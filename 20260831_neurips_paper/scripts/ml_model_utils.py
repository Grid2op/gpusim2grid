# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Random-init (no training, no model.pt checkpoint loading: PFdelta has not
released trained weights) instantiation of a PFdelta model from its own config yaml, for
INFERENCE-ONLY latency/throughput profiling.

CANOS-PF's constructor needs a real dataset-like object at __init__ time (its Encoder infers
per-node/edge-type input dims from it, and the trainer normally injects the real training
dataset wherever a config's `model:` block has a "_include_" placeholder -- see
core/trainers/gnn_trainer.py:customize_model_init_inputs in the pfdelta repo). `InMemoryDataset`
below is a minimal stand-in exposing just what that needs (case_name, indexing,
num_node_features/num_edge_features), built directly from the HeteroData list produced by
ml_pfdelta_bridge.py -- no disk-backed PFDeltaDataset involved.
"""

import os
import sys

import torch
import torch._dynamo
import yaml

# graph_neural_solver's compute_p_joule derives a tensor *size* from batch data
# (`int(edge_batch.max()) + 1`, see core/models/gns.py in the pfdelta repo) -- Inductor
# cannot codegen a guard on that unbacked symbolic value (GuardOnDataDependentSymNode) and
# hard-crashes instead of graph-breaking. This is PFdelta's own unmodified model code (not
# ours to patch, see this module's docstring), so fall back to eager for whatever graph
# region trips this, rather than crashing the whole benchmark run.
torch._dynamo.config.suppress_errors = True

# graph_neural_solver's compute_pg_slack (core/models/gns.py) does a data-dependent
# boolean-mask index (`pg_max_vals[is_slack]`) that Inductor fuses into a kernel with a
# wrong bounds assumption on some torch/CUDA builds -- a device-side assert at *runtime*,
# not a compile-time exception, so suppress_errors above can't catch or recover from it
# (a CUDA device-side assert leaves the whole process's CUDA context unusable). The
# README's own torch.compile investigation already found GNS gets essentially no benefit
# from compilation anyway (falls back to eager for complex-number ops and re-guards every
# Newton-Raphson iteration), so it's not compiled at all -- strictly safer with no real
# throughput cost.
_NO_COMPILE_MODEL_NAMES = {"graph_neural_solver"}

# location were pfdelta paper is cloned
# change is needed
PFDELTA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pfdelta"))
if PFDELTA_ROOT not in sys.path:
    sys.path.insert(0, PFDELTA_ROOT)

from core.utils.registry import registry  # noqa: E402

_registry_loaded = False


def _ensure_registry_loaded():
    """load_registry() globs "core/**/*.py" relative to cwd (see core/utils/main_utils.py in
    the pfdelta repo), so it must run with cwd == pfdelta repo root; restore cwd afterward so
    the rest of this repo's relative paths still work."""
    global _registry_loaded
    if _registry_loaded:
        return
    from core.utils.main_utils import load_registry

    cwd = os.getcwd()
    try:
        os.chdir(PFDELTA_ROOT)
        load_registry()
    finally:
        os.chdir(cwd)
    _registry_loaded = True


class InMemoryDataset:
    """Minimal dataset-like wrapper around a plain list of HeteroData, exposing just the
    duck-typed interface CANOS_PF.__init__ / core.models.canos_utils.Encoder need."""

    def __init__(self, case_name, data_list):
        self.case_name = case_name
        self._data_list = data_list

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    @property
    def num_node_features(self):
        return self[0].num_node_features

    @property
    def num_edge_features(self):
        return self[0].num_edge_features


def load_config(model_yaml_name: str) -> dict:
    """model_yaml_name e.g. "pfnet_task_1_3" (no .yaml, matches core/configs/<name>.yaml and
    the `--config` argument of pfdelta's own main.py)."""
    path = os.path.join(PFDELTA_ROOT, "core", "configs", f"{model_yaml_name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_random_init_model(model_yaml_name: str, dataset: InMemoryDataset = None, device: str = "cpu"):
    """Instantiate the model named by `model_yaml_name`'s config with freshly-initialized
    (untrained) weights -- no model.pt is loaded, since none is released. `dataset` is only
    required for models whose config has a "_include_" dataset placeholder (currently CANOS-PF)."""
    _ensure_registry_loaded()
    config = load_config(model_yaml_name)
    model_cfg = dict(config["model"])
    name = model_cfg.pop("name")

    for key in ("dataset", "data_sample"):
        if model_cfg.get(key) == "_include_":
            if dataset is None:
                raise ValueError(
                    f"{model_yaml_name}: model config needs a '{key}' but none was provided"
                )
            model_cfg[key] = dataset if key == "dataset" else dataset[0]

    model_cls = registry.get_model_class(name)
    model = model_cls(**model_cfg)
    model.eval()
    model.to(device)
    if name not in _NO_COMPILE_MODEL_NAMES:
        model = torch.compile(model)
    return model
