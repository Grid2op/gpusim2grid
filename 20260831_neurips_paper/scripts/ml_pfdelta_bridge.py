# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Bridge between this repo's fast injection-sweep machinery (lightsim2grid) and PFΔ's
model code (../../../pfdelta), for INFERENCE-ONLY use -- no training, no PFDeltaDataset
download/processing, nothing touches disk per scenario.

Why this exists: PFΔ's raw sample schema is {"network": <PowerModels.jl network dict>,
"solution": {"solution": <PowerModels.jl AC-PF result>}} (see
core/datasets/pfdelta_dataset.py:build_heterodata in the pfdelta repo). The static parts of
"network" (branch admittances, bus vmin/vmax, gen pmin/pmax/qmin/qmax, shunts, topology) never
change across an injection sweep -- only pd/qd/pg (inputs) and the solved vm/va/pg/qg
(outputs) do. So PowerModels.jl is invoked only ONCE per case, offline, via
julia_powermodels/dump_pm_static.jl (see that script and matpowerdata/, where the JSON dumps
are saved alongside the source .m/.mat grids).
Every scenario after that is solved here with lightsim2grid (the fastest python available,
matpower order preserving powerflow) and merged into an in-memory copy of the static dict --
PowerModels.jl is never re-invoked, and nothing is written to disk per scenario.

Branch power flows (pf/qf/pt/qt) are NOT computed here: PFΔ's build_heterodata requires them
to exist (as a training target, `edge_label`) but a model's forward() never reads them, so
for this inference/latency-only use they are filled with dummy zeros -- see
`_dummy_branch_solution`.

Alignment: the powerflow solver indexes buses in matpower order (the .m file's row order),
not ascending bus_i. Those two coincide for "clean" cases (e.g. the ACTIVSg family) but not
all -- case3375wp is a verified counterexample -- so assuming bus_i order would silently
mispair generators with the wrong bus on some grids. `load_static_network` renumbers every
bus reference in the static dict to contiguous 1..nbus in matpower order, which is also what
PFΔ's own build_heterodata assumes (plain `bus_i - 1` indexing, unmodified PFΔ code) -- so
after renumbering, every downstream `bus_i - 1` lookup is correct by construction. gen/load
order is already contiguous 1..N in matpower order, which `load_static_network` asserts for
loads rather than risk a silent misalignment.
"""

import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import HeteroData

import lightsim2grid
from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames

PFDELTA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "pfdelta"))
if PFDELTA_ROOT not in sys.path:
    sys.path.insert(0, PFDELTA_ROOT)

from core.datasets.pfdelta_dataset import PFDeltaDataset  # noqa: E402
from core.datasets.pfdelta_variants import (  # noqa: E402
    PFDeltaGNS,
    PFDeltaCANOS,
    PFDeltaPFNet,
)

PF_ALGO = "NR_KLU"
TOL_PF = 1e-8
MAX_ITER = 10

# per-model build_heterodata variant + the `add_bus_type` it expects (must match the
# dataset block of the corresponding core/configs/<model>_task_*.yaml in pfdelta)
MODEL_VARIANTS = {
    "graph_neural_solver": (PFDeltaGNS, False),
    "powerflownet": (PFDeltaPFNet, False),
    "canos_pf": (PFDeltaCANOS, True),
}


def _bare_instance(cls, add_bus_type: bool, pre_transform=None):
    """A real instance of `cls` (a PFDeltaDataset subclass) built via __new__, skipping the
    heavy disk-backed InMemoryDataset __init__ entirely. build_heterodata's implicit
    `super()` calls resolve via `super(__class__, self)`, which requires `self` to actually
    be an instance of the class (isinstance check) -- a plain duck-typed stub object fails
    that check, hence __new__ instead of a lookalike class. build_heterodata (base + all
    three variants) only ever touches self.add_bus_type / self.pre_transform (checked by
    reading pfdelta_dataset.py / pfdelta_variants.py), so nothing else needs to be set."""
    obj = object.__new__(cls)
    obj.add_bus_type = add_bus_type
    obj.pre_transform = pre_transform
    return obj


def load_static_network(case_json_path: str, case_m_path: str) -> Dict[str, Any]:
    """Load the once-per-case PowerModels.jl network dict dumped by
    julia_powermodels/dump_pm_static.jl, and return it alongside derived alignment info.

    case_m_path is needed too (not just the JSON) to get the source .m file's bus ROW order
    -- see the renumbering comment below."""
    with open(case_json_path, "r") as f:
        static = json.load(f)

    # PowerModels omits "rate_a" entirely when the source .m file has RATE_A=0 (this repo's
    # matpowerdata/*.m are vanilla MATPOWER cases with no thermal ratings set, unlike the
    # pglib-opf variants PFdelta itself trained on). build_heterodata unconditionally reads
    # branch["rate_a"] into an (unused-by-any-model's-forward) edge_limits tensor -- for this
    # latency-only benchmark only the shape matters, so backfill a placeholder rather than
    # chase down the exact pglib case file.
    for branch in static["branch"].values():
        branch.setdefault("rate_a", 0.0)

    # Alignment of gen bus id, see description at the top.
    cf = CaseFrames(case_m_path)
    row_order_bus_ids = [int(x) for x in cf.bus["BUS_I"]]
    renumber = {orig: i + 1 for i, orig in enumerate(row_order_bus_ids)}  # orig bus_i -> new 1..nbus

    static["bus"] = {
        str(renumber[int(k)]): {**b, "bus_i": renumber[int(k)]} for k, b in static["bus"].items()
    }
    for gen in static["gen"].values():
        gen["gen_bus"] = renumber[gen["gen_bus"]]
    for load in static["load"].values():
        load["load_bus"] = renumber[load["load_bus"]]
    for branch in static["branch"].values():
        branch["f_bus"] = renumber[branch["f_bus"]]
        branch["t_bus"] = renumber[branch["t_bus"]]
    for shunt in static.get("shunt", {}).values():
        shunt["shunt_bus"] = renumber[shunt["shunt_bus"]]
    for storage in static.get("storage", {}).values():
        storage["storage_bus"] = renumber[storage["storage_bus"]]
    for dcline in static.get("dcline", {}).values():
        dcline["f_bus"] = renumber[dcline["f_bus"]]
        dcline["t_bus"] = renumber[dcline["t_bus"]]
    for switch in static.get("switch", {}).values():
        switch["f_bus"] = renumber[switch["f_bus"]]
        switch["t_bus"] = renumber[switch["t_bus"]]

    nbus = len(static["bus"])
    # trivially list(range(1, nbus + 1)) now, but derived rather than assumed, and this is
    # also what downstream code (check_alignment / solve_scenarios) keys off of -- index i
    # (0-based) -> PM bus key str(bus_ids[i])
    bus_ids = sorted(int(k) for k in static["bus"])
    bus_id_to_ls_idx = {bus_id: i for i, bus_id in enumerate(bus_ids)}

    load_keys = sorted(static["load"].keys(), key=int)
    if [int(k) for k in load_keys] != list(range(1, len(load_keys) + 1)):
        raise ValueError(f"{case_json_path}: PowerModels load ids are not contiguous")

    gen_items_sorted = sorted(static["gen"].items(), key=lambda kv: int(kv[0]))
    active_gen_keys = [k for k, g in gen_items_sorted if g["gen_status"] == 1]

    active_branch_keys = [
        k for k, b in static["branch"].items() if b["br_status"] == 1
    ]

    return {
        "static": static,
        "baseMVA": static["baseMVA"],
        "nbus": nbus,
        "bus_ids": bus_ids,  # index i (0-based, ls bus order) -> PM bus key str(bus_ids[i])
        "bus_id_to_ls_idx": bus_id_to_ls_idx,  # PM bus_i (int) -> lsgrid bus index (0-based)
        "load_keys": load_keys,  # index i (0-based) -> PowerModels load key str(i+1)
        "active_gen_keys": active_gen_keys,  # index j (0-based, ls gen order) -> PM gen key
        "active_branch_keys": active_branch_keys,
    }


def make_lsgrid(case_m_path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lsgrid = init_from_matpower(case_m_path)
    lsgrid.change_algorithm(PF_ALGO)
    return lsgrid


def check_alignment(lsgrid, net_info: Dict[str, Any]) -> None:
    """Also stores `net_info["active_gen_mask"]`, a bool array over lightsim2grid's FULL
    (possibly-disconnected-included) generator array marking which entries are connected
    -- needed downstream (see solve_scenarios) because lightsim2grid.get_generators() /
    get_gen_res_full() always return one entry per matpower gen row, status==0 (disconnected)
    rows included (each still gets a `.connected` flag, see lsgrid.network.from_powermodels'
    _aux_add_gen.py), whereas PowerModels' active_gen_keys only lists status==1 rows -- a
    case with disconnected gens interleaved among connected ones (e.g. case_ACTIVSg500)
    would otherwise silently zip the wrong lightsim2grid result with the wrong PM gen key."""
    static = net_info["static"]
    if lsgrid.total_bus() != net_info["nbus"]:
        raise ValueError(
            f"bus count mismatch: lightsim2grid={lsgrid.total_bus()} vs PowerModels={net_info['nbus']}"
        )
    if len(lsgrid.get_loads()) != len(net_info["load_keys"]):
        raise ValueError(
            f"load count mismatch: lightsim2grid={len(lsgrid.get_loads())} "
            f"vs PowerModels={len(net_info['load_keys'])}"
        )
    all_gens = lsgrid.get_generators()
    connected_gens = [g for g in all_gens if g.connected]
    if len(connected_gens) != len(net_info["active_gen_keys"]):
        raise ValueError(
            f"active gen count mismatch: lightsim2grid={len(connected_gens)} "
            f"vs PowerModels={len(net_info['active_gen_keys'])}"
        )
    ls_gen_bus0 = [g.bus_id for g in connected_gens]
    pm_gen_bus0 = [
        net_info["bus_id_to_ls_idx"][static["gen"][k]["gen_bus"]] for k in net_info["active_gen_keys"]
    ]
    if ls_gen_bus0 != pm_gen_bus0:
        raise ValueError(
            "gen-to-bus alignment mismatch between lightsim2grid and PowerModels static "
            "dict -- do not trust this case's scenarios (see module docstring)"
        )
    net_info["active_gen_mask"] = np.array([g.connected for g in all_gens], dtype=bool)


def _dummy_branch_solution(net_info: Dict[str, Any]) -> Dict[str, Any]:
    """edge_label is a training target build_heterodata requires to exist but no model's
    forward() reads (see module docstring) -- dummy zeros, same dict reused across scenarios."""
    return {k: {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0} for k in net_info["active_branch_keys"]}


def solve_scenarios(
    lsgrid,
    net_info: Dict[str, Any],
    load_p_mw: np.ndarray,
    load_q_mvar: np.ndarray,
    gen_p_mw: np.ndarray,
    v_init_seed: np.ndarray = None,
    warm_start: bool = True,
):
    """Generator yielding one pm_case-shaped dict per converged scenario.

    load_p_mw, load_q_mvar: shape (n_scenarios, n_load) -- the convention returned by
        generate_data_from_matpower.py's get_loads_gens*/get_injection_from_base (rows are
        scenarios), matpower-row-order == PowerModels load-key order along the last axis
        (see load_static_network / check_alignment).
    gen_p_mw: shape (n_scenarios, n_active_gen), same convention for active gens.
    v_init_seed: the starting voltage guess for scenario 0 (subsequent scenarios warm-start
        from the previous one's converged result, same convention as
        InjectionSweepCPP.init_from_n_powerflow in ls_injection.py). Should be the SAME
        flat/dc/file-derived v_init (or the resulting converged V0) used for this case's base
        powerflow (see get_init_method.py) -- flat start (the default here if omitted) does
        not converge for every case (e.g. case_ACTIVSg10k), in which case every scenario
        would silently fail to converge (V.shape[0] == 0 -> skipped) with v_prev stuck at
        the never-updated flat guess for the whole sweep.

    Nothing is written to disk; each yielded dict is discarded by the caller once its
    HeteroData has been built (see build_heterodata_batch), so peak memory is O(1) in the
    number of scenarios, not O(n_scenarios) -- important for large grids.
    """
    static = net_info["static"]
    baseMVA = net_info["baseMVA"]
    branch_solution = _dummy_branch_solution(net_info)
    n_scen, n_load = load_p_mw.shape

    all_true_load = np.ones(n_load, dtype=bool)
    all_true_gen = np.ones(gen_p_mw.shape[1], dtype=bool)

    v_prev = (
        np.array(v_init_seed, dtype=complex)
        if v_init_seed is not None
        else np.ones(lsgrid.total_bus(), dtype=complex)
    )
    for scen in range(n_scen):
        lsgrid.update_loads_p(all_true_load, load_p_mw[scen, :].astype(np.float32))
        lsgrid.update_loads_q(all_true_load, load_q_mvar[scen, :].astype(np.float32))
        lsgrid.update_gens_p(all_true_gen, gen_p_mw[scen, :].astype(np.float32))

        v_init = v_prev.copy() if warm_start else np.ones(lsgrid.total_bus(), dtype=complex)
        V = lsgrid.ac_pf(v_init, MAX_ITER, TOL_PF)
        lsgrid.unset_changes()
        if V.shape[0] == 0:
            continue  # did not converge, skip this scenario
        v_prev = V

        gen_p_res_full, gen_q_res_full, _, _ = lsgrid.get_gen_res_full()
        # get_gen_res_full() is indexed over ALL matpower gen rows (disconnected ones
        # included, see check_alignment); active_gen_mask picks out only the connected
        # ones, in the same order as net_info["active_gen_keys"] (verified in check_alignment)
        gen_p_res = gen_p_res_full[net_info["active_gen_mask"]]
        gen_q_res = gen_q_res_full[net_info["active_gen_mask"]]

        load_dict = {
            net_info["load_keys"][i]: {
                **static["load"][net_info["load_keys"][i]],
                "pd": float(load_p_mw[scen, i]) / baseMVA,
                "qd": float(load_q_mvar[scen, i]) / baseMVA,
            }
            for i in range(n_load)
        }
        network = dict(static)
        network["load"] = load_dict

        # V is indexed in lsgrid bus order, which is ascending-PM-bus-key order (see
        # load_static_network) -- NOT "PM bus key == str(i + 1)", since PM bus_i need not
        # be contiguous 1..nbus (e.g. case_ACTIVSg10k).
        bus_sol = {
            str(net_info["bus_ids"][i]): {"va": float(np.angle(V[i])), "vm": float(np.abs(V[i]))}
            for i in range(net_info["nbus"])
        }
        gen_sol = {
            net_info["active_gen_keys"][j]: {
                "pg": float(gen_p_res[j]) / baseMVA,
                "qg": float(gen_q_res[j]) / baseMVA,
            }
            for j in range(len(net_info["active_gen_keys"]))
        }

        yield {
            "network": network,
            "solution": {"solution": {"bus": bus_sol, "gen": gen_sol, "branch": branch_solution}},
        }


def _apply_gns_extras(data: HeteroData) -> HeteroData:
    """Reimplementation of PFDeltaGNS.build_heterodata's per-model additions -- i.e.
    everything AFTER its `super().build_heterodata(...)` call -- copied from pfdelta's
    core/datasets/pfdelta_variants.py (PFDeltaGNS.build_heterodata, ~line 88), commit
    ddf866f. PFDeltaDataset.build_heterodata's (expensive)
    base construction is IDENTICAL for all models -- building it once and running each model's
    (cheap) additions on a clone avoids the redundant, more expensive, second base build,
    without touching pfdelta itself. NOT kept in sync automatically -- if PFDeltaGNS's
    build_heterodata changes in pfdelta, this needs to be updated to match (verified against
    the original for equivalence when written, see conversation)."""
    num_buses = data["bus"].x.size(0)
    num_gens = data["gen"].generation.size(0)
    num_loads = data["load"].demand.size(0)
    bus_types = data["bus"].bus_type
    x_gns = torch.zeros((num_buses, 2))
    v_buses = torch.zeros(num_buses)
    theta_buses = torch.zeros(num_buses)
    pd_buses = torch.zeros(num_buses)
    qd_buses = torch.zeros(num_buses)
    pg_buses = torch.zeros(num_buses)
    qg_buses = torch.zeros(num_buses)

    for bus_idx in range(num_buses):
        bus_type = bus_types[bus_idx].item()
        pf_x = data["bus"].x[bus_idx]
        bus_demand = data["bus"].bus_demand[bus_idx]
        bus_gen = data["bus"].bus_gen[bus_idx]

        if bus_type == 1:  # PQ bus
            x_gns[bus_idx] = torch.tensor([0.0, 1.0])
            pd, qd = pf_x[0], pf_x[1]
            pg, qg = bus_gen[0], bus_gen[1]
        elif bus_type == 2:  # PV bus
            v = pf_x[1]
            theta = torch.tensor(0.0)
            x_gns[bus_idx] = torch.stack([theta, v])
            pd, qd = bus_demand[0], bus_demand[1]
            pg, qg = bus_gen[0], bus_gen[1]
        elif bus_type == 3:  # Slack bus
            x_gns[bus_idx] = pf_x
            pd, qd = bus_demand[0], bus_demand[1]
            pg, qg = bus_gen[0], bus_gen[1]

        v_buses[bus_idx] = x_gns[bus_idx][1]
        theta_buses[bus_idx] = x_gns[bus_idx][0]
        pd_buses[bus_idx] = pd
        qd_buses[bus_idx] = qd
        pg_buses[bus_idx] = pg
        qg_buses[bus_idx] = qg

    data["bus"].x_gns = x_gns
    data["bus"].v = v_buses
    data["bus"].theta = theta_buses
    data["bus"].pd = pd_buses
    data["bus"].qd = qd_buses
    data["bus"].pg = pg_buses
    data["bus"].qg = qg_buses
    data["bus"].delta_p = torch.zeros_like(v_buses)
    data["bus"].delta_q = torch.zeros_like(v_buses)
    data["gen"].num_nodes = num_gens
    data["load"].num_nodes = num_loads
    return data


def _apply_pfnet_extras(data: HeteroData) -> HeteroData:
    """Reimplementation of PFDeltaPFNet.build_heterodata's per-model additions -- copied from
    pfdelta's core/datasets/pfdelta_variants.py (PFDeltaPFNet.build_heterodata, ~line 394),
    commit ddf866f. Same rationale/caveat as _apply_gns_extras. self.pre_transform is always
    None for our bare instances (see _bare_instance), so that branch of the original is
    omitted here rather than reimplemented."""
    num_buses = data["bus"].x.size(0)
    bus_types = data["bus"].bus_type
    pf_x = data["bus"].x
    pf_y = data["bus"].y
    shunts = data["bus"].shunt
    num_gens = data["gen"].generation.size(0)
    num_loads = data["load"].demand.size(0)

    x_pfnet, y_pfnet = [], []
    for i in range(num_buses):
        bus_type = int(bus_types[i].item())
        one_hot = torch.zeros(4)
        one_hot[bus_type - 1] = 1
        gs, bs = shunts[i]

        if bus_type == 1:  # PQ
            pred_mask = torch.tensor([1, 1, 0, 0, 0, 0])
            va, vm = pf_y[i]
            pd, qd = pf_x[i]
            input_mask = (1 - pred_mask).float()
            input_feats = torch.tensor([vm, va, pd, qd, gs, bs]) * input_mask
            features = torch.cat([one_hot, input_feats])
            y = torch.tensor([vm, va, pd, qd, gs, bs])
        elif bus_type == 2:  # PV
            pred_mask = torch.tensor([0, 1, 0, 1, 0, 0])
            pg_pd, vm = pf_x[i]
            qg_qd, va = pf_y[i]
            input_mask = (1 - pred_mask).float()
            input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
            features = torch.cat([one_hot, input_feats])
            y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])
        elif bus_type == 3:  # Slack
            pred_mask = torch.tensor([0, 0, 1, 1, 0, 0])
            va, vm = pf_x[i]
            pg_pd, qg_qd = pf_y[i]
            input_mask = (1 - pred_mask).float()
            input_feats = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs]) * input_mask
            features = torch.cat([one_hot, input_feats])
            y = torch.tensor([vm, va, pg_pd, qg_qd, gs, bs])

        x_pfnet.append(torch.cat([features, pred_mask]))
        y_pfnet.append(y)

    data["bus"].x = torch.stack(x_pfnet)
    data["bus"].y = torch.stack(y_pfnet)
    data["gen"].num_nodes = num_gens
    data["load"].num_nodes = num_loads

    edge_attrs = []
    for attr in data["bus", "branch", "bus"].edge_attr:
        r, x = attr[0], attr[1]
        b = attr[3] + attr[5]
        tau, angle = attr[6], attr[7]
        edge_attrs.append(torch.tensor([r, x, b, tau, angle]))
    data["bus", "branch", "bus"].edge_attr = torch.stack(edge_attrs)

    return data


def _prune_node_types(data: HeteroData, keep_node_types: set) -> HeteroData:
    """Delete every node type (and any edge type touching it) not in `keep_node_types`.
    Generalized from PFDeltaCANOS.build_heterodata's own prune step (pfdelta_variants.py
    ~284-294, which hardcodes keep_nodes = {"bus", "PV", "PQ", "slack"}) so the same logic can
    also prune PQ/PV/slack off the shared base for graph_neural_solver/powerflownet -- see
    build_heterodata_batch."""
    for node_type in list(data.node_types):
        if node_type not in keep_node_types:
            del data[node_type]
    for edge_type in list(data.edge_types):
        src, _, dst = edge_type
        if src not in keep_node_types or dst not in keep_node_types:
            del data[edge_type]
    return data


# model_name -> its "extras" function, applied (after pruning) to a clone of the shared base
# -- see build_heterodata_batch. canos_pf needs no extras function: its own build_heterodata
# is nothing BUT the prune step (see PFDeltaCANOS.build_heterodata), already covered by
# _prune_node_types below.
_SHARED_BASE_EXTRAS = {
    "graph_neural_solver": _apply_gns_extras,
    "powerflownet": _apply_pfnet_extras,
}

# Node types each model's final HeteroData should retain, after pruning the shared
# add_bus_type=True base (which is the superset: bus/gen/load + PQ/PV/slack, all built by the
# SAME per-bus loop regardless of add_bus_type -- see build_heterodata_batch's docstring for
# why add_bus_type=True is free to always build). graph_neural_solver/powerflownet never had
# PQ/PV/slack in their output (they've always used add_bus_type=False) -- pruning those off
# here keeps their result identical to before, and avoids carrying unused tensors into every
# GPU batch. canos_pf's keep-set matches PFDeltaCANOS.build_heterodata's own keep_nodes.
_KEEP_NODE_TYPES = {
    "graph_neural_solver": {"bus", "gen", "load"},
    "powerflownet": {"bus", "gen", "load"},
    "canos_pf": {"bus", "PV", "PQ", "slack"},
}


class _LazyHeteroDataList:
    """Sequence-like view over `pm_cases` that converts each entry to a HeteroData on first
    __getitem__ access and caches the result, instead of eagerly converting every entry up
    front. PFΔ's build_heterodata is non-trivial per-scenario work; a (model, batch_size)
    sweep only ever reads a max_batches*batch_size-sized prefix of this list (see
    time_model_at_batch_size / --max_batches in ml_injection.py), and once one batch_size
    OOMs every larger one is skipped entirely (see time_model's oom_batch_size) without ever
    touching the rest -- so eagerly converting the full --nb_pf scenarios up front (previous
    behavior) did a lot of conversion work that was never actually used. Duck-typed to work
    as a torch_geometric DataLoader dataset the same way a plain list already did.

    Every model routes through `_shared_base_cache` (module level, keyed by scenario index)
    instead of calling PFDeltaDataset.build_heterodata itself -- see build_heterodata_batch's
    docstring."""

    def __init__(self, model_name: str, pm_cases: list, base_obj):
        self._model_name = model_name
        self._pm_cases = pm_cases
        self._base_obj = base_obj
        self._cache = [None] * len(pm_cases)
        self._n_converted = 0
        self._n_base_built = 0
        self._t_start = None
        self._t_last_log = None

    def __len__(self):
        return len(self._pm_cases)

    def __getitem__(self, idx):
        if self._cache[idx] is not None:
            return self._cache[idx]

        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
            self._t_last_log = now
            print(f"  [{self._model_name}] converting scenarios to HeteroData as needed "
                  f"(up to {len(self._pm_cases)} total; a scenario touched by an earlier "
                  f"model in this process is already cached and skips the expensive part)",
                  flush=True)

        base = _shared_base_cache.get(idx)
        if base is None:
            base = self._base_obj.build_heterodata(self._pm_cases[idx], is_cpf_sample=False)
            _shared_base_cache[idx] = base
            self._n_base_built += 1

        data = _prune_node_types(base.clone(), _KEEP_NODE_TYPES[self._model_name])
        extras_fn = _SHARED_BASE_EXTRAS.get(self._model_name)
        if extras_fn is not None:
            data = extras_fn(data)

        self._cache[idx] = data
        self._n_converted += 1

        # Logged periodically (by wall time, not scenario count, since per-scenario cost varies
        # a lot with grid size) rather than only at the end -- this loop is the actual cost
        # hidden inside what looked like an 8h gap between "converged: N/N" and the first
        # batch-size result line in a real run (case6515rte): forward-pass timing (_timed_pass
        # in ml_injection.py) only starts AFTER the full DataLoader/list(...) materializes,
        # so nothing printed here before meant nothing was visible for however long this took.
        if now - self._t_last_log > 30 or self._n_converted == len(self._pm_cases):
            elapsed = time.perf_counter() - self._t_start
            rate = self._n_converted / elapsed if elapsed > 0 else 0.0
            eta_s = (len(self._pm_cases) - self._n_converted) / rate if rate > 0 else float("inf")
            print(f"  [{self._model_name}] converted {self._n_converted}/{len(self._pm_cases)} "
                  f"scenarios ({self._n_base_built} needed a fresh base build) -- "
                  f"{elapsed:.0f}s elapsed, {rate:.2f} scen/s, ETA {eta_s:.0f}s", flush=True)
            self._t_last_log = time.perf_counter()

        return data


# Keyed by scenario index (0..nb_pf-1), shared across every model's separate
# build_heterodata_batch call within one ml_injection.py process -- see _LazyHeteroDataList.
# One process only ever times one --case (base_launch_ml_batchsize* scripts spawn a fresh
# python process per case), and `pm_cases`'s index i means the same scenario in every model's
# call within that process, so a plain module-level dict (no per-case keying) is safe here.
_shared_base_cache: Dict[int, HeteroData] = {}


def build_heterodata_batch(
    model_name: str,
    pm_cases,
) -> List[HeteroData]:
    """Lazily convert an iterable of pm_case dicts (as yielded by solve_scenarios) into
    HeteroData objects, reimplementing PFΔ's per-model build_heterodata logic on top of ONE
    shared base per scenario instead of a separate call per model -- see _LazyHeteroDataList
    for why this is lazy/cached rather than eager.

    The shared base is always built with add_bus_type=True: reading PFDeltaDataset's own
    build_heterodata (pfdelta_dataset.py), `self.add_bus_type` is only checked twice, both
    late and both cheap (whether to torch.stack() the already-computed PQ/PV/slack lists onto
    the HeteroData, and whether to add their `_link` edges) -- the expensive part (the
    per-bus loop, and the O(n_gen log n_gen) grouping fixed separately) runs identically
    either way (verified: ~697ms vs ~694ms/scenario on case_ACTIVSg10k, see conversation).
    So add_bus_type=True's superset (bus/gen/load/branch + PQ/PV/slack) is free to always
    build, and each model just prunes down to what it actually needs (_KEEP_NODE_TYPES) --
    canos_pf's prune is exactly PFDeltaCANOS.build_heterodata's own logic
    (_prune_node_types), and graph_neural_solver/powerflownet's extras
    (_SHARED_BASE_EXTRAS) are reimplementations of their variant classes' post-base-build
    additions. All three previously needed their own separate (expensive) base build; now
    only the first model to touch a given scenario index pays for it."""
    base_obj = _bare_instance(PFDeltaDataset, add_bus_type=True)
    return _LazyHeteroDataList(model_name, list(pm_cases), base_obj)
