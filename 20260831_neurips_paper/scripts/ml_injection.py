# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Inference-only latency/throughput benchmark for PFdelta's ML models (CANOS-PF, GNS
[graph_neural_solver], PowerFlowNet) on this repo's own injection-sweep scenarios, run on
THIS machine's hardware. PFdelta has not released trained checkpoints, and
even if it had, the paper's own inference-time numbers were measured on different hardware and
are not comparable to timings taken here. Models are therefore random-init (untrained) and
used for throughput profiling only, never for accuracy. Accuracy number are taken from 
the papers themselves.

Data pipeline (see ml_pfdelta_bridge.py's module docstring for the full rationale): topology +
static network parameters come from a PowerModels.jl parse done ONCE offline per case
(julia_powermodels/dump_pm_static.jl / dump_pm_static_all.py ->
matpowerdata/<case>.json); each scenario's pd/qd/pg is sampled the same
way as the other CPU baselines (pandapower, OLF, etc.), then solved with lightsim2grid 
(fastest solver available from python, keeping matpower default ordering)
and merged into an in-memory HeteroData -- PowerModels.jl is never
re-invoked per scenario, nothing is written to disk per scenario.

Batch size: like for other GPU method, forward-pass throughput is batch-size dependent (GPU
utilization in particular; --batch_size 1 badly underuses a GPU). --batch_size accepts one or
more values and every model is timed at every batch size.

Timing methodology (mirrors the one described in the supplementary material of the paper): 
for each (model, batch_size), a separate, explicitly reported
"warmup" pass (--n_warmup_batches batches, default 2 -- pays first-CUDA-call/kernel-compile
cost on GPU, cold-cache cost on CPU) is timed and kept in the output, distinct from the
"full_sweep" pass that follows and re-times the ENTIRE batch set (including the batches used
for warmup) -- that full_sweep is the number to actually use for throughput comparisons.

OOM handling: a (model, batch_size) combination that doesn't fit in VRAM records
{"error": "OOM", "error_detail": ...} in place of its warmup/full_sweep entry and the sweep
moves on to the next batch size / model, rather than aborting the whole run -- expected on a
wide --batch_size sweep (see base_launch_ml_batchsize.sh), not treated as a crash. Once a
batch_size OOMs for a given model, every LARGER batch_size is skipped for that model too
(recorded as {"error": "OOM", "skipped": True, ...}, no forward pass / DataLoader collation
attempted) rather than actually run and left to OOM again -- for torch.compile'd models
(every model except graph_neural_solver, see ml_model_utils.py) each new batch_size is a new
input shape, so running a doomed larger batch_size would also pay for a full Dynamo
recompilation just to OOM anyway.

--max_batches (default 100) caps how many batches full_sweep actually times per (model,
batch_size) -- a throughput estimate from 100 batches is almost as good as one from all of them, and
without this a small --batch_size (e.g. 1) against a large --nb_pf (e.g. 16384) would time 16384
individual forward passes just to report the same samples_per_s a 100-batch estimate already
gives. warmup is unaffected (already just --n_warmup_batches, far below any sensible
--max_batches). The per-batch-size result's "total_batches_available" field records how many
batches batch_size would have produced uncapped, so a capped run is visible in the output JSON
even without cross-referencing the command line.

Usage:
    python ml_injection.py --case case118.m --sample_data_meth sable --nb_pf 2000 \
        --batch_size 8 32 128 --device cpu
    python ml_injection.py --case case118.m --nb_pf 2000 --batch_size 32 --device cuda:0
"""

import argparse
import itertools
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import ml_pfdelta_bridge as br
import ml_model_utils as mu
from generate_data_from_matpower import get_loads_gens, get_loads_gens_sable
from get_init_method import get_init_method
from pp_injection import get_ls_v_init

from scenario_utils import (
    REF_PATH,
    PATH_RESULTS,
    NB_PF_TOTAL,
    SEED
)
STATIC_NET_DIR = REF_PATH
BATCH_SIZES = [32]
N_WARMUP_BATCHES = 2  # separately timed, see module docstring; not folded into full_sweep's total
MAX_BATCHES = 100  # cap on full_sweep's batch count per (model, batch_size); see module docstring

MODEL_CONFIGS = {
    "powerflownet": "pfnet_task_1_3",
    "graph_neural_solver": "gns_task_1_3",
    "canos_pf": "canos_task_1_3",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Inference-only latency/throughput benchmark for PFdelta's ML models "
        "(random-init weights -- no trained checkpoints exist, see module docstring)."
    )
    parser.add_argument("--case", type=str, default="case118.m", help="matpower case file name (under matpowerdata/)")
    parser.add_argument("--sample_data_meth", type=str, default="sable", choices=["fr", "sable"])
    parser.add_argument("--nb_pf", type=int, default=NB_PF_TOTAL, help="number of scenarios to solve/time")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, nargs="+", default=BATCH_SIZES,
                         help="one or more batch sizes; every model is timed at each of them")
    parser.add_argument("--n_warmup_batches", type=int, default=N_WARMUP_BATCHES,
                         help="batches used for the separately-reported warmup pass (see module docstring)")
    parser.add_argument("--max_batches", type=int, default=MAX_BATCHES,
                         help="cap on batches full_sweep times per (model, batch_size); "
                              "<=0 disables the cap (see module docstring)")
    parser.add_argument("--device", type=str, default="cpu", help='"cpu" or e.g. "cuda:0"')
    parser.add_argument("--launch_cpu", action="store_true", default=False,
                         help="Force every torch call (model + batches) onto CPU, overriding "
                              "--device -- for a CPU-only run regardless of what --device says "
                              "or whether a GPU is available.")
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_CONFIGS.keys()),
                         choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--add_to_name", type=str, default="")
    return parser.parse_args()


def get_final_name(path, tmp_nm, add_to_name):
    if add_to_name:
        tmp_nm += add_to_name
    return os.path.join(path, f"{tmp_nm}.json")


def solve_all_scenarios(case_stem, sample_data_meth, nb_pf, seed):
    """Runs the fast (lightsim2grid) part once: base-case solve + scenario sampling + solve.
    Returns (pm_cases, timings_dict). Kept separate from the ML timing loop below so the two
    very different costs (physics solve vs. model forward) are never conflated in the report."""
    case_m_path = os.path.join(REF_PATH, f"{case_stem}.m")
    static_json_path = os.path.join(STATIC_NET_DIR, f"{case_stem}.json")
    if not os.path.exists(static_json_path):
        raise FileNotFoundError(
            f"{static_json_path} not found -- run dump_pm_static_all.py (or "
            f"julia_powermodels/dump_pm_static.jl directly) once for this case first."
        )

    net_info = br.load_static_network(static_json_path, case_m_path)
    lsgrid = br.make_lsgrid(case_m_path)
    br.check_alignment(lsgrid, net_info)

    # flat start doesn't converge for every case (e.g. case_ACTIVSg10k) -- reuse this repo's
    # existing per-case flat/dc/file init policy (see get_init_method.py / ls_injection.py)
    # rather than hardcoding flat, which is what every other *_injection.py script does too.
    v_init = get_ls_v_init(case_m_path, lsgrid, get_init_method(case_stem))
    V0 = lsgrid.ac_pf(v_init, 10, br.TOL_PF)
    lsgrid.unset_changes()
    if V0.shape[0] == 0:
        raise RuntimeError(f"{case_stem}: base powerflow did not converge")
    load_p_init, load_q_init, *_ = lsgrid.get_loads_res_full()
    gen_p_init, *_ = lsgrid.get_gen_res_full()

    prng = np.random.default_rng(seed)
    sample_fun = get_loads_gens_sable if sample_data_meth == "sable" else get_loads_gens
    load_p, load_q, gen_p = sample_fun(load_p_init, load_q_init, gen_p_init, prng, nb_pf)

    beg = time.perf_counter()
    # total=nb_pf (not "unknown"): solve_scenarios' internal loop runs once per requested
    # scenario regardless of convergence (it just `continue`s past non-converged ones without
    # yielding), so nb_pf is the right denominator for the progress bar even though fewer than
    # nb_pf items may come out the other end.
    pm_cases = list(tqdm(
        br.solve_scenarios(lsgrid, net_info, load_p, load_q, gen_p, v_init_seed=V0),
        total=nb_pf, desc=f"{case_stem}: solving scenarios",
    ))
    end = time.perf_counter()

    return pm_cases, {
        "n_requested": int(nb_pf),
        "n_converged": len(pm_cases),
        "total_time": end - beg,
        "scenarios_per_s": len(pm_cases) / max(end - beg, 1e-9),
    }


def _timed_pass(model, batches, device):
    """Times each batch in `batches` individually (forward pass only, no_grad), with a
    torch.cuda.synchronize() bracketing each call on GPU -- without it CUDA calls return as
    soon as the kernel is *launched*, not once it has actually completed, which would time
    launch overhead instead of the real forward-pass cost. Returns per-batch times (s) and
    per-batch sample counts (batch.num_graphs), so both aggregate and per-batch-mean
    ("moyenne des temps moyens", see README's "Points d'attention") metrics can be computed."""
    use_cuda = str(device).startswith("cuda")
    per_batch_time = []
    per_batch_n = []
    for batch in batches:
        batch = batch.to(device)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(batch)
        if use_cuda:
            torch.cuda.synchronize()
        per_batch_time.append(time.perf_counter() - t0)
        per_batch_n.append(batch.num_graphs)
    return per_batch_time, per_batch_n


def _summarize_pass(per_batch_time, per_batch_n):
    total_time = sum(per_batch_time)
    n_samples = sum(per_batch_n)
    ms_per_sample_per_batch = [
        1000.0 * t / n for t, n in zip(per_batch_time, per_batch_n) if n > 0
    ]
    return {
        "n_batches": len(per_batch_time),
        "n_samples": n_samples,
        "total_time": total_time,
        "samples_per_s": n_samples / max(total_time, 1e-9),
        "ms_per_batch_mean": 1000.0 * total_time / max(len(per_batch_time), 1),
        # mean of per-batch (ms/sample) values, ie the "average of averages" the README asks
        # for, robust to a varying number of scenarios per batch (eg the final partial batch)
        "ms_per_sample_batchmean": float(np.mean(ms_per_sample_per_batch)) if ms_per_sample_per_batch else None,
        "ms_per_sample_batchstd": float(np.std(ms_per_sample_per_batch)) if ms_per_sample_per_batch else None,
    }


def _empty_cuda_cache(device):
    # Only reclaims cached-but-unused blocks (PyTorch reuses its own cache regardless) --
    # mostly hygiene, so nvidia-smi/other processes on a shared GPU see accurate free memory
    # before the next (model, batch_size) attempt, and so a later, larger batch size on the
    # SAME model still gets a fair shot rather than starting from an already-fragmented cache.
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def time_model_at_batch_size(model, hetero_list, batch_size, device, n_warmup_batches, max_batches):
    total_batches = -(-len(hetero_list) // batch_size)  # ceil division
    capped = max_batches > 0 and total_batches > max_batches

    # The trailing partial batch (if any) is the LAST one in iteration order, so capping to the
    # FIRST max_batches batches already excludes it whenever capped is True -- only warn about
    # it when it will actually be reached (uncapped, or the cap happens to land past it anyway).
    if len(hetero_list) % batch_size != 0 and not capped:
        print(f"  WARNING: {len(hetero_list)} scenarios is not a multiple of batch_size={batch_size} "
              f"-> the final batch has a different (smaller) shape. With torch.compile, this shape "
              f"wasn't seen during warmup and triggers an extra recompilation INSIDE full_sweep's "
              f"timed pass, which can badly skew the reported throughput. Prefer --nb_pf a multiple "
              f"of every --batch_size value.")

    loader = DataLoader(hetero_list, batch_size=batch_size, shuffle=False)
    # islice (not list(loader) then slicing) so a small batch_size against a large --nb_pf
    # (e.g. batch_size=1, nb_pf=16384) doesn't pay to collate batches never actually timed --
    # see module docstring's "--max_batches" paragraph for why 100 is enough for a throughput
    # estimate.
    batches = list(itertools.islice(loader, max_batches)) if capped else list(loader)
    if capped:
        print(f"  batch_size={batch_size}: capping full_sweep to {max_batches}/{total_batches} "
              f"batches (--max_batches)")
    n_warmup = min(n_warmup_batches, len(batches))

    # A batch size too big for this GPU's VRAM is an expected outcome of a batch-size sweep
    # (see base_launch_ml_batchsize.sh), not a bug -- catch it narrowly (only the dedicated OOM
    # exception, so a genuine crash like the GNS Inductor bug documented in ml_model_utils.py
    # is never mistaken for "just OOM") and record it instead of aborting the whole run.
    try:
        warmup_time, warmup_n = _timed_pass(model, batches[:n_warmup], device)
        warmup_summary = _summarize_pass(warmup_time, warmup_n)

        # full_sweep re-times the WHOLE (possibly capped) batch set (including the ones just
        # used for warmup, same convention as ls_injection.py re-solving scenario 1 inside its
        # own full time_series) -- this, not warmup, is the number to use for throughput
        # comparisons.
        full_time, full_n = _timed_pass(model, batches, device)
        full_summary = _summarize_pass(full_time, full_n)
    except torch.cuda.OutOfMemoryError as e:
        _empty_cuda_cache(device)
        return {"batch_size": batch_size, "error": "OOM", "error_detail": str(e),
                "total_batches_available": total_batches}

    return {
        "batch_size": batch_size,
        "warmup": warmup_summary,
        "full_sweep": full_summary,
        "total_batches_available": total_batches,
    }


def time_model(model_name, config_name, pm_cases, case_stem, batch_sizes, device, n_warmup_batches, max_batches):
    hetero_list = br.build_heterodata_batch(model_name, pm_cases)
    dataset = mu.InMemoryDataset(case_stem, hetero_list)
    try:
        model = mu.build_random_init_model(config_name, dataset=dataset, device=device)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM while loading {model_name} onto {device} -- skipping this model entirely")
        _empty_cuda_cache(device)
        return {"error": "OOM", "error_detail": str(e), "by_batch_size": {}}
    n_params = sum(p.numel() for p in model.parameters())

    by_batch_size = {}
    # Sorted ascending regardless of the order batch_sizes were passed in on the CLI, so
    # "every larger batch_size" (below) is well-defined even if a caller passed them out of
    # order -- every base_launch_ml_batchsize*.sh script already passes them ascending anyway.
    oom_batch_size = None  # smallest batch_size that OOM'd for THIS model; None until one does
    for batch_size in sorted(batch_sizes):
        if oom_batch_size is not None:
            print(f"  batch_size={batch_size}: SKIPPED (batch_size={oom_batch_size} already OOM'd for this model)")
            by_batch_size[str(batch_size)] = {
                "batch_size": batch_size,
                "error": "OOM",
                "skipped": True,
                "error_detail": f"skipped -- batch_size={oom_batch_size} already OOM'd for this model",
            }
            continue

        res = time_model_at_batch_size(model, hetero_list, batch_size, device, n_warmup_batches, max_batches)
        if "error" in res:
            print(f"  batch_size={batch_size}: {res['error']}")
            oom_batch_size = batch_size
        else:
            print(f"  batch_size={batch_size}: warmup {res['warmup']['ms_per_batch_mean']:.2f} ms/batch "
                  f"-> full_sweep {res['full_sweep']['samples_per_s']:.0f} samples/s "
                  f"({res['full_sweep']['ms_per_sample_batchmean']:.3f} ms/sample)")
        by_batch_size[str(batch_size)] = res

    return {"n_params": n_params, "by_batch_size": by_batch_size}


if __name__ == "__main__":
    # stdout is block-buffered by default when redirected to a file (e.g. `> log.txt`), while
    # tqdm's progress bar writes to stderr, which is unbuffered -- without this, print() lines
    # from this module and ml_pfdelta_bridge.py can sit in the buffer for a long time (observed:
    # hours) before actually appearing in the log, making them show up chronologically AFTER
    # tqdm lines that were printed much later in wall-clock time. Line-buffering fixes both the
    # multi-hour visibility gap and the misleading ordering.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # TF32 matmuls are opt-in in PyTorch even on tensor-core GPUs (Ampere/Ada/Hopper) --
    # leaving this off costs real float32-matmul throughput for no benefit here: this
    # benchmark is inference-only latency/throughput on random-init weights (see module
    # docstring), never an accuracy comparison, so TF32's reduced precision is free to take.
    # No-op on GPUs without tensor cores (e.g. T4) and on CPU.
    torch.set_float32_matmul_precision("high")

    args = get_args()
    case_stem, _ = os.path.splitext(args.case)

    nb_pf = args.nb_pf
    batch_sizes = [b for b in args.batch_size if b <= nb_pf]
    dropped = sorted(set(args.batch_size) - set(batch_sizes))
    if dropped:
        print(f"{case_stem}: dropping --batch_size values > nb_pf={nb_pf}: {dropped} "
              f"(would otherwise run as an under-sized batch mislabeled with that batch_size)")

    device = args.device
    if args.launch_cpu:
        if device != "cpu":
            print(f"--launch_cpu set: overriding --device {device} -> cpu")
        device = "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: --device {device} requested but CUDA is not available, falling back to cpu")
        device = "cpu"

    print(f"=== {case_stem}: solving {nb_pf} scenarios ({args.sample_data_meth}, seed={args.seed}) with lightsim2grid ===")
    pm_cases, solve_timings = solve_all_scenarios(case_stem, args.sample_data_meth, nb_pf, args.seed)
    print(f"  converged: {solve_timings['n_converged']}/{solve_timings['n_requested']} "
          f"({solve_timings['scenarios_per_s']:.0f} pf/s)")

    results = {
        "data_generation": solve_timings,
        "device": device,
        "batch_sizes": batch_sizes,
        "n_warmup_batches": args.n_warmup_batches,
        "max_batches": args.max_batches,
        "models": {},
    }

    nm_results = f"ml_injection_{case_stem}_{args.sample_data_meth}_{nb_pf}_{device.replace(':', '')}"
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, args.add_to_name)
    os.makedirs(PATH_RESULTS, exist_ok=True)

    for model_name in tqdm(args.models, desc="models"):
        config_name = MODEL_CONFIGS[model_name]
        print(f"=== {model_name} ({config_name}), device={device} ===")
        try:
            results["models"][model_name] = time_model(
                model_name, config_name, pm_cases, case_stem, batch_sizes, device,
                args.n_warmup_batches, args.max_batches
            )
        except Exception as e:
            # Anything NOT already handled inside time_model (OOM is caught there) -- e.g. a
            # model-specific bug triggered by this case's data (see e.g. the GNS multi-slack-gen
            # IndexError) -- shouldn't take down the rest of the --models sweep. Recorded like the
            # OOM case so the output JSON stays uniform, and the model is skipped entirely.
            print(f"  ERROR while timing {model_name}: {e!r} -- skipping this model entirely")
            traceback.print_exc()
            results["models"][model_name] = {
                "error": "EXCEPTION", "error_detail": repr(e), "by_batch_size": {},
            }
            _empty_cuda_cache(device)
        # Flushed after every model (not just once at the end) -- on a large grid, a single
        # ml_injection.py run can take a long time across every model/batch_size, and without
        # this an interruption (crash, Ctrl-C, OOM outside the caught try/except) partway
        # through would lose every already-finished model's results too, not just the one in
        # progress.
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(results, fp=f, indent=2)
        print(f"  wrote {complete_full_path} (through {model_name})")

    print(f"Wrote {complete_full_path}")
