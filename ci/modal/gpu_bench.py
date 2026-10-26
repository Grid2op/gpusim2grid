"""N-1 security analysis throughput across GPU types, via Modal.

Runs a full N-1 study (every line and trafo tripped once) on open pandapower
grids with ``ContingencyAnalysisGPU``, on several GPU types in parallel, and
reports the best contingencies/second each GPU reaches. The batch size giving
that maximum is searched for on each GPU, in the same container: since a
cold run (new batch size: allocation + cuDSS analysis) and the hot runs that
follow it are cheap compared with a container start, trying a dozen batch
sizes costs a few seconds per grid.

Like ``gpu_tests.py``, nothing is compiled here: it installs the wheels a
build run produced (lightsim2grid + gpusim2grid)::

    G2G_WHEELS=path/to/wheels modal run ci/modal/gpu_bench.py
    G2G_WHEELS=... modal run ci/modal/gpu_bench.py --gpus T4,L40S,H100 \\
        --grids case1354pegase,case9241pegase --out bench.json

Precision is the wheel's (FP64 or FP32): run once per wheel to compare them.
The CI wheel is compiled for sm_75 only; it runs on newer GPUs through PTX
JIT, but for numbers worth publishing build it with e.g.
``-DCMAKE_CUDA_ARCHITECTURES="75;80;89;90"``.

Batch-size search (per GPU, per grid):

1. doubling from ``--min-batch`` up to the study size, stopping after two
   sizes in a row below 90% of the best so far, or at the first failure
   (typically out of device memory);
2. refinement: the geometric midpoints between the best size and its
   neighbours.

Each size is timed on ``--reps`` hot runs (same batch size and contingency
list, so no new analysis); its score is the best of them. The study is tiled
to at least ``--min-cont`` contingencies so a small grid does not cap the
batch size.

Configuration (image-definition time, as in gpu_tests.py):

    G2G_CUDA   nvidia/cuda runtime image version   (default 13.4.2)
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

CUDA = os.environ.get("G2G_CUDA", "13.4.2")
CUDA_MAJOR = CUDA.split(".")[0]
PY = "3.10"  # must match the wheels (ubuntu22.04's python3.10)

DEFAULT_GPUS = "T4,L4,L40S,A100-80GB,H100"
DEFAULT_GRIDS = "case1354pegase,case2869pegase,case9241pegase"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-runtime-ubuntu22.04", add_python=PY)
    .pip_install(f"nvidia-cudss-cu{CUDA_MAJOR}>=0.8,<1", "numpy", "scipy", "pandapower")
    .env({"G2G_CUDA": CUDA})
)
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    WHEELS = os.environ.get("G2G_WHEELS", str(REPO / "ci" / "modal" / "wheels"))
    image = image.add_local_dir(WHEELS, "/wheels")

app = modal.App("gpusim2grid-gpu-bench")


def _install_wheels():
    wheels = sorted(str(p) for p in Path("/wheels").glob("*.whl"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *wheels],
                   check=True)
    # cuDSS ships as a pip wheel: make its lib dir visible before the
    # extension is first imported (the loader reads LD_LIBRARY_PATH at
    # dlopen time only for new processes, so preload it explicitly).
    import ctypes
    import importlib
    cudss = Path(list(importlib.import_module(f"nvidia.cu{CUDA_MAJOR}").__path__)[0]) / "lib"
    for lib in sorted(cudss.glob("libcudss*.so*")):
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # optional plugins (e.g. the MPI/NCCL comm layers)


def _gpu_info():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"], text=True).strip().splitlines()[0]
    name, mem, driver = (s.strip() for s in out.split(","))
    return {"gpu_name": name, "gpu_memory": mem, "driver": driver}


def _load_grid(case):
    import warnings
    import numpy as np
    import pandapower.networks as pn
    from gpusim2grid._ls2g_utils import grid_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = grid_from_pandapower(getattr(pn, case)())
    n_bus = grid.get_bus_vn_kv().shape[0]
    v = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1)
    v = grid.ac_pf(v, 10, 1e-8)
    if v.shape[0] == 0:
        raise RuntimeError(f"{case}: base-case AC power flow did not converge")
    n_branches = len(grid.get_lines()) + len(grid.get_trafos())
    return grid, n_bus, n_branches


class _Bench:
    """One grid on one GPU: rebuilds the solver after a failed size."""

    def __init__(self, grid, conts, nb_iter, strategy, reps):
        self.grid, self.conts = grid, conts
        self.nb_iter, self.strategy, self.reps = nb_iter, strategy, reps
        self.ca = None

    def _make(self):
        from gpusim2grid import ContingencyAnalysisGPU
        self.ca = ContingencyAnalysisGPU(self.grid, nb_iter=self.nb_iter,
                                         max_iter_base=10, tol_base=1e-8)
        self.ca.strategy = self.strategy
        self.ca.add_contingencies_by_branch_id(self.conts)

    def measure(self, batch_size):
        import gc
        import numpy as np
        if self.ca is None:
            self._make()
        n = len(self.conts)
        try:
            t0 = time.perf_counter()
            self.ca.compute(batch_size=batch_size)       # cold: alloc + analysis
            res = self.ca.last_residuals()               # D->H, forces a sync
            t_cold = time.perf_counter() - t0
            ct = self.ca.timings
            cold = (float(ct.t_analysis_ms), ct.cudss_mem_device_peak_bytes / 2**20)
            walls = []
            for _ in range(self.reps):                   # hot: same settings
                t0 = time.perf_counter()
                self.ca.compute(batch_size=batch_size)
                res = self.ca.last_residuals()
                walls.append(time.perf_counter() - t0)
            t = self.ca.timings
        except Exception as exc:  # out of memory, cuDSS failure, ...
            self.ca = None
            gc.collect()
            return {"batch_size": batch_size, "ok": False, "error": repr(exc)[:300]}
        best = min(walls)
        return {
            "batch_size": batch_size,
            "ok": True,
            "used_batch_size": int(self.ca.solver.used_batch_size),
            "n_chunks": int(t.n_chunks),
            "throughput_ctg_s": n / best,
            "wall_ms_best": 1e3 * best,
            "wall_ms_all": [1e3 * w for w in walls],
            "chunks_wall_ms": float(t.t_chunks_total_wall_ms),
            "cold_wall_ms": 1e3 * t_cold,
            "analysis_ms": cold[0],
            "cudss_mem_device_peak_mb": cold[1],
            "n_converged_1e-4": int(np.sum(res < 1e-4)),
            "max_residual": float(np.nanmax(res)) if np.isfinite(res).any() else float("nan"),
        }


def _search(bench, n_cont, min_batch):
    trials = {}

    def run(bs):
        bs = int(max(1, min(bs, n_cont)))
        if bs not in trials:
            trials[bs] = bench.measure(bs)
            r = trials[bs]
            msg = (f"{r['throughput_ctg_s']:10.0f} ctg/s  ({r['wall_ms_best']:.1f} ms, "
                   f"{r['n_chunks']} chunks)") if r["ok"] else f"FAILED {r['error'][:120]}"
            print(f"    batch {bs:6d}: {msg}", flush=True)
        return trials[bs]

    def best():
        ok = [r for r in trials.values() if r["ok"]]
        return max(ok, key=lambda r: r["throughput_ctg_s"]) if ok else None

    # 1. doubling
    bs, below = min_batch, 0
    while True:
        r = run(bs)
        if not r["ok"]:
            break
        b = best()
        below = below + 1 if r["throughput_ctg_s"] < 0.9 * b["throughput_ctg_s"] else 0
        if below >= 2 or bs >= n_cont:
            break
        bs = min(2 * bs, n_cont)

    # 2. refinement around the best: geometric midpoints with its neighbours
    for _ in range(2):
        b = best()
        if b is None:
            break
        sizes = sorted(k for k, r in trials.items() if r["ok"] or k > b["batch_size"])
        i = sizes.index(b["batch_size"])
        cands = []
        if i > 0:
            cands.append(round(math.sqrt(sizes[i - 1] * sizes[i])))
        if i + 1 < len(sizes):
            cands.append(round(math.sqrt(sizes[i] * sizes[i + 1])))
        cands = [c for c in cands if c not in trials]
        if not cands:
            break
        for c in cands:
            run(c)

    return best(), sorted(trials.values(), key=lambda r: r["batch_size"])


@app.function(image=image, gpu="T4", timeout=45 * 60, cpu=4)
def bench_gpu(grids, min_cont, min_batch, nb_iter, strategy, reps):
    _install_wheels()
    info = _gpu_info()
    from gpusim2grid import _gpusim2grid as m, warmup
    info["precision"] = "fp32" if m.is_fp32 else "fp64"
    print(f"== {info['gpu_name']} ({info['gpu_memory']}), {info['precision']}", flush=True)
    warmup()

    out = dict(info, grids={})
    for case in grids:
        try:
            grid, n_bus, n_branches = _load_grid(case)
        except Exception as exc:
            out["grids"][case] = {"error": repr(exc)[:300]}
            continue
        n1 = [[b] for b in range(n_branches)]
        reps_n1 = max(1, math.ceil(min_cont / n_branches))
        conts = n1 * reps_n1
        print(f"  {case}: {n_bus} buses, {n_branches} branches, "
              f"{len(conts)} contingencies (N-1 x{reps_n1})", flush=True)
        bench = _Bench(grid, conts, nb_iter, strategy, reps)
        best, trials = _search(bench, len(conts), min_batch)
        out["grids"][case] = {"n_bus": n_bus, "n_branches": n_branches,
                              "n_contingencies": len(conts), "best": best,
                              "trials": trials}
        del bench
    return out


def _table(results, grids):
    lines = []
    head = "| GPU | precision | " + " | ".join(grids) + " |"
    lines += [head, "|" + "---|" * (2 + len(grids))]
    for r in results:
        cells = []
        for g in grids:
            gr = r.get("grids", {}).get(g, {})
            b = gr.get("best")
            cells.append(f"{b['throughput_ctg_s']:,.0f} ctg/s @ {b['batch_size']}"
                         if b else ("error" if "error" in gr or "error" in r else "-"))
        name = r.get("gpu_name", r.get("requested_gpu", "?"))
        lines.append(f"| {name} | {r.get('precision', '?')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@app.local_entrypoint()
def main(gpus: str = DEFAULT_GPUS, grids: str = DEFAULT_GRIDS, min_cont: int = 20_000,
         min_batch: int = 32, nb_iter: int = 4, strategy: str = "direct_refactor_every",
         reps: int = 3, out: str = "gpu_bench.json"):
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    grid_list = [g.strip() for g in grids.split(",") if g.strip()]
    args = (grid_list, min_cont, min_batch, nb_iter, strategy, reps)
    calls = {g: bench_gpu.with_options(gpu=g).spawn(*args) for g in gpu_list}
    results = []
    for g, call in calls.items():
        try:
            r = call.get()
        except Exception as exc:
            r = {"error": repr(exc)[:300]}
        r["requested_gpu"] = g
        results.append(r)
    meta = {"nb_iter": nb_iter, "strategy": strategy, "min_cont": min_cont,
            "reps": reps, "cuda_image": CUDA}
    Path(out).write_text(json.dumps({"config": meta, "results": results}, indent=1))
    table = _table(results, grid_list)
    Path(out).with_suffix(".md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwritten: {out}, {Path(out).with_suffix('.md')}")
