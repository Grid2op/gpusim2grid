# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
n1_screen.py -- N-1 contingency screening on the GPU, from the command line.

The grid is a pandapower case (``case118``) or a network file pypowsybl reads (``.biidm``,
``.xiidm``, ...): the latter loaded with OpenLoadFlow's loading rules and its permanent current
limits (benchmarks/topology_actions/conti_x_actions_pypowsybl.py's loader). Every connected
branch is a contingency unless ``--contingencies`` names them (a JSON list of branch ids). Prints
the counts, the time and the most loaded contingencies; ``--out`` writes one CSV row per
contingency. ``--trace`` re-runs it under Nsight Systems (``nsys profile``: CUDA, NVTX, OS runtime,
Python sampling) and writes ``traces/<network>_n1_<settings>.nsys-rep`` in the repository, with
an NVTX range per step (load, warmup, setup, screening, results; the ``nvtx`` package, optional).

Run (on a shared machine, through tools/run_limited.sh):
    python examples/n1_screen.py case118
    python examples/n1_screen.py snapshot.biidm --outer-loops --settings perf --out n1.csv
    python examples/n1_screen.py snapshot.biidm --outer-loops --trace
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gpusim2grid import ContingencyAnalysisGPU

# absolute: the editor stubs cover the compiled module, not the package's relative re-export
from gpusim2grid._gpusim2grid import warmup

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

_REPO: Path = Path(__file__).resolve().parents[1]
_LOADER_DIR: Path = _REPO / "benchmarks" / "topology_actions"
#: set in the environment of the run nsys profiles (so it does not re-exec itself again)
_TRACED: str = "GPUSIM2GRID_N1_SCREEN_TRACED"


@contextlib.contextmanager
def step(name: str) -> Iterator[None]:
    """An NVTX range named ``name`` when the ``nvtx`` package is installed (else nothing)."""
    try:
        import nvtx
    except ImportError:
        yield
        return
    with nvtx.annotate(name):
        yield


def run_under_nsys(args: argparse.Namespace) -> None:
    """Re-run this script under ``nsys profile`` (without ``--trace``), into traces/."""
    nsys = shutil.which("nsys")
    if nsys is None:
        raise SystemExit("--trace needs Nsight Systems' nsys on the PATH")
    name = f"{Path(args.network).stem}_n1_{args.settings}{'_outer' if args.outer_loops else ''}"
    out = _REPO / "traces" / name
    out.parent.mkdir(exist_ok=True)
    argv = [a for a in sys.argv[1:] if a != "--trace"]
    cmd = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--python-sampling=true",
        "--force-overwrite=true",
        "-o",
        str(out),
        sys.executable,
        str(Path(__file__).resolve()),
        *argv,
    ]
    print(f"tracing into {out}.nsys-rep", flush=True)
    os.environ[_TRACED] = "1"
    os.execvp(nsys, cmd)


def load(network: str, outer_loops: bool) -> LSGrid:
    """A solved lightsim2grid grid: a pandapower case by name, or a file pypowsybl reads."""
    if Path(network).suffix:
        import pypowsybl as pp

        sys.path.insert(0, str(_LOADER_DIR))
        from conti_x_actions_pypowsybl import load_grid

        grid = load_grid(pp.network.load(network))  # pyright: ignore[reportPrivateImportUsage]
        if not outer_loops:  # a plain Newton base case: the screening runs without the loops
            grid.change_algorithm("NR_KLU")
            if grid.ac_pf(np.asarray(grid.get_V(), dtype=complex), 30, 1e-6).shape[0] == 0:
                raise RuntimeError("the base case diverged without the outer loops")
        return grid
    from _common import load_case

    return load_case(network)["grid"]


def branch_names(grid: LSGrid) -> list[str]:
    return [ln.name for ln in grid.get_lines()] + [t.name for t in grid.get_trafos()]


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n\n")[0])
    p.add_argument("network", help="pandapower case name (case118) or a network file (.biidm, .xiidm, ...)")
    p.add_argument("--contingencies", help="JSON list of branch ids (default: every connected branch)")
    p.add_argument(
        "--settings",
        choices=["perf", "defaults"],
        default="perf",
        help="perf: own LU, continuous, the tail from the start; defaults: cuDSS, chunked",
    )
    p.add_argument("--outer-loops", action="store_true", help="the grid's outer loops on every row")
    p.add_argument("--threshold", type=float, default=1.0, help="violation_threshold (limits x this)")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--nb-iter", type=int, default=30)
    p.add_argument("--top", type=int, default=10, help="most loaded contingencies printed")
    p.add_argument("--out", help="CSV, one row per contingency")
    p.add_argument("--trace", action="store_true", help="an Nsight Systems trace into traces/ (nsys profile)")
    args = p.parse_args()
    if args.trace and not os.environ.get(_TRACED):
        run_under_nsys(args)

    t0 = time.perf_counter()
    with step("load the grid (CPU)"):
        grid = load(args.network, args.outer_loops)
    t_load = time.perf_counter() - t0
    names = branch_names(grid)
    if args.contingencies:
        index = {n: i for i, n in enumerate(names)}
        wanted: list[str] = json.loads(Path(args.contingencies).read_text())
        branches = [index[c] for c in wanted if c in index]
    else:
        n_lines = len(grid.get_lines())
        branches = [i for i, ln in enumerate(grid.get_lines()) if ln.connected_global]
        branches += [n_lines + i for i, t in enumerate(grid.get_trafos()) if t.connected_global]

    with step("warmup (CUDA / cuDSS init)"):
        warmup()
    kw: dict[str, Any] = {"linear_solver": "own_lu"} if args.settings == "perf" else {}
    if args.settings == "perf" or args.outer_loops:
        kw["scheduling"] = "continuous"  # the outer loops run on the continuous schedule only
    if args.outer_loops:
        kw.update(outer_loops=True, init_from_n_powerflow=False, tol=1e-6 / grid.get_sn_mva(), ir_n_steps=1)
    t0 = time.perf_counter()
    with step("session: base case on the GPU + setup"):
        ca = ContingencyAnalysisGPU(
            grid, nb_iter=args.nb_iter, handle_disconnected_grid=True, compute_limit_violations=True, **kw
        )
        if args.settings == "perf":
            ca.tail_max_groups, ca.tail_group_size = args.batch, 1
        ca.violation_threshold = args.threshold
        ca.add_contingencies_by_branch_id([[b] for b in branches])
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    with step("N-1 screening"):
        ca.compute(batch_size=args.batch)
        res: np.ndarray = ca.last_residuals()
    t_solve = time.perf_counter() - t0
    with step("results: max loading, violations"):
        rho, worst = ca.get_max_loading()
        violating: np.ndarray = ca.get_violation_counts()["current"] > 0

    converged = np.isfinite(res) & (res <= 1e-6)
    print(
        f"{args.network}: {grid.get_Ybus_solver().shape[0]} buses, {len(branches)} contingencies "
        f"({args.settings}{', outer loops' if args.outer_loops else ''})"
    )
    print(
        f"  load {t_load:.1f} s, setup {t_setup:.2f} s, screening {t_solve:.2f} s "
        f"({1e3 * t_solve / max(len(branches), 1):.3f} ms / contingency)"
    )
    print(
        f"  converged {int(converged.sum())}, not converged {int((np.isfinite(res) & ~converged).sum())}, "
        f"not simulated {int(np.isnan(res).sum())}, violating a current limit {int(violating.sum())}"
    )
    ranked = np.argsort(np.where(np.isfinite(rho), -rho, np.inf))[: args.top]
    order = [int(k) for k in ranked if np.isfinite(rho[k])]
    if order:
        print("  most loaded (max |I| / limit):")
        for k in order:
            print(f"    {names[branches[k]]:<30} {rho[k]:7.3f}  on {names[worst[k]]}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contingency", "converged", "residual", "max_rho", "branch", "violating"])
            for k, b in enumerate(branches):
                w.writerow(
                    [
                        names[b],
                        bool(converged[k]),
                        res[k],
                        rho[k],
                        names[worst[k]] if worst[k] >= 0 else "",
                        bool(violating[k]),
                    ]
                )
        print(f"  written {args.out}")


if __name__ == "__main__":
    main()
