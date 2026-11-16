# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Chunked vs continuous scheduling of a GPU N-1 contingency analysis.

For each configuration the analysis is built once (construction, base case
and cuDSS warm-up excluded) and ``compute()`` is timed ``--repeat`` times (the
fastest kept). Reported per configuration: wall time of compute(), the rows
whose residual is below ``tol`` (RowStatus CONVERGED), and for the continuous
schedule the rounds run, the slot occupancy and the scheduler's host time.
See docs/dev_notes/continuous_batching.md.
"""

import argparse
import time

import numpy as np

from gpusim2grid import ContingencyAnalysisGPU, RowStatus, warmup
from _grid_setup import load_grid


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid_name", default="case6515rte",
                        help="grid2op env name or pandapower network name (default: case6515rte)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="chunk size / slots S (default: 512)")
    parser.add_argument("--nb_iter", type=int, nargs="+", default=[4, 6],
                        help="chunked nb_iter values (default: 4 6)")
    parser.add_argument("--budget", type=int, default=8,
                        help="continuous: per-row nb_iter budget (default: 8)")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4],
                        help="continuous: nb_iter_per_round values (default: 1 2 4)")
    parser.add_argument("--max_cont", type=int, default=10_000,
                        help="at most this many N-1 contingencies, drawn at random (default: 10000)")
    parser.add_argument("--handle_disconnected_grid", action="store_true",
                        help="solve islanding contingencies on their main component")
    parser.add_argument("--repeat", type=int, default=3,
                        help="compute() calls per configuration, fastest kept (default: 3)")
    return parser


def bench(grid, contingencies, batch_size, repeat, **kwargs):
    ca = ContingencyAnalysisGPU(grid, **kwargs)
    ca.add_contingencies_by_branch_id(contingencies)
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        ca.compute(batch_size=batch_size)
        wall = 1e3 * (time.perf_counter() - t0)
        best = wall if best is None else min(best, wall)
    t = ca.timings
    st = ca.get_row_status()
    return {
        "wall_ms": best,
        "converged": int((st == RowStatus.CONVERGED).sum()),
        "simulated": int((st != RowStatus.NOT_SIMULATED).sum()),
        "rounds": t.n_rounds, "occupancy": t.occupancy,
        "schedule_ms": t.t_schedule.wall_ms, "chunks": t.n_chunks,
        "n_refactorize": t.n_refactorize,
    }


def main(args):
    grid, _, _, n_sub, _, _ = load_grid(args.grid_name)
    n_branches = len(grid.get_lines()) + len(grid.get_trafos())
    ids = np.arange(n_branches)
    if n_branches > args.max_cont:
        ids = np.sort(np.random.default_rng(0).choice(ids, size=args.max_cont, replace=False))
    contingencies = [[int(i)] for i in ids]
    print(f"GPU warm-up: {warmup():.1f} ms")
    common = dict(handle_disconnected_grid=args.handle_disconnected_grid)
    rows = []
    for nb in args.nb_iter:
        r = bench(grid, contingencies, args.batch_size, args.repeat, nb_iter=nb, **common)
        rows.append((f"chunked   nb_iter={nb}", r))
    for k in args.k:
        r = bench(grid, contingencies, args.batch_size, args.repeat, nb_iter=args.budget,
                  scheduling="continuous", nb_iter_per_round=k, **common)
        rows.append((f"continuous k={k} budget={args.budget}", r))

    print(f"\n{args.grid_name}: {n_sub} buses, {len(contingencies)} N-1 rows, "
          f"batch_size = {args.batch_size}")
    print(f"{'configuration':<32}{'wall ms':>10}{'converged':>12}{'rounds':>8}"
          f"{'occupancy':>11}{'sched ms':>10}{'refact.':>9}")
    for name, r in rows:
        occ = f"{100 * r['occupancy']:.1f} %" if r["rounds"] else "-"
        rounds = r["rounds"] if r["rounds"] else f"{r['chunks']} ch."
        print(f"{name:<32}{r['wall_ms']:>10.1f}{r['converged']:>7}/{r['simulated']:<5}"
              f"{rounds:>8}{occ:>11}{r['schedule_ms']:>10.2f}{r['n_refactorize']:>9}")


if __name__ == "__main__":
    main(get_parser().parse_args())
