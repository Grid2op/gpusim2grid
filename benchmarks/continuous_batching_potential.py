# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Upper bound of what continuous batching can save, with no code change.

In chunk mode every row runs exactly ``nb_iter`` Newton iterations and a row's
trajectory does not depend on ``nb_iter`` (no convergence test, the same chunk
composition). So solving the same N-1 analysis with ``nb_iter = 1 ... N`` and
recording, per row, the first ``nb_iter`` whose residual is below ``tol`` gives
the number of iterations each row needs.

From that distribution the script replays, on the host, the continuous
scheduler of docs/dev_notes/continuous_batching.md (S slots, k iterations per
round, a row leaves once converged or out of budget, its slot refilled from
the queue) and compares the batch iterations it would run with chunk mode's
``n_chunks * nb_iter``. It ignores the per-round residual pass and the
scheduling overhead: an upper bound of the gain.
"""

import argparse
import math
import time

import numpy as np

from gpusim2grid import ContingencyAnalysisGPU, warmup
from gpusim2grid.compilation_options import is_fp32
from _grid_setup import load_grid


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid_name", default="case300",
                        help="grid2op env name or pandapower network name (default: case300)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="slots S / chunk size (default: 512)")
    parser.add_argument("--nb_iter_max", type=int, default=8,
                        help="largest nb_iter measured; also the continuous row budget (default: 8)")
    parser.add_argument("--nb_iter_chunk", type=int, default=4,
                        help="chunk-mode nb_iter to compare against (default: 4)")
    parser.add_argument("--tol", type=float, default=None,
                        help="convergence tolerance, ||F||inf < tol (default: 1e-8, 1e-4 in FP32)")
    parser.add_argument("--max_cont", type=int, default=10_000,
                        help="at most this many N-1 contingencies, drawn at random (default: 10000)")
    parser.add_argument("--handle_disconnected_grid", action="store_true",
                        help="solve islanding contingencies on their main component")
    parser.add_argument("--strategy", default="direct_refactor_every",
                        choices=["direct_refactor_every", "direct_base_case_factors"])
    return parser


def needed_iterations(ca, batch_size, nb_iter_max, tol):
    """(n_rows,) first nb_iter with residual < tol; nb_iter_max + 1 if none,
    -1 for a row the pre-check dropped (never simulated). Also the wall time
    of each chunk-mode run."""
    need = None
    wall = {}
    for nb in range(1, nb_iter_max + 1):
        ca.solver.nb_iter = nb
        t0 = time.perf_counter()
        ca.compute(batch_size=batch_size)
        wall[nb] = 1e3 * (time.perf_counter() - t0)
        res = np.asarray(ca.last_residuals(), dtype=float)
        if need is None:
            need = np.full(res.shape, nb_iter_max + 1, dtype=int)
            need[~np.isfinite(res)] = -1   # provisional: NaN at nb = 1 is "not simulated"
            not_sim = ~np.isfinite(res)
        hit = (need == nb_iter_max + 1) & np.isfinite(res) & (res < tol)
        need[hit] = nb
    need[not_sim] = -1
    return need, wall, ca.solver.used_batch_size


def replay_continuous(need, S, k, budget):
    """Rounds the continuous scheduler would run on the measured needs."""
    queue = [int(n) for n in need if n >= 0]
    n_rows = len(queue)
    if n_rows == 0:
        return 0, 0.0
    S = max(1, min(S, n_rows))
    slots = [None] * S          # [needed, iterations done]
    q = 0
    rounds = useful = 0
    for s in range(S):
        slots[s] = [queue[q], 0]
        q += 1
    while any(sl is not None for sl in slots):
        rounds += 1
        for s, sl in enumerate(slots):
            if sl is None:
                continue
            sl[1] += k
            useful += k
            if sl[1] >= sl[0] or sl[1] >= budget:
                if q < n_rows:
                    slots[s] = [queue[q], 0]
                    q += 1
                else:
                    slots[s] = None
    occupancy = useful / (rounds * k * S)
    return rounds, occupancy


def main(args):
    tol = args.tol if args.tol is not None else (1e-4 if is_fp32 else 1e-8)
    grid, _, _, n_sub, _, _ = load_grid(args.grid_name)
    n_lines = len(grid.get_lines())
    n_trafos = len(grid.get_trafos())
    n_branches = n_lines + n_trafos
    ids = np.arange(n_branches)
    if n_branches > args.max_cont:
        ids = np.sort(np.random.default_rng(0).choice(ids, size=args.max_cont, replace=False))
    contingencies = [[int(i)] for i in ids]

    print(f"GPU warm-up: {warmup():.1f} ms")
    ca = ContingencyAnalysisGPU(grid, nb_iter=1, handle_disconnected_grid=args.handle_disconnected_grid)
    ca.strategy = args.strategy
    ca.add_contingencies_by_branch_id(contingencies)

    need, wall, used_bs = needed_iterations(ca, args.batch_size, args.nb_iter_max, tol)
    sim = need >= 0
    n_sim = int(sim.sum())
    print(f"\n{args.grid_name}: {n_sub} buses, {len(contingencies)} N-1 rows, {n_sim} simulated, "
          f"tol = {tol:g}, strategy = {args.strategy}")
    print("iterations needed (first nb_iter with residual < tol):")
    for nb in range(1, args.nb_iter_max + 2):
        c = int((need == nb).sum())
        label = f"> {args.nb_iter_max}" if nb == args.nb_iter_max + 1 else f"{nb:>3}"
        print(f"  {label:>5}: {c:>7}  ({100. * c / max(n_sim, 1):5.1f} %)")

    S = args.batch_size
    n_chunks = math.ceil(n_sim / used_bs) if n_sim else 0
    worst = int(need[sim].max()) if n_sim else 0
    worst = min(worst, args.nb_iter_max)
    print(f"\nchunk mode: {n_chunks} chunk(s) of {used_bs}")
    for nb in sorted({args.nb_iter_chunk, worst}):
        it = n_chunks * nb
        not_conv = int(((need > nb) & sim).sum())
        print(f"  nb_iter = {nb}: {it} batch iterations, {not_conv} row(s) not below tol, "
              f"{wall[nb] if nb in wall else float('nan'):.1f} ms measured")
    print(f"continuous (S = {min(S, max(n_sim, 1))}, budget = {args.nb_iter_max}), "
          "batch iterations = rounds * k, residual pass and scheduling not counted:")
    for k in (1, 2, 4):
        rounds, occ = replay_continuous(need, S, k, args.nb_iter_max)
        print(f"  k = {k}: {rounds} rounds, {rounds * k} batch iterations, occupancy {100. * occ:.1f} %")


if __name__ == "__main__":
    main(get_parser().parse_args())
