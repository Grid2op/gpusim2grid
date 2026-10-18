# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Several small cuDSS handles against one large one: the cost of one Newton round's linear
solve (REFACTORIZATION + SOLVE) on nb_th independent handles of ``--queue_size`` matrices each,
next to one handle of ``nb_th * queue_size`` matrices (docs/dev_notes/outer_loops_performance.md,
"Small queues under the 16-matrix cliff").

cuDSS 0.8.0's uniform batch is cheap below 16 matrices and an order of magnitude dearer from 16
(an NVIDIA-confirmed bug), so nb_th handles of 15 may beat one handle of the same slot count. The
small handles are launched three ways: from one stream (``serial``), one stream each from one
host thread (``streams``), one stream and one host thread each (``threads``, each handle running
its own rounds). Everything runs through ``benchmark_cudss_batch_raw``, the driver's own
``CudssBatchSolver`` on the Jacobian of a solved lightsim2grid grid.

The grid: a pypowsybl network or IIDM file loaded as ``outer_loops.py`` does (``--network``), or
a lightsim2grid binary (``--binary``, ``LSGrid.save_binary``). ``--out`` writes the table as JSON
for the schedule replay.
"""

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from gpusim2grid import _gpusim2grid as _c, warmup

LAUNCHES = ("serial", "streams", "threads")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--network", default="ieee300",
                        help="pypowsybl network name (ieee14, ieee118, ieee300, ...) or an IIDM file")
    parser.add_argument("--binary", default=None, help="a lightsim2grid binary (.l2gb) instead of --network")
    parser.add_argument("--queue_size", type=int, default=15, help="matrices per small handle")
    parser.add_argument("--nb_th", default="1,2,4,8,16,32,64", help="numbers of small handles")
    parser.add_argument("--sizes", default="8,15,16,32,64,128,256,512,1024,2048",
                        help="extra sizes of the one large handle (nb_th * queue_size are always added)")
    parser.add_argument("--rounds", type=int, default=20, help="timed rounds per measurement")
    parser.add_argument("--ir_n_steps", type=int, default=1, help="cuDSS iterative refinement steps per solve")
    parser.add_argument("--reordering_alg", default="default",
                        help="cuDSS reordering ('none' avoids the batch NaN issue some grids hit with 'default')")
    parser.add_argument("--out", default=None, help="JSON file for the results")
    return parser


def jacobian(args):
    """J of the grid at its solution (lightsim2grid's own ledger, CSR, sorted indices)"""
    if args.binary is not None:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
        grid = LSGrid.load_binary(args.binary)
        v0 = np.full(grid.total_bus(), 1.0 + 0j)
        name = args.binary
    else:
        from outer_loops import load
        grid, v0 = load(args.network)
        name = args.network
    v = grid.ac_pf(v0, 30, 1e-8)
    if v.shape[0] == 0:
        raise RuntimeError(f"the power flow of {name} did not converge: {grid.get_algo().get_error()}")
    J = sp.csr_matrix(grid.get_J_solver())
    J.sort_indices()
    return name, J


def bench(J, batch_size, args, n_handles=1, launch="serial"):
    """(ms per round, setup ms, matrices whose solution is not finite: cuDSS's batch NaN issue,
    timed all the same)"""
    from gpusim2grid.contingency_analysis import _resolve_reordering_alg
    r = _c.benchmark_cudss_batch_raw(J.shape[0], J.indptr.tolist(), J.indices.tolist(), J.data.tolist(),
                                     batch_size, args.rounds, ir_n_steps=args.ir_n_steps,
                                     reordering_alg=_resolve_reordering_alg(args.reordering_alg),
                                     n_handles=n_handles, handle_launch=launch)
    if n_handles == 1:
        return r.group_cycle_ms, r.analysis_ms + r.factorize_ms, r.n_nonfinite_slots
    return r.handle_round_ms, r.handle_setup_ms, r.n_nonfinite_slots


def main():
    args = get_parser().parse_args()
    warmup()
    name, J = jacobian(args)
    q = args.queue_size
    nb_th = [int(x) for x in args.nb_th.split(",")]
    sizes = sorted({int(x) for x in args.sizes.split(",")} | {k * q for k in nb_th})
    print(f"{name}: J {J.shape[0]} x {J.shape[0]}, {J.nnz} non-zeros; {args.rounds} rounds per point")

    t0 = time.perf_counter()
    large = {}
    print("\none handle of S matrices")
    print(f"{'S':>6} {'ms/round':>9} {'us/matrix':>10} {'NaN':>5}")
    for S in sizes:
        ms, _, nan = bench(J, S, args)
        large[S] = {"round_ms": ms, "nonfinite": nan}
        print(f"{S:6d} {ms:9.2f} {1e3 * ms / S:10.1f} {nan:5d}")

    small = {launch: {} for launch in LAUNCHES}
    print(f"\nnb_th handles of {q} matrices, ms per round (setup ms: creation + analysis + factorization; "
          f"* = some solutions not finite)")
    print(f"{'nb_th':>5} {'S':>6} {'one handle':>10} " + " ".join(f"{l:>18}" for l in LAUNCHES))
    for k in nb_th:
        cells = []
        for launch in LAUNCHES:
            ms, setup, nan = bench(J, q, args, n_handles=k, launch=launch)
            small[launch][k] = {"round_ms": ms, "setup_ms": setup, "nonfinite": nan}
            cells.append(f"{ms:9.2f}{'*' if nan else ' '}({setup:6.0f})")
        one = large[k * q]
        print(f"{k:5d} {k * q:6d} {one['round_ms']:9.2f}{'*' if one['nonfinite'] else ' '} "
              + " ".join(f"{c:>18}" for c in cells))
    print(f"\n{time.perf_counter() - t0:.0f} s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"grid": name, "dim": J.shape[0], "nnz": J.nnz, "queue_size": q, "rounds": args.rounds,
                       "ir_n_steps": args.ir_n_steps, "reordering_alg": args.reordering_alg,
                       "large_round": {str(k): v for k, v in large.items()},
                       "small_round": {l: {str(k): v for k, v in d.items()} for l, d in small.items()}},
                      f, indent=1)


if __name__ == "__main__":
    main()
