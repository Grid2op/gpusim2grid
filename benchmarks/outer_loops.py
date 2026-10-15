# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The outer loops on the GPU (``ContingencyAnalysisGPU(outer_loops=True)``) against
lightsim2grid's own ``NROuter_KLU`` contingency analysis: N-1 on every line, time per
contingency, and the rows that disagree (outer iterations, status, voltages).

The grid comes from pypowsybl: a built-in network (``--network ieee300``) or an IIDM file
(``--network path/to/snapshot.xiidm``, e.g. one of ``$LS2G_SNAPSHOTS``), loaded with
OpenLoadFlow's default outer loops (``init_from_pypowsybl(olf_rules=True)``).
"""

import argparse
import time
import warnings

import numpy as np

from gpusim2grid import ContingencyAnalysisGPU, warmup


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--network", default="ieee300",
                        help="pypowsybl network name (ieee14, ieee118, ieee300, ...) or an IIDM file")
    parser.add_argument("--n_ctg", type=int, default=1000, help="contingencies (lines, first ones)")
    parser.add_argument("--batch_size", type=int, default=256, help="GPU slots")
    parser.add_argument("--nb_iter", type=int, default=30, help="iteration budget of each inner solve")
    parser.add_argument("--tol_mva", type=float, default=1e-6, help="Newton tolerance, MVA (lightsim2grid's unit)")
    parser.add_argument("--init_from_n", action="store_true",
                        help="the rows resume from the 'n' solve's loops (init_from_n_powerflow)")
    parser.add_argument("--handle_disconnected_grid", action="store_true",
                        help="solve a contingency that islands buses on its main component")
    parser.add_argument("--skip_cpu", action="store_true", help="time the GPU only")
    return parser


def load(network):
    import pypowsybl as pp
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl
    net = pp.network.load(network) if network.endswith((".xiidm", ".iidm", ".xz", ".zip")) \
        else getattr(pp.network, f"create_{network}")()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(net, olf_rules=True)
    grid.change_algorithm("NROuter_KLU")
    return grid


def main(args):
    grid = load(args.network)
    v0 = np.full(grid.total_bus(), 1.0 + 0j)
    t0 = time.perf_counter()
    if grid.ac_pf(v0, args.nb_iter, args.tol_mva).size == 0:
        raise RuntimeError("the 'n' solve did not converge")
    print(f"n solve: {1e3 * (time.perf_counter() - t0):.1f} ms, "
          f"{grid.get_algo().get_outer_loop_stats().nb_outer_iterations} outer iterations, "
          f"loops {[l.name() for l in grid.get_outer_loops()]}")
    n_lines = len(grid.get_lines())
    ctgs = list(range(min(args.n_ctg, n_lines)))
    start = v0 if args.init_from_n else np.asarray(grid.get_V(), dtype=complex)

    warmup()
    t0 = time.perf_counter()
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=args.nb_iter,
                                init_from_n_powerflow=args.init_from_n,
                                handle_disconnected_grid=args.handle_disconnected_grid,
                                tol=args.tol_mva / grid.get_sn_mva())
    ca.add_contingencies_by_branch_id([[c] for c in ctgs])
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    ca.compute(batch_size=args.batch_size)
    t_gpu = time.perf_counter() - t0
    status = np.asarray(ca.get_row_status())
    res = ca.get_outer_results()
    nb_outer = np.array([r.nb_outer_iterations for r in res])
    t = ca.timings
    print(f"GPU: build {1e3 * t_build:.0f} ms, compute {1e3 * t_gpu:.0f} ms "
          f"({1e3 * t_gpu / len(ctgs):.2f} ms / contingency), rounds {t.n_rounds}, "
          f"occupancy {t.occupancy:.2f}")
    print(f"     statuses {dict(zip(*np.unique(status, return_counts=True)))}, "
          f"outer iterations mean {nb_outer.mean():.2f} max {nb_outer.max()}")
    if args.skip_cpu:
        return

    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm("NROuter_KLU")
    batch.init_from_n_powerflow = args.init_from_n
    batch.handle_disconnected_grid = args.handle_disconnected_grid
    for c in ctgs:
        batch.add_n1(c)
    t0 = time.perf_counter()
    batch.compute(start, args.nb_iter, args.tol_mva)
    t_cpu = time.perf_counter() - t0
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res_ls = batch.get_outer_results()
    print(f"CPU (lightsim2grid NROuter_KLU, 1 thread): {1e3 * t_cpu:.0f} ms "
          f"({1e3 * t_cpu / len(ctgs):.2f} ms / contingency) -> GPU speed-up {t_cpu / t_gpu:.1f}x")

    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(ctgs), -1)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    diff_outer = [i for i in range(len(ctgs)) if res[i].nb_outer_iterations != res_ls[i].nb_outer_iterations]
    diff_conv = [i for i in range(len(ctgs)) if (status[i] == 0) != conv[i]]
    both = (status == 0) & conv
    # masked buses: NaN here; compare the finite entries, and that both mask the same ones
    fin = np.isfinite(V[both])
    diff = np.where(fin, np.abs(V[both] - V_ls[both]), 0.)
    err = np.nan_to_num(diff, nan=np.inf).max(axis=1) if both.any() else np.zeros(0)
    print(f"agreement: {len(diff_conv)} rows converge on one side only, {len(diff_outer)} rows with "
          f"another number of outer iterations, max |dV| {err.max() if err.size else 0.:.2e} "
          f"(> 1e-6 on {int((err > 1e-6).sum())} rows), {int((~fin).sum())} masked bus voltages")


if __name__ == "__main__":
    main(get_parser().parse_args())
