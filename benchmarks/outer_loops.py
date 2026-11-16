# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The outer loops on the GPU (``ContingencyAnalysisGPU(outer_loops=True)``) against
lightsim2grid's own ``NROuter_KLU`` contingency analysis: N-1 on every line, time per
contingency, and the rows that disagree (outer iterations, status, voltages).

The grid comes from pypowsybl: a built-in network (``--network ieee300``) or an IIDM file
(``--network path/to/snapshot.xiidm``, e.g. one of ``$LS2G_SNAPSHOTS``), loaded with
OpenLoadFlow's default outer loops (``init_from_pypowsybl(olf_rules=True)``).

Convergence criteria (``--loose``, the default, or ``--tight``), the ones of
``expe_gpus/bench_n1.sh`` so the two benchmarks measure the same thing:

- ``--loose``: OpenLoadFlow's own defaults, read from the installed pypowsybl (1.16.1: 30 outer
  iterations, 30 Newton iterations per inner solve, ``newtonRaphsonConvEpsPerEq`` 1e-4 pu,
  ``slackBusPMaxMismatch`` 1 MW);
- ``--tight``: ``expe_gpus/src/test_launch.py``'s (30 outer, 50 Newton, eps 1e-10 pu,
  ``slackBusPMaxMismatch`` 1e-6 MW).

OpenLoadFlow stops on max |mismatch| < eps (``UNIFORM_CRITERIA``, per unit on its 100 MVA base),
so lightsim2grid gets ``tol = eps * 100`` MVA and gpusim2grid ``tol = eps * 100 / sn_mva`` per unit,
both the same Newton budget; the outer-loop limits (outer iterations, slack mismatch, reactive
mismatch) reach both through the grid (``set_olf_outer_loops``). The 'n' solve uses the same.
``--nb_iter`` / ``--tol_mva`` override the preset's Newton budget / tolerance.

``--distributed_slack_inner_loop``: the grid (and lightsim2grid's reference) runs ``NROuterDist_KLU``
instead of ``NROuter_KLU``, the slack shared inside each inner Newton (gpusim2grid picks that
inner system from the grid's algorithm) rather than by the DistributedSlack outer loop alone.
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
    parser.add_argument("--n_ctg", type=int, default=1000, help="contingencies (lines closed at both ends, at random)")
    parser.add_argument("--seed", type=int, default=0, help="which lines")
    parser.add_argument("--batch_size", type=int, default=256, help="GPU slots")
    conv = parser.add_mutually_exclusive_group()
    conv.add_argument("--loose", dest="conv", action="store_const", const="loose",
                      help="OpenLoadFlow's default convergence criteria (the default)")
    conv.add_argument("--tight", dest="conv", action="store_const", const="tight",
                      help="tight criteria: 50 Newton iterations, 1e-10 pu, slack mismatch 1e-6 MW")
    parser.set_defaults(conv="loose")
    parser.add_argument("--nb_iter", type=int, default=None,
                        help="iteration budget of each inner solve (default: the criteria's)")
    parser.add_argument("--tol_mva", type=float, default=None,
                        help="Newton tolerance, MVA (lightsim2grid's unit; default: the criteria's)")
    parser.add_argument("--ir_n_steps", type=int, default=1,
                        help="cuDSS iterative refinement steps per solve (gpusim2grid's default is 0; "
                             "1 keeps hard inner solves at lightsim2grid's iteration counts)")
    parser.add_argument("--init_from_n", action="store_true",
                        help="the rows resume from the 'n' solve's loops (init_from_n_powerflow)")
    parser.add_argument("--handle_disconnected_grid", action="store_true",
                        help="solve a contingency that islands buses on its main component")
    parser.add_argument("--tail_max_groups", type=int, default=16,
                        help="the continuous tail: at most this many groups of tail_group_size slots "
                             "once the queue is empty (0 = off)")
    parser.add_argument("--tail_group_size", type=int, default=15,
                        help="matrices of the tail's cuDSS handle")
    parser.add_argument("--calibrate_tail", action="store_true",
                        help="print calibrate_tail_max_groups() for this grid and batch size")
    parser.add_argument("--skip_cpu", action="store_true", help="time the GPU only")
    parser.add_argument("--reordering_alg", default="default",
                        help="cuDSS reordering ('ls2g_constrained': lightsim2grid's constrained order, "
                             "docs/dev_notes/batched_lu_and_constrained_ordering.md). Set "
                             "GPUSIM2GRID_COUNT_PIVOTS=1 to count cuDSS's perturbed pivots")
    parser.add_argument("--linear_solver", default="cudss",
                        help="batch linear solver: 'cudss' or 'own_lu' (the batched LU of our own; its order "
                             "from --reordering_alg, 'default' = lightsim2grid's constrained order)")
    parser.add_argument("--distributed_slack_inner_loop", action="store_true",
                        help="NROuterDist_KLU: the slack shared inside each inner Newton (grid and reference)")
    parser.add_argument("--cpu_algo", default=None,
                        help="lightsim2grid's reference algorithm (default NROuter_KLU, NROuterDist_KLU with "
                             "--distributed_slack_inner_loop; *_KLUStaticPivot: its fallback "
                             "factorizations predict the GPU's frozen-pivot failures)")
    return parser


# --tight: expe_gpus/src/test_launch.py's OLF_TIGHT_* (maxOuterLoopIterations stays OpenLoadFlow's)
TIGHT = {"newtonRaphsonConvEpsPerEq": "1e-10", "slackBusPMaxMismatch": "1e-6",
         "maxNewtonRaphsonIterations": "50"}
OLF_KEYS = ("maxOuterLoopIterations", "maxNewtonRaphsonIterations", "newtonRaphsonConvEpsPerEq",
            "slackBusPMaxMismatch")


def convergence(mode):
    """OpenLoadFlow parameters of the criteria (``loose`` or ``tight``), as bench_n1.sh builds
    them, and what lightsim2grid / gpusim2grid get from them: (parameters, Newton budget,
    tolerance in MVA, the criteria for printing)."""
    import pypowsybl as pp
    default = pp.loadflow.get_provider_parameters("OpenLoadFlow")["default"]
    prov = {k: str(default[k]) for k in OLF_KEYS}
    if mode == "tight":
        prov.update(TIGHT)
    elif mode != "loose":
        raise ValueError(f"criteria {mode}: loose or tight")
    prov["newtonRaphsonStoppingCriteriaType"] = "UNIFORM_CRITERIA"
    params = pp.loadflow.Parameters(provider_parameters=prov)
    tol_mva = float(prov["newtonRaphsonConvEpsPerEq"]) * 100.   # per unit on OLF's 100 MVA -> MVA
    return params, int(prov["maxNewtonRaphsonIterations"]), tol_mva, {"mode": mode} | prov


def load(network, params, algo="NROuter_KLU"):
    """The grid as lightsim2grid's own outer-loop comparison builds it
    (utils/olf_outer_compare.py): its default slack generator, half-open lines kept,
    zero-impedance branches fused, OpenLoadFlow's loading rules, its max-voltage-change
    step scaling, ``algo`` (NROuter_KLU or NROuterDist_KLU) with the default loops, their limits from the OpenLoadFlow
    parameters ``params`` (set_olf_outer_loops, as bench_n1.sh). Returns the grid and its start
    (OpenLoadFlow's DC_VALUES: the angles of a DC load flow, 1 pu)."""
    import pypowsybl as pp
    from lightsim2grid.network.from_pypowsybl import (init as init_from_pypowsybl, OlfLoadingParameters,
                                                      set_olf_outer_loops)
    from lightsim2grid.network.from_pypowsybl._aux_add_slack import _default_distributed_slack
    from lightsim2grid.lightsim2grid_cpp import ScalingPolicyType
    if network.endswith((".xiidm", ".iidm", ".xz", ".zip")):
        net = pp.network.load(network, {"iidm.die.with-extensions": "all"})
    else:
        net = getattr(pp.network, f"create_{network}")()
    gen_slack_id = next(iter(_default_distributed_slack(net, net.get_generators())))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id=gen_slack_id, sort_index=False, buses_for_sub=False,
                                   keep_half_open_lines=True, fuse_zero_impedance_branches=True, init_vm_pu=1.,
                                   olf_rules=OlfLoadingParameters(reactive_limits=True, svc_voltage_monitoring=True))
    grid.change_algorithm(algo)
    set_olf_outer_loops(grid, params)   # after the algorithm: changing it resets its loops
    grid.set_keep_vinit_at_group_controlled_buses(True)
    cfg = grid.get_ac_algo_config()
    ip, rp = list(cfg.int_params), list(cfg.real_params)
    ip[0] = int(ScalingPolicyType.MaxVoltageChange)
    rp[0], rp[1] = 1.0, 0.4
    cfg.int_params, cfg.real_params = ip, rp
    grid.set_ac_algo_config(cfg)
    flat = np.full(grid.total_bus(), 1.0 + 0j)
    grid.set_dc_distribute_slack_on_can_participate(True)
    v_dc = grid.dc_pf(flat, 50, 1e-8)
    v0 = np.exp(1j * np.angle(v_dc)) if v_dc.shape[0] > 0 else flat
    return grid, v0


def print_breakdown(t, compute_ms):
    """Where compute() went (wall ms): the Newton rounds on the GPU, the slot loads and
    evictions, and the scheduler's per-round host work split into the residual sync, the
    outer-loop gather, the host model's decisions and the rest; the controls upload and the
    model's slot loads apart; then what no phase covers, from the driver's and the session's
    whole-call wall clocks. Every GPU phase is timed with a sync, so the parts add up to wall
    time."""
    w = lambda *names: sum(getattr(t, n).wall_ms for n in names)
    newton = w("t_spmv", "t_fill_F", "t_fill_J", "t_first_factorize", "t_refactorize",
               "t_solve", "t_update_V")
    load = w("t_tile_V", "t_tile_Ybus", "t_patch_Ybus", "t_tile_Sbus")
    evict = w("t_store_V", "t_violation_check", "t_flow_computation")
    sched = t.t_schedule.wall_ms
    parts = {"res_sync": t.t_sched_res_sync.wall_ms, "oc_gather": t.t_oc_gather.wall_ms,
             "oc_decide": t.t_oc_decide.wall_ms}
    parts["other"] = sched - sum(parts.values())
    rows = [("Newton rounds (GPU)", newton), ("residual pass + evict residuals", t.t_residual.wall_ms),
            ("slot loads", load), ("evictions (store V, checks, flows)", evict),
            ("schedule (host, per round)", sched),
            *[(f"  {k}", v) for k, v in parts.items()],
            ("outer controls upload", t.t_oc_controls.wall_ms),
            ("outer model loads (begin, load)", t.t_oc_load.wall_ms),
            ("tail compaction (moves)", t.t_tail_compact.wall_ms),
            ("driver, untimed", t.t_driver_solve_wall_ms - t.t_chunks_total_wall_ms),
            ("session around solve() (cold build)", t.t_session_run_wall_ms - t.t_driver_solve_wall_ms),
            ("  of which cuDSS ANALYSIS", t.t_analysis_ms),
            ("outside the session (Python)", compute_ms - t.t_session_run_wall_ms)]
    n = max(t.n_rounds, 1)
    print(f"     breakdown of compute() ({compute_ms:.0f} ms, {t.n_rounds} rounds):")
    for name, v in rows:
        print(f"       {name:<38}{v:9.1f} ms {100 * v / compute_ms:5.1f} %  {1e3 * v / n:8.1f} us/round")


def main(args):
    params, nb_iter, tol_mva, crit = convergence(args.conv)
    args.nb_iter = nb_iter if args.nb_iter is None else args.nb_iter
    args.tol_mva = tol_mva if args.tol_mva is None else args.tol_mva
    print(f"convergence: {crit}, Newton budget {args.nb_iter}, tol {args.tol_mva:g} MVA")
    algo = "NROuterDist_KLU" if args.distributed_slack_inner_loop else "NROuter_KLU"
    args.cpu_algo = args.cpu_algo or algo
    grid, v0 = load(args.network, params, algo)
    t0 = time.perf_counter()
    if grid.ac_pf(v0, args.nb_iter, args.tol_mva).size == 0:
        raise RuntimeError("the 'n' solve did not converge")
    print(f"n solve: {1e3 * (time.perf_counter() - t0):.1f} ms, "
          f"{grid.get_algo().get_outer_loop_stats().nb_outer_iterations} outer iterations, "
          f"loops {[l.name() for l in grid.get_outer_loops()]}, algorithm {grid.get_algo_name()}")
    # lines closed at both ends, drawn at random (the same ones for a seed)
    closed = [i for i, line in enumerate(grid.get_lines()) if line.connected_global]
    rng = np.random.default_rng(args.seed)
    ctgs = sorted(rng.choice(closed, size=min(args.n_ctg, len(closed)), replace=False).tolist())
    start = v0 if args.init_from_n else np.asarray(grid.get_V(), dtype=complex)

    warmup()
    t0 = time.perf_counter()
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=args.nb_iter,
                                init_from_n_powerflow=args.init_from_n,
                                handle_disconnected_grid=args.handle_disconnected_grid,
                                tol=args.tol_mva / grid.get_sn_mva(), ir_n_steps=args.ir_n_steps,
                                # the 'n' state is checked against tol_base (pu): never tighter than 1e-8
                                # (tight: the 'n' solve stops right at tol, a recomputed mismatch may be an
                                # ulp above it), as bench_n1.sh
                                max_iter_base=args.nb_iter,
                                tol_base=max(args.tol_mva / grid.get_sn_mva(), 1e-8),
                                reordering_alg=args.reordering_alg, linear_solver=args.linear_solver)
    ca.tail_max_groups, ca.tail_group_size = args.tail_max_groups, args.tail_group_size
    ca.add_contingencies_by_branch_id([[c] for c in ctgs])
    t_build = time.perf_counter() - t0
    if args.calibrate_tail:
        print(f"tail calibration: {ca.calibrate_tail_max_groups(batch_size=min(args.batch_size, len(ctgs)))}")
    t0 = time.perf_counter()
    ca.compute(batch_size=args.batch_size)
    t_gpu = time.perf_counter() - t0
    status = np.asarray(ca.get_row_status())
    res = ca.get_outer_results()
    nb_outer = np.array([r.nb_outer_iterations for r in res])
    t = ca.timings
    print(f"GPU: build {1e3 * t_build:.0f} ms, compute {1e3 * t_gpu:.0f} ms "
          f"({1e3 * t_gpu / len(ctgs):.2f} ms / contingency), rounds {t.n_rounds}, "
          f"occupancy {t.occupancy:.2f}, tail rounds {t.n_tail_rounds} ({t.n_slot_moves} row moves)")
    print(f"     statuses {dict(zip(*np.unique(status, return_counts=True)))}, "
          f"outer iterations mean {nb_outer.mean():.2f} max {nb_outer.max()}")
    print(f"     {args.linear_solver}: reordering {args.reordering_alg}, analysis {t.t_analysis_ms:.0f} ms, "
          f"user perm honoured {t.cudss_user_perm_honoured}, "
          f"LU nnz {t.cudss_lu_nnz}, perturbed pivots {t.cudss_npivots_total} over "
          f"{t.cudss_npivots_reads} factorizations (max {t.cudss_npivots_max})")
    print_breakdown(t, 1e3 * t_gpu)
    if args.skip_cpu:
        return

    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm(args.cpu_algo)
    batch.init_from_n_powerflow = args.init_from_n
    batch.handle_disconnected_grid = args.handle_disconnected_grid
    for c in ctgs:
        batch.add_n1(c)
    t0 = time.perf_counter()
    batch.compute(start, args.nb_iter, args.tol_mva)
    t_cpu = time.perf_counter() - t0
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res_ls = batch.get_outer_results()
    st = batch.get_linear_solver_stats()
    print(f"     CPU linear solver: {st}")
    print(f"CPU (lightsim2grid {args.cpu_algo}, 1 thread): {1e3 * t_cpu:.0f} ms "
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
