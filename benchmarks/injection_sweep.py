"""
Benchmark: GPU batched-injection AC power flow sweep.

Generates log-normal load perturbation scenarios and benchmarks the
InjectionSweepSolver (acpf_nr_gpu_injection).  Scenario 0 is always the
exact base case; scenarios 1..N are random log-normal perturbations.

Load buses (PQ): P and Q scaled by the same per-bus log-normal factor,
  preserving the P/Q ratio at each bus.
Generator buses (PV, net positive P): P scaled uniformly so that
  sum(gen_P_s) / sum(gen_P_ref) = sum(load_P_s) / sum(load_P_ref).
Slack bus: unchanged (absorbs residual imbalance).

Usage
-----
cd gpusim2grid/benchmarks/
python injection_sweep.py
python injection_sweep.py --grid_name case6515rte --n_scenarios 5000
python injection_sweep.py --strategy direct_iter0_only --nb_iter 6
"""

import argparse
import subprocess
import numpy as np

from gpusim2grid.injection_sweep import InjectionSweepSolver
from gpusim2grid.acpf_nr import compute_branch_flows_cpu   # CPU reference for base-case validation
from gpusim2grid.compilation_options import is_fp32
from _grid_setup import load_grid, extract_ac_data


_DEFAULT_SIGMA = np.log(1.05) / 1.96   # ≈ 0.0249 → 95% of multipliers in [0.95, 1.05]


def get_parser():
    parser = argparse.ArgumentParser(description="GPU injection sweep benchmark")
    parser.add_argument(
        "--grid_name", default="l2rpn_idf_2023",
        help="grid2op env name or pandapower network name (default: l2rpn_idf_2023)")
    parser.add_argument(
        "--n_scenarios", type=int, default=1000,
        help="number of random scenarios (+ 1 base case, default: 1000)")
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="scenarios per GPU chunk (default: 100)")
    parser.add_argument(
        "--nb_iter", type=int, default=4,
        help="fixed NR iterations per chunk (default: 4)")
    parser.add_argument(
        "--tol_conv", type=float, default=1e-3,
        help="post-hoc residual threshold for 'converged' (default: 1e-3)")
    parser.add_argument(
        "--strategy", default="direct_refactor_every",
        choices=["direct_refactor_every", "direct_base_case_factors",
                 "direct_iter0_only", "direct_refactor_every_n"],
        help="linear-solve strategy (default: direct_refactor_every)")
    parser.add_argument(
        "--refactor_period", type=int, default=1,
        help="refactor period N for direct_refactor_every_n (default: 1)")
    parser.add_argument(
        "--sigma", type=float, default=_DEFAULT_SIGMA,
        help=f"log-normal σ for per-bus load scaling "
             f"(default: {_DEFAULT_SIGMA:.4f} ≈ 95%% of multipliers in [0.95, 1.05])")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="PRNG seed for scenario generation (default: 0)")
    return parser


def build_injection_scenarios(acSbus_init, pq_index, pv_index, sn_mva,
                               n_random, sigma, rng):
    """Build (p_mw, q_mvar) scenario arrays of shape (n_random + 1, n_bus).

    Scenario 0 is the exact base case (row 0 = acSbus_init * sn_mva).
    Scenarios 1..n_random apply independent per-PQ-bus log-normal factors.

    Returns
    -------
    p_mw, q_mvar : ndarray, shape (n_random + 1, n_bus), float64
        Net active / reactive injection per bus in MW / MVAr.
    """
    n_bus  = acSbus_init.shape[0]
    n_total = n_random + 1

    p_net_base = acSbus_init.real * sn_mva   # MW,   base-case net P per bus
    q_net_base = acSbus_init.imag * sn_mva   # MVAr, base-case net Q per bus

    # Start all scenarios at the base case; scenario 0 stays untouched.
    p_mw   = np.tile(p_net_base, (n_total, 1))   # (n_total, n_bus)
    q_mvar = np.tile(q_net_base, (n_total, 1))

    # Independent log-normal factor per PQ bus per random scenario.
    eps     = rng.normal(0.0, sigma, size=(n_random, len(pq_index)))
    factors = np.exp(eps)                          # (n_random, n_pq)

    # Perturb PQ buses (rows 1..).  Q scales by the same factor → P/Q ratio preserved.
    p_mw  [1:, pq_index] = p_net_base[pq_index] * factors
    q_mvar[1:, pq_index] = q_net_base[pq_index] * factors

    # Scale PV buses with positive net P so that generation tracks load.
    # Buses where net P < 0 at PV type (load dominates the bus) are left at base.
    pv_gen_idx = pv_index[p_net_base[pv_index] > 0]
    total_pq_p_ref = p_net_base[pq_index].sum()      # < 0 for net load (typical)
    if len(pv_gen_idx) > 0 and total_pq_p_ref != 0.0:
        total_pq_p_s = p_mw[1:, pq_index].sum(axis=1)   # (n_random,)
        pv_scale     = total_pq_p_s / total_pq_p_ref      # > 1 when load ↑
        p_mw[1:, pv_gen_idx] = p_net_base[pv_gen_idx] * pv_scale[:, None]

    return p_mw, q_mvar


def main(args):
    grid_name = args.grid_name
    grid, v_init, v_res, n_sub, _, vn_kv = load_grid(grid_name)
    ac_Ybus, acSbus_init, pv_index, pq_index, slack_index, slack_weights, \
        sn_mva, max_iter, tol = extract_ac_data(grid)
    n_bus    = ac_Ybus.shape[0]
    tol_conv = float(args.tol_conv)

    lines   = grid.get_lines()
    trafos  = grid.get_trafos()
    n_lines    = len(lines)
    n_trafos   = len(trafos)
    n_branches = n_lines + n_trafos

    branch_from = np.concat((lines.get_bus_id_side_1(),  trafos.get_bus_id_side_1()))
    branch_to   = np.concat((lines.get_bus_id_side_2(),  trafos.get_bus_id_side_2()))
    yff = np.concat((lines.get_yac_eff_11().copy(), trafos.get_yac_eff_11()))
    yft = np.concat((lines.get_yac_eff_12().copy(), trafos.get_yac_eff_12()))
    ytf = np.concat((lines.get_yac_eff_21().copy(), trafos.get_yac_eff_21()))
    ytt = np.concat((lines.get_yac_eff_22().copy(), trafos.get_yac_eff_22()))

    print(f"Grid: {grid_name}  ({n_bus} buses, {n_lines} lines, {n_trafos} trafos)")

    # Reference branch flows from the lightsim2grid base-case solution.
    ref_or_amps, _ = compute_branch_flows_cpu(
        v_res[:n_bus], branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)

    # --- Scenario generation ---
    n_random = args.n_scenarios
    n_total  = n_random + 1      # scenario 0 = exact base case

    rng = np.random.default_rng(args.seed)
    p_mw, q_mvar = build_injection_scenarios(
        acSbus_init, pq_index, pv_index, sn_mva,
        n_random, args.sigma, rng)

    print(f"Generated {n_random} random scenarios "
          f"(sigma={args.sigma:.4f}, seed={args.seed}; "
          f"scenario 0 = exact base case, total = {n_total})")

    # --- GPU solver ---
    batch_size = args.batch_size
    nb_iter    = args.nb_iter
    print(f"\nRunning injection sweep on GPU "
          f"(strategy={args.strategy}, batch_size={batch_size}, "
          f"nb_iter={nb_iter}, n_scenarios={n_total}) ...")

    solver = InjectionSweepSolver(
        ac_Ybus, v_init, acSbus_init,
        slack_index, slack_weights, pv_index, pq_index,
        batch_size=batch_size, nb_iter=nb_iter,
        max_iter_base=max_iter, tol_base=tol)

    solver.strategy = args.strategy
    if args.strategy == 'direct_refactor_every_n':
        solver.refactor_period = args.refactor_period

    solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, vn_kv, sn_mva)
    solver.set_injections(p_mw, q_mvar, sn_mva)
    solver.run()
    print(f"Effective batch size used: {solver.used_batch_size}")

    # --- Branch flows (GPU kernel, all scenarios at once) ---
    solver.compute_flows()
    or_amps = solver.or_amps.to_numpy().reshape(n_total, n_branches)

    V_results = solver.V_results.to_numpy().reshape(n_total, n_bus)
    residuals  = solver.residuals.to_numpy()    # (n_total,)
    timings    = solver.timings

    # --- Base-case validation (scenario 0) ---
    diff_V = np.abs(V_results[0] - v_res[:n_bus]).max()
    diff_A = np.abs(or_amps[0] - ref_or_amps).max()
    has_error = False
    print(f"\nBase-case scenario (scenario 0) validation:")
    if diff_V > 1e-3:
        has_error = True
    print(f"  Max |V_gpu − V_ref|    : {diff_V:.2e} pu  "
          f"({'OK' if diff_V <= 1e-3 else 'FAIL'})")
    if diff_A > 1.0:
        has_error = True
    print(f"  Max |I_gpu − I_ref|    : {diff_A:.2e} A   "
          f"({'OK' if diff_A <= 1.0 else 'FAIL'})")
    if not has_error:
        print("  Base-case results match lightsim2grid reference")

    # --- Residual statistics ---
    # Guard: the GPU solver stores residual = 0 when the NR fully diverges to NaN.
    # Require both a finite residual below tol_conv AND finite voltages.
    finite_V   = np.isfinite(V_results).all(axis=1)   # (n_total,) bool
    conv_mask  = (residuals < tol_conv) & finite_V
    n_nan_V    = (~finite_V).sum()
    n_converged = conv_mask.sum()
    print(f"\nResults: {n_converged}/{n_total} scenarios converged "
          f"(residual < {tol_conv:.0e})")
    if n_nan_V:
        print(f"  WARNING: {n_nan_V} scenario(s) have non-finite voltages "
              f"(NR diverged; GPU residual is spuriously 0.0 for these)")
    print(f"Max residual : {np.nanmax(residuals[finite_V] if finite_V.any() else residuals):.3e}")
    print(f"Mean residual: {np.nanmean(residuals[finite_V] if finite_V.any() else residuals):.3e}")

    if conv_mask.any():
        max_or  = np.nanmax(or_amps[conv_mask])
        mean_or = np.nanmean(or_amps[conv_mask])
        print(f"Branch flows  : max={max_or:.1f} A, mean={mean_or:.1f} A "
              f"({conv_mask.sum()} converged scenarios)")
    print()

    # --- Timings ---
    print("Timings breakdown:")
    print(f"  Base case NR           : {timings.t_base_case_ms:.2f} ms")
    print(f"  CPU preprocessing      : {timings.t_preprocess_ms:.2f} ms")
    print(f"  Device allocation      : {timings.t_alloc_ms:.2f} ms")
    print(f"  cuDSS analysis         : {timings.t_analysis_ms:.2f} ms")
    print(f"  Source-specific init   : {timings.t_source_init_ms:.2f} ms")
    print(f"  --- per-chunk totals (all {timings.n_chunks} chunks) ---")
    print(f"  Tile V                 : {timings.t_tile_V.wall_ms:.2f} ms")
    print(f"  Tile Ybus              : {timings.t_tile_Ybus.wall_ms:.2f} ms")
    print(f"  Tile Sbus              : {timings.t_tile_Sbus.wall_ms:.2f} ms")
    print(f"  SpMV (Ibus = Ybus·V)   : {timings.t_spmv.wall_ms:.2f} ms")
    print(f"  Fill F                 : {timings.t_fill_F.wall_ms:.2f} ms")
    print(f"  Fill J                 : {timings.t_fill_J.wall_ms:.2f} ms")
    print(f"  Factorize (1st)        : {timings.t_first_factorize.wall_ms:.2f} ms")
    _ms_per_refacto = (timings.t_refactorize.wall_ms / timings.n_refactorize
                       if timings.n_refactorize > 0 else 0.)
    print(f"  Refactorize ({timings.n_refactorize}x)  : "
          f"{timings.t_refactorize.wall_ms:.2f} ms"
          f"  ({_ms_per_refacto:.3f} ms/call)")
    print(f"  Solve                  : {timings.t_solve.wall_ms:.2f} ms")
    print(f"  Update V               : {timings.t_update_V.wall_ms:.2f} ms")
    print(f"  Residual eval          : {timings.t_residual.wall_ms:.2f} ms")
    print(f"  Store V                : {timings.t_store_V.wall_ms:.2f} ms")
    print(f"  Branch flows           : {timings.t_flow_computation.wall_ms:.2f} ms")
    print(f"  Branch data upload     : {timings.t_branch_data_upload_ms:.2f} ms")
    print(f"  Copy flows to host     : {timings.t_copy_flows_to_host_ms:.2f} ms")
    print(f"  Copy V to host         : {timings.t_copy_V_to_host_ms:.2f} ms")
    print(f"  Copy residuals to host : {timings.t_copy_residuals_to_host_ms:.2f} ms")
    print(f"  Total chunk work       : {timings.t_chunks_total_wall_ms:.2f} ms")
    print(f"  Per scenario           : {timings.t_per_contingency_ms:.3f} ms/scen")
    print(f"  Throughput             : {1000./timings.t_per_contingency_ms:.0f} scen/s")
    print()
    print("Coarse aggregation:")
    print(f"  CPU preprocessing (total)  : {timings.t_cpu_preprocess_ms:.2f} ms")
    print(f"  Host->device (total)       : {timings.t_host_to_device_ms:.2f} ms")
    print(f"  GPU compute (total)        : {timings.t_gpu_compute_ms:.2f} ms")
    print(f"  Device->host (total)       : {timings.t_device_to_host_ms:.2f} ms")
    print(f"  Grand total                : {timings.t_grand_total_ms:.2f} ms")
    print()
    t_real_total_ms = (timings.t_preprocess_ms + timings.t_alloc_ms +
                       timings.t_analysis_ms + timings.t_chunks_total_wall_ms)
    t_real_per_scen_ms = t_real_total_ms / n_total
    print("--- Real end-to-end totals (preprocessing + alloc + analysis + chunks) ---")
    print("    (Excludes the base-case NR)")
    print(f"  Total real time        : {t_real_total_ms:.2f} ms")
    print(f"  Per scenario           : {t_real_per_scen_ms:.3f} ms/scen")
    print(f"  Throughput             : {1000. / t_real_per_scen_ms:.0f} scen/s")
    print()
    print(f"Full timings repr: {timings}")


def get_gpu_mdl():
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    line = line_as_bytes.decode("ascii")
    _, line = line.split(":", 1)
    line, _ = line.split("(", 1)
    return line.strip()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print("==================================")
    fp_used = "Float 32 bits (fp32)" if is_fp32 else "Float 64 bits (fp64)"
    print(f"Computation used (cuda side) {fp_used}\n")
    main(args)
    print("==================================")
    model = get_gpu_mdl()
    print(f"GPU used: {model}")
    print("==================================")
