import argparse
import subprocess
import numpy as np
from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
from gpusim2grid import ContingencyAnalysisGPU
from gpusim2grid.compilation_options import is_fp32
from _grid_setup import load_grid


def get_parser():
    parser = argparse.ArgumentParser(description="GPU contingency analysis benchmark")
    parser.add_argument("--grid_name", default="l2rpn_idf_2023",
                        help="grid2op env name or pandapower network name (default: l2rpn_idf_2023)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="contingencies per GPU chunk (default: 10)")
    parser.add_argument("--nb_iter", type=int, default=4,
                        help="fixed NR iterations per chunk (default: 4)")
    parser.add_argument("--tol_conv", type=float, default=1e-3,
                        help="Tolerance to detect, post processing, if something has converged. "
                        "NOT THE SOLVER TOLERANCE (default: 1e-3)")
    parser.add_argument("--nb_max_cont_cpu", type=int, default=1000,
                        help="maximum number of contingencies simulated on the CPU (default: 1000)")
    parser.add_argument("--nb_max_cont_gpu", type=int, default=10_000,
                        help="maximum number of contingencies simulated on the GPU (default: 10000)")
    parser.add_argument("--strategy", default="direct_refactor_every",
                        choices=["direct_refactor_every", "direct_base_case_factors",
                                 "direct_iter0_only", "direct_refactor_every_n"],
                        help="linear-solve strategy (default: direct_refactor_every)")
    parser.add_argument("--refactor_period", type=int, default=1,
                        help="refactor period N for direct_refactor_every_n (default: 1)")
    return parser


def main(args):
    grid_name = args.grid_name
    grid, v_init, v_res, n_sub, v_init_ca, vn_kv = load_grid(grid_name)
    tol_conv = float(args.tol_conv)
    nb_max_cont_cpu = int(args.nb_max_cont_cpu)
    nb_max_cont_gpu = int(args.nb_max_cont_gpu)
    if nb_max_cont_cpu > nb_max_cont_gpu:
        raise ValueError(f"nb_max_cont_cpu(={nb_max_cont_cpu}) should be <= nb_max_cont_gpu(={nb_max_cont_gpu}) ")

    max_cont = max(nb_max_cont_cpu, nb_max_cont_gpu)

    lines = grid.get_lines()
    trafos = grid.get_trafos()
    n_lines = len(lines)
    n_trafos = len(trafos)
    n_branches = n_lines + n_trafos
    
    if n_branches > nb_max_cont_cpu:
        prng = np.random.default_rng(0)
        cont_ids_both = prng.choice(list(range(n_branches)), size=nb_max_cont_cpu, replace=False)
        cont_ids_both = sorted(cont_ids_both)
        if ac_Ybus.shape[0] > nb_max_cont_gpu - nb_max_cont_cpu:
            cont_ids_gpu_only = prng.choice(list(range(n_branches)), size=(nb_max_cont_gpu - nb_max_cont_cpu), replace=False)
            max_ = max_cont
        else:
            cont_ids_gpu_only = np.asarray([el for el in list(range(n_branches)) if not el in cont_ids_both], dtype=int)
            max_ = n_branches
    else:
        cont_ids_both = None
        max_ = None

    ca_ref = ContingencyAnalysisCPP(grid)
    if max_ is not None:
        ca_ref.add_multiple_n1(cont_ids_both)
    else:
        ca_ref.add_all_n1()
    ca_ref.init_from_n_powerflow = True
    ca_ref.compute(v_init_ca, 10, 1e-7)
    V_ref = ca_ref.get_voltages()
    ca_ref.compute_flows()
    ref_amps = 1e3 * ca_ref.get_flows()
    computer = ca_ref
    print(f"For environment: {grid_name} ({computer.nb_solved()} n-1 simulated)")
    print(f"Total time spent in \"computer\" to solve everything: {1e3*computer.total_time():.1f}ms "
        f"({computer.nb_solved() / computer.total_time():.0f} pf / s), "
        f"{1000.*computer.total_time() / computer.nb_solved():.3f} ms / pf)")
    print(f"\t - time to compute the coefficients to simulate line disconnection: {1e3*computer.preprocessing_time():.2f}ms")
    print(f"\t - time to pre process Ybus: {1e3*computer.modif_Ybus_time():.2f}ms")
    print(f"\t - time to perform powerflows: {1e3*computer.solver_time():.2f}ms "
        f"({computer.nb_solved() / computer.solver_time():.0f} pf / s, "
        f"{1000.*computer.solver_time() / computer.nb_solved():.2f} ms / pf)")
    print(f"In addition, it took {1e3*computer.amps_computation_time():.2f} ms to retrieve the current "
        f"from the complex voltages (in total "
        f"{computer.nb_solved() / ( computer.total_time() + computer.amps_computation_time()):.1f} "
        "pf /s, "
        f"{1000.*( computer.total_time() + computer.amps_computation_time()) / computer.nb_solved():.3f} ms / pf)")

    print(f"Grid: {n_sub} buses, {n_lines} lines, {n_trafos} trafos → {n_branches} N-1 possible contingencies")

    # Each contingency disconnects exactly one branch (N-1).
    # cont_branch_ids[c] = [branch_id] for contingency c.
    cont_branch_ids = [[l] for l in range(n_lines)] + [[n_lines + t] for t in range(n_trafos)]
    if max_ is not None:
        print(f"Simulating {max_} / {n_branches} random contingencies")
        n_conts_id_simul = [cont_branch_ids[i] for i in cont_ids_both]
        n_conts_id_simul += [cont_branch_ids[i] for i in cont_ids_gpu_only]
        n_conts_simul = max_
    else:
        n_conts_id_simul = cont_branch_ids
        n_conts_simul = n_branches
        nb_max_cont_cpu = n_branches

    # --- Build and run GPU solver ---
    # batch_size and nb_iter can also be changed after construction via
    # ca.batch_size = N or ca.nb_iter = N before calling compute().
    batch_size = args.batch_size
    nb_iter    = args.nb_iter

    print(f"\nRunning contingency analysis on GPU (strategy={args.strategy}, batch_size={batch_size}, nb_iter={nb_iter}, n cont = {n_conts_simul}) ...")
    ca = ContingencyAnalysisGPU(
        grid, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6)
    # branch data (needed for compute_flows) is extracted automatically from
    # `grid` -- call ca.set_branch_data(...) only in explicit-array mode.

    ca.strategy = args.strategy
    if args.strategy == 'direct_refactor_every_n':
        ca.solver.refactor_period = args.refactor_period
    ca.add_contingencies_by_branch_id(n_conts_id_simul)

    ca.compute(batch_size=batch_size)
    print(f"Effective batch size used: {ca.solver.used_batch_size}")

    V_results = ca.V_results.to_numpy().reshape(n_conts_simul, n_sub)[:nb_max_cont_cpu, :]
    residuals = ca.last_residuals()

    # --- Flow results ---
    ca.compute_flows()
    or_amps = ca.or_amps.to_numpy().reshape(n_conts_simul, n_branches)[:nb_max_cont_cpu, :]
    # ex_amps = ca.ex_amps.to_numpy().reshape(n_conts_simul, n_branches)
    timings = ca.timings   # full timings after compute_flows()
    
    # --- Validate voltages against reference ---
    has_error = False
    for c_id in range(nb_max_cont_cpu):
        diff_ = np.abs(V_results[c_id] - V_ref[c_id][:ca.n_bus]).max()
        l_or_t = "line" if (cont_ids_both[c_id] if cont_ids_both is not None else c_id) < n_lines else "trafo"
        has_converged = np.abs(residuals[c_id]).max() < tol_conv
        has_conv_ref = np.abs(V_ref[c_id]).max() > tol_conv
        reported_id = cont_ids_both[c_id] if cont_ids_both is not None else c_id
        if has_converged and not has_conv_ref:
            print(f"Difference for contingency {reported_id} ({l_or_t}): converged here but not in ref")
            has_error = True
            continue
        if not has_converged and has_conv_ref:
            print(f"Difference for contingency {reported_id} ({l_or_t}): converged in ref but not here")
            has_error = True
            # continue
        if not has_conv_ref:
            continue
        if diff_ > 1e-5:
            print(f"Voltage difference for contingency {reported_id} ({l_or_t}): {diff_:.4e} pu")
            has_error = True
        
        diff_aor = or_amps[c_id] - ref_amps[c_id]
        max_diff_a =  np.abs(diff_aor).max()
        if max_diff_a > 1e-3:
            has_error = True
            print(f"Flow (A) max difference for contingency {reported_id} ({l_or_t}): {max_diff_a:.4e} A")
            
    if not has_error:
        print(f"All contingencies results match between l2g and g2g")

    # validate flows
    converged_mask = residuals < tol_conv
    if converged_mask.any():

        # Verify: tripped branch in each contingency should have zero flow
        n_check = min(5, n_conts_simul)
        zero_ok = True
        for c_id in range(n_check):
            if not converged_mask[c_id]:
                continue
            for l in n_conts_id_simul[c_id]:
                if or_amps[c_id, l] != 0.0: # or ex_amps[c_id, l] != 0.0:
                    print(f"  WARNING: contingency {c_id}, branch {l} should be 0 "
                          f"but or={or_amps[c_id,l]:.2f}")#, ex={ex_amps[c_id,l]:.2f}")
                    zero_ok = False
        if zero_ok:
            print(f"  Tripped-branch flows are correctly zeroed (checked {n_check} contingencies)")

    # --- Report ---
    n_converged = (residuals < 1e-4).sum()
    print(f"\nResults: {n_converged}/{n_conts_simul} contingencies converged (residual < 1e-4)")
    print(f"Max residual : {np.nanmax(residuals):.3e}")
    print(f"Mean residual: {np.nanmean(residuals):.3e}")
    print()

    # --- Timings ---
    print(f"Timings breakdown:")
    print(f"  Base case NR           : {timings.t_base_case_ms:.2f} ms")
    print(f"  CPU preprocessing      : {timings.t_preprocess_ms:.2f} ms")
    print(f"  Device allocation      : {timings.t_alloc_ms:.2f} ms")
    print(f"  cuDSS analysis         : {timings.t_analysis_ms:.2f} ms")
    print(f"  Source-specific init   : {timings.t_source_init_ms:.2f} ms")
    print(f"  --- per-chunk totals (all {timings.n_chunks} chunks) ---")
    print(f"  Tile V                 : {timings.t_tile_V.wall_ms:.2f} ms")
    print(f"  Tile Ybus              : {timings.t_tile_Ybus.wall_ms:.2f} ms")
    print(f"  Patch Ybus             : {timings.t_patch_Ybus.wall_ms:.2f} ms")
    print(f"  SpMV (Ibus = Ybus·V)   : {timings.t_spmv.wall_ms:.2f} ms")
    print(f"  Fill F                 : {timings.t_fill_F.wall_ms:.2f} ms")
    print(f"  Fill J                 : {timings.t_fill_J.wall_ms:.2f} ms")
    print(f"  Factorize (1st)        : {timings.t_first_factorize.wall_ms:.2f} ms")
    _ms_per_refacto = (timings.t_refactorize.wall_ms / timings.n_refactorize
                       if timings.n_refactorize > 0 else 0.)
    print(f"  Refactorize ({timings.n_refactorize}x)  : {timings.t_refactorize.wall_ms:.2f} ms"
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
    print(f"  Per contingency        : {timings.t_per_contingency_ms:.3f} ms/ctg")
    print(f"  Throughput             : {1000./timings.t_per_contingency_ms:.0f} ctg/s")
    print()
    print(f"Coarse aggregation:")
    print(f"  CPU preprocessing (total)  : {timings.t_cpu_preprocess_ms:.2f} ms")
    print(f"  Host->device (total)       : {timings.t_host_to_device_ms:.2f} ms")
    print(f"  GPU compute (total)        : {timings.t_gpu_compute_ms:.2f} ms")
    print(f"  Device->host (total)       : {timings.t_device_to_host_ms:.2f} ms")
    print(f"  Grand total                : {timings.t_grand_total_ms:.2f} ms")
    print()
    t_real_total_ms = (timings.t_preprocess_ms + timings.t_alloc_ms +
                       timings.t_analysis_ms + timings.t_chunks_total_wall_ms)
    t_real_per_ctg_ms = t_real_total_ms / n_conts_simul
    print(f"--- Real end-to-end totals (preprocessing + alloc + analysis + chunks) ---")
    print( "    (Excludes the computation of the NR in the base case)  ")
    print(f"  Total real time        : {t_real_total_ms:.2f} ms")
    print(f"  Per contingency        : {t_real_per_ctg_ms:.3f} ms/ctg")
    print(f"  Throughput             : {1000. / t_real_per_ctg_ms:.0f} ctg/s")
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
    if is_fp32:
        fp_used = "Float 32 bits (fp32)"
    else:
        fp_used = "Float 64 bits (fp64)"
    print(f"Computation used (cuda side) {fp_used}\n")
    main(args)
    print("==================================")
    model = get_gpu_mdl()
    print(f"GPU used: {model}")
    print("==================================")
