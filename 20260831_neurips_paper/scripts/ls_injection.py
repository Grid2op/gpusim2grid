# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.



import time
import warnings

import json
import os
from tqdm import tqdm

from lightsim2grid.network import init_from_matpower
from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
import numpy as np
from matpowercaseframes import CaseFrames

from get_init_method import (
    get_init_method
)

from kcl_checker import (
    KCLChecker,
    summarize_kcl_mismatch,
    summarize_kcl_mismatch_checkpoints,
    kcl_mismatch_per_scenario,
)

from scenario_utils import (
    REF_PATH,
    VERBOSE_TIMINGS,
    PATH_RESULTS,
    TOL_PF,
    get_args,
    get_final_name,
    get_injection_from_base,
)

verbose_lightsim2grid_timing = True
BASE_EXPE_NAME = "ls_injection_{}_{}"
# lightsim2grid's own C++ solver (KLU-backed Newton-Raphson), used directly here (not through
# pandapower) -- this is also the reference grid timing for the whole benchmark suite: every
# other script (olf_injection.py, pp_injection.py, etc.) reuses this same lsgrid + v_init to build
# its own scenarios (lightsim2grid keeps the same order as the matpower grid), 
# so this file's numbers isolate lightsim2grid's own cost with none of
# pandapower's/pypowsybl's extra Python-side overhead.
PF_ALGO = "NR_KLU"


if __name__ == "__main__":
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_flows = args.save_flows
    save_voltages = args.save_voltages
    evaluate_kcl = args.evaluate_kcl
    nb_thread = int(args.nb_threads)
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]

    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total)
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"
    full_path_As = f"{nm_tmp}_{{}}_As.npy"
    for fn in tqdm(grids):
        tmp_res_dict = {}
        path = os.path.join(REF_PATH, fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(path)
        init_method = get_init_method(path)
        if init_method == "flat":
            # use the flat init method
            # preferred in all cases
            v_init = np.ones(lsgrid.total_bus(), dtype=complex)
        elif init_method == "dc":
            # use the dc init method
            # only if "flat" fails
            v_init_dc = np.ones(lsgrid.total_bus(), dtype=complex)
            v_init = lsgrid.dc_pf(v_init_dc, 10, 1e-6)
        elif init_method == "file":
            # init the complex voltages from the
            # data in the matpower casefile
            # only if "flat" and "dc" fails
            cf = CaseFrames(path)
            v_init = cf.bus["VM"] * np.exp(1j * np.deg2rad(cf.bus["VA"]))


        lsgrid.change_algorithm(PF_ALGO)
        # two timed powerflows are run back to back, from the exact same v_init (mirrors
        # pp_injection.py's own "warm-up, then two timed powerflows" split): the first one
        # ("init_powerflow" below) pays for lazy one-time setup inside lightsim2grid (linear
        # solver initialization, symbolic factorization, ...), the second ("following_powerflow")
        # is the steady-state repeated-solve cost -- comparing the two below via
        # nb_factor/nb_refactor is what confirms the first factorizes while the second only
        # refactorizes.
        # this comparison is NOT used for this paper.
        beg_ls = time.perf_counter()
        res_1 = lsgrid.ac_pf(v_init, 10, TOL_PF)
        end_ls = time.perf_counter()
        lsgrid.unset_changes()
        # lightsim2grid's ac_pf() returns an empty array (not an exception) on non-convergence
        if res_1.shape[0] == 0:
            print(f"Error for {fn}: base powerflow did not converge, skipping")
            benchmark_results[fn] = {"error": "base powerflow did not converge"}
            with open(complete_full_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, fp=f, indent=2)
            continue
        if VERBOSE_TIMINGS:
            print(res_1.shape[0])
            print(f"Grid size: {lsgrid.total_bus()}, {len(lsgrid.get_lines()) + len(lsgrid.get_trafos())}, Ybus: {lsgrid.get_Ybus_solver().nnz}")
            print(f"Jac size: {lsgrid.get_solver().get_J().shape}, nnz: {lsgrid.get_solver().get_J().nnz}")
        # recorded once per grid so results across backends (see grid_size in
        # pp_injection.py) can be sanity-checked against each other and against the raw
        # matpower file
        tmp_res_dict["grid_size"] = {
            "total_bus": int(lsgrid.total_bus()),
            "total_branch": len(lsgrid.get_lines()) + len(lsgrid.get_trafos()),
            "Ybus_nnz": int(lsgrid.get_Ybus_solver().nnz)
            }
        tmp_res_dict["jac_size"] = {
            "dim": int(lsgrid.get_solver().get_J().shape[0]),
            "nnz": int(lsgrid.get_solver().get_J().nnz)
        }
        if VERBOSE_TIMINGS:
            print(f"Computation time: {lsgrid.timer_last_ac_pf:.2e}s")
        timers_jac = lsgrid.get_solver().get_timers_jacobian()
        tot_time = end_ls - beg_ls
        stats = lsgrid.get_solver().get_linear_solver_stats()
        nb_factor_first = stats.nb_factorize
        nb_refactor_first = stats.nb_refactorize
        timer_total_nr_ = timers_jac.timer_total_nr
        if VERBOSE_TIMINGS and verbose_lightsim2grid_timing:
            print("--------------------------------------")
            print("Detailed lightsim2grid timings: ")
            print(f"Total time spent in the solver: {1e3 * timers_jac.timer_total_nr:.2e} ms ({100. * timers_jac.timer_total_nr / tot_time:.0f} % of total)")
            print(f"\t Time to pre process Ybus, Sbus etc.: {1e3 * timers_jac.timer_pre_proc:.2e} ms ({100. * timers_jac.timer_pre_proc / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to initialize linear solver {1e3 * timers_jac.timer_initialize:.2e} ms ({100. * timers_jac.timer_initialize / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to compute dS/dV {1e3 * timers_jac.timer_dSbus : .2e} ms ({100. * timers_jac.timer_dSbus / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to fill the Jacobian {1e3 * timers_jac.timer_fillJ:.2e} ms ({100. * timers_jac.timer_fillJ / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to factor the Jacobian linear system: {1e3 * timers_jac.timer_factor:.2e} ms ({100. * timers_jac.timer_factor / timer_total_nr_:.0f} % of time in solver)")
            if nb_factor_first:
                print(f"\t\t {1e3 * timers_jac.timer_factor / nb_factor_first:.2e} ms per factor")
            print(f"\t Time to refactor the Jacobian linear system: {1e3 * timers_jac.timer_refactor:.2e} ms ({100. * timers_jac.timer_refactor / timer_total_nr_:.0f} % of time in solver)")
            if nb_refactor_first:
                print(f"\t\t {1e3 * timers_jac.timer_refactor / nb_refactor_first:.2e} ms per refactor")
            print(f"\t Time to solve the Jacobian linear system: {1e3 * timers_jac.timer_solve:.2e} ms ({100. * timers_jac.timer_solve / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to update Va and Vm {1e3*timers_jac.timer_Va_Vm:.2e} ms ({100. * timers_jac.timer_Va_Vm / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to build p,q mismmatch at each bus {1e3 * timers_jac.timer_mismatch:.2e} ms ({100. * timers_jac.timer_mismatch / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to evaluate p,q mismmatch at each bus {1e3 * timers_jac.timer_Fx:.2e} ms ({100. * timers_jac.timer_Fx / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to evaluate cvg criteria {1e3 * timers_jac.timer_check:.2e} ms ({100. * timers_jac.timer_check / timer_total_nr_:.0f} % of time in solver)")
            print("--------------------------------------\n")
        tmp_res_dict["init_powerflow"] = {
            "timer_last_ac_pf": lsgrid.timer_last_ac_pf,
            "timer_total_nr": timers_jac.timer_total_nr,
            "nb_factor": nb_factor_first,
            "nb_refactor_first": nb_refactor_first,
            "timer_pre_proc": timers_jac.timer_pre_proc,
            "init": timers_jac.timer_initialize,
            "dSbus": timers_jac.timer_dSbus,
            "fillJ": timers_jac.timer_fillJ,
            "factor": timers_jac.timer_factor,
            "refactor": timers_jac.timer_refactor,
            "solve": timers_jac.timer_solve,
            "Va_Vm": timers_jac.timer_Va_Vm,
            "mismatch": timers_jac.timer_mismatch,
            "Fx": timers_jac.timer_Fx,
            "check": timers_jac.timer_check
        }
        if VERBOSE_TIMINGS:
            print("Second powerflow")
        # same v_init, same tolerance, same (now-warmed-up) lsgrid -- this is the
        # "following_powerflow" half of the split described above
        beg_ls = time.perf_counter()
        res_2 = lsgrid.ac_pf(v_init, 10, TOL_PF)
        end_ls = time.perf_counter()
        if res_2.shape[0] == 0:
            raise RuntimeError("Should not happen: if a powerf converges once, it should converge twice")
        lsgrid.unset_changes()
        timers_jac = lsgrid.get_solver().get_timers_jacobian()
        tot_time = end_ls - beg_ls
        stats = lsgrid.get_solver().get_linear_solver_stats()
        nb_factor_second = stats.nb_factorize - nb_factor_first
        nb_refactor_second = stats.nb_refactorize - nb_refactor_first
        if VERBOSE_TIMINGS:
            print(f"Computation time: {lsgrid.timer_last_ac_pf:.2e}s")
        if VERBOSE_TIMINGS and verbose_lightsim2grid_timing:
            print("--------------------------------------")
            print("Detailed lightsim2grid timings: ")
            timer_total_nr_ = timers_jac.timer_total_nr
            print(f"Total time spent in the solver: {1e3 * timers_jac.timer_total_nr:.2e} ms ({100. * timers_jac.timer_total_nr / tot_time:.0f} % of total)")
            print(f"\t Time to pre process Ybus, Sbus etc.: {1e3 * timers_jac.timer_pre_proc:.2e} ms ({100. * timers_jac.timer_pre_proc / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to initialize linear solver {1e3 * timers_jac.timer_initialize:.2e} ms ({100. * timers_jac.timer_initialize / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to compute dS/dV {1e3 * timers_jac.timer_dSbus : .2e} ms ({100. * timers_jac.timer_dSbus / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to fill the Jacobian {1e3 * timers_jac.timer_fillJ:.2e} ms ({100. * timers_jac.timer_fillJ / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to factor the Jacobian linear system: {1e3 * timers_jac.timer_factor:.2e} ms ({100. * timers_jac.timer_factor / timer_total_nr_:.0f} % of time in solver)")
            if nb_factor_second:
                print(f"\t\t {1e3 * timers_jac.timer_factor / nb_factor_second:.2e} ms per factor")
            print(f"\t Time to refactor the Jacobian linear system: {1e3 * timers_jac.timer_refactor:.2e} ms ({100. * timers_jac.timer_refactor / timer_total_nr_:.0f} % of time in solver)")
            if nb_refactor_second:
                print(f"\t\t {1e3 * timers_jac.timer_refactor / nb_refactor_second:.2e} ms per refactor")
            print(f"\t Time to solve the Jacobian linear system: {1e3 * timers_jac.timer_solve:.2e} ms ({100. * timers_jac.timer_solve / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to update Va and Vm {1e3*timers_jac.timer_Va_Vm:.2e} ms ({100. * timers_jac.timer_Va_Vm / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to build p,q mismmatch at each bus {1e3 * timers_jac.timer_mismatch:.2e} ms ({100. * timers_jac.timer_mismatch / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to evaluate p,q mismmatch at each bus {1e3 * timers_jac.timer_Fx:.2e} ms ({100. * timers_jac.timer_Fx / timer_total_nr_:.0f} % of time in solver)")
            print(f"\t Time to evaluate cvg criteria {1e3 * timers_jac.timer_check:.2e} ms ({100. * timers_jac.timer_check / timer_total_nr_:.0f} % of time in solver)")
            print("--------------------------------------\n")
            
        tmp_res_dict["following_powerflow"] = {
            "timer_last_ac_pf": lsgrid.timer_last_ac_pf,
            "timer_total_nr": timers_jac.timer_total_nr,
            "nb_factor": nb_factor_second,
            "nb_refactor_first": nb_refactor_second,
            "timer_pre_proc": timers_jac.timer_pre_proc,
            "init": timers_jac.timer_initialize,
            "dSbus": timers_jac.timer_dSbus,
            "fillJ": timers_jac.timer_fillJ,
            "factor": timers_jac.timer_factor,
            "refactor": timers_jac.timer_refactor,
            "solve": timers_jac.timer_solve,
            "Va_Vm": timers_jac.timer_Va_Vm,
            "mismatch": timers_jac.timer_mismatch,
            "Fx": timers_jac.timer_Fx,
            "check": timers_jac.timer_check
        }  
        if VERBOSE_TIMINGS:
            print("Beginning time series")

        gen_v = None
        # exact same sampling call (same seed, same sample_data_meth, same base lsgrid values)
        # as olf_injection.py / pp_injection.py: the input scenarios are bit-for-bit identical
        # across every backend in this benchmark
        load_p, load_q, gen_p = get_injection_from_base(
            fn,
            lsgrid,
            sample_meth=sample_data_meth,
            seed=seed,
            nb_pf_total=nb_pf_total)

        # matpower/lightsim2grid never split "static generator" (sgen) out of gen the way
        # pandapower does (see pp_injection.py's get_gen_alignment docstring) -- InjectionSweepCPP
        # still takes a separate sgen_p argument for API parity with lightsim2grid's other
        # (pandapower-facing) call sites, so it is passed as an empty (nb_scen, 0) array here
        sgen_p = np.zeros((load_p.shape[0], 0))
        # InjectionSweepCPP is lightsim2grid's own C++ batch solver: it is pure C++ and
        # releases the GIL for the actual solve, so nb_thread below spins up real OS threads
        # (unlike pp_injection.py's run_loop_mode_mp, which has to fall back to multiprocessing
        # because pandapower's own runpp() is Python/numba-JIT and stays GIL-bound)
        ts = InjectionSweepCPP(lsgrid)
        ts.change_algorithm(PF_ALGO)
        # reuse the base case's already-converged v_init as the starting point for every
        # scenario instead of re-running NR from a flat/dc start each time
        ts.init_from_n_powerflow = True
        ts.nb_thread = nb_thread
        if gen_v is not None:
            # eg pf delta_n
            ts.modify_gen_v(gen_v)
        res = ts.compute_Vs(
            gen_p,
            sgen_p,
            load_p,
            load_q,
            v_init,
            100,
            TOL_PF
        )
        if ts.nb_solved() != nb_pf_total:
            print(f"{fn}: {ts.nb_solved()} vs {nb_pf_total}")
            
        if VERBOSE_TIMINGS:
            res_time = 1.
            res_unit = "s"
            if load_p.shape[1] <= 1000:
                # report results in ms if there are less than 1000 loads
                # only affects "verbose" printing
                res_time = 1e3
                res_unit = "ms"
            print(f"Total time spent in \"computer\" to solve everything: {res_time*ts.total_time():.2f}{res_unit} "
                f"({ts.nb_solved() / ts.total_time():.0f} pf / s), "
                f"{1000.*ts.total_time() / ts.nb_solved():.2f} ms / pf)")
            print(f"\t - time to pre process the injections: {res_time * ts.preprocessing_time():.2f}{res_unit}")
            print(f"\t - time to perform powerflows: {res_time * ts.solver_time():.2f} {res_unit} "
                f"({ts.nb_solved() / ts.solver_time():.0f} pf / s, "
                f"{1000.*ts.solver_time() / ts.nb_solved():.2f} ms / pf)")
        tmp_res_dict["time_series"] = {
            "total_time":ts.total_time(),
            "nb_solved": ts.nb_solved(),
            "pre_proc": ts.preprocessing_time(),
            "solver_time": ts.solver_time(),
        }

        # InjectionSweep's rows are independent (BatchInitKind::FromSeed): a row that
        # fails to converge does NOT stop the batch, so failures can be scattered
        # rather than a trailing run -- ts.nb_solved() is not a reliable signal here
        # (it counts every row that reached the solver, converged or not, so for
        # InjectionSweep it is essentially always == nb_pf_total). converged_mask()
        # gives the real per-row flag directly, no need to fetch get_voltages() just
        # to infer it.
        solved_mask = np.array(ts.converged_mask())
        if solved_mask.sum() < nb_pf_total:
            tmp_res_dict["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()

        all_voltages = None
        if save_voltages or evaluate_kcl:
            # unlike olf_injection.py/pp_injection.py, no bus reprojection is needed here:
            # ts.get_voltages() comes straight out of lightsim2grid, already in the same
            # (matpower-derived) bus order as load_p/load_q/gen_p, which is the order
            # KCLChecker itself expects
            all_voltages = ts.get_voltages()

            if evaluate_kcl:
                kcl_checker = KCLChecker(path)
                kcl_mismatch = kcl_checker.check_kcl_injection(
                    load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask], all_voltages[solved_mask])
                tmp_res_dict.update(summarize_kcl_mismatch(kcl_mismatch))
                tmp_res_dict["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
                tmp_res_dict.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))

        benchmark_results[fn] = tmp_res_dict
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)
        case_nm, _ = os.path.splitext(fn)
        if save_voltages:
            np.save(file=full_path_Vs.format(case_nm), arr=all_voltages)
        if save_flows:
            # branch currents (amps) computed lazily from the already-solved voltages,
            # only when actually needed -- not fused into compute_Vs() above
            ampss = ts.compute_flows()
            np.save(full_path_As.format(case_nm),
                     arr=ampss
                     )
