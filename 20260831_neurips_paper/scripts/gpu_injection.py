# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import warnings

from tqdm import tqdm
import os
import numpy as np
from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames
import time

from gpusim2grid import InjectionSweepGPU, warmup

from kcl_checker import (
    KCLChecker,
    summarize_kcl_mismatch,
    summarize_kcl_mismatch_checkpoints,
    kcl_mismatch_per_scenario,
)
from scenario_utils import (
    all_file_names, 
    REF_PATH,
    get_injection_from_base,
    get_default_args,
    PATH_RESULTS,
    get_final_name,
    TOL_PF
)
from get_init_method import get_init_method

PF_ALGO_LS = "NR_KLU"
BASE_EXPE_NAME = "gpu_injection_{}_{}_{}_{}"

# Like ml_injection.py's --max_batches (see its module docstring): sweep.compute() times
# EVERY scenario handed to it in one shot, so a small --batch_size against a large --nb_pf
# would submit thousands of batches to the GPU just to report the same samples/s a 100-batch
# estimate already gives. Capped BEFORE set_injections_from_elements/compute() (not after),
# so the oversized array is never even uploaded to the GPU.
MAX_BATCHES = 100

# Further tightens --max_batches on top of the above for big grids -- a single batch is
# already much more expensive to solve/factorize (bigger Ybus and Jacobian) as the grid
# grows, so even 100 batches of a large grid can take far longer than 100 batches of a small
# one for the same throughput estimate. (bus_count_threshold, max_batches), checked largest
# threshold first so a grid past both thresholds gets the smaller (more restrictive) cap.
SIZE_MAX_BATCHES = [(50_000, 10), (10_000, 30)]

def get_args():
    parser = get_default_args()
    parser.add_argument('--batch_size', type=int, default=100,
                        help="Size of the GPU batch")
    parser.add_argument('--nb_iter', type=int, default=4,
                        help="How many iterations performed by the GPU")
    parser.add_argument('--device', type=int, default=0,
                        help="On which device (GPU) to perform the computations")
    parser.add_argument("--save_kcl",
                        action="store_true",
                        help="Save the kcl mismatch (power balance loss)")
    parser.add_argument("--max_batches", type=int, default=MAX_BATCHES,
                        help="cap on how many batches of --batch_size scenarios are actually "
                             "timed (<=0 disables this CLI cap, but SIZE_MAX_BATCHES's own "
                             "size-based cap still applies -- see --no_cap); see comment above "
                             "MAX_BATCHES")
    parser.add_argument("--no_cap", action="store_true",
                        help="Disable BOTH --max_batches and the size-based SIZE_MAX_BATCHES "
                             "cap: every one of --nb_pf scenarios is run for every grid "
                             "regardless of size, for full-sample (not throughput-sized-subset) "
                             "metrics. Much slower, and more likely to OOM, on the largest grids.")
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    batch_size = args.batch_size
    max_batches = args.max_batches
    nb_iter = args.nb_iter
    device = args.device
    save_flows = args.save_flows
    save_voltages = args.save_voltages
    save_kcl = args.save_kcl
    evaluate_kcl = args.evaluate_kcl
    nb_pf_total = int(args.nb_pf)
    tol_pf = TOL_PF
    
    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total, nb_iter, batch_size)
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"
    full_path_As = f"{nm_tmp}_{{}}_As.npy"
    full_path_kcl = f"{nm_tmp}_{{}}_kcl.npy"

    # Pay CUDA-context creation and the cuSPARSE/cuDSS dlopen + PTX-JIT once here, in a
    # controlled, size-independent call, rather than implicitly inside the first grid's own
    # warmup_sweep = InjectionSweepGPU(...) below (which also runs a real, grid-sized uniform-
    # batch session build) -- same pattern gpusim2grid's own benchmarks/injection_sweep.py
    # uses. See gpusim2grid.warmup()'s own docstring: it exercises this exact uniform-batch
    # cuDSS path (CUDSS_CONFIG_UBATCH_SIZE) on a trivial 2x2 system first.
    print(f"GPU warm-up (CUDA/cuSPARSE/cuDSS init): {warmup(device):.1f} ms")

    for fn in tqdm(all_file_names):
        tmp_res_dict = {}
        OOM = False
        path = os.path.join(REF_PATH, fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(path)
        beg_init = time.perf_counter()
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
        
        lsgrid.change_algorithm(PF_ALGO_LS)
        res = lsgrid.ac_pf(v_init, 10, tol_pf)
        end_init = time.perf_counter()
        
        load_p, load_q, gen_p = get_injection_from_base(
            fn,
            lsgrid,
            sample_data_meth,
            nb_pf_total=nb_pf_total,
            seed=seed)

        # See MAX_BATCHES/SIZE_MAX_BATCHES comments above get_args(): cap to the first
        # `case_max_batches` batches instead of running the full sweep when there would
        # otherwise be too many of them, tightened further for big grids.
        n_bus = lsgrid.total_bus()
        if args.no_cap:
            size_cap = None
            case_max_batches = None
        else:
            size_cap = next((cap for threshold, cap in SIZE_MAX_BATCHES if n_bus >= threshold), None)
            if max_batches <= 0:
                case_max_batches = size_cap  # disabled cap still yields to a size-based one
            elif size_cap is None:
                case_max_batches = max_batches
            else:
                case_max_batches = min(max_batches, size_cap)

        total_batches = -(-load_p.shape[0] // batch_size)  # ceil division
        capped = case_max_batches is not None and total_batches > case_max_batches
        if capped:
            n_capped = case_max_batches * batch_size
            print(f"  {fn}: capping {load_p.shape[0]} scenarios ({total_batches} batches of "
                  f"batch_size={batch_size}) -> first {n_capped} scenarios ({case_max_batches} "
                  f"batches) (--max_batches={max_batches}, n_bus={n_bus} -> size_cap={size_cap})")
            load_p = load_p[:n_capped]
            load_q = load_q[:n_capped]
            gen_p = gen_p[:n_capped]

        if fn == all_file_names[0]:
            # warmup gpu: load the so etc.
            # do it only once for the first case studied
            warmup_sweep = InjectionSweepGPU(
                lsgrid,
                use_bridge=True,
                init_from_n_powerflow=True,
                device=device)
            warmup_sweep.set_injections_from_elements(load_p, load_q, gen_p)
            try:
                warmup_sweep.compute(batch_size=batch_size)
            except Exception:
                OOM = True
        beg_pre_proc = time.perf_counter()
        sweep = InjectionSweepGPU(
            lsgrid,
            use_bridge=True,
            init_from_n_powerflow=True,
            tol_base=tol_pf,
            device=device,
            nb_iter=nb_iter)
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        end_preproc = time.perf_counter()
        
        beg_sweep = time.perf_counter()
        try:
            sweep.compute(batch_size=batch_size)
        except Exception:
            OOM = True
        end_sweep = time.perf_counter()
        
        tmp_res_dict["init_cpu_time"] = end_init - beg_init
        if not OOM:
            tmp_res_dict["pre_process_cpu_time"] = end_preproc - beg_pre_proc
            tmp_res_dict["total_time"] = end_sweep - beg_sweep
            tmp_res_dict["detailed_timings"] = sweep.timings.to_dict()
        tmp_res_dict["nb_powerflow"] = int(load_p.shape[0])
        tmp_res_dict["n_bus"] = int(n_bus)
        tmp_res_dict["total_batches_available"] = total_batches
        tmp_res_dict["max_batches"] = max_batches
        tmp_res_dict["size_cap_max_batches"] = size_cap
        tmp_res_dict["max_batches_effective"] = case_max_batches
        tmp_res_dict["capped"] = capped

        # sweep.V_results.to_numpy() (like or_amps below) comes back flat, NOT already
        # shaped (n_scenario, n_bus) -- see benchmarks/injection_sweep.py's own
        # .reshape(n_total, n_bus) after the identical .to_numpy() call. Reshape once here
        # and reuse for both the KCL check and --save_voltages.
        V_results = None
        if not OOM and (evaluate_kcl or save_voltages):
            V_results = sweep.V_results.to_numpy().reshape(sweep.n_scenarios, sweep.n_bus)

        if evaluate_kcl and not OOM:
            # InjectionSweepGPU exposes no per-scenario convergence signal (unlike
            # InjectionSweepCPP's converged_mask() / PGM's zero-filled node rows) -- a
            # whole-batch failure is already excluded above (OOM), but at a low --nb_iter a
            # single scenario can still individually diverge to a NaN/Inf voltage without
            # tripping OOM at all (confirmed empirically: at --nb_iter 5, WHICH grid hits
            # this is GPU-architecture/driver dependent -- a run on one machine came back
            # clean while another NaN'd, and vice versa for a different grid -- consistent
            # with a handful of scenarios sitting right on the edge of NR divergence, tipped
            # one way or the other by floating-point non-associativity across cuDSS/cuSPARSE
            # builds). A non-finite voltage is therefore excluded from "solved" here too, the
            # same guard gpu_pfdelta.py's ScenarioSweepGPU path applies via get_disconnected()
            # -- otherwise a single such row poisons the whole grid's mean/max/L2 with NaN.
            solved_mask = np.isfinite(V_results).all(axis=1)
            kcl_checker = KCLChecker(path)
            kcl_mismatch = kcl_checker.check_kcl_injection(
                load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask], V_results[solved_mask])
            tmp_res_dict["nb_solved"] = int(solved_mask.sum())
            tmp_res_dict.update(summarize_kcl_mismatch(kcl_mismatch))
            tmp_res_dict["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
            tmp_res_dict.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))

        benchmark_results[fn] = tmp_res_dict
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)

        case_nm, _ = os.path.splitext(fn)
        if save_kcl:
            np.save(
                file=full_path_kcl.format(case_nm),
                arr=sweep.last_residuals()
            )
        if save_voltages and not OOM:
            np.save(
                file=full_path_Vs.format(case_nm),
                arr=V_results
            )
        if save_flows and not OOM:
            sweep.compute_flows()
            ampss = sweep.or_amps.to_numpy().reshape(sweep.n_scenarios, sweep.n_branches)
            np.save(
                file=full_path_As.format(case_nm),
                arr=ampss
            )
