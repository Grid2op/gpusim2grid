# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import json
import time
import warnings

# to make sure the data are always generated the same way
from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames

import pypowsybl.network as pypn
import pypowsybl.loadflow as pypl

import re
import multiprocessing as mp
from tqdm import tqdm
import os
import numpy as np


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
    TOL_PF,
)

from get_init_method import get_init_method

BASE_EXPE_NAME = "olf_injection_{}_{}_{}"  # sample_data_meth, nb_pf_total, use_init_v_values
BASE_EXPE_NAME_NO_NBPF = "olf_injection_{}_{}"  # sample_data_meth, use_init_v_values -- used
    # instead of BASE_EXPE_NAME whenever --add_to_name is set (eg per-grid sizing sweeps, see
    # olf_grid_sizing.py/base_launch_olf.sh): nb_pf_total varies per (grid, nb_threads) cell
    # there, so baking it into the base name would fragment naming for no benefit --
    # add_to_name already disambiguates each cell's own result file.
PF_ALGO_LS = "NR_KLU"

def get_args():
    parser = get_default_args()
    parser.add_argument(
        "--use_init_v_values",
        action="store_true",
        help="Use the same init values as the first powerflow "
        "for the scenario simulation. It is most of the time slower, sometimes much slower")
    parser.add_argument('--nb_threads', type=int, default=1,
                        help=f"Number of concurrent workers used for making the computation "
                             f"(each its own OS process/GraalVM isolate -- see run_multiprocess; "
                             f"named --nb_threads/nb_threads for consistency with the other "
                             f"backends' CLI/output, not because these are Python threads), "
                             f"default to {1}")
    parser.add_argument("--grids", type=str, default=",".join(all_file_names),
                         help="comma-separated matpower filenames to benchmark (default: all "
                              f"{len(all_file_names)} grids in all_file_names) -- lets a caller "
                              "restrict a run to a single grid, e.g. for per-grid nb_pf sizing "
                              "(see olf_grid_sizing.py)")
    return parser

def get_element_alignment(pyp_bus_ids, ls_bus_ids, pyp_row_to_ls, ls_values, pyp_values):
    """Build an array `align` such that `align[i]` is the matpower element
    index corresponding to pypowsybl row `i`.

    pypowsybl and matpower do not always enumerate elements (in particular
    generators) in the same order for a given grid.

    Matching is done by bus (mapped through `pyp_row_to_ls`, the pypowsybl
    bus-row -> matpower bus-id alignment), disambiguating multiple
    elements on the same bus by picking the closest `ls_values`/`pyp_values`
    (eg target_p) pairing from a initialized lightsim2grid lsgrid as lightsim2grid
    is easily available in python and keep the elements in the
    right order.
    """
    pyp_bus_as_ls = np.array([pyp_row_to_ls.get(b, -1) for b in pyp_bus_ids])
    ls_by_bus = {}
    for j, b in enumerate(ls_bus_ids):
        ls_by_bus.setdefault(b, []).append(j)
    pyp_by_bus = {}
    for i, b in enumerate(pyp_bus_as_ls):
        pyp_by_bus.setdefault(b, []).append(i)

    align = np.full(len(pyp_bus_as_ls), -1, dtype=int)
    for b, ls_idxs in ls_by_bus.items():
        pyp_idxs = pyp_by_bus.get(b, [])
        if len(pyp_idxs) != len(ls_idxs):
            raise RuntimeError(
                f"Cannot align elements at bus {b}: {len(ls_idxs)} on the "
                f"lightsim2grid side vs {len(pyp_idxs)} on the pypowsybl side."
            )
        if len(ls_idxs) == 1:
            align[pyp_idxs[0]] = ls_idxs[0]
            continue
        remaining_pyp = list(pyp_idxs)
        for j in ls_idxs:
            pj = ls_values[j]
            best = min(remaining_pyp, key=lambda i: abs(pyp_values[i] - pj))
            align[best] = j
            remaining_pyp.remove(best)
    if (align == -1).any():
        raise RuntimeError("Some pypowsybl elements could not be aligned to a matpower element.")
    return align


def get_natural_order(pyp_grid):
    """When calling pyp_grid.get_buses()
    from a pyp_grid initialized from matpower 
    .mat format, the resulting buses are not in the same order
    as the matpower case frames.
    
    For example in case14.m, bus id (0 indexed) 4 ends up as VL-5_0 (6th row)
    of pypowsybl get_buses()
    
    This functions tries to reorder it...
    
    It uses the names of buses, voltage levels and trafos...
    Might not be "future proof" but worked fine for the paper for all grid sizes.
    """
    trafos = pyp_grid.get_2_windings_transformers()
    buses = pyp_grid.get_buses()
    buses["orig_id"] = [str(re.sub(r"_0$", "", re.sub(r"^VL\-", "", el))) 
                        if el.endswith("_0") else -1
                        for el in buses.index]
    
    mask_issues = buses["orig_id"] == -1
    global_iter = 0
    while mask_issues.any():
        tmp_issues = buses.loc[mask_issues]
        trafo_1 = trafos["bus1_id"].isin(tmp_issues.index)
        trafo_2 = trafos["bus2_id"].isin(tmp_issues.index)
        usefull_trafo_a = trafo_1 & (~trafo_2)
        usefull_trafo_b = trafo_2 & (~trafo_1)
        if not usefull_trafo_a.any() and not usefull_trafo_b.any():
            # side 1 is labeled => I need to label side 2
            print("Error some buses will not be set")
        
        if usefull_trafo_b.any():
            # find the concerned buses
            buses_ids_fixable = tmp_issues.index.isin(trafos["bus2_id"])
            buses_names_fixable = tmp_issues.loc[buses_ids_fixable].index
            for b_nm in buses_names_fixable:
                trafo_anchor = trafos.loc[trafos["bus2_id"] == b_nm]
                tr_nm = trafo_anchor.index[0]
                bus_id = re.sub(r"TWT\-[0-9]+\-", "", tr_nm)
                buses.loc[b_nm,"orig_id"] = bus_id
                
        if usefull_trafo_a.any():
            # find the concerned buses
            buses_ids_fixable = tmp_issues.index.isin(trafos["bus1_id"])
            buses_names_fixable = tmp_issues.loc[buses_ids_fixable].index
            for b_nm in buses_names_fixable:
                trafo_anchor = trafos.loc[trafos["bus1_id"] == b_nm]
                tr_nm = trafo_anchor.index[0]
                bus_id = re.sub(r"TWT\-", "", tr_nm)
                bus_id = re.sub(r"\-[0-9]+$", "", bus_id)
                buses.loc[b_nm,"orig_id"] = bus_id
        mask_issues = buses["orig_id"] == -1
        global_iter += 1
        if global_iter > 10:
            print("Unable to find labels for some buses")
            break
    return buses


def build_ac_params(voltage_init_mode, network_cache_enabled):
    """Shared Parameters constructor for both the warm-up call and the timed per-scenario
    solves, single-process and multiprocessing worker alike.
    """
    return pypl.Parameters(
        voltage_init_mode=voltage_init_mode,
        use_reactive_limits=False,  # disable "outer loop"
        twt_split_shunt_admittance=True,
        component_mode=pypl.ComponentMode.MAIN_SYNCHRONOUS,
        hvdc_ac_emulation=False,  
        transformer_voltage_control_on=False,  # also disable "outer loop"
        provider_parameters={
            # remove all outer loops in OLF (powerflow)
            "outerLoopNames": "",  
            "slackDistributionFailureBehavior": "FAIL",
            # 0 does not work, so we put 1 but this still makes sure there are no outer loops
            "maxOuterLoopIterations": "1",  
            "maxNewtonRaphsonIterations": "30",
            # remove a heuristic: with this heuristic results are not identical
            "generatorsWithZeroMwTargetAreNotStarted": "false",  
            # tighten to match other powerflows TOL_PF
            "newtonRaphsonConvEpsPerEq": f"{TOL_PF:.2e}",  
            "networkCacheEnabled": "true" if network_cache_enabled else "false",
        }
    )


def run_scenario_range(
        grid, scen_params, load_p, load_q, gen_p, load_align, gen_align,
        align_bus_id, start, end, compute_voltages, voltages,
        with_q=True):
    """Run scenarios [start, end) against `grid`, mutating it in place scenario by
    scenario (update_loads/update_generators then run_ac). 
    
    Used both directly (nb_threads <= 1) and inside each
    multiprocessing worker (see _mp_worker), where `load_p`/`load_q`/`gen_p` are already
    that worker's own local [0, end-start) chunk and `voltages` (if any) is that worker's
    own local array -- so this function never needs to know whether it is the only
    consumer of these arrays or one of several concurrent ones.
    """
    attrs = ["p0"]
    if with_q:
        attrs.append("q0")
    load_df = grid.get_loads(attributes=attrs)
    if with_q:
        gen_df = grid.get_generators(attributes=["target_p"])
    total_modif = 0.
    total_pf = 0.
    total_get_data = 0.
    nb_processed = 0
    for scen in range(start, end):
        beg_modif = time.perf_counter()
        load_df["p0"] = load_p[scen][load_align]
        if with_q:
            load_df["q0"] = load_q[scen][load_align]
        grid.update_loads(load_df)
        if with_q:
            gen_df["target_p"] = gen_p[scen][gen_align]
            grid.update_generators(gen_df)
        end_modif = time.perf_counter()
        total_modif += end_modif - beg_modif

        beg_pf = time.perf_counter()
        res = pypl.run_ac(grid, parameters=scen_params)
        end_pf = time.perf_counter()
        total_pf += end_pf - beg_pf
        nb_processed += 1
        # res is a List[ComponentResult], one entry per network component
        # (component_mode=MAIN_SYNCHRONOUS in build_ac_params -> exactly one entry here);
        # it is only ever empty in pathological cases pypowsybl itself doesn't produce, so
        # `if not res` never actually caught a non-convergence -- the real signal is each
        # component's own .status.
        if res[0].status != pypl.ComponentStatus.CONVERGED:
            continue

        if compute_voltages:
            beg_get_data = time.perf_counter()
            df_vl = grid.get_voltage_levels(attributes=["nominal_v"])
            df_bus = grid.get_buses(attributes=["v_mag", "v_angle", "voltage_level_id"])
            cplx_v = df_bus["v_mag"] / df_vl.loc[df_bus["voltage_level_id"], "nominal_v"].values + 0j
            cplx_v *= np.exp(1j * np.deg2rad(df_bus["v_angle"]))
            end_get_data = time.perf_counter()
            total_get_data += end_get_data - beg_get_data
            # save the voltages in the same order as lightsim2grid
            voltages[scen, align_bus_id] = cplx_v.to_numpy()

    return total_modif, total_pf, total_get_data, nb_processed


# Set once per worker process by _worker_init, read by _mp_worker -- process-local
# globals, never shared across workers (each is its own OS process/GraalVM isolate).
# this work because we use "spawn" multi processing method.
_worker_grid = None
_worker_scen_params = None


def _worker_init(fn, init_method, use_init_v_values, barrier):
    """Pool(initializer=...) hook: runs exactly once in each worker process, before that
    worker ever pulls a real task off the queue -- this is where the (expensive, one-time)
    grid load + warm-up belongs, kept out of the timed _mp_worker task function entirely.

    Runs with the "spawn" start method, so this whole module is freshly re-imported in the
    child before this function ever runs -- pypowsybl get a brand new,
    independent GraalVM isolate here, never one inherited from the parent (which is why
    "fork" is not used: the parent's isolate is already live with its own native threads
    by the time workers would be created, and forking a process with a live native isolate
    is unsafe).

    `barrier.wait()` at the end (parties = n_workers + 1, the parent included -- see
    run_multiprocess) is what lets the parent exclude this setup time from its wall_time
    measurement: the parent also calls barrier.wait() right before starting its clock, so
    it only proceeds once every worker has reached this same point.
    """
    global _worker_grid, _worker_scen_params
    if init_method == "flat":
        init_ = pypl.VoltageInitMode.UNIFORM_VALUES
    else:
        init_ = pypl.VoltageInitMode.DC_VALUES

    _worker_grid = pypn.load(os.path.join(REF_PATH, f"{fn}at"))
    pypl.run_ac(_worker_grid, parameters=build_ac_params(init_, network_cache_enabled=True))
    _worker_scen_params = build_ac_params(
        init_ if use_init_v_values else pypl.VoltageInitMode.PREVIOUS_VALUES,
        network_cache_enabled=True,
    )
    barrier.wait()


def _mp_worker(load_p_chunk, load_q_chunk, gen_p_chunk, load_align, gen_align, align_bus_id,
               compute_voltages):
    """The timed part only: solve this worker's own [0, n_local) chunk against the
    Network/Parameters _worker_init already built for this process."""
    n_local = load_p_chunk.shape[0]
    voltages = (np.zeros((n_local, _worker_grid.get_buses().shape[0]), dtype=complex)
                if compute_voltages else None)

    total_modif, total_pf, total_get_data, nb_processed = run_scenario_range(
        _worker_grid, _worker_scen_params, load_p_chunk, load_q_chunk, gen_p_chunk,
        load_align, gen_align, align_bus_id, 0, n_local, compute_voltages, voltages,
    )
    return total_modif, total_pf, total_get_data, nb_processed, voltages


def run_multiprocess(nb_threads, fn, init_method, use_init_v_values,
                      load_p, load_q, gen_p, load_align, gen_align, align_bus_id,
                      compute_voltages):
    """We tested an equivalent threading.Thread implementation but found it was slower and less
    precise on the timing measurments.
    
    When --nb_threads > 1: spawns
    nb_threads processes (multiprocessing.Pool, "spawn" context -- not "fork", since the
    parent already has a live GraalVM isolate loaded by the time this runs). Each worker
    loads its own Network/Parameters once via _worker_init, then solves its own
    [start, end) chunk of load_p/load_q/gen_p (sliced here, so pickling cost to ship data
    to a worker scales with that worker's share of the work, not with nb_scen itself).

    wall_time excludes process spawn / GraalVM isolate startup / grid load / warm-up:
    the Barrier below has n_workers + 1 parties (every worker, plus this function itself),
    so the parent's own barrier.wait() only returns once every worker has finished
    _worker_init -- at that point beg_wall starts, and only the actual scenario-solving
    (pool.starmap dispatching _mp_worker) is timed.
    """
    # spawn so that every process has everything (including graalvm) it needs
    ctx = mp.get_context("spawn")
    nb_scen = load_p.shape[0]
    n_workers = min(nb_threads, nb_scen)
    bounds = np.linspace(0, nb_scen, n_workers + 1).astype(int)
    # this barrier is used to make sure every process are properly initialized
    # before the computation is timed
    barrier = ctx.Barrier(n_workers + 1)

    with ctx.Pool(processes=n_workers, initializer=_worker_init,
                  initargs=(fn, init_method, use_init_v_values, barrier)) as pool:
        args = [
            (load_p[bounds[i]:bounds[i + 1]], load_q[bounds[i]:bounds[i + 1]],
             gen_p[bounds[i]:bounds[i + 1]],
             load_align, gen_align, align_bus_id, compute_voltages)
            for i in range(n_workers)
        ]
        # blocks until every worker has finished _worker_init
        barrier.wait()  
        
        # at this stage all worker has its own "context": grid is loaded, data is there
        # only the powerflow computation is timed
        beg_wall = time.perf_counter()
        results = pool.starmap(_mp_worker, args)
        wall_time = time.perf_counter() - beg_wall
    # agregate all results
    total_modif = sum(r[0] for r in results)
    total_pf = sum(r[1] for r in results)
    total_get_data = sum(r[2] for r in results)
    nb_processed = sum(r[3] for r in results)
    voltages = np.concatenate([r[4] for r in results], axis=0) if compute_voltages else None
    return total_modif, total_pf, total_get_data, nb_processed, wall_time, voltages


if __name__ == "__main__":    
    args = get_args().parse_args()
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_flows = args.save_flows
    save_voltages = args.save_voltages
    evaluate_kcl = args.evaluate_kcl
    need_voltages = save_voltages or evaluate_kcl
    nb_pf_total = int(args.nb_pf)
    tol_pf = TOL_PF
    use_init_v_values = args.use_init_v_values
    nb_threads = int(args.nb_threads)
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]

    benchmark_results = {}
    if add_to_name:
        nm_results = BASE_EXPE_NAME_NO_NBPF.format(
            sample_data_meth,
            "1" if use_init_v_values else "0",
            )
    else:
        nm_results = BASE_EXPE_NAME.format(
            sample_data_meth,
            nb_pf_total,
            "1" if use_init_v_values else "0",
            )
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"
    full_path_As = f"{nm_tmp}_{{}}_As.npy"
    for fn in tqdm(grids):
        tmp_res_dict = {}
        # load the equivalent .mat file
        pyp_grid = pypn.load(os.path.join(REF_PATH, f"{fn}at"))
        # original sin: pypowsybl does not expose the buses
        # in the same order as matpower does
        buses_order = get_natural_order(pyp_grid)
        # we should have something working here
        # but there is a second follow-up...
        # id here are given with matpower names (buses like 1000006)
        # so later we need to convert them to 
        # continuous series of integers in 0..nb_bus
        
        init_method = get_init_method(fn)
        
        ### warm up OLF
        if init_method == "flat":
            init_ = pypl.VoltageInitMode.UNIFORM_VALUES
        else:
            # force DC for pypowsybl
            init_ = pypl.VoltageInitMode.DC_VALUES
        pyp_params = build_ac_params(init_, network_cache_enabled=True)
        # just a warm-up
        pypl.run_ac(pyp_grid, parameters=pyp_params)
        
        #### used to retrieve exact same data as in other experiments
        this_path = os.path.join(REF_PATH, fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(this_path)
        
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
            cf = CaseFrames(this_path)
            v_init = cf.bus["VM"] * np.exp(1j * np.deg2rad(cf.bus["VA"]))
        lsgrid.change_algorithm(PF_ALGO_LS)
        beg_ls = time.perf_counter()
        res_1 = lsgrid.ac_pf(v_init, 10, TOL_PF)
        end_ls = time.perf_counter()
        
        # and i'm not even proud of it...
        align_bus_id = lsgrid._orig_to_ls[buses_order["orig_id"].values.astype(int)]

        # build the load / generator alignment (pypowsybl row -> matpower index):
        # positional order between the two loaders is NOT guaranteed to match,
        # in particular for generators on grids with multiple units per bus
        # (eg case6515rte, confirmed to swap setpoints between physical generators
        # otherwise). See get_element_alignment() docstring.
        pyp_row_to_ls = dict(zip(buses_order.index, align_bus_id))
        ls_gens = lsgrid.get_generators()
        ls_gen_bus = np.array([g.bus_id for g in ls_gens])
        ls_gen_p0 = np.array([g.target_p_mw for g in ls_gens])
        pyp_gen_df_align = pyp_grid.get_generators(attributes=["bus_id", "target_p"])
        gen_align = get_element_alignment(
            pyp_gen_df_align["bus_id"].to_numpy(), ls_gen_bus, pyp_row_to_ls,
            ls_gen_p0, pyp_gen_df_align["target_p"].to_numpy()
        )
        ls_loads = lsgrid.get_loads()
        ls_load_bus = np.array([ld.bus_id for ld in ls_loads])
        ls_load_p0 = np.array([ld.target_p_mw for ld in ls_loads])
        pyp_load_df_align = pyp_grid.get_loads(attributes=["bus_id", "p0"])
        load_align = get_element_alignment(
            pyp_load_df_align["bus_id"].to_numpy(), ls_load_bus, pyp_row_to_ls,
            ls_load_p0, pyp_load_df_align["p0"].to_numpy()
        )

        # # these checks are here to make sure the data are in the same
        # # order as matpower. To that end, as always, we use lightsim2grid.
        # df_vl = pyp_grid.get_voltage_levels(attributes=["nominal_v"])
        # df_bus = pyp_grid.get_buses(attributes=["v_mag", "v_angle", "voltage_level_id"])
        # cplx_v = df_bus["v_mag"] / df_vl.loc[df_bus["voltage_level_id"], "nominal_v"].values + 0j
        # cplx_v *= np.exp(1j * np.deg2rad(df_bus["v_angle"]))
        # volt_for_kcl = cplx_v.to_numpy().copy()
        # volt_for_kcl[align_bus_id] = cplx_v.to_numpy().copy()
        # res = (np.abs(lsgrid.check_solution(volt_for_kcl, False)) >= 0.008).nonzero()
        
        # res_for_kcl = lsgrid.ac_pf(volt_for_kcl, 10, TOL_PF)
        # (np.abs(res_for_kcl - volt_for_kcl) >= 0.002).nonzero()
        
        ### generate exact same injections for all the methods
        gen_v = None
        load_p, load_q, gen_p = get_injection_from_base(
            fn,
            lsgrid,
            sample_meth=sample_data_meth,
            seed=seed,
            nb_pf_total=nb_pf_total)
        
        # run the experiment
        nb_scen = load_p.shape[0]

        if nb_threads <= 1:
            pyp_scen_params = build_ac_params(
                init_ if use_init_v_values else pypl.VoltageInitMode.PREVIOUS_VALUES,
                network_cache_enabled=True,
            )
            if need_voltages:
                voltages = np.zeros(
                    (nb_scen, pyp_grid.get_buses().shape[0]),
                    dtype=complex
                )
            else:
                voltages = None
            total_modif, total_pf, total_get_data, nb_processed = run_scenario_range(
                pyp_grid, pyp_scen_params, load_p, load_q, gen_p,
                load_align, gen_align, align_bus_id, 0, nb_scen,
                need_voltages, voltages, with_q=True
            )
            wall_time = total_modif + total_pf
        else:
            # see run_multiprocess docstring: each worker is its own OS process with its
            # own GraalVM isolate (no shared Network/Parameters/JVM-global cache to race
            # on across workers, unlike the previous threading.Thread implementation).
            (total_modif, total_pf, total_get_data, nb_processed,
             wall_time, voltages) = run_multiprocess(
                nb_threads, fn, init_method, use_init_v_values,
                load_p, load_q, gen_p, load_align, gen_align, align_bus_id,
                need_voltages,
            )

        if evaluate_kcl:
            # voltages is already in matpower bus order (see the align_bus_id
            # scatter in run_scenario_range: "save the voltages in the same order as
            # matpower"), so no bus reprojection is needed here (unlike
            # pp_injection.py). A scenario whose run_ac() didn't report CONVERGED is
            # simply never written, so it stays at its pre-allocated all-zero row -- an
            # unambiguous per-scenario convergence flag.
            solved_mask = np.abs(voltages).sum(axis=1) > 0.
            if solved_mask.sum() < solved_mask.shape[0]:
                tmp_res_dict["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()
            kcl_checker = KCLChecker(this_path)
            kcl_mismatch = kcl_checker.check_kcl_injection(
                load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask], voltages[solved_mask])
            tmp_res_dict.update(summarize_kcl_mismatch(kcl_mismatch))
            tmp_res_dict["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
            tmp_res_dict.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))

        tmp_res_dict["time_series"] = {
            # aggregate CPU time summed across workers when nb_threads > 1 -- NOT
            # elapsed time, see wall_time for that (mirrors the Julia baselines)
            "total_time": total_modif + total_pf,
            "wall_time": wall_time,
            "nb_solved": nb_processed,
            "solver_time": total_pf,
            "modif_time": total_modif,
            "total_get_data": total_get_data,
            "nb_threads": nb_threads,
        }
        
        benchmark_results[fn] = tmp_res_dict
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)
        
        case_nm, _ = os.path.splitext(fn)
        if save_voltages:
            np.save(file=full_path_Vs.format(case_nm), arr=voltages)
