# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import copy
import json
import multiprocessing
import os
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.converter.matpower import from_mpc
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

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


BASE_EXPE_NAME = "pp_injection_{}_{}_{}"
PF_ALGO_LS = "NR_KLU"
PP_ALGO = "nr"
# matpower/lightsim2grid/pypowsybl all build Ybus from the branch's raw pi-equivalent
# parameters; pandapower's default "t" trafo_model re-derives a T-equivalent circuit for
# transformers, which is a different (non-equivalent) Ybus for off-nominal-ratio branches.
# "pi" is what pandapower's own docs recommend "for validation with other software that
# uses the pi-model".
PP_TRAFO_MODEL = "pi"

# pandapower's own pp.runpp(algorithm="nr", ...) silently swaps its Newton-Raphson loop for
# lightsim2grid's own NR_KLU/NRSing_KLU C++ solver whenever lightsim2grid is importable and
# the grid has no exotic FACTS/TDPF/multi-slack features (see
# pandapower.auxiliary._check_lightsim2grid_compatibility, gated by the `lightsim2grid=`
# kwarg, which defaults to "auto"). Confirmed directly: net._options["lightsim2grid"] == True
# for every grid in this benchmark unless explicitly disabled. That means every pp.runpp()
# call up to now was actually timing lightsim2grid, not pandapower's own implementation.
# These three kwarg combos isolate the three real, selectable NR backends pandapower exposes
# through pp.runpp() (a 4th, power-grid-model via pp.runpp_pgm(), is a separate function with
# its own init/tolerance API and is intentionally out of scope here):
#   - lightsim2grid : lightsim2grid's own C++ NR_KLU/NRSing_KLU solver <- used in this experiment
#   - pp_numba      : pandapower's own NR, numba-JIT makeYbus/Jacobian/pfsoln  <- NOT USED, slower
#   - pp_nonumba    : pandapower's own NR, pure-Python/scipy makeYbus/Jacobian/pfsoln <- NOT USED, slower
PP_BACKEND_KWARGS = {
    "lightsim2grid": {"lightsim2grid": True},  # <- used in this experiment, faster
    "pp_numba": {"lightsim2grid": False, "numba": True},  # <- NOT USED, slower
    "pp_nonumba": {"lightsim2grid": False, "numba": False},  # <- NOT USED, slower
}


def get_args():
    parser = get_default_args()
    parser.add_argument(
        "--mode", type=str, default="both", choices=["loop", "timeseries", "both"],
        help="Which pandapower benchmark mode(s) to run: a per-scenario python loop calling "
             "pp.runpp() (like olf_injection.py), pandapower's own timeseries module, or both.")
    parser.add_argument(
        "--pp_backend", type=str, default="lightsim2grid",
        choices=list(PP_BACKEND_KWARGS.keys()),
        help="Which Newton-Raphson backend pp.runpp() should use: lightsim2grid's own C++ "
             "solver (pandapower's default when lightsim2grid is installed), pandapower's own "
             "implementation with numba JIT, or pandapower's own implementation without numba.")
    parser.add_argument('--nb_threads', type=int, default=1,
                        help=f"Number of processes used for making the computation, loop mode "
                             f"only, default to {1} (see run_loop_mode_mp: pandapower's runpp() "
                             f"is pure-Python/GIL-bound, so this uses multiprocessing, not "
                             f"threads, unlike ls_injection.py/pgm_injection.py/exapf_injection.jl)")
    parser.add_argument("--grids", type=str, default=",".join(all_file_names),
                         help="comma-separated matpower filenames to benchmark (default: all "
                              f"{len(all_file_names)} grids in all_file_names) -- lets a caller "
                              "restrict a run to a single grid, e.g. for per-grid nb_pf sizing "
                              "(see pp_grid_sizing.py)")
    return parser


def get_element_alignment(pp_bus_ids, ls_bus_ids, pp_row_to_ls, ls_values, pp_values):
    """Build an array `align` such that `align[i]` is the lightsim2grid element
    index corresponding to pandapower row `i`.

    Mirrors get_element_alignment() in olf_injection.py (duplicated here rather than
    imported, so this file does not need pypowsybl installed). Matching is done by
    bus, disambiguating multiple elements on the same bus by picking the closest
    `ls_values`/`pp_values` pairing (eg target_p).
    """
    pp_bus_as_ls = np.array([pp_row_to_ls.get(b, -1) for b in pp_bus_ids])
    ls_by_bus = {}
    for j, b in enumerate(ls_bus_ids):
        ls_by_bus.setdefault(b, []).append(j)
    pp_by_bus = {}
    for i, b in enumerate(pp_bus_as_ls):
        pp_by_bus.setdefault(b, []).append(i)

    align = np.full(len(pp_bus_as_ls), -1, dtype=int)
    for b, ls_idxs in ls_by_bus.items():
        pp_idxs = pp_by_bus.get(b, [])
        if len(pp_idxs) != len(ls_idxs):
            raise RuntimeError(
                f"Cannot align elements at bus {b}: {len(ls_idxs)} on the "
                f"lightsim2grid side vs {len(pp_idxs)} on the pandapower side."
            )
        if len(ls_idxs) == 1:
            align[pp_idxs[0]] = ls_idxs[0]
            continue
        remaining_pp = list(pp_idxs)
        for j in ls_idxs:
            pj = ls_values[j]
            best = min(remaining_pp, key=lambda i: abs(pp_values[i] - pj))
            align[best] = j
            remaining_pp.remove(best)
    if (align == -1).any():
        raise RuntimeError("Some pandapower elements could not be aligned to a lightsim2grid element.")
    return align


def get_bus_alignment(lsgrid, net):
    """pandapower's matpower converter always sets net.bus.index = matpower_bus_number - 1
    (see pandapower.converter.matpower.from_mpc._adjust_ppc_indices).

    Returns `pp_bus_to_ls` (array, position i -> matpower bus id for net.bus row i)
    and `pp_row_to_ls` (dict, pandapower bus index label -> lightsim2grid bus id).
    """
    pp_bus_to_ls = lsgrid._orig_to_ls[net.bus.index.values + 1]
    pp_row_to_ls = dict(zip(net.bus.index.values, pp_bus_to_ls))
    return pp_bus_to_ls, pp_row_to_ls


def get_gen_alignment(lsgrid, net):
    """pandapower
    splits the generator table table from matpower case 
    into net.ext_grid / net.gen / net.sgen (a generator "on" a non-PV
    bus becomes an sgen). `net._from_ppc_lookups["gen"]` records, for each original matpower
    gen row, which pandapower table/row it ended up in -- so alignment is a direct positional
    lookup here, no bus/value fuzzy-matching needed (unlike loads, see get_load_alignment()).

    matpower also lets several generator units share one PV bus; `_gen_to_which()` in
    pandapower's converter keeps only the *first* such unit as a true (voltage-controlled)
    `gen` and demotes every other co-located unit to a fixed-P/fixed-Q `sgen`, discarding its
    own voltage target and freezing its Q at whatever the matpower file's solved operating
    point happened to be. other solvers (eg. lightsim2grid/pypowsybl) instead keep 
    every co-located unit voltage-regulating -- confirmed to cause large (0.03-0.3 pu) 
    voltage errors on grids with such buses (case3120sp/case3375wp/case_ACTIVSg2000), 
    present even on the unperturbed base
    case, i.e. unrelated to the scenario sweep. 
    
    Promote those gen-origin sgen rows back to
    real net.gen rows here, it uses lightsim2grid lsgrid's 
    own target_vm_pu as the voltage setpoint pandapower
    otherwise has no way to recover (net.sgen carries no vm_pu column at all).
    """
    gen_lookup = net._from_ppc_lookups["gen"]
    is_gen = gen_lookup["element_type"] == "gen"
    is_sgen = gen_lookup["element_type"] == "sgen"
    ls_idx_for_gen = gen_lookup.index[is_gen].to_numpy()
    ls_idx_for_gen_sgen = gen_lookup.index[is_sgen].to_numpy()
    gen_sgen_element_idx = gen_lookup.loc[is_sgen, "element"].to_numpy()

    if len(gen_sgen_element_idx):
        ls_gens = lsgrid.get_generators()
        rows = net.sgen.loc[gen_sgen_element_idx]
        pp.create_gens(
            net, buses=rows["bus"].to_numpy(), p_mw=rows["p_mw"].to_numpy(),
            vm_pu=[ls_gens[i].target_vm_pu for i in ls_idx_for_gen_sgen],
            sn_mva=rows["sn_mva"].to_numpy(),
            max_p_mw=rows["max_p_mw"].to_numpy(), min_p_mw=rows["min_p_mw"].to_numpy(),
            max_q_mvar=rows["max_q_mvar"].to_numpy(), min_q_mvar=rows["min_q_mvar"].to_numpy(),
            in_service=rows["in_service"].to_numpy(), controllable=True,
        )
        net.sgen.drop(index=gen_sgen_element_idx, inplace=True)
        # net.gen row order after this is: original "gen" rows (unchanged) then the newly
        # appended, promoted rows, in the same order as ls_idx_for_gen_sgen / rows above
        ls_idx_for_gen = np.concatenate([ls_idx_for_gen, ls_idx_for_gen_sgen])
    return ls_idx_for_gen


def get_load_alignment(lsgrid, net, pp_row_to_ls):
    """matpower represents loads as bus attributes (PD, QD); pandapower keeps a negative-PD
    bus as an `sgen` instead of a `load`. 
    
    lightsim2grid does not make that split (confirmed:
    it imports these as regular, possibly-negative-p, Load objects). Here also, lightsim2grid
    is used as the reference (as the matpower order is conserved) so lsgrid.get_loads()
    is matched against net.load + net.sgen. Call this after get_gen_alignment(), which
    promotes the gen-origin sgen rows to net.gen -- every sgen row remaining here is bus-origin.
    """
    ls_loads = lsgrid.get_loads()
    ls_load_bus_all = np.array([l.bus_id for l in ls_loads])
    ls_load_p0_all = np.array([l.target_p_mw for l in ls_loads])
    load_connected = ls_load_bus_all != -1
    ls_load_bus = ls_load_bus_all[load_connected]
    ls_load_p0 = ls_load_p0_all[load_connected]

    n_load = len(net.load)
    pp_load_bus = np.concatenate([net.load["bus"].values, net.sgen["bus"].values])
    pp_load_p = np.concatenate([net.load["p_mw"].values, -net.sgen["p_mw"].values])
    load_align = get_element_alignment(pp_load_bus, ls_load_bus, pp_row_to_ls, ls_load_p0, pp_load_p)
    return load_connected, load_align, n_load, net.sgen.index.to_numpy()


def set_pp_slack_angle(net, v_init, pp_row_to_ls):
    """pandapower's ext_grid.va_degree is a hard boundary condition (the slack bus is held at
    that fixed angle throughout the solve), always taken from the matpower file's own VA
    column by from_mpc() -- regardless of the `init` kwarg passed to runpp(), which only
    affects the iteration starting point. We force it to 0 for full reproducibility with 
    voltages from other solvers (eg pypowsybl / OLF or lightsim2grid).
    """
    v_init_arr = np.asarray(v_init)
    for i in net.ext_grid.index:
        pp_bus = net.ext_grid.at[i, "bus"]
        ls_bus = pp_row_to_ls[pp_bus]
        net.ext_grid.at[i, "va_degree"] = np.degrees(np.angle(v_init_arr[ls_bus]))


def bypass_lossy_trafo_conversion(net, cf):
    """pandapower's from_mpc() converts every off-nominal-ratio matpower branch into a
    net.trafo row via a percent-impedance parameterization (vk_percent/vkr_percent/
    i0_percent/pfe_kw, see _from_ppc_branch()) that is NOT an exact round-trip of the raw
    (r, x, b, tap, shift) pi-equivalent -- confirmed, via direct Ybus comparison, to be the
    root cause of a real ~0.26-0.30 pu voltage mismatch vs lightsim2grid/pypowsybl on
    case3120sp/case3375wp (independent of trafo_model="t" vs "pi" and of NR init strategy).

    net.impedance instead models a fully general asymmetric 2-port branch (independent
    from/to series admittance + independent from/to shunt, no "tap" abstraction) whose Ybus
    stamp (see pandapower.pypower.makeYbus.branch_vectors) can represent the *exact* matpower
    pi-equivalent -- including phase shifters -- via a direct analytic inversion:
        y_s = 1/(r+jx); tap = ratio * exp(j*shift)
        Ytt = y_s + j*b/2 ; Yff = Ytt/(tap*conj(tap)) ; Yft = -y_s/conj(tap) ; Ytf = -y_s/tap
        (this is matpower's own makeYbus formula, and pandapower's impedance branch_vectors
        reduces to exactly this form when tap=1: Yff=Ysf+Bcf/2, Ytt=Yst+Bct/2, Yft=-Ysf, Ytf=-Yst)
    so setting rft_pu+j*xft_pu = -1/Yft, rtf_pu+j*xtf_pu = -1/Ytf, gf_pu+j*bf_pu = Yff+Yft,
    gt_pu+j*bt_pu = Ytt+Ytf, and sn_mva = net.sn_mva (so the impedance element's own
    per-unit rescaling in _calc_impedance_parameters_from_dataframe becomes a no-op),
    reproduces the trafo row's Ybus contribution exactly, bit for bit.

    Replaces every net.trafo row that came from an off-nominal-ratio branch with an
    equivalent net.impedance row built directly from the raw matpower data, bypassing
    pandapower's lossy trafo classification entirely. `cf` is the CaseFrames of the same
    matpower file `net` was converted from (cf.branch is in the same row order as the
    ppc/net._from_ppc_lookups["branch"] lookup).
    """
    branch_lookup = net._from_ppc_lookups["branch"]
    is_trafo = branch_lookup["element_type"] == "trafo"
    if not is_trafo.any():
        return
    mpc_rows = branch_lookup.index[is_trafo].to_numpy()
    trafo_idx = branch_lookup.loc[is_trafo, "element"].astype(int).to_numpy()

    br = cf.branch.iloc[mpc_rows]
    r = br["BR_R"].to_numpy(dtype=float)
    x = br["BR_X"].to_numpy(dtype=float)
    b = br["BR_B"].to_numpy(dtype=float)
    ratio = br["TAP"].to_numpy(dtype=float)
    ratio = np.where(ratio == 0., 1., ratio)
    shift = br["SHIFT"].to_numpy(dtype=float)
    in_service = net.trafo.loc[trafo_idx, "in_service"].to_numpy()

    y_s = 1. / (r + 1j * x)
    tap = ratio * np.exp(1j * np.deg2rad(shift))
    Ytt = y_s + 1j * b / 2
    Yff = Ytt / (tap * np.conj(tap))
    Yft = -y_s / np.conj(tap)
    Ytf = -y_s / tap

    z_ft = -1. / Yft
    z_tf = -1. / Ytf
    y_f = Yff + Yft
    y_t = Ytt + Ytf

    # net.bus.index == matpower bus id - 1 (see get_bus_alignment()), so F_BUS/T_BUS map
    # arithmetically -- no need to go through net.trafo's hv/lv bus labelling (which may have
    # from/to swapped relative to the raw matpower row, see _from_ppc_branch's to_vn_is_leq).
    from_bus = br["F_BUS"].to_numpy(dtype=int) - 1
    to_bus = br["T_BUS"].to_numpy(dtype=int) - 1

    pp.create_impedances(
        net, from_buses=from_bus, to_buses=to_bus,
        rft_pu=z_ft.real, xft_pu=z_ft.imag,
        rtf_pu=z_tf.real, xtf_pu=z_tf.imag,
        gf_pu=y_f.real, bf_pu=y_f.imag,
        gt_pu=y_t.real, bt_pu=y_t.imag,
        sn_mva=np.full(len(trafo_idx), net.sn_mva),
        in_service=in_service,
    )
    net.trafo.drop(index=trafo_idx, inplace=True)


def get_ls_v_init(this_path, lsgrid, init_method, cf=None):
    """To make the powerflow converge, the standard 
    "init from a flat complex voltage" strategy does not work.
    
    In this case, we try to either init from the dc approximation, computed by a reference
    solver or from the data directly in the file.
    
    As it was the easiest to use, we chose to use the lightsim2grid for the dc approximation
    when needed.
    """
    if init_method == "flat":
        return np.ones(lsgrid.total_bus(), dtype=complex)
    if init_method == "dc":
        v_init_dc = np.ones(lsgrid.total_bus(), dtype=complex)
        return lsgrid.dc_pf(v_init_dc, 10, 1e-6)
    if cf is None:
        cf = CaseFrames(this_path)
    return cf.bus["VM"] * np.exp(1j * np.deg2rad(cf.bus["VA"]))


def get_pp_init_kwarg(net, init_method, v_init, pp_bus_to_ls):
    """Return the right va and vm used to initialize pandapower
    Newton Raphson algorithm.
    
    They are initialized from flat if the grid converges, or from 
    dc (if flat leads to divergence) or from the votlage
    on the files if dc (and flat) lead to divergence.
    """
    if init_method == "flat":
        return "flat"
    v_init_arr = np.asarray(v_init)
    vm = np.abs(v_init_arr)[pp_bus_to_ls]
    va = np.degrees(np.angle(v_init_arr))[pp_bus_to_ls]
    net["res_bus"] = pd.DataFrame({"vm_pu": vm, "va_degree": va}, index=net.bus.index)
    return "results"


def scenario_arrays(load_p, load_q, gen_p, load_connected, load_align, gen_align_gen):
    """Reindex the (nb_scenario, nb_element) sampled arrays (in matpower element order,
    the same arrays fed to the other solvers eg ls_injection.py / olf_injection.py) into pandapower row order for
    every destination table this benchmark touches (direct check of complex voltages, kcl evaluation etc.)
    """
    load_p_conn = load_p[:, load_connected]
    load_q_conn = load_q[:, load_connected]
    pp_load_p = load_p_conn[:, load_align]
    pp_load_q = load_q_conn[:, load_align]
    pp_gen_p = gen_p[:, gen_align_gen]
    return pp_load_p, pp_load_q, pp_gen_p


def run_loop_mode(net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, tol_pf, compute_voltages,
                   pp_backend_kwargs):
    """Default mode used here as we found it was faster than the time series mode.
    
    A python loop is made to evaluate the scenario one after the others while initializing always from 
    the same complex voltages.
    
    """
    nb_scen = pp_load_p.shape[0]
    total_modif = 0.
    total_pf = 0.
    total_solver = 0.
    nb_solved = 0
    voltages = np.zeros((nb_scen, len(net.bus)), dtype=complex) if compute_voltages else None
    for scen in range(nb_scen):
        beg_modif = time.perf_counter()
        net.load["p_mw"] = pp_load_p[scen, :n_load]
        net.load["q_mvar"] = pp_load_q[scen, :n_load]
        if len(bus_sgen_idx):
            net.sgen.loc[bus_sgen_idx, "p_mw"] = -pp_load_p[scen, n_load:]
            net.sgen.loc[bus_sgen_idx, "q_mvar"] = -pp_load_q[scen, n_load:]
        net.gen["p_mw"] = pp_gen_p[scen]
        end_modif = time.perf_counter()
        total_modif += end_modif - beg_modif

        beg_pf = time.perf_counter()
        try:
            # topology is unchanged between scenarios, only setpoints are: skip the
            # connectivity check on every call to keep the per-scenario cost comparable
            # to what ls_injection.py / olf_injection.py measure
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pp.runpp(net,
                         algorithm=PP_ALGO,
                         init="results",
                         tolerance_mva=tol_pf,
                         check_connectivity=False,
                         trafo_model=PP_TRAFO_MODEL,
                         **pp_backend_kwargs)
        except LoadflowNotConverged:
            continue
        end_pf = time.perf_counter()
        total_pf += end_pf - beg_pf
        # net._ppc["et"]: pandapower's own internal elapsed-time measurement of the powerflow
        # (see pandapower.pf.run_newton_raphson_pf._run_newton_raphson_pf's t0/et, propagated
        # through _powerflow -> _ppci_to_net -> net["_ppc"]["et"]). Unlike `solver_wall_time`
        # below (our external wall-clock around the whole pp.runpp() call, including
        # pd2ppc/results extraction Python/pandas overhead), this only covers the NR solve
        # itself -- the same quantity ls_injection.py reports as "solver_time" (ts.solver_time(),
        # the C++-internal timer) and lightSimBackend.py reports via
        # self._grid.get_computation_time()/self._timer_solver (see lightSimBackend.py:1785).
        # Named "solver_time" here too, for consistency with ls_injection.py's result keys.
        total_solver += net._ppc["et"]
        nb_solved += 1

        if compute_voltages:
            voltages[scen] = (net.res_bus["vm_pu"].to_numpy()
                               * np.exp(1j * np.deg2rad(net.res_bus["va_degree"].to_numpy())))

    res = {
        "total_time": total_modif + total_pf,
        "nb_solved": nb_solved,
        "modif_time": total_modif,
        "solver_wall_time": total_pf,
        "solver_time": total_solver,
    }
    return res, voltages


def _scenario_chunks(nb_scen, n_workers):
    """Split `nb_scen` scenarios into up to `n_workers` contiguous, roughly-equal (start, stop)
    ranges (any remainder spread one-per-chunk over the first chunks). Mirrors
    exapf_injection.jl's split_into_groups()."""
    n_workers = max(1, min(n_workers, nb_scen))
    base, rem = divmod(nb_scen, n_workers)
    chunks = []
    lo = 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        chunks.append((lo, lo + size))
        lo += size
    return chunks


def run_loop_mode_mp(net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, tol_pf, compute_voltages,
                      pp_backend_kwargs, nb_threads):
    """Multiprocessing dispatcher around run_loop_mode(), loop mode only.

    Unlike ls_injection.py's InjectionSweepCPP (C++, releases the GIL) or
    exapf_injection.jl's Threads.@spawn (Julia has no GIL), pp.runpp() is pure-Python /
    numba-JIT -- still bound by Python's GIL at the call boundary -- so real Python threads
    would not run any faster. Instead, nb_threads OS *processes* each get their own deep
    copy of `net` (a pandapower net is mutated in place scenario to scenario, so it cannot
    be shared across workers) and independently run run_loop_mode() over a contiguous slice
    of the scenarios, then results are merged -- the same contiguous-chunk-per-worker
    pattern exapf_injection.jl's process_chunk_group/split_into_groups use for --nthreads.

    wall_time excludes the per-worker copy.deepcopy(net) and process-pool spawn -- these are
    this backend's equivalent of the one-time "worker init" cost olf_injection.py's
    run_multiprocess() excludes via its Barrier (there, loading the Network + warm-up
    run_ac(); here, handing each already-warmed-up net to its worker) -- so the clock starts
    right before pool.starmap() dispatches the actual (already-copied, already-spawned)
    workers, timing only the scenario-solving itself.
    """
    nb_scen = pp_load_p.shape[0]
    if nb_threads <= 1:
        beg = time.perf_counter()
        res, voltages = run_loop_mode(
            net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, tol_pf, compute_voltages,
            pp_backend_kwargs)
        res["wall_time"] = time.perf_counter() - beg
        res["nb_threads"] = 1
        return res, voltages

    chunks = _scenario_chunks(nb_scen, nb_threads)
    tasks = [
        (copy.deepcopy(net), pp_load_p[lo:hi], pp_load_q[lo:hi], pp_gen_p[lo:hi], n_load, bus_sgen_idx,
         tol_pf, compute_voltages, pp_backend_kwargs)
        for lo, hi in chunks
    ]
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        beg = time.perf_counter()
        results = pool.starmap(run_loop_mode, tasks)
    wall_time = time.perf_counter() - beg

    res = {
        "total_time": sum(r["total_time"] for r, _ in results),
        "nb_solved": sum(r["nb_solved"] for r, _ in results),
        "modif_time": sum(r["modif_time"] for r, _ in results),
        "solver_wall_time": sum(r["solver_wall_time"] for r, _ in results),
        "solver_time": sum(r["solver_time"] for r, _ in results),
        "wall_time": wall_time,
        "nb_threads": len(tasks),
    }
    voltages = None
    if compute_voltages:
        voltages = np.concatenate([v for _, v in results], axis=0)
    return res, voltages


def run_timeseries_mode(net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, tol_pf,
                         compute_voltages, pp_backend_kwargs):
    """NOT USED IN THIS BENCHMARK, as it is found slower than the "python loop" 
    counterpart.
    """
    nb_scen = pp_load_p.shape[0]
    time_steps = range(nb_scen)

    load_p_df = pd.DataFrame(pp_load_p[:, :n_load], columns=net.load.index)
    load_q_df = pd.DataFrame(pp_load_q[:, :n_load], columns=net.load.index)
    gen_p_df = pd.DataFrame(pp_gen_p, columns=net.gen.index)
    ConstControl(net, element="load", variable="p_mw", element_index=net.load.index,
                 data_source=DFData(load_p_df), profile_name=net.load.index)
    ConstControl(net, element="load", variable="q_mvar", element_index=net.load.index,
                 data_source=DFData(load_q_df), profile_name=net.load.index)
    ConstControl(net, element="gen", variable="p_mw", element_index=net.gen.index,
                 data_source=DFData(gen_p_df), profile_name=net.gen.index)
    if len(bus_sgen_idx):
        bus_sgen_p_df = pd.DataFrame(-pp_load_p[:, n_load:], columns=bus_sgen_idx)
        bus_sgen_q_df = pd.DataFrame(-pp_load_q[:, n_load:], columns=bus_sgen_idx)
        ConstControl(net, element="sgen", variable="p_mw", element_index=bus_sgen_idx,
                     data_source=DFData(bus_sgen_p_df), profile_name=bus_sgen_idx)
        ConstControl(net, element="sgen", variable="q_mvar", element_index=bus_sgen_idx,
                     data_source=DFData(bus_sgen_q_df), profile_name=bus_sgen_idx)

    ow = OutputWriter(net, output_path=None)
    ow.log_variable("res_bus", "vm_pu")
    if compute_voltages:
        ow.log_variable("res_bus", "va_degree")

    total_solver = [0.]

    def fun_ts(n, **kw):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pp.runpp(n, algorithm=PP_ALGO, init="results", tolerance_mva=tol_pf,
                                      check_connectivity=False, trafo_model=PP_TRAFO_MODEL,
                                      **pp_backend_kwargs, **kw)
        # see the "solver_time" comment in run_loop_mode(): net._ppc["et"] is pandapower's own
        # internal NR solve time, excluding the ConstControl/OutputWriter/pandas bookkeeping
        # that run_timeseries() adds on top and that `total_time` below includes. On
        # divergence pp.runpp() raises above and this line is simply never reached for that
        # time step.
        total_solver[0] += n._ppc["et"]

    beg = time.perf_counter()
    run_timeseries(
        net, time_steps=time_steps, continue_on_divergence=True, verbose=False,
        run=fun_ts,
    )
    end = time.perf_counter()

    vm_pu = ow.output["res_bus.vm_pu"]
    nb_solved = int(vm_pu.notna().all(axis=1).sum())

    voltages = None
    if compute_voltages:
        va_degree = ow.output["res_bus.va_degree"]
        voltages = vm_pu.to_numpy() * np.exp(1j * np.deg2rad(va_degree.to_numpy()))

    res = {
        "total_time": end - beg,
        "nb_solved": nb_solved,
        "solver_time": total_solver[0],
    }
    return res, voltages


def evaluate_and_record_kcl(res, kcl_checker, lsgrid, pp_bus_to_ls, voltages_pp_order, solved_mask,
                             load_p, load_q, gen_p):
    """Adds unsolved_scenarios / KCL summary keys to `res`, in place.

    voltages_pp_order is in net.bus POSITIONAL order (position i -> matpower bus
    pp_bus_to_ls[i], same convention get_bus_alignment()/get_pp_init_kwarg() already use)
    -- re-projected here into matpower bus order, which is what check_kcl_injection
    (and load_p/load_q/gen_p, straight from get_injection_from_base(), never reindexed
    through scenario_arrays()) expect. solved_mask is caller-provided because the two
    modes detect it differently: loop mode continues past a LoadflowNotConverged (so
    failures can be scattered, same as timeseries) and simply never writes that
    scenario's pre-allocated all-zero row; timeseries mode's continue_on_divergence=True
    likewise scatters failures, but OutputWriter fills those rows with NaN, not zero.
    """
    n_bus = lsgrid.total_bus()
    voltages_ls = np.zeros((voltages_pp_order.shape[0], n_bus), dtype=complex)
    voltages_ls[:, pp_bus_to_ls] = np.nan_to_num(voltages_pp_order)

    if solved_mask.sum() < solved_mask.shape[0]:
        res["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()

    kcl_mismatch = kcl_checker.check_kcl_injection(
        load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask], voltages_ls[solved_mask])
    res.update(summarize_kcl_mismatch(kcl_mismatch))
    res["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
    res.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))


if __name__ == "__main__":
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_flows = args.save_flows
    save_voltages = args.save_voltages
    evaluate_kcl = args.evaluate_kcl
    need_voltages = save_voltages or evaluate_kcl
    mode = args.mode
    modes = ["loop", "timeseries"] if mode == "both" else [mode]
    pp_backend = args.pp_backend
    pp_backend_kwargs = PP_BACKEND_KWARGS[pp_backend]
    nb_threads = int(args.nb_threads)
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]

    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total, pp_backend)
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_{{}}_Vs.npy"

    for fn in tqdm(grids):
        tmp_res_dict = {}
        this_path = os.path.join(REF_PATH, fn)
        mat_path = os.path.join(REF_PATH, fn.replace(".m", ".mat"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(this_path)
            net = from_mpc(mat_path, casename_mpc_file="mpc")
            cf = CaseFrames(this_path)
            bypass_lossy_trafo_conversion(net, cf)

        tmp_res_dict["grid_size"] = {
            "total_bus": int(len(net.bus)),
            "total_branch": int(len(net.line) + len(net.trafo) + len(net.impedance)),
        }

        try:
            pp_bus_to_ls, pp_row_to_ls = get_bus_alignment(lsgrid, net)
            gen_align_gen = get_gen_alignment(lsgrid, net)
            load_connected, load_align, n_load, bus_sgen_idx = get_load_alignment(
                lsgrid, net, pp_row_to_ls)
        except RuntimeError as exc:
            print(f"pp_injection: skipping {fn}, could not align elements ({exc})")
            continue

        init_method = get_init_method(fn)

        # lightsim2grid's own base powerflow must run first: get_injection_from_base() below
        # reads lsgrid.get_loads_res_full()/get_gen_res_full(), which are uninitialized until
        # a powerflow has been solved (same prerequisite as in ls_injection.py/olf_injection.py)
        v_init = get_ls_v_init(this_path, lsgrid, init_method, cf=cf)
        lsgrid.change_algorithm(PF_ALGO_LS)
        lsgrid.ac_pf(v_init, 10, TOL_PF)
        lsgrid.unset_changes()

        set_pp_slack_angle(net, v_init, pp_row_to_ls)
        pp_init = get_pp_init_kwarg(net, init_method, v_init, pp_bus_to_ls)

        # warm-up, then two timed powerflows (mirrors the init/following_powerflow split in
        # ls_injection.py; pandapower does not expose a factorize/refactorize breakdown so
        # only the wall-clock time is reported here). All 15 reference grids converge here
        # once seeded from lightsim2grid's own v_init via get_pp_init_kwarg() -- pandapower's
        # own native init="dc" used to fail on some of them (case6515rte/case_ACTIVSg10k),
        # landing at a worse starting point than lightsim2grid's dc_pf(). Kept as a try/except
        # regardless, in case a future grid/scenario genuinely doesn't converge.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pp.runpp(net, algorithm=PP_ALGO, init=pp_init, tolerance_mva=TOL_PF,
                        trafo_model=PP_TRAFO_MODEL, **pp_backend_kwargs)
                beg = time.perf_counter()
                pp.runpp(net, algorithm=PP_ALGO, init=pp_init, tolerance_mva=TOL_PF,
                        trafo_model=PP_TRAFO_MODEL, **pp_backend_kwargs)
                end = time.perf_counter()
                tmp_res_dict["init_powerflow"] = {"timer_total": end - beg, "solver_time": net._ppc["et"]}

                beg = time.perf_counter()
                pp.runpp(net, algorithm=PP_ALGO, init="results", tolerance_mva=TOL_PF,
                        trafo_model=PP_TRAFO_MODEL, **pp_backend_kwargs)
                end = time.perf_counter()
                tmp_res_dict["following_powerflow"] = {"timer_total": end - beg, "solver_time": net._ppc["et"]}
        except LoadflowNotConverged:
            print(f"pp_injection: skipping {fn}, base powerflow did not converge in pandapower")
            continue

        # exact same sampling call (same seed, same sample_data_meth, same base lsgrid values)
        # as ls_injection.py / olf_injection.py: the input scenarios are bit-for-bit identical
        load_p, load_q, gen_p = get_injection_from_base(
            fn, lsgrid, sample_meth=sample_data_meth, seed=seed, nb_pf_total=nb_pf_total)

        pp_load_p, pp_load_q, pp_gen_p = scenario_arrays(
            load_p, load_q, gen_p, load_connected, load_align, gen_align_gen)

        kcl_checker = KCLChecker(this_path) if evaluate_kcl else None

        if "loop" in modes:
            res_loop, voltages_loop = run_loop_mode_mp(
                net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, TOL_PF, need_voltages,
                pp_backend_kwargs, nb_threads)
            if evaluate_kcl:
                # a scenario that LoadflowNotConverged'd is simply never written, so it
                # stays at its pre-allocated all-zero value -- an unambiguous per-scenario
                # flag here (loop mode continues past divergence, see run_loop_mode)
                solved_mask_loop = np.abs(voltages_loop).sum(axis=1) > 0.
                evaluate_and_record_kcl(res_loop, kcl_checker, lsgrid, pp_bus_to_ls, voltages_loop,
                                         solved_mask_loop, load_p, load_q, gen_p)
            tmp_res_dict["loop"] = res_loop
            case_nm, _ = os.path.splitext(fn)
            if save_voltages:
                np.save(file=full_path_Vs.format(case_nm, "loop"), arr=voltages_loop)

        if "timeseries" in modes:
            # reload a fresh net for the timeseries run: run_loop_mode() above already
            # mutated net's load/gen/sgen tables and res_bus state. get_gen_alignment()'s
            # sgen->gen promotion must be redone on this fresh net too.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                net = from_mpc(mat_path, casename_mpc_file="mpc")
                bypass_lossy_trafo_conversion(net, cf)
            pp_bus_to_ls, pp_row_to_ls = get_bus_alignment(lsgrid, net)
            gen_align_gen = get_gen_alignment(lsgrid, net)
            load_connected, load_align, n_load, bus_sgen_idx = get_load_alignment(
                lsgrid, net, pp_row_to_ls)
            pp_load_p, pp_load_q, pp_gen_p = scenario_arrays(
                load_p, load_q, gen_p, load_connected, load_align, gen_align_gen)

            set_pp_slack_angle(net, v_init, pp_row_to_ls)
            pp_init = get_pp_init_kwarg(net, init_method, v_init, pp_bus_to_ls)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pp.runpp(net, algorithm=PP_ALGO, init=pp_init, tolerance_mva=TOL_PF,
                        trafo_model=PP_TRAFO_MODEL, **pp_backend_kwargs)
            res_ts, voltages_ts = run_timeseries_mode(
                net, pp_load_p, pp_load_q, pp_gen_p, n_load, bus_sgen_idx, TOL_PF, need_voltages,
                pp_backend_kwargs)
            if evaluate_kcl:
                # continue_on_divergence=True: failures can be scattered, and a failed
                # step's OutputWriter row is NaN, not zero (see nb_solved above)
                solved_mask_ts = ~np.isnan(voltages_ts).any(axis=1)
                evaluate_and_record_kcl(res_ts, kcl_checker, lsgrid, pp_bus_to_ls, voltages_ts,
                                         solved_mask_ts, load_p, load_q, gen_p)
            tmp_res_dict["timeseries"] = res_ts
            case_nm, _ = os.path.splitext(fn)
            if save_voltages:
                np.save(file=full_path_Vs.format(case_nm, "timeseries"), arr=voltages_ts)

        benchmark_results[fn] = tmp_res_dict
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)
