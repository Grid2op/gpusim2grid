# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
 
"""PF$\\Delta$ test set for case118 -- all 6
combos (N/N-1/N-2 x normal/nose) -- solved with pandapower's own pp.runpp(). 
This exercises pandapower's own alignment/bookkeeping (bus/gen/
load alignment, "pi" trafo/impedance modeling, its own results extraction) as an independent
check of the KCL mismatch reported directly by gpu_pfdelta.py, rather than a
timing benchmark.

Two complications, absent from pp_injection.py's own (fixed-topology, static-gen_v)
benchmarks, are specific to pfdelta:

1. Per-scenario topology. pp_injection.py's bypass_lossy_trafo_conversion() (see its own
   docstring) replaces every off-nominal-ratio net.trafo row with an exactly equivalent
   net.impedance row, built directly from the raw matpower pi-equivalent -- so after that
   call every one of case118's 186 matpower branches lives in net.line or net.impedance,
   never net.trafo. get_pp_branch_targets() below maps each of pfdelta's 186 branch_order
   columns to the (table, row) whose in_service flag models it, so run_loop_mode_pfdelta()
   can flip topology per scenario the same way pp_injection.py's run_loop_mode() flips
   load/gen setpoints per scenario -- confirmed exact (bus-for-bus, against topology.json)
   by _check_pp_branch_targets().

2. Per-scenario generator voltage. Like ls_pfdelta.py/gpu_pfdelta.py, this feeds pfdelta's
   own reported *solved* generator voltage magnitude (gen_v) as the per-row PV/slack target,
   so injections are self-consistent with pfdelta's own solution -- important for "nose"
   rows, whose reactive-limit effects push the real generator voltage far from nominal.
   pp_injection.py's get_gen_alignment() only aligns the PV ("gen") generators; the slack
   lives in net.ext_grid instead and is untouched by pp_injection.py's own benchmarks
   (which never vary gen_v). get_gen_v_targets() below fills in that missing ext_grid half.

3. Per-scenario generator outages, every combo. ls_pfdelta.py's load_pfdelta_rows() drops
   every row with a generator outage by default, because ScenarioSweepCPP has no way to
   take a per-row generator on/off mask (see its docstring) -- for the "nose" scenarios that
   drops roughly half the rows (99/200 for n-1, 135/200 for n-2), and for "normal" scenarios
   a majority of them (14808/29000 for n-1, 13550/20000 for n-2). pandapower has no such
   limitation: net.gen.in_service is just another per-scenario column, exactly like
   p_mw/vm_pu, so EVERY combo below (both "normal" and "nose") is loaded with
   `drop_gen_outage=False` and every row's gen_status is applied to net.gen.in_service in
   run_loop_mode_pfdelta(). One consequence: the "normal" combos' 2000-row subsample (see
   ls_pfdelta.py's RUNS) is now drawn from the FULL population (gen outages included), not
   the gen-outage-free subset ls_pfdelta.py/gpu_pfdelta.py draw theirs from -- so unlike
   before, this script's "normal" rows are no longer the same bit-for-bit scenarios as
   ls_pfdelta.py's/gpu_pfdelta.py's own "normal" rows (a deliberate tradeoff: broader,
   more representative coverage here, at the cost of a direct scenario-for-scenario
   comparison against those two for "normal"). The case118 pfdelta dataset never outages
   the slack unit (confirmed empirically across every topology/scenario/split), so
   net.ext_grid is never touched by this and kcl_checker.KCLChecker's fixed base_ref stays
   valid; check_kcl_scenario(..., gen_status=...) (see kcl_checker.py) dynamically
   reclassifies a generator's bus as PQ for exactly the scenarios where it's outaged,
   instead of leaving its reactive-power balance unchecked there -- the real-power balance
   check (which covers every non-slack bus, PV or PQ, regardless of generator status) was
   always fully rigorous either way.
"""

import copy
import json
import multiprocessing
import os
import time
import warnings

import numpy as np
from tqdm import tqdm

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.converter.matpower import from_mpc
from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames

from get_init_method import get_init_method
from kcl_checker import KCLChecker, summarize_kcl_mismatch, kcl_mismatch_per_scenario
from scenario_utils import (
    TOL_PF,
    REF_PATH,
    PATH_RESULTS,
    get_final_name
)

from pfdelta_utils import (
    TOPOLOGIES,
    RUNS,
    TOL_NOSE,
    MATPOWER_FN,
    PF_ALGO,
    load_pfdelta_rows,
    _load_branch_topology,
    _check_branch_order,
    _topology_dir,
)

from pp_injection import (
    PP_ALGO,
    PP_TRAFO_MODEL,
    PP_BACKEND_KWARGS,
    get_bus_alignment,
    get_gen_alignment,
    get_load_alignment,
    set_pp_slack_angle,
    bypass_lossy_trafo_conversion,
    get_ls_v_init,
    get_pp_init_kwarg,
    scenario_arrays,
    _scenario_chunks,
)


PF_ALGO = "NR_KLU"


def get_pp_branch_targets(net, impedance_offset):
    """For each of the 186 branch_order columns (mpc branch row order -- same as
    topology.json's own f_bus/t_bus/transformer arrays and kcl_checker.KCLChecker's
    self.caseframes.branch row order, confirmed identical, positionally, for case118), the
    (table, row) in `net` (AFTER bypass_lossy_trafo_conversion(net, cf) has run) whose
    in_service flag models that branch:

    - "line": net._from_ppc_lookups["branch"]'s own "element" column, unaffected by the
      trafo -> impedance conversion.
    - a PRE-EXISTING "impedance" row: also straight from that lookup, unaffected -- some
      off-nominal branches (eg phase shifters) are classified as "impedance" directly by
      pandapower's own from_mpc(), before bypass_lossy_trafo_conversion() ever runs.
    - a NEW "impedance" row, for a branch that WAS a "trafo": pp.create_impedances() (inside
      bypass_lossy_trafo_conversion()) appends rows in ascending mpc-row order, immediately
      after whatever pre-existing rows were already in net.impedance, so the k-th (0-based,
      in ascending column order) trafo-classified column lands at row
      `impedance_offset + k`. `impedance_offset` must be `len(net.impedance)` captured
      BEFORE bypass_lossy_trafo_conversion(net, cf) is called.

    Returns (line_cols, line_idx, imp_cols, imp_idx): line_cols/imp_cols are branch_order
    column indices, line_idx/imp_idx the corresponding net.line / net.impedance row index --
    same length and order on each side, ready for a positional net.line.loc[line_idx, ...] /
    net.impedance.loc[imp_idx, ...] assignment. Verified bus-for-bus by
    _check_pp_branch_targets() below.
    """
    branch_lookup = net._from_ppc_lookups["branch"]
    et = branch_lookup["element_type"]
    if (et == "trafo").any() and len(net.trafo) != 0:
        raise RuntimeError(
            "expected bypass_lossy_trafo_conversion() to have converted every 'trafo' "
            f"branch to 'impedance'; net.trafo still has {len(net.trafo)} row(s)"
        )

    line_cols = np.flatnonzero((et == "line").to_numpy())
    line_idx = branch_lookup.loc[line_cols, "element"].to_numpy(dtype=int)

    pre_imp_cols = np.flatnonzero((et == "impedance").to_numpy())
    pre_imp_idx = branch_lookup.loc[pre_imp_cols, "element"].to_numpy(dtype=int)

    trafo_cols = np.flatnonzero((et == "trafo").to_numpy())
    trafo_idx = impedance_offset + np.arange(trafo_cols.size)

    imp_cols = np.concatenate([pre_imp_cols, trafo_cols])
    imp_idx = np.concatenate([pre_imp_idx, trafo_idx])
    return line_cols, line_idx, imp_cols, imp_idx


def _check_pp_branch_targets(net, transformer, line_cols, line_idx, imp_cols, imp_idx):
    """Sanity check (raises on mismatch): confirms get_pp_branch_targets()'s column -> row
    mapping lands in the table topology.json's own "transformer" flag says it should (line
    iff not transformer[col]), and lines up bus-for-bus against topology.json's own
    f_bus/t_bus -- same check ls_pfdelta.py's _check_branch_order() does for lsgrid's
    get_lines()/get_trafos(), including reading f_bus/t_bus from the "n" topology folder
    only (physical bus pairs are a property of the branch, not the contingency scenario).
    """
    with open(os.path.join(_topology_dir("n"), "topology.json")) as f:
        topo = json.load(f)
    f_bus = np.asarray(topo["f_bus"])
    t_bus = np.asarray(topo["t_bus"])

    if set(line_cols.tolist()) != set(np.flatnonzero(~transformer).tolist()):
        raise RuntimeError("pp line columns do not match topology.json's non-transformer columns")
    if set(imp_cols.tolist()) != set(np.flatnonzero(transformer).tolist()):
        raise RuntimeError("pp impedance columns do not match topology.json's transformer columns")

    for col, row in zip(line_cols, line_idx):
        fb, tb = int(net.line.at[row, "from_bus"]), int(net.line.at[row, "to_bus"])
        if {fb + 1, tb + 1} != {int(f_bus[col]), int(t_bus[col])}:
            raise RuntimeError(f"pp/topology line bus mismatch at branch_order col {col}")
    for col, row in zip(imp_cols, imp_idx):
        fb, tb = int(net.impedance.at[row, "from_bus"]), int(net.impedance.at[row, "to_bus"])
        if {fb + 1, tb + 1} != {int(f_bus[col]), int(t_bus[col])}:
            raise RuntimeError(f"pp/topology impedance bus mismatch at branch_order col {col}")


def get_gen_v_targets(net):
    """Companion to pp_injection.py's get_gen_alignment(): that function returns, for each
    net.gen row (including its own gen-origin-sgen promotions), the corresponding lsgrid
    generator index -- but silently drops the slack, which pandapower keeps in net.ext_grid
    instead of net.gen. Returns (ext_grid_ls_idx, ext_grid_element): the lsgrid generator
    index of each slack unit (same ordinal space as gen_v's own columns, see
    ls_pfdelta.py's load_pfdelta_rows) and the net.ext_grid row it corresponds to. Needed
    because pfdelta's per-scenario gen_v carries the slack's own solved voltage too, and
    (unlike pp_injection.py's own benchmarks, which never vary gen_v) that matters here --
    see the module docstring.
    """
    gen_lookup = net._from_ppc_lookups["gen"]
    is_ext_grid = gen_lookup["element_type"] == "ext_grid"
    ext_grid_ls_idx = gen_lookup.index[is_ext_grid].to_numpy()
    ext_grid_element = gen_lookup.loc[is_ext_grid, "element"].to_numpy()
    return ext_grid_ls_idx, ext_grid_element


def run_loop_mode_pfdelta(net, pp_load_p, pp_load_q, pp_gen_p, pp_gen_v, pp_gen_status, ext_grid_element,
                           pp_gen_v_ext, n_load, bus_sgen_idx, line_idx, line_cols, imp_idx, imp_cols,
                           branch_status, base_vm_pu, base_va_degree, tol_pf, compute_voltages,
                           pp_backend_kwargs):
    """Like pp_injection.py's run_loop_mode(), plus per-scenario generator voltage
    (net.gen.vm_pu / net.ext_grid.vm_pu, from pp_gen_v / pp_gen_v_ext), per-scenario
    generator outages (net.gen.in_service, from pp_gen_status -- always all-ones for the
    "normal" combos, see the module docstring's point 3), and per-scenario topology
    (net.line.in_service / net.impedance.in_service, from branch_status via
    line_cols/line_idx/imp_cols/imp_idx -- see get_pp_branch_targets()).

    Unlike run_loop_mode() (whose init="results" warm-starts each scenario from the
    PREVIOUS scenario's own solution -- fine there, since only injections vary), every
    scenario here is reseeded from the fixed base-case solution (base_vm_pu/base_va_degree,
    net.res_bus right after the base powerflow) before calling pp.runpp(), still with
    init="results". This mirrors ScenarioSweepCPP/ScenarioSweepGPU's init_from_n_powerflow=
    True (see ls_pfdelta.py/gpu_pfdelta.py): pfdelta rows can carry wildly different
    topologies from one scenario to the next, and warm-starting NR from an unrelated,
    differently-topologied prior solution converges far worse than starting fresh from the
    base case every time -- confirmed empirically (n-1/n-2 convergence collapsed to ~15-40%
    with plain warm-starting, vs. the expected near-100% once reseeded here).
    """
    nb_scen = pp_load_p.shape[0]
    total_modif = 0.
    total_pf = 0.
    total_solver = 0.
    nb_solved = 0
    voltages = np.zeros((nb_scen, len(net.bus)), dtype=complex) if compute_voltages else None
    has_ext_grid_v = len(ext_grid_element) > 0
    has_imp = len(imp_idx) > 0
    for scen in range(nb_scen):
        beg_modif = time.perf_counter()
        net.res_bus["vm_pu"] = base_vm_pu
        net.res_bus["va_degree"] = base_va_degree
        net.load["p_mw"] = pp_load_p[scen, :n_load]
        net.load["q_mvar"] = pp_load_q[scen, :n_load]
        if len(bus_sgen_idx):
            net.sgen.loc[bus_sgen_idx, "p_mw"] = -pp_load_p[scen, n_load:]
            net.sgen.loc[bus_sgen_idx, "q_mvar"] = -pp_load_q[scen, n_load:]
        net.gen["p_mw"] = pp_gen_p[scen]
        net.gen["vm_pu"] = pp_gen_v[scen]
        net.gen["in_service"] = pp_gen_status[scen].astype(bool)
        if has_ext_grid_v:
            net.ext_grid.loc[ext_grid_element, "vm_pu"] = pp_gen_v_ext[scen]
        net.line.loc[line_idx, "in_service"] = branch_status[scen, line_cols].astype(bool)
        if has_imp:
            net.impedance.loc[imp_idx, "in_service"] = branch_status[scen, imp_cols].astype(bool)
        end_modif = time.perf_counter()
        total_modif += end_modif - beg_modif

        beg_pf = time.perf_counter()
        try:
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
        # see the "solver_time" comment in pp_injection.py's run_loop_mode()
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


def run_loop_mode_pfdelta_mp(net, pp_load_p, pp_load_q, pp_gen_p, pp_gen_v, pp_gen_status, ext_grid_element,
                              pp_gen_v_ext, n_load, bus_sgen_idx, line_idx, line_cols, imp_idx, imp_cols,
                              branch_status, base_vm_pu, base_va_degree, tol_pf, compute_voltages,
                              pp_backend_kwargs, nb_threads):
    """Multiprocessing dispatcher around run_loop_mode_pfdelta(), mirroring pp_injection.py's
    run_loop_mode_mp() (see its own docstring for why processes, not threads): each of
    nb_threads OS processes gets its own deep copy of `net` and independently runs
    run_loop_mode_pfdelta() over a contiguous slice of the (already row-concatenated, all 6
    combos) scenarios, then results are merged.
    """
    nb_scen = pp_load_p.shape[0]
    beg = time.perf_counter()
    if nb_threads <= 1:
        res, voltages = run_loop_mode_pfdelta(
            net, pp_load_p, pp_load_q, pp_gen_p, pp_gen_v, pp_gen_status, ext_grid_element, pp_gen_v_ext,
            n_load, bus_sgen_idx, line_idx, line_cols, imp_idx, imp_cols, branch_status,
            base_vm_pu, base_va_degree, tol_pf, compute_voltages, pp_backend_kwargs)
        res["wall_time"] = time.perf_counter() - beg
        res["nb_threads"] = 1
        return res, voltages

    chunks = _scenario_chunks(nb_scen, nb_threads)
    tasks = [
        (copy.deepcopy(net), pp_load_p[lo:hi], pp_load_q[lo:hi], pp_gen_p[lo:hi], pp_gen_v[lo:hi],
         pp_gen_status[lo:hi], ext_grid_element, pp_gen_v_ext[lo:hi], n_load, bus_sgen_idx, line_idx,
         line_cols, imp_idx, imp_cols, branch_status[lo:hi], base_vm_pu, base_va_degree, tol_pf,
         compute_voltages, pp_backend_kwargs)
        for lo, hi in chunks
    ]
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(run_loop_mode_pfdelta, tasks)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add_to_name", type=str, default="", help="Customize the name of the experiment")
    parser.add_argument("--save_voltages", action="store_true", help="Save the computed voltages (one .npy per combo)")
    parser.add_argument("--nb_threads", type=int, default=1,
                         help="Number of processes used for the loop-mode scenario sweep (see "
                              "pp_injection.py's run_loop_mode_mp: pandapower's runpp() is "
                              "pure-Python/GIL-bound, so this uses multiprocessing, not threads)")
    parser.add_argument("--tol", type=float, default=TOL_NOSE,
                         help=f"Shared pp.runpp() tolerance_mva across all 6 combos (default: "
                              f"{TOL_NOSE:g}, see ls_pfdelta.py's module docstring for why "
                              "'nose' rows need looser than TOL_PF)")
    parser.add_argument("--max_iteration", type=int, default=None,
                         help="pp.runpp()'s own max_iteration cap, shared across all 6 combos "
                              "(default: None, i.e. pandapower's own 'auto' default) -- for "
                              "comparing against gpu_pfdelta.py's --nb_iter sweep")
    args = parser.parse_args()
    nb_threads = int(args.nb_threads)
    pp_backend_kwargs = dict(PP_BACKEND_KWARGS["lightsim2grid"])
    if args.max_iteration is not None:
        # merged into pp_backend_kwargs (rather than threaded through as its own
        # parameter) since it's forwarded via **pp_backend_kwargs to every pp.runpp() call
        # below already -- both the warm-up call and run_loop_mode_pfdelta()'s per-scenario one
        pp_backend_kwargs["max_iteration"] = args.max_iteration

    nm_results = f"pp_pfdelta_{os.path.splitext(MATPOWER_FN)[0]}"
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, args.add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"

    this_path = os.path.join(REF_PATH, MATPOWER_FN)
    mat_path = os.path.join(REF_PATH, MATPOWER_FN.replace(".m", ".mat"))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lsgrid = init_from_matpower(this_path)
        net = from_mpc(mat_path, casename_mpc_file="mpc")
        cf = CaseFrames(this_path)
        # captured BEFORE the conversion -- see get_pp_branch_targets()'s docstring
        impedance_offset = len(net.impedance)
        bypass_lossy_trafo_conversion(net, cf)

    base_mva = lsgrid.get_sn_mva()

    pp_bus_to_ls, pp_row_to_ls = get_bus_alignment(lsgrid, net)
    gen_align_gen = get_gen_alignment(lsgrid, net)
    load_connected, load_align, n_load, bus_sgen_idx = get_load_alignment(lsgrid, net, pp_row_to_ls)
    ext_grid_ls_idx, ext_grid_element = get_gen_v_targets(net)
    line_cols, line_idx, imp_cols, imp_idx = get_pp_branch_targets(net, impedance_offset)

    init_method = get_init_method(this_path)
    v_init = get_ls_v_init(this_path, lsgrid, init_method, cf=cf)
    lsgrid.change_algorithm(PF_ALGO)
    res_base = lsgrid.ac_pf(v_init, 10, TOL_PF)
    if res_base.shape[0] == 0:
        raise RuntimeError(f"{MATPOWER_FN}: base powerflow did not converge")
    lsgrid.unset_changes()

    set_pp_slack_angle(net, v_init, pp_row_to_ls)
    pp_init = get_pp_init_kwarg(net, init_method, v_init, pp_bus_to_ls)
    # warm-up: primes net.res_bus so the loop below's init="results" has a valid starting
    # point on its very first scenario (mirrors pp_injection.py's own two-call calibration)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pp.runpp(net, algorithm=PP_ALGO, init=pp_init, tolerance_mva=TOL_PF,
                 trafo_model=PP_TRAFO_MODEL, **pp_backend_kwargs)
    # fixed base-case solution every scenario is reseeded from -- see
    # run_loop_mode_pfdelta()'s docstring for why (mirrors init_from_n_powerflow=True)
    base_vm_pu = net.res_bus["vm_pu"].to_numpy().copy()
    base_va_degree = net.res_bus["va_degree"].to_numpy().copy()

    kcl_checker = KCLChecker(this_path)
    n_bus_ls = lsgrid.total_bus()

    # Gather all 6 combos first (data loading only), remembering where each one lands in the
    # row-concatenated batch fed to the single loop-mode sweep below -- same pattern as
    # ls_pfdelta.py's own combo_data/combo_slices.
    combo_data = {}
    for topology in TOPOLOGIES:
        transformer = _load_branch_topology(topology)
        _check_branch_order(lsgrid, transformer)
        _check_pp_branch_targets(net, transformer, line_cols, line_idx, imp_cols, imp_idx)
        for run_name, scenario, split, subsample_size in RUNS:
            # every combo keeps every row (incl. generator outages -- net.gen.in_service
            # handles them, see the module docstring's point 3), unlike
            # ls_pfdelta.py/gpu_pfdelta.py's own gen-outage-free rows.
            data = load_pfdelta_rows(topology, scenario, split, base_mva, subsample_size=subsample_size,
                                      drop_gen_outage=False)
            if data is None:
                print(f"{topology}_{run_name}: no (scenario={scenario!r}, split={split!r}) rows, skipping")
                continue
            if (data["gen_status"][:, ext_grid_ls_idx] == 0).any():
                raise RuntimeError(
                    f"{topology}_{run_name}: found a scenario with the slack generator "
                    "outaged -- this script assumes (and case118's pfdelta dataset confirmed, "
                    "empirically, to never do) the slack is never outaged; net.ext_grid is "
                    "never toggled, see the module docstring's point 3")
            combo_data[(topology, run_name)] = data
    if not combo_data:
        raise RuntimeError("no pfdelta rows survived the filters for any (topology, run) combo")

    combo_keys = list(combo_data.keys())
    combo_slices = {}
    offset = 0
    for key in combo_keys:
        n_rows = combo_data[key]["load_p"].shape[0]
        combo_slices[key] = slice(offset, offset + n_rows)
        offset += n_rows

    load_p = np.concatenate([combo_data[k]["load_p"] for k in combo_keys], axis=0)
    load_q = np.concatenate([combo_data[k]["load_q"] for k in combo_keys], axis=0)
    gen_p = np.concatenate([combo_data[k]["gen_p"] for k in combo_keys], axis=0)
    gen_v = np.concatenate([combo_data[k]["gen_v"] for k in combo_keys], axis=0)
    gen_status = np.concatenate([combo_data[k]["gen_status"] for k in combo_keys], axis=0)
    branch_status = np.concatenate([combo_data[k]["branch_status"] for k in combo_keys], axis=0)

    pp_load_p, pp_load_q, pp_gen_p = scenario_arrays(
        load_p, load_q, gen_p, load_connected, load_align, gen_align_gen)
    pp_gen_v = gen_v[:, gen_align_gen]
    pp_gen_v_ext = gen_v[:, ext_grid_ls_idx]
    # trivially all-ones for the "normal" combos (drop_gen_outage=True there), real 0/1 per
    # scenario for "nose" -- see the module docstring's point 3
    pp_gen_status = gen_status[:, gen_align_gen]

    res_loop, voltages_pp_order = run_loop_mode_pfdelta_mp(
        net, pp_load_p, pp_load_q, pp_gen_p, pp_gen_v, pp_gen_status, ext_grid_element, pp_gen_v_ext,
        n_load, bus_sgen_idx, line_idx, line_cols, imp_idx, imp_cols, branch_status,
        base_vm_pu, base_va_degree, args.tol, True, pp_backend_kwargs, nb_threads)

    # loop mode continues past a LoadflowNotConverged (see run_loop_mode_pfdelta), so a
    # failed scenario simply never gets written and stays at its pre-allocated all-zero row
    # -- an unambiguous per-scenario flag, same convention pp_injection.py's own loop mode uses
    solved_mask_all = np.abs(voltages_pp_order).sum(axis=1) > 0.
    voltages_ls_all = np.zeros((offset, n_bus_ls), dtype=complex)
    voltages_ls_all[:, pp_bus_to_ls] = voltages_pp_order

    benchmark_results = {
        "tol": args.tol,
        "nb_rows_total": int(offset),
        "nb_converged_total": int(solved_mask_all.sum()),
        "total_time": res_loop["total_time"],
        "modif_time": res_loop["modif_time"],
        "solver_wall_time": res_loop["solver_wall_time"],
        "solver_time": res_loop["solver_time"],
        "wall_time": res_loop["wall_time"],
        "nb_threads": res_loop["nb_threads"],
    }
    for key in tqdm(combo_keys):
        topology, run_name = key
        result_key = f"{topology}_{run_name}"
        data = combo_data[key]
        sl = combo_slices[key]
        n_rows = sl.stop - sl.start
        solved_mask = solved_mask_all[sl]
        voltages_ls = voltages_ls_all[sl]

        tmp_res_dict = {
            "nb_rows": int(n_rows),
            "nb_converged": int(solved_mask.sum()),
        }
        if solved_mask.sum() < n_rows:
            tmp_res_dict["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()

        if solved_mask.any():
            # gen_status lets check_kcl_scenario() dynamically move a base_pv bus to the
            # reactive-balance check for any scenario where its generator(s) are outaged
            # (see kcl_checker.KCLChecker.check_kcl_scenario()'s docstring and the module
            # docstring's point 3) -- a no-op for the "normal" combos, where gen_status is
            # always all-ones (drop_gen_outage=True there).
            mismatch = kcl_checker.check_kcl_scenario(
                data["load_p"][solved_mask], data["load_q"][solved_mask], data["gen_p"][solved_mask],
                voltages_ls[solved_mask], data["branch_status"][solved_mask],
                gen_status=data["gen_status"][solved_mask])
            tmp_res_dict.update(summarize_kcl_mismatch(mismatch))
            tmp_res_dict.update(kcl_mismatch_per_scenario(mismatch, solved_mask))
        tmp_res_dict["sample_ids"] = data["sample_ids"].tolist()
        tmp_res_dict["lam"] = data["lam"].tolist()

        benchmark_results[result_key] = tmp_res_dict
        if args.save_voltages:
            np.save(file=full_path_Vs.format(result_key), arr=voltages_ls)

    with open(complete_full_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, fp=f, indent=2)
