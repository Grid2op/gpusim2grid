# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""power-grid-model (PGM, https://power-grid-model.readthedocs.io/) as a 6th benchmark
backend: the injection sweep itself, using PGM's own Batch Calculation feature
(`PowerGridModel.calculate_power_flow(update_data=...)`) rather than a Python loop of
single-scenario solves.

Prerequisite: We first validate on the initial (not perturbed) grid suites that PGM converges
and that its results can be "aligned" with matpower data. It was the case for 
5 of the 15 grids in scenario_utils.py. 
`PGM_OK_GRIDS` below is exactly that
5-grid list; this script does not attempt the other 10 by default.
"""

import json
import os
import time
import warnings

import numpy as np
from matpowercaseframes import CaseFrames
from pandapower.converter.matpower import from_mpc
from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter

from power_grid_model import (
    CalculationMethod,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array
)
from tqdm import tqdm

from get_init_method import get_init_method
from lightsim2grid.network import init_from_matpower
from kcl_checker import (
    KCLChecker,
    summarize_kcl_mismatch,
    summarize_kcl_mismatch_checkpoints,
    kcl_mismatch_per_scenario,
)
from scenario_utils import (
    REF_PATH,
    TOL_PF,
    PATH_RESULTS,
    get_final_name,
    get_default_args,
    get_injection_from_base,
)
from pp_injection import (
    get_bus_alignment,
    get_gen_alignment,
    get_load_alignment,
    get_ls_v_init,
    set_pp_slack_angle,
    scenario_arrays,
)

# PGM's default source impedance (sk=1e10 VA) models a finite-strength grid, appropriate for
# distribution studies; matpower's slack is an ideal infinite bus. This is how close to
# "ideal" the override below pushes it -- large enough that the residual is well below TOL_PF.
IDEAL_SOURCE_SK = 1e15


def build_pgm_input(net, cf):
    """Convert `net` (a pandapower net built by `from_mpc`, already `set_pp_slack_angle`-fixed)
    into PGM input_data, plus the pp-bus-row -> lightsim2grid-bus-id array needed to compare
    the result against `lsgrid`'s own solve. See the module docstring for the full rationale.
    """
    branch_lookup = net._from_ppc_lookups["branch"]
    is_bypass = branch_lookup["element_type"].isin(["trafo", "impedance"])
    mpc_rows = branch_lookup.index[is_bypass].to_numpy()
    bypass_kind = branch_lookup.loc[is_bypass, "element_type"].to_numpy()
    bypass_idx = branch_lookup.loc[is_bypass, "element"].astype(int).to_numpy()

    br = cf.branch.iloc[mpc_rows]
    r = br["BR_R"].to_numpy(dtype=float)
    x = br["BR_X"].to_numpy(dtype=float)
    b = br["BR_B"].to_numpy(dtype=float)
    ratio = br["TAP"].to_numpy(dtype=float)
    ratio = np.where(ratio == 0., 1., ratio)
    shift = br["SHIFT"].to_numpy(dtype=float)
    from_bus = br["F_BUS"].to_numpy(dtype=int) - 1
    to_bus = br["T_BUS"].to_numpy(dtype=int) - 1

    in_service = np.ones(len(mpc_rows), dtype=bool)
    is_trafo_row = bypass_kind == "trafo"
    is_imped_row = bypass_kind == "impedance"
    if is_trafo_row.any():
        in_service[is_trafo_row] = net.trafo.loc[bypass_idx[is_trafo_row], "in_service"].to_numpy()
    if is_imped_row.any():
        in_service[is_imped_row] = net.impedance.loc[bypass_idx[is_imped_row], "in_service"].to_numpy()

    # zbase referenced to the "to" side -- see module docstring
    vn_kv_to = net.bus.loc[to_bus, "vn_kv"].to_numpy()
    zbase = vn_kv_to ** 2 / net.sn_mva

    net.trafo.drop(index=bypass_idx[is_trafo_row], inplace=True)
    net.impedance.drop(index=bypass_idx[is_imped_row], inplace=True)

    conv = PandaPowerConverter()
    input_data, extra_info = conv.load_input_data(net)

    input_data["source"] = input_data["source"].copy()
    input_data["source"]["sk"] = IDEAL_SOURCE_SK
    input_data["source"]["rx_ratio"] = 0.0

    pp_bus_to_pgmid = {}
    for pid, info in extra_info.items():
        ref = info.get("id_reference") if isinstance(info, dict) else None
        if ref is not None and str(ref.get("table")).split(".")[-1] == "bus":
            pp_bus_to_pgmid[ref["index"]] = int(pid)

    if "voltage_regulator" in input_data and "source" in input_data:
        src_nodes = set(input_data["source"]["node"].tolist())
        vr = input_data["voltage_regulator"]
        gen_node = dict(zip(input_data["sym_gen"]["id"].tolist(), input_data["sym_gen"]["node"].tolist()))
        keep = np.array([gen_node.get(int(rid), -1) not in src_nodes for rid in vr["regulated_object"]])
        input_data["voltage_regulator"] = vr[keep]

    if "voltage_regulator" in input_data:
        # PGM's own voltage_regulator (and, per its docs, "source" too) can be Q-limited: once
        # a regulated unit would need more/less reactive power than [q_min, q_max] to hold its
        # voltage target, PGM stops regulating it (drops it off "PV"). PV/PQ status here should
        # instead come straight from the matpower input (which bus type each generator is), the
        # same way lightsim2grid treats it for this comparison -- not be re-derived from Q
        # limits mid-solve. The installed power_grid_model_io (1.3.108) already leaves q_min/
        # q_max as NaN unconditionally (never reads net.gen's min_q_mvar/max_q_mvar at all --
        # see PandaPowerConverter._create_pgm_input_voltage_regulators), so this is currently a
        # no-op in practice; set explicitly anyway so a converter upgrade that starts wiring
        # real Q-limits through doesn't silently change this benchmark's PV/PQ assumption.
        input_data["voltage_regulator"] = input_data["voltage_regulator"].copy()
        input_data["voltage_regulator"]["q_min"] = -1e30
        input_data["voltage_regulator"]["q_max"] = 1e30

    if len(mpc_rows):
        gb = initialize_array(DatasetType.input, ComponentType.generic_branch, len(mpc_rows))
        gb["id"] = conv._generate_ids("trafo_gb", mpc_rows)
        gb["from_node"] = [pp_bus_to_pgmid[b_] for b_ in from_bus]
        gb["to_node"] = [pp_bus_to_pgmid[b_] for b_ in to_bus]
        gb["from_status"] = in_service
        gb["to_status"] = in_service
        gb["r1"] = r * zbase
        gb["x1"] = x * zbase
        gb["g1"] = 0.0
        gb["b1"] = b / zbase
        gb["k"] = ratio
        gb["theta"] = np.deg2rad(shift)
        gb["sn"] = net.sn_mva * 1e6
        input_data["generic_branch"] = gb

    id_to_pp_bus = {v: k for k, v in pp_bus_to_pgmid.items()}
    return input_data, id_to_pp_bus, extra_info


PF_ALGO_LS = "NR_KLU"
BASE_EXPE_NAME = "pgm_injection_{}_{}"

# See module docstring: the other 10 canonical grids fail PGM's own base-case AC powerflow
# (pgm_checkpoint.py) and are excluded here until that is root-caused.
PGM_OK_GRIDS = ["case14.m", "case_ieee30.m", "case57.m", "case118.m", "case_ACTIVSg500.m"]


def table_id_map(extra_info, table_name, name_filter):
    """PGM's sym_load holds a ZIP split (3 rows per pandapower load row: const_power/
    const_impedance/const_current -- only "const_power" is non-zero here, pandapower's
    from_mpc() never sets const_z_percent/const_i_percent), and voltage_regulator shares
    the "gen" id_reference table with sym_gen (distinguished only by name: "gen" vs
    "regulator"). So `table_name` alone does not uniquely key a pandapower row --
    `name_filter(name) -> bool` disambiguates. Returns {pandapower row index: pgm id}.
    """
    m = {}
    for pid, info in extra_info.items():
        ref = info.get("id_reference") if isinstance(info, dict) else None
        if ref is not None and str(ref.get("table")).split(".")[-1] == table_name and name_filter(ref.get("name")):
            m[ref["index"]] = int(pid)
    return m


def get_args():
    parser = get_default_args()
    parser.add_argument("--grids", type=str, default=",".join(PGM_OK_GRIDS),
                         help="comma-separated matpower filenames to benchmark "
                              f"(default: the {len(PGM_OK_GRIDS)} grids validated in pgm_checkpoint.py)")
    parser.add_argument('--nb_threads', type=int, default=1, 
                        help=f"Number of threads used for making the computation default to {1}") 
    return parser


def run_one(fn, nb_pf, sample_data_meth, seed, save_voltages,
            nb_threads, evaluate_kcl=False):
    this_path = os.path.join(REF_PATH, fn)
    mat_path = os.path.join(REF_PATH, fn.replace(".m", ".mat"))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lsgrid = init_from_matpower(this_path)
        net = from_mpc(mat_path, casename_mpc_file="mpc")
        cf = CaseFrames(this_path)

    pp_bus_to_ls, pp_row_to_ls = get_bus_alignment(lsgrid, net)
    gen_align_gen = get_gen_alignment(lsgrid, net)
    load_connected, load_align, n_load, bus_sgen_idx = get_load_alignment(lsgrid, net, pp_row_to_ls)

    v_init = get_ls_v_init(this_path, lsgrid, get_init_method(fn), cf=cf)
    set_pp_slack_angle(net, v_init, pp_row_to_ls)

    # get_injection_from_base() reads lsgrid.get_loads_res_full() / get_gen_res_full(), which
    # are only valid once lsgrid has a solved base case -- without this, sampling silently
    # reads whatever was last in that memory (garbage on a freshly-built lsgrid), which in turn
    # feeds PGM nonsense injections and can look like a PGM solver failure (singular matrix /
    # non-convergence) that is actually just this missing base solve.
    lsgrid.change_algorithm(PF_ALGO_LS)
    if lsgrid.ac_pf(v_init, 10, TOL_PF).shape[0] == 0:
        raise RuntimeError(f"{fn}: lightsim2grid base powerflow did not converge")
    lsgrid.unset_changes()

    input_data, id_to_pp_bus, extra_info = build_pgm_input(net, cf)
    load_id_map = table_id_map(extra_info, "load", lambda n: n == "const_power")
    sgen_id_map = table_id_map(extra_info, "sgen", lambda n: n != "regulator")
    gen_id_map = table_id_map(extra_info, "gen", lambda n: n == "gen")

    load_p, load_q, gen_p = get_injection_from_base(
        fn, lsgrid, sample_meth=sample_data_meth, seed=seed, nb_pf_total=nb_pf)
    pp_load_p, pp_load_q, pp_gen_p = scenario_arrays(
        load_p, load_q, gen_p, load_connected, load_align, gen_align_gen)

    load_ids = np.array([load_id_map[i] for i in net.load.index])
    gen_ids = np.array([gen_id_map[i] for i in net.gen.index])
    sgen_ids = np.array([sgen_id_map[i] for i in bus_sgen_idx])

    upd_load = initialize_array(DatasetType.update, ComponentType.sym_load, (nb_pf, len(load_ids)))
    upd_load["id"] = load_ids
    upd_load["p_specified"] = pp_load_p[:, :n_load] * 1e6
    upd_load["q_specified"] = pp_load_q[:, :n_load] * 1e6

    n_gen = len(gen_ids)
    n_sgen = len(sgen_ids)
    upd_gen = initialize_array(DatasetType.update, ComponentType.sym_gen, (nb_pf, n_gen + n_sgen))
    upd_gen["id"] = np.concatenate([gen_ids, sgen_ids])
    upd_gen["p_specified"][:, :n_gen] = pp_gen_p * 1e6
    if n_sgen:
        # bus-origin sgen rows are how pp_injection.py's own alignment represents a
        # negative-PD matpower bus (see get_load_alignment docstring); sign convention
        # is generation-positive throughout (net.sgen.p_mw, PGM sym_gen.p_specified alike).
        upd_gen["p_specified"][:, n_gen:] = -pp_load_p[:, n_load:] * 1e6
        upd_gen["q_specified"][:, n_gen:] = -pp_load_q[:, n_load:] * 1e6

    update_data = {"sym_load": upd_load, "sym_gen": upd_gen}

    model = PowerGridModel(input_data)
    beg = time.perf_counter()
    result = model.calculate_power_flow(
        calculation_method=CalculationMethod.newton_raphson,
        symmetric=True,
        max_iterations=100, error_tolerance=TOL_PF,
        update_data=update_data,
        continue_on_batch_error=True,
        threading=nb_threads,
        # only node voltages are read below -- skip computing/packing branch and injection
        # flow output for every scenario (see PGM's performance guide: restricting
        # output_component_types avoids this otherwise-wasted work inside the timed call).
        output_component_types=[ComponentType.node])
    end = time.perf_counter()
    total_time = end - beg

    node_res = result["node"]
    # continue_on_batch_error=True zero-fills (rather than raising for) any scenario PGM's
    # own NR fails to converge on -- a real |u_pu|==0 pu result never occurs on a converged
    # bus, so this is an unambiguous per-scenario convergence flag.
    solved_mask = np.abs(node_res["u_pu"]).sum(axis=1) > 0.
    nb_solved = int(solved_mask.sum())

    res = {
        "total_time": total_time,
        "nb_solved": nb_solved,
        "n_bus": int(lsgrid.total_bus()),
        "solver_time_per_pf_ms": 1e3 * total_time / nb_pf,
        "pf_per_s": nb_pf / total_time,
    }
    if nb_solved < nb_pf:
        # scattered, not necessarily a trailing run -- see solved_mask comment above
        res["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()

    voltages = None
    if save_voltages or evaluate_kcl:
        pgm_bus_row = np.array([id_to_pp_bus[int(pid)] for pid in node_res["id"][0]])
        pgm_ls_bus = np.array([pp_row_to_ls[b] for b in pgm_bus_row])
        voltages = np.zeros((nb_pf, lsgrid.total_bus()), dtype=complex)
        voltages[:, pgm_ls_bus] = node_res["u_pu"] * np.exp(1j * node_res["u_angle"])

    if evaluate_kcl:
        # load_p/load_q/gen_p (unlike pp_load_p/pp_load_q/pp_gen_p above) are still in
        # matpower order -- exactly what check_kcl_injection expects, and
        # matching the lsgrid bus order `voltages` was just built in. solved_mask (not a
        # prefix slice) is required: continue_on_batch_error=True can zero-fill any
        # scattered scenario, not just a trailing run of them (see module docstring).
        kcl_checker = KCLChecker(this_path)
        kcl_mismatch = kcl_checker.check_kcl_injection(
            load_p[solved_mask], load_q[solved_mask], gen_p[solved_mask], voltages[solved_mask])
        res.update(summarize_kcl_mismatch(kcl_mismatch))
        res["kcl_mismatch_checkpoints"] = summarize_kcl_mismatch_checkpoints(kcl_mismatch)
        res.update(kcl_mismatch_per_scenario(kcl_mismatch, solved_mask))

    return res, voltages


if __name__ == "__main__":
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)
    save_voltages = args.save_voltages
    evaluate_kcl = args.evaluate_kcl
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    nb_threads = int(args.nb_threads)

    benchmark_results = {}
    nm_results = BASE_EXPE_NAME.format(sample_data_meth, nb_pf_total)
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"

    for fn in tqdm(grids):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res, voltages = run_one(fn, nb_pf_total, sample_data_meth, seed, save_voltages,
                                     nb_threads=nb_threads, evaluate_kcl=evaluate_kcl)
        if res["nb_solved"] != nb_pf_total:
            print(f"{fn}: {res['nb_solved']} vs {nb_pf_total} solved")
        benchmark_results[fn] = res
        if save_voltages:
            np.save(file=full_path_Vs.format(fn.replace(".m", "")), arr=voltages)
        with open(complete_full_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, fp=f, indent=2)
