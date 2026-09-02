# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PF$\\Delta$ test set for case118 -- all 6 combos
(N/N-1/N-2 x normal/nose) in one batch -- solved with gpusim2grid.ScenarioSweepGPU

ScenarioSweepGPU has no early-stop-on-convergence the way ScenarioSweepCPP.compute(...,
max_iter, tol) does: it always runs a fixed --nb_iter Newton-Raphson iterations for the
whole batch (uniform, for GPU throughput), then convergence is a post-hoc check
(`sweep.converged(tol)`) against the final residual -- so unlike the CPU port, the
tolerance here doesn't change how many iterations run, only which rows are reported as
converged/KCL-checked afterwards.

Per-scenario generator outages (every combo, unlike pp_pfdelta.py which keeps its
"normal" combos gen-outage-free). A ScenarioSweepGPU has no per-row generator on/off
primitive analogous to set_topology()'s per-row branch list: its PV/PQ/slack split is
fixed for the WHOLE batch, from lsgrid's own generator statuses at CONSTRUCTION time (see
set_gen_v()'s docstring: a disconnected generator's column is only "silently ignored" if
the grid already has it deactivated when the sweep is built). So gen-outage rows are
grouped by their exact (per-generator) on/off pattern; for each distinct pattern, its
offline generator(s) are actually lsgrid.deactivate_gen()'d, the base case is re-solved
(giving that pattern its own correct PV/PQ/slack split AND a warm start -- validated
empirically to matter: a flat start on some patterns fails to converge at all, while
warm-starting from the pattern's own re-solved base case converges every row), a fresh
ScenarioSweepGPU is built and run over just that pattern's rows, then the generator(s)
are lsgrid.reactivate_gen()'d before the next pattern. kcl_checker.check_kcl_scenario()'s
`gen_status` parameter then reports the reactive-balance check correctly for these rows
without any special-casing in the reporting loop below (same mechanism pp_pfdelta.py's
module docstring point 3 uses).

This is markedly more expensive than pp_pfdelta.py's equivalent (a per-row Python loop,
where a generator outage is just another per-scenario column): case118's "normal"/n-1
rows have 53 distinct single-generator-outage patterns, and "normal"/n-2 rows have
~1400 distinct (single- or double-generator) patterns even after the usual 2000-row
subsample -- each one needs its own CPU reconverge + GPU sweep construction, so a full
run does (subsampled) 1000+ extra sweep constructions on top of "nose"'s much smaller 47
(n-1) / 106 (n-2). Run every combo's gen-outage rows anyway (no cap), since the point is
completeness of the reported KCL mismatch, not matching pp_pfdelta.py's timing profile.
"""

import json
import os
import warnings

import numpy as np
from tqdm import tqdm
from lightsim2grid.network import init_from_matpower
from gpusim2grid import ScenarioSweepGPU, warmup

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
)
    
    
def _gpu_branch_ids(branch_status, transformer):
    """branch_ids_per_scenario for ScenarioSweepGPU.set_topology(): per row, the 0-based
    GPU branch ids (lines first, then trafos -- see set_topology()'s docstring) whose
    pfdelta br_status is 0, via the same branch_order -> (line rank | trafo rank) mapping
    ls_pfdelta.py's _check_branch_order() verifies against lsgrid's own
    get_lines()/get_trafos(). `transformer` is the same bool array _load_branch_topology()
    returns; `branch_status` is (n_rows, 186) in raw branch_order column order.
    """
    line_cols = np.flatnonzero(~transformer)
    trafo_cols = np.flatnonzero(transformer)
    gpu_id = np.empty(transformer.shape[0], dtype=int)
    gpu_id[line_cols] = np.arange(line_cols.size)
    gpu_id[trafo_cols] = line_cols.size + np.arange(trafo_cols.size)
    return [gpu_id[np.flatnonzero(row == 0)].tolist() for row in branch_status]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add_to_name", type=str, default="", help="Customize the name of the experiment")
    parser.add_argument("--save_voltages", action="store_true", help="Save the computed voltages (one .npy per combo)")
    parser.add_argument("--device", type=int, default=0, help="On which device (GPU) to perform the computations")
    parser.add_argument("--batch_size", type=int, default=2048, help="GPU batch chunk size for compute()")
    parser.add_argument("--nb_iter", type=int, default=100,
                         help="Fixed Newton-Raphson iteration count for the whole batch (no "
                              "early stop on GPU, unlike ScenarioSweepCPP's max_iter+tol) (default: 100)")
    parser.add_argument("--tol", type=float, default=TOL_NOSE,
                         help=f"Shared post-hoc convergence check across all 6 combos, applied "
                              f"via sweep.converged(tol) after compute() (default: {TOL_NOSE:g}, "
                              "see ls_pfdelta.py's module docstring for why)")
    parser.add_argument("--no_subsample", action="store_true",
                         help="Ignore each RUN's subsample_size (see ls_pfdelta.py's RUNS) and "
                              "use every row surviving the (scenario, split) filter instead of "
                              "just the usual 2000-per-topology 'normal' sample -- KCL is then "
                              "reported over the full pfdelta test set, not a subsample of it")
    args = parser.parse_args()

    nm_results = f"gpu_pfdelta_{os.path.splitext(MATPOWER_FN)[0]}"
    complete_full_path = get_final_name(PATH_RESULTS, nm_results, args.add_to_name)
    nm_tmp, _ = os.path.splitext(complete_full_path)
    full_path_Vs = f"{nm_tmp}_{{}}_Vs.npy"

    print(f"GPU warm-up (CUDA/cuSPARSE/cuDSS init): {warmup(args.device):.1f} ms")

    path = os.path.join(REF_PATH, MATPOWER_FN)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        lsgrid = init_from_matpower(path)
    lsgrid.change_algorithm(PF_ALGO)
    base_mva = lsgrid.get_sn_mva()

    init_method = get_init_method(path)
    if init_method == "flat":
        v_init = np.ones(lsgrid.total_bus(), dtype=complex)
    elif init_method == "dc":
        v_init_dc = np.ones(lsgrid.total_bus(), dtype=complex)
        v_init = lsgrid.dc_pf(v_init_dc, 10, 1e-6)
    else:
        raise RuntimeError(f"{MATPOWER_FN}: unsupported init method {init_method!r} for gpu_pfdelta.py")

    res_base = lsgrid.ac_pf(v_init, 10, TOL_PF)
    if res_base.shape[0] == 0:
        raise RuntimeError(f"{MATPOWER_FN}: base powerflow did not converge")
    lsgrid.unset_changes()

    kcl_checker = KCLChecker(path)

    # Same data-gathering as ls_pfdelta.py's load_pfdelta_rows() (reused directly), plus
    # the GPU-specific branch_status -> branch_ids_per_scenario translation for
    # set_topology(). Unlike ls_pfdelta.py's own callers, drop_gen_outage=False here (see
    # the module docstring).
    combo_data = {}
    for topology in TOPOLOGIES:
        transformer = _load_branch_topology(topology)
        _check_branch_order(lsgrid, transformer)
        for run_name, scenario, split, subsample_size in RUNS:
            if args.no_subsample:
                subsample_size = None
            # Every combo keeps its generator-outage rows too (see the module docstring):
            # handled per-outage-pattern group below, not dropped like ls_pfdelta.py's
            # ScenarioSweepCPP-imposed default.
            data = load_pfdelta_rows(topology, scenario, split, base_mva, subsample_size=subsample_size,
                                      drop_gen_outage=False)
            if data is None:
                print(f"{topology}_{run_name}: no (scenario={scenario!r}, split={split!r}) rows, skipping")
                continue
            data["branch_ids"] = _gpu_branch_ids(data["branch_status"], transformer)
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
    branch_ids_per_scenario = [ids for k in combo_keys for ids in combo_data[k]["branch_ids"]]

    # A row with any generator outage can't share this sweep: ScenarioSweepGPU's PV/PQ/
    # slack split is fixed for the whole batch, from lsgrid's own (fully-connected)
    # generator statuses at construction time -- see the module docstring. Solved
    # separately, per outage pattern, below.
    no_gen_outage = (gen_status == 1).all(axis=1)
    row_idx_main = np.flatnonzero(no_gen_outage)
    row_idx_go = np.flatnonzero(~no_gen_outage)

    # ONE ScenarioSweepGPU call for every gen-outage-free row across all 6 combos -- same
    # rationale as ls_pfdelta.py's single ScenarioSweepCPP call: an "n" row's branch_ids
    # list is just empty.
    sweep = ScenarioSweepGPU(
        lsgrid,
        init_from_n_powerflow=True,
        tol_base=TOL_PF,
        device=args.device,
        nb_iter=args.nb_iter,
    )
    sweep.set_injections_from_elements(load_p[row_idx_main], load_q[row_idx_main], gen_p[row_idx_main])
    sweep.set_gen_v(gen_v[row_idx_main])
    sweep.set_topology([branch_ids_per_scenario[i] for i in row_idx_main])
    sweep.compute(batch_size=args.batch_size)

    n_bus = sweep.n_bus
    voltages_all = np.full((offset, n_bus), np.nan, dtype=complex)
    disconnected_all = np.ones(offset, dtype=bool)
    solved_mask_all = np.zeros(offset, dtype=bool)

    def _scatter_results(rows, sub_sweep):
        """Write one sweep's results into the shared full-size arrays at `rows`.
        get_disconnected() only flags topological islanding -- at --nb_iter as low as 5,
        NR can also diverge to a NaN/Inf voltage on a poorly-conditioned (topology,
        injection) row without being islanded at all, and that's just as unusable for a
        KCL check (check_kcl_scenario() would otherwise silently poison the whole combo's
        aggregate stats: np.nanmean/nanmax per row correctly skip individual NaN columns,
        but a row with EVERY column NaN reduces to NaN itself, and summarize_kcl_mismatch's
        cross-scenario mean() is not NaN-safe). So a non-finite voltage is treated as
        disconnected here too, on top of the solver's own flag.
        """
        v = sub_sweep.V_results.to_numpy().reshape(sub_sweep.n_scenarios, n_bus)
        voltages_all[rows] = v
        disc = (np.asarray(sub_sweep.get_disconnected()) != 0) | ~np.isfinite(v).all(axis=1)
        disconnected_all[rows] = disc
        solved_mask_all[rows] = sub_sweep.converged(tol=args.tol) & ~disc

    _scatter_results(row_idx_main, sweep)

    gen_outage_timings = []
    if row_idx_go.size:
        # Group the gen-outage rows by their exact per-generator on/off pattern (see the
        # module docstring): each group gets its own lsgrid.deactivate_gen() + base-case
        # reconverge + fresh ScenarioSweepGPU, then the generators are reactivated before
        # the next group.
        uniq_patterns, inv = np.unique(gen_status[row_idx_go], axis=0, return_inverse=True)
        print(f"gen-outage rows: {row_idx_go.size} total, {uniq_patterns.shape[0]} distinct "
              "outage pattern(s), solving one ScenarioSweepGPU per pattern")
        for g in tqdm(range(uniq_patterns.shape[0])):
            rows = row_idx_go[inv == g]
            offline_gen_ids = np.flatnonzero(uniq_patterns[g] == 0)

            for gid in offline_gen_ids:
                lsgrid.deactivate_gen(int(gid))
            res_g = lsgrid.ac_pf(v_init, 10, TOL_PF)
            if res_g.shape[0] == 0:
                tqdm.write(f"  outage pattern (gen ids {offline_gen_ids.tolist()}): base "
                           f"powerflow did not converge, marking its {rows.size} row(s) disconnected")
                for gid in offline_gen_ids:
                    lsgrid.reactivate_gen(int(gid))
                continue
            lsgrid.unset_changes()

            sweep_g = ScenarioSweepGPU(
                lsgrid,
                init_from_n_powerflow=True,
                tol_base=TOL_PF,
                device=args.device,
                nb_iter=args.nb_iter,
            )
            sweep_g.set_injections_from_elements(load_p[rows], load_q[rows], gen_p[rows])
            sweep_g.set_gen_v(gen_v[rows])
            sweep_g.set_topology([branch_ids_per_scenario[i] for i in rows])
            sweep_g.compute(batch_size=args.batch_size)
            _scatter_results(rows, sweep_g)
            gen_outage_timings.append(sweep_g.timings.to_dict())

            for gid in offline_gen_ids:
                lsgrid.reactivate_gen(int(gid))
            lsgrid.unset_changes()

    benchmark_results = {
        "tol": args.tol,
        "nb_iter": args.nb_iter,
        "nb_rows_total": int(offset),
        "nb_converged_total": int(solved_mask_all.sum()),
        "nb_disconnected_total": int(disconnected_all.sum()),
        "nb_gen_outage_rows_total": int(row_idx_go.size),
        "timings": sweep.timings.to_dict(),
        "gen_outage_timings": gen_outage_timings,
    }
    for key in combo_keys:
        topology, run_name = key
        result_key = f"{topology}_{run_name}"
        data = combo_data[key]
        sl = combo_slices[key]
        n_rows = sl.stop - sl.start
        solved_mask = solved_mask_all[sl]
        disconnected = disconnected_all[sl]
        voltages = voltages_all[sl]

        tmp_res_dict = {
            "nb_rows": int(n_rows),
            "nb_converged": int(solved_mask.sum()),
            "nb_disconnected": int(disconnected.sum()),
        }
        if solved_mask.sum() < n_rows:
            tmp_res_dict["unsolved_scenarios"] = np.flatnonzero(~solved_mask).tolist()

        # KCL is reported for every row the solver actually attempted (converged at `tol`
        # or not) -- only a truly disconnected/islanded row (voltage undefined) is dropped.
        # A row below `tol` still has a small, informative residual (see ls_pfdelta.py's
        # module docstring): NR ran the full --nb_iter, it just didn't cross the threshold.
        kcl_eligible = ~disconnected
        if kcl_eligible.any():
            # gen_status lets check_kcl_scenario() dynamically move a base_pv bus to the
            # reactive-balance check for any row where its generator(s) are outaged (see
            # its own docstring) -- a no-op for a row with no generator outage, where
            # gen_status is all-ones.
            mismatch = kcl_checker.check_kcl_scenario(
                data["load_p"][kcl_eligible], data["load_q"][kcl_eligible], data["gen_p"][kcl_eligible],
                voltages[kcl_eligible], data["branch_status"][kcl_eligible],
                gen_status=data["gen_status"][kcl_eligible])
            tmp_res_dict.update(summarize_kcl_mismatch(mismatch))
            tmp_res_dict.update(kcl_mismatch_per_scenario(mismatch, kcl_eligible))
            # kcl_mismatch_per_scenario's own "converged_mask" key reflects kcl_eligible
            # (which rows have a KCL entry at all), not `tol`-convergence -- overwrite it
            # with the real, tol-based flag so it isn't misread as "solver converged".
            tmp_res_dict["converged_mask"] = [bool(x) for x in solved_mask]
        tmp_res_dict["sample_ids"] = data["sample_ids"].tolist()
        tmp_res_dict["lam"] = data["lam"].tolist()

        benchmark_results[result_key] = tmp_res_dict
        if args.save_voltages:
            np.save(file=full_path_Vs.format(result_key), arr=voltages)

    with open(complete_full_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, fp=f, indent=2)
