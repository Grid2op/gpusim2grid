# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import os
import json

PF_ALGO = "NR_KLU"

# root of the *full* PF$\Delta$ dataset (n / n-1 / n-2, with contingency.npz), as
# extracted on disk -- distinct from, and not to be confused with,
# generate_data_from_matpower.py's PFDELTA_ROOT (matpower_injection_data/pfdelta_n),
# which only has the unperturbed "n" topology and is used by ls_injection.py's
# "pfdelta_n" sample method. Relative to this file's directory (src/synthetic_data),
# like REF_PATH / PATH_RESULTS in ls_utils.py -- ../../../pfdelta/data is a sibling of
# the expe_gpus repo root, not inside it.
PFDELTA_ROOT = "../../../pfdelta/data"
PFDELTA_CASE_DIR = "case118_all"
MATPOWER_FN = "case118.m"
TOPOLOGIES = ["n", "n-1", "n-2"]

# default shared tol for the single combined ScenarioSweepCPP.compute() call -- looser
# than TOL_PF so "nose" rows (see the module docstring) converge too; overridable via --tol
TOL_NOSE = 1e-4

# (run name, scenario, split, subsample size per topology or None for "take every row
# that survives the no-gen-outage filter") -- sizes match the PF$\Delta$ paper's
# per-topology test-set convention, see the module docstring.
RUNS = [
    ("normal", "simple", "none", 2000),
    ("nose", "nose", "test", None),
]
SUBSAMPLE_SEED = 0


def _topology_dir(topology):
    return os.path.join(PFDELTA_ROOT, PFDELTA_CASE_DIR, topology)


def _load_branch_topology(topology):
    """(f_bus, t_bus, is_transformer) for every one of the 186 branch_order columns of
    this topology's contingency.npz / samples.npz, straight from its topology.json.
    Column i of br_status is a line iff not is_transformer[i], and its position among
    lightsim2grid's own get_lines() (resp. get_trafos()) is its rank among the False
    (resp. True) entries of is_transformer up to and including i -- verified once here
    by cross-checking (f_bus, t_bus) against `lsgrid`'s own lines/trafos, in order.
    """
    with open(os.path.join(_topology_dir(topology), "topology.json")) as f:
        topo = json.load(f)
    return np.asarray(topo["transformer"], dtype=bool)


def _check_branch_order(lsgrid, transformer):
    """Sanity check (raises on mismatch): confirms that selecting br_status's non
    transformer / transformer columns, in order, lines up 1:1 with
    lsgrid.get_lines() / lsgrid.get_trafos(), in order -- see the module docstring.
    """
    with open(os.path.join(_topology_dir("n"), "topology.json")) as f:
        topo = json.load(f)
    f_bus = np.asarray(topo["f_bus"])
    t_bus = np.asarray(topo["t_bus"])

    line_cols = np.flatnonzero(~transformer)
    trafo_cols = np.flatnonzero(transformer)
    lines = lsgrid.get_lines()
    trafos = lsgrid.get_trafos()
    if len(lines) != line_cols.size or len(trafos) != trafo_cols.size:
        raise RuntimeError(
            f"pfdelta branch count mismatch: {line_cols.size} lines / {trafo_cols.size} "
            f"trafos in topology.json, lsgrid has {len(lines)} lines / {len(trafos)} trafos"
        )
    for rank, col in enumerate(line_cols):
        l = lines[rank]
        if {l.bus1_id + 1, l.bus2_id + 1} != {int(f_bus[col]), int(t_bus[col])}:
            raise RuntimeError(f"pfdelta/lsgrid line order mismatch at rank {rank} (branch_order col {col})")
    for rank, col in enumerate(trafo_cols):
        t = trafos[rank]
        if {t.bus1_id + 1, t.bus2_id + 1} != {int(f_bus[col]), int(t_bus[col])}:
            raise RuntimeError(f"pfdelta/lsgrid trafo order mismatch at rank {rank} (branch_order col {col})")


def load_pfdelta_rows(topology, scenario, split, base_mva, subsample_size=None, seed=SUBSAMPLE_SEED,
                       drop_gen_outage=True):
    """Rows matching (scenario, split) for `topology` ("n", "n-1" or "n-2"), ready to feed a
    ScenarioSweepCPP: load_p/load_q/gen_p in MW/MVAr, gen_v in pu (the *solved* voltage
    magnitude at each generator bus, as reported by pfdelta -- same convention as
    generate_data_from_matpower.get_loads_gens_pfdelta_n()), branch_status in raw
    branch_order column order (for kcl_checker.check_kcl_scenario), and line_status/
    trafo_status split out (for ScenarioSweepCPP.set_contingency_lines/trafos).

    By default (`drop_gen_outage=True`, this module's own callers below always use this
    default) rows with ANY generator outage are dropped entirely: ScenarioSweepCPP has no
    way to take a per-row generator on/off mask, so a dropped row's contingency could only
    be reflected as gen_p=0 for the outaged unit -- leaving it, wrongly, still
    voltage-controlling that bus (lightsim2grid's ref/PV/PQ bus split is fixed from the
    case file's own generator statuses, see kcl_checker.KCLChecker.base_ref/base_pv/
    base_pq). Pass `drop_gen_outage=False` for a caller that CAN model a per-row generator
    outage properly (eg pandapower's net.gen.in_service, toggled per scenario in a loop --
    see pp_pfdelta.py) to keep every row instead. Either way, gen_status (raw 0/1, same
    generator-column order as gen_p/gen_v) is always returned, so a `drop_gen_outage=False`
    caller can act on it; pg_sol is already exactly 0 for every outaged unit regardless
    (confirmed empirically), so the P-balance side of a KCL check is correct whether or not
    the caller does anything special with gen_status -- only the generator's own bus's PV/PQ
    classification (and, for a caller that actually simulates it, whether that bus's
    voltage is truly free to float) depends on it.

    If `subsample_size` is given and more rows survive the filter than that, a fixed,
    seeded random subsample of that size is drawn (see the module docstring for why).
    Returns None if no row survives the filter.
    """
    d = np.load(os.path.join(_topology_dir(topology), "samples.npz"), allow_pickle=True)
    c = np.load(os.path.join(_topology_dir(topology), "contingency.npz"), allow_pickle=True)
    if not np.array_equal(d["sample_ids"], c["sample_ids"]):
        raise RuntimeError(f"{topology}: samples.npz / contingency.npz sample_ids do not align")

    mask = (d["scenario"] == scenario) & (d["split"] == split)
    mask_idx = np.flatnonzero(mask)
    if drop_gen_outage:
        no_gen_outage = (c["gen_status"][mask] == 1).all(axis=1)
        mask_idx = mask_idx[no_gen_outage]
    if mask_idx.size == 0:
        return None
    if subsample_size is not None and mask_idx.size > subsample_size:
        rng = np.random.default_rng(seed)
        mask_idx = np.sort(rng.choice(mask_idx, size=subsample_size, replace=False))

    transformer = _load_branch_topology(topology)
    branch_status = c["br_status"][mask_idx]
    return {
        "sample_ids": d["sample_ids"][mask_idx],
        "lam": d["lam"][mask_idx],
        "load_p": d["pd"][mask_idx] * base_mva,
        "load_q": d["qd"][mask_idx] * base_mva,
        "gen_p": d["pg_sol"][mask_idx] * base_mva,
        "gen_v": d["gen_v"][mask_idx],
        "gen_status": c["gen_status"][mask_idx],
        "branch_status": branch_status,
        "line_status": branch_status[:, ~transformer],
        "trafo_status": branch_status[:, transformer],
    }
