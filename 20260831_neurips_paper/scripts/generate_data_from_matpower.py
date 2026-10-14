# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import numpy as np
from scipy.interpolate import interp1d
DEBUG = False

DEBUG_PF_DELTA = False

# root of the pre-extracted PF$\Delta$ dataset (https://arxiv.org/abs/2510.22048), see
# /home/donnotben/Documents/expe_gpus/matpower_injection_data/pfdelta_n/README.md
PFDELTA_ROOT = "../../matpower_injection_data/pfdelta_n"
# only the unperturbed ("N") topology is used here: get_injection_from_base() only ever
# varies loads/gens, never the topology, so it is the only variant that matches what
# ls_injection.py / olf_injection.py / pp_injection.py / export_exapf_data.py do for
# "fr"/"sable" too
PFDELTA_TOPOLOGY = "n"
# matpower file name -> pfdelta case directory name. Only cases confirmed to be the
# *same underlying grid* as the corresponding matpowerdata/*.m file are listed here.
# pfdelta's "500-bus"/"2000-bus" cases are the GOC-500/GOC-2000 grids (ARPA-E Grid
# Optimization Competition, pglib_opf_case500_goc.m / pglib_opf_case2000_goc.m from
# pglib-opf v23.07, https://github.com/power-grid-lib/pglib-opf) -- NOT PSS/E's
# case_ACTIVSg500/case_ACTIVSg2000 also present in matpowerdata/, which happen to have
# the same bus counts but different generator/load counts (90 vs 224 gens, 200 vs 281
# loads for "500"; 544 vs 384 gens, 1125 vs 1010 loads for "2000"). The GOC files
# themselves are copied into matpowerdata/ as case500_goc.m / case2000_goc.m (see
# get_init_method.py for their init methods); case_ACTIVSg500/2000 are deliberately NOT
# mapped here so they get dropped (see has_pfdelta_data()) rather than silently fed
# mismatched injections.
PFDELTA_CASE_DIRS = {
    "case14.m": "case14_all",
    "case_ieee30.m": "case30_all",
    "case57.m": "case57_all",
    "case118.m": "case118_all",
    "case500_goc.m": "case500_all",
    "case2000_goc.m": "case2000_all",
}


def get_pfdelta_samples_path(fn):
    """Path to the pfdelta samples.npz for matpower file `fn`, or None if pfdelta has no
    (matching) data for this grid."""
    case_dir = PFDELTA_CASE_DIRS.get(fn)
    if case_dir is None:
        return None
    path = os.path.join(PFDELTA_ROOT, case_dir, PFDELTA_TOPOLOGY, "samples.npz")
    return path if os.path.isfile(path) else None


def has_pfdelta_data(fn, sample_meth):
    """True unless `sample_meth` is "pfdelta_n" and no (matching) pfdelta data exists for
    `fn`. Callers should `continue` to the next grid when this is False, before doing
    any expensive per-grid setup."""
    return sample_meth != "pfdelta_n" or get_pfdelta_samples_path(fn) is not None


def get_loads_gens_pfdelta_n(fn, lsgrid, nb_pf_total=288):
    """Read pre-solved (Pd, Qd, Pg) scenarios straight from the PF$\\Delta$ dataset,
    instead of synthesizing them like get_loads_gens()/get_loads_gens_sable() do.

    Unlike the "fr"/"sable" methods, this is deterministic (no prng): pfdelta's samples
    were generated once and are just read back here, taking the first `nb_pf_total` of
    them (in the dataset's own, fixed, sample_ids order).

    Column order: pfdelta's `pd`/`qd`/`pg_sol` arrays and lightsim2grid's own load/gen
    order both come from a straight, in-order scan of the matpower file's bus/gen tables
    (loads = bus rows with nonzero Pd/Qd, in row order; gens = gen table rows, in row
    order) -- same reasoning as documented in export_exapf_data.py for ExaPF. Verified
    for case14/case_ieee30/case57/case118 (the four grids listed in PFDELTA_CASE_DIRS):
    load and generator counts match exactly, and feeding a sample's (pd, qd, pg_sol,
    vm-derived gen setpoints) back into lightsim2grid and calling lsgrid.ac_pf() with
    the sample's own (vm, va) as initial guess converges back to that same voltage to
    ~1e-7 pu -- confirming both the column order and the choice of `pg_sol` (see below).

    `pg` vs `pg_sol`: pfdelta stores both an input dispatch `pg` (the pre-solve target
    fed into its data-generation pipeline) and the actual solved dispatch `pg_sol` that
    balances against the reported (vm, va). The two can differ substantially per
    generator (including buses where `pg` is nonzero but `pg_sol` is ~0, e.g. a
    redispatched/curtailed unit). Only `pg_sol` is consistent with `vm`/`va`/`pd`/`qd`;
    using `pg` produces KCL residuals in the hundreds of MW and voltages far from the
    reported solution.
    """
    samples_path = get_pfdelta_samples_path(fn)
    if samples_path is None:
        raise ValueError(f"No pfdelta data available for {fn}")
    file_ = np.load(samples_path)
    base_mva = lsgrid.get_sn_mva()
    nb_load = len(lsgrid.get_loads())
    nb_gen = len(lsgrid.get_generators())
    if file_["pd"].shape[1] != nb_load or file_["pg"].shape[1] != nb_gen:
        raise RuntimeError(
            f"pfdelta data shape mismatch for {fn}: data has {file_['pd'].shape[1]} loads / "
            f"{file_['pg'].shape[1]} gens, lsgrid has {nb_load} loads / {nb_gen} gens"
        )
    nb_avail = file_["pd"].shape[0]
    if nb_pf_total > nb_avail:
        print(f"{fn}: only {nb_avail} pfdelta samples available (< {nb_pf_total} requested), using all of them")
    n = min(nb_pf_total, nb_avail)
    
    # load_bus = np.asarray([el.bus_id for el in lsgrid.get_loads()])
    load_p = file_["pd"][:n] * base_mva
    load_q = file_["qd"][:n] * base_mva
    # load_p = load_p[:, load_bus]
    # load_q = load_q[:, load_bus]
    
    all_v_pu = file_["vm"][:n]
    # NB: "pg" is pfdelta's pre-solve generator dispatch target, NOT what was actually
    # produced by the solved power flow whose (vm, va) we read below -- using it here
    # made every sample's injections inconsistent with its own solved voltages (huge
    # KCL residuals, wrong ac_pf convergence). "pg_sol" is the actual solved dispatch
    # that balances against pd/qd/vm/va, so it's the one to feed into lightsim2grid.
    gen_p = file_["pg_sol"][:n] * base_mva
    gen_bus = np.asarray([el.bus_id for el in lsgrid.get_generators()])
    gen_v_pu = all_v_pu[:, gen_bus]
    # gen_p = gen_p[:, gen_bus]
    
    if DEBUG_PF_DELTA:
        # sanity check: re-solve a handful of samples with lightsim2grid's own ac_pf() and
        # make sure it converges to (essentially) pfdelta's reported voltage. NB: this can
        # NOT be done with lsgrid.check_solution() -- once an ac_pf()/dc_pf() has already
        # run on this `lsgrid` (always true here: ls_injection.py solves the base case
        # before calling this function), check_solution() keeps testing every proposed V
        # against that *first* solve's cached Sbus and silently ignores any update_loads_p
        # / update_gens_p / update_gens_v done afterwards -- so it reports the same bogus,
        # huge residual for every sample regardless of whether the injections are correct.
        all_loads_changed = np.ones(load_p.shape[1], dtype=bool)
        all_gens_changed = np.ones(gen_p.shape[1], dtype=bool)
        nb_check = min(n, 20)
        for ts in range(nb_check):
            lsgrid.update_loads_p(all_loads_changed, load_p[ts])
            lsgrid.update_loads_q(all_loads_changed, load_q[ts])
            lsgrid.update_gens_p(all_gens_changed, gen_p[ts])
            lsgrid.update_gens_v(all_gens_changed, gen_v_pu[ts])  # expects pu
            target_v = file_["vm"][ts] * np.exp(1j * file_["va"][ts])
            res_v = lsgrid.ac_pf(target_v, 10, 1e-8)
            lsgrid.unset_changes()
            if res_v.shape[0] == 0:
                raise RuntimeError(f"Error when reading data for ts {ts}: ac_pf did not converge")
            if np.abs(res_v - target_v).max() >= 1e-5:
                raise RuntimeError(f"Error when reading data for ts {ts}")
        
    return load_p, load_q, gen_p, gen_v_pu


def get_loads_gens(load_p_init, load_q_init, gen_p_init, prng, nb_ts=288):
    # scale loads

    # use some French time series data for loads
    # see https://github.com/BDonnot/data_generation for where to find this file
    coeffs = {"sources": {
    "country": "France",
    "year": "2012",
    "web": "http://clients.rte-france.com/lang/fr/visiteurs/vie/vie_stats_conso_inst.jsp"
    },
    "month": {
    "jan": 1.21,
    "feb": 1.40,
    "mar": 1.05,
    "apr": 1.01,
    "may": 0.86,
    "jun": 0.84,
    "jul": 0.84,
    "aug": 0.79,
    "sep": 0.85,
    "oct": 0.94,
    "nov": 1.01,
    "dec": 1.20
    },
    "day": {
    "mon": 1.01,
    "tue": 1.05,
    "wed": 1.05,
    "thu": 1.05,
    "fri": 1.03,
    "sat": 0.93,
    "sun": 0.88
    },
    "hour": {
    "00:00": 1.00,
    "01:00": 0.93,
    "02:00": 0.91,
    "03:00": 0.86,
    "04:00": 0.84,
    "05:00": 0.85,
    "06:00": 0.90,
    "07:00": 0.97,
    "08:00": 1.03,
    "09:00": 1.06,
    "10:00": 1.08,
    "11:00": 1.09,
    "12:00": 1.09,
    "13:00": 1.09,
    "14:00": 1.06,
    "15:00": 1.03,
    "16:00": 1.00,
    "17:00": 1.00,
    "18:00": 1.04,
    "19:00": 1.09,
    "20:00": 1.05,
    "21:00": 1.01,
    "22:00": 0.99,
    "23:00": 1.03
    }
    }
    # compute the 
    inter_count = int(nb_ts // 24)
    vals = list(coeffs["hour"].values())
    x_final = np.arange(inter_count * len(vals))

    # interpolate them at 5 minutes resolution (instead of 1h)
    vals.append(vals[0])
    vals = np.array(vals) * coeffs["month"]["oct"] * coeffs["day"]["mon"]
    x_interp = inter_count * np.arange(len(vals))
    coeffs = interp1d(x=x_interp, y=vals, kind="cubic")
    all_vals = coeffs(x_final).reshape(-1, 1)
    if DEBUG:
        all_vals[:] = 1
    # do not modify loads / gens with negative values (suspicious in this settings)
    mask_load = load_p_init < 0. 
    mask_gen = gen_p_init < 0. 
    
    # compute the "smooth" loads matrix
    load_p_smooth = all_vals * load_p_init.reshape(1, -1)
    load_q_smooth = all_vals * load_q_init.reshape(1, -1)
    
    # add a bit of noise to it to get the "final" loads matrix
    load_p = load_p_smooth * prng.lognormal(mean=0., sigma=0.003, size=load_p_smooth.shape)
    load_q = load_q_smooth * prng.lognormal(mean=0., sigma=0.003, size=load_q_smooth.shape)
    if DEBUG:
        load_p[:] = load_p_smooth
        load_q[:] = load_q_smooth
    # load_p_smooth[:, mask_load] = load_p_init[mask_load]
    # load_q_smooth[:, mask_load] = load_q_init[mask_load]
    
    # scale generators accordingly
    total_load_per_scenario = load_p.sum(axis=1).reshape(-1, 1) 
    total_load_per_scenario -= gen_p_init[mask_gen].sum()
    scale_coeff = total_load_per_scenario / load_p_init.sum()
    gen_p = scale_coeff * gen_p_init.reshape(1, -1)
    gen_p[:, mask_gen] = gen_p_init[mask_gen]
    return load_p, load_q, gen_p


def get_loads_gens_sable(load_p_init, load_q_init, gen_p_init, prng, nb_ts=288):
    """
    Same waht to generate data than the SABLE paper:
    'SABLE: GPU-Based Power Flow Accelerator for Sparsity-Aware Batched Learning'
    see https://arxiv.org/abs/2606.07099
    
    :param load_p_init: Description
    :param load_q_init: Description
    :param gen_p_init: Description
    :param prng: Description
    :param nb_ts: Description
    """
    # do not modify loads / gens with negative values (suspicious in this settings)
    mask_load = load_p_init < 0. 
    mask_gen = gen_p_init < 0. 
    
    eps_p = prng.uniform(low=-0.1, high=0.1, size=(nb_ts,load_p_init.size))
    eps_q = prng.uniform(low=-0.1, high=0.1, size=(nb_ts,load_p_init.size))
    load_p = load_p_init * (1 + eps_p)
    load_q = load_q_init * (1 + eps_q)
    
    # scale generators accordingly
    # do not change 'negative generator' (suspicious)
    total_load_per_scenario = load_p.sum(axis=1).reshape(-1, 1) 
    total_load_per_scenario -= gen_p_init[mask_gen].sum()
    scale_coeff = total_load_per_scenario / load_p_init.sum()
    gen_p = scale_coeff * gen_p_init.reshape(1, -1)
    gen_p[:, mask_gen] = gen_p_init[mask_gen]
    if DEBUG:
        gen_p[:] = gen_p_init
        load_p[:] = load_p_init
        load_q[:] = load_q_init
    return load_p, load_q, gen_p
    