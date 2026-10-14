# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import os
import numpy as np

from generate_data_from_matpower import (
    get_loads_gens_pfdelta_n,
    get_loads_gens,
    get_loads_gens_sable
)

from get_init_method import get_ratio_base

SEED = 0
NB_PF_TOTAL = 12_000  # by default, actually never really used
REF_PATH = "../matpowerdata"  # by default
VERBOSE_TIMINGS = False
PATH_RESULTS = "../raw_results/"
TOL_PF = 1e-8  # like sable


all_file_names = [
    "case14.m",  # PF DELTA
    "case_ieee30.m",  # PF DELTA
    "case57.m",  # PF DELTA
    "case118.m",      # PF DELTA
    "case_ACTIVSg500.m",
    "case1354pegase.m",  # SABLE   # does not work really well
    "case_ACTIVSg2000.m",
    "case2869pegase.m",
    "case3120sp.m",  # SABLE
    "case3375wp.m",      # init from file
    "case6515rte.m",     # init from dc
    "case9241pegase.m",  # SABLE
    "case_ACTIVSg10k.m",  # init from dc
    "case_ACTIVSg25k.m",  # SABLE
    "case_ACTIVSg70k.m",   # init from file
]


def get_default_args():
    parser = argparse.ArgumentParser(
                        prog='Start experimentation',
                        description='Running the same grid (init from matpower.m files) varying the injections (demand and generation)',
    )
    parser.add_argument('--sample_data_meth', type=str, default="sable",
                        help="How to sample the data (fr, sable or pfdelta_n)")
    parser.add_argument('--add_to_name', type=str, default="",
                        help="Customize the name of the experiment")
    parser.add_argument('--nb_pf', type=int, default=NB_PF_TOTAL,
                        help=f"Number of powerflows to run default {NB_PF_TOTAL}")
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f"Seed to use for prng when generating data default {SEED}")
    parser.add_argument("--save_flows", action="store_true", help="Save the flows")
    parser.add_argument("--save_voltages", action="store_true", help="Save the voltages")
    parser.add_argument("--evaluate_kcl", action="store_true",
                        help="Evaluate the KCL mismatch (mean and max, over buses and "
                             "scenarios) of the computed voltages and save it in the "
                             "results json")
    return parser


def get_args():
    parser = get_default_args()
    parser.add_argument('--nb_threads', type=int, default=1,
                        help=f"Number of threads used for making the computation default to {1}")
    parser.add_argument("--grids", type=str, default=",".join(all_file_names),
                         help="comma-separated matpower filenames to benchmark (default: all "
                              f"{len(all_file_names)} grids in all_file_names) -- lets a caller "
                              "restrict a run to a single grid, e.g. for per-grid nb_pf sizing "
                              "(see ls_grid_sizing.py)")
    return parser


def get_final_name(path, tmp_nm, add_to_name):
    if add_to_name:
        tmp_nm += add_to_name
    final_name = f"{tmp_nm}.json"
    full_path_final = os.path.join(path, final_name)
    return full_path_final


def get_injection_from_base(
    fn,
    lsgrid,
    sample_meth="fr",
    nb_pf_total=NB_PF_TOTAL,
    seed=SEED):
    """
    Sample the injections (demand and generation) from different possibilities.
    Only "sable" was used.
    
    It requires a lightsim2grid "LSGrid" as input as it was the most convenient solver to export
    the matpower data. It is the most convenient because:
    
    1. it keeps the same order as the matpower files
    2. data are easily accessible in python (unlike julia based powermodels or exa-pf)
    3. data are easily accessible from numpy 
    """
    if sample_meth == "pfdelta_n":
        # deterministic: read pre-solved scenarios straight from disk, no prng involved
        # actually never used in the paper.
        return get_loads_gens_pfdelta_n(fn, lsgrid, nb_pf_total)

    load_p_init, load_q_init, *_ = lsgrid.get_loads_res_full()
    gen_p_init, *_ = lsgrid.get_gen_res_full()
    ratio_base = get_ratio_base(fn)
    if abs(ratio_base - 1.) > 1e-6:
        load_p_init = load_p_init.copy()
        load_q_init = load_q_init.copy()
        gen_p_init = gen_p_init.copy()

        load_p_init *= ratio_base
        load_q_init *= ratio_base
        gen_p_init *= ratio_base
    prng = np.random.default_rng(seed)
    if sample_meth == "fr":
        # actually never used in the paper
        sample_fun = get_loads_gens
    elif sample_meth == "sable":
        # the only one used in the paper
        sample_fun = get_loads_gens_sable
    else:
        raise ValueError(f"sample_meth (={sample_meth}) not in ['fr', 'sable', 'pfdelta_n']")
    load_p, load_q, gen_p = sample_fun(load_p_init, load_q_init, gen_p_init, prng, nb_pf_total)
    return load_p, load_q, gen_p
