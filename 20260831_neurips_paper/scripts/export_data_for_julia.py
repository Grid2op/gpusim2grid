# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Export the exact same injection scenarios used by pp_injection.py / olf_injection.py /
# ls_injection.py / pgm_injection.py / rustpower_injection.py , in a format readable 
# from Julia (via NPZ.jl), so that ExaPF.jl can run
# the same benchmark on CPU and GPU (see exapf_injection.jl / powermodels_injection.jl).
#
# raw matpower order is expected from this file (the common natural ordering for all 
# the baselines in this script). This is why lightsim2grid is used as a reference 
# solver here. 
# So the only alignment work needed here is: scatter per-load values onto their bus id to
# get a (nbus, nscen) array, and drop the disconnected generators (bus_id == -1) to get
# ExaPF's / powermodels own (n_active_gen, nscen) array. Both were verified against different
# matpower grids with disconnected generators and / or multiple generators on the same
# bus.

import json
import os
import warnings

import numpy as np
from tqdm import tqdm

from lightsim2grid.network import init_from_matpower
from matpowercaseframes import CaseFrames

from scenario_utils import (
    all_file_names,
    REF_PATH,
    get_injection_from_base,
    get_default_args,
    TOL_PF,
)
from get_init_method import get_init_method

PF_ALGO_LS = "NR_KLU"
PATH_EXAPF_DATA = "../matpower_injection_data"
BASE_DATA_DIRNAME = "exapf_data_{}_{}"


def get_args():
    parser = get_default_args()
    parser.add_argument('--out_dir', type=str, default=PATH_EXAPF_DATA,
                         help="base directory the exported exapf_data_<...> subdirectory is created in "
                              "(default: %(default)s, relative to this script's directory)")
    parser.add_argument('--case', type=str, default="",
                         help="restrict the export to a single case from all_file_names (e.g. "
                              "case_ACTIVSg10k or case_ACTIVSg10k.m); default: export every case")
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()
    nb_pf_total = int(args.nb_pf)
    seed = int(args.seed)
    sample_data_meth = str(args.sample_data_meth)
    add_to_name = str(args.add_to_name)

    case_list = all_file_names
    if args.case:
        case_fn = args.case if args.case.endswith(".m") else args.case + ".m"
        if case_fn not in all_file_names:
            raise SystemExit(f"--case {args.case!r} not found in all_file_names: {all_file_names}")
        case_list = [case_fn]

    dirname = BASE_DATA_DIRNAME.format(sample_data_meth, nb_pf_total)
    if add_to_name:
        dirname += add_to_name
    out_dir = os.path.join(args.out_dir, dirname)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "sample_data_meth": sample_data_meth,
        "nb_pf": nb_pf_total,
        "seed": seed,
        "ref_path": os.path.abspath(REF_PATH),
        "case_order": [],
        "cases": {},
    }
    
    for fn in tqdm(case_list):
        path = os.path.join(REF_PATH, fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            lsgrid = init_from_matpower(path)

        init_method = get_init_method(path)
        nbus = lsgrid.total_bus()
        if init_method == "flat":
            v_init = np.ones(nbus, dtype=complex)
        elif init_method == "dc":
            v_init_dc = np.ones(nbus, dtype=complex)
            v_init = lsgrid.dc_pf(v_init_dc, 10, 1e-6)
        elif init_method == "file":
            cf = CaseFrames(path)
            v_init = cf.bus["VM"].values * np.exp(1j * np.deg2rad(cf.bus["VA"].values))
        else:
            raise RuntimeError(f"Unknown init method {init_method} for {fn}")

        # get_injection_from_base() reads load_p_init/gen_p_init off lsgrid's *powerflow
        # result* buffers (get_loads_res_full()/get_gen_res_full()), which are only
        # populated after a powerflow has actually been run once (otherwise they hold
        # uninitialized memory) -- mirrors the two warm-up ac_pf() calls in ls_injection.py.
        lsgrid.change_algorithm(PF_ALGO_LS)
        res_0 = lsgrid.ac_pf(v_init, 10, TOL_PF)
        lsgrid.unset_changes()
        if res_0.shape[0] == 0:
            print(f"Error: base powerflow did not converge for {fn}, skipping")
            continue

        load_p, load_q, gen_p = get_injection_from_base(
            fn,
            lsgrid,
            sample_meth=sample_data_meth,
            seed=seed,
            nb_pf_total=nb_pf_total,
        )

        # actual number of scenarios: normally == nb_pf_total, except pfdelta which can
        # come up short if fewer samples are available than requested (see
        # get_loads_gens_pfdelta())
        n_scen = load_p.shape[0]

        # scatter per-load values onto their bus (see module docstring for why this is a
        # safe, order-preserving scatter and not a general re-alignment problem)
        loads = lsgrid.get_loads()
        load_bus = np.array([ld.bus_id for ld in loads], dtype=int)
        load_p_bus = np.zeros((nbus, n_scen), dtype=np.float64)
        load_q_bus = np.zeros((nbus, n_scen), dtype=np.float64)
        load_p_bus[load_bus, :] = load_p.T
        load_q_bus[load_bus, :] = load_q.T

        # drop disconnected generators (bus_id == -1) to match ExaPF's own generator
        # indexing (see module docstring)
        gens = lsgrid.get_generators()
        gen_bus = np.array([g.bus_id for g in gens], dtype=int)
        active_mask = gen_bus != -1
        gen_p_active = np.ascontiguousarray(gen_p[:, active_mask].T)  # (n_active_gen, n_scen)

        case_nm, _ = os.path.splitext(fn)
        np.save(os.path.join(out_dir, f"{case_nm}_load_p_mw.npy"), load_p_bus)
        np.save(os.path.join(out_dir, f"{case_nm}_load_q_mvar.npy"), load_q_bus)
        np.save(os.path.join(out_dir, f"{case_nm}_gen_p_mw.npy"), gen_p_active)
        np.save(os.path.join(out_dir, f"{case_nm}_v_init.npy"), v_init.astype(np.complex128))

        meta["case_order"].append(fn)
        meta["cases"][fn] = {
            "nbus": int(nbus),
            "n_gen_active": int(active_mask.sum()),
            "init_method": init_method,
            "n_scen": int(n_scen),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Data exported to {out_dir}")
