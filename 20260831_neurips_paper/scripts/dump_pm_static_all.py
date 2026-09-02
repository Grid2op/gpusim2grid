# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Runs julia_powermodels/dump_pm_static.jl once for every matpower case in matpowerdata/ (or a
given subset), producing the once-per-case static PowerModels.jl network dict every ml_*.py
script needs (see ml_pfdelta_bridge.py's module docstring for why this is a one-time, offline
step -- topology/branch-admittance/limits parsing, no solving, so it does not depend on
--sample_data_meth/--nb_pf/--seed the way the other *_injection.py scripts do). The dumped
<case>.json is written into matpowerdata/, alongside the source .m/.mat grid it was parsed from.

Mirrors ls_injection.py / olf_injection.py's loop-over-all-grids + skip-if-already-done +
tqdm + JSON-trail conventions, but drives a Julia subprocess per grid rather than solving in
Python -- each dump only needs to happen once per case, ever (re-run with --force to redo).

Usage (run from this directory, like every other script in this folder):
    python dump_pm_static_all.py
    python dump_pm_static_all.py --cases case118.m case14.m
    python dump_pm_static_all.py --force
"""

import argparse
import glob
import json
import os
import subprocess
import time

from tqdm import tqdm

REF_PATH = "../matpowerdata"
OUT_DIR = "../matpowerdata"
JULIA_PROJECT = "julia_powermodels"
JULIA_SCRIPT = "julia_powermodels/dump_pm_static.jl"
SUMMARY_PATH = os.path.join(OUT_DIR, "_dump_summary.json")


def get_args():
    parser = argparse.ArgumentParser(
        description="Dump PowerModels.jl's static network dict (once per case, no solving) "
        "for every matpower case, for use by ml_injection.py / ml_pfdelta_bridge.py."
    )
    parser.add_argument(
        "--cases", type=str, nargs="+", default=None,
        help="case file names (e.g. case118.m) to dump; default: every matpowerdata/*.m",
    )
    parser.add_argument("--force", action="store_true", help="redo cases whose JSON already exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.cases is not None:
        case_files = args.cases
    else:
        case_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REF_PATH, "*.m")))

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r") as f:
            summary = json.load(f)

    for fn in tqdm(case_files):
        case_stem, _ = os.path.splitext(fn)
        out_path = os.path.join(OUT_DIR, f"{case_stem}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"{fn}: already dumped ({out_path}), skipping (use --force to redo)")
            continue

        case_path = os.path.join(REF_PATH, fn)
        beg = time.perf_counter()
        proc = subprocess.run(
            [
                "julia", f"--project={JULIA_PROJECT}", JULIA_SCRIPT,
                "--case", case_path, "--out", out_path,
            ],
            capture_output=True, text=True,
        )
        dt = time.perf_counter() - beg

        if proc.returncode != 0:
            print(f"{fn}: FAILED ({dt:.1f}s)\n{proc.stderr[-2000:]}")
            summary[fn] = {"ok": False, "time": dt, "error": proc.stderr[-2000:]}
        else:
            print(f"{fn}: OK ({dt:.1f}s) -- {proc.stdout.strip().splitlines()[-1]}")
            summary[fn] = {"ok": True, "time": dt, "out": out_path}

        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, fp=f, indent=2)

    n_ok = sum(1 for v in summary.values() if v.get("ok"))
    print(f"\n{n_ok}/{len(summary)} cases dumped successfully. Summary: {SUMMARY_PATH}")
