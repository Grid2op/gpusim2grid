# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
 
 
"""One time conversion from .m file to .mat file (required for pypowsybl / OLF).

This uses octave with an installed matpower version.

This should be run from python.
"""

import subprocess
from pathlib import Path

MATPOWER_PATH = "/FULL/PATH/TO/MATPOWER/INSTALL/matpower"

for el in sorted(Path(".").glob("*.m")):
    case_nm = el.stem
    out_mat = f"{case_nm}.mat"
    eval_str = (
        f"addpath(genpath('{MATPOWER_PATH}')); "
        f"mpc = {case_nm}; "
        f"save('-v7', '{out_mat}', 'mpc');"  # force binary MAT v7, explicit var name 'mpc'
    )
    print(f"Converting {case_nm}.m -> {out_mat}")
    result = subprocess.run(
        ["octave", "--no-gui", "--eval", eval_str],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}")
        continue
    size = Path(out_mat).stat().st_size
    if size < 1000:  # sanity check: a real case file shouldn't be tiny
        print(f"  WARNING: {out_mat} is only {size} bytes — likely broken")
    else:
        print(f"  OK ({size} bytes)")