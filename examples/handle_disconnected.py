# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
handle_disconnected.py — solve the largest component of a split grid.

Some N-k contingencies disconnect the grid into several islands. By default such
a contingency is *skipped* (its residual is reported as NaN). With
``handle_disconnected_grid=True`` the GPU instead solves the **largest connected
component** and masks the islanded buses (their voltage is reported as NaN) — the
analogue of lightsim2grid's ``ContingencyAnalysisCPP.handle_disconnected_grid``.

This example screens every N-1 on IEEE 14-bus (one of them splits off an island)
and shows the difference. It uses the recommended ``ContingencyAnalysisGPU``
facade, seeded directly from a solved lightsim2grid grid.

Run:
    python examples/handle_disconnected.py            # default: case14
    python examples/handle_disconnected.py case118    # any pandapower case
"""
import sys

import numpy as np

from _common import load_case
from gpusim2grid import ContingencyAnalysisGPU, optimize_reference_slack


def _screen(grid, cont, handle_disconnected_grid):
    """Run the N-1 screen and return (n_skipped, voltages [n_ctg, n_bus])."""
    ca = ContingencyAnalysisGPU(
        grid, handle_disconnected_grid=handle_disconnected_grid, nb_iter=8)
    ca.add_contingencies_by_branch_id(cont)
    ca.compute(batch_size=256)
    n_bus = grid.get_bus_vn_kv().shape[0]
    V = ca.V_results.to_numpy().reshape(len(cont), n_bus)   # host copy
    n_skipped = int(np.isnan(ca.last_residuals()).sum())
    return n_skipped, V


def main(case="case14"):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    grid = d["grid"]
    n_ctg = len(grid.get_lines()) + len(grid.get_trafos())
    cont = [[c] for c in range(n_ctg)]        # one N-1 per branch

    # Optional: for multi-slack grids, pick the angle-reference slack stranded by
    # the fewest contingencies and re-solve the base case with it, so as few
    # split contingencies as possible have to be skipped. Harmless on a single-
    # slack grid (nothing changes), so it is safe to always call.
    ref = optimize_reference_slack(grid, cont)
    print(f"  reference slack (gridmodel bus id): {ref}")

    skipped_off, _ = _screen(grid, cont, handle_disconnected_grid=False)
    skipped_on, V_on = _screen(grid, cont, handle_disconnected_grid=True)

    # Per contingency, how many buses ended up masked (NaN) in the ON run.
    masked_per_ctg = np.isnan(V_on).sum(axis=1)
    split = np.where(masked_per_ctg > 0)[0]

    print(f"  contingencies screened          : {n_ctg}")
    print(f"  skipped (NaN) — feature OFF      : {skipped_off}")
    print(f"  skipped (NaN) — feature ON       : {skipped_on}")
    print(f"  now solved on largest component  : {skipped_off - skipped_on}")
    if split.size:
        print("  split contingencies (branch_id -> masked buses):")
        for c in split:
            print(f"    branch {int(c):5d} -> {int(masked_per_ctg[c])} bus(es) masked")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case14")
