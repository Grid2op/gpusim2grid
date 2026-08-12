# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
scenario_sweep_scan.py — row-aligned combined topology + injection sweep.

Builds one scenario per load-scaling factor (same as injection_sweep_scan.py),
but every 10th scenario ALSO trips a branch: row `i`'s (P, Q) profile is solved
together with row `i`'s own set of tripped branches, independently of every
other row — the GPU analogue of lightsim2grid's ``ScenarioSweep``. Reports
convergence statistics, how many scenarios were topology-only vs.
injection-only vs. both, and per-scenario timing.

Run:
    python examples/scenario_sweep_scan.py                # default: case14
    python examples/scenario_sweep_scan.py case6515rte     # any pandapower case
    python examples/scenario_sweep_scan.py case14 64 0.5 1.5  # + batch_size, scale range
"""
import sys

import numpy as np

from _common import load_case, print_batch_timings
from gpusim2grid import ScenarioSweepGPU

_RESIDUAL_CONVERGED = 1e-4


def main(case="case14", batch_size=64, scale_lo=0.5, scale_hi=1.5, n_scenarios=200):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    grid = d["grid"]
    sn_mva = grid.get_sn_mva()
    n_branches = len(grid.get_lines()) + len(grid.get_trafos())
    print(f"  n_bus={d['n_bus']}, n_branches={n_branches}, sn_mva={sn_mva}")

    # One scenario per load-scaling factor, applied uniformly to every bus
    # (same injection sweep as injection_sweep_scan.py) ...
    scales = np.linspace(scale_lo, scale_hi, n_scenarios)
    p_base_mw = d["Sbus"].real * sn_mva
    q_base_mvar = d["Sbus"].imag * sn_mva
    p_mw = scales[:, None] * p_base_mw[None, :]
    q_mvar = scales[:, None] * q_base_mvar[None, :]

    # ... AND every 10th scenario also trips one branch (cycling through
    # branch ids), row-aligned with its own injection scale. Every other row
    # gets an empty list: "no branches tripped for this row".
    topology = [[] for _ in range(n_scenarios)]
    tripped_rows = list(range(0, n_scenarios, 10))
    for k, row in enumerate(tripped_rows):
        topology[row] = [k % n_branches]

    sweep = ScenarioSweepGPU(grid, nb_iter=10, max_iter_base=10, tol_base=1e-6)
    sweep.set_injections(p_mw, q_mvar, sn_mva)
    sweep.set_topology(topology)

    print(f"Sweeping {n_scenarios} scenarios (scale in [{scale_lo}, {scale_hi}], "
          f"{len(tripped_rows)} of them also tripping a branch) on the GPU "
          f"(batch_size={batch_size}) ...")
    sweep.compute(batch_size=batch_size)
    sweep.compute_flows()

    residuals = sweep.last_residuals()
    disconnected = sweep.get_disconnected().astype(bool)
    timings = sweep.timings

    finite = np.isfinite(residuals)
    converged = finite & (np.abs(residuals) < _RESIDUAL_CONVERGED)
    n_conv = int(converged.sum())

    print(f"  converged            : {n_conv}/{n_scenarios} ({n_conv / n_scenarios:.1%})")
    print(f"  disconnected (NaN)   : {int(disconnected.sum())}/{n_scenarios} "
          f"(topology trip islanded the grid)")
    print()
    print_batch_timings(timings, unit="scenario")
    print()

    # For the rows that tripped a branch: confirm that branch really does
    # carry zero flow in the result (compute_flows() zeroes it device-side).
    or_amps = sweep.or_amps.to_numpy().reshape(n_scenarios, n_branches)
    bad = [(row, topology[row][0])
           for row in tripped_rows
           if converged[row] and or_amps[row, topology[row][0]] != 0.0]
    print(f"  tripped-branch zero-flow check: {'OK' if not bad else 'FAILED ' + str(bad)}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case14"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    main(case, bs, lo, hi)
