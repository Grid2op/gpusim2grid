# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
injection_sweep_scan.py — batched load-scaling sweep on the GPU.

Builds a scenario per load-scaling factor (same Ybus, varying Sbus), solves the
whole batch on the GPU reusing a single base-case factorization, and reports
convergence statistics, the worst residuals, and per-scenario timing.

Run:
    python examples/injection_sweep_scan.py                # default: case14
    python examples/injection_sweep_scan.py case6515rte     # any pandapower case
    python examples/injection_sweep_scan.py case14 64 0.5 1.5  # + batch_size, scale range
"""
import sys

import numpy as np

from _common import load_case, print_batch_timings
from gpusim2grid import InjectionSweepGPU

_RESIDUAL_CONVERGED = 1e-4


def main(case="case14", batch_size=64, scale_lo=0.5, scale_hi=1.5, n_scenarios=200):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    grid = d["grid"]
    sn_mva = grid.get_sn_mva()
    print(f"  n_bus={d['n_bus']}, sn_mva={sn_mva}")

    # One scenario per load-scaling factor, applied uniformly to every bus.
    scales = np.linspace(scale_lo, scale_hi, n_scenarios)
    p_base_mw = d["Sbus"].real * sn_mva
    q_base_mvar = d["Sbus"].imag * sn_mva
    p_mw = scales[:, None] * p_base_mw[None, :]
    q_mvar = scales[:, None] * q_base_mvar[None, :]

    sweep = InjectionSweepGPU(grid, nb_iter=10, max_iter_base=10, tol_base=1e-6)
    sweep.set_injections(p_mw, q_mvar, sn_mva)

    print(f"Sweeping {n_scenarios} load-scaling scenarios "
          f"(scale in [{scale_lo}, {scale_hi}]) on the GPU (batch_size={batch_size}) ...")
    sweep.compute(batch_size=batch_size)
    sweep.compute_flows()

    residuals = sweep.last_residuals()
    timings = sweep.timings

    finite = np.isfinite(residuals)
    converged = finite & (np.abs(residuals) < _RESIDUAL_CONVERGED)
    n_conv = int(converged.sum())

    print(f"  converged        : {n_conv}/{n_scenarios} ({n_conv / n_scenarios:.1%})")
    print()
    print_batch_timings(timings, unit="scenario")
    print()

    # Show the worst (largest) residuals.
    order = np.argsort(np.abs(residuals))[::-1]
    print("  worst residuals (scale -> |F|inf):")
    for s in order[:5]:
        print(f"    scale {scales[s]:.3f} -> {residuals[s]:.3e}")

    # Sanity check: voltage magnitude at slack should barely move with load scale.
    V = sweep.V_results.to_numpy().reshape(n_scenarios, d["n_bus"])
    vm_min = np.abs(V[converged]).min(axis=1)
    print(f"  min |V| across converged scenarios: {vm_min.min():.4f} .. {vm_min.max():.4f}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case14"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    hi = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    main(case, bs, lo, hi)
