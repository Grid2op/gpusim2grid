"""
case6515rte_screen.py — batched N-1 contingency screen on the GPU.

Builds one contingency per branch (lines then trafos), solves the whole batch
on the GPU reusing a single base-case factorization, and reports convergence
statistics, the worst residuals, and per-contingency timing.

Run:
    python examples/case6515rte_screen.py                 # default: case6515rte
    python examples/case6515rte_screen.py case14          # any pandapower case
    python examples/case6515rte_screen.py case6515rte 256 # case + batch_size
"""
import sys

import numpy as np

from _common import load_case, print_batch_timings
from gpusim2grid import ContingencyAnalysisGPU

_RESIDUAL_CONVERGED = 1e-4


def main(case="case6515rte", batch_size=256, handle_disconnected_grid=True):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    grid = d["grid"]
    n_lines = len(grid.get_lines())
    n_trafos = len(grid.get_trafos())
    n_ctg = n_lines + n_trafos
    print(f"  n_bus={d['n_bus']}, branches={n_ctg} ({n_lines} lines, {n_trafos} trafos)")

    # One N-1 per branch.
    contingencies = [[c] for c in range(n_ctg)]

    ca = ContingencyAnalysisGPU(
        grid,
        nb_iter=10,
        max_iter_base=10,
        tol_base=1e-6,
        handle_disconnected_grid=handle_disconnected_grid,
    )
    ca.add_contingencies_by_branch_id(contingencies)

    print(f"Screening {n_ctg} contingencies on the GPU (batch_size={batch_size}) ...")
    ca.compute(batch_size=batch_size)
    ca.compute_flows()

    residuals = ca.last_residuals()
    timings = ca.timings

    finite = np.isfinite(residuals)
    converged = finite & (np.abs(residuals) < _RESIDUAL_CONVERGED)
    n_conv = int(converged.sum())

    print(f"  converged           : {n_conv}/{n_ctg} ({n_conv / n_ctg:.1%})")
    print(f"  disconnecting (NaN) : {int((~finite).sum())}")
    print()
    print_batch_timings(timings, unit="contingency", unit_plural="contingencies")

    # Show the worst (largest finite residual) non-trivial contingencies.
    order = np.argsort(np.where(finite, np.abs(residuals), -np.inf))[::-1]
    print("  worst finite residuals (branch_id -> |F|inf):")
    for c in order[:5]:
        print(f"    branch {int(c):5d} -> {residuals[c]:.3e}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case6515rte"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(case, bs)
