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

from _common import load_case, branch_data
from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver

_RESIDUAL_CONVERGED = 1e-4


def main(case="case6515rte", batch_size=256):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    branch_args, n_lines, n_trafos = branch_data(d["grid"])
    n_ctg = n_lines + n_trafos
    print(f"  n_bus={d['n_bus']}, branches={n_ctg} ({n_lines} lines, {n_trafos} trafos)")

    # One N-1 per branch.
    contingencies = [[c] for c in range(n_ctg)]

    solver = ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=10, max_iter_base=10, tol_base=1e-6,
    )
    solver.set_branch_data(*branch_args)
    solver.build_contingencies(contingencies)

    print(f"Screening {n_ctg} contingencies on the GPU (batch_size={batch_size}) ...")
    solver.run()

    residuals = solver.residuals.to_numpy()
    timings = solver.timings

    finite = np.isfinite(residuals)
    converged = finite & (np.abs(residuals) < _RESIDUAL_CONVERGED)
    n_conv = int(converged.sum())

    print(f"  converged           : {n_conv}/{n_ctg} ({n_conv / n_ctg:.1%})")
    print(f"  disconnecting (NaN) : {int((~finite).sum())}")
    print(f"  chunks              : {timings.n_chunks}")
    print(f"  total chunk time    : {timings.t_chunks_total_wall_ms:.1f} ms")
    print(f"  per contingency     : {timings.t_per_contingency_ms:.4f} ms")

    # Show the worst (largest finite residual) non-trivial contingencies.
    order = np.argsort(np.where(finite, np.abs(residuals), -np.inf))[::-1]
    print("  worst finite residuals (branch_id -> |F|inf):")
    for c in order[:5]:
        print(f"    branch {int(c):5d} -> {residuals[c]:.3e}")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "case6515rte"
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    main(case, bs)
