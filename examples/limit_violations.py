"""
limit_violations.py — N-1 screen with compute_limit_violations.

lightsim2grid can attach operational limits to a grid: per-bus high/low
voltage (``set_bus_voltage_limits``) and per-side branch thermal current
(``set_line_current_limit_side1/2`` / ``set_trafo_current_limit_side1/2``).
``ContingencyAnalysisGPU(grid, compute_limit_violations=True)`` checks every
contingency against those limits **fused into the GPU solve itself** — a
single thread per contingency, writing only a small, bounded per-contingency
record buffer — so this scales to large N-k batches without ever having to
materialize (or transfer back to the host) the full dense voltage / current
arrays just to threshold-compare them away. This mirrors lightsim2grid's own
``ContingencyAnalysis.compute_limit_violations`` flag.

A non-convergent contingency also shows up in the same per-contingency
violation list (as a ``DIVERGED`` entry) instead of needing a separate
convergence check.

This example configures a simple, illustrative operating envelope (±voltage
band around nominal; thermal limits derived from the worst-case loading seen
in an unconstrained baseline screen, so at least one violation is guaranteed
to show up) and reports what compute_limit_violations finds.

Run:
    python examples/limit_violations.py            # default: case14
    python examples/limit_violations.py case118     # any pandapower case
"""
import sys

import numpy as np

from _common import load_case
from gpusim2grid import ContingencyAnalysisGPU


def _configure_illustrative_limits(grid, ca_baseline, n_ctg, n_lines, n_trafos):
    """Set a simple operating envelope on the grid:
      - bus voltage within +/-6% / -6% of nominal (a common operational band)
      - branch thermal limits at 80% of the worst loading observed across the
        (unconstrained) N-1 baseline screen -- guarantees the branch that is
        most loaded under its own worst contingency will violate, so this
        example always has something to report regardless of the case.
    """
    vn_kv = grid.get_bus_vn_kv()
    grid.set_bus_voltage_limits(0.94 * vn_kv, 1.06 * vn_kv)

    n_bus = ca_baseline.n_bus
    ca_baseline.compute_flows()
    or_amps = ca_baseline.or_amps.to_numpy().reshape(n_ctg, n_lines + n_trafos)
    ex_amps = ca_baseline.ex_amps.to_numpy().reshape(n_ctg, n_lines + n_trafos)
    worst_ka = 1e-3 * np.nanmax(np.maximum(or_amps, ex_amps), axis=0)   # per branch, kA
    limit_ka = 0.8 * worst_ka

    grid.set_line_current_limit_side1(limit_ka[:n_lines])
    grid.set_line_current_limit_side2(limit_ka[:n_lines])
    grid.set_trafo_current_limit_side1(limit_ka[n_lines:])
    grid.set_trafo_current_limit_side2(limit_ka[n_lines:])


def main(case="case14"):
    print(f"Loading {case} and solving the base case on the CPU ...")
    d = load_case(case)
    grid = d["grid"]
    n_lines, n_trafos = len(grid.get_lines()), len(grid.get_trafos())
    n_ctg = n_lines + n_trafos
    cont = [[c] for c in range(n_ctg)]   # one N-1 per branch

    # Unconstrained baseline screen: used only to derive an illustrative
    # thermal envelope below (not needed in normal usage where the grid
    # already carries real operating limits).
    baseline = ContingencyAnalysisGPU(grid, nb_iter=8)
    baseline.add_contingencies_by_branch_id(cont)
    baseline.compute(batch_size=256)
    _configure_illustrative_limits(grid, baseline, n_ctg, n_lines, n_trafos)

    # Bus/branch limits are extracted from `grid` automatically at
    # construction (bridge path); set_limits_from_grid() is also available
    # to (re-)pull them later, e.g. after changing limits on the grid.
    ca = ContingencyAnalysisGPU(grid, compute_limit_violations=True, nb_iter=8)
    ca.add_contingencies_by_branch_id(cont)
    ca.compute(batch_size=256)

    converged = ca.converged()
    violations = ca.get_violations()
    # get_violations()' per-contingency records are capped at
    # violation_capacity; get_violation_counts() reports the TRUE, uncapped
    # count of each type per contingency, so totals stay exact even for a
    # contingency that hit get_violations_truncated().
    counts = ca.get_violation_counts()

    print(f"  contingencies screened      : {n_ctg}")
    print(f"  converged                   : {int(converged.sum())}")
    print(f"  diverged / skipped          : {int((~converged).sum())}")
    print(f"  total LOW_VOLTAGE violations : {int(np.maximum(counts['low_voltage'], 0).sum())}")
    print(f"  total HIGH_VOLTAGE violations: {int(np.maximum(counts['high_voltage'], 0).sum())}")
    print(f"  total CURRENT violations     : {int(np.maximum(counts['current'], 0).sum())}")

    flagged = [c for c in range(n_ctg) if violations[c]]
    print(f"  contingencies with violations: {len(flagged)}")
    for c in flagged[:10]:
        # Records are ordered CURRENT before LOW_VOLTAGE/HIGH_VOLTAGE (thermal
        # violations are first-order, voltage second-order).
        kinds = ", ".join(f"{v.violation_type.name}(id={v.element_id}, "
                          f"value={v.value:.3f}, limit={v.limit:.3f})"
                          for v in violations[c])
        branch = cont[c][0]
        kind = "line" if branch < n_lines else "trafo"
        local_id = branch if branch < n_lines else branch - n_lines
        print(f"    trip {kind} {local_id:3d}: {kinds}")
    if len(flagged) > 10:
        print(f"    ... and {len(flagged) - 10} more")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case14")
