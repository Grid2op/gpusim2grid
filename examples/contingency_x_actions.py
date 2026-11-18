# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
contingency_x_actions.py -- screen remedial actions against N-1 contingencies.

One ``compute()`` solves the contingencies and, as each one finishes, the candidate actions
(``gpusim2grid.actions.Action``: here lines opened and loads shed) of the contingencies that
violate a limit (pypowsybl's AT_LEAST_ONE_VIOLATION_CONDITION, checked on the GPU), each
starting from its contingency's converged state. The actions are then ranked per contingency by
the highest loading they leave.

The grid has no current limits of its own: they are set at 1.05 x the base case's currents, and
``violation_threshold`` keeps a 5 % margin on them (a limit reduction, as pypowsybl's).

Run:
    python examples/contingency_x_actions.py            # default: case118
    python examples/contingency_x_actions.py case300    # any pandapower case
"""

from __future__ import annotations

import sys

import numpy as np
from _common import load_case

from gpusim2grid import ContingencyAnalysisGPU, StrategyCondition
from gpusim2grid.actions import Action

BATCH: int = 256


def main(case: str = "case118") -> None:
    grid = load_case(case)["grid"]
    # current limits: 1.05 x the base case's currents, at least the median one (a lightly loaded
    # line would otherwise get an unrealistically small limit)
    for res, set_limit in (
        (grid.get_line_res1, grid.set_line_current_limit_side1),
        (grid.get_line_res2, grid.set_line_current_limit_side2),
    ):
        amps_ka: np.ndarray = np.asarray(res()[3])
        set_limit(1.05 * np.maximum(amps_ka, np.median(amps_ka)))
    n_lines: int = len(grid.get_lines())

    ca = ContingencyAnalysisGPU(
        grid,
        nb_iter=30,
        compute_limit_violations=True,
        violation_threshold=0.95,  # a 5 % margin on every limit
        strategy_condition=StrategyCondition.ANY_VIOLATION,  # a contingency's actions run if it violates
    )  # each action starts from its own contingency's converged state (strategy_warm_start, default)
    ctgs: list[list[int]] = [[b] for b in range(0, n_lines, 4)]
    ca.add_contingencies_by_branch_id(ctgs)
    n_ctg: int = len(ctgs)

    # candidate actions: open a line (both ends), or shed a load
    actions: list[Action] = [Action(line_bus1={b: -1}, line_bus2={b: -1}) for b in range(1, n_lines, 7)]
    actions += [Action(load_bus={ld: -1}) for ld in range(0, len(grid.get_loads()), 9)]
    ca.set_actions(actions)
    # every (contingency, action) pair, without a Python call per row
    c_idx, a_idx = np.meshgrid(np.arange(n_ctg), np.arange(len(actions)), indexing="ij")
    ca.add_strategies(c_idx.ravel(), a_idx.ravel())
    ctg_of: np.ndarray = c_idx.ravel()

    ca.compute(batch_size=BATCH)
    counts = ca.get_violation_counts()
    violating: np.ndarray = (counts["low_voltage"] + counts["high_voltage"] + counts["current"])[:n_ctg] > 0
    rho_all, branch_all = ca.get_max_loading()  # per row: the contingencies, then the strategies
    rho_ctg, rho, branch = rho_all[:n_ctg], rho_all[n_ctg:], branch_all[n_ctg:]
    ran: np.ndarray = ca.get_strategy_ran()
    converged: np.ndarray = ca.converged()[n_ctg:]
    print(f"{n_ctg} contingencies, {int(violating.sum())} violate a limit")
    print(f"{int(ran.sum())} of {ca.n_strategies} strategies run, {int((ran & converged).sum())} converged")

    # the best action of each violating contingency
    for c in np.flatnonzero(violating)[:10]:
        rows = np.flatnonzero((ctg_of == c) & ran & converged & np.isfinite(rho))
        if not len(rows):
            continue
        best = rows[np.argmin(rho[rows])]
        print(
            f"  contingency {ctgs[c]}: max loading {rho_ctg[c]:.3f} -> {rho[best]:.3f} "
            f"with action {a_idx.ravel()[best]} (most loaded branch {branch[best]})"
        )


if __name__ == "__main__":
    main(*sys.argv[1:2])
