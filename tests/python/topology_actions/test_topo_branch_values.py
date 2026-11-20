# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Step B.1 of docs/dev_notes/topology_actions_plan.md: per-row branch values (phase shifter
angle, ratio). The branch's four Ybus entries change by value; the pattern does not."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import (
    assert_row_matches,
    batch_V_global,
    ca_settings,
    case14,
    make_ca,
    oracle,
    with_current_limits,
)
from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

    from gpusim2grid.actions import Action


pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

BATCH: int = 8


def _actions(grid: LSGrid) -> list[Action]:
    from gpusim2grid.actions import Action

    t0 = grid.get_trafos()[0]
    return [
        Action(trafo_shift_deg={0: 5.0}),
        Action(trafo_shift_deg={0: -8.0}),
        Action(trafo_ratio={0: t0.ratio * 1.03}),
        Action(trafo_ratio={2: grid.get_trafos()[2].ratio * 0.97}, trafo_shift_deg={2: 2.0}),
    ]


@pytest.mark.parametrize("settings", ca_settings())
def test_branch_values_match_lightsim2grid(settings: dict[str, Any], solver_atol: float):
    grid = case14()
    actions = _actions(grid)
    ctgs = [[], [3], [10]]
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    ca = make_ca(grid, settings, batch_size=BATCH)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.compute(batch_size=BATCH)
    assert ca.n_spare_buses == 0
    V = batch_V_global(ca, len(ctgs) + len(strategies), grid.total_bus())
    for r, (c, (a,)) in enumerate(strategies):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
        assert_row_matches(V[len(ctgs) + r], V_ref, solver_atol)


def test_compiled_values_match_a_fresh_copy_per_action():
    """set_actions computes every action's transformer admittances on one copy of the grid,
    putting its transformers back before each action: the values are, bit for bit, those of a
    fresh copy per action -- also for an action touching a transformer an earlier one moved (a
    ratio alone after a shift: the shift must be the grid's again)."""
    from gpusim2grid.contingency_analysis._strategies import compile_action

    grid = case14()
    actions = _actions(grid) + _actions(grid)[::-1]
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions(actions)
    n_lines = len(grid.get_lines())
    assert len(ca._compiled) == len(actions)
    for a, got in zip(actions, ca._compiled, strict=True):
        want = compile_action(grid, a, n_lines)
        assert got.branch_y == want.branch_y
        assert got.base_y == want.base_y


def test_flows_use_the_rows_branch_values(solver_atol: float):
    """compute_flows of a strategy row uses the row's admittance of the changed transformer,
    not the base one: its currents match lightsim2grid's."""
    grid = case14()
    from gpusim2grid.actions import Action

    a = Action(trafo_shift_deg={0: 6.0})
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    ca.compute_flows()
    nb = ca.n_branches
    or_ka: np.ndarray = ca.or_amps.to_numpy().reshape(2, nb) * 1e-3  # A -> kA
    _, ref = oracle(grid, action=a)
    n_lines = len(grid.get_lines())
    ref_ka = np.concatenate([ref.get_line_res1()[3], ref.get_trafo_res1()[3]])
    np.testing.assert_allclose(or_ka[1][n_lines:], ref_ka[n_lines:], atol=10 * solver_atol, rtol=0)
    # the base row is unchanged (the transformer's base current differs from the action's)
    assert abs(or_ka[0][n_lines] - ref_ka[n_lines]) > 1e-4


@pytest.mark.parametrize(("factor", "violates"), [(0.5, False), (0.25, True)])
def test_current_violation_of_the_changed_transformer(factor: float, violates: bool, solver_atol: float):
    """The current violation of a transformer whose value the row changes is checked with the
    row's admittance, not the base one. A +6 deg shift brings transformer 0 from 0.125 kA down
    to 0.056 kA: with limits at 0.5 x the base currents (0.062 kA there) the action relieves it
    (no record, where the base admittance would report one); at 0.25 x it still violates, and
    the record carries lightsim2grid's current."""
    from gpusim2grid.actions import Action
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType, ViolationElementType

    grid = with_current_limits(case14(), factor)
    a = Action(trafo_shift_deg={0: 6.0})
    ca = make_ca(grid, {}, batch_size=BATCH, compute_limit_violations=True)
    ca.solver.violation_capacity = 128
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    _, ref = oracle(grid, action=a)
    ref_ka: float = float(np.asarray(ref.get_trafo_res1()[3])[0])
    rec = [
        v
        for v in ca.get_violations()[1]
        if v.violation_type == LimitViolationType.CURRENT
        and v.element_type == ViolationElementType.TRAFO
        and v.element_id == 0
        and v.side == 1
    ]
    assert len(rec) == int(violates)
    if violates:
        np.testing.assert_allclose(rec[0].value, ref_ka, atol=10 * solver_atol, rtol=0)
        assert rec[0].limit == pytest.approx(float(grid.get_trafos()[0].limit_a1_ka))  # configured limit


def test_changed_branch_results_read_only_their_rows(monkeypatch: pytest.MonkeyPatch):
    """The violations, counts and max loading of the rows that change a branch recompute it from
    those rows' voltages only: never every row's (n_rows x n_bus on the host, 16 GB on France
    200 x 800)."""
    from gpusim2grid import ContingencyAnalysisGPU

    grid = with_current_limits(case14(), 0.25)
    ca = make_ca(grid, {}, batch_size=BATCH, compute_limit_violations=True)
    ca.add_contingencies_by_branch_id([[], [3]])
    ca.set_actions(_actions(grid))
    ca.add_strategies([(c, [a]) for c in range(2) for a in range(4)])
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    expected = (ca.get_violation_counts(), ca.get_violations(), ca.get_max_loading())

    def every_row(self: ContingencyAnalysisGPU) -> Any:
        raise AssertionError("every row's voltages copied to the host")

    monkeypatch.setattr(ContingencyAnalysisGPU, "V_results", property(every_row))
    counts, violations, (rho, branch) = ca.get_violation_counts(), ca.get_violations(), ca.get_max_loading()
    for k in counts:
        np.testing.assert_array_equal(counts[k], expected[0][k])
    assert [len(r) for r in violations] == [len(r) for r in expected[1]]
    np.testing.assert_array_equal(rho, expected[2][0])
    np.testing.assert_array_equal(branch, expected[2][1])


def test_result_rows_select_the_rows():
    """``ResultRows.CONTINGENCIES`` / ``STRATEGIES``: the N-1 rows / the strategy rows of every
    violation query, as the same rows of ``ALL`` (rows that change a branch included)."""
    from gpusim2grid import ResultRows

    grid = with_current_limits(case14(), 0.25)
    ca = make_ca(grid, {}, batch_size=BATCH, compute_limit_violations=True)
    ca.add_contingencies_by_branch_id([[], [3]])
    ca.set_actions(_actions(grid))
    ca.add_strategies([(c, [a]) for c in range(2) for a in range(4)])
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    every = slice(None)
    for rows, part in ((ResultRows.CONTINGENCIES, slice(0, 2)), (ResultRows.STRATEGIES, slice(2, None))):
        counts, all_counts = ca.get_violation_counts(rows), ca.get_violation_counts()
        for k in all_counts:
            np.testing.assert_array_equal(counts[k], all_counts[k][part])
        assert ca.get_violations(rows) == ca.get_violations(ResultRows.ALL)[part]
        np.testing.assert_array_equal(ca.get_violations_truncated(rows), ca.get_violations_truncated()[part])
    assert ca.get_violations(ResultRows.ALL) == ca.get_violations()[every]
    assert sum(len(r) for r in ca.get_violations(ResultRows.STRATEGIES)) > 0
