# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Step A of docs/dev_notes/topology_actions_plan.md: contingency x action rows whose actions
only disconnect elements. A strategy row is the union of its contingency's and its actions'
trips: the same row as one contingency tripping all of them, and lightsim2grid's one-off solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import (
    assert_row_matches,
    batch_V,
    batch_V_global,
    branch_id_of_trafo,
    ca_settings,
    case14,
    case118,
    make_ca,
    oracle,
)
from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

    from gpusim2grid.actions import Action


pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

BATCH: int = 8


def _actions() -> list[Action]:
    from gpusim2grid.actions import Action

    grid = case14()
    gen = next(g for g in grid.get_generators() if not g.is_slack)
    return [
        Action(line_bus1={5: -1}),  # a line, by its end 1
        Action(line_bus2={9: -1}),  # a line, by its end 2
        Action(trafo_bus1={2: -1}),  # a transformer
        Action(load_bus={4: -1}),  # a load
        Action(gen_bus={gen.id: -1}),  # a generator (its bus turns PQ)
        Action(line_bus1={5: -1}, load_bus={4: -1}),  # two elements
    ]


def _equivalent_trips(grid: LSGrid, ctg_branches: list[int], action: Action) -> dict[str, list[int]]:
    """The ``add_contingency`` arguments of one contingency tripping the same elements: a
    branch opened at one end is ``branch_ids_side1`` / ``_side2``, at both ends a trip."""
    side1: set[int] = {int(i) for i, b in action.line_bus1.items() if b == -1}
    side1 |= {branch_id_of_trafo(grid, i) for i, b in action.trafo_bus1.items() if b == -1}
    side2: set[int] = {int(i) for i, b in action.line_bus2.items() if b == -1}
    side2 |= {branch_id_of_trafo(grid, i) for i, b in action.trafo_bus2.items() if b == -1}
    both: set[int] = set(ctg_branches) | (side1 & side2)
    return {
        "branch_ids": sorted(both),
        "branch_ids_side1": sorted(side1 - both),
        "branch_ids_side2": sorted(side2 - both),
        "load_ids": sorted(int(i) for i, b in action.load_bus.items() if b == -1),
        "gen_ids": sorted(int(i) for i, b in action.gen_bus.items() if b == -1),
    }


@pytest.mark.parametrize("settings", ca_settings())
def test_strategy_rows_equal_the_union_of_trips(settings: dict[str, Any]):
    grid = case14()
    ctgs = [[0], [7], [13]]
    actions = _actions()
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]

    ca = make_ca(grid, settings, batch_size=BATCH)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    assert ca.n_strategies == len(strategies)
    ca.compute(batch_size=BATCH)
    V = batch_V(ca, len(ctgs) + len(strategies))

    ref = make_ca(grid, settings, batch_size=BATCH)
    for c, (a,) in strategies:
        ref.add_contingency(**_equivalent_trips(grid, ctgs[c], actions[a]))
    ref.compute(batch_size=BATCH)
    V_ref = batch_V(ref, len(strategies))
    np.testing.assert_allclose(V[len(ctgs) :], V_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("settings", ca_settings())
def test_strategy_rows_match_lightsim2grid(settings: dict[str, Any], solver_atol: float):
    grid = case14()
    ctgs = [[0], [7]]
    actions = _actions()
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    ca = make_ca(grid, settings, batch_size=BATCH)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.compute(batch_size=BATCH)
    V = batch_V_global(ca, len(ctgs) + len(strategies), grid.total_bus())
    for r, (c, (a,)) in enumerate(strategies):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
        assert V_ref is not None
        assert_row_matches(V[len(ctgs) + r], V_ref, solver_atol)


@pytest.mark.parametrize("settings", ca_settings())
def test_element_tripped_twice_counts_once(settings: dict[str, Any], solver_atol: float):
    """The contingency trips line 0 and the action disconnects it too: tripped once (two
    summed patches would subtract its admittance twice)."""
    from gpusim2grid.actions import Action

    grid = case14()
    ca = make_ca(grid, settings, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[0]])
    ca.set_actions([Action(line_bus1={0: -1})])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    V = batch_V(ca, 2)
    np.testing.assert_allclose(V[1], V[0], atol=1e-12, rtol=0)


def test_reconnecting_a_tripped_element_is_unsupported():
    """An action putting back an element its contingency trips (here: line 0 on its bus) is
    reported for that row, not solved and not raised."""
    from gpusim2grid.actions import Action

    grid = case14()
    line = grid.get_lines()[0]
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[0], [3]])
    ca.set_actions([Action(line_bus1={0: int(line.bus1_id)})])
    ca.add_strategies([(0, [0]), (1, [0])])
    ca.compute(batch_size=BATCH)
    res = ca.last_residuals()
    assert np.isnan(res[2]) and np.isfinite(res[3])
    reasons = ca.get_batch_unsupported_reasons()
    assert reasons[2] and not reasons[3]


def test_two_actions_in_one_strategy(solver_atol: float):
    grid = case14()
    actions = _actions()
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[7]])
    ca.set_actions(actions)
    ca.add_strategies([(0, [0, 3])])
    ca.compute(batch_size=BATCH)
    from gpusim2grid.actions import Action

    V = batch_V_global(ca, 2, grid.total_bus())
    V_ref, _ = oracle(grid, branches=[7], action=Action.combine([actions[0], actions[3]]))
    assert_row_matches(V[1], V_ref, solver_atol)


def test_add_strategies_takes_arrays():
    """The product without a Python call per row: (ctg index, action index) arrays, 1 action
    per strategy; rows in the order given."""
    grid = case118()
    from gpusim2grid.actions import Action

    n_lines = len(grid.get_lines())
    ctgs = [[i] for i in range(0, 40, 4)]
    actions = [Action(line_bus1={j: -1}) for j in range(100, n_lines, 13)]
    ca = make_ca(grid, {}, batch_size=64)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    c_idx, a_idx = np.meshgrid(np.arange(len(ctgs)), np.arange(len(actions)), indexing="ij")
    ca.add_strategies(c_idx.ravel(), a_idx.ravel())
    assert ca.n_strategies == len(ctgs) * len(actions)
    ca.compute(batch_size=64)
    assert ca.last_residuals().shape == (len(ctgs) + ca.n_strategies,)
