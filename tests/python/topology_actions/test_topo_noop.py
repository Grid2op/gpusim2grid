# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Principle 1 of docs/dev_notes/topology_actions_plan.md: a batch that does not use the
topology actions takes exactly today's path -- same pattern, same analysis, same voltages to the
bit -- and a strategy whose action changes nothing is its contingency's row (up to rounding)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import Strategy, batch_V, ca_settings, case14, make_ca
from conftest import requires_gpu

if TYPE_CHECKING:
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action


pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

CTGS: list[list[int]] = [[0], [3], [7], [12]]
BATCH: int = 4


def _run(
    settings: dict[str, Any], actions: list[Action] | None = None, strategies: list[Strategy] | None = None
) -> tuple[ContingencyAnalysisGPU, np.ndarray]:
    grid = case14()
    # every row from the base case: the rows themselves are what these tests compare
    ca = make_ca(grid, settings, batch_size=BATCH, strategy_warm_start=False)
    ca.add_contingencies_by_branch_id(CTGS)
    if actions is not None:
        ca.set_actions(actions)
    if strategies is not None:
        ca.add_strategies(strategies)
    ca.compute(batch_size=BATCH)
    n = len(CTGS) + (len(strategies) if strategies else 0)
    return ca, batch_V(ca, n)


@pytest.mark.parametrize("settings", ca_settings())
def test_empty_actions_take_todays_path(settings: dict[str, Any]):
    ca0, V0 = _run(settings)
    ca1, V1 = _run(settings, actions=[], strategies=[])
    # today's code path; two sessions agree up to rounding only (gpusim2grid's first session of a
    # process differs from the later ones in the last bits, ~3e-15, with no action at all)
    np.testing.assert_allclose(V1, V0, atol=1e-12, rtol=0)
    assert ca1.n_spare_buses == 0
    assert ca1.n_strategies == 0


@pytest.mark.parametrize("settings", ca_settings())
def test_no_spare_bus_without_moves(settings: dict[str, Any]):
    """Disconnections and branch values never extend the pattern."""
    from gpusim2grid.actions import Action

    actions = [Action(line_bus1={5: -1}), Action(trafo_shift_deg={0: 3.0}), Action(load_bus={4: -1})]
    ca, _ = _run(settings, actions=actions, strategies=[(0, [0]), (1, [1]), (2, [2])])
    assert ca.n_spare_buses == 0
    np.testing.assert_array_equal(
        np.asarray(ca.batch_bus_ids), np.asarray(case14().id_ac_solver_to_me(), dtype=int)
    )


@pytest.mark.parametrize("settings", ca_settings())
def test_action_keeping_every_bus_is_its_contingency(settings: dict[str, Any]):
    """An action that sets elements to the bus they are on: the strategy row is its
    contingency row (up to rounding: another slot), and the pattern is not extended."""
    from gpusim2grid.actions import Action

    grid = case14()
    load = grid.get_loads()[2]
    line = grid.get_lines()[4]
    same = Action(load_bus={load.id: load.bus_id}, line_bus1={line.id: line.bus1_id})
    ca, V = _run(settings, actions=[same], strategies=[(c, [0]) for c in range(len(CTGS))])
    assert ca.n_spare_buses == 0
    # solved in another slot / chunk: equal up to rounding (the batched factorization does not
    # promise the last bit across chunks, nor the continuous tail across slots)
    np.testing.assert_allclose(V[len(CTGS) :], V[: len(CTGS)], atol=1e-12, rtol=0)


@pytest.mark.parametrize("settings", ca_settings())
def test_contingency_rows_unchanged_by_strategies(settings: dict[str, Any]):
    """Adding strategies (even with bus moves, which extend the pattern) does not change the
    contingency rows beyond rounding: the extra entries are exact zeros for them."""
    from gpusim2grid.actions import Action

    from _topo_helpers import bus2

    grid = case14()
    _, V0 = _run(settings)
    sub = 3
    load = next(el.id for el in grid.get_loads() if el.bus_id == sub)
    line = next(el for el in grid.get_lines() if el.bus2_id == sub)
    split = Action(load_bus={load: bus2(grid, sub)}, line_bus2={line.id: bus2(grid, sub)})
    ca, V = _run(settings, actions=[split], strategies=[(0, [0])])
    assert ca.n_spare_buses == 1
    n0 = V0.shape[1]
    np.testing.assert_allclose(V[: len(CTGS), :n0], V0, atol=1e-12, rtol=0)
