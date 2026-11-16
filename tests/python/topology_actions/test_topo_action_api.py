# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``gpusim2grid.actions.Action``: the action format (docs/dev_notes/topology_actions_plan.md § 7).

Host only: building, validating and applying an action to an ``LSGrid`` (the oracle's own
semantics: ``apply_to`` makes the ``change_bus_*`` / ``deactivate_*`` / tap calls).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from _topo_helpers import bus2, case14, oracle
from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid


pytestmark = [requires_gpu]


def test_action_is_frozen_and_hashable():
    from gpusim2grid.actions import Action

    a = Action(load_bus={2: 17})
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.load_bus = {}
    assert hash(a) == hash(Action(load_bus={2: 17}))
    assert a == Action(load_bus={2: 17})
    assert a != Action(load_bus={2: 3})
    assert Action().is_empty()
    assert not a.is_empty()


def test_apply_to_makes_the_change_bus_calls():
    """apply_to on a copy == the same lightsim2grid calls written out."""
    from gpusim2grid.actions import Action

    grid = case14()
    sub = 3
    load = next(el.id for el in grid.get_loads() if el.bus_id == sub)
    line = next(el for el in grid.get_lines() if el.bus2_id == sub)
    a = Action(load_bus={load: bus2(grid, sub)}, line_bus2={line.id: bus2(grid, sub)})
    V_a, _ = oracle(grid, action=a)

    def by_hand(g: LSGrid) -> None:
        g.change_bus_load(load, bus2(grid, sub))
        g.change_bus2_powerline(line.id, bus2(grid, sub))

    V_h, _ = oracle(grid, apply=by_hand)
    np.testing.assert_array_equal(V_a, V_h)


def test_minus_one_disconnects():
    from gpusim2grid.actions import Action

    grid = case14()
    V_a, _ = oracle(grid, action=Action(line_bus1={5: -1}))
    V_h, _ = oracle(grid, apply=lambda g: g.deactivate_powerline(5))
    np.testing.assert_array_equal(V_a, V_h)


def test_trafo_values_apply():
    from gpusim2grid.actions import Action

    grid = case14()
    t = grid.get_trafos()[0]
    a = Action(trafo_shift_deg={0: 5.0}, trafo_ratio={0: t.ratio * 1.02})
    V_a, _ = oracle(grid, action=a)

    def by_hand(g: LSGrid) -> None:
        g.change_shift_trafo_deg(0, 5.0)
        g.change_ratio_trafo(0, t.ratio * 1.02)

    V_h, _ = oracle(grid, apply=by_hand)
    np.testing.assert_array_equal(V_a, V_h)


@pytest.mark.parametrize(
    "bad",
    [
        dict(load_bus={10_000: 0}),  # unknown element
        dict(line_bus1={0: 10_000}),  # bus out of range
        dict(line_bus1={0: 5}),  # a bus of another substation (line 0's end 1 is in sub 0)
        dict(trafo_ratio={0: -1.0}),  # non-physical ratio
    ],
)
def test_validate_refuses(bad: dict[str, Any]):
    from gpusim2grid.actions import Action

    with pytest.raises(ValueError):
        Action(**bad).validate(case14())


def test_validate_accepts_both_busbars_of_the_substation():
    from gpusim2grid.actions import Action

    grid = case14()
    Action(line_bus1={0: 0}).validate(grid)  # its own bus
    Action(line_bus1={0: bus2(grid, 0)}).validate(grid)  # local bus 2 of the same substation
    Action(line_bus1={0: -1}).validate(grid)  # disconnected


def test_from_partition():
    """The prototype's nodal action: a target partition of a substation. Each group keeps the
    bus holding most of its elements now; the others go to free local buses; listed
    disconnected elements get -1. Every element of the substation must be listed."""
    from gpusim2grid.actions import Action, substation_elements

    grid = case14()
    sub = 3
    elems = substation_elements(grid, sub)  # [("line_bus2", 3), ("trafo_bus1", 0), ...]
    assert len(elems) >= 4
    g1, g2 = elems[: len(elems) // 2], elems[len(elems) // 2 :]
    a = Action.from_partition(grid, sub, [g1, g2])
    moved = [e for e in g2 if e not in g1]
    for kind, el in moved:
        assert getattr(a, kind)[el] == bus2(grid, sub)
    for kind, el in g1:
        assert el not in getattr(a, kind) or getattr(a, kind)[el] == sub

    d = Action.from_partition(grid, sub, [elems[1:]], disconnected=[elems[0]])
    kind, el = elems[0]
    assert getattr(d, kind)[el] == -1

    with pytest.raises(ValueError):  # an element of the substation missing
        Action.from_partition(grid, sub, [elems[1:]])


def test_combine_two_actions():
    """A strategy of several actions: their union; the same element set twice to different
    buses is refused."""
    from gpusim2grid.actions import Action

    a = Action(line_bus1={0: -1})
    b = Action(load_bus={2: 17})
    c = Action.combine([a, b])
    assert c == Action(line_bus1={0: -1}, load_bus={2: 17})
    assert Action.combine([a, a]) == a
    with pytest.raises(ValueError):
        Action.combine([Action(load_bus={2: 17}), Action(load_bus={2: 3})])
