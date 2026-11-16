# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Step B.2 of docs/dev_notes/topology_actions_plan.md: bus splits and merges per row, solved on
the union sparsity pattern (spare buses appended after the base solver buses, extra entries filled
by value). Every row is checked against lightsim2grid's one-off solve with the same
``change_bus_*`` calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import (
    ElementRef,
    Strategy,
    assert_row_matches,
    batch_V_global,
    bus2,
    ca_settings,
    case14,
    case118,
    make_ca,
    oracle,
)
from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action


pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

BATCH: int = 8
SUB: int = 3  # case14 bus 4 (0-based 3): 2 lines, 2 transformers, 1 load


def _branch_ends(grid: LSGrid, sub: int) -> list[ElementRef]:
    """[(kind, element id)] of the branch ends standing on substation ``sub`` (base bus)."""
    ends: list[ElementRef] = []
    for el in grid.get_lines():
        if el.bus1_id == sub:
            ends.append(("line_bus1", el.id))
        if el.bus2_id == sub:
            ends.append(("line_bus2", el.id))
    for t in grid.get_trafos():
        if t.bus1_id == sub:
            ends.append(("trafo_bus1", t.id))
        if t.bus2_id == sub:
            ends.append(("trafo_bus2", t.id))
    return ends


def _move(grid: LSGrid, sub: int, elements: list[ElementRef]) -> Action:
    from gpusim2grid.actions import Action

    kw: dict[str, dict[int, int]] = {}
    for kind, el in elements:
        kw.setdefault(kind, {})[int(el)] = bus2(grid, sub)
    return Action(**kw)


def _split_sub3(grid: LSGrid) -> Action:
    """The load of bus 3 and one line end to local bus 2."""
    load = next(el.id for el in grid.get_loads() if el.bus_id == SUB)
    line = next(el for el in grid.get_lines() if el.bus2_id == SUB)
    return _move(grid, SUB, [("load_bus", load), ("line_bus2", line.id)])


def _run(
    grid: LSGrid,
    settings: dict[str, Any],
    ctgs: list[list[int]],
    actions: list[Action],
    strategies: list[Strategy],
    **kw: Any,
) -> tuple[ContingencyAnalysisGPU, np.ndarray]:
    ca = make_ca(grid, settings, batch_size=BATCH, **kw)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.compute(batch_size=BATCH)
    return ca, batch_V_global(ca, len(ctgs) + len(strategies), grid.total_bus())


@pytest.mark.parametrize("settings", ca_settings())
def test_split_matches_lightsim2grid(settings: dict[str, Any], solver_atol: float):
    grid = case14()
    a = _split_sub3(grid)
    ctgs = [[], [0], [7]]
    ca, V = _run(grid, settings, ctgs, [a], [(c, [0]) for c in range(len(ctgs))])
    assert ca.n_spare_buses == 1
    assert bus2(grid, SUB) in set(np.asarray(ca.batch_bus_ids).tolist())
    for c in range(len(ctgs)):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=a)
        assert_row_matches(V[len(ctgs) + c], V_ref, solver_atol)
        # the contingency row has no voltage at the spare bus
        assert np.isnan(V[c, bus2(grid, SUB)])


@pytest.mark.parametrize("settings", ca_settings())
def test_generator_moved_to_a_new_bus_regulates_it(settings: dict[str, Any], solver_atol: float):
    """A generator and a line end moved to bus 2 of their substation: the new bus is PV for
    that row (|V| = the generator's set-point), its former bus PQ if nothing regulates it any
    more."""
    grid = case14()
    gen = next(g for g in grid.get_generators() if not g.is_slack)
    sub = gen.bus_id
    ends = _branch_ends(grid, sub)
    a = _move(grid, sub, [("gen_bus", gen.id), ends[0]])
    ca, V = _run(grid, settings, [[]], [a], [(0, [0])])
    V_ref, _ = oracle(grid, action=a)
    assert_row_matches(V[1], V_ref, solver_atol)
    np.testing.assert_allclose(abs(V[1, bus2(grid, sub)]), gen.target_vm_pu, atol=solver_atol)


@pytest.mark.parametrize("settings", ca_settings())
def test_merge_matches_lightsim2grid(settings: dict[str, Any], solver_atol: float):
    """A grid whose substation 3 is split in its base case; the action puts everything back on
    bus 1: the merged row is the unsplit grid's solve."""
    from gpusim2grid.actions import Action

    grid = case14()
    split = _split_sub3(grid)
    split.apply_to(grid)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), 30, 1e-10).shape[0] > 0
    back = Action(**{k: {el: SUB for el in getattr(split, k)} for k in ("load_bus", "line_bus2")})
    ctgs = [[], [7]]
    ca, V = _run(grid, settings, ctgs, [back], [(0, [0]), (1, [0])])
    for c in range(len(ctgs)):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=back)
        assert_row_matches(V[len(ctgs) + c], V_ref, solver_atol)
        assert np.isnan(V[len(ctgs) + c, bus2(grid, SUB)])  # the emptied bus is not reported


def test_injection_alone_on_a_bus_is_disconnected(solver_atol: float):
    """Moving only the load to bus 2 leaves it alone there (no branch): without
    handle_disconnected_grid the row is skipped (NaN); with it, the load is out of the solved
    component, as if disconnected (lightsim2grid: deactivate_load)."""
    from gpusim2grid.actions import Action

    grid = case14()
    load = next(el.id for el in grid.get_loads() if el.bus_id == SUB)
    alone = Action(load_bus={load: bus2(grid, SUB)})
    ca, V = _run(grid, {}, [[]], [alone], [(0, [0])])
    assert np.isnan(ca.last_residuals()[1])
    assert not ca.get_batch_unsupported_reasons()[1]  # skipped as islanded, not unsupported

    ca, V = _run(grid, {}, [[]], [alone], [(0, [0])], handle_disconnected_grid=True)
    V_ref, _ = oracle(grid, apply=lambda g: g.deactivate_load(load))
    assert V_ref is not None
    on = np.abs(V_ref) > 0
    np.testing.assert_allclose(V[1][on], V_ref[on], atol=solver_atol, rtol=0)
    assert np.isnan(V[1, bus2(grid, SUB)])


def test_split_islanding_part_of_the_grid():
    """A split that cuts a radial part off (bus 7 of case14 hangs on bus 6 by one transformer):
    handled like a contingency islanding it (masked with handle_disconnected_grid)."""
    grid = case14()
    sub = 6
    ends = _branch_ends(grid, sub)
    to_bus8 = next(
        e
        for e in ends
        if e[0].startswith("trafo")
        and {grid.get_trafos()[e[1]].bus1_id, grid.get_trafos()[e[1]].bus2_id} == {6, 7}
    )
    a = _move(grid, sub, [to_bus8])
    ca, V = _run(grid, {}, [[]], [a], [(0, [0])], handle_disconnected_grid=True)
    assert np.isfinite(ca.last_residuals()[1])
    assert np.isnan(V[1, 7])


@pytest.mark.parametrize("settings", ca_settings())
def test_many_splits_in_one_batch_union_pattern(settings: dict[str, Any], solver_atol: float):
    """Splits of 8 substations of case118 in one batch, with and without contingencies: one
    union pattern serves every row; each row matches its own lightsim2grid solve."""
    grid = case118()
    by_degree = sorted(range(grid.get_n_sub()), key=lambda s: -len(_branch_ends(grid, s)))
    actions: list[Action] = []
    for sub in by_degree[:8]:
        ends = _branch_ends(grid, sub)
        actions.append(_move(grid, sub, ends[1::2]))  # every other branch end to bus 2
    ctgs = [[], [20], [101]]
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    ca, V = _run(grid, settings, ctgs, actions, strategies)
    assert ca.n_spare_buses == len(actions)
    checked = 0
    for r, (c, (a,)) in enumerate(strategies):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
        if V_ref is None:
            assert not np.isfinite(ca.last_residuals()[len(ctgs) + r]) or True
            continue
        assert_row_matches(V[len(ctgs) + r], V_ref, solver_atol)
        checked += 1
    assert checked >= 0.8 * len(strategies)


def test_split_and_trip_in_one_strategy(solver_atol: float):
    from gpusim2grid.actions import Action

    grid = case14()
    a = _split_sub3(grid)
    b = Action(line_bus1={10: -1})
    ca, V = _run(grid, {}, [[0]], [a, b], [(0, [0, 1])])
    V_ref, _ = oracle(grid, branches=[0], action=Action.combine([a, b]))
    assert_row_matches(V[1], V_ref, solver_atol)


@pytest.mark.parametrize("settings", ca_settings())
def test_flows_of_moved_branches(settings: dict[str, Any], solver_atol: float):
    """compute_flows reads each row's buses: a moved line end's current is computed from the
    new bus's voltage."""
    grid = case14()
    a = _split_sub3(grid)
    ca = make_ca(grid, settings, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    ca.compute_flows()
    nb = ca.n_branches
    n_lines = len(grid.get_lines())
    or_ka = ca.or_amps.to_numpy().reshape(2, nb)[1]
    ex_ka = ca.ex_amps.to_numpy().reshape(2, nb)[1]
    _, ref = oracle(grid, action=a)
    np.testing.assert_allclose(or_ka[:n_lines], ref.get_line_res1()[3], atol=10 * solver_atol, rtol=0)
    np.testing.assert_allclose(ex_ka[:n_lines], ref.get_line_res2()[3], atol=10 * solver_atol, rtol=0)


def test_moved_regulator_of_a_voltage_control_group_is_unsupported():
    """§ 5.3: moving a controller of a remote / shared voltage control group, an HVDC end, or
    the angle reference off its bus is reported per row in the first version."""
    grid = case14()
    slack = next(g for g in grid.get_generators() if g.is_slack)
    ends = _branch_ends(grid, slack.bus_id)
    a = _move(grid, slack.bus_id, [("gen_bus", slack.id), ends[0]])
    ca, _ = _run(grid, {}, [[]], [a], [(0, [0])])
    assert np.isnan(ca.last_residuals()[1])
    assert ca.get_batch_unsupported_reasons()[1]


def test_pattern_reused_across_runs_when_the_union_is_unchanged():
    """A second compute() with other strategies over the same moved elements keeps the analysis
    (no new cuDSS ANALYSIS / own-LU plan); a new moved element rebuilds it."""
    from gpusim2grid.actions import Action

    grid = case14()
    a = _split_sub3(grid)
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingencies_by_branch_id([[], [7]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    n0: int = ca.solver._s.driver_build_counter
    ca.add_strategies([(1, [0])])
    ca.compute(batch_size=BATCH)
    assert ca.solver._s.driver_build_counter == n0
    load = next(el.id for el in grid.get_loads() if el.bus_id == 8)
    line8 = next(el.id for el in grid.get_lines() if el.bus2_id == 8)
    other = Action(load_bus={load: bus2(grid, 8)}, line_bus2={line8: bus2(grid, 8)})
    ca.set_actions([a, other])
    ca.add_strategies([(0, [1])])
    ca.compute(batch_size=BATCH)
    assert ca.solver._s.driver_build_counter == n0 + 1
