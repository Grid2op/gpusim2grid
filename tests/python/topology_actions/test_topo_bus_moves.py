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
    with_current_limits,
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
    # contingencies on lines no action moves (a strategy moving the line its contingency trips
    # would put it back: reported per row, test_topo_trips covers that)
    moved: set[int] = {el for a in actions for el in (*a.line_bus1, *a.line_bus2)}
    ctgs = [[]] + [[b] for b in (20, 101, 30, 64, 150) if b not in moved][:2]
    assert len(ctgs) == 3
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    ca, V = _run(grid, settings, ctgs, actions, strategies)
    assert ca.n_spare_buses == len(actions)
    checked = 0
    for r, (c, (a,)) in enumerate(strategies):
        V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
        if V_ref is None:  # lightsim2grid diverges on this row: not compared
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
    or_ka = ca.or_amps.to_numpy().reshape(2, nb)[1] * 1e-3  # A -> kA, as lightsim2grid's results
    ex_ka = ca.ex_amps.to_numpy().reshape(2, nb)[1] * 1e-3
    _, ref = oracle(grid, action=a)
    np.testing.assert_allclose(or_ka[:n_lines], ref.get_line_res1()[3], atol=10 * solver_atol, rtol=0)
    np.testing.assert_allclose(ex_ka[:n_lines], ref.get_line_res2()[3], atol=10 * solver_atol, rtol=0)



@pytest.mark.parametrize("settings", ca_settings())
def test_loading_and_current_violations_of_moved_branches(settings: dict[str, Any]):
    """get_max_loading and the current violations read a moved branch's current at its new bus:
    the row's loading and its violation count are lightsim2grid's."""
    grid = with_current_limits(case14(), 1.05)
    a = _split_sub3(grid)
    ca = make_ca(grid, settings, batch_size=BATCH, compute_limit_violations=True)
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    _, ref = oracle(grid, action=a)
    i1: np.ndarray = np.concatenate([ref.get_line_res1()[3], ref.get_trafo_res1()[3]])
    i2: np.ndarray = np.concatenate([ref.get_line_res2()[3], ref.get_trafo_res2()[3]])
    base1: np.ndarray = np.concatenate([grid.get_line_res1()[3], grid.get_trafo_res1()[3]])
    lim1: np.ndarray = 1.05 * np.maximum(base1, 1e-3)
    base2: np.ndarray = np.concatenate([grid.get_line_res2()[3], grid.get_trafo_res2()[3]])
    lim2: np.ndarray = 1.05 * np.maximum(base2, 1e-3)
    rho_ref: np.ndarray = np.fmax(i1 / lim1, i2 / lim2)
    rho, branch = ca.get_max_loading()
    np.testing.assert_allclose(rho[1], np.nanmax(rho_ref), rtol=1e-9)
    assert branch[1] == int(np.nanargmax(rho_ref))
    n_ref: int = int(np.sum(i1 > lim1 * (1 + 1e-9)) + np.sum(i2 > lim2 * (1 + 1e-9)))
    assert ca.get_violation_counts()["current"][1] == n_ref
    assert sum(v.violation_type.name == "CURRENT" for v in ca.get_violations()[1]) == n_ref

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


def test_generator_moves_lightsim2grid_refuses_are_unsupported():
    """The slack generator and a generator taking part in the slack do not move (lightsim2grid's
    ScenarioSweep refuses them too): their strategy rows are not run, with the reason."""
    grid = case14()
    slack = next(g for g in grid.get_generators() if g.is_slack)
    a = _move(grid, slack.bus_id, [("gen_bus", slack.id)])
    ca, _ = _run(grid, {}, [[]], [a], [(0, [0])])
    reasons = ca.get_batch_unsupported_reasons()
    assert "slack" in reasons[1], reasons



@pytest.mark.parametrize("settings", ca_settings())
def test_closed_end_of_a_half_open_branch_moves(settings: dict[str, Any], solver_atol: float):
    """A line open at one end in the base case: its closed end moves to bus 2 of its substation
    with another branch end (its admittance, Kron-reduced onto the closed end, moves with it);
    voltages and that end's current are lightsim2grid's."""
    from gpusim2grid.actions import Action

    grid = case14()
    # a line whose side-1 substation keeps a meshed bus 1 once two of its branch ends leave
    line: int = next(ln.id for ln in grid.get_lines() if len(_branch_ends(grid, ln.bus1_id)) >= 4)
    Action(line_bus2={line: -1}).apply_to(grid)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), 30, 1e-8).size > 0
    sub: int = grid.get_lines()[line].bus1_id
    others = [e for e in _branch_ends(grid, sub) if e != ("line_bus1", line)]
    a = _move(grid, sub, [("line_bus1", line), others[0]])
    ca, V = _run(grid, settings, [[]], [a], [(0, [0])])
    assert not ca.get_batch_unsupported_reasons()[1]
    V_ref, ref = oracle(grid, action=a)
    assert_row_matches(V[1], V_ref, solver_atol)
    ca.compute_flows()
    or_ka = ca.or_amps.to_numpy().reshape(2, ca.n_branches)[1] * 1e-3
    np.testing.assert_allclose(or_ka[line], ref.get_line_res1()[3][line], atol=10 * solver_atol, rtol=0)


def _case14_with_group() -> LSGrid:
    """The two-busbar case14 with a second generator on bus 1, at the same set-point: the two
    regulate that bus together (a voltage-control group)."""
    import pandapower as pp
    import pandapower.networks as pn

    from _topo_helpers import _two_busbars

    net = pn.case14()
    first = net.gen.index[net.gen.bus == 1][0]
    pp.create_gen(net, bus=1, p_mw=10.0, vm_pu=float(net.gen.vm_pu[first]), min_q_mvar=-50.0, max_q_mvar=50.0)
    return _two_busbars(net)


@pytest.mark.parametrize("settings", ca_settings())
def test_generator_of_a_group_moved_to_a_new_bus(settings: dict[str, Any], solver_atol: float):
    """One of two generators regulating bus 1 together moves to bus 2 of the substation with a
    branch end: it holds the new bus alone, the other one regulates bus 1 on (lightsim2grid's
    one-off solve; its ScenarioSweep refuses this move)."""
    grid = _case14_with_group()
    on_bus1 = [g for g in grid.get_generators() if g.bus_id == 1 and not g.is_slack]
    assert len(on_bus1) == 2
    gen = on_bus1[1]
    a = _move(grid, 1, [("gen_bus", gen.id), _branch_ends(grid, 1)[0]])
    ca, V = _run(grid, settings, [[]], [a], [(0, [0])])
    assert not ca.get_batch_unsupported_reasons()[1]
    V_ref, _ = oracle(grid, action=a)
    assert_row_matches(V[1], V_ref, solver_atol)
    np.testing.assert_allclose(abs(V[1, bus2(grid, 1)]), gen.target_vm_pu, atol=solver_atol)
    np.testing.assert_allclose(abs(V[1, 1]), on_bus1[0].target_vm_pu, atol=solver_atol)


@pytest.mark.parametrize("threshold", [1.0, 0.95], ids=["t1", "t095"])
@pytest.mark.parametrize("reread", [False, True], ids=["ctor_limits", "set_limits_from_grid_after_union"])
def test_spare_bus_has_its_substation_voltage_limits(threshold: float, reread: bool):
    """A spare bus is a busbar of its substation: it is checked against the voltage limits of
    its parent bus (lightsim2grid's limits of that voltage level), at ``violation_threshold``
    too, and its records carry the configured limit -- also when the limits are read off the
    grid again once the union is live."""
    grid = case14()
    vn: np.ndarray = np.asarray(grid.get_bus_vn_kv(), dtype=float)
    vmin, vmax = 0.5 * vn, 1.5 * vn
    for b in (SUB, bus2(grid, SUB)):  # every bus of substation SUB below its vmin
        vmin[b], vmax[b] = 1.2 * vn[b], 1.3 * vn[b]
    grid.set_bus_voltage_limits(vmin, vmax)
    a = _split_sub3(grid)
    ca = make_ca(grid, {}, batch_size=BATCH, compute_limit_violations=True, violation_threshold=threshold)
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    if reread:
        ca.set_limits_from_grid()
        ca.compute(batch_size=BATCH)
    spare: int = int(np.flatnonzero(np.asarray(ca.batch_bus_ids) == bus2(grid, SUB))[0])
    low = {v.element_id: v for v in ca.get_violations()[1] if v.violation_type.name == "LOW_VOLTAGE"}
    assert set(low) == {SUB, spare}, low
    np.testing.assert_allclose(low[spare].limit, vmin[bus2(grid, SUB)], rtol=1e-12)
    V_ref, _ = oracle(grid, action=a)
    assert V_ref is not None
    np.testing.assert_allclose(low[spare].value, abs(V_ref[bus2(grid, SUB)]) * vn[SUB], rtol=1e-6)
    assert ca.get_violation_counts()["low_voltage"][1] == 2
    # the contingency row has no spare bus: only bus SUB
    assert {v.element_id for v in ca.get_violations()[0] if v.violation_type.name == "LOW_VOLTAGE"} == {SUB}
