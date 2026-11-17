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


@pytest.mark.parametrize("settings", ca_settings())
def test_move_needing_no_new_entry_runs(settings: dict[str, Any], solver_atol: float):
    """A grid where generator 2 stands on bus 2 of substation 5 with a branch end; the action
    moves the generator alone back onto bus 5: no spare bus and no Ybus entry the base pattern
    lacks (the union plan is empty), the row still runs on the base pattern and is lightsim2grid's
    one-off solve (bus 5 PV at the generator's set-point, bus 2 PQ)."""
    from gpusim2grid.actions import Action

    grid = case14()
    gen = grid.get_generators()[2]
    sub: int = int(gen.bus_id)
    split = _move(grid, sub, [("gen_bus", gen.id), _branch_ends(grid, sub)[0]])
    split.apply_to(grid)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), 30, 1e-10).size > 0
    back = Action(gen_bus={gen.id: sub})
    ca, V = _run(grid, settings, [[]], [back], [(0, [0])])
    assert not ca.get_batch_unsupported_reasons()[1]
    assert ca.n_spare_buses == 0
    V_ref, _ = oracle(grid, action=back)
    assert_row_matches(V[1], V_ref, solver_atol)
    np.testing.assert_allclose(abs(V[1, sub]), gen.target_vm_pu, atol=solver_atol)


#: case14: generator 2 regulates its own bus 5 at 1.07 pu
GEN_B5, BUS_B5, VM_B5 = 2, 5, 1.07


def _case14_second_regulator(kind: str) -> LSGrid:
    """The two-busbar case14 with a second machine holding bus 5 with generator 2, at its
    set-point: generator 3 (bus 7) regulating it remotely (``"remote"``), or a storage unit on
    bus 5 regulating it (``"storage"``)."""
    import warnings

    import pandapower as pp
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    net = pn.case14()
    if kind == "storage":
        pp.create_storage(net, bus=BUS_B5, p_mw=0.0, max_e_mwh=10.0, q_mvar=0.0)
    n_sub: int = len(net.bus)
    for b in range(n_sub):
        pp.create_bus(net, vn_kv=net.bus.vn_kv.iloc[b], in_service=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net, n_sub=n_sub, n_busbar_per_sub=2)
    if kind == "remote":
        grid.set_gen_regulated_bus(3, BUS_B5)
        grid.change_v_gen(3, VM_B5)
    else:
        grid.init_storages_full(np.array([0.0]), np.array([0.0]), [True], np.array([VM_B5]),
                                np.array([-50.0]), np.array([50.0]), np.array([BUS_B5], dtype=np.int32))
    assert grid.get_generators()[GEN_B5].bus_id == BUS_B5
    assert np.isclose(grid.get_generators()[GEN_B5].target_vm_pu, VM_B5)
    grid.change_algorithm("NR_KLU")
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), 30, 1e-10).size > 0
    return grid


@pytest.mark.parametrize("kind", ["remote", "storage"])
def test_bus_another_machine_holds_stays_pv_when_its_generator_moves(kind: str, solver_atol: float):
    """Generator 2 leaves bus 5 for bus 2 of its substation (with a branch end), while another
    machine -- a remote regulator, a regulating storage unit -- still holds bus 5: bus 5 stays
    at its set-point (lightsim2grid's one-off solve), it is not released to PQ."""
    grid = _case14_second_regulator(kind)
    a = _move(grid, BUS_B5, [("gen_bus", GEN_B5), _branch_ends(grid, BUS_B5)[0]])
    ca, V = _run(grid, {}, [[]], [a], [(0, [0])])
    assert not ca.get_batch_unsupported_reasons()[1]
    V_ref, _ = oracle(grid, action=a)
    assert_row_matches(V[1], V_ref, solver_atol)
    np.testing.assert_allclose(abs(V[1, BUS_B5]), VM_B5, atol=solver_atol)
    np.testing.assert_allclose(abs(V[1, bus2(grid, BUS_B5)]), VM_B5, atol=solver_atol)


@pytest.mark.parametrize("kind", ["remote", "storage"])
def test_generator_moved_onto_a_bus_another_machine_holds_is_unsupported(kind: str):
    """In a grid where generator 2 stands on bus 2 of substation 5, moving it back onto bus 5,
    which a remote regulator / a storage unit holds, is refused per row, as onto a bus another
    generator holds."""
    from gpusim2grid.actions import Action

    grid = _case14_second_regulator(kind)
    split = _move(grid, BUS_B5, [("gen_bus", GEN_B5), _branch_ends(grid, BUS_B5)[0]])
    split.apply_to(grid)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), 30, 1e-10).size > 0
    back = Action(gen_bus={GEN_B5: BUS_B5})
    ca, _ = _run(grid, {}, [[]], [back], [(0, [0])])
    assert "regulator holds" in ca.get_batch_unsupported_reasons()[1]


def test_bus_released_when_its_contingency_trips_the_other_regulator(solver_atol: float):
    """The contingency trips the storage unit holding bus 5 with generator 2, the strategy moves
    generator 2 away: nothing holds bus 5 any more, it is PQ on that row (lightsim2grid's
    one-off solve with the storage unit off)."""
    grid = _case14_second_regulator("storage")
    a = _move(grid, BUS_B5, [("gen_bus", GEN_B5), _branch_ends(grid, BUS_B5)[0]])
    ca = make_ca(grid, {}, batch_size=BATCH)
    ca.add_contingency(storage_ids=[0])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    assert not ca.get_batch_unsupported_reasons()[1]
    V = batch_V_global(ca, 2, grid.total_bus())
    V_ref, _ = oracle(grid, action=a, apply=lambda g: g.deactivate_storage(0))
    assert_row_matches(V[1], V_ref, solver_atol)


#: case14 line 5 (buses 2 - 3): its side 2 stands on substation SUB
LINE_2_3: int = 5


@pytest.mark.parametrize("settings", ca_settings())
@pytest.mark.parametrize("opened_by", ["action", "contingency"])
def test_branch_opened_at_one_end_and_moved_at_the_other(
    settings: dict[str, Any], opened_by: str, solver_atol: float
):
    """Line 5 is opened at its side 1 (by the strategy's action, or by its contingency) and its
    side 2 moved to the new bus of substation SUB (with the split of ``_split_sub3``): its
    Kron-reduced charging lands on the new bus, as in lightsim2grid's one-off solve, and its
    side-2 current is read there -- the move is not dropped."""
    from gpusim2grid.actions import Action

    grid = case14()
    assert grid.get_lines()[LINE_2_3].bus2_id == SUB
    split = _split_sub3(grid)
    moved = Action(line_bus2={LINE_2_3: bus2(grid, SUB)})
    opened = Action(line_bus1={LINE_2_3: -1})
    ca = make_ca(grid, settings, batch_size=BATCH)
    if opened_by == "action":
        ca.add_contingencies_by_branch_id([[]])
        ca.set_actions([Action.combine([split, moved, opened])])
    else:
        ca.add_contingency(branch_ids_side1=[LINE_2_3])
        ca.set_actions([Action.combine([split, moved])])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    assert not ca.get_batch_unsupported_reasons()[1]
    V = batch_V_global(ca, 2, grid.total_bus())
    V_ref, ref = oracle(grid, action=Action.combine([split, moved, opened]))
    assert not ref.get_lines()[LINE_2_3].connected1 and ref.get_lines()[LINE_2_3].bus2_id == bus2(grid, SUB)
    assert_row_matches(V[1], V_ref, solver_atol)
    ca.compute_flows()
    or_ka = ca.or_amps.to_numpy().reshape(2, ca.n_branches)[1] * 1e-3
    ex_ka = ca.ex_amps.to_numpy().reshape(2, ca.n_branches)[1] * 1e-3
    assert or_ka[LINE_2_3] == 0.0
    np.testing.assert_allclose(ex_ka[LINE_2_3], ref.get_line_res2()[3][LINE_2_3], atol=10 * solver_atol, rtol=0)


def test_branch_end_opened_and_moved_by_two_actions_is_unsupported():
    """One action opens line 5 at side 2, another moves that same end: the row cannot decide,
    it is refused, not run with either."""
    from gpusim2grid.actions import Action

    grid = case14()
    split = _split_sub3(grid)
    moved = Action(line_bus2={LINE_2_3: bus2(grid, SUB)})
    ca, _ = _run(grid, {}, [[]], [Action.combine([split, moved]), Action(line_bus2={LINE_2_3: -1})], [(0, [0, 1])])
    assert "line_bus2[5]" in ca.get_batch_unsupported_reasons()[1]


@pytest.mark.parametrize("opened_by", ["contingency", "action"])
def test_bus_left_with_tripped_ends_only_is_empty(opened_by: str, solver_atol: float):
    """Line 3 (buses 1 - 3) is tripped (by the contingency, or opened at its bus-3 end by the
    action), and the action moves every other element of bus 3 to its bus 2: bus 3 holds nothing
    connected any more, it is out of the row (lightsim2grid deactivates it), not an island that
    would skip the row without handle_disconnected_grid."""
    from gpusim2grid.actions import Action

    grid = case14()
    line: int = 3
    assert grid.get_lines()[line].bus2_id == SUB
    others = [e for e in _branch_ends(grid, SUB) if e != ("line_bus2", line)]
    loads = [("load_bus", ld.id) for ld in grid.get_loads() if ld.bus_id == SUB]
    move = _move(grid, SUB, others + loads)
    ca = make_ca(grid, {}, batch_size=BATCH)
    if opened_by == "contingency":
        ca.add_contingencies_by_branch_id([[line]])
        ca.set_actions([move])
        ref_action = Action.combine([move, Action(line_bus1={line: -1}, line_bus2={line: -1})])
    else:
        ca.add_contingencies_by_branch_id([[]])
        ref_action = Action.combine([move, Action(line_bus2={line: -1})])
        ca.set_actions([ref_action])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    assert not ca.get_batch_unsupported_reasons()[1]
    assert np.isfinite(ca.last_residuals()[1])
    V = batch_V_global(ca, 2, grid.total_bus())
    V_ref, ref = oracle(grid, action=ref_action)
    assert not ref.get_bus_status()[SUB]
    assert_row_matches(V[1], V_ref, solver_atol)


@pytest.mark.parametrize("settings", ca_settings())
def test_current_counts_of_moved_branches_past_the_capacity(settings: dict[str, Any]):
    """Current limits at half the base case currents: every branch side violates, far more than
    violation_capacity (2). The row's exact CURRENT count is lightsim2grid's -- the moved
    branch counted once, at its new bus --, its records keep the capacity's most severe and the
    row is reported truncated."""
    grid = with_current_limits(case14(), 0.5)
    a = _split_sub3(grid)
    ca = make_ca(grid, settings, batch_size=BATCH, compute_limit_violations=True)
    ca.solver.violation_capacity = 2
    ca.add_contingencies_by_branch_id([[]])
    ca.set_actions([a])
    ca.add_strategies([(0, [0])])
    ca.compute(batch_size=BATCH)
    _, ref = oracle(grid, action=a)
    i1: np.ndarray = np.concatenate([ref.get_line_res1()[3], ref.get_trafo_res1()[3]])
    i2: np.ndarray = np.concatenate([ref.get_line_res2()[3], ref.get_trafo_res2()[3]])
    base1: np.ndarray = np.concatenate([grid.get_line_res1()[3], grid.get_trafo_res1()[3]])
    base2: np.ndarray = np.concatenate([grid.get_line_res2()[3], grid.get_trafo_res2()[3]])
    lim1, lim2 = 0.5 * np.maximum(base1, 1e-3), 0.5 * np.maximum(base2, 1e-3)
    n_ref: int = int(np.sum(i1 > lim1 * (1 + 1e-9)) + np.sum(i2 > lim2 * (1 + 1e-9)))
    assert n_ref > 2
    assert ca.get_violation_counts()["current"][1] == n_ref
    currents = [v for v in ca.get_violations()[1] if v.violation_type.name == "CURRENT"]
    assert len(currents) == 2
    assert ca.get_violations_truncated()[1]
    # the most severe two of lightsim2grid's
    sev = np.sort(np.concatenate([i1 / lim1, i2 / lim2]) - 1.0)[::-1]
    np.testing.assert_allclose(sorted((v.value / v.limit - 1.0 for v in currents), reverse=True), sev[:2], rtol=1e-6)


def _case14_two_shunts() -> LSGrid:
    """The two-busbar case14 (its shunt on bus 8) with a second shunt on bus 4 drawing active
    power too, so both parts of the admittance are exercised."""
    import pandapower as pp
    import pandapower.networks as pn

    from _topo_helpers import _two_busbars

    net = pn.case14()
    pp.create_shunt(net, bus=4, q_mvar=-6.0, p_mw=1.5)
    return _two_busbars(net)


def _shunt_admittance_reference(grid: LSGrid, shunt: int) -> complex:
    """lightsim2grid's own Ybus diagonal contribution of a shunt: the solver Ybus of the grid
    minus the one of a copy without the shunt (the definition, no convention to guess)."""
    import copy

    bus: int = int(grid.id_me_to_ac_solver()[grid.get_shunts()[shunt].bus_id])
    without = copy.deepcopy(grid)
    without.deactivate_shunt(shunt)
    without.tell_solver_need_reset()
    assert without.ac_pf(np.asarray(grid.get_V(), dtype=complex), 30, 1e-8).size > 0
    return complex(grid.get_Ybus_solver()[bus, bus] - without.get_Ybus_solver()[bus, bus])


def test_shunt_admittance_is_lightsim2grids():
    """The admittance a moved shunt carries is the one lightsim2grid stamps (p and q parts)."""
    from gpusim2grid.contingency_analysis._strategies import _shunt_admittance

    grid = _case14_two_shunts()
    assert len(grid.get_shunts()) == 2
    for s in range(len(grid.get_shunts())):
        y_ref = _shunt_admittance_reference(grid, s)
        assert abs(y_ref.real) > 0 or s == 0
        np.testing.assert_allclose(_shunt_admittance(grid, s), y_ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("settings", ca_settings())
def test_shunt_moves_match_lightsim2grid(settings: dict[str, Any], solver_atol: float):
    """Each shunt moved to bus 2 of its substation with a branch end, with and without a
    contingency: every row is lightsim2grid's one-off solve."""
    grid = _case14_two_shunts()
    actions = []
    for sh in grid.get_shunts():
        # its second branch end ([0] with contingency [7] islands part of the grid)
        actions.append(_move(grid, sh.bus_id, [("shunt_bus", sh.id), _branch_ends(grid, sh.bus_id)[1]]))
    ctgs = [[], [7]]
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    ca, V = _run(grid, settings, ctgs, actions, strategies)
    for r, (c, (a,)) in enumerate(strategies):
        assert not ca.get_batch_unsupported_reasons()[len(ctgs) + r]
        V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
        assert_row_matches(V[len(ctgs) + r], V_ref, solver_atol)
