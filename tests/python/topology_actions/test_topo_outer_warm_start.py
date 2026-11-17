# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Strategies with the outer loops, warm-started from their contingency (``strategy_warm_start``):
each strategy row resumes from its contingency's N-1 state -- its voltages and where its outer
loops ended (the targets they moved, the buses they froze, ...) -- as lightsim2grid's batch row
resuming from a solve (``init_from_n_powerflow``): a batch on the grid with the contingency
applied and solved, the strategy's actions as its row. Without the warm start, a strategy row
solves from scratch, as lightsim2grid's one-off solve."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from gpusim2grid import RowStatus
from gpusim2grid.actions import Action

from _topo_helpers import batch_V_global, make_ca

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    from pandapower.auxiliary import pandapowerNet

    from gpusim2grid import ContingencyAnalysisGPU

pytestmark = pytest.mark.gpu

OUTER: str = "NROuter_KLU"
MAX_IT: int = 30
TOL_MVA: float = 1e-8  # lightsim2grid's tol is in MVA
ATOL: float = 1e-10
CONTINGENCIES: list[list[int]] = [[3], [5], [2]]
# the actions, and the same as a lightsim2grid batch row (ContingencyAnalysisCPP.add_contingency)
ACTIONS: list[Action] = [
    Action(load_bus={2: -1}),  # a load shed
    Action(gen_bus={1: -1}),  # a generator tripped
    Action(line_bus1={13: -1}, line_bus2={13: -1}),  # a line opened
]
ACTION_ROWS: list[dict[str, list[int]]] = [dict(load_ids=[2]), dict(gen_ids=[1]), dict(branch_ids=[13])]
SETTINGS: list[Any] = [
    pytest.param(dict(scheduling="continuous"), id="cudss_continuous"),
    pytest.param(dict(linear_solver="own_lu", scheduling="continuous"), id="own_lu_continuous"),
]


def _case14() -> LSGrid:
    """case14 short of reactive power (ReactiveLimits freezes buses at their limits) and sharing
    the slack on its generators (DistributedSlack), solved with its outer loops."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    net = cast("pandapowerNet", pn.case14())
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid: LSGrid = init_from_pandapower(net)
    n_gen: int = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    grid.set_gen_can_participate_slack([True] * n_gen, np.full(n_gen, 300.0))
    grid.change_algorithm(OUTER)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return grid


@pytest.fixture(scope="module")
def case14() -> LSGrid:
    return _case14()


def _resumed_reference(grid: LSGrid, line: int, action: int, redistribute_slack: bool) -> np.ndarray:
    """lightsim2grid: the contingency applied on a copy and solved with its outer loops (from
    the base voltages), then a batch on that copy whose row is the action, resuming from that
    solve. Global bus ids, 0 where a bus is out of service."""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP

    ref: LSGrid = copy.deepcopy(grid)
    ref.deactivate_powerline(line)
    ref.tell_solver_need_reset()
    V1: np.ndarray = ref.ac_pf(np.asarray(grid.get_V()).copy(), MAX_IT, TOL_MVA)
    assert V1.size > 0, "the contingency diverges in lightsim2grid"
    batch = ContingencyAnalysisCPP(ref)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = True
    batch.redistribute_slack = redistribute_slack
    batch.add_contingency(**ACTION_ROWS[action])
    batch.compute(V1, MAX_IT, TOL_MVA)
    V: np.ndarray = np.asarray(batch.get_voltages())[0]
    assert np.all(np.isfinite(V[np.abs(V) > 0])), "the strategy diverges in lightsim2grid"
    return V


def _scratch_reference(grid: LSGrid, line: int, action: int) -> np.ndarray:
    """lightsim2grid's one-off solve of the contingency and the action, from the base voltages."""
    ref: LSGrid = copy.deepcopy(grid)
    ref.deactivate_powerline(line)
    ACTIONS[action].apply_to(ref)
    ref.tell_solver_need_reset()
    V: np.ndarray = ref.ac_pf(np.asarray(grid.get_V()).copy(), MAX_IT, TOL_MVA)
    assert V.size > 0, "the strategy diverges in lightsim2grid"
    return np.where(np.asarray(ref.get_bus_status(), dtype=bool), V, 0.0)


def _two_phases(grid: LSGrid, settings: dict[str, Any], warm: bool, **kwargs: Any) -> ContingencyAnalysisGPU:
    ca = make_ca(
        grid,
        settings,
        batch_size=4,
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
        **kwargs,
    )
    ca.add_contingencies_by_branch_id(CONTINGENCIES)
    ca.set_actions(ACTIONS)
    ca.add_strategies([(c, [a]) for c in range(len(CONTINGENCIES)) for a in range(len(ACTIONS))])
    ca.strategy_warm_start = warm
    ca.compute(batch_size=4, rows="contingencies")
    ca.compute(batch_size=4, rows="strategies")
    return ca


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("redistribute_slack", [False, True], ids=["mismatch_shared", "loss_shared_first"])
def test_strategy_rows_resume_their_contingency_outer_state(
    case14: LSGrid, settings: dict[str, Any], redistribute_slack: bool
):
    """Every warm-started strategy row is lightsim2grid's batch row resumed from its
    contingency's solve -- with redistribute_slack, the loss its actions add (and only that:
    its contingency's is in the targets it resumes from) shared before its first solve. That
    reference differs from the one-off solve: the loops' state matters on this grid."""
    ca = _two_phases(case14, settings, True, redistribute_slack=redistribute_slack)
    n: int = len(CONTINGENCIES) * len(ACTIONS)
    V: np.ndarray = batch_V_global(ca, n, case14.total_bus())
    gaps: list[float] = []
    for r in range(n):
        c, a = divmod(r, len(ACTIONS))
        line: int = CONTINGENCIES[c][0]
        V_ref: np.ndarray = _resumed_reference(case14, line, a, redistribute_slack)
        on: np.ndarray = np.abs(V_ref) > 0
        np.testing.assert_allclose(V[r][on], V_ref[on], atol=ATOL, rtol=0, err_msg=f"strategy {r}")
        gaps.append(float(np.max(np.abs(V_ref[on] - _scratch_reference(case14, line, a)[on]))))
    assert max(gaps) > 1e-5, "no strategy where resuming the loops changes the result"


@pytest.mark.parametrize("settings", SETTINGS)
def test_without_warm_start_strategy_rows_solve_from_scratch(case14: LSGrid, settings: dict[str, Any]):
    ca = _two_phases(case14, settings, False)
    n: int = len(CONTINGENCIES) * len(ACTIONS)
    V: np.ndarray = batch_V_global(ca, n, case14.total_bus())
    for r in range(n):
        c, a = divmod(r, len(ACTIONS))
        V_ref: np.ndarray = _scratch_reference(case14, CONTINGENCIES[c][0], a)
        on: np.ndarray = np.abs(V_ref) > 0
        np.testing.assert_allclose(V[r][on], V_ref[on], atol=ATOL, rtol=0, err_msg=f"strategy {r}")


def test_resume_needs_the_contingencies_first(case14: LSGrid):
    ca = make_ca(
        case14,
        dict(scheduling="continuous"),
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / case14.get_sn_mva(),
    )
    ca.add_contingencies_by_branch_id(CONTINGENCIES)
    ca.set_actions(ACTIONS)
    ca.add_strategies([(0, [0])])
    ca.strategy_warm_start = True
    with pytest.raises(RuntimeError, match="contingencies"):
        ca.compute(batch_size=4, rows="strategies")


@pytest.fixture(scope="module")
def case14_producing() -> LSGrid:
    """case14 whose generators 1-3 produce (15, 20, 25 MW): tripping one loses power."""
    grid: LSGrid = _case14()
    for gen, mw in ((1, 15.0), (2, 20.0), (3, 25.0)):
        grid.change_p_gen(gen, mw)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.asarray(grid.get_V()).copy(), MAX_IT, TOL_MVA).size > 0
    return grid


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("redistribute_slack", [False, True], ids=["mismatch_shared", "loss_shared_first"])
def test_a_noop_strategy_gives_its_contingency_back(
    case14_producing: LSGrid, settings: dict[str, Any], redistribute_slack: bool
):
    """A strategy that changes nothing, resumed from its contingency, ends where the contingency
    ended: its loss (a generator tripped) is in the targets it resumes from, not shared again."""
    grid: LSGrid = case14_producing
    ca = make_ca(
        grid,
        settings,
        batch_size=4,
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
        redistribute_slack=redistribute_slack,
    )
    for gen in (1, 2, 3):
        ca.add_contingency_gen([gen])
    ca.add_contingencies([([3], [], [], [])])  # and a line
    n_ctg: int = 4
    ca.set_actions([Action()])
    ca.add_strategies([(c, [0]) for c in range(n_ctg)])
    ca.strategy_warm_start = True
    ca.compute(batch_size=4, rows="contingencies")
    V_ctg: np.ndarray = batch_V_global(ca, n_ctg, grid.total_bus())
    ca.compute(batch_size=4, rows="strategies")
    V_str: np.ndarray = batch_V_global(ca, n_ctg, grid.total_bus())
    on: np.ndarray = np.isfinite(V_ctg)
    np.testing.assert_allclose(V_str[on], V_ctg[on], atol=ATOL, rtol=0)


# ---- bus splits under the outer loops ---------------------------------------------------------
SPLIT_CONTINGENCIES: list[list[int]] = [[], [3], [6]]


@pytest.fixture(scope="module")
def case14_two_busbars() -> LSGrid:
    """The outer-loop case14 (short of reactive power, distributed slack) with an empty second
    busbar per substation, for bus splits."""
    import pandapower.networks as pn

    from _topo_helpers import _two_busbars

    net = cast("pandapowerNet", pn.case14())
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    grid: LSGrid = _two_busbars(net)
    n_gen: int = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    grid.set_gen_can_participate_slack([True] * n_gen, np.full(n_gen, 300.0))
    grid.change_algorithm(OUTER)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return grid


def _splits(grid: LSGrid) -> list[Action]:
    """Every other branch end of the three most connected substations moved to busbar 2."""
    from test_topo_bus_moves import _branch_ends, _move

    subs: list[int] = sorted(range(grid.get_n_sub()), key=lambda s: -len(_branch_ends(grid, s)))
    return [_move(grid, s, _branch_ends(grid, s)[1::2]) for s in subs[:3]]


def _start(grid: LSGrid, V: np.ndarray) -> np.ndarray:
    """``V`` with each bus out of service started at its substation's first busbar (the
    voltage the batch gives a spare bus)."""
    n_sub: int = grid.get_n_sub()
    out: np.ndarray = np.asarray(V).copy()
    out[n_sub:] = np.where(np.abs(out[n_sub:]) > 0, out[n_sub:], out[:n_sub])
    return out


def _topo_action(grid: LSGrid, action: Action) -> Any:
    """``action``'s branch and generator moves as a lightsim2grid TopoAction (grid2op set_bus,
    local busbars)."""
    import lightsim2grid.lightsim2grid_cpp as cpp

    ta = cpp.TopoAction()
    n_sub: int = grid.get_n_sub()
    for name, kind in (
        ("line_bus1", cpp.ElementType.line_or),
        ("line_bus2", cpp.ElementType.line_ex),
        ("trafo_bus1", cpp.ElementType.trafo_hv),
        ("trafo_bus2", cpp.ElementType.trafo_lv),
        ("gen_bus", cpp.ElementType.gen),
    ):
        for el, bus in getattr(action, name).items():
            ta.add_element(kind, int(el), int(bus) // n_sub + 1)
    ta.check_validity(grid)
    return ta


def _split_reference(grid: LSGrid, lines: list[int], action: Action, resume: bool) -> np.ndarray | None:
    """lightsim2grid, resumed (a ScenarioSweep row playing the split, resuming from the
    contingency's solve) or one-off; None where it does not converge."""
    import lightsim2grid.lightsim2grid_cpp as cpp

    ref: LSGrid = copy.deepcopy(grid)
    for line in lines:
        ref.deactivate_powerline(line)
    if not resume:
        action.apply_to(ref)
    ref.tell_solver_need_reset()
    V: np.ndarray = ref.ac_pf(_start(grid, np.asarray(grid.get_V())), MAX_IT, TOL_MVA)
    if V.size == 0:
        return None
    if not resume:
        return np.where(np.asarray(ref.get_bus_status(), dtype=bool), V, 0.0)
    sweep = cpp.ScenarioSweepCPP(ref)
    sweep.change_algorithm(OUTER)
    sweep.init_from_n_powerflow = True
    sweep.set_topo_actions([_topo_action(ref, action)])
    sweep.compute(_start(grid, V), MAX_IT, TOL_MVA)
    out: np.ndarray = np.asarray(sweep.get_voltages())[0]
    on: np.ndarray = np.abs(out) > 0
    return out if on.any() and np.all(np.isfinite(out[on])) else None


@pytest.mark.parametrize("settings", SETTINGS)
@pytest.mark.parametrize("warm", [True, False], ids=["warm", "from_scratch"])
def test_bus_splits_under_the_outer_loops(case14_two_busbars: LSGrid, settings: dict[str, Any], warm: bool):
    """Split strategies with the outer loops: each row is lightsim2grid's -- its ScenarioSweep
    row resumed from the contingency's solve (warm), or its one-off solve; a row lightsim2grid
    does not converge does not converge either. A strategy moving the line its contingency trips
    is not run (it would put it back)."""
    grid: LSGrid = case14_two_busbars
    actions: list[Action] = _splits(grid)
    strategies = [(c, [a]) for c in range(len(SPLIT_CONTINGENCIES)) for a in range(len(actions))]
    ca = make_ca(
        grid,
        settings,
        batch_size=4,
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
    )
    ca.add_contingencies_by_branch_id(SPLIT_CONTINGENCIES)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.strategy_warm_start = warm
    ca.compute(batch_size=4, rows="contingencies")
    ca.compute(batch_size=4, rows="strategies")
    V: np.ndarray = batch_V_global(ca, len(strategies), grid.total_bus())
    status: np.ndarray = ca.get_row_status()
    compared: int = 0
    for r, (c, (a,)) in enumerate(strategies):
        lines: list[int] = SPLIT_CONTINGENCIES[c]
        moved: set[int] = {*actions[a].line_bus1, *actions[a].line_bus2}
        if moved & set(lines):
            assert status[r] == RowStatus.NOT_SIMULATED
            continue
        V_ref = _split_reference(grid, lines, actions[a], warm)
        if V_ref is None:
            assert status[r] != RowStatus.CONVERGED, f"strategy {r}"
            continue
        on: np.ndarray = np.abs(V_ref) > 0
        np.testing.assert_allclose(V[r][on], V_ref[on], atol=1e-9, rtol=0, err_msg=f"strategy {r}")
        compared += 1
    assert compared >= 6


# ---- generator moves under the outer loops ----------------------------------------------------
def _case14_gen_moves() -> LSGrid:
    """The two-busbar outer-loop case14 whose generators 1-3 take no part in the slack (a slack
    participant does not move, in lightsim2grid either): generator 0 and the slack share it."""
    import pandapower.networks as pn

    from _topo_helpers import _two_busbars

    net = cast("pandapowerNet", pn.case14())
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    grid: LSGrid = _two_busbars(net)
    n_gen: int = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    participates: list[bool] = [g.id == 0 or g.is_slack for g in grid.get_generators()]
    grid.set_gen_can_participate_slack(participates, np.full(n_gen, 300.0))
    grid.change_algorithm(OUTER)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return grid


@pytest.fixture(scope="module")
def case14_gen_moves() -> LSGrid:
    return _case14_gen_moves()


def _gen_moves(grid: LSGrid) -> list[Action]:
    """Generators 1-3 each moved to busbar 2 of their substation with one branch end (a new bus
    the generator holds PV, its former one left PQ)."""
    from test_topo_bus_moves import _branch_ends, _move

    out: list[Action] = []
    for gen in (grid.get_generators()[g] for g in (1, 2, 3)):
        out.append(_move(grid, gen.bus_id, [("gen_bus", int(gen.id)), _branch_ends(grid, gen.bus_id)[0]]))
    return out


def _frozen_at_end(decisions: list[str]) -> set[int]:
    """The buses a row's ReactiveLimits trace leaves frozen (PV_TO_PQ taken, not released since)."""
    frozen: set[int] = set()
    for d in decisions:
        if "not taken" in d or "ReactiveLimits" not in d:
            continue
        bus = int(d.split("element ")[1].split(",")[0])
        if " PV_TO_PQ" in d:
            frozen.add(bus)
        elif " PQ_TO_PV" in d:
            frozen.discard(bus)
    return frozen


@pytest.mark.parametrize("settings", SETTINGS)
def test_generator_moves_under_the_outer_loops(case14_gen_moves: LSGrid, settings: dict[str, Any]):
    """From scratch, a generator moved to another bus under the outer loops is lightsim2grid's
    one-off solve: its new bus regulated (ReactiveLimits on it), its former one PQ."""
    grid: LSGrid = case14_gen_moves
    actions: list[Action] = _gen_moves(grid)
    strategies = [(c, [a]) for c in range(len(SPLIT_CONTINGENCIES)) for a in range(len(actions))]
    ca = make_ca(
        grid,
        settings,
        batch_size=4,
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
    )
    ca.add_contingencies_by_branch_id(SPLIT_CONTINGENCIES)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.compute(batch_size=4, rows="contingencies")
    ca.compute(batch_size=4, rows="strategies")
    V: np.ndarray = batch_V_global(ca, len(strategies), grid.total_bus())
    status: np.ndarray = ca.get_row_status()
    reasons: list[str] = ca.get_batch_unsupported_reasons()
    compared: int = 0
    for r, (c, (a,)) in enumerate(strategies):
        lines: list[int] = SPLIT_CONTINGENCIES[c]
        assert not reasons[r], reasons[r]
        V_ref = _split_reference(grid, lines, actions[a], False)
        if V_ref is None:
            assert status[r] != RowStatus.CONVERGED, f"strategy {r}"
            continue
        on: np.ndarray = np.abs(V_ref) > 0
        np.testing.assert_allclose(V[r][on], V_ref[on], atol=1e-9, rtol=0, err_msg=f"strategy {r}")
        compared += 1
    assert compared >= 6


@pytest.mark.parametrize("settings", SETTINGS)
def test_generator_moves_resume_their_contingency(case14_gen_moves: LSGrid, settings: dict[str, Any]):
    """Warm-started, a strategy moving a generator resumes from its contingency's outer state: a
    bus its contingency left frozen (still holding a machine) is not frozen again, and the moved
    generator's new bus is at its set-point unless the row's own check froze it.

    No lightsim2grid reference here: its ScenarioSweep with the outer loops leaves a moved
    generator's new bus unregulated (Q = 0, PQ; its one-off solve and its ScenarioSweep without
    outer loops do regulate it), and the batch is its only solve resuming a row."""
    grid: LSGrid = case14_gen_moves
    actions: list[Action] = _gen_moves(grid)
    strategies = [(c, [a]) for c in range(len(SPLIT_CONTINGENCIES)) for a in range(len(actions))]
    ca = make_ca(
        grid,
        settings,
        batch_size=4,
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
        keep_outer_trace=True,
    )
    ca.add_contingencies_by_branch_id(SPLIT_CONTINGENCIES)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    ca.strategy_warm_start = True
    ca.compute(batch_size=4, rows="contingencies")
    ctg_frozen: list[set[int]] = [_frozen_at_end(r.decisions) for r in ca.get_outer_results()]
    ca.compute(batch_size=4, rows="strategies")
    V: np.ndarray = batch_V_global(ca, len(strategies), grid.total_bus())
    status: np.ndarray = ca.get_row_status()
    results = ca.get_outer_results()
    gens = grid.get_generators()
    solver_of: dict[int, int] = {int(b): k for k, b in enumerate(ca.batch_bus_ids)}
    checked: int = 0
    for r, (c, (a,)) in enumerate(strategies):
        if _split_reference(grid, SPLIT_CONTINGENCIES[c], actions[a], False) is None:
            continue  # lightsim2grid does not converge it either, from scratch
        assert status[r] == RowStatus.CONVERGED, f"strategy {r}"
        ((g, new_bus),) = actions[a].gen_bus.items()
        # the traces name buses as the solver does (spares after the base buses)
        new_solver: int = solver_of[new_bus]
        old_bus: int = solver_of[gens[g].bus_id]
        decisions: list[str] = results[r].decisions
        refrozen = {
            int(d.split("element ")[1].split(",")[0])
            for d in decisions
            if " PV_TO_PQ" in d and "not taken" not in d
        }
        assert not (refrozen & (ctg_frozen[c] - {old_bus})), (r, refrozen, ctg_frozen[c])
        if new_solver not in _frozen_at_end(decisions):
            assert abs(V[r][new_bus]) == pytest.approx(gens[g].target_vm_pu, abs=1e-9), r
        checked += 1
    assert checked >= 6
