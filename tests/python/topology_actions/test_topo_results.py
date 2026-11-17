# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Engine side of the strategies (docs/dev_notes/topology_actions_plan.md § 4): the max-loading
reduction on the device, the condition releasing a contingency's strategies, and the warm start
of a strategy from its own contingency's converged state -- one compute()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from gpusim2grid import StrategyCondition
from gpusim2grid._gpusim2grid import have_ls2g_bridge

from _topo_helpers import (
    MAX_IT,
    Strategy,
    assert_row_matches,
    batch_V,
    batch_V_global,
    ca_settings,
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

BATCH: int = 32
CONTINUOUS: list[Any] = [
    pytest.param(dict(scheduling="continuous"), id="cudss_continuous"),
    pytest.param(dict(linear_solver="own_lu", scheduling="continuous"), id="own_lu_continuous"),
]


def _case() -> tuple[LSGrid, list[list[int]], list[Action], list[Strategy]]:
    """case118, 6 contingencies, 5 disconnection actions (a line trip, a line opened at one end,
    a load, a generator, a line and a load together), every pair. Bus moves and branch values
    have their own tests (test_topo_bus_moves.py, test_topo_branch_values.py)."""
    from gpusim2grid.actions import Action

    grid = with_current_limits(case118(), 1.1)
    gen: int = next(g.id for g in grid.get_generators() if not g.is_slack)
    actions = [
        Action(line_bus1={30: -1}, line_bus2={30: -1}),
        Action(line_bus1={64: -1}),
        Action(load_bus={10: -1}),
        Action(gen_bus={gen: -1}),
        Action(line_bus1={70: -1}, line_bus2={70: -1}, load_bus={20: -1}),
    ]
    ctgs = [[2], [9], [40], [77], [120], [150]]
    strategies = [(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))]
    return grid, ctgs, actions, strategies


def _ca(
    grid: LSGrid,
    ctgs: list[list[int]],
    actions: list[Action],
    strategies: list[Strategy],
    settings: dict[str, Any] | None = None,
    **kw: Any,
) -> ContingencyAnalysisGPU:
    ca = make_ca(grid, settings or {}, batch_size=BATCH, **kw)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies(strategies)
    return ca


@pytest.mark.parametrize("settings", ca_settings())
def test_max_loading_matches_the_flows(settings: dict[str, Any]):
    """get_max_loading: per row, max over branches and sides of |I| / limit (limits from the
    grid, NaN limits ignored) and the branch reaching it -- the flows' own reduction."""
    grid, ctgs, actions, strategies = _case()
    ca = _ca(grid, ctgs, actions, strategies, settings)
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    rho, branch = ca.get_max_loading()
    n = len(ctgs) + len(strategies)
    assert rho.shape == (n,) and branch.shape == (n,)
    ca.compute_flows()
    nb = ca.n_branches
    i1: np.ndarray = ca.or_amps.to_numpy().reshape(n, nb) * 1e-3  # A -> kA, as the limits
    i2: np.ndarray = ca.ex_amps.to_numpy().reshape(n, nb) * 1e-3
    lim1 = np.array([ln.limit_a1_ka for ln in grid.get_lines()] + [t.limit_a1_ka for t in grid.get_trafos()])
    lim2 = np.array([ln.limit_a2_ka for ln in grid.get_lines()] + [t.limit_a2_ka for t in grid.get_trafos()])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.fmax(np.where(lim1 > 0, i1 / lim1, np.nan), np.where(lim2 > 0, i2 / lim2, np.nan))
    np.testing.assert_allclose(rho, np.nanmax(r, axis=1), rtol=1e-9)
    np.testing.assert_array_equal(branch, np.nanargmax(r, axis=1))


def test_max_loading_without_flows_on_the_host():
    """The reduction works without compute_flows() (no n_rows x n_branches copy to the host)."""
    grid, ctgs, actions, strategies = _case()
    ca = _ca(grid, ctgs, actions, strategies)
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    rho, _ = ca.get_max_loading()
    assert np.all(np.isfinite(rho[np.isfinite(ca.last_residuals())]))


@pytest.mark.parametrize("settings", CONTINUOUS)
def test_warm_start_same_solution_fewer_iterations(settings: dict[str, Any], solver_atol: float):
    """strategy_warm_start: a strategy row starts from its contingency's converged V and reaches
    the same solution as lightsim2grid's. (Its iteration savings depend on the grid: case118's
    strategies are a few iterations from the base case anyway; see the next test for the start.)"""
    grid, ctgs, actions, strategies = _case()
    n_ctg: int = len(ctgs)
    iters: dict[bool, np.ndarray] = {}
    for warm in (False, True):
        ca = _ca(grid, ctgs, actions, strategies, settings, tol=1e-8, strategy_warm_start=warm)
        ca.compute(batch_size=BATCH)
        iters[warm] = ca.get_iterations()[n_ctg:]
        assert iters[warm].shape == (len(strategies),)
        V = batch_V_global(ca, n_ctg + len(strategies), grid.total_bus())[n_ctg:]
        for r, (c, (a,)) in enumerate(strategies):
            V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
            if V_ref is not None:
                assert_row_matches(V[r], V_ref, solver_atol)
    assert np.all(iters[True] <= MAX_IT)


def _violations(ca: ContingencyAnalysisGPU, rows: slice) -> np.ndarray:
    counts = ca.get_violation_counts()
    return (counts["low_voltage"] + counts["high_voltage"] + counts["current"])[rows] > 0


@pytest.mark.parametrize("settings", CONTINUOUS)
def test_condition_recipe(settings: dict[str, Any]):
    """pypowsybl's AT_LEAST_ONE_VIOLATION_CONDITION, with limits tightened by
    ``violation_threshold`` so that about half the contingencies violate: only the strategies of
    those contingencies run, warm started, and equal the same rows of a run of every strategy."""
    grid, ctgs, actions, strategies = _case()
    n_ctg: int = len(ctgs)
    ctg_of: np.ndarray = np.array([c for c, _ in strategies])

    every = _ca(grid, ctgs, actions, strategies, settings, strategy_warm_start=True)
    every.set_limits_from_grid()
    every.compute(batch_size=BATCH)
    rho_c: np.ndarray = every.get_max_loading()[0][:n_ctg]
    V_every: np.ndarray = batch_V(every, n_ctg + len(strategies))[n_ctg:]

    # limits scaled so every contingency's loading is below 1, then the threshold at the median
    # loading: the contingencies above the median violate
    scale: float = 1.01 * float(np.max(rho_c))
    grid = with_current_limits(case118(), 1.1 * scale)
    ca = _ca(
        grid,
        ctgs,
        actions,
        strategies,
        settings,
        compute_limit_violations=True,
        violation_threshold=float(np.median(rho_c)) / scale,
        strategy_condition=StrategyCondition.ANY_VIOLATION,
        strategy_warm_start=True,
    )
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    violating: np.ndarray = _violations(ca, slice(0, n_ctg))
    assert 0 < violating.sum() < n_ctg
    ran: np.ndarray = ca.get_strategy_ran()
    np.testing.assert_array_equal(ran, violating[ctg_of])
    V: np.ndarray = batch_V(ca, n_ctg + len(strategies))[n_ctg:]
    np.testing.assert_allclose(V[ran], V_every[ran], atol=1e-8, rtol=0)
    assert np.all(np.isnan(V[~ran]))


def test_max_loading_of_a_row_not_solved():
    """A strategy row skipped as islanded (its contingency and its action cut a bus off, no
    handle_disconnected_grid) has no loading: NaN / -1, not the zero currents it reads."""
    from gpusim2grid.actions import Action

    grid = with_current_limits(case118(), 1.1)
    lines = grid.get_lines()
    assert lines[0].bus1_id == lines[1].bus1_id  # lines 0 and 1 are the only two at bus 0
    ca = _ca(grid, [[0]], [Action(line_bus1={1: -1}, line_bus2={1: -1})], [(0, [0])])
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    assert np.isnan(ca.last_residuals()[1])
    rho, branch = ca.get_max_loading()
    assert np.isfinite(rho[0]) and branch[0] >= 0
    assert np.isnan(rho[1]) and branch[1] == -1


@pytest.mark.parametrize("warm", [False, True], ids=["base_start", "warm"])
@pytest.mark.parametrize("shuffled", [False, True], ids=["by_contingency", "shuffled"])
def test_warm_start_starts_from_its_own_contingency(warm: bool, shuffled: bool, residual_atol: float):
    """A strategy whose action changes nothing is its contingency's row: warm started, it begins
    at that row's converged voltages -- its own contingency's, whatever the order the strategies
    come in -- so it is converged at once (at most one Newton iteration) and ends on its
    contingency's voltages; started from the base case, it needs more."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action

    grid = case118()
    line = grid.get_lines()[100]
    same = Action(line_bus1={line.id: int(line.bus1_id)})  # already there: changes nothing
    ctgs = [[2], [40], [77], [120]]
    order: list[int] = [2, 0, 3, 1, 2, 0] if shuffled else [0, 0, 1, 1, 2, 3]
    ca = ContingencyAnalysisGPU(
        grid, nb_iter=MAX_IT, scheduling="continuous", tol=1e-10, strategy_warm_start=warm
    )
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions([same])
    ca.add_strategies([(c, [0]) for c in order])
    ca.compute(batch_size=BATCH)
    n_ctg: int = len(ctgs)
    res: np.ndarray = ca.last_residuals()
    assert np.all(res < residual_atol)
    iters: np.ndarray = ca.get_iterations()
    V: np.ndarray = batch_V(ca, n_ctg + len(order))
    if warm:
        assert np.all(iters[n_ctg:] <= 1), iters
        np.testing.assert_allclose(V[n_ctg:], V[order], atol=1e-9, rtol=0)
    else:
        assert np.all(iters[n_ctg:] >= 2), iters


@pytest.mark.parametrize("settings", CONTINUOUS)
def test_row_status_and_iterations_on_the_strategy_rows(settings: dict[str, Any]):
    """get_row_status / get_row_iterations have one entry per row, contingencies then strategies:
    the session's for the strategies that ran, NOT_SIMULATED / 0 for the ones their contingency
    did not release (``StrategyCondition.ANY_VIOLATION``, current limits at 1.6 x the base
    case: some contingencies stay within them)."""
    from gpusim2grid import RowStatus

    grid, ctgs, actions, strategies = _case()
    grid = with_current_limits(case118(), 1.6)
    ca = _ca(
        grid,
        ctgs,
        actions,
        strategies,
        settings,
        compute_limit_violations=True,
        strategy_condition=StrategyCondition.ANY_VIOLATION,
    )
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    n_ctg: int = len(ctgs)
    picked: np.ndarray = ca.get_strategy_ran()
    np.testing.assert_array_equal(picked, _violations(ca, slice(0, n_ctg))[[c for c, _ in strategies]])
    assert 0 < picked.sum() < len(strategies)
    status: np.ndarray = ca.get_row_status()
    iters: np.ndarray = ca.get_row_iterations()
    assert status.shape == iters.shape == (n_ctg + len(strategies),)
    status, iters = status[n_ctg:], iters[n_ctg:]
    assert np.all(status[~picked] == RowStatus.NOT_SIMULATED)
    assert np.all(iters[~picked] == 0)
    assert np.all(status[picked] == RowStatus.CONVERGED)
    assert np.all(iters[picked] > 0)
    np.testing.assert_array_equal(status[picked], np.asarray(ca._inner.get_row_status())[n_ctg:][picked])


def test_outer_results_on_the_strategy_rows():
    """With the outer loops, get_outer_results has one entry per row: the session's for the
    strategies that ran, None for the ones their contingency did not release."""
    grid, ctgs, actions, strategies = _case()
    grid = with_current_limits(case118(), 1.6)
    grid.change_algorithm("NROuter_KLU")
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, 1e-8).size > 0
    ca = _ca(
        grid,
        ctgs,
        actions,
        strategies,
        dict(scheduling="continuous"),
        outer_loops=True,
        init_from_n_powerflow=False,
        compute_limit_violations=True,
        strategy_condition=StrategyCondition.ANY_VIOLATION,
        strategy_warm_start=True,
    )
    ca.set_limits_from_grid()
    ca.compute(batch_size=BATCH)
    n_ctg: int = len(ctgs)
    picked: np.ndarray = ca.get_strategy_ran()
    assert 0 < picked.sum() < len(strategies)
    res = ca.get_outer_results()
    assert len(res) == n_ctg + len(strategies)
    assert all(r is not None for r in res[:n_ctg])
    assert all(r is None for r, p in zip(res[n_ctg:], picked) if not p)
    run = [r for r, p in zip(res[n_ctg:], picked) if p]
    session = [r for r, p in zip(ca._inner._s.get_outer_results()[n_ctg:], picked) if p]
    assert [r.status for r in run] == [r.status for r in session]
    assert [r.nb_nr_iterations for r in run] == [r.nb_nr_iterations for r in session]


def test_release_needs_the_continuous_schedule():
    """The condition and the warm start (default on) release a strategy as its contingency
    finishes: the continuous schedule. ``scheduling=None`` picks it (a plain N-1 stays chunked);
    an explicit chunked schedule is refused, unless the strategies start from the base case."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action

    grid = case118()
    assert (
        ContingencyAnalysisGPU(grid, strategy_condition=StrategyCondition.ANY_VIOLATION).scheduling
        == "continuous"
    )
    with pytest.raises(ValueError, match="continuous"):
        ContingencyAnalysisGPU(grid, scheduling="chunked", strategy_condition=StrategyCondition.ANY_VIOLATION)
    assert ContingencyAnalysisGPU(grid, strategy_condition="ANY_VIOLATION").strategy_condition is (
        StrategyCondition.ANY_VIOLATION
    )  # its string value converts
    with pytest.raises(ValueError, match="StrategyCondition"):
        ContingencyAnalysisGPU(grid, strategy_condition="ALL")

    def with_strategy(**kw: Any) -> ContingencyAnalysisGPU:
        ca = ContingencyAnalysisGPU(grid, **kw)
        ca.add_contingencies_by_branch_id([[2]])
        ca.set_actions([Action(load_bus={10: -1})])
        ca.add_strategies([(0, [0])])
        return ca

    ca = with_strategy()
    assert ca.strategy_warm_start and ca.scheduling == "chunked"  # nothing released yet
    ca.compute(batch_size=BATCH)
    assert ca.scheduling == "continuous"
    with pytest.raises(ValueError, match="continuous"):
        with_strategy(scheduling="chunked").compute(batch_size=BATCH)
    ca = with_strategy(scheduling="chunked", strategy_warm_start=False)
    ca.compute(batch_size=BATCH)
    assert ca.get_strategy_ran().all()
