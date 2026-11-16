# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Engine side of the two phases (docs/dev_notes/topology_actions_plan.md § 4): the max-loading
reduction on the device, running the contingency rows and the strategy rows as two calls on one
session, a selection of the strategy rows, and the warm start of a strategy from its
contingency's converged state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

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


@pytest.mark.parametrize("settings", ca_settings())
def test_two_calls_equal_one(settings: dict[str, Any]):
    """compute(rows='contingencies') then compute(rows='strategies') on one session: the same
    rows as compute() of everything."""
    grid, ctgs, actions, strategies = _case()
    one = _ca(grid, ctgs, actions, strategies, settings)
    one.compute(batch_size=BATCH)
    V_one = batch_V(one, len(ctgs) + len(strategies))

    two = _ca(grid, ctgs, actions, strategies, settings)
    two.compute(batch_size=BATCH, rows="contingencies")
    V_c = batch_V(two, len(ctgs))
    two.compute(batch_size=BATCH, rows="strategies")
    V_s = batch_V(two, len(strategies))
    np.testing.assert_allclose(V_c, V_one[: len(ctgs)], atol=1e-12, rtol=0)
    np.testing.assert_allclose(V_s, V_one[len(ctgs) :], atol=1e-12, rtol=0)


def test_select_strategies():
    """Only the selected strategy rows run (the others: NaN, not simulated, not disconnected)."""
    grid, ctgs, actions, strategies = _case()
    ca = _ca(grid, ctgs, actions, strategies)
    ca.compute(batch_size=BATCH, rows="contingencies")
    mask = np.zeros(len(strategies), dtype=bool)
    mask[::3] = True
    ca.select_strategies(mask)
    ca.compute(batch_size=BATCH, rows="strategies")
    res = ca.last_residuals()
    assert res.shape == (len(strategies),)
    assert np.all(np.isnan(res[~mask]))
    assert np.all(np.isfinite(res[mask]))
    np.testing.assert_array_equal(ca.get_strategy_ran(), mask)


@pytest.mark.parametrize("settings", ca_settings())
def test_warm_start_same_solution_fewer_iterations(settings: dict[str, Any], solver_atol: float):
    """strategy_warm_start: a strategy row starts from its contingency's converged V and reaches
    the same solution as lightsim2grid's. (Its iteration savings depend on the grid: case118's
    strategies are a few iterations from the base case anyway; see the next test for the start.)"""
    grid, ctgs, actions, strategies = _case()
    iters: dict[bool, np.ndarray] = {}
    for warm in (False, True):
        ca = _ca(grid, ctgs, actions, strategies, settings, tol=1e-8)
        ca.strategy_warm_start = warm
        ca.compute(batch_size=BATCH, rows="contingencies")
        ca.compute(batch_size=BATCH, rows="strategies")
        iters[warm] = ca.get_iterations()
        assert iters[warm].shape == (len(strategies),)
        V = batch_V_global(ca, len(strategies), grid.total_bus())
        for r, (c, (a,)) in enumerate(strategies):
            V_ref, _ = oracle(grid, branches=ctgs[c], action=actions[a])
            if V_ref is not None:
                assert_row_matches(V[r], V_ref, solver_atol)
    assert np.all(iters[True] <= MAX_IT)


def test_two_phase_recipe():
    """The two phases a caller writes with ``rows=``: phase 1, a condition of their own on the
    contingency rows -- here pypowsybl's AT_LEAST_ONE_VIOLATION_CONDITION, with limits
    tightened by ``violation_threshold`` so that about half the contingencies violate -- then
    phase 2 on the strategies of those contingencies, warm started. The strategies that ran
    equal the same rows of a run of every strategy."""
    grid, ctgs, actions, strategies = _case()
    ctg_of: np.ndarray = np.array([c for c, _ in strategies])

    every = _ca(grid, ctgs, actions, strategies)
    every.set_limits_from_grid()
    every.compute(batch_size=BATCH, rows="contingencies")
    rho_c, _ = every.get_max_loading()
    every.compute(batch_size=BATCH, rows="strategies")
    V_every: np.ndarray = batch_V(every, len(strategies))

    # limits scaled so every contingency's loading is below 1, then the threshold at the median
    # loading: the contingencies above the median violate
    scale: float = 1.01 * float(np.max(rho_c))
    grid = with_current_limits(case118(), 1.1 * scale)
    ca = _ca(grid, ctgs, actions, strategies, compute_limit_violations=True)
    ca.set_limits_from_grid()
    ca.violation_threshold = float(np.median(rho_c)) / scale
    ca.compute(batch_size=BATCH, rows="contingencies")
    violating: np.ndarray = ca.get_violation_counts()["current"] > 0
    assert 0 < violating.sum() < len(ctgs)
    ca.select_strategies(violating[ctg_of])
    ca.strategy_warm_start = True
    ca.compute(batch_size=BATCH, rows="strategies")
    ran: np.ndarray = ca.get_strategy_ran()
    np.testing.assert_array_equal(ran, violating[ctg_of])
    V: np.ndarray = batch_V(ca, len(strategies))
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


@pytest.mark.parametrize("warm", [False, True])
def test_warm_start_starts_from_the_contingency(warm: bool, residual_atol: float):
    """A strategy whose action changes nothing is its contingency's row: warm started, it begins
    at that row's converged voltages, so one Newton iteration leaves it converged; started from
    the base case, it does not."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action

    grid = case118()
    line = grid.get_lines()[100]
    same = Action(line_bus1={line.id: int(line.bus1_id)})  # already there: changes nothing
    ctgs = [[2], [40], [77]]
    ca = ContingencyAnalysisGPU(grid, nb_iter=MAX_IT)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions([same])
    ca.add_strategies([(c, [0]) for c in range(len(ctgs))])
    ca.strategy_warm_start = warm
    ca.compute(batch_size=BATCH, rows="contingencies")
    assert np.all(ca.last_residuals() < residual_atol)
    ca.solver.nb_iter = 1  # the engine holds it (mutable, next run)
    ca.compute(batch_size=BATCH, rows="strategies")
    res: np.ndarray = ca.last_residuals()
    if warm:
        assert np.all(res < residual_atol)
    else:
        assert np.all(res > residual_atol)


@pytest.mark.xfail(
    strict=True,
    reason="plan step A4 (deferred): the contingency session rebuilds its "
    "driver when the row count changes between the phases",
)
def test_phases_reuse_the_driver_chunked():
    """In the chunked schedule, phase 2 runs on phase 1's driver (no second analysis). The
    continuous schedule rebuilds its driver every run in any case (upstream A.6)."""
    grid, ctgs, actions, strategies = _case()
    ca = _ca(grid, ctgs, actions, strategies)
    ca.compute(batch_size=BATCH, rows="contingencies")
    n_build: int = ca.solver._s.driver_build_counter
    ca.compute(batch_size=BATCH, rows="strategies")
    assert ca.solver._s.driver_build_counter == n_build
