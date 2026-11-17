# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Streamed strategies: one ``compute()`` on the continuous schedule runs the contingencies and
releases each contingency's strategies when it finishes (``strategy_condition``, the warm start
on the device). Every strategy row is the two-phase one's: the same start, the same state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_topo_outer_warm_start import ACTIONS, CONTINGENCIES, TOL_MVA, _case14

from gpusim2grid import RowStatus
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from gpusim2grid.actions import Action

from _topo_helpers import batch_V_global, case118, make_ca, with_current_limits
from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

    from gpusim2grid import ContingencyAnalysisGPU

pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

BATCH: int = 32
CONTINUOUS: list[Any] = [
    pytest.param(dict(scheduling="continuous"), id="cudss_continuous"),
    pytest.param(dict(linear_solver="own_lu", scheduling="continuous"), id="own_lu_continuous"),
]


def _case118() -> tuple[LSGrid, list[list[int]], list[Action]]:
    grid = with_current_limits(case118(), 1.6)  # contingencies 40 and 77 stay within limits
    ctgs = [[b] for b in (2, 9, 40, 77, 120, 150)]
    actions = [
        Action(line_bus1={30: -1}, line_bus2={30: -1}),
        Action(load_bus={10: -1}),
        Action(line_bus1={70: -1}, line_bus2={70: -1}, load_bus={20: -1}),
    ]
    return grid, ctgs, actions


def _setup(
    grid: LSGrid, ctgs: list[list[int]], actions: list[Action], settings: dict[str, Any], **kw: Any
) -> ContingencyAnalysisGPU:
    ca = make_ca(grid, settings, batch_size=BATCH, compute_limit_violations=True, **kw)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.set_actions(actions)
    ca.add_strategies([(c, [a]) for c in range(len(ctgs)) for a in range(len(actions))])
    ca.strategy_warm_start = True
    return ca


def _two_phases(ca: ContingencyAnalysisGPU, n_bus: int, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """The strategies' voltages and ran mask the explicit two phases give."""
    n_ctg: int = len(ca.get_contingencies())
    ca.compute(batch_size=BATCH, rows="contingencies")
    if condition == "ANY_VIOLATION":
        violating: np.ndarray = (
            ca.get_violation_counts()["low_voltage"]
            + ca.get_violation_counts()["high_voltage"]
            + ca.get_violation_counts()["current"]
            > 0
        )
        ctg_of = np.array([c for c in range(n_ctg) for _ in range(ca.n_strategies // n_ctg)])
        ca.select_strategies(violating[ctg_of])
    ca.compute(batch_size=BATCH, rows="strategies")
    return batch_V_global(ca, ca.n_strategies, n_bus), ca.get_strategy_ran()


def _streamed(ca: ContingencyAnalysisGPU, n_bus: int, condition: str) -> tuple[np.ndarray, np.ndarray]:
    n_ctg: int = len(ca.get_contingencies())
    ca.strategy_condition = condition
    ca.compute(batch_size=BATCH)
    V: np.ndarray = batch_V_global(ca, n_ctg + ca.n_strategies, n_bus)
    return V[n_ctg:], ca.get_strategy_ran()


@pytest.mark.parametrize("settings", CONTINUOUS)
@pytest.mark.parametrize("condition", ["TRUE", "ANY_VIOLATION"])
def test_streamed_strategies_are_the_two_phase_ones(settings: dict[str, Any], condition: str):
    grid, ctgs, actions = _case118()
    n_bus: int = grid.total_bus()
    V2, ran2 = _two_phases(_setup(grid, ctgs, actions, settings), n_bus, condition)
    ca = _setup(grid, ctgs, actions, settings)
    V1, ran1 = _streamed(ca, n_bus, condition)
    np.testing.assert_array_equal(ran1, ran2)
    assert ran1.any() and (condition == "TRUE") == ran1.all()
    np.testing.assert_allclose(V1[ran1], V2[ran2], atol=1e-10, rtol=0, equal_nan=True)
    assert np.all(np.isnan(V1[~ran1]))
    status: np.ndarray = ca.get_row_status()[len(ctgs) :]
    assert np.all(status[~ran1] == RowStatus.NOT_SIMULATED)


@pytest.mark.parametrize("settings", CONTINUOUS)
@pytest.mark.parametrize("redistribute_slack", [False, True], ids=["mismatch_shared", "loss_shared_first"])
def test_streamed_strategies_resume_their_contingency_outer_state(
    settings: dict[str, Any], redistribute_slack: bool
):
    """With the outer loops, a streamed strategy resumes from its contingency's loop state, kept
    in the same run as the contingency finishes: the two-phase rows (checked against
    lightsim2grid in test_topo_outer_warm_start.py)."""
    grid = _case14()
    kw = dict(
        outer_loops=True,
        init_from_n_powerflow=False,
        tol=TOL_MVA / grid.get_sn_mva(),
        redistribute_slack=redistribute_slack,
    )
    n_bus: int = grid.total_bus()
    V2, _ = _two_phases(_setup(grid, CONTINGENCIES, ACTIONS, settings, **kw), n_bus, "TRUE")
    V1, ran1 = _streamed(_setup(grid, CONTINGENCIES, ACTIONS, settings, **kw), n_bus, "TRUE")
    assert ran1.all()
    np.testing.assert_allclose(V1, V2, atol=1e-10, rtol=0)


def test_streaming_needs_the_continuous_schedule_for_a_condition():
    grid, ctgs, actions = _case118()
    ca = _setup(grid, ctgs, actions, {})
    ca.strategy_condition = "ANY_VIOLATION"
    with pytest.raises(NotImplementedError, match="continuous"):
        ca.compute(batch_size=BATCH)
