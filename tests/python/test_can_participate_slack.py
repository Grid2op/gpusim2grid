# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""redistribute_slack with lightsim2grid's "can participate in the slack" flag
(``LSGrid.set_gen_can_participate_slack``): a unit left out of the distributed
slack only because it sat at an active limit takes part in the bounded
redistribution pre-pass -- away from that limit, never across it -- and never in
the Newton solve's slack weights. The reference is lightsim2grid itself
(``consider_only_main_component(True)``) on a 5-bus grid whose leaf bus a
contingency islands.
"""

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [requires_gpu]


def _has_flag():
    try:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
    except ImportError:
        return False
    return hasattr(LSGrid, "set_gen_can_participate_slack")


pytestmark.append(pytest.mark.skipif(not _has_flag(),
                                     reason="needs a lightsim2grid with LSGrid.set_gen_can_participate_slack"))

VN_KV = 138.
CAPPED = 1          # gen 1: at its max_p, out of the slack
LEAF_LINE = 3       # line 1-4: taking it out islands bus 4
MAX_IT, TOL = 30, 1e-11


def _grid(flagged=True, leaf_gen_mw=0., overshoot_mw=0.):
    """lightsim2grid's own test_can_participate_slack grid: buses 0-1-2-3 in a row plus
    a leaf bus 4 off bus 1; 60 MW of load on bus 3, 20 MW on bus 4 (and a `leaf_gen_mw`
    generator there). Gen 0 is the slack; gen 1 sits at its max_p, out of the slack,
    flagged (or not) with the same weight as gen 0."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(5, 1, np.full(5, VN_KV), 0, 0)
    grid.init_powerlines(np.full(4, 0.01), np.full(4, 0.1), np.zeros(4, dtype=complex),
                         np.array([0, 1, 2, 1]), np.array([1, 2, 3, 4]))
    grid.init_loads(np.array([60., 20.]), np.array([10., 5.]), np.array([3, 4]))
    p = [30., 40.] + ([leaf_gen_mw] if leaf_gen_mw else [])
    bus = [0, 2] + ([4] if leaf_gen_mw else [])
    n = len(p)
    grid.init_generators_full(np.array(p), np.full(n, 1.02), np.zeros(n), [True] * n,
                              np.full(n, -1e3), np.full(n, 1e3), np.array(bus))
    grid.set_gen_p_limits(np.zeros(n), np.array([500., 40.] + ([100.] if leaf_gen_mw else [])))
    grid.add_gen_slackbus(0, 0.5)
    if flagged:
        grid.set_gen_can_participate_slack(np.array([False, True] + [False] * (n - 2)),
                                           np.array([0., 0.5] + [0.] * (n - 2)))
        if overshoot_mw:
            grid.set_gen_can_participate_slack_overshoot(np.array([0., overshoot_mw] + [0.] * (n - 2)))
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return grid


def _one_off(flagged, leaf_gen_mw=0., overshoot_mw=0.):
    grid = _grid(flagged, leaf_gen_mw, overshoot_mw)
    grid.deactivate_powerline(LEAF_LINE)
    report = grid.consider_only_main_component(True)
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return V, report


def _gpu(grid):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, tol_base=1e-10, handle_disconnected_grid=True,
                                redistribute_slack=True)
    ca.add_contingencies_by_branch_id([[LEAF_LINE]])
    ca.compute(batch_size=2)
    n_bus = grid.get_Ybus_solver().shape[0]
    V_solver = ca.V_results.to_numpy().reshape(1, n_bus)[0]
    me2s = np.asarray(grid.id_me_to_ac_solver())
    return ca, V_solver[me2s]


def test_data_marks_the_prepass_only_unit():
    from gpusim2grid import _gpusim2grid as _cpp
    grid = _grid(flagged=True)
    d = _cpp._extract_slack_redistribution_data_from_lsgrid(grid, grid.get_Ybus_solver().shape[0])
    assert list(d.el_id) == [0, CAPPED]
    assert list(d.in_slack) == [1, 0]
    np.testing.assert_allclose(d.weight, [0.5, 0.5])


@pytest.mark.parametrize("flagged, leaf_gen_mw", [(True, 0.), (False, 0.), (True, 30.)])
def test_island_matches_lightsim2grid(solver_atol, flagged, leaf_gen_mw):
    """20 MW of load islanded: the flagged unit takes half, down from its max_p (not
    flagged: the slack unit takes it all); an island producing more than it consumes: the
    flagged unit at its max_p cannot go up, the slack unit takes it all."""
    V_ref, report = _one_off(flagged, leaf_gen_mw)
    ca, V = _gpu(_grid(flagged, leaf_gen_mw))
    live = np.arange(4)
    atol = 10 * solver_atol
    np.testing.assert_allclose(np.abs(V[live]), np.abs(V_ref[live]), rtol=0., atol=atol)
    ang = lambda x: np.angle(x[live]) - np.angle(x[0])    # noqa: E731
    np.testing.assert_allclose(ang(V), ang(V_ref), rtol=0., atol=atol)
    rep = ca.get_slack_redistribution_report()
    assert rep["mismatch_mw"][0] == pytest.approx(report.mismatch_mw, abs=1e-9)
    assert rep["nb_participants"][0] == report.nb_participants
    assert rep["nb_participants"][0] == (2 if flagged else 1)


def _has_overshoot():
    try:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
    except ImportError:
        return False
    return hasattr(LSGrid, "set_gen_can_participate_slack_overshoot")


@pytest.mark.skipif(not _has_overshoot(),
                    reason="needs a lightsim2grid with LSGrid.set_gen_can_participate_slack_overshoot")
@pytest.mark.parametrize("overshoot_mw", [15., 25.])
def test_overshoot_matches_lightsim2grid(solver_atol, overshoot_mw):
    """The capped unit sat beyond its max_p in the reference solve: it only leaves it once
    the common shift has used that up (15 MW: it gives 2.5 of the 20 MW; 25 MW: nothing) --
    the data carries the overshoot and the GPU pre-pass gives lightsim2grid's voltages."""
    from gpusim2grid import _gpusim2grid as _cpp
    grid = _grid(True, 0., overshoot_mw)
    d = _cpp._extract_slack_redistribution_data_from_lsgrid(grid, grid.get_Ybus_solver().shape[0])
    np.testing.assert_allclose(d.overshoot_mw, [0., overshoot_mw])
    V_ref, report = _one_off(True, 0., overshoot_mw)
    ca, V = _gpu(_grid(True, 0., overshoot_mw))
    live = np.arange(4)
    atol = 10 * solver_atol
    np.testing.assert_allclose(np.abs(V[live]), np.abs(V_ref[live]), rtol=0., atol=atol)
    ang = lambda x: np.angle(x[live]) - np.angle(x[0])    # noqa: E731
    np.testing.assert_allclose(ang(V), ang(V_ref), rtol=0., atol=atol)
    assert ca.get_slack_redistribution_report()["mismatch_mw"][0] == pytest.approx(report.mismatch_mw, abs=1e-9)

