# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Contingencies tripping shunts, loads and generators, not only branches, and branches
opened at one end only.

``ContingencyAnalysisGPU.add_contingency`` / ``add_contingency_shunt`` / ``_load`` /
``_gen`` (lightsim2grid's ``ContingencyAnalysisCPP.add_contingency``), its
``branch_ids_side1`` / ``branch_ids_side2``, ``ScenarioSweepGPU.set_contingency_shunts`` /
``set_contingency_loads`` and ``BatchPowerFlow``'s ``shunt_status`` / ``load_status``; each
with ``redistribute_slack`` (what a trip takes out of the balance) and with
``outer_loops=True``.

Oracles: lightsim2grid's own batch classes on the same trips, and a one-off
``ac_pf`` on a fresh grid with the elements really deactivated.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge, is_fp32
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [requires_gpu,
              pytest.mark.skipif(not have_ls2g_bridge,
                                 reason="element trips need the lightsim2grid C++ bridge")]

MAX_IT, TOL = 20, 1e-10
# the GPU base case's tolerance: 1e-10 cannot be reached in FP32, which keeps the default
GPU_TOL = {} if is_fp32 else {"tol_base": TOL}

needs_outer = pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                                 reason="lightsim2grid without its outer loops")

# (branches, shunts, loads, generators) per contingency, on pandapower's case14:
# its shunt (bus 9), loads, and generators 1 and 2, each the only one regulating its bus
TRIPS = [
    ((), (0,), (), ()),
    ((), (), (3,), ()),
    ((), (), (), (1,)),
    ((5,), (0,), (2, 4), ()),
    ((7,), (), (1,), (2,)),
]


def _make_grid(algo="NR_KLU"):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(algo)
    n_bus = grid.get_bus_vn_kv().shape[0]
    v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
    v_ref = grid.ac_pf(v0.copy(), MAX_IT, TOL)
    assert v_ref.shape[0] > 0
    return grid, v_ref


def _reference_V(trip):
    grid, v_ref = _make_grid()
    branches, shunts, loads, gens = trip
    n_line = len(grid.get_lines())
    for b in branches:
        if b < n_line:
            grid.deactivate_powerline(int(b))
        else:
            grid.deactivate_trafo(int(b) - n_line)
    for s in shunts:
        grid.deactivate_shunt(int(s))
    for l in loads:
        grid.deactivate_load(int(l))
    for g in gens:
        grid.deactivate_gen(int(g))
    grid.tell_solver_need_reset()
    V = grid.ac_pf(v_ref.copy(), MAX_IT, TOL)
    assert V.shape[0] > 0, "the reference powerflow itself diverged"
    return V


@pytest.fixture(scope="module")
def case():
    grid, v_ref = _make_grid()
    assert len(grid.get_shunts()) > 0
    return {"grid": grid, "v_ref": v_ref,
            "n_bus": grid.get_Ybus_solver().shape[0],
            "buses": np.asarray(grid.id_ac_solver_to_me(), dtype=int)}


def _ls_rows(ls, trips):
    """lightsim2grid's voltages of `trips`, in their order (its rows follow
    my_contingencies())"""
    conts = ls.my_contingencies()
    row = {tuple(tuple(el) for el in c): k for k, c in enumerate(conts)}
    V = np.asarray(ls.get_voltages())
    # my_contingencies' tuples grew with lightsim2grid: 6 lists (the branches opened at one end
    # only as the fifth and sixth), 10 in 1.3.1 (+ static generators, storage units, SVCs, hvdc
    # lines). Pad each trip to the length this version returns.
    n = len(conts[0])
    assert all(len(t) <= n for t in trips)
    return V[[row[tuple(tuple(sorted(el)) for el in (tuple(t) + ((),) * n)[:n])] for t in trips]]


# ------------------------------------------------------------ ContingencyAnalysisGPU
def test_ca_matches_one_off_powerflows_and_lightsim2grid(case, solver_atol, residual_atol):
    """Every trip of TRIPS converges, equals a one-off ac_pf on a grid with those elements
    deactivated and lightsim2grid's ContingencyAnalysisCPP, and moves the state by more
    than that tolerance (the comparison could not pass with the trip ignored)."""
    from gpusim2grid import ContingencyAnalysisGPU
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisGPU(case["grid"], nb_iter=10, **GPU_TOL)
    ca.add_contingencies(TRIPS)
    ca.compute(batch_size=4)
    assert np.all(ca.last_residuals() < residual_atol)
    V = ca.V_results.to_numpy().reshape(len(TRIPS), case["n_bus"])
    assert ca.get_contingencies() == [tuple(sorted(el) for el in t) + ([], []) for t in TRIPS]

    ls = ContingencyAnalysisCPP(case["grid"])
    for t in TRIPS:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * case["v_ref"], MAX_IT, TOL)
    assert all(ls.converged_mask())
    V_ls = _ls_rows(ls, TRIPS)[:, case["buses"]]
    V_n = case["v_ref"][case["buses"]]
    for r, t in enumerate(TRIPS):
        ref = _reference_V(t)[case["buses"]]
        np.testing.assert_allclose(V[r], ref, atol=10 * solver_atol, err_msg=f"trip {t}")
        np.testing.assert_allclose(V[r], V_ls[r], atol=10 * solver_atol, err_msg=f"trip {t}")
        assert np.abs(ref - V_n).max() > 100 * solver_atol   # the trip does move the state


def test_ca_shortcuts_append(case, solver_atol):
    """add_contingency_shunt / _load / _gen append to add_contingencies_by_branch_id's list
    (ids sorted), give the same rows as add_contingencies with those tuples, and refuse an
    unknown id."""
    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(case["grid"], nb_iter=10, **GPU_TOL)
    ca.add_contingencies_by_branch_id([[3], [5]])
    ca.add_contingency_shunt(0)
    ca.add_contingency_load([3, 1])
    ca.add_contingency_gen(1)
    assert ca.get_contingencies() == [([3], [], [], [], [], []), ([5], [], [], [], [], []),
                                      ([], [0], [], [], [], []), ([], [], [1, 3], [], [], []),
                                      ([], [], [], [1], [], [])]
    ca.compute(batch_size=8)
    V = ca.V_results.to_numpy().reshape(5, case["n_bus"])
    ref = ContingencyAnalysisGPU(case["grid"], nb_iter=10, **GPU_TOL)
    ref.add_contingencies([((3,), (), (), ()), ((5,), (), (), ()), ((), (0,), (), ()),
                           ((), (), (1, 3), ()), ((), (), (), (1,))])
    ref.compute(batch_size=8)
    np.testing.assert_allclose(V, ref.V_results.to_numpy().reshape(5, case["n_bus"]), rtol=0., atol=solver_atol)
    with pytest.raises(RuntimeError, match="no shunt with id"):
        ca.add_contingency_shunt(len(case["grid"].get_shunts()))


def test_ca_redistribute_slack_matches_lightsim2grid(solver_atol):
    """The slack pre-pass counts what a tripped load / generator / shunt takes out of the
    balance, as lightsim2grid does. Two slack participants (generators 0 and 1), generator 0
    1 MW above its min_p: the 18.8 MW of loads the first contingency trips saturate it (it
    leaves that row's slack), which moves that row away from the analysis without the
    pre-pass; the second one (a shunt and a generator at 0 MW) loses nothing and stays put."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        net.gen["slack"] = False
        net.gen.loc[[0, 1], "slack"] = True
        net.gen["slack_weight"] = 1.
        net.gen.loc[1, "p_mw"] = 30.      # room to take less
        grid = init_from_pandapower(net)
    grid.change_algorithm("NR_KLU")
    n_gen = len(grid.get_generators())
    assert [g.id for g in grid.get_generators() if g.is_slack] == [0, 1]
    min_p = np.full(n_gen, -1e4)
    min_p[0] = float(grid.get_gen_target_p()[0]) - 1.
    grid.set_gen_p_limits(min_p, np.full(n_gen, 1e4))
    trips = [((), (), (3, 4), ()), ((), (0,), (), (2,))]
    n_bus = grid.get_bus_vn_kv().shape[0]
    v_ref = grid.ac_pf(np.ones(n_bus, dtype=complex), MAX_IT, TOL)
    assert v_ref.shape[0] > 0

    ls = ContingencyAnalysisCPP(grid)
    ls.redistribute_slack = True
    for t in trips:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, redistribute_slack=True, **GPU_TOL)
    ca.add_contingencies(trips)
    ca.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(trips), -1)
    assert np.all(np.isfinite(V))
    np.testing.assert_allclose(V, _ls_rows(ls, trips)[:, buses], atol=10 * solver_atol)
    assert ca.get_slack_redistribution_report()["nb_saturated"][0] >= 1
    off = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    off.add_contingencies(trips)
    off.compute(batch_size=4)
    moved = np.abs(V - off.V_results.to_numpy().reshape(len(trips), -1)).max(axis=1)
    assert moved[0] > 100 * solver_atol
    assert moved[1] < solver_atol


@needs_outer
def test_ca_outer_loops_match_lightsim2grid(solver_atol):
    """outer_loops=True: the rows run the grid's own loops on the edited system and match
    lightsim2grid's batch -- voltages, row statuses and outer iteration counts."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU

    grid, v_ref = _make_grid("NROuter_KLU")
    trips = [((), (0,), (), ()), ((), (), (3,), ()), ((5,), (), (2,), ())]
    ls = ContingencyAnalysisCPP(grid)
    ls.init_from_n_powerflow = True
    for t in trips:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # lightsim2grid's tol is in MVA, the GPU's per unit
        ca = ContingencyAnalysisGPU(grid, outer_loops=True, scheduling="continuous", nb_iter=30,
                                    init_from_n_powerflow=True, tol=TOL / grid.get_sn_mva(), ir_n_steps=1)
    ca.add_contingencies(trips)
    ca.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(trips), -1)
    np.testing.assert_allclose(V, _ls_rows(ls, trips)[:, buses], rtol=0., atol=solver_atol)
    assert np.all(np.asarray(ca.get_row_status()) == 0)   # CONVERGED
    res, res_ls = ca.get_outer_results(), _ls_outer_rows(ls, trips)
    assert [r.nb_outer_iterations for r in res] == [r.nb_outer_iterations for r in res_ls]


def _ls_outer_rows(ls, trips):
    """lightsim2grid's outer results of `trips`, in their order (as _ls_rows)"""
    conts = ls.my_contingencies()
    row = {tuple(tuple(el) for el in c): k for k, c in enumerate(conts)}
    n = len(conts[0])
    res = ls.get_outer_results()
    return [res[row[tuple(tuple(sorted(el)) for el in (tuple(t) + ((),) * n)[:n])]] for t in trips]


# branches opened at one end only (branch_ids_side1 / _side2): case14's line 3 at side 1 (it
# has charging, which shows there), line 5 at side 2, a transformer at side 1 (no shunt
# admittance: half-open or tripped is the same), and line 4 at both ends (a trip)
ONE_SIDE_TRIPS = [((), (), (), (), (3,), ()), ((), (), (), (), (), (5,)),
                  ((), (), (), (), (15,), ()), ((), (), (), (), (4,), (4,))]


def _reference_one_side(trip):
    grid, v_ref = _make_grid()
    grid.set_synch_status_both_side(False)
    n_line = len(grid.get_lines())
    for side, ids in ((1, trip[4]), (2, trip[5])):
        for b in ids:
            if b < n_line:
                getattr(grid, f"deactivate_powerline_side{side}")(int(b))
            else:
                getattr(grid, f"deactivate_trafo_side{side}")(int(b) - n_line)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(v_ref.copy(), MAX_IT, TOL)
    assert V.shape[0] > 0, "the reference powerflow itself diverged"
    return V


@pytest.mark.parametrize("outer_loops", [False, pytest.param(True, marks=needs_outer)])
def test_ca_one_side_trips_match_lightsim2grid(outer_loops, solver_atol):
    """A branch opened at one end keeps its other one, with its charging, as lightsim2grid's
    batch has it (with and without the outer loops). Without them, also as a half-open branch
    of the grid in a one-off ac_pf: line 3 differs from its trip, the transformer (no shunt
    admittance) is its trip. No current is reported at either end of an opened branch."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU

    grid, v_ref = _make_grid("NROuter_KLU" if outer_loops else "NR_KLU")
    assert len(grid.get_lines()) <= 15 < len(grid.get_lines()) + len(grid.get_trafos())
    ls = ContingencyAnalysisCPP(grid)
    ls.init_from_n_powerflow = outer_loops
    for t in ONE_SIDE_TRIPS:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if outer_loops:
            ca = ContingencyAnalysisGPU(grid, outer_loops=True, scheduling="continuous", nb_iter=30,
                                        init_from_n_powerflow=True, tol=TOL / grid.get_sn_mva(), ir_n_steps=1)
        else:
            ca = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    ca.add_contingencies(ONE_SIDE_TRIPS)
    # both ends: a trip
    assert ca.get_contingencies()[3] == ([4], [], [], [], [], [])
    ca.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(ONE_SIDE_TRIPS), -1)
    trips_ls = [((4,), (), (), (), (), ()) if k == 3 else t for k, t in enumerate(ONE_SIDE_TRIPS)]
    V_ls = _ls_rows(ls, trips_ls)
    atol = 10 * solver_atol
    np.testing.assert_allclose(V, V_ls[:, buses], rtol=0., atol=atol)
    if not outer_loops:
        for r, t in enumerate(ONE_SIDE_TRIPS[:3]):
            ref = _reference_one_side(t)[buses]
            np.testing.assert_allclose(V[r], ref, rtol=0., atol=atol, err_msg=f"trip {t}")
            tripped = _reference_V((tuple(t[4]) + tuple(t[5]), (), (), ()))[buses]
            if r == 0:
                # line 3 keeps its charging at the closed end: not the same as its trip (line
                # 5's charging is too small to show)
                assert np.abs(ref - tripped).max() > 100 * solver_atol, f"trip {t}"
            elif r == 2:
                # the transformer has no shunt admittance: half-open or tripped is the same
                np.testing.assert_allclose(ref, tripped, rtol=0., atol=atol, err_msg=f"trip {t}")
    # no current reported at either end of a branch opened at one end (or both)
    ca.compute_flows()
    n_rows = len(ONE_SIDE_TRIPS)
    for amps in (ca.or_amps.to_numpy().reshape(n_rows, -1), ca.ex_amps.to_numpy().reshape(n_rows, -1)):
        assert amps[0, 3] == 0. and amps[1, 5] == 0. and amps[2, 15] == 0. and amps[3, 4] == 0.


# ------------------------------------------------------------------ ScenarioSweepGPU
def test_sweep_shunt_and_load_masks(case, solver_atol):
    """ScenarioSweepGPU's shunt and load masks (the load mask set before the injections, the
    shunt mask after a topology), one row each plus one combining both with a line trip:
    lightsim2grid's ScenarioSweepCPP on the same masks, and a one-off ac_pf for the shunt."""
    from gpusim2grid import ScenarioSweepGPU
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid = case["grid"]
    n_rows, n_shunt, n_load = 3, len(grid.get_shunts()), len(grid.get_loads())
    shunt_off = np.zeros((n_rows, n_shunt), dtype=bool)
    load_off = np.zeros((n_rows, n_load), dtype=bool)
    shunt_off[0, 0] = True
    load_off[1, 3] = True
    shunt_off[2, 0] = True
    load_off[2, [2, 4]] = True
    topology = [[], [], [5]]

    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None, :], n_rows, axis=0)   # noqa: E731
    sw = ScenarioSweepGPU(grid, nb_iter=10, **GPU_TOL)
    sw.set_contingency_loads(load_off)          # before the injections: applied anyway
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_topology(topology)
    sw.set_contingency_shunts(shunt_off)
    sw.compute(batch_size=4)
    V = sw.solver.V_results.to_numpy().reshape(n_rows, case["n_bus"])

    ls = ScenarioSweepCPP(grid)
    line_off = np.zeros((n_rows, len(grid.get_lines())), dtype=bool)
    line_off[2, 5] = True
    ls.set_contingency_lines(line_off)
    ls.set_contingency_shunts(shunt_off)
    ls.set_contingency_loads(load_off)
    ls.compute(1.0 * case["v_ref"], MAX_IT, TOL)
    assert all(ls.converged_mask())
    assert np.all(np.isfinite(V))
    np.testing.assert_allclose(V, np.asarray(ls.get_voltages())[:, case["buses"]], atol=10 * solver_atol)
    np.testing.assert_allclose(V[0], _reference_V(((), (0,), (), ()))[case["buses"]], atol=10 * solver_atol)


def _grid_resistive_shunt(algo):
    """case14 with two slack participants (the external grid and generator 0) and its shunt
    given an active part: tripping it changes the active balance the pre-pass shares."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        net.gen["slack"] = False
        net.gen.loc[0, "slack"] = True
        net.shunt["p_mw"] = 8.
        grid = init_from_pandapower(net)
    grid.change_algorithm(algo)
    v_ref = grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert v_ref.shape[0] > 0
    return grid, v_ref


@pytest.mark.parametrize("outer_loops", [False, True])
def test_sweep_redistribute_slack_counts_a_tripped_shunt(outer_loops, solver_atol):
    """ScenarioSweepGPU's redistribute_slack counts the active power of a shunt its row
    trips (lightsim2grid's term (e), not in the AC Sbus), as ContingencyAnalysisGPU does and
    as lightsim2grid's ScenarioSweep has it."""
    from gpusim2grid import ScenarioSweepGPU, ContingencyAnalysisGPU
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    algo = "NROuter_KLU" if outer_loops else "NR_KLU"
    grid, v_ref = _grid_resistive_shunt(algo)
    n_rows, n_shunt = 2, len(grid.get_shunts())
    shunt_off = np.zeros((n_rows, n_shunt), dtype=bool)
    shunt_off[0, 0] = True
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None, :], n_rows, axis=0)   # noqa: E731
    kw = dict(outer_loops=True, scheduling="continuous", nb_iter=30, init_from_n_powerflow=False,
              tol=TOL / grid.get_sn_mva(), ir_n_steps=1) if outer_loops else dict(nb_iter=10, **GPU_TOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sw = ScenarioSweepGPU(grid, redistribute_slack=True, **kw)
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_contingency_shunts(shunt_off)
    sw.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = sw.solver.V_results.to_numpy().reshape(n_rows, -1)

    ls = ScenarioSweepCPP(grid)
    ls.redistribute_slack = True
    ls.init_from_n_powerflow = False
    ls.modify_gen_p(rep(gen_p))
    ls.modify_load_p(rep(load_p))
    ls.modify_load_q(rep(load_q))
    ls.set_contingency_shunts(shunt_off)
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    np.testing.assert_allclose(V, np.asarray(ls.get_voltages())[:, buses], rtol=0., atol=10 * solver_atol)

    # the same trip through the contingency analysis, which counts it already
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca = ContingencyAnalysisGPU(grid, redistribute_slack=True, **kw)
    ca.add_contingency_shunt(0)
    ca.compute(batch_size=1)
    np.testing.assert_allclose(V[0], ca.V_results.to_numpy(), rtol=0., atol=10 * solver_atol)
    if not outer_loops:
        # the shunt's 8 MW (at 1 pu, load convention) come back to the units
        assert sw.get_slack_redistribution_report()["mismatch_mw"][0] == pytest.approx(-8., abs=1e-6)


# -------------------------------------------------------------------- BatchPowerFlow
def test_batch_power_flow_statuses(case, solver_atol):
    """BatchPowerFlow's shunt_status / load_status (True = connected): lightsim2grid's
    ScenarioSweepCPP with the complementary masks; a tripped load's load_p gets no gradient,
    a connected one's gradient matches a central difference."""
    torch = pytest.importorskip("torch")
    from gpusim2grid.differentiable import BatchPowerFlow
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid = case["grid"]
    n_rows, n_shunt, n_load = 3, len(grid.get_shunts()), len(grid.get_loads())
    shunt_status = np.ones((n_rows, n_shunt), dtype=bool)
    load_status = np.ones((n_rows, n_load), dtype=bool)
    shunt_status[0, 0] = False
    load_status[1, 3] = False
    load_status[2, [2, 4]] = False

    pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=10)
    load_p0 = np.asarray(grid.get_loads_res_full()[0], dtype=np.float64)
    load_p = torch.tensor(np.repeat(load_p0[None, :], n_rows, axis=0), device="cuda", requires_grad=True)
    V = pf(load_p=load_p, shunt_status=torch.as_tensor(shunt_status, device="cuda"),
           load_status=torch.as_tensor(load_status, device="cuda"))

    ls = ScenarioSweepCPP(grid)
    ls.set_contingency_shunts(~shunt_status)
    ls.set_contingency_loads(~load_status)
    ls.compute(1.0 * case["v_ref"], MAX_IT, TOL)
    assert all(ls.converged_mask())
    np.testing.assert_allclose(V.detach().cpu().numpy(), np.asarray(ls.get_voltages())[:, case["buses"]],
                               atol=10 * solver_atol)

    # a tripped load is out of its row: no gradient there
    (V.real.sum() + V.imag.sum()).backward()
    g = load_p.grad.cpu().numpy()
    assert g[1, 3] == 0. and g[2, 2] == 0. and g[2, 4] == 0.
    assert abs(g[0, 3]) > 0.

    # central differences on connected loads (row 0 has its shunt off, row 1 a load off)
    statuses = dict(shunt_status=torch.as_tensor(shunt_status, device="cuda"),
                    load_status=torch.as_tensor(load_status, device="cuda"))
    h = 1e-2 if is_fp32 else 1e-4   # MW

    def f(lp):
        with torch.no_grad():
            Vh = pf(load_p=lp, **statuses)
            return float((Vh.real.sum() + Vh.imag.sum()).item())
    for r, l in ((0, 3), (1, 2), (2, 1)):
        plus = load_p.detach().clone()
        minus = load_p.detach().clone()
        plus[r, l] += h
        minus[r, l] -= h
        fd = (f(plus) - f(minus)) / (2 * h)
        assert g[r, l] == pytest.approx(fd, rel=1e3 * solver_atol, abs=solver_atol), (r, l)
