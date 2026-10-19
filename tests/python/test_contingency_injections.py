# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Contingencies tripping shunts, loads and generators, not only branches.

``ContingencyAnalysisGPU.add_contingency`` / ``add_contingency_shunt`` / ``_load`` /
``_gen`` (lightsim2grid's ``ContingencyAnalysisCPP.add_contingency``),
``ScenarioSweepGPU.set_contingency_shunts`` / ``set_contingency_loads`` and
``BatchPowerFlow``'s ``shunt_status`` / ``load_status``.

Oracles: lightsim2grid's own batch classes on the same trips, and a one-off
``ac_pf`` on a fresh grid with the elements really deactivated.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = [requires_gpu,
              pytest.mark.skipif(not have_ls2g_bridge,
                                 reason="element trips need the lightsim2grid C++ bridge")]

MAX_IT, TOL = 20, 1e-10

# (branches, shunts, loads, generators) per contingency, on pandapower's case14:
# its shunt (bus 9), loads, and generator 1, the only one regulating its bus
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
    row = {tuple(tuple(el) for el in c): k for k, c in enumerate(ls.my_contingencies())}
    V = np.asarray(ls.get_voltages())
    return V[[row[tuple(tuple(sorted(el)) for el in t)] for t in trips]]


# ------------------------------------------------------------ ContingencyAnalysisGPU
def test_ca_matches_one_off_powerflows_and_lightsim2grid(case, solver_atol):
    from gpusim2grid import ContingencyAnalysisGPU
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisGPU(case["grid"], nb_iter=10, tol_base=TOL)
    ca.add_contingencies(TRIPS)
    ca.compute(batch_size=4)
    assert np.all(ca.last_residuals() < 100 * solver_atol)
    V = ca.V_results.to_numpy().reshape(len(TRIPS), case["n_bus"])
    assert ca.get_contingencies() == [tuple(sorted(el) for el in t) for t in TRIPS]

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
        assert np.abs(ref - V_n).max() > 1e-4   # the trip does move the state


def test_ca_shortcuts_append(case):
    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(case["grid"], nb_iter=10, tol_base=TOL)
    ca.add_contingencies_by_branch_id([[3], [5]])
    ca.add_contingency_shunt(0)
    ca.add_contingency_load([3, 1])
    ca.add_contingency_gen(1)
    assert ca.get_contingencies() == [([3], [], [], []), ([5], [], [], []), ([], [0], [], []),
                                      ([], [], [1, 3], []), ([], [], [], [1])]
    ca.compute(batch_size=8)
    assert ca.V_results.to_numpy().shape[0] == 5 * case["n_bus"]
    with pytest.raises(Exception):
        ca.add_contingency_shunt(len(case["grid"].get_shunts()))


def test_ca_redistribute_slack_matches_lightsim2grid(solver_atol):
    """The slack pre-pass counts what a tripped load / generator takes out of the
    balance, as lightsim2grid does (two slack participants: one may saturate)."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        net.gen["slack"] = False
        net.gen.loc[0, "slack"] = True
        grid = init_from_pandapower(net)
    grid.change_algorithm("NR_KLU")
    trips = [((), (), (3, 4), ()), ((), (0,), (), (2,))]
    n_bus = grid.get_bus_vn_kv().shape[0]
    v_ref = grid.ac_pf(np.ones(n_bus, dtype=complex), MAX_IT, TOL)
    assert v_ref.shape[0] > 0

    ls = ContingencyAnalysisCPP(grid)
    ls.redistribute_slack = True
    for t in trips:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, tol_base=TOL, redistribute_slack=True)
    ca.add_contingencies(trips)
    ca.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(trips), -1)
    np.testing.assert_allclose(V, _ls_rows(ls, trips)[:, buses], atol=10 * solver_atol)


def test_ca_outer_loops_match_lightsim2grid(solver_atol):
    """outer_loops=True: the rows run the grid's own loops on the edited system."""
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
        ca = ContingencyAnalysisGPU(grid, outer_loops=True, scheduling="continuous", nb_iter=30,
                                    init_from_n_powerflow=True, tol=TOL, ir_n_steps=1)
    ca.add_contingencies(trips)
    ca.compute(batch_size=4)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(len(trips), -1)
    np.testing.assert_allclose(V, _ls_rows(ls, trips)[:, buses], atol=1e3 * solver_atol)


# ------------------------------------------------------------------ ScenarioSweepGPU
def test_sweep_shunt_and_load_masks(case, solver_atol):
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
    sw = ScenarioSweepGPU(grid, nb_iter=10, tol_base=TOL)
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
    np.testing.assert_allclose(V, np.asarray(ls.get_voltages())[:, case["buses"]], atol=10 * solver_atol)
    np.testing.assert_allclose(V[0], _reference_V(((), (0,), (), ()))[case["buses"]], atol=10 * solver_atol)


# -------------------------------------------------------------------- BatchPowerFlow
def test_batch_power_flow_statuses(case, solver_atol):
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
    np.testing.assert_allclose(V.detach().cpu().numpy(), np.asarray(ls.get_voltages())[:, case["buses"]],
                               atol=10 * solver_atol)

    # a tripped load is out of its row: no gradient there
    (V.real.sum() + V.imag.sum()).backward()
    g = load_p.grad.cpu().numpy()
    assert g[1, 3] == 0. and g[2, 2] == 0. and g[2, 4] == 0.
    assert abs(g[0, 3]) > 0.
