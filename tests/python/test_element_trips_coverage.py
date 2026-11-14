# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""More coverage of contingencies tripping shunts, loads, generators and branches
opened at one end only (complements test_contingency_injections.py):

- a regulating shunt tripped with the outer loops (refused);
- BatchPowerFlow with load, shunt and generator statuses together (forward and
  load_p gradient);
- trips on rows that also island a bus (handle_disconnected_grid), on a
  scenario sweep and on a contingency analysis with branches opened at one end;
- current limits of a branch opened at one end;
- a batch mixing every kind of trip, repeated.

Oracles: lightsim2grid's ContingencyAnalysisCPP / ScenarioSweepCPP on the same
trips, and one-off ``ac_pf`` on a grid with the elements really deactivated.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

from test_contingency_injections import GPU_TOL, MAX_IT, TOL, _make_grid, _ls_rows

pytestmark = [requires_gpu,
              pytest.mark.skipif(not have_ls2g_bridge,
                                 reason="element trips need the lightsim2grid C++ bridge")]

# case14 (pandapower, 0-based buses): bus 7 is fed by transformer 3 (branch 18) only and
# carries generator 3; bus 11 is fed by line 8 (5-11) and line 13 (11-12) and carries load 8
N_LINE = 15
LEAF_TRAFO, LEAF_TRAFO_BUS = N_LINE + 3, 7
BUS11_LINES, BUS11, BUS11_LOAD = (8, 13), 11, 8


def _solver_pos(buses, grid_bus):
    """position, in solver order, of grid bus `grid_bus`"""
    return int(np.flatnonzero(buses == grid_bus)[0])


def _reference_full(trip):
    """one-off ac_pf of case14 with (branches, shunts, loads, gens, side1, side2)
    really deactivated (every bus stays connected)"""
    grid, v_ref = _make_grid()
    branches, shunts, loads, gens, side1, side2 = (tuple(trip) + ((), ()))[:6]
    grid.set_synch_status_both_side(False)
    for b in branches:
        if b < N_LINE:
            grid.deactivate_powerline(int(b))
        else:
            grid.deactivate_trafo(int(b) - N_LINE)
    for side, ids in ((1, side1), (2, side2)):
        for b in ids:
            if b < N_LINE:
                getattr(grid, f"deactivate_powerline_side{side}")(int(b))
            else:
                getattr(grid, f"deactivate_trafo_side{side}")(int(b) - N_LINE)
    for s in shunts:
        grid.deactivate_shunt(int(s))
    for ld in loads:
        grid.deactivate_load(int(ld))
    for g in gens:
        grid.deactivate_gen(int(g))
    grid.tell_solver_need_reset()
    V = grid.ac_pf(v_ref.copy(), MAX_IT, TOL)
    assert V.shape[0] > 0, "the reference powerflow itself diverged"
    return V


# ------------------------------------------------------------------- 1. regulating shunt
def _regulating_shunt_grid():
    """lightsim2grid's own ShuntVoltageControl test grid: pypowsybl's IEEE 14 with two
    linear shunts regulating bus B9 (as test_outer_model_cpu's _shunt_grid), solved
    with NROuter_KLU"""
    pytest.importorskip("pypowsybl")
    from test_outer_model_cpu import _ls2g_tests_dir
    _ls2g_tests_dir()
    try:
        import test_shunt_voltage_control as S
    except ImportError:
        pytest.skip("lightsim2grid's test_shunt_voltage_control is not shipped")
    from lightsim2grid.algorithm import ShuntVoltageControl
    grid = S._grid(S._net("B9", 12.8, (3, 1)))
    grid.clear_outer_loops()
    grid.add_outer_loop(ShuntVoltageControl())
    grid.change_algorithm("NROuter_KLU")
    V = grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    names = [sh.name for sh in grid.get_shunts()]
    return grid, [names.index(n) for n in S.SHUNTS], names


def _outer_ca(grid):
    from gpusim2grid import ContingencyAnalysisGPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ContingencyAnalysisGPU(grid, outer_loops=True, scheduling="continuous", nb_iter=30,
                                      ir_n_steps=1, tol=TOL / grid.get_sn_mva())


def test_outer_loops_trip_of_regulating_shunt_raises_at_compute():
    """With outer_loops=True on a grid whose ShuntVoltageControl loop holds two shunts,
    a contingency tripping one of them makes compute() raise a RuntimeError whose message
    names ShuntVoltageControl (current behaviour: lightsim2grid instead leaves that row
    out as BATCH_UNSUPPORTED). A contingency tripping a line on the same grid computes."""
    grid, reg, _ = _regulating_shunt_grid()
    ca = _outer_ca(grid)
    ca.add_contingency_shunt(reg[0])
    with pytest.raises(RuntimeError, match="ShuntVoltageControl"):
        ca.compute(batch_size=4)

    ok = _outer_ca(grid)
    ok.add_contingencies_by_branch_id([[1]])
    ok.compute(batch_size=4)
    assert np.all(np.isfinite(ok.V_results.to_numpy()))


# ------------------------------------------------------- 2. BatchPowerFlow, all statuses
def test_batch_power_flow_load_shunt_gen_status_together(solver_atol):
    """BatchPowerFlow with load_status, shunt_status and gen_status on the same batch
    (one row trips all three kinds, generator 1 being its bus' only regulator): V equals
    lightsim2grid's ScenarioSweepCPP with the negated masks, and the load_p gradient
    of a weighted |V|^2 loss equals central differences on three entries (rows with a
    generator trip included); a tripped load gets a zero gradient."""
    torch = pytest.importorskip("torch")
    from gpusim2grid.differentiable import BatchPowerFlow
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid, v_ref = _make_grid()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    n_rows, n_shunt, n_load, n_gen = 4, len(grid.get_shunts()), len(grid.get_loads()), len(grid.get_generators())
    shunt_status = np.ones((n_rows, n_shunt), dtype=bool)
    load_status = np.ones((n_rows, n_load), dtype=bool)
    gen_status = np.ones((n_rows, n_gen), dtype=bool)
    shunt_status[0, 0] = False
    load_status[0, 3] = False
    gen_status[0, 1] = False
    load_status[1, [2, 4]] = False
    gen_status[2, 2] = False
    shunt_status[2, 0] = False
    # row 3: nothing tripped

    ls = ScenarioSweepCPP(grid)
    ls.set_contingency_shunts(~shunt_status)
    ls.set_contingency_loads(~load_status)
    ls.set_contingency_gens(~gen_status)
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    V_ls = np.asarray(ls.get_voltages())[:, buses]

    pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=10, **GPU_TOL)
    st = {k: torch.as_tensor(v, device="cuda") for k, v in
          (("shunt_status", shunt_status), ("load_status", load_status), ("gen_status", gen_status))}
    load_p0 = np.asarray(grid.get_loads_res_full()[0], dtype=np.float64)
    w = torch.linspace(0.5, 1.5, pf.n_bus, dtype=torch.float64, device="cuda")

    def loss(lp):
        V = pf(load_p=lp, **st)
        return V, ((V.abs() ** 2) * w).sum()

    load_p = torch.tensor(np.repeat(load_p0[None, :], n_rows, axis=0), device="cuda", requires_grad=True)
    V, L = loss(load_p)
    np.testing.assert_allclose(V.detach().cpu().numpy(), V_ls, atol=10 * solver_atol)
    L.backward()
    g = load_p.grad.detach().clone()
    assert g[0, 3].item() == 0. and g[1, 2].item() == 0. and g[1, 4].item() == 0.

    eps = 1e-6
    for idx in ((0, 5), (2, 1), (3, 7)):
        xp = load_p.detach().clone(); xp[idx] += eps
        xm = load_p.detach().clone(); xm[idx] -= eps
        with torch.no_grad():
            fd = (loss(xp)[1].item() - loss(xm)[1].item()) / (2 * eps)
        an = g[idx].item()
        assert abs(fd) > 0.
        assert abs(an - fd) <= 100 * solver_atol * max(1.0, abs(fd)), (idx, an, fd)


# ---------------------------------------------- 3. scenario sweep, trips + islanded bus
def test_sweep_load_shunt_trips_on_islanding_rows_match_lightsim2grid(solver_atol):
    """ScenarioSweepGPU(handle_disconnected_grid=True), rows tripping loads / shunts
    while set_topology islands bus 11 (lines 8 and 13) or bus 7 (transformer 3), one row
    also tripping the islanded bus' own load: the islanded bus (and only it) is NaN on our
    side, every row's residual is finite, and the other entries equal lightsim2grid's
    ScenarioSweepCPP with handle_disconnected_grid=True on the same masks."""
    from gpusim2grid import ScenarioSweepGPU
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid, v_ref = _make_grid()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    n_bus = len(buses)
    n_rows, n_shunt, n_load = 4, len(grid.get_shunts()), len(grid.get_loads())
    topology = [list(BUS11_LINES), [LEAF_TRAFO], list(BUS11_LINES), []]
    island = [BUS11, LEAF_TRAFO_BUS, BUS11, None]
    shunt_off = np.zeros((n_rows, n_shunt), dtype=bool)
    load_off = np.zeros((n_rows, n_load), dtype=bool)
    shunt_off[0, 0] = True
    shunt_off[1, 0] = True
    load_off[1, 3] = True
    load_off[2, [BUS11_LOAD, 2]] = True
    load_off[3, 4] = True

    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None, :], n_rows, axis=0)   # noqa: E731
    sw = ScenarioSweepGPU(grid, nb_iter=10, handle_disconnected_grid=True, **GPU_TOL)
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_topology(topology)
    sw.set_contingency_shunts(shunt_off)
    sw.set_contingency_loads(load_off)
    sw.compute(batch_size=4)
    V = sw.solver.V_results.to_numpy().reshape(n_rows, n_bus)
    assert np.all(np.isfinite(sw.solver.residuals.to_numpy()))

    ls = ScenarioSweepCPP(grid)
    ls.handle_disconnected_grid = True
    line_off = np.zeros((n_rows, N_LINE), dtype=bool)
    trafo_off = np.zeros((n_rows, len(grid.get_trafos())), dtype=bool)
    for r, ids in enumerate(topology):
        for b in ids:
            if b < N_LINE:
                line_off[r, b] = True
            else:
                trafo_off[r, b - N_LINE] = True
    ls.set_contingency_lines(line_off)
    ls.set_contingency_trafos(trafo_off)
    ls.set_contingency_shunts(shunt_off)
    ls.set_contingency_loads(load_off)
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    V_ls = np.asarray(ls.get_voltages())[:, buses]

    for r in range(n_rows):
        nan = ~np.isfinite(V[r])
        expected = np.zeros(n_bus, dtype=bool)
        if island[r] is not None:
            expected[_solver_pos(buses, island[r])] = True
        np.testing.assert_array_equal(nan, expected, err_msg=f"row {r}")
        np.testing.assert_allclose(V[r, ~nan], V_ls[r, ~nan], atol=10 * solver_atol, err_msg=f"row {r}")


# ------------------------------------- 4. contingency analysis, opened at one end + island
ISLAND_ONE_SIDE = [((), (), (), (), (), (LEAF_TRAFO,)),            # bus 7 islanded, trafo kept on bus 6
                   ((), (), (), (), (LEAF_TRAFO,), ()),            # bus 7 islanded, trafo kept on bus 7
                   ((13,), (), (), (), (), (8,)),                  # bus 11 islanded, line 8 kept on bus 5
                   ((13,), (0,), (3,), (), (8,), ())]              # + a shunt and a load
ISLAND_ONE_SIDE_BUS = [LEAF_TRAFO_BUS, LEAF_TRAFO_BUS, BUS11, BUS11]


def test_ca_one_side_opening_islanding_a_bus_matches_lightsim2grid(solver_atol):
    """ContingencyAnalysisGPU(handle_disconnected_grid=True) with branches opened at one
    end that island a bus (transformer 3 at either side -> bus 7; line 13 tripped and
    line 8 opened at either side -> bus 11, one row also tripping a shunt and a load): the
    islanded bus (and only it) is NaN, every residual is finite, and the other entries
    equal lightsim2grid's ContingencyAnalysisCPP(handle_disconnected_grid=True) given the
    same add_contingency(..., branch_ids_side1/side2)."""
    from gpusim2grid import ContingencyAnalysisGPU
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    grid, v_ref = _make_grid()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    ls = ContingencyAnalysisCPP(grid)
    ls.handle_disconnected_grid = True
    for t in ISLAND_ONE_SIDE:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    V_ls = _ls_rows(ls, ISLAND_ONE_SIDE)[:, buses]

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, handle_disconnected_grid=True, **GPU_TOL)
    ca.add_contingencies(ISLAND_ONE_SIDE)
    ca.compute(batch_size=4)
    V = ca.V_results.to_numpy().reshape(len(ISLAND_ONE_SIDE), -1)
    assert np.all(np.isfinite(ca.last_residuals()))
    for r, bus in enumerate(ISLAND_ONE_SIDE_BUS):
        nan = ~np.isfinite(V[r])
        expected = np.zeros(len(buses), dtype=bool)
        expected[_solver_pos(buses, bus)] = True
        np.testing.assert_array_equal(nan, expected, err_msg=f"row {r}")
        np.testing.assert_allclose(V[r, ~nan], V_ls[r, ~nan], atol=10 * solver_atol, err_msg=f"row {r}")


# ----------------------------------------------- 5. current limits, opened at one end
def test_ca_current_limits_skip_branch_opened_at_one_end():
    """compute_limit_violations with a 1e-6 kA current limit on both sides of every
    branch (so every closed, loaded branch end violates): a branch opened at one end
    (line 3 at side 1, line 5 at side 2, transformer 15 at side 1) and a tripped branch
    (line 7) have no CURRENT record at either side in their row, every other branch has
    both, and the set of CURRENT records (type, id, side) equals lightsim2grid's
    ContingencyAnalysisCPP.get_violations() on the same contingencies and limits."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.contingency_analysis import LimitViolationType, ViolationElementType
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    grid, v_ref = _make_grid()
    n_trafo = len(grid.get_trafos())
    tiny = 1e-6
    grid.set_line_current_limit_side1(np.full(N_LINE, tiny))
    grid.set_line_current_limit_side2(np.full(N_LINE, tiny))
    grid.set_trafo_current_limit_side1(np.full(n_trafo, tiny))
    grid.set_trafo_current_limit_side2(np.full(n_trafo, tiny))
    trips = [((), (), (), (), (3,), ()), ((), (), (), (), (), (5,)),
             ((), (), (), (), (15,), ()), ((7,), (), (), (), (), ())]
    out_of_service = [3, 5, 15, 7]

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_limit_violations=True, **GPU_TOL)
    ca._inner.violation_capacity = 2 * (N_LINE + n_trafo)   # not a facade property
    ca.set_limits_from_grid()
    ca.add_contingencies(trips)
    ca.compute(batch_size=4)
    assert not np.any(ca.get_violations_truncated())

    def key(etype, eid):
        return eid if int(etype) == int(ViolationElementType.LINE) else N_LINE + eid

    ours = []
    for r, viol in enumerate(ca.get_violations()):
        cur = {(key(v.element_type, v.element_id), int(v.side)) for v in viol
               if v.violation_type == LimitViolationType.CURRENT}
        expected = {(b, s) for b in range(N_LINE + n_trafo) if b != out_of_service[r] for s in (1, 2)}
        assert cur == expected, f"row {r}: {sorted(cur ^ expected)}"
        ours.append(cur)

    ls = ContingencyAnalysisCPP(grid)
    ls.compute_limit_violations = True
    for t in trips:
        ls.add_contingency(*[list(el) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    conts = ls.my_contingencies()
    n = len(conts[0])
    row = {tuple(tuple(el) for el in c): k for k, c in enumerate(conts)}
    ls_viol = ls.get_violations()
    for r, t in enumerate(trips):
        k = row[tuple(tuple(sorted(el)) for el in (tuple(t) + ((),) * n)[:n])]
        cur = {(key(v.element_type, v.element_id), int(v.side)) for v in ls_viol[k]
               if int(v.violation_type) == int(LimitViolationType.CURRENT)}
        assert cur == ours[r], f"row {r}: {sorted(cur ^ ours[r])}"


# ---------------------------------------------------- 6. every kind, repeated, one batch
MIXED = [((3,), (0,), (2,), (1,), (), ()),
         ((), (0,), (4, 2), (), (5,), ()),
         ((7,), (), (), (2,), (), (3,)),
         ((), (), (1,), (), (15,), (6,)),
         ((), (0,), (), (2,), (), ()),
         ((5,), (), (3, 3), (), (), (16,))]       # a load named twice: tripped once
MIXED_APPENDED = [((3,), (0,), (2,), (1,), (), ()),                # row 0 again
                  ((), (), (), (1,), (4,), ()),
                  ((), (), (0, 10), (), (), ())]


def test_ca_mixed_repeated_trip_kinds_match_lightsim2grid(solver_atol):
    """One ContingencyAnalysisGPU batch mixing branch trips, shunts, loads, generators
    and branches opened at side 1 / side 2, given by two add_contingencies calls (the
    second repeating a row of the first): get_contingencies() keeps all 9 rows (the load
    named twice once), every residual is below 100 * solver_atol, every row equals
    lightsim2grid's ContingencyAnalysisCPP row of the same contingency and a one-off
    ac_pf with the elements deactivated, and the repeated row equals its first copy."""
    from gpusim2grid import ContingencyAnalysisGPU
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    grid, v_ref = _make_grid()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    trips = MIXED + MIXED_APPENDED

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, **GPU_TOL)
    ca.add_contingencies(MIXED)
    ca.add_contingencies(MIXED_APPENDED)
    assert ca.get_contingencies() == [tuple(sorted(set(el)) for el in t) for t in trips]
    ca.compute(batch_size=4)
    assert np.all(ca.last_residuals() < 100 * solver_atol)
    V = ca.V_results.to_numpy().reshape(len(trips), -1)

    ls = ContingencyAnalysisCPP(grid)
    for t in trips:
        ls.add_contingency(*[sorted(set(el)) for el in t])
    ls.compute(1.0 * v_ref, MAX_IT, TOL)
    assert all(ls.converged_mask())
    V_ls = _ls_rows(ls, [tuple(sorted(set(el)) for el in t) for t in trips])[:, buses]
    for r, t in enumerate(trips):
        np.testing.assert_allclose(V[r], V_ls[r], atol=10 * solver_atol, err_msg=f"trip {t}")
        np.testing.assert_allclose(V[r], _reference_full(t)[buses], atol=10 * solver_atol, err_msg=f"trip {t}")
    np.testing.assert_allclose(V[len(MIXED)], V[0], atol=solver_atol)
