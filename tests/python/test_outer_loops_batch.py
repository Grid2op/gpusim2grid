# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The outer loops on the GPU (``ScenarioSweepGPU(outer_loops=True)``,
docs/dev_notes/continuous_batching.md Part B) against lightsim2grid's own
``NROuter_*`` batch, row by row.

lightsim2grid's batch solves each row with the grid's loops (DistributedSlack,
AcHvdcAcEmulationLimits, VoltageMonitoring, ReactiveLimits): its inner Newton,
then the loops' checks, a re-solve whenever one acts. gpusim2grid runs the
inner solves on the device and the same checks on the host, so a row ends at
the same voltages, after the same number of outer iterations, with the same
targets moved -- when both start from the same voltages (the loops stop on
thresholds, so where they stop depends on where the solve starts): here the
grid's own, its generator buses at their set-points, what lightsim2grid's batch
seeds from the ``v_init`` it is given.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                       reason="gpusim2grid built against a lightsim2grid without NROuter_*"),
    pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity with lightsim2grid's double-precision NR"),
]

OUTER = "NROuter_KLU"
MAX_IT = 30
TOL_MVA = 1e-8          # lightsim2grid's tol is in MVA: ||F||inf < tol / sn_mva
N_ROWS = 8
CONVERGED, MAX_ITER, DIVERGED, NOT_SIMULATED = 0, 1, 2, 3
OUTER_FAILED, OUTER_UNSTABLE, UNREALISTIC = 4, 5, 6


def _case14(q_limit=15., distributed=True):
    """case14, its generators short of reactive power (ReactiveLimits switches
    buses PQ) and sharing the slack in proportion to their maximum output
    (DistributedSlack re-shares whatever a row changes in the balance) --
    lightsim2grid's own test grid (tests/test_outer_batch.py)."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    net = pn.case14()
    net.gen["min_q_mvar"] = -q_limit
    net.gen["max_q_mvar"] = q_limit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 300.))
    if distributed:
        grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
    grid.change_algorithm(OUTER)
    V = grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    assert V.size > 0
    return grid


def _rows(grid, n_rows=N_ROWS, trips=True):
    """load increase per row (+5 % each), line i + 1 tripped on row i"""
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    scale = 1. + 0.05 * np.arange(n_rows)
    rows = {
        "gen_p": np.outer(np.ones(n_rows), gen_p),
        "load_p": np.outer(scale, load_p),
        "load_q": np.outer(np.ones(n_rows), load_q),
        "lines": [[i + 1] for i in range(n_rows)] if trips else [[] for _ in range(n_rows)],
    }
    return rows


def _lightsim2grid(grid, rows):
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    n_rows = rows["load_p"].shape[0]
    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    mask = np.zeros((n_rows, len(grid.get_lines())), dtype=bool)
    for i, lines in enumerate(rows["lines"]):
        mask[i, lines] = True
    batch.set_contingency_lines(mask)
    batch.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL_MVA)
    return batch


def _gpu(grid, rows, batch_size, tol_mva=TOL_MVA, init_from_n_powerflow=False, **kwargs):
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n_powerflow,
                          tol=tol_mva / grid.get_sn_mva(), **kwargs)
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.set_topology(rows["lines"])
    sw.compute(batch_size=batch_size)
    return sw


def _V(sw, grid, n_rows):
    n_bus = grid.get_Ybus_solver().shape[0]
    return sw.V_results.to_numpy().reshape(n_rows, n_bus)


@pytest.fixture(scope="module")
def case14():
    return _case14()


@pytest.mark.parametrize("batch_size", [1, 3, N_ROWS])
def test_rows_match_lightsim2grid_batch(case14, batch_size):
    grid = case14
    rows = _rows(grid)
    ls = _lightsim2grid(grid, rows)
    sw = _gpu(grid, rows, batch_size, keep_outer_trace=True)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)

    V_ls = np.asarray(ls.get_voltages())[:, buses]
    V = _V(sw, grid, N_ROWS)
    conv_ls = np.asarray(ls.converged_mask(), dtype=bool)
    status = np.asarray(sw.get_row_status())
    res_ls = ls.get_outer_results()
    res = sw.get_outer_results()
    for i in range(N_ROWS):
        msg = f"row {i}: gpusim2grid {res[i].decisions} / lightsim2grid status {res_ls[i].status}"
        assert (status[i] == CONVERGED) == bool(conv_ls[i]), msg
        assert int(res[i].status) == int(res_ls[i].status), msg
        assert res[i].nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        if conv_ls[i]:
            np.testing.assert_allclose(V[i], V_ls[i], rtol=0., atol=1e-8, err_msg=msg)
            moved_ls = dict(res_ls[i].gen_target_p)
            moved = dict(res[i].gen_target_p)
            assert moved.keys() == moved_ls.keys(), msg
            for g in moved:
                assert moved[g] == pytest.approx(moved_ls[g], abs=1e-6), msg


def test_a_row_the_loops_act_on_is_resolved(case14):
    """The base grid itself as one row: its loops act (case14's PV buses are
    short of reactive power), and the row ends where lightsim2grid's single
    NROuter solve ends."""
    grid = case14
    rows = _rows(grid, n_rows=1, trips=False)
    rows["load_p"] = rows["load_p"] / 1.   # the grid's own loads
    sw = _gpu(grid, rows, 1, keep_outer_trace=True)
    res = sw.get_outer_results()[0]
    assert res.nb_outer_iterations > 0, res.decisions
    assert sw.get_row_status()[0] == CONVERGED
    ls = _lightsim2grid(grid, rows)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    np.testing.assert_allclose(_V(sw, grid, 1)[0], np.asarray(ls.get_voltages())[0, buses],
                               rtol=0., atol=1e-8)


def test_ample_reactive_limits_only_the_slack_is_shared():
    """Ample reactive limits: ReactiveLimits never acts; the slack's own unit is the
    only participant, so DistributedSlack does not run (as lightsim2grid)."""
    grid = _case14(q_limit=1e4, distributed=False)
    rows = _rows(grid)
    sw = _gpu(grid, rows, 4, keep_outer_trace=True)
    res = sw.get_outer_results()
    assert not any("ReactiveLimits" in d and "not taken" not in d for r in res for d in r.decisions)
    ls = _lightsim2grid(grid, rows)
    _compare_with_batch(_V(sw, grid, N_ROWS), np.asarray(sw.get_row_status()), res, ls, grid, N_ROWS)


def test_refusals(case14):
    from gpusim2grid import ScenarioSweepGPU
    # by design, not planned: at construction and at a run, both being mutable
    with pytest.raises(ValueError, match="continuous.*not planned"):
        ScenarioSweepGPU(case14, outer_loops=True, scheduling="chunked")
    with pytest.raises(ValueError, match="nb_iter_per_round=1.*not planned"):
        ScenarioSweepGPU(case14, outer_loops=True, nb_iter_per_round=2)
    for name, value, match in (("scheduling", "chunked", "continuous.*not planned"),
                               ("nb_iter_per_round", 2, "nb_iter_per_round=1.*not planned")):
        sw = ScenarioSweepGPU(case14, outer_loops=True, init_from_n_powerflow=False)
        rows = _rows(case14, n_rows=2)
        sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
        setattr(sw, name, value)
        with pytest.raises(ValueError, match=match):
            sw.compute(batch_size=2)
    with pytest.raises(ValueError, match="redistribute_slack"):
        ScenarioSweepGPU(case14, outer_loops=True, redistribute_slack=True)
    sw = ScenarioSweepGPU(case14, outer_loops=True, init_from_n_powerflow=False)
    rows = _rows(case14, n_rows=2)
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.strategy = "direct_base_case_factors"
    with pytest.raises(RuntimeError, match="direct_refactor_every"):
        sw.compute(batch_size=2)
    with pytest.raises(RuntimeError, match="outer_loops"):
        ScenarioSweepGPU(case14, init_from_n_powerflow=False).get_outer_results()


def test_grid_without_nrouter_is_refused():
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from gpusim2grid import ScenarioSweepGPU
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    with pytest.raises(RuntimeError, match="NROuter"):
        ScenarioSweepGPU(grid, outer_loops=True)


# --------------------------------------------------------------------------------------
# Every loop and controller configuration of test_outer_model_cpu.py (which checks the
# host port alone against lightsim2grid), now with the inner solves on the GPU: each row
# against lightsim2grid's single NROuter solve of the row's grid, from the same start.
# --------------------------------------------------------------------------------------
def _rows_vs_single_solves(make_grid, scales, tol_mva, batch_size, start=None):
    base = make_grid()
    v0 = np.full(base.total_bus(), 1.0 + 0j) if start is None else start(base)
    assert base.ac_pf(v0, MAX_IT, tol_mva).size > 0
    load_p0 = np.asarray(base.get_load_target_p(), dtype=float)
    load_q0 = np.array([l.target_q_mvar for l in base.get_loads()], dtype=float)
    gen_p0 = np.array([g.target_p_mw for g in base.get_generators()], dtype=float)
    n = len(scales)
    rows = {"gen_p": np.outer(np.ones(n), gen_p0), "load_p": np.outer(scales, load_p0),
            "load_q": np.outer(np.ones(n), load_q0), "lines": [[] for _ in range(n)]}
    sw = _gpu(base, rows, batch_size, tol_mva=tol_mva, keep_outer_trace=True)
    V = _V(sw, base, n)
    status = np.asarray(sw.get_row_status())
    res = sw.get_outer_results()
    for i, s in enumerate(scales):
        row = make_grid()
        for load_id, p in enumerate(load_p0):
            row.change_p_load(load_id, float(p * s))
        V_ref = row.ac_pf(np.asarray(base.get_V(), dtype=complex), MAX_IT, tol_mva)
        stats = row.get_algo().get_outer_loop_stats()
        msg = (f"row {i}: " + "\n".join(res[i].decisions) + "\n---- lightsim2grid\n" +
               "\n".join(str(d) for d in stats.decisions))
        assert res[i].nb_outer_iterations == stats.nb_outer_iterations, msg
        assert int(res[i].status) == int(stats.status), msg
        assert (status[i] == CONVERGED) == (V_ref.size > 0), msg
        if V_ref.size > 0:
            buses = np.asarray(row.id_ac_solver_to_me(), dtype=int)
            np.testing.assert_allclose(V[i], V_ref[buses], rtol=0., atol=1e-8, err_msg=msg)
    return sw


@pytest.mark.parametrize("kwargs", [dict(), dict(p0=-60., pmax12=300., pmax21=20.), dict(pmax12=300.)])
def test_gpu_hvdc_ac_emulation_limits(kwargs):
    from test_outer_model_cpu import _case14_hvdc
    _rows_vs_single_solves(lambda: _case14_hvdc(**kwargs), [1., 1.05, 1.1], TOL_MVA, 2)


@pytest.mark.parametrize("low_pu, high_pu", [(0.90, 1.00), (1.10, 1.20), (0.90, 1.20)])
def test_gpu_voltage_monitoring(low_pu, high_pu):
    from test_outer_model_cpu import _ieee14_standby_svc
    _rows_vs_single_solves(lambda: _ieee14_standby_svc(low_pu, high_pu), [1., 1.05], 1e-10, 2)


@pytest.mark.parametrize("regulators, qmin, qmax", [
    (("B6-G", "B8-G"), (-10., -10.), (10., 10.)),
    (("B6-G", "B8-G"), (-100., -3.), (3., 100.)),
    (("B3-G", "B6-G", "B8-G"), (-100., -100., -3.), (100., 3., 100.)),
    (("B3-G", "B6-G", "B8-G"), (-100., -3., -3.), (3., 100., 100.)),
])
def test_gpu_reactive_limits_on_a_voltage_control_group(regulators, qmin, qmax):
    from test_outer_model_cpu import _ieee14_remote_group
    _rows_vs_single_solves(lambda: _ieee14_remote_group(qmax, regulators, qmin=qmin),
                           [1., 1.05, 1.1], 1e-10, 3)


def test_gpu_ieee300_reactive_limits():
    from test_outer_model_cpu import _ieee300

    def dc_start(grid):
        flat = np.full(grid.total_bus(), 1.0 + 0j)
        return np.exp(1j * np.angle(grid.dc_pf(flat, 30, 1e-12)))

    _rows_vs_single_solves(_ieee300, [1., 1.005, 1.01], 1e-9, 2, start=dc_start)


# --------------------------------------------------------------------------------------
# The two other facades, and init_from_n_powerflow (the rows resume from the "n" solve)
# --------------------------------------------------------------------------------------
def _v_batch(grid, init_from_n):
    """what lightsim2grid's batch is given: rows that do not resume start from it
    (gpusim2grid's start: the grid's V); rows that do resume from the batch's own "n"
    solve, which must land where the grid's did -- so from the same start (flat 1.04)"""
    if init_from_n:
        return np.full(grid.total_bus(), 1.04 + 0j)
    return np.asarray(grid.get_V(), dtype=complex)


def _compare_with_batch(V, status, res, batch, grid, n_rows):
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res_ls = batch.get_outer_results()
    for i in range(n_rows):
        msg = f"row {i}: " + "\n".join(res[i].decisions)
        assert int(res[i].status) == int(res_ls[i].status), msg
        assert res[i].nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        assert (status[i] == CONVERGED) == bool(conv[i]), msg
        if conv[i]:
            np.testing.assert_allclose(V[i], V_ls[i], rtol=0., atol=1e-8, err_msg=msg)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_contingency_analysis_matches_lightsim2grid(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _case14()
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    for line_id in range(N_ROWS):
        batch.add_n1(line_id)
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                                init_from_n_powerflow=init_from_n, tol=TOL_MVA / grid.get_sn_mva())
    ca.add_contingencies_by_branch_id([[i] for i in range(N_ROWS)])
    ca.compute(batch_size=3)
    n_bus = grid.get_Ybus_solver().shape[0]
    _compare_with_batch(ca.V_results.to_numpy().reshape(N_ROWS, n_bus), np.asarray(ca.get_row_status()),
                        ca.get_outer_results(), batch, grid, N_ROWS)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_injection_sweep_matches_lightsim2grid(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _case14()
    rows = _rows(grid, trips=False)
    batch = InjectionSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    sw = InjectionSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                           init_from_n_powerflow=init_from_n, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.compute(batch_size=3)
    n_bus = grid.get_Ybus_solver().shape[0]
    _compare_with_batch(sw.V_results.to_numpy().reshape(N_ROWS, n_bus), np.asarray(sw.get_row_status()),
                        sw.get_outer_results(), batch, grid, N_ROWS)


def test_scenario_sweep_resumes_from_the_n_solve(case14):
    if not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = case14
    rows = _rows(grid)
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = True
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    mask = np.zeros((N_ROWS, len(grid.get_lines())), dtype=bool)
    for i, lines in enumerate(rows["lines"]):
        mask[i, lines] = True
    batch.set_contingency_lines(mask)
    batch.compute(_v_batch(grid, True), MAX_IT, TOL_MVA)
    sw = _gpu(grid, rows, 3, init_from_n_powerflow=True, keep_outer_trace=True)
    _compare_with_batch(_V(sw, grid, N_ROWS), np.asarray(sw.get_row_status()),
                        sw.get_outer_results(), batch, grid, N_ROWS)


# --------------------------------------------------------------------------------------
# The rows' other inputs: islanded buses, generator contingencies, gen_v
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("init_from_n", [False, True])
def test_contingency_analysis_islanding_rows(init_from_n):
    """Every branch of case14 tripped in turn, handle_disconnected_grid: tripping the
    transformer to bus 8 (a generator's) islands it -- solved on the main component,
    its loops not seeing the masked bus -- as lightsim2grid's batch does"""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _case14()
    n_br = len(grid.get_lines()) + len(grid.get_trafos())
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.handle_disconnected_grid = True
    batch.add_all_n1()
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                                handle_disconnected_grid=True, init_from_n_powerflow=init_from_n,
                                tol=TOL_MVA / grid.get_sn_mva())
    ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
    ca.compute(batch_size=4)
    n_bus = grid.get_Ybus_solver().shape[0]
    V = ca.V_results.to_numpy().reshape(n_br, n_bus)
    assert np.isnan(V).any()          # some row masked buses
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    status = np.asarray(ca.get_row_status())
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res, res_ls = ca.get_outer_results(), batch.get_outer_results()
    for i in range(n_br):
        msg = f"branch {i}: " + "\n".join(res[i].decisions)
        assert (status[i] == CONVERGED) == bool(conv[i]), msg
        assert res[i].nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        if conv[i]:
            fin = np.isfinite(V[i])
            np.testing.assert_allclose(V[i][fin], V_ls[i][fin], rtol=0., atol=1e-8, err_msg=msg)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_scenario_sweep_generator_contingencies(init_from_n):
    """Row i disconnects one generator (its bus loses its regulator: PQ on that row) and
    loads +3 %/row; DistributedSlack shares without it"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _case14()
    n_gen = len(grid.get_generators())
    off_rows = list(range(1, n_gen))            # not the slack's
    n_rows = len(off_rows)
    rows = _rows(grid, n_rows=n_rows, trips=False)
    mask = np.zeros((n_rows, n_gen), dtype=bool)
    for r, g in enumerate(off_rows):
        mask[r, g] = True
    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    batch.set_contingency_gens(mask)
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                          init_from_n_powerflow=init_from_n, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=2)
    _compare_with_batch(_V(sw, grid, n_rows), np.asarray(sw.get_row_status()),
                        sw.get_outer_results(), batch, grid, n_rows)


def _gen_v_rows(grid, n_rows, gens, delta=0.01):
    gen_v = np.full((n_rows, len(grid.get_generators())), np.nan)
    for r in range(n_rows):
        for g in gens:
            gen_v[r, g] = grid.get_generators()[g].target_vm_pu + delta * (r - n_rows // 2)
    return gen_v


@pytest.mark.parametrize("init_from_n", [False, True])
def test_scenario_sweep_gen_v(init_from_n):
    """local regulators' set-points varied per row: PV buses re-seeded, the frozen ones
    released against the row's own set-point"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _case14()
    rows = _rows(grid, n_rows=5, trips=False)
    gen_v = _gen_v_rows(grid, 5, gens=[1, 2, 3])
    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    batch.modify_gen_v(np.where(np.isnan(gen_v), [[g.target_vm_pu for g in grid.get_generators()]], gen_v))
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                          init_from_n_powerflow=init_from_n, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=2)
    _compare_with_batch(_V(sw, grid, 5), np.asarray(sw.get_row_status()), sw.get_outer_results(), batch, grid, 5)


def test_scenario_sweep_gen_v_on_a_remote_group():
    """the set-point of a voltage-control group (two remote regulators of bus 9) varied per
    row, one of its controllers short of reactive power"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from test_outer_model_cpu import _ieee14_remote_group
    grid = _ieee14_remote_group((3., 100.), ("B6-G", "B8-G"), qmin=(-100., -3.))
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    names = [g.name for g in grid.get_generators()]
    regs = [names.index("B6-G"), names.index("B8-G")]
    n_rows = 5
    gen_p0 = np.array([g.target_p_mw for g in grid.get_generators()])
    load_p0 = np.asarray(grid.get_load_target_p())
    load_q0 = np.array([l.target_q_mvar for l in grid.get_loads()])
    gen_p = np.outer(np.ones(n_rows), gen_p0)
    load_p = np.outer(1. + 0.02 * np.arange(n_rows), load_p0)
    load_q = np.outer(np.ones(n_rows), load_q0)
    gen_v = _gen_v_rows(grid, n_rows, regs, delta=0.005)
    full_v = np.where(np.isnan(gen_v), [[g.target_vm_pu for g in grid.get_generators()]], gen_v)
    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.modify_gen_p(gen_p)
    batch.modify_load_p(load_p)
    batch.modify_load_q(load_q)
    batch.modify_gen_v(full_v)
    batch.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                          init_from_n_powerflow=False, tol=1e-10 / grid.get_sn_mva())
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=2)
    _compare_with_batch(_V(sw, grid, n_rows), np.asarray(sw.get_row_status()), sw.get_outer_results(),
                        batch, grid, n_rows)


def test_injection_sweep_gen_v():
    from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid = _case14()
    rows = _rows(grid, n_rows=5, trips=False)
    gen_v = _gen_v_rows(grid, 5, gens=[1, 2, 3])
    batch = InjectionSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    batch.modify_gen_v(np.where(np.isnan(gen_v), [[g.target_vm_pu for g in grid.get_generators()]], gen_v))
    batch.compute(_v_batch(grid, False), MAX_IT, TOL_MVA)
    sw = InjectionSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                           init_from_n_powerflow=False, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=2)
    _compare_with_batch(sw.V_results.to_numpy().reshape(5, -1), np.asarray(sw.get_row_status()),
                        sw.get_outer_results(), batch, grid, 5)


# --------------------------------------------------------------------------------------
# PhaseControl: a phase shifter's shift a Newton unknown, its pi block patched by value in
# each slot's Ybus at every Newton step (outer/branch_control_kernels.cuh); the three
# facades against lightsim2grid's own batches
# --------------------------------------------------------------------------------------
def _pst(mode):
    """lightsim2grid's PhaseControl test grid, solved, and the shifter's id: regulating 7 MW
    above its uncontrolled flow, or limiting its current to 80 % of it"""
    import test_outer_model_cpu as cpu
    p1, i1 = cpu._pst_flow()
    target = p1 + 7. if mode == "ACTIVE_POWER_CONTROL" else 0.8 * i1
    grid = cpu._pst_grid(mode, target)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, 1e-10).size > 0
    return grid, cpu._pst_id(grid)


def _check_rows(sw, ls, grid, n_rows, flows=True):
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = _V(sw, grid, n_rows)
    V_ls = np.asarray(ls.get_voltages())[:, buses]
    conv_ls = np.asarray(ls.converged_mask(), dtype=bool)
    status = np.asarray(sw.get_row_status())
    res, res_ls = sw.get_outer_results(), ls.get_outer_results()
    for i in range(n_rows):
        msg = f"row {i}: {res[i].decisions}"
        assert (status[i] == CONVERGED) == bool(conv_ls[i]), msg
        assert int(res[i].status) == int(res_ls[i].status), msg
        assert res[i].nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        assert sorted(res[i].phase_tap) == sorted(res_ls[i].phase_tap), msg
        assert sorted(res[i].ratio_tap) == sorted(res_ls[i].ratio_tap), msg
        assert sorted(res[i].shunt_sections) == sorted(res_ls[i].shunt_sections), msg
        if conv_ls[i]:
            np.testing.assert_allclose(V[i], V_ls[i], rtol=0., atol=1e-8, err_msg=msg)
    if flows:
        # the shifter's current at the taps each row settled on
        sw.compute_flows()
        ls.compute_flows()
        amps = np.asarray(sw.or_amps.to_numpy() if hasattr(sw.or_amps, "to_numpy") else sw.or_amps).reshape(n_rows, -1)
        amps_ls = np.asarray(ls.get_flows())
        ok = conv_ls
        np.testing.assert_allclose(amps[ok], amps_ls[ok] * _flow_unit(amps_ls, amps), rtol=1e-7, atol=1e-6)
    return res


def _flow_unit(amps_ls, amps):
    """lightsim2grid's flows in kA, gpusim2grid's in A"""
    return 1000.


@pytest.mark.parametrize("mode", ["ACTIVE_POWER_CONTROL", "CURRENT_LIMITER"])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_phase_control_scenario_sweep(mode, init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid, pst = _pst(mode)
    n_rows, n_line = 5, len(grid.get_lines())
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.03 * np.arange(n_rows)
    rows = [[], [2], [3], [n_line + pst], []]       # the 4th row opens the shifter
    ls = ScenarioSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    lines = np.zeros((n_rows, n_line), dtype=bool)
    trafos = np.zeros((n_rows, len(grid.get_trafos())), dtype=bool)
    for i, br in enumerate(rows):
        for b in br:
            if b < n_line:
                lines[i, b] = True
            else:
                trafos[i, b - n_line] = True
    ls.set_contingency_lines(lines)
    ls.set_contingency_trafos(trafos)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                          tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    sw.set_topology(rows)
    sw.compute(batch_size=2)
    res = _check_rows(sw, ls, grid, n_rows)
    assert any(r.phase_tap for r in res)
    assert not [t for t, _ in res[3].phase_tap if t == pst]


@pytest.mark.parametrize("mode", ["ACTIVE_POWER_CONTROL", "CURRENT_LIMITER"])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_phase_control_contingency_analysis(mode, init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid, pst = _pst(mode)
    n_line = len(grid.get_lines())
    branches = [0, 2, 3, 5, n_line + pst]           # sorted: lightsim2grid's row order
    ls = ContingencyAnalysisCPP(grid)
    for b in branches:
        ls.add_n1(b)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                                tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    ca.add_contingencies_by_branch_id([[b] for b in branches])
    ca.compute(batch_size=2)
    _check_rows(ca, ls, grid, len(branches))


@pytest.mark.parametrize("mode", ["ACTIVE_POWER_CONTROL", "CURRENT_LIMITER"])
def test_phase_control_injection_sweep(mode):
    from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid, _ = _pst(mode)
    n_rows = 6
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.04 * np.arange(n_rows)
    ls = InjectionSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = InjectionSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                           tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    # fewer slots than rows: a slot's Ybus (an injection sweep never reloads it) carries the
    # next row only once the previous one's taps were undone
    sw.compute(batch_size=2)
    _check_rows(sw, ls, grid, n_rows)


def test_phase_control_current_violations_at_the_moved_tap():
    """The current check (compute_limit_violations) reads a moved shifter's block at the
    taps its row settled on, as lightsim2grid's (its row trafo blocks)"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    grid, pst = _pst("ACTIVE_POWER_CONTROL")
    lim = np.full(len(grid.get_trafos()), np.nan)
    lim[pst] = 0.5 * grid.get_trafos()[pst].res_a1_ka   # violated on every row
    grid.set_trafo_current_limit_side1(lim)
    n_rows = 4
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.03 * np.arange(n_rows)
    ls = ScenarioSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.compute_limit_violations = True
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                          tol=1e-10 / grid.get_sn_mva(), compute_limit_violations=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    sw.compute(batch_size=2)
    res = sw.get_outer_results()
    assert all(r.phase_tap for r in res)      # every row moved it
    for i, (v, v_ls) in enumerate(zip(sw.get_violations(), ls.get_violations())):
        pick = lambda vs: [x.value for x in vs if int(x.element_id) == pst and x.element_type.name == "TRAFO"
                           and x.violation_type.name == "CURRENT"]
        got, ref = pick(v), pick(v_ls)
        assert len(got) == len(ref) == 1, (i, v, v_ls)
        assert got[0] == pytest.approx(ref[0], rel=1e-8)



# --------------------------------------------------------------------------------------
# TransformerVoltageControl: a ratio tap changer's ratio a Newton unknown (BranchControl's
# ratio column and its group's rows), the low voltage generators frozen meanwhile;
# lightsim2grid's _tap_grid (ReactiveLimits too)
# --------------------------------------------------------------------------------------
def _tvc():
    import test_outer_model_cpu as cpu
    grid = cpu._tvc_grid("T4-9-1", 12.6, 0.1, reactive_limits=True, use_initial_tap_position=True,
                         max_controlled_nominal_voltage=120.)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, 1e-10).size > 0
    t = next(i for i, tr in enumerate(grid.get_trafos()) if tr.name == "T4-9-1")
    return grid, t


@pytest.mark.parametrize("init_from_n", [False, True])
def test_transformer_voltage_control_contingency_analysis(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid, trafo = _tvc()
    n_line = len(grid.get_lines())
    branches = [1, 2, 3, 5, n_line + trafo]          # sorted: lightsim2grid's row order
    ls = ContingencyAnalysisCPP(grid)
    for b in branches:
        ls.add_n1(b)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                                tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    ca.add_contingencies_by_branch_id([[b] for b in branches])
    ca.compute(batch_size=2)
    res = _check_rows(ca, ls, grid, len(branches))
    assert any(r.ratio_tap for r in res)
    assert not [t for t, _ in res[-1].ratio_tap if t == trafo]


@pytest.mark.parametrize("init_from_n", [False, True])
def test_transformer_voltage_control_scenario_sweep(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid, _ = _tvc()
    n_rows, n_line = 5, len(grid.get_lines())
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.02 * np.arange(n_rows)
    rows = [[], [1], [2], [], [3]]
    ls = ScenarioSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    lines = np.zeros((n_rows, n_line), dtype=bool)
    for i, br in enumerate(rows):
        lines[i, br] = True
    ls.set_contingency_lines(lines)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                          tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    sw.set_topology(rows)
    sw.compute(batch_size=2)
    _check_rows(sw, ls, grid, n_rows)


def test_transformer_voltage_control_injection_sweep():
    from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid, _ = _tvc()
    n_rows = 6
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.03 * np.arange(n_rows)
    ls = InjectionSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = InjectionSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                           tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    sw.compute(batch_size=2)
    res = _check_rows(sw, ls, grid, n_rows)
    assert any(r.ratio_tap for r in res)



# --------------------------------------------------------------------------------------
# ShuntVoltageControl: a shunt controller's susceptance a Newton unknown (ShuntControl), its
# bus' Ybus diagonal patched by value, then dispatched to sections
# --------------------------------------------------------------------------------------
def _shunts(bus="B9", target_kv=13.2, sections=(3, 1)):
    import test_outer_model_cpu as cpu
    grid = cpu._shunt_grid(bus, target_kv, sections)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, 1e-10).size > 0
    return grid


@pytest.mark.parametrize("init_from_n", [False, True])
def test_shunt_voltage_control_contingency_analysis(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _shunts()
    branches = [0, 1, 2, 3, 5]
    ls = ContingencyAnalysisCPP(grid)
    for b in branches:
        ls.add_n1(b)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                                tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    ca.add_contingencies_by_branch_id([[b] for b in branches])
    ca.compute(batch_size=2)
    res = _check_rows(ca, ls, grid, len(branches), flows=False)
    if not init_from_n:
        assert any(r.shunt_sections for r in res)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_shunt_voltage_control_scenario_sweep(init_from_n):
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _shunts()
    n_rows, n_line = 4, len(grid.get_lines())
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.03 * np.arange(n_rows)
    rows = [[], [1], [], [2]]
    ls = ScenarioSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.init_from_n_powerflow = init_from_n
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    lines = np.zeros((n_rows, n_line), dtype=bool)
    for i, br in enumerate(rows):
        lines[i, br] = True
    ls.set_contingency_lines(lines)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                          tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    sw.set_topology(rows)
    sw.compute(batch_size=2)
    _check_rows(sw, ls, grid, n_rows, flows=False)


def test_shunt_voltage_control_injection_sweep():
    from lightsim2grid.lightsim2grid_cpp import InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid = _shunts()
    n_rows = 6
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    scale = 1. + 0.03 * np.arange(n_rows)
    ls = InjectionSweepCPP(grid)
    ls.change_algorithm(OUTER)
    ls.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    ls.modify_load_p(np.outer(scale, load_p))
    ls.modify_load_q(np.outer(np.ones(n_rows), load_q))
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    sw = InjectionSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                           tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    sw.set_injections_from_elements(np.outer(scale, load_p), np.outer(np.ones(n_rows), load_q),
                                    np.outer(np.ones(n_rows), gen_p))
    # fewer slots than rows: a slot's Ybus diagonal back to the grid's between two rows
    sw.compute(batch_size=2)
    res = _check_rows(sw, ls, grid, n_rows, flows=False)
    assert any(r.shunt_sections for r in res)


def test_all_seven_loops_together():
    """the four default loops, PhaseControl, TransformerVoltageControl and ShuntVoltageControl
    on one grid (a phase shifter, a ratio tap changer and two regulating shunts), OpenLoadFlow's
    order, against lightsim2grid's contingency analysis"""
    import sys, os
    import lightsim2grid
    sys.path.insert(0, os.path.join(os.path.dirname(lightsim2grid.__file__), "tests"))
    import pandas as pd
    import test_shunt_voltage_control as S
    import test_phase_control as P
    from lightsim2grid.algorithm import PhaseControl, TransformerVoltageControl, ShuntVoltageControl
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    net = S._net("B9", 13.2, (3, 1), with_trafo_control=False)
    ptc = pd.DataFrame.from_records(
        index="id", columns=["id", "target_deadband", "regulation_mode", "low_tap", "tap", "regulating", "regulated_side"],
        data=[(P.PST, 0., "ACTIVE_POWER_CONTROL", -10, 0, False, "ONE")])
    steps = pd.DataFrame.from_records(index="id", columns=["id", "b", "g", "r", "x", "rho", "alpha"],
                                      data=[(P.PST, 0., 0., 0., 0.5 * a, 1., float(a)) for a in range(-10, 11)])
    net.create_phase_tap_changers(ptc, steps)
    net.update_phase_tap_changers(id=P.PST, regulation_value=45., regulating=True)
    rtc = pd.DataFrame.from_records(
        index="id", columns=["id", "target_deadband", "target_v", "oltc", "low_tap", "tap", "regulating", "regulated_side"],
        data=[("T4-7-1", 0.1, 12.0, True, 0, 10, False, "TWO")])
    rsteps = pd.DataFrame.from_records(index="id", columns=["id", "b", "g", "r", "x", "rho"],
                                       data=[("T4-7-1", 0., 0., 0., 0., 0.9 + 0.01 * k) for k in range(21)])
    net.create_ratio_tap_changers(rtc, rsteps)
    net.update_ratio_tap_changers(id="T4-7-1", regulating=True)
    grid = S._grid(net)
    for loop in (PhaseControl(), TransformerVoltageControl(), ShuntVoltageControl()):
        grid.add_outer_loop(loop)
    grid.change_algorithm(OUTER)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, 1e-10).size > 0
    names = [l.name() for l in grid.get_outer_loops()]
    assert names[-3:] == ["PhaseControl", "TransformerVoltageControl", "ShuntVoltageControl"], names
    branches = [0, 1, 2, 3]
    ls = ContingencyAnalysisCPP(grid)
    for b in branches:
        ls.add_n1(b)
    ls.change_algorithm(OUTER)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, 1e-10)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                                tol=1e-10 / grid.get_sn_mva(), keep_outer_trace=True)
    ca.add_contingencies_by_branch_id([[b] for b in branches])
    ca.compute(batch_size=2)
    res = _check_rows(ca, ls, grid, len(branches))
    loops_acting = {d.split(" #")[0] for r in res for d in r.decisions if "not taken" not in d}
    assert {"PhaseControl", "TransformerVoltageControl", "ShuntVoltageControl"} <= loops_acting, loops_acting
