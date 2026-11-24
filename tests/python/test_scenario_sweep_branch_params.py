# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ScenarioSweepGPU.set_line_parameters / set_trafo_parameters: each row gives some branches
parameters of their own (lightsim2grid PR #303).

Oracles: the grid itself holding the row's values (``update_powerlines_parameters`` /
``update_trafos_parameters_full`` / ``change_ratio_trafo`` / ``change_shift_trafo``, then a
one-off ``ac_pf``), and lightsim2grid's own ``ScenarioSweepCPP`` given the same parameters --
on a transformer with a tap changer too, where r and x are the NEUTRAL values.
"""
import warnings

import numpy as np
import pytest

from conftest import parity_tol, requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_branch_params", False),
                       reason="needs a lightsim2grid with per-row branch parameters (PR #303)"),
]
needs_fp64 = pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity tolerances are FP64 ones")

MAX_IT, TOL = 20, 1e-10
NAN_C = complex(np.nan, np.nan)
N_ROWS = 6


def _nan(shape):
    return np.full(shape, np.nan)


def _nan_c(shape):
    return np.full(shape, NAN_C, dtype=complex)


def _case14():
    """pandapower's case14, solved (NR_KLU)."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    assert grid.ac_pf(np.ones(n_bus, dtype=complex), MAX_IT, TOL).shape[0] > 0
    return grid


def _tapped_ieee14():
    """pypowsybl's IEEE 14 with a ratio tap changer on T4-9-1, away from its neutral step, whose
    steps correct r and x: a transformer whose r, x at the tap differ from the neutral ones."""
    pp = pytest.importorskip("pypowsybl")
    import pandas as pd
    from lightsim2grid.network import init_from_pypowsybl
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    net = pp.network.create_ieee14()
    rtc = pd.DataFrame.from_records(
        index="id", columns=["id", "target_deadband", "target_v", "oltc", "low_tap", "tap", "regulating",
                             "regulated_side"],
        data=[("T4-9-1", 0.1, 12.6, True, 0, 14, False, "TWO")])
    steps = pd.DataFrame.from_records(
        index="id", columns=["id", "b", "g", "r", "x", "rho"],
        data=[("T4-9-1", 0., 0., 2. * (k - 10), 3. * (k - 10), 0.9 + 0.01 * k) for k in range(21)])
    net.create_ratio_tap_changers(rtc, steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id=0, sort_index=False)
    grid.change_algorithm(AlgorithmType.NR_KLU)
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL).shape[0] > 0
    return grid


class _Params:
    """Some rows of parameters for 3 lines and 2 transformers of `grid`; row 2 changes nothing
    (all NaN)."""

    def __init__(self, grid, line_ids=(0, 3, 7), trafo_ids=(0, 2)):
        rng = np.random.default_rng(0)
        lines, trafos = grid.get_lines(), grid.get_trafos()
        self.line_ids = np.array(line_ids)
        self.trafo_ids = np.array(trafo_ids)
        nl, nt = len(line_ids), len(trafo_ids)
        base_x = np.array([lines[i].x_pu for i in self.line_ids])
        self.line_r = _nan((N_ROWS, nl))
        self.line_x = base_x * rng.uniform(0.5, 2., (N_ROWS, nl))
        self.line_h1 = _nan_c((N_ROWS, nl))
        self.line_h2 = _nan_c((N_ROWS, nl))
        self.line_r[1, :] = [2. * lines[i].r_pu for i in self.line_ids]
        self.line_h1[3, 1] = 0.01j
        self.line_h2[3, 1] = 0.02j
        self.line_x[2, :] = np.nan
        self.trafo_r = _nan((N_ROWS, nt))
        self.trafo_x = _nan((N_ROWS, nt))
        self.trafo_h1 = _nan_c((N_ROWS, nt))
        self.trafo_h2 = _nan_c((N_ROWS, nt))
        self.trafo_ratio = _nan((N_ROWS, nt))
        self.trafo_shift = _nan((N_ROWS, nt))
        self.trafo_x[:, 0] = trafos[int(self.trafo_ids[0])].x_pu * np.linspace(0.6, 1.6, N_ROWS)
        self.trafo_x[2, 0] = np.nan
        self.trafo_ratio[3, 1] = 1.03
        self.trafo_shift[4, :] = 0.05
        self.trafo_r[5, 1] = 3. * trafos[int(self.trafo_ids[1])].r_pu

    def apply(self, sw):
        sw.set_line_parameters(self.line_ids, self.line_r, self.line_x, self.line_h1, self.line_h2)
        sw.set_trafo_parameters(self.trafo_ids, self.trafo_r, self.trafo_x, self.trafo_h1, self.trafo_h2,
                                self.trafo_ratio, self.trafo_shift)

    def grid_of_row(self, grid, row):
        """`grid` holding row `row`'s values, given the way LSGrid takes them (an untapped grid:
        the transformers' r, x read off them are their neutral ones)"""
        g = grid.copy()
        lines, trafos = grid.get_lines(), grid.get_trafos()
        for container, ids, mats, update in (
                (lines, self.line_ids, (self.line_r, self.line_x, self.line_h1, self.line_h2),
                 g.update_powerlines_parameters),
                (trafos, self.trafo_ids, (self.trafo_r, self.trafo_x, self.trafo_h1, self.trafo_h2),
                 g.update_trafos_parameters_full)):
            vects = [np.array([el.r_pu for el in container]), np.array([el.x_pu for el in container]),
                     np.array([el.h1_pu for el in container]), np.array([el.h2_pu for el in container])]
            for k, el_id in enumerate(ids):
                for vect, mat in zip(vects, mats):
                    if not np.isnan(mat[row, k]):
                        vect[el_id] = mat[row, k]
            update(*vects)
        for k, el_id in enumerate(self.trafo_ids):
            if not np.isnan(self.trafo_ratio[row, k]):
                g.change_ratio_trafo(int(el_id), self.trafo_ratio[row, k])
            if not np.isnan(self.trafo_shift[row, k]):
                g.change_shift_trafo(int(el_id), self.trafo_shift[row, k])
        return g


def _sweep(grid, n_rows=N_ROWS, params=None, topology=None, **kwargs):
    from gpusim2grid import ScenarioSweepGPU

    sw = ScenarioSweepGPU(grid, nb_iter=MAX_IT, tol_base=TOL, **kwargs)
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None, :], n_rows, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    if topology is not None:
        sw.set_topology(topology)
    if params is not None:
        params.apply(sw)
    return sw


def _results(sw, grid, n_rows=N_ROWS):
    sw.compute(batch_size=n_rows)
    n_bus = sw.n_bus
    V = sw.solver.V_results.to_numpy().reshape(n_rows, n_bus)
    sw.compute_flows()
    n_br = sw.n_branches
    a_or = sw.or_amps.to_numpy().reshape(n_rows, n_br)
    a_ex = sw.ex_amps.to_numpy().reshape(n_rows, n_br)
    return V, a_or, a_ex


def _assert_amps(gpu_a, ref_ka, atol_pu, err_msg):
    """Currents (A) against a reference solved by another solver (kA): the NR tolerance
    relative to the flows' scale."""
    ref = 1e3 * np.asarray(ref_ka)
    np.testing.assert_allclose(gpu_a, ref, rtol=atol_pu, atol=atol_pu * np.max(np.abs(ref)), err_msg=err_msg)


# --------------------------------------------------------------------- tests
def test_rows_match_the_grid_given_their_values(solver_atol):
    """Every row solves, and reports both sides' currents, as the grid holding its values."""
    grid = _case14()
    p = _Params(grid)
    sw = _sweep(grid, params=p)
    V, a_or, a_ex = _results(sw, grid)
    assert np.all(sw.last_residuals() < solver_atol)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    for row in range(N_ROWS):
        g = p.grid_of_row(grid, row)
        V_ref = g.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
        assert V_ref.shape[0] > 0, f"row {row}: the reference diverged"
        np.testing.assert_allclose(V[row], V_ref[buses], atol=solver_atol, err_msg=f"row {row}")
        ref_or = [el.res_a1_ka for el in g.get_lines()] + [el.res_a1_ka for el in g.get_trafos()]
        ref_ex = [el.res_a2_ka for el in g.get_lines()] + [el.res_a2_ka for el in g.get_trafos()]
        _assert_amps(a_or[row], ref_or, solver_atol, f"row {row}, side 1")
        _assert_amps(a_ex[row], ref_ex, solver_atol, f"row {row}, side 2")


def test_tapped_transformer_matches_lightsim2grid_scenario_sweep(solver_atol):
    """A transformer with a tap changer away from its neutral step: r and x are the NEUTRAL
    values, the step's corrections applied on top -- lightsim2grid's own ScenarioSweep row for
    row (with the same injections), voltages and both sides' currents."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid = _tapped_ieee14()
    t491 = next(i for i, t in enumerate(grid.get_trafos()) if t.name == "T4-9-1")
    p = _Params(grid, line_ids=(0, 3, 7), trafo_ids=(t491, 0))
    p.trafo_r[:, 0] = np.linspace(0.5, 2., N_ROWS) * 0.001
    sw = _sweep(grid, params=p)
    V, a_or, a_ex = _results(sw, grid)

    ls = ScenarioSweepCPP(grid)
    p.apply(ls)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(ls.get_voltages())[:, buses]
    np.testing.assert_allclose(V, V_ls, atol=solver_atol)
    _assert_amps(a_or, ls.compute_flows(), solver_atol, "side 1")
    # the tap's correction matters here: the neutral values given as such are not the grid's
    # values at the tap, so the row really differs from the plain one
    plain = _sweep(grid)
    V_plain, _, _ = _results(plain, grid)
    assert np.max(np.abs(V_plain[0] - V[0])) > 100 * solver_atol


@needs_fp64
def test_a_row_changing_nothing_is_the_plain_row():
    """A row whose cells are all NaN, and a sweep cleared of its parameters, give the plain
    sweep: to rounding (two sessions, see CLAUDE.md)."""
    grid = _case14()
    p = _Params(grid)
    V_plain, a_plain, _ = _results(_sweep(grid), grid)
    sw = _sweep(grid, params=p)
    V, a_or, _ = _results(sw, grid)
    tol = parity_tol("state")
    np.testing.assert_allclose(V[2], V_plain[2], rtol=0, atol=tol)
    np.testing.assert_allclose(a_or[2], a_plain[2], rtol=tol, atol=tol * np.max(a_plain[2]))
    assert np.max(np.abs(V[0] - V_plain[0])) > 1e3 * tol   # (the others do change)
    sw.set_line_parameters([])
    sw.set_trafo_parameters([])
    V2, a2, _ = _results(sw, grid)
    np.testing.assert_allclose(V2, V_plain, rtol=0, atol=tol)
    np.testing.assert_allclose(a2, a_plain, rtol=tol, atol=tol * np.max(a_plain))


def test_with_topology_the_trip_wins(solver_atol):
    """A row tripping a line it gives parameters ignores them; another row trips a line and
    changes others: the grid holding its values with that line off."""
    grid = _case14()
    p = _Params(grid)
    topology = [[] for _ in range(N_ROWS)]
    topology[1] = [3]          # tripped AND given parameters in that row: the trip wins
    topology[4] = [10]         # tripped, other branches given parameters
    sw = _sweep(grid, params=p, topology=topology)
    V, a_or, _ = _results(sw, grid)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    for row, line in ((1, 3), (4, 10)):
        g = p.grid_of_row(grid, row)
        g.deactivate_powerline(line)
        V_ref = g.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
        assert V_ref.shape[0] > 0
        np.testing.assert_allclose(V[row], V_ref[buses], atol=solver_atol, err_msg=f"row {row}")
        assert a_or[row, line] == 0.
        ref_or = [el.res_a1_ka for el in g.get_lines()] + [el.res_a1_ka for el in g.get_trafos()]
        _assert_amps(a_or[row], ref_or, solver_atol, f"row {row}")


def test_current_violations_read_the_rows_block(solver_atol):
    """The fused current check reads a changed branch's own block: its CURRENT record's value
    is the row's current (compute_flows), not what the grid's block would give."""
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType

    grid = _case14()
    p = _Params(grid)
    sw = _sweep(grid, params=p)
    V, a_or, a_ex = _results(sw, grid)
    line = int(p.line_ids[0])
    # a limit just below the smallest current the line carries in rows 0..5 (both sides)
    i_min = min(np.min(a_or[:, line]), np.min(a_ex[:, line]))
    sw2 = _sweep(grid, params=p, compute_limit_violations=True)
    vmin, vmax, _, _, n_lines = sw2._extract_limits_arrays()
    n_br = sw2.n_branches
    lim1, lim2 = np.full(n_br, np.nan), np.full(n_br, np.nan)
    lim1[line] = lim2[line] = 0.9 * i_min * 1e-3
    sw2._inner.set_limits(vmin, vmax, lim1, lim2, n_lines)
    sw2.compute(batch_size=N_ROWS)
    viols = sw2.get_violations()
    for row in range(N_ROWS):
        cur = [v for v in viols[row] if v.violation_type == LimitViolationType.CURRENT and v.element_id == line]
        assert cur, f"row {row}: no CURRENT record on line {line}"
        got = max(v.value for v in cur) * 1e3
        want = max(a_or[row, line], a_ex[row, line])
        np.testing.assert_allclose(got, want, rtol=solver_atol, err_msg=f"row {row}")


def test_setters_refuse(solver_atol):
    grid = _case14()
    sw = _sweep(grid)
    n_line = len(grid.get_lines())
    x = np.ones((N_ROWS, 1))
    with pytest.raises(ValueError, match="out of range"):
        sw.set_line_parameters([n_line], x=x)
    with pytest.raises(ValueError, match="named twice"):
        sw.set_line_parameters([1, 1], x=np.ones((N_ROWS, 2)))
    with pytest.raises(ValueError, match="shape"):
        sw.set_line_parameters([1], x=np.ones((N_ROWS, 2)))
    with pytest.raises(ValueError, match="both 0"):
        sw.set_line_parameters([1], r=np.zeros((N_ROWS, 1)), x=np.zeros((N_ROWS, 1)))
    with pytest.raises(ValueError, match="infinite"):
        sw.set_line_parameters([1], x=np.full((N_ROWS, 1), np.inf))
    with pytest.raises(ValueError, match="positive"):
        sw.set_trafo_parameters([0], ratio=np.zeros((N_ROWS, 1)))
    with pytest.raises(ValueError, match="at least one"):
        sw.set_trafo_parameters([0])
    with pytest.raises(ValueError, match="rows"):
        sw.set_line_parameters([1], x=np.ones((N_ROWS + 1, 1)))
    with pytest.raises(ValueError, match="named"):
        sw.set_line_parameters(["no such line"], x=x)
    # names work, and the call that failed left nothing behind
    name = grid.get_lines()[1].name
    sw.set_line_parameters([name], x=2. * np.full((N_ROWS, 1), grid.get_lines()[1].x_pu))
    sw.compute(batch_size=N_ROWS)
    assert np.all(sw.last_residuals() < solver_atol)


def test_outer_loops_regulated_transformer_is_left_out(solver_atol):
    """Under outer loops, a row changing a transformer a TransformerVoltageControl loop
    regulates is BATCH_UNSUPPORTED (the loop re-stamps it from the grid's values); a row
    changing a line is solved."""
    if not getattr(_cpp, "have_ls2g_outer_loops", False):
        pytest.skip("lightsim2grid without its outer loops")
    import test_outer_loops_batch as olb
    from gpusim2grid import ScenarioSweepGPU

    grid, trafo = olb._tvc()
    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=olb.MAX_IT, init_from_n_powerflow=False,
                          tol=olb.TOL_MVA / grid.get_sn_mva(), ir_n_steps=1)
    n_rows = 3
    load_p, load_q = grid.get_loads_res_full()[:2]
    rep = lambda a: np.repeat(np.asarray(a, dtype=np.float64)[None, :], n_rows, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(grid.get_gen_target_p()))
    x = _nan((n_rows, 1))
    x[1, 0] = 1.1 * grid.get_trafos()[trafo].x_pu
    sw.set_trafo_parameters([trafo], x=x)
    lx = _nan((n_rows, 1))
    lx[2, 0] = 1.2 * grid.get_lines()[0].x_pu
    sw.set_line_parameters([0], x=lx)
    sw.compute(batch_size=n_rows)
    reasons = sw.get_batch_unsupported_reasons()
    assert reasons[0] == "" and reasons[2] == ""
    assert "TransformerVoltageControl" in reasons[1]
    res = sw.last_residuals()
    assert np.isnan(res[1])
    assert res[0] < solver_atol and res[2] < solver_atol
