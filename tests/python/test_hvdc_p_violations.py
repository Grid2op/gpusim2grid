# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_physical_violations, hvdc part -- the droop ("AC emulation") hvdc P-saturation
check: a linear-regime droop line whose theta-driven flow leaves the AC bus
above pmax in the direction it flows (what OpenLoadFlow's HvdcAcEmulationLimits
outer loop saturates). lightsim2grid has no batch counterpart yet, so the pin
is its single solve: the published station injection is -p_flow, and the flow
itself is p0 + k (theta1 - theta2) (controller side, r = 0 here).
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge, reason="hvdc droop data comes through the lightsim2grid C++ bridge")

_BIG_Q = 1e4
MAX_IT, TOL = 30, 1e-10


def _droop_grid(pmax12=300., pmax21=300., p0=10., k_mw_per_deg=5., status=0,
                b1=3, b2=9, lf1=0.01, lf2=0.02):
    """case14 with one droop-enabled VSC-VSC hvdc line (the augmented-features
    helper, with the limits and the regime as parameters), ac-solved."""
    pn = pytest.importorskip("pandapower.networks")
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
        if not hasattr(grid, "init_hvdc_lines"):
            pytest.skip("this lightsim2grid build has no init_hvdc_lines")
        grid.init_hvdc_lines(
            np.array([b1], dtype=np.int32), np.array([b2], dtype=np.int32),
            [0], [0], np.array([lf1]), np.array([lf2]), [False], [False],
            np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]),
            np.array([-_BIG_Q]), np.array([_BIG_Q]), np.array([-_BIG_Q]), np.array([_BIG_Q]),
            np.array([1.0]), np.array([1.0]), [0], np.array([20.0]),
            np.array([0.0]), np.array([0.0]), [True],
            np.array([p0]), np.array([k_mw_per_deg]),
            np.array([pmax12]), np.array([pmax21]))
        if status != 0:
            grid.set_status_droop_hvdc(0, int(status))
        grid.tell_solver_need_reset()
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_model = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_model, dtype=complex), 1, 1e-6)
        V = grid.ac_pf(v0.copy(), MAX_IT, TOL)
        assert V.shape[0] > 0
    return grid, v0


def _expected_from_V(grid, V_solver, tol_mw=0.):
    """The check recomputed from a converged voltage vector (solver
    numbering): [(hvdc id, side, value MW, limit MW)], linear regime only."""
    me2s = np.asarray(grid.id_me_to_ac_solver(), dtype=int)
    out = []
    for h in grid.get_dclines():
        if not h.droop_enabled or h.status_droop != 0:
            continue
        th1 = np.angle(V_solver[me2s[h.bus1_id]]); th2 = np.angle(V_solver[me2s[h.bus2_id]])
        raw = h.droop_p0_mw + h.droop_k_mw_per_rad * (th1 - th2)
        if raw >= 0 and raw > h.pmax_1to2_mw + tol_mw:
            out.append((h.id, 1, raw, h.pmax_1to2_mw))
        elif raw < 0 and -raw > h.pmax_2to1_mw + tol_mw:
            out.append((h.id, 2, -raw, h.pmax_2to1_mw))
    return out


def _hvdc_only(rows):
    """the HVDC records of a physical report (case14's own generator limits do
    get violated on a few contingencies -- those BUS records are the other
    check's business)"""
    from gpusim2grid.contingency_analysis import ViolationElementType
    return [[v for v in row if v.element_type == ViolationElementType.HVDC] for row in rows]


def _got(rows):
    return [[(v.element_id, v.side, v.value, v.limit) for v in row] for row in _hvdc_only(rows)]


def _assert_same(ref, got, atol_mw):
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert [x[:2] for x in a] == [x[:2] for x in b], (a, b)
        for x, y in zip(a, b):
            np.testing.assert_allclose(y[2], x[2], atol=atol_mw)
            np.testing.assert_allclose(y[3], x[3], atol=atol_mw)


def _n1(grid, n=6):
    return [[i] for i in range(n)]


@needs_bridge
@pytest.mark.parametrize("p0,pmax12,pmax21,side", [(10., 15., 300., 1), (-40., 300., 5., 2)])
def test_saturation_detected_both_directions(solver_atol, p0, pmax12, pmax21, side):
    """pmax below the solved flow: one record per row, on the side the flow
    goes, equal to the flow lightsim2grid publishes (-res_p_side) and to the
    droop formula recomputed on lightsim2grid's own contingency voltages."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.contingency_analysis import (
        LimitViolationType, ViolationCategory, ViolationElementType)
    grid, v0 = _droop_grid(pmax12=pmax12, pmax21=pmax21, p0=p0)
    h = grid.get_dclines()[0]
    atol_mw = 50. * float(grid.get_sn_mva()) * solver_atol
    ctgs = _n1(grid)

    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute()

    # base case: the flow lightsim2grid published on the grid itself
    n = _hvdc_only([ca.get_physical_violations_n()])[0]
    assert len(n) == 1
    v = n[0]
    assert v.element_type == ViolationElementType.HVDC
    assert v.violation_type == LimitViolationType.HVDC_P_SATURATION
    assert v.category == ViolationCategory.PHYSICAL
    assert (v.element_id, v.side) == (0, side)
    published = -h.res_p1_mw if side == 1 else -h.res_p2_mw
    np.testing.assert_allclose(v.value, published, atol=atol_mw)
    np.testing.assert_allclose(v.limit, pmax12 if side == 1 else pmax21, atol=atol_mw)
    _assert_same([_expected_from_V(grid, grid.get_V_solver())], _got([n]), atol_mw)

    # contingency rows: recompute on lightsim2grid's own post-contingency V
    ref = ContingencyAnalysisCPP(grid)
    for c in ctgs:
        ref.add_n1(int(c[0]))
    ref.compute(v0.copy(), MAX_IT, TOL)
    Vs = np.asarray(ref.get_voltages())
    n_bus = grid.get_Ybus_solver().shape[0]
    expected = []
    for r in range(len(ctgs)):
        if not ref.converged_mask()[r]:
            expected.append([]); continue
        Vr = Vs[r, :n_bus]
        expected.append(_expected_from_V(grid, Vr))
    got = ca.get_physical_violations()
    assert sum(len(x) for x in got) > 0
    _assert_same(expected, _got(got), atol_mw)
    assert not ca.get_physical_violations_truncated().any()


@needs_bridge
def test_wide_pmax_saturated_regime_and_no_hvdc_report_nothing():
    from gpusim2grid import ContingencyAnalysisGPU
    for kwargs in (dict(pmax12=300.), dict(pmax12=15., status=1)):
        grid, _ = _droop_grid(**kwargs)
        ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
        ca.physical_violation_tol_mva = 0.
        ca.add_contingencies_by_branch_id(_n1(grid, 3))
        ca.compute()
        assert all(row == [] for row in _hvdc_only(ca.get_physical_violations())), kwargs
        assert _hvdc_only([ca.get_physical_violations_n()])[0] == [], kwargs
    # a grid without any hvdc line: the check is on, and silent
    pn = pytest.importorskip("pandapower.networks")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, TOL)
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca.add_contingencies_by_branch_id(_n1(grid, 3))
    ca.compute()
    assert all(row == [] for row in _hvdc_only(ca.get_physical_violations()))
    assert ca.solver._s.get_hvdc_p_violations().count.tolist() == [0, 0, 0]


@needs_bridge
def test_defaults_flag_off_and_tolerance():
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _ = _droop_grid(pmax12=15.)
    ca = ContingencyAnalysisGPU(grid, nb_iter=10)
    assert ca.compute_physical_violations is False
    assert ca.physical_violation_tol_mva == 1e-4
    assert ca.physical_violation_capacity == 16
    ca.add_contingencies_by_branch_id(_n1(grid, 2))
    ca.compute()
    with pytest.raises(RuntimeError):
        ca.get_physical_violations()
    with pytest.raises(RuntimeError):
        ca.physical_violation_tol_mva = float("nan")
    # a tolerance wider than the excess hides the record
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = 100.
    ca.compute()
    assert all(row == [] for row in _hvdc_only(ca.get_physical_violations()))
    ca.physical_violation_tol_mva = 0.
    ca.compute()
    assert all(len(row) == 1 for row in _hvdc_only(ca.get_physical_violations()))
    # independent of the operational check
    assert ca.compute_limit_violations is False


@needs_bridge
def test_scenario_and_injection_sweeps(solver_atol):
    from gpusim2grid import ScenarioSweepGPU, InjectionSweepGPU
    grid, _ = _droop_grid(pmax12=15.)
    atol_mw = 50. * float(grid.get_sn_mva()) * solver_atol
    load_p, load_q = (np.asarray(a, dtype=np.float64) for a in grid.get_loads_res_full()[:2])
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    ks = [0.8, 1.0, 1.2]
    lp = np.stack([k * load_p for k in ks]); lq = np.stack([k * load_q for k in ks])
    gp = np.repeat(gen_p[None, :], len(ks), axis=0)
    n_bus = grid.get_Ybus_solver().shape[0]

    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(lp, lq, gp)
    sw.set_topology([[], [0], []])
    sw.compute()
    V = sw.solver.V_results.to_numpy().reshape(len(ks), n_bus)
    _assert_same([_expected_from_V(grid, V[r]) for r in range(len(ks))],
                 _got(sw.get_physical_violations()), atol_mw)
    assert len(_hvdc_only([sw.get_physical_violations_n()])[0]) == 1

    isw = InjectionSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    isw.physical_violation_tol_mva = 0.
    isw.set_injections_from_elements(lp, lq, gp)
    isw.compute()
    V = isw.solver.V_results.to_numpy().reshape(len(ks), n_bus)
    _assert_same([_expected_from_V(grid, V[r]) for r in range(len(ks))],
                 _got(isw.get_physical_violations()), atol_mw)
    assert len(_hvdc_only([isw.get_physical_violations_n()])[0]) == 1
