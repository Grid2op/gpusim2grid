# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_physical_violations: the standby SVC check (lightsim2grid's
``SvcStandbyCheck.hpp``).

A non-regulating SVC a caller flagged as left idle under its standby automaton
(``LSGrid.set_svc_standby``, what ``bake_outer_loops`` returns handed to
``init_from_pypowsybl(can_be_pv=...)``), whose regulated bus leaves the
automaton's [low, high] voltage thresholds: OpenLoadFlow's
MonitoringVoltageOuterLoop would switch it to voltage control. Reported as
``LOW_VOLTAGE_SVC_STANDBY`` / ``HIGH_VOLTAGE_SVC_STANDBY`` on the SVC, value the
regulated voltage and limit the threshold, both in kV. Nothing is switched on.

gpusim2grid routes it through the PQ -> PV release plan (two entries of
el_type SVC per flagged SVC). The reference is lightsim2grid itself: a single
solve's ``LSGrid.get_physical_violations`` and its batch classes, on a 4-bus
radial feeder with a fixed-Q SVC on bus 2.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid import _gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_gen_pv_release", False),
                       reason="needs the bridge built against a lightsim2grid with can_be_pv"),
]


def _ls_has_svc_standby():
    try:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
    except ImportError:
        return False
    return hasattr(LSGrid, "set_svc_standby")


pytestmark.append(pytest.mark.skipif(not _ls_has_svc_standby(),
                                     reason="needs a lightsim2grid with LSGrid.set_svc_standby"))

SVC, LOW_SB, HIGH_SB = 7, 11, 12   # ViolationElementType.SVC, LOW/HIGH_VOLTAGE_SVC_STANDBY
GEN, LOW_VM = 5, 9                 # ViolationElementType.GENERATOR, LOW_VOLTAGE_AT_MIN_Q
VN_KV = 138.
SVC_BUS = 2
PINNED = 1
MAX_IT, TOL = 30, 1e-11
REACTIVE_POWER_MODE, VOLTAGE_MODE = 2, 1   # SvcContainer.RegulationMode


def _feeder(svc_bus=SVC_BUS, svc_reg_bus=None, svc_mode=REACTIVE_POWER_MODE, leaf=False,
            pinned=False, load_scale=1.):
    """The 4-bus radial feeder 0-1-2-3 (`load_scale` x 80 MW / 60 MVAr load on
    bus 3), gen 0 the PV slack on bus 0, and one SVC (fixed Q = 0) on `svc_bus`.
    `leaf` adds a leaf bus 4 off bus 1 (line 3); `pinned` adds gen 1, a PQ
    machine on bus 1 pinned at its min_q and flagged `can_be_pv` (target 1.10)."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    nb_bus = 5 if leaf else 4
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(nb_bus, 1, np.full(nb_bus, VN_KV), 0, 0)
    fr, to = [0, 1, 2] + ([1] if leaf else []), [1, 2, 3] + ([4] if leaf else [])
    nl = len(fr)
    grid.init_powerlines(np.full(nl, 0.01), np.full(nl, 0.1), np.zeros(nl, dtype=complex),
                         np.array(fr), np.array(to))
    grid.init_loads(np.array([80. * load_scale]), np.array([60. * load_scale]), np.array([3]))
    if pinned:
        grid.init_generators_full(np.array([0., 10.]), np.array([1.02, 1.10]),
                                  np.array([0., -5.]), [True, False],
                                  np.array([-1e3, -5.]), np.array([1e3, 20.]), np.array([0, 1]))
        grid.set_gen_can_be_pv(np.array([False, True]))
    else:
        grid.init_generators_full(np.array([0.]), np.array([1.02]), np.array([0.]), [True],
                                  np.array([-1e3]), np.array([1e3]), np.array([0]))
    grid.add_gen_slackbus(0, 1.)
    reg = svc_bus if svc_reg_bus is None else svc_reg_bus
    grid.init_svcs([svc_mode], np.array([1.0]), np.array([0.]), np.array([0.]),
                   np.array([-1.]), np.array([1.]), np.array([reg], dtype=np.int32),
                   np.array([svc_bus], dtype=np.int32))
    grid.tell_solver_need_reset()
    return grid


def _solve(grid):
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return V


def _flag(grid, low, high):
    grid.set_svc_standby(np.array([True]), np.array([low]), np.array([high]))
    return _solve(grid)


def _outside(high=True, **kwargs):
    """the feeder, its SVC flagged with thresholds the solved voltage of its bus is
    above (`high`) or below"""
    grid = _feeder(**kwargs)
    vm = abs(_solve(grid)[SVC_BUS])
    low, hi = (vm - 0.05, vm - 0.01) if high else (vm + 0.01, vm + 0.05)
    V = _flag(grid, low, hi)
    return grid, V, (low, hi)


def _standby(viols):
    """[(svc_id, type, value, limit)] of the standby records only."""
    return [(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
            for v in viols
            if int(v.element_type) == SVC and int(v.violation_type) in (LOW_SB, HIGH_SB)]


def _release(viols):
    return [(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
            for v in viols if int(v.element_type) == GEN and int(v.violation_type) == LOW_VM]


def _gpu_ca(grid, branches, handle_disconnected_grid=False, tol_vm_pu=0.):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True,
                                handle_disconnected_grid=handle_disconnected_grid)
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = tol_vm_pu
    ca.add_contingencies_by_branch_id([[int(b)] for b in branches])
    ca.compute(batch_size=8)
    return ca


def _assert_same(ref, got, atol_kv):
    assert [x[:2] for x in ref] == [x[:2] for x in got], f"{ref} vs {got}"
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b[2], a[2], atol=atol_kv)
        np.testing.assert_allclose(b[3], a[3], atol=atol_kv)


def _kv_atol(solver_atol):
    return 10. * VN_KV * solver_atol


# ---------------------------------------------------------------- the plan
def test_plan_carries_two_svc_entries():
    grid, _, (low, high) = _outside(high=True)
    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    me2s = np.asarray(grid.id_me_to_ac_solver())
    assert plan.n_entries == 2
    assert list(plan.el_type) == [SVC, SVC]
    assert list(plan.gen_id) == [0, 0]
    assert list(plan.at_min) == [1, 0]
    assert list(plan.reg_bus_solver) == [int(me2s[SVC_BUS])] * 2
    np.testing.assert_allclose(plan.target_vm_pu, [low, high])
    np.testing.assert_allclose(plan.vn_kv, [VN_KV, VN_KV])
    # not flagged: not a candidate
    grid = _feeder()
    _solve(grid)
    assert _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4).n_entries == 0


# -------------------------------------------------------- base ("n") case
@pytest.mark.parametrize("high, expected", [(True, HIGH_SB), (False, LOW_SB)])
def test_n_case_matches_single_solve(solver_atol, high, expected):
    grid, V, (low, hi) = _outside(high=high)
    ref = _standby(grid.get_physical_violations(True, 0., 0.))
    assert len(ref) == 1 and ref[0][1] == expected
    got = _standby(_gpu_ca(grid, [2]).get_physical_violations_n())
    _assert_same(ref, got, _kv_atol(solver_atol))
    me2s = np.asarray(grid.id_me_to_ac_solver())
    assert got[0][2] == pytest.approx(abs(V[me2s[SVC_BUS]]) * VN_KV, abs=_kv_atol(solver_atol))
    assert got[0][3] == pytest.approx((hi if high else low) * VN_KV, rel=solver_atol)


def test_inside_not_flagged_regulating_or_within_tolerance():
    grid = _feeder()
    vm = abs(_solve(grid)[SVC_BUS])
    _flag(grid, vm - 0.01, vm + 0.01)
    assert _standby(_gpu_ca(grid, [2]).get_physical_violations_n()) == []
    grid = _feeder()
    _solve(grid)
    assert _standby(_gpu_ca(grid, [2]).get_physical_violations_n()) == []
    grid, _, _ = _outside(high=True)
    assert _standby(_gpu_ca(grid, [2], tol_vm_pu=1.).get_physical_violations_n()) == []
    grid = _feeder(svc_mode=VOLTAGE_MODE)
    _flag(grid, 1.1, 1.2)
    assert _standby(_gpu_ca(grid, [2]).get_physical_violations_n()) == []


def test_array_mode_plan_equals_grid_plan(solver_atol):
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _, _ = _outside(high=True)
    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    ref = _standby(_gpu_ca(grid, [2]).get_physical_violations_n())
    assert len(ref) == 1
    ca = ContingencyAnalysisGPU(grid, nb_iter=10)
    ca.set_gen_pv_release_capability((plan.gen_id, plan.reg_bus_solver, plan.gen_bus_solver,
                                      plan.at_min, plan.target_vm_pu, plan.vn_kv, plan.el_type))
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = 0.
    ca.add_contingencies_by_branch_id([[2]])
    ca.compute(batch_size=8)
    _assert_same(ref, _standby(ca.get_physical_violations_n()), _kv_atol(solver_atol))


# ------------------------------------------------------------ contingency
def test_stranded_svc_reports_nothing(solver_atol):
    """handle_disconnected_grid: the SVC on the leaf bus 4, regulating bus 1;
    tripping line 3 strands it while the bus it regulates stays live: nothing
    reported -- lightsim2grid's batch agrees."""
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    grid = _feeder(svc_bus=4, svc_reg_bus=1, leaf=True)
    vm = abs(_solve(grid)[1])
    _flag(grid, vm - 0.05, vm - 0.01)
    ref_n = _standby(grid.get_physical_violations(True, 0., 0.))
    assert len(ref_n) == 1
    ls = ContingencyAnalysisCPP(grid)
    ls.compute_physical_violations = True
    ls.physical_violation_tol_mva = 0.
    ls.physical_violation_tol_vm_pu = 0.
    ls.handle_disconnected_grid = True
    ls.add_n1(3)
    ls.compute(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert list(ls.converged_mask()) == [True]
    assert _standby(ls.get_physical_violations()[0]) == []
    gpu = _gpu_ca(grid, [3], handle_disconnected_grid=True)
    assert np.isfinite(gpu.last_residuals()[0])
    assert _standby(gpu.get_physical_violations()[0]) == []
    _assert_same(ref_n, _standby(gpu.get_physical_violations_n()), _kv_atol(solver_atol))


# ---------------------------------------------------------- scenario sweep
def test_scenario_sweep_rows_match_lightsim2grid(solver_atol):
    """Rows whose load moves the SVC's bus inside, above and below its
    thresholds, on a grid that also has a pinned generator whose per-row target
    varies (set_gen_v): the SVC entries keep their thresholds (no row moves
    them), the generator its own row target -- both as lightsim2grid."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    # 70 % of the feeder's load: the full one already sits close to voltage collapse
    base_scale = 0.7
    grid = _feeder(pinned=True, load_scale=base_scale)
    vm = abs(_solve(grid)[SVC_BUS])
    V0 = _flag(grid, vm - 0.01, vm + 0.01)
    scale = base_scale * np.array([1.0, 0.7, 1.2])
    n = scale.size
    load_p = 80. * scale[:, None]
    load_q = 60. * scale[:, None]
    gen_p = np.repeat(np.asarray(grid.get_gen_target_p(), dtype=np.float64)[None, :], n, axis=0)
    gen_v = np.tile([1.02, 1.10], (n, 1))
    gen_v[1, PINNED] = 1.15

    ls = ScenarioSweepCPP(grid)
    ls.compute_physical_violations = True
    ls.physical_violation_tol_mva = 0.
    ls.physical_violation_tol_vm_pu = 0.
    ls.modify_load_p(load_p)
    ls.modify_load_q(load_q)
    ls.modify_gen_p(gen_p)
    ls.modify_gen_v(gen_v)
    ls.compute(V0.copy(), MAX_IT, TOL)
    assert all(ls.converged_mask())
    ref = ls.get_physical_violations()
    ref_sb, ref_gr = [_standby(r) for r in ref], [_release(r) for r in ref]
    assert [[x[1] for x in r] for r in ref_sb] == [[], [HIGH_SB], [LOW_SB]]
    assert all(len(r) == 1 for r in ref_gr)   # the pinned machine, on every row

    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.physical_violation_tol_vm_pu = 0.
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=n)
    got = sw.get_physical_violations()
    for a, b in zip(ref_sb, [_standby(r) for r in got]):
        _assert_same(a, b, _kv_atol(solver_atol))
    for a, b in zip(ref_gr, [_release(r) for r in got]):
        _assert_same(a, b, _kv_atol(solver_atol))
