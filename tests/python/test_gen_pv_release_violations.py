# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_physical_violations: the PQ -> PV release check (lightsim2grid PR
#216, ``GenPvReleaseCheck.hpp``).

A PQ generator a caller flagged as pinned at a reactive limit by an outer loop
(``LSGrid.set_gen_can_be_pv``), sitting at its min_q (resp. max_q), whose
REGULATED bus sits below (resp. above) the target it would hold, absorbs (resp.
produces) too much for that target: OpenLoadFlow's ReactiveLimits loop would
switch it back to PV. Reported as ``LOW_VOLTAGE_AT_MIN_Q`` /
``HIGH_VOLTAGE_AT_MAX_Q`` on the GENERATOR, value the regulated voltage and
limit the target, both in kV. Nothing is switched back.

The reference is lightsim2grid itself: a single solve's
``LSGrid.get_physical_violations`` and its batch classes (ContingencyAnalysisCPP
/ ScenarioSweepCPP with ``compute_physical_violations``), on the same grids as
its own tests -- a 4-bus radial feeder whose PQ machine on bus 1 is pinned at a
reactive limit, and a variant where that machine sits on a leaf and regulates
bus 1 remotely (tripping the leaf's line strands the machine).
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

GEN, LOW_VM, HIGH_VM = 5, 9, 10   # ViolationElementType.GENERATOR, LOW/HIGH_VOLTAGE_AT_M*_Q
VN_KV = 138.
PINNED = 1
MAX_IT, TOL = 30, 1e-11


def _pinned_grid(target_vm=1.10, at_min=True, flagged=True):
    """The 4-bus radial feeder 0-1-2-3 (80 MW / 60 MVAr load on bus 3), gen 0
    the PV slack on bus 0, gen 1 a PQ machine on bus 1 pinned at its min_q (or
    max_q) -- lightsim2grid's own TestGenPvReleaseFromPython grid, solved."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    min_q, max_q = -5., 20.
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, VN_KV), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators_full(np.array([0., 10.]), np.array([1.02, target_vm]),
                              np.array([0., min_q if at_min else max_q]), [True, False],
                              np.array([-1e3, min_q]), np.array([1e3, max_q]), np.array([0, 1]))
    if flagged:
        grid.set_gen_can_be_pv(np.array([False, True]))
    grid.add_gen_slackbus(0, 1.)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return grid, V


def _remote_pinned_grid():
    """The same feeder plus a leaf bus 4 hanging off bus 1 (line 3): the pinned
    machine sits on bus 4 and regulates bus 1 remotely. Tripping line 3
    strands the machine while the bus it regulates stays live."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    min_q, max_q = -5., 20.
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(5, 1, np.full(5, VN_KV), 0, 0)
    grid.init_powerlines(np.full(4, 0.01), np.full(4, 0.1), np.zeros(4, dtype=complex),
                         np.array([0, 1, 2, 1]), np.array([1, 2, 3, 4]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators_full(np.array([0., 10.]), np.array([1.02, 1.10]),
                              np.array([0., min_q]), [True, False],
                              np.array([-1e3, min_q]), np.array([1e3, max_q]), np.array([0, 4]))
    grid.set_gen_regulated_bus(1, 1)
    grid.set_gen_can_be_pv(np.array([False, True]))
    grid.add_gen_slackbus(0, 1.)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return grid, V


def _release(viols):
    """[(gen_id, type, value, limit)] of the release records only."""
    return [(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
            for v in viols
            if int(v.element_type) == GEN and int(v.violation_type) in (LOW_VM, HIGH_VM)]


def _ls_ca(grid, branches, handle_disconnected_grid=False, tol_vm_pu=0.):
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(grid)
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = tol_vm_pu
    ca.handle_disconnected_grid = handle_disconnected_grid
    for b in branches:
        ca.add_n1(int(b))
    ca.compute(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    return ca


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
def test_plan_is_lightsim2grid_selection():
    grid, _ = _pinned_grid(1.10, at_min=True)
    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    me2s = np.asarray(grid.id_me_to_ac_solver())
    assert plan.n_entries == 1
    assert list(plan.gen_id) == [PINNED]
    assert list(plan.reg_bus_solver) == [int(me2s[1])]
    assert list(plan.gen_bus_solver) == [int(me2s[1])]
    assert list(plan.at_min) == [1]
    np.testing.assert_allclose(plan.target_vm_pu, [1.10])
    np.testing.assert_allclose(plan.vn_kv, [VN_KV])
    # not flagged: not a candidate
    grid, _ = _pinned_grid(1.10, at_min=True, flagged=False)
    assert _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4).n_entries == 0


# -------------------------------------------------------- base ("n") case
@pytest.mark.parametrize("target_vm, at_min, expected", [
    (1.10, True, LOW_VM),      # at min_q, regulated bus below the target
    (0.80, False, HIGH_VM),    # at max_q, regulated bus above the target
])
def test_n_case_matches_single_solve(solver_atol, target_vm, at_min, expected):
    grid, V = _pinned_grid(target_vm, at_min=at_min)
    ref = _release(grid.get_physical_violations(True, 0., 0.))
    assert len(ref) == 1 and ref[0][1] == expected
    gpu = _gpu_ca(grid, [2])          # line 2 islands the load: that row is skipped
    got = _release(gpu.get_physical_violations_n())
    _assert_same(ref, got, _kv_atol(solver_atol))
    me2s = np.asarray(grid.id_me_to_ac_solver())
    assert got[0][2] == pytest.approx(abs(V[me2s[1]]) * VN_KV, abs=_kv_atol(solver_atol))
    assert got[0][3] == pytest.approx(target_vm * VN_KV, rel=solver_atol)
    assert _release(gpu.get_physical_violations()[0]) == []   # not simulated


def test_not_flagged_wrong_side_or_within_tolerance():
    grid, _ = _pinned_grid(1.10, at_min=True, flagged=False)
    assert _release(_gpu_ca(grid, [2]).get_physical_violations_n()) == []
    grid, _ = _pinned_grid(0.80, at_min=True)   # voltage above the target: not a release
    assert _release(_gpu_ca(grid, [2]).get_physical_violations_n()) == []
    grid, _ = _pinned_grid(1.10, at_min=True)
    assert _release(_gpu_ca(grid, [2], tol_vm_pu=1.).get_physical_violations_n()) == []


def test_array_mode_plan_equals_grid_plan(solver_atol):
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _ = _pinned_grid(1.10, at_min=True)
    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    ref = _release(_gpu_ca(grid, [2]).get_physical_violations_n())
    ca = ContingencyAnalysisGPU(grid, nb_iter=10)
    ca.set_gen_pv_release_capability((plan.gen_id, plan.reg_bus_solver, plan.gen_bus_solver,
                                      plan.at_min, plan.target_vm_pu, plan.vn_kv))
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = 0.
    ca.add_contingencies_by_branch_id([[2]])
    ca.compute(batch_size=8)
    _assert_same(ref, _release(ca.get_physical_violations_n()), _kv_atol(solver_atol))


# ------------------------------------------------------------ contingency
def test_stranded_machine_releases_nothing(solver_atol):
    """handle_disconnected_grid: a contingency stranding the machine (its own
    bus masked) while the bus it regulates stays live: nothing to release, as
    when the machine is disconnected -- lightsim2grid's batch agrees."""
    grid, _ = _remote_pinned_grid()
    ref_n = _release(grid.get_physical_violations(True, 0., 0.))
    assert len(ref_n) == 1
    ls = _ls_ca(grid, [3], handle_disconnected_grid=True)
    assert list(ls.converged_mask()) == [True]
    assert _release(ls.get_physical_violations()[0]) == []
    gpu = _gpu_ca(grid, [3], handle_disconnected_grid=True)
    assert np.isfinite(gpu.last_residuals()[0])
    assert _release(gpu.get_physical_violations()[0]) == []
    _assert_same(ref_n, _release(gpu.get_physical_violations_n()), _kv_atol(solver_atol))


# ---------------------------------------------------------- scenario sweep
def _rows(grid, n):
    load_p, load_q = (np.asarray(a, dtype=np.float64) for a in grid.get_loads_res_full()[:2])
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    rep = lambda a: np.repeat(a[None, :], n, axis=0)   # noqa: E731
    return rep(load_p), rep(load_q), rep(gen_p)


def test_scenario_sweep_row_targets_match_lightsim2grid(solver_atol):
    """Per row, the target a flagged machine is checked against is that row's
    own gen_v (lightsim2grid reads it off modify_gen_v): a target below the
    regulated voltage on row 1 is no longer a release, a higher one on row 2
    is a bigger one. A row
    that disconnects the machine reports nothing. A hot re-run with other
    targets reports afresh."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    grid, V0 = _pinned_grid(1.10, at_min=True)
    n_gen = len(grid.get_generators())
    targets = np.array([1.10, 0.85, 1.15, 1.10])   # |V(bus 1)| ~ 0.877 pu
    gen_v = np.tile([1.02, 0.], (4, 1))
    gen_v[:, PINNED] = targets
    gens_off = np.zeros((4, n_gen), dtype=bool)
    gens_off[3, PINNED] = True
    load_p, load_q, gen_p = _rows(grid, 4)

    def ref(gv):
        ls = ScenarioSweepCPP(grid)
        ls.compute_physical_violations = True
        ls.physical_violation_tol_mva = 0.
        ls.physical_violation_tol_vm_pu = 0.
        ls.modify_load_p(load_p)
        ls.modify_load_q(load_q)
        ls.modify_gen_p(gen_p)
        ls.modify_gen_v(gv)
        ls.set_contingency_gens(gens_off)
        ls.compute(V0.copy(), MAX_IT, TOL)
        assert all(ls.converged_mask())
        return [_release(r) for r in ls.get_physical_violations()]

    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.physical_violation_tol_vm_pu = 0.
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_gen_v(gen_v)
    sw.set_contingency_gens(gens_off)
    sw.compute(batch_size=4)
    expected = ref(gen_v)
    got = [_release(r) for r in sw.get_physical_violations()]
    assert [len(r) for r in expected] == [1, 0, 1, 0]
    for a, b in zip(expected, got):
        _assert_same(a, b, _kv_atol(solver_atol))

    n_built = sw.source_build_counter
    gen_v2 = gen_v.copy()
    gen_v2[:, PINNED] = [0.85, 1.10, 1.10, 1.15]
    sw.set_gen_v(gen_v2)
    sw.compute(batch_size=4)
    assert sw.source_build_counter == n_built                    # hot
    expected = ref(gen_v2)
    assert [len(r) for r in expected] == [0, 1, 1, 0]
    for a, b in zip(expected, [_release(r) for r in sw.get_physical_violations()]):
        _assert_same(a, b, _kv_atol(solver_atol))


def test_injection_sweep_row_targets(solver_atol):
    from gpusim2grid import InjectionSweepGPU
    grid, _ = _pinned_grid(1.10, at_min=True)
    load_p, load_q, gen_p = _rows(grid, 2)
    gen_v = np.tile([1.02, 0.], (2, 1))
    gen_v[:, PINNED] = [1.10, 0.85]
    sw = InjectionSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.physical_violation_tol_vm_pu = 0.
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=2)
    got = [_release(r) for r in sw.get_physical_violations()]
    ref = _release(grid.get_physical_violations(True, 0., 0.))
    _assert_same(ref, got[0], _kv_atol(solver_atol))
    assert got[1] == []


# ------------------------------------------------------ an SVC frozen at a limit
SVC_EL = 7   # ViolationElementType.SVC


def _frozen_svc_grid(target_vm=1.0, flagged=True):
    """The feeder with gen 0 alone, and a fixed-Q SVC on bus 2 frozen at the absorbing end
    of its range at `target_vm` -- lightsim2grid's own test_svc_can_be_pv grid, solved."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    b_min, b_max = -0.05, 0.5
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, VN_KV), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators_full(np.array([0.]), np.array([1.02]), np.array([0.]), [True],
                              np.array([-1e3]), np.array([1e3]), np.array([0]))
    grid.add_gen_slackbus(0, 1.)
    grid.init_svcs([2], np.array([target_vm]), np.array([b_min * target_vm ** 2 * 100.]),
                   np.array([0.]), np.array([b_min]), np.array([b_max]),
                   np.array([2], dtype=np.int32), np.array([2], dtype=np.int32))
    if flagged:
        grid.set_svc_can_be_pv(np.array([True]))
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    return grid, V


def _svc_release(viols):
    return [(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
            for v in viols
            if int(v.element_type) == SVC_EL and int(v.violation_type) in (LOW_VM, HIGH_VM)]


@pytest.mark.skipif(not hasattr(__import__("lightsim2grid.lightsim2grid_cpp", fromlist=["LSGrid"]).LSGrid,
                                "set_svc_can_be_pv"),
                    reason="needs a lightsim2grid with LSGrid.set_svc_can_be_pv")
def test_frozen_svc_release_matches_single_solve(solver_atol):
    grid, V = _frozen_svc_grid(1.0)
    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    assert list(plan.el_type) == [SVC_EL] and list(plan.standby) == [0] and list(plan.at_min) == [1]
    ref = _svc_release(grid.get_physical_violations(True, 0., 0.))
    assert len(ref) == 1 and ref[0][1] == LOW_VM
    gpu = _gpu_ca(grid, [2])
    _assert_same(ref, _svc_release(gpu.get_physical_violations_n()), _kv_atol(solver_atol))
    assert _release(gpu.get_physical_violations_n()) == []   # not reported on a generator
    # not flagged, or on the other side of its target: nothing
    grid, _ = _frozen_svc_grid(1.0, flagged=False)
    assert _svc_release(_gpu_ca(grid, [2]).get_physical_violations_n()) == []
    grid, _ = _frozen_svc_grid(0.5)
    assert _svc_release(_gpu_ca(grid, [2]).get_physical_violations_n()) == []


# ------------------------------------------- a VSC converter station frozen at a limit
HVDC_EL = 4   # ViolationElementType.HVDC


def _has_hvdc_flag():
    try:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
        import pypowsybl  # noqa: F401
    except ImportError:
        return False
    return hasattr(LSGrid, "set_hvdc_can_be_pv")


@pytest.mark.skipif(not _has_hvdc_flag(), reason="needs pypowsybl and a lightsim2grid with LSGrid.set_hvdc_can_be_pv")
def test_frozen_vsc_station_release_matches_single_solve(solver_atol):
    """lightsim2grid's own test_hvdc_can_be_pv case: VSC2 (side 2 of HVDC1 of the four
    substations network) frozen by the bake at its max_q, its target then lowered below the
    voltage it leaves: released, reported on the HVDC line with side 2."""
    import pypowsybl as pp
    import pypowsybl.loadflow as lf
    from lightsim2grid.network import bake_outer_loops, init_from_pypowsybl
    n = pp.network.create_four_substations_node_breaker_network()
    n.update_vsc_converter_stations(id="VSC2", target_v=412., max_q=130.)
    n.update_vsc_converter_stations(id="VSC2", voltage_regulator_on=True)
    lf.run_ac(n)
    pinned = bake_outer_loops(n)
    n.update_vsc_converter_stations(id="VSC2", target_v=409.)
    grid = init_from_pypowsybl(n, sort_index=False, buses_for_sub=False, can_be_pv=pinned)
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10)
    assert V.shape[0] > 0

    n_bus = grid.get_Ybus_solver().shape[0]
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, n_bus, 1e-4)
    hv = [k for k in range(plan.n_entries) if int(plan.el_type[k]) == HVDC_EL]
    assert len(hv) == 1 and int(plan.side[hv[0]]) == 2 and int(plan.gen_id[hv[0]]) == 0

    def rel(viols):
        return [(int(v.element_id), int(v.side), int(v.violation_type), float(v.value), float(v.limit))
                for v in viols if int(v.element_type) == HVDC_EL and int(v.violation_type) in (LOW_VM, HIGH_VM)]
    ref = rel(grid.get_physical_violations(True, 0., 0.))
    assert [x[:3] for x in ref] == [(0, 2, HIGH_VM)]
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = 0.
    ca.add_contingencies_by_branch_id([[0]])
    ca.compute(batch_size=8)
    got = rel(ca.get_physical_violations_n())
    assert [x[:3] for x in got] == [x[:3] for x in ref]
    np.testing.assert_allclose(got[0][3], ref[0][3], atol=10. * 400. * solver_atol)
    np.testing.assert_allclose(got[0][4], ref[0][4], rtol=solver_atol)
