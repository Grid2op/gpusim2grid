# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ContingencyAnalysisGPU(reactive_limits_outer_loop=True): one pass of
OpenLoadFlow's ReactiveLimits outer loop on the contingencies the physical
checks flag (see ``contingency_analysis/_reactive_limits.py``).

The reference is lightsim2grid itself: for every re-solved contingency, the
same grid rebuilt with the switches applied by hand -- a switched bus'
regulating generators PQ at the limit they were switched at, a released
generator PV at its target -- and solved one-off with the branch tripped. Its
voltages must be the second pass', and its own physical checks (with the
switched generators flagged ``can_be_pv``, so that lightsim2grid reports the
ones that would switch back) the second pass' records.

The grid: an 8-bus meshed feeder (138 kV), the slack on bus 0, two PV machines
with a narrow reactive range (bus 2: [30, 56] MVAr, bus 7: [15, 30] MVAr) and a
machine frozen PQ at its min_q on bus 4 (flagged ``can_be_pv``, target 1.005
pu) that tripping line 1-4 releases. Its N state is within every limit.
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

VN_KV = 138.
LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4), (2, 6), (6, 7), (7, 3), (0, 6), (5, 7)]
LOADS = [(3, 60., 25.), (4, 50., 20.), (5, 40., 15.), (6, 30., 10.), (7, 45., 18.)]
# (bus, p_mw, vm_pu, q_mvar, voltage_regulator_on, min_q, max_q)
GENS = [
    (0, 0., 1.03, 0., True, -500., 500.),   # the slack
    (2, 60., 1.04, 0., True, 30., 56.),     # PV, narrow range
    (4, 20., 1.005, -8., False, -8., 30.),  # PQ frozen at min_q, flagged can_be_pv
    (7, 30., 1.02, 0., True, 15., 30.),     # PV, narrow range
]
CAN_BE_PV = (False, False, True, False)
TOL_MVA, TOL_VM = 1e-3, 1e-4
LOW_Q, HIGH_Q, REL_LOW, REL_HIGH = 5, 6, 9, 10
# what the first pass reports on this grid (checked below): the contingencies
# the loop re-solves, and the one releasing the frozen machine
SWITCHED = {2, 5, 6, 9, 11}
RELEASING = 6


def _build(gens=GENS, can_be_pv=CAN_BE_PV):
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    n_bus = 8
    grid.init_bus(n_bus, 1, np.full(n_bus, VN_KV), 0, 0)
    grid.init_powerlines(np.full(len(LINES), 0.01), np.full(len(LINES), 0.08),
                         np.full(len(LINES), 0.02j),
                         np.array([a for a, _ in LINES]), np.array([b for _, b in LINES]))
    grid.init_loads(np.array([l[1] for l in LOADS]), np.array([l[2] for l in LOADS]),
                    np.array([l[0] for l in LOADS]))
    grid.init_generators_full(np.array([g[1] for g in gens]), np.array([g[2] for g in gens]),
                              np.array([g[3] for g in gens]), [g[4] for g in gens],
                              np.array([g[5] for g in gens]), np.array([g[6] for g in gens]),
                              np.array([g[0] for g in gens]))
    grid.set_gen_can_be_pv(np.array(can_be_pv))
    grid.add_gen_slackbus(0, 1.)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(n_bus, 1.0 + 0j), 30, 1e-12)
    assert V.shape[0] > 0
    return grid, V


def _ca(grid, outer_loop, batch_size=16, min_last_chunk=250):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True,
                                reactive_limits_outer_loop=outer_loop,
                                outer_loop_min_last_chunk=min_last_chunk)
    ca.physical_violation_tol_mva = TOL_MVA
    ca.physical_violation_tol_vm_pu = TOL_VM
    ca.add_contingencies_by_branch_id([[i] for i in range(len(LINES))])
    ca.compute(batch_size=batch_size)
    return ca


def _switched_grid(switches, gens=GENS):
    """The grid with one contingency's switches applied by hand: a switched
    bus' regulating generators PQ at the limit it was switched at (flagged
    can_be_pv, so that lightsim2grid reports their release), a released
    generator PV at its target (no longer flagged)."""
    gens = [list(g) for g in gens]
    can_be_pv = list(CAN_BE_PV)
    for b, (_q, at_min) in switches["to_pq"].items():
        for k, g in enumerate(gens):
            if g[0] == b and g[4]:
                g[4] = False
                g[3] = g[5] if at_min else g[6]
                can_be_pv[k] = True
    for _b, (gen_ids, vm) in switches["to_pv"].items():
        for k in gen_ids:
            gens[k][4] = True
            gens[k][2] = vm
            can_be_pv[k] = False
    return _build([tuple(g) for g in gens], tuple(can_be_pv))[0]


def _records(viols):
    return sorted((int(v.violation_type), int(v.element_id), round(float(v.value), 6))
                  for v in viols)


@pytest.fixture(scope="module")
def runs():
    grid, V0 = _build()
    return grid, V0, _ca(grid, False), _ca(grid, True)


def test_first_pass_is_the_scenario(runs):
    """The grid does what the module docstring says: the N state is clean, the
    contingencies of SWITCHED report a switch, line 1-4 the release."""
    grid, V0, ca_off, _ = runs
    assert ca_off.get_physical_violations_n() == []
    phys = ca_off.get_physical_violations()
    flagged = {i for i, p in enumerate(phys)
               if any(int(v.violation_type) in (LOW_Q, HIGH_Q, REL_LOW, REL_HIGH) for v in p)}
    assert flagged == SWITCHED
    assert any(int(v.violation_type) == REL_LOW for v in phys[RELEASING])


def test_status_and_switches(runs):
    from gpusim2grid import ReactiveLimitsStatus as S
    _, _, _, ca = runs
    st = ca.get_outer_loop_status()
    assert {i for i, s in enumerate(st) if s == S.RECOMPUTED} == SWITCHED
    assert all(st[i] == S.NO_SWITCH for i in range(len(LINES)) if i not in SWITCHED)
    sw = ca.get_outer_loop_switches()
    assert sw[RELEASING]["to_pq"] == {} and list(sw[RELEASING]["to_pv"]) == [4]
    gen_ids, vm = sw[RELEASING]["to_pv"][4]
    assert gen_ids == (2,) and vm == pytest.approx(1.005)
    # contingency 2: bus 2 below its min_q, bus 7 above its max_q
    assert sw[2]["to_pq"] == {2: (30., True), 7: (30., False)}
    assert ca.outer_loop_info["n_rows"] == len(SWITCHED)


def test_second_pass_is_lightsim2grids_switched_solve(runs, solver_atol):
    """Voltages and physical records of every re-solved contingency are a
    one-off lightsim2grid solve of the grid with the switches applied."""
    grid, V0, _, ca = runs
    n = len(LINES)
    V = ca.V_results.to_numpy().reshape(n, -1)
    phys = ca.get_physical_violations()
    sw = ca.get_outer_loop_switches()
    for i in sorted(SWITCHED):
        ref = _switched_grid(sw[i])
        ref.deactivate_powerline(i)
        V_ref = ref.ac_pf(V0.copy(), 30, 1e-12)
        assert V_ref.shape[0] > 0
        np.testing.assert_allclose(V[i], V_ref, atol=solver_atol, rtol=0)
        ls = ref.get_physical_violations(True, TOL_MVA, TOL_VM)
        assert _records(phys[i]) == _records(ls), f"contingency {i}"


def test_switch_back_is_reported(runs):
    """Contingency 2: switching bus 7 to PQ at max_q pulls bus 2 (switched at
    min_q) below its target -- the machine there would regulate again -- and
    the frozen machine of bus 4 is released too: both are reported."""
    _, _, _, ca = runs
    rec = _records(ca.get_physical_violations()[2])
    assert [(t, g) for t, g, _ in rec] == [(REL_LOW, 1), (REL_LOW, 2)]


def test_untouched_rows_and_residuals(runs, solver_atol, residual_atol):
    """A NO_SWITCH contingency keeps the first pass (two sessions agree to
    rounding, not bit for bit); every re-solved one converged in the second
    pass (merged residuals)."""
    _, _, ca_off, ca = runs
    n = len(LINES)
    V_off = ca_off.V_results.to_numpy().reshape(n, -1)
    V_on = ca.V_results.to_numpy().reshape(n, -1)
    keep = [i for i in range(n) if i not in SWITCHED]
    np.testing.assert_allclose(V_on[keep], V_off[keep], atol=solver_atol, rtol=0)
    assert np.all(ca.last_residuals() <= residual_atol)
    phys_off, phys_on = ca_off.get_physical_violations(), ca.get_physical_violations()
    for i in keep:
        assert _records(phys_on[i]) == _records(phys_off[i])


def test_released_bus_is_checked_for_its_reactive_range(solver_atol):
    """A released machine holds its bus again: the second pass checks the
    reactive power it produces against its own range (here max_q 0.2 MVAr,
    below the ~0.46 MVAr it must produce once line 1-4 is out)."""
    gens = [list(g) for g in GENS]
    gens[2][6] = 0.2
    gens = [tuple(g) for g in gens]
    grid, V0 = _build(gens)
    ca = _ca(grid, True)
    sw = ca.get_outer_loop_switches()[RELEASING]
    assert list(sw["to_pv"]) == [4]
    rec = _records(ca.get_physical_violations()[RELEASING])
    assert [(t, b) for t, b, _ in rec] == [(HIGH_Q, 4)]
    ref = _switched_grid(sw, gens)
    ref.deactivate_powerline(RELEASING)
    assert ref.ac_pf(V0.copy(), 30, 1e-12).shape[0] > 0
    assert rec == _records(ref.get_physical_violations(True, TOL_MVA, TOL_VM))


def test_batch_rule():
    """A last, partial chunk runs when it is the only one or large enough."""
    from gpusim2grid.contingency_analysis._reactive_limits import batch_rule
    rows = list(range(1002))
    run, left = batch_rule(rows, 1000, 250)
    assert run == rows[:1000] and left == rows[1000:]
    run, left = batch_rule(list(range(1250)), 1000, 250)
    assert len(run) == 1250 and left == []
    run, left = batch_rule(list(range(999)), 1000, 250)
    assert len(run) == 999 and left == []
    run, left = batch_rule(list(range(7)), 1000, 250)
    assert len(run) == 7 and left == []


def test_left_out_rows_keep_the_first_pass(runs, solver_atol):
    """batch_size 2, min_last_chunk 2: of the five re-solvable contingencies,
    the fifth would be alone in a last chunk -- left out, first pass kept."""
    from gpusim2grid import ReactiveLimitsStatus as S
    grid, _, ca_off, _ = runs
    ca = _ca(grid, True, batch_size=2, min_last_chunk=2)
    st = ca.get_outer_loop_status()
    last = max(SWITCHED)
    assert st[last] == S.LEFT_OUT
    assert {i for i, s in enumerate(st) if s == S.RECOMPUTED} == SWITCHED - {last}
    n = len(LINES)
    np.testing.assert_allclose(ca.V_results.to_numpy().reshape(n, -1)[last],
                               ca_off.V_results.to_numpy().reshape(n, -1)[last],
                               atol=solver_atol, rtol=0)
    assert _records(ca.get_physical_violations()[last]) == \
        _records(ca_off.get_physical_violations()[last])


def _remote_feeder(remote_on=False):
    """test_gen_pv_release_violations.py's remote feeder (the machine on leaf bus
    4 regulates bus 1, released in the N state already) with line 0-1 doubled,
    so that tripping one of them leaves the grid connected. ``remote_on``: the
    machine regulating (the reference of its release)."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(5, 1, np.full(5, VN_KV), 0, 0)
    grid.init_powerlines(np.full(5, 0.01), np.full(5, 0.1), np.zeros(5, dtype=complex),
                         np.array([0, 1, 2, 1, 0]), np.array([1, 2, 3, 4, 1]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators_full(np.array([0., 10.]), np.array([1.02, 1.10]), np.array([0., -5.]),
                              [True, remote_on], np.array([-1e3, -5.]), np.array([1e3, 20.]),
                              np.array([0, 4]))
    grid.set_gen_regulated_bus(1, 1)
    grid.set_gen_can_be_pv(np.array([False, not remote_on]))
    grid.add_gen_slackbus(0, 1.)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.full(5, 1.0 + 0j), 30, 1e-11).shape[0] > 0
    return grid


def test_remote_release(solver_atol):
    """A frozen machine regulating a REMOTE bus is released into its
    VoltageControl group (through lightsim2grid's held controllers, when it has
    them): lightsim2grid's solve with the machine regulating; without them the
    contingency keeps its first pass (UNSUPPORTED)."""
    from gpusim2grid import ContingencyAnalysisGPU, ReactiveLimitsStatus as S
    grid = _remote_feeder()
    ca_off, ca_on = [ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True,
                                            reactive_limits_outer_loop=loop) for loop in (False, True)]
    for ca in (ca_off, ca_on):
        ca.physical_violation_tol_mva = TOL_MVA
        ca.physical_violation_tol_vm_pu = TOL_VM
        ca.add_contingencies_by_branch_id([[0], [4]])
        ca.compute(batch_size=4)
    st, sw = ca_on.get_outer_loop_status(), ca_on.get_outer_loop_switches()
    phys = ca_off.get_physical_violations()
    released = [i for i, p in enumerate(phys) if any(int(v.violation_type) in (REL_LOW, REL_HIGH) for v in p)]
    assert released, "the scenario must release the remote machine somewhere"
    V = ca_on.V_results.to_numpy().reshape(2, -1)
    for i in released:
        if not getattr(_cpp, "have_ls2g_hold_frozen", False):
            assert st[i] == S.UNSUPPORTED
            assert _records(ca_on.get_physical_violations()[i]) == _records(phys[i])
            continue
        assert st[i] == S.RECOMPUTED and list(sw[i]["vc_release"]) == [1]
        ref = _remote_feeder(remote_on=True)
        ref.deactivate_powerline([0, 4][i])
        V_ref = ref.ac_pf(grid.get_V().copy(), 30, 1e-12)
        assert V_ref.shape[0] > 0
        np.testing.assert_allclose(V[i], V_ref, atol=solver_atol, rtol=0)
        assert _records(ca_on.get_physical_violations()[i]) == \
            _records(ref.get_physical_violations(True, TOL_MVA, TOL_VM))


def test_warm_start(runs, solver_atol):
    """The second pass starts each contingency from its first-pass voltages
    (outer_loop_warm_start, the default; ``runs`` uses it): 2 Newton
    iterations then land on the 10-iteration cold result, where a cold start
    with 2 leaves some contingencies DIVERGED."""
    from gpusim2grid import ContingencyAnalysisGPU, ReactiveLimitsStatus as S
    grid, _, _, _ = runs
    n = len(LINES)
    out = {}
    for warm, nb in ((False, 10), (True, 2), (False, 2)):
        ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True,
                                    reactive_limits_outer_loop=True, outer_loop_warm_start=warm,
                                    outer_loop_nb_iter=nb)
        ca.physical_violation_tol_mva = TOL_MVA
        ca.physical_violation_tol_vm_pu = TOL_VM
        ca.add_contingencies_by_branch_id([[i] for i in range(n)])
        ca.compute(batch_size=16)
        out[(warm, nb)] = (ca.get_outer_loop_status(), ca.V_results.to_numpy().reshape(n, -1),
                           ca.get_physical_violations())
    rows = sorted(SWITCHED)
    st_ref, V_ref, rec_ref = out[(False, 10)]
    st, V, rec = out[(True, 2)]
    assert all(st[i] == S.RECOMPUTED for i in rows)
    np.testing.assert_allclose(V[rows], V_ref[rows], atol=solver_atol, rtol=0)
    for i in rows:   # values in kV / MVAr: the solve's accuracy, scaled
        a = sorted(rec[i], key=lambda v: (int(v.violation_type), int(v.element_id)))
        b = sorted(rec_ref[i], key=lambda v: (int(v.violation_type), int(v.element_id)))
        assert [(int(v.violation_type), int(v.element_id)) for v in a] == \
            [(int(v.violation_type), int(v.element_id)) for v in b]
        np.testing.assert_allclose([v.value for v in a], [v.value for v in b], atol=VN_KV * solver_atol, rtol=0)
    assert any(out[(False, 2)][0][i] == S.DIVERGED for i in rows)


def test_session_v_init_rows(solver_atol):
    """ScenarioSweepSession.set_v_init_from_ptr: row i starts from row
    src_rows[i] of another session's voltages, through the active-slot order
    (a skipped row in between) and with a masked bus (NaN there) started from
    the base case. One iteration from a converged row stays on it."""
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = _build()
    loads = grid.get_loads()
    gens = grid.get_generators()

    def sweep(topo, nb_iter):
        ss = ScenarioSweepGPU(grid, nb_iter=nb_iter, handle_disconnected_grid=True)
        ss.set_topology(topo)
        k = len(topo)
        ss.set_injections_from_elements(np.tile([l.target_p_mw for l in loads], (k, 1)),
                                        np.tile([l.target_q_mvar for l in loads], (k, 1)),
                                        np.tile([g.target_p_mw for g in gens], (k, 1)))
        return ss

    # row 2 islands bus 6 (every line touching it)
    topo = [[1], [3], [7, 8, 10], [5], [11]]
    a = sweep(topo, 10)
    a.compute(batch_size=8)
    V_a = a.V_results.to_numpy().reshape(len(topo), -1)
    assert np.isnan(V_a[2, 6]) and np.isfinite(np.delete(V_a[2], 6)).all()

    perm = [3, 2, 0, 4, 1]
    b = sweep([topo[i] for i in perm], 1)
    b.solver._s.set_skipped_rows(np.array([False, False, False, True, False]))
    b.solver._s.set_v_init_from_ptr(a.solver._s.v_results_ptr(), perm)
    assert b.solver._s.has_v_init
    b.compute(batch_size=8)
    V_b = b.V_results.to_numpy().reshape(len(topo), -1)
    for i, src in enumerate(perm):
        if i == 3:
            assert np.isnan(V_b[i]).all()   # skipped
            continue
        np.testing.assert_allclose(V_b[i], V_a[src], atol=solver_atol, rtol=0, equal_nan=True)
    # ... which a single iteration from the base case does not reach
    b.solver._s.clear_v_init()
    b.compute(batch_size=8)
    V_c = b.V_results.to_numpy().reshape(len(topo), -1)
    assert np.nanmax(np.abs(V_c[0] - V_a[3])) > 100 * solver_atol


def test_needs_the_physical_checks():
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _ = _build()
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, reactive_limits_outer_loop=True)
    ca.add_contingencies_by_branch_id([[0]])
    with pytest.raises(RuntimeError, match="compute_physical_violations"):
        ca.compute(batch_size=4)


def test_off_by_default():
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _ = _build()
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    assert ca.reactive_limits_outer_loop is False
    ca.add_contingencies_by_branch_id([[0]])
    ca.compute(batch_size=4)
    with pytest.raises(RuntimeError, match="reactive_limits_outer_loop"):
        ca.get_outer_loop_status()
