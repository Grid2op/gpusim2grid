# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""lightsim2grid's held controllers (``LSGrid.set_hold_frozen_regulators``): a
generator an outer loop froze at a reactive limit (``can_be_pv``, regulation
off) that would regulate a REMOTE bus keeps its seat in that bus' voltage-control
group, pinned at its frozen output.

- Every gpusim2grid solve of such a grid is the solve without the option (same
  voltages, same physical records): the base solve and every batch row pin the
  held controllers.
- A ``ScenarioSweepGPU`` row may release one (``set_vc_controller_releases``):
  it then regulates with its group again -- lightsim2grid's solve of the grid
  where that generator regulates.
- ``reactive_limits_outer_loop`` uses that for the release of a frozen remote
  regulator (``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q``): its second
  pass is built from a held copy of the grid.

The grid is lightsim2grid's own fixture for the option: a 138 kV ring 0-1-2-3
with leaves 4, 5, 6 and 7 behind short lines; gen 0 the slack; gen 1 on leaf 4,
frozen, would regulate bus 1 (a group of held machines only); gen 2 on leaf 5
regulates bus 2 (active) and gen 3 on leaf 6, frozen, would regulate it at the
same set-point (a held machine in an active group); gen 4 on leaf 7, frozen,
would regulate bus 2 at another set-point (left out).
"""

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid import _gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_hold_frozen", False),
                       reason="needs the bridge built against a lightsim2grid with "
                              "set_hold_frozen_regulators"),
]

LINES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (2, 5), (2, 6), (2, 7)]
STEP_UP = {5, 6, 7, 8}
# (bus, p, vm, q, regulating, min_q, max_q, regulated bus)
GENS = [(0, 0., 1.03, 0., True, -500., 500., 0),
        (4, 30., 1.02, 25., False, -20., 25., 1),
        (5, 20., 1.03, 0., True, -40., 40., 2),
        (6, 15., 1.03, -10., False, -10., 30., 2),
        (7, 10., 1.05, 5., False, -5., 5., 2)]
CAN_BE_PV = [False, True, False, True, True]
TOL_MVA, TOL_VM = 1e-3, 1e-4


def _grid(hold=False, released=(), extra_gens=()):
    """The fixture, solved. ``released``: generators turned back into
    regulators (the reference of a release); ``extra_gens``: more generators."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    gens = [list(g) for g in GENS] + [list(g) for g in extra_gens]
    can_be_pv = list(CAN_BE_PV) + [True] * len(extra_gens)
    for k in released:
        gens[k][4], can_be_pv[k] = True, False
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(8, 1, np.full(8, 138.), 0, 0)
    x = np.array([0.03 if k in STEP_UP else 0.08 for k in range(len(LINES))])
    grid.init_powerlines(np.full(len(LINES), 0.01), x, np.full(len(LINES), 0.02j),
                         np.array([a for a, _ in LINES]), np.array([b for _, b in LINES]))
    grid.init_loads(np.array([60., 50., 40.]), np.array([25., 20., 15.]), np.array([1, 2, 3]))
    grid.init_generators_full(np.array([g[1] for g in gens]), np.array([g[2] for g in gens]),
                              np.array([g[3] for g in gens]), [g[4] for g in gens],
                              np.array([g[5] for g in gens]), np.array([g[6] for g in gens]),
                              np.array([g[0] for g in gens]))
    for k, g in enumerate(gens):
        if g[7] != g[0]:
            grid.set_gen_regulated_bus(k, g[7])
    grid.set_gen_can_be_pv(np.array(can_be_pv))
    grid.add_gen_slackbus(0, 1.)
    grid.set_hold_frozen_regulators(hold)
    grid.tell_solver_need_reset()
    V = grid.ac_pf(np.full(8, 1.0 + 0j), 30, 1e-12)
    assert V.shape[0] > 0
    return grid, V


def _records(viols):
    return sorted((int(v.violation_type), int(v.element_id), round(float(v.value), 6))
                  for v in viols)


def _ls_contingency(grid, branch):
    """lightsim2grid's own solve of one contingency (islands masked)."""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(grid)
    ca.handle_disconnected_grid = True
    ca.compute_physical_violations = True
    ca.physical_violation_tol_mva = TOL_MVA
    ca.physical_violation_tol_vm_pu = TOL_VM
    if branch is None:
        ca.add_multiple_n1([])
        ca.add_nk([])
    else:
        ca.add_n1(branch)
    ca.compute(np.full(grid.total_bus(), 1.0 + 0j), 30, 1e-12)
    V = np.asarray(ca.get_voltages())[0].copy()
    V[V == 0.] = np.nan   # a masked bus: 0 there, NaN here
    return V, ca.get_physical_violations()[0]


def _ctrl_of_gen(sess):
    elem = np.asarray(sess.vc_ctrl_elem_id)
    kind = np.asarray(sess.vc_ctrl_kind)
    return {int(e): j for j, (e, k) in enumerate(zip(elem, kind)) if k == 0}


def test_held_controllers_are_read():
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = _grid(hold=True)
    sess = ScenarioSweepGPU(grid, nb_iter=10).solver._s
    held = np.asarray(sess.vc_ctrl_held)
    elem = np.asarray(sess.vc_ctrl_elem_id)
    assert sorted(elem[held == 1].tolist()) == [1, 3]
    ctrl = _ctrl_of_gen(sess)
    q_held = np.asarray(sess.vc_ctrl_q_held)
    assert q_held[ctrl[1]] == pytest.approx(GENS[1][3] / 100.)
    assert q_held[ctrl[3]] == pytest.approx(GENS[3][3] / 100.)


def test_same_solution_with_held_controllers(solver_atol):
    """AcPfGPU (from a flat start), ContingencyAnalysisGPU (islands masked,
    physical checks) and ScenarioSweepGPU give the same answer on the held grid."""
    from gpusim2grid import AcPfGPU, ContingencyAnalysisGPU, ScenarioSweepGPU
    (g_off, V_off), (g_on, V_on) = _grid(), _grid(hold=True)
    ac = AcPfGPU(g_on, max_iter=30, tol=1e-11, init_from_n_powerflow=False)
    np.testing.assert_allclose(ac.solve(), V_off, atol=solver_atol, rtol=0)

    n = len(LINES)
    res = {}
    for name, grid in (("off", g_off), ("on", g_on)):
        ca = ContingencyAnalysisGPU(grid, nb_iter=10, handle_disconnected_grid=True,
                                    compute_physical_violations=True)
        ca.physical_violation_tol_mva, ca.physical_violation_tol_vm_pu = TOL_MVA, TOL_VM
        ca.add_contingencies_by_branch_id([[i] for i in range(n)])
        ca.compute(batch_size=16)
        ss = ScenarioSweepGPU(grid, nb_iter=10, handle_disconnected_grid=True)
        ss.set_topology([[i] for i in range(n)])
        el = [np.tile([float(x.target_p_mw) for x in grid.get_loads()], (n, 1)),
              np.tile([float(x.target_q_mvar) for x in grid.get_loads()], (n, 1)),
              np.tile([float(x.target_p_mw) for x in grid.get_generators()], (n, 1))]
        ss.set_injections_from_elements(*el)
        ss.compute(batch_size=16)
        res[name] = (ca.V_results.to_numpy().reshape(n, -1), [_records(r) for r in ca.get_physical_violations()],
                     ss.V_results.to_numpy().reshape(n, -1))
    for k in (0, 2):
        np.testing.assert_allclose(res["on"][k], res["off"][k], atol=solver_atol, rtol=0, equal_nan=True)
    assert res["on"][1] == res["off"][1]


@pytest.mark.parametrize("branch, released", [
    (None, (1,)),       # the first (and only) controller of a held group
    (0, (1,)),
    (None, (3,)),       # a held machine joining an active group
    (6, (3,)),          # ... whose active controller is islanded: it holds the bus alone
    (None, (1, 3)),
    (2, ()),
])
def test_release_is_lightsim2grids_regulator(branch, released, solver_atol):
    """A row releasing held controllers gives lightsim2grid's solve of the grid
    where those generators regulate."""
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = _grid(hold=True)
    ss = ScenarioSweepGPU(grid, nb_iter=10, handle_disconnected_grid=True)
    sess = ss.solver._s
    ctrl = _ctrl_of_gen(sess)
    rows = [[], [] if branch is None else [branch]]
    ss.set_topology(rows)
    el = [np.tile([float(x.target_p_mw) for x in grid.get_loads()], (2, 1)),
          np.tile([float(x.target_q_mvar) for x in grid.get_loads()], (2, 1)),
          np.tile([float(x.target_p_mw) for x in grid.get_generators()], (2, 1))]
    ss.set_injections_from_elements(*el)
    # row 0: nothing released, the base case; row 1: the release
    sess.set_vc_controller_releases([[], sorted(ctrl[g] for g in released)])
    ss.compute(batch_size=2)
    V = ss.V_results.to_numpy().reshape(2, -1)
    _, V0 = _grid()
    np.testing.assert_allclose(V[0], V0, atol=solver_atol, rtol=0)
    ref, _ = _grid(released=released)
    V_ref, _ = _ls_contingency(ref, branch)
    np.testing.assert_allclose(V[1], V_ref, atol=solver_atol, rtol=0, equal_nan=True)


def test_release_leaving_the_first_held_is_refused():
    """A group of held machines only (gens 1 and 5 regulating bus 1): releasing
    its second one alone would leave the first pinning the voltage row."""
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = _grid(hold=True, extra_gens=[(4, 5., 1.02, 3., False, -5., 3., 1)])
    ss = ScenarioSweepGPU(grid, nb_iter=10)
    sess = ss.solver._s
    ctrl = _ctrl_of_gen(sess)
    first, second = sorted((ctrl[1], ctrl[5]))
    ss.set_injections_from_elements(
        np.array([[float(x.target_p_mw) for x in grid.get_loads()]]),
        np.array([[float(x.target_q_mvar) for x in grid.get_loads()]]),
        np.array([[float(x.target_p_mw) for x in grid.get_generators()]]))
    sess.set_vc_controller_releases([[second]])
    with pytest.raises(RuntimeError, match="either all of them are held"):
        ss.compute(batch_size=1)
    with pytest.raises(RuntimeError, match="not a held controller"):
        sess.set_vc_controller_releases([[0]])


def test_outer_loop_releases_remote_regulators(solver_atol):
    """reactive_limits_outer_loop on the plain grid: every contingency reporting
    the release of a frozen remote regulator is re-solved with it regulating --
    lightsim2grid's voltages and physical records for that grid."""
    from gpusim2grid import ContingencyAnalysisGPU, ReactiveLimitsStatus as S
    grid, _ = _grid()
    n = len(LINES)
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, handle_disconnected_grid=True,
                                compute_physical_violations=True, reactive_limits_outer_loop=True)
    ca.physical_violation_tol_mva, ca.physical_violation_tol_vm_pu = TOL_MVA, TOL_VM
    ca.add_contingencies_by_branch_id([[i] for i in range(n)])
    ca.compute(batch_size=16)
    st, sw = ca.get_outer_loop_status(), ca.get_outer_loop_switches()
    V = ca.V_results.to_numpy().reshape(n, -1)
    phys = ca.get_physical_violations()
    released = {i: tuple(sorted(sw[i]["vc_release"])) for i in range(n)
                if st[i] == S.RECOMPUTED and sw[i]["vc_release"]}
    assert len(released) >= 5 and (1, 3) in released.values()
    for i, gens in released.items():
        assert not sw[i]["to_pq"] and not sw[i]["to_pv"] and not sw[i]["vc_pin"]
        ref, _ = _grid(released=gens)
        V_ref, phys_ref = _ls_contingency(ref, i)
        np.testing.assert_allclose(V[i], V_ref, atol=solver_atol, rtol=0, equal_nan=True)
        assert _records(phys[i]) == _records(phys_ref), f"contingency {i}"
