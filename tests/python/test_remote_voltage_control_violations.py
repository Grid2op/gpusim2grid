"""compute_physical_violations: the remote voltage control check (lightsim2grid's
``RemoteVoltageControlCheck.hpp``).

A generator regulating a REMOTE bus whose own bus leaves the realistic range the
caller set (``LSGrid.set_remote_voltage_control_vm_range``, what
``init_from_pypowsybl`` sets to OpenLoadFlow's defaults): OpenLoadFlow's robust
remote voltage control would switch it to PQ. Reported as
``LOW_VOLTAGE_REMOTE_CONTROL`` / ``HIGH_VOLTAGE_REMOTE_CONTROL`` on the
generator, value its own bus' voltage and limit the bound, both in kV. Nothing
is switched.

gpusim2grid routes it through the PQ -> PV release plan (GENERATOR entries with
standby = 2). The reference is lightsim2grid itself, on IEEE 14 with B6-G
regulating the bus of a load further away.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid import _gpusim2grid as _cpp

pp = pytest.importorskip("pypowsybl")

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_gen_pv_release", False),
                       reason="needs the bridge built against a lightsim2grid with can_be_pv"),
]


def _ls_has_remote_check():
    try:
        from lightsim2grid.lightsim2grid_cpp import LSGrid
    except ImportError:
        return False
    return hasattr(LSGrid, "set_remote_voltage_control_vm_range")


pytestmark.append(pytest.mark.skipif(not _ls_has_remote_check(),
                                     reason="needs a lightsim2grid with set_remote_voltage_control_vm_range"))

GEN, LOW_RC, HIGH_RC = 5, 13, 14   # ViolationElementType.GENERATOR, LOW/HIGH_VOLTAGE_REMOTE_CONTROL
MAX_IT, TOL = 30, 1e-11
VN_KV = 12.   # the controller's own bus
# OLF's former minRealisticVoltage / maxRealisticVoltage (0.8 / 1.2 pu) with its 1.02
# margin: init_from_pypowsybl now reads the installed OLF's defaults (0.5 / 2.0 in recent
# ones), which no target below would leave. Set explicitly so the test does not depend on it.
VM_RANGE = (0.8 * 1.02, 1.2 / 1.02)


def _grid(target_v_kv):
    """IEEE 14, B6-G (12 kV) regulating the 12 kV bus of a load further away, with reactive
    limits wide enough that only its own voltage can stop it."""
    from lightsim2grid.network import init_from_pypowsybl
    n = pp.network.create_ieee14()
    loads = n.get_loads(attributes=["voltage_level_id"])
    load = loads.index[loads["voltage_level_id"] == "VL12"][0]
    n.update_generators(id="B6-G", regulated_element_id=load, target_v=target_v_kv,
                        min_q=-9999., max_q=9999.)
    grid = init_from_pypowsybl(n, gen_slack_id="B1-G", sort_index=True)
    grid.set_remote_voltage_control_vm_range(*VM_RANGE)
    V = grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0
    return grid


def _remote(viols):
    """[(gen_id, type, value, limit)] of the remote control records only."""
    return [(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
            for v in viols
            if int(v.element_type) == GEN and int(v.violation_type) in (LOW_RC, HIGH_RC)]


def _gpu_ca(grid, branches):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_tol_vm_pu = 0.
    ca.add_contingencies_by_branch_id([[int(b)] for b in branches])
    ca.compute(batch_size=8)
    return ca


def _one_off(grid, branch):
    """The remote control records of lightsim2grid's own single solve with ``branch``
    (lines then trafos) disconnected."""
    g = grid.copy()
    n_lines = len(g.get_lines())
    if branch < n_lines:
        g.deactivate_powerline(branch)
    else:
        g.deactivate_trafo(branch - n_lines)
    V = g.ac_pf(np.asarray(grid.get_V()).copy(), MAX_IT, TOL)
    assert V.shape[0] > 0
    return _remote(g.get_physical_violations(True, 0., 0.))


def _assert_same(ref, got, atol_kv):
    assert [x[:2] for x in ref] == [x[:2] for x in got], f"{ref} vs {got}"
    for a, b in zip(ref, got):
        np.testing.assert_allclose(b[2], a[2], atol=atol_kv)
        np.testing.assert_allclose(b[3], a[3], atol=atol_kv)


def test_plan_carries_the_remote_controller():
    grid = _grid(14.2)
    plan = _cpp._extract_gen_pv_release_plan_from_lsgrid(grid, grid.total_bus(), 0.)
    standby = np.asarray(plan.standby)
    remote = np.flatnonzero(standby == 2)
    assert remote.size == 2   # its low and its high bound
    assert set(np.asarray(plan.at_min)[remote]) == {0, 1}
    assert set(np.asarray(plan.el_type)[remote]) == {GEN}


@pytest.mark.parametrize("target_v_kv, nb", [(13.85, 0), (14.2, 1)])
def test_matches_lightsim2grid(solver_atol, target_v_kv, nb):
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    grid = _grid(target_v_kv)
    ref_n = _remote(grid.get_physical_violations(True, 0., 0.))
    assert len(ref_n) == nb
    ls = ContingencyAnalysisCPP(grid)
    ls.compute_physical_violations = True
    ls.physical_violation_tol_mva = 0.
    ls.physical_violation_tol_vm_pu = 0.
    branches = [0, 5]
    for b in branches:
        ls.add_n1(b)
    ls.compute(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    gpu = _gpu_ca(grid, branches)
    atol = 10. * VN_KV * solver_atol
    _assert_same(ref_n, _remote(gpu.get_physical_violations_n()), atol)
    # Every row against lightsim2grid's single solve of that contingency, and against
    # its ContingencyAnalysisCPP where that converged: lightsim2grid master's batch
    # leaves branch 5 unconverged after one iteration with B6-G regulating remotely
    # (the grid stays connected and the single solve converges).
    ls_conv = np.asarray(ls.converged_mask(), dtype=bool)
    for row, b in enumerate(branches):
        got = _remote(gpu.get_physical_violations()[row])
        _assert_same(_one_off(grid, b), got, atol)
        if ls_conv[row]:
            _assert_same(_remote(ls.get_physical_violations()[row]), got, atol)


def test_range_off_reports_nothing():
    grid = _grid(14.2)
    grid.set_remote_voltage_control_vm_range(np.nan, np.nan)
    assert _remote(_gpu_ca(grid, [0]).get_physical_violations_n()) == []
