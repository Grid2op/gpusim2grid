"""Augmented Newton-Raphson features (lightsim2grid parity).

These exercise gpusim2grid solving the SAME augmented system lightsim2grid does
— more than the bare ``[pvpq | pq]`` Jacobian. Phase 2 covers the MultiSlack
extension (distributed slack in the Jacobian). The reference is the solved
``LSGrid`` (itself validated against pypowsybl / OpenLoadFlow upstream).

The augmented path requires the C++ bridge (``have_ls2g_bridge``) since the
NRLedger + J skeleton are read off the solved C++ grid.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="augmented features need the lightsim2grid C++ bridge")


def _solved_multislack_grid(extra_slack_bus=3):
    """case14 with a 2nd ext_grid → genuine distributed slack (N=2), ac-solved."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        pp.create_ext_grid(net, bus=extra_slack_bus, vm_pu=1.0, va_degree=0.0,
                           slack_weight=1.0)
        net.ext_grid.loc[0, "slack_weight"] = 1.0
        pp.runpp(net, distributed_slack=True)
        grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)   # MultiSlackNRSystem
        n_model = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_model, dtype=complex), 1, 1e-6)
        grid.ac_pf(v0.copy(), 30, 1e-10)
    return grid


@requires_gpu
@needs_bridge
def test_acpf_gpu_single_slack_augmented_matches(ieee14_base_case, solver_atol):
    """AcPfGPU via the bridge solves the augmented (slack_absorbed) system and
    matches lightsim2grid even for an ordinary single-slack grid (N=1)."""
    from gpusim2grid import AcPfGPU

    grid = ieee14_base_case["grid"]
    V_ref = grid.get_V_solver()

    ac = AcPfGPU(grid, max_iter=30, tol=1e-10)   # use_bridge auto-detected
    V_gpu = ac.solve()

    # NR_KLU uses MultiSlackNRSystem, so the augmented J carries +1 column
    # (slack_absorbed) over the bare pvpq/pq system even with one slack.
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]
    assert ac.session.dim_J == n_pv + 2 * n_pq + 1

    np.testing.assert_allclose(V_gpu, V_ref, atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_acpf_gpu_distributed_slack_matches(solver_atol):
    """AcPfGPU via the bridge matches lightsim2grid on a genuine multi-slack
    grid (N=2): non-ref slack theta column + slack_absorbed column."""
    from gpusim2grid import AcPfGPU

    grid = _solved_multislack_grid()
    assert grid.get_slack_ids_solver().shape[0] >= 2, "expected >= 2 slacks"

    V_ref = grid.get_V_solver()
    ac = AcPfGPU(grid, max_iter=40, tol=1e-10)
    V_gpu = ac.solve()

    # bare would be n_pv + 2*n_pq; the augmented system adds the slack columns.
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]
    assert ac.session.dim_J > n_pv + 2 * n_pq

    np.testing.assert_allclose(V_gpu, V_ref, atol=10 * solver_atol)
