"""test_diff_augmented.py — finite-difference verification of the
differentiable power-flow adjoint on the AUGMENTED Jacobian.

Covers all four "exotic" lightsim2grid features solved via the NRLedger-driven
bridge: distributed slack (priority), HVDC angle-droop, SVC, and remote
generator voltage control. Reuses the exact grid-building fixtures from
test_augmented_features.py (the same grids already validated for the forward
solve against lightsim2grid).

The finite-difference check is torch.autograd.gradcheck (perturbs each input
element and compares to the analytic adjoint gradient) — same mechanism and
tolerances as the bare-system test in test_adjoint_solve.py::test_gradcheck.
All tests require the lightsim2grid C++ bridge and a GPU; gradcheck is FP64
only (FP32 finite differences are too noisy to validate the adjoint).
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from test_augmented_features import (
    _solved_hvdc_droop_grid,
    _solved_multislack_grid,
    _solve_with_fallback,
)

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping")

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="augmented differentiable PF needs the lightsim2grid C++ bridge")


def _is_fp32() -> bool:
    try:
        from gpusim2grid._gpusim2grid import is_fp32 as _fp32
        return bool(_fp32)
    except ImportError:
        return False


def _gradcheck_grid(grid, max_iter=40, tol=1e-10, eps=1e-5, atol=1e-3, rtol=1e-2,
                    nondet_tol=0.0):
    """gradcheck driver: Sbus (seeded from the grid's own solved state) -> V.real.sum()."""
    from gpusim2grid.differentiable import solve_power_flow

    Sbus0 = grid.get_Sbus_solver()
    Sr = torch.tensor(Sbus0.real, dtype=torch.float64, device="cuda", requires_grad=True)
    Si = torch.tensor(Sbus0.imag, dtype=torch.float64, device="cuda", requires_grad=True)

    def pf_fn(Sr, Si):
        V = solve_power_flow(Sr, Si, max_iter=max_iter, tol=tol, grid=grid)
        return V.real.sum()

    return torch.autograd.gradcheck(pf_fn, (Sr, Si), eps=eps, atol=atol, rtol=rtol,
                                    nondet_tol=nondet_tol)


def _solved_remote_gen_grid():
    """case14 with generator 3 (bus 7) remotely regulating bus 9 (bordered
    VoltageControl), ac-solved (mirrors test_augmented_features.py)."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        if not hasattr(model, "set_gen_regulated_bus"):
            pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
        model.set_gen_regulated_bus(3, 9)   # gen 3 (bus 7) regulates bus 9
        model.tell_solver_need_reset()
        _solve_with_fallback(model, net.bus.shape[0])
    return model


def _solved_svc_grid(slope_pu=0.02, reg_bus=9):
    """case14 with a voltage-mode SVC on bus 10 regulating bus `reg_bus`,
    ac-solved (mirrors test_augmented_features.py)."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        if not hasattr(model, "init_svcs"):
            pytest.skip("this lightsim2grid build has no init_svcs")
        model.init_svcs([1], np.array([1.03]), np.array([0.0]), np.array([slope_pu]),
                        np.array([-100.0]), np.array([100.0]),
                        np.array([reg_bus], dtype=np.int32),
                        np.array([10], dtype=np.int32))   # SVC on bus 10
        model.tell_solver_need_reset()
        _solve_with_fallback(model, net.bus.shape[0])
    return model


@requires_gpu
@needs_bridge
def test_gradcheck_distributed_slack():
    """Priority case: finite-difference gradcheck on a genuine multi-slack
    (N=2) grid — the augmented J carries the slack_absorbed column + extra
    P rows for the slack participants."""
    if _is_fp32():
        pytest.skip("FP32 build: gradcheck requires FP64")

    grid = _solved_multislack_grid()
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]

    from gpusim2grid._gpusim2grid import _make_acpf_session_from_lsgrid
    sess = _make_acpf_session_from_lsgrid(grid, 40, 1e-10)
    assert sess.dim_J > n_pv + 2 * n_pq   # structural sanity: augmented, not bare

    assert _gradcheck_grid(grid)


@requires_gpu
@needs_bridge
def test_gradcheck_hvdc_droop():
    """HVDC angle-droop: adds 0 rows/cols but Va-dependent feature slopes on
    the theta columns of the two connected end buses."""
    if _is_fp32():
        pytest.skip("FP32 build: gradcheck requires FP64")

    grid = _solved_hvdc_droop_grid()
    assert _gradcheck_grid(grid, max_iter=50, tol=1e-11)


@requires_gpu
@needs_bridge
def test_gradcheck_remote_gen_voltage_control():
    """Remote-regulating generator: bordered VoltageControl adds a q-column
    (controller reactive unknown) + a voltage-constraint custom row."""
    if _is_fp32():
        pytest.skip("FP32 build: gradcheck requires FP64")

    grid = _solved_remote_gen_grid()
    n_pv = grid.get_pv().shape[0]
    n_pq = grid.get_pq().shape[0]

    from gpusim2grid._gpusim2grid import _make_acpf_session_from_lsgrid
    sess = _make_acpf_session_from_lsgrid(grid, 50, 1e-11)
    assert sess.dim_J > n_pv + 2 * n_pq

    # VoltageControl's mismatch adjustment uses an atomic add (multiple
    # controllers on the same regulated bus), which introduces float rounding
    # non-determinism across repeated backward calls — same accommodation as
    # test_diff_flows.py::test_gradcheck_full_pipeline (CUDA scatter_add).
    assert _gradcheck_grid(grid, max_iter=50, tol=1e-11, nondet_tol=1e-6)


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("slope_pu", [0.0, 0.02])
def test_gradcheck_svc(slope_pu):
    """Voltage-mode SVC: slope 0 exercises the constant-only feature entries,
    slope > 0 additionally exercises the (v_row, q_col)=slope feature."""
    if _is_fp32():
        pytest.skip("FP32 build: gradcheck requires FP64")

    grid = _solved_svc_grid(slope_pu=slope_pu)
    assert _gradcheck_grid(grid, max_iter=50, tol=1e-11, nondet_tol=1e-6)
