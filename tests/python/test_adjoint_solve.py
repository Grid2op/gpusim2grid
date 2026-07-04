"""
test_adjoint_solve.py — Phase 1 regression + gradient check for the adjoint solve.

Tests:
  1. test_dimensions          — dim_J, n_pvpq, n_pq are consistent.
  2. test_JT_solve_regression — GPU Jᵀ solve matches CPU scipy reference.
  3. test_gradcheck           — torch.autograd.gradcheck on the full pipeline.

All tests are skipped if no GPU is available.
Tests 2 and 3 are additionally skipped in FP32 builds (insufficient precision
to match the FP64 CPU reference at 1e-8 or to pass torch gradcheck).
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed — skipping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fp32() -> bool:
    try:
        from gpusim2grid._gpusim2grid import is_fp32 as _fp32
        return bool(_fp32)
    except ImportError:
        return False


def _make_session(ieee14_base_case):
    from gpusim2grid._gpusim2grid import AcPfNrSession
    d = ieee14_base_case
    return AcPfNrSession(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        10, 1e-6,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdjointSolve:

    def test_dimensions(self, ieee14_base_case):
        """dim_J, n_pvpq, n_pq are internally consistent."""
        sess = _make_session(ieee14_base_case)
        assert sess.dim_J == sess.n_pvpq + sess.n_pq
        assert sess.dim_J > 0
        assert len(sess.pvpq) == sess.n_pvpq
        assert len(sess.pq)   == sess.n_pq

    def test_JT_solve_regression(self, ieee14_base_case):
        """GPU Jᵀ solve matches scipy.sparse.linalg.spsolve(J.T, rhs).

        Skipped in FP32 builds: FP32 linear algebra has insufficient precision
        to match the FP64 CPU reference at the 1e-6 threshold.
        """
        if _is_fp32():
            pytest.skip("FP32 build: Jᵀ regression requires FP64")

        import scipy.sparse.linalg as spla
        from _jacobian_ref import compute_jacobian

        d = ieee14_base_case
        sess = _make_session(d)

        # Build CPU reference Jacobian from the converged voltage
        V_conv = sess.get_v()
        pvpq = np.array(sess.pvpq, dtype=np.intp)
        pq_  = np.array(sess.pq,   dtype=np.intp)
        J_cpu = compute_jacobian(V_conv, d["Ybus"], pvpq, pq_)

        rng = np.random.default_rng(42)
        rhs_np = rng.standard_normal(sess.dim_J).astype(np.float64)

        # CPU reference
        lam_cpu = spla.spsolve(J_cpu.T.tocsr(), rhs_np)

        # GPU solve
        rhs_t = torch.from_numpy(rhs_np).cuda()
        sol_cap = sess.solve_JT_dlpack(rhs_t.__dlpack__())
        lam_gpu = torch.from_dlpack(sol_cap).clone().cpu().numpy()

        max_err = float(np.max(np.abs(lam_gpu - lam_cpu)))
        assert max_err < 1e-6, (
            f"Jᵀ solve max error {max_err:.2e} exceeds 1e-6\n"
            f"  lam_cpu[:5] = {lam_cpu[:5]}\n"
            f"  lam_gpu[:5] = {lam_gpu[:5]}"
        )

    def test_backward_nonzero(self, ieee14_base_case):
        """Backward produces non-zero grad_Sr for a real-part loss."""
        if _is_fp32():
            pytest.skip("FP32 build: this test uses FP64")
        from gpusim2grid.differentiable import PowerFlowFunction

        d = ieee14_base_case
        Sbus_np = d["Sbus"].astype(np.complex128)
        # Use device='cuda' directly so the tensor is a leaf on GPU
        Sr = torch.tensor(Sbus_np.real, dtype=torch.float64, device='cuda', requires_grad=True)
        Si = torch.tensor(Sbus_np.imag, dtype=torch.float64, device='cuda', requires_grad=True)

        Ybus, Vinit = d["Ybus"], d["v_init"].copy().astype(np.complex128)
        pv, pq = d["pv"], d["pq"]
        slack, slack_weights = d["slack"], d["slack_weights"]

        V = PowerFlowFunction.apply(Sr, Si, Ybus, Vinit, pv, pq,
                                    slack, slack_weights, 10, 1e-8)
        # Use V.real.sum() — a real-valued scalar loss
        loss = V.real.sum()
        loss.backward()
        assert Sr.grad is not None, "Sr.grad is None — backward was not called"
        assert torch.any(Sr.grad != 0), f"Sr.grad is all zeros:\n{Sr.grad}"

    def test_gradcheck(self, ieee14_base_case):
        """torch.autograd.gradcheck on PowerFlowFunction (FP64 only).

        Perturbs each element of Sbus_real / Sbus_imag with finite differences
        and compares to the analytic adjoint gradient.  Only meaningful in FP64
        (FP32 finite differences are too noisy to validate the adjoint).
        """
        if _is_fp32():
            pytest.skip("FP32 build: gradcheck requires FP64")

        from gpusim2grid.differentiable import PowerFlowFunction

        d = ieee14_base_case
        Sbus_np = d["Sbus"].astype(np.complex128)

        # Use device='cuda' directly so tensors are GPU leaves (not copies)
        Sr = torch.tensor(Sbus_np.real, dtype=torch.float64, device='cuda', requires_grad=True)
        Si = torch.tensor(Sbus_np.imag, dtype=torch.float64, device='cuda', requires_grad=True)

        # Capture all non-differentiable args (including scipy Ybus) in a
        # closure so gradcheck only sees (Sr, Si) as inputs to perturb.
        Ybus  = d["Ybus"]
        Vinit = d["v_init"].copy().astype(np.complex128)
        pv, pq = d["pv"], d["pq"]
        slack, slack_weights = d["slack"], d["slack_weights"]

        def pf_fn(Sr, Si):
            return PowerFlowFunction.apply(
                Sr, Si, Ybus, Vinit, pv, pq, slack, slack_weights, 10, 1e-8)

        result = torch.autograd.gradcheck(
            pf_fn, (Sr, Si),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-2,
            check_grad_dtypes=True,
        )
        assert result
