"""
test_base_case.py — single-system AC Newton-Raphson correctness.

Each test solves the IEEE 14-bus base case on the GPU and compares against
the lightsim2grid KLU reference.  No contingency patching here — just the
plain solver pipeline.
"""
import numpy as np
import pytest

from conftest import compute_pq_mismatch, requires_gpu, _is_fp32


pytestmark = requires_gpu

# Run-to-run reproducibility tolerance.  FP32 GPU NR is not bit-deterministic
# across calls (reduction-order ULP differences); ~3 × FP32 epsilon is a safe
# floor.  FP64 stays at the original very-tight 1e-10.
_REPRODUCIBILITY_TOL = 1e-6 if _is_fp32() else 1e-10


# ---------------------------------------------------------------------------
# AC Newton-Raphson
# ---------------------------------------------------------------------------

class TestAcNrPowerFlow:
    def _run_gpu(self, d, max_iter=10, tol=1e-6):
        from gpusim2grid.acpf_nr import acpf_nr_gpu
        v_out = np.zeros(d["n_bus"], dtype=complex)
        timings = acpf_nr_gpu(
            v_out,
            d["Ybus"],
            d["v_init"].copy(),
            d["Sbus"],
            d["slack"],
            d["slack_weights"],
            d["pv"],
            d["pq"],
            max_iter,
            tol,
            1,
        )
        return v_out, timings

    def test_voltage_magnitude_matches_reference(self, ieee14_base_case, solver_atol):
        """|V| from GPU NR must match lightsim2grid KLU reference."""
        d = ieee14_base_case
        v_gpu, _ = self._run_gpu(d)
        err = np.abs(np.abs(v_gpu) - np.abs(d["v_ref"]))
        assert err.max() < solver_atol, (
            f"Max |V| error {err.max():.2e} pu exceeds {solver_atol:.0e}"
        )

    def test_voltage_angle_matches_reference(self, ieee14_base_case, solver_atol):
        """Voltage angle from GPU NR must match lightsim2grid KLU reference."""
        d = ieee14_base_case
        v_gpu, _ = self._run_gpu(d)
        # Compare angle differences to the slack (removes global phase offset)
        ref_angle = np.angle(d["v_ref"]) - np.angle(d["v_ref"][d["slack"][0]])
        gpu_angle = np.angle(v_gpu) - np.angle(v_gpu[d["slack"][0]])
        err = np.abs(gpu_angle - ref_angle)
        assert err.max() < solver_atol, (
            f"Max angle error {err.max():.2e} rad exceeds {solver_atol:.0e}"
        )

    def test_power_balance_residual_below_tolerance(self, ieee14_base_case, residual_atol):
        """F(V_gpu) ∞-norm must be below the NR convergence tolerance."""
        d = ieee14_base_case
        tol = 1e-6
        v_gpu, _ = self._run_gpu(d, tol=tol)
        F = compute_pq_mismatch(d["Ybus"], v_gpu, d["Sbus"], d["pv"], d["pq"])
        assert np.abs(F).max() < residual_atol, (
            f"Power balance residual {np.abs(F).max():.2e} pu exceeds {residual_atol:.0e}"
        )

    def test_timings_object_returned(self, ieee14_base_case):
        """acpf_nr_gpu must return a timings object with expected fields."""
        d = ieee14_base_case
        _, timings = self._run_gpu(d)
        # Smoke-test: at least one timing value is positive
        assert timings.t_total.wall_ms > 0, "t_total.wall_ms is not positive"

    def test_second_call_gives_same_result(self, ieee14_base_case, solver_atol):
        """Two consecutive GPU NR calls must return the same voltages (stateless)."""
        d = ieee14_base_case
        v1, _ = self._run_gpu(d)
        v2, _ = self._run_gpu(d)
        diff = np.abs(v1 - v2).max()
        assert diff < _REPRODUCIBILITY_TOL, (
            f"Consecutive AC NR calls returned different results: "
            f"max |v1 - v2| = {diff:.2e} (tol {_REPRODUCIBILITY_TOL:.0e})"
        )

    def test_non_convergence_fewer_iterations(self, ieee14_base_case):
        """With only 1 iteration the solver must not have converged to reference."""
        d = ieee14_base_case
        v_gpu_1iter, _ = self._run_gpu(d, max_iter=1, tol=1e-6)
        v_gpu_full, _  = self._run_gpu(d, max_iter=10, tol=1e-6)
        # 1-iteration result should be meaningfully different (err > 1e-5)
        err = np.abs(v_gpu_1iter - v_gpu_full).max()
        assert err > 1e-5, (
            "1-iteration GPU solve produced the same result as full solve — unexpected"
        )
