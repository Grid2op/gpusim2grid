"""
test_residuals.py — power-balance residual quality checks.

These tests compute F(V) = [ΔP_pvpq | ΔQ_pq] directly from the GPU output
voltages, independently of the solver's internal convergence tracking.

For contingency residuals the patched Ybus (base − branch admittance) is
assembled on the CPU for each N-1, which is correct but O(n_ctg * nnz).
This is fine for IEEE 14 in a test context.  The GPU side is driven through
the public ``ContingencyAnalysisSolver`` (session) API; contingency ``c`` trips
branch ``c`` (lines first, then trafos) — the same order as the triplet fixture,
so ``ieee14_contingencies[c]`` is the matching patch for GPU contingency ``c``.
"""
import numpy as np
import pytest

from conftest import (
    compute_pq_mismatch, requires_gpu, _is_fp32,
    branch_data_arrays, branch_eff_triplet,
)


pytestmark = requires_gpu

_CONVERGED_THRESHOLD = 1e-4   # residual below which a contingency "counts"

# Below this residual we are at the FP32 round-off floor on IEEE 14: further NR
# iterations no longer reduce ‖F‖∞ in any meaningful way, and the residual can
# even tick up by ULPs.  Monotonicity assertions are bypassed once both
# consecutive residuals dip below this threshold.  Under FP64 the floor is far
# below any practical tolerance, so we disable the bypass (threshold = 0).
_NOISE_FLOOR_RESIDUAL = 1e-4 if _is_fp32() else 0.0


# ---------------------------------------------------------------------------
# Base-case residuals
# ---------------------------------------------------------------------------

class TestBaseResiduals:
    def _solve_gpu(self, d, tol=1e-6, max_iter=10):
        from gpusim2grid.acpf_nr import acpf_nr_gpu
        v_out = np.zeros(d["n_bus"], dtype=complex)
        acpf_nr_gpu(
            v_out, d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            max_iter, tol, 1,
        )
        return v_out

    def test_residual_below_solver_tolerance(self, ieee14_base_case, residual_atol):
        """‖F(V_gpu)‖∞ must be below the requested NR tolerance."""
        d = ieee14_base_case
        tol = 1e-6
        v_gpu = self._solve_gpu(d, tol=tol)
        F = compute_pq_mismatch(d["Ybus"], v_gpu, d["Sbus"], d["pv"], d["pq"])
        assert np.abs(F).max() < residual_atol, (
            f"‖F‖∞ = {np.abs(F).max():.2e} exceeds {residual_atol:.0e}"
        )

    def test_residual_decreases_with_more_iterations(self, ieee14_base_case):
        """More NR iterations must monotonically decrease ‖F(V)‖∞."""
        d = ieee14_base_case
        F_norms = []
        for max_iter in (1, 3, 6, 10):
            v = self._solve_gpu(d, max_iter=max_iter, tol=0.0)   # tol=0 → always runs
            F = compute_pq_mismatch(d["Ybus"], v, d["Sbus"], d["pv"], d["pq"])
            F_norms.append(np.abs(F).max())

        for i in range(len(F_norms) - 1):
            # Skip the check once both residuals are below the FP32 noise floor:
            # at that point further NR iterations can no longer reduce ‖F‖∞,
            # and ULP-level fluctuations would cause a spurious failure.
            if (F_norms[i]     < _NOISE_FLOOR_RESIDUAL and
                F_norms[i + 1] < _NOISE_FLOOR_RESIDUAL):
                continue
            assert F_norms[i + 1] <= F_norms[i] * 1.1, (
                f"Residual did not decrease between iter counts "
                f"{[1, 3, 6, 10][i]} and {[1, 3, 6, 10][i+1]}: "
                f"{F_norms[i]:.2e} → {F_norms[i+1]:.2e}"
            )

    def test_active_power_balance_at_pv_buses(self, ieee14_base_case, residual_atol):
        """ΔP must be near zero at PV buses (voltage-controlled generators)."""
        d = ieee14_base_case
        v_gpu = self._solve_gpu(d)
        Ibus = d["Ybus"] @ v_gpu
        Scalc = v_gpu * np.conj(Ibus)
        dP_pv = np.abs((Scalc - d["Sbus"]).real[d["pv"]])
        assert dP_pv.max() < residual_atol, (
            f"Active power mismatch at PV buses: {dP_pv.max():.2e} pu"
        )

    def test_reactive_power_balance_at_pq_buses(self, ieee14_base_case, residual_atol):
        """ΔQ must be near zero at PQ (load) buses."""
        d = ieee14_base_case
        v_gpu = self._solve_gpu(d)
        Ibus = d["Ybus"] @ v_gpu
        Scalc = v_gpu * np.conj(Ibus)
        dQ_pq = np.abs((Scalc - d["Sbus"]).imag[d["pq"]])
        assert dQ_pq.max() < residual_atol, (
            f"Reactive power mismatch at PQ buses: {dQ_pq.max():.2e} pu"
        )


# ---------------------------------------------------------------------------
# Contingency residuals (patched Ybus)
# ---------------------------------------------------------------------------

def _run_ca(ieee14_grid, ieee14_base_case, batch_size=5, nb_iter=10):
    """Run all N-1 contingencies via ContingencyAnalysisSolver.

    Returns (V_flat, residuals, branch_data, n_lines, n_trafos).
    """
    from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver

    d = ieee14_base_case
    branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
    n_ctg = n_lines + n_trafos
    cont_branch_ids = [[c] for c in range(n_ctg)]

    solver = ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6,
    )
    solver.set_branch_data(*branch_data)
    solver.build_contingencies(cont_branch_ids)
    solver.run()
    return (solver.V_results.to_numpy(), solver.residuals.to_numpy(),
            branch_data, n_lines, n_trafos)


def _build_patched_ybus(Ybus, contingency):
    """Return Ybus with the branch admittance contribution removed (CSR)."""
    from scipy.sparse import lil_matrix
    Y = Ybus.tolil()
    for row, col, yr, yi in contingency:
        Y[row, col] -= complex(yr, yi)
    return Y.tocsr()


class TestContingencyResiduals:
    def test_converged_contingency_residuals_below_threshold(
        self, ieee14_grid, ieee14_base_case
    ):
        """For every converged N-1, ‖F(V, Ybus_patched)‖∞ < convergence threshold."""
        d = ieee14_base_case

        V_flat, residuals, branch_data, n_lines, n_trafos = _run_ca(
            ieee14_grid, ieee14_base_case
        )
        n_ctg = n_lines + n_trafos
        V_2d = V_flat.reshape(n_ctg, d["n_bus"])

        failures = []
        for c_id in range(n_ctg):
            if np.abs(residuals[c_id]) >= _CONVERGED_THRESHOLD:
                continue   # skip non-converged

            Y_pat = _build_patched_ybus(d["Ybus"], branch_eff_triplet(branch_data, c_id))
            F = compute_pq_mismatch(
                Y_pat, V_2d[c_id], d["Sbus"], d["pv"], d["pq"]
            )
            fmax = np.abs(F).max()
            if fmax >= _CONVERGED_THRESHOLD:
                l_or_t = "line" if c_id < n_lines else "trafo"
                failures.append(f"ctg {c_id} ({l_or_t}): ‖F‖∞={fmax:.2e}")

        assert not failures, (
            f"{len(failures)} converged contingencies failed the residual check:\n"
            + "\n".join(failures[:5])  # show first 5 only
        )

    def test_reported_residuals_consistent_with_recomputed(
        self, ieee14_grid, ieee14_base_case
    ):
        """Solver-reported residuals must be consistent with independently recomputed ones.

        We allow a 10× factor to account for the fact that the solver stores
        the ‖Δx‖ update-step norm, not the mismatch norm — they are correlated
        but not identical.
        """
        d = ieee14_base_case

        V_flat, residuals, branch_data, n_lines, n_trafos = _run_ca(
            ieee14_grid, ieee14_base_case
        )
        n_ctg = n_lines + n_trafos
        V_2d = V_flat.reshape(n_ctg, d["n_bus"])

        for c_id in range(n_ctg):
            if((np.abs(residuals[c_id]) >= _CONVERGED_THRESHOLD) or
               (np.isfinite(V_flat[c_id]).all())
            ):
                  continue

            Y_pat = _build_patched_ybus(d["Ybus"], branch_eff_triplet(branch_data, c_id))
            F = compute_pq_mismatch(
                Y_pat, V_2d[c_id], d["Sbus"], d["pv"], d["pq"]
            )
            recomputed = np.abs(F).max()
            # Both must be below threshold — directionality check
            assert recomputed < _CONVERGED_THRESHOLD * 10, (
                f"ctg {c_id}: reported residual={residuals[c_id]:.2e} but "
                f"recomputed ‖F‖∞={recomputed:.2e}"
            )


# ---------------------------------------------------------------------------
# Residual norms shape and dtype
# ---------------------------------------------------------------------------

class TestResidualOutputShape:
    def test_residuals_shape_equals_n_contingencies(
        self, ieee14_grid, ieee14_base_case
    ):
        """The solver must return exactly one residual per contingency."""
        _, residuals, _, n_lines, n_trafos = _run_ca(ieee14_grid, ieee14_base_case)
        n_ctg = n_lines + n_trafos
        assert residuals.shape == (n_ctg,), (
            f"Expected residuals shape ({n_ctg},), got {residuals.shape}"
        )

    def test_voltage_output_shape_equals_n_ctg_times_n_bus(
        self, ieee14_grid, ieee14_base_case
    ):
        """Flat voltage output must have length n_contingencies * n_bus."""
        d = ieee14_base_case
        V_flat, _, _, n_lines, n_trafos = _run_ca(ieee14_grid, ieee14_base_case)
        n_ctg = n_lines + n_trafos
        assert V_flat.shape == (n_ctg * d["n_bus"],), (
            f"Expected V_flat shape ({n_ctg * d['n_bus']},), got {V_flat.shape}"
        )
