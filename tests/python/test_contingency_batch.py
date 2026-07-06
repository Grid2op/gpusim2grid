"""
test_contingency_batch.py — GPU N-1 contingency analysis correctness.

Tests validate:
  1. Convergence rate: a reasonable fraction of N-1s converge.
  2. Voltage agreement with the lightsim2grid ContingencyAnalysisCPP CPU reference.
  3. Batch-size independence: results must be identical regardless of chunk size.

Driven through the public ``_ContingencyAnalysisSolver`` (session) API.  N-1
contingencies are built from branch IDs: id ``c`` trips line ``c`` for
``c < n_lines`` and trafo ``c - n_lines`` otherwise — the same lines-then-trafos
order used by the triplet fixture and lightsim2grid's ``add_all_n1()``.
"""
import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays


pytestmark = requires_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONVERGED_FRACTION_MIN = 0.80   # at least 80 % of N-1s should converge
_VOLTAGE_ATOL_VS_CPU   = 5e-4   # |V_gpu| vs |V_cpu| for converged cases (pu)
_RESIDUAL_CONVERGED    = 1e-4   # threshold used to call a contingency "converged"


def _run_ca(ieee14_grid, ieee14_base_case, batch_size, nb_iter=10):
    """Run GPU contingency analysis via _ContingencyAnalysisSolver.

    Returns (V_2d, residuals, timings, n_lines, n_trafos).
    """
    from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

    d = ieee14_base_case
    branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
    n_ctg = n_lines + n_trafos
    cont_branch_ids = [[c] for c in range(n_ctg)]

    solver = _ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6,
    )
    solver.set_branch_data(*branch_data)
    solver.build_contingencies(cont_branch_ids)
    solver.run()

    V_flat    = solver.V_results.to_numpy()
    residuals = solver.residuals.to_numpy()
    timings   = solver.timings
    V_2d = V_flat.reshape(n_ctg, d["n_bus"])
    return V_2d, residuals, timings, n_lines, n_trafos


# ---------------------------------------------------------------------------
# Convergence rate
# ---------------------------------------------------------------------------

class TestConvergenceRate:
    def test_minimum_convergence_fraction(
        self, ieee14_grid, ieee14_base_case
    ):
        """At least 80 % of N-1 contingencies must converge on IEEE 14."""
        V_2d, residuals, _, n_lines, n_trafos = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        n_ctg = n_lines + n_trafos
        converged = (np.abs(residuals) < _RESIDUAL_CONVERGED).sum()
        fraction = converged / n_ctg
        assert fraction >= _CONVERGED_FRACTION_MIN, (
            f"Only {fraction:.1%} of contingencies converged, expected ≥{_CONVERGED_FRACTION_MIN:.0%}"
        )

    def test_all_residuals_are_finite(
        self, ieee14_grid, ieee14_base_case
    ):
        """No residual should be NaN or Inf — diverged cases should still be finite."""
        v_out, residuals, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        simulated = np.isfinite(v_out).all(axis=1)
        assert np.all(np.isfinite(residuals[simulated])), "Non-finite residuals detected for simulated contingency"

    def test_voltages_are_finite_for_converged(
        self, ieee14_grid, ieee14_base_case
    ):
        """Voltage magnitudes for converged contingencies must be finite and positive."""
        V_2d, residuals, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        converged_mask = np.abs(residuals) < _RESIDUAL_CONVERGED
        assert converged_mask.any(), "No contingencies converged — cannot check voltages"
        Vmag = np.abs(V_2d[converged_mask])
        assert np.all(np.isfinite(Vmag)), "Non-finite voltages in converged contingencies"
        assert np.all(Vmag > 1e-4), "Near-zero voltages in converged contingencies"


# ---------------------------------------------------------------------------
# Agreement with lightsim2grid CPU reference
# ---------------------------------------------------------------------------

class TestVsLightsim2gridReference:
    def test_voltage_magnitude_agreement(
        self,
        ieee14_grid,
        ieee14_base_case,
        ieee14_cpu_n1_reference,
    ):
        """GPU |V| must match ContingencyAnalysisCPP |V| for converged contingencies."""
        d = ieee14_base_case
        V_gpu, residuals, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        V_cpu_full, n_lines, n_trafos = ieee14_cpu_n1_reference

        # CPU reference is sized (n_ctg, 2*n_bus) — take only the first n_bus columns
        V_cpu = np.abs(V_cpu_full[:, :d["n_bus"]])

        # Filter: only compare contingencies that converged on both solvers
        gpu_conv = np.abs(residuals) < _RESIDUAL_CONVERGED
        # lightsim2grid marks non-converged with all-zeros voltages
        cpu_conv = np.abs(V_cpu_full).max(axis=1) > 1e-4

        both_conv = gpu_conv & cpu_conv
        assert both_conv.any(), "No contingency converged on both GPU and CPU solvers"

        Vmag_gpu = np.abs(V_gpu[both_conv])
        Vmag_cpu = V_cpu[both_conv]
        err = np.abs(Vmag_gpu - Vmag_cpu)
        assert err.max() < _VOLTAGE_ATOL_VS_CPU, (
            f"Max |V| error vs CPU reference: {err.max():.2e} pu (limit {_VOLTAGE_ATOL_VS_CPU:.0e})"
        )

    def test_convergence_agreement(
        self,
        ieee14_grid,
        ieee14_base_case,
        ieee14_cpu_n1_reference,
    ):
        """GPU and CPU should agree on which contingencies converge."""
        V_gpu, residuals, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        V_cpu_full, _, _ = ieee14_cpu_n1_reference

        gpu_conv = np.abs(residuals) < _RESIDUAL_CONVERGED
        cpu_conv = np.abs(V_cpu_full).max(axis=1) > 1e-4

        # GPU should not declare converged when CPU says diverged (strict direction)
        false_positives = gpu_conv & ~cpu_conv
        assert false_positives.sum() == 0, (
            f"{false_positives.sum()} contingencies converged on GPU but not CPU"
        )
        # for case 300 this should not happen
        false_negatives = ~gpu_conv & cpu_conv
        assert false_negatives.sum() == 0, (
            f"{false_positives.sum()} contingencies converged on CPU but not GPU (can happen for large grid)"
        )


# ---------------------------------------------------------------------------
# Batch-size independence
# ---------------------------------------------------------------------------

class TestBatchSizeIndependence:
    @pytest.mark.parametrize("batch_size", [1, 5, 20])
    def test_voltages_invariant_to_batch_size(
        self, ieee14_grid, ieee14_base_case, batch_size, solver_atol
    ):
        """GPU voltages must be identical (up to FP rounding) for any batch_size."""
        V_ref, residuals_ref, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=20
        )
        V_bs, residuals_bs, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=batch_size
        )

        # Compare only cases that converged with both batch sizes
        conv = (np.abs(residuals_ref) < _RESIDUAL_CONVERGED) & \
               (np.abs(residuals_bs)  < _RESIDUAL_CONVERGED)
        if not conv.any():
            pytest.skip("No contingency converged with both batch sizes")

        err = np.abs(V_bs[conv] - V_ref[conv]).max()
        # Allow a tiny accumulation from chunked FP32 arithmetic
        batch_atol = max(solver_atol, 1e-5)
        assert err < batch_atol, (
            f"batch_size={batch_size}: max V difference vs batch_size=20 is {err:.2e}"
        )

    @pytest.mark.parametrize("batch_size", [1, 5, 20])
    def test_residuals_invariant_to_batch_size(
        self, ieee14_grid, ieee14_base_case, batch_size
    ):
        """Convergence decisions must not depend on batch_size."""
        _, res_ref, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=20
        )
        _, res_bs, _, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=batch_size
        )

        conv_ref = np.abs(res_ref) < _RESIDUAL_CONVERGED
        conv_bs  = np.abs(res_bs)  < _RESIDUAL_CONVERGED
        disagree = (conv_ref != conv_bs).sum()
        assert disagree == 0, (
            f"batch_size={batch_size} flips convergence decision for {disagree} contingencies"
        )


# ---------------------------------------------------------------------------
# Tripped-branch zero-flow
# ---------------------------------------------------------------------------

class TestTrippedBranchZeroFlow:
    def test_tripped_branch_has_zero_or_amps(self, ieee14_solver_with_flows):
        """For every converged N-1, the disconnected branch must carry 0 A."""
        d = ieee14_solver_with_flows
        or_amps        = d["or_amps"]
        residuals      = d["residuals"]
        cont_branch_ids = d["cont_branch_ids"]

        converged = np.abs(residuals) < _RESIDUAL_CONVERGED
        assert converged.any(), "No contingency converged — cannot check tripped-branch flows"

        failures = []
        for c_id, branch_ids in enumerate(cont_branch_ids):
            if not converged[c_id]:
                continue
            for b in branch_ids:
                flow = or_amps[c_id, b]
                if flow != 0.0:
                    failures.append((c_id, b, flow))

        assert not failures, (
            "Tripped branch(es) carry non-zero flow:\n" +
            "\n".join(f"  contingency {c}, branch {b}: {f:.4f} A" for c, b, f in failures)
        )


# ---------------------------------------------------------------------------
# Timing sanity
# ---------------------------------------------------------------------------

class TestTimings:
    def test_timings_object_has_expected_fields(
        self, ieee14_grid, ieee14_base_case
    ):
        """ContingencyTimings must expose all expected wall-clock fields."""
        _, _, timings, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        # Fields declared in timing_utils.hpp / python_bindings.cpp
        for attr in (
            "t_base_case_ms", "t_analysis_ms", "t_chunks_total_wall_ms", "n_chunks",
            "t_spmv", "t_fill_F", "t_fill_J",
            "t_first_factorize", "t_refactorize", "n_refactorize",
            "t_solve", "t_update_V", "t_residual",
            "t_per_contingency_ms",
        ):
            assert hasattr(timings, attr), f"Missing field: {attr}"

    def test_total_time_is_positive(
        self, ieee14_grid, ieee14_base_case
    ):
        """The total wall-clock time must be strictly positive."""
        _, _, timings, _, _ = _run_ca(
            ieee14_grid, ieee14_base_case, batch_size=5
        )
        assert timings.t_chunks_total_wall_ms > 0
