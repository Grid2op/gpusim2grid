"""
test_regression_direct_refactor.py — Phase 1 regression test for DirectRefactorEvery.

Verifies that porting the existing NR loop into ContingencyDriver<DirectRefactorEvery>
produces results that are correct against the lightsim2grid CPU oracle.

The "bit-for-bit" guarantee is structural: the driver calls the same batch kernels
(fill_FP, fill_FQ, fill_J, update_Va, update_Vm) in the same order with the same
arguments as the original nr_iter_step() loop.  The factorize/refactorize policy
is also identical (FACTORIZATION once globally, REFACTORIZATION for all subsequent
calls), controlled by DirectRefactorEvery::factorized_.

Acceptance criteria (from the phase-1 plan):
  1. max|V_gpu| vs cpu oracle < 5e-4 pu for converged contingencies.
  2. All residuals are finite.
  3. Strategy timing breakdown: t_first_factorize > 0 (one FACTORIZATION happened),
     t_refactorize > 0 (REFACTORIZATION happened for subsequent calls),
     n_refactorize == n_chunks * nb_iter - 1.

Driven through the public ``ContingencyAnalysisSolver`` (session) API with the
``direct_refactor_every`` strategy.  N-1 contingencies are built from branch IDs
(lines first, then trafos) — the same order as lightsim2grid's ``add_all_n1()``.
"""

import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu

_VOLTAGE_ATOL_VS_CPU = 5e-4    # |V_gpu| vs |V_cpu| pu for converged cases
_RESIDUAL_CONVERGED  = 1e-4    # threshold to declare a contingency converged


def _run_gpu_ca(ieee14_grid, ieee14_base_case, batch_size=5, nb_iter=10):
    """Run GPU contingency analysis (DirectRefactorEvery).

    Returns (V_2d, residuals, timings, n_ctg).
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
    solver.strategy = 'direct_refactor_every'
    solver.run()

    V_flat    = solver.V_results.to_numpy()
    residuals = solver.residuals.to_numpy()
    timings   = solver.timings
    V_2d = V_flat.reshape(n_ctg, d["n_bus"])
    return V_2d, residuals, timings, n_ctg


class TestDirectRefactorEveryRegression:
    """Structural regression: DirectRefactorEvery via ContingencyDriver."""

    def test_voltages_match_cpu_oracle(
        self,
        ieee14_grid,
        ieee14_base_case,
        ieee14_cpu_n1_reference,
    ):
        """GPU voltages (DirectRefactorEvery) must match the CPU oracle within 5e-4 pu."""
        V_gpu, residuals, _, _ = _run_gpu_ca(ieee14_grid, ieee14_base_case, batch_size=5)

        V_cpu_full, _, _ = ieee14_cpu_n1_reference
        n_bus = ieee14_base_case["n_bus"]
        V_cpu_mag = np.abs(V_cpu_full[:, :n_bus])

        gpu_conv = np.abs(residuals) < _RESIDUAL_CONVERGED
        cpu_conv = np.abs(V_cpu_full).max(axis=1) > 1e-4
        both_conv = gpu_conv & cpu_conv

        assert both_conv.any(), "No contingency converged on both GPU and CPU solvers"

        Vmag_gpu = np.abs(V_gpu[both_conv])
        Vmag_cpu = V_cpu_mag[both_conv]
        err = np.abs(Vmag_gpu - Vmag_cpu).max()
        assert err < _VOLTAGE_ATOL_VS_CPU, (
            f"Max |V| error vs CPU oracle: {err:.2e} pu (limit {_VOLTAGE_ATOL_VS_CPU:.0e})"
        )

    def test_all_residuals_finite(self, ieee14_grid, ieee14_base_case):
        """No residuals should be NaN or Inf after DirectRefactorEvery solve."""
        V_gpu, residuals, _, _ = _run_gpu_ca(ieee14_grid, ieee14_base_case, batch_size=5)

        simulated = np.isfinite(V_gpu).all(axis=1)
        assert np.all(np.isfinite(residuals[simulated])), (
            "Non-finite residuals in simulated (non-disconnected) contingencies"
        )

    def test_factorization_policy(self, ieee14_grid, ieee14_base_case):
        """DirectRefactorEvery: exactly one FACTORIZATION, rest REFACTORIZATION.

        With n_chunks chunks and nb_iter iterations per chunk, the expected number
        of refactorizations is n_chunks * nb_iter - 1 (first call is FACTORIZATION).
        """
        nb_iter = 10
        batch_size = 5

        V_gpu, _, timings, n_ctg = _run_gpu_ca(
            ieee14_grid, ieee14_base_case,
            batch_size=batch_size, nb_iter=nb_iter
        )
        n_chunks = (n_ctg + batch_size - 1) // batch_size

        expected_refactorize = n_chunks * nb_iter - 1
        assert timings.n_refactorize == expected_refactorize, (
            f"Expected {expected_refactorize} refactorizations, "
            f"got {timings.n_refactorize}"
        )

        # t_first_factorize.wall_ms should be nonzero (one FACTORIZATION happened)
        assert timings.t_first_factorize.wall_ms > 0, (
            "t_first_factorize.wall_ms is zero — FACTORIZATION was never called"
        )

        # t_refactorize.wall_ms should be nonzero (REFACTORIZATION happened)
        assert timings.t_refactorize.wall_ms > 0, (
            "t_refactorize.wall_ms is zero — REFACTORIZATION was never called"
        )

    def test_results_stable_across_batch_sizes(
        self, ieee14_grid, ieee14_base_case
    ):
        """DirectRefactorEvery voltages must be consistent across different batch sizes.

        Different chunk sizes produce numerically identical results because each chunk
        re-tiles V and Ybus from the base case (no accumulated error between chunks).
        """
        V_bs1, res_bs1, _, _ = _run_gpu_ca(ieee14_grid, ieee14_base_case, batch_size=1)
        V_bs5, res_bs5, _, _ = _run_gpu_ca(ieee14_grid, ieee14_base_case, batch_size=5)

        conv = (np.abs(res_bs1) < _RESIDUAL_CONVERGED) & \
               (np.abs(res_bs5) < _RESIDUAL_CONVERGED)
        if not conv.any():
            pytest.skip("No contingency converged with both batch sizes")

        err = np.abs(V_bs5[conv] - V_bs1[conv]).max()
        assert err < 1e-5, (
            f"Max |V| difference between batch_size=1 and batch_size=5: {err:.2e}"
        )
