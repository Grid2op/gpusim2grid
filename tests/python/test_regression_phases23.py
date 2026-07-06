"""
test_regression_phases23.py — Regression tests for phases 2 and 3 strategies.

Phase 2: DirectBaseCaseFactors
  - Reuses base-case LU factors for all contingencies (no per-contingency factorize).
  - Cheapest; approximate (uses base-case Jacobian, not contingency Jacobian).
  - Expected: n_refactorize == 0, t_first_factorize.wall_ms == 0 (factorize done
    during construction captured in t_analysis_ms), voltages finite.

Phase 3: DirectIter0Only
  - Fills J at iter==0 per chunk; FACTORIZE/REFACTORIZE once per chunk;
    SOLVE-only for iter 1..nb_iter-1.
  - Expected: n_refactorize == n_chunks - 1, t_first_factorize.wall_ms > 0,
    t_refactorize.wall_ms > 0 (if n_chunks > 1), voltages close to DirectRefactorEvery.
"""

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

_RESIDUAL_CONVERGED = 1e-4


# =============================================================================
# Helper: run contingency analysis via the _ContingencyAnalysisSolver object API
# =============================================================================

def _run_with_strategy(ieee14_base_case, ieee14_grid, branch_ids_per_ctg,
                       strategy, batch_size=5, nb_iter=10, compute_flows=False):
    """Run GPU contingency analysis using the stateful object API.

    Uses set_branch_data + build_contingencies so branch IDs can be passed
    directly.  Returns (V_2d, residuals, timings) or, when compute_flows=True,
    (V_2d, residuals, timings, or_amps) where or_amps has shape (n_ctg, n_branches).
    """
    from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

    d = ieee14_base_case

    solver = _ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter)

    # Extract branch π-model data from lightsim2grid grid
    lines  = ieee14_grid.get_lines()
    trafos = ieee14_grid.get_trafos()
    all_branches = list(lines) + list(trafos)
    n_branches = len(all_branches)

    branch_from = np.array([b.bus1_id for b in all_branches], dtype=np.int32)
    branch_to   = np.array([b.bus2_id for b in all_branches], dtype=np.int32)
    yff = np.array([b.yac_11 for b in all_branches], dtype=complex)
    yft = np.array([b.yac_12 for b in all_branches], dtype=complex)
    ytf = np.array([b.yac_21 for b in all_branches], dtype=complex)
    ytt = np.array([b.yac_22 for b in all_branches], dtype=complex)
    bus_vn_kv = ieee14_grid.get_bus_vn_kv().astype(float)

    solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                           bus_vn_kv, 100.0)
    solver.build_contingencies(branch_ids_per_ctg)
    solver.strategy = strategy
    solver.run()

    n_ctg = len(branch_ids_per_ctg)
    n_bus = d["n_bus"]
    V_2d = solver.V_results.to_numpy().reshape(n_ctg, n_bus)
    residuals = solver.residuals.to_numpy()
    timings = solver.timings

    if compute_flows:
        solver.compute_flows()
        or_amps = solver.or_amps.to_numpy().reshape(n_ctg, n_branches)
        return V_2d, residuals, timings, or_amps

    return V_2d, residuals, timings


# =============================================================================
# Shared contingency set: first 12 single-line N-1 outages
# =============================================================================

@pytest.fixture(scope="module")
def n1_branch_ids(ieee14_grid):
    """12 single-branch N-1 contingencies (first 12 lines of IEEE 14-bus)."""
    n_lines = len(ieee14_grid.get_lines())
    n_ctg = min(12, n_lines)
    return [[i] for i in range(n_ctg)]


# =============================================================================
# Phase 2: DirectBaseCaseFactors
# =============================================================================

class TestDirectBaseCaseFactors:
    """Phase 2 — DirectBaseCaseFactors: reuse base-case LU for all contingencies."""

    def test_dbcf_residuals_finite(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """All residuals must be finite for non-disconnected contingencies."""
        V, residuals, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_base_case_factors', batch_size=5, nb_iter=10)

        finite_V = np.isfinite(V).all(axis=1)
        assert np.all(np.isfinite(residuals[finite_V])), (
            "Non-finite residuals for simulated contingencies (DirectBaseCaseFactors)"
        )

    def test_dbcf_factorization_policy(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectBaseCaseFactors: zero refactorize calls, zero NR-loop factorize time.

        The single FACTORIZE happens inside initialize_from_base() (during
        construction), captured in t_analysis_ms — not in t_first_factorize
        (which tracks factorize calls within the NR loop only).
        """
        _, _, timings = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_base_case_factors', batch_size=5, nb_iter=10)

        assert timings.n_refactorize == 0, (
            f"Expected 0 refactorizations for DirectBaseCaseFactors, "
            f"got {timings.n_refactorize}"
        )
        assert timings.t_first_factorize.wall_ms == 0., (
            "t_first_factorize.wall_ms should be 0 for DirectBaseCaseFactors "
            "(factorize done in initialize_from_base(), captured in t_analysis_ms)"
        )
        assert timings.t_refactorize.wall_ms == 0., (
            "t_refactorize.wall_ms should be 0 for DirectBaseCaseFactors"
        )

    def test_dbcf_voltages_finite(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectBaseCaseFactors voltages must be finite for simulated contingencies.

        This is an approximation strategy (base-case J used for all contingencies).
        We only check finiteness here; accuracy is checked implicitly via residuals.
        """
        V, residuals, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_base_case_factors', batch_size=5, nb_iter=10)

        finite_V = np.isfinite(V).all(axis=1)
        assert finite_V.any(), (
            "All DirectBaseCaseFactors voltages are non-finite — solver diverged"
        )

    def test_dbcf_tripped_branch_zero_flow(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectBaseCaseFactors: disconnected branch must carry 0 A."""
        _, residuals, _, or_amps = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_base_case_factors', batch_size=5, nb_iter=10,
            compute_flows=True)

        converged = np.abs(residuals) < _RESIDUAL_CONVERGED
        assert converged.any(), "No contingency converged — cannot check tripped-branch flows"

        failures = []
        for c_id, branch_ids in enumerate(n1_branch_ids):
            if not converged[c_id]:
                continue
            for b in branch_ids:
                if or_amps[c_id, b] != 0.0:
                    failures.append((c_id, b, or_amps[c_id, b]))

        assert not failures, (
            "Tripped branch(es) carry non-zero flow (DirectBaseCaseFactors):\n" +
            "\n".join(f"  contingency {c}, branch {b}: {f:.4f} A" for c, b, f in failures)
        )


# =============================================================================
# Phase 3: DirectIter0Only
# =============================================================================

class TestDirectIter0Only:
    """Phase 3 — DirectIter0Only: Jacobian frozen at iter==0 per chunk."""

    def test_dio_residuals_finite(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """All residuals must be finite for non-disconnected contingencies."""
        V, residuals, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only', batch_size=5, nb_iter=10)

        finite_V = np.isfinite(V).all(axis=1)
        assert np.all(np.isfinite(residuals[finite_V])), (
            "Non-finite residuals for simulated contingencies (DirectIter0Only)"
        )

    def test_dio_factorization_policy(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectIter0Only: n_refactorize == n_chunks - 1.

        One FACTORIZE for the first chunk (iter==0), then one REFACTORIZE per
        subsequent chunk (iter==0).  All other iterations within each chunk are
        SOLVE-only.
        """
        batch_size = 5
        nb_iter    = 10
        n_ctg      = len(n1_branch_ids)
        n_chunks   = (n_ctg + batch_size - 1) // batch_size

        _, _, timings = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only',
            batch_size=batch_size, nb_iter=nb_iter)

        expected_refactorize = n_chunks - 1
        assert timings.n_refactorize == expected_refactorize, (
            f"Expected {expected_refactorize} refactorizations for DirectIter0Only "
            f"(n_chunks={n_chunks}), got {timings.n_refactorize}"
        )

        assert timings.t_first_factorize.wall_ms > 0., (
            "t_first_factorize.wall_ms is zero — FACTORIZATION was never called"
        )

        if n_chunks > 1:
            assert timings.t_refactorize.wall_ms > 0., (
                f"t_refactorize.wall_ms is zero — REFACTORIZATION was never called "
                f"(expected {n_chunks - 1} refactorizations)"
            )

    def test_dio_fewer_refactorize_than_dre(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectIter0Only does fewer refactorizations than DirectRefactorEvery.

        For nb_iter > 1:
          n_refactorize_dio = n_chunks - 1
          n_refactorize_dre = n_chunks * nb_iter - 1
          saving = (nb_iter - 1) * n_chunks
        """
        batch_size = 5
        nb_iter    = 10
        n_ctg      = len(n1_branch_ids)
        n_chunks   = (n_ctg + batch_size - 1) // batch_size

        _, _, t_dio = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only',
            batch_size=batch_size, nb_iter=nb_iter)

        _, _, t_dre = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_refactor_every',
            batch_size=batch_size, nb_iter=nb_iter)

        assert t_dio.n_refactorize < t_dre.n_refactorize, (
            f"DirectIter0Only should do fewer refactorizations than "
            f"DirectRefactorEvery: {t_dio.n_refactorize} vs {t_dre.n_refactorize}"
        )
        # Exact counts
        assert t_dio.n_refactorize == n_chunks - 1, (
            f"DirectIter0Only: expected {n_chunks - 1}, got {t_dio.n_refactorize}"
        )
        assert t_dre.n_refactorize == n_chunks * nb_iter - 1, (
            f"DirectRefactorEvery: expected {n_chunks * nb_iter - 1}, "
            f"got {t_dre.n_refactorize}"
        )

    def test_dio_voltages_finite(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectIter0Only voltages must be finite for simulated contingencies."""
        V, _, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only', batch_size=5, nb_iter=10)

        finite_V = np.isfinite(V).all(axis=1)
        assert finite_V.any(), (
            "All DirectIter0Only voltages are non-finite — solver diverged"
        )

    def test_dio_tripped_branch_zero_flow(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectIter0Only: disconnected branch must carry 0 A."""
        _, residuals, _, or_amps = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only', batch_size=5, nb_iter=10,
            compute_flows=True)

        converged = np.abs(residuals) < _RESIDUAL_CONVERGED
        assert converged.any(), "No contingency converged — cannot check tripped-branch flows"

        failures = []
        for c_id, branch_ids in enumerate(n1_branch_ids):
            if not converged[c_id]:
                continue
            for b in branch_ids:
                if or_amps[c_id, b] != 0.0:
                    failures.append((c_id, b, or_amps[c_id, b]))

        assert not failures, (
            "Tripped branch(es) carry non-zero flow (DirectIter0Only):\n" +
            "\n".join(f"  contingency {c}, branch {b}: {f:.4f} A" for c, b, f in failures)
        )

    def test_dio_voltages_close_to_dre(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectIter0Only voltages close to DirectRefactorEvery for converged cases.

        With nb_iter=10, the chord Newton (iter0 J) produces similar voltage
        magnitudes as full-refactorize Newton for well-conditioned contingencies.
        Tolerance: 5e-2 pu (looser than phase 1 since DIO is also approximate).
        """
        V_dio, res_dio, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_iter0_only', batch_size=5, nb_iter=10)

        V_dre, res_dre, _ = _run_with_strategy(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_refactor_every', batch_size=5, nb_iter=10)

        conv = (np.abs(res_dio) < _RESIDUAL_CONVERGED) & \
               (np.abs(res_dre) < _RESIDUAL_CONVERGED)

        if not conv.any():
            pytest.skip("No contingency converged with both DIO and DRE strategies")

        err = np.abs(V_dio[conv] - V_dre[conv]).max()
        assert err < 5e-2, (
            f"DirectIter0Only vs DirectRefactorEvery max |V| diff: {err:.3f} pu "
            f"(limit 5e-2 pu)"
        )
