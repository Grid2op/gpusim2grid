"""
test_regression_refactor_every_n.py — Regression tests for DirectRefactorEveryN.

DirectRefactorEveryN
  - Fills J every iteration (needs_fresh_jacobian=True).
  - FACTORIZE on the very first call, then REFACTORIZE every N NR calls,
    SOLVE-only on all other calls.  N is a runtime parameter (refactor_period).
  - N=1 degenerates to DirectRefactorEvery (refactor every call).
  - refactor_period is mutable between run() calls.

Expected refactorize counts for n_chunks chunks and nb_iter iterations per chunk
(total NR calls = n_chunks * nb_iter):
  call 0                          → FACTOR   (not counted in n_refactorize)
  calls k > 0 where k % N == 0   → REFACTORIZE
  n_refactorize = len({k : 1 ≤ k < n_chunks*nb_iter, k % N == 0})
               = floor((n_chunks*nb_iter - 1) / N)
"""

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu

_RESIDUAL_CONVERGED = 1e-4
_BATCH_SIZE = 5
_NB_ITER    = 10


# =============================================================================
# Helper: run contingency analysis with a given strategy + optional refactor_period
# =============================================================================

def _run(ieee14_base_case, ieee14_grid, branch_ids_per_ctg,
         strategy, refactor_period=1, batch_size=_BATCH_SIZE, nb_iter=_NB_ITER,
         compute_flows=False):
    """Run GPU contingency analysis and return (V_2d, residuals, timings[, or_amps])."""
    from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver

    d = ieee14_base_case

    solver = ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter)

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

    solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, 100.0)
    solver.build_contingencies(branch_ids_per_ctg)
    solver.strategy = strategy
    solver.refactor_period = refactor_period
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


def _expected_n_refactorize(n_chunks, nb_iter, period):
    """Number of REFACTORIZE calls for DirectRefactorEveryN.

    call 0 → FACTOR; calls k in [1, n_chunks*nb_iter) with k%period==0 → REFACTORIZE.
    """
    total_calls = n_chunks * nb_iter
    return sum(1 for k in range(1, total_calls) if k % period == 0)


@pytest.fixture(scope="module")
def n1_branch_ids(ieee14_grid):
    """12 single-branch N-1 contingencies (first 12 lines of IEEE 14-bus)."""
    n_lines = len(ieee14_grid.get_lines())
    return [[i] for i in range(min(12, n_lines))]


# =============================================================================
# TestDirectRefactorEveryN
# =============================================================================

class TestDirectRefactorEveryN:
    """Regression tests for the DirectRefactorEveryN / PolicyRefactorEveryN policy."""

    # -------------------------------------------------------------------------
    # Enum / property API
    # -------------------------------------------------------------------------

    def test_enum_exists(self):
        """ContingencySolverType.DirectRefactorEveryN must be present."""
        from gpusim2grid._gpusim2grid import ContingencySolverType
        assert hasattr(ContingencySolverType, "DirectRefactorEveryN")

    def test_default_refactor_period(self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """refactor_period defaults to 1."""
        from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver
        d = ieee14_base_case
        solver = ContingencyAnalysisSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=_BATCH_SIZE, nb_iter=_NB_ITER)
        assert solver.refactor_period == 1

    def test_refactor_period_setter(self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """refactor_period setter stores the value."""
        from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver
        d = ieee14_base_case
        solver = ContingencyAnalysisSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=_BATCH_SIZE, nb_iter=_NB_ITER)
        solver.refactor_period = 3
        assert solver.refactor_period == 3

    # -------------------------------------------------------------------------
    # Refactorize counts
    # -------------------------------------------------------------------------

    def test_n1_refactorize_count_matches_dre(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """DirectRefactorEveryN with period=1 gives same n_refactorize as DirectRefactorEvery."""
        n_ctg    = len(n1_branch_ids)
        n_chunks = (n_ctg + _BATCH_SIZE - 1) // _BATCH_SIZE

        _, _, t_dren = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                            strategy='direct_refactor_every_n', refactor_period=1)
        _, _, t_dre  = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                            strategy='direct_refactor_every')

        assert t_dren.n_refactorize == t_dre.n_refactorize, (
            f"period=1 should give same n_refactorize as DirectRefactorEvery: "
            f"{t_dren.n_refactorize} vs {t_dre.n_refactorize}"
        )
        expected = _expected_n_refactorize(n_chunks, _NB_ITER, period=1)
        assert t_dren.n_refactorize == expected, (
            f"period=1: expected {expected} refactorizations, got {t_dren.n_refactorize}"
        )

    def test_refactorize_count_period2(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """period=2: n_refactorize matches floor((n_chunks*nb_iter - 1) / 2)."""
        n_ctg    = len(n1_branch_ids)
        n_chunks = (n_ctg + _BATCH_SIZE - 1) // _BATCH_SIZE
        period   = 2

        _, _, timings = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                             strategy='direct_refactor_every_n', refactor_period=period)

        expected = _expected_n_refactorize(n_chunks, _NB_ITER, period)
        assert timings.n_refactorize == expected, (
            f"period={period}: expected {expected} refactorizations, "
            f"got {timings.n_refactorize}"
        )

    def test_refactorize_count_period5(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """period=5: n_refactorize matches floor((n_chunks*nb_iter - 1) / 5)."""
        n_ctg    = len(n1_branch_ids)
        n_chunks = (n_ctg + _BATCH_SIZE - 1) // _BATCH_SIZE
        period   = 5

        _, _, timings = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                             strategy='direct_refactor_every_n', refactor_period=period)

        expected = _expected_n_refactorize(n_chunks, _NB_ITER, period)
        assert timings.n_refactorize == expected, (
            f"period={period}: expected {expected} refactorizations, "
            f"got {timings.n_refactorize}"
        )

    def test_refactorize_count_decreases_with_period(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """n_refactorize is non-increasing as refactor_period grows."""
        _, _, t1 = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                        strategy='direct_refactor_every_n', refactor_period=1)
        _, _, t2 = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                        strategy='direct_refactor_every_n', refactor_period=2)
        _, _, t5 = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                        strategy='direct_refactor_every_n', refactor_period=5)

        assert t2.n_refactorize <= t1.n_refactorize, (
            f"period=2 should have ≤ refactorizations than period=1: "
            f"{t2.n_refactorize} > {t1.n_refactorize}"
        )
        assert t5.n_refactorize <= t2.n_refactorize, (
            f"period=5 should have ≤ refactorizations than period=2: "
            f"{t5.n_refactorize} > {t2.n_refactorize}"
        )

    # -------------------------------------------------------------------------
    # Correctness
    # -------------------------------------------------------------------------

    def test_residuals_finite(self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """All residuals must be finite for non-disconnected contingencies (period=2)."""
        V, residuals, _ = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                               strategy='direct_refactor_every_n', refactor_period=2)

        finite_V = np.isfinite(V).all(axis=1)
        assert np.all(np.isfinite(residuals[finite_V])), (
            "Non-finite residuals for simulated contingencies (DirectRefactorEveryN, period=2)"
        )

    def test_voltages_finite(self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """Voltages must be finite for at least some contingencies (period=2)."""
        V, _, _ = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                       strategy='direct_refactor_every_n', refactor_period=2)

        assert np.isfinite(V).all(axis=1).any(), (
            "All voltages non-finite for DirectRefactorEveryN period=2 — solver diverged"
        )

    def test_voltages_close_to_dre(self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """period=2 voltages stay within 5e-2 pu of DirectRefactorEvery for converged cases."""
        V_dren, res_dren, _ = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                                   strategy='direct_refactor_every_n', refactor_period=2)
        V_dre,  res_dre,  _ = _run(ieee14_base_case, ieee14_grid, n1_branch_ids,
                                   strategy='direct_refactor_every')

        conv = (np.abs(res_dren) < _RESIDUAL_CONVERGED) & \
               (np.abs(res_dre)  < _RESIDUAL_CONVERGED)

        if not conv.any():
            pytest.skip("No contingency converged with both strategies")

        err = np.abs(V_dren[conv] - V_dre[conv]).max()
        assert err < 5e-2, (
            f"period=2 vs DirectRefactorEvery max |ΔV|={err:.3f} pu (limit 5e-2 pu)"
        )

    def test_tripped_branch_zero_flow(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """Tripped branch must carry 0 A in flow results (period=2)."""
        _, residuals, _, or_amps = _run(
            ieee14_base_case, ieee14_grid, n1_branch_ids,
            strategy='direct_refactor_every_n', refactor_period=2,
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
            "Tripped branch(es) carry non-zero flow (DirectRefactorEveryN, period=2):\n" +
            "\n".join(f"  contingency {c}, branch {b}: {f:.4f} A" for c, b, f in failures)
        )

    # -------------------------------------------------------------------------
    # Mutability between run() calls
    # -------------------------------------------------------------------------

    def test_refactor_period_mutable_between_runs(
            self, ieee14_base_case, ieee14_grid, n1_branch_ids):
        """Changing refactor_period between run() calls takes effect immediately."""
        from gpusim2grid.contingency_analysis import ContingencyAnalysisSolver

        d = ieee14_base_case
        lines  = ieee14_grid.get_lines()
        trafos = ieee14_grid.get_trafos()
        all_branches = list(lines) + list(trafos)

        branch_from = np.array([b.bus1_id for b in all_branches], dtype=np.int32)
        branch_to   = np.array([b.bus2_id for b in all_branches], dtype=np.int32)
        yff = np.array([b.yac_11 for b in all_branches], dtype=complex)
        yft = np.array([b.yac_12 for b in all_branches], dtype=complex)
        ytf = np.array([b.yac_21 for b in all_branches], dtype=complex)
        ytt = np.array([b.yac_22 for b in all_branches], dtype=complex)
        bus_vn_kv = ieee14_grid.get_bus_vn_kv().astype(float)

        solver = ContingencyAnalysisSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=_BATCH_SIZE, nb_iter=_NB_ITER)
        solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, 100.0)
        solver.build_contingencies(n1_branch_ids)
        solver.strategy = 'direct_refactor_every_n'

        solver.refactor_period = 1
        solver.run()
        r1 = solver.timings.n_refactorize

        solver.refactor_period = 5
        solver.run()
        r5 = solver.timings.n_refactorize

        assert r5 <= r1, (
            f"After changing period 1→5, n_refactorize should decrease: "
            f"{r5} > {r1}"
        )
