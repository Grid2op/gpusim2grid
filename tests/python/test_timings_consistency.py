# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_timings_consistency.py — solver.timings must be a complete accounting.

Wrapping construction -> run() -> compute_flows() -> .to_numpy() calls in an
external stopwatch should not report meaningfully more elapsed time than
timings.t_grand_total_ms; and the coarse aggregation properties
(t_cpu_preprocess_ms, t_host_to_device_ms, t_context_init_ms,
t_gpu_compute_ms, t_device_to_host_ms, t_grand_total_ms) must equal the sum of
their documented components exactly (these are computed fields, no measurement
noise).

A one-time throwaway solve warms up the CUDA context/stream before each
timed measurement: CUDA context creation on a process' first GPU touch is a
one-time, process-lifetime cost (not attributable to any single solve), and
would otherwise make the external-vs-internal comparison spuriously loose.
(gpusim2grid.warmup() does the same thing without a solve; the throwaway solve
is kept here because it also warms this module's own kernels.)
"""
import time

import numpy as np
import pytest

from conftest import requires_gpu, branch_data_arrays

pytestmark = requires_gpu

# Upper-bound slack on (external - t_grand_total_ms): generous enough to
# absorb Python/pybind11 call overhead and measurement noise, but tight
# enough to catch a real untimed gap (e.g. a completely unaccounted D→H
# copy, which is on the order of the whole call, not a few ms/percent).
_MIN_SLACK_MS = 10.0
_REL_SLACK = 0.25

# The external stopwatch must not measure LESS than the internal total by
# more than trivial clock-resolution noise.
_LOWER_EPS_MS = 1.0


def _assert_bracket(external_ms, grand_total_ms):
    assert external_ms >= grand_total_ms - _LOWER_EPS_MS, (
        f"external={external_ms:.3f}ms < t_grand_total_ms={grand_total_ms:.3f}ms "
        "(minus noise floor); a C++ timer cannot exceed an external stopwatch "
        "that fully encloses it -- some GPU/CPU work is unaccounted for.")
    slack = max(_MIN_SLACK_MS, _REL_SLACK * grand_total_ms)
    assert external_ms <= grand_total_ms + slack, (
        f"external={external_ms:.3f}ms far exceeds t_grand_total_ms="
        f"{grand_total_ms:.3f}ms (slack={slack:.3f}ms) -- some real wall-clock "
        "work is happening outside every timed region.")


def _check_batch_aggregation(t):
    assert t.t_cpu_preprocess_ms == pytest.approx(t.t_preprocess_ms, abs=1e-9)
    assert t.t_host_to_device_ms == pytest.approx(
        t.t_alloc_ms + t.t_source_init_ms + t.t_branch_data_upload_ms
        + t.t_violation_setup_ms, abs=1e-9)
    assert t.t_device_to_host_ms == pytest.approx(
        t.t_copy_flows_to_host_ms + t.t_copy_V_to_host_ms
        + t.t_copy_residuals_to_host_ms + t.t_copy_violations_to_host_ms, abs=1e-9)
    assert t.t_gpu_compute_ms == pytest.approx(
        t.t_base_case_solve_only_ms + t.t_analysis_ms + t.t_chunks_total_wall_ms,
        abs=1e-9)
    # t_context_init_ms is its own bucket: CUDA/cuSPARSE/cuDSS one-time init is
    # process warm-up, not this batch's compute, so it sits outside
    # t_gpu_compute_ms while still counting toward the grand total.
    assert t.t_grand_total_ms == pytest.approx(
        t.t_cpu_preprocess_ms + t.t_host_to_device_ms + t.t_context_init_ms
        + t.t_gpu_compute_ms + t.t_device_to_host_ms, abs=1e-9)


def _check_to_dict(t):
    d = t.to_dict()
    assert d["total"] == pytest.approx(t.t_grand_total_ms, abs=1e-9)

    assert d["context_init"]["total"] == pytest.approx(t.t_context_init_ms, abs=1e-9)

    assert d["cpu_preproc"]["total"] == pytest.approx(t.t_cpu_preprocess_ms, abs=1e-9)
    assert d["cpu_preproc"]["preprocess_ms"] == pytest.approx(t.t_preprocess_ms, abs=1e-9)

    assert d["h2d"]["total"] == pytest.approx(t.t_host_to_device_ms, abs=1e-9)
    assert d["h2d"]["alloc_ms"] == pytest.approx(t.t_alloc_ms, abs=1e-9)
    assert d["h2d"]["source_init_ms"] == pytest.approx(t.t_source_init_ms, abs=1e-9)
    assert d["h2d"]["branch_data_upload_ms"] == pytest.approx(t.t_branch_data_upload_ms, abs=1e-9)
    assert d["h2d"]["violation_setup_ms"] == pytest.approx(t.t_violation_setup_ms, abs=1e-9)

    assert d["gpu_compute"]["total"] == pytest.approx(t.t_gpu_compute_ms, abs=1e-9)
    assert d["gpu_compute"]["base_case_solve_only_ms"] == pytest.approx(
        t.t_base_case_solve_only_ms, abs=1e-9)
    assert d["gpu_compute"]["analysis_ms"] == pytest.approx(t.t_analysis_ms, abs=1e-9)
    assert d["gpu_compute"]["spmv"]["wall_ms"] == pytest.approx(t.t_spmv.wall_ms, abs=1e-9)
    assert d["gpu_compute"]["spmv"]["gpu_ms"] == pytest.approx(t.t_spmv.gpu_ms, abs=1e-9)
    assert d["gpu_compute"]["fill_J"]["wall_ms"] == pytest.approx(t.t_fill_J.wall_ms, abs=1e-9)
    assert d["gpu_compute"]["fill_J"]["gpu_ms"] == pytest.approx(t.t_fill_J.gpu_ms, abs=1e-9)

    assert d["d2h"]["total"] == pytest.approx(t.t_device_to_host_ms, abs=1e-9)
    assert d["d2h"]["copy_flows_to_host_ms"] == pytest.approx(t.t_copy_flows_to_host_ms, abs=1e-9)
    assert d["d2h"]["copy_V_to_host_ms"] == pytest.approx(t.t_copy_V_to_host_ms, abs=1e-9)
    assert d["d2h"]["copy_residuals_to_host_ms"] == pytest.approx(
        t.t_copy_residuals_to_host_ms, abs=1e-9)
    assert d["d2h"]["copy_violations_to_host_ms"] == pytest.approx(
        t.t_copy_violations_to_host_ms, abs=1e-9)


def _check_acpf_aggregation(t):
    # t_build_J_ms / t_upload_ms aren't exposed to Python (only their
    # t_cpu_preprocess_ms / t_host_to_device_ms aliases are) -- their
    # correctness is guaranteed by construction in C++. Check what's testable
    # from Python: t_device_to_host_ms's alias, and the two composite sums.
    assert t.t_device_to_host_ms == pytest.approx(t.t_copy_v_to_host_ms, abs=1e-9)
    iters_wall = (t.t_spmv.wall_ms + t.t_fill_F.wall_ms + t.t_fill_J.wall_ms
                  + t.t_first_factorize.wall_ms + t.t_refactorize.wall_ms
                  + t.t_solve.wall_ms + t.t_update_V.wall_ms + t.t_mismatch.wall_ms)
    assert t.t_gpu_compute_ms == pytest.approx(
        t.t_analyze_ms + t.t_prepare_jt_ms + iters_wall, abs=1e-9)
    # t_context_init_ms is its own bucket: CUDA/cuSPARSE/cuDSS one-time init is
    # process warm-up, not this solve's compute, so it sits outside
    # t_gpu_compute_ms while still counting toward the grand total.
    assert t.t_grand_total_ms == pytest.approx(
        t.t_cpu_preprocess_ms + t.t_host_to_device_ms + t.t_context_init_ms
        + t.t_gpu_compute_ms + t.t_device_to_host_ms, abs=1e-9)


# ---------------------------------------------------------------------------
# Single AC power flow (AcPfNrSession) -- nb_iter/converged are meaningful
# here for presolved_v, unlike on the batch solvers' BatchTimings.
# ---------------------------------------------------------------------------

class TestAcPfTimingsConsistency:
    @pytest.mark.parametrize("presolved_v", [False, True])
    def test_grand_total_matches_external_stopwatch(
            self, ieee14_base_case, presolved_v):
        from gpusim2grid._gpusim2grid import AcPfNrSession

        d = ieee14_base_case
        v_seed = d["v_ref"].copy() if presolved_v else d["v_init"].copy()

        # Warm up the CUDA context/stream once (see module docstring).
        AcPfNrSession(d["Ybus"], d["v_init"].copy(), d["Sbus"],
                      d["slack"], d["slack_weights"], d["pv"], d["pq"],
                      10, 1e-8, -1).get_v()

        t0 = time.perf_counter()
        sess = AcPfNrSession(d["Ybus"], v_seed, d["Sbus"],
                             d["slack"], d["slack_weights"], d["pv"], d["pq"],
                             10, 1e-8, -1, presolved_v=presolved_v)
        _ = sess.get_v()
        external_ms = (time.perf_counter() - t0) * 1000.0

        t = sess.timings
        _check_acpf_aggregation(t)
        _assert_bracket(external_ms, t.t_grand_total_ms)

        if presolved_v:
            assert t.nb_iter == 0
            assert t.n_refactorize == 0
            assert t.converged


# ---------------------------------------------------------------------------
# _ContingencyAnalysisSolver
# ---------------------------------------------------------------------------

class TestContingencyTimingsConsistency:
    @pytest.mark.parametrize("presolved_v", [False, True])
    def test_grand_total_matches_external_stopwatch(
            self, ieee14_grid, ieee14_base_case, presolved_v):
        from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver

        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_ctg = n_lines + n_trafos
        cont_branch_ids = [[c] for c in range(n_ctg)]
        v_seed = d["v_ref"].copy() if presolved_v else d["v_init"].copy()

        def _build_and_run(seed, presolved):
            solver = _ContingencyAnalysisSolver(
                d["Ybus"], seed, d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                batch_size=5, nb_iter=10, max_iter_base=10, tol_base=1e-6,
                presolved_v=presolved,
            )
            solver.set_branch_data(*branch_data)
            solver.build_contingencies(cont_branch_ids)
            solver.run()
            solver.compute_flows()
            solver.V_results.to_numpy()
            solver.residuals.to_numpy()
            solver.or_amps.to_numpy()
            solver.ex_amps.to_numpy()
            return solver

        # Warm up the CUDA context/stream once (see module docstring).
        _build_and_run(d["v_init"].copy(), False)

        t0 = time.perf_counter()
        solver = _build_and_run(v_seed, presolved_v)
        external_ms = (time.perf_counter() - t0) * 1000.0

        t = solver.timings
        _check_batch_aggregation(t)
        _check_to_dict(t)
        _assert_bracket(external_ms, t.t_grand_total_ms)

        # compute_limit_violations was never enabled in this run -- these
        # fields must be exact zero, not just small (they're gated on
        # _fused_violations_enabled, never entering their timed segment).
        assert t.t_violation_setup_ms == 0.0
        assert t.t_violation_check.wall_ms == 0.0
        assert t.t_copy_violations_to_host_ms == 0.0


# ---------------------------------------------------------------------------
# _ContingencyAnalysisSolver with compute_limit_violations enabled
# ---------------------------------------------------------------------------

class TestContingencyViolationTimingsConsistency:
    def test_violation_timings_are_populated_and_consistent(
            self, ieee14_grid, ieee14_base_case):
        from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver
        from conftest import branch_data_arrays

        d = ieee14_base_case
        branch_data, n_lines, n_trafos = branch_data_arrays(ieee14_grid)
        n_ctg = n_lines + n_trafos
        cont_branch_ids = [[c] for c in range(n_ctg)]
        n_bus = ieee14_base_case["n_bus"]

        def _build_and_run():
            solver = _ContingencyAnalysisSolver(
                d["Ybus"], d["v_init"].copy(), d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                batch_size=5, nb_iter=10, max_iter_base=10, tol_base=1e-6,
            )
            solver.set_branch_data(*branch_data)
            solver.build_contingencies(cont_branch_ids)
            n_branches_total = solver._s.n_branches
            solver.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                               np.full(n_branches_total, np.nan),
                               np.full(n_branches_total, np.nan), n_lines)
            solver.compute_limit_violations = True
            solver.run()
            solver.compute_flows()
            solver.V_results.to_numpy()
            solver.residuals.to_numpy()
            solver.or_amps.to_numpy()
            solver.ex_amps.to_numpy()
            # Exercise the D->H violation accessors (covers all 10 underlying
            # get_violation_*() calls via the Python wrapper).
            solver.get_violations()
            solver.get_violations_truncated()
            solver.get_violation_counts()
            return solver

        # Warm up the CUDA context/stream once (see module docstring).
        _build_and_run()

        t0 = time.perf_counter()
        solver = _build_and_run()
        external_ms = (time.perf_counter() - t0) * 1000.0

        t = solver.timings
        _check_batch_aggregation(t)
        _check_to_dict(t)
        _assert_bracket(external_ms, t.t_grand_total_ms)

        assert t.t_violation_setup_ms > 0.0
        assert t.t_violation_check.wall_ms > 0.0
        assert t.t_copy_violations_to_host_ms > 0.0


# ---------------------------------------------------------------------------
# _InjectionSweepSolver
# ---------------------------------------------------------------------------

class TestInjectionTimingsConsistency:
    @pytest.mark.parametrize("presolved_v", [False, True])
    def test_grand_total_matches_external_stopwatch(
            self, ieee14_grid, ieee14_base_case, presolved_v):
        from gpusim2grid.injection_sweep import _InjectionSweepSolver

        d = ieee14_base_case
        branch_data, _, _ = branch_data_arrays(ieee14_grid)
        n_bus = d["n_bus"]
        sn_mva = d["grid"].get_sn_mva()
        rng = np.random.default_rng(0)
        n_scen = 8
        p_mw = rng.uniform(-30, 30, (n_scen, n_bus))
        q_mvar = rng.uniform(-10, 10, (n_scen, n_bus))
        v_seed = d["v_ref"].copy() if presolved_v else d["v_init"].copy()

        def _build_and_run(seed, presolved):
            solver = _InjectionSweepSolver(
                d["Ybus"], seed, d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                batch_size=5, nb_iter=10, max_iter_base=10, tol_base=1e-6,
                presolved_v=presolved,
            )
            solver.set_injections(p_mw, q_mvar, sn_mva)
            solver.run()
            solver.set_branch_data(*branch_data)
            solver.compute_flows()
            solver.V_results.to_numpy()
            solver.residuals.to_numpy()
            solver.or_amps.to_numpy()
            solver.ex_amps.to_numpy()
            return solver

        # Warm up the CUDA context/stream once (see module docstring).
        _build_and_run(d["v_init"].copy(), False)

        t0 = time.perf_counter()
        solver = _build_and_run(v_seed, presolved_v)
        external_ms = (time.perf_counter() - t0) * 1000.0

        t = solver.timings
        _check_batch_aggregation(t)
        _check_to_dict(t)
        _assert_bracket(external_ms, t.t_grand_total_ms)
