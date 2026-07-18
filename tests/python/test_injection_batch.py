# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_injection_batch.py — GPU batched-injection PF correctness.

Builds a small sweep of scaled-load scenarios on the IEEE 14-bus grid,
runs them through the new acpf_nr_gpu_injection entry point, and validates:

  1. Per-scenario ‖F(V_gpu)‖∞ residual is below the fp32 tolerance.
  2. Per-scenario |V_gpu| matches the lightsim2grid AC PF reference.
  3. The unit-scale scenario (scale = 1.0) reproduces the base-case voltages.

The strategy is fixed to DirectRefactorEvery in C++ for this phase.
"""
import numpy as np
import pytest

from conftest import requires_gpu, compute_pq_mismatch


pytestmark = requires_gpu


_RESIDUAL_TOL    = 1e-4   # fp32 tolerance, matches contingency suite
_VOLTAGE_ATOL    = 5e-4   # |V_gpu| vs lightsim2grid reference
_BASE_VOLTAGE_TOL = 5e-5  # tighter for the unit-scale scenario


def _build_scenarios(ieee14_base_case, scales):
    """Return (p_mw, q_mvar) arrays of shape (n_scen, n_bus) in MW / MVAr."""
    d = ieee14_base_case
    sn_mva = d["grid"].get_sn_mva()
    n_bus = d["n_bus"]
    n_scen = len(scales)

    # Convert per-unit base Sbus → MW / MVAr then scale per scenario.
    p_base_mw   = d["Sbus"].real * sn_mva
    q_base_mvar = d["Sbus"].imag * sn_mva

    p_mw   = np.zeros((n_scen, n_bus), dtype=np.float64)
    q_mvar = np.zeros((n_scen, n_bus), dtype=np.float64)
    for s, scale in enumerate(scales):
        p_mw[s, :]   = p_base_mw   * scale
        q_mvar[s, :] = q_base_mvar * scale

    return p_mw, q_mvar, sn_mva


def _lightsim_reference(ieee14_grid, ieee14_base_case, scales):
    """Per-scenario AC PF voltages — CPU reference solved hand-rolled in NumPy.

    lightsim2grid's ac_pf doesn't expose a per-unit-Sbus override on the
    fly, so we run the equivalent dense Newton-Raphson directly with NumPy.
    The grid is small enough (IEEE 14-bus) that dense linalg is fine.
    """
    d = ieee14_base_case
    n_bus = d["n_bus"]
    n_scen = len(scales)

    V_ref = np.zeros((n_scen, n_bus), dtype=complex)

    Ybus  = d["Ybus"]
    pv    = d["pv"]
    pq    = d["pq"]
    pvpq  = np.r_[pv, pq]
    n_pv  = len(pv)
    n_pq  = len(pq)

    for s, scale in enumerate(scales):
        V = d["v_init"].copy()
        Sbus_s = d["Sbus"] * scale

        for _ in range(40):
            mis = V * np.conj(Ybus @ V) - Sbus_s
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.abs(F).max() < 1e-10:
                break

            # Build Jacobian dS/dVa, dS/dVm in CSR (small grid — dense is fine).
            Ibus = Ybus @ V
            diagV     = np.diag(V)
            diagIbus  = np.diag(Ibus)
            diagVnorm = np.diag(V / np.abs(V))
            Yh = Ybus.toarray() if hasattr(Ybus, "toarray") else Ybus

            dSdVm = diagV @ np.conj(Yh @ diagVnorm) + np.conj(diagIbus) @ diagVnorm
            dSdVa = 1j * diagV @ np.conj(diagIbus - Yh @ diagV)

            J11 = dSdVa[np.ix_(pvpq, pvpq)].real
            J12 = dSdVm[np.ix_(pvpq, pq)].real
            J21 = dSdVa[np.ix_(pq,    pvpq)].imag
            J22 = dSdVm[np.ix_(pq,    pq)].imag
            J = np.block([[J11, J12], [J21, J22]])

            dx = np.linalg.solve(J, -F)
            dVa = np.zeros(len(V))
            dVm = np.zeros(len(V))
            dVa[pvpq] = dx[:n_pv + n_pq]
            dVm[pq]   = dx[n_pv + n_pq:]
            V = V * (1 + dVm / np.abs(V)) * np.exp(1j * dVa)

        V_ref[s, :] = V

    return V_ref


def _run_injection_gpu(ieee14_base_case, p_mw, q_mvar, sn_mva,
                       batch_size, nb_iter=10, strategy=None,
                       refactor_period=1):
    from gpusim2grid._gpusim2grid import (
        acpf_nr_gpu_injection,
        ContingencySolverType,
    )
    d = ieee14_base_case
    kwargs = {}
    if strategy is not None:
        kwargs["strategy_type"]   = strategy
        kwargs["refactor_period"] = refactor_period
    V_flat, residuals, timings = acpf_nr_gpu_injection(
        d["Ybus"],
        d["v_init"].copy(),
        d["Sbus"],
        d["slack"],
        d["slack_weights"],
        d["pv"],
        d["pq"],
        p_mw,
        q_mvar,
        sn_mva,
        batch_size,
        nb_iter,
        10,    # max_iter_base
        1e-6,  # tol_base
        **kwargs,
    )
    n_scen = p_mw.shape[0]
    V_2d = V_flat.reshape(n_scen, d["n_bus"])
    return V_2d, residuals, timings


class TestInjectionSweepCorrectness:
    """Validate per-scenario residual and voltage agreement vs CPU reference."""

    def test_residuals_below_fp32_tolerance(self, ieee14_grid, ieee14_base_case):
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        V_gpu, residuals, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva, batch_size=5, nb_iter=10)
        max_r = np.abs(residuals).max()
        assert max_r < _RESIDUAL_TOL, (
            f"Max GPU residual across scenarios: {max_r:.2e} (tol {_RESIDUAL_TOL:.0e})"
        )

    def test_voltages_match_cpu_reference(self, ieee14_grid, ieee14_base_case):
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        V_gpu, _, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva, batch_size=5, nb_iter=10)
        V_ref = _lightsim_reference(ieee14_grid, ieee14_base_case, scales)

        err = np.abs(np.abs(V_gpu) - np.abs(V_ref))
        assert err.max() < _VOLTAGE_ATOL, (
            f"Max |V_gpu| - |V_ref| across scenarios: {err.max():.2e} pu "
            f"(tol {_VOLTAGE_ATOL:.0e})"
        )

    def test_unit_scale_reproduces_base_case(self, ieee14_grid, ieee14_base_case):
        scales = [0.8, 1.0, 1.2]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        V_gpu, _, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva, batch_size=3, nb_iter=10)

        V_base_ref = ieee14_base_case["v_ref"]
        err = np.abs(V_gpu[1, :] - V_base_ref)
        assert err.max() < _BASE_VOLTAGE_TOL, (
            f"Unit-scale scenario differs from base case by {err.max():.2e}"
        )

    def test_per_scenario_recomputed_residual(self, ieee14_grid, ieee14_base_case):
        """Hand-compute ‖F‖∞ on V_gpu for each scenario; agree with GPU residual."""
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        V_gpu, residuals, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva, batch_size=5, nb_iter=10)

        d = ieee14_base_case
        for s, scale in enumerate(scales):
            Sbus_s = d["Sbus"] * scale
            F = compute_pq_mismatch(d["Ybus"], V_gpu[s, :], Sbus_s,
                                    d["pv"], d["pq"])
            recomputed = np.abs(F).max()
            assert recomputed < _RESIDUAL_TOL, (
                f"Recomputed residual for scenario {s} (scale={scale}): "
                f"{recomputed:.2e}"
            )


# ---------------------------------------------------------------------------
# Strategy coverage — all four linear-solve strategies must work with the
# InjectionBatch driver, with no new C++ solver code.
# ---------------------------------------------------------------------------

_STRATEGY_VOLTAGE_ATOL = {
    "DirectRefactorEvery":   5e-4,
    "DirectIter0Only":       5e-4,
    "DirectBaseCaseFactors": 5e-2,   # uses base-case J — coarser approximation
    "DirectRefactorEveryN":  5e-4,
}


class TestInjectionStrategies:
    """Each linear-solve strategy must converge and match DirectRefactorEvery
    (or, for DirectBaseCaseFactors, a relaxed tolerance reflecting the
    base-case-Jacobian approximation)."""

    def _scenarios(self, d):
        # Smaller perturbation than the correctness test — DirectBaseCaseFactors
        # uses base-case factors, so very-different operating points stress it.
        return [0.9, 0.95, 1.0, 1.05, 1.1]

    @pytest.mark.parametrize("strategy_name", [
        "DirectRefactorEvery",
        "DirectIter0Only",
        "DirectBaseCaseFactors",
        "DirectRefactorEveryN",
    ])
    def test_strategy_residuals_finite(
        self, ieee14_grid, ieee14_base_case, strategy_name
    ):
        from gpusim2grid._gpusim2grid import ContingencySolverType
        scales = self._scenarios(ieee14_base_case)
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)

        strategy = getattr(ContingencySolverType, strategy_name)
        refactor_period = 2 if strategy_name == "DirectRefactorEveryN" else 1

        V_gpu, residuals, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva,
            batch_size=5, nb_iter=10,
            strategy=strategy, refactor_period=refactor_period,
        )
        assert np.all(np.isfinite(residuals)), (
            f"Non-finite residuals with strategy {strategy_name}"
        )
        assert np.all(np.isfinite(V_gpu)), (
            f"Non-finite voltages with strategy {strategy_name}"
        )

    @pytest.mark.parametrize("strategy_name", [
        "DirectRefactorEvery",
        "DirectIter0Only",
        "DirectBaseCaseFactors",
        "DirectRefactorEveryN",
    ])
    def test_strategy_matches_reference(
        self, ieee14_grid, ieee14_base_case, strategy_name
    ):
        from gpusim2grid._gpusim2grid import ContingencySolverType
        scales = self._scenarios(ieee14_base_case)
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)

        strategy = getattr(ContingencySolverType, strategy_name)
        refactor_period = 2 if strategy_name == "DirectRefactorEveryN" else 1

        V_gpu, _, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva,
            batch_size=5, nb_iter=10,
            strategy=strategy, refactor_period=refactor_period,
        )
        V_ref = _lightsim_reference(ieee14_grid, ieee14_base_case, scales)

        atol = _STRATEGY_VOLTAGE_ATOL[strategy_name]
        err = np.abs(np.abs(V_gpu) - np.abs(V_ref))
        assert err.max() < atol, (
            f"strategy={strategy_name}: max |V_gpu|-|V_ref| = {err.max():.2e} "
            f"pu (tol {atol:.0e})"
        )

    def test_dio_fewer_refactorize_than_dre(
        self, ieee14_grid, ieee14_base_case
    ):
        """DirectIter0Only refactorises at most once per chunk; should be
        strictly fewer than DirectRefactorEvery on >1 iteration runs."""
        from gpusim2grid._gpusim2grid import ContingencySolverType
        scales = self._scenarios(ieee14_base_case)
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)

        _, _, t_dre = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva,
            batch_size=5, nb_iter=5,
            strategy=ContingencySolverType.DirectRefactorEvery,
        )
        _, _, t_dio = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva,
            batch_size=5, nb_iter=5,
            strategy=ContingencySolverType.DirectIter0Only,
        )
        assert t_dio.n_refactorize < t_dre.n_refactorize, (
            f"DirectIter0Only should refactorise less than DirectRefactorEvery: "
            f"{t_dio.n_refactorize} vs {t_dre.n_refactorize}"
        )

    def test_dbcf_no_refactorize(
        self, ieee14_grid, ieee14_base_case
    ):
        """DirectBaseCaseFactors must never refactorise (factor is reused)."""
        from gpusim2grid._gpusim2grid import ContingencySolverType
        scales = self._scenarios(ieee14_base_case)
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)

        _, _, t = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva,
            batch_size=5, nb_iter=5,
            strategy=ContingencySolverType.DirectBaseCaseFactors,
        )
        assert t.n_refactorize == 0, (
            f"DirectBaseCaseFactors should have n_refactorize=0, "
            f"got {t.n_refactorize}"
        )


# ---------------------------------------------------------------------------
# Python facade — gpusim2grid.injection_sweep
#
# Validates the string→enum strategy resolution, the kwarg forwarding, and
# that the facade returns identical results to the raw _gpusim2grid binding.
# ---------------------------------------------------------------------------

class TestInjectionSweepFacade:
    def test_facade_string_strategy(self, ieee14_base_case):
        """String strategy names must be accepted and produce valid results."""
        from gpusim2grid.injection_sweep import acpf_nr_gpu_injection

        scales = [0.9, 1.0, 1.1]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        d = ieee14_base_case

        V_flat, residuals, _ = acpf_nr_gpu_injection(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            p_mw, q_mvar, sn_mva,
            batch_size=3, nb_iter=10,
            strategy='direct_iter0_only',
        )
        assert V_flat.shape == (3 * d["n_bus"],)
        assert np.all(np.isfinite(residuals))
        assert np.abs(residuals).max() < _RESIDUAL_TOL

    def test_facade_enum_strategy(self, ieee14_base_case):
        """Passing the raw ContingencySolverType enum value must also work."""
        from gpusim2grid.injection_sweep import acpf_nr_gpu_injection
        from gpusim2grid._gpusim2grid import ContingencySolverType

        scales = [0.9, 1.0, 1.1]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        d = ieee14_base_case

        V_flat, residuals, _ = acpf_nr_gpu_injection(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            p_mw, q_mvar, sn_mva,
            batch_size=3, nb_iter=10,
            strategy=ContingencySolverType.DirectIter0Only,
        )
        assert np.all(np.isfinite(residuals))

    def test_facade_unknown_strategy_raises(self, ieee14_base_case):
        """An unknown strategy name must raise ValueError."""
        from gpusim2grid.injection_sweep import acpf_nr_gpu_injection

        scales = [1.0]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        d = ieee14_base_case

        with pytest.raises(ValueError, match="Unknown strategy"):
            acpf_nr_gpu_injection(
                d["Ybus"], d["v_init"].copy(), d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                p_mw, q_mvar, sn_mva,
                batch_size=1, nb_iter=4,
                strategy='not_a_real_strategy',
            )

    def test_facade_matches_raw_binding(self, ieee14_base_case):
        """Facade with default strategy must match the raw _gpusim2grid call
        bit-identically (same C++ entry point, same args, deterministic in FP32
        within reproducibility tolerance, identical in FP64)."""
        from gpusim2grid._gpusim2grid import (
            acpf_nr_gpu_injection as raw_call,
            ContingencySolverType,
        )
        from gpusim2grid.injection_sweep import acpf_nr_gpu_injection as facade

        scales = [0.9, 1.0, 1.1]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        d = ieee14_base_case

        V_raw, res_raw, _ = raw_call(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            p_mw, q_mvar, sn_mva, 3, 10, 10, 1e-6,
            ContingencySolverType.DirectRefactorEvery, 1,
        )
        V_fac, res_fac, _ = facade(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            p_mw, q_mvar, sn_mva,
            batch_size=3, nb_iter=10,
            strategy='direct_refactor_every',
        )
        # GPU NR is reproducible up to FP32 epsilon (~1e-7); both paths invoke
        # the same C++ entry point with the same arguments.
        assert np.abs(V_raw - V_fac).max() < 1e-5, (
            "Facade and raw binding disagree on V_results"
        )
        assert np.abs(res_raw - res_fac).max() < 1e-5

    def test_run_injection_sweep_gpu_alias(self, ieee14_base_case):
        """run_injection_sweep_gpu is an alias for acpf_nr_gpu_injection."""
        from gpusim2grid.injection_sweep import (
            acpf_nr_gpu_injection,
            run_injection_sweep_gpu,
        )
        assert run_injection_sweep_gpu is acpf_nr_gpu_injection


# ---------------------------------------------------------------------------
# Stateful solver — _InjectionSweepSolver
#
# Validates that the stateful wrapper reuses the base case across runs and
# produces results identical to the one-shot entry point.
# ---------------------------------------------------------------------------

class TestInjectionSweepSolver:
    def _make_solver(self, ieee14_base_case, batch_size=5, nb_iter=10):
        from gpusim2grid.injection_sweep import _InjectionSweepSolver
        d = ieee14_base_case
        return _InjectionSweepSolver(
            d["Ybus"], d["v_init"].copy(), d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            batch_size=batch_size, nb_iter=nb_iter,
            max_iter_base=10, tol_base=1e-6,
        )

    def test_run_matches_one_shot(self, ieee14_base_case):
        """Stateful solver result must match the one-shot acpf_nr_gpu_injection."""
        scales = [0.9, 1.0, 1.1]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        d = ieee14_base_case

        solver = self._make_solver(ieee14_base_case, batch_size=3, nb_iter=10)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.run()
        V_stateful = solver.V_results.to_numpy()
        res_stateful = solver.residuals.to_numpy()

        V_oneshot, res_oneshot, _ = _run_injection_gpu(
            ieee14_base_case, p_mw, q_mvar, sn_mva, batch_size=3, nb_iter=10)

        assert np.abs(V_stateful.reshape(3, d["n_bus"]) - V_oneshot).max() < 1e-5
        assert np.abs(res_stateful - res_oneshot).max() < 1e-5

    def test_metadata_and_residuals(self, ieee14_base_case):
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, scales)
        solver = self._make_solver(ieee14_base_case, batch_size=5, nb_iter=10)
        solver.set_injections(p_mw, q_mvar, sn_mva)
        solver.run()

        assert solver.n_scenarios == 5
        assert solver.n_bus == ieee14_base_case["n_bus"]
        res = solver.residuals.to_numpy()
        assert res.shape == (5,)
        assert np.all(np.isfinite(res))
        assert np.abs(res).max() < _RESIDUAL_TOL

    def test_rerun_with_new_injections(self, ieee14_base_case):
        """A second set_injections + run on the same solver must work and
        reflect the new scenario count, reusing the base case."""
        solver = self._make_solver(ieee14_base_case, batch_size=4, nb_iter=10)

        p1, q1, sn = _build_scenarios(ieee14_base_case, [0.9, 1.0, 1.1])
        solver.set_injections(p1, q1, sn)
        solver.run()
        assert solver.n_scenarios == 3

        p2, q2, _ = _build_scenarios(ieee14_base_case, [0.85, 0.95, 1.05, 1.15])
        solver.set_injections(p2, q2, sn)
        solver.run()
        assert solver.n_scenarios == 4
        assert np.all(np.isfinite(solver.residuals.to_numpy()))

    def test_strategy_switch_between_runs(self, ieee14_base_case):
        """Switching strategy between runs must take effect and converge."""
        p_mw, q_mvar, sn_mva = _build_scenarios(ieee14_base_case, [0.9, 1.0, 1.1])
        solver = self._make_solver(ieee14_base_case, batch_size=3, nb_iter=10)
        solver.set_injections(p_mw, q_mvar, sn_mva)

        solver.strategy = 'direct_refactor_every'
        solver.run()
        n_ref_dre = solver.timings.n_refactorize

        solver.strategy = 'direct_iter0_only'
        solver.run()
        n_ref_dio = solver.timings.n_refactorize

        assert np.all(np.isfinite(solver.residuals.to_numpy()))
        assert n_ref_dio < n_ref_dre, (
            f"DirectIter0Only should refactorise less: {n_ref_dio} vs {n_ref_dre}"
        )

    def test_run_before_set_injections_raises(self, ieee14_base_case):
        solver = self._make_solver(ieee14_base_case)
        with pytest.raises(Exception):
            solver.run()
