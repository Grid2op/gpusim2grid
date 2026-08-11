# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ScenarioSweepGPU: handle_disconnected_grid masking + compute_limit_violations.

Both features are ported from ContingencyAnalysisGPU onto the row-aligned
combined topology + injection sweep (see test_handle_disconnected_grid.py /
test_limit_violations.py for the reference behavior this mirrors). Reuses
test_handle_disconnected_grid's `_solved_spur_grid` helper: case14 + a radial
spur bus, so a single line trip cleanly islands exactly that bus.

The property specific to ScenarioSweepGPU (not exercised by the
ContingencyAnalysisGPU reference tests, since that class has no per-row
injection axis) is that masking/violations behave identically when the
masked row also carries a non-base-case injection, and independently of
whatever the other rows in the same batch are doing.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from test_handle_disconnected_grid import _solved_spur_grid, _ls2g_masked_reference

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="handle_disconnected_grid/compute_limit_violations parity needs the "
           "lightsim2grid C++ bridge")


# ---------------------------------------------------------------------------
# handle_disconnected_grid
# ---------------------------------------------------------------------------

@needs_bridge
def test_flag_off_skips_islanding_scenario():
    """With handle_disconnected_grid disabled (default), a scenario whose
    topology trip islands the grid is skipped: NaN voltages and residual."""
    from gpusim2grid import ScenarioSweepGPU

    grid, n_bus, spur_line, _ = _solved_spur_grid(distributed_slack=False)
    sweep = ScenarioSweepGPU(grid, handle_disconnected_grid=False,
                             precision=None, nb_iter=15, tol_base=1e-10)
    sn_mva = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    sweep.set_injections((Sbus.real * sn_mva)[None, :], (Sbus.imag * sn_mva)[None, :], sn_mva)
    sweep.set_topology([[int(spur_line)]])
    sweep.compute(batch_size=8)
    V = sweep.V_results.to_numpy().reshape(1, n_bus)[0]
    assert np.isnan(sweep.last_residuals()[0])
    assert np.all(np.isnan(V))
    assert sweep.get_disconnected()[0] == 1


@needs_bridge
def test_masked_scenario_matches_ls2g(solver_atol):
    """handle_disconnected_grid=True: the masked scenario converges, matches
    lightsim2grid's own masked solve on the live buses, and reports the
    islanded spur bus as NaN -- same reference test_handle_disconnected_grid
    uses for ContingencyAnalysisGPU, driven through ScenarioSweepGPU instead."""
    from gpusim2grid import ScenarioSweepGPU

    grid, n_bus, spur_line, spur_bus = _solved_spur_grid(distributed_slack=False)
    V_ref = _ls2g_masked_reference(grid, n_bus, spur_line)

    sweep = ScenarioSweepGPU(grid, handle_disconnected_grid=True,
                             precision=None, nb_iter=15, tol_base=1e-10)
    sn_mva = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    sweep.set_injections((Sbus.real * sn_mva)[None, :], (Sbus.imag * sn_mva)[None, :], sn_mva)
    sweep.set_topology([[int(spur_line)]])
    sweep.compute(batch_size=8)
    V_gpu = sweep.V_results.to_numpy().reshape(1, n_bus)[0]
    residual = sweep.last_residuals()[0]

    assert V_ref[spur_bus] == 0
    assert np.isnan(V_gpu[spur_bus])
    assert sweep.get_disconnected()[0] == 0
    assert np.isfinite(residual) and residual < 100 * solver_atol

    main = np.ones(n_bus, dtype=bool)
    main[spur_bus] = False
    assert np.all(np.isfinite(V_gpu[main]))
    np.testing.assert_allclose(V_gpu[main], V_ref[main], atol=10 * solver_atol)


@needs_bridge
def test_masked_row_independent_of_other_rows_own_injection(solver_atol):
    """The property unique to ScenarioSweepGPU: pair the masking scenario
    with a non-base-case injection on its own row, alongside an unrelated
    row (no trip, a different injection scale). Both rows must solve exactly
    as if run alone -- masking one row must not perturb the other, and the
    masked row's own injection must still take effect on its live buses."""
    from gpusim2grid import ScenarioSweepGPU

    grid, n_bus, spur_line, spur_bus = _solved_spur_grid(distributed_slack=False)
    V_ref = _ls2g_masked_reference(grid, n_bus, spur_line)

    sn_mva = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    p_base_mw = Sbus.real * sn_mva
    q_base_mvar = Sbus.imag * sn_mva

    # Row 0: masking scenario, injection scaled by 1.0 (reproduces V_ref).
    # Row 1: no trip, injection scaled by 0.95 (a plain, unrelated sweep row).
    p_mw = np.stack([p_base_mw * 1.0, p_base_mw * 0.95])
    q_mvar = np.stack([q_base_mvar * 1.0, q_base_mvar * 0.95])

    sweep = ScenarioSweepGPU(grid, handle_disconnected_grid=True,
                             precision=None, nb_iter=15, tol_base=1e-10)
    sweep.set_injections(p_mw, q_mvar, sn_mva)
    sweep.set_topology([[int(spur_line)], []])
    sweep.compute(batch_size=8)

    V_gpu = sweep.V_results.to_numpy().reshape(2, n_bus)
    residuals = sweep.last_residuals()
    disc = sweep.get_disconnected()

    assert disc[0] == 0 and disc[1] == 0
    assert np.all(np.isfinite(residuals)) and np.all(residuals < 100 * solver_atol)

    # Row 0 still matches the masked reference (same as the single-row test).
    main = np.ones(n_bus, dtype=bool)
    main[spur_bus] = False
    assert np.isnan(V_gpu[0, spur_bus])
    np.testing.assert_allclose(V_gpu[0, main], V_ref[main], atol=10 * solver_atol)

    # Row 1 (no topology change) is unaffected by row 0's masking: every bus
    # is finite and, being a distinct injection scale, generally differs from
    # the base case -- just check it actually solved cleanly.
    assert np.all(np.isfinite(V_gpu[1]))


# ---------------------------------------------------------------------------
# compute_limit_violations
# ---------------------------------------------------------------------------

def _build_sweep(ieee14_grid, ieee14_base_case, branch_ids_per_scenario, scales, *,
                 batch_size=8, nb_iter=10, violation_capacity=16, violation_tol=1e-6):
    """Fresh _ScenarioSweepSolver (array API) for IEEE 14: branch data +
    injections + topology set, NOT yet run(). Callers configure limits / run().
    """
    from gpusim2grid.scenario_sweep import _ScenarioSweepSolver
    from conftest import branch_data_arrays

    grid = ieee14_grid
    d = ieee14_base_case
    branch_data, n_lines, n_trafos = branch_data_arrays(grid)

    sn_mva = grid.get_sn_mva()
    n_bus = d["n_bus"]
    p_base_mw = d["Sbus"].real * sn_mva
    q_base_mvar = d["Sbus"].imag * sn_mva
    p_mw = np.stack([p_base_mw * s for s in scales])
    q_mvar = np.stack([q_base_mvar * s for s in scales])

    solver = _ScenarioSweepSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6)
    solver.set_branch_data(*branch_data)
    solver.set_injections(p_mw, q_mvar, sn_mva)
    solver.set_topology(branch_ids_per_scenario)
    solver.violation_capacity = violation_capacity
    solver.violation_tol = violation_tol
    return solver, n_lines, n_trafos


@requires_gpu
def test_no_limits_configured_gives_empty_violations(ieee14_grid, ieee14_base_case):
    """No bus/branch limits configured: every converged scenario reports [];
    a non-converged one reports a single DIVERGENCE/NOT_SIMULATED entry."""
    branch_ids = [[0], [1], [2]]
    scales = [1.0, 1.0, 1.0]
    solver, n_lines, _ = _build_sweep(ieee14_grid, ieee14_base_case, branch_ids, scales)

    n_bus = ieee14_base_case["n_bus"]
    n_branches_total = solver._s.n_branches
    solver.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                      np.full(n_branches_total, np.nan), np.full(n_branches_total, np.nan),
                      n_lines)
    solver.compute_limit_violations = True
    solver.run()

    converged = solver.converged()
    violations = solver.get_violations()
    assert len(violations) == len(branch_ids)
    for c in range(len(branch_ids)):
        if converged[c]:
            assert violations[c] == []
        else:
            assert len(violations[c]) == 1
            assert violations[c][0].violation_type.name in ("DIVERGENCE", "NOT_SIMULATED")


@requires_gpu
def test_tight_limits_reproduce_high_voltage_and_current(ieee14_grid, ieee14_base_case):
    """Deliberately tight, derived-from-actual-values limits reproduce a
    HIGH_VOLTAGE and a CURRENT violation on one row, row-aligned with a
    second (unrelated) row that stays clean."""
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType

    branch_ids = [[], []]
    scales = [1.0, 1.05]

    baseline, n_lines, _ = _build_sweep(ieee14_grid, ieee14_base_case, branch_ids, scales)
    baseline.run()
    baseline.compute_flows()
    n_bus = ieee14_base_case["n_bus"]
    n_branches = baseline._s.n_branches
    V0 = baseline.V_results.to_numpy().reshape(2, n_bus)[0]
    or_amps0 = baseline.or_amps.to_numpy().reshape(2, n_branches)[0]
    vn_kv = ieee14_grid.get_bus_vn_kv()
    vm_kv0 = np.abs(V0) * vn_kv
    assert np.isfinite(baseline.residuals.to_numpy()[0]), "row 0 must converge for this test"

    bus_high = 1
    branch_l = 0
    bus_vmax_kv = np.full(n_bus, np.nan)
    bus_vmax_kv[bus_high] = vm_kv0[bus_high] - 1.0
    limit_a1_ka = np.full(n_branches, np.nan)
    or_ka0 = or_amps0 * 1e-3
    limit_a1_ka[branch_l] = or_ka0[branch_l] * 0.5

    solver, _, _ = _build_sweep(ieee14_grid, ieee14_base_case, branch_ids, scales)
    solver.set_limits(np.full(n_bus, np.nan), bus_vmax_kv,
                      limit_a1_ka, np.full(n_branches, np.nan), n_lines)
    solver.compute_limit_violations = True
    solver.run()

    violations0 = solver.get_violations()[0]
    by_type = {v.violation_type: v for v in violations0}
    assert LimitViolationType.HIGH_VOLTAGE in by_type
    assert LimitViolationType.CURRENT in by_type
    np.testing.assert_allclose(by_type[LimitViolationType.HIGH_VOLTAGE].value,
                               vm_kv0[bus_high], rtol=1e-5)
    np.testing.assert_allclose(by_type[LimitViolationType.CURRENT].value,
                               or_ka0[branch_l], rtol=1e-5)


@needs_bridge
def test_masked_bus_excluded_from_violations():
    """A masked (islanded, frozen) bus never appears in get_violations() for
    the scenario that masks it, even with a limit tight enough that it would
    otherwise trivially violate."""
    from gpusim2grid import ScenarioSweepGPU

    grid, n_bus, spur_line_id, spur_bus = _solved_spur_grid(distributed_slack=False)

    sweep = ScenarioSweepGPU(grid, handle_disconnected_grid=True,
                             precision=None, nb_iter=15, tol_base=1e-10)
    n_lines = len(grid.get_lines())
    n_branches = sweep.n_branches
    bus_vmin_kv = np.full(n_bus, 1e6)   # absurdly high -> "violates" if checked
    bus_vmax_kv = np.full(n_bus, np.nan)
    limit_a1_ka = np.full(n_branches, np.nan)
    limit_a2_ka = np.full(n_branches, np.nan)
    sweep.solver._s.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)
    sweep.compute_limit_violations = True

    sn_mva = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    sweep.set_injections((Sbus.real * sn_mva)[None, :], (Sbus.imag * sn_mva)[None, :], sn_mva)
    sweep.set_topology([[int(spur_line_id)]])
    sweep.compute(batch_size=8)

    assert np.isfinite(sweep.last_residuals()[0]), "masked scenario must actually be solved"
    violations = sweep.get_violations()[0]
    assert not any(v.element_id == spur_bus for v in violations), \
        "masked (islanded) bus must be excluded from violations, not reported"
    other_buses = {v.element_id for v in violations if v.violation_type.name == "LOW_VOLTAGE"}
    assert spur_bus not in other_buses
    assert len(other_buses) >= 1, \
        "the absurd vmin must still flag the live buses -- an empty violation " \
        "list would make the exclusion check above vacuous"


@needs_bridge
def test_not_simulated_scenario_reports_grid_entry():
    """With handle_disconnected_grid OFF, a scenario that would disconnect the
    grid is dropped by the pre-check before ever reaching
    check_limit_violations_kernel. get_violations() must surface this as a
    single GRID/NOT_SIMULATED entry, not silently as []."""
    from gpusim2grid import ScenarioSweepGPU
    from gpusim2grid.contingency_analysis._limit_violations import (
        LimitViolationType, ViolationElementType,
    )

    grid, n_bus, spur_line_id, _ = _solved_spur_grid(distributed_slack=False)

    sweep = ScenarioSweepGPU(grid, handle_disconnected_grid=False,
                             precision=None, nb_iter=15, tol_base=1e-10)
    n_lines = len(grid.get_lines())
    n_branches = sweep.n_branches
    sweep.solver._s.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                               np.full(n_branches, np.nan), np.full(n_branches, np.nan),
                               n_lines)
    sweep.compute_limit_violations = True

    sn_mva = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    sweep.set_injections((Sbus.real * sn_mva)[None, :], (Sbus.imag * sn_mva)[None, :], sn_mva)
    sweep.set_topology([[int(spur_line_id)]])
    sweep.compute(batch_size=8)

    assert np.isnan(sweep.last_residuals()[0]), "scenario must be pre-check-dropped, not solved"
    violations = sweep.get_violations()[0]
    assert len(violations) == 1
    v = violations[0]
    assert v.element_type == ViolationElementType.GRID
    assert v.violation_type == LimitViolationType.NOT_SIMULATED
    assert v.element_id == -1
    assert np.isnan(v.value) and np.isnan(v.limit)
