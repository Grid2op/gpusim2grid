# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_limit_violations: fused on-device per-contingency voltage /
current / divergence check.

Mirrors lightsim2grid's ``ContingencyAnalysis.compute_limit_violations`` flag
(same name/semantics), but computed fused into each chunk's power flow on the
GPU rather than as a per-contingency CPU loop — see
``check_limit_violations_kernel`` (contingency/violation_kernels.cu) and
``ContingencyAnalysisSession::set_limits()``. Only a bounded, per-contingency
compact buffer (``O(n_contingencies * violation_capacity)``) is ever produced
for the batch case; the dense ``V_results``/``or_amps``/``ex_amps`` are never
required just to compute violations.

The pre-contingency ("n") case is a single voltage vector, not a batch, and
is computed in pure Python/numpy (``compute_violations_n``) — no GPU/memory
concern there, so it is exercised separately (and does not need
``requires_gpu``, though it's still guarded here since it lives in a
GPU-focused module for discoverability).
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu


# ---------------------------------------------------------------------------
# Pure-numpy unit test of the classification math (no GPU needed)
# ---------------------------------------------------------------------------

def test_compute_violations_n_classification():
    """Hand-built synthetic case exercising every LimitViolationType,
    including the "not configured" (None) skip paths, with no GPU/grid."""
    from gpusim2grid.contingency_analysis._limit_violations import (
        compute_violations_n, ViolationElementType, LimitViolationType,
    )

    # 2-bus, 1-branch toy system. V in pu; bus_vn_kv chosen so vm_kv is
    # trivial to reason about (vn_kv=100 -> vm_kv = |V|*100).
    V = np.array([1.10 + 0j, 0.80 + 0j])          # bus 0: 110 kV, bus 1: 80 kV
    bus_vn_kv   = np.array([100.0, 100.0])
    bus_vmin_kv = np.array([90.0, np.nan])         # bus 0: no low-voltage violation; bus 1: unconfigured
    bus_vmax_kv = np.array([105.0, np.nan])        # bus 0: 110 > 105 -> HIGH_VOLTAGE

    branch_from = np.array([0])
    branch_to   = np.array([1])
    # yff_eff/yft_eff/ytf_eff/ytt_eff chosen so I_or is large enough to exceed limit_a1_ka;
    # keep it simple: a pure shunt-like self-admittance at "from", nothing
    # at "to" (I_or = yff_eff*V_from, I_ex = 0).
    yff_eff = np.array([1.0 + 0j])
    yft_eff = np.array([0.0 + 0j])
    ytf_eff = np.array([0.0 + 0j])
    ytt_eff = np.array([0.0 + 0j])
    sn_mva = 100.0
    limit_a1_ka = np.array([1e-6])   # guaranteed to be exceeded
    limit_a2_ka = np.array([np.nan]) # not configured -> no ex-side check

    violations = compute_violations_n(
        V, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
        branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
        limit_a1_ka, limit_a2_ka, sn_mva, n_lines=1)

    types_seen = {(v.element_type, v.violation_type) for v in violations}
    assert (ViolationElementType.BUS, LimitViolationType.HIGH_VOLTAGE) in types_seen
    assert (ViolationElementType.LINE, LimitViolationType.CURRENT) in types_seen
    # bus 1 has no limit configured (NaN/NaN) -> never reported
    assert not any(v.element_type == ViolationElementType.BUS and v.element_id == 1
                   for v in violations)
    # side/element_id sanity for the current violation
    cur = next(v for v in violations if v.violation_type == LimitViolationType.CURRENT)
    assert cur.element_id == 0 and cur.side == 1

    # DIVERGENCE (finite residual > tol) short-circuits everything else.
    diverged = compute_violations_n(
        V, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
        branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
        limit_a1_ka, limit_a2_ka, sn_mva, n_lines=1,
        residual=1.0, tol=1e-6)
    assert len(diverged) == 1
    assert diverged[0].element_type == ViolationElementType.GRID
    assert diverged[0].violation_type == LimitViolationType.DIVERGENCE
    assert diverged[0].element_id == -1
    assert diverged[0].value == 1.0 and diverged[0].limit == 1e-6

    # A NaN residual (solver ran but the result is unusable) is ALSO
    # DIVERGENCE, not silently dropped -- there is no NOT_SIMULATED path for
    # compute_violations_n (see its own docstring).
    nan_residual = compute_violations_n(
        V, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
        branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
        limit_a1_ka, limit_a2_ka, sn_mva, n_lines=1,
        residual=np.nan, tol=1e-6)
    assert len(nan_residual) == 1
    assert nan_residual[0].element_type == ViolationElementType.GRID
    assert nan_residual[0].violation_type == LimitViolationType.DIVERGENCE
    assert nan_residual[0].element_id == -1
    assert np.isnan(nan_residual[0].value) and nan_residual[0].limit == 1e-6

    # None limits (nothing configured at all) -> no violations, no crash.
    empty = compute_violations_n(
        V, bus_vn_kv, None, None,
        branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
        None, None, sn_mva, n_lines=1)
    assert empty == []


# ---------------------------------------------------------------------------
# GPU-backed helpers
# ---------------------------------------------------------------------------

def _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids, *,
                  batch_size=8, nb_iter=10, violation_capacity=16, violation_tol=1e-6):
    """Fresh _ContingencyAnalysisSolver (array API) for IEEE 14, branch data +
    contingencies set, NOT yet run(). Callers configure limits / run().
    """
    from gpusim2grid.contingency_analysis import _ContingencyAnalysisSolver
    from conftest import branch_data_arrays

    grid = ieee14_grid
    d = ieee14_base_case
    branch_data, n_lines, n_trafos = branch_data_arrays(grid)

    solver = _ContingencyAnalysisSolver(
        d["Ybus"], d["v_init"].copy(), d["Sbus"],
        d["slack"], d["slack_weights"], d["pv"], d["pq"],
        batch_size=batch_size, nb_iter=nb_iter, max_iter_base=10, tol_base=1e-6)
    solver.set_branch_data(*branch_data)
    solver.build_contingencies(cont_branch_ids)
    solver.violation_capacity = violation_capacity
    solver.violation_tol = violation_tol
    return solver, n_lines, n_trafos


@requires_gpu
def test_no_limits_configured_gives_empty_violations(ieee14_grid, ieee14_base_case):
    """Default case14 has no bus/branch limits configured anywhere: every
    converged contingency reports [], and converged() matches finite/small
    residuals."""
    cont_branch_ids = [[0], [1], [2]]
    solver, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)

    n_bus = ieee14_base_case["n_bus"]
    bus_vmin_kv = np.full(n_bus, np.nan)
    bus_vmax_kv = np.full(n_bus, np.nan)
    n_branches_total = solver._s.n_branches
    limit_a1_ka = np.full(n_branches_total, np.nan)
    limit_a2_ka = np.full(n_branches_total, np.nan)

    solver.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)
    solver.compute_limit_violations = True
    solver.run()

    converged = solver.converged()
    violations = solver.get_violations()
    assert len(violations) == len(cont_branch_ids)
    for c in range(len(cont_branch_ids)):
        if converged[c]:
            assert violations[c] == []
        else:
            # Either genuinely diverged (solved, residual > tol) or dropped
            # by the pre-check before ever being solved (residual == NaN) --
            # IEEE14 single-line trips may hit either, so accept both.
            assert len(violations[c]) == 1
            assert violations[c][0].violation_type.name in ("DIVERGENCE", "NOT_SIMULATED")


@requires_gpu
def test_tight_limits_reproduce_low_high_voltage_and_current(ieee14_grid, ieee14_base_case):
    """Deliberately tight, but derived-from-actual-values, limits reproduce a
    LOW_VOLTAGE, a HIGH_VOLTAGE and a CURRENT violation on the same
    contingency, with reported values matching the unconstrained dense path."""
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType

    cont_branch_ids = [[0]]

    # Pass 1 (unconstrained): read the actual converged values to derive
    # guaranteed-to-violate thresholds (no fragile numeric luck).
    baseline, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)
    baseline.run()
    baseline.compute_flows()
    n_bus = ieee14_base_case["n_bus"]
    n_branches = baseline._s.n_branches
    V0 = baseline.V_results.to_numpy().reshape(1, n_bus)[0]
    or_amps0 = baseline.or_amps.to_numpy().reshape(1, n_branches)[0]
    vn_kv = ieee14_grid.get_bus_vn_kv()
    vm_kv0 = np.abs(V0) * vn_kv
    assert np.isfinite(baseline.residuals.to_numpy()[0]), "contingency 0 must converge for this test"

    bus_low, bus_high = 0, 1
    assert bus_low != bus_high
    # Use a branch NOT tripped by this contingency (a tripped branch's amps
    # are zeroed device-side, so its current would trivially never violate).
    branch_l = next(l for l in range(n_branches) if l not in cont_branch_ids[0])

    bus_vmin_kv = np.full(n_bus, np.nan)
    bus_vmax_kv = np.full(n_bus, np.nan)
    bus_vmin_kv[bus_low] = vm_kv0[bus_low] + 1.0     # observed is below this -> LOW_VOLTAGE
    bus_vmax_kv[bus_high] = vm_kv0[bus_high] - 1.0   # observed is above this -> HIGH_VOLTAGE

    limit_a1_ka = np.full(n_branches, np.nan)
    limit_a2_ka = np.full(n_branches, np.nan)
    or_ka0 = or_amps0 * 1e-3
    limit_a1_ka[branch_l] = or_ka0[branch_l] * 0.5   # guaranteed to be exceeded

    # Pass 2: same contingency, now with tight limits configured.
    solver, _, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)
    solver.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)
    solver.compute_limit_violations = True
    solver.run()

    violations = solver.get_violations()[0]
    by_type = {v.violation_type: v for v in violations}
    assert LimitViolationType.LOW_VOLTAGE in by_type
    assert LimitViolationType.HIGH_VOLTAGE in by_type
    assert LimitViolationType.CURRENT in by_type

    # Current is checked before voltage in the kernel (thermal/current
    # violations are first-order, voltage second-order) -- the CURRENT entry
    # must appear before both voltage entries in get_violations()'s order.
    types_in_order = [v.violation_type for v in violations]
    assert types_in_order.index(LimitViolationType.CURRENT) < \
        types_in_order.index(LimitViolationType.LOW_VOLTAGE)
    assert types_in_order.index(LimitViolationType.CURRENT) < \
        types_in_order.index(LimitViolationType.HIGH_VOLTAGE)

    lv = by_type[LimitViolationType.LOW_VOLTAGE]
    assert lv.element_id == bus_low
    np.testing.assert_allclose(lv.value, vm_kv0[bus_low], rtol=1e-5)

    hv = by_type[LimitViolationType.HIGH_VOLTAGE]
    assert hv.element_id == bus_high
    np.testing.assert_allclose(hv.value, vm_kv0[bus_high], rtol=1e-5)

    cur = by_type[LimitViolationType.CURRENT]
    assert cur.side == 1
    np.testing.assert_allclose(cur.value, or_ka0[branch_l], rtol=1e-5)

    # Uncapped per-type totals agree with the detail-record counts (nothing
    # was truncated in this test: 1 low + 1 high + 1 current).
    counts = solver.get_violation_counts()
    assert counts["low_voltage"][0] == 1
    assert counts["high_voltage"][0] == 1
    assert counts["current"][0] == 1


@requires_gpu
def test_diverged_via_tight_violation_tol(ieee14_grid, ieee14_base_case):
    """An absurdly tight violation_tol turns every (otherwise converged)
    contingency into a single GRID/DIVERGENCE entry -- isolates the
    divergence path from actual solver (non-)convergence, precision-agnostic."""
    from gpusim2grid.contingency_analysis._limit_violations import (
        LimitViolationType, ViolationElementType,
    )

    cont_branch_ids = [[0], [1]]
    solver, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids,
                                      violation_tol=1e-30)
    n_bus = ieee14_base_case["n_bus"]
    n_branches = solver._s.n_branches
    solver.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                      np.full(n_branches, np.nan), np.full(n_branches, np.nan), n_lines)
    solver.compute_limit_violations = True
    solver.run()

    residuals = solver.residuals.to_numpy()
    violations = solver.get_violations()
    for c in range(len(cont_branch_ids)):
        assert len(violations[c]) == 1
        v = violations[c][0]
        assert v.element_type == ViolationElementType.GRID
        assert v.violation_type == LimitViolationType.DIVERGENCE
        assert v.element_id == -1
        np.testing.assert_allclose(v.value, residuals[c], rtol=1e-6)
        assert v.limit == pytest.approx(1e-30)


@requires_gpu
def test_truncation(ieee14_grid, ieee14_base_case):
    """violation_capacity=1 with >=2 simultaneous violations: truncated flag
    set, and the one kept record is a valid (not garbage) entry."""
    cont_branch_ids = [[0]]
    baseline, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)
    # Read n_branches BEFORE run(): once solver_ exists, n_branches() reports
    # the driver's own n_branches_, which is only populated by a driver-level
    # branch-data upload (compute_flows() or the compute_limit_violations
    # path) -- neither of which this baseline run triggers.
    n_branches = baseline._s.n_branches
    baseline.run()
    n_bus = ieee14_base_case["n_bus"]
    V0 = baseline.V_results.to_numpy().reshape(1, n_bus)[0]
    assert np.all(np.isfinite(V0)), "contingency 0 must converge for this test"
    vn_kv = ieee14_grid.get_bus_vn_kv()
    vm_kv0 = np.abs(V0) * vn_kv

    # Tight vmax on every bus -> every bus violates HIGH_VOLTAGE.
    bus_vmax_kv = vm_kv0 - 1.0
    bus_vmin_kv = np.full(n_bus, np.nan)

    solver, _, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids,
                                 violation_capacity=1)
    solver.set_limits(bus_vmin_kv, bus_vmax_kv,
                      np.full(n_branches, np.nan), np.full(n_branches, np.nan), n_lines)
    solver.compute_limit_violations = True
    solver.run()

    assert n_bus > 1, "need >=2 violations to exercise truncation with capacity=1"
    truncated = solver.get_violations_truncated()
    assert truncated[0]
    violations = solver.get_violations()[0]
    assert len(violations) == 1
    assert violations[0].violation_type.name == "HIGH_VOLTAGE"

    # The uncapped per-type totals must stay exact (every bus violates
    # HIGH_VOLTAGE here) even though the detail-record buffer only kept 1 --
    # this is the whole point of tracking them separately from
    # get_violation_count()/get_violations(), which are capped at K.
    counts = solver.get_violation_counts()
    assert counts["high_voltage"][0] == n_bus
    assert counts["low_voltage"][0] == 0
    assert counts["current"][0] == 0


@requires_gpu
def test_truncation_keeps_most_severe_per_type(ieee14_grid, ieee14_base_case):
    """violation_capacity is per TYPE: with several violations of each type,
    the kept records are, for CURRENT, LOW_VOLTAGE and HIGH_VOLTAGE each, the
    violation_capacity of largest |value / limit - 1|, most severe first, the
    types one after the other in that order -- checked against an uncapped
    run of the same batch."""
    from gpusim2grid.contingency_analysis import LimitViolationType
    cont_branch_ids = [[0], [3], [5]]
    baseline, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)
    n_branches = baseline._s.n_branches
    baseline.run()
    n_bus = ieee14_base_case["n_bus"]
    V0 = baseline.V_results.to_numpy().reshape(len(cont_branch_ids), n_bus)[0]
    assert np.all(np.isfinite(V0))
    vm_kv0 = np.abs(V0) * ieee14_grid.get_bus_vn_kv()

    # even buses: vmax below the solved voltage, odd ones: vmin above it, each
    # by a different margin (so every row has several of both, no tie)
    rng = np.random.default_rng(0)
    bus_vmin_kv = np.full(n_bus, np.nan)
    bus_vmax_kv = np.full(n_bus, np.nan)
    even, odd = np.arange(0, n_bus, 2), np.arange(1, n_bus, 2)
    bus_vmax_kv[even] = vm_kv0[even] * (1. - rng.uniform(0.01, 0.2, even.size))
    bus_vmin_kv[odd] = vm_kv0[odd] * (1. + rng.uniform(0.01, 0.2, odd.size))
    # tiny, varied current limits: most branch ends violate
    limit_a1_ka = rng.uniform(1e-3, 5e-2, n_branches)
    limit_a2_ka = rng.uniform(1e-3, 5e-2, n_branches)

    def run(capacity):
        solver, _, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids,
                                     violation_capacity=capacity)
        solver.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)
        solver.compute_limit_violations = True
        solver.run()
        return solver

    full = run(n_bus + 2 * n_branches)   # nothing dropped
    assert not full.get_violations_truncated().any()
    K = 2
    capped = run(K)
    trunc = capped.get_violations_truncated()
    order = [LimitViolationType.CURRENT, LimitViolationType.LOW_VOLTAGE,
             LimitViolationType.HIGH_VOLTAGE]
    sev = lambda v: abs(v.value / v.limit - 1.)
    key = lambda v: (int(v.element_type), v.element_id, v.side, int(v.violation_type))
    for r, (all_rows, kept) in enumerate(zip(full.get_violations(), capped.get_violations())):
        expected, n_of = [], []
        for t in order:
            of_type = [v for v in all_rows if v.violation_type == t]
            n_of.append(len(of_type))
            expected += sorted(of_type, key=sev, reverse=True)[:K]
        assert min(n_of) > K, f"row {r}: the test needs more than K violations of each type"
        assert [key(v) for v in kept] == [key(v) for v in expected], f"row {r}"
        np.testing.assert_allclose([v.value for v in kept], [v.value for v in expected], rtol=1e-9)
        assert trunc[r]
    # the exact per-type totals do not depend on the capacity
    for name, got in capped.get_violation_counts().items():
        np.testing.assert_array_equal(got, full.get_violation_counts()[name])


@requires_gpu
def test_compute_limit_violations_clear_on_change(ieee14_grid, ieee14_base_case):
    """Mirrors lightsim2grid's set_compute_limit_violations: no-op if
    unchanged, else clears previous results; get_violation_*() raises after
    the flag is toggled off until the next run()."""
    cont_branch_ids = [[0]]
    solver, n_lines, _ = _build_solver(ieee14_grid, ieee14_base_case, cont_branch_ids)
    n_bus = ieee14_base_case["n_bus"]
    n_branches = solver._s.n_branches
    solver.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                      np.full(n_branches, np.nan), np.full(n_branches, np.nan), n_lines)

    solver.compute_limit_violations = True
    solver.run()
    solver.get_violations()   # does not raise

    solver.compute_limit_violations = False
    with pytest.raises(RuntimeError):
        solver.get_violations()

    solver.compute_limit_violations = True
    with pytest.raises(Exception):
        # limits/results were cleared by the flag toggle; get_violation_* on
        # the C++ session raises until run() is called again.
        solver._s.get_violation_count()

    solver.run()
    solver.get_violations()   # works again after re-running


@requires_gpu
def test_handle_disconnected_grid_masks_excluded_from_violations(solver_atol):
    """A masked (islanded, frozen) bus never appears in get_violations() for
    the contingency that masks it, even with a limit tight enough that it
    would otherwise trivially violate -- the fused kernel's NaN-propagation
    exclusion (see check_limit_violations_kernel) verified end-to-end through
    the actual device masking path, not just the classification math."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._gpusim2grid import have_ls2g_bridge
    from test_handle_disconnected_grid import _solved_spur_grid

    if not have_ls2g_bridge:
        pytest.skip("handle_disconnected_grid needs the lightsim2grid C++ bridge")

    grid, n_bus, spur_line_id, spur_bus = _solved_spur_grid(distributed_slack=False)

    ca = ContingencyAnalysisGPU(grid, handle_disconnected_grid=True,
                                precision=None, nb_iter=15, tol_base=1e-10)
    n_lines = len(grid.get_lines())
    n_branches = ca.n_branches
    # Tight vmin high enough to violate every live bus AND the masked bus (if
    # it were checked) -- the masked bus's NaN voltage must exclude it.
    bus_vmin_kv = np.full(n_bus, 1e6)   # kV -- absurdly high, guaranteed "violation" if checked
    bus_vmax_kv = np.full(n_bus, np.nan)
    limit_a1_ka = np.full(n_branches, np.nan)
    limit_a2_ka = np.full(n_branches, np.nan)
    ca.solver._s.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)
    ca.compute_limit_violations = True

    ca.add_contingencies_by_branch_id([[int(spur_line_id)]])
    ca.compute(batch_size=8)

    assert np.isfinite(ca.last_residuals()[0]), "masked contingency must actually be solved, not skipped"
    violations = ca.get_violations()[0]
    assert not any(v.element_id == spur_bus for v in violations), \
        "masked (islanded) bus must be excluded from violations, not reported"
    # Every OTHER bus legitimately violates the absurd vmin.
    other_buses = {v.element_id for v in violations if v.violation_type.name == "LOW_VOLTAGE"}
    assert spur_bus not in other_buses
    assert len(other_buses) >= 1, \
        "the absurd vmin must still flag the live buses -- an empty violation " \
        "list would make the exclusion check above vacuous"


@requires_gpu
def test_not_simulated_contingency_reports_grid_entry():
    """With handle_disconnected_grid OFF (default), a contingency that would
    disconnect the grid is dropped by the pre-check before ever reaching
    check_limit_violations_kernel (residual/V stay NaN -- see
    test_handle_disconnected_grid.test_flag_off_skips_the_contingency).
    get_violations() must surface this as a single GRID/NOT_SIMULATED entry
    (value=limit=nan, the solver was never invoked), not silently as []."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid._gpusim2grid import have_ls2g_bridge
    from gpusim2grid.contingency_analysis._limit_violations import (
        LimitViolationType, ViolationElementType,
    )
    from test_handle_disconnected_grid import _solved_spur_grid

    if not have_ls2g_bridge:
        pytest.skip("this setup needs the lightsim2grid C++ bridge")

    grid, n_bus, spur_line_id, _ = _solved_spur_grid(distributed_slack=False)

    ca = ContingencyAnalysisGPU(grid, handle_disconnected_grid=False,
                                precision=None, nb_iter=15, tol_base=1e-10)
    n_lines = len(grid.get_lines())
    n_branches = ca.n_branches
    ca.solver._s.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                            np.full(n_branches, np.nan), np.full(n_branches, np.nan),
                            n_lines)
    ca.compute_limit_violations = True

    ca.add_contingencies_by_branch_id([[int(spur_line_id)]])
    ca.compute(batch_size=8)

    assert np.isnan(ca.last_residuals()[0]), "contingency must be pre-check-dropped, not solved"
    violations = ca.get_violations()[0]
    assert len(violations) == 1
    v = violations[0]
    assert v.element_type == ViolationElementType.GRID
    assert v.violation_type == LimitViolationType.NOT_SIMULATED
    assert v.element_id == -1
    assert np.isnan(v.value) and np.isnan(v.limit)


@requires_gpu
def test_set_limits_from_grid_without_configured_limits(ieee14_base_case):
    """A grid whose current limits were never configured (lightsim2grid then
    returns EMPTY limit vectors, not NaN-filled ones) must still give
    n_lines + n_trafos NaN entries, so set_limits_from_grid() works and the
    fused check reports no violation."""
    from gpusim2grid import ScenarioSweepGPU, ContingencyAnalysisGPU
    grid = ieee14_base_case["grid"]
    n_branch = len(grid.get_lines()) + len(grid.get_trafos())
    sw = ScenarioSweepGPU(grid, nb_iter=6, tol_base=1e-10)
    _, _, a1, a2, n_lines = sw._extract_limits_arrays()
    assert a1.shape == (n_branch,) and a2.shape == (n_branch,)
    assert np.all(np.isnan(a1)) and np.all(np.isnan(a2))
    assert n_lines == len(grid.get_lines())
    sw.set_limits_from_grid()
    sw.compute_limit_violations = True
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p())
    rep = lambda a: np.repeat(np.asarray(a)[None, :], 2, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.compute(batch_size=2)
    assert sw.get_violations() == [[], []]
    # the constructor-time path (bridge factory) must accept it too
    ca = ContingencyAnalysisGPU(grid, nb_iter=6, tol_base=1e-10, compute_limit_violations=True)
    assert ca is not None
