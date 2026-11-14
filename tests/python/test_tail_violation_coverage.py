# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The limit / physical checks under the continuous schedule and its tail,
and the operational check's selection corner cases.

- The tail (``tail_max_groups`` / ``tail_group_size``) moves live rows into
  other slots; the checks run on the leaving slots and read the slot's own
  tripped-branch table entry through ``d_slot_active``. Records, counts and
  truncation must not depend on where a row sat (test_continuous_tail.py only
  compares statuses, iterations and V).
- ``check_limit_violations_kernel`` (one 256-thread block per row) keeps the
  first K ties in scan order through a block scan and ``s_tie_base`` carried
  across tiles: a tie run longer than one tile is exercised here.
- A batch mixing truncated rows with DIVERGENCE and NOT_SIMULATED rows.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = requires_gpu

PREC = "fp32" if _cpp.is_fp32 else "fp64"
CONVERGED, MAX_ITER, DIVERGED, NOT_SIMULATED = 0, 1, 2, 3
TAIL_CONFIGS = ((0, 15), (10 ** 6, 3), (10 ** 6, 1))   # (tail_max_groups, tail_group_size)


# =============================================================================
# helpers
# =============================================================================
def _grid(case):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(getattr(pn, case)())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n = grid.get_bus_vn_kv().shape[0]
    V = grid.ac_pf(grid.dc_pf(np.ones(n, dtype=complex), 1, 1e-6), 30, 1e-10)
    assert V.shape[0] > 0
    return grid


@pytest.fixture(scope="module")
def case118():
    return _grid("case118")


def _solver_vm_kv(grid):
    """|V| (kV) of the grid's own solution, in AC-solver bus numbering."""
    V = np.asarray(grid.get_V_solver())
    me2s = np.asarray(grid.id_me_to_ac_solver())
    vn = np.full(V.shape[0], np.nan)
    ok = me2s >= 0
    vn[me2s[ok]] = np.asarray(grid.get_bus_vn_kv())[ok]
    assert np.all(np.isfinite(vn))
    return np.abs(V) * vn


def _tight_limits(grid, n_branches, seed=0):
    """Voltage limits within +-3 % of the base |V| and small current limits:
    every converged row has many violations of each type, none within
    rounding of its limit."""
    vm_kv = _solver_vm_kv(grid)
    n_bus = vm_kv.shape[0]
    rng = np.random.default_rng(seed)
    vmin = vm_kv * (1. + rng.uniform(-0.03, 0.03, n_bus))
    vmax = vm_kv * (1. + rng.uniform(-0.03, 0.03, n_bus))
    vmax[vmax < vmin] = np.nan
    a1 = rng.uniform(0.02, 0.4, n_branches)
    a2 = rng.uniform(0.02, 0.4, n_branches)
    return vmin, vmax, a1, a2


def _key(v):
    return (int(v.element_type), int(v.element_id), int(v.side), int(v.violation_type))


def _assert_same_violations(got, ref, status, rtol, what):
    """Per row: the same records in the same (ranked) order, their values
    equal to rounding (a DIVERGENCE record's value is a residual: only its
    kind is compared), their limits exactly."""
    assert len(got) == len(ref)
    for r, (a, b) in enumerate(zip(got, ref)):
        assert [_key(v) for v in a] == [_key(v) for v in b], f"{what}, row {r}"
        if status[r] == CONVERGED:
            np.testing.assert_allclose([v.value for v in a], [v.value for v in b],
                                       rtol=rtol, err_msg=f"{what}, row {r}")
            np.testing.assert_array_equal([v.limit for v in a], [v.limit for v in b],
                                          err_msg=f"{what}, row {r}")


def _operational(obj):
    return (np.asarray(obj.get_row_status()), obj.get_violations(),
            obj.get_violations_truncated(), obj.get_violation_counts())


def _assert_same_operational(got, ref, rtol, what):
    st, viol, trunc, counts = got
    st0, viol0, trunc0, counts0 = ref
    np.testing.assert_array_equal(st, st0, err_msg=what)
    np.testing.assert_array_equal(trunc, trunc0, err_msg=what)
    for name in counts0:
        np.testing.assert_array_equal(counts[name], counts0[name], err_msg=f"{what}, {name}")
    _assert_same_violations(viol, viol0, st0, rtol, what)


# =============================================================================
# 1. the tail with compute_limit_violations
# =============================================================================
def test_tail_keeps_operational_violations_contingency_analysis(case118, solver_atol):
    """N-1 plus some N-2 of case118 (handle_disconnected_grid), tight limits,
    violation_capacity 3: with the tail on (groups of 3 and 1, so rows move:
    n_tail_rounds > 0 and n_slot_moves > 0) every row's status, truncated
    flag, per-type counts and ranked violation records equal those of the run
    with the tail off (n_tail_rounds == 0); some rows are truncated."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    ctgs = [[i] for i in range(nbr)] + [[i, i + 7] for i in range(0, nbr - 7, 9)]
    limits = _tight_limits(grid, nbr)
    out, tim = {}, {}
    for mg, G in TAIL_CONFIGS:
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous",
                                    handle_disconnected_grid=True, compute_limit_violations=True)
        ca.tail_max_groups, ca.tail_group_size = mg, G
        ca.set_limits(*limits, len(grid.get_lines()))
        ca.solver.violation_capacity = 3
        ca.add_contingencies_by_branch_id(ctgs)
        ca.compute(batch_size=64)
        out[(mg, G)] = _operational(ca)
        tim[(mg, G)] = ca.timings
    ref = out.pop((0, 15))
    assert tim[(0, 15)].n_tail_rounds == 0
    assert np.sum(ref[0] == CONVERGED) > 0.8 * len(ctgs)
    assert ref[2].sum() > 0.5 * len(ctgs), "most rows must be truncated"
    for key, got in out.items():
        assert tim[key].n_tail_rounds > 0 and tim[key].n_slot_moves > 0, key
        _assert_same_operational(got, ref, solver_atol, str(key))


def test_tail_keeps_operational_violations_scenario_sweep(case118, solver_atol):
    """ScenarioSweepGPU on case118, rows with varied injections and their own
    branch trips (single and double, handle_disconnected_grid), tight
    limits, violation_capacity 3: with the tail on (rows moved:
    n_tail_rounds > 0 and n_slot_moves > 0) statuses, truncated flags,
    per-type counts and ranked records equal the tail-off run's."""
    from gpusim2grid import ScenarioSweepGPU
    grid = case118
    n_lines = len(grid.get_lines())
    nbr = n_lines + len(grid.get_trafos())
    n_rows = 90
    sn = grid.get_sn_mva()
    Sbus = np.asarray(grid.get_Sbus_solver())
    rng = np.random.default_rng(3)
    scale = rng.uniform(0.7, 1.2, size=(n_rows, 1)) * rng.uniform(0.95, 1.05, (n_rows, Sbus.size))
    p, q = Sbus.real * sn * scale, Sbus.imag * sn * scale
    topo = [[] if r % 5 == 0 else
            [int(rng.integers(nbr))] if r % 3 else
            [int(x) for x in rng.choice(nbr, 2, replace=False)]
            for r in range(n_rows)]
    limits = _tight_limits(grid, nbr, seed=1)
    out, tim = {}, {}
    for mg, G in TAIL_CONFIGS:
        sw = ScenarioSweepGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous",
                              handle_disconnected_grid=True, compute_limit_violations=True)
        sw.tail_max_groups, sw.tail_group_size = mg, G
        sw.solver.set_limits(*limits, n_lines)
        sw.solver.violation_capacity = 3
        sw.set_injections(p, q, sn)
        sw.set_topology(topo)
        sw.compute(batch_size=32)
        out[(mg, G)] = _operational(sw)
        tim[(mg, G)] = sw.timings
    ref = out.pop((0, 15))
    assert tim[(0, 15)].n_tail_rounds == 0
    assert np.sum(ref[0] == CONVERGED) > 0.8 * n_rows
    assert ref[2].sum() > 0.5 * n_rows, "most rows must be truncated"
    for key, got in out.items():
        assert tim[key].n_tail_rounds > 0 and tim[key].n_slot_moves > 0, key
        _assert_same_operational(got, ref, solver_atol, str(key))


# =============================================================================
# 2. the tail with compute_physical_violations
# =============================================================================
def test_tail_keeps_physical_violations(case118, solver_atol):
    """ScenarioSweepGPU on case118 with loads scaled up (machines short of
    reactive power), branch trips, compute_physical_violations (bus Q check
    among them), capacity 2: with the tail on (n_slot_moves > 0) every row's
    status, physical records (same keys in the same ranked order, values to
    rounding) and truncated flags equal the tail-off run's; there are bus
    LOW_Q/HIGH_Q records."""
    from gpusim2grid import ScenarioSweepGPU
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType
    grid = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    n_rows = 90
    sn = grid.get_sn_mva()
    Sbus = np.asarray(grid.get_Sbus_solver())
    rng = np.random.default_rng(4)
    scale = rng.uniform(1.0, 1.4, size=(n_rows, 1)) * rng.uniform(0.95, 1.05, (n_rows, Sbus.size))
    p, q = Sbus.real * sn * scale, Sbus.imag * sn * scale
    topo = [[] if r % 4 == 0 else [int(rng.integers(nbr))] for r in range(n_rows)]
    out, tim = {}, {}
    for mg, G in TAIL_CONFIGS:
        sw = ScenarioSweepGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous",
                              handle_disconnected_grid=True, compute_physical_violations=True)
        sw.tail_max_groups, sw.tail_group_size = mg, G
        sw.physical_violation_capacity = 2
        sw.set_injections(p, q, sn)
        sw.set_topology(topo)
        sw.compute(batch_size=32)
        out[(mg, G)] = (np.asarray(sw.get_row_status()), sw.get_physical_violations(),
                        sw.get_physical_violations_truncated())
        tim[(mg, G)] = sw.timings
    st0, phys0, trunc0 = out.pop((0, 15))
    assert np.sum(st0 == CONVERGED) > 0.8 * n_rows
    q_types = {LimitViolationType.LOW_Q, LimitViolationType.HIGH_Q}
    n_q = sum(v.violation_type in q_types for row in phys0 for v in row)
    assert n_q > 0, "the scaled loads must give bus Q records"
    for key, (st, phys, trunc) in out.items():
        assert tim[key].n_tail_rounds > 0 and tim[key].n_slot_moves > 0, key
        np.testing.assert_array_equal(st, st0, err_msg=str(key))
        np.testing.assert_array_equal(trunc, trunc0, err_msg=str(key))
        # physical values are MVA / kV: compare to rounding relative to the value
        _assert_same_violations(phys, phys0, st0, solver_atol, f"physical {key}")


# =============================================================================
# 3. live rows that never fit the tail
# =============================================================================
def test_tail_stays_off_while_live_rows_exceed_its_groups(case118, solver_atol):
    """40 identical rows on 16 slots all converge in the same round, so the
    queue empties with 8 live rows that leave together. With groups of 2,
    tail_max_groups = 3 (6 slots < 8 live rows) leaves the tail off
    (n_tail_rounds == 0, n_slot_moves == 0), tail_max_groups = 4 (8 slots)
    turns it on (n_tail_rounds > 0); statuses and iterations are those of the
    tail_max_groups = 0 run, and V too to rounding."""
    from gpusim2grid import InjectionSweepGPU
    grid = case118
    n_rows = 40
    sn = grid.get_sn_mva()
    Sbus = np.asarray(grid.get_Sbus_solver())
    p = np.repeat((Sbus.real * sn * 1.1)[None], n_rows, 0)
    q = np.repeat((Sbus.imag * sn * 1.1)[None], n_rows, 0)
    out, tim = {}, {}
    for mg in (0, 3, 4):
        sw = InjectionSweepGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous")
        sw.tail_max_groups, sw.tail_group_size = mg, 2
        sw.set_injections(p, q, sn)
        sw.compute(batch_size=16)
        out[mg] = (np.asarray(sw.get_row_status()), np.asarray(sw.get_row_iterations()),
                   sw.V_results.to_numpy().reshape(n_rows, -1))
        tim[mg] = sw.timings
    st0, it0, V0 = out[0]
    assert np.all(st0 == CONVERGED)
    assert np.unique(it0).size == 1, "identical rows must leave in the same round"
    assert tim[3].n_tail_rounds == 0 and tim[3].n_slot_moves == 0
    assert tim[4].n_tail_rounds > 0
    for mg in (3, 4):
        st, it, V = out[mg]
        np.testing.assert_array_equal(st, st0)
        np.testing.assert_array_equal(it, it0)
        np.testing.assert_allclose(V, V0, rtol=0, atol=solver_atol)


# =============================================================================
# 4. ties spanning several tiles of the violation kernel
# =============================================================================
@pytest.fixture(scope="module")
def case300_arrays():
    grid = _grid("case300")
    n_bus = grid.get_Ybus_solver().shape[0]
    slack = grid.get_slack_ids()
    slack_weights = np.zeros(n_bus)
    slack_weights[slack] = 1.0 / slack.shape[0]
    d = {"n_bus": n_bus, "Ybus": grid.get_Ybus_solver().copy(),
         "Sbus": grid.get_Sbus_solver().copy(), "pv": grid.get_pv(), "pq": grid.get_pq(),
         "slack": slack, "slack_weights": slack_weights,
         "v_init": np.asarray(grid.get_V_solver()).copy()}
    return grid, d


def test_ties_longer_than_a_tile_keep_scan_order(case300_arrays, solver_atol, residual_atol):
    """case300 (300 buses, 411 branches, VIOL_BLOCK = 256): vmin = +inf on
    every bus gives one LOW_VOLTAGE record per bus, all of severity exactly 1
    (a 300-long tie run over two bus tiles); a 0 kA limit on every branch end
    gives more than 512 CURRENT records, all of severity +inf (a tie run over
    two branch tiles). The uncapped records equal compute_violations_n's on the same V
    (as sets) and the per-type counts equal its totals. For K = 5, 200 and
    290 (below both tie counts; 290 needs bus ties of the second tile) and
    600 (CURRENT ties of the second branch tile; LOW_VOLTAGE then uncapped)
    the kept records are the documented ranking's first K per type -- buses
    0..min(K, n_bus)-1 for LOW_VOLTAGE, K branch ends in increasing scan
    order for CURRENT --, the counts are unchanged and every row is
    truncated."""
    from conftest import branch_data_arrays
    from gpusim2grid.contingency_analysis._limit_violations import compute_violations_n
    from test_limit_violations import _build_solver, _ranked, _record_key, _scan_key
    grid, d = case300_arrays
    n_bus = d["n_bus"]
    violation_tol = 10. * residual_atol
    branch_data, n_lines, _ = branch_data_arrays(grid)
    ctgs = [[1], [8], [n_lines + 27]]
    baseline, _, _ = _build_solver(grid, d, ctgs, batch_size=4, violation_tol=violation_tol)
    n_br = baseline._s.n_branches
    assert n_bus > 256 and n_br > 256
    baseline.run()
    V = baseline.V_results.to_numpy().reshape(len(ctgs), n_bus)
    assert np.all(np.isfinite(V))

    # LOW_VOLTAGE: every bus, severity 1 (a bus reports one voltage type only:
    # no vmax, so nothing competes with the tie run)
    vmin = np.full(n_bus, np.inf)
    vmax = np.full(n_bus, np.nan)
    a1 = np.zeros(n_br)                           # CURRENT: every branch end, severity +inf
    a2 = np.zeros(n_br)

    def run(capacity):
        s, _, _ = _build_solver(grid, d, ctgs, batch_size=4, violation_capacity=capacity,
                                violation_tol=violation_tol)
        s.set_limits(vmin, vmax, a1, a2, n_lines)
        s.compute_limit_violations = True
        s.run()
        return s

    full = run(n_bus + 2 * n_br)
    assert not full.get_violations_truncated().any()
    sn_mva = branch_data[-1]
    V_full = full.V_results.to_numpy().reshape(len(ctgs), n_bus)
    counts = full.get_violation_counts()
    for r, rows in enumerate(full.get_violations()):
        l1, l2 = a1.copy(), a2.copy()
        l1[ctgs[r]] = np.nan
        l2[ctgs[r]] = np.nan
        ref = compute_violations_n(V_full[r], grid.get_bus_vn_kv(), vmin, vmax,
                                   *branch_data[:6], l1, l2, sn_mva, n_lines)
        assert {_record_key(v) for v in rows} == {_record_key(v) for v in ref}, f"row {r}"
        n_of = {t: sum(v.violation_type.name == t for v in ref)
                for t in ("CURRENT", "LOW_VOLTAGE", "HIGH_VOLTAGE")}
        assert n_of["LOW_VOLTAGE"] == n_bus and n_of["CURRENT"] > 2 * 256
        assert n_of["HIGH_VOLTAGE"] == 0
        assert counts["low_voltage"][r] == n_of["LOW_VOLTAGE"]
        assert counts["high_voltage"][r] == n_of["HIGH_VOLTAGE"]
        assert counts["current"][r] == n_of["CURRENT"]

    for K in (5, 200, 290, 600):
        capped = run(K)
        for r, (rows, kept) in enumerate(zip(full.get_violations(), capped.get_violations())):
            expected = _ranked(rows, n_lines, K)
            assert [_record_key(v) for v in kept] == [_record_key(v) for v in expected], \
                f"row {r}, K={K}"
            low = [v.element_id for v in kept if v.violation_type.name == "LOW_VOLTAGE"]
            assert low == list(range(min(K, n_bus))), f"row {r}, K={K}: buses in scan order"
            cur = [_scan_key(v, n_lines) for v in kept if v.violation_type.name == "CURRENT"]
            assert cur == sorted(cur) and len(cur) == K, f"row {r}, K={K}: ends in scan order"
            np.testing.assert_allclose([v.value for v in kept], [v.value for v in expected],
                                       rtol=solver_atol)
        for name, got in capped.get_violation_counts().items():
            np.testing.assert_array_equal(got, counts[name], err_msg=f"K={K}, {name}")
        assert capped.get_violations_truncated().all()


# =============================================================================
# 5. the operational check, continuous vs chunked
# =============================================================================
def test_operational_check_continuous_equals_chunked(case118, solver_atol):
    """N-1 of case118 (handle_disconnected_grid), tight limits, capacity 3:
    the continuous schedule (tail off and tail on) gives the chunked
    schedule's statuses, truncated flags, per-type counts and ranked records
    (values to solver_atol: continuous rows stop at tol, chunked ones run
    nb_iter iterations)."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    ctgs = [[i] for i in range(nbr)]
    limits = _tight_limits(grid, nbr, seed=2)
    out = {}
    for sched, mg in (("chunked", 0), ("continuous", 0), ("continuous", 16)):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, scheduling=sched,
                                    handle_disconnected_grid=True, compute_limit_violations=True)
        ca.tail_max_groups = mg
        ca.set_limits(*limits, len(grid.get_lines()))
        ca.solver.violation_capacity = 3
        ca.add_contingencies_by_branch_id(ctgs)
        ca.compute(batch_size=32)
        out[(sched, mg)] = _operational(ca)
    ref = out.pop(("chunked", 0))
    assert np.sum(ref[0] == CONVERGED) > 0.8 * nbr
    assert ref[2].sum() > 0.5 * nbr
    for key, got in out.items():
        _assert_same_operational(got, ref, solver_atol, str(key))


# =============================================================================
# 6. truncated rows next to DIVERGENCE and NOT_SIMULATED rows
# =============================================================================
@pytest.mark.parametrize("scheduling", ["chunked", "continuous"])
def test_truncation_next_to_divergence_and_not_simulated(case118, scheduling):
    """N-1 of case118 without handle_disconnected_grid (islanding trips are
    NOT_SIMULATED), nb_iter = 3 and violation_tol = tol (rows not converged
    after 3 iterations are DIVERGENCE), tight limits, capacity 2. Each row's
    entry matches its status: NOT_SIMULATED -> one GRID/NOT_SIMULATED record
    (NaN value/limit), counts -1, not truncated; MAX_ITER -> one
    GRID/DIVERGENCE record (value = its residual > limit = violation_tol),
    counts 0, not truncated; CONVERGED -> no GRID record, truncated, counts
    >= the kept records of each type. All three kinds occur."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    K = 2
    ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=3, scheduling=scheduling,
                                handle_disconnected_grid=False, compute_limit_violations=True)
    ca.set_limits(*_tight_limits(grid, nbr, seed=3), len(grid.get_lines()))
    ca.solver.violation_capacity = K
    ca.solver.violation_tol = ca.tol
    ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
    ca.compute(batch_size=32)
    st = np.asarray(ca.get_row_status())
    res = ca.last_residuals()
    viol = ca.get_violations()
    trunc = ca.get_violations_truncated()
    counts = ca.get_violation_counts()
    for kind in (CONVERGED, MAX_ITER, NOT_SIMULATED):
        assert np.any(st == kind), f"the batch needs a row of status {kind}: {np.bincount(st)}"
    assert not np.any(st == DIVERGED)
    for r in range(nbr):
        rec = viol[r]
        c = {k: int(v[r]) for k, v in counts.items()}
        if st[r] == NOT_SIMULATED:
            assert len(rec) == 1 and rec[0].violation_type.name == "NOT_SIMULATED", r
            assert rec[0].element_type.name == "GRID" and rec[0].element_id == -1
            assert np.isnan(rec[0].value) and np.isnan(rec[0].limit)
            assert not trunc[r], r
            assert set(c.values()) == {-1}, (r, c)
        elif st[r] == MAX_ITER:
            assert len(rec) == 1 and rec[0].violation_type.name == "DIVERGENCE", r
            assert rec[0].element_type.name == "GRID" and rec[0].element_id == -1
            np.testing.assert_allclose(rec[0].value, res[r], rtol=1e-6)
            assert rec[0].value > rec[0].limit
            np.testing.assert_allclose(rec[0].limit, ca.tol, rtol=1e-6)
            assert not trunc[r], r
            assert set(c.values()) == {0}, (r, c)
        else:
            assert rec and all(v.element_type.name != "GRID" for v in rec), r
            assert trunc[r], r
            n_kept = {"current": 0, "low_voltage": 0, "high_voltage": 0}
            for v in rec:
                n_kept[v.violation_type.name.lower()] += 1
            for k in n_kept:
                assert n_kept[k] == min(K, c[k]), (r, k, n_kept, c)
