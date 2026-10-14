# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""scheduling='continuous' (docs/dev_notes/continuous_batching.md, Part A).

``batch_size`` slots; every ``nb_iter_per_round`` iterations each row is
checked, and one that converged (``||F||inf < tol``), diverged or used its
``nb_iter`` budget leaves at once, its slot refilled from the queue. Only the
data movement differs from the chunked schedule, so:

- a row's result is the chunked one's (to the convergence tolerance), whatever
  the slots, the order of the rows and the rows it shared slots with --
  including rows carrying masks, pins, gen_v overrides and generator
  contingencies cycling through one slot (reload hygiene);
- its iteration count is lightsim2grid's at ``nb_iter_per_round = 1``
  (one-off ``ac_pf`` from the same start; lightsim2grid takes its ``tol`` in
  MVA, ``||F||inf < tol / sn_mva``);
- the violation and physical-check records are the chunked ones, as sets.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import is_fp32

pytestmark = requires_gpu
fp64_only = pytest.mark.skipif(
    bool(is_fp32), reason="iteration-count parity with lightsim2grid's double-precision NR")

CONVERGED, MAX_ITER, DIVERGED, NOT_SIMULATED = 0, 1, 2, 3
PREC = "fp32" if is_fp32 else "fp64"
# FP32: a row leaves at its tol (1e-3 by default, the residual floor being
# ~1e-4), i.e. ~1e-4 from a 10-iteration chunked solve, and a status decided
# that close to the floor is not reproducible across schedules -- so there the
# schedules are compared on the rows both call converged, at 10 x solver_atol.
V_FACTOR = 10 if is_fp32 else 1


def _grid(case):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(getattr(pn, case)())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n = grid.get_bus_vn_kv().shape[0]
    V = grid.ac_pf(grid.dc_pf(np.ones(n, dtype=complex), 1, 1e-6), 30, 1e-8)
    assert V.shape[0] > 0
    return grid, V


@pytest.fixture(scope="module")
def case118():
    return _grid("case118")


@pytest.fixture(scope="module")
def case14():
    return _grid("case14")


def _scaled_injections(grid, n_rows, seed=0, spread=(0.6, 1.4)):
    n_bus = grid.get_Ybus_solver().shape[0]
    sn = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    rng = np.random.default_rng(seed)
    scale = rng.uniform(*spread, size=(n_rows, 1)) * rng.uniform(0.9, 1.1, size=(n_rows, n_bus))
    return Sbus.real * sn * scale, Sbus.imag * sn * scale, sn


def _injection_sweep(grid, p, q, sn, batch_size, **kwargs):
    from gpusim2grid import InjectionSweepGPU
    sw = InjectionSweepGPU(grid, precision=PREC, **kwargs)
    sw.set_injections(p, q, sn)
    sw.compute(batch_size=batch_size)
    n_bus = grid.get_Ybus_solver().shape[0]
    return sw, sw.V_results.to_numpy().reshape(p.shape[0], n_bus)


def _records(viols):
    """A row's records as a sorted list of (type, element_type, id, side,
    value) -- record order is a ranking, compared as sets."""
    return sorted((int(v.violation_type), int(v.element_type), int(v.element_id),
                   int(v.side), float(v.value)) for v in viols)


def _assert_same_records(a, b, rtol):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert len(ra) == len(rb), (ra, rb)
        for x, y in zip(_records(ra), _records(rb)):
            assert x[:4] == y[:4], (x, y)
            np.testing.assert_allclose(x[4], y[4], rtol=rtol)


# =============================================================================
# Injection sweep
# =============================================================================
def test_injection_sweep_matches_chunked(case118, solver_atol, residual_atol):
    grid, _ = case118
    p, q, sn = _scaled_injections(grid, 37)
    sw_c, V_c = _injection_sweep(grid, p, q, sn, 8, nb_iter=10)
    sw_k, V_k = _injection_sweep(grid, p, q, sn, 8, nb_iter=10, scheduling="continuous",
                                 nb_iter_per_round=2)
    st_c, st_k = sw_c.get_row_status(), sw_k.get_row_status()
    if not is_fp32:
        assert np.all(st_c == CONVERGED) and np.all(st_k == CONVERGED)
    both = (st_c == CONVERGED) & (st_k == CONVERGED)
    assert both.sum() > 0.8 * len(both)
    np.testing.assert_allclose(V_k[both], V_c[both], atol=V_FACTOR * solver_atol)
    assert np.all(sw_k.last_residuals()[st_k == CONVERGED] < sw_k.tol)

    # chunked: every row ran nb_iter; continuous: a multiple of k, at least k
    assert np.all(sw_c.get_row_iterations() == 10)
    it = sw_k.get_row_iterations()
    assert np.all(it % 2 == 0) and np.all(it >= 2) and np.all(it <= 10)

    t = sw_k.timings
    assert t.scheduling == 1 and t.n_chunks == 0 and t.chunk_size == 8
    assert t.nb_iter_per_round == 2 and t.n_rounds > 0
    assert 0. < t.occupancy <= 1.
    assert t.n_refactorize == t.n_rounds * 2 - 1        # direct_refactor_every
    assert "schedule" in t.to_dict()["gpu_compute"]
    assert sw_c.timings.scheduling == 0 and sw_c.timings.n_rounds == 0


def test_injection_sweep_row_independent_of_slots_and_order(case118, solver_atol):
    """A row's trajectory depends on that row alone: same V and the same
    iteration count whatever the slot count and the order of the rows."""
    grid, _ = case118
    p, q, sn = _scaled_injections(grid, 23, seed=1)
    perm = np.random.default_rng(2).permutation(23)
    ref_sw, ref_V = _injection_sweep(grid, p, q, sn, 23, nb_iter=10, scheduling="continuous",
                                     nb_iter_per_round=1)
    ref_it = ref_sw.get_row_iterations()
    # FP32: the slots' rounding differs with their count, so a row whose residual
    # sits on tol may take one more iteration
    same_it = (np.testing.assert_array_equal if not is_fp32 else
               lambda a, b: np.testing.assert_array_less(np.abs(a - b), 2))
    for S in (1, 4, 9):
        sw, V = _injection_sweep(grid, p, q, sn, S, nb_iter=10, scheduling="continuous",
                                 nb_iter_per_round=1)
        np.testing.assert_allclose(V, ref_V, atol=V_FACTOR * solver_atol)
        same_it(sw.get_row_iterations(), ref_it)
    sw, V = _injection_sweep(grid, p[perm], q[perm], sn, 4, nb_iter=10,
                             scheduling="continuous", nb_iter_per_round=1)
    np.testing.assert_allclose(V, ref_V[perm], atol=V_FACTOR * solver_atol)
    same_it(sw.get_row_iterations(), ref_it[perm])


def test_failing_row_does_not_disturb_its_neighbours(case118, solver_atol):
    """A row that cannot converge (loads x 8) ends MAX_ITER / DIVERGED; the
    rows sharing its slots are solved as without it."""
    grid, _ = case118
    p, q, sn = _scaled_injections(grid, 9, seed=3)
    bad_p, bad_q = p.copy(), q.copy()
    bad_p[4] *= 8.
    bad_q[4] *= 8.
    sw, V = _injection_sweep(grid, bad_p, bad_q, sn, 2, nb_iter=8, scheduling="continuous",
                             nb_iter_per_round=2)
    st = sw.get_row_status()
    assert st[4] in (MAX_ITER, DIVERGED)
    others = np.r_[0:4, 5:9]
    assert np.all(st[others] == CONVERGED)
    _, V_ref = _injection_sweep(grid, p[others], q[others], sn, 2, nb_iter=8,
                                scheduling="continuous", nb_iter_per_round=2)
    np.testing.assert_allclose(V[others], V_ref, atol=solver_atol)
    if st[4] == MAX_ITER:
        assert sw.get_row_iterations()[4] == 8


def test_gen_v_reload_hygiene(case14, solver_atol):
    """Rows with and without a gen_v override cycling through one slot."""
    grid, _ = case14
    p, q, sn = _scaled_injections(grid, 8, seed=4, spread=(0.9, 1.1))
    n_gen = len(grid.get_generators())
    gen_v = np.full((8, n_gen), np.nan)
    gen_v[1::2, 1] = 1.035          # every other row moves generator 1
    out = {}
    for sched, S in (("chunked", 8), ("continuous", 1), ("continuous", 3)):
        from gpusim2grid import InjectionSweepGPU
        sw = InjectionSweepGPU(grid, precision=PREC, nb_iter=10, scheduling=sched, nb_iter_per_round=1)
        sw.set_injections(p, q, sn)
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=S)
        out[(sched, S)] = sw.V_results.to_numpy().reshape(8, -1)
        assert np.all(sw.get_row_status() == CONVERGED)
    for key, V in out.items():
        np.testing.assert_allclose(V, out[("chunked", 8)], atol=V_FACTOR * solver_atol,
                                   err_msg=str(key))


def test_remote_gen_v_reload_hygiene(solver_atol):
    """gen_v rows driving a VoltageControl group's set-point (per-slot v_set:
    gen 3 on bus 7 remotely regulating bus 9 of case14) interleaved with rows
    that keep the base set-point, cycled through one slot: each row is the
    one-off lightsim2grid solve at its set-point."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    from gpusim2grid import InjectionSweepGPU, ScenarioSweepGPU
    from gpusim2grid._gpusim2grid import have_ls2g_bridge
    if not have_ls2g_bridge:
        pytest.skip("VoltageControl needs the lightsim2grid C++ bridge")
    gen_remote, reg_bus = 3, 9

    def remote_case14(v_target=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = init_from_pandapower(pn.case14())
        if not hasattr(model, "set_gen_regulated_bus"):
            pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
        model.set_gen_regulated_bus(gen_remote, reg_bus)
        if v_target is not None:
            model.change_v_gen(gen_remote, float(v_target))
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NR_KLU)
        V = model.ac_pf(np.ones(14, dtype=complex), 40, 1e-11)
        assert V.shape[0] > 0
        return model, V

    model, V_base = remote_case14()
    targets = [np.nan, 1.03, np.nan, 1.055, 1.04, np.nan]
    refs = np.stack([V_base if np.isnan(v) else remote_case14(v)[1] for v in targets])
    n = len(targets)
    gen_v = np.full((n, len(model.get_generators())), np.nan)
    gen_v[:, gen_remote] = targets
    sn = model.get_sn_mva()
    S = np.asarray(model.get_Sbus_solver())
    p, q = np.tile(S.real * sn, (n, 1)), np.tile(S.imag * sn, (n, 1))
    for cls in (ScenarioSweepGPU, InjectionSweepGPU):
        sw = cls(model, precision=PREC, nb_iter=20, tol_base=1e-11, scheduling="continuous", nb_iter_per_round=1)
        sw.set_injections(p, q, sn)
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=1)
        assert np.all(sw.get_row_status() == CONVERGED), cls.__name__
        V = sw.solver.V_results.to_numpy().reshape(n, -1)
        np.testing.assert_allclose(V, refs, atol=10 * solver_atol, err_msg=cls.__name__)


def test_edge_cases(case118, solver_atol):
    grid, _ = case118
    p, q, sn = _scaled_injections(grid, 3, seed=5)
    # fewer rows than slots: the capacity is the row count
    sw, V = _injection_sweep(grid, p, q, sn, 64, nb_iter=10, scheduling="continuous")
    assert sw.solver._s.used_batch_size == 3
    assert np.all(sw.get_row_status() == CONVERGED)
    # a round longer than the budget: one round, the budget is rounded up to it
    sw, V2 = _injection_sweep(grid, p, q, sn, 2, nb_iter=2, scheduling="continuous",
                              nb_iter_per_round=4)
    np.testing.assert_array_equal(sw.get_row_iterations(), 4)
    np.testing.assert_allclose(V2, V, atol=solver_atol)


# =============================================================================
# Contingency analysis
# =============================================================================
def _ca(grid, ctgs, batch_size, **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, precision=PREC, **kwargs)
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute(batch_size=batch_size)
    return ca


def test_contingency_analysis_matches_chunked(case118, solver_atol):
    """N-1 + some N-2 with islands solved on their main component, limit and
    physical checks: same statuses, masks, voltages and records."""
    grid, _ = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    ctgs = [[i] for i in range(nbr)] + [[i, i + 1] for i in range(0, nbr - 1, 5)]
    kw = dict(nb_iter=10, handle_disconnected_grid=True, compute_limit_violations=True,
              compute_physical_violations=True)
    ca_c = _ca(grid, ctgs, 16, **kw)
    ca_k = _ca(grid, ctgs, 16, scheduling="continuous", nb_iter_per_round=2, **kw)
    n = len(ctgs)
    st_c, st_k = ca_c.get_row_status(), ca_k.get_row_status()
    if not is_fp32:
        np.testing.assert_array_equal(st_k, st_c)
    np.testing.assert_array_equal(st_k == NOT_SIMULATED, st_c == NOT_SIMULATED)
    ok = (st_c == CONVERGED) & (st_k == CONVERGED)
    assert ok.sum() > 0.8 * n
    V_c = ca_c.V_results.to_numpy().reshape(n, -1)[ok]
    V_k = ca_k.V_results.to_numpy().reshape(n, -1)[ok]
    np.testing.assert_array_equal(np.isnan(V_k), np.isnan(V_c))
    m = ~np.isnan(V_c)
    np.testing.assert_allclose(V_k[m], V_c[m], atol=V_FACTOR * solver_atol)
    if not is_fp32:   # FP32: a record on its threshold may come or go
        sel = np.flatnonzero(ok)
        _assert_same_records([ca_k.get_violations()[i] for i in sel],
                             [ca_c.get_violations()[i] for i in sel], rtol=solver_atol)
        _assert_same_records([ca_k.get_physical_violations()[i] for i in sel],
                             [ca_c.get_physical_violations()[i] for i in sel], rtol=solver_atol)


def test_every_row_dropped(case14):
    """No active row at all: nothing is solved, every row NOT_SIMULATED."""
    grid, _ = case14
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    probe = _ca(grid, [[i] for i in range(nbr)], 32, nb_iter=4)
    islanding = np.flatnonzero(probe.get_row_status() == NOT_SIMULATED)
    assert islanding.size > 0, "case14 has a branch whose trip islands a bus"
    ca = _ca(grid, [[int(i)] for i in islanding], 4, nb_iter=4, scheduling="continuous")
    np.testing.assert_array_equal(ca.get_row_status(), NOT_SIMULATED)
    np.testing.assert_array_equal(ca.get_row_iterations(), 0)
    assert np.all(np.isnan(ca.last_residuals()))
    assert ca.timings.n_rounds == 0


def test_chunked_reports_row_outcome(case14):
    grid, _ = case14
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    ca = _ca(grid, [[i] for i in range(nbr)], 8, nb_iter=1)
    st, it = ca.get_row_status(), ca.get_row_iterations()
    dropped = st == NOT_SIMULATED
    assert np.all(it[dropped] == 0) and np.all(it[~dropped] == 1)
    # the status is the final residual's against tol
    res = ca.last_residuals()
    np.testing.assert_array_equal(st[~dropped],
                                  np.where(res[~dropped] < ca.tol, CONVERGED, MAX_ITER))
    assert np.any(st[~dropped] == MAX_ITER)           # one iteration is not enough for all
    ca.solver.nb_iter = 10
    ca.compute(batch_size=8)
    st = ca.get_row_status()
    assert np.all(st[~dropped] == CONVERGED) and np.all(st[dropped] == NOT_SIMULATED)


@fp64_only
@pytest.mark.parametrize("case_name", ["case14", "case118"])
def test_iterations_and_voltages_match_lightsim2grid(case_name, solver_atol):
    """At nb_iter_per_round = 1 a row's iteration count is the one of
    lightsim2grid's own Newton-Raphson from the same start (the converged base
    case), and its voltages are lightsim2grid's."""
    grid, V0 = _grid(case_name)
    sn = grid.get_sn_mva()
    n_line = len(grid.get_lines())
    ca = _ca(grid, [[l] for l in range(n_line)], 7, nb_iter=10, scheduling="continuous",
             nb_iter_per_round=1)
    tol = ca.tol
    st, it = ca.get_row_status(), ca.get_row_iterations()
    V = ca.V_results.to_numpy().reshape(n_line, -1)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    n_checked = 0
    for line in np.flatnonzero(st == CONVERGED):
        grid.deactivate_powerline(int(line))
        try:
            V_ls = grid.ac_pf(V0.copy(), 10, tol * sn)       # lightsim2grid's tol is in MVA
            nb_ls = grid.get_solver().get_nb_iter()
        finally:
            grid.reactivate_powerline(int(line))
        assert V_ls.shape[0] > 0
        assert it[line] == nb_ls, f"line {line}: {it[line]} vs lightsim2grid's {nb_ls}"
        np.testing.assert_allclose(V[line], V_ls[buses], atol=solver_atol)
        n_checked += 1
    assert n_checked > 0.8 * n_line


def test_tol_above_violation_tol_warns(case14):
    grid, _ = case14
    with pytest.warns(RuntimeWarning, match="violation_tol"):
        _ca(grid, [[0]], 4, nb_iter=10, scheduling="continuous", tol=1e-3,
            compute_limit_violations=True)


# =============================================================================
# Scenario sweep: every per-row feature cycling through few slots
# =============================================================================
def test_scenario_sweep_reload_hygiene(case14, solver_atol):
    """Plain rows, a branch trip, an island solved on its main component, a
    generator contingency (bus released PV -> PQ, slack re-weighted) and gen_v
    overrides, interleaved and cycled through 1 and 2 slots: every row is the
    chunked one."""
    from gpusim2grid import ScenarioSweepGPU
    grid, _ = case14
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    probe = _ca(grid, [[i] for i in range(nbr)], 32, nb_iter=4)
    island = int(np.flatnonzero(probe.get_row_status() == NOT_SIMULATED)[0])
    gens = grid.get_generators()
    n_gen = len(gens)
    non_slack = [g for g in range(n_gen) if not gens[g].is_slack]
    load_p, load_q = (np.asarray(a, dtype=float) for a in grid.get_loads_res_full()[:2])
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)

    kinds = ["plain", "trip", "island", "gen_off", "gen_v", "island+gen_v"] * 2
    n = len(kinds)
    topo = [[] for _ in range(n)]
    gen_off = np.zeros((n, n_gen), dtype=bool)
    gen_v = np.full((n, n_gen), np.nan)
    rng = np.random.default_rng(6)
    lp = np.repeat(load_p[None], n, 0) * rng.uniform(0.9, 1.1, (n, 1))
    lq = np.repeat(load_q[None], n, 0) * rng.uniform(0.9, 1.1, (n, 1))
    gp = np.repeat(gen_p[None], n, 0)
    for r, kind in enumerate(kinds):
        if kind == "trip":
            topo[r] = [3]
        if "island" in kind:
            topo[r] = [island]
        if kind == "gen_off":
            gen_off[r, non_slack[r % len(non_slack)]] = True
        if "gen_v" in kind:
            gen_v[r, non_slack[0]] = 1.03 + 0.002 * r

    def run(sched, S):
        sw = ScenarioSweepGPU(grid, precision=PREC, nb_iter=10, handle_disconnected_grid=True,
                              scheduling=sched, nb_iter_per_round=1)
        sw.set_injections_from_elements(lp, lq, gp)
        sw.set_topology(topo)
        sw.set_contingency_gens(gen_off)
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=S)
        return sw.get_row_status(), sw.V_results.to_numpy().reshape(n, -1)

    st_ref, V_ref = run("chunked", n)
    assert np.all(st_ref == CONVERGED)
    for S in (1, 2):
        st, V = run("continuous", S)
        np.testing.assert_array_equal(st, st_ref)
        np.testing.assert_array_equal(np.isnan(V), np.isnan(V_ref))
        m = ~np.isnan(V_ref)
        np.testing.assert_allclose(V[m], V_ref[m], atol=solver_atol, err_msg=f"S={S}")


# =============================================================================
# The reactive-limit outer loop's second pass
# =============================================================================
@pytest.mark.skipif(bool(is_fp32), reason="the outer loop's first pass needs residuals "
                    "below violation_tol (1e-6), out of an FP32 build's reach")
def test_reactive_limits_outer_loop_continuous(solver_atol):
    """The second pass of reactive_limits_outer_loop runs with the analysis'
    scheduling: the same re-solved rows and results as a chunked analysis that
    leaves nothing out -- and no LEFT_OUT row, there being no last partial
    chunk."""
    from gpusim2grid import _gpusim2grid as _cpp
    if not getattr(_cpp, "have_ls2g_gen_pv_release", False):
        pytest.skip("needs the bridge built against a lightsim2grid with can_be_pv")
    from gpusim2grid import ContingencyAnalysisGPU, ReactiveLimitsStatus
    from test_reactive_limits_outer_loop import _build, LINES, TOL_MVA, TOL_VM

    grid, _ = _build()

    def run(sched, batch_size):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, compute_physical_violations=True,
                                    reactive_limits_outer_loop=True, scheduling=sched,
                                    nb_iter_per_round=1)
        ca.physical_violation_tol_mva = TOL_MVA
        ca.physical_violation_tol_vm_pu = TOL_VM
        ca.add_contingencies_by_branch_id([[i] for i in range(len(LINES))])
        ca.compute(batch_size=batch_size)
        return ca

    ca_ref = run("chunked", 16)          # one chunk: nothing left out
    ca_left = run("chunked", 4)          # a last partial chunk of the second pass left out
    ca_k = run("continuous", 4)
    st_ref, st_k = ca_ref.get_outer_loop_status(), ca_k.get_outer_loop_status()
    assert np.any(ca_left.get_outer_loop_status() == ReactiveLimitsStatus.LEFT_OUT)
    assert not np.any(st_k == ReactiveLimitsStatus.LEFT_OUT)
    np.testing.assert_array_equal(st_k, st_ref)
    assert np.any(st_k == ReactiveLimitsStatus.RECOMPUTED)
    n = len(LINES)
    np.testing.assert_allclose(ca_k.V_results.to_numpy().reshape(n, -1),
                               ca_ref.V_results.to_numpy().reshape(n, -1),
                               atol=V_FACTOR * solver_atol)
    if not is_fp32:
        _assert_same_records(ca_k.get_physical_violations(), ca_ref.get_physical_violations(),
                             rtol=solver_atol)
    # a re-solved row reports its second pass' outcome
    np.testing.assert_array_equal(ca_k.get_row_status(), CONVERGED)
    rec = st_k == ReactiveLimitsStatus.RECOMPUTED
    assert np.all(ca_k.get_row_iterations()[rec] >= 1)


# =============================================================================
# Refusals and validation
# =============================================================================
def test_refusals(case14, monkeypatch):
    from gpusim2grid import ContingencyAnalysisGPU, ScenarioSweepGPU
    grid, _ = case14
    with pytest.raises(ValueError, match="scheduling"):
        ContingencyAnalysisGPU(grid, precision=PREC, scheduling="eager")
    with pytest.raises(ValueError, match="nb_iter_per_round"):
        ContingencyAnalysisGPU(grid, precision=PREC, scheduling="continuous", nb_iter_per_round=0)
    with pytest.raises(ValueError, match="tol"):
        ContingencyAnalysisGPU(grid, precision=PREC, tol=0.)

    for strategy in ("direct_iter0_only", "direct_refactor_every_n"):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=4)
        ca.add_contingencies_by_branch_id([[0], [1]])
        ca.strategy = strategy
        with pytest.raises(ValueError, match=strategy):
            ca.scheduling = "continuous"
        # ... and the other order: refused at compute()
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=4, scheduling="continuous")
        ca.add_contingencies_by_branch_id([[0], [1]])
        ca.strategy = strategy
        with pytest.raises(ValueError, match=strategy):
            ca.compute(batch_size=2)

    ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=4, scheduling="continuous")
    ca.add_contingencies_by_branch_id([[0], [1]])
    monkeypatch.setenv("GPUSIM2GRID_USE_BLOCKDIAG", "1")
    with pytest.raises(ValueError, match="uniform"):
        ca.compute(batch_size=2)
    monkeypatch.delenv("GPUSIM2GRID_USE_BLOCKDIAG")

    sw = ScenarioSweepGPU(grid, precision=PREC, nb_iter=4, scheduling="continuous")
    n_bus = grid.get_Ybus_solver().shape[0]
    Sbus = grid.get_Sbus_solver()
    sn = grid.get_sn_mva()
    sw.set_injections(np.repeat(Sbus.real[None] * sn, 2, 0), np.repeat(Sbus.imag[None] * sn, 2, 0), sn)
    # keep_final_jacobian is accepted (and does nothing): the batched adjoint
    # rebuilds each row's Jacobian (tests/python/test_batch_power_flow.py)
    sw.solver.keep_final_jacobian = True
    sw.compute(batch_size=2)
    assert sw.get_row_status().shape == (2,)
    assert n_bus > 0
