# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The tail of a continuous run (``tail_max_groups`` / ``tail_group_size``,
docs/dev_notes/outer_loops_performance.md, step 3).

Once the queue is empty and the live rows fit in ``tail_max_groups`` groups of
``tail_group_size`` slots, they are moved into the first slots and the rounds
run on those groups only, the linear solve on a small cuDSS handle cycled over
them. Only where the rows sit and which handle factorizes them change, so every
row ends as without the tail: same status, same iteration count, same V up to
cuDSS's rounding. Small groups (1, 3) make the rows move and the groups cycle
as much as possible; a missed per-slot buffer in the move shows here.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = requires_gpu

PREC = "fp32" if _cpp.is_fp32 else "fp64"
CONVERGED, NOT_SIMULATED = 0, 3


def _grid(case, distributed=False):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(getattr(pn, case)())
    if distributed:
        for g in grid.get_generators():
            grid.add_gen_slackbus(g.id, 1.)
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n = grid.get_bus_vn_kv().shape[0]
    V = grid.ac_pf(np.ones(n, dtype=complex), 30, 1e-10)
    assert V.shape[0] > 0
    return grid


@pytest.fixture(scope="module")
def case118():
    return _grid("case118")


def _same_rows(runs, ref, solver_atol):
    """every run: the reference's statuses and iterations; on its converged
    rows its NaN pattern and V (a row that never converges amplifies the other
    handle's rounding over its iterations: its last iterate is not compared)"""
    st0, it0, V0 = ref
    conv = st0 == CONVERGED
    for key, (st, it, V) in runs.items():
        np.testing.assert_array_equal(st, st0, err_msg=str(key))
        np.testing.assert_array_equal(it, it0, err_msg=str(key))
        np.testing.assert_array_equal(np.isnan(V[conv]), np.isnan(V0[conv]), err_msg=str(key))
        m = conv[:, None] & ~np.isnan(V0)
        np.testing.assert_allclose(V[m], V0[m], rtol=0, atol=solver_atol, err_msg=str(key))


def _outcome(obj, n_rows):
    return (np.asarray(obj.get_row_status()), np.asarray(obj.get_row_iterations()),
            obj.V_results.to_numpy().reshape(n_rows, -1))


# =============================================================================
# The three batch facades: the tail changes nothing but where the rows sit
# =============================================================================
def test_contingency_analysis(case118, solver_atol):
    """N-1 of case118 with handle_disconnected_grid (masked islands, NaN V)."""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case118
    nbr = len(grid.get_lines()) + len(grid.get_trafos())
    runs, timings = {}, {}
    for mg, G in ((0, 15), (16, 15), (10 ** 6, 3), (10 ** 6, 1)):
        ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous",
                                    handle_disconnected_grid=True)
        ca.tail_max_groups, ca.tail_group_size = mg, G
        ca.add_contingencies_by_branch_id([[i] for i in range(nbr)])
        ca.compute(batch_size=64)
        runs[(mg, G)] = _outcome(ca, nbr)
        timings[(mg, G)] = ca.timings
    ref = runs.pop((0, 15))
    assert np.any(ref[0] == NOT_SIMULATED) or np.any(np.isnan(ref[2]))
    _same_rows(runs, ref, solver_atol)
    t_off = timings[(0, 15)]
    assert t_off.n_tail_rounds == 0 and t_off.n_slot_moves == 0
    for key in runs:
        t = timings[key]
        assert t.n_rounds == t_off.n_rounds, key
        assert 0 < t.n_tail_rounds <= t.n_rounds and t.n_slot_moves > 0, key
        assert t.occupancy > t_off.occupancy, key
        assert "tail_compact" in t.to_dict()["gpu_compute"]


def test_injection_sweep(case118, solver_atol):
    from gpusim2grid import InjectionSweepGPU
    grid = case118
    n_rows = 40
    n_bus = grid.get_Ybus_solver().shape[0]
    sn = grid.get_sn_mva()
    Sbus = grid.get_Sbus_solver()
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.6, 1.4, size=(n_rows, 1)) * rng.uniform(0.9, 1.1, size=(n_rows, n_bus))
    p, q = Sbus.real * sn * scale, Sbus.imag * sn * scale
    runs = {}
    for mg, G in ((0, 15), (10 ** 6, 3), (10 ** 6, 1)):
        sw = InjectionSweepGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous")
        sw.tail_max_groups, sw.tail_group_size = mg, G
        sw.set_injections(p, q, sn)
        sw.compute(batch_size=16)
        runs[(mg, G)] = _outcome(sw, n_rows)
        assert (sw.timings.n_tail_rounds > 0) == (mg > 0)
    _same_rows(runs, runs.pop((0, 15)), solver_atol)


def test_scenario_sweep(solver_atol):
    """Rows tripping a branch, cutting off a leaf bus with its generator (an
    island solved on the main component), disconnecting generators (a bus
    released PV -> PQ, the slack re-weighted) and moving set-points, with the
    lost power shared beforehand (redistribute_slack): per-slot Sbus with its
    correction, slack weights, masks, pins and v_set all move with their row."""
    from gpusim2grid import ScenarioSweepGPU
    grid = _grid("case14", distributed=True)
    gens = grid.get_generators()
    n_gen = len(gens)
    n_line = len(grid.get_lines())
    leaf = [ln.id for ln in grid.get_lines() if 7 in (ln.bus1_id, ln.bus2_id)]
    leaf += [n_line + t.id for t in grid.get_trafos() if 7 in (t.bus1_id, t.bus2_id)]
    others = list(range(1, n_gen))   # every generator shares the slack: all but the first
    load_p, load_q = (np.asarray(a, dtype=float) for a in grid.get_loads_res_full()[:2])
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)

    kinds = ["plain", "trip", "island", "gen_off", "gen_v", "island+gen_v"] * 4
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
            topo[r] = list(leaf)
        if kind == "gen_off":
            gen_off[r, others[r % len(others)]] = True
        if "gen_v" in kind:
            gen_v[r, others[0]] = 1.03 + 0.001 * r

    runs = {}
    for mg, G in ((0, 15), (10 ** 6, 3), (10 ** 6, 1)):
        sw = ScenarioSweepGPU(grid, precision=PREC, nb_iter=10, handle_disconnected_grid=True,
                              redistribute_slack=True, scheduling="continuous")
        sw.tail_max_groups, sw.tail_group_size = mg, G
        sw.set_injections_from_elements(lp, lq, gp)
        sw.set_topology(topo)
        sw.set_contingency_gens(gen_off)
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=8)
        runs[(mg, G)] = _outcome(sw, n)
        assert (sw.timings.n_slot_moves > 0) == (mg > 0)
    ref = runs.pop((0, 15))
    assert np.all(ref[0] == CONVERGED)
    _same_rows(runs, ref, solver_atol)


# =============================================================================
# The outer loops: the model's slot state moves with its row
# =============================================================================
@pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                    reason="gpusim2grid built against a lightsim2grid without NROuter_*")
@pytest.mark.skipif(bool(_cpp.is_fp32), reason="the outer loops run in double precision")
def test_outer_loops(solver_atol):
    """case14 short of reactive power, sharing the slack (ReactiveLimits and
    DistributedSlack act on most rows, re-solving them in their slot)."""
    from test_outer_loops_batch import _case14, _rows, _gpu, N_ROWS
    import test_outer_loops_batch as T
    grid = _case14()
    rows = _rows(grid, n_rows=3 * N_ROWS)
    rows["lines"] = [[1 + i % 8] for i in range(3 * N_ROWS)]
    n = 3 * N_ROWS
    runs, res = {}, {}
    for mg, G in ((0, 15), (10 ** 6, 3), (10 ** 6, 1)):
        from gpusim2grid import ScenarioSweepGPU
        sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=T.MAX_IT, init_from_n_powerflow=False,
                              tol=T.TOL_MVA / grid.get_sn_mva(), keep_outer_trace=True)
        sw.tail_max_groups, sw.tail_group_size = mg, G
        sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
        sw.set_topology(rows["lines"])
        sw.compute(batch_size=8)
        runs[(mg, G)] = _outcome(sw, n)
        res[(mg, G)] = sw.get_outer_results()
        assert (sw.timings.n_slot_moves > 0) == (mg > 0)
    assert any(r.nb_outer_iterations > 0 for r in res[(0, 15)])
    _same_rows(runs, runs.pop((0, 15)), solver_atol)
    for key in runs:
        for i in range(n):
            a, b = res[key][i], res[(0, 15)][i]
            assert a.nb_outer_iterations == b.nb_outer_iterations, (key, i)
            assert a.nb_nr_iterations == b.nb_nr_iterations, (key, i)
            assert a.decisions == b.decisions, (key, i)


# =============================================================================
# Settings
# =============================================================================
def test_settings_and_when_the_tail_stays_off(case118):
    from gpusim2grid import ContingencyAnalysisGPU
    grid = case118
    ca = ContingencyAnalysisGPU(grid, precision=PREC, nb_iter=10, scheduling="continuous")
    assert (ca.tail_max_groups, ca.tail_group_size) == (16, 15)
    with pytest.raises(ValueError):
        ca.tail_max_groups = -1
    with pytest.raises(ValueError):
        ca.tail_group_size = 0
    ca.add_contingencies_by_branch_id([[i] for i in range(40)])
    ca.tail_group_size = 2
    # S <= group size: nothing to gain
    ca.compute(batch_size=2)
    assert ca.timings.n_tail_rounds == 0
    # a policy keeping its factors across iterations: no tail
    ca.strategy = "direct_base_case_factors"
    ca.compute(batch_size=16)
    assert ca.timings.n_tail_rounds == 0
    # a chunked run: no tail
    ca.strategy = "direct_refactor_every"
    ca.scheduling = "chunked"
    ca.compute(batch_size=16)
    assert ca.timings.n_tail_rounds == 0
    ca.scheduling = "continuous"
    ca.compute(batch_size=16)
    assert ca.timings.n_tail_rounds > 0


def test_calibrate_tail_max_groups(case118):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(case118, precision=PREC, nb_iter=10, scheduling="continuous")
    ca.tail_group_size = 4
    c = ca.calibrate_tail_max_groups(n_repeat=2, batch_size=32)
    assert (c.batch_size, c.group_size, c.n_repeat) == (32, 4, 2)
    assert c.t_full_ms > 0 and c.t_group_ms > 0
    assert c.max_groups == int(np.floor(c.t_full_ms / c.t_group_ms))
    assert ca.tail_max_groups == 16          # measured, not set
    assert ca.calibrate_tail_max_groups(n_repeat=1, batch_size=4).max_groups == 0
