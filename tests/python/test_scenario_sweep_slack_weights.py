# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ScenarioSweepGPU.set_gen_slack_weights / set_storage_slack_weights: each row gives some units
distributed-slack weights of their own (lightsim2grid PR #303).

Oracles: the grid itself holding the row's weights (``add_gen_slackbus`` /
``remove_gen_slackbus``, then a one-off ``ac_pf``; voltages compared up to a common rotation,
since the batch keeps the angle reference it picked while a grid taking its reference unit out
of the slack moves it), and lightsim2grid's own ``ScenarioSweepCPP`` given the same weights --
alone, with ``redistribute_slack``, with ``compute_physical_violations`` (the generator
active-power check) and under its outer loops.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_bridge", False), reason="needs the lightsim2grid bridge"),
]
needs_ls2g_weights = pytest.mark.skipif(
    not getattr(_cpp, "have_ls2g_branch_params", False),
    reason="lightsim2grid's own ScenarioSweep with per-row slack weights (PR #303) is the oracle")

MAX_IT, TOL = 20, 1e-10
N_ROWS = 5
GEN_IDS = np.array([0, 2, 4])


def _weights():
    """5 rows: row 0 changes nothing (all NaN), row 3 takes three generators out of the slack
    (the reference bus's among them)"""
    w = np.full((N_ROWS, 3), np.nan)
    w[1] = [10., 0., 1.]
    w[2, 1] = 0.
    w[3] = [0., 0., 0.]
    w[4, 0] = 50.
    return w


def _gap_up_to_rotation(a, b):
    """largest gap between two voltage vectors once their best common rotation is taken out"""
    rot = np.vdot(b, a)
    rot = rot / abs(rot) if abs(rot) > 0. else 1.
    return np.max(np.abs(a - b * rot))


def _case14_all_slack():
    """pandapower's case14, every generator sharing the slack (weight 1 + id), solved."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    for g in range(len(grid.get_generators())):
        grid.add_gen_slackbus(g, 1. + g)
    grid.change_algorithm(AlgorithmType.NR_KLU)
    grid.tell_solver_need_reset()
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL).shape[0] > 0
    return grid


def _feeder_colocated():
    """test_gen_p_violations' 4-bus radial feeder (80 MW / 60 MVAr on bus 3) with one more
    generator: generator 2, on bus 0 next to generator 0, takes NO part in the grid's slack
    (generators 0 and 1 share it) -- only a row's own weight brings it in, so the grid's
    active-power plan and pre-pass data lack it. Every generator has an active range the rows'
    shares leave. Solved; returns (grid, v0)."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid

    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, 138.), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators(np.array([0., 10., 5.]), np.array([1.02, 1.05, 1.02]),
                         np.full(3, -1e3), np.full(3, 1e3), np.array([0, 1, 0]))
    grid.add_gen_slackbus(0, 1.)
    grid.add_gen_slackbus(1, 1.)
    grid.set_gen_p_limits(np.zeros(3), np.array([30., 30., 10.]))
    grid.tell_solver_need_reset()
    v0 = grid.ac_pf(np.ones(4, dtype=complex), MAX_IT, TOL)
    assert v0.shape[0] > 0
    return grid, v0


COLOCATED_W = np.array([[np.nan], [1.], [3.]])   # generator 2's weight per row: none, 1, 3


def _colocated_reference(grid, v0, **options):
    """lightsim2grid's ScenarioSweep on _feeder_colocated with COLOCATED_W, the active-power
    check on (tolerance 0); `options` its attributes (e.g. redistribute_slack)"""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    import test_gen_p_violations as gpv

    n_rows = COLOCATED_W.shape[0]
    ls = ScenarioSweepCPP(grid)
    for k, v in options.items():
        setattr(ls, k, v)
    gpv._ls_enable(ls, 0.)
    ls.modify_load_p(np.repeat(_elements(grid)[0][None, :], n_rows, axis=0))
    ls.set_gen_slack_weights(np.array([2]), COLOCATED_W)
    ls.compute(v0.copy(), MAX_IT, TOL)
    assert np.all(ls.converged_mask())
    ref = gpv._p_records(ls.get_physical_violations())
    # generator 2 (only the rows' weights make it share) is reported: the test checks something
    assert any(rec[1] == 2 for row in ref for rec in row)
    return ref


def _elements(grid):
    load_p, load_q = grid.get_loads_res_full()[:2]
    return (np.asarray(load_p, dtype=np.float64), np.asarray(load_q, dtype=np.float64),
            np.asarray(grid.get_gen_target_p(), dtype=np.float64))


def _sweep(grid, n_rows=N_ROWS, gen_ids=GEN_IDS, weights=None, **kwargs):
    from gpusim2grid import ScenarioSweepGPU

    sw = ScenarioSweepGPU(grid, nb_iter=MAX_IT, tol_base=TOL, **kwargs)
    load_p, load_q, gen_p = _elements(grid)
    rep = lambda a: np.repeat(a[None, :], n_rows, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    if weights is not None:
        sw.set_gen_slack_weights(gen_ids, weights)
    return sw


def _V(sw, n_rows=N_ROWS):
    sw.compute(batch_size=n_rows)
    return sw.solver.V_results.to_numpy().reshape(n_rows, sw.n_bus)


def _grid_of_row(grid, weights, row, gen_ids=GEN_IDS):
    """`grid` holding row `row`'s weights, given the way LSGrid takes them"""
    g = grid.copy()
    for k, gen_id in enumerate(gen_ids):
        w = weights[row, k]
        if np.isnan(w):
            continue
        if w > 0.:
            g.add_gen_slackbus(int(gen_id), float(w))
        else:
            g.remove_gen_slackbus(int(gen_id))
    g.tell_solver_need_reset()
    return g


# --------------------------------------------------------------------- tests
def test_rows_match_the_grid_given_their_weights(solver_atol):
    """Every row solves as the grid holding its weights (up to the angle reference)."""
    grid = _case14_all_slack()
    w = _weights()
    V = _V(_sweep(grid, weights=w))
    V_plain = _V(_sweep(grid))
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    for row in range(N_ROWS):
        ref = _grid_of_row(grid, w, row)
        V_ref = ref.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
        assert V_ref.shape[0] > 0, f"row {row}: the reference diverged"
        assert _gap_up_to_rotation(V[row], V_ref[buses]) < solver_atol, f"row {row}"
        if row > 0:   # the weights did act
            assert _gap_up_to_rotation(V[row], V_plain[row]) > 100 * solver_atol, f"row {row}"


@needs_ls2g_weights
def test_rows_match_lightsim2grid_scenario_sweep(solver_atol):
    """Row for row lightsim2grid's ScenarioSweep given the same weights (same angle reference),
    together with generator contingencies (a weighted unit the row disconnects takes no share)."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP

    grid = _case14_all_slack()
    w = _weights()
    n_gen = len(grid.get_generators())
    gen_off = np.zeros((N_ROWS, n_gen), dtype=bool)
    gen_off[1, 1] = True     # a participant the row gives no weight of its own, disconnected
    gen_off[4, 0] = True     # the row gives generator 0 a weight of 50 and disconnects it: no share
    sw = _sweep(grid, weights=w)
    sw.set_contingency_gens(gen_off)
    V = _V(sw)
    ls = ScenarioSweepCPP(grid)
    ls.set_gen_slack_weights(GEN_IDS, w)
    ls.set_contingency_gens(gen_off)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
    assert np.all(ls.converged_mask())
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(ls.get_voltages())[:, buses]
    for row in range(N_ROWS):
        assert _gap_up_to_rotation(V[row], V_ls[row]) < solver_atol, f"row {row}"


def test_storage_weights(solver_atol):
    """A storage unit's row weight: the grid holding it (add_storage_slackbus)."""
    pypo = pytest.importorskip("pypowsybl")
    from lightsim2grid.network import init_from_pypowsybl
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    net = pypo.network.create_ieee14()
    net.create_batteries(id="BAT", voltage_level_id="VL2", bus_id="B2", target_p=10., target_q=0.,
                         min_p=-300., max_p=300.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id={"B1-G": 1., "B2-G": 1.}, sort_index=False,
                                   buses_for_sub=False)
    if not hasattr(grid, "add_storage_slackbus"):
        pytest.skip("this lightsim2grid build has no storage slack")
    grid.change_algorithm(AlgorithmType.NR_KLU)
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL).shape[0] > 0
    sw = _sweep(grid, n_rows=3)
    w = np.array([[np.nan], [2.], [0.5]])
    sw.set_storage_slack_weights(["BAT"], w)
    V = _V(sw, n_rows=3)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    for row in range(3):
        ref = grid.copy()
        if not np.isnan(w[row, 0]):
            ref.add_storage_slackbus(0, float(w[row, 0]))
        ref.tell_solver_need_reset()
        V_ref = ref.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
        assert V_ref.shape[0] > 0
        assert _gap_up_to_rotation(V[row], V_ref[buses]) < solver_atol, f"row {row}"


@needs_ls2g_weights
def test_redistribute_slack_shares_with_the_rows_weights(solver_atol):
    """With redistribute_slack, the bounded pre-pass shares what a row loses with its weights:
    lightsim2grid's ScenarioSweep with the same option and weights."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    import test_gen_p_violations as gpv

    grid, v0 = gpv._case14_shared_slack(margin_mw=2.)
    gens = grid.get_generators()
    n_gen = len(gens)
    slack_gens = [g for g in range(n_gen) if gens[g].is_slack]
    gen_ids = np.array(slack_gens)
    w = np.full((N_ROWS, len(gen_ids)), np.nan)
    w[1, 0] = 10.
    w[2, 1] = 0.
    w[3] = 1.
    w[4, -1] = 0.
    # every row loses a generator that is not in the slack: something to share
    lost = next(g for g in range(n_gen) if not gens[g].is_slack)
    gen_off = np.zeros((N_ROWS, n_gen), dtype=bool)
    gen_off[:, lost] = True
    sw = _sweep(grid, gen_ids=gen_ids, weights=w, redistribute_slack=True)
    sw.set_contingency_gens(gen_off)
    V = _V(sw)
    ls = ScenarioSweepCPP(grid)
    ls.redistribute_slack = True
    ls.set_gen_slack_weights(gen_ids, w)
    ls.set_contingency_gens(gen_off)
    ls.compute(v0.copy(), MAX_IT, TOL)
    assert np.all(ls.converged_mask())
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(ls.get_voltages())[:, buses]
    for row in range(N_ROWS):
        assert _gap_up_to_rotation(V[row], V_ls[row]) < solver_atol, f"row {row}"


@needs_ls2g_weights
def test_gen_p_check_follows_the_rows_weights(solver_atol):
    """The slack machines' active-power check splits each bus' share with the row's weights:
    lightsim2grid's ScenarioSweep records."""
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    import test_gen_p_violations as gpv

    grid, v0 = gpv._case14_shared_slack(margin_mw=0.05)
    gens = grid.get_generators()
    gen_ids = np.array([g for g in range(len(gens)) if gens[g].is_slack])
    w = np.full((N_ROWS, len(gen_ids)), np.nan)
    w[1, 0] = 10.
    w[2, 1] = 0.
    w[3] = [1., 4., 0.5]
    w[4, -1] = 0.
    load_p, load_q, gen_p = _elements(grid)
    load_p = np.repeat(load_p[None, :], N_ROWS, axis=0) * (1. + 0.01 * np.arange(N_ROWS))[:, None]
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, nb_iter=MAX_IT, tol_base=TOL, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(load_p, np.repeat(load_q[None, :], N_ROWS, axis=0),
                                    np.repeat(gen_p[None, :], N_ROWS, axis=0))
    sw.set_gen_slack_weights(gen_ids, w)
    sw.compute(batch_size=N_ROWS)
    got = gpv._p_records(sw.get_physical_violations())

    ls = ScenarioSweepCPP(grid)
    gpv._ls_enable(ls, 0.)
    ls.modify_load_p(load_p)
    ls.set_gen_slack_weights(gen_ids, w)
    ls.compute(v0.copy(), MAX_IT, TOL)
    ref = gpv._p_records(ls.get_physical_violations())
    gpv._assert_same(ref, got, gpv._mw_atol(grid, solver_atol))
    assert any(ref), "nothing reported: the test checks nothing"
    # the weights decide what is reported: row 0 (none given) differs from a weighted row
    assert sorted(x[:3] for x in ref[0]) != sorted(x[:3] for x in ref[1])


@needs_ls2g_weights
def test_gen_p_plan_follows_weights_set_while_the_checks_are_off(solver_atol):
    """A plan built while the active-power check was on, the check turned off, weights set
    bringing in a unit the grid's plan lacks, the check back on: the plan lists that unit
    (lightsim2grid's ScenarioSweep records). Turning the check on keeps a plan it finds, so the
    weights must have refreshed it while it was off."""
    import test_gen_p_violations as gpv

    grid, v0 = _feeder_colocated()
    ref = _colocated_reference(grid, v0)
    n_rows = COLOCATED_W.shape[0]
    sw = _sweep(grid, n_rows=n_rows, compute_physical_violations=True)   # the grid's plan
    sw.physical_violation_tol_mva = 0.
    sw.compute_physical_violations = False
    sw.set_gen_slack_weights([2], COLOCATED_W)
    sw.compute_physical_violations = True
    sw.compute(batch_size=n_rows)
    got = gpv._p_records(sw.get_physical_violations())
    gpv._assert_same(ref, got, gpv._mw_atol(grid, solver_atol))


@needs_ls2g_weights
def test_refused_weights_leave_the_previous_ones(monkeypatch, solver_atol):
    """set_gen_slack_weights refused while it re-reads what depends on the row-weighted units
    (a lightsim2grid without PR #303 refuses the active-power plan of such units): the call
    changes nothing, the next compute() is the one the previous weights give."""
    import test_gen_p_violations as gpv

    grid, v0 = _feeder_colocated()
    ref = _colocated_reference(grid, v0)
    n_rows = COLOCATED_W.shape[0]
    sw = _sweep(grid, n_rows=n_rows, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_gen_slack_weights([2], COLOCATED_W)

    def refuse(*args, **kwargs):
        raise RuntimeError("the generator active-power check of rows giving slack weights of their own "
                           "needs a lightsim2grid with per-row slack weights (lightsim2grid PR #303)")

    with monkeypatch.context() as m:
        m.setattr(_cpp, "_extract_gen_p_plan_from_lsgrid", refuse)
        with pytest.raises(RuntimeError, match="PR #303"):
            sw.set_gen_slack_weights([2], np.full((n_rows, 1), 7.))
    sw.compute(batch_size=n_rows)
    gpv._assert_same(ref, gpv._p_records(sw.get_physical_violations()), gpv._mw_atol(grid, solver_atol))


def test_explicit_prepass_data_lacking_a_weighted_unit_is_refused():
    """redistribute_slack builds the solve's row weights from its pre-pass data: data set
    explicitly that lacks a unit a row weights would silently drop its share (the same row
    without the pre-pass keeps it). set_gen_slack_weights leaves such data alone, compute()
    refuses it; the grid's data (set_slack_redistribution_data_from_grid) lists the unit."""
    grid, _ = _feeder_colocated()
    n_rows = COLOCATED_W.shape[0]
    sw = _sweep(grid, n_rows=n_rows, redistribute_slack=True)
    sw.set_gen_slack_weights([2], COLOCATED_W)
    # the grid's data without the rows' units: generator 2 is not listed
    sw.set_slack_redistribution_data(_cpp._extract_slack_redistribution_data_from_lsgrid(grid, sw.n_bus))
    with pytest.raises(ValueError, match="pre-pass data does not list it"):
        sw.compute(batch_size=n_rows)
    sw.set_gen_slack_weights([2], COLOCATED_W)   # leaves the caller's data as it is
    with pytest.raises(ValueError, match="pre-pass data does not list it"):
        sw.compute(batch_size=n_rows)
    sw.set_slack_redistribution_data_from_grid()
    sw.compute(batch_size=n_rows)
    assert np.all(np.isfinite(sw.solver.V_results.to_numpy()))


@needs_ls2g_weights
def test_explicit_gen_p_plan_lacking_a_weighted_unit_is_refused():
    """The active-power check's plan set explicitly (set_gen_p_capability) is the caller's:
    set_gen_slack_weights does not replace it by the grid's, and compute() refuses a row
    weighting a unit it lacks -- it would neither check that unit nor count it in its bus' total."""
    grid, _ = _feeder_colocated()
    n_rows = COLOCATED_W.shape[0]
    sw = _sweep(grid, n_rows=n_rows)
    sw.set_gen_p_capability(_cpp._extract_gen_p_plan_from_lsgrid(grid, sw.n_bus))   # without generator 2
    sw.compute_physical_violations = True   # keeps the plan it finds
    sw.set_gen_slack_weights([2], COLOCATED_W)
    with pytest.raises(ValueError, match="plan does not list it"):
        sw.compute(batch_size=n_rows)
    # the grid's plan lists it
    sw.set_gen_p_capability_from_grid()
    sw.compute(batch_size=n_rows)


@needs_ls2g_weights
@pytest.mark.parametrize("init_from_n", [False, True])
def test_outer_loops_share_with_the_rows_weights(init_from_n, solver_atol):
    """Under NROuter_*, the DistributedSlack loop shares with the row's weights (and a weighted
    row starts from the grid's targets with init_from_n_powerflow): lightsim2grid's ScenarioSweep
    under the same algorithm."""
    if not getattr(_cpp, "have_ls2g_outer_loops", False):
        pytest.skip("lightsim2grid without its outer loops")
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    from lightsim2grid.scenarioSweep import ScenarioSweepCPP
    import test_outer_loops_batch as olb
    from gpusim2grid import ScenarioSweepGPU

    grid = olb._case14()
    n_gen = len(grid.get_generators())
    gen_ids = np.arange(n_gen)
    w = np.full((N_ROWS, n_gen), np.nan)
    w[1, 0] = 900.
    w[2, 1] = 0.
    w[3, :] = 1.
    w[4, 2] = 50.
    load_p = np.asarray(grid.get_load_target_p())
    load_p = np.repeat(load_p[None, :], N_ROWS, axis=0) * (1. + 0.05 * np.arange(N_ROWS))[:, None]
    load_q = np.repeat(np.array([l.target_q_mvar for l in grid.get_loads()], dtype=float)[None, :], N_ROWS, axis=0)
    gen_p = np.repeat(np.array([g.target_p_mw for g in grid.get_generators()])[None, :], N_ROWS, axis=0)

    ls = ScenarioSweepCPP(grid)
    ls.change_algorithm(olb.OUTER)
    ls.init_from_n_powerflow = init_from_n
    # the same inputs on both sides: set_injections_from_elements sets every generator's target, so
    # a resumed row does not take the targets the grid's loops moved for them (modify_gen_p, the same)
    ls.modify_load_p(load_p)
    ls.modify_load_q(load_q)
    ls.modify_gen_p(gen_p)
    ls.set_gen_slack_weights(gen_ids, w)
    ls.compute(np.asarray(grid.get_V(), dtype=complex), olb.MAX_IT, olb.TOL_MVA)

    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=olb.MAX_IT, init_from_n_powerflow=init_from_n,
                          tol=olb.TOL_MVA / grid.get_sn_mva(), ir_n_steps=1)
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_gen_slack_weights(gen_ids, w)
    sw.compute(batch_size=N_ROWS)
    conv_ls = np.asarray(ls.converged_mask(), dtype=bool)
    assert conv_ls.all()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = sw.solver.V_results.to_numpy().reshape(N_ROWS, sw.n_bus)
    V_ls = np.asarray(ls.get_voltages())[:, buses]
    for row in range(N_ROWS):
        assert _gap_up_to_rotation(V[row], V_ls[row]) < solver_atol, f"row {row}"
    res, res_ls = sw.get_outer_results(), ls.get_outer_results()
    for row in range(N_ROWS):
        assert res[row].nb_outer_iterations == res_ls[row].nb_outer_iterations, f"row {row}"


def test_refusals():
    """What the setters and compute() refuse."""
    import test_gen_p_violations as gpv

    grid, _ = gpv._case14_shared_slack()
    gens = grid.get_generators()
    n_gen = len(gens)
    off_slack = next(g for g in range(n_gen) if not gens[g].is_slack)
    sw = _sweep(grid)
    with pytest.raises(ValueError, match="negative or infinite"):
        sw.set_gen_slack_weights([0], np.full((N_ROWS, 1), -1.))
    with pytest.raises(ValueError, match="out of range"):
        sw.set_gen_slack_weights([n_gen], np.ones((N_ROWS, 1)))
    with pytest.raises(ValueError, match="named twice"):
        sw.set_gen_slack_weights([0, 0], np.ones((N_ROWS, 2)))
    with pytest.raises(ValueError, match="shape"):
        sw.set_gen_slack_weights([0], np.ones((N_ROWS, 2)))
    with pytest.raises(ValueError, match="rows"):
        sw.set_gen_slack_weights([0], np.ones((N_ROWS + 1, 1)))
    with pytest.raises(ValueError, match="named"):
        sw.set_gen_slack_weights(["no such generator"], np.ones((N_ROWS, 1)))
    # a positive weight to a generator on a bus that does not share the grid's slack (unless
    # that bus also holds a participant)
    slack_buses = {gens[g].bus_id for g in range(n_gen) if gens[g].is_slack}
    if gens[off_slack].bus_id not in slack_buses:
        w = np.full((N_ROWS, 1), np.nan)
        w[2, 0] = 1.
        sw.set_gen_slack_weights([off_slack], w)
        with pytest.raises(ValueError, match="no unit takes part"):
            sw.compute(batch_size=N_ROWS)
        # ... 0 there is fine (it takes it out: it was not in)
        w[2, 0] = 0.
        sw.set_gen_slack_weights([off_slack], w)
        sw.compute(batch_size=N_ROWS)
