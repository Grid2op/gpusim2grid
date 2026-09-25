# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_physical_violations, active part -- the per-machine active power of
the distributed slack (lightsim2grid's GenPCheck.hpp parity: generators AND
storage units, LOW_P / HIGH_P on a GENERATOR / STORAGE) on the three batch
facades.

The slack is solved inside the Jacobian by participation factors that know
nothing about limits, so a participating machine's converged active power --
its target plus its share of the imbalance -- can leave its [min_p, max_p].
The reference is lightsim2grid itself (ContingencyAnalysisCPP /
InjectionSweepCPP / ScenarioSweepCPP with ``compute_physical_violations``),
whose per-row records must be reproduced record for record, and a single
``ac_pf`` for the published value (``res_p_mw``: the check re-derives that very
number). Both sides report the container id, so no bus mapping is needed; the
bus reactive records that share the report are filtered out here (they are
pinned in test_bus_q_violations.py).
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from test_bus_q_violations import (
    MAX_IT, TOL, _solve, _ls_enable, _ls_physical, _ls_physical_n, _Ref, _all_n1, _elements,
)

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="the active-power plan is built by lightsim2grid through the C++ bridge")

GEN, STO = 5, 6          # ViolationElementType.GENERATOR / STORAGE
HIGH_P, LOW_P = 7, 8     # LimitViolationType


# ------------------------------------------------------------------ grids
def _feeder_grid(w0=1., w1=1.):
    """upstream's fixture: the 4-bus radial feeder 0-1-2-3 with the 80 MW / 60
    MVAr load on bus 3, and the slack SHARED between gen 0 (bus 0, target 0
    MW) and gen 1 (bus 1, target 10 MW). Unsolved."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, 138.), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators(np.array([0., 10.]), np.array([1.02, 1.05]),
                         np.full(2, -1e3), np.full(2, 1e3), np.array([0, 1]))
    grid.add_gen_slackbus(0, w0)
    grid.add_gen_slackbus(1, w1)
    grid.tell_solver_need_reset()
    return grid


def _feeder_one_slack_bus(w0=1., w1=3.):
    """the feeder with BOTH generators on bus 0 sharing the slack: ONE slack
    bus, so the tuple-mode path (whose trivial ledger has no distributed
    slack: several slack ids become several angle references, and
    ``extract_grid_arrays`` hands out uniform weights) solves the very same
    problem as lightsim2grid -- while the split of that bus' residual between
    its two machines is still by their own factors. Unsolved."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, 138.), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators(np.array([0., 10.]), np.array([1.02, 1.02]),
                         np.full(2, -1e3), np.full(2, 1e3), np.array([0, 0]))
    grid.add_gen_slackbus(0, w0)
    grid.add_gen_slackbus(1, w1)
    grid.tell_solver_need_reset()
    return grid


def _feeder_storage_grid(w_gen=1., w_sto=1., target_p_mw=-10.):
    """upstream's fixture: the same feeder with the slack SHARED between gen 0
    (bus 0) and a battery on bus 1 discharging ``-target_p_mw`` MW (the
    storage container works in the LOAD convention). Unsolved."""
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    grid = LSGrid()
    grid.set_sn_mva(100.)
    grid.set_init_vm_pu(1.0)
    grid.init_bus(4, 1, np.full(4, 138.), 0, 0)
    grid.init_powerlines(np.full(3, 0.01), np.full(3, 0.1), np.zeros(3, dtype=complex),
                         np.array([0, 1, 2]), np.array([1, 2, 3]))
    grid.init_loads(np.array([80.]), np.array([60.]), np.array([3]))
    grid.init_generators(np.array([0.]), np.array([1.02]),
                         np.full(1, -1e3), np.full(1, 1e3), np.array([0]))
    grid.init_storages_full(np.array([target_p_mw]), np.array([0.]), [True],
                            np.array([1.05]), np.array([-1e3]), np.array([1e3]),
                            np.array([1], dtype=np.int32))
    grid.add_gen_slackbus(0, w_gen)
    grid.add_storage_slackbus(0, w_sto)
    grid.tell_solver_need_reset()
    return grid


def _ac_pf_res_p(grid):
    """every machine's converged active power as a single ac_pf publishes it,
    GENERATOR convention (a storage unit's res_p_mw negated): the number the
    check re-derives. Leaves the grid solved."""
    V = grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL)
    assert V.shape[0] > 0
    gens = np.array([g.res_p_mw for g in grid.get_generators()])
    stos = np.array([-s.res_p_mw for s in grid.get_storages()])
    return gens, stos


def _case14_shared_slack(margin_mw=0.05):
    """pandapower case14 with the slack SHARED between the ext_grid (the last
    generator of the loader) and two pandapower generators (gens 1 and 2,
    weights 2 / 1, the ext_grid keeping 1), every participant given an active
    range ``margin_mw`` around its base-case output: any N-1 that changes the
    losses pushes some of them out. Returns (grid, v0)."""
    pn = pytest.importorskip("pandapower.networks")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.add_gen_slackbus(1, 2.)
    grid.add_gen_slackbus(2, 1.)
    grid.tell_solver_need_reset()
    gens_p, _ = _ac_pf_res_p(grid)
    gens = grid.get_generators()
    n_gen = len(gens)
    assert sum(g.is_slack for g in gens) == 3
    pmin = np.full(n_gen, np.nan); pmax = np.full(n_gen, np.nan)
    for g in range(n_gen):
        if gens[g].is_slack:
            pmin[g] = gens_p[g] - margin_mw
            pmax[g] = gens_p[g] + margin_mw
    grid.set_gen_p_limits(pmin, pmax)
    grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0


def _ieee14_storage_slack(margin_mw=0.05):
    """IEEE 14 (pypowsybl) with a battery on bus B3 (limits read off the
    IIDM battery's own min_p / max_p) sharing the distributed slack with
    generators B1-G and B2-G, every participant then given a tight range
    around its base-case output. Returns (grid, v0)."""
    pypo = pytest.importorskip("pypowsybl")
    from lightsim2grid.network import init_from_pypowsybl
    net = pypo.network.create_ieee14()
    net.create_batteries(id="BAT", voltage_level_id="VL3", bus_id="B3", target_p=10.,
                         target_q=0., min_p=-300., max_p=300.)
    net.update_loads(id="B3-L", p0=net.get_loads().at["B3-L", "p0"] + 50.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id={"B1-G": 1., "B2-G": 1.},
                                   sort_index=False, buses_for_sub=False)
    if not hasattr(grid, "add_storage_slackbus") or not hasattr(grid, "set_storage_p_limits"):
        pytest.skip("this lightsim2grid build has no storage slack / storage active limits")
    # the loader read the IIDM battery's own range
    sto = grid.get_storages()[0]
    assert sto.min_p_mw == -300. and sto.max_p_mw == 300.
    grid.add_storage_slackbus(0, 0.25)
    grid.tell_solver_need_reset()
    gens_p, stos_p = _ac_pf_res_p(grid)
    gens = grid.get_generators()
    n_gen = len(gens)
    pmin = np.full(n_gen, np.nan); pmax = np.full(n_gen, np.nan)
    for g in range(n_gen):
        if gens[g].is_slack:
            pmin[g] = gens_p[g] - margin_mw
            pmax[g] = gens_p[g] + margin_mw
    grid.set_gen_p_limits(pmin, pmax)
    grid.set_storage_p_limits(np.array([stos_p[0] - margin_mw]), np.array([stos_p[0] + margin_mw]))
    grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0


# ------------------------------------------------------------------ helpers
def _p_records(rows):
    """[(element_type, element_id, type, value, limit)] per row, the active
    records only (the bus reactive ones share the report)."""
    return [[(int(v.element_type), int(v.element_id), int(v.violation_type),
              float(v.value), float(v.limit))
             for v in row if int(v.element_type) in (GEN, STO)] for row in rows]


def _assert_ranked(rows):
    """gpusim2grid's documented order: LOW_P records, then HIGH_P ones
    (generators and storage units ranked together), each type sorted by
    |value - limit|, largest first."""
    group = {LOW_P: 0, HIGH_P: 1}
    for r, row in enumerate(rows):
        groups = [group[x[2]] for x in row]
        assert groups == sorted(groups), f"row {r}: types not grouped LOW_P then HIGH_P: {row}"
        for x, y in zip(row, row[1:]):
            if x[2] == y[2]:
                assert abs(x[3] - x[4]) >= abs(y[3] - y[4]), f"row {r}: not most severe first: {row}"


def _assert_same(ref, got, atol_mw):
    """Same records as lightsim2grid (which reports them in container order;
    ours are ranked by type and severity, so compared as sets)."""
    assert len(ref) == len(got)
    _assert_ranked(got)
    for r, (a, b) in enumerate(zip(ref, got)):
        a, b = sorted(a), sorted(b)
        assert [x[:3] for x in a] == [x[:3] for x in b], f"row {r}: {a} vs {b}"
        for x, y in zip(a, b):
            np.testing.assert_allclose(y[3], x[3], atol=atol_mw, err_msg=f"row {r} value")
            np.testing.assert_allclose(y[4], x[4], atol=atol_mw, err_msg=f"row {r} limit")


def _mw_atol(grid, solver_atol):
    return 50. * float(grid.get_sn_mva()) * solver_atol


def _ref_ca(grid, v0, contingencies, tol_mw=0.):
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(grid)
    _ls_enable(ca, tol_mw)
    for c in contingencies:
        if len(c) == 1:
            ca.add_n1(int(c[0]))
        else:
            ca.add_nk([int(x) for x in c])
    ca.compute(v0.copy(), MAX_IT, TOL)
    return _Ref(ca, contingencies)


def _gpu_ca(grid, contingencies, tol_mw=0., **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True, **kwargs)
    ca.physical_violation_tol_mva = tol_mw
    ca.add_contingencies_by_branch_id(contingencies)
    ca.compute(batch_size=8)
    return ca


def _one_row_is(grid, tol_mw=0.):
    """the base injections as a one-row injection sweep: row 0 IS the base
    case, and so is the "n" report"""
    from gpusim2grid import InjectionSweepGPU
    load_p, load_q, gen_p = _elements(grid)
    sw = InjectionSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = tol_mw
    sw.set_injections_from_elements(load_p[None, :], load_q[None, :], gen_p[None, :])
    sw.compute()
    return sw


# ------------------------------------------------------------- contingency
@needs_bridge
def test_ca_n1_matches_lightsim2grid(solver_atol):
    from gpusim2grid.contingency_analysis import ViolationCategory, ViolationElementType
    grid, v0 = _case14_shared_slack()
    ctgs = _all_n1(grid)
    ref = _ref_ca(grid, v0, ctgs)
    gpu = _gpu_ca(grid, ctgs)

    got = gpu.get_physical_violations()
    ref_rows = _p_records(ref.get_physical_violations())
    assert sum(len(r) for r in ref_rows) > 0, "an N-1 changes the losses: some share must move"
    _assert_same(ref_rows, _p_records(got), _mw_atol(grid, solver_atol))
    for row in got:
        for v in row:
            if v.element_type == ViolationElementType.GENERATOR:
                assert v.category == ViolationCategory.PHYSICAL
                assert v.side == 0
    # the base case sits inside its own margin: nothing on either side
    assert _p_records([ref.get_physical_violations_n()]) == [[]]
    assert _p_records([gpu.get_physical_violations_n()]) == [[]]
    assert not gpu.get_physical_violations_truncated().any()
    assert gpu.solver.timings.t_gen_p_check.wall_ms >= 0.
    assert gpu.has_gen_p_capability


@needs_bridge
def test_wide_tolerance_hides_everything_and_flag_off_raises():
    from gpusim2grid import ContingencyAnalysisGPU
    grid, v0 = _case14_shared_slack()
    ctgs = _all_n1(grid)[:5]
    gpu = _gpu_ca(grid, ctgs, tol_mw=1e6)
    assert all(len(r) == 0 for r in _p_records(gpu.get_physical_violations()))
    ca = ContingencyAnalysisGPU(grid, nb_iter=10)
    assert ca.has_gen_p_capability is False
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute()
    with pytest.raises(RuntimeError):
        ca.get_physical_violations()


# ------------------------------------------------ the number itself (ac_pf)
@needs_bridge
def test_reports_the_power_ac_pf_publishes(solver_atol):
    """each generator's converged active power is its target plus its own
    share (weights 1 / 3): the very res_p_mw a single ac_pf publishes"""
    from gpusim2grid.contingency_analysis import LimitViolationType, ViolationElementType
    grid = _feeder_grid(1., 3.)
    p_ref, _ = _ac_pf_res_p(grid)
    pmax = p_ref[1] - 1.
    grid.set_gen_p_limits(np.array([np.nan, np.nan]), np.array([np.nan, pmax]))
    grid.tell_solver_need_reset()
    _solve(grid)
    sw = _one_row_is(grid)
    for rows in (sw.get_physical_violations(), [sw.get_physical_violations_n()]):
        recs = _p_records(rows)[0]
        assert len(recs) == 1
        et, eid, vt, value, limit = recs[0]
        assert et == ViolationElementType.GENERATOR and eid == 1
        assert vt == LimitViolationType.HIGH_P and limit == pmax
        np.testing.assert_allclose(value, p_ref[1], atol=_mw_atol(grid, solver_atol))
    # the other side
    pmin = p_ref[0] + 1.
    grid.set_gen_p_limits(np.array([pmin, np.nan]), np.array([np.nan, np.nan]))
    grid.tell_solver_need_reset()
    _solve(grid)
    recs = _p_records([_one_row_is(grid).get_physical_violations_n()])[0]
    assert len(recs) == 1 and recs[0][:3] == (GEN, 0, LOW_P)
    np.testing.assert_allclose(recs[0][3], p_ref[0], atol=_mw_atol(grid, solver_atol))
    # within the limits, or no limit at all: nothing
    grid.set_gen_p_limits(np.full(2, -1e3), p_ref + 5.)
    grid.tell_solver_need_reset(); _solve(grid)
    assert _p_records([_one_row_is(grid).get_physical_violations_n()]) == [[]]
    grid.set_gen_p_limits(np.array([]), np.array([]))
    grid.tell_solver_need_reset(); _solve(grid)
    sw = _one_row_is(grid)
    assert _p_records([sw.get_physical_violations_n()]) == [[]]
    assert sw.solver._s.physical_checks.gen_p_plan.n_entries == 0


@needs_bridge
def test_each_machine_at_its_own_share(solver_atol):
    """uneven weights (1 / 3): gen 1 takes three quarters of the imbalance --
    both reported, each against its own limit"""
    grid = _feeder_grid(1., 3.)
    p_ref, _ = _ac_pf_res_p(grid)
    np.testing.assert_allclose(p_ref[1] - 10., 3. * p_ref[0], rtol=1e-6)   # three times gen 0's share
    grid.set_gen_p_limits(np.full(2, np.nan), p_ref - 1.)
    grid.tell_solver_need_reset(); _solve(grid)
    # both 1 MW above their max_p: a tie in severity, so compare by id
    recs = sorted(_p_records([_one_row_is(grid).get_physical_violations_n()])[0])
    assert [r[:3] for r in recs] == [(GEN, 0, HIGH_P), (GEN, 1, HIGH_P)]
    np.testing.assert_allclose([r[3] for r in recs], p_ref, atol=_mw_atol(grid, solver_atol))


@needs_bridge
def test_a_non_participant_is_never_reported():
    """a machine outside the distribution keeps its target exactly: a limit
    below it would be an input error, not something the solve produced"""
    grid = _feeder_grid(1., 1.)
    grid.remove_gen_slackbus(1)
    grid.set_gen_p_limits(np.full(2, np.nan), np.array([np.nan, 1.]))   # target 10 > 1
    grid.tell_solver_need_reset(); _solve(grid)
    sw = _one_row_is(grid)
    assert _p_records([sw.get_physical_violations_n()]) == [[]]
    assert _p_records(sw.get_physical_violations()) == [[]]


# ------------------------------------------------------------ storage units
@needs_bridge
def test_storage_reports_the_power_ac_pf_publishes(solver_atol):
    """the same check on a STORAGE unit, value and limit in the GENERATOR
    convention (the negated res_p_mw), element_type STORAGE"""
    from gpusim2grid.contingency_analysis import LimitViolationType, ViolationElementType
    grid = _feeder_storage_grid()
    if not hasattr(grid, "set_storage_p_limits"):
        pytest.skip("this lightsim2grid build has no storage active limits")
    _, p_sto = _ac_pf_res_p(grid)
    p_sto = p_sto[0]
    assert p_sto > 10.                       # discharging its 10 MW plus a share of the losses/load
    pmax = p_sto - 1.
    grid.set_storage_p_limits(np.array([np.nan]), np.array([pmax]))
    grid.tell_solver_need_reset(); _solve(grid)
    sw = _one_row_is(grid)
    for rows in (sw.get_physical_violations(), [sw.get_physical_violations_n()]):
        recs = _p_records(rows)[0]
        assert len(recs) == 1
        et, eid, vt, value, limit = recs[0]
        assert et == ViolationElementType.STORAGE and eid == 0
        assert vt == LimitViolationType.HIGH_P and limit == pmax
        np.testing.assert_allclose(value, p_sto, atol=_mw_atol(grid, solver_atol))
    # below min_p
    pmin = p_sto + 1.
    grid.set_storage_p_limits(np.array([pmin]), np.array([np.nan]))
    grid.tell_solver_need_reset(); _solve(grid)
    recs = _p_records([_one_row_is(grid).get_physical_violations_n()])[0]
    assert len(recs) == 1 and recs[0][:3] == (STO, 0, LOW_P)
    np.testing.assert_allclose(recs[0][3], p_sto, atol=_mw_atol(grid, solver_atol))


@needs_bridge
def test_storage_load_convention_is_not_used():
    """a range around the unit's output in the GENERATOR convention holds it:
    were the check reading the LOAD-convention res_p_mw, -p_sto would sit far
    outside [p_sto - 5, p_sto + 5] and be reported"""
    grid = _feeder_storage_grid()
    if not hasattr(grid, "set_storage_p_limits"):
        pytest.skip("this lightsim2grid build has no storage active limits")
    _, p_sto = _ac_pf_res_p(grid)
    grid.set_storage_p_limits(np.array([p_sto[0] - 5.]), np.array([p_sto[0] + 5.]))
    grid.tell_solver_need_reset(); _solve(grid)
    sw = _one_row_is(grid)
    assert _p_records([sw.get_physical_violations_n()]) == [[]]
    # ... and the two families share ONE slack: the generator's share is a
    # fraction of the raw participation of the whole grid, battery included
    grid.remove_storage_slackbus(0)
    grid.set_storage_p_limits(np.array([np.nan]), np.array([1.]))
    grid.tell_solver_need_reset(); _solve(grid)
    assert _p_records([_one_row_is(grid).get_physical_violations_n()]) == [[]]


@needs_bridge
def test_scenario_sweep_with_storage_matches_lightsim2grid(solver_atol):
    """Rows mixing a topology change, a generator contingency (the row's
    shares are the survivors' -- the battery included -- renormalised), a
    load change and a change of the generators' OWN set-points (per-row
    targets), on a persistent driver."""
    ScenarioSweepCPP = pytest.importorskip("lightsim2grid.scenarioSweep").ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    grid, v0 = _ieee14_storage_slack()
    gens = grid.get_generators()
    n_gen, n_line = len(gens), len(grid.get_lines())
    b2 = [g.name for g in gens].index("B2-G")
    load_p, load_q, gen_p = _elements(grid)
    n_rows = 4
    rep = lambda a, k=1.0: np.repeat((k * a)[None, :], n_rows, axis=0)   # noqa: E731
    lines_off = np.zeros((n_rows, n_line), dtype=bool); lines_off[1, 0] = True
    gens_off = np.zeros((n_rows, n_gen), dtype=bool); gens_off[2, b2] = True
    gp = rep(gen_p); gp[3] *= 0.97          # row 3: other set-points, same load
    lp = rep(load_p); lq = rep(load_q); lp[3] *= 0.98; lq[3] *= 0.98

    def ref_run(gp_, lp_, lq_):
        ls = ScenarioSweepCPP(grid)
        _ls_enable(ls, 0.)
        ls.set_contingency_lines(lines_off)
        ls.set_contingency_gens(gens_off)
        ls.modify_load_p(lp_); ls.modify_load_q(lq_); ls.modify_gen_p(gp_)
        ls.compute(v0.copy(), MAX_IT, TOL)
        assert all(ls.converged_mask())
        return _p_records(_ls_physical(ls)), _p_records([_ls_physical_n(ls)])

    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(lp, lq, gp)
    sw.set_topology([[], [0], [], []])
    sw.set_contingency_gens(gens_off)
    sw.compute(batch_size=n_rows)
    ref_rows, ref_n = ref_run(gp, lp, lq)
    got = _p_records(sw.get_physical_violations())
    atol = _mw_atol(grid, solver_atol)
    assert all(len(r) > 0 for r in ref_rows[1:]), "the fixture must move some share on every changed row"
    assert any(x[0] == STO for r in ref_rows for x in r), "the battery must be reported somewhere"
    _assert_same(ref_rows, got, atol)
    _assert_same(ref_n, _p_records([sw.get_physical_violations_n()]), atol)
    # the disconnected participant is never reported on its row
    assert all(x[:2] != (GEN, b2) for x in got[2])
    # hot re-run: other set-points (the per-row targets follow), fresh report
    gp2 = gp * 1.02
    sw.set_injections_from_elements(lp, lq, gp2)
    sw.compute(batch_size=n_rows)
    ref_rows, _ = ref_run(gp2, lp, lq)
    _assert_same(ref_rows, _p_records(sw.get_physical_violations()), atol)
    # warm re-run: drop the generator contingency
    sw.set_contingency_gens(np.zeros((n_rows, n_gen), dtype=bool))
    sw.compute(batch_size=n_rows)
    ls = ScenarioSweepCPP(grid)
    _ls_enable(ls, 0.)
    ls.set_contingency_lines(lines_off)
    ls.modify_load_p(lp); ls.modify_load_q(lq); ls.modify_gen_p(gp2)
    ls.compute(v0.copy(), MAX_IT, TOL)
    _assert_same(_p_records(_ls_physical(ls)), _p_records(sw.get_physical_violations()), atol)


# --------------------------------------------------------- injection sweep
@needs_bridge
def test_injection_sweep_targets_vary_per_row(solver_atol):
    """a row's generator produces ITS set-point plus its share: the check
    reads the row's own gen_p (per-row targets), like lightsim2grid"""
    InjectionSweepCPP = pytest.importorskip("lightsim2grid.lightsim2grid_cpp").InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid, v0 = _case14_shared_slack()
    load_p, load_q, gen_p = _elements(grid)
    ks = [1.0, 0.95, 1.05]
    lp = np.repeat(load_p[None, :], len(ks), axis=0)
    lq = np.repeat(load_q[None, :], len(ks), axis=0)
    gp = np.stack([k * gen_p for k in ks])

    ls = InjectionSweepCPP(grid)
    _ls_enable(ls, 0.)
    ls.modify_load_p(lp); ls.modify_load_q(lq); ls.modify_gen_p(gp)
    ls.compute(v0.copy(), MAX_IT, TOL)
    assert all(ls.converged_mask())
    ref_rows = _p_records(_ls_physical(ls))
    assert ref_rows[0] == [] and all(len(r) > 0 for r in ref_rows[1:])

    sw = InjectionSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(lp, lq, gp)
    sw.compute()
    atol = _mw_atol(grid, solver_atol)
    _assert_same(ref_rows, _p_records(sw.get_physical_violations()), atol)
    _assert_same(_p_records([_ls_physical_n(ls)]), _p_records([sw.get_physical_violations_n()]), atol)
    assert sw.solver.timings.t_physical_setup_ms > 0.
    # a per-bus set_injections carries no per-generator set-point: the
    # check falls back to the grid's own targets, so the same Sbus reported
    # with scaled targets is now reported against the BASE ones
    from gpusim2grid._ls2g_utils import build_bus_injections
    p_mw, q_mvar = build_bus_injections(sw._elements, lp, lq, gp)
    sw.set_injections(p_mw, q_mvar, sw._elements.sn_mva)
    sw.compute()
    got = _p_records(sw.get_physical_violations())
    assert got[0] == []
    for r in (1, 2):
        # target + share: the base target instead of the row's own, so the
        # value moves by exactly (1 - k) * target
        for (et, eid, vt, value, limit) in got[r]:
            ref = next(x for x in ref_rows[r] if x[:2] == (et, eid))
            np.testing.assert_allclose(value - ref[3], (1. - ks[r]) * gen_p[eid], atol=atol)


# --------------------------------------------------------------- array mode
@needs_bridge
def test_tuple_mode_matches_lightsim2grid(solver_atol):
    """Explicit-array mode: the plan is handed in as arrays (or the
    GenPPlanData object) and the report equals lightsim2grid's; left unset,
    the check reports nothing and raises nothing. One slack BUS holding two
    machines (weights 1 / 3): the tuple path has no distributed slack, see
    the fixture -- but the split of the bus' residual is per machine."""
    InjectionSweepCPP = pytest.importorskip("lightsim2grid.lightsim2grid_cpp").InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    from gpusim2grid import _gpusim2grid as _cpp
    from gpusim2grid._ls2g_utils import (
        extract_grid_arrays, extract_injection_elements, build_bus_injections)
    grid = _feeder_one_slack_bus()
    p_ref, _ = _ac_pf_res_p(grid)
    np.testing.assert_allclose(p_ref[1] - 10., 3. * p_ref[0], rtol=1e-6)
    grid.set_gen_p_limits(np.full(2, np.nan), p_ref - 1.)     # both out, on the base case already
    grid.tell_solver_need_reset()
    v0 = _solve(grid)
    load_p, load_q, gen_p = _elements(grid)
    ks = [1.0, 0.9, 0.8]   # the feeder is near its nose already
    lp = np.stack([k * load_p for k in ks]); lq = np.stack([k * load_q for k in ks])
    gp = np.repeat(gen_p[None, :], len(ks), axis=0)

    ls = InjectionSweepCPP(grid)
    _ls_enable(ls, 0.)
    ls.modify_load_p(lp); ls.modify_load_q(lq); ls.modify_gen_p(gp)
    ls.compute(v0.copy(), MAX_IT, TOL)
    assert all(ls.converged_mask())
    ref = _p_records(_ls_physical(ls))
    assert [x[:3] for x in ref[0]] == [(GEN, 0, HIGH_P), (GEN, 1, HIGH_P)]   # the base row at least
    ref_n = _p_records([_ls_physical_n(ls)])

    d = extract_grid_arrays(grid, max_iter=MAX_IT, tol=TOL)
    n_bus = d["n_bus"]
    p_mw, q_mvar = build_bus_injections(extract_injection_elements(grid, n_bus), lp, lq, gp)
    q_plan = _cpp._extract_bus_q_plan_from_lsgrid(grid, n_bus)
    plan = _cpp._extract_gen_p_plan_from_lsgrid(grid, n_bus)
    assert plan.n_entries == 2 and plan.n_part == 2   # every participant has a limit
    assert list(plan.bus_solver) == [0, 0] and list(plan.slack_weight) == [1., 3.]
    arrays = (plan.el_type, plan.el_id, plan.bus_solver, plan.slack_weight, plan.min_p_mw,
              plan.max_p_mw, plan.target_p_mw, plan.part_el_type, plan.part_el_id,
              plan.part_bus_solver, plan.part_weight, plan.sn_mva)
    atol = _mw_atol(grid, solver_atol)
    for handed in (None, plan, arrays):
        sw = InjectionSweepGPU(
            (d["Ybus"], d["v_converged"], d["Sbus"], d["slack"], d["slack_weights"], d["pv"], d["pq"]),
            nb_iter=10)
        with pytest.raises(RuntimeError):
            sw.set_gen_p_capability_from_grid()   # no grid in tuple mode
        sw.compute_physical_violations = True
        sw.physical_violation_tol_mva = 0.
        sw.set_bus_q_capability(q_plan)
        if handed is not None:
            sw.set_gen_p_capability(handed)
        sw.set_injections(p_mw, q_mvar, float(grid.get_sn_mva()))
        sw.compute()
        got = _p_records(sw.get_physical_violations())
        if handed is None:
            assert all(len(r) == 0 for r in got)   # no plan = nothing has a limit
        else:
            _assert_same(ref, got, atol)
            _assert_same(ref_n, _p_records([sw.get_physical_violations_n()]), atol)
    # a plan that does not fit the grid is refused
    bad = list(arrays); bad[2] = np.array([10**6, 0], dtype=np.int32)
    with pytest.raises(RuntimeError):
        sw.set_gen_p_capability(tuple(bad))
    bad = list(arrays); bad[0] = np.array([1, 5], dtype=np.int32)
    with pytest.raises(RuntimeError):
        sw.set_gen_p_capability(tuple(bad))
