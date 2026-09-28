# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``redistribute_slack`` on ContingencyAnalysisGPU / ScenarioSweepGPU
(lightsim2grid PR #216 parity, ``slack_redistribution.hpp``).

With it, the active power a row loses (an island cut off with
``handle_disconnected_grid``, a scenario-sweep generator contingency) is shared
on the remaining slack units BEFORE the solve, OpenLoadFlow-style: each unit
clamped to its ``[min_p, max_p]`` and never crossing 0 MW, a clamped unit
leaving the pool and that row's distributed slack. The batch must land where
lightsim2grid's own batch lands with the same option, and where its one-off
path does (a copy of the grid with the elements really removed,
``consider_only_main_component(True)`` / ``redistribute_active_power``,
``ac_pf``).

Grids: lightsim2grid's own test grid (case14, distributed slack on every
generator, the leaf bus 7 behind trafo 3 carrying a 40 MW generator), and the
pypowsybl IEEE 14 with a battery taking part in the slack. Voltages are
compared in magnitude and in angle relative to bus 0: when the first slack
unit saturates, lightsim2grid's one-off moves its angle reference, gpusim2grid
keeps its structural one (a constant angle shift).
"""

import os
import sys
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid C++ bridge"),
]

LEAF_BUS = 7
LEAF_P_MW = 40.
MAX_IT, TOL = 30, 1e-10


def _needs_upstream():
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    if not hasattr(ContingencyAnalysisCPP, "redistribute_slack"):
        pytest.skip("this lightsim2grid build has no redistribute_slack (PR #216)")


def _solve(grid):
    n = grid.get_bus_vn_kv().shape[0]
    V = grid.ac_pf(np.ones(n, dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0, "lightsim2grid diverged"
    return V


class Case14:
    """lightsim2grid's TestContingencyAnalysisRedistributeSlack grid, solved."""

    def __init__(self, limits=None):
        import pandapower.networks as pn
        from lightsim2grid.network import init_from_pandapower
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.grid = init_from_pandapower(pn.case14())
        gens = self.grid.get_generators()
        self.n_gen = len(gens)
        self.n_line = len(self.grid.get_lines())
        self.leaf_gen = [g.id for g in gens if g.bus_id == LEAF_BUS][0]
        branch = [ln.id for ln in self.grid.get_lines() if LEAF_BUS in (ln.bus1_id, ln.bus2_id)]
        branch += [self.n_line + t.id for t in self.grid.get_trafos()
                   if LEAF_BUS in (t.bus1_id, t.bus2_id)]
        assert len(branch) == 1
        self.leaf_branch = branch[0]
        for g in gens:
            self.grid.add_gen_slackbus(g.id, 1.)
        self.grid.change_p_gen(self.leaf_gen, LEAF_P_MW)
        self.targets = np.array([g.target_p_mw for g in self.grid.get_generators()])
        if limits is not None:
            self.grid.set_gen_p_limits(*limits(self))
        self.grid.tell_solver_need_reset()
        self.V0 = _solve(self.grid)
        self.s2me = np.asarray(self.grid.id_ac_solver_to_me(), dtype=int)
        self.n_bus = len(self.s2me)

    def two_clamped(self, extra=(3., 3.)):
        """two of the units that stay can only take `extra` MW more"""
        others = [g for g in range(self.n_gen) if g != self.leaf_gen]
        max_p = np.full(self.n_gen, np.inf)
        for g, e in zip(others[:2], extra):
            max_p[g] = self.targets[g] + e
        return np.full(self.n_gen, -np.inf), max_p

    def one_off_island(self):
        ref = self.grid.copy()
        if self.leaf_branch < self.n_line:
            ref.deactivate_powerline(self.leaf_branch)
        else:
            ref.deactivate_trafo(self.leaf_branch - self.n_line)
        report = ref.consider_only_main_component(True)
        return _solve(ref), report


def _in_solver(V_me, s2me):
    """a grid-numbered voltage vector (lightsim2grid's) in solver numbering"""
    return np.asarray(V_me)[s2me]


def _assert_same_state(V_gpu, V_ref, live, atol):
    np.testing.assert_allclose(np.abs(V_gpu[live]), np.abs(V_ref[live]), rtol=0., atol=atol)
    ang = lambda V: np.angle(V[live]) - np.angle(V[0])    # noqa: E731
    np.testing.assert_allclose(ang(V_gpu), ang(V_ref), rtol=0., atol=atol)


def _gpu_ca(case, branches, redistribute=True, **kw):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(case.grid, nb_iter=10, tol_base=1e-10,
                                handle_disconnected_grid=True,
                                redistribute_slack=redistribute, **kw)
    ca.add_contingencies_by_branch_id([[int(b)] for b in branches])
    ca.compute(batch_size=8)
    V = ca.V_results.to_numpy().reshape(len(branches), case.n_bus)
    return ca, V


def _ls_ca(case, branches, redistribute=True, physical=False):
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(case.grid)
    for b in branches:
        ca.add_n1(int(b))
    ca.handle_disconnected_grid = True
    ca.redistribute_slack = redistribute
    if physical:
        ca.compute_physical_violations = True
        ca.physical_violation_tol_mva = 0.
    ca.compute(1. * case.V0, MAX_IT, TOL)
    rows = {tuple(sorted(int(x) for x in c)): i for i, c in enumerate(ca.my_defaults())}
    V = np.asarray(ca.get_voltages())
    return ca, [V[rows[(int(b),)]] for b in branches], [rows[(int(b),)] for b in branches]


# ------------------------------------------------------------ the pure loop
def _olf_oracle(inj, w, lo, hi, mismatch, eps=1e-6):
    """A plain Python re-implementation of OLF's GenerationActivePowerDistributionStep
    (the same rules as lightsim2grid's test_main_component_slack oracle)."""
    inj = np.array(inj, dtype=float)
    n = inj.size
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, np.inf)
    lo = np.where(inj < 0., lo, np.maximum(lo, 0.))
    hi = np.where(inj < 0., np.minimum(hi, 0.), hi)
    active = np.ones(n, dtype=bool)
    sat = np.zeros(n, dtype=bool)
    remaining = mismatch
    rounds = 0
    while active.any() and abs(remaining) > eps and rounds <= n + 1:
        rounds += 1
        f = w[active].sum()
        if f <= 0.:
            break
        done = 0.
        for k in range(n):
            if not active[k]:
                continue
            old = inj[k]
            cand = old + remaining * w[k] / f
            if remaining > 0. and cand >= hi[k]:
                cand = old if old > hi[k] else hi[k]
                active[k] = False
                sat[k] = True
            elif remaining < 0. and cand <= lo[k]:
                cand = old if old < lo[k] else lo[k]
                active[k] = False
                sat[k] = True
            done += cand - old
            inj[k] = cand
        remaining -= done
    if not active.any():
        sat[:] = False
    return inj, sat, remaining


def test_distribute_matches_the_olf_loop():
    from gpusim2grid._gpusim2grid import _slack_distribute
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(200):
        n = int(rng.integers(1, 8))
        inj = rng.uniform(-50., 100., n)
        w = rng.uniform(0.1, 3., n)
        lo = np.where(rng.random(n) < 0.3, np.nan, inj - rng.uniform(0., 30., n))
        hi = np.where(rng.random(n) < 0.3, np.nan, inj + rng.uniform(0., 30., n))
        cases.append((inj, w, lo, hi, float(rng.uniform(-150., 150.))))
    # the edges: everything saturated, already beyond a bound, nothing to share
    cases.append((np.array([10., 20.]), np.array([1., 1.]), np.array([0., 0.]),
                  np.array([11., 21.]), 50.))
    cases.append((np.array([10., 20.]), np.array([1., 1.]), np.full(2, np.nan),
                  np.array([5., 30.]), 4.))
    cases.append((np.array([10., 20.]), np.array([1., 1.]), np.full(2, np.nan),
                  np.full(2, np.nan), 0.))
    for inj, w, lo, hi, mis in cases:
        p, sat, *rep = _slack_distribute(inj, w, lo, hi, mis)
        p_ref, sat_ref, remaining = _olf_oracle(inj, w, lo, hi, mis)
        np.testing.assert_allclose(p, p_ref, rtol=0., atol=1e-9)
        assert list(np.asarray(sat, dtype=bool)) == list(sat_ref)
        assert rep[4] == pytest.approx(remaining, abs=1e-9)   # not_distributed_mw


# ---------------------------------------------------------- contingency analysis
@pytest.mark.parametrize("limits", [None, "two_clamped", "reference"])
def test_ca_island_matches_lightsim2grid(solver_atol, limits):
    """The island of the leaf bus: gpusim2grid's row equals lightsim2grid's
    batch with the same option, and its one-off path."""
    _needs_upstream()

    def lim(case):
        if limits == "two_clamped":
            return case.two_clamped()
        max_p = np.full(case.n_gen, np.inf)
        max_p[0] = case.targets[0] + 1.   # the first slack unit saturates
        return np.full(case.n_gen, -np.inf), max_p

    case = Case14(None if limits is None else lim)
    ca, V = _gpu_ca(case, [case.leaf_branch])
    _, V_ls, _ = _ls_ca(case, [case.leaf_branch])
    V_one, report = case.one_off_island()
    live = np.arange(case.n_bus) != LEAF_BUS
    atol = 10 * solver_atol
    assert np.isnan(V[0][LEAF_BUS])
    _assert_same_state(V[0], _in_solver(V_ls[0], case.s2me), live, atol)
    _assert_same_state(V[0], _in_solver(V_one, case.s2me), live, atol)
    rep = ca.get_slack_redistribution_report()
    assert rep["mismatch_mw"][0] == pytest.approx(report.mismatch_mw, abs=1e-9)
    assert rep["mismatch_mw"][0] == pytest.approx(LEAF_P_MW, abs=1e-9)
    assert rep["nb_saturated"][0] == report.nb_saturated
    assert rep["nb_saturated"][0] == {None: 0, "two_clamped": 2, "reference": 1}[limits]


def test_ca_limits_bite_and_nothing_lost_is_unchanged(solver_atol):
    """With the limits biting the option changes the islanding row; a row that
    islands nothing, or a batch where no row loses anything, is unchanged --
    compared to rounding: two separate sessions are only reproducible to
    ~1e-15 (batched GPU reductions), whatever the option."""
    _needs_upstream()
    case = Case14(lambda c: c.two_clamped())
    branches = [case.leaf_branch, 0]
    _, V_on = _gpu_ca(case, branches, redistribute=True)
    _, V_off = _gpu_ca(case, branches, redistribute=False)
    live = np.arange(case.n_bus) != LEAF_BUS
    assert np.max(np.abs(V_on[0][live] - V_off[0][live])) > 1e-6
    np.testing.assert_allclose(V_on[1], V_off[1], rtol=0., atol=solver_atol)
    _, V_nothing = _gpu_ca(case, [0], redistribute=True)
    _, V_nothing_off = _gpu_ca(case, [0], redistribute=False)
    np.testing.assert_allclose(V_nothing, V_nothing_off, rtol=0., atol=solver_atol)


def test_ca_saturated_units_are_not_reported(solver_atol):
    """The saturated units sit at max_p: the slack active-power check does not
    report them above it, and every row reports what lightsim2grid's batch
    reports."""
    _needs_upstream()
    from gpusim2grid.contingency_analysis import LimitViolationType as LVT
    case = Case14(lambda c: c.two_clamped())
    branches = [case.leaf_branch, 0]
    ca, _ = _gpu_ca(case, branches, compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.compute(batch_size=8)
    ls, _, rows = _ls_ca(case, branches, physical=True)
    got = ca.get_physical_violations()
    ref = ls.get_physical_violations()
    p_types = (int(LVT.HIGH_P), int(LVT.LOW_P))
    key = lambda v: (int(v.element_type), int(v.element_id), int(v.violation_type))   # noqa: E731
    for i, r in enumerate(rows):
        g = sorted(key(v) for v in got[i] if int(v.violation_type) in p_types)
        e = sorted(key(v) for v in ref[r] if int(v.violation_type) in p_types)
        assert g == e
    assert [v for v in got[0] if int(v.violation_type) in p_types] == []


def test_ca_hvdc_island(solver_atol):
    """The island holds a (non-droop) HVDC converter station that rectifies:
    its set-point is part of what the row loses (a consumption: the share is
    negative), and the row equals lightsim2grid's batch."""
    _needs_upstream()
    import lightsim2grid
    tests_dir = os.path.join(os.path.dirname(lightsim2grid.__file__), "tests")
    if not os.path.isfile(os.path.join(tests_dir, "_aux_make_hvdc.py")):
        pytest.skip("lightsim2grid's test helpers are not installed")
    sys.path.insert(0, tests_dir)
    try:
        from _aux_make_hvdc import make_case14_hvdc
    finally:
        sys.path.remove(tests_dir)
    psp = 30.
    _, grid = make_case14_hvdc(3, LEAF_BUS, converters_mode=1, p_setpoint=psp)
    case = Case14.__new__(Case14)
    case.grid = grid
    gens = grid.get_generators()
    case.n_gen = len(gens)
    case.n_line = len(grid.get_lines())
    branch = [ln.id for ln in grid.get_lines() if LEAF_BUS in (ln.bus1_id, ln.bus2_id)]
    branch += [case.n_line + t.id for t in grid.get_trafos() if LEAF_BUS in (t.bus1_id, t.bus2_id)]
    case.leaf_branch = branch[0]
    for g in gens:
        grid.add_gen_slackbus(g.id, 1.)
    targets = np.array([g.target_p_mw for g in gens])
    min_p = np.full(case.n_gen, -np.inf)
    producing = [g.id for g in gens if g.target_p_mw > 0. and g.bus_id != LEAF_BUS]
    min_p[producing[0]] = targets[producing[0]] - 2.
    grid.set_gen_p_limits(min_p, np.full(case.n_gen, np.inf))
    grid.tell_solver_need_reset()
    case.V0 = _solve(grid)
    case.s2me = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    case.n_bus = len(case.s2me)

    ca, V = _gpu_ca(case, [case.leaf_branch])
    _, V_ls, _ = _ls_ca(case, [case.leaf_branch])
    live = np.arange(case.n_bus) != LEAF_BUS
    _assert_same_state(V[0], _in_solver(V_ls[0], case.s2me), live, 10 * solver_atol)
    rep = ca.get_slack_redistribution_report()
    assert rep["mismatch_mw"][0] == pytest.approx(-psp, abs=1e-6)
    assert rep["nb_saturated"][0] >= 1


def test_refusals():
    _needs_upstream()
    case = Case14(lambda c: c.two_clamped())
    with pytest.raises(RuntimeError, match="distributed slack"):
        _gpu_ca(case, [case.leaf_branch], use_distributed_slack=False)
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(case.grid, nb_iter=10, redistribute_slack=True)
    ca.strategy = "direct_base_case_factors"
    ca.add_contingencies_by_branch_id([[0]])
    with pytest.raises(RuntimeError, match="direct_base_case_factors"):
        ca.compute(batch_size=8)


# ---------------------------------------------------------------- scenario sweep
class SweepCase(Case14):
    """lightsim2grid's TestScenarioSweepRedistributeSlack grid: a non-slack
    generator on bus 2 producing 20 MW, the others all in the slack."""

    def __init__(self, limits=None):
        def setup(case):
            gens = case.grid.get_generators()
            case.non_slack = [g.id for g in gens if g.bus_id == 2][0]
            case.grid.remove_gen_slackbus(case.non_slack)
            case.grid.change_p_gen(case.non_slack, 20.)
            case.targets = np.array([g.target_p_mw for g in case.grid.get_generators()])
            case.slack_gen = [g.id for g in gens if g.bus_id == 1][0]
            lo = np.full(case.n_gen, -np.inf)
            hi = np.full(case.n_gen, np.inf)
            if limits is not None:
                lo, hi = limits(case)
            return lo, hi
        super().__init__(setup)

    def elements(self, n_rows, gen_scale=None):
        load_p, load_q = (np.asarray(a, dtype=np.float64) for a in self.grid.get_loads_res_full()[:2])
        gen_p = np.asarray(self.grid.get_gen_target_p(), dtype=np.float64)
        rep = lambda a: np.repeat(a[None, :], n_rows, axis=0)   # noqa: E731
        gp = rep(gen_p)
        if gen_scale is not None:
            gp = gp * np.asarray(gen_scale, dtype=np.float64)[:, None]
        return rep(load_p), rep(load_q), gp


def _gpu_ss(case, gen_mask, gen_p_rows=None, topology=None, redistribute=True, **kw):
    from gpusim2grid import ScenarioSweepGPU
    n = gen_mask.shape[0]
    load_p, load_q, gen_p = case.elements(n)
    if gen_p_rows is not None:
        gen_p = gen_p_rows
    sw = ScenarioSweepGPU(case.grid, nb_iter=10, tol_base=1e-10,
                          redistribute_slack=redistribute, **kw)
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.set_contingency_gens(gen_mask)
    if topology is not None:
        sw.set_topology(topology)
    sw.compute(batch_size=n)
    return sw, sw.solver.V_results.to_numpy().reshape(n, case.n_bus)


def _ls_ss(case, gen_mask, gen_p_rows=None, branches_off=None, redistribute=True):
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    n = gen_mask.shape[0]
    load_p, load_q, gen_p = case.elements(n)
    sw = ScenarioSweepCPP(case.grid)
    sw.modify_load_p(load_p)
    sw.modify_load_q(load_q)
    sw.modify_gen_p(gen_p if gen_p_rows is None else gen_p_rows)
    sw.set_contingency_gens(gen_mask)
    if branches_off is not None:
        # lightsim2grid's masks are per family (lines, then trafos)
        sw.set_contingency_lines(branches_off[:, :case.n_line])
        sw.set_contingency_trafos(branches_off[:, case.n_line:])
        sw.handle_disconnected_grid = True
    sw.redistribute_slack = redistribute
    sw.compute(1. * case.V0, MAX_IT, TOL)
    assert all(sw.converged_mask())
    return np.asarray(sw.get_voltages())


def _one_off_gen_off(case, gen_off, row_p=None):
    ref = case.grid.copy()
    if row_p is not None:
        for g in range(case.n_gen):
            ref.change_p_gen(g, float(row_p[g]))
    lost = float(row_p[gen_off]) if row_p is not None else float(case.targets[gen_off])
    ref.deactivate_gen(gen_off)
    report = ref.redistribute_active_power(lost)
    return _solve(ref), report


@pytest.mark.parametrize("which", ["slack_gen", "non_slack"])
def test_ss_gen_off_matches_lightsim2grid(solver_atol, which):
    _needs_upstream()

    def lim(case):
        off = case.slack_gen if which == "slack_gen" else case.non_slack
        others = [g for g in range(case.n_gen) if g not in (off, case.non_slack)]
        hi = np.full(case.n_gen, np.inf)
        hi[others[0]] = case.targets[others[0]] + (2. if which == "slack_gen" else 1.)
        return np.full(case.n_gen, -np.inf), hi

    case = SweepCase(lim)
    off = case.slack_gen if which == "slack_gen" else case.non_slack
    mask = np.zeros((1, case.n_gen), dtype=bool)
    mask[0, off] = True
    sw, V = _gpu_ss(case, mask)
    V_ls = _ls_ss(case, mask)
    V_one, report = _one_off_gen_off(case, off)
    every = np.ones(case.n_bus, dtype=bool)
    atol = 10 * solver_atol
    _assert_same_state(V[0], _in_solver(V_ls[0], case.s2me), every, atol)
    _assert_same_state(V[0], _in_solver(V_one, case.s2me), every, atol)
    rep = sw.get_slack_redistribution_report()
    assert rep["nb_saturated"][0] == report.nb_saturated == 1
    assert rep["mismatch_mw"][0] == pytest.approx(report.mismatch_mw, abs=1e-9)


def test_ss_row_setpoints_are_the_rows_own(solver_atol):
    """The lost power, and the units' starting points, are the ROW's gen_p."""
    _needs_upstream()

    def lim(case):
        others = [g for g in range(case.n_gen) if g not in (case.slack_gen, case.non_slack)]
        hi = np.full(case.n_gen, np.inf)
        hi[others[0]] = case.targets[others[0]] + 2.
        return np.full(case.n_gen, -np.inf), hi

    case = SweepCase(lim)
    gen_p = np.vstack((case.targets, 1.1 * case.targets))
    mask = np.zeros((2, case.n_gen), dtype=bool)
    mask[:, case.slack_gen] = True
    _, V = _gpu_ss(case, mask, gen_p_rows=gen_p)
    V_ls = _ls_ss(case, mask, gen_p_rows=gen_p)
    every = np.ones(case.n_bus, dtype=bool)
    for r in range(2):
        _assert_same_state(V[r], _in_solver(V_ls[r], case.s2me), every, 10 * solver_atol)
        V_one, _ = _one_off_gen_off(case, case.slack_gen, row_p=gen_p[r])
        _assert_same_state(V[r], _in_solver(V_one, case.s2me), every, 10 * solver_atol)


def test_ss_island_and_gen_off_in_one_row(solver_atol):
    """A row islanding the leaf bus AND disconnecting a slack generator loses
    both; the rows equal lightsim2grid's batch."""
    _needs_upstream()

    def lim(case):
        others = [g for g in range(case.n_gen)
                  if g not in (case.slack_gen, case.non_slack, case.leaf_gen)]
        hi = np.full(case.n_gen, np.inf)
        hi[others[0]] = case.targets[others[0]] + 3.
        return np.full(case.n_gen, -np.inf), hi

    case = SweepCase(lim)
    mask = np.zeros((3, case.n_gen), dtype=bool)
    mask[1, case.slack_gen] = True
    mask[2, case.slack_gen] = True
    n_branch = case.n_line + len(case.grid.get_trafos())
    branches_off = np.zeros((3, n_branch), dtype=bool)
    branches_off[0, case.leaf_branch] = True
    branches_off[2, case.leaf_branch] = True
    topology = [[case.leaf_branch], [], [case.leaf_branch]]
    sw, V = _gpu_ss(case, mask, topology=topology, handle_disconnected_grid=True)
    V_ls = _ls_ss(case, mask, branches_off=branches_off)
    live = np.arange(case.n_bus) != LEAF_BUS
    for r in range(3):
        on = live if r != 1 else np.ones(case.n_bus, dtype=bool)
        _assert_same_state(V[r], _in_solver(V_ls[r], case.s2me), on, 10 * solver_atol)
    rep = sw.get_slack_redistribution_report()
    assert rep["mismatch_mw"][2] == pytest.approx(LEAF_P_MW + case.targets[case.slack_gen], abs=1e-9)


def test_ss_hot_runs_and_flag_toggle(solver_atol):
    """New injections on a live driver stay hot (no source rebuild) and are
    redistributed afresh; toggling the flag needs no rebuild either, and off
    gives back exactly the plain run."""
    _needs_upstream()

    def lim(case):
        others = [g for g in range(case.n_gen) if g not in (case.slack_gen, case.non_slack)]
        hi = np.full(case.n_gen, np.inf)
        hi[others[0]] = case.targets[others[0]] + 2.
        return np.full(case.n_gen, -np.inf), hi

    case = SweepCase(lim)
    mask = np.zeros((2, case.n_gen), dtype=bool)
    mask[:, case.slack_gen] = True
    sw, V_a = _gpu_ss(case, mask)
    n_src, n_drv = sw.source_build_counter, sw.driver_build_counter
    load_p, load_q, gen_p = case.elements(2)
    gen_p2 = np.vstack((case.targets, 1.1 * case.targets))
    sw.set_injections_from_elements(load_p, load_q, gen_p2)
    sw.compute(batch_size=2)
    V_b = sw.solver.V_results.to_numpy().reshape(2, case.n_bus)
    assert (sw.source_build_counter, sw.driver_build_counter) == (n_src, n_drv)
    V_ls = _ls_ss(case, mask, gen_p_rows=gen_p2)
    every = np.ones(case.n_bus, dtype=bool)
    for r in range(2):
        _assert_same_state(V_b[r], _in_solver(V_ls[r], case.s2me), every, 10 * solver_atol)

    sw.redistribute_slack = False
    sw.compute(batch_size=2)
    V_off = sw.solver.V_results.to_numpy().reshape(2, case.n_bus).copy()
    assert (sw.source_build_counter, sw.driver_build_counter) == (n_src, n_drv)
    _, V_plain = _gpu_ss(case, mask, gen_p_rows=gen_p2, redistribute=False)
    np.testing.assert_allclose(V_off, V_plain, rtol=0., atol=10 * solver_atol)
    assert np.max(np.abs(V_off - V_b)) > 1e-6
    sw.redistribute_slack = True
    sw.compute(batch_size=2)
    np.testing.assert_allclose(sw.solver.V_results.to_numpy().reshape(2, case.n_bus), V_b,
                               rtol=0., atol=10 * solver_atol)


def test_ss_per_bus_injections_fall_back_to_grid_targets(solver_atol):
    """set_injections (per bus) carries no generator set-point: the pre-pass
    reads the grid's own -- the same as set_injections_from_elements with the
    grid's targets."""
    _needs_upstream()
    from gpusim2grid import ScenarioSweepGPU
    from gpusim2grid._ls2g_utils import build_bus_injections, extract_injection_elements

    def lim(case):
        others = [g for g in range(case.n_gen) if g not in (case.slack_gen, case.non_slack)]
        hi = np.full(case.n_gen, np.inf)
        hi[others[0]] = case.targets[others[0]] + 2.
        return np.full(case.n_gen, -np.inf), hi

    case = SweepCase(lim)
    mask = np.zeros((1, case.n_gen), dtype=bool)
    mask[0, case.slack_gen] = True
    _, V_ref = _gpu_ss(case, mask)
    el = extract_injection_elements(case.grid, case.n_bus)
    load_p, load_q, gen_p = case.elements(1)
    p_mw, q_mvar = build_bus_injections(el, load_p, load_q, gen_p, gen_off=mask)
    sw = ScenarioSweepGPU(case.grid, nb_iter=10, tol_base=1e-10, redistribute_slack=True)
    sw.set_injections(p_mw, q_mvar, el.sn_mva)
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=1)
    np.testing.assert_allclose(sw.solver.V_results.to_numpy().reshape(1, case.n_bus), V_ref,
                               rtol=0., atol=10 * solver_atol)


def _ieee14_battery(sto_margin_mw=1.):
    """pypowsybl IEEE 14 with a battery on bus B3 sharing the slack with B1-G
    and B2-G; the battery can only move `sto_margin_mw` around its set-point."""
    pypo = pytest.importorskip("pypowsybl")
    from lightsim2grid.network import init_from_pypowsybl
    net = pypo.network.create_ieee14()
    net.create_batteries(id="BAT", voltage_level_id="VL3", bus_id="B3", target_p=10.,
                         target_q=0., min_p=-300., max_p=300.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id={"B1-G": 1., "B2-G": 1.},
                                   sort_index=False, buses_for_sub=False)
    if not hasattr(grid, "add_storage_slackbus") or not hasattr(grid, "set_storage_p_limits"):
        pytest.skip("this lightsim2grid build has no storage slack / storage active limits")
    grid.add_storage_slackbus(0, 1.)
    sto_p = -float(grid.get_storages()[0].target_p_mw)   # generator convention
    grid.set_storage_p_limits(np.array([sto_p - sto_margin_mw]), np.array([sto_p + sto_margin_mw]))
    grid.tell_solver_need_reset()
    return grid


def test_ss_storage_unit_saturates(solver_atol):
    """A battery of the slack saturates first and leaves the row's slack; the
    generator left takes the rest -- as lightsim2grid's batch."""
    _needs_upstream()
    grid = _ieee14_battery()
    V0 = _solve(grid)
    case = Case14.__new__(Case14)
    case.grid, case.V0 = grid, V0
    case.n_gen = len(grid.get_generators())
    case.s2me = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    case.n_bus = len(case.s2me)
    case.elements = lambda n, gen_scale=None: SweepCase.elements(case, n, gen_scale)
    names = [g.name for g in grid.get_generators()]
    off = names.index("B2-G") if "B2-G" in names else 1
    mask = np.zeros((1, case.n_gen), dtype=bool)
    mask[0, off] = True
    sw, V = _gpu_ss(case, mask)
    V_ls = _ls_ss(case, mask)
    every = np.ones(case.n_bus, dtype=bool)
    _assert_same_state(V[0], _in_solver(V_ls[0], case.s2me), every, 10 * solver_atol)
    rep = sw.get_slack_redistribution_report()
    assert rep["nb_saturated"][0] >= 1 and not rep["all_saturated"][0]


def test_ss_refuses_device_injections():
    _needs_upstream()
    torch = pytest.importorskip("torch")
    from gpusim2grid import ScenarioSweepGPU
    case = SweepCase()
    sw = ScenarioSweepGPU(case.grid, nb_iter=10, redistribute_slack=True)
    load_p, load_q, gen_p = case.elements(1)
    from gpusim2grid._ls2g_utils import build_bus_injections, extract_injection_elements
    el = extract_injection_elements(case.grid, case.n_bus)
    p_mw, q_mvar = build_bus_injections(el, load_p, load_q, gen_p)
    S = torch.tensor((p_mw + 1j * q_mvar) / el.sn_mva, dtype=torch.complex128, device="cuda")
    sw._inner._s.set_injections_dlpack(S.__dlpack__(), torch.cuda.current_stream().cuda_stream)
    with pytest.raises(RuntimeError, match="host"):
        sw.compute(batch_size=1)


# ---------------------------------------------------------------- BatchPowerFlow
def _bpf_case():
    """SweepCase with one unit that can only take 2 MW more: rows 0 / 3 island
    the leaf bus, rows 1 / 3 disconnect the slack generator of bus 1, row 2
    loses nothing."""
    def lim(case):
        # the clamped unit and the ones perturbed below all produce: a unit at
        # exactly 0 MW sits on OLF's "never cross 0 MW" kink
        others = [g for g in range(case.n_gen)
                  if g not in (case.slack_gen, case.non_slack, case.leaf_gen)]
        case.clamped_gen = max(others, key=lambda g: case.targets[g])
        assert case.targets[case.clamped_gen] > 0.
        hi = np.full(case.n_gen, np.inf)
        hi[case.clamped_gen] = case.targets[case.clamped_gen] + 2.
        return np.full(case.n_gen, -np.inf), hi
    case = SweepCase(lim)
    n = 4
    trafo_status = np.ones((n, len(case.grid.get_trafos())), dtype=bool)
    leaf_trafo = case.leaf_branch - case.n_line
    assert leaf_trafo >= 0
    trafo_status[[0, 3], leaf_trafo] = False
    gen_status = np.ones((n, case.n_gen), dtype=bool)
    gen_status[[1, 3], case.slack_gen] = False
    return case, trafo_status, gen_status


def _bpf(case, **kw):
    from gpusim2grid.differentiable import BatchPowerFlow
    return BatchPowerFlow.from_lsgrid(case.grid, nb_iter=20, tol_base=1e-11,
                                      handle_disconnected_grid=True, **kw)


def test_batch_power_flow_forward_matches_the_sweep(solver_atol):
    """BatchPowerFlow runs the pre-pass itself (torch + host loop) and hands the
    saturated units in: its rows equal ScenarioSweepGPU's own pre-pass."""
    _needs_upstream()
    torch = pytest.importorskip("torch")
    case, trafo_status, gen_status = _bpf_case()
    pf = _bpf(case, redistribute_slack=True)
    load_p, load_q, gen_p = case.elements(4)
    V = pf(load_p=torch.as_tensor(load_p, device="cuda"), load_q=torch.as_tensor(load_q, device="cuda"),
           gen_p=torch.as_tensor(gen_p, device="cuda"),
           trafo_status=torch.as_tensor(trafo_status, device="cuda"),
           gen_status=torch.as_tensor(gen_status, device="cuda")).detach().cpu().numpy()
    topology = [[case.leaf_branch] if not trafo_status[r].all() else [] for r in range(4)]
    # BatchPowerFlow keeps the grid's reference (see its from_lsgrid)
    _, V_sw = _gpu_ss(case, ~gen_status, topology=topology, handle_disconnected_grid=True,
                      reference_slack="grid")
    np.testing.assert_allclose(np.nan_to_num(V), np.nan_to_num(V_sw), rtol=0., atol=10 * solver_atol)
    # ... and differs from the plain distributed slack (the limit bites)
    pf_off = _bpf(case)
    V_off = pf_off(load_p=torch.as_tensor(load_p, device="cuda"),
                   load_q=torch.as_tensor(load_q, device="cuda"),
                   gen_p=torch.as_tensor(gen_p, device="cuda"),
                   trafo_status=torch.as_tensor(trafo_status, device="cuda"),
                   gen_status=torch.as_tensor(gen_status, device="cuda")).detach().cpu().numpy()
    assert np.nanmax(np.abs(V_off[1] - V[1])) > 1e-6


@pytest.mark.skipif(bool(__import__("gpusim2grid._gpusim2grid", fromlist=["x"]).is_fp32),
                    reason="finite-difference gradient checks need the FP64 build")
def test_batch_power_flow_gradient_is_the_finite_difference():
    """Every differentiable input, against central differences -- the entries
    the pre-pass touches included. With the clamped set fixed, the free units
    absorb what was lost by their weights, which the Newton slack (same free
    units, same weights) would do anyway: the island's generator and the
    disconnected one have a ZERO derivative, and so does a clamped unit's own
    set-point (its injection sits at the bound) -- which a pre-pass treated as a
    constant correction would get wrong (without the option that set-point
    moves the state like any participant's). A free unit, a load, a reactive
    load and a voltage set-point keep a live derivative."""
    _needs_upstream()
    torch = pytest.importorskip("torch")
    case, trafo_status, gen_status = _bpf_case()
    pf = _bpf(case, redistribute_slack=True)
    load_p, load_q, gen_p = (torch.as_tensor(a, device="cuda") for a in case.elements(4))
    gen_v = torch.full((4, case.n_gen), float("nan"), dtype=torch.float64, device="cuda")
    pv = [g for g, e in enumerate(case.grid.get_generators())
          if e.voltage_regulator_on and g not in (case.slack_gen, case.leaf_gen)][0]
    gen_v[:, pv] = float(case.grid.get_generators()[pv].target_vm_pu)
    ts = torch.as_tensor(trafo_status, device="cuda")
    gs = torch.as_tensor(gen_status, device="cuda")
    gen_buses = {g.bus_id for g in case.grid.get_generators()}
    loads = [ld.id for ld in case.grid.get_loads() if ld.bus_id not in gen_buses]   # on PQ buses

    def loss(lp, lq, gp, gv):
        V = pf(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv, trafo_status=ts, gen_status=gs)
        n = pf.n_bus
        wr = torch.linspace(0.5, 1.5, n, dtype=torch.float64, device="cuda")
        wi = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device="cuda")
        ok = torch.isfinite(V.real) & torch.isfinite(V.imag)
        Vs = torch.where(ok, V, torch.zeros_like(V))
        return (Vs.real * wr).sum() + (Vs.imag * wi).sum()

    inputs = [load_p.clone().requires_grad_(True), load_q.clone().requires_grad_(True),
              gen_p.clone().requires_grad_(True), gen_v.clone().requires_grad_(True)]
    loss(*inputs).backward()
    grads = [x.grad for x in inputs]

    checks = [  # (input index, row, column, step, expected: "zero" / "live")
        (2, 0, case.leaf_gen, 1e-3, "zero"),     # the island's generator
        (2, 1, case.slack_gen, 1e-3, "zero"),    # the disconnected generator
        (2, 3, case.leaf_gen, 1e-3, "zero"),     # island + disconnected generator
        (2, 0, case.clamped_gen, 1e-3, "zero"),  # clamped: its injection sits at the bound
        (2, 1, case.clamped_gen, 1e-3, "zero"),
        (2, 0, case.slack_gen, 1e-3, "live"),    # free units (40 MW each)
        (2, 1, case.leaf_gen, 1e-3, "live"),
        (2, 2, case.slack_gen, 1e-3, "live"),    # a row that loses nothing
        (2, 2, case.clamped_gen, 1e-3, "live"),  # ... where nothing is clamped
        (0, 0, loads[0], 1e-3, "live"),
        (0, 1, loads[1], 1e-3, "live"),
        (1, 1, loads[1], 1e-3, "live"),
        (3, 1, pv, 1e-6, "live"),
    ]

    def fd(model_loss, i, r, c, h):
        vals = []
        for sign in (1.0, -1.0):
            args = [x.detach().clone() for x in inputs]
            args[i][r, c] += sign * h
            vals.append(float(model_loss(*args)))
        return (vals[0] - vals[1]) / (2 * h)

    with torch.no_grad():
        for i, r, c, h, expected in checks:
            d = fd(loss, i, r, c, h)
            an = float(grads[i][r, c])
            if expected == "live":
                assert abs(d) > 1e-6, (i, r, c, d)
            else:
                assert abs(d) < 1e-7, (i, r, c, d)
            assert an == pytest.approx(d, rel=1e-4, abs=1e-7), (i, r, c, an, d)

    # without the option, the clamped unit's set-point does move the state:
    # the zero above is the pre-pass' doing
    pf_off = _bpf(case)

    def loss_off(lp, lq, gp, gv):
        V = pf_off(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv, trafo_status=ts, gen_status=gs)
        ok = torch.isfinite(V.real) & torch.isfinite(V.imag)
        Vs = torch.where(ok, V, torch.zeros_like(V))
        n = pf_off.n_bus
        return ((Vs.real * torch.linspace(0.5, 1.5, n, dtype=torch.float64, device="cuda")).sum()
                + (Vs.imag * torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device="cuda")).sum())

    with torch.no_grad():
        assert abs(fd(loss_off, 2, 1, case.clamped_gen, 1e-3)) > 1e-6
