# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""compute_physical_violations, reactive part -- the per-bus reactive-capability
check (lightsim2grid PR #206 parity) on the three batch facades.

The reference is lightsim2grid itself: ContingencyAnalysisCPP /
ScenarioSweepCPP / InjectionSweepCPP with ``compute_physical_violations =
True`` on the SAME grid, whose per-row reports (bus, LOW_Q/HIGH_Q, q produced,
summed capability) must be reproduced record for record. None of these grids
has a droop hvdc line, so the physical report IS the reactive one here (the
hvdc part is pinned in test_hvdc_p_violations.py). lightsim2grid reports the
grid-model bus id, gpusim2grid the solver one (like its voltage records), so
the reference ids go through ``id_me_to_ac_solver``.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="the reactive-capability plan is built by lightsim2grid through the C++ bridge")

MAX_IT, TOL = 20, 1e-11
LOW_Q, HIGH_Q = 5, 6     # LimitViolationType


# ------------------------------------------------------------------ grids
def _solve(grid):
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    grid.change_algorithm(AlgorithmType.NR_KLU)
    v0 = np.full(grid.total_bus(), grid.get_init_vm_pu() + 0j)
    V = grid.ac_pf(v0.copy(), MAX_IT, TOL)
    assert V.shape[0] > 0, "lightsim2grid diverged"
    return v0


def _tight_net(max_q_mvar):
    """pandapower case14 with every generator's reactive range narrowed to
    +/- max_q_mvar (upstream's fixture): the buses they hold then ask for far
    more reactive power than they own. Limits influence nothing but the
    report."""
    pn = pytest.importorskip("pandapower.networks")
    net = pn.case14()
    net.gen["min_q_mvar"] = -max_q_mvar
    net.gen["max_q_mvar"] = max_q_mvar
    net.ext_grid["min_q_mvar"] = -max_q_mvar
    net.ext_grid["max_q_mvar"] = max_q_mvar
    return net


def _case14_tight_q(max_q_mvar=5.):
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(_tight_net(max_q_mvar))
    v0 = _solve(grid)
    return grid, v0


def _case14_default():
    pn = pytest.importorskip("pandapower.networks")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(pn.case14())
    v0 = _solve(grid)
    return grid, v0


def _tight_spur_grid(max_q_mvar=5.):
    """Tight case14 plus a radial spur bus (one line, one load): tripping the
    spur line islands exactly that bus. Returns (grid, v0, spur_line_id)."""
    pp = pytest.importorskip("pandapower")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        net = _tight_net(max_q_mvar)
        spur_bus = pp.create_bus(net, vn_kv=float(net.bus.vn_kv.iloc[0]))
        pp.create_line_from_parameters(
            net, from_bus=0, to_bus=spur_bus, length_km=1.0,
            r_ohm_per_km=0.05, x_ohm_per_km=0.2, c_nf_per_km=0.0, max_i_ka=1.0)
        pp.create_load(net, bus=spur_bus, p_mw=5.0, q_mvar=2.0)
        spur_line_id = len(net.line) - 1
        grid = init_from_pandapower(net)
    v0 = _solve(grid)
    return grid, v0, spur_line_id


def _tight_voltage_control_grid(max_q_mvar=5.):
    """Tight case14 with a bordered VoltageControl group: a second generator
    co-located with gen 3 on bus 7, both regulating remote bus 9 -- the
    reactive output of those two is a solved unknown (Q_c), not part of the
    residual, which is exactly what the raw-residual shortcut must handle."""
    pp = pytest.importorskip("pandapower")
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        net = _tight_net(max_q_mvar)
        pp.create_gen(net, bus=7, p_mw=0.0, vm_pu=1.09, controllable=True,
                      min_q_mvar=-max_q_mvar, max_q_mvar=max_q_mvar)
        grid = init_from_pandapower(net)
        if not hasattr(grid, "set_gen_regulated_bus"):
            pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
        grid.set_gen_regulated_bus(3, 9)
        grid.set_gen_regulated_bus(4, 9)
        grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0


def _tight_svc_grid(max_q_mvar=5., b_max=0.05):
    """Tight case14 with a voltage-mode SVC on bus 10 holding bus 9 (remote):
    its capability is a SUSCEPTANCE range, worth b * |V|^2 * sn_mva at the
    row's own voltage."""
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(_tight_net(max_q_mvar))
        if not hasattr(grid, "init_svcs"):
            pytest.skip("this lightsim2grid build has no init_svcs")
        grid.init_svcs([1], np.array([1.03]), np.array([0.0]), np.array([0.0]),
                       np.array([-b_max]), np.array([b_max]),
                       np.array([9], dtype=np.int32), np.array([10], dtype=np.int32))
        grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0


# ------------------------------------------------------------------ helpers
def _ls_physical(computer):
    """lightsim2grid's per-row physical report, whichever name the installed
    build has (compute_physical_violations, or PR #206's first
    compute_physical_violations)."""
    if hasattr(computer, "get_physical_violations"):
        return computer.get_physical_violations()
    return computer.get_bus_q_violations()


def _ls_physical_n(computer):
    if hasattr(computer, "get_physical_violations_n"):
        return computer.get_physical_violations_n()
    return computer.get_bus_q_violations_n()


def _ls_enable(computer, tol):
    if hasattr(computer, "compute_physical_violations"):
        computer.compute_physical_violations = True
        computer.physical_violation_tol_mva = tol
    else:   # PR #206's first naming
        setattr(computer, "compute_bus_q_violations", True)
        setattr(computer, "bus_q_violation_tol_mvar", tol)


def _me2s(grid):
    return np.asarray(grid.id_me_to_ac_solver(), dtype=int)


def _ref_rows(rows, me2s):
    """lightsim2grid records -> [(solver bus, type, value, limit)] per row."""
    return [[(int(me2s[v.element_id]), int(v.violation_type), float(v.value), float(v.limit))
             for v in row] for row in rows]


def _got_rows(rows):
    return [[(int(v.element_id), int(v.violation_type), float(v.value), float(v.limit))
             for v in row] for row in rows]


def _assert_ranked(rows):
    """gpusim2grid's documented order: LOW_Q records, then HIGH_Q ones, each
    type sorted by |value - limit|, largest first."""
    for r, row in enumerate(rows):
        types = [x[1] for x in row]
        assert types == sorted(types), f"row {r}: types not grouped LOW_Q then HIGH_Q: {row}"
        for x, y in zip(row, row[1:]):
            if x[1] == y[1]:
                assert abs(x[2] - x[3]) >= abs(y[2] - y[3]), f"row {r}: not most severe first: {row}"


def _assert_same(ref, got, atol_mvar):
    """Same records as lightsim2grid (which reports them in plan order; ours
    are ranked by type and severity, so compared as sets)."""
    assert len(ref) == len(got)
    _assert_ranked(got)
    for r, (a, b) in enumerate(zip(ref, got)):
        a, b = sorted(a), sorted(b)
        assert [x[:2] for x in a] == [x[:2] for x in b], f"row {r}: {a} vs {b}"
        for x, y in zip(a, b):
            np.testing.assert_allclose(y[2], x[2], atol=atol_mvar, err_msg=f"row {r} value")
            np.testing.assert_allclose(y[3], x[3], atol=atol_mvar, err_msg=f"row {r} limit")


def _mvar_atol(grid, solver_atol):
    # a voltage tolerance turns into a reactive-power one through sn_mva
    return 50. * float(grid.get_sn_mva()) * solver_atol


class _Ref:
    """lightsim2grid's ContingencyAnalysisCPP report, put back in the CALLER's
    contingency order (the C++ object keeps its own, ``my_defaults()``)."""
    def __init__(self, ca, contingencies):
        defaults = [sorted(int(x) for x in c) for c in ca.my_defaults()]
        order = [defaults.index(sorted(int(x) for x in c)) for c in contingencies]
        rows = _ls_physical(ca)
        conv = ca.converged_mask()
        self.rows = [rows[i] for i in order]
        self.converged = np.asarray([conv[i] for i in order], dtype=bool)
        self.rows_n = _ls_physical_n(ca)

    def get_physical_violations(self):
        return self.rows

    def get_physical_violations_n(self):
        return self.rows_n

    def converged_mask(self):
        return self.converged


def _ref_ca(grid, v0, contingencies, tol_mvar=0., handle_disconnected_grid=False):
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP
    ca = ContingencyAnalysisCPP(grid)
    _ls_enable(ca, tol_mvar)
    ca.handle_disconnected_grid = handle_disconnected_grid
    for c in contingencies:
        if len(c) == 1:
            ca.add_n1(int(c[0]))
        else:
            ca.add_nk([int(x) for x in c])
    ca.compute(v0.copy(), MAX_IT, TOL)
    return _Ref(ca, contingencies)


def _gpu_ca(grid, contingencies, tol_mvar=0., **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True, **kwargs)
    ca.physical_violation_tol_mva = tol_mvar
    ca.add_contingencies_by_branch_id(contingencies)
    ca.compute(batch_size=8)
    return ca


def _all_n1(grid):
    return [[i] for i in range(len(grid.get_lines()) + len(grid.get_trafos()))]


# ------------------------------------------------------------- contingency
@needs_bridge
@pytest.mark.parametrize("use_distributed_slack", [True, False])
def test_ca_n1_matches_lightsim2grid(solver_atol, use_distributed_slack):
    from gpusim2grid.contingency_analysis import ViolationCategory, ViolationElementType
    grid, v0 = _case14_tight_q()
    ctgs = _all_n1(grid)
    ref = _ref_ca(grid, v0, ctgs)
    gpu = _gpu_ca(grid, ctgs, use_distributed_slack=use_distributed_slack)

    got = gpu.get_physical_violations()
    assert sum(len(r) for r in got) > 0, "+/- 5 MVAr per machine cannot hold case14's voltages"
    _assert_same(_ref_rows(ref.get_physical_violations(), _me2s(grid)), _got_rows(got),
                 _mvar_atol(grid, solver_atol))
    for row in got:
        for v in row:
            assert v.element_type == ViolationElementType.BUS
            assert v.category == ViolationCategory.PHYSICAL
            assert v.side == 0
    # the base ("n") case is the very state lightsim2grid solved
    _assert_same(_ref_rows([ref.get_physical_violations_n()], _me2s(grid)),
                 _got_rows([gpu.get_physical_violations_n()]), _mvar_atol(grid, solver_atol))
    assert not gpu.get_physical_violations_truncated().any()
    assert gpu.solver.timings.t_bus_q_check.wall_ms >= 0.


@needs_bridge
def test_real_limits_and_wide_tolerance(solver_atol):
    """case14's own reactive ranges: the base case holds every generator
    inside them (nothing reported), a few N-1 rows genuinely do not (gen 1
    beyond its 50 MVAr when line 0 or 1 is out) -- and lightsim2grid says the
    same. A tolerance wider than any excess hides everything."""
    grid, v0 = _case14_default()
    ctgs = _all_n1(grid)[:5]
    gpu = _gpu_ca(grid, ctgs)
    ref = _ref_ca(grid, v0, ctgs)
    ref_rows = _ref_rows(ref.get_physical_violations(), _me2s(grid))
    assert any(len(r) > 0 for r in ref_rows) and any(len(r) == 0 for r in ref_rows)
    _assert_same(ref_rows, _got_rows(gpu.get_physical_violations()), _mvar_atol(grid, solver_atol))
    assert gpu.get_physical_violations_n() == [] and ref.get_physical_violations_n() == []

    grid, _ = _case14_tight_q()
    gpu = _gpu_ca(grid, ctgs, tol_mvar=1e6)
    assert all(len(r) == 0 for r in gpu.get_physical_violations())
    assert gpu.get_physical_violations_n() == []


@needs_bridge
def test_defaults_flag_off_raises_and_validation():
    from gpusim2grid import ContingencyAnalysisGPU
    grid, _ = _case14_tight_q()
    ca = ContingencyAnalysisGPU(grid, nb_iter=10)
    assert ca.compute_physical_violations is False
    assert ca.physical_violation_tol_mva == 1e-4
    assert ca.physical_violation_capacity == 16
    assert ca.has_bus_q_capability is False
    ca.add_contingencies_by_branch_id([[0], [1]])
    ca.compute()
    with pytest.raises(RuntimeError):
        ca.get_physical_violations()
    with pytest.raises(RuntimeError):
        ca.get_physical_violations_n()
    with pytest.raises(RuntimeError):
        ca.physical_violation_tol_mva = -1.
    with pytest.raises(RuntimeError):
        ca.physical_violation_capacity = 0
    # turning the flag on after construction pulls the plan off the grid and
    # takes effect at the next compute(), the registered contingencies kept
    ca.compute_physical_violations = True
    assert ca.has_bus_q_capability
    ca.compute()
    assert len(ca.get_physical_violations()) == 2
    assert all(len(r) > 0 for r in ca.get_physical_violations())


@needs_bridge
def test_independent_of_compute_limit_violations():
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.contingency_analysis import LimitViolationType, ViolationCategory
    grid, _ = _case14_tight_q()
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_limit_violations=True,
                                compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.add_contingencies_by_branch_id([[0], [1], [2]])
    ca.compute()
    phys = {LimitViolationType.LOW_Q, LimitViolationType.HIGH_Q}
    for row in ca.get_violations():
        assert all(v.violation_type not in phys for v in row)
        assert all(v.category != ViolationCategory.PHYSICAL for v in row)
    for row in ca.get_physical_violations():
        assert len(row) > 0
        assert all(v.violation_type in phys for v in row)
    # the reactive check works without the operational one, too
    ca2 = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca2.add_contingencies_by_branch_id([[0]])
    ca2.compute()
    assert ca2.compute_limit_violations is False
    with pytest.raises(RuntimeError):
        ca2.get_violations()
    assert len(ca2.get_physical_violations()[0]) > 0


@needs_bridge
def test_capacity_truncation(solver_atol):
    grid, v0 = _case14_tight_q()
    ctgs = _all_n1(grid)[:6]
    ref = _ref_ca(grid, v0, ctgs)
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=10, compute_physical_violations=True)
    ca.physical_violation_tol_mva = 0.
    ca.physical_violation_capacity = 1
    ca.add_contingencies_by_branch_id(ctgs)
    ca.compute()
    got = ca.get_physical_violations()
    trunc = ca.get_physical_violations_truncated()
    ref_rows = _ref_rows(ref.get_physical_violations(), _me2s(grid))
    saw_truncation = False
    for r, (a, b) in enumerate(zip(ref_rows, got)):
        # capacity 1 is per TYPE: the single most severe LOW_Q and HIGH_Q
        kept = []
        for vtype in (LOW_Q, HIGH_Q):
            of_type = [x for x in a if x[1] == vtype]
            if of_type:
                kept.append(max(of_type, key=lambda x: abs(x[2] - x[3]))[:2])
        assert [(v.element_id, int(v.violation_type)) for v in b] == kept, f"row {r}"
        n_of = [sum(x[1] == t for x in a) for t in (LOW_Q, HIGH_Q)]
        assert bool(trunc[r]) == (max(n_of) > 1)
        saw_truncation |= bool(trunc[r])
    assert saw_truncation, "the test grid must have a row with several violations of one type"


@needs_bridge
def test_islanding_contingency(solver_atol):
    """A contingency that islands a bus: skipped (empty entry, never simulated)
    without handle_disconnected_grid, solved on the main component with it --
    and then equal to lightsim2grid's own masked solve."""
    grid, v0, spur = _tight_spur_grid()
    ctgs = [[spur], [0]]
    # without the mode: the islanding row is NOT_SIMULATED -> empty report
    gpu = _gpu_ca(grid, ctgs)
    got = gpu.get_physical_violations()
    assert got[0] == []
    assert np.isnan(gpu.last_residuals()[0])                     # never simulated
    assert gpu.solver._s.get_bus_q_violations().count[0] == -1   # the sentinel behind it
    assert len(got[1]) > 0
    # with it: the spur bus is masked, the rest is checked as usual
    ref = _ref_ca(grid, v0, ctgs, handle_disconnected_grid=True)
    gpu = _gpu_ca(grid, ctgs, handle_disconnected_grid=True)
    assert np.isfinite(gpu.last_residuals()[0])
    _assert_same(_ref_rows(ref.get_physical_violations(), _me2s(grid)),
                 _got_rows(gpu.get_physical_violations()), _mvar_atol(grid, solver_atol))


@needs_bridge
@pytest.mark.parametrize("make", [_tight_voltage_control_grid, _tight_svc_grid])
def test_voltage_control_grids_match(solver_atol, make):
    """Bordered VoltageControl (remote generators, a voltage-mode SVC): the
    controllers' reactive output is a solved unknown, and lightsim2grid adds
    it back onto its mismatch -- the GPU's raw residual must land on the same
    per-bus reactive power, SVC susceptance capability included."""
    grid, v0 = make()
    ctgs = _all_n1(grid)[:8]
    ref = _ref_ca(grid, v0, ctgs)
    gpu = _gpu_ca(grid, ctgs)
    ref_rows = _ref_rows(ref.get_physical_violations(), _me2s(grid))
    got_rows = _got_rows(gpu.get_physical_violations())
    # a row lightsim2grid's KLU did not converge on has an empty reference
    # entry by construction (nothing to compare); the GPU may well converge
    # there -- keep the rows both sides solved
    conv = np.asarray(ref.converged_mask(), dtype=bool)
    assert conv.sum() >= 4
    ref_rows = [r for r, c in zip(ref_rows, conv) if c]
    got_rows = [r for r, c in zip(got_rows, conv) if c]
    assert sum(len(r) for r in ref_rows) > 0
    _assert_same(ref_rows, got_rows, _mvar_atol(grid, solver_atol))
    _assert_same(_ref_rows([ref.get_physical_violations_n()], _me2s(grid)),
                 _got_rows([gpu.get_physical_violations_n()]), _mvar_atol(grid, solver_atol))


STORAGE_BUS, STORAGE_MAX_Q = 9, 0.5


def _tight_storage_grid(max_q_mvar=5.):
    """Tight case14 with a storage unit regulating its own bus 9 (a load bus,
    no generator): a PV bus like a local generator's, whose [min_q, max_q]
    joins that bus' capability as a fixed term (no row disconnects it)."""
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(_tight_net(max_q_mvar))
        if not hasattr(grid, "init_storages_full"):
            pytest.skip("this lightsim2grid build has no init_storages_full")
        grid.init_storages_full(np.array([0.0]), np.array([0.0]), [True], np.array([1.035]),
                                np.array([-STORAGE_MAX_Q]), np.array([STORAGE_MAX_Q]),
                                np.array([STORAGE_BUS], dtype=np.int32))
        grid.tell_solver_need_reset()
    v0 = _solve(grid)
    return grid, v0


@needs_bridge
def test_storage_unit_capability_matches(solver_atol):
    """A voltage-regulating storage unit holds its bus like a local generator:
    the bridge must carry its reactive range into the plan, or the GPU
    compares that bus against nothing and disagrees with lightsim2grid."""
    grid, v0 = _tight_storage_grid()
    ctgs = _all_n1(grid)[:8]
    ref = _ref_ca(grid, v0, ctgs)
    gpu = _gpu_ca(grid, ctgs)
    atol = _mvar_atol(grid, solver_atol)
    conv = np.asarray(ref.converged_mask(), dtype=bool)
    assert conv.sum() >= 4
    ref_rows = [r for r, c in zip(_ref_rows(ref.get_physical_violations(), _me2s(grid)), conv) if c]
    got_rows = [r for r, c in zip(_got_rows(gpu.get_physical_violations()), conv) if c]
    _assert_same(ref_rows, got_rows, atol)
    ref_n = _ref_rows([ref.get_physical_violations_n()], _me2s(grid))
    _assert_same(ref_n, _got_rows([gpu.get_physical_violations_n()]), atol)
    # the storage unit's bus is actually reported, against its own range alone
    sto_solver = int(_me2s(grid)[STORAGE_BUS])
    at_sto = [x for x in ref_n[0] if x[0] == sto_solver]
    assert len(at_sto) == 1, "the fixture needs the storage unit's bus to be reported"
    np.testing.assert_allclose(abs(at_sto[0][3]), STORAGE_MAX_Q, atol=1e-9)


@needs_bridge
def test_tuple_mode_matches_bridge(solver_atol):
    """Explicit-array mode: the plan is handed in as arrays (or the
    BusQPlanData object) and the report equals the bridge path's."""
    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid import _gpusim2grid as _cpp
    from gpusim2grid._ls2g_utils import extract_grid_arrays, extract_branch_data
    grid, _ = _case14_tight_q()
    ctgs = _all_n1(grid)[:6]
    ref = _gpu_ca(grid, ctgs)

    d = extract_grid_arrays(grid, max_iter=MAX_IT, tol=TOL)
    branch_args, _, _ = extract_branch_data(grid)
    plan = _cpp._extract_bus_q_plan_from_lsgrid(grid, grid.get_Ybus_solver().shape[0])
    arrays = (plan.bus_solver, plan.qmin_fixed_mvar, plan.qmax_fixed_mvar, plan.n_fixed,
              plan.bmin_sum_pu, plan.bmax_sum_pu, plan.gen_start, plan.gen_id,
              plan.gen_qmin_mvar, plan.gen_qmax_mvar, plan.sn_mva)
    for handed in (plan, arrays):
        ca = ContingencyAnalysisGPU(
            (d["Ybus"], d["v_converged"], d["Sbus"], d["slack"], d["slack_weights"], d["pv"], d["pq"]),
            nb_iter=10)
        ca.set_branch_data(*branch_args)
        with pytest.raises(RuntimeError):
            ca.set_bus_q_capability_from_grid()   # no grid in tuple mode
        ca.compute_physical_violations = True
        ca.add_contingencies_by_branch_id(ctgs)
        with pytest.raises(RuntimeError):
            ca.compute()                          # flag on, no plan yet
        ca.set_bus_q_capability(handed)
        ca.physical_violation_tol_mva = 0.
        ca.compute()
        _assert_same(_got_rows(ref.get_physical_violations()), _got_rows(ca.get_physical_violations()),
                     _mvar_atol(grid, solver_atol))
    # a plan that does not fit the grid is refused
    bad = list(arrays); bad[0] = np.array([10**6], dtype=np.int32)
    bad[3] = np.array([0], dtype=np.int32); bad[1] = bad[2] = bad[4] = bad[5] = np.zeros(1)
    bad[6] = np.array([0, 0], dtype=np.int32)
    with pytest.raises(RuntimeError):
        ca.set_bus_q_capability(tuple(bad))


def test_enum_parity_with_lightsim2grid():
    ls = pytest.importorskip("lightsim2grid.lightsim2grid_cpp")
    from gpusim2grid.contingency_analysis import (
        LimitViolationType, ViolationCategory, ViolationElementType, LimitViolation,
        violation_category)
    if not hasattr(ls, "ViolationCategory"):
        pytest.skip("this lightsim2grid build predates ViolationCategory")
    for name, member in ls.LimitViolationType.__members__.items():
        assert int(LimitViolationType[name]) == int(member), name
        assert int(violation_category(LimitViolationType[name])) == int(ls.violation_category(member)), name
    for name, member in ls.ViolationCategory.__members__.items():
        assert int(ViolationCategory[name]) == int(member), name
    for name, member in ls.ViolationElementType.__members__.items():
        assert int(ViolationElementType[name]) == int(member), name
    v = LimitViolation(ViolationElementType.BUS, 3, 0, LimitViolationType.HIGH_Q, 12., 5.)
    assert v.category == ViolationCategory.PHYSICAL
    assert LimitViolation(ViolationElementType.GRID, -1, 0, LimitViolationType.DIVERGENCE,
                          1., 1e-6).category == ViolationCategory.SOLVER
    assert LimitViolation(ViolationElementType.LINE, 0, 1, LimitViolationType.CURRENT,
                          1., 0.5).category == ViolationCategory.OPERATIONAL


# ---------------------------------------------------------- scenario sweep
def _elements(grid):
    load_p, load_q = grid.get_loads_res_full()[:2]
    return (np.asarray(load_p, dtype=np.float64), np.asarray(load_q, dtype=np.float64),
            np.asarray(grid.get_gen_target_p(), dtype=np.float64))


@needs_bridge
def test_scenario_sweep_matches_lightsim2grid(solver_atol):
    """Rows mixing a topology change, generator contingencies (a disconnected
    generator leaves its bus' capability; the bus of an only generator turns
    PQ and is not checked at all) and an injection change, on a persistent
    driver (a second, warm run must not show the first run's records)."""
    ScenarioSweepCPP = pytest.importorskip("lightsim2grid.scenarioSweep").ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    grid, v0 = _case14_tight_q()
    me2s = _me2s(grid)
    gens = grid.get_generators()
    n_gen, n_line = len(gens), len(grid.get_lines())
    non_slack = [g for g in range(n_gen) if not gens[g].is_slack]
    n_rows = 2 + len(non_slack)
    lines_off = np.zeros((n_rows, n_line), dtype=bool)
    lines_off[1, 0] = True
    gens_off = np.zeros((n_rows, n_gen), dtype=bool)
    for r, g in enumerate(non_slack):
        gens_off[2 + r, g] = True
    load_p, load_q, gen_p = _elements(grid)
    rep = lambda a, k=1.0: np.repeat((k * a)[None, :], n_rows, axis=0)   # noqa: E731

    def ref_run(k_load):
        ls = ScenarioSweepCPP(grid)
        _ls_enable(ls, 0.)
        ls.set_contingency_lines(lines_off)
        ls.set_contingency_gens(gens_off)
        ls.modify_load_p(rep(load_p, k_load))
        ls.modify_load_q(rep(load_q, k_load))
        ls.compute(v0.copy(), MAX_IT, TOL)
        assert all(ls.converged_mask())
        return _ref_rows(_ls_physical(ls), me2s), _ref_rows([_ls_physical_n(ls)], me2s)

    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_topology([[], [0]] + [[] for _ in non_slack])
    sw.set_contingency_gens(gens_off)
    sw.compute(batch_size=n_rows)
    ref_rows, ref_n = ref_run(1.0)
    got = sw.get_physical_violations()
    atol = _mvar_atol(grid, solver_atol)
    _assert_same(ref_rows, _got_rows(got), atol)
    _assert_same(ref_n, _got_rows([sw.get_physical_violations_n()]), atol)
    # a released bus (its only generator off) is never reported on that row
    for r, g in enumerate(non_slack):
        bus = int(me2s[gens[g].bus_id])
        assert all(v.element_id != bus for v in got[2 + r])
    # hot re-run with other injections: fresh report, no stale row
    sw.set_injections_from_elements(rep(load_p, 1.1), rep(load_q, 1.1), rep(gen_p))
    sw.compute(batch_size=n_rows)
    ref_rows, _ = ref_run(1.1)
    _assert_same(ref_rows, _got_rows(sw.get_physical_violations()), atol)
    # warm re-run: drop the generator contingencies (structure released)
    sw.set_contingency_gens(np.zeros((n_rows, n_gen), dtype=bool))
    sw.compute(batch_size=n_rows)
    ls = ScenarioSweepCPP(grid)
    _ls_enable(ls, 0.)
    ls.set_contingency_lines(lines_off)
    ls.modify_load_p(rep(load_p, 1.1))
    ls.modify_load_q(rep(load_q, 1.1))
    ls.compute(v0.copy(), MAX_IT, TOL)
    _assert_same(_ref_rows(_ls_physical(ls), me2s), _got_rows(sw.get_physical_violations()), atol)


@needs_bridge
def test_scenario_sweep_skipped_row_is_empty():
    from gpusim2grid import ScenarioSweepGPU
    grid, v0, spur = _tight_spur_grid()
    load_p, load_q, gen_p = _elements(grid)
    rep = lambda a: np.repeat(a[None, :], 2, axis=0)   # noqa: E731
    sw = ScenarioSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_topology([[spur], []])
    sw.compute()
    got = sw.get_physical_violations()
    assert got[0] == [] and len(got[1]) > 0
    assert sw.get_disconnected()[0] == 1


# --------------------------------------------------------- injection sweep
@needs_bridge
def test_injection_sweep_matches_lightsim2grid(solver_atol):
    InjectionSweepCPP = pytest.importorskip("lightsim2grid.lightsim2grid_cpp").InjectionSweepCPP
    from gpusim2grid import InjectionSweepGPU
    grid, v0 = _case14_tight_q()
    load_p, load_q, gen_p = _elements(grid)
    ks = [0.9, 1.0, 1.1, 50.0]   # the last one diverges on both sides
    lp = np.stack([k * load_p for k in ks]); lq = np.stack([k * load_q for k in ks])
    gp = np.repeat(gen_p[None, :], len(ks), axis=0)

    ls = InjectionSweepCPP(grid)
    _ls_enable(ls, 0.)
    ls.modify_load_p(lp); ls.modify_load_q(lq); ls.modify_gen_p(gp)
    ls.compute(v0.copy(), MAX_IT, TOL)
    conv = np.asarray(ls.converged_mask(), dtype=bool)
    assert conv[:3].all() and not conv[3]

    sw = InjectionSweepGPU(grid, nb_iter=10, compute_physical_violations=True)
    sw.physical_violation_tol_mva = 0.
    sw.set_injections_from_elements(lp, lq, gp)
    sw.compute()
    got = sw.get_physical_violations()
    assert got[3] == [] and sw.last_residuals()[3] > sw.solver.violation_tol \
        if hasattr(sw.solver, "violation_tol") else got[3] == []
    atol = _mvar_atol(grid, solver_atol)
    _assert_same(_ref_rows(_ls_physical(ls), _me2s(grid))[:3], _got_rows(got)[:3], atol)
    _assert_same(_ref_rows([_ls_physical_n(ls)], _me2s(grid)),
                 _got_rows([sw.get_physical_violations_n()]), atol)
    assert sw.solver.timings.t_physical_setup_ms > 0.
