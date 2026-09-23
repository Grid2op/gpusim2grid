# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""gen_v acts on the bus a generator REGULATES, and tied set-points share one
derivative.

Two things used to be wrong, both found by finite differences on real RTE
snapshots:

* a remote voltage regulator (``regulated_bus_id`` != its own bus, the bordered
  ``VoltageControl`` extension) had its ``gen_v`` keyed on its OWN bus. That bus
  is not Vm-fixed (at best a slack bus whose |V| is still an unknown), so the
  reseed was overwritten by the first Newton step: the set-point was silently
  ignored, forward and gradient alike. It now drives the group's per-row
  ``v_set`` in ``|V_reg| + s.Q - v_set = 0``, and its gradient is lambda at that
  row;
* generators regulating the same bus must agree on a row, so only the
  derivative along the tie exists; each column used to receive all of it (a
  group moved together saw n times the gradient). It is now split equally
  between the columns that apply on that row.

Every check is a finite difference, or a one-off lightsim2grid solve with the
set-point actually changed.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge, is_fp32

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed -- skipping")

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge, reason="VoltageControl needs the lightsim2grid C++ bridge")
fp64_only = pytest.mark.skipif(
    bool(is_fp32), reason="finite-difference gradient checks need the FP64 build")

RDT = torch.float32 if is_fp32 else torch.float64
GEN_REMOTE, OWN_BUS, REG_BUS = 3, 7, 9      # case14: gen 3 on bus 7 regulates bus 9
TRAFO_BEHIND = 3                            # trips bus 7 off the grid
NB_ITER, TOL = 20, 1e-11
EPS = 1e-6


def _remote_case14(v_target=None):
    """Solved case14 with gen 3 (bus 7) remotely regulating bus 9, optionally
    at another set-point (the one-off reference of a batch row)."""
    pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
    if not hasattr(model, "set_gen_regulated_bus"):
        pytest.skip("this lightsim2grid build has no set_gen_regulated_bus")
    model.set_gen_regulated_bus(GEN_REMOTE, REG_BUS)
    if v_target is not None:
        model.change_v_gen(GEN_REMOTE, float(v_target))     # pu
    model.tell_solver_need_reset()
    model.change_algorithm(AlgorithmType.NR_KLU)
    V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), 40, TOL)
    assert V.shape[0] > 0, "lightsim2grid diverged"
    return model, V


def _pf(grid, **kw):
    from gpusim2grid.differentiable import BatchPowerFlow
    kw.setdefault("nb_iter", NB_ITER)
    kw.setdefault("tol_base", TOL)
    return BatchPowerFlow.from_lsgrid(grid, **kw)


def _loss(pf, V):
    """A loss touching every bus magnitude and angle, and every branch flow."""
    n = pf.n_bus
    wr = torch.linspace(0.5, 1.5, n, dtype=RDT, device="cuda")
    wi = torch.linspace(-1.0, 1.0, n, dtype=RDT, device="cuda")
    ok = torch.isfinite(V.real) & torch.isfinite(V.imag)
    Vs = torch.where(ok, V, torch.zeros_like(V))
    p1 = pf.compute_flows(Vs)["p_or_mw"]
    p1 = torch.where(torch.isfinite(p1), p1, torch.zeros_like(p1))
    wf = torch.linspace(-1.0, 1.0, p1.shape[1], dtype=RDT, device="cuda")
    return (Vs.real * wr).sum() + (Vs.imag * wi).sum() + 1e-3 * (p1 * wf).sum()


def _analytic_and_fd(pf, gen_v, moves, **inputs):
    """Gradient of _loss w.r.t. gen_v, and for each (row, [cols]) in `moves` the
    central difference moving those columns TOGETHER; returns
    [(sum of the analytic gradient over cols, fd)]."""
    gv = gen_v.clone().requires_grad_(True)
    _loss(pf, pf(gen_v=gv, **inputs)).backward()
    out = []
    for row, cols in moves:
        vals = []
        for sign in (1.0, -1.0):
            x = gen_v.clone()
            x[row, cols] += sign * EPS
            with torch.no_grad():
                vals.append(_loss(pf, pf(gen_v=x, **inputs)).item())
        out.append((gv.grad[row, cols].sum().item(), (vals[0] - vals[1]) / (2 * EPS)))
    return gv.grad, out


# ---------------------------------------------------------------------------
# the key
# ---------------------------------------------------------------------------

@needs_bridge
def test_gen_v_bus_is_the_regulated_bus():
    from gpusim2grid._ls2g_utils import extract_injection_elements
    model, _ = _remote_case14()
    n_bus = model.get_Ybus_solver().shape[0]
    me2s = np.asarray(model.id_me_to_ac_solver())
    el = extract_injection_elements(model, n_bus)
    assert el.gen_bus[GEN_REMOTE] == me2s[OWN_BUS]
    assert el.gen_v_bus[GEN_REMOTE] == me2s[REG_BUS]
    local = [g for g, e in enumerate(model.get_generators())
             if e.voltage_regulator_on and g != GEN_REMOTE]
    assert all(el.gen_v_bus[g] == el.gen_bus[g] for g in local)


# ---------------------------------------------------------------------------
# forward: a remote regulator's set-point is applied, on every batch facade
# ---------------------------------------------------------------------------

@needs_bridge
def test_remote_setpoint_matches_one_off_solves(solver_atol):
    from gpusim2grid import InjectionSweepGPU, ScenarioSweepGPU
    model, V_base = _remote_case14()
    v_rows = [1.03, 1.055]
    refs = [_remote_case14(v)[1] for v in v_rows]
    assert abs(abs(refs[0][REG_BUS]) - abs(V_base[REG_BUS])) > 1e-3   # really moves
    for v, ref in zip(v_rows, refs):
        assert abs(ref[REG_BUS]) == pytest.approx(v, abs=1e-8)

    n = len(v_rows)
    gen_v = np.full((n, len(model.get_generators())), np.nan)
    gen_v[:, GEN_REMOTE] = v_rows
    ref = np.stack(refs)

    pf = _pf(model)
    V = pf(gen_v=torch.as_tensor(gen_v, dtype=RDT, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(V, ref, atol=10 * solver_atol)

    sn = model.get_sn_mva()
    S = np.asarray(model.get_Sbus_solver())
    p = np.tile(S.real * sn, (n, 1))
    q = np.tile(S.imag * sn, (n, 1))
    n_bus = S.shape[0]
    for cls in (ScenarioSweepGPU, InjectionSweepGPU):
        sw = cls(model, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections(p, q, sn)
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=4)
        V = sw.solver.V_results.to_numpy().reshape(n, n_bus)
        np.testing.assert_allclose(V, ref, atol=10 * solver_atol, err_msg=cls.__name__)


# ---------------------------------------------------------------------------
# gradients vs finite differences
# ---------------------------------------------------------------------------

@needs_bridge
@fp64_only
def test_remote_setpoint_gradient_matches_finite_differences():
    model, _ = _remote_case14()
    pf = _pf(model)
    n = 2
    gens = model.get_generators()
    gen_v = torch.full((n, len(gens)), float("nan"), dtype=RDT, device="cuda")
    gen_v[:, GEN_REMOTE] = torch.tensor([1.03, 1.05], dtype=RDT)
    gen_v[:, 1] = float(gens[1].target_vm_pu)      # a local regulator, for company
    load_p = pf._load_p_base[None, :].expand(n, -1) * torch.tensor([[1.0], [1.08]], dtype=RDT, device="cuda")
    grad, pairs = _analytic_and_fd(pf, gen_v, [(0, [GEN_REMOTE]), (1, [GEN_REMOTE]), (1, [1])],
                                   load_p=load_p)
    for an, fd in pairs:
        assert abs(fd) > 1e-3                        # a real sensitivity ...
        assert an == pytest.approx(fd, rel=1e-5, abs=1e-7)   # ... and the right one


@needs_bridge
@fp64_only
def test_stranded_remote_setpoint_has_no_gradient(solver_atol):
    """handle_disconnected_grid: tripping the trafo behind the controller's own
    bus strands it; its voltage row becomes Q_c == 0, v_set leaves the system,
    and both the finite difference and the gradient are 0 on that row."""
    model, _ = _remote_case14()
    pf = _pf(model, handle_disconnected_grid=True)
    n = 2
    gen_v = torch.full((n, len(model.get_generators())), float("nan"), dtype=RDT, device="cuda")
    gen_v[:, GEN_REMOTE] = 1.04
    line_status = torch.ones(n, pf.n_line, dtype=torch.bool, device="cuda")
    trafo_status = torch.ones(n, pf.n_trafo, dtype=torch.bool, device="cuda")
    trafo_status[1, TRAFO_BEHIND] = False
    grad, pairs = _analytic_and_fd(pf, gen_v, [(0, [GEN_REMOTE]), (1, [GEN_REMOTE])],
                                   line_status=line_status, trafo_status=trafo_status)
    assert pf.get_disconnected().tolist() == [0, 0]
    (an0, fd0), (an1, fd1) = pairs
    assert abs(fd0) > 1e-3 and an0 == pytest.approx(fd0, rel=1e-5, abs=1e-7)
    assert abs(fd1) < 1e-6 and an1 == 0.0


def _two_gen_case14():
    """case14 with a 2nd generator on bus 5, regulating it with the first."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        pp.create_gen(net, bus=5, p_mw=10.0, vm_pu=float(net.gen.vm_pu.iloc[2]),
                      controllable=True, min_q_mvar=-50., max_q_mvar=50.)
        grid = init_from_pandapower(net)
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
    assert grid.ac_pf(v0.copy(), 30, 1e-10).shape[0] > 0
    shared = [g for g, e in enumerate(grid.get_generators()) if e.bus_id == 5]
    assert len(shared) == 2
    return grid, shared


@fp64_only
def test_tied_setpoints_share_one_derivative():
    """Two generators regulating one bus: moving both together is the only
    feasible direction, and the two gradients sum to its finite difference
    (they used to be each equal to it). With one of them NaN on a row, the
    other owns the whole derivative there."""
    grid, (g_a, g_b) = _two_gen_case14()
    pf = _pf(grid)
    n = 2
    gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
    gen_v[0, [g_a, g_b]] = 1.03
    gen_v[1, g_a] = 1.045                   # g_b NaN on row 1: only g_a applies
    grad, pairs = _analytic_and_fd(pf, gen_v, [(0, [g_a, g_b]), (1, [g_a])])
    for an, fd in pairs:
        assert abs(fd) > 1e-3
        assert an == pytest.approx(fd, rel=1e-5, abs=1e-7)
    assert grad[0, g_a].item() == pytest.approx(grad[0, g_b].item(), rel=1e-12)
    assert grad[1, g_b].item() == 0.0


# ---------------------------------------------------------------------------
# conflicts: a group set-point is one more value that must agree
# ---------------------------------------------------------------------------

def test_conflicting_rows_with_voltage_control_groups():
    from gpusim2grid._ls2g_utils import conflicting_gen_v_rows
    nan = float("nan")
    # gen 0 -> Vm-fixed bus 0; gens 1, 2 -> bus 2 held by group 0 (free);
    # gen 3 -> bus 3 held by group 1, pinned at 1.01 by an SVC / station member
    gen_bus = np.array([0, 2, 2, 3])
    fixed = np.array([True, False, False, False])
    group = np.array([-1, -1, 0, 1])
    pinned = np.array([nan, 1.01])
    gen_v = np.array([
        [1.0, 1.02, 1.02, 1.01],       # agree, pinned value kept   -> ok
        [1.0, 1.02, 1.03, 1.01],       # two remote regulators disagree -> conflict
        [1.0, 1.02, 1.02, 1.03],       # moves a pinned set-point   -> conflict
        [1.0, 1.02, nan, nan],         # nothing to compare         -> ok
    ])
    np.testing.assert_array_equal(
        conflicting_gen_v_rows(gen_v, gen_bus, fixed, vc_group_of_bus=group,
                               vc_pinned_v_set=pinned),
        [False, True, True, False])
    # without the group map, a column on a non-fixed bus is not driven at all
    np.testing.assert_array_equal(
        conflicting_gen_v_rows(gen_v, gen_bus, fixed), [False, False, False, False])
