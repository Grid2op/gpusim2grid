# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The host side of the outer loops (outer/ls2g_outer_model.cpp), without a GPU.

The model -- the port of lightsim2grid's NROuter_* driver and of its loops --
decides on every converged inner solve; here those solves are a small dense
Newton in numpy instead of the device's (the test-only ``_OuterModelHarness``
binding). Each row must end where lightsim2grid's own single ``NROuter_KLU``
solve of the row's grid ends, after the same number of outer iterations, with
the same decisions and the same targets moved. A difference here is a
difference in the port, not in the GPU solve.
"""

import warnings

import numpy as np
import pytest

import gpusim2grid._gpusim2grid as _cpp

pytestmark = pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                                reason="gpusim2grid built against a lightsim2grid without NROuter_*")

OUTER = "NROuter_KLU"
MAX_IT = 30
TOL_MVA = 1e-8
CONVERGED, MAX_ITER, DIVERGED = 0, 1, 2
RESOLVE, DONE = 0, 1


def _case14(q_limit=15., distributed=True, load_scale=1., line_off=None):
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    net = pn.case14()
    net.gen["min_q_mvar"] = -q_limit
    net.gen["max_q_mvar"] = q_limit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 300.))
    if distributed:
        grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
    grid.change_algorithm(OUTER)
    if load_scale != 1.:
        for load_id, p in enumerate(np.asarray(grid.get_load_target_p())):
            grid.change_p_load(load_id, float(p) * load_scale)
    if line_off is not None:
        grid.deactivate_powerline(line_off)
    return grid


def _newton(Ybus, V, Sbus, ref, vm_unknown, tol, max_it):
    """A dense polar Newton: P at every bus but the reference, Q (and a Vm
    unknown) where vm_unknown. Returns (V, converged, iterations); lightsim2grid
    checks the residual before each step, so does this one."""
    n = V.shape[0]
    th_idx = np.array([b for b in range(n) if b != ref])
    vm_idx = np.flatnonzero(vm_unknown)
    for it in range(max_it + 1):
        Ibus = Ybus @ V
        mis = V * np.conj(Ibus) - Sbus
        F = np.concatenate([mis.real[th_idx], mis.imag[vm_idx]])
        if np.max(np.abs(F)) < tol:
            return V, True, it
        if it == max_it:
            return V, False, it
        dV = V / np.abs(V)
        dS_dVm = np.diag(V) @ np.conj(Ybus @ np.diag(dV)) + np.diag(np.conj(Ibus)) @ np.diag(dV)
        dS_dVa = 1j * np.diag(V) @ np.conj(np.diag(Ibus) - Ybus @ np.diag(V))
        J = np.block([
            [dS_dVa.real[np.ix_(th_idx, th_idx)], dS_dVm.real[np.ix_(th_idx, vm_idx)]],
            [dS_dVa.imag[np.ix_(vm_idx, th_idx)], dS_dVm.imag[np.ix_(vm_idx, vm_idx)]],
        ])
        dx = np.linalg.solve(J, -F)
        Va = np.angle(V)
        Vm = np.abs(V)
        Va[th_idx] += dx[:th_idx.size]
        Vm[vm_idx] += dx[th_idx.size:]
        V = Vm * np.exp(1j * Va)
    return V, False, max_it


def _run_row(h, row, Ybus, Sbus_row, ref, tol_mva=TOL_MVA, sn_mva=100.):
    """one row through the model, the inner solves in numpy"""
    pin_bus = np.asarray(h.pin_bus, dtype=int)
    watch = np.asarray(h.watch_bus, dtype=int)
    pq = np.asarray(h.pq, dtype=int)
    realistic = np.asarray(h.realistic_bus, dtype=int)
    n = Ybus.shape[0]
    V = np.asarray(h.V0).copy()
    h.load(0, row)
    status = None
    for _ in range(100):
        c = h.controls(0)
        Sbus = Sbus_row.copy()
        for b, dp, dq in zip(c["sbus_bus"], c["sbus_dp"], c["sbus_dq"]):
            Sbus[b] += dp + 1j * dq
        for b, vm in zip(c["vm_reset_bus"], c["vm_reset_vm"]):
            V[b] = vm * V[b] / abs(V[b])
        h.clear_vm_reset(0)
        vm_unknown = np.zeros(n, dtype=bool)
        vm_unknown[pq] = True
        pinned = pin_bus[np.asarray(c["pin"], dtype=bool)]
        free = pin_bus[~np.asarray(c["pin"], dtype=bool)]
        vm_unknown[free] = True
        vm_unknown[pinned] = False
        V, conv, nr = _newton(Ybus, V, Sbus, ref, vm_unknown, tol_mva / sn_mva, MAX_IT)
        S = V * np.conj(Ybus @ V)
        cand = [b for b in realistic if vm_unknown[b]]
        unrealistic = bool(np.any((np.abs(V[cand]) < 0.5) | (np.abs(V[cand]) > 2.))) if cand else False
        nxt, status = h.on_solved(0, conv, nr, CONVERGED if conv else MAX_ITER,
                                  list(V[watch]), list(S[watch]), list(Sbus[watch]), [], unrealistic)
        if nxt == DONE:
            return V, status
    raise AssertionError("the row never ended")


@pytest.mark.parametrize("load_scale, line_off", [(1., None), (1.05, 1), (1.1, 2), (1.2, 3),
                                                  (1.3, 5), (1.0, 7)])
def test_host_model_matches_single_nrouter_solve(load_scale, line_off):
    base = _case14()
    base.ac_pf(np.full(base.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    h = _cpp._OuterModelHarness(base, keep_trace=True)
    h.begin(1, 1)

    # the row's own grid, solved by lightsim2grid from the same start
    row = _case14(load_scale=load_scale, line_off=line_off)
    v_start = np.asarray(base.get_V(), dtype=complex)
    V_ref = row.ac_pf(v_start, MAX_IT, TOL_MVA)
    stats = row.get_algo().get_outer_loop_stats()
    Ybus = row.get_Ybus_solver().toarray()
    Sbus_row = np.asarray(row.get_Sbus_solver())
    ref = int(np.asarray(h.slack)[0])

    V, status = _run_row(h, 0, Ybus, Sbus_row, ref)
    res = h.row_result(0)
    msg = "\n".join(res.decisions) + "\n---- lightsim2grid\n" + "\n".join(str(d) for d in stats.decisions)
    assert res.nb_outer_iterations == stats.nb_outer_iterations, msg
    assert int(res.status) == int(stats.status), msg
    if V_ref.size == 0:
        assert status != CONVERGED, msg
        return
    assert status == CONVERGED, msg
    buses = np.asarray(row.id_ac_solver_to_me(), dtype=int)
    np.testing.assert_allclose(V, V_ref[buses], rtol=0., atol=1e-8, err_msg=msg)
    # the targets DistributedSlack moved: what each generator produces, but the one
    # on the reference bus, which also takes what the loop left on the slack
    to_solver = np.asarray(row.id_me_to_ac_solver(), dtype=int)
    for g, mw in res.gen_target_p:
        gen = row.get_generators()[g]
        if to_solver[gen.bus_id] == ref:
            continue
        assert mw == pytest.approx(gen.res_p_mw, abs=1e-6), msg


def test_ample_reactive_limits_only_the_slack_is_shared():
    """Ample reactive limits: ReactiveLimits never acts; the slack's own unit is the
    only participant, so DistributedSlack does not run (lightsim2grid's
    DistributedSlackLoop::_is_needed asks for a unit off the slack bus): the slack
    bus keeps the row's extra load."""
    base = _case14(q_limit=1e4, distributed=False)
    base.ac_pf(np.full(base.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    h = _cpp._OuterModelHarness(base, keep_trace=True)
    h.begin(1, 1)
    row = _case14(q_limit=1e4, distributed=False, load_scale=1.1)
    V_ref = row.ac_pf(np.asarray(base.get_V(), dtype=complex), MAX_IT, TOL_MVA)
    stats = row.get_algo().get_outer_loop_stats()
    V, status = _run_row(h, 0, row.get_Ybus_solver().toarray(), np.asarray(row.get_Sbus_solver()),
                         int(np.asarray(h.slack)[0]))
    res = h.row_result(0)
    assert status == CONVERGED
    assert res.nb_outer_iterations == stats.nb_outer_iterations == 0
    assert not any("ReactiveLimits" in d or "DistributedSlack" in d for d in res.decisions)
    buses = np.asarray(row.id_ac_solver_to_me(), dtype=int)
    np.testing.assert_allclose(V, V_ref[buses], rtol=0., atol=1e-8)


def _ieee300(load_scale=1.):
    """pypowsybl's IEEE 300-bus grid with ReactiveLimits alone, as lightsim2grid's
    own test against OpenLoadFlow builds it (every unit regulates its own bus)"""
    pp = pytest.importorskip("pypowsybl")
    import sys, os
    import lightsim2grid
    sys.path.insert(0, os.path.join(os.path.dirname(lightsim2grid.__file__), "tests"))
    from _olf_reference import reference_parameters
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl, OlfLoadingParameters
    from lightsim2grid.network.from_pypowsybl._aux_add_slack import _default_distributed_slack
    net = pp.network.create_ieee300()
    if load_scale != 1.:
        loads = net.get_loads()
        net.update_loads(id=loads.index, p0=loads["p0"] * load_scale)
    gen = net.get_generators()
    slack_gen = next(iter(_default_distributed_slack(net, gen)))
    params = reference_parameters(
        provider={"slackBusSelectionMode": "NAME", "slackBusesIds": gen.loc[slack_gen, "bus_id"],
                  "outerLoopNames": "ReactiveLimits",
                  "newtonRaphsonConvEpsPerEq": "1e-9", "maxNewtonRaphsonIterations": "50"},
        distributed_slack=False, use_reactive_limits=True, read_slack_bus=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id=slack_gen, sort_index=False, buses_for_sub=False,
                                   keep_half_open_lines=True, fuse_zero_impedance_branches=True,
                                   olf_rules=OlfLoadingParameters(reactive_limits=True), olf_parameters=params)
    grid.change_algorithm(OUTER)
    return grid


@pytest.mark.parametrize("load_scale", [1., 1.01, 1.03])
def test_ieee300_reactive_limits(load_scale):
    """Many PV -> PQ switches over several outer iterations, releases, moved limits.
    At 1.03 the second inner solve diverges, on both sides."""
    base = _ieee300()
    flat = np.full(base.total_bus(), 1.0 + 0j)
    V0 = np.exp(1j * np.angle(base.dc_pf(flat, 30, 1e-12)))
    assert base.ac_pf(V0, 30, 1e-9).size > 0
    h = _cpp._OuterModelHarness(base, keep_trace=True)
    h.begin(1, 1)
    row = _ieee300(load_scale)
    V_ref = row.ac_pf(np.asarray(base.get_V(), dtype=complex), 30, 1e-9)
    stats = row.get_algo().get_outer_loop_stats()
    ref = int(np.asarray(h.slack)[0])
    V, status = _run_row(h, 0, row.get_Ybus_solver().toarray(), np.asarray(row.get_Sbus_solver()), ref,
                         tol_mva=1e-9, sn_mva=row.get_sn_mva())
    res = h.row_result(0)
    ls_trace = [str(d) for d in stats.decisions]
    msg = "\n".join(res.decisions) + "\n---- lightsim2grid\n" + "\n".join(ls_trace)
    assert stats.nb_outer_iterations > 0
    assert res.nb_outer_iterations == stats.nb_outer_iterations, msg
    assert len(res.decisions) == len(ls_trace), msg
    assert res.nb_nr_iterations == sum(max(n, 1) for n in stats.nr_iterations), msg
    if V_ref.size == 0:
        assert status == MAX_ITER, msg
        return
    assert status == CONVERGED, msg
    buses = np.asarray(row.id_ac_solver_to_me(), dtype=int)
    np.testing.assert_allclose(V, V_ref[buses], rtol=0., atol=1e-7, err_msg=msg)


# --------------------------------------------------------------------------------------
# The full inner system: VoltageControl (lightsim2grid's held-controller formulation)
# and the droop hvdc lines, Jacobian by central differences
# --------------------------------------------------------------------------------------
def _hvdc_recv(p, side1, lf1, lf2, r):
    lf_c, lf_r = (lf1, lf2) if side1 else (lf2, lf1)
    line_in = (1. - lf_c) * p
    return (1. - lf_r) * (line_in - r * line_in * line_in)


def _hvdc_flows(st, raw, lf1, lf2, r, pmax12, pmax21):
    if st == 0:
        if raw >= 0.:
            return raw, -_hvdc_recv(raw, True, lf1, lf2, r)
        return -_hvdc_recv(-raw, False, lf1, lf2, r), -raw
    if st > 0:
        return pmax12, -_hvdc_recv(pmax12, True, lf1, lf2, r)
    return -_hvdc_recv(pmax21, False, lf1, lf2, r), pmax21


class _FullSystem:
    def __init__(self, h, Ybus):
        self.L = h.ledger
        self.Y = Ybus
        n = Ybus.shape[0]
        self.n = n
        self.ref = int(np.asarray(h.slack)[0])
        self.pin_bus = np.asarray(h.pin_bus, dtype=int)
        L = self.L
        self.p_bus = np.flatnonzero(np.asarray(L["p_row_of_bus"]) >= 0)
        self.th_bus = np.flatnonzero(np.asarray(L["theta_col_of_bus"]) >= 0)
        self.vm_col = np.asarray(L["vm_col_of_bus"]) >= 0
        self.nc = len(L["vc_bus"])
        self.q_plan = np.zeros(self.nc)
        if len(L["vc_q_held"]) == self.nc:
            self.q_plan = np.asarray(L["vc_q_held"], dtype=float)
        # BranchControl: the transformers whose block follows their shift / ratio, the ones
        # solving for their shift (a column, a row)
        self.nb = len(L["bc_trafo"])
        self.bc_col = np.flatnonzero(np.asarray(L["bc_shift_col"], dtype=int) >= 0) if self.nb else np.zeros(0, int)
        self.bc_rcol = np.flatnonzero(np.asarray(L["bc_ratio_col"], dtype=int) >= 0) if self.nb else np.zeros(0, int)
        yeff = np.asarray(L["bc_yeff0"], dtype=float).reshape(-1, 4, 2) if self.nb else np.zeros((0, 4, 2))
        self.bc_yeff = yeff[:, :, 0] + 1j * yeff[:, :, 1]
        # the slot's BranchControl state (BranchControl::Entry), set by controls()
        self.bc_a = np.zeros(self.nb)
        self.bc_rho = np.ones(self.nb)
        self.bc_a_target = np.zeros(self.nb)
        self.bc_rho_target = np.ones(self.nb)
        self.bc_on = np.zeros(self.nb, dtype=bool)
        self.bc_ron = np.zeros(self.nb, dtype=bool)
        self.bc_out = np.zeros(self.nb, dtype=bool)
        self.bc_a_tap = np.zeros(self.nb)
        self.bc_rho_tap = np.ones(self.nb)
        self.bc_tap = np.zeros((self.nb, 4), dtype=complex)

    def apply_bc_controls(self, c, first):
        """NROuterInner::apply_controls on BranchControl: re-taps, then the controls"""
        if self.nb == 0:
            return
        tap = np.asarray(c["bc_tap"], dtype=float).reshape(-1, 4, 2)
        self.bc_out = np.asarray(c["bc_out"], dtype=bool)
        if first:   # BranchControl::_reset: the grid's taps
            self.bc_a_tap = np.asarray(c["bc_a_tap"], dtype=float).copy()
            self.bc_rho_tap = np.asarray(c["bc_rho_tap"], dtype=float).copy()
            self.bc_tap = tap[:, :, 0] + 1j * tap[:, :, 1]
            self.bc_a, self.bc_rho = self.bc_a_tap.copy(), self.bc_rho_tap.copy()
            self.bc_a_target = self.bc_a.copy()
            self.bc_rho_target = self.bc_rho.copy()
            self.bc_on[:] = False
            self.bc_ron[:] = False
        for e in c["bc_retap"]:
            self.bc_a_tap[e] = c["bc_a_tap"][e]
            self.bc_rho_tap[e] = c["bc_rho_tap"][e]
            self.bc_tap[e] = tap[e, :, 0] + 1j * tap[e, :, 1]
            self.bc_a[e], self.bc_rho[e] = self.bc_a_tap[e], self.bc_rho_tap[e]
            self.bc_a_target[e] = self.bc_a[e]
            self.bc_rho_target[e] = self.bc_rho[e]
        for e, req in enumerate(c["bc_ctl"]):
            if self.bc_out[e] or req < 0:
                continue
            self.bc_on[e] = req == 1
            if req == 0:
                self.bc_a_target[e] = self.bc_a[e]
        for e, req in enumerate(c["bc_rctl"]):
            if self.bc_out[e] or req < 0 or e not in self.bc_rcol:
                continue
            self.bc_ron[e] = req == 1
            if req == 0:
                self.bc_rho_target[e] = self.bc_rho[e]

    def block(self, e, a, rho):
        """BranchControl::_block_at"""
        if self.bc_out[e]:
            return np.zeros(4, dtype=complex)
        d = self.L["bc_sign"][e] * (a - self.bc_a_tap[e])
        rot = np.exp(1j * d)
        f = (rho / self.bc_rho_tap[e]) ** self.L["bc_rexp"][e]
        t = self.bc_tap[e]
        return np.array([t[0] * f * f, t[1] * f * rot, t[2] * f * np.conj(rot), t[3]])

    def ybus(self, a, rho=None):
        """the slot's Ybus at shifts `a` and ratios `rho`: the row's (the grid's blocks) patched
        by value"""
        rho = self.bc_rho if rho is None else rho
        Y = self.Y.astype(complex).copy()
        for e in range(self.nb):
            if self.bc_out[e]:
                continue
            b1, b2 = self.L["bc_bus1"][e], self.L["bc_bus2"][e]
            d = self.block(e, a[e], rho[e]) - self.bc_yeff[e]
            Y[b1, b1] += d[0]
            Y[b1, b2] += d[1]
            Y[b2, b1] += d[2]
            Y[b2, b2] += d[3]
        return Y

    def residual(self, V, q, Sbus, c, vm_bus, a=None, rho=None):
        L = self.L
        a = self.bc_a if a is None else a
        rho = self.bc_rho if rho is None else rho
        Y = self.ybus(a, rho) if self.nb else self.Y
        S = V * np.conj(Y @ V)
        mis = S - Sbus
        Va = np.angle(V)
        for e in range(len(L["hvdc_bus1"])):
            b1, b2 = L["hvdc_bus1"][e], L["hvdc_bus2"][e]
            raw = L["hvdc_p0"][e] + L["hvdc_k"][e] * (Va[b1] - Va[b2])
            p1, p2 = _hvdc_flows(c["hvdc_status"][e], raw, L["hvdc_lf1"][e], L["hvdc_lf2"][e],
                                 L["hvdc_r"][e], L["hvdc_pmax12"][e], L["hvdc_pmax21"][e])
            mis[b1] += p1
            mis[b2] += p2
        held_q = np.asarray(c["held_q"], dtype=float)
        plan_held = np.asarray(c["plan_held"], dtype=bool)
        for j in range(self.nc):
            mis[L["vc_bus"][j]] -= 1j * (q[j] - (self.q_plan[j] if plan_held[j] else 0.))
        rows = [mis.real[self.p_bus], mis.imag[vm_bus]]
        held = plan_held | np.isfinite(held_q)
        hq = np.where(np.isfinite(held_q), held_q, self.q_plan)
        for g in range(len(L["vc_reg_bus"])):
            first, cnt = L["vc_grp_start"][g], L["vc_grp_count"][g]
            ref = next((off for off in range(cnt) if not held[first + off]), -1)
            Q = lambda off: q[first + off]
            if ref < 0:
                rows.append([Q(0) - hq[first]])
            else:
                slope = sum(L["vc_slope"][first + off] * Q(off) for off in range(cnt))
                rows.append([abs(V[L["vc_reg_bus"][g]]) + slope - c["v_set"][g]])
            for m in range(1, cnt):
                j = first + m
                if ref < 0 or held[j]:
                    rows.append([Q(m) - hq[j]])
                elif m == ref:
                    rows.append([Q(0) - hq[first]])
                else:
                    rows.append([L["vc_weight"][first + ref] * Q(m) - L["vc_weight"][j] * Q(ref)])
        # BranchControl's shift rows: the shift held, or the regulated side's P on target
        for e in self.bc_col:
            if not self.bc_on[e]:
                rows.append([a[e] - self.bc_a_target[e]])
                continue
            y = self.block(e, a[e], rho[e])
            V1, V2 = V[L["bc_bus1"][e]], V[L["bc_bus2"][e]]
            p = (V1 * np.conj(y[0] * V1 + y[1] * V2)).real if L["bc_side"][e] == 1 else \
                (V2 * np.conj(y[2] * V1 + y[3] * V2)).real
            rows.append([p - L["bc_p_target"][e]])
        # the ratio groups' rows: per member its ratio held, the first one on holds the bus'
        # voltage, any other one on shares the mean ratio
        for g, members in enumerate(L["bc_grp_members"]):
            mean = np.mean([rho[m] for m in members])
            first_on = True
            for m in members:
                if not self.bc_ron[m]:
                    rows.append([rho[m] - self.bc_rho_target[m]])
                elif first_on:
                    first_on = False
                    rows.append([abs(V[L["bc_grp_bus"][g]]) - L["bc_grp_target"][g]])
                else:
                    rows.append([mean - rho[m]])
        return np.concatenate([np.asarray(r, dtype=float) for r in rows])

    def solve(self, V, q, Sbus, c, tol, max_it):
        vm_unknown = self.vm_col.copy()
        vm_unknown[self.pin_bus[np.asarray(c["pin"], dtype=bool)]] = False
        vm_bus = np.flatnonzero(vm_unknown)
        nt, nv = self.th_bus.size, vm_bus.size

        nq, na = len(q), self.bc_col.size

        def unpack(x):
            Va, Vm = np.angle(V0).copy(), np.abs(V0).copy()
            Va[self.th_bus] = x[:nt]
            Vm[vm_bus] = x[nt:nt + nv]
            a = self.bc_a.copy()
            a[self.bc_col] = x[nt + nv + nq:nt + nv + nq + na]
            rho = self.bc_rho.copy()
            rho[self.bc_rcol] = x[nt + nv + nq + na:]
            return Vm * np.exp(1j * Va), x[nt + nv:nt + nv + nq], a, rho

        V0 = V.copy()
        x = np.concatenate([np.angle(V)[self.th_bus], np.abs(V)[vm_bus], q, self.bc_a[self.bc_col],
                            self.bc_rho[self.bc_rcol]])
        for it in range(max_it + 1):
            Vx, qx, ax, rx = unpack(x)
            F = self.residual(Vx, qx, Sbus, c, vm_bus, ax, rx)
            if np.max(np.abs(F)) < tol or it == max_it:
                self.bc_a, self.bc_rho = ax, rx
                return Vx, qx, np.max(np.abs(F)) < tol, it, vm_unknown
            J = np.empty((F.size, x.size))
            for k in range(x.size):
                d = np.zeros(x.size)
                d[k] = 1e-7
                Vp, qp, ap, rp = unpack(x + d)
                Vm_, qm, am, rm = unpack(x - d)
                J[:, k] = (self.residual(Vp, qp, Sbus, c, vm_bus, ap, rp) -
                           self.residual(Vm_, qm, Sbus, c, vm_bus, am, rm)) / 2e-7
            x = x - np.linalg.solve(J, F)
        raise AssertionError("unreachable")


def _run_row_full(h, row, Ybus, Sbus_row, tol_pu):
    sysm = _FullSystem(h, Ybus)
    watch = np.asarray(h.watch_bus, dtype=int)
    realistic = np.asarray(h.realistic_bus, dtype=int)
    V = np.asarray(h.V0).copy()
    h.load(0, row)
    q = np.where(np.asarray(h.controls(0)["plan_held"], dtype=bool), sysm.q_plan, 0.)
    first = True
    for _ in range(100):
        c = h.controls(0)
        Sbus = Sbus_row.copy()
        for b, dp, dq in zip(c["sbus_bus"], c["sbus_dp"], c["sbus_dq"]):
            Sbus[b] += dp + 1j * dq
        for b, vm in zip(c["vm_reset_bus"], c["vm_reset_vm"]):
            V[b] = vm * V[b] / abs(V[b])
        sysm.apply_bc_controls(c, first)
        first = False
        h.clear_vm_reset(0)
        V, q, conv, nr, vm_unknown = sysm.solve(V, q, Sbus, c, tol_pu, MAX_IT)
        S = V * np.conj((sysm.ybus(sysm.bc_a, sysm.bc_rho) if sysm.nb else Ybus) @ V)
        cand = [b for b in realistic if vm_unknown[b]]
        unrealistic = bool(np.any((np.abs(V[cand]) < 0.5) | (np.abs(V[cand]) > 2.))) if cand else False
        nxt, status = h.on_solved(0, conv, nr, CONVERGED if conv else MAX_ITER,
                                  list(V[watch]), list(S[watch]), list(Sbus[watch]), list(q), unrealistic,
                                  list(sysm.bc_a), list(sysm.bc_rho))
        if nxt == DONE:
            return V, status
    raise AssertionError("the row never ended")


def _compare_single_solve(make_grid, tol_mva=TOL_MVA, start=None, branch_off=None, **row_kwargs):
    """the model on make_grid()'s rows (its inner solves in numpy) against
    lightsim2grid's single NROuter solve of the same grid, from the same start;
    `branch_off`: the branches the row opens (lines then trafos), told to the model
    (make_grid(**row_kwargs) opens them on the reference)"""
    base = make_grid()
    v0 = np.full(base.total_bus(), 1.0 + 0j) if start is None else start(base)
    assert base.ac_pf(v0, MAX_IT, tol_mva).size > 0
    h = _cpp._OuterModelHarness(base, keep_trace=True)
    if branch_off is not None:
        h.set_branch_off([list(branch_off)])
    h.begin(1, 1)
    row = make_grid(**row_kwargs)
    V_ref = row.ac_pf(np.asarray(base.get_V(), dtype=complex), MAX_IT, tol_mva)
    stats = row.get_algo().get_outer_loop_stats()
    V, status = _run_row_full(h, 0, row.get_Ybus_solver().toarray(), np.asarray(row.get_Sbus_solver()),
                              tol_mva / row.get_sn_mva())
    res = h.row_result(0)
    ls_trace = [str(d) for d in stats.decisions]
    msg = "\n".join(res.decisions) + "\n---- lightsim2grid\n" + "\n".join(ls_trace)
    assert res.nb_outer_iterations == stats.nb_outer_iterations, msg
    assert int(res.status) == int(stats.status), msg
    assert len(res.decisions) == len(ls_trace), msg
    assert (status == CONVERGED) == (V_ref.size > 0), msg
    if V_ref.size > 0:
        buses = np.asarray(row.id_ac_solver_to_me(), dtype=int)
        np.testing.assert_allclose(V, V_ref[buses], rtol=0., atol=1e-8, err_msg=msg)
    return res, stats


def _ls2g_tests_dir():
    import os
    import sys
    import lightsim2grid
    d = os.path.join(os.path.dirname(lightsim2grid.__file__), "tests")
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def _case14_hvdc(**kwargs):
    _ls2g_tests_dir()
    from _aux_make_hvdc import make_case14_hvdc
    params = dict(loss_factor1=0.011, loss_factor2=0.011, droop_enabled=True, p0=10.,
                  droop_mw_per_deg=5., pmax12=15., pmax21=300.)
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _, model = make_case14_hvdc(3, 9, **params)
    model.change_algorithm(OUTER)
    return model


@pytest.mark.parametrize("kwargs", [dict(), dict(p0=-60., pmax12=300., pmax21=20.), dict(pmax12=300.)])
def test_hvdc_ac_emulation_limits(kwargs):
    """lightsim2grid's own cases: the droop line saturates 1 -> 2, 2 -> 1, or stays linear"""
    res, stats = _compare_single_solve(lambda **kw: _case14_hvdc(**{**kwargs, **kw}))
    saturates = kwargs.get("pmax12", 15.) < 300. or kwargs.get("pmax21", 300.) < 300.
    assert (len(res.hvdc_regime) > 0) == saturates
    if saturates:
        assert any("SATURATE" in d for d in res.decisions)


def _ieee14_standby_svc(low_pu, high_pu, b0=0.):
    pp = pytest.importorskip("pypowsybl")
    _ls2g_tests_dir()
    from _olf_reference import reference_parameters
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl, OlfLoadingParameters
    net = pp.network.create_ieee14()
    vn = float(net.get_voltage_levels().loc["VL9", "nominal_v"])
    net.create_static_var_compensators(id="SVC9", voltage_level_id="VL9", bus_id="B9",
                                       connectable_bus_id="B9", b_min=-0.5, b_max=0.5,
                                       regulation_mode="VOLTAGE", target_v=vn, target_q=0.,
                                       regulating=True)
    net.create_extensions("standbyAutomaton", id="SVC9", b0=b0, standby=True,
                          low_voltage_threshold=low_pu * vn, high_voltage_threshold=high_pu * vn,
                          low_voltage_setpoint=0.98 * vn, high_voltage_setpoint=1.02 * vn)
    params = reference_parameters(
        provider={"slackBusSelectionMode": "NAME",
                  "slackBusesIds": net.get_generators().loc["B1-G", "bus_id"],
                  "svcVoltageMonitoring": "true",
                  "newtonRaphsonConvEpsPerEq": "1e-12", "maxNewtonRaphsonIterations": "50"},
        distributed_slack=True, use_reactive_limits=True, read_slack_bus=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id="B1-G", sort_index=False, buses_for_sub=False,
                                   olf_rules=OlfLoadingParameters(reactive_limits=True),
                                   olf_parameters=params)
    grid.change_algorithm(OUTER)
    return grid


@pytest.mark.parametrize("low_pu, high_pu", [(0.90, 1.00), (1.10, 1.20), (0.90, 1.20)])
def test_voltage_monitoring(low_pu, high_pu):
    """a standby SVC switched on above its high threshold, below its low one, or left
    idle -- with the default loops around it (DistributedSlack, ReactiveLimits)"""
    res, stats = _compare_single_solve(lambda: _ieee14_standby_svc(low_pu, high_pu), tol_mva=1e-10)
    switched = any("SWITCH_ON" in d and "not taken" not in d for d in res.decisions)
    assert switched == ((low_pu, high_pu) != (0.90, 1.20))


def _ieee14_remote_group(qmax, regulators=("B6-G", "B8-G"), scale=1., qmin=None):
    """pypowsybl's IEEE14, the generators `regulators` all holding load bus B9 at
    1.05 pu (one voltage-control group), with reactive limits `qmax` (MVAr, one per
    regulator): ReactiveLimits freezes the group's controllers at their limits"""
    pp = pytest.importorskip("pypowsybl")
    _ls2g_tests_dir()
    from _olf_reference import reference_parameters
    from lightsim2grid.network.from_pypowsybl import init as init_from_pypowsybl, OlfLoadingParameters
    net = pp.network.create_ieee14()
    vn9 = float(net.get_voltage_levels().loc["VL9", "nominal_v"])
    ids = list(regulators)
    net.update_generators(id=ids, regulated_element_id=["B9-L"] * len(ids), target_v=[1.05 * vn9] * len(ids))
    qmin = [-q for q in qmax] if qmin is None else list(qmin)
    net.update_generators(id=ids, min_q=qmin, max_q=list(qmax))
    if scale != 1.:
        loads = net.get_loads()
        net.update_loads(id=loads.index, p0=loads["p0"] * scale, q0=loads["q0"] * scale)
    params = reference_parameters(
        provider={"slackBusSelectionMode": "NAME",
                  "slackBusesIds": net.get_generators().loc["B1-G", "bus_id"],
                  "newtonRaphsonConvEpsPerEq": "1e-12", "maxNewtonRaphsonIterations": "50"},
        distributed_slack=True, use_reactive_limits=True, read_slack_bus=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pypowsybl(net, gen_slack_id="B1-G", sort_index=False, buses_for_sub=False,
                                   olf_rules=OlfLoadingParameters(reactive_limits=True),
                                   olf_parameters=params)
    grid.change_algorithm(OUTER)
    return grid


@pytest.mark.parametrize("regulators, qmax", [
    (("B6-G", "B8-G"), (100., 100.)),          # nothing frozen
    (("B6-G", "B8-G"), (10., 10.)),            # the whole group frozen
    (("B6-G", "B8-G"), (3., 100.)),            # its first controller only
    (("B6-G", "B8-G"), (100., 3.)),            # its second one only
    (("B3-G", "B6-G", "B8-G"), (10., 3., 100.)),   # sharing against the last one
    (("B3-G", "B6-G", "B8-G"), (100., 3., 3.)),
])
def test_reactive_limits_on_a_voltage_control_group(regulators, qmax):
    res, stats = _compare_single_solve(lambda **kw: _ieee14_remote_group(qmax, regulators, **kw),
                                       tol_mva=1e-10, scale=1.05)


@pytest.mark.parametrize("regulators, qmin, qmax", [
    # the same range (the same share), one of them close to its max: it alone freezes,
    # and the group regulates on with the others
    (("B6-G", "B8-G"), (-100., -3.), (3., 100.)),
    (("B6-G", "B8-G"), (-3., -100.), (100., 3.)),
    (("B3-G", "B6-G", "B8-G"), (-100., -100., -3.), (100., 3., 100.)),
    (("B3-G", "B6-G", "B8-G"), (-40., -100., -3.), (40., 3., 100.)),
    # the group's first controller held, the sharing taken against its second one
    (("B3-G", "B6-G", "B8-G"), (-100., -3., -3.), (3., 100., 100.)),
])
def test_reactive_limits_holds_part_of_a_group(regulators, qmin, qmax):
    res, stats = _compare_single_solve(
        lambda **kw: _ieee14_remote_group(qmax, regulators, qmin=qmin, **kw), tol_mva=1e-10, scale=1.05)
    assert any("PV_TO_PQ" in d for d in res.decisions), res.decisions


# --------------------------------------------------------------------------------------
# init_from_n_powerflow: every row resumes from where the "n" solve left its loops
# --------------------------------------------------------------------------------------
@pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_resume", False),
                    reason="lightsim2grid without its loops' state accessors")
@pytest.mark.parametrize("q_limit", [15., 30.])
def test_rows_resume_from_the_n_solve(q_limit):
    """lightsim2grid's ScenarioSweepCPP with init_from_n_powerflow: the rows start from the
    "n" solve's voltages and loops (its frozen buses frozen again at each row's own limit);
    every row sets its generators' targets (modify_gen_p), so the targets DistributedSlack
    moved in "n" are not taken"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    n_rows = 6
    base = _case14(q_limit=q_limit)
    v0 = np.full(base.total_bus(), 1.04 + 0j)
    assert base.ac_pf(v0, MAX_IT, TOL_MVA).size > 0
    assert base.get_algo().get_outer_loop_stats().nb_outer_iterations > 0
    gen_p = np.array([g.target_p_mw for g in base.get_generators()])
    load_p = np.asarray(base.get_load_target_p())
    scales = 1. + 0.04 * np.arange(n_rows)
    batch = ScenarioSweepCPP(base)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = True
    batch.modify_gen_p(np.outer(np.ones(n_rows), gen_p))
    batch.modify_load_p(np.outer(scales, load_p))
    mask = np.zeros((n_rows, len(base.get_lines())), dtype=bool)
    for i in range(n_rows):
        mask[i, i + 1] = True
    batch.set_contingency_lines(mask)
    batch.compute(v0, MAX_IT, TOL_MVA)
    V_ls = np.asarray(batch.get_voltages())
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res_ls = batch.get_outer_results()

    h = _cpp._OuterModelHarness(base, keep_trace=True, resume_from_n=True)
    h.begin(1, n_rows)
    h.set_row_targets(n_rows, list(np.outer(np.ones(n_rows), gen_p).ravel()))
    ref = int(np.asarray(h.slack)[0])
    buses = np.asarray(base.id_ac_solver_to_me(), dtype=int)
    resumed = 0
    for i in range(n_rows):
        row = _case14(q_limit=q_limit, load_scale=scales[i], line_off=i + 1)
        row.ac_pf(v0, MAX_IT, TOL_MVA)   # its Ybus and Sbus
        V, status = _run_row_full(h, i, row.get_Ybus_solver().toarray(), np.asarray(row.get_Sbus_solver()),
                                  TOL_MVA / row.get_sn_mva())
        res = h.row_result(i)
        msg = f"row {i}:\n" + "\n".join(res.decisions)
        assert res.nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        assert int(res.status) == int(res_ls[i].status), msg
        assert (status == CONVERGED) == bool(conv[i]), msg
        if conv[i]:
            np.testing.assert_allclose(V, V_ls[i][buses], rtol=0., atol=1e-8, err_msg=msg)
        # a bus the "n" solve froze is checked for its release before any switch
        first_rl = next((d for d in res.decisions if d.startswith("ReactiveLimits")), "")
        resumed += "KEPT_PQ" in first_rl or "PQ_TO_PV" in first_rl
    assert resumed > 0


# --------------------------------------------------------------------------------------
# PhaseControl: a phase shifter regulating its active power (its shift a Newton unknown,
# then rounded to the closest tap), or limiting its current (one tap at a time); its pi
# block patched by value at every Newton iteration (BranchControl)
# --------------------------------------------------------------------------------------
def _pst_module():
    pytest.importorskip("pypowsybl")
    _ls2g_tests_dir()
    import test_phase_control as P
    return P


def _pst_grid(mode, target, load_scale=1., line_off=None, pst_off=False, regulating=True):
    """lightsim2grid's PhaseControl test grid (IEEE 14, a phase tap changer on T5-6-1, -10 ..
    10 degrees) with its default loops and PhaseControl"""
    P = _pst_module()
    from lightsim2grid.algorithm import PhaseControl
    grid = P._grid(P._net(mode, target, regulating=regulating))
    grid.add_outer_loop(PhaseControl())
    grid.change_algorithm(OUTER)
    if load_scale != 1.:
        for load_id, p in enumerate(np.asarray(grid.get_load_target_p())):
            grid.change_p_load(load_id, float(p) * load_scale)
    if line_off is not None:
        grid.deactivate_powerline(line_off)
    if pst_off:
        grid.deactivate_trafo(_pst_id(grid))
    return grid


def _pst_id(grid):
    return next(i for i, t in enumerate(grid.get_trafos()) if t.name == _pst_module().PST)


def _pst_flow():
    """the shifter's flow and current with no control (MW, kA)"""
    grid = _pst_grid("ACTIVE_POWER_CONTROL", 0., regulating=False)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), MAX_IT, 1e-10).size > 0
    t = grid.get_trafos()[_pst_id(grid)]
    return t.res_p1_mw, t.res_a1_ka * 1000.


def _check_pst(mode, target, branch_off=None, **row_kwargs):
    res, stats = _compare_single_solve(lambda **kw: _pst_grid(mode, target, **kw), tol_mva=1e-10,
                                       branch_off=branch_off, **row_kwargs)
    row = _pst_grid(mode, target, **row_kwargs)
    row.ac_pf(np.full(row.total_bus(), 1. + 0j), MAX_IT, 1e-10)
    # the taps it settled on: those of lightsim2grid's single solve of the row's grid
    ref = [(i, t.res_phase_tap_position) for i, t in enumerate(row.get_trafos())
           if t.res_phase_tap_position != t.phase_tap_position]
    assert sorted(res.phase_tap) == sorted(ref), (res.phase_tap, ref)
    return res, stats


@pytest.mark.parametrize("delta_mw", [7., -12.3, 30.])
def test_phase_control_active_power(delta_mw):
    p1, _ = _pst_flow()
    res, _ = _check_pst("ACTIVE_POWER_CONTROL", p1 + delta_mw)
    assert res.phase_tap   # rounded somewhere off the grid's tap


@pytest.mark.parametrize("factor, moves", [(0.8, True), (0.95, True), (1.1, False)])
def test_phase_control_current_limiter(factor, moves):
    _, i1 = _pst_flow()
    res, _ = _check_pst("CURRENT_LIMITER", factor * i1)
    assert bool(res.phase_tap) == moves


@pytest.mark.parametrize("row_kwargs", [dict(load_scale=1.05), dict(line_off=2)])
def test_phase_control_rows(row_kwargs):
    p1, _ = _pst_flow()
    branch_off = [row_kwargs["line_off"]] if "line_off" in row_kwargs else None
    _check_pst("ACTIVE_POWER_CONTROL", p1 + 7., branch_off=branch_off, **row_kwargs)


def test_phase_control_row_opening_the_shifter():
    # the row has no such transformer: nothing of it in Ybus, no tap for it
    p1, _ = _pst_flow()
    grid = _pst_grid("ACTIVE_POWER_CONTROL", p1 + 7.)
    branch_off = [len(grid.get_lines()) + _pst_id(grid)]
    res, _ = _check_pst("ACTIVE_POWER_CONTROL", p1 + 7., branch_off=branch_off, pst_off=True)
    assert not res.phase_tap


@pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_resume", False),
                    reason="lightsim2grid without its loops' state accessors")
@pytest.mark.parametrize("mode", ["ACTIVE_POWER_CONTROL", "CURRENT_LIMITER"])
def test_phase_control_rows_resume_from_the_n_solve(mode):
    """lightsim2grid's ContingencyAnalysisCPP with init_from_n_powerflow: each row starts from
    the tap the "n" solve settled on (OuterLoopAlgo::_apply_carried_positions), PhaseControl
    itself restarting (its controllers on again); the row opening the shifter has none"""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    p1, i1 = _pst_flow()
    target = p1 + 7. if mode == "ACTIVE_POWER_CONTROL" else 0.8 * i1
    base = _pst_grid(mode, target)
    v0 = np.full(base.total_bus(), 1. + 0j)
    assert base.ac_pf(v0, MAX_IT, 1e-10).size > 0
    assert base.get_trafos()[_pst_id(base)].res_phase_tap_position != 0   # a tap to carry
    n_line = len(base.get_lines())
    branches = [0, 2, 3, n_line + _pst_id(base)]    # sorted: the batch's row order
    batch = ContingencyAnalysisCPP(base)
    for b in branches:
        batch.add_n1(b)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = True
    batch.compute(v0, MAX_IT, 1e-10)
    V_ls = np.asarray(batch.get_voltages())
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    res_ls = batch.get_outer_results()

    h = _cpp._OuterModelHarness(base, keep_trace=True, resume_from_n=True)
    h.set_branch_off([[b] for b in branches])
    h.begin(1, len(branches))
    buses = np.asarray(base.id_ac_solver_to_me(), dtype=int)
    for i, b in enumerate(branches):
        row = _pst_grid(mode, target, line_off=b if b < n_line else None, pst_off=b >= n_line)
        row.ac_pf(v0, MAX_IT, 1e-10)   # its Ybus and Sbus
        V, status = _run_row_full(h, i, row.get_Ybus_solver().toarray(), np.asarray(row.get_Sbus_solver()),
                                  1e-10 / row.get_sn_mva())
        res = h.row_result(i)
        msg = f"row {i} (branch {b}):\n" + "\n".join(res.decisions)
        assert res.nb_outer_iterations == res_ls[i].nb_outer_iterations, msg
        assert int(res.status) == int(res_ls[i].status), msg
        assert sorted(res.phase_tap) == sorted(res_ls[i].phase_tap), msg
        assert (status == CONVERGED) == bool(conv[i]), msg
        if conv[i]:
            np.testing.assert_allclose(V, V_ls[i][buses], rtol=0., atol=1e-8, err_msg=msg)


# --------------------------------------------------------------------------------------
# TransformerVoltageControl: a ratio tap changer regulating a bus' voltage (its ratio a
# Newton unknown, then rounded), the low voltage generators frozen meanwhile; lightsim2grid's
# own test grid (IEEE 14, a tap changer on a transformer, 0.9 .. 1.1 in 21 steps)
# --------------------------------------------------------------------------------------
def _tvc_grid(trafo, target_kv, deadband_kv, reactive_limits=False, line_off=None, trafo_off=False,
              load_scale=1., **loop_kw):
    pytest.importorskip("pypowsybl")
    _ls2g_tests_dir()
    import test_transformer_voltage_control as T
    from lightsim2grid.algorithm import ReactiveLimits, TransformerVoltageControl
    grid = T._grid(T._net(trafo, target_kv, deadband_kv), reactive_limits=reactive_limits)
    grid.clear_outer_loops()
    if reactive_limits:
        grid.add_outer_loop(ReactiveLimits())
    grid.add_outer_loop(TransformerVoltageControl(**loop_kw))
    grid.change_algorithm(OUTER)
    if load_scale != 1.:
        for load_id, p in enumerate(np.asarray(grid.get_load_target_p())):
            grid.change_p_load(load_id, float(p) * load_scale)
    if line_off is not None:
        grid.deactivate_powerline(line_off)
    if trafo_off:
        grid.deactivate_trafo(next(i for i, t in enumerate(grid.get_trafos()) if t.name == trafo))
    return grid


def _check_tvc(trafo, target_kv, deadband_kv, branch_off=None, row_kwargs=None, **grid_kw):
    row_kwargs = row_kwargs or {}
    make = lambda **kw: _tvc_grid(trafo, target_kv, deadband_kv, **grid_kw, **kw)
    res, stats = _compare_single_solve(make, tol_mva=1e-10, branch_off=branch_off, **row_kwargs)
    base = make()
    base.ac_pf(np.full(base.total_bus(), 1. + 0j), MAX_IT, 1e-10)
    row = make(**row_kwargs)
    row.ac_pf(np.asarray(base.get_V(), dtype=complex), MAX_IT, 1e-10)
    ref = [(i, t.res_ratio_tap_position) for i, t in enumerate(row.get_trafos())
           if t.res_ratio_tap_position != t.ratio_tap_position]
    assert sorted(res.ratio_tap) == sorted(ref), (res.ratio_tap, ref, res.decisions)
    return res


@pytest.mark.parametrize("target_kv, deadband_kv", [(12.6, 0.1), (12.4, 0.05), (12.5, 0.02), (12.0, 0.1)])
def test_transformer_voltage_control(target_kv, deadband_kv):
    # switched on, solved, rounded (at 12.6 kV back onto the grid's own tap)
    res = _check_tvc("T4-9-1", target_kv, deadband_kv)
    assert any("ROUND_TAP" in d for d in res.decisions), res.decisions


def test_transformer_voltage_control_with_reactive_limits():
    _check_tvc("T4-9-1", 12.3, 0.05, reactive_limits=True)


def test_transformer_voltage_control_hidden_by_a_generator():
    res = _check_tvc("T5-6-1", 12.6, 0.1)
    assert not res.ratio_tap


@pytest.mark.parametrize("loop_kw", [dict(use_initial_tap_position=True, max_controlled_nominal_voltage=120.),
                                     dict(max_controlled_nominal_voltage=120.)])
def test_transformer_voltage_control_frozen_generators(loop_kw):
    # lightsim2grid's _tap_grid: ReactiveLimits too, the generators of up to 120 kV frozen
    _check_tvc("T4-9-1", 12.6, 0.1, reactive_limits=True, **loop_kw)


@pytest.mark.parametrize("row_kwargs, branch_off", [(dict(load_scale=1.05), None), (dict(line_off=1), [1])])
def test_transformer_voltage_control_rows(row_kwargs, branch_off):
    _check_tvc("T4-9-1", 12.6, 0.1, branch_off=branch_off, row_kwargs=row_kwargs, reactive_limits=True,
               use_initial_tap_position=True, max_controlled_nominal_voltage=120.)


def test_transformer_voltage_control_row_opening_the_transformer():
    grid = _tvc_grid("T4-9-1", 12.6, 0.1)
    t = next(i for i, tr in enumerate(grid.get_trafos()) if tr.name == "T4-9-1")
    res = _check_tvc("T4-9-1", 12.6, 0.1, branch_off=[len(grid.get_lines()) + t], row_kwargs=dict(trafo_off=True))
    assert not res.ratio_tap
