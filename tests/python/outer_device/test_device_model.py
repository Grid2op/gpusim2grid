# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The device outer-loop model (``outer_loops_backend='device'``) against the host one, in lockstep.

Both models are driven through the test-only ``_OuterModelHarness`` (``backend='host'`` /
``'device'``) on the same rows: the inner solves are the numpy full-system Newton of
test_outer_model_cpu.py, run on the host model's controls, and every converged state is handed to
both. After every call the two must agree: the same next step and row status, the same controls
(pins, held controllers, set-points, regimes, Sbus correction, |V| resets, slack weights), and at the
end the same row result. That is the parity bar of docs/outer_loops_on_device.md (the same
decisions as the host model, which test_outer_model_cpu.py pins against lightsim2grid)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import test_outer_model_cpu as cpu
from _outer_device_helpers import needs_fp64, needs_outer_loops

import gpusim2grid._gpusim2grid as _cpp

from conftest import parity_tol, requires_gpu

pytestmark = [needs_outer_loops, needs_fp64, requires_gpu]

#: controls compared to rounding: sums of the same terms, maybe in another order


def _sparse(bus: list[int], *vals: list[float]) -> dict[int, tuple[float, ...]]:
    return {int(b): tuple(float(v[k]) for v in vals) for k, b in enumerate(bus)}


def _assert_same_controls(host: dict[str, Any], dev: dict[str, Any], where: str) -> None:
    for key in (
        "pin",
        "plan_held",
        "hvdc_status",
        "bc_out",
        "bc_retap",
        "bc_ctl",
        "bc_rctl",
        "sc_set",
        "sc_ctl",
    ):
        assert list(dev[key]) == list(host[key]), f"{where}: controls[{key!r}]"
    for key in ("held_q", "v_set", "slack_w", "bc_tap", "bc_a_tap", "bc_rho_tap", "sc_b_new"):
        h, d = np.asarray(host[key], dtype=float), np.asarray(dev[key], dtype=float)
        assert h.shape == d.shape, f"{where}: controls[{key!r}] shape"
        np.testing.assert_allclose(
            d, h, rtol=0, atol=parity_tol("lockstep"), equal_nan=True, err_msg=f"{where}: controls[{key!r}]"
        )
    for bus, vals in (("sbus_bus", ("sbus_dp", "sbus_dq")), ("vm_reset_bus", ("vm_reset_vm",))):
        h = _sparse(host[bus], *(host[v] for v in vals))
        d = _sparse(dev[bus], *(dev[v] for v in vals))
        assert sorted(d) == sorted(h), f"{where}: controls[{bus!r}]"
        for b, hv in h.items():
            np.testing.assert_allclose(
                d[b], hv, rtol=0, atol=parity_tol("lockstep"), err_msg=f"{where}: controls[{bus!r}] at {b}"
            )


def _assert_same_result(host: Any, dev: Any, where: str) -> None:
    for key in (
        "status",
        "unrealistic_state",
        "failed_loop",
        "nb_outer_iterations",
        "nb_nr_iterations",
        "hvdc_regime",
    ):
        assert getattr(dev, key) == getattr(host, key), f"{where}: row_result.{key}"
    # the decision trace (keep_outer_trace), line by line
    assert list(dev.decisions) == list(host.decisions), f"{where}\ndevice:\n" + "\n".join(dev.decisions)
    assert sorted(dev.shunt_sections) == sorted(host.shunt_sections), f"{where}: row_result.shunt_sections"
    assert sorted(dev.phase_tap) == sorted(host.phase_tap), f"{where}: row_result.phase_tap"
    assert sorted(dev.ratio_tap) == sorted(host.ratio_tap), f"{where}: row_result.ratio_tap"
    h, d = dict(host.gen_target_p), dict(dev.gen_target_p)
    assert sorted(d) == sorted(h), f"{where}: row_result.gen_target_p ids"
    mw_tol = parity_tol("lockstep_mw")
    for g, mw in h.items():
        assert d[g] == pytest.approx(mw, rel=0, abs=mw_tol), f"{where}: row_result.gen_target_p[{g}]"


def _lockstep(
    make_grid: Callable[..., Any],
    tol_mva: float = cpu.TOL_MVA,
    resume: bool = False,
    overrides: dict[str, Any] | None = None,
    **row_kwargs: Any,
) -> Any:
    """Every outer iteration of one row, host and device side by side (see the module docstring);
    returns the host's row result. ``overrides``: the row's RowOverrides (``set_row_overrides``
    keywords), what the row's grid (``make_grid(**row_kwargs)``) changes for the loops."""
    base = make_grid()
    assert base.ac_pf(np.full(base.total_bus(), 1.0 + 0j), cpu.MAX_IT, tol_mva).size > 0
    host = _cpp._OuterModelHarness(base, keep_trace=True, resume_from_n=resume)
    dev = _cpp._OuterModelHarness(base, keep_trace=True, resume_from_n=resume, backend="device")
    for h in (host, dev):
        if overrides:
            h.set_row_overrides(**overrides)
        h.begin(1, 1)
    row = make_grid(**row_kwargs)
    # lightsim2grid's own solve of the row builds its solver data (Ybus, Sbus) as the host test does
    row.ac_pf(np.asarray(base.get_V(), dtype=complex), cpu.MAX_IT, tol_mva)
    ybus = row.get_Ybus_solver().toarray()
    sbus_row = np.asarray(row.get_Sbus_solver())
    tol_pu = tol_mva / row.get_sn_mva()
    sysm = cpu._FullSystem(host, ybus)
    watch = np.asarray(host.watch_bus, dtype=int)
    assert list(dev.watch_bus) == list(watch)
    realistic = np.asarray(host.realistic_bus, dtype=int)
    V = np.asarray(host.V0).copy()
    for h in (host, dev):
        h.load(0, 0)
    q = np.where(np.asarray(host.controls(0)["plan_held"], dtype=bool), sysm.q_plan, 0.0)
    first = True
    for it in range(100):
        c = host.controls(0)
        where = f"outer iteration {it}\n" + "\n".join(host.row_result(0).decisions)
        _assert_same_controls(c, dev.controls(0), where)
        sbus = sbus_row.copy()
        for b, dp, dq in zip(c["sbus_bus"], c["sbus_dp"], c["sbus_dq"], strict=True):
            sbus[b] += dp + 1j * dq
        for b, vm in zip(c["vm_reset_bus"], c["vm_reset_vm"], strict=True):
            V[b] = vm * V[b] / abs(V[b])
        sysm.apply_bc_controls(c, first)
        sysm.apply_sc_controls(c, first)
        first = False
        for h in (host, dev):
            h.clear_vm_reset(0)
        V, q, conv, nr, vm_unknown = sysm.solve(V, q, sbus, c, tol_pu, cpu.MAX_IT)
        S = V * np.conj(ybus @ V)
        cand = [b for b in realistic if vm_unknown[b]]
        unrealistic = bool(np.any((np.abs(V[cand]) < 0.5) | (np.abs(V[cand]) > 2.0))) if cand else False
        args = (
            0,
            conv,
            nr,
            cpu.CONVERGED if conv else cpu.MAX_ITER,
            list(V[watch]),
            list(S[watch]),
            list(sbus[watch]),
            list(q),
            unrealistic,
            list(sysm.bc_a),
            list(sysm.bc_rho),
            list(sysm.sc_b),
        )
        nxt_h = host.on_solved(*args, slack_absorbed=float(sysm.sa))
        nxt_d = dev.on_solved(*args, slack_absorbed=float(sysm.sa))
        assert nxt_d == nxt_h, f"{where}\n(next, status): device {nxt_d}, host {nxt_h}"
        if nxt_h[0] == cpu.DONE:
            res = host.row_result(0)
            _assert_same_result(res, dev.row_result(0), "\n".join(res.decisions))
            return res
    raise AssertionError("the row never ended")


def _case14_two_regulated_busbars(move: bool = False, load_scale: float = 1.0) -> Any:
    """case14 with a second generator in generator 1's substation, on its second busbar (with one
    line end), regulating the same voltage: both busbars are controller buses of the grid's
    ReactiveLimits plan. ``move``: that generator moved onto busbar 1 (a re-merge: the bus generator
    1 holds gains a generator). Only generator 0 and the slack share the slack."""
    import warnings

    import pandapower as pp
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    net = pn.case14()
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    net.load["p_mw"] *= load_scale
    n_sub = len(net.bus)
    for b in range(n_sub):
        pp.create_bus(net, vn_kv=net.bus.vn_kv.iloc[b], in_service=False)
    sub = int(net.gen.bus.iloc[1])
    net.bus.loc[n_sub + sub, "in_service"] = True
    vm_pu = float(net.gen.vm_pu.iloc[1])
    pp.create_gen(net, n_sub + sub, p_mw=10.0, vm_pu=vm_pu, min_q_mvar=-5.0, max_q_mvar=5.0)
    line = int(net.line.index[(net.line.from_bus == sub) | (net.line.to_bus == sub)][0])
    end = "from_bus" if int(net.line.from_bus.loc[line]) == sub else "to_bus"
    net.line.loc[line, end] = n_sub + sub
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        g = init_from_pandapower(net, n_sub=n_sub, n_busbar_per_sub=2)
    n_gen = len(g.get_generators())
    g.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    participates = [gen.id == 0 or gen.is_slack for gen in g.get_generators()]
    g.set_gen_can_participate_slack(participates, np.full(n_gen, 300.0))
    if move:
        g.change_bus_gen(_merged_gen(g), sub)
    g.change_algorithm(cpu.OUTER)
    g.tell_solver_need_reset()
    return g


def _merged_gen(g: Any) -> int:
    """the generator standing on a second busbar"""
    return next(int(x.id) for x in g.get_generators() if int(x.bus_id) >= g.get_n_sub())


@pytest.mark.parametrize("load_scale", [1.0, 1.15])
def test_generator_moved_onto_a_regulated_bus(load_scale: float):
    """a row moving a generator onto a bus the grid's ReactiveLimits plan already has (generator 1's,
    a re-merge; the facades refuse it, the model takes it as RowOverrides::gen_bus): its plan bus has
    one more generator than the grid's. The device tables carry it, off on the rows not moving it
    there, instead of refusing the row at its load"""
    base = _case14_two_regulated_busbars()
    assert base.ac_pf(np.full(base.total_bus(), 1.0 + 0j), cpu.MAX_IT, cpu.TOL_MVA).size > 0
    gen = _merged_gen(base)
    sub = int(base.get_generators()[gen].bus_id) - base.get_n_sub()
    to_solver = np.asarray(base.id_me_to_ac_solver(), dtype=int)
    _lockstep(
        _case14_two_regulated_busbars,
        overrides=dict(gen_bus=[[(gen, int(to_solver[sub]))]]),
        move=True,
        load_scale=load_scale,
    )


# ---- DistributedSlack and ReactiveLimits on local buses ----------------------------------------


def test_ample_limits_nothing_acts():
    res = _lockstep(lambda **kw: cpu._case14(q_limit=1e4, distributed=False, **kw), load_scale=1.1)
    assert res.nb_outer_iterations == 0


@pytest.mark.parametrize("load_scale", [1.0, 1.1, 1.3])
def test_case14_short_of_reactive_power(load_scale: float):
    """DistributedSlack shares what the row changes, ReactiveLimits switches local buses PQ"""
    _lockstep(lambda **kw: cpu._case14(**kw), load_scale=load_scale)


@pytest.mark.parametrize("load_scale", [1.0, 1.05])
def test_ieee300_reactive_limits(load_scale: float):
    _lockstep(lambda **kw: cpu._ieee300(**kw), load_scale=load_scale)


# ---- the other loops and the group buses -------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{}, {"p0": -60.0, "pmax12": 300.0, "pmax21": 20.0}])
def test_hvdc_ac_emulation_limits(kwargs: dict[str, float]):
    _lockstep(lambda **kw: cpu._case14_hvdc(**{**kwargs, **kw}))


@pytest.mark.parametrize("low_pu, high_pu", [(0.90, 1.00), (1.10, 1.20)])
def test_voltage_monitoring(low_pu: float, high_pu: float):
    _lockstep(lambda: cpu._ieee14_standby_svc(low_pu, high_pu), tol_mva=1e-10)


@pytest.mark.parametrize(
    "regulators, qmax", [(("B6-G", "B8-G"), (100.0, 3.0)), (("B3-G", "B6-G", "B8-G"), (100.0, 3.0, 3.0))]
)
def test_reactive_limits_on_a_voltage_control_group(regulators: tuple[str, ...], qmax: tuple[float, ...]):
    _lockstep(lambda **kw: cpu._ieee14_remote_group(qmax, regulators, **kw), tol_mva=1e-10, scale=1.05)


@pytest.mark.skipif(
    not getattr(_cpp, "have_ls2g_outer_resume", False), reason="lightsim2grid without its loops' state"
)
@pytest.mark.parametrize("load_scale", [1.0, 1.1])
def test_rows_resume_from_the_n_solve(load_scale: float):
    _lockstep(lambda **kw: cpu._case14(**kw), resume=True, load_scale=load_scale)


@pytest.mark.parametrize(
    "regulators, qmin, qmax",
    [
        (("B6-G", "B8-G"), (-100.0, -3.0), (3.0, 100.0)),
        (("B3-G", "B6-G", "B8-G"), (-40.0, -100.0, -3.0), (40.0, 3.0, 100.0)),
        (("B3-G", "B6-G", "B8-G"), (-100.0, -3.0, -3.0), (3.0, 100.0, 100.0)),
    ],
)
def test_reactive_limits_holds_part_of_a_group(
    regulators: tuple[str, ...], qmin: tuple[float, ...], qmax: tuple[float, ...]
):
    """a group's controllers frozen one by one (held by value), the group regulating on with the others"""
    res = _lockstep(
        lambda **kw: cpu._ieee14_remote_group(qmax, regulators, qmin=qmin, **kw), tol_mva=1e-10, scale=1.05
    )
    assert any("PV_TO_PQ" in d for d in res.decisions), res.decisions


# ---- a row whose contingency masks a voltage-controlled bus (handle_disconnected_grid) ----------


@pytest.mark.parametrize("load_scale", [1.0, 1.2, 1.3])
def test_row_masking_a_voltage_controlled_bus(load_scale: float):
    """test_outer_model_cpu.py's masked row (trafo 6-7 open, bus 7 and its generator stranded), in
    lockstep through its simple Newton: the row's ReactiveLimits plan is the grid's without the masked bus"""
    trafo, gen, bus = 3, 3, 7
    extra_load = (bus, 0.0, 30.0)
    base = cpu._case14(distributed=False, extra_load=extra_load)
    assert base.ac_pf(np.full(base.total_bus(), 1.04 + 0j), cpu.MAX_IT, cpu.TOL_MVA).size > 0
    to_base = np.asarray(base.id_me_to_ac_solver(), dtype=int)
    masked = int(to_base[bus])
    host = _cpp._OuterModelHarness(base, keep_trace=True)
    dev = _cpp._OuterModelHarness(base, keep_trace=True, backend="device")
    for h in (host, dev):
        h.set_row_overrides([[len(base.get_lines()) + trafo]], masked=[[masked]])
        h.begin(1, 1)
    row = cpu._case14(distributed=False, load_scale=load_scale, extra_load=extra_load)
    row.deactivate_trafo(trafo)
    row.deactivate_gen(gen)
    row.deactivate_load(len(row.get_loads()) - 1)
    row.deactivate_bus(bus)
    assert row.ac_pf(np.asarray(base.get_V(), dtype=complex), cpu.MAX_IT, cpu.TOL_MVA).size > 0
    row_to_me = np.asarray(row.id_ac_solver_to_me(), dtype=int)
    idx = to_base[row_to_me]
    n = len(np.asarray(base.id_ac_solver_to_me()))
    ybus = np.zeros((n, n), dtype=complex)
    ybus[np.ix_(idx, idx)] = row.get_Ybus_solver().toarray()
    sbus_row = np.zeros(n, dtype=complex)
    sbus_row[idx] = np.asarray(row.get_Sbus_solver())
    sbus_row[masked] = np.asarray(base.get_Sbus_solver())[masked]
    ref = int(np.asarray(host.slack)[0])
    pin_bus = np.asarray(host.pin_bus, dtype=int)
    watch = np.asarray(host.watch_bus, dtype=int)
    pq = np.asarray(host.pq, dtype=int)
    realistic = np.asarray(host.realistic_bus, dtype=int)
    V = np.asarray(host.V0).copy()
    for h in (host, dev):
        h.load(0, 0)
    for it in range(100):
        c = host.controls(0)
        where = f"outer iteration {it}\n" + "\n".join(host.row_result(0).decisions)
        _assert_same_controls(c, dev.controls(0), where)
        sbus = sbus_row.copy()
        for b, dp, dq in zip(c["sbus_bus"], c["sbus_dp"], c["sbus_dq"], strict=True):
            sbus[b] += dp + 1j * dq
        for b, vm in zip(c["vm_reset_bus"], c["vm_reset_vm"], strict=True):
            V[b] = vm * V[b] / abs(V[b])
        for h in (host, dev):
            h.clear_vm_reset(0)
        vm_unknown = np.zeros(n, dtype=bool)
        vm_unknown[pq] = True
        pin = np.asarray(c["pin"], dtype=bool)
        vm_unknown[pin_bus[~pin]] = True
        vm_unknown[pin_bus[pin]] = False
        V, conv, nr = cpu._newton(
            ybus, V, sbus, ref, vm_unknown, cpu.TOL_MVA / 100.0, cpu.MAX_IT, skip=(masked,)
        )
        S = V * np.conj(ybus @ V)
        cand = [b for b in realistic if vm_unknown[b] and b != masked]
        unrealistic = bool(np.any((np.abs(V[cand]) < 0.5) | (np.abs(V[cand]) > 2.0))) if cand else False
        args = (
            0,
            conv,
            nr,
            cpu.CONVERGED if conv else cpu.MAX_ITER,
            list(V[watch]),
            list(S[watch]),
            list(sbus[watch]),
            [],
            unrealistic,
        )
        nxt_h, nxt_d = host.on_solved(*args), dev.on_solved(*args)
        assert nxt_d == nxt_h, f"{where}\n(next, status): device {nxt_d}, host {nxt_h}"
        if nxt_h[0] == cpu.DONE:
            res = host.row_result(0)
            _assert_same_result(res, dev.row_result(0), "\n".join(res.decisions))
            return
    raise AssertionError("the row never ended")


# ---- a distributed-slack Newton (NROuterDist_*) --------------------------------------------------


@pytest.mark.parametrize("load_scale, line_off", [(1.0, None), (1.1, None), (1.1, 3), (1.3, None)])
def test_dist_newton(load_scale: float, line_off: int | None):
    """the Newton shares the slack on the units by the weights each solve sets (controls' slack_w);
    DistributedSlack parks the unit its share pushes past its limit"""
    _lockstep(lambda **kw: cpu._case14_dist(**kw), load_scale=load_scale, line_off=line_off)


@pytest.mark.parametrize("fail_on_residue", [True, False])
def test_dist_every_unit_at_its_limit(fail_on_residue: bool):
    def make(**kw: Any) -> Any:
        n_gen = len(cpu._case14_dist(capped=False, only_ds=True).get_generators())
        return cpu._case14_dist(
            capped=False, only_ds=True, max_p=np.full(n_gen, 45.0), fail_on_residue=fail_on_residue, **kw
        )

    _lockstep(make, load_scale=1.2)


def test_dist_only_the_loop():
    _lockstep(lambda **kw: cpu._case14_dist(only_ds=True, **kw), load_scale=1.1)


@pytest.mark.skipif(
    not getattr(_cpp, "have_ls2g_outer_resume", False), reason="lightsim2grid without its loops' state"
)
@pytest.mark.parametrize("load_scale", [1.03, 1.09])
def test_dist_rows_resume_with_the_parked_units(load_scale: float):
    _lockstep(lambda **kw: cpu._case14_dist(**kw), resume=True, load_scale=load_scale)


# ---- ShuntVoltageControl (ShuntControl) -----------------------------------------------------------


@pytest.mark.parametrize(
    "bus, target_kv, sections",
    [("B9", 12.8, (3, 1)), ("B9", 13.2, (3, 1)), ("B9", 12.4, (3, 1)), ("B10", 12.8, (5, 0))],
)
def test_shunt_voltage_control(bus: str, target_kv: float, sections: tuple[int, int]):
    """the regulating shunts of a bus one susceptance unknown in the first solve, then dispatched to
    sections (ShuntControl's controls compared every iteration)"""
    res = _lockstep(lambda **kw: cpu._shunt_grid(bus, target_kv, sections, **kw), tol_mva=1e-10)
    assert any("ROUND_SECTIONS" in d for d in res.decisions), res.decisions


def test_shunt_voltage_control_hidden_by_a_generator():
    res = _lockstep(lambda **kw: cpu._shunt_grid("B6", 12.8, (0, 2), **kw), tol_mva=1e-10)
    assert not res.decisions


@pytest.mark.parametrize("load_scale", [1.05, 1.1])
def test_shunt_voltage_control_rows(load_scale: float):
    _lockstep(lambda **kw: cpu._shunt_grid("B9", 13.2, (3, 1), **kw), tol_mva=1e-10, load_scale=load_scale)


# ---- PhaseControl (BranchControl) ---------------------------------------------------------------


@pytest.mark.parametrize("delta_mw", [7.0, -12.3, 30.0])
def test_phase_control_active_power(delta_mw: float):
    """a phase shifter regulating its active power: its shift a Newton unknown, then rounded to the
    closest tap (BranchControl's controls compared every iteration)"""
    p1, _ = cpu._pst_flow()
    res = _lockstep(lambda **kw: cpu._pst_grid("ACTIVE_POWER_CONTROL", p1 + delta_mw, **kw), tol_mva=1e-10)
    assert res.phase_tap


@pytest.mark.parametrize("factor", [0.8, 0.95, 1.1])
def test_phase_control_current_limiter(factor: float):
    """a current limiter one tap at a time, the way that lowers its current"""
    _, i1 = cpu._pst_flow()
    _lockstep(lambda **kw: cpu._pst_grid("CURRENT_LIMITER", factor * i1, **kw), tol_mva=1e-10)


@pytest.mark.parametrize("load_scale", [1.05, 1.1])
def test_phase_control_rows(load_scale: float):
    p1, _ = cpu._pst_flow()
    _lockstep(
        lambda **kw: cpu._pst_grid("ACTIVE_POWER_CONTROL", p1 + 7.0, **kw),
        tol_mva=1e-10,
        load_scale=load_scale,
    )


# ---- TransformerVoltageControl (BranchControl's ratio columns) -------------------------------------


@pytest.mark.parametrize("target_kv, deadband_kv", [(12.6, 0.1), (12.4, 0.05), (12.5, 0.02), (12.0, 0.1)])
def test_transformer_voltage_control(target_kv: float, deadband_kv: float):
    """a ratio tap changer regulating a bus: switched on, its ratio solved, rounded to a tap"""
    res = _lockstep(lambda **kw: cpu._tvc_grid("T4-9-1", target_kv, deadband_kv, **kw), tol_mva=1e-10)
    assert any("ROUND_TAP" in d for d in res.decisions), res.decisions


def test_transformer_voltage_control_with_reactive_limits():
    _lockstep(lambda **kw: cpu._tvc_grid("T4-9-1", 12.3, 0.05, reactive_limits=True, **kw), tol_mva=1e-10)


def test_transformer_voltage_control_hidden_by_a_generator():
    _lockstep(lambda **kw: cpu._tvc_grid("T5-6-1", 12.6, 0.1, **kw), tol_mva=1e-10)


@pytest.mark.parametrize(
    "loop_kw",
    [
        dict(use_initial_tap_position=True, max_controlled_nominal_voltage=120.0),
        dict(max_controlled_nominal_voltage=120.0),
    ],
)
def test_transformer_voltage_control_frozen_generators(loop_kw: dict[str, Any]):
    """the generators of up to 120 kV frozen while the transformer acts (suspended for ReactiveLimits)"""
    _lockstep(
        lambda **kw: cpu._tvc_grid("T4-9-1", 12.6, 0.1, reactive_limits=True, **loop_kw, **kw), tol_mva=1e-10
    )


@pytest.mark.parametrize("load_scale", [1.05, 1.1])
def test_transformer_voltage_control_rows(load_scale: float):
    _lockstep(
        lambda **kw: cpu._tvc_grid(
            "T4-9-1",
            12.6,
            0.1,
            reactive_limits=True,
            use_initial_tap_position=True,
            max_controlled_nominal_voltage=120.0,
            **kw,
        ),
        tol_mva=1e-10,
        load_scale=load_scale,
    )


def test_shunt_voltage_control_hidden_by_a_transformer():
    _lockstep(lambda **kw: cpu._shunt_grid("B9", 13.2, (0, 2), with_trafo_control=True, **kw), tol_mva=1e-10)
