# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``reactive_limits_outer_loop`` -- one pass of OpenLoadFlow's ``ReactiveLimits``
outer loop on the rows the physical checks flagged.

The first pass is the batch as usual, with ``compute_physical_violations``. The
rows it reports a reactive-limit switch on are then re-solved ONCE, in a second,
smaller batch (a :class:`ScenarioSweepGPU` built from the same grid, one row per
flagged contingency), with the switches applied:

- ``LOW_Q`` / ``HIGH_Q`` on a bus its own machines hold (a Vm-fixed bus outside
  any VoltageControl group, no SVC on it): the bus turns PQ, its machines
  producing their summed limit -- the Vm column and Q equation it needs are
  reserved once for the whole second batch (``add_switchable_vm_buses``, what
  ``set_contingency_gens`` uses), every other row keeping the bus PV;
- ``LOW_Q`` / ``HIGH_Q`` on EVERY controller bus of a VoltageControl group
  made of generators only (typically a generator behind its step-up
  transformer regulating the HV bus), all in the same direction: each
  controller is held at its own limit (``set_vc_controller_pins``: the group's
  voltage row and sharing rows are rewritten by value) and the regulated bus
  floats;
- ``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q`` on a generator frozen at
  a limit that regulates its own bus, a PQ bus outside any group: the bus is
  held PV again at the generator's target (its Q equation pinned, |V| set), the
  frozen Q leaving the injection;
- the same records on a frozen generator regulating a REMOTE bus: the second
  pass is built from a copy of the grid with lightsim2grid's
  ``set_hold_frozen_regulators`` on, where such a machine already sits in its
  voltage-control group, held at its frozen output (every row pins it, the
  frozen Q staying in the injection); the row releases it
  (``set_vc_controller_releases``) and it takes part in its group again.

Anything else a row reports (a group only part of which saturated, a group
holding an SVC / hvdc station, a remote regulator lightsim2grid does not hold
or one that would leave its group's first controller held, an SVC, a truncated
report) makes the row UNSUPPORTED: it is not re-solved and keeps its first-pass
result. The other physical checks (slack
active power, hvdc saturation, standby SVCs) neither trigger nor block a
re-solve; they are simply reported again by the second pass.

The second pass reports, with the new labels: a switched bus is no longer
checked for LOW_Q / HIGH_Q on that row, and gets a release check instead (its
machines at min_q with the bus below target, or at max_q above it, would
regulate again: ``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q`` on one of
its generators, against the voltage of the bus it regulates -- the remote one
for a group); a released generator's own release check is dropped and the
bus it holds again is checked for LOW_Q / HIGH_Q against its range (a released
held machine against its own range: its frozen output, still in the injection,
is offset out of the check and put back into the records). A row that
still reports something after that pass needs another outer iteration (on the
CPU); nothing is iterated here.
"""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ._limit_violations import LimitViolationType, ViolationElementType

__all__ = ["ReactiveLimitsStatus"]

_LOW_Q = int(LimitViolationType.LOW_Q)
_HIGH_Q = int(LimitViolationType.HIGH_Q)
_REL_LOW = int(LimitViolationType.LOW_VOLTAGE_AT_MIN_Q)
_REL_HIGH = int(LimitViolationType.HIGH_VOLTAGE_AT_MAX_Q)
_EL_GEN = int(ViolationElementType.GENERATOR)


class ReactiveLimitsStatus(IntEnum):
    """What ``reactive_limits_outer_loop`` did with a row (see
    :meth:`ContingencyAnalysisGPU.get_outer_loop_status`)."""
    #: nothing to switch (no LOW_Q / HIGH_Q / release record, or the row was not
    #: simulated / did not converge): the first pass is the result
    NO_SWITCH = 0
    #: re-solved with its switches: its voltages, residual and violations are the
    #: second pass'
    RECOMPUTED = 1
    #: a switch this loop does not handle (a VoltageControl group only part of
    #: which saturated or holding an SVC / station, the release of a remote
    #: regulator, an SVC, a truncated report): first pass kept, needs the CPU
    UNSUPPORTED = 2
    #: supported, but left out by the batch rule (a last, partial chunk smaller
    #: than ``outer_loop_min_last_chunk``): first pass kept, needs the CPU
    LEFT_OUT = 3
    #: re-solved, but the second pass did not converge: first pass kept, needs
    #: the CPU
    DIVERGED = 4


@dataclass
class _Context:
    """What the switch decisions and the second pass need of the grid, read
    once (solver bus numbering)."""
    n_bus: int
    is_vm_fixed: np.ndarray        # (n_bus,) bool
    vc_group: np.ndarray           # (n_bus,) int, -1 = none
    q_row: np.ndarray              # (n_bus,) int, unextended ledger
    vm_col: np.ndarray             # (n_bus,) int
    bus_vn_kv: np.ndarray          # (n_bus,) float
    gen_bus: np.ndarray            # (n_gen,) int, -1 = disconnected
    gen_reg_bus: np.ndarray        # (n_gen,) int, -1 = none
    gen_vreg_on: np.ndarray        # (n_gen,) bool
    gen_target_q: np.ndarray       # (n_gen,) MVAr
    gen_target_vm: np.ndarray      # (n_gen,) pu
    gen_min_q: np.ndarray          # (n_gen,) MVAr
    gen_max_q: np.ndarray          # (n_gen,) MVAr
    base_load_p: np.ndarray
    base_load_q: np.ndarray
    base_gen_p: np.ndarray
    # the grid's own plans, as arrays (the second pass appends to copies)
    bq: dict = field(default_factory=dict)
    gr: dict = field(default_factory=dict)
    bq_entry_of_bus: dict = field(default_factory=dict)   # bus -> entry of the reactive plan
    gr_entry_of_gen: dict = field(default_factory=dict)   # gen -> GENERATOR entry of the release plan
    # VoltageControl groups (ledger order): per controller its bus / kind /
    # group / element id, per group its controllers, regulated bus and v_set
    vc_ctrl_bus: np.ndarray = None
    vc_ctrl_kind: np.ndarray = None
    vc_ctrl_group: np.ndarray = None
    vc_ctrl_elem: np.ndarray = None
    vc_ctrl_held: np.ndarray = None                       # lightsim2grid's held controllers
    vc_grp_ctrls: list = field(default_factory=list)      # group -> ACTIVE controller indices
    vc_grp_first: list = field(default_factory=list)      # group -> its first controller (held or not)
    vc_grp_buses: list = field(default_factory=list)      # group -> set of active controller buses
    vc_grp_reg_bus: np.ndarray = None
    vc_grp_vset: np.ndarray = None
    vc_ctrls_at_bus: dict = field(default_factory=dict)   # bus -> active controller indices
    vc_held_of_gen: dict = field(default_factory=dict)    # gen -> its held controller


def _plan_arrays_bq(plan):
    return {
        "bus_solver": np.asarray(plan.bus_solver, dtype=np.int32).copy(),
        "qmin_fixed": np.asarray(plan.qmin_fixed_mvar, dtype=np.float64).copy(),
        "qmax_fixed": np.asarray(plan.qmax_fixed_mvar, dtype=np.float64).copy(),
        "n_fixed": np.asarray(plan.n_fixed, dtype=np.int32).copy(),
        "bmin": np.asarray(plan.bmin_sum_pu, dtype=np.float64).copy(),
        "bmax": np.asarray(plan.bmax_sum_pu, dtype=np.float64).copy(),
        "gen_start": np.asarray(plan.gen_start, dtype=np.int32).copy(),
        "gen_id": np.asarray(plan.gen_id, dtype=np.int32).copy(),
        "gen_qmin": np.asarray(plan.gen_qmin_mvar, dtype=np.float64).copy(),
        "gen_qmax": np.asarray(plan.gen_qmax_mvar, dtype=np.float64).copy(),
        "sn_mva": float(plan.sn_mva),
    }


def _plan_arrays_gr(plan):
    n = int(plan.n_entries)
    el_type = np.asarray(plan.el_type, dtype=np.int32)
    standby = np.asarray(plan.standby, dtype=np.int32)
    side = np.asarray(plan.side, dtype=np.int32)
    return {
        "gen_id": np.asarray(plan.gen_id, dtype=np.int32).copy(),
        "reg_bus": np.asarray(plan.reg_bus_solver, dtype=np.int32).copy(),
        "gen_bus": np.asarray(plan.gen_bus_solver, dtype=np.int32).copy(),
        "at_min": np.asarray(plan.at_min, dtype=np.int32).copy(),
        "target_vm_pu": np.asarray(plan.target_vm_pu, dtype=np.float64).copy(),
        "vn_kv": np.asarray(plan.vn_kv, dtype=np.float64).copy(),
        "el_type": el_type.copy() if el_type.size else np.full(n, _EL_GEN, dtype=np.int32),
        "standby": standby.copy() if standby.size else np.zeros(n, dtype=np.int32),
        "side": side.copy() if side.size else np.zeros(n, dtype=np.int32),
    }


def build_context(grid, sweep):
    """Read, off the lightsim2grid grid and the second-pass ScenarioSweepGPU
    ``sweep`` (built from it, physical checks on), everything the switch
    decisions need. Called once, before the second pass ever reserved a bus
    (the ledger maps are the unextended ones)."""
    eng = sweep.solver
    sess = eng._s
    n_bus = int(eng.n_bus)
    me2s = np.asarray(grid.id_me_to_ac_solver())
    s2me = np.asarray(grid.id_ac_solver_to_me())
    vn_model = np.asarray(grid.get_bus_vn_kv(), dtype=np.float64)
    bus_vn_kv = vn_model[s2me[:n_bus]]

    def to_solver(model_bus):
        return int(me2s[model_bus]) if 0 <= model_bus < me2s.size else -1

    gens = grid.get_generators()
    el = sweep._elements
    gen_bus = np.asarray(el.gen_bus, dtype=np.int64)
    gen_reg_bus = np.array([to_solver(int(g.regulated_bus_id)) if g.connected else -1 for g in gens],
                           dtype=np.int64)
    loads = grid.get_loads()
    ctx = _Context(
        n_bus=n_bus,
        is_vm_fixed=np.asarray(eng.is_vm_fixed_bus, dtype=bool),
        vc_group=np.asarray(eng.vc_group_of_bus, dtype=np.int64),
        q_row=np.asarray(eng.q_row_of_bus, dtype=np.int64),
        vm_col=np.asarray(eng.vm_col_of_bus, dtype=np.int64),
        bus_vn_kv=bus_vn_kv,
        gen_bus=gen_bus,
        gen_reg_bus=gen_reg_bus,
        gen_vreg_on=np.asarray(el.gen_vreg_on, dtype=bool),
        gen_target_q=np.asarray(el.gen_target_q_mvar, dtype=np.float64),
        gen_target_vm=np.array([float(g.target_vm_pu) for g in gens], dtype=np.float64),
        gen_min_q=np.array([float(g.min_q_mvar) for g in gens], dtype=np.float64),
        gen_max_q=np.array([float(g.max_q_mvar) for g in gens], dtype=np.float64),
        base_load_p=np.array([float(l.target_p_mw) for l in loads], dtype=np.float64),
        base_load_q=np.array([float(l.target_q_mvar) for l in loads], dtype=np.float64),
        base_gen_p=np.array([float(g.target_p_mw) for g in gens], dtype=np.float64),
    )
    ctx.bq = _plan_arrays_bq(sess.physical_checks.bus_q_plan)
    ctx.gr = _plan_arrays_gr(sess.physical_checks.gen_pv_release_plan)
    ctx.bq_entry_of_bus = {int(b): k for k, b in enumerate(ctx.bq["bus_solver"])}
    # the releases only: a remote voltage control entry (standby 2) names a generator too
    ctx.gr_entry_of_gen = {int(g): k for k, (g, e, c) in enumerate(zip(ctx.gr["gen_id"], ctx.gr["el_type"],
                                                                       ctx.gr["standby"]))
                           if int(e) == _EL_GEN and int(c) == 0}
    ctx.vc_ctrl_bus = np.asarray(sess.vc_ctrl_bus, dtype=np.int64)
    ctx.vc_ctrl_kind = np.asarray(sess.vc_ctrl_kind, dtype=np.int64)
    ctx.vc_ctrl_group = np.asarray(sess.vc_ctrl_group, dtype=np.int64)
    ctx.vc_ctrl_elem = np.asarray(sess.vc_ctrl_elem_id, dtype=np.int64)
    held = np.asarray(getattr(sess, "vc_ctrl_held", []), dtype=bool)
    ctx.vc_ctrl_held = held if held.size == ctx.vc_ctrl_bus.size else np.zeros(ctx.vc_ctrl_bus.size, dtype=bool)
    start = np.asarray(sess.vc_grp_start, dtype=np.int64)
    count = np.asarray(sess.vc_grp_count, dtype=np.int64)
    ctx.vc_grp_ctrls = [[j for j in range(int(s), int(s + c)) if not ctx.vc_ctrl_held[j]]
                        for s, c in zip(start, count)]
    ctx.vc_grp_first = [int(s) for s in start]
    ctx.vc_grp_buses = [{int(ctx.vc_ctrl_bus[j]) for j in js} for js in ctx.vc_grp_ctrls]
    ctx.vc_grp_reg_bus = np.asarray(sess.vc_reg_bus, dtype=np.int64)
    ctx.vc_grp_vset = np.asarray(sess.vc_v_set, dtype=np.float64)
    for j, b in enumerate(ctx.vc_ctrl_bus):
        if ctx.vc_ctrl_held[j]:
            if ctx.vc_ctrl_kind[j] == 0:
                ctx.vc_held_of_gen[int(ctx.vc_ctrl_elem[j])] = j
        else:
            ctx.vc_ctrls_at_bus.setdefault(int(b), []).append(j)
    return ctx


def _bus_switchable(ctx, b):
    """A bus whose LOW_Q / HIGH_Q this loop can act on: held by its own
    machines (Vm-fixed, no VoltageControl group), at least one generator among
    them, no SVC susceptance."""
    if b < 0 or b >= ctx.n_bus or not ctx.is_vm_fixed[b] or ctx.vc_group[b] >= 0:
        return False
    k = ctx.bq_entry_of_bus.get(b)
    if k is None:
        return False
    if ctx.bq["bmin"][k] != 0. or ctx.bq["bmax"][k] != 0.:
        return False
    return ctx.bq["gen_start"][k + 1] > ctx.bq["gen_start"][k]


def _bus_group(ctx, b):
    """The VoltageControl group whose controllers alone hold bus ``b`` (every
    generator of its reactive-plan entry a GEN controller of that one group, no
    station / storage unit / SVC at the bus, the group itself made of
    generators only), else -1."""
    ctrls = ctx.vc_ctrls_at_bus.get(b)
    k = ctx.bq_entry_of_bus.get(b)
    if not ctrls or k is None:
        return -1
    if ctx.bq["n_fixed"][k] != 0 or ctx.bq["bmin"][k] != 0. or ctx.bq["bmax"][k] != 0.:
        return -1
    groups = {int(ctx.vc_ctrl_group[j]) for j in ctrls}
    if len(groups) != 1 or any(ctx.vc_ctrl_kind[j] != 0 for j in ctrls):
        return -1
    g = groups.pop()
    if any(ctx.vc_ctrl_kind[j] != 0 for j in ctx.vc_grp_ctrls[g]) or not ctx.vc_grp_ctrls[g]:
        return -1
    gens = {int(x) for x in ctx.bq["gen_id"][ctx.bq["gen_start"][k]:ctx.bq["gen_start"][k + 1]]}
    if gens != {int(ctx.vc_ctrl_elem[j]) for j in ctrls}:
        return -1
    return g


def _gen_releasable(ctx, g):
    """A generator whose release this loop can act on: frozen PQ at a limit
    (its target Q in the injection), regulating its own bus, itself a PQ bus
    outside any VoltageControl group and hosting no group controller."""
    if g < 0 or g >= ctx.gen_bus.size or ctx.gr_entry_of_gen.get(g) is None:
        return False
    b = int(ctx.gen_bus[g])
    if b < 0 or ctx.gen_vreg_on[g] or int(ctx.gen_reg_bus[g]) != b:
        return False
    if ctx.vc_ctrls_at_bus.get(b):
        return False
    return (not ctx.is_vm_fixed[b]) and ctx.vc_group[b] < 0 and ctx.q_row[b] >= 0 and ctx.vm_col[b] >= 0


def _held_releasable(ctx, g):
    """The held controller of generator ``g`` (a frozen remote regulator
    lightsim2grid keeps in its group) whose release this loop can act on: no
    other machine checked at its bus. None otherwise."""
    j = ctx.vc_held_of_gen.get(g)
    if j is None or ctx.gr_entry_of_gen.get(g) is None:
        return None
    b = int(ctx.vc_ctrl_bus[j])
    if b < 0 or ctx.bq_entry_of_bus.get(b) is not None or ctx.vc_ctrls_at_bus.get(b):
        return None
    return j


@dataclass
class RowSwitches:
    #: bus -> (summed limit its machines produce, MVAr; True when at min_q)
    to_pq: dict
    #: bus -> (released generators, |V| pu)
    to_pv: dict
    #: VoltageControl group -> True when its controllers are held at min_q
    #: (False: at max_q)
    vc_pin: dict = field(default_factory=dict)
    #: held controller (lightsim2grid's set_hold_frozen_regulators) -> the
    #: generator it releases
    vc_release: dict = field(default_factory=dict)


def plan_switches(ctx, bq_res, gr_res, converged):
    """Per row of the first pass: (status, RowSwitches or None). ``bq_res`` /
    ``gr_res`` are the raw ``BusQViolationsResult`` / ``GenPvReleaseViolationsResult``,
    ``converged`` the (n_rows,) bool of the first pass."""
    n_rows = converged.size
    status = np.full(n_rows, int(ReactiveLimitsStatus.NO_SWITCH), dtype=np.int64)
    switches = [None] * n_rows
    q_cnt = np.asarray(bq_res.count)
    r_cnt = np.asarray(gr_res.count)
    q_tr = np.asarray(bq_res.truncated).astype(bool)
    r_tr = np.asarray(gr_res.truncated).astype(bool)
    q_bus, q_type, q_lim = np.asarray(bq_res.bus_id), np.asarray(bq_res.type), np.asarray(bq_res.limit)
    r_gid, r_el, r_type = np.asarray(gr_res.gen_id), np.asarray(gr_res.el_type), np.asarray(gr_res.type)
    r_lim = np.asarray(gr_res.limit)
    q_stride, r_stride = int(bq_res.stride), int(gr_res.stride)
    cand = np.flatnonzero(converged & ((q_cnt > 0) | (r_cnt > 0)))
    for r in cand:
        r = int(r)
        # the release records of this row that are not a release of a frozen
        # machine (a standby SVC's switch on) are another loop's: not handled
        rel = [(int(r_gid[i]), int(r_el[i]), int(r_type[i]), float(r_lim[i]))
               for i in range(r * r_stride, r * r_stride + max(int(r_cnt[r]), 0))]
        if q_tr[r] or r_tr[r]:
            status[r] = ReactiveLimitsStatus.UNSUPPORTED
            continue
        ok = True
        to_pq = {}
        grp_sat = {}   # group -> {controller bus: at_min}
        for i in range(r * q_stride, r * q_stride + max(int(q_cnt[r]), 0)):
            b = int(q_bus[i])
            if _bus_switchable(ctx, b):
                to_pq[b] = (float(q_lim[i]), int(q_type[i]) == _LOW_Q)
                continue
            g = _bus_group(ctx, b)
            if g < 0:
                ok = False
                break
            grp_sat.setdefault(g, {})[b] = int(q_type[i]) == _LOW_Q
        vc_pin = {}
        if ok:
            for g, sat in grp_sat.items():
                # the whole group, in one direction
                if set(sat) != ctx.vc_grp_buses[g] or len(set(sat.values())) != 1:
                    ok = False
                    break
                vc_pin[g] = next(iter(sat.values()))
        to_pv = {}
        vc_release = {}
        if ok:
            for g, el, vt, lim in rel:
                if el != _EL_GEN or vt not in (_REL_LOW, _REL_HIGH):
                    ok = False
                    break
                if not _gen_releasable(ctx, g):
                    j = _held_releasable(ctx, g)
                    if j is None:
                        ok = False
                        break
                    vc_release[j] = g
                    continue
                b = int(ctx.gen_bus[g])
                vm = lim / ctx.gr["vn_kv"][ctx.gr_entry_of_gen[g]]
                if b in to_pv:
                    gens, vm0 = to_pv[b]
                    if abs(vm0 - vm) > 1e-9:
                        ok = False   # two targets for one bus
                        break
                    to_pv[b] = (gens + (g,), vm0)
                else:
                    to_pv[b] = ((g,), vm)
        if ok and set(to_pq) & set(to_pv):
            ok = False
        if ok and vc_release:
            rel_groups = {int(ctx.vc_ctrl_group[j]) for j in vc_release}
            for grp in rel_groups:
                first = ctx.vc_grp_first[grp]
                # a held first controller left held would pin the whole group
                # through its voltage row; a group both pinned and released has
                # no single labelling
                if (ctx.vc_ctrl_held[first] and first not in vc_release) or grp in vc_pin:
                    ok = False
                    break
            rel_buses = {int(ctx.vc_ctrl_bus[j]) for j in vc_release}
            if rel_buses & (set(to_pq) | set(to_pv)):
                ok = False
        if not ok:
            status[r] = ReactiveLimitsStatus.UNSUPPORTED
            continue
        if not to_pq and not to_pv and not vc_pin and not vc_release:
            continue
        switches[r] = RowSwitches(to_pq=to_pq, to_pv=to_pv, vc_pin=vc_pin, vc_release=vc_release)
    return status, switches


def batch_rule(rows, batch_size, min_last_chunk):
    """(rows re-solved, rows left out): chunks of ``batch_size``; a last,
    partial chunk runs only when it is the only one or holds at least
    ``min_last_chunk`` rows."""
    rows = list(rows)
    n = len(rows)
    if n <= batch_size:
        return rows, []
    n_full = (n // batch_size) * batch_size
    rest = n - n_full
    keep = n_full + (rest if rest >= min_last_chunk else 0)
    return rows[:keep], rows[keep:]


def _held_releases_by_bus(ctx, sw):
    """bus -> sorted generators of the held controllers ``sw`` releases there."""
    out = {}
    for j, g in sw.vc_release.items():
        out.setdefault(int(ctx.vc_ctrl_bus[j]), []).append(int(g))
    return {b: tuple(sorted(gs)) for b, gs in out.items()}


def held_release_offsets(ctx, row_switches):
    """Per second-pass row, ``{bus: MVAr}``: the frozen output of the held
    machines it releases at that bus, offset out of the reactive check (the
    records' value and limit get it back, see :func:`fix_held_release_records`)."""
    return [{b: float(ctx.gen_target_q[list(gs)].sum()) for b, gs in _held_releases_by_bus(ctx, sw).items()}
            for sw in row_switches]


def fix_held_release_records(viols, offsets):
    """The LOW_Q / HIGH_Q records of a released held machine's bus, with its
    frozen output put back into value and limit (``offsets`` from
    :func:`held_release_offsets` for that row)."""
    if not offsets:
        return viols
    from dataclasses import replace
    out = []
    for v in viols:
        if (int(v.element_type) == int(ViolationElementType.BUS)
                and int(v.violation_type) in (_LOW_Q, _HIGH_Q) and int(v.element_id) in offsets):
            off = offsets[int(v.element_id)]
            v = replace(v, value=v.value + off, limit=v.limit + off)
        out.append(v)
    return out


def _augmented_plans(ctx, row_switches):
    """The second pass' reactive and release plans (the grid's plus one entry
    per released bus / switched bus) and their per-row skip masks."""
    n2 = len(row_switches)
    bq, gr = ctx.bq, ctx.gr
    n_check = bq["bus_solver"].size
    n_rel = gr["gen_id"].size

    # released buses: one reactive entry per distinct (bus, released machines,
    # held) -- a released held machine's frozen output is still in the
    # injection, so its entry is offset by it
    extra_q = {}
    for sw in row_switches:
        for b, (gens, _vm) in sw.to_pv.items():
            extra_q.setdefault((b, tuple(sorted(gens)), False), len(extra_q))
        for b, gens in _held_releases_by_bus(ctx, sw).items():
            extra_q.setdefault((b, gens, True), len(extra_q))
    # switched buses: one release entry per distinct (machine bus, regulated
    # bus, at_min) -- a group's controller buses check the bus it regulates
    extra_r = {}
    for sw in row_switches:
        for b, (_lim, at_min) in sw.to_pq.items():
            extra_r.setdefault((b, b, bool(at_min)), len(extra_r))
        for g, at_min in sw.vc_pin.items():
            for c in sorted(ctx.vc_grp_buses[g]):
                extra_r.setdefault((c, int(ctx.vc_grp_reg_bus[g]), bool(at_min)), len(extra_r))

    # ---- reactive plan
    ek = sorted(extra_q, key=extra_q.get)
    add_gens = [list(k[1]) for k in ek]
    add_cnt = np.array([len(g) for g in add_gens], dtype=np.int32)
    gen_start = np.concatenate([bq["gen_start"],
                                bq["gen_start"][-1] + np.cumsum(add_cnt, dtype=np.int32)]).astype(np.int32)
    flat = np.array([g for gs in add_gens for g in gs], dtype=np.int32)
    ne = len(ek)
    offset = np.array([-ctx.gen_target_q[list(k[1])].sum() if k[2] else 0. for k in ek])
    bq_plan = (
        np.concatenate([bq["bus_solver"], np.array([k[0] for k in ek], dtype=np.int32)]),
        np.concatenate([bq["qmin_fixed"], offset]),
        np.concatenate([bq["qmax_fixed"], offset]),
        np.concatenate([bq["n_fixed"], np.zeros(ne, dtype=np.int32)]),
        np.concatenate([bq["bmin"], np.zeros(ne)]),
        np.concatenate([bq["bmax"], np.zeros(ne)]),
        gen_start,
        np.concatenate([bq["gen_id"], flat]),
        np.concatenate([bq["gen_qmin"], ctx.gen_min_q[flat]]),
        np.concatenate([bq["gen_qmax"], ctx.gen_max_q[flat]]),
        bq["sn_mva"],
    )
    bq_skip = np.zeros((n2, n_check + ne), dtype=bool)
    bq_skip[:, n_check:] = True
    for i, sw in enumerate(row_switches):
        for b in sw.to_pq:
            bq_skip[i, ctx.bq_entry_of_bus[b]] = True
        for g in sw.vc_pin:
            for c in ctx.vc_grp_buses[g]:
                bq_skip[i, ctx.bq_entry_of_bus[c]] = True
        for b, (gens, _vm) in sw.to_pv.items():
            bq_skip[i, n_check + extra_q[(b, tuple(sorted(gens)), False)]] = False
        for b, gens in _held_releases_by_bus(ctx, sw).items():
            bq_skip[i, n_check + extra_q[(b, gens, True)]] = False

    # ---- release plan
    rk = sorted(extra_r, key=extra_r.get)
    rep, target = [], []
    for b, reg, _at_min in rk:
        k = ctx.bq_entry_of_bus[b]
        rep.append(int(bq["gen_id"][bq["gen_start"][k]]))   # the bus' first generator reports it
        if reg == b:
            target.append(ctx.gen_target_vm[rep[-1]])
        else:   # a group's set-point
            target.append(ctx.vc_grp_vset[int(ctx.vc_group[reg])])
    rep = np.array(rep, dtype=np.int32)
    nr = len(rk)
    gbus = np.array([k[0] for k in rk], dtype=np.int32)
    rbus = np.array([k[1] for k in rk], dtype=np.int32)
    gr_plan = (
        np.concatenate([gr["gen_id"], rep]),
        np.concatenate([gr["reg_bus"], rbus]),
        np.concatenate([gr["gen_bus"], gbus]),
        np.concatenate([gr["at_min"], np.array([int(k[2]) for k in rk], dtype=np.int32)]),
        np.concatenate([gr["target_vm_pu"], np.asarray(target, dtype=np.float64)]),
        np.concatenate([gr["vn_kv"], ctx.bus_vn_kv[rbus] if nr else np.zeros(0)]),
        np.concatenate([gr["el_type"], np.full(nr, _EL_GEN, dtype=np.int32)]),
        np.concatenate([gr["standby"], np.zeros(nr, dtype=np.int32)]),
        np.concatenate([gr["side"], np.zeros(nr, dtype=np.int32)]),
    )
    gr_skip = np.zeros((n2, n_rel + nr), dtype=bool)
    gr_skip[:, n_rel:] = True
    for i, sw in enumerate(row_switches):
        for b, (gens, _vm) in sw.to_pv.items():
            for g in gens:
                gr_skip[i, ctx.gr_entry_of_gen[g]] = True
        for g in sw.vc_release.values():
            gr_skip[i, ctx.gr_entry_of_gen[g]] = True
        for b, (_lim, at_min) in sw.to_pq.items():
            gr_skip[i, n_rel + extra_r[(b, b, bool(at_min))]] = False
        for g, at_min in sw.vc_pin.items():
            for c in ctx.vc_grp_buses[g]:
                gr_skip[i, n_rel + extra_r[(c, int(ctx.vc_grp_reg_bus[g]), bool(at_min))]] = False
    return bq_plan, bq_skip, gr_plan, gr_skip


def run_second_pass(sweep, ctx, topology_rows, row_switches, batch_size, v_init=None):
    """Configure the second-pass ScenarioSweepGPU ``sweep`` (one row per entry of
    ``row_switches``, with the branches ``topology_rows`` trips) and solve it.
    ``v_init``: ``(device pointer, source rows)`` of the voltages each row starts
    from (the first pass' own, see ScenarioSweepSession.set_v_init_from_ptr);
    None: the base case."""
    n2 = len(row_switches)
    eng = sweep.solver
    sess = eng._s
    sweep.set_topology([list(map(int, t)) for t in topology_rows])

    # injections: the grid's own, plus what the switches change
    dq_rows, dq_bus, dq_val = [], [], []
    to_pq_lists, to_pv_lists, pin_lists, release_lists = [], [], [], []
    sn = ctx.bq["sn_mva"]
    for i, sw in enumerate(row_switches):
        # a saturated group: every controller at its own limit (pu, generator
        # convention) -- its Q is an unknown, so nothing changes in the injection
        pins = []
        for g, at_min in sw.vc_pin.items():
            for j in ctx.vc_grp_ctrls[g]:
                gen = int(ctx.vc_ctrl_elem[j])
                q = ctx.gen_min_q[gen] if at_min else ctx.gen_max_q[gen]
                pins.append((int(j), float(q) / sn))
        pin_lists.append(pins)
        # a released held machine: its frozen Q stays in the injection (the
        # controller mismatch offsets it), nothing to change there
        release_lists.append(sorted(int(j) for j in sw.vc_release))
        pq_list = []
        for b, (lim, _at_min) in sw.to_pq.items():
            dq_rows.append(i); dq_bus.append(b); dq_val.append(lim)   # the machines at their limit
            pq_list.append(int(b))
        pv_list = []
        for b, (gens, vm) in sw.to_pv.items():
            for g in gens:   # the frozen Q leaves the injection
                dq_rows.append(i); dq_bus.append(b); dq_val.append(-ctx.gen_target_q[g])
            pv_list.append((int(b), float(vm)))
        to_pq_lists.append(sorted(pq_list))
        to_pv_lists.append(sorted(pv_list))
    sweep._q_delta = (np.asarray(dq_rows, dtype=np.int64), np.asarray(dq_bus, dtype=np.int64),
                      np.asarray(dq_val, dtype=np.float64))
    sweep.set_injections_from_elements(np.tile(ctx.base_load_p, (n2, 1)),
                                       np.tile(ctx.base_load_q, (n2, 1)),
                                       np.tile(ctx.base_gen_p, (n2, 1)))
    sess.set_pv_pq_switches(to_pq_lists, to_pv_lists)
    sess.set_vc_controller_pins(pin_lists)
    if ctx.vc_ctrl_held.any():
        sess.set_vc_controller_releases(release_lists)

    bq_plan, bq_skip, gr_plan, gr_skip = _augmented_plans(ctx, row_switches)
    eng.set_bus_q_capability(bq_plan)
    eng.set_gen_pv_release_capability(gr_plan)
    sess.set_bus_q_row_skip(bq_skip)
    sess.set_gen_pv_release_row_skip(gr_skip)
    sweep._push_gen_pv_release_targets()
    if v_init is None:
        sess.clear_v_init()
    else:
        sess.set_v_init_from_ptr(int(v_init[0]), list(v_init[1]))
    sweep.compute(batch_size=batch_size)
