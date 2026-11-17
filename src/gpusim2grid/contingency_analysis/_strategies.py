# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Strategy rows of ``ContingencyAnalysisGPU`` (docs/dev_notes/topology_actions_plan.md, step A).

A strategy is a contingency followed by one or several ``gpusim2grid.actions.Action``. Its row
trips the union of what the contingency and the actions disconnect (an element tripped twice is
tripped once), so it is solved as one more contingency row of the session. What a row needs and
the batch cannot do yet is reported per row (``""`` = runs), never raised for the batch:

- an action putting back an element its contingency trips (a faulted element stays out);
- bus moves (step B.2: union pattern) and transformer values (step B.1), until they land.

``RowMap`` maps the rows the session solved back to the rows the caller sees (contingencies,
then strategies), the others NaN."""

from __future__ import annotations

import copy
import dataclasses
import enum
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ..actions import DISCONNECTED, Action, ElementRef, current_buses

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid


class StrategyCondition(str, enum.Enum):
    """When a contingency's strategies run (pypowsybl's operator strategy conditions), checked
    on the device as the contingency finishes."""

    #: every strategy of the contingency (pypowsybl's TrueCondition)
    TRUE = "TRUE"
    #: only when the contingency violates a limit (pypowsybl's AnyViolationCondition), the limit
    #: checks on (``compute_limit_violations``) at ``violation_threshold``
    ANY_VIOLATION = "ANY_VIOLATION"

    @property
    def code(self) -> int:
        """The session's code (``ContingencyAnalysisSession.set_streaming``)."""
        return 0 if self is StrategyCondition.TRUE else 1


class ResultRows(str, enum.Enum):
    """Which rows a result query covers (``ContingencyAnalysisGPU.get_violations`` ...); an
    array of row indices picks single rows."""

    #: every row: the contingencies, then the strategies
    ALL = "ALL"
    #: the contingency rows (the N-1 results)
    CONTINGENCIES = "CONTINGENCIES"
    #: the strategy rows
    STRATEGIES = "STRATEGIES"

__all__ = [
    "BranchY",
    "CompiledAction",
    "RowBranch",
    "RowMap",
    "ResultRows",
    "Strategy",
    "StrategyCondition",
    "StrategyRow",
    "compile_action",
    "strategy_row",
]

#: one contingency row of the session: the ten lists of ``ContingencyAnalysisGPU.add_contingencies``
#: (branch, shunt, load, gen, branch side 1, branch side 2, sgen, storage, svc, hvdc ids)
CtgElements = tuple[list[int], ...]
#: one strategy: (contingency index, action indices)
Strategy = tuple[int, tuple[int, ...]]

#: a branch's pi-model admittance, lightsim2grid's ``yac_eff_11, _12, _21, _22``
BranchY = tuple[complex, complex, complex, complex]
#: a branch as one row has it: its admittance and its (from, to) solver buses there
RowBranch = tuple[BranchY, int, int]
#: one Ybus value change: (row, col, re, im) in solver numbering, subtracted (as a trip's)
Triplet = tuple[int, int, float, float]

T = TypeVar("T")

_SLOT: dict[str, int] = {"shunt_bus": 1, "load_bus": 2, "gen_bus": 3, "sgen_bus": 6, "storage_bus": 7}


@dataclasses.dataclass(frozen=True)
class CompiledAction:
    """An action read against the grid's current state.

    ``trips``: what it disconnects, as the ten contingency lists (one-end openings in the side
    lists); ``placed``: the element ends it puts on a bus (also on the bus they are on: that
    is a no-op, but not on an element its contingency trips); ``branch_y``: the new admittance
    of the branches whose values it changes (lines then transformers numbering; ``base_y`` their
    admittance now); ``unsupported``: why a row with it cannot run yet (``""``: it can)."""

    trips: CtgElements
    placed: frozenset[ElementRef]
    branch_y: dict[int, BranchY] = dataclasses.field(default_factory=dict)
    base_y: dict[int, BranchY] = dataclasses.field(default_factory=dict)
    #: element ends it moves to another bus: {end: global bus} (step B.2)
    moves: dict[ElementRef, int] = dataclasses.field(default_factory=dict)
    #: the injection (pu, generator convention) of every load / static generator it moves
    injection: dict[ElementRef, complex] = dataclasses.field(default_factory=dict)
    #: the Ybus diagonal admittance (pu) of every shunt it moves
    shunt_y: dict[int, complex] = dataclasses.field(default_factory=dict)
    #: the voltage set-point (pu) of every voltage-regulating generator it moves
    gen_vm: dict[int, float] = dataclasses.field(default_factory=dict)
    unsupported: str = ""


def _empty() -> list[list[int]]:
    return [[] for _ in range(10)]


def compile_action(
    grid: LSGrid, action: Action, n_lines: int, current: dict[ElementRef, int] | None = None
) -> CompiledAction:
    """``action`` against ``grid`` (its current buses and transformer values). ``current``: the
    grid's ``current_buses``, read once for many actions (it walks every element end)."""
    if current is None:
        current = current_buses(grid)
    trips = _empty()
    placed: set[ElementRef] = set()
    moves: dict[ElementRef, int] = {}
    injection: dict[ElementRef, complex] = {}
    shunt_y: dict[int, complex] = {}
    why = ""
    for name, el, bus in action.bus_items():
        now = current[(name, el)]
        if bus == DISCONNECTED:
            if now == DISCONNECTED:
                continue
            if name in ("line_bus1", "line_bus2", "trafo_bus1", "trafo_bus2"):
                branch = el + (n_lines if name.startswith("trafo") else 0)
                trips[4 if name.endswith("1") else 5].append(branch)
            else:
                trips[_SLOT[name]].append(el)
        else:
            placed.add((name, el))
            if bus == now:
                continue
            if now == DISCONNECTED:
                why = why or f"{name}[{el}] is disconnected now: reconnecting an element is not supported yet"
            elif name in _MOVE_UNSUPPORTED:
                why = why or f"{name}[{el}]: {_MOVE_UNSUPPORTED[name]}"
            else:
                moves[(name, el)] = int(bus)
    sn_mva: float = float(grid.get_sn_mva())
    gen_vm: dict[int, float] = {}
    for (name, el), _ in moves.items():
        if name == "gen_bus":
            gen = grid.get_generators()[el]
            why = why or _gen_move_unsupported(grid, el)
            if gen.voltage_regulator_on:
                # its reactive output is free where it regulates: only its active power moves
                injection[(name, el)] = complex(gen.target_p_mw, 0.0) / sn_mva
                gen_vm[el] = float(gen.target_vm_pu)
            else:
                injection[(name, el)] = complex(gen.target_p_mw, gen.target_q_mvar) / sn_mva
        elif name == "load_bus":
            ld = grid.get_loads()[el]
            injection[(name, el)] = -complex(ld.target_p_mw, ld.target_q_mvar) / sn_mva
        elif name == "sgen_bus":
            sg = grid.get_static_generators()[el]
            injection[(name, el)] = complex(sg.target_p_mw, sg.target_q_mvar) / sn_mva
        elif name == "shunt_bus":
            shunt_y[el] = _shunt_admittance(grid, el)
    branch_y: dict[int, BranchY] = {}
    base_y: dict[int, BranchY] = {}
    trafos = grid.get_trafos()
    changed: set[int] = {el for el, r in action.trafo_ratio.items() if r != trafos[el].ratio}
    changed |= {
        el
        for el, deg in action.trafo_shift_deg.items()
        if not np.isclose(np.deg2rad(deg), trafos[el].shift_rad, rtol=0.0, atol=1e-15)
    }
    if changed:
        # lightsim2grid's own admittance after the change (a copy: the grid is left as it is)
        changed_grid = copy.deepcopy(grid)
        for el, ratio in action.trafo_ratio.items():
            changed_grid.change_ratio_trafo(int(el), float(ratio))
        for el, deg in action.trafo_shift_deg.items():
            changed_grid.change_shift_trafo_deg(int(el), float(deg))
        new_trafos = changed_grid.get_trafos()
        for el in sorted(changed):
            branch_y[n_lines + el] = _y(new_trafos[el])
            base_y[n_lines + el] = _y(trafos[el])
    lists: CtgElements = tuple(sorted(set(t)) for t in trips)
    return CompiledAction(lists, frozenset(placed), branch_y, base_y, moves, injection, shunt_y, gen_vm, why)


#: element kinds whose moves the first version leaves out (reported per row)
_MOVE_UNSUPPORTED: dict[str, str] = {
    "storage_bus": "storage unit moves are not supported yet",
}


def _gen_move_unsupported(grid: LSGrid, el: int) -> str:
    """Why generator ``el`` cannot move to another bus ("" when it can): the slack, a slack
    participant, a regulator of a remote bus (lightsim2grid's ScenarioSweep refuses them too; it
    also refuses a member of a voltage-control group, which moves here)."""
    gens = grid.get_generators()
    gen = gens[el]
    name = f"gen_bus[{el}]"
    if gen.is_slack:
        return f"{name}: the slack generator cannot move"
    if gen.can_participate_slack:
        return f"{name}: a generator taking part in the slack cannot move"
    if not gen.voltage_regulator_on:
        return ""
    if gen.regulated_bus_id != gen.bus_id:
        return f"{name}: a generator regulating a remote bus cannot move"
    # one of a voltage-control group (other machines regulating its bus) leaves the group and
    # holds its new bus alone: the session drops it from the group on that row
    return ""


def _shunt_admittance(grid: LSGrid, shunt: int) -> complex:
    """The shunt's Ybus diagonal admittance (pu): lightsim2grid's own, the solver Ybus with the
    shunt minus without it (no sign convention to guess)."""
    bus: int = int(grid.id_me_to_ac_solver()[grid.get_shunts()[shunt].bus_id])
    without = copy.deepcopy(grid)
    without.deactivate_shunt(shunt)
    without.tell_solver_need_reset()
    with_v = np.asarray(grid.get_V(), dtype=complex)
    without.ac_pf(with_v.copy(), 30, 1e-8)
    return complex(grid.get_Ybus_solver()[bus, bus] - without.get_Ybus_solver()[bus, bus])


def _y(branch: Any) -> BranchY:
    return (
        complex(branch.yac_eff_11),
        complex(branch.yac_eff_12),
        complex(branch.yac_eff_21),
        complex(branch.yac_eff_22),
    )


def _ctg_ends(ctg: CtgElements, n_lines: int) -> set[ElementRef]:
    """The element ends a contingency row takes out."""
    out: set[ElementRef] = set()

    def branch_ends(b: int, sides: tuple[str, ...]) -> None:
        kind = "line" if b < n_lines else "trafo"
        el = b if b < n_lines else b - n_lines
        for s in sides:
            out.add((f"{kind}_bus{s}", el))

    for b in ctg[0]:
        branch_ends(b, ("1", "2"))
    for b in ctg[4]:
        branch_ends(b, ("1",))
    for b in ctg[5]:
        branch_ends(b, ("2",))
    for name, slot in _SLOT.items():
        out |= {(name, el) for el in ctg[slot]}
    return out


@dataclasses.dataclass(frozen=True)
class UnionPlan:
    """The spare buses and union Ybus entries the moves of a batch need (solver numbering: the
    n base solver buses, then the spares). Empty: today's base pattern."""

    n_solver: int
    spare_global: list[int]
    parents: list[int]
    entries: list[tuple[int, int]]
    #: global bus -> solver bus, spares included
    solver_of_global: dict[int, int]

    @property
    def n_spare(self) -> int:
        return len(self.spare_global)

    def is_empty(self) -> bool:
        return not self.spare_global and not self.entries


def plan_union(
    grid: LSGrid,
    compiled: Sequence[CompiledAction],
    combos: Iterable[tuple[int, ...]],
    n_lines: int,
    branch_from: np.ndarray,
    branch_to: np.ndarray,
) -> UnionPlan:
    """Spare buses (moved-to buses no base solver bus is) and the union entries of every action
    combination a strategy uses (the moved stamps of each branch, a moved injection's diagonal).

    The pattern is built ONCE, over every strategy registered on the analysis, before the first
    run: phase 1, phase 2 and every chunk of either solve on it (a row whose actions move nothing
    carries zeros at the extra entries), and it is rebuilt only when the actions / strategies
    change. The prototype (gpu-security-analysis) builds a smaller union per block of strategies
    instead: less fill per row, one pattern per block. If building it upfront ever shows (many
    actions, a large grid), it could be streamed while the GPU works on the first rows -- but
    rows already on the device were analysed on the pattern known when they were loaded, so that
    needs a join of patterns (a pattern that only grows, and a new analysis when it does), not
    just a background thread."""
    me_to_solver = np.asarray(grid.id_me_to_ac_solver(), dtype=np.int64)
    n_solver: int = int(grid.get_Ybus_solver().shape[0])
    current = current_buses(grid)
    solver_of_global: dict[int, int] = {}
    spare_global: list[int] = []
    parents: list[int] = []
    for a in compiled:
        for ref, g in a.moves.items():
            if g in solver_of_global or me_to_solver[g] >= 0:
                continue
            solver_of_global[g] = n_solver + len(spare_global)
            spare_global.append(g)
            parents.append(int(me_to_solver[current[ref]]))

    def solver(g: int) -> int:
        return solver_of_global[g] if g in solver_of_global else int(me_to_solver[g])

    entries: set[tuple[int, int]] = {(s, s) for s in solver_of_global.values()}
    for combo in combos:
        moves: dict[ElementRef, int] = {}
        for k in combo:
            moves.update(compiled[k].moves)
        for f, t in _moved_branch_ends(moves, n_lines, branch_from, branch_to, solver).values():
            ends = [x for x in (f, t) if x >= 0]
            entries |= {(i, j) for i in ends for j in ends}
        for (name, _), g in moves.items():
            if not name.startswith(("line", "trafo")):
                entries.add((solver(g), solver(g)))
    if not spare_global:
        # merges only: the entries the base pattern already has are not new
        Y = grid.get_Ybus_solver()
        entries = {(i, j) for i, j in entries if Y[i, j] == 0}
    return UnionPlan(n_solver, spare_global, parents, sorted(entries), {**solver_of_global})


def _moved_branch_ends(
    moves: dict[ElementRef, int], n_lines: int, branch_from: np.ndarray, branch_to: np.ndarray, solver: Any
) -> dict[int, tuple[int, int]]:
    """{branch: (new from, new to)} in solver numbering for every branch an end of which moves."""
    out: dict[int, tuple[int, int]] = {}
    for (name, el), g in moves.items():
        if not name.startswith(("line", "trafo")):
            continue
        b = el + (n_lines if name.startswith("trafo") else 0)
        f, t = out.get(b, (int(branch_from[b]), int(branch_to[b])))
        out[b] = (solver(g), t) if name.endswith("1") else (f, solver(g))
    return out


@dataclasses.dataclass(frozen=True)
class TopologyContext:
    """What the rows' moves need of the grid, once per run: the union plan, branch ends and
    admittances (solver numbering), the element ends per solver bus in the base case."""

    plan: UnionPlan
    n_lines: int
    branch_from: np.ndarray
    branch_to: np.ndarray
    branch_y: list[BranchY]
    occupancy: np.ndarray
    me_to_solver: np.ndarray
    current: dict[ElementRef, int]
    #: solver bus -> the machines holding its voltage in the base case (see _regulators)
    regulators: dict[int, frozenset[ElementRef]] = dataclasses.field(default_factory=dict)

    def solver(self, g: int) -> int:
        return self.plan.solver_of_global[g] if g in self.plan.solver_of_global else int(self.me_to_solver[g])


def topology_context(
    grid: LSGrid, plan: UnionPlan, n_lines: int, branch_from: np.ndarray, branch_to: np.ndarray
) -> TopologyContext:
    """The context of a run's rows (``plan``: its union, set on the session)."""
    me_to_solver = np.asarray(grid.id_me_to_ac_solver(), dtype=np.int64)
    occupancy = np.zeros(plan.n_solver + plan.n_spare, dtype=np.int64)
    for _, bus in current_buses(grid).items():
        if bus >= 0 and me_to_solver[bus] >= 0:
            occupancy[me_to_solver[bus]] += 1
    branch_y: list[BranchY] = [_y(ln) for ln in grid.get_lines()] + [_y(t) for t in grid.get_trafos()]
    current = current_buses(grid)
    return TopologyContext(plan, n_lines, branch_from, branch_to, branch_y, occupancy, me_to_solver, current,
                           _regulators(grid, me_to_solver))


def _regulators(grid: LSGrid, me_to_solver: np.ndarray) -> dict[int, frozenset[ElementRef]]:
    """Solver bus -> every machine holding its voltage in the base case, keyed on the bus it
    REGULATES (a remote regulator holds its regulated bus, not its own): the voltage-regulating
    generators (not treated as off), storage units and VSC converter stations, and the SVCs in
    voltage mode -- the controllers the session counts. ``("hvdc_station1" / "2", line id)``
    names a station."""
    regulators: dict[int, set[ElementRef]] = {}

    def add(bus: int, ref: ElementRef) -> None:
        if 0 <= bus < len(me_to_solver) and me_to_solver[bus] >= 0:
            regulators.setdefault(int(me_to_solver[bus]), set()).add(ref)

    # lightsim2grid's GeneratorContainer::_treated_as_off: a pseudo-off generator holds nothing
    turned_off_pv: bool = bool(grid.get_turnedoff_gen_pv())
    for g in grid.get_generators():
        pseudo_off = not g.is_slack and abs(g.slack_weight) < 1e-7 and abs(g.target_p_mw) < 1e-7
        if g.connected and g.voltage_regulator_on and not (pseudo_off and not turned_off_pv):
            add(int(g.regulated_bus_id), ("gen_bus", int(g.id)))
    for s in grid.get_storages():
        if s.connected and s.voltage_regulator_on:
            add(int(s.bus_id), ("storage_bus", int(s.id)))  # a unit holds its own bus only
    for sv in grid.get_svcs():
        if sv.connected and int(sv.regulation_mode) == 1:  # VOLTAGE
            add(int(sv.regulated_bus_id), ("svc", int(sv.id)))
    for h in grid.get_dclines():
        for side, st in (("1", h.station1), ("2", h.station2)):
            if st.connected and st.voltage_regulator_on and int(st.converter_type) == 0:  # VSC
                add(int(st.bus_id), (f"hvdc_station{side}", int(h.id)))
    return {b: frozenset(r) for b, r in regulators.items()}


@dataclasses.dataclass(frozen=True)
class StrategyRow:
    """The session row of a strategy: ``elements`` (the ten contingency lists, ``None`` when it
    cannot run, ``reason`` saying why), ``triplets`` (its Ybus value changes, subtracted) and
    ``branch_y`` (the new admittance of the branches it changes, for its flows)."""

    elements: CtgElements | None
    triplets: list[Triplet] = dataclasses.field(default_factory=list)
    branch_y: dict[int, BranchY] = dataclasses.field(default_factory=dict)
    reason: str = ""
    #: BASE buses its moves leave with no element: always masked (a spare bus the row does not
    #: use is masked by the session itself, so it is never listed here)
    empty_buses: list[int] = dataclasses.field(default_factory=list)
    #: (bus, dS pu) added to the row's Sbus (moved injections)
    sbus: list[tuple[int, complex]] = dataclasses.field(default_factory=list)
    #: the new (from, to) solver buses of the branches its moves move (for their flows)
    branch_ends: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)
    #: the admittance of the moved branches whose block is not the base one (opened at one end
    #: by the row, their other end moved: Kron-reduced onto it), for their flows
    branch_ends_y: dict[int, BranchY] = dataclasses.field(default_factory=dict)
    #: base PV buses its moved generators leave without a regulator (PQ on this row)
    pv_release: list[int] = dataclasses.field(default_factory=list)
    #: (bus, |V| pu) its moved generators hold PV
    pv_set: list[tuple[int, float]] = dataclasses.field(default_factory=list)
    #: (generator, solver bus) of every generator it moves (the outer loops see it there)
    gen_bus: list[tuple[int, int]] = dataclasses.field(default_factory=list)


def strategy_row(
    ctg: CtgElements,
    actions: Sequence[CompiledAction],
    n_lines: int,
    branch_from: np.ndarray,
    branch_to: np.ndarray,
    topo: TopologyContext | None = None,
) -> StrategyRow:
    """The session row of a strategy: its contingency's elements and its actions' trips, and the
    Ybus changes of the branch values its actions set (``branch_from`` / ``branch_to``: solver
    buses of every branch, -1 for a Kron-reduced end)."""
    for a in actions:
        if a.unsupported:
            return StrategyRow(None, reason=a.unsupported)
    # an element end its contingency trips stays out whatever the actions do with it (the
    # contingency's breakers stay open, as in OpenLoadFlow's security analysis): their moves of
    # it are dropped, the rest of each action applies
    out_ends = _ctg_ends(ctg, n_lines)
    actions = [
        dataclasses.replace(a, moves={ref: g for ref, g in a.moves.items() if ref not in out_ends})
        if a.placed & out_ends else a
        for a in actions
    ]
    lists = [set(x) for x in ctg]
    for a in actions:
        for k in range(10):
            lists[k] |= set(a.trips[k])
    # opened at both ends one way or another: a trip (as add_contingencies has it)
    both = lists[0] | (lists[4] & lists[5])
    lists[0] = both
    lists[4] -= both
    lists[5] -= both
    # branch values: a branch the row opens keeps no admittance to change
    opened: set[int] = lists[0] | lists[4] | lists[5]
    branch_y: dict[int, BranchY] = {}
    base_y: dict[int, BranchY] = {}
    for a in actions:
        for b, y in a.branch_y.items():
            if b in opened:
                continue
            if b in branch_y and branch_y[b] != y:
                return StrategyRow(None, reason=f"two of its actions set different values on branch {b}")
            branch_y[b] = y
            base_y[b] = a.base_y[b]
    triplets: list[Triplet] = []
    for b, y in branch_y.items():
        f, t = int(branch_from[b]), int(branch_to[b])
        for (i, j), y0, y1 in zip(((f, f), (f, t), (t, f), (t, t)), base_y[b], y):
            if i >= 0 and j >= 0:
                d = y0 - y1  # subtracted from the row's Ybus: base - new
                triplets.append((i, j, d.real, d.imag))
    # bus moves (step B.2): stamps from the old ends to the new, moved injections, empty buses
    moves: dict[ElementRef, int] = {}
    injection: dict[ElementRef, complex] = {}
    shunt_y: dict[int, complex] = {}
    gen_vm: dict[int, float] = {}
    for a in actions:
        gen_vm.update(a.gen_vm)
        for ref, g in a.moves.items():
            if moves.get(ref, g) != g:
                name, el = ref
                return StrategyRow(None, reason=f"two of its actions move {name}[{el}] to different buses")
            moves[ref] = g
        injection.update(a.injection)
        shunt_y.update(a.shunt_y)
    if not moves:
        return StrategyRow(tuple(sorted(x) for x in lists), triplets, branch_y)
    if topo is None:
        return StrategyRow(None, reason="bus moves need the union pattern of the batch")
    occupancy: np.ndarray = topo.occupancy.copy()
    sbus: list[tuple[int, complex]] = []
    moved = _moved_branch_ends(moves, topo.n_lines, topo.branch_from, topo.branch_to, topo.solver)
    branch_ends: dict[int, tuple[int, int]] = {}
    branch_ends_y: dict[int, BranchY] = {}
    left: set[int] = set()  # the base buses its moves take an element end off
    for b, (f2, t2) in moved.items():
        if b in lists[0]:
            continue  # the row trips it: nothing left to move
        if b in branch_y:
            return StrategyRow(None, reason=f"branch {b} is moved and its value changed by one strategy")
        f, t = int(topo.branch_from[b]), int(topo.branch_to[b])
        open_side: int = 1 if b in lists[4] else 2 if b in lists[5] else 0
        if open_side:
            # opened at one end by the row (its actions or its contingency), moved at the other:
            # the session's one-end trip leaves the kept end's Kron-reduced diagonal at its base
            # bus (append_one_side_trips); it moves to the kept end's new bus. The branch stays
            # one of the row's opened ones (flows, the outer loops' view).
            kind, el = ("line", b) if b < topo.n_lines else ("trafo", b - topo.n_lines)
            if (f"{kind}_bus{open_side}", el) in moves:
                return StrategyRow(None, reason=f"{kind}_bus{open_side}[{el}] is both opened and moved by its actions")
            y = topo.branch_y[b]
            y_oo, y_ok, y_ko, y_kk = (y[0], y[1], y[2], y[3]) if open_side == 1 else (y[3], y[2], y[1], y[0])
            if f < 0 or t < 0 or y_oo == 0:
                continue  # nothing stamped at the opened end: the session's one-end trip does it all
            y_red: complex = y_kk - y_ko * y_ok / y_oo
            kept_old, kept_new = (t, t2) if open_side == 1 else (f, f2)
            triplets.append((kept_old, kept_old, y_red.real, y_red.imag))  # subtracted: out of the old bus
            triplets.append((kept_new, kept_new, -y_red.real, -y_red.imag))  # added at the new one
            branch_ends[b] = (-1, kept_new) if open_side == 1 else (kept_new, -1)
            branch_ends_y[b] = (0j, 0j, 0j, y_red) if open_side == 1 else (y_red, 0j, 0j, 0j)
            occupancy[kept_old] -= 1
            occupancy[kept_new] += 1
            left.add(kept_old)
            continue
        branch_ends[b] = (f2, t2)
        # a half-open branch (an end open in the base case, -1): its block is lightsim2grid's
        # effective one, Kron-reduced onto the closed end, (y, 0, 0, 0) -- only that end moves
        # (placing the open one is a reconnection, refused by compile_action)
        y = topo.branch_y[b]
        for (i, j), yv in zip(((f, f), (f, t), (t, f), (t, t)), y):
            if i >= 0 and j >= 0:
                triplets.append((i, j, yv.real, yv.imag))  # subtracted: out of the old ends
        for (i, j), yv in zip(((f2, f2), (f2, t2), (t2, f2), (t2, t2)), y):
            if i >= 0 and j >= 0:
                triplets.append((i, j, -yv.real, -yv.imag))  # added at the new ones
        for old, new in ((f, f2), (t, t2)):
            if old != new and old >= 0 and new >= 0:
                occupancy[old] -= 1
                occupancy[new] += 1
                left.add(old)
    for (name, el), g in moves.items():
        if name.startswith(("line", "trafo")):
            continue
        old = int(topo.me_to_solver[topo.current[(name, el)]])
        new = topo.solver(g)
        if name == "shunt_bus":
            y_sh = shunt_y[el]
            triplets += [(old, old, y_sh.real, y_sh.imag), (new, new, -y_sh.real, -y_sh.imag)]
        else:
            s_inj = injection[(name, el)]
            sbus += [(old, -s_inj), (new, s_inj)]
        occupancy[old] -= 1
        occupancy[new] += 1
        left.add(old)
    # the element ends the row takes out (its contingency's and its actions' trips) hold no bus
    # any more: a bus its moves leave with only those is empty too
    for ref in _ctg_ends(tuple(lists), topo.n_lines):
        bus = topo.current.get(ref, DISCONNECTED)
        if ref not in moves and bus >= 0 and topo.me_to_solver[bus] >= 0:
            occupancy[topo.me_to_solver[bus]] -= 1
    # the base buses its moves empty (the spares are masked by the session unless used); a bus
    # its trips alone empty is left to the session, as on its contingency's row
    empty = [b for b in sorted(left) if b < topo.plan.n_solver and occupancy[b] <= 0]
    pv_release, pv_set, why = _voltage_control_of_moves(moves, gen_vm, lists, topo)
    if why:
        return StrategyRow(None, reason=why)
    gen_bus = sorted((el, topo.solver(g)) for (name, el), g in moves.items() if name == "gen_bus")
    return StrategyRow(tuple(sorted(x) for x in lists), triplets, branch_y, empty_buses=empty, sbus=sbus,
                       branch_ends=branch_ends, branch_ends_y=branch_ends_y, pv_release=pv_release,
                       pv_set=pv_set, gen_bus=gen_bus)


def _voltage_control_of_moves(
    moves: dict[ElementRef, int], gen_vm: dict[int, float], lists: list[set[int]], topo: TopologyContext
) -> tuple[list[int], list[tuple[int, float]], str]:
    """The PV buses a row's moved regulating generators leave without a regulator (released),
    the buses they hold PV at their set-point, and why the row cannot run ("" when it can): a
    bus another regulator holds, or two set-points on one bus."""
    moved: dict[int, int] = {
        el: topo.solver(g) for (name, el), g in moves.items() if name == "gen_bus" and el in gen_vm
    }
    if not moved:
        return [], [], ""
    # what the row takes away from a bus: the generators it moves, and the controllers it trips
    gone: set[ElementRef] = {("gen_bus", el) for el in moved} | {("gen_bus", g) for g in lists[3]}
    gone |= {("storage_bus", s) for s in lists[7]} | {("svc", s) for s in lists[8]}
    gone |= {(f"hvdc_station{side}", h) for h in lists[9] for side in ("1", "2")}
    release: list[int] = []
    for el in moved:
        old = int(topo.me_to_solver[topo.current[("gen_bus", el)]])
        if old in topo.regulators and not (topo.regulators[old] - gone):
            release.append(old)
    pv: dict[int, float] = {}
    for el, new in moved.items():
        if topo.regulators.get(new, frozenset()) - gone:
            return [], [], f"gen_bus[{el}] is moved onto a bus another regulator holds"
        if new in pv and pv[new] != gen_vm[el]:
            return [], [], f"two generators moved onto bus {new} hold different set-points"
        pv[new] = gen_vm[el]
    return sorted(set(release)), sorted(pv.items()), ""


@dataclasses.dataclass
class RowMap:
    """The rows the session solved (``session_rows``: their index in the caller's rows) among
    the ``n_out`` rows the caller sees; ``reasons``: why a row did not run (``""`` for the rows
    that ran and the ones not asked for); ``released``: per session row, whether it ran (a
    strategy whose contingency failed the condition did not; None: every one)."""

    n_out: int
    session_rows: np.ndarray
    reasons: list[str]
    released: np.ndarray | None = None

    def session_pos(self, rows: np.ndarray) -> np.ndarray:
        """Per caller row, its position among the session rows; -1 for a row that did not run."""
        out: np.ndarray = np.full(self.n_out, -1, dtype=np.int64)
        pos, run = self._run_rows()
        out[run] = pos
        return out[np.asarray(rows, dtype=np.int64)]

    def _run_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """(the session positions that ran, their caller rows)."""
        if self.released is None:
            return np.arange(len(self.session_rows)), self.session_rows
        pos: np.ndarray = np.flatnonzero(self.released)
        return pos, self.session_rows[pos]

    @property
    def ran(self) -> np.ndarray:
        out = np.zeros(self.n_out, dtype=bool)
        out[self._run_rows()[1]] = True
        return out

    def expand(self, values: np.ndarray, fill: Any) -> np.ndarray:
        """``values`` (one per session row) on the caller's rows that ran, ``fill`` elsewhere."""
        values = np.asarray(values)
        out = np.full((self.n_out,) + values.shape[1:], fill, dtype=values.dtype)
        pos, rows = self._run_rows()
        out[rows] = values[pos]
        return out

    def expand_list(self, values: Sequence[T], fill: Callable[[], T]) -> list[T]:
        out: list[T] = [fill() for _ in range(self.n_out)]
        for i, r in zip(*self._run_rows()):
            out[int(r)] = values[int(i)]
        return out
