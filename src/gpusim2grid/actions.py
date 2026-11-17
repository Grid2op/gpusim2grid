# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Remedial actions on a lightsim2grid grid: element bus moves, disconnections, branch values.

An :class:`Action` says, per element, which bus it goes to, in lightsim2grid's own
``change_bus_*`` convention: a **global bus id**, local bus ``k`` (1-based) of substation ``s``
being ``s + (k - 1) * n_sub``, and ``-1`` for disconnected. Transformer values are set with
lightsim2grid's ``change_ratio_trafo`` / ``change_shift_trafo_deg``. It is a value: frozen,
hashable, comparable::

    from gpusim2grid.actions import Action

    n_sub = grid.get_n_sub()
    split = Action(load_bus={2: 3 + n_sub}, line_bus2={3: 3 + n_sub})  # to bus 2 of substation 3
    trip = Action(line_bus1={5: -1}, line_bus2={5: -1})                 # line 5 out
    pst = Action(trafo_shift_deg={0: 5.0})
    both = Action.combine([split, trip])
    both.validate(grid)          # ValueError on an unknown element, a bus elsewhere, ...
    both.apply_to(grid_copy)     # the same change_bus_* / deactivate_* calls, e.g. for a one-off ac_pf

    # a target partition of a substation (every element listed; the group holding most of a
    # bus's elements keeps that bus, the others take free local buses)
    from gpusim2grid.actions import substation_elements
    elems = substation_elements(grid, 3)       # [("line_bus2", 3), ("trafo_bus1", 0), ...]
    a = Action.from_partition(grid, 3, [elems[:3], elems[3:]])

``-1`` on one end of a line or transformer opens that end only: the branch stays energized
from its other end and keeps its charging (lightsim2grid's ``deactivate_powerline_side1``, a
contingency's ``branch_ids_side1``, OpenLoadFlow's breaker opening at one end); on both ends the
branch trips. ``ContingencyAnalysisGPU.set_actions`` runs actions in batch
(docs/dev_notes/topology_actions_plan.md).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

__all__ = [
    "BUS_FIELDS",
    "DISCONNECTED",
    "VALUE_FIELDS",
    "Action",
    "ElementRef",
    "FrozenMap",
    "current_buses",
    "substation_elements",
]

#: ``Action`` fields mapping an element id to a global bus id (``-1``: disconnected)
BUS_FIELDS: tuple[str, ...] = (
    "line_bus1",
    "line_bus2",
    "trafo_bus1",
    "trafo_bus2",
    "load_bus",
    "gen_bus",
    "sgen_bus",
    "shunt_bus",
    "storage_bus",
)
#: ``Action`` fields mapping a transformer id to a value
VALUE_FIELDS: tuple[str, ...] = ("trafo_ratio", "trafo_shift_deg")

#: one element end, e.g. ``("line_bus2", 3)``: the ``Action`` field and the element id
ElementRef = tuple[str, int]

DISCONNECTED: int = -1


class FrozenMap(Mapping[int, Any]):
    """An immutable, hashable ``{int: value}`` mapping (the fields of an ``Action``)."""

    __slots__ = ("_d", "_h")

    def __init__(self, data: Mapping[int, Any] | None = None) -> None:
        self._d: dict[int, Any] = {int(k): v for k, v in (data or {}).items()}
        self._h: int | None = None

    def __getitem__(self, key: int) -> Any:
        return self._d[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __hash__(self) -> int:
        if self._h is None:
            self._h = hash(frozenset(self._d.items()))
        return self._h

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMap):
            return self._d == other._d
        if isinstance(other, Mapping):
            return self._d == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self._d)


def _frozen_bus(m: Mapping[int, int] | FrozenMap) -> FrozenMap:
    return m if isinstance(m, FrozenMap) else FrozenMap({int(k): int(v) for k, v in m.items()})


def _frozen_value(m: Mapping[int, float] | FrozenMap) -> FrozenMap:
    return m if isinstance(m, FrozenMap) else FrozenMap({int(k): float(v) for k, v in m.items()})


@dataclasses.dataclass(frozen=True)
class Action:
    """One remedial action (see the module docstring for the conventions).

    Each ``*_bus`` field maps an element id to the global bus id it goes to (``-1``:
    disconnected); ``trafo_ratio`` / ``trafo_shift_deg`` map a transformer id to its new ratio /
    phase shift in degrees. Elements not named keep their state."""

    line_bus1: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    line_bus2: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    trafo_bus1: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    trafo_bus2: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    load_bus: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    gen_bus: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    sgen_bus: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    shunt_bus: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    storage_bus: Mapping[int, int] = dataclasses.field(default_factory=FrozenMap)
    trafo_ratio: Mapping[int, float] = dataclasses.field(default_factory=FrozenMap)
    trafo_shift_deg: Mapping[int, float] = dataclasses.field(default_factory=FrozenMap)

    def __post_init__(self) -> None:
        for name in BUS_FIELDS:
            object.__setattr__(self, name, _frozen_bus(getattr(self, name)))
        for name in VALUE_FIELDS:
            object.__setattr__(self, name, _frozen_value(getattr(self, name)))

    # ------------------------------------------------------------------ queries
    def is_empty(self) -> bool:
        """True when the action names no element."""
        return not any(getattr(self, name) for name in BUS_FIELDS + VALUE_FIELDS)

    def bus_items(self) -> Iterator[tuple[str, int, int]]:
        """``(field, element id, global bus)`` for every element the action places or disconnects."""
        for name in BUS_FIELDS:
            for el, bus in getattr(self, name).items():
                yield name, el, bus

    # ------------------------------------------------------------- combination
    @classmethod
    def combine(cls, actions: Sequence[Action]) -> Action:
        """The union of ``actions`` (a strategy of several actions). An element given two
        different buses, or a transformer two different values, raises ``ValueError``."""
        merged: dict[str, dict[int, Any]] = {name: {} for name in BUS_FIELDS + VALUE_FIELDS}
        for a in actions:
            for name in BUS_FIELDS + VALUE_FIELDS:
                for el, v in getattr(a, name).items():
                    prev = merged[name].get(el)
                    if prev is not None and prev != v:
                        raise ValueError(f"{name}[{el}] is set to both {prev} and {v} by these actions")
                    merged[name][el] = v
        return cls(**merged)

    @classmethod
    def from_partition(
        cls,
        grid: LSGrid,
        sub: int,
        groups: Sequence[Sequence[ElementRef]],
        disconnected: Sequence[ElementRef] = (),
    ) -> Action:
        """The action putting the elements of substation ``sub`` on the target partition
        ``groups`` (elements sharing a bus), ``disconnected`` ones on no bus.

        Every element of the substation (``substation_elements``) must be listed, once. Each
        bus in use now is kept by the group holding most of its elements (ties: the earlier
        group); a group keeps at most one bus; the groups left take the free local buses in
        order. Elements already where they go are not named in the action."""
        current = current_buses(grid)
        listed: list[ElementRef] = [e for g in groups for e in g] + list(disconnected)
        expected = set(substation_elements(grid, sub))
        if len(set(listed)) != len(listed):
            raise ValueError("an element is listed twice in the partition")
        if set(listed) != expected:
            missing = sorted(expected - set(listed))
            extra = sorted(set(listed) - expected)
            raise ValueError(
                f"the partition of substation {sub} must list its elements exactly: "
                f"missing {missing}, not in it {extra}"
            )
        n_sub: int = grid.get_n_sub()
        n_local: int = grid.total_bus() // n_sub
        # each group's bus: the bus now holding most of its elements, if no larger group claims it
        claims: list[tuple[int, int, int]] = []  # (-count, group index, bus)
        for gi, g in enumerate(groups):
            counts: dict[int, int] = {}
            for e in g:
                b = current[e]
                if b >= 0:
                    counts[b] = counts.get(b, 0) + 1
            claims += [(-c, gi, b) for b, c in counts.items()]
        bus_of_group: dict[int, int] = {}
        taken: set[int] = set()
        for _, gi, b in sorted(claims):
            if gi not in bus_of_group and b not in taken:
                bus_of_group[gi] = b
                taken.add(b)
        free = [sub + k * n_sub for k in range(n_local) if sub + k * n_sub not in taken]
        for gi in range(len(groups)):
            if gi not in bus_of_group:
                if not free:
                    raise ValueError(f"substation {sub} has no free local bus left for group {gi}")
                bus_of_group[gi] = free.pop(0)
                taken.add(bus_of_group[gi])
        moves: dict[str, dict[int, int]] = {name: {} for name in BUS_FIELDS}
        for gi, g in enumerate(groups):
            for name, el in g:
                if current[(name, el)] != bus_of_group[gi]:
                    moves[name][el] = bus_of_group[gi]
        for name, el in disconnected:
            if current[(name, el)] != DISCONNECTED:
                moves[name][el] = DISCONNECTED
        return cls(**moves)

    # ------------------------------------------------------------ on a grid
    def validate(self, grid: LSGrid) -> None:
        """Raise ``ValueError`` if an element id is unknown, a bus is out of range or in another
        substation than the element's, or a transformer value is not physical."""
        n_sub: int = grid.get_n_sub()
        total: int = grid.total_bus()
        subs = _element_substations(grid)
        for name, el, bus in self.bus_items():
            key = (name, el)
            if key not in subs:
                raise ValueError(f"{name}: no element {el} in this grid")
            if bus == DISCONNECTED:
                continue
            if not 0 <= bus < total:
                raise ValueError(f"{name}[{el}]: bus {bus} out of range [0, {total})")
            if bus % n_sub != subs[key]:
                raise ValueError(
                    f"{name}[{el}]: bus {bus} is in substation {bus % n_sub}, "
                    f"the element is in substation {subs[key]}"
                )
        n_trafo: int = len(grid.get_trafos())
        for name in VALUE_FIELDS:
            for el, v in getattr(self, name).items():
                if not 0 <= el < n_trafo:
                    raise ValueError(f"{name}: no transformer {el} in this grid")
                if name == "trafo_ratio" and not v > 0.0:
                    raise ValueError(f"trafo_ratio[{el}] = {v}: a ratio must be positive")

    def apply_to(self, grid: LSGrid) -> None:
        """Make the action's lightsim2grid calls on ``grid`` (``change_bus*``, ``deactivate_*``,
        ``change_ratio_trafo``, ``change_shift_trafo_deg``): what a row of the batch solves.

        A branch end the action moves never closes the other end: a half-open branch stays
        half-open, as a switch move leaves it (lightsim2grid's ``change_bus*`` alone would
        reconnect it, grid2op's ``set_bus``); that end is opened again after the move."""
        placed: set[ElementRef] = {(name, el) for name, el, bus in self.bus_items() if bus != DISCONNECTED}
        moved_branches: set[ElementRef] = {
            (_OTHER_END[name], el) for name, el in placed if name in _OTHER_END
        } - placed
        open_before: set[ElementRef] = {
            ref for ref, _, bus in _element_ends(grid) if ref in moved_branches and bus == DISCONNECTED
        }
        for name, el, bus in self.bus_items():
            if bus == DISCONNECTED:
                getattr(grid, _DEACTIVATE[name])(int(el))
            else:
                getattr(grid, _CHANGE_BUS[name])(int(el), int(bus))
        for name, el in sorted(open_before):
            getattr(grid, _DEACTIVATE[name])(int(el))
        for el, ratio in self.trafo_ratio.items():
            grid.change_ratio_trafo(int(el), float(ratio))
        for el, deg in self.trafo_shift_deg.items():
            grid.change_shift_trafo_deg(int(el), float(deg))


#: the other end of a branch end
_OTHER_END: dict[str, str] = {
    "line_bus1": "line_bus2",
    "line_bus2": "line_bus1",
    "trafo_bus1": "trafo_bus2",
    "trafo_bus2": "trafo_bus1",
}
_CHANGE_BUS: dict[str, str] = {
    "line_bus1": "change_bus1_powerline",
    "line_bus2": "change_bus2_powerline",
    "trafo_bus1": "change_bus1_trafo",
    "trafo_bus2": "change_bus2_trafo",
    "load_bus": "change_bus_load",
    "gen_bus": "change_bus_gen",
    "sgen_bus": "change_bus_sgen",
    "shunt_bus": "change_bus_shunt",
    "storage_bus": "change_bus_storage",
}
_DEACTIVATE: dict[str, str] = {
    "line_bus1": "deactivate_powerline_side1",
    "line_bus2": "deactivate_powerline_side2",
    "trafo_bus1": "deactivate_trafo_side1",
    "trafo_bus2": "deactivate_trafo_side2",
    "load_bus": "deactivate_load",
    "gen_bus": "deactivate_gen",
    "sgen_bus": "deactivate_sgen",
    "shunt_bus": "deactivate_shunt",
    "storage_bus": "deactivate_storage",
}


def _element_ends(grid: LSGrid) -> Iterator[tuple[ElementRef, int, int]]:
    """``((field, id), substation, global bus or -1)`` of every element end of ``grid``."""
    for ln in grid.get_lines():
        yield ("line_bus1", ln.id), ln.sub1_id, ln.bus1_id if ln.connected1 else DISCONNECTED
        yield ("line_bus2", ln.id), ln.sub2_id, ln.bus2_id if ln.connected2 else DISCONNECTED
    for t in grid.get_trafos():
        yield ("trafo_bus1", t.id), t.sub1_id, t.bus1_id if t.connected1 else DISCONNECTED
        yield ("trafo_bus2", t.id), t.sub2_id, t.bus2_id if t.connected2 else DISCONNECTED
    for name, container in (
        ("load_bus", grid.get_loads()),
        ("gen_bus", grid.get_generators()),
        ("sgen_bus", grid.get_static_generators()),
        ("shunt_bus", grid.get_shunts()),
        ("storage_bus", grid.get_storages()),
    ):
        for el in container:
            yield (name, el.id), el.sub_id, el.bus_id if el.connected else DISCONNECTED


def _element_substations(grid: LSGrid) -> dict[ElementRef, int]:
    return {ref: int(sub) for ref, sub, _ in _element_ends(grid)}


def current_buses(grid: LSGrid) -> dict[ElementRef, int]:
    """The global bus of every element end of ``grid`` (``-1``: disconnected)."""
    return {ref: int(bus) for ref, _, bus in _element_ends(grid)}


def substation_elements(grid: LSGrid, sub: int) -> list[ElementRef]:
    """The element ends of substation ``sub`` connected now (lines, transformers, then
    injections), in a stable order: what ``Action.from_partition`` expects to be listed."""
    return [ref for ref, s, bus in _element_ends(grid) if s == sub and bus != DISCONNECTED]
