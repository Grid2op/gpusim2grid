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

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ..actions import DISCONNECTED, Action, ElementRef, current_buses

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

__all__ = ["CompiledAction", "RowMap", "Strategy", "compile_action", "strategy_row"]

#: one contingency row of the session: the ten lists of ``ContingencyAnalysisGPU.add_contingencies``
#: (branch, shunt, load, gen, branch side 1, branch side 2, sgen, storage, svc, hvdc ids)
CtgElements = tuple[list[int], ...]
#: one strategy: (contingency index, action indices)
Strategy = tuple[int, tuple[int, ...]]

T = TypeVar("T")

_SLOT: dict[str, int] = {"shunt_bus": 1, "load_bus": 2, "gen_bus": 3, "sgen_bus": 6, "storage_bus": 7}


@dataclasses.dataclass(frozen=True)
class CompiledAction:
    """An action read against the grid's current state.

    ``trips``: what it disconnects, as the ten contingency lists (one-end openings in the side
    lists); ``placed``: the element ends it puts on a bus (also on the bus they are on: that
    is a no-op, but not on an element its contingency trips); ``unsupported``: why a row with
    it cannot run yet (``""``: it can)."""

    trips: CtgElements
    placed: frozenset[ElementRef]
    unsupported: str = ""


def _empty() -> list[list[int]]:
    return [[] for _ in range(10)]


def compile_action(grid: LSGrid, action: Action, n_lines: int) -> CompiledAction:
    """``action`` against ``grid`` (its current buses and transformer values)."""
    current = current_buses(grid)
    trips = _empty()
    placed: set[ElementRef] = set()
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
            if bus != now and not why:
                why = (
                    f"{name}[{el}] moves from bus {now} to bus {bus}: bus moves need the union "
                    "sparsity pattern (step B.2), not available yet"
                )
    trafos = grid.get_trafos()
    for el, ratio in action.trafo_ratio.items():
        if ratio != trafos[el].ratio and not why:
            why = f"trafo_ratio[{el}]: per-row branch values (step B.1) are not available yet"
    for el, deg in action.trafo_shift_deg.items():
        if not np.isclose(np.deg2rad(deg), trafos[el].shift_rad, rtol=0.0, atol=1e-15) and not why:
            why = f"trafo_shift_deg[{el}]: per-row branch values (step B.1) are not available yet"
    return CompiledAction(tuple(sorted(set(t)) for t in trips), frozenset(placed), why)


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


def strategy_row(
    ctg: CtgElements, actions: Sequence[CompiledAction], n_lines: int
) -> tuple[CtgElements | None, str]:
    """The session row of a strategy (its contingency's elements and its actions' trips), or
    ``(None, why)`` when it cannot run."""
    for a in actions:
        if a.unsupported:
            return None, a.unsupported
    out_ends = _ctg_ends(ctg, n_lines)
    for a in actions:
        back = sorted(a.placed & out_ends)
        if back:
            return None, f"puts back {back[0][0]}[{back[0][1]}], which its contingency trips"
    lists = [set(x) for x in ctg]
    for a in actions:
        for k in range(10):
            lists[k] |= set(a.trips[k])
    # opened at both ends one way or another: a trip (as add_contingencies has it)
    both = lists[0] | (lists[4] & lists[5])
    lists[0] = both
    lists[4] -= both
    lists[5] -= both
    return tuple(sorted(x) for x in lists), ""


@dataclasses.dataclass
class RowMap:
    """The rows the session solved (``session_rows``: their index in the caller's rows) among
    the ``n_out`` rows the caller sees; ``reasons``: why a row did not run (``""`` for the rows
    that ran and the ones not asked for)."""

    n_out: int
    session_rows: np.ndarray
    reasons: list[str]

    @property
    def ran(self) -> np.ndarray:
        out = np.zeros(self.n_out, dtype=bool)
        out[self.session_rows] = True
        return out

    def expand(self, values: np.ndarray, fill: Any) -> np.ndarray:
        """``values`` (session rows first) on the caller's rows, ``fill`` elsewhere."""
        values = np.asarray(values)
        out = np.full((self.n_out,) + values.shape[1:], fill, dtype=values.dtype)
        out[self.session_rows] = values
        return out

    def expand_list(self, values: Sequence[T], fill: Callable[[], T]) -> list[T]:
        out: list[T] = [fill() for _ in range(self.n_out)]
        for i, r in enumerate(self.session_rows):
            out[int(r)] = values[i]
        return out
