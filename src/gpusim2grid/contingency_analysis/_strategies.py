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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from ..actions import DISCONNECTED, Action, ElementRef, current_buses

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

__all__ = ["BranchY", "CompiledAction", "RowMap", "Strategy", "StrategyRow", "compile_action", "strategy_row"]

#: one contingency row of the session: the ten lists of ``ContingencyAnalysisGPU.add_contingencies``
#: (branch, shunt, load, gen, branch side 1, branch side 2, sgen, storage, svc, hvdc ids)
CtgElements = tuple[list[int], ...]
#: one strategy: (contingency index, action indices)
Strategy = tuple[int, tuple[int, ...]]

#: a branch's pi-model admittance, lightsim2grid's ``yac_eff_11, _12, _21, _22``
BranchY = tuple[complex, complex, complex, complex]
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
    return CompiledAction(tuple(sorted(set(t)) for t in trips), frozenset(placed), branch_y, base_y, why)


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
class StrategyRow:
    """The session row of a strategy: ``elements`` (the ten contingency lists, ``None`` when it
    cannot run, ``reason`` saying why), ``triplets`` (its Ybus value changes, subtracted) and
    ``branch_y`` (the new admittance of the branches it changes, for its flows)."""

    elements: CtgElements | None
    triplets: list[Triplet] = dataclasses.field(default_factory=list)
    branch_y: dict[int, BranchY] = dataclasses.field(default_factory=dict)
    reason: str = ""


def strategy_row(
    ctg: CtgElements,
    actions: Sequence[CompiledAction],
    n_lines: int,
    branch_from: np.ndarray,
    branch_to: np.ndarray,
) -> StrategyRow:
    """The session row of a strategy: its contingency's elements and its actions' trips, and the
    Ybus changes of the branch values its actions set (``branch_from`` / ``branch_to``: solver
    buses of every branch, -1 for a Kron-reduced end)."""
    for a in actions:
        if a.unsupported:
            return StrategyRow(None, reason=a.unsupported)
    out_ends = _ctg_ends(ctg, n_lines)
    for a in actions:
        back = sorted(a.placed & out_ends)
        if back:
            name, el = back[0]
            return StrategyRow(None, reason=f"puts back {name}[{el}], which its contingency trips")
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
    return StrategyRow(tuple(sorted(x) for x in lists), triplets, branch_y)


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
