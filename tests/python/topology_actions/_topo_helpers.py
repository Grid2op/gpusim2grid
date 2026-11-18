# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helpers of the topology-action tests (docs/dev_notes/topology_actions_plan.md).

Grids with two busbars per substation (lightsim2grid's ``n_busbar_per_sub=2``: global bus
``s + n_sub`` is local bus 2 of substation ``s``), the lightsim2grid oracle (a one-off
``ac_pf`` of a deep copy with the same ``change_bus_*`` / ``deactivate_*`` / tap calls), and
the mapping of a batch's voltage columns back to global bus ids.
"""

from __future__ import annotations

import copy
import functools
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid
    from pandapower.auxiliary import pandapowerNet

    from gpusim2grid import ContingencyAnalysisGPU
    from gpusim2grid.actions import Action

MAX_IT: int = 30
TOL: float = 1e-10

#: one strategy row: (contingency index, action indices)
Strategy = tuple[int, list[int]]
#: one element end of an ``Action`` field, e.g. ``("line_bus2", 3)``
ElementRef = tuple[str, int]


def _two_busbars(net: pandapowerNet) -> LSGrid:
    """``net`` with an empty second busbar per substation, solved with NR_KLU."""
    import pandapower as pp
    from lightsim2grid.network import init_from_pandapower

    n_sub: int = len(net.bus)
    for b in range(n_sub):
        pp.create_bus(net, vn_kv=net.bus.vn_kv.iloc[b], in_service=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net, n_sub=n_sub, n_busbar_per_sub=2)
    grid.change_algorithm("NR_KLU")
    V: np.ndarray = grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0, "base case diverged"
    return grid


@functools.cache
def _case14() -> LSGrid:
    import pandapower.networks as pn

    return _two_busbars(cast("pandapowerNet", pn.case14()))


@functools.cache
def _case118() -> LSGrid:
    import pandapower.networks as pn

    return _two_busbars(cast("pandapowerNet", pn.case118()))


def _fresh(grid: LSGrid) -> LSGrid:
    """A deep copy, solved again (a copy does not keep its solver's state)."""
    g = copy.deepcopy(grid)
    g.tell_solver_need_reset()
    V: np.ndarray = g.ac_pf(np.ones(g.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0, "base case diverged"
    return g


def case14() -> LSGrid:
    """A fresh solved copy of the two-busbar IEEE 14 (tests may mutate it)."""
    return _fresh(_case14())


def case118() -> LSGrid:
    """A fresh solved copy of the two-busbar IEEE 118."""
    return _fresh(_case118())


def _n_lines(grid: LSGrid) -> int:
    return len(grid.get_lines())


def branch_id_of_trafo(grid: LSGrid, trafo_id: int) -> int:
    """lightsim2grid's branch numbering: lines, then transformers."""
    return _n_lines(grid) + int(trafo_id)


def bus2(grid: LSGrid, sub: int) -> int:
    """Global id of local bus 2 of substation ``sub``."""
    return int(sub) + grid.get_n_sub()


def oracle(
    grid: LSGrid,
    branches: Iterable[int] = (),
    action: Action | None = None,
    apply: Callable[[LSGrid], None] | None = None,
) -> tuple[np.ndarray | None, LSGrid]:
    """lightsim2grid's one-off solve: deep copy, trip ``branches`` (lines then trafos), apply
    ``action`` (a ``gpusim2grid.actions.Action``, through its own ``apply_to``) or ``apply``
    (a callable on the copy, for tests that write the calls out), solve from a flat start.

    Returns ``(V_global, ref_grid)``: V over the ``total_bus()`` global buses (0 where a bus is
    not in service -- lightsim2grid leaves the flat start there), or ``(None, ref_grid)`` when
    the reference diverges."""
    ref = copy.deepcopy(grid)
    nl: int = _n_lines(grid)
    for br in branches:
        if br < nl:
            ref.deactivate_powerline(int(br))
        else:
            ref.deactivate_trafo(int(br) - nl)
    if action is not None:
        action.apply_to(ref)
    if apply is not None:
        apply(ref)
    ref.tell_solver_need_reset()
    V: np.ndarray = ref.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    if V.shape[0] == 0:
        return None, ref
    in_service: np.ndarray = np.asarray(ref.get_bus_status(), dtype=bool)
    return np.where(in_service, V, 0.0), ref


def with_current_limits(grid: LSGrid, factor: float) -> LSGrid:
    """``grid`` with every line / transformer current limit at ``factor`` x its base case
    current (both sides), so that the loadings of the tests are known and finite."""
    for res, setter in (
        (grid.get_line_res1, grid.set_line_current_limit_side1),
        (grid.get_line_res2, grid.set_line_current_limit_side2),
        (grid.get_trafo_res1, grid.set_trafo_current_limit_side1),
        (grid.get_trafo_res2, grid.set_trafo_current_limit_side2),
    ):
        amps_ka: np.ndarray = np.asarray(res()[3], dtype=float)
        setter(factor * np.maximum(amps_ka, 1e-3))
    return grid


def batch_V(ca: ContingencyAnalysisGPU, n_rows: int) -> np.ndarray:
    """The batch's voltages, ``(n_rows, n_batch_bus)``, solver columns."""
    return ca.V_results.to_numpy().reshape(n_rows, -1)


def batch_V_global(ca: ContingencyAnalysisGPU, n_rows: int, total_bus: int) -> np.ndarray:
    """The batch's voltages on global bus ids, ``(n_rows, total_bus)``: NaN where the batch
    has no column for that bus (``ca.batch_bus_ids`` maps a column to its global bus)."""
    V: np.ndarray = batch_V(ca, n_rows)
    out: np.ndarray = np.full((n_rows, total_bus), np.nan + 1j * np.nan)
    out[:, np.asarray(ca.batch_bus_ids, dtype=int)] = V
    return out


def assert_row_matches(V_row_global: np.ndarray, V_ref: np.ndarray | None, atol: float) -> None:
    """Every bus energized in the reference matches; every bus the reference leaves out of
    service is NaN (or absent) in the batch row. ``V_ref`` is ``oracle``'s: ``None`` (the
    reference diverged) fails."""
    assert V_ref is not None, "the lightsim2grid reference diverged"
    on: np.ndarray = np.abs(V_ref) > 0
    np.testing.assert_allclose(V_row_global[on], V_ref[on], atol=atol, rtol=0)
    assert np.all(np.isnan(V_row_global[~on])), "a bus out of service in the reference has a voltage"


def ca_settings() -> list[Any]:
    """The two solver set-ups every feature is checked with (``pytest.param`` each): the
    defaults, and the performance set-up (own LU, continuous scheduling, the tail from the
    start)."""
    return [
        pytest.param({}, id="defaults"),
        pytest.param(dict(linear_solver="own_lu", scheduling="continuous"), id="own_lu_continuous"),
    ]


def make_ca(
    grid: LSGrid, settings: dict[str, Any], batch_size: int | None = None, **kwargs: Any
) -> ContingencyAnalysisGPU:
    """A ContingencyAnalysisGPU with ``settings`` (see ``ca_settings``); the tail starts with the
    first eviction (``tail_max_groups=batch_size``, ``tail_group_size=1``) when continuous."""
    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(grid, nb_iter=MAX_IT, **settings, **kwargs)
    if settings.get("scheduling") == "continuous" and batch_size is not None:
        ca.tail_max_groups = batch_size
        ca.tail_group_size = 1
    return ca
