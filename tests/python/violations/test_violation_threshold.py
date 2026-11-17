# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``violation_threshold``: lightsim2grid's knob of the same name and semantics
(``ContingencyAnalysis`` / ``ScenarioSweep.violation_threshold``), on ``ContingencyAnalysisGPU``
and ``ScenarioSweepGPU``.

A float in ``]0, 1]`` (default 1.0, today's behaviour) moving every checked limit towards a
healthy anchor: ``effective = t * limit + (1 - t) * anchor``, anchor 0 for currents, the bus
nominal voltage (clamped into ``[vmin, vmax]``) for both voltage bounds. The reported value and
limit are never rescaled; ``violation_rel_tol`` still applies on top. Lowering it invalidates
computed results, raising it does not. The oracle is lightsim2grid's ``ContingencyAnalysisCPP``
with the same threshold.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from conftest import requires_gpu

if TYPE_CHECKING:
    from lightsim2grid.lightsim2grid_cpp import LSGrid

pytestmark = requires_gpu

MAX_IT: int = 30
TOL: float = 1e-10
#: (element_type, element_id, side, violation_type) of a violation record
ViolationKey = tuple[int, int, int, int]


def _grid() -> LSGrid:
    """IEEE 14 with current limits at 1.15 x the base case's currents and voltage bounds at
    0.97 / 1.05 x nominal: a few violations at t = 1, more as t goes down."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.change_algorithm("NR_KLU")
    V: np.ndarray = grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0
    i_or: np.ndarray = np.asarray(grid.get_line_res1()[3])
    i_ex: np.ndarray = np.asarray(grid.get_line_res2()[3])
    grid.set_line_current_limit_side1(1.15 * i_or)
    grid.set_line_current_limit_side2(1.15 * i_ex)
    vn: np.ndarray = np.asarray(grid.get_bus_vn_kv())
    grid.set_bus_voltage_limits(0.97 * vn, 1.05 * vn)
    return grid


def _solved_copy(grid: LSGrid) -> LSGrid:
    """A deep copy, solved again (a copy does not keep its solver's state)."""
    g = copy.deepcopy(grid)
    g.tell_solver_need_reset()
    V: np.ndarray = g.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0
    return g


def _keys(records: list[Any]) -> set[ViolationKey]:
    return {(int(v.element_type), int(v.element_id), int(v.side), int(v.violation_type)) for v in records}


def _ls2g(grid: LSGrid, threshold: float) -> tuple[list[set[ViolationKey]], list[Any]]:
    """lightsim2grid's violations per N-1 line contingency (in line order) and its "n" ones."""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP

    n_lines: int = len(grid.get_lines())
    ca = ContingencyAnalysisCPP(_solved_copy(grid))
    # flags first: toggling compute_limit_violations clears lightsim2grid's contingency list
    ca.compute_limit_violations = True
    ca.violation_threshold = threshold
    for line_id in range(n_lines):
        ca.add_n1(line_id)
    defaults: list[list[int]] = [sorted(int(x) for x in d) for d in ca.my_defaults()]
    order: list[int] = [defaults.index([b]) for b in range(n_lines)]
    ca.compute(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL)
    rows: list[Any] = ca.get_violations()
    return [_keys(rows[i]) for i in order], list(ca.get_violations_n())


def _gpu(grid: LSGrid, threshold: float, cls_name: str = "ContingencyAnalysisGPU") -> Any:
    import gpusim2grid

    n_lines: int = len(grid.get_lines())
    obj = getattr(gpusim2grid, cls_name)(
        grid, nb_iter=MAX_IT, compute_limit_violations=True, violation_threshold=threshold
    )
    obj.solver.violation_capacity = 128  # every record (the default keeps the 16 most severe per type)
    if cls_name == "ContingencyAnalysisGPU":
        obj.add_contingencies_by_branch_id([[b] for b in range(n_lines)])
    else:
        load_p: np.ndarray = np.asarray(grid.get_load_target_p(), dtype=float)
        load_q: np.ndarray = np.array([ld.target_q_mvar for ld in grid.get_loads()], dtype=float)
        gen_p: np.ndarray = np.asarray(grid.get_gen_target_p(), dtype=float)
        obj.set_injections_from_elements(
            np.tile(load_p, (n_lines, 1)), np.tile(load_q, (n_lines, 1)), np.tile(gen_p, (n_lines, 1))
        )
        obj.set_topology([[b] for b in range(n_lines)])
    obj.set_limits_from_grid()
    obj.compute(batch_size=n_lines)
    return obj


@pytest.mark.parametrize("cls_name", ["ContingencyAnalysisGPU", "ScenarioSweepGPU"])
def test_argument(cls_name: str):
    """A constructor argument, default 1.0, validated to ]0, 1]."""
    import gpusim2grid

    cls = getattr(gpusim2grid, cls_name)
    grid = _grid()
    assert cls(grid, compute_limit_violations=True).violation_threshold == 1.0
    for bad in (0.0, -0.5, 1.0 + 1e-9, 2.0, "x"):
        with pytest.raises(ValueError):
            cls(grid, compute_limit_violations=True, violation_threshold=bad)
    assert cls(grid, compute_limit_violations=True, violation_threshold=0.8).violation_threshold == 0.8


@pytest.mark.parametrize("threshold", [1.0, 0.9, 0.75])
def test_matches_lightsim2grid(threshold: float):
    """Every N-1 line contingency of IEEE 14: the same violation records as lightsim2grid's
    with the same threshold."""
    grid = _grid()
    ref, _ = _ls2g(grid, threshold)
    ca = _gpu(grid, threshold)
    got: list[set[ViolationKey]] = [_keys(r) for r in ca.get_violations()]
    assert got == ref


def test_lower_threshold_is_stricter():
    grid = _grid()
    counts: dict[float, np.ndarray] = {}
    for t in (1.0, 0.8):
        c = _gpu(grid, t).get_violation_counts()
        counts[t] = c["current"] + c["low_voltage"] + c["high_voltage"]
    assert np.all(counts[0.8] >= counts[1.0])
    assert counts[0.8].sum() > counts[1.0].sum()


def test_reported_limit_is_not_rescaled():
    """A CURRENT record carries the configured limit, not threshold x limit."""
    from gpusim2grid.contingency_analysis._limit_violations import LimitViolationType

    grid = _grid()
    ca = _gpu(grid, 0.8)
    lim1: np.ndarray = np.array([ln.limit_a1_ka for ln in grid.get_lines()])
    cur = [
        v
        for row in ca.get_violations()
        for v in row
        if v.violation_type == LimitViolationType.CURRENT and v.side == 1
    ]
    assert cur
    for v in cur:
        np.testing.assert_allclose(v.limit, lim1[v.element_id], rtol=1e-12)
        assert v.value > 0.8 * v.limit


def test_voltage_anchor_is_the_clamped_nominal_voltage():
    """A bus whose band does not bracket its nominal voltage: the anchor is clamped into the
    band (lightsim2grid's rule), checked on the "n" case against lightsim2grid."""
    grid = _grid()
    vn: np.ndarray = np.asarray(grid.get_bus_vn_kv())
    vmin, vmax = 0.97 * vn, 1.05 * vn
    vmin[3], vmax[3] = 1.01 * vn[3], 1.06 * vn[3]  # band above nominal at bus 3
    grid.set_bus_voltage_limits(vmin, vmax)
    for t in (1.0, 0.5):
        _, ref_n = _ls2g(grid, t)
        ca = _gpu(grid, t)
        assert _keys(ca.get_violations_n()) == _keys(ref_n)


def test_scenario_sweep_matches_contingency_analysis():
    grid = _grid()
    a = _gpu(grid, 0.85)
    b = _gpu(grid, 0.85, "ScenarioSweepGPU")
    assert [_keys(r) for r in a.get_violations()] == [_keys(r) for r in b.get_violations()]
