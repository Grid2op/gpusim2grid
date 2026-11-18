# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Remedial actions with ``outer_loops_backend='device'``: strategies warm-started from their
contingency (``strategy_warm_start``), each strategy row resuming from where its contingency's
loops ended, decided on the device -- the same rows as ``'host'``, a strategy tripping a
generator (its row's own ReactiveLimits plan) included."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import MAX_IT, TOL_MVA, case14_outer, needs_outer_loops

from gpusim2grid.actions import Action

from conftest import requires_gpu

pytestmark = [needs_outer_loops, requires_gpu]

CONTINGENCIES: list[list[int]] = [[3], [5], [2]]
ACTIONS: list[Action] = [
    Action(load_bus={2: -1}),  # a load shed
    Action(line_bus1={13: -1}, line_bus2={13: -1}),  # a line opened
]
VTOL: float = 1e-9


@pytest.fixture(scope="module")
def grid() -> Any:
    return case14_outer()


def _run(grid: Any, backend: str, actions: list[Action], warm: bool = True) -> Any:
    from gpusim2grid import ContingencyAnalysisGPU

    ca = ContingencyAnalysisGPU(
        grid,
        outer_loops=True,
        nb_iter=MAX_IT,
        tol=TOL_MVA / grid.get_sn_mva(),
        init_from_n_powerflow=False,
        scheduling="continuous",
        strategy_warm_start=warm,
        outer_loops_backend=backend,
    )
    ca.tail_max_groups, ca.tail_group_size = 4, 1
    ca.add_contingencies_by_branch_id(CONTINGENCIES)
    ca.set_actions(actions)
    ca.add_strategies([(c, [a]) for c in range(len(CONTINGENCIES)) for a in range(len(actions))])
    ca.compute(batch_size=4)
    return ca


@pytest.mark.parametrize("warm", [True, False])
def test_strategies_device_equals_host(grid: Any, warm: bool):
    host, dev = _run(grid, "host", ACTIONS, warm), _run(grid, "device", ACTIONS, warm)
    st_h, st_d = np.asarray(host.get_row_status()), np.asarray(dev.get_row_status())
    np.testing.assert_array_equal(st_d, st_h)
    n_rows = st_h.size
    assert n_rows == len(CONTINGENCIES) * (1 + len(ACTIONS))
    conv = st_h == 0
    v_h = host.V_results.to_numpy().reshape(n_rows, -1)
    v_d = dev.V_results.to_numpy().reshape(n_rows, -1)
    np.testing.assert_allclose(v_d[conv], v_h[conv], rtol=0, atol=VTOL, equal_nan=True)
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"


def test_a_strategy_tripping_a_generator(grid: Any):
    """the action gives its row its own ReactiveLimits plan (the generator leaves its bus' machines,
    or the bus leaves the plan): per-slot targets / machines on the device, the same rows"""
    actions = [Action(gen_bus={1: -1}), Action(gen_bus={2: -1})]
    host, dev = _run(grid, "host", actions), _run(grid, "device", actions)
    st_h = np.asarray(host.get_row_status())
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), st_h)
    n_rows = st_h.size
    conv = st_h == 0
    np.testing.assert_allclose(
        dev.V_results.to_numpy().reshape(n_rows, -1)[conv],
        host.V_results.to_numpy().reshape(n_rows, -1)[conv],
        rtol=0,
        atol=VTOL,
        equal_nan=True,
    )
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"


def test_strategy_rows_from_their_point_template(grid: Any, monkeypatch: pytest.MonkeyPatch):
    """a strategy row is a device copy of its point's template (the first strategy row of that
    contingency, loaded through the host model); in check mode every one is also loaded through
    the host and compared field by field (the device model throws on a difference)"""
    monkeypatch.setenv("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE", "1")
    host, dev = _run(grid, "host", ACTIONS), _run(grid, "device", ACTIONS)
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), np.asarray(host.get_row_status()))


def test_strategies_on_a_dist_newton():
    """NROuterDist_*: a strategy row resumes with the units its contingency left parked"""
    import test_outer_model_cpu as cpu

    grid = cpu._case14_dist()
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    host, dev = _run(grid, "host", ACTIONS), _run(grid, "device", ACTIONS)
    st_h = np.asarray(host.get_row_status())
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), st_h)
    conv = st_h == 0
    n_rows = st_h.size
    np.testing.assert_allclose(
        dev.V_results.to_numpy().reshape(n_rows, -1)[conv],
        host.V_results.to_numpy().reshape(n_rows, -1)[conv],
        rtol=0,
        atol=VTOL,
    )
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"


def test_strategies_on_shunt_voltage_control():
    """a strategy row resumes with the sections its contingency's ShuntVoltageControl left"""
    import test_outer_model_cpu as cpu

    grid = cpu._shunt_grid("B9", 12.8, (3, 1))
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    host, dev = _run(grid, "host", ACTIONS), _run(grid, "device", ACTIONS)
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), np.asarray(host.get_row_status()))
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"
        assert sorted(rd.shunt_sections) == sorted(rh.shunt_sections), f"row {k}"


@pytest.mark.parametrize("which", ["phase", "ratio"])
def test_strategies_on_tap_changers(which: str):
    """a strategy row resumes with the taps its contingency's PhaseControl / TransformerVoltageControl left"""
    import test_outer_model_cpu as cpu

    if which == "phase":
        p1, _ = cpu._pst_flow()
        grid = cpu._pst_grid("ACTIVE_POWER_CONTROL", p1 + 7.0)
    else:
        grid = cpu._tvc_grid("T4-9-1", 12.4, 0.05)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    host, dev = _run(grid, "host", ACTIONS), _run(grid, "device", ACTIONS)
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), np.asarray(host.get_row_status()))
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"
        assert sorted(rd.phase_tap) == sorted(rh.phase_tap), f"row {k}"
        assert sorted(rd.ratio_tap) == sorted(rh.ratio_tap), f"row {k}"
