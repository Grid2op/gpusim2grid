# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``outer_loops_backend='device'`` through the facades: the same rows as ``'host'``.

The batch runs on the real driver (continuous schedule, the inner solves on the GPU) with the
device model deciding; every row must end with the host backend's status, outer and Newton
iteration counts, targets and, on converged rows, voltages to rounding (two sessions are only
reproducible to ~1e-15, see CLAUDE.md)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import MAX_IT, TOL_MVA, case14_outer, needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops, requires_gpu]

N_ROWS: int = 8
VTOL: float = 1e-9


@pytest.fixture(scope="module")
def grid() -> Any:
    return case14_outer()


def _same_rows(host: Any, dev: Any, n_rows: int) -> None:
    st_h, st_d = np.asarray(host.get_row_status()), np.asarray(dev.get_row_status())
    np.testing.assert_array_equal(st_d, st_h)
    conv = st_h == 0
    v_h = host.V_results.to_numpy().reshape(n_rows, -1)
    v_d = dev.V_results.to_numpy().reshape(n_rows, -1)
    np.testing.assert_allclose(v_d[conv], v_h[conv], rtol=0, atol=VTOL, equal_nan=True)
    for k, (rh, rd) in enumerate(zip(host.get_outer_results(), dev.get_outer_results(), strict=True)):
        assert rd.status == rh.status, f"row {k}"
        assert rd.nb_outer_iterations == rh.nb_outer_iterations, f"row {k}"
        assert rd.nb_nr_iterations == rh.nb_nr_iterations, f"row {k}"
        assert dict(rd.gen_target_p).keys() == dict(rh.gen_target_p).keys(), f"row {k}"
        for g, mw in dict(rh.gen_target_p).items():
            assert dict(rd.gen_target_p)[g] == pytest.approx(mw, rel=0, abs=1e-6), f"row {k}, generator {g}"


@pytest.mark.parametrize("init_from_n", [False, True])
def test_contingency_analysis(grid: Any, init_from_n: bool):
    from gpusim2grid import ContingencyAnalysisGPU

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        ca.add_contingencies_by_branch_id([[i] for i in range(N_ROWS)])
        ca.compute(batch_size=3)
        return ca

    _same_rows(run("host"), run("device"), N_ROWS)


def test_scenario_sweep_topologies(grid: Any):
    from gpusim2grid import ScenarioSweepGPU

    def run(backend: str) -> Any:
        ss = ScenarioSweepGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        sn = grid.get_sn_mva()
        sbus = np.asarray(grid.get_Sbus_solver()) * sn
        ss.set_injections(np.tile(sbus.real, (N_ROWS, 1)), np.tile(sbus.imag, (N_ROWS, 1)), sn)
        ss.set_topology([[i] for i in range(N_ROWS)])
        ss.compute(batch_size=4)
        return ss

    _same_rows(run("host"), run("device"), N_ROWS)


def test_contingency_analysis_with_the_tail(grid: Any):
    """every branch of case14, a batch smaller than the rows and a tail from early on: the tail
    compaction moves live rows between slots (the device model's slot moves)"""
    from gpusim2grid import ContingencyAnalysisGPU

    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            handle_disconnected_grid=True,
            linear_solver="own_lu",
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 2
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=8)
        return ca

    host, dev = run("host"), run("device")
    assert dev.timings.n_slot_moves > 0, "the tail compaction moved no row"
    _same_rows(host, dev, n_br)


def test_rows_never_simulated(grid: Any):
    """without handle_disconnected_grid, a contingency islanding a bus is not simulated: its row's
    outer result is the default one on both backends (no iteration, no target)"""
    from gpusim2grid import ContingencyAnalysisGPU

    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            outer_loops_backend=backend,
        )
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=4)
        return ca

    host, dev = run("host"), run("device")
    assert (np.asarray(host.get_row_status()) == 3).any(), "no contingency islands a bus"
    _same_rows(host, dev, n_br)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_template_load_equals_the_host_load(grid: Any, init_from_n: bool, monkeypatch: pytest.MonkeyPatch):
    """a row that changes nothing is loaded on the device from the first such row's slot (M3); in
    check mode every one is also loaded through the host model and compared field by field (the
    device model throws on a difference)"""
    from gpusim2grid import ContingencyAnalysisGPU

    n_br = len(grid.get_lines()) + len(grid.get_trafos())
    monkeypatch.setenv("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE", "1")

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=4)
        return ca

    _same_rows(run("host"), run("device"), n_br)
