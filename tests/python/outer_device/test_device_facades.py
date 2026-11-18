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


def test_scenario_sweep_generator_contingencies(grid: Any):
    """rows disconnecting generators (set_contingency_gens): each row's own ReactiveLimits plan,
    its own DistributedSlack units, a bus whose last generator goes turning PQ"""
    from gpusim2grid import ScenarioSweepGPU

    n_gen = len(grid.get_generators())

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
        mask = np.zeros((N_ROWS, n_gen), dtype=bool)
        for r in range(N_ROWS):
            mask[r, r % n_gen] = True
        ss.set_contingency_gens(mask)
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


@pytest.mark.parametrize("qmax", [(100.0, 3.0, 3.0), (40.0, 100.0, 100.0)])
def test_generator_contingencies_on_a_voltage_control_group(qmax: tuple[float, ...]):
    """every generator tripped in turn, on a grid where three of them hold a remote bus together:
    a tripped controller leaves its group bus' controllers (the row's own ReactiveLimits plan)"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    grid = cpu._ieee14_remote_group(qmax, ("B3-G", "B6-G", "B8-G"), scale=1.05)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, TOL_MVA).size > 0
    n_gen = len(grid.get_generators())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            init_from_n_powerflow=False,
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        for g in range(n_gen):
            ca.add_contingency(gen_ids=[g])
        ca.compute(batch_size=3)
        return ca

    _same_rows(run("host"), run("device"), n_gen)


def _element_rows(grid: Any, n_rows: int) -> dict[str, np.ndarray]:
    """per row: the loads scaled up by 3 % a row, the generators' targets moved (row r: generator
    r % n_gen up by 5 MW)"""
    gen_p = np.array([g.target_p_mw for g in grid.get_generators()])
    load_p = np.asarray(grid.get_load_target_p())
    load_q = np.array([ld.target_q_mvar for ld in grid.get_loads()], dtype=float)
    rows_gen_p = np.tile(gen_p, (n_rows, 1))
    for r in range(n_rows):
        rows_gen_p[r, r % gen_p.size] += 5.0
    scale = 1.0 + 0.03 * np.arange(n_rows)
    return {"gen_p": rows_gen_p, "load_p": np.outer(scale, load_p), "load_q": np.outer(scale, load_q)}


@pytest.mark.parametrize("with_gen_v", [False, True])
def test_scenario_sweep_rows_with_their_own_targets(grid: Any, with_gen_v: bool):
    """set_injections_from_elements: every row its own generator targets (and, with gen_v, its own
    voltage set-points: its own ReactiveLimits targets)"""
    from gpusim2grid import ScenarioSweepGPU

    rows = _element_rows(grid, N_ROWS)
    n_gen = rows["gen_p"].shape[1]

    def run(backend: str) -> Any:
        ss = ScenarioSweepGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        ss.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
        if with_gen_v:
            gen_v = np.full((N_ROWS, n_gen), np.nan)
            for r in range(N_ROWS):
                gen_v[r, (r + 1) % n_gen] = 1.03 + 0.005 * r
            ss.set_gen_v(gen_v)
        ss.set_topology([[i] for i in range(N_ROWS)])
        ss.compute(batch_size=4)
        return ss

    _same_rows(run("host"), run("device"), N_ROWS)


def test_injection_sweep_rows_with_their_own_targets(grid: Any):
    from gpusim2grid import InjectionSweepGPU

    rows = _element_rows(grid, N_ROWS)

    def run(backend: str) -> Any:
        sw = InjectionSweepGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            scheduling="continuous",
            outer_loops_backend=backend,
        )
        sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
        sw.compute(batch_size=4)
        return sw

    _same_rows(run("host"), run("device"), N_ROWS)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_contingency_analysis_dist_newton(init_from_n: bool):
    """NROuterDist_*: the Newton shares the slack by the weights the device writes per slot, a
    unit pushed past its limit parked (resumed rows: parked from the start)"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    grid = cpu._case14_dist()
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    n_br = len(grid.get_lines()) + len(grid.get_trafos())

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
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=6)
        return ca

    _same_rows(run("host"), run("device"), n_br)


@pytest.mark.parametrize("dist", [False, True])
def test_decision_trace(grid: Any, dist: bool):
    """keep_outer_trace: each row's decisions, recorded on the device and flushed every round,
    line by line the host's (tail moves included)"""
    from gpusim2grid import ContingencyAnalysisGPU

    if dist:
        import test_outer_model_cpu as cpu

        grid = cpu._case14_dist()
        assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            handle_disconnected_grid=True,
            keep_outer_trace=True,
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=6)
        return ca

    host, dev = run("host"), run("device")
    _same_rows(host, dev, n_br)
    res_h, res_d = host.get_outer_results(), dev.get_outer_results()
    assert any(r.decisions for r in res_h)
    for k, (rh, rd) in enumerate(zip(res_h, res_d, strict=True)):
        _same_decisions(list(rh.decisions), list(rd.decisions), f"row {k}")


def _same_decisions(host: list[str], dev: list[str], where: str) -> None:
    """the same decisions: the text (loop, iteration, action, taken, element) exactly, the value
    and limit to rounding (two sessions agree to ~1e-15 only, see CLAUDE.md)"""
    import re

    pat = re.compile(r"^(.*), value=(\S+), limit=(\S+)$")
    assert len(dev) == len(host), f"{where}: {len(dev)} decisions, host {len(host)}\n" + "\n".join(dev)
    for h, d in zip(host, dev, strict=True):
        mh, md = pat.match(h), pat.match(d)
        assert mh is not None and md is not None, (h, d)
        assert md.group(1) == mh.group(1), f"{where}: {d} != {h}"
        for j in (2, 3):
            np.testing.assert_allclose(
                float(md.group(j)), float(mh.group(j)), rtol=1e-5, atol=1e-9, err_msg=f"{where}: {d} != {h}"
            )


@pytest.mark.parametrize("init_from_n", [False, True])
def test_shunt_voltage_control(init_from_n: bool):
    """ShuntVoltageControl: the shunt controllers' susceptances patched per slot by the driver from
    the controls the device writes, the sections each row ends with"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    grid = cpu._shunt_grid("B9", 12.8, (3, 1))
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=1e-10 / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=6)
        return ca

    host, dev = run("host"), run("device")
    _same_rows(host, dev, n_br)
    res_h, res_d = host.get_outer_results(), dev.get_outer_results()
    assert any(r.shunt_sections for r in res_h)
    for k, (rh, rd) in enumerate(zip(res_h, res_d, strict=True)):
        assert sorted(rd.shunt_sections) == sorted(rh.shunt_sections), f"row {k}"


@pytest.mark.parametrize("init_from_n", [False, True])
@pytest.mark.parametrize("mode", ["ACTIVE_POWER_CONTROL", "CURRENT_LIMITER"])
def test_phase_control(mode: str, init_from_n: bool):
    """PhaseControl: the shifter's block patched per slot by the driver from the taps and controls the
    device writes (the row opening it included), the taps each row ends with and its flows there"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    p1, i1 = cpu._pst_flow()
    grid = cpu._pst_grid(mode, p1 + 7.0 if mode == "ACTIVE_POWER_CONTROL" else 0.8 * i1)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=1e-10 / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=6)
        ca.compute_flows()
        return ca

    host, dev = run("host"), run("device")
    _same_rows(host, dev, n_br)
    res_h, res_d = host.get_outer_results(), dev.get_outer_results()
    assert any(r.phase_tap for r in res_h)
    for k, (rh, rd) in enumerate(zip(res_h, res_d, strict=True)):
        assert sorted(rd.phase_tap) == sorted(rh.phase_tap), f"row {k}"
    conv = np.asarray(host.get_row_status()) == 0
    for name in ("or_amps", "ex_amps"):
        f_h = getattr(host, name).to_numpy().reshape(n_br, -1)[conv]
        f_d = getattr(dev, name).to_numpy().reshape(n_br, -1)[conv]
        np.testing.assert_allclose(f_d, f_h, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("init_from_n", [False, True])
@pytest.mark.parametrize(
    "grid_kw",
    [
        dict(),
        dict(reactive_limits=True, use_initial_tap_position=True, max_controlled_nominal_voltage=120.0),
    ],
)
def test_transformer_voltage_control(grid_kw: dict[str, Any], init_from_n: bool):
    """TransformerVoltageControl: the ratio tap changer's ratio column switched on by the device,
    the low voltage generators frozen meanwhile, rounded; the taps and flows each row ends with"""
    import test_outer_model_cpu as cpu

    from gpusim2grid import ContingencyAnalysisGPU

    grid = cpu._tvc_grid("T4-9-1", 12.6, 0.1, **grid_kw)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.0 + 0j), MAX_IT, 1e-10).size > 0
    n_br = len(grid.get_lines()) + len(grid.get_trafos())

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=1e-10 / grid.get_sn_mva(),
            init_from_n_powerflow=init_from_n,
            handle_disconnected_grid=True,
            keep_outer_trace=True,
            outer_loops_backend=backend,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
        ca.compute(batch_size=6)
        ca.compute_flows()
        return ca

    host, dev = run("host"), run("device")
    _same_rows(host, dev, n_br)
    res_h, res_d = host.get_outer_results(), dev.get_outer_results()
    assert any(r.ratio_tap for r in res_h)
    for k, (rh, rd) in enumerate(zip(res_h, res_d, strict=True)):
        assert sorted(rd.ratio_tap) == sorted(rh.ratio_tap), f"row {k}"
        _same_decisions(list(rh.decisions), list(rd.decisions), f"row {k}")
    conv = np.asarray(host.get_row_status()) == 0
    for name in ("or_amps", "ex_amps"):
        f_h = getattr(host, name).to_numpy().reshape(n_br, -1)[conv]
        f_d = getattr(dev, name).to_numpy().reshape(n_br, -1)[conv]
        np.testing.assert_allclose(f_d, f_h, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name)
