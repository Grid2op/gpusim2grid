# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ScenarioSweepGPU: the batch driver persists across compute() calls.

compute() picks one of three paths against the live driver (see
ScenarioSweepSession::run in scenario_sweep_session.cu):

  cold : first call, or n_scenarios / batch_size / strategy / cuDSS config /
         base state changed -> new driver (allocation + cuDSS ANALYSIS + a
         first FACTORIZATION on the first iteration);
  warm : only the topology (or the generator mask) changed -> new batch source
         on the live driver (CPU connectivity + patch upload), REFACTORIZE only;
  hot  : only injections / gen_v changed -> one device gather of the new rows.

The observable proof is in the counters (driver_build_counter /
source_build_counter) and in the one-time timing fields, which a reused driver
reports as 0 (t_analysis_ms, t_alloc_ms, t_first_factorize); results must be
identical to a fresh session's on every path.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge
from test_handle_disconnected_grid import _solved_spur_grid
from test_timings_consistency import _check_to_dict


def _check_batch_aggregation(t):
    """Like test_timings_consistency's helper, but valid on the bridge path too
    (t_ground_truth_check_ms is part of the CPU preprocessing bucket there)."""
    assert t.t_cpu_preprocess_ms == pytest.approx(
        t.t_preprocess_ms + t.t_ground_truth_check_ms, abs=1e-9)
    assert t.t_host_to_device_ms == pytest.approx(
        t.t_alloc_ms + t.t_source_init_ms + t.t_branch_data_upload_ms
        + t.t_violation_setup_ms, abs=1e-9)
    assert t.t_gpu_compute_ms == pytest.approx(
        t.t_base_case_solve_only_ms + t.t_analysis_ms + t.t_chunks_total_wall_ms,
        abs=1e-9)
    assert t.t_grand_total_ms == pytest.approx(
        t.t_cpu_preprocess_ms + t.t_host_to_device_ms + t.t_context_init_ms
        + t.t_gpu_compute_ms + t.t_device_to_host_ms, abs=1e-9)

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge, reason="needs the lightsim2grid C++ bridge")

NB_ITER = 10
TOL = 1e-10


def _elements(grid):
    load_p, load_q = (np.asarray(a, dtype=np.float64) for a in grid.get_loads_res_full()[:2])
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    return load_p, load_q, gen_p


def _rows(grid, scales):
    load_p, load_q, gen_p = _elements(grid)
    scales = np.asarray(scales, dtype=np.float64)[:, None]
    n = scales.shape[0]
    return (np.tile(load_p, (n, 1)) * scales, np.tile(load_q, (n, 1)) * scales,
            np.tile(gen_p, (n, 1)))


def _fresh(grid, scales, topology=None, gen_v=None, **kw):
    """A brand-new session solving the same rows: the reference for every path."""
    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL, **kw)
    sw.set_injections_from_elements(*_rows(grid, scales))
    if topology is not None:
        sw.set_topology(topology)
    if gen_v is not None:
        sw.set_gen_v(gen_v)
    V = _np(sw.compute(batch_size=len(scales)))
    return V, sw


def _np(capsule):
    torch = pytest.importorskip("torch")
    return torch.from_dlpack(capsule).cpu().numpy().copy()


def _one_time_costs_zero(t):
    assert t.t_analysis_ms == 0.0
    assert t.t_alloc_ms == 0.0
    assert t.t_context_init_ms == 0.0
    assert t.t_first_factorize.gpu_ms == 0.0 and t.t_first_factorize.wall_ms == 0.0


class TestHotPath:
    def test_second_compute_reuses_the_driver(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        scales1, scales2 = [1.0, 1.1, 0.9], [0.95, 1.05, 1.2]

        sw.set_injections_from_elements(*_rows(grid, scales1))
        V1 = _np(sw.compute(batch_size=3))
        t1 = sw.timings
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 1
        assert t1.t_analysis_ms > 0.0
        assert t1.t_first_factorize.wall_ms > 0.0
        assert t1.n_refactorize == NB_ITER - 1
        _check_batch_aggregation(t1)
        _check_to_dict(t1)

        sw.set_injections_from_elements(*_rows(grid, scales2))
        V2 = _np(sw.compute(batch_size=3))
        t2 = sw.timings
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 1
        assert sw.run_counter == 2
        _one_time_costs_zero(t2)
        assert t2.t_source_init_ms == 0.0
        assert t2.n_refactorize == NB_ITER          # refactorize only, no first factorize
        assert t2.t_refactorize.wall_ms > 0.0
        _check_batch_aggregation(t2)
        _check_to_dict(t2)

        Vref1, _ = _fresh(grid, scales1)
        Vref2, _ = _fresh(grid, scales2)
        np.testing.assert_allclose(V1, Vref1, atol=solver_atol)
        np.testing.assert_allclose(V2, Vref2, atol=solver_atol)
        assert not np.allclose(V1, V2)

    def test_results_overwritten_in_place(self, ieee14_base_case):
        """v_results_dlpack aliases memory the next (reusing) compute() overwrites."""
        torch = pytest.importorskip("torch")
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.1]))
        view = torch.from_dlpack(sw.compute(batch_size=2))
        snap = view.clone()
        sw.set_injections_from_elements(*_rows(grid, [0.9, 1.2]))
        sw.compute(batch_size=2)
        assert not torch.equal(view, snap)          # same memory, new values
        assert torch.equal(view, torch.from_dlpack(sw.solver.v_results_dlpack()))

    def test_gen_v_hot_update(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        n_gen = len(grid.get_generators())
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.05]))
        gv1 = np.full((2, n_gen), np.nan); gv1[0, 1] = 1.03
        gv2 = np.full((2, n_gen), np.nan); gv2[1, 2] = 1.02
        sw.set_gen_v(gv1)
        V1 = _np(sw.compute(batch_size=2))
        sw.set_gen_v(gv2)
        V2 = _np(sw.compute(batch_size=2))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 1
        _one_time_costs_zero(sw.timings)
        np.testing.assert_allclose(V1, _fresh(grid, [1.0, 1.05], gen_v=gv1)[0], atol=solver_atol)
        np.testing.assert_allclose(V2, _fresh(grid, [1.0, 1.05], gen_v=gv2)[0], atol=solver_atol)
        # Dropping the override: back to the base-case voltages.
        sw.solver.clear_gen_v()
        V3 = _np(sw.compute(batch_size=2))
        np.testing.assert_allclose(V3, _fresh(grid, [1.0, 1.05])[0], atol=solver_atol)
        assert sw.driver_build_counter == 1

    def test_limit_violations_across_hot_runs(self, ieee14_base_case):
        """The fused violation check (whose per-row sentinels are reset every
        run) stays correct when the driver is reused."""
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        n_lines = len(grid.get_lines())

        def _with_limits(sw):
            n_bus, n_bra = sw.n_bus, sw.n_branches
            sw.solver.set_limits(np.full(n_bus, np.nan), np.full(n_bus, np.nan),
                                 np.full(n_bra, 0.05), np.full(n_bra, 0.05), n_lines)
            sw.compute_limit_violations = True
            return sw

        sw = _with_limits(ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL))
        for scales in ([1.0, 1.1], [1.2, 0.8]):
            sw.set_injections_from_elements(*_rows(grid, scales))
            sw.compute(batch_size=2)
            ref = _with_limits(ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL))
            ref.set_injections_from_elements(*_rows(grid, scales))
            ref.compute(batch_size=2)
            for key in ("low_voltage", "high_voltage", "current"):
                np.testing.assert_array_equal(sw.get_violation_counts()[key],
                                              ref.get_violation_counts()[key])
            # Same records; the values differ by rounding noise only (a
            # refactorized solve vs a first factorization).
            for got, exp in zip(sw.get_violations(), ref.get_violations()):
                assert len(got) == len(exp) > 0
                for g, e in zip(got, exp):
                    assert (g.element_type, g.element_id, g.side, g.violation_type,
                            g.limit) == (e.element_type, e.element_id, e.side,
                                         e.violation_type, e.limit)
                    assert g.value == pytest.approx(e.value, rel=1e-9)
        assert sw.driver_build_counter == 1


class TestWarmPath:
    @needs_bridge
    def test_new_topology_rebuilds_only_the_source(self, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        scales = [1.0, 1.05, 0.95, 1.1]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, scales))

        topo_a = [[], [3], [], [int(spur_line)]]        # row 3 islands the spur bus
        sw.set_topology(topo_a)
        Va = _np(sw.compute(batch_size=4))
        assert list(sw.get_disconnected()) == [0, 0, 0, 1]
        assert np.all(np.isnan(Va[3]))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 1

        topo_b = [[int(spur_line)], [], [5], []]        # row 0 islands, row 3 recovers
        sw.set_topology(topo_b)
        Vb = _np(sw.compute(batch_size=4))
        assert list(sw.get_disconnected()) == [1, 0, 0, 0]
        assert np.all(np.isnan(Vb[0])) and np.all(np.isfinite(Vb[3]))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 2
        t = sw.timings
        _one_time_costs_zero(t)
        assert t.t_preprocess_ms > 0.0 or t.t_source_init_ms >= 0.0   # source rebuilt
        assert t.n_refactorize == NB_ITER
        _check_batch_aggregation(t)

        Vref_a = _fresh(grid, scales, topology=topo_a)[0]
        Vref_b = _fresh(grid, scales, topology=topo_b)[0]
        np.testing.assert_allclose(Va[:3], Vref_a[:3], atol=solver_atol)
        np.testing.assert_allclose(Vb[1:], Vref_b[1:], atol=solver_atol)

    @needs_bridge
    def test_handle_disconnected_grid_warm_path(self, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, spur_bus = _solved_spur_grid(distributed_slack=False)
        me_to_solver = np.asarray(grid.id_me_to_ac_solver())
        spur_solver = int(me_to_solver[spur_bus]) if me_to_solver.size else int(spur_bus)
        scales = [1.0, 1.1]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL,
                              handle_disconnected_grid=True)
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.set_topology([[], []])
        V0 = _np(sw.compute(batch_size=2))
        sw.set_topology([[int(spur_line)], []])
        V1 = _np(sw.compute(batch_size=2))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 2
        assert list(sw.get_disconnected()) == [0, 0]
        assert np.isnan(V1[0, spur_solver]) and np.all(np.isfinite(V1[1]))
        ref0 = _fresh(grid, scales, topology=[[], []], handle_disconnected_grid=True)[0]
        ref1 = _fresh(grid, scales, topology=[[int(spur_line)], []],
                      handle_disconnected_grid=True)[0]
        np.testing.assert_allclose(V0, ref0, atol=solver_atol)
        np.testing.assert_allclose(V1[1], ref1[1], atol=solver_atol)
        finite = np.isfinite(ref1[0])
        np.testing.assert_allclose(V1[0][finite], ref1[0][finite], atol=solver_atol)


class TestColdPath:
    def test_shape_and_config_changes_rebuild(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)

        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.1]))
        sw.compute(batch_size=2)
        assert sw.driver_build_counter == 1

        # n_scenarios changed -> cold
        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.1, 0.9]))
        V = _np(sw.compute(batch_size=3))
        assert sw.driver_build_counter == 2
        assert sw.timings.t_analysis_ms > 0.0
        np.testing.assert_allclose(V, _fresh(grid, [1.0, 1.1, 0.9])[0], atol=solver_atol)

        # strategy changed -> cold
        sw.strategy = "direct_iter0_only"
        sw.compute(batch_size=3)
        assert sw.driver_build_counter == 3

        # reordering changed -> cold
        sw.reordering_alg = "amd"
        sw.compute(batch_size=3)
        assert sw.driver_build_counter == 4

        # nb_iter changed -> NOT a rebuild, but takes effect
        sw.nb_iter = NB_ITER + 3
        sw.compute(batch_size=3)
        assert sw.driver_build_counter == 4
        assert sw.timings.nb_iter == NB_ITER + 3

        # batch_size changed -> cold
        sw.compute(batch_size=2)
        assert sw.driver_build_counter == 5

    @needs_bridge
    def test_fixed_batch_capacity_keeps_one_chunk(self, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        scales = [1.0, 1.05, 0.95, 1.1, 1.02]
        topo = [[int(spur_line)], [], [int(spur_line)], [], []]   # 2 islanded rows
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.solver.fixed_batch_capacity = True
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.set_topology(topo)
        V = _np(sw.compute(batch_size=5))
        t = sw.timings
        assert t.n_chunks == 1
        assert t.chunk_size == 5 and sw.solver.used_batch_size == 5
        assert sw.solver.capacity == 5 and sw.solver.n_active == 3
        np.testing.assert_array_equal(sw.active_to_orig, [1, 3, 4])
        ref = _fresh(grid, scales, topology=topo)[0]
        for r in (1, 3, 4):
            np.testing.assert_allclose(V[r], ref[r], atol=solver_atol)
        assert np.all(np.isnan(V[0])) and np.all(np.isnan(V[2]))


class TestDevicePath:
    def test_set_injections_dlpack_matches_numpy(self, ieee14_base_case, solver_atol):
        torch = pytest.importorskip("torch")
        from gpusim2grid import ScenarioSweepGPU
        from gpusim2grid._gpusim2grid import is_fp32
        from gpusim2grid._ls2g_utils import build_bus_injections
        grid = ieee14_base_case["grid"]
        scales = [1.0, 1.1, 0.9]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        el = sw._elements
        p_mw, q_mvar = build_bus_injections(el, *_rows(grid, scales))
        S = torch.tensor((p_mw + 1j * q_mvar) / el.sn_mva,
                         dtype=torch.complex64 if is_fp32 else torch.complex128,
                         device="cuda")
        sw.solver.set_injections_dlpack(S.__dlpack__(), torch.cuda.current_stream().cuda_stream)
        V_dev = _np(sw.compute(batch_size=3))
        ref = _fresh(grid, scales)[0]
        np.testing.assert_allclose(V_dev, ref, atol=solver_atol)

        # The tensor was consumed by copy: freeing it changes nothing.
        del S
        torch.cuda.synchronize()
        V_again = _np(sw.compute(batch_size=3))
        np.testing.assert_allclose(V_again, ref, atol=solver_atol)
        assert sw.driver_build_counter == 1

        # Validation.
        bad = torch.zeros(3, sw.n_bus + 1, dtype=torch.complex128, device="cuda")
        with pytest.raises(RuntimeError, match="extent"):
            sw.solver.set_injections_dlpack(bad.__dlpack__())
        wrong_dtype = torch.zeros(3, sw.n_bus, dtype=torch.float64, device="cuda")
        with pytest.raises(RuntimeError, match="dtype"):
            sw.solver.set_injections_dlpack(wrong_dtype.__dlpack__())
