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


class TestStateKeptAcrossRuns:
    """Session state that must survive (or be refreshed by) a reused driver."""

    @needs_bridge
    @pytest.mark.parametrize("change", ["injections", "gen_v"])
    def test_disconnected_flags_survive_a_hot_run(self, change):
        # Only a new batch source recomputes which rows are islanded; a hot
        # run keeps the live source, so it must keep those flags too.
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        scales = [1.0, 1.05, 0.95]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.set_topology([[], [int(spur_line)], []])        # row 1 islands the spur bus
        V1 = _np(sw.compute(batch_size=3))
        assert list(sw.get_disconnected()) == [0, 1, 0]
        assert sw.timings.n_disconnected == 1
        assert np.all(np.isnan(V1[1]))

        if change == "injections":
            sw.set_injections_from_elements(*_rows(grid, [0.9, 1.1, 1.02]))
        else:
            gen_v = np.full((3, len(grid.get_gen_target_p())), np.nan)
            gen_v[:, 0] = 1.02
            sw.set_gen_v(gen_v)
        V2 = _np(sw.compute(batch_size=3))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 1   # hot
        assert np.all(np.isnan(V2[1]))
        assert list(sw.get_disconnected()) == [0, 1, 0]
        assert sw.timings.n_disconnected == 1

    def test_set_branch_data_after_a_run_reaches_the_driver(self, ieee14_base_case, solver_atol):
        # The driver uploads the branch admittances once and survives across
        # runs: a later set_branch_data() must still reach it.
        from gpusim2grid import ScenarioSweepGPU
        from gpusim2grid._ls2g_utils import extract_branch_data
        grid = ieee14_base_case["grid"]
        scales = [1.0, 1.1]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.compute(batch_size=2)
        sw.compute_flows()
        or1 = sw.or_amps.to_numpy().copy()
        ex1 = sw.ex_amps.to_numpy().copy()
        assert np.all(np.isfinite(or1)) and np.any(or1 > 0.0)

        # Branch currents are linear in the admittances; Ybus (hence V) is
        # untouched by set_branch_data, so doubling them doubles the currents.
        (b_from, b_to, yff, yft, ytf, ytt, vn_kv, sn_mva), _, _ = extract_branch_data(grid)
        sw.set_branch_data(b_from, b_to, 2 * np.asarray(yff), 2 * np.asarray(yft),
                           2 * np.asarray(ytf), 2 * np.asarray(ytt), vn_kv, sn_mva)
        sw.compute(batch_size=2)
        assert sw.driver_build_counter == 1                 # same driver
        sw.compute_flows()
        np.testing.assert_allclose(sw.or_amps.to_numpy(), 2 * or1, rtol=solver_atol, atol=solver_atol)
        np.testing.assert_allclose(sw.ex_amps.to_numpy(), 2 * ex1, rtol=solver_atol, atol=solver_atol)

    @needs_bridge
    def test_warm_run_outgrowing_the_cold_capacity_rebuilds_the_driver(self, solver_atol):
        # The cold run sizes the chunk capacity for ITS active rows. A warm
        # run with far more active rows must not be split into one-row chunks.
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        spur = int(spur_line)
        scales = [1.0, 1.05, 0.95, 1.1, 0.9, 1.02]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, scales))

        sw.set_topology([[spur]] * 5 + [[]])               # 1 active row of 6
        sw.compute(batch_size=6)
        assert sw.solver.n_active == 1 and sw.solver.capacity == 1
        assert sw.driver_build_counter == 1

        # mild growth (1 -> 2 active): one extra chunk, no new analysis
        topo_mild = [[spur]] * 4 + [[], []]
        sw.set_topology(topo_mild)
        V_mild = _np(sw.compute(batch_size=6))
        assert sw.driver_build_counter == 1 and sw.source_build_counter == 2
        assert sw.solver.capacity == 1

        # all 6 active: 6 chunks at the live capacity vs 1 fresh -> rebuild
        topo_all = [[]] * 6
        sw.set_topology(topo_all)
        V_all = _np(sw.compute(batch_size=6))
        assert sw.driver_build_counter == 2
        assert sw.solver.n_active == 6 and sw.solver.capacity == 6
        assert list(sw.get_disconnected()) == [0] * 6

        ref_mild = _fresh(grid, scales, topology=topo_mild)[0]
        ref_all = _fresh(grid, scales, topology=topo_all)[0]
        np.testing.assert_allclose(V_mild[4:], ref_mild[4:], atol=solver_atol)
        np.testing.assert_allclose(V_all, ref_all, atol=solver_atol)

    @needs_bridge
    def test_turning_limit_violations_off_disarms_the_reused_driver(self, ieee14_base_case):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL,
                              compute_limit_violations=True)
        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.1]))
        sw.compute(batch_size=2)
        t_on = sw.timings.t_violation_check
        assert t_on.wall_ms > 0.0 or t_on.gpu_ms > 0.0

        sw.compute_limit_violations = False
        sw.compute(batch_size=2)
        assert sw.driver_build_counter == 1                 # same driver
        t_off = sw.timings.t_violation_check
        assert t_off.wall_ms == 0.0 and t_off.gpu_ms == 0.0
        with pytest.raises(RuntimeError, match="compute_limit_violations"):
            sw.get_violations()

    def test_set_branch_data_refuses_ids_the_topology_still_trips(self, ieee14_base_case,
                                                                  solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        from gpusim2grid._ls2g_utils import extract_branch_data
        grid = ieee14_base_case["grid"]
        scales = [1.0, 1.1]
        topo = [[3], []]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.set_topology(topo)
        V1 = _np(sw.compute(batch_size=2))

        args, _, _ = extract_branch_data(grid)
        short = tuple(np.asarray(a)[:3] for a in args[:6]) + tuple(args[6:])
        with pytest.raises(RuntimeError, match="out of range"):
            sw.set_branch_data(*short)

        # refused before anything was replaced: same topology, same answer
        V2 = _np(sw.compute(batch_size=2))
        np.testing.assert_allclose(V2, V1, atol=solver_atol)
        np.testing.assert_allclose(V2, _fresh(grid, scales, topology=topo)[0],
                                   atol=solver_atol)

    @needs_bridge
    def test_set_branch_data_with_more_branches(self, ieee14_base_case, solver_atol):
        # One extra (zero-admittance) branch: the flow buffers are resized,
        # the old per-branch limits are dropped (set_limits() is asked for
        # again instead of reading past their end), the topology still holds.
        from gpusim2grid import ScenarioSweepGPU
        from gpusim2grid._ls2g_utils import extract_branch_data
        grid = ieee14_base_case["grid"]
        topo = [[3], []]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL,
                              compute_limit_violations=True)
        sw.set_injections_from_elements(*_rows(grid, [1.0, 1.1]))
        sw.set_topology(topo)
        V1 = _np(sw.compute(batch_size=2))
        sw.compute_flows()
        or1 = sw.or_amps.to_numpy().reshape(2, -1).copy()
        n_bra = or1.shape[1]

        (b_from, b_to, yff, yft, ytf, ytt, vn_kv, sn_mva), _, _ = extract_branch_data(grid)
        z = np.zeros(1, dtype=np.asarray(yff).dtype)
        sw.set_branch_data(np.append(b_from, 0), np.append(b_to, 1),
                           np.append(yff, z), np.append(yft, z),
                           np.append(ytf, z), np.append(ytt, z), vn_kv, sn_mva)
        with pytest.raises(RuntimeError, match="set_limits"):
            sw.compute(batch_size=2)

        sw.compute_limit_violations = False
        V2 = _np(sw.compute(batch_size=2))
        np.testing.assert_allclose(V2, V1, atol=solver_atol)
        sw.compute_flows()
        or2 = sw.or_amps.to_numpy().reshape(2, -1)
        assert or2.shape == (2, n_bra + 1)
        np.testing.assert_allclose(or2[:, :n_bra], or1, rtol=solver_atol, atol=solver_atol)
        assert np.all(or2[:, n_bra] == 0.0)
        assert or2[0, 3] == 0.0                             # still tripped in row 0

    @needs_bridge
    def test_keep_final_jacobian_rebuilds_when_one_chunk_is_possible(self, solver_atol):
        # keep_final_jacobian needs one chunk. The cold run's capacity (2: two
        # of four rows islanded) would split a later all-active topology in two
        # chunks, which is under the "twice as many chunks" rebuild threshold;
        # the forward must still get the one chunk batch_size allows.
        from gpusim2grid import ScenarioSweepGPU
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        spur = int(spur_line)
        scales = [1.0, 1.05, 0.95, 1.1]
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.solver.keep_final_jacobian = True
        sw.set_injections_from_elements(*_rows(grid, scales))
        sw.set_topology([[spur], [spur], [], []])
        sw.compute(batch_size=4)
        assert sw.solver.capacity == 2

        topo = [[]] * 4
        sw.set_topology(topo)
        V = _np(sw.compute(batch_size=4))                # used to raise
        assert sw.driver_build_counter == 2 and sw.solver.capacity == 4
        np.testing.assert_allclose(V, _fresh(grid, scales, topology=topo)[0],
                                   atol=solver_atol)
