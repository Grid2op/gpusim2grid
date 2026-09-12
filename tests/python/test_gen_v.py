# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_gen_v.py — InjectionSweepGPU.set_gen_v / ScenarioSweepGPU.set_gen_v.

Mirrors lightsim2grid's own ``modify_gen_v``: unlike every other
``set_injections*`` input, this does NOT feed Sbus -- it only re-seeds |V| at
each voltage-regulating generator's own AC-solver bus right before that row's
solve (a PV/slack bus's magnitude is never an NR unknown, so it never moves
once seeded). See CLAUDE.md's "Session objects and Python facades" section.

Reference: lightsim2grid's own ``LSGrid.change_v_gen`` + ``ac_pf`` on an
independent grid instance -- if gpusim2grid's set_gen_v reproduces that whole
voltage vector, the augmented-Jacobian-vs-bare distinction (whichever this
grid happens to use) doesn't matter; the physics either way is "reseed |V| at
that bus, then solve."
"""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = requires_gpu


def _make_second_grid():
    """A second, independently-constructed+solved IEEE14 grid, so mutating its
    generator setpoints never touches the shared session-scoped ieee14_grid."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    v_init = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
    v_ref = grid.ac_pf(v_init.copy(), 20, 1e-10)
    return grid, v_ref


def _pick_pv_generator(grid, n_bus):
    from gpusim2grid._ls2g_utils import extract_injection_elements

    elements = extract_injection_elements(grid, n_bus)
    pv_buses = set(grid.get_pv().tolist())
    candidates = [g for g in range(elements.n_gen) if elements.gen_bus[g] in pv_buses]
    assert candidates, "no PV-classified generator found on this grid"
    return elements, candidates[0], int(elements.gen_bus[candidates[0]])


@pytest.fixture(scope="module")
def gen_v_grid():
    grid, v_ref = _make_second_grid()
    n_bus = grid.get_bus_vn_kv().shape[0]
    elements, gen_id, bus_id = _pick_pv_generator(grid, n_bus)
    return {
        "grid": grid, "v_ref": v_ref, "n_bus": n_bus,
        "elements": elements, "gen_id": gen_id, "bus_id": bus_id,
    }


def _base_case_injections(grid):
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = grid.get_gen_target_p()
    return load_p[None, :].copy(), load_q[None, :].copy(), gen_p[None, :].copy()


def _cpu_reference_v(grid_data, new_vm):
    """Independent grid instance, generator setpoint changed, re-solved."""
    grid2, v1 = _make_second_grid()
    grid2.change_v_gen(grid_data["gen_id"], float(new_vm))
    return grid2.ac_pf(v1.copy(), 20, 1e-10)


class TestScenarioSweepGenV:
    def test_matches_cpu_reference(self, gen_v_grid, solver_atol):
        from gpusim2grid import ScenarioSweepGPU

        d = gen_v_grid
        old_vm = float(d["grid"].get_generators()[d["gen_id"]].target_vm_pu)
        new_vm = old_vm + 0.03
        v_cpu = _cpu_reference_v(d, new_vm)

        sweep = ScenarioSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p, load_q, gen_p = _base_case_injections(d["grid"])
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[]])

        gen_v = np.full((1, d["elements"].n_gen), np.nan)
        gen_v[0, d["gen_id"]] = new_vm
        sweep.set_gen_v(gen_v)
        sweep.compute(batch_size=4)

        residuals = sweep.last_residuals()
        assert np.all(residuals < 1e-6), residuals

        V_gpu = sweep.solver.V_results.to_numpy().reshape(1, d["n_bus"])[0]
        assert abs(V_gpu[d["bus_id"]]) == pytest.approx(new_vm, abs=1e-6)
        np.testing.assert_allclose(V_gpu, v_cpu, atol=max(solver_atol, 1e-6))

    def test_unset_reproduces_base_case(self, gen_v_grid, solver_atol):
        from gpusim2grid import ScenarioSweepGPU

        d = gen_v_grid
        sweep = ScenarioSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p, load_q, gen_p = _base_case_injections(d["grid"])
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[]])
        sweep.compute(batch_size=4)

        V = sweep.solver.V_results.to_numpy().reshape(1, d["n_bus"])[0]
        np.testing.assert_allclose(V, d["v_ref"], atol=max(solver_atol, 1e-6))

    def test_all_nan_reproduces_base_case(self, gen_v_grid, solver_atol):
        from gpusim2grid import ScenarioSweepGPU

        d = gen_v_grid
        sweep = ScenarioSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p, load_q, gen_p = _base_case_injections(d["grid"])
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[]])
        sweep.set_gen_v(np.full((1, d["elements"].n_gen), np.nan))
        sweep.compute(batch_size=4)

        V = sweep.solver.V_results.to_numpy().reshape(1, d["n_bus"])[0]
        np.testing.assert_allclose(V, d["v_ref"], atol=max(solver_atol, 1e-6))

    def test_row_independent_targets(self, gen_v_grid, solver_atol):
        """Two rows, two different targets: each row's own reseed must not
        leak into the other (exercises ScenarioSweepBatch's active-slot
        permutation of the gen_v override)."""
        from gpusim2grid import ScenarioSweepGPU

        d = gen_v_grid
        old_vm = float(d["grid"].get_generators()[d["gen_id"]].target_vm_pu)
        vm_a, vm_b = old_vm + 0.02, old_vm - 0.02
        v_a = _cpu_reference_v(d, vm_a)
        v_b = _cpu_reference_v(d, vm_b)

        sweep = ScenarioSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p1, load_q1, gen_p1 = _base_case_injections(d["grid"])
        load_p = np.repeat(load_p1, 2, axis=0)
        load_q = np.repeat(load_q1, 2, axis=0)
        gen_p = np.repeat(gen_p1, 2, axis=0)
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[], []])

        gen_v = np.full((2, d["elements"].n_gen), np.nan)
        gen_v[0, d["gen_id"]] = vm_a
        gen_v[1, d["gen_id"]] = vm_b
        sweep.set_gen_v(gen_v)
        sweep.compute(batch_size=4)

        assert np.all(sweep.last_residuals() < 1e-6)
        V = sweep.solver.V_results.to_numpy().reshape(2, d["n_bus"])
        np.testing.assert_allclose(V[0], v_a, atol=max(solver_atol, 1e-6))
        np.testing.assert_allclose(V[1], v_b, atol=max(solver_atol, 1e-6))

    def test_gen_bus_length_mismatch_raises(self, gen_v_grid):
        from gpusim2grid import ScenarioSweepGPU

        d = gen_v_grid
        sweep = ScenarioSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p, load_q, gen_p = _base_case_injections(d["grid"])
        sweep.set_injections_from_elements(load_p, load_q, gen_p)
        sweep.set_topology([[]])
        with pytest.raises(RuntimeError):
            sweep._inner.set_gen_v(np.zeros((1, d["elements"].n_gen + 1)),
                                   d["elements"].gen_bus)


class TestInjectionSweepGenV:
    def test_matches_cpu_reference(self, gen_v_grid, solver_atol):
        from gpusim2grid import InjectionSweepGPU

        d = gen_v_grid
        old_vm = float(d["grid"].get_generators()[d["gen_id"]].target_vm_pu)
        new_vm = old_vm - 0.025
        v_cpu = _cpu_reference_v(d, new_vm)

        isw = InjectionSweepGPU(d["grid"], nb_iter=8, tol_base=1e-10)
        load_p, load_q, gen_p = _base_case_injections(d["grid"])
        isw.set_injections_from_elements(load_p, load_q, gen_p)

        gen_v = np.full((1, d["elements"].n_gen), np.nan)
        gen_v[0, d["gen_id"]] = new_vm
        isw.set_gen_v(gen_v)
        isw.compute(batch_size=4)

        assert np.all(isw.last_residuals() < 1e-6)
        V_gpu = isw.solver.V_results.to_numpy().reshape(1, d["n_bus"])[0]
        np.testing.assert_allclose(V_gpu, v_cpu, atol=max(solver_atol, 1e-6))

    def test_tuple_mode_rejects_set_gen_v(self, ieee14_base_case):
        """Explicit-array mode has no loads/generators to read gen_bus from."""
        from gpusim2grid import InjectionSweepGPU

        d = ieee14_base_case
        isw = InjectionSweepGPU(
            (d["Ybus"], d["v_init"].copy(), d["Sbus"], d["slack"],
             d["slack_weights"], d["pv"], d["pq"]),
            nb_iter=4, init_from_n_powerflow=False)
        with pytest.raises(RuntimeError):
            isw.set_gen_v(np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# Two connected generators on one bus with different set-points: |V| at a
# bus is unique, so no solution satisfies both. lightsim2grid's set_vm lets
# the last one silently win; gpusim2grid refuses the row instead.
# ---------------------------------------------------------------------------

def _two_gen_case14():
    """case14 with a 2nd generator on the bus of pp gen 2 (bus 5). Returns
    (solved grid, [gen ids on bus 5], solver bus id of bus 5)."""
    pp = pytest.importorskip("pandapower")
    import warnings
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        pp.create_gen(net, bus=5, p_mw=10.0, vm_pu=float(net.gen.vm_pu.iloc[2]),
                      controllable=True, min_q_mvar=-50., max_q_mvar=50.)
        grid = init_from_pandapower(net)
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
    assert grid.ac_pf(v0.copy(), 30, 1e-10).shape[0] > 0
    gens = grid.get_generators()
    shared = [g for g in range(len(gens)) if gens[g].bus_id == 5]
    assert len(shared) == 2
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    return grid, shared, int(np.flatnonzero(buses == 5)[0])


def test_conflicting_gen_v_rows_helper():
    from gpusim2grid._ls2g_utils import conflicting_gen_v_rows
    gen_bus = np.array([0, 3, 3, 3, -1, 5])      # gens 1,2,3 share bus 3; gen 4 off
    fixed = np.array([True, False, False, True, False, False])   # bus 5 is PQ
    nan = float("nan")
    gen_v = np.array([
        [1.0, 1.02, 1.02, nan, 1.5, 1.1],    # equal set-points          -> ok
        [1.0, 1.02, 1.03, nan, 1.5, 1.1],    # 1.02 vs 1.03 on bus 3     -> conflict
        [1.0, 1.02, nan, 1.04, 1.5, 1.1],    # 1.02 vs 1.04              -> conflict
        [1.0, nan, nan, 1.04, 1.5, 1.1],     # one applied               -> ok
        [1.0, 1.02, 1.02 + 1e-10, nan, 1.5, 1.1],   # within tolerance  -> ok
    ])
    np.testing.assert_array_equal(
        conflicting_gen_v_rows(gen_v, gen_bus, fixed),
        [False, True, True, False, False])
    # a generator taken out by a contingency does not take part
    gen_off = np.zeros(gen_v.shape, dtype=bool)
    gen_off[1, 2] = True
    np.testing.assert_array_equal(
        conflicting_gen_v_rows(gen_v, gen_bus, fixed, gen_off=gen_off),
        [False, False, True, False, False])


@requires_gpu
class TestConflictingSetpoints:
    def test_scenario_sweep_row_is_not_simulated(self, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        from gpusim2grid.contingency_analysis._limit_violations import (
            LimitViolationType, ViolationElementType)
        grid, (g_a, g_b), b5 = _two_gen_case14()
        n_bus = grid.get_Ybus_solver().shape[0]
        load_p, load_q = grid.get_loads_res_full()[:2]
        gen_p = np.asarray(grid.get_gen_target_p())
        n = 3
        rep = lambda a: np.repeat(np.asarray(a)[None, :], n, axis=0)   # noqa: E731
        n_gen = len(grid.get_generators())
        gen_v = np.full((n, n_gen), np.nan)
        gen_v[:, [g_a, g_b]] = 1.02
        gen_v[1, g_b] = 1.03                  # row 1: infeasible

        sw = ScenarioSweepGPU(grid, nb_iter=10, tol_base=1e-10)
        sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
        sw.set_gen_v(gen_v)
        sw.set_limits_from_grid()          # case14 has no limits: all NaN
        sw.compute_limit_violations = True
        sw.compute(batch_size=n)
        V = sw.solver.V_results.to_numpy().reshape(n, n_bus)
        assert sw.get_disconnected().tolist() == [0, 1, 0]
        viol = sw.get_violations()
        assert viol[0] == [] and viol[2] == []
        assert len(viol[1]) == 1 and viol[1][0].element_type == ViolationElementType.GRID
        assert viol[1][0].violation_type == LimitViolationType.NOT_SIMULATED
        assert np.all(np.isnan(V[1])) and np.isnan(sw.last_residuals()[1])
        assert np.all(np.isfinite(V[[0, 2]]))
        np.testing.assert_allclose(np.abs(V[0, b5]), 1.02, atol=solver_atol)

        # same set-points again -> the row is back (warm reset of the skip)
        gen_v[1, g_b] = 1.02
        sw.set_gen_v(gen_v)
        sw.compute(batch_size=n)
        assert sw.get_disconnected().tolist() == [0, 0, 0]
        V2 = sw.solver.V_results.to_numpy().reshape(n, n_bus)
        np.testing.assert_allclose(V2[1], V[0], atol=solver_atol)

        # the conflict disappears when one of the two is disconnected
        gen_v[1, g_b] = 1.03
        mask = np.zeros((n, n_gen), dtype=bool)
        mask[1, g_b] = True
        sw.set_gen_v(gen_v)
        sw.set_contingency_gens(mask)
        sw.compute(batch_size=n)
        assert sw.get_disconnected().tolist() == [0, 0, 0]
        V3 = sw.solver.V_results.to_numpy().reshape(n, n_bus)
        np.testing.assert_allclose(np.abs(V3[1, b5]), 1.02, atol=solver_atol)

    def test_injection_sweep_raises(self):
        from gpusim2grid import InjectionSweepGPU
        grid, (g_a, g_b), _ = _two_gen_case14()
        n_gen = len(grid.get_generators())
        gen_v = np.full((2, n_gen), np.nan)
        gen_v[:, [g_a, g_b]] = 1.02
        isw = InjectionSweepGPU(grid, nb_iter=8, tol_base=1e-10)
        isw.set_gen_v(gen_v)                  # equal: fine
        gen_v[1, g_a] = 1.0
        with pytest.raises(ValueError, match=r"rows \[1\]"):
            isw.set_gen_v(gen_v)
