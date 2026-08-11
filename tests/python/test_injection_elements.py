# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
test_injection_elements.py — element-level injections for InjectionSweepGPU.

``InjectionSweepGPU.set_injections_from_elements(load_p, load_q, gen_p)`` takes
the same per-element matrices as lightsim2grid's own ``TimeSeries.compute_Vs``
and assembles the per-bus Sbus itself, reading which AC-solver bus each load /
generator sits on.  This validates:

  1. Feeding the grid's own base values reproduces ``get_Sbus_solver()`` exactly.
  2. The element path and the per-bus ``set_injections`` path agree on V.
  3. A perturbed scenario matches a lightsim2grid CPU solve of the same case.
  4. Disconnected loads are ignored whatever their column holds.
  5. Shape / mode misuse raises a clear error.
"""
import numpy as np
import pytest

from conftest import requires_gpu

from gpusim2grid import InjectionSweepGPU
from gpusim2grid._ls2g_utils import (
    build_bus_injections,
    extract_injection_elements,
)

pytestmark = requires_gpu


_NB_ITER = 12    # plenty for the small perturbations used here


def _fresh_grid():
    """A private, solved IEEE-14 grid — safe to mutate.

    The session-scoped ``ieee14_grid`` fixture is shared, so tests that
    deactivate or re-target elements must not touch it.
    """
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    _solve(grid)
    return grid


def _solve(grid):
    """Run the CPU AC power flow; return V in grid-model numbering."""
    n_bus_model = grid.get_bus_vn_kv().shape[0]
    v0 = grid.dc_pf(np.ones(n_bus_model, dtype=complex), 1, 1e-6)
    v = grid.ac_pf(v0.copy(), 20, 1e-10)
    assert v.shape[0] == n_bus_model, "CPU AC power flow diverged"
    return v


def _base_injections(grid):
    """(load_p, load_q, gen_p) as lightsim2grid itself stamped them into Sbus.

    Loads are pure PQ, so their *results* are the targets (0 when
    disconnected).  Generators are not symmetric — a slack gen's ``res_p``
    carries the absorbed slack mismatch — so gen P comes from the targets.
    """
    load_res = grid.get_loads_res_full()
    return (np.asarray(load_res[0], dtype=float),
            np.asarray(load_res[1], dtype=float),
            np.asarray(grid.get_gen_target_p(), dtype=float))


def _to_solver_numbering(grid, v_model):
    """Gather a model-numbered per-bus vector onto AC-solver buses."""
    solver_to_me = np.asarray(grid.id_ac_solver_to_me())
    if solver_to_me.size == 0:
        return v_model      # identity numbering
    return v_model[solver_to_me]


# ---------------------------------------------------------------------------
# 1. Base-case identity (assembly only, no GPU solve)
# ---------------------------------------------------------------------------

def test_base_values_reproduce_base_sbus(ieee14_grid, ieee14_base_case):
    """The grid's own values must rebuild get_Sbus_solver() to round-off."""
    grid = ieee14_grid
    n_bus = ieee14_base_case["n_bus"]
    sn_mva = grid.get_sn_mva()

    elements = extract_injection_elements(grid, n_bus)
    load_p, load_q, gen_p = _base_injections(grid)

    n_scen = 3
    p_mw, q_mvar = build_bus_injections(
        elements,
        np.tile(load_p, (n_scen, 1)),
        np.tile(load_q, (n_scen, 1)),
        np.tile(gen_p, (n_scen, 1)))

    assert p_mw.shape == (n_scen, n_bus)
    assert q_mvar.shape == (n_scen, n_bus)

    sbus_pu = (p_mw + 1j * q_mvar) / sn_mva
    ref = np.asarray(grid.get_Sbus_solver())
    assert np.abs(sbus_pu - ref).max() < 1e-12


def test_scaled_loads_shift_sbus_by_the_expected_amount(ieee14_grid,
                                                        ieee14_base_case):
    """Doubling every load doubles the load part of Sbus, nothing else."""
    grid = ieee14_grid
    n_bus = ieee14_base_case["n_bus"]
    sn_mva = grid.get_sn_mva()

    elements = extract_injection_elements(grid, n_bus)
    load_p, load_q, gen_p = _base_injections(grid)

    p_base, q_base = build_bus_injections(
        elements, load_p[None, :], load_q[None, :], gen_p[None, :])
    p_dbl, q_dbl = build_bus_injections(
        elements, 2.0 * load_p[None, :], 2.0 * load_q[None, :], gen_p[None, :])

    # The delta must be exactly -(one more copy of every load), scattered.
    expected = -(elements.scatter_load
                 @ (load_p[elements.load_sel] + 1j * load_q[elements.load_sel]))
    got = (p_dbl - p_base) + 1j * (q_dbl - q_base)
    assert np.abs(got[0] - expected).max() < 1e-9 * max(1.0, sn_mva)


# ---------------------------------------------------------------------------
# 2. Element path vs per-bus path
# ---------------------------------------------------------------------------

def test_element_path_matches_bus_level_api(ieee14_grid, ieee14_base_case,
                                            solver_atol):
    """Same scenarios expressed both ways must give the same voltages."""
    grid = ieee14_grid
    n_bus = ieee14_base_case["n_bus"]
    sn_mva = grid.get_sn_mva()

    load_p, load_q, gen_p = _base_injections(grid)
    scales = np.array([1.0, 0.95, 1.05, 1.10])
    n_scen = scales.shape[0]
    load_p_s = load_p[None, :] * scales[:, None]
    load_q_s = load_q[None, :] * scales[:, None]
    gen_p_s = np.tile(gen_p, (n_scen, 1))

    # Reference per-bus arrays, assembled by hand from the same snapshot.
    elements = extract_injection_elements(grid, n_bus)
    p_mw, q_mvar = build_bus_injections(elements, load_p_s, load_q_s, gen_p_s)

    sweep_bus = InjectionSweepGPU(grid, nb_iter=_NB_ITER)
    sweep_bus.set_injections(p_mw, q_mvar, sn_mva)
    sweep_bus.compute(batch_size=n_scen)
    v_bus = sweep_bus.V_results.to_numpy().reshape(n_scen, n_bus).copy()

    sweep_el = InjectionSweepGPU(grid, nb_iter=_NB_ITER)
    sweep_el.set_injections_from_elements(load_p_s, load_q_s, gen_p_s)
    sweep_el.compute(batch_size=n_scen)
    v_el = sweep_el.V_results.to_numpy().reshape(n_scen, n_bus).copy()

    assert np.abs(v_el - v_bus).max() <= solver_atol

    # Scenario 0 is the untouched base case.
    assert np.abs(v_el[0] - ieee14_base_case["v_ref"][:n_bus]).max() <= solver_atol


def test_n_loads_n_gens_match_the_grid(ieee14_grid):
    grid = ieee14_grid
    sweep = InjectionSweepGPU(grid, nb_iter=_NB_ITER)
    assert sweep.n_loads == len(grid.get_loads())
    assert sweep.n_gens == len(grid.get_generators())


# ---------------------------------------------------------------------------
# 3. Cross-check against a lightsim2grid CPU solve of the same case
# ---------------------------------------------------------------------------

def test_perturbed_scenario_matches_lightsim2grid(solver_atol):
    """Perturb loads + a non-slack gen; GPU row must match a CPU re-solve."""
    grid = _fresh_grid()
    n_bus = grid.get_Ybus_solver().shape[0]
    load_p, load_q, gen_p = _base_injections(grid)

    rng = np.random.default_rng(0)
    factors = np.exp(rng.normal(0.0, 0.02, size=load_p.shape[0]))
    load_p_new = load_p * factors
    load_q_new = load_q * factors

    gen_is_slack = np.array([g.is_slack for g in grid.get_generators()], dtype=bool)
    gen_p_new = gen_p.copy()
    movable = np.flatnonzero((~gen_is_slack) & (gen_p > 0))
    assert movable.size > 0, "IEEE-14 should have a non-slack generator with P>0"
    gen_p_new[movable] *= 1.02

    # Row 0 = base case, row 1 = the perturbation.
    load_p_s = np.stack([load_p, load_p_new])
    load_q_s = np.stack([load_q, load_q_new])
    gen_p_s = np.stack([gen_p, gen_p_new])

    sweep = InjectionSweepGPU(grid, nb_iter=_NB_ITER)
    sweep.set_injections_from_elements(load_p_s, load_q_s, gen_p_s)
    sweep.compute(batch_size=2)
    v_gpu = sweep.V_results.to_numpy().reshape(2, n_bus).copy()
    assert (sweep.last_residuals() < 1e-6).all()

    # Same case on the CPU, on a private grid so nothing else is disturbed.
    ref_grid = _fresh_grid()
    for i in range(load_p.shape[0]):
        ref_grid.change_p_load(i, load_p_new[i])
        ref_grid.change_q_load(i, load_q_new[i])
    for i in movable:
        ref_grid.change_p_gen(int(i), gen_p_new[i])
    v_ref = _to_solver_numbering(ref_grid, _solve(ref_grid))

    assert np.abs(v_gpu[1] - v_ref[:n_bus]).max() <= solver_atol


# ---------------------------------------------------------------------------
# 4. Disconnected elements
# ---------------------------------------------------------------------------

def test_disconnected_load_column_is_ignored():
    """A deactivated load contributes nothing, whatever its column holds."""
    grid = _fresh_grid()
    grid.deactivate_load(2)
    _solve(grid)

    n_bus = grid.get_Ybus_solver().shape[0]
    sn_mva = grid.get_sn_mva()
    elements = extract_injection_elements(grid, n_bus)
    assert 2 not in elements.load_sel.tolist()

    load_p, load_q, gen_p = _base_injections(grid)
    assert load_p[2] == 0.0 and load_q[2] == 0.0   # results are zeroed too

    p_a, q_a = build_bus_injections(
        elements, load_p[None, :], load_q[None, :], gen_p[None, :])

    garbage_p, garbage_q = load_p.copy(), load_q.copy()
    garbage_p[2], garbage_q[2] = 1e6, -1e6
    p_b, q_b = build_bus_injections(
        elements, garbage_p[None, :], garbage_q[None, :], gen_p[None, :])

    assert np.array_equal(p_a, p_b)
    assert np.array_equal(q_a, q_b)

    # ...and the base case is still reproduced with the load out.
    sbus_pu = (p_a + 1j * q_a) / sn_mva
    assert np.abs(sbus_pu - np.asarray(grid.get_Sbus_solver())).max() < 1e-12


# ---------------------------------------------------------------------------
# 5. Misuse
# ---------------------------------------------------------------------------

def test_shape_validation(ieee14_grid, ieee14_base_case):
    grid = ieee14_grid
    elements = extract_injection_elements(grid, ieee14_base_case["n_bus"])
    load_p, load_q, gen_p = _base_injections(grid)

    with pytest.raises(ValueError, match="must be 2-D"):
        build_bus_injections(elements, load_p, load_q[None, :], gen_p[None, :])

    with pytest.raises(ValueError, match="columns while the grid counts"):
        build_bus_injections(elements, load_p[None, :-1], load_q[None, :-1],
                             gen_p[None, :])

    with pytest.raises(ValueError, match="same number of rows"):
        build_bus_injections(elements, np.tile(load_p, (2, 1)),
                             np.tile(load_q, (2, 1)), gen_p[None, :])

    with pytest.raises(ValueError, match="at least one row"):
        build_bus_injections(elements, load_p[None, :][:0], load_q[None, :][:0],
                             gen_p[None, :][:0])


def test_tuple_grid_mode_rejects_element_injections(ieee14_base_case):
    d = ieee14_base_case
    sweep = InjectionSweepGPU(
        (d["Ybus"], d["v_ref"], d["Sbus"], d["slack"], d["slack_weights"],
         d["pv"], d["pq"]), nb_iter=_NB_ITER)

    with pytest.raises(RuntimeError, match="needs a lightsim2grid grid"):
        sweep.set_injections_from_elements(np.zeros((1, 1)), np.zeros((1, 1)),
                                           np.zeros((1, 1)))
    with pytest.raises(RuntimeError, match="no loads"):
        _ = sweep.n_loads
    with pytest.raises(RuntimeError, match="no generators"):
        _ = sweep.n_gens
