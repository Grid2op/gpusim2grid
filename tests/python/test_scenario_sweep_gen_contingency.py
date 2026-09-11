# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ScenarioSweepGPU.set_contingency_gens — per-row generator disconnection.

The GPU counterpart of lightsim2grid's ``ScenarioSweep.set_contingency_gens``
(PR #193): a ``(n_scenarios, n_gen)`` bool mask takes a generator's injection
out of its row, re-weights the distributed slack without it, and -- when the
LAST generator locally regulating a bus goes -- turns that bus from PV to PQ
for that row only. The Jacobian pattern is the union over the rows (one Vm
column + Q equation reserved per bus that can flip, derived from the mask and
rebuilt once when that set changes); rows where the bus is still PV identity-
pin its Q row.

Oracles: a one-off ``ac_pf`` on a fresh grid with the generators actually
deactivated (lightsim2grid's own test oracle), and lightsim2grid's
``ScenarioSweepCPP.set_contingency_gens`` where the installed build has it.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = requires_gpu

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge,
    reason="generator contingencies need the lightsim2grid C++ bridge")

MAX_IT, TOL = 20, 1e-10


def _make_grid():
    """A fresh, solved case14 (NR_KLU). Returns (grid, v_ref)."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = init_from_pandapower(pn.case14())
    grid.change_algorithm(AlgorithmType.NR_KLU)
    n_bus = grid.get_bus_vn_kv().shape[0]
    v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
    v_ref = grid.ac_pf(v0.copy(), MAX_IT, TOL)
    assert v_ref.shape[0] > 0
    return grid, v_ref


def _reference_V(gens_off, lines_off=(), gen_p=None, load_p=None, load_q=None):
    """One-off powerflow on a fresh grid with those elements really removed."""
    grid, v1 = _make_grid()
    if gen_p is not None:
        for g, p in enumerate(gen_p):
            grid.change_p_gen(int(g), float(p))
    if load_p is not None:
        for l, p in enumerate(load_p):
            grid.change_p_load(int(l), float(p))
    if load_q is not None:
        for l, q in enumerate(load_q):
            grid.change_q_load(int(l), float(q))
    for g in gens_off:
        grid.deactivate_gen(int(g))
    for l in lines_off:
        grid.deactivate_powerline(int(l))
    grid.tell_solver_need_reset()
    V = grid.ac_pf(v1.copy(), MAX_IT, TOL)
    assert V.shape[0] > 0, "the reference powerflow itself diverged"
    return V


@pytest.fixture(scope="module")
def case():
    grid, v_ref = _make_grid()
    gens = grid.get_generators()
    n_gen = len(gens)
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=np.float64)
    return {
        "grid": grid, "v_ref": v_ref,
        "n_bus": grid.get_Ybus_solver().shape[0],
        "buses": np.asarray(grid.id_ac_solver_to_me(), dtype=int),
        "n_gen": n_gen,
        "bus_of_gen": [g.bus_id for g in gens],
        "slack_gens": [g for g in range(n_gen) if gens[g].is_slack],
        "load_p": np.asarray(load_p, dtype=np.float64),
        "load_q": np.asarray(load_q, dtype=np.float64),
        "gen_p": gen_p,
        "n_line": len(grid.get_lines()),
    }


def _sweep(case, n_rows, gen_mask=None, topology=None, **kwargs):
    from gpusim2grid import ScenarioSweepGPU

    sw = ScenarioSweepGPU(case["grid"], nb_iter=10, tol_base=TOL, **kwargs)
    rep = lambda a: np.repeat(a[None, :], n_rows, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(case["load_p"]), rep(case["load_q"]),
                                    rep(case["gen_p"]))
    sw.set_topology(topology if topology is not None else [[] for _ in range(n_rows)])
    if gen_mask is not None:
        sw.set_contingency_gens(gen_mask)
    return sw


def _V(sw, n_rows, n_bus):
    return sw.solver.V_results.to_numpy().reshape(n_rows, n_bus)


def _assert_row(case, V_row, gens_off, lines_off=(), atol=1e-6, **inj):
    ref = _reference_V(gens_off, lines_off, **inj)
    np.testing.assert_allclose(V_row, ref[case["buses"]], atol=atol,
                               err_msg=f"generators {list(gens_off)} off")


# --------------------------------------------------------------------- tests
@requires_gpu
@needs_bridge
def test_gen_n1_matches_one_off_powerflow(case, solver_atol):
    """One row per non-slack generator, each disconnecting it."""
    rows = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    mask = np.zeros((len(rows), case["n_gen"]), dtype=bool)
    for r, g in enumerate(rows):
        mask[r, g] = True
    sw = _sweep(case, len(rows), mask)
    sw.compute(batch_size=4)
    assert np.all(sw.last_residuals() < 100 * solver_atol)
    V = _V(sw, len(rows), case["n_bus"])
    for r, g in enumerate(rows):
        _assert_row(case, V[r], [g], atol=10 * solver_atol)
    # every disconnected generator was the only one on its bus -> all reserved
    reserved = set(sw.reserved_switchable_buses.tolist())
    assert reserved == {case["bus_of_gen"][g] for g in rows}
    assert sw.dim_J == sw.solver._s.dim_J


@requires_gpu
@needs_bridge
def test_matches_lightsim2grid_scenario_sweep(case, solver_atol):
    """Same rows through lightsim2grid's own ScenarioSweepCPP."""
    ScenarioSweepCPP = pytest.importorskip("lightsim2grid.scenarioSweep").ScenarioSweepCPP
    if not hasattr(ScenarioSweepCPP, "set_contingency_gens"):
        pytest.skip("this lightsim2grid build has no set_contingency_gens")
    rows = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    mask = np.zeros((len(rows), case["n_gen"]), dtype=bool)
    for r, g in enumerate(rows):
        mask[r, g] = True
    sw = _sweep(case, len(rows), mask)
    sw.compute(batch_size=4)
    V = _V(sw, len(rows), case["n_bus"])

    ls = ScenarioSweepCPP(case["grid"])
    ls.set_contingency_gens(mask)
    ls.compute(1.0 * case["v_ref"], MAX_IT, TOL)
    assert all(ls.converged_mask())
    np.testing.assert_allclose(V, np.asarray(ls.get_voltages())[:, case["buses"]],
                               atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_bus_stays_pv_while_one_gen_remains(solver_atol):
    """Two generators on one bus: one off keeps it PV, both off turns it PQ."""
    pp = pytest.importorskip("pandapower")
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    from gpusim2grid import ScenarioSweepGPU

    def make():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = pn.case14()
            # a 2nd generator on bus 5 (same bus as gen 2), same setpoint
            pp.create_gen(net, bus=5, p_mw=10.0, vm_pu=float(net.gen.vm_pu.iloc[2]),
                          controllable=True, min_q_mvar=-50., max_q_mvar=50.)
            grid = init_from_pandapower(net)
        grid.change_algorithm(AlgorithmType.NR_KLU)
        n_bus = grid.get_bus_vn_kv().shape[0]
        v0 = grid.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
        v_ref = grid.ac_pf(v0.copy(), MAX_IT, TOL)
        assert v_ref.shape[0] > 0
        return grid, v_ref

    grid, v_ref = make()
    gens = grid.get_generators()
    n_gen = len(gens)
    shared = [g for g in range(n_gen) if gens[g].bus_id == 5]
    assert len(shared) == 2
    g_a, g_b = shared
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    n_bus = grid.get_Ybus_solver().shape[0]

    mask = np.zeros((3, n_gen), dtype=bool)
    mask[0, g_a] = True
    mask[1, g_b] = True
    mask[2, [g_a, g_b]] = True

    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p())
    sw = ScenarioSweepGPU(grid, nb_iter=10, tol_base=TOL)
    rep = lambda a: np.repeat(np.asarray(a)[None, :], 3, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    assert sw.reserved_switchable_buses.tolist() == [5]
    V = _V(sw, 3, n_bus)
    assert np.all(sw.last_residuals() < 100 * solver_atol)

    for row, gens_off in enumerate([[g_a], [g_b], [g_a, g_b]]):
        g2, v1 = make()
        for g in gens_off:
            g2.deactivate_gen(int(g))
        g2.tell_solver_need_reset()
        ref = g2.ac_pf(v1.copy(), MAX_IT, TOL)
        assert ref.shape[0] > 0
        np.testing.assert_allclose(V[row], ref[buses], atol=10 * solver_atol)

    # the distinction is real: |V| pinned in rows 0/1, free in row 2
    b = int(np.flatnonzero(buses == 5)[0])
    assert abs(V[0][b]) == pytest.approx(abs(V[1][b]), abs=1e-8)
    assert abs(abs(V[0][b]) - abs(V[2][b])) > 1e-4


@requires_gpu
@needs_bridge
def test_empty_mask_is_bit_identical(case):
    """An all-False mask must reproduce a sweep that never set one, exactly."""
    n = 3
    scale = np.array([0.9, 1.0, 1.1])
    from gpusim2grid import ScenarioSweepGPU

    def run(with_mask):
        sw = ScenarioSweepGPU(case["grid"], nb_iter=8, tol_base=TOL)
        sw.set_injections_from_elements(
            scale[:, None] * case["load_p"][None, :],
            scale[:, None] * case["load_q"][None, :],
            np.repeat(case["gen_p"][None, :], n, axis=0))
        if with_mask:
            sw.set_contingency_gens(np.zeros((n, case["n_gen"]), dtype=bool))
        sw.compute(batch_size=4)
        return _V(sw, n, case["n_bus"]), sw.dim_J, sw.reserved_switchable_buses

    V0, d0, r0 = run(False)
    V1, d1, r1 = run(True)
    # same structure (no bus reserved, no rebuild) ...
    assert d0 == d1 and r0.size == 0 and r1.size == 0
    # ... and the same numbers, up to cuDSS's own run-to-run reproducibility
    # (~1e-15 even without any mask; nothing else may differ here).
    np.testing.assert_allclose(V0, V1, atol=1e-13, rtol=0)


@requires_gpu
@needs_bridge
def test_rebuild_on_change(case, solver_atol):
    """The reserved set follows the mask: grows, shrinks, back to nothing."""
    n_bus = case["n_bus"]
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    g_a, g_b = non_slack[0], non_slack[1]
    sw = _sweep(case, 2)
    dim0 = sw.dim_J

    mask = np.zeros((2, case["n_gen"]), dtype=bool)
    mask[0, g_a] = True
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    assert sw.reserved_switchable_buses.tolist() == [case["bus_of_gen"][g_a]]
    assert sw.dim_J == dim0 + 1
    V = _V(sw, 2, n_bus)
    _assert_row(case, V[0], [g_a], atol=10 * solver_atol)
    _assert_row(case, V[1], [], atol=10 * solver_atol)

    mask[1, g_b] = True
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    assert sw.reserved_switchable_buses.tolist() == sorted(
        [case["bus_of_gen"][g_a], case["bus_of_gen"][g_b]])
    assert sw.dim_J == dim0 + 2
    V = _V(sw, 2, n_bus)
    _assert_row(case, V[0], [g_a], atol=10 * solver_atol)
    _assert_row(case, V[1], [g_b], atol=10 * solver_atol)

    sw.set_contingency_gens(np.zeros((2, case["n_gen"]), dtype=bool))
    sw.compute(batch_size=4)
    assert sw.reserved_switchable_buses.size == 0 and sw.dim_J == dim0
    V = _V(sw, 2, n_bus)
    np.testing.assert_allclose(V[0], case["v_ref"][case["buses"]], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_with_injections(case, solver_atol):
    """This row's own injections AND this row's own generator out."""
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    n = 4
    rng = np.random.default_rng(0)
    scale = 1.0 + 0.1 * rng.uniform(-1, 1, size=(n, 1))
    load_p = scale * case["load_p"][None, :]
    load_q = scale * case["load_q"][None, :]
    gen_p = np.repeat(case["gen_p"][None, :], n, axis=0)
    mask = np.zeros((n, case["n_gen"]), dtype=bool)
    for r in range(n):
        mask[r, non_slack[r % len(non_slack)]] = True

    from gpusim2grid import ScenarioSweepGPU
    sw = ScenarioSweepGPU(case["grid"], nb_iter=10, tol_base=TOL)
    # mask BEFORE the injections: the facade must re-assemble either way
    sw.set_contingency_gens(mask)
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    sw.compute(batch_size=4)
    V = _V(sw, n, case["n_bus"])
    for r in range(n):
        _assert_row(case, V[r], [non_slack[r % len(non_slack)]], atol=10 * solver_atol,
                    gen_p=gen_p[r], load_p=load_p[r], load_q=load_q[r])


@requires_gpu
@needs_bridge
def test_combined_with_line_contingency(case, solver_atol):
    """A line trip and a generator disconnection on the same row."""
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    g = non_slack[1]
    line = 5
    mask = np.zeros((2, case["n_gen"]), dtype=bool)
    mask[0, g] = True
    sw = _sweep(case, 2, mask, topology=[[line], [line]])
    sw.compute(batch_size=4)
    assert np.all(sw.get_disconnected() == 0)
    V = _V(sw, 2, case["n_bus"])
    _assert_row(case, V[0], [g], [line], atol=10 * solver_atol)
    _assert_row(case, V[1], [], [line], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_only_slack_gen_off_matches_lightsim2grid(case, solver_atol):
    """Disconnecting the ONLY slack generator: its weight leaves, the reference
    bus keeps the whole share (the slack bus SET never moves). A plain one-off
    ac_pf has no slack left and diverges, so the oracle is lightsim2grid's own
    sweep, which applies the same contract."""
    ScenarioSweepCPP = pytest.importorskip("lightsim2grid.scenarioSweep").ScenarioSweepCPP
    if not hasattr(ScenarioSweepCPP, "set_contingency_gens"):
        pytest.skip("this lightsim2grid build has no set_contingency_gens")
    g = case["slack_gens"][0]
    mask = np.zeros((1, case["n_gen"]), dtype=bool)
    mask[0, g] = True
    sw = _sweep(case, 1, mask)
    sw.compute(batch_size=4)
    assert sw.last_residuals()[0] < 100 * solver_atol
    V = _V(sw, 1, case["n_bus"])

    ls = ScenarioSweepCPP(case["grid"])
    ls.set_contingency_gens(mask)
    ls.compute(1.0 * case["v_ref"], MAX_IT, TOL)
    assert ls.converged_mask()[0]
    np.testing.assert_allclose(V[0], np.asarray(ls.get_voltages())[0][case["buses"]],
                               atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_distributed_slack_participant_off(solver_atol):
    """Two slack participants: disconnecting one re-weights the distributed
    slack onto the survivor, exactly like a one-off solve without it."""
    from test_augmented_features import _solved_multislack_grid
    from gpusim2grid import ScenarioSweepGPU

    grid = _solved_multislack_grid()
    gens = grid.get_generators()
    n_gen = len(gens)
    slack_gens = [g for g in range(n_gen) if gens[g].is_slack and gens[g].slack_weight != 0]
    assert len(slack_gens) == 2
    n_bus = grid.get_Ybus_solver().shape[0]
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p())
    V_n = grid.get_V()

    mask = np.zeros((2, n_gen), dtype=bool)
    mask[0, slack_gens[0]] = True
    mask[1, slack_gens[1]] = True
    sw = ScenarioSweepGPU(grid, nb_iter=12, tol_base=TOL)
    rep = lambda a: np.repeat(np.asarray(a)[None, :], 2, axis=0)   # noqa: E731
    sw.set_injections_from_elements(rep(load_p), rep(load_q), rep(gen_p))
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    assert np.all(sw.last_residuals() < 100 * solver_atol)
    V = _V(sw, 2, n_bus)

    for row, g in enumerate(slack_gens):
        grid2 = _solved_multislack_grid()
        grid2.deactivate_gen(int(g))
        grid2.tell_solver_need_reset()
        ref = grid2.ac_pf(V_n.copy(), 30, TOL)
        assert ref.shape[0] > 0, "the reference itself diverged"
        np.testing.assert_allclose(V[row], ref[buses], atol=10 * solver_atol,
                                   err_msg=f"slack generator {g} off")


@requires_gpu
@needs_bridge
def test_with_handle_disconnected_grid(solver_atol):
    """A row that both islands a bus (masked) and drops a generator."""
    from test_handle_disconnected_grid import _solved_spur_grid
    from gpusim2grid import ScenarioSweepGPU

    grid, n_bus, spur_line, spur_bus = _solved_spur_grid(distributed_slack=False)
    gens = grid.get_generators()
    n_gen = len(gens)
    g = next(i for i in range(n_gen) if not gens[i].is_slack)
    load_p, load_q = grid.get_loads_res_full()[:2]
    gen_p = np.asarray(grid.get_gen_target_p())
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)

    mask = np.zeros((1, n_gen), dtype=bool)
    mask[0, g] = True
    sw = ScenarioSweepGPU(grid, handle_disconnected_grid=True, nb_iter=12, tol_base=TOL)
    sw.set_injections_from_elements(load_p[None, :], load_q[None, :], gen_p[None, :])
    sw.set_topology([[int(spur_line)]])
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    assert sw.get_disconnected()[0] == 0
    assert sw.last_residuals()[0] < 100 * solver_atol
    V = _V(sw, 1, n_bus)[0]
    assert np.isnan(V[spur_bus])

    # reference: the spur line AND the generator really removed. The islanded
    # spur load must go too (a one-off solve refuses a load on a bus outside
    # the main component; the masked GPU row simply freezes that bus).
    grid2, _, _, _ = _solved_spur_grid(distributed_slack=False)
    v_n = grid2.get_V()
    spur_load = [i for i, l in enumerate(grid2.get_loads()) if l.bus_id == spur_bus]
    grid2.deactivate_load(int(spur_load[0]))
    grid2.deactivate_powerline(int(spur_line))
    grid2.deactivate_gen(int(g))
    grid2.tell_solver_need_reset()
    ref = grid2.ac_pf(v_n.copy(), 30, TOL)
    assert ref.shape[0] > 0
    main = np.ones(n_bus, dtype=bool)
    main[spur_bus] = False
    np.testing.assert_allclose(V[main], ref[buses][main], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_remote_controller_is_refused():
    pp = pytest.importorskip("pandapower")  # noqa: F841
    import pandapower.networks as pn
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid.lightsim2grid_cpp import AlgorithmType
    from gpusim2grid import ScenarioSweepGPU

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = pn.case14()
        model = init_from_pandapower(net)
        model.set_gen_regulated_bus(3, 9)   # gen 3 (bus 7) regulates bus 9
        model.tell_solver_need_reset()
        model.change_algorithm(AlgorithmType.NR_KLU)
        V = model.ac_pf(np.ones(net.bus.shape[0], dtype=complex), 40, 1e-11)
        assert V.shape[0] > 0
    n_gen = len(model.get_generators())
    sw = ScenarioSweepGPU(model, nb_iter=8, tol_base=1e-10)
    mask = np.zeros((1, n_gen), dtype=bool)
    mask[0, 3] = True
    with pytest.raises(RuntimeError, match="remote"):
        sw.set_contingency_gens(mask)
    # a generator that is NOT a remote controller is still fine on this grid
    mask[:] = False
    mask[0, 0] = True
    sw.set_contingency_gens(mask)


@requires_gpu
@needs_bridge
def test_tuple_mode_rejects(ieee14_base_case):
    from gpusim2grid import ScenarioSweepGPU

    d = ieee14_base_case
    sw = ScenarioSweepGPU(
        (d["Ybus"], d["v_init"].copy(), d["Sbus"], d["slack"],
         d["slack_weights"], d["pv"], d["pq"]),
        nb_iter=4, init_from_n_powerflow=False)
    with pytest.raises(RuntimeError):
        sw.set_contingency_gens(np.zeros((1, 1), dtype=bool))


@requires_gpu
@needs_bridge
def test_shape_validation(case):
    sw = _sweep(case, 2)
    with pytest.raises(ValueError):
        sw.set_contingency_gens(np.zeros((2, case["n_gen"] + 1), dtype=bool))
    with pytest.raises(ValueError):
        sw.set_contingency_gens(np.zeros(case["n_gen"], dtype=bool))
    with pytest.raises(RuntimeError):
        sw.set_contingency_gens(np.zeros((3, case["n_gen"]), dtype=bool))  # 3 != 2 rows


@requires_gpu
@needs_bridge
def test_row_permutation_independence(case, solver_atol):
    """Rows are independent: permuting them permutes the results."""
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    n = len(non_slack)
    mask = np.zeros((n, case["n_gen"]), dtype=bool)
    for r, g in enumerate(non_slack):
        mask[r, g] = True
    sw = _sweep(case, n, mask)
    sw.compute(batch_size=2)
    V = _V(sw, n, case["n_bus"])
    perm = np.array([n - 1 - i for i in range(n)])
    sw2 = _sweep(case, n, mask[perm])
    sw2.compute(batch_size=3)
    V2 = _V(sw2, n, case["n_bus"])
    np.testing.assert_allclose(V2, V[perm], atol=solver_atol)


@requires_gpu
@needs_bridge
@pytest.mark.parametrize("strategy", ["direct_refactor_every", "direct_iter0_only",
                                      "direct_refactor_every_n"])
def test_strategies_match(case, solver_atol, strategy):
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    g = non_slack[0]
    mask = np.zeros((1, case["n_gen"]), dtype=bool)
    mask[0, g] = True
    sw = _sweep(case, 1, mask)
    sw.solver.strategy = strategy
    if strategy == "direct_refactor_every_n":
        sw.solver.refactor_period = 2
    sw.solver.nb_iter = 15
    sw.compute(batch_size=4)
    assert sw.last_residuals()[0] < 100 * solver_atol
    _assert_row(case, _V(sw, 1, case["n_bus"])[0], [g], atol=10 * solver_atol)


@requires_gpu
@needs_bridge
def test_base_case_factors_strategy_raises(case):
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    mask = np.zeros((1, case["n_gen"]), dtype=bool)
    mask[0, non_slack[0]] = True
    sw = _sweep(case, 1, mask)
    sw.solver.strategy = "direct_base_case_factors"
    with pytest.raises(RuntimeError, match="direct_base_case_factors"):
        sw.compute(batch_size=4)


@requires_gpu
@needs_bridge
def test_set_gen_v_with_released_bus(case, solver_atol):
    """set_gen_v on a still-connected generator while another bus is released."""
    non_slack = [g for g in range(case["n_gen"]) if g not in case["slack_gens"]]
    g_off, g_v = non_slack[0], non_slack[1]
    gens = case["grid"].get_generators()
    new_vm = float(gens[g_v].target_vm_pu) + 0.02
    mask = np.zeros((1, case["n_gen"]), dtype=bool)
    mask[0, g_off] = True
    sw = _sweep(case, 1, mask)
    gen_v = np.full((1, case["n_gen"]), np.nan)
    gen_v[0, g_v] = new_vm
    sw.set_gen_v(gen_v)
    sw.compute(batch_size=4)
    assert sw.last_residuals()[0] < 100 * solver_atol
    V = _V(sw, 1, case["n_bus"])[0]

    grid2, v1 = _make_grid()
    grid2.change_v_gen(int(g_v), new_vm)
    grid2.deactivate_gen(int(g_off))
    grid2.tell_solver_need_reset()
    ref = grid2.ac_pf(v1.copy(), MAX_IT, TOL)
    assert ref.shape[0] > 0
    np.testing.assert_allclose(V, ref[case["buses"]], atol=10 * solver_atol)
