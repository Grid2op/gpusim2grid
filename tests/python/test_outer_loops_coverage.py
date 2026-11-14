# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Outer-loop paths test_outer_loops_batch.py leaves uncovered, each against lightsim2grid's
own batch (``ScenarioSweepCPP`` / ``ContingencyAnalysisCPP`` with its ``NROuter_*`` algorithm):
the ReactiveLimits loop's robust mode switched off or given its own realistic band, the
violations of a row the loops reject on a scenario sweep, the host model's cached load state
(rows that mask buses, set voltages, disconnect generators, or change nothing, in one batch),
the DistributedSlack share of a unit on a bus a row strands, and redistribute_slack with a
storage unit taking part in the slack.
"""

import warnings

import numpy as np
import pytest

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp
from test_outer_loops_batch import (OUTER, MAX_IT, TOL_MVA, CONVERGED, OUTER_FAILED, OUTER_UNSTABLE,
                                    UNREALISTIC, _case14, _rows, _v_batch, _V)

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                       reason="gpusim2grid built against a lightsim2grid without NROuter_*"),
    pytest.mark.skipif(bool(_cpp.is_fp32), reason="parity with lightsim2grid's double-precision NR"),
]

ATOL_V = 1e-8   # test_outer_loops_batch.py's parity tolerance (FP64 only, see pytestmark)


def _skip_without_resume(init_from_n):
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")


def _compare_rows(V, status, res, V_ls, conv_ls, res_ls, rows=None, rows_ls=None, decisions=False):
    """per row: converged on both sides or on neither, same outer status and outer iteration
    count, the finite entries of V equal to lightsim2grid's (a masked bus is NaN on the GPU);
    with ``decisions`` the outer decisions too (needs keep_outer_trace on both sides)"""
    rows = range(len(status)) if rows is None else rows
    rows_ls = rows if rows_ls is None else rows_ls
    for i, k in zip(rows, rows_ls):
        msg = f"row {i}: " + "\n".join(res[i].decisions)
        assert (status[i] == CONVERGED) == bool(conv_ls[k]), msg
        assert int(res[i].status) == int(res_ls[k].status), msg
        assert res[i].nb_outer_iterations == res_ls[k].nb_outer_iterations, msg
        if decisions:
            assert list(res[i].decisions) == list(res_ls[k].decisions), msg
        if conv_ls[k]:
            fin = np.isfinite(V[i])
            assert fin.any(), msg
            np.testing.assert_allclose(V[i][fin], V_ls[k][fin], rtol=0., atol=ATOL_V, err_msg=msg)


# --------------------------------------------------------------------------------------
# 1. ReactiveLimits robust mode: switched off, or the band the loop's own parameters give
# --------------------------------------------------------------------------------------
def _remote_controller_grid(driver_robust=True, driver_vmax=None, loop_kwargs=None):
    """case14 whose bus 8 generator (a wide reactive range) regulates bus 7 remotely (its own
    bus at about 1.156 pu, test_outer_loops_batch._case14_remote_controller), the outer-loop
    driver's robust flag / maximum realistic voltage set as given, and -- with loop_kwargs --
    the grid's four default loops rebuilt with ReactiveLimits(**loop_kwargs)"""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.algorithm import DistributedSlack, HvdcAcEmulationLimits, VoltageMonitoring, ReactiveLimits
    net = pn.case14()
    net.gen["min_q_mvar"] = -15.
    net.gen["max_q_mvar"] = 15.
    net.gen.loc[net.gen.bus == 7, ["min_q_mvar", "max_q_mvar"]] = (-300., 300.)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 300.))
    grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
    gen_id = next(g.id for g in grid.get_generators() if g.bus_id == 7)
    grid.set_gen_regulated_bus(gen_id, 6)
    params = grid.get_outer_loop_driver_params()
    params.voltage_remote_control_robust_mode = driver_robust
    if driver_vmax is not None:
        params.max_realistic_voltage = driver_vmax
    grid.set_outer_loop_driver_params(params)
    if loop_kwargs is not None:
        grid.clear_outer_loops()
        grid.add_outer_loop(DistributedSlack())
        grid.add_outer_loop(HvdcAcEmulationLimits())
        grid.add_outer_loop(VoltageMonitoring())
        grid.add_outer_loop(ReactiveLimits(**loop_kwargs))
    grid.change_algorithm(OUTER)
    V = grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    assert V.size > 0
    return grid


def _n1_both(grid, init_from_n):
    """every branch tripped in turn, handle_disconnected_grid, on lightsim2grid's
    ContingencyAnalysisCPP and on ContingencyAnalysisGPU (keep_outer_trace)"""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    n_br = len(grid.get_lines()) + len(grid.get_trafos())
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.handle_disconnected_grid = True
    batch.add_all_n1()
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                                handle_disconnected_grid=True, init_from_n_powerflow=init_from_n,
                                tol=TOL_MVA / grid.get_sn_mva())
    ca.add_contingencies_by_branch_id([[i] for i in range(n_br)])
    ca.compute(batch_size=4)
    n_bus = grid.get_Ybus_solver().shape[0]
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(n_br, n_bus)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    return (V, np.asarray(ca.get_row_status()), ca.get_outer_results(),
            V_ls, np.asarray(batch.converged_mask(), dtype=bool), batch.get_outer_results())


def _nb_unrealistic_switches(res):
    return sum(any("PV_TO_PQ_UNREALISTIC" in d for d in r.decisions) for r in res)


@pytest.mark.parametrize("where", ["driver", "loop"])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_robust_mode_off_never_releases_the_remote_controller(where, init_from_n):
    """The robust flag OFF -- on the outer-loop driver's parameters (the loop left at its
    default, -1 = the driver's), or on the ReactiveLimits loop's own parameters (overriding a
    driver that says on) -- with a realistic band (1.16 / 1.02) the remote controller's own bus
    sits above: no row makes a PV_TO_PQ_UNREALISTIC decision, and every row matches
    lightsim2grid's ContingencyAnalysisCPP (convergence, outer status and iteration count,
    finite V entries)"""
    _skip_without_resume(init_from_n)
    if where == "driver":
        grid = _remote_controller_grid(driver_robust=False, driver_vmax=1.16)
    else:
        grid = _remote_controller_grid(driver_robust=True, driver_vmax=1.16, loop_kwargs={"robust_mode": False})
    V, status, res, V_ls, conv_ls, res_ls = _n1_both(grid, init_from_n)
    assert _nb_unrealistic_switches(res) == 0
    assert conv_ls.any()
    _compare_rows(V, status, res, V_ls, conv_ls, res_ls)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_robust_mode_band_from_the_loop_parameters(init_from_n):
    """The ReactiveLimits loop's own max_realistic_voltage (1.16) overriding the driver's band
    (its default, 2 pu), the driver's robust flag on: the loop releases the remote controller
    whose own bus sits above 1.16 / 1.02: without init_from_n some row records
    PV_TO_PQ_UNREALISTIC (resumed from the "n" solve the controller is already released there,
    so that count is not asserted), and every row matches lightsim2grid's
    ContingencyAnalysisCPP (convergence, outer status and iteration count, finite V entries)"""
    _skip_without_resume(init_from_n)
    grid = _remote_controller_grid(driver_robust=True, loop_kwargs={"max_realistic_voltage": 1.16})
    V, status, res, V_ls, conv_ls, res_ls = _n1_both(grid, init_from_n)
    if not init_from_n:
        assert _nb_unrealistic_switches(res) > 0
    assert conv_ls.any()
    _compare_rows(V, status, res, V_ls, conv_ls, res_ls)


def test_robust_mode_driver_band_ignored_when_the_loop_sets_its_own():
    """The converse: the driver's band at 1.16 (which would release the remote controller) but
    the loop's own max_realistic_voltage at 2 pu: no row records PV_TO_PQ_UNREALISTIC, and
    every row matches lightsim2grid's ContingencyAnalysisCPP"""
    grid = _remote_controller_grid(driver_robust=True, driver_vmax=1.16, loop_kwargs={"max_realistic_voltage": 2.})
    V, status, res, V_ls, conv_ls, res_ls = _n1_both(grid, False)
    assert _nb_unrealistic_switches(res) == 0
    assert conv_ls.any()
    _compare_rows(V, status, res, V_ls, conv_ls, res_ls)


# --------------------------------------------------------------------------------------
# 2. A scenario-sweep row the outer loops reject: one GRID/DIVERGENCE violation
# --------------------------------------------------------------------------------------
def test_scenario_sweep_outer_failure_is_divergence():
    """ScenarioSweepGPU rows the outer loops reject (UNREALISTIC: the realistic band just under
    the base case's lowest voltage, loads raised by 5 % per row) with every bus limit biting
    (vmax at half its nominal voltage): each such row's get_violations is exactly one
    GRID/DIVERGENCE entry with value = limit = NaN (as lightsim2grid's ScenarioSweepCPP gives
    them), its get_violation_counts are all 0, get_violations_truncated is False and
    converged() is False; every converged row (convergence matches lightsim2grid's) reports
    high-voltage violations and converged() True"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    from gpusim2grid.contingency_analysis import LimitViolationType, ViolationElementType
    grid = _case14()
    params = grid.get_outer_loop_driver_params()
    params.min_realistic_voltage = float(np.abs(grid.get_V()).min()) - 0.005
    grid.set_outer_loop_driver_params(params)
    vn = np.asarray(grid.get_bus_vn_kv(), dtype=float)
    grid.set_bus_voltage_limits(0.1 * vn, 0.5 * vn)
    assert grid.ac_pf(np.asarray(grid.get_V(), dtype=complex), MAX_IT, TOL_MVA).size > 0
    n_rows = 8
    rows = _rows(grid, n_rows=n_rows, trips=False)

    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.compute_limit_violations = True
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    batch.compute(_v_batch(grid, False), MAX_IT, TOL_MVA)

    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=False,
                          compute_limit_violations=True, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.compute(batch_size=3)
    status = np.asarray(sw.get_row_status())
    conv = np.asarray(batch.converged_mask(), dtype=bool)
    rejected = np.isin(status, (OUTER_FAILED, OUTER_UNSTABLE, UNREALISTIC))
    assert (status == UNREALISTIC).any(), status
    assert (status == CONVERGED).any(), status
    v_ls, v_gpu = batch.get_violations(), sw.get_violations()
    counts, truncated, converged = sw.get_violation_counts(), sw.get_violations_truncated(), sw.converged()
    for i in range(n_rows):
        assert (status[i] == CONVERGED) == bool(conv[i]), (i, status)
        if status[i] == CONVERGED:
            assert counts["high_voltage"][i] > 0, i
            assert converged[i], i
        if rejected[i]:
            assert len(v_ls[i]) == 1 and f"{v_ls[i][0].violation_type}".endswith("DIVERGENCE"), i
            assert np.isnan(v_ls[i][0].value) and np.isnan(v_ls[i][0].limit), i
            assert len(v_gpu[i]) == 1, (i, v_gpu[i])
            el = v_gpu[i][0]
            assert el.violation_type == LimitViolationType.DIVERGENCE, i
            assert el.element_type == ViolationElementType.GRID, i
            assert np.isnan(el.value) and np.isnan(el.limit), i
            assert all(counts[k][i] == 0 for k in ("low_voltage", "high_voltage", "current")), (i, counts)
            assert not truncated[i], i
            assert not converged[i], i


# --------------------------------------------------------------------------------------
# 3. The host model's cached load state (units0_, resume0_*, rl_plan_masked): rows that
#    change what it is built from next to rows that do not, in one batch
# --------------------------------------------------------------------------------------
def _gen_targets_match(res, res_ls, rows, rows_ls=None, atol=1e-6):
    """same generators with a moved target (gen_target_p) on both sides, at the same values"""
    rows_ls = rows if rows_ls is None else rows_ls
    for i, k in zip(rows, rows_ls):
        mine, ref = dict(res[i].gen_target_p), dict(res_ls[k].gen_target_p)
        assert sorted(mine) == sorted(ref), (i, mine, ref)
        for g in ref:
            assert mine[g] == pytest.approx(ref[g], abs=atol), (i, g, mine[g], ref[g])


@pytest.mark.parametrize("injections", ["elements", "per_bus"])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_cached_load_state_rows_in_one_batch(init_from_n, injections):
    """One ScenarioSweepGPU outer batch (handle_disconnected_grid) mixing rows that rebuild the
    model's per-row state with rows that take the cached one: (0) bus 13 islanded AND a
    generator's set-point moved (set_gen_v, bus 7's), (1) bus 13 islanded AND a generator disconnected
    (set_contingency_gens), (2) bus 13 islanded only, (3) nothing changed, (4) the generator
    bus 7 islanded only, (5) bus 7 islanded AND a set-point moved (bus 2's), (6) bus 7's set-point moved only,
    (7) a generator disconnected only. Injections either per element (every row sets its
    generators' targets) or per bus (set_injections, the grid's targets: no row sets one, so
    the resumed targets are the cached ones). Every row matches lightsim2grid's
    ScenarioSweepCPP given the same inputs (modify_gen_v, set_contingency_gens /
    _lines / _trafos, handle_disconnected_grid=True): convergence, outer status and
    iteration count, finite V entries, and the generators' moved targets (gen_target_p)"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    from gpusim2grid._ls2g_utils import extract_injection_elements, build_bus_injections
    _skip_without_resume(init_from_n)
    grid = _case14()
    gens = grid.get_generators()
    n_gen, n_line = len(gens), len(grid.get_lines())
    gen_of_bus = {g.bus_id: g.id for g in gens}
    assert 7 in gen_of_bus                       # bus 7 carries a generator (not at a limit)
    g_v7, g_v2, g_off = gen_of_bus[7], gen_of_bus[2], gen_of_bus[5]
    leaf_lines = [el.id for el in grid.get_lines() if 13 in (el.bus1_id, el.bus2_id)]
    leaf_trafo = [el.id for el in grid.get_trafos() if 7 in (el.bus1_id, el.bus2_id)]
    assert len(leaf_trafo) == 1
    n_rows = 8
    rows = _rows(grid, n_rows=n_rows, trips=False)
    if injections == "per_bus":
        rows["load_p"][:] = rows["load_p"][0]    # the grid's own injections on every row
    lines = [leaf_lines, leaf_lines, leaf_lines, [], [], [], [], []]
    trafos = [[], [], [], [], leaf_trafo, leaf_trafo, [], []]
    gen_v = np.full((n_rows, n_gen), np.nan)
    for r, g in ((0, g_v7), (5, g_v2), (6, g_v7)):
        gen_v[r, g] = gens[g].target_vm_pu + 0.01
    gen_off = np.zeros((n_rows, n_gen), dtype=bool)
    gen_off[[1, 7], g_off] = True

    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.handle_disconnected_grid = True
    if injections == "elements":
        batch.modify_gen_p(rows["gen_p"])
        batch.modify_load_p(rows["load_p"])
        batch.modify_load_q(rows["load_q"])
    batch.modify_gen_v(np.where(np.isnan(gen_v), [[g.target_vm_pu for g in gens]], gen_v))
    mask_l = np.zeros((n_rows, n_line), dtype=bool)
    mask_t = np.zeros((n_rows, len(grid.get_trafos())), dtype=bool)
    for r in range(n_rows):
        mask_l[r, lines[r]] = True
        mask_t[r, trafos[r]] = True
    batch.set_contingency_lines(mask_l)
    batch.set_contingency_trafos(mask_t)
    batch.set_contingency_gens(gen_off)
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)

    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                          keep_outer_trace=True, handle_disconnected_grid=True, tol=TOL_MVA / grid.get_sn_mva())
    if injections == "elements":
        sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    else:
        n_bus = grid.get_Ybus_solver().shape[0]
        p_mw, q_mvar = build_bus_injections(extract_injection_elements(grid, n_bus), rows["load_p"],
                                            rows["load_q"], rows["gen_p"], gen_off=gen_off)
        sw.set_injections(p_mw, q_mvar, grid.get_sn_mva())
    sw.set_topology([lines[r] + [n_line + t for t in trafos[r]] for r in range(n_rows)])
    sw.set_gen_v(gen_v)
    sw.set_contingency_gens(gen_off)
    sw.compute(batch_size=3)

    V = _V(sw, grid, n_rows)
    assert np.isnan(V[[0, 1, 2, 4, 5]]).any(axis=1).all()     # the islanding rows masked a bus
    assert np.isfinite(V[[3, 6, 7]]).all()
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    conv_ls = np.asarray(batch.converged_mask(), dtype=bool)
    res, res_ls = sw.get_outer_results(), batch.get_outer_results()
    assert conv_ls.all(), conv_ls
    _compare_rows(V, np.asarray(sw.get_row_status()), res, V_ls, conv_ls, res_ls)
    _gen_targets_match(res, res_ls, range(n_rows))


# --------------------------------------------------------------------------------------
# 4. DistributedSlack shares nothing with a unit on a bus the row strands
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("init_from_n", [False, True])
def test_distributed_slack_no_share_on_a_stranded_bus(init_from_n):
    """case14 (every generator taking part in the DistributedSlack loop, weight 300): rows
    tripping the transformer to bus 7 strand that bus' generator (handle_disconnected_grid),
    one at the grid's loads and one at +10 % (a mismatch the loop shares), next to a row
    changing nothing (every row sets its generators' targets, so a resumed row takes no
    target the "n" solve moved for them). On the stranding rows the stranded generator has no
    gen_target_p entry at all while every other participant has one; on the row changing
    nothing it has one. Every row matches lightsim2grid's ScenarioSweepCPP
    (convergence, outer status and iteration count, finite V entries, gen_target_p)"""
    from lightsim2grid.lightsim2grid_cpp import ScenarioSweepCPP
    from gpusim2grid import ScenarioSweepGPU
    _skip_without_resume(init_from_n)
    grid = _case14()
    gens = grid.get_generators()
    n_gen, n_line = len(gens), len(grid.get_lines())
    stranded = next(g.id for g in gens if g.bus_id == 7)
    leaf_trafo = [el.id for el in grid.get_trafos() if 7 in (el.bus1_id, el.bus2_id)]
    n_rows = 3
    rows = _rows(grid, n_rows=n_rows, trips=False)
    rows["load_p"][:] = rows["load_p"][0]
    rows["load_p"][1] *= 1.10
    trafos = [leaf_trafo, leaf_trafo, []]
    STRANDING, NOTHING = (0, 1), 2
    assert n_gen > 2

    batch = ScenarioSweepCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.handle_disconnected_grid = True
    batch.modify_gen_p(rows["gen_p"])
    batch.modify_load_p(rows["load_p"])
    batch.modify_load_q(rows["load_q"])
    mask_t = np.zeros((n_rows, len(grid.get_trafos())), dtype=bool)
    for r in range(n_rows):
        mask_t[r, trafos[r]] = True
    batch.set_contingency_trafos(mask_t)
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)

    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=MAX_IT, init_from_n_powerflow=init_from_n,
                          keep_outer_trace=True, handle_disconnected_grid=True, tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
    sw.set_topology([[n_line + t for t in trafos[r]] for r in range(n_rows)])
    sw.compute(batch_size=2)

    V = _V(sw, grid, n_rows)
    assert np.isnan(V[list(STRANDING)]).any(axis=1).all()
    res = sw.get_outer_results()
    for r in STRANDING:
        moved = dict(res[r].gen_target_p)
        msg = f"row {r}: {moved}\n" + "\n".join(res[r].decisions)
        assert stranded not in moved, msg
        assert sorted(moved) == sorted(set(range(n_gen)) - {stranded}), msg
    assert stranded in dict(res[NOTHING].gen_target_p)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    conv_ls = np.asarray(batch.converged_mask(), dtype=bool)
    assert conv_ls.all(), conv_ls
    res_ls = batch.get_outer_results()
    _compare_rows(V, np.asarray(sw.get_row_status()), res, V_ls, conv_ls, res_ls)
    _gen_targets_match(res, res_ls, range(n_rows))


# --------------------------------------------------------------------------------------
# 5. redistribute_slack with a storage unit taking part in the slack
# --------------------------------------------------------------------------------------
def _case14_with_storage(sto_bus=3, weight=300., p_range=(-50., 50.)):
    """_case14 plus one storage unit on a load bus (not regulating, at 0 MW), flagged as a
    DistributedSlack participant (set_storage_can_participate_slack, weight 300, the
    generators' weight) with active limits [-50, 50] MW (generator convention), solved again
    with the grid's outer loops"""
    grid = _case14()
    grid.init_storages_full(np.array([0.]), np.array([0.]), [False], np.array([1.]),
                            np.array([-50.]), np.array([50.]), np.array([sto_bus], dtype=np.int32))
    grid.set_storage_p_limits(np.array([p_range[0]]), np.array([p_range[1]]))
    grid.set_storage_can_participate_slack([True], np.array([weight]))
    grid.tell_solver_need_reset()
    grid.change_algorithm(OUTER)
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return grid


@pytest.mark.parametrize("p_range", [(-50., 50.), (-1., 1.)])
@pytest.mark.parametrize("init_from_n", [False, True])
def test_redistribute_slack_with_a_storage_participant(init_from_n, p_range):
    """ContingencyAnalysisGPU, outer loops and redistribute_slack, on case14 with a storage
    unit taking part in the slack, its active range wide or narrow (+-1 MW: its share
    clamped at a bound): contingencies losing a generator, a load, and a line (nothing lost).
    The generator / load contingencies share their lost power (a SHARE_LOST_P decision), the
    line one does not; the storage unit moves on the generator / load contingencies (a
    storage_target_p entry) and every row matches lightsim2grid's ContingencyAnalysisCPP with
    redistribute_slack=True: convergence, outer status and iteration count, finite V entries,
    gen_target_p and storage_target_p"""
    from lightsim2grid.lightsim2grid_cpp import ContingencyAnalysisCPP
    from gpusim2grid import ContingencyAnalysisGPU
    _skip_without_resume(init_from_n)
    grid = _case14_with_storage(p_range=p_range)
    gen_40mw = [el.id for el in grid.get_generators() if el.target_p_mw > 0. and not el.is_slack]
    ctgs = [([], [], [], gen_40mw), ([], [], [3], []), ([3], [], [], [])]
    batch = ContingencyAnalysisCPP(grid)
    batch.change_algorithm(OUTER)
    batch.init_from_n_powerflow = init_from_n
    batch.handle_disconnected_grid = True
    batch.redistribute_slack = True
    for branches, shunts, loads, gens in ctgs:
        batch.add_contingency(branches, shunts, loads, gens)
    batch.compute(_v_batch(grid, init_from_n), MAX_IT, TOL_MVA)
    ca = ContingencyAnalysisGPU(grid, outer_loops=True, nb_iter=MAX_IT, keep_outer_trace=True,
                                handle_disconnected_grid=True, init_from_n_powerflow=init_from_n,
                                redistribute_slack=True, tol=TOL_MVA / grid.get_sn_mva())
    ca.add_contingencies(ctgs)
    ca.compute(batch_size=2)
    n_ctg, n_bus = len(ctgs), grid.get_Ybus_solver().shape[0]
    V = ca.V_results.to_numpy().reshape(n_ctg, n_bus)
    # lightsim2grid orders its contingencies (test_outer_loops_batch's redistribute_slack test)
    conts = batch.my_contingencies()
    n_lists = len(conts[0])
    key = lambda c: tuple(tuple(sorted(int(e) for e in el)) for el in (tuple(c) + ((),) * n_lists)[:n_lists])
    row_ls = {key(c): k for k, c in enumerate(conts)}
    rows_ls = [row_ls[key(c)] for c in ctgs]
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V_ls = np.asarray(batch.get_voltages())[:, buses]
    conv_ls = np.asarray(batch.converged_mask(), dtype=bool)
    assert conv_ls.all(), conv_ls
    res, res_ls = ca.get_outer_results(), batch.get_outer_results()
    _compare_rows(V, np.asarray(ca.get_row_status()), res, V_ls, conv_ls, res_ls, range(n_ctg), rows_ls)
    _gen_targets_match(res, res_ls, range(n_ctg), rows_ls)
    for i, k in zip(range(n_ctg), rows_ls):
        mine, ref = dict(res[i].storage_target_p), dict(res_ls[k].storage_target_p)
        msg = f"contingency {ctgs[i]}: {mine} vs {ref}\n" + "\n".join(res[i].decisions)
        assert sorted(mine) == sorted(ref), msg
        for s in ref:
            assert mine[s] == pytest.approx(ref[s], abs=1e-6), msg
        assert any("SHARE_LOST_P" in d for d in res[i].decisions) == (i < 2), msg
        if i < 2:
            assert 0 in mine, msg
            assert p_range[0] - 1e-9 <= -mine[0] <= p_range[1] + 1e-9, msg   # load convention
