# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Contingencies tripping static generators, storage units, SVCs and HVDC lines
(lightsim2grid PR #292), on ContingencyAnalysisGPU and ScenarioSweepGPU.

The oracle is lightsim2grid's: a one-off powerflow of the grid with the element
really deactivated. What the batch cannot do yet is reported per row
(``get_batch_unsupported_reasons``), not raised for the whole batch.
"""
import copy
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge

pytestmark = [requires_gpu, pytest.mark.skipif(not have_ls2g_bridge, reason="needs the lightsim2grid bridge")]

MAX_IT, TOL = 30, 1e-10
OUTER = "NROuter_KLU"
SGEN, STORAGE, SVC, HVDC = "sgen", "storage", "svc", "hvdc"
TUPLE_POS = {SGEN: 6, STORAGE: 7, SVC: 8, HVDC: 9}


def _net():
    import pandapower.networks as pn
    return pn.case14()


def _finish(grid, algo="NR_KLU"):
    grid.tell_solver_need_reset()
    grid.change_algorithm(algo)
    assert grid.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL).shape[0] > 0
    return grid


def _base():
    from lightsim2grid.network import init_from_pandapower
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return init_from_pandapower(_net())


def _grid_sgen():
    import pandapower as pp
    from lightsim2grid.network import init_from_pandapower
    net = _net()
    pp.create_sgen(net, 4, p_mw=5., q_mvar=2.)
    pp.create_sgen(net, 10, p_mw=3., q_mvar=-1.)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return _finish(init_from_pandapower(net))


REGULATING, PQ = 0, 1
BUS_REGULATING, BUS_PQ = 12, 13


def _grid_storage(algo="NR_KLU"):
    """one unit alone regulating its bus, one that does not regulate"""
    grid = _base()
    grid.init_storages_full(np.array([5., -3.]), np.array([0., 1.]), [True, False],
                            np.array([1.03, 1.]), np.array([-50., -50.]), np.array([50., 50.]),
                            np.array([BUS_REGULATING, BUS_PQ], dtype=np.int32))
    return _finish(grid, algo)


FIXED_Q, OFF_B0, VOLTAGE = 0, 1, 2   # REACTIVE_POWER, OFF with a b0, VOLTAGE


def _grid_svc():
    grid = _base()
    grid.init_svcs([2, 0, 1], np.array([1., 1., 1.03]), np.array([5., 0., 0.]),
                   np.zeros(3), np.full(3, -0.5), np.full(3, 0.5),
                   np.array([12, 13, 4], dtype=np.int32), np.array([12, 13, 4], dtype=np.int32))
    grid.set_svc_b0(np.array([0.02, 0.03, 0.]))
    return _finish(grid)


DROOP, HV_REGULATING, FIXED = 0, 1, 2   # AC emulation, regulating at both ends, LCC at fixed P and Q


def _grid_hvdc(algo="NR_KLU"):
    grid = _base()
    big = np.full(3, 1e3)
    grid.init_hvdc_lines(np.array([9, 10, 12], dtype=np.int32), np.array([3, 11, 13], dtype=np.int32),
                         [0, 0, 1], [0, 0, 1], np.ones(3), np.ones(3),
                         [True, True, False], [False, True, False],
                         np.array([1.03, 1.03, 1.]), np.array([1., 1.03, 1.]),
                         np.zeros(3), np.array([3., 0., 0.]), -big, big, -big, big,
                         np.array([1., 1., 0.9]), np.array([1., 1., 0.9]),
                         [0, 0, 0], np.array([10., 5., 3.]), np.ones(3), np.full(3, 12.),
                         [True, False, False], np.array([5., 0., 0.]), np.array([2., 0., 0.]),
                         np.full(3, 100.), np.full(3, 100.))
    return _finish(grid, algo)


def _grid_two_stations_on_a_gen_bus():
    """two regulating stations on the bus of a generator: three controllers of that bus"""
    grid = _base()
    gen = next(g for g in grid.get_generators() if not g.is_slack)
    big = np.full(2, 1e3)
    grid.init_hvdc_lines(np.array([gen.bus_id, gen.bus_id], dtype=np.int32), np.array([13, 12], dtype=np.int32),
                         [0, 0], [0, 0], np.ones(2), np.ones(2), [True, True], [False, False],
                         np.full(2, gen.target_vm_pu), np.ones(2), np.zeros(2), np.zeros(2), -big, big, -big, big,
                         np.ones(2), np.ones(2), [0, 0], np.array([1., 2.]), np.ones(2), np.full(2, 12.),
                         [False, False], np.zeros(2), np.zeros(2), np.full(2, 100.), np.full(2, 100.))
    return _finish(grid), gen.id, gen.bus_id, gen.target_vm_pu


def _deactivate(grid, kind, el_id):
    {SGEN: grid.deactivate_sgen, STORAGE: grid.deactivate_storage, SVC: grid.deactivate_svc,
     HVDC: grid.deactivate_dcline}[kind](int(el_id))


def _n_el(grid, kind):
    return len({SGEN: grid.get_static_generators, STORAGE: grid.get_storages, SVC: grid.get_svcs,
                HVDC: grid.get_dclines}[kind]())


def _reference(grid, kind=None, ids=(), branches=(), gens=()):
    ref = copy.deepcopy(grid)
    n_line = len(grid.get_lines())
    for br in branches:
        if br < n_line:
            ref.deactivate_powerline(int(br))
        else:
            ref.deactivate_trafo(int(br) - n_line)
    for g in gens:
        ref.deactivate_gen(int(g))
    for el in ids:
        _deactivate(ref, kind, el)
    ref.tell_solver_need_reset()
    V = ref.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V.shape[0] > 0, "the reference powerflow itself diverged"
    return V[np.asarray(grid.id_ac_solver_to_me(), dtype=int)]


def _ctg(kind, ids, branches=(), gens=()):
    """the 10-tuple of a contingency"""
    c = [list(branches), [], [], list(gens), [], [], [], [], [], []]
    c[TUPLE_POS[kind]] = list(ids)
    return tuple(c)


def _analysis(grid, trips, kind, **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU
    kwargs.setdefault("nb_iter", 15)
    kwargs.setdefault("tol_base", TOL)
    ca = ContingencyAnalysisGPU(grid, **kwargs)
    ca.add_contingencies([_ctg(kind, ids, branches) for ids, branches in trips])
    ca.compute(batch_size=4)
    n_bus = grid.get_Ybus_solver().shape[0]
    return ca, ca.V_results.to_numpy().reshape(len(trips), n_bus)


def _sweep(grid, trips, kind, **kwargs):
    from gpusim2grid import ScenarioSweepGPU
    kwargs.setdefault("nb_iter", 15)
    kwargs.setdefault("tol_base", TOL)
    n = len(trips)
    sw = ScenarioSweepGPU(grid, **kwargs)
    load_p = np.asarray(grid.get_load_target_p(), dtype=float)
    load_q = np.array([el.target_q_mvar for el in grid.get_loads()], dtype=float)
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)
    sw.set_injections_from_elements(np.tile(load_p, (n, 1)), np.tile(load_q, (n, 1)), np.tile(gen_p, (n, 1)))
    mask = np.zeros((n, _n_el(grid, kind)), dtype=bool)
    for r, (ids, _) in enumerate(trips):
        mask[r, list(ids)] = True
    getattr(sw, f"set_contingency_{kind}s")(mask)
    sw.set_topology([list(branches) for _, branches in trips])
    sw.compute(batch_size=4)
    return sw, sw.solver.V_results.to_numpy().reshape(n, grid.get_Ybus_solver().shape[0])


CASES = [
    (SGEN, _grid_sgen, [([0], []), ([1], []), ([0, 1], []), ([0], [5])]),
    (STORAGE, _grid_storage, [([REGULATING], []), ([PQ], []), ([REGULATING, PQ], []), ([PQ], [5])]),
    (SVC, _grid_svc, [([FIXED_Q], []), ([OFF_B0], []), ([FIXED_Q, OFF_B0], []), ([FIXED_Q], [5])]),
    (HVDC, _grid_hvdc, [([FIXED], []), ([HV_REGULATING], []), ([FIXED, HV_REGULATING], []), ([FIXED], [5]),
                        ([DROOP], []), ([DROOP, FIXED], []), ([DROOP], [5])]),
]


@pytest.mark.parametrize("runner", [_analysis, _sweep], ids=["analysis", "sweep"])
@pytest.mark.parametrize("kind, make, trips", CASES, ids=[c[0] for c in CASES])
def test_matches_a_one_off_powerflow(runner, kind, make, trips):
    grid = make()
    session, V = runner(grid, trips, kind)
    assert not any(session.get_batch_unsupported_reasons())
    for r, (ids, branches) in enumerate(trips):
        np.testing.assert_allclose(V[r], _reference(grid, kind, ids, branches), rtol=0., atol=1e-7,
                                   err_msg=f"row {r}: {kind} {ids}, branches {branches}")


@pytest.mark.parametrize("runner", [_analysis, _sweep], ids=["analysis", "sweep"])
def test_a_storage_unit_of_the_slack_leaves_it(runner):
    grid = _grid_storage()
    for gen in grid.get_generators():
        grid.add_gen_slackbus(gen.id, 1.)
    grid.add_storage_slackbus(PQ, 1.)
    _finish(grid)
    trips = [([PQ], [])]
    session, V = runner(grid, trips, STORAGE)
    np.testing.assert_allclose(V[0], _reference(grid, STORAGE, [PQ]), rtol=0., atol=1e-7)


@pytest.mark.parametrize("gens_off, hvdcs, moves", [([True], [], False), ([], [0, 1], False), ([True], [0], False),
                                                    ([True], [0, 1], True)],
                         ids=["gen", "stations", "gen+one", "all"])
def test_controllers_of_a_bus_generator_and_stations(gens_off, hvdcs, moves):
    """the bus turns PQ only when the row takes out every one of its controllers"""
    from gpusim2grid import ContingencyAnalysisGPU
    grid, gen_id, bus, vm = _grid_two_stations_on_a_gen_bus()
    gens = [gen_id] if gens_off else []
    ca = ContingencyAnalysisGPU(grid, nb_iter=15, tol_base=TOL)
    c = list(_ctg(HVDC, hvdcs, gens=gens))
    ca.add_contingencies([tuple(c)])
    ca.compute(batch_size=1)
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    V = ca.V_results.to_numpy().reshape(1, buses.shape[0])[0]
    assert (abs(abs(V[list(buses).index(bus)]) - vm) > 1e-4) == moves
    np.testing.assert_allclose(V, _reference(grid, HVDC, hvdcs, gens=gens), rtol=0., atol=1e-7)


def _grid_storage_on_group_bus():
    grid = _grid_storage()
    remote = next(g.id for g in grid.get_generators() if not g.is_slack)
    grid.set_gen_regulated_bus(remote, BUS_REGULATING)
    grid.change_v_gen(remote, 1.03)
    return _finish(grid), remote


def _grid_hvdc_station_in_group():
    """two stations on a bus a remote generator regulates too: they are all of its group"""
    grid, gen_id, bus, vm = _grid_two_stations_on_a_gen_bus()
    other = next(g.id for g in grid.get_generators() if not g.is_slack and g.id != gen_id)
    grid.set_gen_regulated_bus(other, bus)
    grid.change_v_gen(other, vm)
    return _finish(grid)


def _grid_standby_svc():
    grid = _base()
    grid.init_svcs([2], np.ones(1), np.array([5.]), np.zeros(1), np.full(1, -0.5), np.full(1, 0.5),
                   np.array([4], dtype=np.int32), np.array([4], dtype=np.int32))
    grid.set_svc_standby([True], np.full(1, 0.9), np.full(1, 1.1))
    return _finish(grid)


UNSUPPORTED = [
    (SVC, _grid_standby_svc, [0], "standby"),
    (STORAGE, lambda: _grid_storage_on_group_bus()[0], [REGULATING], "group"),
]


@pytest.mark.parametrize("runner", [_analysis, _sweep], ids=["analysis", "sweep"])
@pytest.mark.parametrize("kind, make, bad, word", UNSUPPORTED, ids=[c[0] for c in UNSUPPORTED])
def test_what_the_batch_cannot_do_is_reported_per_row(runner, kind, make, bad, word):
    from gpusim2grid.contingency_analysis import LimitViolationType
    grid = make()
    # (an element the batch can take out, or nothing: the second row opens a line anyway)
    ok = {STORAGE: [PQ]}.get(kind, [])
    trips = [(bad, []), (ok, [5])]
    session, V = runner(grid, trips, kind, compute_limit_violations=True)
    reasons = session.get_batch_unsupported_reasons()
    assert word in reasons[0] and reasons[1] == ""
    assert np.isnan(V[0]).all()
    np.testing.assert_allclose(V[1], _reference(grid, kind, ok, [5]), rtol=0., atol=1e-7)
    v0 = session.get_violations()[0]
    assert [el.violation_type for el in v0] == [LimitViolationType.BATCH_UNSUPPORTED]


# ----- controllers of a voltage-control group ---------------------------------------------
GROUP = (1, 2, 3)
GROUP_TRIPS = ([1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3])
REG_BUS = 9


def _grid_group(group=GROUP, reg_bus=REG_BUS, algo="NR_KLU", q_limits=None):
    from lightsim2grid.network import init_from_pandapower
    net = _net()
    for gen_pos, (min_q, max_q) in (q_limits or {}).items():
        net.gen.loc[net.gen.index[gen_pos - 1], ["min_q_mvar", "max_q_mvar"]] = [min_q, max_q]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    for gen_id in group:
        grid.set_gen_regulated_bus(gen_id, reg_bus)
        grid.change_v_gen(gen_id, 1.04)
    return _finish(grid, algo)


def _gens_in_analysis(grid, trips, **kwargs):
    from gpusim2grid import ContingencyAnalysisGPU
    kwargs.setdefault("nb_iter", 15)
    kwargs.setdefault("tol_base", TOL)
    ca = ContingencyAnalysisGPU(grid, **kwargs)
    ca.add_contingencies([([], [], [], list(g)) for g in trips])
    ca.compute(batch_size=4)
    return ca, ca.V_results.to_numpy().reshape(len(trips), grid.get_Ybus_solver().shape[0])


def _gens_in_sweep(grid, trips, **kwargs):
    from gpusim2grid import ScenarioSweepGPU
    kwargs.setdefault("nb_iter", 15)
    kwargs.setdefault("tol_base", TOL)
    n = len(trips)
    sw = ScenarioSweepGPU(grid, **kwargs)
    load_p = np.asarray(grid.get_load_target_p(), dtype=float)
    load_q = np.array([el.target_q_mvar for el in grid.get_loads()], dtype=float)
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)
    sw.set_injections_from_elements(np.tile(load_p, (n, 1)), np.tile(load_q, (n, 1)), np.tile(gen_p, (n, 1)))
    mask = np.zeros((n, gen_p.shape[0]), dtype=bool)
    for r, g in enumerate(trips):
        mask[r, list(g)] = True
    sw.set_contingency_gens(mask)
    sw.compute(batch_size=4)
    return sw, sw.solver.V_results.to_numpy().reshape(n, grid.get_Ybus_solver().shape[0])


@pytest.mark.parametrize("runner", [_gens_in_analysis, _gens_in_sweep], ids=["analysis", "sweep"])
@pytest.mark.parametrize("kwargs", [{}, {"handle_disconnected_grid": True}], ids=["plain", "islands"])
def test_generators_of_a_group_match_a_one_off_powerflow(runner, kwargs):
    grid = _grid_group()
    session, V = runner(grid, GROUP_TRIPS, **kwargs)
    assert not any(session.get_batch_unsupported_reasons()), session.get_batch_unsupported_reasons()
    for r, gens in enumerate(GROUP_TRIPS):
        np.testing.assert_allclose(V[r], _reference(grid, gens=gens), rtol=0., atol=1e-7, err_msg=f"row {r}: {gens}")


def test_a_group_of_one():
    grid = _grid_group(group=(3,))
    session, V = _gens_in_analysis(grid, [[3]])
    np.testing.assert_allclose(V[0], _reference(grid, gens=[3]), rtol=0., atol=1e-7)


def test_the_regulated_bus_goes_free_when_nobody_is_left():
    grid = _grid_group()
    session, V = _gens_in_analysis(grid, [GROUP, [1]])
    buses = list(np.asarray(grid.id_ac_solver_to_me(), dtype=int))
    assert abs(abs(V[0][buses.index(REG_BUS)]) - 1.04) > 1e-3
    assert abs(V[1][buses.index(REG_BUS)]) == pytest.approx(1.04, abs=1e-8)


def test_a_local_generator_of_a_group():
    """generator 1 stands on the bus generator 3 regulates remotely: both are of the group"""
    bus_of_1 = _base().get_generators()[1].bus_id
    grid = _grid_group(group=(1, 3), reg_bus=bus_of_1)
    trips = [[1], [3], [1, 3]]
    session, V = _gens_in_analysis(grid, trips)
    assert not any(session.get_batch_unsupported_reasons())
    for r, gens in enumerate(trips):
        np.testing.assert_allclose(V[r], _reference(grid, gens=gens), rtol=0., atol=1e-7, err_msg=f"{gens}")


def test_a_row_that_trips_no_controller_next_to_one_that_does():
    grid = _grid_group()
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=15, tol_base=TOL)
    ca.add_contingencies([([], [], [], [1]), ([5], [], [], [])])
    ca.compute(batch_size=2)
    V = ca.V_results.to_numpy().reshape(2, -1)
    np.testing.assert_allclose(V[0], _reference(grid, gens=[1]), rtol=0., atol=1e-7)
    np.testing.assert_allclose(V[1], _reference(grid, branches=[5]), rtol=0., atol=1e-7)


def test_an_islanded_controller_next_to_a_tripped_one():
    """the islanded one is gone too: the row trips the live controller, nobody is left"""
    n_line = len(_base().get_lines())
    grid = _grid_group(group=(2, 3), reg_bus=8)
    trafo = [n_line + t.id for t in grid.get_trafos() if 7 in (t.bus1_id, t.bus2_id)][0]
    from gpusim2grid import ContingencyAnalysisGPU
    ca = ContingencyAnalysisGPU(grid, nb_iter=15, tol_base=TOL, handle_disconnected_grid=True)
    ca.add_contingencies([([trafo], [], [], [2])])
    ca.compute(batch_size=1)
    V = ca.V_results.to_numpy().reshape(1, -1)[0]
    assert not any(ca.get_batch_unsupported_reasons())
    ref = copy.deepcopy(grid)
    ref.deactivate_trafo(trafo - n_line)
    ref.deactivate_gen(2)
    ref.consider_only_main_component(False)
    ref.tell_solver_need_reset()
    V_ref = ref.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL)
    assert V_ref.size > 0
    buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
    live = np.isfinite(V)
    assert not live[list(buses).index(7)]
    np.testing.assert_allclose(V[live], V_ref[buses][live], rtol=0., atol=1e-7)


def test_a_voltage_mode_svc():
    grid = _grid_svc()
    trips = [([VOLTAGE], []), ([VOLTAGE, FIXED_Q], []), ([VOLTAGE], [5])]
    for runner in (_analysis, _sweep):
        session, V = runner(grid, trips, SVC)
        assert not any(session.get_batch_unsupported_reasons())
        for r, (ids, branches) in enumerate(trips):
            np.testing.assert_allclose(V[r], _reference(grid, SVC, ids, branches), rtol=0., atol=1e-7, err_msg=f"{ids}")


def test_stations_of_a_control_group():
    grid = _grid_hvdc_station_in_group()
    other_gen_trips = [([0], []), ([0, 1], [])]
    session, V = _analysis(grid, other_gen_trips, HVDC)
    assert not any(session.get_batch_unsupported_reasons())
    for r, (ids, _) in enumerate(other_gen_trips):
        np.testing.assert_allclose(V[r], _reference(grid, HVDC, ids), rtol=0., atol=1e-7, err_msg=f"{ids}")


def test_with_the_outer_loops_a_generator_of_a_group():
    from lightsim2grid.algorithm import ReactiveLimits
    grid = _grid_group(algo=OUTER)
    grid.clear_outer_loops()
    grid.add_outer_loop(ReactiveLimits())
    _finish(grid, OUTER)
    session, V = _gens_in_analysis(grid, GROUP_TRIPS, outer_loops=True, scheduling="continuous", nb_iter=MAX_IT,
                                   ir_n_steps=1, init_from_n_powerflow=False, tol=1e-8 / grid.get_sn_mva())
    assert not any(session.get_batch_unsupported_reasons()), session.get_batch_unsupported_reasons()
    for r, gens in enumerate(GROUP_TRIPS):
        np.testing.assert_allclose(V[r], _reference(grid, gens=gens), rtol=0., atol=1e-6, err_msg=f"row {r}: {gens}")


def _grid_two_controllers_on_a_bus(algo):
    """two generators on one bus regulating a remote one, with a third elsewhere, the group short of
    reactive power: ReactiveLimits freezes the bus the row keeps a unit on"""
    import pandapower as pp
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.algorithm import ReactiveLimits
    net = _net()
    bus = int(net.gen.bus.iloc[1])
    pp.create_gen(net, bus, p_mw=5., vm_pu=float(net.gen.vm_pu.iloc[1]), min_q_mvar=-5., max_q_mvar=5.)
    net.gen.loc[net.gen.index[1], ["min_q_mvar", "max_q_mvar"]] = [-5., 5.]
    net.gen.loc[net.gen.index[2], ["min_q_mvar", "max_q_mvar"]] = [-5., 5.]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    on_bus = [g.id for g in grid.get_generators() if g.bus_id == bus]
    other = next(g.id for g in grid.get_generators() if not g.is_slack and g.bus_id != bus)
    assert len(on_bus) == 2
    group = on_bus + [other]
    for g in group:
        grid.set_gen_regulated_bus(g, REG_BUS)
        grid.change_v_gen(g, 1.06)
    grid.clear_outer_loops()
    grid.add_outer_loop(ReactiveLimits())
    return _finish(grid, algo), on_bus


def test_a_tripped_controller_stays_out_when_its_bus_is_frozen():
    """found on a real grid: ReactiveLimits freezing the bus also wrote its limit over the
    controller the row had tripped, which came back to life at that limit"""
    grid, on_bus = _grid_two_controllers_on_a_bus(OUTER)
    trips = [[on_bus[0]], [on_bus[1]]]
    session, V = _gens_in_analysis(grid, trips, outer_loops=True, scheduling="continuous", nb_iter=MAX_IT, ir_n_steps=1,
                                   init_from_n_powerflow=False, keep_outer_trace=True, tol=1e-8 / grid.get_sn_mva())
    assert not any(session.get_batch_unsupported_reasons()), session.get_batch_unsupported_reasons()
    res = session.get_outer_results()
    for r, gens in enumerate(trips):
        assert any("PV_TO_PQ" in d and "not taken" not in d for d in res[r].decisions), res[r].decisions
        np.testing.assert_allclose(V[r], _reference(grid, gens=gens), rtol=0., atol=1e-6, err_msg=f"{gens}")


def test_controllers_follow_their_row_through_the_continuous_tail():
    from gpusim2grid import ContingencyAnalysisGPU
    grid = _grid_group()
    conts = [([b], [], [], [1 + (b % 3)]) for b in range(7)] * 2
    ca = ContingencyAnalysisGPU(grid, scheduling="continuous", nb_iter=MAX_IT, tol=1e-9 / grid.get_sn_mva(), tol_base=TOL)
    ca.tail_max_groups = 1
    ca.tail_group_size = 4
    ca.add_contingencies(conts)
    ca.compute(batch_size=12)
    assert ca.timings.n_slot_moves > 0
    V = ca.V_results.to_numpy().reshape(len(conts), grid.get_Ybus_solver().shape[0])
    for r, c in enumerate(conts):
        np.testing.assert_allclose(V[r], _reference(grid, gens=c[3], branches=c[0]), rtol=0., atol=1e-7, err_msg=f"{c}")


def test_a_sweep_can_drop_and_regain_its_controller_trips():
    """the entries a tripped controller needs are reserved from the masks, on every run: the same
    session then matches a one-off powerflow with the trips gone and back"""
    from gpusim2grid import ScenarioSweepGPU
    grid = _grid_group()
    load_p = np.asarray(grid.get_load_target_p(), dtype=float)[None, :]
    load_q = np.array([el.target_q_mvar for el in grid.get_loads()], dtype=float)[None, :]
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)[None, :]
    sw = ScenarioSweepGPU(grid, nb_iter=15, tol_base=TOL)
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    mask = np.zeros_like(gen_p, dtype=bool)
    for trip in (True, False, True):
        mask[0, 1] = trip
        sw.set_contingency_gens(mask)
        sw.compute(batch_size=1)
        V = sw.solver.V_results.to_numpy().reshape(1, -1)[0]
        np.testing.assert_allclose(V, _reference(grid, gens=[1] if trip else []), rtol=0., atol=1e-7, err_msg=f"{trip}")


def test_a_controller_with_physical_checks_is_reported():
    grid = _grid_group()
    session, V = _gens_in_analysis(grid, [[1]], compute_physical_violations=True)
    assert "physical" in session.get_batch_unsupported_reasons()[0]


def test_a_controller_with_gradients_is_reported():
    """BatchPowerFlow's rows, backward included, do not know how to read such a row yet"""
    from gpusim2grid import ScenarioSweepGPU
    grid = _grid_group()
    gen_p = np.asarray(grid.get_gen_target_p(), dtype=float)[None, :]
    load_p = np.asarray(grid.get_load_target_p(), dtype=float)[None, :]
    load_q = np.array([el.target_q_mvar for el in grid.get_loads()], dtype=float)[None, :]
    sw = ScenarioSweepGPU(grid, nb_iter=15, tol_base=TOL)
    sw.set_injections_from_elements(load_p, load_q, gen_p)
    mask = np.zeros_like(gen_p, dtype=bool)
    mask[0, 1] = True
    sw.set_contingency_gens(mask)
    sw.solver.keep_final_jacobian = True
    sw.compute(batch_size=1)
    assert "gradient" in sw.get_batch_unsupported_reasons()[0]


@pytest.mark.parametrize("kind, make, ids, lost", [
    (SGEN, _grid_sgen, [0], 5.), (STORAGE, _grid_storage, [PQ], 3.), (HVDC, _grid_hvdc, [FIXED], None)],
    ids=[SGEN, STORAGE, HVDC])
def test_redistribute_slack_counts_what_the_element_loses(kind, make, ids, lost):
    """the pre-pass' term: a static generator's P, minus a storage unit's (load convention:
    the -3 MW unit produced 3 MW), both converter stations' set-points"""
    grid = make()
    nb_gen = len(grid.get_generators())
    for g in grid.get_generators():
        grid.add_gen_slackbus(g.id, 1.)
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 1e3))
    _finish(grid)
    ca, V = _analysis(grid, [(ids, [])], kind, redistribute_slack=True)
    assert not any(ca.get_batch_unsupported_reasons())
    report = ca.get_slack_redistribution_report()
    mismatch = report["mismatch_mw"][0]
    if lost is None:
        # both stations' set-points: what the row takes out of the grid's balance
        ref = copy.deepcopy(grid)
        _deactivate(ref, kind, ids[0])
        ref.tell_solver_need_reset()
        assert ref.ac_pf(np.ones(grid.total_bus(), dtype=complex), MAX_IT, TOL).shape[0] > 0
        expected = float(np.sum(np.real(grid.get_Sbus_solver())) - np.sum(np.real(ref.get_Sbus_solver()))) * grid.get_sn_mva()
        assert mismatch == pytest.approx(expected, abs=1e-7)
        # the line's own losses come back: the units at 0 MW cannot go lower and leave the
        # slack, so this row is not the plain one-off solve
        assert mismatch < 0. and report["nb_saturated"][0] > 0
        return
    assert mismatch == pytest.approx(lost, abs=1e-8)
    # no limit is reached: the pre-pass does not change the converged state
    np.testing.assert_allclose(V[0], _reference(grid, kind, ids), rtol=0., atol=1e-6)


def test_redistribute_slack_a_storage_unit_of_the_slack_leaves_it_on_a_sweep():
    """the pre-pass' weights too: a tripped storage unit keeps no share of the row's slack"""
    grid = _grid_storage()
    for gen in grid.get_generators():
        grid.add_gen_slackbus(gen.id, 1.)
    grid.add_storage_slackbus(PQ, 1.)
    nb_gen = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 1e3))
    _finish(grid)
    session, V = _sweep(grid, [([PQ], [])], STORAGE, redistribute_slack=True)
    np.testing.assert_allclose(V[0], _reference(grid, STORAGE, [PQ]), rtol=0., atol=1e-6)


def test_the_ten_tuple_and_the_shortcuts():
    from gpusim2grid import ContingencyAnalysisGPU
    grid = _grid_sgen()
    ca = ContingencyAnalysisGPU(grid, nb_iter=15, tol_base=TOL)
    ca.add_contingency_sgen(1)
    ca.add_contingency(branch_ids=[5], sgen_ids=[0, 1])
    assert ca.get_contingencies() == [([], [], [], [], [], [], [1], [], [], []),
                                      ([5], [], [], [], [], [], [0, 1], [], [], [])]
    ca.compute(batch_size=2)
    V = ca.V_results.to_numpy().reshape(2, -1)
    np.testing.assert_allclose(V[0], _reference(grid, SGEN, [1]), rtol=0., atol=1e-7)
    np.testing.assert_allclose(V[1], _reference(grid, SGEN, [0, 1], [5]), rtol=0., atol=1e-7)
    with pytest.raises(Exception):
        ca.add_contingency_sgen(2)


@pytest.mark.parametrize("kind, make, ids", [(STORAGE, _grid_storage, [PQ]), (STORAGE, _grid_storage, [REGULATING]),
                                             (HVDC, _grid_hvdc, [FIXED])],
                         ids=["storage_pq", "storage_regulating", "hvdc_fixed"])
def test_with_the_outer_loops(kind, make, ids):
    """the row is lightsim2grid's one-off NROuter solve of the grid without the element"""
    from lightsim2grid.algorithm import ReactiveLimits
    grid = make()
    grid.clear_outer_loops()
    grid.add_outer_loop(ReactiveLimits())
    _finish(grid, OUTER)
    session, V = _analysis(grid, [(ids, [])], kind, outer_loops=True, scheduling="continuous", nb_iter=MAX_IT,
                           ir_n_steps=1, init_from_n_powerflow=False, tol=1e-8 / grid.get_sn_mva())
    if kind == STORAGE and ids == [REGULATING] and session.get_batch_unsupported_reasons()[0]:
        assert "reserved" in session.get_batch_unsupported_reasons()[0]
        return
    assert not any(session.get_batch_unsupported_reasons())
    np.testing.assert_allclose(V[0], _reference(grid, kind, ids), rtol=0., atol=1e-6)


def _grid_droop_only(algo="NR_KLU", p_max=100., limits=True):
    """one line in AC emulation between buses 9 and 3, no station regulating a voltage"""
    grid = _base()
    big = np.full(1, 1e3)
    grid.init_hvdc_lines(np.array([9], dtype=np.int32), np.array([3], dtype=np.int32), [0], [0], np.ones(1), np.ones(1),
                         [False], [False], np.ones(1), np.ones(1), np.zeros(1), np.zeros(1), -big, big, -big, big,
                         np.ones(1), np.ones(1), [0], np.array([10.]), np.ones(1), np.full(1, 12.),
                         [True], np.array([5.]), np.array([2.]), np.full(1, p_max), np.full(1, p_max))
    if limits:
        from lightsim2grid.algorithm import HvdcAcEmulationLimits, ReactiveLimits
        grid.clear_outer_loops()
        grid.add_outer_loop(HvdcAcEmulationLimits())
        grid.add_outer_loop(ReactiveLimits())
    return _finish(grid, algo)


def test_a_line_in_ac_emulation_with_the_outer_loops():
    """the row is a one-off NROuter solve of the grid without the line; its limits loop skips it
    (the second row, keeping the line, is the one-off solve of the grid itself)"""
    grid = _grid_droop_only(OUTER)
    session, V = _analysis(grid, [([0], []), ([], [5])], HVDC, outer_loops=True, scheduling="continuous",
                           nb_iter=MAX_IT, ir_n_steps=1, init_from_n_powerflow=False, tol=1e-8 / grid.get_sn_mva())
    assert not any(session.get_batch_unsupported_reasons())
    np.testing.assert_allclose(V[0], _reference(grid, HVDC, [0]), rtol=0., atol=1e-6)
    np.testing.assert_allclose(V[1], _reference(grid, branches=[5]), rtol=0., atol=1e-6)


def test_a_saturated_line_can_be_tripped():
    """at its limit, the line injects pmax: the trip takes that out too"""
    grid = _grid_droop_only(p_max=1., limits=False)
    grid.set_status_droop_hvdc(0, 1)
    _finish(grid)
    session, V = _analysis(grid, [([0], [])], HVDC)
    np.testing.assert_allclose(V[0], _reference(grid, HVDC, [0]), rtol=0., atol=1e-7)


def test_physical_checks_report_a_line_in_ac_emulation():
    grid = _grid_droop_only(limits=False)
    session, V = _analysis(grid, [([0], [])], HVDC, compute_physical_violations=True)
    assert "physical" in session.get_batch_unsupported_reasons()[0]


def test_a_regulating_station_with_the_outer_loops_is_reported():
    """OuterRowView has no way to say a line is out before lightsim2grid #292"""
    from lightsim2grid.algorithm import ReactiveLimits
    grid = _grid_hvdc()
    grid.clear_outer_loops()
    grid.add_outer_loop(ReactiveLimits())
    _finish(grid, OUTER)
    session, V = _analysis(grid, [([HV_REGULATING], [])], HVDC, outer_loops=True, scheduling="continuous",
                           nb_iter=MAX_IT, ir_n_steps=1, init_from_n_powerflow=False)
    assert "outer" in session.get_batch_unsupported_reasons()[0]
    assert np.isnan(V[0]).all()


@pytest.mark.parametrize("kind, make, ids", [(STORAGE, _grid_storage, [REGULATING]),
                                             (HVDC, _grid_hvdc, [HV_REGULATING])], ids=[STORAGE, HVDC])
def test_physical_checks_report_the_rows_they_cannot_check(kind, make, ids):
    grid = make()
    session, V = _analysis(grid, [(ids, [])], kind, compute_physical_violations=True)
    assert "physical" in session.get_batch_unsupported_reasons()[0]


def test_a_tripped_line_follows_its_row_through_the_continuous_tail():
    """the per-slot status moves with its row when the tail compacts the slots"""
    from gpusim2grid import ContingencyAnalysisGPU
    grid = _grid_droop_only(limits=False)
    branches = list(range(7))
    conts = [_ctg(HVDC, [0], [b]) for b in branches] + [_ctg(HVDC, [], [b]) for b in branches]
    ca = ContingencyAnalysisGPU(grid, scheduling="continuous", nb_iter=MAX_IT, tol=1e-9 / grid.get_sn_mva(),
                                tol_base=TOL)
    ca.tail_max_groups = 1
    ca.tail_group_size = 4
    ca.add_contingencies(conts)
    ca.compute(batch_size=12)
    assert ca.timings.n_slot_moves > 0
    V = ca.V_results.to_numpy().reshape(len(conts), grid.get_Ybus_solver().shape[0])
    for r, c in enumerate(conts):
        ids, br = c[TUPLE_POS[HVDC]], c[0]
        np.testing.assert_allclose(V[r], _reference(grid, HVDC, ids, br), rtol=0., atol=1e-7, err_msg=f"row {r}")
