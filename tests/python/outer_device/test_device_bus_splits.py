# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Bus-split strategies (spare buses) with ``outer_loops_backend='device'``: the same rows as
``'host'`` (test_topo_outer_warm_start.py's grid and splits, the host checked there against
lightsim2grid)."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import MAX_IT, OUTER, TOL_MVA, needs_fp64, needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops, needs_fp64, requires_gpu]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "topology_actions"))

SPLIT_CONTINGENCIES: list[list[int]] = [[], [3], [6]]
VTOL: float = 1e-9


@pytest.fixture(scope="module")
def grid() -> Any:
    import pandapower.networks as pn

    from _topo_helpers import _two_busbars

    net = pn.case14()
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        g = _two_busbars(net)
    n_gen = len(g.get_generators())
    g.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    g.set_gen_can_participate_slack([True] * n_gen, np.full(n_gen, 300.0))
    g.change_algorithm("NROuter_KLU")
    g.tell_solver_need_reset()
    assert g.ac_pf(np.full(g.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    return g


def _same_strategies(
    grid: Any,
    actions: list[Any],
    warm: bool,
    trace: bool = False,
    handle_disconnected_grid: bool = False,
    contingencies: list[list[int]] | None = None,
) -> tuple[Any, Any]:
    from gpusim2grid import ContingencyAnalysisGPU

    contingencies = SPLIT_CONTINGENCIES if contingencies is None else contingencies
    strategies = [(c, [a]) for c in range(len(contingencies)) for a in range(len(actions))]

    def run(backend: str) -> Any:
        ca = ContingencyAnalysisGPU(
            grid,
            outer_loops=True,
            nb_iter=MAX_IT,
            tol=TOL_MVA / grid.get_sn_mva(),
            init_from_n_powerflow=False,
            scheduling="continuous",
            strategy_warm_start=warm,
            outer_loops_backend=backend,
            keep_outer_trace=trace,
            handle_disconnected_grid=handle_disconnected_grid,
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id(contingencies)
        ca.set_actions(actions)
        ca.add_strategies(strategies)
        ca.compute(batch_size=4)
        return ca

    host, dev = run("host"), run("device")
    st_h = np.asarray(host.get_row_status())
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), st_h)
    n = st_h.size
    conv = st_h == 0
    np.testing.assert_allclose(
        dev.V_results.to_numpy().reshape(n, -1)[conv],
        host.V_results.to_numpy().reshape(n, -1)[conv],
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
        if trace:
            from test_device_facades import _same_decisions

            _same_decisions(list(rh.decisions), list(rd.decisions), f"row {k}")
    return host, dev


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "from_scratch"])
def test_bus_splits_device_equals_host(grid: Any, warm: bool):
    """branches moved to a spare bus (the session's buses grow: set_spare_buses)"""
    from test_topo_outer_warm_start import _splits

    _same_strategies(grid, _splits(grid), warm)


def test_generator_moves_device_equals_host():
    """a generator moved to a spare bus with a branch end: its new bus regulated (ReactiveLimits
    on it), its former one PQ -- the row's own ReactiveLimits plan"""
    from test_topo_outer_warm_start import _case14_gen_moves, _gen_moves

    g = _case14_gen_moves()
    _same_strategies(g, _gen_moves(g), False)


def test_moved_slack_participants_device_equals_host(grid: Any):
    """generators 1-3 moved to a spare bus each, on the module's grid where every generator takes
    part in the slack: DistributedSlack moves a moved generator's target on its new bus (the host's
    ds_unit_bus, as lightsim2grid's solve-view units), on the device too. The device read the
    grid's bus of every unit, so it put a moved one's share back on its former bus (and the
    NROuterDist weights with it)"""
    from test_topo_outer_warm_start import _gen_moves

    actions = _gen_moves(grid)
    moved = [int(grid.get_generators()[g].id) for g in (1, 2, 3)]
    host, dev = _same_strategies(grid, actions, False, trace=True)
    n_ctg = len(SPLIT_CONTINGENCIES)
    rows_h, rows_d = host.get_outer_results(), dev.get_outer_results()
    shared = 0
    for c in range(n_ctg):
        for a, gen in enumerate(moved):
            k = n_ctg + c * len(actions) + a
            th, td = dict(rows_h[k].gen_target_p), dict(rows_d[k].gen_target_p)
            assert td.keys() == th.keys(), f"row {k}"
            for g, mw in th.items():
                assert td[g] == pytest.approx(mw, rel=0, abs=1e-6), f"row {k}, generator {g}"
            shared += gen in th
    # the loop did move the moved generators (else this compares nothing the bug touches)
    assert shared > 0


def _case14_gen_on_busbar2() -> Any:
    """test_topo_outer_warm_start.py's generator-move grid with generator 2 standing on the second
    busbar of its substation (bus 19, with one line end), so a regulated bus of the grid comes after
    a spare bus another generator may move to (lightsim2grid orders its ReactiveLimits plan by bus
    id: a real grid's spare buses sit between its in-service ones)"""
    import pandapower as pp
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    net = pn.case14()
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    n_sub = len(net.bus)
    for b in range(n_sub):
        pp.create_bus(net, vn_kv=net.bus.vn_kv.iloc[b], in_service=False)
    sub = int(net.gen.bus.iloc[2])
    net.bus.loc[n_sub + sub, "in_service"] = True
    net.gen.loc[net.gen.index[2], "bus"] = n_sub + sub
    line = int(net.line.index[(net.line.from_bus == sub) | (net.line.to_bus == sub)][0])
    end = "from_bus" if int(net.line.from_bus.loc[line]) == sub else "to_bus"
    net.line.loc[line, end] = n_sub + sub
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        g = init_from_pandapower(net, n_sub=n_sub, n_busbar_per_sub=2)
    n_gen = len(g.get_generators())
    g.set_gen_p_limits(np.zeros(n_gen), np.full(n_gen, 300.0))
    participates = [gen.id == 0 or gen.is_slack for gen in g.get_generators()]
    g.set_gen_can_participate_slack(participates, np.full(n_gen, 300.0))
    g.change_algorithm(OUTER)
    g.tell_solver_need_reset()
    assert g.ac_pf(np.full(g.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA).size > 0
    assert g.get_generators()[2].bus_id == n_sub + sub
    return g


def test_generator_moved_before_a_regulated_bus_device_equals_host():
    """generator 1 moved to a spare bus whose id is below a bus the grid regulates (bus 19): the
    row's ReactiveLimits plan lists its new bus between the grid's ones, the device keeps the row's
    order (its decisions trace too)"""
    from test_topo_bus_moves import _branch_ends, _move

    g = _case14_gen_on_busbar2()
    gen = g.get_generators()[1]
    assert int(gen.bus_id) + g.get_n_sub() < int(g.get_generators()[2].bus_id)
    action = _move(g, int(gen.bus_id), [("gen_bus", int(gen.id)), _branch_ends(g, int(gen.bus_id))[0]])
    _same_strategies(g, [action], False, trace=True)


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "from_scratch"])
def test_rows_masking_only_spare_buses_are_copied_on_the_device(grid: Any, warm: bool):
    """with handle_disconnected_grid every row masks the spare buses it leaves empty (islands of
    their own): masks that change nothing for the loops. The device copies such rows from a
    template instead of setting each up on the host: only the rows capturing a template are
    (the phantom row, the first plain row, the first strategy of each point when warm)"""
    from test_topo_outer_warm_start import _splits

    host, dev = _same_strategies(grid, _splits(grid), warm, handle_disconnected_grid=True)
    if os.environ.get("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE"):
        pytest.skip("the check mode loads the rows through the host on purpose (to compare)")
    n_rows = len(np.asarray(host.get_row_status()))
    assert host.timings.n_oc_host_loads >= n_rows
    n_points = len(SPLIT_CONTINGENCIES) if warm else 0
    assert dev.timings.n_oc_host_loads <= 2 + n_points, (dev.timings.n_oc_host_loads, n_rows)


def test_strategies_of_an_islanding_contingency_are_copied_on_the_device(grid: Any):
    """a contingency islanding a bus (the transformer feeding case14's leaf bus 7): every strategy
    resuming from it masks that bus too. Its point's template is captured from its first strategy, masks
    kept, and the others -- masking at least those buses -- are copied from it on the device"""
    from test_topo_outer_warm_start import _splits

    n_lines = len(grid.get_lines())
    leaf = [n_lines + t.id for t in grid.get_trafos() if 7 in (t.bus1_id, t.bus2_id)]
    assert len(leaf) == 1
    ctgs = [[], [3], leaf]
    _, dev = _same_strategies(grid, _splits(grid), True, handle_disconnected_grid=True, contingencies=ctgs)
    if os.environ.get("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE"):
        pytest.skip("the check mode loads the rows through the host on purpose (to compare)")
    assert dev.timings.n_oc_host_loads <= 2 + len(ctgs), dev.timings.n_oc_host_loads


@pytest.mark.parametrize("before_a_regulated_bus", [False, True], ids=["moves", "before_a_regulated_bus"])
def test_generator_moves_filter_the_grid_plan(before_a_regulated_bus: bool, monkeypatch: pytest.MonkeyPatch):
    """a row moving generators takes its ReactiveLimits plan and DistributedSlack units from the grid's
    (generators leaving / joining a bus, a bus newly regulated) instead of rebuilding them: with
    GPUSIM2GRID_OUTER_CHECK_RL_PLAN each is compared with build_bus_q_plan / collect_units (a difference
    raises), and the rows stay the host's"""
    from test_topo_bus_moves import _branch_ends, _move
    from test_topo_outer_warm_start import _case14_gen_moves, _gen_moves

    monkeypatch.setenv("GPUSIM2GRID_OUTER_CHECK_RL_PLAN", "1")
    if before_a_regulated_bus:
        g = _case14_gen_on_busbar2()
        gen = g.get_generators()[1]
        actions = [_move(g, int(gen.bus_id), [("gen_bus", int(gen.id)), _branch_ends(g, int(gen.bus_id))[0]])]
    else:
        g = _case14_gen_moves()
        actions = _gen_moves(g)
    _same_strategies(g, actions, True, handle_disconnected_grid=True)
