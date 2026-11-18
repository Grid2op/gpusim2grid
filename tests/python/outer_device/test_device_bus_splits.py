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
from _outer_device_helpers import MAX_IT, TOL_MVA, needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops, requires_gpu]

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


def _same_strategies(grid: Any, actions: list[Any], warm: bool) -> None:
    from gpusim2grid import ContingencyAnalysisGPU

    strategies = [(c, [a]) for c in range(len(SPLIT_CONTINGENCIES)) for a in range(len(actions))]

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
        )
        ca.tail_max_groups, ca.tail_group_size = 4, 1
        ca.add_contingencies_by_branch_id(SPLIT_CONTINGENCIES)
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
