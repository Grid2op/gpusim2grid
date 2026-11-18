# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Gradients through the outer loops with ``outer_loops_backend='device'``
(``BatchPowerFlow.from_lsgrid(outer_loops=True, outer_loops_backend='device')``): every row's
adjoint view of its last solve (``pq_buses``, ``vset_lost_groups``, ``gen_p_fixed``,
``gen_p_slope``) and so every gradient the same as ``'host'`` -- whose gradients
test_outer_gradients.py checks against central differences."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _outer_device_helpers import needs_outer_loops

from conftest import requires_gpu

pytestmark = [needs_outer_loops, requires_gpu]

GTOL: float = 1e-8


TOL_MVA: float = 1e-8
N_ROWS: int = 6


def _grad_grid(
    distributed: bool = False, reactive_limits: bool = True, moved_fraction: float | None = None
) -> Any:
    """test_outer_gradients.py's grid (that module needs torch): case14 short of reactive power,
    with ``distributed`` the slack shared in the Newton, the first generator capped just above
    its target"""
    import warnings

    import pandapower.networks as pn
    from lightsim2grid.algorithm import DistributedSlack, ReactiveLimits
    from lightsim2grid.network import init_from_pandapower

    net = pn.case14()
    net.gen["min_q_mvar"] = -15.0
    net.gen["max_q_mvar"] = 15.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    p_max = np.full(nb_gen, 300.0)
    if distributed:
        grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.0))
        p_max[0] = float(grid.get_gen_target_p()[0]) + 1.0
    grid.set_gen_p_limits(np.zeros(nb_gen), p_max)
    grid.clear_outer_loops()
    if reactive_limits:
        grid.add_outer_loop(ReactiveLimits())
    if distributed:
        grid.add_outer_loop(
            DistributedSlack() if moved_fraction is None else DistributedSlack(moved_fraction=moved_fraction)
        )
    grid.change_algorithm("NROuterDist_KLU" if distributed else "NROuter_KLU")
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, TOL_MVA).size > 0
    return grid


def _element_rows(grid: Any, zero_gen_p_to: float | None = None) -> dict[str, np.ndarray]:
    """test_outer_gradients.py's _inputs"""
    rng = np.random.default_rng(1)
    base_lp = np.asarray(grid.get_load_target_p(), dtype=float)
    base_gp = np.asarray(grid.get_gen_target_p(), dtype=float)
    if zero_gen_p_to is not None:
        base_gp = np.where(base_gp == 0.0, zero_gen_p_to, base_gp)
    load_p = base_lp[None, :] * (1.1 + 0.1 * rng.standard_normal((N_ROWS, base_lp.size)))
    return {
        "load_p": load_p,
        "load_q": 0.3 * load_p,
        "gen_p": base_gp[None, :] * (1.0 + 0.05 * rng.standard_normal((N_ROWS, base_gp.size))),
    }


def _same_adjoint_rows(
    grid: Any, rows: dict[str, np.ndarray], gen_off: np.ndarray | None = None
) -> list[Any]:
    """a scenario sweep keeping what the adjoint needs (BatchPowerFlow's session setting): the
    rows' adjoint view of their last solve, host vs device"""
    from gpusim2grid import ScenarioSweepGPU

    def run(backend: str) -> Any:
        ss = ScenarioSweepGPU(
            grid,
            outer_loops=True,
            nb_iter=30,
            tol=TOL_MVA / grid.get_sn_mva(),
            ir_n_steps=1,
            reference_slack="grid",
            outer_loops_backend=backend,
        )
        ss._inner.keep_final_jacobian = True
        ss.set_injections_from_elements(rows["load_p"], rows["load_q"], rows["gen_p"])
        if gen_off is not None:
            ss.set_contingency_gens(gen_off)
        ss.compute(batch_size=4)
        return ss

    host, dev = run("host"), run("device")
    np.testing.assert_array_equal(np.asarray(dev.get_row_status()), np.asarray(host.get_row_status()))
    res_h, res_d = host.get_outer_results(), dev.get_outer_results()
    _compare_results(res_h, res_d)
    n = rows["gen_p"].shape[0]
    conv = np.asarray(host.get_row_status()) == 0
    np.testing.assert_allclose(
        dev.V_results.to_numpy().reshape(n, -1)[conv],
        host.V_results.to_numpy().reshape(n, -1)[conv],
        rtol=0,
        atol=1e-9,
    )
    return res_h


def _compare_results(res_h: list[Any], res_d: list[Any]) -> None:
    for k, (rh, rd) in enumerate(zip(res_h, res_d, strict=True)):
        assert (rd.status, rd.nb_outer_iterations, rd.nb_nr_iterations) == (
            rh.status,
            rh.nb_outer_iterations,
            rh.nb_nr_iterations,
        ), f"row {k}"
        assert list(rd.pq_buses) == list(rh.pq_buses), f"row {k}: pq_buses"
        assert list(rd.vset_lost_groups) == list(rh.vset_lost_groups), f"row {k}: vset_lost_groups"
        assert list(rd.gen_p_fixed) == list(rh.gen_p_fixed), f"row {k}: gen_p_fixed"
        assert [(g, r) for g, r, _ in rd.gen_p_slope] == [(g, r) for g, r, _ in rh.gen_p_slope], f"row {k}"
        np.testing.assert_allclose(
            [c for _, _, c in rd.gen_p_slope], [c for _, _, c in rh.gen_p_slope], rtol=0, atol=1e-12
        )


def _curves(grid: Any) -> Any:
    grid.set_gen_capability_curves(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 100.0, 0.0, 100.0]),
        np.array([-10.0, -30.0, -10.0, -30.0]),
        np.array([5.0, 25.0, 5.0, 25.0]),
    )
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, TOL_MVA).size > 0
    return grid


def test_adjoint_rows_reactive_limits():
    grid = _grad_grid()
    res = _same_adjoint_rows(grid, _element_rows(grid))
    assert any(r.pq_buses for r in res)


def test_adjoint_rows_distributed_slack():
    grid = _grad_grid(distributed=True)
    res = _same_adjoint_rows(grid, _element_rows(grid, zero_gen_p_to=5.0))
    assert any(0 in r.gen_p_fixed for r in res)


def test_adjoint_rows_parked_at_the_last_check():
    grid = _grad_grid(distributed=True, reactive_limits=False, moved_fraction=1e9)
    _same_adjoint_rows(grid, _element_rows(grid, zero_gen_p_to=5.0))


def test_adjoint_rows_capability_curve_slope():
    grid = _curves(_grad_grid())
    res = _same_adjoint_rows(grid, _element_rows(grid))
    assert any(r.gen_p_slope for r in res)


def test_adjoint_rows_generator_contingency():
    grid = _grad_grid()
    rows = _element_rows(grid)
    gen_off = np.zeros_like(rows["gen_p"], dtype=bool)
    gen_off[1, 2] = True
    _same_adjoint_rows(grid, rows, gen_off)


# ---- the gradients themselves (torch) --------------------------------------------------------------


def _pair(grid: Any, **kwargs: Any) -> tuple[Any, Any]:
    import test_outer_gradients as og

    from gpusim2grid.differentiable import BatchPowerFlow

    def make(backend: str) -> Any:
        return BatchPowerFlow.from_lsgrid(
            grid,
            outer_loops=True,
            nb_iter=30,
            tol=og.TOL_MVA / grid.get_sn_mva(),
            ir_n_steps=1,
            outer_loops_backend=backend,
            **kwargs,
        )

    return make("host"), make("device")


def _same_gradients(grid: Any, inputs: dict[str, np.ndarray], **call: Any) -> list[Any]:
    torch = pytest.importorskip("torch")
    import test_outer_gradients as og

    host, dev = _pair(grid)
    V_h, g_h = og._grads(host, inputs, **call)
    V_d, g_d = og._grads(dev, inputs, **call)
    res_h, res_d = host.sweep.get_outer_results(), dev.sweep.get_outer_results()
    _compare_results(res_h, res_d)
    torch.testing.assert_close(V_d, V_h, rtol=0, atol=1e-10, equal_nan=True)
    for name in g_h:
        np.testing.assert_allclose(g_d[name], g_h[name], rtol=0, atol=GTOL, err_msg=name)
    return res_h


def test_gradients_reactive_limits():
    import test_outer_gradients as og

    grid = og._grid()
    res = _same_gradients(grid, og._inputs(grid))
    assert any(r.pq_buses for r in res)


def test_gradients_distributed_slack():
    import test_outer_gradients as og

    grid = og._grid(distributed=True)
    res = _same_gradients(grid, og._inputs(grid, zero_gen_p_to=5.0))
    assert any(0 in r.gen_p_fixed for r in res)


def test_gradients_parked_at_the_last_check():
    import test_outer_gradients as og

    grid = og._grid(distributed=True, reactive_limits=False, moved_fraction=1e9)
    _same_gradients(grid, og._inputs(grid, zero_gen_p_to=5.0))


def test_gradients_capability_curve_slope():
    import test_outer_gradients as og

    grid = og._grid()
    grid.set_gen_capability_curves(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 100.0, 0.0, 100.0]),
        np.array([-10.0, -30.0, -10.0, -30.0]),
        np.array([5.0, 25.0, 5.0, 25.0]),
    )
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, og.TOL_MVA).size > 0
    res = _same_gradients(grid, og._inputs(grid))
    assert any(r.gen_p_slope for r in res)


def test_gradients_generator_contingency():
    torch = pytest.importorskip("torch")
    import test_outer_gradients as og

    grid = og._grid()
    inputs = og._inputs(grid)
    gen_status = np.ones_like(inputs["gen_p"], dtype=bool)
    gen_status[1, 2] = False
    _same_gradients(grid, inputs, gen_status=torch.as_tensor(gen_status, device="cuda"))
