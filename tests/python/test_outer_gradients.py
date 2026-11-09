# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Gradients through the outer loops (``BatchPowerFlow.from_lsgrid(outer_loops=True)``).

As lightsim2grid's (``BatchCPUPowerFlow`` with an ``NROuter_*`` / ``NROuterDist_*``
algorithm, its test ``TestBatchCPUPowerFlowOuterLoops``): the gradient is the one of
each row's last inner solve, every decision of its loops held -- a bus switched PQ
stays PQ, a held controller stays held, a unit the distributed slack parked stays at its
bound. Where a small enough step flips none of them, that is what a central difference
of the whole run measures, which is what these tests check, for every input.
"""

import warnings

import numpy as np
import pytest
import torch

from conftest import requires_gpu
import gpusim2grid._gpusim2grid as _cpp

pytestmark = [
    requires_gpu,
    pytest.mark.skipif(not getattr(_cpp, "have_ls2g_outer_loops", False),
                       reason="gpusim2grid built against a lightsim2grid without NROuter_*"),
    pytest.mark.skipif(bool(_cpp.is_fp32), reason="finite differences need double precision"),
]

N_SCEN = 4
TOL_MVA = 1e-10


def _grid(distributed=False, init_from_n=False):
    """lightsim2grid's test grid: case14 short of reactive power (ReactiveLimits switches buses
    PQ), its only loop -- with `distributed`, the slack shared in the Newton (NROuterDist_KLU)
    with DistributedSlack, the first generator capped just above its target, so that the
    Newton's share parks it there"""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower
    from lightsim2grid.algorithm import ReactiveLimits, DistributedSlack
    net = pn.case14()
    net.gen["min_q_mvar"] = -15.
    net.gen["max_q_mvar"] = 15.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen = len(grid.get_generators())
    p_max = np.full(nb_gen, 300.)
    if distributed:
        grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.))
        p_max[0] = float(grid.get_gen_target_p()[0]) + 1.
    grid.set_gen_p_limits(np.zeros(nb_gen), p_max)
    grid.clear_outer_loops()
    grid.add_outer_loop(ReactiveLimits())
    if distributed:
        grid.add_outer_loop(DistributedSlack())
    grid.change_algorithm("NROuterDist_KLU" if distributed else "NROuter_KLU")
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, TOL_MVA).size > 0
    return grid


def _pf(grid, init_from_n=False, **kwargs):
    from gpusim2grid.differentiable import BatchPowerFlow
    return BatchPowerFlow.from_lsgrid(grid, outer_loops=True, keep_outer_trace=True, nb_iter=30,
                                      tol=TOL_MVA / grid.get_sn_mva(), init_from_n_powerflow=init_from_n,
                                      ir_n_steps=1, **kwargs)


def _inputs(grid, zero_gen_p_to=None):
    rng = np.random.default_rng(1)
    base_lp = np.asarray(grid.get_load_target_p(), dtype=float)
    base_gp = np.asarray(grid.get_gen_target_p(), dtype=float)
    if zero_gen_p_to is not None:
        base_gp = np.where(base_gp == 0., zero_gen_p_to, base_gp)
    n_load, n_gen = base_lp.shape[0], base_gp.shape[0]
    load_p = base_lp[None, :] * (1.1 + 0.1 * rng.standard_normal((N_SCEN, n_load)))
    return {
        "load_p": load_p,
        "load_q": 0.3 * load_p,
        "gen_p": base_gp[None, :] * (1. + 0.05 * rng.standard_normal((N_SCEN, n_gen))),
        "gen_v": np.tile(np.asarray([g.target_vm_pu for g in grid.get_generators()]), (N_SCEN, 1)),
    }


def _loss(V):
    """a fixed real linear functional of V (as lightsim2grid's test)"""
    n_bus = V.shape[1]
    b = torch.arange(n_bus, dtype=torch.float64, device=V.device)
    wr, wi = 1. + 0.5 * b, -0.75 + 0.25 * b
    Vf = torch.where(torch.isfinite(V.real), V, torch.zeros_like(V))
    return (wr * Vf.real + wi * Vf.imag).sum()


def _t(x, grad=False):
    return torch.tensor(x, dtype=torch.float64, device="cuda", requires_grad=grad)


def _grads(pf, inputs, **kwargs):
    tensors = {k: _t(v, grad=True) for k, v in inputs.items()}
    V = pf(**tensors, **kwargs)
    _loss(V).backward()
    return V, {k: t.grad.detach().cpu().numpy() for k, t in tensors.items()}


def _fd(pf, inputs, name, idx, delta, **kwargs):
    out = []
    for sign in (+1., -1.):
        perturbed = {k: v.copy() for k, v in inputs.items()}
        perturbed[name][idx] += sign * delta
        with torch.no_grad():
            out.append(float(_loss(pf(**{k: _t(v) for k, v in perturbed.items()}, **kwargs))))
    return (out[0] - out[1]) / (2. * delta)


def _check_all(pf, inputs, grads, names=("load_p", "load_q", "gen_p", "gen_v"), atol=1e-6, atol_v=1e-5,
               **kwargs):
    for name in names:
        delta = 1e-6 if name == "gen_v" else 1e-4
        tol = atol_v if name == "gen_v" else atol
        g = grads[name]
        for idx in np.ndindex(*g.shape):
            fd = _fd(pf, inputs, name, idx, delta, **kwargs)
            assert g[idx] == pytest.approx(fd, abs=tol), f"{name}{idx}: adjoint {g[idx]}, central difference {fd}"


@pytest.mark.parametrize("init_from_n", [False, True])
def test_reactive_limits(init_from_n):
    """ReactiveLimits switches buses PQ in the rows: a switched bus' generators get no gen_v
    gradient, a pinned one keeps its through the pinned row"""
    grid = _grid()
    pf = _pf(grid, init_from_n)
    inputs = _inputs(grid)
    V, grads = _grads(pf, inputs)
    assert torch.isfinite(V).all()
    res = pf.sweep.get_outer_results()
    assert any(r.pq_buses for r in res), [r.decisions for r in res]
    assert (grads["gen_v"] == 0.).any() and (grads["gen_v"] != 0.).any()
    _check_all(pf, inputs, grads)


def test_distributed_slack_in_the_newton():
    """NROuterDist_*: the Newton carries the distributed slack, so does its gradient; the
    generator the share parks at its bound gets no gen_p gradient where it is parked, and
    keeps it where its own target is already beyond the bound"""
    grid = _grid(distributed=True)
    pf = _pf(grid)
    inputs = _inputs(grid, zero_gen_p_to=5.)
    cap = float(grid.get_gen_target_p()[0]) + 1.
    V, grads = _grads(pf, inputs)
    assert torch.isfinite(V).all()
    res = pf.sweep.get_outer_results()
    parked = [0 in r.gen_p_fixed for r in res]
    assert any(parked), [r.decisions for r in res]
    for r in range(N_SCEN):
        if inputs["gen_p"][r, 0] < cap:
            assert parked[r]
            assert grads["gen_p"][r, 0] == pytest.approx(0., abs=1e-12)
        else:
            assert abs(grads["gen_p"][r, 0]) > 1e-4
    _check_all(pf, inputs, grads, names=("load_p", "gen_p"))


def test_generator_contingency():
    """rows that disconnect generators (gen_status) with the loops on"""
    grid = _grid()
    pf = _pf(grid)
    inputs = _inputs(grid)
    gen_status = np.ones_like(inputs["gen_p"], dtype=bool)
    gen_status[1, 2] = False
    st = torch.as_tensor(gen_status, device="cuda")
    V, grads = _grads(pf, inputs, gen_status=st)
    assert grads["gen_p"][1, 2] == 0. and grads["gen_v"][1, 2] == 0.
    for name, idx, delta, tol in (("load_p", (1, 0), 1e-4, 1e-6), ("load_p", (0, 3), 1e-4, 1e-6),
                                  ("gen_v", (0, 1), 1e-6, 1e-5), ("gen_v", (1, 1), 1e-6, 1e-5),
                                  ("gen_v", (3, 0), 1e-6, 1e-5)):
        fd = _fd(pf, inputs, name, idx, delta, gen_status=st)
        assert grads[name][idx] == pytest.approx(fd, abs=tol), f"{name}{idx}"


def _redistribute_rows(grid):
    """_inputs' rows (their loads ~10 % up: each row loses that against the grid's set-points,
    which the share of its DistributedSlack loop pushes the capped generator 0 to its bound),
    row 1 also disconnecting generator 2, row 2 islanding the load on bus 14"""
    inputs = _inputs(grid, zero_gen_p_to=5.)
    gen_status = np.ones_like(inputs["gen_p"], dtype=bool)
    gen_status[1, 2] = False
    line_status = np.ones((N_SCEN, len(grid.get_lines())), dtype=bool)
    line_status[2, [el.id for el in grid.get_lines() if 13 in (el.bus1_id, el.bus2_id)]] = False
    return inputs, {"gen_status": torch.as_tensor(gen_status, device="cuda"),
                    "line_status": torch.as_tensor(line_status, device="cuda")}


def test_redistribute_slack_forward_is_the_sessions():
    """BatchPowerFlow measures what each row loses in torch: the same rows as ScenarioSweepGPU's
    own redistribute_slack, whose DistributedSlack loop shares it before the first solve"""
    from gpusim2grid import ScenarioSweepGPU
    grid = _grid(distributed=True)
    pf = _pf(grid, handle_disconnected_grid=True, redistribute_slack=True)
    inputs, status = _redistribute_rows(grid)
    inputs.pop("gen_v")
    with torch.no_grad():
        V = pf(**{k: _t(v) for k, v in inputs.items()}, **status).cpu().numpy()
    res = pf.sweep.get_outer_results()
    assert all(any("SHARE_LOST_P" in d for d in r.decisions) for r in res), [r.decisions for r in res]

    sw = ScenarioSweepGPU(grid, outer_loops=True, nb_iter=30, init_from_n_powerflow=False,
                          handle_disconnected_grid=True, redistribute_slack=True, ir_n_steps=1,
                          reference_slack="grid", tol=TOL_MVA / grid.get_sn_mva())
    sw.set_injections_from_elements(inputs["load_p"], inputs["load_q"], inputs["gen_p"])
    lines = status["line_status"].cpu().numpy()
    sw.set_topology([np.flatnonzero(~row).tolist() for row in lines])
    sw.set_contingency_gens(~status["gen_status"].cpu().numpy())
    sw.compute()
    V_sw = np.asarray(sw.V_results.to_numpy()).reshape(V.shape)
    np.testing.assert_array_equal(np.isfinite(V), np.isfinite(V_sw))
    fin = np.isfinite(V)
    np.testing.assert_allclose(V[fin], V_sw[fin], rtol=0., atol=1e-10)


@pytest.mark.parametrize("init_from_n", [False, True])
def test_redistribute_slack_distributed_in_the_newton(init_from_n):
    """redistribute_slack around a distributed-slack Newton: the free units take what a row
    loses by the Newton's own weights, so that share moves nothing (its derivative is the
    Newton slack's); a generator the share puts at its bound sits there whatever its set-point
    -- no gen_p gradient -- and one already beyond it keeps its own"""
    if init_from_n and not getattr(_cpp, "have_ls2g_outer_resume", False):
        pytest.skip("lightsim2grid without its loops' state accessors")
    grid = _grid(distributed=True)
    pf = _pf(grid, init_from_n, handle_disconnected_grid=True, redistribute_slack=True)
    inputs, status = _redistribute_rows(grid)
    cap = float(grid.get_gen_target_p()[0]) + 1.
    V, grads = _grads(pf, inputs, **status)
    res = pf.sweep.get_outer_results()
    assert all(any("SHARE_LOST_P" in d for d in r.decisions) for r in res), [r.decisions for r in res]
    assert torch.isfinite(V[[0, 1, 3]]).all()
    for r in range(N_SCEN):
        if inputs["gen_p"][r, 0] < cap:
            assert 0 in res[r].gen_p_fixed, res[r].decisions
            assert grads["gen_p"][r, 0] == pytest.approx(0., abs=1e-12)
        else:
            assert abs(grads["gen_p"][r, 0]) > 1e-4
    assert grads["gen_p"][1, 2] == 0. and grads["gen_v"][1, 2] == 0.
    _check_all(pf, inputs, grads, **status)


def test_redistribute_slack_turned_off():
    """turning redistribute_slack off hands the session nothing to share any more: the next
    forward is the one of a model built without it"""
    grid = _grid(distributed=True)
    inputs, status = _redistribute_rows(grid)
    tensors = {k: _t(v) for k, v in inputs.items()}
    pf = _pf(grid, handle_disconnected_grid=True, redistribute_slack=True)
    with torch.no_grad():
        pf(**tensors, **status)
        pf.redistribute_slack = False
        V = pf(**tensors, **status).cpu().numpy()
        V_ref = _pf(grid, handle_disconnected_grid=True)(**tensors, **status).cpu().numpy()
    assert not any(any("SHARE_LOST_P" in d for d in r.decisions) for r in pf.sweep.get_outer_results())
    fin = np.isfinite(V_ref)
    np.testing.assert_allclose(V[fin], V_ref[fin], rtol=0., atol=1e-10)


def test_tap_changer_settled_taps():
    """TransformerVoltageControl moves a ratio tap in the rows: the gradient is the one at the
    taps the loops settled on (the row's last Ybus, put back by the adjoint), and compute_flows
    uses those taps, as the session's own flows"""
    from test_outer_model_cpu import _tvc_grid
    from gpusim2grid.differentiable import BatchPowerFlow
    grid = _tvc_grid("T4-9-1", 12.4, 0.05)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), 30, TOL_MVA).size > 0
    pf = BatchPowerFlow.from_lsgrid(grid, outer_loops=True, keep_outer_trace=True, nb_iter=30,
                                    tol=TOL_MVA / grid.get_sn_mva(), init_from_n_powerflow=False,
                                    ir_n_steps=1)
    base_lp = np.asarray(grid.get_load_target_p(), dtype=float)
    scales = np.array([1., 1.03, 1.06])
    inputs = {"load_p": np.outer(scales, base_lp),
              "gen_v": np.tile(np.asarray([g.target_vm_pu for g in grid.get_generators()]), (len(scales), 1))}
    V, grads = _grads(pf, inputs)
    assert torch.isfinite(V).all()
    res = pf.sweep.get_outer_results()
    assert any(r.ratio_tap for r in res), [r.decisions for r in res]
    # the flows at the settled taps
    with torch.no_grad():
        V0 = pf(**{k: _t(v) for k, v in inputs.items()})
        amps = pf.compute_flows(V0)["i_or_a"].cpu().numpy()
    pf.sweep.compute_flows()
    amps_sw = np.asarray(pf.sweep.or_amps.to_numpy()).reshape(len(scales), -1)
    np.testing.assert_allclose(amps, amps_sw, rtol=1e-8, atol=1e-6)
    _check_all(pf, inputs, grads, names=("load_p", "gen_v"))


def test_capability_curve_slope():
    """a generator frozen at a limit of its capability curve: its reactive output follows its
    target P (the row's), which the gen_p gradient owes (lightsim2grid's gen_p_indirect_grad)"""
    grid = _grid()
    # the first two generators (buses 1 and 2, frozen at their max in the rows): max Q rising with P
    grid.set_gen_capability_curves(np.array([0, 0, 1, 1]), np.array([0., 100., 0., 100.]),
                                   np.array([-10., -30., -10., -30.]), np.array([5., 25., 5., 25.]))
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, TOL_MVA).size > 0
    pf = _pf(grid)
    inputs = _inputs(grid)
    V, grads = _grads(pf, inputs)
    assert torch.isfinite(V).all()
    res = pf.sweep.get_outer_results()
    assert any(r.gen_p_slope for r in res), [r.decisions for r in res]
    _check_all(pf, inputs, grads, names=("gen_p", "load_q"))


def test_distributed_slack_parked_at_a_check_that_does_not_resolve():
    """NROuterDist_*: a unit DistributedSlack parks at a check whose move is too small to
    re-solve still shared the slack in the Newton whose Jacobian the backward reads: its
    set-point reaches the voltages (lightsim2grid 46897582)"""
    from lightsim2grid.algorithm import DistributedSlack
    grid = _grid(distributed=True)
    grid.clear_outer_loops()
    grid.add_outer_loop(DistributedSlack(moved_fraction=1e9))     # no move ever re-solves
    assert grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), 30, TOL_MVA).size > 0
    pf = _pf(grid)
    inputs = _inputs(grid)
    V, grads = _grads(pf, inputs)
    assert torch.isfinite(V).all()
    res = pf.sweep.get_outer_results()
    assert any("PARK" in d for r in res for d in r.decisions), [r.decisions for r in res]
    assert all(r.nb_outer_iterations == 0 for r in res), [r.decisions for r in res]
    assert not any(r.gen_p_fixed for r in res), [r.gen_p_fixed for r in res]
    assert (grads["gen_p"][:, 0] != 0.).all()
    _check_all(pf, inputs, grads, names=("gen_p",))


def test_compute_flows_refuses_rows_it_cannot_match_to_their_taps():
    """Another row count than the forward's: nothing says which row is which, the taps the
    loops moved would be silently lost (lightsim2grid d2a80040)"""
    from test_outer_model_cpu import _tvc_grid
    from gpusim2grid.differentiable import BatchPowerFlow
    grid = _tvc_grid("T4-9-1", 12.4, 0.05)
    assert grid.ac_pf(np.full(grid.total_bus(), 1. + 0j), 30, TOL_MVA).size > 0
    pf = BatchPowerFlow.from_lsgrid(grid, outer_loops=True, nb_iter=30, tol=TOL_MVA / grid.get_sn_mva(),
                                    init_from_n_powerflow=False, ir_n_steps=1)
    base_lp = np.asarray(grid.get_load_target_p(), dtype=float)
    with torch.no_grad():
        V = pf(load_p=_t(np.outer(np.array([1., 1.03, 1.06]), base_lp)))
    assert pf._trafo_moved is not None and len(pf._trafo_moved[1]) > 0
    full = pf.compute_flows(V)["i_or_a"]
    with pytest.raises(ValueError, match="taps"):
        pf.compute_flows(V[1:])
    assert full.shape[0] == 3
