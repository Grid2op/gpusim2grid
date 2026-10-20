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


def _pf(grid, init_from_n=False):
    from gpusim2grid.differentiable import BatchPowerFlow
    return BatchPowerFlow.from_lsgrid(grid, outer_loops=True, keep_outer_trace=True, nb_iter=30,
                                      tol=TOL_MVA / grid.get_sn_mva(), init_from_n_powerflow=init_from_n,
                                      ir_n_steps=1)


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


def test_refusals():
    from gpusim2grid.differentiable import BatchPowerFlow
    with pytest.raises(ValueError, match="redistribute_slack"):
        BatchPowerFlow.from_lsgrid(_grid(), outer_loops=True, redistribute_slack=True)


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
