# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""gpusim2grid.differentiable.BatchPowerFlow: batched, differentiable AC power
flow driven by lightsim2grid-style per-element inputs.

Forward is checked against ScenarioSweepGPU (the numpy path of the same
session class); every gradient is checked against FINITE DIFFERENCES --
torch.autograd.gradcheck plus explicit central differences on a scalar loss
(FP64 build only: FP32 finite differences are too noisy). The reuse contract
(no cuDSS analysis / first factorization on repeated calls; the transposed
system built lazily on the first backward and only refactorized afterwards) is
asserted on the session counters.
"""
import warnings

import numpy as np
import pytest

from conftest import requires_gpu
from gpusim2grid._gpusim2grid import have_ls2g_bridge, is_fp32
from test_handle_disconnected_grid import _solved_spur_grid

pytestmark = requires_gpu

torch = pytest.importorskip("torch", reason="PyTorch not installed -- skipping")

needs_bridge = pytest.mark.skipif(
    not have_ls2g_bridge, reason="needs the lightsim2grid C++ bridge")
fp64_only = pytest.mark.skipif(
    bool(is_fp32), reason="finite-difference gradient checks need the FP64 build")

NB_ITER = 12
TOL = 1e-10
RDT = torch.float32 if is_fp32 else torch.float64


def _pf(grid, **kw):
    from gpusim2grid.differentiable import BatchPowerFlow
    kw.setdefault("nb_iter", NB_ITER)
    kw.setdefault("tol_base", TOL)
    return BatchPowerFlow.from_lsgrid(grid, **kw)


def _base_inputs(pf, n_scen, scales=None):
    scales = np.ones(n_scen) if scales is None else np.asarray(scales, dtype=np.float64)
    s = torch.tensor(scales, dtype=RDT, device="cuda")[:, None]
    load_p = (pf._load_p_base[None, :] * s).clone()
    load_q = (pf._load_q_base[None, :] * s).clone()
    gen_p = pf._gen_p_base[None, :].expand(n_scen, -1).clone()
    return load_p, load_q, gen_p


def _all_connected(pf, n_scen):
    return (torch.ones(n_scen, pf.n_line, dtype=torch.bool, device="cuda"),
            torch.ones(n_scen, pf.n_trafo, dtype=torch.bool, device="cuda"))


def _central_diff(f, x, idx, eps=1e-6):
    xp = x.detach().clone(); xp[idx] += eps
    xm = x.detach().clone(); xm[idx] -= eps
    return (f(xp) - f(xm)) / (2 * eps)


def _fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords, tol=1e-4,
              gen_status=None):
    """Central differences of a weighted |V|^2 loss vs the analytic gradient."""
    n_bus = pf.n_bus
    w = torch.linspace(0.5, 1.5, n_bus, dtype=RDT, device="cuda")

    def loss_of(lp, lq, gp, gv):
        V = pf(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv, line_status=line_status,
               gen_status=gen_status)
        valid = torch.isfinite(V.real)
        Vs = torch.where(valid, V, torch.zeros_like(V))
        return ((Vs.abs() ** 2) * w).sum()

    ins = {"load_p": load_p, "load_q": load_q, "gen_p": gen_p, "gen_v": gen_v}
    grads = {k: v.clone().requires_grad_(True) for k, v in ins.items() if v is not None}
    args = {k: grads.get(k) for k in ins}
    loss_of(args["load_p"], args["load_q"], args["gen_p"], args["gen_v"]).backward()
    for name, idx in coords:
        def f(x, name=name):
            a = {k: (v.detach() if v is not None else None) for k, v in args.items()}
            a[name] = x
            with torch.no_grad():
                return loss_of(a["load_p"], a["load_q"], a["gen_p"], a["gen_v"]).item()
        fd = _central_diff(f, grads[name], idx)
        an = grads[name].grad[idx].item()
        assert abs(an - fd) <= tol * max(1.0, abs(fd)), (name, idx, an, fd)
    return grads


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

class TestForward:
    def test_matches_scenario_sweep_mixed_batch(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        n = 4
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9, 1.05])
        line_status, trafo_status = _all_connected(pf, n)
        line_status[1, 3] = False               # row 1: trip line 3
        trafo_status[2, 0] = False              # row 2: trip trafo 0
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[3, 1] = 1.03                      # row 3: a PV set-point change
        V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_v=gen_v,
               line_status=line_status, trafo_status=trafo_status)
        assert V.shape == (n, pf.n_bus) and V.is_cuda
        assert np.all(pf.last_residuals() < 1e-6)

        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(load_p.cpu().numpy(), load_q.cpu().numpy(),
                                        gen_p.cpu().numpy())
        sw.set_topology([[], [3], [pf.n_line + 0], []])
        sw.set_gen_v(gen_v.cpu().numpy())
        Vref = torch.from_dlpack(sw.compute(batch_size=n)).cpu().numpy()
        np.testing.assert_allclose(V.cpu().numpy(), Vref, atol=solver_atol)
        # Each input differs from the others' rows (the batch is really mixed).
        assert not np.allclose(Vref[0], Vref[1]) and not np.allclose(Vref[0], Vref[2])

    def test_none_inputs_reproduce_the_base_case(self, ieee14_base_case, solver_atol):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        V = pf(gen_p=pf._gen_p_base[None, :].expand(3, -1))
        V_n = np.asarray(grid.get_V_solver())
        for r in range(3):
            np.testing.assert_allclose(V[r].cpu().numpy(), V_n, atol=solver_atol)
        with pytest.raises(ValueError, match="at least one input"):
            pf()

    def test_status_false_means_tripped(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        line_status, _ = _all_connected(pf, 2)
        line_status[0, 5] = False
        V = pf(line_status=line_status)
        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        lp, lq, gp = (a.cpu().numpy() for a in _base_inputs(pf, 2))
        sw.set_injections_from_elements(lp, lq, gp)
        sw.set_topology([[5], []])
        Vref = torch.from_dlpack(sw.compute(batch_size=2)).cpu().numpy()
        np.testing.assert_allclose(V.cpu().numpy(), Vref, atol=solver_atol)
        assert not np.allclose(Vref[0], Vref[1])

    @needs_bridge
    def test_islanded_row_is_nan(self, solver_atol):
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        pf = _pf(grid)
        line_status, _ = _all_connected(pf, 3)
        line_status[1, int(spur_line)] = False
        V = pf(line_status=line_status)
        assert np.all(np.isnan(V[1].cpu().numpy()))
        assert np.all(np.isfinite(V[[0, 2]].cpu().numpy()))
        assert list(pf.get_disconnected()) == [0, 1, 0]
        assert pf.timings.n_chunks == 1

    def test_compute_flows_batched(self, ieee14_base_case):
        from gpusim2grid.differentiable import compute_flows
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        V = pf(gen_p=pf._gen_p_base[None, :].expand(2, -1))
        flows = pf.compute_flows(V)
        assert flows["p_or_mw"].shape == (2, pf.n_branch)
        single = compute_flows(V[0], pf._yff, pf._yft, pf._ytf, pf._ytt,
                               pf._branch_from, pf._branch_to, pf._bus_vn_kv, pf.sn_mva)
        torch.testing.assert_close(flows["i_or_a"][0], single["i_or_a"])


# ---------------------------------------------------------------------------
# Reuse contract
# ---------------------------------------------------------------------------

class TestReuse:
    def test_forward_reuse_and_lazy_adjoint(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
        line_status, trafo_status = _all_connected(pf, n)
        line_status[1, 3] = False

        # forward only: the driver is built, nothing adjoint-related exists
        V = pf(load_p=load_p, gen_p=gen_p, line_status=line_status, trafo_status=trafo_status)
        t = pf.timings
        assert pf.sweep.driver_build_counter == 1 and pf.sweep.source_build_counter == 1
        assert t.t_analysis_ms > 0 and t.n_refactorize == NB_ITER - 1
        assert t.adjoint_n_analysis == 0 and not pf.sweep.solver.adjoint_ready

        # same shape, new injections -> hot path
        load_p2 = (load_p * 1.02).requires_grad_(True)
        V2 = pf(load_p=load_p2, gen_p=gen_p, line_status=line_status, trafo_status=trafo_status)
        t = pf.timings
        assert pf.sweep.driver_build_counter == 1 and pf.sweep.source_build_counter == 1
        assert t.t_analysis_ms == 0.0 and t.t_alloc_ms == 0.0 and t.t_preprocess_ms == 0.0
        assert t.t_first_factorize.wall_ms == 0.0 and t.n_refactorize == NB_ITER
        assert t.adjoint_n_analysis == 0

        # first backward: J^T built once (analysis + factorization), solved once
        V2.real.sum().backward()
        t = pf.timings
        assert pf.sweep.solver.adjoint_ready
        assert (t.adjoint_n_analysis, t.adjoint_n_factorize,
                t.adjoint_n_refactorize, t.adjoint_n_solve) == (1, 1, 0, 1)
        assert t.t_adjoint_build_ms > 0 and t.t_adjoint_first_factorize.wall_ms > 0

        # second backward on the SAME forward: solve only
        V2 = pf(load_p=load_p2, gen_p=gen_p, line_status=line_status, trafo_status=trafo_status)
        loss = V2.real.sum()
        loss.backward(retain_graph=True)
        loss.backward()
        t = pf.timings
        assert (t.adjoint_n_analysis, t.adjoint_n_factorize,
                t.adjoint_n_refactorize, t.adjoint_n_solve) == (1, 1, 1, 3)

        # new topology (warm) + backward: refactorize only
        line_status[2, 7] = False
        V3 = pf(load_p=load_p2, gen_p=gen_p, line_status=line_status, trafo_status=trafo_status)
        assert pf.sweep.driver_build_counter == 1 and pf.sweep.source_build_counter == 2
        assert pf.timings.t_analysis_ms == 0.0
        V3.real.sum().backward()
        t = pf.timings
        assert (t.adjoint_n_analysis, t.adjoint_n_factorize, t.adjoint_n_refactorize) == (1, 1, 2)

        # new n_scen: cold rebuild, the adjoint is rebuilt lazily on the next backward
        lp4 = _base_inputs(pf, 5)[0].requires_grad_(True)
        V4 = pf(load_p=lp4)
        assert pf.sweep.driver_build_counter == 2
        assert pf.timings.adjoint_n_analysis == 0
        V4.real.sum().backward()
        assert pf.timings.adjoint_n_analysis == 1

    def test_backward_after_another_forward_raises(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        load_p, _, gen_p = _base_inputs(pf, 2, [1.0, 1.1])
        lp = load_p.clone().requires_grad_(True)
        V1 = pf(load_p=lp, gen_p=gen_p)
        pf(load_p=load_p * 1.01, gen_p=gen_p)
        with pytest.raises(RuntimeError, match="snapshot_jacobian"):
            V1.real.sum().backward()

    def test_snapshot_jacobian_allows_two_forwards(self, ieee14_base_case, solver_atol):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, snapshot_jacobian=True)
        pf_ref = _pf(grid)
        load_p, _, gen_p = _base_inputs(pf, 2, [1.0, 1.1])
        gen_v = torch.full((2, pf.n_gen), 1.02, dtype=RDT, device="cuda")

        lp = load_p.clone().requires_grad_(True)
        gv = gen_v.clone().requires_grad_(True)
        V1 = pf(load_p=lp, gen_p=gen_p, gen_v=gv)
        pf(load_p=load_p * 1.3, gen_p=gen_p)              # overwrites the chunk buffers
        (V1.abs() ** 2).sum().backward()                   # ... but the snapshot survives

        lp_ref = load_p.clone().requires_grad_(True)
        gv_ref = gen_v.clone().requires_grad_(True)
        (pf_ref(load_p=lp_ref, gen_p=gen_p, gen_v=gv_ref).abs() ** 2).sum().backward()
        torch.testing.assert_close(lp.grad, lp_ref.grad, atol=10 * solver_atol, rtol=1e-6)
        torch.testing.assert_close(gv.grad, gv_ref.grad, atol=10 * solver_atol, rtol=1e-6)

    def test_non_default_stream(self, ieee14_base_case, solver_atol):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        load_p, _, gen_p = _base_inputs(pf, 3, [1.0, 1.1, 0.9])

        lp = load_p.clone().requires_grad_(True)
        V = pf(load_p=lp, gen_p=gen_p)
        V.abs().sum().backward()

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            lp_s = (load_p.clone() * 1.0).requires_grad_(True)
            V_s = pf(load_p=lp_s, gen_p=gen_p)
            V_s.abs().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(V_s, V, atol=solver_atol, rtol=0)
        torch.testing.assert_close(lp_s.grad, lp.grad, atol=10 * solver_atol, rtol=1e-6)


# ---------------------------------------------------------------------------
# Jacobian / adjoint primitives
# ---------------------------------------------------------------------------

class TestAdjointPrimitives:
    @needs_bridge
    @fp64_only
    def test_converged_jacobian_matches_single_system(self, ieee14_base_case):
        """Row 0 (base injections, no trip) of the kept batched J must equal the
        single-system converged J of AcPfNrSession on the same augmented ledger."""
        from scipy.sparse import csr_matrix
        from gpusim2grid._gpusim2grid import _make_acpf_session_from_lsgrid
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        load_p, _, gen_p = _base_inputs(pf, 2, [1.0, 1.15])
        lp = load_p.clone().requires_grad_(True)
        pf(load_p=lp, gen_p=gen_p)          # requires_grad -> keep_final_jacobian
        sol = pf.sweep.solver
        outer, inner = sol.j_skeleton()
        J_all = torch.from_dlpack(sol.j_values_dlpack()).cpu().numpy()
        assert J_all.shape == (2, sol.nnz_J)
        n = sol.dim_J
        J0 = csr_matrix((J_all[0], inner, outer), shape=(n, n)).toarray()

        ref = _make_acpf_session_from_lsgrid(grid, 30, 1e-10)
        o_r, i_r, v_r = ref.get_J()
        J_ref = csr_matrix((np.asarray(v_r), np.asarray(i_r), np.asarray(o_r)),
                           shape=(ref.dim_J, ref.dim_J)).toarray()
        assert J_ref.shape == J0.shape
        np.testing.assert_allclose(J0, J_ref, atol=1e-8, rtol=1e-8)

    @needs_bridge
    @fp64_only
    def test_solve_JT_batch_matches_spsolve(self):
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        pf = _pf(grid)
        n = 4
        load_p, _, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9, 1.05])
        line_status, _ = _all_connected(pf, n)
        line_status[1, 3] = False
        line_status[2, int(spur_line)] = False          # islanded row
        lp = load_p.clone().requires_grad_(True)
        pf(load_p=lp, gen_p=gen_p, line_status=line_status)
        sol = pf.sweep.solver
        dim_J = sol.dim_J
        rng = np.random.default_rng(0)
        rhs = rng.standard_normal((n, dim_J))
        rhs[0, 3] = np.nan                               # non-finite entries count as 0
        lam_cap, gvm = sol.solve_JT_batch_dlpack(
            torch.tensor(rhs, device="cuda").__dlpack__())
        assert gvm is None
        lam = torch.from_dlpack(lam_cap).cpu().numpy()
        outer, inner = sol.j_skeleton()
        J_all = torch.from_dlpack(sol.j_values_dlpack()).cpu().numpy()
        a2o = sol.get_active_to_orig()
        assert list(a2o) == [0, 1, 3]
        rhs_clean = np.nan_to_num(rhs)
        for slot, r in enumerate(a2o):
            J = csr_matrix((J_all[slot], inner, outer), shape=(dim_J, dim_J))
            expected = spsolve(J.T.tocsr(), rhs_clean[r])
            np.testing.assert_allclose(lam[r], expected, atol=1e-8, rtol=1e-8)
        assert np.all(lam[2] == 0.0)                     # dropped row


# ---------------------------------------------------------------------------
# Gradients vs finite differences
# ---------------------------------------------------------------------------

@fp64_only
class TestGradients:
    def test_gradcheck_ieee14_mixed_topology(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        # gradcheck interleaves many forwards before it back-propagates the
        # first output: that needs the per-forward Jacobian snapshots.
        pf = _pf(grid, nb_iter=15, snapshot_jacobian=True)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
        line_status, trafo_status = _all_connected(pf, n)
        line_status[1, 3] = False
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[:, 1] = 1.04
        gen_v[2, 2] = 1.0
        inputs = tuple(x.clone().requires_grad_(True) for x in (load_p, load_q, gen_p, gen_v))

        def f_real(lp, lq, gp, gv):
            return pf(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv,
                      line_status=line_status, trafo_status=trafo_status).real.sum()

        def f_abs(lp, lq, gp, gv):
            return pf(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv,
                      line_status=line_status, trafo_status=trafo_status).abs().sum()

        assert torch.autograd.gradcheck(f_real, inputs, eps=1e-5, atol=1e-3, rtol=1e-2,
                                        nondet_tol=1e-7)
        assert torch.autograd.gradcheck(f_abs, inputs, eps=1e-5, atol=1e-3, rtol=1e-2,
                                        nondet_tol=1e-7)

    @needs_bridge
    def test_gradcheck_with_islanded_row(self):
        grid, _, spur_line, _ = _solved_spur_grid(distributed_slack=False)
        pf = _pf(grid, nb_iter=15, snapshot_jacobian=True)
        n = 3
        load_p, _, gen_p = _base_inputs(pf, n, [1.0, 1.05, 0.95])
        line_status, _ = _all_connected(pf, n)
        line_status[1, int(spur_line)] = False
        lp = load_p.clone().requires_grad_(True)
        gp = gen_p.clone().requires_grad_(True)

        def f(lp, gp):
            V = pf(load_p=lp, gen_p=gp, line_status=line_status)
            valid = torch.isfinite(V.real)
            return torch.where(valid, V, torch.zeros_like(V)).real.sum()

        assert torch.autograd.gradcheck(f, (lp, gp), eps=1e-5, atol=1e-3, rtol=1e-2,
                                        nondet_tol=1e-7)
        # the islanded row gets exactly zero gradient
        f(lp, gp).backward()
        assert torch.all(lp.grad[1] == 0) and torch.all(gp.grad[1] == 0)
        assert torch.any(lp.grad[0] != 0) and torch.any(lp.grad[2] != 0)

    def _fd_check(self, *args, **kwargs):
        return _fd_check(*args, **kwargs)


    def test_central_differences_ieee14_distributed_slack(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, nb_iter=15)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
        line_status, _ = _all_connected(pf, n)
        line_status[2, 3] = False
        gen_v = torch.tensor([[1.06, 1.045, 1.01, 1.07, 1.09]] * n, dtype=RDT, device="cuda")
        gen_v[1, 3] = float("nan")
        coords = [("load_p", (0, 2)), ("load_p", (2, 5)), ("load_q", (1, 4)),
                  ("gen_p", (0, 1)), ("gen_p", (2, 3)),
                  ("gen_v", (0, 0)),      # slack generator: Vm-fixed too
                  ("gen_v", (1, 1)), ("gen_v", (2, 2)), ("gen_v", (2, 4)),
                  ("gen_v", (1, 3))]      # NaN entry -> gradient must be 0
        grads = self._fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords)
        assert grads["gen_v"].grad[1, 3] == 0.0
        assert torch.all(grads["gen_v"].grad[0] != 0)

    @needs_bridge
    def test_central_differences_hvdc_droop_and_remote_vc(self):
        from test_augmented_features import _solved_hvdc_droop_grid
        from test_diff_augmented import _solved_remote_gen_grid
        for grid in (_solved_hvdc_droop_grid(), _solved_remote_gen_grid()):
            pf = _pf(grid, nb_iter=20)
            n = 2
            load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.05])
            line_status, _ = _all_connected(pf, n)
            line_status[1, 2] = False
            gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
            gen_v[:, 1] = 1.045
            coords = [("load_p", (0, 3)), ("load_q", (1, 1)), ("gen_p", (1, 2)),
                      ("gen_v", (0, 1)), ("gen_v", (1, 1))]
            self._fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords)

    def test_gen_v_disconnected_and_colocated(self, solver_atol):
        """A disconnected generator's gen_v is ignored (zero gradient); two
        generators on the same bus report the same (bus) gradient."""
        pp = pytest.importorskip("pandapower")
        import pandapower.networks as pn
        from lightsim2grid.network import init_from_pandapower
        from lightsim2grid.lightsim2grid_cpp import AlgorithmType
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net = pn.case14()
            pp.create_gen(net, bus=int(net.gen.bus.iloc[1]), p_mw=5.0,
                          vm_pu=float(net.gen.vm_pu.iloc[1]))      # gen 4: co-located with gen 1
            pp.runpp(net)
            grid = init_from_pandapower(net)      # gens: pp gens 0..4, then the ext_grid
            grid.deactivate_gen(3)                # disconnected gen (bus 7 turns PQ)
            grid.tell_solver_need_reset()
            grid.change_algorithm(AlgorithmType.NR_KLU)
            n_model = grid.get_bus_vn_kv().shape[0]
            v0 = grid.dc_pf(np.ones(n_model, dtype=complex), 1, 1e-6)
            grid.ac_pf(v0.copy(), 30, 1e-10)
        pf = _pf(grid, nb_iter=15)
        assert pf.n_gen == 6 and int(pf._gen_bus_all[3]) == -1
        assert int(pf._gen_bus_all[4]) == int(pf._gen_bus_all[1])
        n = 2
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1])
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[:, 1] = 1.05
        gen_v[:, 4] = 1.05                 # the co-located generator: same value
        gen_v[:, 3] = 1.2                  # disconnected: must be ignored
        V_ignore = pf(load_p=load_p, load_q=load_q, gen_p=gen_p,
                      gen_v=torch.where(torch.arange(pf.n_gen, device="cuda") == 3,
                                        torch.full_like(gen_v, float("nan")), gen_v))
        gv = gen_v.clone().requires_grad_(True)
        V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_v=gv)
        torch.testing.assert_close(V, V_ignore, atol=solver_atol, rtol=0)
        (V.abs() ** 2).sum().backward()
        assert torch.all(gv.grad[:, 3] == 0)
        torch.testing.assert_close(gv.grad[:, 1], gv.grad[:, 4])
        assert torch.all(gv.grad[:, 1] != 0)

    @needs_bridge
    def test_matches_single_system_adjoint_on_untripped_row(self, ieee14_base_case):
        """Row 0 (no trip): the batched Sbus gradient equals the single-system
        PowerFlowFunction one on the same augmented system."""
        from gpusim2grid.differentiable import solve_power_flow
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, nb_iter=15)
        n = 2
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1])
        line_status, _ = _all_connected(pf, n)
        line_status[1, 3] = False
        lp = load_p.clone().requires_grad_(True)
        lq = load_q.clone().requires_grad_(True)
        V = pf(load_p=lp, load_q=lq, gen_p=gen_p, line_status=line_status)
        V[0].real.sum().backward()

        # Single system at the same Sbus: gradient w.r.t. Sbus, mapped to loads.
        Sbus0 = np.asarray(grid.get_Sbus_solver())
        Sr = torch.tensor(Sbus0.real, dtype=torch.float64, device="cuda", requires_grad=True)
        Si = torch.tensor(Sbus0.imag, dtype=torch.float64, device="cuda", requires_grad=True)
        V1 = solve_power_flow(Sr, Si, max_iter=30, tol=1e-10, grid=grid)
        V1.real.sum().backward()
        el = pf.sweep._elements
        load_bus = el.load_bus[el.load_sel]
        expected_lp = -Sr.grad[load_bus] / el.sn_mva
        expected_lq = -Si.grad[load_bus] / el.sn_mva
        torch.testing.assert_close(lp.grad[0, el.load_sel], expected_lp, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(lq.grad[0, el.load_sel], expected_lq, atol=1e-7, rtol=1e-5)
        assert torch.all(lp.grad[1] == 0)          # row 1 got no cotangent


# ---------------------------------------------------------------------------
# Call-to-call state: a call without a contingency input after one with it
# must reset the session (same inputs -> same answer, whatever ran before).
# ---------------------------------------------------------------------------

def _two_gen_case14():
    """case14 with a 2nd generator on the bus of pp gen 2 (bus 5): the bus stays
    PV while one of the two is off and turns PQ when both are. Returns
    (solved grid, [gen ids on bus 5])."""
    pp = pytest.importorskip("pandapower")
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
    return grid, shared


class TestCallToCallState:

    def _loss_grad(self, pf, load_p, load_q, gen_p, **kw):
        lp = load_p.clone().requires_grad_(True)
        lq = load_q.clone().requires_grad_(True)
        gp = gen_p.clone().requires_grad_(True)
        V = pf(load_p=lp, load_q=lq, gen_p=gp, **kw)
        valid = torch.isfinite(V.real)
        (torch.where(valid, V, torch.zeros_like(V)).abs() ** 2).sum().backward()
        return V.detach(), lp.grad, lq.grad, gp.grad

    def test_topology_is_reset_by_a_call_without_status(self, ieee14_base_case, solver_atol):
        grid = ieee14_base_case["grid"]
        pf, ref = _pf(grid), _pf(grid)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.05, 0.95])

        # 1. never any topology
        V0 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        assert (pf.sweep.driver_build_counter, pf.sweep.source_build_counter) == (1, 1)

        # 2. trips on two rows (warm)
        ls, ts = _all_connected(pf, n)
        ls[1, 3] = False
        ls[2, 0] = False
        V1 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, line_status=ls, trafo_status=ts).clone()
        assert (pf.sweep.driver_build_counter, pf.sweep.source_build_counter) == (1, 2)
        torch.testing.assert_close(V1[0], V0[0], atol=solver_atol, rtol=0)
        assert not torch.allclose(V1[1], V0[1], atol=1e-4)
        assert not torch.allclose(V1[2], V0[2], atol=1e-4)

        # 3. no status inputs again: the trips must be gone (warm reset, no analysis)
        V2 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        assert (pf.sweep.driver_build_counter, pf.sweep.source_build_counter) == (1, 3)
        assert pf.timings.t_analysis_ms == 0.0
        torch.testing.assert_close(V2, V0, atol=solver_atol, rtol=0)

        # 4. and stays gone on the hot path
        V3 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        assert pf.sweep.source_build_counter == 3
        torch.testing.assert_close(V3, V0, atol=solver_atol, rtol=0)

        # 5. all-True masks mean the same as no masks
        ls, ts = _all_connected(pf, n)
        V4 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, line_status=ls, trafo_status=ts).clone()
        torch.testing.assert_close(V4, V0, atol=solver_atol, rtol=0)

        # 6. gradients after the reset match a session that never saw a trip
        out = self._loss_grad(pf, load_p, load_q, gen_p)
        exp = self._loss_grad(ref, load_p, load_q, gen_p)
        for a, b in zip(out, exp):
            torch.testing.assert_close(a, b, atol=10 * solver_atol, rtol=1e-6)

        # 7. a new row count without status after trips (the session would
        #    refuse a stale trip list) -> fresh answer
        lp5, lq5, gp5 = _base_inputs(pf, 5, [1.0, 1.02, 0.98, 1.05, 0.95])
        ls5, ts5 = _all_connected(pf, 5)
        ls5[4, 6] = False
        pf(load_p=lp5, load_q=lq5, gen_p=gp5, line_status=ls5, trafo_status=ts5)
        V6 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        torch.testing.assert_close(V6, V0, atol=solver_atol, rtol=0)

    @needs_bridge
    def test_gen_status_is_reset_by_a_call_without_it(self, ieee14_base_case, solver_atol):
        grid = ieee14_base_case["grid"]
        pf, ref = _pf(grid), _pf(grid)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.05, 0.95])
        V0 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        assert pf.reserved_switchable_buses.size == 0

        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[1, 1] = False
        gs[2, 2] = False
        V1 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_status=gs).clone()
        assert pf.reserved_switchable_buses.size == 2      # the two buses that can flip
        assert pf.sweep.driver_build_counter == 2           # structure changed: cold
        torch.testing.assert_close(V1[0], V0[0], atol=solver_atol, rtol=0)
        assert not torch.allclose(V1[1], V0[1], atol=1e-4)

        # no gen_status: the mask is cleared and the reserved structure released
        V2 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        assert pf.reserved_switchable_buses.size == 0
        assert pf.sweep.driver_build_counter == 3
        assert pf.sweep.solver.dim_J == ref.sweep.solver.dim_J
        torch.testing.assert_close(V2, V0, atol=solver_atol, rtol=0)
        V3 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()   # hot
        assert pf.sweep.driver_build_counter == 3
        torch.testing.assert_close(V3, V0, atol=solver_atol, rtol=0)

        # all-True == no mask (no rebuild either)
        V4 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p,
                gen_status=torch.ones_like(gs)).clone()
        assert pf.sweep.driver_build_counter == 3
        torch.testing.assert_close(V4, V0, atol=solver_atol, rtol=0)

        out = self._loss_grad(pf, load_p, load_q, gen_p)
        exp = self._loss_grad(ref, load_p, load_q, gen_p)
        for a, b in zip(out, exp):
            torch.testing.assert_close(a, b, atol=10 * solver_atol, rtol=1e-6)

        # new row count, no gen_status, after a masked call
        lp5, lq5, gp5 = _base_inputs(pf, 5)
        gs5 = torch.ones(5, pf.n_gen, dtype=torch.bool, device="cuda")
        gs5[3, 1] = False
        pf(load_p=lp5, load_q=lq5, gen_p=gp5, gen_status=gs5)
        V6 = pf(load_p=load_p, load_q=load_q, gen_p=gen_p).clone()
        torch.testing.assert_close(V6, V0, atol=solver_atol, rtol=0)


# ---------------------------------------------------------------------------
# Generator contingencies (gen_status)
# ---------------------------------------------------------------------------

@needs_bridge
class TestGenStatus:

    def test_forward_matches_scenario_sweep(self, ieee14_base_case, solver_atol):
        from gpusim2grid import ScenarioSweepGPU
        grid = ieee14_base_case["grid"]
        pf = _pf(grid)
        n = 4
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.05, 0.95, 1.02])
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[1, 1] = False
        gs[2, 2] = False
        gs[3, [1, 3]] = False
        ls, ts = _all_connected(pf, n)
        ls[2, 3] = False
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[:, 1] = 1.03            # off on rows 1 and 3: must be ignored there
        gen_v[3, 2] = 1.02
        V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_v=gen_v,
               line_status=ls, trafo_status=ts, gen_status=gs)
        assert torch.isfinite(V).all()
        assert pf.reserved_switchable_buses.size == 3

        sw = ScenarioSweepGPU(grid, nb_iter=NB_ITER, tol_base=TOL)
        sw.set_injections_from_elements(load_p.cpu().numpy(), load_q.cpu().numpy(),
                                        gen_p.cpu().numpy())
        sw.set_topology([[], [], [3], []])
        sw.set_contingency_gens(~gs.cpu().numpy())
        gv = gen_v.cpu().numpy().copy()
        gv[~gs.cpu().numpy()] = np.nan
        sw.set_gen_v(gv)
        sw.compute(batch_size=n)
        V_ref = sw.solver.V_results.to_numpy().reshape(n, pf.n_bus)
        np.testing.assert_allclose(V.cpu().numpy(), V_ref, atol=solver_atol)

        # the disconnected generator's P is really gone: same as gen_p = 0 there
        gp0 = gen_p.clone()
        gp0[~gs] = 0.0
        V_gp0 = pf(load_p=load_p, load_q=load_q, gen_p=gp0, gen_v=gen_v,
                   line_status=ls, trafo_status=ts, gen_status=gs)
        torch.testing.assert_close(V_gp0, V, atol=solver_atol, rtol=0)

    def test_forward_matches_one_off_powerflow(self, solver_atol):
        """Row with both generators of a bus off (bus -> PQ) vs a fresh grid
        with them deactivated, and one generator off (bus stays PV)."""
        grid, (g_a, g_b) = _two_gen_case14()
        pf = _pf(grid, nb_iter=15)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n)
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[0, g_a] = False
        gs[1, [g_a, g_b]] = False
        V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_status=gs)
        assert pf.reserved_switchable_buses.tolist() == [5]
        assert pf.sweep.solver.get_row_pv_to_pq() == [[], [5], []]
        buses = np.asarray(grid.id_ac_solver_to_me(), dtype=int)
        for row, off in enumerate([[g_a], [g_a, g_b], []]):
            g2, _ = _two_gen_case14()
            for g in off:
                g2.deactivate_gen(int(g))
            g2.tell_solver_need_reset()
            n_bus = g2.get_bus_vn_kv().shape[0]
            v0 = g2.dc_pf(np.ones(n_bus, dtype=complex), 1, 1e-6)
            ref = g2.ac_pf(v0.copy(), 30, 1e-10)
            assert ref.shape[0] > 0
            np.testing.assert_allclose(V[row].cpu().numpy(), ref[buses], atol=10 * solver_atol)

    @fp64_only
    def test_central_differences_released_bus(self, ieee14_base_case):
        """Rows releasing a bus (its only generator off): the Q equation of that
        bus is live (load_q there has a gradient), the off generator's gen_p /
        gen_v have none, everything else matches finite differences."""
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, nb_iter=15)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
        line_status, _ = _all_connected(pf, n)
        line_status[2, 3] = False
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[0, 1] = False
        gs[2, [2, 3]] = False
        gen_v = torch.tensor([[1.06, 1.045, 1.01, 1.07, 1.09]] * n, dtype=RDT, device="cuda")
        bus_g1 = int(pf._gen_bus_all[1])
        loads_at_g1 = torch.nonzero(pf._load_bus_sel == bus_g1).flatten()
        assert loads_at_g1.numel() == 1            # case14: one load on that bus
        l1 = int(pf._load_sel[loads_at_g1[0]])
        coords = [("load_p", (0, 2)), ("load_q", (0, l1)), ("load_q", (1, l1)),
                  ("load_p", (2, 5)), ("load_q", (2, 4)),
                  ("gen_p", (0, 2)), ("gen_p", (1, 1)), ("gen_p", (2, 1)),
                  ("gen_p", (0, 1)), ("gen_v", (0, 1)),      # off: 0
                  ("gen_v", (0, 0)), ("gen_v", (0, 2)), ("gen_v", (1, 1)),
                  ("gen_v", (2, 1)), ("gen_v", (2, 4))]
        grads = _fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords,
                          gen_status=gs)
        assert grads["gen_p"].grad[0, 1] == 0.0 and grads["gen_v"].grad[0, 1] == 0.0
        assert torch.all(grads["gen_p"].grad[2, [2, 3]] == 0)
        assert torch.all(grads["gen_v"].grad[2, [2, 3]] == 0)
        assert grads["load_q"].grad[0, l1] != 0.0          # released: Q equation live
        assert grads["gen_v"].grad[1, 1] != 0.0

    @fp64_only
    def test_central_differences_pinned_reserved_bus(self):
        """A reserved bus that is still PV on a row (one of its two generators
        off) identity-pins its Q equation: no Q gradient there, and the
        surviving generator's gen_v gradient is the plain PV one."""
        grid, (g_a, g_b) = _two_gen_case14()
        pf = _pf(grid, nb_iter=15)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.05, 0.95])
        line_status, _ = _all_connected(pf, n)
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[0, g_a] = False                   # bus 5 pinned (g_b keeps it PV)
        gs[1, [g_a, g_b]] = False            # bus 5 released
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[:, [g_a, g_b]] = 1.01
        gen_v[:, 1] = 1.045
        bus5 = int(pf._gen_bus_all[g_a])
        loads_at_5 = torch.nonzero(pf._load_bus_sel == bus5).flatten()
        assert loads_at_5.numel() == 1
        l5 = int(pf._load_sel[loads_at_5[0]])
        coords = [("load_q", (0, l5)), ("load_q", (1, l5)), ("load_q", (2, l5)),
                  ("load_p", (0, l5)), ("load_p", (1, 3)),
                  ("gen_p", (0, g_b)), ("gen_p", (1, 1)),
                  ("gen_v", (0, g_b)),        # the surviving generator of the pinned bus
                  ("gen_v", (0, 1)), ("gen_v", (1, 1)),
                  ("gen_v", (1, g_a)), ("gen_v", (1, g_b))]     # released: 0
        # (no per-column FD on row 2: both generators write the same bus, the
        #  last column wins, so a single column's FD is 0 by construction --
        #  both report the bus gradient, checked below.)
        grads = _fd_check(pf, load_p, load_q, gen_p, gen_v, line_status, coords,
                          gen_status=gs)
        assert grads["load_q"].grad[0, l5] == 0.0            # pinned: frozen Q equation
        assert grads["load_q"].grad[2, l5] == 0.0            # plain PV bus
        assert grads["load_q"].grad[1, l5] != 0.0            # released
        assert torch.all(grads["gen_v"].grad[1, [g_a, g_b]] == 0)
        assert grads["gen_v"].grad[0, g_a] == 0.0
        assert grads["gen_v"].grad[0, g_b] != 0.0
        torch.testing.assert_close(grads["gen_v"].grad[2, g_a], grads["gen_v"].grad[2, g_b])

    @fp64_only
    def test_gradcheck_with_gen_status(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, nb_iter=15, snapshot_jacobian=True)
        n = 3
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1, 0.9])
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[0, 1] = False
        gs[2, 3] = False
        gen_v = torch.full((n, pf.n_gen), float("nan"), dtype=RDT, device="cuda")
        gen_v[:, 2] = 1.01
        gen_v[1, 1] = 1.04
        inputs = tuple(x.clone().requires_grad_(True) for x in (load_p, load_q, gen_p, gen_v))

        def f(lp, lq, gp, gv):
            V = pf(load_p=lp, load_q=lq, gen_p=gp, gen_v=gv, gen_status=gs)
            return torch.view_as_real(V)

        assert torch.autograd.gradcheck(f, inputs, eps=1e-5, atol=1e-3, rtol=1e-2,
                                        nondet_tol=1e-7)

    def test_snapshot_backward_after_mask_change_raises(self, ieee14_base_case):
        grid = ieee14_base_case["grid"]
        pf = _pf(grid, snapshot_jacobian=True)
        n = 2
        load_p, load_q, gen_p = _base_inputs(pf, n, [1.0, 1.1])
        gs = torch.ones(n, pf.n_gen, dtype=torch.bool, device="cuda")
        gs[1, 1] = False
        lp = load_p.clone().requires_grad_(True)
        V1 = pf(load_p=lp, load_q=load_q, gen_p=gen_p, gen_status=gs)
        gs2 = gs.clone()
        gs2[0, 1] = False                    # same reserved set, other rows pinned
        pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_status=gs2)
        with pytest.raises(RuntimeError, match="batch structure"):
            V1.real.sum().backward()
