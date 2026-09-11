# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
BatchPowerFlow — batched, differentiable AC power flow for PyTorch, driven
by the same per-element inputs as lightsim2grid's ``ScenarioSweep``.

    pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=6)
    V  = pf(load_p=..., load_q=..., gen_p=..., gen_v=...,
            line_status=..., trafo_status=...)        # complex (n_scen, n_bus)

Every input is a ``(n_scenarios, n_elements)`` tensor (or ``None`` = the
grid's own base-case value in every row); row ``i`` is one independent
scenario: its own injections, its own voltage set-points and its own set of
disconnected branches (``line_status`` / ``trafo_status`` are boolean masks
with **True = connected**, grid2op's convention; a row trips the branches
whose mask is False). The whole batch is solved in ONE GPU pass by a
:class:`gpusim2grid.ScenarioSweepGPU` session that this object keeps alive,
so consecutive calls with the same number of rows reuse everything: no
cuDSS analysis, no first factorization, no re-upload of the grid, only the
new rows move to the device (see ``ScenarioSweepGPU.driver_build_counter``).

Differentiable inputs: ``load_p``, ``load_q``, ``gen_p`` (through Sbus) and
``gen_v`` (through the seeded voltage magnitude). ``line_status`` /
``trafo_status`` are discrete and carry no gradient.

Adjoint math (batched)
----------------------
Each row ``s`` solves ``G(x_s; Sbus_s, Vm_s) = 0`` -- Newton-Raphson on the
augmented system lightsim2grid poses (see ``_power_flow_op.py``) -- and
returns ``V_s`` (a function of the unknowns ``x_s`` and, at Vm-fixed buses,
of the set-point directly). With the code Jacobian ``J_s = dS_calc/dx``
(positive diagonal, see ``_power_flow_op.py`` for the sign discussion):

    x̄_s   = projection of ḡV_s onto the theta / Vm unknowns
            (theta_col_of_bus / vm_col_of_bus maps, same formulas as the
            single-system op, vectorised over rows)
    λ_s    = J_sᵀ⁻¹ x̄_s                    (batched cuDSS solve of the
                                            explicitly transposed system)
    Sbus̄_s = +λ_s  gathered by p_row_of_bus / q_row_of_bus (real / imag)

then autograd maps Sbus̄ back to ``load_p`` / ``load_q`` / ``gen_p`` through
the (affine, natively differentiable) element→bus assembly done in torch.

For ``gen_v``: a Vm-fixed bus ``k`` keeps ``|V_k| = gen_v`` throughout the
solve, so ``V_k = gen_v · e^{jθ_k*}`` with ``θ_k*`` (and every other unknown)
depending on ``gen_v`` through ``G``. Hence

    ḡ(gen_v)_k = Re( e^{-jθ_k} · ḡV_k )            (direct term, Python side)
               - λ_s · ∂S_calc/∂Vm_k               (indirect term,
                                                     gen_v_adjoint_kernel)

where ``∂S_calc/∂Vm_k`` is the dS/dVm column ``fill_J`` never stores for a
Vm-fixed bus, evaluated on the row's own patched Ybus. The minus sign is the
implicit-function sign: ``dx/dVm_k = -J⁻¹ ∂S_calc/∂Vm_k``. Only generators
whose own bus is Vm-fixed (pv or slack) get a non-zero gradient -- exactly
the ones ``set_gen_v`` acts on; a NaN entry (= "keep the base-case voltage")
gets 0.

Row bookkeeping: the session works in *active-slot* order (rows the
connectivity pre-check drops are compacted out); all of that stays in C++
(``solve_JT_batch_dlpack`` takes and returns ORIGINAL row order). Dropped rows
are NaN in ``V`` and get a zero gradient; callers mask them out of the loss.

Lazy Jᵀ: nothing adjoint-related is built until the first ``backward()``,
which creates the transposed pattern + J→Jᵀ position map, the buffers and a
second cuDSS batch context (one ANALYSIS + one FACTORIZATION). Every later
backward only permutes the values, REFACTORIZES (once per new forward) and
SOLVES.

Jacobian lifetime: by default backward reads the converged Jacobians (and,
for ``gen_v``, the patched Ybus values) straight from the session's chunk
buffers, which the *next* forward overwrites. A ``run_counter`` guard turns a
forward→forward→backward(first) pattern into a clear error; pass
``snapshot_jacobian=True`` to clone those buffers in every forward instead
(one D2D copy of ``capacity x nnz_J`` reals and ``capacity x nnz_Y``
complex, kept until backward).
"""

import numpy as np
import torch
from torch import Tensor

from .. import _gpusim2grid as _cpp
from .._ls2g_utils import extract_branch_data
from ..scenario_sweep.gpu_facade import ScenarioSweepGPU
from ._flows import compute_flows as _compute_flows_torch


__all__ = ["BatchPowerFlow"]


class BatchPowerFlow:
    """Batched differentiable AC power flow (see the module docstring).

    Build it with :meth:`from_lsgrid`; call it like a function (or use
    :meth:`forward`).
    """

    def __init__(self, sweep, *, snapshot_jacobian=False):
        if sweep._elements is None:
            raise ValueError(
                "BatchPowerFlow needs a ScenarioSweepGPU built from a lightsim2grid "
                "grid (explicit-array/tuple mode has no loads/generators to map).")
        self._sweep = sweep
        self._solver = sweep.solver                 # _ScenarioSweepSolver
        self._solver.fixed_batch_capacity = True    # always one chunk (adjoint)
        self.snapshot_jacobian = bool(snapshot_jacobian)

        self._dev = torch.device("cuda", _device_index(sweep))
        self._rdtype = torch.float32 if bool(_cpp.is_fp32) else torch.float64
        self._cdtype = torch.complex64 if bool(_cpp.is_fp32) else torch.complex128

        el = sweep._elements
        dev, rdt = self._dev, self._rdtype
        sn = float(el.sn_mva)
        self.sn_mva = sn
        self.n_bus = int(el.n_bus)
        self.n_load = int(el.n_load)
        self.n_gen = int(el.n_gen)
        grid = sweep._grid
        self.n_line = len(grid.get_lines())
        self.n_trafo = len(grid.get_trafos())
        self.n_branch = self.n_line + self.n_trafo

        # Element -> bus assembly (mirrors _ls2g_utils.build_bus_injections):
        # Sbus_pu = const + Σ gen_p/sn − Σ (load_p + j load_q)/sn.
        self._const_re = torch.as_tensor(np.ascontiguousarray(el.const_mw.real) / sn, dtype=rdt, device=dev)
        self._const_im = torch.as_tensor(np.ascontiguousarray(el.const_mw.imag) / sn, dtype=rdt, device=dev)
        self._gen_sel = torch.as_tensor(np.asarray(el.gen_sel, dtype=np.int64), device=dev)
        self._gen_bus_sel = torch.as_tensor(np.asarray(el.gen_bus[el.gen_sel], dtype=np.int64), device=dev)
        self._load_sel = torch.as_tensor(np.asarray(el.load_sel, dtype=np.int64), device=dev)
        self._load_bus_sel = torch.as_tensor(np.asarray(el.load_bus[el.load_sel], dtype=np.int64), device=dev)
        self._gen_bus_all = torch.as_tensor(np.asarray(el.gen_bus, dtype=np.int64), device=dev)
        self._gen_bus_np = np.ascontiguousarray(el.gen_bus, dtype=np.int32)
        self._is_vm_fixed = torch.as_tensor(self._solver.is_vm_fixed_bus, device=dev)

        # Base-case per-element values: what a ``None`` input means.
        self._load_p_base = torch.tensor(np.array(el.load_p_base), dtype=rdt, device=dev)
        self._load_q_base = torch.tensor(np.array(el.load_q_base), dtype=rdt, device=dev)
        self._gen_p_base = torch.tensor(np.array(el.gen_p_base), dtype=rdt, device=dev)

        # Branch data for compute_flows() (lines-then-trafos, solver numbering).
        (b_from, b_to, yff, yft, ytf, ytt, vn_kv, _sn), _, _ = extract_branch_data(grid)
        self._branch_from = torch.as_tensor(np.asarray(b_from, dtype=np.int64), device=dev)
        self._branch_to = torch.as_tensor(np.asarray(b_to, dtype=np.int64), device=dev)
        self._yff = torch.as_tensor(np.asarray(yff), dtype=self._cdtype, device=dev)
        self._yft = torch.as_tensor(np.asarray(yft), dtype=self._cdtype, device=dev)
        self._ytf = torch.as_tensor(np.asarray(ytf), dtype=self._cdtype, device=dev)
        self._ytt = torch.as_tensor(np.asarray(ytt), dtype=self._cdtype, device=dev)
        self._bus_vn_kv = torch.as_tensor(np.asarray(vn_kv), dtype=rdt, device=dev)

        # Call-to-call state.
        self._last_n_scen = None
        self._topology_mask = None      # (n_scen, n_branch) bool, True = tripped
        self._topology_in_session = False
        self._pending_topology = None   # ragged list to hand to the session on the next run
        self._gen_v_in_session = False

    # ------------------------------------------------------------------ build
    @classmethod
    def from_lsgrid(cls, grid, *, nb_iter=4, handle_disconnected_grid=False,
                    strategy="direct_refactor_every", snapshot_jacobian=False,
                    device=None, reordering_alg=None, matching_alg=None,
                    pivot_epsilon_alg=None, use_distributed_slack=True,
                    scaling_max_voltage_change=None, max_dVa=None, max_dVm=None,
                    init_from_n_powerflow=True, max_iter_base=10, tol_base=1e-8,
                    precision=None):
        """Build from a *solved* lightsim2grid grid (``grid.ac_pf`` done).

        The keyword arguments are :class:`gpusim2grid.ScenarioSweepGPU`'s
        (same meaning), plus ``strategy`` (linear-solve strategy string) and
        ``snapshot_jacobian`` (see the module docstring). ``nb_iter`` is the
        fixed Newton-Raphson iteration count per row: raise it (and lower
        ``tol_base``) when gradients must be accurate -- the adjoint is exact
        only at a converged solution.
        """
        sweep = ScenarioSweepGPU(
            grid, init_from_n_powerflow=init_from_n_powerflow, precision=precision,
            nb_iter=nb_iter, max_iter_base=max_iter_base, tol_base=tol_base,
            device=device, handle_disconnected_grid=handle_disconnected_grid,
            reordering_alg=reordering_alg, matching_alg=matching_alg,
            pivot_epsilon_alg=pivot_epsilon_alg,
            scaling_max_voltage_change=scaling_max_voltage_change,
            max_dVa=max_dVa, max_dVm=max_dVm,
            use_distributed_slack=use_distributed_slack)
        sweep.strategy = strategy
        return cls(sweep, snapshot_jacobian=snapshot_jacobian)

    # --------------------------------------------------------------- forward
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, load_p=None, load_q=None, gen_p=None, gen_v=None,
                line_status=None, trafo_status=None):
        """Solve one scenario per row; returns complex ``V`` ``(n_scen, n_bus)``
        on the GPU (per-unit, AC-solver bus numbering; NaN rows = scenarios the
        connectivity pre-check dropped).

        load_p, load_q : (n_scen, n_load) MW / MVAr        (differentiable)
        gen_p          : (n_scen, n_gen)  MW               (differentiable)
        gen_v          : (n_scen, n_gen)  vm_pu            (differentiable; NaN
                         = keep the base-case voltage for that (row, gen))
        line_status    : (n_scen, n_line)  bool, True = connected
        trafo_status   : (n_scen, n_trafo) bool, True = connected
        ``None`` = the grid's base-case value in every row. At least one
        input must be given (it fixes n_scen).
        """
        n_scen = self._infer_n_scen(load_p, load_q, gen_p, gen_v, line_status, trafo_status)
        load_p = self._as_input(load_p, self.n_load, self._load_p_base, n_scen, "load_p")
        load_q = self._as_input(load_q, self.n_load, self._load_q_base, n_scen, "load_q")
        gen_p = self._as_input(gen_p, self.n_gen, self._gen_p_base, n_scen, "gen_p")
        gen_v = None if gen_v is None else self._as_input(gen_v, self.n_gen, None, n_scen, "gen_v")

        self._apply_topology(line_status, trafo_status, n_scen)

        if n_scen != self._last_n_scen:
            # Capacity == n_scen: one chunk, whatever rows get islanded.
            self._solver.batch_size = n_scen
            self._last_n_scen = n_scen

        # Element -> bus, in plain (differentiable) torch.
        inv_sn = 1.0 / self.sn_mva
        P = self._const_re.unsqueeze(0).expand(n_scen, -1).clone()
        Q = self._const_im.unsqueeze(0).expand(n_scen, -1).clone()
        if self._gen_sel.numel():
            P = P.index_add(1, self._gen_bus_sel, gen_p[:, self._gen_sel] * inv_sn)
        if self._load_sel.numel():
            P = P.index_add(1, self._load_bus_sel, -load_p[:, self._load_sel] * inv_sn)
            Q = Q.index_add(1, self._load_bus_sel, -load_q[:, self._load_sel] * inv_sn)

        return _BatchPowerFlowOp.apply(P, Q, gen_v, self)

    # --------------------------------------------------------------- results
    def get_disconnected(self):
        """(n_scen,) int: 1 where the last call dropped the row (NaN)."""
        return self._sweep.get_disconnected()

    def last_residuals(self):
        """(n_scen,) float: ``‖F‖∞`` of each row after the last call."""
        return self._solver.residuals.to_numpy()

    def converged(self, tol=1e-6):
        """(n_scen,) bool: residual <= tol after the last call."""
        return self.last_residuals() <= tol

    @property
    def timings(self):
        """:class:`BatchTimings` of the last call (+ cumulative adjoint counters)."""
        return self._solver.timings

    @property
    def sweep(self):
        """The underlying :class:`gpusim2grid.ScenarioSweepGPU` (escape hatch)."""
        return self._sweep

    @property
    def device(self):
        return self._dev

    def compute_flows(self, V):
        """Branch flows (``p_or_mw``, ``q_or_mvar``, ``p_ex_mw``, ``q_ex_mvar``,
        ``i_or_a``, ``i_ex_a``) of shape ``(n_scen, n_branch)`` from ``V``
        ``(n_scen, n_bus)``; pure torch, differentiable. Branches a row tripped
        are NOT zeroed here (mask them with the status inputs if needed)."""
        return _compute_flows_torch(V, self._yff, self._yft, self._ytf, self._ytt,
                                    self._branch_from, self._branch_to,
                                    self._bus_vn_kv, self.sn_mva)

    # --------------------------------------------------------------- helpers
    def _infer_n_scen(self, *inputs):
        n = None
        for x in inputs:
            if x is None:
                continue
            shape = tuple(x.shape) if hasattr(x, "shape") else np.shape(x)
            if len(shape) != 2:
                raise ValueError(
                    f"every input must be 2-D (n_scenarios, n_elements); got shape {shape}")
            if n is None:
                n = int(shape[0])
            elif int(shape[0]) != n:
                raise ValueError(
                    f"all inputs must share the same number of rows; got {n} and {shape[0]}")
        if n is None:
            raise ValueError(
                "BatchPowerFlow needs at least one input (load_p, load_q, gen_p, gen_v, "
                "line_status or trafo_status) to know the number of scenarios")
        if n <= 0:
            raise ValueError("the number of scenarios must be > 0")
        return n

    def _as_input(self, x, n_cols, base, n_scen, name):
        if x is None:
            return base.unsqueeze(0).expand(n_scen, n_cols)
        if isinstance(x, Tensor):
            t = x.to(device=self._dev, dtype=self._rdtype)
        else:
            t = torch.as_tensor(np.asarray(x), dtype=self._rdtype, device=self._dev)
        if t.shape != (n_scen, n_cols):
            raise ValueError(
                f"'{name}' must have shape ({n_scen}, {n_cols}), got {tuple(t.shape)}")
        return t

    def _as_status(self, x, n_cols, n_scen, name):
        if x is None:
            return torch.ones(n_scen, n_cols, dtype=torch.bool, device=self._dev)
        if isinstance(x, Tensor):
            t = x.to(device=self._dev, dtype=torch.bool)
        else:
            t = torch.as_tensor(np.asarray(x, dtype=bool), device=self._dev)
        if t.shape != (n_scen, n_cols):
            raise ValueError(
                f"'{name}' must have shape ({n_scen}, {n_cols}), got {tuple(t.shape)}")
        return t

    def _apply_topology(self, line_status, trafo_status, n_scen):
        """Decide what the session's topology must be for this call; the
        ragged list (if any) is handed over by the op AFTER the injections
        (the session checks the row counts against them)."""
        self._pending_topology = None
        if line_status is None and trafo_status is None:
            # No trips wanted. Only touch the session if it still holds trips
            # or its row count is stale.
            if self._topology_mask is not None or (
                    self._topology_in_session and n_scen != self._last_n_scen):
                self._pending_topology = [[] for _ in range(n_scen)]
            self._topology_mask = None
            return

        tripped = torch.cat([
            ~self._as_status(line_status, self.n_line, n_scen, "line_status"),
            ~self._as_status(trafo_status, self.n_trafo, n_scen, "trafo_status")], dim=1)
        prev = self._topology_mask
        if prev is not None and prev.shape == tripped.shape and torch.equal(prev, tripped):
            return   # unchanged: the session keeps its topology (hot path)

        # Dense mask -> ragged list of tripped branch ids (lines-then-trafos).
        nz = tripped.nonzero()
        if nz.numel():
            rows = nz[:, 0].cpu().numpy()
            cols = nz[:, 1].cpu().numpy()
            counts = np.bincount(rows, minlength=n_scen)
            splits = np.split(cols, np.cumsum(counts)[:-1])
            ragged = [c.tolist() for c in splits]
        else:
            ragged = [[] for _ in range(n_scen)]
        self._pending_topology = ragged
        self._topology_mask = tripped.clone()


def _device_index(sweep):
    """CUDA ordinal the session lives on (the facade normalises ``device``)."""
    try:
        return int(torch.from_dlpack(sweep.solver.v_base_dlpack()).device.index)
    except Exception:   # pragma: no cover - defensive
        return 0


class _BatchPowerFlowOp(torch.autograd.Function):
    """(P, Q) per-unit bus injections (+ optional gen_v) -> V, with the
    batched adjoint backward. Not public: :class:`BatchPowerFlow` builds
    (P, Q) from the element inputs so autograd gets their gradients for free."""

    @staticmethod
    def forward(ctx, P: Tensor, Q: Tensor, gen_v, pf: BatchPowerFlow) -> Tensor:
        solver = pf._solver
        dev = pf._dev
        stream = torch.cuda.current_stream(dev).cuda_stream

        S = torch.complex(P, Q).contiguous()
        solver.set_injections_dlpack(S.__dlpack__(), stream)
        if pf._pending_topology is not None:
            pf._sweep.set_topology(pf._pending_topology)
            pf._topology_in_session = True
            pf._pending_topology = None
        if gen_v is not None:
            solver.set_gen_v_dlpack(gen_v.detach().contiguous().__dlpack__(), pf._gen_bus_np, stream)
            pf._gen_v_in_session = True
        elif pf._gen_v_in_session:
            solver.clear_gen_v()
            pf._gen_v_in_session = False

        want_gen_v = gen_v is not None and ctx.needs_input_grad[2]
        needs_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or want_gen_v
        solver.keep_final_jacobian = bool(needs_grad)
        solver.run()

        V = torch.from_dlpack(solver.v_results_dlpack()).clone()

        ctx.pf = pf
        ctx.run_id = solver.run_counter
        ctx.want_gen_v = bool(want_gen_v)
        ctx.J = ctx.Y = None
        if needs_grad and pf.snapshot_jacobian:
            ctx.J = torch.from_dlpack(solver.j_values_dlpack()).clone()
            if want_gen_v:
                ctx.Y = torch.from_dlpack(solver.ybus_values_dlpack()).clone()
        ctx.save_for_backward(V, gen_v)
        return V

    @staticmethod
    def backward(ctx, grad_V: Tensor):
        pf = ctx.pf
        solver = pf._solver
        V, gen_v = ctx.saved_tensors
        dev = V.device
        rdtype = pf._rdtype
        n_scen, n_bus = V.shape

        if ctx.J is None and solver.run_counter != ctx.run_id:
            raise RuntimeError(
                "BatchPowerFlow.backward: another forward() ran after the one this "
                "gradient belongs to, overwriting its Jacobians on the GPU. Call "
                "backward() before the next forward(), or build the model with "
                "snapshot_jacobian=True to keep a copy per forward.")

        theta_col = torch.as_tensor(solver.theta_col_of_bus, device=dev)
        vm_col = torch.as_tensor(solver.vm_col_of_bus, device=dev)
        p_row = torch.as_tensor(solver.p_row_of_bus, device=dev)
        q_row = torch.as_tensor(solver.q_row_of_bus, device=dev)
        dim_J = int(solver.dim_J)

        # Cotangent projection (same formulas as _power_flow_op.py, per row).
        # NaN buses (islanded rows / masked components) carry no cotangent.
        valid = torch.isfinite(V.real) & torch.isfinite(V.imag)
        gV = torch.where(valid, grad_V, torch.zeros_like(grad_V))
        Vs = torch.where(valid, V, torch.ones_like(V))
        Vn = Vs / Vs.abs()
        proj_th = (Vs.conj() * gV).imag.to(rdtype)
        proj_vm = (Vn.conj() * gV).real.to(rdtype)

        xbar = torch.zeros(n_scen, dim_J, dtype=rdtype, device=dev)
        th_mask = theta_col >= 0
        vm_mask = vm_col >= 0
        xbar[:, theta_col[th_mask]] = proj_th[:, th_mask]
        xbar[:, vm_col[vm_mask]] = proj_vm[:, vm_mask]
        xbar = xbar.contiguous()

        stream = torch.cuda.current_stream(dev).cuda_stream
        j_cap = y_cap = v_cap = None
        if ctx.J is not None:
            j_cap = ctx.J.__dlpack__()
            v_cap = V.detach().contiguous().__dlpack__()
            if ctx.Y is not None:
                y_cap = ctx.Y.__dlpack__()
        lam_cap, gvm_cap = solver.solve_JT_batch_dlpack(
            xbar.__dlpack__(), j_cap, y_cap, v_cap, ctx.want_gen_v, stream)
        lam = torch.from_dlpack(lam_cap).clone()

        # Sbus gradient: +λ (see _power_flow_op.py on the sign).
        grad_P = torch.zeros(n_scen, n_bus, dtype=rdtype, device=dev)
        grad_Q = torch.zeros(n_scen, n_bus, dtype=rdtype, device=dev)
        p_mask = p_row >= 0
        q_mask = q_row >= 0
        grad_P[:, p_mask] = lam[:, p_row[p_mask]]
        grad_Q[:, q_mask] = lam[:, q_row[q_mask]]

        grad_gen_v = None
        if ctx.want_gen_v:
            gvm = torch.from_dlpack(gvm_cap).clone()          # indirect term (sign included)
            g_bus = proj_vm + gvm                              # + direct term
            gen_bus = pf._gen_bus_all
            safe_bus = gen_bus.clamp(min=0)
            ok = (gen_bus >= 0) & pf._is_vm_fixed[safe_bus]
            g = g_bus[:, safe_bus]
            g = torch.where(ok.unsqueeze(0), g, torch.zeros_like(g))
            g = torch.where(torch.isnan(gen_v), torch.zeros_like(g), g)
            grad_gen_v = g

        return grad_P, grad_Q, grad_gen_v, None
