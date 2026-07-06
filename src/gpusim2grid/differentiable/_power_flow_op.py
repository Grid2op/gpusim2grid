# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
PowerFlowFunction — torch.autograd.Function wrapping the GPU NR AC power flow.

Forward:  Runs AcPfNrSession, returns complex V [n_bus] on GPU. When ``grid``
          is given (a solved lightsim2grid grid), the session solves the SAME
          augmented system lightsim2grid does (distributed slack / HVDC
          angle-droop / SVC / remote generator voltage control), via the
          NRLedger read off the grid. Otherwise it solves the bare
          ``[pvpq | pq]`` system from raw Ybus/Vinit/pv/pq/slack arrays.
Backward: Solves Jᵀλ = x̄ via the adjoint method (implicit function theorem),
          scatters λ back to grad_Sbus.

Adjoint math
------------
NR solves F(x, Sbus) = 0 at convergence, where x is the NR unknown vector
(dimension ``dim_J``): angle (theta) unknowns and voltage-magnitude (Vm)
unknowns, one of each per bus that owns one (bare system: theta for pvpq
buses, Vm for pq buses; augmented system: whatever the NRLedger assigns —
extra buses can gain a theta/Vm unknown, e.g. a non-reference distributed-slack
bus, and extra columns can exist that own no bus, e.g. ``slack_absorbed`` or a
VoltageControl reactive unknown).

Given dL/dV (complex grad_V), project to the real x-space using the
bus-indexed NRLedger column maps (``theta_col_of_bus`` / ``vm_col_of_bus``,
length n_bus, -1 where the bus owns no such column):
    x̄[theta_col_of_bus[i]] = Im(conj(V[i]) * grad_V[i])   for every bus i with
                              theta_col_of_bus[i] >= 0
    x̄[vm_col_of_bus[i]]    = Re(conj(V[i]/|V[i]|) * grad_V[i])   for every bus
                              i with vm_col_of_bus[i] >= 0

This reduces exactly to the bare-system pvpq/pq positional formulas when no
augmented ledger is active (the trivial ledger sets theta_col_of_bus[bus] =
position in sorted(pvpq), vm_col_of_bus[bus] = n_pvpq + position in pq).

Adjoint solve:  Jᵀ λ = x̄   (shape-generic: dim_J, whatever it is)

Scatter to Sbus gradient, using the bus-indexed NRLedger row maps
(``p_row_of_bus`` / ``q_row_of_bus``, same -1 convention):
    grad_Sbus.real[i] = λ[p_row_of_bus[i]]   for every bus i with p_row_of_bus[i] >= 0
    grad_Sbus.imag[i] = λ[q_row_of_bus[i]]   for every bus i with q_row_of_bus[i] >= 0

Rows/columns that don't map back to a bus (e.g. the MultiSlack
``slack_absorbed`` column, or a VoltageControl voltage-constraint/sharing row)
simply carry zero cotangent going in and are never gathered coming out.

Sign note: the code Jacobian J_code = dS_calc/dx has POSITIVE diagonal (it is
dP_calc/dθ not dF/dθ = -dP_calc/dθ). This flips the IFT sign relative to the
textbook formulation, giving +J_code^{-T} xbar rather than -J_IFT^{-T} xbar.

Sbus is split into real/imag parts because torch.autograd.Function requires
real tensors for gradient tracking across all supported PyTorch versions.

Out of scope: no gradient support is exposed for slack weights, HVDC droop
parameters, or controller voltage setpoints — only Sbus (P, Q injections).

# TODO: Ybus_values gradient hook — currently returns None (less useful for
#       the RL topology-switching use case, where topology is discrete).
"""

import numpy as np
import torch
from torch import Tensor


class PowerFlowFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        Sbus_real: Tensor,     # real [n_bus], requires_grad
        Sbus_imag: Tensor,     # real [n_bus], requires_grad
        Ybus=None,             # scipy sparse (n_bus × n_bus) complex; bare path only
        Vinit=None,            # numpy complex [n_bus]; bare path only
        pv=None,               # numpy int [n_pv]; bare path only
        pq=None,               # numpy int [n_pq]; bare path only
        slack_ids=None,        # numpy int [n_slack]; bare path only
        slack_weights=None,    # numpy float [n_slack]; bare path only
        max_iter: int = 10,
        tol: float = 1e-6,
        grid=None,             # solved lightsim2grid grid; augmented path
    ) -> Tensor:
        from gpusim2grid._gpusim2grid import AcPfNrSession

        Sbus_np = (
            Sbus_real.detach().cpu().numpy()
            + 1j * Sbus_imag.detach().cpu().numpy()
        )

        if grid is not None:
            from gpusim2grid._gpusim2grid import (
                _make_acpf_session_from_lsgrid_with_sbus,
            )
            session = _make_acpf_session_from_lsgrid_with_sbus(
                grid, Sbus_np, max_iter, tol,
            )
        else:
            session = AcPfNrSession(
                Ybus, Vinit, Sbus_np,
                slack_ids, slack_weights,
                pv, pq,
                max_iter, tol,
            )

        # clone() materialises d_V so subsequent NR calls (if any) don't race
        V = torch.from_dlpack(session.v_dlpack()).clone()

        ctx.session = session   # keep alive for backward (owns cuDSS factors)
        ctx.save_for_backward(V)
        return V  # complex [n_bus] on CUDA

    @staticmethod
    def backward(ctx, grad_V: Tensor):
        session = ctx.session
        V, = ctx.saved_tensors
        dev = grad_V.device
        n_bus = V.shape[0]

        if V.dtype == torch.complex64:
            rdtype = torch.float32
        else:
            rdtype = torch.float64

        # ------------------------------------------------------------------
        # Bus-indexed NRLedger maps (length n_bus, -1 sentinel). Identical to
        # the bare pvpq/pq positional layout when no augmented ledger is
        # active — see the module docstring.
        # ------------------------------------------------------------------
        theta_col = torch.as_tensor(session.theta_col_of_bus, dtype=torch.long, device=dev)
        vm_col    = torch.as_tensor(session.vm_col_of_bus,    dtype=torch.long, device=dev)
        p_row     = torch.as_tensor(session.p_row_of_bus,     dtype=torch.long, device=dev)
        q_row     = torch.as_tensor(session.q_row_of_bus,     dtype=torch.long, device=dev)

        theta_mask = theta_col >= 0
        vm_mask    = vm_col >= 0

        # ------------------------------------------------------------------
        # Build x̄ — project complex grad_V to the real adjoint state vector
        # ------------------------------------------------------------------
        # theta columns:  ∂V_i/∂θ_i = j·V_i  →  x̄ = Im(conj(V_i)·ḡV_i)
        V_th, gV_th = V[theta_mask], grad_V[theta_mask]
        xbar_theta = (V_th.conj() * gV_th).imag.to(rdtype)

        # Vm columns: ∂V_i/∂|V_i| = V_i/|V_i|  →  x̄ = Re(conj(V_i/|V_i|)·ḡV_i)
        V_vm, gV_vm = V[vm_mask], grad_V[vm_mask]
        xbar_vm = (V_vm.conj() / V_vm.abs() * gV_vm).real.to(rdtype)

        xbar = torch.zeros(session.dim_J, dtype=rdtype, device=dev)
        xbar.scatter_(0, theta_col[theta_mask], xbar_theta)
        xbar.scatter_(0, vm_col[vm_mask], xbar_vm)

        # ------------------------------------------------------------------
        # Adjoint solve: Jᵀ λ = x̄  (via the CUDA primitive)
        # ------------------------------------------------------------------
        xbar_r = xbar.contiguous()

        rhs_cap = xbar_r.__dlpack__()
        sol_cap = session.solve_JT_dlpack(rhs_cap)
        lam = torch.from_dlpack(sol_cap).clone()  # real [dim_J]; clone before reuse

        # ------------------------------------------------------------------
        # Scatter λ back to grad_Sbus (real and imag parts separately)
        # ------------------------------------------------------------------
        # The code Jacobian J_code = dS_calc/dx (positive diagonal) is the
        # NEGATIVE of the mathematical IFT Jacobian dF/dx.  This flips the IFT
        # sign: grad_Sbus = +J_code^{-T} xbar = +lam  (not -lam).
        p_mask, q_mask = p_row >= 0, q_row >= 0
        grad_Sr = torch.zeros(n_bus, dtype=lam.dtype, device=lam.device)
        grad_Si = torch.zeros(n_bus, dtype=lam.dtype, device=lam.device)
        grad_Sr[p_mask] = lam[p_row[p_mask]]
        grad_Si[q_mask] = lam[q_row[q_mask]]

        # Return one gradient per forward argument (11 total).
        # Ybus, Vinit, pv, pq, slack_ids, slack_weights, max_iter, tol, grid → None
        # TODO: Ybus_values gradient hook (return non-None when needed)
        return (grad_Sr, grad_Si,
                None, None, None, None, None, None, None, None, None)


def solve_power_flow(
    Sbus_real: Tensor,
    Sbus_imag: Tensor,
    Ybus=None,
    Vinit=None,
    pv=None,
    pq=None,
    slack_ids=None,
    slack_weights=None,
    max_iter: int = 10,
    tol: float = 1e-6,
    grid=None,
) -> Tensor:
    """
    Differentiable single AC NR power flow.

    Returns complex V [n_bus] on GPU.  Sbus_real and Sbus_imag are the
    real and imaginary parts of the complex injection vector in per-unit;
    both must be float tensors with requires_grad=True for gradients to flow.

    Two mutually exclusive modes:
      - Bare: pass Ybus, Vinit, pv, pq, slack_ids, slack_weights (raw arrays).
        Solves the bare [pvpq | pq] system.
      - Augmented: pass a solved lightsim2grid ``grid`` instead. Solves the
        same augmented system lightsim2grid does (distributed slack / HVDC
        angle-droop / SVC / remote voltage control), via the NRLedger read off
        the grid. Requires the lightsim2grid C++ bridge
        (``gpusim2grid._gpusim2grid.have_ls2g_bridge``).
    """
    return PowerFlowFunction.apply(
        Sbus_real, Sbus_imag,
        Ybus, Vinit, pv, pq, slack_ids, slack_weights,
        max_iter, tol, grid,
    )
