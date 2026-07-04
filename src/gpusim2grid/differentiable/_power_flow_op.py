# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
PowerFlowFunction — torch.autograd.Function wrapping the GPU NR AC power flow.

Forward:  Runs AcPfNrSession, returns complex V [n_bus] on GPU.
Backward: Solves Jᵀλ = x̄ via the adjoint method (implicit function theorem),
          scatters λ back to grad_Sbus.

Adjoint math
------------
NR solves F(x, Sbus) = 0 at convergence, where:
    x = [θ_pvpq ; |V|_pq]   (real, dim_J = n_pvpq + n_pq)

Given dL/dV (complex grad_V), project to the real x-space:
    x̄[k]          = Im(conj(V[pvpq[k]]) * grad_V[pvpq[k]])   k < n_pvpq
    x̄[n_pvpq + k] = Re(conj(V[pq[k]]/|V[pq[k]]|) * grad_V[pq[k]])

Adjoint solve:  Jᵀ λ = x̄

Scatter to Sbus gradient:
    grad_Sbus.real[pvpq[k]] += λ[k]            k < n_pvpq
    grad_Sbus.imag[pq[k]]   += λ[n_pvpq + k]  k < n_pq

Sign note: the code Jacobian J_code = dS_calc/dx has POSITIVE diagonal (it is
dP_calc/dθ not dF/dθ = -dP_calc/dθ). This flips the IFT sign relative to the
textbook formulation, giving +J_code^{-T} xbar rather than -J_IFT^{-T} xbar.

Sbus is split into real/imag parts because torch.autograd.Function requires
real tensors for gradient tracking across all supported PyTorch versions.

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
        Ybus,                  # scipy sparse (n_bus × n_bus) complex
        Vinit,                 # numpy complex [n_bus]
        pv,                    # numpy int [n_pv]
        pq,                    # numpy int [n_pq]
        slack_ids,             # numpy int [n_slack]
        slack_weights,         # numpy float [n_slack]
        max_iter: int,
        tol: float,
    ) -> Tensor:
        from gpusim2grid._gpusim2grid import AcPfNrSession

        Sbus_np = (
            Sbus_real.detach().cpu().numpy()
            + 1j * Sbus_imag.detach().cpu().numpy()
        )
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

        pvpq = torch.tensor(session.pvpq, dtype=torch.long, device=grad_V.device)
        pq_  = torch.tensor(session.pq,   dtype=torch.long, device=grad_V.device)
        n_pvpq = session.n_pvpq
        n_pq   = session.n_pq
        n_bus  = V.shape[0]

        # ------------------------------------------------------------------
        # Build x̄ — project complex grad_V to the real adjoint state vector
        # ------------------------------------------------------------------
        # θ block (pvpq buses):  ∂V_k/∂θ_k = j·V_k  →  x̄_k = Im(conj(V_k)·ḡV_k)
        V_pvpq  = V[pvpq]
        gV_pvpq = grad_V[pvpq]
        xbar_theta = (V_pvpq.conj() * gV_pvpq).imag  # real [n_pvpq]

        # |V| block (pq buses): ∂V_k/∂|V_k| = V_k/|V_k|  →  x̄_k = Re(conj(V_k/|V_k|)·ḡV_k)
        V_pq  = V[pq_]
        gV_pq = grad_V[pq_]
        xbar_vm = (V_pq.conj() / V_pq.abs() * gV_pq).real  # real [n_pq]

        xbar = torch.cat([xbar_theta, xbar_vm])  # real [dim_J]

        # ------------------------------------------------------------------
        # Adjoint solve: Jᵀ λ = x̄  (via the CUDA primitive)
        # ------------------------------------------------------------------
        # Match the real dtype to the solver precision
        if V.dtype == torch.complex64:
            rdtype = torch.float32
        else:
            rdtype = torch.float64
        xbar_r = xbar.to(dtype=rdtype).contiguous()

        rhs_cap = xbar_r.__dlpack__()
        sol_cap = session.solve_JT_dlpack(rhs_cap)
        lam = torch.from_dlpack(sol_cap).clone()  # real [dim_J]; clone before reuse

        # ------------------------------------------------------------------
        # Scatter λ back to grad_Sbus (real and imag parts separately)
        # ------------------------------------------------------------------
        grad_Sr = torch.zeros(n_bus, dtype=lam.dtype, device=lam.device)
        grad_Si = torch.zeros(n_bus, dtype=lam.dtype, device=lam.device)

        # grad_Sbus.real[pvpq] -= λ[:n_pvpq]
        # The code Jacobian J_code = dS_calc/dx (positive diagonal) is the
        # NEGATIVE of the mathematical IFT Jacobian dF/dx.  This flips the IFT
        # sign: grad_Sr = +J_code^{-T} xbar = +lam  (not -lam).
        grad_Sr.scatter_add_(0, pvpq, +lam[:n_pvpq])
        grad_Si.scatter_add_(0, pq_, +lam[n_pvpq:])

        # Return one gradient per forward argument (10 total).
        # Ybus, Vinit, pv, pq, slack_ids, slack_weights, max_iter, tol → None
        # TODO: Ybus_values gradient hook (return non-None when needed)
        return grad_Sr, grad_Si, None, None, None, None, None, None, None, None


def solve_power_flow(
    Sbus_real: Tensor,
    Sbus_imag: Tensor,
    Ybus,
    Vinit,
    pv,
    pq,
    slack_ids,
    slack_weights,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> Tensor:
    """
    Differentiable single AC NR power flow.

    Returns complex V [n_bus] on GPU.  Sbus_real and Sbus_imag are the
    real and imaginary parts of the complex injection vector in per-unit;
    both must be float tensors with requires_grad=True for gradients to flow.
    """
    return PowerFlowFunction.apply(
        Sbus_real, Sbus_imag,
        Ybus, Vinit, pv, pq, slack_ids, slack_weights,
        max_iter, tol,
    )