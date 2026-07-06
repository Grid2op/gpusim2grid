"""
differentiable_pf.py — derivatives through a single AC power flow.

Wraps the GPU Newton-Raphson solver in a torch.autograd.Function and
differentiates a scalar loss (total voltage magnitude) with respect to the bus
injections, using the adjoint method (the converged Jacobian factorization is
reused for the backward solve — no second forward solve).

Currently the differentiable path is limited to a single power flow (not the
batched / contingency path). Requires PyTorch with CUDA.

Run:
    python examples/differentiable_pf.py
"""
import numpy as np

from _common import load_case
from gpusim2grid.compilation_options import is_fp32


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable PyTorch build.")
    if is_fp32:
        print("NOTE: gpusim2grid was built in FP32; gradients use FP64 internally "
              "and are most accurate against an FP64 build.")

    from gpusim2grid.differentiable import solve_power_flow

    d = load_case("case14")
    Sbus = d["Sbus"].astype(np.complex128)

    # Injections are the differentiable inputs: split into real/imag GPU leaves.
    Sr = torch.tensor(Sbus.real, dtype=torch.float64, device="cuda", requires_grad=True)
    Si = torch.tensor(Sbus.imag, dtype=torch.float64, device="cuda", requires_grad=True)

    # Forward: complex voltages V [n_bus] on the GPU.
    V = solve_power_flow(
        Sr, Si,
        d["Ybus"], d["v_init"].copy().astype(np.complex128),
        d["pv"], d["pq"], d["slack"], d["slack_weights"],
        max_iter=10, tol=1e-8,
    )

    # A real scalar loss, then backprop to the injections.
    loss = V.abs().sum()
    loss.backward()

    print(f"loss (sum |V|)        : {loss.item():.6f}")
    print(f"grad wrt P (||.||2)   : {Sr.grad.norm().item():.6e}")
    print(f"grad wrt Q (||.||2)   : {Si.grad.norm().item():.6e}")
    print(f"largest |dLoss/dP|    : {Sr.grad.abs().max().item():.6e}")


if __name__ == "__main__":
    main()
