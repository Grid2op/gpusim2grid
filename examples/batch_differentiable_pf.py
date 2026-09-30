# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
batch_differentiable_pf.py — a batch of power flows as a differentiable
PyTorch layer, driven by lightsim2grid-style per-element inputs.

``BatchPowerFlow`` takes, per scenario (row), the load / generator injections,
the generator voltage set-points and the line / trafo statuses, solves the whole
batch in ONE GPU pass, and back-propagates through it with the adjoint method.
Here a tiny model learns a generator redispatch + voltage set-points that keep
every bus of the IEEE 14-bus grid near 1.0 pu under random load scalings and
random N-1 line trips.

The point of the timings printed at the end: the first call builds the GPU
batch driver (cuDSS analysis + first factorization); every later call with the
same batch size reuses it (only the new rows move to the GPU, the Jacobians are
refactorized), and the transposed system the backward needs is built once, on
the first backward, then only refactorized.

Requires PyTorch with CUDA. Run:
    python examples/batch_differentiable_pf.py
"""
import numpy as np

from _common import load_case
from gpusim2grid.compilation_options import is_fp32


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable PyTorch build.")
    if is_fp32:
        print("NOTE: gpusim2grid was built in FP32; gradients are most accurate "
              "against an FP64 build.")

    from gpusim2grid.differentiable import BatchPowerFlow

    grid = load_case("case14")["grid"]
    pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=8, tol_base=1e-10)
    dev = pf.device
    rdt = torch.float32 if is_fp32 else torch.float64

    n_scen = 64
    rng = np.random.default_rng(0)
    # Random load scalings and one random line trip per row (True = connected).
    scales = torch.tensor(rng.uniform(0.8, 1.2, size=(n_scen, 1)), dtype=rdt, device=dev)
    load_p = pf._load_p_base[None, :] * scales
    load_q = pf._load_q_base[None, :] * scales
    line_status = torch.ones(n_scen, pf.n_line, dtype=torch.bool, device=dev)
    line_status[torch.arange(n_scen), torch.tensor(rng.integers(2, 12, size=n_scen))] = False

    # Learnable per-generator redispatch (MW) and voltage set-points (pu),
    # shared across the batch; the slack absorbs the mismatch.
    dp = torch.zeros(pf.n_gen, dtype=rdt, device=dev, requires_grad=True)
    vset = torch.full((pf.n_gen,), 1.04, dtype=rdt, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dp, vset], lr=5e-3)

    for step in range(15):
        opt.zero_grad()
        gen_p = (pf._gen_p_base + dp)[None, :].expand(n_scen, -1)
        gen_v = vset[None, :].expand(n_scen, -1)
        V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_v=gen_v,
               line_status=line_status)
        valid = torch.isfinite(V.real)                       # islanded rows are NaN
        vm = torch.where(valid, V, torch.ones_like(V)).abs()
        loss = ((vm - 1.0) ** 2 * valid).sum() / valid.sum() + 1e-4 * (dp ** 2).sum()
        loss.backward()
        opt.step()
        t = pf.timings
        print(f"step {step:2d}  loss {loss.item():.3e}  "
              f"driver builds {pf.sweep.driver_build_counter}  "
              f"analysis {t.t_analysis_ms:6.2f} ms  first-factorize {t.t_first_factorize.wall_ms:5.2f} ms  "
              f"refactorize x{t.n_refactorize}  "
              f"adjoint: analysis x{t.adjoint_n_analysis} factorize x{t.adjoint_n_factorize} "
              f"refactorize x{t.adjoint_n_refactorize} solve x{t.adjoint_n_solve}")

    print(f"learned redispatch (MW): {dp.detach().cpu().numpy().round(3)}")
    print(f"learned set-points (pu): {vset.detach().cpu().numpy().round(4)}")


if __name__ == "__main__":
    main()
