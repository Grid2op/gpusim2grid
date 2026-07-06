# Examples

Runnable examples for `gpusim2grid`. Each requires a CUDA-capable NVIDIA GPU and
an installed build of the package, plus `lightsim2grid` / `pandapower` for the
grid data:

```bash
pip install -e ".[test]"   # pulls in lightsim2grid + pandapower
python examples/ieee14_basic.py
```

(Run them from the repository root so the shared `examples/_common.py` helper is
importable.)

| File | What it shows |
|---|---|
| `ieee14_basic.py` | End-to-end single AC power flow on IEEE 14-bus, compared against the lightsim2grid KLU reference. |
| `case6515rte_screen.py` | Batched N-1 contingency screen reusing one base-case factorization; reports convergence stats and timing. Takes optional `<case> <batch_size>` arguments. |
| `injection_sweep_scan.py` | Batched load-scaling sweep (same Ybus, many Sbus) reusing one base-case factorization; reports convergence stats and timing. Takes optional `<case> <batch_size> <scale_lo> <scale_hi>` arguments. |
| `handle_disconnected.py` | Solve the largest connected component of a grid-splitting contingency (islanded buses reported as NaN) instead of skipping it, via the `ContingencyAnalysisGPU` facade. Takes an optional `<case>` argument. |
| `limit_violations.py` | N-1 screen with `compute_limit_violations=True`: fused, on-device per-contingency bus voltage / branch current / divergence check, reported as a bounded per-contingency violation list. Takes an optional `<case>` argument. |
| `distributed_slack.py` | Augmented solve: distributed slack carried in the Jacobian (via the lightsim2grid C++ bridge), matched against the CPU reference. Same path also covers HVDC droop, SVC, and remote voltage control. Needs a bridge-enabled build. |
| `differentiable_pf.py` | Derivatives through a single power flow (adjoint method) via the PyTorch autograd integration. Requires PyTorch with CUDA. |

`_common.py` is a shared helper (not a standalone example): it wraps
lightsim2grid/pandapower to produce the plain NumPy/SciPy arrays the solver
consumes.
