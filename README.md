# gpusim2grid

GPU-accelerated power flow for the [Grid2Op](https://github.com/Grid2Op) ecosystem.

`gpusim2grid` runs AC power flow on the GPU and scales it to large batches, so that
thousands of scenarios can be solved in parallel on a single device. It is built
around batched sparse direct factorization (cuDSS), and is the GPU companion to
[`lightsim2grid`](https://github.com/Grid2op/lightsim2grid), which it uses for CPU
preprocessing and as a reference oracle.

> **Status:** research code, early release. No PyPI package yet — **install from
> source only** (see [installation](#installation)). The public API is still moving
> and may change between versions.

> **Hardware:** this package requires a CUDA-capable NVIDIA device (i.e. an NVIDIA
> GPU). It cannot run on CPU-only machines or on non-NVIDIA accelerators.

Please also read the [`DISCLAIMER.md`](DISCLAIMER.md) on the intended scope and
limitations of this solver before using it for power-system analysis.

## What it does

In order of maturity:

1. **AC power flow on the GPU** — a Newton-Raphson solver for the full nonlinear AC
   power flow.
2. **Contingency analysis on the GPU** — screen thousands of N-k contingencies in
   parallel, reusing one symbolic factorization across the batch.
3. **Injection sweep on the GPU** — solve the same grid across many injection
   scenarios (varying loads/generation) in a single batched call.

Both single (FP32) and double (FP64) precision are supported and work well; the
precision is selectable at build time (see the documentation).

## Alpha features

These work today but are early and **subject to change**:

- **Derivatives through the power flow** — differentiate through the solver. Currently
  limited to a **single** power flow (not yet the batched / contingency path).
- **PyTorch integration via DLPack** — zero-copy export of GPU results as PyTorch
  tensors. JAX support is coming soon through the same path.
  See [DLPack](https://github.com/dmlc/dlpack) and its
  [license](https://github.com/dmlc/dlpack/blob/main/LICENSE).

Coming soon:

- **DC power flow** on the GPU.
- **JAX** interop through DLPack.

## ⚠️ Important warning

**This is a raw Newton-Raphson solver. There is no outer loop.** It does not enforce
reactive power limits, adjust tap / phase-shifter control, apply distributed slack, or
perform any other outer-loop correction. Voltages are the solution of the bare AC
equations as posed. If your problem relies on outer-loop behaviour, results will
differ from a full load-flow tool. See [`DISCLAIMER.md`](DISCLAIMER.md) for the full
list of limitations and for pointers to other open-source tools that cover them.

## Requirements

- A **CUDA-capable NVIDIA GPU**.
- The **[NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**
  12.x+.
- **[cuDSS](https://developer.nvidia.com/cudss) >= 0.8.0** — required for the sparse
  direct solve.
- **cuSPARSE** (ships with the CUDA Toolkit).
- Python 3.9+, a C++17 compiler, CMake >= 3.15.
- **[`lightsim2grid`](https://github.com/Grid2op/lightsim2grid) (hard dependency)** —
  CPU preprocessing and reference oracle. Integration with `lightsim2grid` is expected
  to deepen in future releases.

> NVIDIA libraries (cuDSS, cuSPARSE) are **not redistributed** with this package.
> Install the CUDA Toolkit and cuDSS yourself before building, using the links above.

## Installation

Install from source (no PyPI release yet):

```bash
git clone https://github.com/Grid2Op/gpusim2grid.git
cd gpusim2grid
pip install .
```

For detailed installation options — selecting float precision, linking against a
specific CUDA version, and other build customization — see the dedicated installation
page in the [documentation](#building-the-docs).

## Quickstart

```python
import gpusim2grid as g2g

# Base case solved to convergence on CPU via lightsim2grid,
# then the batched scenarios are solved on the GPU.
solver = g2g.ContingencyAnalysisSolver(...)

results = solver.run(
    contingencies=...,   # list[list[tuple[row, col, val]]]
    batch_size=...,
    nb_iter=...,         # fixed iteration count, no convergence stop
)

V = results.voltages          # on GPU
residuals = results.residuals # per-scenario ‖F‖∞
timings = results.timings     # ContingencyTimings, per-phase
```

Zero-copy export to PyTorch:

```python
import torch
V_torch = torch.from_dlpack(results.voltages_dlpack())
```

See [`examples/`](examples/):

- `ieee14_basic.py` — end-to-end AC power flow on the IEEE 14-bus case.
- `case6515rte_screen.py` — batched contingency screen with residuals.
- `differentiable_pf.py` — derivatives through a single power flow via the adjoint method.

## How it works

[2-3 sentence overview: base-case solve, chunked batch loop with absolute Ybus
patches, amortized symbolic factorization, post-loop residual.]

See the [API documentation](https://gpusim2grid.readthedocs.io) for the full design.

## Running the tests

Python integration tests use pytest and are validated against `lightsim2grid` /
`pandapower` as CPU oracles:

```bash
pip install -e ".[test]"
pytest
```

The IEEE 14-bus case is the canonical regression grid.

> C++/CUDA unit tests are not set up yet — contributions welcome (see
> [roadmap](#roadmap)).

## Building the docs

Documentation is hosted on Read the Docs. It is built from Python docstrings and the
pybind11 binding docstrings — there is nothing C++-specific to build separately.

```bash
pip install -e ".[docs]"
cd docs
make html        # output in docs/_build/html
```

## Contributing

Contributions are welcome — issues, ideas, and pull requests alike. A good PR:

- targets a single concern and keeps the diff focused;
- adds or updates tests for any behaviour change;
- keeps the public API stable, or flags the break explicitly;
- passes the test suite locally before review.

If you are planning a larger change, please open an issue first so we can align on
design. See the [roadmap](#roadmap) below for areas where help is especially welcome.

## Roadmap

Directions we plan to pursue — **any help or ideas are very welcome**:

- **Extend derivatives to the injection sweep**, and later to the full contingency
  analysis path (currently differentiation is limited to a single power flow).
- **Best action selector** — given one or several grid snapshots and a list of
  candidate actions, find the best action(s) to apply, by evaluating the candidates in
  batch on the GPU.
- **Performance** — for example, reusing the KLU ordering already computed by
  `lightsim2grid` during the base-case solve, instead of recomputing an ordering on
  the GPU.
- Deeper `lightsim2grid` integration and a simpler, more ergonomic public API.

### Explored but not yet implemented

We have considered the following directions and record them here as future work; we
simply have not had the time to implement them yet:

- **Reinforcement learning integration** — using the batched GPU solver (and its
  derivatives) as a fast, differentiable environment / inner loop for RL agents
  operating on power grids.

## License

Licensed under the **Mozilla Public License 2.0 (MPL-2.0)** — see [`LICENSE`](LICENSE)
and [`DISCLAIMER.md`](DISCLAIMER.md).

Copyright © RTE (<https://www.rte-france.com>). Released as part of the
[Linux Foundation Energy (LF Energy)](https://www.lfenergy.org/) ecosystem.

NVIDIA CUDA libraries (CUDA Toolkit, cuDSS, cuSPARSE) are dependencies, not
redistributed components, and remain under their respective NVIDIA licenses. DLPack is
used under its own [license](https://github.com/dmlc/dlpack/blob/main/LICENSE).

## Acknowledgements

Developed at RTE. Part of the [Grid2Op](https://github.com/Grid2Op) ecosystem and built
on [`lightsim2grid`](https://github.com/Grid2op/lightsim2grid). Uses
[Eigen](https://eigen.tuxfamily.org), [KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse),
[cuDSS](https://developer.nvidia.com/cudss), and cuSPARSE.
