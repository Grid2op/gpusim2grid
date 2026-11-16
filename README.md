# gpusim2grid

GPU-accelerated power flow for the [Grid2Op](https://github.com/Grid2Op) ecosystem.

`gpusim2grid` runs AC power flow on the GPU and scales it to large batches, so that
thousands of scenarios can be solved in parallel on a single device. It is built
around batched sparse direct factorization (cuDSS), and is the GPU companion to
[`lightsim2grid`](https://github.com/Grid2op/lightsim2grid), which it uses for CPU
preprocessing and as a reference oracle.

> **Status:** research code, early release, distributed on PyPI as a source
> distribution that `pip` compiles on your machine (see [installation](#installation)).
> The public API is still moving and may change between versions.

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
4. **Scenario sweep on the GPU** — row-aligned combination of the two: row `i`'s
   injection profile is solved together with row `i`'s own set of tripped
   branches, independently of every other row. Mirrors lightsim2grid's own
   `ScenarioSweep`.

The three batch workloads run in one of two schedules: **chunked** (every row of a
chunk runs the same number of iterations) or **continuous** (`scheduling="continuous"`:
a row leaves its slot as soon as it converges and the next row takes it, so the
batch never waits for its slowest row).

Both single (FP32) and double (FP64) precision are supported and work well; the
precision is selectable at build time (see the documentation).

When seeded from a `lightsim2grid` grid — the default path for the `AcPfGPU`,
`ContingencyAnalysisGPU`, `InjectionSweepGPU` and `ScenarioSweepGPU` facades —
gpusim2grid solves the **same augmented Newton-Raphson system that lightsim2grid
poses**, including the in-Jacobian power-system controls modelled there:

- **distributed slack** (slack mismatch shared across weighted slack buses),
- **HVDC angle-droop**,
- **SVC** (static var compensator, voltage mode with optional slope),
- **remote generator voltage control** (a generator regulating a non-local bus).

These match the lightsim2grid CPU solution to rounding (~1e-15), on both the single
solve and the batched contingency / injection-sweep / scenario-sweep paths (a droop
HVDC line with a resistance matches to the Newton tolerance: gpusim2grid takes its
Jacobian slope exactly). The low-level array entry points (`acpf_nr_gpu`, the
`*Session` classes constructed from raw NumPy/SciPy arrays) solve only the bare
`[pvpq | pq]` system without these controls.

## Alpha features

These work today but are early and **subject to change**:

- **Derivatives through the power flow** — the adjoint method, on a single power flow
  and on a whole batch (`BatchPowerFlow`, a PyTorch layer): gradients with respect to
  loads, generator set-points and voltage set-points, with branch and generator
  statuses as discrete inputs, and through the outer loops (the adjoint of each row's
  last inner solve, the loops' decisions held).
- **PyTorch integration via DLPack** — zero-copy export of GPU results as PyTorch
  tensors. JAX support is coming soon through the same path.
  See [DLPack](https://github.com/dmlc/dlpack) and its
  [license](https://github.com/dmlc/dlpack/blob/main/LICENSE).
- **Outer loops per row** (`outer_loops=True` on the three batch facades) — the grid's
  own lightsim2grid outer loops (OpenLoadFlow's reactive limits, distributed slack,
  HVDC AC-emulation limits, voltage monitoring, phase / transformer / shunt voltage
  control) run on the host for every row, its inner solves on the GPU.
- **Limit checks** — bus voltage and branch current violations
  (`compute_limit_violations`) and the conditions OpenLoadFlow's outer loops act on
  (`compute_physical_violations`: reactive limits, PQ → PV releases, slack active-power
  limits, HVDC saturation), checked on the GPU per row.
- **Own batched LU** (`linear_solver="own_lu"`, experimental) — a batched sparse LU in
  place of cuDSS in the batch solves.

Coming soon:

- **DC power flow** on the GPU.
- **JAX** interop through DLPack.

## ⚠️ Important warning

**By default this is a raw Newton-Raphson solver, with no outer loop.** It does not
enforce reactive-power limits (no PV→PQ switching), adjust tap / phase-shifter setpoints,
or perform any other outer-loop correction. The *in-Jacobian* controls that
lightsim2grid models — distributed slack, HVDC angle-droop, SVC, and remote
generator voltage control — **are** solved when you go through the lightsim2grid
bridge (the default facade path), and then match lightsim2grid. The three batch
facades can also run the grid's own lightsim2grid outer loops on every row
(`outer_loops=True`, a grid solved with an `NROuter_*` algorithm: OpenLoadFlow's
reactive limits, distributed slack, HVDC AC-emulation limits, voltage monitoring,
phase / transformer / shunt voltage control); the single solve (`AcPfGPU`) cannot.
Without them, if your problem relies on outer-loop behaviour (Q limits, tap
changing, …), results will differ from a full load-flow tool. See
[`DISCLAIMER.md`](DISCLAIMER.md) for the full list of limitations and for pointers
to other open-source tools that cover them.

## Requirements

- A **CUDA-capable NVIDIA GPU** and its driver.
- Python 3.9+ and a C++17 compiler.
- Everything else comes from PyPI:
  - **[`lightsim2grid`](https://github.com/Grid2op/lightsim2grid) >= 1.3.1 (hard
    dependency)** — CPU preprocessing and reference oracle. Its wheel ships the C++
    core and headers (KLU's included) that gpusim2grid's bridge compiles against.
    Integration with `lightsim2grid` is expected to deepen in future releases.
  - **[cuDSS](https://developer.nvidia.com/cudss) 1.x** (`nvidia-cudss-cu12` /
    `nvidia-cudss-cu13`) — the sparse direct solve.
  - the **CUDA 13 toolkit** (`cuda-toolkit`), compiler included. PyPI has no
    CUDA 12 compiler: with CUDA 12, `nvcc` (12.4 or later) comes from the
    [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installer,
    the libraries and cuDSS from PyPI.

  `pip` installs them itself (see below).

## Installation

```bash
pip install "gpusim2grid[cuda13]"
```

gpusim2grid is a source distribution: `pip` compiles it, in an isolated environment
holding PyPI's CUDA 13 toolkit (`nvcc` included), cuDSS, lightsim2grid, pybind11 and
cmake, and the `cuda13` extra installs the CUDA libraries and cuDSS it runs on, which
it finds next to itself (no `LD_LIBRARY_PATH`). You need an NVIDIA driver recent
enough for CUDA 13 (>= 580) and a C++17 compiler. `CUDA_REAL_FLOAT=1` builds the FP32
version.

With CUDA 12, PyPI has no compiler: name the toolkit's (12.4 or later),
`CUDACXX=/usr/local/cuda-12.4/bin/nvcc pip install "gpusim2grid[cuda12]"`.

From a clone, `pip install ".[cuda13]"` does the same. To compile against the
`lightsim2grid` installed in your environment (built from source, or pinned) instead
of the newest one on PyPI, build without isolation:
`pip install --no-build-isolation ".[cuda13]"`, with scikit-build-core, pybind11 and
cmake installed. `import gpusim2grid` checks that the CUDA libraries, the driver and
lightsim2grid match the build.

For the details — the CUDA major chosen from the environment, a toolkit or cuDSS not
from PyPI, GPU architectures — see the installation page of the
[documentation](#building-the-docs).

## Quickstart

```python
import numpy as np
import gpusim2grid as g2g
from lightsim2grid.network import init_from_pandapower
import pandapower.networks as pn

# Solve the base case on the CPU via lightsim2grid, then screen the
# contingencies on the GPU reusing that single base-case factorization.
grid = init_from_pandapower(pn.case14())
grid.ac_pf(np.ones(grid.get_bus_vn_kv().shape[0], dtype=complex), 10, 1e-8)

ca = g2g.ContingencyAnalysisGPU(grid, nb_iter=4)
ca.add_contingencies_by_branch_id([[12], [4], [12, 4]])    # branch ids, lines-then-trafos

V = ca.compute(batch_size=512)   # DLPack capsule, shape (n_ctg, n_bus), complex
residuals = ca.last_residuals()  # per-contingency ‖F‖∞ (NaN = skipped)
timings = ca.timings             # BatchTimings, per-phase breakdown
```

When an N-k contingency splits the grid, opt into solving the largest connected
component (islanded buses reported as ``NaN``) instead of skipping it. The angle
reference then moves, by default, to the slack participant stranded by the fewest
contingencies (`reference_slack="auto"`):

```python
ca = g2g.ContingencyAnalysisGPU(grid, handle_disconnected_grid=True, nb_iter=4)
ca.add_contingencies_by_branch_id([[12], [4], [12, 4]])
V = ca.compute(batch_size=512)
```

Rows of very different difficulty are better served by the continuous schedule,
each row stopping at its own tolerance within a budget of `nb_iter` iterations:

```python
ca = g2g.ContingencyAnalysisGPU(grid, scheduling="continuous", nb_iter=30)
```

Zero-copy export to PyTorch (the capsule aliases live GPU memory — clone it
before the next ``compute()`` if you need a snapshot):

```python
import torch
V_torch = torch.from_dlpack(ca.compute(batch_size=512))
```

Row `i`'s injection paired with row `i`'s own topology, independently of every
other row — the GPU analogue of lightsim2grid's `ScenarioSweep`:

```python
sweep = g2g.ScenarioSweepGPU(grid, nb_iter=4)
sweep.set_injections_from_elements(load_p, load_q, gen_p)  # (n_scen, n_load/n_gen)
sweep.set_topology([[], [3], [], [3, 12]])                 # one branch-id list per row
V = sweep.compute(batch_size=512)
disconnected = sweep.get_disconnected()  # which rows a topology trip islanded
```

For PyTorch users, the same batch is available as a differentiable layer that
takes lightsim2grid-style per-element inputs (`True = connected` statuses),
solves every row in one GPU pass and back-propagates through it with the
adjoint method — consecutive calls with the same batch size reuse the whole GPU
setup (no new cuDSS analysis, refactorization only), and the transposed system
the backward needs is built lazily on the first `backward()`:

```python
from gpusim2grid.differentiable import BatchPowerFlow
pf = BatchPowerFlow.from_lsgrid(grid, nb_iter=8)
V = pf(load_p=load_p, load_q=load_q, gen_p=gen_p, gen_v=gen_v,   # (n_scen, n_elem) tensors
       line_status=line_status, trafo_status=trafo_status,        # bool, True = connected
       gen_status=gen_status)
loss = (V.abs() - 1.0).pow(2).sum()
loss.backward()   # gradients w.r.t. load_p, load_q, gen_p and gen_v
```

See [`examples/`](examples/):

- `ieee14_basic.py` — end-to-end AC power flow on the IEEE 14-bus case.
- `case6515rte_screen.py` — batched contingency screen with residuals.
- `injection_sweep_scan.py` — batched load-scaling injection sweep.
- `handle_disconnected.py` — solve the largest component of a grid-splitting contingency.
- `limit_violations.py` — fused per-contingency bus voltage / branch current limit checking.
- `distributed_slack.py` — augmented solve (distributed slack in the Jacobian) via the lightsim2grid bridge.
- `differentiable_pf.py` — derivatives through a single power flow via the adjoint method.
- `batch_differentiable_pf.py` — `BatchPowerFlow`: a batch of scenarios (injections, set-points, branch statuses) as one differentiable PyTorch layer, trained over a few steps.
- `pypowsybl_ca_and_differentiable_pf.ipynb` — a narrated walk-through on a real snapshot loaded with pypowsybl: N-1 analysis with limit violations and `BatchPowerFlow` gradients.

## How it works

The base case is solved once (on the CPU via lightsim2grid, or on the GPU) to
get the converged voltages and the Jacobian sparsity pattern. The batch is then
solved in GPU slots: each scenario starts from the base-case voltages, its Ybus
is patched in place (contingencies subtract the tripped-branch admittances;
injection sweeps vary Sbus; the scenario sweep does both, row-aligned), and the
Newton-Raphson iterations run in parallel over the block-diagonal system — a
fixed number per chunk of rows, or, in the continuous schedule, until each row
converges, a freed slot taking the next row at once. The cuDSS symbolic
factorization is computed **once** and reused across the whole batch, and
across `compute()` calls while nothing that shapes it changes — only numeric
refactorization/solve happen per iteration. A residual pass reports the final
‖F‖∞ per scenario.

See the [API documentation](https://gpusim2grid.readthedocs.io) for the full design.

## Running the tests

Python integration tests use pytest and are validated against `lightsim2grid` /
`pandapower` as CPU oracles:

```bash
pip install -e ".[test]"
pytest
```

The IEEE 14-bus case is the canonical regression grid.

C++/CUDA unit tests (`tests/cpp`, no external framework) are opt-in, in a separate build
directory; a test needing a GPU is skipped without one:

```bash
cmake -S . -B build_cpptests -DGPUSIM2GRID_BUILD_TESTS=ON -DSKBUILD_PROJECT_NAME=gpusim2grid
cmake --build build_cpptests && ctest --test-dir build_cpptests
```

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

- **Extend derivatives**: a JAX front-end on the same DLPack primitives, and the
  plain contingency-analysis path.
- **Best action selector** — given one or several grid snapshots and a list of
  candidate actions, find the best action(s) to apply, by evaluating the candidates in
  batch on the GPU.
- **Performance** — a single kernel assembling the Jacobian and the mismatch straight
  into the batched LU's own layout, and the LU itself (factor and solve fused).
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
redistributed components, and remain under their respective NVIDIA licenses. The vendored
DLPack header (`src/_cpp/dlpack/dlpack.h`) is under the Apache License 2.0, its
[license](src/_cpp/dlpack/LICENSE) next to it.

## Acknowledgements

Developed at RTE. Part of the [Grid2Op](https://github.com/Grid2Op) ecosystem and built
on [`lightsim2grid`](https://github.com/Grid2op/lightsim2grid). Uses
[Eigen](https://eigen.tuxfamily.org), [KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse),
[cuDSS](https://developer.nvidia.com/cudss), and cuSPARSE. The batched LU behind
`linear_solver='own_lu'` is Ralf Cortes Do Nascimento's contribution.
