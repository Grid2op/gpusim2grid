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
4. **Scenario sweep on the GPU** — row-aligned combination of the two: row `i`'s
   injection profile is solved together with row `i`'s own set of tripped
   branches, independently of every other row. Mirrors lightsim2grid's own
   `ScenarioSweep`.

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

These match the lightsim2grid CPU solution bit-for-bit, on both the single solve
and the batched contingency / injection-sweep / scenario-sweep paths. The
low-level array entry points (`acpf_nr_gpu`, the `*Session` classes constructed
from raw NumPy/SciPy arrays) solve only the bare `[pvpq | pq]` system without
these controls.

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
reactive-power limits (no PV→PQ switching), adjust tap / phase-shifter setpoints, or
perform any other outer-loop correction. The *in-Jacobian* controls that
lightsim2grid models — distributed slack, HVDC angle-droop, SVC, and remote
generator voltage control — **are** solved when you go through the lightsim2grid
bridge (the default facade path), and then match lightsim2grid. But if your problem
relies on outer-loop behaviour (Q limits, tap changing, …), results will differ from
a full load-flow tool. See [`DISCLAIMER.md`](DISCLAIMER.md) for the full list of
limitations and for pointers to other open-source tools that cover them.

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
pip install --no-build-isolation .
```

This automatically builds the zero-copy C++ bridge to `lightsim2grid`
(`lightsim2grid_core`) — CMake locates it via the Python interpreter it is
building against, since `lightsim2grid` is a build-system requirement. No
`CMAKE_ARGS` needed for the common case; see the
[installation docs](#building-the-docs) if you need to point at a different
`lightsim2grid` install. `--no-build-isolation` matters here too: it makes
gpusim2grid's own build use the `lightsim2grid` already installed in your
current environment (matching Eigen/SuiteSparse/compile flags) instead of
pip resolving a fresh, isolated one that may not match.

> **`lightsim2grid` should also be built from source.** The PyPI wheel does
> not ship the C++ headers (`LSGrid.hpp` and friends) the bridge needs, only
> the compiled extension — so build it yourself first, with
> `--no-build-isolation` so it's built against the same environment
> gpusim2grid will use. It also needs its own submodules (SuiteSparse,
> Eigen) checked out, or it won't compile:
> ```bash
> git clone --recurse-submodules https://github.com/Grid2op/lightsim2grid.git
> cd lightsim2grid
> pip install --no-build-isolation .
> ```
> (or the equivalent `uv pip install --no-build-isolation .`). See the
> [installation docs](#building-the-docs) for details on matching Eigen /
> SuiteSparse versions between the two builds.

For detailed installation options — selecting float precision, linking against a
specific CUDA version, and other build customization — see the dedicated installation
page in the [documentation](#building-the-docs).

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
ca.add_contingencies_by_branch_id([[12], [40], [12, 40]])  # branch ids, lines-then-trafos

V = ca.compute(batch_size=512)   # DLPack capsule, shape (n_ctg, n_bus), complex
residuals = ca.last_residuals()  # per-contingency ‖F‖∞ (NaN = skipped)
timings = ca.timings             # BatchTimings, per-phase breakdown
```

When an N-k contingency splits the grid, opt into solving the largest connected
component (islanded buses reported as ``NaN``) instead of skipping it:

```python
# Pick the angle-reference slack stranded by the fewest contingencies and
# re-solve the base case with it (minimises skips), then enable the masking.
cont = [[c] for c in range(ca.n_branches)]
g2g.optimize_reference_slack(grid, cont)
ca = g2g.ContingencyAnalysisGPU(grid, handle_disconnected_grid=True, nb_iter=4)
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
sweep.set_topology([[], [3], [], [3, 40]])                 # one branch-id list per row
V = sweep.compute(batch_size=512)
disconnected = sweep.get_disconnected()  # which rows a topology trip islanded
```

See [`examples/`](examples/):

- `ieee14_basic.py` — end-to-end AC power flow on the IEEE 14-bus case.
- `case6515rte_screen.py` — batched contingency screen with residuals.
- `injection_sweep_scan.py` — batched load-scaling injection sweep.
- `scenario_sweep_scan.py` — row-aligned combined topology + injection sweep.
- `handle_disconnected.py` — solve the largest component of a grid-splitting contingency.
- `limit_violations.py` — fused per-contingency bus voltage / branch current limit checking.
- `distributed_slack.py` — augmented solve (distributed slack in the Jacobian) via the lightsim2grid bridge.
- `differentiable_pf.py` — derivatives through a single power flow via the adjoint method.

## How it works

The base case is solved once (on the CPU via lightsim2grid, or on the GPU) to
get the converged voltages and the Jacobian sparsity pattern. The batch is then
processed in GPU chunks: each scenario starts from the base-case voltages, its
Ybus is patched in place (contingencies subtract the tripped-branch admittances;
injection sweeps vary Sbus; the scenario sweep does both, row-aligned), and a
fixed number of Newton-Raphson iterations run in parallel over the
block-diagonal system. The cuDSS symbolic factorization is computed **once**
and reused across the whole batch — only numeric refactorization/solve happen
per iteration. A post-loop residual pass reports the final ‖F‖∞ per scenario.

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

Copyright © <!--RTE (<https://www.rte-france.com>) REMOVE FOR DOUBLE BLIND, WILL BE ADDED BACK LATER --> . Released as part of the
[Linux Foundation Energy (LF Energy)](https://www.lfenergy.org/) ecosystem. 

NVIDIA CUDA libraries (CUDA Toolkit, cuDSS, cuSPARSE) are dependencies, not
redistributed components, and remain under their respective NVIDIA licenses. DLPack is
used under its own [license](https://github.com/dmlc/dlpack/blob/main/LICENSE).

<!-- ## Acknowledgements

REMOVE FOR DOUBLE BLIND, WILL BE ADDED BACK LATER

Developed at RTE. Part of the [Grid2Op](https://github.com/Grid2Op) ecosystem and built
on [`lightsim2grid`](https://github.com/Grid2op/lightsim2grid). Uses
[Eigen](https://eigen.tuxfamily.org), [KLU](https://github.com/DrTimothyAldenDavis/SuiteSparse),
[cuDSS](https://developer.nvidia.com/cudss), and cuSPARSE. -->
