# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

GPU-accelerated AC power flow for electrical grids. The Python package is `gpusim2grid`; the compiled CUDA extension is `gpusim2grid._gpusim2grid`. Three workloads, all built on a single-system Newton-Raphson (NR) core:

- **Single AC NR solve** (`AcPfGPU`)
- **N-k contingency analysis** (`ContingencyAnalysisGPU`) — trip branches, re-solve in parallel
- **Injection sweep** (`InjectionSweepGPU`) — same Ybus, many (P, Q) profiles in parallel

Backed by cuDSS (sparse direct solver), cuSPARSE, cuBLAS/cuSOLVER, and Eigen (header-only, vendored at `src/eigen/`). Grid data (Ybus, Sbus, pv/pq/slack arrays, branch admittances) comes from lightsim2grid/pandapower; this package only consumes those arrays.

## Build

```bash
source env_compile.sh       # exports CUDAToolkit_ROOT, cudss_ROOT, LD_LIBRARY_PATH — edit paths for your machine
pip install .               # FP64 (double) by default
CUDA_REAL_FLOAT=1 pip install .    # FP32 (float)
CUDA_REAL_DOUBLE=1 pip install .   # FP64, explicit
```

`scikit-build-core` (pyproject.toml) → CMake → NVCC. Requires CUDA toolkit + cuDSS ≥ 0.8.0 installed and discoverable (see `env_compile.sh` for the expected layout). Precision is a **compile-time** choice (`-DUSE_FLOAT_PRECISION=ON` or the env vars above); query it at runtime via `gpusim2grid._gpusim2grid.is_fp32` or `compilation_options.is_fp32()`.

## Tests

```bash
pytest tests/python/                              # full suite
pytest tests/python/test_contingency_batch.py     # one file
pytest tests/python/ -m "not slow"                 # skip >10s tests
```

Tests **require a CUDA GPU and the installed extension** — `conftest.py` defines a `requires_gpu` skip marker. Reference values come from lightsim2grid (KLU CPU solver, `ContingencyAnalysisCPP`). Tolerances are precision-aware via the `solver_atol` / `residual_atol` fixtures (`1e-4` FP32, `1e-6` FP64) — never hardcode tolerances in new tests. The shared IEEE 14-bus grid and its solved base case are session-scoped fixtures.

## Benchmarks

```bash
python benchmarks/contingency_analysis.py
python benchmarks/injection_sweep.py
bash   benchmarks/run_sweep_combined.sh    # full FP32+FP64 strategy sweep
```

`run_*_sweep.py` files each sweep one linear-solve strategy; `_grid_setup.py` and `_aux_sweep.py` are shared helpers (underscore prefix = not a standalone entry point).

## Architecture

```
Python API (src/gpusim2grid/)
  └── pybind11 module _gpusim2grid (src/_cpp/python_bindings.cpp)
        └── CUDA/C++ solvers (src/_cpp/**/*.cu,*.cpp)
              └── cuDSS, cuSPARSE, cuBLAS/cuSOLVER, Eigen
```

### The NR core and the policy/strategy pattern

The central abstraction is a **templated NR loop** (`src/_cpp/contingency/driver.cuh`, `run_nr_loop<Policy>`) that runs a fixed number of iterations over a block-diagonal batch. All physics kernels (`fill_FP`, `fill_FQ`, `fill_J`, `update_Va`, `update_Vm` in `acpf_nr_kernels.{cuh,cu}`) are called by the driver, never by policies. A **Policy** decides only *when* to factorize/refactorize/solve the linear system (which `CudssBatchSolver` owns). Strategies live in `src/_cpp/contingency/strategies/`:

| Strategy enum / string | Policy file | Behavior |
|---|---|---|
| `DirectRefactorEvery` / `'direct_refactor_every'` (default) | `policy_refactor_every.cuh` | Fill J every iter; factor once, refactor every iter. Most accurate. |
| `DirectIter0Only` / `'direct_iter0_only'` | `policy_iter0_only.cuh` | Fill/factor J at iter 0 per chunk; solve-only afterward. |
| `DirectBaseCaseFactors` / `'direct_base_case_factors'` | `policy_base_case_factors.cuh` | Reuse base-case LU factors; cheapest, approximate. |
| `DirectRefactorEveryN` / `'direct_refactor_every_n'` | `policy_refactor_every_n.cuh` | Refactor every N iters (`refactor_period`). |

**Adding a new strategy = adding a policy header + an enum value + the string mapping in the Python facades; no kernel changes.** The Python-side string→enum map (`_STRATEGY_MAP`) is duplicated in `contingency_analysis/__init__.py` and `injection_sweep/__init__.py` — keep them in sync.

### Batch sources

`src/_cpp/contingency/batch_sources/` adapts the two workloads to the shared block-diagonal batch the NR loop consumes: `contingency_batch.cuh` patches Ybus (subtracts tripped-branch admittances), `injection_batch.{cuh,cu}` varies Sbus from per-scenario (P, Q). Contingencies that would disconnect the Ybus graph are skipped and their residuals set to NaN.

### Session objects and Python facades

C++ session classes (`AcPfNrSession`, `ContingencyAnalysisSession`, `InjectionSweepSession`) keep state (converted indices, base-case factors, device buffers) on the GPU so a base-case solve is reused across many `run()` calls.

For each workload there is exactly **one** public entry point — `AcPfGPU`, `ContingencyAnalysisGPU`, `InjectionSweepGPU` (all exported from the top-level `gpusim2grid` package) — whose `grid` argument accepts either a solved lightsim2grid grid (extracted zero-copy via the C++ bridge when compiled in, else via the `_ls2g_utils.py` Python fallback) or an explicit `(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)` array tuple for callers without a lightsim2grid grid; `isinstance(grid, (tuple, list))` picks the branch in each facade's `__init__`. In tuple mode, branch data (needed for `compute_flows`/`build_contingencies`) is supplied afterward via `set_branch_data()`, and `use_bridge=True` is invalid (there's no grid to bridge to).

Each facade is implemented in its sibling package's `gpu_facade.py` (e.g. `contingency_analysis/gpu_facade.py`) and delegates in tuple mode to that package's internal engine class — `_ContingencyAnalysisSolver` / `_InjectionSweepSolver` (Python wrappers adding `_normalize_device`, the string strategy map, and `DeviceBuffer` lazy `to_numpy()` D→H copies) or the raw `AcPfNrSession` C++ binding for `AcPfGPU`. These are underscore-prefixed (except `AcPfNrSession`, a compiled binding rather than a Python ergonomics class) precisely because they are *not* a second public entry point — don't reach for them directly in new examples/docs; use the `*GPU` facade with a tuple `grid` instead. `batch_size`, `nb_iter`, `strategy`, `refactor_period` are mutable and take effect on the *next* `run()`; `max_iter_base` / `tol_base` only apply at construction.

### Zero-copy interop (DLPack)

`dlpack_export.{cu,cuh,hpp}` exports device voltage buffers as DLPack capsules (`v_base_dlpack()`, `v_results_dlpack()`) for zero-copy `torch.from_dlpack()` / `jax.dlpack.from_dlpack()`. **These capsules alias live GPU memory** — a subsequent `run()` overwrites them in place; clone the tensor if you need a snapshot, and keep the session object alive while the tensor is in use.

### Differentiable power flow

`src/gpusim2grid/differentiable/` wraps `AcPfNrSession` in a `torch.autograd.Function` (`PowerFlowFunction`, `solve_power_flow`). Backward uses the adjoint method (implicit function theorem): it solves Jᵀλ = x̄ via `AcPfNrSession.solve_JT_dlpack` reusing the converged factorization. Sbus is split into real/imag tensors because autograd needs real tensors. Sign conventions and the x-space projection are documented at the top of `_power_flow_op.py` — read it before touching gradients.

### Types and timing

- `dtypes.hpp` defines `real_t` (`float` or `double` per compile flag) and the Eigen/complex aliases. **Use these aliases in new solver code; never hardcode `float`/`double`.**
- `timing_utils.hpp` defines `TimingEntry` (`gpu_ms` via CUDA events vs `wall_ms` after sync), `AcPfTimings` (single solve), and `BatchTimings` (contingency + injection; exposed to Python under the legacy alias `ContingencyTimings`). Timings carry per-phase breakdowns and are surfaced on every session.

## Conventions

- **Branch ordering is lines-then-trafos**: branch index `c < n_lines` is line `c`; `c >= n_lines` is trafo `c - n_lines`. This matches lightsim2grid's `add_all_n1()` and is assumed by `build_contingencies`, branch-flow output, and the test reference helpers in `conftest.py`.
- The pybind module is `_gpusim2grid` (not the CMake `project()` name). New bindings go in `python_bindings.cpp` and must register enums *before* any binding that uses them as a default argument.
