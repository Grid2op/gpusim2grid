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
pip install --no-build-isolation .               # FP64 (double) by default
CUDA_REAL_FLOAT=1 pip install --no-build-isolation .    # FP32 (float)
CUDA_REAL_DOUBLE=1 pip install --no-build-isolation .   # FP64, explicit
```

`--no-build-isolation` matters because gpusim2grid's C++ bridge build needs
to see the `lightsim2grid` already installed in the current environment
(built from source, matching Eigen/SuiteSparse/compile flags) — not a fresh
one pip would otherwise resolve into an isolated build environment.

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

### Augmented Jacobian (ledger-driven NR)

When solving through the lightsim2grid bridge (the default `grid=<solved LSGrid>` path on all three facades), gpusim2grid does not solve the bare `[pvpq | pq]` system — it reproduces the **same augmented NR system lightsim2grid poses**, including in-Jacobian power-system controls: distributed slack (`MultiSlack`), HVDC angle-droop, and SVC / remote generator voltage control (`VoltageControl`). Results match lightsim2grid bit-for-bit (to ~1e-15) on both the single solve and the batched contingency/injection-sweep paths.

- **`LedgerData`** (`src/_cpp/ledger_data.hpp`) is the host-side description of the augmented Jacobian: the J sparsity skeleton (CSR, structure only), bus-keyed row/column maps (`p_row_of_bus`, `q_row_of_bus`, `theta_col_of_bus`, `vm_col_of_bus`, `q_col_of_bus`), and per-extension feature data (MultiSlack weights, HVDC droop parameters, VoltageControl controller data). It mirrors lightsim2grid's own `NRSystem`/`NRLedger`, which stays the single source of truth for the augmented structure — gpusim2grid only does the batched *numeric* fill/solve on the GPU.
- **`ls2g_bridge.{hpp,cpp}`** (`extract_ledger_data`, `make_acpf_session_from_lsgrid`, and the CA/IS equivalents) build a `LedgerData` off a *solved* `ls2g::LSGrid` — reading `get_J_solver()`, the ledger row/col accessors, and per-extension data straight off the C++ object (zero-copy, no scipy marshalling). `nullptr`/empty `LedgerData` reproduces the legacy feature-free (bare pvpq/pq) system exactly.
- **`use_distributed_slack`** (bool, default `True`, construction-time only, on all three facades): picks between the two slack formulations. `False` calls `drop_multislack_augmentation()` (`ls2g_bridge.cpp`) on the extracted ledger, deleting exactly what MultiSlack owns — the P equation of *every* participant, the theta unknown of every non-reference participant, and the `slack_absorbed` column — so `dim_J` shrinks by `n_slack` and lands on the bare `[pvpq | pq]` size. **HVDC droop and VoltageControl are preserved either way**; this switch only touches the slack. Deleting (rather than rebuilding) is valid because the augmented sparsity pattern is a strict superset of the bare one, so no entry ever has to be added. With one participant it is an exact reformulation (that row/column pair is block-triangular). With several it is a genuine model change, and combined with `init_from_n_powerflow=True` it then legitimately fails the residual check. Note it is *not* a workaround for the cuDSS batch-reordering NaN bug — measured on case300 the bare system is markedly *more* affected (833/1440 vs ~10/1440 NaN at `reordering_alg='default'`); `reordering_alg='none'` and `strategy='direct_base_case_factors'` still fix it in both modes.
- **Feature-free invariant**: the trivial ledger built for the tuple-`grid` (no-lightsim2grid) path reproduces the legacy layout exactly — P/theta on sorted(pvpq) at rows/cols `[0, n_pvpq)`, Q/vm on pq in input order at `[n_pvpq, dim_J)`. Don't break this when touching ledger code — it's what keeps the bare path bit-identical.
- **Feature kernels** are small, conditionally-gated additions layered on the same driver: MultiSlack (`adjust_slack_mismatch`/`fill_slack_feature`/`update_slack_absorbed`, gated on `slack_col>=0`), HVDC droop (`hvdc_adjust_mismatch`/`hvdc_fill_feature`, additive J slopes — forces `d_J_values` to be zeroed before `fill_J` when active), and VoltageControl (`vc_adjust_mismatch`/`vc_vrow`/`vc_share`/`vc_apply_step`, a bordered formulation that adds custom rows). Shared inline helpers `nr_feature_{mismatch,zero_J,fill_J,update}` (`nr_iter_step.cuh`) are called from both the single-system step and the batched `driver.cuh` loop, so CA/InjectionSweep get the augmented system "for free" once the base session carries a `LedgerData*`.
- **`presolved_v` fast path**: when the bridge is told the grid's own `V` is already converged (`init_from_n_powerflow=True`, the `AcPfGPU` default), the GPU trusts it instead of re-running NR from scratch — it factorizes J once at V0 and, for MultiSlack/VoltageControl, recovers the `slack_absorbed`/controller-Q running state (not a pure function of V) two ways: by default it's seeded directly from lightsim2grid's own converged ground truth (`LedgerData::slack_absorbed_gt`/`vc_q_gt`), needing no cuDSS solve at all; only when that ground truth isn't available (array/tuple mode) or `debug_base_case=True` is explicitly requested does it fall back to deriving that state with one solve-only correction, *without moving V* either way. It then verifies the residual. `V` stays bit-identical to the CPU solution.
- **`handle_disconnected_grid`** (contingency analysis only): the batch shares one fixed symbolic structure, so masking a contingency-islanded bus is value-only — identity its J row(s), zero its F row(s), report its V as NaN — solved on the largest connected component instead of skipping the whole scenario. Contingencies that strand the angle reference or a controller (HVDC/SVC/remote-gen) bus are still conservatively NaN-skipped. `optimize_reference_slack(grid, contingency_branch_ids)` (top-level export) re-picks and re-solves the base case with the reference slack least likely to be stranded, minimizing those skips.
- **Differentiable path**: the adjoint backward (`_power_flow_op.py`) indexes through the same bus-keyed maps (`p_row_of_bus`/`q_row_of_bus`/`theta_col_of_bus`/`vm_col_of_bus`, exposed on `AcPfNrSession`) rather than positional pvpq/pq slots, so gradients are correct on both the bare and augmented systems without separate code paths.

### cuDSS solver configuration

Three construction-time-only knobs select cuDSS's analysis-phase behavior, exposed identically on all three facades (`reordering_alg`, `matching_alg`, `pivot_epsilon_alg`, all strings): `ReorderingAlg` (`'default'|'btf_colamd'|'colamd'|'amd'|'nested_dissection'|'none'`), `MatchingAlg` (`'none'|'max_diag_count'|'max_min_diag'|'max_min_diag_alt'|'max_diag_sum'|'max_diag_product'|'auto'`), `PivotEpsilonAlg` (`'default'|'scaled'|'static'`). The string→enum resolvers (`_resolve_reordering_alg`/`_resolve_matching_alg`/`_resolve_pivot_epsilon_alg`) are defined once in `contingency_analysis/__init__.py` and imported by `injection_sweep` and `acpf_nr` — a single source of truth (unlike `_STRATEGY_MAP`, which is duplicated). **Batch-mode restriction**: `MatchingAlg` values other than `NoMatching` (e.g. `MaxDiagProduct`, `Auto`) can silently produce NaN in uniform-batch mode (and even single-system in some cases) — `NoMatching` is the only value validated safe everywhere. `BtfColamd`/`Colamd` reordering only work single-system, not uniform-batch.

### Contingency limit violations

`ContingencyAnalysisGPU.compute_limit_violations` (`_limit_violations.py`, `contingency/violation_kernels.cu`, `limit_violation_types.hpp`) fuses bus/branch limit checking into the batched GPU kernel: `ViolationElementType` (`BUS`/`LINE`/`TRAFO`/`GRID`) and `LimitViolationType` (`LOW_VOLTAGE`/`HIGH_VOLTAGE`/`CURRENT`/`NOT_SIMULATED`/`DIVERGENCE`) mirror lightsim2grid's own codes exactly (`LimitViolation.hpp`, `improve_const_ref` branch), including `GRID`/`NOT_SIMULATED`/`DIVERGENCE` — but the two `GRID` violation types are written from two different layers, not both from the kernel: `DIVERGENCE` is written by `check_limit_violations_kernel` itself, for a contingency it actually ran the solver on but whose residual is NaN or exceeds `violation_tol` (the kernel already computes this residual check as a precondition to trusting `V`, so folding it into the same compact output avoids a second round trip); `NOT_SIMULATED` is written by the Python session layer (`get_violations()`) for a contingency the pre-check dropped before it ever reached that kernel (`BatchPfDriver`'s `d_violation_count` `-1` sentinel) — the solver was never invoked at all. gpusim2grid additionally populates `value`/`limit` with the actual residual/tol for `DIVERGENCE` entries; lightsim2grid's own convention leaves those NaN/unused for `GRID`. These codes are not pybind-bound (the kernel writes raw ints; the Python facade mirrors them as `enum.IntEnum`) — keep the three in sync by construction if you touch any of them.

### Zero-copy interop (DLPack)

`dlpack_export.{cu,cuh,hpp}` exports device voltage buffers as DLPack capsules (`v_base_dlpack()`, `v_results_dlpack()`) for zero-copy `torch.from_dlpack()` / `jax.dlpack.from_dlpack()`. **These capsules alias live GPU memory** — a subsequent `run()` overwrites them in place; clone the tensor if you need a snapshot, and keep the session object alive while the tensor is in use.

### Differentiable power flow

`src/gpusim2grid/differentiable/` wraps `AcPfNrSession` in a `torch.autograd.Function` (`PowerFlowFunction`, `solve_power_flow`). Backward uses the adjoint method (implicit function theorem): it solves Jᵀλ = x̄ via `AcPfNrSession.solve_JT_dlpack` reusing the converged factorization. Sbus is split into real/imag tensors because autograd needs real tensors. Sign conventions and the x-space projection are documented at the top of `_power_flow_op.py` — read it before touching gradients.

### Types and timing

- `dtypes.hpp` defines `real_t` (`float` or `double` per compile flag) and the Eigen/complex aliases. **Use these aliases in new solver code; never hardcode `float`/`double`.**
- `timing_utils.hpp` defines `TimingEntry` (`gpu_ms` via CUDA events vs `wall_ms` after sync), `AcPfTimings` (single solve), and `BatchTimings` (contingency + injection; exposed to Python under the legacy alias `ContingencyTimings`). Timings carry per-phase breakdowns and are surfaced on every session.
- **One-time CUDA init is its own bucket, not compute.** CUDA-context creation plus the cuSPARSE/cuDSS `dlopen`+PTX-JIT costs tens of ms on the first session in a process and scales with nothing. It is measured as `t_context_init_ms` — from `AcPfNrState::ctor_clock_` (the *first-declared member*, because the `CudaStream` member triggers context creation before the constructor body runs) through the cuSPARSE/cuDSS descriptor setup, plus the batch solver's own `CudssBatchSolver::context_init_ms()`. It is excluded from `t_gpu_compute_ms()` and added as a separate term of `t_grand_total_ms()` / the `to_dict()` `"context_init"` bucket. **Call `gpusim2grid.warmup()` (top-level export, `warmup.cu`) before any timed region** — it pays that cost up front via a throwaway 2×2 cuSPARSE SpMV + cuDSS analyze/factorize/solve (single-system *and* uniform-batch) and short-circuits on repeat calls. Both `benchmarks/contingency_analysis.py` and `benchmarks/injection_sweep.py` call it, which also covers the `run_*_sweep.py` runners (they spawn those two as subprocesses).
- Beware the two timing scopes when comparing against an external stopwatch: the base-case solve (`t_base_case_solve_only_ms`) happens at **construction**, while `t_analysis_ms` and the per-chunk totals happen inside `run()`/`compute()`. `t_gpu_compute_ms()` sums all three, so it is only comparable to a stopwatch spanning construction *through* `compute()`.

## Conventions

- **Branch ordering is lines-then-trafos**: branch index `c < n_lines` is line `c`; `c >= n_lines` is trafo `c - n_lines`. This matches lightsim2grid's `add_all_n1()` and is assumed by `build_contingencies`, branch-flow output, and the test reference helpers in `conftest.py`.
- The pybind module is `_gpusim2grid` (not the CMake `project()` name). New bindings go in `python_bindings.cpp` and must register enums *before* any binding that uses them as a default argument.
