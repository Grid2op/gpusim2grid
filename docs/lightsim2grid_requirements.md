# What gpusim2grid requires from lightsim2grid

This document is the contract between **gpusim2grid** (GPU batched AC power flow)
and **lightsim2grid** (CPU physics + base-case solve). gpusim2grid is a
*companion* package: lightsim2grid owns the grid model, the admittances, and the
base-case (N) Newton-Raphson solve; gpusim2grid consumes the solved state and
runs the batched GPU phase (contingencies / injection sweeps).

It lists every lightsim2grid API gpusim2grid depends on, its current status, and
the small set of additions still needed. Verified against the
`n1_full_compute` branch (lightsim2grid `0.13.2.dev0`).

---

## 1. Build / packaging requirements

gpusim2grid links lightsim2grid's C++ core at compile time (no Python round-trip
for Ybus/Sbus/V). This reuses lightsim2grid's installed CMake package, exactly as
the `examples/external_algorithm` CI job (`test_plugin_against_installed`) does.

| Requirement | API | Status |
|---|---|---|
| Locate the installed CMake package from Python | `lightsim2grid.get_cmake_dir()` → `.../share/cmake/lightsim2grid_core` | ✅ present |
| Header-only include dir (alt. to find_package) | `lightsim2grid.get_include_dir()` | ✅ present |
| CMake package config | `find_package(lightsim2grid_core CONFIG)` | ✅ present (`lightsim2grid_coreConfig.cmake`) |
| Exported CMake target | `lightsim2grid::core` (INTERFACE, headers + `liblightsim2grid_core.so`) | ✅ present |
| Bundled Eigen inside the install include dir | `${lightsim2grid_core_INCLUDE_DIRS}` contains `Eigen/` | ✅ present |
| Key public headers | `LSGrid.hpp`, `powerflow_algorithm/BaseAlgo.hpp`, `AlgorithmRegistry.hpp`, `Utils.hpp` | ✅ present |

**gpusim2grid build invocation**

```bash
LS2G_CMAKE=$(python -c "import lightsim2grid; print(lightsim2grid.get_cmake_dir())")
pip install -v . --no-build-isolation \
  -C cmake.args="-DLIGHTSIM2GRID_CMAKE_DIR=$LS2G_CMAKE"
```

gpusim2grid uses the Eigen shipped *inside* `lightsim2grid_core` when found
(falling back to its vendored `src/eigen/` otherwise) so both sides share one
Eigen version — important because they exchange `Eigen::SparseMatrix` /
`Eigen::Ref` types directly across the ABI boundary.

**ABI rule**: gpusim2grid must be compiled against the *same* lightsim2grid it
will import at runtime (same compiler/stdlib). The pybind cross-module cast of a
Python `LSGrid` to `ls2g::LSGrid&` only works when the loaded
`lightsim2grid_cpp` extension matches the headers gpusim2grid compiled against.
Mixing a stale `.so` with newer headers crashes or fails the cast.

---

## 2. C++ types (from `Utils.hpp`)

gpusim2grid relies on these being stable:

```cpp
using real_type = double;                                   // host real
using cplx_type = std::complex<real_type>;                  // host complex
using RealVect  = Eigen::Matrix<real_type, Eigen::Dynamic, 1>;
using CplxVect  = Eigen::Matrix<cplx_type, Eigen::Dynamic, 1>;
using IntVect   = Eigen::Matrix<int,       Eigen::Dynamic, 1>;
```

`real_type` is `double` on both sides (gpusim2grid casts to `float` internally
when compiled FP32). No host-side precision negotiation is needed.

---

## 3. C++ accessors used by the zero-copy bridge (`LSGrid`)

All of these are read off a **solved** `LSGrid` (after `ac_pf`). gpusim2grid uses
the **`*_solver`-numbered** variants because they are consistent with
`get_Ybus_solver()` (internal AC-solver bus ordering). Verified signatures:

| Data | Accessor | Signature | Status |
|---|---|---|---|
| Admittance matrix | `get_Ybus_solver()` | `Eigen::SparseMatrix<cplx_type>` (by value, copy) | ✅ |
| Power injections | `get_Sbus_solver()` | `Eigen::Ref<const CplxVect>` | ✅ |
| Converged voltage | `get_V_solver()` | `Eigen::Ref<const CplxVect>` | ✅ |
| PV bus indices | `get_pv_solver_numpy()` | `Eigen::Ref<const IntVect>` | ✅ |
| PQ bus indices | `get_pq_solver_numpy()` | `Eigen::Ref<const IntVect>` | ✅ |
| Slack bus indices | `get_slack_ids_solver_numpy()` | `Eigen::Ref<const IntVect>` | ✅ |
| Slack weights | `get_slack_weights_solver()` | `Eigen::Ref<const RealVect>` | ✅ |
| NR Jacobian (skeleton + values) | `get_J_solver()` | `Eigen::Ref<const Eigen::SparseMatrix<real_type>>` | ✅ |
| Nominal bus voltages | `get_bus_vn_kv()` | `Eigen::Ref<const RealVect>` | ✅ |
| System base | `get_sn_mva()` | `real_type` | ✅ |
| Solver↔model bus relabeling | `id_me_to_ac_solver()` / `id_ac_solver_to_me()` | bus-id vectors | ✅ |

### Branch (π-model) admittances — for contingency Ybus patching

From the line / transformer containers. gpusim2grid translates a *branch removal*
into an exact Ybus patch using these effective admittances (never recomputed —
tap reference side, asymmetric shunts, etc. are error-prone):

| Accessor on `get_powerlines_as_data()` / `get_trafos_as_data()` | Signature | Status |
|---|---|---|
| `yac_eff_11()`, `yac_eff_12()`, `yac_eff_21()`, `yac_eff_22()` | `Eigen::Ref<const CplxVect>` | ✅ |
| `get_bus_id_side_1_numpy()`, `get_bus_id_side_2_numpy()` | `Eigen::Ref<const IntVect>` (**global** ids) | ✅ |

> ⚠ **Numbering gotcha.** `yac_eff_*` and the J/Ybus are in *solver* numbering,
> but `get_bus_id_side_*` returns **global** (model) bus ids. When isolated buses
> exist (`id_ac_solver_to_me()` non-empty), the branch endpoints must be
> relabeled to solver numbering via `id_me_to_ac_solver()` before patching Ybus.
> For fully-connected grids the two coincide. gpusim2grid must apply this
> relabeling in the bridge; lightsim2grid already exposes the maps.

### Branch ordering convention

Branches are **lines-then-trafos**: branch index `c < n_lines` is line `c`;
`c >= n_lines` is trafo `c - n_lines`. This matches `add_all_n1()` and is assumed
by gpusim2grid's contingency builder and branch-flow output. Stability of this
ordering is part of the contract.

---

## 4. C++ accessors needed for the optional `GpuNR` plugin (Phase D)

To register gpusim2grid as a lightsim2grid solver (`grid.change_algorithm("GpuNR")`):

| Requirement | API | Status |
|---|---|---|
| Abstract solver base | `ls2g::BaseAlgo` (override `compute_pf`, optionally `get_J`) | ✅ |
| Registration helper | `ls2g::AlgorithmRegistrar(name, factory)` (fires at dlopen) | ✅ |
| Registry | `AlgorithmRegistry::instance()`, `available_algorithm_names()` | ✅ |
| Plugin loader (Python) | `lightsim2grid.load_algorithm_plugin(path)` | ✅ |
| `Custom` algorithm enum value for plugins | `AlgorithmType::Custom` | ✅ |

`compute_pf` receives `(Ybus, V, Sbus, slack_ids, slack_weights, pv, pq, max_iter,
tol)` and must populate `V_`, `Va_`, `Vm_`, `n_`, `nr_iter_`, `err_`. This is a
stable interface; no change required.

---

## 5. Additions still needed upstream (lightsim2grid side)

| # | Need | Severity | Detail |
|---|---|---|---|
| 0 | **`lightsim2grid_core` exports `-DKLU_SOLVER_AVAILABLE` but does not ship the SuiteSparse headers it then requires** | **blocking for any consumer that includes `LSGrid.hpp`** | The installed CMake target sets `INTERFACE_COMPILE_DEFINITIONS "KLU_SOLVER_AVAILABLE"`, so `LSGrid.hpp` → `linear_solvers/KLUSolver.hpp` does `#include "cs.h"` / `"klu.h"`. But `INTERFACE_INCLUDE_DIRECTORIES` only contains `lightsim2grid/include` — the SuiteSparse headers (`cs.h`, `klu.h`, `amd.h`, `colamd.h`, `btf.h`, `SuiteSparse_config.h`) are **not** installed. Result: `fatal error: cs.h: No such file or directory`. The `external_algorithm` example never hits this because it only includes `AlgorithmRegistry.hpp` + `BaseAlgo.hpp`. **Fix options (pick one):** (a) install the SuiteSparse public headers into the package and add their dir to `INTERFACE_INCLUDE_DIRECTORIES`; (b) keep the linear-solver headers out of `LSGrid.hpp`'s public include surface (PIMPL / forward-declare) so consumers don't transitively need them; (c) document that consumers must supply the SuiteSparse include dirs and the exact set required. **Current gpusim2grid workaround:** point at the vendored `_lightsim2grid/SuiteSparse` submodule's `Include` dirs (see `src/_cpp/CMakeLists.txt`, `GPUSIM2GRID_SUITESPARSE_DIR`). This must compile *with* `KLU_SOLVER_AVAILABLE` to keep the `LSGrid` ABI identical to the loaded `lightsim2grid_cpp` — undefining it would change member layout and corrupt the pybind cross-cast. |
| 1 | **KLU fill-reducing permutation getter** | needed for Phase F | Add `KLULinearSolver::get_permutation() -> IntVect` (and surface it on `LSGrid`, e.g. `get_solver_permutation()`). gpusim2grid would feed it to cuDSS via `CUDSS_ALG_USER` + user perm, skipping the cuDSS ANALYSIS phase. **Not present today** (`KLUSolver.hpp` exposes no permutation). |
| 2 | *(nice-to-have)* Guarantee `get_J_solver()` row/col ordering is exactly `[pvpq | pq]` and documented | Phase C | gpusim2grid reuses the J skeleton directly; an explicit ordering guarantee (or the `get_*_to_J_col` maps on the active algo) avoids re-deriving it from Ybus. The maps exist on `BaseAlgo` (`get_theta_to_J_col_python`, `get_vm_to_J_col_python`, `get_q_to_J_col_python`); confirming they are populated for `NR_KLU`/`NR_SparseLU` is enough. |
| 3 | *(nice-to-have)* Keep `get_Ybus_solver()` returning CSR with a stable sparsity pattern across an N-k batch | Phase B/C | The pattern is the batch invariant gpusim2grid uploads once. Already true in practice; worth stating as a contract. |
| 4 | **Augmented-Jacobian ledger ROW maps on `LSGrid`** | needed for the augmented-NR integration (distributed slack / HVDC / SVC / remote voltage control) | gpusim2grid rebuilds the GPU dS scatter + residual layout of the *augmented* J from the ledger. The column maps already exist; the row counterpart was missing. **Added** (this work): `NRLedger::p_row_of_bus()/q_row_of_bus()` → `NRSystem::p_to_J_row()/q_to_J_row()` → `BaseAlgo::get_p_to_J_row_python()/get_q_to_J_row_python()` (throwing defaults; overridden in `NRAlgo`) → `LSGrid::get_p_to_J_row_solver()/get_q_to_J_row_solver()` (plus `get_{theta,vm,q}_to_J_col_solver()` surfaced uniformly). All in solver bus numbering, valid after an NR solve. |
| 5 | **Per-extension feature-entry tables + solver data on `LSGrid`** | needed for augmented-NR Phases 2–4 | To recompute feature values / mismatch adjustments per GPU iteration, expose (in solver numbering, after a solve): MultiSlack `slack_col` + per-slack P-row + feature J positions + free-Vm slack set; `HvdcDroopSolverData` + per-line feature J positions + end `p_row`/`theta_col`; `VoltageControlSolverData` + `q_cols`/`q_rows`/`v_rows`/`vm_cols`/`share_rows` + resolved feature positions. Added incrementally with each phase. |

Apart from #0 (worked around) and #1 (Phase F only), everything gpusim2grid needs
already exists on `n1_full_compute`. **The zero-copy bridge is implemented and
working** against this lightsim2grid: a Python `LSGrid` is cross-cast to
`ls2g::LSGrid&` inside the gpusim2grid pybind module and Ybus/Sbus/V/pv/pq/slack
plus branch admittances are read straight off the C++ object.

### Runtime linkage note

`_gpusim2grid.so` ends up with a `NEEDED liblightsim2grid_core.so`. gpusim2grid
finds it two ways: an `$ORIGIN/../lightsim2grid` RUNPATH (both packages sit under
`site-packages/`), and by importing `lightsim2grid` first in
`gpusim2grid/__init__.py` (so the soname is already resolved in-process). For
this to work the gpusim2grid extension and the loaded `lightsim2grid_cpp` must be
the **same** lightsim2grid build (matching ABI) — see §1.

---

## 6. Python API requirements (extraction fallback path)

When the C++ bridge is not used (or for the single-solve `AcPfGPU`), gpusim2grid
extracts the same arrays through the Python bindings. Required:

| Purpose | Python API |
|---|---|
| pandapower → grid | `lightsim2grid.network.init_from_pandapower(net)` |
| select solver | `grid.change_algorithm("NR_KLU")` (string) or `change_algorithm(AlgorithmType.NR_KLU)` |
| list solvers | `grid.available_solver_names()` |
| base solve | `grid.dc_pf(Vinit, max_iter, tol)`, `grid.ac_pf(Vinit, max_iter, tol)` |
| arrays | `get_Ybus_solver()` (scipy CSR), `get_Sbus_solver()`, `get_pv()`, `get_pq()`, `get_slack_ids()` |
| branches | `get_lines()` / `get_trafos()` → `.get_yac_eff_11/12/21/22()`, `.get_bus_id_side_1/2()` |
| scalars | `get_bus_vn_kv()`, `get_sn_mva()` |

> Note the **API rename** on `n1_full_compute`: `lightsim2grid.gridmodel` is
> deprecated in favour of `lightsim2grid.network`; the grid class is `LSGrid`
> (was `GridModel`); solver selection is `change_algorithm(...)` with registered
> names like `"NR_KLU"` (the old `change_solver(SolverType.KLU)` enum path no
> longer accepts `SolverType` directly). gpusim2grid's `_ls2g_utils` handles both
> old and new forms.

---

## 7. Version pinning

- Minimum lightsim2grid: the first release containing `get_cmake_dir()`,
  `lightsim2grid_core` CMake package, `LSGrid`, and `lightsim2grid.network`
  (i.e. the `n1_full_compute` line → `>= 0.13.2`).
- gpusim2grid should declare `lightsim2grid >= 0.13.2` in `pyproject.toml` and,
  for source builds, document the `LIGHTSIM2GRID_CMAKE_DIR` invocation above.
