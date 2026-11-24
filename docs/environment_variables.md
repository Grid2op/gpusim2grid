# Environment variables

None of these is needed for normal use: every option of a run is a constructor argument or an
attribute of the facades. The variables below select what a build compiles, or turn on
diagnostics and check modes. **Default** is the value used when the variable is not set.

## Build time

Read by `pip install` (the in-tree build backend and CMake), not at runtime. Details in
[Installation](installation.rst).

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_CUDA` | `12`, `13` | the CUDA major of the `nvcc` that `CUDACXX` / `CUDAToolkit_ROOT` names, else `13` (PyPI's toolkit) | The CUDA major to build for. Any other value, or one contradicting the named `nvcc`, is an error. |
| `GPUSIM2GRID_CUDSS_FROM_PYPI` | `0`, `1` | `1` | `1`: cuDSS from the `nvidia-cudss-cuXX` wheel. `0`: from `cudss_ROOT` / `cudss_DIR`. |
| `CUDA_REAL_FLOAT` | not set, or set (any value, `0` included) | not set | Set: FP32 build. Wins over `CUDA_REAL_DOUBLE` and `-DUSE_FLOAT_PRECISION`. |
| `CUDA_REAL_DOUBLE` | not set, or set (any value) | not set | Set: FP64 build (already the default when neither is set). Query the precision at runtime with `gpusim2grid._gpusim2grid.is_fp32`. |

## At import

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_SKIP_BUILD_CHECKS` | `0`, `1` | `0` | `1`: skip `import gpusim2grid`'s checks (the loaded cuDSS / cuSPARSE built for another CUDA major, a too old driver, another lightsim2grid version than the build's). |

## Performance

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_ASYNC_TIMERS` | `0`, `1` | `0` | `0`: every timed phase waits for the GPU (`gpu_ms` measured). `1`: the per-phase timers stop waiting, so the host keeps the GPU fed; their `gpu_ms` reads 0 and `wall_ms` is the host's enqueue time. No result changes. Read once per process. |

## `linear_solver="own_lu"`

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_OWN_LU_EPS` | a number `≥ 0` | `1e-12` (FP64 build), `1e-5` (FP32 build) | The relative pivot clamp: a pivot below `x · max\|J\|` is clamped and counted in `cudss_npivots_*`. `0`: no clamp. |
| `GPUSIM2GRID_OWN_LU_FUSE` | `0`, `1` | `1` | `1`: the runs of narrow levels of the triangular solves fused into one kernel each (when the device supports it). `0`: every level its own launch (same bits, more launches). |
| `GPUSIM2GRID_OWN_LU_STATS` | `0`, `1` | `0` | `1`: prints the widths of the factorization and solve launches (how many are narrow, split or fusable). |
| `GPUSIM2GRID_OWN_LU_TOP` | `0`, `1` | `0` | Prototype. `1`: `linear_solver='own_lu'` uses `top_lu::TopLu` (`src/_cpp/fused_lu/`, built when present): the last elimination levels factored and solved in shared memory, one block per matrix. `0`: own_lu unchanged. |
| `GPUSIM2GRID_TOP_LU_M` | an integer `≥ 0` | `0` | With `GPUSIM2GRID_OWN_LU_TOP=1`: the largest trailing block (pivots). `0`: as large as one block's shared memory holds. |
| `GPUSIM2GRID_TOP_LU_M_LARGE` | an integer `≥ 0` | `104` | With `GPUSIM2GRID_OWN_LU_TOP=1`: the trailing block used above `GPUSIM2GRID_TOP_LU_NB` active matrices (a large one costs more than the launches it replaces there). |
| `GPUSIM2GRID_TOP_LU_NB` | an integer | `768` | With `GPUSIM2GRID_OWN_LU_TOP=1`: up to this many active matrices the factorization takes the large trailing block, above it the `GPUSIM2GRID_TOP_LU_M_LARGE` one. |
| `GPUSIM2GRID_TOP_LU_FUSE_NB` | an integer | `0` | With `GPUSIM2GRID_OWN_LU_TOP=1`: the runs of narrow solve levels are fused up to this many active matrices (issue #44: one launch per level is faster above). |
| `GPUSIM2GRID_TOP_LU_PDL` | `0`, `1` | `1` | With `GPUSIM2GRID_OWN_LU_TOP=1`: every launch with programmatic dependent launch (a kernel starts before its predecessor ends and waits in-kernel for its results). `0`: plain launches, the same results. |
| `GPUSIM2GRID_TOP_LU_STATS` | `0`, `1` | `0` | With `GPUSIM2GRID_OWN_LU_TOP=1`: prints the launches left outside the trailing block (count, chain, split per level). |

## Outer loops (`outer_loops=True`)

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_OC_PROFILE` | `0`, `1` | `0` | `1`: wall time and call count of each part of the outer-loop model (host loads of rows, decisions, checks), printed on stderr when the model is destroyed. With `outer_loops_backend="device"` also the parts of a row loaded through the host model (load, image, fill, upload) and why it was. |
| `GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE` | `0`, `1` | `0` | `1`, check mode, `outer_loops_backend="device"`: every row the device would copy from a template (a plain row, a strategy's point, a masked row, a row sharing a lost power, an empty slot) is also loaded through the host model and compared field by field; a difference throws. Slow: every row goes through the host. |
| `GPUSIM2GRID_OUTER_CHECK_RL_PLAN` | `0`, `1` | `0` | `1`, check mode, both backends: every filtered ReactiveLimits plan and DistributedSlack unit list a row gets is compared with a full rebuild (`build_bus_q_plan` / `collect_units`); a difference throws. |

## Linear solver diagnostics

| Variable | Values | Default | Effect |
|---|---|---|---|
| `GPUSIM2GRID_DUMP_J` | a directory | none (nothing written) | Every batch solver writes its J skeleton, its orderings (the user permutation, and cuDSS's own after ANALYSIS) and the J values of its first factorizations there, as raw little-endian files (`h<k>_*.i32` / `.f64`, `.f32` on an FP32 build, plus `h<k>_meta.txt`). See [the batched LU note](dev_notes/batched_lu_and_constrained_ordering.md). |
| `GPUSIM2GRID_DUMP_J_MAX` | an integer `≥ 0` | `8` | How many factorizations `GPUSIM2GRID_DUMP_J` writes per solver. |
| `GPUSIM2GRID_ND_NO_POSTMOVE` | `0`, `1` | `0` | `1`, `reordering_alg="ls2g_constrained_nd"`: cuDSS's own order handed back unchanged. `0`: post-moved by lightsim2grid's ordering constraints. |
