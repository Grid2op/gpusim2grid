# Environment variables

None of these is needed for normal use: every option of a run is a constructor argument or an
attribute of the facades. The variables below select what a build compiles, or turn on
diagnostics and check modes. A variable is on when it is set to anything but empty or `0`, unless
its line says otherwise. **Default** is the behaviour when the variable is unset.

## Build time

Read by `pip install` (the in-tree build backend and CMake), not at runtime. Details in
[Installation](installation.rst).

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_CUDA=12` / `13` | unset: the major of the `nvcc` that `CUDACXX` / `CUDAToolkit_ROOT` names, else CUDA 13 from PyPI | The CUDA major to build for. Any other value is an error, and so is one that contradicts the named `nvcc`. |
| `GPUSIM2GRID_CUDSS_FROM_PYPI=0` | `1`: cuDSS from the `nvidia-cudss-cuXX` wheel | `0` takes cuDSS from `cudss_ROOT` / `cudss_DIR` instead. Only `0` turns it off. |
| `CUDA_REAL_FLOAT` / `CUDA_REAL_DOUBLE` | unset: FP64 (CMake's `USE_FLOAT_PRECISION=OFF`) | FP32 / FP64 build. Tested for being **set**, whatever its value (`CUDA_REAL_FLOAT=0` still builds FP32); `CUDA_REAL_FLOAT` wins over `CUDA_REAL_DOUBLE`, both over `-DUSE_FLOAT_PRECISION`. Query it at runtime with `gpusim2grid._gpusim2grid.is_fp32`. |

## At import

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_SKIP_BUILD_CHECKS=1` | off: the checks run | Skip `import gpusim2grid`'s checks: the loaded cuDSS / cuSPARSE built for another CUDA major, a too old driver, another lightsim2grid version than the build's. Only `1` turns it on. |

## Performance

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_ASYNC_TIMERS=1` | off: every timed phase waits for the GPU (`gpu_ms` measured) | The per-phase timers stop waiting for the GPU after each timed phase, so the host keeps the GPU fed. Their `gpu_ms` then reads 0 and `wall_ms` is the host's enqueue time; no result changes. Read once per process. |

## `linear_solver="own_lu"`

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_OWN_LU_EPS=<x>` | `1e-12` (FP64), `1e-5` (FP32) | The relative pivot clamp: a pivot below `x · max\|J\|` is clamped and counted in `cudss_npivots_*`; `0` = no clamp. |
| `GPUSIM2GRID_OWN_LU_FUSE=0` | on: the runs of narrow levels of the triangular solves fused into one kernel each (when the device supports the fused kernel) | `0` gives every level its own launch (same bits, more launches). Only a value starting with `0` turns it off. |
| `GPUSIM2GRID_OWN_LU_STATS=1` | off | Diagnostic: prints the widths of the factorization and solve launches (how many are narrow, split or fusable). |

## Outer loops (`outer_loops=True`)

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_OC_PROFILE=1` | off | Diagnostic: wall time and call count of each part of the outer-loop model (host loads of rows, decisions, checks), printed on stderr when the model is destroyed. With `outer_loops_backend="device"` also the parts of a row loaded through the host model (load, image, fill, upload) and why it was. |
| `GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE=1` | off | Check mode, `outer_loops_backend="device"`: every row the device would copy from a template (a plain row, a strategy's point, a masked row, a row sharing a lost power, an empty slot) is also loaded through the host model and compared field by field; a difference throws. Slow: every row goes through the host. |
| `GPUSIM2GRID_OUTER_CHECK_RL_PLAN=1` | off | Check mode, both backends: every filtered ReactiveLimits plan and DistributedSlack unit list a row gets is compared with a full rebuild (`build_bus_q_plan` / `collect_units`); a difference throws. |

## Linear solver diagnostics

| Variable | Default | Effect |
|---|---|---|
| `GPUSIM2GRID_DUMP_J=<dir>` | unset or empty: nothing written | Every batch solver writes its J skeleton, its orderings (the user permutation, and cuDSS's own after ANALYSIS) and the J values of its first factorizations to `<dir>` as raw little-endian files (`h<k>_*.i32` / `.f64`, `.f32` on an FP32 build, plus `h<k>_meta.txt`). See [the batched LU note](dev_notes/batched_lu_and_constrained_ordering.md). |
| `GPUSIM2GRID_DUMP_J_MAX=<n>` | `8` | How many factorizations `GPUSIM2GRID_DUMP_J` writes per solver. |
| `GPUSIM2GRID_ND_NO_POSTMOVE=1` | off: lightsim2grid's ordering constraints applied to cuDSS's order | Diagnostic, `reordering_alg="ls2g_constrained_nd"`: hand cuDSS's own order back unchanged, without the post-move by lightsim2grid's ordering constraints. |
