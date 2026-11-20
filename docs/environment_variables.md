# Environment variables

None of these is needed for normal use: every option of a run is a constructor argument or an
attribute of the facades. The variables below select what a build compiles, or turn on
diagnostics and check modes. A variable is on when it is set to anything but empty or `0`, unless
its line says otherwise.

## Build time

Read by `pip install` (the in-tree build backend and CMake), not at runtime. Details in
[Installation](installation.rst).

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_CUDA=12` / `13` | The CUDA major to build for. Unset: the major of the `nvcc` that `CUDACXX` / `CUDAToolkit_ROOT` names, else CUDA 13 from PyPI. |
| `GPUSIM2GRID_CUDSS_FROM_PYPI=0` | Take cuDSS from `cudss_ROOT` / `cudss_DIR` instead of the `nvidia-cudss-cuXX` wheel. |
| `CUDA_REAL_FLOAT=1` / `CUDA_REAL_DOUBLE=1` | FP32 / FP64 build (FP64 by default). Query it at runtime with `gpusim2grid._gpusim2grid.is_fp32`. |

## At import

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_SKIP_BUILD_CHECKS=1` | Skip `import gpusim2grid`'s checks: the loaded cuDSS / cuSPARSE built for another CUDA major, a too old driver, another lightsim2grid version than the build's. |

## Performance

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_ASYNC_TIMERS=1` | The per-phase timers stop waiting for the GPU after each timed phase, so the host keeps the GPU fed. Their `gpu_ms` then reads 0 and `wall_ms` is the host's enqueue time; no result changes. Read once per process. |

## `linear_solver="own_lu"`

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_OWN_LU_EPS=<x>` | The relative pivot clamp: a pivot below `x · max\|J\|` is clamped and counted in `cudss_npivots_*`. Default `1e-12` (FP64), `1e-5` (FP32); `0` = no clamp. |
| `GPUSIM2GRID_OWN_LU_FUSE=0` | Every level of the triangular solves gets its own launch instead of fusing the runs of narrow levels into one kernel (same bits, more launches). |
| `GPUSIM2GRID_OWN_LU_STATS=1` | Diagnostic: prints the widths of the factorization and solve launches (how many are narrow, split or fusable). |

## Outer loops (`outer_loops=True`)

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_OC_PROFILE=1` | Diagnostic: wall time and call count of each part of the outer-loop model (host loads of rows, decisions, checks), printed on stderr when the model is destroyed. With `outer_loops_backend="device"` also the parts of a row loaded through the host model (load, image, fill, upload). |
| `GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE=1` | Check mode, `outer_loops_backend="device"`: every row the device would copy from a template (a plain row, a strategy's point, a masked row, an empty slot) is also loaded through the host model and compared field by field; a difference throws. Slow: every row goes through the host. |
| `GPUSIM2GRID_OUTER_CHECK_RL_PLAN=1` | Check mode, both backends: every filtered ReactiveLimits plan and DistributedSlack unit list a row gets is compared with a full rebuild (`build_bus_q_plan` / `collect_units`); a difference throws. |

## Linear solver diagnostics

| Variable | Effect |
|---|---|
| `GPUSIM2GRID_DUMP_J=<dir>` | Every batch solver writes its J skeleton, its orderings (the user permutation, and cuDSS's own after ANALYSIS) and the J values of its first factorizations to `<dir>` as raw little-endian files (`h<k>_*.i32` / `.f64`, `.f32` on an FP32 build, plus `h<k>_meta.txt`). See [the batched LU note](dev_notes/batched_lu_and_constrained_ordering.md). |
| `GPUSIM2GRID_DUMP_J_MAX=<n>` | How many factorizations `GPUSIM2GRID_DUMP_J` writes per solver (default 8). |
| `GPUSIM2GRID_ND_NO_POSTMOVE=1` | Diagnostic, `reordering_alg="ls2g_constrained_nd"`: hand cuDSS's own order back unchanged, without the post-move by lightsim2grid's ordering constraints. |
