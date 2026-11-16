# Where a batched Newton call spends its time, and what to change

**Design note, not documentation.** Written on 2026-11-16, following up the parked GPU benchmark
(branch `gpu_benchmark_suite`, `docs/dev_notes/gpu_benchmark_suite.md` there), which found
gpusim2grid behind p3s's `lean` backend. Measurements on the RTX A1000 6 GB laptop GPU, FP64 unless
stated, case2869_pegase (2 869 buses, nnz(Ybus) 10 805, dim J 5 228, nnz(J) 36 602), the suite's own
time series and N-1 rows, `batch_size=512`, `linear_solver='own_lu'`, `use_distributed_slack=False`,
tol 1e-8 pu. Laptop numbers: the ratios matter, not the absolute speed.

## 1. Two readings of the benchmark that were wrong

- **"Chunked is ~10x slower than continuous"** came from the benchmark note's table, which put
  *cuDSS* chunked next to *own LU* continuous. With the same solver (results JSON of the suite):

  | case, workload | own LU chunked (5 it.) | own LU continuous (3.1 it. mean) | cuDSS chunked | cuDSS continuous |
  |---|---|---|---|---|
  | 1354 time series 10k | 1 805 | 1 345 | 12 639 | 8 305 |
  | 2869 time series 10k | 4 179 | 3 015 | 23 704 | 15 092 |
  | 2869 N-1 3 779 | 1 576 | 1 135 | 8 946 | 5 591 |

  Chunked costs the same per Newton iteration (slightly less); it loses only by running a fixed
  `nb_iter`. On case2869 time series (3 779 rows) 97 % of the rows converge in 3 iterations, the rest
  in 4: `nb_iter` 5 / 4 / 3 = 1 592 / 1 280 / 976 ms against 1 175 ms continuous; 3 leaves ~3 % of
  the rows unconverged. Chunked can only win when the rows need the same number of iterations.
- **"Time series is where we lose to p3s"** compared absolute times of 10 000 and 3 779 rows. Per row
  (µs, own LU continuous vs p3s lean): 2869 time series 301 vs 151, 2869 N-1 300 vs 275; 1354
  time series 134 vs 61, N-1 167 vs 160. gpusim2grid costs the same per row on both workloads; p3s is
  ~2x cheaper on time series than on its own N-1. On N-1 we are level only because p3s is slower
  there.

The per-slot copy of Ybus an injection sweep keeps (`InjectionBatch` tiles it once into
`d_Ybus_values_batch`, p3s passes `y_shared=1`) is not the cause: the whole SpMV is ~7 % of a call.

## 2. The profile (nsys, one call, time series continuous, 996 ms of kernels)

| group | kernels | ms | share |
|---|---|---|---|
| J / F / V assembly (ours) | `fill_J` 269, `update_Va`+`update_Vm` 108, cuSPARSE SpMV 72, `fill_FP`/`fill_FQ` 58, misc ~25 | ~535 | 54 % |
| own LU layout glue | `lu_tr_in` 64, `lu_scaled_tol` 27, `lu_tr_out` 8 | 99 | 10 % |
| LU factor + solves | `lu_left_*` 255, `lu_fwd`/`lu_bwd` 109 | 364 | 37 % |

N-1 gives the same picture (`fill_J` 262, V update 106). The library's own timers (`timings.to_dict()`)
agree with nsys. Outside the kernels: an injection sweep rebuilds its whole host Sbus at every
`compute()` (52 ms, `injection_sweep_session.cu`, `h_Sbus_all`) and re-uploads it (17 ms) even when
the injections did not change; the D→H copy of V takes 55 ms (~3 GB/s) on both workloads.

`fill_J` costs 10.8 ms per round of 512 slots, ~21 µs per matrix, against 1.3 ms for the SpMV over
the same Ybus: one thread per (slot, Ybus entry), a 64-bit integer div/mod to find both, a 12-step
binary search for the row, `|V_j|` (hypot) and two divisions for `V/|V|`, four scattered writes.

## 3. FP64 arithmetic or memory? (FP32 rebuild)

ncu could not run (`ERR_NVGPUCTRPERM`: needs root or `NVreg_RestrictProfilingPermission=0`), so the
test was a FP32 build of the same source, profiled on the same rows with chunked `nb_iter=4` (same
launches one for one; FP32 solves its own base case, `init_from_n_powerflow=False`, its 1e-4 check
rejects the FP64 voltages). This GPU measured: DRAM copy 123 GB/s, DGEMM 119 GFLOP/s, SGEMM 3 056
GFLOP/s. A kernel limited by memory should gain ~2x, one limited by FP64 arithmetic much more.

| kernel | FP64 ms | FP32 ms | ratio | limited by |
|---|---|---|---|---|
| `fill_J` | 347.5 | 34.5 | 10.1 | FP64 arithmetic |
| `update_Va` | 76.2 | 7.0 | 10.9 | FP64 arithmetic |
| `update_Vm` | 63.0 | 6.8 | 9.3 | FP64 arithmetic |
| LU factor (`lu_left_*`) | 332.6 | 179.6 | 1.9 | memory / latency |
| LU solves | 142.2 | 89.1 | 1.6 | memory / latency |
| `lu_tr_in` | 92.7 | 54.6 | 1.7 | memory |
| SpMV | 51.3 | 28.5 | 1.8 | memory |
| `lu_scaled_tol` | 35.3 | 18.4 | 1.9 | memory |
| `fill_FP` + `fill_FQ` | 47.0 | 24.0 | 2.0 | memory |
| all kernels | 1 213 | 456 | 2.7 | |

In FP32 `fill_J` and the V update sit at the memory bound (e.g. `update_Va` 0.22 ms per launch
against ~0.24 ms of traffic). The LU is not FP64-bound. On an A100/H100 (FP64 at half rate) these
kernels would be memory-bound: the rewrite of their arithmetic matters most on consumer GPUs, the
layout work (below) everywhere.

## 4. How p3s does it

`p3s/cuda/nr_polar_kernels.cu` (`eval_FJ`, `apply_step`), FP64 too:

- V in **polar form** per system (`Vm`, `cs`, `sn`): no transcendental, division or hypot per
  entry; `apply_step` updates Va/Vm and recomputes `sincos` once per bus.
- **One thread per (row, system), the system index fastest**: a warp is 32 systems on one row, Ybus
  is read once for all (broadcast when shared), V reads and J writes are contiguous.
- J written **directly in the LU's interleaved layout** (`J[pos * S + c]`): no transpose.
- **One loop** gives the injection, the mismatch and J: no SpMV, no separate `fill_F`, no residual
  pass.

Our assembly + glue (~630 ms) costs about p3s's whole call on the same rows (~570 ms). Even with a
free assembly our LU alone (364 ms) would still be slightly behind: the LU is a later step.

## 5. Step 1 (done, branch `perf_v_update`): the V update as a product

The step is applied to the complex V: `Va += dva` ⇔ `V *= exp(j·dva)` (one `sincos` of the small
step), `Vm += dvm` ⇔ `V *= 1 + dvm/|V|` (one `rsqrt` of |V|²). No new buffer, no change to the call
sites or to anything else reading V; a zero step leaves V bit-identical. Fully polar state is left
for step 2, where `fill_J` would stop paying its own hypot and divisions.

- Kernels 139 → 34 ms (memory bound ~28 ms; the rest is `update_Va`'s FP64 `sincos`), a whole call
  −7 % (time series continuous 1 167 → 1 088 ms, N-1 1 128 → 1 054 ms).
- Against the previous build on four configurations (own LU / cuDSS × time series / N-1 × chunked /
  continuous): max |dV| 5.8e-13 on converged rows, identical iteration counts and NaN pattern (one
  N-1 row diverges both ways). Full suite green.

## 6. Step 2, first part (done, branch `perf_v_update`): `fill_J`'s arithmetic

Kept the slot-major output and the signature; three changes, each measured (chunked `nb_iter=4`,
`fill_J` alone, 347 ms before):

- **Fewer FP64 operations**: with `t = V_i·conj(Y_ij·V_j)` and `inv = 1/|V|`, dS/dVa = −j·t and
  dS/dVm = t·inv_j off the diagonal; the diagonal adds `c = V_i·conj(I_i)` (dS/dVa += j·c,
  dS/dVm += c·inv_i). 2 complex products and one `rsqrt` per entry instead of 4, a `hypot` and two
  divisions. Plus a 2D grid (entries on x, slots on y, grid-stride past 65 535): no 64-bit div/mod
  per thread, and an entry feeding no J position leaves at once. → 131 ms.
- **The diagonal as an added term, not a second path**: a warp holds ~8 diagonal entries out of 32
  and ran both full branches serially. → 76 ms, at the FP64 memory bound (~65-70 ms: 2x the FP32
  time).
- Tried and dropped: a precomputed row index per Ybus entry instead of the binary search (75.6 vs
  76.5 ms). A first experiment pointed at the search, but it had changed the diagonal pattern too:
  divergence was the cost.

Same checks as step 1 (max |dV| 6.3e-13 against the original build, identical iteration counts and
NaN pattern, full suite green).

**Where a call stands** (time series continuous, 3 779 rows): whole call 1 167 → 894 ms; kernels
996 → 711 ms: LU factor 257 + solves 111 (52 %), SpMV 77, `fill_F` 58, `fill_J` 58, V update 27,
residual 9, `lu_tr_in` 64, `lu_scaled_tol` 27, `lu_tr_out` 8.

## 7. Next steps

- **One assembly kernel in the style of `eval_FJ`**, replacing `fill_J`, `fill_FP`/`fill_FQ` and the
  SpMV: system index fastest, the patched or shared Ybus through a stride (stride 0 for an
  injection sweep, like `sbus_stride`), J written in own LU's interleaved layout (which removes
  `lu_tr_in` for J and lets `lu_scaled_tol` read J where it is written). The design cost is
  everything else that writes J or F by slot-major position: the feature stamps (MultiSlack, HVDC
  droop, VoltageControl, BranchControl / ShuntControl, outer-loop pins and holds), the masks
  (`nr_apply_J_masks` / `nr_apply_F_masks`), `keep_final_jacobian` and the adjoint, which read J
  slot-major. cuDSS keeps the slot-major path.
- **Skip the Sbus rebuild** of an injection sweep when its injections did not change (69 ms a call).
- **Cheaper, independent of the J layout** (F side only; features and masks write F after `fill_F`,
  so nothing else moves):
  - *SpMV + `fill_F` as one kernel* (135 ms of a call): the injection, `Ibus` (still read by
    `fill_J`'s diagonal) and the P/Q mismatch in one pass, Ybus through a stride. cuSPARSE runs at the
    memory bound on the block-diagonal matrix (88 MB of values + 22 MB of column indices per round);
    a shared Ybus (time series) leaves mostly V / Ibus / Sbus traffic, a per-slot one at least drops
    the block-diagonal indices and the second read of V and Ibus by `fill_F`.
  - *The continuous residual pass computes F twice*: `_residual_pass` runs SpMV, `fill_F`, the
    feature mismatch and the F masks at V_k, and the next round's Newton step runs the same sequence
    at the same V_k for every slot that was not reloaded (or re-controlled by the outer loops):
    ~half of the SpMV + `fill_F` time in continuous mode. Reusing F needs either a per-slot dirty
    flag in those kernels or recomputing only on rounds that reloaded a slot.

Then the LU itself (fusing factor and solve, interleaved storage end to end).

## Appendix: reproducing the profile

The driver used the suite's grid and scenario code (`git archive gpu_benchmark_suite
benchmarks/suite | tar -x -C <dir>`, cache in `~/.cache/gpusim2grid-bench`), built one
`InjectionSweepGPU` / `ContingencyAnalysisGPU` with `benchmarks.suite.g2g_runner.facade_kwargs`, made
one untimed `compute()`, then three inside an NVTX range `PROFILE`:

```bash
nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture=PROFILE \
     --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range-end=stop python driver.py ...
nsys stats -r cuda_gpu_kern_sum -f csv report.nsys-rep
```

A second build (e.g. FP32) can run beside the editable install: build a wheel with
`CUDA_REAL_FLOAT=1 uv build --wheel --no-build-isolation -C build-dir=<dir>`, unzip it, and before
importing `gpusim2grid` drop the `_gpusim2grid_editable` finder from `sys.meta_path` and put the
unzipped directory first on `sys.path`.
