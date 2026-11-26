# N-1 with outer loops: what is left in the large rounds

**Design note, not documentation.** Written on 2026-11-26 while reviewing PR #46 (own_lu_top,
device outer loops). Not implemented: two ideas parked behind another one to check first.

Setting: `bench_n1.sh` on PtFige-20260911-1230 (8 803 contingencies, loose convergence),
`linear_solver='own_lu'` with `GPUSIM2GRID_OWN_LU_TOP=1`, `outer_loops_backend='device'`,
`batch_size=4096`, tail 1 x 4096, RTX PRO 6000 Blackwell (FP64 at 1/64 of the FP32 rate). LU nnz
140 628, nnz(J) 97 670, dim J 14 364. Session run 0.357 s after PR #46 and the follow-ups of branch
`perf/pr46-followups` (no pinned gather buffer with the device model, J zeroed only where `fill_J`
does not write, own_lu plan + TopLu symbolic part cached per Jacobian pattern).

## Where the time is

The 4 large rounds (4 096 live slots) are 0.237 s of the 0.335 s driver solve, ~59 ms each, the GPU
busy all along (nsys: busy = span). One of them:

| kernels | ms |
|---|---|
| own_lu factorization below the trailing block (`lu_left_u` / `_l` / `_split`) | ~15 |
| J slot-major -> interleaved (`lu_tr_in_vals`) | 4.5 |
| triangular solves outside T (`lu_fwd` / `_bwd` / `_split`), twice (ir_n_steps=1) | ~12 |
| trailing block (`top_factor`, `top_fwd` / `top_bwd`, `top_outside`, `top_fo`) | ~3.5 |
| refinement residual (`lu_residual_i`) + layout transposes of b / x | ~5 |
| `spmv_fill_F_kernel`, 3 passes (residual pass, check before the first step, Newton) | 7.3 |
| `fill_J_kernel` | 4.1 |
| outer model (`on_solved`, `share_lost`, ...), V update, residual norms, gathers | ~7 |

All of it is bandwidth-bound at this batch: the LU factors alone are 140 628 x 4 096 x 8 B = 4.6 GB,
each slot's Ybus ~0.9 MB (x 4 096 = 3.8 GB per SpMV pass), J 3.2 GB. No cheap win is left there;
what follows changes what is read, not how.

## 1. Shared base Ybus + per-slot patches in `spmv_fill_F` / `fill_J` (estimate: -40 ms)

The biggest remaining lever, and a large refactor. A contingency slot's Ybus differs from the base
one only at the entries of its tripped branches (a few per row), yet `spmv_fill_F_kernel` and
`fill_J_kernel` read the slot's full tiled copy (`d_Ybus_values_batch`, stride nnz_Y). Reading the
base values once (L2-resident) and correcting afterwards would make both V-bound:

- SpMV + F: run on the base values (stride 0, as `ybus_shared` already does for an injection
  sweep), then a correction kernel over the slot's patch entries (`I_i += dY_ij V_j`) and a
  recomputation of F at the buses they touch. Mind `d_F_raw` / `d_F_clean` (the F cache of the
  continuous schedule).
- `fill_J`: fill from the base values, then recompute the J entries of the patched Ybus entries and
  the diagonal of their rows from the patched values, before the feature stamps (HVDC adds onto
  them) and the masks.
- Keep the tiled per-slot Ybus for every other reader (flows, gen_v adjoint, the adjoint rebuild,
  violation checks); only the two hot kernels change.
- Not with BranchControl / ShuntControl (`n_bc` / `n_sc` > 0: they patch Ybus by value per slot):
  stay on the per-slot path there, as `ybus_shared` does.
- The patch list must follow the slots: loads, outer-loop re-solves and `_compact_tail` moves (a new
  per-slot buffer must be added there).

Estimate: spmv_fill_F 37 ms + fill_J 21 ms per run today, mostly the per-slot Ybus reads.

## 2. `fill_J` writing J in the LU's interleaved layout (estimate: -20 ms)

`lu_tr_in_vals` transposes J from slot-major (`J[b * nnz_J + p]`) into own_lu's interleaved layout
(`A[p * B + b]`) at every factorization: 23 ms per run, pure bandwidth. `fill_J_kernel` could write
the interleaved layout directly (its writes are scattered either way: neighbouring threads are
neighbouring Ybus entries, not slots), but every kernel that reads or writes J would change with it:
the feature stamps (slack, HVDC, VoltageControl, BranchControl / ShuntControl), the masks and J
overrides, the outer-loop pins and holds, `zero_J_positions_kernel`, the clamp's max |J|, the
slot-major refinement residual, the adjoint (J -> J^T gather, `_solve_JT_rebuilt`), the J dumps and
`j_values_dlpack`. A stride pair (row stride, slot stride) passed to every one of them would keep
both layouts available (cuDSS needs slot-major).

## 3. Memory- or compute-bound: Nsight Compute on one large round

`sudo ncu` (hardware counters are admin-only on pf9sodsia004, `RmProfilingAdminOnly: 1`; root needs no
driver change), GPU 0 only, clocks untouched, one complete large round of the setting above (652
kernels, 4096 slots; FP64, own_lu_top, device outer loops), sections SpeedOfLight,
ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, Occupancy. ms under ncu (61.8 ms for the round, 55
under nsys); % of peak, time-weighted over the kernel's launches in the round.

**Memory-bound, ~66% of the round** (DRAM 72-94% of peak: at the roofline, only fewer bytes help):

| kernel | ms | DRAM % | SM % | FP64 pipe % |
|---|---|---|---|---|
| `lu_left_u` (factorization, U step) | 6.65 | 89 | 19 | 19 |
| `lu_bwd` (backward solve, 148 levels) | 5.44 | 83 | 31 | 34 |
| `lu_left_l` (factorization, L step) | 5.34 | 84 | 57 | 59 |
| `lu_tr_in_vals` (J -> interleaved) | 4.49 | 89 | 11 | 11 |
| `lu_left_split` | 3.78 | 76 | 28 | 31 |
| `lu_fwd_split` (occupancy 47%) | 3.35 | 75 | 11 | 12 |
| `lu_fwd` | 3.24 | 91 | 12 | 12 |
| `lu_residual_i` (refinement residual) | 3.14 | 93 | 15 | 15 |
| b / x transposes, `top_outside`, `top_fo`, gathers, `update_Vm`, `lu_bwd_split` (occupancy 33%) | ~6 | 72-94 | low | low |

**FP64-compute-bound, ~22%** (FP64 pipe 63-85%; FP64 is 1/64 of the FP32 rate on this card):

| kernel | ms | DRAM % | FP64 pipe % |
|---|---|---|---|
| `spmv_fill_F_kernel` (3 passes) | 7.29 | 69 | 80 |
| `fill_J_kernel` (DRAM-saturated too) | 4.07 | 88 | 75 |
| `update_Va_kernel` | 1.16 | 60 | 85 |
| `compute_residuals_kernel` | 1.10 | 69 | 83 |
| `reduce_step_norms`, `oc_unrealistic`, `vc_hold_rows` | ~0.7 | 43-79 | 63-85 |

**Latency-bound, ~6%** (memory and SM both low, low occupancy): the device outer model (`on_solved`
1.55 ms at 4-14%, `share_lost` 0.57, `materialize`, `mask_rows`) and `top_fwd`: serial per-row work and
barriers. **Mixed, ~5%**: `top_factor` (1.66 ms, 26% memory / 41% SM: one block per matrix, a barrier
per pivot), `top_bwd`, `template_rows`.

Cross-check: the same kernels timed by nsys in an FP32 build (same launch configurations, batch 4096)
against the FP64 profile: 1.8-2.0x on the memory-bound ones (the int index arrays do not shrink),
2.2-2.8x on the FP64-bound ones (`spmv_fill_F` 2.77x, `update_Va` 2.6x, `fill_J` 1.96x), 1.4-1.7x on the
latency / occupancy-bound split levels and reductions.

## 4. What a lower precision would buy

**Full FP32 build** (estimate from section 3): a large round ~55 -> ~30 ms; the run 0.357 -> ~0.22 s
(4 large rounds 237 -> ~130 ms, the two medium ones ~58 -> ~35, the tail ~35 -> ~30, host + construction
+ device outer model ~27 unchanged). Two conditions, neither holding today:

- FP32 does not converge at the loose tolerance (1e-4 pu) on PtFige: every row ran its 30 Newton
  iterations, 0 converged (the bus currents are sums of Ybus terms up to ~1e3-1e5 pu that cancel;
  float leaves ~1e-4-1e-3 pu). `--tight` (1e-10 pu) is out of reach anyway.
- `outer_loops_backend='device'` is refused in an FP32 build (`_outer_loops.py`): the device model is
  written in double and works in the driver's `real_t` buffers in place (`device_model_f64`). Not a
  numerical necessity -- the host model already decides on FP32 states, widened to double -- but the
  port is needed (double inside with float <-> double at the boundary, keeping lockstep with the host
  model; or the model itself in `real_t`, lockstep then lost near thresholds). With the host backend
  (~+0.3 s on PtFige) the gain is gone.

Measuring it also needed a shim: the facades default to `precision="fp64"`, which an FP32 extension
refuses, and `bench_n1.sh` passes none.

**Mixed precision, the realistic one:** V, F, the convergence test and the refinement residual stay in
FP64, only the LU (factorization, triangular solves, the J / b / x transposes) in FP32. Newton takes an
inexact step anyway, and `ir_n_steps=1` -- already what the bench runs -- is exactly the refinement of a
float LU against a double residual. The LU's memory-bound kernels are ~35 ms of a large round (nsys)
at ~1.85x: about -16 ms per large round, ~-80 ms per run (estimate), the convergence criterion
untouched and the outer loops on the device in FP64. To check: the number of refinement steps a float
LU needs on PtFige's J (its clamped pivots at FP32's 1e-5 relative clamp), and whether Newton's
iteration counts -- hence the outer loops' decisions -- stay those of the FP64 run.

## 5. The split levels (done) and `top_factor` (parked)

**Split levels, done** (`ac835e8`): `lu_left_split` / `lu_fwd_split` / `lu_bwd_split` declared a static
`red[32][33]` (8.4 KB) whatever their number of warps K, which capped them at 10 blocks per SM (42%
occupancy for K = 2, 83% for K = 4, Nsight Compute's shared-memory limiter); the array is now K x 33
values of dynamic shared memory. Same sums in the same order. PtFige N-1: 0.357 -> 0.354 s, less than
the ~5 ms expected: above ~75% of peak DRAM a higher occupancy buys little, and these kernels are
latency-bound in the tail anyway.

**`top_factor`, parked.** In the large rounds it factors the capped trailing block (`tops_[1]`, 104
pivots): one block of 512 threads per matrix, 39.7 KB of dynamic shared memory, hence 2 blocks per SM
(66% occupancy; a third does not fit in this GPU's ~100 KB per SM), 26% memory / 41% SM: limited by
its two block barriers per pivot. 1.66 ms per large round, 0.3-1 ms per tail round. Tried:

- **One barrier per pivot** (every thread clamps the pivot and takes its reciprocal itself, the
  previous reciprocal kept in a register; the same bits): +7 ms per run. 512 FP64 divisions per pivot
  cost more than the barrier on a card whose FP64 rate is 1/64 of FP32.
- **A smaller capped block** (`GPUSIM2GRID_TOP_LU_M_LARGE`, PtFige N-1 session run, median of 5): 64 ->
  0.356 s, 80 -> 0.355, 96 -> 0.355, 104 (the default) -> 0.354, 160 -> 0.367. What a smaller block saves
  in shared memory / barriers, the pivots it hands back to the per-level launches cost again: the
  default stays.

What would be left is a different kernel, e.g. one warp per matrix (`__syncwarp` instead of block
barriers, several matrices per block, the trailing block's pattern sized to fit), or a pivot sequence
avoiding the barrier without a per-thread FP64 division (warp 0 computes the reciprocal and the other
warps wait on a flag rather than a full barrier). Estimate: ~0.5-1 ms per large round plus a little in
the tail, i.e. -3 to -6 ms per run (~1-2%): not worth a day before the larger levers of sections 1, 2
and 4 (shared Ybus, J in the LU's layout, mixed precision).
