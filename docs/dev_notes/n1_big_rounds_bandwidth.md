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
