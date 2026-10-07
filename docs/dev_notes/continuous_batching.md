# Continuous batching: refill slots instead of solving chunks to the end

**Design note, not documentation.** Proposal written on 2026-10-07. **Part A is implemented**
(2026-10-07: `scheduling="continuous"`, see `docs/api.rst` "Batch scheduling" and CLAUDE.md);
the review that preceded it corrected a few points of the proposal, marked *Review* below. A.6
item 1 (the batched adjoint) is implemented too, differently from the proposal (see there); A.6
item 2 and Part B are not. It is not part of the built documentation (`docs/*.rst`).

- **Part A** works on the current code, with no outer loop, and can start now. Its A.6 lists
  what the first version leaves out, to bring back once it is implemented and tested, before
  Part B.
- **Part B** extends it to OpenLoadFlow-style outer loops. It depends on lightsim2grid's outer
  loop work (branch `dev_outerloops_refacto`, design note
  `docs/dev_notes/outer_loops_fixed_sparsity.md` in that repo), and its batch API is still
  open.

## Why

`BatchPfDriver::solve` today works chunk by chunk:

- it cuts the active rows into `ceil(n_active / batch_size)` chunks;
- `_solve_chunk` prepares one chunk (`prepare_Ybus_batch`, `prepare_Sbus_batch`,
  `fill_*_buffers`);
- it runs `run_nr_loop` for a fixed `nb_iter`, with no convergence test;
- it then post-processes every slot: residual, NaN masking, limit and physical checks,
  storing V, branch flows.

So every row costs `nb_iter` iterations, whatever it needs. `nb_iter` must be set for the
hardest row, and a row that needs more is reported as not converged.

With outer loops, rows need different numbers of re-solves. A chunk solved to the end then
waits for its slowest row, and cuDSS uniform batch costs the same per slot whether or not the
slot still has work. An RTE/AssistFlux GPU security-analysis prototype measured this
(`_gpu_security_anlysis_assistflux/docs/continuous_batching.md`, in the lightsim2grid tree):

- on France, the tail of nearly empty batches was its first bottleneck (about 90 ms per call on
  a 64-slot batch with about 9 active rows);
- carrying rows over between batches cut the total time by 25 to 44 %.

Continuous batching keeps a fixed set of slots. A row enters from a queue, leaves as soon as
it is done, and its slot is refilled at once. Only data movement changes: the kernels, the
Jacobian pattern and the cuDSS analysis stay the same.

---

# Part A: design for the current code (no outer loops)

## A.1 What the mode does

Its inputs are:

- `S`, the capacity (the batch size);
- `k`, the number of Newton iterations per round, fixed for the whole run;
- `max_iter`, the iteration budget of a row;
- `tol`, the convergence tolerance.

```
fill the S slots with the first active rows (active order)
repeat:
    k Newton iterations on all S slots            run_nr_loop(nb_iter = k), as today
    per-slot ||F||inf at the current V            device
    D->H: the S residuals                         the only sync of the round
    host, for each slot holding a row:
        residual not finite             -> DIVERGED, leaves
        residual <= tol                 -> CONVERGED, leaves
        iterations >= max_iter          -> MAX_ITER, leaves (V and residual kept, as today)
        otherwise                       -> stays and keeps its V
    eviction (device, leaving slots only):
        NaN-mask, limit / physical checks, flows, write V and residual to the results
    load the next rows of the queue into the freed slots (phantom once the queue is empty)
until the queue is empty and no slot holds a row
```

- **Convergence test.** It is lightsim2grid's rule, ‖F‖∞ < `tol` (strict, `BaseAlgo.cpp`), with
  F after the masks: the same F `compute_residuals_kernel` reduces today. `violation_tol` keeps its
  current meaning (trusting V in the checks).
  - *Review:* the comparison is strict, and `tol` is per unit (what `get_residuals()` reports);
    lightsim2grid's public `tol` is in MVA (its test is ‖F‖∞ < `tol / sn_mva`, `LSGrid.cpp`,
    `BaseBatchSweep.cpp`). Default 1e-8 (1e-3 in FP32, whose residual floor
    is around 1e-4 on case118).
  - *Review:* a row loaded already converged does **not** leave with 0 iterations: a round runs
    k iterations before its check, so a row runs at least k. With `k = 1` the counts are
    lightsim2grid's one-off `ac_pf` ones otherwise (pinned in the tests).
- **Iteration count.** A row's count is a multiple of `k`, counted from its load. A row that
  converges in the middle of a round finishes the round; that is harmless and only refines it.
    With `k = 1` the count is exact. The count depends on the row alone, never on the other slots.
  The budget rounds up: a row leaves at the first check with iterations ≥ `nb_iter`, so it can
  run `ceil(nb_iter / k) * k` of them.
- **Cost of the residual pass.** It is one extra SpMV + F per round, on top of the `k`
  iterations. With small `k` this is visible. It can be removed later: the next round's first
  iteration could reuse that F instead of computing it again (the slots just loaded would still
  need their own). The first version pays it.
- **Phantom slots.** Once the queue is empty, freed slots hold the base case, as the padded
  last chunk does today. They converge at once and are never written to the results
  (`d_slot_row = -1`).
- **Results.** The buffers are today's, in original row order: `d_V_results`, `d_residuals`, the
  violation and physical-check outputs. There are two new per-row outputs: the iteration count
  and the status (CONVERGED / MAX_ITER / DIVERGED, plus today's NOT_SIMULATED for rows the
  connectivity pre-check drops before they enter the queue). A MAX_ITER row whose residual
  exceeds `violation_tol` gets its DIVERGENCE record exactly as today.

## A.2 What changes in the code

### Rows indexed by row, slots by indirection

Every per-chunk input is laid out today in active-slot order and **sliced per chunk**:

| input | today |
|---|---|
| branch-trip patches | `chunk_ranges_` + `apply_contingencies_kernel` (chunk-relative slot ids) |
| masks: identity rows, NaN V, `jov`, `str`, `vcp` | `MaskStreams`, one `ChunkPatchRange` per chunk |
| Sbus rows | `d_Sbus_all` sliced from `chunk * batch_size` |
| `gen_v` reseed and `v_set` | `apply_gen_v_kernel` / `GenVsetSlots::prepare`, `row_offset = chunk_idx * batch_size` |
| `vm_reseed` | `vr_ranges_` |
| slack weights and Sbus corrections | `SlotSlackRedistribution::prepare_weights(chunk_idx, ...)` |
| warm start V | `d_V_init_all` sliced per chunk |
| tripped branches (limit check) | `TrippedBranchTable`, indexed by global active slot |

A chunk is a contiguous range of active rows; in this mode a slot may hold any row. The
proposal:

- every per-row input stays on the device **in active-row order, indexed by row**, with a
  CSR-like `row_ptr` for the variable-length streams;
- the batch holds `d_slot_row[S]`, the active row of each slot (-1 for a phantom slot).

Each consumer then falls in one of two kinds.

1. **Written when a slot is loaded.** A row being loaded into a slot gets:
   - the base V and base Ybus values tiled into that slot;
   - its Ybus patches, its Sbus row, its `gen_v` / `vm_reseed` / `v_set`;
   - its slack weights and corrections, its warm-start V;
   - the initial `slack_absorbed` and controller Q.

   This becomes a new source hook, `load_slots(ctx, d_load_slots, n_load, ...)`, whose kernels
   run over the listed slots only and find the row's segment through `d_slot_row` and
   `row_ptr`. Today's `prepare_Ybus_batch` + `prepare_Sbus_batch` is then `load_slots` of every
   slot of a chunk. Per-row dense data that the Newton kernels read with a stride
   (`sbus_stride`, `slack_w_stride`, `vc_vset_stride`) keeps a per-slot copy written at load,
   so the Newton kernels do not change.
2. **Read at every iteration.** These are the mask streams (`nr_apply_F_masks`,
   `nr_apply_J_masks`, `nr_mask_v_nan`), which today walk a flat chunk stream of
   `(slot, row / position)` entries. Two options:
      - **(a)** the kernels read `d_slot_row` and walk that row's segment. Nothing is rebuilt, at
     the price of some load imbalance between slots.
   - **(b)** after every round that loaded a slot, rebuild a flat stream for the batch (a
     gather plus an exclusive scan over the per-slot counts).

   *Review (implemented): (b).* The host holds the per-row counts and the slot → row table, so
   it computes the offsets itself and ships them; one device gather (`SegmentGather`,
   `contingency/segment_gather.cuh`) builds the batch stream, with no extra sync. The five mask
   kernels stay as they are, and an islanded row's hundreds of masked buses are not one thread's
   serial loop. The same gather serves the load-time streams (patches, `vm_reseed`, dP), and the
   host builders already give the per-row layout when run with a chunk size of 1.

In short:

- what is copied per slot at chunk start today is copied per slot at load;
- what is walked as a chunk stream during the iterations is walked through `d_slot_row`.

The shared pieces to convert are `MaskStreams`, `SlotSlackRedistribution`, `GenVsetSlots` and
`TrippedBranchTable`.

### Post-solve kernels: one eviction map

These kernels all take `(c_start, actual_batch, d_result_map)` and write slot `s` at
`d_result_map[c_start + s]` (or `c_start + s`):

- `compute_residuals_kernel`, `check_limit_violations_kernel`;
- `check_bus_q_violations_kernel`, `check_hvdc_p_violations_kernel`,
  `check_gen_p_violations_kernel`, `check_gen_pv_release_violations_kernel`;
- `compute_branch_flows_kernel`, `scatter_V_results_kernel`.

Replace that triple with one array, `d_evict_row[S]`: the original row index of slot `s` when
it leaves this round, -1 otherwise (skip). Chunk mode passes exactly what it computes today,
so its results stay bit-identical. Doing this first, on its own, makes the rest incremental
(see A.4).

*Review (implemented):* an array of original rows is not enough: `check_limit_violations_kernel`
reads the `TrippedBranchTable` by **active** row. The kernels take `d_slot_active[S]` (the active
row of each slot, -1 = skip; `nullptr` = today's `c_start + s`) and keep
`out_c = d_result_map ? d_result_map[a] : a`.

The residual pass of a round also writes a per-slot buffer, `d_slot_res[S]`, which the host
reads for every slot. `d_residuals[row]` is only written for the slots leaving.

### NaN masking in place, on leaving slots only

`nr_mask_v_nan` writes NaN into the masked buses of `d_V_batch`, before the checks and the
store. In chunk mode the chunk is finished by then. In this mode it **must run on the leaving
slots only**: the masked buses of a slot that stays would be destroyed. A NaN in V poisons the
next SpMV through the zeroed Ybus entries of a tripped branch, the same reason the warm start
gathers the base V for masked buses (`gather_v_rows_kernel`). The leaving slots are reloaded
before the next round, so masking them in place is fine. The checks then read `d_V_batch`, the
slot's patched Ybus values and its Sbus, as today.

### Per-slot Newton state

Today, at the start of a chunk, `init_slack_absorbed_kernel` and the controller-Q init
(`d_vc_q_batch`, tiled from `d_vc_qoff` or zeroed) run over every slot. In this mode they run
for the loaded slots only. A slot that stays keeps that state, since it is part of its Newton
iterate.

### Scheduler

The scheduler lives on the host, in `BatchPfDriver::solve`, as a second loop next to the chunk
loop.

- **Queue:** the active rows, in order.
- **Per round:**
  1. `run_nr_loop(nb_iter = k)`, with the policy as today;
  2. the residual pass;
  3. D→H of `d_slot_res`, then the host's decisions;
  4. H→D of `d_evict_row` and of the load list;
  5. the eviction kernels, then `load_slots`.
- **Syncs:** one stream sync per round. The per-slot iteration counters are host
  bookkeeping.
  - *Review:* not a cost model: `CudaTimer::stop_ms` already synchronizes after every phase
    of every iteration. The scheduler's own host time is measured (`t_schedule`).
- **Determinism:** rows are loaded in active order into the freed slots, in increasing slot
  order, so a row's trajectory depends only on that row.

### Linear solver

Uniform cuDSS mode does one ANALYSIS, then only values change. `PolicyRefactorEvery::factorized_`
is already global across chunks: the rows of chunk 2 are refactorized today on the pivots of
chunk 1's first factorization. A reloaded slot therefore adds no new kind of numerical risk.
`direct_base_case_factors` keeps working, since it uses the base factors throughout.

**Refused (raise at configuration):**

- `direct_iter0_only`: it fills and factorizes J at the first iteration of a chunk and reuses it
  for the whole solve, which assumes every slot starts together;
- `direct_refactor_every_n`: its period counts the batch's iterations, not a row's;
- the cuDSS `BlockDiag` / `NonUniform` modes (`GPUSIM2GRID_USE_BATCH_MODE`): they re-analyse per
  chunk in `CudssBatchSolver::begin_chunk`.

### Also not supported in the first version

Both of these come back once the mode is implemented and tested, before Part B (see A.6).

- `keep_final_jacobian`, and so the batched adjoint behind `BatchPowerFlow`, **raises** in
  continuous mode. It needs a single chunk and each row's final J.
  *(Implemented since, see A.6 item 1.)*
- `ScenarioSweepSession`'s **warm and hot reuse paths are not taken**: in continuous mode every
  `run()` takes the cold path (a new `BatchPfDriver`, with its cuDSS ANALYSIS).

### The three batch sources

- **`InjectionBatch`** is the simplest: Ybus is shared, so loading a slot is V plus the Sbus row
  (plus `gen_v`). It is the first end-to-end target for the scheduler.
- **`ScenarioSweepBatch`** has everything: patches, masks, `gen_v`, `vm_reseed`, slack weights
  and corrections, warm-start V.
- **`ContingencyBatch`** is a subset of the scenario sweep (shared Sbus, or a per-row correction
  under `redistribute_slack`).

## A.3 API (proposal; names to settle)

These apply to the three batch facades and their sessions:

- `scheduling = "chunked"` (default, today's behaviour, untouched) or `"continuous"`;
- `nb_iter_per_round` (`k >= 1`, default 2), fixed for a run; `k = 1` checks after every
  iteration;
  - *Decided (implemented):* default 1. The residual pass is cheap next to a refactorization:
    on case6515rte N-1 (2876 rows, S = 512), k = 1 / 2 / 4 took 6.7 / 8.6 / 9.2 s, chunked
    nb_iter = 4 / 6 took 7.5 / 11.2 s.
- `max_iter` and `tol`, used only in continuous mode;
- **open:** whether setting today's `nb_iter` raises or is ignored in this mode, since it means
  nothing there;
- *Decided (implemented):* no `max_iter`: in continuous mode `nb_iter` is each row's budget.
  `tol` applies to both schedules (it decides the reported status in chunk mode too).
  `get_row_status()` (`RowStatus`: CONVERGED, MAX_ITER, DIVERGED, NOT_SIMULATED) and
  `get_row_iterations()` are filled by both schedules;
- new outputs: the iteration count and the status of each row;
- new timings:
  - `n_rounds`;
  - `occupancy`: useful row-iterations / (`S` × iterations run), the figure that tells whether
    the mode pays;
  - the time spent on the host and in syncs for the scheduling.

## A.4 Phasing

*Implemented in this order; chunk mode was checked after phases 1 and 3 against golden dumps of
the whole test suite (every session's outputs), bitwise where the baseline is, within its
run-to-run noise elsewhere. Phase 0 on case6515rte (4000 N-1 rows, S = 512): 2.7 / 16 / 72 /
9 / 0.3 % of the rows need 1 / 2 / 3 / 4 / 5-6 iterations; chunk mode needs 48 batch iterations
for all to converge (nb_iter = 6), the continuous replay 25 at k = 1
(`benchmarks/continuous_batching_potential.py`).*

0. **Measure, with no code change.** Run today's mode with `nb_iter = 1 ... 8` on the target
   grids and count, for each value, the rows whose residual is at most `tol`. A row's trajectory
   does not depend on `nb_iter`, so this gives the distribution of the iterations each row needs:
   the upper bound of the gain without outer loops.
1. **Eviction map.** Generalise the post-solve kernels to `d_evict_row`. Chunk mode stays
   bit-identical, and the existing test suite guards it.
2. **Scheduler.** The scheduler, the residual pass, the refusals, and `InjectionBatch::load_slots`.
3. **Row-indexed sources.** Per-row streams (`MaskStreams`, `SlotSlackRedistribution`,
   `GenVsetSlots`, `TrippedBranchTable`), then `ScenarioSweepBatch`, then `ContingencyBatch`.
4. **Facades and benchmark.** Facade options, outputs and timings; benchmark against chunk mode
   on the French grids.

## A.5 Tests

- **Against chunk mode** with a large `nb_iter`: the same V and residual per row, to
  tolerance, and the same violation and physical-check records (compared as sets) for the
  converged rows.
- **Invariance:** a row's result does not depend on `S` or on the row order, to rounding.
  Batched reductions make two sessions reproducible only to about 1e-15, as today.
- **Against lightsim2grid,** row by row (`ContingencyAnalysisCPP`, `ScenarioSweep`), at the same
  tolerance. With `k = 1` the iteration counts should match lightsim2grid's.
- **Reload hygiene:** a row with masks, pins or overrides loaded into a slot that held a plain row,
  and the reverse, leaves nothing behind.
- **Statuses:** a row forced to diverge ends DIVERGED or MAX_ITER without disturbing its
  neighbours.
- **Edge cases:** `n_rows < S`; no row at all; every row dropped by the pre-check; `k = 1`;
  `k > max_iter`.
- **Refusals:** each refused strategy, mode or option raises.
- **Precision:** tolerances from the `solver_atol` / `residual_atol` fixtures (FP32 and FP64).

## A.6 TODO once the feature is implemented and tested (before Part B)

The first version leaves these out (see "Also not supported in the first version"). Bring them
back once Part A is implemented and tested, and before starting the outer loops.

1. **`keep_final_jacobian`, and the batched adjoint behind `BatchPowerFlow`.** Today:
   - `_solve_chunk` refills `d_J_values_batch` at V_final after the loop
     (`nr_fill_J_at_current_V`), and `solve()` throws when there is more than one chunk;
   - `solve_JT_batch` then reads the chunk buffers (alias mode, `run_counter` guard) or cloned
     copies (snapshot mode: `j_values_dlpack()` / `ybus_values_dlpack()` / V).

   In continuous mode a row's final state exists only at its eviction. So:
   - refill J at V_final for the **leaving slots only**, in the eviction step;
   - copy their J values, their patched Ybus values and their V into per-row storage, in
     original row order: a snapshot by construction, so the alias mode has nothing to alias;
   - `BatchAdjoint` then solves Jᵀλ chunk by chunk over the stored rows (its own
     `CudssBatchSolver` over Jᵀ, one ANALYSIS, rows packed in any order);
   - the per-row mask streams feed `zero_identity_rows_kernel`, and the stored Ybus values feed
     `gen_v_adjoint_kernel`.

   Cost: `n_rows × (nnz_J + 2·nnz_Y + n_bus)` values kept until the backward pass, against one
   chunk's worth today. It should be optional, and refused (or chunked to host memory) above a
   size budget. Gradients are checked against finite differences, as for every adjoint here.

   *Review (implemented, 2026-10-07): nothing is stored; the backward rebuilds J.* A row's J
   depends on its converged V, its patched Ybus values, its per-slot slack weights and its mask
   streams (identity rows, `jov` overrides) -- not on Sbus, `slack_absorbed` or the controllers'
   Q. The run keeps V anyway (`d_V_results`; the caller's V snapshot in snapshot mode) and the
   batch source holds the rest, so storing J and Ybus would cost `n_rows × (nnz_J + 2·nnz_Y)`
   for values that can be recomputed exactly. `BatchPfDriver::_solve_JT_rebuilt` (taken when
   the last `solve()` was continuous) works over adjoint chunks of `S` active rows:
   - the source's own `load_slots` fills the slots with the chunk's rows (phantom tail = base
     case): patched Ybus, slack weights, starting V;
   - the rows' converged V is copied over it, finite entries only: a masked bus is NaN in the
     results and kept its loaded value through the solve (identity row, `dx = 0`), so the slot
     holds the row's final state entry for entry -- and no NaN reaches the SpMV;
   - SpMV + `nr_fill_J_at_current_V` with the rows' mask streams bound (`bind_slots` of that
     slot table): the forward's own fill sequence;
   - Jᵀ permuted, refactorized (FACTORIZE once per driver), solved; the identity rows zeroed
     and λ scattered to original rows; the `gen_v` contraction on the NaN-masked V and the
     reloaded Ybus.

   Nothing changes in the forward (`keep_final_jacobian` has no effect there) and the extra
   memory is nil; the backward pays one reload + SpMV + J fill per chunk, small next to the
   factorization. The Jᵀ factors of a single-chunk backward on the driver's own V survive for a
   second backward on the same forward, as in the chunked path. Refused after a continuous run:
   J / Ybus snapshots (`solve_JT_batch`'s `d_J_ext` / `d_Ybus_ext`, `j_values_dlpack()` /
   `ybus_values_dlpack()`). `BatchPowerFlow.from_lsgrid(scheduling="continuous",
   batch_size=S)` exposes it; its snapshot is V alone. Pinned in
   `tests/python/test_batch_power_flow.py::TestContinuousScheduling`: gradients equal to the
   chunked ones for S = 1, 3, n (trips, `gen_v`, generator contingencies with released and
   pinned buses), islanded rows masked (`handle_disconnected_grid`) or dropped (non-identity
   active map), central differences, Jᵀ reuse counters, the guards.

   What this leaves for item 2: the backward needs the forward's batch source, and today every
   continuous `run()` is cold, so a backward must come before the next forward even with a V
   snapshot (`source_build_counter` guard; `gradcheck` cannot run there). Once hot runs keep
   the source, a V snapshot survives an injection-only forward -- **except** for the per-row
   slack weights under `redistribute_slack`, which `run()` re-sets on the live source on every
   run (the saturated units follow the injections): the hot path must then either keep the
   weights of the forward the snapshot belongs to, or the guard must also cover them.
2. **`ScenarioSweepSession`'s warm and hot reuse paths.**
   - Add the scheduling options (`scheduling`, `nb_iter_per_round`, `max_iter`, `tol`) to the
     `ScenarioSweepDriverConfig` snapshot, so that changing them forces a cold run.
   - With row-indexed data:
     - **warm** (topology or generator mask changed) means rebuilding the row-indexed streams.
       There are no chunk ranges to match, so `forced_batch_size` only fixes the capacity `S`;
     - **hot** (injections or `gen_v` only) means updating `d_Sbus_orig` and the per-row
       `gen_v`, which `load_slots` reads at the next load.
   - `fixed_batch_capacity` becomes implicit: the capacity is always `S`.
   - `mark_reused` and the `driver_build_counter` / `source_build_counter` counters keep their
     meaning. `BatchPowerFlow`'s snapshot guard (`source_build_counter`) has to hold once item 1
     is in.

---

# Part B: extension to outer loops (future)

This part waits on lightsim2grid (see B.6), and comes after Part A's TODO list (A.6). What Part A
must already do so that B is cheap is in B.7.

## B.1 Row lifecycle

Once its inner solve has converged, a row has two outcomes instead of one: **fully converged**
(terminal) or **needs an outer loop**.

```
            load (default controls, base V)
 waiting ───────────────────────────────▶ running ── k iterations ──┐
    ▲                                       ▲  not converged, budget left
    │                                       └─────────────────────┘
    │                                                                │ converged
    │                                                                ▼
    │                       DIVERGED / MAX_ITER ◀── budget hit   checks on GPU (eviction)
    │                                                                │
    │  re-enter (new controls, warm V)                    stable ────┴──── needs action
    └─────────────────────────── values modified ◀── decide + patch (async stream)
                                                                     │
                                              DONE ◀─────────────────┘ (stable)
```

The terminal statuses follow lightsim2grid's `OuterLoopStats`:

- done (STABLE);
- diverged, or not converged within `max_iter`;
- an outer loop FAILED (e.g. the slack could not be distributed);
- the outer-iteration cap was reached (UNSTABLE);
- unrealistic voltage.

## B.2 Running every check at once, then routing

At eviction, one set of kernels evaluates every loop's trigger on the converged row. That is
the physical checks of today, realigned on OpenLoadFlow's rules, plus the new triggers. The
driver of OpenLoadFlow 2.3.0, mirrored by lightsim2grid's `OuterLoopAlgo::compute_pf`, is
sequential:

- it checks the current loop again after each re-solve until that loop is stable, then moves
  on;
- a pass stops at the last loop that was unstable (never reset between passes);
- passes repeat while one of them re-solved.

Evaluating every trigger at once is still equivalent, as long as:

1. each loop's detection **is** the trigger its check acts on. This is lightsim2grid's
   `BaseOuterLoop` contract: the trigger is computed in one place, `detect`, reused by `check`;
2. a check that finds the loop stable changes nothing.

The decision is then "the first triggered loop from the row's current position in its pass".
That needs a small per-row driver state: current loop, last unstable loop, pass number, outer
iteration count, Newton iterations of the current pass.

Two kinds of loops act without a violation, and are encoded as "act on the first visit, then
follow the state":

- `PhaseControl` and `ShuntVoltageControl` act at their first visit regardless;
- `TransformerVoltageControl` runs a stage machine (INITIAL / CONTROL / COMPLETE).

The unrealistic-voltage check (on the buses whose magnitude is a Newton unknown) runs on every
solve, or in robust mode only after the last loop able to fix it, exactly as lightsim2grid's
driver does.

## B.3 Decide and patch on a second stream; park the row

- **Streams.** The checks run on the main stream at eviction. Deciding and patching run on a
  second stream, ordered with CUDA events, so a parked row is loaded only once its patch is done.
  The host only moves row indices between queues and reads counts once per round. The row data
  stay on the device.
- **Parking.** A row waiting for its patch **leaves its slot**, which is refilled. Keeping it
  frozen in place would idle the slot for a round. The parked state is:
  - its V;
  - the Newton unknowns beyond V: `slack_absorbed`, controller Q, and later α / ρ / B;
  - its **control state** (see B.7);
  - its per-row loop state: e.g. the PV→PQ switch count of each bus, the units' current targets,
    the transformer stage.
- **Re-entry.** Its Ybus and Sbus are rebuilt from its row inputs plus its control state, through
  the **same load path as Part A**:
  - a first load is "default control state + base V";
  - a re-entry is "modified control state + warm V".

  Nothing else is stored per parked row. Memory is bounded by the rows in flight (slots + parked).
- **Iteration budget.** The per-row iteration counter is the *inner* one: it restarts at each
  re-entry.

## B.4 Porting the decisions, not reusing lightsim2grid's loop code

Deciding on the GPU means each loop's *action* becomes a kernel, not only its detection:

- `DistributedSlack`, `AcHvdcAcEmulationLimits` and `VoltageMonitoring` are small;
- `ReactiveLimits` is the large one: `maxPqPvSwitch` per bus, keeping the strongest PV bus, robust
  mode, voltage-control groups, capability curves read at the moved target P.

So "reusing lightsim2grid" means:

- the loop list and order (`LSGrid::get_outer_loops()`, each loop's `name()` / `get_params()`);
- the driver parameters (`OuterLoopDriverParams`: `max_outer_iterations`, robust mode, realistic
  band);
- the plans built by its headers, as gpusim2grid already does for detection (`BusQCheck.hpp`,
  `GenPvReleaseCheck.hpp`, `GenPCheck.hpp`, ...);
- parity tests, row by row, against its `NROuter_*` algorithms.

lightsim2grid's loop classes themselves are not called. Calling them per row on the host
(through an `OuterControls` subclass) would give parity by construction, but would need each
row's V on the host at every outer iteration, and the AssistFlux prototype measured host-side
checks at 30 s out of 68 s before moving them to kernels.

A rare case can still be sent to the host with only that row's data: a truncated report, or a
group configuration the kernels do not handle.

## B.5 Device mechanisms, per loop

The table follows OpenLoadFlow's default order. The first four loops are on by default.

| loop | what it changes in the solve | already in gpusim2grid | new |
|---|---|---|---|
| DistributedSlack | Sbus values (units' targets; a non-regulating unit's target Q follows its limits) | per-row Sbus correction, verbatim `distribute` port (`slack_redistribution.hpp`) | apply it again from the cumulative mismatch of the slack bus |
| AcHvdcAcEmulationLimits | the droop regime of each line | droop kernels, the HVDC P check | per-row regime (today shared by the whole batch) |
| VoltageMonitoring | an idle standby SVC's "Q = 0" row becomes its voltage row | held controllers, per-row `v_set` (`GenVsetSlots`), the standby check | enrol the monitors as held controllers in the ledger |
| ReactiveLimits | PV↔PQ, group controllers held at a limit, V = 1 restart | `set_pv_pq_switches`, `set_vc_controller_pins` / `_releases`, `Contingency::pinned_buses`, `vm_reseed` | applied as control state at re-entry instead of per-run inputs |
| PhaseControl, TransformerVoltageControl, ShuntVoltageControl (off by default) | new unknowns α / ρ / B; the transformer or shunt Ybus entries patched every Newton iteration | per-slot Ybus values | `BranchControl` / `ShuntControl` as kernels: the Ybus patch, dS/dα and dS/dρ, the custom rows, a limiter's current |

The inner Newton of lightsim2grid's outer-loop algorithms has a single slack:
`use_distributed_slack=False` (`drop_multislack_augmentation`) already reproduces it. The
Jacobian pattern comes from lightsim2grid's `NROuter_*` system: the union of everything its
loops reserve.

## B.6 What waits on lightsim2grid

- **The ledger** of an `NROuter_*` solve (its union pattern), with accessors for the
  `BranchControl` / `ShuntControl` columns and rows.
- **Row-aware loops.** Detection already honours `OuterContext::masked` and `id_me_to_solver`,
  but the actions read the grid directly. For instance, `DistributedSlackLoop::_initialize`
  collects its units through `grid.id_me_to_ac_solver()`, and `BranchControl::_add_entry` reads
  the transformer's status off the grid. The parity reference must take a row's tripped branches,
  islanded buses and disconnected generators into account.
- **Detection realigned on OpenLoadFlow.** This is a breaking change of the check outputs, with
  new trigger kinds (`SLACK_MISMATCH`, `UNREALISTIC_VOLTAGE`, `REACTIVE_LIMIT_MOVED`, ...). The
  gpusim2grid kernels and enums follow it.
- **The starting state of a contingency row.** OpenLoadFlow's security analysis restores the N
  state after the loops (PV / PQ labels, moved targets, taps). The other option is to start from
  scratch. Today's workflow (`bake_outer_loops` + `can_be_pv` + detection + the one-shot second
  pass of `reactive_limits_outer_loop`) approximates the first. The second pass becomes a special
  case of B, to be replaced by it.

## B.7 What Part A must already do so that B is cheap

1. **Separate a row's inputs from its control state.** The inputs never change (injections,
   trips, `gen_off`, `gen_v`). The control state can change: pinned / switched buses, held and
   released controllers, Sbus deltas, `v_set`, HVDC regimes, `vm_reseed`. Its default is derived
   from the inputs. The load path reads both. In Part A the control state is always the default.
2. **Let the load path take an optional warm V and warm unknowns** (`slack_absorbed`,
   controller Q) per row. Part A only uses the warm V, for `set_v_init_from_ptr`.
3. **Leave the eviction checks' records on the device in a per-row form the scheduler can read,**
   not only in the user-facing result buffers.
4. **Keep the status enum and the per-row iteration counter extensible:** the counter becomes
   the inner counter, and the status gains the outer outcomes.
