# Continuous batching: refill slots instead of solving chunks to the end

**Design note, not documentation.** Proposal written on 2026-10-07. **Part A is implemented**
(2026-10-07: `scheduling="continuous"`, see `docs/api.rst` "Batch scheduling" and CLAUDE.md);
the review that preceded it corrected a few points of the proposal, marked *Review* below. A.6
item 1 (the batched adjoint) is implemented too, differently from the proposal (see there); A.6
item 2 and Part B are not. It is not part of the built documentation (`docs/*.rst`).

- **Part A** works on the current code, with no outer loop, and can start now. Its A.6 lists
  what the first version leaves out, to bring back once it is implemented and tested, before
  Part B.
- **Part B** extends it to OpenLoadFlow-style outer loops. It follows lightsim2grid's outer
  loop work (branch `dev_outer_batch_perf` of `../lightsim2grid_outerloops`, design note
  `docs/dev_notes/outer_loops_fixed_sparsity.md` in that repo), whose batches run the loops
  per row since. Part B was reviewed against it on 2026-10-15; the scope decisions are in B.0.

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

What still waits on lightsim2grid is in B.6. What Part A must already do so that B is cheap,
and how much of it is done, is in B.7.

## B.0 Decisions (2026-10-15)

Scope of the first version, decided before planning:

1. **Loops.** The four OpenLoadFlow runs by default, in its order: `DistributedSlack`,
   `AcHvdcAcEmulationLimits`, `VoltageMonitoring`, `ReactiveLimits`. `PhaseControl`,
   `TransformerVoltageControl` and `ShuntVoltageControl` are left out (an issue to open). They
   also need accessors lightsim2grid does not expose yet (B.6).
2. **Facades.** All three batch facades: `ScenarioSweepGPU`, `InjectionSweepGPU` and
   `ContingencyAnalysisGPU`. The scenario sweep is the general case (an injection sweep is a
   scenario sweep without trips), so it is built and validated there first.
3. **Scheduling.** Continuous only. With the outer loops on, chunked scheduling is refused.
4. **Starting state of a row**, chosen by the caller:
   - from the N solve (default): a row starts from the N solve's loop state, as lightsim2grid's
     batches resume from their "n" solve (frozen buses, SVCs switched on, HVDC regimes, moved
     targets, V);
   - or from a voltage and grid state the caller provides, as the facades accept today.
5. **Today's workflow stays.** `bake_outer_loops` + `can_be_pv` + detection + the one-shot
   `reactive_limits_outer_loop` are kept alongside. Whether they are deprecated is decided
   later, once both can be compared.
6. **Parity.** Against lightsim2grid's `NROuter_*` batches, row by row, on French snapshots,
   to a small tolerance. lightsim2grid against OpenLoadFlow is a separate question, not
   addressed here.
7. **Decisions on the host first.** The detection runs on the GPU (B.2). The decision of a routed
   row runs on the host, with only that row's data copied back. Moving the decisions into
   kernels comes later (an issue to open).
8. **Zero pivots are not an issue for cuDSS.** Its REFACTORIZATION does not reuse the pivots of
   the previous factorization: it pivots as a FACTORIZATION does. The refactorization fallbacks
   lightsim2grid's KLU hits on rows rewritten by value (`speed_optimisation.md`) do not carry over.
9. **Jacobian size.** Every switchable bus is reserved up front (the union pattern of B.5).
   This is accepted on the CPU, so it is accepted on the GPU.
10. **No gradients** with the outer loops on in the first version: `BatchPowerFlow` and the
    batched adjoint refuse them (an issue to open).

Issues to open: the three remaining loops (1), the decisions as kernels (7), gradients
through the outer loops (10).

Prerequisites found while reviewing:

- **A lightsim2grid with `NROuter_*`.** The one in `.venv` (`~/Documents/lightsim2grid`,
  `dev_1.3.0`) has no `NROuter_*` algorithm. gpusim2grid must be built against the
  outer-loop branch (`../lightsim2grid_outerloops`, `dev_outer_batch_perf`) until it is merged.
- **The bridge on an `NROuter_*` grid is unverified.** `extract_ledger_data`
  (`ls2g_bridge.cpp`) reads the ledger through `LSGrid`'s solver accessors, which `NROuter_*`
  forwards to its inner Newton (`OuterLoopAlgo.hpp:190-215`). Still to check: the slack and
  VoltageControl accessors, and the pv labelling, since `NROuter_*` moves the other slack buses
  into `pv` internally (B.5).
- **The detection realignment can ship first, on its own.** lightsim2grid added
  `ViolationCategory::CONTROL` and the types 16-20 (`SLACK_MISMATCH`, `PHASE_CONTROL_P`,
  `PHASE_LIMITER_CURRENT`, `TRANSFORMER_VOLTAGE_DEADBAND`, `SHUNT_VOLTAGE_CONTROL`), and moved
  the types 9-12 and 15 into `CONTROL` (`batch_algorithm/LimitViolation.hpp:216-226`).
  gpusim2grid stops at 15 and still files those as `PHYSICAL`.

## B.1 Row lifecycle

*Reviewed 2026-10-15 against lightsim2grid `dev_outer_batch_perf`.*

Once its inner solve has converged, a row is detected (B.2) and has two outcomes instead of
one: **done** (no loop would act) or **routed to a loop**. A routed loop either acts (the row
is patched and re-queued) or ends the row.

```
            load (default controls, base V)
 waiting ───────────────────────────────▶ running ── k iterations ──┐
    ▲                                       ▲  not converged, budget left
    │                                       └─────────────────────┘
    │                                                                │ converged
    │                                                                ▼
    │                DIVERGED / MAX_ITER ◀── budget hit     detection at eviction (B.2)
    │                                                                │
    │                                          no loop would act ────┴──── first detected loop
    │                                                  │                          │
    │                                                  ▼                          ▼
    │                                          unrealistic check           decide (B.2, B.3)
    │                                           ok │      │ fails           │     │       │
    │                                              ▼      ▼               FAILED  cap   UNSTABLE
    │                                            DONE  UNREALISTIC          │     │       │
    │                                                                       ▼     ▼       │
    │  re-enter (new control state, warm V)                  OUTER_FAILED  OUTER_UNSTABLE │
    └──────────────────────────────── patched ◀───────────────────────────────────────────┘
```

The unrealistic-voltage check runs where B.2 says (every solve, or in robust mode after the
last loop able to fix it); the diagram shows it only at the end.

### Statuses

lightsim2grid's outcome of an outer-loop solve is **not one enum**. `OuterLoopStatus` has three
values only, `STABLE`, `UNSTABLE` and `FAILED` (`outer_loop/BaseOuterLoop.hpp:41`). The rest
comes from the inner solve's `ErrorType` and a flag. Per row, its batches publish the four
together in `OuterRowResult` (`batch_algorithm/BaseBatchSweep.hpp:73-96`): `status`,
`unrealistic_state`, `failed_loop` (the loop's name), plus `nb_outer_iterations` and
`nb_nr_iterations`. `OuterLoopAlgo.tpp:166-186` maps them as follows:

| outcome | `status` | `unrealistic_state` | `ErrorType` |
|---|---|---|---|
| done | `STABLE` | false | `NoError` |
| an inner solve diverged or used its `max_iter` | as left | false | the inner error (`TooManyIterations`, `SingularMatrix`, ...) |
| a loop failed (e.g. a slack residue it could not share) | `FAILED`, `failed_loop` set | false | `OuterLoopFailed` |
| `max_outer_iterations` reached | `UNSTABLE` | false | `TooManyIterations`, the **same code** as an inner `max_iter`: tell them apart by `status` |
| unrealistic voltage | **`STABLE`** | true | `UnrealisticState` |

The last line follows the code. The comment on `OuterLoopStats::unrealistic_state`
(`BaseOuterLoop.hpp:217`) says the status is then `FAILED`: the code leaves it `STABLE` and
the comment is wrong, or the reverse. Settle it upstream before pinning parity tests on it.

### In gpusim2grid

`RowStatus` (`contingency/row_status.hpp`) is append-only, and already says the outer loops add
after its four values (`Converged`, `MaxIter`, `Diverged`, `NotSimulated`). Proposal (names to
settle):

- the inner outcomes keep their values. `MaxIter` / `Diverged` mean the *last* inner solve
  failed; the iteration counter is the inner one and restarts at each re-entry (B.3);
- `OuterFailed`, with a per-row `failed_loop` (index into the loop list);
- `OuterUnstable` (`max_outer_iterations` reached): its own value, not lightsim2grid's
  `TooManyIterations` shared with `MaxIter`;
- `Unrealistic`: its own value. The row is not `Converged`, whatever upstream decides for
  `status`;
- per row, next to the status: `nb_outer_iterations` and the Newton iterations summed over
  every inner solve, as `OuterRowResult`;
- for parity tests, the sparse per-row result of `OuterRowResult`: moved generator and
  storage targets, HVDC regimes, phase and ratio taps, shunt sections.

## B.2 Detecting every loop at once, then routing

*Reviewed 2026-10-15 against lightsim2grid `dev_outer_batch_perf`; file:line references are
to that branch.*

### The driver to reproduce

The driver of OpenLoadFlow 2.3.0, mirrored by lightsim2grid's `OuterLoopAlgo::compute_pf`
(`OuterLoopAlgo.tpp:112-162`), is sequential:

- it checks the current loop again after each re-solve until that loop is stable, then moves
  on;
- a pass stops at the last loop that was unstable (`last_unstable`, never reset between
  passes);
- passes repeat while one of them re-solved;
- a check returns `STABLE`, `UNSTABLE` (act, re-solve) or `FAILED` (terminal for the row).

### Fused detection at eviction

At eviction, the kernels evaluate every loop's detection on the converged row. A detection is
a pure function of the row's V, its slot's Ybus and its loop state, so the order of the
detections does not matter and they can be fused. Most of them read the same quantity, the
bus power `V·conj(Ybus·V)` of the slot:

- ReactiveLimits reads its reactive part;
- DistributedSlack reads its active part at the slack;
- VoltageMonitoring, TransformerVoltageControl and the release tests read |V|;
- HvdcAcEmulationLimits reads the angles.

One kernel computes that power once per slot (today the bus Q and gen P checks each recompute
it) and then evaluates the per-loop tests. This is the physical checks of today, realigned on
OpenLoadFlow's rules, plus the new triggers.

### Routing: the first detected loop

The row is routed to **the first detected loop from its current position in its pass**. Its
action changes the row's equations; the row is re-queued and re-solved.

This is the sequential driver itself if and only if *detected* means **"its check would
return `UNSTABLE`"** ("would act"). lightsim2grid's `_detect` means something weaker, "a
violation exists": it feeds `get_physical_violations` and acts on nothing. For three loops the
two differ, and the GPU detection must use the stricter test:

| loop | where `_detect` and the check differ | what the GPU detection adds |
|---|---|---|
| ReactiveLimits | not detected, yet `UNSTABLE`: a frozen bus whose limit moved (its capability curve read at a new target P), `ReactiveLimitsLoop.cpp:516-526` | a "limit moved" test: `frozen_q` against the limit at the current target P |
| ReactiveLimits | detected, yet `STABLE`: `max_pq_pv_switch` or "keep the strongest PV bus" blocks every switch | both rules; they need the per-bus PV→PQ switch count, part of the row's loop state (B.3) |
| DistributedSlack | detected, yet `STABLE`: the units move by less than `moved_fraction · p_residue_eps` (the move is kept), or `FAILED` on a residue that cannot be shared, `DistributedSlackLoop.cpp:155-178` | run `distribute` within the detection: it is cheap and the action needs its result anyway |

Two more rules keep the routing exact:

- **Quiet transitions of the skipped loops.** A loop the router passes over can still change
  its state. `TransformerVoltageControl` at its INITIAL stage with nothing outside half a
  deadband switches to COMPLETE and returns `STABLE` (`TransformerVoltageControlLoop.cpp:478-480`),
  and never acts again. The router applies such transitions to every loop it skips: here a
  per-row stage flag. Without it, TransformerVoltageControl could act later, once
  ReactiveLimits moved the voltages, where lightsim2grid does not.
- **A `STABLE` that changed something.** DistributedSlack's small move changes the targets, and
  so the limits ReactiveLimits reads at target P, after the later loops were detected. The
  difference is at tolerance level. For bit parity, apply the move and detect the loops after
  it again before routing; V is unchanged, so this is cheap.

### Synchronous and asynchronous detections

- **Synchronous** detections (device kernels) are evaluated together and read in loop order.
- **Asynchronous** ones (a decision run off the main stream or on the host, such as
  DistributedSlack's `distribute`, host-side today) must all have finished before the row is
  routed: the row waits for every detection before its new state is decided.

### Per-row driver state

The routing needs a small per-row state, parked with the row (B.3):

- the current loop, `last_unstable`, the pass number;
- the per-loop count of `UNSTABLE` results (lightsim2grid's `ctx.iteration`), which the
  first-visit loops below read;
- the outer-iteration count and the Newton iterations of the current pass;
- the loops' own state: TransformerVoltageControl's stage, DistributedSlack's current set-points,
  ReactiveLimits' per-bus state and switch count.

### Loops that act without a violation

These are encoded as "act at count 0, then follow the state":

- `PhaseControl` returns `UNSTABLE` at count 0 whenever a phase shifter exists
  (`PhaseControlLoop.cpp:119-143`);
- `ShuntVoltageControl` acts at count 0 if any visible controller is handled
  (`ShuntVoltageControlLoop.cpp:163-185`);
- `TransformerVoltageControl` runs a stage machine (INITIAL / CONTROL / COMPLETE). Its INITIAL
  step acts only if some group is outside half its deadband, not regardless.

### Unrealistic voltage

The unrealistic-voltage check (on the buses whose magnitude is a Newton unknown) runs on every
solve. In robust mode it runs only after the last loop able to fix it, ReactiveLimits or
TransformerVoltageControl, with its flag reset at every pass (`OuterLoopAlgo.tpp:99-106,
147-155`).

## B.3 Decide and patch on a second stream; park the row

- **Streams.** The checks run on the main stream at eviction. Deciding and patching run on a
  second stream, ordered with CUDA events, so a parked row is loaded only once its patch is done.
  In the first version the decision runs on the host (B.0, item 7): only the routed rows' data
  are copied back, and the patch is uploaded on the second stream. Otherwise the host only moves
  row indices between queues and reads counts once per round.
- **Parking.** A row waiting for its patch **leaves its slot**, which is refilled. Keeping it
  frozen in place would idle the slot for a round. The parked state is:
  - its V;
  - no other Newton unknown. With the single slack of B.5 there is no `slack_absorbed`, and the
    controller Q enters F linearly, so the Newton iterate does not depend on where it starts
    (as for today's warm start). Later, α / ρ / B would need parking: they are nonlinear;
  - its **control state** (see B.7);
  - its driver state (B.2): current loop, `last_unstable`, pass, per-loop count of `UNSTABLE`
    results, outer iterations, Newton iterations;
  - its loops' state. For the four loops of B.0 this is, in lightsim2grid:
    - `DistributedSlack`: the units' current set-points (`current_mw_`), and the moved targets
      in `OuterInjections`;
    - `AcHvdcAcEmulationLimits`: the regime of each line (`frozen_dir`, -1 / 0 / +1);
    - `VoltageMonitoring`: which monitors are switched on (`switched_on_`);
    - `ReactiveLimits`: per controller bus, `state`, `realistic`, `frozen_q`, `nb_pv_pq`,
      `target_q`, `target_p`, `monitor_svc` (`ReactiveLimitsLoop.hpp:98-153`);
    - the controls' own state (`OuterControls.hpp`): PV/PQ of each bus, held controllers,
      standby SVCs, HVDC regimes, `reset_vm`.

    The loops left out add their own: `TransformerVoltageControl`'s stage, ratios and frozen
    generators, the taps and the sections.
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
- the driver parameters (`OuterLoopDriverParams`: `max_outer_iterations`, the robust mode
  `voltage_remote_control_robust_mode`, the realistic band `min/max_realistic_voltage`,
  `min_nominal_voltage_realistic_check`, `resume_outer_state`; `OuterLoopDriver.hpp:26-38`);
- the plans built by the check headers the loops themselves build on: `BusQCheck` (ReactiveLimits),
  `HvdcPCheck` (AcHvdcAcEmulationLimits), `SvcStandbyCheck` (VoltageMonitoring).
  `GenPCheck` and `GenPvReleaseCheck` serve only the batch physical checks: ReactiveLimits runs
  its own release test, and the robust mode's `LOW/HIGH_VOLTAGE_REMOTE_CONTROL` come from its
  own detection;
- parity tests, row by row, against its `NROuter_*` batches (B.0, item 6). Its per-solve trace
  of decisions (`OuterLoopStats::decisions`) is the oracle when a row differs.

lightsim2grid's loop classes themselves are not called. Calling them per row on the host
(through an `OuterControls` subclass) would give parity by construction, but would need each
row's V on the host at every outer iteration, and the AssistFlux prototype measured host-side
checks at 30 s out of 68 s before moving them to kernels.

The first version still decides on the host (B.0, item 7), but only for the rows the GPU
detection routed, with only their data: the checks themselves, the 30 s above, stay on the
device. The decisions move into kernels later. Even then, a rare case can go to the host the
same way: a truncated report, or a group configuration the kernels do not handle.

## B.5 Device mechanisms, per loop

The table follows OpenLoadFlow's default order. The first four loops are on by default, and
are the scope of the first version (B.0, item 1); the last row is left to an issue.

| loop | what it changes in the solve | already in gpusim2grid | new |
|---|---|---|---|
| DistributedSlack | Sbus values (units' targets; a non-regulating unit's target Q follows its limits) | per-row Sbus correction, verbatim `distribute` port (`slack_redistribution.hpp`) | apply it again from the cumulative mismatch of the slack bus |
| AcHvdcAcEmulationLimits | the droop regime of each line | droop kernels, the HVDC P check | per-row regime (today shared by the whole batch) |
| VoltageMonitoring | an idle standby SVC's "Q = 0" row becomes its voltage row | held controllers, per-row `v_set` (`GenVsetSlots`), the standby check | enrol the monitors as held controllers in the ledger |
| ReactiveLimits | PV↔PQ, group controllers held at a limit, V = 1 restart | `set_pv_pq_switches`, `set_vc_controller_pins` / `_releases`, `Contingency::pinned_buses`, `vm_reseed` | applied as control state at re-entry instead of per-run inputs |
| PhaseControl, TransformerVoltageControl, ShuntVoltageControl (off by default) | new unknowns α / ρ / B; the transformer or shunt Ybus entries patched every Newton iteration | per-slot Ybus values | `BranchControl` / `ShuntControl` as kernels: the Ybus patch, dS/dα and dS/dρ, the custom rows, a limiter's current |

The Jacobian pattern comes from lightsim2grid's `NROuter_*` system: the union of everything its
loops reserve.

### The slack: single slack, as lightsim2grid

*Reviewed 2026-10-15 against lightsim2grid `dev_outer_batch_perf`.*

With the outer loops on, gpusim2grid uses a **single-slack inner Newton**, as lightsim2grid's
`NROuter_*` algorithms do, and shares the slack only through the `DistributedSlack` loop. A
distributed slack inside the Newton *and* the outer loops together is not part of
lightsim2grid yet. It is only a proposal there (`docs/dev_notes/slack_handling_refactor.md`,
"A distributed-slack Newton inside `NROuter_*`": the Newton shares by `slack_absorbed`, and the
loop only caps units at their limits and redistributes what they no longer take). Follow
lightsim2grid when it lands; until then, no parity reference exists for it.

What "single slack" means changed with lightsim2grid commit `148e2e52`
(`OuterLoopAlgo.tpp:47-58`):

- only the reference bus, `slack_ids(0)`, is P-free, with its angle as the datum;
- every **other** slack bus is moved into `pv` and **keeps its P equation**. It also gets a Vm
  unknown and a Q equation when no local regulator pins its voltage
  (`VoltageControlPlan::free_vm_slack_buses`). Left as slack buses, they would take whatever P
  closes the balance, and a target the loop moves onto one of their units would change
  nothing.

`use_distributed_slack=False` does **not** reproduce this. `drop_multislack_augmentation`
(`ls2g_bridge.cpp:344-372`) removes the P equation of *every* slack bus, which is
`NRSing_*`'s layout (several P-free buses at fixed angles). It matches `NROuter_*` only on a
grid with one slack bus. So:

- **take the ledger from lightsim2grid's `NROuter_*` solve**, not from a transform of the
  `NR_*` ledger. On an `NROuter_*` grid, `get_J_solver()` and the ledger's bus maps are
  forwarded to the inner Newton (`OuterLoopAlgo.hpp:190-215`), so `extract_ledger_data`
  reads them as they are. This also brings the union pattern and the other slack buses' Vm
  columns, and gives parity by construction. The `BranchControl` / `ShuntControl` slots are
  not readable yet (B.6);
- with the outer loops on, **refuse** `use_distributed_slack=True` and `redistribute_slack`,
  as lightsim2grid's batches refuse `redistribute_slack` with an `NROuter_*` algorithm
  (`BaseBatchSweep.cpp:504`: it needs an algorithm whose `distributes_slack()` is true). The `DistributedSlack` loop is
  OpenLoadFlow's version of that pre-pass;
- the MultiSlack kernels (`slack_col < 0`) and the per-row slack weights (`slack_w_stride`) are
  then simply off. The `DistributedSlack` loop acts only on Sbus values (its row above).

## B.6 What waits on lightsim2grid

*Reviewed 2026-10-15 against `dev_outer_batch_perf`: most of the list has landed.*

- **The ledger of an `NROuter_*` solve: partly available.** `get_J_solver()` and the bus maps
  are forwarded to the inner Newton (`OuterLoopAlgo.hpp:190-215`), so the bridge can read them
  (to verify, B.0). The `BranchControl` / `ShuntControl` columns and rows, and those of the
  held and standby controllers, are private with no getter (`NRSystem.hpp:1776-1789`). The
  first ones matter only for the loops left out; the held / standby rows are needed for
  `VoltageMonitoring` and `ReactiveLimits`.
- **Row-aware loops: half done.** `OuterRowView` / `OuterGridView` hand a row's trips,
  islanded buses, disconnected units and targets to the loops, and
  `DistributedSlackLoop::_initialize` goes through it now. `BranchControl::_add_entry` still
  reads the transformer's status off the grid (`NRSystemBranchControl.cpp:43-48`): it matters
  only for the loops left out.
- **Detection realigned on OpenLoadFlow: done.** The new types are 16-20 and the
  `CONTROL` category (B.0, prerequisites). `UNREALISTIC_VOLTAGE` and `REACTIVE_LIMIT_MOVED`,
  named here before, do not exist. An unrealistic voltage is the `unrealistic_state` flag
  (B.1), and a moved limit is a decision of ReactiveLimits, not a violation type.
- **The starting state of a row: done.** A batch row seeded from the "n" solve resumes from a
  copy of that solve's loops: frozen buses, SVCs switched on, HVDC regimes, taps and sections,
  the moved targets unless the row sets its own, and V (`OuterLoopAlgo.tpp:44-46`). A TimeSeries
  row chains on the previous one, and any other row starts from the grid. B.0 item 4 follows
  this. Today's workflow (`bake_outer_loops` + `can_be_pv` + detection + the one-shot second
  pass of `reactive_limits_outer_loop`) approximates the default and is kept (B.0, item 5).
- **Still open upstream:** whether an unrealistic stop is `STABLE` or `FAILED` (B.1), and the
  slack refactor (B.5).

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

*State of the code, 2026-10-15 (`dev_outerloops`):*

| item | state |
|---|---|
| 1. inputs apart from the control state | **Not done, and the largest piece of Part B.** The control state is still a per-run input baked into the source when it is built. The `ScenarioSweepBatch` constructor builds the pins, stranded groups, `vm_reseed`, masks and patches of every row at once from `std::vector<Contingency>` (`scenario_sweep_batch.cuh:233-298`). The per-row streams are then gathered by `SegmentGather` from immutable host segments. Changing one row's control state means rebuilding the whole source, or a cold rebuild when the reserved buses change (`scenario_sweep_session.cu:186`). Only the slack correction and weights, `gen_v` and `v_init` can be set on the live source. |
| 2. warm V and warm unknowns | **Partial, and enough.** Warm V works in continuous mode, but in `ScenarioSweepBatch` only (`load_slots`, `scenario_sweep_batch.cu:175-183`). `ContingencyBatch` and `InjectionBatch` have none, and B.0 item 2 needs all three facades: either these sources get it, or those facades go through `ScenarioSweepBatch` (to settle in the plan). No warm unknowns are needed (B.3). |
| 3. check records the scheduler can read | **Partial.** `_solve_continuous` runs every check at eviction (`batch_pf_driver.cu:1667`, `1704-1900`), but the records exist only in the user-facing top-K buffers. The scheduler reads only the residuals. A per-row summary of what was detected is still to add. |
| 4. extensible status and counter | **Done.** `RowStatus` is append-only (`row_status.hpp`), with one counter per row, reset when a slot is loaded. |

Also not done: A.6 item 2. Every continuous run is cold (the sessions rebuild the driver), and
`nb_iter_per_round` / `tol` are not in the driver config snapshot.

*First version, 2026-10-15 (`ScenarioSweepGPU(outer_loops=True)`, CLAUDE.md "Outer loops"):*

- **Item 1** is solved per slot rather than per row: a row stays in its slot from its first
  load to its end, so its control state (`outer::SlotControls`) exists only for the rows in
  flight and is written to dense per-slot device buffers (pins, held controllers, `v_set`,
  hvdc regimes) plus a sparse Sbus correction re-applied over the row's own Sbus. The
  immutable per-row segments are untouched: the reserved-bus pins moved out of the mask
  streams.
- **Item 2** is not needed: a re-solved row keeps its V in its slot (B.3's parking is left
  for when the decisions are asynchronous).
- **Item 3**: the scheduler gathers, per converged slot, V, the computed and scheduled power
  at the buses the loops read and the controllers' Q, and the host runs the loops' exact
  checks on them (B.2's "would act" superset filter on the device is not there yet: every
  converged row comes back, a few kB each).
