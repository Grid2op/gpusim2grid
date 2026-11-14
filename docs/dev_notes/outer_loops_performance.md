# Outer loops on the GPU: where the time goes, and what to change

**Design note, not documentation.** Written on 2026-10-16 from an N-1 benchmark on real
transmission-grid snapshots (`ContingencyAnalysisGPU(outer_loops=True)`, continuous scheduling,
batch sizes 1 to 16000, two GPUs), compared with the same grids without outer loops (OpenLoadFlow
solve + `bake_outer_loops`, one plain batch with the detection on the device, chunked and
continuous). The timers (commits `a2ec4b2`, `12112cb`) and steps 1 to 3 are implemented.
The measured figures stay with the benchmark, outside this repository, as real-grid data does.

## How to read the timings

`BatchTimings` of a continuous run, all wall times:

- **Newton rounds**: `t_spmv`, `t_fill_F`, `t_fill_J`, `t_first_factorize`, `t_refactorize`,
  `t_solve`, `t_update_V` (+ `t_residual`, the residual pass of every round).
- **Slot loads / evictions**: `t_tile_V`, `t_tile_Ybus`, `t_patch_Ybus`, `t_tile_Sbus`;
  `t_store_V`, `t_violation_check`, the physical checks, `t_flow_computation`.
- **Scheduler, per round** (`t_schedule`): `t_sched_res_sync` (the D->H of the S residuals),
  `t_oc_gather` (gather of the converged leaving slots for the outer model), `t_oc_decide`
  (`OuterHook::on_solved`); the rest of `t_schedule` is the decision loop and the H->D of the
  eviction map / load list.
- **Outer model, apart**: `t_oc_controls` (`_oc_write_controls`), `t_oc_load` (`OuterHook::begin`,
  `_oc_setup`, `OuterHook::load` on every (re)loaded slot, phantom slots included).
- **What nothing covers**: `t_chunks_total_wall_ms` is the *sum* of the phases above, not a wall
  clock. `t_driver_solve_wall_ms − t_chunks_total_wall_ms` is the driver's untimed part,
  `t_session_run_wall_ms − t_driver_solve_wall_ms` the session's (source and driver build, cuDSS
  ANALYSIS of a cold run, the outer rows' inputs), `compute()` minus `t_session_run_wall_ms` the
  Python side. On the benchmark all three are negligible: the phases account for the run.

Per row, `OuterRowResult.nb_nr_iterations` is the Newton iterations summed over every inner
solve; `get_row_iterations()` is only the last inner solve's count with the outer loops (the
counter restarts on every re-solve), all of them without.

## What the benchmark shows

1. **The outer loops do about twice the Newton work** of the baked single solve (the re-solves the
   loops trigger). That part is intrinsic: at full occupancy it alone sets a floor of about a third
   to a half of today's best outer-loop time.
2. **The GPU cost of a round does not depend on the mode**: at equal slot count the outer loops and
   the plain batch pay the same refactorization + solve per round. The gap is in the number of
   rounds and in host work.
3. **The tail.** Every round refactorizes and solves all S slots (`_solve_continuous` runs
   `run_nr_loop` on the whole batch whatever `n_live` is). Once the queue is empty the occupancy
   collapses: with S at least the number of rows, the run lasts as many rounds as the longest row
   has Newton iterations. Two things make that row long: a row whose inner solve ends `MaxIter`
   uses the whole `nb_iter` budget (and may come after other solves), and the natural spread of
   the iteration totals (a long tail even among the converged rows). Replaying the schedule from
   the per-row iteration histogram, in random row order, reproduces the measured round counts
   within about 10 %, so the replay can price a scheduler change before it is built.
4. **cuDSS 0.8.0 uniform batch has a cost cliff at 16 matrices**: one round of 8 matrices is cheap,
   16 is more than an order of magnitude dearer, and from there the cost barely grows up to about
   a thousand. NVIDIA confirmed it as a bug:
   https://forums.developer.nvidia.com/t/surprising-behaviour-for-cudss-uniform-batch-mode/382920.
   Consequences while it lasts: shrinking the batch to a few hundred slots saves little; only
   handles of fewer than 16 matrices are cheap (15 included, see step 4); a few small solves in a
   row beat one large batch up to several hundred live rows.
5. **The operational violation check is the first host-visible cost.**
   `check_limit_violations_kernel` runs one thread per row: that thread scans every branch and bus
   and keeps the most severe `capacity` records per type by sorted insertion in global memory, so a
   row with many violations shifts records on every insertion. It costs tens of ms per launch and
   is launched in every round where a row leaves. It dominates every mode (baked too) at small
   batch sizes and is a noticeable share at the best one.
6. **`t_oc_load`** (the outer model's per-row setup) does not depend on the batch size, grows a
   little with it (phantom slots are reloaded at the end), and is the largest host item at the best
   batch size; it is roughly twice as large on a slower host CPU.
7. **`t_oc_gather`** costs a few ms per call: large with few slots (one call per converged inner
   solve), negligible from a few hundred slots.
8. **Separate correctness issue, T4 only**: at batch sizes 128 and 256 a large share of the rows
   diverge, in both modes and reproducibly; the other sizes and the other GPU are fine. It looks
   like the known cuDSS batch NaN issue (CLAUDE.md, `use_distributed_slack`): rerun those sizes
   with `reordering_alg='none'` to check.

## What to change, in order

### 1. The violation check, one block per row

*Done: `check_limit_violations_kernel` (`violation_kernels.cu`) as described below, with an 8-bit
MSD radix select, a per-slot global scratch for the collected records, and one `__noinline__`
evaluator per element type (the passes must agree on every severity to the last bit). Same records
as before, bit for bit on case300. Laptop timing on case300 (400 N-1 rows): the whole run's
check went from 13–1043 ms (K = 3–2000) to 6–11 ms. On the real snapshots it is an order of
magnitude or more cheaper in the outer-loop runs.*

Threads of the block split the branches and buses; the per-type totals by block reduction. The
records must stay what they are: per type the `capacity` most severe (`|value/limit − 1|`), ties in
scan order. Scan order is a total order on the candidates (branch index then side for CURRENT, bus
index for the voltages), so a tie broken on that key reproduces it. One way: count per type; if the
count fits, compact the candidates and sort them in the block; otherwise select the K-th severity
(radix select on the bits of a positive float), keep the candidates above it and the ties by key.
Pinned by the existing limit-violation parity tests and their ranking check (`_assert_ranked`). The
physical-check kernels share the `TopKByType` pattern but are not costly in this benchmark (with the
outer loops they are refused anyway).

### 2. `t_oc_load`

*Done. Profiled on the CPU alone, through `_OuterModelHarness` (its `set_branch_off` now also
takes the rows' masked buses) on a snapshot saved with `LSGrid.save_binary`, under callgrind (a
release build unstripped: `-C cmake.build-type=RelWithDebInfo` with `-O3`; `pybind11_add_module`
strips every other build type). None of the suspects below: `collect_units`, which scans every
generator, ran twice per row and once per phantom (82 % of a load), and `build_bus_q_plan` for a row
masking buses (two thirds of what was left). Now: the participants are collected once (a row
changes them only by disconnecting units), the groups built once per load, the resumed targets
taken once for the rows that set none, a masked-only row's plan is the grid's filtered
(`rl_plan_masked`, checked field by field against `build_rl_plan` on every masked row of the
snapshot), and `materialize` merges its two maps without a copy. A load is ~4x cheaper, a phantom
~40x; `t_oc_load` of the outer contingency analysis ~4.6x. Same statuses and iteration counts on
every row, V to cuDSS' run-to-run rounding. What is left is mostly the copy of the resumed
`target_delta` map and of the filtered plan's entries. The benchmark snapshots have no regulating
phase shifter, so `PhaseControlLoop::shifters`' union-find per shifter and per row (lightsim2grid's,
called by `load`) was not measured: a grid with many of them would need a cheaper `regulates`.*

Profile `Ls2gOuterModel::load` first. Suspects, per loaded row: the BranchControl reset (a
`pi_block_at` / `ratio_at` per controlled transformer, a few hundred on a real grid), the
ShuntControl reset, `build_rl_plan` for a row that masks buses, `set_active`. Most of it is the
grid's own state, the same for every row that opens no controlled transformer: compute it once in
the model and copy it. A phantom load (row −1) needs none of it.

### 3. The tail on a 15-matrix cuDSS handle

*Done (`tail_max_groups`, default 16, and `tail_group_size`, default 15, on the three batch facades;
`BatchPfDriver::_compact_tail`, `TailSolve` in `contingency/driver.cuh`).* Once the queue is empty
and the live rows fit in `tail_max_groups` groups of G = `tail_group_size` slots, the live rows of
slots past `G · groups` move into phantom slots below it, and every later round runs on those slots
only:
- the Newton kernels and a prefix of the block-diagonal SpMV (`_build_spmv`);
- the residual pass and the evictions;
- the linear solve, on one more cuDSS handle analysed for G matrices. It is pointed at each
  contiguous group in turn (`CudssBatchSolver::bind_slice`: `cudssMatrixSetValues` on A, x and b),
  then refactorized and solved. Nothing has to survive between groups, since the policy refactorizes
  every iteration; the tail is off for the other policies.

Moving a row carries what its next iterations read or carry:
- V, its Ybus row (the contingency patches and the BranchControl / ShuntControl stamps, with their
  `applied` record);
- `slack_absorbed`, the controllers' Q;
- the source's per-slot rows (`move_slots`: Sbus with its corrections, slack weights, `v_set`);
- the outer controls, device and host mirrors;
- the outer model's slot (`OuterHook::move`, which re-points the row view's `masked` list);
- the host's iteration counter and slot table.

J, F, dx and Ibus are rebuilt every iteration. The masks follow the new slot table at the next
bind. A moved row's result is its own up to cuDSS's rounding: statuses, iteration counts and outer
decisions are unchanged, V agrees to 1e-14 on the converged rows (`tests/python/test_continuous_tail.py`,
and the whole continuous / outer-loop suite passes with groups of 1).

Checked first with `benchmark_cudss_batch_raw(n_groups=...)`, on a laptop GPU only, which says
nothing of the target hardware's timings, only of the mechanism:
- **Cycling is free.** One 15-matrix handle cycled over 16 groups costs the same per group as one
  group.
- **Below 16 matrices the cost grew linearly with the count** there, so 15 per group costs no more
  per matrix than 8, in half the calls. To re-measure on the RTX 6000 and the T4, together with
  `calibrate_tail_max_groups()` on a real snapshot.

The replay of the benchmark files (extrapolating the cost of 15 matrices from 8) puts the tail at
several times cheaper at the best batch size on both GPUs, and most of the rounds of a run are tail
rounds there. The gain is flat beyond about 16 groups, whence the default.

`tail_max_groups` is a fixed knob. `calibrate_tail_max_groups()` measures what it should be on a
GPU and a grid: REFACTORIZATION + SOLVE of the base case's J on a handle of S matrices and on one of
G, `max_groups = floor(t_full / t_group)`. That counts cuDSS only, so it is conservative.

The 15 is cuDSS 0.8.0's cliff (point 4 above). The compaction itself stays useful without the
cliff: a fixed cuDSS would make G a handle of `n_live` matrices, or one cheap per-matrix size.

Later, if wanted:
- Call the calibration inside the driver: time the first round's full solve and one group solve on
  the real J, and decide the switch from that, instead of the fixed knob.
- Re-check the per-group cost against `n_groups` handles on separate streams; the
  `n_handles` / `handle_launch` options of the raw benchmark measure it.

### 4. Small queues under the 16-matrix cliff (to measure first)

*Proposal of 2026-10-18; the benchmark and the replay variant exist, the driver change does not.*

The cliff sits between 15 and 16 matrices: a handle of **15** is as cheap as one of 8. Step 3
therefore wants handles of 15, not 8. The idea goes further and uses small handles from the start,
not only in the tail: **nb_th queues of 15 slots, each with its own cuDSS handle**. A row that
enters the batch joins a queue with a free slot, and a queue runs when it is full.

- **"Runs when full" is the steady state of continuous scheduling.** A row that leaves frees a
  slot of its queue, and the same round boundary refills it from the global queue. So every queue
  stays full while rows are waiting. The rule only matters in the tail, where a queue never fills
  again: it must either run partly empty (a whole 15-matrix handle for its live rows; no slot is
  moved) or merge with the other partial queues (`ceil(n_live / 15)` handles; slot moves, the same
  state copy as step 3). The replay prices both (`queues` and `queues+merge`).
- **No slot has to move in the steady state.** J, F and dx are already slot-contiguous, so handle h
  is just analysed on slots `[15h, 15h + 15)` of the existing buffers (what
  `CudssBatchSolver::bind_slice` does for one cycled handle). The Newton kernels keep running over
  all S slots in one launch; only the linear solve is split.
- **What decides it: can nb_th small handles beat one large one per round?** Run one after the
  other, they cost `nb_th · c(15)` against `c(15 · nb_th)`. With the cliff (c(8) ≈ 1.5 ms, c(16..1024)
  nearly flat around 80 ms on the RTX 6000), serial handles win up to about 50 of them (750 slots),
  as long as c(15) ≈ c(8). Beyond that, they win only if the handles overlap on the GPU. A small
  batch's REFACTORIZATION + SOLVE is mostly launch latency and cuDSS's own host work. It may not
  overlap on separate streams driven from one host thread when cuDSS synchronises internally. One
  host thread per handle, each with its own stream ("threads" in the user's sense), can overlap. A
  cycled handle (`bind_slice`, step 3) costs the same as serial handles but cannot overlap: a
  handle's data object holds the factors of one slice at a time.
- **The "several queues" objection below does not carry over.** That objection was about a few large
  queues: a slow row keeps a whole large batch running, and two batches only help when one cannot
  fill the GPU. A 15-matrix queue held by one slow row costs one cheap handle, and a 15-matrix batch
  is far from filling the GPU. Each handle has its own ANALYSIS (`handle_setup_ms` measures them all;
  they run once per driver). The factor memory is that of nb_th · 15 matrices, the same as one
  handle of S.
- **With host threads, two designs.** (a) The main thread runs the Newton kernels over S, then
  hands the nb_th linear solves to workers and joins them before `update_V`. The schedule stays as
  it is, and only the solve phase is threaded. (b) Each thread runs its own continuous loop on its
  queue: its own kernels, residual pass and D→H, with a shared, locked row queue. The outer model's
  `on_solved` / `load` would then be called concurrently, so `Ls2gOuterModel` would need per-slot
  locking or one model per thread. Try (a) first; (b) only pays if the residual sync or `t_schedule`
  dominates once the solve is split.

**Measuring it.**
`benchmarks/cudss_small_handles.py` times one round's linear solve (REFACTORIZATION + SOLVE, plus
the rhs reset that stands for the refill of F) on nb_th handles of `--queue_size` (15) matrices. It
compares them with one handle of `nb_th · 15` matrices on the Jacobian of a solved grid
(`--network` as in `outer_loops.py`, or `--binary` from a `.l2gb`). The small handles run
`serial` (one stream), `streams` (one stream each, one host thread) and `threads` (one stream and one
host thread each, every handle running its own rounds). All three go through
`benchmark_cudss_batch_raw(..., n_handles=, handle_launch=)` and are pinned for correctness in
`tests/python/test_cudss_factor_stats.py` (handle h must solve its own slice). Run it on the target
GPUs (RTX 6000, T4; the laptop's numbers say nothing), with `--out`. Then feed the JSON to the
schedule replay (`replay_continuous_schedule.py --handles <json> --launch threads`). The replay
prices `queues` / `queues+merge` against one handle of the same S. A round of k queues costs the
measured k-handle solve plus the rest of the Newton round at `15k` slots (the measured round minus
its cuDSS phases). Caveats: `get_J_solver()` is lightsim2grid's single-solve J, a little smaller than
the outer-loop union pattern (the cliff is structural, the absolute ms a bit low). Use the
`ir_n_steps` of the run being replayed. The replay's one-handle cost at S = 15 is interpolated across
the cliff (8 → 16), so ignore that column on the `nb_th = 1` line. Go if the best launch beats one
handle of the same S per round, and `queues+merge` (or `queues`, which needs no move) beats today's
best S in the replay.

## Not to pursue

- `CUDSS_DATA_UBATCH_MASK` (cuDSS 0.8.0) to skip idle matrices: it does not work yet; re-check on a
  later cuDSS.
- cuDSS factorization / solve algorithm knobs against the 16-matrix cliff: it is a cuDSS bug
  (above), wait for the fix.
- Several large queues of different sizes summing to S from the start: a slow row still lands in
  one at random and keeps that whole batch running; two batches on two streams only help when one
  cannot fill the GPU, which it does at large S; and it doubles the analyses and the factor memory.
  (Many queues of 15, under the cuDSS cliff, are a different case: step 4.)
- A second, smaller driver prepared asynchronously: migrating a row between drivers needs the same
  state copy as a slot move, plus a second set of buffers; the analysis it would hide is small next
  to the tail.
- Lowering `nb_iter` is a benchmark setting, not a code change: it is safe down to the largest
  per-inner-solve count of the converged rows, which the histograms bound from above (they sum over
  the inner solves).
