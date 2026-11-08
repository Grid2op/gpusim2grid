# Outer loops: getting the host out of the Newton rounds

**Design note, not documentation.** Written on 2026-11-08 from an N-1 benchmark
(`../expe_gpus/bench_n1.sh`, "loose" mode: 30 outer / 30 Newton iterations, tol 0.01 MVA,
slack mismatch 1 MW) on an RTX 6000. It ran `linear_solver='own_lu'`, batch 2048 and a tail of
4 x 64, with the outer loops on, against lightsim2grid on one core (KLU, static pivot + min-fill).
It follows `outer_loops_performance.md`, whose steps 1-3 are done, and Part B of
`continuous_batching.md`. Nothing here is implemented yet. The two snapshots are called A and B
below; real-grid data stays outside the repository.

**Constraint (2026-11-08): no extra host threads.** Spreading the host model's per-slot calls over
threads (`outer_loops_performance.md` step 4 (b) and the like) is not an option. The goal is to
remove host work, not to spread it, so every idea below runs on the one host thread.

**Order agreed (2026-11-08).** Make each component cheaper first (4.1, starting with
`on_solved`), before anything asynchronous or on the device. The lazy state with its device filter
and the device decisions (4.2, 4.3, 4.5, 4.6) come later. The overlap (4.4) comes last, if ever:
the tail works against it (see there).

## 1. What a run spends

| | A (8803 rows) | B (9455 rows) |
|---|---|---|
| lightsim2grid N-1 | 34.3 s | 40.2 s |
| gpusim2grid `compute` | 1.98 s | 2.05 s |
| outer iterations / Newton iterations (GPU) | 4440 / 21266 | 2850 / 21001 |
| rounds (of them in the tail) | 41 (29) | 38 (26) |

The breakdown printed by the benchmark lists `schedule` beside `oc_decide` and `oc_gather`, but
**`t_schedule` already contains them** (and `t_sched_res_sync`): see the comment on `TimingEntry
t_schedule` in `timing_utils.hpp`. On A, 0.497 + 0.071 + 0.001 = 0.569 = `schedule`. Without
the double count, and with the ~10 kernel phases' totals added up:

| bucket (wall s) | A | B | what it is |
|---|---|---|---|
| `oc_load` | 0.556 | 0.689 | `Ls2gOuterModel::load` on every (re)loaded slot, phantoms included, + `begin` |
| `oc_decide` | 0.497 | 0.427 | `Ls2gOuterModel::on_solved` on every slot that left a round |
| `oc_gather` | 0.071 | 0.070 | gather kernel + D->H + sync of the converged slots |
| `oc_controls` | 0.039 | 0.043 | `_oc_write_controls`: host packing, H->D, Sbus re-read, sync |
| rest of `schedule` | ~0.001 | ~0.002 | eviction loop, load list, H->D |
| **host model, total** | **~1.16** | **~1.23** | |
| GPU phases | ~0.71 | ~0.71 | solve 0.23, refactorize 0.18, fill_J 0.11, update_V 0.09 (0.05 GPU), residual 0.04, rest |
| session outside the driver | ~0.10 | ~0.10 | source/driver build, analysis 0.03, the outer rows' inputs |

The driver's phases add up to its wall time (1.87 of 1.88 s on A), so nothing large is untimed.

**The GPU is idle almost 60 % of the run.** Each round is: Newton step on the GPU (~17 ms), then
`on_solved` for every slot that left (~12 ms), then `load` for every refilled slot (~13 ms). It all
runs one after the other on one thread, and nothing overlaps.

**Per call the host model is already cheap.** There are about 13k `on_solved` calls (rows +
outer iterations, so ~37 µs each) and about 11k loads (rows + phantoms, ~50 µs each). The cost
comes from the number of calls, not from one slow call. The 4.6x of `outer_loops_performance.md`
step 2 has already been taken on `load`.

## 2. CPU <-> GPU latency: what it costs today, what it would cost

Today every round crosses the bus several times, with a host wait each time:

| crossing | per round | measured total (A) |
|---|---|---|
| residuals D->H + sync (`t_sched_res_sync`) | 1 | 0.001 s (~25 µs each) |
| gather of converged slots: kernel, D->H, sync (`t_oc_gather`) | 1 when a row converged | 0.071 s |
| controls of re-entering / loaded slots: H->D, Sbus re-read, sync (`t_oc_controls`) | 1-2 | 0.039 s |
| eviction map / load list H->D | 1 | in `schedule` |
| **every timed phase**: `CudaTimer::stop_ms` calls `cudaEventSynchronize` | ~10 | spread over all phases |

So **the transfers and their syncs cost ~0.11 s per run at ~40 rounds, about a tenth of the
host model's compute**. The estimate given before this note (0.8-0.9 s once the host is out of the
way) was GPU 0.71 + these ~0.11 s, but did not say so. Latency becomes a real item in two cases:

- **More rounds.** Latency scales with rounds, not rows: tight mode, smaller batches, or a longer
  tail. Recheck it on the tight-mode runs.
- **Any overlap.** The per-phase timers' `cudaEventSynchronize` makes the host wait for the GPU
  ~10 times per round. While they are on, no host work can run under a Newton step. They also hide
  launch overhead in the wall time (`update_V`: 0.090 wall vs 0.050 GPU). An overlap scheme needs
  a timing mode without per-phase syncs: events read at the end of the round, or only
  `t_round_wall`.

The own LU itself does not sync in a round. It syncs only at setup and in `clamped_counts()`, and
that is read only under `GPUSIM2GRID_COUNT_PIVOTS`. Check cuDSS and the rest of the round with
`nsys` before an overlap is built: one hidden `cudaStreamSynchronize` is enough to serialise it.

**Moving decisions to the device does not remove latency by itself.** As long as the host
schedules rounds, there is at least one D->H + sync per round (the residuals, or a count of rows
leaving). That one costs ~25 µs, which is nothing. What costs is moving per-slot payloads and doing
work on them on the host. Removing the last per-round sync is idea 7, and it only pays when the
rounds are many.

## 3. What to measure first

Without these, the gains below are guesses.

1. **Which loop decides what, how often.** Run with `keep_outer_trace=True` and make a histogram
   of `decisions` over rows: per loop, the number of UNSTABLE results; the number of rows with
   `nb_outer_iterations == 0`; the number of outer iterations per acting row. This orders ideas 4
   and 6. The guess is that most rows never act: 4440 outer iterations over 8803 rows, at most
   5 per row.
2. **Profile `on_solved`** the way `load` was profiled (step 2 of `outer_loops_performance.md`):
   callgrind through `_OuterModelHarness` on a `save_binary` snapshot, RelWithDebInfo build. Split
   it into `check` per loop, `materialize`, `finish`, `adjoint_info`.
3. **Split `on_solved` time into "Resolve" calls and "Done" calls.** These are two counters in the
   driver loop.
4. **An `nsys` timeline of one run**, to see the host gaps and confirm the syncs listed above.
5. **The tight-mode runs**: more Newton iterations per row means more GPU time per round, and
   changes the host/GPU ratio.
6. **Host and GPU time per round, steady state vs tail.** Today there are only run totals. 29 of
   the 41 rounds are tail rounds on few rows, and the tail may hold most of the run. This split
   decides whether 4.4 can work at all. Add a per-round record (live rows, GPU ms, `oc_decide`,
   `oc_load`) next to the totals.

## 4. Ideas, in order

### 4.1 Trim the host model (no design change)

Cheap and exact. Expected: tens of percent of `oc_decide` and some of `oc_load`.

*First pass on `on_solved`, 2026-11-08 (uncommitted).* Measured with `GPUSIM2GRID_OC_PROFILE=1`, a
new diagnostic that prints wall time and calls per part of the model on stderr when the model is
destroyed, and with callgrind (`--toggle-collect='*Ls2gOuterModel::on_solved*'`, RelWithDebInfo
build). Snapshot: the `.l2gb` of `outer_loops_performance.md` step 2, 10235 N-1 rows, ~600
ReactiveLimits buses per row, laptop, the bench_n1 loose settings. Before the changes,
`on_solved` was 70 % `check_rl`, 14 % `finish` and 12 % `check_hvdc`. What was done:

- **`check_hvdc` / `check_vm`**: the `Zero(n_bus)` vector built on every call (all of
  `check_hvdc`'s time) became a persistent zeroed scratch; only the entries written are reset.
  lightsim2grid's own loops have the same pattern (the "memset / malloc on the hot path" of its
  `speed_optimisation.md`).
- **ReactiveLimits limits**: each `RlBus` carries the generator part of its limits computed at
  the base targets, in the same order, so bit for bit. They are valid for the grid plan and its
  masked copies, on a row that sets no generator target. `move_gen` marks the buses of the
  generator it moves, and only those are recomputed (86 % of the bus evaluations hit the base
  sums). The recomputation goes through a per-generator memo of the last (target P -> limits),
  with `gen_override` inlined.
- **`std::map` -> a sorted flat map** for a row's `target_delta` / `freeze_q` (same keys, order and
  sums). This made `materialize` 2x cheaper and the resumed targets' copy in `load` a memcpy.
  There is also a per-bus frozen flag, so `mis_bus` searches only frozen buses.
- **`finish`**: the slot's dense generator targets are moved into the result instead of being
  re-packed as (id, value) pairs (6296 per resumed row). `RowResult::gens_target_p_mw` is now
  dense by generator id (NaN: not moved), `gen_ids` is gone, `gen_target_p_mw(gen_id)` reads one generator and `gen_targets()` gives the pairs.
  The Python `gen_target_p` is unchanged.

Results: `oc_decide` 435 -> ~268 ms (-38 %), `oc_load` ~778 -> ~680 ms as a side effect. Statuses,
outer and Newton counts and NaN pattern are identical on every row. V differs by at most 6e-16,
the GPU's own run-to-run noise (two runs of the same code differ as much). The outer-loop suites
give the same result as before the change (one failure on `main`'s code too:
`test_contingency_analysis_redistribute_slack[True]`).

What is left in `on_solved` (~255 ms): `check_rl` ~14 µs per call (~24 ns per bus, mostly
`mis_bus` and data-dependent loads; nothing large left to cache), `materialize` 25 ms, DS 15 ms,
`finish` 15 ms.

*Second pass, on `load`, 2026-11-08 (uncommitted).* `load` was ~70 µs per row, and 31 % of the rows
(those that mask buses) paid most of it: ~59 µs copying the ReactiveLimits entries, ~59 µs scanning
every generator for the DistributedSlack units, and freeing those entries at the next reset. The
`units0_` copy itself is ~1 µs (~270 units). What was done:

- **A masked row shares the grid's `BusQPlan`** (`RlPlan::plan` is a `shared_ptr`, `RlBus::entry`
  indexes it); only its filtered `RlBus` list is its own.
- **A masked row filters `units0_`** by its masked buses instead of running `collect_units`.
- **`RlBus` is split**: the static part stays in the plan, shared by the slots, and a slot holds a
  32-byte `RlBusState` per bus. The capability-curve slopes moved to a per-slot flat map keyed by
  bus index, so the adjoint still reads them in bus order.
- **`RlPlan::state0`** holds the buses' target P at the base targets; `init_loops` only sums the
  buses a row moved or that have storage units.

Results: `load` ~70 -> ~22 µs per row, `oc_load` ~680 -> ~226 ms. `check_rl` also went from ~15 to
~11.5 µs per call, `oc_decide` ~268 -> ~225 ms: the static data no longer has one copy per slot.
Statuses, outer and Newton counts and NaN pattern are identical on every row, max |dV| 5e-16
against the code before both passes.

What is left in `load` (~22 µs): DistributedSlack init ~4 µs (lightsim2grid's `lost_p_to_share_mw`
scans every generator, on rows that have nothing to find), the resumed targets' copy ~4 µs (one
vector per row, kept in the results), the reset ~2.5 µs, `init_loops`' ReactiveLimits part ~2.7 µs.
No single item stands out any more.

- **`adjoint_info` runs on every row that finishes** (`finish()`, `ls2g_outer_model.cpp`), even
  without gradients. Gate it on the adjoint being requested (`keep_outer_adjoint_`). The model
  needs an option for this.
- **`finish()` builds the user-facing `RowResult` eagerly.** It walks every controlled transformer
  (with a `pi_block_at` for each moved one) and every shunt controller. Instead, keep the row's
  final positions as flat arrays and build the result in `get_outer_results()` /
  `row_trafo_blocks()`, outside the timed loop.
- **`load` copies the resumed `target_delta` map and the filtered plan's entries** (what step 2
  left). Make them flat vectors, or reference the shared grid version copy-on-write: most rows
  never change them.
- **`load` resets a dozen per-slot vectors with `assign`.** Copy-assign one `SlotState` template
  prepared once per run (grid defaults, resumed state) and then apply the row's deltas.
- **The gather conversion loop** (`gbuf`, float -> double per converged slot) is small; fold it
  into `RowView`.

### 4.2 A lazy host state: a row the loops never touch never reaches the host

This idea removes the most CPU without porting decisions.

`load(slot, row)` is a pure function of the row's inputs and the grid, so building it late gives
the same result. Split a row's life in two:

- **Fast path, no host involvement.** A fresh row's slot controls are the grid's defaults plus
  the row's deltas (masked buses, units off, `gen_v`, released buses). They are written **by a
  kernel** from per-row inputs already on the device, not packed on the host by
  `_oc_write_controls`. Its counters (`total_nr`, first-solve flag) live per slot on the device.
- **Slow path, on first need.** When the device filter (4.3) cannot prove that no loop acts on a
  converged solve, the host builds the slot's `SlotState` then (today's `load`, unchanged) and
  calls `on_solved` with the gathered data.

A row that converges with nothing to do is finished on the device: status `Converged`, outer
iterations 0, Newton count from the device counter. Its `RowResult` is the grid's (resumed)
defaults and is written in bulk at the end. It costs no `load`, no `on_solved` and no gather.

Expected, if 3.1 confirms that most rows never act: most of the ~11k loads and up to the ~8.8k
"Done" `on_solved` calls disappear. That is roughly 60-75 % of the 1.1 s, with the exact figure
depending on the histogram.

### 4.3 The device "nothing acts" filter (`continuous_batching.md` B.2)

This is what 4.2 needs. It has to be a **conservative superset** of "some loop would return
UNSTABLE or FAILED". A false positive costs one host call (today's cost); a false negative breaks
parity. The filter can therefore start crude and be tightened loop by loop, tracking the
false-positive rate.

For each loop, what the device must prove on a converged solve of a fresh row (resumed or not):

| loop | proves "STABLE, nothing changed" when | notes |
|---|---|---|
| DistributedSlack | slack mismatch below the threshold under which the loop does not move units | the "STABLE but moved" case (B.2 table) must go to the host |
| AcHvdcAcEmulationLimits | no droop line past its pmax in its regime | already a device check (HVDC P) |
| VoltageMonitoring | no standby SVC's bus outside its thresholds | already a device check (standby) |
| ReactiveLimits | every PV bus inside its capability by a margin, no frozen bus whose release test triggers, no frozen bus whose limit moved | capability at the row's target P: a piecewise-linear curve per machine, tabulated on the device; a resumed row with frozen buses goes to the host until this is proven exact |
| PhaseControl | no regulating phase shifter visible to the row | it acts at count 0 whenever one exists, so such grids always go to the host on the first solve |
| TransformerVoltageControl | INITIAL stage with nothing outside half a deadband | that case is a quiet INITIAL->COMPLETE switch: one per-slot stage flag on the device |
| ShuntVoltageControl | no visible handled controller | acts at count 0 otherwise |
| unrealistic voltage | flag clear | already computed on the device (`oc_unrealistic_kernel`) |

The routing subtleties of B.2 (first-visit loops, `last_unstable`, passes) only matter after a
loop has acted, and by then the row is on the slow path. The fast path only covers "first solve,
nothing anywhere would act". It is a single, easy-to-state condition, and it is the case that
dominates if most rows never act.

### 4.4 Overlap on one thread: decide one round late

The host work that remains (rows that do act) can still run under the GPU without a second
thread:

1. launch round r asynchronously (kernels + solve, **no per-phase syncs**, see section 2);
2. on the same thread, while it runs, process the rows routed at the end of round r-1: their
   lazy `load`, `on_solved`, the control packing;
3. sync on round r's residuals.

A routed row cannot wait in its slot for a decision taken during the next round. It **leaves and
is parked** (B.3: its V and its control state), its slot is refilled from the queue, and it
re-enters later with its warm V through the load path. Parking was postponed in the first version
because the decision was synchronous. It costs one row of V per parked row on the device, and the
re-entry path already exists (`set_v_init_from_ptr`-style warm V).

A round then costs max(GPU, host) instead of their sum. On its own this caps at the host time
(~1.1 s today), so **it only pays after 4.1-4.2 have brought the host below the GPU time**.

**The tail works against it.** Hiding the host under the GPU needs two disjoint sets of rows at
every moment: one on the GPU, one being decided on the host. That means at least twice as many
rows in flight as either side handles at once. The steady state has them, since the queue refills
every freed slot. The tail does not: 29 of 41 rounds there run few rows, which often all leave
together and are all decided before the next round can start. In those rounds the row that is
being decided is the round's only work, so nothing can overlap it. If, as suspected, the tail
holds most of the run time (3.6), overlap hides only the steady-state share of the host time.
Removing host work (4.1-4.3) helps both phases. Hence the order at the top.

### 4.5 Device `load` for the slow path too

Once 4.2 exists, the slot controls of a re-entering row come from the host model's
`materialize`. For the frequent decisions (PV->PQ of one bus, an hvdc regime flip, a standby SVC
switched on), the device could apply the delta directly, so the full `SlotControls` is not
re-uploaded. This matters only if 3.1 shows many outer iterations per acting row.

### 4.6 Decisions on the device, the frequent ones first (`continuous_batching.md` B.4)

Port only what 3.1 shows to be frequent, and keep the rest on the host through the same route as
the filter. Rough difficulty:

- **Easy:** an hvdc regime flip, a standby SVC switched on, TVC's stage machine.
- **Medium:** DistributedSlack's `distribute`: sort + clamp per row, one block per row, with the
  same ordering as upstream for bit parity.
- **Large:** ReactiveLimits:
  - **Medium** for a PV->PQ switch of a local bus outside a group, which needs per-bus switch
    counters on the device for `max_pq_pv_switch` and "keep the strongest PV".
  - **Hard** for groups, held controllers, capability curves at moved targets and the robust
    mode.
- **Leave on the host:** tap and section changes (BranchControl / ShuntControl dispatch).

This gives up the line-by-line host port of lightsim2grid, the property that makes parity hold
by construction today. Each ported decision needs its own parity test against lightsim2grid's
decision trace. Do it only for what 4.2 + 4.4 leave visible.

### 4.7 A device-resident scheduler (last)

Evictions, the queue pointer (an atomic counter) and the refill list are decided on the device.
The round, including the load of refilled slots, is captured as a CUDA graph. This removes the
last per-round D->H + sync and the launch overhead of ~10 phases. At ~40 rounds it is a few ms,
so it only matters if tight mode or a smaller S multiplies the rounds.

## 5. What to expect

These are rough estimates for snapshot A from the counts above, not measurements:

| state | host model | run (compute) |
|---|---|---|
| today | ~1.16 s | 1.98 s |
| + 4.1 trimming | ~0.9 s | ~1.7 s |
| + 4.2/4.3 lazy state + filter (most rows never act) | ~0.3-0.4 s | ~1.1-1.2 s |
| + 4.4 overlap (host < GPU per round) | hidden in the steady state only | ~0.85 s at best |
| floor: GPU phases + transfers + session | 0 | ~0.8 s |

The 4.4 line is optimistic: it hides only the host time of steady-state rounds (see the tail in
4.4). The floor is reached only by removing host work, not by hiding it. That is about 40x lightsim2grid's single core, against ~17x today. Below that floor the only lever
is the GPU itself: its rounds (41, of which 29 are tail rounds) and its per-round cost.

## 6. Not to pursue

- **More host threads**: excluded (see the constraint at the top).
- **Porting the whole host model to the device at once**: this means seven loops, plans built by
  lightsim2grid code, branchy per-row work that suits a GPU poorly, and the parity-by-construction
  property lost. Port by measured frequency instead (4.6).
