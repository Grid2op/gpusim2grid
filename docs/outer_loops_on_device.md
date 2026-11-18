---
orphan: true
---

# Outer loops on the device: working document

**Working document for branch `feat/gpu-outer-loop-kernels`. Keep it current.** The goal of the
branch is to port every outer loop to CUDA kernels, behind `outer_loops_backend='device'`, while
`'host'` keeps today's behaviour exactly. All the work (M0 to M6, the M1 kernel fixes included) is
done on this branch and goes in its one PR. Update the status table below whenever a milestone
moves, and replace estimates with measurements as they come in. Started 2026-11-17 from `feat/topology-actions` at `a42e7bf` (PR #41 included).

It follows `dev_notes/outer_loops_host_work.md` (4.1 done there) and Part B of
`dev_notes/continuous_batching.md`. It borrows the design of gpu-security-analysis (GSA below),
whose outer loops already run on the device. The other notes it names are in `docs/dev_notes/`.

## Status

| milestone | what | status | result |
|---|---|---|---|
| M0 | measure: NVTX per outer phase, sync-free timing, PtFige decision counts, feature census | NVTX ranges done; census done (§5, M0); decision counts and sync-free timing pending an idle GPU | PtFige needs all of ReactiveLimits: M2's scope widened |
| M1a | `oc_unrealistic_kernel` one block per slot | done: full suite unchanged (905 passed, the 2 known failures) | timing pending an idle GPU (expected −0.9 s on PtFige) |
| M1b | why one `check_limit_violations_kernel` row takes 1.3 ms | not started | |
| M2a | `outer_loops_backend`, feature scan raising `NotImplementedError`, device tables, per-slot state, `oc_dev_load_kernel` | option + census + refusal done (`tests/python/outer_device/`); tables and state next | |
| M2b | the four checks on the device, evaluate-only, shadow-compared with the host model | superseded by the lockstep harness tests (M2c) | |
| M2c | device sequencing and apply | device model behind `OuterHook` (`device_outer_model.cu`, one thread per slot): 20 lockstep tests pass (case14, ieee300, hvdc, monitoring, groups incl. held controllers, resumed and masked rows); wired into the three facades (`test_device_facades.py`); PtFige N-1, first 500 rows: statuses, outer / Newton counts identical, max dV 7e-13 (device 13.1 s vs host 1.0 s: per-slot transfers, M3) | |
| M2d | outputs, `get_outer_results()`, timings | not started | |
| M3 | the round without the host: one D->H per round, device refill | step 1 done: `DeviceOuterHook` batched `load_rows` / `decide` / `write_controls`, slot-major state pools; PtFige N-1 identical on all 10530 rows (dV 1.4e-12), outer-loop time ~2.6 s vs ~1.5 s host (load 1.24 s, decide 0.83 s, controls 0.52 s); step 2 done: a row that changes nothing loads from a template slot on the device (check mode GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE=1 passes on the whole PtFige N-1): load 1.24 -> 0.60 s, total 6.9 s vs host 4.8 s; the decision kernel one block per slot, the ReactiveLimits buses evaluated in parallel: decide 0.75 -> 0.28 s (device 6.6 s vs host 4.7 s); masked rows load on the device too (the template plus a mask kernel; check mode passes on the whole PtFige N-1): no host-model load left in an N-1 but the template's; persistent buffers. Load-independent counts per PtFige N-1, host backend vs device: H2D 288 MB vs 13.6 MB, D2H 713 MB vs 1.2 MB, cudaMalloc/Free 234/129 vs 187/64, syncs 1291 vs 1194, oc_* host time ~1.5 s vs ~0.67 s. The device model then moved onto the driver's stream (no legacy-stream implicit syncs, no cudaDeviceSynchronize): syncs 1126, oc_load 195 -> 30 ms; same comparison run: device 3.8 s vs host 4.7 s, every row identical. GPU busy in our trace 43 % vs 33 % (load-dependent: the GPU is shared). Then (CUDA-API backtraces, `claude_data/outer_device/scripts/sync_callers.py`): grow-only pinned
buffers, double-buffered and pre-sized segment-gather staging, `par_nosync` fills, amortised device
buffers, batched tail moves, and `GPUSIM2GRID_ASYNC_TIMERS=1` (opt-in: no per-phase timer waits).
With it, PtFige N-1: syncs 1126 -> 447, D2D copies 4535 -> 32, H2D copies 2105 -> 604,
cudaMalloc/Free 187/64 -> 159/31, GPU busy in our trace 43 % -> 67 %.

**Towards ~100 % GPU busy** (what the idle time left is, and what removes it):

| idle owner | what removes it |
|---|---|
| gaps between kernels (~48k of ~8 us, mostly the own LU's ~1000 dependent level kernels per factorization / solve) | tried: the own LU's factorize / solve as cached CUDA graphs (`claude_data/outer_device/own_lu_graphs.patch`, 2026-11-17): host launch calls 49k -> 27k, but the GPU-side gaps and the wall time unchanged (with async timers the host was already ahead; the gaps are the dependent levels' own boundaries). Not kept. What would remove them: fewer dependent kernels (fusing levels, a persistent kernel walking the levels) |
| the round's host turn (decide D2H + sync, evictions, refills) | a device-resident scheduler (§4.7): evictions and refills decided on the device |
| per-phase timer waits | `GPUSIM2GRID_ASYNC_TIMERS=1` as the default for runs that do not read per-phase GPU times (user's call) | | |
| M4 | the rest of the four default loops, by the census | in progress: rows disconnecting generators (generator contingencies, `set_contingency_gens`, strategies tripping one) -- the row's own ReactiveLimits plan mapped onto the grid's (per-slot target Q / target \|V\| / base flag, generator- and controller-off masks; refused if it changes anything else); PtFige generator N-1 (5771 rows, group controllers included) `'device'` == `'host'`, max \|dV\| 1.6e-12. Rows with their own generator targets / voltage set-points (`set_injections_from_elements`, `set_gen_v`, every injection-sweep row): the row's targets per slot (the loops' value, else the row's, else the grid's, as `gen_in_force` / `gen_override`). Such rows still load through the host model. A distributed-slack Newton (`NROuterDist_*`, 2026-11-17): per-slot weights written as a control (`materialize` -> `ControlTargets::slack_w` -> the driver's `d_oc_slack_w_`), each converged solve's `slack_absorbed` (gathered at `stride-2`) folded into the units' targets (`ds_fold`), `mis_bus`'s `+ sa * w`, `check_ds_in_newton` (PARK / UNPARK, the unsolved target Q), parked units carried in strategy points; lockstep on lightsim2grid's dist grids, facades cold and resumed | |
| M5 | PhaseControl, TransformerVoltageControl, ShuntVoltageControl | ShuntVoltageControl done (2026-11-17): the controllers' shunts and section tables, the grid's groups (a row only hides some, per slot); `check_svc` / `svc_dispatch` / `sc_set_sections` on the device, the controls into the driver's `sc_controls_kernel` (`ControlTargets::sc_*`), the sections in results and strategy points, resumed sections; lockstep + facades + strategies. PhaseControl done (2026-11-17): BranchControl's transformers as device tables (each (ratio tap, phase tap)'s pi block, ratio, shift; the phase taps' shifts), their per-slot state retapped in `materialize`, `bc_current` / `check_pc` on the device, the controls into the driver's `bc_controls_kernel` (`ControlTargets::bc_*`), the final taps per row (results, strategy points, `row_trafo_blocks` -> flows); a row opening branches or masking buses loads through the host on such grids (its transformers / shifters / shunt groups). TransformerVoltageControl done (2026-11-17): its groups (a row only hides some), the row-fixed half of its freeze evaluated by the host load (candidates, target |V|, load; the voltage limit; the row's closed branches), the dynamic half on the device (`tvc_freeze` / `tvc_release`, suspended buses ReactiveLimits skips, the union-find of `tvc_fix_controls`, the shared ratio ranges, `check_tvc`'s three steps); lockstep + facades + strategies. Every loop lightsim2grid runs is on the device: the census marks every feature supported | |
| M6 | the host-only options (trace, adjoint, resume points, spare buses); `'device'` becomes the default | trace done (2026-11-17): every decision recorded per slot (`rec`, the host's 18 actions), flushed each round with the points and formatted as `record()` does; lockstep traces line by line == host, facades (values to rounding). Adjoint done (2026-11-17): the capability slopes recorded per slot at freeze time (`gen_limit_slope`, the curve's), the units parked as the last solve had them (`ds_mark_solved`), read back at a row's end with the points (`export_points` -> `Ls2gOuterModel::device_adjoint_info`, `adjoint_info` line by line); rows' adjoint views == host (`test_device_gradients.py`; its torch gradient comparisons need torch, absent here). Resume points done (2026-11-17): a contingency's point read off its slot at its end (`DeviceOuterHook::export_points` -> the host model's ResumePoint), its strategies loaded through the host model from it -- only the first strategy row of each point, the others device copies of it (per-point templates; check mode compares each to a host load); warm-started strategies `'device'` == `'host'` (`test_device_strategies.py`); spare buses (2026-11-17): bus splits (`set_spare_buses`: the host model takes the spares, the device tables are rebuilt from it) and generator moves (the device tables' ReactiveLimits plan grows by the buses the rows' moved generators bring, `device_prepare_rows`, generator membership per (bus, generator) position); `test_device_bus_splits.py` | |

Open decisions: §10. Last update: 2026-11-17.

## Why a new plan

Two things change from `outer_loops_host_work.md`:

- 4.6 ("decisions on the device") moves forward. It was last in that note because the host model
  looked cheap. On the 2024-05-20 07:00 French snapshot (PtFige) the host still holds the GPU idle
  for 1.6 s of a 6.1 s N-1 run.
- 4.2/4.3 (a lazy host state with a device filter) is dropped. GSA shows that moving the state itself
  is simpler. A filter would still need the host to rebuild a `SlotState` partway through a row, and
  the design below never does that.

## 1. What the trace shows

`traces/PtFige-20240520-0700_n1_perf_outer.nsys-rep` (2026-11-17, unloaded GPU, own LU, batch 2048,
~7.4k buses, J ~25k nonzeros). Exported with `nsys export --type sqlite` and read with a short
script over `CUPTI_ACTIVITY_KIND_*` within the `N-1 screening` NVTX range:

| | time | |
|---|---|---|
| N-1 screening | 6.07 s | |
| GPU busy | 4.14 s (68 %) | |
| `oc_unrealistic_kernel` | 0.92 s | 78 calls at ~12 ms, **1-7 blocks**: one thread per slot loops over every candidate bus |
| own LU (all `lu_*`) | ~1.5 s | |
| `fill_J_kernel` | 0.55 s | 8.3 ms per call over 2048 slots |
| `check_limit_violations_kernel` | 0.22 s | one block per row; one row alone is 1.3 ms (per-row latency) |
| update V, SpMV, F | ~0.6 s | |
| GPU idle | 1.93 s | |
| between a D->H and the next H->D | **1.61 s** | 72 gaps, ~22 ms each: gather download -> `on_solved` + `load` + packing -> controls upload |
| other gaps | ~0.3 s | launch latency (106k gaps < 1 ms), slot loads |
| actual copies | 0.09 s | D->H 71 ms, H->D 14 ms |

- **The copies are cheap.** What costs is the host compute between them.
- **`cudaEventSynchronize` (2.9 s API time) is a symptom.** The per-phase timers make the host wait
  for the GPU about ten times per round.
- **There are two separate problems.** The host model's turn (1.6 s) is one. A device-side outer
  kernel written one thread per slot (0.9 s) is the other.

## 2. How GSA runs its outer loops

Sources: `src/gpu_security_analysis/engine/outer/{driver,reactive_limits,slack,...}.py`,
`engine/kernels/{outer,distribute}.cu`, `docs/continuous_batching.md` §3, `docs/olf_parity_review.md`.

- **The controller state lives on the device**, per (scenario, controller): PV/PQ (`gvc`), limit type
  (`qlt`), frozen Q (`genq`), PQ->PV counter (`pvpq`), voltage targets, generator control type, P.
  It is uploaded once per bank at `reset()`. A strategy row inherits its contingency's final state,
  gathered on the device by parent index. A row's arrival costs no host work.
- **The model's tables are static, uploaded once per run** (`_static_tables`): controller -> generators
  (CSR), capability curves (padded point tables), forced target Q, HVDC parameters, standby thresholds.
- **Sequencing stays on the host but is vectorized**, never per row (`on_converged`). `cur`, `inside`,
  `last_unstable`, `nr_pass` and `total` are numpy arrays over the bank. The converged slots are
  grouped by the loop they are at, and each loop's check runs once, as kernels, for the whole group.
  It returns one bool per row (one D->H per loop and pass), and the host advances the state machine
  in numpy.
- **Each decision is a kernel**:
  - `qlim_decide`: one thread per (slot, controller). Capability curves, merged-controller key shares
    and robust mode are all inside.
  - `qlim_keep_one`: one thread per slot, the "keep the strongest PV" rule.
  - `qlim_apply`: writes the new state and the reactive targets.
  - `qlim_setv`: the robust mode's |V| reset.
  - `distribute_kernel`, `monitoring_kernel`, `hvdc_limits_kernel`.
  - `route_kernel`: rebuilds the voltage-control routing of J by value inside the union pattern.
- **Parity.** The old host code is kept as `testing/outer_reference.py`, and
  `tests/oracle/test_outer_gpu_vs_reference.py` compares the GPU path with it. The target is OLF 2.3.0.
- **Coverage.** Five loops: DistributedSlack, FreezingHvdc, AcHvdcAcEmulationLimits,
  VoltageMonitoring, ReactiveLimits. No transformer, phase or shunt control. Unsupported structures
  raise `NotImplementedError` for the whole batch.
- **Result.** "During the NR loops the GPU is practically never idle" (their §3.3). The GPU idles
  during chunk preparation, not in the loops.

What to take from it, and what not to:

| GSA | here |
|---|---|
| state on the device, static tables once per run | **take** |
| one kernel per loop over the slots that reached it | **take** |
| host oracle kept and differential tests | **take**: `Ls2gOuterModel` is our oracle, in a shadow mode |
| slowest kernels one thread per scenario: `route_kernel` 25 ms per call (6.4 s on France), `distribute_kernel` over 6732 generators (2.6 s) | **avoid**: our `oc_unrealistic_kernel` has the same flaw (M1). Every new kernel is one block per slot, or one thread per (slot, element) with block reductions |
| parity bugs from porting "outside-in" (robust mode is on by default but was absent from the parameter file, and nobody read it) | **avoid**: port `check_*` line by line, every branch, and test on rows that stress rare mechanisms (§7) |
| OLF as the reference | ours is lightsim2grid, through `Ls2gOuterModel` |
| `NotImplementedError` for the whole batch | **take**, but raised at construction from a scan of the grid's features (§4.2), not partway through a run |
| sequencing in numpy, one D->H per loop and pass | a per-slot state machine on the device, one D->H per round (§4.4) |

## 3. Where our host time goes, in code

Per round, `BatchPfDriver::_solve_continuous` with `outer_hook_` set:

1. `_residual_pass`, then a D->H of the residuals.
2. For the converged leaving slots, `_oc_gather`: `oc_gather_kernel` (V, S, Sbus at
   `Layout::watch_bus`, the controllers' Q), `oc_unrealistic_kernel`, the BranchControl/ShuntControl
   gathers, then a D->H and a sync.
3. `OuterHook::on_solved` for each such slot (`Ls2gOuterModel::on_solved_impl` -> `drive` ->
   `check_ds` / `check_hvdc` / `check_vm` / `check_rl` ... -> `materialize`, or `finish`).
4. `OuterHook::load` for each refilled slot (a `SlotState` built from `slot0_`, `init_loops`, the
   plans).
5. `_oc_write_controls`: the dense `SlotControls` arrays for those slots, the Sbus row re-read plus
   the sparse correction, an H->D and a sync.

Steps 3-5 are the 22 ms gap. The device design removes all three, and step 2 shrinks to the per-slot
outputs (status, counters).

## 4. Design

### 4.1 One backend per run

A new option, `outer_loops_backend = 'host' | 'device'` (the three batch facades, their engines and
sessions, `BatchPowerFlow`; construction time), picks the path for the whole run:

- **`'host'`** (the default until M6): today's behaviour, exactly. `Ls2gOuterModel` runs the loops on
  the host, unchanged and bit-identical. It is also the oracle of every device test (§7).
- **`'device'`**: every row runs on the device, from load to finish. No row goes through
  `OuterHook::load` / `on_solved`. The two paths never mix within one run.

When the device path covers everything (M6), `'device'` becomes the default, and `'host'` stays,
unchanged, for the tests and as a reference.

### 4.2 Unsupported: `NotImplementedError` for the whole batch

The goal is full coverage, so there is no fallback path: anything the device path does not
implement yet stops the run.

- **At construction**, a feature scan (`outer::device_support(grid, ledger, options)`, host, reading
  what `Ls2gOuterModel`'s constructor already reads) lists every feature the run may need: the loops,
  voltage-control groups and remote regulators, held controllers, SVC entries in the ReactiveLimits
  plan, standby monitors, capability curves, `NROuterDist_*`, resumed frozen buses, masked rows,
  units disconnected or `gen_v` set by rows, the host-only options (`keep_outer_trace`,
  `keep_final_jacobian`, `keep_row_points` / `set_row_resume`, spare buses). With `'device'`, the
  session raises `NotImplementedError` naming every missing feature, before any GPU work.
- **Row inputs set after construction** (`set_row_overrides`, the rows' targets) are scanned again at
  `run()`, before the first round.
- **On the device**, each kernel keeps an error flag for any case the scan should have refused. It is
  read once per round with the outcome download, and a set flag throws (a bug in the scan, reported
  with the row and the case).

What a feature needs, and the order it is ported in, comes from M0's census (§5). Until a feature is
ported, a grid that has it runs only with `'host'`.

### 4.3 Device data

**Static, per run** (built on the host from `Ls2gOuterModel`'s own plans so the content is identical,
uploaded once):

| table | content | source |
|---|---|---|
| RL buses | `bus_solver`, `reg_bus_solver`, `local`, `pin`, `target_vm`, `nominal_v`, `target_q`, `base_min_q`, `base_max_q`, `group`, `has_svc`, `monitor_svc` | `RlPlan::buses` of the grid plan (`rl0_`) |
| RL generators | CSR bus -> generators, curve point tables (padded, like GSA's `pts_*`), `min_q`/`max_q` | `RlPlan::plan->buses[].gen_ids`, the grid's curves |
| DS units | `injection_mw`, `min_mw`, `max_mw`, raw weight, kind, bus, in upstream's order | `units0_` |
| HVDC plan | per line: buses, `p0`, `k`, loss factors, `r`, `pmax12`, `pmax21`, ledger index | the grid's `HvdcPPlan` |
| standby SVCs | buses, thresholds, targets, controller index | `monitors_` |
| loop list | the active loops in order, their parameters (`LoopDecl`) | `loops_` |

**Per slot** (SoA, `[S × ...]`, moved by `_compact_tail` like every per-slot buffer, see CLAUDE.md):

| state | host field (`SlotState`) |
|---|---|
| RL per bus: `state` (-1, 0, +1), `frozen_q`, `nb_pv_pq`, `realistic`, `target_p` | `rl_state[]` |
| DS per unit: `current_mw`, `parked` (M4) | `ds_current_mw`, `ds_parked` |
| hvdc regime per line | `hvdc_regime`, `hvdc_plan.lines[].frozen_dir` |
| standby target per SVC (NaN idle) | `svc_on` |
| sequencing: `i`, `last_unstable`, `last_status`, `first`, `changed`, `nb_outer`, `nb_passes`, `total_nr`, `old_total`, `nr_steps`, `v_version`, `done_at[loop]`, `loop_iter[loop]`, `pass_check_unrealistic`, `check_unrealistic`, `solver_ok`, `unrealistic` | the "driver state" block |
| the controls | already on the device: `d_oc_pin_`, held Q, `v_set`, hvdc status, the Sbus correction, |V| resets |

**Per row results** (written at finish): status, `RowStatus`, `nb_outer`, `nb_nr`, unrealistic flag,
failed loop index, DS units' final MW `[n_rows × n_units]`, hvdc regimes. A dense
`[n_rows × n_gen]` target array would be ~0.5 GB at 10k rows × 6.7k generators, so the units' own
order is used. `get_outer_results()` builds `RowResult` from these on read, as `row_result()` does
today.

The controls the device writes go straight into the driver's `d_oc_*` arrays. Every device write
must clear `oc_mirrors_on_device_` for those slots, or update the host mirrors (the CLAUDE.md
invariant on `_oc_write_controls`).

### 4.4 Kernels and the round

Per round, after `_residual_pass`, on the device slots that just converged:

1. `oc_dev_seq_kernel` (one thread per slot) ports `on_solved_impl` / `drive` / `finish`: it advances
   to the loop to check or ends the row. It writes a per-slot "loop to check" code (or done).
2. One kernel per loop on the slots at that loop (compacted lists built on the device by a scan):
   - **ReactiveLimits.**
     - `rl_eval_kernel`, one thread per (slot, RL bus): `_evaluate`. That is `mis_bus` on the
       device from V, S and Sbus (the gather's quantities, read in place, never copied), the limits
       (`base_*` at first), the to_pq / to_pv / kept / moved classification and the robust-mode test.
       `remaining_pv` comes from a block reduction.
     - `rl_apply_kernel`, one block per slot: "keep the strongest PV" (an argmin over to_pq by
       (nominal_v, target_p, bus), as `check_rl`), the `max_switch` gate, then freeze/release. It
       writes `pin`, the Sbus Q correction, |V| resets and the RL state, plus a per-slot `changed`.
   - **DistributedSlack.** `ds_check_kernel`, one block per slot: the slack-bus mismatch, then a port
     of `slack_redistribution::distribute` (proportional to the raw weights, clamped, a clamped unit
     leaves the pool, iterated) with block reductions. The order of units and sums stays upstream's.
     Then the moved test and the P correction.
   - **HvdcLimits** (one thread per (slot, line)) and **VoltageMonitoring** (one thread per
     (slot, SVC)): `check_hvdc_p_violations` / `check_svc_standby_violations` re-expressed.
     `hvdc_flows_pu` already exists on the device.
   - Each writes UNSTABLE / STABLE / FAILED per slot and its controls.
3. Steps 1-2 repeat on the device until every converged slot has either a re-solve or an end. A
   pass that only finds STABLE loops continues to the next loop without leaving the device. This
   is a bounded loop of launches, at most n_loops × max_passes, with an early exit when nothing is
   pending. A device-side flag avoids a host round-trip for each iteration.
4. One D->H per round: the per-slot outcome (`Resolve` / `Done`), the error flags and the row
   status of finished slots. This replaces the gather download. The host only frees slots and
   refills them.
5. Refilled device slots are loaded by `oc_dev_load_kernel` (one block per slot): the slot's state
   from the static `state0` and the row's per-row inputs on the device, and the grid's default
   controls. No `OuterHook::load`, no host packing.

**Bit parity.** The host model's decisions compare sums against thresholds (`q < q_min - eps`,
`moved > 0.9·eps`, `|mismatch| > 1 MW`). A device sum in a different order, or contracted into FMAs
(nvcc's default), can flip a borderline decision. So:

- The decision kernels sum in the host's order (one thread per sum where it matters, or a fixed-order
  reduction whose host counterpart is changed to match).
- They are compiled with `-fmad=false`, or written with `__dadd_rn` / `__dmul_rn`.
- The shadow mode (§7) reports any decision that differs, with its margin.

### 4.6 The device model behind `outer::OuterHook` (how M2 is built)

The device model (`outer/device_outer_model.{hpp,cu}`, factory `make_device_outer_model`) implements
the same `outer::OuterHook` interface as `Ls2gOuterModel`, with its state in device arrays and its
decisions in kernels:

- **The driver runs it unchanged** at first, through the per-slot calls (`load`, `controls`,
  `on_solved`): exact from day one, not yet fast (each call crosses the bus). M3 then adds batched
  entry points (load / decide on a slot list, controls written straight into the driver's `d_oc_*`
  arrays) and the driver calls those when its hook is a device model.
- **Tests in lockstep** (`tests/python/outer_device/test_device_model.py`): the test-only
  `_OuterModelHarness` takes `backend='host' | 'device'`; both models get the same converged states
  (test_outer_model_cpu.py's numpy full-system Newton, on the host model's controls) and must give the
  same next step, row status and controls after every call, and the same row result. That is "the
  same decisions as the host model", without a device decision trace. Every scenario of
  test_outer_model_cpu.py is a candidate: the device model is done when they all pass in lockstep.
- **One thread per slot first.** The per-slot logic (`drive` / `on_solved_impl` / `finish`, the
  checks, `materialize`) is ported as `__device__` functions that follow the host functions line by
  line, the per-slot `std::vector`s and flat maps becoming fixed-capacity per-slot arrays. Sums run in
  the host's order and the file is compiled without FMA contraction, so a borderline threshold
  decides the same way. The hot loops (the ReactiveLimits evaluation over ~600 buses and their
  generators' curves) are parallelised in M3, once parity is pinned.
- **What it needs from lightsim2grid**, ported as device functions: the capability curve
  (`LSGrid::gen_limits_at` / `curve_at`: segment search, linear interpolation, extrapolation from
  the end segments, crossed limits averaged; fixed limits when no outer target),
  `gen_target_q_at_outer_target` (the raw target Q clamped into the limits at the moved target) and
  `slack_redistribution::distribute` (inline in `SlackRedistribution.hpp`). Reference sources:
  lightsim2grid v1.3.1 `src/core/LSGrid.cpp` (the version the bridge builds against).
- **Static tables** are filled on the host from the host model's own plans (the RL plan, the
  DistributedSlack units, the hvdc plan, the monitors, the curves), so both models start from the same
  data; a CUDA-free struct (`outer/device_tables.hpp`) carries them to the `.cu`.
- **Loading a row, M2:** the device model keeps a host `Ls2gOuterModel`; `load` runs the host
  model's own `load` and uploads the slot state it built (one way, once per row): masks, resume
  points and every other row input are then exact by construction, and only the decisions
  (`on_solved`: sequencing, checks, `materialize`, `finish`) are kernels. **M3:** a row that
  changes nothing loads on the device from one template per run (the grid's defaults, or its
  resume point with `init_from_n_powerflow`), so the host `load` leaves the round.

### 4.5 What does not change

- The inner Newton, the strategies, BranchControl/ShuntControl kernels, the adjoint path and the
  host model itself. The host model stays the oracle and the `'host'` backend.
- Every run with `'host'` is bit-identical to today.

## 5. Milestones

Each milestone has an exit criterion. Phases M2-M3 are gated on M0's numbers.

### M0: measure (needs a GPU, ~0.5 day)

- NVTX ranges around `oc_gather`, `oc_decide` (the `on_solved` loop), `oc_load`, `oc_controls`, and
  one per round. A timing mode without per-phase `cudaEventSynchronize` (events read at round end).
- PtFige with `keep_outer_trace=True` and `GPUSIM2GRID_OC_PROFILE=1`: per loop, the actions and first
  actions; rows that never re-solve; outer iterations per acting row.
- **The feature census**, on PtFige and the test grids: which of the features the scan lists (§4.2)
  each grid has, and how many rows exercise each one (non-local RL buses acting, SVC entries,
  limits moved by a target change, masked rows, resumed frozen non-local buses). It orders M4.
- Is PtFige solved with `NROuter_*` or `NROuterDist_*`? That decides whether `check_ds_in_newton` is
  in M2 or M4.
- **Exit:** a table of these numbers, added to this note. It fixes the order of M2 and M4 and the expected gain.

*Census, 2026-11-17* (`outer_device_census`, PtFige-20240520-0700, 7365 solver buses, `NROuter_*`
single-slack Newton; `claude_data/outer_device/scripts/census.py`):

| feature | count |
|---|---|
| loops | DistributedSlack, AcHvdcAcEmulationLimits, VoltageMonitoring, ReactiveLimits |
| ReactiveLimits local controller buses | 544 |
| ReactiveLimits voltage-control group buses (remote regulators) | 79 |
| ReactiveLimits robust mode on group buses | on |
| ReactiveLimits buses with SVCs / storage units / HVDC stations | 4 / 2 / 4 |
| ReactiveLimits voltage-monitor buses | 1 |
| generators with a reactive capability curve | 6301 (all) |
| generators with forceTargetQInReactiveLimits | 6301 (all) |
| DistributedSlack units | 249 |
| HVDC lines in AC emulation | 0 (the loop is there, nothing to act on) |
| standby SVCs | 1 |
| with `init_from_n_powerflow=True`: frozen local / group buses | 5 / 3 |

**Consequence:** with the whole-batch refusal, a French grid runs on the device only once all of
ReactiveLimits is ported: groups (held controllers by value), capability curves at moved targets,
the SVC / storage / station terms, the monitors and the robust mode. A "local buses only" M2 would
never run it. M2 therefore takes all of ReactiveLimits, single-slack DistributedSlack (with the
forced target Q of moved generators), VoltageMonitoring and AcHvdcAcEmulationLimits, plus resumed
rows; M4 keeps masked rows, rows that disconnect units or set `gen_v`, and `NROuterDist_*`.

### M1: fix the existing device kernels (~1 day)

- **M1a** `oc_unrealistic_kernel`: one block per slot, threads strided over the candidates,
  `__syncthreads_or`. The flag is an OR, so the output is identical. Expected −0.9 s on PtFige.
  *Done* (`outer_kernels.cu`, `.cuh`, the launch in `batch_pf_driver.cu`).
- **M1b** (look only) `check_limit_violations_kernel`: 1.3 ms for one row alone (`grid 1` in the
  trace). Find where a block spends that (branch scan, radix select).
- **Exit:** outer-loop suites green; PtFige statuses, iteration counts and V identical; nsys shows
  the kernel under 0.5 ms per call.

### M2: device rows for the default loops (~2-3 weeks)

Built in four steps, each one testable without changing results:

- **M2a, backend, scan, tables and state.** `outer_loops_backend`, the feature scan raising
  `NotImplementedError` (§4.2), static tables and per-slot SoA, `oc_dev_load_kernel`.
  - Shadow test: after a load, the device state equals the host `SlotState` (RL state, DS units,
    controls) for every row of the test grids.
- **M2b, checks in evaluate-only mode.** The four check kernels run beside the host model on the
  same converged state and write their decisions to a buffer. The host model still decides.
  - Shadow test: identical decisions (to_pq / to_pv / moved per bus, DS moved MW, hvdc and SVC
    switches) on every `on_solved` of the suites and of PtFige.
  - This is where parity is earned, with no risk to results.
- **M2c, device decides.** Sequencing kernel, apply kernels, controls written on the device, the
  device error flag.
  - Test: statuses, `nb_outer`, `nb_nr`, NaN pattern and V identical to the host path on the suites
    and PtFige (V to rounding where M2b proved identical decisions).
- **M2d, outputs.** Results by row, `get_outer_results()` from them, timings (`t_oc_dev_*`).
- **M2's scope** (from the census, M0): all of ReactiveLimits (local and group buses, curves, SVC /
  storage / station terms, monitors, robust mode), single-slack DistributedSlack, VoltageMonitoring,
  AcHvdcAcEmulationLimits, resumed rows. A grid needing more raises until M4.
- **Exit:** the test grids that need no more than M2's scope run with `'device'`, with outcomes
  identical to `'host'`; nsys shows no ~22 ms host gap in their rounds.

### M3: the round without the host (~1 week)

*Step 1 (2026-11-17):* the device model also implements `outer::DeviceOuterHook`
(`device_outer_model.hpp`), which `_solve_continuous` uses when its hook is one:

- `load_rows(slots, rows)`: the host model loads each row, the slot states go up as one staging
  buffer per pool and a scatter kernel (the state pools are slot-major: a slot is one row of each
  of the int / byte / double pools, so a load or a tail move is three copies);
- `decide(...)`: every slot leaving the round in one launch, on the gathered rows in
  `d_oc_gather_` (no gather download); only the next step and status of each slot come back;
- `write_controls(...)`: the dense controls written in place into the driver's `d_oc_*`
  arrays; the Sbus correction and the |V| resets laid out in fixed blocks per slot (bus -1 =
  unused) that the driver applies with its own `oc_sbus_delta_kernel` / `apply_vm_reseed_kernel`,
  so their arithmetic is the host path's. `oc_mirrors_on_device_` is cleared.

What is left on the host per round: the eviction loop and the host model's `load` of the
refilled rows (step 2: a template load on the device for rows that change nothing).

- Remove the per-phase syncs from the round in the timing-free mode, one D->H per round (§4.4 step 4).
- Device slot refills from a device queue pointer (`outer_loops_host_work.md` 4.7, the easy half).
- **Exit:** GPU idle within the N-1 range under ~0.3 s on PtFige.

### M4: the rest of the four default loops

In the census's order, probably:

- non-local RL buses (held controllers by value: `held_q`, the `vc_hold_*` kernels already exist);
- limits at moved targets (capability curves on the device, GSA's `gen_q_limits` as a template);
- SVC entries;
- masked rows (their plan = the grid plan minus masked buses: a per-slot skip mask);
- rows that disconnect units or set `gen_v` (their own plan and DistributedSlack units);
- `NROuterDist_*` (`check_ds_in_newton`, parking, `ds_fold`, `calibrate_slack_absorbed`);
- resumed rows' frozen non-local buses, regimes and switched-on SVCs (`init_from_n_powerflow=True`).

Each item gets its own shadow test and lightsim2grid test, then the scan stops refusing it.
**Exit:** PtFige and the French snapshots run with `'device'`, outcomes identical to `'host'`.

### M5: the other three loops

PhaseControl, TransformerVoltageControl (its INITIAL / CONTROL / COMPLETE stages, its generator
freeze through ReactiveLimits' freeze), ShuntVoltageControl. The device side of their controls
already exists (BranchControl / ShuntControl kernels patch Ybus by value); what moves is their
checks and dispatch (tap and section choice), `materialize`'s share of them, and `row_trafo_blocks`.

### M6: the host-only options, then `'device'` by default

- `keep_outer_trace`: a compact device decision log (loop, action, element, value, limit), turned into
  the strings at read time.
- `keep_final_jacobian`: the outer adjoint (`adjoint_info`'s terms from the device state, `_oa_save`
  unchanged).
- `keep_row_points` / `set_row_resume`: resume points kept on the device.
- spare buses (topology actions).

**Exit:** the whole outer-loop test suite runs with `'device'`; `'device'` becomes the default.

## Goals stated by the user (2026-11-17)

- **The GPU busy close to 100 % of a run** (ideally): the measure is the GPU-busy share of the
  run and the GPU-idle gaps our host turn leaves (count, total), read off our own nsys traces.
  That needs a device-resident round (M3's last item): no host wait between rounds.
- **Speedups claimed with load-independent metrics** (H2D / D2H copies and bytes, syncs,
  allocations, launches, rows the host touches), the GPUs being shared: wall time only as support.
- **Remedial actions** (strategies resuming from their contingency's final state, spare buses)
  are coming use cases: the device state must support a row starting from another row's final
  slot (a device-to-device copy, as the template), not only from the grid's.

## 6. Expected gain on PtFige (estimate, to be replaced by M0/M2 measurements)

| | N-1 screening |
|---|---|
| today | 6.07 s (measured) |
| + M1a | ~5.15 s |
| + M2-M4 (no host turn left) | ~3.6-3.9 s: GPU 3.2 s + launch gaps ~0.3 s + the device checks |
| + M3 | the launch gaps and timer syncs, a few hundred ms more |

After that the inner Newton is the floor (own LU 1.5 s, `fill_J` 0.55 s, V/F/SpMV 0.6 s).
`newton_kernels_performance.md` is the lever there, not the outer loops.

## 7. Tests and parity

- **Oracle:** `Ls2gOuterModel`, unchanged, already pinned against lightsim2grid's `NROuter_KLU`
  (`test_outer_model_cpu.py`) and batches (`test_outer_loops_batch.py`).
- **Shadow mode** `GPUSIM2GRID_OUTER_DEVICE_SHADOW=1`: the host model runs on every slot too,
  fed the same gathered state. Any differing decision is reported with the row, loop, bus, both
  values and the threshold margin. M2b runs in this mode only. Afterwards it is a debug tool and a CI
  test on the small grids.
- **New tests** `tests/python/test_outer_device.py`: device path vs host path (statuses, counters, V,
  `get_outer_results()`) on every grid of `test_outer_loops_batch.py`, plus one test per refused feature
  (each raises `NotImplementedError` naming it with `'device'`, and runs with `'host'`). Tolerances come from the `solver_atol` /
  `residual_atol` fixtures.
- **Rows that stress the rare branches** (GSA's lesson): robust mode on a remote controller pushed
  out of [0.8, 1.2], `max_pq_pv_switch` reached, "keep the strongest PV" with ties, a DS FAILED
  residue, an hvdc release, a standby SVC switched on, a limit moved by a DS share, the 30-outer-
  iteration budget.
- **Rule:** compare with the host path and lightsim2grid, never with the previous version of this
  branch (GSA's "compensating errors").

## 8. Risks

- **Bit parity of borderline decisions** (§4.4). Mitigated by the summation order, no FMA, and
  shadow mode. If some sums cannot be made identical, document the margin and accept rounding-level
  differences only where lightsim2grid itself is that sensitive.
- **Two implementations of four loops to keep in sync** with lightsim2grid's evolution. The shadow
  test in CI is what keeps them honest. Each upstream loop change needs a device change or a new
  unsupported reason.
- **Time to coverage.** Until M4, a grid with an unported feature cannot use `'device'` at all (no
  partial gain). French grids likely need non-local RL buses (72 of 585 groups have remote
  regulators on one RTE snapshot), so they come right after M2's first scope.
- **A scan that misses a feature** gives wrong results silently, not an error. The device error flag
  (§4.2) and the shadow test on every grid are the guard.
- **Memory** of per-slot SoA: RL buses (~600) × S, DS units (~270) × S. That is small next to J and
  the LU at S = 2048.

## 9. Out of scope

- More host threads (the constraint of `outer_loops_host_work.md`).
- Porting the inner Newton changes of `newton_kernels_performance.md`. They are separate.

## 10. Decisions

Decided (2026-11-17):

- The branch ports every outer loop to CUDA kernels: the four default loops (M2, M4), the other three
  (M5) and the host-only options (M6).
- Everything, M1 included, is done on this branch and goes in its one PR.
- `outer_loops_backend = 'host' | 'device'`; `'host'` keeps today's behaviour exactly.
- Anything `'device'` does not implement yet raises `NotImplementedError` for the whole batch (no
  host fallback within a run).

Open:

1. Parity: the device loops are a second implementation, checked against `Ls2gOuterModel` (shadow
   mode) and lightsim2grid. Is "identical decisions, V to rounding" the acceptance criterion?
2. A GPU for M0 and M1.
