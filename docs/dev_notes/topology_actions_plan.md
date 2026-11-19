# Topology actions: contingency × remedial action screening on the GPU

**Design note, not documentation.** Plan written on 2026-11-16, before any code. It is not part
of the built documentation (`docs/*.rst`). Status of each step: § 9.

## 1. Why

The roadmap's **best action selector** ("given one or several grid snapshots and a list of
candidate actions, find the best action(s) to apply, by evaluating the candidates in batch on the
GPU") is, in operation, a *contingency × remedial action* study: for every contingency `c` of a
list and every candidate action `a` (open/close a switch, change a phase shifter tap, ...), solve
the grid with `c` and `a` applied and rank the actions by the loading they leave.

A private RTE prototype of that study (`gpu-security-analysis`, whose batched LU became
`linear_solver='own_lu'` here) runs it on a full French transmission snapshot (~7 400 buses):
200 contingencies × 800 candidate actions = 160 000 load flows per snapshot, about 30 s per
snapshot on one GPU, results checked against OpenLoadFlow. This note plans how to bring the
same use case to gpusim2grid, reusing what it already has (row patches, masks, structural
supersets, continuous batching, own LU) rather than porting that code.

The candidate actions of that case, on one snapshot (2024-05-20 07:00): each switch action
applied alone to the node-breaker network, the bus partitions of pypowsybl's bus view compared
before / after (bus ids are not comparable: pypowsybl renames the buses of a voltage level whose
topology changes). 1 106 actions:

| action kind | actions | what changes in a bus-branch model |
|---|---|---|
| no change on this snapshot (e.g. closing a closed switch) | 466 | nothing |
| open the breaker of one feeder (line 376, transformer 92, load 48, generator + load 21, generator 4, shunt 3) | 544 | one element (or a bay) disconnected |
| open a coupler: one bus becomes two | 32 | elements move to a **new** bus of the substation |
| close a coupler: two buses become one | 21 | two buses **merge** |
| phase shifter tap | 4 | a branch admittance changes (value only) |
| element not in this snapshot | 39 | — |

Each candidate ("strategy") is 1 or 2 actions, 800 per contingency, 160 000 in all. By what they
do (strategies whose contingency exists on the snapshot or not):

| strategies that... | count | share | step |
|---|---|---|---|
| only disconnect elements (possibly with a no-op action) | 105 600 | 66 % | A |
| change nothing (the row is its contingency) | 37 200 | 23 % | A |
| split a bus | 6 400 | 4 % | B.2 |
| merge two buses (with or without a disconnection) | 4 200 | 2.6 % | B.2 |
| change a phase shifter tap | 800 | 0.5 % | B.1 |
| name a missing element | 5 800 | 3.6 % | — |

Step A alone (§ 4) therefore covers ~89 % of the rows; the union pattern (§ 5) the remaining
~7 %, which is where the topology screening's value is.

## 2. What gpusim2grid has today

- **Per-row topology = branch / element trips.** `ContingencyAnalysisGPU.add_contingency(
  branch_ids, shunt_ids, load_ids, gen_ids, sgen_ids, storage_ids, svc_ids, hvdc_ids)` and
  `ScenarioSweepGPU.set_topology` / `set_contingency_*`. A row's Ybus is the base Ybus plus
  a list of value **triplets** `(row, col, delta)` resolved to CSR positions of the *base*
  pattern (`Triplet`, `resolve_indices`, `contingency_analysis_helper.hpp`); a tripped
  injection is an Sbus delta (`element_trips::sbus_delta`). Removal only: a triplet whose
  `(row, col)` is not in the base pattern cannot be resolved.
- **One sparsity pattern per batch.** One cuDSS ANALYSIS (or one own-LU plan) per batch; every
  slot has the same Ybus and J pattern.
- **Structural supersets** (`ledger_extend.{hpp,cpp}`): the CA / SS sessions may already solve a
  J whose pattern is a strict superset of lightsim2grid's (stranded-controller slots, switchable
  Vm buses appended at the end of the ledger), filled per slot by value.
- **Masks** (`handle_disconnected_grid`): a bus outside the main component is identity-pinned
  (J row = e_i, F = 0) and reported NaN, by value, per slot.
- **Batching**: chunked or continuous (`scheduling='continuous'`, the tail), own LU.
- **Results**: `compute_flows`, fused limit violations (most severe per row and type).

What is **missing** for the use case:

| # | gap | consequence today |
|---|---|---|
| G1 | a contingency × action **product** | the caller materializes 160 000 rows; nothing is shared between rows of one contingency or one action |
| G2 | **value changes** of a branch (phase shifter tap, ratio, impedance) per row | not expressible (only "trip") |
| G3 | **bus splits / merges** per row, i.e. a row whose Ybus/J pattern differs from the base one | not expressible: needs a union pattern (§ 5) |
| G4 | **switches** (node-breaker actions) | no representation; lightsim2grid's detailed topology (§ 6) is not released |
| G5 | **warm start** of `(c, a)` from the converged state of `c` | every row starts from the base case |
| G6 | **memory** of a dense per-row Sbus (ScenarioSweep): 160 000 × 7 400 × 16 B ≈ 19 GB | ScenarioSweep cannot hold the case; ContingencyAnalysis can (sparse Sbus deltas) |
| G7 | a **per-row reduction** (max loading, most loaded branch) without the full flows on the host | `compute_flows` is n_rows × n_branch on the host (160 000 × 11 000 × 8 B = 14 GB) |

## 2b. The two codebases side by side

`gpu-security-analysis` (GSA below, the private prototype) and gpusim2grid (G2G) solve the same
batched Newton problem with different scopes. What differs, and what is worth bringing over:

| topic | GSA | G2G today | bring over? |
|---|---|---|---|
| model source | OpenLoadFlow's `LfNetwork`, exported by a Java bridge: OLF's equations, statuses and violation filter exactly | lightsim2grid grid (pypowsybl / pandapower / ...), lightsim2grid's equations, `olf_rules` approximations | **no**: G2G stays lightsim2grid-based (its oracle) |
| topology per row | any bus partition: switches merged per situation (zero-impedance merges), nodal actions (target partition per voltage level) | element trips only (value triplets on the base pattern) | **yes**: § 5 (spare buses, moves) |
| sparsity | one pattern per block = **union** over its scenarios (`union_pattern`); scenarios differ by routing arrays (which bus row/column an element writes to, identity rows) | one base pattern per batch; supersets only for controller slots / switchable Vm | **yes**: § 5.2, union pattern with a bit-identical no-op; banks later |
| actions | `SWITCH` (open/close), `PHASE_TAP_CHANGER_POSITION`, nodal actions | none (contingencies, injections, generator v-set) | **yes**: § 4 – § 7 |
| switch model | pypowsybl node-breaker view through OLF; contingency propagation (a contingency opens the switches up to the nearest breakers) | none | **outside the solver and later** (§ 6): lightsim2grid's detailed topology; propagation belongs there too |
| contingency × strategy | two phases as OLF's `AcSecurityAnalysis`: contingencies first, then each strategy starts from its contingency's final state | one phase, every row from the base case | **yes**: § 4 product + warm start, streamed (a contingency releases its strategies as it finishes) |
| strategy conditions | `TRUE_CONDITION`, `AT_LEAST_ONE_VIOLATION_CONDITION` (+ subject filter): a strategy runs only if its contingency violates a limit | n/a | **yes**: `strategy_condition=StrategyCondition.ANY_VIOLATION` (constructor), checked on the device from the contingency's violation counts as it finishes |
| results | result processors reduced on the GPU while the analysis runs (`MaxRho`: max loading + branch per situation), flows optional | V, flows on the host (n_rows × n_branch), most severe violation records | **yes**: max-loading reduction (§ 4, G7) |
| batching | continuous, tail by smaller banks (2048 → 512 → ... → 64) | continuous, tail by compaction (own LU prefix) | no: G2G's tail is the better one |
| linear solver | cuDSS uniform batch or own batched LU | cuDSS or `own_lu` (ported from GSA) | already there |
| outer loops | OLF's (distributed slack, reactive limits, voltage monitoring, HVDC AC emulation limits), checked on the GPU | lightsim2grid's seven loops per row | no; moves + outer loops is a later step (§ 5.3) |
| host preparation | numba, a producer thread preparing the next block while the GPU solves | C++, GIL released in the long calls | to check on the 160 000-row case (§ 8.2): if host prep shows, overlap it |
| several snapshots | streaming API: snapshots in a queue, the GPU busy across them | sessions reuse their setup across `run()` | partly there; with the union pattern, a series whose union does not change keeps its analysis (the no-op check gives it) |
| API | drop-in for `pypowsybl.security.SecurityAnalysis` (statuses, branch results, violations in pypowsybl's layout) | lightsim2grid-style facades | no (a thin facade could live in a separate package) |

In short, the four things to introduce, in order: the **product** of contingencies and actions
with a warm start and an optional violation condition (§ 4); **per-row branch values** (taps,
§ 5.1); the **union pattern** with spare buses for splits and merges, free when unused (§ 5.2);
and later **switch actions**, translated to bus moves by lightsim2grid (§ 6).

## 2c. Where the time will go (estimated before measuring)

End-to-end timings are not taken yet (the GPUs of the development machine are shared with
another job). The design below is driven by where the prototype lost its time on this very case,
and by gpusim2grid's own measurements on French N-1 (`outer_loops_host_work.md` § 1: 9 k rows).
Target: the prototype's 13.4 s of GPU time / 19.3 s end to end per snapshot (185 contingencies +
142 635 strategies, own LU, batch 2048, one GPU alone).

| # | expected bottleneck at 160 000 rows | estimate | evidence | design answer |
|---|---|---|---|---|
| 1 | **host outer-loop model per row** (`Ls2gOuterModel::load` ~50 µs, `on_solved` ~37 µs per call) | ≥ 160 k × 87 µs ≈ **14 s**, GPU idle meanwhile — alone above the target | `outer_loops_host_work.md` § 1 (GPU idle 60 % of an N-1 run) | a strategy row **inherits its contingency's outer state** (decisions, frozen buses) with its V (§ 4 warm start): the prototype's restarted strategies converge in 1–2 Newton iterations (p99 2) and rarely re-enter a loop, so fewer `on_solved` per row; then the device-side decisions that note already plans (its 4.2–4.6) |
| 2 | host construction of 160 k rows (per-row triplets, CSR lookups, masks, connectivity) | O(rows × trips) today | prototype: building each strategy from scratch cost 4.6 s per snapshot, 0.11 s once done incrementally from its contingency (DFS forest + bridges once per contingency) | product rows (§ 4): a contingency's patch / mask / connectivity is built **once**, a strategy only adds its action's (pre-resolved) triplets; connectivity of `(c, a)` from `c`'s components |
| 3 | Python per-row calls (`add_contingency` × 160 k) | ~3 s at ~20 µs per call | — | `set_actions` once (A objects) + `add_strategies` with arrays |
| 4 | results to the host: `compute_flows` = 160 k × 11 k branches × 2 sides × 8 B | **28 GB** | — | per-row **max loading** reduced on the device at eviction (the prototype's `MaxRho` processor); flows only for selected rows |
| 5 | dense per-row Sbus (ScenarioSweep path) | 19 GB | — | the ContingencyAnalysis path (sparse Sbus deltas per row) |
| 6 | strategies that need not run | 100 % run today | prototype: `AT_LEAST_ONE_VIOLATION_CONDITION` skips every strategy of a contingency without violation (all of them on this snapshot: max rho 0.978) | `StrategyCondition.ANY_VIOLATION` (§ 4) |
| 7 | union pattern fill (§ 5.2) | unknown: 53 substations touched by splits / merges on this snapshot, a few entries each | prototype: one union pattern per bank; `pattern_growth_limit` fallback when a block's union grows | measure nnz(J) and the own-LU plan's nnz(L+U) of the union **on the host** (no GPU needed); banks only if it grows |
| 8 | tail: the few rows with many outer restarts | prototype: 1–3 % of strategies with 4–6 restarts kept small banks busy (22 s before its fix) | `continuous_batching.md` (both repositories) | gpusim2grid's tail compaction (own LU prefix), from the first eviction: `tail_max_groups=batch_size, tail_group_size=1` |

**What to reuse from the prototype** (ideas and semantics, not its code — it is CuPy / numba around
OpenLoadFlow's model):

- its **nodal action** semantics (a target partition of one voltage level: groups sharing a bus,
  disconnected elements, every element listed) as a constructor of the action format here:
  `Action.from_partition(grid, sub, groups, disconnected)` (§ 7);
- its **partition-based** reading of switch actions (bus partitions compared, not bus ids) for
  the end-to-end benchmark's translation of the French case (§ 6);
- incremental construction of a strategy from its contingency (#2), warm start from the
  contingency state (#1), conditions (#6), the on-device max-loading reduction (#4);
- its France 200 × 800 inputs and per-situation results (statuses, max rho, worst branch) as the
  end-to-end reference (§ 8).

### Side question: result buffers, DLPack and CuPy

Results reach Python two ways today: `DeviceBuffer` (a lazy handle whose `to_numpy()` copies
device → host through the session) and DLPack capsules (`compute()`'s voltages, the
differentiable layer). DLPack is not a library but an exchange format (a C struct: pointer,
device, dtype, shape, strides; gpusim2grid vendors its one header): `cupy.from_dlpack`,
`torch.from_dlpack`, `jax.dlpack.from_dlpack` take a capsule **without copying**, so the caller
keeps the data on the GPU with the array library of its choice. The capsule aliases live
session memory: the next `compute()` overwrites it.

CuPy is **not** a dependency of gpusim2grid (numpy, scipy, lightsim2grid; PyTorch optional) and
should not become one: its wheels are per CUDA major, and DLPack already gives any caller a
CuPy array. What this work changes:

| result | size per row | today | with topology actions |
|---|---|---|---|
| residuals, reasons, max loading, iterations | O(1) | `DeviceBuffer` / numpy | same; the row map expands them on the host (cheap) |
| V | n_bus (7 400 on France) | stored for every row on the device (`d_V_results`, 19 GB at 160 k rows) | one start per contingency with strategies (22 MB, the warm start), the rows' own lazily (`V_results`) |
| branch currents | 2 × n_branch (22 000) | `compute_flows()`: every row on the device (28.7 GB), copied whole | reduced at eviction by a processor (max loading, ~60-100 ms for 160 k rows, estimated); per-row flows only on request |

The first version of `get_max_loading()` reduces on the host from `compute_flows()`. The
device-side reduction ("result processor", as the prototype's `MaxRho`) hooks in `_evict`, the
one place every schedule evicts a row, right after the violation check, reusing its per-branch
current computation (shared helper, identical numbers).

## 3. Principles

1. **No-op = today's path.** A batch without the new features takes exactly today's code path:
   same pattern, same analysis, same kernels, no extra work (pinned by tests). Results compare
   up to rounding, not to the bit: gpusim2grid's first session of a process already differs
   from the later ones by ~3e-15, with no action at all. A strategy whose action changes
   nothing is its contingency's row up to rounding (another slot or chunk). Each feature is opt-in by
   the data the caller gives, not by a flag, and costs nothing when unused.
2. **Bus-branch in the solver, switches outside it.** The solver sees per-row *element → bus*
   assignments in lightsim2grid's own bus labelling (`n_busbar_per_sub`, `change_bus_*`), i.e.
   grid2op's `set_bus` semantics. Switches are translated to that before the solve (§ 6).
3. **lightsim2grid is the oracle.** Each row must equal a one-off lightsim2grid solve of the
   same grid with the same `change_bus_*` / `deactivate_*` / tap calls, at the suite's
   `solver_atol` / `residual_atol`.
4. **Report, don't raise, per row.** What a row asks and the batch cannot do yet goes through
   `row_unsupported_` (NaN row, `BATCH_UNSUPPORTED`), as the element trips do.

## 4. Step A: contingency × action product (G1, G5, G6, G7), no new physics

Actions that only disconnect elements (~70 % of the case) are contingencies already: row
`(c, a)` = trips of `c` ∪ trips of `a`. That runs **today** through `ContingencyAnalysisGPU`
(sparse Sbus deltas, no dense Sbus) — first measurement, § 8.2.

Two layers, as in the prototype (its analysis structure, gpusim2grid's engine):

```python
# the engine: rows on one session, one pattern (the union, § 5), one compute()
ca = g2g.ContingencyAnalysisGPU(
    grid, ...,
    compute_limit_violations=True,
    violation_threshold=0.95,             # lightsim2grid's knob
    strategy_condition=g2g.StrategyCondition.ANY_VIOLATION,   # pypowsybl's condition, on the device
    strategy_warm_start=True,             # (default) each strategy from its own contingency's state
)
ca.add_contingencies_by_branch_id(...)            # C rows, as today
ca.set_actions(actions)                           # A gpusim2grid.actions.Action (§ 7)
ca.add_strategies(ctg_idx, action_idx)            # or [(c, [a, ...]), ...]; rows after the C
ca.compute(batch_size=...)                        # contingencies, then their strategies, streamed
rho, branch = ca.get_max_loading()                # on the device, per row
```

**Streamed strategies.** On the continuous schedule, the contingency rows enter the slots first
and each releases its strategies into the queue as soon as it finishes (the driver's `RowFeed`,
slots topped up after each eviction): no wait for the whole N-1. The condition is read on the
device from the contingency's violation counts; the warm start copies the contingency's converged
voltages into its start group (and, with the outer loops, keeps its resume point) at that moment.
With `StrategyCondition.TRUE` and no warm start nothing depends on a contingency, every row is
queued at once (either schedule). The security-analysis parameters (`strategy_condition`,
`strategy_warm_start`, `violation_threshold`) are constructor arguments, fixed for the instance.

**No second class.** The strategies are this engine API, not a separate "security analysis"
object: gpusim2grid has one public entry point per workload, lightsim2grid deprecated its
`SecurityAnalysis` name for `ContingencyAnalysis`. Named contingencies / actions / strategies in
pypowsybl's shape belong in a thin layer outside gpusim2grid.

**`violation_threshold`** (lightsim2grid parity, same name and semantics as its
`ContingencyAnalysis` / `ScenarioSweep.violation_threshold`), on `ContingencyAnalysisGPU` and
`ScenarioSweepGPU`: a float in `]0, 1]` (default 1.0 = today) moving every checked limit towards
a healthy anchor, `effective = t · limit + (1 − t) · anchor` (anchor 0 for currents, the bus
nominal voltage clamped into `[vmin, vmax]` for both voltage bounds); records keep the
configured limit; a constructor argument. It is what makes the violation condition usable with
a security margin (pypowsybl's limit reductions).

- **Union, not sum.** A row applies the union of the element statuses of `c` and `a`: an
  element both trip is tripped once (two triplet lists summed would subtract its admittance
  twice). An action reconnecting what `c` trips is a conflict: reported per row
  (`BATCH_UNSUPPORTED`), not solved.
- **No materialization.** Host data stays O(C + A): the per-row segment of `(c, a)` is the
  concatenation of `c`'s and `a`'s segments (dedup on the host when both touch an element),
  built when the row is loaded into a slot (`load_slots` / `SegmentGather` already work row
  by row).
- **Connectivity** of `(c, a)`: the base DFS tree answers "removed edges" sets of any size
  (`compute_component_masks`), so a pair is one more query, no new graph search.
- **Warm start (G5)**: `(c, a)` starts from `V_c`, written when `c` finishes in the same run.
  ScenarioSweep's per-row start (`set_v_init_from_orig`) holds a dense `n_rows × n_bus` buffer
  on the device (17 GB for the French strategies); every strategy of a contingency starts from
  the same V, so the contingency session keeps **one start per contingency** (C × n_bus, 22 MB
  there) and a row → contingency index, gathered at slot load. On by default
  (`strategy_warm_start=True`, OpenLoadFlow's semantics): it changes the iterates, not the
  solution. Expected: fewer Newton
  iterations, since most actions are local. With the outer loops, the row also resumes from where
  its contingency's loops ended (moved targets, frozen buses, switched SVCs, hvdc regimes, taps,
  sections, parked units): the outer model keeps one resume point per contingency row as it ends
  (`OuterHook::keep_row_points`) and each strategy row, loaded after it, points at its own
  contingency's (`set_row_resume`) -- lightsim2grid's batch row resuming from a solve (`init_from_n_powerflow`)
  on the grid with the contingency applied; its DistributedSlack shares only the loss its actions
  add. Checked against lightsim2grid's `ContingencyAnalysisCPP` / `ScenarioSweepCPP` (topology
  actions) resumed from the contingency's solve (`test_topo_outer_warm_start.py`).
- **One run**: contingencies and strategies share one driver and one analysis per
  `compute()`. A later `compute()` with other rows rebuilds them (the continuous schedule
  rebuilds every run in both sessions, `continuous_batching.md` A.6, not done upstream).
- **Reduction (G7)**: `get_max_loading()` → per row `(max |I|/I_max, branch id)` computed on the
  device from the slot's V, before eviction; flows only on request for selected rows.
- **Perf settings** for every measurement here: `linear_solver='own_lu'`,
  `scheduling='continuous'`, the tail from the start (`tail_max_groups=batch_size`,
  `tail_group_size=1`: with the own LU the group size is only the compaction granularity;
  `tail_group_size=batch_size` turns the tail off, `tail_ok` requires G < S).

## 5. Step B: union sparsity pattern (G2, G3) — the core addition

### 5.1 Value changes (G2): no pattern change

A phase shifter tap / ratio change is a new branch admittance `(yff, yft, ytf, ytt)'` on the 4
entries the branch already has: four triplets `y' − y`, resolved in the base pattern. API:
`set_actions` takes `branch_admittance={branch_id: (yff, yft, ytf, ytt)}` per action (and a
helper from lightsim2grid's `change_trafo_phase_tap` / `change_shift_trafo` that computes it).
Flows and limits read the row's admittance (`compute_flows` today reads the base `yff_eff`...).

### 5.2 Bus splits and merges (G3): spare buses, union pattern

A **split** moves some elements of bus `b` to a bus `b'` the base case does not use (bus 2 of
the substation in lightsim2grid's labelling). A **merge** puts the elements of `b1` and `b2`
on one bus. Both change the Ybus pattern. gpusim2grid keeps **one pattern per batch** and makes
it the **union** of what the batch's rows need:

1. **Spare buses.** Every `(substation, local bus)` some row uses and the base case does not
   gets a solver bus, appended after the base buses (no existing index moves, as
   `add_switchable_vm_buses` appends its rows). It carries a P and a Q equation, a theta and a
   Vm unknown (a PQ bus; a generator moving there makes it PV by value, the switchable-Vm
   mechanism already does PV ↔ PQ per row).
2. **Union Ybus pattern.** For each moved branch end `(branch, side)` from `b` to `b'`: the
   entries `(b', b')`, `(b', j)`, `(j, b')` with `j` the other end. The base values there are 0.
3. **Union J pattern** derived from the union Ybus by the ledger (`P`/`Q` rows × `theta`/`Vm`
   columns of the touched buses), appended like the existing supersets.
4. **Per row, by value only**: the moved branch's admittance is subtracted from `b`'s entries
   and added to `b'`'s (triplets, now resolvable in the union pattern); a moved injection is an
   Sbus delta from `b` to `b'`; a spare bus no row element sits on is masked (identity row,
   NaN), the existing mask mechanism; so is a base bus every element of which left.
5. **Merges as splits of the base**: a coupler closing between `b1` and `b2` is solved by
   moving `b2`'s elements to `b1` (the same triplets), `b2` then empty and masked. No
   zero-impedance branch, no V1 = V2 row.
6. **Efficient no-op.** No spare bus and no new entry ⇒ the union is the base pattern, the
   ledger is not extended, the analysis is the one of today: bit-identical (test). Above that,
   the extra cost is the fill of the union pattern, paid by every row of the batch.
7. **Built once, upfront.** The union covers every strategy registered on the analysis and is
   built before the first run; the contingencies, the strategies and every chunk solve on it. (The prototype
   builds a smaller union per block of strategies.) If building it upfront ever shows, it can be
   streamed while the GPU works on the first rows, which needs a join of patterns: one that only
   grows, with a new analysis when it does.
8. **Pattern banks (later).** When the union grows too much (many splits spread over the grid),
   group the rows by pattern signature into banks, one analysis each (the prototype's
   "one pattern per bank"). Not in the first version: measure the union's fill first (§ 8.3).

Per-row data then: the row's bus of every moved element end (for flows, limits, connectivity:
the connectivity check runs on the union graph with the row's removed / moved edges; a split
can island a bus, then the mask mode applies).

### 5.3 What stays out of the first version (reported per row)

- a moved element that is a voltage-control group member, an HVDC end or an SVC (the bridge's
  group data is per bus): `BATCH_UNSUPPORTED`;
- outer loops (`outer_loops=True`) with moves: the loops' per-bus plans are base-bus keyed;
- gradients (`BatchPowerFlow`) through moves;
- the angle reference moved off its bus.

## 6. Switches (G4): later, in lightsim2grid

The solver never sees switches (principle 2), and gpusim2grid gets **no switch API in these
PRs**: its stable interface is the per-element bus move (`Action`, § 7).

- **Later, from lightsim2grid**: its *detailed topology* (branch `dev-switches`, unreleased:
  `init_from_pypowsybl(net, detailed_topology=True)`, `update_switches` projects switch
  positions onto element buses with pypowsybl's own bus rule) gives exactly the moves of a
  switch action. Once released, a switch action becomes an `Action` on the host, once per
  action, with no pypowsybl at run time, and contingency propagation can live there too.
- **Meanwhile, for validation only**: the end-to-end benchmark
  (`benchmarks/topology_actions/`, § 8) translates the French case's switch actions itself:
  apply each to a pypowsybl variant, compare the bus **partitions** of the touched voltage
  levels before / after (pypowsybl renames the buses of a voltage level whose topology
  changes), write the new partition as moves (`Action.from_partition`). Not public API.

## 7. Step D: action format and results

- An action (`gpusim2grid.actions.Action`, frozen): per element kind, `{element id: global bus
  id}` in lightsim2grid's own `change_bus_*` convention (local bus `k` of substation `s` is
  `s + (k − 1) · n_sub`), `-1` = disconnected: `line_bus1`, `line_bus2`, `trafo_bus1`,
  `trafo_bus2`, `load_bus`, `gen_bus`, `sgen_bus`, `shunt_bus`, `storage_bus` (`-1` on one end of
  a branch opens that end only, as a contingency's `branch_ids_side1` / OpenLoadFlow's breaker
  at one end; on both ends the branch trips); and branch values
  `trafo_ratio` / `trafo_shift_deg` (`change_ratio_trafo` / `change_shift_trafo_deg`).
  `Action.apply_to(grid)` makes exactly those calls on an `LSGrid` (the oracle's semantics);
  `Action.validate(grid)` refuses an unknown element, a bus out of range or in another
  substation. `Action.from_partition(grid, sub, groups, disconnected=())` builds one from a target
  partition of a substation (the prototype's nodal action: each group keeps the bus now holding
  most of its elements, the others take free local buses). Disconnection-only actions are a
  special case of moves.
- On `ContingencyAnalysisGPU`: `set_actions(list[Action])`, `add_strategies([(ctg_index,
  [action indices]), ...])`; rows = the contingencies, then the strategies; `batch_bus_ids`
  (global bus of every voltage column: the base solver buses, then the spare ones),
  `n_spare_buses` (0 on the no-op path), `get_max_loading()` → `(rho, branch)` per row,
  `compute()` (contingencies, then their strategies), `get_strategy_ran()`, `get_iterations()`;
  constructor arguments `strategy_condition`, `strategy_warm_start`, `violation_threshold`.
- Results per row: status (converged / diverged / not simulated / unsupported), residual,
  max loading + its branch, the violations; full V / flows on request.

## 8. Validation and measurements

### 8.1 Data

- **Unit / oracle**: IEEE 14 and 118 (pandapower → lightsim2grid, `n_busbar_per_sub=2`): random
  splits, merges, disconnections and taps, each row vs a one-off lightsim2grid `ac_pf` after
  the same `change_bus_*` / `deactivate_*` calls. Fixtures and tolerances of the suite.
- **No-op**: every existing test unchanged; a batch with an empty action list bit-identical to
  the same batch without `set_actions`.
- **End to end** (not in CI, `benchmarks/topology_actions/`): the French 200 × 800 case above,
  one snapshot (data not committed), vs the prototype's results (statuses, max loading, flows),
  with its time per snapshot next to gpusim2grid's.

### 8.2 First measurement (step A, today's code)

Disconnection-only strategies through `ContingencyAnalysisGPU` as plain contingencies:
time, iterations, statuses and flows vs the prototype. Gives the baseline of everything else
and the share of time spent outside the GPU (host prep of 160 000 rows).

**Result (2026-11-16, after steps A, A2, B.1)**, `benchmarks/topology_actions/conti_x_actions_pypowsybl.py`
on a French snapshot (2024-05-20 07:00, 190 contingencies, 152 000 strategies, 800 action sets
translated in 79 s), against the prototype's per-situation results (OpenLoadFlow parity, no
contingency propagation); gpusim2grid with OpenLoadFlow's loading rules and the outer loops on
every row, the permanent current limits of the network:

| kind | situations | not run yet | status agrees | \|Δ max rho\| p50 | p99 | max | same worst branch |
|---|---|---|---|---|---|---|---|
| contingency | 185 | 0 | 100 % | 1.6e-8 | 1.3e-6 | 1.0e-5 | 100 % |
| disconnection | 71 780 | 0 | 100 % | 1.9e-8 | 9.4e-6 | 1.8e-4 | 100 % |
| no change | 58 275 | 0 | 100 % | 1.5e-8 | 1.6e-6 | 1.0e-5 | 100 % |
| phase shifter tap | 740 | 0 | 100 % | 1.5e-8 | 8.0e-7 | 1.0e-5 | 100 % |
| bus split / merge | 11 840 | 11 840 (B.2) | | | | | |

29 action sets name elements this snapshot lacks. Times (a GPU shared with another job:
indicative only): phase 1 0.4 s, phase 2 115 s for 146 490 strategies in chunks of 10 000.

**Result (2026-11-17, B.2 done, phase 2 warm-started from the contingencies, outer state
included)**, same inputs, GPU 1 (free):

| kind | situations | not run | status agrees | \|Δ max rho\| p50 | p99 | max | same worst branch |
|---|---|---|---|---|---|---|---|
| contingency | 185 | 0 | 100 % | 1.6e-8 | 1.3e-6 | 1.0e-5 | 100 % |
| disconnection | 71 780 | 0 | 100 % | 1.7e-8 | 6.0e-6 | 2.0e-4 | 100 % |
| no change | 58 275 | 0 | 100 % | 1.5e-8 | 1.6e-6 | 1.0e-5 | 100 % |
| phase shifter tap | 740 | 0 | 100 % | 1.5e-8 | 3.5e-7 | 1.0e-5 | 100 % |
| bus split / merge | 11 840 | 0 | 100 % | 2.3e-8 | 6.5e-6 | 2.6e-5 | 100 % |

Phase 1 0.5 s, phase 2 82 s. The moves include 570 strategies moving the closed end of a
half-open branch and 380 moving a generator (one of them a member of a voltage-control group).

lightsim2grid note (no reference for warm generator moves under the outer loops): its
ScenarioSweep with ``NROuter_*`` leaves a generator a topological action moves unregulated on
its new bus (Q = 0, PQ), from scratch or resumed; its one-off solve and its ScenarioSweep
without outer loops regulate it. The from-scratch rows are checked against the one-off solve,
the warm ones against the resume rules (test_topo_outer_warm_start.py).

### 8.3 Union pattern cost

On the French case: base nnz(J) vs union nnz(J) with all the case's splits, LU fill (own LU
plan), factor / solve time per batch, before and after.

## 9. Order of work and PRs

| step | content | PR | status (2026-11-17) |
|---|---|---|---|
| 0 | this note; build + suite on this machine; § 8.2 measurement script | 1 | note, tests, tooling done; § 8.2 waits for a free GPU |
| D | `Action` format (`set_bus`-style moves, branch values), results | 1 | done |
| A | strategy rows (union of trips), max loading, `violation_threshold` | 1 | done (Python row map over today's session; max loading reduced on the device) |
| A2 | warm start: one start V per contingency on the device, gathered at slot load | 1 | done (C++ unit tests for the gather) |
| A6 | streamed strategies: one `compute()`, a contingency releases its strategies as it finishes (`strategy_condition` on the device, warm start and outer resume point kept in the same run); the two-phase `rows=` / `select_strategies` API removed | 1 | done |
| B.1 | per-row branch admittance (taps) | 2 | done (row triplets in C++; the changed branch's flows / current violations / max loading from the row's own block on the device, `set_row_branch_blocks`) |
| A5 | warm start with the outer loops: each strategy resumes from its contingency's loop state; per-row accessors (`get_row_status`, `get_row_iterations`, `get_outer_results`, physical checks) on the caller's rows | 1 | done |
| B.2 | union pattern + spare buses + moves (splits / merges), no-op = today's path | 3 | done: S1-S3 (also under the outer loops, warm or not), S4 moved-branch flows / violations, S5 generator moves (PV on the new bus, group members, under the outer loops: spares switchable in the outer model), half-open branches' closed ends; the contingency wins over an action's move of its elements |
| — | switch actions → moves (lightsim2grid detailed topology), pattern banks; groups / gradients with moves | later | |

The tests come first (`tests/python/topology_actions/`, red until their step lands), one file
per step:

| file | step | pins |
|---|---|---|
| `test_topo_action_api.py` | D | `Action`: frozen, `apply_to` = lightsim2grid's calls, `validate`, `from_partition`, `combine` |
| `test_topo_noop.py` | principle 1 | no actions / no-op actions: today's path, no spare bus |
| `test_topo_trips.py` | A | strategy rows = union of trips, lightsim2grid, a trip counted once, the contingency wins over an action, conflicts per row, array input |
| `test_topo_branch_values.py` | B.1 | taps / ratios vs lightsim2grid, flows with the row's admittance |
| `test_topo_bus_moves.py` | B.2 | splits, merges, a generator moved (PV), a group member moved, the closed end of a half-open branch, islands, 8 splits in one union pattern, flows / loading / violations of moved branches, refused moves, analysis reuse |
| `test_topo_results.py` | A, A6 | max loading on the device, warm start (lightsim2grid's solution, each strategy from its own contingency whatever the order), the violation condition (`violation_threshold`), per-row statuses / outer results with strategies not released, the continuous schedule required |
| `test_topo_outer_warm_start.py` | A5 | strategies resumed from their own contingency's outer-loop state vs lightsim2grid's resumed batch (also with the strategies shuffled), a no-op strategy gives its contingency back (its loss not shared twice), splits and generator moves under the outer loops (warm and from scratch) |
| `../violations/test_violation_threshold.py` | A | `violation_threshold` vs lightsim2grid's `ContingencyAnalysisCPP`, both facades, clamped voltage anchor, records not rescaled, the constructor argument |

Code conventions for the new modules (those of the prototype): `from __future__ import
annotations`, typed signatures and module-level variables, frozen dataclasses for values,
`__all__`, a usage docstring per module; ruff (`[tool.ruff]` in `pyproject.toml`) on the new and
changed files only, with `ANN` on the package's new modules. On a shared machine the tests run
through `tools/run_limited.sh` (memory / core caps, the GPU left to `CUDA_VISIBLE_DEVICES`):

```bash
CUDA_VISIBLE_DEVICES=2 tools/run_limited.sh -m 32 -c 8 -- \
    python -m pytest tests/python/topology_actions tests/python/violations
```

## 10. Limits (to be measured, estimated here)

- **10× contingencies** (2 000 × 800 = 1.6 M rows): step A's host data is O(C + A), fine; the
  GPU time grows linearly. The union pattern does not depend on C.
- **10× actions**: the union pattern grows with the *distinct* moved element ends, not with
  the number of actions; spread over many substations it adds fill to every row (→ banks).
- **Bigger networks**: the union's extra entries are local (a few per moved end); its effect
  on the LU fill is not local and must be measured (§ 8.3).
- **Different actions per contingency** (sparse pairs): `add_strategies` takes any pairs; the
  union is over the pairs actually given.
- **Outliers**: rows that need many iterations (an action close to voltage collapse) keep the
  tail busy; continuous batching + tail is the lever already there.
- **Outer loops**: a row with moves and `outer_loops=True` is unsupported in the first version;
  that is the biggest functional gap against OpenLoadFlow-like results.
- **Injection sweeps** (same topology, different loads / generation): orthogonal; a later
  product of the three axes.
