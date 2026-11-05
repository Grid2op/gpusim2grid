# A batched LU of our own, and lightsim2grid's constrained ordering

**Design note, not documentation.** Written on 2026-11-03, after reading olf-gpu's batched LU
(github.com/gautierbureau/olf-gpu at `9d2bd515`, `native/cuda/lu/`) and lightsim2grid's
constrained ordering (`_lightsim2grid` at `ce7efa11`, `KLUStaticPivotSolver`, `OrderingConstraints`;
the paper plan `paper_plan_fixed_pattern_nr.md` of the `lightsim2grid_outerloops` checkout). Nothing
here is implemented except the prototype of the last section; the numbers quoted are those
repositories', not measurements of gpusim2grid.

## 1. What olf-gpu did

olf-gpu (OpenLoadFlow's Newton on the GPU, same uniform-batch cuDSS as here) added a batched sparse
LU of its own on 2026-10-30 (`e2419e87`, `e61249df`, `c3214e82`), opt-in
(`GpuLoadFlowParameters.linearSolver = BATCHED_LU`, "mode 7"; cuDSS stays the default and the robust
single-matrix retry).

- **Not a new ordering**: it runs cuDSS's ANALYSIS and reads `CUDSS_DATA_PERM_REORDER_ROW/COL`
  back (`cudss_order.cu`), so its fill is cuDSS's `LU_NNZ` exactly. It replaces the numeric
  factorization and the solve only.
- **Fast engine** (`scalar_lu_plan`, `scalar_lu`): left-looking, one GPU item per LU entry
  gathering its contributions in pivot order (no atomics, deterministic), level-scheduled
  triangular solves, values interleaved by lane (`entry * capacity + lane`, a warp = 32 lanes of one
  entry), **no pivoting** (a pivot below 1e-13 is clamped and the lane flagged, as is a multiplier
  above 1e8). It factorizes only the lanes of a lane map: a converged scenario costs nothing.
- **Pivoting engine** (`supernodal_lu_plan`, `supernodal_lu`): supernodes of the symmetrized
  Cholesky structure, complete pivoting inside each supernode's diagonal block, per lane, as cuDSS
  pivots (`CUDSS_PIVOT_AUTO` → local block). It takes the flagged lanes only.
- **Iterative refinement** reimplemented as cuDSS's SOLVE phase does it (`IR_N_STEPS` inner steps),
  without which their LU fell behind cuDSS on ill-conditioned busbar systems.
- **Why it is faster**: cuDSS's uniform batch "factorizes every slot of the bank, an idle one as an
  identity system" (rte6515, 1000 branch outages: 2.2 s of a 4.5 s batched solve).
- **Measured** (RTX A1000 laptop, warm, end to end): rte6515 1000 outages 5970 → 4329 ms (×1.4),
  a real grid with 470 contingencies 13002 → 7251 ms (×1.8). Same statuses as the CPU, steps equal to
  cuDSS's to ~1e-12. Lanes needing the pivoting engine: 0 of 5416 on rte6515, 16 of 9306 on the
  real grid.

Their API already matches our layout: `factorizeLanes(lu, vals, laneMap, width, stream)` reads slot
`s`'s CSR values at `vals + s * nnz`, `solveLanes` reads/writes `b`/`x` at `+ s * n`, exactly
`d_J_values_batch` / `d_F` / `d_dx`. FP64 only (hard-coded `double`). MPL-2.0, like gpusim2grid.

### What it would buy here, without anything else

The gain olf-gpu measured is mostly "do not factorize idle lanes". Continuous scheduling already
keeps the slots full (`docs/dev_notes/continuous_batching.md`), so here it reduces to:

1. **Per-lane speed** against cuDSS's uniform batch at the same S, unknown.
2. **The tail**: no 16-matrix cliff and any subset of lanes, so the tail could run on a shrinking
   lane map instead of `_compact_tail` + `TailSolve` groups of 15. A simplification as much as a
   speed-up; weaker under cuDSS 1.0.0, which removes the cliff (`cudss_ubatch_mask.md`).

## 2. What lightsim2grid's constrained ordering solves

A refactorization keeps the pivot sequence of the last factorization. When the outer loops switch a
row **by value** between two forms -- a voltage row `Vm(reg) = target` ↔ `Q_c = q`, a ratio-group row
`Vm(bus) = target` ↔ `ratio = held` -- a pivot chosen in one form sits on an entry the other zeroes,
and the refactorization meets an exact zero pivot (a "fallback": KLU factorizes again).

`OrderingConstraints` (`linear_solvers/OrderingConstraints.hpp`), declared by the Newton system
(`NRSystem::ordering_constraints()`, the `declare_ordering_constraints` of `BranchControl`,
`ShuntControl`, VoltageControl): "eliminate these follow indices after those lead ones" (the leads
are Vm columns, the follows controller unknowns and custom rows), plus `in_order` for a shared
VoltageControl group (its Q columns in the controllers' order). `constrained_order`
(`KLUStaticPivotSolver.cpp`, `LS2G_API`) post-moves an unconstrained AMD (or min-fill) order of
`A + Aᵀ` so every constraint holds; the Vm pivots on the (never switched) Q rows then fill the
switched row's other entry in, and **one pivot order is valid in both forms**. The order is
**symmetric** (`klu_analyze_given(n, Ap, Ai, perm, perm)`), designed for pivots on the diagonal
("these rows sit on the diagonal of the column they pivot on").

From the paper plan (one core, KLU static-pivot refactorization):

- Real snapshot, 1000 N-1 with the outer loops: fallbacks 1781 → 0, batch −18 % on static pivot.
- 100-grid run, all 7 loops, before `in_order`: 46 148 fallbacks (≈ 0.05 per row), nearly all on
  shared VoltageControl groups. With `in_order`, 12 grids: 14 434 → 444, "about 1 per 300 rows in the
  worst case". Converged states identical to ~3e-12.
- Cost: the constraints add about 25–30 % refactorization flops over unconstrained AMD, ~97 % of it
  from ratio groups. Public grids without outer loops: no constraint is declared, nothing changes.

## 3. Why the two fit together

olf-gpu's fast engine **is** a static-pivot refactorization, on the GPU: one elimination order for
every lane, no pivoting, a fallback (its pivoting engine) for the lanes whose pivots go bad. It
borrows cuDSS's order, which knows nothing of the row forms. lightsim2grid's ordering is exactly what
makes such an engine safe when the rows switch form, and it was designed for the same diagonal-pivot,
symmetric-order setting.

The GPU needs it more than the CPU: in a batch the lanes sit in **different** forms at the same time
(per-slot pins, held controllers, BranchControl rows, `handle_disconnected_grid` rewrites) on one
shared order. A frozen pivot that one form zeroes costs

- **under cuDSS**: a perturbed pivot (`CUDSS_DATA_NPIVOTS`; cuDSS still pivots inside its
  supernodes, so not every such case is a perturbation). A plausible cause, **not verified**, of the
  inner solves that diverge under `outer_loops=True` unless `ir_n_steps=1`;
- **under olf-gpu's LU**: that lane on the slow pivoting engine.

The ordering needs no new LU to be tried: cuDSS takes a user permutation (`CUDSS_DATA_USER_PERM`,
present in the 0.8.0 and 1.0.0 headers, `cudss_data_types.h`; that the uniform batch honours it is
to be checked).

### Where it should matter in gpusim2grid

| mode | expected effect |
|---|---|
| plain N-1 / injection sweep, no outer loops | none: no constraint declared (and olf-gpu saw 0 pivoted lanes on rte6515) |
| `outer_loops=True` | the case it was built for. The outer ledger *is* lightsim2grid's (`make_*_outer_session_from_lsgrid` poses it as it is), so its indices and constraints carry over unchanged |
| `handle_disconnected_grid`, `set_contingency_gens`, the reactive-limits second pass | probably: rows rewritten by value the same way (stranded group → `Q_first = 0`, pinned `Q_c = t`, PV→PQ). Not measured. Our ledger extensions (`reserve_stranded_controller_slots`, `add_switchable_vm_buses`, `move_reference`) change the pattern: lightsim2grid's order applies only where the pattern is still its own |

lightsim2grid's fallback count with `NROuter_KLUStaticPivot` (constrained ordering on) is already a
prediction of the lanes that would need pivoting on the GPU: same ledger, one order for the batch,
diagonal pivots.

## 4. Plan

0. **Speed probe** (½ day): time olf-gpu's `factorizeLanes` + `solveLanes` against our uniform batch
   on real Jacobians at S = 8 … 1000 (a `backend=` on `benchmark_cudss_batch_raw`). Go / no-go for
   step 2.
1. **lightsim2grid's order through cuDSS** (≈ 1 day, the prototype below): no new LU. Hand cuDSS the
   constrained order, compare perturbed pivots, `LU_NNZ`, iterations and the `ir_n_steps=0` failures
   under `outer_loops=True`.
2. **Port olf-gpu's LU** behind `CudssBatchSolver` (≈ 3–4 days after step 0, + 1–1.5 days to feed it
   step 1's order instead of cuDSS's):
   - vendor `lu/` pinned to `9d2bd515`, with shims for its `device_buf` / `check` / stats helpers
     (½ day);
   - a backend inside `CudssBatchSolver` rather than templated policies: `initialize` still runs
     the ANALYSIS (for the order) and builds the `LuPlan`; `factor` / `refactor` → `factorizeLanes`;
     `solve` → `solveLanes`; `bind_slice` → a lane map (1 day);
   - `ir_n_steps`: r = b − Jx, J d = r, x += d on our batched SpMV, as their
     `solveWithInternalRefinement` (½ day);
   - `linear_solver = 'cudss' | 'batched_lu'` in `BatchDriverConfig` and the three batch facades,
     refused in FP32 and for strategies other than `direct_refactor_every` (½ day);
   - batch tests on both backends (same results to rounding, same iterations and statuses), benchmark
     on real grids (1 day).
   - Left out at first: FP32, `direct_base_case_factors` / `direct_iter0_only` (factor reuse across
     slots), the Jᵀ adjoint and the single-system base case (stay on cuDSS).

Open questions: does the uniform batch honour `CUDSS_DATA_USER_PERM`; what convention does it expect
(new → old as AMD's `P`, or its inverse); does a GPU pay the 25–30 % extra fill on every lane more
than it saves on fallback lanes (the paper's cost model, `N_fb / N_ref > Δc_ref / (c_ref + c_fac)`,
with a far more expensive `c_fac` on a GPU); how often our own ledger extensions break the pattern
match.

## 5. Prototype: lightsim2grid's permutation through cuDSS (2026-11-03)

### What was built

- lightsim2grid (`b8bf92bb`, pushed to `dev_gradients`): `BaseAlgo::get_ordering_constraints()`,
  read-only, through `NRAlgo` / `OuterLoopAlgo` / `AlgorithmSelector` like `get_group_v_row`.
- `extract_ledger_data` reads the constraints into `LedgerData::ord_*` (flattened);
  `drop_multislack_augmentation` clears them (it deletes J indices). Appending extensions and
  `move_reference` keep them valid (constraints only name Vm columns and controller unknowns).
- `reordering_alg='ls2g_constrained'`: lightsim2grid's own `constrained_order` (AMD + post-move) on
  the **final** J skeleton, through a hook the bridge installs (`user_ordering.hpp`; the bridge is
  optional, CUDA code cannot link lightsim2grid).
- `reordering_alg='ls2g_constrained_nd'`: cuDSS's own order of that skeleton (a throwaway single-system
  ANALYSIS, `cudss_reorder_perm`) post-moved by the same constraints (`apply_ordering_constraints`,
  `ledger_extend.cpp`, a copy of lightsim2grid's post-move: it only exports it fused with AMD /
  min-fill). `GPUSIM2GRID_ND_NO_POSTMOVE=1` hands cuDSS's order back unchanged (diagnostic).
- Both computed in `AcPfNrState` and handed to every forward/adjoint batch solver
  (`CudssBatchSolver::initialize(..., user_perm)` → `CUDSS_DATA_USER_PERM`); the single-system base
  case keeps cuDSS's default. Refused in array mode and with `use_distributed_slack=False`.
- Diagnostics on `BatchTimings`: `cudss_user_perm_honoured` (cuDSS's `PERM_REORDER_ROW/COL` after
  ANALYSIS == the permutation given), and with `GPUSIM2GRID_COUNT_PIVOTS=1`
  `cudss_npivots_total/max/reads` (`CUDSS_DATA_NPIVOTS` after each factorization, host sync; **the
  tail's own handle is not counted**, so a continuous run reports only its non-tail rounds).
- `benchmarks/outer_loops.py --reordering_alg ... --cpu_algo ...`; `tests/python/test_ls2g_constrained_ordering.py`
  (cuDSS uses the order, same statuses / outer iterations / V as cuDSS's own order on case14 with a remote
  group, refusal of a renumbered ledger).

### Measured

French snapshot (`recollement-auto-20230119-1900-enrichi`, loops DistributedSlack /
AcHvdcAcEmulationLimits / VoltageMonitoring / ReactiveLimits), `ContingencyAnalysisGPU(outer_loops=True)`,
300 random N-1, batch 128, cuDSS 1.0.0, RTX A1000 laptop (relative numbers only). Statuses
(213 converged, 1 max-iter, 86 not simulated) and outer iterations identical in every run.

| order | user perm honoured | LU nnz | perturbed pivots (7 factorizations) | Newton rounds |
|---|---|---|---|---|
| cuDSS default | – | 176 018 | 2 688 (384 each) | 2.0 s |
| cuDSS `nested_dissection` | – | 176 018 | 2 688 | 2.0 s |
| cuDSS `amd` | – | 150 516 | 2 688 | 17.2 s |
| `ls2g_constrained` (AMD + post-move) | yes | 150 664 | **0** | 17.0 s |
| `ls2g_constrained_nd` (cuDSS ND + post-move) | yes | 246 956 | **0** | 11.9 s |
| cuDSS ND handed back unchanged (`GPUSIM2GRID_ND_NO_POSTMOVE`) | yes | 246 672 | 896 | 12.1 s |
| (inverse of cuDSS's ND handed back) | – | – | – | factorization fails (status 2) |

`ir_n_steps=1` changes none of the counts (Newton rounds +25–30 %). lightsim2grid on the same rows:
`NROuter_KLU` and `NROuter_KLUStaticPivot` both 0 fallback factorizations in 1 529 refactorizations,
agreeing with the GPU to 1.8e-12.

### What it shows

1. **cuDSS's uniform batch honours `CUDSS_DATA_USER_PERM`** (1.0.0), with the AMD convention (new →
   old): the inverse gives a factorization that does not fit.
2. **lightsim2grid's constraints remove every pivot cuDSS perturbs** (2 688 → 0), whatever the base
   order; AMD alone does not (2 688). They are structural (3 per system, in every factorization, on a
   case where KLU needs no fallback), i.e. pivots on entries the reserved union pattern holds at zero.
3. **But cuDSS loses its nested-dissection structure under any user permutation**: handed its *own*
   order back, its fill grows 40 % and its rounds are 6× slower. Its ND path uses more than the
   permutation it reports (separator tree / supernode partition, presumably). An AMD-type order is
   slow in cuDSS whatever its source (8.7×, with *less* fill): its elimination tree has too little
   parallelism for the GPU. **Through cuDSS, the constrained order costs far more than it saves.**
4. On this case the perturbed pivots are harmless (same statuses, outer iterations and V to 1.8e-12,
   with or without refinement). The `ir_n_steps=1` divergence that motivated the idea is not
   reproduced here; it needs a harder case: rows the outer loops switch so that a frozen pivot breaks
   (shared VoltageControl groups, ratio groups: the paper's worst grid, 9 008 fallbacks on 9 115 rows
   before `in_order`), and TransformerVoltageControl / PhaseControl / ShuntVoltageControl, which this
   snapshot does not run.

### Consequences for the plan

- Step 1 (the constrained order through cuDSS) is closed: correct, removes the perturbations, too slow.
- The constraints remain the right input for a factorization whose schedule we own (step 2, olf-gpu's
  LU). Before porting it, measure the **symbolic** fill and elimination-tree height of the candidate
  orders outside cuDSS (cuDSS ND, ND + post-move, AMD + post-move, min-fill + post-move) on real
  Jacobians: the 246 956 above includes cuDSS's user-permutation penalty, not the order's own fill.
- Find a case with perturbation trouble first (a grid with shared groups or the transformer loops, and
  the `ir_n_steps=0` divergence): without one, the ordering has nothing to fix on the GPU.

## 6. A third LU: gpu-security-analysis' (2026-11-05), and the go / no-go

A colleague's batched LU (github.com/RTE-AssistFlux/gpu-security-analysis, private, `engine/lu.py`,
`engine/kernels/lu.cu`, its `docs/own_lu.md`) differs from olf-gpu's on the point that matters here: it
**computes its own symbolic factorization for any order it is given** (Cholesky fill of the symmetrized
permuted pattern, level schedules, CPU, once per pattern), instead of borrowing cuDSS's. Same engine
otherwise: left-looking, one item per LU entry in pivot order, values interleaved by matrix, static
diagonal pivots with a relative clamp, no fallback, plus "split" kernels sharing the long contribution
chains of the narrow levels near the root over up to 32 warps (their largest gain). Measured there:
France, batch 2048, Blackwell: refactorization 13.9 vs 94 ms, solve 5.35 vs 47.3 ms against cuDSS 1.0.0;
MMD beats cuDSS's own order in it (fewer contributions despite more levels).

### Go / no-go on our Jacobians

`GPUSIM2GRID_DUMP_J` (section 7) on the snapshot of section 5 (300 N-1, outer loops, batch 128): 896 batch
Jacobians (dim 14 082, nnz 98 024) from every factorization of the run, their LU run as is (CuPy),
the orders read back from cuDSS. Laptop RTX A1000, cuDSS 1.0.0: timings are ratios only.

| order | LU nnz | levels | contributions | pivots < 1e-8·max\|A\| (no clamp) | max residual |
|---|---|---|---|---|---|
| ls2g_constrained | 150 664 | 180 | 597k | 0 (smallest 6.6e-8) | 6.7e-16 |
| MMD | 152 040 | 180 | 633k | 2 688 = 3 exact zeros per matrix | NaN |
| cuDSS default | 176 018 | 116 | 923k | 2 688, the same | NaN |
| ls2g_constrained_nd | 246 956 | 187 | 2 254k | 889 (none < 1e-12) | 6.3e-16 |

- The 384 pivots cuDSS perturbs per batch are these 3 exact, structural zeros per matrix.
- Clamp 1e-12·max|A| + 2 refinement steps brings MMD / cuDSS's order to 6e-16; a clamp of 1e-5 clamps
  genuine small pivots of the constrained order (6.6e-8) and stalls at 3e-6: keep it tiny.
- cuDSS's own solve on these matrices: residual 3.5e-7 (its perturbed pivots), 3e-15 with `ir_n_steps=1`.
- Refactorization + solve, ms (own LU with the constrained order / cuDSS uniform batch): S = 15: 5.0 /
  15.7 (3.1x); S = 128: 16.1 / 130 (8.1x); S = 512: 56 / 525 (9.4x). The ND + post-move order is 2.6x
  slower than AMD + post-move in this LU.

Go: with lightsim2grid's constrained order the static-pivot engine needs no fallback on this case.

On "refactorization": cuDSS's REFACTORIZATION costs as much as a FACTORIZATION in uniform batch (measured,
user, 2026-11-05): it redoes its local-block pivot search every time, and still perturbs those 3 zeros. The
own LU has a single numeric phase with a fixed pivot sequence, i.e. a static-pivot refactorization every
time. A fallback, if a case needs one, belongs there: refactor() counts the pivots it clamps per lane
(done), a flagged lane then gets a real factorization with pivoting (inside supernodes, which keeps the
shared pattern) or a slow path for that iteration (not done).

## 7. Implementation, first increment (branch `dev_customLU`)

- `src/_cpp/own_lu/lu_plan.{hpp,cpp}`: the symbolic plan, CUDA-free, a line-by-line port of their
  `LuPlan` (Liu's Cholesky rows, levels, left-looking items, level slices).
- `src/_cpp/own_lu/batched_lu.{cuh,cu}`: their kernels templated on `real_t` (FP32 builds), the U steps
  counting the pivots they clamp per matrix, iterative refinement (`r = b − A x` on the slot's CSR, solve,
  add) for `ir_n_steps`.
- `CudssBatchMode::OwnLu` (`GPUSIM2GRID_OWN_LU=1`, `GPUSIM2GRID_OWN_LU_EPS`, default 1e-12): cuDSS's
  uniform ANALYSIS still runs, for its order only (PERM_REORDER_ROW, i.e. the constrained order with
  `reordering_alg='ls2g_constrained'`); factor / refactor / solve / bind_slice go to the own LU, so the
  forward, the continuous tail, the Jᵀ adjoint and `benchmark_cudss_batch_raw` all use it. The single-system
  base case stays on cuDSS. `cudss_lu_nnz` reports the own LU's, `cudss_npivots_*` (with
  `GPUSIM2GRID_COUNT_PIVOTS`) its clamped pivots. `cudss_mode_is_uniform` covers both for the checks.
- `GPUSIM2GRID_DUMP_J=<dir>` (`GPUSIM2GRID_DUMP_J_MAX`, default 8): every batch solver writes its skeleton,
  orders and the J values of its first factorizations (raw files), for probes like section 6.
- Section 5's case with the own LU (`benchmarks/outer_loops.py`, `ir_n_steps=0`): statuses and outer
  iterations identical to cuDSS, max |dV| against lightsim2grid 1.7e-12 with cuDSS's order (2 688 clamped)
  and 1.8e-12 with the constrained one (0 clamped); Newton rounds 1 960 → 630 ms (laptop).
- Tests: `tests/python/test_own_lu.py` (raw solve, CA chunked / continuous with tail, injection sweep,
  outer loops with both orders and the clamp count, BatchPowerFlow gradient by central differences).

Next: the order without cuDSS's ANALYSIS (lightsim2grid's `constrained_order` is already computed by
`AcPfNrState`; AMD for the array mode); a facade option instead of the variable; Jacobian kernels writing
the interleaved layout (the transposes); a lane map for the tail instead of `_compact_tail` groups; the
adjoint reusing J's factors (Jᵀ = UᵀLᵀ) instead of a second factorization; timings on the RTX 6000.
