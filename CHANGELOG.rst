Change Log
===========

[0.3.1] (unreleased)
--------------------
- [IMPROVED] ``linear_solver='own_lu'``: its symbolic plan is built with counting sorts instead of a sort of every
  contribution (PtFige: 28 -> 12 ms per driver build; the same plan).
- [IMPROVED] ``ContingencyAnalysisGPU`` with ``outer_loops`` and ``redistribute_slack``: each row's lost power sums
  its own tripped generators, no longer every generator of the grid (PtFige N-1: 62 -> <1 ms of host per run).
- [IMPROVED] Prototype own_lu_top: programmatic dependent launch hides the gap between its dependent kernels
  (PtFige N-1: 0.518 -> 0.514 s; ``GPUSIM2GRID_TOP_LU_PDL=0``: off).
- [ADDED] Prototype ``GPUSIM2GRID_OWN_LU_TOP=1``: own_lu with its trailing elimination levels in shared memory
  (``src/_cpp/fused_lu/``; PtFige N-1 with outer loops 0.597 -> 0.517 s). Experimental, not tested in the suite yet.
- [FIXED] ``outer_loops_backend``: the host outer-loop model is compiled without FMA contraction, like the device
  one, so a borderline decision goes the same way on both on a ``-march=native`` build.
- [FIXED] ``outer_loops_backend='device'`` with ``redistribute_slack``: a row's lost power is shared on the GPU
  after its template copy, no longer by a host load per row (PtFige N-1: 1.85 -> 1.05 s, host 1.63 s).
- [FIXED] Build against a lightsim2grid compiled with AVX (``-march=native``): nvcc's Eigen ABI probe checks the
  alignment only, since Eigen never vectorizes in a CUDA file.
- [IMPROVED] N-1 x actions setup (host work before the first GPU round): 2.8 -> 0.9 s on PtFige, 36k strategies;
  connectivity certified without a search per row, generator trips walked per row, rows built without GC passes.
- [ADDED] ``docs/environment_variables.md``: every environment variable gpusim2grid reads, in one page.
- [ADDED] ``GPUSIM2GRID_ASYNC_TIMERS=1``: per-phase timers no longer wait for the GPU (their GPU times read 0), so
  the host keeps the GPU fed; opt-in.
- [IMPROVED] Strategies changing a branch (transformer values, bus moves): its current check, flows and max
  loading are computed on the GPU from the row's own block; no voltages come back to the host.
- [IMPROVED] ``ContingencyAnalysisGPU.set_actions``: the grid is walked once and copied once for all actions, not
  per action (771 actions on a French grid: 25 s -> 1.2 s).
- [IMPROVED] Fewer device synchronisations per round: grow-only pinned buffers, double-buffered segment uploads,
  no stream sync after a fill.
- [IMPROVED] ``linear_solver='own_lu'``: the triangular solves' runs of narrow levels run as one kernel each (the
  same bits; a third fewer kernel launches on a 7k-bus N-1).
- [IMPROVED] ``outer_loops=True``: a row disconnecting units or tripping lines takes its ReactiveLimits plan and
  DistributedSlack units by filtering the grid's (the same lists, a third of a row's load).
- [ADDED] ``outer_loops_backend`` (``OuterLoopsBackend``, a str enum: ``HOST`` default, today's path) on the three
  batch facades and ``BatchPowerFlow``: ``DEVICE`` runs all seven outer loops in CUDA kernels (FP64 builds;
  ``outer_device_census``).
- [ADDED] ``timings.n_oc_host_loads``: the rows whose outer-loop state the host model set up (``DEVICE``: the rows
  not copied from a template on the GPU), a load-independent measure of the CPU's share.
- [IMPROVED] ``outer_loops=True``: the unrealistic-voltage check runs one block per converged row instead of
  one thread (12 ms per call on a 7k-bus grid, a fifth of an N-1 run's GPU time).
- [IMPROVED] The Newton voltage update applies its step as a product (no ``atan2``/``sin``/``cos`` per bus):
  4x faster in FP64 on an RTX A1000.
- [IMPROVED] The Jacobian fill: fewer FP64 operations per entry, no warp divergence on the diagonal (4.6x faster).
- [IMPROVED] Ybus·V and the P/Q mismatch in one kernel instead of cuSPARSE's SpMV and two fills: time series 2x faster.
- [IMPROVED] ``scheduling="continuous"`` keeps its driver across ``compute()`` calls on the three batch facades
  (not with ``outer_loops``): no allocation or cuDSS analysis.
- [IMPROVED] ``scheduling="continuous"``: a round's first Newton step reuses the residual pass's mismatch.
- [IMPROVED] Batch drivers no longer allocate the block-diagonal Ybus CSR (batch × nnz_Y ints) unless cuSPARSE's
  SpMV is used.
- [FIXED] ``outer_loops=True``: an inner solve already within tolerance takes no Newton step, as lightsim2grid's;
  the extra step changed the loops' decisions.
- [BREAKING] ``outer_loops=True``: ``nb_nr_iterations`` / ``get_row_iterations()`` count the steps taken, 0 for an
  inner solve converged from its start.
- [ADDED] ``gpusim2grid.actions.Action``: a remedial action as element bus moves (lightsim2grid's
  ``change_bus_*`` convention), disconnections and transformer ratio / phase shift values.
- [ADDED] ``violation_threshold`` argument of ``ContingencyAnalysisGPU`` / ``ScenarioSweepGPU``:
  lightsim2grid's knob, limits moved towards a healthy anchor; records keep the configured limit.
- [ADDED] ``ContingencyAnalysisGPU`` strategies: ``set_actions`` / ``add_strategies`` (a contingency then
  actions, solved as one more row) in the same ``compute()`` as the contingencies: a contingency
  releases its strategies as it finishes (continuous schedule), arguments ``strategy_condition``
  (pypowsybl's ``"TRUE"`` / ``"ANY_VIOLATION"``, checked on the device) and ``strategy_warm_start``
  (default True: a strategy starts from its own contingency's converged voltages, as OpenLoadFlow's
  security analysis); ``get_max_loading`` reduced on
  the device (one value and one branch per row come back); ``get_violations`` /
  ``get_violation_counts`` / ``get_violations_truncated`` take a ``ResultRows`` (``ALL``,
  ``CONTINGENCIES``, ``STRATEGIES``) and build only those rows' records; the rows whose actions change
  a branch bring only their own voltages to the host.
- [ADDED] Strategy actions may change transformer values (``trafo_ratio`` / ``trafo_shift_deg``): per-row
  Ybus values on the base pattern; the changed transformer's flows and current violations use the row's.
- [ADDED] Strategy actions may move elements between the buses of a substation (bus splits and merges):
  spare buses and one union sparsity pattern for the run, unused spares masked per row; a moved branch's
  flows and current violations on its new buses; a moved voltage-regulating generator holds its new bus
  PV (its former bus PQ when nothing else regulates it), a group member leaves its group; the closed end
  of a half-open branch moves with its Kron-reduced admittance. All of it with ``outer_loops=True`` too.
- [CHANGED] An element end a strategy's contingency trips stays out whatever its actions do with it (the
  action's move of it is dropped, the rest applies), instead of the row being reported unsupported.
- [CHANGED] ``Action.apply_to`` leaves a half-open branch half-open when it moves the closed end
  (lightsim2grid's ``change_bus*`` alone reconnects the open one).
- [ADDED] ``strategy_warm_start`` with ``outer_loops=True``: a strategy resumes from its contingency's
  outer-loop state too (moved targets, frozen buses, regimes, taps), kept as the contingency finishes,
  as lightsim2grid's batch row resuming from a solve; ``get_row_status`` / ``get_row_iterations`` /
  ``get_outer_results`` / the physical checks have one entry per row (contingencies, then strategies).
- [ADDED] C++/CUDA unit tests (``tests/cpp``, ctest), opt-in with ``-DGPUSIM2GRID_BUILD_TESTS=ON``; a
  test needing a GPU is skipped without one.

[0.3.0] 2026-11-16
--------------------
- [ADDED] An sdist installable with ``pip install gpusim2grid[cuda13]``: the isolated build takes nvcc and
  cuDSS from PyPI, and the libraries find the ``nvidia`` wheels without ``LD_LIBRARY_PATH``.
- [ADDED] ``gpusim2grid[cuda12]`` builds with the CUDA 12 nvcc ``CUDACXX`` names (the build's CUDA major is
  picked from the environment by an in-tree backend: an extra never reaches the build).
- [ADDED] ``.github/workflows/publish.yml``: a published GitHub release ``vX.Y.Z`` uploads the sdist to PyPI
  (trusted publishing), once a GPU-less ``pip install`` of it builds; manual dispatch rehearses on TestPyPI.
- [ADDED] ``import gpusim2grid`` refuses CUDA libraries of another major than the build, a too old driver, and
  another lightsim2grid version than the bridge's.
- [ADDED] CI: one build leg and a GPU test subset pin lightsim2grid to 1.3.1, the minimum supported, so a
  change needing a newer one is caught.
- [FIXED] build: Eigen's SIMD settings are read off ``lightsim2grid_core`` and matched, not only ``-march=native``:
  a compiler defaulting to AVX2 (ubuntu-26.04 runner) built an extension PyPI's lightsim2grid refused to load.
- [FIXED] the first cuDSS call of a process (``warmup()`` or any session) left the calling thread in the "C"
  locale (NVRTC, cuDSS's JIT, never restores it): Python then read ASCII as its default file encoding.
- [FIXED] ``ContingencyAnalysisGPU.add_contingenc*``: a refused contingency (an id out of range) stayed in the
  list, so every later call raised too; the session's own list is left as it was.
- [FIXED] ``ScenarioSweepGPU.set_contingency_loads``: a mask disconnecting nothing clears it (per-bus
  ``set_injections`` stayed refused until the facade was rebuilt).
- [FIXED] ``BatchPowerFlow(outer_loops=True)``: the backward of a forward with no simulated row raised;
  it gives zero gradients, as without the outer loops.
- [FIXED] ``BatchPowerFlow(outer_loops=True)``: ``load_status`` reaches the loops' view of the row, as
  ``ScenarioSweepGPU.set_contingency_loads``.
- [IMPROVED] ``outer_loops=True``: the per-slot controls go up for the slots that changed, not whole twice
  a round; ``InjectionSweepGPU`` hands the rows' generator targets over as an array.
- [IMPROVED] The automatic reference slack checks that lightsim2grid's ordering constraints name no index
  it moves (pinned against lightsim2grid's own ledger for that reference).
- [IMPROVED] The GIL is released during the long calls (session construction, ``run``, ``compute_flows``,
  ``resolve``, adjoint solves, result downloads, ``warmup``), so other Python threads keep running.
- [FIXED] ``benchmarks/contingency_analysis.py`` stopped on a ``NameError`` on grids with more branches
  than ``nb_max_cont_cpu``, and ran some contingencies twice.
- [FIXED] ``InjectionSweepGPU``: a reused driver ignored, or misapplied, the ``set_gen_v`` columns of
  remote regulators added or changed since it was built.
- [IMPROVED] ``outer_loops=True``: a row can trip an HVDC line with a station regulating a voltage
  (it was ``BATCH_UNSUPPORTED``).
- [FIXED] ``outer_loops=True``: a tripped controller came back at a reactive limit when ``ReactiveLimits``
  froze a bus it shared with another controller of its group.
- [IMPROVED] (diagnostic) ``GPUSIM2GRID_OC_PROFILE=1`` also counts the rows never re-solved, the loops
  that act and the frozen buses resumed.
- [ADDED] A contingency or a row can trip an SVC with a standby automaton (lightsim2grid leaves it out): its
  release-check entries and the ``VoltageMonitoring`` loop skip it for that row.
- [IMPROVED] ``outer_loops=True``: the rows' adjoint terms (``pq_buses``, ``gen_p_fixed``, ...) are only computed
  with gradients; ``get_outer_results()`` leaves them empty otherwise.
- [IMPROVED] ``outer_loops=True``: a finished row's taps, sections and regimes are listed when read, not in the run.
- [IMPROVED] ``outer_loops=True``: a slot's load starts from a state prepared once per run, reusing its memory.
- [ADDED] A row can trip a controller of a voltage-control group (a remote or group generator, an SVC in voltage
  mode, a converter station): it leaves its group by value, still one analysis, the regulated bus free when none is left.
- [ADDED] A contingency or a row can trip an HVDC line in AC emulation: no flow, no slope, still one analysis.
- [FIXED] ``ScenarioSweepGPU`` with ``redistribute_slack``: a tripped storage unit kept its share of the row's slack.
- [ADDED] ``ContingencyAnalysisGPU`` / ``ScenarioSweepGPU``: a contingency can trip static generators, storage units,
  SVCs and HVDC lines (lightsim2grid #292): ``add_contingency(sgen_ids=, ...)``, ``set_contingency_sgens`` etc.
- [BREAKING] What the batch cannot do (a remote generator, an SVC in voltage mode, ...) is reported per row
  (``get_batch_unsupported_reasons``, a ``BATCH_UNSUPPORTED`` violation) instead of raising for the whole batch.
- [BREAKING] ``ContingencyAnalysisGPU.get_contingencies()`` returns 10 lists, as lightsim2grid's ``my_contingencies()``.
- [FIXED] Tripping a generator next to a storage unit or HVDC station still regulating its bus turned that bus PQ.
- [FIXED] ``outer_loops=True``: a voltage-control group losing a controller to an island, the others held at a
  reactive limit, failed: the islanded controller is now gone and the regulated bus free.
- [FIXED] ``outer_loops=True`` with ``redistribute_slack``: after a contingency's lost-power share, ``DistributedSlack``
  shares the rest from the row's own targets (lightsim2grid 1.3.1, OpenLoadFlow): a unit put at a bound stays there.
- [FIXED] ``outer_loops=True`` with ``redistribute_slack`` and ``init_from_n_powerflow``: a row losing a unit the "n"
  solve moved shares that power, even when it loses nothing at the grid's targets.
- [FIXED] ``outer_loops=True``: ``DistributedSlack(run_without_participant=True)`` is honoured (lightsim2grid #269).
- [FIXED] Gradients with ``NROuterDist_*``: a unit parked at a check that did not re-solve keeps its ``gen_p`` gradient.
- [FIXED] ``BatchPowerFlow.compute_flows`` refuses a ``V`` of another row count when the outer loops moved taps,
  instead of silently using the grid's taps.
- [ADDED] Continuous scheduling: ``round_records`` (rows and times of each round), and ``t_sess_*_ms`` for what
  a contingency analysis does around the driver's solve.
- [IMPROVED] Continuous scheduling: no base-case load of a freed slot the tail overwrites or drops
  (``n_base_loads`` / ``n_base_loads_skipped``); results unchanged.
- [FIXED] ``outer_loops=True``: ``scaling_max_voltage_change`` / ``max_dVa`` / ``max_dVm`` were silently ignored at
  construction (the grid's own setting was always used).
- [IMPROVED] MaxVoltageChange step scaling: one block per slot instead of one thread per entry (about 4x cheaper),
  now timed in ``t_update_V``; results unchanged.
- [IMPROVED] ``outer_loops=True``: the host model's row load is ~3x and its decisions ~2x cheaper (shared
  ReactiveLimits plans, base limit sums, flat per-row state); results unchanged.
- [ADDED] Continuous scheduling timings: ``t_main_phase_wall_ms`` (every slot in use) and ``t_tail_phase_wall_ms``,
  with the live rows solved in each (``n_main_row_rounds``, ``n_tail_row_rounds``).
- [ADDED] (diagnostic) ``GPUSIM2GRID_OC_PROFILE=1``: wall time per part of the outer model, on stderr.
- [BREAKING] ``SlackRedistributionData.gen_target_p_mw`` is now ``gens_target_p_mw``. In C++, ``RowResult``
  has a dense ``gens_target_p_mw`` and ``gen_target_p_mw(gen_id)``; ``gen_ids`` is gone.
- [ADDED] ``gpusim2grid_core``: the solvers as a C++ shared library without Python (headers, CMake package
  ``gpusim2grid::core``, ``gpusim2grid.get_cmake_dir()``), for C++ / JNI callers; the Python module links it.
- [ADDED] ``benchmark_cudss_batch_raw(n_active=)``: with ``linear_solver='own_lu'``, only the first
  ``n_active`` matrices are factorized and solved, the others left untouched.
- [ADDED] (experimental) ``linear_solver='own_lu'`` on the three batch facades and ``BatchPowerFlow``: a batched LU
  of our own instead of cuDSS, ordered on the host (lightsim2grid's constrained order by default): no pivot to perturb.
  The LU itself is Ralf Cortes Do Nascimento's contribution (ported from gpu-security-analysis).
- [IMPROVED] ``linear_solver='own_lu'``: the continuous tail runs on the forward solver's first slots (active
  prefix) instead of a second solver cycled over groups of ``tail_group_size``.
- [ADDED] lightsim2grid's constrained order now starts from the grid's own fill-reducing order (AMD or
  ``MinFill``); ``reordering_alg='ls2g_constrained_minfill'`` forces minimum fill.
- [ADDED] (diagnostic) ``GPUSIM2GRID_DUMP_J=<dir>``: every batch solver dumps its Jacobians and orders.
- [ADDED] ``BatchTimings.cudss_user_perm_honoured`` and, with ``GPUSIM2GRID_COUNT_PIVOTS=1``,
  ``cudss_npivots_*``: whether cuDSS used the order given, and how many pivots it perturbed.
- [ADDED] (prototype) ``reordering_alg='ls2g_constrained'`` / ``'ls2g_constrained_nd'``: lightsim2grid's
  constrained order given to cuDSS. Removes its perturbed pivots but slows it 6-9x; see the dev note.
- [ADDED] ``BatchPowerFlow``: ``redistribute_slack`` with ``outer_loops=True``, the lost power measured in
  torch and shared by the rows' ``DistributedSlack`` loop; a unit that share puts at a bound gets no ``gen_p``
  gradient.
- [ADDED] ``BatchPowerFlow.from_lsgrid(outer_loops=True)``: gradients through the outer loops, as lightsim2grid
  -- each row's last inner solve, its loops' decisions held; ``compute_flows`` at the taps they settled on in
  the forward that returned ``V`` (``taps_from=`` for a clone).
- [ADDED] ``outer_loops=True`` on a grid solved with lightsim2grid's ``NROuterDist_*``: the inner Newton shares
  the slack, ``DistributedSlack`` only parks the units it pushes past a limit, as lightsim2grid.
- [FIXED] ``reordering_alg`` changed after construction: the batch solvers get that algorithm's order (they kept
  the construction value's lightsim2grid order, with cuDSS and the own LU alike).
- [FIXED] ``benchmark_cudss_batch_raw(linear_solver=OwnLu)`` checks the CSR before ordering it (a malformed one
  crashed the interpreter).
- [FIXED] ``calibrate_tail_max_groups()`` with ``linear_solver='own_lu'``: measured on the prefix the tail runs
  (k groups = one call), no longer ``t_full / t_group``, which advised far too few groups.
- [IMPROVED] ``linear_solver='own_lu'`` with a ``reordering_alg`` / ``matching_alg`` it cannot take is refused at
  construction or by the setter making the combination, not at the first ``compute()``.
- [FIXED] ``linear_solver='own_lu'`` in an FP32 build clamps tiny pivots at ``1e-5 * max|J|`` (was ``1e-12``,
  below float resolution: garbage factors, refinement diverging).
- [FIXED] ``ContingencyAnalysisGPU`` / ``ScenarioSweepGPU``: ``violation_capacity`` and ``violation_tol`` set on
  the facade reach the session (they were plain attributes, silently ignored).
- [FIXED] ``violation_capacity`` below 1 is refused (``ValueError``): the operational check wrote its
  DIVERGENCE record into an empty buffer, an illegal address.
- [FIXED] ``benchmark_cudss_batch_raw`` refuses a CSR matrix whose ``indptr`` or column indices are out of range.
- [FIXED] Contingencies: a branch named twice, or in ``branch_ids`` and a side list, and a repeated shunt /
  load / generator id are tripped once, as lightsim2grid (the admittance was subtracted twice).
- [FIXED] ``ScenarioSweepGPU.set_contingency_shunts`` on a new row count no longer refuses the "no trip"
  topology it filled in itself; ``set_contingency_loads`` checks its columns and its rows at ``compute()``.
- [FIXED] ``redistribute_slack``: a ``ScenarioSweepGPU`` row tripping a resistive shunt shares its active power.
- [FIXED] ``outer_loops=True``: a row the loops gave up on counts no violation, is not truncated and is not
  ``converged()``, as its ``get_violations()`` entry says.
- [FIXED] ``outer_loops=True`` with ``redistribute_slack``: the lost-power share keeps the row's targets as the
  base, and a row gets its loss even when it is 0 at the grid's targets (lightsim2grid 1.3.1).
- [FIXED] A generator contingency no longer turns PQ a bus a storage unit or an HVDC station still regulates.
- [ADDED] ``benchmark_cudss_batch_raw(n_handles=, handle_launch=)`` and ``benchmarks/cudss_small_handles.py``:
  several independent cuDSS handles per round, timed against one large one.
- [ADDED] ``ContingencyAnalysisGPU.add_contingency(branch_ids_side1=, branch_ids_side2=)``: a branch opened at
  one end only keeps its charging from the other, as in lightsim2grid.
- [ADDED] ``ContingencyAnalysisGPU`` / ``ScenarioSweepGPU``: ``redistribute_slack`` with ``outer_loops=True``, the
  ``DistributedSlack`` loop sharing what a row loses before its first solve (lightsim2grid's).
- [FIXED] ``outer_loops=True``: the ``ReactiveLimits`` robust mode reads the realistic voltage band and
  the robust flag off the outer-loop driver when the loop leaves them at their defaults, as lightsim2grid.
  A remote controller whose own bus is unrealistic now stops regulating, at its target Q.
- [ADDED] Contingencies that trip shunts, loads and generators, as in lightsim2grid:
  ``ContingencyAnalysisGPU.add_contingency(branches, shunts, loads, gens)`` and the
  ``add_contingency_shunt`` / ``_load`` / ``_gen`` shortcuts, ``ScenarioSweepGPU.set_contingency_shunts`` /
  ``set_contingency_loads``, and ``BatchPowerFlow``'s ``shunt_status`` / ``load_status``. A regulating
  shunt cannot be tripped with ``outer_loops=True`` yet.
- [FIXED] ``outer_loops=True``: ``DistributedSlack`` gives no share to a unit on a bus the row strands
  (``handle_disconnected_grid``), as lightsim2grid does.
- [FIXED] ``get_violations()``: a row the outer loops gave up on (``OUTER_FAILED``, ``OUTER_UNSTABLE``,
  ``UNREALISTIC``) gets one ``GRID`` / ``DIVERGENCE`` entry, as lightsim2grid, not the violations of
  its last state.
- [IMPROVED] ``scheduling="continuous"``: once the queue is empty, the last rows are moved into the
  first slots and solved in groups of 15 (``tail_max_groups``, ``tail_group_size``). cuDSS 0.8.0's batch
  costs over 10x more from 16 matrices. ``calibrate_tail_max_groups()`` measures the setting.
- [ADDED] ``benchmark_cudss_batch_raw(n_groups=...)``: one cuDSS handle cycled over several groups of
  matrices, as the continuous tail does.
- [IMPROVED] ``outer_loops=True``: loading a row into its slot is much cheaper on the host; the slack
  participants, resumed targets and reactive-limits plan are built once per grid, not per row.
- [IMPROVED] ``compute_limit_violations``: the operational check runs one GPU block per row, not one thread,
  same records; it cost tens of ms per call on a real grid, every continuous round.
- [FIXED] ``ContingencyAnalysisSession.overwrite_rows`` / ``ScenarioSweepSession.set_v_init_from`` (was
  ``set_v_init_from_ptr``) take the source session, not raw pointers: a source row past its rows is refused.
- [FIXED] ``reactive_limits_outer_loop``: the second pass checks the limits given to the analysis
  (``set_limits``), not the grid's own.
- [FIXED] ``outer_loops=True`` accepts ``scheduling=BatchScheduling.Continuous``, not only the string.
- [FIXED] ``compute_flows`` / ``get_V_results`` on the three batch sessions: rows × branches (or buses) past
  ``INT32_MAX`` overflowed (large grids, many rows); sizes and offsets are now 64-bit.
- [FIXED] ``outer_loops=True``: parallel transformers of ``PhaseControl`` / ``TransformerVoltageControl`` lost
  Ybus patches (racing threads), leaving rows at their iteration budget or on wrong blocks.
- [ADDED] ``benchmark_cudss_batch_raw`` and ``solve_cudss_raw`` take ``ir_n_steps``.
- [ADDED] ``benchmarks/outer_loops.py`` (the outer loops against lightsim2grid's ``NROuter_KLU``),
  ``benchmarks/continuous_batching.py`` and ``continuous_batching_potential.py`` (chunked vs continuous).
- [ADDED] ``BatchTimings.t_sched_res_sync`` / ``t_oc_gather`` / ``t_oc_decide`` / ``t_oc_controls`` /
  ``t_oc_load``: where a continuous run's per-round host time goes, outer loops included.
- [ADDED] ``BatchTimings.t_driver_solve_wall_ms`` / ``t_session_run_wall_ms``: whole-call wall clocks,
  to find the time no phase covers.
- [ADDED] ``ir_n_steps`` on the four facades and ``BatchPowerFlow``: cuDSS iterative refinement after each
  solve (default 0). Use 1 with ``outer_loops=True``: hard inner solves then converge as lightsim2grid's do.
- [BREAKING] The release records (types 9-12 and 15) are now ``ViolationCategory.CONTROL``, as in
  lightsim2grid; the types 16-20 of its outer-loop detection are mirrored.
- [ADDED] ``outer_loops=True`` on the three batch facades: lightsim2grid's ``NROuter_*`` outer loops
  on every row (continuous scheduling), the inner solves on the GPU; ``get_outer_results()``.
- [ADDED] ``outer_loops=True`` with ``handle_disconnected_grid``, ``set_contingency_gens`` and
  ``set_gen_v``: each row's loops see its masks, units off and set-points as lightsim2grid's do.
- [IMPROVED] ``reactive_limits_outer_loop`` is deprecated: use ``outer_loops=True``, which runs
  lightsim2grid's ``ReactiveLimits`` loop (PV <-> PQ) on every row until it is stable.
- [FIXED] ``outer_loops=True`` with ``init_from_n_powerflow=False``: rows start at every voltage
  source's set-point (stations, SVCs, storage units too) and use the grid's Newton step scaling.
- [ADDED] ``outer_loops=True`` with lightsim2grid's ``ShuntVoltageControl`` loop: the regulating shunts'
  susceptance solved then dispatched to sections. All seven OpenLoadFlow loops now run on the GPU.
- [ADDED] ``outer_loops=True`` with lightsim2grid's ``TransformerVoltageControl`` loop: ratios solved then
  rounded, the low voltage generators frozen meanwhile, as lightsim2grid.
- [ADDED] ``outer_loops=True`` with lightsim2grid's ``PhaseControl`` loop: phase shifters solved for
  their shift then rounded, current limiters moved tap by tap; flows and current limits at the moved taps.
- [FIXED] ``outer_loops=True``: ``DistributedSlack`` does not run when only the slack bus' own units
  participate, as lightsim2grid now (they would only take what that bus already absorbs).
- [ADDED] ``scheduling="continuous"`` on the three batch facades: a row leaves its slot once
  converged, diverged or out of its ``nb_iter`` budget, and the slot is refilled at once.
- [ADDED] ``get_row_status()`` / ``get_row_iterations()`` (``RowStatus``) and ``tol``: each row's
  outcome, in both schedules.
- [ADDED] ``BatchPowerFlow.from_lsgrid(scheduling="continuous", batch_size=...)``: gradients through a
  continuous run. The backward rebuilds each row's Jacobian from its converged voltages, so
  nothing is kept from the forward.
- [ADDED] Report an hvdc line frozen at its AC-emulation limit whose droop asks for less
  (``HVDC_AC_EMULATION_RELEASE``, lightsim2grid's ``set_hvdc_ac_emulation_frozen``).
- [FIXED] ``redistribute_slack``: a unit capped well beyond its active limit stays there until the
  shift has used that up (lightsim2grid's ``set_gen_can_participate_slack_overshoot``).
- [ADDED] Report a generator holding a remote bus from an unrealistic own-bus voltage
  (``LOW_VOLTAGE_REMOTE_CONTROL`` / ``HIGH_VOLTAGE_REMOTE_CONTROL``), lightsim2grid's
  ``set_remote_voltage_control_vm_range``, as OpenLoadFlow's robust remote voltage control does.
- [ADDED] ``ContingencyAnalysisGPU(reactive_limits_outer_loop=True)`` (opt-in): one pass of
  OpenLoadFlow's ``ReactiveLimits`` loop, re-solving the contingencies a PV bus or a local release
  flags (``get_outer_loop_status``).
- [IMPROVED] ``reactive_limits_outer_loop`` also re-solves a voltage-control group of generators
  whose every controller saturated: each held at its limit, the regulated bus floating.
- [IMPROVED] ``reactive_limits_outer_loop`` also releases a frozen remote regulator, through
  lightsim2grid's held controllers (``set_hold_frozen_regulators``, lightsim2grid PR #220),
  pinned on every other row.
- [IMPROVED] ``reactive_limits_outer_loop``: the second pass starts each contingency from its
  first-pass voltages (``outer_loop_warm_start``, default on); ``outer_loop_nb_iter`` sets its
  Newton iterations.
- [IMPROVED] the build takes the SuiteSparse headers lightsim2grid >= 1.3.0 ships with its core:
  a lightsim2grid installed from a wheel no longer needs a SuiteSparse checkout on the side.
- [BREAKING] lightsim2grid >= 1.3.1 is required, to build and to run: its PyPI wheel is all the bridge needs.
- [ADDED] ``cuda12`` / ``cuda13`` extras: the CUDA libraries and cuDSS (``nvidia-cudss-cu12/-cu13>=1.0.0.35,<2``) from PyPI.
- [IMPROVED] the build finds a CUDA toolkit installed from PyPI (``cuda-toolkit``, libraries named by soname only)
  and cuDSS 1.x's wheel (``libcudss.so.1``); the docs install everything from PyPI.
- [IMPROVED] CI: CUDA 12.4 and the latest CUDA 13 on PyPI, each with cuDSS 1.x, and lightsim2grid, all from PyPI.

[0.2.0] 2026-xx-yy
--------------------
- [ADDED] ``redistribute_slack`` (``ContingencyAnalysisGPU``, ``ScenarioSweepGPU``,
  ``BatchPowerFlow``): OpenLoadFlow's bounded slack pre-pass on what a row loses, as
  lightsim2grid PR #216.
- [IMPROVED] ``redistribute_slack`` on a scenario sweep (``ScenarioSweepGPU``, ``BatchPowerFlow``)
  also shares the active imbalance each row's injections create against the grid's set-points
  (``SlackRedistributionData.base_p_mw``), as lightsim2grid.
- [ADDED] ``reference_slack="auto"`` (default with ``handle_disconnected_grid``): the angle
  reference is the slack unit stranded by the fewest rows.
- [ADDED] ``LimitViolationType`` mirrors lightsim2grid 1.2's ``LOW_VOLTAGE_REMOTE_CONTROL`` /
  ``HIGH_VOLTAGE_REMOTE_CONTROL`` / ``HVDC_AC_EMULATION_RELEASE`` (category PHYSICAL, not computed yet).
- [ADDED] PQ -> PV release check (``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q``) and
  ``physical_violation_tol_vm_pu``.
- [ADDED] ``violation_rel_tol`` (default ``1e-9``): a value on its operational limit up to
  rounding is not reported, as lightsim2grid.
- [IMPROVED] ``handle_disconnected_grid`` solves a row stranding every controller of a
  voltage-control group of any size, and skips one stranding a regulated bus whose controllers
  stay live.
- [FIXED] ``ScenarioSweepGPU``: a hot re-run lost the row masks, so ``BatchPowerFlow`` gave a
  stranded voltage-control group a ``gen_v`` gradient.
- [ADDED] Physical check of the idle SVCs under a standby automaton (lightsim2grid's
  ``SvcStandbyCheck``, ``LSGrid.set_svc_standby``): a regulated bus outside the automaton's
  thresholds is reported as ``LOW_VOLTAGE_SVC_STANDBY`` / ``HIGH_VOLTAGE_SVC_STANDBY`` on the new
  ``ViolationElementType.SVC``. Routed through the PQ -> PV release plan.
- [ADDED] The PQ -> PV release check for the SVCs lightsim2grid flags as frozen at a reactive limit
  (``LSGrid.set_svc_can_be_pv``): reported as ``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q``
  on the SVC. The records of the release plan carry their element type (``el_type``), the standby
  entries are marked by ``GenPvReleasePlanData.standby``.
- [ADDED] ``redistribute_slack`` shares on the units lightsim2grid flags "can participate in the
  slack" (``LSGrid.set_gen_can_participate_slack``) too, in the pre-pass only
  (``SlackRedistributionData.in_slack``).
- [FIXED] The PQ -> PV release plan (lightsim2grid's ``build_gen_pv_release_plan``) pins a flagged
  generator at the nearer of its reactive limits, not only within ``physical_violation_tol_mva`` of
  one: the facades no longer rebuild it when that tolerance changes.
- [ADDED] The PQ -> PV release of the VSC converter stations lightsim2grid flags as frozen at a
  reactive limit (``LSGrid.set_hvdc_can_be_pv``): reported on the HVDC line with ``side`` the
  station's end (``GenPvReleasePlanData.side``, ``GenPvReleaseViolationsResult.side``).
