Change Log
===========

[0.3.0] 2026-xx-yy
--------------------
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
  -- each row's last inner solve, its loops' decisions held; ``compute_flows`` at the taps they settled on.
- [ADDED] ``outer_loops=True`` on a grid solved with lightsim2grid's ``NROuterDist_*``: the inner Newton shares
  the slack, ``DistributedSlack`` only parks the units it pushes past a limit, as lightsim2grid.
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
