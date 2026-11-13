Change Log
===========

[0.3.0] 2026-xx-yy
--------------------
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
