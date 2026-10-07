Change Log
===========

[0.2.0] 2026-xx-yy
--------------------
- [ADDED] ``scheduling="continuous"`` on the three batch facades: a row leaves its slot once
  converged, diverged or out of its ``nb_iter`` budget, and the slot is refilled at once.
- [ADDED] ``get_row_status()`` / ``get_row_iterations()`` (``RowStatus``) and ``tol``: each row's
  outcome, in both schedules.
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
- [ADDED] ``redistribute_slack`` (``ContingencyAnalysisGPU``, ``ScenarioSweepGPU``,
  ``BatchPowerFlow``): OpenLoadFlow's bounded slack pre-pass on what a row loses, as
  lightsim2grid PR #216.
- [IMPROVED] ``redistribute_slack`` on a scenario sweep (``ScenarioSweepGPU``, ``BatchPowerFlow``)
  also shares the active imbalance each row's injections create against the grid's set-points
  (``SlackRedistributionData.base_p_mw``), as lightsim2grid.
- [ADDED] ``reference_slack="auto"`` (default with ``handle_disconnected_grid``): the angle
  reference is the slack unit stranded by the fewest rows.
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
