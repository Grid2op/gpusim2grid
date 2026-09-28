Change Log
===========

[0.2.0] 2026-xx-yy
--------------------
- [ADDED] ``redistribute_slack`` (``ContingencyAnalysisGPU``, ``ScenarioSweepGPU``,
  ``BatchPowerFlow``): OpenLoadFlow's bounded slack pre-pass on what a row loses, as
  lightsim2grid PR #216.
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
