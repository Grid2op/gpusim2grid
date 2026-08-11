API reference
=============

The high-level Python wrappers are the recommended entry points. The low-level
CUDA bindings (``gpusim2grid._gpusim2grid``) are documented at the bottom and are
only fully rendered when the docs are built on a machine where the compiled
extension is importable.

GPU facades
-----------

These are the top-level entry points exported from the ``gpusim2grid`` package,
and each is the *single* entry point for its workload: their ``grid`` argument
accepts either a solved lightsim2grid grid (seeding the GPU batch from the CPU
base case, zero-copy via the C++ bridge where available) or an explicit
``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)`` array tuple for
callers without a lightsim2grid grid.

.. autoclass:: gpusim2grid.ContingencyAnalysisGPU
   :members:

.. autoclass:: gpusim2grid.InjectionSweepGPU
   :members:

.. autoclass:: gpusim2grid.ScenarioSweepGPU
   :members:

.. autoclass:: gpusim2grid.AcPfGPU
   :members:

.. autofunction:: gpusim2grid.optimize_reference_slack

Contingency analysis
--------------------

.. automodule:: gpusim2grid.contingency_analysis
   :members:
   :undoc-members:

Limit violations
~~~~~~~~~~~~~~~~~

:class:`~gpusim2grid.ContingencyAnalysisGPU` supports an opt-in
``compute_limit_violations`` check (in both grid and explicit-array
construction modes), mirroring lightsim2grid's
``ContingencyAnalysis.compute_limit_violations`` flag: every contingency is
checked against per-bus voltage limits (kV) and per-side branch thermal
current limits (kA), configured on the grid via lightsim2grid's
``set_bus_voltage_limits`` / ``set_line_current_limit_side1/2`` /
``set_trafo_current_limit_side1/2``.

Unlike a host-side pass over the fully materialized batch results, the check
runs **fused into each chunk of the GPU solve** — one thread per contingency,
reading the chunk-local voltages that are already resident on device — and
writes only a small, bounded ``(n_contingencies, violation_capacity)`` record
buffer. The full dense ``V_results`` / ``or_amps`` / ``ex_amps`` arrays are
never required just to compute violations, which keeps both device memory use
and the eventual host transfer independent of grid/batch size — the point of
running this on GPU in the first place. A contingency that fails to converge
(``residual > violation_tol``) is folded into the same per-contingency list as
a ``GRID``/``DIVERGENCE`` entry instead of needing a separate convergence
check; one the pre-check dropped before it was ever solved gets a ``GRID``/
``NOT_SIMULATED`` entry instead.

Per contingency, branch current is checked **before** bus voltage (thermal
violations are generally a first-order operational concern, voltage a
second-order one), so :meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violations`
reports ``CURRENT`` entries ahead of ``LOW_VOLTAGE`` / ``HIGH_VOLTAGE`` ones
for the same contingency. Since the per-violation *records* are capped at
``violation_capacity``,
:meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violation_counts` additionally
reports the **true, uncapped** count of each of the three types
(``low_voltage`` / ``high_voltage`` / ``current``) per contingency — these
totals keep counting past the cap, so they stay accurate even when
:meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violations_truncated` is
``True`` for a contingency.

See :meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violations`,
:meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violation_counts`,
:meth:`~gpusim2grid.ContingencyAnalysisGPU.converged`, and the pre-contingency
("n" case, computed on the CPU since it is a single voltage vector, not a
batch) :meth:`~gpusim2grid.ContingencyAnalysisGPU.get_violations_n` /
:meth:`~gpusim2grid.ContingencyAnalysisGPU.converged_n`. Example:
:doc:`examples` ("N-1 screen with limit violations").

.. autoclass:: gpusim2grid.contingency_analysis.ViolationElementType
   :members:

.. autoclass:: gpusim2grid.contingency_analysis.LimitViolationType
   :members:

.. autoclass:: gpusim2grid.contingency_analysis.LimitViolation
   :members:

Injection sweep
---------------

.. automodule:: gpusim2grid.injection_sweep
   :members:
   :undoc-members:

Scenario sweep
--------------

Row-aligned combination of contingency analysis and injection sweep: row
``i``'s (P, Q) profile (:meth:`~gpusim2grid.ScenarioSweepGPU.set_injections` /
:meth:`~gpusim2grid.ScenarioSweepGPU.set_injections_from_elements`) is solved
together with row ``i``'s own set of tripped branches
(:meth:`~gpusim2grid.ScenarioSweepGPU.set_topology`), independently of every
other row. Mirrors lightsim2grid's own ``ScenarioSweep``. Deliberately a
separate class from :class:`~gpusim2grid.InjectionSweepGPU` /
:class:`~gpusim2grid.ContingencyAnalysisGPU` rather than an extension of
either, since the usage pattern (one topology + injection pair per row)
differs from both of theirs (a shared base case with a distinct scenario
set). ``set_topology`` is optional — a row never covered defaults to "no
branches tripped", so :class:`~gpusim2grid.ScenarioSweepGPU` also works as a
plain injection sweep.

Quick start:

.. code-block:: python

    import numpy as np
    from gpusim2grid import ScenarioSweepGPU

    # grid is a solved lightsim2grid grid (grid.ac_pf(...) already called).
    sweep = ScenarioSweepGPU(grid, nb_iter=10)

    # Per-element injections, mirroring lightsim2grid's own TimeSeries API:
    # one row per scenario, one column per load / generator.
    sweep.set_injections_from_elements(load_p, load_q, gen_p)

    # One branch-id list per scenario (lines-then-trafos), row-aligned with
    # the injections above. An empty list means "nothing tripped this row".
    sweep.set_topology([[], [3], [], [3, 40]])

    V_batch = sweep.compute(batch_size=512)   # DLPack (n_scenarios, n_bus)
    residuals = sweep.last_residuals()
    disconnected = sweep.get_disconnected()   # which rows were skipped (NaN)

``handle_disconnected_grid`` and ``compute_limit_violations`` are supported
identically to :class:`~gpusim2grid.ContingencyAnalysisGPU` — see the
"Limit violations" section above for the full semantics
(:class:`~gpusim2grid.contingency_analysis.ViolationElementType` /
:class:`~gpusim2grid.contingency_analysis.LimitViolationType` /
:class:`~gpusim2grid.contingency_analysis.LimitViolation` are reused as-is,
not redefined here) and :doc:`examples`
("Largest-component solve of a split grid" / "N-1 screen with limit
violations") for the equivalent ``ContingencyAnalysisGPU`` walkthroughs — the
only difference on :class:`~gpusim2grid.ScenarioSweepGPU` is that each row
also carries its own injection.

.. automodule:: gpusim2grid.scenario_sweep
   :members:
   :undoc-members:

Single AC power flow
--------------------

.. automodule:: gpusim2grid.acpf_nr
   :members:
   :undoc-members:

Differentiable power flow (alpha)
---------------------------------

.. automodule:: gpusim2grid.differentiable
   :members:
   :undoc-members:

Utilities
---------

.. automodule:: gpusim2grid.compilation_options
   :members:

.. automodule:: gpusim2grid.utils
   :members:

Low-level CUDA extension
------------------------

.. automodule:: gpusim2grid._gpusim2grid
   :members:
   :undoc-members:
