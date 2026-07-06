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
a ``DIVERGED`` entry instead of needing a separate convergence check.

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
