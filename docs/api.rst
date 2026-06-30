API reference
=============

The high-level Python wrappers are the recommended entry points. The low-level
CUDA bindings (``gpusim2grid._gpusim2grid``) are documented at the bottom and are
only fully rendered when the docs are built on a machine where the compiled
extension is importable.

GPU facades (lightsim2grid-driven)
----------------------------------

These are the top-level entry points exported from the ``gpusim2grid`` package.
They are seeded directly from a solved lightsim2grid grid and run the batched
solve on the GPU.

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
