Examples
========

Runnable scripts live in the ``examples/`` directory of the repository. Each
requires a CUDA-capable GPU and an installed build of the package, plus
``lightsim2grid`` / ``pandapower`` for the grid data (``pip install -e ".[test]"``).
Run them from the repository root so the shared ``examples/_common.py`` helper is
importable.

Single AC power flow (IEEE 14-bus)
----------------------------------

End-to-end single AC power flow, compared against the lightsim2grid KLU
reference.

.. literalinclude:: ../examples/ieee14_basic.py
   :language: python

Batched N-1 contingency screen
------------------------------

Screens one contingency per branch on the GPU, reusing a single base-case
factorization, and reports convergence statistics and timing.

.. literalinclude:: ../examples/case6515rte_screen.py
   :language: python

Largest-component solve of a split grid
---------------------------------------

Screens every N-1 and, for the contingency that disconnects an island, solves
the largest connected component (reporting the islanded buses as ``NaN``) instead
of skipping it — via the ``ContingencyAnalysisGPU`` facade and
``handle_disconnected_grid``.

.. literalinclude:: ../examples/handle_disconnected.py
   :language: python

Augmented solve: distributed slack
----------------------------------

Adds a second weighted slack to the IEEE 14-bus case so the active-power slack
is genuinely shared, then solves the **same augmented Newton-Raphson system
lightsim2grid poses** on the GPU (via the C++ bridge) and matches the CPU
reference. The same path also covers HVDC angle-droop, SVC, and remote
generator voltage control.

.. literalinclude:: ../examples/distributed_slack.py
   :language: python

Differentiable power flow (alpha)
---------------------------------

Differentiates a scalar loss with respect to the bus injections through a single
power flow, using the adjoint method via the PyTorch integration.

.. literalinclude:: ../examples/differentiable_pf.py
   :language: python
