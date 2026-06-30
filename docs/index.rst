gpusim2grid
===========

GPU-accelerated AC power flow for the `Grid2Op <https://github.com/Grid2Op>`_
ecosystem. ``gpusim2grid`` runs AC power flow on the GPU and scales it to large
batches — thousands of scenarios solved in parallel on a single device — built
around batched sparse direct factorization (cuDSS) and using
`lightsim2grid <https://github.com/Grid2op/lightsim2grid>`_ for CPU
preprocessing and as a reference oracle.

.. warning::

   Research code, early release. The public API is still moving and may change
   between versions. This package requires a CUDA-capable NVIDIA GPU; it cannot
   run on CPU-only machines or on non-NVIDIA accelerators.

.. warning::

   This is a raw Newton-Raphson solver with **no outer loop**: it does not
   enforce reactive-power limits (no PV→PQ switching) or adjust tap/phase-shifter
   setpoints. The in-Jacobian controls that lightsim2grid models — distributed
   slack, HVDC angle-droop, SVC, and remote generator voltage control — *are*
   solved when seeded from a lightsim2grid grid (the default facade path), and
   then match lightsim2grid. See the project ``DISCLAIMER.md`` before using it for
   power-system analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
