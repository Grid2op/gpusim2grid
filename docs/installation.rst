Installation
============

There is no PyPI release yet — install from source.

Requirements
------------

* A CUDA-capable NVIDIA GPU.
* The `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_.
* `cuDSS <https://developer.nvidia.com/cudss>`_ >= 0.8.0 (sparse direct solve).
* cuSPARSE (ships with the CUDA Toolkit).
* Python >= 3.9, a C++17 compiler, CMake >= 3.15.
* `lightsim2grid <https://github.com/Grid2op/lightsim2grid>`_ — CPU
  preprocessing and reference oracle (hard dependency).

NVIDIA libraries (cuDSS, cuSPARSE) are **not** redistributed with this package;
install the CUDA Toolkit and cuDSS yourself before building.

From source
-----------

.. code-block:: bash

   git clone https://github.com/Grid2Op/gpusim2grid.git
   cd gpusim2grid
   source env_compile.sh   # exports CUDAToolkit_ROOT, cudss_ROOT, LD_LIBRARY_PATH — edit for your machine
   pip install .

The lightsim2grid C++ bridge
-----------------------------

Extracting a ``lightsim2grid`` grid into the GPU solver has two paths:

* a **zero-copy C++ bridge** (``lightsim2grid_core``), used when
  ``gpusim2grid._gpusim2grid`` exposes ``_make_ca_session_from_lsgrid`` and
  friends;
* a **Python fallback** (``_ls2g_utils.py``), used otherwise.

The C++ bridge is built automatically: at configure time, CMake asks the
Python interpreter it is building against for
``lightsim2grid.get_cmake_dir()`` and locates ``lightsim2grid_core`` from
there. Since ``lightsim2grid`` is a build-system requirement (see
``pyproject.toml``), this works out of the box with a plain
``pip install .`` / ``uv pip install .`` — no ``CMAKE_ARGS`` needed.

If you need the bridge to build against a *different* ``lightsim2grid``
install than the one importable by that interpreter, override the
detection explicitly:

.. code-block:: bash

   CMAKE_ARGS="-DLIGHTSIM2GRID_CMAKE_DIR=$(python -c \
     'import lightsim2grid; print(lightsim2grid.get_cmake_dir())')" \
     pip install .

When ``lightsim2grid_core`` cannot be found (bridge disabled), the build
falls back to the vendored Eigen copy (``src/eigen/``) and the Python
fallback path is used at runtime instead — everything still works, just
without the zero-copy fast path.

Selecting floating-point precision
----------------------------------

Precision is chosen at **build time**. The default is FP64 (double).

.. code-block:: bash

   pip install .                      # FP64 (double), default
   CUDA_REAL_FLOAT=1 pip install .    # FP32 (float)
   CUDA_REAL_DOUBLE=1 pip install .   # FP64 (double), explicit

Equivalently, pass ``-DUSE_FLOAT_PRECISION=ON`` to CMake. Query the compiled
precision at runtime with :func:`gpusim2grid.compilation_options.is_fp32`.

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html        # output in docs/_build/html

The docs build from the Python docstrings and, when the compiled extension is
importable, the pybind11 binding docstrings. On a CPU-only machine the extension
is mocked, so only the Python API is rendered.
