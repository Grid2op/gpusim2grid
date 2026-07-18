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
   pip install --no-build-isolation .

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
``pip install --no-build-isolation .`` / ``uv pip install --no-build-isolation .``
— no ``CMAKE_ARGS`` needed. ``--no-build-isolation`` is what makes the build
see the ``lightsim2grid`` already installed in your current environment,
instead of pip resolving a fresh, isolated one that may not match it.

If you need the bridge to build against a *different* ``lightsim2grid``
install than the one importable by that interpreter, override the
detection explicitly:

.. code-block:: bash

   CMAKE_ARGS="-DLIGHTSIM2GRID_CMAKE_DIR=$(python -c \
     'import lightsim2grid; print(lightsim2grid.get_cmake_dir())')" \
     pip install --no-build-isolation .

When ``lightsim2grid_core`` cannot be found (bridge disabled), the build
falls back to the vendored Eigen copy (``src/eigen/``) and the Python
fallback path is used at runtime instead — the Python fallback is exercised
against the bridge path for parity in ``tests/python/test_new_api.py``
(``use_bridge=False`` vs ``use_bridge=True``), but this project has no CI
yet, so that comparison is only ever run by hand, not automatically on
every change.

SuiteSparse headers for the bridge
-----------------------------------

When the bridge is enabled, ``LSGrid.hpp`` transitively pulls in KLU headers
(``cs.h``, ``klu.h``, ``amd.h``, ``colamd.h``, ``btf.h``,
``SuiteSparse_config.h``) that the installed ``lightsim2grid_core`` package
does not ship. CMake looks for them in order:

1. A system SuiteSparse dev package (e.g. Debian/Ubuntu's
   ``libsuitesparse-dev``), auto-detected via ``find_path``. Set
   ``SUITESPARSE_INCLUDE_DIR`` (env var or ``-D``) if it's installed
   somewhere non-standard.
2. A SuiteSparse *source checkout*'s per-module ``Include/`` layout, via
   ``-DGPUSIM2GRID_SUITESPARSE_DIR=/path/to/SuiteSparse``.

If neither is found, configuration prints a warning and the bridge may fail
to compile with a ``cs.h``/``klu.h`` not found error — install
``libsuitesparse-dev`` (or point at a checkout) to fix it.

It's best to build against the **same Eigen and SuiteSparse** (same
versions, same compile flags) that your ``lightsim2grid`` build used —
mismatched ABI/struct layouts between the two can fail silently rather than
at compile time. The only configuration currently exercised and known to
work is:

* ``lightsim2grid`` installed from source (``pip install --no-build-isolation .``),
* ``gpusim2grid`` installed from source the same way.

Other configurations (prebuilt ``lightsim2grid`` wheel, mixed Eigen/SuiteSparse
versions, ...) may work but are not tested.

Selecting floating-point precision
----------------------------------

Precision is chosen at **build time**. The default is FP64 (double).

.. code-block:: bash

   pip install --no-build-isolation .                      # FP64 (double), default
   CUDA_REAL_FLOAT=1 pip install --no-build-isolation .    # FP32 (float)
   CUDA_REAL_DOUBLE=1 pip install --no-build-isolation .   # FP64 (double), explicit

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
