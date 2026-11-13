Installation
============

There is no PyPI release of gpusim2grid yet — install from source. Everything
it builds against comes from PyPI wherever PyPI has it.

Requirements
------------

* A CUDA-capable NVIDIA GPU and its driver.
* Python >= 3.9 and a C++17 compiler.
* From PyPI:

  * `lightsim2grid <https://github.com/Grid2op/lightsim2grid>`_ >= 1.3.1 — CPU
    preprocessing and reference oracle (hard dependency). Its wheel ships
    ``lightsim2grid_core``, its headers and the KLU headers the bridge compiles
    against: no lightsim2grid or SuiteSparse checkout is needed.
  * `cuDSS <https://developer.nvidia.com/cudss>`_ 1.x (sparse direct solve):
    ``nvidia-cudss-cu12`` or ``nvidia-cudss-cu13``, ``>=1.0.0.35,<2``.
  * the CUDA libraries (cudart, cuBLAS, cuSOLVER, cuSPARSE), and for CUDA 13 the
    compiler too: ``cuda-toolkit``. ``pip install ".[cuda12]"`` / ``".[cuda13]"``
    lists the runtime ones.
  * ``scikit-build-core``, ``pybind11`` and ``cmake``.

* PyPI has no CUDA 12 compiler (``nvidia-cuda-nvcc-cu12`` ships ``ptxas`` only):
  with CUDA 12, ``nvcc`` (12.4 or later) comes from the
  `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ installer.

CI builds both: CUDA 12.4 with ``nvidia-cudss-cu12``, and the latest CUDA 13 on
PyPI with ``nvidia-cudss-cu13`` (``.github/workflows/build.yml``).

From source, CUDA 13
--------------------

.. code-block:: bash

   git clone https://github.com/Grid2Op/gpusim2grid.git
   cd gpusim2grid
   pip install "lightsim2grid>=1.3.1" "nvidia-cudss-cu13>=1.0.0.35,<2" \
       "cuda-toolkit[nvcc,crt,cudart,cccl,cublas,cusolver,cusparse,nvjitlink,nvvm]>=13,<14" \
       scikit-build-core pybind11 cmake
   CUDA_ROOT=$(python -c "import nvidia.cu13 as m; print(list(m.__path__)[0])")
   CUDACXX=$CUDA_ROOT/bin/nvcc CUDAToolkit_ROOT=$CUDA_ROOT cudss_DIR=$PWD/ci/cmake/cudss \
       pip install --no-build-isolation .
   export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH

The CUDA 13 wheels put the whole toolkit, and cuDSS, under one prefix
(``<site-packages>/nvidia/cu13``). Its libraries carry their soname only
(``libcudart.so.13``); ``src/_cpp/CMakeLists.txt`` points CMake's
``FindCUDAToolkit`` at them. They are not on the loader's path: set
``LD_LIBRARY_PATH`` as above before importing gpusim2grid.

From source, CUDA 12
--------------------

.. code-block:: bash

   pip install "lightsim2grid>=1.3.1" "nvidia-cudss-cu12>=1.0.0.35,<2" \
       scikit-build-core pybind11 cmake
   # nvcc from the CUDA 12.4+ toolkit on PATH (or CUDACXX=/path/to/nvcc)
   cudss_DIR=$PWD/ci/cmake/cudss pip install --no-build-isolation .
   export LD_LIBRARY_PATH=$(python -c "import nvidia.cu12 as m; print(list(m.__path__)[0])")/lib:$LD_LIBRARY_PATH

``ci/cmake/cudss/`` is a small CMake package for the cuDSS wheel, which ships
none (``find_package(cudss)`` finds NVIDIA's tarball or apt packages without
it: leave ``cudss_DIR`` unset and set ``cudss_ROOT`` for those). It takes the
wheel matching nvcc's CUDA major in the building Python, or
``CUDSS_WHEEL_ROOT`` (``<site-packages>/nvidia/cuXX``).

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
(``use_bridge=False`` vs ``use_bridge=True``), a GPU test: CI's GPU-less
build legs skip it, the opt-in Modal job runs it.

SuiteSparse headers for the bridge
-----------------------------------

When the bridge is enabled, ``LSGrid.hpp`` transitively pulls in KLU headers
(``cs.h``, ``klu.h``, ``amd.h``, ``colamd.h``, ``btf.h``,
``SuiteSparse_config.h``). CMake looks for them in order:

1. ``SUITESPARSE_INCLUDE_DIR`` (env var or ``-D``): a directory holding
   ``klu.h`` and the others directly.
2. The ones ``lightsim2grid_core`` ships (lightsim2grid >= 1.3.0, its wheel
   included), already on its include path: the usual case, nothing to set.
3. The SuiteSparse checkout next to the lightsim2grid *source* the Python's
   lightsim2grid was built from (per-module ``Include/`` layout), or
   ``-DGPUSIM2GRID_SUITESPARSE_DIR=/path/to/SuiteSparse``.
4. A system SuiteSparse dev package (e.g. Debian/Ubuntu's
   ``libsuitesparse-dev``).

If none is found, configuration stops with an error listing them.

The bridge is compiled with the flags ``lightsim2grid_core`` exports (its
``-O3`` / ``-march=native``), so that both agree on the layout of the objects
they share.

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
