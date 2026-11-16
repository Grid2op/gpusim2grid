# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Checks run by ``import gpusim2grid``: the runtime matches what the extension
was built with (``_build_info.py``, written by CMake).

* the cuDSS / cuSPARSE loaded in the process are of the CUDA major nvcc was:
  their sonames are the same for CUDA 12 and 13 (``libcudss.so.1``,
  ``libcusparse.so.12``), so the loader happily mixes them, e.g. a CUDA 13 build
  next to the ``cuda12`` extra, or a CUDA 12 ``LD_LIBRARY_PATH`` (which wins
  over the rpath). Their own dependencies tell them apart: ``libcudss.so.1``
  NEEDs ``libcublas.so.<major>``, a CUDA 12 cuSPARSE ``libnvJitLink.so.12``;
* the driver supports that CUDA major (the runtime is linked statically);
* lightsim2grid is the version the bridge was compiled against: the bridge
  reads its C++ classes' layout, which a new version may change (the Eigen
  SIMD flags are checked separately, by the extension itself).

``GPUSIM2GRID_SKIP_BUILD_CHECKS=1`` skips them all.
"""

from __future__ import annotations

import os
import re

_CUDA_DEP = re.compile(r"^lib(cublas|cublasLt|nvrtc|nvJitLink|cudart)\.so\.(\d+)$")


def cuda_major_mismatches(built_major, libs):
    """Messages for the loaded libraries of another CUDA major than ``built_major``.

    ``libs``: ``[(path, [DT_NEEDED, ...]), ...]`` (``_loaded_library_needs``).
    """
    out = []
    for path, needed in libs:
        for dep in needed:
            m = _CUDA_DEP.match(dep)
            if m and int(m.group(2)) != built_major:
                out.append(f"{path} is a CUDA {m.group(2)} library (it needs {dep})")
                break
    return out


def driver_mismatch(built_major, driver_version):
    """Message when the driver is too old for ``built_major`` (0 = no driver: not checked)."""
    if driver_version and driver_version // 1000 < built_major:
        return (f"the NVIDIA driver supports CUDA up to {driver_version // 1000}."
                f"{driver_version % 1000 // 10}, this build needs CUDA {built_major}")
    return None


def lightsim2grid_mismatch(built_version, runtime_version):
    """Message when lightsim2grid is not the version the bridge was compiled against."""
    if built_version and runtime_version and built_version != runtime_version:
        return (f"lightsim2grid {runtime_version} is installed, gpusim2grid was compiled "
                f"against lightsim2grid {built_version}")
    return None


def _remedy(major):
    other = 12 if major == 13 else 13
    return (
        f"\n\nThis gpusim2grid was built with CUDA {major}. Either install the matching "
        f"libraries:\n    pip install \"gpusim2grid[cuda{major}]\"\n"
        f"(and take any other CUDA {other} directory out of LD_LIBRARY_PATH), or rebuild "
        f"gpusim2grid for what is installed, without pip's wheel cache, which would hand "
        f"back this build:\n"
        f"    pip install --force-reinstall --no-deps --no-cache-dir \"gpusim2grid[cuda{major}]\"\n"
        f"    GPUSIM2GRID_CUDA=12 CUDACXX=/path/to/cuda-12/bin/nvcc pip install "
        f"--force-reinstall --no-deps --no-cache-dir \"gpusim2grid[cuda12]\"\n"
        "See the installation page of the documentation (docs/installation.rst). "
        "GPUSIM2GRID_SKIP_BUILD_CHECKS=1 skips this check.")


def check_build():
    """Raise ImportError when the runtime does not match the build (see the module doc)."""
    if os.environ.get("GPUSIM2GRID_SKIP_BUILD_CHECKS", "").strip() == "1":
        return
    try:
        from . import _build_info as info
    except ImportError:  # a build that predates _build_info.py
        return
    try:
        from . import _gpusim2grid
    except ImportError as exc:
        raise ImportError(
            f"gpusim2grid: the compiled extension (built with CUDA {info.CUDA_MAJOR}) "
            f"cannot be loaded: {exc}\nIts CUDA libraries and cuDSS come from "
            f"`pip install \"gpusim2grid[cuda{info.CUDA_MAJOR}]\"`, or from a toolkit on "
            f"LD_LIBRARY_PATH.") from exc

    problems = cuda_major_mismatches(
        info.CUDA_MAJOR, _gpusim2grid._loaded_library_needs(["libcudss", "libcusparse"]))
    drv = driver_mismatch(info.CUDA_MAJOR, _gpusim2grid._cuda_driver_version())
    if drv:
        problems.append(drv)
    if problems:
        raise ImportError("gpusim2grid: CUDA mismatch:\n  - " + "\n  - ".join(problems)
                          + _remedy(info.CUDA_MAJOR))

    if info.LIGHTSIM2GRID_VERSION:
        try:
            import lightsim2grid
            runtime = getattr(lightsim2grid, "__version__", "")
        except ImportError:
            runtime = ""
        msg = lightsim2grid_mismatch(info.LIGHTSIM2GRID_VERSION, runtime)
        if msg:
            raise ImportError(
                f"gpusim2grid: {msg}. Its C++ bridge reads lightsim2grid's classes and "
                f"must be recompiled against the installed version:\n"
                f"    pip install --force-reinstall --no-deps --no-cache-dir "
                f"--no-build-isolation gpusim2grid\n"
                f"(needs scikit-build-core, pybind11 and cmake installed), or install lightsim2grid=={info.LIGHTSIM2GRID_VERSION}. "
                f"GPUSIM2GRID_SKIP_BUILD_CHECKS=1 skips this check.")
