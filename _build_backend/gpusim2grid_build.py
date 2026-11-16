# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""In-tree PEP 517 backend: scikit-build-core, with the CUDA build requirements
chosen when the build starts.

An extra (``gpusim2grid[cuda12]``) never reaches the build backend, and
``build-system.requires`` is one fixed list, so the CUDA major is picked here,
from the environment:

1. ``GPUSIM2GRID_CUDA=12`` / ``13``, explicitly;
2. else the version of the nvcc ``CUDACXX`` names, else of
   ``$CUDAToolkit_ROOT/bin/nvcc``;
3. else CUDA 13.

An isolated build also takes cmake from PyPI. CUDA 13 without a named compiler
takes the toolkit from PyPI
(``cuda-toolkit[nvcc,...]``, found by CMakeLists.txt); PyPI has no CUDA 12
compiler (``nvidia-cuda-nvcc-cu12`` ships ``ptxas`` only), so CUDA 12 needs a
system nvcc (``CUDACXX``, ``CUDAToolkit_ROOT`` or ``PATH``). Either way cuDSS
comes from PyPI (``nvidia-cudss-cuXX``) unless ``GPUSIM2GRID_CUDSS_FROM_PYPI=0``
(cuDSS installed otherwise, found through ``cudss_ROOT`` / ``cudss_DIR``).

The nvcc found on ``PATH`` alone does not choose the major: a default
``pip install "gpusim2grid[cuda13]"`` builds with CUDA 13 whatever toolkit the
machine has. Keep the requirements in sync with pyproject.toml's extras.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from scikit_build_core import build as _skb
from scikit_build_core.build import *  # noqa: F401,F403  (the PEP 517 hooks)

CUDA_TOOLKIT_REQ = {
    13: "cuda-toolkit[nvcc,crt,cudart,cccl,cublas,cusolver,cusparse,nvjitlink,nvvm]>=13,<14",
}
CUDSS_REQ = {
    12: "nvidia-cudss-cu12>=1.0.0.35,<2",
    13: "nvidia-cudss-cu13>=1.0.0.35,<2",
}
SUPPORTED = (12, 13)
# A CMake that knows the CUDA 13 toolkit (ubuntu 22.04's 3.22 does not): an isolated
# build takes PyPI's, a --no-build-isolation build keeps the one it finds.
CMAKE_REQ = "cmake>=3.24"


class CudaSelectionError(RuntimeError):
    pass


def nvcc_major(nvcc: str) -> int:
    """CUDA major of an nvcc executable (``nvcc --version``)."""
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CudaSelectionError(f"gpusim2grid: cannot run '{nvcc} --version': {exc}") from exc
    m = re.search(r"release (\d+)\.(\d+)", out)
    if not m:
        raise CudaSelectionError(f"gpusim2grid: no CUDA release in '{nvcc} --version':\n{out}")
    return int(m.group(1))


def named_nvcc(env) -> str | None:
    """The nvcc the environment names for CMake (CUDACXX, then CUDAToolkit_ROOT)."""
    if env.get("CUDACXX"):
        return env["CUDACXX"]
    if env.get("CUDAToolkit_ROOT"):
        return os.path.join(env["CUDAToolkit_ROOT"], "bin", "nvcc")
    return None


def cuda_requirements(env=None, which=shutil.which, nvcc_major=nvcc_major) -> list[str]:
    """The CUDA build requirements for this environment (see the module docstring)."""
    env = os.environ if env is None else env
    explicit = env.get("GPUSIM2GRID_CUDA", "").strip()
    if explicit and explicit not in {str(v) for v in SUPPORTED}:
        raise CudaSelectionError(
            f"gpusim2grid: GPUSIM2GRID_CUDA={explicit!r}, expected one of {SUPPORTED}")
    nvcc = named_nvcc(env)
    major = nvcc_major(nvcc) if nvcc else None
    if explicit and major is not None and int(explicit) != major:
        raise CudaSelectionError(
            f"gpusim2grid: GPUSIM2GRID_CUDA={explicit} but {nvcc} is CUDA {major}")
    if major is None:
        major = int(explicit) if explicit else 13
    if major not in SUPPORTED:
        raise CudaSelectionError(f"gpusim2grid: {nvcc} is CUDA {major}, supported: {SUPPORTED}")

    reqs = [CMAKE_REQ]
    if nvcc is None:
        if major in CUDA_TOOLKIT_REQ:
            reqs.append(CUDA_TOOLKIT_REQ[major])
        else:
            path_nvcc = which("nvcc")
            path_major = nvcc_major(path_nvcc) if path_nvcc else None
            if path_major != major:
                found = f"the nvcc on PATH is CUDA {path_major}" if path_nvcc else "no nvcc on PATH"
                raise CudaSelectionError(
                    f"gpusim2grid: a CUDA {major} build needs a CUDA {major} toolkit installed on "
                    f"this machine (PyPI has no CUDA {major} compiler), and {found}. Install one "
                    f"and point CUDACXX at its nvcc, e.g. "
                    f"CUDACXX=/usr/local/cuda-{major}/bin/nvcc pip install \"gpusim2grid[cuda{major}]\"")
    if env.get("GPUSIM2GRID_CUDSS_FROM_PYPI", "1").strip() != "0":
        reqs.append(CUDSS_REQ[major])
    return reqs


def get_requires_for_build_wheel(config_settings=None):
    return [*_skb.get_requires_for_build_wheel(config_settings), *cuda_requirements()]


def get_requires_for_build_editable(config_settings=None):
    return [*_skb.get_requires_for_build_editable(config_settings), *cuda_requirements()]
