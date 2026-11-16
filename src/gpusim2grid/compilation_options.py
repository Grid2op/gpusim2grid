# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from ._gpusim2grid import is_fp32

try:  # written by CMake at configure time
    from ._build_info import (CUDA_VERSION as cuda_version, CUDA_MAJOR as cuda_major,
                              CUDA_ARCHITECTURES as cuda_architectures,
                              CUDSS_VERSION as cudss_version,
                              LIGHTSIM2GRID_VERSION as lightsim2grid_version)
except ImportError:  # a build that predates it
    cuda_version = cuda_major = cuda_architectures = cudss_version = None
    lightsim2grid_version = None

__all__ = ["is_fp32", "cuda_version", "cuda_major", "cuda_architectures", "cudss_version",
           "lightsim2grid_version"]