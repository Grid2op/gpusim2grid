# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

__version__ = "0.2.0-rc0"

# Import lightsim2grid first so liblightsim2grid_core.so is loaded into the
# process: the compiled _gpusim2grid extension links against it (the zero-copy
# bridge), and resolving the soname this way avoids depending on an rpath to
# lightsim2grid's install dir. Harmless if the build has no bridge.
try:  # pragma: no cover - environment dependent
    import lightsim2grid as _lightsim2grid  # noqa: F401
except ImportError:
    pass

from .contingency_analysis import (ContingencyAnalysisGPU, optimize_reference_slack,
                                   ReactiveLimitsStatus, RowStatus)
from .injection_sweep import InjectionSweepGPU
from .scenario_sweep import ScenarioSweepGPU
from .acpf_nr import AcPfGPU
from ._gpusim2grid import warmup

__all__ = [
    "ContingencyAnalysisGPU",
    "optimize_reference_slack",
    "ReactiveLimitsStatus",
    "RowStatus",
    "InjectionSweepGPU",
    "ScenarioSweepGPU",
    "AcPfGPU",
    "warmup",
    "get_cmake_dir",
]


def get_cmake_dir() -> str:
    """Return the CMake config directory of ``gpusim2grid_core``, gpusim2grid's solvers as a
    C++ shared library without Python (imported target ``gpusim2grid::core``)::

        cmake -Dgpusim2grid_core_DIR=$(python -c "import gpusim2grid; print(gpusim2grid.get_cmake_dir())")

    Only installed by a build with the lightsim2grid bridge.
    """
    import os
    from . import _gpusim2grid
    # next to the compiled module (in an editable install, not next to this file)
    return os.path.join(os.path.dirname(os.path.abspath(_gpusim2grid.__file__)), "share", "cmake", "gpusim2grid_core")