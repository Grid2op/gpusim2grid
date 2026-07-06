# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

__version__ = "0.0.0-dev11"

# Import lightsim2grid first so liblightsim2grid_core.so is loaded into the
# process: the compiled _gpusim2grid extension links against it (the zero-copy
# bridge), and resolving the soname this way avoids depending on an rpath to
# lightsim2grid's install dir. Harmless if the build has no bridge.
try:  # pragma: no cover - environment dependent
    import lightsim2grid as _lightsim2grid  # noqa: F401
except ImportError:
    pass

from .contingency_analysis import ContingencyAnalysisGPU, optimize_reference_slack
from .injection_sweep import InjectionSweepGPU
from .acpf_nr import AcPfGPU

__all__ = [
    "ContingencyAnalysisGPU",
    "optimize_reference_slack",
    "InjectionSweepGPU",
    "AcPfGPU",
]