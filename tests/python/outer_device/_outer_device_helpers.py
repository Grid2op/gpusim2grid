# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared helpers of the ``outer_loops_backend='device'`` tests (docs/outer_loops_on_device.md)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import gpusim2grid._gpusim2grid as _cpp

__all__ = ["MAX_IT", "OUTER", "TOL_MVA", "case14_outer", "needs_fp64", "needs_outer_loops"]

OUTER: str = "NROuter_KLU"
MAX_IT: int = 30
TOL_MVA: float = 1e-8

#: the outer loops need the bridge and a lightsim2grid with NROuter_*
needs_outer_loops = pytest.mark.skipif(
    not getattr(_cpp, "have_ls2g_outer_loops", False),
    reason="gpusim2grid built without the bridge or against a lightsim2grid without NROuter_*",
)

#: ``outer_loops_backend='device'`` works in double: an FP32 build refuses it at construction
#: (``_resolve_outer_loops_backend``), and the host-vs-device parity bounds (``TOL_MVA``, V to
#: rounding) are FP64 ones. Every test that runs the device backend, or compares to that rounding,
#: carries it; ``test_backend_option.py`` keeps its refusal tests running on FP32 builds.
needs_fp64 = pytest.mark.skipif(bool(_cpp.is_fp32), reason="outer_loops_backend='device' needs an FP64 build")


def case14_outer(q_limit: float = 15.0, distributed: bool = True) -> Any:
    """case14 solved with its outer loops: its generators short of reactive power (ReactiveLimits
    switches buses PQ), sharing the slack in proportion to their maximum output (DistributedSlack)
    -- the grid of test_outer_loops_batch.py."""
    import pandapower.networks as pn
    from lightsim2grid.network import init_from_pandapower

    net = pn.case14()
    net.gen["min_q_mvar"] = -q_limit
    net.gen["max_q_mvar"] = q_limit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        grid = init_from_pandapower(net)
    nb_gen: int = len(grid.get_generators())
    grid.set_gen_p_limits(np.zeros(nb_gen), np.full(nb_gen, 300.0))
    if distributed:
        grid.set_gen_can_participate_slack([True] * nb_gen, np.full(nb_gen, 300.0))
    grid.change_algorithm(OUTER)
    v = grid.ac_pf(np.full(grid.total_bus(), 1.04 + 0j), MAX_IT, TOL_MVA)
    assert v.size > 0
    return grid
