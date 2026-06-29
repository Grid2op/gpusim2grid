# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Single AC power flow on the GPU, driven from a lightsim2grid grid."""

from ._gpusim2grid import AcPfNrSession as _AcPfNrSession
from ._ls2g_utils import (
    extract_grid_arrays,
    grid_from_pandapower,
    _validate_precision,
)
from .contingency_analysis import _normalize_device

__all__ = ["AcPfGPU"]


class AcPfGPU:
    """Single-system AC Newton-Raphson on the GPU, seeded from a CPU solve.

    Seeds the GPU solver with lightsim2grid's CPU-converged voltage so the
    base-case factorization is computed at the correct operating point.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid
    precision : {"fp64", "fp32", None}, default "fp64"
    max_iter, tol : int, float
    device : None | int | "cuda" | "cuda:N"

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> ac = AcPfGPU(grid)
    >>> V = ac.solve()
    """

    def __init__(self, grid, *, precision="fp64", max_iter=10, tol=1e-8,
                 device=None):
        _validate_precision(precision)
        d = extract_grid_arrays(grid, max_iter=max_iter, tol=tol)
        self._s = _AcPfNrSession(
            d["Ybus"], d["v_converged"], d["Sbus"],
            d["slack"], d["slack_weights"], d["pv"], d["pq"],
            int(max_iter), float(tol), _normalize_device(device))

    def solve(self):
        """Return the solved complex voltage vector (host copy)."""
        return self._s.get_v()

    def v_dlpack(self):
        """Zero-copy DLPack capsule of the solved voltage, shape [n_bus]."""
        return self._s.v_dlpack()

    def solve_JT_dlpack(self, rhs):
        """Solve Jᵀ x = rhs reusing the converged factorization (adjoint)."""
        return self._s.solve_JT_dlpack(rhs)

    @property
    def timings(self):
        """AcPfTimings from construction."""
        return self._s.timings

    @property
    def session(self):
        """The underlying ``AcPfNrSession`` (escape hatch)."""
        return self._s

    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)
