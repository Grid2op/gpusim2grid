# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Single AC power flow on the GPU, driven from a lightsim2grid grid."""

from .. import _gpusim2grid as _cpp
from .._gpusim2grid import AcPfNrSession as _AcPfNrSession
from .._ls2g_utils import (
    extract_grid_arrays,
    grid_from_pandapower,
    _validate_precision,
)
from ..contingency_analysis import _normalize_device

__all__ = ["AcPfGPU"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


class AcPfGPU:
    """Single-system AC Newton-Raphson on the GPU, seeded from a CPU solve.

    Seeds the GPU solver with lightsim2grid's CPU-converged voltage so the
    base-case factorization is computed at the correct operating point.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) this solves the **same augmented system lightsim2grid poses**: the
    in-Jacobian power-system controls — distributed slack, HVDC angle-droop, SVC,
    and remote generator voltage control — are reproduced bit-for-bit from the
    NRLedger read off the solved grid. When the bridge is unavailable (or
    ``use_bridge=False``) it falls back to extracting the grid arrays in Python
    and solving only the bare ``[pvpq | pq]`` system (no distributed slack in the
    Jacobian).

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, or tuple
        Either the grid whose base case is (or will be) solved on the CPU, or
        an explicit ``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)``
        array tuple for callers without a lightsim2grid grid. In the latter
        case ``use_bridge`` must not be True (no grid to bridge to), and only
        the bare ``[pvpq | pq]`` system is solved.
    precision : {"fp64", "fp32", None}, default "fp64"
    max_iter, tol : int, float
    device : None | int | "cuda" | "cuda:N"
    init_from_n_powerflow : bool, default True
        Trust the CPU-converged voltage (``grid.get_V_solver()``) as already
        solved: validate ``‖F(V0)‖∞`` once and skip the GPU NR loop entirely
        (raises ``RuntimeError`` if the residual check fails). When False, the
        GPU runs up to ``max_iter`` iterations from that same V0 seed.
    use_bridge : bool or None, default None
        Force (``True``) or disable (``False``) the zero-copy lightsim2grid C++
        bridge. ``None`` auto-detects it. Only the bridge path solves the
        augmented system; the Python-array fallback solves the bare system.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> ac = AcPfGPU(grid)
    >>> V = ac.solve()

    Explicit-array path (no lightsim2grid grid available):

    >>> ac = AcPfGPU((Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq))
    >>> V = ac.solve()
    """

    def __init__(self, grid, *, precision="fp64", max_iter=10, tol=1e-8,
                 device=None, init_from_n_powerflow=True, use_bridge=None):
        _validate_precision(precision)

        if isinstance(grid, (tuple, list)):
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._s = _AcPfNrSession(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                int(max_iter), float(tol), _normalize_device(device),
                presolved_v=bool(init_from_n_powerflow))
            return

        if use_bridge is None:
            use_bridge = _have_bridge()
        if use_bridge:
            # Zero-copy C++ bridge: solve the SAME augmented system lightsim2grid
            # does (distributed slack / future extensions), reading the NRLedger +
            # J skeleton off the solved grid. The grid must already be ac-solved.
            self._s = _cpp._make_acpf_session_from_lsgrid(
                grid, int(max_iter), float(tol), _normalize_device(device),
                bool(init_from_n_powerflow))
        else:
            # Python extraction fallback: bare [pvpq|pq] system (no distributed
            # slack in the Jacobian).
            d = extract_grid_arrays(grid, max_iter=max_iter, tol=tol)
            self._s = _AcPfNrSession(
                d["Ybus"], d["v_converged"], d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                int(max_iter), float(tol), _normalize_device(device),
                presolved_v=bool(init_from_n_powerflow))

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
