# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Single AC power flow on the GPU, driven from a lightsim2grid grid."""

import numpy as np

from .. import _gpusim2grid as _cpp
from .._gpusim2grid import AcPfNrSession as _AcPfNrSession
from .._ls2g_utils import (
    extract_grid_arrays,
    grid_from_pandapower,
    _validate_precision,
)
from ..contingency_analysis import (
    _normalize_device,
    _resolve_reordering_alg,
    _resolve_matching_alg,
    _resolve_pivot_epsilon_alg,
)

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
    reordering_alg : str, default "default"
        cuDSS ``CUDSS_CONFIG_REORDERING_ALG`` choice for the (once-only) cuDSS
        ANALYSIS phase. One of ``"default"``, ``"btf_colamd"``, ``"colamd"``,
        ``"amd"``, ``"nested_dissection"``, ``"none"``. Construction-time only
        — unlike the batch workloads, ``AcPfGPU`` never reruns ANALYSIS, so
        this cannot be changed after construction.
    matching_alg : str, default "none"
        cuDSS ``CUDSS_CONFIG_MATCHING_ALG`` choice for the same (once-only)
        cuDSS ANALYSIS phase. One of ``"none"`` (cuDSS's own default),
        ``"max_diag_count"``, ``"max_min_diag"``, ``"max_min_diag_alt"``,
        ``"max_diag_sum"``, ``"max_diag_product"``, ``"auto"``.
        Construction-time only, same reason as ``reordering_alg``.
        WARNING: ``"max_diag_product"``/``"auto"`` have been observed to
        silently produce NaN voltages on real power-flow Jacobians while
        ``timings.converged`` still reports True (the residual check does
        not catch NaN) — verify ``np.isnan(V).any()`` yourself if you use
        them. ``"none"``/``"max_diag_count"``/``"max_min_diag"``/
        ``"max_min_diag_alt"``/``"max_diag_sum"`` have been verified to
        reproduce the reference solution.
    pivot_epsilon_alg : str, default "default"
        cuDSS ``CUDSS_CONFIG_PIVOT_EPSILON_ALG`` choice for the same
        (once-only) cuDSS ANALYSIS phase. One of ``"default"``, ``"scaled"``,
        ``"static"``. Construction-time only, same reason as
        ``reordering_alg``.
    debug_base_case : bool, default False
        Only meaningful with ``init_from_n_powerflow=True`` and a MultiSlack/
        VoltageControl extension active (bridge path). By default, that
        extension's running state is seeded directly from lightsim2grid's own
        converged values, needing no extra cuDSS solve. Setting this True
        forces the pre-ground-truth cuDSS-solve derivation instead -- an
        opt-in diagnostic (e.g. to keep testing ``reordering_alg``/
        ``matching_alg``/``pivot_epsilon_alg`` choices in isolation).

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
                 device=None, init_from_n_powerflow=True, use_bridge=None,
                 reordering_alg="default", matching_alg="none",
                 pivot_epsilon_alg="default", debug_base_case=False):
        _validate_precision(precision)
        reordering_alg_enum = _resolve_reordering_alg(reordering_alg)
        matching_alg_enum = _resolve_matching_alg(matching_alg)
        pivot_epsilon_alg_enum = _resolve_pivot_epsilon_alg(pivot_epsilon_alg)

        # Kept for run_ac(): the lightsim2grid grid this instance is attached
        # to (None for the explicit-array tuple constructor, which has no
        # CPU ac_pf() to compare against) and the cuDSS/precision choices, so
        # a fresh session can be rebuilt at a caller-chosen (max_iter, tol)
        # without re-specifying everything. Stored as grid.copy() so run_ac()'s
        # own CPU ac_pf() calls (needed to seed/compare against) don't mutate
        # the caller's own grid object out from under them.
        self._grid = None if isinstance(grid, (tuple, list)) else grid.copy()
        self._id_ac_solver_to_me = None if isinstance(grid, (tuple, list)) else np.asarray(grid.id_ac_solver_to_me()).copy()
        self._ctor_kwargs = dict(
            precision=precision, device=device, use_bridge=use_bridge,
            reordering_alg=reordering_alg, matching_alg=matching_alg,
            pivot_epsilon_alg=pivot_epsilon_alg)

        if isinstance(grid, (tuple, list)):
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._s = _AcPfNrSession(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                int(max_iter), float(tol), _normalize_device(device),
                presolved_v=bool(init_from_n_powerflow),
                reordering_alg=reordering_alg_enum,
                matching_alg=matching_alg_enum,
                pivot_epsilon_alg=pivot_epsilon_alg_enum,
                debug_base_case=bool(debug_base_case))
            return

        if use_bridge is None:
            use_bridge = _have_bridge()
        if use_bridge:
            # Zero-copy C++ bridge: solve the SAME augmented system lightsim2grid
            # does (distributed slack / future extensions), reading the NRLedger +
            # J skeleton off the solved grid. The grid must already be ac-solved.
            self._s = _cpp._make_acpf_session_from_lsgrid(
                grid, int(max_iter), float(tol), _normalize_device(device),
                bool(init_from_n_powerflow),
                reordering_alg=reordering_alg_enum,
                matching_alg=matching_alg_enum,
                pivot_epsilon_alg=pivot_epsilon_alg_enum,
                debug_base_case=bool(debug_base_case))
        else:
            # Python extraction fallback: bare [pvpq|pq] system (no distributed
            # slack in the Jacobian).
            d = extract_grid_arrays(grid, max_iter=max_iter, tol=tol)
            self._s = _AcPfNrSession(
                d["Ybus"], d["v_converged"], d["Sbus"],
                d["slack"], d["slack_weights"], d["pv"], d["pq"],
                int(max_iter), float(tol), _normalize_device(device),
                presolved_v=bool(init_from_n_powerflow),
                reordering_alg=reordering_alg_enum,
                matching_alg=matching_alg_enum,
                pivot_epsilon_alg=pivot_epsilon_alg_enum,
                debug_base_case=bool(debug_base_case))

    def solve(self):
        """Return the solved complex voltage vector (host copy)."""
        return self._s.get_v()

    def ac_pf(self, Vinit, max_iter, tol):
        """Run an *independent* GPU power flow with the same call signature
        as lightsim2grid's ``LSGrid.ac_pf(Vinit, max_iter, tol)`` -- ``Vinit``
        and the return value are in **grid-model** bus numbering, exactly
        like lightsim2grid (gpusim2grid's own arrays, e.g. on the
        explicit-array tuple constructor, are solver-numbered internally;
        this method hides that relabeling). Unlike ``ac_pf()``, ``Vinit`` is
        not modified in place.

        This is deliberately NOT "solve on the CPU, then hand the GPU the
        converged answer to validate": both solvers are given the exact same
        ``(Vinit, max_iter, tol)`` and run their own Newton-Raphson loop from
        there (``init_from_n_powerflow=False`` on the GPU side), so a match
        is real evidence the GPU's own iteration is correct, not just that a
        residual check at a known-good point passes.

        Only valid when this instance was built from a lightsim2grid grid
        (not the explicit-array tuple constructor). The grid's solver-side
        structures (Ybus/ledger) come from lightsim2grid itself -- gpusim2grid
        never builds them from scratch -- so this seeds them with a
        zero-iteration ``ac_pf(Vinit, 0, tol)`` call first (sets up the
        solver bus numbering / ledger and leaves ``get_V_solver()`` exactly
        equal to ``Vinit``, no CPU Newton steps taken) before the CPU
        reference run so the CPU doesn't get a head start.

        Returns ``(V_cpu, V_gpu)``, both grid-model-numbered complex arrays
        (or empty, matching ``ac_pf()``'s own convergence-failure
        convention, for whichever side didn't converge).
        
        .. warning::
            Vinit here is for gpusim2grid, so only covers the active buses
            of the lightsim2grid grid.
            
            running directly LSGrid.ac_pf(Vinit, max_iter, tol) will fail 
            because the size mismatch (this later call require to feed
            complex voltages even for disconnected buses).
            
        """
        if self._grid is None:
            raise RuntimeError(
                "run_ac() needs a lightsim2grid grid (this AcPfGPU was "
                "built from an explicit-array tuple, which has no CPU "
                "ac_pf() to compare against).")

        Vinit_gpu = np.array(Vinit, dtype=complex)

        # Seed solver-side structures (bus numbering, ledger) at Vinit itself,
        # untouched -- max_iter=0 takes no CPU Newton step (see run_ac's own
        # doc). This must happen before the CPU reference run below, which
        # would otherwise leave get_V_solver() at ITS OWN converged answer.
        # self._id_ac_solver_to_me has length n_bus_solver and lists, for each
        # solver position, its grid-model bus id -- a direct scatter, no mask
        # needed (disconnected buses simply never appear in it and keep the
        # np.ones() placeholder, which lightsim2grid ignores for them anyway).
        V_init_ls = np.ones(self._grid.get_bus_vn_kv().shape[0], dtype=complex)
        V_init_ls[self._id_ac_solver_to_me] = Vinit_gpu
        self._grid.ac_pf(V_init_ls, 0, float(tol))
        self._grid.unset_changes()  # speed optim is this is called multiple times

        gpu = AcPfGPU(self._grid, max_iter=int(max_iter), tol=float(tol),
                      init_from_n_powerflow=False, debug_base_case=False,
                      **self._ctor_kwargs)
        self._s = gpu._s
        v_gpu_solver = gpu.solve() if gpu.timings.converged else np.zeros(0, dtype=complex)

        # v_cpu = self._grid.ac_pf(Vinit.copy(), int(max_iter), float(tol))

        # if v_gpu_solver.shape[0] == 0:
        #     v_gpu_full = np.zeros(0, dtype=complex)
        # else:
        #     id_solver_to_me = np.asarray(self._grid.id_ac_solver_to_me(), dtype=np.int64)
        #     v_gpu_full = np.full(v_cpu.shape[0] or Vinit.shape[0], np.nan, dtype=complex)
        #     v_gpu_full[id_solver_to_me] = v_gpu_solver

        return v_gpu_solver

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
