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
    _check_ir_n_steps,
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
        produce NaN voltages on real power-flow Jacobians (reported as
        ``timings.converged == False``) — do not rely on them. ``"none"``/``"max_diag_count"``/``"max_min_diag"``/
        ``"max_min_diag_alt"``/``"max_diag_sum"`` have been verified to
        reproduce the reference solution.
    pivot_epsilon_alg : str, default "default"
        cuDSS ``CUDSS_CONFIG_PIVOT_EPSILON_ALG`` choice for the same
        (once-only) cuDSS ANALYSIS phase. One of ``"default"``, ``"scaled"``,
        ``"static"``. Construction-time only, same reason as
        ``reordering_alg``.
    ir_n_steps : int, default 0
        cuDSS ``CUDSS_CONFIG_IR_N_STEPS``: steps of iterative refinement after
        each linear solve (forward and adjoint). ``0`` (default) solves
        without refinement. Construction-time only.
    debug_base_case : bool, default False
        Only meaningful with ``init_from_n_powerflow=True`` and a MultiSlack/
        VoltageControl extension active (bridge path). By default, that
        extension's running state is seeded directly from lightsim2grid's own
        converged values, needing no extra cuDSS solve. Setting this True
        forces the pre-ground-truth cuDSS-solve derivation instead -- an
        opt-in diagnostic (e.g. to keep testing ``reordering_alg``/
        ``matching_alg``/``pivot_epsilon_alg`` choices in isolation).
    scaling_max_voltage_change : bool or None, default None
        NR step-scaling, mirrors lightsim2grid's own
        ``MaxVoltageChangeScalingPolicy``: after solving for the Newton step,
        scale the WHOLE step by ``alpha <= 1`` so ``max|dtheta| <= max_dVa``
        and ``max|dVm| <= max_dVm`` before applying it anywhere -- matching
        lightsim2grid's own damped trajectory instead of taking the full raw
        step every iteration. Only takes effect with ``init_from_n_powerflow=
        False`` (the GPU actually iterates) and the bridge path. ``None``
        (default) is opt-in by inheritance: mirrors whatever the ``grid``'s
        own ``get_ac_algo_config()`` is already set to (e.g. via
        lightsim2grid's ``set_ac_algo_config``), so behavior only changes for
        grids the caller explicitly configured with damping -- everything
        else stays bit-for-bit as before. Pass ``True``/``False`` to force it
        on/off regardless of the grid's own config. Without it, an undamped
        GPU Newton step can converge onto a different (sometimes spurious)
        root than lightsim2grid's own when seeded far from the solution
        (e.g. a flat/DC start) -- observed on real RTE grids.
    max_dVa, max_dVm : float or None, default None
        ``MaxVoltageChangeScalingPolicy`` thresholds (radians / pu). ``None``
        inherits the grid's own configured values (or lightsim2grid's own
        defaults, 0.5 / 0.1, if forcing ``scaling_max_voltage_change=True``
        with no grid to inherit from). Ignored unless step-scaling is active.
    use_distributed_slack : bool, default True
        Selects which of the two slack formulations the GPU solves.

        ``True`` keeps the ``MultiSlack`` augmentation lightsim2grid poses:
        the mismatch is shared across participants per ``slack_weights``,
        carried by one extra ``slack_absorbed`` column plus the P equation of
        every participant.

        ``False`` drops exactly those rows/columns and solves the classic
        single-slack ``[pvpq | pq]`` system instead, shrinking ``dim_J`` by the
        participant count. **Every other in-Jacobian control is preserved
        either way** -- HVDC angle-droop and SVC / remote generator voltage
        control are untouched by this switch.

        With a single (non-distributed) slack, ``False`` is an exact
        reformulation: that row/column pair is block-triangular, so no other
        row's solution moves and only ``slack_absorbed`` stops being reported.
        With several participants it is a genuine model change, and combining
        it with ``init_from_n_powerflow=True`` then legitimately raises on the
        residual check -- the grid's own converged ``V`` solves the other
        system.

        Only meaningful on the lightsim2grid-bridge path; the explicit-array
        and Python-fallback paths have no ledger and always solve the bare
        system.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> ac = AcPfGPU(grid)
    >>> V = ac.solve()

    Explicit-array path (no lightsim2grid grid available):

    >>> ac = AcPfGPU((Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq))
    >>> V = ac.solve()

    Many power flows on one grid: build once, then :meth:`resolve`

    >>> t = ac.resolve(V0, Sbus_new)
    >>> V = ac.solve() if t.converged else None
    """

    def __init__(self, grid, *, precision="fp64", max_iter=10, tol=1e-8,
                 device=None, init_from_n_powerflow=True, use_bridge=None,
                 reordering_alg="default", matching_alg="none",
                 pivot_epsilon_alg="default", ir_n_steps=0, debug_base_case=False,
                 scaling_max_voltage_change=None, max_dVa=None, max_dVm=None,
                 use_distributed_slack=True):
        _validate_precision(precision)
        reordering_alg_enum = _resolve_reordering_alg(reordering_alg)
        matching_alg_enum = _resolve_matching_alg(matching_alg)
        pivot_epsilon_alg_enum = _resolve_pivot_epsilon_alg(pivot_epsilon_alg)
        ir_n_steps = _check_ir_n_steps(ir_n_steps)
        # (mask, |V|) of the Vm-fixed buses, read off the construction solve by
        # the first resolve() -- see _vm_setpoints().
        self._vm_fixed = None

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
                ir_n_steps=ir_n_steps,
                debug_base_case=bool(debug_base_case),
                # No grid to inherit a scaling policy from -- None means off.
                scaling_max_voltage_change=bool(scaling_max_voltage_change),
                max_dVa=0.5 if max_dVa is None else float(max_dVa),
                max_dVm=0.1 if max_dVm is None else float(max_dVm))
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
                ir_n_steps=ir_n_steps,
                debug_base_case=bool(debug_base_case),
                # None => -1/-1.0 sentinels: inherit the grid's own
                # get_ac_algo_config() (opt-in by construction). True/False
                # force it on/off regardless of what the grid is configured
                # with; an explicit max_dVa/max_dVm overrides just that value.
                scaling_max_voltage_change_override=(
                    -1 if scaling_max_voltage_change is None
                    else int(bool(scaling_max_voltage_change))),
                max_dVa_override=-1.0 if max_dVa is None else float(max_dVa),
                max_dVm_override=-1.0 if max_dVm is None else float(max_dVm),
                use_distributed_slack=bool(use_distributed_slack))
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
                ir_n_steps=ir_n_steps,
                debug_base_case=bool(debug_base_case),
                # No grid config to inherit here either (Python-side fallback,
                # not the bridge) -- None means off.
                scaling_max_voltage_change=bool(scaling_max_voltage_change),
                max_dVa=0.5 if max_dVa is None else float(max_dVa),
                max_dVm=0.1 if max_dVm is None else float(max_dVm))

    def solve(self):
        """Return the solved complex voltage vector (host copy)."""
        return self._s.get_v()

    def resolve(self, Vinit, Sbus=None, *, max_iter=None, tol=None):
        """Solve again from ``Vinit`` on this instance's GPU state.

        Reuses everything construction built -- Ybus, the (augmented)
        Jacobian structure, the cuSPARSE descriptor and the cuDSS handle with
        its ANALYSIS -- so only the Newton-Raphson iterations run. Much
        cheaper than building a new ``AcPfGPU`` per power flow.

        Parameters
        ----------
        Vinit : (n_bus,) complex
            Start voltage, AC-solver bus numbering (same as ``solve()``).
        Sbus : (n_bus,) complex or None
            New per-unit injections; ``None`` keeps the current ones. Ybus
            and the pv/pq labelling cannot change.
        max_iter, tol : int, float or None
            ``None`` uses the constructor's.

        ``Vinit`` is taken as given, like at construction: ``|V|`` at the
        Vm-fixed buses (PV, slack) is their voltage set-point and the Newton
        loop never moves it. :meth:`ac_pf` writes the grid's set-points there
        first, as lightsim2grid's ``ac_pf`` does.

        Always iterates (``init_from_n_powerflow`` only applies at
        construction); the distributed-slack / voltage-control running state
        restarts as at construction. ``solve()``, ``v_dlpack()`` (overwritten
        in place) and ``solve_JT_dlpack()`` follow the new solution.

        Returns
        -------
        AcPfTimings
            This solve's timings (also ``self.timings``); check ``converged``.
        """
        self._vm_setpoints()   # captured before the first solve moves V
        return self._s.resolve(
            np.ascontiguousarray(Vinit, dtype=complex),
            None if Sbus is None else np.ascontiguousarray(Sbus, dtype=complex),
            -1 if max_iter is None else int(max_iter),
            -1.0 if tol is None else float(tol))

    def ac_pf(self, Vinit, max_iter, tol):
        """GPU power flow with the call signature of lightsim2grid's
        ``LSGrid.ac_pf(Vinit, max_iter, tol)``, run by :meth:`resolve` on this
        instance's state (the GPU iterates from ``Vinit``; no CPU solve).

        Unlike ``LSGrid.ac_pf``, ``Vinit`` and the result are in **AC-solver**
        bus numbering (``n_bus_solver`` long, connected buses only, the order
        of ``solve()``), not grid-model numbering. ``Vinit`` is not modified.

        As lightsim2grid does, the voltage set-points are written into
        ``|V|`` at the Vm-fixed buses (PV, slack) before solving; the angles,
        and the other buses, start from ``Vinit``.

        Returns the converged voltage, or an empty array if the GPU did not
        converge (``ac_pf``'s own convention).
        """
        mask, vm = self._vm_setpoints()
        V = np.array(Vinit, dtype=complex)
        V[mask] = vm[mask] * np.exp(1j * np.angle(V[mask]))
        t = self.resolve(V, max_iter=max_iter, tol=tol)
        return self._s.get_v() if t.converged else np.zeros(0, dtype=complex)

    def _vm_setpoints(self):
        """``(mask, |V|)``: the buses without a Vm unknown and their voltage
        magnitude. NR never moves |V| there, so the construction solve holds
        the set-points (lightsim2grid's own seeding, or the caller's tuple
        ``Vinit``); read once, before any :meth:`resolve`."""
        if self._vm_fixed is None:
            mask = np.asarray(self._s.vm_col_of_bus) < 0
            self._vm_fixed = (mask, np.abs(self._s.get_v()))
        return self._vm_fixed

    def v_dlpack(self):
        """Zero-copy DLPack capsule of the solved voltage, shape [n_bus]."""
        return self._s.v_dlpack()

    def solve_JT_dlpack(self, rhs):
        """Solve Jᵀ x = rhs reusing the converged factorization (adjoint)."""
        return self._s.solve_JT_dlpack(rhs)

    @property
    def timings(self):
        """AcPfTimings of the last solve (construction, or the last
        :meth:`resolve`)."""
        return self._s.timings

    @property
    def session(self):
        """The underlying ``AcPfNrSession`` (escape hatch)."""
        return self._s

    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)
