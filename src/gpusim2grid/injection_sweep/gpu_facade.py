# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU injection sweep driven directly from a lightsim2grid grid.

``InjectionSweepGPU`` reuses the CPU base-case solve, keeps the same Ybus, and
re-solves the network on the GPU for many (P, Q) injection profiles in parallel.
It is a thin facade over :class:`_InjectionSweepSolver`.
"""

from . import (
    _InjectionSweepSolver,
    _normalize_device,
    _resolve_reordering_alg,
    _resolve_matching_alg,
    _resolve_pivot_epsilon_alg,
)
from .._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    grid_from_pandapower,
    _validate_precision,
)
from .. import _gpusim2grid as _cpp

__all__ = ["InjectionSweepGPU"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


class InjectionSweepGPU:
    """Batch injection sweep on the GPU, seeded from a CPU base-case solve.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) every scenario is solved on the **same augmented system lightsim2grid
    poses** — distributed slack, HVDC angle-droop, SVC, and remote generator
    voltage control are carried through the Jacobian as Sbus varies. When the
    bridge is unavailable (or ``use_bridge=False``) the Python-array fallback
    solves only the bare ``[pvpq | pq]`` system.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, or tuple
        Either the grid whose base case is solved on the CPU, or an explicit
        ``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)`` array tuple
        for callers without a lightsim2grid grid. In the latter case,
        ``use_bridge`` must not be True and branch data (if needed for
        :meth:`compute_flows`) must be supplied via :meth:`set_branch_data`.
    init_from_n_powerflow : bool, default True
        Seed each scenario with the CPU-converged base-case voltage V0,
        skipping the GPU base-case NR loop (see :class:`ContingencyAnalysisGPU`).
    precision : {"fp64", "fp32", None}, default "fp64"
        Validated against the compiled extension.
    nb_iter : int, default 4
        Newton-Raphson iterations per scenario in the batch phase.
    max_iter_base, tol_base : int, float
        Base-case CPU solve / GPU fallback settings.
    device : None | int | "cuda" | "cuda:N"
        Target CUDA device.
    auto_branch_data : bool, default True
        Extract and store π-model branch admittances so :meth:`compute_flows`
        works without a manual ``set_branch_data`` call.
    reordering_alg : str, optional
        cuDSS ``CUDSS_CONFIG_REORDERING_ALG`` choice, applied ONCE at
        construction to BOTH the base-case solve AND the batch solver used by
        :meth:`compute` -- single source of truth. ``None`` (default) leaves
        it at the session's own default (``'default'``). The
        :attr:`reordering_alg` mutable property can still be changed
        afterward, but only ever affects the batch solver on the next
        :meth:`compute` (the base-case solve is fixed once built).
    matching_alg : str, optional
        cuDSS ``CUDSS_CONFIG_MATCHING_ALG`` choice, same construction-time
        scope as ``reordering_alg`` above; ``None`` (default) leaves it at
        ``'none'``.
    pivot_epsilon_alg : str, optional
        cuDSS ``CUDSS_CONFIG_PIVOT_EPSILON_ALG`` choice, same construction-time
        scope as ``reordering_alg`` above; ``None`` (default) leaves it at
        ``'default'``.
    debug_base_case : bool, default False
        Only meaningful with ``init_from_n_powerflow=True`` and a MultiSlack/
        VoltageControl extension active (bridge path). By default, that
        extension's running state is seeded directly from lightsim2grid's own
        converged values, needing no cuDSS solve at all for the base case.
        Setting this True forces the pre-ground-truth cuDSS-solve derivation
        instead -- an opt-in diagnostic (e.g. to keep testing
        ``reordering_alg``/``matching_alg``/``pivot_epsilon_alg`` choices in
        isolation).
    scaling_max_voltage_change : bool or None, default None
        NR step-scaling, mirrors lightsim2grid's own
        ``MaxVoltageChangeScalingPolicy``: after solving for the Newton step,
        scale it by ``alpha <= 1`` so ``max|dtheta| <= max_dVa`` and
        ``max|dVm| <= max_dVm`` before applying it anywhere. Applied to BOTH
        the base-case solve AND the batch solver used by :meth:`compute` --
        each scenario in the batch gets its OWN alpha from its own max step,
        not one alpha shared across the whole chunk. ``None`` (default) is
        opt-in by inheritance: mirrors whatever ``grid``'s own
        ``get_ac_algo_config()`` is already set to. Pass ``True``/``False``
        to force it on/off regardless of the grid's own config. Without it,
        an undamped GPU Newton step can converge onto a different (sometimes
        spurious) root when seeded far from the solution (e.g.
        ``init_from_n_powerflow=False`` from a DC warm-start) -- observed on
        real RTE grids.
    max_dVa, max_dVm : float or None, default None
        ``MaxVoltageChangeScalingPolicy`` thresholds (radians / pu). ``None``
        inherits the grid's own configured values (or lightsim2grid's own
        defaults, 0.5 / 0.1, if forcing ``scaling_max_voltage_change=True``
        with no grid to inherit from). Ignored unless step-scaling is active.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> grid.ac_pf(Vinit, max_iter=10, tol=1e-8)
    >>> sweep = InjectionSweepGPU(grid, nb_iter=4)
    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)   # (n_scen, n_bus)
    >>> V_batch = sweep.compute(batch_size=512)      # DLPack (n_scen, n_bus)
    >>> residuals = sweep.last_residuals()

    Explicit-array path (no lightsim2grid grid available):

    >>> sweep = InjectionSweepGPU(
    ...     (Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq), nb_iter=4)
    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)
    >>> V_batch = sweep.compute(batch_size=512)
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 auto_branch_data=True, use_bridge=None, reordering_alg=None,
                 matching_alg=None, pivot_epsilon_alg=None,
                 debug_base_case=False,
                 scaling_max_voltage_change=None, max_dVa=None, max_dVm=None):
        _validate_precision(precision)

        # Single source of truth, resolved once here and applied at
        # construction time to BOTH the base-case solve and the batch solver
        # (see _InjectionSweepSolver's identical-shaped ctor) -- None
        # (default) leaves each at the session's own default.
        _reordering_alg = 'default' if reordering_alg is None else reordering_alg
        _matching_alg = 'none' if matching_alg is None else matching_alg
        _pivot_epsilon_alg = 'default' if pivot_epsilon_alg is None else pivot_epsilon_alg

        if isinstance(grid, (tuple, list)):
            # Explicit-array mode: no grid to seed/extract branch data from.
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._inner = _InjectionSweepSolver(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                batch_size=100, nb_iter=nb_iter,
                max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                presolved_v=bool(init_from_n_powerflow),
                reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                pivot_epsilon_alg=_pivot_epsilon_alg,
                debug_base_case=bool(debug_base_case),
                # No grid to inherit a scaling policy from -- None means off.
                scaling_max_voltage_change=bool(scaling_max_voltage_change),
                max_dVa=0.5 if max_dVa is None else float(max_dVa),
                max_dVm=0.1 if max_dVm is None else float(max_dVm))
            # No grid to auto-extract branch data from: call set_branch_data()
            # manually if compute_flows() is needed.
        else:
            if use_bridge is None:
                use_bridge = _have_bridge()

            if use_bridge:
                session = _cpp._make_is_session_from_lsgrid(
                    grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                    int(max_iter_base), float(tol_base), _normalize_device(device),
                    bool(auto_branch_data),
                    reordering_alg=_resolve_reordering_alg(_reordering_alg),
                    matching_alg=_resolve_matching_alg(_matching_alg),
                    pivot_epsilon_alg=_resolve_pivot_epsilon_alg(_pivot_epsilon_alg),
                    debug_base_case=bool(debug_base_case),
                    scaling_max_voltage_change_override=(
                        -1 if scaling_max_voltage_change is None
                        else int(bool(scaling_max_voltage_change))),
                    max_dVa_override=-1.0 if max_dVa is None else float(max_dVa),
                    max_dVm_override=-1.0 if max_dVm is None else float(max_dVm))
                self._inner = _InjectionSweepSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg)
            else:
                d = extract_grid_arrays(grid, max_iter=max_iter_base, tol=tol_base)
                vinit = d["v_converged"] if init_from_n_powerflow else d["v_init"]

                self._inner = _InjectionSweepSolver(
                    d["Ybus"], vinit, d["Sbus"],
                    d["slack"], d["slack_weights"], d["pv"], d["pq"],
                    batch_size=100, nb_iter=nb_iter,
                    max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                    presolved_v=init_from_n_powerflow,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg,
                    debug_base_case=bool(debug_base_case),
                    # Python-array fallback: no C++ bridge to inherit the
                    # grid's algo config through, same as the tuple path.
                    scaling_max_voltage_change=bool(scaling_max_voltage_change),
                    max_dVa=0.5 if max_dVa is None else float(max_dVa),
                    max_dVm=0.1 if max_dVm is None else float(max_dVm))

                if auto_branch_data:
                    branch_args, _, _ = extract_branch_data(grid)
                    self._inner.set_branch_data(*branch_args)

        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances (explicit-array mode only).

        Grid mode extracts this automatically (unless ``auto_branch_data``
        was set to False at construction); only needed when ``grid`` was an
        explicit-array tuple. Required before :meth:`compute_flows`.
        """
        self._inner.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                    bus_vn_kv, sn_mva)

    def set_injections(self, p_mw, q_mvar, sn_mva):
        """Store the (n_scenarios, n_bus) MW / MVAr injection arrays."""
        self._inner.set_injections(p_mw, q_mvar, sn_mva)

    def compute(self, batch_size=512):
        """Solve every scenario; return DLPack (n_scenarios, n_bus) complex.

        The capsule aliases live GPU memory; clone before the next ``compute()``
        for a snapshot.  Residuals are cached for :meth:`last_residuals`.
        """
        self._inner.batch_size = int(batch_size)
        self._inner.run()
        self._last_residuals = self._inner.residuals
        return self._inner.v_results_dlpack()

    def last_residuals(self):
        """``‖F‖∞`` per scenario from the most recent :meth:`compute`."""
        if self._last_residuals is None:
            raise RuntimeError("Call compute() before last_residuals().")
        return self._last_residuals.to_numpy()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute()."""
        self._inner.compute_flows()

    # ----------------------------------------------------------- pass-through
    @property
    def or_amps(self):
        return self._inner.or_amps

    @property
    def ex_amps(self):
        return self._inner.ex_amps

    @property
    def V_results(self):
        return self._inner.V_results

    @property
    def strategy(self):
        return self._inner.strategy

    @strategy.setter
    def strategy(self, value):
        self._inner.strategy = value

    @property
    def reordering_alg(self):
        """cuDSS CUDSS_CONFIG_REORDERING_ALG choice (str). Takes effect on the
        next compute() (which always reruns cuDSS ANALYSIS). One of 'default'
        (default), 'amd', 'nested_dissection', 'none'. 'btf_colamd'/'colamd'
        are rejected by cuDSS (CUDSS_STATUS_NOT_SUPPORTED) in this class's
        uniform-batch mode -- they only work on AcPfGPU's single-system solve."""
        return self._inner.reordering_alg

    @reordering_alg.setter
    def reordering_alg(self, value):
        self._inner.reordering_alg = value

    @property
    def matching_alg(self):
        """cuDSS CUDSS_CONFIG_MATCHING_ALG choice (str). Takes effect on the
        next compute() (which always reruns cuDSS ANALYSIS). 'none' (default)
        is the only value cuDSS accepts in this class's uniform-batch mode --
        every other value raises RuntimeError (CUDSS_STATUS_NOT_SUPPORTED)."""
        return self._inner.matching_alg

    @matching_alg.setter
    def matching_alg(self, value):
        self._inner.matching_alg = value

    @property
    def pivot_epsilon_alg(self):
        """cuDSS CUDSS_CONFIG_PIVOT_EPSILON_ALG choice (str). Takes effect on
        the next compute() (which always reruns cuDSS ANALYSIS). One of
        'default' (default), 'scaled', 'static'."""
        return self._inner.pivot_epsilon_alg

    @pivot_epsilon_alg.setter
    def pivot_epsilon_alg(self, value):
        self._inner.pivot_epsilon_alg = value

    @property
    def timings(self):
        return self._inner.timings

    @property
    def n_scenarios(self):
        return self._inner.n_scenarios

    @property
    def n_branches(self):
        return self._inner.n_branches

    @property
    def n_bus(self):
        return self._inner.n_bus

    @property
    def solver(self):
        """The underlying :class:`_InjectionSweepSolver` (escape hatch)."""
        return self._inner

    # --------------------------------------------------------------- factory
    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)
