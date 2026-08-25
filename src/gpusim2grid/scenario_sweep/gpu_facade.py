# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU row-aligned combined topology + injection sweep, driven directly from
a lightsim2grid grid.

``ScenarioSweepGPU`` reuses the CPU base-case solve and re-solves the network
on the GPU for many (injection, topology) pairs in parallel: row `i` of the
injection matrices is solved together with row `i` of the tripped-branch
lists set via :meth:`ScenarioSweepGPU.set_topology`, independently of every
other row. It is a thin facade over :class:`_ScenarioSweepSolver`.

This mirrors lightsim2grid's own ``ScenarioSweep`` (see the ``BaseBatchSweep``
policy refactor), added as a distinct class rather than folded into
:class:`gpusim2grid.InjectionSweepGPU` or
:class:`gpusim2grid.ContingencyAnalysisGPU` because the usage pattern differs
from both: one injection/topology pair per row, not a shared base case with a
distinct scenario set.

``handle_disconnected_grid`` and ``compute_limit_violations`` are supported
identically to :class:`gpusim2grid.ContingencyAnalysisGPU` -- see that
class's docs for the full semantics.
"""

import numpy as np

from . import (
    _ScenarioSweepSolver,
    _normalize_device,
    _resolve_reordering_alg,
    _resolve_matching_alg,
    _resolve_pivot_epsilon_alg,
)
from .._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    extract_injection_elements,
    build_bus_injections,
    grid_from_pandapower,
    _validate_precision,
)
from .. import _gpusim2grid as _cpp

__all__ = ["ScenarioSweepGPU"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


class ScenarioSweepGPU:
    """Batch row-aligned topology + injection sweep on the GPU, seeded from a
    CPU base-case solve.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) every scenario is solved on the **same augmented system lightsim2grid
    poses** — distributed slack, HVDC angle-droop, SVC, and remote generator
    voltage control are carried through the Jacobian as Sbus/topology vary.
    When the bridge is unavailable (or ``use_bridge=False``) the Python-array
    fallback solves only the bare ``[pvpq | pq]`` system.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, or tuple
        Either the grid whose base case is solved on the CPU, or an explicit
        ``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)`` array tuple
        for callers without a lightsim2grid grid. In the latter case,
        ``use_bridge`` must not be True and branch data (needed for
        :meth:`set_topology` and :meth:`compute_flows`) must be supplied via
        :meth:`set_branch_data`.
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
    reordering_alg, matching_alg, pivot_epsilon_alg : str, optional
        cuDSS config choices, applied ONCE at construction to BOTH the
        base-case solve AND the batch solver used by :meth:`compute` --
        single source of truth. ``None`` (default) leaves each at the
        session's own default. See :class:`InjectionSweepGPU` for the full
        description of each knob (identical semantics here).
    debug_base_case : bool, default False
        Opt-in diagnostic; see :class:`InjectionSweepGPU`.
    scaling_max_voltage_change, max_dVa, max_dVm : bool/float or None
        NR step-scaling; see :class:`InjectionSweepGPU`.
    use_distributed_slack : bool, default True
        Selects which of the two slack formulations the GPU solves; see
        :class:`InjectionSweepGPU`.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> grid.ac_pf(Vinit, max_iter=10, tol=1e-8)
    >>> sweep = ScenarioSweepGPU(grid, nb_iter=4)
    >>> sweep.set_injections_from_elements(load_p, load_q, gen_p)
    ...                                 # (n_scen, n_load) / (n_scen, n_gen)
    >>> sweep.set_topology([[3], [], [3, 40]])  # one branch-id list per row
    >>> sweep.set_gen_v(gen_v)                  # optional: (n_scen, n_gen) vm_pu
    >>> V_batch = sweep.compute(batch_size=512)      # DLPack (n_scen, n_bus)
    >>> residuals = sweep.last_residuals()
    >>> disconnected = sweep.get_disconnected()      # which rows were skipped

    The per-bus injection form is still available for callers that already
    assembled Sbus themselves, in AC-solver bus numbering:

    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)   # (n_scen, n_bus)

    ``set_topology`` is optional: a row never covered defaults to "no
    branches tripped" (a plain injection sweep for that row).

    Explicit-array path (no lightsim2grid grid available):

    >>> sweep = ScenarioSweepGPU(
    ...     (Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq), nb_iter=4)
    >>> sweep.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, sn_mva)
    >>> sweep.set_injections(p_mw, q_mvar, sn_mva)
    >>> sweep.set_topology([[3], [], [3, 40]])
    >>> V_batch = sweep.compute(batch_size=512)
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 use_bridge=None, handle_disconnected_grid=False,
                 compute_limit_violations=False, reordering_alg=None,
                 matching_alg=None, pivot_epsilon_alg=None,
                 debug_base_case=False,
                 scaling_max_voltage_change=None, max_dVa=None, max_dVm=None,
                 use_distributed_slack=True):
        _validate_precision(precision)

        _reordering_alg = 'default' if reordering_alg is None else reordering_alg
        _matching_alg = 'none' if matching_alg is None else matching_alg
        _pivot_epsilon_alg = 'default' if pivot_epsilon_alg is None else pivot_epsilon_alg

        if isinstance(grid, (tuple, list)):
            # Explicit-array mode: no grid to seed/extract branch data from.
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            self._grid = None
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._inner = _ScenarioSweepSolver(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                batch_size=100, nb_iter=nb_iter,
                max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                presolved_v=bool(init_from_n_powerflow),
                reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                pivot_epsilon_alg=_pivot_epsilon_alg,
                debug_base_case=bool(debug_base_case),
                scaling_max_voltage_change=bool(scaling_max_voltage_change),
                max_dVa=0.5 if max_dVa is None else float(max_dVa),
                max_dVm=0.1 if max_dVm is None else float(max_dVm))
            # No grid to auto-extract branch/limit data from: call
            # set_branch_data() manually before set_topology()/compute_flows()
            # (and set_limits() before run() when compute_limit_violations
            # is wanted).
            self._n_branches = 0
            self._inner.compute_limit_violations = bool(compute_limit_violations)
        else:
            # Retained for set_limits_from_grid()/converged_n()/get_violations_n(),
            # which need to read the grid's own (bus/branch limit, base-case V)
            # state after construction.
            self._grid = grid

            if use_bridge is None:
                use_bridge = _have_bridge()

            if use_bridge:
                # Branch data is always set -- set_topology() needs it, not
                # just compute_flows() -- see ls2g_bridge.cpp:make_ss_session_from_lsgrid.
                # When compute_limit_violations is True, the bridge also pulls
                # bus/branch limits off the grid and enables the fused check.
                session = _cpp._make_ss_session_from_lsgrid(
                    grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                    int(max_iter_base), float(tol_base), _normalize_device(device),
                    bool(compute_limit_violations),
                    reordering_alg=_resolve_reordering_alg(_reordering_alg),
                    matching_alg=_resolve_matching_alg(_matching_alg),
                    pivot_epsilon_alg=_resolve_pivot_epsilon_alg(_pivot_epsilon_alg),
                    debug_base_case=bool(debug_base_case),
                    scaling_max_voltage_change_override=(
                        -1 if scaling_max_voltage_change is None
                        else int(bool(scaling_max_voltage_change))),
                    max_dVa_override=-1.0 if max_dVa is None else float(max_dVa),
                    max_dVm_override=-1.0 if max_dVm is None else float(max_dVm),
                    use_distributed_slack=bool(use_distributed_slack))
                self._inner = _ScenarioSweepSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg)
                self._n_branches = self._inner._s.n_branches
            else:
                d = extract_grid_arrays(grid, max_iter=max_iter_base, tol=tol_base)
                vinit = d["v_converged"] if init_from_n_powerflow else d["v_init"]

                self._inner = _ScenarioSweepSolver(
                    d["Ybus"], vinit, d["Sbus"],
                    d["slack"], d["slack_weights"], d["pv"], d["pq"],
                    batch_size=100, nb_iter=nb_iter,
                    max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                    presolved_v=init_from_n_powerflow,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg,
                    debug_base_case=bool(debug_base_case),
                    scaling_max_voltage_change=bool(scaling_max_voltage_change),
                    max_dVa=0.5 if max_dVa is None else float(max_dVa),
                    max_dVm=0.1 if max_dVm is None else float(max_dVm))

                # Branch admittances come straight from lightsim2grid (never
                # recomputed) so branch-removal topology becomes exact Ybus
                # patches -- always extracted, set_topology() needs it.
                branch_args, _, _ = extract_branch_data(grid)
                self._inner.set_branch_data(*branch_args)
                self._n_branches = len(branch_args[0])

                # The array-based (non-bridge) path has no C++-side grid
                # access, so compute_limit_violations is enabled here in
                # Python instead of inside the C++ bridge call above.
                if compute_limit_violations:
                    self.set_limits_from_grid()
                    self._inner.compute_limit_violations = True

        # Solve the largest connected component of a split grid (masking the
        # rest as NaN) instead of skipping such scenarios. Works on both the
        # bridge and the array path (mutable property on the underlying
        # session).
        self._inner.handle_disconnected_grid = bool(handle_disconnected_grid)

        # Snapshot the load/gen -> solver-bus wiring so set_injections_from_
        # elements() can assemble Sbus itself -- see InjectionSweepGPU's
        # identical rationale.
        if isinstance(grid, (tuple, list)):
            self._elements = None   # no loads/generators to read
        else:
            self._elements = extract_injection_elements(grid, self._inner.n_bus)

        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances (explicit-array mode only).

        Grid mode extracts this automatically; only needed when ``grid`` was
        an explicit-array tuple. Required before :meth:`set_topology` and
        :meth:`compute_flows`.
        """
        self._inner.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                    bus_vn_kv, sn_mva)
        self._n_branches = len(branch_from)

    def set_injections(self, p_mw, q_mvar, sn_mva):
        """Store the (n_scenarios, n_bus) MW / MVAr injection arrays.

        ``p_mw``/``q_mvar`` are *net per-bus* injections in **AC-solver** bus
        numbering.  Prefer :meth:`set_injections_from_elements` when ``grid``
        is a lightsim2grid grid — it takes per-element data and does this
        assembly for you.
        """
        self._inner.set_injections(p_mw, q_mvar, sn_mva)

    def set_injections_from_elements(self, load_p, load_q, gen_p):
        """Store per-element injections, mirroring lightsim2grid's own batch API.

        Same inputs as lightsim2grid's ``TimeSeries.compute_Vs`` /
        :meth:`InjectionSweepGPU.set_injections_from_elements`: one row per
        step (scenario), one column per element. See that method's docstring
        for the full semantics (constant term, disconnected-element handling,
        the "snapshotted at construction" caveat).
        """
        if self._elements is None:
            raise RuntimeError(
                "set_injections_from_elements() needs a lightsim2grid grid; "
                "explicit-array (tuple) mode has no loads/generators to read. "
                "Use set_injections(p_mw, q_mvar, sn_mva) instead.")
        p_mw, q_mvar = build_bus_injections(self._elements,
                                            load_p, load_q, gen_p)
        self._inner.set_injections(p_mw, q_mvar, self._elements.sn_mva)

    def set_gen_v(self, gen_v):
        """Store per-scenario generator target voltage magnitude (vm_pu, NOT kV).

        Mirrors lightsim2grid's ``modify_gen_v``: unlike
        :meth:`set_injections_from_elements`, this does NOT feed Sbus at
        all -- a PV/slack bus's magnitude is never part of Newton-Raphson's
        unknown vector, so it never moves during a solve once seeded. This
        only re-seeds ``|V|`` at each generator's own AC-solver bus,
        immediately before that scenario's solve, keeping whatever angle is
        already seeded there.

        Parameters
        ----------
        gen_v : (n_scenarios, n_gens) — target vm_pu per generator,
            row-aligned with :meth:`set_injections_from_elements` /
            :meth:`set_injections` / :meth:`set_topology`. NaN leaves that
            (scenario, generator) untouched. Only applied to generators
            whose OWN bus is voltage-fixed (PV or slack) in this session's
            base case -- a disconnected, reactive-only ("PQ"), or remotely
            voltage-regulating (SVC / VoltageControl) generator's column is
            silently ignored, mirroring lightsim2grid's own
            ``voltage_regulator_on_``-gated behavior. Left unset entirely
            (the default), every scenario keeps the grid's own base-case
            voltage.

        Notes
        -----
        The generator -> bus wiring is snapshotted at construction, like
        :meth:`set_injections_from_elements` -- mutating the grid
        afterwards does not propagate.
        """
        if self._elements is None:
            raise RuntimeError(
                "set_gen_v() needs a lightsim2grid grid; explicit-array "
                "(tuple) mode has no generators to read.")
        self._inner.set_gen_v(gen_v, self._elements.gen_bus)

    def set_topology(self, branch_ids_per_scenario):
        """Define each scenario's topology as branch removals.

        Parameters
        ----------
        branch_ids_per_scenario : list[list[int]]
            One inner list per scenario, holding the branch indices to trip
            in that scenario, row-aligned with :meth:`set_injections` /
            :meth:`set_injections_from_elements`. Indices are 0-based, lines
            first then trafos (``c < n_lines`` is line ``c``; ``c >= n_lines``
            is trafo ``c - n_lines``). An empty inner list means "nothing
            tripped for this row".

        Notes
        -----
        Requires :meth:`set_branch_data` (automatic in grid mode) to have
        been called first. Optional: if never called, :meth:`compute`
        defaults every scenario to "no branches tripped".
        """
        self._inner.set_topology(branch_ids_per_scenario)

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

    def get_disconnected(self):
        """Per-scenario disconnected flag: (n_scenarios,) int array, 1 == the
        topology change islanded the grid (scenario skipped/NaN), 0 == solved.
        Must be called after :meth:`compute`."""
        return self._inner.get_disconnected()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute().

        Each scenario's own tripped branches are reported as exactly zero.
        """
        self._inner.compute_flows()

    # ------------------------------------------------- compute_limit_violations
    def _extract_limits_arrays(self):
        """(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines) off
        self._grid: bulk C++ extraction (bus arrays relabeled to AC-solver
        numbering) when the bridge is available, else the pure-Python
        fallback iterating grid.get_lines()/get_trafos(). NaN = not
        configured. Shared by set_limits_from_grid() and get_violations_n()."""
        if self._grid is None:
            raise RuntimeError(
                "requires a lightsim2grid grid (this session was built from "
                "an explicit-array tuple); use set_limits() instead.")
        grid = self._grid
        n_lines = len(grid.get_lines())
        n_bus_solver = self._inner._s.n_bus
        if _have_bridge():
            bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka = \
                _cpp._extract_limits_from_lsgrid(grid, n_bus_solver)
        else:
            me_to_solver = grid.id_me_to_ac_solver()
            bus_vmin_model = grid.get_bus_vmin_kv()
            bus_vmax_model = grid.get_bus_vmax_kv()
            bus_vmin_kv = np.full(n_bus_solver, np.nan)
            bus_vmax_kv = np.full(n_bus_solver, np.nan)
            if len(bus_vmin_model) > 0:
                for grid_id, solver_id in enumerate(me_to_solver):
                    if solver_id >= 0:
                        bus_vmin_kv[solver_id] = bus_vmin_model[grid_id]
                        bus_vmax_kv[solver_id] = bus_vmax_model[grid_id]
            limit_a1_ka = np.array([l.limit_a1_ka for l in grid.get_lines()] +
                                    [t.limit_a1_ka for t in grid.get_trafos()])
            limit_a2_ka = np.array([l.limit_a2_ka for l in grid.get_lines()] +
                                    [t.limit_a2_ka for t in grid.get_trafos()])
        return bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines

    def set_limits_from_grid(self):
        """Extract bus voltage (kV) / branch current (kA) limits straight off
        the lightsim2grid grid and configure them via the underlying solver's
        ``set_limits()``.

        Bus arrays are relabeled from grid-model to AC-solver bus numbering
        (same map ``set_branch_data`` uses for branch endpoints); branch
        arrays are lines-then-trafos. NaN = not configured for that element.

        Not needed when ``compute_limit_violations=True`` was passed to the
        constructor and the bridge path is in use (the C++ bridge already
        does this internally) -- call this when limits changed on the grid
        after construction, or when using the array (non-bridge) path.
        """
        bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines = \
            self._extract_limits_arrays()
        self._inner.set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)

    @property
    def compute_limit_violations(self):
        """bool: fused per-chunk voltage/current/divergence check (see
        set_limits_from_grid()). Default False. Takes effect on the next
        compute()."""
        return self._inner.compute_limit_violations

    @compute_limit_violations.setter
    def compute_limit_violations(self, value):
        self._inner.compute_limit_violations = value

    def get_violations(self):
        """list[list[LimitViolation]]: one entry per scenario (row order
        matches set_injections()/set_topology()'s input rows). Requires
        compute() with compute_limit_violations=True."""
        return self._inner.get_violations()

    def get_violations_truncated(self):
        """(n_scenarios,) bool ndarray: True where more than violation_capacity
        violations were found for that scenario (clamped)."""
        return self._inner.get_violations_truncated()

    def get_violation_counts(self):
        """dict of (n_scenarios,) int ndarrays with keys 'low_voltage',
        'high_voltage', 'current': the TRUE, uncapped count of violations of
        each type per scenario (-1 = not simulated). Unlike
        get_violations()'s records (capped at violation_capacity), these
        totals stay exact even when get_violations_truncated() is True."""
        return self._inner.get_violation_counts()

    def converged(self, tol=None):
        """(n_scenarios,) bool ndarray: residual <= tol (defaults to violation_tol).
        Independent of compute_limit_violations -- requires compute()."""
        return self._inner.converged(tol)

    def converged_n(self, tol=None):
        """bool: whether the pre-scenario ('n') CPU base-case solve
        converged. Construction already validates this against tol_base
        (raising RuntimeError otherwise), so this always returns True --
        exists for API parity with lightsim2grid's converged_n(). tol is
        accepted for signature symmetry with converged() but unused."""
        return True

    def get_violations_n(self):
        """list[LimitViolation] for the pre-scenario ('n') case: a single
        voltage vector, not a batch, so this is pure Python/CPU (no GPU,
        no memory/transfer concern -- see
        gpusim2grid.contingency_analysis._limit_violations.compute_violations_n).
        Requires set_limits_from_grid() (or a compute_limit_violations=True
        construction) to have been called; with no limits configured,
        returns []."""
        from ..contingency_analysis._limit_violations import compute_violations_n

        grid = self._grid
        bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines = \
            self._extract_limits_arrays()
        if np.all(np.isnan(bus_vmin_kv)) and np.all(np.isnan(bus_vmax_kv)):
            bus_vmin_kv = bus_vmax_kv = None
        if np.all(np.isnan(limit_a1_ka)) and np.all(np.isnan(limit_a2_ka)):
            limit_a1_ka = limit_a2_ka = None

        branch_args, _, _ = extract_branch_data(grid)
        branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, sn_mva = branch_args
        V_n = grid.get_V_solver()

        return compute_violations_n(
            V_n, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
            branch_from, branch_to, yff, yft, ytf, ytt,
            limit_a1_ka, limit_a2_ka, sn_mva, n_lines)

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
    def n_loads(self):
        """Column count :meth:`set_injections_from_elements` expects for load_p/q."""
        if self._elements is None:
            raise RuntimeError(
                "explicit-array (tuple) mode has no loads; n_loads is undefined.")
        return self._elements.n_load

    @property
    def n_gens(self):
        """Column count :meth:`set_injections_from_elements` expects for gen_p."""
        if self._elements is None:
            raise RuntimeError(
                "explicit-array (tuple) mode has no generators; n_gens is "
                "undefined.")
        return self._elements.n_gen

    @property
    def solver(self):
        """The underlying :class:`_ScenarioSweepSolver` (escape hatch)."""
        return self._inner

    # --------------------------------------------------------------- factory
    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)
