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

Deferred scope (matches lightsim2grid's own ScenarioSweepCPP v1): no
handle_disconnected_grid masking, no compute_limit_violations. A scenario
whose topology change disconnects the grid is skipped (NaN).
"""

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
                 use_bridge=None, reordering_alg=None,
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
            # No grid to auto-extract branch data from: call set_branch_data()
            # manually before set_topology()/compute_flows().
            self._n_branches = 0
        else:
            if use_bridge is None:
                use_bridge = _have_bridge()

            if use_bridge:
                # Branch data is always set -- set_topology() needs it, not
                # just compute_flows() -- see ls2g_bridge.cpp:make_ss_session_from_lsgrid.
                session = _cpp._make_ss_session_from_lsgrid(
                    grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                    int(max_iter_base), float(tol_base), _normalize_device(device),
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
