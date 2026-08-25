# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
gpusim2grid.scenario_sweep — Python facade for the row-aligned combined
topology + injection sweep.

Mirrors gpusim2grid.injection_sweep's structure: a thin `_ScenarioSweepSolver`
engine class wrapping the C++ `ScenarioSweepSession` binding, reusing the
same `'string' -> enum` maps and device/DeviceBuffer helpers as
gpusim2grid.contingency_analysis / gpusim2grid.injection_sweep (single source
of truth, not re-duplicated here).

Row `i` of the injection matrices (``set_injections`` / ``set_injections_from_
elements``) is solved together with row `i` of ``set_topology``'s branch-trip
lists, independently of every other row -- matching lightsim2grid's own
``ScenarioSweep`` (see CLAUDE.md's Augmented Jacobian section and the
"ScenarioSweepGPU" workload table).

``handle_disconnected_grid`` and ``compute_limit_violations`` are supported
identically to ``ContingencyAnalysisGPU`` -- see that class's docs for the
full semantics; this module reuses the same ``ViolationElementType`` /
``LimitViolationType`` / ``LimitViolation`` types (defined once in
``gpusim2grid.contingency_analysis._limit_violations``, not re-duplicated
here).
"""

__all__ = []

import numpy as np

from .._gpusim2grid import (
    ContingencySolverType as _ContingencySolverType,
    ScenarioSweepSession as _ScenarioSweepSession,
)

# Shared with contingency_analysis/__init__.py (single source of truth for
# the reordering-alg string<->enum map and the device normalizer), same
# pattern injection_sweep already follows.
from ..contingency_analysis import (
    _resolve_reordering_alg,
    _resolve_matching_alg,
    _resolve_pivot_epsilon_alg,
)
from ..contingency_analysis._limit_violations import (
    ViolationElementType,
    LimitViolationType,
    LimitViolation,
)
from ..injection_sweep import _normalize_device, _DeviceBuffer


_STRATEGY_MAP = {
    'direct_refactor_every':    _ContingencySolverType.DirectRefactorEvery,
    'direct_base_case_factors': _ContingencySolverType.DirectBaseCaseFactors,
    'direct_iter0_only':        _ContingencySolverType.DirectIter0Only,
    'direct_refactor_every_n':  _ContingencySolverType.DirectRefactorEveryN,
}


def _resolve_strategy(strategy):
    """Accept either a string or a raw ContingencySolverType value."""
    if isinstance(strategy, _ContingencySolverType):
        return strategy
    if isinstance(strategy, str):
        try:
            return _STRATEGY_MAP[strategy]
        except KeyError:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from: {list(_STRATEGY_MAP)}"
            )
    raise TypeError(
        f"strategy must be a str or ContingencySolverType, got {type(strategy).__name__}"
    )


class _ScenarioSweepSolver:
    """Stateful GPU row-aligned combined topology + injection sweep.

    The base-case Newton-Raphson is solved once at construction; subsequent
    ``set_injections()`` / ``set_topology()`` + ``run()`` calls reuse that
    base case, so a sweep that re-runs with different injections/topology --
    or a different ``batch_size`` / ``nb_iter`` / ``strategy`` -- does not
    re-solve the base case.

    Parameters
    ----------
    Ybus : scipy.sparse complex (n_bus, n_bus)
    Vinit : (n_bus,) complex128 — base-case warm-start voltages.
    Sbus : (n_bus,) complex128 — per-unit injections for the base-case NR.
    slack_ids, slack_weights, pv, pq : index/weight arrays (as in the other solvers).
    batch_size : int, optional — scenarios per GPU chunk (default 100).
    nb_iter : int, optional — fixed NR iterations per chunk (default 4).
    max_iter_base, tol_base : base-case NR convergence settings.

    Notes
    -----
    The following properties are **mutable** and take effect on the next
    :meth:`run` call: ``batch_size``, ``nb_iter``, ``strategy``,
    ``refactor_period``, ``reordering_alg``, ``matching_alg``,
    ``pivot_epsilon_alg``.

    Examples
    --------
    .. code-block:: python

        solver = _ScenarioSweepSolver(Ybus, Vinit, Sbus,
                                      slack_ids, slack_weights, pv, pq)
        solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                               bus_vn_kv, sn_mva)
        solver.set_injections(p_mw, q_mvar, sn_mva)     # (n_scen, n_bus) MW / MVAr
        solver.set_topology([[3], [], [3, 40]])         # one branch-id list per row
        solver.run()
        V   = solver.V_results.to_numpy()               # lazy D->H
        res = solver.residuals.to_numpy()
    """

    def __init__(self, Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                 batch_size=100, nb_iter=4, max_iter_base=10, tol_base=1e-6,
                 device=None, handle_disconnected_grid=False, presolved_v=False,
                 reordering_alg='default',
                 matching_alg='none', pivot_epsilon_alg='default',
                 debug_base_case=False,
                 scaling_max_voltage_change=False, max_dVa=0.5, max_dVm=0.1):
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = 'direct_refactor_every'
        self._reordering_alg = reordering_alg
        self._matching_alg = matching_alg
        self._pivot_epsilon_alg = pivot_epsilon_alg
        self._s = _ScenarioSweepSession(
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
            int(batch_size), int(nb_iter), self._max_iter_base, self._tol_base,
            _normalize_device(device), presolved_v=bool(presolved_v),
            reordering_alg=_resolve_reordering_alg(reordering_alg),
            matching_alg=_resolve_matching_alg(matching_alg),
            pivot_epsilon_alg=_resolve_pivot_epsilon_alg(pivot_epsilon_alg),
            debug_base_case=bool(debug_base_case),
            scaling_max_voltage_change=bool(scaling_max_voltage_change),
            max_dVa=float(max_dVa), max_dVm=float(max_dVm))
        self._s.handle_disconnected_grid = bool(handle_disconnected_grid)

    @classmethod
    def _wrap_session(cls, session, max_iter_base=1, tol_base=1e-6,
                      strategy='direct_refactor_every', reordering_alg='default',
                      matching_alg='none', pivot_epsilon_alg='default'):
        """Wrap an already-constructed C++ ScenarioSweepSession (zero-copy
        lightsim2grid bridge). Reuses all wrapper ergonomics."""
        self = cls.__new__(cls)
        self._s = session
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = strategy
        self._reordering_alg = reordering_alg
        self._matching_alg = matching_alg
        self._pivot_epsilon_alg = pivot_epsilon_alg
        return self

    # --- mutable config ---
    @property
    def batch_size(self):
        return self._s.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._s.batch_size = int(value)

    @property
    def nb_iter(self):
        return self._s.nb_iter

    @nb_iter.setter
    def nb_iter(self, value):
        self._s.nb_iter = int(value)

    @property
    def scaling_max_voltage_change(self):
        """NR step-scaling (mirrors lightsim2grid's own
        MaxVoltageChangeScalingPolicy); bool, takes effect on the next run().
        Each batch slot gets its own alpha from its own max|dtheta|/max|dvm|.
        """
        return self._s.scaling_max_voltage_change

    @scaling_max_voltage_change.setter
    def scaling_max_voltage_change(self, value):
        self._s.scaling_max_voltage_change = bool(value)

    @property
    def max_dVa(self):
        """MaxVoltageChangeScalingPolicy max angle step (rad); only
        meaningful with scaling_max_voltage_change=True."""
        return self._s.max_dVa

    @max_dVa.setter
    def max_dVa(self, value):
        self._s.max_dVa = float(value)

    @property
    def max_dVm(self):
        """MaxVoltageChangeScalingPolicy max voltage-magnitude step (pu);
        only meaningful with scaling_max_voltage_change=True."""
        return self._s.max_dVm

    @max_dVm.setter
    def max_dVm(self, value):
        self._s.max_dVm = float(value)

    @property
    def strategy(self):
        """Linear-solve strategy (str). Takes effect on the next run()."""
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._s.strategy_type = _resolve_strategy(value)
        self._strategy = value if isinstance(value, str) else str(value)

    @property
    def reordering_alg(self):
        """CUDSS_CONFIG_REORDERING_ALG choice (str or ReorderingAlg). Takes
        effect on the next run() (which always reruns cuDSS ANALYSIS).

        NOTE: cuDSS rejects 'btf_colamd'/'colamd' with CUDSS_STATUS_NOT_SUPPORTED
        in this session's uniform-batch mode -- only 'default', 'amd',
        'nested_dissection', 'none' are supported here. 'btf_colamd'/'colamd'
        work only on AcPfGPU's single-system solve."""
        return self._reordering_alg

    @reordering_alg.setter
    def reordering_alg(self, value):
        self._s.reordering_alg = _resolve_reordering_alg(value)
        self._reordering_alg = value if isinstance(value, str) else str(value)

    @property
    def matching_alg(self):
        """CUDSS_CONFIG_MATCHING_ALG choice (str or MatchingAlg). Takes effect
        on the next run() (which always reruns cuDSS ANALYSIS).

        NOTE: 'none' (default) is the only value cuDSS accepts in this
        session's uniform-batch mode -- every other value raises RuntimeError
        (CUDSS_STATUS_NOT_SUPPORTED). They only work on AcPfGPU's
        single-system solve, and even there 'max_diag_product'/'auto' have
        been observed to silently produce NaN voltages."""
        return self._matching_alg

    @matching_alg.setter
    def matching_alg(self, value):
        self._s.matching_alg = _resolve_matching_alg(value)
        self._matching_alg = value if isinstance(value, str) else str(value)

    @property
    def pivot_epsilon_alg(self):
        """CUDSS_CONFIG_PIVOT_EPSILON_ALG choice (str or PivotEpsilonAlg).
        Takes effect on the next run() (which always reruns cuDSS ANALYSIS).
        One of 'default' (default), 'scaled', 'static'."""
        return self._pivot_epsilon_alg

    @pivot_epsilon_alg.setter
    def pivot_epsilon_alg(self, value):
        self._s.pivot_epsilon_alg = _resolve_pivot_epsilon_alg(value)
        self._pivot_epsilon_alg = value if isinstance(value, str) else str(value)

    @property
    def handle_disconnected_grid(self):
        """bool: solve the largest connected component of a scenario's split
        grid (masking the rest as NaN) instead of skipping it. Scenarios that
        strand the angle reference or a controller bus are still skipped.
        Incompatible with the 'direct_base_case_factors' strategy. Takes
        effect on the next run()."""
        return self._s.handle_disconnected_grid

    @handle_disconnected_grid.setter
    def handle_disconnected_grid(self, value):
        self._s.handle_disconnected_grid = bool(value)

    @property
    def refactor_period(self):
        return self._s.refactor_period

    @refactor_period.setter
    def refactor_period(self, value):
        self._s.refactor_period = int(value)

    @property
    def used_batch_size(self):
        return self._s.used_batch_size

    # --- branch data (needed for set_topology AND compute_flows) ---
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances. Required before set_topology()
        and before compute_flows()."""
        self._s.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                bus_vn_kv, sn_mva)

    # --- inputs ---
    def set_injections(self, p_mw, q_mvar, sn_mva):
        """Store the (n_scenarios, n_bus) MW / MVAr injection arrays.

        Converted to per-unit complex Sbus on the next ``run()``. Fixes
        n_scenarios. May be called repeatedly.
        """
        p = np.ascontiguousarray(p_mw, dtype=np.float64)
        q = np.ascontiguousarray(q_mvar, dtype=np.float64)
        self._s.set_injections(p, q, float(sn_mva))

    def set_gen_v(self, gen_v, gen_bus):
        """Per-scenario generator target voltage magnitude (vm_pu, NOT kV),
        (n_scenarios, n_gen). Does NOT feed Sbus -- see the C++
        ScenarioSweepSession.set_gen_v binding for the full semantics.
        gen_bus: (n_gen,) AC-solver bus id per generator, -1 for a
        disconnected one.
        """
        v = np.ascontiguousarray(gen_v, dtype=np.float64)
        b = np.ascontiguousarray(gen_bus, dtype=np.int32)
        self._s.set_gen_v(v, b)

    def set_topology(self, branch_ids_per_scenario):
        """Build topology from a list-of-lists of branch indices, row-aligned
        with set_injections().

        Each inner list is the set of branch IDs (lines-then-trafos) to trip
        in that scenario. Requires set_branch_data() to have been called
        first. Optional: if never called, run() defaults every scenario to
        "no branches tripped" (a plain injection sweep).
        """
        self._s.set_topology(branch_ids_per_scenario)

    def run(self):
        """Run all scenarios; fills V_results and residuals on device."""
        self._s.run()

    def compute_flows(self):
        """Compute branch flows for all scenarios from stored V_results.

        Requires run() and set_branch_data() to have been called.
        Results are then available via solver.or_amps and solver.ex_amps.
        """
        self._s.compute_flows()

    @property
    def compute_limit_violations(self):
        """bool: fused per-chunk voltage/current/divergence check (see
        set_limits()). Mirrors ContingencyAnalysisGPU's flag of the same
        name. Writes only a bounded compact buffer -- never the full dense
        V_results/or_amps/ex_amps. Changing this clears any previously
        computed violation results. Default False. Takes effect on the next
        run()."""
        return self._s.compute_limit_violations

    @compute_limit_violations.setter
    def compute_limit_violations(self, value):
        self._s.compute_limit_violations = bool(value)   # C++ setter handles the clear-on-change

    def set_limits(self, bus_vmin_kv, bus_vmax_kv, branch_limit_a1_ka, branch_limit_a2_ka, n_lines):
        """Configure per-bus voltage (kV) / per-branch current (kA) limits.

        NaN = not configured for that element (matches lightsim2grid's
        convention). Required before run() when compute_limit_violations is
        True. n_lines splits the lines-then-trafos branch ordering for
        LimitViolation.element_type/element_id de-concatenation.
        """
        self._s.set_limits(bus_vmin_kv, bus_vmax_kv, branch_limit_a1_ka,
                           branch_limit_a2_ka, int(n_lines))

    @property
    def violation_tol(self):
        """float: residual tolerance for the DIVERGENCE check, independent of
        tol_base. Takes effect on the next run()."""
        return self._s.violation_tol

    @violation_tol.setter
    def violation_tol(self, value):
        self._s.violation_tol = float(value)

    @property
    def violation_capacity(self):
        """int: max violation records kept per scenario (K). Bounds the
        compact output at n_scenarios * K regardless of grid size. Takes
        effect on the next run(); default 16."""
        return self._s.violation_capacity

    @violation_capacity.setter
    def violation_capacity(self, value):
        self._s.violation_capacity = int(value)

    def get_violations(self):
        """list[list[LimitViolation]]: one entry per scenario (row order
        matches set_injections()/set_topology()'s input rows). A
        not-simulated (disconnected / masked-skip) scenario gets a single
        GRID/NOT_SIMULATED entry (value=limit=nan -- the solver was never
        invoked, there is no residual to report); a non-converged one gets a
        single GRID/DIVERGENCE entry instead (value=residual, limit=tol).
        Requires run() with compute_limit_violations=True."""
        if not self.compute_limit_violations:
            raise RuntimeError(
                "get_violations() requires compute_limit_violations=True "
                "(set solver.compute_limit_violations = True before run()).")
        counts = self._s.get_violation_count()
        etype  = self._s.get_violation_element_type()
        eid    = self._s.get_violation_element_id()
        side   = self._s.get_violation_side()
        vtype  = self._s.get_violation_type()
        value  = self._s.get_violation_value()
        limit  = self._s.get_violation_limit()
        K = self.violation_capacity
        out = []
        for c, cnt in enumerate(counts):
            if cnt < 0:
                # Pre-check (graph connectivity) dropped this scenario before
                # it ever reached check_limit_violations_kernel -- the solver
                # was never invoked (BatchPfDriver's d_violation_count -1
                # sentinel). Never written by the kernel itself.
                out.append([LimitViolation(ViolationElementType.GRID, -1, 0,
                                            LimitViolationType.NOT_SIMULATED,
                                            float('nan'), float('nan'))])
                continue
            base = c * K
            out.append([
                LimitViolation(ViolationElementType(int(etype[base + i])), int(eid[base + i]),
                               int(side[base + i]), LimitViolationType(int(vtype[base + i])),
                               float(value[base + i]), float(limit[base + i]))
                for i in range(cnt)
            ])
        return out

    def get_violations_truncated(self):
        """(n_scenarios,) bool ndarray: True where more than
        violation_capacity violations were found for that scenario
        (clamped -- raise violation_capacity if this matters for your use
        case). Requires run() with compute_limit_violations=True."""
        return self._s.get_violation_truncated().astype(bool)

    def get_violation_counts(self):
        """dict of (n_scenarios,) int ndarrays with keys 'low_voltage',
        'high_voltage', 'current': the TRUE, uncapped count of violations of
        each type per scenario, -1 for a not-simulated (disconnected /
        masked-skip) scenario.

        Unlike get_violations()'s per-violation records (capped at
        violation_capacity), these totals are always exact -- they keep
        counting past the cap, so they remain reliable even when
        get_violations_truncated() is True for a scenario. Requires run()
        with compute_limit_violations=True."""
        if not self.compute_limit_violations:
            raise RuntimeError(
                "get_violation_counts() requires compute_limit_violations=True "
                "(set solver.compute_limit_violations = True before run()).")
        return {
            "low_voltage":  self._s.get_violation_count_low_voltage(),
            "high_voltage": self._s.get_violation_count_high_voltage(),
            "current":      self._s.get_violation_count_current(),
        }

    def converged(self, tol=None):
        """(n_scenarios,) bool ndarray: residual <= tol (defaults to
        violation_tol). Independent of compute_limit_violations -- residuals
        are always O(n_scenarios) and always computed (see residuals);
        this does not require compute_limit_violations to be enabled."""
        t = self.violation_tol if tol is None else tol
        return self.residuals.to_numpy() <= t

    # --- results ---
    @property
    def V_results(self):
        """DeviceBuffer: (n_scenarios * n_bus,) complex voltages (per-unit)."""
        n_scen = self._s.n_scenarios
        n_bus = self._s.n_bus
        return _DeviceBuffer(self._s.get_V_results, (n_scen * n_bus,), 'complex128')

    @property
    def residuals(self):
        """DeviceBuffer: (n_scenarios,) ‖F‖∞ residuals."""
        return _DeviceBuffer(self._s.get_residuals, (self._s.n_scenarios,), 'float64')

    @property
    def or_amps(self):
        """DeviceBuffer: (n_scenarios * n_branches,) origin terminal amps.

        Available after compute_flows() has been called.
        """
        n_scen = self._s.n_scenarios
        n_bra  = self._s.n_branches
        return _DeviceBuffer(self._s.get_or_amps, (n_scen * n_bra,), 'float64')

    @property
    def ex_amps(self):
        """DeviceBuffer: (n_scenarios * n_branches,) extremity terminal amps.

        Available after compute_flows() has been called.
        """
        n_scen = self._s.n_scenarios
        n_bra  = self._s.n_branches
        return _DeviceBuffer(self._s.get_ex_amps, (n_scen * n_bra,), 'float64')

    @property
    def n_branches(self):
        """Number of branches (lines + trafos). Available after set_branch_data()."""
        return self._s.n_branches

    @property
    def timings(self):
        """BatchTimings from the most recent run() / compute_flows()."""
        return self._s.get_timings()

    def get_disconnected(self):
        """Per-scenario disconnected flag: (n_scenarios,) int array, 1 == the
        topology change islanded the grid (scenario skipped/NaN), 0 == solved.
        Empty before run() has been called."""
        return self._s.get_disconnected()

    def v_base_dlpack(self):
        """Zero-copy DLPack capsule of base-case voltages, shape [n_bus].

        Pass to ``torch.from_dlpack()`` or ``jax.dlpack.from_dlpack()`` for a
        zero-copy GPU tensor.  The solver must remain alive while the tensor
        is in use.
        """
        return self._s.v_base_dlpack()

    def v_results_dlpack(self):
        """Zero-copy DLPack capsule of batch voltages, shape [n_scenarios, n_bus].

        Requires ``run()`` to have been called.  The capsule aliases live GPU
        memory — calling ``run()`` again overwrites it in place.  Clone before
        a subsequent ``run()`` if a snapshot is needed.
        """
        return self._s.v_results_dlpack()

    @property
    def n_scenarios(self):
        return self._s.n_scenarios

    @property
    def n_bus(self):
        return self._s.n_bus


# Not in __all__: documented separately as gpusim2grid.ScenarioSweepGPU (see
# docs/api.rst) to avoid duplicate autodoc entries for the same class.
from .gpu_facade import ScenarioSweepGPU
