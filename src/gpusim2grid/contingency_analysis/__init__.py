# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

__all__ = [
    "ViolationElementType",
    "LimitViolationType",
    "LimitViolation",
]

from .._gpusim2grid import (
    ContingencyAnalysisSession as _ContingencyAnalysisSession,
    ContingencySolverType as _ContingencySolverType,
    ReorderingAlg as _ReorderingAlg,
    MatchingAlg as _MatchingAlg,
    PivotEpsilonAlg as _PivotEpsilonAlg,
)

from ._limit_violations import ViolationElementType, LimitViolationType, LimitViolation

def _normalize_device(device):
    """Normalize a device specifier to an int for the C++ ctor.

    None  → -1  (use current device)
    int   → itself
    "cuda"     → 0
    "cuda:N"   → N
    """
    if device is None:
        return -1
    if isinstance(device, int):
        return device
    if device == "cuda":
        return 0
    if isinstance(device, str) and device.startswith("cuda:"):
        return int(device[5:])
    raise ValueError(
        f"Unrecognized device specifier {device!r}. "
        "Expected None, int, 'cuda', or 'cuda:N'.")


_STRATEGY_MAP = {
    'direct_refactor_every':    _ContingencySolverType.DirectRefactorEvery,
    'direct_base_case_factors': _ContingencySolverType.DirectBaseCaseFactors,
    'direct_iter0_only':        _ContingencySolverType.DirectIter0Only,
    'direct_refactor_every_n':  _ContingencySolverType.DirectRefactorEveryN,
}


# Shared with injection_sweep/__init__.py and acpf_nr/gpu_facade.py (imported
# from here, same as _normalize_device) rather than duplicated per-module like
# _STRATEGY_MAP above: CUDSS_CONFIG_REORDERING_ALG is a solver-wide concept,
# not specific to the batch/contingency workload.
_REORDERING_ALG_MAP = {
    'default':           _ReorderingAlg.Default,
    'btf_colamd':        _ReorderingAlg.BtfColamd,
    'colamd':            _ReorderingAlg.Colamd,
    'amd':               _ReorderingAlg.Amd,
    'nested_dissection': _ReorderingAlg.NestedDissection,
    'none':              _ReorderingAlg.NoReordering,
}


def _resolve_reordering_alg(value):
    """Accept either a string (mapped via _REORDERING_ALG_MAP) or a raw
    ReorderingAlg value."""
    if isinstance(value, _ReorderingAlg):
        return value
    if isinstance(value, str):
        try:
            return _REORDERING_ALG_MAP[value]
        except KeyError:
            raise ValueError(
                f"Unknown reordering_alg {value!r}. "
                f"Choose from: {list(_REORDERING_ALG_MAP)}")
    raise TypeError(
        f"reordering_alg must be a str or ReorderingAlg, got {type(value).__name__}")


# Shared with injection_sweep/__init__.py and acpf_nr/gpu_facade.py, same
# rationale as _REORDERING_ALG_MAP above: CUDSS_CONFIG_MATCHING_ALG is also a
# solver-wide concept, not specific to the batch/contingency workload.
_MATCHING_ALG_MAP = {
    'none':              _MatchingAlg.NoMatching,
    'max_diag_count':    _MatchingAlg.MaxDiagCount,
    'max_min_diag':      _MatchingAlg.MaxMinDiag,
    'max_min_diag_alt':  _MatchingAlg.MaxMinDiagAlt,
    'max_diag_sum':      _MatchingAlg.MaxDiagSum,
    'max_diag_product':  _MatchingAlg.MaxDiagProduct,
    'auto':              _MatchingAlg.Auto,
}


def _resolve_matching_alg(value):
    """Accept either a string (mapped via _MATCHING_ALG_MAP) or a raw
    MatchingAlg value."""
    if isinstance(value, _MatchingAlg):
        return value
    if isinstance(value, str):
        try:
            return _MATCHING_ALG_MAP[value]
        except KeyError:
            raise ValueError(
                f"Unknown matching_alg {value!r}. "
                f"Choose from: {list(_MATCHING_ALG_MAP)}")
    raise TypeError(
        f"matching_alg must be a str or MatchingAlg, got {type(value).__name__}")


# Shared with injection_sweep/__init__.py and acpf_nr/gpu_facade.py, same
# rationale as _REORDERING_ALG_MAP/_MATCHING_ALG_MAP above:
# CUDSS_CONFIG_PIVOT_EPSILON_ALG is also a solver-wide concept, not specific
# to the batch/contingency workload.
_PIVOT_EPSILON_ALG_MAP = {
    'default': _PivotEpsilonAlg.Default,
    'scaled':  _PivotEpsilonAlg.Scaled,
    'static':  _PivotEpsilonAlg.Static,
}


def _resolve_pivot_epsilon_alg(value):
    """Accept either a string (mapped via _PIVOT_EPSILON_ALG_MAP) or a raw
    PivotEpsilonAlg value."""
    if isinstance(value, _PivotEpsilonAlg):
        return value
    if isinstance(value, str):
        try:
            return _PIVOT_EPSILON_ALG_MAP[value]
        except KeyError:
            raise ValueError(
                f"Unknown pivot_epsilon_alg {value!r}. "
                f"Choose from: {list(_PIVOT_EPSILON_ALG_MAP)}")
    raise TypeError(
        f"pivot_epsilon_alg must be a str or PivotEpsilonAlg, got "
        f"{type(value).__name__}")


class DeviceBuffer:
    """Handle to GPU-resident data; transfers to host on demand via to_numpy()."""

    def __init__(self, fetch_fn, shape, dtype):
        self._fetch = fetch_fn
        self._shape = shape
        self._dtype = dtype

    def to_numpy(self):
        """Copy data from device to host and return a numpy array."""
        return self._fetch()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"DeviceBuffer(shape={self._shape}, dtype={self._dtype!r})"


class _ContingencyAnalysisSolver:
    """Stateful GPU N-k contingency analysis solver.

    Parameters
    ----------
    Ybus : scipy.sparse complex matrix (n_bus × n_bus)
        Base-case admittance matrix.
    Vinit : (n_bus,) complex128
        Initial voltage vector (flat-start or warm-start).
    Sbus : (n_bus,) complex128
        Scheduled complex power injections.
    slack_ids : (n_slack,) int32
        Slack bus indices.
    slack_weights : (n_slack,) float64
        Slack bus weights.
    pv : (n_pv,) int32
        PV bus indices.
    pq : (n_pq,) int32
        PQ bus indices.
    batch_size : int, optional
        Contingencies processed per GPU chunk (default: 100).
        Can be changed at any time via ``solver.batch_size = N``.
    nb_iter : int, optional
        Fixed NR iterations per chunk — no convergence check mid-loop
        (default: 4).  3–5 is typically sufficient.
        Can be changed at any time via ``solver.nb_iter = N``.
    max_iter_base : int, optional
        Max NR iterations for the base-case solve (default: 10).
        Only used at construction time.
    tol_base : float, optional
        Convergence tolerance ‖F‖∞ for the base-case solve (default: 1e-6).
        Only used at construction time.
    presolved_v : bool, optional
        If True, ``Vinit`` is trusted as already converged (e.g. lightsim2grid's
        CPU solve): the GPU base-case NR loop is skipped entirely after a single
        validation ``‖F(Vinit)‖∞`` check (raises ``RuntimeError`` if it fails).
        ``max_iter_base`` is then only used for reference/introspection, not to
        drive an iteration count. Default False.
    reordering_alg, matching_alg, pivot_epsilon_alg : str, optional
        cuDSS config, applied ONCE at construction to BOTH the base-case solve
        above and the batch solver's own copy (consumed by :meth:`run`) --
        single source of truth. The mutable properties of the same name only
        ever affect the latter afterward (the base-case solve is fixed once
        built). Defaults ``'default'``, ``'none'``, ``'default'``.
    debug_base_case : bool, optional
        Only meaningful with ``presolved_v=True`` and a MultiSlack/
        VoltageControl extension active (bridge/lightsim2grid ledger). Forces
        the pre-ground-truth cuDSS-solve derivation of the extension's running
        state for the base-case solve, even when lightsim2grid's own converged
        values are available -- an opt-in diagnostic (e.g. to keep testing
        cuDSS config choices in isolation). Default False.

    Notes
    -----
    The following properties are **mutable** and take effect on the next
    :meth:`run` call:

    - ``batch_size`` (*int*): Contingencies per GPU chunk.
    - ``nb_iter`` (*int*): Fixed NR iterations per chunk.
    - ``strategy`` (*str*): Linear-solve strategy.  One of
      ``'direct_refactor_every'`` (default), ``'direct_base_case_factors'``,
      ``'direct_iter0_only'``, ``'direct_refactor_every_n'``.
    - ``refactor_period`` (*int*): Period N for ``'direct_refactor_every_n'``
      (default 1, equivalent to ``'direct_refactor_every'``).
    - ``reordering_alg`` (*str*): cuDSS ``CUDSS_CONFIG_REORDERING_ALG`` choice.
      One of ``'default'`` (default), ``'amd'``, ``'nested_dissection'``,
      ``'none'``. ``'btf_colamd'``/``'colamd'`` are also accepted but cuDSS
      rejects them (``CUDSS_STATUS_NOT_SUPPORTED``) in this session's
      uniform-batch mode; they only work on ``AcPfGPU``'s single-system solve.
    - ``matching_alg`` (*str*): cuDSS ``CUDSS_CONFIG_MATCHING_ALG`` choice.
      ``'none'`` (default) is the only value cuDSS accepts in this session's
      uniform-batch mode -- every other value (``'max_diag_count'``,
      ``'max_min_diag'``, ``'max_min_diag_alt'``, ``'max_diag_sum'``,
      ``'max_diag_product'``, ``'auto'``) raises ``RuntimeError``
      (``CUDSS_STATUS_NOT_SUPPORTED``). They only work on ``AcPfGPU``'s
      single-system solve, and even there ``'max_diag_product'``/``'auto'``
      have been observed to silently produce NaN voltages.
    - ``pivot_epsilon_alg`` (*str*): cuDSS ``CUDSS_CONFIG_PIVOT_EPSILON_ALG``
      choice. One of ``'default'`` (default), ``'scaled'``, ``'static'``.
    - ``max_iter_base``, ``tol_base``: Stored for reference only; do not
      rerun the base case.
    - ``compute_limit_violations`` (*bool*): Fused per-chunk voltage/current/
      divergence check (see :meth:`set_limits`). Default False.
    - ``violation_tol`` (*float*), ``violation_capacity`` (*int*): DIVERGED
      tolerance and max records kept per contingency (default 16).

    Examples
    --------
    .. code-block:: python

        solver = _ContingencyAnalysisSolver(Ybus, Vinit, Sbus,
                                           slack_ids, slack_weights, pv, pq)
        solver.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                               bus_vn_kv, sn_mva)
        solver.build_contingencies([[l] for l in range(n_lines)])
        solver.run()
        V   = solver.V_results.to_numpy()   # lazy D→H
        res = solver.residuals.to_numpy()
        solver.compute_flows()
        or_amps = solver.or_amps.to_numpy()
        ex_amps = solver.ex_amps.to_numpy()
        timings = solver.timings   # full timings after compute_flows()

        # Tune and re-run without rebuilding:
        solver.batch_size = 200
        solver.nb_iter    = 5
        solver.run()
        solver.compute_flows()
        timings2 = solver.timings
    """

    def __init__(self, Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                 batch_size=100, nb_iter=4, max_iter_base=10, tol_base=1e-6,
                 device=None, handle_disconnected_grid=False, presolved_v=False,
                 reordering_alg='default', matching_alg='none',
                 pivot_epsilon_alg='default', debug_base_case=False,
                 scaling_max_voltage_change=False, max_dVa=0.5, max_dVm=0.1):
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = 'direct_refactor_every'
        self._reordering_alg = reordering_alg
        self._matching_alg = matching_alg
        self._pivot_epsilon_alg = pivot_epsilon_alg
        # Single source of truth, set once at construction: forwarded to BOTH
        # the base-case solve (this call) AND the batch solver's own members
        # (consumed by run() -- see the reordering_alg/matching_alg/
        # pivot_epsilon_alg property setters below, which only ever affect
        # the latter, going forward).
        self._s = _ContingencyAnalysisSession(
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
        """Wrap an already-constructed C++ ContingencyAnalysisSession.

        Used by the zero-copy lightsim2grid bridge, which builds the session in
        C++ directly from a solved LSGrid. Reuses all wrapper ergonomics.
        reordering_alg/matching_alg/pivot_epsilon_alg here are bookkeeping only
        (what the caller already passed to the C++ builder at construction) --
        purely so the Python-side property getters below report the truth;
        they do NOT re-apply these to the (already-built) session. Same for
        scaling_max_voltage_change/max_dVa/max_dVm -- no bookkeeping needed
        for those since they're read straight off self._s (a plain bool/float
        pair, unlike reordering_alg's str<->enum resolution).
        """
        self = cls.__new__(cls)
        self._s = session
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = strategy
        self._reordering_alg = reordering_alg
        self._matching_alg = matching_alg
        self._pivot_epsilon_alg = pivot_epsilon_alg
        return self

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
        if value not in _STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy {value!r}. "
                f"Choose from: {list(_STRATEGY_MAP)}")
        self._strategy = value
        self._s.strategy_type = _STRATEGY_MAP[value]

    @property
    def reordering_alg(self):
        """CUDSS_CONFIG_REORDERING_ALG choice (str or ReorderingAlg). Takes
        effect on the next run() (which always reruns cuDSS ANALYSIS).
        One of 'default' (default), 'btf_colamd', 'colamd', 'amd',
        'nested_dissection', 'none'.

        NOTE: cuDSS rejects 'btf_colamd'/'colamd' with CUDSS_STATUS_NOT_SUPPORTED
        in this session's uniform-batch mode -- only 'default', 'amd',
        'nested_dissection', 'none' are supported here. 'btf_colamd'/'colamd'
        work only on AcPfGPU's single-system solve."""
        return self._reordering_alg

    @reordering_alg.setter
    def reordering_alg(self, value):
        self._reordering_alg = value
        self._s.reordering_alg = _resolve_reordering_alg(value)

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
        self._matching_alg = value
        self._s.matching_alg = _resolve_matching_alg(value)

    @property
    def pivot_epsilon_alg(self):
        """CUDSS_CONFIG_PIVOT_EPSILON_ALG choice (str or PivotEpsilonAlg).
        Takes effect on the next run() (which always reruns cuDSS ANALYSIS).
        One of 'default' (default), 'scaled', 'static'."""
        return self._pivot_epsilon_alg

    @pivot_epsilon_alg.setter
    def pivot_epsilon_alg(self, value):
        self._pivot_epsilon_alg = value
        self._s.pivot_epsilon_alg = _resolve_pivot_epsilon_alg(value)

    @property
    def handle_disconnected_grid(self):
        """bool: solve the largest connected component of a split grid (masking the
        rest as NaN) instead of skipping the contingency. Contingencies that strand
        the angle reference or a controller bus are still skipped. Incompatible with
        the 'direct_base_case_factors' strategy. Takes effect on the next run()."""
        return self._s.handle_disconnected_grid

    @handle_disconnected_grid.setter
    def handle_disconnected_grid(self, value):
        self._s.handle_disconnected_grid = bool(value)

    @property
    def refactor_period(self):
        """Refactor period N for 'direct_refactor_every_n'. Takes effect on the next run()."""
        return self._s.refactor_period

    @refactor_period.setter
    def refactor_period(self, value):
        self._s.refactor_period = int(value)

    @property
    def max_iter_base(self):
        return self._max_iter_base

    @max_iter_base.setter
    def max_iter_base(self, value):
        self._max_iter_base = int(value)

    @property
    def tol_base(self):
        return self._tol_base

    @tol_base.setter
    def tol_base(self, value):
        self._tol_base = float(value)

    @property
    def used_batch_size(self):
        return self._s.used_batch_size
    
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances.  Required before build_contingencies
        and compute_flows."""
        self._s.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                bus_vn_kv, sn_mva)

    def build_contingencies(self, branch_ids_per_ctg):
        """Build contingencies from a list-of-lists of branch indices.

        Each inner list is the set of branch IDs to trip in that contingency.
        Requires set_branch_data() to have been called first.
        """
        self._s.build_contingencies(branch_ids_per_ctg)

    def run(self):
        """Run all contingencies.  Fills V_results and residuals on device.

        Does not return timings — call solver.timings after compute_flows()
        for a complete breakdown including flow computation time.
        """
        self._s.run()

    def compute_flows(self):
        """Compute branch flows for all contingencies from stored V_results.

        Tripped branches are zeroed device-side.
        Requires run() and set_branch_data() to have been called.
        """
        self._s.compute_flows()

    @property
    def compute_limit_violations(self):
        """bool: fused per-chunk voltage/current/divergence check (see
        set_limits()). Mirrors lightsim2grid's ContingencyAnalysis flag of the
        same name. Writes only a bounded compact buffer -- never the full
        dense V_results/or_amps/ex_amps. Changing this clears any previously
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
        """float: residual tolerance for the DIVERGED check, independent of
        tol_base. Takes effect on the next run()."""
        return self._s.violation_tol

    @violation_tol.setter
    def violation_tol(self, value):
        self._s.violation_tol = float(value)

    @property
    def violation_capacity(self):
        """int: max violation records kept per contingency (K). Bounds the
        compact output at n_contingencies * K regardless of grid size. Takes
        effect on the next run(); default 16."""
        return self._s.violation_capacity

    @violation_capacity.setter
    def violation_capacity(self, value):
        self._s.violation_capacity = int(value)

    def get_violations(self):
        """list[list[LimitViolation]]: one entry per contingency (row order
        matches build_contingencies()'s input list). [] for a not-simulated
        (disconnected / masked-skip) contingency; a non-converged one gets a
        single DIVERGED entry and nothing else. Requires run() with
        compute_limit_violations=True."""
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
                out.append([])
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
        """(n_contingencies,) bool ndarray: True where more than
        violation_capacity violations were found for that contingency
        (clamped -- raise violation_capacity if this matters for your use
        case). Requires run() with compute_limit_violations=True."""
        return self._s.get_violation_truncated().astype(bool)

    def get_violation_counts(self):
        """dict of (n_contingencies,) int ndarrays with keys 'low_voltage',
        'high_voltage', 'current': the TRUE, uncapped count of violations of
        each type per contingency, -1 for a not-simulated (disconnected /
        masked-skip) contingency.

        Unlike get_violations()'s per-violation records (capped at
        violation_capacity), these totals are always exact -- they keep
        counting past the cap, so they remain reliable even when
        get_violations_truncated() is True for a contingency. Requires run()
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
        """(n_contingencies,) bool ndarray: residual <= tol (defaults to
        violation_tol). Independent of compute_limit_violations -- residuals
        are always O(n_contingencies) and always computed (see residuals);
        this does not require compute_limit_violations to be enabled."""
        t = self.violation_tol if tol is None else tol
        return self.residuals.to_numpy() <= t

    @property
    def V_results(self):
        """DeviceBuffer: (n_contingencies * n_bus,) complex voltages."""
        n_ctg = self._s.n_contingencies
        n_bus = self._s.n_bus
        return DeviceBuffer(self._s.get_V_results, (n_ctg * n_bus,), 'complex128')

    @property
    def residuals(self):
        """DeviceBuffer: (n_contingencies,) ‖F‖∞ residuals."""
        return DeviceBuffer(self._s.get_residuals, (self._s.n_contingencies,), 'float64')

    @property
    def or_amps(self):
        """DeviceBuffer: (n_contingencies * n_branches,) origin terminal amps."""
        n_ctg = self._s.n_contingencies
        n_bra = self._s.n_branches
        return DeviceBuffer(self._s.get_or_amps, (n_ctg * n_bra,), 'float64')

    @property
    def ex_amps(self):
        """DeviceBuffer: (n_contingencies * n_branches,) extremity terminal amps."""
        n_ctg = self._s.n_contingencies
        n_bra = self._s.n_branches
        return DeviceBuffer(self._s.get_ex_amps, (n_ctg * n_bra,), 'float64')

    @property
    def timings(self):
        """ContingencyTimings from the most recent run()."""
        return self._s.get_timings()

    def v_base_dlpack(self):
        """Zero-copy DLPack capsule of base-case voltages, shape [n_bus].

        The capsule aliases live GPU memory owned by this solver.  Pass to
        ``torch.from_dlpack()`` or ``jax.dlpack.from_dlpack()`` for a
        zero-copy GPU tensor.  The solver must remain alive while the tensor
        is in use.
        """
        return self._s.v_base_dlpack()

    def v_results_dlpack(self):
        """Zero-copy DLPack capsule of batch voltages, shape [n_contingencies, n_bus].

        Requires ``run()`` to have been called.  The capsule aliases live GPU
        memory — calling ``run()`` again overwrites it in place.  Clone the
        tensor before a subsequent ``run()`` if a snapshot is needed.
        """
        return self._s.v_results_dlpack()


# Not in __all__: documented separately as gpusim2grid.ContingencyAnalysisGPU /
# gpusim2grid.optimize_reference_slack (see docs/api.rst) to avoid duplicate
# autodoc entries for the same classes/functions.
from .gpu_facade import ContingencyAnalysisGPU, optimize_reference_slack