# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

__all__ = [
    "ContingencyAnalysisSolver",
]

from .._gpusim2grid import (
    ContingencyAnalysisSession as _ContingencyAnalysisSession,
    ContingencySolverType as _ContingencySolverType,
)

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


class ContingencyAnalysisSolver:
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
    - ``max_iter_base``, ``tol_base``: Stored for reference only; do not
      rerun the base case.

    Examples
    --------
    .. code-block:: python

        solver = ContingencyAnalysisSolver(Ybus, Vinit, Sbus,
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
                 device=None):
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = 'direct_refactor_every'
        self._s = _ContingencyAnalysisSession(
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
            int(batch_size), int(nb_iter), self._max_iter_base, self._tol_base,
            _normalize_device(device))

    @classmethod
    def _wrap_session(cls, session, max_iter_base=1, tol_base=1e-6,
                      strategy='direct_refactor_every'):
        """Wrap an already-constructed C++ ContingencyAnalysisSession.

        Used by the zero-copy lightsim2grid bridge, which builds the session in
        C++ directly from a solved LSGrid. Reuses all wrapper ergonomics.
        """
        self = cls.__new__(cls)
        self._s = session
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = strategy
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