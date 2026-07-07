# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
gpusim2grid.injection_sweep — Python facade for the batched-injection
power flow sweep.

Re-exports the C++ binding `acpf_nr_gpu_injection` plus a `'string' → enum`
mapping for the linear-solve strategy, mirroring the convention used by
gpusim2grid.contingency_analysis.

Quick start
-----------

.. code-block:: python

    from gpusim2grid.injection_sweep import acpf_nr_gpu_injection

    V_flat, residuals, timings = acpf_nr_gpu_injection(
        Ybus, v_init, Sbus_base,
        slack_ids, slack_weights, pv, pq,
        p_mw,        # (n_scenarios, n_bus) float64 — MW
        q_mvar,      # (n_scenarios, n_bus) float64 — MVAr
        sn_mva,      # system base apparent power
        batch_size=64,
        nb_iter=4,
        strategy='direct_refactor_every',   # or 'direct_iter0_only', etc.
    )

``strategy`` may be passed as either a string (mapped via ``_STRATEGY_MAP``)
or the raw ``ContingencySolverType`` enum value.
"""

__all__ = [
    "acpf_nr_gpu_injection",
    "run_injection_sweep_gpu",
]

import numpy as np

from .._gpusim2grid import (
    acpf_nr_gpu_injection as _acpf_nr_gpu_injection,
    ContingencySolverType as _ContingencySolverType,
    InjectionSweepSession as _InjectionSweepSession,
)

# Shared with contingency_analysis/__init__.py (single source of truth for
# the reordering-alg string↔enum map, same as _normalize_device).
from ..contingency_analysis import _resolve_reordering_alg


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


def acpf_nr_gpu_injection(
    Ybus, Vinit, Sbus,
    slack_ids, slack_weights, pv, pq,
    p_mw, q_mvar, sn_mva,
    batch_size, nb_iter,
    max_iter_base=10, tol_base=1e-6,
    strategy='direct_refactor_every',
    refactor_period=1,
):
    """Batched-injection AC PF sweep on GPU.

    Parameters
    ----------
    Ybus : scipy.sparse complex (n_bus, n_bus)
    Vinit : (n_bus,) complex128 — warm-start voltages for the base case.
    Sbus : (n_bus,) complex128 — per-unit injections used for the base-case NR.
    slack_ids, slack_weights, pv, pq : as in run_contingency_analysis_gpu.
    p_mw, q_mvar : (n_scenarios, n_bus) float64, row-major.
        Per-scenario active and reactive power in PHYSICAL units (MW / MVAr).
        Internally divided by ``sn_mva`` to form per-unit complex Sbus.
    sn_mva : float — system base apparent power, used for MW/MVAr → pu.
    batch_size, nb_iter : as in the contingency entry point.
    max_iter_base, tol_base : base-case NR convergence settings.
    strategy : 'direct_refactor_every' | 'direct_iter0_only'
               | 'direct_base_case_factors' | 'direct_refactor_every_n'
               | ContingencySolverType — selects the linear-solve policy.
    refactor_period : int — only used by 'direct_refactor_every_n'.

    Returns
    -------
    (V_results, residuals, timings) tuple, where
        V_results : (n_scenarios * n_bus,) complex128
        residuals : (n_scenarios,) float64
        timings   : BatchTimings (a.k.a. ContingencyTimings)
    """
    strategy_enum = _resolve_strategy(strategy)
    return _acpf_nr_gpu_injection(
        Ybus, Vinit, Sbus,
        slack_ids, slack_weights, pv, pq,
        p_mw, q_mvar, sn_mva,
        batch_size, nb_iter,
        max_iter_base, tol_base,
        strategy_enum, refactor_period,
    )


# Alias matching the contingency-side naming convention
# (run_contingency_analysis_gpu  ↔  run_injection_sweep_gpu).
run_injection_sweep_gpu = acpf_nr_gpu_injection


def _normalize_device(device):
    """Normalize a device specifier to an int for the C++ ctor."""
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


class _DeviceBuffer:
    """Handle to GPU-resident data; transfers to host on demand via to_numpy()."""

    def __init__(self, fetch_fn, shape, dtype):
        self._fetch = fetch_fn
        self._shape = shape
        self._dtype = dtype

    def to_numpy(self):
        return self._fetch()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"DeviceBuffer(shape={self._shape}, dtype={self._dtype!r})"


class _InjectionSweepSolver:
    """Stateful GPU batched-injection power flow solver.

    The base-case Newton-Raphson is solved once at construction; subsequent
    ``set_injections()`` + ``run()`` calls reuse that base case (its single-system
    J sparsity, scatter maps, and converged voltages), so a parameter sweep that
    re-runs with different injections — or a different ``batch_size`` / ``nb_iter``
    / ``strategy`` — does not re-solve the base case.

    Parameters
    ----------
    Ybus : scipy.sparse complex (n_bus, n_bus)
    Vinit : (n_bus,) complex128 — base-case warm-start voltages.
    Sbus : (n_bus,) complex128 — per-unit injections for the base-case NR.
    slack_ids, slack_weights, pv, pq : index/weight arrays (as in the contingency solver).
    batch_size : int, optional — scenarios per GPU chunk (default 100).
    nb_iter : int, optional — fixed NR iterations per chunk (default 4).
    max_iter_base, tol_base : base-case NR convergence settings.

    Notes
    -----
    The following properties are **mutable** and take effect on the next
    :meth:`run` call: ``batch_size``, ``nb_iter``, ``strategy``,
    ``refactor_period``, ``reordering_alg``.

    Examples
    --------
    .. code-block:: python

        solver = _InjectionSweepSolver(Ybus, Vinit, Sbus,
                                      slack_ids, slack_weights, pv, pq)
        solver.set_injections(p_mw, q_mvar, sn_mva)   # (n_scen, n_bus) MW / MVAr
        solver.run()
        V   = solver.V_results.to_numpy()             # lazy D→H
        res = solver.residuals.to_numpy()
        timings = solver.timings

        # Sweep a second injection set without rebuilding the base case:
        solver.set_injections(p_mw2, q_mvar2, sn_mva)
        solver.strategy = 'direct_iter0_only'
        solver.run()
    """

    def __init__(self, Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                 batch_size=100, nb_iter=4, max_iter_base=10, tol_base=1e-6,
                 device=None, presolved_v=False):
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = 'direct_refactor_every'
        self._reordering_alg = 'default'
        self._s = _InjectionSweepSession(
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
            int(batch_size), int(nb_iter), self._max_iter_base, self._tol_base,
            _normalize_device(device), presolved_v=bool(presolved_v))

    @classmethod
    def _wrap_session(cls, session, max_iter_base=1, tol_base=1e-6,
                      strategy='direct_refactor_every', reordering_alg='default'):
        """Wrap an already-constructed C++ InjectionSweepSession (zero-copy
        lightsim2grid bridge). Reuses all wrapper ergonomics."""
        self = cls.__new__(cls)
        self._s = session
        self._max_iter_base = int(max_iter_base)
        self._tol_base = float(tol_base)
        self._strategy = strategy
        self._reordering_alg = reordering_alg
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
    def refactor_period(self):
        return self._s.refactor_period

    @refactor_period.setter
    def refactor_period(self, value):
        self._s.refactor_period = int(value)

    @property
    def used_batch_size(self):
        return self._s.used_batch_size

    # --- branch data (optional, needed for compute_flows) ---
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances.  Required before compute_flows().

        Parameters match _ContingencyAnalysisSolver.set_branch_data() exactly:
        branch_from, branch_to : (n_branches,) int
        yff, yft, ytf, ytt     : (n_branches,) complex — π-model admittances
        bus_vn_kv              : (n_bus,) float — nominal voltage in kV per bus
        sn_mva                 : float — system base apparent power (MVA)
        """
        self._s.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                bus_vn_kv, sn_mva)

    # --- inputs ---
    def set_injections(self, p_mw, q_mvar, sn_mva):
        """Store the (n_scenarios, n_bus) MW / MVAr injection arrays.

        Converted to per-unit complex Sbus on the next ``run()``.  May be
        called repeatedly to sweep different injection sets reusing the base case.
        """
        p = np.ascontiguousarray(p_mw, dtype=np.float64)
        q = np.ascontiguousarray(q_mvar, dtype=np.float64)
        self._s.set_injections(p, q, float(sn_mva))

    def run(self):
        """Run all scenarios; fills V_results and residuals on device."""
        self._s.run()

    def compute_flows(self):
        """Compute branch flows for all scenarios from stored V_results.

        Requires run() and set_branch_data() to have been called.
        Results are then available via solver.or_amps and solver.ex_amps.
        """
        self._s.compute_flows()

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


# Not in __all__: documented separately as gpusim2grid.InjectionSweepGPU (see
# docs/api.rst) to avoid duplicate autodoc entries for the same class.
from .gpu_facade import InjectionSweepGPU