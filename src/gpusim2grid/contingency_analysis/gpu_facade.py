# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU contingency analysis driven directly from a lightsim2grid grid.

``ContingencyAnalysisGPU`` is the GPU sibling of lightsim2grid's
``ContingencyAnalysis(grid)``: it takes the solved grid object, reuses the
CPU base-case (N) power flow, and runs the batched contingencies on the GPU.
It is a thin, batch-oriented facade over :class:`_ContingencyAnalysisSolver`.
"""

import numpy as np

from . import _ContingencyAnalysisSolver, _normalize_device
from .._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    grid_from_pandapower,
    _validate_precision,
)
from .. import _gpusim2grid as _cpp

__all__ = ["ContingencyAnalysisGPU", "optimize_reference_slack"]


def _have_bridge():
    return getattr(_cpp, "have_ls2g_bridge", False)


def optimize_reference_slack(grid, contingency_branch_ids, *, Vinit=None,
                             max_iter=30, tol=1e-10):
    """Choose the angle-reference slack that minimises skipped contingencies, set
    it on the grid, and re-solve the base AC power flow (CPU, in lightsim2grid).

    For ``handle_disconnected_grid`` the reference slack is fixed for the whole
    GPU batch, and any contingency that strands it is skipped (NaN). lightsim2grid
    can pick the slack stranded by the *fewest* of the given contingencies and
    re-solve the base case with it as the angle reference; the GPU companion then
    inherits that reference (read off the solved grid) and skips as few split
    contingencies as possible. Call this **before** building a
    :class:`ContingencyAnalysisGPU` (bridge / multi-slack path).

    Requires a lightsim2grid whose ``ContingencyAnalysisCPP`` exposes
    ``pick_reference_slack`` and whose grid exposes ``set_reference_slack_bus``.

    Parameters
    ----------
    grid : lightsim2grid LSGrid
        Grid to re-solve in place (its slack ordering is updated).
    contingency_branch_ids : list[list[int]]
        Branch-removal contingencies (lines-then-trafos), as passed to
        :meth:`ContingencyAnalysisGPU.add_contingencies_by_branch_id`.
    Vinit : (n_bus,) complex, optional
        Base-case warm start; defaults to a flat 1.0 start.
    max_iter, tol : int, float
        Base-case AC solve settings.

    Returns
    -------
    int
        The chosen reference bus id (gridmodel numbering), or -1 if the grid has
        no slack to choose from.
    """
    from lightsim2grid.contingencyAnalysis import ContingencyAnalysisCPP

    ca = ContingencyAnalysisCPP(grid)
    for ids in contingency_branch_ids:
        ca.add_nk([int(i) for i in ids])
    ref = ca.pick_reference_slack()
    if ref is not None and ref >= 0:
        grid.set_reference_slack_bus(int(ref))
        n_bus = grid.get_bus_vn_kv().shape[0]
        v = Vinit if Vinit is not None else np.ones(n_bus, dtype=complex)
        grid.ac_pf(v, int(max_iter), float(tol))
    return ref


class ContingencyAnalysisGPU:
    """Batch N-k contingency analysis on the GPU, seeded from a CPU solve.

    By default (``use_bridge=None`` auto-detects the compiled lightsim2grid
    bridge) every contingency is solved on the **same augmented system
    lightsim2grid poses** — distributed slack, HVDC angle-droop, SVC, and remote
    generator voltage control are carried through the Jacobian under the Ybus
    patch. When the bridge is unavailable (or ``use_bridge=False``) the
    Python-array fallback solves only the bare ``[pvpq | pq]`` system.

    Parameters
    ----------
    grid : lightsim2grid GridModel / LSGrid, or tuple
        Either the grid whose base case has been (or will be) solved on the
        CPU (lightsim2grid owns the physics; gpusim2grid only runs the GPU
        batch), or an explicit
        ``(Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq)`` array tuple
        for callers without a lightsim2grid grid. In the latter case,
        ``use_bridge`` must not be True, branch data must be supplied via
        :meth:`set_branch_data`, and limits (if any) via :meth:`set_limits`.
    init_from_n_powerflow : bool, default True
        Seed every contingency with the CPU-converged base-case voltage V0
        (mirrors lightsim2grid's flag of the same name).  This skips the GPU
        base-case Newton-Raphson loop: at V0 the base mismatch is already below
        tolerance, so a single GPU step fills J at the right operating point and
        factorizes once.  When ``False``, the GPU runs ``max_iter_base`` NR
        iterations from a DC warm-start.
    precision : {"fp64", "fp32", None}, default "fp64"
        Validated against the compiled extension; precision is a build-time
        choice.  ``None`` accepts whatever was compiled.
    nb_iter : int, default 4
        Newton-Raphson iterations per contingency in the batch phase.
    max_iter_base, tol_base : int, float
        Base-case CPU solve / GPU fallback settings.
    device : None | int | "cuda" | "cuda:N"
        Target CUDA device.
    compute_limit_violations : bool, default False
        Enable the fused per-chunk voltage/current/divergence check (mirrors
        lightsim2grid's ``ContingencyAnalysis`` flag of the same name). Bus
        voltage / branch current limits are extracted from ``grid`` (via
        the bridge, or :meth:`set_limits_from_grid` for the array path).
        See :meth:`get_violations` / :meth:`converged`.

    Examples
    --------
    >>> grid = init_from_pandapower(net)
    >>> grid.ac_pf(Vinit, max_iter=10, tol=1e-8)
    >>> ca = ContingencyAnalysisGPU(grid, init_from_n_powerflow=True, nb_iter=4)
    >>> ca.add_contingencies_by_branch_id([[12], [40], [12, 40]])
    >>> V_batch = ca.compute(batch_size=512)   # DLPack (n_ctg, n_bus) complex
    >>> residuals = ca.last_residuals()        # ‖F‖∞ per contingency

    Explicit-array path (no lightsim2grid grid available):

    >>> ca = ContingencyAnalysisGPU(
    ...     (Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq), nb_iter=4)
    >>> ca.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, sn_mva)
    >>> ca.add_contingencies_by_branch_id([[12], [40], [12, 40]])
    >>> V_batch = ca.compute(batch_size=512)
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 use_bridge=None, handle_disconnected_grid=False,
                 compute_limit_violations=False):
        _validate_precision(precision)

        if isinstance(grid, (tuple, list)):
            # Explicit-array mode: no lightsim2grid grid to seed from, extract
            # branch/limit data from, or bridge to.
            if use_bridge:
                raise ValueError(
                    "use_bridge=True requires `grid` to be a lightsim2grid "
                    "grid object, not an explicit-array tuple.")
            self._grid = None
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq = grid
            self._inner = _ContingencyAnalysisSolver(
                Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
                batch_size=100, nb_iter=nb_iter,
                max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                presolved_v=bool(init_from_n_powerflow))
            # No grid to auto-extract branch/limit data from: the caller must
            # call set_branch_data() (and set_limits() before compute() when
            # compute_limit_violations is wanted).
            self._n_branches = None
            self._inner.compute_limit_violations = bool(compute_limit_violations)
        else:
            # Retained for set_limits_from_grid()/converged_n()/get_violations_n(),
            # which need to read the grid's own (bus/branch limit, base-case V)
            # state after construction -- not mutated by this class beyond what
            # already happens (optimize_reference_slack is the only mutator, and
            # that's a separate free function called BEFORE construction).
            self._grid = grid

            if use_bridge is None:
                use_bridge = _have_bridge()

            if use_bridge:
                # Zero-copy: extract everything in C++ off the solved LSGrid
                # (no scipy CSR marshalling, branch data set automatically).
                # When compute_limit_violations is True, the bridge also pulls
                # bus/branch limits off the grid and enables the fused check --
                # see ls2g_bridge.cpp:make_ca_session_from_lsgrid.
                session = _cpp._make_ca_session_from_lsgrid(
                    grid, bool(init_from_n_powerflow), 100, int(nb_iter),
                    int(max_iter_base), float(tol_base), _normalize_device(device),
                    bool(compute_limit_violations))
                self._inner = _ContingencyAnalysisSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base)
                self._n_branches = self._inner._s.n_branches
            else:
                # TODO(bug): no v_init= forwarded here, so extract_grid_arrays()
                # / _ensure_solved() always re-solves from its own DC warm-start
                # with (max_iter_base, tol_base), silently discarding/overriding
                # any Vinit + solver settings the caller already used in a prior
                # grid.ac_pf() call on this same grid object (array/non-bridge
                # path only -- the bridge path has no such re-solve, see
                # ls2g_bridge.cpp). See also the pv/pq/slack numbering TODO in
                # extract_grid_arrays() (_ls2g_utils.py).
                d = extract_grid_arrays(grid, max_iter=max_iter_base, tol=tol_base)
                # Seed from the CPU base-case solution and either trust it as already
                # converged (presolved_v, no GPU NR loop) or run the full GPU base
                # solve from the DC start.
                vinit = d["v_converged"] if init_from_n_powerflow else d["v_init"]

                self._inner = _ContingencyAnalysisSolver(
                    d["Ybus"], vinit, d["Sbus"],
                    d["slack"], d["slack_weights"], d["pv"], d["pq"],
                    batch_size=100, nb_iter=nb_iter,
                    max_iter_base=max_iter_base, tol_base=tol_base, device=device,
                    presolved_v=init_from_n_powerflow)

                # Branch admittances come straight from lightsim2grid (never
                # recomputed) so branch-removal contingencies become exact Ybus
                # patches.
                branch_args, _, _ = extract_branch_data(grid)
                self._inner.set_branch_data(*branch_args)
                self._n_branches = len(branch_args[0])

                # The array-based (non-bridge) path has no C++-side grid access,
                # so compute_limit_violations is enabled here in Python instead
                # of inside the C++ bridge call above.
                if compute_limit_violations:
                    self.set_limits_from_grid()
                    self._inner.compute_limit_violations = True

        # Solve the largest connected component of a split grid (masking the rest
        # as NaN) instead of skipping such contingencies. Works on both the bridge
        # and the array path (mutable property on the underlying session).
        self._inner.handle_disconnected_grid = bool(handle_disconnected_grid)

        self._nb_iter = int(nb_iter)
        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def set_branch_data(self, branch_from, branch_to, yff, yft, ytf, ytt,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances (explicit-array mode only).

        Grid mode extracts this automatically at construction; only needed
        when ``grid`` was an explicit-array tuple. Required before
        :meth:`add_contingencies_by_branch_id` and :meth:`compute_flows`.
        """
        self._inner.set_branch_data(branch_from, branch_to, yff, yft, ytf, ytt,
                                    bus_vn_kv, sn_mva)
        self._n_branches = len(branch_from)

    def set_limits(self, bus_vmin_kv, bus_vmax_kv, branch_limit_a1_ka,
                  branch_limit_a2_ka, n_lines):
        """Configure bus voltage / branch current limits (explicit-array mode
        only). Grid mode should use :meth:`set_limits_from_grid` instead."""
        self._inner.set_limits(bus_vmin_kv, bus_vmax_kv, branch_limit_a1_ka,
                               branch_limit_a2_ka, int(n_lines))

    def add_contingencies_by_branch_id(self, branch_ids_per_ctg):
        """Define contingencies as branch removals.

        Parameters
        ----------
        branch_ids_per_ctg : list[list[int]]
            One inner list per contingency, holding the branch indices to trip.
            Indices are 0-based, lines first then trafos (``c < n_lines`` is
            line ``c``; ``c >= n_lines`` is trafo ``c - n_lines``).
        """
        self._inner.build_contingencies(branch_ids_per_ctg)

    def compute(self, batch_size=512):
        """Solve every contingency and return the batched voltages.

        Returns a DLPack capsule of shape ``(n_contingencies, n_bus)``,
        complex, aliasing live GPU memory.  Pass to ``torch.from_dlpack`` /
        ``jax.dlpack.from_dlpack``; clone before the next ``compute()`` for a
        snapshot.  Residuals are cached for :meth:`last_residuals`.
        """
        self._inner.batch_size = int(batch_size)
        self._inner.run()
        self._last_residuals = self._inner.residuals
        return self._inner.v_results_dlpack()

    def last_residuals(self):
        """``‖F‖∞`` per contingency from the most recent :meth:`compute`."""
        if self._last_residuals is None:
            raise RuntimeError("Call compute() before last_residuals().")
        return self._last_residuals.to_numpy()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute()."""
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
        """list[list[LimitViolation]]: one entry per contingency (row order
        matches add_contingencies_by_branch_id). Requires compute() with
        compute_limit_violations=True."""
        return self._inner.get_violations()

    def get_violations_truncated(self):
        """(n_ctg,) bool ndarray: True where more than violation_capacity
        violations were found for that contingency (clamped)."""
        return self._inner.get_violations_truncated()

    def get_violation_counts(self):
        """dict of (n_ctg,) int ndarrays with keys 'low_voltage',
        'high_voltage', 'current': the TRUE, uncapped count of violations of
        each type per contingency (-1 = not simulated). Unlike
        get_violations()'s records (capped at violation_capacity), these
        totals stay exact even when get_violations_truncated() is True."""
        return self._inner.get_violation_counts()

    def converged(self, tol=None):
        """(n_ctg,) bool ndarray: residual <= tol (defaults to violation_tol).
        Independent of compute_limit_violations -- requires compute()."""
        return self._inner.converged(tol)

    def converged_n(self, tol=None):
        """bool: whether the pre-contingency ('n') CPU base-case solve
        converged. Construction already validates this against tol_base
        (raising RuntimeError otherwise), so this always returns True --
        exists for API parity with lightsim2grid's converged_n(). tol is
        accepted for signature symmetry with converged() but unused."""
        return True

    def get_violations_n(self):
        """list[LimitViolation] for the pre-contingency ('n') case: a single
        voltage vector, not a batch, so this is pure Python/CPU (no GPU,
        no memory/transfer concern -- see
        gpusim2grid.contingency_analysis._limit_violations.compute_violations_n).
        Requires set_limits_from_grid() (or a compute_limit_violations=True
        construction) to have been called; with no limits configured,
        returns []."""
        from ._limit_violations import compute_violations_n

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
        """DeviceBuffer: (n_ctg * n_branches,) origin terminal amps."""
        return self._inner.or_amps

    @property
    def ex_amps(self):
        """DeviceBuffer: (n_ctg * n_branches,) extremity terminal amps."""
        return self._inner.ex_amps

    @property
    def V_results(self):
        """DeviceBuffer: (n_ctg * n_bus,) complex voltages (lazy D->H)."""
        return self._inner.V_results

    @property
    def strategy(self):
        """Linear-solve strategy (str). Takes effect on the next compute()."""
        return self._inner.strategy

    @strategy.setter
    def strategy(self, value):
        self._inner.strategy = value

    @property
    def timings(self):
        """BatchTimings from the most recent compute() / compute_flows()."""
        return self._inner.timings

    @property
    def n_branches(self):
        return self._n_branches

    @property
    def n_contingencies(self):
        return self._inner._s.n_contingencies

    @property
    def n_bus(self):
        return self._inner._s.n_bus

    @property
    def solver(self):
        """The underlying :class:`_ContingencyAnalysisSolver` (escape hatch)."""
        return self._inner

    # --------------------------------------------------------------- factory
    @classmethod
    def from_pandapower(cls, net, solver_type="KLU", **kwargs):
        """Build directly from a pandapower network (convenience).

        Converts to a lightsim2grid grid, solves the base case on the CPU, then
        constructs the GPU analysis.
        """
        grid = grid_from_pandapower(net, solver_type=solver_type)
        return cls(grid, **kwargs)
