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

from . import (
    _ContingencyAnalysisSolver,
    PhysicalChecksFacadeMixin,
    SlackRedistributionFacadeMixin,
    SchedulingFacadeMixin,
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
from ._reactive_limits import ReactiveLimitsStatus

__all__ = ["ContingencyAnalysisGPU", "optimize_reference_slack", "ReactiveLimitsStatus"]


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

    The ``*GPU`` facades now make the same choice themselves by default
    (``reference_slack="auto"``, lightsim2grid PR #216's batch rule), without
    touching the grid; this function remains for callers that want the grid
    itself re-solved with that reference, or use ``reference_slack="grid"``.
    Note that the reference it forces on the grid is then kept by the facades.

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


class ContingencyAnalysisGPU(PhysicalChecksFacadeMixin, SlackRedistributionFacadeMixin,
                             SchedulingFacadeMixin):
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
        extension's running state (e.g. distributed-slack ``slack_absorbed``)
        is seeded directly from lightsim2grid's own converged values, needing
        no cuDSS solve at all for the base case. Setting this True forces the
        pre-ground-truth cuDSS-solve derivation instead -- an opt-in
        diagnostic (e.g. to keep testing ``reordering_alg``/``matching_alg``/
        ``pivot_epsilon_alg`` choices in isolation, or to cross-validate the
        GPU's own Newton-derived state against lightsim2grid's).
    scaling_max_voltage_change : bool or None, default None
        NR step-scaling, mirrors lightsim2grid's own
        ``MaxVoltageChangeScalingPolicy``: after solving for the Newton step,
        scale it by ``alpha <= 1`` so ``max|dtheta| <= max_dVa`` and
        ``max|dVm| <= max_dVm`` before applying it anywhere. Applied to BOTH
        the base-case solve AND the batch solver used by :meth:`compute` --
        each contingency/scenario in the batch gets its OWN alpha from its
        own max step, not one alpha shared across the whole chunk (a "hard"
        contingency is damped on its own terms). ``None`` (default) is
        opt-in by inheritance: mirrors whatever ``grid``'s own
        ``get_ac_algo_config()`` is already set to, so behavior only changes
        for grids the caller explicitly configured with damping. Pass
        ``True``/``False`` to force it on/off regardless of the grid's own
        config. Without it, an undamped GPU Newton step can converge onto a
        different (sometimes spurious) root when seeded far from the
        solution (e.g. ``init_from_n_powerflow=False`` from a DC warm-start)
        -- observed on real RTE grids. The mutable ``scaling_max_voltage_change``/
        ``max_dVa``/``max_dVm`` properties on the returned object only ever
        affect the batch solver afterward (same scope as ``reordering_alg``).
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
    reactive_limits_outer_loop : bool, default False
        Opt-in: after the batch, re-solve ONCE the contingencies whose
        physical checks report a reactive-limit switch this loop handles --
        ``LOW_Q`` / ``HIGH_Q`` on a bus its own machines hold (switched to
        PQ at the limit) or on every controller bus of a voltage-control group
        of generators (each held at its own limit, the regulated bus
        floating), and ``LOW_VOLTAGE_AT_MIN_Q`` / ``HIGH_VOLTAGE_AT_MAX_Q`` on
        a generator frozen at a limit that regulates its own PQ bus (held PV
        again) or a remote one (released into its group, with lightsim2grid's
        held controllers) -- one pass of OpenLoadFlow's ``ReactiveLimits``
        outer loop, in a second, smaller batch. Their voltages, residuals and
        violations (operational and physical, with the new labels) then
        replace the first pass'. A contingency reporting a switch the loop
        does not handle (a group only part of which saturated or holding an
        SVC / station, an SVC) is not re-solved. See
        :meth:`get_outer_loop_status` and ``_reactive_limits.py``. Needs a
        lightsim2grid grid (bridge path) and ``compute_physical_violations``.
        Mutable; takes effect on the next :meth:`compute`.
    outer_loop_min_last_chunk : int, default 250
        The second pass runs in chunks of ``batch_size``; a last, partial
        chunk is only run when it is the only one or holds at least this many
        contingencies -- otherwise they are left out
        (``ReactiveLimitsStatus.LEFT_OUT``, first pass kept), rather than
        paying a whole batch for a handful. Mutable.
    outer_loop_warm_start : bool, default True
        The second pass starts each contingency from its first-pass voltages
        (a device-to-device copy; a masked bus from the base case), as
        OpenLoadFlow continues from the current state after an outer-loop
        action, instead of from the base case: only the switches are left to
        converge. Mutable.
    outer_loop_nb_iter : int or None, default None
        Newton iterations of the second pass; None = :attr:`nb_iter`. With
        the warm start, fewer usually suffice; a row that does not converge
        is ``DIVERGED`` and keeps its first pass. Mutable.
    scheduling : {"chunked", "continuous"}, default "chunked"
        How the rows go through the batch. ``"chunked"``: ``ceil(n_rows /
        batch_size)`` chunks, every row runs exactly ``nb_iter`` Newton
        iterations. ``"continuous"``: ``batch_size`` slots; every
        ``nb_iter_per_round`` iterations each row is checked, and one that
        converged (``||F||inf < tol``), diverged or used its ``nb_iter``
        budget leaves at once, its slot refilled from the queue -- a row pays
        the iterations it needs, not the hardest row's. ``nb_iter`` is then
        each row's budget (a row runs a multiple of ``nb_iter_per_round``, at
        least one round). Continuous refuses the ``'direct_iter0_only'`` /
        ``'direct_refactor_every_n'`` strategies. Mutable.
    nb_iter_per_round : int, default 1
        Continuous scheduling: iterations between two convergence checks. A
        check (one SpMV + mismatch) is cheap next to a refactorization, so 1
        is usually fastest: on case6515rte N-1, 1 / 2 / 4 took 6.7 / 8.6 /
        9.2 s. Mutable.
    tol : float or None, default None
        A row has converged when ``||F||inf < tol``, per unit like
        :meth:`last_residuals`: when it leaves (continuous) and its
        :class:`RowStatus` (both schedules, :meth:`get_row_status`).
        lightsim2grid compares the same way but takes its ``tol`` in MVA
        (``||F||inf < tol / sn_mva``). ``None`` = 1e-8 (1e-3 in an FP32
        build). Mutable.

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
    >>> ca.set_branch_data(branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, bus_vn_kv, sn_mva)
    >>> ca.add_contingencies_by_branch_id([[12], [40], [12, 40]])
    >>> V_batch = ca.compute(batch_size=512)
    """

    def __init__(self, grid, *, init_from_n_powerflow=True, precision="fp64",
                 nb_iter=4, max_iter_base=10, tol_base=1e-8, device=None,
                 use_bridge=None, handle_disconnected_grid=False,
                 compute_limit_violations=False, reordering_alg=None,
                 matching_alg=None, pivot_epsilon_alg=None,
                 debug_base_case=False,
                 scaling_max_voltage_change=None, max_dVa=None, max_dVm=None,
                 use_distributed_slack=True,
                 compute_physical_violations=False, redistribute_slack=False,
                 reference_slack="auto", reactive_limits_outer_loop=False,
                 outer_loop_min_last_chunk=250, outer_loop_warm_start=True,
                 outer_loop_nb_iter=None, scheduling="chunked", nb_iter_per_round=1,
                 tol=None):
        _validate_precision(precision)
        # the second pass of reactive_limits_outer_loop is a ScenarioSweepGPU
        # built (lazily) from the same grid with the same construction options
        self._rl_ctor_kwargs = dict(
            init_from_n_powerflow=init_from_n_powerflow, precision=precision,
            nb_iter=nb_iter, max_iter_base=max_iter_base, tol_base=tol_base,
            device=device, use_bridge=use_bridge, reordering_alg=reordering_alg,
            matching_alg=matching_alg, pivot_epsilon_alg=pivot_epsilon_alg,
            debug_base_case=debug_base_case,
            scaling_max_voltage_change=scaling_max_voltage_change,
            max_dVa=max_dVa, max_dVm=max_dVm, use_distributed_slack=use_distributed_slack)
        self._rl_sweep = None
        self._rl_ctx = None
        self._rl_result = None
        self._ctg_branch_ids = None
        self.reactive_limits_outer_loop = reactive_limits_outer_loop
        self.outer_loop_min_last_chunk = outer_loop_min_last_chunk
        self.outer_loop_warm_start = outer_loop_warm_start
        self.outer_loop_nb_iter = outer_loop_nb_iter

        # Single source of truth, resolved once here and applied at
        # construction time to BOTH the base-case solve and the batch solver
        # (see _ContingencyAnalysisSolver's identical-shaped ctor) -- None
        # (default) leaves each at the session's own default.
        _reordering_alg = 'default' if reordering_alg is None else reordering_alg
        _matching_alg = 'none' if matching_alg is None else matching_alg
        _pivot_epsilon_alg = 'default' if pivot_epsilon_alg is None else pivot_epsilon_alg

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
                presolved_v=bool(init_from_n_powerflow),
                reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                pivot_epsilon_alg=_pivot_epsilon_alg,
                debug_base_case=bool(debug_base_case),
                # No grid to inherit a scaling policy from -- None means off.
                scaling_max_voltage_change=bool(scaling_max_voltage_change),
                max_dVa=0.5 if max_dVa is None else float(max_dVa),
                max_dVm=0.1 if max_dVm is None else float(max_dVm))
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
                    bool(compute_limit_violations),
                    reordering_alg=_resolve_reordering_alg(_reordering_alg),
                    matching_alg=_resolve_matching_alg(_matching_alg),
                    pivot_epsilon_alg=_resolve_pivot_epsilon_alg(_pivot_epsilon_alg),
                    debug_base_case=bool(debug_base_case),
                    # None => -1/-1.0 sentinels: inherit the grid's own
                    # get_ac_algo_config() (opt-in by construction). See
                    # AcPfGPU's identical pattern.
                    scaling_max_voltage_change_override=(
                        -1 if scaling_max_voltage_change is None
                        else int(bool(scaling_max_voltage_change))),
                    max_dVa_override=-1.0 if max_dVa is None else float(max_dVa),
                    max_dVm_override=-1.0 if max_dVm is None else float(max_dVm),
                    use_distributed_slack=bool(use_distributed_slack))
                self._inner = _ContingencyAnalysisSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg)
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
                    presolved_v=init_from_n_powerflow,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg,
                    debug_base_case=bool(debug_base_case),
                    # Python-array fallback: no C++ bridge to inherit the
                    # grid's algo config through, same as the tuple path.
                    scaling_max_voltage_change=bool(scaling_max_voltage_change),
                    max_dVa=0.5 if max_dVa is None else float(max_dVa),
                    max_dVm=0.1 if max_dVm is None else float(max_dVm))

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

        # Post-solve physical checks (compute_physical_violations: bus reactive
        # capability + hvdc droop saturation): the bus-Q plan is pulled off the
        # grid here when there is one -- see PhysicalChecksFacadeMixin.
        # Batch scheduling (chunked / continuous), see SchedulingEngineMixin.
        self._inner._init_scheduling(scheduling, nb_iter_per_round, tol)

        self._apply_physical_checks_kwargs(compute_physical_violations)

        # OLF-style bounded slack redistribution of the island a contingency
        # cuts off, and the reference slack the fewest contingencies strand --
        # see SlackRedistributionFacadeMixin.
        if redistribute_slack:
            self.redistribute_slack = True
        self.reference_slack = reference_slack

        self._nb_iter = int(nb_iter)
        self._init_from_n_powerflow = bool(init_from_n_powerflow)
        self._last_residuals = None

    # ------------------------------------------------------------------ spec
    def set_branch_data(self, branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
                        bus_vn_kv, sn_mva):
        """Store π-model branch admittances (explicit-array mode only).

        Grid mode extracts this automatically at construction; only needed
        when ``grid`` was an explicit-array tuple. Required before
        :meth:`add_contingencies_by_branch_id` and :meth:`compute_flows`.
        """
        self._inner.set_branch_data(branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
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
        self._ctg_branch_ids = [list(map(int, ids)) for ids in branch_ids_per_ctg]

    def compute(self, batch_size=512):
        """Solve every contingency and return the batched voltages.

        Returns a DLPack capsule of shape ``(n_contingencies, n_bus)``,
        complex, aliasing live GPU memory.  Pass to ``torch.from_dlpack`` /
        ``jax.dlpack.from_dlpack``; clone before the next ``compute()`` for a
        snapshot.  Residuals are cached for :meth:`last_residuals`.

        With :attr:`reactive_limits_outer_loop`, the contingencies it re-solves
        have their second-pass voltages and residuals in these buffers (see
        :meth:`get_outer_loop_status`).
        """
        self._inner.batch_size = int(batch_size)
        self._rl_result = None
        self._inner.run()
        self._last_residuals = self._inner.residuals
        if self._reactive_limits_outer_loop:
            self._run_outer_loop(int(batch_size))
        return self._inner.v_results_dlpack()

    # --------------------------------------------- reactive_limits_outer_loop
    @property
    def reactive_limits_outer_loop(self):
        """bool: re-solve once, with the switches applied, the contingencies
        whose physical checks report a reactive-limit switch this loop handles
        (see the constructor's doc). Default False; takes effect on the next
        :meth:`compute`."""
        return self._reactive_limits_outer_loop

    @reactive_limits_outer_loop.setter
    def reactive_limits_outer_loop(self, value):
        if bool(value) != value:
            raise ValueError("The `reactive_limits_outer_loop` attribute must be a boolean.")
        self._reactive_limits_outer_loop = bool(value)

    @property
    def outer_loop_min_last_chunk(self):
        """int: a last, partial chunk of the second pass runs only when it is
        the only one or holds at least this many contingencies (default 250)."""
        return self._outer_loop_min_last_chunk

    @outer_loop_min_last_chunk.setter
    def outer_loop_min_last_chunk(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("outer_loop_min_last_chunk must be >= 0")
        self._outer_loop_min_last_chunk = value

    @property
    def outer_loop_warm_start(self):
        """bool: the second pass starts each contingency from its first-pass
        voltages (default True) instead of the base case."""
        return self._outer_loop_warm_start

    @outer_loop_warm_start.setter
    def outer_loop_warm_start(self, value):
        if bool(value) != value:
            raise ValueError("The `outer_loop_warm_start` attribute must be a boolean.")
        self._outer_loop_warm_start = bool(value)

    @property
    def outer_loop_nb_iter(self):
        """int or None: Newton iterations of the second pass (None:
        :attr:`nb_iter`)."""
        return self._outer_loop_nb_iter

    @outer_loop_nb_iter.setter
    def outer_loop_nb_iter(self, value):
        if value is not None:
            value = int(value)
            if value < 1:
                raise ValueError("outer_loop_nb_iter must be >= 1 (or None)")
        self._outer_loop_nb_iter = value

    def get_outer_loop_status(self):
        """(n_contingencies,) int ndarray of :class:`ReactiveLimitsStatus`, from
        the last :meth:`compute` made with :attr:`reactive_limits_outer_loop`:
        NO_SWITCH (nothing to switch, first pass final), RECOMPUTED (second
        pass), UNSUPPORTED / LEFT_OUT / DIVERGED (first pass kept -- a switch
        the loop does not handle, left out by the batch rule, the second pass
        did not converge)."""
        if self._rl_result is None:
            raise RuntimeError(
                "get_outer_loop_status() needs a compute() made with "
                "reactive_limits_outer_loop=True.")
        return self._rl_result["status"].copy()

    def get_outer_loop_switches(self):
        """list, one entry per contingency, from the last :meth:`compute` made
        with :attr:`reactive_limits_outer_loop`: None where nothing was
        switched (every status but RECOMPUTED / LEFT_OUT / DIVERGED), else a
        dict ``{"to_pq": {bus: (q_mvar, at_min)}, "to_pv": {bus: (gen_ids,
        vm_pu)}, "vc_pin": {group: at_min}}`` -- the solver buses switched to PQ
        with the summed reactive power their machines produce (their min_q sum
        when ``at_min``, else their max_q sum), the buses held PV again at
        ``vm_pu`` by the released generators ``gen_ids``, and the
        VoltageControl groups ``{group: (at_min, gen_ids)}`` whose every
        controller (the generators ``gen_ids``) is held at its own min_q
        (``at_min``) / max_q, the bus they regulated floating; and, under
        ``"vc_release"``, ``{gen_id: group}`` the frozen remote regulators
        released into their VoltageControl group."""
        if self._rl_result is None:
            raise RuntimeError(
                "get_outer_loop_switches() needs a compute() made with "
                "reactive_limits_outer_loop=True.")
        ctx = self._rl_ctx

        def _pins(sw):
            return {g: (at_min, tuple(int(ctx.vc_ctrl_elem[j]) for j in ctx.vc_grp_ctrls[g]))
                    for g, at_min in sw.vc_pin.items()}
        return [None if sw is None else {"to_pq": dict(sw.to_pq), "to_pv": dict(sw.to_pv),
                                         "vc_pin": _pins(sw),
                                         "vc_release": {int(g): int(ctx.vc_ctrl_group[j])
                                                        for j, g in sw.vc_release.items()}}
                for sw in self._rl_result["switches"]]

    @property
    def outer_loop_info(self):
        """dict about the last second pass (None before one): ``n_rows`` the
        contingencies it re-solved, ``time_s`` its wall time (decision +
        solve + merge), ``timings`` its session's BatchTimings (None when no
        row was re-solved)."""
        if self._rl_result is None:
            return None
        return {k: self._rl_result[k] for k in ("n_rows", "time_s", "timings")}

    def _second_pass_sweep(self):
        from ._reactive_limits import build_context
        if self._rl_sweep is None:
            from ..scenario_sweep.gpu_facade import ScenarioSweepGPU
            if self._grid is None:
                raise RuntimeError(
                    "reactive_limits_outer_loop needs a lightsim2grid grid (this session was "
                    "built from an explicit-array tuple).")
            grid = self._hold_frozen_grid()
            sweep = ScenarioSweepGPU(grid, compute_physical_violations=True,
                                     **self._rl_ctor_kwargs)
            self._rl_ctx = build_context(grid, sweep)
            self._rl_sweep = sweep
        return self._rl_sweep

    def _hold_frozen_grid(self):
        """The grid of the second pass: a copy with lightsim2grid's
        ``set_hold_frozen_regulators`` on when it has a frozen remote regulator
        (one the release of which a row may then ask for), else the grid
        itself. Same solution either way."""
        grid = self._grid
        if not (getattr(_cpp, "have_ls2g_hold_frozen", False)
                and hasattr(grid, "set_hold_frozen_regulators")):
            return grid
        if grid.get_hold_frozen_regulators():
            return grid
        if not any(g.connected and g.can_be_pv and not g.voltage_regulator_on
                   and g.regulated_bus_id != g.bus_id for g in grid.get_generators()):
            return grid
        held = grid.copy()
        held.set_hold_frozen_regulators(True)
        V = held.ac_pf(np.asarray(grid.get_V()).copy(), int(self._rl_ctor_kwargs["max_iter_base"]),
                       float(self._rl_ctor_kwargs["tol_base"]))
        if V.shape[0] == 0:
            return grid
        return held

    def _sync_second_pass(self, sweep):
        """The mutable settings of this analysis, mirrored on the second pass."""
        src, dst = self._inner, sweep.solver
        dst.nb_iter = src.nb_iter if self._outer_loop_nb_iter is None else self._outer_loop_nb_iter
        dst.strategy = src.strategy
        dst.refactor_period = src.refactor_period
        dst.nb_iter_per_round = src.nb_iter_per_round
        dst.tol = src.tol
        dst.scheduling = src.scheduling
        dst.handle_disconnected_grid = src.handle_disconnected_grid
        if src.compute_limit_violations and not dst.compute_limit_violations:
            sweep.set_limits_from_grid()
        dst.compute_limit_violations = src.compute_limit_violations
        dst.violation_tol = src.violation_tol
        dst.violation_rel_tol = src.violation_rel_tol
        dst.violation_capacity = src.violation_capacity
        sweep.physical_violation_tol_mva = self.physical_violation_tol_mva
        sweep.physical_violation_tol_vm_pu = self.physical_violation_tol_vm_pu
        sweep.physical_violation_capacity = self.physical_violation_capacity
        if sweep.redistribute_slack != self.redistribute_slack:
            sweep.redistribute_slack = self.redistribute_slack
        sweep.reference_slack = self.reference_slack

    def _run_outer_loop(self, batch_size):
        import time
        from ._reactive_limits import (plan_switches, batch_rule, run_second_pass,
                                       held_release_offsets, fix_held_release_records)
        beg = time.perf_counter()
        if not self.compute_physical_violations:
            raise RuntimeError(
                "reactive_limits_outer_loop needs compute_physical_violations=True (the "
                "switches are read off the physical checks).")
        if self._ctg_branch_ids is None:
            raise RuntimeError("reactive_limits_outer_loop: no contingency was added "
                               "through add_contingencies_by_branch_id().")
        sess = self._inner._s
        res = self.last_residuals()
        conv = np.isfinite(res) & (res <= self._inner.violation_tol)
        sweep = self._second_pass_sweep()
        ctx = self._rl_ctx
        status, switches = plan_switches(ctx, sess.get_bus_q_violations(),
                                         sess.get_gen_pv_release_violations(), conv)
        cand = [r for r, sw in enumerate(switches) if sw is not None]
        if self.scheduling == "continuous":
            # no chunk, so no last partial chunk to leave out: every row re-solved
            run_rows, left = cand, []
        else:
            run_rows, left = batch_rule(cand, batch_size, self._outer_loop_min_last_chunk)
        status[left] = int(ReactiveLimitsStatus.LEFT_OUT)
        out = {"status": status, "n_rows": len(run_rows), "timings": None, "switches": switches,
               "viol": {}, "viol_trunc": {}, "viol_counts": {}, "phys": {}, "phys_trunc": {}}
        if run_rows:
            self._sync_second_pass(sweep)
            v_init = (sess.v_results_ptr(), list(map(int, run_rows))) if self._outer_loop_warm_start else None
            run_second_pass(sweep, ctx, [self._ctg_branch_ids[r] for r in run_rows],
                            [switches[r] for r in run_rows], batch_size, v_init=v_init)
            out["timings"] = sweep.timings
            res2 = sweep.last_residuals()
            conv2 = np.isfinite(res2) & (res2 <= self._inner.violation_tol)
            ok = np.flatnonzero(conv2)
            status[np.asarray(run_rows)[~conv2]] = int(ReactiveLimitsStatus.DIVERGED)
            dst = [int(run_rows[i]) for i in ok]
            status[dst] = int(ReactiveLimitsStatus.RECOMPUTED)
            s2 = sweep.solver._s
            sess.overwrite_rows(dst, [int(i) for i in ok], s2.v_results_ptr(), s2.residuals_ptr())
            # ... and their outcome: the second pass' iterations and status
            sess.overwrite_row_outcomes(dst, [int(v) for v in np.asarray(s2.get_row_iterations())[ok]],
                                        [int(v) for v in np.asarray(s2.get_row_status())[ok]])
            phys = sweep.get_physical_violations()
            phys_tr = sweep.get_physical_violations_truncated()
            viol = sweep.get_violations() if sweep.solver.compute_limit_violations else None
            if viol is not None:
                viol_tr = sweep.get_violations_truncated()
                viol_cnt = sweep.get_violation_counts()
            offsets = held_release_offsets(ctx, [switches[r] for r in run_rows])
            for i, r in zip(ok, dst):
                out["phys"][r] = fix_held_release_records(phys[i], offsets[i])
                out["phys_trunc"][r] = bool(phys_tr[i])
                if viol is not None:
                    out["viol"][r] = viol[i]
                    out["viol_trunc"][r] = bool(viol_tr[i])
                    out["viol_counts"][r] = {k: int(v[i]) for k, v in viol_cnt.items()}
        out["time_s"] = time.perf_counter() - beg
        self._rl_result = out

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

    @property
    def violation_rel_tol(self):
        """float: relative margin a value must clear past its limit to be
        reported by :attr:`compute_limit_violations` -- lightsim2grid's
        ``violation_rel_tol``, same default (``1e-9``) and semantics:
        CURRENT when ``ka > limit * (1 + tol)``, HIGH_VOLTAGE when
        ``v > vmax * (1 + tol)``, LOW_VOLTAGE when ``v < vmin * (1 - tol)``.
        It keeps a value that sits ON its limit by construction (a bus a
        regulator holds exactly at its vmax) from being reported or not
        depending on the last bit of the solve -- which made the GPU, the
        lightsim2grid batch and its one-off solve disagree. ``0`` gives the
        bare strict comparisons. Also applies to :meth:`get_violations_n`.
        In [0, 1[; takes effect on the next compute(). An FP32 build cannot
        resolve 1e-9 (use ~1e-6 there)."""
        return self._inner.violation_rel_tol

    @violation_rel_tol.setter
    def violation_rel_tol(self, value):
        self._inner.violation_rel_tol = value

    def get_violations(self):
        """list[list[LimitViolation]]: one entry per contingency (row order
        matches add_contingencies_by_branch_id). Requires compute() with
        compute_limit_violations=True. With reactive_limits_outer_loop, a
        RECOMPUTED contingency's entry is its second pass'."""
        out = self._inner.get_violations()
        if self._rl_result is not None:
            for r, v in self._rl_result["viol"].items():
                out[r] = v
        return out

    def get_violations_truncated(self):
        """(n_ctg,) bool ndarray: True where more than violation_capacity
        violations were found for that contingency (clamped)."""
        out = self._inner.get_violations_truncated()
        if self._rl_result is not None:
            for r, v in self._rl_result["viol_trunc"].items():
                out[r] = v
        return out

    def get_violation_counts(self):
        """dict of (n_ctg,) int ndarrays with keys 'low_voltage',
        'high_voltage', 'current': the TRUE, uncapped count of violations of
        each type per contingency (-1 = not simulated). Unlike
        get_violations()'s records (capped at violation_capacity), these
        totals stay exact even when get_violations_truncated() is True."""
        out = self._inner.get_violation_counts()
        if self._rl_result is not None:
            for r, cnt in self._rl_result["viol_counts"].items():
                for k, v in cnt.items():
                    out[k][r] = v
        return out

    def get_physical_violations(self):
        """See :meth:`PhysicalChecksEngineMixin.get_physical_violations`. With
        reactive_limits_outer_loop, a RECOMPUTED contingency's entry is its
        second pass' (with the new labels, see ``_reactive_limits.py``)."""
        out = self._inner.get_physical_violations()
        if self._rl_result is not None:
            for r, v in self._rl_result["phys"].items():
                out[r] = v
        return out

    def get_physical_violations_truncated(self):
        """(n_ctg,) bool ndarray: True where one of the physical checks kept
        only its most severe records on that row."""
        out = self._inner.get_physical_violations_truncated()
        if self._rl_result is not None:
            for r, v in self._rl_result["phys_trunc"].items():
                out[r] = v
        return out

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
        branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, bus_vn_kv, sn_mva = branch_args
        V_n = grid.get_V_solver()

        return compute_violations_n(
            V_n, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
            branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
            limit_a1_ka, limit_a2_ka, sn_mva, n_lines,
            rel_tol=self._inner.violation_rel_tol)

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
