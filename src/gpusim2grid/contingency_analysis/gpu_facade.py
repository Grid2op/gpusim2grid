# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU contingency analysis driven directly from a lightsim2grid grid.

``ContingencyAnalysisGPU`` is the GPU sibling of lightsim2grid's
``ContingencyAnalysis(grid)``: it takes the solved grid object, reuses the
CPU base-case (N) power flow, and runs the batched contingencies on the GPU.
It is a thin, batch-oriented facade over :class:`_ContingencyAnalysisSolver`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._scheduling import RowStatus, check_linear_solver_args
from . import (
    DeviceBuffer,
    LimitViolationsFacadeMixin,
    _ContingencyAnalysisSolver,
    PhysicalChecksFacadeMixin,
    SlackRedistributionFacadeMixin,
    SchedulingFacadeMixin,
    _normalize_device,
    _resolve_reordering_alg,
    _resolve_matching_alg,
    _resolve_pivot_epsilon_alg,
    _check_ir_n_steps,
)
from .._ls2g_utils import (
    extract_grid_arrays,
    extract_branch_data,
    grid_from_pandapower,
    _validate_precision,
)
from .. import _gpusim2grid as _cpp
from .._outer_loops import _check_outer_loops_args
from ._reactive_limits import ReactiveLimitsStatus
from ..actions import current_buses
from ._strategies import (RowBranch, RowMap, StrategyCondition, TopologyContext, UnionPlan, compile_action, plan_union,
                          strategy_row, topology_context)
from ._violation_threshold import ViolationThresholdFacadeMixin

if TYPE_CHECKING:
    from ..actions import Action
    from ._limit_violations import LimitViolation
    from ._strategies import CompiledAction, CtgElements, Strategy

#: what ``compute(rows=...)`` runs


def _check_reactive_limits_outer_loop(value):
    if bool(value) != value:
        raise ValueError("The `reactive_limits_outer_loop` attribute must be a boolean.")
    return bool(value)


def _warn_reactive_limits_outer_loop_deprecated():
    # stacklevel 3: the caller of the constructor / of the attribute assignment
    warnings.warn(
        "reactive_limits_outer_loop is deprecated and will be removed in a future version: "
        "use outer_loops=True instead, which runs lightsim2grid's ReactiveLimits outer loop "
        "(PV <-> PQ) on every row until it is stable, with the other default outer loops.",
        DeprecationWarning, stacklevel=3)


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
                             SchedulingFacadeMixin, LimitViolationsFacadeMixin,
                             ViolationThresholdFacadeMixin):
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
    ir_n_steps : int, default 0
        cuDSS ``CUDSS_CONFIG_IR_N_STEPS``: steps of iterative refinement after
        each linear solve, base case and batch (forward and adjoint). ``0``
        (default) solves without refinement. Static pivoting can leave the
        Jacobian's factors a little inexact, which Newton usually absorbs.
        **With** ``outer_loops=True`` **use 1**: each inner solve must stop at
        the iteration lightsim2grid's does, and a hard one (e.g. after
        TransformerVoltageControl releases a held remote voltage controller)
        can diverge without refinement where lightsim2grid's KLU converges.
        One step costs one more residual and solve per Newton iteration. Mutable:
        takes effect on the next :meth:`compute` (the base case keeps the
        construction value).
    linear_solver : {'cudss', 'own_lu'}, default 'cudss'
        Who factorizes and solves the batch Jacobians (forward, continuous tail,
        adjoint); ``'own_lu'`` is the batched LU of our own, its order chosen on
        the host from ``reordering_alg`` (lightsim2grid's constrained order by
        default). See :attr:`linear_solver`; mutable, next run.
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
        .. deprecated:: 0.3.0
            Use ``outer_loops=True``: lightsim2grid's own outer loops,
            ``ReactiveLimits`` (PV <-> PQ) among them, run on every row until
            they are stable.

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
    scheduling : {"chunked", "continuous"} or None, default None
        ``None``: ``"chunked"``, or ``"continuous"`` with ``outer_loops`` or with strategies
        released by their contingency (``strategy_condition``, ``strategy_warm_start``).
        How the rows go through the batch. ``"chunked"``: ``ceil(n_rows /
        batch_size)`` chunks, every row runs exactly ``nb_iter`` Newton
        iterations. ``"continuous"``: ``batch_size`` slots; every
        ``nb_iter_per_round`` iterations each row is checked, and one that
        converged (``||F||inf < tol``), diverged or used its ``nb_iter``
        budget leaves at once, its slot refilled from the queue -- a row pays
        the iterations it needs, not the hardest row's. ``nb_iter`` is then
        each row's budget (a row runs a multiple of ``nb_iter_per_round``, at
        least one round). Continuous refuses the ``'direct_iter0_only'`` /
        ``'direct_refactor_every_n'`` strategies. With ``outer_loops``, only
        ``"continuous"`` is supported, by design (not planned to change): a
        row the loops change is re-solved in its own slot while the others
        go on, which a chunk cannot do. Mutable.
    nb_iter_per_round : int, default 1
        Continuous scheduling: iterations between two convergence checks. A
        check (one SpMV + mismatch) is cheap next to a refactorization, so 1
        is usually fastest: on case6515rte N-1, 1 / 2 / 4 took 6.7 / 8.6 /
        9.2 s. With ``outer_loops``, only 1 is supported, by design (not
        planned to change): each inner solve must stop at the iteration
        lightsim2grid's does, so that the loops decide on the same state.
        Mutable.
    tol : float or None, default None
        A row has converged when ``||F||inf < tol``, per unit like
        :meth:`last_residuals`: when it leaves (continuous) and its
        :class:`RowStatus` (both schedules, :meth:`get_row_status`).
        lightsim2grid compares the same way but takes its ``tol`` in MVA
        (``||F||inf < tol / sn_mva``). ``None`` = 1e-8 (1e-3 in an FP32
        build). Mutable.
    outer_loops, keep_outer_trace : bool, default False
        Run the grid's own outer loops (lightsim2grid's ``NROuter_*`` / ``NROuterDist_*``) on every
        row; see :class:`ScenarioSweepGPU`. Requires ``scheduling="continuous"`` and
        ``nb_iter_per_round=1``, by design (neither is planned to change); pass
        ``ir_n_steps=1`` with it (see there). Each contingency keeps the grid's targets. With
        ``init_from_n_powerflow`` (default) the rows resume from where the
        grid's own solve left its loops. :meth:`get_outer_results`.
    violation_threshold : float in ``]0, 1]``, default 1.0
        lightsim2grid's knob of the same name and semantics: every checked limit moves towards a
        healthy anchor, ``t * limit + (1 - t) * anchor`` (0 for currents, the bus nominal voltage
        clamped into ``[vmin, vmax]`` for voltages); records keep the configured limit.
    strategy_condition : StrategyCondition, default StrategyCondition.TRUE
        When a contingency's strategies (:meth:`add_strategies`) run, pypowsybl's condition:
        all of them, or only when the contingency violates a limit (``ANY_VIOLATION``:
        ``compute_limit_violations`` on, ``violation_threshold`` applied), checked on the device
        as the contingency finishes. Its string value converts.
    strategy_warm_start : bool, default True
        A strategy row starts from its own contingency's converged state, in the same
        :meth:`compute` (the contingency finishing first): its voltages and, with
        ``outer_loops``, where its loops ended (moved targets, frozen buses, switched SVCs, hvdc
        regimes, taps and sections), as a second solve on the grid with the contingency applied
        whose loss its DistributedSlack does not share again. A contingency that did not
        converge (or whose loops did not end stable) leaves its strategies the base start.
        False: every strategy from the base case. ``"ANY_VIOLATION"`` and the warm start
        release a strategy when its contingency finishes: they need
        ``scheduling="continuous"`` (``None`` picks it when there are strategies).

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
                 matching_alg=None, pivot_epsilon_alg=None, ir_n_steps=0,
                 linear_solver="cudss", debug_base_case=False,
                 scaling_max_voltage_change=None, max_dVa=None, max_dVm=None,
                 use_distributed_slack=True,
                 compute_physical_violations=False, redistribute_slack=False,
                 reference_slack="auto", reactive_limits_outer_loop=False,
                 outer_loop_min_last_chunk=250, outer_loop_warm_start=True,
                 outer_loop_nb_iter=None, scheduling=None, nb_iter_per_round=1,
                 tol=None, outer_loops=False, keep_outer_trace=False, violation_threshold=1.0,
                 strategy_condition=StrategyCondition.TRUE, strategy_warm_start=True):
        _validate_precision(precision)
        check_linear_solver_args(linear_solver, reordering_alg, matching_alg)
        self._init_violation_threshold(violation_threshold)
        strategy_condition = StrategyCondition(strategy_condition)   # ValueError if not one
        self._outer_loops = bool(outer_loops)
        # a strategy released by its contingency (the condition, the warm start) needs the slots:
        # scheduling=None picks them, at the constructor for the condition, at the first compute()
        # with strategies for the warm start (a plain N-1 keeps its chunked default)
        self._scheduling_auto: bool = scheduling is None
        if scheduling is None:
            scheduling = ("continuous" if self._outer_loops or strategy_condition is not StrategyCondition.TRUE
                          else "chunked")
        if strategy_condition is not StrategyCondition.TRUE and scheduling != "continuous":
            raise ValueError("strategy_condition=ANY_VIOLATION releases a contingency's strategies as it "
                             "finishes: it needs scheduling='continuous'")
        if self._outer_loops:
            _check_outer_loops_args(grid, use_bridge, scheduling, nb_iter_per_round,
                                    compute_physical_violations=compute_physical_violations,
                                    reactive_limits_outer_loop=reactive_limits_outer_loop)
        # the second pass of reactive_limits_outer_loop is a ScenarioSweepGPU
        # built (lazily) from the same grid with the same construction options
        self._rl_ctor_kwargs = dict(
            init_from_n_powerflow=init_from_n_powerflow, precision=precision,
            nb_iter=nb_iter, max_iter_base=max_iter_base, tol_base=tol_base,
            device=device, use_bridge=use_bridge, reordering_alg=reordering_alg,
            matching_alg=matching_alg, pivot_epsilon_alg=pivot_epsilon_alg,
            ir_n_steps=ir_n_steps,
            debug_base_case=debug_base_case,
            scaling_max_voltage_change=scaling_max_voltage_change,
            max_dVa=max_dVa, max_dVm=max_dVm, use_distributed_slack=use_distributed_slack)
        self._rl_sweep = None
        # the last limits given (_set_limits; None = the bridge's, off the grid)
        # and the generation of the ones the second pass has (_sync_second_pass)
        self._limits_args = None
        self._limits_gen = 0
        self._limits_synced = -1
        self._rl_ctx = None
        self._rl_result = None
        self._ctg_branch_ids = None
        self._ctg_elements = None
        if reactive_limits_outer_loop:
            _warn_reactive_limits_outer_loop_deprecated()
        self._reactive_limits_outer_loop = _check_reactive_limits_outer_loop(reactive_limits_outer_loop)
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
        _ir_n_steps = _check_ir_n_steps(ir_n_steps)

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
                ir_n_steps=_ir_n_steps,
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

            if self._outer_loops:
                # the grid's own outer loops (see ScenarioSweepGPU's outer_loops);
                # every contingency keeps the grid's targets
                session = _cpp._make_ca_outer_session_from_lsgrid(
                    grid, 100, int(nb_iter), _normalize_device(device),
                    bool(compute_limit_violations),
                    reordering_alg=_resolve_reordering_alg(_reordering_alg),
                    matching_alg=_resolve_matching_alg(_matching_alg),
                    pivot_epsilon_alg=_resolve_pivot_epsilon_alg(_pivot_epsilon_alg),
                    ir_n_steps=_ir_n_steps,
                    keep_trace=bool(keep_outer_trace),
                    resume_from_n=bool(init_from_n_powerflow),
                    # None => the grid's own step scaling, as without the outer loops
                    scaling_max_voltage_change_override=(
                        -1 if scaling_max_voltage_change is None
                        else int(bool(scaling_max_voltage_change))),
                    max_dVa_override=-1.0 if max_dVa is None else float(max_dVa),
                    max_dVm_override=-1.0 if max_dVm is None else float(max_dVm))
                self._inner = _ContingencyAnalysisSolver._wrap_session(
                    session, max_iter_base=max_iter_base, tol_base=tol_base,
                    reordering_alg=_reordering_alg, matching_alg=_matching_alg,
                    pivot_epsilon_alg=_pivot_epsilon_alg)
                self._n_branches = self._inner._s.n_branches
            elif use_bridge:
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
                    ir_n_steps=_ir_n_steps,
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
                    ir_n_steps=_ir_n_steps,
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
        # Who factorizes / solves the batch Jacobians, see
        # SchedulingFacadeMixin.linear_solver (the base case stays on cuDSS).
        self._inner.linear_solver = linear_solver
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

        # topology actions (docs/dev_notes/topology_actions_plan.md, step A): actions,
        # strategies (contingency index, action indices), the condition and warm start that
        # release them, and the map from the rows the session solved to the caller's
        self._actions: list[Action] = []
        self._compiled: list[CompiledAction] = []
        self._strategies: list[Strategy] = []
        self._rowmap: RowMap | None = None
        self._flows_done: bool = False
        self._strategy_warm_start: bool = bool(strategy_warm_start)
        self._strategy_condition: StrategyCondition = strategy_condition
        # the union pattern the session holds for the bus moves (None: the base pattern)
        self._union: UnionPlan | None = None
        # the rows of the last run whose actions change branch values: {row: {branch: new y}}
        # per caller row, the branches whose admittance or ends it changes: (y, from, to), solver
        # buses (step B.1 values, B.2 moves) -- their flows and current checks are the host's
        self._row_branch: dict[int, dict[int, RowBranch]] = {}
        self._branch_args: tuple[Any, ...] | None = None

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
        self._set_limits(bus_vmin_kv, bus_vmax_kv, branch_limit_a1_ka,
                         branch_limit_a2_ka, int(n_lines))

    def _set_limits(self, *args):
        """Every limit given to this analysis goes through here: the second pass
        of reactive_limits_outer_loop is given the same ones (_sync_second_pass)."""
        self._inner.set_limits(*args)
        self._limits_args = args
        self._limits_gen += 1

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
        # a branch named twice is tripped once (the session does the same)
        self._ctg_branch_ids = [list(dict.fromkeys(map(int, ids))) for ids in branch_ids_per_ctg]
        self._ctg_elements = [(list(ids),) + tuple([] for _ in range(9)) for ids in self._ctg_branch_ids]

    # ------------------------------------- shunts, loads and generators tripped
    @staticmethod
    def _as_ids(ids):
        if ids is None:
            return []
        if isinstance(ids, (int, np.integer)):
            return [int(ids)]
        return sorted(set(int(el) for el in ids))

    def _push_contingencies(self, elements):
        """Hand `elements` (the whole list) to the session, and keep it only once the
        session took it: a refused list (an id out of range, an element this session
        has no data for) leaves the facade's own, and the session's, as they were."""
        b, s, l, g, b1, b2, sg, st, sv, hv = ([list(c[k]) for c in elements] for k in range(10))
        if any(sg) or any(st) or any(sv) or any(hv):
            self._inner.build_contingencies_full(b, s, l, g, b1, b2, sg, st, sv, hv)
        elif any(b1) or any(b2):
            self._inner.build_contingencies_full(b, s, l, g, b1, b2)
        elif any(s) or any(l) or any(g):
            self._inner.build_contingencies_full(b, s, l, g)
        else:
            self._inner.build_contingencies(b)
        self._ctg_elements = elements
        self._ctg_branch_ids = b

    def add_contingencies(self, contingencies):
        """Append contingencies, each a tuple ``(branch_ids, shunt_ids, load_ids,
        gen_ids)`` tripping those elements together (lightsim2grid's
        ``ContingencyAnalysisCPP.add_contingency``), optionally followed by
        ``branch_ids_side1, branch_ids_side2``, then by ``sgen_ids, storage_ids, svc_ids,
        hvdc_ids`` (the ten lists of lightsim2grid's ``my_contingencies()``). Any of them
        may be empty, not all.
        Appended to what :meth:`add_contingencies_by_branch_id` (which replaces
        the whole list) or a previous ``add_contingenc*`` registered.

        - a shunt leaves the contingency's admittance matrix (its diagonal
          coefficient);
        - a load leaves its injection (its active and reactive power);
        - a generator too (its active power, and the reactive setpoint of one that does
          not regulate voltage), and the distributed slack is re-weighted without it;
          when it was the last one regulating its own bus, that bus turns PQ for the
          contingency. A controller of a voltage-control group (regulating a remote bus,
          or on a bus a group holds) leaves its group, the regulated bus going free when
          nobody is left; such a contingency is reported BATCH_UNSUPPORTED (see
          :meth:`get_batch_unsupported_reasons`) with ``compute_physical_violations``,
          when the controller is held at a reactive limit (lightsim2grid's held
          regulators), or when it takes out every controller of a group regulating a
          slack bus;
        - a branch of ``branch_ids_side1`` (``branch_ids_side2``) is opened at side 1 (2)
          only: its other end stays connected and keeps the branch's charging, as
          OpenLoadFlow models a branch whose breaker opens at one end. Its flows read 0,
          its current is not checked and the outer loops see it as open. Named at both
          sides, or also in ``branch_ids``, it is opened at both;
        - a static generator leaves its injection;
        - a storage unit too, and its share of the distributed slack, the last controller
          of a bus turning it PQ; one on a bus a control group holds is BATCH_UNSUPPORTED;
        - a static var compensator: its fixed reactive injection and its b0; one in voltage
          mode leaves its control group; one with a standby automaton leaves its automaton's
          check and the VoltageMonitoring loop (lightsim2grid leaves that row out); one
          frozen at a limit is BATCH_UNSUPPORTED;
        - an HVDC line stops transmitting, both converter stations leaving the injection:
          a line in AC emulation carries no flow any more, a station in a control group
          leaves it, and the bus a regulating station was the last to hold turns PQ.

        With ``compute_physical_violations``, the elements those checks cannot leave out of
        a row yet are BATCH_UNSUPPORTED: a storage unit they check, an SVC in voltage mode,
        an HVDC line in AC emulation or with a regulating station. With the outer loops, a
        shunt that regulates a voltage is BATCH_UNSUPPORTED (its sections are the
        ShuntVoltageControl loop's). Needs a grid-mode instance (the element data comes
        from the lightsim2grid grid).
        """
        if getattr(self, "_ctg_elements", None) is None:
            self._ctg_elements = []
        new = []
        for cont in contingencies:
            if len(cont) not in (4, 6, 10):
                raise ValueError("each contingency must be (branch_ids, shunt_ids, load_ids, gen_ids), "
                                 "optionally followed by (branch_ids_side1, branch_ids_side2), then "
                                 "(sgen_ids, storage_ids, svc_ids, hvdc_ids)")
            cont = [self._as_ids(el) for el in cont] + [[] for _ in range(10 - len(cont))]
            # opened at both ends one way or another: a trip (as lightsim2grid has it)
            both = set(cont[0]) | (set(cont[4]) & set(cont[5]))
            cont[0] = sorted(both)
            cont[4] = [el for el in cont[4] if el not in both]
            cont[5] = [el for el in cont[5] if el not in both]
            cont = tuple(cont)
            if not any(cont):
                raise ValueError("a contingency must trip something (every list is empty)")
            new.append(cont)
        self._push_contingencies(self._ctg_elements + new)

    def add_contingency(self, branch_ids=(), shunt_ids=(), load_ids=(), gen_ids=(),
                        branch_ids_side1=(), branch_ids_side2=(), sgen_ids=(), storage_ids=(),
                        svc_ids=(), hvdc_ids=()):
        """Append one contingency tripping these elements together; see
        :meth:`add_contingencies`."""
        self.add_contingencies([(branch_ids, shunt_ids, load_ids, gen_ids, branch_ids_side1, branch_ids_side2,
                                 sgen_ids, storage_ids, svc_ids, hvdc_ids)])

    def add_contingency_shunt(self, shunt_ids):
        """Append one contingency tripping these shunts only (one id, or several)."""
        self.add_contingencies([((), shunt_ids, (), ())])

    def add_contingency_load(self, load_ids):
        """Append one contingency tripping these loads only (one id, or several)."""
        self.add_contingencies([((), (), load_ids, ())])

    def add_contingency_gen(self, gen_ids):
        """Append one contingency tripping these generators only (one id, or several)."""
        self.add_contingencies([((), (), (), gen_ids)])

    def add_contingency_sgen(self, sgen_ids):
        """Append one contingency tripping these static generators only (one id, or several)."""
        self.add_contingency(sgen_ids=sgen_ids)

    def add_contingency_storage(self, storage_ids):
        """Append one contingency tripping these storage units only (one id, or several)."""
        self.add_contingency(storage_ids=storage_ids)

    def add_contingency_svc(self, svc_ids):
        """Append one contingency tripping these static var compensators only (one id, or several)."""
        self.add_contingency(svc_ids=svc_ids)

    def add_contingency_hvdc(self, hvdc_ids):
        """Append one contingency tripping these HVDC lines only (one id, or several)."""
        self.add_contingency(hvdc_ids=hvdc_ids)

    def get_batch_unsupported_reasons(self):
        """list[str], per contingency: why the batch left it out (``""``: it did not), after
        :meth:`compute`. Such a row has NaN voltages and a ``BATCH_UNSUPPORTED`` violation."""
        if self._rowmap is None:
            return self._inner.get_batch_unsupported_reasons()
        rm = self._rowmap
        out = list(rm.reasons)
        if len(rm.session_rows):
            for i, why in enumerate(self._inner.get_batch_unsupported_reasons()):
                out[int(rm.session_rows[i])] = why
        return out

    def get_contingencies(self):
        """Every contingency registered, in row order: ``(branch_ids, shunt_ids,
        load_ids, gen_ids, branch_ids_side1, branch_ids_side2, sgen_ids, storage_ids,
        svc_ids, hvdc_ids)``."""
        return [tuple(list(el) for el in c) for c in (getattr(self, "_ctg_elements", None) or [])]

    @property
    def outer_loops(self):
        """bool: whether the grid's own outer loops run on every contingency (see
        :class:`ScenarioSweepGPU`'s ``outer_loops``; each contingency keeps the
        grid's targets)."""
        return self._outer_loops

    def get_outer_results(self):
        """Per contingency, the outcome of its outer loops (see
        :meth:`ScenarioSweepGPU.get_outer_results`). Needs ``outer_loops=True``."""
        if not self._outer_loops:
            raise RuntimeError("get_outer_results() needs outer_loops=True")
        return self._session_rows_or_empty(self._inner._s.get_outer_results, lambda: None)

    def get_row_status(self) -> np.ndarray:
        """(n_rows,) int ndarray of :class:`RowStatus` codes of the last :meth:`compute`, on
        its rows (contingencies, then strategies); NOT_SIMULATED for a strategy not run (its
        contingency failed :attr:`strategy_condition`, or unsupported: see
        :meth:`get_batch_unsupported_reasons`)."""
        return self._session_array(self._inner.get_row_status, int(RowStatus.NOT_SIMULATED))

    def get_row_iterations(self) -> np.ndarray:
        """(n_rows,) int ndarray: the Newton iterations each row of the last :meth:`compute`
        ran, on its rows; 0 for a strategy not run."""
        return self._session_array(self._inner.get_row_iterations, 0)

    def compute(self, batch_size: int = 512) -> Any:
        """Solve every contingency and return the batched voltages.

        Returns a DLPack capsule of shape ``(n_contingencies, n_bus)``,
        complex, aliasing live GPU memory.  Pass to ``torch.from_dlpack`` /
        ``jax.dlpack.from_dlpack``; clone before the next ``compute()`` for a
        snapshot.  Residuals are cached for :meth:`last_residuals`.

        With :attr:`reactive_limits_outer_loop`, the contingencies it re-solves
        have their second-pass voltages and residuals in these buffers (see
        :meth:`get_outer_loop_status`).

        With strategies (:meth:`add_strategies`), one call runs the contingencies and their
        strategies: a contingency releases its strategies when it finishes, if
        :attr:`strategy_condition` holds, each starting from the contingency's state with
        :attr:`strategy_warm_start` (both need ``scheduling="continuous"``). The results cover
        the contingencies, then the strategies (NaN for a strategy that did not run), and the
        voltages come back as :attr:`V_results` (``(n_rows * n_bus,)``, copied to the host only
        by its ``to_numpy()``).
        """
        self._push_effective_limits()
        self._flows_done = False
        if self._strategies:
            return self._compute_rows(int(batch_size))
        self._rowmap = None
        self._inner.batch_size = int(batch_size)
        self._rl_result = None
        self._inner.run()
        self._last_residuals = self._inner.residuals
        if self._reactive_limits_outer_loop:
            self._run_outer_loop(int(batch_size))
        return self._inner.v_results_dlpack()


    # ------------------------------------------------------ topology actions
    # docs/dev_notes/topology_actions_plan.md, step A: strategy rows (a contingency and
    # actions), streamed after their contingency, max loading, violation_threshold
    def set_actions(self, actions: Sequence[Action]) -> None:
        """The actions strategies refer to (:class:`gpusim2grid.actions.Action`, by index).
        Validated against the grid; replaces the previous list (the strategies keep their
        indices, which must stay in range)."""
        from ..actions import Action

        if self._grid is None:
            raise RuntimeError("set_actions needs a grid-mode instance (a lightsim2grid grid)")
        actions = list(actions)
        for i, a in enumerate(actions):
            if not isinstance(a, Action):
                raise TypeError(f"actions[{i}] is a {type(a).__name__}, not a gpusim2grid.actions.Action")
            a.validate(self._grid)
        n_lines: int = len(self._grid.get_lines())
        if any(max(acts, default=-1) >= len(actions) for _, acts in self._strategies):
            raise ValueError("a strategy refers to an action index the new list does not have")
        self._actions = actions
        current = current_buses(self._grid)  # once: it walks every element end of the grid
        self._compiled = [compile_action(self._grid, a, n_lines, current) for a in actions]
        # branch ends (solver numbering), nominal voltages, sn_mva: row triplets and flow fix-ups
        self._branch_args = extract_branch_data(self._grid)[0]
        self._rowmap = None

    def add_strategies(
        self,
        strategies: Sequence[tuple[int, Sequence[int]]] | np.ndarray,
        action_idx: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        """Append strategies: ``[(ctg_index, [action indices]), ...]``, or two arrays
        ``(ctg_idx, action_idx)`` of one action each (the product without a Python call per
        row). A strategy's row is its contingency's plus its actions' disconnections."""
        if action_idx is not None:
            c = np.asarray(strategies, dtype=np.int64).ravel()
            a = np.asarray(action_idx, dtype=np.int64).ravel()
            if c.shape != a.shape:
                raise ValueError("ctg_idx and action_idx must have the same length")
            new: list[Strategy] = [(int(ci), (int(ai),)) for ci, ai in zip(c, a)]
        else:
            new = [(int(ci), tuple(int(x) for x in acts)) for ci, acts in strategies]
        n_ctg: int = len(self._ctg_elements or [])
        for ci, acts in new:
            if not 0 <= ci < n_ctg:
                raise ValueError(f"contingency index {ci} out of range [0, {n_ctg})")
            if not acts or not all(0 <= x < len(self._actions) for x in acts):
                raise ValueError(f"action indices {list(acts)} out of range [0, {len(self._actions)})")
        self._strategies += new
        self._rowmap = None

    @property
    def n_strategies(self) -> int:
        """int: the strategies registered (:meth:`add_strategies`)."""
        return len(self._strategies)

    @property
    def n_spare_buses(self) -> int:
        """int: buses the batch adds for bus moves (0: the base pattern, today's path)."""
        return self._union.n_spare if self._union is not None else 0

    @property
    def batch_bus_ids(self) -> np.ndarray:
        """(n_bus,) int ndarray: the global (lightsim2grid) bus id of every voltage column."""
        if self._grid is None:
            return np.arange(self.n_bus)
        base = np.asarray(self._grid.id_ac_solver_to_me(), dtype=np.int64)
        if self._union is None:
            return base
        return np.concatenate([base, np.asarray(self._union.spare_global, dtype=np.int64)])

    def get_strategy_ran(self) -> np.ndarray:
        """(n_strategies,) bool: the strategies the last run solved."""
        if self._rowmap is None:
            raise RuntimeError("no strategy run yet")
        return self._rowmap.ran[self._rowmap.n_out - len(self._strategies):]

    @property
    def strategy_condition(self) -> StrategyCondition:
        """StrategyCondition: when a contingency's strategies run (the constructor's)."""
        return self._strategy_condition

    @property
    def strategy_warm_start(self) -> bool:
        """bool: a strategy starts from its own contingency's converged state (the constructor's
        ``strategy_warm_start``)."""
        return self._strategy_warm_start

    def get_iterations(self) -> np.ndarray:
        """(n_rows,) int: Newton iterations of every row of the last run (-1: not run)."""
        it = np.asarray(self._inner._s.get_row_iterations(), dtype=np.int64) \
            if self._rowmap is None or len(self._rowmap.session_rows) else np.zeros(0, dtype=np.int64)
        return it if self._rowmap is None else self._rowmap.expand(it, -1)

    def get_max_loading(self) -> tuple[np.ndarray, np.ndarray]:
        """Per row of the last run: ``(rho, branch)``, the highest ``|I| / limit`` over the
        branches and their two sides (the grid's current limits, not the threshold's; NaN
        limits ignored) and the branch reaching it (lines then transformers). NaN / -1 for a
        row that was not solved (did not run, skipped as islanded, unsupported, NaN residual)
        or has no limit."""
        if self._last_residuals is None:
            raise RuntimeError("Call compute() before get_max_loading().")
        _, _, lim1, lim2, _ = self._configured_limits()
        if self.n_branches is None:
            raise RuntimeError("get_max_loading needs branch data (set_branch_data in array mode)")
        nb: int = self.n_branches
        rm = self._rowmap
        residuals: DeviceBuffer = self._last_residuals
        # reduce the rows the session solved only, then expand the two per-row results: the flows
        # of the caller's rows that did not run would be n_rows x n_branches of NaN
        n_session: int = len(rm.session_rows) if rm is not None else int(residuals.shape[0])
        if n_session == 0:
            assert rm is not None
            return np.full(rm.n_out, np.nan), np.full(rm.n_out, -1, dtype=np.int64)
        # reduced on the device (one rho and one branch per row come back); the branches whose
        # values or ends a row changes are left out there and taken from the row's own
        # admittance and buses here (the session's flows use the base ones)
        lim1_a: np.ndarray = np.where(np.asarray(lim1, dtype=float) > 0, np.asarray(lim1, dtype=float), np.nan)
        lim2_a: np.ndarray = np.where(np.asarray(lim2, dtype=float) > 0, np.asarray(lim2, dtype=float), np.nan)
        changed_rows: dict[int, dict[int, RowBranch]] = {}
        skip: list[int] = []
        if rm is not None and self._row_branch:
            pos: dict[int, int] = {int(r): k for k, r in enumerate(rm.session_rows)}
            for row, changed in self._row_branch.items():
                k = pos[row]
                changed_rows[k] = changed
                skip += [k * nb + b for b in changed]
        rho_d, branch_d = self._inner._s.compute_max_loading(lim1_a, lim2_a, sorted(skip))
        rho: np.ndarray = np.asarray(rho_d, dtype=float)
        branch: np.ndarray = np.asarray(branch_d, dtype=np.int64)
        if changed_rows:
            keys: list[int] = sorted(changed_rows)
            V: np.ndarray = np.asarray(self._inner._s.get_V_rows(keys)).reshape(len(keys), -1)
            for i, k in enumerate(keys):
                for b, rb in sorted(changed_rows[k].items()):
                    loads = [self._branch_amps(V[i], rb, side) * 1e-3 / lim
                             for side, lim in ((1, lim1_a[b]), (2, lim2_a[b])) if np.isfinite(lim)]
                    if loads and (not np.isfinite(rho[k]) or max(loads) > rho[k]
                                  or (max(loads) == rho[k] and b < branch[k])):
                        rho[k], branch[k] = max(loads), b
        # a row not solved (skipped, unsupported, NaN residual) has no loading
        bad: np.ndarray = ~np.isfinite(residuals.to_numpy())
        rho[bad] = np.nan
        branch[bad | ~np.isfinite(rho)] = -1
        if rm is None:
            return rho, branch
        return rm.expand(rho, np.nan), rm.expand(branch, -1)

    # ------------------------------------------------- topology actions, inside
    def _compute_rows(self, batch_size: int) -> DeviceBuffer:
        """Run the contingency rows and the supported strategy rows through the session in one
        run; map them back. A strategy that depends on its contingency (the warm start, a
        condition) is released by it as it finishes (the session's feed, continuous schedule)."""
        if self._reactive_limits_outer_loop:
            raise NotImplementedError("strategies with reactive_limits_outer_loop (deprecated): "
                                      "use outer_loops=True")
        stream: bool = self._strategy_warm_start or self._strategy_condition is not StrategyCondition.TRUE
        if stream and self.scheduling != "continuous":
            if not self._scheduling_auto:
                raise ValueError("strategy_warm_start (default True) / strategy_condition=ANY_VIOLATION "
                                 "release a strategy as its contingency finishes: they need "
                                 "scheduling='continuous' (or strategy_warm_start=False)")
            self.scheduling = "continuous"
        ctg: list[CtgElements] = list(self._ctg_elements or [])
        if not ctg:
            raise RuntimeError("add contingencies before compute()")
        n_lines: int = len(self._grid.get_lines()) if self._grid is not None else 0
        topo: TopologyContext | None = self._prepare_union(n_lines)
        n_ctg: int = len(ctg)
        n_out: int = n_ctg + len(self._strategies)
        reasons: list[str] = [""] * n_out
        # the contingencies are session rows 0..n_ctg-1: a strategy's start is its contingency's row
        session_rows: list[int] = list(range(n_ctg))
        session_list: list[CtgElements] = list(ctg)
        # per session row, its Ybus value changes (the contingency rows have none)
        session_triplets: list[list[tuple[int, int, float, float]]] = [[] for _ in ctg]
        # per session row, the buses it leaves empty (always masked; the session masks the spares
        # a row does not use) and its Sbus moves
        session_empty: list[list[int]] = [[] for _ in ctg]
        session_sbus: list[list[tuple[int, complex]]] = [[] for _ in ctg]
        # per session row, the voltage control of its moved generators
        session_release: list[list[int]] = [[] for _ in ctg]
        session_pv: list[list[tuple[int, float]]] = [[] for _ in ctg]
        session_gen_bus: list[list[tuple[int, int]]] = [[] for _ in ctg]
        # per session row, the session row of its contingency (-1: a contingency row)
        session_parent: list[int] = [-1] * n_ctg
        self._row_branch = {}
        off: int = n_ctg
        for k, (ci, acts) in enumerate(self._strategies):
            assert self._branch_args is not None  # set by set_actions
            bf, bt = self._branch_args[0], self._branch_args[1]
            row = strategy_row(ctg[ci], [self._compiled[a] for a in acts], n_lines, bf, bt, topo)
            if row.elements is None:
                reasons[off + k] = row.reason
                continue
            session_rows.append(off + k)
            session_list.append(row.elements)
            session_triplets.append(row.triplets)
            session_empty.append(row.empty_buses)
            session_sbus.append(row.sbus)
            session_release.append(row.pv_release)
            session_pv.append(row.pv_set)
            session_gen_bus.append(row.gen_bus)
            session_parent.append(ci)
            changed: dict[int, RowBranch] = {b: (y, int(bf[b]), int(bt[b])) for b, y in row.branch_y.items()}
            if row.branch_ends:
                assert topo is not None  # moves run on the union pattern
                changed.update({b: (topo.branch_y[b], f2, t2) for b, (f2, t2) in row.branch_ends.items()})
            if changed:
                self._row_branch[off + k] = changed
        self._rowmap = RowMap(n_out, np.asarray(session_rows, dtype=np.int64), reasons)
        self._rl_result = None
        session = self._inner._s
        self._push_rows(session_list)
        if any(session_triplets):
            session.add_row_triplets(session_triplets)
        if any(session_empty):
            session.set_row_empty_buses(session_empty)
        if any(session_sbus):
            session.add_row_sbus_deltas(session_sbus)
        if any(session_release) or any(session_pv) or any(session_gen_bus):
            session.set_row_pv_moves(session_release, session_pv, session_gen_bus)
        self._inner.batch_size = int(batch_size)
        if stream:
            # the condition and the warm start on the device: each strategy from its own
            # contingency's converged voltages and, with the outer loops, where its loops ended
            # (a second solve on the grid with the contingency applied: its loss shared once)
            session.set_streaming(session_parent, self._strategy_condition.code, self._strategy_warm_start)
        else:
            session.clear_streaming()
        self._inner.run()
        self._last_residuals = self._inner.residuals
        if stream:   # a strategy whose contingency failed the condition did not run
            self._rowmap.released = np.asarray(session.streamed_rows, dtype=bool)
        return self.V_results

    def _prepare_union(self, n_lines: int) -> TopologyContext | None:
        """The union pattern of every registered strategy's bus moves, set on the session when it
        changes; None: no move, the base
        pattern (today's path). Built once, upfront, for the whole run: see plan_union for why,
        and for streaming it while the GPU works (a join of patterns)."""
        if self._grid is None or not any(a.moves for a in self._compiled):
            return self._prepare_union_none()
        assert self._branch_args is not None  # set by set_actions
        bf, bt = self._branch_args[0], self._branch_args[1]
        combos = {acts for _, acts in self._strategies}
        plan = plan_union(self._grid, self._compiled, combos, n_lines, bf, bt)
        if plan.is_empty():
            return self._prepare_union_none()
        if self._union is None or (plan.parents, plan.entries) != (self._union.parents, self._union.entries):
            self._inner._s.set_spare_buses(plan.parents, plan.entries, plan.spare_global)
        self._union = plan
        return topology_context(self._grid, plan, n_lines, bf, bt)

    def _prepare_union_none(self) -> None:
        """Back to the base pattern (today's path) if the session holds a union."""
        if self._union is not None:
            self._inner._s.set_spare_buses([], [])
            self._union = None
        return None

    def _push_rows(self, elements: Sequence[CtgElements]) -> None:
        """Hand the session these rows (the ten lists each) without changing the
        contingencies registered by the caller."""
        saved = (self._ctg_elements, self._ctg_branch_ids)
        try:
            self._push_contingencies(list(elements))
        finally:
            self._ctg_elements, self._ctg_branch_ids = saved

    def _changed_branch_violations(
        self, rows: list[list[LimitViolation]]
    ) -> tuple[list[list[LimitViolation]], dict[int, int]]:
        """``rows`` with the CURRENT records of the branches whose values or ends a row changes
        replaced by the ones of its own admittance and buses (the device check uses the base ones), and the
        change of each such row's current count. Records keep the effective limit here
        (``_restore_limits`` puts the configured one back); a row's current records stay most
        severe first."""
        from ._limit_violations import LimitViolation, LimitViolationType, ViolationElementType

        if not self._row_branch:
            return rows, {}
        _, _, lim1, lim2, n_lines = self._configured_limits()
        t: float = self._violation_threshold
        rel_tol: float = float(self._inner.violation_rel_tol)
        V: np.ndarray = self.V_results.to_numpy().reshape(len(rows), -1)
        delta: dict[int, int] = {}
        for r, changed in self._row_branch.items():

            def branch_of(v: LimitViolation) -> int:
                return v.element_id + (int(n_lines) if v.element_type == ViolationElementType.TRAFO else 0)

            current = LimitViolationType.CURRENT
            kept = [v for v in rows[r] if not (v.violation_type == current and branch_of(v) in changed)]
            added: list[LimitViolation] = []
            for b, rb in changed.items():
                is_trafo: bool = b >= int(n_lines)
                for side, lim in ((1, lim1), (2, lim2)):
                    limit_ka = float(lim[b]) * t
                    if not np.isfinite(limit_ka):
                        continue
                    ka: float = self._branch_amps(V[r], rb, side) * 1e-3
                    if ka > limit_ka * (1.0 + rel_tol):
                        added.append(LimitViolation(
                            ViolationElementType.TRAFO if is_trafo else ViolationElementType.LINE,
                            b - int(n_lines) if is_trafo else b, side, current, ka, limit_ka))
            removed: int = len(rows[r]) - len(kept)
            currents = sorted([v for v in kept if v.violation_type == current] + added,
                              key=lambda v: -abs(v.value / v.limit - 1.0))
            others = [v for v in kept if v.violation_type != current]
            rows[r] = currents + others
            delta[r] = len(added) - removed
        return rows, delta

    def _flow_buffer(self, buf: DeviceBuffer, side: int) -> DeviceBuffer:
        """The amps of one side on the caller's rows, the branches whose values or ends a row
        changes recomputed with its own admittance and buses (the session's flows use the base ones)."""
        nb: int | None = self.n_branches
        if nb is None:
            return buf
        expanded = self._expanded_buffer(buf, nb, np.nan)
        if not self._row_branch:
            return expanded

        def fetch() -> np.ndarray:
            amps: np.ndarray = expanded.to_numpy().reshape(-1, nb).copy()
            V: np.ndarray = self.V_results.to_numpy().reshape(amps.shape[0], -1)
            for r, changed in self._row_branch.items():
                for b, rb in changed.items():
                    amps[r, b] = self._branch_amps(V[r], rb, side)
            return amps.ravel()

        return DeviceBuffer(fetch, expanded.shape, expanded.dtype)

    def _branch_amps(self, V: np.ndarray, rb: RowBranch, side: int) -> float:
        """|I| (A) at ``side`` of a branch of a row, ``rb`` its admittance and (from, to) solver
        buses there, as compute_branch_flows_kernel: a Kron-reduced end has V = 0 and no current;
        the base current uses the nominal voltage of the side's bus (the other end's when it is
        Kron-reduced; a spare bus has its parent's)."""
        assert self._branch_args is not None
        bus_vn_kv, sn_mva = self._branch_args[6], self._branch_args[7]
        y, f, t = rb
        own, other = (f, t) if side == 1 else (t, f)
        if own < 0:
            return 0.0
        vf: complex = complex(V[f]) if f >= 0 else 0j
        vt: complex = complex(V[t]) if t >= 0 else 0j
        i_pu: complex = y[0] * vf + y[1] * vt if side == 1 else y[2] * vf + y[3] * vt
        return abs(i_pu) * float(sn_mva) * 1e3 / (np.sqrt(3.0) * self._bus_vn_kv(bus_vn_kv, own if own >= 0 else other))

    def _bus_vn_kv(self, bus_vn_kv: np.ndarray, bus: int) -> float:
        """The nominal voltage of a solver bus, a spare bus (past the base ones) its parent's."""
        n_base: int = len(bus_vn_kv)
        if bus >= n_base:
            assert self._union is not None  # a spare bus exists only with a union plan
            bus = self._union.parents[bus - n_base]
        return float(bus_vn_kv[bus])

    def _expanded_buffer(self, buf: DeviceBuffer, width: int, fill: Any) -> DeviceBuffer:
        """``buf`` (session rows x width, flat) on the caller's rows when a row map is live."""
        rm = self._rowmap
        if rm is None:
            return buf

        def fetch() -> np.ndarray:
            vals = buf.to_numpy().reshape(-1, width) if len(rm.session_rows) else \
                np.zeros((0, width), dtype=np.asarray(fill).dtype)
            return rm.expand(vals, fill).ravel()

        return DeviceBuffer(fetch, (rm.n_out * width,), buf.dtype)

    def _session_array(self, getter: Callable[[], np.ndarray], fill: Any) -> np.ndarray:
        if self._rowmap is None:
            return getter()
        rm = self._rowmap
        vals = getter() if len(rm.session_rows) else np.zeros(0, dtype=np.asarray(fill).dtype)
        return rm.expand(vals, fill)

    def _session_rows_or_empty(self, getter: Callable[[], list[Any]], fill: Callable[[], Any]) -> list[Any]:
        if self._rowmap is None:
            return getter()
        rm = self._rowmap
        vals = getter() if len(rm.session_rows) else []
        return rm.expand_list(vals, fill)

    # --------------------------------------------- reactive_limits_outer_loop
    @property
    def reactive_limits_outer_loop(self):
        """bool: re-solve once, with the switches applied, the contingencies
        whose physical checks report a reactive-limit switch this loop handles
        (see the constructor's doc). Default False; takes effect on the next
        :meth:`compute`. Deprecated: use ``outer_loops=True``."""
        return self._reactive_limits_outer_loop

    @reactive_limits_outer_loop.setter
    def reactive_limits_outer_loop(self, value):
        value = _check_reactive_limits_outer_loop(value)
        if value:
            _warn_reactive_limits_outer_loop_deprecated()
        self._reactive_limits_outer_loop = value

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
        dst.tail_max_groups = src.tail_max_groups
        dst.tail_group_size = src.tail_group_size
        dst.tol = src.tol
        dst.scheduling = src.scheduling
        dst.ir_n_steps = src.ir_n_steps
        dst.linear_solver = src.linear_solver
        dst.handle_disconnected_grid = src.handle_disconnected_grid
        # the limits this analysis checks, not the (copied) grid's: the second
        # pass' records replace the first pass' of the rows it re-solves
        if src.compute_limit_violations and self._limits_synced != self._limits_gen:
            if self._limits_args is None:   # set by the bridge, off the grid
                sweep.set_limits_from_grid()
            else:
                dst.set_limits(*self._limits_args)
            self._limits_synced = self._limits_gen
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
        if any(any(c[1:]) for c in (getattr(self, "_ctg_elements", None) or [])):
            raise RuntimeError("reactive_limits_outer_loop does not support contingencies tripping "
                               "shunts, loads or generators, or opening a branch at one end: use "
                               "outer_loops=True.")
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
            v_init = (sess, list(map(int, run_rows))) if self._outer_loop_warm_start else None
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
            sess.overwrite_rows(dst, [int(i) for i in ok], s2)
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
        if self._rowmap is not None:
            return self._rowmap.expand(self._last_residuals.to_numpy(), np.nan)
        return self._last_residuals.to_numpy()

    def compute_flows(self):
        """Compute branch currents (``or_amps`` / ``ex_amps``) after compute()."""
        if self._rowmap is None or len(self._rowmap.session_rows):
            self._inner.compute_flows()
        self._flows_done = True

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
        self._set_limits(bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka, n_lines)

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
        compute_limit_violations=True. With reactive_limits_outer_loop, a
        RECOMPUTED contingency's entry is its second pass'."""
        out = self._session_rows_or_empty(self._inner.get_violations, list)
        if self._rl_result is not None:
            for r, v in self._rl_result["viol"].items():
                out[r] = v
        out, _ = self._changed_branch_violations(out)
        return self._restore_limits(out)

    def get_violations_truncated(self):
        """(n_ctg,) bool ndarray: True where more than violation_capacity
        violations were found for that contingency (clamped)."""
        out = self._session_array(self._inner.get_violations_truncated, False)
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
        if self._rowmap is None:
            out = self._inner.get_violation_counts()
        else:
            raw = self._inner.get_violation_counts() if len(self._rowmap.session_rows) else None
            out = {k: self._rowmap.expand(raw[k] if raw else np.zeros(0, dtype=np.int64), -1)
                   for k in ("low_voltage", "high_voltage", "current")}
            if self._row_branch:
                rows = self._session_rows_or_empty(self._inner.get_violations, list)
                _, delta = self._changed_branch_violations(rows)
                for r, d in delta.items():
                    out["current"][r] += d
        if self._rl_result is not None:
            for r, cnt in self._rl_result["viol_counts"].items():
                for k, v in cnt.items():
                    out[k][r] = v
        return out

    def get_physical_violations(self):
        """See :meth:`PhysicalChecksEngineMixin.get_physical_violations`. With
        reactive_limits_outer_loop, a RECOMPUTED contingency's entry is its
        second pass' (with the new labels, see ``_reactive_limits.py``)."""
        out = self._session_rows_or_empty(self._inner.get_physical_violations, list)
        if self._rl_result is not None:
            for r, v in self._rl_result["phys"].items():
                out[r] = v
        return out

    def get_physical_violations_truncated(self):
        """(n_ctg,) bool ndarray: True where one of the physical checks kept
        only its most severe records on that row."""
        out = self._session_array(self._inner.get_physical_violations_truncated, False)
        if self._rl_result is not None:
            for r, v in self._rl_result["phys_trunc"].items():
                out[r] = v
        return out

    def converged(self, tol=None):
        """(n_ctg,) bool ndarray: residual <= tol (defaults to violation_tol).
        Independent of compute_limit_violations -- requires compute()."""
        if self._rowmap is not None:
            t = self._inner.violation_tol if tol is None else tol
            res = self.last_residuals()
            return np.isfinite(res) & (res <= t)
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

        def compute_n(vmin: np.ndarray | None, vmax: np.ndarray | None,
                      a1: np.ndarray | None, a2: np.ndarray | None) -> list:
            return compute_violations_n(
                V_n, bus_vn_kv, vmin, vmax,
                branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
                a1, a2, sn_mva, n_lines,
                rel_tol=self._inner.violation_rel_tol)

        return self._threshold_violations_n(
            compute_n, bus_vn_kv, (bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka), len(branch_from))

    # ----------------------------------------------------------- pass-through
    @property
    def or_amps(self):
        """DeviceBuffer: (n_ctg * n_branches,) origin terminal amps."""
        return self._flow_buffer(self._inner.or_amps, side=1)

    @property
    def ex_amps(self):
        """DeviceBuffer: (n_ctg * n_branches,) extremity terminal amps."""
        return self._flow_buffer(self._inner.ex_amps, side=2)

    @property
    def V_results(self):
        """DeviceBuffer: (n_ctg * n_bus,) complex voltages (lazy D->H)."""
        return self._expanded_buffer(self._inner.V_results, self.n_bus, np.nan + 1j * np.nan)

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
    def ir_n_steps(self):
        """int: cuDSS ``CUDSS_CONFIG_IR_N_STEPS``, iterative refinement steps per
        batch solve (0 = none). Takes effect on the next :meth:`compute`."""
        return self._inner.ir_n_steps

    @ir_n_steps.setter
    def ir_n_steps(self, value):
        self._inner.ir_n_steps = value

    @property
    def timings(self):
        """BatchTimings from the most recent compute() / compute_flows()."""
        return self._inner.timings

    @property
    def n_branches(self) -> int | None:
        """int: branches (lines then transformers); None in array mode before set_branch_data()."""
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
