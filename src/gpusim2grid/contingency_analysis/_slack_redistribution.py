# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``redistribute_slack`` -- the OpenLoadFlow-style bounded redistribution of
what a batch row loses (lightsim2grid PR #216), as the contingency-analysis
and scenario-sweep engines and their ``*GPU`` facades expose it (one mixin
each, so the API is written once).

The distributed slack of the Newton solve shares whatever imbalance a row
leaves by fixed per-bus weights, with no limit: a row that loses a big
generator (a scenario-sweep generator contingency, or an island cut off with
``handle_disconnected_grid``) pushes the remaining machines past their
``max_p``, where OpenLoadFlow's ``DistributedSlack`` outer loop stops each one
at its bound and re-shares the excess. With the option on, the part of that
imbalance known BEFORE the solve -- the set-points of what the row takes out --
is shared on the remaining slack units the way OLF does it (proportionally to
their weight, each clamped to its ``[min_p, max_p]`` and never crossing 0 MW,
a clamped unit leaving the pool), the saturated units leave that row's
distributed slack, and the solve only shares what is left (the change in the
losses) on the units that can still move. The pre-pass runs on the host, once
per ``compute()`` (``slack_redistribution.hpp``); the limits are lightsim2grid's
``set_gen_p_limits`` / ``set_storage_p_limits`` -- without limits nothing
saturates and the converged state is unchanged. Off by default.

Lives under ``contingency_analysis/`` next to ``_physical_checks`` for the same
reason: it is shared by the two batch sessions that know contingencies.
"""

import numpy as np

__all__ = ["SlackRedistributionEngineMixin", "SlackRedistributionFacadeMixin"]


class SlackRedistributionEngineMixin:
    """For the engine classes: expects ``self._s`` to be the compiled
    session."""

    @property
    def redistribute_slack(self):
        """bool: share what each row loses on the remaining slack units BEFORE
        the solve, OpenLoadFlow's ``DistributedSlack`` way (bounded by each
        unit's ``[min_p, max_p]``, a saturated unit leaving that row's slack) --
        lightsim2grid's option of the same name. Needs
        :meth:`set_slack_redistribution_data` and the distributed slack. Default
        False; takes effect on the next run()."""
        return self._s.redistribute_slack

    @redistribute_slack.setter
    def redistribute_slack(self, value):
        if bool(value) != value:
            raise ValueError("The `redistribute_slack` attribute must be a boolean.")
        self._s.redistribute_slack = bool(value)

    @property
    def has_slack_redistribution_data(self):
        """bool: whether the slack participants of the pre-pass are set."""
        return self._s.has_slack_redistribution_data

    def set_slack_redistribution_data(self, data):
        """Hand in the slack participants of the pre-pass -- a
        ``SlackRedistributionData`` (``_gpusim2grid._extract_slack_
        redistribution_data_from_lsgrid``; the facades do it from the grid) or,
        in array mode, the tuple of its constructor arguments ``(kind, el_id,
        bus_solver, weight, min_p_mw, max_p_mw, target_p_mw, gen_bus_solver,
        gen_target_p_mw, n_sto, shunt_p_mw, sn_mva[, in_slack[, base_p_mw]])``: one entry per
        unit of the distributed slack (generators then storage units, by id;
        kind 5 / 6, container id, solver bus, raw weight, limits with NaN =
        none, set-point in the GENERATOR convention), then every generator's
        solver bus (-1: not in the solved system) and set-point, the storage
        count, the shunts' active power at 1 pu per solver bus (MW) and sn_mva;
        the optional ``in_slack`` (one per unit, all 1 when absent) is 0 for a
        unit of the pre-pass only (lightsim2grid's "can participate in the
        slack", left out of the slack only because it sat at an active
        limit); the optional ``base_p_mw`` (NaN when absent) is the active
        injection of the grid's own set-points summed over the solved system
        (MW): a scenario sweep pre-shares what each row's injections take out
        of it, and shares no such term without it."""
        from .._gpusim2grid import SlackRedistributionData
        if not isinstance(data, SlackRedistributionData):
            (kind, el_id, bus_solver, weight, min_p, max_p, target_p,
             gen_bus, gen_target_p, n_sto, shunt_p, sn_mva, *optional) = data
            i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)       # noqa: E731
            f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)     # noqa: E731
            data = SlackRedistributionData(i32(kind), i32(el_id), i32(bus_solver), f64(weight),
                                           f64(min_p), f64(max_p), f64(target_p), i32(gen_bus),
                                           f64(gen_target_p), int(n_sto), f64(shunt_p),
                                           float(sn_mva), i32(optional[0] if optional else []),
                                           float(optional[1]) if len(optional) > 1 else float("nan"))
        self._s.set_slack_redistribution_data(data)

    @property
    def auto_reference_slack(self):
        """bool: with ``handle_disconnected_grid`` and the distributed slack,
        make the angle reference the slack participant stranded by the fewest
        rows (lightsim2grid PR #216's batch rule; ties: higher weight, then
        lower bus), so fewer rows are skipped for stranding it. A change moves
        the reference and rebuilds the base state; the solution is the same up
        to a constant angle shift. A reference forced on the grid
        (``LSGrid.set_reference_slack_bus``) is kept."""
        return self._s.auto_reference_slack

    @auto_reference_slack.setter
    def auto_reference_slack(self, value):
        self._s.auto_reference_slack = bool(value)

    @property
    def reference_bus(self):
        """int: the solver bus the current base state uses as angle reference
        (-1: none, e.g. without the distributed slack)."""
        return self._s.reference_bus

    def get_slack_redistribution_report(self):
        """dict of (n_rows,) arrays, what the pre-pass did on each row of the
        last run() (lightsim2grid's ``SlackRedistributionReport``, one entry
        per row, caller's row order): ``mismatch_mw`` (what had to be shared,
        > 0: the units inject more), ``not_distributed_mw`` (what no unit could
        take), ``nb_participants``, ``nb_saturated`` (units that reached a bound
        and left the row's slack), ``nb_rounds`` (0: nothing shared) and
        ``all_saturated`` (every unit hit its bound: all stay in the slack).
        All zero with the option off."""
        rep = self._s.get_slack_redistribution_report()
        return {
            "mismatch_mw": np.asarray(rep.mismatch_mw),
            "not_distributed_mw": np.asarray(rep.not_distributed_mw),
            "nb_participants": np.asarray(rep.nb_participants),
            "nb_saturated": np.asarray(rep.nb_saturated),
            "nb_rounds": np.asarray(rep.nb_rounds),
            "all_saturated": np.asarray(rep.all_saturated).astype(bool),
        }


class SlackRedistributionFacadeMixin:
    """For the ``*GPU`` facades: expects ``self._inner`` (an engine carrying
    :class:`SlackRedistributionEngineMixin`) and ``self._grid`` (the
    lightsim2grid grid, or None in explicit-array mode)."""

    def set_slack_redistribution_data_from_grid(self, grid=None):
        """Read the slack participants of the pre-pass off the lightsim2grid
        grid (the one this object was built from, unless ``grid`` is given):
        the generators and storage units of the distributed slack with their
        ``[min_p, max_p]`` (``set_gen_p_limits`` / ``set_storage_p_limits``),
        every generator's set-point and the shunts' active power. Needs the
        compiled bridge. Done automatically when ``redistribute_slack`` is
        turned on (so limits set on the grid after construction are seen)."""
        from .. import _gpusim2grid as _cpp
        grid = self._grid if grid is None else grid
        if grid is None:
            raise RuntimeError(
                "set_slack_redistribution_data_from_grid() requires a lightsim2grid grid "
                "(this session was built from an explicit-array tuple); use "
                "set_slack_redistribution_data(...) with the arrays instead.")
        if not getattr(_cpp, "have_ls2g_bridge", False):
            raise RuntimeError(
                "set_slack_redistribution_data_from_grid() needs gpusim2grid compiled with "
                "the lightsim2grid bridge; use set_slack_redistribution_data(...) with the "
                "arrays instead.")
        self._inner.set_slack_redistribution_data(
            _cpp._extract_slack_redistribution_data_from_lsgrid(grid, self._inner._s.n_bus))

    def set_slack_redistribution_data(self, data):
        """Explicit-array mode counterpart of
        :meth:`set_slack_redistribution_data_from_grid` -- see the engine's
        ``set_slack_redistribution_data`` for the arrays."""
        self._inner.set_slack_redistribution_data(data)

    @property
    def redistribute_slack(self):
        """bool: share what each row loses -- the island a contingency cuts
        off with ``handle_disconnected_grid``, and (scenario sweep) the
        generators a row disconnects and the imbalance its own injections
        create against the grid's set-points -- on the remaining units of the
        distributed slack BEFORE the solve, as OpenLoadFlow's
        ``DistributedSlack`` outer loop does: proportionally to their weight,
        each one clamped to its ``[min_p, max_p]`` and never crossing 0 MW, a
        clamped unit leaving the pool and that row's distributed slack; the
        solve then only shares what is left (the change in the losses) on the
        units that can still move (a line / trafo that leaves the grid
        connected loses no injection). lightsim2grid's option of the same name
        (``ContingencyAnalysis`` / ``ScenarioSweep``). Needs the distributed
        slack (``use_distributed_slack=True``) and limits on the grid
        (``set_gen_p_limits`` / ``set_storage_p_limits``) to clamp anything;
        turning it on reads them off the grid. Default False; takes effect on
        the next ``compute()``."""
        return self._inner.redistribute_slack

    @redistribute_slack.setter
    def redistribute_slack(self, value):
        if bool(value) != value:
            raise ValueError("The `redistribute_slack` attribute must be a boolean.")
        if value and self._grid is not None:
            self.set_slack_redistribution_data_from_grid()
        self._inner.redistribute_slack = bool(value)

    def get_slack_redistribution_report(self):
        """dict of (n_rows,) arrays: what the pre-pass did on each row of the
        last ``compute()`` -- see the engine's method of the same name."""
        return self._inner.get_slack_redistribution_report()

    @property
    def reference_slack(self):
        """str: how the angle reference of a ``handle_disconnected_grid``
        batch is chosen -- ``"auto"`` (default): the distributed-slack
        participant stranded by the fewest rows, as lightsim2grid's own batch
        does since PR #216 (a reference forced on the grid with
        ``set_reference_slack_bus`` is kept); ``"grid"``: the grid's own, fixed
        (a row stranding it is skipped). Only matters with the distributed
        slack and ``handle_disconnected_grid``; a change of reference moves the
        angles by a constant and costs one base-case rebuild."""
        return "auto" if self._inner.auto_reference_slack else "grid"

    @reference_slack.setter
    def reference_slack(self, value):
        if value not in ("auto", "grid"):
            raise ValueError(f"reference_slack must be 'auto' or 'grid', got {value!r}")
        self._inner.auto_reference_slack = (value == "auto")

    @property
    def reference_bus(self):
        """int: the solver bus the current base state uses as angle reference."""
        return self._inner.reference_bus
