# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""``compute_physical_violations`` -- the opt-in post-solve PHYSICAL checks, as
the three batch engines and their ``*GPU`` facades expose them (one mixin
each, so the API is written once). Same names and contract as lightsim2grid's:
one flag for the whole category, every record of ``get_physical_violations()``
has ``category == ViolationCategory.PHYSICAL`` -- the converged solution
assumes a control the equipment cannot hold -- and the next physical limit
needs no new flag.

Today the category holds two checks, both detection-only (no bus is switched
PV -> PQ, no line is saturated, no row is re-solved):

- per-bus reactive capability (``LOW_Q`` / ``HIGH_Q`` on a ``BUS``): the
  condition PowSyBl OpenLoadFlow's ``ReactiveLimits`` outer loop acts on;
- droop hvdc P-saturation (``HVDC_P_SATURATION`` on an ``HVDC``):
  OpenLoadFlow's ``HvdcAcEmulationLimits`` outer loop.

The work happens on the device, fused into each chunk
(``check_bus_q_violations_kernel`` / ``check_hvdc_p_violations_kernel``), and
the records are kept apart from ``get_violations()`` (operational limits)
exactly as lightsim2grid keeps them. Lives under ``contingency_analysis/``
next to ``_limit_violations`` for the same historical reason; it is
workload-agnostic.
"""

import numpy as np

from ._limit_violations import (
    bus_q_violations_from_result,
    hvdc_p_violations_from_result,
)

__all__ = ["PhysicalChecksEngineMixin", "PhysicalChecksFacadeMixin"]


def _merge_rows(*per_check):
    """One PHYSICAL list per row: the checks' records concatenated in a fixed
    order (bus reactive capability first, then hvdc saturation)."""
    return [sum(rows, []) for rows in zip(*per_check)]


class PhysicalChecksEngineMixin:
    """For the engine classes (``_ContingencyAnalysisSolver`` & co): expects
    ``self._s`` to be the compiled session, whose ``physical_checks`` is the
    live ``PhysicalChecksConfig``."""

    @property
    def compute_physical_violations(self):
        """bool: report, per converged row, the PHYSICAL violations -- the buses
        whose voltage-holding machines (regulating generators, hvdc converter
        stations, voltage-mode SVCs) had to produce more (or less) reactive
        power than the SUM of what they own (``LOW_Q`` / ``HIGH_Q``; per bus,
        not per machine), and the linear-regime droop hvdc lines whose flow
        exceeds pmax in the direction it flows (``HVDC_P_SATURATION``). Same
        contract as lightsim2grid's flag of the same name: every record has
        ``category == PHYSICAL``. Detection only, nothing is enforced. Needs
        :meth:`set_bus_q_capability` (the facades do it from the grid). Default
        False; changing it drops the previous report; takes effect on the next
        run()."""
        return self._s.physical_checks.compute_physical_violations

    @compute_physical_violations.setter
    def compute_physical_violations(self, value):
        self._s.physical_checks.compute_physical_violations = bool(value)

    @property
    def physical_violation_tol_mva(self):
        """float: slack (MVA) on every comparison -- MVAr for the reactive check
        (a violation needs ``q_bus < sum(min_q) - tol`` or ``q_bus > sum(max_q)
        + tol``), MW for the hvdc one (``p_flow > pmax + tol``). Default 1e-4
        (lightsim2grid's); finite and >= 0, any change drops the previous
        report."""
        return self._s.physical_checks.physical_violation_tol_mva

    @physical_violation_tol_mva.setter
    def physical_violation_tol_mva(self, value):
        self._s.physical_checks.physical_violation_tol_mva = float(value)

    @property
    def physical_violation_capacity(self):
        """int: records kept per row and per check (bounds each compact output
        at n_rows * capacity; a row with more is flagged by
        :meth:`get_physical_violations_truncated`). Default 16."""
        return self._s.physical_checks.physical_violation_capacity

    @physical_violation_capacity.setter
    def physical_violation_capacity(self, value):
        self._s.physical_checks.physical_violation_capacity = int(value)

    @property
    def has_bus_q_capability(self):
        """bool: whether :meth:`set_bus_q_capability` was called."""
        return self._s.physical_checks.has_bus_q_capability

    def set_bus_q_capability(self, plan):
        """Hand in the routing of the reactive-capability check: which buses a
        machine holds, and what each can produce -- a ``BusQPlanData`` (as
        returned by ``_gpusim2grid._extract_bus_q_plan_from_lsgrid``, i.e.
        lightsim2grid's own ``build_bus_q_plan``; the facades' ``set_bus_q_
        capability_from_grid()`` does that call) or, in array mode, the tuple
        of its constructor arguments ``(bus_solver, qmin_fixed_mvar,
        qmax_fixed_mvar, n_fixed, bmin_sum_pu, bmax_sum_pu, gen_start, gen_id,
        gen_qmin_mvar, gen_qmax_mvar, sn_mva)`` -- solver bus numbering, MVAr
        for generators / hvdc stations, pu susceptance for SVCs, generators as
        a per-bus CSR of container ids (the columns of a generator-contingency
        mask). An empty plan (no bus) is accepted. Validated against n_bus;
        drops any previous report."""
        from .._gpusim2grid import BusQPlanData
        if not isinstance(plan, BusQPlanData):
            (bus_solver, qmin_fixed, qmax_fixed, n_fixed, bmin_sum, bmax_sum,
             gen_start, gen_id, gen_qmin, gen_qmax, sn_mva) = plan
            i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
            f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
            plan = BusQPlanData(i32(bus_solver), f64(qmin_fixed), f64(qmax_fixed),
                                i32(n_fixed), f64(bmin_sum), f64(bmax_sum),
                                i32(gen_start), i32(gen_id), f64(gen_qmin), f64(gen_qmax),
                                float(sn_mva))
        self._s.set_bus_q_capability(plan)

    def get_physical_violations(self):
        """list[list[LimitViolation]]: one entry per row (caller's row order),
        every record of category PHYSICAL -- the reactive-capability records
        first (element_type BUS, element_id the SOLVER bus id, violation_type
        LOW_Q / HIGH_Q, value the reactive power the machines holding that bus
        had to produce in MVAr, limit their summed capability), then the hvdc
        ones (element_type HVDC, element_id the grid hvdc id, side 1 = would
        saturate 1->2 / 2 = 2->1, value the flow leaving that AC bus in MW,
        limit pmax). A row that was never simulated (compacted out) or did not
        converge has an EMPTY entry, not a sentinel -- ask ``converged()`` /
        ``get_disconnected()`` to tell that apart from "converged, no
        violation" (lightsim2grid parity). Requires run() with
        compute_physical_violations=True."""
        return _merge_rows(bus_q_violations_from_result(self._s.get_bus_q_violations()),
                           hvdc_p_violations_from_result(self._s.get_hvdc_p_violations()))

    def get_physical_violations_n(self):
        """list[LimitViolation]: the same for the base ("n") case every row is
        solved from (empty when the base solve did not converge)."""
        return _merge_rows(bus_q_violations_from_result(self._s.get_bus_q_violations_n()),
                           hvdc_p_violations_from_result(self._s.get_hvdc_p_violations_n()))[0]

    def get_physical_violations_truncated(self):
        """(n_rows,) bool ndarray: True where one of the checks found more than
        physical_violation_capacity violations on that row (records clamped)."""
        bq = np.asarray(self._s.get_bus_q_violations().truncated).astype(bool)
        hp = np.asarray(self._s.get_hvdc_p_violations().truncated).astype(bool)
        return bq | hp


class PhysicalChecksFacadeMixin:
    """For the ``*GPU`` facades: expects ``self._inner`` (an engine carrying
    :class:`PhysicalChecksEngineMixin`) and ``self._grid`` (the lightsim2grid
    grid, or None in explicit-array mode)."""

    def _apply_physical_checks_kwargs(self, compute_physical_violations):
        """Constructor tail: apply the opt-in flag (which pulls the bus
        capability plan off the grid when there is one)."""
        if compute_physical_violations:
            self.compute_physical_violations = True

    def set_bus_q_capability_from_grid(self, grid=None):
        """Build the routing of the reactive-capability check off the
        lightsim2grid grid (the one this object was built from, unless ``grid``
        is given) with lightsim2grid's OWN ``build_bus_q_plan`` -- which buses a
        voltage-regulating generator, hvdc converter station or voltage-mode SVC
        holds, and what each can produce -- and hand it to the session. Needs
        the compiled bridge (any solved LSGrid will do, independently of
        ``use_bridge``). Done automatically when ``compute_physical_violations``
        is turned on and no plan was set yet; call it again if the generators'
        reactive limits changed on the grid. In explicit-array mode use
        :meth:`set_bus_q_capability` instead."""
        from .. import _gpusim2grid as _cpp
        grid = self._grid if grid is None else grid
        if grid is None:
            raise RuntimeError(
                "set_bus_q_capability_from_grid() requires a lightsim2grid grid "
                "(this session was built from an explicit-array tuple); use "
                "set_bus_q_capability(...) with the plan arrays instead.")
        if not getattr(_cpp, "have_ls2g_bridge", False):
            raise RuntimeError(
                "set_bus_q_capability_from_grid() needs gpusim2grid compiled with the "
                "lightsim2grid bridge (the plan is built by lightsim2grid's own "
                "build_bus_q_plan); use set_bus_q_capability(...) with the plan arrays.")
        self._inner.set_bus_q_capability(
            _cpp._extract_bus_q_plan_from_lsgrid(grid, self._inner._s.n_bus))

    def set_bus_q_capability(self, plan):
        """Explicit-array mode counterpart of :meth:`set_bus_q_capability_from_grid`
        -- see the engine's ``set_bus_q_capability`` for the arrays."""
        self._inner.set_bus_q_capability(plan)

    @property
    def compute_physical_violations(self):
        """bool: the PHYSICAL checks -- per-bus reactive capability (LOW_Q /
        HIGH_Q) and droop hvdc P-saturation (HVDC_P_SATURATION), see
        ``get_physical_violations()``; lightsim2grid's flag of the same name.
        Turning it on pulls the bus capability plan off the grid if none was set
        yet. Default False; takes effect on the next compute()."""
        return self._inner.compute_physical_violations

    @compute_physical_violations.setter
    def compute_physical_violations(self, value):
        value = bool(value)
        if value and not self._inner.has_bus_q_capability and self._grid is not None:
            self.set_bus_q_capability_from_grid()
        self._inner.compute_physical_violations = value

    @property
    def physical_violation_tol_mva(self):
        """float: slack (MVA) on every physical comparison; default 1e-4."""
        return self._inner.physical_violation_tol_mva

    @physical_violation_tol_mva.setter
    def physical_violation_tol_mva(self, value):
        self._inner.physical_violation_tol_mva = value

    @property
    def physical_violation_capacity(self):
        """int: records kept per row and per check; default 16."""
        return self._inner.physical_violation_capacity

    @physical_violation_capacity.setter
    def physical_violation_capacity(self, value):
        self._inner.physical_violation_capacity = value

    @property
    def has_bus_q_capability(self):
        """bool: whether the reactive-capability plan is set."""
        return self._inner.has_bus_q_capability

    def get_physical_violations(self):
        """list[list[LimitViolation]]: per row (caller's row order), every
        PHYSICAL violation -- the buses whose machines had to produce more (or
        less) reactive power than the SUM of what they own (LOW_Q / HIGH_Q,
        element_id the SOLVER bus id, MVAr) followed by the linear-regime droop
        hvdc lines whose flow exceeds pmax (HVDC_P_SATURATION, element_id the
        grid hvdc id, side 1 = would saturate 1->2 / 2 = 2->1, MW). A
        never-simulated or non-converged row has an EMPTY entry. Kept apart from
        ``get_violations()`` (operational limits). Requires compute() with
        compute_physical_violations=True."""
        return self._inner.get_physical_violations()

    def get_physical_violations_n(self):
        """list[LimitViolation]: the same for the base ("n") case."""
        return self._inner.get_physical_violations_n()

    def get_physical_violations_truncated(self):
        """(n_rows,) bool ndarray: rows where a check found more than
        physical_violation_capacity violations."""
        return self._inner.get_physical_violations_truncated()
