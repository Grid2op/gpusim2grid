# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Types and pre-contingency ("n") helper for compute_limit_violations.

The batch (contingency) case is computed on-device, fused into each chunk's
power flow (see ContingencyAnalysisSession::set_limits() /
check_limit_violations_kernel) -- these types are the Python-facing mirror of
that kernel's compact int-coded output. The pre-contingency case is a single
voltage vector, not a batch, so it stays pure Python/CPU (compute_violations_n
below).
"""

__all__ = [
    "ViolationElementType",
    "LimitViolationType",
    "ViolationCategory",
    "violation_category",
    "LimitViolation",
    "compute_violations_n",
    "bus_q_violations_from_result",
    "hvdc_p_violations_from_result",
    "gen_p_violations_from_result",
]

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ViolationElementType(IntEnum):
    """Mirrors lightsim2grid's ls2g::ViolationElementType exactly (including
    GRID, NOT_SIMULATED's element, and HVDC / GENERATOR / STORAGE, the elements
    the PHYSICAL checks of ``compute_physical_violations`` report on)."""
    BUS = 0
    LINE = 1
    TRAFO = 2
    GRID = 3  # the whole grid/contingency, not a specific element
    HVDC = 4  # an hvdc line (compute_physical_violations)
    #: A generator / a storage unit, by its own container id: its ACTIVE power
    #: only (LOW_P / HIGH_P, the distributed slack asked it for more than it
    #: has). A reactive violation is reported on the BUS instead, because how a
    #: bus' reactive power is divided between its machines is a convention,
    #: while the active one is divided by the participation factors the caller
    #: chose. A STORAGE record's value/limit are in the GENERATOR convention
    #: (positive = injected), like its min_q/max_q and unlike the LOAD-convention
    #: target_p_mw lightsim2grid stores for it.
    GENERATOR = 5
    STORAGE = 6


class LimitViolationType(IntEnum):
    """Mirrors lightsim2grid's ls2g::LimitViolationType exactly, including
    NOT_SIMULATED/DIVERGENCE (added on lightsim2grid's improve_const_ref
    branch, alongside ViolationElementType.GRID). Both are written under
    element_type=GRID, but from two different layers:

    - DIVERGENCE is written by the fused GPU kernel itself
      (check_limit_violations_kernel) for a contingency it actually ran the
      solver on but whose residual is NaN or exceeds violation_tol -- V is
      unreliable, so folding this into the same compact output avoids a
      second round trip to get_residuals() for callers of get_violations().
    - NOT_SIMULATED is written by the Python session layer (get_violations())
      for a contingency the pre-check dropped before it ever reached that
      kernel (graph connectivity; see BatchPfDriver's d_violation_count -1
      sentinel) -- the solver was never invoked at all.

    gpusim2grid additionally populates value/limit with the actual
    residual/tol for DIVERGENCE entries; lightsim2grid's own convention
    leaves value/limit NaN/unused for both GRID violation types."""
    LOW_VOLTAGE = 0
    HIGH_VOLTAGE = 1
    CURRENT = 2
    NOT_SIMULATED = 3
    DIVERGENCE = 4
    #: The reactive power the machines holding ONE BUS' voltage had to produce
    #: went BELOW / ABOVE the SUM of what they own (lightsim2grid PR #206,
    #: ``compute_physical_violations``). Category PHYSICAL: a machine cannot
    #: produce reactive power it does not have, so the converged solution is
    #: not a state the grid can reach. Reported, never enforced.
    LOW_Q = 5
    HIGH_Q = 6
    #: A droop ("AC emulation") hvdc line in linear regime whose flow exceeds
    #: pmax in the direction it flows -- OpenLoadFlow's HvdcAcEmulationLimits
    #: outer loop would saturate it (``compute_physical_violations``). Category
    #: PHYSICAL. On a GENERATOR / STORAGE: the distributed slack -- solved
    #: inside the Jacobian by participation factors that know nothing about
    #: limits -- asked the machine for more than its max_p_mw (lightsim2grid's
    #: GenPCheck.hpp; OpenLoadFlow's DistributedSlack outer loop).
    HIGH_P = 7               # lightsim2grid's name for it (LimitViolation.hpp)
    HVDC_P_SATURATION = 7    # alias: the name it was introduced under here
    #: ... and the other way: a slack GENERATOR / STORAGE below its min_p_mw
    #: (an hvdc line's two directions are two HIGH_P with a different side).
    LOW_P = 8


class ViolationCategory(IntEnum):
    """What KIND of statement a violation is (mirrors lightsim2grid's
    ``ViolationCategory`` exactly) -- a pure function of its type, see
    :func:`violation_category` / :attr:`LimitViolation.category`.

    OPERATIONAL : a limit an operator chose and the grid CAN leave (a bus
        outside its voltage band, a branch above its rating): a reachable
        state nobody wants to sit in. LOW_VOLTAGE, HIGH_VOLTAGE, CURRENT.
    PHYSICAL : a limit of the equipment itself, which nothing can leave: the
        converged solution is NOT physically realizable, the control it assumes
        cannot happen. LOW_Q, HIGH_Q, HIGH_P (= HVDC_P_SATURATION), LOW_P.
    SOLVER : not a limit at all, what the solver did. NOT_SIMULATED, DIVERGENCE.
    """
    OPERATIONAL = 0
    PHYSICAL = 1
    SOLVER = 2


def violation_category(violation_type):
    """The :class:`ViolationCategory` of a :class:`LimitViolationType`."""
    t = LimitViolationType(int(violation_type))
    if t in (LimitViolationType.LOW_VOLTAGE, LimitViolationType.HIGH_VOLTAGE,
             LimitViolationType.CURRENT):
        return ViolationCategory.OPERATIONAL
    if t in (LimitViolationType.LOW_Q, LimitViolationType.HIGH_Q,
             LimitViolationType.HVDC_P_SATURATION, LimitViolationType.LOW_P):
        return ViolationCategory.PHYSICAL
    return ViolationCategory.SOLVER


@dataclass(frozen=True)
class LimitViolation:
    """Mirrors lightsim2grid's ls2g::LimitViolation field-for-field.

    element_id : grid-model bus id (solver numbering, see the module docstring
        of contingency_analysis/gpu_facade.py for the numbering caveat) for BUS;
        LOCAL (own-type, 0-based) id for LINE/TRAFO -- i.e. de-concatenated
        from gpusim2grid's lines-then-trafos branch numbering, NOT the same
        as the branch_ids_per_ctg index. -1 for GRID (NOT_SIMULATED /
        DIVERGENCE; whole-system, no specific element).
    side : 0 for BUS/GRID; 1 or 2 for LINE/TRAFO (1 = origin/"or"
        terminal, matching limit_a1_ka/or_amps; 2 = extremity/"ex" terminal,
        matching limit_a2_ka/ex_amps).
    value / limit : kV for LOW_VOLTAGE/HIGH_VOLTAGE, kA for CURRENT; for
        GRID (NOT_SIMULATED / DIVERGENCE), gpusim2grid populates
        residual/tol (lightsim2grid's own convention leaves these NaN/unused
        for GRID).

    The two PHYSICAL checks add (see :class:`ViolationCategory`):

    LOW_Q / HIGH_Q (``compute_physical_violations``) : element_type BUS,
        element_id the SOLVER bus id (unlike lightsim2grid, which reports the
        grid-model id -- gpusim2grid's own voltage records use solver
        numbering, and so does its V array), side 0, value the reactive
        power the machines holding that bus had to produce (MVAr), limit
        their SUMMED capability (MVAr).
    HVDC_P_SATURATION (``compute_physical_violations``) : element_type HVDC,
        element_id the grid hvdc id, side 1 (would saturate 1->2: the flow
        leaving bus 1 exceeds pmax_1to2) or 2 (2->1), value that flow (MW),
        limit pmax (MW).
    LOW_P / HIGH_P on a GENERATOR / STORAGE (``compute_physical_violations``) :
        element_id the container id of that family, side 0, value the
        machine's converged active power -- its target plus its share of the
        distributed slack (MW, GENERATOR convention for both families), limit
        its min_p_mw / max_p_mw.
    """
    element_type: ViolationElementType
    element_id: int
    side: int
    violation_type: LimitViolationType
    value: float
    limit: float

    @property
    def category(self):
        """:class:`ViolationCategory` of this violation (derived from its type)."""
        return violation_category(self.violation_type)


def _rows_from_flat(count, stride, make):
    """Split a flat per-row record buffer into one list per row: row r owns
    slots [r*stride, r*stride + count[r]); a negative count (the row was
    never simulated) and a zero one (simulated, nothing to report -- or not
    converged: upstream reports an EMPTY entry there, never a sentinel) both
    give an empty list."""
    out = []
    for r, cnt in enumerate(count):
        cnt = int(cnt)
        base = r * stride
        out.append([make(base + i) for i in range(max(cnt, 0))])
    return out


def bus_q_violations_from_result(res):
    """list[list[LimitViolation]] from a ``BusQViolationsResult`` (the raw
    output of ``get_bus_q_violations[_n]()`` on a batch session)."""
    bus_id, vtype, value, limit = res.bus_id, res.type, res.value, res.limit
    return _rows_from_flat(res.count, res.stride, lambda i: LimitViolation(
        ViolationElementType.BUS, int(bus_id[i]), 0, LimitViolationType(int(vtype[i])),
        float(value[i]), float(limit[i])))


def hvdc_p_violations_from_result(res):
    """list[list[LimitViolation]] from an ``HvdcPViolationsResult`` (the raw
    output of ``get_hvdc_p_violations[_n]()`` on a batch session)."""
    hvdc_id, side, value, limit = res.hvdc_id, res.side, res.value, res.limit
    return _rows_from_flat(res.count, res.stride, lambda i: LimitViolation(
        ViolationElementType.HVDC, int(hvdc_id[i]), int(side[i]),
        LimitViolationType.HVDC_P_SATURATION, float(value[i]), float(limit[i])))


def gen_p_violations_from_result(res):
    """list[list[LimitViolation]] from a ``GenPViolationsResult`` (the raw
    output of ``get_gen_p_violations[_n]()`` on a batch session)."""
    etype, eid, vtype, value, limit = res.element_type, res.element_id, res.type, res.value, res.limit
    return _rows_from_flat(res.count, res.stride, lambda i: LimitViolation(
        ViolationElementType(int(etype[i])), int(eid[i]), 0, LimitViolationType(int(vtype[i])),
        float(value[i]), float(limit[i])))


def compute_violations_n(V, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
                          branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff,
                          branch_limit_a1_ka, branch_limit_a2_ka, sn_mva,
                          n_lines, residual=None, tol=None):
    """Pre-contingency ("n") limit-violation check: a single voltage vector,
    not a batch, so this is pure numpy -- no GPU, no memory/transfer concern.

    Reuses gpusim2grid.acpf_nr.compute_branch_flows_cpu (the same Amps-based
    formula the GPU kernel mirrors) and divides by 1000 for kA, matching the
    fused kernel's unit convention (see check_limit_violations_kernel).

    Parameters
    ----------
    V : (n_bus,) complex
        Converged bus voltages (solver numbering), e.g. grid.get_V_solver().
    bus_vn_kv, bus_vmin_kv, bus_vmax_kv : (n_bus,) float or None
        Nominal / limit voltages in kV. None (or all-NaN limits) disables the
        bus-voltage check.
    branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, sn_mva
        Same arguments as compute_branch_flows_cpu (lines-then-trafos order).
    branch_limit_a1_ka, branch_limit_a2_ka : (n_branches,) float or None
        Per-side current limits in kA. None (or all-NaN) disables the
        current check.
    n_lines : int
        Splits the lines-then-trafos branch ordering for element_type/
        element_id de-concatenation.
    residual, tol : float or None
        If both given and isnan(residual) or residual > tol, a single GRID/
        DIVERGENCE entry is returned and no other check is run (V is assumed
        unreliable), mirroring the fused kernel's behavior for the batch
        case. There is no NOT_SIMULATED equivalent here -- unlike the batch
        (contingency) case, the pre-contingency "n" power flow this helper
        checks is always actually run, never pre-check-dropped.

    Returns
    -------
    list[LimitViolation]
    """
    from gpusim2grid.acpf_nr import compute_branch_flows_cpu

    out = []

    if residual is not None and tol is not None and (np.isnan(residual) or residual > tol):
        out.append(LimitViolation(ViolationElementType.GRID, -1, 0,
                                   LimitViolationType.DIVERGENCE,
                                   float(residual), float(tol)))
        return out

    if bus_vmin_kv is not None and bus_vmax_kv is not None:
        vm_kv = np.abs(V) * bus_vn_kv
        with np.errstate(invalid="ignore"):
            low = ~np.isnan(bus_vmin_kv) & (vm_kv < bus_vmin_kv)
            high = ~np.isnan(bus_vmax_kv) & (vm_kv > bus_vmax_kv)
        for b in np.nonzero(low)[0]:
            out.append(LimitViolation(ViolationElementType.BUS, int(b), 0,
                                       LimitViolationType.LOW_VOLTAGE,
                                       float(vm_kv[b]), float(bus_vmin_kv[b])))
        for b in np.nonzero(high)[0]:
            out.append(LimitViolation(ViolationElementType.BUS, int(b), 0,
                                       LimitViolationType.HIGH_VOLTAGE,
                                       float(vm_kv[b]), float(bus_vmax_kv[b])))

    if branch_limit_a1_ka is not None and branch_limit_a2_ka is not None:
        or_amps, ex_amps = compute_branch_flows_cpu(
            V, branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, bus_vn_kv, sn_mva)
        or_ka, ex_ka = or_amps * 1e-3, ex_amps * 1e-3
        for l in range(len(branch_from)):
            etype = ViolationElementType.LINE if l < n_lines else ViolationElementType.TRAFO
            eid = l if l < n_lines else l - n_lines
            lim1, lim2 = branch_limit_a1_ka[l], branch_limit_a2_ka[l]
            if not np.isnan(lim1) and or_ka[l] > lim1:
                out.append(LimitViolation(etype, eid, 1, LimitViolationType.CURRENT,
                                           float(or_ka[l]), float(lim1)))
            if not np.isnan(lim2) and ex_ka[l] > lim2:
                out.append(LimitViolation(etype, eid, 2, LimitViolationType.CURRENT,
                                           float(ex_ka[l]), float(lim2)))

    return out
