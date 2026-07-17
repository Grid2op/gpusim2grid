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
    "LimitViolation",
    "compute_violations_n",
]

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ViolationElementType(IntEnum):
    """Mirrors lightsim2grid's ls2g::ViolationElementType exactly (including
    GRID, added on lightsim2grid's improve_const_ref branch)."""
    BUS = 0
    LINE = 1
    TRAFO = 2
    GRID = 3  # the whole grid/contingency, not a specific element


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
    """
    element_type: ViolationElementType
    element_id: int
    side: int
    violation_type: LimitViolationType
    value: float
    limit: float


def compute_violations_n(V, bus_vn_kv, bus_vmin_kv, bus_vmax_kv,
                          branch_from, branch_to, yff, yft, ytf, ytt,
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
    branch_from, branch_to, yff, yft, ytf, ytt, sn_mva
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
            V, branch_from, branch_to, yff, yft, ytf, ytt, bus_vn_kv, sn_mva)
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
