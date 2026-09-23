// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef LIMIT_VIOLATION_TYPES_HPP
#define LIMIT_VIOLATION_TYPES_HPP

// =============================================================================
// contingency/limit_violation_types.hpp
//
// Shared int codes for compute_limit_violations, mirroring lightsim2grid's
// ls2g::ViolationElementType / ls2g::LimitViolationType (LimitViolation.hpp,
// improve_const_ref branch) exactly, GRID/NOT_SIMULATED/DIVERGENCE included.
// The fused GPU kernel (check_limit_violations_kernel) already computes a
// per-contingency residual check as a precondition to trusting V for the
// bus/branch checks, so folding a GRID-element-type record into the same
// compact output avoids a second round trip for callers of get_violations().
// Note the two codes are written from two different layers, not both from
// the kernel: DIVERGENCE is written by check_limit_violations_kernel itself
// (a contingency it was actually invoked on, but whose residual is NaN or
// exceeds tol); NOT_SIMULATED is written by the Python session layer
// (contingency_analysis/__init__.py's get_violations()) for a contingency
// the pre-check dropped before the chunk loop ever reached this kernel --
// see BatchPfDriver's d_violation_count -1 sentinel.
//
// Not pybind-bound: the kernel writes raw ints, and the Python facade mirrors
// these values as plain enum.IntEnum. Kept in one named place so the codes
// used by the kernels (violation_kernels.cu's constexpr mirror), the Python
// facade (_limit_violations.py), and this doc all agree by construction --
// THREE places to update for any new code.
// =============================================================================

// HVDC (=4): the element a droop P-saturation violation is reported on.
// GENERATOR (=5) / STORAGE (=6): a machine carrying the distributed slack whose
// converged ACTIVE power left its own [min_p, max_p] (lightsim2grid's
// GenPCheck.hpp; the two families take a share under the same rule, so one
// check walks both). A storage unit's value/limit are in the GENERATOR
// convention (positive = injected), like an IIDM battery's own min_p/max_p and
// unlike the LOAD-convention target_p_mw lightsim2grid stores for it.
enum class ViolationElementType : int {
    BUS = 0, LINE = 1, TRAFO = 2, GRID = 3, HVDC = 4, GENERATOR = 5, STORAGE = 6
};
enum class LimitViolationType  : int {
    LOW_VOLTAGE = 0, HIGH_VOLTAGE = 1, CURRENT = 2,
    NOT_SIMULATED = 3,  // pre-check (graph connectivity) skipped it, solver never invoked
    DIVERGENCE = 4,     // solver was invoked but did not converge (or produced NaN)
    // The reactive power the machines holding ONE BUS' voltage had to produce
    // left the SUM of what they own (lightsim2grid PR #206,
    // compute_physical_violations). Not a limit one may choose to exceed: the
    // converged solution is not a state the grid can reach. Written by
    // check_bus_q_violations_kernel; element_id is the SOLVER bus id (unlike
    // lightsim2grid, which reports the grid-model id), value/limit in MVAr.
    LOW_Q = 5,
    HIGH_Q = 6,
    // A droop ("AC emulation") HVDC line in LINEAR regime whose theta-driven
    // flow exceeds pmax in the direction it flows -- OpenLoadFlow's
    // HvdcAcEmulationLimits outer loop would saturate it (compute_hvdc_p_
    // violations). element_type HVDC, element_id the grid hvdc id, side 1
    // (saturates 1->2, p1 > pmax_1to2) or 2 (saturates 2->1, p2 > pmax_2to1),
    // value/limit in MW. Written by check_hvdc_p_violations_kernel.
    // On a GENERATOR / STORAGE (check_gen_p_violations_kernel): the distributed
    // slack -- solved inside the Jacobian by participation factors that know
    // nothing about limits -- asked the machine for more than its max_p_mw
    // (lightsim2grid's name for both cases, HIGH_P).
    HVDC_P_SATURATION = 7,
    HIGH_P = 7,
    // ... and the other way: a slack GENERATOR / STORAGE below its min_p_mw
    // (an hvdc line's two directions are two HIGH_P with a different side).
    LOW_P = 8
};

// What KIND of statement a violation is -- a pure function of its type, so the
// two can never disagree (mirrors lightsim2grid's ViolationCategory exactly).
//   OPERATIONAL : a limit the grid CAN leave (voltage band, current rating).
//   PHYSICAL    : a limit of the equipment itself, which nothing can leave: the
//                 converged solution is not physically realizable (LOW_Q,
//                 HIGH_Q, HIGH_P / HVDC_P_SATURATION, LOW_P).
//   SOLVER      : not a limit at all (NOT_SIMULATED, DIVERGENCE).
enum class ViolationCategory : int { OPERATIONAL = 0, PHYSICAL = 1, SOLVER = 2 };

inline ViolationCategory violation_category(LimitViolationType t) noexcept
{
    switch (t) {
        case LimitViolationType::LOW_VOLTAGE:
        case LimitViolationType::HIGH_VOLTAGE:
        case LimitViolationType::CURRENT:
            return ViolationCategory::OPERATIONAL;
        case LimitViolationType::LOW_Q:
        case LimitViolationType::HIGH_Q:
        case LimitViolationType::HVDC_P_SATURATION:   // == HIGH_P
        case LimitViolationType::LOW_P:
            return ViolationCategory::PHYSICAL;
        default:  // NOT_SIMULATED, DIVERGENCE
            return ViolationCategory::SOLVER;
    }
}

#endif  // LIMIT_VIOLATION_TYPES_HPP
