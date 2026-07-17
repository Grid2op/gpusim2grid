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
// used by the kernel (violation_kernels.cu), the Python facade
// (_limit_violations.py), and this doc all agree by construction.
// =============================================================================

enum class ViolationElementType : int { BUS = 0, LINE = 1, TRAFO = 2, GRID = 3 };
enum class LimitViolationType  : int {
    LOW_VOLTAGE = 0, HIGH_VOLTAGE = 1, CURRENT = 2,
    NOT_SIMULATED = 3,  // pre-check (graph connectivity) skipped it, solver never invoked
    DIVERGENCE = 4      // solver was invoked but did not converge (or produced NaN)
};

#endif  // LIMIT_VIOLATION_TYPES_HPP
