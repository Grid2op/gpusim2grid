// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef LIMIT_VIOLATION_TYPES_HPP
#define LIMIT_VIOLATION_TYPES_HPP

// =============================================================================
// contingency/limit_violation_types.hpp
//
// Shared int codes for compute_limit_violations, mirroring lightsim2grid's
// ls2g::ViolationElementType / ls2g::LimitViolationType (LimitViolation.hpp)
// exactly for element_type and the first three violation_type values.
// DIVERGED (=3) is a gpusim2grid-only extension: the fused GPU kernel already
// computes a per-contingency residual check as a precondition to trusting V
// for the bus/branch checks, so folding a DIVERGED record into the same
// compact output avoids a second round trip for callers of get_violations().
//
// Not pybind-bound: the kernel writes raw ints, and the Python facade mirrors
// these values as plain enum.IntEnum. Kept in one named place so the codes
// used by the kernel (violation_kernels.cu), the Python facade
// (_limit_violations.py), and this doc all agree by construction.
// =============================================================================

enum class ViolationElementType : int { BUS = 0, LINE = 1, TRAFO = 2 };
enum class LimitViolationType  : int { LOW_VOLTAGE = 0, HIGH_VOLTAGE = 1, CURRENT = 2, DIVERGED = 3 };

#endif  // LIMIT_VIOLATION_TYPES_HPP
