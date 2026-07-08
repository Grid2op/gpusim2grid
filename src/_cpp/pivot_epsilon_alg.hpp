// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PIVOT_EPSILON_ALG_HPP
#define PIVOT_EPSILON_ALG_HPP

// =============================================================================
// pivot_epsilon_alg.hpp
//
// PivotEpsilonAlg mirrors cudssPivotEpsilonAlg_t (cudss_data_types.h) and
// selects CUDSS_CONFIG_PIVOT_EPSILON_ALG ahead of CUDSS_PHASE_ANALYSIS.
// Deliberately CUDA-free (no cudss.h include), same rationale as
// reordering_alg.hpp / matching_alg.hpp: usable from pure-C++ headers
// (acpf_nr.hpp, *_session.hpp, ls2g_bridge.hpp) without pulling CUDA/cuDSS
// headers into host-only translation units. The mapping to the real
// cudssPivotEpsilonAlg_t values lives in cuda_utils.h
// (to_cudss_pivot_epsilon_alg()), which is only included from CUDA
// translation units.
//
// Unlike MatchingAlg, there is no "off" value here: Default maps 1:1 to
// cuDSS's own default, same treatment as ReorderingAlg::Default.
// =============================================================================

enum class PivotEpsilonAlg {
    Default,
    Scaled,
    Static
};

#endif  // PIVOT_EPSILON_ALG_HPP
