// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef MATCHING_ALG_HPP
#define MATCHING_ALG_HPP

// =============================================================================
// matching_alg.hpp
//
// MatchingAlg mirrors cudssMatchingAlg_t (cudss_data_types.h) and selects
// CUDSS_CONFIG_MATCHING_ALG ahead of CUDSS_PHASE_ANALYSIS. Deliberately
// CUDA-free (no cudss.h include), same rationale as reordering_alg.hpp: usable
// from pure-C++ headers (acpf_nr.hpp, *_session.hpp, ls2g_bridge.hpp) without
// pulling CUDA/cuDSS headers into host-only translation units. The mapping to
// the real cudssMatchingAlg_t values lives in cuda_utils.h
// (to_cudss_matching_alg()), which is only included from CUDA translation
// units.
//
// Unlike ReorderingAlg, cuDSS has no separate "default" matching value:
// CUDSS_MATCHING_ALG_NONE (matching disabled) IS cuDSS's own default, so
// MatchingAlg::None is this enum's default member, mapping 1:1.
// =============================================================================

enum class MatchingAlg {
    None,
    MaxDiagCount,
    MaxMinDiag,
    MaxMinDiagAlt,
    MaxDiagSum,
    MaxDiagProduct,
    Auto
};

#endif  // MATCHING_ALG_HPP
