// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef REORDERING_ALG_HPP
#define REORDERING_ALG_HPP

// =============================================================================
// reordering_alg.hpp
//
// ReorderingAlg mirrors cudssReorderingAlg_t (cudss_data_types.h) and selects
// CUDSS_CONFIG_REORDERING_ALG ahead of CUDSS_PHASE_ANALYSIS (reordering +
// symbolic factorization). Deliberately CUDA-free (no cudss.h include) so it
// can be used from pure-C++ headers (acpf_nr.hpp, *_session.hpp, ls2g_bridge.hpp)
// without pulling CUDA/cuDSS headers into host-only translation units.
// The mapping to the real cudssReorderingAlg_t values lives in cuda_utils.h
// (to_cudss_reordering_alg()), which is only included from CUDA translation
// units.
// =============================================================================

enum class ReorderingAlg {
    Default,
    BtfColamd,
    Colamd,
    Amd,
    NestedDissection,
    None
};

#endif  // REORDERING_ALG_HPP
