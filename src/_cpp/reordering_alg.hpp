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
    None,
    // lightsim2grid's constrained order (prototype, docs/dev_notes/
    // batched_lu_and_constrained_ordering.md): an AMD order of the final J
    // skeleton post-moved by the grid's own OrderingConstraints
    // (ls2g::constrained_order), handed to cuDSS as CUDSS_DATA_USER_PERM.
    // Needs the lightsim2grid bridge; the batch solvers only (the single-system
    // base case keeps cuDSS's default).
    Ls2gConstrained,
    // The same constraints post-moving cuDSS's OWN order of that skeleton (its
    // default, nested dissection, read back from a throwaway ANALYSIS) instead
    // of AMD: AMD's tree has too little parallelism for cuDSS (8.7x slower
    // Newton rounds on a French snapshot, see the dev note).
    Ls2gConstrainedNd
};

#endif  // REORDERING_ALG_HPP
