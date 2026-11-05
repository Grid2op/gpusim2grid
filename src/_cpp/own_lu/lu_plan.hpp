// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// own_lu/lu_plan.hpp -- symbolic part of the batched LU of our own
// =============================================================================
//
// A batched sparse LU for the uniform batch (every matrix one sparsity
// pattern), ported from gpu-security-analysis' `engine/lu.py` (RTE-AssistFlux,
// docs/own_lu.md there; docs/dev_notes/batched_lu_and_constrained_ordering.md
// here). Everything symbolic is fixed once per pattern, on the host, CUDA-free:
//
//   - a symmetric elimination order (new -> old, the convention of AMD's P and
//     of CUDSS_DATA_USER_PERM / PERM_REORDER_ROW): lightsim2grid's constrained
//     order, or cuDSS's own;
//   - the LU pattern: the Cholesky fill of the symmetrized permuted pattern, a
//     superset of the LU fill without pivoting (no pivoting at all: static
//     diagonal pivots, as a KLU refactorization);
//   - one CSR holding both factors: row i = its L part [rp[i], dp[i]), the
//     pivot dp[i], its U part (dp[i], rp[i+1]);
//   - the level schedules: of the left-looking factorization (each LU entry is
//     computed once, gathering its contributions L(i,k)·U(k,j) in pivot order:
//     deterministic, no atomics; step 2·level[p] = the U entries of row p, the
//     pivot included, step 2·level[p]+1 = the L entries of column p), and of
//     the two triangular solves.
//
// The device side (batched_lu.cuh) only streams values along these lists.
// =============================================================================

#ifndef GPUSIM2GRID_OWN_LU_PLAN_HPP
#define GPUSIM2GRID_OWN_LU_PLAN_HPP

#include <vector>

namespace own_lu {

// One launch of a level schedule: `count` consecutive items from `offset`.
// `max_chain`: the longest run of contributions of an item (entries of a row
// for the solves), what decides the split kernels near the root.
struct LevelSlice {
    int  offset    = 0;
    int  count     = 0;
    int  max_chain = 0;
    bool is_u      = false;  // factorization only: U step (else L step)
};

struct LuPlan {
    int n = 0;          // dimension
    int nnz = 0;        // non-zeros of A (the pattern handed in)
    int n_ent = 0;      // non-zeros of the LU pattern (L + U + diagonal)
    int levels = 0;     // factorization levels (elimination tree height + 1)
    long long contributions = 0;  // multiply-adds per matrix

    std::vector<int> order;       // new -> old
    std::vector<int> inv_order;   // old -> new
    std::vector<int> rp, dp, ci;  // LU CSR, in the permuted numbering
    std::vector<int> a_ent;       // A entry (CSR position) -> LU entry

    // Factorization items in schedule order: target entry (~entry for a fill
    // entry, i.e. one not in A: its sum starts at 0, so nothing is zeroed),
    // contributions [cptr[i], cptr[i+1]) into cl / cu (L and U entries), and
    // the pivot entry of the item's step.
    std::vector<int> f_tgt, f_cptr, f_dg;
    std::vector<int> cl, cu;
    std::vector<LevelSlice> factor_steps;

    // Triangular solves: rows in level order, one slice per level.
    std::vector<int> fw_rows, bw_rows;
    std::vector<LevelSlice> fw_levels, bw_levels;
};

// order: new -> old, a permutation of [0, n). Throws std::invalid_argument on a
// pattern / order mismatch.
LuPlan build_lu_plan(int n, const std::vector<int>& indptr, const std::vector<int>& indices,
                     const std::vector<int>& order);

}  // namespace own_lu

#endif  // GPUSIM2GRID_OWN_LU_PLAN_HPP
