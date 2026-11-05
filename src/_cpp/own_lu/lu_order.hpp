// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// own_lu/lu_order.hpp -- the elimination order of the batched LU of our own
// =============================================================================
//
// The own LU never pivots: its order is everything (lu_plan.hpp). Chosen on
// the host from the session's reordering_alg, with no cuDSS ANALYSIS:
//
//   default            lightsim2grid's constrained order when the grid declares
//                      ordering constraints (AMD post-moved so that a row the
//                      outer loops switch by value keeps a valid pivot), else
//                      AMD
//   amd                AMD of A + A^T, constraints ignored
//   none               the identity
//   ls2g_constrained / ls2g_constrained_nd
//                      the order AcPfNrState already computed for cuDSS
//                      (user_perm, required)
//   anything else      std::invalid_argument
//
// AMD is lightsim2grid's (SuiteSparse, through the constrained-order hook with
// no constraint) when the bridge is compiled in, Eigen's otherwise. CUDA-free.
// =============================================================================

#ifndef GPUSIM2GRID_OWN_LU_ORDER_HPP
#define GPUSIM2GRID_OWN_LU_ORDER_HPP

#include <vector>

#include "../reordering_alg.hpp"

struct LedgerData;

namespace own_lu {

// AMD of the pattern of A + A^T (new -> old).
std::vector<int> amd_order(int n, const std::vector<int>& indptr, const std::vector<int>& indices);

// ord_ledger: the J skeleton and the grid's ordering constraints (nullptr in
// array mode); user_perm: AcPfNrState's order for the ls2g_* algorithms.
std::vector<int> choose_order(ReorderingAlg alg, int n, const std::vector<int>& indptr,
                              const std::vector<int>& indices, const LedgerData* ord_ledger,
                              const std::vector<int>& user_perm);

}  // namespace own_lu

#endif  // GPUSIM2GRID_OWN_LU_ORDER_HPP
