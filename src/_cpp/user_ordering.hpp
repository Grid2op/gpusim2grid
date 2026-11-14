// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef USER_ORDERING_HPP
#define USER_ORDERING_HPP

// =============================================================================
// user_ordering.hpp -- prototype (docs/dev_notes/batched_lu_and_constrained_ordering.md)
//
// ReorderingAlg::Ls2gConstrained hands cuDSS lightsim2grid's constrained order
// of the batch Jacobian: ls2g::constrained_order (an AMD order of A + A^T
// post-moved so the grid's OrderingConstraints hold) on the FINAL J skeleton,
// with the constraints the bridge read into LedgerData. The function lives in
// lightsim2grid, which only the optional bridge links: the bridge installs it
// here at load time (ls2g_bridge.cpp); CUDA code calls it through the hook.
// CUDA-free.
// =============================================================================

#include <vector>

struct LedgerData;

// perm[k] = the J index eliminated k-th (AMD's P, new -> old). False on failure.
using ConstrainedOrderFn = bool (*)(int n, const int* row_ptr, const int* col_idx,
                                    const LedgerData& ledger, std::vector<int>& perm);

// Null when gpusim2grid was built without the lightsim2grid bridge.
ConstrainedOrderFn& constrained_order_hook();

// The order for this ledger's J skeleton, or throws (no bridge, constraints not
// read / invalidated, lightsim2grid failure).
std::vector<int> ls2g_constrained_perm(const LedgerData& ledger);

// The ledger's ordering constraints applied to a GIVEN base order (new -> old):
// lightsim2grid's post-move (the second half of ls2g::constrained_order,
// KLUStaticPivotSolver.cpp at b8bf92bb), copied here because lightsim2grid only
// exports it fused with its AMD / min-fill base. Throws like
// ls2g_constrained_perm when the constraints are not available. To upstream as a
// lightsim2grid function if the prototype is kept.
std::vector<int> apply_ordering_constraints(std::vector<int> perm, const LedgerData& ledger);

#endif  // USER_ORDERING_HPP
