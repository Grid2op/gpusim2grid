// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef LINEAR_SOLVER_HPP
#define LINEAR_SOLVER_HPP

// =============================================================================
// linear_solver.hpp
//
// Which engine factorizes and solves the batch Jacobians (forward, continuous
// tail, J^T adjoint). CUDA-free, like reordering_alg.hpp.
//
//   Cudss  cuDSS's uniform batch (CudssBatchSolver's default).
//   OwnLu  the batched LU of our own (own_lu/, docs/dev_notes/
//          batched_lu_and_constrained_ordering.md): static diagonal pivots
//          along an order chosen on the host (own_lu/lu_order.hpp), no cuDSS
//          call at all. The single-system base case stays on cuDSS.
// =============================================================================

enum class LinearSolver {
    Cudss,
    OwnLu
};

#endif  // LINEAR_SOLVER_HPP
