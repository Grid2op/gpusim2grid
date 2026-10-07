// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ROW_STATUS_HPP
#define ROW_STATUS_HPP

// =============================================================================
// contingency/row_status.hpp
//
// CUDA-free enums of the batch schedulers (see docs/dev_notes/
// continuous_batching.md), shared by BatchPfDriver, the sessions and the
// Python bindings.
//
//   BatchScheduling : Chunked    -- ceil(n_active / batch_size) chunks, each
//                                   solved for exactly nb_iter iterations.
//                     Continuous -- batch_size slots; every round runs
//                                   nb_iter_per_round iterations on all of
//                                   them, a row leaves as soon as it has
//                                   converged (residual < tol), diverged or
//                                   used its nb_iter budget, and its slot is
//                                   refilled from the queue at once.
//   RowStatus       : the outcome of one row. Kept small and append-only: the
//                     outer loops (Part B of the design note) add their own
//                     outcomes after these.
// =============================================================================

enum class BatchScheduling : int {
    Chunked    = 0,
    Continuous = 1,
};

enum class RowStatus : int {
    Converged    = 0,   // ||F||inf < tol
    MaxIter      = 1,   // nb_iter budget used, residual finite but >= tol
    Diverged     = 2,   // residual not finite
    NotSimulated = 3,   // dropped by the pre-check, never solved
};

// Default convergence tolerance of a row, ||F||inf < tol with F in per unit
// (lightsim2grid compares the same way but takes its tol in MVA: tol / sn_mva).
// 1e-8 in double precision. A float build cannot reach it: its residual floor
// is around 1e-4 already on case118 (rows hover at 1.2-1.7e-4 whatever the
// iteration count), so it uses 1e-3.
constexpr double default_row_tol(bool fp32) { return fp32 ? 1e-3 : 1e-8; }

#endif  // ROW_STATUS_HPP
