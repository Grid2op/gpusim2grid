// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/policy_iter0_only.cuh
// =============================================================================
//
// PolicyIter0Only — NR policy for ContingencyDriver.
//
// Behaviour
// ---------
//   needs_fresh_jacobian = false
//   needs_iter0_jacobian = true  → driver calls fill_J_kernel at iter==0 only.
//
//   solve_iter:
//     iter==0 of each chunk (iter0_done_ == false):
//       set_values + FACTORIZE (first chunk) or REFACTORIZE (subsequent) + SOLVE.
//       Sets iter0_done_ = true.
//     iter>0 of each chunk (iter0_done_ == true):
//       SOLVE only — reuses the LU factors from iter==0.
//
// "Lagged Jacobian" / chord Newton: the Jacobian is frozen within each chunk.
// Factorization cost is nb_iter times cheaper than PolicyRefactorEvery.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef POLICY_ITER0_ONLY_CUH
#define POLICY_ITER0_ONLY_CUH

#include "cudss_batch_solver.cuh"
#include "../../cuda_utils.h"      // CudaTimer
#include "../../dtypes.hpp"         // cuda_real_type
#include "../../nr_iter_step.cuh"  // NrIterTimings
#include "../../acpf_nr_state.cuh" // AcPfNrState

struct PolicyIter0Only {

    static constexpr bool needs_fresh_jacobian = false;
    static constexpr bool needs_iter0_jacobian = true;

    bool factorized_ = false;  // global: FACTORIZE first call, REFACTORIZE thereafter
    bool iter0_done_ = false;  // per-chunk: reset to false in begin_chunk()

    // initialize_from_base() — no-op: builds fresh factors from the contingency J.
    void initialize_from_base(CudssBatchSolver&, const AcPfNrState&,
                               cuda_real_type*, int, int, cudaStream_t) {}

    // begin_chunk() — reset per-chunk flag so the next solve_iter triggers factor/refactor.
    void begin_chunk() { iter0_done_ = false; }

    // =========================================================================
    // solve_iter()
    //
    // iter==0 (iter0_done_ == false):
    //   set_values + FACTORIZE or REFACTORIZE + SOLVE.  Sets iter0_done_ = true.
    //   Returns true if REFACTORIZE was done.
    //
    // iter>0 (iter0_done_ == true):
    //   SOLVE only — reuses factors from this chunk's iter==0.
    //   Returns false.
    // =========================================================================
    bool solve_iter(CudssBatchSolver& solver,
                    cuda_real_type*   d_J_values_batch,
                    cudaStream_t      /* cs */,
                    CudaTimer&        timer,
                    NrIterTimings&    t)
    {
        if (!iter0_done_) {
            solver.set_values(d_J_values_batch);

            timer.start();
            const bool is_refactor = factorized_;
            if (!factorized_) {
                solver.factor();
                t.t_first_factorize += timer.stop_ms();
                factorized_ = true;
            } else {
                solver.refactor();
                t.t_refactorize += timer.stop_ms();
            }
            iter0_done_ = true;

            timer.start();
            solver.solve();
            t.t_solve += timer.stop_ms();

            return is_refactor;
        } else {
            timer.start();
            solver.solve();
            t.t_solve += timer.stop_ms();
            return false;
        }
    }
};

#endif // POLICY_ITER0_ONLY_CUH