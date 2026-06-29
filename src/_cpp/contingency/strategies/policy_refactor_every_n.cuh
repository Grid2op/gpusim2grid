// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/policy_refactor_every_n.cuh
// =============================================================================
//
// PolicyRefactorEveryN — NR policy for ContingencyDriver.
//
// Behaviour
// ---------
//   needs_fresh_jacobian = true  → driver calls fill_J_kernel every NR iteration.
//   needs_iter0_jacobian = false
//
//   solve_iter:
//     Call 0 (globally first):            set_values + FACTORIZE + SOLVE.
//     Subsequent calls where call % n == 0: set_values + REFACTORIZE + SOLVE.
//     All other calls:                    set_values + SOLVE only.
//
// call_count_ is global across all chunks — begin_chunk() is a no-op.
// Setting n=1 degenerates to the same behaviour as PolicyRefactorEvery.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef POLICY_REFACTOR_EVERY_N_CUH
#define POLICY_REFACTOR_EVERY_N_CUH

#include "cudss_batch_solver.cuh"
#include "../../cuda_utils.h"      // CudaTimer
#include "../../dtypes.hpp"         // cuda_real_type
#include "../../nr_iter_step.cuh"  // NrIterTimings
#include "../../acpf_nr_state.cuh" // AcPfNrState

struct PolicyRefactorEveryN {

    static constexpr bool needs_fresh_jacobian = true;
    static constexpr bool needs_iter0_jacobian = false;

    int  n_          = 1;
    int  call_count_ = 0;
    bool factorized_ = false;

    explicit PolicyRefactorEveryN(int n) : n_(n < 1 ? 1 : n) {}

    // initialize_from_base() — no-op: this policy always builds fresh factors.
    void initialize_from_base(CudssBatchSolver&, const AcPfNrState&,
                               cuda_real_type*, int, int, cudaStream_t) {}

    // begin_chunk() — no-op: call_count_ is global across all chunks.
    void begin_chunk() {}

    // =========================================================================
    // solve_iter()
    //
    // Call 0:               set_values + FACTORIZE + SOLVE.
    // call_count_ % n_ == 0: set_values + REFACTORIZE + SOLVE.
    // Otherwise:            set_values + SOLVE only (reuse existing factors).
    // Returns true if REFACTORIZE was done (used by driver to count n_refactorize).
    // =========================================================================
    bool solve_iter(CudssBatchSolver& solver,
                    cuda_real_type*   d_J_values_batch,
                    cudaStream_t      /* cs */,
                    CudaTimer&        timer,
                    NrIterTimings&    t)
    {
        solver.set_values(d_J_values_batch);

        timer.start();
        bool is_refactor = false;
        if (!factorized_) {
            solver.factor();
            t.t_first_factorize += timer.stop_ms();
            factorized_ = true;
        } else if (call_count_ % n_ == 0) {
            solver.refactor();
            t.t_refactorize += timer.stop_ms();
            is_refactor = true;
        } else {
            timer.stop_ms();  // no factorize/refactorize on this call
        }

        ++call_count_;

        timer.start();
        solver.solve();
        t.t_solve += timer.stop_ms();

        return is_refactor;
    }
};

#endif // POLICY_REFACTOR_EVERY_N_CUH