// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/policy_refactor_every.cuh
// =============================================================================
//
// PolicyRefactorEvery — NR policy for ContingencyDriver.
//
// Behaviour
// ---------
//   needs_fresh_jacobian = true  → driver calls fill_J_kernel every NR iteration.
//   needs_iter0_jacobian = false
//
//   solve_iter:
//     First call ever (across all chunks): set_values + FACTORIZE + SOLVE.
//     All subsequent calls:                set_values + REFACTORIZE + SOLVE.
//
// This reproduces the behaviour of the original DirectRefactorEvery strategy,
// now without owning any cuDSS resources.  The CudssBatchSolver instance is
// passed into solve_iter() by the driver.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef POLICY_REFACTOR_EVERY_CUH
#define POLICY_REFACTOR_EVERY_CUH

#include "cudss_batch_solver.cuh"
#include "../../cuda_utils.h"      // CudaTimer
#include "../../dtypes.hpp"         // cuda_real_type
#include "../../nr_iter_step.cuh"  // NrIterTimings
#include "../../acpf_nr_state.cuh" // AcPfNrState

struct PolicyRefactorEvery {

    static constexpr bool needs_fresh_jacobian = true;
    static constexpr bool needs_iter0_jacobian = false;

    bool factorized_ = false;  // global: FACTORIZE first call, REFACTORIZE thereafter

    // initialize_from_base() — no-op: this policy always builds fresh factors.
    void initialize_from_base(CudssBatchSolver&, const AcPfNrState&,
                               cuda_real_type*, int, int, cudaStream_t) {}

    // begin_chunk() — no-op: factorized_ is global across all chunks.
    void begin_chunk() {}

    // =========================================================================
    // solve_iter()
    //
    // set_values + FACTORIZE (first call) or REFACTORIZE (all subsequent) + SOLVE.
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
        const bool is_refactor = factorized_;
        if (!factorized_) {
            solver.factor();
            t.t_first_factorize += timer.stop_ms();
            factorized_ = true;
        } else {
            solver.refactor();
            t.t_refactorize += timer.stop_ms();
        }

        timer.start();
        solver.solve();
        t.t_solve += timer.stop_ms();

        return is_refactor;
    }
};

#endif // POLICY_REFACTOR_EVERY_CUH