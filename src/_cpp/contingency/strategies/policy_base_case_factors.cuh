// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/policy_base_case_factors.cuh
// =============================================================================
//
// PolicyBaseCaseFactors — NR policy for ContingencyDriver.
//
// Behaviour
// ---------
//   needs_fresh_jacobian = false  → driver never calls fill_J_kernel.
//   needs_iter0_jacobian = false  → driver never calls fill_J_kernel.
//
//   initialize_from_base():
//     Tiles d_J_values_base (base-case J) into every batch slot, then calls
//     solver.set_values() + solver.factor().  This is the only factorization
//     ever performed.
//
//   solve_iter:
//     SOLVE only — the d_J_values_batch argument is intentionally ignored.
//     LU factors were set once during initialize_from_base() and never change.
//
// This is the "frozen Jacobian" / "dishonest Newton" approximation.  Accuracy
// is coarser than PolicyRefactorEvery but the factorize cost is paid only once.
//
// Timing note
// -----------
// The one-time factor() call happens inside initialize_from_base(), which is
// invoked from ContingencyAnalysisSolver's constructor before the
// cs.synchronize() + t_analysis_ms_ wall-clock capture.  Consequently the
// factorization cost is absorbed into t_analysis_ms_ (not t_first_factorize).
// t_first_factorize and t_refactorize are both zero for this policy.
// n_refactorize is always zero.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef POLICY_BASE_CASE_FACTORS_CUH
#define POLICY_BASE_CASE_FACTORS_CUH

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "cudss_batch_solver.cuh"
#include "../../cuda_utils.h"      // CudaTimer
#include "../../dtypes.hpp"         // cuda_real_type
#include "../../nr_iter_step.cuh"  // NrIterTimings
#include "../../acpf_nr_state.cuh" // AcPfNrState

struct PolicyBaseCaseFactors {

    static constexpr bool needs_fresh_jacobian = false;
    static constexpr bool needs_iter0_jacobian = false;

    // =========================================================================
    // initialize_from_base()
    //
    // Tiles d_J_values_base (single-system base-case J values) into every slot
    // of d_J_values_batch, then FACTORIZES using those values.
    //
    // All GPU work is async on cs; the caller (ContingencyAnalysisSolver ctor)
    // synchronizes the stream immediately after this call.
    // =========================================================================
    void initialize_from_base(CudssBatchSolver&     solver,
                               const AcPfNrState&    base,
                               cuda_real_type*       d_J_values_batch,
                               int                   ubatch_size,
                               int                   nnz_J,
                               cudaStream_t          cs)
    {
        const cuda_real_type* src =
            thrust::raw_pointer_cast(base.d_J_values_base.data());

        for (int b = 0; b < ubatch_size; ++b) {
            cudaMemcpyAsync(
                d_J_values_batch + b * nnz_J,
                src,
                static_cast<size_t>(nnz_J) * sizeof(cuda_real_type),
                cudaMemcpyDeviceToDevice,
                cs);
        }

        solver.set_values(d_J_values_batch);
        solver.factor();
    }

    // begin_chunk() — no-op: factors are global across all chunks.
    void begin_chunk() {}

    // =========================================================================
    // solve_iter()
    //
    // SOLVE only — LU factors are already set from the base case.
    // The d_J_values_batch argument is intentionally ignored.
    // Returns false (no REFACTORIZATION; n_refactorize never incremented).
    // =========================================================================
    bool solve_iter(CudssBatchSolver& solver,
                    cuda_real_type*   /* d_J_values_batch — ignored */,
                    cudaStream_t      /* cs */,
                    CudaTimer&        timer,
                    NrIterTimings&    t)
    {
        timer.start();
        solver.solve();
        t.t_solve += timer.stop_ms();
        return false;
    }
};

#endif // POLICY_BASE_CASE_FACTORS_CUH