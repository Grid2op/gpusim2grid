// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/cudss_batch_solver.cuh
// =============================================================================
//
// CudssBatchSolver — owns the cuDSS uniform-batch context and the three
// matrix/vector descriptors (A, x, b).  Exposes four operations:
//
//   initialize()   — create handle/config/data, build descriptors, run ANALYSIS
//   set_values()   — update the J values pointer in dss_A_ (CPU call, no GPU work)
//   factor()       — CUDSS_PHASE_FACTORIZATION  (first numeric factorization)
//   refactor()     — CUDSS_PHASE_REFACTORIZATION (subsequent refactorizations)
//   solve()        — CUDSS_PHASE_SOLVE
//
// This struct owns only the linear-algebra machinery.  It holds no state about
// *when* to factorize — that is left entirely to the policy that uses it.
//
// Destruction order
// -----------------
// Declared members top-to-bottom; destroyed bottom-to-top by C++ rules.
// dss_ must be declared first so it is destroyed last: CudssContext::~CudssContext
// calls cudssDataDestroy(handle, data) which requires a live handle.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef CUDSS_BATCH_SOLVER_CUH
#define CUDSS_BATCH_SOLVER_CUH

#include <chrono>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cudss.h>

#include "../../cuda_utils.h"  // CudssContext, CudssDescriptor
#include "../../dtypes.hpp"     // cuda_real_type, CUDSS_R_TYPE

struct CudssBatchSolver {

    // =========================================================================
    // cuDSS uniform-batch context (owned)
    //
    // Declaration order (destruction is reverse):
    //   dss_b_ (last declared) → destroyed first
    //   dss_x_
    //   dss_A_
    //   dss_  (first declared) → destroyed last
    // =========================================================================
    CudssContext    dss_;
    CudssDescriptor dss_A_;
    CudssDescriptor dss_x_;
    CudssDescriptor dss_b_;

    // Wall-clock ms of the handle/config/data creation inside initialize();
    // see context_init_ms() below.
    double t_context_init_ms_ = 0.;

    // =========================================================================
    // initialize()
    //
    // Creates the cuDSS context, builds the three descriptors, sets the
    // uniform-batch size, and runs the one-time ANALYSIS phase (reordering +
    // symbolic factorization).
    //
    // Parameters (all device pointers unless noted):
    //   ubatch_size       — number of systems in the uniform batch
    //   dim_J             — Jacobian dimension (n_pvpq + n_pq)
    //   nnz_J             — number of non-zeros in J (per system)
    //   d_J_outer         — J row-pointer array  (single-system, size dim_J+1)
    //   d_J_inner         — J column-index array (single-system, size nnz_J)
    //   d_J_values_batch  — J values batch buffer (ubatch_size * nnz_J)
    //   d_F_batch         — F (RHS) batch buffer  (ubatch_size * dim_J)
    //   d_dx_batch        — dx (solution) batch buffer (ubatch_size * dim_J)
    //   cs                — CUDA stream on which cuDSS will be bound
    //   reordering_alg    — CUDSS_CONFIG_REORDERING_ALG choice (default: cuDSS's own default)
    //   matching_alg      — CUDSS_CONFIG_MATCHING_ALG choice (default: cuDSS's own default, matching off)
    //   pivot_epsilon_alg — CUDSS_CONFIG_PIVOT_EPSILON_ALG choice (default: cuDSS's own default)
    //
    // Async: ANALYSIS is launched on cs; the caller must synchronize cs after
    // this call (or before consuming any results).
    // =========================================================================
    void initialize(int ubatch_size, int dim_J, int nnz_J,
                    int*            d_J_outer,
                    int*            d_J_inner,
                    cuda_real_type* d_J_values_batch,
                    cuda_real_type* d_F_batch,
                    cuda_real_type* d_dx_batch,
                    cudaStream_t    cs,
                    ReorderingAlg   reordering_alg = ReorderingAlg::Default,
                    MatchingAlg     matching_alg = MatchingAlg::None,
                    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default)
    {
        auto chk = [](cudssStatus_t s, const char* msg) {
            if (s != CUDSS_STATUS_SUCCESS)
                throw std::runtime_error(
                    std::string("[CudssBatchSolver::initialize] ") + msg
                    + ": cuDSS status=" + std::to_string(static_cast<int>(s)));
        };

        // Context creation is timed separately from ANALYSIS below: on a
        // process' first cuDSS call it dlopens + JITs the backend (tens of ms,
        // independent of problem size), which would otherwise inflate what
        // callers read as symbolic-analysis cost. See context_init_ms().
        auto t_ctx_start = std::chrono::steady_clock::now();

        chk(cudssCreate(&dss_.handle),                "cudssCreate");
        chk(cudssSetStream(dss_.handle, cs),          "cudssSetStream");
        chk(cudssConfigCreate(&dss_.config),          "cudssConfigCreate");
        chk(cudssDataCreate(dss_.handle, &dss_.data), "cudssDataCreate");

        t_context_init_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_ctx_start).count();

        dss_A_.create_csr(dim_J, nnz_J, d_J_outer, d_J_inner, d_J_values_batch);
        dss_x_.create_dn(dim_J, d_dx_batch);
        dss_b_.create_dn(dim_J, d_F_batch);

        chk(cudssConfigSet(dss_.config, CUDSS_CONFIG_UBATCH_SIZE,
                           &ubatch_size, sizeof(ubatch_size)), "cudssConfigSet");
        dss_.set_reordering_alg(reordering_alg);
        dss_.set_matching_alg(matching_alg);
        dss_.set_pivot_epsilon_alg(pivot_epsilon_alg);

        dss_.analyze(dss_A_, dss_x_, dss_b_);
    }

    // =========================================================================
    // set_values()
    //
    // Notifies cuDSS of the current J values pointer.  CPU call — no GPU work
    // is launched.  Must be called before the first factor() or refactor() when
    // new J values have been written into d_J_values_batch.
    // =========================================================================
    void set_values(cuda_real_type* d_J_values_batch)
    {
        dss_A_.set_values(d_J_values_batch);
    }

    // =========================================================================
    // factor()   — CUDSS_PHASE_FACTORIZATION  (first numeric factorization)
    // refactor() — CUDSS_PHASE_REFACTORIZATION (subsequent refactorizations)
    // solve()    — CUDSS_PHASE_SOLVE
    //
    // All three are async on the stream bound at initialize() time.
    // =========================================================================
    // Wall-clock ms spent creating the cuDSS handle/config/data in
    // initialize() — pure library/context setup, no ANALYSIS. Dominated by
    // first-touch dlopen + JIT on the first cuDSS use in the process, ~0
    // afterwards.
    double context_init_ms() const { return t_context_init_ms_; }

    void factor()   { dss_.factorize  (dss_A_, dss_x_, dss_b_); }
    void refactor() { dss_.refactorize(dss_A_, dss_x_, dss_b_); }
    void solve()    { dss_.solve      (dss_A_, dss_x_, dss_b_); }

    // Non-copyable, non-movable (cuDSS descriptors hold raw device pointers).
    CudssBatchSolver() = default;
    CudssBatchSolver(const CudssBatchSolver&)            = delete;
    CudssBatchSolver& operator=(const CudssBatchSolver&) = delete;
    CudssBatchSolver(CudssBatchSolver&&)                 = delete;
    CudssBatchSolver& operator=(CudssBatchSolver&&)      = delete;
};

#endif // CUDSS_BATCH_SOLVER_CUH