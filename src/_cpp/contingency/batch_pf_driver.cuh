// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BATCH_PF_DRIVER_CUH
#define BATCH_PF_DRIVER_CUH

// =============================================================================
// contingency/batch_pf_driver.cuh
//
// Templated batched-PF chunk driver — shared by N-k contingency analysis and
// the injection sweep.  Parameterised by a small NON-VIRTUAL BatchSource
// policy that owns the per-chunk perturbation:
//
//   ContingencyBatch : tile V + tile Ybus + apply per-element patches.
//                      Sbus is the shared base-case (sbus_stride = 0).
//   InjectionBatch   : tile V only (Ybus tiled once at construction).
//                      Sbus is per-scenario (sbus_stride = n_bus).
//
// All other state (V_batch / Ybus_batch / J / dx / F buffers, cuDSS context,
// SpMV descriptor, linear-solve policy std::variant, residual + branch-flow
// outputs, dimensions, timings) is shared and lives on this class.
//
// The constructor performs:
//   1.  source.cpu_preprocess_ms()  → recorded into t_preprocess_ms_
//   2.  block-diagonal CSR structure upload (outer/inner only)
//   3.  chunk-sized device buffer allocation
//   4.  full-result buffer allocation (V_results + residuals)
//   5.  cuSPARSE block-diagonal SpMV descriptor
//   6.  CudssBatchSolver::initialize  (cuDSS ANALYSIS — once)
//   7.  std::visit policy.initialize_from_base
//   8.  source.initialize(ctx, cs)   — source-specific one-time work
//
// Per-chunk loop (in _solve_chunk):
//   a.  source.prepare_Ybus_batch(ctx, chunk_idx, actual_batch, …)
//   b.  source.prepare_Sbus_batch(ctx, chunk_idx, actual_batch, …)
//   c.  NrIterBuffers built with source.d_Sbus_ptr() + source.sbus_stride()
//   d.  std::visit policy → run_nr_loop<Policy>
//   e.  final SpMV + fill_F + per-element ‖F‖∞ + store-V (+ optional flows)
//
// Member declaration order is the same as the historic
// ContingencyAnalysisSolver: cs FIRST so it is destroyed LAST.
// =============================================================================

#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "../cuda_utils.h"
#include "../cu_complex_utils.h"
#include "../timing_utils.hpp"
#include "../acpf_nr_state.cuh"
#include "../contingency_analysis_helper.hpp"   // ContingencySolverType
#include "strategies/cudss_batch_solver.cuh"
#include "strategies/policy_refactor_every.cuh"
#include "strategies/policy_base_case_factors.cuh"
#include "strategies/policy_iter0_only.cuh"
#include "strategies/policy_refactor_every_n.cuh"

#include <variant>

#include "Eigen/Core"
#include "Eigen/SparseCore"

// -----------------------------------------------------------------------------
// BatchPfDriverContext — non-owning per-chunk view passed to the BatchSource
// hooks.  Plain struct (no templates) to keep BatchSource types decoupled from
// BatchPfDriver's template parameter.
// -----------------------------------------------------------------------------
struct BatchPfDriverContext {
    AcPfNrState&     base;
    cudaComplexType* d_V_batch;             // [batch_size × n_bus]
    cudaComplexType* d_Ybus_values_batch;   // [batch_size × nnz_Y]
    int              batch_size;
    int              n_bus;
    int              nnz_Y;
};

// =============================================================================
// BatchPfDriver<BatchSource>
// =============================================================================
template <typename BatchSource>
struct BatchPfDriver {

    // -------------------------------------------------------------------------
    // CUDA stream  (declared FIRST — destroyed LAST)
    // -------------------------------------------------------------------------
    CudaStream cs;

    // -------------------------------------------------------------------------
    // Reference to base-case state (not owned; must outlive this object)
    // -------------------------------------------------------------------------
    AcPfNrState& base;

    // -------------------------------------------------------------------------
    // Per-element batch source (owned by value — moved in at construction)
    // -------------------------------------------------------------------------
    BatchSource source_;

    // -------------------------------------------------------------------------
    // Dimensions
    //
    // n_contingencies is the count of batch elements; kept under this name
    // for backwards compatibility with the existing public surface
    // (ContingencyAnalysisSession reads it; BatchTimings exposes it).
    // For the injection sweep it is n_scenarios.
    // -------------------------------------------------------------------------
    int n_contingencies = 0;   // size of the result buffers (full set)
    int n_active_       = 0;    // systems actually solved (disconnected dropped)
    int batch_size_     = 0;
    int nb_iter_        = 0;
    int n_chunks_       = 0;    // ceil(n_active_ / batch_size_)

    // -------------------------------------------------------------------------
    // One-time construction timings (wall-clock ms)
    // -------------------------------------------------------------------------
    double t_preprocess_ms_ = 0.;
    double t_alloc_ms_      = 0.;
    double t_analysis_ms_   = 0.;

    // -------------------------------------------------------------------------
    // Block-diagonal Ybus structure (outer/inner only — values tiled per chunk)
    // -------------------------------------------------------------------------
    thrust::device_vector<int> d_Ybus_batch_outer;
    thrust::device_vector<int> d_Ybus_batch_inner;

    // -------------------------------------------------------------------------
    // Chunk-sized working buffers
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V_batch;
    thrust::device_vector<cudaComplexType> d_Ybus_values_batch;
    thrust::device_vector<cudaComplexType> d_Ibus_batch;
    thrust::device_vector<cuda_real_type>  d_F_batch;
    thrust::device_vector<cuda_real_type>  d_dx_batch;
    thrust::device_vector<cuda_real_type>  d_J_values_batch;

    // -------------------------------------------------------------------------
    // Full result buffers
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V_results;
    thrust::device_vector<cuda_real_type>  d_residuals;

    // -------------------------------------------------------------------------
    // Optional branch-flow outputs (only meaningful for contingency-style
    // features that mutate topology).  Both InjectionBatch and ContingencyBatch
    // can co-exist with this: InjectionBatch simply never calls set_branch_data.
    // -------------------------------------------------------------------------
    bool _has_branch_data = false;
    int  n_branches_      = 0;

    thrust::device_vector<int>             d_branch_from;
    thrust::device_vector<int>             d_branch_to;
    thrust::device_vector<cudaComplexType> d_yff;
    thrust::device_vector<cudaComplexType> d_yft;
    thrust::device_vector<cudaComplexType> d_ytf;
    thrust::device_vector<cudaComplexType> d_ytt;
    thrust::device_vector<cuda_real_type>  d_base_current_A;

    thrust::device_vector<cuda_real_type>  d_or_amps_results;
    thrust::device_vector<cuda_real_type>  d_ex_amps_results;

    // -------------------------------------------------------------------------
    // cuSPARSE batched SpMV
    // -------------------------------------------------------------------------
    CuSpMV spmv_batch;

    // -------------------------------------------------------------------------
    // Linear solver + NR policy variant
    // -------------------------------------------------------------------------
    CudssBatchSolver linear_solver_;

    std::variant<PolicyRefactorEvery,
                 PolicyBaseCaseFactors,
                 PolicyIter0Only,
                 PolicyRefactorEveryN> policy_;

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------
    //   base_state       — must outlive this object; cuDSS ANALYSIS already done.
    //   source           — moved into source_; its initialize(ctx, cs) is invoked
    //                      once construction reaches the GPU-setup phase.
    //   n_elements       — total contingencies / scenarios in this run.
    //   Ybus_rm_outer/inner — host RowMajor CSR of the base Ybus (used to build
    //                      the block-diagonal outer/inner; not retained).
    //   batch_size       — systems per chunk.
    //   nb_iter          — fixed NR iterations per chunk.
    //   strategy_type    — selects which Policy alternative emplaces into policy_.
    //   refactor_period  — for DirectRefactorEveryN.
    // -------------------------------------------------------------------------
    BatchPfDriver(
        AcPfNrState&              base_state,
        BatchSource               source,
        int                       n_contingencies_in,    // count of batch elements
        const int*                Ybus_rm_outer,
        const int*                Ybus_rm_inner,
        int                       batch_size,
        int                       nb_iter,
        ContingencySolverType     strategy_type   = ContingencySolverType::DirectRefactorEvery,
        int                       refactor_period = 1);

    ~BatchPfDriver() = default;

    // -------------------------------------------------------------------------
    // solve()  — iterate chunks; fills d_V_results + d_residuals.
    // -------------------------------------------------------------------------
    BatchTimings solve();

    // -------------------------------------------------------------------------
    // copy_results_to_host  — syncs cs and copies V_results + residuals.
    // -------------------------------------------------------------------------
    void copy_results_to_host(CplxVect& V_out, RealVect& res_out) const;

    // -------------------------------------------------------------------------
    // Branch-flow data  (no-op for features that do not call it)
    // -------------------------------------------------------------------------
    void set_branch_data(
        Eigen::Ref<const Eigen::VectorXi> branch_from,
        Eigen::Ref<const Eigen::VectorXi> branch_to,
        Eigen::Ref<const CplxVect>        yff,
        Eigen::Ref<const CplxVect>        yft,
        Eigen::Ref<const CplxVect>        ytf,
        Eigen::Ref<const CplxVect>        ytt,
        Eigen::Ref<const RealVect>        bus_vn_kv,
        double                            sn_mva);

    void copy_flow_results_to_host(RealVect& or_amps_out,
                                    RealVect& ex_amps_out) const;

    // -------------------------------------------------------------------------
    // DLPack / zero-copy accessors
    // -------------------------------------------------------------------------
    const cudaComplexType* d_V_results_ptr() const {
        return thrust::raw_pointer_cast(d_V_results.data());
    }
    int  n_bus()            const { return base.n_bus; }
    int  device_id()        const { return base.device_id_; }
    int  n_contingencies_() const { return n_contingencies; }
    void synchronize()            { cs.synchronize(); }

    // Non-copyable, non-movable
    BatchPfDriver(const BatchPfDriver&)            = delete;
    BatchPfDriver& operator=(const BatchPfDriver&) = delete;
    BatchPfDriver(BatchPfDriver&&)                 = delete;
    BatchPfDriver& operator=(BatchPfDriver&&)      = delete;

    // -------------------------------------------------------------------------
    // Context view  — built once in the ctor and reused per chunk by sources.
    // -------------------------------------------------------------------------
    BatchPfDriverContext make_context();

private:
    void _solve_chunk(int c_start, int actual_batch, BatchTimings& t);
};

#endif // BATCH_PF_DRIVER_CUH