// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_pf_driver.cu
//
// Out-of-line method definitions for BatchPfDriver<BatchSource> + explicit
// template instantiations for the two supported sources.
// =============================================================================

#include "batch_pf_driver.cuh"
#include "batch_sources/contingency_batch.cuh"
#include "batch_sources/injection_batch.cuh"
#include "driver.cuh"                       // run_nr_loop<Policy>
#include "../acpf_nr_kernels.cuh"
#include "../nr_iter_step.cuh"              // NrIterBuffers, BS

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <cusparse.h>
#include <cudss.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>

// =============================================================================
// Error-checking macros (scoped local — same pattern as contingency_solver.cu)
// =============================================================================
#define CHK_CUDA_BPF(call)                                                    \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(                                         \
                std::string("[batch_pf] CUDA: ") + cudaGetErrorString(_e));   \
    } while(0)

#define CHK_CSP_BPF(call)                                                     \
    do {                                                                      \
        cusparseStatus_t _s = (call);                                         \
        if (_s != CUSPARSE_STATUS_SUCCESS)                                    \
            throw std::runtime_error(                                         \
                std::string("[batch_pf] cuSPARSE: ")                          \
                + cusparseGetErrorString(_s));                                \
    } while(0)

namespace {
inline double bpf_ms_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}
}

// =============================================================================
// Constructor
// =============================================================================
template <typename BatchSource>
BatchPfDriver<BatchSource>::BatchPfDriver(
    AcPfNrState&          base_state,
    BatchSource           source,
    int                   n_contingencies_in,
    const int*            Ybus_rm_outer,
    const int*            Ybus_rm_inner,
    int                   batch_size,
    int                   nb_iter,
    ContingencySolverType strategy_type,
    int                   refactor_period)
    : base(base_state)
    , source_(std::move(source))
    , n_contingencies(n_contingencies_in)
    , n_active_(source_.n_active())
    , batch_size_(batch_size)
    , nb_iter_(nb_iter)
      // Chunks are formed over the ACTIVE set: disconnected contingencies are
      // compacted out by the source and never occupy a batch slot.
    , n_chunks_((source_.n_active() + batch_size - 1) / batch_size)
{
    // Pin this driver's stream and allocations to the same device as base.
    CHK_CUDA_BPF(cudaSetDevice(base.device_id_));

    // Emplace the requested policy alternative.
    if (strategy_type == ContingencySolverType::DirectBaseCaseFactors)
        policy_.template emplace<PolicyBaseCaseFactors>();
    else if (strategy_type == ContingencySolverType::DirectIter0Only)
        policy_.template emplace<PolicyIter0Only>();
    else if (strategy_type == ContingencySolverType::DirectRefactorEveryN)
        policy_.template emplace<PolicyRefactorEveryN>(refactor_period);
    // else: leave as default-constructed PolicyRefactorEvery

    base.cs.synchronize();

    const int n_bus  = base.n_bus;
    const int n_pq   = base.n_pq;  (void)n_pq;
    const int n_pvpq = base.n_pvpq; (void)n_pvpq;
    const int dim_J  = base.dim_J;
    const int nnz_Y  = base.nnz_Y;
    const int nnz_J  = base.nnz_J;

    // Source-owned host preprocessing time was captured in the source ctor.
    t_preprocess_ms_ = source_.cpu_preprocess_ms();

    // -------------------------------------------------------------------------
    // Build block-diagonal CSR structure (outer/inner only).  Values are
    // tiled per chunk (ContingencyBatch) or once at construction (InjectionBatch).
    // -------------------------------------------------------------------------
    auto t_cpu_start = std::chrono::steady_clock::now();
    std::vector<int> h_batch_outer, h_batch_inner;
    build_blockdiag_csr(n_bus, nnz_Y,
                        Ybus_rm_outer, Ybus_rm_inner,
                        batch_size_,
                        h_batch_outer, h_batch_inner);
    t_preprocess_ms_ += bpf_ms_since(t_cpu_start);

    // -------------------------------------------------------------------------
    // Upload block-diagonal structure
    // -------------------------------------------------------------------------
    upload_h2d(d_Ybus_batch_outer, h_batch_outer.data(), h_batch_outer.size(), cs);
    upload_h2d(d_Ybus_batch_inner, h_batch_inner.data(), h_batch_inner.size(), cs);

    // -------------------------------------------------------------------------
    // Allocate chunk-sized working buffers
    // -------------------------------------------------------------------------
    {
        auto t_alloc_start = std::chrono::steady_clock::now();

        d_V_batch.resize(static_cast<size_t>(batch_size_) * n_bus);
        d_Ybus_values_batch.resize(static_cast<size_t>(batch_size_) * nnz_Y);
        d_Ibus_batch.resize(static_cast<size_t>(batch_size_) * n_bus);
        d_F_batch.resize(static_cast<size_t>(batch_size_) * dim_J);
        d_dx_batch.resize(static_cast<size_t>(batch_size_) * dim_J);
        d_J_values_batch.resize(static_cast<size_t>(batch_size_) * nnz_J);

        // Per-slot augmented-feature running state (allocated only when active).
        if (base.slack_col >= 0)
            d_slack_absorbed_batch.resize(batch_size_);
        if (base.n_vc_ctrl > 0)
            d_vc_q_batch.resize(static_cast<size_t>(batch_size_) * base.n_vc_ctrl);

        d_V_results.resize(static_cast<size_t>(n_contingencies) * n_bus);
        d_residuals.resize(n_contingencies, cuda_real_type(0));

        cs.synchronize();
        t_alloc_ms_ = bpf_ms_since(t_alloc_start);
    }

    // -------------------------------------------------------------------------
    // cuSPARSE block-diagonal SpMV descriptor (for the maximum batch_size).
    // The last chunk may be smaller, but the descriptor stays valid since the
    // block structure is identical within each block.
    // -------------------------------------------------------------------------
    CHK_CSP_BPF(cusparseCreate(&spmv_batch.handle));
    CHK_CSP_BPF(cusparseSetPointerMode(spmv_batch.handle, CUSPARSE_POINTER_MODE_HOST));
    CHK_CSP_BPF(cusparseSetStream(spmv_batch.handle, cs));

    {
        const int bd_rows = batch_size_ * n_bus;
        const int bd_cols = batch_size_ * n_bus;
        const int bd_nnz  = batch_size_ * nnz_Y;

        CHK_CSP_BPF(cusparseCreateConstCsr(
            &spmv_batch.mat,
            bd_rows, bd_cols, bd_nnz,
            thrust::raw_pointer_cast(d_Ybus_batch_outer.data()),
            thrust::raw_pointer_cast(d_Ybus_batch_inner.data()),
            thrust::raw_pointer_cast(d_Ybus_values_batch.data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_TYPE));

        CHK_CSP_BPF(cusparseCreateConstDnVec(
            &spmv_batch.vec_x,
            bd_cols,
            thrust::raw_pointer_cast(d_V_batch.data()),
            CUDA_C_TYPE));

        CHK_CSP_BPF(cusparseCreateDnVec(
            &spmv_batch.vec_y,
            bd_rows,
            thrust::raw_pointer_cast(d_Ibus_batch.data()),
            CUDA_C_TYPE));

        CHK_CSP_BPF(cusparseSpMV_bufferSize(
            spmv_batch.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_cplx_one, spmv_batch.mat, spmv_batch.vec_x,
            &h_cplx_zero, spmv_batch.vec_y,
            CUDA_C_TYPE, CUSPARSE_SPMV_ALG_DEFAULT,
            &spmv_batch.buf.size));

        if (spmv_batch.buf.size > 0)
            CHK_CUDA_BPF(cudaMalloc(&spmv_batch.buf.ptr, spmv_batch.buf.size));
    }

    // -------------------------------------------------------------------------
    // cuDSS uniform-batch context + ANALYSIS + policy init + source init
    // -------------------------------------------------------------------------
    auto t_cudss_start = std::chrono::steady_clock::now();

    linear_solver_.initialize(
        batch_size_,
        dim_J, nnz_J,
        thrust::raw_pointer_cast(base.d_J_outer.data()),
        thrust::raw_pointer_cast(base.d_J_inner.data()),
        thrust::raw_pointer_cast(d_J_values_batch.data()),
        thrust::raw_pointer_cast(d_F_batch.data()),
        thrust::raw_pointer_cast(d_dx_batch.data()),
        cs);

    std::visit([&](auto& policy) {
        policy.initialize_from_base(
            linear_solver_, base,
            thrust::raw_pointer_cast(d_J_values_batch.data()),
            batch_size_, nnz_J, cs);
    }, policy_);

    // Source-specific one-time setup (e.g. InjectionBatch tiles base Ybus here).
    {
        BatchPfDriverContext ctx = make_context();
        source_.initialize(ctx, cs);
    }

    cs.synchronize();
    t_analysis_ms_ = bpf_ms_since(t_cudss_start);
}

// =============================================================================
// make_context
// =============================================================================
template <typename BatchSource>
BatchPfDriverContext BatchPfDriver<BatchSource>::make_context()
{
    return BatchPfDriverContext{
        base,
        thrust::raw_pointer_cast(d_V_batch.data()),
        thrust::raw_pointer_cast(d_Ybus_values_batch.data()),
        batch_size_,
        base.n_bus,
        base.nnz_Y,
    };
}

// =============================================================================
// copy_results_to_host
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::copy_results_to_host(
    CplxVect& V_out, RealVect& res_out) const
{
    cs.synchronize();
    const int n_bus = base.n_bus;

    thrust::host_vector<cudaComplexType> h_V = d_V_results;
    V_out.resize(n_contingencies * n_bus);
    for (int i = 0; i < n_contingencies * n_bus; ++i)
        V_out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[i].x),
            static_cast<eigen_real_type>(h_V[i].y));

    thrust::host_vector<cuda_real_type> h_res = d_residuals;
    res_out.resize(n_contingencies);
    for (int i = 0; i < n_contingencies; ++i)
        res_out(i) = static_cast<eigen_real_type>(h_res[i]);
}

// =============================================================================
// set_branch_data  /  copy_flow_results_to_host
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff,
    Eigen::Ref<const CplxVect>        yft,
    Eigen::Ref<const CplxVect>        ytf,
    Eigen::Ref<const CplxVect>        ytt,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double                            sn_mva)
{
    n_branches_ = static_cast<int>(branch_from.size());

    {
        std::vector<int> h_from(n_branches_), h_to(n_branches_);
        for (int l = 0; l < n_branches_; ++l) {
            h_from[l] = branch_from(l);
            h_to[l]   = branch_to(l);
        }
        upload_h2d(d_branch_from, h_from.data(), n_branches_, cs);
        upload_h2d(d_branch_to,   h_to.data(),   n_branches_, cs);

        const double sqrt3 = std::sqrt(3.0);
        std::vector<cuda_real_type> h_base(n_branches_);
        for (int l = 0; l < n_branches_; ++l) {
            const double vn = bus_vn_kv(h_from[l]);
            h_base[l] = static_cast<cuda_real_type>(
                sn_mva * 1e6 / (sqrt3 * vn * 1e3));
        }
        upload_h2d(d_base_current_A, h_base.data(), n_branches_, cs);
    }

    {
        std::vector<cudaComplexType> h_yff(n_branches_), h_yft(n_branches_),
                                     h_ytf(n_branches_), h_ytt(n_branches_);
        for (int l = 0; l < n_branches_; ++l) {
            h_yff[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(yff(l).real()),
                static_cast<cuda_real_type>(yff(l).imag()));
            h_yft[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(yft(l).real()),
                static_cast<cuda_real_type>(yft(l).imag()));
            h_ytf[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(ytf(l).real()),
                static_cast<cuda_real_type>(ytf(l).imag()));
            h_ytt[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(ytt(l).real()),
                static_cast<cuda_real_type>(ytt(l).imag()));
        }
        upload_h2d(d_yff, h_yff.data(), n_branches_, cs);
        upload_h2d(d_yft, h_yft.data(), n_branches_, cs);
        upload_h2d(d_ytf, h_ytf.data(), n_branches_, cs);
        upload_h2d(d_ytt, h_ytt.data(), n_branches_, cs);
    }

    d_or_amps_results.assign(
        static_cast<size_t>(n_contingencies) * n_branches_, cuda_real_type(0));
    d_ex_amps_results.assign(
        static_cast<size_t>(n_contingencies) * n_branches_, cuda_real_type(0));

    _has_branch_data = true;
}

template <typename BatchSource>
void BatchPfDriver<BatchSource>::copy_flow_results_to_host(
    RealVect& or_amps_out, RealVect& ex_amps_out) const
{
    if (!_has_branch_data) {
        or_amps_out.resize(0);
        ex_amps_out.resize(0);
        return;
    }

    cs.synchronize();
    const int n = n_contingencies * n_branches_;
    thrust::host_vector<cuda_real_type> h_or = d_or_amps_results;
    thrust::host_vector<cuda_real_type> h_ex = d_ex_amps_results;

    or_amps_out.resize(n);
    ex_amps_out.resize(n);
    for (int i = 0; i < n; ++i) {
        or_amps_out(i) = static_cast<eigen_real_type>(h_or[i]);
        ex_amps_out(i) = static_cast<eigen_real_type>(h_ex[i]);
    }
}

// =============================================================================
// solve
// =============================================================================
template <typename BatchSource>
BatchTimings BatchPfDriver<BatchSource>::solve()
{
    BatchTimings t;
    t.n_contingencies  = n_contingencies;
    t.n_chunks         = n_chunks_;
    t.chunk_size       = batch_size_;
    t.nb_iter          = nb_iter_;
    t.t_preprocess_ms  = t_preprocess_ms_;
    t.t_alloc_ms       = t_alloc_ms_;
    t.t_analysis_ms    = t_analysis_ms_;

    // When the source has compacted disconnected contingencies out of the batch
    // (n_active_ < n_contingencies), the chunk loop only writes the connected
    // result slots.  Pre-fill the full result buffers with NaN so the dropped
    // (disconnected) slots are reported as NaN, matching the historic contract.
    if (n_active_ < n_contingencies) {
        static const cuda_real_type nan_r =
            std::numeric_limits<cuda_real_type>::quiet_NaN();
        const cudaComplexType nan_c = { nan_r, nan_r };
        thrust::fill(thrust::cuda::par.on(static_cast<cudaStream_t>(cs)),
                     d_V_results.begin(), d_V_results.end(), nan_c);
        thrust::fill(thrust::cuda::par.on(static_cast<cudaStream_t>(cs)),
                     d_residuals.begin(), d_residuals.end(), nan_r);
    }

    // Chunk loop runs over the ACTIVE set (c_start / actual_batch in active-slot
    // space); _solve_chunk scatters each result to its original index.
    for (int chunk = 0; chunk < n_chunks_; ++chunk) {
        const int c_start      = chunk * batch_size_;
        const int actual_batch = std::min(batch_size_, n_active_ - c_start);
        _solve_chunk(c_start, actual_batch, t);
    }

    cs.synchronize();
    return t;
}

// =============================================================================
// _solve_chunk
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::_solve_chunk(
    int c_start, int actual_batch, BatchTimings& t)
{
    const int n_bus  = base.n_bus;
    const int n_pvpq = base.n_pvpq;
    const int n_pq   = base.n_pq;
    const int dim_J  = base.dim_J;
    const int nnz_Y  = base.nnz_Y;
    const int nnz_J  = base.nnz_J;

    const int chunk = c_start / batch_size_;

    // Active-slot → original-index map (nullptr when no compaction: identity).
    const int* d_result_map = source_.d_result_map();

    CudaTimer timer(cs);

    // -------------------------------------------------------------------------
    // ①  BatchSource::prepare_Ybus_batch  — tile V (+ optionally tile Ybus
    //     and apply patches, depending on the source).
    // ②  BatchSource::prepare_Sbus_batch  — no-op (contingency) or row-slice
    //     copy + phantom-pad (injection).
    // -------------------------------------------------------------------------
    BatchPfDriverContext ctx = make_context();
    source_.prepare_Ybus_batch(ctx, chunk, actual_batch, cs, timer, t);
    source_.prepare_Sbus_batch(ctx, chunk, actual_batch, cs, timer, t);

    // -------------------------------------------------------------------------
    // ③  Fixed NR loop
    // -------------------------------------------------------------------------
    const cudaComplexType* d_Sbus_for_NR = source_.d_Sbus_ptr(ctx);
    const int sbus_stride = BatchSource::sbus_stride(n_bus);

    const NrIterBuffers buf {
        thrust::raw_pointer_cast(d_J_values_batch.data()),
        thrust::raw_pointer_cast(d_V_batch.data()),
        thrust::raw_pointer_cast(d_Ibus_batch.data()),
        thrust::raw_pointer_cast(d_F_batch.data()),
        thrust::raw_pointer_cast(d_dx_batch.data()),
        thrust::raw_pointer_cast(d_Ybus_values_batch.data()),
        thrust::raw_pointer_cast(base.d_Ybus_outer.data()),
        thrust::raw_pointer_cast(base.d_Ybus_inner.data()),
        thrust::raw_pointer_cast(base.d_map_j11.data()),
        thrust::raw_pointer_cast(base.d_map_j12.data()),
        thrust::raw_pointer_cast(base.d_map_j21.data()),
        thrust::raw_pointer_cast(base.d_map_j22.data()),
        thrust::raw_pointer_cast(base.d_p_buses.data()),
        thrust::raw_pointer_cast(base.d_p_rows.data()),
        base.n_p,
        thrust::raw_pointer_cast(base.d_q_buses.data()),
        thrust::raw_pointer_cast(base.d_q_rows.data()),
        base.n_q,
        thrust::raw_pointer_cast(base.d_theta_buses.data()),
        thrust::raw_pointer_cast(base.d_theta_cols.data()),
        base.n_theta,
        thrust::raw_pointer_cast(base.d_vm_buses.data()),
        thrust::raw_pointer_cast(base.d_vm_cols.data()),
        base.n_vm,
        d_Sbus_for_NR,
        sbus_stride,
        // ---- MultiSlack (shared feature data on base; per-slot state here) ----
        base.slack_col,
        base.n_slack,
        thrust::raw_pointer_cast(base.d_slack_prow.data()),
        thrust::raw_pointer_cast(base.d_slack_w.data()),
        thrust::raw_pointer_cast(base.d_slack_feat_pos.data()),
        thrust::raw_pointer_cast(d_slack_absorbed_batch.data()),
        // ---- HVDC angle-droop (all shared on base) ----
        base.n_hvdc,
        /*zero_J_before_fill=*/(base.n_hvdc > 0),
        thrust::raw_pointer_cast(base.d_hvdc_bus1.data()),
        thrust::raw_pointer_cast(base.d_hvdc_bus2.data()),
        thrust::raw_pointer_cast(base.d_hvdc_status.data()),
        thrust::raw_pointer_cast(base.d_hvdc_p0.data()),
        thrust::raw_pointer_cast(base.d_hvdc_k.data()),
        thrust::raw_pointer_cast(base.d_hvdc_lf1.data()),
        thrust::raw_pointer_cast(base.d_hvdc_lf2.data()),
        thrust::raw_pointer_cast(base.d_hvdc_r.data()),
        thrust::raw_pointer_cast(base.d_hvdc_pmax12.data()),
        thrust::raw_pointer_cast(base.d_hvdc_pmax21.data()),
        thrust::raw_pointer_cast(base.d_hvdc_prow1.data()),
        thrust::raw_pointer_cast(base.d_hvdc_prow2.data()),
        thrust::raw_pointer_cast(base.d_hvdc_h11.data()),
        thrust::raw_pointer_cast(base.d_hvdc_h12.data()),
        thrust::raw_pointer_cast(base.d_hvdc_h21.data()),
        thrust::raw_pointer_cast(base.d_hvdc_h22.data()),
        // ---- VoltageControl (shared feature data on base; per-slot q here) ----
        base.n_vc_ctrl,
        base.n_vc_grp,
        base.n_vc_share,
        base.n_vc_feat,
        thrust::raw_pointer_cast(base.d_vc_qrow.data()),
        thrust::raw_pointer_cast(base.d_vc_qcol.data()),
        thrust::raw_pointer_cast(base.d_vc_slope.data()),
        thrust::raw_pointer_cast(base.d_vc_reg_bus.data()),
        thrust::raw_pointer_cast(base.d_vc_vrow.data()),
        thrust::raw_pointer_cast(base.d_vc_grp_start.data()),
        thrust::raw_pointer_cast(base.d_vc_grp_count.data()),
        thrust::raw_pointer_cast(base.d_vc_vset.data()),
        thrust::raw_pointer_cast(base.d_vc_sh_row.data()),
        thrust::raw_pointer_cast(base.d_vc_sh_first.data()),
        thrust::raw_pointer_cast(base.d_vc_sh_other.data()),
        thrust::raw_pointer_cast(base.d_vc_sh_wfirst.data()),
        thrust::raw_pointer_cast(base.d_vc_sh_wother.data()),
        thrust::raw_pointer_cast(base.d_vc_feat_pos.data()),
        thrust::raw_pointer_cast(base.d_vc_feat_val.data()),
        thrust::raw_pointer_cast(d_vc_q_batch.data()),
    };

    // Re-initialise the per-slot feature state for this chunk (slack_absorbed =
    // Re(Σ Sbus_slot); controller reactive injection = 0). The NR loop runs over
    // the full padded batch, so initialise batch_size_ slots.
    if (base.slack_col >= 0)
        init_slack_absorbed_kernel<<<(batch_size_ + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_slack_absorbed_batch.data()),
            d_Sbus_for_NR, sbus_stride, n_bus, batch_size_);
    if (base.n_vc_ctrl > 0)
        CHK_CUDA_BPF(cudaMemsetAsync(
            thrust::raw_pointer_cast(d_vc_q_batch.data()), 0,
            d_vc_q_batch.size() * sizeof(cuda_real_type), cs));

    std::visit([&](auto& policy) {
        run_nr_loop(
            policy, linear_solver_, spmv_batch, buf,
            n_bus, n_pvpq, n_pq, dim_J, nnz_Y, nnz_J,
            /*batch_size=*/batch_size_,
            /*nb_iter=*/nb_iter_,
            cs, timer, t);
    }, policy_);

    // -------------------------------------------------------------------------
    // ④  Post-loop: final SpMV + fill_F + per-element ‖F‖∞
    // -------------------------------------------------------------------------
    timer.start();
    {
        spmv_batch.spmv();

        if (actual_batch > 0) {
            fill_FP_kernel<<<
                (actual_batch * base.n_p + BS - 1) / BS, BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_F_batch.data()),
                thrust::raw_pointer_cast(d_V_batch.data()),
                thrust::raw_pointer_cast(d_Ibus_batch.data()),
                d_Sbus_for_NR,
                thrust::raw_pointer_cast(base.d_p_buses.data()),
                thrust::raw_pointer_cast(base.d_p_rows.data()),
                base.n_p, n_bus, dim_J, actual_batch, sbus_stride);
            fill_FQ_kernel<<<
                (actual_batch * base.n_q + BS - 1) / BS, BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_F_batch.data()),
                thrust::raw_pointer_cast(d_V_batch.data()),
                thrust::raw_pointer_cast(d_Ibus_batch.data()),
                d_Sbus_for_NR,
                thrust::raw_pointer_cast(base.d_q_buses.data()),
                thrust::raw_pointer_cast(base.d_q_rows.data()),
                base.n_q, n_bus, dim_J, actual_batch, sbus_stride);
            // Augmented-feature contributions to the final residual (slack /
            // HVDC mismatch + VC bordered custom rows), using the converged state.
            nr_feature_mismatch(buf, n_bus, dim_J, actual_batch, cs);

            compute_residuals_kernel<<<
                actual_batch, BS,
                static_cast<size_t>(BS) * sizeof(cuda_real_type), cs>>>(
                thrust::raw_pointer_cast(d_residuals.data()),
                thrust::raw_pointer_cast(d_F_batch.data()),
                dim_J, actual_batch, c_start, d_result_map);
        }
    }
    t.t_residual += timer.stop_ms();

    // -------------------------------------------------------------------------
    // ⑤  Store V results.  Without compaction the active slots map contiguously
    //     to result indices (single D→D memcpy).  With compaction the original
    //     indices are non-contiguous, so scatter through d_result_map instead.
    // -------------------------------------------------------------------------
    timer.start();
    if (actual_batch > 0) {
        if (d_result_map) {
            const int total = actual_batch * n_bus;
            scatter_V_results_kernel<<<(total + BS - 1) / BS, BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_V_results.data()),
                thrust::raw_pointer_cast(d_V_batch.data()),
                d_result_map, c_start, n_bus, actual_batch);
            CHK_CUDA_BPF(cudaGetLastError());
        } else {
            CHK_CUDA_BPF(cudaMemcpyAsync(
                thrust::raw_pointer_cast(d_V_results.data())
                    + static_cast<ptrdiff_t>(c_start) * n_bus,
                thrust::raw_pointer_cast(d_V_batch.data()),
                static_cast<size_t>(actual_batch) * n_bus * sizeof(cudaComplexType),
                cudaMemcpyDeviceToDevice, cs));
        }
    }
    t.t_store_V += timer.stop_ms();

    // -------------------------------------------------------------------------
    // ⑥  Optional branch-flow computation
    // -------------------------------------------------------------------------
    if (_has_branch_data && actual_batch > 0) {
        timer.start();
        const int total = actual_batch * n_branches_;
        compute_branch_flows_kernel<<<(total + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_V_batch.data()),
            thrust::raw_pointer_cast(d_branch_from.data()),
            thrust::raw_pointer_cast(d_branch_to.data()),
            thrust::raw_pointer_cast(d_yff.data()),
            thrust::raw_pointer_cast(d_yft.data()),
            thrust::raw_pointer_cast(d_ytf.data()),
            thrust::raw_pointer_cast(d_ytt.data()),
            thrust::raw_pointer_cast(d_base_current_A.data()),
            thrust::raw_pointer_cast(d_or_amps_results.data()),
            thrust::raw_pointer_cast(d_ex_amps_results.data()),
            n_bus, n_branches_, c_start, actual_batch, d_result_map);
        CHK_CUDA_BPF(cudaGetLastError());
        t.t_flow_computation += timer.stop_ms();
    }
}

// =============================================================================
// Explicit template instantiations
// =============================================================================
template struct BatchPfDriver<ContingencyBatch>;
template struct BatchPfDriver<InjectionBatch>;