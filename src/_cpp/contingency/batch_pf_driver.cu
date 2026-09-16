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
#include "batch_sources/scenario_sweep_batch.cuh"
#include "driver.cuh"                       // run_nr_loop<Policy>
#include "../acpf_nr_kernels.cuh"
#include "../nr_iter_step.cuh"              // NrIterBuffers, BS
#include "../batch_dims_check.hpp"          // check_batch_stride
#include "violation_kernels.cuh"            // check_limit_violations_kernel

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
// blockdiag_csr_kernel
//   Block-diagonal CSR structure of the batched Ybus, written from the
//   single-system arrays already on the device: row (b*n_bus + r) starts at
//   b*nnz + outer[r], entry (b*nnz + i) sits in column b*n_bus + inner[i], and
//   the last row pointer is batch_size*nnz. The inner array is batch_size*nnz
//   ints -- close to a gigabyte at a 10k batch on a 7k-bus grid -- so it is
//   generated where it lives rather than built on the host (a quarter of a
//   second of scattered writes) and pushed through a pageable H→D copy.
//   batch_size*nnz and batch_size*n_bus fit an int (check_batch_stride).
// =============================================================================
__global__ void blockdiag_csr_kernel(
    int n_bus, int nnz, int batch_size,
    const int* __restrict__ outer, const int* __restrict__ inner,
    int* __restrict__ batch_outer, int* __restrict__ batch_inner)
{
    const long long tid     = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    const long long n_outer = static_cast<long long>(batch_size) * n_bus + 1;
    const long long n_inner = static_cast<long long>(batch_size) * nnz;
    if (tid < n_inner) {
        const int b = static_cast<int>(tid / nnz);
        const int i = static_cast<int>(tid - static_cast<long long>(b) * nnz);
        batch_inner[tid] = inner[i] + b * n_bus;
    }
    if (tid < n_outer) {
        if (tid == n_outer - 1) {
            batch_outer[tid] = batch_size * nnz;
        } else {
            const int b = static_cast<int>(tid / n_bus);
            const int r = static_cast<int>(tid - static_cast<long long>(b) * n_bus);
            batch_outer[tid] = outer[r] + b * nnz;
        }
    }
}

// Declared in strategies/cudss_batch_solver.cuh: the same skeleton, for the
// Jacobian of the experimental block-diagonal cuDSS mode.
void cudss_blockdiag_csr(int n, int nnz, int batch,
                         const int* outer, const int* inner,
                         int* batch_outer, int* batch_inner,
                         cudaStream_t cs)
{
    const long long n_outer = static_cast<long long>(batch) * n + 1;
    const long long n_inner = static_cast<long long>(batch) * nnz;
    blockdiag_csr_kernel<<<nr_grid_size(std::max(n_outer, n_inner), BS), BS, 0, cs>>>(
        n, nnz, batch, outer, inner, batch_outer, batch_inner);
    CHK_CUDA_BPF(cudaGetLastError());
}

// =============================================================================
// Constructor
// =============================================================================
template <typename BatchSource>
BatchPfDriver<BatchSource>::BatchPfDriver(
    AcPfNrState&          base_state,
    BatchSource           source,
    int                   n_contingencies_in,
    int                   batch_size,
    int                   nb_iter,
    ContingencySolverType strategy_type,
    int                   refactor_period,
    ReorderingAlg         reordering_alg,
    MatchingAlg           matching_alg,
    PivotEpsilonAlg       pivot_epsilon_alg,
    bool                  scaling_max_voltage_change,
    double                max_dVa,
    double                max_dVm)
    : base(base_state)
    , source_(std::move(source))
    , n_contingencies(n_contingencies_in)
    , n_active_(source_.n_active())
    , batch_size_(batch_size)
    , nb_iter_(nb_iter)
      // Chunks are formed over the ACTIVE set: disconnected contingencies are
      // compacted out by the source and never occupy a batch slot.
    , n_chunks_((source_.n_active() + batch_size - 1) / batch_size)
    , scaling_max_voltage_change_(scaling_max_voltage_change)
    , max_dVa_(static_cast<cuda_real_type>(max_dVa))
    , max_dVm_(static_cast<cuda_real_type>(max_dVm))
    , reordering_alg_(reordering_alg)
    , matching_alg_(matching_alg)
    , pivot_epsilon_alg_(pivot_epsilon_alg)
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

    // -------------------------------------------------------------------------
    // Reject batch_size * per-system-stride combinations that hit a hard
    // 32-bit index limit still standing after the ptrdiff_t widening of
    // gpusim2grid's own kernels/launch arithmetic (see the "tid/b widened to
    // ptrdiff_t" comments in acpf_nr_kernels.cu, nr_iter_step.cuh, driver.cuh):
    //   - n_bus, nnz_Y: blockdiag_csr_kernel (below) writes ONE literal
    //     block-diagonal Ybus matrix and BatchPfDriver hands its outer/inner
    //     arrays to cusparseCreateConstCsr with CUSPARSE_INDEX_32I explicitly
    //     -- a hard cuSPARSE requirement gpusim2grid's own arithmetic cannot
    //     lift (see batch_dims_check.hpp's own doc).
    //   - nnz_J, dim_J: our own fill/update/feature kernels are now 64-bit
    //     safe at any batch_size, but cuDSS's own internal indexing into the
    //     batched d_J_values_batch/d_F_batch/d_dx_batch buffers (via
    //     CUDSS_CONFIG_UBATCH_SIZE) is closed-source and unverified at these
    //     scales -- kept as a safety margin, not a known corruption vector.
    // n_contingencies-scaled products (copy_results_to_host, branch-flow and
    // violation output buffers) no longer need this guard: those loops and
    // kernels were widened to 64-bit alongside the rest of this pass.
    // -------------------------------------------------------------------------
    check_batch_stride("BatchPfDriver", "batch_size", batch_size_, "nnz_J", nnz_J);
    check_batch_stride("BatchPfDriver", "batch_size", batch_size_, "dim_J", dim_J);
    check_batch_stride("BatchPfDriver", "batch_size", batch_size_, "n_bus", n_bus);
    check_batch_stride("BatchPfDriver", "batch_size", batch_size_, "nnz_Y", nnz_Y);

    // Source-owned host preprocessing time was captured in the source ctor.
    // Nothing else on the host: the block-diagonal CSR structure below is
    // generated on the device.
    t_preprocess_ms_ = source_.cpu_preprocess_ms();

    // -------------------------------------------------------------------------
    // Block-diagonal CSR structure (outer/inner only; values are tiled per
    // chunk by ContingencyBatch/ScenarioSweepBatch or once at construction by
    // InjectionBatch) + chunk-sized working buffers, all under the t_alloc_ms_
    // wall-clock window.
    // -------------------------------------------------------------------------
    {
        auto t_alloc_start = std::chrono::steady_clock::now();

        const long long n_batch_outer = static_cast<long long>(batch_size_) * n_bus + 1;
        const long long n_batch_inner = static_cast<long long>(batch_size_) * nnz_Y;
        d_Ybus_batch_outer.resize(static_cast<size_t>(n_batch_outer));
        d_Ybus_batch_inner.resize(static_cast<size_t>(n_batch_inner));
        blockdiag_csr_kernel<<<nr_grid_size(std::max(n_batch_outer, n_batch_inner), BS), BS, 0, cs>>>(
            n_bus, nnz_Y, batch_size_,
            thrust::raw_pointer_cast(base.d_Ybus_outer.data()),
            thrust::raw_pointer_cast(base.d_Ybus_inner.data()),
            thrust::raw_pointer_cast(d_Ybus_batch_outer.data()),
            thrust::raw_pointer_cast(d_Ybus_batch_inner.data()));
        CHK_CUDA_BPF(cudaGetLastError());

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

        // NR step-scaling scratch (one max|dtheta|/max|dvm| pair per slot).
        if (scaling_max_voltage_change_) {
            d_scale_max_dtheta_batch.resize(batch_size_);
            d_scale_max_dvm_batch.resize(batch_size_);
        }

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
    // cuDSS uniform-batch context + ANALYSIS + policy init
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
        cs,
        reordering_alg,
        matching_alg,
        pivot_epsilon_alg);

    std::visit([&](auto& policy) {
        policy.initialize_from_base(
            linear_solver_, base,
            thrust::raw_pointer_cast(d_J_values_batch.data()),
            batch_size_, nnz_J, cs);
    }, policy_);

    cs.synchronize();
    // Split the cuDSS context creation (first-touch dlopen/JIT on the first
    // cuDSS use in the process) out of the ANALYSIS cost -- see
    // CudssBatchSolver::context_init_ms() and BatchTimings::t_context_init_ms.
    t_context_init_ms_ = linear_solver_.context_init_ms();
    t_analysis_ms_ = bpf_ms_since(t_cudss_start) - t_context_init_ms_;

    // -------------------------------------------------------------------------
    // Source-specific one-time setup (flat-patch/mask H→D upload for
    // ContingencyBatch; full Sbus_all H→D upload + one-time Ybus D→D tiling
    // for InjectionBatch). Timed separately from cuDSS ANALYSIS above so
    // t_analysis_ms_ isn't a mix of unrelated GPU compute + transfer.
    // -------------------------------------------------------------------------
    auto t_source_start = std::chrono::steady_clock::now();
    {
        BatchPfDriverContext ctx = make_context();
        source_.initialize(ctx, cs);
    }
    cs.synchronize();
    t_source_init_ms_ = bpf_ms_since(t_source_start);
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

    // Pure host loop (no cuSPARSE/cuDSS 32-bit index constraint involved) --
    // widened to a 64-bit count so n_contingencies * n_bus itself, computed
    // once here, can't silently wrap before V_out.resize()/the loop bound
    // even sees it.
    const long long n_v = static_cast<long long>(n_contingencies) * n_bus;
    thrust::host_vector<cudaComplexType> h_V = d_V_results;
    V_out.resize(n_v);
    for (long long i = 0; i < n_v; ++i)
        V_out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[i].x),
            static_cast<eigen_real_type>(h_V[i].y));

    thrust::host_vector<cuda_real_type> h_res = d_residuals;
    res_out.resize(n_contingencies);
    for (int i = 0; i < n_contingencies; ++i)
        res_out(i) = static_cast<eigen_real_type>(h_res[i]);
}

// =============================================================================
// upload_branch_admittances  /  set_branch_data  /  copy_flow_results_to_host
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::upload_branch_admittances(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff_eff,
    Eigen::Ref<const CplxVect>        yft_eff,
    Eigen::Ref<const CplxVect>        ytf_eff,
    Eigen::Ref<const CplxVect>        ytt_eff,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double                            sn_mva)
{
    n_branches_ = static_cast<int>(branch_from.size());

    // No check_batch_stride guard needed here: compute_branch_flows_kernel's
    // and check_limit_violations_kernel's n_branches_-scaled offsets, and
    // copy_flow_results_to_host's host loop, were all widened to 64-bit
    // alongside the rest of this pass -- neither cuSPARSE nor cuDSS is
    // involved on this path (see the constructor's own check_batch_stride
    // calls for the ones that still are).

    auto t_upload_start = std::chrono::steady_clock::now();
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
            // branch_from can be -1 (Kron-reduced / half-open-line endpoint,
            // see check_limit_violations_kernel's own bf/bt>=0 guard) --
            // bus_vn_kv(-1) is an out-of-bounds read (UB, not caught by
            // Eigen without assertions), and if it happens to read something
            // near zero it makes h_base[l] = +inf, which then poisons the
            // *live* side's reported current (ka_or/ka_ex = finite * inf)
            // into an "infinite" CURRENT violation. Fall back to the other
            // endpoint, which is always valid when branch_from isn't (a
            // branch can't have both ends Kron-reduced and still appear
            // here).
            const int vn_bus = (h_from[l] >= 0) ? h_from[l] : h_to[l];
            const double vn = (vn_bus >= 0) ? bus_vn_kv(vn_bus) : 0.0;
            h_base[l] = (vn_bus >= 0)
                ? static_cast<cuda_real_type>(sn_mva * 1e6 / (sqrt3 * vn * 1e3))
                : cuda_real_type(0);
        }
        upload_h2d(d_base_current_A, h_base.data(), n_branches_, cs);
    }

    {
        std::vector<cudaComplexType> h_yff_eff(n_branches_), h_yft_eff(n_branches_),
                                     h_ytf_eff(n_branches_), h_ytt_eff(n_branches_);
        for (int l = 0; l < n_branches_; ++l) {
            h_yff_eff[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(yff_eff(l).real()),
                static_cast<cuda_real_type>(yff_eff(l).imag()));
            h_yft_eff[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(yft_eff(l).real()),
                static_cast<cuda_real_type>(yft_eff(l).imag()));
            h_ytf_eff[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(ytf_eff(l).real()),
                static_cast<cuda_real_type>(ytf_eff(l).imag()));
            h_ytt_eff[l] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(ytt_eff(l).real()),
                static_cast<cuda_real_type>(ytt_eff(l).imag()));
        }
        upload_h2d(d_yff_eff, h_yff_eff.data(), n_branches_, cs);
        upload_h2d(d_yft_eff, h_yft_eff.data(), n_branches_, cs);
        upload_h2d(d_ytf_eff, h_ytf_eff.data(), n_branches_, cs);
        upload_h2d(d_ytt_eff, h_ytt_eff.data(), n_branches_, cs);
    }

    // Full per-bus nominal kV (distinct from the per-branch-endpoint use above
    // folded into d_base_current_A): needed by the fused compute_limit_violations
    // kernel's bus-voltage check, which visits every bus, not just branch ends.
    {
        const int n_bus = base.n_bus;
        std::vector<cuda_real_type> h_bus_vn(n_bus);
        for (int b = 0; b < n_bus; ++b)
            h_bus_vn[b] = static_cast<cuda_real_type>(bus_vn_kv(b));
        upload_h2d(d_bus_vn_kv, h_bus_vn.data(), n_bus, cs);
    }

    cs.synchronize();
    t_branch_data_upload_ms_ = bpf_ms_since(t_upload_start);
    _has_branch_admittances = true;
}

template <typename BatchSource>
void BatchPfDriver<BatchSource>::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff_eff,
    Eigen::Ref<const CplxVect>        yft_eff,
    Eigen::Ref<const CplxVect>        ytf_eff,
    Eigen::Ref<const CplxVect>        ytt_eff,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double                            sn_mva)
{
    upload_branch_admittances(branch_from, branch_to, yff_eff, yft_eff, ytf_eff, ytt_eff, bus_vn_kv, sn_mva);

    d_or_amps_results.assign(
        static_cast<size_t>(n_contingencies) * n_branches_, cuda_real_type(0));
    d_ex_amps_results.assign(
        static_cast<size_t>(n_contingencies) * n_branches_, cuda_real_type(0));

    _has_branch_data = true;
}

// =============================================================================
// set_violation_limits  (compute_limit_violations)
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::set_violation_limits(
    Eigen::Ref<const RealVect> bus_vmin_kv,
    Eigen::Ref<const RealVect> bus_vmax_kv,
    Eigen::Ref<const RealVect> branch_limit_a1_ka,
    Eigen::Ref<const RealVect> branch_limit_a2_ka,
    double tol,
    int    K,
    int    n_lines)
{
    if (!_has_branch_admittances)
        throw std::runtime_error(
            "BatchPfDriver::set_violation_limits: call upload_branch_admittances "
            "(via ContingencyAnalysisSession::set_branch_data) before enabling "
            "compute_limit_violations.");

    // No check_batch_stride guard needed here: check_limit_violations_kernel's
    // out_c * K output-slice offset was widened to 64-bit (ptrdiff_t)
    // alongside the rest of this pass, and the n_contingencies*K output
    // buffers below are already sized via a size_t-safe assign().

    auto t_setup_start = std::chrono::steady_clock::now();

    const int n_bus = base.n_bus;
    auto to_dev = [&](thrust::device_vector<cuda_real_type>& d,
                       Eigen::Ref<const RealVect> h, int n) {
        std::vector<cuda_real_type> tmp(n);
        for (int i = 0; i < n; ++i) tmp[i] = static_cast<cuda_real_type>(h(i));
        upload_h2d(d, tmp.data(), n, cs);
    };
    to_dev(d_bus_vmin_kv, bus_vmin_kv, n_bus);
    to_dev(d_bus_vmax_kv, bus_vmax_kv, n_bus);
    to_dev(d_branch_limit_a1_ka, branch_limit_a1_ka, n_branches_);
    to_dev(d_branch_limit_a2_ka, branch_limit_a2_ka, n_branches_);

    violation_tol_      = static_cast<cuda_real_type>(tol);
    violation_capacity_ = K;
    n_lines_            = n_lines;

    const size_t n_out = static_cast<size_t>(n_contingencies) * static_cast<size_t>(K);
    d_viol_element_type.assign(n_out, 0);
    d_viol_element_id.assign(n_out, 0);
    d_viol_side.assign(n_out, 0);
    d_viol_type.assign(n_out, 0);
    d_viol_value.assign(n_out, cuda_real_type(0));
    d_viol_limit.assign(n_out, cuda_real_type(0));
    // -1 sentinel: "not yet simulated" (overwritten by every active slot's own
    // chunk write; slots never revisited by _solve_chunk -- e.g. disconnected/
    // masked-skip contingencies excluded from the active set -- keep -1).
    d_violation_count.assign(static_cast<size_t>(n_contingencies), -1);
    d_violation_truncated.assign(static_cast<size_t>(n_contingencies), 0);
    d_violation_count_low_voltage.assign(static_cast<size_t>(n_contingencies), -1);
    d_violation_count_high_voltage.assign(static_cast<size_t>(n_contingencies), -1);
    d_violation_count_current.assign(static_cast<size_t>(n_contingencies), -1);

    cs.synchronize();
    t_violation_setup_ms_ = bpf_ms_since(t_setup_start);
    _fused_violations_enabled = true;
}

// =============================================================================
// set_bus_q_check / run_bus_q_check_n  (compute_physical_violations)
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::set_bus_q_check(
    const BusQPlanData& plan, double tol_mvar, int K_q, double residual_tol,
    const unsigned char* d_gen_off, int n_gen)
{
    plan.validate(base.n_bus);
    if (K_q <= 0)
        throw std::runtime_error("BatchPfDriver::set_bus_q_check: K_q (physical_violation_capacity) must be > 0.");
    if (!(tol_mvar >= 0.) || !std::isfinite(tol_mvar))
        throw std::runtime_error("BatchPfDriver::set_bus_q_check: tol_mvar must be a finite, non-negative number.");
    if (d_gen_off != nullptr && n_gen > 0) {
        for (int p = 0; p < plan.n_gen_entries(); ++p)
            if (plan.gen_id(p) >= n_gen)
                throw std::runtime_error(
                    "BatchPfDriver::set_bus_q_check: a generator id of the plan is outside the "
                    "generator-contingency mask's columns.");
    }

    auto t_setup_start = std::chrono::steady_clock::now();

    auto to_dev_r = [&](thrust::device_vector<cuda_real_type>& d, const RealVect& h) {
        std::vector<cuda_real_type> tmp(static_cast<size_t>(h.size()));
        for (Eigen::Index i = 0; i < h.size(); ++i) tmp[static_cast<size_t>(i)] = static_cast<cuda_real_type>(h(i));
        upload_h2d(d, tmp.data(), static_cast<int>(h.size()), cs);
    };
    auto to_dev_i = [&](thrust::device_vector<int>& d, const Eigen::VectorXi& h) {
        upload_h2d(d, h.data(), static_cast<int>(h.size()), cs);
    };
    to_dev_i(d_bq_bus_solver, plan.bus_solver);
    to_dev_i(d_bq_n_fixed,    plan.n_fixed);
    to_dev_i(d_bq_gen_start,  plan.gen_start);
    to_dev_i(d_bq_gen_id,     plan.gen_id);
    to_dev_r(d_bq_qmin_fixed, plan.qmin_fixed_mvar);
    to_dev_r(d_bq_qmax_fixed, plan.qmax_fixed_mvar);
    to_dev_r(d_bq_bmin_sum,   plan.bmin_sum_pu);
    to_dev_r(d_bq_bmax_sum,   plan.bmax_sum_pu);
    to_dev_r(d_bq_gen_qmin,   plan.gen_qmin_mvar);
    to_dev_r(d_bq_gen_qmax,   plan.gen_qmax_mvar);

    bus_q_n_check_      = plan.n_check;
    bus_q_capacity_     = K_q;
    bus_q_n_gen_        = (d_gen_off != nullptr) ? n_gen : 0;
    d_bq_gen_off_       = (n_gen > 0) ? d_gen_off : nullptr;
    bus_q_tol_mvar_     = static_cast<cuda_real_type>(tol_mvar);
    bus_q_sn_mva_       = static_cast<cuda_real_type>(plan.sn_mva);
    bus_q_residual_tol_ = static_cast<cuda_real_type>(residual_tol);

    const size_t n_out = static_cast<size_t>(n_contingencies) * static_cast<size_t>(K_q);
    d_bq_out_bus_id.assign(n_out, 0);
    d_bq_out_type.assign(n_out, 0);
    d_bq_out_value.assign(n_out, cuda_real_type(0));
    d_bq_out_limit.assign(n_out, cuda_real_type(0));
    // -1 sentinel: "never simulated" (a slot compacted out of the active set);
    // every simulated row overwrites it with 0..K_q.
    d_bq_count.assign(static_cast<size_t>(n_contingencies), -1);
    d_bq_truncated.assign(static_cast<size_t>(n_contingencies), 0);
    d_bq_n_bus_id.assign(static_cast<size_t>(K_q), 0);
    d_bq_n_type.assign(static_cast<size_t>(K_q), 0);
    d_bq_n_value.assign(static_cast<size_t>(K_q), cuda_real_type(0));
    d_bq_n_limit.assign(static_cast<size_t>(K_q), cuda_real_type(0));
    d_bq_n_count.assign(1, 0);
    d_bq_n_truncated.assign(1, 0);

    cs.synchronize();
    t_bus_q_setup_ms_ = bpf_ms_since(t_setup_start);
    _bus_q_enabled = true;
}

template <typename BatchSource>
void BatchPfDriver<BatchSource>::run_bus_q_check_n()
{
    if (!_bus_q_enabled)
        throw std::runtime_error("BatchPfDriver::run_bus_q_check_n: call set_bus_q_check first.");
    auto t_start = std::chrono::steady_clock::now();
    check_bus_q_violations_kernel<<<1, BS, 0, cs>>>(
        thrust::raw_pointer_cast(base.d_V_base.data()),
        thrust::raw_pointer_cast(base.d_Ybus_values.data()),
        thrust::raw_pointer_cast(base.d_Ybus_outer.data()),
        thrust::raw_pointer_cast(base.d_Ybus_inner.data()),
        thrust::raw_pointer_cast(base.d_Sbus.data()), /*sbus_stride=*/0,
        /*d_residuals=*/nullptr, bus_q_residual_tol_,
        bus_q_n_check_,
        thrust::raw_pointer_cast(d_bq_bus_solver.data()),
        thrust::raw_pointer_cast(d_bq_qmin_fixed.data()),
        thrust::raw_pointer_cast(d_bq_qmax_fixed.data()),
        thrust::raw_pointer_cast(d_bq_n_fixed.data()),
        thrust::raw_pointer_cast(d_bq_bmin_sum.data()),
        thrust::raw_pointer_cast(d_bq_bmax_sum.data()),
        thrust::raw_pointer_cast(d_bq_gen_start.data()),
        thrust::raw_pointer_cast(d_bq_gen_id.data()),
        thrust::raw_pointer_cast(d_bq_gen_qmin.data()),
        thrust::raw_pointer_cast(d_bq_gen_qmax.data()),
        /*d_gen_off=*/nullptr, /*n_gen=*/0,
        bus_q_sn_mva_, bus_q_tol_mvar_,
        base.n_bus, base.nnz_Y,
        /*c_start=*/0, /*actual_batch=*/1, bus_q_capacity_,
        /*d_result_map=*/nullptr,
        thrust::raw_pointer_cast(d_bq_n_bus_id.data()),
        thrust::raw_pointer_cast(d_bq_n_type.data()),
        thrust::raw_pointer_cast(d_bq_n_value.data()),
        thrust::raw_pointer_cast(d_bq_n_limit.data()),
        thrust::raw_pointer_cast(d_bq_n_count.data()),
        thrust::raw_pointer_cast(d_bq_n_truncated.data()));
    CHK_CUDA_BPF(cudaGetLastError());
    cs.synchronize();
    t_bus_q_setup_ms_ += bpf_ms_since(t_start);
}

// =============================================================================
// set_hvdc_p_check / run_hvdc_p_check_n  (compute_physical_violations)
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::set_hvdc_p_check(double tol_mw, double sn_mva, int K_p, double residual_tol)
{
    if (K_p <= 0)
        throw std::runtime_error("BatchPfDriver::set_hvdc_p_check: K_p (physical_violation_capacity) must be > 0.");
    if (!(tol_mw >= 0.) || !std::isfinite(tol_mw))
        throw std::runtime_error("BatchPfDriver::set_hvdc_p_check: tol_mw must be a finite, non-negative number.");
    if (!(sn_mva > 0.))
        throw std::runtime_error("BatchPfDriver::set_hvdc_p_check: sn_mva must be > 0.");

    auto t_setup_start = std::chrono::steady_clock::now();
    hvdc_p_capacity_     = K_p;
    hvdc_p_tol_pu_       = static_cast<cuda_real_type>(tol_mw / sn_mva);
    hvdc_p_sn_mva_       = static_cast<cuda_real_type>(sn_mva);
    hvdc_p_residual_tol_ = static_cast<cuda_real_type>(residual_tol);

    const size_t n_out = static_cast<size_t>(n_contingencies) * static_cast<size_t>(K_p);
    d_hp_out_hvdc_id.assign(n_out, 0);
    d_hp_out_side.assign(n_out, 0);
    d_hp_out_value.assign(n_out, cuda_real_type(0));
    d_hp_out_limit.assign(n_out, cuda_real_type(0));
    d_hp_count.assign(static_cast<size_t>(n_contingencies), -1);   // see set_bus_q_check
    d_hp_truncated.assign(static_cast<size_t>(n_contingencies), 0);
    d_hp_n_hvdc_id.assign(static_cast<size_t>(K_p), 0);
    d_hp_n_side.assign(static_cast<size_t>(K_p), 0);
    d_hp_n_value.assign(static_cast<size_t>(K_p), cuda_real_type(0));
    d_hp_n_limit.assign(static_cast<size_t>(K_p), cuda_real_type(0));
    d_hp_n_count.assign(1, 0);
    d_hp_n_truncated.assign(1, 0);

    cs.synchronize();
    t_hvdc_p_setup_ms_ = bpf_ms_since(t_setup_start);
    _hvdc_p_enabled = true;
}

template <typename BatchSource>
void BatchPfDriver<BatchSource>::run_hvdc_p_check_n()
{
    if (!_hvdc_p_enabled)
        throw std::runtime_error("BatchPfDriver::run_hvdc_p_check_n: call set_hvdc_p_check first.");
    auto t_start = std::chrono::steady_clock::now();
    check_hvdc_p_violations_kernel<<<1, BS, 0, cs>>>(
        thrust::raw_pointer_cast(base.d_V_base.data()),
        /*d_residuals=*/nullptr, hvdc_p_residual_tol_,
        base.n_hvdc,
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
        thrust::raw_pointer_cast(base.d_hvdc_id.data()),
        hvdc_p_sn_mva_, hvdc_p_tol_pu_,
        base.n_bus,
        /*c_start=*/0, /*actual_batch=*/1, hvdc_p_capacity_,
        /*d_result_map=*/nullptr,
        thrust::raw_pointer_cast(d_hp_n_hvdc_id.data()),
        thrust::raw_pointer_cast(d_hp_n_side.data()),
        thrust::raw_pointer_cast(d_hp_n_value.data()),
        thrust::raw_pointer_cast(d_hp_n_limit.data()),
        thrust::raw_pointer_cast(d_hp_n_count.data()),
        thrust::raw_pointer_cast(d_hp_n_truncated.data()));
    CHK_CUDA_BPF(cudaGetLastError());
    cs.synchronize();
    t_hvdc_p_setup_ms_ += bpf_ms_since(t_start);
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
    // Pure host loop -- widened to a 64-bit count for the same reason as
    // copy_results_to_host's n_v above.
    const long long n = static_cast<long long>(n_contingencies) * n_branches_;
    thrust::host_vector<cuda_real_type> h_or = d_or_amps_results;
    thrust::host_vector<cuda_real_type> h_ex = d_ex_amps_results;

    or_amps_out.resize(n);
    ex_amps_out.resize(n);
    for (long long i = 0; i < n; ++i) {
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
    ++n_solves_;
    if (keep_final_jacobian_ && n_chunks_ > 1)
        throw std::runtime_error(
            "[batch_pf] keep_final_jacobian requires the whole batch to be "
            "solved in ONE chunk (only the last chunk's Jacobian survives in "
            "the chunk buffer): raise batch_size to at least the number of "
            "active elements (" + std::to_string(n_active_) + ").");
    t.n_contingencies  = n_contingencies;
    t.n_chunks         = n_chunks_;
    t.chunk_size       = batch_size_;
    t.nb_iter          = nb_iter_;
    t.t_preprocess_ms  = t_preprocess_ms_;
    t.t_alloc_ms       = t_alloc_ms_;
    t.t_analysis_ms    = t_analysis_ms_;
    t.t_source_init_ms = t_source_init_ms_;
    t.t_context_init_ms = t_context_init_ms_;

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
    // Per-chunk ANALYSIS of the non-uniform cuDSS modes (0 in uniform mode).
    const double t_chunk_analysis_start = linear_solver_.analysis_ms();
    for (int chunk = 0; chunk < n_chunks_; ++chunk) {
        const int c_start      = chunk * batch_size_;
        const int actual_batch = std::min(batch_size_, n_active_ - c_start);
        _solve_chunk(c_start, actual_batch, t);
    }
    t.t_analysis_ms += linear_solver_.analysis_ms() - t_chunk_analysis_start;

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

    NrIterBuffers buf {
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

    // NR step-scaling (MaxVoltageChange) -- off (nullptr scratch) unless
    // enabled, matching every other opt-in extension above.
    buf.scaling_max_voltage_change = scaling_max_voltage_change_;
    buf.max_dVa = max_dVa_;
    buf.max_dVm = max_dVm_;
    if (scaling_max_voltage_change_) {
        buf.d_scale_max_dtheta = thrust::raw_pointer_cast(d_scale_max_dtheta_batch.data());
        buf.d_scale_max_dvm    = thrust::raw_pointer_cast(d_scale_max_dvm_batch.data());
    }

    // handle_disconnected_grid: attach this chunk's mask slice (no-op for the
    // injection sweep / when the mode is off). d_J_outer is the shared skeleton.
    source_.fill_mask_buffers(buf, chunk, thrust::raw_pointer_cast(base.d_J_outer.data()));

    // Per-slot distributed-slack weights (ScenarioSweep generator
    // contingencies only; no-op elsewhere -- buf.d_slack_w stays base's
    // shared array with slack_w_stride 0).
    source_.fill_slack_w_buffers(buf, chunk);

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
                nr_grid_size((long long)actual_batch * base.n_p, BS), BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_F_batch.data()),
                thrust::raw_pointer_cast(d_V_batch.data()),
                thrust::raw_pointer_cast(d_Ibus_batch.data()),
                d_Sbus_for_NR,
                thrust::raw_pointer_cast(base.d_p_buses.data()),
                thrust::raw_pointer_cast(base.d_p_rows.data()),
                base.n_p, n_bus, dim_J, actual_batch, sbus_stride);
            fill_FQ_kernel<<<
                nr_grid_size((long long)actual_batch * base.n_q, BS), BS, 0, cs>>>(
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
            // handle_disconnected_grid / PV pins / stranded controllers: the
            // same row rewrites the NR loop applied, so the frozen component
            // and the pinned rows do not pollute the ‖F‖∞ residual.
            nr_apply_F_masks(buf, nnz_J, dim_J, actual_batch, cs);

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
    // ④b Converged Jacobian (opt-in, differentiable wrapper): the loop left
    //     J(V_{nb_iter-1}) in d_J_values_batch; refill it at V_final using the
    //     SpMV output of step ④ (d_Ibus_batch = Ybus · V_final for the whole
    //     padded batch). Same fill sequence as the loop -- one definition
    //     (nr_fill_J_at_current_V). Must run BEFORE step ⑤'s NaN masking of
    //     d_V_batch. The forward factors are untouched (cuDSS keeps them in
    //     its own data object); the batched adjoint reads the values later.
    // -------------------------------------------------------------------------
    if (keep_final_jacobian_) {
        timer.start();
        nr_fill_J_at_current_V(buf, n_bus, dim_J, nnz_Y, nnz_J, batch_size_, cs);
        t.t_fill_J += timer.stop_ms();
    }

    // -------------------------------------------------------------------------
    // ⑤  Store V results.  Without compaction the active slots map contiguously
    //     to result indices (single D→D memcpy).  With compaction the original
    //     indices are non-contiguous, so scatter through d_result_map instead.
    // -------------------------------------------------------------------------
    timer.start();
    if (actual_batch > 0) {
        // handle_disconnected_grid: report the frozen (masked) buses' voltages as
        // NaN before the result store (no-op when the mode is off).
        nr_mask_v_nan(buf, n_bus, cs);
    }
    t.t_store_V += timer.stop_ms();

    // compute_limit_violations: fused per-contingency voltage/current/
    // divergence check, reading d_V_batch (chunk-local, post mask-NaN)
    // directly -- never touches d_V_results or any O(actual_batch*n_branches)
    // buffer. d_residuals already holds this chunk's just-written residuals
    // (step ④ above). No-op (skipped entirely) unless set_violation_limits()
    // was called -- timed separately into t_violation_check so it isn't
    // folded into t_store_V's "D→D copy" cost.
    if (actual_batch > 0 && _fused_violations_enabled) {
        timer.start();
        const TrippedBranchTable trip = source_.tripped_branch_table();
        check_limit_violations_kernel<<<(actual_batch + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_V_batch.data()),
            thrust::raw_pointer_cast(d_residuals.data()),
            violation_tol_,
            thrust::raw_pointer_cast(d_bus_vn_kv.data()),
            thrust::raw_pointer_cast(d_bus_vmin_kv.data()),
            thrust::raw_pointer_cast(d_bus_vmax_kv.data()),
            thrust::raw_pointer_cast(d_branch_from.data()),
            thrust::raw_pointer_cast(d_branch_to.data()),
            thrust::raw_pointer_cast(d_yff_eff.data()),
            thrust::raw_pointer_cast(d_yft_eff.data()),
            thrust::raw_pointer_cast(d_ytf_eff.data()),
            thrust::raw_pointer_cast(d_ytt_eff.data()),
            thrust::raw_pointer_cast(d_base_current_A.data()),
            thrust::raw_pointer_cast(d_branch_limit_a1_ka.data()),
            thrust::raw_pointer_cast(d_branch_limit_a2_ka.data()),
            trip.d_start, trip.d_count, trip.d_branch_flat,
            n_bus, n_branches_, n_lines_,
            c_start, actual_batch, violation_capacity_,
            d_result_map,
            thrust::raw_pointer_cast(d_viol_element_type.data()),
            thrust::raw_pointer_cast(d_viol_element_id.data()),
            thrust::raw_pointer_cast(d_viol_side.data()),
            thrust::raw_pointer_cast(d_viol_type.data()),
            thrust::raw_pointer_cast(d_viol_value.data()),
            thrust::raw_pointer_cast(d_viol_limit.data()),
            thrust::raw_pointer_cast(d_violation_count.data()),
            thrust::raw_pointer_cast(d_violation_truncated.data()),
            thrust::raw_pointer_cast(d_violation_count_low_voltage.data()),
            thrust::raw_pointer_cast(d_violation_count_high_voltage.data()),
            thrust::raw_pointer_cast(d_violation_count_current.data()));
        CHK_CUDA_BPF(cudaGetLastError());
        t.t_violation_check += timer.stop_ms();
    }

    // compute_physical_violations: the per-bus reactive-capability check on the
    // converged, post mask-NaN chunk voltages, this chunk's patched Ybus values
    // and its Sbus (shared or per slot, see sbus_stride). Launched whenever the
    // check is on -- even with an empty plan -- so every simulated row gets a
    // count (0), distinct from the -1 "never simulated" sentinel.
    if (actual_batch > 0 && _bus_q_enabled) {
        timer.start();
        check_bus_q_violations_kernel<<<(actual_batch + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_V_batch.data()),
            thrust::raw_pointer_cast(d_Ybus_values_batch.data()),
            thrust::raw_pointer_cast(base.d_Ybus_outer.data()),
            thrust::raw_pointer_cast(base.d_Ybus_inner.data()),
            d_Sbus_for_NR, sbus_stride,
            thrust::raw_pointer_cast(d_residuals.data()), bus_q_residual_tol_,
            bus_q_n_check_,
            thrust::raw_pointer_cast(d_bq_bus_solver.data()),
            thrust::raw_pointer_cast(d_bq_qmin_fixed.data()),
            thrust::raw_pointer_cast(d_bq_qmax_fixed.data()),
            thrust::raw_pointer_cast(d_bq_n_fixed.data()),
            thrust::raw_pointer_cast(d_bq_bmin_sum.data()),
            thrust::raw_pointer_cast(d_bq_bmax_sum.data()),
            thrust::raw_pointer_cast(d_bq_gen_start.data()),
            thrust::raw_pointer_cast(d_bq_gen_id.data()),
            thrust::raw_pointer_cast(d_bq_gen_qmin.data()),
            thrust::raw_pointer_cast(d_bq_gen_qmax.data()),
            d_bq_gen_off_, bus_q_n_gen_,
            bus_q_sn_mva_, bus_q_tol_mvar_,
            n_bus, nnz_Y,
            c_start, actual_batch, bus_q_capacity_,
            d_result_map,
            thrust::raw_pointer_cast(d_bq_out_bus_id.data()),
            thrust::raw_pointer_cast(d_bq_out_type.data()),
            thrust::raw_pointer_cast(d_bq_out_value.data()),
            thrust::raw_pointer_cast(d_bq_out_limit.data()),
            thrust::raw_pointer_cast(d_bq_count.data()),
            thrust::raw_pointer_cast(d_bq_truncated.data()));
        CHK_CUDA_BPF(cudaGetLastError());
        t.t_bus_q_check += timer.stop_ms();
    }

    // compute_physical_violations: droop P-saturation check on the same voltages
    // (the per-line data is the base state's own, shared across slots).
    if (actual_batch > 0 && _hvdc_p_enabled) {
        timer.start();
        check_hvdc_p_violations_kernel<<<(actual_batch + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_V_batch.data()),
            thrust::raw_pointer_cast(d_residuals.data()), hvdc_p_residual_tol_,
            base.n_hvdc,
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
            thrust::raw_pointer_cast(base.d_hvdc_id.data()),
            hvdc_p_sn_mva_, hvdc_p_tol_pu_,
            n_bus,
            c_start, actual_batch, hvdc_p_capacity_,
            d_result_map,
            thrust::raw_pointer_cast(d_hp_out_hvdc_id.data()),
            thrust::raw_pointer_cast(d_hp_out_side.data()),
            thrust::raw_pointer_cast(d_hp_out_value.data()),
            thrust::raw_pointer_cast(d_hp_out_limit.data()),
            thrust::raw_pointer_cast(d_hp_count.data()),
            thrust::raw_pointer_cast(d_hp_truncated.data()));
        CHK_CUDA_BPF(cudaGetLastError());
        t.t_hvdc_p_check += timer.stop_ms();
    }

    timer.start();
    if (actual_batch > 0) {
        if (d_result_map) {
            scatter_V_results_kernel<<<nr_grid_size((long long)actual_batch * n_bus, BS), BS, 0, cs>>>(
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
        compute_branch_flows_kernel<<<nr_grid_size((long long)actual_batch * n_branches_, BS), BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_V_batch.data()),
            thrust::raw_pointer_cast(d_branch_from.data()),
            thrust::raw_pointer_cast(d_branch_to.data()),
            thrust::raw_pointer_cast(d_yff_eff.data()),
            thrust::raw_pointer_cast(d_yft_eff.data()),
            thrust::raw_pointer_cast(d_ytf_eff.data()),
            thrust::raw_pointer_cast(d_ytt_eff.data()),
            thrust::raw_pointer_cast(d_base_current_A.data()),
            thrust::raw_pointer_cast(d_or_amps_results.data()),
            thrust::raw_pointer_cast(d_ex_amps_results.data()),
            n_bus, n_branches_, c_start, actual_batch, d_result_map);
        CHK_CUDA_BPF(cudaGetLastError());
        t.t_flow_computation += timer.stop_ms();
    }
}

// =============================================================================
// Batched adjoint kernels
// =============================================================================

// JT_values[b*nnz + perm[i]] = J_values[b*nnz + i] for every slot b: the
// value permutation of the shared J→Jᵀ position map (BatchAdjoint::d_J_to_JT).
__global__ void transpose_gather_values_kernel(
          cuda_real_type* __restrict__ dst,
    const cuda_real_type* __restrict__ src,
    const int*            __restrict__ perm,
    int nnz, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / nnz;
    const int       i   = static_cast<int>(tid % nnz);
    if (b >= batch) return;
    dst[b * nnz + perm[i]] = src[b * nnz + i];
}

// Zero the adjoint vector at this chunk's identity rows (handle_disconnected_
// grid masked P/Q rows and the PV-pinned Q rows of generator contingencies).
// An identity row r = e_c decouples: λ_r appears in the single Jᵀ equation of
// its column c, so it absorbs x̄_c − Σ_{r'≠r} J[r',c] λ_{r'} and every other λ
// is exactly the reduced (unpinned) system's. That λ_r is not a real
// sensitivity -- the equation is frozen, F_r ≡ 0 whatever Sbus -- and it must
// not reach the outputs: as λ_{q_k} it would be reported as a Q gradient at a
// bus whose Q equation is off, and gen_v_adjoint_kernel would contract it
// with ∂Q_k/∂Vm_j for every neighbour j (a row the bare system never has).
// For a masked row the value is already 0 (its column carries no live entry).
__global__ void zero_identity_rows_kernel(
          cuda_real_type* __restrict__ d_sol,       // [batch × dim_J] (slot order)
    const int*            __restrict__ d_mask_slot,
    const int*            __restrict__ d_mask_row,
    int dim_J, int n_entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    d_sol[static_cast<ptrdiff_t>(d_mask_slot[i]) * dim_J + d_mask_row[i]] = cuda_real_type(0);
}

// gen_v adjoint: for every (slot, Vm-fixed bus k) the contraction of the
// adjoint vector with the dS_calc/dVm_k column (which J never stores for a
// Vm-fixed bus):
//   λ̃_i = λ[p_row(i)] + j λ[q_row(i)]      (0 where the bus owns no row)
//   w_i  = conj(λ̃_i) V_i
//   g_k  = − Re( e^{−jθ_k} · Σ_{i∈col k} conj(Y_ik) w_i + conj(λ̃_k) e^{jθ_k} conj(I_k) )
// Y_ik is read through the Ybus transpose-position map (row k's pattern is
// column k's pattern: the pattern is symmetric); I_k = Σ_j Y_kj V_j is
// recomputed inline from row k. Non-finite V (masked buses) count as 0; a
// non-finite V_k gives g_k = 0. The minus sign is the implicit-function
// sign: dx/dVm_k = −J⁻¹ c_k (see _batch_pf.py's docstring).
__global__ void gen_v_adjoint_kernel(
          cuda_real_type*  __restrict__ d_gvm,          // [batch × n_bus] out (slot order)
    const cuda_real_type*  __restrict__ d_lam,          // [batch × dim_J] (slot order)
    const cudaComplexType* __restrict__ d_V,            // [batch × n_bus] (slot order)
    const cudaComplexType* __restrict__ d_Yvals,        // [batch × nnz_Y] (slot order)
    const int*             __restrict__ d_Y_outer,
    const int*             __restrict__ d_Y_inner,
    const int*             __restrict__ d_YT_pos,
    const int*             __restrict__ d_p_row_of_bus,
    const int*             __restrict__ d_q_row_of_bus,
    const char*            __restrict__ d_is_vm_fixed,
    int n_bus, int nnz_Y, int dim_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_bus;
    const int       k   = static_cast<int>(tid % n_bus);
    if (b >= batch) return;

    cuda_real_type g = 0;
    if (d_is_vm_fixed[k]) {
        const cudaComplexType* V  = d_V + b * n_bus;
        const cudaComplexType* Yv = d_Yvals + b * nnz_Y;
        const cuda_real_type*  lam = d_lam + b * dim_J;

        auto lam_tilde = [&](int i) -> cudaComplexType {
            const int pr = d_p_row_of_bus[i], qr = d_q_row_of_bus[i];
            return CudaFunHelper::my_make_cuComplex(
                pr >= 0 ? lam[pr] : cuda_real_type(0),
                qr >= 0 ? lam[qr] : cuda_real_type(0));
        };
        auto finite_or_zero = [](cudaComplexType v) -> cudaComplexType {
            if (!isfinite(v.x) || !isfinite(v.y))
                return CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
            return v;
        };

        const cudaComplexType Vk = V[k];
        const cuda_real_type  mag = CudaFunHelper::my_cuCabs(Vk);
        if (isfinite(Vk.x) && isfinite(Vk.y) && mag > cuda_real_type(0)) {
            const cudaComplexType e = CudaFunHelper::my_make_cuComplex(Vk.x / mag, Vk.y / mag);
            cudaComplexType Ik  = CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
            cudaComplexType acc = Ik;
            for (int p = d_Y_outer[k]; p < d_Y_outer[k + 1]; ++p) {
                const int j = d_Y_inner[p];
                const cudaComplexType Vj = finite_or_zero(V[j]);
                Ik = CudaFunHelper::my_cuCadd(Ik, CudaFunHelper::my_cuCmul(Yv[p], Vj));
                const int pT = d_YT_pos[p];
                if (pT >= 0) {
                    const cudaComplexType Yjk = Yv[pT];
                    const cudaComplexType wj  = CudaFunHelper::my_cuCmul(
                        CudaFunHelper::my_cuConj(lam_tilde(j)), Vj);
                    acc = CudaFunHelper::my_cuCadd(acc,
                        CudaFunHelper::my_cuCmul(CudaFunHelper::my_cuConj(Yjk), wj));
                }
            }
            const cudaComplexType t1 = CudaFunHelper::my_cuCmul(CudaFunHelper::my_cuConj(e), acc);
            const cudaComplexType t2 = CudaFunHelper::my_cuCmul(
                CudaFunHelper::my_cuCmul(CudaFunHelper::my_cuConj(lam_tilde(k)), e),
                CudaFunHelper::my_cuConj(Ik));
            g = -(t1.x + t2.x);
        }
    }
    d_gvm[b * n_bus + k] = g;
}

// =============================================================================
// _prepare_adjoint — first solve_JT_batch() call only.
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::_prepare_adjoint()
{
    auto t_build_start = std::chrono::steady_clock::now();
    adjoint_ = std::make_unique<BatchAdjoint>();
    BatchAdjoint& A = *adjoint_;

    const int dim_J = base.dim_J;
    const int nnz_J = base.nnz_J;

    // Host transpose of the shared J skeleton: counting sort over columns.
    thrust::host_vector<int> h_outer(base.d_J_outer);
    thrust::host_vector<int> h_inner(base.d_J_inner);
    std::vector<int> h_JT_outer(static_cast<size_t>(dim_J) + 1, 0);
    std::vector<int> h_JT_inner(static_cast<size_t>(nnz_J), 0);
    std::vector<int> h_map(static_cast<size_t>(nnz_J), 0);
    for (int p = 0; p < nnz_J; ++p) ++h_JT_outer[static_cast<size_t>(h_inner[p]) + 1];
    for (int j = 0; j < dim_J; ++j) h_JT_outer[static_cast<size_t>(j) + 1] += h_JT_outer[static_cast<size_t>(j)];
    std::vector<int> fill(h_JT_outer.begin(), h_JT_outer.end() - 1);
    for (int i = 0; i < dim_J; ++i)
        for (int p = h_outer[i]; p < h_outer[i + 1]; ++p) {
            const int j   = h_inner[p];
            const int pos = fill[static_cast<size_t>(j)]++;
            h_JT_inner[static_cast<size_t>(pos)] = i;   // rows come out sorted
            h_map[static_cast<size_t>(p)] = pos;
        }
    upload_h2d(A.d_JT_outer, h_JT_outer.data(), h_JT_outer.size(), cs);
    upload_h2d(A.d_JT_inner, h_JT_inner.data(), h_JT_inner.size(), cs);
    upload_h2d(A.d_J_to_JT,  h_map.data(),      h_map.size(),      cs);

    A.d_JT_values.resize(static_cast<size_t>(batch_size_) * nnz_J);
    A.d_rhs.resize(static_cast<size_t>(batch_size_) * dim_J);
    A.d_sol.resize(static_cast<size_t>(batch_size_) * dim_J);
    A.d_sol_full.resize(static_cast<size_t>(n_contingencies) * dim_J);
    zero_d(A.d_rhs, cs);
    zero_d(A.d_sol, cs);
    zero_d(A.d_sol_full, cs);

    // Second uniform-batch cuDSS context over the transposed skeleton, same
    // config as the forward (so a reordering workaround applies to both).
    A.solver.initialize(
        batch_size_, dim_J, nnz_J,
        thrust::raw_pointer_cast(A.d_JT_outer.data()),
        thrust::raw_pointer_cast(A.d_JT_inner.data()),
        thrust::raw_pointer_cast(A.d_JT_values.data()),
        thrust::raw_pointer_cast(A.d_rhs.data()),
        thrust::raw_pointer_cast(A.d_sol.data()),
        cs, reordering_alg_, matching_alg_, pivot_epsilon_alg_);
    A.n_analysis = 1;
    cs.synchronize();
    A.t_build_ms = bpf_ms_since(t_build_start);
}

template <typename BatchSource>
void BatchPfDriver<BatchSource>::_prepare_gen_v_adjoint(const std::vector<char>& is_vm_fixed_bus)
{
    BatchAdjoint& A = *adjoint_;
    const int n_bus = base.n_bus;
    const int nnz_Y = base.nnz_Y;
    if (static_cast<int>(is_vm_fixed_bus.size()) != n_bus)
        throw std::runtime_error(
            "[batch_pf] solve_JT_batch: is_vm_fixed_bus must have n_bus entries");

    // Ybus transpose-position map: for entry p = (i, j), the position of
    // (j, i) in the same CSR (−1 if the pattern is not symmetric there).
    thrust::host_vector<int> h_outer(base.d_Ybus_outer);
    thrust::host_vector<int> h_inner(base.d_Ybus_inner);
    std::vector<int> h_pos(static_cast<size_t>(nnz_Y), -1);
    for (int i = 0; i < n_bus; ++i)
        for (int p = h_outer[i]; p < h_outer[i + 1]; ++p) {
            const int j = h_inner[p];
            // binary search for column i in row j (rows are column-sorted)
            int lo = h_outer[j], hi = h_outer[j + 1] - 1;
            while (lo <= hi) {
                const int mid = (lo + hi) / 2;
                const int c   = h_inner[mid];
                if (c == i)     { h_pos[static_cast<size_t>(p)] = mid; break; }
                else if (c < i) lo = mid + 1;
                else            hi = mid - 1;
            }
        }
    upload_h2d(A.d_Ybus_T_pos, h_pos.data(), h_pos.size(), cs);
    upload_h2d(A.d_p_row_of_bus, base.h_p_row_of_bus.data(), base.h_p_row_of_bus.size(), cs);
    upload_h2d(A.d_q_row_of_bus, base.h_q_row_of_bus.data(), base.h_q_row_of_bus.size(), cs);
    upload_h2d(A.d_is_vm_fixed_bus, is_vm_fixed_bus.data(), is_vm_fixed_bus.size(), cs);
    A.d_gvm.resize(static_cast<size_t>(batch_size_) * n_bus);
    A.d_gvm_full.resize(static_cast<size_t>(n_contingencies) * n_bus);
    zero_d(A.d_gvm, cs);
    zero_d(A.d_gvm_full, cs);
    cs.synchronize();
    A.gen_v_ready = true;
}

// =============================================================================
// solve_JT_batch
// =============================================================================
template <typename BatchSource>
void BatchPfDriver<BatchSource>::solve_JT_batch(
    const cuda_real_type*    d_rhs_orig,
    const cuda_real_type*    d_J_ext,
    bool                     want_gen_v_grad,
    const cudaComplexType*   d_Ybus_ext,
    const cudaComplexType*   d_V_ext_orig,
    const std::vector<char>& is_vm_fixed_bus)
{
    if (n_solves_ == 0)
        throw std::runtime_error(
            "[batch_pf] solve_JT_batch: call solve() (a forward run) first");
    if (!adjoint_) _prepare_adjoint();
    BatchAdjoint& A = *adjoint_;

    const int dim_J = base.dim_J;
    const int nnz_J = base.nnz_J;
    const int n_bus = base.n_bus;
    const int nnz_Y = base.nnz_Y;
    const int* d_map = source_.d_result_map();   // nullptr → identity
    CudaTimer timer(cs);

    // ① Permute J → Jᵀ values and (re)factorize, unless this driver's own J
    //    was already permuted + factorized since its last solve() (a second
    //    backward on the same forward, e.g. retain_graph=True).
    const bool own_J = (d_J_ext == nullptr);
    const bool need_refactor = !A.factorized || !own_J || A.factorized_for_solve != n_solves_;
    if (need_refactor) {
        const cuda_real_type* src = own_J ? thrust::raw_pointer_cast(d_J_values_batch.data()) : d_J_ext;
        transpose_gather_values_kernel<<<
            nr_grid_size((long long)batch_size_ * nnz_J, BS), BS, 0, cs>>>(
            thrust::raw_pointer_cast(A.d_JT_values.data()), src,
            thrust::raw_pointer_cast(A.d_J_to_JT.data()), nnz_J, batch_size_);
        CHK_CUDA_BPF(cudaGetLastError());
        A.solver.set_values(thrust::raw_pointer_cast(A.d_JT_values.data()));
        A.solver.prepare_factorization();
        timer.start();
        if (!A.factorized) {
            A.solver.factor();
            A.t_first_factorize += timer.stop_ms();
            ++A.n_factorize;
            A.factorized = true;
        } else {
            A.solver.refactor();
            A.t_refactorize += timer.stop_ms();
            ++A.n_refactorize;
        }
        A.factorized_for_solve = own_J ? n_solves_ : -1;
    }

    // ② Right-hand side: original order → slot order (non-finite → 0), phantom
    //    slots zero (their base-case Jᵀ then yields λ = 0).
    zero_d(A.d_rhs, cs);
    launch_gather_rows(thrust::raw_pointer_cast(A.d_rhs.data()), d_rhs_orig, d_map,
                       dim_J, n_active_, /*zero_nonfinite=*/true, cs);
    CHK_CUDA_BPF(cudaGetLastError());

    // ③ Solve, drop the identity-row components (see zero_identity_rows_kernel;
    //    the driver is one chunk here -- keep_final_jacobian requires it -- so
    //    the chunk-0 slice is the whole batch) and scatter back to original order.
    timer.start();
    A.solver.solve();
    A.t_solve += timer.stop_ms();
    ++A.n_solve;
    {
        NrIterBuffers mb{};
        source_.fill_mask_buffers(mb, /*chunk_idx=*/0,
                                  thrust::raw_pointer_cast(base.d_J_outer.data()));
        if (mb.n_mask_rows > 0) {
            zero_identity_rows_kernel<<<(mb.n_mask_rows + BS - 1) / BS, BS, 0, cs>>>(
                thrust::raw_pointer_cast(A.d_sol.data()),
                mb.d_mask_slot, mb.d_mask_row, dim_J, mb.n_mask_rows);
            CHK_CUDA_BPF(cudaGetLastError());
        }
    }
    zero_d(A.d_sol_full, cs);
    launch_scatter_rows(thrust::raw_pointer_cast(A.d_sol_full.data()),
                        thrust::raw_pointer_cast(A.d_sol.data()), d_map,
                        dim_J, n_active_, cs);
    CHK_CUDA_BPF(cudaGetLastError());

    // ④ gen_v contraction (optional).
    if (want_gen_v_grad) {
        if (!A.gen_v_ready) _prepare_gen_v_adjoint(is_vm_fixed_bus);
        const cudaComplexType* d_V_slots;
        if (d_V_ext_orig) {
            A.d_V_ext_slots.resize(static_cast<size_t>(batch_size_) * n_bus);
            launch_gather_rows(thrust::raw_pointer_cast(A.d_V_ext_slots.data()), d_V_ext_orig,
                               d_map, n_bus, n_active_, /*zero_nonfinite=*/false, cs);
            d_V_slots = thrust::raw_pointer_cast(A.d_V_ext_slots.data());
        } else {
            d_V_slots = thrust::raw_pointer_cast(d_V_batch.data());
        }
        const cudaComplexType* d_Y_slots =
            d_Ybus_ext ? d_Ybus_ext : thrust::raw_pointer_cast(d_Ybus_values_batch.data());
        gen_v_adjoint_kernel<<<nr_grid_size((long long)n_active_ * n_bus, BS), BS, 0, cs>>>(
            thrust::raw_pointer_cast(A.d_gvm.data()),
            thrust::raw_pointer_cast(A.d_sol.data()),
            d_V_slots, d_Y_slots,
            thrust::raw_pointer_cast(base.d_Ybus_outer.data()),
            thrust::raw_pointer_cast(base.d_Ybus_inner.data()),
            thrust::raw_pointer_cast(A.d_Ybus_T_pos.data()),
            thrust::raw_pointer_cast(A.d_p_row_of_bus.data()),
            thrust::raw_pointer_cast(A.d_q_row_of_bus.data()),
            thrust::raw_pointer_cast(A.d_is_vm_fixed_bus.data()),
            n_bus, nnz_Y, dim_J, n_active_);
        CHK_CUDA_BPF(cudaGetLastError());
        zero_d(A.d_gvm_full, cs);
        launch_scatter_rows(thrust::raw_pointer_cast(A.d_gvm_full.data()),
                            thrust::raw_pointer_cast(A.d_gvm.data()), d_map,
                            n_bus, n_active_, cs);
        CHK_CUDA_BPF(cudaGetLastError());
    }

    cs.synchronize();
}

// =============================================================================
// Explicit template instantiations
// =============================================================================
template struct BatchPfDriver<ContingencyBatch>;
template struct BatchPfDriver<InjectionBatch>;
template struct BatchPfDriver<ScenarioSweepBatch>;