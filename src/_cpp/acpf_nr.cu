// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// acpf_nr.cu  —  AC Newton-Raphson power flow, refactored around AcPfNrState
// =============================================================================
//
// Structure
// ---------
//   §1  Thrust functors        (AbsFunctor)
//   §2  CPU helper             (build_J_structure)
//   §3  AcPfNrState constructor  (Phase 0: setup + ANALYSIS + NR loop)
//   §4  AcPfNrState::copy_V_to_host
//   §5  acpf_nr_gpu()            (thin public wrapper, Python-facing)
//   §6  AcPfNrSession            (stateful wrapper for DLPack zero-copy export)
//   §7  AcPfNrState::prepare_JT / solve_JT  (adjoint primitive)
//
// Device kernels (fill_FP, fill_FQ, fill_J, update_Va, update_Vm,
//                 compute_residuals) live in acpf_nr_kernels.cuh and are
// shared with contingency_solver.cu; called here with actual_batch=1.
//
// Stream discipline
// -----------------
//   Every GPU operation is enqueued on cs (the CudaStream member):
//     • cuSPARSE SpMV  — handle bound via cusparseSetStream(spmv.handle, cs)
//     • cuDSS phases   — handle bound via cudssSetStream(dss.handle, cs)
//     • Thrust reductions — thrust::cuda::par.on(cs)
//     • Custom kernels — launched with <<<grid, BS, 0, cs>>>
//
//   Alpha/beta scalars for cuSPARSE are host-side (h_cplx_one / h_cplx_zero),
//   which requires CUSPARSE_POINTER_MODE_HOST.  No CudaDeviceScalar needed.
//
//   The broad cudaDeviceSynchronize() between update_Va and update_Vm is
//   replaced by a narrow event sync (e_va_done) that only blocks until that
//   one point on cs is reached, leaving other streams free.
//
//   timer.stop_ms() calls cudaEventSynchronize internally, so the redundant
//   cudaDeviceSynchronize() after cuDSS factorize/solve is dropped.
//
// See acpf_nr_state.cuh for the full class declaration and design notes.
// =============================================================================

#include "acpf_nr_state.cuh"
#include "acpf_nr.hpp"
#include "acpf_nr_kernels.cuh"
#include "nr_iter_step.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cusparse.h>
#include <cudss.h>

#include <Eigen/SparseCore>

#include <algorithm>   // std::sort, std::lower_bound
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking macros — throw on failure so the AcPfNrState constructor
// unwinds cleanly through RAII destructors.
// ---------------------------------------------------------------------------
#define CHK_CUDA_AC(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("[acpf_nr] CUDA error: ")                          \
                + cudaGetErrorString(_e));                                     \
    } while(0)

#define CHK_CSP_AC(call)                                                       \
    do {                                                                       \
        cusparseStatus_t _s = (call);                                          \
        if (_s != CUSPARSE_STATUS_SUCCESS)                                     \
            throw std::runtime_error(                                          \
                std::string("[acpf_nr] cuSPARSE error: ")                      \
                + cusparseGetErrorString(_s));                                 \
    } while(0)

#define CHK_DSS_AC(call)                                                       \
    do {                                                                       \
        cudssStatus_t _s = (call);                                             \
        if (_s != CUDSS_STATUS_SUCCESS)                                        \
            throw std::runtime_error(                                          \
                std::string("[acpf_nr] cuDSS error, status=")                  \
                + std::to_string(_s));                                         \
    } while(0)

// BS = 256 is defined in nr_iter_step.cuh (included above).

// =============================================================================
// §1  Thrust functors  (named structs required for C++14 / nvcc compatibility)
// =============================================================================

struct AbsFunctor {
    __host__ __device__
    cuda_real_type operator()(cuda_real_type x) const { return fabs(x); }
};

// =============================================================================
// §2  CPU helper: build_J_structure
//   Derives the J CSR sparsity skeleton and four scatter maps from the
//   RowMajor Ybus restricted to the pvpq/pq subsets.
//   Identical logic to the original acpf_nr.cu; reproduced here so this
//   translation unit is self-contained.
// =============================================================================
static void build_J_structure(
    Eigen::SparseMatrix<eigen_real_type, Eigen::RowMajor>& J_csr,
    std::vector<int>& map_j11,
    std::vector<int>& map_j12,
    std::vector<int>& map_j21,
    std::vector<int>& map_j22,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus,
    const Eigen::VectorXi& pvpq,
    const Eigen::VectorXi& pq)
{
    const int n_bus  = Ybus.rows();
    const int n_pvpq = pvpq.size();
    const int n_pq   = pq.size();
    const int dim_J  = n_pvpq + n_pq;
    const int nnz_Y  = Ybus.nonZeros();

    std::vector<int> pvpq_inv(n_bus, -1), pq_inv(n_bus, -1);
    for (int i = 0; i < n_pvpq; ++i) pvpq_inv[pvpq(i)] = i;
    for (int i = 0; i < n_pq;   ++i) pq_inv[pq(i)]     = i;

    struct Contrib { int jrow, jcol, ybus_k; };
    std::vector<Contrib> c11, c12, c21, c22;

    int k = 0;
    for (int outer = 0; outer < Ybus.outerSize(); ++outer) {
        for (Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>::InnerIterator
             it(Ybus, outer); it; ++it, ++k)
        {
            int i = (int)it.row(), j = (int)it.col();
            int ri = pvpq_inv[i], rq = pq_inv[i];
            int ci = pvpq_inv[j], cq = pq_inv[j];
            if (ri >= 0 && ci >= 0) c11.push_back({ri,          ci,          k});
            if (ri >= 0 && cq >= 0) c12.push_back({ri,          n_pvpq + cq, k});
            if (rq >= 0 && ci >= 0) c21.push_back({n_pvpq + rq, ci,          k});
            if (rq >= 0 && cq >= 0) c22.push_back({n_pvpq + rq, n_pvpq + cq, k});
        }
    }

    std::vector<Eigen::Triplet<eigen_real_type>> triplets;
    triplets.reserve(c11.size() + c12.size() + c21.size() + c22.size());
    for (auto& c : c11) triplets.push_back({c.jrow, c.jcol, 0.});
    for (auto& c : c12) triplets.push_back({c.jrow, c.jcol, 0.});
    for (auto& c : c21) triplets.push_back({c.jrow, c.jcol, 0.});
    for (auto& c : c22) triplets.push_back({c.jrow, c.jcol, 0.});

    J_csr.resize(dim_J, dim_J);
    J_csr.setFromTriplets(triplets.begin(), triplets.end());
    J_csr.makeCompressed();

    map_j11.assign(nnz_Y, -1);
    map_j12.assign(nnz_Y, -1);
    map_j21.assign(nnz_Y, -1);
    map_j22.assign(nnz_Y, -1);

    auto find_J_pos = [&](int row, int col) -> int {
        int start = J_csr.outerIndexPtr()[row];
        int end   = J_csr.outerIndexPtr()[row + 1];
        const int* inner = J_csr.innerIndexPtr();
        auto it = std::lower_bound(inner + start, inner + end, col);
        if (it == inner + end || *it != col) return -1;
        return (int)(it - inner);
    };

    for (auto& c : c11) map_j11[c.ybus_k] = find_J_pos(c.jrow, c.jcol);
    for (auto& c : c12) map_j12[c.ybus_k] = find_J_pos(c.jrow, c.jcol);
    for (auto& c : c21) map_j21[c.ybus_k] = find_J_pos(c.jrow, c.jcol);
    for (auto& c : c22) map_j22[c.ybus_k] = find_J_pos(c.jrow, c.jcol);
}

// =============================================================================
// §4  AcPfNrState constructor
// =============================================================================

AcPfNrState::AcPfNrState(
    const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
    Eigen::Ref<const CplxVect>                  Vinit,
    Eigen::Ref<const CplxVect>                  Sbus_host,
    Eigen::Ref<const Eigen::VectorXi>           pv,
    Eigen::Ref<const Eigen::VectorXi>           pq_in,
    int                                         max_iter,
    eigen_real_type                             tol,
    int                                         device)
{
    // =========================================================================
    // Device selection — must happen before any stream/allocation.
    // =========================================================================
    if (device >= 0) {
        int count = 0;
        CHK_CUDA_AC(cudaGetDeviceCount(&count));
        if (device >= count)
            throw std::runtime_error(
                "[acpf_nr] device index " + std::to_string(device)
                + " out of range (device count = " + std::to_string(count) + ")");
        CHK_CUDA_AC(cudaSetDevice(device));
    }
    CHK_CUDA_AC(cudaGetDevice(&device_id_));

    // Wall clock for the mixed CPU+GPU init phase; chrono is appropriate here
    // because the work is heterogeneous (CPU preprocessing + device uploads +
    // ANALYSIS launch).
    auto t_wall_start = std::chrono::steady_clock::now();

    // =========================================================================
    // 4.1  Scalar dimensions
    // =========================================================================
    n_bus  = Ybus.rows();
    n_pv   = pv.size();
    n_pq   = pq_in.size();
    n_pvpq = n_pv + n_pq;
    dim_J  = n_pvpq + n_pq;
    nnz_Y  = Ybus.nonZeros();

    // =========================================================================
    // 4.2  CPU preprocessing
    // =========================================================================

    // pvpq = sorted union of pv and pq (slack buses absent)
    Eigen::VectorXi pvpq_host(n_pvpq);
    pvpq_host << pv, pq_in;
    std::sort(pvpq_host.data(), pvpq_host.data() + n_pvpq);

    // *** Must convert to RowMajor BEFORE building scatter maps ***
    // build_J_structure iterates in CSR (outer=row) order; passing a ColMajor
    // matrix would produce incorrect scatter maps (the bug that was fixed
    // previously by introducing Ybus_rm).
    Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor> Ybus_rm = Ybus;

    Eigen::SparseMatrix<eigen_real_type, Eigen::RowMajor> J_csr;
    std::vector<int> h_map_j11, h_map_j12, h_map_j21, h_map_j22;
    build_J_structure(J_csr, h_map_j11, h_map_j12, h_map_j21, h_map_j22,
                      Ybus_rm, pvpq_host, pq_in);
    nnz_J = J_csr.nonZeros();
    timings.t_build_J_ms = ms_since(t_wall_start);

    // =========================================================================
    // 4.3  Upload to device  (all on stream cs)
    //
    //   upload_h2d / copy_d2d / zero_d (from cuda_utils.h) wrap
    //   cudaMemcpyAsync / cudaMemsetAsync so every transfer is explicitly
    //   stream-ordered.  thrust::copy with par.on(cs) is NOT used here:
    //   thrust routes all copies involving raw pointers through the GPU
    //   code path (treating them as D→D) regardless of where the data
    //   actually lives, which causes cudaErrorInvalidValue at runtime.
    // =========================================================================

    auto t_upload_start = std::chrono::steady_clock::now();

    // Bus-index arrays (H→D)
    upload_h2d(d_pvpq, pvpq_host.data(),       n_pvpq, cs);
    upload_h2d(d_pq,   pq_in.data(),           n_pq,   cs);

    // Scatter maps (H→D)
    upload_h2d(d_map_j11, h_map_j11.data(), nnz_Y, cs);
    upload_h2d(d_map_j12, h_map_j12.data(), nnz_Y, cs);
    upload_h2d(d_map_j21, h_map_j21.data(), nnz_Y, cs);
    upload_h2d(d_map_j22, h_map_j22.data(), nnz_Y, cs);

    // Ybus (RowMajor CSR, FP32 complex)  (H→D)
    {
        thrust::host_vector<cudaComplexType> h_yv(nnz_Y);
        for (int i = 0; i < nnz_Y; ++i)
            h_yv[i] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(Ybus_rm.valuePtr()[i].real()),
                static_cast<cuda_real_type>(Ybus_rm.valuePtr()[i].imag()));
        upload_h2d(d_Ybus_values, h_yv.data(),              nnz_Y,                  cs);
        upload_h2d(d_Ybus_outer,  Ybus_rm.outerIndexPtr(),  Ybus_rm.outerSize() + 1, cs);
        upload_h2d(d_Ybus_inner,  Ybus_rm.innerIndexPtr(),  nnz_Y,                  cs);
    }

    // Sbus  (H→D)
    {
        thrust::host_vector<cudaComplexType> h_s(n_bus);
        for (int i = 0; i < n_bus; ++i)
            h_s[i] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(Sbus_host(i).real()),
                static_cast<cuda_real_type>(Sbus_host(i).imag()));
        upload_h2d(d_Sbus, h_s.data(), n_bus, cs);
    }

    // V (initial)  (H→D)
    {
        thrust::host_vector<cudaComplexType> h_v(n_bus);
        for (int i = 0; i < n_bus; ++i)
            h_v[i] = CudaFunHelper::my_make_cuComplex(
                static_cast<cuda_real_type>(Vinit(i).real()),
                static_cast<cuda_real_type>(Vinit(i).imag()));
        upload_h2d(d_V, h_v.data(), n_bus, cs);
    }

    // Working vectors — allocate then zero-fill on cs
    //   d_Ibus: overwritten by the first SpMV before it is read; no fill needed.
    d_Ibus.resize(n_bus);
    d_F.resize(dim_J);   zero_d(d_F,  cs);
    d_dx.resize(dim_J);  zero_d(d_dx, cs);

    // Jacobian skeleton (H→D)
    upload_h2d(d_J_outer, J_csr.outerIndexPtr(), J_csr.outerSize() + 1, cs);
    upload_h2d(d_J_inner, J_csr.innerIndexPtr(), nnz_J,                 cs);
    d_J_values.resize(nnz_J);  zero_d(d_J_values, cs);

    timings.t_upload_ms = ms_since(t_upload_start);
    timings.t_init_ms   = ms_since(t_wall_start);

    // =========================================================================
    // 4.4  cuSPARSE SpMV descriptor  (Ybus · d_V → d_Ibus)
    //
    //   CUSPARSE_POINTER_MODE_HOST: alpha/beta are host-side constants
    //   (h_cplx_one / h_cplx_zero from cu_complex_utils.h).
    //   This avoids the CudaDeviceScalar used in the original code and
    //   sidesteps the device-pointer aliasing bug that caused silent
    //   context corruption.
    // =========================================================================
    CHK_CSP_AC(cusparseCreate(&spmv.handle));
    CHK_CSP_AC(cusparseSetPointerMode(spmv.handle, CUSPARSE_POINTER_MODE_HOST));
    CHK_CSP_AC(cusparseSetStream(spmv.handle, cs));  // bind to our stream

    CHK_CSP_AC(cusparseCreateConstCsr(
        &spmv.mat, n_bus, n_bus, nnz_Y,
        thrust::raw_pointer_cast(d_Ybus_outer.data()),
        thrust::raw_pointer_cast(d_Ybus_inner.data()),
        thrust::raw_pointer_cast(d_Ybus_values.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_TYPE));

    CHK_CSP_AC(cusparseCreateConstDnVec(
        &spmv.vec_x, n_bus,
        thrust::raw_pointer_cast(d_V.data()), CUDA_C_TYPE));

    CHK_CSP_AC(cusparseCreateDnVec(
        &spmv.vec_y, n_bus,
        thrust::raw_pointer_cast(d_Ibus.data()), CUDA_C_TYPE));

    CHK_CSP_AC(cusparseSpMV_bufferSize(
        spmv.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_cplx_one, spmv.mat, spmv.vec_x,
        &h_cplx_zero, spmv.vec_y,
        CUDA_C_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, &spmv.buf.size));

    if (spmv.buf.size > 0)
        CHK_CUDA_AC(cudaMalloc(&spmv.buf.ptr, spmv.buf.size));

    // =========================================================================
    // 4.5  cuDSS context + ANALYSIS  (symbolic factorisation, paid once)
    // =========================================================================
    CHK_DSS_AC(cudssCreate(&dss.handle));
    CHK_DSS_AC(cudssSetStream(dss.handle, cs));  // bind to our stream
    CHK_DSS_AC(cudssConfigCreate(&dss.config));
    CHK_DSS_AC(cudssDataCreate(dss.handle, &dss.data));

    // Sparse J descriptor — values pointer is updated each iteration via
    // cudssMatrixSetValues(); the shape and structure pointers are fixed.
    dss_A.create_csr(dim_J, nnz_J,
        thrust::raw_pointer_cast(d_J_outer.data()),
        thrust::raw_pointer_cast(d_J_inner.data()),
        thrust::raw_pointer_cast(d_J_values.data()));

    // Dense dx (solution) and F (RHS)
    dss_x.create_dn(dim_J, thrust::raw_pointer_cast(d_dx.data()));
    dss_b.create_dn(dim_J, thrust::raw_pointer_cast(d_F.data()));

    // ANALYSIS: reordering + symbolic factorisation.
    // Wall-clock timing via chrono (mixed CPU/GPU work; a CUDA event pair
    // would only capture the GPU tail).
    {
        auto t_analyse = std::chrono::steady_clock::now();
        dss.analyze(dss_A, dss_x, dss_b);
        timings.t_analyze_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_analyse).count();
    }

    // =========================================================================
    // 4.6  Newton-Raphson loop  (all computation on cs)
    //
    //   Convergence is checked AFTER each full NR step (option B): nr_iter_step
    //   runs SpMV + fill_F + fill_J + factorize + solve + update_V, and the
    //   mismatch left in d_F is tested immediately after.
    //
    //   Trade-off: the iteration on which ‖F‖∞ first drops below tol has
    //   already performed a fill_J + factorize + solve + update_V that was
    //   technically unnecessary (dx ≈ 0 for a converged system).  This is
    //   acceptable: one extra factorize per solve call is negligible compared
    //   to the overall NR cost, and it keeps the loop body uniform with
    //   contingency_solver.cu.
    // =========================================================================

    // Non-owning pointer bundle — views into the single-system device vectors.
    const NrIterBuffers buf {
        thrust::raw_pointer_cast(d_J_values.data()),
        thrust::raw_pointer_cast(d_V.data()),
        thrust::raw_pointer_cast(d_Ibus.data()),
        thrust::raw_pointer_cast(d_F.data()),
        thrust::raw_pointer_cast(d_dx.data()),
        thrust::raw_pointer_cast(d_Ybus_values.data()),
        thrust::raw_pointer_cast(d_Ybus_outer.data()),
        thrust::raw_pointer_cast(d_Ybus_inner.data()),
        thrust::raw_pointer_cast(d_map_j11.data()),
        thrust::raw_pointer_cast(d_map_j12.data()),
        thrust::raw_pointer_cast(d_map_j21.data()),
        thrust::raw_pointer_cast(d_map_j22.data()),
        thrust::raw_pointer_cast(d_pvpq.data()),
        thrust::raw_pointer_cast(d_pq.data()),
        thrust::raw_pointer_cast(d_Sbus.data()),
        /*sbus_stride=*/0,   // single-system NR: shared Sbus
    };

    CudaTimer timer(cs);  // stream-aware: events recorded on cs

    for (int iter = 0; iter < max_iter; ++iter)
    {
        timings.nb_iter++;

        NrIterTimings step;
        nr_iter_step(spmv, dss, dss_A, dss_x, dss_b, buf,
                     n_bus, n_pvpq, n_pq, dim_J, nnz_Y, nnz_J,
                     /*actual_batch=*/1, cs, timer,
                     /*first_factorize=*/(iter == 0), step);
        timings.t_spmv            += step.t_spmv;
        timings.t_fill_F          += step.t_fill_F;
        timings.t_fill_J          += step.t_fill_J;
        timings.t_first_factorize += step.t_first_factorize;
        timings.t_refactorize     += step.t_refactorize;
        if (iter > 0) timings.n_refactorize++;
        timings.t_solve           += step.t_solve;
        timings.t_update_V        += step.t_update_V;

        // ‖F‖∞ convergence check on the mismatch left by nr_iter_step.
        // d_F holds −[ΔP, ΔQ] of the just-updated V; no extra SpMV needed.
        timer.start();
        cuda_real_type norm_F = thrust::transform_reduce(
            thrust::cuda::par.on(cs),
            d_F.begin(), d_F.end(),
            AbsFunctor{},
            cuda_real_type(0.),
            thrust::maximum<cuda_real_type>());
        timings.t_mismatch += timer.stop_ms();

        if (norm_F < static_cast<cuda_real_type>(tol)) {
            timings.converged = true;
            break;
        }
    }
    // §4.8 (post-loop final check) is no longer needed: nr_iter_step always
    // recomputes d_F before the convergence test above, so timings.converged
    // is accurate whether the loop exited via break or by exhausting max_iter.

    // =========================================================================
    // 4.9  Snapshot base-case solution (D→D, on cs)
    //
    //   copy_d2d (cuda_utils.h) wraps cudaMemcpyAsync(DeviceToDevice) and
    //   keeps the copies ordered after all NR work on cs.
    // =========================================================================
    copy_d2d(d_V_base,        d_V,        cs);
    copy_d2d(d_J_values_base, d_J_values, cs);

    // cuDSS context intentionally left alive:
    //   ContingencyAnalysisSolver will call REFACTORIZATION / SOLVE directly
    //   on dss / dss_A / dss_b / dss_x, amortising ANALYSIS across the batch.

    // Build and factor Jᵀ for the adjoint solve (implicit function theorem).
    prepare_JT();
}

// =============================================================================
// §5  AcPfNrState::copy_V_to_host
// =============================================================================

void AcPfNrState::copy_V_to_host(Eigen::Ref<CplxVect> V_out) const
{
    // Synchronise cs before the D→H transfer.  The host_vector assignment
    // below uses the default stream; without this sync it could race ahead
    // of any pending work on cs and read stale device data.
    cs.synchronize();

    thrust::host_vector<cudaComplexType> h_V = d_V;   // D→H, default stream
    V_out.resize(n_bus);
    for (int i = 0; i < n_bus; ++i)
        V_out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[i].x),
            static_cast<eigen_real_type>(h_V[i].y));
}

// =============================================================================
// §6  acpf_nr_gpu  —  thin public wrapper  (Python-facing)
// =============================================================================

AcPfTimings acpf_nr_gpu(
    Eigen::Ref<CplxVect>                        V_out,
    const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
    Eigen::Ref<const CplxVect>                  Vinit,
    Eigen::Ref<const CplxVect>                  Sbus,
    Eigen::Ref<const Eigen::VectorXi>           slack_ids,
    Eigen::Ref<const RealVect>                  slack_weights,
    Eigen::Ref<const Eigen::VectorXi>           pv,
    Eigen::Ref<const Eigen::VectorXi>           pq,
    int                                         max_iter,
    eigen_real_type                             tol,
    int                                         /*nb_solve*/,  // unused for now
    int                                         device
)
{
    (void)slack_ids;
    (void)slack_weights;

    AcPfNrState state(Ybus, Vinit, Sbus, pv, pq, max_iter, tol, device);
    state.copy_V_to_host(V_out);
    return state.timings;
}

// =============================================================================
// §7  AcPfNrSession  —  stateful wrapper; keeps AcPfNrState alive for DLPack
// =============================================================================

AcPfNrSession::AcPfNrSession(
    const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
    Eigen::Ref<const CplxVect>                  Vinit,
    Eigen::Ref<const CplxVect>                  Sbus,
    Eigen::Ref<const Eigen::VectorXi>           slack_ids,
    Eigen::Ref<const RealVect>                  slack_weights,
    Eigen::Ref<const Eigen::VectorXi>           pv,
    Eigen::Ref<const Eigen::VectorXi>           pq,
    int                                         max_iter,
    eigen_real_type                             tol,
    int                                         device
)
{
    (void)slack_ids;
    (void)slack_weights;
    state_ = std::make_shared<AcPfNrState>(Ybus, Vinit, Sbus, pv, pq,
                                            max_iter, tol, device);
}

// =============================================================================
// §7  AcPfNrState adjoint helpers — prepare_JT() and solve_JT()
//
// cuDSS 0.7 exposes CUDSS_CONFIG_SOLVE_MODE for a future transpose mode but
// documents "Other values are not supported" — only 0 (forward solve) works.
// We therefore build Jᵀ explicitly via cusparseCsr2cscEx2 (which converts a
// CSR matrix to CSC, and CSC(A) == CSR(Aᵀ) for any square matrix) and factor
// it with a dedicated CudssContext.  Both the transpose and its LU factors are
// computed once at the end of the constructor and reused across all backward
// passes.
// =============================================================================

void AcPfNrState::prepare_JT()
{
    // Allocate device storage for Jᵀ CSR and work vectors
    d_JT_outer.resize(dim_J + 1);
    d_JT_inner.resize(nnz_J);
    d_JT_values.resize(nnz_J);
    d_JT_rhs.resize(dim_J);   zero_d(d_JT_rhs, cs);
    d_JT_sol.resize(dim_J);   zero_d(d_JT_sol, cs);

    // -------------------------------------------------------------------------
    // Transpose J → Jᵀ via cusparseCsr2cscEx2
    // A temporary cuSPARSE handle is used here (one-time setup); the main
    // SpMV handle (spmv.handle) is typed for complex Ybus and must not be
    // reused for real-valued J operations.
    // -------------------------------------------------------------------------
    cusparseHandle_t csp = nullptr;
    CHK_CSP_AC(cusparseCreate(&csp));
    CHK_CSP_AC(cusparseSetStream(csp, cs));

    size_t buf_sz = 0;
    CHK_CSP_AC(cusparseCsr2cscEx2_bufferSize(
        csp,
        dim_J, dim_J, nnz_J,
        thrust::raw_pointer_cast(d_J_values_base.data()),
        thrust::raw_pointer_cast(d_J_outer.data()),
        thrust::raw_pointer_cast(d_J_inner.data()),
        thrust::raw_pointer_cast(d_JT_values.data()),
        thrust::raw_pointer_cast(d_JT_outer.data()),
        thrust::raw_pointer_cast(d_JT_inner.data()),
        static_cast<cudaDataType_t>(CudaFunHelper::CUDSS_R_TYPE),  // CUDA_R_32F or CUDA_R_64F
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT,
        &buf_sz));

    void* buf = nullptr;
    if (buf_sz > 0)
        CHK_CUDA_AC(cudaMalloc(&buf, buf_sz));

    CHK_CSP_AC(cusparseCsr2cscEx2(
        csp,
        dim_J, dim_J, nnz_J,
        thrust::raw_pointer_cast(d_J_values_base.data()),
        thrust::raw_pointer_cast(d_J_outer.data()),
        thrust::raw_pointer_cast(d_J_inner.data()),
        thrust::raw_pointer_cast(d_JT_values.data()),
        thrust::raw_pointer_cast(d_JT_outer.data()),
        thrust::raw_pointer_cast(d_JT_inner.data()),
        static_cast<cudaDataType_t>(CudaFunHelper::CUDSS_R_TYPE),
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG_DEFAULT,
        buf));

    if (buf) CHK_CUDA_AC(cudaFree(buf));
    CHK_CSP_AC(cusparseDestroy(csp));

    // -------------------------------------------------------------------------
    // Factor Jᵀ with a fresh CudssContext
    // Stream must be bound immediately after cudssCreate and before
    // cudssConfigCreate / cudssDataCreate (mirrors the pattern for dss above).
    // -------------------------------------------------------------------------
    CHK_DSS_AC(cudssCreate(&dss_T.handle));
    CHK_DSS_AC(cudssSetStream(dss_T.handle, cs));
    CHK_DSS_AC(cudssConfigCreate(&dss_T.config));
    CHK_DSS_AC(cudssDataCreate(dss_T.handle, &dss_T.data));

    dss_AT.create_csr(static_cast<int64_t>(dim_J),
                      static_cast<int64_t>(nnz_J),
                      thrust::raw_pointer_cast(d_JT_outer.data()),
                      thrust::raw_pointer_cast(d_JT_inner.data()),
                      thrust::raw_pointer_cast(d_JT_values.data()));
    dss_bT.create_dn(static_cast<int64_t>(dim_J),
                     thrust::raw_pointer_cast(d_JT_rhs.data()));
    dss_xT.create_dn(static_cast<int64_t>(dim_J),
                     thrust::raw_pointer_cast(d_JT_sol.data()));

    dss_T.analyze(dss_AT, dss_xT, dss_bT);
    dss_T.factorize(dss_AT, dss_xT, dss_bT);
}

void AcPfNrState::solve_JT(const cuda_real_type* d_rhs, cuda_real_type* d_sol)
{
    // Copy caller's rhs into the internal d_JT_rhs buffer (D→D, stream-ordered)
    CHK_CUDA_AC(cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_JT_rhs.data()),
        d_rhs,
        static_cast<std::size_t>(dim_J) * sizeof(cuda_real_type),
        cudaMemcpyDeviceToDevice,
        cs));

    // Update the RHS pointer in the descriptor (values in d_JT_rhs may change
    // between backward calls even though the sparsity pattern is fixed).
    dss_bT.set_values(thrust::raw_pointer_cast(d_JT_rhs.data()));

    // Solve Jᵀ λ = rhs;  solution written to d_JT_sol.
    dss_T.solve(dss_AT, dss_xT, dss_bT);

    // Copy solution out to caller's buffer (D→D, stream-ordered)
    CHK_CUDA_AC(cudaMemcpyAsync(
        d_sol,
        thrust::raw_pointer_cast(d_JT_sol.data()),
        static_cast<std::size_t>(dim_J) * sizeof(cuda_real_type),
        cudaMemcpyDeviceToDevice,
        cs));
}

AcPfTimings AcPfNrSession::timings() const {
    return state_->timings;
}

CplxVect AcPfNrSession::get_v() const {
    CplxVect out(state_->n_bus);
    state_->copy_V_to_host(out);
    return out;
}

int AcPfNrSession::n_pvpq() const { return state_->n_pvpq; }
int AcPfNrSession::n_pq()   const { return state_->n_pq; }
int AcPfNrSession::dim_J()  const { return state_->dim_J; }

std::vector<int> AcPfNrSession::pvpq() const {
    thrust::host_vector<int> h(state_->d_pvpq);
    return std::vector<int>(h.begin(), h.end());
}

std::vector<int> AcPfNrSession::pq() const {
    thrust::host_vector<int> h(state_->d_pq);
    return std::vector<int>(h.begin(), h.end());
}