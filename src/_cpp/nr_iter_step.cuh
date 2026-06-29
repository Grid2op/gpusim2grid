// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// nr_iter_step.cuh — One Newton-Raphson iteration step
// =============================================================================
//
// Shared by acpf_nr.cu (actual_batch=1) and contingency_solver.cu
// (actual_batch=N).  Eliminates the duplicated kernel-launch boilerplate.
//
// Usage
// -----
//   1. Fill an NrIterBuffers with raw pointers cast from thrust::device_vectors.
//   2. Call nr_iter_step() inside the NR loop.
//   3. Accumulate the returned NrIterTimings into the caller's timing struct:
//        timings.t_spmv += step.t_spmv;  // ... × 6
//
// Block-size constant
// -------------------
//   BS = 256 is defined here and shared with callers.  Do NOT redefine it in
//   files that include this header.
// =============================================================================

#ifndef NR_ITER_STEP_CUH
#define NR_ITER_STEP_CUH

#include "acpf_nr_kernels.cuh"
#include "cuda_utils.h"
#include "timing_utils.hpp"
#include "dtypes.hpp"
#include "cu_complex_utils.h"

static constexpr int BS = 256;   // block size for all NR kernels

// -----------------------------------------------------------------------------
// NrIterBuffers — non-owning raw pointer bundle
//
// Per-batch fields use stride actual_batch × per-system size:
//   d_V[b * n_bus + i],  d_F[b * dim_J + k],  etc.
//
// Shared single-system fields have no b-stride (same arrays for all systems):
//   d_Ybus_outer, d_Ybus_inner, d_map_j*, d_pvpq, d_pq, d_Sbus.
// -----------------------------------------------------------------------------
struct NrIterBuffers {
    // per-batch — written by kernels
    cuda_real_type*  d_J_values;         // [actual_batch × nnz_J]
    cudaComplexType* d_V;                // [actual_batch × n_bus]  (in-place update)
    cudaComplexType* d_Ibus;             // [actual_batch × n_bus]  (SpMV output)
    cuda_real_type*  d_F;                // [actual_batch × dim_J]
    cuda_real_type*  d_dx;               // [actual_batch × dim_J]  (solve output)

    // per-batch — read-only (base or patched Ybus values)
    const cudaComplexType* d_Ybus_values;  // [actual_batch × nnz_Y]

    // shared single-system — read-only
    const int*             d_Ybus_outer;
    const int*             d_Ybus_inner;
    const int*             d_map_j11;
    const int*             d_map_j12;
    const int*             d_map_j21;
    const int*             d_map_j22;
    const int*             d_pvpq;
    const int*             d_pq;

    // Sbus pointer + stride control (see acpf_nr_kernels.cuh):
    //   sbus_stride = 0     → d_Sbus is a single-system (n_bus,) buffer
    //                         shared across all systems in the batch
    //   sbus_stride = n_bus → d_Sbus is a per-batch (actual_batch × n_bus)
    //                         layout indexed as d_Sbus[b * n_bus + bus]
    const cudaComplexType* d_Sbus;
    int                    sbus_stride;
};

// -----------------------------------------------------------------------------
// NrIterTimings — per-step timing accumulators (gpu_ms + wall_ms each)
//
// Field names deliberately match those in AcPfTimings and BatchTimings
// so callers can aggregate with:  outer_t.t_spmv += step.t_spmv;
// -----------------------------------------------------------------------------
struct NrIterTimings {
    TimingEntry t_spmv;
    TimingEntry t_fill_F;
    TimingEntry t_fill_J;
    TimingEntry t_first_factorize;   // populated only when first_factorize==true
    TimingEntry t_refactorize;       // populated only when first_factorize==false
    TimingEntry t_solve;
    TimingEntry t_update_V;
};

// -----------------------------------------------------------------------------
// nr_iter_step
//
// Executes one Newton-Raphson iteration on stream cs:
//   ①  SpMV:        d_Ibus = Ybus · d_V
//   ②  Fill F:      −[ΔP(pvpq), ΔQ(pq)]
//   ③  Fill J:      Jacobian numeric values from V, Ibus, Ybus
//   ④  Factorize:   full (first_factorize=true) or numeric-only refactorize
//   ⑤  Solve:       d_dx = J⁻¹ · d_F
//   ⑥  Update V:    Va (pvpq) then Vm (pq) in-place
//
// first_factorize=true  → dss.factorize()   (CUDSS_PHASE_FACTORIZATION)
// first_factorize=false → dss.refactorize() (CUDSS_PHASE_REFACTORIZATION)
//
// Both update kernels are launched on the same stream and are therefore
// serialised by CUDA without any explicit event.
// -----------------------------------------------------------------------------
inline void nr_iter_step(
    CuSpMV&          spmv,
    CudssContext&    dss,
    CudssDescriptor& dss_A,
    CudssDescriptor& dss_x,
    CudssDescriptor& dss_b,
    const NrIterBuffers& buf,
    int n_bus, int n_pvpq, int n_pq, int dim_J, int nnz_Y, int nnz_J,
    int actual_batch,
    cudaStream_t cs,
    CudaTimer&   timer,
    bool         first_factorize,
    NrIterTimings& t)
{
    // ①  SpMV: d_Ibus = Ybus · d_V
    timer.start();
    spmv.spmv();
    t.t_spmv += timer.stop_ms();

    // ②  Fill F: −[ΔP(pvpq), ΔQ(pq)]
    timer.start();
    fill_FP_kernel<<<(actual_batch * n_pvpq + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_pvpq,
        n_pvpq, n_bus, dim_J, actual_batch, buf.sbus_stride);
    fill_FQ_kernel<<<(actual_batch * n_pq + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_pq,
        n_pvpq, n_pq, n_bus, dim_J, actual_batch, buf.sbus_stride);
    t.t_fill_F += timer.stop_ms();

    // ③  Fill J values; notify cuDSS that the values pointer changed
    timer.start();
    fill_J_kernel<<<(actual_batch * nnz_Y + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_J_values, buf.d_V, buf.d_Ibus,
        buf.d_Ybus_outer, buf.d_Ybus_inner, buf.d_Ybus_values,
        buf.d_map_j11, buf.d_map_j12, buf.d_map_j21, buf.d_map_j22,
        n_bus, nnz_Y, nnz_J, actual_batch);
    dss_A.set_values(buf.d_J_values);
    t.t_fill_J += timer.stop_ms();

    // ④  Factorize / refactorize
    timer.start();
    if (first_factorize) dss.factorize(dss_A, dss_x, dss_b);
    else                 dss.refactorize(dss_A, dss_x, dss_b);
    if (first_factorize) t.t_first_factorize += timer.stop_ms();
    else                 t.t_refactorize     += timer.stop_ms();

    // ⑤  Solve: d_dx = J⁻¹ · d_F
    timer.start();
    dss.solve(dss_A, dss_x, dss_b);
    t.t_solve += timer.stop_ms();

    // ⑥  Update V in-place: Va first, then Vm.
    //     Both kernels run on cs — CUDA stream ordering serialises them without
    //     an explicit CPU-side event.
    timer.start();
    update_Va_kernel<<<(actual_batch * n_pvpq + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_pvpq,
        n_pvpq, n_bus, dim_J, actual_batch);
    update_Vm_kernel<<<(actual_batch * n_pq + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_pq,
        n_pvpq, n_pq, n_bus, dim_J, actual_batch);
    t.t_update_V += timer.stop_ms();
}

#endif  // NR_ITER_STEP_CUH