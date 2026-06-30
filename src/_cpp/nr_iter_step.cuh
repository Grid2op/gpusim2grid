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
//   d_Ybus_outer, d_Ybus_inner, d_map_j*, ledger pair lists, d_Sbus.
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

    // NRLedger pair lists (shared single-system) — drive the augmented J fill_F
    // and update_V kernels. (bus, row/col) pairs in registration order; counts
    // may exceed n_pvpq/n_pq once extensions add equations/unknowns. Feature-free:
    //   p/theta == sorted(pvpq) @ rows/cols [0, n_pvpq);
    //   q/vm    == pq           @ rows/cols [n_pvpq, n_pvpq + n_pq).
    const int*             d_p_buses;
    const int*             d_p_rows;
    int                    n_p;
    const int*             d_q_buses;
    const int*             d_q_rows;
    int                    n_q;
    const int*             d_theta_buses;
    const int*             d_theta_cols;
    int                    n_theta;
    const int*             d_vm_buses;
    const int*             d_vm_cols;
    int                    n_vm;

    // Sbus pointer + stride control (see acpf_nr_kernels.cuh):
    //   sbus_stride = 0     → d_Sbus is a single-system (n_bus,) buffer
    //                         shared across all systems in the batch
    //   sbus_stride = n_bus → d_Sbus is a per-batch (actual_batch × n_bus)
    //                         layout indexed as d_Sbus[b * n_bus + bus]
    const cudaComplexType* d_Sbus;
    int                    sbus_stride;

    // ---- MultiSlack (distributed slack) — inactive when slack_col < 0 --------
    // d_slack_absorbed is the only MUTABLE per-batch-slot state ([actual_batch]);
    // the rest are shared single-system arrays of length n_slack. All nullptr /
    // slack_col=-1 when the extension is absent (kernels then skipped entirely).
    int                    slack_col   = -1;
    int                    n_slack     = 0;
    const int*             d_slack_prow     = nullptr;   // [n_slack] P row of each slack
    const cuda_real_type*  d_slack_w        = nullptr;   // [n_slack] slack weight
    const int*             d_slack_feat_pos = nullptr;   // [n_slack] J pos of (p_row, slack_col)
    cuda_real_type*        d_slack_absorbed = nullptr;   // [actual_batch] running state

    // ---- HVDC angle-droop — inactive when n_hvdc == 0 -----------------------
    // When active the J value array is zeroed before each fill_J (the droop
    // slopes accumulate onto / beside the dS entries), hence zero_J_before_fill.
    int                    n_hvdc      = 0;
    bool                   zero_J_before_fill = false;
    const int*             d_hvdc_bus1  = nullptr;
    const int*             d_hvdc_bus2  = nullptr;
    const int*             d_hvdc_status= nullptr;
    const cuda_real_type*  d_hvdc_p0    = nullptr;
    const cuda_real_type*  d_hvdc_k     = nullptr;
    const cuda_real_type*  d_hvdc_lf1   = nullptr;
    const cuda_real_type*  d_hvdc_lf2   = nullptr;
    const cuda_real_type*  d_hvdc_r     = nullptr;
    const cuda_real_type*  d_hvdc_pmax12= nullptr;
    const cuda_real_type*  d_hvdc_pmax21= nullptr;
    const int*             d_hvdc_prow1 = nullptr;
    const int*             d_hvdc_prow2 = nullptr;
    const int*             d_hvdc_h11   = nullptr;
    const int*             d_hvdc_h12   = nullptr;
    const int*             d_hvdc_h21   = nullptr;
    const int*             d_hvdc_h22   = nullptr;

    // ---- VoltageControl (remote gen + SVC) — inactive when n_vc_ctrl == 0 ----
    int                    n_vc_ctrl   = 0;
    int                    n_vc_grp    = 0;
    int                    n_vc_share  = 0;
    int                    n_vc_feat   = 0;
    const int*             d_vc_qrow      = nullptr;
    const int*             d_vc_qcol      = nullptr;
    const cuda_real_type*  d_vc_slope     = nullptr;
    const int*             d_vc_reg_bus   = nullptr;
    const int*             d_vc_vrow      = nullptr;
    const int*             d_vc_grp_start = nullptr;
    const int*             d_vc_grp_count = nullptr;
    const cuda_real_type*  d_vc_vset      = nullptr;
    const int*             d_vc_sh_row    = nullptr;
    const int*             d_vc_sh_first  = nullptr;
    const int*             d_vc_sh_other  = nullptr;
    const cuda_real_type*  d_vc_sh_wfirst = nullptr;
    const cuda_real_type*  d_vc_sh_wother = nullptr;
    const int*             d_vc_feat_pos  = nullptr;
    const cuda_real_type*  d_vc_feat_val  = nullptr;
    cuda_real_type*        d_vc_q         = nullptr;   // [actual_batch * n_vc_ctrl] running state
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
// Augmented-feature launch helpers — shared by the single-system nr_iter_step
// and the batched run_nr_loop / post-loop residual, so the extension wiring
// lives in ONE place. Each is a no-op when its extension is inactive.
//   nr_feature_mismatch : run AFTER fill_FP/FQ (MultiSlack/HVDC/VC mismatch +
//                         VC bordered custom rows).
//   nr_feature_zero_J   : zero J before the dS fill_J when an additive feature
//                         (HVDC) is active.
//   nr_feature_fill_J   : run AFTER fill_J (MultiSlack/HVDC/VC feature entries).
//   nr_feature_update   : run AFTER update_Va/Vm (slack_absorbed + VC q states).
// -----------------------------------------------------------------------------
inline void nr_feature_mismatch(const NrIterBuffers& buf,
                                int n_bus, int dim_J, int batch, cudaStream_t cs)
{
    if (buf.slack_col >= 0)
        adjust_slack_mismatch_kernel<<<(batch * buf.n_slack + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_slack_absorbed, buf.d_slack_prow, buf.d_slack_w,
            buf.n_slack, dim_J, batch);
    if (buf.n_hvdc > 0)
        hvdc_adjust_mismatch_kernel<<<(batch * buf.n_hvdc + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_hvdc_bus1, buf.d_hvdc_bus2, buf.d_hvdc_status,
            buf.d_hvdc_p0, buf.d_hvdc_k, buf.d_hvdc_lf1, buf.d_hvdc_lf2, buf.d_hvdc_r,
            buf.d_hvdc_pmax12, buf.d_hvdc_pmax21, buf.d_hvdc_prow1, buf.d_hvdc_prow2,
            buf.n_hvdc, n_bus, dim_J, batch);
    if (buf.n_vc_ctrl > 0) {
        vc_adjust_mismatch_kernel<<<(batch * buf.n_vc_ctrl + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_vc_q, buf.d_vc_qrow, buf.n_vc_ctrl, dim_J, batch);
        vc_vrow_kernel<<<(batch * buf.n_vc_grp + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_vc_q, buf.d_vc_slope, buf.d_vc_reg_bus, buf.d_vc_vrow,
            buf.d_vc_grp_start, buf.d_vc_grp_count, buf.d_vc_vset,
            buf.n_vc_grp, buf.n_vc_ctrl, n_bus, dim_J, batch);
        if (buf.n_vc_share > 0)
            vc_share_kernel<<<(batch * buf.n_vc_share + BS - 1) / BS, BS, 0, cs>>>(
                buf.d_F, buf.d_vc_q, buf.d_vc_sh_row, buf.d_vc_sh_first, buf.d_vc_sh_other,
                buf.d_vc_sh_wfirst, buf.d_vc_sh_wother, buf.n_vc_share, buf.n_vc_ctrl,
                dim_J, batch);
    }
}

inline void nr_feature_zero_J(const NrIterBuffers& buf, int nnz_J, int batch, cudaStream_t cs)
{
    if (buf.zero_J_before_fill)
        cudaMemsetAsync(buf.d_J_values, 0,
                        static_cast<size_t>(batch) * nnz_J * sizeof(cuda_real_type), cs);
}

inline void nr_feature_fill_J(const NrIterBuffers& buf,
                              int n_bus, int nnz_J, int batch, cudaStream_t cs)
{
    if (buf.slack_col >= 0)
        fill_slack_feature_kernel<<<(batch * buf.n_slack + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_J_values, buf.d_slack_feat_pos, buf.d_slack_w, buf.n_slack, nnz_J, batch);
    if (buf.n_hvdc > 0)
        hvdc_fill_feature_kernel<<<(batch * buf.n_hvdc + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_J_values, buf.d_V, buf.d_hvdc_bus1, buf.d_hvdc_bus2, buf.d_hvdc_status,
            buf.d_hvdc_p0, buf.d_hvdc_k, buf.d_hvdc_lf1, buf.d_hvdc_lf2,
            buf.d_hvdc_h11, buf.d_hvdc_h12, buf.d_hvdc_h21, buf.d_hvdc_h22,
            buf.n_hvdc, n_bus, nnz_J, batch);
    if (buf.n_vc_feat > 0)
        fill_slack_feature_kernel<<<(batch * buf.n_vc_feat + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_J_values, buf.d_vc_feat_pos, buf.d_vc_feat_val, buf.n_vc_feat, nnz_J, batch);
}

inline void nr_feature_update(const NrIterBuffers& buf, int dim_J, int batch, cudaStream_t cs)
{
    if (buf.slack_col >= 0)
        update_slack_absorbed_kernel<<<(batch + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_slack_absorbed, buf.d_dx, buf.slack_col, dim_J, batch);
    if (buf.n_vc_ctrl > 0)
        vc_apply_step_kernel<<<(batch * buf.n_vc_ctrl + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_vc_q, buf.d_dx, buf.d_vc_qcol, buf.n_vc_ctrl, dim_J, batch);
}

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

    // ②  Fill F: −[ΔP, ΔQ] scattered into the ledger P/Q rows
    timer.start();
    fill_FP_kernel<<<(actual_batch * buf.n_p + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_p_buses, buf.d_p_rows,
        buf.n_p, n_bus, dim_J, actual_batch, buf.sbus_stride);
    fill_FQ_kernel<<<(actual_batch * buf.n_q + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_q_buses, buf.d_q_rows,
        buf.n_q, n_bus, dim_J, actual_batch, buf.sbus_stride);
    nr_feature_mismatch(buf, n_bus, dim_J, actual_batch, cs);
    t.t_fill_F += timer.stop_ms();

    // ③  Fill J values; notify cuDSS that the values pointer changed.
    //     When an additive feature (HVDC droop) is active, J must be zeroed first
    //     (the dS fill assigns; the droop slopes accumulate onto / beside it).
    timer.start();
    nr_feature_zero_J(buf, nnz_J, actual_batch, cs);
    fill_J_kernel<<<(actual_batch * nnz_Y + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_J_values, buf.d_V, buf.d_Ibus,
        buf.d_Ybus_outer, buf.d_Ybus_inner, buf.d_Ybus_values,
        buf.d_map_j11, buf.d_map_j12, buf.d_map_j21, buf.d_map_j22,
        n_bus, nnz_Y, nnz_J, actual_batch);
    nr_feature_fill_J(buf, n_bus, nnz_J, actual_batch, cs);
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
    update_Va_kernel<<<(actual_batch * buf.n_theta + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_theta_buses, buf.d_theta_cols,
        buf.n_theta, n_bus, dim_J, actual_batch);
    update_Vm_kernel<<<(actual_batch * buf.n_vm + BS - 1) / BS, BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_vm_buses, buf.d_vm_cols,
        buf.n_vm, n_bus, dim_J, actual_batch);
    nr_feature_update(buf, dim_J, actual_batch, cs);
    t.t_update_V += timer.stop_ms();
}

#endif  // NR_ITER_STEP_CUH