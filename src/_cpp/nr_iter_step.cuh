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

#include <limits>

static constexpr int BS = 256;   // block size for all NR kernels

// -----------------------------------------------------------------------------
// nr_grid_size
//
// ceil(total_threads / block), computed in 64-bit before narrowing to the
// unsigned int a CUDA <<<grid, block>>> launch expects. Every launch below
// computes total_threads as (batch or actual_batch) * <per-system stride>
// (n_p, n_q, nnz_Y, dim_J, n_slack, n_hvdc, n_ctrl, n_grp, n_share, …); doing
// that multiplication in plain `int` -- as every one of these call sites did
// before -- silently wraps once the product exceeds INT32_MAX, launching too
// few blocks and leaving the tail of the batch never computed (wrong results,
// no crash, no error). The corresponding kernel bodies (acpf_nr_kernels.cu)
// already widened their own tid/offset arithmetic to ptrdiff_t for the same
// reason; this is the launch-site half of that fix. Narrowing to unsigned int
// here is safe for any workload that actually fits in GPU memory -- CUDA's
// own gridDim.x hardware limit (2^31-1) is far above what real batch sizes
// reach before exhausting device memory first.
// -----------------------------------------------------------------------------
inline unsigned int nr_grid_size(long long total_threads, int block) {
    return static_cast<unsigned int>((total_threads + block - 1) / block);
}

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

    // ---- handle_disconnected_grid masking (per-chunk slice) -----------------
    // Identity-row entries (slot, J row, diag nnz pos) freeze the masked buses'
    // NR equations; masked-voltage entries (slot, bus) flag NaN reporting. All
    // null / counts 0 when the mode is off → the masking launches are skipped
    // and the feature-free / legacy path stays bit-identical.
    const int*             d_J_outer_mask = nullptr;   // [dim_J+1] shared J skeleton outer
    const int*             d_mask_slot    = nullptr;   // [n_mask_rows]
    const int*             d_mask_row     = nullptr;   // [n_mask_rows]
    const int*             d_mask_diag    = nullptr;   // [n_mask_rows]
    int                    n_mask_rows    = 0;
    const int*             d_maskv_slot   = nullptr;   // [n_mask_v]
    const int*             d_maskv_bus    = nullptr;   // [n_mask_v]
    int                    n_mask_v       = 0;

    // ---- NR step-scaling (MaxVoltageChange) -- inactive (alpha=1, no kernels
    // launched) unless enabled. Mirrors lightsim2grid's own
    // MaxVoltageChangeScalingPolicy: after solving J*dx=F, scale dx by
    // alpha<=1 so max|dtheta|<=max_dVa and max|dvm|<=max_dVm BEFORE applying
    // it anywhere (Va/Vm AND any extension state columns), matching
    // NRAlgo.tpp's apply_step(coeff * F). See LedgerData::
    // scaling_max_voltage_change's own doc for why this exists: an undamped
    // full Newton step can converge onto a different (sometimes spurious)
    // root than lightsim2grid's own damped trajectory when seeded far from
    // the solution.
    bool                   scaling_max_voltage_change = false;
    cuda_real_type         max_dVa = static_cast<cuda_real_type>(0.5);
    cuda_real_type         max_dVm = static_cast<cuda_real_type>(0.1);
    cuda_real_type*        d_scale_max_dtheta = nullptr;  // [actual_batch] scratch, zeroed each call
    cuda_real_type*        d_scale_max_dvm    = nullptr;  // [actual_batch] scratch, zeroed each call
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
        adjust_slack_mismatch_kernel<<<nr_grid_size((long long)batch * buf.n_slack, BS), BS, 0, cs>>>(
            buf.d_F, buf.d_slack_absorbed, buf.d_slack_prow, buf.d_slack_w,
            buf.n_slack, dim_J, batch);
    if (buf.n_hvdc > 0)
        hvdc_adjust_mismatch_kernel<<<nr_grid_size((long long)batch * buf.n_hvdc, BS), BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_hvdc_bus1, buf.d_hvdc_bus2, buf.d_hvdc_status,
            buf.d_hvdc_p0, buf.d_hvdc_k, buf.d_hvdc_lf1, buf.d_hvdc_lf2, buf.d_hvdc_r,
            buf.d_hvdc_pmax12, buf.d_hvdc_pmax21, buf.d_hvdc_prow1, buf.d_hvdc_prow2,
            buf.n_hvdc, n_bus, dim_J, batch);
    if (buf.n_vc_ctrl > 0) {
        vc_adjust_mismatch_kernel<<<nr_grid_size((long long)batch * buf.n_vc_ctrl, BS), BS, 0, cs>>>(
            buf.d_F, buf.d_vc_q, buf.d_vc_qrow, buf.n_vc_ctrl, dim_J, batch);
        vc_vrow_kernel<<<nr_grid_size((long long)batch * buf.n_vc_grp, BS), BS, 0, cs>>>(
            buf.d_F, buf.d_V, buf.d_vc_q, buf.d_vc_slope, buf.d_vc_reg_bus, buf.d_vc_vrow,
            buf.d_vc_grp_start, buf.d_vc_grp_count, buf.d_vc_vset,
            buf.n_vc_grp, buf.n_vc_ctrl, n_bus, dim_J, batch);
        if (buf.n_vc_share > 0)
            vc_share_kernel<<<nr_grid_size((long long)batch * buf.n_vc_share, BS), BS, 0, cs>>>(
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
        fill_slack_feature_kernel<<<nr_grid_size((long long)batch * buf.n_slack, BS), BS, 0, cs>>>(
            buf.d_J_values, buf.d_slack_feat_pos, buf.d_slack_w, buf.n_slack, nnz_J, batch);
    if (buf.n_hvdc > 0)
        hvdc_fill_feature_kernel<<<nr_grid_size((long long)batch * buf.n_hvdc, BS), BS, 0, cs>>>(
            buf.d_J_values, buf.d_V, buf.d_hvdc_bus1, buf.d_hvdc_bus2, buf.d_hvdc_status,
            buf.d_hvdc_p0, buf.d_hvdc_k, buf.d_hvdc_lf1, buf.d_hvdc_lf2,
            buf.d_hvdc_h11, buf.d_hvdc_h12, buf.d_hvdc_h21, buf.d_hvdc_h22,
            buf.n_hvdc, n_bus, nnz_J, batch);
    if (buf.n_vc_feat > 0)
        fill_slack_feature_kernel<<<nr_grid_size((long long)batch * buf.n_vc_feat, BS), BS, 0, cs>>>(
            buf.d_J_values, buf.d_vc_feat_pos, buf.d_vc_feat_val, buf.n_vc_feat, nnz_J, batch);
}

inline void nr_feature_update(const NrIterBuffers& buf, int dim_J, int batch, cudaStream_t cs)
{
    if (buf.slack_col >= 0)
        update_slack_absorbed_kernel<<<(batch + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_slack_absorbed, buf.d_dx, buf.slack_col, dim_J, batch);
    if (buf.n_vc_ctrl > 0)
        vc_apply_step_kernel<<<nr_grid_size((long long)batch * buf.n_vc_ctrl, BS), BS, 0, cs>>>(
            buf.d_vc_q, buf.d_dx, buf.d_vc_qcol, buf.n_vc_ctrl, dim_J, batch);
}

// -----------------------------------------------------------------------------
// handle_disconnected_grid launch helpers (no-ops when the mode is off).
//   nr_apply_bus_mask : freeze masked buses' J rows to identity + zero their F.
//                       Must run AFTER fill_F / fill_J + all feature stamps.
//   nr_mask_v_nan     : overwrite masked buses' voltages with NaN before store.
// -----------------------------------------------------------------------------
inline void nr_apply_bus_mask(const NrIterBuffers& buf,
                              int nnz_J, int dim_J, int batch, cudaStream_t cs)
{
    (void)batch;
    if (buf.n_mask_rows > 0)
        apply_bus_mask_kernel<<<(buf.n_mask_rows + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_J_values, buf.d_F, buf.d_mask_slot, buf.d_mask_row, buf.d_mask_diag,
            buf.d_J_outer_mask, nnz_J, dim_J, buf.n_mask_rows);
}

inline void nr_mask_v_nan(const NrIterBuffers& buf, int n_bus, cudaStream_t cs)
{
    if (buf.n_mask_v > 0) {
        const cuda_real_type nan_val = std::numeric_limits<cuda_real_type>::quiet_NaN();
        mask_V_nan_kernel<<<(buf.n_mask_v + BS - 1) / BS, BS, 0, cs>>>(
            buf.d_V, buf.d_maskv_slot, buf.d_maskv_bus, nan_val, n_bus, buf.n_mask_v);
    }
}

// -----------------------------------------------------------------------------
// nr_iter_step_fill_F
//
// Step ② alone: −[ΔP(pvpq), ΔQ(pq)] scattered into the ledger P/Q rows, at
// whatever d_V/d_Ibus/feature-state (d_slack_absorbed, d_vc_q) currently hold.
// Extracted so a caller that already has a valid d_Ibus (V unchanged since the
// last SpMV) can refresh d_F after updating feature state alone — e.g. the
// presolved-V fast path, after seeding slack_absorbed/vc_q via one linear
// solve without moving V (see nr_iter_step_prepare's fast-path note below).
// -----------------------------------------------------------------------------
inline void nr_iter_step_fill_F(
    const NrIterBuffers& buf,
    int n_bus, int dim_J,
    int actual_batch,
    cudaStream_t cs,
    CudaTimer&   timer,
    NrIterTimings& t)
{
    timer.start();
    fill_FP_kernel<<<nr_grid_size((long long)actual_batch * buf.n_p, BS), BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_p_buses, buf.d_p_rows,
        buf.n_p, n_bus, dim_J, actual_batch, buf.sbus_stride);
    fill_FQ_kernel<<<nr_grid_size((long long)actual_batch * buf.n_q, BS), BS, 0, cs>>>(
        buf.d_F, buf.d_V, buf.d_Ibus, buf.d_Sbus, buf.d_q_buses, buf.d_q_rows,
        buf.n_q, n_bus, dim_J, actual_batch, buf.sbus_stride);
    nr_feature_mismatch(buf, n_bus, dim_J, actual_batch, cs);
    t.t_fill_F += timer.stop_ms();
}

// -----------------------------------------------------------------------------
// nr_iter_step_prepare
//
// Steps ①-④ of one Newton-Raphson iteration on stream cs:
//   ①  SpMV:        d_Ibus = Ybus · d_V
//   ②  Fill F:      −[ΔP(pvpq), ΔQ(pq)]   (residual AT THE CURRENT d_V)
//   ③  Fill J:      Jacobian numeric values from V, Ibus, Ybus (AT THE CURRENT d_V)
//   ④  Factorize:   full (first_factorize=true) or numeric-only refactorize
//
// Leaves d_F holding the mismatch at the current d_V and dss_A holding the
// factorized J at that same d_V — both ready either for nr_iter_step_correct
// (the normal path) or for a caller that wants to stop here (the presolved-V
// fast path: validate ‖F‖, keep V exactly as supplied, never call solve/update).
//
// first_factorize=true  → dss.factorize()   (CUDSS_PHASE_FACTORIZATION)
// first_factorize=false → dss.refactorize() (CUDSS_PHASE_REFACTORIZATION)
//
// use_cudss=false skips BOTH dss_A.set_values() and factorize/refactorize --
// used by AcPfNrState's presolved_v fast path when this instance has no
// cuDSS context at all (base_case_only_ + ground-truth extension seeding, no
// debug override; see acpf_nr.cu). d_J_values itself is still populated
// (steps ①-③ always run), since direct_base_case_factors only ever needs the
// raw values, never a factorization of this particular context.
// -----------------------------------------------------------------------------
inline void nr_iter_step_prepare(
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
    NrIterTimings& t,
    bool         use_cudss = true)
{
    (void)n_pvpq; (void)n_pq;

    // ①  SpMV: d_Ibus = Ybus · d_V
    timer.start();
    spmv.spmv();
    t.t_spmv += timer.stop_ms();

    // ②  Fill F: −[ΔP, ΔQ] scattered into the ledger P/Q rows
    nr_iter_step_fill_F(buf, n_bus, dim_J, actual_batch, cs, timer, t);

    // ③  Fill J values; notify cuDSS that the values pointer changed (skipped
    //     when use_cudss=false -- no cuDSS context/descriptor exists to notify).
    //     When an additive feature (HVDC droop) is active, J must be zeroed first
    //     (the dS fill assigns; the droop slopes accumulate onto / beside it).
    timer.start();
    nr_feature_zero_J(buf, nnz_J, actual_batch, cs);
    fill_J_kernel<<<nr_grid_size((long long)actual_batch * nnz_Y, BS), BS, 0, cs>>>(
        buf.d_J_values, buf.d_V, buf.d_Ibus,
        buf.d_Ybus_outer, buf.d_Ybus_inner, buf.d_Ybus_values,
        buf.d_map_j11, buf.d_map_j12, buf.d_map_j21, buf.d_map_j22,
        n_bus, nnz_Y, nnz_J, actual_batch);
    nr_feature_fill_J(buf, n_bus, nnz_J, actual_batch, cs);
    if (use_cudss) dss_A.set_values(buf.d_J_values);
    t.t_fill_J += timer.stop_ms();

    // ④  Factorize / refactorize (skipped entirely when use_cudss=false)
    if (use_cudss) {
        timer.start();
        if (first_factorize) dss.factorize(dss_A, dss_x, dss_b);
        else                 dss.refactorize(dss_A, dss_x, dss_b);
        if (first_factorize) t.t_first_factorize += timer.stop_ms();
        else                 t.t_refactorize     += timer.stop_ms();
    }
}

// -----------------------------------------------------------------------------
// nr_iter_step_correct
//
// Steps ⑤-⑥ of one Newton-Raphson iteration on stream cs:
//   ⑤  Solve:       d_dx = J⁻¹ · d_F
//   ⑥  Update V:    Va (pvpq) then Vm (pq) in-place
//
// Requires nr_iter_step_prepare to have just been called (dss_A factorized,
// d_F holding the mismatch at the pre-update d_V).
//
// Both update kernels are launched on the same stream and are therefore
// serialised by CUDA without any explicit event.
// -----------------------------------------------------------------------------
inline void nr_iter_step_correct(
    CudssContext&    dss,
    CudssDescriptor& dss_A,
    CudssDescriptor& dss_x,
    CudssDescriptor& dss_b,
    const NrIterBuffers& buf,
    int n_bus, int dim_J,
    int actual_batch,
    cudaStream_t cs,
    CudaTimer&   timer,
    NrIterTimings& t)
{
    // ⑤  Solve: d_dx = J⁻¹ · d_F
    timer.start();
    dss.solve(dss_A, dss_x, dss_b);
    t.t_solve += timer.stop_ms();

    // ⑤b Step-scaling (MaxVoltageChange), if enabled -- rescale the WHOLE dx
    //     vector BEFORE applying it anywhere, so Va/Vm and any extension state
    //     move together, exactly like lightsim2grid's own apply_step(coeff*F).
    //     Folded into the same t.t_solve bucket (no new timing field).
    if (buf.scaling_max_voltage_change) {
        cudaMemsetAsync(buf.d_scale_max_dtheta, 0, actual_batch * sizeof(cuda_real_type), cs);
        cudaMemsetAsync(buf.d_scale_max_dvm,    0, actual_batch * sizeof(cuda_real_type), cs);
        const int total = buf.n_theta + buf.n_vm;
        reduce_step_norms_kernel<<<nr_grid_size((long long)actual_batch * total, BS), BS, 0, cs>>>(
            buf.d_dx, buf.d_theta_cols, buf.d_vm_cols, buf.n_theta, buf.n_vm,
            dim_J, actual_batch, buf.d_scale_max_dtheta, buf.d_scale_max_dvm);
        apply_step_scale_kernel<<<nr_grid_size((long long)actual_batch * dim_J, BS), BS, 0, cs>>>(
            buf.d_dx, buf.d_scale_max_dtheta, buf.d_scale_max_dvm,
            buf.max_dVa, buf.max_dVm, dim_J, actual_batch);
    }

    // ⑥  Update V in-place: Va first, then Vm.
    //     Both kernels run on cs — CUDA stream ordering serialises them without
    //     an explicit CPU-side event.
    timer.start();
    update_Va_kernel<<<nr_grid_size((long long)actual_batch * buf.n_theta, BS), BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_theta_buses, buf.d_theta_cols,
        buf.n_theta, n_bus, dim_J, actual_batch);
    update_Vm_kernel<<<nr_grid_size((long long)actual_batch * buf.n_vm, BS), BS, 0, cs>>>(
        buf.d_V, buf.d_dx, buf.d_vm_buses, buf.d_vm_cols,
        buf.n_vm, n_bus, dim_J, actual_batch);
    nr_feature_update(buf, dim_J, actual_batch, cs);
    t.t_update_V += timer.stop_ms();
}

// -----------------------------------------------------------------------------
// nr_iter_step
//
// Executes one full Newton-Raphson iteration: nr_iter_step_prepare() followed
// by nr_iter_step_correct(). Thin wrapper kept so existing call sites
// (acpf_nr.cu's normal loop, contingency/batch_pf_driver.cuh's per-chunk loop)
// are unchanged.
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
    nr_iter_step_prepare(spmv, dss, dss_A, dss_x, dss_b, buf,
                         n_bus, n_pvpq, n_pq, dim_J, nnz_Y, nnz_J,
                         actual_batch, cs, timer, first_factorize, t);
    nr_iter_step_correct(dss, dss_A, dss_x, dss_b, buf,
                         n_bus, dim_J, actual_batch, cs, timer, t);
}

#endif  // NR_ITER_STEP_CUH