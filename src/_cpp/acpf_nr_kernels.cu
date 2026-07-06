// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// acpf_nr_kernels.cu
// =============================================================================

#include "acpf_nr_kernels.cuh"

static constexpr int BS = 256;   // block size for all kernels in this file

// Portable real atomic add (the device default arch may predate hardware double
// atomicAdd). Native atomicAdd for float and for double on SM >= 6.0; a 64-bit
// CAS fallback for double on older targets. Used by the HVDC droop kernels,
// whose contributions can overlap onto shared J positions / mismatch rows.
__device__ __forceinline__ void atomic_add_real(float* addr, float val) {
    atomicAdd(addr, val);
}
__device__ __forceinline__ void atomic_add_real(double* addr, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(addr, val);
#else
    unsigned long long int* p = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// =============================================================================
// compute_branch_flows_kernel
// =============================================================================
__global__ void compute_branch_flows_kernel(
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ d_branch_from,
    const int*             __restrict__ d_branch_to,
    const cudaComplexType* __restrict__ d_yff,
    const cudaComplexType* __restrict__ d_yft,
    const cudaComplexType* __restrict__ d_ytf,
    const cudaComplexType* __restrict__ d_ytt,
    const cuda_real_type*  __restrict__ d_base_current_A,
          cuda_real_type*  __restrict__ d_or_amps,
          cuda_real_type*  __restrict__ d_ex_amps,
    int n_bus,
    int n_branches,
    int c_start,
    int actual_batch,
    const int* __restrict__ d_result_map)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_branches;   // contingency index in batch
    const int l   = tid % n_branches;   // branch index
    if (b >= actual_batch) return;

    const cudaComplexType Vi = d_V[b * n_bus + d_branch_from[l]];
    const cudaComplexType Vj = d_V[b * n_bus + d_branch_to[l]];

    // I_or = yff * Vi + yft * Vj  (origin / from-bus terminal current)
    const cudaComplexType I_or = CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_yff[l], Vi),
        CudaFunHelper::my_cuCmul(d_yft[l], Vj));

    // I_ex = ytf * Vi + ytt * Vj  (extremity / to-bus terminal current)
    const cudaComplexType I_ex = CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_ytf[l], Vi),
        CudaFunHelper::my_cuCmul(d_ytt[l], Vj));

    // Map the chunk-relative slot to its original result index (identity when
    // d_result_map is null, e.g. the full-batch session call or injection).
    const int out_c   = d_result_map ? d_result_map[c_start + b] : (c_start + b);
    const int out_idx = out_c * n_branches + l;
    d_or_amps[out_idx] = CudaFunHelper::my_cuCabs(I_or) * d_base_current_A[l];
    d_ex_amps[out_idx] = CudaFunHelper::my_cuCabs(I_ex) * d_base_current_A[l];
}

// =============================================================================
// scatter_V_results_kernel
//
// Copies a chunk's converged voltages from the compact batch buffer into the
// full-size result buffer at each system's ORIGINAL contingency index.  Used
// when disconnected contingencies have been compacted out of the batch, so the
// active-slot index differs from the result index.  One thread per (slot, bus).
// =============================================================================
__global__ void scatter_V_results_kernel(
          cudaComplexType* __restrict__ d_V_results,
    const cudaComplexType* __restrict__ d_V_batch,
    const int*             __restrict__ d_result_map,
    int c_start,
    int n_bus,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = actual_batch * n_bus;
    if (tid >= total) return;

    const int local_c = tid / n_bus;   // active-slot index in this chunk
    const int bus     = tid % n_bus;
    const int out_c   = d_result_map[c_start + local_c];   // original index
    d_V_results[out_c * n_bus + bus] = d_V_batch[local_c * n_bus + bus];
}

// =============================================================================
// apply_bus_mask_kernel
// =============================================================================
__global__ void apply_bus_mask_kernel(
          cuda_real_type* __restrict__ d_J_values,
          cuda_real_type* __restrict__ d_F,
    const int*           __restrict__ d_mask_slot,
    const int*           __restrict__ d_mask_row,
    const int*           __restrict__ d_mask_diag,
    const int*           __restrict__ d_J_outer,
    int nnz_J,
    int dim_J,
    int n_entries)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;

    const int slot = d_mask_slot[tid];
    const int row  = d_mask_row[tid];
    const int diag = d_mask_diag[tid];

    cuda_real_type* J = d_J_values + static_cast<ptrdiff_t>(slot) * nnz_J;
    const int row_beg = d_J_outer[row];
    const int row_end = d_J_outer[row + 1];
    for (int k = row_beg; k < row_end; ++k) J[k] = cuda_real_type(0);
    if (diag >= 0) J[diag] = cuda_real_type(1);

    d_F[static_cast<ptrdiff_t>(slot) * dim_J + row] = cuda_real_type(0);
}

// =============================================================================
// mask_V_nan_kernel
// =============================================================================
__global__ void mask_V_nan_kernel(
          cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ d_maskv_slot,
    const int*             __restrict__ d_maskv_bus,
    cuda_real_type nan_val,
    int n_bus,
    int n_entries)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;
    const int slot = d_maskv_slot[tid];
    const int bus  = d_maskv_bus[tid];
    d_V[static_cast<ptrdiff_t>(slot) * n_bus + bus] =
        CudaFunHelper::my_make_cuComplex(nan_val, nan_val);
}

// =============================================================================
// zero_branch_flows_kernel
// =============================================================================
__global__ void zero_branch_flows_kernel(
          cuda_real_type* __restrict__ d_or_amps,
          cuda_real_type* __restrict__ d_ex_amps,
    const int*           __restrict__ d_zero_indices,
    int n_entries)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    const int idx = d_zero_indices[i];
    d_or_amps[idx] = cuda_real_type(0);
    d_ex_amps[idx] = cuda_real_type(0);
}

// =============================================================================
// apply_contingencies_kernel
// =============================================================================
__global__ void apply_contingencies_kernel(
          cudaComplexType* __restrict__ value_ptr_batch,
    const int*             __restrict__ ctg_id,
    const int*             __restrict__ k_idx,
    const cuda_real_type*  __restrict__ delta_re,
    const cuda_real_type*  __restrict__ delta_im,
    int nnz_Y,
    int n_updates)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_updates) return;

    // Offset to the start of this contingency's value block.
    // No atomicAdd needed: each (ctg_id, k) pair is unique by construction.
    const int base = ctg_id[i] * nnz_Y;
    cudaComplexType& entry = value_ptr_batch[base + k_idx[i]];
    entry = CudaFunHelper::my_cuCsub(
        entry,
        CudaFunHelper::my_make_cuComplex(delta_re[i], delta_im[i]));
}

// =============================================================================
// fill_FP_kernel
// =============================================================================
__global__ void fill_FP_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ p_buses,
    const int*             __restrict__ p_rows,
    int n_p,
    int n_bus,
    int dim_J,
    int actual_batch,
    int sbus_stride)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_p;   // contingency index
    const int k   = tid % n_p;   // P-equation slot
    if (b >= actual_batch) return;

    const int bus = p_buses[k];
    const int row = p_rows[k];
    const cudaComplexType S_calc = CudaFunHelper::my_cuCmul(
        d_V   [b * n_bus + bus],
        CudaFunHelper::my_cuConj(d_Ibus[b * n_bus + bus]));
    const cuda_real_type dP =
        CudaFunHelper::my_cuCreal(S_calc) -
        CudaFunHelper::my_cuCreal(d_Sbus[b * sbus_stride + bus]);
    d_F[b * dim_J + row] = -dP;
}

// =============================================================================
// fill_FQ_kernel
// =============================================================================
__global__ void fill_FQ_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ q_buses,
    const int*             __restrict__ q_rows,
    int n_q,
    int n_bus,
    int dim_J,
    int actual_batch,
    int sbus_stride)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_q;
    const int k   = tid % n_q;
    if (b >= actual_batch) return;

    const int bus = q_buses[k];
    const int row = q_rows[k];
    const cudaComplexType S_calc = CudaFunHelper::my_cuCmul(
        d_V   [b * n_bus + bus],
        CudaFunHelper::my_cuConj(d_Ibus[b * n_bus + bus]));
    const cuda_real_type dQ =
        CudaFunHelper::my_cuCimag(S_calc) -
        CudaFunHelper::my_cuCimag(d_Sbus[b * sbus_stride + bus]);
    d_F[b * dim_J + row] = -dQ;
}

// =============================================================================
// fill_J_kernel
// =============================================================================
__global__ void fill_J_kernel(
          cuda_real_type*  __restrict__ d_J_values,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const int*             __restrict__ d_Ybus_outer,        // single-system
    const int*             __restrict__ d_Ybus_inner,        // single-system
    const cudaComplexType* __restrict__ d_Ybus_values_batch, // [actual_batch * nnz_Y]
    const int*             __restrict__ d_map_j11,           // single-system
    const int*             __restrict__ d_map_j12,
    const int*             __restrict__ d_map_j21,
    const int*             __restrict__ d_map_j22,
    int n_bus,
    int nnz_Y,
    int nnz_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / nnz_Y;   // contingency index
    const int k   = tid % nnz_Y;   // nnz slot within one system
    if (b >= actual_batch) return;

    // Row recovery: binary search on the single-system outer array using k.
    // Row recovery uses the single-system outer array — no b offset needed.
    int lo = 0, hi = n_bus;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (d_Ybus_outer[mid + 1] <= k) lo = mid + 1;
        else                            hi = mid;
    }
    const int i = lo;                  // local row within one system
    const int j = d_Ybus_inner[k];    // local col within one system

    const cudaComplexType V_i  = d_V[b * n_bus + i];
    const cudaComplexType V_j  = d_V[b * n_bus + j];
    const cudaComplexType Y_ij = d_Ybus_values_batch[b * nnz_Y + k];  // patched value
    const cudaComplexType I_i  = d_Ibus[b * n_bus + i];

    // ---- dS/dVa ---------------------------------------------------------------
    cudaComplexType dSdVa;
    if (i == j) {
        cudaComplexType tmp = CudaFunHelper::my_cuConj(
            CudaFunHelper::my_cuCsub(I_i, CudaFunHelper::my_cuCmul(Y_ij, V_i)));
        tmp = CudaFunHelper::my_cuCmul(V_i, tmp);
        // multiply by j: (a + jb) * j = -b + ja
        dSdVa = CudaFunHelper::my_make_cuComplex(
            -CudaFunHelper::my_cuCimag(tmp),
             CudaFunHelper::my_cuCreal(tmp));
    } else {
        cudaComplexType tmp = CudaFunHelper::my_cuConj(
            CudaFunHelper::my_cuCmul(Y_ij, V_j));
        tmp = CudaFunHelper::my_cuCmul(V_i, tmp);
        // multiply by -j: (a + jb) * (-j) = b - ja
        dSdVa = CudaFunHelper::my_make_cuComplex(
             CudaFunHelper::my_cuCimag(tmp),
            -CudaFunHelper::my_cuCreal(tmp));
    }

    // ---- dS/dVm ------------------------------------------------------------
    cudaComplexType dSdVm;
    if (i == j) {
        const cuda_real_type absVi = CudaFunHelper::my_cuCabs(V_i);
        const cudaComplexType Vnorm_i = CudaFunHelper::my_make_cuComplex(
            CudaFunHelper::my_cuCreal(V_i) / absVi,
            CudaFunHelper::my_cuCimag(V_i) / absVi);
        const cudaComplexType t1 = CudaFunHelper::my_cuCmul(
            V_i, CudaFunHelper::my_cuConj(CudaFunHelper::my_cuCmul(Y_ij, Vnorm_i)));
        const cudaComplexType t2 = CudaFunHelper::my_cuCmul(
            CudaFunHelper::my_cuConj(I_i), Vnorm_i);
        dSdVm = CudaFunHelper::my_cuCadd(t1, t2);
    } else {
        const cuda_real_type absVj = CudaFunHelper::my_cuCabs(V_j);
        const cudaComplexType Vnorm_j = CudaFunHelper::my_make_cuComplex(
            CudaFunHelper::my_cuCreal(V_j) / absVj,
            CudaFunHelper::my_cuCimag(V_j) / absVj);
        dSdVm = CudaFunHelper::my_cuCmul(
            V_i, CudaFunHelper::my_cuConj(CudaFunHelper::my_cuCmul(Y_ij, Vnorm_j)));
    }

    // ---- Scatter into d_J_values (batch-offset by b * nnz_J) --------------
    const int J_base = b * nnz_J;
    if (d_map_j11[k] >= 0) d_J_values[J_base + d_map_j11[k]] = CudaFunHelper::my_cuCreal(dSdVa);
    if (d_map_j12[k] >= 0) d_J_values[J_base + d_map_j12[k]] = CudaFunHelper::my_cuCreal(dSdVm);
    if (d_map_j21[k] >= 0) d_J_values[J_base + d_map_j21[k]] = CudaFunHelper::my_cuCimag(dSdVa);
    if (d_map_j22[k] >= 0) d_J_values[J_base + d_map_j22[k]] = CudaFunHelper::my_cuCimag(dSdVm);
}

// =============================================================================
// update_Va_kernel
// =============================================================================
__global__ void update_Va_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ theta_buses,
    const int*             __restrict__ theta_cols,
    int n_theta,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_theta;
    const int k   = tid % n_theta;
    if (b >= actual_batch) return;

    const int bus = theta_buses[k];
    const int col = theta_cols[k];
    const cudaComplexType V = d_V[b * n_bus + bus];
    const cuda_real_type vm = CudaFunHelper::my_cuCabs(V);
    const cuda_real_type va = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V),
        CudaFunHelper::my_cuCreal(V)) + d_dx[b * dim_J + col];

    d_V[b * n_bus + bus] = CudaFunHelper::my_make_cuComplex(
        vm * CudaFunHelper::my_cos(va),
        vm * CudaFunHelper::my_sin(va));
}

// =============================================================================
// update_Vm_kernel
// =============================================================================
__global__ void update_Vm_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ vm_buses,
    const int*             __restrict__ vm_cols,
    int n_vm,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_vm;
    const int k   = tid % n_vm;
    if (b >= actual_batch) return;

    const int bus = vm_buses[k];
    const int col = vm_cols[k];
    const cudaComplexType V = d_V[b * n_bus + bus];
    // Vm_new = Vm_old + dx[vm col];  Va unchanged (already updated)
    const cuda_real_type vm = CudaFunHelper::my_cuCabs(V) + d_dx[b * dim_J + col];
    const cuda_real_type va = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V),
        CudaFunHelper::my_cuCreal(V));

    d_V[b * n_bus + bus] = CudaFunHelper::my_make_cuComplex(
        vm * CudaFunHelper::my_cos(va),
        vm * CudaFunHelper::my_sin(va));
}

// =============================================================================
// MultiSlack kernels (Phase 2)
// =============================================================================
__global__ void adjust_slack_mismatch_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_slack_absorbed,
    const int*            __restrict__ d_slack_prow,
    const cuda_real_type* __restrict__ d_slack_w,
    int n_slack,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_slack;
    const int k   = tid % n_slack;
    if (b >= actual_batch) return;
    // mis += sa·weight  ⇒  residual d_F = −real(mis) gains −sa·weight
    d_F[b * dim_J + d_slack_prow[k]] -= d_slack_absorbed[b] * d_slack_w[k];
}

__global__ void fill_slack_feature_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const int*            __restrict__ d_slack_feat_pos,
    const cuda_real_type* __restrict__ d_slack_w,
    int n_slack,
    int nnz_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_slack;
    const int k   = tid % n_slack;
    if (b >= actual_batch) return;
    d_J_values[b * nnz_J + d_slack_feat_pos[k]] = d_slack_w[k];
}

__global__ void init_slack_absorbed_kernel(
          cuda_real_type*  __restrict__ d_slack_absorbed,
    const cudaComplexType* __restrict__ d_Sbus,
    int sbus_stride,
    int n_bus,
    int actual_batch)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= actual_batch) return;
    const cudaComplexType* sb = d_Sbus + static_cast<size_t>(b) * sbus_stride;
    cuda_real_type s = static_cast<cuda_real_type>(0.);
    for (int i = 0; i < n_bus; ++i) s += CudaFunHelper::my_cuCreal(sb[i]);
    d_slack_absorbed[b] = s;
}

__global__ void update_slack_absorbed_kernel(
          cuda_real_type* __restrict__ d_slack_absorbed,
    const cuda_real_type* __restrict__ d_dx,
    int slack_col,
    int dim_J,
    int actual_batch)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= actual_batch) return;
    d_slack_absorbed[b] += d_dx[b * dim_J + slack_col];
}

// =============================================================================
// HVDC angle-droop kernels (Phase 3)
// =============================================================================

// Active power received by the non-controller side (HvdcDroopSolverData::recv_pu)
__device__ __forceinline__ cuda_real_type hvdc_recv_pu(
    cuda_real_type p_ctrl_abs, bool side1_ctrl,
    cuda_real_type lf1, cuda_real_type lf2, cuda_real_type r)
{
    const cuda_real_type lf_ctrl = side1_ctrl ? lf1 : lf2;
    const cuda_real_type lf_recv = side1_ctrl ? lf2 : lf1;
    const cuda_real_type line_in = (static_cast<cuda_real_type>(1.) - lf_ctrl) * p_ctrl_abs;
    return (static_cast<cuda_real_type>(1.) - lf_recv) * (line_in - r * line_in * line_in);
}

// The two active flows leaving the AC buses into the HVDC (HvdcDroopSolverData::flows_pu)
__device__ __forceinline__ void hvdc_flows_pu(
    int st, cuda_real_type raw,
    cuda_real_type lf1, cuda_real_type lf2, cuda_real_type r,
    cuda_real_type pmax12, cuda_real_type pmax21,
    cuda_real_type& p1_flow, cuda_real_type& p2_flow)
{
    if (st == 0) {
        if (raw >= static_cast<cuda_real_type>(0.)) {
            p1_flow =  raw;
            p2_flow = -hvdc_recv_pu(raw, true, lf1, lf2, r);
        } else {
            p1_flow = -hvdc_recv_pu(-raw, false, lf1, lf2, r);
            p2_flow = -raw;
        }
    } else if (st > 0) {
        p1_flow =  pmax12;
        p2_flow = -hvdc_recv_pu(pmax12, true, lf1, lf2, r);
    } else {
        p1_flow = -hvdc_recv_pu(pmax21, false, lf1, lf2, r);
        p2_flow =  pmax21;
    }
}

__global__ void hvdc_adjust_mismatch_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ bus1,
    const int*             __restrict__ bus2,
    const int*             __restrict__ status,
    const cuda_real_type*  __restrict__ p0,
    const cuda_real_type*  __restrict__ k,
    const cuda_real_type*  __restrict__ lf1,
    const cuda_real_type*  __restrict__ lf2,
    const cuda_real_type*  __restrict__ r,
    const cuda_real_type*  __restrict__ pmax12,
    const cuda_real_type*  __restrict__ pmax21,
    const int*             __restrict__ prow1,
    const int*             __restrict__ prow2,
    int n_hvdc,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_hvdc;
    const int e   = tid % n_hvdc;
    if (b >= actual_batch) return;

    const cudaComplexType V1 = d_V[b * n_bus + bus1[e]];
    const cudaComplexType V2 = d_V[b * n_bus + bus2[e]];
    const cuda_real_type th1 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
    const cuda_real_type th2 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
    const cuda_real_type raw = p0[e] + k[e] * (th1 - th2);

    cuda_real_type p1_flow, p2_flow;
    hvdc_flows_pu(status[e], raw, lf1[e], lf2[e], r[e], pmax12[e], pmax21[e],
                  p1_flow, p2_flow);

    // mis(bus) += p_flow  ⇒  residual d_F[p_row] -= p_flow
    if (prow1[e] >= 0) atomic_add_real(&d_F[b * dim_J + prow1[e]], -p1_flow);
    if (prow2[e] >= 0) atomic_add_real(&d_F[b * dim_J + prow2[e]], -p2_flow);
}

__global__ void hvdc_fill_feature_kernel(
          cuda_real_type*  __restrict__ d_J_values,
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ bus1,
    const int*             __restrict__ bus2,
    const int*             __restrict__ status,
    const cuda_real_type*  __restrict__ p0,
    const cuda_real_type*  __restrict__ k,
    const cuda_real_type*  __restrict__ lf1,
    const cuda_real_type*  __restrict__ lf2,
    const int*             __restrict__ h11,
    const int*             __restrict__ h12,
    const int*             __restrict__ h21,
    const int*             __restrict__ h22,
    int n_hvdc,
    int n_bus,
    int nnz_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_hvdc;
    const int e   = tid % n_hvdc;
    if (b >= actual_batch) return;
    if (status[e] != 0) return;   // saturated: constant injection, zero slopes

    const cudaComplexType V1 = d_V[b * n_bus + bus1[e]];
    const cudaComplexType V2 = d_V[b * n_bus + bus2[e]];
    const cuda_real_type th1 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
    const cuda_real_type th2 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
    const cuda_real_type raw = p0[e] + k[e] * (th1 - th2);
    const cuda_real_type loss_mult =
        (static_cast<cuda_real_type>(1.) - lf1[e]) * (static_cast<cuda_real_type>(1.) - lf2[e]);
    // dp1 = dp_side1/dtheta1, dp2 = dp_side2/dtheta1; d/dtheta2 = -d/dtheta1
    const cuda_real_type dp1 = (raw >= static_cast<cuda_real_type>(0.)) ? k[e] : k[e] * loss_mult;
    const cuda_real_type dp2 = (raw <  static_cast<cuda_real_type>(0.)) ? -k[e] : -k[e] * loss_mult;

    if (h11[e] >= 0) atomic_add_real(&d_J_values[b * nnz_J + h11[e]],  dp1);
    if (h12[e] >= 0) atomic_add_real(&d_J_values[b * nnz_J + h12[e]], -dp1);
    if (h21[e] >= 0) atomic_add_real(&d_J_values[b * nnz_J + h21[e]],  dp2);
    if (h22[e] >= 0) atomic_add_real(&d_J_values[b * nnz_J + h22[e]], -dp2);
}

// =============================================================================
// VoltageControl kernels (Phase 4)
// =============================================================================
__global__ void vc_adjust_mismatch_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,
    const int*            __restrict__ d_vc_qrow,
    int n_ctrl,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_ctrl;
    const int j   = tid % n_ctrl;
    if (b >= actual_batch) return;
    // mis(c.bus) -= i·Q_c  ⇒  residual d_F[q_row] += Q_c
    atomic_add_real(&d_F[b * dim_J + d_vc_qrow[j]], d_vc_q[b * n_ctrl + j]);
}

__global__ void vc_vrow_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_vc_q,
    const cuda_real_type*  __restrict__ d_vc_slope,
    const int*             __restrict__ d_vc_reg_bus,
    const int*             __restrict__ d_vc_vrow,
    const int*             __restrict__ d_vc_grp_start,
    const int*             __restrict__ d_vc_grp_count,
    const cuda_real_type*  __restrict__ d_vc_vset,
    int n_grp,
    int n_ctrl,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_grp;
    const int g   = tid % n_grp;
    if (b >= actual_batch) return;

    const cudaComplexType Vr = d_V[b * n_bus + d_vc_reg_bus[g]];
    const cuda_real_type vm  = CudaFunHelper::my_cuCabs(Vr);
    cuda_real_type slope_term = static_cast<cuda_real_type>(0.);
    const int first = d_vc_grp_start[g], cnt = d_vc_grp_count[g];
    for (int off = 0; off < cnt; ++off) {
        const int j = first + off;
        slope_term += d_vc_slope[j] * d_vc_q[b * n_ctrl + j];
    }
    // F_v = Vm(reg) + Σ s_c·Q_c − v_set ;  residual d_F = −F_v (custom row: assign)
    d_F[b * dim_J + d_vc_vrow[g]] = -(vm + slope_term - d_vc_vset[g]);
}

__global__ void vc_share_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,
    const int*            __restrict__ d_sh_row,
    const int*            __restrict__ d_sh_first,
    const int*            __restrict__ d_sh_other,
    const cuda_real_type* __restrict__ d_sh_wfirst,
    const cuda_real_type* __restrict__ d_sh_wother,
    int n_share,
    int n_ctrl,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_share;
    const int s   = tid % n_share;
    if (b >= actual_batch) return;
    const cuda_real_type q_first = d_vc_q[b * n_ctrl + d_sh_first[s]];
    const cuda_real_type q_other = d_vc_q[b * n_ctrl + d_sh_other[s]];
    // F_k = w_1·Q_{k+1} − w_{k+1}·Q_1 ;  residual d_F = −F_k (custom row: assign)
    d_F[b * dim_J + d_sh_row[s]] = -(d_sh_wfirst[s] * q_other - d_sh_wother[s] * q_first);
}

__global__ void vc_apply_step_kernel(
          cuda_real_type* __restrict__ d_vc_q,
    const cuda_real_type* __restrict__ d_dx,
    const int*            __restrict__ d_vc_qcol,
    int n_ctrl,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_ctrl;
    const int j   = tid % n_ctrl;
    if (b >= actual_batch) return;
    d_vc_q[b * n_ctrl + j] += d_dx[b * dim_J + d_vc_qcol[j]];
}

// =============================================================================
// compute_residuals_kernel
//
// One block per contingency; threads in the block cooperate to find ‖F_b‖∞
// via shared-memory reduction, then thread 0 writes the result.
// =============================================================================
__global__ void compute_residuals_kernel(
          cuda_real_type* __restrict__ d_residuals,
    const cuda_real_type* __restrict__ d_F,
    int dim_J,
    int actual_batch,
    int c_start,
    const int* __restrict__ d_result_map)
{
    // One block handles one contingency.
    const int b = blockIdx.x;
    if (b >= actual_batch) return;

    extern __shared__ cuda_real_type sdata[];

    const cuda_real_type* F_b = d_F + b * dim_J;
    cuda_real_type local_max = cuda_real_type(0);

    // Each thread scans its portion of F_b.
    for (int i = threadIdx.x; i < dim_J; i += blockDim.x) {
        cuda_real_type v = F_b[i];
        if (v < cuda_real_type(0)) v = -v;
        if (v > local_max) local_max = v;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree reduction within the block.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int out = d_result_map ? d_result_map[c_start + b] : (c_start + b);
        d_residuals[out] = sdata[0];
    }
}