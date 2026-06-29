// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// acpf_nr_kernels.cu
// =============================================================================

#include "acpf_nr_kernels.cuh"

static constexpr int BS = 256;   // block size for all kernels in this file

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
    const int*             __restrict__ pvpq,
    int n_pvpq,
    int n_bus,
    int dim_J,
    int actual_batch,
    int sbus_stride)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_pvpq;   // contingency index
    const int k   = tid % n_pvpq;   // pvpq slot
    if (b >= actual_batch) return;

    const int bus = pvpq[k];
    const cudaComplexType S_calc = CudaFunHelper::my_cuCmul(
        d_V   [b * n_bus + bus],
        CudaFunHelper::my_cuConj(d_Ibus[b * n_bus + bus]));
    const cuda_real_type dP =
        CudaFunHelper::my_cuCreal(S_calc) -
        CudaFunHelper::my_cuCreal(d_Sbus[b * sbus_stride + bus]);
    d_F[b * dim_J + k] = -dP;
}

// =============================================================================
// fill_FQ_kernel
// =============================================================================
__global__ void fill_FQ_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ pq_idx,
    int n_pvpq,
    int n_pq,
    int n_bus,
    int dim_J,
    int actual_batch,
    int sbus_stride)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_pq;
    const int k   = tid % n_pq;
    if (b >= actual_batch) return;

    const int bus = pq_idx[k];
    const cudaComplexType S_calc = CudaFunHelper::my_cuCmul(
        d_V   [b * n_bus + bus],
        CudaFunHelper::my_cuConj(d_Ibus[b * n_bus + bus]));
    const cuda_real_type dQ =
        CudaFunHelper::my_cuCimag(S_calc) -
        CudaFunHelper::my_cuCimag(d_Sbus[b * sbus_stride + bus]);
    d_F[b * dim_J + n_pvpq + k] = -dQ;
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
    const int*             __restrict__ pvpq,
    int n_pvpq,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_pvpq;
    const int k   = tid % n_pvpq;
    if (b >= actual_batch) return;

    const int bus = pvpq[k];
    const cudaComplexType V = d_V[b * n_bus + bus];
    const cuda_real_type vm = CudaFunHelper::my_cuCabs(V);
    const cuda_real_type va = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V),
        CudaFunHelper::my_cuCreal(V)) + d_dx[b * dim_J + k];

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
    const int*             __restrict__ pq_idx,
    int n_pvpq,
    int n_pq,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int b   = tid / n_pq;
    const int k   = tid % n_pq;
    if (b >= actual_batch) return;

    const int bus = pq_idx[k];
    const cudaComplexType V = d_V[b * n_bus + bus];
    // Vm_new = Vm_old + dx[n_pvpq + k];  Va unchanged (already updated)
    const cuda_real_type vm = CudaFunHelper::my_cuCabs(V) + d_dx[b * dim_J + n_pvpq + k];
    const cuda_real_type va = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V),
        CudaFunHelper::my_cuCreal(V));

    d_V[b * n_bus + bus] = CudaFunHelper::my_make_cuComplex(
        vm * CudaFunHelper::my_cos(va),
        vm * CudaFunHelper::my_sin(va));
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