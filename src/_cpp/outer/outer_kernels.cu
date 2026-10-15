// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/outer_kernels.cu -- see outer_kernels.cuh
// =============================================================================

#include "outer_kernels.cuh"

#include <cmath>

namespace {

__device__ __forceinline__ bool oc_held(const cuda_real_type* held_q, const unsigned char* plan_held,
                                        ptrdiff_t at)
{
    return (plan_held != nullptr && plan_held[at] != 0) || isfinite(held_q[at]);
}

// the first controller of a group still regulating (lightsim2grid's _ref_active),
// -1 when every one is held
__device__ __forceinline__ int oc_ref_active(const cuda_real_type* held_q, const unsigned char* plan_held,
                                             ptrdiff_t base, int first, int cnt)
{
    for (int off = 0; off < cnt; ++off)
        if (!oc_held(held_q, plan_held, base + first + off)) return off;
    return -1;
}

}  // namespace

__global__ void oc_pin_rows_kernel(
          cuda_real_type* __restrict__ d_J_values,
          cuda_real_type* __restrict__ d_F,
    const unsigned char*  __restrict__ d_pin,
    const int*            __restrict__ d_pin_row,
    const int*            __restrict__ d_pin_diag,
    const int*            __restrict__ d_J_outer,
    int n_pin, int nnz_J, int dim_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * n_pin) return;
    if (!d_pin[tid]) return;
    const ptrdiff_t b = tid / n_pin;
    const int       k = static_cast<int>(tid % n_pin);
    const int row = d_pin_row[k];
    if (row < 0) return;
    if (d_J_values != nullptr) {
        cuda_real_type* J = d_J_values + b * nnz_J;
        for (int p = d_J_outer[row]; p < d_J_outer[row + 1]; ++p) J[p] = cuda_real_type(0);
        if (d_pin_diag[k] >= 0) J[d_pin_diag[k]] = cuda_real_type(1);
    }
    if (d_F != nullptr) d_F[b * dim_J + row] = cuda_real_type(0);
}

__global__ void vc_hold_fill_J_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const cuda_real_type* __restrict__ d_held_q,
    const unsigned char*  __restrict__ d_plan_held,
    const cuda_real_type* __restrict__ d_slope,
    const cuda_real_type* __restrict__ d_weight,
    const int*            __restrict__ d_slope_pos,
    const int*            __restrict__ d_vm_pos,
    const int*            __restrict__ d_rowpos_off,
    const int*            __restrict__ d_rowpos,
    const int*            __restrict__ d_grp_start,
    const int*            __restrict__ d_grp_count,
    int n_ctrl, int n_grp, int nnz_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * n_grp) return;
    const ptrdiff_t b = tid / n_grp;
    const int       g = static_cast<int>(tid % n_grp);
    cuda_real_type* J = d_J_values + b * nnz_J;
    const ptrdiff_t base = b * n_ctrl;
    const int first = d_grp_start[g];
    const int cnt   = d_grp_count[g];
    const int ref   = oc_ref_active(d_held_q, d_plan_held, base, first, cnt);
    const cuda_real_type one = cuda_real_type(1), zero = cuda_real_type(0);

    // nobody regulates: the voltage row pins the first controller
    for (int off = 0; off < cnt; ++off) {
        const int j = first + off;
        if (d_slope_pos[j] >= 0)
            J[d_slope_pos[j]] = ref >= 0 ? d_slope[j] : (off == 0 ? one : zero);
    }
    if (d_vm_pos[g] >= 0) J[d_vm_pos[g]] = ref >= 0 ? one : zero;
    // the sharing rows, taken against the first ACTIVE controller
    for (int m = 1; m < cnt; ++m) {
        const int j = first + m;
        const int* P = d_rowpos + d_rowpos_off[j];   // P[o]: (row of j, q_col of first + o)
        for (int o = 0; o < cnt; ++o) if (P[o] >= 0) J[P[o]] = zero;
        if (ref < 0 || oc_held(d_held_q, d_plan_held, base + j)) {
            if (P[m] >= 0) J[P[m]] = one;                 // Q_m = held
        } else if (m == ref) {
            if (P[0] >= 0) J[P[0]] = one;                 // the first controller is held: pinned here
        } else {
            if (P[m] >= 0)   J[P[m]]   =  d_weight[first + ref];   // w_ref.Q_m - w_m.Q_ref
            if (P[ref] >= 0) J[P[ref]] = -d_weight[j];
        }
    }
}

__global__ void vc_hold_rows_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_vc_q,
    const cuda_real_type*  __restrict__ d_held_q,
    const unsigned char*   __restrict__ d_plan_held,
    const cuda_real_type*  __restrict__ d_q_plan,
    const cuda_real_type*  __restrict__ d_vset,
    int vset_stride,
    const cuda_real_type*  __restrict__ d_slope,
    const cuda_real_type*  __restrict__ d_weight,
    const int*             __restrict__ d_reg_bus,
    const int*             __restrict__ d_vrow,
    const int*             __restrict__ d_sh_row,
    const int*             __restrict__ d_grp_start,
    const int*             __restrict__ d_grp_count,
    int n_ctrl, int n_grp, int n_bus, int dim_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * n_grp) return;
    const ptrdiff_t b = tid / n_grp;
    const int       g = static_cast<int>(tid % n_grp);
    cuda_real_type* F = d_F + b * dim_J;
    const ptrdiff_t base = b * n_ctrl;
    const int first = d_grp_start[g];
    const int cnt   = d_grp_count[g];
    const int ref   = oc_ref_active(d_held_q, d_plan_held, base, first, cnt);
    auto Q  = [&](int off) { return d_vc_q[base + first + off]; };
    // the output a held controller holds: its value, or the plan's frozen one
    auto hq = [&](int off) {
        const cuda_real_type v = d_held_q[base + first + off];
        if (isfinite(v)) return v;
        return d_q_plan != nullptr ? d_q_plan[first + off] : cuda_real_type(0);
    };
    if (ref < 0) {
        F[d_vrow[g]] = -(Q(0) - hq(0));
    } else {
        const cuda_real_type vm = CudaFunHelper::my_cuCabs(d_V[b * n_bus + d_reg_bus[g]]);
        cuda_real_type slope_term = cuda_real_type(0);
        for (int off = 0; off < cnt; ++off) slope_term += d_slope[first + off] * Q(off);
        F[d_vrow[g]] = -(vm + slope_term - d_vset[b * vset_stride + g]);
    }
    for (int m = 1; m < cnt; ++m) {
        const int row = d_sh_row[first + m];
        if (row < 0) continue;
        if (ref < 0 || oc_held(d_held_q, d_plan_held, base + first + m)) {
            F[row] = -(Q(m) - hq(m));
        } else if (m == ref) {
            F[row] = -(Q(0) - hq(0));
        } else {
            F[row] = -(d_weight[first + ref] * Q(m) - d_weight[first + m] * Q(ref));
        }
    }
}

__global__ void oc_gather_kernel(
          cuda_real_type*  __restrict__ d_out,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    int sbus_stride,
    const cuda_real_type*  __restrict__ d_vc_q,
    const int*             __restrict__ d_slots,
    const int*             __restrict__ d_watch,
    int n_list, int n_watch, int n_ctrl, int n_bus, int stride)
{
    const int per = n_watch + n_ctrl;
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(n_list) * per) return;
    const int i = static_cast<int>(tid / per);
    const int w = static_cast<int>(tid % per);
    const ptrdiff_t s = d_slots[i];
    cuda_real_type* out = d_out + static_cast<ptrdiff_t>(i) * stride;
    if (w >= n_watch) {
        const int c = w - n_watch;
        out[6 * n_watch + c] = d_vc_q[s * n_ctrl + c];
        return;
    }
    const int bus = d_watch[w];
    const cudaComplexType V  = d_V[s * n_bus + bus];
    const cudaComplexType I  = d_Ibus[s * n_bus + bus];
    const cudaComplexType Sc = CudaFunHelper::my_cuCmul(V, CudaFunHelper::my_cuConj(I));
    const cudaComplexType Sb = d_Sbus[s * sbus_stride + bus];
    out[0 * n_watch + w] = CudaFunHelper::my_cuCreal(V);
    out[1 * n_watch + w] = CudaFunHelper::my_cuCimag(V);
    out[2 * n_watch + w] = CudaFunHelper::my_cuCreal(Sc);
    out[3 * n_watch + w] = CudaFunHelper::my_cuCimag(Sc);
    out[4 * n_watch + w] = CudaFunHelper::my_cuCreal(Sb);
    out[5 * n_watch + w] = CudaFunHelper::my_cuCimag(Sb);
}

__global__ void oc_unrealistic_kernel(
          cuda_real_type*  __restrict__ d_out,
    const cudaComplexType* __restrict__ d_V,
    const unsigned char*   __restrict__ d_pin,
    const int*             __restrict__ d_cand_bus,
    const int*             __restrict__ d_cand_pin,
    const int*             __restrict__ d_slots,
    int n_list, int n_cand, int n_pin, int n_bus, int stride,
    cuda_real_type vmin, cuda_real_type vmax)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_list) return;
    const ptrdiff_t s = d_slots[i];
    cuda_real_type flag = cuda_real_type(0);
    for (int c = 0; c < n_cand; ++c) {
        const int k = d_cand_pin[c];
        if (k >= 0 && d_pin != nullptr && d_pin[s * n_pin + k]) continue;   // PV: not an unknown
        const cuda_real_type vm = CudaFunHelper::my_cuCabs(d_V[s * n_bus + d_cand_bus[c]]);
        if (vm < vmin || vm > vmax) { flag = cuda_real_type(1); break; }
    }
    d_out[static_cast<ptrdiff_t>(i) * stride + stride - 1] = flag;
}

__global__ void oc_sbus_delta_kernel(
          cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ d_slot,
    const int*             __restrict__ d_bus,
    const cuda_real_type*  __restrict__ d_dp,
    const cuda_real_type*  __restrict__ d_dq,
    int n, int n_bus)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    cudaComplexType& s = d_Sbus[static_cast<ptrdiff_t>(d_slot[t]) * n_bus + d_bus[t]];
    s = CudaFunHelper::my_make_cuComplex(CudaFunHelper::my_cuCreal(s) + d_dp[t],
                                         CudaFunHelper::my_cuCimag(s) + d_dq[t]);
}
