// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/branch_control_kernels.cu -- see branch_control_kernels.cuh. The formulas
// follow lightsim2grid's NRSystemBranchControl.cpp line by line.
// =============================================================================

#include "branch_control_kernels.cuh"

#include <cmath>

namespace {

using C = CudaFunHelper;

__device__ __forceinline__ cudaComplexType cx(cuda_real_type re, cuda_real_type im) { return C::my_make_cuComplex(re, im); }
__device__ __forceinline__ cudaComplexType cmul(cudaComplexType a, cudaComplexType b) { return C::my_cuCmul(a, b); }
__device__ __forceinline__ cudaComplexType cadd(cudaComplexType a, cudaComplexType b) { return C::my_cuCadd(a, b); }
__device__ __forceinline__ cudaComplexType csub(cudaComplexType a, cudaComplexType b) { return C::my_cuCsub(a, b); }
__device__ __forceinline__ cudaComplexType cconj(cudaComplexType a) { return C::my_cuConj(a); }
__device__ __forceinline__ cudaComplexType cscale(cudaComplexType a, cuda_real_type k) {
    return cx(C::my_cuCreal(a) * k, C::my_cuCimag(a) * k);
}
__device__ __forceinline__ cuda_real_type re(cudaComplexType a) { return C::my_cuCreal(a); }
__device__ __forceinline__ cuda_real_type im(cudaComplexType a) { return C::my_cuCimag(a); }
// j * a
__device__ __forceinline__ cudaComplexType jmul(cudaComplexType a) { return cx(-im(a), re(a)); }

// BranchControl::_block_at: the taps' block at shift a and ratio rho
__device__ __forceinline__ void block_at(const BcStatic& st, const BcSlots& sl, ptrdiff_t idx, int e,
                                         cuda_real_type a, cuda_real_type rho, cudaComplexType blk[4])
{
    const cuda_real_type d = st.sign[e] * (a - sl.a_tap[idx]);
    const cudaComplexType rot = cx(cos(d), sin(d));
    const cuda_real_type f = pow(rho / sl.rho_tap[idx], st.rexp[e]);
    const cudaComplexType* tap = sl.tap + 4 * idx;
    blk[0] = cscale(tap[0], f * f);
    blk[1] = cmul(cscale(tap[1], f), rot);
    blk[2] = cmul(cscale(tap[2], f), cconj(rot));
    blk[3] = tap[3];
}

// BranchControl::_patch
__device__ __forceinline__ void patch(cudaComplexType* Y, const BcStatic& st, const BcSlots& sl,
                                      ptrdiff_t s, ptrdiff_t idx, int e, int nnz_Y)
{
    cudaComplexType blk[4];
    block_at(st, sl, idx, e, sl.a[idx], sl.rho[idx], blk);
    cudaComplexType* Ys = Y + s * nnz_Y;
    cudaComplexType* applied = sl.applied + 4 * idx;
    for (int k = 0; k < 4; ++k) {
        const int p = st.ypos[4 * e + k];
        Ys[p] = cadd(Ys[p], csub(blk[k], applied[k]));
        applied[k] = blk[k];
    }
}

}  // namespace

__global__ void bc_mismatch_kernel(cuda_real_type* __restrict__ d_F, const cudaComplexType* __restrict__ d_V,
                                   BcStatic st, BcSlots sl, int n_bus, int dim_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * st.n_bc) return;
    const ptrdiff_t b = tid / st.n_bc;
    const int e = static_cast<int>(tid % st.n_bc);
    if (st.col[e] < 0) return;
    const ptrdiff_t idx = tid;
    cuda_real_type F;
    if (!sl.on[idx]) {
        F = sl.a[idx] - sl.a_target[idx];
    } else {
        // the block at the current shift is the one stamped (BranchControl::fill_custom_rows at dx = 0)
        const cudaComplexType* y = sl.applied + 4 * idx;
        const cudaComplexType V1 = d_V[b * n_bus + st.b1[e]];
        const cudaComplexType V2 = d_V[b * n_bus + st.b2[e]];
        const cuda_real_type p = st.side[e] == 1
            ? re(cmul(V1, cconj(cadd(cmul(y[0], V1), cmul(y[1], V2)))))
            : re(cmul(V2, cconj(cadd(cmul(y[2], V1), cmul(y[3], V2)))));
        F = p - st.p_target[e];
    }
    d_F[b * dim_J + st.row[e]] = -F;
}

__global__ void bc_fill_J_kernel(cuda_real_type* __restrict__ d_J, const cudaComplexType* __restrict__ d_V,
                                 BcStatic st, BcSlots sl, int n_bus, int nnz_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * st.n_bc) return;
    const ptrdiff_t b = tid / st.n_bc;
    const int e = static_cast<int>(tid % st.n_bc);
    const bool has_r = st.rcol != nullptr && st.rcol[e] >= 0;
    if (st.col[e] < 0 && !has_r) return;
    const ptrdiff_t idx = tid;
    cuda_real_type* J = d_J + b * nnz_J;
    const cudaComplexType* y = sl.applied + 4 * idx;
    const cudaComplexType V1 = d_V[b * n_bus + st.b1[e]];
    const cudaComplexType V2 = d_V[b * n_bus + st.b2[e]];
    const cudaComplexType v1_y12_v2 = cmul(V1, cconj(cmul(y[1], V2)));
    const cudaComplexType v2_y21_v1 = cmul(V2, cconj(cmul(y[2], V1)));
    if (has_r) {
        // dy11/drho = 2 e y11 / rho, dy12/drho = e y12 / rho, dy21/drho = e y21 / rho
        const cuda_real_type k = st.rexp[e] / sl.rho[idx];
        const cudaComplexType dS1 = cscale(cadd(cscale(cmul(V1, cconj(cmul(y[0], V1))), cuda_real_type(2)), v1_y12_v2), k);
        const cudaComplexType dS2 = cscale(v2_y21_v1, k);
        const int* jr = st.jpos_r + 4 * e;
        if (jr[0] >= 0) J[jr[0]] = re(dS1);
        if (jr[1] >= 0) J[jr[1]] = im(dS1);
        if (jr[2] >= 0) J[jr[2]] = re(dS2);
        if (jr[3] >= 0) J[jr[3]] = im(dS2);
    }
    if (st.col[e] < 0) return;
    const int* jp = st.jpos + 9 * e;
    // S1 = V1 conj(y11 V1 + y12 V2), S2 = V2 conj(y21 V1 + y22 V2); dy12/da = j s y12,
    // dy21/da = -j s y21
    const cudaComplexType dS1 = cscale(jmul(v1_y12_v2), -st.sign[e]);
    const cudaComplexType dS2 = cscale(jmul(v2_y21_v1), st.sign[e]);
    if (jp[0] >= 0) J[jp[0]] = re(dS1);
    if (jp[1] >= 0) J[jp[1]] = im(dS1);
    if (jp[2] >= 0) J[jp[2]] = re(dS2);
    if (jp[3] >= 0) J[jp[3]] = im(dS2);
    for (int k = 5; k < 9; ++k) if (jp[k] >= 0) J[jp[k]] = cuda_real_type(0);
    if (!sl.on[idx]) {
        J[jp[4]] = cuda_real_type(1);
        return;
    }
    const cuda_real_type vm1 = C::my_cuCabs(V1);
    const cuda_real_type vm2 = C::my_cuCabs(V2);
    cuda_real_type d_th1, d_th2, d_vm1, d_vm2, d_a;
    if (st.side[e] == 1) {
        const cudaComplexType S1 = cmul(V1, cconj(cadd(cmul(y[0], V1), cmul(y[1], V2))));
        d_th1 = re(jmul(v1_y12_v2));
        d_th2 = -d_th1;
        d_vm1 = re(cadd(S1, cscale(cconj(y[0]), vm1 * vm1))) / vm1;
        d_vm2 = re(v1_y12_v2) / vm2;
        d_a = re(dS1);
    } else {
        const cudaComplexType S2 = cmul(V2, cconj(cadd(cmul(y[2], V1), cmul(y[3], V2))));
        d_th2 = re(jmul(v2_y21_v1));
        d_th1 = -d_th2;
        d_vm2 = re(cadd(S2, cscale(cconj(y[3]), vm2 * vm2))) / vm2;
        d_vm1 = re(v2_y21_v1) / vm1;
        d_a = re(dS2);
    }
    J[jp[4]] = d_a;
    if (jp[5] >= 0) J[jp[5]] = d_th1;
    if (jp[6] >= 0) J[jp[6]] = d_th2;
    if (jp[7] >= 0) J[jp[7]] = d_vm1;
    if (jp[8] >= 0) J[jp[8]] = d_vm2;
}

__global__ void ctl_group_mismatch_kernel(cuda_real_type* __restrict__ d_F, const cudaComplexType* __restrict__ d_V,
                                          CtlGroups g, const cuda_real_type* __restrict__ x,
                                          const cuda_real_type* __restrict__ x_target,
                                          const unsigned char* __restrict__ on, int n_ent,
                                          int n_bus, int dim_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * g.n_grp) return;
    const ptrdiff_t b = tid / g.n_grp;
    const int k = static_cast<int>(tid % g.n_grp);
    const int s0 = g.start[k], n = g.start[k + 1] - s0;
    cuda_real_type mean = 0;
    for (int j = 0; j < n; ++j) mean += x[b * n_ent + g.member[s0 + j]];
    mean /= static_cast<cuda_real_type>(n);
    bool first_on = true;
    for (int j = 0; j < n; ++j) {
        const ptrdiff_t idx = b * n_ent + g.member[s0 + j];
        cuda_real_type F;
        if (!on[idx]) {
            F = x[idx] - x_target[idx];
        } else if (first_on) {
            first_on = false;
            F = C::my_cuCabs(d_V[b * n_bus + g.bus[k]]) - g.target[k];
        } else {
            F = mean - x[idx];
        }
        d_F[b * dim_J + g.row[s0 + j]] = -F;
    }
}

__global__ void ctl_group_fill_J_kernel(cuda_real_type* __restrict__ d_J, CtlGroups g,
                                        const unsigned char* __restrict__ on, int n_ent, int nnz_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * g.n_grp) return;
    const ptrdiff_t b = tid / g.n_grp;
    const int k = static_cast<int>(tid % g.n_grp);
    const int s0 = g.start[k], n = g.start[k + 1] - s0;
    cuda_real_type* J = d_J + b * nnz_J;
    const int* jc = g.jcol + g.jstart[k];
    const cuda_real_type inv_n = cuda_real_type(1) / static_cast<cuda_real_type>(n);
    bool first_on = true;
    for (int j = 0; j < n; ++j) {
        const ptrdiff_t idx = b * n_ent + g.member[s0 + j];
        // every form's positions zero, then the row's own
        for (int m = 0; m < n; ++m) if (jc[j * n + m] >= 0) J[jc[j * n + m]] = cuda_real_type(0);
        if (g.jvm[s0 + j] >= 0) J[g.jvm[s0 + j]] = cuda_real_type(0);
        if (!on[idx]) {
            if (jc[j * n + j] >= 0) J[jc[j * n + j]] = cuda_real_type(1);
        } else if (first_on) {
            first_on = false;
            if (g.jvm[s0 + j] >= 0) J[g.jvm[s0 + j]] = cuda_real_type(1);
        } else {
            for (int m = 0; m < n; ++m)
                if (jc[j * n + m] >= 0) J[jc[j * n + m]] = m == j ? inv_n - cuda_real_type(1) : inv_n;
        }
    }
}

__global__ void bc_step_kernel(const cuda_real_type* __restrict__ d_dx, cudaComplexType* __restrict__ d_Ybus,
                               BcStatic st, BcSlots sl, int dim_J, int nnz_Y, int batch)
{
    // one thread per slot, its transformers in order: parallel ones patch the same Ybus
    // coefficients (as BranchControl::apply_step, in its order)
    const ptrdiff_t b = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    for (int e = 0; e < st.n_bc; ++e) {
        const ptrdiff_t idx = b * st.n_bc + e;
        const bool has_r = st.rcol != nullptr && st.rcol[e] >= 0;
        if ((st.col[e] < 0 && !has_r) || sl.out[idx]) continue;
        const cuda_real_type da = st.col[e] >= 0 ? d_dx[b * dim_J + st.col[e]] : cuda_real_type(0);
        const cuda_real_type dr = has_r ? d_dx[b * dim_J + st.rcol[e]] : cuda_real_type(0);
        if (da == cuda_real_type(0) && dr == cuda_real_type(0)) continue;
        sl.a[idx] += da;
        sl.rho[idx] += dr;
        if (d_Ybus != nullptr) patch(d_Ybus, st, sl, b, idx, e, nnz_Y);
    }
}

__global__ void bc_controls_kernel(cudaComplexType* __restrict__ d_Ybus, BcStatic st, BcSlots sl,
                                   const int* __restrict__ d_slots, int n_list, bool fresh,
                                   const unsigned char* __restrict__ d_out,
                                   const unsigned char* __restrict__ d_retap,
                                   const cudaComplexType* __restrict__ d_tap,
                                   const cuda_real_type* __restrict__ d_a_tap,
                                   const cuda_real_type* __restrict__ d_rho_tap,
                                   const signed char* __restrict__ d_ctl,
                                   const signed char* __restrict__ d_rctl,
                                   int nnz_Y)
{
    // one thread per listed slot, its transformers in order (see bc_step_kernel)
    const ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_list) return;
    const ptrdiff_t s = d_slots[i];
    for (int e = 0; e < st.n_bc; ++e) {
    const ptrdiff_t tid = i * st.n_bc + e;
    const ptrdiff_t idx = s * st.n_bc + e;
    if (fresh) {
        // BranchControl::_reset (and drop_out_of_row)
        sl.out[idx] = d_out[tid];
        for (int k = 0; k < 4; ++k) {
            sl.tap[4 * idx + k] = st.tap0[4 * e + k];
            sl.applied[4 * idx + k] = sl.out[idx] ? cx(0, 0) : st.yeff0[4 * e + k];
        }
        sl.a_tap[idx] = st.a_tap0[e];
        sl.rho_tap[idx] = st.rho_tap0[e];
        sl.a[idx] = st.a_tap0[e];
        sl.rho[idx] = st.rho_tap0[e];
        sl.a_target[idx] = st.a_tap0[e];
        sl.rho_target[idx] = st.rho_tap0[e];
        sl.on[idx] = 0;
        sl.ron[idx] = 0;
    }
    if (sl.out[idx]) continue;
    if (d_retap[tid]) {
        // BranchControl::_retap
        for (int k = 0; k < 4; ++k) sl.tap[4 * idx + k] = d_tap[4 * tid + k];
        sl.a_tap[idx] = d_a_tap[tid];
        sl.rho_tap[idx] = d_rho_tap[tid];
        sl.a[idx] = sl.a_tap[idx];
        sl.rho[idx] = sl.rho_tap[idx];
        sl.a_target[idx] = sl.a_tap[idx];
        sl.rho_target[idx] = sl.rho_tap[idx];
    }
    // BranchControl::set_control_on (a transformer without a shift column has no control),
    // then set_ratio_control_on (the same for the ratio column)
    if (st.col[e] >= 0 && d_ctl[tid] >= 0) {
        sl.on[idx] = d_ctl[tid] == 1;
        if (d_ctl[tid] == 0) sl.a_target[idx] = sl.a[idx];
    }
    if (st.rcol != nullptr && st.rcol[e] >= 0 && d_rctl != nullptr && d_rctl[tid] >= 0) {
        sl.ron[idx] = d_rctl[tid] == 1;
        if (d_rctl[tid] == 0) sl.rho_target[idx] = sl.rho[idx];
    }
    if (d_Ybus != nullptr) patch(d_Ybus, st, sl, s, idx, e, nnz_Y);
    }
}

__global__ void bc_restore_kernel(cudaComplexType* __restrict__ d_Ybus, BcStatic st, BcSlots sl,
                                  const int* __restrict__ d_evict, int nnz_Y, int batch)
{
    // one thread per slot, its transformers in order (see bc_step_kernel)
    const ptrdiff_t s = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= batch) return;
    if (d_evict != nullptr && d_evict[s] < 0) return;
    cudaComplexType* Ys = d_Ybus + s * nnz_Y;
    for (int e = 0; e < st.n_bc; ++e) {
        const ptrdiff_t idx = s * st.n_bc + e;
        if (sl.out[idx]) continue;
        cudaComplexType* applied = sl.applied + 4 * idx;
        for (int k = 0; k < 4; ++k) {
            const int p = st.ypos[4 * e + k];
            Ys[p] = cadd(Ys[p], csub(st.yeff0[4 * e + k], applied[k]));
            applied[k] = st.yeff0[4 * e + k];
        }
    }
}

__global__ void bc_gather_kernel(cuda_real_type* __restrict__ d_out, BcStatic st, BcSlots sl,
                                 const int* __restrict__ d_slots, int n_list, int offset, int stride)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(n_list) * st.n_bc) return;
    const ptrdiff_t i = tid / st.n_bc;
    const int e = static_cast<int>(tid % st.n_bc);
    const ptrdiff_t idx = static_cast<ptrdiff_t>(d_slots[i]) * st.n_bc + e;
    d_out[i * stride + offset + e] = sl.a[idx];
    d_out[i * stride + offset + st.n_bc + e] = sl.rho[idx];
}

// =============================================================================
// ShuntControl (NRSystemShuntControl.cpp)
// =============================================================================
namespace {
__device__ __forceinline__ void sc_patch(cudaComplexType* Y, const ScStatic& st, const ScSlots& sl,
                                         ptrdiff_t s, ptrdiff_t idx, int e, int nnz_Y)
{
    cudaComplexType* p = Y + s * nnz_Y + st.ypos[e];
    *p = cadd(*p, cx(0, sl.b[idx] - sl.b_applied[idx]));
    sl.b_applied[idx] = sl.b[idx];
}
}  // namespace

__global__ void sc_fill_J_kernel(cuda_real_type* __restrict__ d_J, const cudaComplexType* __restrict__ d_V,
                                 ScStatic st, ScSlots sl, int n_bus, int nnz_J, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * st.n_sc) return;
    const ptrdiff_t b = tid / st.n_sc;
    const int e = static_cast<int>(tid % st.n_sc);
    if (st.jq[e] < 0) return;
    // S = V conj(j B V) = -j B |V|^2
    const cudaComplexType V = d_V[b * n_bus + st.bus[e]];
    d_J[b * nnz_J + st.jq[e]] = -(re(V) * re(V) + im(V) * im(V));
    (void)sl;
}

__global__ void sc_step_kernel(const cuda_real_type* __restrict__ d_dx, cudaComplexType* __restrict__ d_Ybus,
                               ScStatic st, ScSlots sl, int dim_J, int nnz_Y, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * st.n_sc) return;
    const ptrdiff_t b = tid / st.n_sc;
    const int e = static_cast<int>(tid % st.n_sc);
    if (st.col[e] < 0) return;
    const cuda_real_type db = d_dx[b * dim_J + st.col[e]];
    if (db == cuda_real_type(0)) return;
    sl.b[tid] += db;
    if (d_Ybus != nullptr) sc_patch(d_Ybus, st, sl, b, tid, e, nnz_Y);
}

__global__ void sc_controls_kernel(cudaComplexType* __restrict__ d_Ybus, ScStatic st, ScSlots sl,
                                   const int* __restrict__ d_slots, int n_list, bool fresh,
                                   const unsigned char* __restrict__ d_set,
                                   const cuda_real_type* __restrict__ d_b_new,
                                   const signed char* __restrict__ d_ctl, int nnz_Y)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(n_list) * st.n_sc) return;
    const ptrdiff_t i = tid / st.n_sc;
    const int e = static_cast<int>(tid % st.n_sc);
    const ptrdiff_t s = d_slots[i];
    const ptrdiff_t idx = s * st.n_sc + e;
    if (fresh) {
        sl.b[idx] = st.b0[e];
        sl.b_applied[idx] = st.b0[e];
        sl.b_target[idx] = st.b0[e];
        sl.on[idx] = 0;
    }
    if (d_set[tid]) {   // ShuntControl::set_sections
        sl.b[idx] = d_b_new[tid];
        sl.b_target[idx] = d_b_new[tid];
    }
    if (st.col[e] >= 0 && d_ctl[tid] >= 0) {   // ShuntControl::set_control_on
        sl.on[idx] = d_ctl[tid] == 1;
        if (d_ctl[tid] == 0) sl.b_target[idx] = sl.b[idx];
    }
    if (d_Ybus != nullptr) sc_patch(d_Ybus, st, sl, s, idx, e, nnz_Y);
}

__global__ void sc_restore_kernel(cudaComplexType* __restrict__ d_Ybus, ScStatic st, ScSlots sl,
                                  const int* __restrict__ d_evict, int nnz_Y, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(batch) * st.n_sc) return;
    const ptrdiff_t s = tid / st.n_sc;
    const int e = static_cast<int>(tid % st.n_sc);
    if (d_evict != nullptr && d_evict[s] < 0) return;
    cudaComplexType* p = d_Ybus + s * nnz_Y + st.ypos[e];
    *p = cadd(*p, cx(0, st.b0[e] - sl.b_applied[tid]));
    sl.b_applied[tid] = st.b0[e];
}

__global__ void sc_gather_kernel(cuda_real_type* __restrict__ d_out, ScStatic st, ScSlots sl,
                                 const int* __restrict__ d_slots, int n_list, int offset, int stride)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(n_list) * st.n_sc) return;
    const ptrdiff_t i = tid / st.n_sc;
    const int e = static_cast<int>(tid % st.n_sc);
    d_out[i * stride + offset + e] = sl.b[static_cast<ptrdiff_t>(d_slots[i]) * st.n_sc + e];
}
