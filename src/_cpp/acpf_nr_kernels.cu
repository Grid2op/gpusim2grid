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
//
// Bit-reinterpretation note (applies to every CAS-based function in this
// file): __double_as_longlong/__longlong_as_double/__float_as_int/
// __int_as_float are dedicated CUDA intrinsics for exactly this
// bit-for-bit reinterpretation -- well-defined, not UB (the CUDA
// equivalent of memcpy/std::bit_cast). atomicCAS() itself, though, has no
// float/double overload, so it can only be driven through a
// reinterpret_cast<unsigned long long int*>/reinterpret_cast<int*> of the
// double*/float* address -- that pointer-cast-and-dereference is classic
// strict-aliasing type punning, UB by the letter of the C++ standard CUDA
// C++ inherits. It's the exact idiom NVIDIA's own CUDA C++ Programming
// Guide uses for this same double-atomicAdd fallback, and nvcc does not
// do the aliasing-based reordering that would break it in device code --
// but it is not standard-legal, only a de facto-safe, universally-used
// CUDA idiom. The initial reads below go through the intrinsics on
// *addr (the correctly-typed pointer) specifically to avoid adding a
// second, avoidable instance of that same punned dereference.
__device__ __forceinline__ void atomic_add_real(float* addr, float val) {
    atomicAdd(addr, val);
}
__device__ __forceinline__ void atomic_add_real(double* addr, double val) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(addr, val);
#else
    unsigned long long int* p = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = __double_as_longlong(*addr), assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// CUDA has no built-in atomicMax for float/double (only for integer types),
// so this reimplements it as a compare-and-swap loop: read the current value,
// bail out if it's already >= val, otherwise try to swap in val and retry on
// contention. This is just "atomicMax", specialized to NON-NEGATIVE
// floats/doubles -- for those, comparing the bit pattern as if it were an
// integer gives the same ordering as comparing the float/double value
// itself (true only when the sign bit is always 0), so the integer
// atomicCAS above can drive the whole loop. Works on all compute
// capabilities, unlike the native double atomicAdd this file also needs a
// fallback for. Used by reduce_step_norms_kernel below, which only ever
// feeds it fabs(dx) (always non-negative) to find, per batch slot, the
// largest |dtheta|/|dvm| step any bus is about to take -- the max-voltage-
// change step-scaling feature needs that per-slot maximum to compute how
// much to shrink the whole Newton step by.
__device__ __forceinline__ void atomic_max_nonneg(float* addr, float val) {
    int* p = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr), assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(p, assumed, __float_as_int(val));
    } while (assumed != old);
}
__device__ __forceinline__ void atomic_max_nonneg(double* addr, double val) {
    unsigned long long int* p = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = __double_as_longlong(*addr), assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(p, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// =============================================================================
// override_branch_flows_kernel
// =============================================================================
__global__ void override_branch_flows_kernel(
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ d_row,
    const int*             __restrict__ d_branch,
    const cudaComplexType* __restrict__ d_blk,
    const int*             __restrict__ d_branch_from,
    const int*             __restrict__ d_branch_to,
    const cuda_real_type*  __restrict__ d_base_current_A,
    const cuda_real_type*  __restrict__ d_base_current_ex_A,
          cuda_real_type*  __restrict__ d_or_amps,
          cuda_real_type*  __restrict__ d_ex_amps,
    int n_bus, int n_branches, int n,
    const int*             __restrict__ d_ft,
    const cuda_real_type*  __restrict__ d_base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ptrdiff_t r = d_row[i];
    const int l = d_branch[i];
    const int bf = d_ft ? d_ft[2 * i] : d_branch_from[l];
    const int bt = d_ft ? d_ft[2 * i + 1] : d_branch_to[l];
    const cudaComplexType zero = CudaFunHelper::my_make_cuComplex(0., 0.);
    const cudaComplexType Vi = (bf >= 0) ? d_V[r * n_bus + bf] : zero;
    const cudaComplexType Vj = (bt >= 0) ? d_V[r * n_bus + bt] : zero;
    const cudaComplexType* y = d_blk + 4 * static_cast<ptrdiff_t>(i);
    const cudaComplexType I_or = (bf >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(y[0], Vi), CudaFunHelper::my_cuCmul(y[1], Vj)) : zero;
    const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(y[2], Vi), CudaFunHelper::my_cuCmul(y[3], Vj)) : zero;
    const ptrdiff_t out_idx = r * n_branches + l;
    d_or_amps[out_idx] = CudaFunHelper::my_cuCabs(I_or) * (d_base ? d_base[2 * i] : d_base_current_A[l]);
    d_ex_amps[out_idx] = CudaFunHelper::my_cuCabs(I_ex) * (d_base ? d_base[2 * i + 1] : d_base_current_ex_A[l]);
}

// =============================================================================
// compute_branch_flows_kernel
// =============================================================================
__global__ void compute_branch_flows_kernel(
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ d_branch_from,
    const int*             __restrict__ d_branch_to,
    const cudaComplexType* __restrict__ d_yff_eff,
    const cudaComplexType* __restrict__ d_yft_eff,
    const cudaComplexType* __restrict__ d_ytf_eff,
    const cudaComplexType* __restrict__ d_ytt_eff,
    const cuda_real_type*  __restrict__ d_base_current_A,
    const cuda_real_type*  __restrict__ d_base_current_ex_A,
          cuda_real_type*  __restrict__ d_or_amps,
          cuda_real_type*  __restrict__ d_ex_amps,
    int n_bus,
    int n_branches,
    int c_start,
    int actual_batch,
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active)
{
    // tid/b widened to ptrdiff_t: see fill_J_kernel's own note. Here
    // actual_batch * n_branches (this launch's thread count) and b * n_bus /
    // out_c * n_branches (the offsets below) are the at-risk products.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_branches;   // contingency index in batch
    const int        l  = static_cast<int>(tid % n_branches);   // branch index
    if (b >= actual_batch) return;
    const int a = d_slot_active ? d_slot_active[b] : static_cast<int>(c_start + b);
    if (a < 0) return;

    // branch_from/branch_to are in AC-solver bus numbering (see
    // concat_busids_to_solver, ls2g_bridge.cpp): a side that lightsim2grid
    // Kron-reduced away (isolated / half-open line, keep_half_open_lines) is
    // relabeled to -1. That bus has no voltage in the solved system -- treat
    // it as V=0 -- and no terminal current on that side to report -- 0, not
    // computed from the pi-model formula, since the terminal itself doesn't
    // exist. Reading d_V[... + (-1)] without this guard is an out-of-bounds
    // read (confirmed via compute-sanitizer on a real half-open-line grid).
    const int bf = d_branch_from[l];
    const int bt = d_branch_to[l];
    const cudaComplexType Vi = (bf >= 0) ? d_V[b * n_bus + bf] : CudaFunHelper::my_make_cuComplex(0., 0.);
    const cudaComplexType Vj = (bt >= 0) ? d_V[b * n_bus + bt] : CudaFunHelper::my_make_cuComplex(0., 0.);

    // I_or = yff_eff * Vi + yft_eff * Vj  (origin / from-bus terminal current)
    const cudaComplexType I_or = (bf >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_yff_eff[l], Vi),
        CudaFunHelper::my_cuCmul(d_yft_eff[l], Vj)) : CudaFunHelper::my_make_cuComplex(0., 0.);

    // I_ex = ytf_eff * Vi + ytt_eff * Vj  (extremity / to-bus terminal current)
    const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_ytf_eff[l], Vi),
        CudaFunHelper::my_cuCmul(d_ytt_eff[l], Vj)) : CudaFunHelper::my_make_cuComplex(0., 0.);

    // Map the chunk-relative slot to its original result index (identity when
    // d_result_map is null, e.g. the full-batch session call or injection).
    // out_c itself (a contingency index < n_contingencies) fits int32; only
    // out_c * n_branches (the flat offset) needs the 64-bit product.
    const int out_c   = d_result_map ? d_result_map[a] : a;
    const ptrdiff_t out_idx = static_cast<ptrdiff_t>(out_c) * n_branches + l;
    d_or_amps[out_idx] = CudaFunHelper::my_cuCabs(I_or) * d_base_current_A[l];
    d_ex_amps[out_idx] = CudaFunHelper::my_cuCabs(I_ex) * d_base_current_ex_A[l];
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
    int actual_batch,
    const int*             __restrict__ d_slot_active)
{
    // tid/total/local_c widened to ptrdiff_t -- actual_batch * n_bus (this
    // launch's thread count) and local_c/out_c * n_bus (the offsets below)
    // are the at-risk products; see fill_J_kernel's own note.
    const ptrdiff_t tid   = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t total = static_cast<ptrdiff_t>(actual_batch) * n_bus;
    if (tid >= total) return;

    const ptrdiff_t local_c = tid / n_bus;   // active-slot index in this chunk
    const int       bus     = static_cast<int>(tid % n_bus);
    const int a = d_slot_active ? d_slot_active[local_c] : static_cast<int>(c_start + local_c);
    if (a < 0) return;
    const int out_c = d_result_map ? d_result_map[a] : a;   // original index
    d_V_results[static_cast<ptrdiff_t>(out_c) * n_bus + bus] = d_V_batch[local_c * n_bus + bus];
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
// apply_J_overrides_kernel
// =============================================================================
__global__ void apply_J_overrides_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const int*           __restrict__ d_jov_slot,
    const int*           __restrict__ d_jov_pos,
    const cuda_real_type* __restrict__ d_jov_val,
    int nnz_J,
    int n_entries)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;
    d_J_values[static_cast<ptrdiff_t>(d_jov_slot[tid]) * nnz_J + d_jov_pos[tid]] = d_jov_val[tid];
}

// =============================================================================
// vc_stranded_vrow_kernel
// =============================================================================
__global__ void vc_stranded_vrow_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,
    const int*            __restrict__ d_vc_vrow,
    const int*            __restrict__ d_vc_grp_start,
    const int*            __restrict__ d_str_slot,
    const int*            __restrict__ d_str_grp,
    int n_ctrl,
    int dim_J,
    int n_entries)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;
    const ptrdiff_t slot = d_str_slot[tid];
    const int       g    = d_str_grp[tid];
    // F_v = Q_first (pin the first controller's reactive injection to 0; the
    // group's sharing rows pin the others);
    // residual d_F = -F_v (custom row: assign), replacing vc_vrow_kernel's.
    d_F[slot * dim_J + d_vc_vrow[g]] = -d_vc_q[slot * n_ctrl + d_vc_grp_start[g]];
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
    int n_entries,
    const int*             __restrict__ d_slot_active)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;
    const int slot = d_maskv_slot[tid];
    // continuous batching: only the slots leaving this round (a slot that
    // stays keeps iterating on its masked buses' finite values)
    if (d_slot_active && d_slot_active[slot] < 0) return;
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
    const long long*     __restrict__ d_zero_indices,
    long long n_entries)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_entries) return;
    const long long idx = d_zero_indices[i];
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
    const ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_updates) return;

    // Offset to the start of this contingency's value block. ctg_id[i] * nnz_Y
    // is the same at-risk product as fill_J_kernel's J_base -- widen it too.
    // No atomicAdd needed: each (ctg_id, k) pair is unique by construction.
    const ptrdiff_t base = static_cast<ptrdiff_t>(ctg_id[i]) * nnz_Y;
    cudaComplexType& entry = value_ptr_batch[base + k_idx[i]];
    entry = CudaFunHelper::my_cuCsub(
        entry,
        CudaFunHelper::my_make_cuComplex(delta_re[i], delta_im[i]));
}

// =============================================================================
// apply_gen_v_kernel
// =============================================================================
__global__ void apply_gen_v_kernel(
          cudaComplexType* __restrict__ d_V_batch,
    const cuda_real_type*  __restrict__ d_gen_v_all,
    const int*              __restrict__ d_active_bus,
    int row_offset,
    int k_active,
    int actual_batch,
    int n_bus)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t r   = tid / k_active;
    const int       j   = static_cast<int>(tid % k_active);
    if (r >= actual_batch) return;

    const cuda_real_type target =
        d_gen_v_all[static_cast<ptrdiff_t>(row_offset + r) * k_active + j];
    if (isnan(target)) return;

    const int bus = d_active_bus[j];
    cudaComplexType v = d_V_batch[r * n_bus + bus];
    cuda_real_type mag = CudaFunHelper::my_cuCabs(v);
    if (mag < cuda_real_type(1e-10)) {
        // matches GeneratorContainer::_set_vm_impl: a collapsed magnitude is
        // forced to 1 first, so the scale below still lands on `target`.
        v   = CudaFunHelper::my_make_cuComplex(cuda_real_type(1), cuda_real_type(0));
        mag = cuda_real_type(1);
    }
    const cuda_real_type scale = target / mag;
    d_V_batch[r * n_bus + bus] = CudaFunHelper::my_make_cuComplex(
        CudaFunHelper::my_cuCreal(v) * scale,
        CudaFunHelper::my_cuCimag(v) * scale);
}

// =============================================================================
// vc_pinned_ctrl_rows_kernel
// =============================================================================
__global__ void vc_pinned_ctrl_rows_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,
    const int*            __restrict__ d_slot,
    const int*            __restrict__ d_row,
    const int*            __restrict__ d_ctrl,
    const cuda_real_type* __restrict__ d_target,
    int n_ctrl,
    int dim_J,
    int n_entries)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_entries) return;
    const ptrdiff_t slot = d_slot[k];
    d_F[slot * dim_J + d_row[k]] = -(d_vc_q[slot * n_ctrl + d_ctrl[k]] - d_target[k]);
}

// =============================================================================
// apply_vm_reseed_kernel
// =============================================================================
__global__ void apply_vm_reseed_kernel(
          cudaComplexType* __restrict__ d_V_batch,
    const int*             __restrict__ d_slot,
    const int*             __restrict__ d_bus,
    const cuda_real_type*  __restrict__ d_vm,
    int n_entries,
    int n_bus)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_entries || d_bus[k] < 0) return;   // bus -1: an unused entry of a fixed block
    const ptrdiff_t at = static_cast<ptrdiff_t>(d_slot[k]) * n_bus + d_bus[k];
    cudaComplexType v = d_V_batch[at];
    cuda_real_type mag = CudaFunHelper::my_cuCabs(v);
    if (mag < cuda_real_type(1e-10)) {
        v   = CudaFunHelper::my_make_cuComplex(cuda_real_type(1), cuda_real_type(0));
        mag = cuda_real_type(1);
    }
    const cuda_real_type scale = d_vm[k] / mag;
    d_V_batch[at] = CudaFunHelper::my_make_cuComplex(
        CudaFunHelper::my_cuCreal(v) * scale,
        CudaFunHelper::my_cuCimag(v) * scale);
}

// =============================================================================
// gather_v_rows_kernel
// =============================================================================
__global__ void gather_v_rows_kernel(
          cudaComplexType* __restrict__ d_dst,
    const cudaComplexType* __restrict__ d_src,
    const int*             __restrict__ d_map,
    const cudaComplexType* __restrict__ d_fallback,
    int n_bus,
    int n_rows)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t r   = tid / n_bus;
    const int       b   = static_cast<int>(tid % n_bus);
    if (r >= n_rows) return;
    const ptrdiff_t src_r = d_map ? d_map[r] : r;
    const cudaComplexType v = d_src[src_r * n_bus + b];
    d_dst[r * n_bus + b] = (isfinite(CudaFunHelper::my_cuCreal(v)) && isfinite(CudaFunHelper::my_cuCimag(v)))
                         ? v : d_fallback[b];
}

// =============================================================================
// tile_vc_vset_kernel / apply_gen_vset_kernel
// =============================================================================
__global__ void tile_vc_vset_kernel(
          cuda_real_type* __restrict__ d_vset_batch,
    const cuda_real_type* __restrict__ d_vset_base,
    int n_grp,
    int batch_size)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_grp;
    const int       g   = static_cast<int>(tid % n_grp);
    if (b >= batch_size) return;
    d_vset_batch[b * n_grp + g] = d_vset_base[g];
}

__global__ void apply_gen_vset_kernel(
          cuda_real_type* __restrict__ d_vset_batch,
    const cuda_real_type* __restrict__ d_gen_v_all,
    const int*            __restrict__ d_active_group,
    int row_offset,
    int k_active,
    int actual_batch,
    int n_grp)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t r   = tid / k_active;
    const int       j   = static_cast<int>(tid % k_active);
    if (r >= actual_batch) return;
    const int g = d_active_group[j];
    if (g < 0) return;
    const cuda_real_type target =
        d_gen_v_all[static_cast<ptrdiff_t>(row_offset + r) * k_active + j];
    if (isnan(target)) return;
    d_vset_batch[r * n_grp + g] = target;
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_p;   // contingency index
    const int       k   = static_cast<int>(tid % n_p);   // P-equation slot
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_q;
    const int       k   = static_cast<int>(tid % n_q);
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
// spmv_fill_F_kernel -- see the header. One thread per (bus row, slot), a
// warp being 32 consecutive rows of one slot.
// =============================================================================
__global__ void spmv_fill_F_kernel(
          cuda_real_type*  __restrict__ d_F,
          cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Sbus,
    int sbus_stride,
    const int*             __restrict__ d_Ybus_outer,
    const int*             __restrict__ d_Ybus_inner,
    const cudaComplexType* __restrict__ d_Ybus_values_batch,
    int ybus_stride,
    const int*             __restrict__ d_p_row_of_bus,
    const int*             __restrict__ d_q_row_of_bus,
    const int*             __restrict__ d_bus_order,
          cuda_real_type*  __restrict__ d_F_raw,
    const int*             __restrict__ d_F_clean,
    int write_raw,
    int n_bus, int dim_J, int n_slots)
{
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_bus || static_cast<int>(blockIdx.y) >= n_slots) return;
    const int i = d_bus_order != nullptr ? d_bus_order[t] : t;
    const int k0 = d_Ybus_outer[i], k1 = d_Ybus_outer[i + 1];
    const int pr = d_p_row_of_bus[i], qr = d_q_row_of_bus[i];

    using H = CudaFunHelper;
    for (ptrdiff_t b = blockIdx.y; b < n_slots; b += gridDim.y) {
        // F_raw by thread: [slot][t][P, Q], coalesced whatever the bus order
        cuda_real_type* raw = d_F_raw + (b * n_bus + t) * 2;
        if (d_F_clean != nullptr && d_F_clean[b] != 0) {
            // F_raw holds this slot's rows at its current state (Ibus too)
            if (pr >= 0) d_F[b * dim_J + pr] = raw[0];
            if (qr >= 0) d_F[b * dim_J + qr] = raw[1];
            continue;
        }
        const cudaComplexType* V = d_V + b * n_bus;
        const cudaComplexType* Y = d_Ybus_values_batch + b * ybus_stride;
        cuda_real_type ir = 0, ii = 0;
        for (int k = k0; k < k1; ++k) {
            const cudaComplexType y = Y[k];
            const cudaComplexType v = V[d_Ybus_inner[k]];
            ir += H::my_cuCreal(y) * H::my_cuCreal(v) - H::my_cuCimag(y) * H::my_cuCimag(v);
            ii += H::my_cuCreal(y) * H::my_cuCimag(v) + H::my_cuCimag(y) * H::my_cuCreal(v);
        }
        d_Ibus[b * n_bus + i] = H::my_make_cuComplex(ir, ii);
        if ((pr & qr) < 0) continue;   // both < 0: no P nor Q row
        const cudaComplexType Vi = V[i];
        const cudaComplexType sb = d_Sbus[b * sbus_stride + i];
        const cuda_real_type vr = H::my_cuCreal(Vi), vi = H::my_cuCimag(Vi);
        // S = V_i * conj(I_i)
        if (pr >= 0) {
            const cuda_real_type f = -((vr * ir + vi * ii) - H::my_cuCreal(sb));
            d_F[b * dim_J + pr] = f;
            if (write_raw) raw[0] = f;
        }
        if (qr >= 0) {
            const cuda_real_type f = -((vi * ir - vr * ii) - H::my_cuCimag(sb));
            d_F[b * dim_J + qr] = f;
            if (write_raw) raw[1] = f;
        }
    }
}

// =============================================================================
// fill_J_kernel
//
// One thread per Ybus entry k (blockIdx.x), the batch slots on blockIdx.y
// (grid-stride past the 65535 limit): no 64-bit div/mod per thread, and the
// row search / map loads are done once for all the slots a thread visits.
// With t = V_i * conj(Y_ij * V_j) and inv = 1 / |V|:
//   dS/dVa = -j t,  dS/dVm = t * inv_j          (off-diagonal)
//   diagonal (j == i), with c = V_i conj(I_i): dS/dVa += j c, dS/dVm += c * inv_i
// the same derivatives as before (lightsim2grid's dSbus_dV), with 2 (resp. 3)
// complex products and one rsqrt instead of 4 products, a hypot and two
// divisions, and with the diagonal as a short added branch instead of a second
// full path (warp divergence): that arithmetic, not memory, set the cost in
// FP64 on GPUs with a reduced FP64 rate (~10x the FP32 time on an RTX A1000).
// =============================================================================
__global__ void zero_J_positions_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const int*            __restrict__ pos,
    int n_pos,
    int nnz_J,
    int actual_batch)
{
    const int a = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (a >= n_pos) return;
    const int p = pos[a];
    for (ptrdiff_t b = blockIdx.y; b < actual_batch; b += gridDim.y)
        d_J_values[b * nnz_J + p] = static_cast<cuda_real_type>(0.);
}

namespace {
__device__ __forceinline__ double j_rsqrt(double x) { return rsqrt(x); }
__device__ __forceinline__ float  j_rsqrt(float x)  { return rsqrtf(x); }
}  // namespace

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
    const int k = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;   // entry of one system
    if (k >= nnz_Y || static_cast<int>(blockIdx.y) >= actual_batch) return;

    const int m11 = d_map_j11[k], m12 = d_map_j12[k], m21 = d_map_j21[k], m22 = d_map_j22[k];
    if ((m11 & m12 & m21 & m22) < 0) return;   // all four < 0: the entry feeds no J position

    // Row recovery: binary search on the single-system outer array (once per
    // thread, whatever the number of slots it visits; a precomputed row array
    // measured no faster).
    int lo = 0, hi = n_bus;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (d_Ybus_outer[mid + 1] <= k) lo = mid + 1;
        else                            hi = mid;
    }
    const int i = lo;
    const int j = d_Ybus_inner[k];

    using H = CudaFunHelper;
    for (ptrdiff_t b = blockIdx.y; b < actual_batch; b += gridDim.y) {
        const cudaComplexType V_i  = d_V[b * n_bus + i];
        const cudaComplexType Y_ij = d_Ybus_values_batch[b * nnz_Y + k];  // patched value

        // Off-diagonal and diagonal share t = V_i conj(Y_ij V_j) (j == i on the
        // diagonal); the diagonal only adds c = V_i conj(I_i): one short
        // divergent branch instead of two full ones (a warp holds ~8 diagonal
        // entries out of 32 and ran both paths).
        const cudaComplexType V_j = d_V[b * n_bus + j];
        const cudaComplexType t = H::my_cuCmul(V_i, H::my_cuConj(H::my_cuCmul(Y_ij, V_j)));
        const cuda_real_type re = H::my_cuCreal(V_j), im = H::my_cuCimag(V_j);
        const cuda_real_type inv = j_rsqrt(re * re + im * im);
        cuda_real_type dPa =  H::my_cuCimag(t);   // Re/Im of dS/dVa = -j t (+ j c)
        cuda_real_type dQa = -H::my_cuCreal(t);
        cuda_real_type pm  =  H::my_cuCreal(t);   // dS/dVm = (t (+ c)) * inv
        cuda_real_type qm  =  H::my_cuCimag(t);
        if (i == j) {
            const cudaComplexType c = H::my_cuCmul(V_i, H::my_cuConj(d_Ibus[b * n_bus + i]));
            dPa -= H::my_cuCimag(c);
            dQa += H::my_cuCreal(c);
            pm  += H::my_cuCreal(c);
            qm  += H::my_cuCimag(c);
        }
        const cuda_real_type dPm = pm * inv, dQm = qm * inv;

        cuda_real_type* J = d_J_values + b * nnz_J;
        if (m11 >= 0) J[m11] = dPa;
        if (m12 >= 0) J[m12] = dPm;
        if (m21 >= 0) J[m21] = dQa;
        if (m22 >= 0) J[m22] = dQm;
    }
}

// The bit-pattern view compute_residuals_kernel and reduce_step_norms_kernel take their max on.
template <class T> struct ResBits;
template <> struct ResBits<double> {
    using type = unsigned long long;
    static constexpr type inf_bits = 0x7FF0000000000000ULL;
    __device__ static type abs_bits(double x) { return static_cast<type>(__double_as_longlong(x)) & 0x7FFFFFFFFFFFFFFFULL; }
    __device__ static double from_bits(type u) { return __longlong_as_double(static_cast<long long>(u)); }
};
template <> struct ResBits<float> {
    using type = unsigned int;
    static constexpr type inf_bits = 0x7F800000U;
    __device__ static type abs_bits(float x) { return __float_as_uint(x) & 0x7FFFFFFFU; }
    __device__ static float from_bits(type u) { return __uint_as_float(u); }
};

// =============================================================================
// NR step-scaling (MaxVoltageChange), mirrors lightsim2grid's own
// MaxVoltageChangeScalingPolicy (ScalingPolicies.hpp): after solving J*dx=F,
// scale the WHOLE dx vector by alpha<=1 so max|dtheta|<=max_dVa and
// max|dvm|<=max_dVm BEFORE applying it anywhere (Va/Vm and any extension
// state columns), exactly matching NRAlgo.tpp's apply_step(coeff * F). Two
// passes, both no-ops unless the caller enables scaling (see NrIterBuffers'
// own doc):
//   reduce_step_norms_kernel : per-batch-slot max|dx| restricted to the
//                              theta/vm columns (CAS-based atomic max --
//                              values are always non-negative so the plain
//                              bit-pattern comparison in atomic_max_nonneg is
//                              safe).
//   apply_step_scale_kernel  : rescales dx in place from those two maxima.
// Both run ONE BLOCK PER SLOT (grid = actual_batch blocks of BS threads), the
// threads of a block sharing the slot's entries cyclically: a Newton step is
// almost never scaled, and one thread per entry cost a CAS per entry and a
// rewrite of every dx (~40 ms of a 2.3 s run on a 7k-bus grid). Each thread
// keeps its own maxima and publishes them once; apply leaves at once when the
// slot's alpha is 1.
// =============================================================================
__global__ void reduce_step_norms_kernel(
    const cuda_real_type* __restrict__ d_dx,
    const int*             __restrict__ theta_cols,
    const int*             __restrict__ vm_cols,
    int n_theta, int n_vm,
    int dim_J, int actual_batch,
    cuda_real_type* __restrict__ d_max_dtheta,   // [actual_batch], zeroed before launch
    cuda_real_type* __restrict__ d_max_dvm)      // [actual_batch], zeroed before launch
{
    const ptrdiff_t b = blockIdx.x;  // widened; see fill_J_kernel's own note
    if (b >= actual_batch) return;
    const cuda_real_type* dx = d_dx + b * dim_J;
    // `if (!(m >= a)) m = a` (atomic_max_nonneg's own test) on the integer pipe, |v| as v's bit pattern
    // with the sign cleared (see compute_residuals_kernel): m >= a is false exactly when one of them is
    // NaN or a's pattern exceeds m's -- the same values, NaN included (a later element replaces a NaN, as
    // before), without the FP64 compares (1/64 of the FP32 rate on some GPUs).
    using U = ResBits<cuda_real_type>;
    typename U::type mt = 0, mv = 0;
    for (int k = threadIdx.x; k < n_theta; k += blockDim.x) {
        const typename U::type a = U::abs_bits(dx[theta_cols[k]]);
        if (mt > U::inf_bits || a > U::inf_bits || a > mt) mt = a;
    }
    for (int k = threadIdx.x; k < n_vm; k += blockDim.x) {
        const typename U::type a = U::abs_bits(dx[vm_cols[k]]);
        if (mv > U::inf_bits || a > U::inf_bits || a > mv) mv = a;
    }
    atomic_max_nonneg(&d_max_dtheta[b], U::from_bits(mt));
    atomic_max_nonneg(&d_max_dvm[b], U::from_bits(mv));
}

__global__ void apply_step_scale_kernel(
          cuda_real_type* __restrict__ d_dx,
    const cuda_real_type* __restrict__ d_max_dtheta,
    const cuda_real_type* __restrict__ d_max_dvm,
    cuda_real_type max_dVa, cuda_real_type max_dVm,
    int dim_J, int actual_batch)
{
    const ptrdiff_t b = blockIdx.x;  // widened; see fill_J_kernel's own note
    if (b >= actual_batch) return;

    cuda_real_type alpha = static_cast<cuda_real_type>(1.);
    const cuda_real_type mdt = d_max_dtheta[b];
    if (mdt > max_dVa) alpha = alpha < (max_dVa / mdt) ? alpha : (max_dVa / mdt);
    const cuda_real_type mdv = d_max_dvm[b];
    if (mdv > max_dVm) alpha = alpha < (max_dVm / mdv) ? alpha : (max_dVm / mdv);
    if (alpha == static_cast<cuda_real_type>(1.)) return;  // the usual case

    cuda_real_type* dx = d_dx + b * dim_J;
    for (int j = threadIdx.x; j < dim_J; j += blockDim.x) dx[j] *= alpha;
}

// =============================================================================
// update_Va_kernel / update_Vm_kernel
//
// V is kept as a complex number, so a step in polar coordinates is applied as
// a product, never through atan2 / |V| / sin / cos of the full angle:
//   Va += dva  <=>  V *= exp(j dva)          (one sincos of the small step)
//   Vm += dvm  <=>  V *= 1 + dvm / |V|       (one rsqrt of |V|^2)
// The former formulation (atan2 + hypot + sin + cos, in each of the two
// kernels) was limited by FP64 arithmetic on GPUs with a reduced FP64 rate:
// ~10x slower than FP32 on an RTX A1000, where every other kernel of the
// Newton loop is ~2x (memory). Same values to rounding; a zero step leaves
// V bit-identical (masked buses, identity rows).
// =============================================================================
namespace {
// |x| <= 0.25 rad -- a Newton angle step, nearly always -- by the Taylor series to x^11 / x^12 (truncation
// ~1e-17 relative, below half an ulp; Horner with FMAs, ~13 FP64 operations against ~35 for the library's
// argument reduction and polynomials); larger steps by sincos. Both ~1 ulp: V to rounding.
__device__ __forceinline__ void v_sincos(double x, double* s, double* c)
{
    if (fabs(x) <= 0.25) {
        const double x2 = x * x;
        double ps = -1.0 / 39916800.0;           // -1/11!
        ps = fma(ps, x2, 1.0 / 362880.0);         //  1/9!
        ps = fma(ps, x2, -1.0 / 5040.0);          // -1/7!
        ps = fma(ps, x2, 1.0 / 120.0);            //  1/5!
        ps = fma(ps, x2, -1.0 / 6.0);             // -1/3!
        *s = fma(ps * x2, x, x);
        double pc = 1.0 / 479001600.0;            //  1/12!
        pc = fma(pc, x2, -1.0 / 3628800.0);       // -1/10!
        pc = fma(pc, x2, 1.0 / 40320.0);          //  1/8!
        pc = fma(pc, x2, -1.0 / 720.0);           // -1/6!
        pc = fma(pc, x2, 1.0 / 24.0);             //  1/4!
        pc = fma(pc, x2, -0.5);                   // -1/2!
        *c = fma(pc, x2, 1.0);
        return;
    }
    sincos(x, s, c);
}
__device__ __forceinline__ void v_sincos(float x, float* s, float* c)    { sincosf(x, s, c); }
__device__ __forceinline__ double v_rsqrt(double x) { return rsqrt(x); }
__device__ __forceinline__ float  v_rsqrt(float x)  { return rsqrtf(x); }
}  // namespace

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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_theta;
    const int       k   = static_cast<int>(tid % n_theta);
    if (b >= actual_batch) return;

    const cuda_real_type dva = d_dx[b * dim_J + theta_cols[k]];
    if (dva == static_cast<cuda_real_type>(0)) return;
    cuda_real_type s, c;
    v_sincos(dva, &s, &c);
    cudaComplexType& V = d_V[b * n_bus + theta_buses[k]];
    V = CudaFunHelper::my_cuCmul(V, CudaFunHelper::my_make_cuComplex(c, s));
}

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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_vm;
    const int       k   = static_cast<int>(tid % n_vm);
    if (b >= actual_batch) return;

    const cuda_real_type dvm = d_dx[b * dim_J + vm_cols[k]];
    if (dvm == static_cast<cuda_real_type>(0)) return;
    cudaComplexType& V = d_V[b * n_bus + vm_buses[k]];
    const cuda_real_type re = CudaFunHelper::my_cuCreal(V);
    const cuda_real_type im = CudaFunHelper::my_cuCimag(V);
    const cuda_real_type n2 = re * re + im * im;
    if (n2 == static_cast<cuda_real_type>(0)) {
        // |V| = 0: no angle to keep; the former atan2(0, 0) = 0 gave V = dvm.
        V = CudaFunHelper::my_make_cuComplex(dvm, static_cast<cuda_real_type>(0));
        return;
    }
    const cuda_real_type scale = static_cast<cuda_real_type>(1) + dvm * v_rsqrt(n2);
    V = CudaFunHelper::my_make_cuComplex(scale * re, scale * im);
}

// update_Va_kernel then update_Vm_kernel for one (slot, bus), V kept in registers between them: the
// same operations on the same values, one read and one write of V instead of two each.
__global__ void update_V_fused_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ theta_col_of_bus,
    const int*             __restrict__ vm_col_of_bus,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_bus;
    const int       i   = static_cast<int>(tid - b * n_bus);
    if (b >= actual_batch) return;
    const int tc = theta_col_of_bus[i], vc = vm_col_of_bus[i];
    const cuda_real_type zero = static_cast<cuda_real_type>(0);
    const cuda_real_type dva = tc >= 0 ? d_dx[b * dim_J + tc] : zero;
    const cuda_real_type dvm = vc >= 0 ? d_dx[b * dim_J + vc] : zero;
    if (dva == zero && dvm == zero) return;
    cudaComplexType V = d_V[b * n_bus + i];
    if (dva != zero) {
        cuda_real_type s, c;
        v_sincos(dva, &s, &c);
        V = CudaFunHelper::my_cuCmul(V, CudaFunHelper::my_make_cuComplex(c, s));
    }
    if (dvm != zero) {
        const cuda_real_type re = CudaFunHelper::my_cuCreal(V);
        const cuda_real_type im = CudaFunHelper::my_cuCimag(V);
        const cuda_real_type n2 = re * re + im * im;
        if (n2 == zero) {
            V = CudaFunHelper::my_make_cuComplex(dvm, zero);   // as update_Vm_kernel
        } else {
            const cuda_real_type scale = static_cast<cuda_real_type>(1) + dvm * v_rsqrt(n2);
            V = CudaFunHelper::my_make_cuComplex(scale * re, scale * im);
        }
    }
    d_V[b * n_bus + i] = V;
}

// =============================================================================
// MultiSlack kernels (Phase 2)
// =============================================================================
__global__ void adjust_slack_mismatch_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_slack_absorbed,
    const int*            __restrict__ d_slack_prow,
    const cuda_real_type* __restrict__ d_slack_w,
    int w_stride,
    int n_slack,
    int dim_J,
    int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_slack;
    const int       k   = static_cast<int>(tid % n_slack);
    if (b >= actual_batch) return;
    // mis += sa·weight  ⇒  residual d_F = −real(mis) gains −sa·weight
    // w_stride == 0: shared weights; == n_slack: this slot's own weights.
    d_F[b * dim_J + d_slack_prow[k]] -= d_slack_absorbed[b] * d_slack_w[b * w_stride + k];
}

__global__ void fill_slack_feature_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const int*            __restrict__ d_slack_feat_pos,
    const cuda_real_type* __restrict__ d_slack_w,
    int w_stride,
    int n_slack,
    int nnz_J,
    int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note -- this kernel
    // is also reused for VC feature stamping (fill_slack_feature_kernel is
    // called with VC's flat pos/value pairs too, w_stride 0), so the
    // nnz_J-scaled offset below is exactly as much at risk as fill_J_kernel's
    // own J_base.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_slack;
    const int       k   = static_cast<int>(tid % n_slack);
    if (b >= actual_batch) return;
    d_J_values[b * nnz_J + d_slack_feat_pos[k]] = d_slack_w[b * w_stride + k];
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
    // b widened to ptrdiff_t: one thread per batch slot (no division), but
    // b * dim_J below is still the same at-risk product as elsewhere in this
    // file once actual_batch * dim_J grows large.
    const ptrdiff_t b = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= actual_batch) return;
    d_slack_absorbed[b] += d_dx[b * dim_J + slack_col];
}

// =============================================================================
// HVDC angle-droop kernels (Phase 3)
// =============================================================================

// hvdc_recv_pu / hvdc_flows_pu live in acpf_nr_kernels.cuh (shared with the
// droop P-saturation check, violation_kernels.cu).

__global__ void hvdc_adjust_mismatch_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ bus1,
    const int*             __restrict__ bus2,
    const int*             __restrict__ status,
    int                                 status_stride,   // 0: shared, n_hvdc: per slot
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_hvdc;
    const int       e   = static_cast<int>(tid % n_hvdc);
    if (b >= actual_batch) return;

    const cudaComplexType V1 = d_V[b * n_bus + bus1[e]];
    const cudaComplexType V2 = d_V[b * n_bus + bus2[e]];
    const cuda_real_type th1 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
    const cuda_real_type th2 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
    const cuda_real_type raw = p0[e] + k[e] * (th1 - th2);

    cuda_real_type p1_flow, p2_flow;
    hvdc_flows_pu(status[b * status_stride + e], raw, lf1[e], lf2[e], r[e], pmax12[e], pmax21[e],
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
    int                                 status_stride,   // 0: shared, n_hvdc: per slot
    const cuda_real_type*  __restrict__ p0,
    const cuda_real_type*  __restrict__ k,
    const cuda_real_type*  __restrict__ lf1,
    const cuda_real_type*  __restrict__ lf2,
    const cuda_real_type*  __restrict__ r,
    const int*             __restrict__ h11,
    const int*             __restrict__ h12,
    const int*             __restrict__ h21,
    const int*             __restrict__ h22,
    int n_hvdc,
    int n_bus,
    int nnz_J,
    int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_hvdc;
    const int       e   = static_cast<int>(tid % n_hvdc);
    if (b >= actual_batch) return;
    if (status[b * status_stride + e] != 0) return;   // saturated: constant injection, zero slopes

    const cudaComplexType V1 = d_V[b * n_bus + bus1[e]];
    const cudaComplexType V2 = d_V[b * n_bus + bus2[e]];
    const cuda_real_type th1 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
    const cuda_real_type th2 = CudaFunHelper::my_atan2(
        CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
    const cuda_real_type raw = p0[e] + k[e] * (th1 - th2);
    const cuda_real_type one = static_cast<cuda_real_type>(1.);
    const bool side1_ctrl = raw >= static_cast<cuda_real_type>(0.);
    // Exact derivative of hvdc_recv_pu w.r.t. the controller flow |raw|:
    //   recv = (1-lf_recv)·(line_in - r·line_in²),  line_in = (1-lf_ctrl)·|raw|
    //   recv' = (1-lf_recv)(1-lf_ctrl)·(1 - 2·r·line_in)
    const cuda_real_type lf_ctrl = side1_ctrl ? lf1[e] : lf2[e];
    const cuda_real_type line_in = (one - lf_ctrl) * (side1_ctrl ? raw : -raw);
    const cuda_real_type recv_slope =
        (one - lf1[e]) * (one - lf2[e]) * (one - static_cast<cuda_real_type>(2.) * r[e] * line_in);
    // dp1 = dp_side1/dtheta1, dp2 = dp_side2/dtheta1; d/dtheta2 = -d/dtheta1.
    // Controller side: p = ±raw, slope ±k. Receiving side: p = -recv(|raw|),
    // and d|raw|/dtheta1 = ±k, so its slope is ∓k·recv'.
    const cuda_real_type dp1 = side1_ctrl ?  k[e] :  k[e] * recv_slope;
    const cuda_real_type dp2 = side1_ctrl ? -k[e] * recv_slope : -k[e];

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
    const cuda_real_type* __restrict__ d_vc_qoff,
    int n_ctrl,
    int dim_J,
    int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_ctrl;
    const int       j   = static_cast<int>(tid % n_ctrl);
    if (b >= actual_batch) return;
    // mis(c.bus) -= i·(Q_c - q_off_c)  ⇒  residual d_F[q_row] += Q_c - q_off_c: a held
    // controller's frozen output is already in Sbus
    const cuda_real_type off = d_vc_qoff ? d_vc_qoff[j] : cuda_real_type(0);
    atomic_add_real(&d_F[b * dim_J + d_vc_qrow[j]], d_vc_q[b * n_ctrl + j] - off);
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
    int vset_stride,
    int n_grp,
    int n_ctrl,
    int n_bus,
    int dim_J,
    int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_grp;
    const int       g   = static_cast<int>(tid % n_grp);
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
    const cuda_real_type vset = d_vc_vset[b * vset_stride + g];
    d_F[b * dim_J + d_vc_vrow[g]] = -(vm + slope_term - vset);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_share;
    const int       s   = static_cast<int>(tid % n_share);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_ctrl;
    const int       j   = static_cast<int>(tid % n_ctrl);
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
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active)
{
    // One block handles one contingency. b widened to ptrdiff_t: blockIdx.x
    // itself is a valid small block index, but b * dim_J below is the same
    // at-risk product as fill_J_kernel's own J_base once actual_batch *
    // dim_J grows large.
    const ptrdiff_t b = blockIdx.x;
    if (b >= actual_batch) return;
    // The slot's active row (d_slot_active, -1 = skip: the whole block returns)
    // or the chunk's c_start + b.
    const int a = d_slot_active ? d_slot_active[b] : static_cast<int>(c_start + b);
    if (a < 0) return;

    extern __shared__ cuda_real_type sdata[];

    // ‖F_b‖∞ on the integer pipe (FP64 runs at 1/64 of FP32 on some GPUs; this kernel was FP64-bound):
    // |v|'s bit pattern is v's with the sign bit cleared, non-negative reals order as their patterns, and
    // a NaN's pattern exceeds +inf's -- so the max of the patterns is max|v| when no element is NaN, and
    // above +inf's when some is. NaN must PROPAGATE: a slot whose F is NaN somewhere (e.g. a NaN cuDSS
    // solve poisoning V) reports NaN, never a finite residual that would look converged. The same values
    // as the former floating-point max (-0 and +0 alike give +0; a NaN anywhere gives NaN).
    using U = ResBits<cuda_real_type>;
    const cuda_real_type* F_b = d_F + b * dim_J;
    typename U::type m = 0;
    for (int i = threadIdx.x; i < dim_J; i += blockDim.x) {
        const typename U::type u = U::abs_bits(F_b[i]);
        m = u > m ? u : m;
    }
    typename U::type* su = reinterpret_cast<typename U::type*>(sdata);   // the same size as a real
    su[threadIdx.x] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const typename U::type o = su[threadIdx.x + stride];
            if (o > su[threadIdx.x]) su[threadIdx.x] = o;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int out = d_result_map ? d_result_map[a] : a;
        d_residuals[out] = su[0] > U::inf_bits ? cuda_real_type(NAN) : U::from_bits(su[0]);
    }
}