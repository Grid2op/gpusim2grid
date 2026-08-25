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
    // tid/b widened to ptrdiff_t: see fill_J_kernel's own note. Here
    // actual_batch * n_branches (this launch's thread count) and b * n_bus /
    // out_c * n_branches (the offsets below) are the at-risk products.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_branches;   // contingency index in batch
    const int        l  = static_cast<int>(tid % n_branches);   // branch index
    if (b >= actual_batch) return;

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

    // I_or = yff * Vi + yft * Vj  (origin / from-bus terminal current)
    const cudaComplexType I_or = (bf >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_yff[l], Vi),
        CudaFunHelper::my_cuCmul(d_yft[l], Vj)) : CudaFunHelper::my_make_cuComplex(0., 0.);

    // I_ex = ytf * Vi + ytt * Vj  (extremity / to-bus terminal current)
    const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(d_ytf[l], Vi),
        CudaFunHelper::my_cuCmul(d_ytt[l], Vj)) : CudaFunHelper::my_make_cuComplex(0., 0.);

    // Map the chunk-relative slot to its original result index (identity when
    // d_result_map is null, e.g. the full-batch session call or injection).
    // out_c itself (a contingency index < n_contingencies) fits int32; only
    // out_c * n_branches (the flat offset) needs the 64-bit product.
    const int out_c   = d_result_map ? d_result_map[c_start + b] : static_cast<int>(c_start + b);
    const ptrdiff_t out_idx = static_cast<ptrdiff_t>(out_c) * n_branches + l;
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
    // tid/total/local_c widened to ptrdiff_t -- actual_batch * n_bus (this
    // launch's thread count) and local_c/out_c * n_bus (the offsets below)
    // are the at-risk products; see fill_J_kernel's own note.
    const ptrdiff_t tid   = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t total = static_cast<ptrdiff_t>(actual_batch) * n_bus;
    if (tid >= total) return;

    const ptrdiff_t local_c = tid / n_bus;   // active-slot index in this chunk
    const int       bus     = static_cast<int>(tid % n_bus);
    const int out_c   = d_result_map[c_start + local_c];   // original index
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
    // tid/b promoted to 64-bit: actual_batch * nnz_Y (the total thread count
    // for this launch) can exceed INT32_MAX/UINT32_MAX on large grids well
    // before it exceeds any CUDA hardware grid-size limit, and blockIdx.x /
    // blockDim.x / threadIdx.x are native 32-bit CUDA types -- tid itself
    // would silently wrap before ever reaching the b*nnz_J offset below.
    // Every downstream `b * <stride>` (n_bus, nnz_Y, nnz_J) promotes to
    // ptrdiff_t automatically once b is ptrdiff_t, so nothing past this line
    // needs its own cast. k stays plain int: it indexes single-system arrays
    // of width nnz_Y, which fits int32 by construction.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / nnz_Y;   // contingency index
    const int       k   = static_cast<int>(tid % nnz_Y);   // nnz slot within one system
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
    // b is already ptrdiff_t (see tid/b above), so this product is computed
    // in 64-bit -- no separate cast needed.
    const ptrdiff_t J_base = b * nnz_J;
    if (d_map_j11[k] >= 0) d_J_values[J_base + d_map_j11[k]] = CudaFunHelper::my_cuCreal(dSdVa);
    if (d_map_j12[k] >= 0) d_J_values[J_base + d_map_j12[k]] = CudaFunHelper::my_cuCreal(dSdVm);
    if (d_map_j21[k] >= 0) d_J_values[J_base + d_map_j21[k]] = CudaFunHelper::my_cuCimag(dSdVa);
    if (d_map_j22[k] >= 0) d_J_values[J_base + d_map_j22[k]] = CudaFunHelper::my_cuCimag(dSdVm);
}

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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = n_theta + n_vm;
    const ptrdiff_t b = tid / total;
    if (b >= actual_batch) return;
    const int k = static_cast<int>(tid % total);
    if (k < n_theta) {
        const cuda_real_type v = d_dx[b * dim_J + theta_cols[k]];
        atomic_max_nonneg(&d_max_dtheta[b], v < 0 ? -v : v);
    } else {
        const cuda_real_type v = d_dx[b * dim_J + vm_cols[k - n_theta]];
        atomic_max_nonneg(&d_max_dvm[b], v < 0 ? -v : v);
    }
}

__global__ void apply_step_scale_kernel(
          cuda_real_type* __restrict__ d_dx,
    const cuda_real_type* __restrict__ d_max_dtheta,
    const cuda_real_type* __restrict__ d_max_dvm,
    cuda_real_type max_dVa, cuda_real_type max_dVm,
    int dim_J, int actual_batch)
{
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / dim_J;
    if (b >= actual_batch) return;
    const int j = static_cast<int>(tid % dim_J);

    cuda_real_type alpha = static_cast<cuda_real_type>(1.);
    const cuda_real_type mdt = d_max_dtheta[b];
    if (mdt > max_dVa) alpha = alpha < (max_dVa / mdt) ? alpha : (max_dVa / mdt);
    const cuda_real_type mdv = d_max_dvm[b];
    if (mdv > max_dVm) alpha = alpha < (max_dVm / mdv) ? alpha : (max_dVm / mdv);

    d_dx[b * dim_J + j] *= alpha;
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_theta;
    const int       k   = static_cast<int>(tid % n_theta);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_vm;
    const int       k   = static_cast<int>(tid % n_vm);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_slack;
    const int       k   = static_cast<int>(tid % n_slack);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note -- this kernel
    // is also reused for VC feature stamping (fill_slack_feature_kernel is
    // called with VC's flat pos/value pairs too), so the nnz_J-scaled offset
    // below is exactly as much at risk as fill_J_kernel's own J_base.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_slack;
    const int       k   = static_cast<int>(tid % n_slack);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_hvdc;
    const int       e   = static_cast<int>(tid % n_hvdc);
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
    // tid/b widened to ptrdiff_t; see fill_J_kernel's own note.
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t b   = tid / n_ctrl;
    const int       j   = static_cast<int>(tid % n_ctrl);
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
    const int* __restrict__ d_result_map)
{
    // One block handles one contingency. b widened to ptrdiff_t: blockIdx.x
    // itself is a valid small block index, but b * dim_J below is the same
    // at-risk product as fill_J_kernel's own J_base once actual_batch *
    // dim_J grows large.
    const ptrdiff_t b = blockIdx.x;
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
        const int out = d_result_map ? d_result_map[c_start + b] : static_cast<int>(c_start + b);
        d_residuals[out] = sdata[0];
    }
}