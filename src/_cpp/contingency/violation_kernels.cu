// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/violation_kernels.cu
// =============================================================================

#include "violation_kernels.cuh"

#include <math.h>   // isnan

namespace {

// Element-type / violation-type raw codes, mirrored from
// contingency/limit_violation_types.hpp (kept as plain ints here since device
// code writes them straight into int output buffers; see that header for the
// canonical enum this must stay in sync with).
constexpr int ELEM_BUS   = 0;
constexpr int ELEM_LINE  = 1;
constexpr int ELEM_TRAFO = 2;
constexpr int ELEM_GRID  = 3;

constexpr int VIOL_LOW_VOLTAGE  = 0;
constexpr int VIOL_HIGH_VOLTAGE = 1;
constexpr int VIOL_CURRENT      = 2;
// VIOL_NOT_SIMULATED (=3) is never written by this kernel -- it's reserved
// for contingencies the pre-check dropped before the chunk loop ever reached
// this kernel at all; see get_violations() (contingency_analysis/__init__.py)
// and BatchPfDriver's d_violation_count -1 sentinel.
constexpr int VIOL_DIVERGENCE   = 4;
// compute_physical_violations (check_bus_q_violations_kernel)
constexpr int VIOL_LOW_Q        = 5;
constexpr int VIOL_HIGH_Q       = 6;
// compute_physical_violations (check_hvdc_p_violations_kernel) writes no type
// code at all: every record it emits is element type HVDC (=4), violation type
// HVDC_P_SATURATION (=7), stamped by the Python session layer.

// Row gate shared by the two post-solve "physical" checks: a row the solver ran
// on but that did not converge gets an EMPTY report (upstream parity), never a
// record. nullptr residuals = no gate (the base-case "n" report).
__device__ __forceinline__ bool row_not_converged(const cuda_real_type* d_residuals,
                                                  int out_c, cuda_real_type residual_tol)
{
    if (d_residuals == nullptr) return false;
    const cuda_real_type r = d_residuals[out_c];
    return isnan(r) || r > residual_tol;
}

}  // namespace

__global__ void check_limit_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type          tol,
    const cuda_real_type*  __restrict__ d_bus_vn_kv,
    const cuda_real_type*  __restrict__ d_bus_vmin_kv,
    const cuda_real_type*  __restrict__ d_bus_vmax_kv,
    const int*             __restrict__ d_branch_from,
    const int*             __restrict__ d_branch_to,
    const cudaComplexType* __restrict__ d_yff_eff,
    const cudaComplexType* __restrict__ d_yft_eff,
    const cudaComplexType* __restrict__ d_ytf_eff,
    const cudaComplexType* __restrict__ d_ytt_eff,
    const cuda_real_type*  __restrict__ d_base_current_A,
    const cuda_real_type*  __restrict__ d_branch_limit_a1_ka,
    const cuda_real_type*  __restrict__ d_branch_limit_a2_ka,
    const int*             __restrict__ d_trip_start,
    const int*             __restrict__ d_trip_count,
    const int*             __restrict__ d_trip_branch_flat,
    int n_bus, int n_branches, int n_lines,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
          int*             __restrict__ d_out_element_type,
          int*             __restrict__ d_out_element_id,
          int*             __restrict__ d_out_side,
          int*             __restrict__ d_out_type,
          cuda_real_type*  __restrict__ d_out_value,
          cuda_real_type*  __restrict__ d_out_limit,
          int*             __restrict__ d_out_count,
          int*             __restrict__ d_out_truncated,
          int*             __restrict__ d_out_count_low_voltage,
          int*             __restrict__ d_out_count_high_voltage,
          int*             __restrict__ d_out_count_current)
{
    // local_c widened to ptrdiff_t: local_c * n_bus below (the d_V offset) is
    // the same at-risk product as fill_J_kernel's own J_base once
    // actual_batch * n_bus grows large (see acpf_nr_kernels.cu's note). base
    // (out_c * K, the per-contingency output slice offset) gets the same
    // treatment below.
    const ptrdiff_t local_c = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = c_start + static_cast<int>(local_c);   // GLOBAL active-slot id (tripped-branch table)
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    // base widened to ptrdiff_t: out_c * K (this contingency's output slice
    // offset) is the same at-risk product as fill_J_kernel's own J_base once
    // n_contingencies * K grows large.
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * K;
    int  cnt       = 0;      // DETAIL records written so far, capped at K
    bool truncated = false;
    // TRUE, uncapped per-type totals -- incremented on every violation found,
    // independent of whether a detail record could still be written.
    int n_low = 0, n_high = 0, n_current = 0;

    // Local (single-thread-owned, no atomics needed) push into this
    // contingency's exclusive output slice. Silently stops writing DETAIL
    // records past K (sets `truncated`); callers still bump the relevant
    // n_low/n_high/n_current counter themselves regardless of this return.
    auto push = [&](int etype, int eid, int side, int vtype,
                    cuda_real_type value, cuda_real_type limit) {
        if (cnt >= K) { truncated = true; return; }
        d_out_element_type[base + cnt] = etype;
        d_out_element_id[base + cnt]   = eid;
        d_out_side[base + cnt]         = side;
        d_out_type[base + cnt]         = vtype;
        d_out_value[base + cnt]        = value;
        d_out_limit[base + cnt]        = limit;
        ++cnt;
    };

    // ---- 1. DIVERGENCE --------------------------------------------------
    // V is unreliable for a diverged system; do not scan buses/branches on it.
    // isnan(residual) is included alongside residual > tol: this kernel only
    // ever runs for a contingency that WAS actually solved (NR iterations
    // executed in _solve_chunk), so any unusable residual here -- whether a
    // large finite value or NaN -- means the solver ran and failed, i.e.
    // DIVERGENCE, never NOT_SIMULATED (that code covers a contingency
    // dropped before it ever reached this kernel, handled at the Python
    // session layer instead -- see this file's own top-of-file note).
    const cuda_real_type residual = d_residuals[out_c];
    if (isnan(residual) || residual > tol) {
        push(ELEM_GRID, -1, 0, VIOL_DIVERGENCE, residual, tol);
        d_out_count[out_c]              = cnt;
        d_out_truncated[out_c]          = 0;
        d_out_count_low_voltage[out_c]  = 0;
        d_out_count_high_voltage[out_c] = 0;
        d_out_count_current[out_c]      = 0;
        return;
    }

    // ---- 2. Branch current (checked first: thermal/current violations are
    // generally first-order operational concerns, voltage second-order) ----
    // Scans every branch regardless of `truncated` so n_current stays exact
    // even once the K-slot detail buffer is full.
    const int t_start = d_trip_start ? d_trip_start[slot_global] : 0;
    const int t_count = d_trip_count ? d_trip_count[slot_global] : 0;
    for (int l = 0; l < n_branches; ++l) {
        bool tripped = false;
        for (int ti = 0; ti < t_count; ++ti)
            if (d_trip_branch_flat[t_start + ti] == l) { tripped = true; break; }
        if (tripped) continue;   // Ybus was patched but yff_eff/yft_eff/ytf_eff/ytt_eff weren't -- would report a phantom current

        const cuda_real_type lim1 = d_branch_limit_a1_ka[l];
        const cuda_real_type lim2 = d_branch_limit_a2_ka[l];
        if (isnan(lim1) && isnan(lim2)) continue;   // no limit configured for this branch

        // branch_from/branch_to are in AC-solver bus numbering (see
        // concat_busids_to_solver, ls2g_bridge.cpp): a side that lightsim2grid
        // Kron-reduced away (isolated / half-open line, keep_half_open_lines)
        // is relabeled to -1. That bus has no voltage in the solved system --
        // treat it as V=0 -- and no terminal current on that side to report --
        // 0, not computed from the pi-model formula, since the terminal
        // itself doesn't exist. Reading d_V[... + (-1)] without this guard is
        // an out-of-bounds read (confirmed via compute-sanitizer on a real
        // half-open-line grid).
        const int bf = d_branch_from[l];
        const int bt = d_branch_to[l];
        const cudaComplexType Vi = (bf >= 0) ? d_V[local_c * n_bus + bf] : CudaFunHelper::my_make_cuComplex(0., 0.);
        const cudaComplexType Vj = (bt >= 0) ? d_V[local_c * n_bus + bt] : CudaFunHelper::my_make_cuComplex(0., 0.);
        const cudaComplexType I_or = (bf >= 0) ? CudaFunHelper::my_cuCadd(
            CudaFunHelper::my_cuCmul(d_yff_eff[l], Vi), CudaFunHelper::my_cuCmul(d_yft_eff[l], Vj))
            : CudaFunHelper::my_make_cuComplex(0., 0.);
        const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
            CudaFunHelper::my_cuCmul(d_ytf_eff[l], Vi), CudaFunHelper::my_cuCmul(d_ytt_eff[l], Vj))
            : CudaFunHelper::my_make_cuComplex(0., 0.);
        // *0.001: gpusim2grid's d_base_current_A is Amps-based; limits are kA.
        const cuda_real_type ka_or = CudaFunHelper::my_cuCabs(I_or) * d_base_current_A[l] * cuda_real_type(0.001);
        const cuda_real_type ka_ex = CudaFunHelper::my_cuCabs(I_ex) * d_base_current_A[l] * cuda_real_type(0.001);

        const int etype = (l < n_lines) ? ELEM_LINE : ELEM_TRAFO;
        const int eid    = (l < n_lines) ? l : (l - n_lines);
        if (!isnan(lim1) && ka_or > lim1) { ++n_current; push(etype, eid, 1, VIOL_CURRENT, ka_or, lim1); }
        if (!isnan(lim2) && ka_ex > lim2) { ++n_current; push(etype, eid, 2, VIOL_CURRENT, ka_ex, lim2); }
    }

    // ---- 3. Bus voltage --------------------------------------------------
    // Scans every bus regardless of `truncated`, same reason as above.
    for (int b = 0; b < n_bus; ++b) {
        const cuda_real_type vmin = d_bus_vmin_kv[b];
        const cuda_real_type vmax = d_bus_vmax_kv[b];
        if (isnan(vmin) && isnan(vmax)) continue;   // no limit configured for this bus

        const cudaComplexType Vb = d_V[local_c * n_bus + b];
        // vm_kv is NaN when Vb is NaN (masked bus, written by nr_mask_v_nan
        // before this kernel launches) -- NaN comparisons below are false in
        // IEEE754, so masked buses are excluded for free.
        const cuda_real_type vm_kv = CudaFunHelper::my_cuCabs(Vb) * d_bus_vn_kv[b];
        if (!isnan(vmin) && vm_kv < vmin) {
            ++n_low;
            push(ELEM_BUS, b, 0, VIOL_LOW_VOLTAGE, vm_kv, vmin);
        } else if (!isnan(vmax) && vm_kv > vmax) {
            ++n_high;
            push(ELEM_BUS, b, 0, VIOL_HIGH_VOLTAGE, vm_kv, vmax);
        }
    }

    d_out_count[out_c]              = cnt;
    d_out_truncated[out_c]          = truncated ? 1 : 0;
    d_out_count_low_voltage[out_c]  = n_low;
    d_out_count_high_voltage[out_c] = n_high;
    d_out_count_current[out_c]      = n_current;
}


// =============================================================================
// check_bus_q_violations_kernel -- see violation_kernels.cuh
// =============================================================================
__global__ void check_bus_q_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Yvals,
    const int*             __restrict__ d_Y_outer,
    const int*             __restrict__ d_Y_inner,
    const cudaComplexType* __restrict__ d_Sbus,
    int                                 sbus_stride,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type                      residual_tol,
    int                                 n_check,
    const int*             __restrict__ d_bus_solver,
    const cuda_real_type*  __restrict__ d_qmin_fixed,
    const cuda_real_type*  __restrict__ d_qmax_fixed,
    const int*             __restrict__ d_n_fixed,
    const cuda_real_type*  __restrict__ d_bmin_sum,
    const cuda_real_type*  __restrict__ d_bmax_sum,
    const int*             __restrict__ d_gen_start,
    const int*             __restrict__ d_gen_id,
    const cuda_real_type*  __restrict__ d_gen_qmin,
    const cuda_real_type*  __restrict__ d_gen_qmax,
    const unsigned char*   __restrict__ d_gen_off,
    int                                 n_gen,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_mvar,
    int n_bus, int nnz_Y,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
          int*             __restrict__ d_out_bus_id,
          int*             __restrict__ d_out_type,
          cuda_real_type*  __restrict__ d_out_value,
          cuda_real_type*  __restrict__ d_out_limit,
          int*             __restrict__ d_out_count,
          int*             __restrict__ d_out_truncated)
{
    // local_c / base widened to ptrdiff_t: see check_limit_violations_kernel.
    const ptrdiff_t local_c = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = c_start + static_cast<int>(local_c);
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * K;

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V  = d_V     + local_c * n_bus;
    const cudaComplexType* Yv = d_Yvals + local_c * nnz_Y;
    const cudaComplexType* Sb = d_Sbus  + local_c * sbus_stride;
    const unsigned char*   off = d_gen_off ? d_gen_off + static_cast<ptrdiff_t>(out_c) * n_gen : nullptr;

    int  cnt       = 0;
    bool truncated = false;
    auto push = [&](int bus, int vtype, cuda_real_type value, cuda_real_type limit) {
        if (cnt >= K) { truncated = true; return; }
        d_out_bus_id[base + cnt] = bus;
        d_out_type[base + cnt]   = vtype;
        d_out_value[base + cnt]  = value;
        d_out_limit[base + cnt]  = limit;
        ++cnt;
    };
    auto finite_or_zero = [](cudaComplexType v) -> cudaComplexType {
        if (!isfinite(v.x) || !isfinite(v.y))
            return CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
        return v;
    };

    for (int kk = 0; kk < n_check; ++kk) {
        const int b = d_bus_solver[kk];
        const cudaComplexType Vb = V[b];
        if (!isfinite(Vb.x) || !isfinite(Vb.y)) continue;   // masked (stranded) bus

        // ---- what this bus' machines can produce, this row --------------
        int nb_live = d_n_fixed[kk];
        cuda_real_type q_min = d_qmin_fixed[kk];
        cuda_real_type q_max = d_qmax_fixed[kk];
        for (int p = d_gen_start[kk]; p < d_gen_start[kk + 1]; ++p) {
            if (off != nullptr && off[d_gen_id[p]]) continue;   // disconnected by this row
            ++nb_live;
            q_min += d_gen_qmin[p];
            q_max += d_gen_qmax[p];
        }
        const cuda_real_type v2_sn = (Vb.x * Vb.x + Vb.y * Vb.y) * sn_mva;
        q_min += d_bmin_sum[kk] * v2_sn;
        q_max += d_bmax_sum[kk] * v2_sn;
        if (nb_live == 0) continue;   // nothing holds this bus in this row

        // ---- ... and what it had to produce: the raw reactive residual ---
        cudaComplexType Ib = CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
        for (int p = d_Y_outer[b]; p < d_Y_outer[b + 1]; ++p) {
            const cudaComplexType Vj = finite_or_zero(V[d_Y_inner[p]]);
            Ib = CudaFunHelper::my_cuCadd(Ib, CudaFunHelper::my_cuCmul(Yv[p], Vj));
        }
        const cudaComplexType S = CudaFunHelper::my_cuCmul(Vb, CudaFunHelper::my_cuConj(Ib));
        const cuda_real_type q_bus = (S.y - Sb[b].y) * sn_mva;
        if (!isfinite(q_bus)) continue;

        if (isfinite(q_min) && q_bus < q_min - tol_mvar) {
            push(b, VIOL_LOW_Q, q_bus, q_min);
        } else if (isfinite(q_max) && q_bus > q_max + tol_mvar) {
            push(b, VIOL_HIGH_Q, q_bus, q_max);
        }
    }

    d_out_count[out_c]     = cnt;
    d_out_truncated[out_c] = truncated ? 1 : 0;
}

// =============================================================================
// check_hvdc_p_violations_kernel -- see violation_kernels.cuh
// =============================================================================
__global__ void check_hvdc_p_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type                      residual_tol,
    int                                 n_hvdc,
    const int*             __restrict__ d_bus1,
    const int*             __restrict__ d_bus2,
    const int*             __restrict__ d_status,
    const cuda_real_type*  __restrict__ d_p0,
    const cuda_real_type*  __restrict__ d_k,
    const cuda_real_type*  __restrict__ d_lf1,
    const cuda_real_type*  __restrict__ d_lf2,
    const cuda_real_type*  __restrict__ d_r,
    const cuda_real_type*  __restrict__ d_pmax12,
    const cuda_real_type*  __restrict__ d_pmax21,
    const int*             __restrict__ d_hvdc_id,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_pu,
    int n_bus,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
          int*             __restrict__ d_out_hvdc_id,
          int*             __restrict__ d_out_side,
          cuda_real_type*  __restrict__ d_out_value,
          cuda_real_type*  __restrict__ d_out_limit,
          int*             __restrict__ d_out_count,
          int*             __restrict__ d_out_truncated)
{
    const ptrdiff_t local_c = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = c_start + static_cast<int>(local_c);
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * K;

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V = d_V + local_c * n_bus;
    int  cnt       = 0;
    bool truncated = false;
    auto push = [&](int hid, int side, cuda_real_type value, cuda_real_type limit) {
        if (cnt >= K) { truncated = true; return; }
        d_out_hvdc_id[base + cnt] = hid;
        d_out_side[base + cnt]    = side;
        d_out_value[base + cnt]   = value;
        d_out_limit[base + cnt]   = limit;
        ++cnt;
    };

    for (int e = 0; e < n_hvdc; ++e) {
        if (d_status[e] != 0) continue;   // saturated: pinned at pmax by construction
        const cudaComplexType V1 = V[d_bus1[e]];
        const cudaComplexType V2 = V[d_bus2[e]];
        if (!isfinite(V1.x) || !isfinite(V1.y) || !isfinite(V2.x) || !isfinite(V2.y)) continue;
        const cuda_real_type th1 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
        const cuda_real_type th2 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
        const cuda_real_type raw = d_p0[e] + d_k[e] * (th1 - th2);
        cuda_real_type p1_flow, p2_flow;
        hvdc_flows_pu(0, raw, d_lf1[e], d_lf2[e], d_r[e], d_pmax12[e], d_pmax21[e], p1_flow, p2_flow);
        if (raw >= cuda_real_type(0)) {
            const cuda_real_type pmax = d_pmax12[e];
            if (isfinite(pmax) && p1_flow > pmax + tol_pu)
                push(d_hvdc_id[e], 1, p1_flow * sn_mva, pmax * sn_mva);
        } else {
            const cuda_real_type pmax = d_pmax21[e];
            if (isfinite(pmax) && p2_flow > pmax + tol_pu)
                push(d_hvdc_id[e], 2, p2_flow * sn_mva, pmax * sn_mva);
        }
    }

    d_out_count[out_c]     = cnt;
    d_out_truncated[out_c] = truncated ? 1 : 0;
}
