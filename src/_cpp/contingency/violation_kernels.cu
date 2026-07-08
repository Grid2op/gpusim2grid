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

constexpr int VIOL_LOW_VOLTAGE  = 0;
constexpr int VIOL_HIGH_VOLTAGE = 1;
constexpr int VIOL_CURRENT      = 2;
constexpr int VIOL_DIVERGED     = 3;

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
    const cudaComplexType* __restrict__ d_yff,
    const cudaComplexType* __restrict__ d_yft,
    const cudaComplexType* __restrict__ d_ytf,
    const cudaComplexType* __restrict__ d_ytt,
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
    const int local_c = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = c_start + local_c;   // GLOBAL active-slot id (tripped-branch table)
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    const int base  = out_c * K;
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

    // ---- 1. DIVERGED ---------------------------------------------------
    // V is unreliable for a diverged system; do not scan buses/branches on it.
    const cuda_real_type residual = d_residuals[out_c];
    if (residual > tol) {
        push(ELEM_BUS, -1, 0, VIOL_DIVERGED, residual, tol);
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
        if (tripped) continue;   // Ybus was patched but yff/yft/ytf/ytt weren't -- would report a phantom current

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
            CudaFunHelper::my_cuCmul(d_yff[l], Vi), CudaFunHelper::my_cuCmul(d_yft[l], Vj))
            : CudaFunHelper::my_make_cuComplex(0., 0.);
        const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
            CudaFunHelper::my_cuCmul(d_ytf[l], Vi), CudaFunHelper::my_cuCmul(d_ytt[l], Vj))
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
