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
// compute_physical_violations (check_gen_p_violations_kernel)
constexpr int ELEM_GENERATOR = 5;
constexpr int ELEM_STORAGE   = 6;

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
// compute_physical_violations (check_gen_p_violations_kernel): HIGH_P is the
// same code as HVDC_P_SATURATION (lightsim2grid's one name for both).
constexpr int VIOL_HIGH_P       = 7;
constexpr int VIOL_LOW_P        = 8;

// check_limit_violations_kernel's record groups, in output order (one per
// type; see N_OPERATIONAL_VIOLATION_GROUPS)
constexpr int GRP_CURRENT      = 0;
constexpr int GRP_LOW_VOLTAGE  = 1;
constexpr int GRP_HIGH_VOLTAGE = 2;

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

// Keeps, per violation type ("group"), the K most severe records of ONE row, each
// group sorted most severe first (see violation_kernels.cuh, "Records kept").
// While the row is scanned, group g owns slots [base + g*K, base + g*K + K) of
// the row's exclusive slice; finish() then packs the groups to the front of the
// slice, in group order, and returns the record count. Severity is recomputed
// from the stored value/limit (sev), so no key is stored; mv(dst, src) moves
// the kernel's own extra fields (value/limit are moved here). A candidate is
// rejected in O(1) once its group is full and it is no more severe than the
// group's least severe record; ties keep the record seen first (scan order), so
// the output stays deterministic.
template <int NG, class Sev, class Move>
struct TopKByType {
    ptrdiff_t       base;
    int             K;
    cuda_real_type* value;
    cuda_real_type* limit;
    Sev             sev;
    Move            mv;
    int             cnt[NG];
    bool            truncated;

    __device__ TopKByType(ptrdiff_t base_, int K_, cuda_real_type* value_,
                          cuda_real_type* limit_, Sev sev_, Move mv_)
        : base(base_), K(K_), value(value_), limit(limit_), sev(sev_), mv(mv_), truncated(false)
    {
        for (int g = 0; g < NG; ++g) cnt[g] = 0;
    }

    __device__ void move(ptrdiff_t dst, ptrdiff_t src) {
        value[dst] = value[src];
        limit[dst] = limit[src];
        mv(dst, src);
    }

    // The slot a record of group g with severity `key` goes to (the records it
    // outranks already shifted down by one), or -1 to drop it.
    __device__ ptrdiff_t reserve(int g, cuda_real_type key) {
        const ptrdiff_t gb = base + static_cast<ptrdiff_t>(g) * K;
        int n = cnt[g];
        if (n == K) {
            truncated = true;
            if (!(key > sev(value[gb + K - 1], limit[gb + K - 1]))) return -1;
            n = K - 1;   // the least severe kept record makes room
        } else {
            cnt[g] = n + 1;
        }
        int pos = n;
        while (pos > 0 && key > sev(value[gb + pos - 1], limit[gb + pos - 1])) {
            move(gb + pos, gb + pos - 1);
            --pos;
        }
        return gb + pos;
    }

    __device__ int finish() {
        int out = 0;
        for (int g = 0; g < NG; ++g) {
            const ptrdiff_t gb = base + static_cast<ptrdiff_t>(g) * K;
            for (int i = 0; i < cnt[g]; ++i, ++out)
                if (base + out != gb + i) move(base + out, gb + i);
        }
        return out;
    }
};

template <int NG, class Sev, class Move>
__device__ TopKByType<NG, Sev, Move> make_topk(ptrdiff_t base, int K, cuda_real_type* value,
                                               cuda_real_type* limit, Sev sev, Move mv)
{
    return TopKByType<NG, Sev, Move>(base, K, value, limit, sev, mv);
}

// Severity keys. Voltage and current: how far the ratio value/limit is from 1
// (|value/limit - 1|, so a LOW_VOLTAGE ranks by how far BELOW its limit it
// fell) -- a relative measure, so a 400 kV bus does not outrank a 63 kV one
// by its size alone. The physical checks (MVAr, MW): the absolute excess.
struct SevRatio {
    __device__ __forceinline__ cuda_real_type operator()(cuda_real_type value, cuda_real_type limit) const {
        return fabs(value / limit - cuda_real_type(1));
    }
};
struct SevAbs {
    __device__ __forceinline__ cuda_real_type operator()(cuda_real_type value, cuda_real_type limit) const {
        return fabs(value - limit);
    }
};

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
    const cuda_real_type*  __restrict__ d_base_current_ex_A,
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
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_OPERATIONAL_VIOLATION_GROUPS * K);
    // TRUE, uncapped per-type totals -- incremented on every violation found,
    // independent of whether a detail record could still be kept.
    int n_low = 0, n_high = 0, n_current = 0;

    // The K most severe records of each type (groups: GRP_CURRENT,
    // GRP_LOW_VOLTAGE, GRP_HIGH_VOLTAGE), in this contingency's exclusive
    // output slice -- single-thread-owned, no atomics.
    auto topk = make_topk<N_OPERATIONAL_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevRatio{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_element_type[dst] = d_out_element_type[src];
            d_out_element_id[dst]   = d_out_element_id[src];
            d_out_side[dst]         = d_out_side[src];
            d_out_type[dst]         = d_out_type[src];
        });
    auto push = [&](int group, int etype, int eid, int side, int vtype,
                    cuda_real_type value, cuda_real_type limit) {
        const ptrdiff_t at = topk.reserve(group, SevRatio{}(value, limit));
        if (at < 0) return;
        d_out_element_type[at] = etype;
        d_out_element_id[at]   = eid;
        d_out_side[at]         = side;
        d_out_type[at]         = vtype;
        d_out_value[at]        = value;
        d_out_limit[at]        = limit;
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
        d_out_element_type[base] = ELEM_GRID;
        d_out_element_id[base]   = -1;
        d_out_side[base]         = 0;
        d_out_type[base]         = VIOL_DIVERGENCE;
        d_out_value[base]        = residual;
        d_out_limit[base]        = tol;
        d_out_count[out_c]              = 1;
        d_out_truncated[out_c]          = 0;
        d_out_count_low_voltage[out_c]  = 0;
        d_out_count_high_voltage[out_c] = 0;
        d_out_count_current[out_c]      = 0;
        return;
    }

    // ---- 2. Branch current (checked first: thermal/current violations are
    // generally first-order operational concerns, voltage second-order) ----
    // Scans every branch whatever is already kept, so n_current stays exact
    // (and the kept records are the most severe ones).
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
        const cuda_real_type ka_ex = CudaFunHelper::my_cuCabs(I_ex) * d_base_current_ex_A[l] * cuda_real_type(0.001);

        const int etype = (l < n_lines) ? ELEM_LINE : ELEM_TRAFO;
        const int eid    = (l < n_lines) ? l : (l - n_lines);
        if (!isnan(lim1) && ka_or > lim1) { ++n_current; push(GRP_CURRENT, etype, eid, 1, VIOL_CURRENT, ka_or, lim1); }
        if (!isnan(lim2) && ka_ex > lim2) { ++n_current; push(GRP_CURRENT, etype, eid, 2, VIOL_CURRENT, ka_ex, lim2); }
    }

    // ---- 3. Bus voltage --------------------------------------------------
    // Scans every bus, same reason as above.
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
            push(GRP_LOW_VOLTAGE, ELEM_BUS, b, 0, VIOL_LOW_VOLTAGE, vm_kv, vmin);
        } else if (!isnan(vmax) && vm_kv > vmax) {
            ++n_high;
            push(GRP_HIGH_VOLTAGE, ELEM_BUS, b, 0, VIOL_HIGH_VOLTAGE, vm_kv, vmax);
        }
    }

    d_out_count[out_c]              = topk.finish();
    d_out_truncated[out_c]          = topk.truncated ? 1 : 0;
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
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_BUS_Q_VIOLATION_GROUPS * K);

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V  = d_V     + local_c * n_bus;
    const cudaComplexType* Yv = d_Yvals + local_c * nnz_Y;
    const cudaComplexType* Sb = d_Sbus  + local_c * sbus_stride;
    const unsigned char*   off = d_gen_off ? d_gen_off + static_cast<ptrdiff_t>(out_c) * n_gen : nullptr;

    // the K largest excesses of each type (groups: LOW_Q, then HIGH_Q)
    auto topk = make_topk<N_BUS_Q_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevAbs{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_bus_id[dst] = d_out_bus_id[src];
            d_out_type[dst]   = d_out_type[src];
        });
    auto push = [&](int group, int bus, int vtype, cuda_real_type value, cuda_real_type limit) {
        const ptrdiff_t at = topk.reserve(group, SevAbs{}(value, limit));
        if (at < 0) return;
        d_out_bus_id[at] = bus;
        d_out_type[at]   = vtype;
        d_out_value[at]  = value;
        d_out_limit[at]  = limit;
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
            push(0, b, VIOL_LOW_Q, q_bus, q_min);
        } else if (isfinite(q_max) && q_bus > q_max + tol_mvar) {
            push(1, b, VIOL_HIGH_Q, q_bus, q_max);
        }
    }

    d_out_count[out_c]     = topk.finish();
    d_out_truncated[out_c] = topk.truncated ? 1 : 0;
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
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_HVDC_P_VIOLATION_GROUPS * K);

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V = d_V + local_c * n_bus;
    // the K largest excesses (one type: HIGH_P, either side)
    auto topk = make_topk<N_HVDC_P_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevAbs{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_hvdc_id[dst] = d_out_hvdc_id[src];
            d_out_side[dst]    = d_out_side[src];
        });
    auto push = [&](int hid, int side, cuda_real_type value, cuda_real_type limit) {
        const ptrdiff_t at = topk.reserve(0, SevAbs{}(value, limit));
        if (at < 0) return;
        d_out_hvdc_id[at] = hid;
        d_out_side[at]    = side;
        d_out_value[at]   = value;
        d_out_limit[at]   = limit;
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

    d_out_count[out_c]     = topk.finish();
    d_out_truncated[out_c] = topk.truncated ? 1 : 0;
}


// =============================================================================
// check_gen_p_violations_kernel -- see violation_kernels.cuh
// =============================================================================
__global__ void check_gen_p_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Yvals,
    const int*             __restrict__ d_Y_outer,
    const int*             __restrict__ d_Y_inner,
    const cudaComplexType* __restrict__ d_Sbus,
    int                                 sbus_stride,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type                      residual_tol,
    int                                 n_hvdc,
    const int*             __restrict__ d_hvdc_bus1,
    const int*             __restrict__ d_hvdc_bus2,
    const int*             __restrict__ d_hvdc_status,
    const cuda_real_type*  __restrict__ d_hvdc_p0,
    const cuda_real_type*  __restrict__ d_hvdc_k,
    const cuda_real_type*  __restrict__ d_hvdc_lf1,
    const cuda_real_type*  __restrict__ d_hvdc_lf2,
    const cuda_real_type*  __restrict__ d_hvdc_r,
    const cuda_real_type*  __restrict__ d_hvdc_pmax12,
    const cuda_real_type*  __restrict__ d_hvdc_pmax21,
    int                                 n_entries,
    const int*             __restrict__ d_el_type,
    const int*             __restrict__ d_el_id,
    const int*             __restrict__ d_bus_solver,
    const int*             __restrict__ d_bus_slot,
    const cuda_real_type*  __restrict__ d_weight,
    const cuda_real_type*  __restrict__ d_min_p,
    const cuda_real_type*  __restrict__ d_max_p,
    const cuda_real_type*  __restrict__ d_target_base,
    int                                 n_part_bus,
    const int*             __restrict__ d_part_bus,
    const int*             __restrict__ d_part_start,
    const int*             __restrict__ d_part_el_type,
    const int*             __restrict__ d_part_el_id,
    const cuda_real_type*  __restrict__ d_part_weight,
    const unsigned char*   __restrict__ d_gen_off,
    int                                 n_gen,
    const cuda_real_type*  __restrict__ d_targets,
    int                                 target_stride,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_mw,
    int n_bus, int nnz_Y,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
          int*             __restrict__ d_out_element_type,
          int*             __restrict__ d_out_element_id,
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
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_GEN_P_VIOLATION_GROUPS * K);

    d_out_count[out_c]     = 0;
    d_out_truncated[out_c] = 0;
    if (row_not_converged(d_residuals, out_c, residual_tol)) return;

    const cudaComplexType* V   = d_V     + local_c * n_bus;
    const cudaComplexType* Yv  = d_Yvals + local_c * nnz_Y;
    const cudaComplexType* Sb  = d_Sbus  + local_c * sbus_stride;
    const unsigned char*   off = d_gen_off ? d_gen_off + static_cast<ptrdiff_t>(out_c) * n_gen : nullptr;
    const cuda_real_type*  tgt = d_targets ? d_targets + static_cast<ptrdiff_t>(out_c) * target_stride : nullptr;
    const cuda_real_type   eps = cuda_real_type(1e-12);

    auto bus_live = [&](int b) -> bool {
        const cudaComplexType Vb = V[b];
        return isfinite(Vb.x) && isfinite(Vb.y);   // masked (stranded) bus otherwise
    };
    // a row takes a GENERATOR out of the distribution; a storage unit is never
    // disconnected by one, so it always keeps its share
    auto machine_live = [&](int type, int id) -> bool {
        return !(type == ELEM_GENERATOR && off != nullptr && id < n_gen && off[id]);
    };
    // the live raw participation of the participants grouped under slot s
    auto slot_raw_w = [&](int s) -> cuda_real_type {
        cuda_real_type w = 0;
        for (int p = d_part_start[s]; p < d_part_start[s + 1]; ++p)
            if (machine_live(d_part_el_type[p], d_part_el_id[p])) w += d_part_weight[p];
        return w;
    };
    auto finite_or_zero = [](cudaComplexType v) -> cudaComplexType {
        if (!isfinite(v.x) || !isfinite(v.y))
            return CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
        return v;
    };

    // ---- the row's total raw participation -----------------------------------
    // Over the machines it actually leaves participating, both families: nothing
    // left distributing anything means nothing to report (upstream parity).
    cuda_real_type total_raw_w = 0;
    for (int s = 0; s < n_part_bus; ++s) {
        if (!bus_live(d_part_bus[s])) continue;
        total_raw_w += slot_raw_w(s);
    }
    if (!(fabs(total_raw_w) > eps)) return;

    // the K largest excesses of each type (groups: LOW_P, then HIGH_P),
    // generators and storage units ranked together
    auto topk = make_topk<N_GEN_P_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevAbs{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_element_type[dst] = d_out_element_type[src];
            d_out_element_id[dst]   = d_out_element_id[src];
            d_out_type[dst]         = d_out_type[src];
        });
    auto push = [&](int group, int etype, int eid, int vtype, cuda_real_type value, cuda_real_type limit) {
        const ptrdiff_t at = topk.reserve(group, SevAbs{}(value, limit));
        if (at < 0) return;
        d_out_element_type[at] = etype;
        d_out_element_id[at]   = eid;
        d_out_type[at]         = vtype;
        d_out_value[at]        = value;
        d_out_limit[at]        = limit;
    };

    for (int k = 0; k < n_entries; ++k) {
        const int b = d_bus_solver[k];
        if (!bus_live(b)) continue;
        const int etype = d_el_type[k];
        const int eid   = d_el_id[k];
        if (!machine_live(etype, eid)) continue;   // disconnected by this row: produces nothing

        // its target, plus its share of what its bus had to make up -- exactly
        // as lightsim2grid's GeneratorContainer::set_p_slack computes it after
        // a single solve. Generator convention throughout (a storage unit's
        // target was negated when the plan was built).
        cuda_real_type p_mw = d_target_base[k];
        if (tgt != nullptr) {
            const cuda_real_type t = tgt[k];
            if (!isnan(t)) p_mw = t;
        }
        const int slot = d_bus_slot[k];
        const cuda_real_type bus_raw_w = (slot >= 0) ? slot_raw_w(slot) : cuda_real_type(0);
        if (fabs(bus_raw_w) > eps) {
            // the active power the slack machines of this bus produced on top
            // of their targets: the raw active residual of the slot's patched
            // Ybus ...
            const cudaComplexType Vb = V[b];
            cudaComplexType Ib = CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
            for (int p = d_Y_outer[b]; p < d_Y_outer[b + 1]; ++p) {
                const cudaComplexType Vj = finite_or_zero(V[d_Y_inner[p]]);
                Ib = CudaFunHelper::my_cuCadd(Ib, CudaFunHelper::my_cuCmul(Yv[p], Vj));
            }
            const cudaComplexType S = CudaFunHelper::my_cuCmul(Vb, CudaFunHelper::my_cuConj(Ib));
            cuda_real_type mis = S.x - Sb[b].x;
            // ... plus the angle-droop hvdc flows leaving it (a converter
            // station's published injection is -p_flow: not the slack's doing)
            for (int e = 0; e < n_hvdc; ++e) {
                const int b1 = d_hvdc_bus1[e], b2 = d_hvdc_bus2[e];
                if (b1 != b && b2 != b) continue;
                const cudaComplexType V1 = V[b1];
                const cudaComplexType V2 = V[b2];
                if (!isfinite(V1.x) || !isfinite(V1.y) || !isfinite(V2.x) || !isfinite(V2.y)) continue;
                const cuda_real_type th1 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
                const cuda_real_type th2 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
                const cuda_real_type raw = d_hvdc_p0[e] + d_hvdc_k[e] * (th1 - th2);
                cuda_real_type p1_flow, p2_flow;
                hvdc_flows_pu(d_hvdc_status[e], raw, d_hvdc_lf1[e], d_hvdc_lf2[e], d_hvdc_r[e],
                              d_hvdc_pmax12[e], d_hvdc_pmax21[e], p1_flow, p2_flow);
                if (b1 == b) mis += p1_flow;
                if (b2 == b) mis += p2_flow;
            }
            p_mw += mis * sn_mva * d_weight[k] / bus_raw_w;
        }
        if (!isfinite(p_mw)) continue;

        const cuda_real_type pmin = d_min_p[k];
        const cuda_real_type pmax = d_max_p[k];
        if (isfinite(pmin) && p_mw < pmin - tol_mw) {
            push(0, etype, eid, VIOL_LOW_P, p_mw, pmin);
        } else if (isfinite(pmax) && p_mw > pmax + tol_mw) {
            push(1, etype, eid, VIOL_HIGH_P, p_mw, pmax);
        }
    }

    d_out_count[out_c]     = topk.finish();
    d_out_truncated[out_c] = topk.truncated ? 1 : 0;
}
