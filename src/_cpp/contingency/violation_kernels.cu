// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/violation_kernels.cu
// =============================================================================

#include "violation_kernels.cuh"

#include <math.h>   // isnan

#include <type_traits>   // std::conditional

#include <cub/block/block_scan.cuh>

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
// the SVC entries of the release plan (check_gen_pv_release_violations_kernel)
constexpr int ELEM_SVC       = 7;

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
// compute_physical_violations (check_hvdc_p_violations_kernel): element type HVDC
// (=4), violation type HVDC_P_SATURATION (=7, VIOL_HIGH_P below) or, for a line
// frozen at its limit, VIOL_HVDC_AC_EMULATION_RELEASE.
// compute_physical_violations (check_gen_p_violations_kernel): HIGH_P is the
// same code as HVDC_P_SATURATION (lightsim2grid's one name for both).
constexpr int VIOL_HIGH_P       = 7;
constexpr int VIOL_LOW_P        = 8;
// compute_physical_violations (check_gen_pv_release_violations_kernel)
constexpr int VIOL_LOW_VOLTAGE_AT_MIN_Q  = 9;
constexpr int VIOL_HIGH_VOLTAGE_AT_MAX_Q = 10;
// the standby SVC check, on the SVC entries of the same plan (see limit_violation_types.hpp)
constexpr int VIOL_LOW_VOLTAGE_SVC_STANDBY  = 11;
constexpr int VIOL_HIGH_VOLTAGE_SVC_STANDBY = 12;
// the remote voltage control check, on the remote control entries of the same plan
constexpr int VIOL_LOW_VOLTAGE_REMOTE_CONTROL  = 13;
constexpr int VIOL_HIGH_VOLTAGE_REMOTE_CONTROL = 14;
constexpr int VIOL_HVDC_AC_EMULATION_RELEASE = 15;
// the `standby` column of the release plan (GenPvReleasePlanData::CHECK_*)
constexpr int CHECK_SVC_STANDBY    = 1;
constexpr int CHECK_REMOTE_CONTROL = 2;

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

// ---- check_limit_violations_kernel's helpers ------------------------------------

// The read-only inputs of one row, gathered once per block.
struct ViolRowArgs {
    const cudaComplexType* V;            // the slot's voltages, [n_bus]
    const cuda_real_type*  vn_kv;
    const cuda_real_type*  vmin_kv;
    const cuda_real_type*  vmax_kv;
    const int*             from;
    const int*             to;
    const cudaComplexType* yff;
    const cudaComplexType* yft;
    const cudaComplexType* ytf;
    const cudaComplexType* ytt;
    const cuda_real_type*  base_A;
    const cuda_real_type*  base_ex_A;
    const cuda_real_type*  lim1;
    const cuda_real_type*  lim2;
    const int*             trip;         // the row's tripped branches
    int                    t_count;
    const int*             branch_bc;    // null: every branch has its shared *_eff block
    const cudaComplexType* bc_applied;   // the slot's own blocks, [n_bc * 4]
    cuda_real_type         up;           // 1 + rel_tol (HIGH_VOLTAGE / CURRENT)
    cuda_real_type         dn;           // 1 - rel_tol (LOW_VOLTAGE)
};

// The CURRENT violations of branch l: bit s of the result set when side s+1
// violates, its value / limit / severity in value[s] / limit[s] / sev[s].
// noinline, as eval_bus: every pass of the kernel evaluates an element through
// this one function, so the passes agree on each severity to the last bit
// (inlined at several call sites, the compiler could contract the complex
// products into FMAs differently).
__device__ __noinline__ int eval_branch(const ViolRowArgs& a, int l, cuda_real_type* value,
                                        cuda_real_type* limit, cuda_real_type* sev)
{
    for (int ti = 0; ti < a.t_count; ++ti)
        if (a.trip[ti] == l) return 0;   // Ybus was patched but yff_eff/yft_eff/ytf_eff/ytt_eff weren't -- would report a phantom current

    const cuda_real_type lim1 = a.lim1[l];
    const cuda_real_type lim2 = a.lim2[l];
    if (isnan(lim1) && isnan(lim2)) return 0;   // no limit configured for this branch

    // branch_from/branch_to are in AC-solver bus numbering (see
    // concat_busids_to_solver, ls2g_bridge.cpp): a side that lightsim2grid
    // Kron-reduced away (isolated / half-open line, keep_half_open_lines)
    // is relabeled to -1. That bus has no voltage in the solved system --
    // treat it as V=0 -- and no terminal current on that side to report --
    // 0, not computed from the pi-model formula, since the terminal
    // itself doesn't exist. Reading V[-1] without this guard is an
    // out-of-bounds read (confirmed via compute-sanitizer on a real
    // half-open-line grid).
    const int bf = a.from[l];
    const int bt = a.to[l];
    const cudaComplexType Vi = (bf >= 0) ? a.V[bf] : CudaFunHelper::my_make_cuComplex(0., 0.);
    const cudaComplexType Vj = (bt >= 0) ? a.V[bt] : CudaFunHelper::my_make_cuComplex(0., 0.);
    // a transformer whose taps the outer loops move: the block the slot holds now
    cudaComplexType yff = a.yff[l], yft = a.yft[l], ytf = a.ytf[l], ytt = a.ytt[l];
    if (a.branch_bc != nullptr && a.branch_bc[l] >= 0) {
        const cudaComplexType* y = a.bc_applied + 4 * a.branch_bc[l];
        yff = y[0]; yft = y[1]; ytf = y[2]; ytt = y[3];
    }
    const cudaComplexType I_or = (bf >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(yff, Vi), CudaFunHelper::my_cuCmul(yft, Vj))
        : CudaFunHelper::my_make_cuComplex(0., 0.);
    const cudaComplexType I_ex = (bt >= 0) ? CudaFunHelper::my_cuCadd(
        CudaFunHelper::my_cuCmul(ytf, Vi), CudaFunHelper::my_cuCmul(ytt, Vj))
        : CudaFunHelper::my_make_cuComplex(0., 0.);
    // *0.001: gpusim2grid's d_base_current_A is Amps-based; limits are kA.
    const cuda_real_type ka_or = CudaFunHelper::my_cuCabs(I_or) * a.base_A[l] * cuda_real_type(0.001);
    const cuda_real_type ka_ex = CudaFunHelper::my_cuCabs(I_ex) * a.base_ex_A[l] * cuda_real_type(0.001);

    // beyond the limit by more than rel_tol (violation_rel_tol, lightsim2grid's
    // OperationalCheck.hpp): a value ON its limit up to rounding is not reported
    int m = 0;
    if (!isnan(lim1) && ka_or > lim1 * a.up) {
        value[0] = ka_or; limit[0] = lim1; sev[0] = SevRatio{}(ka_or, lim1); m |= 1;
    }
    if (!isnan(lim2) && ka_ex > lim2 * a.up) {
        value[1] = ka_ex; limit[1] = lim2; sev[1] = SevRatio{}(ka_ex, lim2); m |= 2;
    }
    return m;
}

// The voltage violation of bus b: its group (GRP_LOW_VOLTAGE / GRP_HIGH_VOLTAGE)
// with its value / limit / severity, or -1.
__device__ __noinline__ int eval_bus(const ViolRowArgs& a, int b, cuda_real_type* value,
                                     cuda_real_type* limit, cuda_real_type* sev)
{
    const cuda_real_type vmin = a.vmin_kv[b];
    const cuda_real_type vmax = a.vmax_kv[b];
    if (isnan(vmin) && isnan(vmax)) return -1;   // no limit configured for this bus

    // vm_kv is NaN when V[b] is NaN (masked bus, written by nr_mask_v_nan
    // before this kernel launches) -- NaN comparisons below are false in
    // IEEE754, so masked buses are excluded for free.
    const cuda_real_type vm_kv = CudaFunHelper::my_cuCabs(a.V[b]) * a.vn_kv[b];
    if (!isnan(vmin) && vm_kv < vmin * a.dn) {
        value[0] = vm_kv; limit[0] = vmin; sev[0] = SevRatio{}(vm_kv, vmin);
        return GRP_LOW_VOLTAGE;
    }
    if (!isnan(vmax) && vm_kv > vmax * a.up) {
        value[0] = vm_kv; limit[0] = vmax; sev[0] = SevRatio{}(vm_kv, vmax);
        return GRP_HIGH_VOLTAGE;
    }
    return -1;
}

// A severity is >= 0 and never NaN (a zero limit gives +inf), so its bit
// pattern, read as an unsigned integer of the same width, orders as it does.
using sev_bits_t = std::conditional<sizeof(cuda_real_type) == 8, unsigned long long, unsigned int>::type;
__device__ __forceinline__ sev_bits_t sev_bits(cuda_real_type s) {
    if constexpr (sizeof(cuda_real_type) == 8)
        return static_cast<sev_bits_t>(__double_as_longlong(s));
    else
        return static_cast<sev_bits_t>(__float_as_uint(s));
}

constexpr int VIOL_RADIX_BITS = 8;
constexpr int VIOL_RADIX = 1 << VIOL_RADIX_BITS;

__device__ __forceinline__ int warp_sum(int x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
    return x;
}

}  // namespace

// One block per row (see violation_kernels.cuh): count, select, collect, emit.
__global__ void __launch_bounds__(VIOL_BLOCK) check_limit_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type          tol,
    cuda_real_type          rel_tol,
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
    const int* __restrict__ d_slot_active,
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
          int*             __restrict__ d_out_count_current,
          cuda_real_type*  __restrict__ d_scratch_real,
          int*             __restrict__ d_scratch_key,
    const int*             __restrict__ d_branch_bc,
    const cudaComplexType* __restrict__ d_bc_applied,
    int n_bc)
{
    // local_c widened to ptrdiff_t: local_c * n_bus below (the d_V offset) is
    // the same at-risk product as fill_J_kernel's own J_base once
    // actual_batch * n_bus grows large (see acpf_nr_kernels.cu's note). base
    // (out_c * K, the per-contingency output slice offset) gets the same
    // treatment below.
    const ptrdiff_t local_c = blockIdx.x;
    if (local_c >= actual_batch) return;
    const int tid = threadIdx.x;

    // GLOBAL active-slot id (tripped-branch table): the slot's own row when
    // d_slot_active is given (continuous batching; -1 = not leaving), else
    // the chunk's c_start + local_c. Uniform over the block, as every early
    // return below.
    const int slot_global = d_slot_active ? d_slot_active[local_c]
                                          : c_start + static_cast<int>(local_c);
    if (slot_global < 0) return;
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    // base widened to ptrdiff_t: out_c * K (this contingency's output slice
    // offset) is the same at-risk product as fill_J_kernel's own J_base once
    // n_contingencies * K grows large.
    constexpr int NG = N_OPERATIONAL_VIOLATION_GROUPS;
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (NG * K);

    // ---- DIVERGENCE --------------------------------------------------------
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
        if (tid == 0) {
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
        }
        return;
    }

    const int t_start = d_trip_start ? d_trip_start[slot_global] : 0;
    ViolRowArgs a;
    a.V = d_V + local_c * n_bus;
    a.vn_kv = d_bus_vn_kv; a.vmin_kv = d_bus_vmin_kv; a.vmax_kv = d_bus_vmax_kv;
    a.from = d_branch_from; a.to = d_branch_to;
    a.yff = d_yff_eff; a.yft = d_yft_eff; a.ytf = d_ytf_eff; a.ytt = d_ytt_eff;
    a.base_A = d_base_current_A; a.base_ex_A = d_base_current_ex_A;
    a.lim1 = d_branch_limit_a1_ka; a.lim2 = d_branch_limit_a2_ka;
    a.trip = d_trip_branch_flat ? d_trip_branch_flat + t_start : nullptr;
    a.t_count = d_trip_count ? d_trip_count[slot_global] : 0;
    a.branch_bc = d_branch_bc;
    a.bc_applied = d_bc_applied ? d_bc_applied + 4 * local_c * n_bc : nullptr;
    a.up = cuda_real_type(1) + rel_tol;
    a.dn = cuda_real_type(1) - rel_tol;

    using BlockScan = cub::BlockScan<int, VIOL_BLOCK>;
    __shared__ typename BlockScan::TempStorage s_scan;
    __shared__ int        s_cnt[NG];        // TRUE, uncapped per-type totals
    __shared__ int        s_kept[NG];       // records collected so far
    __shared__ int        s_need[NG];       // ties at the threshold still to keep
    __shared__ int        s_tie_base[NG];   // ties seen in the previous tiles
    __shared__ bool       s_sel[NG];        // more than K: keep only the K most severe
    __shared__ sev_bits_t s_prefix[NG];     // the threshold's bits found so far
    __shared__ sev_bits_t s_mask[NG];
    __shared__ int        s_hist[NG][VIOL_RADIX];
    if (tid < NG) {
        s_cnt[tid] = 0; s_kept[tid] = 0; s_tie_base[tid] = 0;
        s_prefix[tid] = 0; s_mask[tid] = 0;
    }
    __syncthreads();

    cuda_real_type val[2], lim[2], sv[2];

    // ---- 1. count ------------------------------------------------------------
    // Every branch end and bus, whatever is kept, so the totals stay exact.
    {
        int c[NG] = {0, 0, 0};
        for (int l = tid; l < n_branches; l += VIOL_BLOCK)
            c[GRP_CURRENT] += __popc(eval_branch(a, l, val, lim, sv));
        for (int b = tid; b < n_bus; b += VIOL_BLOCK) {
            const int g = eval_bus(a, b, val, lim, sv);
            if (g >= 0) ++c[g];
        }
        for (int g = 0; g < NG; ++g) {
            const int w = warp_sum(c[g]);
            if ((tid & 31) == 0 && w != 0) atomicAdd(&s_cnt[g], w);
        }
    }
    __syncthreads();
    if (tid < NG) { s_sel[tid] = s_cnt[tid] > K; s_need[tid] = K; }
    __syncthreads();
    const bool sel_cur = s_sel[GRP_CURRENT];
    const bool sel_v   = s_sel[GRP_LOW_VOLTAGE] || s_sel[GRP_HIGH_VOLTAGE];

    // ---- 2. select -----------------------------------------------------------
    // Per type with more than K: the bits of the K-th largest severity T, by an
    // MSD radix select (one digit per pass, the three types at once); s_need
    // ends as K - #{sev > T}, the ties at T to keep.
    if (sel_cur || sel_v) {
        for (int shift = 8 * static_cast<int>(sizeof(sev_bits_t)) - VIOL_RADIX_BITS; shift >= 0;
             shift -= VIOL_RADIX_BITS) {
            for (int i = tid; i < NG * VIOL_RADIX; i += VIOL_BLOCK) (&s_hist[0][0])[i] = 0;
            __syncthreads();
            if (sel_cur) {
                for (int l = tid; l < n_branches; l += VIOL_BLOCK) {
                    const int m = eval_branch(a, l, val, lim, sv);
                    for (int s = 0; s < 2; ++s) {
                        if (!((m >> s) & 1)) continue;
                        const sev_bits_t u = sev_bits(sv[s]);
                        if ((u & s_mask[GRP_CURRENT]) == s_prefix[GRP_CURRENT])
                            atomicAdd(&s_hist[GRP_CURRENT][(u >> shift) & (VIOL_RADIX - 1)], 1);
                    }
                }
            }
            if (sel_v) {
                for (int b = tid; b < n_bus; b += VIOL_BLOCK) {
                    const int g = eval_bus(a, b, val, lim, sv);
                    if (g < 0 || !s_sel[g]) continue;
                    const sev_bits_t u = sev_bits(sv[0]);
                    if ((u & s_mask[g]) == s_prefix[g])
                        atomicAdd(&s_hist[g][(u >> shift) & (VIOL_RADIX - 1)], 1);
                }
            }
            __syncthreads();
            if (tid < NG && s_sel[tid]) {
                // the digit holding the s_need-th largest, from the top
                const int need = s_need[tid];
                int cum = 0, d = VIOL_RADIX - 1;
                for (; d > 0; --d) {
                    const int h = s_hist[tid][d];
                    if (cum + h >= need) break;
                    cum += h;
                }
                s_need[tid]    = need - cum;
                s_prefix[tid] |= static_cast<sev_bits_t>(d) << shift;
                s_mask[tid]   |= static_cast<sev_bits_t>(VIOL_RADIX - 1) << shift;
            }
            __syncthreads();
        }
    }

    // ---- 3. collect ----------------------------------------------------------
    // The kept records -- all of a type with at most K, else those above T and
    // the first s_need ties at T in scan order (tiles in element order, a block
    // scan inside each) -- go to the slot's scratch, in any order: (severity,
    // value, limit) and the scan key (2*l + side - 1 for CURRENT, b for a bus).
    const ptrdiff_t sbase = local_c * (NG * K);
    cuda_real_type* scr_sev = d_scratch_real;
    cuda_real_type* scr_val = d_scratch_real + static_cast<ptrdiff_t>(actual_batch) * NG * K;
    cuda_real_type* scr_lim = scr_val + static_cast<ptrdiff_t>(actual_batch) * NG * K;
    auto collect = [&](int g, int key, cuda_real_type s, cuda_real_type v, cuda_real_type lm) {
        const int pos = atomicAdd(&s_kept[g], 1);
        if (pos >= K) return;   // cannot happen (the passes agree bit for bit); never overflow
        const ptrdiff_t at = sbase + static_cast<ptrdiff_t>(g) * K + pos;
        scr_sev[at] = s; scr_val[at] = v; scr_lim[at] = lm; d_scratch_key[at] = key;
    };
    for (int t0 = 0; t0 < n_branches; t0 += VIOL_BLOCK) {
        const int l = t0 + tid;
        const int m = (l < n_branches) ? eval_branch(a, l, val, lim, sv) : 0;
        int r = 0, agg = 0;
        if (sel_cur) {
            int ties = 0;
            for (int s = 0; s < 2; ++s)
                if (((m >> s) & 1) && sev_bits(sv[s]) == s_prefix[GRP_CURRENT]) ++ties;
            BlockScan(s_scan).ExclusiveSum(ties, r, agg);
            r += s_tie_base[GRP_CURRENT];
        }
        for (int s = 0; s < 2; ++s) {
            if (!((m >> s) & 1)) continue;
            bool keep = true;
            if (sel_cur) {
                const sev_bits_t u = sev_bits(sv[s]);
                keep = u > s_prefix[GRP_CURRENT]
                    || (u == s_prefix[GRP_CURRENT] && r++ < s_need[GRP_CURRENT]);
            }
            if (keep) collect(GRP_CURRENT, 2 * l + s, sv[s], val[s], lim[s]);
        }
        if (sel_cur) {
            __syncthreads();
            if (tid == 0) s_tie_base[GRP_CURRENT] += agg;
            __syncthreads();
        }
    }
    for (int t0 = 0; t0 < n_bus; t0 += VIOL_BLOCK) {
        const int b = t0 + tid;
        const int g = (b < n_bus) ? eval_bus(a, b, val, lim, sv) : -1;
        const bool tie = g >= 0 && s_sel[g] && sev_bits(sv[0]) == s_prefix[g];
        // LOW_VOLTAGE ties in the low half, HIGH_VOLTAGE ones in the high half
        // (at most VIOL_BLOCK of each per tile)
        int r = 0, agg = 0;
        if (sel_v) BlockScan(s_scan).ExclusiveSum(tie ? (g == GRP_LOW_VOLTAGE ? 1 : 1 << 16) : 0, r, agg);
        if (g >= 0) {
            bool keep = true;
            if (s_sel[g]) {
                const int rank = s_tie_base[g] + (g == GRP_LOW_VOLTAGE ? (r & 0xffff) : (r >> 16));
                const sev_bits_t u = sev_bits(sv[0]);
                keep = u > s_prefix[g] || (tie && rank < s_need[g]);
            }
            if (keep) collect(g, b, sv[0], val[0], lim[0]);
        }
        if (sel_v) {
            __syncthreads();
            if (tid == 0) {
                s_tie_base[GRP_LOW_VOLTAGE]  += agg & 0xffff;
                s_tie_base[GRP_HIGH_VOLTAGE] += agg >> 16;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    // ---- 4. emit -------------------------------------------------------------
    // Each kept record goes straight to its final place: the types packed to
    // the front of the row's slice in group order, each sorted most severe
    // first, ties in scan order (its rank among its type's kept records).
    int kept[NG], off[NG];
    for (int g = 0, o = 0; g < NG; ++g) {
        kept[g] = min(s_kept[g], K);
        off[g] = o;
        o += kept[g];
    }
    const int n_kept = off[NG - 1] + kept[NG - 1];
    for (int j = tid; j < n_kept; j += VIOL_BLOCK) {
        int g = NG - 1;
        while (j < off[g]) --g;
        const ptrdiff_t gs = sbase + static_cast<ptrdiff_t>(g) * K;
        const ptrdiff_t at = gs + (j - off[g]);
        const cuda_real_type s = scr_sev[at];
        const int key = d_scratch_key[at];
        int rank = 0;
        for (int q = 0; q < kept[g]; ++q) {
            const cuda_real_type sq = scr_sev[gs + q];
            rank += (sq > s) || (sq == s && d_scratch_key[gs + q] < key);
        }
        const ptrdiff_t o = base + off[g] + rank;
        if (g == GRP_CURRENT) {
            const int l = key >> 1;
            d_out_element_type[o] = (l < n_lines) ? ELEM_LINE : ELEM_TRAFO;
            d_out_element_id[o]   = (l < n_lines) ? l : (l - n_lines);
            d_out_side[o]         = (key & 1) + 1;
            d_out_type[o]         = VIOL_CURRENT;
        } else {
            d_out_element_type[o] = ELEM_BUS;
            d_out_element_id[o]   = key;
            d_out_side[o]         = 0;
            d_out_type[o]         = (g == GRP_LOW_VOLTAGE) ? VIOL_LOW_VOLTAGE : VIOL_HIGH_VOLTAGE;
        }
        d_out_value[o] = scr_val[at];
        d_out_limit[o] = scr_lim[at];
    }

    if (tid == 0) {
        d_out_count[out_c]              = n_kept;
        d_out_truncated[out_c]          = (s_sel[GRP_CURRENT] || sel_v) ? 1 : 0;
        d_out_count_low_voltage[out_c]  = s_cnt[GRP_LOW_VOLTAGE];
        d_out_count_high_voltage[out_c] = s_cnt[GRP_HIGH_VOLTAGE];
        d_out_count_current[out_c]      = s_cnt[GRP_CURRENT];
    }
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
    const unsigned char*   __restrict__ d_row_skip,
    int                                 skip_stride,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_mvar,
    int n_bus, int nnz_Y,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active,
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

    const int slot_global = d_slot_active ? d_slot_active[local_c]
                                          : c_start + static_cast<int>(local_c);
    if (slot_global < 0) return;
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
    const unsigned char*   skip = d_row_skip ? d_row_skip + static_cast<ptrdiff_t>(out_c) * skip_stride : nullptr;

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
        if (skip != nullptr && skip[kk]) continue;   // not held by its machines on this row
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
    int                                 n_frz,
    const int*             __restrict__ d_frz_bus1,
    const int*             __restrict__ d_frz_bus2,
    const int*             __restrict__ d_frz_dir,
    const int*             __restrict__ d_frz_id,
    const cuda_real_type*  __restrict__ d_frz_p0,
    const cuda_real_type*  __restrict__ d_frz_k,
    const cuda_real_type*  __restrict__ d_frz_limit,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_pu,
    int n_bus,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active,
          int*             __restrict__ d_out_hvdc_id,
          int*             __restrict__ d_out_side,
          int*             __restrict__ d_out_type,
          cuda_real_type*  __restrict__ d_out_value,
          cuda_real_type*  __restrict__ d_out_limit,
          int*             __restrict__ d_out_count,
          int*             __restrict__ d_out_truncated)
{
    const ptrdiff_t local_c = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = d_slot_active ? d_slot_active[local_c]
                                          : c_start + static_cast<int>(local_c);
    if (slot_global < 0) return;
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_HVDC_P_VIOLATION_GROUPS * K);

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V = d_V + local_c * n_bus;
    // the K largest excesses per type: HIGH_P (either side), then the releases
    auto topk = make_topk<N_HVDC_P_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevAbs{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_hvdc_id[dst] = d_out_hvdc_id[src];
            d_out_side[dst]    = d_out_side[src];
            d_out_type[dst]    = d_out_type[src];
        });
    auto push = [&](int hid, int side, cuda_real_type value, cuda_real_type limit, int grp = 0) {
        const ptrdiff_t at = topk.reserve(grp, SevAbs{}(value, limit));
        if (at < 0) return;
        d_out_hvdc_id[at] = hid;
        d_out_side[at]    = side;
        d_out_type[at]    = grp == 0 ? VIOL_HIGH_P : VIOL_HVDC_AC_EMULATION_RELEASE;
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

    // the lines frozen at their limit: released when their droop asks for less
    for (int e = 0; e < n_frz; ++e) {
        const cudaComplexType V1 = V[d_frz_bus1[e]];
        const cudaComplexType V2 = V[d_frz_bus2[e]];
        if (!isfinite(V1.x) || !isfinite(V1.y) || !isfinite(V2.x) || !isfinite(V2.y)) continue;
        const cuda_real_type th1 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V1), CudaFunHelper::my_cuCreal(V1));
        const cuda_real_type th2 = CudaFunHelper::my_atan2(CudaFunHelper::my_cuCimag(V2), CudaFunHelper::my_cuCreal(V2));
        const cuda_real_type raw = d_frz_p0[e] + d_frz_k[e] * (th1 - th2);
        const int dir = d_frz_dir[e];
        const cuda_real_type flow  = dir > 0 ? raw : -raw;
        const cuda_real_type limit = d_frz_limit[e];
        if (isfinite(flow) && isfinite(limit) && flow < limit - tol_pu)
            push(d_frz_id[e], dir > 0 ? 1 : 2, flow * sn_mva, limit * sn_mva, 1);
    }

    d_out_count[out_c]     = topk.finish();
    d_out_truncated[out_c] = topk.truncated ? 1 : 0;
}


// =============================================================================
// check_gen_pv_release_violations_kernel -- see violation_kernels.cuh
// =============================================================================
__global__ void check_gen_pv_release_violations_kernel(
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_residuals,
    cuda_real_type                      residual_tol,
    int                                 n_entries,
    const int*             __restrict__ d_gen_id,
    const int*             __restrict__ d_reg_bus,
    const int*             __restrict__ d_gen_bus,
    const int*             __restrict__ d_at_min,
    const cuda_real_type*  __restrict__ d_target_base,
    const cuda_real_type*  __restrict__ d_vn_kv,
    const int*             __restrict__ d_el_type,
    const int*             __restrict__ d_standby,
    const int*             __restrict__ d_side,
    const unsigned char*   __restrict__ d_gen_off,
    int                                 n_gen,
    const cuda_real_type*  __restrict__ d_targets,
    int                                 target_stride,
    const unsigned char*   __restrict__ d_row_skip,
    int                                 skip_stride,
    cuda_real_type                      tol_vm_pu,
    int n_bus,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active,
          int*             __restrict__ d_out_gen_id,
          int*             __restrict__ d_out_type,
          int*             __restrict__ d_out_el_type,
          int*             __restrict__ d_out_side,
          cuda_real_type*  __restrict__ d_out_value,
          cuda_real_type*  __restrict__ d_out_limit,
          int*             __restrict__ d_out_count,
          int*             __restrict__ d_out_truncated)
{
    const ptrdiff_t local_c = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_c >= actual_batch) return;

    const int slot_global = d_slot_active ? d_slot_active[local_c]
                                          : c_start + static_cast<int>(local_c);
    if (slot_global < 0) return;
    const int out_c = d_result_map ? d_result_map[slot_global] : slot_global;
    const ptrdiff_t base = static_cast<ptrdiff_t>(out_c) * (N_GEN_PV_RELEASE_VIOLATION_GROUPS * K);

    if (row_not_converged(d_residuals, out_c, residual_tol)) {
        d_out_count[out_c]     = 0;
        d_out_truncated[out_c] = 0;
        return;
    }

    const cudaComplexType* V = d_V + local_c * n_bus;
    const unsigned char* off = d_gen_off ? d_gen_off + static_cast<ptrdiff_t>(out_c) * n_gen : nullptr;
    const cuda_real_type* tgt = d_targets ? d_targets + static_cast<ptrdiff_t>(out_c) * target_stride : nullptr;
    const unsigned char* skip = d_row_skip ? d_row_skip + static_cast<ptrdiff_t>(out_c) * skip_stride : nullptr;
    // groups: LOW_VOLTAGE_AT_MIN_Q (0), HIGH_VOLTAGE_AT_MAX_Q (1) on the generators and
    // the frozen SVCs, LOW_VOLTAGE_SVC_STANDBY (2), HIGH_VOLTAGE_SVC_STANDBY (3) on the
    // standby SVCs, LOW_VOLTAGE_REMOTE_CONTROL (4), HIGH_VOLTAGE_REMOTE_CONTROL (5) on the
    // remote controllers
    auto topk = make_topk<N_GEN_PV_RELEASE_VIOLATION_GROUPS>(
        base, K, d_out_value, d_out_limit, SevRatio{},
        [&](ptrdiff_t dst, ptrdiff_t src) {
            d_out_gen_id[dst]  = d_out_gen_id[src];
            d_out_type[dst]    = d_out_type[src];
            d_out_el_type[dst] = d_out_el_type[src];
            d_out_side[dst]    = d_out_side[src];
        });
    auto push = [&](int grp, int gid, int type, int el, int side, cuda_real_type value, cuda_real_type limit) {
        const ptrdiff_t at = topk.reserve(grp, SevRatio{}(value, limit));
        if (at < 0) return;
        d_out_gen_id[at]  = gid;
        d_out_type[at]    = type;
        d_out_el_type[at] = el;
        d_out_side[at]    = side;
        d_out_value[at]   = value;
        d_out_limit[at]   = limit;
    };

    for (int k = 0; k < n_entries; ++k) {
        if (skip != nullptr && skip[k]) continue;   // not pinned at a limit on this row
        const int gid = d_gen_id[k];
        // an SVC entry (frozen at a limit, or standby) or an HVDC one (a frozen VSC
        // station): no row disconnects it nor moves its target / thresholds
        const int el = (d_el_type != nullptr) ? d_el_type[k] : ELEM_GENERATOR;
        const bool is_gen = (el == ELEM_GENERATOR);
        const int check = (d_standby != nullptr) ? d_standby[k] : 0;
        const bool is_standby = (el == ELEM_SVC) && (check == CHECK_SVC_STANDBY);
        // a generator holding a remote bus: its own bus (d_reg_bus) against a fixed bound,
        // the remote one (d_gen_bus) only for its mask
        const bool is_remote = is_gen && (check == CHECK_REMOTE_CONTROL);
        const int side = (d_side != nullptr) ? d_side[k] : 0;
        if (is_gen && off != nullptr && gid < n_gen && off[gid]) continue;   // disconnected by the row
        const cudaComplexType Vr = V[d_reg_bus[k]];
        const cudaComplexType Vg = V[d_gen_bus[k]];
        // masked (stranded) regulated bus or own bus: nothing to release
        if (!isfinite(Vr.x) || !isfinite(Vr.y) || !isfinite(Vg.x) || !isfinite(Vg.y)) continue;
        cuda_real_type target = d_target_base[k];
        if (is_gen && !is_remote && tgt != nullptr) {
            const cuda_real_type t_row = tgt[k];
            if (!isnan(t_row)) target = t_row;
        }
        if (!isfinite(target) || !(target > cuda_real_type(0))) continue;
        const cuda_real_type vm = CudaFunHelper::my_cuCabs(Vr);
        const cuda_real_type vn = d_vn_kv[k];
        if (d_at_min[k]) {
            // a generator absorbing all it can, still below the target: absorbs too
            // much; an idle SVC below its low threshold: its automaton switches it on
            if (vm < target - tol_vm_pu) {
                if (is_standby)     push(2, gid, VIOL_LOW_VOLTAGE_SVC_STANDBY, el, side, vm * vn, target * vn);
                else if (is_remote) push(4, gid, VIOL_LOW_VOLTAGE_REMOTE_CONTROL, el, side, vm * vn, target * vn);
                else                push(0, gid, VIOL_LOW_VOLTAGE_AT_MIN_Q, el, side, vm * vn, target * vn);
            }
        } else if (vm > target + tol_vm_pu) {
            if (is_standby)     push(3, gid, VIOL_HIGH_VOLTAGE_SVC_STANDBY, el, side, vm * vn, target * vn);
            else if (is_remote) push(5, gid, VIOL_HIGH_VOLTAGE_REMOTE_CONTROL, el, side, vm * vn, target * vn);
            else                push(1, gid, VIOL_HIGH_VOLTAGE_AT_MAX_Q, el, side, vm * vn, target * vn);
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
    const unsigned char*   __restrict__ d_no_share_gen,
    int                                 n_ns_gen,
    const unsigned char*   __restrict__ d_no_share_sto,
    int                                 n_ns_sto,
    cuda_real_type                      sn_mva,
    cuda_real_type                      tol_mw,
    int n_bus, int nnz_Y,
    int c_start, int actual_batch, int K,
    const int* __restrict__ d_result_map,
    const int* __restrict__ d_slot_active,
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

    const int slot_global = d_slot_active ? d_slot_active[local_c]
                                          : c_start + static_cast<int>(local_c);
    if (slot_global < 0) return;
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
    const unsigned char*   nsg = d_no_share_gen ? d_no_share_gen + static_cast<ptrdiff_t>(out_c) * n_ns_gen : nullptr;
    const unsigned char*   nss = d_no_share_sto ? d_no_share_sto + static_cast<ptrdiff_t>(out_c) * n_ns_sto : nullptr;
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
    // ... and the redistribute_slack pre-pass takes a unit it saturated out of
    // the row's distributed slack: it produces its (clamped) set-point and
    // takes no share (upstream's takes_no_share)
    auto takes_share = [&](int type, int id) -> bool {
        if (!machine_live(type, id)) return false;
        if (type == ELEM_GENERATOR) return !(nsg != nullptr && id < n_ns_gen && nsg[id]);
        return !(nss != nullptr && id < n_ns_sto && nss[id]);
    };
    // the live raw participation of the participants grouped under slot s
    auto slot_raw_w = [&](int s) -> cuda_real_type {
        cuda_real_type w = 0;
        for (int p = d_part_start[s]; p < d_part_start[s + 1]; ++p)
            if (takes_share(d_part_el_type[p], d_part_el_id[p])) w += d_part_weight[p];
        return w;
    };
    auto finite_or_zero = [](cudaComplexType v) -> cudaComplexType {
        if (!isfinite(v.x) || !isfinite(v.y))
            return CudaFunHelper::my_make_cuComplex(cuda_real_type(0), cuda_real_type(0));
        return v;
    };

    // ---- the row's total raw participation -----------------------------------
    // Over the machines it actually leaves participating, both families. When
    // nothing is left sharing, every unit sits at its (row) set-point and is
    // still checked there (upstream's anything_shared).
    cuda_real_type total_raw_w = 0;
    for (int s = 0; s < n_part_bus; ++s) {
        if (!bus_live(d_part_bus[s])) continue;
        total_raw_w += slot_raw_w(s);
    }
    const bool anything_shared = fabs(total_raw_w) > eps;

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
        const bool shares = anything_shared && takes_share(etype, eid);
        const cuda_real_type bus_raw_w = (shares && slot >= 0) ? slot_raw_w(slot) : cuda_real_type(0);
        if (shares && fabs(bus_raw_w) > eps) {
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
