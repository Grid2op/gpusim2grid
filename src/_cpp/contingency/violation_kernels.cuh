// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef VIOLATION_KERNELS_CUH
#define VIOLATION_KERNELS_CUH

// =============================================================================
// contingency/violation_kernels.cuh
//
// check_limit_violations_kernel — the fused, on-device compute_limit_violations
// check. Invoked once per chunk, right after nr_mask_v_nan() (see
// BatchPfDriver::_solve_chunk), reading the chunk-local d_V_batch directly —
// never the full dense d_V_results/d_or_amps_results/d_ex_amps_results. Writes
// a bounded, per-contingency compact output (O(n_contingencies * K)) so the
// memory footprint and D->H transfer cost stay small regardless of batch size,
// mirroring lightsim2grid's opt-in compute_limit_violations flag but computed
// entirely on-device instead of as a per-element CPU loop.
// =============================================================================

#include "../dtypes.hpp"
#include "../cu_complex_utils.h"
#include "tripped_branch_table.hpp"

// -----------------------------------------------------------------------------
// check_limit_violations_kernel
//
// One thread PER CONTINGENCY (active slot) in the current chunk -- not per
// bus, not per (contingency, branch). Each thread owns output slice
// [out_c*K, out_c*K+K) exclusively (out_c = original contingency index), so
// there is no cross-thread write hazard and no atomics anywhere in this
// kernel. n_contingencies is typically large enough on its own to saturate
// the GPU at one-thread-per-contingency; total FLOPs are identical to a
// per-bus/per-branch layout either way, so this trades finer-grained
// parallelism for avoiding atomic-append machinery entirely.
//
// Checks, in order, matching lightsim2grid's ContingencyAnalysis.cpp
// (check_bus_voltage_violations / check_current_violations) semantics and
// units exactly (kV for bus, kA for branch current). Current is checked
// BEFORE bus voltage -- thermal/current violations are generally first-order
// operational concerns, voltage second-order:
//
//   1. DIVERGED: if d_residuals[out_c] > tol, write ONE record
//      {element_type=BUS(placeholder, unused), element_id=-1, side=0,
//      violation_type=DIVERGED, value=residual, limit=tol} to slot 0, set
//      count=1, the three per-type totals to 0, and RETURN immediately -- V
//      is unreliable for a diverged system, do not run the bus/branch loops
//      on it. This is a gpusim2grid extension beyond lightsim2grid's own
//      semantics (which keeps `converged()` and `get_violations()` as two
//      separate accessors); folding it into the same compact array costs
//      nothing extra here since the residual check is already needed as a
//      precondition to trusting V, and it lets a caller iterating
//      get_violations() alone see *why* a contingency has no useful data
//      without a second round trip to get_residuals().
//
//   2. Branch current, for each branch l in [0, n_branches):
//        skip if l appears in this active slot's tripped-branch list
//        (d_trip_start/d_trip_count/d_trip_branch_flat, indexed by the
//        GLOBAL active-slot id c_start+local_c -- typically 0-4 entries,
//        linear scan).
//        I_or = yff[l]*V[from] + yft[l]*V[to];  ka_or = |I_or|*base_current_A[l]*0.001
//        I_ex = ytf[l]*V[from] + ytt[l]*V[to];  ka_ex = |I_ex|*base_current_A[l]*0.001
//        (the /1000 converts gpusim2grid's existing Amps-basis
//        d_base_current_A into kA, matching branch_limit_a1_ka/a2_ka's units;
//        base_current_A itself is untouched, this kernel is the only place
//        that rescales it.)
//        if !isnan(limit1[l]) && ka_or > limit1[l] -> CURRENT, side=1 (d_out_count_current++)
//        if !isnan(limit2[l]) && ka_ex > limit2[l] -> CURRENT, side=2 (d_out_count_current++)
//
//   3. Bus voltage, for each bus b in [0, n_bus):
//        vm_kv = |V[local_c*n_bus + b]| * d_bus_vn_kv[b]
//        (V already has masked buses set to NaN by nr_mask_v_nan, which ran
//        before this kernel launches in _solve_chunk -- NaN comparisons are
//        false in IEEE754, so masked buses are excluded with NO special
//        casing here.)
//        if !isnan(vmin[b]) && vm_kv < vmin[b]  -> LOW_VOLTAGE (d_out_count_low_voltage++)
//        else if !isnan(vmax[b]) && vm_kv > vmax[b] -> HIGH_VOLTAGE (d_out_count_high_voltage++)
//
// Detail-record capacity: on the (K+1)-th write attempt for a contingency,
// stop writing further detail records (clamp) and set d_out_truncated[out_c]
// = 1 -- but keep SCANNING every bus/branch regardless, so the three
// per-type totals (d_out_count_low_voltage/high_voltage/current) are always
// the TRUE, uncapped violation counts for that contingency, independent of
// violation_capacity (K). This is the reason these loops do not stop early
// on truncation like the detail-record buffer does.
//
// element_id de-concatenation (lines-then-trafos -> lightsim2grid's own-type
// local id, per LimitViolation.hpp's documented convention):
//   element_type = (l < n_lines) ? LINE : TRAFO
//   element_id   = (l < n_lines) ? l    : l - n_lines
// n_lines is passed in as a plain int (branch ordering is a session-level
// invariant, not derivable from n_branches alone).
//
// Parameters
// ----------
// d_V              : [actual_batch*n_bus] complex, chunk-local, post mask-NaN
// d_residuals      : [n_contingencies] real, ORIGINAL-index space (already
//                    written earlier this chunk by compute_residuals_kernel)
// tol              : DIVERGED threshold (independent of tol_base)
// d_bus_vn_kv      : [n_bus] real, nominal per-bus kV
// d_bus_vmin_kv/d_bus_vmax_kv : [n_bus] real, kV, NaN = unconfigured
// d_branch_from/to : [n_branches] int, terminal bus indices
// d_yff/yft/ytf/ytt: [n_branches] complex, pi-model admittances
// d_base_current_A : [n_branches] real, Amps-basis (kernel /1000 for kA)
// d_branch_limit_a1_ka/a2_ka : [n_branches] real, kA, NaN = unconfigured
// d_trip_start/d_trip_count/d_trip_branch_flat : tripped-branch lookup table,
//                    indexed by GLOBAL active-slot id (c_start + local_c), or
//                    all-nullptr when the batch source never trips branches
//                    (e.g. injection sweep -- this kernel is never launched
//                    from that path, but the parameters stay generic)
// n_bus, n_branches, n_lines : dimensions
// c_start, actual_batch : this chunk's active-slot offset / size
// K                : violation_capacity, output slots per contingency
// d_result_map     : [n_active] active-slot -> original-index map, or nullptr
//                    for identity (c_start + local_c directly)
// d_out_*          : [n_contingencies * K] compact SoA output (see
//                    ContingencyAnalysisSession::get_violation_*())
// d_out_count      : [n_contingencies]; -1 = not simulated, else 0..K
//                    (number of DETAIL records written, capped at K)
// d_out_truncated  : [n_contingencies]; 0/1
// d_out_count_low_voltage/high_voltage/current : [n_contingencies]; -1 = not
//                    simulated, else the TRUE, UNCAPPED count of violations
//                    of that type (independent of K / violation_capacity --
//                    always accurate even when d_out_truncated is 1).
// -----------------------------------------------------------------------------
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
          int*             __restrict__ d_out_count_current);

#endif  // VIOLATION_KERNELS_CUH
