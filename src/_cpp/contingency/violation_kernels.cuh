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
#include "../acpf_nr_kernels.cuh"   // hvdc_flows_pu (shared with the NR loop)
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
//   1. DIVERGENCE: if isnan(d_residuals[out_c]) || d_residuals[out_c] > tol,
//      write ONE record {element_type=GRID, element_id=-1, side=0,
//      violation_type=DIVERGENCE, value=residual, limit=tol} to slot 0, set
//      count=1, the three per-type totals to 0, and RETURN immediately -- V
//      is unreliable for a diverged system, do not run the bus/branch loops
//      on it. This kernel only ever runs for a contingency actually solved
//      this chunk, so both the NaN and the large-finite-residual case mean
//      the same thing here (solver ran, didn't converge) -- NOT_SIMULATED
//      (the OTHER new lightsim2grid code, for a contingency the pre-check
//      dropped before it ever reached this kernel) is written by the Python
//      session layer instead (get_violations(), off BatchPfDriver's
//      d_violation_count -1 sentinel), never by this kernel. Folding
//      DIVERGENCE into the same compact array costs nothing extra here
//      since the residual check is already needed as a precondition to
//      trusting V, and it lets a caller iterating get_violations() alone
//      see *why* a contingency has no useful data without a second round
//      trip to get_residuals(). Mirrors lightsim2grid's own
//      ViolationElementType::GRID / LimitViolationType::DIVERGENCE
//      (LimitViolation.hpp) exactly, though gpusim2grid additionally
//      populates value/limit with the actual residual/tol (lightsim2grid
//      itself leaves those NaN/unused for GRID entries).
//
//   2. Branch current, for each branch l in [0, n_branches):
//        skip if l appears in this active slot's tripped-branch list
//        (d_trip_start/d_trip_count/d_trip_branch_flat, indexed by the
//        GLOBAL active-slot id c_start+local_c -- typically 0-4 entries,
//        linear scan).
//        I_or = yff_eff[l]*V[from] + yft_eff[l]*V[to];  ka_or = |I_or|*base_current_A[l]*0.001
//        I_ex = ytf_eff[l]*V[from] + ytt_eff[l]*V[to];  ka_ex = |I_ex|*base_current_A[l]*0.001
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
// tol              : DIVERGENCE threshold (independent of tol_base)
// d_bus_vn_kv      : [n_bus] real, nominal per-bus kV
// d_bus_vmin_kv/d_bus_vmax_kv : [n_bus] real, kV, NaN = unconfigured
// d_branch_from/to : [n_branches] int, terminal bus indices
// d_yff_eff/yft_eff/ytf_eff/ytt_eff: [n_branches] complex, pi-model admittances
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
          int*             __restrict__ d_out_count_current);

// -----------------------------------------------------------------------------
// check_bus_q_violations_kernel  (compute_physical_violations, lightsim2grid
// PR #206 parity -- see BusQCheck.hpp there and bus_q_check_data.hpp here)
//
// One thread PER active slot, same ownership / no-atomics layout as
// check_limit_violations_kernel above: records land in this row's exclusive
// slice [out_c*K, out_c*K+K) in PLAN order (the order build_bus_q_plan lists
// the buses, ascending grid bus id), so the output is deterministic.
//
// For each checked bus k (solver id b = d_bus_solver[k]):
//
//   what the machines holding b had to produce (MVAr):
//       q_bus = ( imag( V_b . conj( sum_j Y_bj V_j ) ) - imag(Sbus_b) ) . sn_mva
//   i.e. the RAW reactive residual of the converged state. This is exactly
//   lightsim2grid's `imag(mis_bus) + sum Q_c`: its mis_bus is the raw residual
//   with each VoltageControl controller's -i.Q_c folded in, and the check adds
//   the Q_c back. Nothing else touches the imaginary part of the residual (the
//   distributed slack and the HVDC droop act on P rows only), and a regulating
//   machine's Q is never in Sbus -- so the raw residual IS its output, whether
//   it pins its own bus (classical PV) or belongs to a bordered control group.
//   A NaN neighbour voltage (a bus this row masked) counts as 0, exactly like
//   gen_v_adjoint_kernel: its coupling entries were patched out of Ybus by the
//   islanding trip, so the value is immaterial, but 0 * NaN would poison the
//   sum. A checked bus whose OWN V is NaN (masked) is skipped.
//
//   what they can produce, THIS row (MVAr):
//       q_min = sum over LIVE generators of gen_qmin + qmin_fixed
//             + bmin_sum . |V_b|^2 . sn_mva          (idem q_max)
//   a generator is live unless d_gen_off[out_c * n_gen + gen_id] (a
//   ScenarioSweep generator contingency; nullptr = none). nb_live == 0 (every
//   generator off, no station / SVC) => the bus is an ordinary PQ bus in this
//   row and is not checked (the residual there is nobody's output).
//
//   violation:  q_bus < q_min - tol_mvar  -> LOW_Q  (value q_bus, limit q_min)
//         else  q_bus > q_max + tol_mvar  -> HIGH_Q (value q_bus, limit q_max)
//   a non-finite limit disables that side; a non-finite q_bus skips the bus.
//
// Row gate: when d_residuals != nullptr and the row's residual is NaN or
// exceeds residual_tol, the row gets count = 0 and NOTHING else -- upstream
// reports an EMPTY entry for a non-converged row (never a DIVERGENCE record,
// which stays in get_violations()). d_residuals == nullptr disables the gate
// (the base-case "n" report, gated by the caller on the base solve's own
// convergence flag).
//
// Sbus indexing follows fill_FQ_kernel: d_Sbus[local_c * sbus_stride + b],
// stride 0 for a source whose Sbus is shared (ContingencyBatch), n_bus for a
// per-slot dense row (InjectionBatch / ScenarioSweepBatch). Launched even when
// n_check == 0 so every simulated row gets count 0 (distinct from the -1
// "never simulated" sentinel seeded by BatchPfDriver::set_bus_q_check).
//
// Capacity: the (K+1)-th record sets d_out_truncated[out_c] = 1 and is
// dropped (upstream has no cap; ours is bus_q_violation_capacity).
// -----------------------------------------------------------------------------
__global__ void check_bus_q_violations_kernel(
    const cudaComplexType* __restrict__ d_V,            // [actual_batch × n_bus], slot order, post mask-NaN
    const cudaComplexType* __restrict__ d_Yvals,        // [actual_batch × nnz_Y], slot order, patched
    const int*             __restrict__ d_Y_outer,      // [n_bus + 1] shared skeleton
    const int*             __restrict__ d_Y_inner,      // [nnz_Y]
    const cudaComplexType* __restrict__ d_Sbus,         // see sbus_stride
    int                                 sbus_stride,
    const cuda_real_type*  __restrict__ d_residuals,    // [n_rows] ORIGINAL order, or nullptr
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
    const unsigned char*   __restrict__ d_gen_off,      // [n_rows × n_gen] ORIGINAL order, or nullptr
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
          int*             __restrict__ d_out_truncated);

// -----------------------------------------------------------------------------
// check_hvdc_p_violations_kernel  (compute_physical_violations)
//
// OpenLoadFlow's HvdcAcEmulationLimits outer loop, first pass, as a post-solve
// detection: a droop ("AC emulation") HVDC line in LINEAR regime (status == 0)
// whose theta-driven flow leaves the AC bus above pmax in the direction it
// flows would be saturated by that loop (and the row re-solved with the flow
// pinned at pmax). Nothing is enforced here; it only reports.
//
// One thread per active slot, looping over the n_hvdc droop lines of the
// session (shared single-system arrays, see AcPfNrState). Per line e:
//     raw = p0 + k . (theta1 - theta2)                 (pu)
//     (p1, p2) = hvdc_flows_pu(status=0, raw, ...)     the flows LEAVING the AC
//                                                      buses into the hvdc
//     raw >= 0 and p1 > pmax12 + tol_pu -> side 1 (saturates 1->2), value p1, limit pmax12
//     raw <  0 and p2 > pmax21 + tol_pu -> side 2 (saturates 2->1), value p2, limit pmax21
// value / limit are reported in MW (x sn_mva); element_id is the GRID hvdc id
// (d_hvdc_id); element_type HVDC. A saturated line (status != 0) is pinned at
// pmax by construction and is not checked; a line with a NaN end voltage
// (masked) is skipped. Same row gate / result map / capacity / sentinel
// conventions as check_bus_q_violations_kernel above.
// -----------------------------------------------------------------------------
__global__ void check_hvdc_p_violations_kernel(
    const cudaComplexType* __restrict__ d_V,            // [actual_batch × n_bus], slot order
    const cuda_real_type*  __restrict__ d_residuals,    // [n_rows] ORIGINAL order, or nullptr
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
          int*             __restrict__ d_out_truncated);

#endif  // VIOLATION_KERNELS_CUH
