// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ACPF_NR_KERNELS_CUH
#define ACPF_NR_KERNELS_CUH

// =============================================================================
// acpf_nr_kernels.cuh
//
// Declarations for all batch AC NR device kernels.
// Used by both acpf_nr.cu (single-system, actual_batch=1) and
// contingency_solver.cu (multi-system batch).
//
// Thread-layout convention
// ------------------------
// Every kernel maps a flat thread index to a (b, k) pair:
//
//   b = tid / <per_system_count>   — which system in the batch [0, actual_batch)
//   k = tid % <per_system_count>   — which element within that system
//
// Launch width is always actual_batch * <per_system_count>, rounded up to BS.
//
// Memory layout convention
// ------------------------
// All batch buffers interleave systems contiguously:
//   buf[b * stride + k]
// where stride is the per-system count (n_bus, dim_J, nnz_Y, nnz_J).
//
// Single-system reference arrays (pvpq, pq, scatter maps, Ybus outer/inner)
// are read-only and shared across all systems in the batch — no tiling needed.
//
// Sbus indexing is controlled by an `sbus_stride` parameter passed to the
// fill_FP/FQ kernels:
//   sbus_stride = 0      → shared single-system Sbus  (single-system NR,
//                          contingency analysis: every batch element reads
//                          the same n_bus-sized array)
//   sbus_stride = n_bus  → per-batch Sbus row         (injection sweep:
//                          d_Sbus laid out as actual_batch × n_bus)
// The kernels index as `d_Sbus[b * sbus_stride + bus]`, so stride 0 falls
// back to the existing shared-pointer access pattern with no extra work.
// =============================================================================

#include "dtypes.hpp"
#include "cu_complex_utils.h"   // cudaComplexType, cuda_real_type, CudaFunHelper

// ---------------------------------------------------------------------------
// apply_contingencies_kernel
//
// Patches d_Ybus_values_batch in-place after it has been tiled from base-case
// values.  One thread per (contingency, triplet) pair within the current chunk.
//
// No atomicAdd is needed: by construction each (ctg_id, k) pair is unique
// — a single contingency cannot modify the same Ybus entry twice.
//
// Parameters
// ----------
// value_ptr_batch : [actual_batch * nnz_Y] complex — tiled Ybus values buffer
// ctg_id          : [n_updates] — chunk-relative contingency index [0, actual_batch)
// k_idx           : [n_updates] — CSR flat index into a single-system values array
// delta_re/im     : [n_updates] — complex delta to SUBTRACT from the entry
// nnz_Y           : nnz of one Ybus (stride between contingency value blocks)
// n_updates       : total number of patch entries in this chunk
// ---------------------------------------------------------------------------
__global__ void apply_contingencies_kernel(
          cudaComplexType* __restrict__ value_ptr_batch,
    const int*             __restrict__ ctg_id,
    const int*             __restrict__ k_idx,
    const cuda_real_type*  __restrict__ delta_re,
    const cuda_real_type*  __restrict__ delta_im,
    int nnz_Y,
    int n_updates);

// ---------------------------------------------------------------------------
// apply_gen_v_kernel
//
// Re-seeds |V| at each (chunk-relative row, active generator) pair to the
// caller's target vm_pu, keeping whatever angle tile_V already wrote there --
// mirrors lightsim2grid's GeneratorContainer::set_vm / modify_gen_v: a
// PV/slack bus's magnitude is never part of the NR unknown vector, so this is
// a value-only reseed of the initial guess, not a Jacobian change. One thread
// per (row, active generator) pair; d_active_bus/d_gen_v_all only ever list
// generators whose own bus is Vm-fixed (see GenVOverride's own doc), so no
// bus-type check is needed here.
//
// Parameters
// ----------
// d_V_batch    : [actual_batch * n_bus] complex — already tiled from base V
// d_gen_v_all  : [n_rows * k_active] real, row-major, row = GLOBAL scenario
//                index (row_offset..row_offset+actual_batch); NaN = leave
//                this (row, gen) untouched
// d_active_bus : [k_active] — AC-solver bus id of each active generator
//                column, same order as d_gen_v_all's columns
// row_offset   : first global row covered by this chunk
// k_active     : number of active generator columns
// actual_batch : batch slots actually used in this chunk
// n_bus        : bus count (d_V_batch's per-row stride)
// ---------------------------------------------------------------------------
__global__ void apply_gen_v_kernel(
          cudaComplexType* __restrict__ d_V_batch,
    const cuda_real_type*  __restrict__ d_gen_v_all,
    const int*              __restrict__ d_active_bus,
    int row_offset,
    int k_active,
    int actual_batch,
    int n_bus);

// ---------------------------------------------------------------------------
// fill_FP_kernel
//
// Stores −ΔP at the ledger P-equation row of each P bus.
//   ΔP = Re(V · conj(Ibus)) − Re(Sbus)
//   d_F[b * dim_J + p_rows[k]] = −ΔP,   bus = p_buses[k]
//
// Ledger-driven (augmented J): the (bus, row) pairs come from the NRLedger's
// P-equation list, so a bus' P row is no longer assumed to be its slot index.
// For a feature-free grid p_buses == sorted(pvpq) and p_rows == [0, n_pvpq),
// reproducing the legacy layout exactly.
//
// Sbus is indexed as d_Sbus[b * sbus_stride + bus]; stride 0 falls back to the
// shared single-system access pattern.
//
// Parameters
// ----------
// d_F        : [actual_batch * dim_J] — mismatch RHS
// d_V        : [actual_batch * n_bus] — current voltages
// d_Ibus     : [actual_batch * n_bus] — result of batched SpMV (Ybus · V)
// d_Sbus     : [n_bus] or [actual_batch * n_bus] — scheduled injections
// p_buses    : [n_p] — bus owning each P equation
// p_rows     : [n_p] — J row of each P equation
// n_p, n_bus, dim_J, actual_batch : dimensions
// sbus_stride: 0 (shared) or n_bus (per-batch row)
// ---------------------------------------------------------------------------
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
    int sbus_stride);

// ---------------------------------------------------------------------------
// fill_FQ_kernel
//
// Stores −ΔQ at the ledger Q-equation row of each Q bus.
//   ΔQ = Im(V · conj(Ibus)) − Im(Sbus)
//   d_F[b * dim_J + q_rows[k]] = −ΔQ,   bus = q_buses[k]
//
// Ledger-driven counterpart of fill_FP_kernel. For a feature-free grid
// q_buses == pq and q_rows == [n_pvpq, n_pvpq + n_pq), reproducing the legacy
// layout. Sbus indexed as d_Sbus[b * sbus_stride + bus].
// ---------------------------------------------------------------------------
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
    int sbus_stride);

// ---------------------------------------------------------------------------
// fill_J_kernel
//
// Fills the Jacobian for a batch of systems (actual_batch=1 for single-system NR).
// One thread per (contingency b, Ybus nnz k).
//
// Row recovery: binary search on the SINGLE-SYSTEM d_Ybus_outer using k.
//   Reusing the single-system outer avoids repeating the n_bus-wide search
//   in the block-diagonal outer (which would require subtracting b * nnz).
// Column: d_Ybus_inner[k]  (single-system).
// Values: d_Ybus_values_batch[b * nnz_Y + k]  (per-contingency patched copy).
// Scatter maps: d_map_j{11,12,21,22}[k]  (single-system, shared).
//   Output: d_J_values[b * nnz_J + map_jXX[k]]
//
// Parameters
// ----------
// d_J_values        : [actual_batch * nnz_J] — Jacobian values to fill
// d_V               : [actual_batch * n_bus]
// d_Ibus            : [actual_batch * n_bus]
// d_Ybus_outer      : [n_bus + 1]            — single-system RowMajor outer
// d_Ybus_inner      : [nnz_Y]                — single-system RowMajor inner
// d_Ybus_values_batch: [actual_batch * nnz_Y] — per-contingency patched values
// d_map_j{11,12,21,22}: [nnz_Y]             — scatter maps (single-system)
// n_bus, nnz_Y, nnz_J, actual_batch
// ---------------------------------------------------------------------------
__global__ void fill_J_kernel(
          cuda_real_type*  __restrict__ d_J_values,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const int*             __restrict__ d_Ybus_outer,
    const int*             __restrict__ d_Ybus_inner,
    const cudaComplexType* __restrict__ d_Ybus_values_batch,
    const int*             __restrict__ d_map_j11,
    const int*             __restrict__ d_map_j12,
    const int*             __restrict__ d_map_j21,
    const int*             __restrict__ d_map_j22,
    int n_bus,
    int nnz_Y,
    int nnz_J,
    int actual_batch);

// ---------------------------------------------------------------------------
// NR step-scaling (MaxVoltageChange) -- see acpf_nr_kernels.cu's own doc
// above the definitions. Both no-ops unless the caller opts in (see
// NrIterBuffers::scaling_max_voltage_change).
// ---------------------------------------------------------------------------
__global__ void reduce_step_norms_kernel(
    const cuda_real_type* __restrict__ d_dx,
    const int*             __restrict__ theta_cols,
    const int*             __restrict__ vm_cols,
    int n_theta, int n_vm,
    int dim_J, int actual_batch,
    cuda_real_type* __restrict__ d_max_dtheta,
    cuda_real_type* __restrict__ d_max_dvm);

__global__ void apply_step_scale_kernel(
          cuda_real_type* __restrict__ d_dx,
    const cuda_real_type* __restrict__ d_max_dtheta,
    const cuda_real_type* __restrict__ d_max_dvm,
    cuda_real_type max_dVa, cuda_real_type max_dVm,
    int dim_J, int actual_batch);

// ---------------------------------------------------------------------------
// update_Va_kernel
//
// Applies angle corrections to each bus owning a theta unknown.
//   Va_new = Va_old + dx[b * dim_J + theta_cols[k]],   bus = theta_buses[k]
//   V_new  = |V| * exp(j * Va_new)
// Ledger-driven: the (bus, col) pairs come from the NRLedger's theta-unknown
// list. Feature-free: theta_buses == sorted(pvpq), theta_cols == [0, n_pvpq).
// Must complete before update_Vm_kernel (same stream → implicit order).
// ---------------------------------------------------------------------------
__global__ void update_Va_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ theta_buses,
    const int*             __restrict__ theta_cols,
    int n_theta,
    int n_bus,
    int dim_J,
    int actual_batch);

// ---------------------------------------------------------------------------
// update_Vm_kernel
//
// Applies magnitude corrections to each bus owning a vm unknown.
//   Vm_new = Vm_old + dx[b * dim_J + vm_cols[k]],   bus = vm_buses[k]
//   V_new  = Vm_new * exp(j * Va)
// Feature-free: vm_buses == pq, vm_cols == [n_pvpq, n_pvpq + n_pq).
// Reads Va from d_V after update_Va_kernel has written it (same stream).
// ---------------------------------------------------------------------------
__global__ void update_Vm_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ vm_buses,
    const int*             __restrict__ vm_cols,
    int n_vm,
    int n_bus,
    int dim_J,
    int actual_batch);

// ===========================================================================
// MultiSlack (distributed slack in the Jacobian) — Phase 2
//
// Three small per-slack kernels, launched only when the extension is active
// (slack_col >= 0). They implement, on the GPU, MultiSlack::adjust_mismatch /
// fill_feature_values / apply_step from lightsim2grid's NRSystem:
//   • adjust_slack_mismatch_kernel : mis += slack_absorbed · slack_weight  ⇒
//                                    d_F[p_row] -= slack_absorbed · weight
//   • fill_slack_feature_kernel    : J(slack_p_row, slack_col) = slack_weight
//   • update_slack_absorbed_kernel : slack_absorbed += dx[slack_col]
//
// Each slack bus owns a distinct P row / feature position, so within one batch
// slot the n_slack threads touch distinct entries — no atomics needed.
// ===========================================================================

// d_F[b*dim_J + slack_prow[k]] -= slack_absorbed[b] * slack_w[k]
// Run AFTER fill_FP/FQ (which wrote the bare −ΔP), using the pre-step
// slack_absorbed so the residual matches lightsim2grid's _residual.
__global__ void adjust_slack_mismatch_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_slack_absorbed,  // [actual_batch]
    const int*            __restrict__ d_slack_prow,       // [n_slack]
    const cuda_real_type* __restrict__ d_slack_w,          // [n_slack]
    int n_slack,
    int dim_J,
    int actual_batch);

// d_J_values[b*nnz_J + slack_feat_pos[k]] = slack_w[k]
// Run AFTER each fill_J (those J positions carry no dS contribution, so a plain
// assignment restamps the constant slack-weight coefficients).
__global__ void fill_slack_feature_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const int*            __restrict__ d_slack_feat_pos,   // [n_slack]
    const cuda_real_type* __restrict__ d_slack_w,          // [n_slack]
    int n_slack,
    int nnz_J,
    int actual_batch);

// Per-batch-slot slack_absorbed initial value = Re(Σ Sbus_slot) (one thread per
// slot). sbus_stride=0 → shared Sbus (every slot sums the same base injection);
// sbus_stride=n_bus → per-scenario Sbus row. Mirrors MultiSlack::update_state.
__global__ void init_slack_absorbed_kernel(
          cuda_real_type*  __restrict__ d_slack_absorbed,  // [actual_batch]
    const cudaComplexType* __restrict__ d_Sbus,
    int sbus_stride,
    int n_bus,
    int actual_batch);

// slack_absorbed[b] += dx[b*dim_J + slack_col]   (one thread per batch slot)
__global__ void update_slack_absorbed_kernel(
          cuda_real_type* __restrict__ d_slack_absorbed,   // [actual_batch]
    const cuda_real_type* __restrict__ d_dx,               // [actual_batch*dim_J]
    int slack_col,
    int dim_J,
    int actual_batch);

// ===========================================================================
// HVDC angle-droop ("AC emulation") — Phase 3
//
// The extension claims no row/column. Per connected droop line it adds, on the
// GPU (mirroring lightsim2grid's Hvdc extension):
//   • hvdc_adjust_mismatch_kernel : the theta-dependent droop flows leaving each
//       end bus into the HVDC  ⇒  d_F[p_row(end)] -= p_flow
//   • hvdc_fill_feature_kernel    : the (piecewise-constant) dP/dtheta slopes
//       ADDED onto the four (p_row(end), theta_col(end)) J positions.
//
// Two lines may share an end bus / J position, so both kernels use atomicAdd.
// Because the feature slopes ACCUMULATE onto (and some HVDC-only positions are
// NOT covered by) the dS fill, d_J_values must be zeroed before fill_J whenever
// HVDC is active (NrIterBuffers::zero_J_before_fill).
//
// Per-line arrays (length n_hvdc, shared single-system); the regime `status` is
// frozen (0 linear, +1 sat 1→2, -1 sat 2→1). h11/h12/h21/h22 and prow1/prow2 are
// -1 when the corresponding row/column does not exist (a slack end).
// ===========================================================================
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
    int actual_batch);

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
    int actual_batch);

// ===========================================================================
// VoltageControl (remote gen + SVC) — Phase 4 (bordered formulation)
//
// Per group of N controllers regulating reg_bus at v_set: N reactive-injection
// columns Q_c, 1 voltage-constraint custom row, N-1 reactive-sharing custom rows.
// Controllers / regulated bus stay PQ. The (constant) Jacobian feature entries
// are stamped via the shared fill_slack_feature_kernel (flat pos/value pairs).
// These kernels implement adjust_mismatch / fill_custom_rows / apply_step:
//   • vc_adjust_mismatch_kernel : d_F[q_row(c)] += Q_c   (reactive injection)
//   • vc_vrow_kernel            : d_F[v_row]  = -(Vm(reg) + Σ s_c·Q_c − v_set)
//   • vc_share_kernel           : d_F[sh_row] = -(w_1·Q_{k+1} − w_{k+1}·Q_1)
//   • vc_apply_step_kernel      : Q_c += dx[q_col(c)]
// All VC feature positions are feature-only (no dS overlap), and the custom rows
// are not bus rows, so the kernels assign (custom rows) / atomic-add (q rows).
// d_vc_q is the per-batch-slot running reactive injection (gen convention).
// ===========================================================================
__global__ void vc_adjust_mismatch_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,     // [actual_batch * n_ctrl]
    const int*            __restrict__ d_vc_qrow,  // [n_ctrl]
    int n_ctrl,
    int dim_J,
    int actual_batch);

__global__ void vc_vrow_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_vc_q,        // [actual_batch * n_ctrl]
    const cuda_real_type*  __restrict__ d_vc_slope,    // [n_ctrl]
    const int*             __restrict__ d_vc_reg_bus,  // [n_grp]
    const int*             __restrict__ d_vc_vrow,     // [n_grp]
    const int*             __restrict__ d_vc_grp_start,// [n_grp]
    const int*             __restrict__ d_vc_grp_count,// [n_grp]
    const cuda_real_type*  __restrict__ d_vc_vset,     // [n_grp]
    int n_grp,
    int n_ctrl,
    int n_bus,
    int dim_J,
    int actual_batch);

__global__ void vc_share_kernel(
          cuda_real_type* __restrict__ d_F,
    const cuda_real_type* __restrict__ d_vc_q,        // [actual_batch * n_ctrl]
    const int*            __restrict__ d_sh_row,      // [n_share]
    const int*            __restrict__ d_sh_first,    // [n_share] controller idx (group ref)
    const int*            __restrict__ d_sh_other,    // [n_share] controller idx (k+1)
    const cuda_real_type* __restrict__ d_sh_wfirst,   // [n_share] w of ref
    const cuda_real_type* __restrict__ d_sh_wother,   // [n_share] w of (k+1)
    int n_share,
    int n_ctrl,
    int dim_J,
    int actual_batch);

__global__ void vc_apply_step_kernel(
          cuda_real_type* __restrict__ d_vc_q,       // [actual_batch * n_ctrl]
    const cuda_real_type* __restrict__ d_dx,         // [actual_batch * dim_J]
    const int*            __restrict__ d_vc_qcol,    // [n_ctrl]
    int n_ctrl,
    int dim_J,
    int actual_batch);

// ---------------------------------------------------------------------------
// apply_bus_mask_kernel  (handle_disconnected_grid mode)
//
// Forces the masked buses' Newton-Raphson equations to identity rows so the
// largest connected component can be solved while the rest of the (split) grid
// is "frozen". One thread per (slot, row) identity entry:
//   • zero every J value in that row's nnz range  [J_outer[row], J_outer[row+1])
//   • write 1 at diag_nnz_pos (the (row, own-col) position)  [skipped if < 0]
//   • zero the RHS  d_F[slot * dim_J + row]
// so the linear solve returns dx = 0 for the masked unknowns and the live block
// is left untouched (cross-couplings to masked columns are ≈ 0 after the cut).
//
// Must run AFTER fill_F / fill_J and all feature stamps so it overrides them.
// Reused (J writes harmless) in the post-loop to zero masked F rows before the
// residual reduction. No-op when n_entries == 0 (mode off → bit-identical).
//
// d_J_values : [actual_batch * nnz_J]
// d_F        : [actual_batch * dim_J]
// d_mask_slot/d_mask_row/d_mask_diag : [n_entries] — batch slot, J row, diag pos
// d_J_outer  : [dim_J + 1] — shared single-system J skeleton row pointers
// ---------------------------------------------------------------------------
__global__ void apply_bus_mask_kernel(
          cuda_real_type* __restrict__ d_J_values,
          cuda_real_type* __restrict__ d_F,
    const int*           __restrict__ d_mask_slot,
    const int*           __restrict__ d_mask_row,
    const int*           __restrict__ d_mask_diag,
    const int*           __restrict__ d_J_outer,
    int nnz_J,
    int dim_J,
    int n_entries);

// ---------------------------------------------------------------------------
// mask_V_nan_kernel  (handle_disconnected_grid mode)
//
// Overwrites the voltage of every masked (frozen) bus with NaN+NaNj so the
// reported result distinguishes the unsolved component. One thread per
// (slot, bus) entry. Launched after the residual reduction, before the result
// store. No-op when n_entries == 0.
// ---------------------------------------------------------------------------
__global__ void mask_V_nan_kernel(
          cudaComplexType* __restrict__ d_V,
    const int*             __restrict__ d_maskv_slot,
    const int*             __restrict__ d_maskv_bus,
    cuda_real_type nan_val,
    int n_bus,
    int n_entries);

// ---------------------------------------------------------------------------
// compute_residuals_kernel
//
// After the final SpMV + fill_F pass, computes ‖F‖∞ for each contingency
// and writes it into d_residuals[c_start + b].
//
// One block per contingency (blockDim.x threads reduce over dim_J entries
// via shared memory).  This avoids the O(dim_J) sequential scan of a
// one-thread-per-contingency approach while staying simpler than a full
// Thrust reduction.
//
// Shared memory requirement: blockDim.x * sizeof(cuda_real_type).
// Recommended launch: <<<actual_batch, 256, 256*sizeof(cuda_real_type), cs>>>
//
// Parameters
// ----------
// d_residuals : [n_contingencies] — global residual output (c_start offset applied)
// d_F         : [actual_batch * dim_J] — mismatch vector after final fill_F
// dim_J       : length of F for one system
// actual_batch: number of contingencies in this chunk
// c_start     : active-slot offset for indexing into d_result_map
// d_result_map: [n_active] active-slot → original-index map, or nullptr for
//               identity (write residual at c_start + b directly)
// ---------------------------------------------------------------------------
__global__ void compute_residuals_kernel(
          cuda_real_type* __restrict__ d_residuals,
    const cuda_real_type* __restrict__ d_F,
    int dim_J,
    int actual_batch,
    int c_start,
    const int* __restrict__ d_result_map);

// ---------------------------------------------------------------------------
// scatter_V_results_kernel
//
// Copies a chunk's converged voltages from the compact batch buffer into the
// full-size result buffer at each system's ORIGINAL contingency index (via
// d_result_map).  Used when disconnected contingencies are compacted out of
// the batch, so the active-slot index differs from the result index.
//
// Parameters
// ----------
// d_V_results  : [n_contingencies * n_bus] complex — full-size output
// d_V_batch    : [actual_batch * n_bus]    complex — this chunk's converged V
// d_result_map : [n_active] active-slot → original-index map (must be non-null)
// c_start      : active-slot offset for this chunk
// n_bus        : buses per system
// actual_batch : number of active systems in this chunk
// ---------------------------------------------------------------------------
__global__ void scatter_V_results_kernel(
          cudaComplexType* __restrict__ d_V_results,
    const cudaComplexType* __restrict__ d_V_batch,
    const int*             __restrict__ d_result_map,
    int c_start,
    int n_bus,
    int actual_batch);

// ---------------------------------------------------------------------------
// compute_branch_flows_kernel
//
// Computes the current magnitude (in amperes) at the origin and extremity
// terminals of every branch for each contingency in the current chunk.
// Uses the π-model admittance parameters:
//   I_or = yff * V[from] + yft * V[to]   (origin / from-bus terminal)
//   I_ex = ytf * V[from] + ytt * V[to]   (extremity / to-bus terminal)
//
// The per-unit magnitude is multiplied by d_base_current_A[l] to give A:
//   d_base_current_A[l] = sn_mva * 1e6 / (sqrt(3) * bus_vn_kv[from[l]] * 1e3)
// (pre-computed on the host in set_branch_data and uploaded once).
//
// Thread layout: one thread per (b, l) pair.
//   b = tid / n_branches  — contingency index in batch [0, actual_batch)
//   l = tid % n_branches  — branch index [0, n_branches)
//
// Results are written into the full-size result buffers at offset
// out_c * n_branches + l, where out_c is the original contingency index
// (via d_result_map, or c_start + b when the map is null).
//
// Parameters
// ----------
// d_V              : [actual_batch * n_bus] complex — converged voltages
// d_branch_from/to : [n_branches] int — terminal bus indices
// d_yff/yft/ytf/ytt: [n_branches] complex — π-model admittances
// d_base_current_A : [n_branches] real — pre-computed I_base in A per branch
// d_or_amps        : [n_contingencies * n_branches] real — output origin amps
// d_ex_amps        : [n_contingencies * n_branches] real — output extremity amps
// n_bus, n_branches, c_start, actual_batch : dimensions / offsets
// d_result_map     : [n_active] active-slot → original-index map, or nullptr
//                    for identity (write at c_start + b directly)
// ---------------------------------------------------------------------------
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
    const int* __restrict__ d_result_map);

// ---------------------------------------------------------------------------
// zero_branch_flows_kernel
//
// Zeros out selected entries in the flow result buffers.
// Used after compute_branch_flows_kernel to clear flows for disconnected
// branches (a branch that is tripped in contingency c carries zero current).
//
// Parameters
// ----------
// d_or_amps      : [n_contingencies * n_branches] real — origin amps buffer
// d_ex_amps      : [n_contingencies * n_branches] real — extremity amps buffer
// d_zero_indices : [n_entries] flat indices c * n_branches + l to zero
// n_entries      : length of d_zero_indices
// ---------------------------------------------------------------------------
__global__ void zero_branch_flows_kernel(
          cuda_real_type* __restrict__ d_or_amps,
          cuda_real_type* __restrict__ d_ex_amps,
    const int*           __restrict__ d_zero_indices,
    int n_entries);

#endif // ACPF_NR_KERNELS_CUH