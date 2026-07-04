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
// fill_FP_kernel
//
// Stores −ΔP at d_F[b * dim_J + k] for each pvpq bus k of contingency b.
//   ΔP = Re(V · conj(Ibus)) − Re(Sbus)
//
// Sbus is the shared base-case injection (not modified by contingencies).
//
// Parameters
// ----------
// d_F        : [actual_batch * dim_J] — mismatch RHS; ΔP written to [0, n_pvpq)
// d_V        : [actual_batch * n_bus] — current voltages
// d_Ibus     : [actual_batch * n_bus] — result of batched SpMV (Ybus · V)
// d_Sbus     : [n_bus] or [actual_batch * n_bus] — scheduled injections,
//              indexed as d_Sbus[b * sbus_stride + bus]
// pvpq       : [n_pvpq]              — sorted non-slack bus indices
// n_pvpq, n_bus, dim_J, actual_batch : dimensions
// sbus_stride: 0 (shared) or n_bus (per-batch row)
// ---------------------------------------------------------------------------
__global__ void fill_FP_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ pvpq,
    int n_pvpq,
    int n_bus,
    int dim_J,
    int actual_batch,
    int sbus_stride);

// ---------------------------------------------------------------------------
// fill_FQ_kernel
//
// Stores −ΔQ at d_F[b * dim_J + n_pvpq + k] for each pq bus k.
//   ΔQ = Im(V · conj(Ibus)) − Im(Sbus)
//
// Sbus is indexed as d_Sbus[b * sbus_stride + bus]; stride 0 falls back to
// the shared single-system access pattern.
// ---------------------------------------------------------------------------
__global__ void fill_FQ_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ pq_idx,
    int n_pvpq,
    int n_pq,
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
// update_Va_kernel
//
// Applies angle corrections dx[b * dim_J + k] to d_V[b * n_bus + pvpq[k]].
//   Va_new = Va_old + dx[k]
//   V_new  = |V| * exp(j * Va_new)
// Must complete before update_Vm_kernel (same stream → implicit order).
// ---------------------------------------------------------------------------
__global__ void update_Va_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ pvpq,
    int n_pvpq,
    int n_bus,
    int dim_J,
    int actual_batch);

// ---------------------------------------------------------------------------
// update_Vm_kernel
//
// Applies magnitude corrections dx[b * dim_J + n_pvpq + k] to pq buses.
//   Vm_new = Vm_old + dx[n_pvpq + k]
//   V_new  = Vm_new * exp(j * Va)
// Reads Va from d_V after update_Va_kernel has written it (same stream).
// ---------------------------------------------------------------------------
__global__ void update_Vm_kernel(
          cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_dx,
    const int*             __restrict__ pq_idx,
    int n_pvpq,
    int n_pq,
    int n_bus,
    int dim_J,
    int actual_batch);

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