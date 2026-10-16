// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/branch_control_kernels.cuh -- lightsim2grid's BranchControl NR extension
// (NRSystemBranchControl.cpp) per slot: the transformers the outer loops
// (PhaseControl, TransformerVoltageControl) handle, whose pi block lives in the
// slot's Ybus and is PATCHED BY VALUE whenever their shift a, their ratio rho or
// a tap moves -- the difference to the block stamped there, so whatever else sits
// on those coefficients keeps its part. A phase controller solving for its shift
// owns a column (a) and a row, holding by value
//     F = a - a_target                 (control off)
//     F = P_side(V, a) - p_target      (control on)
// and dS/da sits in the P and Q rows of its two buses. Everything is a by-value
// edit on lightsim2grid's union pattern: one cuDSS analysis serves the batch.
//
// Static data (per transformer, LedgerData::bc_* order) in BcStatic, the running
// state (per slot and transformer, BranchControl::Entry) in BcSlots. Both empty
// (n_bc == 0) unless the ledger has such transformers: every launch is skipped.
// =============================================================================

#ifndef GPUSIM2GRID_BRANCH_CONTROL_KERNELS_CUH
#define GPUSIM2GRID_BRANCH_CONTROL_KERNELS_CUH

#include "../cu_complex_utils.h"
#include "../dtypes.hpp"

struct BcStatic {
    int n_bc = 0;
    const int* b1 = nullptr;              // [n_bc] solver buses
    const int* b2 = nullptr;
    const int* side = nullptr;            // [n_bc] the regulated side (1, 2)
    const int* row = nullptr;             // [n_bc] the shift row, -1 without a shift column
    const int* col = nullptr;             // [n_bc] the shift column, -1 without
    const int* ypos = nullptr;            // [4 n_bc] Ybus value positions: (b1,b1) (b1,b2) (b2,b1) (b2,b2)
    // [9 n_bc] J positions (-1 absent): (p1,col) (q1,col) (p2,col) (q2,col) (row,col)
    // (row,th1) (row,th2) (row,vm1) (row,vm2)
    const int* jpos = nullptr;
    const cuda_real_type* sign = nullptr; // [n_bc] TrafoContainer::shift_sign
    const cuda_real_type* rexp = nullptr; // [n_bc] TrafoContainer::ratio_exponent
    const cuda_real_type* p_target = nullptr;   // [n_bc] pu
    const cudaComplexType* tap0 = nullptr;      // [4 n_bc] the block at the grid's taps
    const cudaComplexType* yeff0 = nullptr;     // [4 n_bc] the block the container stamps
    const cuda_real_type* a_tap0 = nullptr;     // [n_bc] the grid's taps' shift and ratio
    const cuda_real_type* rho_tap0 = nullptr;
    // the ratio columns (TransformerVoltageControl): per transformer its column (-1 none) and
    // the J positions of dS/drho, (p1,rcol) (q1,rcol) (p2,rcol) (q2,rcol)
    const int* rcol = nullptr;            // [n_bc]
    const int* jpos_r = nullptr;          // [4 n_bc]
    // its solved ratio groups: group g's members are grp_member[grp_start[g] .. grp_start[g+1]),
    // one row each (grp_row, same order); grp_jvm the J position of (row, vm_col(bus)) per
    // member, grp_jcol of (row_j, rcol_m) per (j, m) pair, flat from grp_jstart[g]
    int n_grp = 0;
    const int* grp_start = nullptr;       // [n_grp + 1]
    const int* grp_member = nullptr;      // flat
    const int* grp_row = nullptr;         // flat
    const int* grp_jvm = nullptr;         // flat
    const int* grp_jstart = nullptr;      // [n_grp]
    const int* grp_jcol = nullptr;        // flat, n_j^2 per group
    const int* grp_bus = nullptr;         // [n_grp] the regulated bus
    const cuda_real_type* grp_target = nullptr;   // [n_grp] pu
};

struct BcSlots {   // per slot, [batch * n_bc] (blocks [batch * n_bc * 4])
    cuda_real_type*  a = nullptr;
    cuda_real_type*  rho = nullptr;
    cuda_real_type*  a_target = nullptr;
    cuda_real_type*  a_tap = nullptr;
    cuda_real_type*  rho_tap = nullptr;
    cudaComplexType* tap = nullptr;       // the block at the slot's taps
    cudaComplexType* applied = nullptr;   // the block stamped in the slot's Ybus now
    cuda_real_type*  rho_target = nullptr;
    unsigned char*   on = nullptr;        // phase control on
    unsigned char*   ron = nullptr;       // ratio (voltage) control on
    unsigned char*   out = nullptr;       // the row does not have it: nothing stamped, never patched
};

// One thread per (slot, transformer with a shift column): the shift row's residual
// (assign; d_F = -F).
__global__ void bc_mismatch_kernel(cuda_real_type* __restrict__ d_F, const cudaComplexType* __restrict__ d_V,
                                   BcStatic st, BcSlots sl, int n_bus, int dim_J, int batch);

// One thread per (slot, transformer with a shift column): dS/da in the P and Q rows of
// its buses and the shift row, both forms (every position assigned).
__global__ void bc_fill_J_kernel(cuda_real_type* __restrict__ d_J, const cudaComplexType* __restrict__ d_V,
                                 BcStatic st, BcSlots sl, int n_bus, int nnz_J, int batch);

// One thread per (slot, solved ratio group): its rows' residuals (assign), BranchControl::
// fill_custom_rows: per member, off "rho = rho_target", the first one on "Vm(bus) = target",
// any other one on "mean(rho) = rho".
__global__ void bc_group_mismatch_kernel(cuda_real_type* __restrict__ d_F, const cudaComplexType* __restrict__ d_V,
                                         BcStatic st, BcSlots sl, int n_bus, int dim_J, int batch);

// One thread per (slot, solved ratio group): its rows' J entries, every position assigned.
__global__ void bc_group_fill_J_kernel(cuda_real_type* __restrict__ d_J, BcStatic st, BcSlots sl,
                                       int nnz_J, int batch);

// One thread per (slot, transformer): the Newton step on the shift (BranchControl::
// apply_step), then the block patched into the slot's Ybus (d_Ybus may be null: the
// state moves, nothing is patched).
__global__ void bc_step_kernel(const cuda_real_type* __restrict__ d_dx, cudaComplexType* __restrict__ d_Ybus,
                               BcStatic st, BcSlots sl, int dim_J, int nnz_Y, int batch);

// One thread per (listed slot, transformer): what the outer-loop model asks of it before
// the slot is solved (outer::SlotControls, lightsim2grid's NROuterInner::apply_controls).
// `fresh`: the slot was just loaded -- BranchControl::_reset at the grid's taps, out from
// d_out (nothing stamped then), the block the container stamped taken as applied. Then
// a re-tap (d_retap set: the taps' block, shift, ratio from d_tap / d_a_tap / d_rho_tap,
// the shift and ratio back to them), then the requested control (d_ctl: -1 untouched,
// 0 off -- its target where the shift is --, 1 on; d_rctl the same for the ratio), then the
// block patched into Ybus.
// The per-listed-slot inputs are [n_list * n_bc] (blocks [n_list * n_bc * 4]).
__global__ void bc_controls_kernel(cudaComplexType* __restrict__ d_Ybus, BcStatic st, BcSlots sl,
                                   const int* __restrict__ d_slots, int n_list, bool fresh,
                                   const unsigned char* __restrict__ d_out,
                                   const unsigned char* __restrict__ d_retap,
                                   const cudaComplexType* __restrict__ d_tap,
                                   const cuda_real_type* __restrict__ d_a_tap,
                                   const cuda_real_type* __restrict__ d_rho_tap,
                                   const signed char* __restrict__ d_ctl,
                                   const signed char* __restrict__ d_rctl,
                                   int nnz_Y);

// One thread per (slot, transformer), the slots whose d_evict entry is >= 0 (null:
// every slot): the slot's Ybus back to what it held when the row was loaded (the
// container's block, nothing for a transformer the row opens), for the next row.
__global__ void bc_restore_kernel(cudaComplexType* __restrict__ d_Ybus, BcStatic st, BcSlots sl,
                                  const int* __restrict__ d_evict, int nnz_Y, int batch);

// One thread per (listed slot, transformer): its shift and ratio into the gather buffer
// (row i of `d_out`, stride `stride`, the shifts at `offset`, the ratios after them).
__global__ void bc_gather_kernel(cuda_real_type* __restrict__ d_out, BcStatic st, BcSlots sl,
                                 const int* __restrict__ d_slots, int n_list, int offset, int stride);

#endif  // GPUSIM2GRID_BRANCH_CONTROL_KERNELS_CUH
