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

// The solved groups of a continuous control (BranchControl's ratio groups, ShuntControl's
// groups): group g's members are member[start[g] .. start[g+1]) (indices into the control's
// entries), one row each (row, same order); jvm the J position of (row, vm_col(bus)) per member,
// jcol of (row_j, col_m) per (j, m) pair, flat from jstart[g]. Each row holds by value
//     x_j = x_target_j               (its control off)
//     Vm(bus) = target               (the first one on)
//     mean(x) = x_j                  (any other one on)
// (BranchControl / ShuntControl::fill_custom_rows).
struct CtlGroups {
    int n_grp = 0;
    const int* start = nullptr;       // [n_grp + 1]
    const int* member = nullptr;      // flat
    const int* row = nullptr;         // flat
    const int* jvm = nullptr;         // flat
    const int* jstart = nullptr;      // [n_grp]
    const int* jcol = nullptr;        // flat, n_j^2 per group
    const int* bus = nullptr;         // [n_grp] the regulated bus
    const cuda_real_type* target = nullptr;   // [n_grp] pu
};

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
    CtlGroups grp;                        // its solved ratio groups (the members' ratio columns)
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

// ---- ShuntControl (ShuntVoltageControl): the regulating shunts of a bus are ONE controller
// with a susceptance B (pu; its bus' Ybus diagonal holds j B), patched by value; a solved
// group's controller owns a column, dQ/dB = -|V|^2 in its bus' Q row
struct ScStatic {
    int n_sc = 0;
    const int* bus = nullptr;             // [n_sc] solver bus
    const int* col = nullptr;             // [n_sc] the susceptance column, -1 none
    const int* jq = nullptr;              // [n_sc] J position of (q_row, col), -1 none
    const int* ypos = nullptr;            // [n_sc] Ybus position of (bus, bus)
    const cuda_real_type* b0 = nullptr;   // [n_sc] the grid's susceptance
    CtlGroups grp;                        // its solved groups
};
struct ScSlots {   // per slot, [batch * n_sc]
    cuda_real_type* b = nullptr;
    cuda_real_type* b_applied = nullptr;  // what the slot's Ybus holds
    cuda_real_type* b_target = nullptr;
    unsigned char*  on = nullptr;
};

// One thread per (slot, controller with a column): dQ/dB.
__global__ void sc_fill_J_kernel(cuda_real_type* __restrict__ d_J, const cudaComplexType* __restrict__ d_V,
                                 ScStatic st, ScSlots sl, int n_bus, int nnz_J, int batch);
// One thread per (slot, controller with a column): the step on B, then its Ybus patch.
__global__ void sc_step_kernel(const cuda_real_type* __restrict__ d_dx, cudaComplexType* __restrict__ d_Ybus,
                               ScStatic st, ScSlots sl, int dim_J, int nnz_Y, int batch);
// One thread per (listed slot, controller): NROuterInner::apply_controls on ShuntControl. `fresh`:
// ShuntControl::_reset at the grid's sections; then the sections the loop switched (d_set:
// B = d_b_new, its target too), then the requested control (d_ctl -1 / 0 off, target where B
// is / 1 on), then the Ybus patch. Inputs [n_list * n_sc].
__global__ void sc_controls_kernel(cudaComplexType* __restrict__ d_Ybus, ScStatic st, ScSlots sl,
                                   const int* __restrict__ d_slots, int n_list, bool fresh,
                                   const unsigned char* __restrict__ d_set,
                                   const cuda_real_type* __restrict__ d_b_new,
                                   const signed char* __restrict__ d_ctl, int nnz_Y);
// One thread per (slot, controller), the slots whose d_evict entry is >= 0 (null: every slot):
// the slot's Ybus back to the grid's susceptance.
__global__ void sc_restore_kernel(cudaComplexType* __restrict__ d_Ybus, ScStatic st, ScSlots sl,
                                  const int* __restrict__ d_evict, int nnz_Y, int batch);
// One thread per (listed slot, controller): its B into the gather buffer at `offset`.
__global__ void sc_gather_kernel(cuda_real_type* __restrict__ d_out, ScStatic st, ScSlots sl,
                                 const int* __restrict__ d_slots, int n_list, int offset, int stride);

// One thread per (slot, group): its rows' residuals (assign), see CtlGroups. x, x_target,
// on: the control's per slot values ([batch * n_ent]).
__global__ void ctl_group_mismatch_kernel(cuda_real_type* __restrict__ d_F, const cudaComplexType* __restrict__ d_V,
                                          CtlGroups g, const cuda_real_type* __restrict__ x,
                                          const cuda_real_type* __restrict__ x_target,
                                          const unsigned char* __restrict__ on, int n_ent,
                                          int n_bus, int dim_J, int batch);

// One thread per (slot, group): its rows' J entries, every position assigned.
__global__ void ctl_group_fill_J_kernel(cuda_real_type* __restrict__ d_J, CtlGroups g,
                                        const unsigned char* __restrict__ on, int n_ent, int nnz_J, int batch);

// One thread per slot, its transformers in order (parallel transformers patch the same Ybus
// coefficients): the Newton step on the shift and the ratio (BranchControl::
// apply_step), then the block patched into the slot's Ybus (d_Ybus may be null: the
// state moves, nothing is patched).
__global__ void bc_step_kernel(const cuda_real_type* __restrict__ d_dx, cudaComplexType* __restrict__ d_Ybus,
                               BcStatic st, BcSlots sl, int dim_J, int nnz_Y, int batch);

// One thread per listed slot, its transformers in order: what the outer-loop model asks of them before
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

// One thread per slot, its transformers in order; the slots whose d_evict entry is >= 0
// (null: every slot): the slot's Ybus back to what it held when the row was loaded (the
// container's block, nothing for a transformer the row opens), for the next row.
__global__ void bc_restore_kernel(cudaComplexType* __restrict__ d_Ybus, BcStatic st, BcSlots sl,
                                  const int* __restrict__ d_evict, int nnz_Y, int batch);

// One thread per (listed slot, transformer): its shift and ratio into the gather buffer
// (row i of `d_out`, stride `stride`, the shifts at `offset`, the ratios after them).
__global__ void bc_gather_kernel(cuda_real_type* __restrict__ d_out, BcStatic st, BcSlots sl,
                                 const int* __restrict__ d_slots, int n_list, int offset, int stride);

#endif  // GPUSIM2GRID_BRANCH_CONTROL_KERNELS_CUH
