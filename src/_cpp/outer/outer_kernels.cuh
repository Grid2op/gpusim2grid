// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/outer_kernels.cuh -- the per-slot control state of the outer loops on
// the device (outer/outer_hook.hpp's SlotControls), and what the scheduler
// gathers from a converged slot for the host to decide on.
//
// Each NR kernel here is a by-value edit on lightsim2grid's union pattern (the
// ledger of an NROuter_* solve), so one cuDSS analysis still serves the batch:
//   - oc_pin_rows_kernel       : the switchable buses still PV in their slot,
//                                Q row identity-pinned (NRSystem::set_pv_pinned_buses)
//   - vc_hold_fill_J_kernel /  : VoltageControl with controllers held by value
//     vc_hold_rows_kernel        (lightsim2grid's may_hold_ formulation,
//                                VoltageControl::_fill_*_holding)
// =============================================================================

#ifndef GPUSIM2GRID_OUTER_KERNELS_CUH
#define GPUSIM2GRID_OUTER_KERNELS_CUH

#include "../cu_complex_utils.h"
#include "../dtypes.hpp"

// One thread per (slot, switchable bus): a bus whose flag is set has its Q row
// replaced by the identity on its Vm column (J; nullptr skips) and a zero
// residual (F; nullptr skips).
__global__ void oc_pin_rows_kernel(
          cuda_real_type* __restrict__ d_J_values,
          cuda_real_type* __restrict__ d_F,
    const unsigned char*  __restrict__ d_pin,       // [batch * n_pin]
    const int*            __restrict__ d_pin_row,   // [n_pin]
    const int*            __restrict__ d_pin_diag,  // [n_pin]
    const int*            __restrict__ d_J_outer,
    int n_pin, int nnz_J, int dim_J, int batch);

// One thread per (slot, group): the J values of the group's bordered block when
// controllers may be held by value. Assigns every position it owns (an SVC's
// slope or the first controller's (v_row, q_col) slot, (v_row, vm_col(reg)),
// and each sharing row's entry on every controller of the group).
__global__ void vc_hold_fill_J_kernel(
          cuda_real_type* __restrict__ d_J_values,
    const cuda_real_type* __restrict__ d_held_q,     // [batch * n_ctrl] NaN: not held by value
    const unsigned char*  __restrict__ d_plan_held,  // [batch * n_ctrl]
    const cuda_real_type* __restrict__ d_slope,      // [n_ctrl]
    const cuda_real_type* __restrict__ d_weight,     // [n_ctrl]
    const int*            __restrict__ d_slope_pos,  // [n_ctrl] (v_row, q_col), -1 none
    const int*            __restrict__ d_vm_pos,     // [n_grp] (v_row, vm_col(reg)), -1 none
    const int*            __restrict__ d_rowpos_off, // [n_ctrl] -1 for a first controller
    const int*            __restrict__ d_rowpos,     // flat, grp_count entries per sharing row
    const int*            __restrict__ d_grp_start,
    const int*            __restrict__ d_grp_count,
    int n_ctrl, int n_grp, int nnz_J, int batch);

// One thread per (slot, group): the voltage row and the sharing rows' residuals.
__global__ void vc_hold_rows_kernel(
          cuda_real_type*  __restrict__ d_F,
    const cudaComplexType* __restrict__ d_V,
    const cuda_real_type*  __restrict__ d_vc_q,       // [batch * n_ctrl]
    const cuda_real_type*  __restrict__ d_held_q,     // [batch * n_ctrl]
    const unsigned char*   __restrict__ d_plan_held,  // [batch * n_ctrl]
    const cuda_real_type*  __restrict__ d_q_plan,     // [n_ctrl] a plan-held controller's frozen output, or nullptr
    const cuda_real_type*  __restrict__ d_vset,       // [n_grp] or per slot
    int vset_stride,
    const cuda_real_type*  __restrict__ d_slope,
    const cuda_real_type*  __restrict__ d_weight,
    const int*             __restrict__ d_reg_bus,
    const int*             __restrict__ d_vrow,
    const int*             __restrict__ d_sh_row,     // [n_ctrl] -1 for a first controller
    const int*             __restrict__ d_grp_start,
    const int*             __restrict__ d_grp_count,
    int n_ctrl, int n_grp, int n_bus, int dim_J, int batch);

// One thread per (listed slot, watched bus): V, V.conj(Ibus) and Sbus there, as
// cuda_real_type, out[i * stride + {0..5} * n_watch + w]; then the slot's
// controller Q at out[i * stride + 6 * n_watch + c].
__global__ void oc_gather_kernel(
          cuda_real_type*  __restrict__ d_out,
    const cudaComplexType* __restrict__ d_V,
    const cudaComplexType* __restrict__ d_Ibus,
    const cudaComplexType* __restrict__ d_Sbus,
    int sbus_stride,
    const cuda_real_type*  __restrict__ d_vc_q,
    const int*             __restrict__ d_slots,      // [n_list]
    const int*             __restrict__ d_watch,      // [n_watch]
    int n_list, int n_watch, int n_ctrl, int n_bus, int stride);

// One thread per (listed slot): 1 when some candidate bus whose magnitude is a
// Newton unknown in that slot (not a pinned switchable bus) is outside
// [vmin, vmax], written to out[i * stride + stride - 1].
__global__ void oc_unrealistic_kernel(
          cuda_real_type*  __restrict__ d_out,
    const cudaComplexType* __restrict__ d_V,
    const unsigned char*   __restrict__ d_pin,        // [S * n_pin] or nullptr
    const int*             __restrict__ d_cand_bus,   // [n_cand]
    const int*             __restrict__ d_cand_pin,   // [n_cand] its switchable index, -1 none
    const int*             __restrict__ d_slots,
    int n_list, int n_cand, int n_pin, int n_bus, int stride,
    cuda_real_type vmin, cuda_real_type vmax);

// One thread per entry: d_Sbus[slot * n_bus + bus] += (dp, dq).
__global__ void oc_sbus_delta_kernel(
          cudaComplexType* __restrict__ d_Sbus,
    const int*             __restrict__ d_slot,
    const int*             __restrict__ d_bus,
    const cuda_real_type*  __restrict__ d_dp,
    const cuda_real_type*  __restrict__ d_dq,
    int n, int n_bus);

#endif  // GPUSIM2GRID_OUTER_KERNELS_CUH
