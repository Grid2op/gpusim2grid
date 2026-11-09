// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SLOT_VC_HOLD_CUH
#define SLOT_VC_HOLD_CUH

// =============================================================================
// contingency/batch_sources/slot_vc_hold.cuh
//
// SlotVcHold — the controllers of the voltage-control groups each slot has lost
// (lightsim2grid PR #292; shared by ContingencyBatch and ScenarioSweepBatch): one a
// row trips, or on a bus it masks, is pinned at Q = 0 and never the reference of its
// group's sharing, the regulated bus going free when nobody is left. This is the
// outer loops' holding formulation (vc_hold_*_kernel) by value, on the entries
// reserve_tripped_controller_slots reserved, so one analysis still serves the batch.
// Nothing is allocated when no row trips one (the regular kernels run, bit-identical).
// =============================================================================

#include <algorithm>
#include <limits>
#include <vector>
#include <thrust/device_vector.h>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../nr_iter_step.cuh"      // NrIterBuffers
#include "../slot_schedule.cuh"        // SlotLoadView, SlotMoveView

struct SlotVcHold {
    std::vector<cuda_real_type>            h_all;      // [n_active * n_ctrl], NaN: not held
    int                                    n_ctrl = 0;
    thrust::device_vector<cuda_real_type>  d_all;      // n_active x n_ctrl
    thrust::device_vector<cuda_real_type>  d_batch;    // batch_size x n_ctrl
    thrust::device_vector<cuda_real_type>  d_none;     // n_ctrl NaN: a phantom slot
    thrust::device_vector<unsigned char>   d_no_plan;  // batch_size x n_ctrl zeros

    bool any() const { return !h_all.empty() && n_ctrl > 0; }

    // `held_orig[r]`: the controllers row r has lost, ORIGINAL order (empty outer: none)
    void set_host(int n_ctrl_, const std::vector<std::vector<int>>& held_orig,
                  const std::vector<int>& active_to_orig)
    {
        h_all.clear();
        n_ctrl = n_ctrl_;
        if (held_orig.empty() || n_ctrl <= 0) return;
        h_all.assign(active_to_orig.size() * static_cast<size_t>(n_ctrl),
                     std::numeric_limits<cuda_real_type>::quiet_NaN());
        for (size_t a = 0; a < active_to_orig.size(); ++a) {
            const int o = active_to_orig[a];
            if (o < 0 || o >= static_cast<int>(held_orig.size())) continue;
            for (int c : held_orig[static_cast<size_t>(o)])
                h_all[a * static_cast<size_t>(n_ctrl) + static_cast<size_t>(c)] = static_cast<cuda_real_type>(0.);
        }
    }

    void upload(int batch_capacity, cudaStream_t cs)
    {
        if (!any()) { d_all.clear(); d_batch.clear(); d_none.clear(); d_no_plan.clear(); return; }
        upload_h2d(d_all, h_all.data(), h_all.size(), cs);
        d_batch.resize(static_cast<size_t>(batch_capacity) * n_ctrl);
        const std::vector<cuda_real_type> none(static_cast<size_t>(n_ctrl), std::numeric_limits<cuda_real_type>::quiet_NaN());
        upload_h2d(d_none, none.data(), none.size(), cs);
        const std::vector<unsigned char> no_plan(static_cast<size_t>(batch_capacity) * n_ctrl, 0);
        upload_h2d(d_no_plan, no_plan.data(), no_plan.size(), cs);
    }

    void load(const SlotLoadView& L, cudaStream_t cs)
    {
        if (!any()) return;
        launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_batch.data()),
                                    static_cast<const cuda_real_type*>(thrust::raw_pointer_cast(d_all.data())),
                                    static_cast<const cuda_real_type*>(thrust::raw_pointer_cast(d_none.data())),
                                    L, n_ctrl, cs);
    }

    void move_slots(const SlotMoveView& M, cudaStream_t cs)
    {
        if (!any() || d_batch.empty()) return;
        launch_move_slot_rows(thrust::raw_pointer_cast(d_batch.data()), n_ctrl, M, cs);
    }

    void fill(NrIterBuffers& buf) const
    {
        if (!any()) return;
        buf.vc_may_hold    = true;
        buf.d_oc_held_q    = thrust::raw_pointer_cast(d_batch.data());
        buf.d_oc_plan_held = thrust::raw_pointer_cast(d_no_plan.data());
    }
};

#endif  // SLOT_VC_HOLD_CUH
