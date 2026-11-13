// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SLOT_HVDC_STATUS_CUH
#define SLOT_HVDC_STATUS_CUH

// =============================================================================
// contingency/batch_sources/slot_hvdc_status.cuh
//
// SlotHvdcStatus — the regime of the lines in AC emulation, per slot, for a batch
// source whose rows trip some of them (lightsim2grid PR #292; shared by
// ContingencyBatch and ScenarioSweepBatch, like SlotSlackRedistribution): a tripped
// line carries no flow and adds no slope (HVDC_STATUS_TRIPPED), on the Jacobian
// entries already declared for every regime, so one analysis still serves the batch.
// No row tripping one: nothing is allocated and every slot reads the base case's
// shared array (stride 0, bit-identical). The outer loops own each slot's regimes
// instead (BatchPfDriver::d_oc_hvdc_).
// =============================================================================

#include <algorithm>
#include <vector>
#include <thrust/device_vector.h>

#include "../../ledger_data.hpp"       // HVDC_STATUS_TRIPPED
#include "../../cuda_utils.h"
#include "../../nr_iter_step.cuh"      // NrIterBuffers
#include "../slot_schedule.cuh"        // SlotLoadView, SlotMoveView

struct SlotHvdcStatus {
    std::vector<int>           h_all;    // [n_active * n_hvdc], active order
    int                        n_hvdc = 0;
    thrust::device_vector<int> d_all;    // n_active x n_hvdc
    thrust::device_vector<int> d_batch;  // batch_size x n_hvdc

    bool any() const { return !h_all.empty() && n_hvdc > 0; }

    // `off_orig[r]`: the entries (positions in `base_status`) row r trips, ORIGINAL order
    void set_host(const std::vector<int>& base_status, const std::vector<std::vector<int>>& off_orig,
                  const std::vector<int>& active_to_orig)
    {
        h_all.clear();
        n_hvdc = static_cast<int>(base_status.size());
        bool any_off = false;
        for (int o : active_to_orig)
            if (o >= 0 && o < static_cast<int>(off_orig.size()) && !off_orig[static_cast<size_t>(o)].empty()) { any_off = true; break; }
        if (!any_off || n_hvdc == 0) return;
        h_all.resize(active_to_orig.size() * static_cast<size_t>(n_hvdc));
        for (size_t a = 0; a < active_to_orig.size(); ++a) {
            std::copy(base_status.begin(), base_status.end(), h_all.begin() + static_cast<ptrdiff_t>(a) * n_hvdc);
            const int o = active_to_orig[a];
            if (o < 0 || o >= static_cast<int>(off_orig.size())) continue;
            for (int e : off_orig[static_cast<size_t>(o)])
                h_all[a * static_cast<size_t>(n_hvdc) + static_cast<size_t>(e)] = HVDC_STATUS_TRIPPED;
        }
    }

    void upload(int batch_capacity, cudaStream_t cs)
    {
        if (any()) {
            upload_h2d(d_all, h_all.data(), h_all.size(), cs);
            d_batch.resize(static_cast<size_t>(batch_capacity) * n_hvdc);
        } else {
            d_all.clear();
            d_batch.clear();
        }
    }

    // Load path: the loaded rows' regimes into their slots, a phantom slot the base case's
    void load(const SlotLoadView& L, const int* d_base, cudaStream_t cs)
    {
        if (!any()) return;
        launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_batch.data()),
                                    static_cast<const int*>(thrust::raw_pointer_cast(d_all.data())), d_base,
                                    L, n_hvdc, cs);
    }

    void move_slots(const SlotMoveView& M, cudaStream_t cs)
    {
        if (!any() || d_batch.empty()) return;
        launch_move_slot_rows(thrust::raw_pointer_cast(d_batch.data()), n_hvdc, M, cs);
    }

    void fill(NrIterBuffers& buf) const
    {
        if (!any()) return;
        buf.d_hvdc_status      = thrust::raw_pointer_cast(d_batch.data());
        buf.hvdc_status_stride = n_hvdc;
    }
};

#endif  // SLOT_HVDC_STATUS_CUH
