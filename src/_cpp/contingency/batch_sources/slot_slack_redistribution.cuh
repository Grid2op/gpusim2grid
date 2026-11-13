// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SLOT_SLACK_REDISTRIBUTION_CUH
#define SLOT_SLACK_REDISTRIBUTION_CUH

// =============================================================================
// contingency/batch_sources/slot_slack_redistribution.cuh
//
// SlotSlackRedistribution — the per-row distributed-slack data of a batch
// source, shared by ContingencyBatch and ScenarioSweepBatch (like MaskStreams):
//
//   weights : per-row slack weights on the base participant layout
//             ([n_active x n_slack], ACTIVE-row order), copied into the loaded
//             slots of d_w_batch (a phantom slot takes base's shared weights)
//             and handed to NrIterBuffers (slack_w_stride = n_slack). Empty =
//             every slot keeps base's weights (stride 0, bit-identical).
//             Produced by a ScenarioSweep generator contingency and/or the
//             redistribute_slack pre-pass (saturated units out).
//   dP      : the redistribute_slack pre-pass' Sbus correction, (bus, dP pu)
//             entries, one segment per ACTIVE row -- gathered for the loaded
//             slots (segment_gather.cuh) and scatter-ADDED into their Sbus
//             after it was filled (never into a buffer that outlives the run,
//             so a re-run never adds it twice).
//
// Host setters take ORIGINAL-row-order data plus the source's active_to_orig
// map; upload() pushes both on the driver's stream (re-callable on a live
// source, the ScenarioSweep hot / warm paths).
// =============================================================================

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <thrust/device_vector.h>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../cu_complex_utils.h"
#include "../../contingency_analysis_helper.hpp"   // ChunkPatchRange
#include "../../nr_iter_step.cuh"                  // NrIterBuffers, BS
#include "../segment_gather.cuh"                   // RowSegments, SegmentGather
#include "../slot_schedule.cuh"                    // SlotLoadView

// d_Sbus[slot * n_bus + bus].x += dp, one thread per entry (entries are unique
// per (slot, bus): no two threads touch the same element).
template <typename C, typename R>
__global__ void add_sbus_delta_kernel(C* __restrict__ d_Sbus,
                                      const int* __restrict__ d_slot,
                                      const int* __restrict__ d_bus,
                                      const R*   __restrict__ d_val,
                                      int n_bus, int n_entries)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_entries) return;
    d_Sbus[static_cast<ptrdiff_t>(d_slot[tid]) * n_bus + d_bus[tid]].x += d_val[tid];
}

inline void ssr_chk_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("[slot_slack_redistribution] CUDA error in ") + what
                                 + ": " + cudaGetErrorString(e));
}

struct SlotSlackRedistribution {
    // ---- per-row weights ----------------------------------------------------
    std::vector<cuda_real_type>           h_w_all;   // [n_active * n_slack], active order
    int                                   n_slack = 0;
    thrust::device_vector<cuda_real_type> d_w_all;   // n_active × n_slack
    thrust::device_vector<cuda_real_type> d_w_batch; // batch_size × n_slack

    // ---- Sbus correction, one segment per active row --------------------------
    std::vector<int>             h_dp_bus;          // solver bus
    std::vector<cuda_real_type>  h_dp_val;          // dP, pu
    RowSegments                  dp_seg;
    thrust::device_vector<int>            d_dp_bus;
    thrust::device_vector<cuda_real_type> d_dp_val;
    SegmentGather                         dp_gather;   // the loaded slots' entries
    thrust::device_vector<int>            b_dp_bus;
    thrust::device_vector<cuda_real_type> b_dp_val;

    bool has_weights() const { return !h_w_all.empty() && n_slack > 0; }
    bool has_dp()      const { return !h_dp_bus.empty(); }

    // Per-row weights, [n_rows * n_slack] in ORIGINAL row order (empty = none).
    void set_weights_host(const std::vector<cuda_real_type>& orig, int n_slack_,
                          const std::vector<int>& active_to_orig, int n_rows)
    {
        h_w_all.clear();
        n_slack = n_slack_;
        if (orig.empty() || n_slack <= 0) return;
        if (static_cast<long long>(orig.size()) != static_cast<long long>(n_rows) * n_slack)
            throw std::runtime_error(
                "[slot_slack_redistribution] per-row slack weights must be (n_rows x n_slack)");
        h_w_all.resize(active_to_orig.size() * static_cast<size_t>(n_slack));
        for (size_t slot = 0; slot < active_to_orig.size(); ++slot) {
            const int o = active_to_orig[slot];
            std::copy(orig.begin() + static_cast<ptrdiff_t>(o) * n_slack,
                      orig.begin() + static_cast<ptrdiff_t>(o + 1) * n_slack,
                      h_w_all.begin() + static_cast<ptrdiff_t>(slot) * n_slack);
        }
    }

    // Per-row Sbus corrections, ORIGINAL row order (rows_orig[r] = sorted
    // (bus, dP pu) list of row r; an empty outer vector = none), laid out per
    // ACTIVE row.
    void set_dp_host(const std::vector<std::vector<std::pair<int, double>>>& rows_orig,
                     const std::vector<int>& active_to_orig)
    {
        h_dp_bus.clear(); h_dp_val.clear(); dp_seg.clear();
        if (rows_orig.empty()) return;
        bool any = false;
        for (const auto& r : rows_orig) if (!r.empty()) { any = true; break; }
        if (!any) return;
        std::vector<ChunkPatchRange> ranges(active_to_orig.size(), ChunkPatchRange{0, 0});
        for (size_t a = 0; a < active_to_orig.size(); ++a) {
            const int start = static_cast<int>(h_dp_bus.size());
            const int o = active_to_orig[a];
            if (o >= 0 && o < static_cast<int>(rows_orig.size()))
                for (const auto& bd : rows_orig[static_cast<size_t>(o)]) {
                    h_dp_bus.push_back(bd.first);
                    h_dp_val.push_back(static_cast<cuda_real_type>(bd.second));
                }
            ranges[a] = {start, static_cast<int>(h_dp_bus.size()) - start};
        }
        dp_seg.from_ranges(ranges);
    }

    void clear()
    {
        h_w_all.clear(); n_slack = 0;
        h_dp_bus.clear(); h_dp_val.clear(); dp_seg.clear();
    }

    // H→D of both, and the per-slot weight buffer sized for `batch_capacity`.
    void upload(int batch_capacity, cudaStream_t cs)
    {
        if (has_weights()) {
            upload_h2d(d_w_all, h_w_all.data(), h_w_all.size(), cs);
            d_w_batch.resize(static_cast<size_t>(batch_capacity) * n_slack);
        } else {
            d_w_all.clear();
            d_w_batch.clear();
        }
        if (has_dp()) {
            upload_h2d(d_dp_bus, h_dp_bus.data(), h_dp_bus.size(), cs);
            upload_h2d(d_dp_val, h_dp_val.data(), h_dp_val.size(), cs);
        }
    }

    // Load path: the loaded rows' weights into their slots, a phantom slot
    // taking base's shared weights.
    void load_weights(const SlotLoadView& L, const cuda_real_type* d_base_w, cudaStream_t cs)
    {
        if (!has_weights()) return;
        launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_w_batch.data()),
                                    thrust::raw_pointer_cast(d_w_all.data()), d_base_w,
                                    L, n_slack, cs);
        ssr_chk_cuda(cudaGetLastError(), "slack weight rows");
    }

    // Load path: add the loaded rows' corrections to their slots' Sbus (which
    // the caller has just filled).
    void apply_dp(cudaComplexType* d_Sbus_batch, const SlotLoadView& L, int n_bus, cudaStream_t cs)
    {
        if (!has_dp()) return;
        dp_gather.plan(dp_seg, L.h_slot, L.h_row, L.n, cs);
        if (dp_gather.total <= 0) return;
        dp_gather.gather(b_dp_bus, d_dp_bus, cs);
        dp_gather.gather(b_dp_val, d_dp_val, cs);
        add_sbus_delta_kernel<<<(dp_gather.total + BS - 1) / BS, BS, 0, cs>>>(
            d_Sbus_batch, dp_gather.out_slot(),
            thrust::raw_pointer_cast(b_dp_bus.data()),
            thrust::raw_pointer_cast(b_dp_val.data()),
            n_bus, dp_gather.total);
        ssr_chk_cuda(cudaGetLastError(), "Sbus correction scatter");
    }

    void fill(NrIterBuffers& buf) const
    {
        if (!has_weights()) return;
        buf.d_slack_w      = thrust::raw_pointer_cast(d_w_batch.data());
        buf.slack_w_stride = n_slack;
    }
};

#endif  // SLOT_SLACK_REDISTRIBUTION_CUH
