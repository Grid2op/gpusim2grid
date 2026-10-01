// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CONTINGENCY_BATCH_CUH
#define CONTINGENCY_BATCH_CUH

// =============================================================================
// contingency/batch_sources/contingency_batch.cuh
//
// ContingencyBatch — BatchSource policy for N-k contingency analysis.
//
// Per-chunk behaviour
// -------------------
//   prepare_Ybus_batch :
//     ① tile base d_V into d_V_batch                  (one tile_kernel launch)
//     ② tile base d_Ybus_values into d_Ybus_values_batch
//     ③ apply_contingencies_kernel: subtract Ybus deltas for this chunk's
//        contingencies (chunk-relative ctg_id, no atomicAdd by construction)
//
//   prepare_Sbus_batch : no-op (Sbus is the shared base — sbus_stride = 0).
//
// Per-batch element data (host preprocessing in ctor, GPU upload in initialize):
//   d_flat_ctg_id, d_flat_k, d_flat_delta_re, d_flat_delta_im — SoA flat patches
//   chunk_ranges_ — per-chunk slice into the flat arrays
// =============================================================================

#include <chrono>
#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../cu_complex_utils.h"
#include "../../timing_utils.hpp"
#include "../../acpf_nr_kernels.cuh"      // apply_contingencies_kernel
#include "../../acpf_nr_state.cuh"
#include "../../contingency_analysis_helper.hpp"
#include "../../nr_iter_step.cuh"         // BS
#include "../tripped_branch_table.hpp"    // TrippedBranchTable
#include "../mask_streams.cuh"            // MaskStreams
#include "slot_slack_redistribution.cuh"  // SlotSlackRedistribution

// Forward declaration to avoid circular include.
struct BatchPfDriverContext;

// Helper: ms_since for chrono start points (mirrors usage elsewhere).
inline double cb_ms_since(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

struct ContingencyBatch {

    // -------------------------------------------------------------------------
    // sbus_stride = 0  → fill_FP/FQ kernels treat d_Sbus as a single-system
    // shared (n_bus,) buffer, the historic contingency behaviour. Only when the
    // redistribute_slack pre-pass corrected some row's injection does every
    // slot get its own Sbus row (d_Sbus_batch, stride n_bus) -- a batch that
    // needs none keeps the shared buffer, bit-identical.
    // -------------------------------------------------------------------------
    int sbus_stride(int n_bus) const { return slack_.has_dp() ? n_bus : 0; }

    // -------------------------------------------------------------------------
    // Host-side preprocessing results (filled in the ctor; uploaded in initialize)
    // -------------------------------------------------------------------------
    std::vector<int>            h_flat_ctg_id_;
    std::vector<int>            h_flat_k_;
    std::vector<cuda_real_type> h_flat_delta_re_;
    std::vector<cuda_real_type> h_flat_delta_im_;
    std::vector<ChunkPatchRange> chunk_ranges_;

    // -------------------------------------------------------------------------
    // Compaction: disconnected contingencies are excluded from the batch.
    //   n_total_       — number of contingencies in the original list
    //   active_to_orig_— [n_active] active-slot index → original index
    //   d_active_to_orig — device copy, used by the driver to scatter results
    // -------------------------------------------------------------------------
    int                         n_total_ = 0;
    std::vector<int>            active_to_orig_;
    thrust::device_vector<int>  d_active_to_orig;

    // Effective per-chunk size, rebalanced over the ACTIVE (simulated) count.
    // Read back by the entry point and handed to the driver so its chunking
    // and buffer sizing match this source's chunk_ranges_.
    int                         used_batch_size_ = 0;

    // -------------------------------------------------------------------------
    // Device-side flat patch arrays (uploaded in initialize)
    // -------------------------------------------------------------------------
    thrust::device_vector<int>            d_flat_ctg_id;
    thrust::device_vector<int>            d_flat_k;
    thrust::device_vector<cuda_real_type> d_flat_delta_re;
    thrust::device_vector<cuda_real_type> d_flat_delta_im;

    // -------------------------------------------------------------------------
    // handle_disconnected_grid masking data (empty / no-op when the mode is off).
    // Identity-row entries and masked-voltage entries, flat over chunks with a
    // ChunkPatchRange per chunk (same chunking as h_flat_*).
    // -------------------------------------------------------------------------
    bool        mask_mode_ = false;
    MaskStreams mask_;

    // -------------------------------------------------------------------------
    // compute_limit_violations: per-active-slot (global, not per-chunk)
    // tripped-branch lookup table — see build_tripped_branch_table. Built
    // unconditionally (cheap: O(n_active + total_trips) ints) so it costs
    // nothing when nobody enables compute_limit_violations.
    // -------------------------------------------------------------------------
    std::vector<int> h_trip_branch_flat_, h_trip_start_, h_trip_count_;
    thrust::device_vector<int> d_trip_branch_flat, d_trip_start, d_trip_count;

    // -------------------------------------------------------------------------
    // Per-row distributed slack (redistribute_slack, see
    // slot_slack_redistribution.cuh): the saturated units' per-row weights and
    // the per-row Sbus correction; empty = base weights / shared base Sbus.
    // Set on the host (set_slack_redistribution_host) BEFORE the source is
    // handed to its driver, uploaded by initialize().
    // -------------------------------------------------------------------------
    SlotSlackRedistribution                slack_;
    thrust::device_vector<cudaComplexType> d_Sbus_batch;   // batch_size × n_bus, only with a correction

    // Preprocess timing captured at construction (CPU work only).
    double t_preprocess_ms = 0.0;

    // -------------------------------------------------------------------------
    // Constructor (host-only): resolve_indices, check_connectivity, and
    // build_flat_patches.  No GPU work here — initialize(ctx, cs) uploads.
    //
    //   contingencies : modified in-place (triplets sorted/merged; .disconnected set)
    //   Ybus_rm_outer / Ybus_rm_inner : host RowMajor CSR (n_bus+1, nnz_Y)
    //   Ybus_rm       : full RowMajor SparseMatrix for check_connectivity
    //   max_batch_size: upper bound on systems per chunk (the user batch_size).
    //                   The effective per-chunk size is rebalanced over the
    //                   ACTIVE (connected) count and exposed via used_batch_size().
    // -------------------------------------------------------------------------
    ContingencyBatch(std::vector<Contingency>& contingencies,
                     const int*                Ybus_rm_outer,
                     const int*                Ybus_rm_inner,
                     const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                     int                       max_batch_size,
                     const MaskConfig*         mask_cfg = nullptr,
                     bool                      mask_mode = true)
    {
        auto t_start = std::chrono::steady_clock::now();
        n_total_ = static_cast<int>(contingencies.size());
        resolve_indices(contingencies, Ybus_rm_outer, Ybus_rm_inner);

        // handle_disconnected_grid: largest-component masking (mask the split-off
        // buses, only skip when the reference / a controller is stranded). Legacy
        // path: skip any contingency that disconnects the graph.
        // mask_cfg without mask_mode: no largest-component masking, only the
        // per-row pins (lightsim2grid's held controllers) need its positions
        mask_mode_ = (mask_cfg != nullptr) && mask_mode;
        if (mask_mode_)
            compute_component_masks(contingencies, Ybus_rm, *mask_cfg);
        else
            check_connectivity(contingencies, Ybus_rm);

        // Rebalance the chunk size over the ACTIVE (simulated) contingencies —
        // the ones actually solved — rather than the full input count.  This
        // keeps the per-chunk buffers no larger than needed and the chunks
        // evenly filled when many contingencies are dropped (disconnected /
        // reference-stranding).
        int n_active = 0;
        for (const auto& ctg : contingencies)
            if (!ctg.disconnected) ++n_active;
        const int n_chunks = (n_active + max_batch_size - 1) / max_batch_size;
        used_batch_size_ = n_chunks > 0 ? (n_active + n_chunks - 1) / n_chunks : 1;

        build_flat_patches(contingencies, used_batch_size_,
                           h_flat_ctg_id_, h_flat_k_,
                           h_flat_delta_re_, h_flat_delta_im_,
                           chunk_ranges_, active_to_orig_);

        bool any_pins = false;
        for (const auto& ctg : contingencies)
            if (!ctg.pinned_buses.empty() || !ctg.vc_pinned_ctrl.empty()) { any_pins = true; break; }
        if (mask_mode_ || (mask_cfg != nullptr && any_pins))
            build_mask_entries(contingencies, active_to_orig_, used_batch_size_,
                               *mask_cfg, mask_.h);

        build_tripped_branch_table(contingencies, active_to_orig_,
                                   h_trip_branch_flat_, h_trip_start_, h_trip_count_);

        t_preprocess_ms = cb_ms_since(t_start);
    }

    // Effective per-chunk size (rebalanced over the active/simulated count).
    int used_batch_size() const { return used_batch_size_; }
    const std::vector<int>& active_to_orig() const { return active_to_orig_; }

    // redistribute_slack: per-row weights ([n_ctg * n_slack], ORIGINAL order;
    // empty = base weights) and Sbus corrections (per ORIGINAL row, sorted
    // (bus, dP pu); empty = none). Host only -- initialize() uploads.
    void set_slack_redistribution_host(const std::vector<cuda_real_type>& w_orig, int n_slack,
                                       const std::vector<std::vector<std::pair<int, double>>>& dp_orig)
    {
        slack_.set_weights_host(w_orig, n_slack, active_to_orig_, n_total_);
        slack_.set_dp_host(dp_orig, active_to_orig_, used_batch_size_);
    }

    ContingencyBatch(ContingencyBatch&&) noexcept = default;
    ContingencyBatch(const ContingencyBatch&) = delete;
    ContingencyBatch& operator=(const ContingencyBatch&) = delete;
    ContingencyBatch& operator=(ContingencyBatch&&) = delete;

    // -------------------------------------------------------------------------
    // initialize  — upload flat-patch SoA arrays to device on the driver's stream.
    // Called once from BatchPfDriver's ctor after `cs` exists.
    // -------------------------------------------------------------------------
    void initialize(BatchPfDriverContext& ctx, cudaStream_t cs);
    void initialize_patches(cudaStream_t cs) {
        upload_h2d(d_flat_ctg_id,   h_flat_ctg_id_.data(),   h_flat_ctg_id_.size(),   cs);
        upload_h2d(d_flat_k,         h_flat_k_.data(),         h_flat_k_.size(),         cs);
        upload_h2d(d_flat_delta_re,  h_flat_delta_re_.data(),  h_flat_delta_re_.size(),  cs);
        upload_h2d(d_flat_delta_im,  h_flat_delta_im_.data(),  h_flat_delta_im_.size(),  cs);
        // Upload the active→original map only when compaction actually drops
        // some contingency; otherwise d_result_map() returns nullptr (identity).
        if (!active_to_orig_.empty()
                && static_cast<int>(active_to_orig_.size()) < n_total_)
            upload_h2d(d_active_to_orig, active_to_orig_.data(),
                       active_to_orig_.size(), cs);

        // handle_disconnected_grid masking entries (only when any bus is masked).
        mask_.upload(cs);

        // compute_limit_violations tripped-branch table (see the ctor's
        // build_tripped_branch_table call). h_trip_start_/h_trip_count_ are
        // always sized n_active (possibly all-zero counts); h_trip_branch_flat_
        // may be empty when no contingency in this batch trips any branch.
        if (!h_trip_start_.empty()) {
            upload_h2d(d_trip_start, h_trip_start_.data(), h_trip_start_.size(), cs);
            upload_h2d(d_trip_count, h_trip_count_.data(), h_trip_count_.size(), cs);
            if (!h_trip_branch_flat_.empty())
                upload_h2d(d_trip_branch_flat, h_trip_branch_flat_.data(),
                           h_trip_branch_flat_.size(), cs);
        }
    }

    // -------------------------------------------------------------------------
    // fill_mask_buffers — write this chunk's handle_disconnected_grid mask slice
    // into the NrIterBuffers. Sets null / 0 when the mode is off or the chunk
    // masks nothing, leaving the masking launches as no-ops. d_J_outer is the
    // shared base-case J skeleton row-pointer array (used by the mask kernel).
    // -------------------------------------------------------------------------
    void fill_mask_buffers(NrIterBuffers& buf, int chunk_idx, const int* d_J_outer) const
    {
        if (!mask_.any()) return;
        mask_.fill(buf, chunk_idx, d_J_outer);
    }

    // BatchSource concept: the shared base-case slack weights on every slot,
    // unless the redistribute_slack pre-pass took saturated units out of some
    // row's distributed slack.
    void fill_slack_w_buffers(NrIterBuffers& buf, int /*chunk_idx*/) const { slack_.fill(buf); }
    void fill_vc_vset_buffers(NrIterBuffers& /*buf*/, int /*chunk_idx*/) const {}

    // -------------------------------------------------------------------------
    // Active-set interface (consumed by BatchPfDriver to compact the batch).
    //   n_active()    — number of connected contingencies actually solved.
    //   d_result_map()— device active-slot → original-index map, or nullptr
    //                   when no contingency was dropped (identity mapping, so
    //                   the driver can use the contiguous fast path).
    // -------------------------------------------------------------------------
    int n_active() const { return static_cast<int>(active_to_orig_.size()); }

    const int* d_result_map() const {
        return (static_cast<int>(active_to_orig_.size()) < n_total_
                && !d_active_to_orig.empty())
               ? thrust::raw_pointer_cast(d_active_to_orig.data())
               : nullptr;
    }

    // -------------------------------------------------------------------------
    // tripped_branch_table — device pointers into this batch's tripped-branch
    // lookup table, indexed by GLOBAL active-slot id (see
    // build_tripped_branch_table). Consumed by check_limit_violations_kernel
    // to skip branches tripped by the contingency it is currently checking.
    // -------------------------------------------------------------------------
    TrippedBranchTable tripped_branch_table() const {
        return TrippedBranchTable{
            thrust::raw_pointer_cast(d_trip_start.data()),
            thrust::raw_pointer_cast(d_trip_count.data()),
            thrust::raw_pointer_cast(d_trip_branch_flat.data())};
    }

    // -------------------------------------------------------------------------
    // prepare_Ybus_batch  — called once per chunk, BEFORE the NR loop.
    //
    //   ① tile d_V_base → d_V_batch    (one tile_kernel launch, on cs)
    //   ② tile d_Ybus_values → d_Ybus_values_batch
    //   ③ apply_contingencies_kernel for this chunk's patch slice
    //
    // Phantom slots (when actual_batch < batch_size) start from base data and
    // receive no patches, so they run a trivial NR that converges in one step.
    // -------------------------------------------------------------------------
    void prepare_Ybus_batch(BatchPfDriverContext& ctx,
                            int                  chunk_idx,
                            int                  /*actual_batch*/,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&  t)
    {
        // ①  Tile V (errors surface through the next CUDA call)
        timer.start();
        launch_tile(ctx.d_V_batch,
                    thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                    ctx.n_bus, ctx.batch_size, cs);
        t.t_tile_V += timer.stop_ms();

        // ②  Tile Ybus values
        timer.start();
        launch_tile(ctx.d_Ybus_values_batch,
                    thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data()),
                    ctx.nnz_Y, ctx.batch_size, cs);
        t.t_tile_Ybus += timer.stop_ms();

        // ③  Apply contingency patches
        timer.start();
        {
            const ChunkPatchRange& pr = chunk_ranges_[chunk_idx];
            if (pr.count > 0) {
                apply_contingencies_kernel<<<(pr.count + BS - 1) / BS, BS, 0, cs>>>(
                    ctx.d_Ybus_values_batch,
                    thrust::raw_pointer_cast(d_flat_ctg_id.data())  + pr.start,
                    thrust::raw_pointer_cast(d_flat_k.data())        + pr.start,
                    thrust::raw_pointer_cast(d_flat_delta_re.data()) + pr.start,
                    thrust::raw_pointer_cast(d_flat_delta_im.data()) + pr.start,
                    ctx.nnz_Y, pr.count);
            }
        }
        t.t_patch_Ybus += timer.stop_ms();
    }

    // -------------------------------------------------------------------------
    // prepare_Sbus_batch  — a no-op for a plain contingency analysis (Sbus is
    // the shared base case: d_Sbus_ptr returns base.d_Sbus, sbus_stride = 0).
    // With a redistribute_slack correction: base Sbus tiled into every slot,
    // this chunk's corrections added, and the per-row weights sliced.
    // -------------------------------------------------------------------------
    void prepare_Sbus_batch(BatchPfDriverContext& ctx,
                            int                  chunk_idx,
                            int                  actual_batch,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&        t);

    // -------------------------------------------------------------------------
    // d_Sbus_ptr  — the base-case Sbus pointer (shared across all batch
    // elements), or this chunk's per-slot copy when some row was corrected.
    // Used by the driver to populate NrIterBuffers.d_Sbus.
    // -------------------------------------------------------------------------
    const cudaComplexType* d_Sbus_ptr(const BatchPfDriverContext& ctx) const;

    // CPU preprocess time captured at construction.
    double cpu_preprocess_ms() const { return t_preprocess_ms; }
};

#endif // CONTINGENCY_BATCH_CUH