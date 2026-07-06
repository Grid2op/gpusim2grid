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
//     ① tile base d_V into d_V_batch                  (batch_size D→D copies)
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
    // shared (n_bus,) buffer.  This matches the historic contingency behaviour.
    // -------------------------------------------------------------------------
    static int sbus_stride(int /*n_bus*/) { return 0; }

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
    bool                         mask_mode_ = false;
    std::vector<int>             h_mask_slot_, h_mask_row_, h_mask_diag_;
    std::vector<ChunkPatchRange> mask_row_ranges_;
    std::vector<int>             h_maskv_slot_, h_maskv_bus_;
    std::vector<ChunkPatchRange> maskv_ranges_;
    thrust::device_vector<int>   d_mask_slot, d_mask_row, d_mask_diag;
    thrust::device_vector<int>   d_maskv_slot, d_maskv_bus;

    // -------------------------------------------------------------------------
    // compute_limit_violations: per-active-slot (global, not per-chunk)
    // tripped-branch lookup table — see build_tripped_branch_table. Built
    // unconditionally (cheap: O(n_active + total_trips) ints) so it costs
    // nothing when nobody enables compute_limit_violations.
    // -------------------------------------------------------------------------
    std::vector<int> h_trip_branch_flat_, h_trip_start_, h_trip_count_;
    thrust::device_vector<int> d_trip_branch_flat, d_trip_start, d_trip_count;

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
                     const MaskConfig*         mask_cfg = nullptr)
    {
        auto t_start = std::chrono::steady_clock::now();
        n_total_ = static_cast<int>(contingencies.size());
        resolve_indices(contingencies, Ybus_rm_outer, Ybus_rm_inner);

        // handle_disconnected_grid: largest-component masking (mask the split-off
        // buses, only skip when the reference / a controller is stranded). Legacy
        // path: skip any contingency that disconnects the graph.
        mask_mode_ = (mask_cfg != nullptr);
        if (mask_mode_)
            compute_component_masks(contingencies, Ybus_rm,
                                    mask_cfg->is_reference_bus,
                                    mask_cfg->is_controller_bus);
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

        if (mask_mode_)
            build_mask_entries(contingencies, active_to_orig_, used_batch_size_,
                               mask_cfg->row_info,
                               h_mask_slot_, h_mask_row_, h_mask_diag_, mask_row_ranges_,
                               h_maskv_slot_, h_maskv_bus_, maskv_ranges_);

        build_tripped_branch_table(contingencies, active_to_orig_,
                                   h_trip_branch_flat_, h_trip_start_, h_trip_count_);

        t_preprocess_ms = cb_ms_since(t_start);
    }

    // Effective per-chunk size (rebalanced over the active/simulated count).
    int used_batch_size() const { return used_batch_size_; }

    ContingencyBatch(ContingencyBatch&&) noexcept = default;
    ContingencyBatch(const ContingencyBatch&) = delete;
    ContingencyBatch& operator=(const ContingencyBatch&) = delete;
    ContingencyBatch& operator=(ContingencyBatch&&) = delete;

    // -------------------------------------------------------------------------
    // initialize  — upload flat-patch SoA arrays to device on the driver's stream.
    // Called once from BatchPfDriver's ctor after `cs` exists.
    // -------------------------------------------------------------------------
    void initialize(BatchPfDriverContext& /*ctx*/, cudaStream_t cs) {
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
        if (!h_mask_slot_.empty()) {
            upload_h2d(d_mask_slot, h_mask_slot_.data(), h_mask_slot_.size(), cs);
            upload_h2d(d_mask_row,  h_mask_row_.data(),  h_mask_row_.size(),  cs);
            upload_h2d(d_mask_diag, h_mask_diag_.data(), h_mask_diag_.size(), cs);
        }
        if (!h_maskv_slot_.empty()) {
            upload_h2d(d_maskv_slot, h_maskv_slot_.data(), h_maskv_slot_.size(), cs);
            upload_h2d(d_maskv_bus,  h_maskv_bus_.data(),  h_maskv_bus_.size(),  cs);
        }

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
        if (!mask_mode_) return;
        buf.d_J_outer_mask = d_J_outer;

        if (chunk_idx < static_cast<int>(mask_row_ranges_.size())) {
            const ChunkPatchRange& r = mask_row_ranges_[static_cast<size_t>(chunk_idx)];
            if (r.count > 0) {
                buf.d_mask_slot = thrust::raw_pointer_cast(d_mask_slot.data()) + r.start;
                buf.d_mask_row  = thrust::raw_pointer_cast(d_mask_row.data())  + r.start;
                buf.d_mask_diag = thrust::raw_pointer_cast(d_mask_diag.data()) + r.start;
                buf.n_mask_rows = r.count;
            }
        }
        if (chunk_idx < static_cast<int>(maskv_ranges_.size())) {
            const ChunkPatchRange& r = maskv_ranges_[static_cast<size_t>(chunk_idx)];
            if (r.count > 0) {
                buf.d_maskv_slot = thrust::raw_pointer_cast(d_maskv_slot.data()) + r.start;
                buf.d_maskv_bus  = thrust::raw_pointer_cast(d_maskv_bus.data())  + r.start;
                buf.n_mask_v     = r.count;
            }
        }
    }

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
    //   ① tile d_V_base → d_V_batch    (batch_size D→D copies, on cs)
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
        // ①  Tile V
        timer.start();
        {
            cudaComplexType* const       dst = ctx.d_V_batch;
            const cudaComplexType* const src =
                thrust::raw_pointer_cast(ctx.base.d_V_base.data());
            const size_t nbytes = static_cast<size_t>(ctx.n_bus) * sizeof(cudaComplexType);
            for (int b = 0; b < ctx.batch_size; ++b) {
                cudaError_t e = cudaMemcpyAsync(
                    dst + static_cast<ptrdiff_t>(b) * ctx.n_bus,
                    src, nbytes, cudaMemcpyDeviceToDevice, cs);
                (void)e;   // errors surface through the next CUDA call
            }
        }
        t.t_tile_V += timer.stop_ms();

        // ②  Tile Ybus values
        timer.start();
        {
            cudaComplexType* const       dst = ctx.d_Ybus_values_batch;
            const cudaComplexType* const src =
                thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data());
            const size_t nbytes = static_cast<size_t>(ctx.nnz_Y) * sizeof(cudaComplexType);
            for (int b = 0; b < ctx.batch_size; ++b) {
                cudaError_t e = cudaMemcpyAsync(
                    dst + static_cast<ptrdiff_t>(b) * ctx.nnz_Y,
                    src, nbytes, cudaMemcpyDeviceToDevice, cs);
                (void)e;
            }
        }
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
    // prepare_Sbus_batch  — no-op for contingency analysis.  Sbus is the shared
    // base-case (d_Sbus_ptr returns base.d_Sbus; sbus_stride = 0).
    // -------------------------------------------------------------------------
    void prepare_Sbus_batch(BatchPfDriverContext& /*ctx*/,
                            int                  /*chunk_idx*/,
                            int                  /*actual_batch*/,
                            cudaStream_t         /*cs*/,
                            CudaTimer&           /*timer*/,
                            BatchTimings&  /*t*/)
    { }

    // -------------------------------------------------------------------------
    // d_Sbus_ptr  — returns the base-case Sbus pointer (shared across all
    // batch elements).  Used by the driver to populate NrIterBuffers.d_Sbus.
    // -------------------------------------------------------------------------
    const cudaComplexType* d_Sbus_ptr(const BatchPfDriverContext& ctx) const {
        return thrust::raw_pointer_cast(ctx.base.d_Sbus.data());
    }

    // CPU preprocess time captured at construction.
    double cpu_preprocess_ms() const { return t_preprocess_ms; }
};

#endif // CONTINGENCY_BATCH_CUH