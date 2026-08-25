// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SCENARIO_SWEEP_BATCH_CUH
#define SCENARIO_SWEEP_BATCH_CUH

// =============================================================================
// contingency/batch_sources/scenario_sweep_batch.cuh
//
// ScenarioSweepBatch — BatchSource policy for the row-aligned combined
// topology + injection sweep (ScenarioSweepGPU): row `i` pairs its own
// contingency (line/trafo trip) with its own (P, Q) injection, independently
// of every other row. Composes the two existing BatchSource policies rather
// than inventing new device mechanics:
//
//   - Ybus side: identical to ContingencyBatch — per-active-slot Ybus value
//     patches (apply_contingencies_kernel), compaction of scenarios whose
//     topology disconnects the grid. Two connectivity modes, selected by
//     whether a MaskConfig is passed to the constructor (same convention as
//     ContingencyBatch): nullptr selects the legacy check_connectivity
//     skip-if-split path (a disconnecting scenario is skipped/NaN); non-null
//     selects compute_component_masks' handle_disconnected_grid mode (solve
//     the largest connected component, freezing the rest as NaN, only
//     skipping when the angle reference or a controller bus is stranded).
//   - Sbus side: identical to InjectionBatch's dense per-scenario (n_scenario
//     × n_bus) complex Sbus, sbus_stride = n_bus. The one wrinkle: since
//     compaction can drop rows, h_Sbus_all_ is built ALREADY PERMUTED into
//     active-slot order at construction (h_Sbus_all_[slot] = original row
//     active_to_orig_[slot]) — prepare_Sbus_batch's row-slice copy is then
//     verbatim InjectionBatch code, needing no extra indirection.
//
// Per-chunk behaviour
// -------------------
//   prepare_Ybus_batch : tile V + tile Ybus + apply_contingencies_kernel for
//                         this chunk's patch slice (verbatim ContingencyBatch).
//   prepare_Sbus_batch : row-slice copy of d_Sbus_all + phantom-pad with
//                         base.d_Sbus for any unfilled tail slots (verbatim
//                         InjectionBatch).
//
// fill_mask_buffers / tripped_branch_table are real implementations, verbatim
// ContingencyBatch's — the fused masking kernels (nr_apply_bus_mask/
// nr_mask_v_nan, driver.cuh) and the compute_limit_violations kernel
// (check_limit_violations_kernel, batch_pf_driver.cu) are already generic
// over any BatchSource, so no changes are needed outside this file/session to
// support either handle_disconnected_grid or compute_limit_violations here.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../cu_complex_utils.h"
#include "../../timing_utils.hpp"
#include "../../acpf_nr_kernels.cuh"      // apply_contingencies_kernel, apply_gen_v_kernel
#include "../../acpf_nr_state.cuh"
#include "../../contingency_analysis_helper.hpp"
#include "../../nr_iter_step.cuh"         // BS
#include "../gen_v_override.hpp"          // GenVOverride
#include "../tripped_branch_table.hpp"    // TrippedBranchTable

struct BatchPfDriverContext;

inline double ssb_ms_since(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

struct ScenarioSweepBatch {

    // sbus_stride = n_bus → per-scenario row indexing in fill_FP/FQ kernels,
    // identical to InjectionBatch.
    static int sbus_stride(int n_bus) { return n_bus; }

    // -------------------------------------------------------------------------
    // Host-side Ybus-patch preprocessing (filled in the ctor; uploaded in
    // initialize()) — identical fields/semantics to ContingencyBatch.
    // -------------------------------------------------------------------------
    std::vector<int>            h_flat_ctg_id_;
    std::vector<int>            h_flat_k_;
    std::vector<cuda_real_type> h_flat_delta_re_;
    std::vector<cuda_real_type> h_flat_delta_im_;
    std::vector<ChunkPatchRange> chunk_ranges_;

    int                         n_total_ = 0;
    std::vector<int>            active_to_orig_;
    thrust::device_vector<int>  d_active_to_orig;

    // Effective per-chunk size, rebalanced over the ACTIVE (simulated) count
    // — read back by the session and handed to the driver.
    int                         used_batch_size_ = 0;

    thrust::device_vector<int>            d_flat_ctg_id;
    thrust::device_vector<int>            d_flat_k;
    thrust::device_vector<cuda_real_type> d_flat_delta_re;
    thrust::device_vector<cuda_real_type> d_flat_delta_im;

    // -------------------------------------------------------------------------
    // handle_disconnected_grid masking data (empty / no-op when the mode is
    // off). Identity-row entries and masked-voltage entries, flat over chunks
    // with a ChunkPatchRange per chunk (same chunking as h_flat_*). Verbatim
    // ContingencyBatch's fields — see that file's own doc.
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
    // nothing when nobody enables compute_limit_violations. Verbatim
    // ContingencyBatch's fields.
    // -------------------------------------------------------------------------
    std::vector<int> h_trip_branch_flat_, h_trip_start_, h_trip_count_;
    thrust::device_vector<int> d_trip_branch_flat, d_trip_start, d_trip_count;

    // -------------------------------------------------------------------------
    // Host-side per-scenario Sbus, ALREADY PERMUTED into active-slot order at
    // construction (see class doc). n_scenarios_ is the ORIGINAL (pre-
    // compaction) row count — kept for bookkeeping / error messages only; the
    // device-resident d_Sbus_all has n_active() rows.
    // -------------------------------------------------------------------------
    std::vector<cudaComplexType> h_Sbus_all_;
    int                          n_scenarios_ = 0;

    thrust::device_vector<cudaComplexType> d_Sbus_all;     // n_active × n_bus
    thrust::device_vector<cudaComplexType> d_Sbus_batch;   // batch_size × n_bus

    // -------------------------------------------------------------------------
    // set_gen_v() override (see ScenarioSweepSession::set_gen_v's doc), ALREADY
    // PERMUTED into active-slot order at construction — same rationale as
    // h_Sbus_all_ above. k_active() == 0 (the default) is a cheap no-op.
    // -------------------------------------------------------------------------
    GenVOverride gen_v_override_;
    thrust::device_vector<int>            d_gv_active_bus;
    thrust::device_vector<cuda_real_type> d_gv_all;

    // Preprocess timing captured at construction (CPU work only).
    double t_preprocess_ms = 0.0;

    // -------------------------------------------------------------------------
    // Constructor (host-only): resolve_indices, check_connectivity, and
    // build_flat_patches — identical sequence to ContingencyBatch, always the
    // legacy (skip-if-split) connectivity path (no MaskConfig / handle_
    // disconnected_grid support for this source, see class doc). Then
    // permutes h_Sbus_all_orig into active-slot order.
    //
    //   contingencies   : modified in-place (triplets sorted/merged; .disconnected
    //                     set) — one entry per scenario, row-aligned with
    //                     h_Sbus_all_orig.
    //   h_Sbus_all_orig : (n_scenarios × n_bus) per-unit complex Sbus, ORIGINAL
    //                     (pre-compaction) row order, row-aligned with
    //                     `contingencies`. Taken by rvalue reference since the
    //                     caller has no further use for it; read-only here
    //                     (copied, row-permuted, into h_Sbus_all_), not moved.
    //   max_batch_size  : upper bound on systems per chunk (the user batch_size).
    //   mask_cfg        : handle_disconnected_grid mode when non-null (see
    //                     class doc); nullptr selects the legacy
    //                     check_connectivity skip-if-split path.
    //   gen_v_override_orig : optional set_gen_v() data, ORIGINAL (pre-
    //                     compaction) row order, row-aligned with
    //                     h_Sbus_all_orig — permuted into active-slot order
    //                     below, same as h_Sbus_all_orig itself.
    // -------------------------------------------------------------------------
    ScenarioSweepBatch(std::vector<Contingency>& contingencies,
                       const int*                Ybus_rm_outer,
                       const int*                Ybus_rm_inner,
                       const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                       std::vector<cudaComplexType>&& h_Sbus_all_orig,
                       int                       max_batch_size,
                       const MaskConfig*         mask_cfg = nullptr,
                       GenVOverride&&            gen_v_override_orig = GenVOverride{})
    {
        auto t_start = std::chrono::steady_clock::now();
        n_total_ = static_cast<int>(contingencies.size());
        n_scenarios_ = n_total_;
        resolve_indices(contingencies, Ybus_rm_outer, Ybus_rm_inner);

        mask_mode_ = (mask_cfg != nullptr);
        if (mask_mode_)
            compute_component_masks(contingencies, Ybus_rm,
                                    mask_cfg->is_reference_bus,
                                    mask_cfg->is_controller_bus);
        else
            check_connectivity(contingencies, Ybus_rm);

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

        // Permute Sbus rows into active-slot order so prepare_Sbus_batch's
        // contiguous-slice copy (verbatim from InjectionBatch) stays correct
        // even though the driver's chunk loop runs over active-slot space.
        const int n_bus = static_cast<int>(Ybus_rm.rows());
        h_Sbus_all_.resize(static_cast<size_t>(active_to_orig_.size()) * n_bus);
        for (size_t slot = 0; slot < active_to_orig_.size(); ++slot) {
            const int orig = active_to_orig_[slot];
            std::copy(
                h_Sbus_all_orig.begin() + static_cast<ptrdiff_t>(orig) * n_bus,
                h_Sbus_all_orig.begin() + static_cast<ptrdiff_t>(orig + 1) * n_bus,
                h_Sbus_all_.begin() + static_cast<ptrdiff_t>(slot) * n_bus);
        }

        // Permute set_gen_v() rows into active-slot order too, mirroring
        // h_Sbus_all_ above (same active_to_orig_ mapping) — see
        // GenVOverride's own doc. The active-bus column list itself is
        // row-independent, so it carries over unchanged.
        if (gen_v_override_orig.k_active() > 0) {
            const ptrdiff_t k = gen_v_override_orig.k_active();
            gen_v_override_.h_active_bus = std::move(gen_v_override_orig.h_active_bus);
            gen_v_override_.h_gen_v_all.resize(active_to_orig_.size() * static_cast<size_t>(k));
            for (size_t slot = 0; slot < active_to_orig_.size(); ++slot) {
                const int orig = active_to_orig_[slot];
                std::copy(
                    gen_v_override_orig.h_gen_v_all.begin() + static_cast<ptrdiff_t>(orig) * k,
                    gen_v_override_orig.h_gen_v_all.begin() + static_cast<ptrdiff_t>(orig + 1) * k,
                    gen_v_override_.h_gen_v_all.begin() + static_cast<ptrdiff_t>(slot) * k);
            }
        }

        t_preprocess_ms = ssb_ms_since(t_start);
    }

    int used_batch_size() const { return used_batch_size_; }

    ScenarioSweepBatch(ScenarioSweepBatch&&) noexcept = default;
    ScenarioSweepBatch(const ScenarioSweepBatch&) = delete;
    ScenarioSweepBatch& operator=(const ScenarioSweepBatch&) = delete;
    ScenarioSweepBatch& operator=(ScenarioSweepBatch&&) = delete;

    // -------------------------------------------------------------------------
    // initialize — upload flat Ybus-patch arrays AND the (already active-
    // permuted) Sbus rows; allocate the per-chunk Sbus buffer. Unlike
    // InjectionBatch, Ybus is NOT tiled once here — it varies per scenario, so
    // it is re-tiled + patched every chunk in prepare_Ybus_batch (like
    // ContingencyBatch).
    // -------------------------------------------------------------------------
    void initialize(BatchPfDriverContext& ctx, cudaStream_t cs);

    // -------------------------------------------------------------------------
    // fill_mask_buffers — write this chunk's handle_disconnected_grid mask
    // slice into the NrIterBuffers. Sets null / 0 when the mode is off or the
    // chunk masks nothing, leaving the masking launches as no-ops. Verbatim
    // ContingencyBatch::fill_mask_buffers.
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
    // Active-set interface (consumed by BatchPfDriver to compact the batch) —
    // identical semantics to ContingencyBatch.
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
    // lookup table, indexed by GLOBAL active-slot id. Consumed by
    // check_limit_violations_kernel to skip branches tripped by the scenario
    // it is currently checking. Verbatim ContingencyBatch::tripped_branch_table.
    // -------------------------------------------------------------------------
    TrippedBranchTable tripped_branch_table() const {
        return TrippedBranchTable{
            thrust::raw_pointer_cast(d_trip_start.data()),
            thrust::raw_pointer_cast(d_trip_count.data()),
            thrust::raw_pointer_cast(d_trip_branch_flat.data())};
    }

    // -------------------------------------------------------------------------
    // prepare_Ybus_batch — tile V + tile Ybus + apply this chunk's patches.
    // Verbatim ContingencyBatch::prepare_Ybus_batch.
    // -------------------------------------------------------------------------
    void prepare_Ybus_batch(BatchPfDriverContext& ctx,
                            int                  chunk_idx,
                            int                  actual_batch,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&  t);

    // -------------------------------------------------------------------------
    // prepare_Sbus_batch — row-slice copy + phantom pad. Verbatim
    // InjectionBatch::prepare_Sbus_batch (operates on whatever row order
    // d_Sbus_all holds, which is active-slot order here).
    // -------------------------------------------------------------------------
    void prepare_Sbus_batch(BatchPfDriverContext& ctx,
                            int                  chunk_idx,
                            int                  actual_batch,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&  t);

    const cudaComplexType* d_Sbus_ptr(const BatchPfDriverContext& /*ctx*/) const {
        return thrust::raw_pointer_cast(d_Sbus_batch.data());
    }

    double cpu_preprocess_ms() const { return t_preprocess_ms; }
};

#endif // SCENARIO_SWEEP_BATCH_CUH
