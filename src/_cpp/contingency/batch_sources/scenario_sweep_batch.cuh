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
//     topology disconnects the grid (check_connectivity → active_to_orig_).
//     handle_disconnected_grid masking is NOT supported here (deferred scope,
//     see CLAUDE.md) — a disconnecting scenario is skipped/NaN exactly like
//     ContingencyBatch's legacy (mask_cfg == nullptr) path; there is no
//     MaskConfig parameter at all.
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
// Deferred scope (matches lightsim2grid's own ScenarioSweepCPP v1): no
// fill_mask_buffers / tripped_branch_table content — compute_limit_violations
// and handle_disconnected_grid masking are not wired up for this source.
// =============================================================================

#include <algorithm>
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
    // Host-side per-scenario Sbus, ALREADY PERMUTED into active-slot order at
    // construction (see class doc). n_scenarios_ is the ORIGINAL (pre-
    // compaction) row count — kept for bookkeeping / error messages only; the
    // device-resident d_Sbus_all has n_active() rows.
    // -------------------------------------------------------------------------
    std::vector<cudaComplexType> h_Sbus_all_;
    int                          n_scenarios_ = 0;

    thrust::device_vector<cudaComplexType> d_Sbus_all;     // n_active × n_bus
    thrust::device_vector<cudaComplexType> d_Sbus_batch;   // batch_size × n_bus

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
    // -------------------------------------------------------------------------
    ScenarioSweepBatch(std::vector<Contingency>& contingencies,
                       const int*                Ybus_rm_outer,
                       const int*                Ybus_rm_inner,
                       const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                       std::vector<cudaComplexType>&& h_Sbus_all_orig,
                       int                       max_batch_size)
    {
        auto t_start = std::chrono::steady_clock::now();
        n_total_ = static_cast<int>(contingencies.size());
        n_scenarios_ = n_total_;
        resolve_indices(contingencies, Ybus_rm_outer, Ybus_rm_inner);
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
    // fill_mask_buffers — no-op (handle_disconnected_grid deferred, see class
    // doc); leaves NrIterBuffers' mask fields at their off defaults.
    // -------------------------------------------------------------------------
    void fill_mask_buffers(NrIterBuffers& /*buf*/, int /*chunk_idx*/,
                           const int* /*d_J_outer*/) const {}

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

    // compute_limit_violations is out of scope for this source (see class
    // doc) — the all-null table is never consulted since callers never
    // enable the fused violation check on a ScenarioSweepSession.
    TrippedBranchTable tripped_branch_table() const { return TrippedBranchTable{nullptr, nullptr, nullptr}; }

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
