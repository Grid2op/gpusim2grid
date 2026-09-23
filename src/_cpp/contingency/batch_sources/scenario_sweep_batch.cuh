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
//     compaction can drop rows, d_Sbus_all holds the rows in ACTIVE-slot
//     order (d_Sbus_all[slot] = original row active_to_orig_[slot]) —
//     prepare_Sbus_batch's row-slice copy is then verbatim InjectionBatch
//     code. The rows are NOT permuted on the host any more: the session owns
//     a canonical ORIGINAL-row-order device buffer (filled either from numpy
//     or straight from a torch tensor) and set_sbus_from_orig() gathers it
//     into active-slot order with one kernel (gather_rows_kernel), on every
//     run() path — cold (new driver), warm (new source on a live driver) or
//     hot (new injections only). See ScenarioSweepSession::run().
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
//
// Driver persistence
// ------------------
// A source can be built for an ALREADY LIVE BatchPfDriver (warm path): pass
// forced_batch_size = the driver's capacity so the chunk ranges built here
// match the driver's chunk loop (the driver refuses a mismatch), then
// BatchPfDriver::replace_source() move-assigns it in — hence the defaulted
// move assignment below.
// =============================================================================

#include <algorithm>
#include <chrono>
#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../cu_complex_utils.h"
#include "../../timing_utils.hpp"
#include "../../acpf_nr_kernels.cuh"      // apply_contingencies_kernel, apply_gen_v_kernel, gather kernels
#include "../../acpf_nr_state.cuh"
#include "../../contingency_analysis_helper.hpp"
#include "../../nr_iter_step.cuh"         // BS
#include "../gen_v_override.hpp"          // GenVOverride
#include "gen_vset_slots.cuh"             // GenVsetSlots
#include "../tripped_branch_table.hpp"    // TrippedBranchTable
#include "../mask_streams.cuh"            // MaskStreams

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
    int                         n_bus_   = 0;
    std::vector<int>            active_to_orig_;
    // ALWAYS uploaded (even without compaction, where it is the identity):
    // the Sbus / gen_v / adjoint row gathers index through it. d_result_map()
    // still returns nullptr when identity so the driver keeps its contiguous
    // result-store fast path.
    thrust::device_vector<int>  d_active_to_orig;

    // Effective per-chunk size, rebalanced over the ACTIVE (simulated) count
    // — read back by the session and handed to the driver — or forced to the
    // live driver's capacity (warm path).
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
    // Also carries the per-row PV pins (Contingency::pinned_buses) of the
    // generator-contingency feature, which are built whether or not
    // handle_disconnected_grid is on (see the ctor).
    bool        mask_mode_ = false;
    MaskStreams mask_;

    // -------------------------------------------------------------------------
    // Per-row distributed-slack weights (generator contingencies, see
    // ScenarioSweepSession::set_contingency_gens). h_slack_w_all_ is
    // [n_active * n_slack], ALREADY PERMUTED into active-slot order; empty when
    // no row re-weights the slack (every slot then keeps base's shared
    // weights, slack_w_stride 0 -- bit-identical).
    // -------------------------------------------------------------------------
    std::vector<cuda_real_type>            h_slack_w_all_;
    int                                    n_slack_ = 0;
    thrust::device_vector<cuda_real_type>  d_slack_w_all;     // n_active × n_slack
    thrust::device_vector<cuda_real_type>  d_slack_w_batch;   // batch_size × n_slack

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
    // Device-resident per-scenario Sbus in ACTIVE-slot order (n_active × n_bus),
    // filled by set_sbus_from_orig() from the session's original-order buffer.
    // n_scenarios_ is the ORIGINAL (pre-compaction) row count — bookkeeping /
    // error messages only.
    // -------------------------------------------------------------------------
    int                          n_scenarios_ = 0;

    thrust::device_vector<cudaComplexType> d_Sbus_all;     // n_active × n_bus
    thrust::device_vector<cudaComplexType> d_Sbus_batch;   // batch_size × n_bus

    // -------------------------------------------------------------------------
    // set_gen_v() override (see ScenarioSweepSession::set_gen_v's doc), in
    // active-slot order. Two ways in: the host path (ctor argument / set_gen_v,
    // permuted here) and the device path (set_gen_v_from_orig, one gather
    // kernel from the session's original-order device buffer). Only
    // gen_v_override_.h_active_bus is meaningful on the device path (its
    // k_active() gates prepare_Ybus_batch's reseed); h_gen_v_all stays empty.
    // -------------------------------------------------------------------------
    GenVOverride gen_v_override_;
    thrust::device_vector<int>            d_gv_active_bus;
    thrust::device_vector<int>            d_gv_active_col;   // device path only
    thrust::device_vector<cuda_real_type> d_gv_all;
    GenVsetSlots                          gv_vset_;          // VoltageControl set-point columns

    // Preprocess timing captured at construction (CPU work only).
    double t_preprocess_ms = 0.0;

    // -------------------------------------------------------------------------
    // Constructor (host-only): resolve_indices, check_connectivity /
    // compute_component_masks, and build_flat_patches — identical sequence to
    // ContingencyBatch — then permutes the optional gen_v / slack-weight rows
    // into active-slot order. Sbus is NOT taken here: call set_sbus_from_orig()
    // after initialize() (the session does).
    //
    //   contingencies   : modified in-place (triplets sorted/merged; .disconnected
    //                     set) — one entry per scenario.
    //   max_batch_size  : upper bound on systems per chunk (the user batch_size).
    //   mask_cfg        : the session's mask configuration (ALWAYS given: its
    //                     row_info also drives the per-row PV pins, which do
    //                     not depend on handle_disconnected_grid).
    //   mask_mode       : handle_disconnected_grid mode when true (see class
    //                     doc); false selects the legacy check_connectivity
    //                     skip-if-split path.
    //   gen_v_override_orig : optional set_gen_v() data, ORIGINAL (pre-
    //                     compaction) row order — permuted into active-slot
    //                     order below.
    //   h_slack_w_orig  : optional per-row slack weights, [n_scenarios *
    //                     n_slack] in ORIGINAL row order (empty: every row
    //                     keeps the base weights) -- permuted below too.
    //   forced_batch_size : 0 → rebalance used_batch_size_ over the active
    //                     count (cold path); > 0 → use exactly this chunk
    //                     size (a live driver's capacity, warm path; also the
    //                     session's fixed_batch_capacity mode).
    // -------------------------------------------------------------------------
    ScenarioSweepBatch(std::vector<Contingency>& contingencies,
                       const int*                Ybus_rm_outer,
                       const int*                Ybus_rm_inner,
                       const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                       int                       max_batch_size,
                       const MaskConfig&         mask_cfg,
                       bool                      mask_mode,
                       GenVOverride&&            gen_v_override_orig = GenVOverride{},
                       std::vector<cuda_real_type>&& h_slack_w_orig = std::vector<cuda_real_type>{},
                       int                       n_slack = 0,
                       int                       forced_batch_size = 0)
    {
        auto t_start = std::chrono::steady_clock::now();
        n_total_ = static_cast<int>(contingencies.size());
        n_scenarios_ = n_total_;
        n_bus_ = static_cast<int>(Ybus_rm.rows());
        resolve_indices(contingencies, Ybus_rm_outer, Ybus_rm_inner);

        mask_mode_ = mask_mode;
        if (mask_mode_)
            compute_component_masks(contingencies, Ybus_rm, mask_cfg);
        else
            check_connectivity(contingencies, Ybus_rm);

        int n_active = 0;
        for (const auto& ctg : contingencies)
            if (!ctg.disconnected) ++n_active;
        if (forced_batch_size > 0) {
            used_batch_size_ = forced_batch_size;
        } else {
            const int n_chunks = (n_active + max_batch_size - 1) / max_batch_size;
            used_batch_size_ = n_chunks > 0 ? (n_active + n_chunks - 1) / n_chunks : 1;
        }

        build_flat_patches(contingencies, used_batch_size_,
                           h_flat_ctg_id_, h_flat_k_,
                           h_flat_delta_re_, h_flat_delta_im_,
                           chunk_ranges_, active_to_orig_);

        // Mask / pin / stranded streams: needed in mask mode AND whenever some
        // row pins a switchable bus (generator contingencies).
        bool any_pins = false;
        for (const auto& ctg : contingencies)
            if (!ctg.pinned_buses.empty()) { any_pins = true; break; }
        if (mask_mode_ || any_pins)
            build_mask_entries(contingencies, active_to_orig_, used_batch_size_,
                               mask_cfg, mask_.h);

        build_tripped_branch_table(contingencies, active_to_orig_,
                                   h_trip_branch_flat_, h_trip_start_, h_trip_count_);

        // Permute set_gen_v() rows into active-slot order (host path) — see
        // GenVOverride's own doc. The active-bus column list itself is
        // row-independent, so it carries over unchanged.
        if (gen_v_override_orig.k_active() > 0)
            _permute_gen_v_rows(std::move(gen_v_override_orig));

        // Permute the per-row slack weights into active-slot order too.
        n_slack_ = n_slack;
        if (!h_slack_w_orig.empty() && n_slack > 0) {
            if (static_cast<int>(h_slack_w_orig.size()) != n_total_ * n_slack)
                throw std::runtime_error(
                    "[scenario_sweep_batch] per-row slack weights must be "
                    "(n_scenarios x n_slack)");
            h_slack_w_all_.resize(active_to_orig_.size() * static_cast<size_t>(n_slack));
            for (size_t slot = 0; slot < active_to_orig_.size(); ++slot) {
                const int orig = active_to_orig_[slot];
                std::copy(
                    h_slack_w_orig.begin() + static_cast<ptrdiff_t>(orig) * n_slack,
                    h_slack_w_orig.begin() + static_cast<ptrdiff_t>(orig + 1) * n_slack,
                    h_slack_w_all_.begin() + static_cast<ptrdiff_t>(slot) * n_slack);
            }
        }

        t_preprocess_ms = ssb_ms_since(t_start);
    }

    int used_batch_size() const { return used_batch_size_; }
    const std::vector<int>& active_to_orig() const { return active_to_orig_; }
    const int* d_active_to_orig_ptr() const {
        return d_active_to_orig.empty() ? nullptr
                                        : thrust::raw_pointer_cast(d_active_to_orig.data());
    }

    ScenarioSweepBatch(ScenarioSweepBatch&&) noexcept = default;
    ScenarioSweepBatch& operator=(ScenarioSweepBatch&&) noexcept = default;
    ScenarioSweepBatch(const ScenarioSweepBatch&) = delete;
    ScenarioSweepBatch& operator=(const ScenarioSweepBatch&) = delete;

    // -------------------------------------------------------------------------
    // initialize — upload flat Ybus-patch arrays, masks, the tripped table and
    // the active map; size the Sbus buffers (values come from
    // set_sbus_from_orig()). Unlike InjectionBatch, Ybus is NOT tiled once
    // here — it varies per scenario, so it is re-tiled + patched every chunk
    // in prepare_Ybus_batch (like ContingencyBatch).
    // -------------------------------------------------------------------------
    void initialize(BatchPfDriverContext& ctx, cudaStream_t cs);

    // -------------------------------------------------------------------------
    // set_sbus_from_orig — gather the session's ORIGINAL-row-order per-unit
    // Sbus (n_scenarios × n_bus, device) into d_Sbus_all (active-slot order).
    // Requires initialize() (d_active_to_orig uploaded, d_Sbus_all sized).
    // -------------------------------------------------------------------------
    void set_sbus_from_orig(const cudaComplexType* d_Sbus_orig, cudaStream_t cs);

    // -------------------------------------------------------------------------
    // set_gen_v — host path hot update: permute the original-order override
    // into active-slot order and upload (replaces whatever was configured).
    // set_gen_v_from_orig — device path: gather the selected generator columns
    //   of the session's original-order (n_scenarios × n_gen) device matrix
    //   into active-slot order with one kernel. active_cols/active_bus/
    //   active_vc_group are the driven generator columns, the bus each
    //   regulates and the VoltageControl group it drives (gen_v_active_columns).
    // clear_gen_v — drop any override (rows keep the base-case voltage).
    // -------------------------------------------------------------------------
    void set_gen_v(GenVOverride&& gen_v_override_orig, cudaStream_t cs);
    void set_gen_v_from_orig(const cuda_real_type* d_gen_v_orig, int n_gen,
                             const std::vector<int>& active_cols,
                             const std::vector<int>& active_bus,
                             const std::vector<int>& active_vc_group,
                             cudaStream_t cs);
    void clear_gen_v() { gen_v_override_ = GenVOverride{}; gv_vset_.clear(); }

    // -------------------------------------------------------------------------
    // fill_mask_buffers — write this chunk's handle_disconnected_grid mask
    // slice into the NrIterBuffers. Sets null / 0 when the mode is off or the
    // chunk masks nothing, leaving the masking launches as no-ops. Verbatim
    // ContingencyBatch::fill_mask_buffers.
    // -------------------------------------------------------------------------
    void fill_mask_buffers(NrIterBuffers& buf, int chunk_idx, const int* d_J_outer) const
    {
        if (!mask_.any()) return;
        mask_.fill(buf, chunk_idx, d_J_outer);
    }

    // -------------------------------------------------------------------------
    // fill_slack_w_buffers — point buf.d_slack_w at this chunk's per-slot
    // weights (sliced by prepare_Sbus_batch) when some row re-weighted the
    // slack; otherwise leave base's shared array (stride 0, bit-identical).
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // fill_vc_vset_buffers — point buf.d_vc_vset at this chunk's per-slot
    // VoltageControl set-points (built by prepare_Ybus_batch) when a gen_v
    // column drives a group; otherwise base's shared array (stride 0).
    // -------------------------------------------------------------------------
    void fill_vc_vset_buffers(NrIterBuffers& buf, int /*chunk_idx*/) const
    {
        gv_vset_.fill(buf);
    }

    void fill_slack_w_buffers(NrIterBuffers& buf, int /*chunk_idx*/) const
    {
        if (h_slack_w_all_.empty() || n_slack_ <= 0) return;
        buf.d_slack_w      = thrust::raw_pointer_cast(d_slack_w_batch.data());
        buf.slack_w_stride = n_slack_;
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

private:
    // Host permutation of an original-order override into active-slot order
    // (h_gen_v_all rows follow active_to_orig_; the bus list is row-independent).
    void _permute_gen_v_rows(GenVOverride&& orig)
    {
        const ptrdiff_t k = orig.k_active();
        gen_v_override_.h_active_bus = std::move(orig.h_active_bus);
        gen_v_override_.h_active_vc_group = std::move(orig.h_active_vc_group);
        gen_v_override_.h_gen_v_all.resize(active_to_orig_.size() * static_cast<size_t>(k));
        for (size_t slot = 0; slot < active_to_orig_.size(); ++slot) {
            const int o = active_to_orig_[slot];
            std::copy(
                orig.h_gen_v_all.begin() + static_cast<ptrdiff_t>(o) * k,
                orig.h_gen_v_all.begin() + static_cast<ptrdiff_t>(o + 1) * k,
                gen_v_override_.h_gen_v_all.begin() + static_cast<ptrdiff_t>(slot) * k);
        }
    }
};

#endif // SCENARIO_SWEEP_BATCH_CUH
