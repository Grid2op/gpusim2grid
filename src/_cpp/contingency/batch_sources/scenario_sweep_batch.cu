// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/scenario_sweep_batch.cu
// =============================================================================

#include "scenario_sweep_batch.cuh"
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)

#include <stdexcept>
#include <string>

namespace {
inline void _chk_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("[scenario_sweep_batch] CUDA error in ") + what
            + ": " + cudaGetErrorString(e));
    }
}
}

// =============================================================================
// initialize — upload flat Ybus-patch arrays, masks, tripped table, active
// map; size the Sbus buffers. Mirrors ContingencyBatch::initialize (patch
// upload) + InjectionBatch::initialize (buffer alloc), minus InjectionBatch's
// one-time Ybus tiling (Ybus varies per chunk here) and minus the Sbus upload
// (set_sbus_from_orig gathers it from the session's device buffer).
// =============================================================================
void ScenarioSweepBatch::initialize(BatchPfDriverContext& ctx, cudaStream_t cs)
{
    upload_h2d(d_flat_ctg_id,   h_flat_ctg_id_.data(),   h_flat_ctg_id_.size(),   cs);
    upload_h2d(d_flat_k,         h_flat_k_.data(),         h_flat_k_.size(),         cs);
    upload_h2d(d_flat_delta_re,  h_flat_delta_re_.data(),  h_flat_delta_re_.size(),  cs);
    upload_h2d(d_flat_delta_im,  h_flat_delta_im_.data(),  h_flat_delta_im_.size(),  cs);
    // Always uploaded (identity included): the row gathers index through it.
    if (!active_to_orig_.empty())
        upload_h2d(d_active_to_orig, active_to_orig_.data(),
                   active_to_orig_.size(), cs);
    else
        d_active_to_orig.clear();

    // handle_disconnected_grid masking / PV-pin / stranded-controller entries
    // (only when any exist).
    mask_.upload(cs);

    // compute_limit_violations tripped-branch table (see the ctor's
    // build_tripped_branch_table call). h_trip_start_/h_trip_count_ are
    // always sized n_active (possibly all-zero counts); h_trip_branch_flat_
    // may be empty when no scenario in this batch trips any branch.
    if (!h_trip_start_.empty()) {
        upload_h2d(d_trip_start, h_trip_start_.data(), h_trip_start_.size(), cs);
        upload_h2d(d_trip_count, h_trip_count_.data(), h_trip_count_.size(), cs);
        if (!h_trip_branch_flat_.empty())
            upload_h2d(d_trip_branch_flat, h_trip_branch_flat_.data(),
                       h_trip_branch_flat_.size(), cs);
    }

    if (n_bus_ != ctx.n_bus)
        throw std::runtime_error(
            "[scenario_sweep_batch] bus count does not match the driver's");
    d_Sbus_all.resize(static_cast<size_t>(n_active()) * ctx.n_bus);
    d_Sbus_batch.resize(static_cast<size_t>(ctx.batch_size) * ctx.n_bus);

    // set_gen_v() overrides given to the ctor (host path), if any.
    if (gen_v_override_.k_active() > 0 && !gen_v_override_.h_gen_v_all.empty()) {
        upload_h2d(d_gv_active_bus, gen_v_override_.h_active_bus.data(),
                   gen_v_override_.h_active_bus.size(), cs);
        upload_h2d(d_gv_all, gen_v_override_.h_gen_v_all.data(),
                   gen_v_override_.h_gen_v_all.size(), cs);
        gv_vset_.upload(gen_v_override_.h_active_vc_group, cs);
    }

    // Per-row slack weights, if any (generator contingencies).
    if (!h_slack_w_all_.empty() && n_slack_ > 0) {
        if (n_slack_ != ctx.base.n_slack)
            throw std::runtime_error(
                "[scenario_sweep_batch] per-row slack weight count does not "
                "match the base case's participant count");
        upload_h2d(d_slack_w_all, h_slack_w_all_.data(), h_slack_w_all_.size(), cs);
        d_slack_w_batch.resize(static_cast<size_t>(ctx.batch_size) * n_slack_);
    }
}

// =============================================================================
// set_sbus_from_orig — one gather kernel, original row order → active slots.
// =============================================================================
void ScenarioSweepBatch::set_sbus_from_orig(const cudaComplexType* d_Sbus_orig,
                                            cudaStream_t cs)
{
    const int n_act = n_active();
    if (n_act <= 0) return;
    if (d_Sbus_all.size() != static_cast<size_t>(n_act) * n_bus_)
        throw std::runtime_error(
            "[scenario_sweep_batch] set_sbus_from_orig: call initialize() first");
    launch_gather_rows(thrust::raw_pointer_cast(d_Sbus_all.data()), d_Sbus_orig,
                       d_active_to_orig_ptr(), n_bus_, n_act,
                       /*zero_nonfinite=*/false, cs);
    _chk_cuda(cudaGetLastError(), "Sbus row gather");
}

// =============================================================================
// set_gen_v (host path) / set_gen_v_from_orig (device path)
// =============================================================================
void ScenarioSweepBatch::set_gen_v(GenVOverride&& gen_v_override_orig, cudaStream_t cs)
{
    gen_v_override_ = GenVOverride{};
    gv_vset_.clear();
    if (gen_v_override_orig.k_active() <= 0) return;
    _permute_gen_v_rows(std::move(gen_v_override_orig));
    upload_h2d(d_gv_active_bus, gen_v_override_.h_active_bus.data(),
               gen_v_override_.h_active_bus.size(), cs);
    upload_h2d(d_gv_all, gen_v_override_.h_gen_v_all.data(),
               gen_v_override_.h_gen_v_all.size(), cs);
    gv_vset_.upload(gen_v_override_.h_active_vc_group, cs);
}

void ScenarioSweepBatch::set_gen_v_from_orig(const cuda_real_type* d_gen_v_orig, int n_gen,
                                             const std::vector<int>& active_cols,
                                             const std::vector<int>& active_bus,
                                             const std::vector<int>& active_vc_group,
                                             cudaStream_t cs)
{
    gen_v_override_ = GenVOverride{};
    gv_vset_.clear();
    const int k = static_cast<int>(active_cols.size());
    if (k <= 0 || active_bus.size() != active_cols.size()
        || active_vc_group.size() != active_cols.size()) return;
    gen_v_override_.h_active_bus = active_bus;   // k_active() > 0 gates the reseed
    gen_v_override_.h_active_vc_group = active_vc_group;
    upload_h2d(d_gv_active_bus, active_bus.data(), active_bus.size(), cs);
    gv_vset_.upload(active_vc_group, cs);
    upload_h2d(d_gv_active_col, active_cols.data(), active_cols.size(), cs);
    const int n_act = n_active();
    d_gv_all.resize(static_cast<size_t>(n_act) * k);
    launch_gather_cols_rows(thrust::raw_pointer_cast(d_gv_all.data()), d_gen_v_orig,
                            d_active_to_orig_ptr(),
                            thrust::raw_pointer_cast(d_gv_active_col.data()),
                            n_gen, k, n_act, cs);
    _chk_cuda(cudaGetLastError(), "gen_v gather");
}

// =============================================================================
// prepare_Ybus_batch — verbatim ContingencyBatch::prepare_Ybus_batch.
// =============================================================================
void ScenarioSweepBatch::prepare_Ybus_batch(BatchPfDriverContext& ctx,
                                            int                  chunk_idx,
                                            int                  actual_batch,
                                            cudaStream_t         cs,
                                            CudaTimer&           timer,
                                            BatchTimings&  t)
{
    // ①  Tile V
    timer.start();
    launch_tile(ctx.d_V_batch,
                thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                ctx.n_bus, ctx.batch_size, cs);
    _chk_cuda(cudaGetLastError(), "tile V");
    t.t_tile_V += timer.stop_ms();

    // ②  Tile Ybus values
    timer.start();
    launch_tile(ctx.d_Ybus_values_batch,
                thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data()),
                ctx.nnz_Y, ctx.batch_size, cs);
    _chk_cuda(cudaGetLastError(), "tile Ybus");
    t.t_tile_Ybus += timer.stop_ms();

    // ③  Apply this chunk's contingency (branch-trip) patches
    timer.start();
    {
        const ChunkPatchRange& pr = chunk_ranges_[static_cast<size_t>(chunk_idx)];
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

    // ④  Re-seed generator target voltages (set_gen_v()), if configured.
    //     Rows here are already in active-slot order (see the ctor), so the
    //     row offset is a plain chunk*batch_size slice, same as ③ above.
    if (gen_v_override_.k_active() > 0) {
        const int k = gen_v_override_.k_active();
        const int row_offset = chunk_idx * ctx.batch_size;
        apply_gen_v_kernel<<<nr_grid_size((long long)actual_batch * k, BS), BS, 0, cs>>>(
            ctx.d_V_batch,
            thrust::raw_pointer_cast(d_gv_all.data()),
            thrust::raw_pointer_cast(d_gv_active_bus.data()),
            row_offset, k, actual_batch, ctx.n_bus);
        // ⑤  ... and the VoltageControl set-points those columns drive.
        gv_vset_.prepare(thrust::raw_pointer_cast(ctx.base.d_vc_vset.data()),
                         ctx.base.n_vc_grp,
                         thrust::raw_pointer_cast(d_gv_all.data()), k,
                         row_offset, actual_batch, ctx.batch_size, cs);
    }
}

// =============================================================================
// prepare_Sbus_batch — verbatim InjectionBatch::prepare_Sbus_batch, operating
// on d_Sbus_all in whatever row order it holds (active-slot order here).
// =============================================================================
void ScenarioSweepBatch::prepare_Sbus_batch(BatchPfDriverContext& ctx,
                                            int                  chunk_idx,
                                            int                  actual_batch,
                                            cudaStream_t         cs,
                                            CudaTimer&           timer,
                                            BatchTimings&        t)
{
    const int n_bus     = ctx.n_bus;
    const int c_start   = chunk_idx * ctx.batch_size;

    timer.start();

    if (actual_batch > 0) {
        const cudaComplexType* const src =
            thrust::raw_pointer_cast(d_Sbus_all.data())
            + static_cast<ptrdiff_t>(c_start) * n_bus;
        cudaComplexType* const dst =
            thrust::raw_pointer_cast(d_Sbus_batch.data());
        const size_t nbytes =
            static_cast<size_t>(actual_batch) * n_bus * sizeof(cudaComplexType);
        _chk_cuda(cudaMemcpyAsync(dst, src, nbytes,
                                   cudaMemcpyDeviceToDevice, cs),
                  "Sbus row-slice copy");
    }

    if (actual_batch < ctx.batch_size) {
        const cudaComplexType* const src_base =
            thrust::raw_pointer_cast(ctx.base.d_Sbus.data());
        const size_t row_bytes =
            static_cast<size_t>(n_bus) * sizeof(cudaComplexType);
        for (int b = actual_batch; b < ctx.batch_size; ++b) {
            cudaComplexType* const dst =
                thrust::raw_pointer_cast(d_Sbus_batch.data())
                + static_cast<ptrdiff_t>(b) * n_bus;
            _chk_cuda(cudaMemcpyAsync(dst, src_base, row_bytes,
                                       cudaMemcpyDeviceToDevice, cs),
                      "Sbus phantom pad");
        }
    }

    // Per-row slack weights: same row-slice + phantom-pad as Sbus above, the
    // phantom slots taking base's shared weights.
    if (!h_slack_w_all_.empty() && n_slack_ > 0) {
        const int nsl = n_slack_;
        if (actual_batch > 0) {
            _chk_cuda(cudaMemcpyAsync(
                thrust::raw_pointer_cast(d_slack_w_batch.data()),
                thrust::raw_pointer_cast(d_slack_w_all.data())
                    + static_cast<ptrdiff_t>(c_start) * nsl,
                static_cast<size_t>(actual_batch) * nsl * sizeof(cuda_real_type),
                cudaMemcpyDeviceToDevice, cs),
                "slack weight row-slice copy");
        }
        for (int b = actual_batch; b < ctx.batch_size; ++b) {
            _chk_cuda(cudaMemcpyAsync(
                thrust::raw_pointer_cast(d_slack_w_batch.data())
                    + static_cast<ptrdiff_t>(b) * nsl,
                thrust::raw_pointer_cast(ctx.base.d_slack_w.data()),
                static_cast<size_t>(nsl) * sizeof(cuda_real_type),
                cudaMemcpyDeviceToDevice, cs),
                "slack weight phantom pad");
        }
    }

    t.t_tile_Sbus += timer.stop_ms();
}
