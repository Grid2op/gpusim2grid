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
// initialize — upload flat Ybus-patch arrays + active-permuted Sbus rows;
// allocate the per-chunk Sbus buffer. Mirrors ContingencyBatch::initialize
// (patch upload) + InjectionBatch::initialize (Sbus upload / buffer alloc),
// minus InjectionBatch's one-time Ybus tiling (Ybus varies per chunk here).
// =============================================================================
void ScenarioSweepBatch::initialize(BatchPfDriverContext& ctx, cudaStream_t cs)
{
    upload_h2d(d_flat_ctg_id,   h_flat_ctg_id_.data(),   h_flat_ctg_id_.size(),   cs);
    upload_h2d(d_flat_k,         h_flat_k_.data(),         h_flat_k_.size(),         cs);
    upload_h2d(d_flat_delta_re,  h_flat_delta_re_.data(),  h_flat_delta_re_.size(),  cs);
    upload_h2d(d_flat_delta_im,  h_flat_delta_im_.data(),  h_flat_delta_im_.size(),  cs);
    if (!active_to_orig_.empty()
            && static_cast<int>(active_to_orig_.size()) < n_total_)
        upload_h2d(d_active_to_orig, active_to_orig_.data(),
                   active_to_orig_.size(), cs);

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

    if (static_cast<int>(h_Sbus_all_.size())
            != n_active() * ctx.n_bus) {
        throw std::runtime_error(
            "[scenario_sweep_batch] h_Sbus_all_ size does not match n_active * n_bus");
    }
    upload_h2d(d_Sbus_all, h_Sbus_all_.data(), h_Sbus_all_.size(), cs);
    d_Sbus_batch.resize(static_cast<size_t>(ctx.batch_size) * ctx.n_bus);

    // set_gen_v() overrides, if any -- see GenVOverride's own doc.
    if (gen_v_override_.k_active() > 0) {
        upload_h2d(d_gv_active_bus, gen_v_override_.h_active_bus.data(),
                   gen_v_override_.h_active_bus.size(), cs);
        upload_h2d(d_gv_all, gen_v_override_.h_gen_v_all.data(),
                   gen_v_override_.h_gen_v_all.size(), cs);
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
