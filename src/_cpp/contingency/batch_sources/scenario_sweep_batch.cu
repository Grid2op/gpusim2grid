// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/scenario_sweep_batch.cu
// =============================================================================

#include "scenario_sweep_batch.cuh"
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)
#include "../../acpf_nr_kernels.cuh"   // apply_gen_v_kernel

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

    if (has_gen_v_) {
        if (static_cast<int>(h_gen_v_all_.size()) != n_active() * ctx.n_bus) {
            throw std::runtime_error(
                "[scenario_sweep_batch] h_gen_v_all_ size does not match n_active * n_bus");
        }
        upload_h2d(d_gen_v_all, h_gen_v_all_.data(), h_gen_v_all_.size(), cs);
    }
}

// =============================================================================
// prepare_Ybus_batch — tile V + gen_v reseed (InjectionBatch-style, new),
// then verbatim ContingencyBatch::prepare_Ybus_batch (tile Ybus + patches).
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
    {
        cudaComplexType* const       dst = ctx.d_V_batch;
        const cudaComplexType* const src =
            thrust::raw_pointer_cast(ctx.base.d_V_base.data());
        const size_t nbytes = static_cast<size_t>(ctx.n_bus) * sizeof(cudaComplexType);
        for (int b = 0; b < ctx.batch_size; ++b) {
            _chk_cuda(cudaMemcpyAsync(
                dst + static_cast<ptrdiff_t>(b) * ctx.n_bus,
                src, nbytes, cudaMemcpyDeviceToDevice, cs),
                "tile V");
        }
    }
    // Re-seed |V| from this chunk's gen_v row-slice (active-slot order, same
    // c_start convention as prepare_Sbus_batch below). No-op when set_gen_v()
    // was never called.
    if (has_gen_v_ && actual_batch > 0) {
        const cuda_real_type* const src =
            thrust::raw_pointer_cast(d_gen_v_all.data())
            + static_cast<ptrdiff_t>(chunk_idx) * ctx.batch_size * ctx.n_bus;
        apply_gen_v_kernel<<<nr_grid_size((long long)actual_batch * ctx.n_bus, BS), BS, 0, cs>>>(
            ctx.d_V_batch, src, ctx.n_bus, actual_batch);
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
            _chk_cuda(cudaMemcpyAsync(
                dst + static_cast<ptrdiff_t>(b) * ctx.nnz_Y,
                src, nbytes, cudaMemcpyDeviceToDevice, cs),
                "tile Ybus");
        }
    }
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

    t.t_tile_Sbus += timer.stop_ms();
}
