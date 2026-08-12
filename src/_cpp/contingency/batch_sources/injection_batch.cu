// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/injection_batch.cu
// =============================================================================

#include "injection_batch.cuh"
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)
#include "../../acpf_nr_kernels.cuh"   // apply_gen_v_kernel
#include "../../nr_iter_step.cuh"      // BS, nr_grid_size

#include <stdexcept>
#include <string>

namespace {
inline void _chk_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("[injection_batch] CUDA error in ") + what
            + ": " + cudaGetErrorString(e));
    }
}
}

void InjectionBatch::initialize(BatchPfDriverContext& ctx, cudaStream_t cs)
{
    // Sanity: host array must match the configured (n_scenarios × n_bus).
    if (static_cast<int>(h_Sbus_all_.size())
            != n_scenarios_ * ctx.n_bus) {
        throw std::runtime_error(
            "[injection_batch] h_Sbus_all_ size does not match n_scenarios * n_bus");
    }

    // Upload the full (n_scenario × n_bus) Sbus.
    upload_h2d(d_Sbus_all,
               h_Sbus_all_.data(),
               h_Sbus_all_.size(),
               cs);

    // Allocate per-chunk Sbus buffer.
    d_Sbus_batch.resize(static_cast<size_t>(ctx.batch_size) * ctx.n_bus);

    // Upload the (n_scenario × n_bus) gen_v reseed targets, if set_gen_v()
    // was called. No per-chunk buffer: prepare_Ybus_batch reads a row-slice
    // of d_gen_v_all directly (this source never reorders scenarios).
    if (has_gen_v_) {
        if (static_cast<int>(h_gen_v_all_.size()) != n_scenarios_ * ctx.n_bus) {
            throw std::runtime_error(
                "[injection_batch] h_gen_v_all_ size does not match n_scenarios * n_bus");
        }
        upload_h2d(d_gen_v_all, h_gen_v_all_.data(), h_gen_v_all_.size(), cs);
    }

    // Tile base Ybus values into every batch slot — ONCE.
    {
        cudaComplexType* const       dst = ctx.d_Ybus_values_batch;
        const cudaComplexType* const src =
            thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data());
        const size_t nbytes = static_cast<size_t>(ctx.nnz_Y) * sizeof(cudaComplexType);
        for (int b = 0; b < ctx.batch_size; ++b) {
            _chk_cuda(cudaMemcpyAsync(
                dst + static_cast<ptrdiff_t>(b) * ctx.nnz_Y,
                src, nbytes, cudaMemcpyDeviceToDevice, cs),
                "tile base Ybus");
        }
    }
}

void InjectionBatch::prepare_Ybus_batch(BatchPfDriverContext& ctx,
                                        int                  chunk_idx,
                                        int                  actual_batch,
                                        cudaStream_t         cs,
                                        CudaTimer&           timer,
                                        BatchTimings&  t)
{
    // Tile V from base (every chunk starts from the converged base voltage).
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
    // Re-seed |V| from this chunk's gen_v row-slice, mirroring lightsim2grid's
    // own `V = Vinit_solver; _apply_step_gen_v(...);` order. No-op when
    // set_gen_v() was never called.
    if (has_gen_v_ && actual_batch > 0) {
        const cuda_real_type* const src =
            thrust::raw_pointer_cast(d_gen_v_all.data())
            + static_cast<ptrdiff_t>(chunk_idx) * ctx.batch_size * ctx.n_bus;
        apply_gen_v_kernel<<<nr_grid_size((long long)actual_batch * ctx.n_bus, BS), BS, 0, cs>>>(
            ctx.d_V_batch, src, ctx.n_bus, actual_batch);
    }
    t.t_tile_V += timer.stop_ms();
    // No t_tile_Ybus / t_patch_Ybus updates — Ybus is permanent.
}

void InjectionBatch::prepare_Sbus_batch(BatchPfDriverContext& ctx,
                                        int                  chunk_idx,
                                        int                  actual_batch,
                                        cudaStream_t         cs,
                                        CudaTimer&           timer,
                                        BatchTimings&        t)
{
    const int n_bus     = ctx.n_bus;
    const int c_start   = chunk_idx * ctx.batch_size;

    timer.start();

    // ②  Copy the (actual_batch × n_bus) row-slice into d_Sbus_batch.
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

    // ③  Phantom-pad remaining slots with base.d_Sbus.  Phantom NR uses base
    //     V + base Ybus + base Sbus → converged in one step, results discarded.
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