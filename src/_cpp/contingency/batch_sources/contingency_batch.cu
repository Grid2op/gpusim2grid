// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/contingency_batch.cu — the ContingencyBatch
// members that need the complete BatchPfDriverContext
// =============================================================================

// the driver first: the header's inline members need BatchPfDriverContext complete
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)
#include "contingency_batch.cuh"

#include <stdexcept>
#include <string>

// =============================================================================
// initialize — upload the flat-patch arrays / masks / tripped table (see
// initialize_patches), and the redistribute_slack data when there is any.
// =============================================================================
void ContingencyBatch::initialize(BatchPfDriverContext& ctx, cudaStream_t cs)
{
    initialize_patches(cs);
    if (slack_.has_weights() && slack_.n_slack != ctx.base.n_slack)
        throw std::runtime_error(
            "[contingency_batch] per-row slack weight count does not match the base "
            "case's participant count");
    slack_.upload(ctx.batch_size, cs);
    if (slack_.has_dp())
        d_Sbus_batch.resize(static_cast<size_t>(ctx.batch_size) * ctx.n_bus);
}

// =============================================================================
// prepare_Sbus_batch — base Sbus into every slot + this chunk's corrections
// (only with a correction), and this chunk's per-row weights (only with some).
// =============================================================================
void ContingencyBatch::prepare_Sbus_batch(BatchPfDriverContext& ctx,
                                          int                  chunk_idx,
                                          int                  actual_batch,
                                          cudaStream_t         cs,
                                          CudaTimer&           timer,
                                          BatchTimings&        t)
{
    if (!slack_.has_dp() && !slack_.has_weights()) return;
    timer.start();
    if (slack_.has_dp()) {
        cudaComplexType* dst = thrust::raw_pointer_cast(d_Sbus_batch.data());
        launch_tile(dst, thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                    ctx.n_bus, ctx.batch_size, cs);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("[contingency_batch] CUDA error in tile Sbus: ")
                                     + cudaGetErrorString(e));
        slack_.apply_dp(dst, chunk_idx, ctx.n_bus, cs);
    }
    slack_.prepare_weights(chunk_idx, actual_batch, ctx.batch_size,
                           thrust::raw_pointer_cast(ctx.base.d_slack_w.data()), cs);
    t.t_tile_Sbus += timer.stop_ms();
}

const cudaComplexType* ContingencyBatch::d_Sbus_ptr(const BatchPfDriverContext& ctx) const
{
    return slack_.has_dp() ? thrust::raw_pointer_cast(d_Sbus_batch.data())
                           : thrust::raw_pointer_cast(ctx.base.d_Sbus.data());
}
