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

namespace {
inline void cb_chk_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("[contingency_batch] CUDA error in ") + what
                                 + ": " + cudaGetErrorString(e));
}
}

// =============================================================================
// load_slots — fill the listed slots with their rows (see the header's doc).
// =============================================================================
void ContingencyBatch::load_slots(BatchPfDriverContext& ctx, const SlotLoadView& L,
                                  cudaStream_t cs, CudaTimer& timer, BatchTimings& t)
{
    // ①  base V
    timer.start();
    launch_gather_rows_to_slots(ctx.d_V_batch, static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                                L, ctx.n_bus, cs);
    t.t_tile_V += timer.stop_ms();

    // ②  base Ybus values
    timer.start();
    launch_gather_rows_to_slots(ctx.d_Ybus_values_batch, static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data()),
                                L, ctx.nnz_Y, cs);
    t.t_tile_Ybus += timer.stop_ms();

    // ③  the loaded rows' patches
    timer.start();
    patch_gather_.plan(patch_seg_, L.h_slot, L.h_row, L.n, cs);
    if (patch_gather_.total > 0) {
        patch_gather_.gather(b_flat_k, d_flat_k, cs);
        patch_gather_.gather(b_flat_delta_re, d_flat_delta_re, cs);
        patch_gather_.gather(b_flat_delta_im, d_flat_delta_im, cs);
        apply_contingencies_kernel<<<(patch_gather_.total + BS - 1) / BS, BS, 0, cs>>>(
            ctx.d_Ybus_values_batch,
            patch_gather_.out_slot(),
            thrust::raw_pointer_cast(b_flat_k.data()),
            thrust::raw_pointer_cast(b_flat_delta_re.data()),
            thrust::raw_pointer_cast(b_flat_delta_im.data()),
            ctx.nnz_Y, patch_gather_.total);
    }
    t.t_patch_Ybus += timer.stop_ms();
    cb_chk_cuda(cudaGetLastError(), "V / Ybus / patches");

    // ④  Sbus: the shared base case, unless a redistribute_slack correction
    //     exists (base Sbus per slot, then the row's correction); the row's
    //     slack weights when some row was re-weighted.
    if (!slack_.has_dp() && !slack_.has_weights()) return;
    timer.start();
    if (slack_.has_dp()) {
        cudaComplexType* dst = thrust::raw_pointer_cast(d_Sbus_batch.data());
        launch_gather_rows_to_slots(dst, static_cast<const cudaComplexType*>(nullptr),
                                    thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                                    L, ctx.n_bus, cs);
        cb_chk_cuda(cudaGetLastError(), "Sbus rows");
        slack_.apply_dp(dst, L, ctx.n_bus, cs);
    }
    slack_.load_weights(L, thrust::raw_pointer_cast(ctx.base.d_slack_w.data()), cs);
    t.t_tile_Sbus += timer.stop_ms();
}

const cudaComplexType* ContingencyBatch::d_Sbus_ptr(const BatchPfDriverContext& ctx) const
{
    return slack_.has_dp() ? thrust::raw_pointer_cast(d_Sbus_batch.data())
                           : thrust::raw_pointer_cast(ctx.base.d_Sbus.data());
}
