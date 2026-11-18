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
#include "../v_groups.cuh"         // v_groups_by_active, launch_gather_groups_to_slots

#include <stdexcept>
#include <string>

// =============================================================================
// initialize — upload the flat-patch arrays / masks / tripped table (see
// initialize_patches), and the redistribute_slack data when there is any.
// =============================================================================
void ContingencyBatch::initialize(BatchPfDriverContext& ctx, cudaStream_t cs)
{
    initialize_patches(cs);
    if (!h_vr_bus_.empty()) {
        upload_h2d(d_vr_bus, h_vr_bus_.data(), h_vr_bus_.size(), cs);
        upload_h2d(d_vr_vm,  h_vr_vm_.data(),  h_vr_vm_.size(),  cs);
    }
    if (slack_.has_weights() && slack_.n_slack != ctx.base.n_slack)
        throw std::runtime_error(
            "[contingency_batch] per-row slack weight count does not match the base "
            "case's participant count");
    slack_.upload(ctx.batch_size, cs);
    hvdc_.upload(ctx.batch_size, cs);
    vc_hold_.upload(ctx.batch_size, cs);
    if (slack_.has_dp() || slot_sbus_)
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
// set_v_groups — per-row starting voltages by group (see the header's doc).
// =============================================================================
void ContingencyBatch::set_v_groups(const cudaComplexType* d_groups, int n_groups,
                                    const std::vector<int>& group_of_orig,
                                    const cudaComplexType* d_fallback, cudaStream_t cs)
{
    has_v_groups_ = false;
    if (d_groups == nullptr || d_fallback == nullptr || n_groups <= 0) return;
    const std::vector<int> by_active = v_groups_by_active(group_of_orig, active_to_orig_, n_total_, n_groups);
    upload_h2d(d_group_of_active, by_active.data(), by_active.size(), cs);
    d_V_groups_   = d_groups;
    d_V_fallback_ = d_fallback;
    has_v_groups_ = true;
}

// =============================================================================
// load_slots — fill the listed slots with their rows (see the header's doc).
// =============================================================================
void ContingencyBatch::load_slots(BatchPfDriverContext& ctx, const SlotLoadView& L,
                                  cudaStream_t cs, CudaTimer& timer, BatchTimings& t)
{
    // ①  base V, or the row's group's (strategy_warm_start)
    timer.start();
    if (has_v_groups_ && L.n > 0 && ctx.n_bus > 0) {
        cb_chk_cuda(launch_gather_groups_to_slots(ctx.d_V_batch, d_V_groups_,
                                                  thrust::raw_pointer_cast(d_group_of_active.data()),
                                                  d_V_fallback_, L.d_slot, L.d_row, L.n, ctx.n_bus, cs),
                    "group V gather");
    } else {
        launch_gather_rows_to_slots(ctx.d_V_batch, static_cast<const cudaComplexType*>(nullptr),
                                    thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                                    L, ctx.n_bus, cs);
    }
    // ... the |V| of the buses a row holds PV (a generator moved there), its Q row pinned
    vr_gather_.plan(vr_seg_, L.h_slot, L.h_row, L.n, cs);
    if (vr_gather_.total > 0) {
        vr_gather_.gather(b_vr_bus, d_vr_bus, cs);
        vr_gather_.gather(b_vr_vm, d_vr_vm, cs);
        apply_vm_reseed_kernel<<<(vr_gather_.total + BS - 1) / BS, BS, 0, cs>>>(
            ctx.d_V_batch, vr_gather_.out_slot(),
            thrust::raw_pointer_cast(b_vr_bus.data()),
            thrust::raw_pointer_cast(b_vr_vm.data()),
            vr_gather_.total, ctx.n_bus);
        cb_chk_cuda(cudaGetLastError(), "vm reseed");
    }
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
    hvdc_.load(L, thrust::raw_pointer_cast(ctx.base.d_hvdc_status.data()), cs);
    vc_hold_.load(L, cs);
    if (!slack_.has_dp() && !slack_.has_weights() && !slot_sbus_) return;
    timer.start();
    if (slack_.has_dp() || slot_sbus_) {
        cudaComplexType* dst = thrust::raw_pointer_cast(d_Sbus_batch.data());
        launch_gather_rows_to_slots(dst, static_cast<const cudaComplexType*>(nullptr),
                                    thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                                    L, ctx.n_bus, cs);
        cb_chk_cuda(cudaGetLastError(), "Sbus rows");
        if (slack_.has_dp()) slack_.apply_dp(dst, L, ctx.n_bus, cs);
    }
    slack_.load_weights(L, thrust::raw_pointer_cast(ctx.base.d_slack_w.data()), cs);
    t.t_tile_Sbus += timer.stop_ms();
}

const cudaComplexType* ContingencyBatch::d_Sbus_ptr(const BatchPfDriverContext& ctx) const
{
    return (slack_.has_dp() || slot_sbus_) ? thrust::raw_pointer_cast(d_Sbus_batch.data())
                                           : thrust::raw_pointer_cast(ctx.base.d_Sbus.data());
}

void ContingencyBatch::reload_sbus(BatchPfDriverContext& ctx, const SlotLoadView& L, cudaStream_t cs)
{
    if (!slot_sbus_)
        throw std::runtime_error("[contingency_batch] reload_sbus needs a per-slot Sbus (slot_sbus_)");
    launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_Sbus_batch.data()),
                                static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                                L, ctx.n_bus, cs);
    cb_chk_cuda(cudaGetLastError(), "Sbus reload");
    // the row's own correction (the loads / generators its contingency trips) is part
    // of the system its loops solve
    if (slack_.has_dp()) slack_.apply_dp(thrust::raw_pointer_cast(d_Sbus_batch.data()), L, ctx.n_bus, cs);
}
