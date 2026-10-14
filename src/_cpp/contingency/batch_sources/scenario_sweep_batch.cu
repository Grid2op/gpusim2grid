// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/scenario_sweep_batch.cu
// =============================================================================

#include "scenario_sweep_batch.cuh"
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)

#include <limits>
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

    // per-slot |V| reseeds of the reactive-limit outer loop (only when any exist)
    if (!h_vr_bus_.empty()) {
        upload_h2d(d_vr_bus,  h_vr_bus_.data(),  h_vr_bus_.size(),  cs);
        upload_h2d(d_vr_vm,   h_vr_vm_.data(),   h_vr_vm_.size(),   cs);
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

    // Per-row slack weights / Sbus correction: set on the live source by
    // set_slack_redistribution (the session does, on every run() path).
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
// set_v_init_from_orig — one gather kernel, original row order → active slots.
// =============================================================================
void ScenarioSweepBatch::set_v_init_from_orig(const cudaComplexType* d_V_orig,
                                              const cudaComplexType* d_V_fallback,
                                              cudaStream_t cs)
{
    const int n_act = n_active();
    has_v_init_ = false;
    if (n_act <= 0 || d_V_orig == nullptr) return;
    d_V_init_all.resize(static_cast<size_t>(n_act) * n_bus_);
    const long long total = static_cast<long long>(n_act) * n_bus_;
    gather_v_rows_kernel<<<static_cast<unsigned>((total + BS - 1) / BS), BS, 0, cs>>>(
        thrust::raw_pointer_cast(d_V_init_all.data()), d_V_orig, d_active_to_orig_ptr(),
        d_V_fallback, n_bus_, n_act);
    _chk_cuda(cudaGetLastError(), "V init row gather");
    has_v_init_ = true;
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
// load_slots — fill the listed slots with their rows (see the header's doc),
// in the order the per-chunk preparation always used: V (+ warm start), Ybus,
// patches, gen_v reseed + v_set, |V| reseeds, then Sbus, its correction and
// the slack weights.
// =============================================================================
void ScenarioSweepBatch::load_slots(BatchPfDriverContext& ctx, const SlotLoadView& L,
                                    cudaStream_t cs, CudaTimer& timer, BatchTimings& t)
{
    const int n_bus = ctx.n_bus;
    const int S     = ctx.batch_size;

    // ①  V: the row's own starting point (set_v_init_from_orig) when given,
    //     else the base-case V; a phantom slot always the base case.
    timer.start();
    launch_gather_rows_to_slots(ctx.d_V_batch,
                                has_v_init_ ? thrust::raw_pointer_cast(d_V_init_all.data())
                                            : static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                                L, n_bus, cs);
    _chk_cuda(cudaGetLastError(), "V rows");
    t.t_tile_V += timer.stop_ms();

    // ②  base Ybus values
    timer.start();
    launch_gather_rows_to_slots(ctx.d_Ybus_values_batch, static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data()),
                                L, ctx.nnz_Y, cs);
    _chk_cuda(cudaGetLastError(), "tile Ybus");
    t.t_tile_Ybus += timer.stop_ms();

    // ③  the loaded rows' contingency (branch-trip) patches
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

    // ④  Re-seed generator target voltages (set_gen_v()), if configured: the
    //     loaded rows' columns (active-slot order already, see the ctor) into
    //     a per-slot scratch that is NaN on every other slot, so the reseed
    //     (NaN = leave alone) touches the loaded slots only.
    if (gen_v_override_.k_active() > 0) {
        const int k = gen_v_override_.k_active();
        if (d_gv_slots.size() < static_cast<size_t>(S) * k)
            d_gv_slots.resize(static_cast<size_t>(S) * k);
        cuda_real_type* d_gv = thrust::raw_pointer_cast(d_gv_slots.data());
        launch_fill_value(d_gv, std::numeric_limits<cuda_real_type>::quiet_NaN(),
                          static_cast<ptrdiff_t>(S) * k, cs);
        launch_gather_rows_to_slots(d_gv, thrust::raw_pointer_cast(d_gv_all.data()),
                                    static_cast<const cuda_real_type*>(nullptr), L, k, cs);
        apply_gen_v_kernel<<<nr_grid_size((long long)S * k, BS), BS, 0, cs>>>(
            ctx.d_V_batch, d_gv,
            thrust::raw_pointer_cast(d_gv_active_bus.data()),
            /*row_offset=*/0, k, S, n_bus);
        // ⑤  ... and the VoltageControl set-points those columns drive.
        gv_vset_.size_for(S, ctx.base.n_vc_grp);
        gv_vset_.load(thrust::raw_pointer_cast(ctx.base.d_vc_vset.data()), d_gv, k, L, S, cs);
    }

    // ⑥  |V| of the buses the reactive-limit outer loop holds PV again on a
    //     row (their Q row is pinned, so this value stays put).
    vr_gather_.plan(vr_seg_, L.h_slot, L.h_row, L.n, cs);
    if (vr_gather_.total > 0) {
        vr_gather_.gather(b_vr_bus, d_vr_bus, cs);
        vr_gather_.gather(b_vr_vm, d_vr_vm, cs);
        apply_vm_reseed_kernel<<<(vr_gather_.total + BS - 1) / BS, BS, 0, cs>>>(
            ctx.d_V_batch, vr_gather_.out_slot(),
            thrust::raw_pointer_cast(b_vr_bus.data()),
            thrust::raw_pointer_cast(b_vr_vm.data()),
            vr_gather_.total, n_bus);
        _chk_cuda(cudaGetLastError(), "vm reseed");
    }

    // ⑦  Sbus rows (base for a phantom slot), the redistribute_slack
    //     correction on top, and the per-row slack weights.
    timer.start();
    launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_Sbus_batch.data()),
                                thrust::raw_pointer_cast(d_Sbus_all.data()),
                                thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                                L, n_bus, cs);
    _chk_cuda(cudaGetLastError(), "Sbus rows");
    slack_.apply_dp(thrust::raw_pointer_cast(d_Sbus_batch.data()), L, n_bus, cs);
    slack_.load_weights(L, thrust::raw_pointer_cast(ctx.base.d_slack_w.data()), cs);
    t.t_tile_Sbus += timer.stop_ms();
}
