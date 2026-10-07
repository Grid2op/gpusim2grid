// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/batch_sources/injection_batch.cu
// =============================================================================

#include "injection_batch.cuh"
#include "../batch_pf_driver.cuh"   // BatchPfDriverContext (complete type)

#include <limits>
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

    // Tile base Ybus values into every batch slot — ONCE.
    launch_tile(ctx.d_Ybus_values_batch,
                thrust::raw_pointer_cast(ctx.base.d_Ybus_values.data()),
                ctx.nnz_Y, ctx.batch_size, cs);
    _chk_cuda(cudaGetLastError(), "tile base Ybus");

    // set_gen_v() overrides, if any -- see GenVOverride's own doc.
    if (gen_v_override_.k_active() > 0) {
        upload_h2d(d_gv_active_bus, gen_v_override_.h_active_bus.data(),
                   gen_v_override_.h_active_bus.size(), cs);
        upload_h2d(d_gv_all, gen_v_override_.h_gen_v_all.data(),
                   gen_v_override_.h_gen_v_all.size(), cs);
        gv_vset_.upload(gen_v_override_.h_active_vc_group, cs);
        gv_vset_.size_for(ctx.batch_size, ctx.base.n_vc_grp);
        d_gv_slots.resize(static_cast<size_t>(ctx.batch_size) * gen_v_override_.k_active());
    }
}

void InjectionBatch::load_slots(BatchPfDriverContext& ctx,
                                const SlotLoadView&   L,
                                cudaStream_t          cs,
                                CudaTimer&            timer,
                                BatchTimings&         t)
{
    const int n_bus = ctx.n_bus;
    const int S     = ctx.batch_size;

    // ①  Base V into every loaded slot (each row starts from the converged
    //     base voltage).
    timer.start();
    launch_gather_rows_to_slots(ctx.d_V_batch, static_cast<const cudaComplexType*>(nullptr),
                                thrust::raw_pointer_cast(ctx.base.d_V_base.data()),
                                L, n_bus, cs);
    _chk_cuda(cudaGetLastError(), "tile V");
    t.t_tile_V += timer.stop_ms();
    // No t_tile_Ybus / t_patch_Ybus updates — Ybus is permanent.

    // Re-seed generator target voltages (set_gen_v()), if configured: the
    // loaded rows' columns into a per-slot scratch that is NaN on every other
    // slot, so the reseed (NaN = leave alone) touches the loaded slots only.
    if (gen_v_override_.k_active() > 0) {
        const int k = gen_v_override_.k_active();
        cuda_real_type* d_gv = thrust::raw_pointer_cast(d_gv_slots.data());
        launch_fill_value(d_gv, std::numeric_limits<cuda_real_type>::quiet_NaN(),
                          static_cast<ptrdiff_t>(S) * k, cs);
        launch_gather_rows_to_slots(d_gv, thrust::raw_pointer_cast(d_gv_all.data()),
                                    static_cast<const cuda_real_type*>(nullptr), L, k, cs);
        apply_gen_v_kernel<<<nr_grid_size((long long)S * k, BS), BS, 0, cs>>>(
            ctx.d_V_batch, d_gv,
            thrust::raw_pointer_cast(d_gv_active_bus.data()),
            /*row_offset=*/0, k, S, n_bus);
        gv_vset_.load(thrust::raw_pointer_cast(ctx.base.d_vc_vset.data()), d_gv, k, L, S, cs);
    }

    // ②  The rows' Sbus into the loaded slots; a phantom slot gets base.d_Sbus
    //     (phantom NR uses base V + base Ybus + base Sbus → converged in one
    //     step, results discarded).
    timer.start();
    launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_Sbus_batch.data()),
                                thrust::raw_pointer_cast(d_Sbus_all.data()),
                                thrust::raw_pointer_cast(ctx.base.d_Sbus.data()),
                                L, n_bus, cs);
    _chk_cuda(cudaGetLastError(), "Sbus rows");
    t.t_tile_Sbus += timer.stop_ms();
}
