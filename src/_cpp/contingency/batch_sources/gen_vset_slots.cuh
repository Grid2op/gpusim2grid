// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GEN_VSET_SLOTS_CUH
#define GEN_VSET_SLOTS_CUH

// =============================================================================
// contingency/batch_sources/gen_vset_slots.cuh
//
// GenVsetSlots — the VoltageControl half of a gen_v override, shared by
// InjectionBatch and ScenarioSweepBatch. A generator regulating the bus of a
// VoltageControl group (a remote regulator, or a local one on a
// group-controlled bus) does not fix |V| anywhere: its set-point is the
// group's v_set in the bordered row |V_reg| + s.Q - v_set = 0. So a batch row
// that moves it needs its own v_set: load() writes the base per-group v_set
// into the loaded slots and each non-NaN group column on top
// (apply_gen_vset_kernel), and fill() points NrIterBuffers::d_vc_vset at the
// result with a per-slot stride.
// Inactive (no kernel, stride 0, bit-identical) unless some active gen_v
// column drives a group.
// =============================================================================

#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../acpf_nr_kernels.cuh"   // tile_vc_vset_kernel, apply_gen_vset_kernel
#include "../../nr_iter_step.cuh"      // NrIterBuffers, BS, nr_grid_size
#include "../slot_schedule.cuh"         // SlotLoadView, launch_gather_rows_to_slots

struct GenVsetSlots {
    thrust::device_vector<int>            d_active_group;   // [k_active]
    thrust::device_vector<cuda_real_type> d_vset_batch;     // [batch_size * n_grp]
    bool active = false;
    int  n_grp  = 0;

    // group[j]: the VoltageControl group gen_v column j drives, -1 for none
    void upload(const std::vector<int>& group, cudaStream_t cs)
    {
        active = false;
        for (int g : group) if (g >= 0) { active = true; break; }
        if (active) upload_h2d(d_active_group, group.data(), group.size(), cs);
    }

    void clear() { active = false; }

    // Size the per-slot buffer for a batch of `capacity` slots. Without it
    // fill() would see an empty buffer and leave every slot on the base v_set
    // -- call it once the capacity is known, before load().
    void size_for(int capacity, int n_vc_grp)
    {
        n_grp = n_vc_grp;
        if (!active || n_grp <= 0) return;
        d_vset_batch.resize(static_cast<size_t>(capacity) * n_grp);
    }

    // Load path: the base v_set into the loaded slots, then every non-NaN
    // group column of d_gen_v_slots ([S * k_active], the gen_v of the slots
    // being loaded, NaN on every other slot -- which keep their own v_set).
    void load(const cuda_real_type* d_vset_base, const cuda_real_type* d_gen_v_slots,
              int k_active, const SlotLoadView& L, int S, cudaStream_t cs)
    {
        if (!active || n_grp <= 0 || k_active <= 0 || d_vset_batch.empty()) return;
        launch_gather_rows_to_slots(thrust::raw_pointer_cast(d_vset_batch.data()),
                                    static_cast<const cuda_real_type*>(nullptr), d_vset_base,
                                    L, n_grp, cs);
        if (S > 0)
            apply_gen_vset_kernel<<<nr_grid_size((long long)S * k_active, BS), BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_vset_batch.data()), d_gen_v_slots,
                thrust::raw_pointer_cast(d_active_group.data()),
                /*row_offset=*/0, k_active, S, n_grp);
    }

    void fill(NrIterBuffers& buf) const
    {
        if (!active || n_grp <= 0 || d_vset_batch.empty()) return;
        buf.d_vc_vset      = thrust::raw_pointer_cast(d_vset_batch.data());
        buf.vc_vset_stride = n_grp;
    }
};

#endif // GEN_VSET_SLOTS_CUH
