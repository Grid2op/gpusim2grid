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
// that moves it needs its own v_set: this tiles the base per-group v_set over
// the chunk and writes each non-NaN group column in (apply_gen_vset_kernel),
// then points NrIterBuffers::d_vc_vset at the result with a per-slot stride.
// Inactive (no kernel, stride 0, bit-identical) unless some active gen_v
// column drives a group.
// =============================================================================

#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../acpf_nr_kernels.cuh"   // tile_vc_vset_kernel, apply_gen_vset_kernel
#include "../../nr_iter_step.cuh"      // NrIterBuffers, BS, nr_grid_size

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

    // This chunk's slots: base v_set everywhere (phantom slots included),
    // then every non-NaN group column of rows [row_offset, row_offset + actual_batch).
    void prepare(const cuda_real_type* d_vset_base, int n_vc_grp,
                 const cuda_real_type* d_gen_v_all, int k_active,
                 int row_offset, int actual_batch, int batch_size, cudaStream_t cs)
    {
        n_grp = n_vc_grp;
        if (!active || n_grp <= 0 || k_active <= 0) return;
        d_vset_batch.resize(static_cast<size_t>(batch_size) * n_grp);
        tile_vc_vset_kernel<<<nr_grid_size((long long)batch_size * n_grp, BS), BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_vset_batch.data()), d_vset_base, n_grp, batch_size);
        if (actual_batch > 0)
            apply_gen_vset_kernel<<<nr_grid_size((long long)actual_batch * k_active, BS), BS, 0, cs>>>(
                thrust::raw_pointer_cast(d_vset_batch.data()), d_gen_v_all,
                thrust::raw_pointer_cast(d_active_group.data()),
                row_offset, k_active, actual_batch, n_grp);
    }

    void fill(NrIterBuffers& buf) const
    {
        if (!active || n_grp <= 0 || d_vset_batch.empty()) return;
        buf.d_vc_vset      = thrust::raw_pointer_cast(d_vset_batch.data());
        buf.vc_vset_stride = n_grp;
    }
};

#endif // GEN_VSET_SLOTS_CUH
