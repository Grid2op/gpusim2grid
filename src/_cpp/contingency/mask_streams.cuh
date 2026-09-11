// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef MASK_STREAMS_CUH
#define MASK_STREAMS_CUH

// =============================================================================
// contingency/mask_streams.cuh
//
// MaskStreams — the device-side counterpart of MaskEntries (see
// contingency_analysis_helper.hpp), shared by ContingencyBatch and
// ScenarioSweepBatch: owns the uploaded flat streams and hands the current
// chunk's slices to NrIterBuffers. Every stream is chunk-sliced with one
// ChunkPatchRange per chunk (chunk-relative slot ids), so fill() only offsets
// a base pointer per stream -- the kernels need no offset arithmetic.
//
//   rows : identity-row entries (masked P/Q rows + PV-pinned Q rows)
//   v    : masked-voltage NaN entries
//   jov  : per-slot J value overrides (stranded lone controller)
//   str  : stranded rows (F[v_row] = -Q_c)
// =============================================================================

#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "../cuda_utils.h"
#include "../contingency_analysis_helper.hpp"   // MaskEntries, ChunkPatchRange
#include "../nr_iter_step.cuh"                  // NrIterBuffers

struct MaskStreams {
    MaskEntries h;

    thrust::device_vector<int>            d_slot, d_row, d_diag;
    thrust::device_vector<int>            d_v_slot, d_v_bus;
    thrust::device_vector<int>            d_jov_slot, d_jov_pos;
    thrust::device_vector<cuda_real_type> d_jov_val;
    thrust::device_vector<int>            d_str_slot, d_str_grp;

    bool any() const { return h.any(); }

    // H→D upload of every non-empty stream (on the driver's stream).
    void upload(cudaStream_t cs)
    {
        if (!h.slot.empty()) {
            upload_h2d(d_slot, h.slot.data(), h.slot.size(), cs);
            upload_h2d(d_row,  h.row.data(),  h.row.size(),  cs);
            upload_h2d(d_diag, h.diag.data(), h.diag.size(), cs);
        }
        if (!h.v_slot.empty()) {
            upload_h2d(d_v_slot, h.v_slot.data(), h.v_slot.size(), cs);
            upload_h2d(d_v_bus,  h.v_bus.data(),  h.v_bus.size(),  cs);
        }
        if (!h.jov_slot.empty()) {
            upload_h2d(d_jov_slot, h.jov_slot.data(), h.jov_slot.size(), cs);
            upload_h2d(d_jov_pos,  h.jov_pos.data(),  h.jov_pos.size(),  cs);
            upload_h2d(d_jov_val,  h.jov_val.data(),  h.jov_val.size(),  cs);
        }
        if (!h.str_slot.empty()) {
            upload_h2d(d_str_slot, h.str_slot.data(), h.str_slot.size(), cs);
            upload_h2d(d_str_grp,  h.str_grp.data(),  h.str_grp.size(),  cs);
        }
    }

    // Point the NrIterBuffers mask fields at this chunk's slices. Leaves the
    // fields at their off defaults (null / 0) for a stream with nothing in
    // this chunk, so the corresponding launch is skipped. d_J_outer is the
    // shared base-case J skeleton row-pointer array (used by the mask kernel).
    void fill(NrIterBuffers& buf, int chunk_idx, const int* d_J_outer) const
    {
        buf.d_J_outer_mask = d_J_outer;
        auto slice = [&](const std::vector<ChunkPatchRange>& ranges) -> ChunkPatchRange {
            if (chunk_idx < 0 || chunk_idx >= static_cast<int>(ranges.size()))
                return ChunkPatchRange{0, 0};
            return ranges[static_cast<size_t>(chunk_idx)];
        };
        {
            const ChunkPatchRange r = slice(h.row_ranges);
            if (r.count > 0) {
                buf.d_mask_slot = thrust::raw_pointer_cast(d_slot.data()) + r.start;
                buf.d_mask_row  = thrust::raw_pointer_cast(d_row.data())  + r.start;
                buf.d_mask_diag = thrust::raw_pointer_cast(d_diag.data()) + r.start;
                buf.n_mask_rows = r.count;
            }
        }
        {
            const ChunkPatchRange r = slice(h.v_ranges);
            if (r.count > 0) {
                buf.d_maskv_slot = thrust::raw_pointer_cast(d_v_slot.data()) + r.start;
                buf.d_maskv_bus  = thrust::raw_pointer_cast(d_v_bus.data())  + r.start;
                buf.n_mask_v     = r.count;
            }
        }
        {
            const ChunkPatchRange r = slice(h.jov_ranges);
            if (r.count > 0) {
                buf.d_jov_slot = thrust::raw_pointer_cast(d_jov_slot.data()) + r.start;
                buf.d_jov_pos  = thrust::raw_pointer_cast(d_jov_pos.data())  + r.start;
                buf.d_jov_val  = thrust::raw_pointer_cast(d_jov_val.data())  + r.start;
                buf.n_jov      = r.count;
            }
        }
        {
            const ChunkPatchRange r = slice(h.str_ranges);
            if (r.count > 0) {
                buf.d_str_slot = thrust::raw_pointer_cast(d_str_slot.data()) + r.start;
                buf.d_str_grp  = thrust::raw_pointer_cast(d_str_grp.data())  + r.start;
                buf.n_str      = r.count;
            }
        }
    }
};

#endif // MASK_STREAMS_CUH
