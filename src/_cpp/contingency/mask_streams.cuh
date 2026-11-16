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
// ScenarioSweepBatch. The entries are kept per ACTIVE ROW (MaskEntries built
// with a chunk size of 1: one range per row), and bind() gathers the rows the
// batch slots hold into one slot-tagged stream per kind (see
// segment_gather.cuh), whose pointers and counts it hands to NrIterBuffers --
// the per-entry kernels need no offset arithmetic and do not change.
//
//   rows : identity-row entries (masked P/Q rows + PV-pinned Q rows)
//   v    : masked-voltage NaN entries
//   jov  : per-slot J value overrides (stranded lone controller)
//   str  : stranded rows (F[v_row] = -Q_c)
//   vcp  : controller rows held at a fixed Q (F[row] = -(Q_c - target))
//
// The chunked schedule binds once per chunk (rows c*S + s in slot s, the same
// entries in the same order as a chunk slice); the continuous schedule after
// every round that loaded a slot.
// =============================================================================

#include <thrust/device_vector.h>
#include <vector>

#include "../dtypes.hpp"
#include "../cuda_utils.h"
#include "../contingency_analysis_helper.hpp"   // MaskEntries, ChunkPatchRange
#include "../nr_iter_step.cuh"                  // NrIterBuffers
#include "segment_gather.cuh"                   // RowSegments, SegmentGather
#include "slot_schedule.cuh"                    // SlotTableView

struct MaskStreams {
    MaskEntries h;   // per active row (chunk size 1)

    // per-row device data (the slot columns of h are not needed: bind() writes
    // the slot ids)
    RowSegments seg_row, seg_v, seg_jov, seg_str, seg_vcp;
    thrust::device_vector<int>            d_row, d_diag;
    thrust::device_vector<int>            d_v_bus;
    thrust::device_vector<int>            d_jov_pos;
    thrust::device_vector<cuda_real_type> d_jov_val;
    thrust::device_vector<int>            d_str_grp;
    thrust::device_vector<int>            d_vcp_row, d_vcp_ctrl;
    thrust::device_vector<cuda_real_type> d_vcp_target;

    // the batch streams of the current slot table
    SegmentGather g_row, g_v, g_jov, g_str, g_vcp;
    thrust::device_vector<int>            b_row, b_diag;
    thrust::device_vector<int>            b_v_bus;
    thrust::device_vector<int>            b_jov_pos;
    thrust::device_vector<cuda_real_type> b_jov_val;
    thrust::device_vector<int>            b_str_grp;
    thrust::device_vector<int>            b_vcp_row, b_vcp_ctrl;
    thrust::device_vector<cuda_real_type> b_vcp_target;

    std::vector<int> iota_;   // slot ids 0 .. S-1 (host)

    bool any() const { return h.any(); }

    // H→D upload of every non-empty stream (on the driver's stream).
    void upload(cudaStream_t cs)
    {
        seg_row.from_ranges(h.row_ranges);
        seg_v.from_ranges(h.v_ranges);
        seg_jov.from_ranges(h.jov_ranges);
        seg_str.from_ranges(h.str_ranges);
        seg_vcp.from_ranges(h.vcp_ranges);
        if (!h.slot.empty()) {
            upload_h2d(d_row,  h.row.data(),  h.row.size(),  cs);
            upload_h2d(d_diag, h.diag.data(), h.diag.size(), cs);
        }
        if (!h.v_slot.empty())
            upload_h2d(d_v_bus, h.v_bus.data(), h.v_bus.size(), cs);
        if (!h.jov_slot.empty()) {
            upload_h2d(d_jov_pos, h.jov_pos.data(), h.jov_pos.size(), cs);
            upload_h2d(d_jov_val, h.jov_val.data(), h.jov_val.size(), cs);
        }
        if (!h.str_slot.empty())
            upload_h2d(d_str_grp, h.str_grp.data(), h.str_grp.size(), cs);
        if (!h.vcp_slot.empty()) {
            upload_h2d(d_vcp_row,    h.vcp_row.data(),    h.vcp_row.size(),    cs);
            upload_h2d(d_vcp_ctrl,   h.vcp_ctrl.data(),   h.vcp_ctrl.size(),   cs);
            upload_h2d(d_vcp_target, h.vcp_target.data(), h.vcp_target.size(), cs);
        }
    }

    // Gather the streams of the rows the slots hold (T) and point the
    // NrIterBuffers mask fields at them. A kind with no entry in the batch
    // leaves its fields at their off defaults (null / 0), so the
    // corresponding launch is skipped. d_J_outer is the shared base-case J
    // skeleton row-pointer array (used by the mask kernel).
    void bind(const SlotTableView& T, NrIterBuffers& buf, const int* d_J_outer, cudaStream_t cs)
    {
        buf.d_J_outer_mask = d_J_outer;
        if (static_cast<int>(iota_.size()) < T.S) {
            const int old = static_cast<int>(iota_.size());
            iota_.resize(static_cast<size_t>(T.S));
            for (int s = old; s < T.S; ++s) iota_[static_cast<size_t>(s)] = s;
        }
        const int* slots = iota_.data();

        g_row.plan(seg_row, slots, T.h_slot_row, T.S, cs);
        if (g_row.total > 0) {
            g_row.gather(b_row, d_row, cs);
            g_row.gather(b_diag, d_diag, cs);
            buf.d_mask_slot = g_row.out_slot();
            buf.d_mask_row  = thrust::raw_pointer_cast(b_row.data());
            buf.d_mask_diag = thrust::raw_pointer_cast(b_diag.data());
            buf.n_mask_rows = g_row.total;
        }
        g_v.plan(seg_v, slots, T.h_slot_row, T.S, cs);
        if (g_v.total > 0) {
            g_v.gather(b_v_bus, d_v_bus, cs);
            buf.d_maskv_slot = g_v.out_slot();
            buf.d_maskv_bus  = thrust::raw_pointer_cast(b_v_bus.data());
            buf.n_mask_v     = g_v.total;
        }
        g_jov.plan(seg_jov, slots, T.h_slot_row, T.S, cs);
        if (g_jov.total > 0) {
            g_jov.gather(b_jov_pos, d_jov_pos, cs);
            g_jov.gather(b_jov_val, d_jov_val, cs);
            buf.d_jov_slot = g_jov.out_slot();
            buf.d_jov_pos  = thrust::raw_pointer_cast(b_jov_pos.data());
            buf.d_jov_val  = thrust::raw_pointer_cast(b_jov_val.data());
            buf.n_jov      = g_jov.total;
        }
        g_str.plan(seg_str, slots, T.h_slot_row, T.S, cs);
        if (g_str.total > 0) {
            g_str.gather(b_str_grp, d_str_grp, cs);
            buf.d_str_slot = g_str.out_slot();
            buf.d_str_grp  = thrust::raw_pointer_cast(b_str_grp.data());
            buf.n_str      = g_str.total;
        }
        g_vcp.plan(seg_vcp, slots, T.h_slot_row, T.S, cs);
        if (g_vcp.total > 0) {
            g_vcp.gather(b_vcp_row, d_vcp_row, cs);
            g_vcp.gather(b_vcp_ctrl, d_vcp_ctrl, cs);
            g_vcp.gather(b_vcp_target, d_vcp_target, cs);
            buf.d_vcp_slot   = g_vcp.out_slot();
            buf.d_vcp_row    = thrust::raw_pointer_cast(b_vcp_row.data());
            buf.d_vcp_ctrl   = thrust::raw_pointer_cast(b_vcp_ctrl.data());
            buf.d_vcp_target = thrust::raw_pointer_cast(b_vcp_target.data());
            buf.n_vcp        = g_vcp.total;
        }
    }
};

#endif // MASK_STREAMS_CUH
