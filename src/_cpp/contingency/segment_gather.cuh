// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SEGMENT_GATHER_CUH
#define SEGMENT_GATHER_CUH

// =============================================================================
// contingency/segment_gather.cuh
//
// The variable-length per-row streams of the batch sources (Ybus patches,
// handle_disconnected_grid masks, |V| reseeds, redistribute_slack Sbus
// corrections) are kept in ACTIVE-ROW order with one segment per row
// (RowSegments: a CSR row pointer over the flat arrays the host builders emit
// when called with a chunk size of 1). A SegmentGather turns the segments of
// a list of (slot, row) pairs into one flat, slot-tagged stream -- exactly the
// layout the per-entry kernels (apply_contingencies_kernel, the mask kernels,
// apply_vm_reseed_kernel, add_sbus_delta_kernel) consume:
//
//   plan()   : on the host (which holds the row pointer and the slot -> row
//              table), the non-empty pairs' (slot, src start, dst start) and
//              the total; one H->D of that table from page-locked memory;
//              one kernel expanding it into out_slot[e] / out_src[e].
//   gather() : dst[e] = src[out_src[e]] for each field of the stream.
//
// Gathering the rows c*S .. c*S+n-1 of a stream into slots 0 .. n-1 gives,
// entry for entry, the chunk slice the builders emit with a chunk size of S.
// Device buffers only grow (no allocation on a steady schedule).
// =============================================================================

#include <algorithm>
#include <vector>

#include <thrust/device_vector.h>

#include "../contingency_analysis_helper.hpp"   // ChunkPatchRange
#include "../cuda_utils.h"                      // CudaEvent
#include "slot_schedule.cuh"                    // PinnedBuffer, BS, nr_grid_size

// -----------------------------------------------------------------------------
// RowSegments — CSR row pointer of a flat per-row stream.
// -----------------------------------------------------------------------------
struct RowSegments {
    std::vector<int> row_ptr;   // [n_rows + 1]; empty = no row has an entry

    // One ChunkPatchRange per row (a builder run with a chunk size of 1).
    void from_ranges(const std::vector<ChunkPatchRange>& ranges)
    {
        row_ptr.clear();
        if (ranges.empty()) return;
        row_ptr.resize(ranges.size() + 1);
        row_ptr[0] = ranges[0].start;
        for (size_t r = 0; r < ranges.size(); ++r)
            row_ptr[r + 1] = ranges[r].start + ranges[r].count;
    }
    bool any() const { return !row_ptr.empty() && row_ptr.back() > row_ptr.front(); }
    int  count(int row) const
    {
        if (row < 0 || row_ptr.empty()) return 0;
        return row_ptr[static_cast<size_t>(row) + 1] - row_ptr[static_cast<size_t>(row)];
    }
    int  start(int row) const { return row_ptr[static_cast<size_t>(row)]; }
    void clear() { row_ptr.clear(); }
};

// -----------------------------------------------------------------------------
// Kernels (templates: see slot_schedule.cuh)
// -----------------------------------------------------------------------------

// One thread per output entry e < total: the pair p owning e (last p with
// dst[p] <= e; pairs are non-empty and sorted by dst), then
// out_slot[e] = slot[p], out_src[e] = src[p] + (e - dst[p]).
template <typename I>
__global__ void expand_segments_kernel(const I* __restrict__ d_pairs, int n_pairs, int total,
                                       I* __restrict__ out_slot, I* __restrict__ out_src)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const I* slot = d_pairs;
    const I* src  = d_pairs + n_pairs;
    const I* dst  = d_pairs + 2 * n_pairs;
    int lo = 0, hi = n_pairs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (dst[mid] <= e) lo = mid; else hi = mid - 1;
    }
    out_slot[e] = slot[lo];
    out_src[e]  = src[lo] + (e - dst[lo]);
}

template <typename T>
__global__ void index_gather_kernel(T* __restrict__ dst, const T* __restrict__ src,
                                    const int* __restrict__ idx, int n)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n) dst[e] = src[idx[e]];
}

// -----------------------------------------------------------------------------
// SegmentGather
// -----------------------------------------------------------------------------
struct SegmentGather {
    int total   = 0;
    int n_pairs = 0;

    PinnedBuffer<int>          h_pairs;     // [3 * n_pairs]: slot | src | dst
    thrust::device_vector<int> d_pairs;
    thrust::device_vector<int> d_out_slot;  // [>= total]
    thrust::device_vector<int> d_out_src;   // [>= total]
    CudaEvent                  uploaded;    // the last H->D of h_pairs
    bool                       recorded = false;

    // Pairs (h_slot[i], h_row[i]), i < n (row -1 = none). Computes the plan,
    // ships it and expands it on cs. total == 0 → nothing launched.
    void plan(const RowSegments& seg, const int* h_slot, const int* h_row, int n,
              cudaStream_t cs)
    {
        total = 0;
        n_pairs = 0;
        if (!seg.any() || n <= 0) return;
        if (recorded) uploaded.synchronize();   // h_pairs may still be in flight
        // count first (pinned buffers only grow)
        int np = 0;
        for (int i = 0; i < n; ++i)
            if (seg.count(h_row[i]) > 0) ++np;
        if (np == 0) return;
        if (h_pairs.size < static_cast<size_t>(3 * np))
            h_pairs.resize(static_cast<size_t>(3 * std::max(np, 64)));
        int* slot = h_pairs.ptr;
        int* src  = h_pairs.ptr + np;
        int* dst  = h_pairs.ptr + 2 * np;
        int p = 0, acc = 0;
        for (int i = 0; i < n; ++i) {
            const int c = seg.count(h_row[i]);
            if (c <= 0) continue;
            slot[p] = h_slot[i];
            src[p]  = seg.start(h_row[i]);
            dst[p]  = acc;
            acc += c;
            ++p;
        }
        n_pairs = np;
        total   = acc;
        if (d_pairs.size() < static_cast<size_t>(3 * np))
            d_pairs.resize(static_cast<size_t>(3 * std::max(np, 64)));
        if (d_out_slot.size() < static_cast<size_t>(total)) {
            const size_t cap = static_cast<size_t>(total) + static_cast<size_t>(total) / 2 + 64;
            d_out_slot.resize(cap);
            d_out_src.resize(cap);
        }
        CHK_CUDA_SEG(cudaMemcpyAsync(thrust::raw_pointer_cast(d_pairs.data()), h_pairs.ptr,
                                     static_cast<size_t>(3 * np) * sizeof(int),
                                     cudaMemcpyHostToDevice, cs));
        uploaded.record(cs);
        recorded = true;
        expand_segments_kernel<int><<<(total + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(d_pairs.data()), np, total,
            thrust::raw_pointer_cast(d_out_slot.data()),
            thrust::raw_pointer_cast(d_out_src.data()));
    }

    const int* out_slot() const { return thrust::raw_pointer_cast(d_out_slot.data()); }

    // One field of the stream: dst[e] = src[out_src[e]], e < total.
    template <typename T>
    void gather(thrust::device_vector<T>& dst, const thrust::device_vector<T>& src,
                cudaStream_t cs) const
    {
        if (total <= 0) return;
        if (dst.size() < static_cast<size_t>(total))
            dst.resize(static_cast<size_t>(total) + static_cast<size_t>(total) / 2 + 64);
        index_gather_kernel<T><<<(total + BS - 1) / BS, BS, 0, cs>>>(
            thrust::raw_pointer_cast(dst.data()), thrust::raw_pointer_cast(src.data()),
            thrust::raw_pointer_cast(d_out_src.data()), total);
    }

private:
    static void CHK_CUDA_SEG(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("[segment_gather] CUDA error: ")
                                     + cudaGetErrorString(e));
    }
};

#endif  // SEGMENT_GATHER_CUH
