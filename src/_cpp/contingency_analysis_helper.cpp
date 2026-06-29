// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_helper.cpp
// =============================================================================

#include "contingency_analysis_helper.hpp"

#include <algorithm>        // std::lower_bound, std::min
#include <cassert>
#include <queue>            // std::queue (BFS)
#include <stdexcept>
#include <string>
#include <unordered_set>    // std::unordered_set (O(1) flat-index lookup)
#include <utility>          // std::pair
#include <vector>

// ---------------------------------------------------------------------------
// csr_find_k
// ---------------------------------------------------------------------------
int csr_find_k(const int* row_ptr, const int* col_ind, int row, int col)
{
    const int* begin = col_ind + row_ptr[row];
    const int* end   = col_ind + row_ptr[row + 1];
    const int* it    = std::lower_bound(begin, end, col);

    // In debug builds, verify the entry actually exists.
    // In release builds the assert compiles away but the UB is on the caller
    // (passing a (row, col) pair that is structurally zero in Ybus is a bug).
    assert(it != end && *it == col
           && "csr_find_k: entry (row, col) not found — "
              "contingency modifies a structural zero in Ybus");

    return static_cast<int>(it - col_ind);
}

// ---------------------------------------------------------------------------
// resolve_indices
// ---------------------------------------------------------------------------
void resolve_indices(
    std::vector<Contingency>& contingencies,
    const int*                row_ptr,
    const int*                col_ind)
{
    for (auto& ctg : contingencies)
        for (auto& t : ctg.triplets)
            t.k = csr_find_k(row_ptr, col_ind, t.row, t.col);
}

// ---------------------------------------------------------------------------
// build_flat_patches
// ---------------------------------------------------------------------------
void build_flat_patches(
    std::vector<Contingency>&    contingencies,
    int                          batch_size,
    std::vector<int>&            h_flat_ctg_id,
    std::vector<int>&            h_flat_k,
    std::vector<cuda_real_type>& h_flat_delta_re,
    std::vector<cuda_real_type>& h_flat_delta_im,
    std::vector<ChunkPatchRange>& chunk_ranges,
    std::vector<int>&            active_to_orig)
{
    // Merge triplets with identical k within each contingency.
    // Sort by k, then sum consecutive entries with the same k.
    // This handles N-2 contingencies and parallel lines sharing buses.
    for (auto& ctg : contingencies)
    {
        auto& tr = ctg.triplets;

        // Sort by CSR flat index k
        std::sort(tr.begin(), tr.end(),
                  [](const Triplet& a, const Triplet& b){ return a.k < b.k; });

        // Reduce-by-key: accumulate consecutive entries with identical k
        std::vector<Triplet> merged;
        merged.reserve(tr.size());
        for (const auto& t : tr)
        {
            if (!merged.empty() && merged.back().k == t.k)
            {
                merged.back().delta_re += t.delta_re;
                merged.back().delta_im += t.delta_im;
            }
            else
            {
                merged.push_back(t);
            }
        }
        tr = std::move(merged);
    }

    // Compaction: build the list of ACTIVE (connected) contingencies.  Each
    // active slot maps back to its original index via active_to_orig so the
    // caller can scatter results into the full-size output and leave the
    // disconnected slots as NaN.  Disconnected contingencies never enter the
    // batch and so consume no factorize / solve work.
    const int n_cont = static_cast<int>(contingencies.size());
    active_to_orig.clear();
    active_to_orig.reserve(n_cont);
    for (int i = 0; i < n_cont; ++i)
        if (!contingencies[i].disconnected)
            active_to_orig.push_back(i);

    const int n_active = static_cast<int>(active_to_orig.size());
    const int n_chunks = (n_active + batch_size - 1) / batch_size;

    // Pre-allocate: count total patches once to avoid repeated reallocation.
    int total_patches = 0;
    for (int orig : active_to_orig)
        total_patches += static_cast<int>(contingencies[orig].triplets.size());

    h_flat_ctg_id.clear();   h_flat_ctg_id.reserve(total_patches);
    h_flat_k.clear();         h_flat_k.reserve(total_patches);
    h_flat_delta_re.clear();  h_flat_delta_re.reserve(total_patches);
    h_flat_delta_im.clear();  h_flat_delta_im.reserve(total_patches);
    chunk_ranges.resize(n_chunks);

    int flat_idx = 0;
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        const int a_start      = chunk * batch_size;
        const int a_end        = std::min(a_start + batch_size, n_active);
        const int actual_batch = a_end - a_start;
        const int chunk_start  = flat_idx;

        for (int local_c = 0; local_c < actual_batch; ++local_c)
        {
            const auto& ctg = contingencies[active_to_orig[a_start + local_c]];
            for (const auto& t : ctg.triplets)
            {
                // Validate that resolve_indices() was called first.
                if (t.k < 0)
                    throw std::logic_error(
                        "build_flat_patches: triplet has k == -1; "
                        "call resolve_indices() before build_flat_patches()");

                // ctg_id is chunk-relative so the kernel needs no extra offset.
                h_flat_ctg_id.push_back(local_c);
                h_flat_k.push_back(t.k);

                // Cast from double to cuda_real_type (FP32 or FP64 at build time).
                h_flat_delta_re.push_back(static_cast<cuda_real_type>(t.delta_re));
                h_flat_delta_im.push_back(static_cast<cuda_real_type>(t.delta_im));
                ++flat_idx;
            }
        }

        chunk_ranges[chunk] = {chunk_start, flat_idx - chunk_start};
    }
}

// ---------------------------------------------------------------------------
// check_connectivity
// ---------------------------------------------------------------------------
void check_connectivity(
    std::vector<Contingency>&                                     contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm)
{
    const int  n_bus  = Ybus_rm.rows();
    const int* outer  = Ybus_rm.outerIndexPtr();
    const int* inner  = Ybus_rm.innerIndexPtr();

    // Build adjacency once from off-diagonal Ybus non-zeros.
    // adj[u] = list of (neighbor_bus, flat_CSR_index) pairs.
    // The flat CSR index is used to identify edges removed by a contingency.
    std::vector<std::vector<std::pair<int, int> > > adj(
        static_cast<size_t>(n_bus));
    for (int u = 0; u < n_bus; ++u) {
        for (int idx = outer[u]; idx < outer[u + 1]; ++idx) {
            int v = inner[idx];
            if (v != u)  // off-diagonal only — these are the graph edges
                adj[static_cast<size_t>(u)].push_back(std::make_pair(v, idx));
        }
    }

    // Threshold below which a Ybus entry is considered zero after the patch.
    const double eps = 1e-10;

    const eigen_cplx_type* values = Ybus_rm.valuePtr();

    std::vector<bool> visited(static_cast<size_t>(n_bus));
    std::queue<int>   q;

    for (auto& ctg : contingencies) {
        // Collect flat CSR indices of off-diagonal entries whose post-contingency
        // value would be (near) zero.  Only those edges are removed from the graph.
        std::unordered_set<int> removed;
        for (const auto& t : ctg.triplets) {
            if (t.row == t.col) continue;   // diagonal — not a graph edge
            const eigen_cplx_type new_val =
                values[t.k] - eigen_cplx_type(t.delta_re, t.delta_im);
            if (std::abs(new_val) < eps)
                removed.insert(t.k);
        }

        if (removed.empty()) continue;  // no edges removed → still connected

        // BFS from node 0 over the base graph, skipping removed edges.
        std::fill(visited.begin(), visited.end(), false);
        while (!q.empty()) q.pop();  // clear queue (reuse to avoid realloc)

        visited[0] = true;
        q.push(0);
        int count = 1;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            const std::vector<std::pair<int, int> >& nbrs =
                adj[static_cast<size_t>(u)];
            for (size_t i = 0; i < nbrs.size(); ++i) {
                int v      = nbrs[i].first;
                int flat_k = nbrs[i].second;
                if (!visited[static_cast<size_t>(v)]
                        && removed.count(flat_k) == 0) {
                    visited[static_cast<size_t>(v)] = true;
                    ++count;
                    q.push(v);
                }
            }
        }

        if (count < n_bus)
            ctg.disconnected = true;
    }
}

// ---------------------------------------------------------------------------
// build_blockdiag_csr
// ---------------------------------------------------------------------------
void build_blockdiag_csr(
    int                  n_bus,
    int                  nnz,
    const int*           single_outer,
    const int*           single_inner,
    int                  batch_size,
    std::vector<int>&    h_batch_outer,
    std::vector<int>&    h_batch_inner)
{
    h_batch_outer.resize(static_cast<size_t>(batch_size) * n_bus + 1);
    h_batch_inner.resize(static_cast<size_t>(batch_size) * nnz);

    for (int b = 0; b < batch_size; ++b)
    {
        // Row pointers for block b: single_outer[r] shifted by b * nnz so that
        // row (b * n_bus + r) starts at flat position b * nnz + single_outer[r].
        for (int r = 0; r < n_bus; ++r)
            h_batch_outer[b * n_bus + r] = single_outer[r] + b * nnz;

        // Column indices for block b: each column j within the block maps to
        // global column b * n_bus + j in the block-diagonal matrix.
        for (int i = 0; i < nnz; ++i)
            h_batch_inner[b * nnz + i] = single_inner[i] + b * n_bus;
    }

    // Sentinel: last row pointer is the total number of non-zeros.
    h_batch_outer[batch_size * n_bus] = batch_size * nnz;
}