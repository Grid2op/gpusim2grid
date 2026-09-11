// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_helper.cpp
// =============================================================================

#include "contingency_analysis_helper.hpp"

#include <algorithm>        // std::lower_bound, std::min, std::sort, std::max_element
#include <cmath>            // std::abs
#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>          // std::pair
#include <vector>

// ---------------------------------------------------------------------------
// build_contingency_from_branch_ids
// ---------------------------------------------------------------------------
Contingency build_contingency_from_branch_ids(
    const std::vector<int>&   branch_ids,
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect> yff_eff,
    Eigen::Ref<const CplxVect> yft_eff,
    Eigen::Ref<const CplxVect> ytf_eff,
    Eigen::Ref<const CplxVect> ytt_eff)
{
    const int n_branches = static_cast<int>(branch_from.size());

    Contingency ctg;
    ctg.tripped_branches = branch_ids;
    ctg.triplets.reserve(branch_ids.size() * 4);

    for (int l : branch_ids) {
        if (l < 0 || l >= n_branches)
            throw std::runtime_error(
                "build_contingency_from_branch_ids: branch index out of range");

        const int i = branch_from(l);
        const int j = branch_to(l);

        // A negative endpoint means that side of the branch is isolated in
        // the AC-solver's bus numbering (id_me_to_ac_solver relabels an
        // unattached bus to a negative id -- e.g. the open side of a
        // half-open line with keep_half_open_lines=True). Such a bus has no
        // row/col in Ybus_solver at all, so there is nothing to patch there:
        // emitting a triplet for it would make resolve_indices()'s
        // csr_find_k() index row_ptr[negative] (UB) and produce a garbage
        // flat CSR index downstream. Skip any triplet touching that endpoint;
        // the branch's contribution at its still-connected endpoint (if any)
        // is preserved.
        const bool i_valid = i >= 0;
        const bool j_valid = j >= 0;

        // π-model Ybus modifications to SUBTRACT for this branch trip:
        //   (i,i) → yff_eff   (ii self-admittance at from-bus)
        //   (j,j) → ytt_eff   (jj self-admittance at to-bus)
        //   (i,j) → yft_eff   (ij mutual admittance)
        //   (j,i) → ytf_eff   (ji mutual admittance)
        if (i_valid)
            ctg.triplets.push_back({i, i,  yff_eff(l).real(),  yff_eff(l).imag()});
        if (j_valid)
            ctg.triplets.push_back({j, j,  ytt_eff(l).real(),  ytt_eff(l).imag()});
        if (i_valid && j_valid) {
            ctg.triplets.push_back({i, j,  yft_eff(l).real(),  yft_eff(l).imag()});
            ctg.triplets.push_back({j, i,  ytf_eff(l).real(),  ytf_eff(l).imag()});
        }
    }
    return ctg;
}

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
// Connectivity queries
//
// Both check_connectivity and compute_component_masks answer the same
// question for every contingency: does removing this contingency's edges split
// the Ybus graph, and if so, which buses fall off the main component? The
// original implementation answered it with a full BFS per contingency --
// O(n_bus + nnz_Y) each, with an unordered_set probe on every edge -- which on
// a 7k-bus grid costs ~100 us per contingency and adds up to more than a
// second over an N-1 study, more than the GPU solve itself.
//
// The graph never changes within a batch, so the structure that decides the
// answer is built ONCE (ConnectivityIndex) and each contingency is reduced to
// a few array lookups:
//
//   * A DFS tree rooted at bus 0, with preorder numbering (tin/tout) and
//     low-links, marks every edge as tree/non-tree and every tree edge as
//     bridge/non-bridge (Tarjan). Removing only NON-tree edges leaves the
//     spanning tree intact, so the graph stays connected -- no search needed.
//   * A single removed edge that IS a bridge splits the graph in exactly two:
//     the DFS subtree below it (a contiguous preorder range [tin, tout)) and
//     the rest. The masked side is read straight off the preorder array.
//   * Several removed edges (N-k) cut the tree into pieces: the root piece and
//     one subtree per removed tree edge. Only NON-tree edges leaving those
//     subtrees can glue pieces back together, so the subtrees -- typically a
//     few antenna buses -- are scanned and the pieces merged with a union-find.
//     The work is proportional to the cut-off subtrees, not to the grid.
//   * Anything else (a cut-off side larger than half the grid, a removed set
//     that is not symmetric in Ybus, a base graph that is not connected to
//     begin with) falls back to a component labelling BFS, written with flat
//     arrays and a byte mark per removed edge instead of the hash set.
//
// The decisions are identical to the per-contingency BFS: the fast paths only
// fire where the DFS tree proves the answer, and the tie-break for "largest
// component" (first component in bus order wins, i.e. bus 0's side) is
// reproduced by never masking bus 0's side unless it is strictly the smaller.
//
// One deliberate difference: the removed-edge test now sums the deltas that
// several triplets of the same contingency put on the same Ybus entry before
// comparing with the base value (which is what build_flat_patches merges and
// the GPU applies). Tripping both circuits of a double line in one N-2
// contingency used to leave that edge "present" for the connectivity check
// because each circuit's delta alone does not zero the entry.
// ---------------------------------------------------------------------------
namespace {

// Threshold below which a Ybus entry is considered zero after the patch.
constexpr double kRemovedEdgeEps = 1e-10;

struct ConnectivityIndex {
    int        n_bus = 0;
    int        nnz   = 0;
    const int* outer = nullptr;   // Ybus_rm CSR row pointers
    const int* inner = nullptr;   // Ybus_rm CSR column indices

    std::vector<int>  row_of;      // [nnz]   row i of flat entry k
    std::vector<int>  k_rev;       // [nnz]   flat index of (j,i) for entry (i,j); -1 if absent / diagonal
    std::vector<char> is_tree;     // [nnz]   entry is a DFS-tree edge (both directions)
    std::vector<char> is_bridge;   // [nnz]   entry is a bridge (both directions)
    std::vector<int>  tree_child;  // [nnz]   deeper endpoint of a tree edge, -1 otherwise
    std::vector<int>  tin, tout;   // [n_bus] preorder index / end of subtree (exclusive)
    std::vector<int>  preorder;    // [n_bus] bus at each preorder position
    bool              base_connected = false;

    explicit ConnectivityIndex(const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm)
        : n_bus(static_cast<int>(Ybus_rm.rows()))
        , nnz(static_cast<int>(Ybus_rm.nonZeros()))
        , outer(Ybus_rm.outerIndexPtr())
        , inner(Ybus_rm.innerIndexPtr())
        , row_of(static_cast<size_t>(nnz))
        , k_rev(static_cast<size_t>(nnz), -1)
        , is_tree(static_cast<size_t>(nnz), 0)
        , is_bridge(static_cast<size_t>(nnz), 0)
        , tree_child(static_cast<size_t>(nnz), -1)
        , tin(static_cast<size_t>(n_bus), -1)
        , tout(static_cast<size_t>(n_bus), -1)
        , preorder(static_cast<size_t>(n_bus), -1)
    {
        for (int i = 0; i < n_bus; ++i)
            for (int p = outer[i]; p < outer[i + 1]; ++p) {
                row_of[static_cast<size_t>(p)] = i;
                const int j = inner[p];
                if (j == i) continue;
                const int* b = inner + outer[j];
                const int* e = inner + outer[j + 1];
                const int* it = std::lower_bound(b, e, i);
                if (it != e && *it == i) k_rev[static_cast<size_t>(p)] = static_cast<int>(it - inner);
            }
        if (n_bus == 0) { base_connected = true; return; }

        // Iterative DFS from bus 0: preorder numbering, low-links, bridges.
        std::vector<int> low(static_cast<size_t>(n_bus), 0);
        std::vector<int> parent_k(static_cast<size_t>(n_bus), -1);  // flat index of (parent -> bus)
        std::vector<int> it_pos(static_cast<size_t>(n_bus), 0);     // next CSR position to scan
        std::vector<int> stack;
        stack.reserve(static_cast<size_t>(n_bus));
        int counter = 0;
        tin[0] = low[0] = counter; preorder[counter++] = 0;
        it_pos[0] = outer[0];
        stack.push_back(0);
        while (!stack.empty()) {
            const int u = stack.back();
            int& p = it_pos[static_cast<size_t>(u)];
            if (p < outer[u + 1]) {
                const int k = p++;
                const int v = inner[k];
                if (v == u) continue;                                 // diagonal
                if (parent_k[static_cast<size_t>(u)] >= 0
                        && k_rev[static_cast<size_t>(k)] == parent_k[static_cast<size_t>(u)])
                    continue;                                         // the edge we came down
                if (tin[static_cast<size_t>(v)] < 0) {
                    tin[static_cast<size_t>(v)] = low[static_cast<size_t>(v)] = counter;
                    preorder[counter++] = v;
                    parent_k[static_cast<size_t>(v)] = k;
                    it_pos[static_cast<size_t>(v)] = outer[v];
                    is_tree[static_cast<size_t>(k)] = 1;
                    tree_child[static_cast<size_t>(k)] = v;
                    const int kr = k_rev[static_cast<size_t>(k)];
                    if (kr >= 0) { is_tree[static_cast<size_t>(kr)] = 1; tree_child[static_cast<size_t>(kr)] = v; }
                    stack.push_back(v);
                } else {
                    low[static_cast<size_t>(u)] = std::min(low[static_cast<size_t>(u)], tin[static_cast<size_t>(v)]);
                }
            } else {
                stack.pop_back();
                tout[static_cast<size_t>(u)] = counter;
                const int pk = parent_k[static_cast<size_t>(u)];
                if (pk >= 0) {
                    const int par = row_of[static_cast<size_t>(pk)];
                    low[static_cast<size_t>(par)] = std::min(low[static_cast<size_t>(par)], low[static_cast<size_t>(u)]);
                    if (low[static_cast<size_t>(u)] > tin[static_cast<size_t>(par)]) {
                        is_bridge[static_cast<size_t>(pk)] = 1;
                        const int kr = k_rev[static_cast<size_t>(pk)];
                        if (kr >= 0) is_bridge[static_cast<size_t>(kr)] = 1;
                    }
                }
            }
        }
        base_connected = (counter == n_bus);
    }

    bool in_subtree(int bus, int child) const {
        const int t = tin[static_cast<size_t>(bus)];
        return t >= tin[static_cast<size_t>(child)] && t < tout[static_cast<size_t>(child)];
    }
};

// Per-batch scratch: the removed-edge extraction, the N-k piece merging, and
// the BFS fallback all reuse these buffers across contingencies.
struct ConnectivityScratch {
    std::vector<int>    removed;       // flat indices removed by the current contingency
    std::vector<int>    acc_k;         // merged (k, delta) of the current contingency
    std::vector<double> acc_re, acc_im;
    std::vector<char>   removed_mark;  // [nnz] byte per flat index, set only while scanning
    std::vector<int>    cut_child;     // children of the removed tree edges, sorted by tin
    std::vector<int>    outer_lo, outer_hi;   // preorder ranges of the outermost cut subtrees
    std::vector<int>    uf;            // union-find parent over the pieces (0 = root piece)
    std::vector<int>    comp;          // [n_bus] component label (fallback)
    std::vector<int>    queue;         // [n_bus] BFS queue (fallback)
    std::vector<int>    count;         // component sizes (fallback)

    ConnectivityScratch(int n_bus, int nnz)
        : removed_mark(static_cast<size_t>(nnz), 0)
        , comp(static_cast<size_t>(n_bus), -1)
        , queue(static_cast<size_t>(n_bus), 0)
    {}

    // Off-diagonal flat indices whose post-contingency value is ~0. Deltas of
    // several triplets on the same entry are summed first (see the note above).
    void collect_removed(const Contingency& ctg, const eigen_cplx_type* values) {
        removed.clear();
        acc_k.clear(); acc_re.clear(); acc_im.clear();
        for (const auto& t : ctg.triplets) {
            if (t.row == t.col) continue;
            size_t idx = 0;
            for (; idx < acc_k.size(); ++idx) if (acc_k[idx] == t.k) break;
            if (idx == acc_k.size()) { acc_k.push_back(t.k); acc_re.push_back(0.); acc_im.push_back(0.); }
            acc_re[idx] += t.delta_re;
            acc_im[idx] += t.delta_im;
        }
        for (size_t idx = 0; idx < acc_k.size(); ++idx) {
            const eigen_cplx_type new_val = values[acc_k[idx]] - eigen_cplx_type(acc_re[idx], acc_im[idx]);
            if (std::abs(new_val) < kRemovedEdgeEps) removed.push_back(acc_k[idx]);
        }
    }

    bool is_removed(int k) const {
        for (int r : removed) if (r == k) return true;
        return false;
    }

    int uf_find(int a) {
        while (uf[static_cast<size_t>(a)] != a) {
            uf[static_cast<size_t>(a)] = uf[static_cast<size_t>(uf[static_cast<size_t>(a)])];
            a = uf[static_cast<size_t>(a)];
        }
        return a;
    }
    void uf_unite(int a, int b) {
        a = uf_find(a); b = uf_find(b);
        if (a != b) uf[static_cast<size_t>(std::max(a, b))] = std::min(a, b);
    }
};

// Decides, without a search, whether the current removed set splits the
// graph, and if so which buses fall off bus 0's side. Returns false when the
// DFS tree cannot settle it cheaply (the caller then labels components with a
// BFS). On true: `masked` holds the sorted buses outside the main component,
// empty when the graph stays connected.
bool fast_masked_set(const ConnectivityIndex& ix, ConnectivityScratch& sc, std::vector<int>& masked)
{
    masked.clear();
    if (sc.removed.empty()) return true;          // graph untouched
    if (!ix.base_connected) return false;

    // The removal must be symmetric for the undirected reasoning to hold.
    // Collect the removed TREE edges (their child endpoints); a removal that
    // touches no tree edge leaves the spanning tree -- and connectivity -- intact.
    sc.cut_child.clear();
    int n_und = 0, single_k = -1;
    for (int k : sc.removed) {
        const int kr = ix.k_rev[static_cast<size_t>(k)];
        if (kr < 0 || !sc.is_removed(kr)) return false;
        if (ix.row_of[static_cast<size_t>(k)] < ix.inner[k]) {
            ++n_und;
            single_k = k;
            if (ix.is_tree[static_cast<size_t>(k)]) sc.cut_child.push_back(ix.tree_child[static_cast<size_t>(k)]);
        }
    }
    if (sc.cut_child.empty()) return true;         // spanning tree survives
    const int n_bus = ix.n_bus;

    if (n_und == 1) {
        // One edge: it disconnects the graph iff it is a bridge, and then the
        // two components are the subtree below it and bus 0's side.
        if (!ix.is_bridge[static_cast<size_t>(single_k)]) return true;
        const int child = sc.cut_child[0];
        const int t0 = ix.tin[static_cast<size_t>(child)], t1 = ix.tout[static_cast<size_t>(child)];
        const int sub_size = t1 - t0;
        if (n_bus - sub_size >= sub_size) {
            masked.assign(ix.preorder.begin() + t0, ix.preorder.begin() + t1);
            std::sort(masked.begin(), masked.end());
        } else {
            masked.reserve(static_cast<size_t>(n_bus - sub_size));
            for (int b = 0; b < n_bus; ++b)
                if (!ix.in_subtree(b, child)) masked.push_back(b);
        }
        return true;
    }

    // Several edges: cut the tree at every removed tree edge. Piece 0 is bus
    // 0's side; piece i (1-based, children sorted by preorder) is the subtree
    // below the i-th cut minus the cuts nested inside it. Only non-tree edges
    // leaving a cut subtree can reconnect pieces, so the subtrees are scanned
    // and the pieces merged with a union-find.
    std::sort(sc.cut_child.begin(), sc.cut_child.end(),
              [&](int a, int b) { return ix.tin[static_cast<size_t>(a)] < ix.tin[static_cast<size_t>(b)]; });
    const int t = static_cast<int>(sc.cut_child.size());
    auto piece_of = [&](int bus) -> int {
        const int tb = ix.tin[static_cast<size_t>(bus)];
        for (int i = t - 1; i >= 0; --i) {
            const int c = sc.cut_child[static_cast<size_t>(i)];
            if (ix.tin[static_cast<size_t>(c)] <= tb && tb < ix.tout[static_cast<size_t>(c)]) return i + 1;
        }
        return 0;
    };

    // Outermost cut subtrees (the ranges actually scanned) and their total
    // size. If they cover more than half the grid, the plain BFS is no more
    // expensive and keeps the largest-component bookkeeping simple.
    sc.outer_lo.clear(); sc.outer_hi.clear();
    int scanned = 0;
    for (int i = 0; i < t; ++i) {
        const int c = sc.cut_child[static_cast<size_t>(i)];
        const int lo = ix.tin[static_cast<size_t>(c)], hi = ix.tout[static_cast<size_t>(c)];
        if (!sc.outer_hi.empty() && lo < sc.outer_hi.back()) continue;   // nested in the previous range
        sc.outer_lo.push_back(lo); sc.outer_hi.push_back(hi);
        scanned += hi - lo;
    }
    if (2 * scanned > n_bus) return false;

    sc.uf.resize(static_cast<size_t>(t) + 1);
    for (int i = 0; i <= t; ++i) sc.uf[static_cast<size_t>(i)] = i;
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 1;
    for (size_t r = 0; r < sc.outer_lo.size(); ++r) {
        for (int pos = sc.outer_lo[r]; pos < sc.outer_hi[r]; ++pos) {
            const int u  = ix.preorder[static_cast<size_t>(pos)];
            const int pu = piece_of(u);
            for (int p = ix.outer[u]; p < ix.outer[u + 1]; ++p) {
                const int v = ix.inner[p];
                if (v == u || ix.is_tree[static_cast<size_t>(p)] || sc.removed_mark[static_cast<size_t>(p)]) continue;
                sc.uf_unite(pu, piece_of(v));
            }
        }
    }
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 0;

    bool all_joined = true;
    for (int i = 1; i <= t; ++i) if (sc.uf_find(i) != 0) { all_joined = false; break; }
    if (all_joined) return true;

    // Bus 0's side keeps at least n_bus - scanned >= n_bus / 2 buses, so it is
    // the main component (on an exact tie it is the first in bus order, which
    // is what the labelling path picks). Everything not joined to it is masked.
    for (size_t r = 0; r < sc.outer_lo.size(); ++r)
        for (int pos = sc.outer_lo[r]; pos < sc.outer_hi[r]; ++pos) {
            const int u = ix.preorder[static_cast<size_t>(pos)];
            if (sc.uf_find(piece_of(u)) != 0) masked.push_back(u);
        }
    std::sort(masked.begin(), masked.end());
    return true;
}

// Fallback: label the connected components of the graph minus the removed
// (directed) entries, BFS from every unvisited bus in increasing order.
// Returns the number of components; sc.comp holds the labels.
int label_components(const ConnectivityIndex& ix, ConnectivityScratch& sc)
{
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 1;
    std::fill(sc.comp.begin(), sc.comp.end(), -1);
    int nb_comp = 0;
    for (int start = 0; start < ix.n_bus; ++start) {
        if (sc.comp[static_cast<size_t>(start)] != -1) continue;
        sc.comp[static_cast<size_t>(start)] = nb_comp;
        int head = 0, tail = 0;
        sc.queue[tail++] = start;
        while (head < tail) {
            const int u = sc.queue[head++];
            for (int p = ix.outer[u]; p < ix.outer[u + 1]; ++p) {
                const int v = ix.inner[p];
                if (v == u || sc.removed_mark[static_cast<size_t>(p)]) continue;
                if (sc.comp[static_cast<size_t>(v)] == -1) {
                    sc.comp[static_cast<size_t>(v)] = nb_comp;
                    sc.queue[tail++] = v;
                }
            }
        }
        ++nb_comp;
    }
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 0;
    return nb_comp;
}

// Fallback for check_connectivity: number of buses reachable from bus 0.
int count_reachable_from_0(const ConnectivityIndex& ix, ConnectivityScratch& sc)
{
    if (ix.n_bus == 0) return 0;
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 1;
    std::fill(sc.comp.begin(), sc.comp.end(), -1);
    sc.comp[0] = 0;
    int head = 0, tail = 0;
    sc.queue[tail++] = 0;
    while (head < tail) {
        const int u = sc.queue[head++];
        for (int p = ix.outer[u]; p < ix.outer[u + 1]; ++p) {
            const int v = ix.inner[p];
            if (v == u || sc.removed_mark[static_cast<size_t>(p)]) continue;
            if (sc.comp[static_cast<size_t>(v)] == -1) {
                sc.comp[static_cast<size_t>(v)] = 0;
                sc.queue[tail++] = v;
            }
        }
    }
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 0;
    return tail;
}

}  // namespace

// ---------------------------------------------------------------------------
// check_connectivity
// ---------------------------------------------------------------------------
void check_connectivity(
    std::vector<Contingency>&                                     contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm)
{
    const ConnectivityIndex ix(Ybus_rm);
    ConnectivityScratch     sc(ix.n_bus, ix.nnz);
    const eigen_cplx_type*  values = Ybus_rm.valuePtr();
    std::vector<int>        masked;

    for (auto& ctg : contingencies) {
        if (ctg.skip) { ctg.disconnected = true; continue; }
        sc.collect_removed(ctg, values);
        if (fast_masked_set(ix, sc, masked)) {
            if (!masked.empty()) ctg.disconnected = true;
        } else if (count_reachable_from_0(ix, sc) < ix.n_bus) {
            ctg.disconnected = true;
        }
    }
}

// ---------------------------------------------------------------------------
// compute_component_masks
// ---------------------------------------------------------------------------
void compute_component_masks(
    std::vector<Contingency>&                                     contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
    const MaskConfig&                                             cfg)
{
    const ConnectivityIndex ix(Ybus_rm);
    ConnectivityScratch     sc(ix.n_bus, ix.nnz);
    const eigen_cplx_type*  values = Ybus_rm.valuePtr();
    const int               n_bus  = ix.n_bus;

    const int n_grp  = static_cast<int>(cfg.vc_grp_count.size());
    const int n_ctrl = static_cast<int>(cfg.vc_bus.size());
    std::vector<int>  n_masked_ctrl(static_cast<size_t>(n_grp), 0);
    std::vector<char> is_masked(static_cast<size_t>(n_bus), 0);

    for (auto& ctg : contingencies) {
        ctg.masked_buses.clear();
        ctg.stranded_groups.clear();
        if (ctg.skip) { ctg.disconnected = true; continue; }
        sc.collect_removed(ctg, values);

        if (!fast_masked_set(ix, sc, ctg.masked_buses)) {
            const int nb_comp = label_components(ix, sc);
            if (nb_comp <= 1) continue;   // still connected → nothing masked

            // Largest component by bus count (first one wins a tie).
            sc.count.assign(static_cast<size_t>(nb_comp), 0);
            for (int b = 0; b < n_bus; ++b) ++sc.count[static_cast<size_t>(sc.comp[static_cast<size_t>(b)])];
            const int main_comp = static_cast<int>(std::distance(
                sc.count.begin(), std::max_element(sc.count.begin(), sc.count.end())));
            for (int b = 0; b < n_bus; ++b)
                if (sc.comp[static_cast<size_t>(b)] != main_comp) ctg.masked_buses.push_back(b);
        }
        if (ctg.masked_buses.empty()) continue;   // graph untouched / still connected

        // Every bus outside the main component is masked. Stranding the angle
        // reference or a hard controller bus (HVDC end, regulated bus) has no
        // value-only fallback on the fixed batch structure → skip the
        // contingency (NaN), reusing the `disconnected` compaction path.
        bool skip = false;
        std::fill(is_masked.begin(), is_masked.end(), 0);
        for (int b : ctg.masked_buses) {
            is_masked[static_cast<size_t>(b)] = 1;
            if ((!cfg.is_reference_bus.empty()       && cfg.is_reference_bus[static_cast<size_t>(b)]) ||
                (!cfg.is_hard_controller_bus.empty() && cfg.is_hard_controller_bus[static_cast<size_t>(b)])) {
                skip = true;
                break;
            }
        }

        // VoltageControl groups: count the masked controllers of each group.
        //   count == 1, lone controller  → repurpose its voltage row (stranded)
        //   every controller masked      → skip (nobody left to hold the row)
        //   some of several masked       → nothing to do (sharing rows keep
        //                                  the masked column coupled)
        if (!skip && n_grp > 0) {
            std::fill(n_masked_ctrl.begin(), n_masked_ctrl.end(), 0);
            for (int j = 0; j < n_ctrl; ++j) {
                const int b = cfg.vc_bus[static_cast<size_t>(j)];
                if (b >= 0 && b < n_bus && is_masked[static_cast<size_t>(b)])
                    ++n_masked_ctrl[static_cast<size_t>(cfg.vc_group[static_cast<size_t>(j)])];
            }
            for (int g = 0; g < n_grp && !skip; ++g) {
                const int nm = n_masked_ctrl[static_cast<size_t>(g)];
                if (nm == 0) continue;
                const int cnt = cfg.vc_grp_count[static_cast<size_t>(g)];
                if (cnt == 1) {
                    // Needs the reserved (v_row, q_col) slot; without it the
                    // column would be structurally singular → skip (defensive:
                    // the CA/SS factories always reserve it).
                    if (cfg.vc_vrow_qcol_pos.empty() ||
                        cfg.vc_vrow_qcol_pos[static_cast<size_t>(g)] < 0)
                        skip = true;
                    else
                        ctg.stranded_groups.push_back(g);
                } else if (nm >= cnt) {
                    skip = true;
                }
            }
        }
        if (skip) {
            ctg.disconnected = true;
            ctg.masked_buses.clear();
            ctg.stranded_groups.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// build_mask_entries
// ---------------------------------------------------------------------------
void build_mask_entries(
    const std::vector<Contingency>& contingencies,
    const std::vector<int>&         active_to_orig,
    int                             batch_size,
    const MaskConfig&               cfg,
    MaskEntries&                    out)
{
    const MaskRowInfo& row_info = cfg.row_info;
    out = MaskEntries{};

    const int n_active = static_cast<int>(active_to_orig.size());
    const int n_chunks = batch_size > 0 ? (n_active + batch_size - 1) / batch_size : 0;
    out.row_ranges.assign(static_cast<size_t>(n_chunks), ChunkPatchRange{0, 0});
    out.v_ranges.assign(static_cast<size_t>(n_chunks), ChunkPatchRange{0, 0});
    out.jov_ranges.assign(static_cast<size_t>(n_chunks), ChunkPatchRange{0, 0});
    out.str_ranges.assign(static_cast<size_t>(n_chunks), ChunkPatchRange{0, 0});

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int a_start   = chunk * batch_size;
        const int a_end     = std::min(a_start + batch_size, n_active);
        const int row_start = static_cast<int>(out.slot.size());
        const int v_start   = static_cast<int>(out.v_slot.size());
        const int jov_start = static_cast<int>(out.jov_slot.size());
        const int str_start = static_cast<int>(out.str_slot.size());

        for (int local_c = 0; local_c < a_end - a_start; ++local_c) {
            const Contingency& ctg = contingencies[active_to_orig[a_start + local_c]];
            for (int bus : ctg.masked_buses) {
                // Identity-row entries for this bus' P and Q equations.
                if (row_info.p_row[static_cast<size_t>(bus)] >= 0) {
                    out.slot.push_back(local_c);
                    out.row.push_back(row_info.p_row[static_cast<size_t>(bus)]);
                    out.diag.push_back(row_info.p_diag_pos[static_cast<size_t>(bus)]);
                }
                if (row_info.q_row[static_cast<size_t>(bus)] >= 0) {
                    out.slot.push_back(local_c);
                    out.row.push_back(row_info.q_row[static_cast<size_t>(bus)]);
                    out.diag.push_back(row_info.q_diag_pos[static_cast<size_t>(bus)]);
                }
                // Masked-voltage entry (reported as NaN).
                out.v_slot.push_back(local_c);
                out.v_bus.push_back(bus);
            }
            // PV pins: ONLY the Q row goes to identity (dVm = 0); the P row and
            // the voltage stay live (lightsim2grid's set_pv_pinned_buses).
            for (int bus : ctg.pinned_buses) {
                if (row_info.q_row[static_cast<size_t>(bus)] >= 0) {
                    out.slot.push_back(local_c);
                    out.row.push_back(row_info.q_row[static_cast<size_t>(bus)]);
                    out.diag.push_back(row_info.q_diag_pos[static_cast<size_t>(bus)]);
                }
            }
            // Stranded lone controllers: repurpose the voltage row by value.
            for (int g : ctg.stranded_groups) {
                const int pq = cfg.vc_vrow_qcol_pos[static_cast<size_t>(g)];
                const int pv = cfg.vc_vrow_vmcol_pos[static_cast<size_t>(g)];
                if (pq >= 0) {
                    out.jov_slot.push_back(local_c);
                    out.jov_pos.push_back(pq);
                    out.jov_val.push_back(static_cast<cuda_real_type>(1.));
                }
                if (pv >= 0) {
                    out.jov_slot.push_back(local_c);
                    out.jov_pos.push_back(pv);
                    out.jov_val.push_back(static_cast<cuda_real_type>(0.));
                }
                out.str_slot.push_back(local_c);
                out.str_grp.push_back(g);
            }
        }

        out.row_ranges[static_cast<size_t>(chunk)] =
            {row_start, static_cast<int>(out.slot.size()) - row_start};
        out.v_ranges[static_cast<size_t>(chunk)] =
            {v_start, static_cast<int>(out.v_slot.size()) - v_start};
        out.jov_ranges[static_cast<size_t>(chunk)] =
            {jov_start, static_cast<int>(out.jov_slot.size()) - jov_start};
        out.str_ranges[static_cast<size_t>(chunk)] =
            {str_start, static_cast<int>(out.str_slot.size()) - str_start};
    }
}

// ---------------------------------------------------------------------------
// build_tripped_branch_table
// ---------------------------------------------------------------------------
void build_tripped_branch_table(
    const std::vector<Contingency>& contingencies,
    const std::vector<int>&         active_to_orig,
    std::vector<int>&               h_trip_branch_flat,
    std::vector<int>&               h_trip_start,
    std::vector<int>&               h_trip_count)
{
    const int n_active = static_cast<int>(active_to_orig.size());
    h_trip_start.resize(static_cast<size_t>(n_active));
    h_trip_count.resize(static_cast<size_t>(n_active));
    h_trip_branch_flat.clear();

    for (int slot = 0; slot < n_active; ++slot) {
        const auto& tb = contingencies[static_cast<size_t>(active_to_orig[slot])].tripped_branches;
        h_trip_start[static_cast<size_t>(slot)] = static_cast<int>(h_trip_branch_flat.size());
        h_trip_count[static_cast<size_t>(slot)] = static_cast<int>(tb.size());
        h_trip_branch_flat.insert(h_trip_branch_flat.end(), tb.begin(), tb.end());
    }
}
