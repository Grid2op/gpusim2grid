// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_helper.cpp
// =============================================================================

#include "contingency_analysis_helper.hpp"
#include "ledger_data.hpp"   // LedgerData (choose_reference_bus_from_ledger)

#include <algorithm>        // std::lower_bound, std::min, std::sort, std::max_element
#include <cmath>            // std::abs
#include <cstdint>
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

    // a branch named twice is tripped once (lightsim2grid's contingency is a set): its
    // admittance subtracted twice would leave -y in Ybus and the edge in the graph
    std::vector<int> unique_ids;
    unique_ids.reserve(branch_ids.size());
    for (int l : branch_ids)
        if (std::find(unique_ids.begin(), unique_ids.end(), l) == unique_ids.end()) unique_ids.push_back(l);

    Contingency ctg;
    ctg.tripped_branches = unique_ids;
    ctg.triplets.reserve(unique_ids.size() * 4);

    for (int l : unique_ids) {
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
// append_one_side_trips
// ---------------------------------------------------------------------------
void append_one_side_trips(
    Contingency&              ctg,
    const std::vector<int>&   branch_ids,
    int                       open_side,
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect> yff_eff,
    Eigen::Ref<const CplxVect> yft_eff,
    Eigen::Ref<const CplxVect> ytf_eff,
    Eigen::Ref<const CplxVect> ytt_eff)
{
    if (open_side != 1 && open_side != 2)
        throw std::runtime_error("append_one_side_trips: the side opened is 1 or 2");
    const int n_branches = static_cast<int>(branch_from.size());
    for (int l : branch_ids) {
        if (l < 0 || l >= n_branches)
            throw std::runtime_error("append_one_side_trips: branch index out of range");
        ctg.tripped_branches.push_back(l);
        const int b_open = open_side == 1 ? branch_from(l) : branch_to(l);
        const int b_kept = open_side == 1 ? branch_to(l) : branch_from(l);
        if (b_open < 0) continue;   // already open at that end
        if (b_kept < 0) {
            // open at its other end in the base grid: it goes entirely (its stamped
            // block is the open end's Kron-reduced diagonal)
            const auto y = open_side == 1 ? yff_eff(l) : ytt_eff(l);
            ctg.triplets.push_back({b_open, b_open, y.real(), y.imag()});
            continue;
        }
        // both ends closed: the stamped block is the raw one
        const auto y_oo = open_side == 1 ? yff_eff(l) : ytt_eff(l);
        const auto y_ok = open_side == 1 ? yft_eff(l) : ytf_eff(l);
        const auto y_ko = open_side == 1 ? ytf_eff(l) : yft_eff(l);
        if (std::abs(y_oo) == 0.) continue;   // off in the base grid: nothing stamped
        // y_kk - (y_kk - y_ko * y_ok / y_oo): the kept end keeps its Kron-reduced part
        const auto y_kept = y_ko * y_ok / y_oo;
        ctg.triplets.push_back({b_open, b_open, y_oo.real(), y_oo.imag()});
        ctg.triplets.push_back({b_kept, b_kept, y_kept.real(), y_kept.imag()});
        ctg.triplets.push_back({b_open, b_kept, y_ok.real(), y_ok.imag()});
        ctg.triplets.push_back({b_kept, b_open, y_ko.real(), y_ko.imag()});
    }
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

// |z| < kRemovedEdgeEps, on the squared modulus: std::abs of a complex is a hypot call, and
// the per-row searches below test every entry of the matrix
inline bool removed_edge(const eigen_cplx_type& z)
{
    return std::norm(z) < kRemovedEdgeEps * kRemovedEdgeEps;
}

struct ConnectivityIndex {
    int        n_bus = 0;
    int        nnz   = 0;
    const int* outer = nullptr;   // Ybus_rm CSR row pointers
    const int* inner = nullptr;   // Ybus_rm CSR column indices

    std::vector<char> live;        // [nnz]   entry is an edge of the graph (see the ctor)
    std::vector<int>  row_of;      // [nnz]   row i of flat entry k (-1 outside the graph)
    std::vector<int>  k_rev;       // [nnz]   flat index of (j,i) for entry (i,j); -1 if absent / diagonal
    std::vector<char> is_tree;     // [nnz]   entry is a DFS-tree edge (both directions)
    std::vector<char> is_bridge;   // [nnz]   entry is a bridge (both directions)
    std::vector<int>  tree_child;  // [nnz]   deeper endpoint of a tree edge, -1 otherwise
    std::vector<int>  tin, tout;   // [n_bus] preorder index / end of subtree (exclusive)
    std::vector<int>  preorder;    // [n_bus] bus at each preorder position
    bool              base_connected = false;
    bool              skip_zero      = false;   // zero entries are no edges (the ctor's flag)
    // Cycle-space labels (certify_removal): per edge (both directions), a random 64-bit word for
    // a non-tree edge, and for a tree edge the XOR of the non-tree edges leaving the subtree below
    // it. An edge cut XORs to 0, so removed edges with independent labels hold no cut.
    // labels_ok: the graph is connected and every edge has both directions.
    std::vector<uint64_t> label;   // [nnz]
    bool              labels_ok = false;

    // The graph: the first n_graph buses (the others are spare buses of the union pattern,
    // see check_connectivity), an edge per off-diagonal entry between two of them -- with
    // skip_zero, only the nonzero ones (the union's entries are explicit zeros in the base
    // case). Without spare buses (n_graph = all, skip_zero false): every off-diagonal entry
    // of the pattern, today's graph.
    ConnectivityIndex(const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                      int n_graph, bool skip_zero)
        : n_bus(n_graph)
        , nnz(static_cast<int>(Ybus_rm.nonZeros()))
        , outer(Ybus_rm.outerIndexPtr())
        , inner(Ybus_rm.innerIndexPtr())
        , live(static_cast<size_t>(nnz), 0)
        , row_of(static_cast<size_t>(nnz), -1)
        , k_rev(static_cast<size_t>(nnz), -1)
        , is_tree(static_cast<size_t>(nnz), 0)
        , is_bridge(static_cast<size_t>(nnz), 0)
        , tree_child(static_cast<size_t>(nnz), -1)
        , tin(static_cast<size_t>(n_bus), -1)
        , tout(static_cast<size_t>(n_bus), -1)
        , preorder(static_cast<size_t>(n_bus), -1)
        , skip_zero(skip_zero)
    {
        const eigen_cplx_type* values = Ybus_rm.valuePtr();
        for (int i = 0; i < n_bus; ++i)
            for (int p = outer[i]; p < outer[i + 1]; ++p) {
                row_of[static_cast<size_t>(p)] = i;
                const int j = inner[p];
                if (j == i || j >= n_bus) continue;
                if (skip_zero && removed_edge(values[p])) continue;
                live[static_cast<size_t>(p)] = 1;
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
                if (!live[static_cast<size_t>(k)]) continue;         // diagonal / not an edge
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
        if (base_connected) set_labels(parent_k);
    }

    void set_labels(const std::vector<int>& parent_k)
    {
        // splitmix64's finalizer: a fixed word per entry, nonzero but for one input in 2^64
        const auto mix = [](uint64_t z) {
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            return z ^ (z >> 31);
        };
        label.assign(static_cast<size_t>(nnz), 0);
        std::vector<uint64_t> sub(static_cast<size_t>(n_bus), 0);   // XOR of the non-tree edges at a bus
        for (int i = 0; i < n_bus; ++i)
            for (int p = outer[i]; p < outer[i + 1]; ++p) {
                if (!live[static_cast<size_t>(p)]) continue;
                const int kr = k_rev[static_cast<size_t>(p)];
                if (kr < 0 || !live[static_cast<size_t>(kr)]) return;   // a one-way edge: no labels
                const int j = inner[p];
                if (j < i || is_tree[static_cast<size_t>(p)]) continue;
                const uint64_t w = mix(0x9e3779b97f4a7c15ull * (static_cast<uint64_t>(p) + 1));
                label[static_cast<size_t>(p)] = label[static_cast<size_t>(kr)] = w;
                sub[static_cast<size_t>(i)] ^= w;
                sub[static_cast<size_t>(j)] ^= w;
            }
        // children before parents: the XOR over a subtree is what leaves it (inner edges cancel)
        for (int pos = n_bus - 1; pos >= 1; --pos) {
            const int v = preorder[static_cast<size_t>(pos)];
            const int pk = parent_k[static_cast<size_t>(v)];
            label[static_cast<size_t>(pk)] = label[static_cast<size_t>(k_rev[static_cast<size_t>(pk)])] = sub[static_cast<size_t>(v)];
            sub[static_cast<size_t>(row_of[static_cast<size_t>(pk)])] ^= sub[static_cast<size_t>(v)];
        }
        labels_ok = true;
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
            if (removed_edge(new_val)) removed.push_back(acc_k[idx]);
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
                if (!ix.live[static_cast<size_t>(p)] || ix.is_tree[static_cast<size_t>(p)]
                        || sc.removed_mark[static_cast<size_t>(p)]) continue;
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
                if (!ix.live[static_cast<size_t>(p)] || sc.removed_mark[static_cast<size_t>(p)]) continue;
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
            if (!ix.live[static_cast<size_t>(p)] || sc.removed_mark[static_cast<size_t>(p)]) continue;
            if (sc.comp[static_cast<size_t>(v)] == -1) {
                sc.comp[static_cast<size_t>(v)] = 0;
                sc.queue[tail++] = v;
            }
        }
    }
    for (int k : sc.removed) sc.removed_mark[static_cast<size_t>(k)] = 0;
    return tail;
}

// ---- topology actions: rows with bus moves on the union pattern -------------
// One merged Ybus change of a row: flat index, row, col, summed delta (subtracted).
struct MergedDelta {
    int             k;
    int             row;
    int             col;
    eigen_cplx_type delta;
};

void merged_deltas(const Contingency& ctg, std::vector<MergedDelta>& out)
{
    out.clear();
    for (const auto& t : ctg.triplets) {
        auto it = std::find_if(out.begin(), out.end(), [&](const MergedDelta& m) { return m.k == t.k; });
        if (it == out.end()) out.push_back({t.k, t.row, t.col, eigen_cplx_type(t.delta_re, t.delta_im)});
        else it->delta += eigen_cplx_type(t.delta_re, t.delta_im);
    }
}

// A row the graph of the first n_graph buses cannot answer: it touches a spare bus (a
// triplet or an injection), empties a bus of the graph, or adds a stamp where the base
// value is 0 (a merge onto a bus the branch did not reach).
bool row_has_moves(const Contingency& ctg, const eigen_cplx_type* values, int n_graph,
                   const std::vector<MergedDelta>& acc)
{
    for (int b : ctg.injection_buses) if (b >= n_graph) return true;
    for (int b : ctg.empty_buses) if (b < n_graph) return true;
    for (const auto& m : acc) {
        if (m.row >= n_graph || m.col >= n_graph) return true;
        if (m.row != m.col && removed_edge(values[m.k]) && !removed_edge(values[m.k] - m.delta))
            return true;
    }
    return false;
}

// The row's empty buses: the given ones and every spare bus (>= n_graph) no triplet and no
// injection of the row touches. Sorted, unique.
std::vector<int> effective_empty(const Contingency& ctg, int n_graph, int n_total)
{
    std::vector<int> out = ctg.empty_buses;
    if (n_total > n_graph) {
        std::vector<char> used(static_cast<size_t>(n_total - n_graph), 0);
        const auto use = [&](int b) { if (b >= n_graph && b < n_total) used[static_cast<size_t>(b - n_graph)] = 1; };
        for (const auto& t : ctg.triplets) { use(t.row); use(t.col); }
        for (int b : ctg.injection_buses) use(b);
        for (int s = n_graph; s < n_total; ++s)
            if (!used[static_cast<size_t>(s - n_graph)]) out.push_back(s);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// Components of a row on its own edge set: the nonzero off-diagonal entries of its Ybus
// (the base values, minus its merged deltas where it has some), its empty buses left out
// (label -1). BFS from every unlabelled bus in increasing order; returns the number of
// components. `delta_at`: scratch, one entry per Ybus entry, all -1 (left so on return).
int label_row_components(const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
                         const std::vector<MergedDelta>& acc, const std::vector<int>& empty,
                         std::vector<int>& comp, std::vector<int>& queue, std::vector<int>& delta_at)
{
    const int n_total = static_cast<int>(Ybus_rm.rows());
    const int* outer = Ybus_rm.outerIndexPtr();
    const int* inner = Ybus_rm.innerIndexPtr();
    const eigen_cplx_type* values = Ybus_rm.valuePtr();
    // the row's changes by flat index (merged_deltas: one per entry), looked up in O(1): every
    // entry of the matrix is visited, once per moved row
    delta_at.resize(static_cast<size_t>(Ybus_rm.nonZeros()), -1);
    for (size_t i = 0; i < acc.size(); ++i) delta_at[static_cast<size_t>(acc[i].k)] = static_cast<int>(i);
    const auto value_of = [&](int k) {
        const int i = delta_at[static_cast<size_t>(k)];
        return i >= 0 ? values[k] - acc[static_cast<size_t>(i)].delta : values[k];
    };
    constexpr int kEmpty = -2;
    comp.assign(static_cast<size_t>(n_total), -1);
    queue.resize(static_cast<size_t>(n_total));
    for (int b : empty) if (b >= 0 && b < n_total) comp[static_cast<size_t>(b)] = kEmpty;
    int nb_comp = 0;
    for (int start = 0; start < n_total; ++start) {
        if (comp[static_cast<size_t>(start)] != -1) continue;
        comp[static_cast<size_t>(start)] = nb_comp;
        int head = 0, tail = 0;
        queue[static_cast<size_t>(tail++)] = start;
        while (head < tail) {
            const int u = queue[static_cast<size_t>(head++)];
            for (int p = outer[u]; p < outer[u + 1]; ++p) {
                const int v = inner[p];
                if (v == u || comp[static_cast<size_t>(v)] != -1) continue;
                if (removed_edge(value_of(p))) continue;
                comp[static_cast<size_t>(v)] = nb_comp;
                queue[static_cast<size_t>(tail++)] = v;
            }
        }
        ++nb_comp;
    }
    for (int& c : comp) if (c == kEmpty) c = -1;
    for (const auto& m : acc) delta_at[static_cast<size_t>(m.k)] = -1;
    return nb_comp;
}

// ---- cycle-space certificate: "still connected" without a search -------------
// A row's removed edges (and the buses it empties, all of whose edges go with them) split the
// graph into c components, which gives c - 1 independent cuts inside the removed set, each with
// labels XOR-ing to 0: the removed labels have at least c - 1 dependencies, whatever the random
// words. So `n_isolated` emptied buses and exactly `n_isolated` dependencies mean c = n_isolated
// + 1: the rest is one piece. Exact when it answers; more dependencies (an island, or a chance
// collision of the words) leave the answer to the search. O(removed edges) per row, where the
// search visits the whole grid: most rows of a batch island nothing.
struct CertScratch {
    std::vector<int> edges;      // flat indices of the removed edges, either direction
    std::vector<int> uf;         // union-find over {the base graph, the spare buses}
    std::vector<int> delta_at;   // [nnz] the row's delta at each entry (index in acc), else -1
};

bool removed_labels_have(const ConnectivityIndex& ix, std::vector<int>& edges, int n_dependencies)
{
    for (int& k : edges) k = std::min(k, ix.k_rev[static_cast<size_t>(k)]);   // one per edge
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    uint64_t basis[64] = {};
    int dependencies = 0;
    for (int k : edges) {
        uint64_t x = ix.label[static_cast<size_t>(k)];
        while (x != 0) {
            int h = 63;
            while (!((x >> h) & 1ull)) --h;
            if (basis[h] == 0) { basis[h] = x; break; }
            x ^= basis[h];
        }
        if (x == 0 && ++dependencies > n_dependencies) return false;
    }
    return dependencies == n_dependencies;
}

// A row without moves (its removed entries `sc.removed`): certainly still connected.
bool certify_removal(const ConnectivityIndex& ix, ConnectivityScratch& sc, CertScratch& cs)
{
    if (!ix.labels_ok) return false;
    cs.edges.clear();
    for (int k : sc.removed) {
        if (!ix.live[static_cast<size_t>(k)]) continue;   // no edge to remove
        if (!sc.is_removed(ix.k_rev[static_cast<size_t>(k)])) return false;   // one way only
        cs.edges.push_back(k);
    }
    return removed_labels_have(ix, cs.edges, 0);
}

// A row with moves (bus_moves: its merged deltas `acc`, its empty buses `empty`, sorted):
// certainly one piece once its empty buses are left out. Its base graph minus the edges it
// removes and the buses it empties must be one piece (the labels), and every spare bus it uses
// must reach the base graph through edges it stamps (both ways nonzero). An edge it adds between
// base buses only joins more: not needed. `cs.delta_at` is filled with the row's deltas by the
// caller (certify_moves).
bool certify_moves_filled(const ConnectivityIndex& ix, const eigen_cplx_type* values, int n_graph, int n_total,
                          const std::vector<MergedDelta>& acc, const std::vector<int>& empty, CertScratch& cs)
{
    const auto is_empty = [&](int b) { return std::binary_search(empty.begin(), empty.end(), b); };
    // the row's value of the entry (row, col): its base value minus its delta there, 0 off the pattern
    const auto value_at = [&](int row, int col) {
        const int* b = ix.inner + ix.outer[row];
        const int* e = ix.inner + ix.outer[row + 1];
        const int* it = std::lower_bound(b, e, col);
        if (it == e || *it != col) return eigen_cplx_type(0);
        const int k = static_cast<int>(it - ix.inner);
        const int i = cs.delta_at[static_cast<size_t>(k)];
        return i >= 0 ? values[k] - acc[static_cast<size_t>(i)].delta : values[k];
    };
    const auto new_value = [&](const MergedDelta& m) { return values[m.k] - m.delta; };
    cs.edges.clear();
    for (const auto& m : acc) {
        if (m.row == m.col || m.row >= n_graph || m.col >= n_graph || !ix.live[static_cast<size_t>(m.k)]) continue;
        if (!removed_edge(new_value(m))) continue;
        if (!removed_edge(value_at(m.col, m.row))) return false;   // one way only
        cs.edges.push_back(m.k);
    }
    int n_isolated = 0;
    for (int b : empty) {
        if (b >= n_graph) continue;
        ++n_isolated;
        for (int p = ix.outer[b]; p < ix.outer[b + 1]; ++p)
            if (ix.live[static_cast<size_t>(p)]) cs.edges.push_back(p);
    }
    if (!removed_labels_have(ix, cs.edges, n_isolated)) return false;
    if (n_total == n_graph) return true;
    // the spare buses it uses: joined to the base graph (node 0) by its stamps
    cs.uf.resize(static_cast<size_t>(n_total - n_graph + 1));
    for (size_t i = 0; i < cs.uf.size(); ++i) cs.uf[i] = static_cast<int>(i);
    const auto node = [n_graph](int b) { return b < n_graph ? 0 : b - n_graph + 1; };
    const auto root = [&](int a) {
        while (cs.uf[static_cast<size_t>(a)] != a) a = cs.uf[static_cast<size_t>(a)] = cs.uf[static_cast<size_t>(cs.uf[static_cast<size_t>(a)])];
        return a;
    };
    for (const auto& m : acc) {
        if (m.row == m.col || (m.row < n_graph && m.col < n_graph)) continue;
        if (is_empty(m.row) || is_empty(m.col) || removed_edge(new_value(m))) continue;
        if (removed_edge(value_at(m.col, m.row))) continue;   // not an edge both ways
        const int a = root(node(m.row)), c = root(node(m.col));
        if (a != c) cs.uf[static_cast<size_t>(std::max(a, c))] = std::min(a, c);
    }
    for (int sp = n_graph; sp < n_total; ++sp)
        if (!is_empty(sp) && root(node(sp)) != 0) return false;
    return true;
}

// certify_moves_filled with the row's deltas indexed by entry (merged_deltas: one per entry), so
// the reverse of an entry is found in O(log degree), not by a scan of the row's deltas
bool certify_moves(const ConnectivityIndex& ix, const eigen_cplx_type* values, int n_graph, int n_total,
                   const std::vector<MergedDelta>& acc, const std::vector<int>& empty, CertScratch& cs)
{
    if (!ix.labels_ok || !ix.skip_zero) return false;   // (the search reads values: zeros no edges)
    cs.delta_at.resize(static_cast<size_t>(ix.nnz), -1);
    for (size_t i = 0; i < acc.size(); ++i) cs.delta_at[static_cast<size_t>(acc[i].k)] = static_cast<int>(i);
    const bool ok = certify_moves_filled(ix, values, n_graph, n_total, acc, empty, cs);
    for (const auto& m : acc) cs.delta_at[static_cast<size_t>(m.k)] = -1;
    return ok;
}

}  // namespace

// ---------------------------------------------------------------------------
// check_connectivity
// ---------------------------------------------------------------------------
void check_connectivity(
    std::vector<Contingency>&                                     contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
    int                                                           n_graph,
    bool                                                          union_pattern,
    ConnectivityStats*                                            stats)
{
    const int n_total = static_cast<int>(Ybus_rm.rows());
    if (n_graph < 0 || n_graph > n_total) n_graph = n_total;
    const ConnectivityIndex ix(Ybus_rm, n_graph, /*skip_zero=*/union_pattern || n_graph < n_total);
    ConnectivityScratch     sc(ix.n_bus, ix.nnz);
    const eigen_cplx_type*  values = Ybus_rm.valuePtr();
    std::vector<int>        masked;
    std::vector<MergedDelta> acc;
    std::vector<int>        comp, queue, delta_at;
    CertScratch             cs;

    for (auto& ctg : contingencies) {
        ctg.empty_effective = effective_empty(ctg, n_graph, n_total);
        if (ctg.skip) { ctg.disconnected = true; continue; }
        merged_deltas(ctg, acc);
        if (row_has_moves(ctg, values, n_graph, acc)) {
            // bus moves: the row's own graph; its empty buses are no island
            if (certify_moves(ix, values, n_graph, n_total, acc, ctg.empty_effective, cs)) {
                if (stats) ++stats->certified_moves;
                continue;
            }
            if (label_row_components(Ybus_rm, acc, ctg.empty_effective, comp, queue, delta_at) > 1)
                ctg.disconnected = true;
            continue;
        }
        sc.collect_removed(ctg, values);
        if (certify_removal(ix, sc, cs)) {
            if (stats) ++stats->certified_removal;
            continue;
        }
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
    const MaskConfig&                                             cfg,
    int                                                           n_graph,
    bool                                                          union_pattern,
    ConnectivityStats*                                            stats)
{
    const int n_total = static_cast<int>(Ybus_rm.rows());
    if (n_graph < 0 || n_graph > n_total) n_graph = n_total;
    const ConnectivityIndex ix(Ybus_rm, n_graph, /*skip_zero=*/union_pattern || n_graph < n_total);
    ConnectivityScratch     sc(ix.n_bus, ix.nnz);
    const eigen_cplx_type*  values = Ybus_rm.valuePtr();
    const int               n_bus  = n_total;   // masked buses may be spare ones (a moved island)
    std::vector<MergedDelta> acc;
    std::vector<int>        comp, queue, delta_at;
    CertScratch             cs;

    const int n_grp  = static_cast<int>(cfg.vc_grp_count.size());
    const int n_ctrl = static_cast<int>(cfg.vc_bus.size());
    std::vector<int>  n_masked_ctrl(static_cast<size_t>(n_grp), 0);
    std::vector<int>  n_active_ctrl(static_cast<size_t>(n_grp), 0);
    std::vector<char> pinned_ctrl(static_cast<size_t>(n_ctrl), 0);
    std::vector<char> is_masked(static_cast<size_t>(n_bus), 0);

    for (auto& ctg : contingencies) {
        ctg.masked_buses.clear();
        ctg.stranded_groups.clear();
        ctg.empty_effective = effective_empty(ctg, n_graph, n_total);
        if (ctg.skip) { ctg.disconnected = true; continue; }
        merged_deltas(ctg, acc);

        if (row_has_moves(ctg, values, n_graph, acc)) {
            // bus moves: the row's own graph, its empty buses left out (masked anyway)
            if (certify_moves(ix, values, n_graph, n_total, acc, ctg.empty_effective, cs)) {
                if (stats) ++stats->certified_moves;
                continue;
            }
            const int nb_comp = label_row_components(Ybus_rm, acc, ctg.empty_effective, comp, queue, delta_at);
            if (nb_comp <= 1) continue;
            std::vector<int> count(static_cast<size_t>(nb_comp), 0);
            for (int c : comp) if (c >= 0) ++count[static_cast<size_t>(c)];
            const int main_comp = static_cast<int>(std::distance(
                count.begin(), std::max_element(count.begin(), count.end())));
            for (int b = 0; b < n_total; ++b)
                if (comp[static_cast<size_t>(b)] >= 0 && comp[static_cast<size_t>(b)] != main_comp)
                    ctg.masked_buses.push_back(b);
        } else {
            sc.collect_removed(ctg, values);
            if (certify_removal(ix, sc, cs)) {   // connected: nothing masked
                if (stats) ++stats->certified_removal;
                continue;
            }
            if (!fast_masked_set(ix, sc, ctg.masked_buses)) {
                const int nb_comp = label_components(ix, sc);
                if (nb_comp <= 1) continue;   // still connected → nothing masked

                // Largest component by bus count (first one wins a tie).
                sc.count.assign(static_cast<size_t>(nb_comp), 0);
                for (int b = 0; b < ix.n_bus; ++b) ++sc.count[static_cast<size_t>(sc.comp[static_cast<size_t>(b)])];
                const int main_comp = static_cast<int>(std::distance(
                    sc.count.begin(), std::max_element(sc.count.begin(), sc.count.end())));
                for (int b = 0; b < ix.n_bus; ++b)
                    if (sc.comp[static_cast<size_t>(b)] != main_comp) ctg.masked_buses.push_back(b);
            }
        }
        if (ctg.masked_buses.empty()) continue;   // graph untouched / still connected

        // Every bus outside the main component is masked. Stranding the angle
        // reference or a hard controller bus (HVDC end) has no value-only
        // fallback on the fixed batch structure → skip the contingency (NaN),
        // reusing the `disconnected` compaction path.
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
        //   every controller masked         → repurpose its voltage row into
        //                                     Q_first == 0 (stranded), whether
        //                                     the regulated bus is masked or not
        //   regulated bus masked, a live one → skip (the frozen magnitude of a
        //                                     masked bus cannot reach v_set)
        //   some of several masked          → nothing to do (the live ones hold
        //                                     the bus, the sharing rows keep
        //                                     the masked columns coupled)
        // A controller the row holds at a fixed Q (Contingency::vc_pinned_ctrl, and
        // a held one -- lightsim2grid's set_hold_frozen_regulators -- the row does
        // not release) holds no bus: only the free ones count, and a group with
        // none is neither stranded nor a reason to skip.
        if (!skip && n_grp > 0) {
            std::fill(n_masked_ctrl.begin(), n_masked_ctrl.end(), 0);
            std::fill(n_active_ctrl.begin(), n_active_ctrl.end(), 0);
            std::fill(pinned_ctrl.begin(), pinned_ctrl.end(), 0);
            if (!cfg.vc_held.empty()) {
                for (int j = 0; j < n_ctrl; ++j)
                    pinned_ctrl[static_cast<size_t>(j)] = cfg.vc_held[static_cast<size_t>(j)] ? 1 : 0;
                for (int j : ctg.vc_released)
                    if (j >= 0 && j < n_ctrl) pinned_ctrl[static_cast<size_t>(j)] = 0;
            }
            for (const auto& pc : ctg.vc_pinned_ctrl)
                if (pc.first >= 0 && pc.first < n_ctrl) pinned_ctrl[static_cast<size_t>(pc.first)] = 1;
            for (int j : ctg.vc_tripped)
                if (j >= 0 && j < n_ctrl) pinned_ctrl[static_cast<size_t>(j)] = 1;
            for (int j = 0; j < n_ctrl; ++j) {
                if (pinned_ctrl[static_cast<size_t>(j)]) continue;
                const int grp = cfg.vc_group[static_cast<size_t>(j)];
                ++n_active_ctrl[static_cast<size_t>(grp)];
                const int b = cfg.vc_bus[static_cast<size_t>(j)];
                if (b >= 0 && b < n_bus && is_masked[static_cast<size_t>(b)])
                    ++n_masked_ctrl[static_cast<size_t>(grp)];
            }
            for (int g = 0; g < n_grp && !skip; ++g) {
                const int nm  = n_masked_ctrl[static_cast<size_t>(g)];
                const int cnt = n_active_ctrl[static_cast<size_t>(g)];
                const int reg = cfg.vc_reg_bus.empty() ? -1 : cfg.vc_reg_bus[static_cast<size_t>(g)];
                const bool reg_masked = reg >= 0 && reg < n_bus && is_masked[static_cast<size_t>(reg)];
                if (cnt > 0 && nm >= cnt) {
                    // Needs the reserved (v_row, q_col_first) slot; without it
                    // the column would be structurally singular → skip
                    // (defensive: the CA/SS factories always reserve it).
                    if (cfg.vc_vrow_qcol_pos.empty() ||
                        cfg.vc_vrow_qcol_pos[static_cast<size_t>(g)] < 0)
                        skip = true;
                    else
                        ctg.stranded_groups.push_back(g);
                } else if (reg_masked && cnt > nm) {
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
// choose_reference_bus
// ---------------------------------------------------------------------------
int choose_reference_bus(
    const std::vector<Contingency>&                               contingencies,
    const int*                                                    Ybus_rm_outer,
    const int*                                                    Ybus_rm_inner,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
    const std::vector<int>&                                       cand_bus,
    const std::vector<double>&                                    cand_w,
    int                                                           forced)
{
    std::vector<int> cands;
    for (size_t k = 0; k < cand_bus.size(); ++k)
        if (k < cand_w.size() && cand_w[k] > 0.) cands.push_back(static_cast<int>(k));
    if (cands.empty())
        for (size_t k = 0; k < cand_bus.size(); ++k) cands.push_back(static_cast<int>(k));
    if (cands.empty()) return -1;
    if (forced >= 0)
        for (int k : cands)
            if (cand_bus[static_cast<size_t>(k)] == forced) return forced;

    // the plain split of every row: no reference / controller / skip rule
    std::vector<Contingency> ctgs;
    ctgs.reserve(contingencies.size());
    for (const auto& c : contingencies) {
        Contingency x;
        x.triplets = c.triplets;
        ctgs.push_back(std::move(x));
    }
    resolve_indices(ctgs, Ybus_rm_outer, Ybus_rm_inner);
    const MaskConfig plain;
    compute_component_masks(ctgs, Ybus_rm, plain);

    const int n_bus = static_cast<int>(Ybus_rm.rows());
    std::vector<int> strand(static_cast<size_t>(n_bus), 0);
    for (const auto& c : ctgs)
        for (int b : c.masked_buses)
            if (b >= 0 && b < n_bus) ++strand[static_cast<size_t>(b)];

    int best = -1, best_strand = -1;
    double best_w = -1.;
    for (int k : cands) {
        const int b = cand_bus[static_cast<size_t>(k)];
        const int s = (b >= 0 && b < n_bus) ? strand[static_cast<size_t>(b)] : 0;
        const double w = cand_w[static_cast<size_t>(k)];
        const bool better = best < 0 || s < best_strand ||
                            (s == best_strand && w > best_w) ||
                            (s == best_strand && w == best_w && b < best);
        if (better) { best = b; best_strand = s; best_w = w; }
    }
    return best;
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
    out.vcp_ranges.assign(static_cast<size_t>(n_chunks), ChunkPatchRange{0, 0});

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int a_start   = chunk * batch_size;
        const int a_end     = std::min(a_start + batch_size, n_active);
        const int row_start = static_cast<int>(out.slot.size());
        const int v_start   = static_cast<int>(out.v_slot.size());
        const int jov_start = static_cast<int>(out.jov_slot.size());
        const int str_start = static_cast<int>(out.str_slot.size());
        const int vcp_start = static_cast<int>(out.vcp_slot.size());

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
            // Stranded groups: repurpose the voltage row by value into
            // "Q_first == 0" -- 1 on the first controller's column, 0 on the
            // regulated magnitude and on the other controllers' slope slots
            // (an SVC's; a generator has none -- and lightsim2grid v1 refuses an
            // SVC sharing its regulated bus, so that last loop mirrors upstream's
            // generic rule without a case that reaches it today). The sharing
            // rows are untouched.
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
                if (!cfg.vc_grp_start.empty() && !cfg.vc_ctrl_vrow_qcol_pos.empty()) {
                    const int first = cfg.vc_grp_start[static_cast<size_t>(g)];
                    const int cnt   = cfg.vc_grp_count[static_cast<size_t>(g)];
                    for (int j = first + 1; j < first + cnt; ++j) {
                        const int ps = cfg.vc_ctrl_vrow_qcol_pos[static_cast<size_t>(j)];
                        if (ps < 0) continue;
                        out.jov_slot.push_back(local_c);
                        out.jov_pos.push_back(ps);
                        out.jov_val.push_back(static_cast<cuda_real_type>(0.));
                    }
                }
                out.str_slot.push_back(local_c);
                out.str_grp.push_back(g);
            }
            // Controllers held at a fixed reactive injection (the reactive-limit
            // outer loop): a group's first controller through its voltage row
            // ("Q_first = t": 1 on its q_col, 0 on the regulated magnitude and on
            // the other controllers' slope slots -- the stranded-group rewrite
            // with a target), any other one through its sharing row ("Q_c = t":
            // 1 on its own q_col, 0 on the first controller's). The regulated
            // bus floats once every controller of the group is held.
            auto jov = [&](int pos, double val) {
                if (pos < 0) return;
                out.jov_slot.push_back(local_c);
                out.jov_pos.push_back(pos);
                out.jov_val.push_back(static_cast<cuda_real_type>(val));
            };
            // (the groups the row loses a controller of hold their pins by value, SlotVcHold)
            std::vector<char> held_by_value;
            if (cfg.vc_hold_by_value && !ctg.vc_pinned_ctrl.empty()) {
                held_by_value.assign(cfg.vc_grp_start.size(), 0);
                for (int j : ctg.vc_tripped)
                    if (j >= 0 && j < static_cast<int>(cfg.vc_group.size()))
                        held_by_value[static_cast<size_t>(cfg.vc_group[static_cast<size_t>(j)])] = 1;
                for (size_t j = 0; j < cfg.vc_bus.size(); ++j)
                    if (std::find(ctg.masked_buses.begin(), ctg.masked_buses.end(), cfg.vc_bus[j]) !=
                        ctg.masked_buses.end())
                        held_by_value[static_cast<size_t>(cfg.vc_group[j])] = 1;
            }
            for (const auto& pc : ctg.vc_pinned_ctrl) {
                const int j = pc.first;
                const int g = cfg.vc_group[static_cast<size_t>(j)];
                if (!held_by_value.empty() && held_by_value[static_cast<size_t>(g)]) continue;
                const int first = cfg.vc_grp_start[static_cast<size_t>(g)];
                int row;
                if (j == first) {
                    row = cfg.vc_vrow[static_cast<size_t>(g)];
                    jov(cfg.vc_vrow_qcol_pos[static_cast<size_t>(g)], 1.);
                    jov(cfg.vc_vrow_vmcol_pos[static_cast<size_t>(g)], 0.);
                    const int cnt = cfg.vc_grp_count[static_cast<size_t>(g)];
                    for (int k = first + 1; k < first + cnt; ++k)
                        jov(cfg.vc_ctrl_vrow_qcol_pos[static_cast<size_t>(k)], 0.);
                } else {
                    row = cfg.vc_ctrl_sh_row[static_cast<size_t>(j)];
                    jov(cfg.vc_ctrl_sh_self_pos[static_cast<size_t>(j)], 1.);
                    jov(cfg.vc_ctrl_sh_first_pos[static_cast<size_t>(j)], 0.);
                }
                out.vcp_slot.push_back(local_c);
                out.vcp_row.push_back(row);
                out.vcp_ctrl.push_back(j);
                out.vcp_target.push_back(static_cast<cuda_real_type>(pc.second));
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
        out.vcp_ranges[static_cast<size_t>(chunk)] =
            {vcp_start, static_cast<int>(out.vcp_slot.size()) - vcp_start};
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
    std::vector<int>&               h_trip_count,
    std::vector<int>*               h_trip_own,
    std::vector<RowBranchBlock>*    h_own)
{
    const int n_active = static_cast<int>(active_to_orig.size());
    h_trip_start.resize(static_cast<size_t>(n_active));
    h_trip_count.resize(static_cast<size_t>(n_active));
    h_trip_branch_flat.clear();
    const bool own = h_trip_own != nullptr && h_own != nullptr;
    if (own) { h_trip_own->clear(); h_own->clear(); }

    for (int slot = 0; slot < n_active; ++slot) {
        const Contingency& ctg = contingencies[static_cast<size_t>(active_to_orig[slot])];
        const auto& tb = ctg.tripped_branches;
        const size_t n_own = own ? ctg.own_branches.size() : 0;
        h_trip_start[static_cast<size_t>(slot)] = static_cast<int>(h_trip_branch_flat.size());
        h_trip_count[static_cast<size_t>(slot)] = static_cast<int>(tb.size() + n_own);
        // the own blocks first: a branch both tripped and listed with its own block (opened at
        // one end, its other end moved) takes its block (the first match wins)
        if (own)
            for (const RowBranchBlock& ob : ctg.own_branches) {
                h_trip_branch_flat.push_back(ob.branch);
                h_trip_own->push_back(static_cast<int>(h_own->size()));
                h_own->push_back(ob);
            }
        h_trip_branch_flat.insert(h_trip_branch_flat.end(), tb.begin(), tb.end());
        if (own) h_trip_own->insert(h_trip_own->end(), tb.size(), -1);
    }
}

// ---------------------------------------------------------------------------
// choose_reference_bus_from_ledger
// ---------------------------------------------------------------------------
int choose_reference_bus_from_ledger(
    const std::vector<Contingency>&                               contingencies,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>& Ybus_rm,
    const LedgerData&                                             ld,
    int                                                           forced,
    int                                                           fallback)
{
    std::vector<int>    cand_bus;
    std::vector<double> cand_w;
    for (int b = 0; b < ld.n_bus && b < static_cast<int>(ld.slack_weights.size()); ++b)
        if (ld.p_row_of_bus[static_cast<size_t>(b)] >= 0 && ld.slack_weights[static_cast<size_t>(b)] != 0.) {
            cand_bus.push_back(b);
            cand_w.push_back(ld.slack_weights[static_cast<size_t>(b)]);
        }
    const int choice = choose_reference_bus(contingencies, Ybus_rm.outerIndexPtr(),
                                            Ybus_rm.innerIndexPtr(), Ybus_rm,
                                            cand_bus, cand_w, forced);
    return choice < 0 ? fallback : choice;
}
