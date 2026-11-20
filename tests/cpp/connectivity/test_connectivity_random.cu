// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// check_connectivity / compute_component_masks against a plain breadth-first search of each
// row's own graph, on random meshed grids (a ring, chords, parallel branches, pendant chains:
// bridges) with spare buses: rows tripping branches, rows moving branch ends of a bus onto a
// spare (the bus emptied or not), merges onto a union entry, and mixes of them. Whatever path a
// row takes (the DFS tree, the cycle-space certificate, a search), the answer must be the
// search's. Host only: needs no GPU.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "contingency_analysis_helper.hpp"
#include "dtypes.hpp"

#include "Eigen/SparseCore"

namespace {

using YbusRm = Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>;

struct Rng {   // splitmix64: the same rows on every platform
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

struct Grid {
    int n_graph = 0, n_spare = 0;
    std::vector<std::pair<int, int>> branches;   // unit admittance each (parallel ones repeat)
    YbusRm y;
};

// n buses on a ring, `chords` random chords (some doubling an existing branch), pendant chains
// hung off the ring (their branches are bridges); spare buses with explicit-zero union entries
// to a few base buses each, and a few explicit zeros between base buses (merge targets)
Grid make_grid(Rng& rng, int n_ring, int chords, int chains, int n_spare, bool zeros)
{
    Grid g;
    for (int i = 0; i < n_ring; ++i) g.branches.emplace_back(i, (i + 1) % n_ring);
    for (int c = 0; c < chords; ++c) {
        const int a = rng.below(n_ring), b = rng.below(n_ring);
        if (a != b) g.branches.emplace_back(std::min(a, b), std::max(a, b));
    }
    int n = n_ring;
    for (int c = 0; c < chains; ++c) {
        int at = rng.below(n);
        const int len = 1 + rng.below(3);
        for (int k = 0; k < len; ++k) { g.branches.emplace_back(at, n); at = n++; }
    }
    g.n_graph = n;
    g.n_spare = n_spare;
    const int n_total = n + n_spare;
    std::map<std::pair<int, int>, double> v;
    for (auto [a, b] : g.branches) {
        v[{a, a}] += 1.; v[{b, b}] += 1.; v[{a, b}] -= 1.; v[{b, a}] -= 1.;
    }
    for (int s = n; s < n_total; ++s) {
        v[{s, s}] += 0.;
        for (int k = 0; k < 4; ++k) {
            const int b = rng.below(n);
            v[{s, b}] += 0.; v[{b, s}] += 0.;
        }
    }
    for (int k = 0; k < (zeros ? 6 : 0); ++k) {
        const int a = rng.below(n), b = rng.below(n);
        if (a != b && !v.count({a, b})) { v[{a, b}] = 0.; v[{b, a}] = 0.; }
    }
    std::vector<Eigen::Triplet<eigen_cplx_type>> t;
    for (const auto& [ij, x] : v) t.emplace_back(ij.first, ij.second, eigen_cplx_type(x, 0.));
    g.y = YbusRm(n_total, n_total);
    g.y.setFromTriplets(t.begin(), t.end());
    g.y.makeCompressed();
    return g;
}

// branch (i, j)'s unit stamp: sign +1 takes it out, -1 puts it in
void stamp(Contingency& c, int i, int j, double sign)
{
    c.triplets.push_back({i, i, sign * 1., 0.});
    c.triplets.push_back({j, j, sign * 1., 0.});
    c.triplets.push_back({i, j, sign * -1., 0.});
    c.triplets.push_back({j, i, sign * -1., 0.});
}

bool has_entry(const YbusRm& y, int i, int j)
{
    for (YbusRm::InnerIterator it(y, i); it; ++it) if (it.col() == j) return true;
    return false;
}

// a random row: branch trips, a bus split onto a spare, a merge, or a mix
Contingency make_row(Rng& rng, const Grid& g)
{
    Contingency c;
    const int kind = rng.below(4);   // 0 trips, 1 split, 2 split + trips, 3 merge + trips
    const int nb = static_cast<int>(g.branches.size());
    std::vector<char> used(static_cast<size_t>(nb), 0);
    auto trip = [&](int count) {
        for (int k = 0; k < count; ++k) {
            const int e = rng.below(nb);
            if (used[static_cast<size_t>(e)]) continue;
            used[static_cast<size_t>(e)] = 1;
            stamp(c, g.branches[static_cast<size_t>(e)].first, g.branches[static_cast<size_t>(e)].second, +1.);
        }
    };
    if (kind == 0) trip(1 + rng.below(6));
    if ((kind == 1 || kind == 2) && g.n_spare > 0) {
        const int s = g.n_graph + rng.below(g.n_spare);
        // a bus with union entries to s: some of its branch ends move to s (only those whose
        // other end has a union entry with s too, as a real split's plan reserves)
        std::vector<int> cand;
        for (int b = 0; b < g.n_graph; ++b) if (has_entry(g.y, b, s)) cand.push_back(b);
        if (!cand.empty()) {
            const int b = cand[static_cast<size_t>(rng.below(static_cast<int>(cand.size())))];
            int moved = 0, kept = 0;
            for (int e = 0; e < nb; ++e) {
                auto [x, z] = g.branches[static_cast<size_t>(e)];
                if (x != b && z != b) continue;
                const int other = x == b ? z : x;
                if (other == s || !has_entry(g.y, s, other) || rng.below(2)) { ++kept; continue; }
                used[static_cast<size_t>(e)] = 1;
                stamp(c, b, other, +1.);
                stamp(c, s, other, -1.);
                ++moved;
            }
            if (moved > 0 && kept == 0 && rng.below(2)) c.empty_buses.push_back(b);   // nothing left on b
        }
        if (kind == 2) trip(1 + rng.below(3));
    }
    if (kind == 3) {
        // a stamp where the base value is 0 (a branch end merged onto a union entry)
        for (int k = 0; k < 20; ++k) {
            const int a = rng.below(g.n_graph);
            for (YbusRm::InnerIterator it(g.y, a); it; ++it) {
                const int b = static_cast<int>(it.col());
                if (b != a && b < g.n_graph && std::abs(it.value()) == 0.) { stamp(c, a, b, -1.); k = 20; break; }
            }
        }
        trip(1 + rng.below(4));
    }
    return c;
}

// The search: the row's graph (base minus its summed changes, its empty buses and every spare it
// does not touch left out), components in increasing bus order, the largest kept (the first of
// equal ones). Returns the buses outside it, sorted; `n_comp` the number of components.
std::vector<int> reference_masked(const Grid& g, const Contingency& c, int& n_comp, std::vector<int>& empty)
{
    const int n_total = g.n_graph + g.n_spare;
    std::map<std::pair<int, int>, double> val;
    for (int i = 0; i < n_total; ++i)
        for (YbusRm::InnerIterator it(g.y, i); it; ++it) val[{i, static_cast<int>(it.col())}] = it.value().real();
    std::set<int> touched;
    for (const auto& t : c.triplets) { val[{t.row, t.col}] -= t.delta_re; touched.insert(t.row); touched.insert(t.col); }
    empty = c.empty_buses;
    for (int s = g.n_graph; s < n_total; ++s) if (!touched.count(s)) empty.push_back(s);
    std::sort(empty.begin(), empty.end());
    empty.erase(std::unique(empty.begin(), empty.end()), empty.end());
    std::vector<char> gone(static_cast<size_t>(n_total), 0);
    for (int b : empty) gone[static_cast<size_t>(b)] = 1;
    std::vector<std::vector<int>> adj(static_cast<size_t>(n_total));
    for (const auto& [ij, x] : val)
        if (ij.first != ij.second && std::abs(x) >= 1e-10) adj[static_cast<size_t>(ij.first)].push_back(ij.second);
    std::vector<int> comp(static_cast<size_t>(n_total), -1), size;
    n_comp = 0;
    for (int s = 0; s < n_total; ++s) {
        if (gone[static_cast<size_t>(s)] || comp[static_cast<size_t>(s)] >= 0) continue;
        std::vector<int> q{s};
        comp[static_cast<size_t>(s)] = n_comp;
        for (size_t h = 0; h < q.size(); ++h)
            for (int v : adj[static_cast<size_t>(q[h])])
                if (!gone[static_cast<size_t>(v)] && comp[static_cast<size_t>(v)] < 0) {
                    comp[static_cast<size_t>(v)] = n_comp;
                    q.push_back(v);
                }
        size.push_back(static_cast<int>(q.size()));
        ++n_comp;
    }
    std::vector<int> masked;
    if (n_comp <= 1) return masked;
    const int main = static_cast<int>(std::max_element(size.begin(), size.end()) - size.begin());
    for (int b = 0; b < n_total; ++b)
        if (comp[static_cast<size_t>(b)] >= 0 && comp[static_cast<size_t>(b)] != main) masked.push_back(b);
    return masked;
}

// the largest component is not unique: which one a path keeps is not pinned
bool tied(const Grid& g, const Contingency& c)
{
    int n_comp = 0;
    std::vector<int> empty;
    const std::vector<int> masked = reference_masked(g, c, n_comp, empty);
    const int n_live = g.n_graph + g.n_spare - static_cast<int>(empty.size());
    return n_comp > 1 && 2 * static_cast<int>(masked.size()) >= n_live;
}

// union_pattern: explicit zeros (no edges) between base buses, merge targets
void check_grid(uint64_t seed, int n_ring, int chords, int chains, int n_spare, bool union_pattern, int n_rows)
{
    Rng rng{seed};
    const Grid g = make_grid(rng, n_ring, chords, chains, n_spare, union_pattern);
    std::vector<Contingency> rows;
    for (int r = 0; r < n_rows; ++r) rows.push_back(make_row(rng, g));
    resolve_indices(rows, g.y.outerIndexPtr(), g.y.innerIndexPtr());
    std::vector<Contingency> conn = rows, masks = rows;
    ConnectivityStats conn_stats, mask_stats;
    check_connectivity(conn, g.y, g.n_graph, union_pattern, &conn_stats);
    compute_component_masks(masks, g.y, MaskConfig{}, g.n_graph, union_pattern, &mask_stats);
    int n_split = 0, n_islands = 0;
    for (int r = 0; r < n_rows; ++r) {
        int n_comp = 0;
        std::vector<int> empty;
        const std::vector<int> want = reference_masked(g, rows[static_cast<size_t>(r)], n_comp, empty);
        const std::string where = "seed " + std::to_string(seed) + " row " + std::to_string(r);
        n_split += rows[static_cast<size_t>(r)].triplets.size() > 0 && n_comp > 1;
        n_islands += n_comp > 1;
        CHECK_MSG(conn[static_cast<size_t>(r)].disconnected == (n_comp > 1), where + ": check_connectivity");
        CHECK_MSG(conn[static_cast<size_t>(r)].empty_effective == empty, where + ": empty buses");
        if (!tied(g, rows[static_cast<size_t>(r)]))
            CHECK_MSG(masks[static_cast<size_t>(r)].masked_buses == want, where + ": masked buses");
        else
            CHECK_MSG(masks[static_cast<size_t>(r)].masked_buses.size() == want.size(), where + ": masked count");
        CHECK_MSG(!masks[static_cast<size_t>(r)].disconnected, where + ": a mask, no skip");
    }
    // the rows reach both answers (else the case tests nothing)
    CHECK(n_islands > n_rows / 50);
    CHECK(n_islands < n_rows - n_rows / 50);
    (void)n_split;
    // the certificate answers rows of both kinds (else only the search is tested); with zeros no
    // edges (spare buses, a union pattern) the rows with moves too
    const std::string where = "seed " + std::to_string(seed);
    for (const ConnectivityStats* st : {&conn_stats, &mask_stats}) {
        CHECK_MSG(st->certified_removal > n_rows / 50, where + ": certified removals");
        if (union_pattern || n_spare > 0)
            CHECK_MSG(st->certified_moves > n_rows / 50, where + ": certified moves");
    }
    // both functions answer the same rows by the certificate (same index, same rows)
    CHECK(conn_stats.certified_removal == mask_stats.certified_removal);
    CHECK(conn_stats.certified_moves == mask_stats.certified_moves);
}

}  // namespace

TEST_CASE("random meshed grids with spare buses: every row as the search says")
{
    check_grid(1, 40, 20, 6, 3, true, 3000);
    check_grid(2, 120, 90, 15, 6, true, 3000);
    check_grid(3, 25, 4, 10, 2, true, 3000);   // sparse: long cycles, many bridges
}

TEST_CASE("without spare buses: merges onto explicit zeros, and the plain pattern as before")
{
    check_grid(4, 60, 40, 8, 0, true, 3000);
    check_grid(5, 60, 40, 8, 0, false, 3000);
}

GPUSIM2GRID_TEST_MAIN(false)
