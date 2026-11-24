// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// own_lu/lu_plan.cpp -- see lu_plan.hpp. A line-by-line port of
// gpu-security-analysis' LuPlan (engine/lu.py): _cholesky_rows, _levels,
// _left_looking, the item sort and the level slices.
//
// Original batched LU (algorithm, symbolic plan and kernels): Ralf Cortes Do
// Nascimento (RTE), contributed from gpu-security-analysis.

#include "lu_plan.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace own_lu {

namespace {

// Position of (i, j) in the LU CSR (binary search in row i), -1 if absent.
inline int entry_of(const std::vector<int>& rp, const std::vector<int>& ci, int i, int j)
{
    const auto b = ci.begin() + rp[i], e = ci.begin() + rp[i + 1];
    const auto it = std::lower_bound(b, e, j);
    return (it != e && *it == j) ? static_cast<int>(it - ci.begin()) : -1;
}

// Strict lower rows of the Cholesky factor of a symmetric pattern (CSR, sorted
// columns): elimination tree, then each row's etree paths (Liu).
void cholesky_rows(int n, const std::vector<int>& srp, const std::vector<int>& sci,
                   std::vector<int>& lrp, std::vector<int>& lci)
{
    std::vector<int> parent(n, -1), anc(n, -1);
    for (int i = 0; i < n; ++i) {
        for (int p = srp[i]; p < srp[i + 1]; ++p) {
            const int k = sci[p];
            if (k >= i) break;
            int j = k;
            while (anc[j] != -1 && anc[j] != i) {
                const int nxt = anc[j];
                anc[j] = i;
                j = nxt;
            }
            if (anc[j] == -1) {
                anc[j] = i;
                parent[j] = i;
            }
        }
    }
    std::vector<int> mark(n, -1);
    lrp.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        mark[i] = i;
        for (int p = srp[i]; p < srp[i + 1]; ++p) {
            int j = sci[p];
            if (j >= i) break;
            while (mark[j] != i) {
                ++lrp[i + 1];
                mark[j] = i;
                j = parent[j];
            }
        }
    }
    for (int i = 0; i < n; ++i) lrp[i + 1] += lrp[i];
    lci.assign(lrp[n], 0);
    std::fill(mark.begin(), mark.end(), -1);
    for (int i = 0; i < n; ++i) {
        mark[i] = i;
        int q = lrp[i];
        for (int p = srp[i]; p < srp[i + 1]; ++p) {
            int j = sci[p];
            if (j >= i) break;
            while (mark[j] != i) {
                lci[q++] = j;
                mark[j] = i;
                j = parent[j];
            }
        }
        std::sort(lci.begin() + lrp[i], lci.begin() + lrp[i + 1]);
    }
}

}  // namespace

LuPlan build_lu_plan(int n, const std::vector<int>& indptr, const std::vector<int>& indices,
                     const std::vector<int>& order)
{
    if (n <= 0 || static_cast<int>(indptr.size()) != n + 1
        || static_cast<int>(order.size()) != n
        || static_cast<long long>(indices.size()) != static_cast<long long>(indptr[n]))
        throw std::invalid_argument("[own_lu::build_lu_plan] pattern / order size mismatch");

    LuPlan P;
    P.n = n;
    P.nnz = indptr[n];
    P.order = order;
    P.inv_order.assign(n, -1);
    for (int k = 0; k < n; ++k) {
        const int o = order[k];
        if (o < 0 || o >= n || P.inv_order[o] != -1)
            throw std::invalid_argument("[own_lu::build_lu_plan] order is not a permutation");
        P.inv_order[o] = k;
    }
    const std::vector<int>& inv = P.inv_order;

    // ---- symmetrized permuted pattern s = q + q^T, q = A[order][:, order] ----
    std::vector<int> srp(n + 1, 0);
    for (int r = 0; r < n; ++r)
        for (int p = indptr[r]; p < indptr[r + 1]; ++p) {
            ++srp[inv[r] + 1];
            ++srp[inv[indices[p]] + 1];
        }
    for (int i = 0; i < n; ++i) srp[i + 1] += srp[i];
    std::vector<int> sci(srp[n]);
    {
        std::vector<int> fill(srp.begin(), srp.end() - 1);
        for (int r = 0; r < n; ++r)
            for (int p = indptr[r]; p < indptr[r + 1]; ++p) {
                const int i = inv[r], j = inv[indices[p]];
                sci[fill[i]++] = j;
                sci[fill[j]++] = i;
            }
    }
    {   // sort + unique each row, compact
        std::vector<int> nrp(n + 1, 0);
        int w = 0;
        for (int i = 0; i < n; ++i) {
            auto b = sci.begin() + srp[i], e = sci.begin() + srp[i + 1];
            std::sort(b, e);
            const auto u = std::unique(b, e);
            for (auto it = b; it != u; ++it) sci[w++] = *it;
            nrp[i + 1] = w;
        }
        sci.resize(w);
        srp.swap(nrp);
    }

    // ---- LU pattern: L (Cholesky rows) + L^T + I, one sorted CSR ----
    std::vector<int> lrp, lci;
    cholesky_rows(n, srp, sci, lrp, lci);
    std::vector<int> ucnt(n + 1, 0);  // transpose: row i gets every k > i with i in L row k
    for (int k = 0; k < n; ++k)
        for (int p = lrp[k]; p < lrp[k + 1]; ++p) ++ucnt[lci[p] + 1];
    for (int i = 0; i < n; ++i) ucnt[i + 1] += ucnt[i];
    std::vector<int> uci(ucnt[n]);
    {
        std::vector<int> fill(ucnt.begin(), ucnt.end() - 1);
        for (int k = 0; k < n; ++k)  // k ascending: each U row comes out sorted
            for (int p = lrp[k]; p < lrp[k + 1]; ++p) uci[fill[lci[p]]++] = k;
    }
    P.rp.assign(n + 1, 0);
    P.dp.assign(n, 0);
    for (int i = 0; i < n; ++i)
        P.rp[i + 1] = P.rp[i] + (lrp[i + 1] - lrp[i]) + 1 + (ucnt[i + 1] - ucnt[i]);
    P.n_ent = P.rp[n];
    P.ci.resize(P.n_ent);
    std::vector<int> row_of(P.n_ent);
    for (int i = 0; i < n; ++i) {
        int q = P.rp[i];
        for (int p = lrp[i]; p < lrp[i + 1]; ++p) P.ci[q++] = lci[p];
        P.dp[i] = q;
        P.ci[q++] = i;
        for (int p = ucnt[i]; p < ucnt[i + 1]; ++p) P.ci[q++] = uci[p];
        for (int e = P.rp[i]; e < P.rp[i + 1]; ++e) row_of[e] = i;
    }
    const std::vector<int>& rp = P.rp;
    const std::vector<int>& dp = P.dp;
    const std::vector<int>& ci = P.ci;

    // ---- levels: factorization / forward, backward ----
    std::vector<int> level(n, 0), blevel(n, 0);
    for (int i = 0; i < n; ++i) {
        int m = -1;
        for (int e = rp[i]; e < dp[i]; ++e) m = std::max(m, level[ci[e]]);
        level[i] = m + 1;
    }
    for (int i = n - 1; i >= 0; --i) {
        int m = -1;
        for (int e = dp[i] + 1; e < rp[i + 1]; ++e) m = std::max(m, blevel[ci[e]]);
        blevel[i] = m + 1;
    }
    const int n_lev = n ? *std::max_element(level.begin(), level.end()) + 1 : 0;
    const int n_blev = n ? *std::max_element(blevel.begin(), blevel.end()) + 1 : 0;
    P.levels = n_lev;

    // ---- A entries in the LU pattern ----
    P.a_ent.resize(P.nnz);
    std::vector<char> in_a(P.n_ent, 0);
    for (int r = 0; r < n; ++r)
        for (int p = indptr[r]; p < indptr[r + 1]; ++p) {
            const int e = entry_of(rp, ci, inv[r], inv[indices[p]]);
            if (e < 0) throw std::logic_error("[own_lu::build_lu_plan] A entry missing from the LU pattern");
            P.a_ent[p] = e;
            in_a[e] = 1;
        }

    // ---- left-looking items. Every contribution L(i,k) U(k,j) goes to the entry (i,j), whose step
    //      2 level[min(i,j)] + (i > j) is a function of the entry alone: the items are the entries ordered by
    //      (step, entry), each with its contributions in ascending k -- two counting sorts, linear. An entry is
    //      an item when it has a contribution; every diagonal and every L entry is one anyway (it must be
    //      computed even without contributions: its placeholder). ----
    // The contributions of U row k: (i, j) over its columns; row i holds every column of U row k (the
    // pattern is closed under elimination), so (i, j) is found by walking row i along them.
    auto for_each_contribution = [&](auto&& fn) {
        for (int k = 0; k < n; ++k) {
            const int u0 = dp[k] + 1, u1 = rp[k + 1];  // U row k: columns j > k = rows i > k of L col k
            if (u0 >= u1) continue;
            for (int a = u0; a < u1; ++a) {
                const int i = ci[a];
                const int lik = entry_of(rp, ci, i, k);
                int e = entry_of(rp, ci, i, ci[u0]);
                if (lik < 0 || e < 0) throw std::logic_error("[own_lu::build_lu_plan] pattern not closed");
                for (int c = u0; c < u1; ++c) {
                    const int j = ci[c];
                    while (ci[e] < j) ++e;
                    if (ci[e] != j) throw std::logic_error("[own_lu::build_lu_plan] pattern not closed");
                    fn(e, lik, c);
                }
            }
        }
    };
    // one pass: the contributions in visit order (k ascending), then scattered by entry -- stable: each entry's
    // contributions stay in k order
    std::vector<int> te, tl, tc;
    {
        long long total = 0;
        for (int k = 0; k < n; ++k) {
            const long long c = rp[k + 1] - dp[k] - 1;
            total += c * c;
        }
        te.reserve(static_cast<size_t>(total));
        tl.reserve(static_cast<size_t>(total));
        tc.reserve(static_cast<size_t>(total));
    }
    for_each_contribution([&](int e, int lik, int c) { te.push_back(e); tl.push_back(lik); tc.push_back(c); });
    std::vector<int> cstart(static_cast<size_t>(P.n_ent) + 1, 0);
    for (const int e : te) ++cstart[static_cast<size_t>(e) + 1];
    for (int e = 0; e < P.n_ent; ++e) cstart[static_cast<size_t>(e) + 1] += cstart[static_cast<size_t>(e)];
    std::vector<int> ecl(te.size()), ecu(te.size());
    {
        std::vector<int> fill(cstart.begin(), cstart.end() - 1);
        for (size_t q = 0; q < te.size(); ++q) {
            const int w = fill[static_cast<size_t>(te[q])]++;
            ecl[static_cast<size_t>(w)] = tl[q];
            ecu[static_cast<size_t>(w)] = tc[q];
        }
    }
    // the items by (step, entry): entries ascending into their step's bucket
    const int n_steps = 2 * n_lev;
    std::vector<int> scount(static_cast<size_t>(n_steps) + 1, 0), estep(P.n_ent, -1);
    for (int e = 0; e < P.n_ent; ++e) {
        const int i = row_of[e], j = ci[e];
        const bool item = cstart[static_cast<size_t>(e) + 1] > cstart[static_cast<size_t>(e)] || j <= i;  // diagonal / L
        if (!item) continue;
        estep[e] = 2 * level[std::min(i, j)] + (i > j ? 1 : 0);
        ++scount[static_cast<size_t>(estep[e]) + 1];
    }
    for (int s2 = 0; s2 < n_steps; ++s2) scount[static_cast<size_t>(s2) + 1] += scount[static_cast<size_t>(s2)];
    std::vector<int> items(scount.back());
    {
        std::vector<int> fill(scount.begin(), scount.end() - 1);
        for (int e = 0; e < P.n_ent; ++e)
            if (estep[e] >= 0) items[static_cast<size_t>(fill[static_cast<size_t>(estep[e])]++)] = e;
    }
    const size_t n_items = items.size();
    std::vector<int> item_step(n_items);
    P.f_tgt.resize(n_items);
    P.f_dg.resize(n_items);
    P.f_cptr.resize(n_items + 1);
    P.cl.resize(ecl.size());
    P.cu.resize(ecu.size());
    P.f_cptr[0] = 0;
    int w = 0;
    for (size_t q = 0; q < n_items; ++q) {
        const int e = items[q];
        P.f_tgt[q] = in_a[e] ? e : ~e;
        P.f_dg[q] = dp[std::min(row_of[e], ci[e])];
        item_step[q] = estep[e];
        const int c0 = cstart[static_cast<size_t>(e)], c1 = cstart[static_cast<size_t>(e) + 1];
        std::copy(ecl.begin() + c0, ecl.begin() + c1, P.cl.begin() + w);
        std::copy(ecu.begin() + c0, ecu.begin() + c1, P.cu.begin() + w);
        w += c1 - c0;
        P.f_cptr[q + 1] = w;
    }
    P.contributions = static_cast<long long>(P.cl.size());

    for (size_t a = 0; a < item_step.size();) {
        size_t b = a;
        int mc = 0;
        while (b < item_step.size() && item_step[b] == item_step[a]) {
            mc = std::max(mc, P.f_cptr[b + 1] - P.f_cptr[b]);
            ++b;
        }
        P.factor_steps.push_back({static_cast<int>(a), static_cast<int>(b - a), mc, item_step[a] % 2 == 0});
        a = b;
    }

    // ---- triangular solve levels ----
    auto slices = [n](const std::vector<int>& lev, int n_levels, const std::vector<int>& len,
                      std::vector<int>& rows, std::vector<LevelSlice>& out) {
        rows.resize(n);
        std::iota(rows.begin(), rows.end(), 0);
        std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) { return lev[a] < lev[b]; });
        size_t a = 0;
        for (int l = 0; l < n_levels; ++l) {
            size_t b = a;
            int mc = 0;
            while (b < rows.size() && lev[rows[b]] == l) mc = std::max(mc, len[rows[b++]]);
            if (b > a) out.push_back({static_cast<int>(a), static_cast<int>(b - a), mc, false});
            a = b;
        }
    };
    std::vector<int> fw_len(n), bw_len(n);
    for (int i = 0; i < n; ++i) {
        fw_len[i] = dp[i] - rp[i];
        bw_len[i] = rp[i + 1] - dp[i] - 1;
    }
    slices(level, n_lev, fw_len, P.fw_rows, P.fw_levels);
    slices(blevel, n_blev, bw_len, P.bw_rows, P.bw_levels);
    return P;
}

}  // namespace own_lu
