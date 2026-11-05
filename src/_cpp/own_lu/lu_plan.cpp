// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// own_lu/lu_plan.cpp -- see lu_plan.hpp. A line-by-line port of
// gpu-security-analysis' LuPlan (engine/lu.py): _cholesky_rows, _levels,
// _left_looking, the item sort and the level slices.

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

    // ---- left-looking items: one record per contribution, then one
    //      placeholder per diagonal and per L entry (so that every entry that
    //      must be computed has an item even without contributions) ----
    struct Rec { int tgt, lo, up, step; };
    std::vector<Rec> recs;
    {
        long long total = 0;
        for (int k = 0; k < n; ++k) {
            const long long c = rp[k + 1] - dp[k] - 1;
            total += c * c;
        }
        recs.reserve(static_cast<size_t>(total) + n + (P.n_ent - n) / 2);
    }
    for (int k = 0; k < n; ++k) {
        const int u0 = dp[k] + 1, u1 = rp[k + 1];  // U row k: columns j > k = rows i > k of L col k
        for (int a = u0; a < u1; ++a) {
            const int i = ci[a];
            const int lik = entry_of(rp, ci, i, k);
            for (int c = u0; c < u1; ++c) {
                const int j = ci[c];
                const int p = std::min(i, j);
                recs.push_back({entry_of(rp, ci, i, j), lik, c, 2 * level[p] + (i > j ? 1 : 0)});
            }
        }
    }
    for (int p = 0; p < n; ++p) recs.push_back({dp[p], -1, -1, 2 * level[p]});
    for (int i = 0; i < n; ++i)
        for (int e = rp[i]; e < dp[i]; ++e) recs.push_back({e, -1, -1, 2 * level[ci[e]] + 1});
    std::stable_sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) {
        return a.step != b.step ? a.step < b.step : a.tgt < b.tgt;
    });

    std::vector<int> item_step;
    P.f_cptr.push_back(0);
    for (size_t q = 0; q < recs.size(); ++q) {
        const Rec& r = recs[q];
        if (q == 0 || r.tgt != recs[q - 1].tgt || r.step != recs[q - 1].step) {
            if (q != 0) P.f_cptr.push_back(static_cast<int>(P.cl.size()));
            const int piv = std::min(row_of[r.tgt], ci[r.tgt]);
            P.f_tgt.push_back(in_a[r.tgt] ? r.tgt : ~r.tgt);
            P.f_dg.push_back(dp[piv]);
            item_step.push_back(r.step);
        }
        if (r.lo >= 0) {
            P.cl.push_back(r.lo);
            P.cu.push_back(r.up);
        }
    }
    P.f_cptr.push_back(static_cast<int>(P.cl.size()));
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
