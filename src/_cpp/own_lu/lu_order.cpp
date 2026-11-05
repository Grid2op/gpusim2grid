// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// own_lu/lu_order.cpp -- see lu_order.hpp.

#include "lu_order.hpp"

#include <numeric>
#include <stdexcept>
#include <string>

#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

#include "../ledger_data.hpp"
#include "../user_ordering.hpp"

namespace own_lu {

std::vector<int> amd_order(int n, const std::vector<int>& indptr, const std::vector<int>& indices)
{
    if (constrained_order_hook()) {   // lightsim2grid's AMD: its constrained order, no constraint
        LedgerData none;
        std::vector<int> perm;
        if (constrained_order_hook()(n, indptr.data(), indices.data(), none, perm)
            && static_cast<int>(perm.size()) == n)
            return perm;
    }
    // Eigen's AMD (symmetrizes A + A^T itself); its indices() are new -> old
    // (checked against the fill: the inverse convention explodes it)
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A(n, n);
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(indices.size());
    for (int r = 0; r < n; ++r)
        for (int p = indptr[r]; p < indptr[r + 1]; ++p) t.emplace_back(r, indices[p], 1.0);
    A.setFromTriplets(t.begin(), t.end());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> pm;
    Eigen::AMDOrdering<int> amd;
    amd(A, pm);
    return std::vector<int>(pm.indices().data(), pm.indices().data() + n);
}

std::vector<int> choose_order(ReorderingAlg alg, int n, const std::vector<int>& indptr,
                              const std::vector<int>& indices, const LedgerData* ord_ledger,
                              const std::vector<int>& user_perm)
{
    switch (alg) {
    case ReorderingAlg::Default:
        if (ord_ledger && ord_ledger->ord_constraints_read && constrained_order_hook())
            return ls2g_constrained_perm(*ord_ledger);
        return amd_order(n, indptr, indices);
    case ReorderingAlg::Amd:
        return amd_order(n, indptr, indices);
    case ReorderingAlg::None: {
        std::vector<int> id(n);
        std::iota(id.begin(), id.end(), 0);
        return id;
    }
    case ReorderingAlg::Ls2gConstrained:
    case ReorderingAlg::Ls2gConstrainedNd:
        if (static_cast<int>(user_perm.size()) != n)
            throw std::invalid_argument("linear_solver='own_lu': no lightsim2grid order computed "
                                        "for this reordering_alg");
        return user_perm;
    default:
        throw std::invalid_argument(
            "linear_solver='own_lu' takes reordering_alg 'default', 'amd', 'none', "
            "'ls2g_constrained' or 'ls2g_constrained_nd' (it never pivots: a symmetric order "
            "computed on the host; cuDSS's 'nested_dissection' / 'colamd' / 'btf_colamd' are not)");
    }
}

}  // namespace own_lu
