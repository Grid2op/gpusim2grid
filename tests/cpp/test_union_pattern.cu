// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// extend_ledger_union (ledger_extend.hpp): the union pattern of the topology actions -- spare
// buses appended as PQ buses, the dS pattern of every Ybus entry the base pattern lacks.
// Host only: needs no GPU.

#include <utility>
#include <vector>

#include "check.hpp"
#include "dtypes.hpp"
#include "ledger_data.hpp"
#include "ledger_extend.hpp"

#include "Eigen/SparseCore"

namespace {

using YbusRm = Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>;

// A Ybus pattern of n buses from its (row, col) entries (values 1, compressed).
YbusRm ybus(int n, const std::vector<std::pair<int, int>>& entries)
{
    std::vector<Eigen::Triplet<eigen_cplx_type>> t;
    for (const auto& [i, j] : entries) t.emplace_back(i, j, eigen_cplx_type(1.));
    YbusRm y(n, n);
    y.setFromTriplets(t.begin(), t.end());
    y.makeCompressed();
    return y;
}

// 3 buses in a line 0 - 1 - 2: bus 0 the slack (no equation, no unknown), buses 1 and 2 PQ
// (P row / theta col 0, 2; Q row / Vm col 1, 3). J = the dS pattern of the Ybus pattern.
const std::vector<std::pair<int, int>> kBase{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}, {2, 1}, {2, 2}};

LedgerData three_bus_ledger()
{
    LedgerData ld;
    ld.n_bus = 3;
    ld.p_row_of_bus = {-1, 0, 2};
    ld.theta_col_of_bus = {-1, 0, 2};
    ld.q_row_of_bus = {-1, 1, 3};
    ld.vm_col_of_bus = {-1, 1, 3};
    ld.q_col_of_bus = {-1, -1, -1};
    ld.p_buses = {1, 2};     ld.p_rows = {0, 2};
    ld.theta_buses = {1, 2}; ld.theta_cols = {0, 2};
    ld.q_buses = {1, 2};     ld.q_rows = {1, 3};
    ld.vm_buses = {1, 2};    ld.vm_cols = {1, 3};
    ld.dim_J = 4;
    // rows 0, 1 (bus 1): cols of buses 1, 2; rows 2, 3 (bus 2): cols of buses 1, 2
    ld.J_outer = {0, 4, 8, 12, 16};
    ld.J_inner = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    return ld;
}

bool has(const LedgerData& ld, int row, int col)
{
    for (int k = ld.J_outer[static_cast<size_t>(row)]; k < ld.J_outer[static_cast<size_t>(row) + 1]; ++k)
        if (ld.J_inner[static_cast<size_t>(k)] == col) return true;
    return false;
}

}  // namespace

TEST_CASE("no spare bus, no new entry: the ledger is unchanged")
{
    LedgerData ld = three_bus_ledger();
    const LedgerData before = ld;
    extend_ledger_union(ld, 0, ybus(3, kBase), ybus(3, kBase));
    CHECK(ld.n_bus == before.n_bus);
    CHECK(ld.dim_J == before.dim_J);
    CHECK(ld.J_outer == before.J_outer);
    CHECK(ld.J_inner == before.J_inner);
    CHECK(ld.n_spare_buses == 0);
}

TEST_CASE("a spare bus split from bus 2, with line 1-2's end 2 moved onto it")
{
    // spare bus 3: entries (3, 3), (1, 3), (3, 1) (the moved end) and the base ones
    std::vector<std::pair<int, int>> uni = kBase;
    uni.insert(uni.end(), {{3, 3}, {1, 3}, {3, 1}});
    LedgerData ld = three_bus_ledger();
    extend_ledger_union(ld, 1, ybus(4, uni), ybus(3, kBase));
    CHECK(ld.n_bus == 4);
    CHECK(ld.n_spare_buses == 1);
    CHECK(ld.dim_J == 6);
    // a PQ bus appended: P row / theta col 4, Q row / Vm col 5
    CHECK(ld.p_row_of_bus[3] == 4);
    CHECK(ld.theta_col_of_bus[3] == 4);
    CHECK(ld.q_row_of_bus[3] == 5);
    CHECK(ld.vm_col_of_bus[3] == 5);
    CHECK(ld.q_col_of_bus.size() == 4);
    // dS of the new entries: (3, 3), (3, 1) on rows 4, 5; (1, 3) on rows 0, 1
    for (int r : {4, 5})
        for (int c : {4, 5, 0, 1}) CHECK(has(ld, r, c));
    for (int r : {0, 1})
        for (int c : {4, 5}) CHECK(has(ld, r, c));
    // nothing between the spare and bus 2 (no entry joins them)
    CHECK(!has(ld, 4, 2));
    CHECK(!has(ld, 2, 4));
    // every base entry kept
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) CHECK(has(ld, r, c));
}

TEST_CASE("a merge: a new entry between existing buses 0 and 2 (no spare bus)")
{
    // the slack bus 0 has no row / col: only the dS of (2, 0) and (0, 2) that exist
    std::vector<std::pair<int, int>> uni = kBase;
    uni.insert(uni.end(), {{0, 2}, {2, 0}});
    LedgerData ld = three_bus_ledger();
    const LedgerData before = ld;
    extend_ledger_union(ld, 0, ybus(3, uni), ybus(3, kBase));
    CHECK(ld.n_bus == 3);
    CHECK(ld.dim_J == 4);
    CHECK(ld.J_inner == before.J_inner);   // rows / cols of bus 0 do not exist: nothing to add
}

TEST_CASE("sizes are checked")
{
    LedgerData ld = three_bus_ledger();
    CHECK_THROWS(extend_ledger_union(ld, 1, ybus(3, kBase), ybus(3, kBase)));    // union not 4 x 4
    CHECK_THROWS(extend_ledger_union(ld, 0, ybus(3, kBase), ybus(2, {{0, 0}})));  // base not n_bus
}

GPUSIM2GRID_TEST_MAIN(false)
