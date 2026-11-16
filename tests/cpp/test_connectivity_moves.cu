// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// check_connectivity / compute_component_masks on a union pattern (topology actions, bus
// moves): spare buses left out of the graph, explicit zeros not edges, rows with moves
// labelled on their own edge set, empty buses never islands. Host only: needs no GPU.

#include <utility>
#include <vector>

#include "check.hpp"
#include "contingency_analysis_helper.hpp"
#include "dtypes.hpp"

#include "Eigen/SparseCore"

namespace {

using YbusRm = Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>;

// Buses 0 - 1 - 2 in a line (unit branches), spare bus 3 (n_graph = 3); explicit zeros at the
// union entries (1, 3), (3, 1), (3, 3) (line 1-2's end 2 may move onto 3) and (0, 2), (2, 0)
// (a merge may join 0 and 2).
YbusRm union_ybus()
{
    std::vector<Eigen::Triplet<eigen_cplx_type>> t{
        {0, 0, 1.}, {0, 1, -1.}, {1, 0, -1.}, {1, 1, 2.}, {1, 2, -1.}, {2, 1, -1.}, {2, 2, 1.},
        {1, 3, 0.}, {3, 1, 0.},  {3, 3, 0.},  {0, 2, 0.}, {2, 0, 0.}};
    YbusRm y(4, 4);
    y.setFromTriplets(t.begin(), t.end());
    y.makeCompressed();
    return y;
}

constexpr int kNGraph = 3;

// The triplets of branch (i, j)'s unit stamp, subtracted (sign +1: removed) or added (-1).
void stamp(Contingency& c, int i, int j, double sign)
{
    c.triplets.push_back({i, i, sign * 1., 0.});
    c.triplets.push_back({j, j, sign * 1., 0.});
    c.triplets.push_back({i, j, sign * -1., 0.});
    c.triplets.push_back({j, i, sign * -1., 0.});
}

std::vector<Contingency> resolved(std::vector<Contingency> rows, const YbusRm& y)
{
    resolve_indices(rows, y.outerIndexPtr(), y.innerIndexPtr());
    return rows;
}

}  // namespace

TEST_CASE("no move: the spare bus is empty, the row connected")
{
    const YbusRm y = union_ybus();
    auto rows = resolved({Contingency{}}, y);
    check_connectivity(rows, y, kNGraph);
    CHECK(!rows[0].disconnected);
    CHECK((rows[0].empty_effective == std::vector<int>{3}));
}

TEST_CASE("an explicit zero is no edge: tripping 1-2 islands bus 2")
{
    const YbusRm y = union_ybus();
    Contingency c;
    stamp(c, 1, 2, +1.);   // the (0, 2) union entry, zero, must not hold bus 2
    auto rows = resolved({c}, y);
    check_connectivity(rows, y, kNGraph);
    CHECK(rows[0].disconnected);
}

TEST_CASE("line 1-2's end 2 moved onto the spare: connected once bus 2 is empty")
{
    const YbusRm y = union_ybus();
    Contingency c;
    stamp(c, 1, 2, +1.);   // off bus 2
    stamp(c, 1, 3, -1.);   // onto spare 3
    Contingency empty = c;
    empty.empty_buses = {2};
    auto rows = resolved({c, empty}, y);
    check_connectivity(rows, y, kNGraph);
    CHECK(rows[0].disconnected);           // bus 2 keeps an element: an island
    CHECK(!rows[1].disconnected);          // bus 2 has none: masked, not an island
    CHECK((rows[1].empty_effective == std::vector<int>{2}));   // spare 3 is used
}

TEST_CASE("a stamp added where the base value is 0 reconnects (a merge)")
{
    const YbusRm y = union_ybus();
    Contingency c;
    stamp(c, 1, 2, +1.);
    stamp(c, 0, 2, -1.);   // bus 2 now hangs on bus 0
    auto rows = resolved({c}, y);
    check_connectivity(rows, y, kNGraph);
    CHECK(!rows[0].disconnected);
}

TEST_CASE("mask mode: the island of a move row is masked, its empty buses are not islands")
{
    const YbusRm y = union_ybus();
    Contingency c;
    stamp(c, 1, 2, +1.);
    stamp(c, 1, 3, -1.);
    auto rows = resolved({c}, y);
    MaskConfig cfg;   // no reference / controller information: nothing to skip on
    compute_component_masks(rows, y, cfg, kNGraph);
    CHECK(!rows[0].disconnected);
    CHECK((rows[0].masked_buses == std::vector<int>{2}));
}

TEST_CASE("without spare buses the graph is the whole pattern, as before")
{
    // the same line without the union entries: n_graph = all, explicit zeros count (today)
    std::vector<Eigen::Triplet<eigen_cplx_type>> t{
        {0, 0, 1.}, {0, 1, -1.}, {1, 0, -1.}, {1, 1, 2.}, {1, 2, -1.}, {2, 1, -1.}, {2, 2, 1.},
        {0, 2, 0.}, {2, 0, 0.}};
    YbusRm y(3, 3);
    y.setFromTriplets(t.begin(), t.end());
    y.makeCompressed();
    Contingency c;
    stamp(c, 1, 2, +1.);
    auto rows = resolved({c}, y);
    check_connectivity(rows, y);
    CHECK(!rows[0].disconnected);   // the zero (0, 2) entry is an edge of the pattern
    CHECK(rows[0].empty_effective.empty());
}

TEST_CASE("a merge without spare bus: union entries are explicit zeros, not edges")
{
    // the 3 base buses with the (0, 2) / (2, 0) union entries, no spare bus (n_graph = all)
    std::vector<Eigen::Triplet<eigen_cplx_type>> t{
        {0, 0, 1.}, {0, 1, -1.}, {1, 0, -1.}, {1, 1, 2.}, {1, 2, -1.}, {2, 1, -1.}, {2, 2, 1.},
        {0, 2, 0.}, {2, 0, 0.}};
    YbusRm y(3, 3);
    y.setFromTriplets(t.begin(), t.end());
    y.makeCompressed();
    Contingency trip;
    stamp(trip, 1, 2, +1.);
    Contingency merge = trip;
    stamp(merge, 0, 2, -1.);   // line 1-2's end 1 moved onto bus 0
    auto rows = resolved({trip, merge}, y);
    check_connectivity(rows, y, /*n_graph=*/-1, /*union_pattern=*/true);
    CHECK(rows[0].disconnected);     // the zero (0, 2) does not hold bus 2
    CHECK(!rows[1].disconnected);    // the merged stamp does
}

GPUSIM2GRID_TEST_MAIN(false)
