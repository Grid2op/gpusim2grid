// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// v_groups_by_active (contingency/v_groups.cuh): the group of every ACTIVE row of a
// batch, from the group of every original row (ContingencyBatch::set_v_groups).
// Host only: needs no GPU.

#include <vector>

#include "check.hpp"
#include "contingency/v_groups.cuh"

TEST_CASE("identity: every row active")
{
    const std::vector<int> groups{0, 1, -1, 2};
    const std::vector<int> active{0, 1, 2, 3};
    CHECK(v_groups_by_active(groups, active, 4, 3) == groups);
}

TEST_CASE("compaction: active rows follow active_to_orig")
{
    // rows 1 and 3 compacted out (islanded / skipped); groups of rows 0, 2, 4
    const std::vector<int> groups{5, 0, 1, 2, -1};
    const std::vector<int> active{0, 2, 4};
    CHECK((v_groups_by_active(groups, active, 5, 6) == std::vector<int>{5, 1, -1}));
}

TEST_CASE("group -1 (the base case) is kept")
{
    const std::vector<int> groups{-1, -1};
    CHECK((v_groups_by_active(groups, {1, 0}, 2, 1) == std::vector<int>{-1, -1}));
}

TEST_CASE("no active row")
{
    CHECK(v_groups_by_active({0, 0}, {}, 2, 1).empty());
}

TEST_CASE("a group out of range throws")
{
    CHECK_THROWS(v_groups_by_active({0, 3}, {0, 1}, 2, 3));   // 3 >= n_groups
    CHECK_THROWS(v_groups_by_active({0, -2}, {0, 1}, 2, 3));  // below -1
}

TEST_CASE("an out-of-range group of a compacted row is not checked")
{
    // only active rows start from a group: row 1 is not active
    CHECK((v_groups_by_active({0, 7}, {0}, 2, 1) == std::vector<int>{0}));
}

TEST_CASE("a group list of the wrong size throws")
{
    CHECK_THROWS(v_groups_by_active({0, 0, 0}, {0, 1}, 2, 1));
}

TEST_CASE("an active row outside the original rows throws")
{
    CHECK_THROWS(v_groups_by_active({0, 0}, {0, 2}, 2, 1));
}

GPUSIM2GRID_TEST_MAIN(false)
