// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// outer::check_device_features (outer/device_outer_model.hpp): the census of a grid's outer loops
// refuses outer_loops_backend='device' for the whole batch, naming every feature the grid has and
// the device does not implement. Host only.

#include <string>
#include <vector>

#include "check.hpp"
#include "outer/device_outer_model.hpp"

using outer::DeviceFeature;

namespace {

// the message check_device_features throws, "" when it does not
std::string refusal(const std::vector<DeviceFeature>& features)
{
    try {
        outer::check_device_features(features);
    } catch (const outer::DeviceNotImplemented& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST_CASE("every feature supported: no refusal")
{
    CHECK(refusal({{"loop ReactiveLimits", 1, true}, {"DistributedSlack units", 5, true}}).empty());
}

TEST_CASE("an unsupported feature the grid lacks: no refusal")
{
    CHECK(refusal({{"loop ReactiveLimits", 1, true}, {"a feature to come", 0, false}}).empty());
}

TEST_CASE("an unsupported feature the grid has: refused, naming it and its count")
{
    const std::string msg = refusal({{"loop ReactiveLimits", 1, true}, {"a feature to come", 2, false},
                                     {"another one", 3, false}});
    CHECK(msg.find("a feature to come (2)") != std::string::npos);
    CHECK(msg.find("another one (3)") != std::string::npos);
    CHECK(msg.find("outer_loops_backend='host'") != std::string::npos);
    // what the device runs is not listed
    CHECK(msg.find("loop ReactiveLimits") == std::string::npos);
}

GPUSIM2GRID_TEST_MAIN(false)
