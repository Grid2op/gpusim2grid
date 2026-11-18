// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/ls2g_outer_model.hpp -- lightsim2grid's outer loops as an outer::OuterHook.
//
// Built by the bridge off a solved NROuter_* grid (its loops, their parameters,
// the driver's): a host port of OuterLoopAlgo::compute_pf (the passes, the
// "last unstable" stop, the unrealistic-voltage check) and of its four default
// loops -- DistributedSlack, AcHvdcAcEmulationLimits, VoltageMonitoring,
// ReactiveLimits -- deciding on what the device gathered from each converged
// row. The plans and checks are lightsim2grid's own (BusQCheck, HvdcPCheck,
// SvcStandbyCheck, slack_redistribution::distribute), called on a copy of the
// grid. Compiled only with the bridge (it needs lightsim2grid's headers).
// =============================================================================

#ifndef GPUSIM2GRID_LS2G_OUTER_MODEL_HPP
#define GPUSIM2GRID_LS2G_OUTER_MODEL_HPP

#include <memory>
#include <string>
#include <vector>

#include "outer_hook.hpp"
#include "device_tables.hpp"

namespace ls2g { class LSGrid; }
struct LedgerData;

namespace outer {

struct Ls2gModelOptions {
    // keep each row's decision trace (RowResult::decisions)
    bool keep_trace = false;
    // every row resumes from where the grid's own (last) solve left its loops --
    // lightsim2grid's batch with init_from_n_powerflow: the targets they moved (but
    // where the row sets its own), the buses they froze (frozen again at the row's own
    // limit), the SVCs switched on, the hvdc regimes. Needs a lightsim2grid exposing
    // the loops' state (LS2G_HAS_OUTER_LOOP_STATE_ACCESSORS).
    bool resume_from_n = false;
};

// whether the lightsim2grid built against exposes the loops' state (resume_from_n)
bool ls2g_outer_resume_available();

// One feature of a grid's outer loops, as the model sees it (docs/outer_loops_on_device.md,
// §4.2): how many of it the grid has (0: none) and whether outer_loops_backend='device'
// implements it. A feature the grid has and the device does not implement refuses 'device'.
struct DeviceFeature {
    std::string name;
    int count = 0;
    bool supported = false;
};

// The features of `grid`'s outer loops (the model built on `ledger` with `options`), every
// one listed, those the grid lacks with count 0. Read off the model's own plans, so the
// census sees exactly what the host model runs.
// The device model of `grid` (outer_loops_backend='device', outer/device_outer_model.hpp): a
// host model kept for the loads and the tables, the decisions in kernels.
std::shared_ptr<OuterHook> make_device_outer_model(const ls2g::LSGrid& grid,
                                                   const LedgerData& ledger,
                                                   const Ls2gModelOptions& options);

std::vector<DeviceFeature> outer_device_census(const ls2g::LSGrid& grid,
                                               const LedgerData& ledger,
                                               const Ls2gModelOptions& options);

// The model of `grid` (solved, its algorithm an NROuter_* one), on the ledger the
// session solves (`ledger`, extracted from that same grid). Throws when the grid
// carries a loop this port does not have.
std::shared_ptr<OuterHook> make_ls2g_outer_model(const ls2g::LSGrid& grid,
                                                 const LedgerData& ledger,
                                                 const Ls2gModelOptions& options);

}  // namespace outer

#endif  // GPUSIM2GRID_LS2G_OUTER_MODEL_HPP
