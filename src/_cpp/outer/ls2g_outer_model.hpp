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

#include "outer_hook.hpp"

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

// The model of `grid` (solved, its algorithm an NROuter_* one), on the ledger the
// session solves (`ledger`, extracted from that same grid). Throws when the grid
// carries a loop this port does not have.
std::shared_ptr<OuterHook> make_ls2g_outer_model(const ls2g::LSGrid& grid,
                                                 const LedgerData& ledger,
                                                 const Ls2gModelOptions& options);

}  // namespace outer

#endif  // GPUSIM2GRID_LS2G_OUTER_MODEL_HPP
