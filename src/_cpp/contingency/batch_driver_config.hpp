// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BATCH_DRIVER_CONFIG_HPP
#define BATCH_DRIVER_CONFIG_HPP

#include "../contingency_analysis_helper.hpp"   // ContingencySolverType
#include "../reordering_alg.hpp"
#include "../matching_alg.hpp"
#include "../pivot_epsilon_alg.hpp"
#include "row_status.hpp"                  // BatchScheduling

// =============================================================================
// BatchDriverConfig — the construction-time shape of a session's live batch
// driver (BatchPfDriver). A session's run() compares its current settings
// against the snapshot taken when the driver was built: any difference means
// the driver cannot be reused (a "cold" run rebuilds it, with a new cuDSS
// ANALYSIS); otherwise run() keeps it and only swaps what changed (the
// source, or the injections). The config members are plain read/write
// attributes on the Python side, so a snapshot comparison is the only robust
// way to notice a change. A field a workload has no use for stays at its
// default (e.g. mask_mode on an injection sweep).
// =============================================================================
struct BatchDriverConfig {
    int    n_scenarios     = -1;   // rows of the batch (contingencies / scenarios)
    int    batch_size      = 0;
    int    refactor_period = 1;
    ContingencySolverType strategy = ContingencySolverType::DirectRefactorEvery;
    ReorderingAlg   reordering_alg    = ReorderingAlg::Default;
    MatchingAlg     matching_alg      = MatchingAlg::None;
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default;
    bool   scaling_max_voltage_change = false;
    double max_dVa = 0.5, max_dVm = 0.1;
    bool   mask_mode            = false;   // handle_disconnected_grid
    bool   fixed_batch_capacity = false;
    int    base_state_generation = -1;
    int    rows_generation       = -1;     // ContingencyAnalysisSession: contingency list version
    // The schedule: a chunked run after a continuous one must not take the
    // hot path on a capacity built for the other.
    BatchScheduling scheduling = BatchScheduling::Chunked;

    bool operator==(const BatchDriverConfig& o) const {
        return n_scenarios == o.n_scenarios && batch_size == o.batch_size
            && refactor_period == o.refactor_period && strategy == o.strategy
            && reordering_alg == o.reordering_alg && matching_alg == o.matching_alg
            && pivot_epsilon_alg == o.pivot_epsilon_alg
            && scaling_max_voltage_change == o.scaling_max_voltage_change
            && max_dVa == o.max_dVa && max_dVm == o.max_dVm
            && mask_mode == o.mask_mode && fixed_batch_capacity == o.fixed_batch_capacity
            && base_state_generation == o.base_state_generation
            && rows_generation == o.rows_generation
            && scheduling == o.scheduling;
    }
    bool operator!=(const BatchDriverConfig& o) const { return !(*this == o); }
};

#endif  // BATCH_DRIVER_CONFIG_HPP
