// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ledger_data.hpp — host-side description of the augmented NR Jacobian
// =============================================================================
//
// Read off a solved lightsim2grid LSGrid (via the bridge) or built trivially
// from pv/pq for a feature-free grid. Consumed by AcPfNrState to set up the GPU
// augmented system: the J sparsity skeleton (structure only — values are
// recomputed on the GPU), the NRLedger row/column maps that locate each bus'
// P/Q equation and theta/Vm unknown, and the optional per-extension feature
// data (Phase 2: MultiSlack distributed slack).
//
// Pure host types (std::vector / int) — included by both the host bridge .cpp
// and the nvcc translation units. No CUDA / Eigen dependency.
// =============================================================================

#ifndef LEDGER_DATA_HPP
#define LEDGER_DATA_HPP

#include <vector>

struct LedgerData {
    int n_bus = 0;
    int dim_J = 0;   // augmented Jacobian dimension (== NRLedger size)

    // Augmented J sparsity skeleton in RowMajor CSR (structure only). The GPU
    // uploads these as d_J_outer / d_J_inner and resolves the dS scatter maps +
    // feature positions against this exact ordering; the lightsim2grid J values
    // are irrelevant (fill_J recomputes them).
    std::vector<int> J_outer;   // size dim_J + 1
    std::vector<int> J_inner;   // size nnz_J

    // NRLedger bus-keyed maps (size n_bus, -1 when the bus owns no such row/col).
    std::vector<int> p_row_of_bus;
    std::vector<int> q_row_of_bus;
    std::vector<int> theta_col_of_bus;
    std::vector<int> vm_col_of_bus;

    // ---- MultiSlack (distributed slack in the Jacobian) — optional -----------
    // slack_col >= 0 enables the extension. slack_weights has size n_bus, sums to
    // 1, and is nonzero only at slack participant buses (each of which owns a P
    // equation in the augmented ledger). The slack_absorbed initial value is
    // Re(sum Sbus), computed by AcPfNrState from the per-system Sbus.
    int                 slack_col = -1;
    std::vector<double> slack_weights;   // size n_bus (empty when inactive)

    bool has_multislack() const { return slack_col >= 0; }
    int  nnz_J()          const { return static_cast<int>(J_inner.size()); }
};

#endif  // LEDGER_DATA_HPP
