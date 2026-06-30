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
    std::vector<int> q_col_of_bus;     // controller reactive-unknown columns (VoltageControl)

    // ---- MultiSlack (distributed slack in the Jacobian) — optional -----------
    // slack_col >= 0 enables the extension. slack_weights has size n_bus, sums to
    // 1, and is nonzero only at slack participant buses (each of which owns a P
    // equation in the augmented ledger). The slack_absorbed initial value is
    // Re(sum Sbus), computed by AcPfNrState from the per-system Sbus.
    int                 slack_col = -1;
    std::vector<double> slack_weights;   // size n_bus (empty when inactive)

    bool has_multislack() const { return slack_col >= 0; }
    int  nnz_J()          const { return static_cast<int>(J_inner.size()); }

    // ---- HVDC angle-droop ("AC emulation") — optional ------------------------
    // Per CONNECTED droop-enabled HVDC line, in solver bus numbering and per-unit
    // (HvdcDroopSolverData read off the grid via fill_hvdc_droop_data_from_grid).
    // status is the frozen saturation regime (0 linear, +1 sat 1→2, -1 sat 2→1).
    // The extension claims no row/column; AcPfNrState derives the end P rows /
    // theta columns / feature J positions from the ledger maps + skeleton.
    std::vector<int>    hvdc_bus1, hvdc_bus2, hvdc_status;
    std::vector<double> hvdc_p0, hvdc_k, hvdc_lf1, hvdc_lf2, hvdc_r,
                        hvdc_pmax12, hvdc_pmax21;

    int  n_hvdc()       const { return static_cast<int>(hvdc_bus1.size()); }
    bool has_hvdc()     const { return !hvdc_bus1.empty(); }

    // ---- VoltageControl (remote gen + SVC) — optional ------------------------
    // Bordered formulation (VoltageControlSolverData, solver numbering, pu). Per
    // group of N controllers regulating reg_bus at v_set: N q-unknown columns,
    // 1 voltage-constraint custom row, N-1 reactive-sharing custom rows. Only the
    // physics is carried here; AcPfNrState reconstructs the claimed columns
    // (q_col/vm_col from the ledger maps) and the custom rows (the last
    // n_controllers rows, allocated v_row then share_rows per group) + feature
    // J positions. kind: 0 = GEN, 1 = SVC (sloped).
    std::vector<int>    vc_bus, vc_kind, vc_group;       // per controller
    std::vector<double> vc_slope, vc_weight;             // per controller
    std::vector<int>    vc_reg_bus, vc_grp_start, vc_grp_count;  // per group
    std::vector<double> vc_v_set;                        // per group

    int  vc_n_controllers() const { return static_cast<int>(vc_bus.size()); }
    int  vc_n_groups()      const { return static_cast<int>(vc_reg_bus.size()); }
    bool has_voltage_control() const { return !vc_bus.empty(); }
};

#endif  // LEDGER_DATA_HPP
