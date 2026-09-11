// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef MASK_CONFIG_BUILDER_CUH
#define MASK_CONFIG_BUILDER_CUH

// =============================================================================
// mask_config_builder.cuh
//
// build_mask_config — the handle_disconnected_grid MaskConfig of a session,
// built once from its base AcPfNrState (per-bus identity-row metadata, angle
// reference, VC row positions) and the ledger (HVDC ends, VoltageControl
// groups). Shared by ContingencyAnalysisSession and ScenarioSweepSession so
// the classification of a masked bus lives in exactly one place:
//
//   is_reference_bus       : the bus(es) with no theta column anchor the angle
//                            and cannot be frozen → skip when stranded.
//   is_hard_controller_bus : HVDC converter ends and every VoltageControl
//                            REGULATED bus -- their feature equations
//                            reference the live block with no value-only
//                            fallback → skip when stranded.
//   vc_*                   : group topology, so a masked CONTROLLER bus can be
//                            classified per group (see compute_component_masks).
// =============================================================================

#include <vector>

#include "acpf_nr_state.cuh"
#include "contingency_analysis_helper.hpp"   // MaskConfig
#include "ledger_data.hpp"

inline MaskConfig build_mask_config(const AcPfNrState& base, const LedgerData* ledger)
{
    MaskConfig cfg;
    const int n_bus = base.n_bus;
    cfg.row_info.p_row      = base.h_p_row_of_bus;
    cfg.row_info.q_row      = base.h_q_row_of_bus;
    cfg.row_info.p_diag_pos = base.h_p_diag_pos;
    cfg.row_info.q_diag_pos = base.h_q_diag_pos;

    cfg.is_reference_bus.assign(static_cast<size_t>(n_bus), 0);
    for (int b = 0; b < n_bus; ++b)
        if (base.h_theta_col_of_bus[static_cast<size_t>(b)] < 0)
            cfg.is_reference_bus[static_cast<size_t>(b)] = 1;

    cfg.is_hard_controller_bus.assign(static_cast<size_t>(n_bus), 0);
    if (ledger != nullptr) {
        auto mark = [&](int b) {
            if (b >= 0 && b < n_bus) cfg.is_hard_controller_bus[static_cast<size_t>(b)] = 1;
        };
        for (int b : ledger->hvdc_bus1)  mark(b);
        for (int b : ledger->hvdc_bus2)  mark(b);
        for (int b : ledger->vc_reg_bus) mark(b);

        cfg.vc_bus            = ledger->vc_bus;
        cfg.vc_group          = ledger->vc_group;
        cfg.vc_grp_count      = ledger->vc_grp_count;
        cfg.vc_vrow           = base.h_vc_vrow;
        cfg.vc_vrow_qcol_pos  = base.h_vc_vrow_qcol_pos;
        cfg.vc_vrow_vmcol_pos = base.h_vc_vrow_vmcol_pos;
    }
    return cfg;
}

#endif  // MASK_CONFIG_BUILDER_CUH
