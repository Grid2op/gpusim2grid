// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ledger_extend.hpp — host-side STRUCTURAL extensions of a LedgerData
// =============================================================================
//
// The J skeleton a LedgerData carries is, by default, exactly lightsim2grid's
// own (read verbatim off the solved LSGrid). Two batch features need the GPU
// batch to solve a system whose pattern is a strict SUPERSET of that skeleton
// -- every extra entry is filled per slot by value, so one shared cuDSS
// analysis still serves the whole batch:
//
//   • reserve_stranded_controller_slots  (lightsim2grid PR #192 parity)
//       one structural zero at (v_row, q_col) for every singleton (one-
//       controller) GEN VoltageControl group, so handle_disconnected_grid can
//       repurpose that group's bordered voltage row into "Q_c == 0" when a
//       contingency strands the controller's own bus. An SVC's slot already
//       exists (it carries the slope); a lone remote-regulating generator's
//       does not unless lightsim2grid was told set_may_mask_voltage_control.
//
//   • add_switchable_vm_buses            (lightsim2grid PR #193 parity)
//       one Vm column + one Q equation, with the full dS pattern, for every
//       PV bus whose local voltage-regulating generators a ScenarioSweep row
//       may disconnect. Rows where the bus is still PV identity-pin that Q
//       row (dVm = 0); rows where it lost its last controller solve it as PQ.
//       Appended at the END of the ledger (indices dim_J, dim_J+1, ...) so no
//       existing index moves -- which is why the VoltageControl custom rows
//       must first be frozen explicitly (materialize_vc_custom_rows) rather
//       than reconstructed as "the last n_controllers rows".
//
// All three are idempotent and host-only (no CUDA), so a session can re-run
// them on a stored copy of the base ledger whenever its switchable set changes.
// =============================================================================

#ifndef LEDGER_EXTEND_HPP
#define LEDGER_EXTEND_HPP

#include <vector>

#include "dtypes.hpp"
#include "ledger_data.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"

// Freeze the VoltageControl custom-row indices (vc_v_rows / vc_share_rows) from
// the CURRENT dim_J using the legacy "last n_controllers rows, v_row then the
// count-1 sharing rows per group" rule. No-op when already populated or when
// the ledger carries no VoltageControl. Must run BEFORE any row is appended.
void materialize_vc_custom_rows(LedgerData& ld);

// Insert the (v_row, q_col_first) entry for every singleton GEN group whose
// slot is missing (structure only; value 0 in the normal case, 1 when the
// group is stranded -- see acpf_nr.cu / build_mask_entries). Calls
// materialize_vc_custom_rows first.
void reserve_stranded_controller_slots(LedgerData& ld);

// Append a Vm column + a Q equation for every bus in `buses` (sorted, unique,
// AC-solver numbering) that does not already own one, with the dS pattern
// derived from Ybus_rm's structure (the same RowMajor Ybus AcPfNrState's
// build_scatter_maps_aug iterates). Records them in ld.switchable_vm_buses.
// Calls materialize_vc_custom_rows first. Throws on an out-of-range bus.
void add_switchable_vm_buses(
    LedgerData&                                                     ld,
    const std::vector<int>&                                         buses,
    const Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor>&    Ybus_rm);

// Position of (row, col) in the ledger's CSR skeleton, -1 if absent.
int ledger_find_J_pos(const LedgerData& ld, int row, int col);

#endif  // LEDGER_EXTEND_HPP
