// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef OUTER_ROW_SEED_HPP
#define OUTER_ROW_SEED_HPP

// =============================================================================
// outer/outer_row_seed.hpp
//
// Where an outer-loop row starts when its loops start from scratch: the grid's
// voltages with every voltage source's bus at its set-point
// (BaseBatchSolverSynch::_solve_n_case's seed_vm_keeping -- with
// set_keep_vinit_at_group_controlled_buses, the buses a voltage-control group
// holds keep the magnitude they start from). Shared by extract_outer_inputs
// (init_from_n_powerflow=False: every row starts there) and the host model
// (init_from_n_powerflow=True: the rows that do not resume, see
// Ls2gOuterModel::fresh_vm_), so both seed the same buses to the same values.
// =============================================================================

#include <vector>

#include <LSGrid.hpp>

namespace outer {

inline void seed_rows_vm(const ls2g::LSGrid& grid, Eigen::Ref<ls2g::CplxVect> V)
{
    std::vector<int> held;
    if (grid.get_keep_vinit_at_group_controlled_buses())
        held = grid.get_ac_layout().voltage_control.group_controlled_solver_buses(
            grid.get_ac_layout().id_me_to_solver);
    std::vector<ls2g::cplx_type> kept;
    kept.reserve(held.size());
    for (int b : held) kept.push_back(V(b));
    // LSGrid's own seed: every voltage source, not only the generators (a station or a
    // storage unit a loop froze in the grid's solve is off its set-point there)
    const auto& me_to_solver = grid.get_ac_layout().id_me_to_solver;
    grid.get_generators().set_vm(V, me_to_solver);
    grid.get_dclines().set_vm(V, me_to_solver);
    grid.get_svcs().set_vm(V, me_to_solver);
    grid.get_storages().set_vm(V, me_to_solver);
    for (size_t k = 0; k < held.size(); ++k) V(held[k]) = kept[k];
}

}  // namespace outer

#endif  // OUTER_ROW_SEED_HPP
