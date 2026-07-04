// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ls2g_bridge.hpp  —  zero-copy construction from a solved lightsim2grid LSGrid
// =============================================================================
//
// These factories build a gpusim2grid batch session directly from a *solved*
// lightsim2grid LSGrid, pulling Ybus / Sbus / V / pv / pq / slack and the
// π-model branch admittances straight off the C++ object — no scipy CSR
// marshalling, no recomputation of physics.
//
// lightsim2grid owns the base-case (N) power flow on the CPU. With
// init_from_n_powerflow=true the converged voltage (get_V_solver()) seeds the
// GPU base case and a single GPU NR step factorizes J at the right operating
// point (max_iter_base = 1). With false, the GPU re-solves the base case in
// max_iter_base iterations from that same seed.
//
// This header is CUDA-free and only depends on the (CUDA-free) session headers
// plus lightsim2grid's LSGrid.hpp, so it compiles with the host C++ compiler.
// =============================================================================

#ifndef LS2G_BRIDGE_HPP
#define LS2G_BRIDGE_HPP

#include <memory>

#include <LSGrid.hpp>  // ls2g::LSGrid (from lightsim2grid_core)

#include "contingency_analysis_session.hpp"
#include "injection_sweep_session.hpp"

// Build a ContingencyAnalysisSession from a solved LSGrid (branch data set).
std::shared_ptr<ContingencyAnalysisSession>
make_ca_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device);

// Build an InjectionSweepSession from a solved LSGrid (branch data set when
// with_branch_data=true so compute_flows() works without extra setup).
std::shared_ptr<InjectionSweepSession>
make_is_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    bool   with_branch_data);

#endif  // LS2G_BRIDGE_HPP
