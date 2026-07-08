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
// GPU base case and is trusted as already solved: AcPfNrState's presolved_v
// fast path validates ||F(V0)||_inf, factorizes J at V0 once, and skips the NR
// loop entirely — V stays bit-identical to V0 (see acpf_nr.cu §4.6). With
// false, the GPU re-solves the base case in max_iter_base iterations from that
// same seed.
//
// This header is CUDA-free and only depends on the (CUDA-free) session headers
// plus lightsim2grid's LSGrid.hpp, so it compiles with the host C++ compiler.
// =============================================================================

#ifndef LS2G_BRIDGE_HPP
#define LS2G_BRIDGE_HPP

#include <memory>
#include <tuple>

#include <LSGrid.hpp>  // ls2g::LSGrid (from lightsim2grid_core)

#include "contingency_analysis_session.hpp"
#include "injection_sweep_session.hpp"
#include "acpf_nr.hpp"      // AcPfNrSession
#include "ledger_data.hpp"  // LedgerData

// Build a LedgerData (augmented-J description) from a solved LSGrid: the J
// sparsity skeleton (get_J_solver), the NRLedger row/col maps, and the
// MultiSlack slack_col + slack weights. All in solver bus numbering. Empty /
// slack_col=-1 reproduces the feature-free system.
//
// When presolved_v is true (the caller is about to trust the grid's own V as
// already converged -- init_from_n_powerflow), this ALSO:
//   1. Verifies the grid is actually solved via LSGrid::check_solution() on
//      its own get_V_solver() -- throws std::runtime_error if the Kirchhoff
//      mismatch's inf-norm exceeds tol. Runs unconditionally whenever
//      presolved_v is requested, regardless of whether any extension is
//      active; timed into the returned LedgerData::t_ground_truth_check_ms.
//   2. If MultiSlack and/or VoltageControl is active, populates
//      slack_absorbed_gt/vc_q_gt from LSGrid::get_slack_absorbed_solver()/
//      get_controller_q_solver() -- lightsim2grid's own converged extension
//      state, letting AcPfNrState's presolved_v fast path seed it directly
//      instead of deriving it via a cuDSS solve.
LedgerData extract_ledger_data(const ls2g::LSGrid& grid,
                               bool presolved_v = false, double tol = 0.0);

// Build a single-system AcPfNrSession from a solved LSGrid, solving the same
// augmented system lightsim2grid does (distributed slack / future extensions).
//
// When init_from_n_powerflow=true (default) the CPU-converged get_V_solver()
// is trusted as already solved: one fill_F/fill_J/FACTORIZE validates
// ||F(V0)||_inf and prepares J's factors, but the GPU NR loop never runs and
// V stays bit-identical to V0 (see AcPfNrState's presolved_v path). When
// false, the GPU runs up to max_iter iterations from that same V0 seed.
std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    int    max_iter,
    double tol,
    int    device,
    bool   init_from_n_powerflow = true,
    // DEBUG ONLY -- see AcPfNrState's own doc (acpf_nr_state.cuh). No-op
    // unless init_from_n_powerflow=true. Returns a session whose get_J()/
    // get_F() expose J(V0) and the RAW F(V0, state=0) (no cuDSS-based state
    // correction, no throwing residual check), for an external solver to
    // redo that correction step independently of cuDSS on the exact same
    // data.
    bool   diag_stop_before_state_correction = false,
    // CUDSS_CONFIG_REORDERING_ALG / CUDSS_CONFIG_MATCHING_ALG /
    // CUDSS_CONFIG_PIVOT_EPSILON_ALG choices; see AcPfNrState's own doc.
    // Ctor-time only -- ANALYSIS runs once and is never rerun for this
    // session.
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
    // Opt-in diagnostic (see AcPfNrState's own doc): force the pre-ground-truth
    // cuDSS-solve derivation of slack_absorbed/vc_q even when lightsim2grid's
    // own converged values are available. No-op unless init_from_n_powerflow.
    bool   debug_base_case = false);

// Same as make_acpf_session_from_lsgrid, but with a caller-supplied Sbus
// (solver numbering) instead of the grid's own Sbus. The ledger structure is
// still read off the grid; only the numeric injections differ. Used by the
// differentiable power-flow path, where Sbus is a torch leaf tensor.
std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid_with_sbus(
    const ls2g::LSGrid& grid,
    Eigen::Ref<const CplxVect> Sbus,
    int    max_iter,
    double tol,
    int    device,
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default);

// Build a ContingencyAnalysisSession from a solved LSGrid (branch data set).
// When compute_limit_violations=true, also pulls bus/branch limits off the
// grid (extract_limits_from_lsgrid below) and enables the session's fused
// on-device violation check.
std::shared_ptr<ContingencyAnalysisSession>
make_ca_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    bool   compute_limit_violations = false,
    // Single source of truth for base_state_'s AND the batch solver's cuDSS
    // config (see ContingencyAnalysisSession's own doc).
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
    // Opt-in diagnostic forwarded to base_state_ -- see AcPfNrState's own doc.
    bool   debug_base_case = false);

// Extract compute_limit_violations limits off a solved LSGrid: bus voltage
// limits (kV, relabeled from grid-model to AC-solver bus numbering via
// id_me_to_ac_solver_numpy(), same map used for branch endpoints) and branch
// current limits (kA, lines-then-trafos, straight concat -- a bulk C++
// accessor exists on the line/trafo containers, unlike the Python bindings).
// NaN = not configured (matches lightsim2grid's convention). n_bus_solver
// must be the session's n_bus (Ybus_solver dimension) -- passed in rather
// than re-derived since every caller already has it on hand.
// Returns (bus_vmin_kv, bus_vmax_kv, limit_a1_ka, limit_a2_ka).
std::tuple<RealVect, RealVect, RealVect, RealVect>
extract_limits_from_lsgrid(const ls2g::LSGrid& grid, int n_bus_solver);

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
    bool   with_branch_data,
    // Single source of truth for base_state_'s AND the batch solver's cuDSS
    // config (see InjectionSweepSession's own doc).
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
    // Opt-in diagnostic forwarded to base_state_ -- see AcPfNrState's own doc.
    bool   debug_base_case = false);

#endif  // LS2G_BRIDGE_HPP
