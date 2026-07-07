// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ls2g_bridge.cpp  —  extract solved state from an LSGrid into a GPU session
// =============================================================================

#include "ls2g_bridge.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// Concatenate two complex vectors (lines then trafos ordering).
CplxVect concat_cplx(Eigen::Ref<const ls2g::CplxVect> a,
                     Eigen::Ref<const ls2g::CplxVect> b)
{
    CplxVect out(a.size() + b.size());
    out.head(a.size()) = a;
    out.tail(b.size()) = b;
    return out;
}

// Concatenate two branch bus-id vectors (global ids), relabeling each to the
// AC-solver bus numbering used by Ybus_solver. The relabel map is global→solver
// (id_me_to_ac_solver); isolated buses map to a negative id. When the grid has
// no relabeling (fully connected, identity), this is a straight copy.
Eigen::VectorXi concat_busids_to_solver(
    Eigen::Ref<const ls2g::IntVect> a,
    Eigen::Ref<const ls2g::IntVect> b,
    const std::vector<int>&         me_to_solver)
{
    const int na = static_cast<int>(a.size());
    const int nb = static_cast<int>(b.size());
    Eigen::VectorXi out(na + nb);

    auto relabel = [&](int global_id) -> int {
        if (me_to_solver.empty()) return global_id;  // identity numbering
        if (global_id < 0 || global_id >= static_cast<int>(me_to_solver.size()))
            return global_id;
        return me_to_solver[global_id];
    };

    for (int i = 0; i < na; ++i) out[i]      = relabel(a[i]);
    for (int i = 0; i < nb; ++i) out[na + i] = relabel(b[i]);
    return out;
}

// Pull the eight set_branch_data() arguments off the grid (lines then trafos).
struct BranchData {
    Eigen::VectorXi branch_from, branch_to;
    CplxVect        yff, yft, ytf, ytt;
    RealVect        bus_vn_kv;
    double          sn_mva;
};

BranchData extract_branch_data(const ls2g::LSGrid& grid)
{
    const auto& lines  = grid.get_powerlines_as_data();
    const auto& trafos = grid.get_trafos_as_data();

    // Relabel branch endpoints (global → AC-solver) so they index Ybus_solver.
    std::vector<int> me_to_solver = grid.id_me_to_ac_solver_numpy();

    BranchData bd;
    bd.branch_from = concat_busids_to_solver(
        lines.get_bus_id_side_1_numpy(), trafos.get_bus_id_side_1_numpy(),
        me_to_solver);
    bd.branch_to = concat_busids_to_solver(
        lines.get_bus_id_side_2_numpy(), trafos.get_bus_id_side_2_numpy(),
        me_to_solver);
    bd.yff = concat_cplx(lines.yac_eff_11(), trafos.yac_eff_11());
    bd.yft = concat_cplx(lines.yac_eff_12(), trafos.yac_eff_12());
    bd.ytf = concat_cplx(lines.yac_eff_21(), trafos.yac_eff_21());
    bd.ytt = concat_cplx(lines.yac_eff_22(), trafos.yac_eff_22());
    // TODO(bug): bus_vn_kv is left in grid-MODEL bus numbering, unlike
    // branch_from/branch_to just above (relabeled via me_to_solver) and unlike
    // bus_vmin_kv/bus_vmax_kv in extract_limits() (also relabeled). Whenever
    // id_me_to_ac_solver is not the identity -- e.g. isolated buses excluded
    // under KLU/the augmented multi-slack system -- this mismatches V/
    // branch_from in size (crash) or, if sizes coincide, silently pairs the
    // wrong nominal voltage with the wrong bus in compute_branch_flows_cpu /
    // get_violations_n. Needs the same me_to_solver remap applied here.
    bd.bus_vn_kv = grid.get_bus_vn_kv();
    bd.sn_mva    = const_cast<ls2g::LSGrid&>(grid).get_sn_mva();
    return bd;
}

// Convert an ls2g IntVect (Eigen) to a std::vector<int>.
std::vector<int> to_int_vector(const ls2g::IntVect& v)
{
    return std::vector<int>(v.data(), v.data() + v.size());
}

// compute_limit_violations limits pulled off the grid (lines-then-trafos for
// branch limits; solver bus numbering for bus limits), for
// ContingencyAnalysisSession::set_limits().
struct LimitData {
    RealVect bus_vmin_kv, bus_vmax_kv;    // solver numbering, size n_bus_solver
    RealVect limit_a1_ka, limit_a2_ka;    // lines-then-trafos, size n_branches
};

// n_bus_solver is passed in (rather than re-derived from get_Ybus_solver(),
// which is non-const and returns a full copy) since every caller already has
// it on hand: make_ca_session_from_lsgrid from its own Ybus, and the
// standalone pybind wrapper from the already-constructed session's n_bus().
LimitData extract_limits(const ls2g::LSGrid& grid, int n_bus_solver)
{
    const auto& lines  = grid.get_powerlines_as_data();
    const auto& trafos = grid.get_trafos_as_data();

    LimitData ld;

    // Branch limits: bulk C++ accessor exists (TwoSidesContainer_rxh_A::
    // get_limit_a1_ka/a2_ka) -- straight concat, same head/tail pattern as
    // concat_cplx above. NaN entries ("not configured") pass through as-is.
    {
        Eigen::Ref<const ls2g::RealVect> l1_lines  = lines.get_limit_a1_ka();
        Eigen::Ref<const ls2g::RealVect> l1_trafos = trafos.get_limit_a1_ka();
        ld.limit_a1_ka.resize(l1_lines.size() + l1_trafos.size());
        ld.limit_a1_ka.head(l1_lines.size())  = l1_lines;
        ld.limit_a1_ka.tail(l1_trafos.size()) = l1_trafos;

        Eigen::Ref<const ls2g::RealVect> l2_lines  = lines.get_limit_a2_ka();
        Eigen::Ref<const ls2g::RealVect> l2_trafos = trafos.get_limit_a2_ka();
        ld.limit_a2_ka.resize(l2_lines.size() + l2_trafos.size());
        ld.limit_a2_ka.head(l2_lines.size())  = l2_lines;
        ld.limit_a2_ka.tail(l2_trafos.size()) = l2_trafos;
    }

    // Bus limits: grid.get_bus_vmin_kv()/get_bus_vmax_kv() return an EMPTY
    // array when never configured (not a NaN-filled n_bus array) -- and are
    // in grid-model bus numbering, requiring the same model->solver relabel
    // ls2g_bridge.cpp already applies to branch endpoints
    // (concat_busids_to_solver). Buses with no model-side entry at all (empty
    // input) or isolated (negative solver id) are reported as NaN =
    // "not configured", matching the per-element NaN convention.
    ls2g::RealVect vmin_model = grid.get_bus_vmin_kv();
    ls2g::RealVect vmax_model = grid.get_bus_vmax_kv();
    std::vector<int> me_to_solver = grid.id_me_to_ac_solver_numpy();

    const eigen_real_type nan_val = std::numeric_limits<eigen_real_type>::quiet_NaN();
    ld.bus_vmin_kv = RealVect::Constant(n_bus_solver, nan_val);
    ld.bus_vmax_kv = RealVect::Constant(n_bus_solver, nan_val);
    if (vmin_model.size() > 0) {
        for (size_t grid_id = 0; grid_id < me_to_solver.size(); ++grid_id) {
            const int solver_id = me_to_solver[grid_id];
            if (solver_id >= 0 && solver_id < n_bus_solver) {
                ld.bus_vmin_kv(solver_id) = vmin_model(static_cast<Eigen::Index>(grid_id));
                ld.bus_vmax_kv(solver_id) = vmax_model(static_cast<Eigen::Index>(grid_id));
            }
        }
    }
    return ld;
}

}  // namespace

LedgerData extract_ledger_data(const ls2g::LSGrid& grid)
{
    LedgerData ld;

    // Augmented J sparsity skeleton in RowMajor CSR (structure only). get_J_solver
    // returns the solved augmented J in lightsim2grid's default (ColMajor) storage;
    // convert to RowMajor so outer = row (gpusim2grid's CSR convention).
    Eigen::SparseMatrix<eigen_real_type>                  J_cm = grid.get_J_solver();
    Eigen::SparseMatrix<eigen_real_type, Eigen::RowMajor> J    = J_cm;
    J.makeCompressed();
    ld.dim_J = static_cast<int>(J.rows());
    ld.J_outer.assign(J.outerIndexPtr(), J.outerIndexPtr() + ld.dim_J + 1);
    ld.J_inner.assign(J.innerIndexPtr(), J.innerIndexPtr() + J.nonZeros());

    // NRLedger bus→row/col maps (solver numbering, size n_bus, -1 absent).
    ld.p_row_of_bus     = to_int_vector(grid.get_p_to_J_row_solver());
    ld.q_row_of_bus     = to_int_vector(grid.get_q_to_J_row_solver());
    ld.theta_col_of_bus = to_int_vector(grid.get_theta_to_J_col_solver());
    ld.vm_col_of_bus    = to_int_vector(grid.get_vm_to_J_col_solver());
    ld.q_col_of_bus     = to_int_vector(grid.get_q_to_J_col_solver());
    ld.n_bus            = static_cast<int>(ld.p_row_of_bus.size());

    // NRLedger compact (bus, row/col) registration pair lists -- preserve every
    // registration (unlike the bus-keyed maps above, "last registration wins").
    // NRSystem's own residual assembly iterates these; the augmented-Jacobian
    // scatter/residual reconstruction below must match it exactly.
    ld.p_buses     = to_int_vector(grid.get_p_buses_solver());
    ld.p_rows      = to_int_vector(grid.get_p_rows_solver());
    ld.q_buses     = to_int_vector(grid.get_q_buses_solver());
    ld.q_rows      = to_int_vector(grid.get_q_rows_solver());
    ld.theta_buses = to_int_vector(grid.get_theta_buses_solver());
    ld.theta_cols  = to_int_vector(grid.get_theta_cols_solver());
    ld.vm_buses    = to_int_vector(grid.get_vm_buses_solver());
    ld.vm_cols     = to_int_vector(grid.get_vm_cols_solver());

    // MultiSlack: slack_col (-1 when distributed slack inactive) + slack weights.
    ld.slack_col = grid.get_slack_col_solver();
    if (ld.slack_col >= 0) {
        ls2g::RealVect sw = grid.get_slack_weights_solver();
        ld.slack_weights.assign(sw.data(), sw.data() + sw.size());
    }

    // HVDC angle-droop: pull the connected droop lines (solver numbering, pu) via
    // the same path the NRSystem's Hvdc extension uses. Empty when none.
    {
        ls2g::HvdcDroopSolverData h;
        ls2g::fill_hvdc_droop_data_from_grid(&grid, h, /*ac=*/true);
        const int nh = h.size();
        auto to_iv = [](const Eigen::VectorXi& v) {
            return std::vector<int>(v.data(), v.data() + v.size());
        };
        auto to_dv = [](const ls2g::RealVect& v) {
            return std::vector<double>(v.data(), v.data() + v.size());
        };
        if (nh > 0) {
            ld.hvdc_bus1   = to_iv(h.bus1);
            ld.hvdc_bus2   = to_iv(h.bus2);
            ld.hvdc_status = to_iv(h.status);
            ld.hvdc_p0     = to_dv(h.p0);
            ld.hvdc_k      = to_dv(h.k);
            ld.hvdc_lf1    = to_dv(h.lf1);
            ld.hvdc_lf2    = to_dv(h.lf2);
            ld.hvdc_r      = to_dv(h.r);
            ld.hvdc_pmax12 = to_dv(h.pmax12);
            ld.hvdc_pmax21 = to_dv(h.pmax21);
            ld.hvdc_connected1.assign(h.connected1.begin(), h.connected1.end());
            ld.hvdc_connected2.assign(h.connected2.begin(), h.connected2.end());
        }
    }

    // VoltageControl (remote-regulating generators + voltage-mode SVCs): pull the
    // bordered-block physics (solver numbering, pu). Empty when none active.
    {
        ls2g::VoltageControlSolverData v;
        grid.fill_voltage_control_solver_data(v, /*ac=*/true);
        auto to_iv = [](const Eigen::VectorXi& a) {
            return std::vector<int>(a.data(), a.data() + a.size());
        };
        auto to_dv = [](const ls2g::RealVect& a) {
            return std::vector<double>(a.data(), a.data() + a.size());
        };
        if (v.n_controllers() > 0) {
            ld.vc_bus       = to_iv(v.bus);
            ld.vc_kind      = to_iv(v.kind);
            ld.vc_group     = to_iv(v.group);
            ld.vc_slope     = to_dv(v.slope);
            ld.vc_weight    = to_dv(v.weight);
            ld.vc_reg_bus   = to_iv(v.reg_bus);
            ld.vc_grp_start = to_iv(v.grp_start);
            ld.vc_grp_count = to_iv(v.grp_count);
            ld.vc_v_set     = to_dv(v.v_set);
        }
    }
    return ld;
}

std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    int    max_iter,
    double tol,
    int    device,
    bool   init_from_n_powerflow,
    bool   diag_stop_before_state_correction,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg)
{
    auto& g = const_cast<ls2g::LSGrid&>(grid);
    Eigen::SparseMatrix<eigen_cplx_type> Ybus = g.get_Ybus_solver();
    if (Ybus.rows() == 0)
        throw std::runtime_error(
            "make_acpf_session_from_lsgrid: empty Ybus — has the grid been solved "
            "(ac_pf) before being handed to gpusim2grid?");

    CplxVect        V0    = grid.get_V_solver();
    CplxVect        Sbus  = grid.get_Sbus_solver();
    Eigen::VectorXi slack = grid.get_slack_ids_solver_numpy();
    RealVect        sw    = grid.get_slack_weights_solver();
    Eigen::VectorXi pv    = grid.get_pv_solver_numpy();
    Eigen::VectorXi pq    = grid.get_pq_solver_numpy();

    LedgerData ledger = extract_ledger_data(grid);

    return std::make_shared<AcPfNrSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq, max_iter, tol, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow,
        diag_stop_before_state_correction,
        reordering_alg, matching_alg);
}

std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid_with_sbus(
    const ls2g::LSGrid& grid,
    Eigen::Ref<const CplxVect> Sbus,
    int    max_iter,
    double tol,
    int    device,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg)
{
    // Same as make_acpf_session_from_lsgrid, but with a caller-supplied Sbus
    // (solver numbering) instead of the grid's own get_Sbus_solver(). Used by
    // the differentiable power-flow path: the ledger's structural content
    // (row/col maps, sparsity, HVDC/VC constant params, slack weights) depends
    // only on topology/control configuration, not on the numeric Sbus of the
    // solve that produced it, so it can be reused across many different Sbus
    // values — exactly like the injection-sweep path already does.
    auto& g = const_cast<ls2g::LSGrid&>(grid);
    Eigen::SparseMatrix<eigen_cplx_type> Ybus = g.get_Ybus_solver();
    if (Ybus.rows() == 0)
        throw std::runtime_error(
            "make_acpf_session_from_lsgrid_with_sbus: empty Ybus — has the grid "
            "been solved (ac_pf) before being handed to gpusim2grid?");
    if (Sbus.size() != Ybus.rows())
        throw std::runtime_error(
            "make_acpf_session_from_lsgrid_with_sbus: Sbus size does not match "
            "the solver bus count.");

    CplxVect        V0    = grid.get_V_solver();
    Eigen::VectorXi slack = grid.get_slack_ids_solver_numpy();
    RealVect        sw    = grid.get_slack_weights_solver();
    Eigen::VectorXi pv    = grid.get_pv_solver_numpy();
    Eigen::VectorXi pq    = grid.get_pq_solver_numpy();

    LedgerData ledger = extract_ledger_data(grid);

    return std::make_shared<AcPfNrSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq, max_iter, tol, device, &ledger,
        /*presolved_v=*/false, /*diag_stop_before_state_correction=*/false,
        reordering_alg, matching_alg);
}

std::shared_ptr<ContingencyAnalysisSession>
make_ca_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    bool   compute_limit_violations)
{
    // get_Ybus_solver() is non-const (returns a copy) — cast away constness;
    // we only read it.
    auto& g = const_cast<ls2g::LSGrid&>(grid);
    Eigen::SparseMatrix<eigen_cplx_type> Ybus = g.get_Ybus_solver();
    if (Ybus.rows() == 0)
        throw std::runtime_error(
            "make_ca_session_from_lsgrid: empty Ybus — has the grid been solved "
            "(ac_pf) before being handed to gpusim2grid?");

    CplxVect        V0    = grid.get_V_solver();
    CplxVect        Sbus  = grid.get_Sbus_solver();
    Eigen::VectorXi slack = grid.get_slack_ids_solver_numpy();
    RealVect        sw    = grid.get_slack_weights_solver();
    Eigen::VectorXi pv    = grid.get_pv_solver_numpy();
    Eigen::VectorXi pq    = grid.get_pq_solver_numpy();

    LedgerData ledger = extract_ledger_data(grid);
    auto session = std::make_shared<ContingencyAnalysisSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, max_iter_base, tol_base, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow);

    BranchData bd = extract_branch_data(grid);
    session->set_branch_data(bd.branch_from, bd.branch_to,
                             bd.yff, bd.yft, bd.ytf, bd.ytt,
                             bd.bus_vn_kv, bd.sn_mva);

    if (compute_limit_violations) {
        session->set_compute_limit_violations(true);
        LimitData ld = extract_limits(grid, static_cast<int>(Ybus.rows()));
        const int n_lines = static_cast<int>(grid.get_powerlines_as_data().nb());
        session->set_limits(ld.bus_vmin_kv, ld.bus_vmax_kv,
                            ld.limit_a1_ka, ld.limit_a2_ka, n_lines);
    }
    return session;
}

std::tuple<RealVect, RealVect, RealVect, RealVect>
extract_limits_from_lsgrid(const ls2g::LSGrid& grid, int n_bus_solver)
{
    LimitData ld = extract_limits(grid, n_bus_solver);
    return std::make_tuple(ld.bus_vmin_kv, ld.bus_vmax_kv, ld.limit_a1_ka, ld.limit_a2_ka);
}

std::shared_ptr<InjectionSweepSession>
make_is_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    bool   with_branch_data)
{
    auto& g = const_cast<ls2g::LSGrid&>(grid);
    Eigen::SparseMatrix<eigen_cplx_type> Ybus = g.get_Ybus_solver();
    if (Ybus.rows() == 0)
        throw std::runtime_error(
            "make_is_session_from_lsgrid: empty Ybus — has the grid been solved "
            "(ac_pf) before being handed to gpusim2grid?");

    CplxVect        V0    = grid.get_V_solver();
    CplxVect        Sbus  = grid.get_Sbus_solver();
    Eigen::VectorXi slack = grid.get_slack_ids_solver_numpy();
    RealVect        sw    = grid.get_slack_weights_solver();
    Eigen::VectorXi pv    = grid.get_pv_solver_numpy();
    Eigen::VectorXi pq    = grid.get_pq_solver_numpy();

    LedgerData ledger = extract_ledger_data(grid);
    auto session = std::make_shared<InjectionSweepSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, max_iter_base, tol_base, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow);

    if (with_branch_data) {
        BranchData bd = extract_branch_data(grid);
        session->set_branch_data(bd.branch_from, bd.branch_to,
                                 bd.yff, bd.yft, bd.ytf, bd.ytt,
                                 bd.bus_vn_kv, bd.sn_mva);
    }
    return session;
}
