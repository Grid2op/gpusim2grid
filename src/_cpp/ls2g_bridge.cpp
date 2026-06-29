// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ls2g_bridge.cpp  —  extract solved state from an LSGrid into a GPU session
// =============================================================================

#include "ls2g_bridge.hpp"

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
    bd.bus_vn_kv = grid.get_bus_vn_kv();
    bd.sn_mva    = const_cast<ls2g::LSGrid&>(grid).get_sn_mva();
    return bd;
}

// Convert an ls2g IntVect (Eigen) to a std::vector<int>.
std::vector<int> to_int_vector(const ls2g::IntVect& v)
{
    return std::vector<int>(v.data(), v.data() + v.size());
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
    ld.n_bus            = static_cast<int>(ld.p_row_of_bus.size());

    // MultiSlack: slack_col (-1 when distributed slack inactive) + slack weights.
    ld.slack_col = grid.get_slack_col_solver();
    if (ld.slack_col >= 0) {
        ls2g::RealVect sw = grid.get_slack_weights_solver();
        ld.slack_weights.assign(sw.data(), sw.data() + sw.size());
    }
    return ld;
}

std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    int    max_iter,
    double tol,
    int    device)
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
        Ybus, V0, Sbus, slack, sw, pv, pq, max_iter, tol, device, &ledger);
}

std::shared_ptr<ContingencyAnalysisSession>
make_ca_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device)
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

    const int base_iters = init_from_n_powerflow ? 1 : max_iter_base;

    auto session = std::make_shared<ContingencyAnalysisSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, base_iters, tol_base, device);

    BranchData bd = extract_branch_data(grid);
    session->set_branch_data(bd.branch_from, bd.branch_to,
                             bd.yff, bd.yft, bd.ytf, bd.ytt,
                             bd.bus_vn_kv, bd.sn_mva);
    return session;
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

    const int base_iters = init_from_n_powerflow ? 1 : max_iter_base;

    auto session = std::make_shared<InjectionSweepSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, base_iters, tol_base, device);

    if (with_branch_data) {
        BranchData bd = extract_branch_data(grid);
        session->set_branch_data(bd.branch_from, bd.branch_to,
                                 bd.yff, bd.yft, bd.ytf, bd.ytt,
                                 bd.bus_vn_kv, bd.sn_mva);
    }
    return session;
}
