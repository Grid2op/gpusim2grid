// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// ls2g_bridge.cpp  —  extract solved state from an LSGrid into a GPU session
// =============================================================================

#include "ls2g_bridge.hpp"
#include "ledger_extend.hpp"   // materialize_vc_custom_rows, reserve_stranded_controller_slots
#include "timing_utils.hpp"    // ms_since

#include <batch_algorithm/BusQCheck.hpp>   // ls2g::bus_q_check::build_bus_q_plan (lightsim2grid >= PR #206)

#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
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

// Resolve the (scaling_max_voltage_change, max_dVa, max_dVm) triple to hand to
// AcPfNrState. By default (all three *_override args at their sentinel: -1
// for the tri-state bool, negative for the doubles) this mirrors whatever the
// grid's OWN get_ac_algo_config() already has set -- opt-in, and it inherits
// lightsim2grid's own configured values rather than introducing a separate
// gpusim2grid-level default. A caller (the Python facades) can force it on/off
// or override max_dVa/max_dVm regardless of what the grid itself is
// configured with. int_params[0]==1 is ls2g::ScalingPolicyType::MaxVoltageChange
// (ScalingPolicies.hpp) -- not named via the enum here to avoid depending on
// ScalingPolicies.hpp being transitively visible from this translation unit.
std::tuple<bool, double, double> resolve_scaling_policy(
    const ls2g::LSGrid& grid,
    int scaling_max_voltage_change_override,
    double max_dVa_override,
    double max_dVm_override)
{
    const ls2g::AlgoConfig cfg = grid.get_ac_algo_config();
    const bool grid_max_vchange =
        !cfg.int_params.empty() && cfg.int_params[0] == 1;
    const double grid_max_dVa = cfg.real_params.size() > 0 ? cfg.real_params[0] : 0.5;
    const double grid_max_dVm = cfg.real_params.size() > 1 ? cfg.real_params[1] : 0.1;

    const bool scaling = (scaling_max_voltage_change_override >= 0)
        ? (scaling_max_voltage_change_override == 1) : grid_max_vchange;
    const double max_dVa = (max_dVa_override >= 0.0) ? max_dVa_override : grid_max_dVa;
    const double max_dVm = (max_dVm_override >= 0.0) ? max_dVm_override : grid_max_dVm;
    return {scaling, max_dVa, max_dVm};
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
    CplxVect        yff_eff, yft_eff, ytf_eff, ytt_eff;
    RealVect        bus_vn_kv;
    double          sn_mva;
};

BranchData extract_branch_data(const ls2g::LSGrid& grid, int n_bus_solver)
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
    bd.yff_eff = concat_cplx(lines.yac_eff_11(), trafos.yac_eff_11());
    bd.yft_eff = concat_cplx(lines.yac_eff_12(), trafos.yac_eff_12());
    bd.ytf_eff = concat_cplx(lines.yac_eff_21(), trafos.yac_eff_21());
    bd.ytt_eff = concat_cplx(lines.yac_eff_22(), trafos.yac_eff_22());
    // bus_vn_kv must be relabeled global(model)->AC-solver the same way
    // branch_from/branch_to above and bus_vmin_kv/bus_vmax_kv in
    // extract_limits() are, or it silently pairs the wrong nominal voltage
    // with the wrong bus (used below to derive d_base_current_A) whenever
    // id_me_to_ac_solver is not the identity -- e.g. isolated/Kron-reduced
    // buses from keep_half_open_lines or a disconnected grid. A solver bus
    // id with no model-side entry can't happen (every solver bus originates
    // from some model bus); left at 0 defensively.
    {
        ls2g::RealVect vn_model = grid.get_bus_vn_kv();
        RealVect vn_solver = RealVect::Zero(n_bus_solver);
        for (size_t grid_id = 0; grid_id < me_to_solver.size(); ++grid_id) {
            const int solver_id = me_to_solver[grid_id];
            if (solver_id >= 0 && solver_id < n_bus_solver)
                vn_solver(solver_id) = vn_model(static_cast<Eigen::Index>(grid_id));
        }
        bd.bus_vn_kv = vn_solver;
    }
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

    const eigen_real_type nan_val = std::numeric_limits<eigen_real_type>::quiet_NaN();

    // Branch limits: bulk C++ accessor exists (TwoSidesContainer_rxh_A::
    // get_limit_a1_ka/a2_ka) -- straight concat, same head/tail pattern as
    // concat_cplx above. NaN entries ("not configured") pass through as-is.
    // Like the bus limits below, a container whose limits were NEVER
    // configured hands back an EMPTY vector, not a NaN-filled nb() one (e.g.
    // any pandapower case without thermal limits): pad it to nb() NaNs per
    // container, so the concatenation is always n_lines + n_trafos long and
    // lines/trafos never misalign when only one of the two is configured.
    {
        auto padded = [&](Eigen::Ref<const ls2g::RealVect> v, Eigen::Index nb) -> RealVect {
            if (v.size() == nb) return RealVect(v);
            if (v.size() != 0)
                throw std::runtime_error(
                    "extract_limits: a branch current-limit vector has " +
                    std::to_string(v.size()) + " entries for " + std::to_string(nb) +
                    " elements");
            return RealVect::Constant(nb, nan_val);
        };
        const Eigen::Index nl = static_cast<Eigen::Index>(lines.nb());
        const Eigen::Index nt = static_cast<Eigen::Index>(trafos.nb());
        const RealVect l1_lines  = padded(lines.get_limit_a1_ka(),  nl);
        const RealVect l1_trafos = padded(trafos.get_limit_a1_ka(), nt);
        ld.limit_a1_ka.resize(nl + nt);
        ld.limit_a1_ka.head(nl) = l1_lines;
        ld.limit_a1_ka.tail(nt) = l1_trafos;

        const RealVect l2_lines  = padded(lines.get_limit_a2_ka(),  nl);
        const RealVect l2_trafos = padded(trafos.get_limit_a2_ka(), nt);
        ld.limit_a2_ka.resize(nl + nt);
        ld.limit_a2_ka.head(nl) = l2_lines;
        ld.limit_a2_ka.tail(nt) = l2_trafos;
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

// Pick the ledger a session is built on, honouring use_distributed_slack.
//
// use_distributed_slack=true (default) → the augmented ledger, i.e. the same
// system lightsim2grid poses. false → nullptr, which AcPfNrState turns into
// the trivial feature-free ledger (bare [pvpq | pq]); that is exactly what
// "remove the slack row/column" means, since MultiSlack contributes exactly
// one extra row (the reference bus' P equation) and one extra column
// (slack_absorbed) on top of that bare layout, whatever the participant count.
//
// The other in-Jacobian controls live in the very same ledger, so dropping it
// would drop them too. Refuse rather than silently change the physics.
const LedgerData* select_ledger(const LedgerData& ledger,
                                bool              use_distributed_slack,
                                const char*       who)
{
    if (use_distributed_slack) return &ledger;
    if (ledger.has_hvdc() || ledger.has_voltage_control())
        throw std::runtime_error(
            std::string(who) + ": use_distributed_slack=false is not supported on a "
            "grid carrying HVDC-droop or VoltageControl (SVC / remote generator "
            "voltage control) features -- they share the augmented Jacobian with "
            "the MultiSlack row/column and cannot be kept while it is dropped. "
            "Use use_distributed_slack=true, or remove those controls from the "
            "grid first.");
    return nullptr;
}

// Remap one bus-keyed map in place through old→new (negatives stay negative).
void remap_map(std::vector<int>& m, const std::vector<int>& old_to_new)
{
    for (int& v : m) v = (v < 0) ? -1 : old_to_new[v];
}

// Remap a compact (bus, row_or_col) pair list, dropping every registration
// whose row/col no longer exists.
void remap_pairs(std::vector<int>& buses, std::vector<int>& idx,
                 const std::vector<int>& old_to_new)
{
    std::vector<int> kb, ki;
    kb.reserve(buses.size());
    ki.reserve(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) {
        const int v = idx[k];
        const int nv = (v < 0) ? -1 : old_to_new[v];
        if (nv < 0) continue;
        kb.push_back(buses[k]);
        ki.push_back(nv);
    }
    buses.swap(kb);
    idx.swap(ki);
}

// The per-generator "local / remote voltage controller" predicates changed
// name in lightsim2grid: dev_1.0.1 has GeneratorContainer::
// gen_is_local_voltage_controller / gen_is_voltage_controller (the latter
// meaning REMOTE), the later VoltageSourceContainer refactor renamed them
// is_local_voltage_controller / is_remote_voltage_controller. Same gating
// either way (status, regulator on, regulates_remote, pseudo-off rule).
// Detect whichever the installed headers provide so the bridge builds
// against both.
template <class C, class = void>
struct has_new_vc_predicates : std::false_type {};
template <class C>
struct has_new_vc_predicates<
    C, std::void_t<decltype(std::declval<const C&>().is_remote_voltage_controller(0))>>
    : std::true_type {};

template <class C>
bool gen_local_vreg(const C& gens, int g)
{
    if constexpr (has_new_vc_predicates<C>::value) return gens.is_local_voltage_controller(g);
    else                                           return gens.gen_is_local_voltage_controller(g);
}

template <class C>
bool gen_remote_vreg(const C& gens, int g)
{
    if constexpr (has_new_vc_predicates<C>::value) return gens.is_remote_voltage_controller(g);
    else                                           return gens.gen_is_voltage_controller(g);
}

}  // namespace

LedgerData drop_multislack_augmentation(const LedgerData& in,
                                        const Eigen::VectorXi& pv,
                                        const Eigen::VectorXi& pq)
{
    if (!in.has_multislack()) return in;

    const int old_dim = in.dim_J;
    const int n_bus   = static_cast<int>(in.p_row_of_bus.size());

    // Buses of the bare layout: those that own a P equation without MultiSlack.
    std::vector<char> in_pvpq(n_bus, 0);
    for (Eigen::Index i = 0; i < pv.size(); ++i) in_pvpq[pv[i]] = 1;
    for (Eigen::Index i = 0; i < pq.size(); ++i) in_pvpq[pq[i]] = 1;

    // Surplus rows/cols = everything the bare layout would not have.
    std::vector<char> row_drop(old_dim, 0), col_drop(old_dim, 0);
    int n_row_drop = 0, n_col_drop = 0;
    col_drop[in.slack_col] = 1;
    ++n_col_drop;
    for (int b = 0; b < n_bus; ++b) {
        if (in_pvpq[b]) continue;
        const int pr = in.p_row_of_bus[b];
        const int tc = in.theta_col_of_bus[b];
        if (pr >= 0 && !row_drop[pr]) { row_drop[pr] = 1; ++n_row_drop; }
        if (tc >= 0 && !col_drop[tc]) { col_drop[tc] = 1; ++n_col_drop; }
    }
    if (n_row_drop != n_col_drop)
        throw std::runtime_error(
            "drop_multislack_augmentation: the MultiSlack augmentation is not "
            "square (" + std::to_string(n_row_drop) + " surplus rows vs "
            + std::to_string(n_col_drop) + " surplus columns) — the ledger does "
            "not have the layout this transform assumes. Use "
            "use_distributed_slack=true.");

    // old → new index maps (-1 = deleted).
    std::vector<int> row_new(old_dim, -1), col_new(old_dim, -1);
    for (int i = 0, r = 0, c = 0; i < old_dim; ++i) {
        if (!row_drop[i]) row_new[i] = r++;
        if (!col_drop[i]) col_new[i] = c++;
    }
    const int new_dim = old_dim - n_row_drop;

    LedgerData out = in;
    out.dim_J = new_dim;

    // Skeleton: drop the surplus rows, and the surplus columns of every kept
    // row. Nothing is ever added — the augmented pattern is a superset.
    out.J_outer.clear();
    out.J_inner.clear();
    out.J_outer.reserve(static_cast<size_t>(new_dim) + 1);
    out.J_inner.reserve(in.J_inner.size());
    out.J_outer.push_back(0);
    for (int r = 0; r < old_dim; ++r) {
        if (row_drop[r]) continue;
        for (int k = in.J_outer[r]; k < in.J_outer[r + 1]; ++k) {
            const int c = in.J_inner[k];
            if (!col_drop[c]) out.J_inner.push_back(col_new[c]);
        }
        out.J_outer.push_back(static_cast<int>(out.J_inner.size()));
    }

    remap_map(out.p_row_of_bus,     row_new);
    remap_map(out.q_row_of_bus,     row_new);
    remap_map(out.theta_col_of_bus, col_new);
    remap_map(out.vm_col_of_bus,    col_new);
    remap_map(out.q_col_of_bus,     col_new);

    remap_pairs(out.p_buses,     out.p_rows,     row_new);
    remap_pairs(out.q_buses,     out.q_rows,     row_new);
    remap_pairs(out.theta_buses, out.theta_cols, col_new);
    remap_pairs(out.vm_buses,    out.vm_cols,    col_new);

    // VoltageControl keeps its own per-controller Q columns (never surplus:
    // they are not theta columns and not slack_col). Its custom rows are
    // reconstructed downstream as the LAST vc_n_controllers() rows of J, which
    // stays correct under deletion because only P rows — all of which precede
    // them — are removed.
    for (int& c : out.vc_q_col) {
        if (c < 0) continue;
        const int nc = col_new[c];
        if (nc < 0)
            throw std::runtime_error(
                "drop_multislack_augmentation: a VoltageControl Q column "
                "collided with the MultiSlack augmentation — cannot drop the "
                "distributed slack without also dropping that controller.");
        c = nc;
    }

    // Explicit VC custom rows (only present once ledger_extend.hpp froze them;
    // the factories drop the slack BEFORE that, so normally empty here).
    for (int& r : out.vc_v_rows) if (r >= 0) r = row_new[r];
    for (auto& rows : out.vc_share_rows)
        for (int& r : rows) if (r >= 0) r = row_new[r];
    if (!out.switchable_vm_buses.empty())
        throw std::runtime_error(
            "drop_multislack_augmentation: must run before add_switchable_vm_buses");

    // MultiSlack itself is gone: the feature kernels are gated on slack_col>=0.
    out.slack_col = -1;
    out.slack_weights.clear();
    out.slack_absorbed_gt = 0.0;
    return out;
}

GenContingencyData extract_gen_contingency_data(const ls2g::LSGrid& grid, int n_bus_solver)
{
    const ls2g::GeneratorContainer& gens = grid.get_generators();
    const int n_gen = gens.nb();
    const std::vector<int> me_to_solver = grid.id_me_to_ac_solver_numpy();
    const std::set<int> group_buses = grid.get_group_controlled_buses();  // GRID bus ids

    GenContingencyData d;
    d.n_gen = n_gen;
    d.bus.assign(n_gen, -1);
    d.status.assign(n_gen, 0);
    d.local_vreg.assign(n_gen, 0);
    d.remote_vreg.assign(n_gen, 0);
    d.on_group_bus.assign(n_gen, 0);
    d.slack_participant.assign(n_gen, 0);
    d.slack_weight.assign(n_gen, 0.0);
    d.vreg_on.assign(n_gen, 0);
    d.target_q_mvar.assign(n_gen, 0.0);

    for (int g = 0; g < n_gen; ++g) {
        const ls2g::GenInfo gi(gens, g);
        d.status[g]        = gi.connected ? 1 : 0;
        d.slack_weight[g]  = static_cast<double>(gi.slack_weight);
        d.vreg_on[g]       = gi.voltage_regulator_on ? 1 : 0;
        d.target_q_mvar[g] = static_cast<double>(gi.target_q_mvar);
        if (!gi.connected) continue;
        const int bus_me = gi.bus_id;
        int bus_solver = bus_me;
        if (!me_to_solver.empty()) {
            bus_solver = (bus_me >= 0 && bus_me < static_cast<int>(me_to_solver.size()))
                         ? me_to_solver[bus_me] : -1;
        }
        if (bus_solver < 0 || bus_solver >= n_bus_solver) bus_solver = -1;
        d.bus[g]          = bus_solver;
        d.local_vreg[g]   = gen_local_vreg(gens, g)  ? 1 : 0;
        d.remote_vreg[g]  = gen_remote_vreg(gens, g) ? 1 : 0;
        d.on_group_bus[g] = (bus_me >= 0 && group_buses.count(bus_me) > 0) ? 1 : 0;
        d.slack_participant[g] =
            (gi.is_slack && std::abs(static_cast<double>(gi.slack_weight)) > 1e-12) ? 1 : 0;
    }
    return d;
}

LedgerData extract_ledger_data(const ls2g::LSGrid& grid, bool presolved_v, double tol)
{
    LedgerData ld;

    // Augmented J sparsity skeleton in RowMajor CSR (structure only). get_J_solver
    // hands back lightsim2grid's compressed ColMajor J; its index arrays are
    // transposed into gpusim2grid's CSR convention (outer = row) with one
    // counting sort -- walking the columns in order leaves every row's column
    // indices sorted -- rather than converting through a second Eigen matrix
    // and makeCompressed.
    {
        const Eigen::Ref<const Eigen::SparseMatrix<eigen_real_type>> J_cm = grid.get_J_solver();
        ld.dim_J = static_cast<int>(J_cm.rows());
        const int  nnz_J    = static_cast<int>(J_cm.nonZeros());
        const int* cm_outer = J_cm.outerIndexPtr();
        const int* cm_inner = J_cm.innerIndexPtr();
        ld.J_outer.assign(static_cast<size_t>(ld.dim_J) + 1, 0);
        for (int p = 0; p < nnz_J; ++p) ++ld.J_outer[static_cast<size_t>(cm_inner[p]) + 1];
        for (int r = 0; r < ld.dim_J; ++r) ld.J_outer[r + 1] += ld.J_outer[r];
        ld.J_inner.resize(static_cast<size_t>(nnz_J));
        std::vector<int> head(ld.J_outer.begin(), ld.J_outer.end() - 1);
        for (int c = 0; c < static_cast<int>(J_cm.outerSize()); ++c)
            for (int p = cm_outer[c]; p < cm_outer[c + 1]; ++p)
                ld.J_inner[static_cast<size_t>(head[cm_inner[p]]++)] = c;
    }

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
            // Grid hvdc id of each entry, replaying LSGrid::fill_hvdc_droop_solver_data's
            // own selection (droop enabled AND globally connected, ascending id) --
            // what compute_physical_violations reports. Only trusted when the replay
            // lands on the same count; otherwise left empty (entry index).
            {
                const ls2g::HvdcLineContainer& hv = grid.get_dclines();
                const std::vector<bool>& droop_on = hv.get_droop_enabled();
                const std::vector<bool>& glob_on  = hv.get_status_global();
                std::vector<int> ids;
                for (int hid = 0; hid < hv.nb(); ++hid)
                    if (droop_on[hid] && glob_on[hid]) ids.push_back(hid);
                if (static_cast<int>(ids.size()) == nh) ld.hvdc_id = ids;
            }
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
            // Per-controller Q column -- NOT ledger.q_col_of_bus (bus-keyed,
            // collides whenever two controllers share a bus). See LedgerData::
            // vc_q_col's own doc.
            ld.vc_q_col     = to_iv(grid.get_controller_q_col_solver());
        }
    }

    // presolved_v (init_from_n_powerflow) precondition: verify the grid is
    // actually solved before trusting anything derived from its converged
    // state, and (when an extension is active) pull lightsim2grid's own
    // converged extension state as ground truth for AcPfNrState's
    // presolved_v fast path to seed slack_absorbed/vc_q from directly,
    // instead of deriving them via a cuDSS solve. Mandatory and unconditional
    // whenever presolved_v is requested -- regardless of whether an
    // extension happens to be active.
    if (presolved_v) {
        auto& g = const_cast<ls2g::LSGrid&>(grid);
        auto t0 = std::chrono::steady_clock::now();
        // check_solution(V, check_q_limits=false): "check the kirchoff law"
        // against the CALLER-supplied V as-is, without lightsim2grid's own
        // NR-initialization heuristics (see LSGrid.hpp).
        //
        // MUST be grid-model (original) bus numbering -- get_V(), NOT
        // get_V_solver(). check_solution() internally calls pre_process_solver()
        // with V_proposed.size() as the ORIGINAL bus count and rebuilds its own
        // id_me_to_ac_solver_ mapping from it; passing the already-reduced
        // solver-numbering vector (get_V_solver(), size == n_bus_solver, smaller
        // whenever any bus is isolated/disconnected) makes that mapping too
        // small, and later bus-id lookups (e.g. fill_hvdc_droop_solver_data)
        // index past the end of it -- heap corruption ("free(): corrupted
        // unsorted chunks"), confirmed via gdb on a real ~47k-bus / 7270-solver-
        // bus grid with distributed slack + SVCs + HVDC droop (reproduces with
        // plain lightsim2grid alone, no gpusim2grid involved). get_V() relabels
        // back to the full original numbering LSGrid.cpp's own check_solution()
        // comment already warns about this exact class of bug.
        ls2g::CplxVect mismatch = g.check_solution(grid.get_V(), false);
        ld.t_ground_truth_check_ms = ms_since(t0);

        // check_solution() scales its result by sn_mva_ into physical (MW/MVAr)
        // units (LSGrid.cpp: "if (sn_mva_ != 1) res *= sn_mva_"), but `tol`/
        // `tol_base` is a per-unit ||F||_inf tolerance everywhere else in this
        // codebase (the GPU-side check, ac_pf's own convergence criterion, ...).
        // Undo that scaling before comparing so both sides are the same unit.
        const double sn_mva = static_cast<double>(g.get_sn_mva());
        const double norm_inf_pu = (mismatch.size() > 0
            ? mismatch.cwiseAbs().maxCoeff() : 0.0) / (sn_mva > 0. ? sn_mva : 1.);
        if (norm_inf_pu > tol) {
            throw std::runtime_error(
                "extract_ledger_data: presolved_v/init_from_n_powerflow requested "
                "but LSGrid::check_solution() reports ||mismatch||_inf (per-unit) = " +
                std::to_string(norm_inf_pu) + " exceeds tol = " + std::to_string(tol) +
                ". The grid is not actually solved for its own Ybus/Sbus (or was "
                "mutated since its last ac_pf()) -- re-solve before trusting Vinit, "
                "or disable init_from_n_powerflow.");
        }

        if (ld.slack_col >= 0) {
            ld.slack_absorbed_gt = static_cast<double>(g.get_slack_absorbed_solver());
            ld.has_ext_state_ground_truth = true;
        }
        if (!ld.vc_bus.empty()) {
            ls2g::RealVect qgt = g.get_controller_q_solver();
            ld.vc_q_gt.assign(qgt.data(), qgt.data() + qgt.size());
            ld.has_ext_state_ground_truth = true;
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
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg,
    bool   debug_base_case,
    int    scaling_max_voltage_change_override,
    double max_dVa_override,
    double max_dVm_override,
    bool   use_distributed_slack)
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

    LedgerData ledger = extract_ledger_data(grid, init_from_n_powerflow, tol);
    if (!use_distributed_slack)
        ledger = drop_multislack_augmentation(ledger, pv, pq);

    const auto [scaling, max_dVa, max_dVm] = resolve_scaling_policy(
        grid, scaling_max_voltage_change_override, max_dVa_override, max_dVm_override);

    return std::make_shared<AcPfNrSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq, max_iter, tol, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow,
        diag_stop_before_state_correction,
        reordering_alg, matching_alg, pivot_epsilon_alg, debug_base_case,
        scaling, max_dVa, max_dVm);
}

std::shared_ptr<AcPfNrSession>
make_acpf_session_from_lsgrid_with_sbus(
    const ls2g::LSGrid& grid,
    Eigen::Ref<const CplxVect> Sbus,
    int    max_iter,
    double tol,
    int    device,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg)
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
        reordering_alg, matching_alg, pivot_epsilon_alg);
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
    bool   compute_limit_violations,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg,
    bool   debug_base_case,
    int    scaling_max_voltage_change_override,
    double max_dVa_override,
    double max_dVm_override,
    bool   use_distributed_slack)
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

    LedgerData ledger = extract_ledger_data(grid, init_from_n_powerflow, tol_base);
    if (!use_distributed_slack)
        ledger = drop_multislack_augmentation(ledger, pv, pq);
    // handle_disconnected_grid: one structural zero per lone remote-regulating
    // generator so a contingency stranding its own bus can repurpose its
    // voltage row by value (lightsim2grid PR #192). Negligible cost, always on
    // for the batch sessions (handle_disconnected_grid is a mutable property).
    reserve_stranded_controller_slots(ledger);
    const auto [scaling, max_dVa, max_dVm] = resolve_scaling_policy(
        grid, scaling_max_voltage_change_override, max_dVa_override, max_dVm_override);
    auto session = std::make_shared<ContingencyAnalysisSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, max_iter_base, tol_base, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow,
        reordering_alg, matching_alg, pivot_epsilon_alg, debug_base_case,
        scaling, max_dVa, max_dVm);

    BranchData bd = extract_branch_data(grid, static_cast<int>(Ybus.rows()));
    session->set_branch_data(bd.branch_from, bd.branch_to,
                             bd.yff_eff, bd.yft_eff, bd.ytf_eff, bd.ytt_eff,
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

BusQPlanData extract_bus_q_plan_from_lsgrid(const ls2g::LSGrid& grid, int n_bus_solver)
{
    // lightsim2grid's own routing, on the labelling the session solves in (the
    // AC cache of a solved grid) and the same controller list the ledger was
    // read from (extract_ledger_data).
    ls2g::VoltageControlSolverData ctrl;
    grid.fill_voltage_control_solver_data(ctrl, /*ac=*/true);
    ls2g::bus_q_check::BusQPlan plan;
    ls2g::bus_q_check::build_bus_q_plan(grid, grid.id_me_to_ac_solver(), ctrl, plan);

    const ls2g::GeneratorContainer& gens  = grid.get_generators();
    const ls2g::SvcContainer&       svcs  = grid.get_svcs();
    const ls2g::HvdcLineContainer&  hvdcs = grid.get_dclines();

    BusQPlanData out;
    out.sn_mva = static_cast<double>(grid.get_sn_mva());
    std::vector<int>    bus_solver, n_fixed, gen_start{0}, gen_id;
    std::vector<double> qmin_fixed, qmax_fixed, bmin_sum, bmax_sum, gen_qmin, gen_qmax;
    for (const ls2g::bus_q_check::BusQEntry& e : plan.buses) {
        if (e.bus_solver < 0 || e.bus_solver >= n_bus_solver) continue;   // not in the solved system
        double qmin = 0., qmax = 0., bmin = 0., bmax = 0.;
        for (const auto& st : e.station_ids) {
            qmin += static_cast<double>(hvdcs.get_station_min_q_mvar(st.first, st.second));
            qmax += static_cast<double>(hvdcs.get_station_max_q_mvar(st.first, st.second));
        }
        for (int svc : e.svc_ids) {
            bmin += static_cast<double>(svcs.get_b_min(svc));
            bmax += static_cast<double>(svcs.get_b_max(svc));
        }
        for (int g : e.gen_ids) {
            gen_id.push_back(g);
            gen_qmin.push_back(static_cast<double>(gens.get_min_q(g)));
            gen_qmax.push_back(static_cast<double>(gens.get_max_q(g)));
        }
        bus_solver.push_back(e.bus_solver);
        n_fixed.push_back(static_cast<int>(e.station_ids.size() + e.svc_ids.size()));
        qmin_fixed.push_back(qmin); qmax_fixed.push_back(qmax);
        bmin_sum.push_back(bmin);   bmax_sum.push_back(bmax);
        gen_start.push_back(static_cast<int>(gen_id.size()));
    }
    auto iv = [](const std::vector<int>& v) { return Eigen::VectorXi::Map(v.data(), static_cast<Eigen::Index>(v.size())).eval(); };
    auto rv = [](const std::vector<double>& v) {
        RealVect r(static_cast<Eigen::Index>(v.size()));
        for (size_t i = 0; i < v.size(); ++i) r(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(v[i]);
        return r;
    };
    out.n_check         = static_cast<int>(bus_solver.size());
    out.bus_solver      = iv(bus_solver);
    out.n_fixed         = iv(n_fixed);
    out.gen_start       = iv(gen_start);
    out.gen_id          = iv(gen_id);
    out.qmin_fixed_mvar = rv(qmin_fixed);
    out.qmax_fixed_mvar = rv(qmax_fixed);
    out.bmin_sum_pu     = rv(bmin_sum);
    out.bmax_sum_pu     = rv(bmax_sum);
    out.gen_qmin_mvar   = rv(gen_qmin);
    out.gen_qmax_mvar   = rv(gen_qmax);
    out.validate(n_bus_solver);
    return out;
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
    bool   with_branch_data,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg,
    bool   debug_base_case,
    int    scaling_max_voltage_change_override,
    double max_dVa_override,
    double max_dVm_override,
    bool   use_distributed_slack)
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

    LedgerData ledger = extract_ledger_data(grid, init_from_n_powerflow, tol_base);
    if (!use_distributed_slack)
        ledger = drop_multislack_augmentation(ledger, pv, pq);
    const auto [scaling, max_dVa, max_dVm] = resolve_scaling_policy(
        grid, scaling_max_voltage_change_override, max_dVa_override, max_dVm_override);
    auto session = std::make_shared<InjectionSweepSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, max_iter_base, tol_base, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow,
        reordering_alg, matching_alg, pivot_epsilon_alg, debug_base_case,
        scaling, max_dVa, max_dVm);

    if (with_branch_data) {
        BranchData bd = extract_branch_data(grid, static_cast<int>(Ybus.rows()));
        session->set_branch_data(bd.branch_from, bd.branch_to,
                                 bd.yff_eff, bd.yft_eff, bd.ytf_eff, bd.ytt_eff,
                                 bd.bus_vn_kv, bd.sn_mva);
    }
    return session;
}

std::shared_ptr<ScenarioSweepSession>
make_ss_session_from_lsgrid(
    const ls2g::LSGrid& grid,
    bool   init_from_n_powerflow,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    bool   compute_limit_violations,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg,
    bool   debug_base_case,
    int    scaling_max_voltage_change_override,
    double max_dVa_override,
    double max_dVm_override,
    bool   use_distributed_slack)
{
    auto& g = const_cast<ls2g::LSGrid&>(grid);
    Eigen::SparseMatrix<eigen_cplx_type> Ybus = g.get_Ybus_solver();
    if (Ybus.rows() == 0)
        throw std::runtime_error(
            "make_ss_session_from_lsgrid: empty Ybus — has the grid been solved "
            "(ac_pf) before being handed to gpusim2grid?");

    CplxVect        V0    = grid.get_V_solver();
    CplxVect        Sbus  = grid.get_Sbus_solver();
    Eigen::VectorXi slack = grid.get_slack_ids_solver_numpy();
    RealVect        sw    = grid.get_slack_weights_solver();
    Eigen::VectorXi pv    = grid.get_pv_solver_numpy();
    Eigen::VectorXi pq    = grid.get_pq_solver_numpy();

    LedgerData ledger = extract_ledger_data(grid, init_from_n_powerflow, tol_base);
    if (!use_distributed_slack)
        ledger = drop_multislack_augmentation(ledger, pv, pq);
    // See make_ca_session_from_lsgrid. The switchable Vm buses of generator
    // contingencies are NOT reserved here: the session derives them from the
    // mask it is given later and extends a copy of this ledger itself.
    reserve_stranded_controller_slots(ledger);
    const auto [scaling, max_dVa, max_dVm] = resolve_scaling_policy(
        grid, scaling_max_voltage_change_override, max_dVa_override, max_dVm_override);
    auto session = std::make_shared<ScenarioSweepSession>(
        Ybus, V0, Sbus, slack, sw, pv, pq,
        batch_size, nb_iter, max_iter_base, tol_base, device, &ledger,
        /*presolved_v=*/init_from_n_powerflow,
        reordering_alg, matching_alg, pivot_epsilon_alg, debug_base_case,
        scaling, max_dVa, max_dVm);

    // Per-generator snapshot for set_contingency_gens() (cheap, always).
    session->set_gen_contingency_data(
        extract_gen_contingency_data(grid, static_cast<int>(Ybus.rows())));

    // Always set — set_topology() needs branch admittances to build each
    // scenario's Ybus triplets, not just compute_flows().
    BranchData bd = extract_branch_data(grid, static_cast<int>(Ybus.rows()));
    session->set_branch_data(bd.branch_from, bd.branch_to,
                             bd.yff_eff, bd.yft_eff, bd.ytf_eff, bd.ytt_eff,
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
