// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// scenario_sweep_session.cu
// =============================================================================

#include "scenario_sweep_session.hpp"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/scenario_sweep_batch.cuh"
#include "contingency/gen_v_override.hpp"   // GenVOverride, build_gen_v_override
#include "acpf_nr_kernels.cuh"   // compute_branch_flows_kernel, zero_branch_flows_kernel
#include "cu_complex_utils.h"
#include "cuda_utils.h"          // ms_since
#include "ledger_data.hpp"       // LedgerData (mask_cfg_ controller-bus setup)
#include "ledger_extend.hpp"     // add_switchable_vm_buses
#include "mask_config_builder.cuh"   // build_mask_config

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static constexpr int SESSION_BS = 256;

// =============================================================================
// Destructor — defined here where AcPfNrState / BatchPfDriver<ScenarioSweepBatch>
// are complete types, enabling unique_ptr to call delete correctly.
// =============================================================================
ScenarioSweepSession::~ScenarioSweepSession() = default;

// =============================================================================
// Constructor — base-case NR to convergence.
// =============================================================================
ScenarioSweepSession::ScenarioSweepSession(
    const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
    Eigen::Ref<const CplxVect>                  Vinit,
    Eigen::Ref<const CplxVect>                  Sbus,
    Eigen::Ref<const Eigen::VectorXi>           slack_ids,
    Eigen::Ref<const RealVect>                  slack_weights,
    Eigen::Ref<const Eigen::VectorXi>           pv,
    Eigen::Ref<const Eigen::VectorXi>           pq,
    int    batch_size,
    int    nb_iter,
    int    max_iter_base,
    double tol_base,
    int    device,
    const LedgerData* ledger,
    bool   presolved_v,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg,
    bool   debug_base_case,
    bool   scaling_max_voltage_change,
    double max_dVa,
    double max_dVm)
    : Ybus_rm_(Ybus)
    , batch_size_(batch_size)
    , nb_iter_(nb_iter)
    , reordering_alg_(reordering_alg)
    , matching_alg_(matching_alg)
    , pivot_epsilon_alg_(pivot_epsilon_alg)
    , scaling_max_voltage_change_(scaling_max_voltage_change)
    , max_dVa_(max_dVa)
    , max_dVm_(max_dVm)
    , Ybus_cm_(Ybus)
    , Vinit_(Vinit)
    , Sbus_(Sbus)
    , slack_ids_(slack_ids)
    , pv_(pv)
    , pq_(pq)
    , max_iter_base_(max_iter_base)
    , tol_base_(tol_base)
    , device_(device)
    , presolved_v_(presolved_v)
    , debug_base_case_(debug_base_case)
{
    (void)slack_weights;
    if (ledger != nullptr) base_ledger_ = std::make_unique<LedgerData>(*ledger);

    _build_base_state(std::vector<int>{});

    // Vm-fixed bus mask for set_gen_v(): a bus in pv or slack_ids has |V|
    // fixed by construction (not an NR unknown) in both the bare and the
    // augmented-ledger system -- see set_gen_v()'s own doc. A switchable bus
    // (generator contingencies) keeps its reseed too: it is only the NR start
    // value on a row that releases it.
    {
        const int n_bus = base_state_->n_bus;
        h_is_vm_fixed_bus_.assign(static_cast<size_t>(n_bus), 0);
        for (Eigen::Index i = 0; i < pv.size(); ++i) {
            const int b = pv(i);
            if (b >= 0 && b < n_bus) h_is_vm_fixed_bus_[static_cast<size_t>(b)] = 1;
        }
        for (Eigen::Index i = 0; i < slack_ids.size(); ++i) {
            const int b = slack_ids(i);
            if (b >= 0 && b < n_bus) h_is_vm_fixed_bus_[static_cast<size_t>(b)] = 1;
        }
    }
}

// =============================================================================
// _build_base_state — base-case NR on the (possibly extended) ledger.
// =============================================================================
void ScenarioSweepSession::_build_base_state(const std::vector<int>& switchable_buses)
{
    // solver_ references *base_state_: drop it first. Its d_V_results /
    // DLPack views die with it (run() rebuilds it anyway).
    solver_.reset();
    has_violations_result_ = false;

    const LedgerData* ledger_ptr = nullptr;
    LedgerData ext;
    if (base_ledger_) {
        ext = *base_ledger_;
        if (!switchable_buses.empty())
            add_switchable_vm_buses(ext, switchable_buses, Ybus_rm_);
        ledger_ptr = &ext;
    }

    auto t_base_start = std::chrono::steady_clock::now();
    base_state_ = std::make_unique<AcPfNrState>(
        Ybus_cm_, Vinit_, Sbus_, pv_, pq_,
        max_iter_base_,
        static_cast<eigen_real_type>(tol_base_),
        device_, ledger_ptr, presolved_v_,
        /*diag_stop_before_state_correction=*/false,
        reordering_alg_, matching_alg_, pivot_epsilon_alg_,
        debug_base_case_, /*base_case_only=*/true,
        scaling_max_voltage_change_, max_dVa_, max_dVm_);
    t_base_case_ms_ = ms_since(t_base_start);

    // handle_disconnected_grid mask configuration (per-bus identity-row
    // metadata + angle reference + VC group topology / row positions); its
    // row_info also drives the per-row PV pins of generator contingencies, so
    // it is rebuilt with the base state. Shared builder with
    // ContingencyAnalysisSession.
    mask_cfg_ = build_mask_config(*base_state_, ledger_ptr);
    reserved_buses_ = base_state_->h_switchable_buses;
}

// =============================================================================
// set_gen_contingency_data / set_contingency_gens
// =============================================================================
void ScenarioSweepSession::set_gen_contingency_data(const GenContingencyData& data)
{
    gen_data_     = data;
    has_gen_data_ = (data.n_gen > 0);
}

void ScenarioSweepSession::set_contingency_gens(Eigen::Ref<const BoolMat> mask)
{
    if (!has_gen_data_)
        throw std::runtime_error(
            "ScenarioSweepSession::set_contingency_gens: this session has no "
            "generator data (explicit-array/tuple mode, or a build without the "
            "lightsim2grid bridge). Generator contingencies need a session built "
            "from a lightsim2grid grid.");
    if (mask.cols() != gen_data_.n_gen)
        throw std::runtime_error(
            "ScenarioSweepSession::set_contingency_gens: the mask has " +
            std::to_string(mask.cols()) + " columns but the grid has " +
            std::to_string(gen_data_.n_gen) + " generators");
    if (mask.rows() <= 0)
        throw std::runtime_error(
            "ScenarioSweepSession::set_contingency_gens: n_scenarios must be > 0");
    if (has_injections_ && static_cast<int>(mask.rows()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession::set_contingency_gens: row count must match "
            "set_injections()'s n_scenarios");

    // Only a generator pinning its OWN bus is in scope. A remote controller,
    // or a bus a control group holds (remote generator, SVC, HVDC station),
    // lives in the VoltageControl extension, whose own Jacobian rows and
    // columns nothing here reserves or releases (same rule as lightsim2grid).
    for (int g = 0; g < gen_data_.n_gen; ++g) {
        bool ever_off = false;
        for (Eigen::Index r = 0; r < mask.rows() && !ever_off; ++r)
            if (mask(r, g)) ever_off = true;
        if (!ever_off) continue;
        if (gen_data_.remote_vreg[g])
            throw std::runtime_error(
                "ScenarioSweepSession::set_contingency_gens: generator " +
                std::to_string(g) + " regulates the voltage of a remote bus. Only "
                "generators regulating their own bus can be disconnected for now: "
                "remote voltage control is not supported by this feature yet.");
        if (gen_data_.on_group_bus[g])
            throw std::runtime_error(
                "ScenarioSweepSession::set_contingency_gens: generator " +
                std::to_string(g) + " stands on a bus whose voltage a control group "
                "holds (a remote generator, an SVC or an HVDC converter station). "
                "Only generators regulating their own bus can be disconnected for "
                "now: remote voltage control is not supported by this feature yet.");
    }

    gen_off_     = mask;
    has_gen_off_ = true;
}

int ScenarioSweepSession::dim_J() const
{
    return base_state_ ? base_state_->dim_J : 0;
}

// =============================================================================
// _prepare_gen_contingency — mirrors BaseBatchSweep::_maybe_prepare_gen_contingency
// =============================================================================
void ScenarioSweepSession::_prepare_gen_contingency(
    std::vector<int>&              required,
    std::vector<std::vector<int>>& row_pv_to_pq,
    std::vector<std::vector<int>>& row_slack_off) const
{
    required.clear();
    row_pv_to_pq.assign(static_cast<size_t>(n_scenarios_), std::vector<int>());
    row_slack_off.assign(static_cast<size_t>(n_scenarios_), std::vector<int>());
    if (!has_gen_off_) return;

    const int n_gen  = gen_data_.n_gen;
    const int n_rows = std::min(n_scenarios_, static_cast<int>(gen_off_.rows()));

    // Which bus does each LOCAL voltage controller pin? (-1: none)
    std::map<int, std::vector<int>> gens_of_bus;
    for (int g = 0; g < n_gen; ++g) {
        if (!gen_data_.status[g] || !gen_data_.local_vreg[g]) continue;
        const int b = gen_data_.bus[g];
        if (b < 0) continue;
        gens_of_bus[b].push_back(g);
    }

    std::set<int> req;
    for (int r = 0; r < n_rows; ++r) {
        for (const auto& kv : gens_of_bus) {
            bool all_off = true;
            for (int g : kv.second)
                if (!gen_off_(r, g)) { all_off = false; break; }
            if (all_off) {
                row_pv_to_pq[static_cast<size_t>(r)].push_back(kv.first);
                req.insert(kv.first);
            }
        }
        for (int g = 0; g < n_gen; ++g)
            if (gen_off_(r, g) && gen_data_.status[g] && gen_data_.slack_participant[g])
                row_slack_off[static_cast<size_t>(r)].push_back(g);
    }

    // Only a bus that owns no Vm unknown / Q equation in the base ledger needs
    // a reserved pair (add_switchable_vm_buses skips the others anyway, so
    // this keeps `required` comparable to reserved_buses_).
    for (int b : req) {
        if (base_ledger_ &&
            (base_ledger_->vm_col_of_bus[static_cast<size_t>(b)] >= 0 ||
             base_ledger_->q_row_of_bus[static_cast<size_t>(b)] >= 0))
            continue;
        required.push_back(b);
    }
    std::sort(required.begin(), required.end());
}

// =============================================================================
// _row_slack_weights — mirrors BaseBatchSweep::_row_slack_weights /
// GeneratorContainer::get_slack_weights_solver_without
// =============================================================================
std::vector<cuda_real_type> ScenarioSweepSession::_row_slack_weights(
    const std::vector<std::vector<int>>& row_slack_off) const
{
    std::vector<cuda_real_type> out;
    const int n_slack = base_state_->n_slack;
    if (n_slack <= 0) return out;
    bool any = false;
    for (const auto& v : row_slack_off) if (!v.empty()) { any = true; break; }
    if (!any) return out;

    const int n_bus = base_state_->n_bus;
    const std::vector<int>& slack_bus = base_state_->h_slack_bus;
    const std::vector<cuda_real_type>& base_w = base_state_->h_slack_w;
    int ref_bus = -1;
    for (int b = 0; b < n_bus; ++b)
        if (mask_cfg_.is_reference_bus[static_cast<size_t>(b)]) { ref_bus = b; break; }
    if (ref_bus < 0 && slack_ids_.size() > 0) ref_bus = slack_ids_(0);

    out.assign(static_cast<size_t>(n_scenarios_) * n_slack, static_cast<cuda_real_type>(0.));
    std::vector<double> w(static_cast<size_t>(n_bus));
    std::vector<char>   off(static_cast<size_t>(gen_data_.n_gen));
    for (int r = 0; r < n_scenarios_; ++r) {
        cuda_real_type* row = out.data() + static_cast<ptrdiff_t>(r) * n_slack;
        if (row_slack_off[static_cast<size_t>(r)].empty()) {
            std::copy(base_w.begin(), base_w.end(), row);
            continue;
        }
        std::fill(off.begin(), off.end(), 0);
        for (int g : row_slack_off[static_cast<size_t>(r)]) off[static_cast<size_t>(g)] = 1;
        std::fill(w.begin(), w.end(), 0.0);
        double sum = 0.0;
        for (int g = 0; g < gen_data_.n_gen; ++g) {
            if (!gen_data_.status[g] || !gen_data_.slack_participant[g] || off[static_cast<size_t>(g)]) continue;
            const int b = gen_data_.bus[g];
            if (b < 0) continue;
            w[static_cast<size_t>(b)] += gen_data_.slack_weight[g];
            sum += gen_data_.slack_weight[g];
        }
        if (std::abs(sum) < 1e-12) {
            // Every participant off: the reference bus keeps the whole share
            // (the angle reference is a property of the batch, picked once).
            std::fill(w.begin(), w.end(), 0.0);
            if (ref_bus >= 0 && ref_bus < n_bus) w[static_cast<size_t>(ref_bus)] = 1.0;
        } else {
            for (double& x : w) x /= sum;
        }
        for (int k = 0; k < n_slack; ++k) {
            const int b = slack_bus[static_cast<size_t>(k)];
            row[k] = static_cast<cuda_real_type>(w[static_cast<size_t>(b)]);
            w[static_cast<size_t>(b)] = 0.0;   // consumed
        }
        // A survivor bus that is not a base participant cannot exist (the
        // survivors are a subset of the base participants); the degenerate
        // "reference gets 1" case can, when the reference carries no base
        // weight (several slacks) -- refuse rather than silently drop it.
        for (int b = 0; b < n_bus; ++b)
            if (w[static_cast<size_t>(b)] != 0.0)
                throw std::runtime_error(
                    "ScenarioSweepSession: row " + std::to_string(r) + " re-weights "
                    "the distributed slack onto bus " + std::to_string(b) + ", which "
                    "owns no slack column entry in the base case (a reference "
                    "bus with zero base weight). Keep at least one participating "
                    "generator connected on that row.");
    }
    return out;
}

// =============================================================================
// set_gen_v
// =============================================================================
void ScenarioSweepSession::set_gen_v(
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gen_v,
    Eigen::Ref<const Eigen::VectorXi> gen_bus)
{
    if (gen_v.cols() != gen_bus.size())
        throw std::runtime_error(
            "ScenarioSweepSession::set_gen_v: gen_v's column count must "
            "equal gen_bus's length (one entry per generator)");
    if (gen_v.rows() <= 0)
        throw std::runtime_error(
            "ScenarioSweepSession::set_gen_v: n_scenarios must be > 0");

    gen_v_     = gen_v;
    gen_bus_   = gen_bus;
    has_gen_v_ = true;
}

// =============================================================================
// set_branch_data
// =============================================================================
void ScenarioSweepSession::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff,
    Eigen::Ref<const CplxVect>        yft,
    Eigen::Ref<const CplxVect>        ytf,
    Eigen::Ref<const CplxVect>        ytt,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double sn_mva)
{
    h_branch_from_   = branch_from;
    h_branch_to_     = branch_to;
    h_yff_           = yff;
    h_yft_           = yft;
    h_ytf_           = ytf;
    h_ytt_           = ytt;
    h_bus_vn_kv_     = bus_vn_kv;
    sn_mva_          = sn_mva;
    has_branch_data_ = true;
}

// =============================================================================
// set_injections
// =============================================================================
void ScenarioSweepSession::set_injections(
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
    double sn_mva)
{
    const int n_bus = base_state_->n_bus;

    if (p_mw.rows() != q_mvar.rows() || p_mw.cols() != q_mvar.cols())
        throw std::runtime_error(
            "ScenarioSweepSession::set_injections: p_mw and q_mvar must have the same shape");
    if (p_mw.cols() != n_bus)
        throw std::runtime_error(
            "ScenarioSweepSession::set_injections: second dim must equal n_bus");
    if (p_mw.rows() <= 0)
        throw std::runtime_error(
            "ScenarioSweepSession::set_injections: n_scenarios must be > 0");
    if (sn_mva <= 0.0)
        throw std::runtime_error(
            "ScenarioSweepSession::set_injections: sn_mva must be > 0");

    p_mw_           = p_mw;
    q_mvar_         = q_mvar;
    sn_mva_         = sn_mva;
    n_scenarios_    = static_cast<int>(p_mw.rows());
    has_injections_ = true;
}

// =============================================================================
// set_topology
// =============================================================================
void ScenarioSweepSession::set_topology(
    const std::vector<std::vector<int>>& branch_ids_per_scenario)
{
    if (!has_branch_data_)
        throw std::runtime_error(
            "ScenarioSweepSession: call set_branch_data() before set_topology()");
    if (has_injections_
            && static_cast<int>(branch_ids_per_scenario.size()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession::set_topology: row count must match "
            "set_injections()'s n_scenarios");

    contingencies_.clear();
    contingencies_.reserve(branch_ids_per_scenario.size());
    tripped_branches_per_scenario_.clear();
    tripped_branches_per_scenario_.reserve(branch_ids_per_scenario.size());

    for (const auto& branch_ids : branch_ids_per_scenario) {
        contingencies_.push_back(build_contingency_from_branch_ids(
            branch_ids,
            h_branch_from_, h_branch_to_,
            h_yff_, h_yft_, h_ytf_, h_ytt_));
        tripped_branches_per_scenario_.push_back(branch_ids);
    }
    has_topology_ = true;
}

// =============================================================================
// set_limits
// =============================================================================
void ScenarioSweepSession::set_limits(
    Eigen::Ref<const RealVect> bus_vmin_kv,
    Eigen::Ref<const RealVect> bus_vmax_kv,
    Eigen::Ref<const RealVect> branch_limit_a1_ka,
    Eigen::Ref<const RealVect> branch_limit_a2_ka,
    int n_lines)
{
    const int n_bus = base_state_->n_bus;
    if (bus_vmin_kv.size() != n_bus || bus_vmax_kv.size() != n_bus)
        throw std::runtime_error(
            "ScenarioSweepSession::set_limits: bus_vmin_kv/bus_vmax_kv "
            "must have size n_bus");
    if (has_branch_data_) {
        const int n_bra = static_cast<int>(h_branch_from_.size());
        if (branch_limit_a1_ka.size() != n_bra || branch_limit_a2_ka.size() != n_bra)
            throw std::runtime_error(
                "ScenarioSweepSession::set_limits: branch_limit_a1_ka/"
                "branch_limit_a2_ka must have size n_branches");
    }

    h_bus_vmin_kv_ = bus_vmin_kv;
    h_bus_vmax_kv_ = bus_vmax_kv;
    h_branch_limit_a1_ka_ = branch_limit_a1_ka;
    h_branch_limit_a2_ka_ = branch_limit_a2_ka;
    n_lines_ = n_lines;
    has_limits_ = true;
    has_violations_result_ = false;
}

// =============================================================================
// run
// =============================================================================
void ScenarioSweepSession::run()
{
    if (!has_injections_)
        throw std::runtime_error(
            "ScenarioSweepSession: call set_injections() before run()");

    if (has_gen_v_ && static_cast<int>(gen_v_.rows()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession: set_gen_v()'s row count no longer "
            "matches set_injections()'s n_scenarios -- call set_gen_v() "
            "again after changing set_injections()");

    if (has_topology_
            && static_cast<int>(contingencies_.size()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession: set_topology()'s row count no longer "
            "matches set_injections()'s n_scenarios — call set_topology() "
            "again after changing set_injections()");

    if (handle_disconnected_grid_ &&
        strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
        throw std::runtime_error(
            "ScenarioSweepSession: handle_disconnected_grid is incompatible "
            "with the 'direct_base_case_factors' strategy (it reuses the unmasked "
            "base-case factors). Use 'direct_refactor_every' (default), "
            "'direct_iter0_only', or 'direct_refactor_every_n'.");

    if (has_gen_off_ && static_cast<int>(gen_off_.rows()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession: set_contingency_gens()'s row count no longer "
            "matches set_injections()'s n_scenarios -- call set_contingency_gens() "
            "again after changing set_injections()");

    // Generator contingencies: derive what the mask needs -- the buses that
    // must own a reserved Vm column + Q equation (union over rows), the
    // per-row PV→PQ releases, the per-row slack participants taken out -- and
    // rebuild the base state whenever the reserved set differs from the
    // current one (mirrors lightsim2grid's _maybe_prepare_gen_contingency,
    // whose "n" warm-up solve rebuilds the sparsity each compute()).
    std::vector<int>              required;
    std::vector<std::vector<int>> row_pv_to_pq, row_slack_off;
    _prepare_gen_contingency(required, row_pv_to_pq, row_slack_off);
    if (required != reserved_buses_)
        _build_base_state(required);

    if (has_gen_off_ &&
        strategy_type_ == ContingencySolverType::DirectBaseCaseFactors) {
        bool any_effect = !reserved_buses_.empty();
        for (const auto& v : row_slack_off) if (!v.empty()) { any_effect = true; break; }
        if (any_effect)
            throw std::runtime_error(
                "ScenarioSweepSession: set_contingency_gens is incompatible with "
                "the 'direct_base_case_factors' strategy (it reuses the base-case "
                "factors, which cannot release a bus' voltage pinning per row). "
                "Use 'direct_refactor_every' (default), 'direct_iter0_only', or "
                "'direct_refactor_every_n'.");
    }

    if (!has_topology_) {
        // Default: no branches tripped for any scenario (pure injection sweep).
        contingencies_.assign(static_cast<size_t>(n_scenarios_), Contingency{});
        tripped_branches_per_scenario_.assign(
            static_cast<size_t>(n_scenarios_), std::vector<int>{});
    }

    // Reset disconnected flags from any previous run() — contingencies_ is
    // mutated in place across runs.
    for (auto& ctg : contingencies_) {
        ctg.disconnected = false;
        ctg.masked_buses.clear();
        ctg.stranded_groups.clear();
        ctg.pinned_buses.clear();
    }

    // Per-row PV pins: every reserved bus stays PV (identity Q row) except
    // the ones this row turned PQ. Nothing reserved → nothing to pin.
    if (!reserved_buses_.empty()) {
        for (int r = 0; r < n_scenarios_; ++r) {
            const std::vector<int>& to_pq = row_pv_to_pq[static_cast<size_t>(r)];
            std::vector<int>& pinned = contingencies_[static_cast<size_t>(r)].pinned_buses;
            for (int b : reserved_buses_)
                if (!std::binary_search(to_pq.begin(), to_pq.end(), b)) pinned.push_back(b);
        }
    }
    std::vector<cuda_real_type> h_slack_w_rows = _row_slack_weights(row_slack_off);

    const int n_bus = base_state_->n_bus;

    // Build host-side per-unit complex Sbus_all, ORIGINAL (pre-compaction) row
    // order — ScenarioSweepBatch permutes into active-slot order internally.
    auto t_sbus_start = std::chrono::steady_clock::now();
    std::vector<cudaComplexType> h_Sbus_all(
        static_cast<size_t>(n_scenarios_) * static_cast<size_t>(n_bus));
    const double inv_sn = 1.0 / sn_mva_;
    for (int s = 0; s < n_scenarios_; ++s) {
        for (int b = 0; b < n_bus; ++b) {
            h_Sbus_all[static_cast<size_t>(s) * n_bus + b] =
                CudaFunHelper::my_make_cuComplex(
                    static_cast<cuda_real_type>(p_mw_(s, b)   * inv_sn),
                    static_cast<cuda_real_type>(q_mvar_(s, b) * inv_sn));
        }
    }
    t_sbus_build_ms_ = ms_since(t_sbus_start);

    // Host preprocessing (resolve_indices + connectivity/masking +
    // build_flat_patches + Sbus active-order permute), mutates contingencies_
    // in-place so disconnected flags are observable below.
    GenVOverride gen_v_override;
    if (has_gen_v_)
        gen_v_override = build_gen_v_override(gen_v_, gen_bus_, h_is_vm_fixed_bus_);

    ScenarioSweepBatch source(
        contingencies_,
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        Ybus_rm_,
        std::move(h_Sbus_all),
        batch_size_,
        mask_cfg_,
        handle_disconnected_grid_,
        std::move(gen_v_override),
        std::move(h_slack_w_rows),
        base_state_->n_slack);
    used_batch_size_ = source.used_batch_size();

    // (Re-)construct the solver — allows run() to be called multiple times.
    solver_ = std::make_unique<ScenarioSweepSolver>(
        *base_state_,
        std::move(source),
        n_scenarios_,
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        used_batch_size_,
        nb_iter_,
        strategy_type_,
        refactor_period_,
        reordering_alg_,
        matching_alg_,
        pivot_epsilon_alg_,
        scaling_max_voltage_change_,
        max_dVa_,
        max_dVm_);

    // compute_limit_violations: the fused per-chunk kernel needs branch
    // admittances + limits on device BEFORE solve() runs its chunk loop
    // (unlike compute_flows(), which uploads AFTER solve() and only for
    // callers who explicitly want full flows). Mirrors
    // ContingencyAnalysisSession::run() exactly.
    double t_admittance_upload_ms = 0.;
    double t_limits_setup_ms      = 0.;
    if (compute_limit_violations_) {
        if (!has_branch_data_)
            throw std::runtime_error(
                "ScenarioSweepSession: compute_limit_violations requires "
                "set_branch_data() to have been called first.");
        if (!has_limits_)
            throw std::runtime_error(
                "ScenarioSweepSession: compute_limit_violations requires "
                "set_limits() to have been called first.");

        solver_->upload_branch_admittances(
            h_branch_from_, h_branch_to_, h_yff_, h_yft_, h_ytf_, h_ytt_,
            h_bus_vn_kv_, sn_mva_);
        t_admittance_upload_ms = solver_->branch_data_upload_ms();

        solver_->set_violation_limits(
            h_bus_vmin_kv_, h_bus_vmax_kv_,
            h_branch_limit_a1_ka_, h_branch_limit_a2_ka_,
            violation_tol_, violation_capacity_, n_lines_);
        t_limits_setup_ms = solver_->violation_setup_ms();
    }

    timings_ = solver_->solve();
    timings_.t_base_case_ms  = t_base_case_ms_;
    timings_.t_preprocess_ms += base_state_->timings.t_build_J_ms;
    timings_.t_alloc_ms      += base_state_->timings.t_upload_ms;
    if (compute_limit_violations_) {
        timings_.t_branch_data_upload_ms += t_admittance_upload_ms;
        timings_.t_violation_setup_ms    += t_limits_setup_ms;
    }
    timings_.t_context_init_ms += base_state_->timings.t_context_init_ms;
    timings_.t_base_case_solve_only_ms =
        t_base_case_ms_ - base_state_->timings.t_build_J_ms
                         - base_state_->timings.t_upload_ms
                         - base_state_->timings.t_context_init_ms;
    timings_.t_ground_truth_check_ms = base_state_->timings.t_ground_truth_check_ms;

    // Scenarios whose topology change disconnects the grid are compacted out
    // of the batch by the source and never solved; the driver pre-fills their
    // result slots with NaN. Count them for the timing report.
    int n_disconnected = 0;
    for (const auto& ctg : contingencies_)
        if (ctg.disconnected) ++n_disconnected;
    timings_.n_disconnected = n_disconnected;
    has_violations_result_ = compute_limit_violations_;

    solver_->cs.synchronize();
}

// =============================================================================
// compute_flows
// =============================================================================
void ScenarioSweepSession::compute_flows()
{
    if (!has_branch_data_)
        throw std::runtime_error(
            "ScenarioSweepSession: call set_branch_data() before compute_flows()");
    if (!solver_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() before compute_flows()");

    solver_->set_branch_data(
        h_branch_from_, h_branch_to_,
        h_yff_, h_yft_, h_ytf_, h_ytt_,
        h_bus_vn_kv_, sn_mva_);
    timings_.t_branch_data_upload_ms += solver_->branch_data_upload_ms();

    const int n_scen = solver_->n_contingencies;
    const int n_bra  = solver_->n_branches_;
    const int n_bus  = base_state_->n_bus;
    const int total  = n_scen * n_bra;
    const cudaStream_t cs = solver_->cs;

    CudaTimer flow_timer(cs);
    flow_timer.start();

    compute_branch_flows_kernel<<<(total + SESSION_BS - 1) / SESSION_BS, SESSION_BS, 0, cs>>>(
        thrust::raw_pointer_cast(solver_->d_V_results.data()),
        thrust::raw_pointer_cast(solver_->d_branch_from.data()),
        thrust::raw_pointer_cast(solver_->d_branch_to.data()),
        thrust::raw_pointer_cast(solver_->d_yff.data()),
        thrust::raw_pointer_cast(solver_->d_yft.data()),
        thrust::raw_pointer_cast(solver_->d_ytf.data()),
        thrust::raw_pointer_cast(solver_->d_ytt.data()),
        thrust::raw_pointer_cast(solver_->d_base_current_A.data()),
        thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
        thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
        n_bus, n_bra, 0, n_scen, /*d_result_map=*/nullptr);

    // Zero flows for each scenario's own tripped branches (they carry no
    // current by definition) — same pattern as ContingencyAnalysisSession.
    std::vector<int> h_zero;
    for (int s = 0; s < n_scen; ++s)
        for (int l : tripped_branches_per_scenario_[static_cast<size_t>(s)])
            h_zero.push_back(s * n_bra + l);

    thrust::device_vector<int> d_zero;   // kept alive until after sync
    if (!h_zero.empty()) {
        const int n_z = static_cast<int>(h_zero.size());
        upload_h2d(d_zero, h_zero.data(), static_cast<size_t>(n_z), cs);
        zero_branch_flows_kernel<<<(n_z + SESSION_BS - 1) / SESSION_BS, SESSION_BS, 0, cs>>>(
            thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
            thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
            thrust::raw_pointer_cast(d_zero.data()),
            n_z);
    }

    timings_.t_flow_computation += flow_timer.stop_ms();
    // d_zero destroyed here, after sync (stop_ms() synchronizes)

    const int n = n_scen * n_bra;
    auto t_copy_start = std::chrono::steady_clock::now();
    {
        thrust::host_vector<cuda_real_type> h_or = solver_->d_or_amps_results;
        thrust::host_vector<cuda_real_type> h_ex = solver_->d_ex_amps_results;
        h_or_amps_.resize(n);
        h_ex_amps_.resize(n);
        for (int i = 0; i < n; ++i) {
            h_or_amps_(i) = static_cast<eigen_real_type>(h_or[static_cast<size_t>(i)]);
            h_ex_amps_(i) = static_cast<eigen_real_type>(h_ex[static_cast<size_t>(i)]);
        }
    }
    timings_.t_copy_flows_to_host_ms = ms_since(t_copy_start);
}

// =============================================================================
// Metadata
// =============================================================================
int ScenarioSweepSession::n_scenarios() const
{
    return solver_ ? solver_->n_contingencies : n_scenarios_;
}

int ScenarioSweepSession::n_bus() const
{
    return base_state_ ? base_state_->n_bus : 0;
}

int ScenarioSweepSession::n_branches() const
{
    return solver_ ? solver_->n_branches_
                   : static_cast<int>(h_branch_from_.size());
}

// =============================================================================
// Result accessors
// =============================================================================
CplxVect ScenarioSweepSession::get_V_results() const
{
    if (!solver_)
        throw std::runtime_error("ScenarioSweepSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    const int n = solver_->n_contingencies * base_state_->n_bus;
    thrust::host_vector<cudaComplexType> h_V = solver_->d_V_results;
    CplxVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].x),
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].y));
    timings_.t_copy_V_to_host_ms = ms_since(t_copy_start);
    return out;
}

RealVect ScenarioSweepSession::get_residuals() const
{
    if (!solver_)
        throw std::runtime_error("ScenarioSweepSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    const int n = solver_->n_contingencies;
    thrust::host_vector<cuda_real_type> h_res = solver_->d_residuals;
    RealVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = static_cast<eigen_real_type>(h_res[static_cast<size_t>(i)]);
    timings_.t_copy_residuals_to_host_ms = ms_since(t_copy_start);
    return out;
}

RealVect ScenarioSweepSession::get_or_amps() const
{
    if (h_or_amps_.size() == 0)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() and compute_flows() first");
    return h_or_amps_;
}

RealVect ScenarioSweepSession::get_ex_amps() const
{
    if (h_ex_amps_.size() == 0)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() and compute_flows() first");
    return h_ex_amps_;
}

Eigen::VectorXi ScenarioSweepSession::get_disconnected() const
{
    Eigen::VectorXi out(static_cast<Eigen::Index>(contingencies_.size()));
    for (size_t i = 0; i < contingencies_.size(); ++i)
        out(static_cast<Eigen::Index>(i)) = contingencies_[i].disconnected ? 1 : 0;
    return out;
}

// =============================================================================
// compute_limit_violations D→H result accessors
// =============================================================================
Eigen::VectorXi ScenarioSweepSession::get_violation_element_type() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_element_type;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_element_id() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_element_id;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_side() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_side;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_type() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_type;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

RealVect ScenarioSweepSession::get_violation_value() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<cuda_real_type> h = solver_->d_viol_value;
    RealVect out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(h[i]);
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

RealVect ScenarioSweepSession::get_violation_limit() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<cuda_real_type> h = solver_->d_viol_limit;
    RealVect out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(h[i]);
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_count() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_truncated() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_truncated;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_count_low_voltage() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_low_voltage;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_count_high_voltage() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_high_voltage;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ScenarioSweepSession::get_violation_count_current() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ScenarioSweepSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_current;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}
