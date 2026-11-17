// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_session.cu
// =============================================================================

#include "contingency_analysis_session.hpp"

#include <deque>
#include <map>
#include <tuple>
#include "gen_contingency_rows.hpp"   // gen_contingency::prepare_rows / check_rows / row_slack_weights
#include "ledger_extend.hpp"     // move_reference, ledger_reference_bus, add_switchable_vm_buses
#include "contingency/physical_checks_impl.cuh"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/contingency_batch.cuh"
#include "acpf_nr_kernels.cuh"
#include "contingency_analysis_helper.hpp"
#include "ledger_data.hpp"
#include "mask_config_builder.cuh"   // build_mask_config
#include "cuda_utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <algorithm>   // std::max, std::sort, std::set_union
#include <iterator>    // std::back_inserter
#include <cmath>       // std::abs
#include <stdexcept>
#include <string>
#include <limits>
#include <utility>   // std::move

static constexpr int SESSION_BS = 256;

// the device copy of the generators each contingency trips (n_ctg x n_gen uint8),
// what the physical checks read a disconnected machine off
struct ContingencyAnalysisDeviceData {
    thrust::device_vector<unsigned char> d_gen_off;
    int n_gen = 0;
    thrust::device_vector<cudaComplexType> d_V_groups;   // n_v_groups_ x n_bus starting voltages
};

// =============================================================================
// Destructor — defined here where AcPfNrState/ContingencyAnalysisSolver
// are complete types, enabling unique_ptr to call delete correctly.
// =============================================================================
ContingencyAnalysisSession::~ContingencyAnalysisSession() = default;

// =============================================================================
// Constructor
// =============================================================================
ContingencyAnalysisSession::ContingencyAnalysisSession(
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
    int ir_n_steps,
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
    , ir_n_steps_(ir_n_steps)
    , scaling_max_voltage_change_(scaling_max_voltage_change)
    , max_dVa_(max_dVa)
    , max_dVm_(max_dVm)
{
    (void)slack_ids;
    (void)slack_weights;
    h_Sbus_base_ = Sbus;   // what the redistribute_slack pre-pass reads a row's island off
    Ybus_cm_         = Ybus;
    Vinit_           = Vinit;
    pv_              = pv;
    pq_              = pq;
    max_iter_base_   = max_iter_base;
    tol_base_        = tol_base;
    device_          = device;
    presolved_v_     = presolved_v;
    debug_base_case_ = debug_base_case;
    if (ledger != nullptr) base_ledger_ = std::make_unique<LedgerData>(*ledger);
    grid_ref_bus_ = base_ledger_ ? ledger_reference_bus(*base_ledger_) : -1;
    ref_bus_      = grid_ref_bus_;
    dev_ = std::make_unique<ContingencyAnalysisDeviceData>();
    _build_base_state();
}

// =============================================================================
// _build_base_state — base-case NR on the stored inputs (the ledger's angle
// reference moved to ref_bus_ when the automatic reference picked another).
// =============================================================================
void ContingencyAnalysisSession::_build_base_state(const std::vector<int>& switchable_buses)
{
    solver_.reset();   // it references *base_state_
    const LedgerData* ledger_ptr = nullptr;
    LedgerData moved;
    if (base_ledger_) {
        ledger_ptr = base_ledger_.get();
        if ((ref_bus_ >= 0 && ref_bus_ != grid_ref_bus_) || !switchable_buses.empty() || reserved_vc_trip_
                || union_active_) {
            moved = *base_ledger_;
            // the ledger has the original buses until extend_ledger_union: their Ybus first
            const auto& Ybus_base_rm = union_active_ ? Ybus_rm_orig_ : Ybus_rm_;
            if (ref_bus_ >= 0 && ref_bus_ != grid_ref_bus_) move_reference(moved, ref_bus_, Ybus_base_rm);
            // bus moves (set_spare_buses): the spare buses and the union pattern, appended
            if (union_active_) {
                extend_ledger_union(moved, static_cast<int>(spare_parents_.size()), Ybus_rm_, Ybus_rm_orig_);
                // the outer loops: every spare switchable (PQ unless a row's moved generator
                // holds it PV, the model's controls per slot)
                if (outer_)
                    for (size_t k = 0; k < spare_parents_.size(); ++k)
                        moved.switchable_vm_buses.push_back(n_orig_bus_ + static_cast<int>(k));
            }
            // a generator trip releasing a bus it held alone: a Vm column + Q row for it,
            // pinned on the rows that keep it PV (see run())
            if (!switchable_buses.empty()) add_switchable_vm_buses(moved, switchable_buses, Ybus_rm_);
            if (reserved_vc_trip_) reserve_tripped_controller_slots(moved);
            ledger_ptr = &moved;
        }
    }
    auto t_base_start = std::chrono::steady_clock::now();
    base_state_ = std::make_unique<AcPfNrState>(
        Ybus_cm_, Vinit_, h_Sbus_base_, pv_, pq_,
        max_iter_base_,
        static_cast<eigen_real_type>(tol_base_),
        device_, ledger_ptr, presolved_v_,
        /*diag_stop_before_state_correction=*/false,
        reordering_alg_, matching_alg_, pivot_epsilon_alg_, ir_n_steps_,
        debug_base_case_, /*base_case_only=*/true,
        scaling_max_voltage_change_, max_dVa_, max_dVm_);
    t_base_case_ms_ = ms_since(t_base_start);
    ++base_state_generation_;

    // Build the handle_disconnected_grid mask configuration from the base case
    // (per-bus identity-row metadata + angle reference + VC row positions) and
    // the ledger (HVDC ends → skip-if-stranded, VC group topology → a group
    // whose controllers are all stranded is recovered instead). Cheap; only
    // consulted by run() when handle_disconnected_grid_ is enabled.
    mask_cfg_ = build_mask_config(*base_state_, ledger_ptr);
    reserved_buses_ = base_state_->h_switchable_buses;
    if (outer_ && ledger_ptr != nullptr)
        outer_->set_spare_buses(*ledger_ptr, union_active_ ? spare_grid_bus_ : std::vector<int>());
}

int ContingencyAnalysisSession::_wanted_reference_bus()
{
    if (grid_ref_bus_ < 0 || !auto_reference_slack_ || !handle_disconnected_grid_ || !has_contingencies_)
        return grid_ref_bus_;
    // bus moves (set_spare_buses): the rows' triplets name spare buses the base ledger the
    // choice reads does not have -- the grid's own reference is kept
    if (union_active_) return grid_ref_bus_;
    if (!auto_ref_valid_) {
        auto_ref_choice_ = choose_reference_bus_from_ledger(contingencies_, Ybus_rm_, *base_ledger_,
                                                            forced_ref_bus_, grid_ref_bus_);
        auto_ref_valid_ = true;
    }
    return auto_ref_choice_;
}

// =============================================================================
// set_branch_data
// =============================================================================
void ContingencyAnalysisSession::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff_eff,
    Eigen::Ref<const CplxVect>        yft_eff,
    Eigen::Ref<const CplxVect>        ytf_eff,
    Eigen::Ref<const CplxVect>        ytt_eff,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double sn_mva)
{
    h_branch_from_ = branch_from;
    h_branch_to_   = branch_to;
    h_yff_eff_         = yff_eff;
    h_yft_eff_         = yft_eff;
    h_ytf_eff_         = ytf_eff;
    h_ytt_eff_         = ytt_eff;
    h_bus_vn_kv_   = bus_vn_kv;
    sn_mva_        = sn_mva;
    has_branch_data_ = true;
}

// =============================================================================
// build_contingencies
// =============================================================================
void ContingencyAnalysisSession::build_contingencies(
    const std::vector<std::vector<int>>& branch_ids_per_ctg)
{
    if (!has_branch_data_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call set_branch_data() before build_contingencies()");

    const std::vector<std::vector<int>> none(branch_ids_per_ctg.size());
    build_contingencies_full(branch_ids_per_ctg, none, none, none);
}

namespace {
// sorted, without repeats: lightsim2grid keeps each list of a contingency in a std::set
std::vector<std::vector<int>> as_sets(const std::vector<std::vector<int>>& lists)
{
    std::vector<std::vector<int>> out(lists);
    for (auto& l : out) {
        std::sort(l.begin(), l.end());
        l.erase(std::unique(l.begin(), l.end()), l.end());
    }
    return out;
}
}  // namespace

void ContingencyAnalysisSession::build_contingencies_full(
    const std::vector<std::vector<int>>& branch_ids_per_ctg_in,
    const std::vector<std::vector<int>>& shunt_ids_per_ctg_in,
    const std::vector<std::vector<int>>& load_ids_per_ctg_in,
    const std::vector<std::vector<int>>& gen_ids_per_ctg_in,
    const std::vector<std::vector<int>>& branch_ids_side1_per_ctg_in,
    const std::vector<std::vector<int>>& branch_ids_side2_per_ctg_in,
    const std::vector<std::vector<int>>& sgen_ids_per_ctg_in,
    const std::vector<std::vector<int>>& storage_ids_per_ctg_in,
    const std::vector<std::vector<int>>& svc_ids_per_ctg_in,
    const std::vector<std::vector<int>>& hvdc_ids_per_ctg_in)
{
    // Each list as a set, and a branch opened at both ends one way or another (named in
    // branch_ids and at one end, or at both ends) a trip, as lightsim2grid's
    // _checked_contingency has it: subtracted twice, its admittance would leave -y.
    const std::vector<std::vector<int>> shunt_ids_per_ctg = as_sets(shunt_ids_per_ctg_in);
    const std::vector<std::vector<int>> load_ids_per_ctg  = as_sets(load_ids_per_ctg_in);
    const std::vector<std::vector<int>> gen_ids_per_ctg   = as_sets(gen_ids_per_ctg_in);
    const std::vector<std::vector<int>> sgen_ids_per_ctg    = as_sets(sgen_ids_per_ctg_in);
    const std::vector<std::vector<int>> storage_ids_per_ctg = as_sets(storage_ids_per_ctg_in);
    const std::vector<std::vector<int>> svc_ids_per_ctg     = as_sets(svc_ids_per_ctg_in);
    const std::vector<std::vector<int>> hvdc_ids_per_ctg    = as_sets(hvdc_ids_per_ctg_in);
    std::vector<std::vector<int>> branch_ids_per_ctg       = as_sets(branch_ids_per_ctg_in);
    std::vector<std::vector<int>> branch_ids_side1_per_ctg = as_sets(branch_ids_side1_per_ctg_in);
    std::vector<std::vector<int>> branch_ids_side2_per_ctg = as_sets(branch_ids_side2_per_ctg_in);
    if (branch_ids_side1_per_ctg.size() == branch_ids_per_ctg.size() &&
        branch_ids_side2_per_ctg.size() == branch_ids_per_ctg.size()) {
        for (size_t c = 0; c < branch_ids_per_ctg.size(); ++c) {
            auto& both = branch_ids_per_ctg[c];
            auto& s1 = branch_ids_side1_per_ctg[c];
            auto& s2 = branch_ids_side2_per_ctg[c];
            std::vector<int> on_both_sides;
            std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(on_both_sides));
            std::vector<int> merged;
            std::set_union(both.begin(), both.end(), on_both_sides.begin(), on_both_sides.end(),
                           std::back_inserter(merged));
            both = merged;
            const auto drop_tripped = [&](std::vector<int>& side) {
                side.erase(std::remove_if(side.begin(), side.end(),
                                          [&](int b) { return std::binary_search(both.begin(), both.end(), b); }),
                           side.end());
            };
            drop_tripped(s1);
            drop_tripped(s2);
        }
    } else {
        // one side list only (or none): a branch it shares with branch_ids is a trip
        for (auto* sides : {&branch_ids_side1_per_ctg, &branch_ids_side2_per_ctg})
            if (sides->size() == branch_ids_per_ctg.size())
                for (size_t c = 0; c < branch_ids_per_ctg.size(); ++c) {
                    const auto& both = branch_ids_per_ctg[c];
                    auto& side = (*sides)[c];
                    side.erase(std::remove_if(side.begin(), side.end(),
                                              [&](int b) { return std::binary_search(both.begin(), both.end(), b); }),
                               side.end());
                }
    }

    static const char* who = "ContingencyAnalysisSession::build_contingencies_full";
    if (!has_branch_data_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call set_branch_data() before build_contingencies()");
    const int n_ctg = static_cast<int>(branch_ids_per_ctg.size());
    if (static_cast<int>(shunt_ids_per_ctg.size()) != n_ctg || static_cast<int>(load_ids_per_ctg.size()) != n_ctg ||
        static_cast<int>(gen_ids_per_ctg.size()) != n_ctg)
        throw std::runtime_error(std::string(who) + ": the branch, shunt, load and generator lists must "
                                 "have one entry per contingency each");
    for (const auto* sides : {&branch_ids_side1_per_ctg, &branch_ids_side2_per_ctg})
        if (!sides->empty() && static_cast<int>(sides->size()) != n_ctg)
            throw std::runtime_error(std::string(who) + ": the branches opened at one end must be given for "
                                     "every contingency, or for none");
    bool any_shunt = false, any_load = false, any_gen = false;
    for (int c = 0; c < n_ctg; ++c) {
        any_shunt = any_shunt || !shunt_ids_per_ctg[c].empty();
        any_load  = any_load  || !load_ids_per_ctg[c].empty();
        any_gen   = any_gen   || !gen_ids_per_ctg[c].empty();
    }
    if ((any_shunt || any_load || any_gen) && !has_trip_data_)
        throw std::runtime_error(std::string(who) + ": tripping shunts, loads or generators needs the "
                                 "element data of a session built from a lightsim2grid grid "
                                 "(set_element_trip_data, set by the bridge factories)");
    if (any_gen && !has_gen_data_)
        throw std::runtime_error(std::string(who) + ": tripping generators needs the generator data of a "
                                 "session built from a lightsim2grid grid (set_gen_contingency_data)");
    for (int c = 0; c < n_ctg; ++c) {
        element_trips::check_ids(shunt_ids_per_ctg[c], trip_data_.n_shunt, "shunt", who);
        element_trips::check_ids(load_ids_per_ctg[c], trip_data_.n_load, "load", who);
        element_trips::check_ids(gen_ids_per_ctg[c], std::min(trip_data_.n_gen, gen_data_.n_gen), "generator", who);
    }

    for (const auto* more : {&sgen_ids_per_ctg, &storage_ids_per_ctg, &svc_ids_per_ctg, &hvdc_ids_per_ctg})
        if (!more->empty() && static_cast<int>(more->size()) != n_ctg)
            throw std::runtime_error(std::string(who) + ": the static generators, storage units, SVCs and hvdc "
                                     "lines must be given for every contingency, or for none");
    std::vector<MoreTrips> more_trips(static_cast<size_t>(n_ctg));
    for (int c = 0; c < n_ctg; ++c) {
        MoreTrips& t = more_trips[static_cast<size_t>(c)];
        if (!sgen_ids_per_ctg.empty())    t.sgens    = sgen_ids_per_ctg[c];
        if (!storage_ids_per_ctg.empty()) t.storages = storage_ids_per_ctg[c];
        if (!svc_ids_per_ctg.empty())     t.svcs     = svc_ids_per_ctg[c];
        if (!hvdc_ids_per_ctg.empty())    t.hvdcs    = hvdc_ids_per_ctg[c];
        if (t.any() && !has_trip_data_)
            throw std::runtime_error(std::string(who) + ": tripping these elements needs the element data of a "
                                     "session built from a lightsim2grid grid");
        if (t.any()) element_trips::check_ids(t, trip_data_, who);
    }

    // Built aside and committed at the end: a list refused below (a branch id out of
    // range) leaves the session's own contingencies as they were.
    std::vector<Contingency>      contingencies;
    std::vector<std::vector<int>> disconnected_per_ctg;
    decltype(ctg_sbus_ds_)        ctg_sbus_ds(static_cast<size_t>(n_ctg));
    contingencies.reserve(n_ctg);
    disconnected_per_ctg.reserve(n_ctg);

    for (int c = 0; c < n_ctg; ++c) {
        Contingency ctg = build_contingency_from_branch_ids(
            branch_ids_per_ctg[c],
            h_branch_from_, h_branch_to_,
            h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_);
        // the branches opened at one end only: their kept end stays, Kron-reduced
        if (!branch_ids_side1_per_ctg.empty())
            append_one_side_trips(ctg, branch_ids_side1_per_ctg[c], 1, h_branch_from_, h_branch_to_,
                                  h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_);
        if (!branch_ids_side2_per_ctg.empty())
            append_one_side_trips(ctg, branch_ids_side2_per_ctg[c], 2, h_branch_from_, h_branch_to_,
                                  h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_);
        // a shunt: its diagonal coefficient leaves the row's Ybus (merged with a branch
        // end on the same diagonal by build_flat_patches)
        for (const auto& t : element_trips::shunt_triplets(trip_data_, shunt_ids_per_ctg[c]))
            ctg.triplets.push_back({t.bus, t.bus, t.re, t.im});
        const MoreTrips& more = more_trips[static_cast<size_t>(c)];
        if (more.any())   // an SVC's b0 leaves the row's Ybus
            for (const auto& t : element_trips::svc_triplets(trip_data_, more.svcs))
                ctg.triplets.push_back({t.bus, t.bus, t.re, t.im});
        // a load / generator / ...: its injection leaves the row's Sbus
        if (!load_ids_per_ctg[c].empty() || !gen_ids_per_ctg[c].empty() || more.any())
            ctg_sbus_ds[static_cast<size_t>(c)] = element_trips::sbus_delta(
                trip_data_, load_ids_per_ctg[c], gen_ids_per_ctg[c],
                gen_data_.bus, gen_data_.vreg_on, gen_data_.target_q_mvar, &more);
        // every branch it opens, at one end or both: no flow reported
        disconnected_per_ctg.push_back(ctg.tripped_branches);
        contingencies.push_back(std::move(ctg));
    }
    contingencies_        = std::move(contingencies);
    disconnected_per_ctg_ = std::move(disconnected_per_ctg);
    ctg_sbus_ds_          = std::move(ctg_sbus_ds);
    row_pv_release_.clear();
    row_pv_set_.clear();
    row_gen_bus_.clear();
    clear_streaming();   // its parents index the old rows
    ctg_shunts_ = shunt_ids_per_ctg;
    ctg_loads_  = load_ids_per_ctg;
    ctg_gens_   = gen_ids_per_ctg;
    ctg_more_   = std::move(more_trips);
    has_gen_trips_ = any_gen;
    has_contingencies_ = true;
    auto_ref_valid_    = false;
    ++contingencies_generation_;   // the live driver's source is stale now
}

BatchDriverConfig ContingencyAnalysisSession::_current_config() const
{
    BatchDriverConfig c;
    c.n_scenarios                = static_cast<int>(contingencies_.size());
    c.batch_size                 = batch_size_;
    c.refactor_period            = refactor_period_;
    c.strategy                   = strategy_type_;
    c.reordering_alg             = reordering_alg_;
    c.linear_solver              = linear_solver_;
    c.matching_alg               = matching_alg_;
    c.pivot_epsilon_alg          = pivot_epsilon_alg_;
    c.ir_n_steps                 = ir_n_steps_;
    c.scaling_max_voltage_change = scaling_max_voltage_change_;
    c.max_dVa                    = max_dVa_;
    c.max_dVm                    = max_dVm_;
    c.mask_mode                  = handle_disconnected_grid_;
    c.base_state_generation      = base_state_generation_;
    c.rows_generation            = contingencies_generation_;
    c.scheduling                 = scheduling_;
    return c;
}

// =============================================================================
// run
// =============================================================================
void ContingencyAnalysisSession::run()
{
    const auto t_run0 = std::chrono::steady_clock::now();
    if (!has_contingencies_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call build_contingencies() before run()");

    if (handle_disconnected_grid_ &&
        strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
        throw std::runtime_error(
            "ContingencyAnalysisSession: handle_disconnected_grid is incompatible "
            "with the 'direct_base_case_factors' strategy (it reuses the unmasked "
            "base-case factors). Use 'direct_refactor_every' (default), "
            "'direct_iter0_only', or 'direct_refactor_every_n'.");
    const bool continuous = (scheduling_ == BatchScheduling::Continuous);
    if (continuous)   // before any allocation / cuDSS ANALYSIS
        check_continuous_scheduling(strategy_type_, nb_iter_, nb_iter_per_round_, tol_);
    check_tail_settings(tail_max_groups_, tail_group_size_);
    if (outer_) {
        auto refuse = [](const std::string& what) {
            throw std::runtime_error("ContingencyAnalysisSession: the outer loops do not support " + what + " yet");
        };
        check_outer_loops_scheduling("ContingencyAnalysisSession", continuous, nb_iter_per_round_);
        if (strategy_type_ != ContingencySolverType::DirectRefactorEvery)
            refuse("a strategy other than 'direct_refactor_every'");
        if (phys_.compute_physical_violations) refuse("compute_physical_violations");
    }

    // What the batch cannot do yet leaves the row out (NaN, BATCH_UNSUPPORTED), not the whole batch
    std::vector<std::string> unsupported(contingencies_.size());
    for (size_t c = 0; c < contingencies_.size(); ++c) {
        std::string why;
        for (int g : ctg_gens_[c]) {
            why = gen_contingency::gen_unsupported(gen_data_, g, phys_.compute_physical_violations, false);
            if (!why.empty()) break;
        }
        if (why.empty() && ctg_more_[c].any())
            why = element_trips::more_unsupported(trip_data_, ctg_more_[c], phys_.compute_physical_violations);
        // a regulating shunt's sections are the ShuntVoltageControl loop's, which does not
        // know of a row taking it out
        if (why.empty() && outer_)
            for (int sh : ctg_shunts_[c])
                if (trip_data_.shunt_regulating[static_cast<size_t>(sh)] && trip_data_.shunt_bus[static_cast<size_t>(sh)] >= 0) {
                    why = "disconnects shunt " + std::to_string(sh) + ", which regulates a voltage: the "
                          "ShuntVoltageControl loop owns its sections";
                    break;
                }
        std::vector<int> vc_tripped;
        if (why.empty() && base_ledger_) {
            // ... and the generators its actions move to another bus (set_row_pv_moves): one of a
            // voltage-control group leaves the group (the others regulate its bus on), and holds
            // its new bus on its own (a local PV bus, pinned per row)
            std::vector<int> gens = ctg_gens_[c];
            if (c < row_gen_bus_.size())
                for (const auto& gb : row_gen_bus_[c]) gens.push_back(gb.first);
            std::sort(gens.begin(), gens.end());
            gens.erase(std::unique(gens.begin(), gens.end()), gens.end());
            vc_tripped = gen_contingency::tripped_controllers(*base_ledger_, gens, ctg_more_[c].svcs, ctg_more_[c].hvdcs);
            why = gen_contingency::tripped_controllers_unsupported(*base_ledger_, vc_tripped);
        }
        if (!why.empty()) vc_tripped.clear();
        contingencies_[c].vc_tripped = vc_tripped;
        unsupported[c] = why;
    }

    // Generator trips: the buses some contingency releases (they need a reserved
    // Vm column + Q equation), each contingency's releases and the slack
    // participants it takes out (lightsim2grid's _maybe_prepare_gen_contingency)
    const int n_ctg_all = static_cast<int>(contingencies_.size());
    const auto ctg_gen_off = [this, &unsupported](int r, int g) {
        if (!unsupported[static_cast<size_t>(r)].empty()) return false;
        const std::vector<int>& gens = ctg_gens_[static_cast<size_t>(r)];
        return std::find(gens.begin(), gens.end(), g) != gens.end();
    };
    const auto ctg_sto_off = [this, &unsupported](int r, int s) {
        if (!unsupported[static_cast<size_t>(r)].empty()) return false;
        const std::vector<int>& ids = ctg_more_[static_cast<size_t>(r)].storages;
        return std::find(ids.begin(), ids.end(), s) != ids.end();
    };
    const auto ctg_hvdc_off = [this, &unsupported](int r, int h) {
        if (!unsupported[static_cast<size_t>(r)].empty()) return false;
        const std::vector<int>& ids = ctg_more_[static_cast<size_t>(r)].hvdcs;
        return std::find(ids.begin(), ids.end(), h) != ids.end();
    };
    bool any_controller_trip = has_gen_trips_;
    for (const MoreTrips& t : ctg_more_) any_controller_trip = any_controller_trip || !t.storages.empty() || !t.hvdcs.empty() || !t.svcs.empty();
    std::vector<int>              required;
    std::vector<std::vector<int>> row_slack_off;
    row_pv_to_pq_.assign(static_cast<size_t>(n_ctg_all), std::vector<int>());
    row_slack_off.assign(static_cast<size_t>(n_ctg_all), std::vector<int>());
    if (any_controller_trip) {
        gen_contingency::prepare_rows(gen_data_, n_ctg_all, ctg_gen_off, ctg_sto_off, ctg_hvdc_off, base_ledger_.get(),
                                      required, row_pv_to_pq_, row_slack_off);
        if (outer_ && !required.empty()) {
            // a row releasing a bus the outer loops' ledger did not reserve (ReactiveLimits
            // reserves every bus its own machines hold)
            for (size_t r = 0; r < row_pv_to_pq_.size(); ++r) {
                for (int b : row_pv_to_pq_[r])
                    if (std::binary_search(required.begin(), required.end(), b)) {
                        unsupported[r] = "releases bus " + std::to_string(b) + ", which the outer loops' ledger has not reserved";
                        break;
                    }
                if (!unsupported[r].empty()) { row_pv_to_pq_[r].clear(); row_slack_off[r].clear(); }
            }
            required.clear();
        }
        if (strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
            throw std::runtime_error(
                "ContingencyAnalysisSession: tripping generators is incompatible with the "
                "'direct_base_case_factors' strategy (it reuses the base-case factors, which cannot "
                "release a bus' voltage pinning per row). Use 'direct_refactor_every' (default), "
                "'direct_iter0_only', or 'direct_refactor_every_n'.");
    }

    // generator moves (set_row_pv_moves): the PV buses a row releases are reserved as a
    // generator trip's (PQ on that row, pinned PV on the others); the buses it holds PV are
    // pinned below
    if (!row_pv_release_.empty() || !row_pv_set_.empty()) {
        for (size_t r = 0; r < contingencies_.size(); ++r) {
            const bool moves = (r < row_pv_release_.size() && !row_pv_release_[r].empty())
                            || (r < row_pv_set_.size() && !row_pv_set_[r].empty());
            if (!moves || !unsupported[r].empty()) continue;
            if (outer_) {
                // the outer loops' ledger reserved its switchable buses up front: a released bus
                // and a bus held PV must be among them (a moved generator's own bus is: every bus
                // a machine holds, every spare)
                const std::vector<int>& sw = base_state_->h_switchable_buses;
                const auto reserved = [&sw](int b) { return std::binary_search(sw.begin(), sw.end(), b); };
                for (int b : r < row_pv_release_.size() ? row_pv_release_[r] : std::vector<int>())
                    if (!reserved(b)) { unsupported[r] = "releases bus " + std::to_string(b) + ", which the outer loops' ledger has not reserved"; break; }
                for (const auto& bv : r < row_pv_set_.size() ? row_pv_set_[r] : std::vector<std::pair<int, double>>())
                    if (unsupported[r].empty() && !reserved(bv.first))
                        unsupported[r] = "holds bus " + std::to_string(bv.first) + " PV, which the outer loops' ledger has not reserved";
                if (!unsupported[r].empty()) continue;
            }
            if (r >= row_pv_release_.size()) continue;
            std::vector<int>& to_pq = row_pv_to_pq_[r];
            for (int b : row_pv_release_[r]) { to_pq.push_back(b); if (!outer_) required.push_back(b); }
            std::sort(to_pq.begin(), to_pq.end());
            to_pq.erase(std::unique(to_pq.begin(), to_pq.end()), to_pq.end());
        }
        std::sort(required.begin(), required.end());
        required.erase(std::unique(required.begin(), required.end()), required.end());
        if (!required.empty() && strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
            throw std::runtime_error("ContingencyAnalysisSession: a generator moved off its bus is incompatible "
                                     "with the 'direct_base_case_factors' strategy (it cannot release a bus' "
                                     "voltage pinning per row)");
    }

    // a row's status is settled with its masks: a change means a new source
    if (unsupported != row_unsupported_) { row_unsupported_ = unsupported; ++contingencies_generation_; }
    for (size_t c = 0; c < contingencies_.size(); ++c) contingencies_[c].skip = !row_unsupported_[c].empty();

    // Automatic reference slack: the participant the fewest contingencies
    // strand; a change moves it in the ledger and rebuilds the base state --
    // as does a change of the buses the generator trips release (the outer loops'
    // ledger carries its own)
    const int want_ref = _wanted_reference_bus();
    bool want_vc_trip = false;
    for (size_t c = 0; c < contingencies_.size(); ++c)
        want_vc_trip = want_vc_trip || (!contingencies_[c].vc_tripped.empty() && row_unsupported_[c].empty());
    if (want_ref != ref_bus_ || (!outer_ && (required != reserved_buses_ || want_vc_trip != reserved_vc_trip_))) {
        reserved_vc_trip_ = want_vc_trip;
        ref_bus_ = want_ref;
        _build_base_state(outer_ ? std::vector<int>() : required);
    }

    // Hot run: same settings, same contingency list as the live driver --
    // keep it (allocations, block-diagonal structure, cuDSS ANALYSIS, the
    // policy's factorization state) and its source (connectivity, masks,
    // patches already on the device); only solve() runs again. Otherwise
    // build both, as the first run() does. The continuous schedule reuses them
    // too (each solve() reloads every slot from the source's per-row data),
    // except with the outer loops, whose repeated runs on one driver are not
    // tested yet.
    const BatchDriverConfig cfg = _current_config();
    const bool cold = !solver_ || cfg != driver_cfg_ || (continuous && outer_);
    const double t_sess_ref_ms = ms_since(t_run0);
    double t_sess_source_ms = 0., t_sess_prepass_ms = 0., t_sess_driver_ctor_ms = 0.;

    if (cold) {
        // Reset any masking flags set on a previous run() (the contingency
        // list is mutated in place across runs; compute_component_masks /
        // check_connectivity expect to start from a clean state). A hot run
        // keeps them: they describe the live source.
        // lightsim2grid's held controllers (set_hold_frozen_regulators) are
        // pinned on every contingency, as in the grid's own solve (AcPfNrState
        // pins them too)
        // (the outer loops pin every held controller per slot themselves)
        std::vector<std::pair<int, double>> held_pins;
        for (size_t j = 0; j < base_state_->h_vc_held.size() && !outer_; ++j)
            if (base_state_->h_vc_held[j])
                held_pins.emplace_back(static_cast<int>(j), base_state_->h_vc_q_held[j]);
        for (auto& ctg : contingencies_) {
            ctg.disconnected = false;
            ctg.masked_buses.clear();
            ctg.stranded_groups.clear();
            ctg.pinned_buses.clear();
            ctg.vm_reseed.clear();
            ctg.vc_pinned_ctrl = held_pins;
        }
        // the generator trips: every reserved bus stays PV (identity Q row) on a
        // contingency that does not release it
        bool any_gen_pin = false;
        if (!reserved_buses_.empty() && !outer_) {
            for (int r = 0; r < n_ctg_all; ++r) {
                const std::vector<int>& to_pq = row_pv_to_pq_[static_cast<size_t>(r)];
                std::vector<int>& pinned = contingencies_[static_cast<size_t>(r)].pinned_buses;
                for (int b : reserved_buses_)
                    if (!std::binary_search(to_pq.begin(), to_pq.end(), b)) pinned.push_back(b);
                if (!pinned.empty()) any_gen_pin = true;
            }
        }
        // ... and the buses a row's moved generators hold PV: Q row pinned, |V| at the set-point
        if (!outer_)
            for (size_t r = 0; r < row_pv_set_.size() && r < contingencies_.size(); ++r) {
                if (!row_unsupported_[r].empty()) continue;
                Contingency& ctg = contingencies_[r];
                for (const auto& bv : row_pv_set_[r]) {
                    ctg.pinned_buses.push_back(bv.first);
                    ctg.vm_reseed.push_back(bv);
                    any_gen_pin = true;
                }
            }

        // Host preprocessing (resolve_indices + connectivity/masking +
        // build_flat_patches) captured by the BatchSource; mutates
        // contingencies_ in-place so that disconnected flags are observable
        // below for the disconnected count. The source rebalances the chunk
        // size over the ACTIVE (simulated) count; we read it back so the
        // driver chunks consistently with the source.
        const auto t_source0 = std::chrono::steady_clock::now();
        // bus moves (set_spare_buses): every row has empty buses (the spare ones it does not
        // use, at least), always masked
        bool any_empty = !spare_parents_.empty();
        for (const auto& ctg : contingencies_) any_empty = any_empty || !ctg.empty_buses.empty();
        ContingencyBatch source(
            contingencies_,
            Ybus_rm_.outerIndexPtr(),
            Ybus_rm_.innerIndexPtr(),
            Ybus_rm_,
            batch_size_,
            (handle_disconnected_grid_ || !held_pins.empty() || any_gen_pin || any_empty) ? &mask_cfg_ : nullptr,
            handle_disconnected_grid_,
            static_cast<int>(spare_parents_.size()),
            union_active_);
        t_sess_source_ms = ms_since(t_source0);
        const auto t_prepass0 = std::chrono::steady_clock::now();
        // Continuous: batch_size_ slots, never more than the simulated rows.
        used_batch_size_ = continuous ? std::max(1, std::min(batch_size_, source.n_active()))
                                      : source.used_batch_size();
        // the outer loops correct each row's Sbus: one per slot
        source.slot_sbus_ = static_cast<bool>(outer_);

        // redistribute_slack: the pre-pass of what each contingency's island took
        // out, on the masks the source just computed -- handed to the source
        // before its driver uploads it.
        // Each contingency's own Sbus correction (the loads / generators it trips) and
        // distributed slack (without the participants it trips), and with
        // redistribute_slack the pre-pass of what it loses -- its island, its tripped
        // elements -- merged into that correction.
        const int n_bus = base_state_->n_bus;
        int ref_bus = -1;
        for (int b = 0; b < n_bus && b < static_cast<int>(mask_cfg_.is_reference_bus.size()); ++b)
            if (mask_cfg_.is_reference_bus[static_cast<size_t>(b)]) { ref_bus = b; break; }
        std::vector<cuda_real_type> w_rows;
        std::vector<std::vector<std::pair<int, std::complex<double>>>> ds_rows = ctg_sbus_ds_;
        // With the outer loops, the Newton has a single slack: the DistributedSlack loop shares
        // what a row loses before its first solve (RowOverrides::lost_p_mw), from the targets in
        // force -- lightsim2grid's batch does the same
        row_lost_p_mw_.clear();
        if (redistribute_slack_) {
            _check_redistribute_slack();
            const double sn = slack_rd_.sn_mva;
            slack_rows_.assign(contingencies_.size(), slack_redistribution::RowResult{});
            if (outer_) row_lost_p_mw_.assign(contingencies_.size(), std::numeric_limits<double>::quiet_NaN());
            std::vector<char> is_masked(static_cast<size_t>(n_bus), 0);
            std::vector<double> row_p(static_cast<size_t>(n_bus));
            for (int r = 0; r < n_ctg_all; ++r) {
                const Contingency& ctg = contingencies_[static_cast<size_t>(r)];
                if (ctg.disconnected) continue;
                const auto g_off = [&](int g) { return ctg_gen_off(r, g); };
                const auto g_p   = [this](int g) { return static_cast<double>(slack_rd_.gens_target_p_mw(g)); };
                const auto s_off = [&](int s) { return ctg_sto_off(r, s); };
                // the island read off the row's own Sbus: a tripped load / generator on it nets to 0
                for (int b : ctg.masked_buses) row_p[static_cast<size_t>(b)] = static_cast<double>(h_Sbus_base_(b).real()) * sn;
                for (const auto& bd : ctg_sbus_ds_[static_cast<size_t>(r)])
                    row_p[static_cast<size_t>(bd.first)] = static_cast<double>(h_Sbus_base_(bd.first).real()) * sn + bd.second.real() * sn;
                for (int b : ctg.masked_buses) is_masked[static_cast<size_t>(b)] = 1;
                const auto bus_masked = [&](int b) { return b >= 0 && b < n_bus && is_masked[static_cast<size_t>(b)] != 0; };
                const double lost =
                    slack_redistribution::lost_mw(slack_rd_, ctg.masked_buses, g_off, g_p,
                                                  [&](int b) { return row_p[static_cast<size_t>(b)]; })
                    + (has_trip_data_ ? element_trips::trip_lost_mw(trip_data_, ctg_loads_[static_cast<size_t>(r)],
                                                                    ctg_shunts_[static_cast<size_t>(r)], bus_masked)
                                        + element_trips::more_lost_mw(trip_data_, ctg_more_[static_cast<size_t>(r)])
                                      : 0.);
                if (outer_) {
                    // the loop gets the row's loss even when it is nothing at the grid's targets: it counts
                    // the units the row loses at the targets in force, which a row resumed from
                    // the "n" solve takes from that solve, and decides itself what is left to share
                    row_lost_p_mw_[static_cast<size_t>(r)] = lost;
                } else {
                    slack_rows_[static_cast<size_t>(r)] = slack_redistribution::redistribute_row(
                        slack_rd_, lost, bus_masked, g_off, g_p, slack_redistribution::DEFAULT_EPS_MW, s_off);
                }
                for (int b : ctg.masked_buses) is_masked[static_cast<size_t>(b)] = 0;
            }
            if (outer_) slack_rows_.clear();
        }
        if (redistribute_slack_ && !outer_) {
            w_rows = slack_redistribution::build_row_weights(
                slack_rd_, slack_rows_, n_ctg_all, ctg_gen_off, n_bus,
                base_state_->h_slack_bus, base_state_->h_slack_w, ref_bus, ctg_sto_off);
            for (size_t r = 0; r < slack_rows_.size(); ++r) {
                // one entry per bus: the pre-pass' dP joins the trips' dS
                std::vector<std::pair<int, std::complex<double>>>& ds = ds_rows[r];
                for (const auto& bd : slack_rows_[r].dp_pu) ds.emplace_back(bd.first, std::complex<double>(bd.second, 0.));
                std::sort(ds.begin(), ds.end(), [](const std::pair<int, std::complex<double>>& a,
                                                   const std::pair<int, std::complex<double>>& b) { return a.first < b.first; });
                std::vector<std::pair<int, std::complex<double>>> merged;
                for (const auto& e : ds) {
                    if (!merged.empty() && merged.back().first == e.first) merged.back().second += e.second;
                    else merged.push_back(e);
                }
                ds.swap(merged);
            }
        } else {
            slack_rows_.clear();
            // with the outer loops, a distributed-slack Newton's weights are its model's, per slot
            if (!outer_)
                w_rows = gen_contingency::row_slack_weights(gen_data_, row_slack_off, n_ctg_all, n_bus,
                                                            base_state_->h_slack_bus, base_state_->h_slack_w,
                                                            ref_bus, "ContingencyAnalysisSession", ctg_sto_off);
        }
        bool any_ds = false;
        for (const auto& ds : ds_rows) if (!ds.empty()) { any_ds = true; break; }
        // the lines in AC emulation each contingency trips (the outer loops own the slots' regimes)
        if (!outer_) {
            std::vector<std::vector<int>> droop_off(contingencies_.size());
            for (size_t r = 0; r < contingencies_.size(); ++r)
                if (row_unsupported_[r].empty() && !ctg_more_[r].hvdcs.empty())
                    droop_off[r] = element_trips::droop_entries(base_state_->h_hvdc_id, ctg_more_[r].hvdcs);
            source.set_hvdc_off_host(base_state_->h_hvdc_status, droop_off);
            // ... and the controllers each has lost: tripped, or on a bus it masks
            if (reserved_vc_trip_) {
                std::vector<std::vector<std::pair<int, double>>> lost(contingencies_.size());
                for (size_t r = 0; r < contingencies_.size(); ++r)
                    lost[r] = gen_contingency::held_by_value(
                        *base_ledger_, gen_contingency::lost_controllers(*base_ledger_, contingencies_[r].vc_tripped,
                                                                         contingencies_[r].masked_buses),
                        contingencies_[r].vc_pinned_ctrl);
                source.set_vc_hold_host(base_state_->n_vc_ctrl, lost);
            }
        }
        if (!w_rows.empty() || any_ds)
            source.set_row_sbus_host(w_rows, base_state_->n_slack,
                                     any_ds ? ds_rows : std::vector<std::vector<std::pair<int, std::complex<double>>>>());

        t_sess_prepass_ms = ms_since(t_prepass0);
        // New driver: allocation, block-diagonal structure, cuDSS ANALYSIS.
        // Free the old one first so the two never coexist on the device.
        const auto t_ctor0 = std::chrono::steady_clock::now();
        solver_.reset();
        solver_ = std::make_unique<ContingencyAnalysisSolver>(
            *base_state_,
            std::move(source),
            static_cast<int>(contingencies_.size()),
            used_batch_size_,
            nb_iter_,
            strategy_type_,
            refactor_period_,
            reordering_alg_,
            matching_alg_,
            pivot_epsilon_alg_,
            ir_n_steps_,
            scaling_max_voltage_change_,
            max_dVa_,
            max_dVm_,
            scheduling_,
            nb_iter_per_round_,
            tol_,
            linear_solver_);
        t_sess_driver_ctor_ms = ms_since(t_ctor0);
        driver_cfg_ = cfg;
        ++driver_build_counter_;
    } else {
        // One-time costs (allocation, ANALYSIS, source preprocessing) are
        // not paid again: report them as 0 for this run.
        solver_->mark_reused(/*hot=*/true);
    }
    const auto t_overrides0 = std::chrono::steady_clock::now();
    // per-run knobs, no rebuild needed
    solver_->nb_iter_           = nb_iter_;
    solver_->nb_iter_per_round_ = nb_iter_per_round_;
    solver_->tol_               = tol_;
    solver_->tail_max_groups_   = tail_max_groups_;
    solver_->tail_group_size_   = tail_group_size_;

    // compute_limit_violations: the fused per-chunk kernel needs branch
    // admittances + limits on device BEFORE solve() runs its chunk loop
    // (unlike compute_flows(), which uploads AFTER solve() and only for
    // callers who explicitly want full flows). This intentionally calls
    // upload_branch_admittances() (NOT the full set_branch_data(), which
    // would also allocate the O(n_ctg*n_branches) dense or_amps/ex_amps
    // buffers this path is meant to avoid).
    double t_admittance_upload_ms = 0.;
    double t_limits_setup_ms      = 0.;
    if (compute_limit_violations_) {
        if (!has_branch_data_)
            throw std::runtime_error(
                "ContingencyAnalysisSession: compute_limit_violations requires "
                "set_branch_data() to have been called first.");
        if (!has_limits_)
            throw std::runtime_error(
                "ContingencyAnalysisSession: compute_limit_violations requires "
                "set_limits() to have been called first.");

        solver_->upload_branch_admittances(
            h_branch_from_, h_branch_to_, h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_,
            _bus_array_with_spares(h_bus_vn_kv_, /*from_parent=*/true, 0.), sn_mva_);
        t_admittance_upload_ms = solver_->branch_data_upload_ms();

        // a spare bus is a busbar of its parent's substation: its voltage limits
        const double nan = std::numeric_limits<double>::quiet_NaN();
        solver_->set_violation_limits(
            _bus_array_with_spares(h_bus_vmin_kv_, /*from_parent=*/true, nan),
            _bus_array_with_spares(h_bus_vmax_kv_, /*from_parent=*/true, nan),
            h_branch_limit_a1_ka_, h_branch_limit_a2_ka_,
            violation_tol_, violation_rel_tol_, violation_capacity_, n_lines_);
        t_limits_setup_ms = solver_->violation_setup_ms();
    }

    // Post-solve physical checks (compute_physical_violations / compute_hvdc_p_
    // violations): plan / outputs on device before the chunk loop, base-case
    // report now. No generator contingencies here (nullptr mask).
    // redistribute_slack: a unit the pre-pass moved is checked at its new
    // set-point, and one it saturated takes no share (upstream's GenPCheck).
    RealMatRM rd_targets;
    std::vector<unsigned char> rd_ns_gen, rd_ns_sto;
    const bool rd_gp = redistribute_slack_ && phys_.compute_physical_violations &&
        gen_p_check_inputs(slack_rd_, slack_rows_, phys_.gen_p_plan.el_type, phys_.gen_p_plan.el_id,
                           RealMatRM(), rd_targets, rd_ns_gen, rd_ns_sto);
    // the generators each contingency trips leave their bus' summed capability
    const unsigned char* d_gen_off_ptr = nullptr;
    int n_gen_off = 0;
    if (phys_.compute_physical_violations && has_gen_trips_) {
        n_gen_off = gen_data_.n_gen;
        const size_t want = contingencies_.size() * static_cast<size_t>(n_gen_off);
        if (cold || dev_->d_gen_off.size() != want || dev_->n_gen != n_gen_off) {
            std::vector<unsigned char> h(want, 0);
            for (size_t r = 0; r < ctg_gens_.size(); ++r)
                for (int g : ctg_gens_[r]) h[r * static_cast<size_t>(n_gen_off) + static_cast<size_t>(g)] = 1;
            dev_->d_gen_off.assign(h.begin(), h.end());
            dev_->n_gen = n_gen_off;
        }
        d_gen_off_ptr = thrust::raw_pointer_cast(dev_->d_gen_off.data());
    }
    // the release plan's entries of the SVCs a contingency trips (a standby one's automaton) leave it
    int gr_skip_cols = 0;
    std::vector<unsigned char> gr_skip;
    if (phys_.compute_physical_violations)
        gr_skip = release_skip_with_svc_trips(
            phys_.gen_pv_release_plan, static_cast<int>(contingencies_.size()), std::vector<unsigned char>(), 0,
            [this](int r, int s) {
                if (static_cast<size_t>(r) >= ctg_more_.size()) return false;
                const std::vector<int>& ids = ctg_more_[static_cast<size_t>(r)].svcs;
                return std::find(ids.begin(), ids.end(), s) != ids.end(); },
            gr_skip_cols);
    physical_checks::RowTargets row_targets{(rd_gp && rd_targets.size() > 0) ? &rd_targets : nullptr, true,
                                            nullptr, false};
    row_targets.gen_pv_release_skip = &gr_skip;
    row_targets.gen_pv_release_skip_cols = gr_skip_cols;
    row_targets.gen_pv_release_skip_dirty = true;
    const physical_checks::SetupTimes t_phys = physical_checks::before_solve(
        phys_, *solver_, "ContingencyAnalysisSession", violation_tol_, sn_mva_,
        base_state_->timings.converged, d_gen_off_ptr, n_gen_off, row_targets);
    if (rd_gp)
        solver_->upload_gen_p_no_share(rd_ns_gen, slack_rd_.n_gen, rd_ns_sto, slack_rd_.n_sto);

    // Run all chunks; fills d_V_results and d_residuals on device (and, when
    // compute_limit_violations_ is set, the compact violation buffers too —
    // see check_limit_violations_kernel in _solve_chunk).
    // solve() carries preprocess/alloc/analysis timings from the ctor.
    // The outer loops: the model decides on every row (a contingency keeps the
    // grid's targets).
    solver_->outer_hook_ = outer_.get();
    if (outer_) {
        outer_->set_row_targets(static_cast<int>(contingencies_.size()), {}, {});
        // the buses each contingency masks, as its loops see them
        outer::RowOverrides ov;
        if (handle_disconnected_grid_) {
            ov.masked.resize(contingencies_.size());
            for (size_t r = 0; r < contingencies_.size(); ++r) ov.masked[r] = contingencies_[r].masked_buses;
        }
        // ... and the branches it opens (a transformer it opens holds nothing in its Ybus)
        ov.branch_off.resize(contingencies_.size());
        for (size_t r = 0; r < contingencies_.size(); ++r) ov.branch_off[r] = contingencies_[r].tripped_branches;
        // ... and the generators and loads it trips
        if (has_gen_trips_) {
            const int n_gen = gen_data_.n_gen;
            ov.gen_off.assign(contingencies_.size() * static_cast<size_t>(n_gen), 0);
            for (size_t r = 0; r < ctg_gens_.size(); ++r)
                for (int g : ctg_gens_[r]) ov.gen_off[r * static_cast<size_t>(n_gen) + static_cast<size_t>(g)] = 1;
        }
        if (any_controller_trip || !row_pv_release_.empty()) ov.pq_buses = row_pv_to_pq_;
        // ... and the generators it places on another bus (topology actions)
        if (!row_gen_bus_.empty()) {
            ov.gen_bus.assign(contingencies_.size(), {});
            for (size_t r = 0; r < row_gen_bus_.size() && r < contingencies_.size(); ++r)
                if (row_unsupported_[r].empty()) ov.gen_bus[r] = row_gen_bus_[r];
        }
        // ... and the controllers of the voltage-control groups it trips
        ov.vc_tripped.resize(contingencies_.size());
        for (size_t r = 0; r < contingencies_.size(); ++r) ov.vc_tripped[r] = contingencies_[r].vc_tripped;
        // ... and the lines in AC emulation it trips
        ov.hvdc_off.resize(contingencies_.size());
        for (size_t r = 0; r < ctg_more_.size(); ++r) ov.hvdc_off[r] = ctg_more_[r].hvdcs;
        // ... and the storage units it trips
        bool any_storage = false;
        for (const MoreTrips& t : ctg_more_) any_storage = any_storage || !t.storages.empty();
        if (any_storage) {
            const size_t n_sto = static_cast<size_t>(trip_data_.n_storage);
            ov.storage_off.assign(contingencies_.size() * n_sto, 0);
            for (size_t r = 0; r < ctg_more_.size(); ++r)
                for (int st : ctg_more_[r].storages) ov.storage_off[r * n_sto + static_cast<size_t>(st)] = 1;
        }
        bool any_load = false;
        for (const auto& l : ctg_loads_) if (!l.empty()) { any_load = true; break; }
        if (any_load) {
            const int n_load = trip_data_.n_load;
            ov.load_off.assign(contingencies_.size() * static_cast<size_t>(n_load), 0);
            for (size_t r = 0; r < ctg_loads_.size(); ++r)
                for (int l : ctg_loads_[r]) ov.load_off[r * static_cast<size_t>(n_load) + static_cast<size_t>(l)] = 1;
        }
        // ... and what it loses, for its DistributedSlack loop to share (redistribute_slack)
        ov.lost_p_mw = row_lost_p_mw_;
        outer_->set_row_overrides(std::move(ov));
    }
    const double t_sess_overrides_ms = ms_since(t_overrides0);
    // starting voltages by group (strategy_warm_start), else the base case
    if (has_v_groups_) {
        if (v_group_of_row_.size() != contingencies_.size())
            throw std::runtime_error("ContingencyAnalysisSession: set_streaming() gave "
                                     + std::to_string(v_group_of_row_.size()) + " groups for "
                                     + std::to_string(contingencies_.size()) + " rows");
        solver_->source_.set_v_groups(thrust::raw_pointer_cast(dev_->d_V_groups.data()), n_v_groups_,
                                      v_group_of_row_, thrust::raw_pointer_cast(base_state_->d_V_base.data()),
                                      static_cast<cudaStream_t>(base_state_->cs));
    } else {
        solver_->source_.clear_v_groups();
    }
    // streamed strategies: the feed the continuous schedule takes its rows from
    std::unique_ptr<RowFeed> feed;
    solver_->row_feed_ = nullptr;
    if (!stream_parent_.empty()) {
        if (!continuous)
            throw std::runtime_error("ContingencyAnalysisSession: streamed strategies need the continuous schedule");
        feed = _make_stream_feed();
        solver_->row_feed_ = feed.get();
    }
    timings_ = solver_->solve();
    solver_->row_feed_ = nullptr;
    const auto t_after0 = std::chrono::steady_clock::now();

    // Wire in base-case and one-time setup timings captured before/during
    // solver construction.
    timings_.t_base_case_ms  = t_base_case_ms_;
    // Fold base-case sub-phases into the contingency-solver aggregates so that
    // t_preprocess_ms and t_alloc_ms give a complete picture of all one-time
    // CPU work and H→D transfers, not just the contingency-solver's own share.
    timings_.t_preprocess_ms += base_state_->timings.t_build_J_ms;
    timings_.t_alloc_ms      += base_state_->timings.t_upload_ms;
    // compute_limit_violations' pre-solve() admittance upload would otherwise
    // be clobbered by the plain `timings_ = solver_->solve()` assignment above
    // — fold it in here, in the same neighborhood as the other one-time-setup
    // folding above.
    if (compute_limit_violations_) {
        timings_.t_branch_data_upload_ms += t_admittance_upload_ms;
        timings_.t_violation_setup_ms    += t_limits_setup_ms;
    }
    // The base case's own cuSPARSE/cuDSS context creation joins the batch
    // solver's (already in t_context_init_ms from solve()) in the shared
    // one-time-init bucket, instead of masquerading as base-case solve time:
    // on the first session in a process this is where CUDA-context creation
    // and the cuSPARSE/cuDSS dlopen+JIT land (tens of ms).
    timings_.t_context_init_ms += base_state_->timings.t_context_init_ms;
    // Non-overlapping remainder of t_base_case_ms (cuDSS analyze + NR
    // iterations, or the presolved_v validation step): the build_J/upload and
    // context-init shares are already folded above, so this avoids
    // double-counting them in
    // BatchTimings::t_gpu_compute_ms()/t_grand_total_ms().
    timings_.t_base_case_solve_only_ms =
        t_base_case_ms_ - base_state_->timings.t_build_J_ms
                         - base_state_->timings.t_upload_ms
                         - base_state_->timings.t_context_init_ms;
    // Informational only (measured before base_state_ existed) -- see
    // AcPfTimings::t_ground_truth_check_ms.
    timings_.t_ground_truth_check_ms = base_state_->timings.t_ground_truth_check_ms;

    // Disconnected contingencies are compacted out of the batch by the source
    // and never solved; the driver pre-fills their result slots with NaN.  Here
    // we only need to count them for the timing report.
    const int n_ctg = solver_->n_contingencies;
    int n_disconnected = 0;
    for (int i = 0; i < n_ctg; ++i)
        if (contingencies_[static_cast<size_t>(i)].disconnected)
            ++n_disconnected;

    timings_.n_disconnected = n_disconnected;
    has_violations_result_ = compute_limit_violations_;
    physical_checks::after_solve(phys_, timings_, t_phys);
    timings_.t_sess_ref_ms         = t_sess_ref_ms;
    timings_.t_sess_source_ms      = t_sess_source_ms;
    timings_.t_sess_prepass_ms     = t_sess_prepass_ms;
    timings_.t_sess_driver_ctor_ms = t_sess_driver_ctor_ms;
    timings_.t_sess_overrides_ms   = t_sess_overrides_ms;
    timings_.t_sess_after_ms       = ms_since(t_after0);
    timings_.t_session_run_wall_ms = ms_since(t_run0);
}

// =============================================================================
// compute_flows
// =============================================================================
void ContingencyAnalysisSession::compute_flows()
{
    _compute_flows_device();
    // D→H download of flow results — timed separately.
    const Eigen::Index n = static_cast<Eigen::Index>(solver_->n_contingencies) * solver_->n_branches_;
    auto t_copy_start = std::chrono::steady_clock::now();
    {
        // Through the driver's pinned staging buffer (download_into).
        h_or_amps_.resize(n);
        h_ex_amps_.resize(n);
        solver_->download_into(h_or_amps_.data(), static_cast<size_t>(n), solver_->d_or_amps_results);
        solver_->download_into(h_ex_amps_.data(), static_cast<size_t>(n), solver_->d_ex_amps_results);
    }
    timings_.t_copy_flows_to_host_ms = ms_since(t_copy_start);
}

namespace {
// One block per row: the highest |I| / limit over the branches and their two sides (amps in A,
// limits in kA; a NaN / non-positive limit ignored) and the first branch reaching it.
__global__ void max_loading_kernel(const cuda_real_type* __restrict__ or_amps, const cuda_real_type* __restrict__ ex_amps,
                                   const double* __restrict__ lim1_ka, const double* __restrict__ lim2_ka,
                                   int n_rows, int n_bra, double* __restrict__ rho, int* __restrict__ branch)
{
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const size_t off = static_cast<size_t>(r) * static_cast<size_t>(n_bra);
    double best = -1.;
    int best_b = -1;
    for (int b = threadIdx.x; b < n_bra; b += blockDim.x) {
        const double l1 = lim1_ka[b], l2 = lim2_ka[b];
        double v = -1.;
        if (l1 > 0.) v = fmax(v, or_amps[off + b] * 1e-3 / l1);
        if (l2 > 0.) v = fmax(v, ex_amps[off + b] * 1e-3 / l2);
        if (v > best) { best = v; best_b = b; }      // b increases: the first one kept on ties
    }
    __shared__ double s_v[256];
    __shared__ int s_b[256];
    s_v[threadIdx.x] = best;
    s_b[threadIdx.x] = best_b;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double ov = s_v[threadIdx.x + stride];
            const int ob = s_b[threadIdx.x + stride];
            if (ov > s_v[threadIdx.x] || (ov == s_v[threadIdx.x] && ob >= 0 && (s_b[threadIdx.x] < 0 || ob < s_b[threadIdx.x]))) {
                s_v[threadIdx.x] = ov;
                s_b[threadIdx.x] = ob;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rho[r] = s_b[0] >= 0 ? s_v[0] : nan("");
        branch[r] = s_b[0];
    }
}
}  // namespace

std::pair<RealVect, std::vector<int>> ContingencyAnalysisSession::compute_max_loading(
    Eigen::Ref<const RealVect> lim1_ka, Eigen::Ref<const RealVect> lim2_ka, const std::vector<long long>& skip)
{
    _compute_flows_device();
    const int n_rows = solver_->n_contingencies;
    const int n_bra = solver_->n_branches_;
    if (lim1_ka.size() != n_bra || lim2_ka.size() != n_bra)
        throw std::runtime_error("ContingencyAnalysisSession::compute_max_loading: one limit per branch and side");
    const cudaStream_t cs = solver_->cs;
    // the branches the caller computes itself (their values / ends change on that row): out of the max
    thrust::device_vector<long long> d_skip;
    if (!skip.empty()) {
        const long long n_z = static_cast<long long>(skip.size());
        for (long long k : skip)
            if (k < 0 || k >= static_cast<long long>(n_rows) * n_bra)
                throw std::runtime_error("ContingencyAnalysisSession::compute_max_loading: skip entry out of range");
        upload_h2d(d_skip, skip.data(), skip.size(), cs);
        zero_branch_flows_kernel<<<static_cast<unsigned>((n_z + SESSION_BS - 1) / SESSION_BS), SESSION_BS, 0, cs>>>(
            thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
            thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
            thrust::raw_pointer_cast(d_skip.data()), n_z);
    }
    thrust::device_vector<double> d_l1, d_l2, d_rho(static_cast<size_t>(std::max(n_rows, 1)));
    thrust::device_vector<int> d_br(static_cast<size_t>(std::max(n_rows, 1)));
    upload_h2d(d_l1, lim1_ka.data(), static_cast<size_t>(n_bra), cs);
    upload_h2d(d_l2, lim2_ka.data(), static_cast<size_t>(n_bra), cs);
    if (n_rows > 0)
        max_loading_kernel<<<static_cast<unsigned>(n_rows), 256, 0, cs>>>(
            thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
            thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
            thrust::raw_pointer_cast(d_l1.data()), thrust::raw_pointer_cast(d_l2.data()),
            n_rows, n_bra, thrust::raw_pointer_cast(d_rho.data()), thrust::raw_pointer_cast(d_br.data()));
    RealVect rho(n_rows);
    std::vector<int> branch(static_cast<size_t>(n_rows));
    if (n_rows > 0) {
        cudaMemcpyAsync(rho.data(), thrust::raw_pointer_cast(d_rho.data()), sizeof(double) * n_rows, cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(branch.data(), thrust::raw_pointer_cast(d_br.data()), sizeof(int) * n_rows, cudaMemcpyDeviceToHost, cs);
    }
    cudaStreamSynchronize(cs);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ContingencyAnalysisSession::compute_max_loading: ") + cudaGetErrorString(e));
    // (the host flows of a previous compute_flows() are untouched: the zeroing is on the device)
    return {rho, branch};
}

void ContingencyAnalysisSession::_compute_flows_device()
{
    if (!has_branch_data_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call set_branch_data() before compute_flows()");
    if (!solver_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() before compute_flows()");

    // Upload branch admittances to device and allocate result buffers.
    solver_->set_branch_data(
        h_branch_from_, h_branch_to_,
        h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_,
        _bus_array_with_spares(h_bus_vn_kv_, /*from_parent=*/true, 0.), sn_mva_);
    timings_.t_branch_data_upload_ms += solver_->branch_data_upload_ms();

    // Launch one kernel over ALL contingencies using d_V_results as input.
    const int n_ctg  = solver_->n_contingencies;
    const int n_bra  = solver_->n_branches_;
    const int n_bus  = base_state_->n_bus;
    // 64-bit: n_rows * n_branches passes INT32_MAX on large grids (70k buses, ~20k rows)
    const long long total = static_cast<long long>(n_ctg) * n_bra;
    const cudaStream_t cs = solver_->cs;

    CudaTimer flow_timer(cs);
    flow_timer.start();

    compute_branch_flows_kernel<<<static_cast<unsigned>((total + SESSION_BS - 1) / SESSION_BS), SESSION_BS, 0, cs>>>(
        thrust::raw_pointer_cast(solver_->d_V_results.data()),
        thrust::raw_pointer_cast(solver_->d_branch_from.data()),
        thrust::raw_pointer_cast(solver_->d_branch_to.data()),
        thrust::raw_pointer_cast(solver_->d_yff_eff.data()),
        thrust::raw_pointer_cast(solver_->d_yft_eff.data()),
        thrust::raw_pointer_cast(solver_->d_ytf_eff.data()),
        thrust::raw_pointer_cast(solver_->d_ytt_eff.data()),
        thrust::raw_pointer_cast(solver_->d_base_current_A.data()),
        thrust::raw_pointer_cast(solver_->d_base_current_ex_A.data()),
        thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
        thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
        n_bus, n_bra, 0, n_ctg, /*d_result_map=*/nullptr);
    // the transformers the outer loops left on other taps: at the row's final taps
    solver_->override_outer_trafo_flows();

    // Zero flows for tripped branches (they carry no current by definition).
    // Build flat indices: c * n_bra + l for each (c, l) where branch l is
    // disconnected in contingency c.
    std::vector<long long> h_zero;   // flat row * n_bra + l: 64-bit
    for (int c = 0; c < n_ctg; ++c)
        for (int l : disconnected_per_ctg_[static_cast<size_t>(c)])
            h_zero.push_back(static_cast<long long>(c) * n_bra + l);

    thrust::device_vector<long long> d_zero;   // kept alive until after sync
    if (!h_zero.empty()) {
        const long long n_z = static_cast<long long>(h_zero.size());
        upload_h2d(d_zero, h_zero.data(), static_cast<size_t>(n_z), cs);
        zero_branch_flows_kernel<<<static_cast<unsigned>((n_z + SESSION_BS - 1) / SESSION_BS), SESSION_BS, 0, cs>>>(
            thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
            thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
            thrust::raw_pointer_cast(d_zero.data()),
            n_z);
    }

    // stop_ms() records + synchronizes the stop event, so this also serves as
    // the sync point that lets d_zero be safely destroyed below.
    timings_.t_flow_computation += flow_timer.stop_ms();
    // d_zero destroyed here, after sync
}

// =============================================================================
// set_limits
// =============================================================================
void ContingencyAnalysisSession::set_limits(
    Eigen::Ref<const RealVect> bus_vmin_kv,
    Eigen::Ref<const RealVect> bus_vmax_kv,
    Eigen::Ref<const RealVect> branch_limit_a1_ka,
    Eigen::Ref<const RealVect> branch_limit_a2_ka,
    int n_lines)
{
    const int n_bus = base_state_->n_bus;
    // with spare buses, limits over the original buses are fine: each spare one gets its
    // parent's (a busbar of the same substation)
    const bool orig_size = !spare_parents_.empty() && bus_vmin_kv.size() == n_orig_bus_
                           && bus_vmax_kv.size() == n_orig_bus_;
    if ((bus_vmin_kv.size() != n_bus || bus_vmax_kv.size() != n_bus) && !orig_size)
        throw std::runtime_error(
            "ContingencyAnalysisSession::set_limits: bus_vmin_kv/bus_vmax_kv "
            "must have size n_bus");
    if (has_branch_data_) {
        const int n_bra = static_cast<int>(h_branch_from_.size());
        if (branch_limit_a1_ka.size() != n_bra || branch_limit_a2_ka.size() != n_bra)
            throw std::runtime_error(
                "ContingencyAnalysisSession::set_limits: branch_limit_a1_ka/"
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
// compute_limit_violations D→H result accessors
// =============================================================================
Eigen::VectorXi ContingencyAnalysisSession::get_violation_element_type() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_element_type;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_element_id() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_element_id;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_side() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_side;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_type() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_viol_type;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

RealVect ContingencyAnalysisSession::get_violation_value() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<cuda_real_type> h = solver_->d_viol_value;
    RealVect out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(h[i]);
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

RealVect ContingencyAnalysisSession::get_violation_limit() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<cuda_real_type> h = solver_->d_viol_limit;
    RealVect out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(h[i]);
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_count() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_truncated() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_truncated;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_count_low_voltage() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_low_voltage;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_count_high_voltage() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_high_voltage;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi ContingencyAnalysisSession::get_violation_count_current() const
{
    if (!has_violations_result_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() with compute_limit_violations=True first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    thrust::host_vector<int> h = solver_->d_violation_count_current;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = h[i];
    timings_.t_copy_violations_to_host_ms += ms_since(t_copy_start);
    return out;
}

// =============================================================================
// Metadata accessors
// =============================================================================
int ContingencyAnalysisSession::n_contingencies() const
{
    return solver_ ? solver_->n_contingencies
                   : static_cast<int>(contingencies_.size());
}

int ContingencyAnalysisSession::n_bus() const
{
    return base_state_ ? base_state_->n_bus : 0;
}

int ContingencyAnalysisSession::n_branches() const
{
    return solver_ ? solver_->n_branches_
                   : static_cast<int>(h_branch_from_.size());
}

// =============================================================================
// D→H result accessors
// =============================================================================
CplxVect ContingencyAnalysisSession::get_V_results() const
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    // 64-bit: n_rows * n_bus passes INT32_MAX on large grids (70k buses x 30k rows)
    const Eigen::Index n = static_cast<Eigen::Index>(solver_->n_contingencies) * base_state_->n_bus;
    CplxVect out(n);
    solver_->download_into(out.data(), static_cast<size_t>(n), solver_->d_V_results);   // pinned, pipelined
    timings_.t_copy_V_to_host_ms = ms_since(t_copy_start);
    return out;
}

CplxVect ContingencyAnalysisSession::get_V_rows(const std::vector<int>& rows) const
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    solver_->cs.synchronize();
    const int n_bus = base_state_->n_bus;
    CplxVect out(static_cast<Eigen::Index>(rows.size()) * n_bus);
    // the device's precision (float2 in an FP32 build), widened row by row
    std::vector<cudaComplexType> row_buf(static_cast<size_t>(n_bus));
    for (size_t k = 0; k < rows.size(); ++k) {
        const int r = rows[k];
        if (r < 0 || r >= solver_->n_contingencies)
            throw std::runtime_error("ContingencyAnalysisSession::get_V_rows: row " + std::to_string(r) + " out of range");
        cudaMemcpy(row_buf.data(),
                   thrust::raw_pointer_cast(solver_->d_V_results.data()) + static_cast<size_t>(r) * n_bus,
                   sizeof(cudaComplexType) * static_cast<size_t>(n_bus), cudaMemcpyDeviceToHost);
        for (int b = 0; b < n_bus; ++b)
            out[static_cast<Eigen::Index>(k) * n_bus + b] = CplxVect::Scalar(row_buf[static_cast<size_t>(b)].x,
                                                                              row_buf[static_cast<size_t>(b)].y);
    }
    return out;
}

RealVect ContingencyAnalysisSession::get_residuals() const
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    const int n = solver_->n_contingencies;
    RealVect out(n);
    solver_->download_into(out.data(), static_cast<size_t>(n), solver_->d_residuals);   // pinned, pipelined
    timings_.t_copy_residuals_to_host_ms = ms_since(t_copy_start);
    return out;
}

void ContingencyAnalysisSession::overwrite_rows(const std::vector<int>& dst_rows,
                                                const std::vector<int>& src_rows,
                                                std::uintptr_t d_V_src, std::uintptr_t d_res_src,
                                                int n_src_rows, int n_bus_src)
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: call run() first");
    if (dst_rows.size() != src_rows.size())
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: dst_rows and src_rows "
                                 "must have the same length");
    if (dst_rows.empty()) return;
    if (d_V_src == 0 || d_res_src == 0)
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: null source buffer");
    const int n_rows = solver_->n_contingencies;
    for (int r : dst_rows)
        if (r < 0 || r >= n_rows)
            throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: row " +
                                     std::to_string(r) + " outside [0, n_contingencies)");
    if (n_bus_src != base_state_->n_bus)
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: the source has " +
                                 std::to_string(n_bus_src) + " buses, this session " +
                                 std::to_string(base_state_->n_bus));
    for (int r : src_rows)
        if (r < 0 || r >= n_src_rows)
            throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: source row " +
                                     std::to_string(r) + " outside [0, " +
                                     std::to_string(n_src_rows) + ")");
    const cudaStream_t cs = static_cast<cudaStream_t>(solver_->cs);
    thrust::device_vector<int> d_dst, d_src;
    upload_h2d(d_dst, dst_rows.data(), dst_rows.size(), cs);
    upload_h2d(d_src, src_rows.data(), src_rows.size(), cs);
    const int n = static_cast<int>(dst_rows.size());
    launch_copy_rows(thrust::raw_pointer_cast(solver_->d_V_results.data()),
                     reinterpret_cast<const cudaComplexType*>(d_V_src),
                     thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_src.data()),
                     base_state_->n_bus, n, cs);
    launch_copy_rows(thrust::raw_pointer_cast(solver_->d_residuals.data()),
                     reinterpret_cast<const cuda_real_type*>(d_res_src),
                     thrust::raw_pointer_cast(d_dst.data()), thrust::raw_pointer_cast(d_src.data()),
                     1, n, cs);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ContingencyAnalysisSession::overwrite_rows: ") +
                                 cudaGetErrorString(e));
    solver_->cs.synchronize();
}

void ContingencyAnalysisSession::overwrite_row_outcomes(const std::vector<int>& rows,
                                                        const std::vector<int>& iterations,
                                                        const std::vector<int>& status)
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_row_outcomes: call run() first");
    if (rows.size() != iterations.size() || rows.size() != status.size())
        throw std::runtime_error("ContingencyAnalysisSession::overwrite_row_outcomes: rows, "
                                 "iterations and status must have the same length");
    const int n_rows = static_cast<int>(solver_->row_iters_.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const int r = rows[i];
        if (r < 0 || r >= n_rows)
            throw std::runtime_error("ContingencyAnalysisSession::overwrite_row_outcomes: row " +
                                     std::to_string(r) + " outside [0, n_contingencies)");
        solver_->row_iters_[static_cast<size_t>(r)]  = iterations[i];
        solver_->row_status_[static_cast<size_t>(r)] = status[i];
    }
}

Eigen::VectorXi ContingencyAnalysisSession::get_row_iterations() const
{
    if (!solver_) return Eigen::VectorXi();
    const std::vector<int>& v = solver_->row_iterations();
    return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<Eigen::Index>(v.size()));
}

Eigen::VectorXi ContingencyAnalysisSession::get_row_status() const
{
    if (!solver_) return Eigen::VectorXi();
    const std::vector<int>& v = solver_->row_status();
    return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<Eigen::Index>(v.size()));
}

// =============================================================================
// Streamed strategies (set_streaming): a RowFeed over the session's rows
// =============================================================================
namespace {
struct StreamFeed final : RowFeed {
    std::vector<int> a2o;                            // active -> original row
    std::vector<int> parent;                         // per original row (-1: a contingency row)
    std::vector<std::vector<int>> children;          // per original contingency row: its active strategy rows
    std::vector<int> group;                          // per original contingency row: its V group (-1: none)
    std::deque<int>  ready;
    long long        pending = 0;
    int              condition = 0;
    bool             warm = false;
    int              n_bus = 0;
    const int*             d_count = nullptr;        // violation counts, per original row
    const std::vector<int>* status = nullptr;        // the driver's row_status_
    const cudaComplexType* d_V_results = nullptr;
    const cudaComplexType* d_V_base = nullptr;
    cudaComplexType*       d_V_groups = nullptr;
    std::vector<char>*     released = nullptr;       // per original row

    int next() override {
        if (ready.empty()) return -1;
        const int a = ready.front();
        ready.pop_front();
        return a;
    }
    bool more() const override { return !ready.empty() || pending > 0; }

    // contingency `o` is over (count: its violations, -1 never simulated; converged): its
    // strategies released or dropped, their V group written
    void settle(int o, int count, bool converged, cudaStream_t cs) {
        std::vector<int>& kids = children[static_cast<size_t>(o)];
        if (kids.empty()) return;
        pending -= static_cast<long long>(kids.size());
        const bool go = condition == 0 || count > 0;
        if (go) {
            const int g = group[static_cast<size_t>(o)];
            if (warm && g >= 0) {
                const cudaComplexType* src = converged ? d_V_results + static_cast<size_t>(o) * n_bus : d_V_base;
                cudaMemcpyAsync(d_V_groups + static_cast<size_t>(g) * n_bus, src,
                                sizeof(cudaComplexType) * static_cast<size_t>(n_bus), cudaMemcpyDeviceToDevice, cs);
            }
            for (int a : kids) {
                ready.push_back(a);
                (*released)[static_cast<size_t>(a2o[static_cast<size_t>(a)])] = 1;
            }
        }
        kids.clear();
    }

    void finished(const std::vector<int>& rows, cudaStream_t cs) override {
        std::vector<int> ctg;
        for (int a : rows) {
            const int o = a2o[static_cast<size_t>(a)];
            if (parent[static_cast<size_t>(o)] < 0 && !children[static_cast<size_t>(o)].empty()) ctg.push_back(o);
        }
        if (ctg.empty()) return;
        std::vector<int> count(ctg.size(), -1);
        if (d_count != nullptr)
            for (size_t i = 0; i < ctg.size(); ++i)
                cudaMemcpyAsync(&count[i], d_count + ctg[i], sizeof(int), cudaMemcpyDeviceToHost, cs);
        cudaStreamSynchronize(cs);
        for (size_t i = 0; i < ctg.size(); ++i)
            settle(ctg[i], count[i], (*status)[static_cast<size_t>(ctg[i])] == static_cast<int>(RowStatus::Converged), cs);
    }
};
}  // namespace

std::unique_ptr<RowFeed> ContingencyAnalysisSession::_make_stream_feed()
{
    auto f = std::make_unique<StreamFeed>();
    const size_t n_rows = contingencies_.size();
    f->a2o = solver_->source_.active_to_orig();
    if (f->a2o.empty() && solver_->n_active_ == static_cast<int>(n_rows)) {
        f->a2o.resize(n_rows);
        for (size_t i = 0; i < n_rows; ++i) f->a2o[i] = static_cast<int>(i);
    }
    f->parent = stream_parent_;
    f->children.assign(n_rows, {});
    f->group = stream_group_of_ctg_;
    f->condition = stream_condition_;
    f->warm = stream_warm_;
    f->n_bus = base_state_->n_bus;
    if (stream_condition_ != 0) {
        if (solver_->d_violation_count.size() != n_rows)
            throw std::runtime_error("ContingencyAnalysisSession: the ANY_VIOLATION condition needs the limit "
                                     "checks (compute_limit_violations)");
        f->d_count = thrust::raw_pointer_cast(solver_->d_violation_count.data());
    }
    f->status = &solver_->row_status_;
    f->d_V_results = thrust::raw_pointer_cast(solver_->d_V_results.data());
    f->d_V_base = thrust::raw_pointer_cast(base_state_->d_V_base.data());
    f->d_V_groups = stream_warm_ ? thrust::raw_pointer_cast(dev_->d_V_groups.data()) : nullptr;
    stream_released_.assign(n_rows, 0);
    f->released = &stream_released_;
    std::vector<char> active(n_rows, 0);
    for (size_t a = 0; a < f->a2o.size(); ++a) {
        const int o = f->a2o[a];
        active[static_cast<size_t>(o)] = 1;
        const int p = f->parent[static_cast<size_t>(o)];
        if (p < 0) {
            f->ready.push_back(static_cast<int>(a));
            stream_released_[static_cast<size_t>(o)] = 1;
        } else {
            f->children[static_cast<size_t>(p)].push_back(static_cast<int>(a));
            ++f->pending;
        }
    }
    // a contingency that never runs (islanded, unsupported): settled now, never simulated
    const cudaStream_t cs = static_cast<cudaStream_t>(solver_->cs);
    for (size_t o = 0; o < n_rows; ++o)
        if (f->parent[o] < 0 && !active[o]) f->settle(static_cast<int>(o), -1, false, cs);
    return f;
}

void ContingencyAnalysisSession::set_streaming(const std::vector<int>& parent_of_row, int condition, bool warm)
{
    static const char* who = "ContingencyAnalysisSession::set_streaming";
    if (!has_contingencies_) throw std::runtime_error(std::string(who) + ": call build_contingencies() first");
    const size_t n = contingencies_.size();
    if (parent_of_row.size() != n)
        throw std::runtime_error(std::string(who) + ": one entry per row (" + std::to_string(n) + ")");
    if (condition != 0 && condition != 1)
        throw std::runtime_error(std::string(who) + ": condition 0 (TRUE) or 1 (ANY_VIOLATION)");
    // each contingency row with strategies gets a V group (written when it finishes)
    std::vector<int> group(n, -1), group_of_row(n, -1);
    int n_groups = 0;
    for (size_t r = 0; r < n; ++r) {
        const int p = parent_of_row[r];
        if (p < 0) continue;
        if (p >= static_cast<int>(n) || parent_of_row[static_cast<size_t>(p)] >= 0)
            throw std::runtime_error(std::string(who) + ": row " + std::to_string(r) + "'s contingency row "
                                     + std::to_string(p) + " is not a contingency row");
        if (group[static_cast<size_t>(p)] < 0) group[static_cast<size_t>(p)] = n_groups++;
        group_of_row[r] = group[static_cast<size_t>(p)];
    }
    stream_parent_ = parent_of_row;
    stream_condition_ = condition;
    stream_warm_ = warm;
    stream_group_of_ctg_ = group;
    if (warm && n_groups > 0) {
        dev_->d_V_groups.resize(static_cast<size_t>(n_groups) * base_state_->n_bus);
        n_v_groups_ = n_groups;
        has_v_groups_ = true;
        v_group_of_row_ = group_of_row;
    } else {
        has_v_groups_ = false;
    }
    // the outer loops: each strategy resumes from its contingency's loops, kept as it finishes
    if (outer_) {
        outer_->keep_row_points(warm);
        outer_->set_row_resume(warm ? parent_of_row : std::vector<int>());
    }
    ++contingencies_generation_;
}

void ContingencyAnalysisSession::clear_streaming()
{
    stream_parent_.clear();
    stream_group_of_ctg_.clear();
    has_v_groups_ = false;
    n_v_groups_ = 0;
    v_group_of_row_.clear();
    dev_->d_V_groups.clear();
    dev_->d_V_groups.shrink_to_fit();
    if (outer_) {
        outer_->keep_row_points(false);
        outer_->set_row_resume({});
    }
}

void ContingencyAnalysisSession::add_row_triplets(
    const std::vector<std::vector<std::tuple<int, int, double, double>>>& rows)
{
    static const char* who = "ContingencyAnalysisSession::add_row_triplets";
    if (!has_contingencies_)
        throw std::runtime_error(std::string(who) + ": call build_contingencies() first");
    if (rows.size() != contingencies_.size())
        throw std::runtime_error(std::string(who) + ": " + std::to_string(rows.size()) + " lists for "
                                 + std::to_string(contingencies_.size()) + " rows");
    const int n_bus = base_state_->n_bus;
    for (const auto& row : rows)
        for (const auto& t : row)
            if (std::get<0>(t) < 0 || std::get<0>(t) >= n_bus || std::get<1>(t) < 0 || std::get<1>(t) >= n_bus)
                throw std::runtime_error(std::string(who) + ": entry (" + std::to_string(std::get<0>(t)) + ", "
                                         + std::to_string(std::get<1>(t)) + ") outside the " + std::to_string(n_bus)
                                         + " solver buses");
    for (size_t c = 0; c < rows.size(); ++c)
        for (const auto& t : rows[c])
            contingencies_[c].triplets.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t)});
    ++contingencies_generation_;   // the live driver's source is stale now
}

// =============================================================================
// set_spare_buses / set_row_empty_buses / add_row_sbus_deltas -- topology actions,
// bus moves on a union pattern (docs/dev_notes/topology_actions_plan.md, step B.2)
// =============================================================================
void ContingencyAnalysisSession::set_spare_buses(const std::vector<int>& parents,
                                                 const std::vector<std::pair<int, int>>& entries,
                                                 const std::vector<int>& spare_grid_bus)
{
    static const char* who = "ContingencyAnalysisSession::set_spare_buses";
    if (!base_ledger_)
        throw std::runtime_error(std::string(who) + ": needs a session built from a lightsim2grid grid (a ledger)");
    if (!union_active_) {   // the original base, kept to restore it
        n_orig_bus_      = static_cast<int>(Ybus_cm_.rows());
        Ybus_cm_orig_    = Ybus_cm_;
        Ybus_rm_orig_    = Ybus_rm_;
        Vinit_orig_      = Vinit_;
        Sbus_base_orig_  = h_Sbus_base_;
        pq_orig_         = pq_;
    }
    const int n = n_orig_bus_;
    const int m = static_cast<int>(parents.size());
    for (int b : parents)
        if (b < 0 || b >= n)
            throw std::runtime_error(std::string(who) + ": parent bus " + std::to_string(b) + " outside [0, "
                                     + std::to_string(n) + ")");
    for (const auto& [i, j] : entries)
        if (i < 0 || j < 0 || i >= n + m || j >= n + m)
            throw std::runtime_error(std::string(who) + ": entry (" + std::to_string(i) + ", " + std::to_string(j)
                                     + ") outside the " + std::to_string(n + m) + " buses");
    if (outer_ && m > 0 && spare_grid_bus.size() != parents.size())
        throw std::runtime_error(std::string(who) + ": with the outer loops, each spare bus needs its grid bus "
                                 "(spare_grid_bus)");
    const bool active = m > 0 || !entries.empty();   // a merge needs entries, no spare bus
    // the same union as now: keep the pattern, the analysis and the driver (no rebuild)
    if (active == union_active_ && parents == spare_parents_ && entries == spare_entries_) return;
    if (!active) {   // back to the original base: today's path
        Ybus_cm_ = Ybus_cm_orig_;
        Ybus_rm_ = Ybus_rm_orig_;
        Vinit_ = Vinit_orig_;
        h_Sbus_base_ = Sbus_base_orig_;
        pq_ = pq_orig_;
    } else {
        // the union: every base entry (its value), every requested entry and every spare
        // diagonal as an explicit zero (setFromTriplets sums, never prunes)
        std::vector<Eigen::Triplet<eigen_cplx_type>> t;
        t.reserve(static_cast<size_t>(Ybus_cm_orig_.nonZeros()) + entries.size() + static_cast<size_t>(m));
        for (int c = 0; c < Ybus_cm_orig_.outerSize(); ++c)
            for (Eigen::SparseMatrix<eigen_cplx_type>::InnerIterator it(Ybus_cm_orig_, c); it; ++it)
                t.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
        for (const auto& [i, j] : entries) t.emplace_back(i, j, eigen_cplx_type(0.));
        for (int k = 0; k < m; ++k) t.emplace_back(n + k, n + k, eigen_cplx_type(0.));
        Eigen::SparseMatrix<eigen_cplx_type> Y(n + m, n + m);
        Y.setFromTriplets(t.begin(), t.end());
        Y.makeCompressed();
        Ybus_cm_ = Y;
        Ybus_rm_ = Y;
        Ybus_rm_.makeCompressed();
        // a spare bus starts from its parent's voltage, injects nothing, is PQ
        Vinit_.resize(n + m);
        Vinit_.head(n) = Vinit_orig_;
        h_Sbus_base_.resize(n + m);
        h_Sbus_base_.head(n) = Sbus_base_orig_;
        for (int k = 0; k < m; ++k) {
            Vinit_(n + k) = Vinit_orig_(parents[static_cast<size_t>(k)]);
            h_Sbus_base_(n + k) = eigen_cplx_type(0., 0.);
        }
        pq_.resize(pq_orig_.size() + m);
        pq_.head(pq_orig_.size()) = pq_orig_;
        for (int k = 0; k < m; ++k) pq_(pq_orig_.size() + k) = n + k;
    }
    spare_parents_ = parents;
    spare_entries_ = entries;
    spare_grid_bus_ = active ? spare_grid_bus : std::vector<int>();
    union_active_  = active;
    ++contingencies_generation_;          // the rows resolve on the new Ybus at the next run()
    // the base state (and no driver) on the new buses; the outer loops' ledger carries its
    // own switchable buses (spares included, below)
    _build_base_state(outer_ ? std::vector<int>() : reserved_buses_);
}

void ContingencyAnalysisSession::set_row_empty_buses(const std::vector<std::vector<int>>& rows)
{
    static const char* who = "ContingencyAnalysisSession::set_row_empty_buses";
    if (!has_contingencies_) throw std::runtime_error(std::string(who) + ": call build_contingencies() first");
    if (rows.size() != contingencies_.size())
        throw std::runtime_error(std::string(who) + ": " + std::to_string(rows.size()) + " lists for "
                                 + std::to_string(contingencies_.size()) + " rows");
    // base buses only: a spare bus is masked by default in every row that does not use it
    const int n_base = base_state_->n_bus - static_cast<int>(spare_parents_.size());
    for (size_t c = 0; c < rows.size(); ++c) {
        std::vector<int> buses = rows[c];
        for (int b : buses)
            if (b < 0 || b >= n_base)
                throw std::runtime_error(std::string(who) + ": bus " + std::to_string(b) + " is not one of the "
                                         + std::to_string(n_base) + " base buses (spare buses are masked by "
                                         "default where a row does not use them)");
        std::sort(buses.begin(), buses.end());
        buses.erase(std::unique(buses.begin(), buses.end()), buses.end());
        contingencies_[c].empty_buses = std::move(buses);
    }
    ++contingencies_generation_;
}

void ContingencyAnalysisSession::set_row_pv_moves(const std::vector<std::vector<int>>& release,
                                                  const std::vector<std::vector<std::pair<int, double>>>& pv,
                                                  const std::vector<std::vector<std::pair<int, int>>>& gen_bus)
{
    static const char* who = "ContingencyAnalysisSession::set_row_pv_moves";
    if (!has_contingencies_) throw std::runtime_error(std::string(who) + ": call build_contingencies() first");
    if (release.size() != contingencies_.size() || pv.size() != contingencies_.size())
        throw std::runtime_error(std::string(who) + ": one release list and one pv list per row ("
                                 + std::to_string(contingencies_.size()) + " rows)");
    const int n_bus = base_state_->n_bus;
    const int n_base = n_bus - static_cast<int>(spare_parents_.size());
    for (size_t r = 0; r < release.size(); ++r) {
        for (int b : release[r])
            if (b < 0 || b >= n_base)
                throw std::runtime_error(std::string(who) + ": released bus " + std::to_string(b)
                                         + " is not one of the " + std::to_string(n_base) + " base buses");
        for (const auto& bv : pv[r])
            if (bv.first < 0 || bv.first >= n_bus || !(bv.second > 0.))
                throw std::runtime_error(std::string(who) + ": PV bus " + std::to_string(bv.first)
                                         + " out of range or a non-positive set-point");
    }
    row_pv_release_ = release;
    for (auto& rel : row_pv_release_) {
        std::sort(rel.begin(), rel.end());
        rel.erase(std::unique(rel.begin(), rel.end()), rel.end());
    }
    row_pv_set_ = pv;
    if (!gen_bus.empty() && gen_bus.size() != contingencies_.size())
        throw std::runtime_error(std::string(who) + ": one gen_bus list per row");
    for (const auto& row : gen_bus)
        for (const auto& [g, b] : row)
            if (g < 0 || b < 0 || b >= n_bus)
                throw std::runtime_error(std::string(who) + ": generator " + std::to_string(g) + " placed on bus "
                                         + std::to_string(b) + ", outside the solver buses");
    row_gen_bus_ = gen_bus;
    ++contingencies_generation_;
}

void ContingencyAnalysisSession::add_row_sbus_deltas(
    const std::vector<std::vector<std::pair<int, std::complex<double>>>>& rows)
{
    static const char* who = "ContingencyAnalysisSession::add_row_sbus_deltas";
    if (!has_contingencies_) throw std::runtime_error(std::string(who) + ": call build_contingencies() first");
    if (rows.size() != contingencies_.size())
        throw std::runtime_error(std::string(who) + ": " + std::to_string(rows.size()) + " lists for "
                                 + std::to_string(contingencies_.size()) + " rows");
    const int n_bus = base_state_->n_bus;
    if (ctg_sbus_ds_.size() != contingencies_.size()) ctg_sbus_ds_.resize(contingencies_.size());
    for (size_t c = 0; c < rows.size(); ++c) {
        if (rows[c].empty()) continue;
        // one entry per bus, sorted (the device adds them without atomics): merge with the trips'
        std::map<int, std::complex<double>> merged(ctg_sbus_ds_[c].begin(), ctg_sbus_ds_[c].end());
        for (const auto& [b, ds] : rows[c]) {
            if (b < 0 || b >= n_bus)
                throw std::runtime_error(std::string(who) + ": bus " + std::to_string(b) + " outside the "
                                         + std::to_string(n_bus) + " solver buses");
            merged[b] += ds;
            contingencies_[c].injection_buses.push_back(b);
        }
        ctg_sbus_ds_[c].assign(merged.begin(), merged.end());
        auto& inj = contingencies_[c].injection_buses;
        std::sort(inj.begin(), inj.end());
        inj.erase(std::unique(inj.begin(), inj.end()), inj.end());
    }
    ++contingencies_generation_;
}

RealVect ContingencyAnalysisSession::_bus_array_with_spares(const RealVect& a, bool from_parent, double fill) const
{
    const int n_bus = base_state_ ? base_state_->n_bus : static_cast<int>(a.size());
    if (spare_parents_.empty() || a.size() >= n_bus) return a;
    RealVect out(n_bus);
    out.head(a.size()) = a;
    for (int b = static_cast<int>(a.size()); b < n_bus; ++b) {
        const int k = b - n_orig_bus_;
        out(b) = (from_parent && k >= 0 && k < static_cast<int>(spare_parents_.size()))
                     ? a(spare_parents_[static_cast<size_t>(k)]) : fill;
    }
    return out;
}

int ContingencyAnalysisSession::result_rows() const
{
    return solver_ ? solver_->n_contingencies : 0;
}

std::uintptr_t ContingencyAnalysisSession::v_results_ptr() const
{
    return solver_ ? reinterpret_cast<std::uintptr_t>(thrust::raw_pointer_cast(solver_->d_V_results.data()))
                   : 0;
}

RealVect ContingencyAnalysisSession::get_or_amps() const
{
    if (h_or_amps_.size() == 0)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() and compute_flows() first");
    return h_or_amps_;
}

RealVect ContingencyAnalysisSession::get_ex_amps() const
{
    if (h_ex_amps_.size() == 0)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call run() and compute_flows() first");
    return h_ex_amps_;
}

// =============================================================================
// Post-solve physical checks (compute_physical_violations / compute_hvdc_p_
// violations) -- shared glue in contingency/physical_checks_impl.cuh
// =============================================================================
void ContingencyAnalysisSession::set_bus_q_capability(const BusQPlanData& plan)
{
    phys_.set_bus_q_plan(plan, base_state_->n_bus);
}

BusQViolationsResult ContingencyAnalysisSession::get_bus_q_violations() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_bus_q(phys_, *solver_, /*n_case=*/false, "ContingencyAnalysisSession",
                                        timings_.t_copy_violations_to_host_ms);
}

BusQViolationsResult ContingencyAnalysisSession::get_bus_q_violations_n() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_bus_q(phys_, *solver_, /*n_case=*/true, "ContingencyAnalysisSession",
                                        timings_.t_copy_violations_to_host_ms);
}

HvdcPViolationsResult ContingencyAnalysisSession::get_hvdc_p_violations() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_hvdc_p(phys_, *solver_, /*n_case=*/false, "ContingencyAnalysisSession",
                                         timings_.t_copy_violations_to_host_ms);
}

HvdcPViolationsResult ContingencyAnalysisSession::get_hvdc_p_violations_n() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_hvdc_p(phys_, *solver_, /*n_case=*/true, "ContingencyAnalysisSession",
                                         timings_.t_copy_violations_to_host_ms);
}

void ContingencyAnalysisSession::set_slack_redistribution_data(const SlackRedistributionData& data)
{
    data.validate(base_state_->n_bus);
    slack_rd_     = data;
    has_slack_rd_ = true;
    ++contingencies_generation_;   // a live pre-pass was built on the old data
}

SlackRedistributionReport ContingencyAnalysisSession::get_slack_redistribution_report() const
{
    return make_slack_redistribution_report(slack_rows_);
}

void ContingencyAnalysisSession::_check_redistribute_slack() const
{
    if (!has_slack_rd_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: redistribute_slack needs the slack participants -- call "
            "set_slack_redistribution_data() first (the ContingencyAnalysisGPU facade does it "
            "from a lightsim2grid grid).");
    // with the outer loops, the DistributedSlack loop's participants share it, not the Newton's
    if (!outer_ && base_state_->n_slack <= 0)
        throw std::runtime_error(
            "ContingencyAnalysisSession: redistribute_slack needs the distributed slack "
            "(use_distributed_slack=True): without it no participant takes a share.");
    if (strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
        throw std::runtime_error(
            "ContingencyAnalysisSession: redistribute_slack is incompatible with the "
            "'direct_base_case_factors' strategy (it reuses the base-case factors, which cannot "
            "take a saturated unit out of a contingency's slack).");
    // the pre-pass reads a row's island off the per-unit base Sbus with
    // slack_rd_.sn_mva: another base would mis-scale every lost MW
    if (std::abs(slack_rd_.sn_mva - sn_mva_) > 1e-9 * std::max(1.0, std::abs(sn_mva_)))
        throw std::runtime_error(
            "ContingencyAnalysisSession: the slack participants were given with another sn_mva "
            "than the branch data's (set_branch_data).");
}

void ContingencyAnalysisSession::set_gen_p_capability(const GenPPlanData& plan)
{
    phys_.set_gen_p_plan(plan, base_state_->n_bus);
}

GenPViolationsResult ContingencyAnalysisSession::get_gen_p_violations() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_gen_p(phys_, *solver_, /*n_case=*/false, "ContingencyAnalysisSession",
                                        timings_.t_copy_violations_to_host_ms);
}

GenPViolationsResult ContingencyAnalysisSession::get_gen_p_violations_n() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_gen_p(phys_, *solver_, /*n_case=*/true, "ContingencyAnalysisSession",
                                        timings_.t_copy_violations_to_host_ms);
}

void ContingencyAnalysisSession::set_gen_pv_release_capability(const GenPvReleasePlanData& plan)
{
    phys_.set_gen_pv_release_plan(plan, base_state_->n_bus);
}

GenPvReleaseViolationsResult ContingencyAnalysisSession::get_gen_pv_release_violations() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_gen_pv_release(phys_, *solver_, /*n_case=*/false, "ContingencyAnalysisSession",
                                                 timings_.t_copy_violations_to_host_ms);
}

GenPvReleaseViolationsResult ContingencyAnalysisSession::get_gen_pv_release_violations_n() const
{
    if (!solver_) throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    return physical_checks::fetch_gen_pv_release(phys_, *solver_, /*n_case=*/true, "ContingencyAnalysisSession",
                                                 timings_.t_copy_violations_to_host_ms);
}
