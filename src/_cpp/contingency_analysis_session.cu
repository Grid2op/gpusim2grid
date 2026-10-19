// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_session.cu
// =============================================================================

#include "contingency_analysis_session.hpp"
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

#include <algorithm>   // std::max
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
        if ((ref_bus_ >= 0 && ref_bus_ != grid_ref_bus_) || !switchable_buses.empty()) {
            moved = *base_ledger_;
            if (ref_bus_ >= 0 && ref_bus_ != grid_ref_bus_) move_reference(moved, ref_bus_, Ybus_rm_);
            // a generator trip releasing a bus it held alone: a Vm column + Q row for it,
            // pinned on the rows that keep it PV (see run())
            if (!switchable_buses.empty()) add_switchable_vm_buses(moved, switchable_buses, Ybus_rm_);
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
}

int ContingencyAnalysisSession::_wanted_reference_bus()
{
    if (grid_ref_bus_ < 0 || !auto_reference_slack_ || !handle_disconnected_grid_ || !has_contingencies_)
        return grid_ref_bus_;
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

void ContingencyAnalysisSession::build_contingencies_full(
    const std::vector<std::vector<int>>& branch_ids_per_ctg,
    const std::vector<std::vector<int>>& shunt_ids_per_ctg,
    const std::vector<std::vector<int>>& load_ids_per_ctg,
    const std::vector<std::vector<int>>& gen_ids_per_ctg)
{
    static const char* who = "ContingencyAnalysisSession::build_contingencies_full";
    if (!has_branch_data_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call set_branch_data() before build_contingencies()");
    const int n_ctg = static_cast<int>(branch_ids_per_ctg.size());
    if (static_cast<int>(shunt_ids_per_ctg.size()) != n_ctg || static_cast<int>(load_ids_per_ctg.size()) != n_ctg ||
        static_cast<int>(gen_ids_per_ctg.size()) != n_ctg)
        throw std::runtime_error(std::string(who) + ": the branch, shunt, load and generator lists must "
                                 "have one entry per contingency each");
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

    contingencies_.clear();
    contingencies_.reserve(n_ctg);
    disconnected_per_ctg_.clear();
    disconnected_per_ctg_.reserve(n_ctg);
    ctg_shunts_ = shunt_ids_per_ctg;
    ctg_loads_  = load_ids_per_ctg;
    ctg_gens_   = gen_ids_per_ctg;
    ctg_sbus_ds_.assign(static_cast<size_t>(n_ctg), {});
    has_gen_trips_ = any_gen;

    for (int c = 0; c < n_ctg; ++c) {
        Contingency ctg = build_contingency_from_branch_ids(
            branch_ids_per_ctg[c],
            h_branch_from_, h_branch_to_,
            h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_);
        // a shunt: its diagonal coefficient leaves the row's Ybus (merged with a branch
        // end on the same diagonal by build_flat_patches)
        for (const auto& t : element_trips::shunt_triplets(trip_data_, shunt_ids_per_ctg[c]))
            ctg.triplets.push_back({t.bus, t.bus, t.re, t.im});
        // a load / generator: its injection leaves the row's Sbus
        if (!load_ids_per_ctg[c].empty() || !gen_ids_per_ctg[c].empty())
            ctg_sbus_ds_[static_cast<size_t>(c)] = element_trips::sbus_delta(
                trip_data_, load_ids_per_ctg[c], gen_ids_per_ctg[c],
                gen_data_.bus, gen_data_.vreg_on, gen_data_.target_q_mvar);
        contingencies_.push_back(std::move(ctg));
        disconnected_per_ctg_.push_back(branch_ids_per_ctg[c]);
    }
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
        // a regulating shunt's sections are the ShuntVoltageControl loop's, which does not
        // know of a row taking it out (lightsim2grid refuses it the same way)
        for (size_t c = 0; c < ctg_shunts_.size(); ++c)
            for (int sh : ctg_shunts_[c])
                if (trip_data_.shunt_regulating[static_cast<size_t>(sh)] && trip_data_.shunt_bus[static_cast<size_t>(sh)] >= 0)
                    throw std::runtime_error(
                        "ContingencyAnalysisSession: contingency " + std::to_string(c) + " trips shunt " +
                        std::to_string(sh) + ", which regulates a voltage, with the outer loops: the "
                        "ShuntVoltageControl loop owns its sections and does not support a row taking it out yet");
    }

    // Generator trips: the buses some contingency releases (they need a reserved
    // Vm column + Q equation), each contingency's releases and the slack
    // participants it takes out (lightsim2grid's _maybe_prepare_gen_contingency)
    const int n_ctg_all = static_cast<int>(contingencies_.size());
    const auto ctg_gen_off = [this](int r, int g) {
        const std::vector<int>& gens = ctg_gens_[static_cast<size_t>(r)];
        return std::find(gens.begin(), gens.end(), g) != gens.end();
    };
    std::vector<int>              required;
    std::vector<std::vector<int>> row_slack_off;
    row_pv_to_pq_.assign(static_cast<size_t>(n_ctg_all), std::vector<int>());
    row_slack_off.assign(static_cast<size_t>(n_ctg_all), std::vector<int>());
    if (has_gen_trips_) {
        gen_contingency::check_rows(gen_data_, n_ctg_all, ctg_gen_off, "ContingencyAnalysisSession");
        gen_contingency::prepare_rows(gen_data_, n_ctg_all, ctg_gen_off, base_ledger_.get(),
                                      required, row_pv_to_pq_, row_slack_off);
        if (outer_ && !required.empty())
            throw std::runtime_error(
                "ContingencyAnalysisSession: a generator trip releases bus " + std::to_string(required.front()) +
                ", which the outer loops' ledger has not reserved (ReactiveLimits reserves every bus "
                "its own machines hold)");
        if (strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
            throw std::runtime_error(
                "ContingencyAnalysisSession: tripping generators is incompatible with the "
                "'direct_base_case_factors' strategy (it reuses the base-case factors, which cannot "
                "release a bus' voltage pinning per row). Use 'direct_refactor_every' (default), "
                "'direct_iter0_only', or 'direct_refactor_every_n'.");
    }

    // Automatic reference slack: the participant the fewest contingencies
    // strand; a change moves it in the ledger and rebuilds the base state --
    // as does a change of the buses the generator trips release (the outer loops'
    // ledger carries its own)
    const int want_ref = _wanted_reference_bus();
    if (want_ref != ref_bus_ || (!outer_ && required != reserved_buses_)) {
        ref_bus_ = want_ref;
        _build_base_state(outer_ ? std::vector<int>() : required);
    }

    // Hot run: same settings, same contingency list as the live driver --
    // keep it (allocations, block-diagonal structure, cuDSS ANALYSIS, the
    // policy's factorization state) and its source (connectivity, masks,
    // patches already on the device); only solve() runs again. Otherwise
    // build both, as the first run() does. The continuous schedule always
    // rebuilds (its reuse is a later step, docs/dev_notes/continuous_batching.md
    // A.6).
    const BatchDriverConfig cfg = _current_config();
    const bool cold = !solver_ || cfg != driver_cfg_ || continuous;

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

        // Host preprocessing (resolve_indices + connectivity/masking +
        // build_flat_patches) captured by the BatchSource; mutates
        // contingencies_ in-place so that disconnected flags are observable
        // below for the disconnected count. The source rebalances the chunk
        // size over the ACTIVE (simulated) count; we read it back so the
        // driver chunks consistently with the source.
        ContingencyBatch source(
            contingencies_,
            Ybus_rm_.outerIndexPtr(),
            Ybus_rm_.innerIndexPtr(),
            Ybus_rm_,
            batch_size_,
            (handle_disconnected_grid_ || !held_pins.empty() || any_gen_pin) ? &mask_cfg_ : nullptr,
            handle_disconnected_grid_);
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
                const auto g_p   = [this](int g) { return static_cast<double>(slack_rd_.gen_target_p_mw(g)); };
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
                                      : 0.);
                if (outer_) {
                    if (std::abs(lost) > slack_redistribution::DEFAULT_EPS_MW) row_lost_p_mw_[static_cast<size_t>(r)] = lost;
                } else {
                    slack_rows_[static_cast<size_t>(r)] = slack_redistribution::redistribute_row(slack_rd_, lost, bus_masked, g_off, g_p);
                }
                for (int b : ctg.masked_buses) is_masked[static_cast<size_t>(b)] = 0;
            }
            if (outer_) slack_rows_.clear();
        }
        if (redistribute_slack_ && !outer_) {
            w_rows = slack_redistribution::build_row_weights(
                slack_rd_, slack_rows_, n_ctg_all, ctg_gen_off, n_bus,
                base_state_->h_slack_bus, base_state_->h_slack_w, ref_bus);
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
            w_rows = gen_contingency::row_slack_weights(gen_data_, row_slack_off, n_ctg_all, n_bus,
                                                        base_state_->h_slack_bus, base_state_->h_slack_w,
                                                        ref_bus, "ContingencyAnalysisSession");
        }
        bool any_ds = false;
        for (const auto& ds : ds_rows) if (!ds.empty()) { any_ds = true; break; }
        if (!w_rows.empty() || any_ds)
            source.set_row_sbus_host(w_rows, base_state_->n_slack,
                                     any_ds ? ds_rows : std::vector<std::vector<std::pair<int, std::complex<double>>>>());

        // New driver: allocation, block-diagonal structure, cuDSS ANALYSIS.
        // Free the old one first so the two never coexist on the device.
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
            tol_);
        driver_cfg_ = cfg;
        ++driver_build_counter_;
    } else {
        // One-time costs (allocation, ANALYSIS, source preprocessing) are
        // not paid again: report them as 0 for this run.
        solver_->mark_reused(/*hot=*/true);
    }
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
            h_bus_vn_kv_, sn_mva_);
        t_admittance_upload_ms = solver_->branch_data_upload_ms();

        solver_->set_violation_limits(
            h_bus_vmin_kv_, h_bus_vmax_kv_,
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
    const physical_checks::SetupTimes t_phys = physical_checks::before_solve(
        phys_, *solver_, "ContingencyAnalysisSession", violation_tol_, sn_mva_,
        base_state_->timings.converged, d_gen_off_ptr, n_gen_off,
        physical_checks::RowTargets{(rd_gp && rd_targets.size() > 0) ? &rd_targets : nullptr, true,
                                    nullptr, false});
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
            ov.pq_buses = row_pv_to_pq_;
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
    timings_ = solver_->solve();

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
    timings_.t_session_run_wall_ms = ms_since(t_run0);
}

// =============================================================================
// compute_flows
// =============================================================================
void ContingencyAnalysisSession::compute_flows()
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
        h_bus_vn_kv_, sn_mva_);
    timings_.t_branch_data_upload_ms += solver_->branch_data_upload_ms();

    // Launch one kernel over ALL contingencies using d_V_results as input.
    const int n_ctg  = solver_->n_contingencies;
    const int n_bra  = solver_->n_branches_;
    const int n_bus  = base_state_->n_bus;
    const int total  = n_ctg * n_bra;
    const cudaStream_t cs = solver_->cs;

    CudaTimer flow_timer(cs);
    flow_timer.start();

    compute_branch_flows_kernel<<<(total + SESSION_BS - 1) / SESSION_BS, SESSION_BS, 0, cs>>>(
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
    std::vector<int> h_zero;
    for (int c = 0; c < n_ctg; ++c)
        for (int l : disconnected_per_ctg_[static_cast<size_t>(c)])
            h_zero.push_back(c * n_bra + l);

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

    // stop_ms() records + synchronizes the stop event, so this also serves as
    // the sync point that lets d_zero be safely destroyed below.
    timings_.t_flow_computation += flow_timer.stop_ms();
    // d_zero destroyed here, after sync

    // D→H download of flow results — timed separately.
    const int n = n_ctg * n_bra;
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
    if (bus_vmin_kv.size() != n_bus || bus_vmax_kv.size() != n_bus)
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
    const int n = solver_->n_contingencies * base_state_->n_bus;
    CplxVect out(n);
    solver_->download_into(out.data(), static_cast<size_t>(n), solver_->d_V_results);   // pinned, pipelined
    timings_.t_copy_V_to_host_ms = ms_since(t_copy_start);
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
                                                std::uintptr_t d_V_src, std::uintptr_t d_res_src)
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
    for (int r : src_rows)
        if (r < 0)
            throw std::runtime_error("ContingencyAnalysisSession::overwrite_rows: negative source row");
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
