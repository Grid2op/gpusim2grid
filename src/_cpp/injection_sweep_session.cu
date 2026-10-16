// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// injection_sweep_session.cu
// =============================================================================

#include "injection_sweep_session.hpp"
#include <limits>
#include "contingency/physical_checks_impl.cuh"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/injection_batch.cuh"
#include "contingency/gen_v_override.hpp"   // GenVOverride, build_gen_v_override
#include "ledger_data.hpp"                    // LedgerData (VoltageControl groups)
#include "acpf_nr_kernels.cuh"   // compute_branch_flows_kernel
#include "cu_complex_utils.h"
#include "cuda_utils.h"          // ms_since

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

static constexpr int SESSION_BS = 256;

// =============================================================================
// Destructor — defined here where AcPfNrState / BatchPfDriver<InjectionBatch>
// are complete types, enabling unique_ptr to call delete correctly.
// =============================================================================
InjectionSweepSession::~InjectionSweepSession() = default;

// =============================================================================
// Constructor — base-case NR to convergence.
// =============================================================================
InjectionSweepSession::InjectionSweepSession(
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
    (void)slack_weights;
    // lightsim2grid's held controllers (set_hold_frozen_regulators) are pinned per
    // row by the mask streams, which an injection sweep does not have
    // (the outer loops pin every held controller per slot themselves)
    if (ledger != nullptr && !ledger->outer_loops)
        for (int h : ledger->vc_held)
            if (h)
                throw std::runtime_error(
                    "InjectionSweepSession: this grid holds frozen remote regulators in their "
                    "voltage-control group (lightsim2grid's set_hold_frozen_regulators), which "
                    "InjectionSweepGPU does not support: turn the option off on the grid, or "
                    "use ScenarioSweepGPU.");

    auto t_base_start = std::chrono::steady_clock::now();
    base_state_ = std::make_unique<AcPfNrState>(
        Ybus, Vinit, Sbus, pv, pq,
        max_iter_base,
        static_cast<eigen_real_type>(tol_base),
        device, ledger, presolved_v,
        /*diag_stop_before_state_correction=*/false,
        reordering_alg, matching_alg, pivot_epsilon_alg, ir_n_steps,
        debug_base_case, /*base_case_only=*/true,
        scaling_max_voltage_change, max_dVa, max_dVm);
    t_base_case_ms_ = ms_since(t_base_start);

    // Bus maps for set_gen_v(): Vm-fixed buses and the VoltageControl group
    // regulating each bus (see build_gen_v_bus_maps / GenVOverride).
    build_gen_v_bus_maps(base_state_->n_bus, pv, slack_ids,
                         base_state_->h_vm_col_of_bus,
                         ledger ? ledger->vc_reg_bus : std::vector<int>{},
                         h_is_vm_fixed_bus_, h_vc_group_of_bus_);
    // the outer loops' switchable buses keep a gen_v re-seed (see ScenarioSweepSession's ctor)
    if (ledger != nullptr && ledger->outer_loops)
        for (int b : ledger->switchable_vm_buses)
            if (b >= 0 && b < static_cast<int>(h_is_vm_fixed_bus_.size()))
                h_is_vm_fixed_bus_[static_cast<size_t>(b)] = 1;
    if (ledger) {
        h_vc_v_set_ = ledger->vc_v_set;
        h_vc_group_fixed_.assign(ledger->vc_reg_bus.size(), 0);
        for (size_t j = 0; j < ledger->vc_kind.size(); ++j)
            if (ledger->vc_kind[j] != 0)
                h_vc_group_fixed_[static_cast<size_t>(ledger->vc_group[j])] = 1;
    }
}

// =============================================================================
// set_gen_v
// =============================================================================
void InjectionSweepSession::set_gen_v(
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gen_v,
    Eigen::Ref<const Eigen::VectorXi> gen_bus)
{
    if (gen_v.cols() != gen_bus.size())
        throw std::runtime_error(
            "InjectionSweepSession::set_gen_v: gen_v's column count must "
            "equal gen_bus's length (one entry per generator)");
    if (gen_v.rows() <= 0)
        throw std::runtime_error(
            "InjectionSweepSession::set_gen_v: n_scenarios must be > 0");

    gen_v_     = gen_v;
    gen_bus_   = gen_bus;
    has_gen_v_ = true;
    gen_v_dirty_ = true;
}

// =============================================================================
// set_injections
// =============================================================================
void InjectionSweepSession::set_injections(
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
    double sn_mva)
{
    const int n_bus = base_state_->n_bus;

    if (p_mw.rows() != q_mvar.rows() || p_mw.cols() != q_mvar.cols())
        throw std::runtime_error(
            "InjectionSweepSession::set_injections: p_mw and q_mvar must have the same shape");
    if (p_mw.cols() != n_bus)
        throw std::runtime_error(
            "InjectionSweepSession::set_injections: second dim must equal n_bus");
    if (p_mw.rows() <= 0)
        throw std::runtime_error(
            "InjectionSweepSession::set_injections: n_scenarios must be > 0");
    if (sn_mva <= 0.0)
        throw std::runtime_error(
            "InjectionSweepSession::set_injections: sn_mva must be > 0");

    p_mw_          = p_mw;
    q_mvar_        = q_mvar;
    sn_mva_        = sn_mva;
    n_scenarios_   = static_cast<int>(p_mw.rows());
    has_injections_ = true;
    injections_dirty_ = true;
}

BatchDriverConfig InjectionSweepSession::_current_config() const
{
    BatchDriverConfig c;
    c.n_scenarios                = n_scenarios_;
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
    c.scheduling                 = scheduling_;
    return c;
}

// =============================================================================
// run
// =============================================================================
void InjectionSweepSession::run()
{
    const auto t_run0 = std::chrono::steady_clock::now();
    if (!has_injections_)
        throw std::runtime_error(
            "InjectionSweepSession: call set_injections() before run()");

    if (has_gen_v_ && static_cast<int>(gen_v_.rows()) != n_scenarios_)
        throw std::runtime_error(
            "InjectionSweepSession: set_gen_v()'s row count no longer "
            "matches set_injections()'s n_scenarios -- call set_gen_v() "
            "again after changing set_injections()");

    const int n_bus = base_state_->n_bus;
    const bool continuous = (scheduling_ == BatchScheduling::Continuous);
    if (continuous)   // before any allocation / cuDSS ANALYSIS
        check_continuous_scheduling(strategy_type_, nb_iter_, nb_iter_per_round_, tol_);
    if (outer_) {
        auto refuse = [](const std::string& what) {
            throw std::runtime_error("InjectionSweepSession: the outer loops do not support " + what + " yet");
        };
        check_outer_loops_scheduling("InjectionSweepSession", continuous, nb_iter_per_round_);
        if (strategy_type_ != ContingencySolverType::DirectRefactorEvery)
            refuse("a strategy other than 'direct_refactor_every'");
        if (phys_.compute_physical_violations) refuse("compute_physical_violations");
    }

    // Hot run: same settings and row count as the live driver -- keep it
    // (allocations, cuDSS ANALYSIS, the policy's factorization state); new
    // injections / gen_v, if any, go into its live source. Otherwise build a
    // new one, as the first run() does. The continuous schedule always
    // rebuilds (its reuse is a later step, docs/dev_notes/continuous_batching.md
    // A.6).
    const BatchDriverConfig cfg = _current_config();
    const bool cold = !solver_ || cfg != driver_cfg_ || continuous;
    const bool new_rows = cold || injections_dirty_ || gen_v_dirty_;

    std::vector<cudaComplexType> h_Sbus_all;
    GenVOverride gen_v_override;
    t_sbus_build_ms_ = 0.;
    if (new_rows) {
        // Host-side per-unit complex Sbus_all ((n_scenarios × n_bus) row-major).
        auto t_sbus_start = std::chrono::steady_clock::now();
        h_Sbus_all.resize(static_cast<size_t>(n_scenarios_) * static_cast<size_t>(n_bus));
        const double inv_sn = 1.0 / sn_mva_;
        for (int s = 0; s < n_scenarios_; ++s) {
            for (int b = 0; b < n_bus; ++b) {
                h_Sbus_all[static_cast<size_t>(s) * n_bus + b] =
                    CudaFunHelper::my_make_cuComplex(
                        static_cast<cuda_real_type>(p_mw_(s, b)   * inv_sn),
                        static_cast<cuda_real_type>(q_mvar_(s, b) * inv_sn));
            }
        }
        if (has_gen_v_)
            gen_v_override = build_gen_v_override(gen_v_, gen_bus_, h_is_vm_fixed_bus_, h_vc_group_of_bus_);
        t_sbus_build_ms_ = ms_since(t_sbus_start);
    }

    double t_rows_upload_ms = 0.;
    if (cold) {
        // Effective per-chunk batch size: balance the last chunk (same rule as
        // run_injection_sweep_gpu / run_contingency_analysis_gpu). Continuous:
        // batch_size_ slots, never more than the rows.
        if (continuous) {
            used_batch_size_ = std::max(1, std::min(batch_size_, n_scenarios_));
        } else {
            const size_t n_elem   = static_cast<size_t>(n_scenarios_);
            const size_t max_bs   = static_cast<size_t>(batch_size_);
            const size_t n_chunks = (n_elem + max_bs - 1) / max_bs;
            used_batch_size_ = static_cast<int>(
                n_chunks > 0 ? (n_elem + n_chunks - 1) / n_chunks : n_elem);
        }

        InjectionBatch source(std::move(h_Sbus_all), n_scenarios_, t_sbus_build_ms_,
                              std::move(gen_v_override));

        // New driver: allocation, cuDSS ANALYSIS. Free the old one first so
        // the two never coexist on the device.
        solver_.reset();
        solver_ = std::make_unique<InjectionSweepSolver>(
            *base_state_,
            std::move(source),
            n_scenarios_,
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
        solver_->mark_reused(/*hot=*/true);
        if (new_rows) {
            auto t_up = std::chrono::steady_clock::now();
            solver_->source_.set_rows(std::move(h_Sbus_all), std::move(gen_v_override),
                                      static_cast<cudaStream_t>(solver_->cs));
            solver_->cs.synchronize();
            t_rows_upload_ms = ms_since(t_up);
        }
    }
    injections_dirty_ = false;
    gen_v_dirty_      = false;
    // per-run knobs, no rebuild needed
    solver_->nb_iter_           = nb_iter_;
    solver_->nb_iter_per_round_ = nb_iter_per_round_;
    solver_->tol_               = tol_;

    // Post-solve physical checks (compute_physical_violations / compute_hvdc_p_
    // violations); an injection sweep never disconnects a generator (nullptr mask).
    const physical_checks::SetupTimes t_phys = physical_checks::before_solve(
        phys_, *solver_, "InjectionSweepSession", violation_tol_, sn_mva_,
        base_state_->timings.converged, /*d_gen_off=*/nullptr, /*n_gen=*/0,
        physical_checks::RowTargets{&gen_p_targets_, gen_p_targets_dirty_,
                                    &gen_pv_release_targets_, gen_pv_release_targets_dirty_});
    // The targets reached the driver only if the checks ran their setup: with
    // them off, keep the flags so turning them back on on this same driver
    // uploads what changed meanwhile.
    if (phys_.compute_physical_violations) {
        gen_p_targets_dirty_ = false;
        gen_pv_release_targets_dirty_ = false;
    }

    // The outer loops: the model decides on every row, from the rows' own targets.
    solver_->outer_hook_ = outer_.get();
    if (outer_) {
        outer_->set_row_targets(n_scenarios_, outer_gen_p_, {});
        // the rows' generator set-points, and the groups' v_set they drive
        outer::RowOverrides ov;
        const int n_gen = static_cast<int>(gen_bus_.size());
        if (has_gen_v_ && gen_v_.rows() == n_scenarios_ && gen_v_.cols() == n_gen) {
            ov.gen_vm.resize(static_cast<size_t>(n_scenarios_) * n_gen);
            for (int r = 0; r < n_scenarios_; ++r)
                for (int g = 0; g < n_gen; ++g) ov.gen_vm[static_cast<size_t>(r) * n_gen + g] = gen_v_(r, g);
            const GenVOverride gvo = build_gen_v_override(gen_v_, gen_bus_, h_is_vm_fixed_bus_, h_vc_group_of_bus_);
            const int n_grp = base_state_->n_vc_grp;
            if (gvo.has_vc_columns() && n_grp > 0) {
                const int k = gvo.k_active();
                ov.vc_v_set.assign(static_cast<size_t>(n_scenarios_) * n_grp, std::numeric_limits<double>::quiet_NaN());
                for (int r = 0; r < n_scenarios_; ++r)
                    for (int j = 0; j < k; ++j) {
                        const int grp = gvo.h_active_vc_group[static_cast<size_t>(j)];
                        const double val = static_cast<double>(gvo.h_gen_v_all[static_cast<size_t>(r) * k + j]);
                        if (grp >= 0 && std::isfinite(val)) ov.vc_v_set[static_cast<size_t>(r) * n_grp + grp] = val;
                    }
            }
        }
        outer_->set_row_overrides(std::move(ov));
    }

    timings_ = solver_->solve();
    if (!cold) {
        // mark_reused zeroed the source's own costs; a hot run that loaded
        // new rows still paid for building and uploading them.
        timings_.t_preprocess_ms += t_sbus_build_ms_;
        timings_.t_alloc_ms      += t_rows_upload_ms;
    }
    timings_.t_base_case_ms  = t_base_case_ms_;
    timings_.t_preprocess_ms += base_state_->timings.t_build_J_ms;
    timings_.t_alloc_ms      += base_state_->timings.t_upload_ms;
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
    timings_.n_disconnected   = 0;
    physical_checks::after_solve(phys_, timings_, t_phys);

    solver_->cs.synchronize();
    timings_.t_session_run_wall_ms = ms_since(t_run0);
}

// =============================================================================
// set_branch_data
// =============================================================================
void InjectionSweepSession::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff_eff,
    Eigen::Ref<const CplxVect>        yft_eff,
    Eigen::Ref<const CplxVect>        ytf_eff,
    Eigen::Ref<const CplxVect>        ytt_eff,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double sn_mva)
{
    h_branch_from_   = branch_from;
    h_branch_to_     = branch_to;
    h_yff_eff_           = yff_eff;
    h_yft_eff_           = yft_eff;
    h_ytf_eff_           = ytf_eff;
    h_ytt_eff_           = ytt_eff;
    h_bus_vn_kv_     = bus_vn_kv;
    sn_mva_          = sn_mva;
    has_branch_data_ = true;
}

// =============================================================================
// compute_flows
// =============================================================================
void InjectionSweepSession::compute_flows()
{
    if (!has_branch_data_)
        throw std::runtime_error(
            "InjectionSweepSession: call set_branch_data() before compute_flows()");
    if (!solver_)
        throw std::runtime_error(
            "InjectionSweepSession: call run() before compute_flows()");

    // Upload branch admittances to device and allocate result buffers.
    solver_->set_branch_data(
        h_branch_from_, h_branch_to_,
        h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_,
        h_bus_vn_kv_, sn_mva_);
    timings_.t_branch_data_upload_ms += solver_->branch_data_upload_ms();

    // Launch one kernel over ALL scenarios using d_V_results as input.
    // No tripped branches to zero — unlike contingency analysis, every
    // branch carries current in every scenario.
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
        thrust::raw_pointer_cast(solver_->d_yff_eff.data()),
        thrust::raw_pointer_cast(solver_->d_yft_eff.data()),
        thrust::raw_pointer_cast(solver_->d_ytf_eff.data()),
        thrust::raw_pointer_cast(solver_->d_ytt_eff.data()),
        thrust::raw_pointer_cast(solver_->d_base_current_A.data()),
        thrust::raw_pointer_cast(solver_->d_base_current_ex_A.data()),
        thrust::raw_pointer_cast(solver_->d_or_amps_results.data()),
        thrust::raw_pointer_cast(solver_->d_ex_amps_results.data()),
        n_bus, n_bra, 0, n_scen, /*d_result_map=*/nullptr);
    // the transformers the outer loops left on other taps: at the row's final taps
    solver_->override_outer_trafo_flows();

    // stop_ms() records + synchronizes the stop event.
    timings_.t_flow_computation += flow_timer.stop_ms();

    // D→H download of flow results — timed separately.
    const int n = n_scen * n_bra;
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
// Metadata
// =============================================================================
int InjectionSweepSession::n_scenarios() const
{
    return solver_ ? solver_->n_contingencies : n_scenarios_;
}

int InjectionSweepSession::n_bus() const
{
    return base_state_ ? base_state_->n_bus : 0;
}

int InjectionSweepSession::n_branches() const
{
    return solver_ ? solver_->n_branches_
                   : static_cast<int>(h_branch_from_.size());
}

// =============================================================================
// Result accessors
// =============================================================================
CplxVect InjectionSweepSession::get_V_results() const
{
    if (!solver_)
        throw std::runtime_error("InjectionSweepSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    const int n = solver_->n_contingencies * base_state_->n_bus;
    CplxVect out(n);
    solver_->download_into(out.data(), static_cast<size_t>(n), solver_->d_V_results);   // pinned, pipelined
    timings_.t_copy_V_to_host_ms = ms_since(t_copy_start);
    return out;
}

Eigen::VectorXi InjectionSweepSession::get_row_iterations() const
{
    if (!solver_) return Eigen::VectorXi();
    const std::vector<int>& v = solver_->row_iterations();
    return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<Eigen::Index>(v.size()));
}

Eigen::VectorXi InjectionSweepSession::get_row_status() const
{
    if (!solver_) return Eigen::VectorXi();
    const std::vector<int>& v = solver_->row_status();
    return Eigen::Map<const Eigen::VectorXi>(v.data(), static_cast<Eigen::Index>(v.size()));
}

RealVect InjectionSweepSession::get_residuals() const
{
    if (!solver_)
        throw std::runtime_error("InjectionSweepSession: call run() first");
    solver_->cs.synchronize();
    auto t_copy_start = std::chrono::steady_clock::now();
    const int n = solver_->n_contingencies;
    RealVect out(n);
    solver_->download_into(out.data(), static_cast<size_t>(n), solver_->d_residuals);   // pinned, pipelined
    timings_.t_copy_residuals_to_host_ms = ms_since(t_copy_start);
    return out;
}

RealVect InjectionSweepSession::get_or_amps() const
{
    if (h_or_amps_.size() == 0)
        throw std::runtime_error(
            "InjectionSweepSession: call run() and compute_flows() first");
    return h_or_amps_;
}

RealVect InjectionSweepSession::get_ex_amps() const
{
    if (h_ex_amps_.size() == 0)
        throw std::runtime_error(
            "InjectionSweepSession: call run() and compute_flows() first");
    return h_ex_amps_;
}

// =============================================================================
// Post-solve physical checks (compute_physical_violations / compute_hvdc_p_
// violations) -- shared glue in contingency/physical_checks_impl.cuh
// =============================================================================
void InjectionSweepSession::set_bus_q_capability(const BusQPlanData& plan)
{
    phys_.set_bus_q_plan(plan, base_state_->n_bus);
}

BusQViolationsResult InjectionSweepSession::get_bus_q_violations() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_bus_q(phys_, *solver_, /*n_case=*/false, "InjectionSweepSession",
                                        timings_.t_copy_violations_to_host_ms);
}

BusQViolationsResult InjectionSweepSession::get_bus_q_violations_n() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_bus_q(phys_, *solver_, /*n_case=*/true, "InjectionSweepSession",
                                        timings_.t_copy_violations_to_host_ms);
}

HvdcPViolationsResult InjectionSweepSession::get_hvdc_p_violations() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_hvdc_p(phys_, *solver_, /*n_case=*/false, "InjectionSweepSession",
                                         timings_.t_copy_violations_to_host_ms);
}

HvdcPViolationsResult InjectionSweepSession::get_hvdc_p_violations_n() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_hvdc_p(phys_, *solver_, /*n_case=*/true, "InjectionSweepSession",
                                         timings_.t_copy_violations_to_host_ms);
}

void InjectionSweepSession::set_gen_p_capability(const GenPPlanData& plan)
{
    phys_.set_gen_p_plan(plan, base_state_->n_bus);
}

GenPViolationsResult InjectionSweepSession::get_gen_p_violations() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_gen_p(phys_, *solver_, /*n_case=*/false, "InjectionSweepSession",
                                        timings_.t_copy_violations_to_host_ms);
}

GenPViolationsResult InjectionSweepSession::get_gen_p_violations_n() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_gen_p(phys_, *solver_, /*n_case=*/true, "InjectionSweepSession",
                                        timings_.t_copy_violations_to_host_ms);
}

void InjectionSweepSession::set_gen_p_targets(Eigen::Ref<const RealMatRM> targets)
{
    gen_p_targets_       = targets;
    gen_p_targets_dirty_ = true;
}

void InjectionSweepSession::set_gen_pv_release_capability(const GenPvReleasePlanData& plan)
{
    phys_.set_gen_pv_release_plan(plan, base_state_->n_bus);
}

GenPvReleaseViolationsResult InjectionSweepSession::get_gen_pv_release_violations() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_gen_pv_release(phys_, *solver_, /*n_case=*/false, "InjectionSweepSession",
                                                 timings_.t_copy_violations_to_host_ms);
}

GenPvReleaseViolationsResult InjectionSweepSession::get_gen_pv_release_violations_n() const
{
    if (!solver_) throw std::runtime_error("InjectionSweepSession: call run() first");
    return physical_checks::fetch_gen_pv_release(phys_, *solver_, /*n_case=*/true, "InjectionSweepSession",
                                                 timings_.t_copy_violations_to_host_ms);
}

void InjectionSweepSession::set_gen_pv_release_targets(Eigen::Ref<const RealMatRM> targets)
{
    gen_pv_release_targets_       = targets;
    gen_pv_release_targets_dirty_ = true;
}
