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
#include "acpf_nr_kernels.cuh"   // compute_branch_flows_kernel, zero_branch_flows_kernel
#include "cu_complex_utils.h"
#include "cuda_utils.h"          // ms_since
#include "ledger_data.hpp"       // LedgerData (mask_cfg_ controller-bus setup)

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <stdexcept>
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
{
    (void)slack_ids;
    (void)slack_weights;

    auto t_base_start = std::chrono::steady_clock::now();
    base_state_ = std::make_unique<AcPfNrState>(
        Ybus, Vinit, Sbus, pv, pq,
        max_iter_base,
        static_cast<eigen_real_type>(tol_base),
        device, ledger, presolved_v,
        /*diag_stop_before_state_correction=*/false,
        reordering_alg, matching_alg, pivot_epsilon_alg,
        debug_base_case, /*base_case_only=*/true,
        scaling_max_voltage_change, max_dVa, max_dVm);
    t_base_case_ms_ = ms_since(t_base_start);

    // Build the handle_disconnected_grid mask configuration once from the base
    // case (per-bus identity-row metadata + angle reference) and the ledger
    // (controller buses → skip-if-stranded). Cheap; only consulted by run()
    // when handle_disconnected_grid_ is enabled. Verbatim
    // ContingencyAnalysisSession's identical ctor block.
    {
        const int n_bus = base_state_->n_bus;
        mask_cfg_.row_info.p_row      = base_state_->h_p_row_of_bus;
        mask_cfg_.row_info.q_row      = base_state_->h_q_row_of_bus;
        mask_cfg_.row_info.p_diag_pos = base_state_->h_p_diag_pos;
        mask_cfg_.row_info.q_diag_pos = base_state_->h_q_diag_pos;

        mask_cfg_.is_reference_bus.assign(static_cast<size_t>(n_bus), 0);
        for (int b = 0; b < n_bus; ++b)
            if (base_state_->h_theta_col_of_bus[static_cast<size_t>(b)] < 0)
                mask_cfg_.is_reference_bus[static_cast<size_t>(b)] = 1;

        mask_cfg_.is_controller_bus.assign(static_cast<size_t>(n_bus), 0);
        if (ledger != nullptr) {
            auto mark = [&](int b){ if (b >= 0 && b < n_bus) mask_cfg_.is_controller_bus[static_cast<size_t>(b)] = 1; };
            for (int b : ledger->hvdc_bus1)  mark(b);
            for (int b : ledger->hvdc_bus2)  mark(b);
            for (int b : ledger->vc_bus)     mark(b);
            for (int b : ledger->vc_reg_bus) mark(b);
        }
    }
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
// set_gen_v
// =============================================================================
void ScenarioSweepSession::set_gen_v(
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> vm_pu)
{
    const int n_bus = base_state_->n_bus;

    if (vm_pu.cols() != n_bus)
        throw std::runtime_error(
            "ScenarioSweepSession::set_gen_v: second dim must equal n_bus");
    if (vm_pu.rows() <= 0)
        throw std::runtime_error(
            "ScenarioSweepSession::set_gen_v: n_scenarios must be > 0");
    if (has_injections_ && static_cast<int>(vm_pu.rows()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession::set_gen_v: row count must match "
            "set_injections()'s n_scenarios");

    gen_v_     = vm_pu;
    has_gen_v_ = true;
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

    if (has_topology_
            && static_cast<int>(contingencies_.size()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession: set_topology()'s row count no longer "
            "matches set_injections()'s n_scenarios — call set_topology() "
            "again after changing set_injections()");

    if (has_gen_v_ && static_cast<int>(gen_v_.rows()) != n_scenarios_)
        throw std::runtime_error(
            "ScenarioSweepSession: set_gen_v()'s row count no longer matches "
            "set_injections()'s n_scenarios -- call set_gen_v() again after "
            "changing set_injections()");

    if (handle_disconnected_grid_ &&
        strategy_type_ == ContingencySolverType::DirectBaseCaseFactors)
        throw std::runtime_error(
            "ScenarioSweepSession: handle_disconnected_grid is incompatible "
            "with the 'direct_base_case_factors' strategy (it reuses the unmasked "
            "base-case factors). Use 'direct_refactor_every' (default), "
            "'direct_iter0_only', or 'direct_refactor_every_n'.");

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
    }

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

    // Build host-side per-unit real gen_v_all, ORIGINAL (pre-compaction) row
    // order -- ScenarioSweepBatch permutes into active-slot order internally,
    // same as h_Sbus_all above. Empty when set_gen_v() was never called.
    std::vector<cuda_real_type> h_gen_v_all;
    if (has_gen_v_) {
        h_gen_v_all.resize(static_cast<size_t>(n_scenarios_) * static_cast<size_t>(n_bus));
        for (int s = 0; s < n_scenarios_; ++s)
            for (int b = 0; b < n_bus; ++b)
                h_gen_v_all[static_cast<size_t>(s) * n_bus + b] =
                    static_cast<cuda_real_type>(gen_v_(s, b));
    }

    // Host preprocessing (resolve_indices + connectivity/masking +
    // build_flat_patches + Sbus active-order permute), mutates contingencies_
    // in-place so disconnected flags are observable below.
    ScenarioSweepBatch source(
        contingencies_,
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        Ybus_rm_,
        std::move(h_Sbus_all),
        batch_size_,
        handle_disconnected_grid_ ? &mask_cfg_ : nullptr,
        std::move(h_gen_v_all));
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
