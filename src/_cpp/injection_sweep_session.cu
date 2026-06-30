// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// injection_sweep_session.cu
// =============================================================================

#include "injection_sweep_session.hpp"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/injection_batch.cuh"
#include "acpf_nr_kernels.cuh"   // compute_branch_flows_kernel
#include "cu_complex_utils.h"
#include "cuda_utils.h"          // ms_since

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    const LedgerData* ledger)
    : Ybus_rm_(Ybus)
    , batch_size_(batch_size)
    , nb_iter_(nb_iter)
{
    (void)slack_ids;
    (void)slack_weights;

    auto t_base_start = std::chrono::steady_clock::now();
    base_state_ = std::make_unique<AcPfNrState>(
        Ybus, Vinit, Sbus, pv, pq,
        max_iter_base,
        static_cast<eigen_real_type>(tol_base),
        device, ledger);
    t_base_case_ms_ = ms_since(t_base_start);
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
}

// =============================================================================
// run
// =============================================================================
void InjectionSweepSession::run()
{
    if (!has_injections_)
        throw std::runtime_error(
            "InjectionSweepSession: call set_injections() before run()");

    const int n_bus = base_state_->n_bus;

    // Effective per-chunk batch size: balance the last chunk (same rule as
    // run_injection_sweep_gpu / run_contingency_analysis_gpu).
    {
        const size_t n_elem   = static_cast<size_t>(n_scenarios_);
        const size_t max_bs   = static_cast<size_t>(batch_size_);
        const size_t n_chunks = (n_elem + max_bs - 1) / max_bs;
        used_batch_size_ = static_cast<int>(
            n_chunks > 0 ? (n_elem + n_chunks - 1) / n_chunks : n_elem);
    }

    // Build host-side per-unit complex Sbus_all ((n_scenarios × n_bus) row-major).
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

    InjectionBatch source(std::move(h_Sbus_all), n_scenarios_, t_sbus_build_ms_);

    // (Re-)construct the solver — allows run() to be called multiple times.
    solver_ = std::make_unique<InjectionSweepSolver>(
        *base_state_,
        std::move(source),
        n_scenarios_,
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        used_batch_size_,
        nb_iter_,
        strategy_type_,
        refactor_period_);

    timings_ = solver_->solve();
    timings_.t_base_case_ms  = t_base_case_ms_;
    timings_.t_preprocess_ms += base_state_->timings.t_build_J_ms;
    timings_.t_alloc_ms      += base_state_->timings.t_upload_ms;
    timings_.n_disconnected   = 0;

    solver_->cs.synchronize();
}

// =============================================================================
// set_branch_data
// =============================================================================
void InjectionSweepSession::set_branch_data(
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
        h_yff_, h_yft_, h_ytf_, h_ytt_,
        h_bus_vn_kv_, sn_mva_);

    // Launch one kernel over ALL scenarios using d_V_results as input.
    // No tripped branches to zero — unlike contingency analysis, every
    // branch carries current in every scenario.
    const int n_scen = solver_->n_contingencies;
    const int n_bra  = solver_->n_branches_;
    const int n_bus  = base_state_->n_bus;
    const int total  = n_scen * n_bra;
    const cudaStream_t cs = solver_->cs;

    auto t_flow_start = std::chrono::steady_clock::now();

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

    solver_->cs.synchronize();
    timings_.t_flow_computation.wall_ms += ms_since(t_flow_start);

    // D→H download of flow results — timed separately.
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
    const int n = solver_->n_contingencies * base_state_->n_bus;
    thrust::host_vector<cudaComplexType> h_V = solver_->d_V_results;
    CplxVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].x),
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].y));
    return out;
}

RealVect InjectionSweepSession::get_residuals() const
{
    if (!solver_)
        throw std::runtime_error("InjectionSweepSession: call run() first");
    solver_->cs.synchronize();
    const int n = solver_->n_contingencies;
    thrust::host_vector<cuda_real_type> h_res = solver_->d_residuals;
    RealVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = static_cast<eigen_real_type>(h_res[static_cast<size_t>(i)]);
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