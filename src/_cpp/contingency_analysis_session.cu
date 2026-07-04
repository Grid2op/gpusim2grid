// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency_analysis_session.cu
// =============================================================================

#include "contingency_analysis_session.hpp"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/contingency_batch.cuh"
#include "acpf_nr_kernels.cuh"
#include "contingency_analysis_helper.hpp"
#include "cuda_utils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <stdexcept>
#include <limits>
#include <utility>   // std::move

static constexpr int SESSION_BS = 256;

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
    int    device)
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
        device);
    t_base_case_ms_ = ms_since(t_base_start);
}

// =============================================================================
// set_branch_data
// =============================================================================
void ContingencyAnalysisSession::set_branch_data(
    Eigen::Ref<const Eigen::VectorXi> branch_from,
    Eigen::Ref<const Eigen::VectorXi> branch_to,
    Eigen::Ref<const CplxVect>        yff,
    Eigen::Ref<const CplxVect>        yft,
    Eigen::Ref<const CplxVect>        ytf,
    Eigen::Ref<const CplxVect>        ytt,
    Eigen::Ref<const RealVect>        bus_vn_kv,
    double sn_mva)
{
    h_branch_from_ = branch_from;
    h_branch_to_   = branch_to;
    h_yff_         = yff;
    h_yft_         = yft;
    h_ytf_         = ytf;
    h_ytt_         = ytt;
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

    const int n_branches = static_cast<int>(h_branch_from_.size());
    const int n_ctg      = static_cast<int>(branch_ids_per_ctg.size());

    contingencies_.clear();
    contingencies_.reserve(n_ctg);
    disconnected_per_ctg_.clear();
    disconnected_per_ctg_.reserve(n_ctg);

    for (int c = 0; c < n_ctg; ++c) {
        Contingency ctg;
        for (int l : branch_ids_per_ctg[c]) {
            if (l < 0 || l >= n_branches)
                throw std::runtime_error(
                    "build_contingencies: branch index out of range");

            const int i = h_branch_from_(l);
            const int j = h_branch_to_(l);

            // π-model Ybus modifications to SUBTRACT for this branch trip:
            //   (i,i) → yff   (ii self-admittance at from-bus)
            //   (j,j) → ytt   (jj self-admittance at to-bus)
            //   (i,j) → yft   (ij mutual admittance)
            //   (j,i) → ytf   (ji mutual admittance)
            ctg.triplets.push_back({i, i,  h_yff_(l).real(),  h_yff_(l).imag()});
            ctg.triplets.push_back({j, j,  h_ytt_(l).real(),  h_ytt_(l).imag()});
            ctg.triplets.push_back({i, j,  h_yft_(l).real(),  h_yft_(l).imag()});
            ctg.triplets.push_back({j, i,  h_ytf_(l).real(),  h_ytf_(l).imag()});
        }
        contingencies_.push_back(std::move(ctg));
        disconnected_per_ctg_.push_back(branch_ids_per_ctg[c]);
    }
    has_contingencies_ = true;
}

// =============================================================================
// run
// =============================================================================
void ContingencyAnalysisSession::run()
{
    if (!has_contingencies_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: call build_contingencies() before run()");

    // Host preprocessing (resolve_indices + check_connectivity + build_flat_patches)
    // captured by the BatchSource; mutates contingencies_ in-place so that
    // disconnected flags are observable below for the disconnected count.
    // The source rebalances the chunk size over the ACTIVE (simulated) count;
    // we read it back so the driver chunks consistently with the source.
    ContingencyBatch source(
        contingencies_,
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        Ybus_rm_,
        batch_size_);
    used_batch_size_ = source.used_batch_size();

    // (Re-)construct the solver — allows run() to be called multiple times.
    solver_ = std::make_unique<ContingencyAnalysisSolver>(
        *base_state_,
        std::move(source),
        static_cast<int>(contingencies_.size()),
        Ybus_rm_.outerIndexPtr(),
        Ybus_rm_.innerIndexPtr(),
        used_batch_size_,
        nb_iter_,
        strategy_type_,
        refactor_period_);

    // Run all chunks; fills d_V_results and d_residuals on device.
    // solve() carries preprocess/alloc/analysis timings from the ctor.
    timings_ = solver_->solve();

    // Wire in base-case and one-time setup timings captured before/during
    // solver construction.
    timings_.t_base_case_ms  = t_base_case_ms_;
    // Fold base-case sub-phases into the contingency-solver aggregates so that
    // t_preprocess_ms and t_alloc_ms give a complete picture of all one-time
    // CPU work and H→D transfers, not just the contingency-solver's own share.
    timings_.t_preprocess_ms += base_state_->timings.t_build_J_ms;
    timings_.t_alloc_ms      += base_state_->timings.t_upload_ms;

    // Disconnected contingencies are compacted out of the batch by the source
    // and never solved; the driver pre-fills their result slots with NaN.  Here
    // we only need to count them for the timing report.
    const int n_ctg = solver_->n_contingencies;
    int n_disconnected = 0;
    for (int i = 0; i < n_ctg; ++i)
        if (contingencies_[static_cast<size_t>(i)].disconnected)
            ++n_disconnected;

    timings_.n_disconnected = n_disconnected;
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
        h_yff_, h_yft_, h_ytf_, h_ytt_,
        h_bus_vn_kv_, sn_mva_);

    // Launch one kernel over ALL contingencies using d_V_results as input.
    const int n_ctg  = solver_->n_contingencies;
    const int n_bra  = solver_->n_branches_;
    const int n_bus  = base_state_->n_bus;
    const int total  = n_ctg * n_bra;
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
        n_bus, n_bra, 0, n_ctg, /*d_result_map=*/nullptr);

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

    solver_->cs.synchronize();
    // d_zero destroyed here, after sync

    timings_.t_flow_computation.wall_ms += ms_since(t_flow_start);

    // D→H download of flow results — timed separately.
    const int n = n_ctg * n_bra;
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
    const int n = solver_->n_contingencies * base_state_->n_bus;
    thrust::host_vector<cudaComplexType> h_V = solver_->d_V_results;
    CplxVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = eigen_cplx_type(
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].x),
            static_cast<eigen_real_type>(h_V[static_cast<size_t>(i)].y));
    return out;
}

RealVect ContingencyAnalysisSession::get_residuals() const
{
    if (!solver_)
        throw std::runtime_error("ContingencyAnalysisSession: call run() first");
    solver_->cs.synchronize();
    const int n = solver_->n_contingencies;
    thrust::host_vector<cuda_real_type> h_res = solver_->d_residuals;
    RealVect out(n);
    for (int i = 0; i < n; ++i)
        out(i) = static_cast<eigen_real_type>(h_res[static_cast<size_t>(i)]);
    return out;
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