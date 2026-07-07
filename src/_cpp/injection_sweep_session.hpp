// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INJECTION_SWEEP_SESSION_HPP
#define INJECTION_SWEEP_SESSION_HPP

// =============================================================================
// injection_sweep_session.hpp
//
// InjectionSweepSession — stateful Python-facing solver for the batched
// injection power flow sweep.  Mirrors ContingencyAnalysisSession.
//
// Lifecycle
// ---------
//   1. Construct → base-case NR runs immediately (AcPfNrState), so the
//      single-system cuDSS ANALYSIS and converged base voltages are reused
//      across every subsequent sweep.
//   2. set_injections(p_mw, q_mvar, sn_mva) → store host injection arrays.
//   3. run() → BatchPfDriver<InjectionBatch> constructed + solve() called;
//      d_V_results / d_residuals filled on device.
//
// Device data stays resident between steps; callers pull results via
// get_V_results() / get_residuals(), each performing a synchronous D→H copy
// on demand.
//
// Why a session (vs. the one-shot run_injection_sweep_gpu): the base-case NR
// — and the single-system J sparsity / scatter maps it builds — is solved
// exactly once at construction.  Re-running with a new injection matrix (or a
// new batch_size / strategy) reuses that base state.
//
// AcPfNrState and BatchPfDriver<InjectionBatch> are forward-declared to keep
// CUDA headers out of this pure-C++ header.  The destructor is declared here
// and defined in the .cu where the complete types are visible.
// =============================================================================

#include "dtypes.hpp"
#include "timing_utils.hpp"
#include "contingency_analysis_helper.hpp"   // ContingencySolverType
#include "reordering_alg.hpp"
#include "matching_alg.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"

#include <memory>

// Forward-declare CUDA-dependent types.
struct AcPfNrState;
struct LedgerData;
struct InjectionBatch;
template <typename BatchSource> struct BatchPfDriver;
using InjectionSweepSolver = BatchPfDriver<InjectionBatch>;

struct InjectionSweepSession {

    // Destructor defined in the .cu where the complete types are visible.
    ~InjectionSweepSession();

    // =========================================================================
    // Owned GPU state
    // =========================================================================
    std::unique_ptr<AcPfNrState>        base_state_;
    std::unique_ptr<InjectionSweepSolver> solver_;   // null until run()

    // RowMajor Ybus copy — needed by BatchPfDriver to build the block-diag CSR.
    Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor> Ybus_rm_;

    // =========================================================================
    // Configuration (stored for run())
    // =========================================================================
    int        batch_size_      = 0;
    int        used_batch_size_ = 0;
    int        nb_iter_         = 0;
    int        refactor_period_ = 1;
    ContingencySolverType strategy_type_ = ContingencySolverType::DirectRefactorEvery;
    ReorderingAlg reordering_alg_ = ReorderingAlg::Default;
    MatchingAlg matching_alg_ = MatchingAlg::None;

    // =========================================================================
    // Host injection data (stored by set_injections(), consumed by run())
    //
    // p_mw_ / q_mvar_ are (n_scenarios × n_bus) row-major physical-unit
    // arrays; sn_mva_ converts them to per-unit complex Sbus inside run().
    // =========================================================================
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_mw_;
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_mvar_;
    double sn_mva_           = 100.0;
    int    n_scenarios_      = 0;
    bool   has_injections_   = false;

    // =========================================================================
    // Host branch data (stored for compute_flows())
    // =========================================================================
    Eigen::VectorXi h_branch_from_;
    Eigen::VectorXi h_branch_to_;
    CplxVect        h_yff_, h_yft_, h_ytf_, h_ytt_;
    RealVect        h_bus_vn_kv_;
    bool            has_branch_data_ = false;

    // Host-side flow result storage (filled by compute_flows()).
    RealVect h_or_amps_;
    RealVect h_ex_amps_;

    // Base-case NR time captured at construction (before run() is called).
    double t_base_case_ms_ = 0.;

    // Host-side preprocessing time for the per-unit Sbus build (run()).
    double t_sbus_build_ms_ = 0.;

    // Timings from the most recent run(). Mutable so the const
    // get_V_results()/get_residuals() accessors can record their own D→H
    // transfer time (t_copy_V_to_host_ms / t_copy_residuals_to_host_ms)
    // without relaxing their constness.
    mutable BatchTimings timings_;

    // =========================================================================
    // Constructor — runs base-case NR to convergence (AcPfNrState construction).
    // =========================================================================
    InjectionSweepSession(
        const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
        Eigen::Ref<const CplxVect>                  Vinit,
        Eigen::Ref<const CplxVect>                  Sbus,
        Eigen::Ref<const Eigen::VectorXi>           slack_ids,
        Eigen::Ref<const RealVect>                  slack_weights,
        Eigen::Ref<const Eigen::VectorXi>           pv,
        Eigen::Ref<const Eigen::VectorXi>           pq,
        int    batch_size,
        int    nb_iter,
        int    max_iter_base = 10,
        double tol_base      = 1e-6,
        int    device        = -1,
        const LedgerData* ledger = nullptr,  // augmented-J description (bridge path)
        bool   presolved_v   = false  // trust Vinit as already converged; see AcPfNrState
    );

    // =========================================================================
    // set_injections
    //   Store the (n_scenarios × n_bus) physical-unit injection arrays.
    //   Must be called before run().  May be called repeatedly to sweep
    //   different injection sets reusing the same base case.
    // =========================================================================
    void set_injections(
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
        double sn_mva
    );

    // =========================================================================
    // set_branch_data
    //   Store host copies of π-model admittances.  Must be called before
    //   compute_flows().
    // =========================================================================
    void set_branch_data(
        Eigen::Ref<const Eigen::VectorXi> branch_from,
        Eigen::Ref<const Eigen::VectorXi> branch_to,
        Eigen::Ref<const CplxVect>        yff,
        Eigen::Ref<const CplxVect>        yft,
        Eigen::Ref<const CplxVect>        ytf,
        Eigen::Ref<const CplxVect>        ytt,
        Eigen::Ref<const RealVect>        bus_vn_kv,
        double sn_mva
    );

    // =========================================================================
    // run
    //   Constructs BatchPfDriver<InjectionBatch> + runs the chunk loop.
    //   Fills d_V_results and d_residuals on device.  May be called multiple
    //   times (e.g. after changing batch_size / nb_iter / strategy_type or
    //   after a new set_injections()).
    // =========================================================================
    void run();

    // =========================================================================
    // compute_flows
    //   Computes branch flows for ALL scenarios at once from d_V_results.
    //   Requires run() and set_branch_data() to have been called.
    // =========================================================================
    void compute_flows();

    // =========================================================================
    // Metadata accessors
    // =========================================================================
    int n_scenarios() const;
    int n_bus() const;
    int n_branches() const;

    // =========================================================================
    // Result accessors — synchronous D→H copy on demand.
    // =========================================================================
    CplxVect get_V_results()  const;   // (n_scenarios * n_bus,)      complex
    RealVect get_residuals()  const;   // (n_scenarios,)               real
    RealVect get_or_amps()    const;   // (n_scenarios * n_branches,) real
    RealVect get_ex_amps()    const;   // (n_scenarios * n_branches,) real
    BatchTimings get_timings() const { return timings_; }

    // Non-copyable, non-movable (owns CUDA resources via unique_ptr)
    InjectionSweepSession(const InjectionSweepSession&)            = delete;
    InjectionSweepSession& operator=(const InjectionSweepSession&) = delete;
    InjectionSweepSession(InjectionSweepSession&&)                 = delete;
    InjectionSweepSession& operator=(InjectionSweepSession&&)      = delete;
};

#endif // INJECTION_SWEEP_SESSION_HPP