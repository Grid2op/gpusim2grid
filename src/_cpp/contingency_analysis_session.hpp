// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CONTINGENCY_ANALYSIS_SESSION_HPP
#define CONTINGENCY_ANALYSIS_SESSION_HPP

// =============================================================================
// contingency_analysis_session.hpp
//
// ContingencyAnalysisSession — stateful Python-facing solver that owns the
// full pipeline from base-case NR through contingency chunk loop.
//
// Lifecycle
// ---------
//   1. Construct → base-case NR runs immediately (AcPfNrState).
//   2. set_branch_data() → store host copies of π-model admittances.
//   3. build_contingencies(list[list[int]]) → convert branch IDs to Ybus
//      triplets; record which branches are tripped per contingency.
//   4. run() → ContingencyAnalysisSolver constructed + solve() called;
//      d_V_results / d_residuals filled on device.
//   5. compute_flows() → flows computed for ALL contingencies at once from
//      d_V_results; disconnected entries zeroed device-side.
//
// Device data stays resident between steps; callers pull results via
// get_V_results() / get_residuals() / get_or_amps() / get_ex_amps(),
// which each perform a synchronous D→H copy on demand.
//
// AcPfNrState and ContingencyAnalysisSolver are forward-declared to keep
// CUDA headers out of this file (they live in .cuh files).  The destructor
// is declared but not defaulted here; it is defined in the .cu file where
// the complete types are visible, enabling unique_ptr to call delete safely.
// =============================================================================

#include "dtypes.hpp"
#include "timing_utils.hpp"
#include "contingency_analysis_helper.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"

#include <memory>
#include <vector>

// Forward-declare CUDA-dependent types to avoid pulling CUDA headers
// into any pure-C++ translation unit that includes this header.
//
// ContingencyAnalysisSolver is now a type alias for BatchPfDriver<ContingencyBatch>;
// the forward template declaration + alias is fully sufficient for
// std::unique_ptr<ContingencyAnalysisSolver> since the destructor is defined
// in the .cu where the complete type is visible.
struct AcPfNrState;
struct LedgerData;
struct ContingencyBatch;
template <typename BatchSource> struct BatchPfDriver;
using ContingencyAnalysisSolver = BatchPfDriver<ContingencyBatch>;

struct ContingencyAnalysisSession {

    // Destructor defined in .cu where AcPfNrState/ContingencyAnalysisSolver
    // are complete.
    ~ContingencyAnalysisSession();

    // =========================================================================
    // Owned GPU state
    // =========================================================================
    std::unique_ptr<AcPfNrState>               base_state_;
    std::unique_ptr<ContingencyAnalysisSolver>  solver_;   // null until run()

    // RowMajor Ybus copy — needed by resolve_indices in run()
    Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor> Ybus_rm_;

    // =========================================================================
    // Configuration (stored for run())
    // =========================================================================
    int        batch_size_      = 0;
    int        used_batch_size_ = 0;
    int        nb_iter_         = 0;
    int        refactor_period_ = 1;
    ContingencySolverType strategy_type_   = ContingencySolverType::DirectRefactorEvery;

    // =========================================================================
    // Host branch data (stored for compute_flows() and build_contingencies())
    // =========================================================================
    Eigen::VectorXi h_branch_from_;
    Eigen::VectorXi h_branch_to_;
    CplxVect        h_yff_, h_yft_, h_ytf_, h_ytt_;
    RealVect        h_bus_vn_kv_;
    double          sn_mva_          = 100.0;
    bool            has_branch_data_ = false;

    // Host-side flow result storage (filled by compute_flows()).
    RealVect h_or_amps_;
    RealVect h_ex_amps_;

    // =========================================================================
    // Contingency data (populated by build_contingencies())
    // =========================================================================
    std::vector<Contingency>      contingencies_;
    std::vector<std::vector<int>> disconnected_per_ctg_;  // tripped branch IDs per ctg
    bool has_contingencies_ = false;

    // Base-case NR time captured at construction (before run() is called).
    double t_base_case_ms_ = 0.;

    // Timings accumulated across run() + compute_flows().
    BatchTimings timings_;

    // =========================================================================
    // Constructor
    // Runs base-case NR to convergence (AcPfNrState construction).
    // =========================================================================
    ContingencyAnalysisSession(
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
        const LedgerData* ledger = nullptr   // augmented-J description (bridge path)
    );

    // =========================================================================
    // set_branch_data
    // Store host copies of π-model admittances.  Must be called before
    // build_contingencies() and before compute_flows().
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
    // build_contingencies
    // Converts branch-ID lists to Ybus triplet vectors for each contingency.
    // Requires set_branch_data() to have been called.
    // =========================================================================
    void build_contingencies(const std::vector<std::vector<int>>& branch_ids_per_ctg);

    // =========================================================================
    // run
    // Constructs ContingencyAnalysisSolver + runs the chunk loop.
    // Fills d_V_results and d_residuals on device.
    // Disconnected contingencies receive NaN residuals / voltages on device.
    // Timings are accumulated into timings_; retrieve via get_timings() after
    // compute_flows() for a complete picture.
    // =========================================================================
    void run();

    // =========================================================================
    // compute_flows
    // Computes branch flows for ALL contingencies at once from d_V_results.
    // Tripped-branch entries are zeroed device-side.
    // Requires run() and set_branch_data() to have been called.
    // =========================================================================
    void compute_flows();

    // =========================================================================
    // Metadata accessors
    // =========================================================================
    int n_contingencies() const;
    int n_bus() const;
    int n_branches() const;

    // =========================================================================
    // Result accessors — synchronous D→H copy on demand.
    // =========================================================================
    CplxVect get_V_results()  const;   // (n_contingencies * n_bus,)  complex
    RealVect get_residuals()  const;   // (n_contingencies,)           real
    RealVect get_or_amps()    const;   // (n_contingencies * n_branches,) real
    RealVect get_ex_amps()    const;   // (n_contingencies * n_branches,) real
    BatchTimings get_timings() const { return timings_; }

    // Non-copyable, non-movable (owns CUDA resources via unique_ptr)
    ContingencyAnalysisSession(const ContingencyAnalysisSession&)            = delete;
    ContingencyAnalysisSession& operator=(const ContingencyAnalysisSession&) = delete;
    ContingencyAnalysisSession(ContingencyAnalysisSession&&)                 = delete;
    ContingencyAnalysisSession& operator=(ContingencyAnalysisSession&&)      = delete;
};

#endif // CONTINGENCY_ANALYSIS_SESSION_HPP