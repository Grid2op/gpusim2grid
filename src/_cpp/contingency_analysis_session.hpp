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
#include "reordering_alg.hpp"
#include "matching_alg.hpp"

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
    ReorderingAlg reordering_alg_ = ReorderingAlg::Default;
    MatchingAlg matching_alg_ = MatchingAlg::None;

    // handle_disconnected_grid: when true, a contingency that splits the grid is
    // solved on its largest connected component (the rest is frozen and reported
    // as NaN) instead of being skipped — unless it strands the angle reference or
    // a controller bus, which is still skipped. Mutable; takes effect on the next
    // run(). mask_cfg_ is built once in the ctor from the base case + ledger.
    bool       handle_disconnected_grid_ = false;
    MaskConfig mask_cfg_;

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
    // compute_limit_violations (opt-in; fused on-device per-chunk check --
    // see set_compute_limit_violations()/set_limits()/run()).
    // =========================================================================
    bool     compute_limit_violations_ = false;
    double   violation_tol_            = 1e-6;   // dedicated; independent of tol_base
    int      violation_capacity_       = 16;     // K; bounds memory at n_ctg*K regardless
                                                  // of n_bus/n_branches -- see set_limits()
    bool     has_limits_               = false;
    bool     has_violations_result_    = false;
    int      n_lines_                  = 0;      // branch ordering split (lines-then-trafos)

    RealVect h_bus_vmin_kv_, h_bus_vmax_kv_;               // [n_bus], solver numbering
    RealVect h_branch_limit_a1_ka_, h_branch_limit_a2_ka_; // [n_branches], lines-then-trafos

    // =========================================================================
    // Contingency data (populated by build_contingencies())
    // =========================================================================
    std::vector<Contingency>      contingencies_;
    std::vector<std::vector<int>> disconnected_per_ctg_;  // tripped branch IDs per ctg
    bool has_contingencies_ = false;

    // Base-case NR time captured at construction (before run() is called).
    double t_base_case_ms_ = 0.;

    // Timings accumulated across run() + compute_flows(). Mutable so the const
    // get_V_results()/get_residuals() accessors can record their own D→H
    // transfer time (t_copy_V_to_host_ms / t_copy_residuals_to_host_ms)
    // without relaxing their constness.
    mutable BatchTimings timings_;

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
        const LedgerData* ledger = nullptr,  // augmented-J description (bridge path)
        bool   presolved_v   = false  // trust Vinit as already converged; see AcPfNrState
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
    // compute_limit_violations
    // Opt-in fused per-chunk voltage/current/divergence check (mirrors
    // lightsim2grid's ContingencyAnalysis.compute_limit_violations flag/name).
    // When enabled, requires set_branch_data() and set_limits() to have been
    // called before run(); the check then runs on-device, per chunk, writing
    // only a bounded O(n_contingencies * violation_capacity) compact buffer --
    // the full dense d_V_results/d_or_amps_results/d_ex_amps_results are never
    // required for this path (though set_branch_data()/compute_flows() remain
    // available, unchanged, for callers who also want full flows).
    // Changing the flag is a no-op if unchanged; otherwise it clears any
    // previously computed violation results (mirrors lightsim2grid exactly).
    // =========================================================================
    bool get_compute_limit_violations() const { return compute_limit_violations_; }
    void set_compute_limit_violations(bool val) {
        if (val == compute_limit_violations_) return;
        compute_limit_violations_ = val;
        has_violations_result_ = false;
    }

    // =========================================================================
    // set_limits
    // Configure per-bus voltage (kV, solver numbering) and per-branch current
    // (kA, lines-then-trafos) limits for compute_limit_violations. NaN = not
    // configured for that element (matches lightsim2grid's convention).
    // n_lines splits the lines-then-trafos branch ordering for
    // LimitViolation.element_type/element_id de-concatenation. Required
    // before run() when compute_limit_violations is True.
    // =========================================================================
    void set_limits(
        Eigen::Ref<const RealVect> bus_vmin_kv,
        Eigen::Ref<const RealVect> bus_vmax_kv,
        Eigen::Ref<const RealVect> branch_limit_a1_ka,
        Eigen::Ref<const RealVect> branch_limit_a2_ka,
        int n_lines);

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

    // =========================================================================
    // compute_limit_violations result accessors — synchronous D→H copy on
    // demand, all cheap (O(n_contingencies * violation_capacity) or
    // O(n_contingencies)). Throw if run() hasn't been called with
    // compute_limit_violations=True.
    // =========================================================================
    Eigen::VectorXi get_violation_element_type() const;
    Eigen::VectorXi get_violation_element_id()   const;
    Eigen::VectorXi get_violation_side()         const;
    Eigen::VectorXi get_violation_type()         const;
    RealVect        get_violation_value()        const;
    RealVect        get_violation_limit()        const;
    Eigen::VectorXi get_violation_count()        const;
    Eigen::VectorXi get_violation_truncated()    const;

    // TRUE, uncapped per-type violation totals (independent of
    // violation_capacity/K, unlike get_violation_count() above which is
    // capped at K): -1 = not simulated, else the exact count.
    Eigen::VectorXi get_violation_count_low_voltage()  const;
    Eigen::VectorXi get_violation_count_high_voltage() const;
    Eigen::VectorXi get_violation_count_current()      const;

    // Non-copyable, non-movable (owns CUDA resources via unique_ptr)
    ContingencyAnalysisSession(const ContingencyAnalysisSession&)            = delete;
    ContingencyAnalysisSession& operator=(const ContingencyAnalysisSession&) = delete;
    ContingencyAnalysisSession(ContingencyAnalysisSession&&)                 = delete;
    ContingencyAnalysisSession& operator=(ContingencyAnalysisSession&&)      = delete;
};

#endif // CONTINGENCY_ANALYSIS_SESSION_HPP