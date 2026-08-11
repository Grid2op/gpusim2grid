// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SCENARIO_SWEEP_SESSION_HPP
#define SCENARIO_SWEEP_SESSION_HPP

// =============================================================================
// scenario_sweep_session.hpp
//
// ScenarioSweepSession — stateful Python-facing solver for the row-aligned
// combined topology + injection sweep. Mirrors InjectionSweepSession and
// ContingencyAnalysisSession, composing both: row `i` of the injection
// matrices is solved together with row `i` of the topology (tripped-branch)
// list, independently of every other row.
//
// Lifecycle
// ---------
//   1. Construct → base-case NR runs immediately (AcPfNrState).
//   2. set_branch_data(...) → store host copies of π-model admittances
//      (needed for BOTH topology triplet construction and compute_flows()).
//   3. set_injections(p_mw, q_mvar, sn_mva) → store per-scenario injections;
//      fixes n_scenarios().
//   4. set_topology(branch_ids_per_scenario) → OPTIONAL; one branch-id list
//      per scenario (lines-then-trafos, same convention as
//      ContingencyAnalysisSession::build_contingencies). Row count must match
//      set_injections()'s n_scenarios. If never called, run() defaults every
//      scenario to "no branches tripped" (a plain injection sweep).
//   5. run() → BatchPfDriver<ScenarioSweepBatch> constructed + solve() called;
//      d_V_results / d_residuals filled on device. A scenario whose topology
//      change disconnects the grid is skipped (NaN) — see
//      ScenarioSweepBatch's doc; handle_disconnected_grid masking and
//      compute_limit_violations are NOT supported by this session (deferred
//      scope, see CLAUDE.md).
//   6. compute_flows() → flows for ALL scenarios at once from d_V_results;
//      each scenario's own tripped branches are zeroed device-side.
//
// Device data stays resident between steps; callers pull results via
// get_V_results() / get_residuals() / get_or_amps() / get_ex_amps(), each
// performing a synchronous D→H copy on demand.
// =============================================================================

#include "dtypes.hpp"
#include "timing_utils.hpp"
#include "contingency_analysis_helper.hpp"   // ContingencySolverType, Contingency
#include "reordering_alg.hpp"
#include "matching_alg.hpp"
#include "pivot_epsilon_alg.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"

#include <memory>
#include <vector>

// Forward-declare CUDA-dependent types to keep CUDA headers out of this file.
struct AcPfNrState;
struct LedgerData;
struct ScenarioSweepBatch;
template <typename BatchSource> struct BatchPfDriver;
using ScenarioSweepSolver = BatchPfDriver<ScenarioSweepBatch>;

struct ScenarioSweepSession {

    // Destructor defined in the .cu where AcPfNrState/ScenarioSweepSolver are
    // complete types.
    ~ScenarioSweepSession();

    // =========================================================================
    // Owned GPU state
    // =========================================================================
    std::unique_ptr<AcPfNrState>          base_state_;
    std::unique_ptr<ScenarioSweepSolver>  solver_;   // null until run()

    // RowMajor Ybus copy — needed to build the block-diag CSR + resolve_indices.
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
    PivotEpsilonAlg pivot_epsilon_alg_ = PivotEpsilonAlg::Default;

    // NR step-scaling (MaxVoltageChange), forwarded to BOTH base_state_'s
    // AcPfNrState AND the batch driver used by run() -- see the other
    // sessions' own doc for the rationale (per-batch-slot alpha).
    bool   scaling_max_voltage_change_ = false;
    double max_dVa_ = 0.5;
    double max_dVm_ = 0.1;

    // =========================================================================
    // Host branch data (set_branch_data()) — needed for set_topology()'s
    // triplet construction AND compute_flows().
    // =========================================================================
    Eigen::VectorXi h_branch_from_;
    Eigen::VectorXi h_branch_to_;
    CplxVect        h_yff_, h_yft_, h_ytf_, h_ytt_;
    RealVect        h_bus_vn_kv_;
    double          sn_mva_          = 100.0;
    bool            has_branch_data_ = false;

    // handle_disconnected_grid: when true, a scenario whose topology change
    // splits the grid is solved on its largest connected component (the rest
    // is frozen and reported as NaN) instead of being skipped — unless it
    // strands the angle reference or a controller bus, which is still
    // skipped. Mutable; takes effect on the next run(). mask_cfg_ is built
    // once in the ctor from the base case + ledger.
    bool       handle_disconnected_grid_ = false;
    MaskConfig mask_cfg_;

    // =========================================================================
    // compute_limit_violations (opt-in; fused on-device per-chunk check --
    // see set_compute_limit_violations()/set_limits()/run()).
    // =========================================================================
    bool     compute_limit_violations_ = false;
    double   violation_tol_            = 1e-6;   // dedicated; independent of tol_base
    int      violation_capacity_       = 16;     // K; bounds memory at n_scenarios*K
    bool     has_limits_               = false;
    bool     has_violations_result_    = false;
    int      n_lines_                  = 0;      // branch ordering split (lines-then-trafos)

    RealVect h_bus_vmin_kv_, h_bus_vmax_kv_;               // [n_bus], solver numbering
    RealVect h_branch_limit_a1_ka_, h_branch_limit_a2_ka_; // [n_branches], lines-then-trafos

    // =========================================================================
    // Host injection data (set_injections()) — (n_scenarios × n_bus)
    // row-major physical-unit arrays, AC-solver bus numbering.
    // =========================================================================
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> p_mw_;
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> q_mvar_;
    int    n_scenarios_    = 0;
    bool   has_injections_ = false;

    // =========================================================================
    // Host topology data (set_topology()) — one Contingency (and tripped-
    // branch list, for compute_flows' zero-flow step) per scenario, row-
    // aligned with p_mw_/q_mvar_. has_topology_ == false means "run()
    // defaults every scenario to no trips" (pure injection sweep).
    // =========================================================================
    std::vector<Contingency>      contingencies_;
    std::vector<std::vector<int>> tripped_branches_per_scenario_;
    bool has_topology_ = false;

    // Host-side flow result storage (filled by compute_flows()).
    RealVect h_or_amps_;
    RealVect h_ex_amps_;

    // Base-case NR time captured at construction (before run() is called).
    double t_base_case_ms_ = 0.;

    // Host-side preprocessing time for the per-unit Sbus build (run()).
    double t_sbus_build_ms_ = 0.;

    // Timings from the most recent run()/compute_flows(). Mutable so the const
    // accessors can record their own D→H transfer time.
    mutable BatchTimings timings_;

    // =========================================================================
    // Constructor — runs base-case NR to convergence.
    // =========================================================================
    ScenarioSweepSession(
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
        const LedgerData* ledger = nullptr,
        bool   presolved_v   = false,
        ReorderingAlg reordering_alg = ReorderingAlg::Default,
        MatchingAlg matching_alg = MatchingAlg::None,
        PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
        bool   debug_base_case = false,
        bool   scaling_max_voltage_change = false,
        double max_dVa = 0.5,
        double max_dVm = 0.1
    );

    // =========================================================================
    // set_branch_data — store host copies of π-model admittances. Must be
    // called before set_topology() and before compute_flows().
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
    // set_injections — store the (n_scenarios × n_bus) physical-unit
    // injection arrays. Must be called before run(). Fixes n_scenarios().
    // =========================================================================
    void set_injections(
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
        double sn_mva
    );

    // =========================================================================
    // set_topology — one branch-id list per scenario (lines-then-trafos),
    // row-aligned with set_injections(). Requires set_branch_data() first.
    // Optional: if never called, run() defaults every scenario to "no
    // branches tripped".
    // =========================================================================
    void set_topology(const std::vector<std::vector<int>>& branch_ids_per_scenario);

    // =========================================================================
    // run — constructs BatchPfDriver<ScenarioSweepBatch> + runs the chunk
    // loop. A scenario whose topology change disconnects the grid is skipped
    // (NaN residual/voltage) — unless handle_disconnected_grid_ is set, in
    // which case only scenarios stranding the angle reference or a
    // controller bus are left as NaN (the rest solve on their largest
    // connected component, masked buses reported as NaN).
    // =========================================================================
    void run();

    // =========================================================================
    // compute_flows — branch flows for ALL scenarios at once from
    // d_V_results; each scenario's own tripped branches are zeroed
    // device-side. Requires run() and set_branch_data().
    // =========================================================================
    void compute_flows();

    // =========================================================================
    // compute_limit_violations
    // Opt-in fused per-chunk voltage/current/divergence check (mirrors
    // ContingencyAnalysisSession's flag of the same name). When enabled,
    // requires set_branch_data() and set_limits() to have been called before
    // run(); the check then runs on-device, per chunk, writing only a
    // bounded O(n_scenarios * violation_capacity) compact buffer.
    // Changing the flag is a no-op if unchanged; otherwise it clears any
    // previously computed violation results.
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
    // configured for that element. n_lines splits the lines-then-trafos
    // branch ordering for LimitViolation.element_type/element_id
    // de-concatenation. Required before run() when compute_limit_violations
    // is True.
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

    // Per-scenario disconnected flag (1 == topology change islanded the grid,
    // scenario skipped/NaN; 0 == solved). Size n_scenarios(); empty before
    // run() has been called.
    Eigen::VectorXi get_disconnected() const;

    // =========================================================================
    // compute_limit_violations result accessors — synchronous D→H copy on
    // demand, all cheap (O(n_scenarios * violation_capacity) or
    // O(n_scenarios)). Throw if run() hasn't been called with
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
    ScenarioSweepSession(const ScenarioSweepSession&)            = delete;
    ScenarioSweepSession& operator=(const ScenarioSweepSession&) = delete;
    ScenarioSweepSession(ScenarioSweepSession&&)                 = delete;
    ScenarioSweepSession& operator=(ScenarioSweepSession&&)      = delete;
};

#endif // SCENARIO_SWEEP_SESSION_HPP
