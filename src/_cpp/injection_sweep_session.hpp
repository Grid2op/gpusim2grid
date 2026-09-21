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
#include "pivot_epsilon_alg.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "contingency/physical_checks_data.hpp"  // PhysicalChecksConfig, BusQPlanData, *ViolationsResult

#include <memory>
#include <vector>

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
    PivotEpsilonAlg pivot_epsilon_alg_ = PivotEpsilonAlg::Default;

    // NR step-scaling (MaxVoltageChange), forwarded to BOTH base_state_'s
    // AcPfNrState AND the batch driver used by run() -- see AcPfNrState's own
    // doc. Computed PER BATCH SLOT (each scenario gets its own alpha from its
    // own max|dtheta|/max|dvm|, not a single alpha shared across the chunk).
    bool   scaling_max_voltage_change_ = false;
    double max_dVa_ = 0.5;
    double max_dVm_ = 0.1;

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
    // Vm-fixed bus mask (pv ∪ slack_ids without a ledger Vm unknown, built
    // once at construction, see build_gen_v_bus_maps) --
    // consulted by set_gen_v() below. See that method's own doc.
    // =========================================================================
    std::vector<char> h_is_vm_fixed_bus_;
    std::vector<int>  h_vc_group_of_bus_;   // VoltageControl group regulating each bus, -1: none
    std::vector<int>    h_vc_group_fixed_;  // per group: holds a member gen_v cannot move (SVC / station)
    std::vector<double> h_vc_v_set_;        // per group: base set-point (pu)

    // =========================================================================
    // Host generator target-voltage override (set_gen_v()) -- optional; see
    // that method's own doc. Empty (has_gen_v_ == false) means every row
    // keeps using the grid's own base-case voltage, exactly as before this
    // setter existed.
    // =========================================================================
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gen_v_;
    Eigen::VectorXi gen_bus_;
    bool   has_gen_v_        = false;

    // =========================================================================
    // Host branch data (stored for compute_flows())
    // =========================================================================
    Eigen::VectorXi h_branch_from_;
    Eigen::VectorXi h_branch_to_;
    CplxVect        h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_;
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

    // post-solve physical checks (see physical_checks() above)
    PhysicalChecksConfig phys_;
    // Residual gate of the physical checks (a row whose ||F||inf is NaN or above
    // it reports nothing), the same role violation_tol_ plays on the other two
    // sessions. Independent of tol_base.
    double violation_tol_ = 1e-6;

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
        bool   presolved_v   = false,  // trust Vinit as already converged; see AcPfNrState
        // cuDSS alg choices, forwarded to BOTH base_state_'s AcPfNrState AND
        // this session's own reordering_alg_/matching_alg_/pivot_epsilon_alg_
        // members (used by the batch solver in run()) -- single source of
        // truth set once at construction; see AcPfNrState's own doc.
        ReorderingAlg reordering_alg = ReorderingAlg::Default,
        MatchingAlg matching_alg = MatchingAlg::None,
        PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
        // Opt-in diagnostic forwarded to base_state_ (see AcPfNrState's own
        // doc): force the pre-ground-truth cuDSS-solve derivation of
        // slack_absorbed/vc_q even when lightsim2grid's own converged values
        // are available. Default false (prefer ground truth).
        bool   debug_base_case = false,
        // NR step-scaling (MaxVoltageChange), forwarded to BOTH base_state_'s
        // AcPfNrState AND scaling_max_voltage_change_/max_dVa_/max_dVm_ (used
        // by the batch solver in run(), computed per batch slot -- see the
        // members' own doc). Off by default; see AcPfNrState's own doc.
        bool   scaling_max_voltage_change = false,
        double max_dVa = 0.5,
        double max_dVm = 0.1
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
    // set_gen_v
    //   Per-scenario generator target voltage magnitude (vm_pu, NOT kV),
    //   (n_scenarios x n_gen). Unlike set_injections(), this does NOT feed
    //   Sbus. gen_bus[g] is the AC-solver bus generator g REGULATES
    //   (InjectionElements.gen_v_bus: its regulated_bus_id, -1 when it
    //   regulates nothing). Two cases are driven:
    //     * that bus is Vm-fixed (h_is_vm_fixed_bus_: pv ∪ slack with no Vm
    //       unknown): |V| is re-seeded there right before the chunk's solve,
    //       keeping the angle -- never an NR unknown, so it stays put
    //       (lightsim2grid's modify_gen_v / GeneratorContainer::set_vm);
    //     * that bus is regulated by a VoltageControl group (a remote
    //       regulator, or a local one on a group-controlled bus): the value is
    //       that group's v_set for the row, in the bordered row
    //       |V_reg| + s.Q - v_set = 0 (and |V_reg| is re-seeded as a start).
    //   Any other column is ignored. NaN entries in gen_v leave that (row,
    //   gen) untouched. Left unset entirely (the default), every row keeps
    //   the grid's own base-case voltage and set-points.
    //   gen_bus must have one entry per generator. May be called repeatedly, and in either
    //   order relative to set_injections() -- the row count is only checked
    //   against set_injections()'s n_scenarios at the next run().
    // =========================================================================
    void set_gen_v(
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gen_v,
        Eigen::Ref<const Eigen::VectorXi> gen_bus);

    // =========================================================================
    // set_branch_data
    //   Store host copies of π-model admittances.  Must be called before
    //   compute_flows().
    // =========================================================================
    void set_branch_data(
        Eigen::Ref<const Eigen::VectorXi> branch_from,
        Eigen::Ref<const Eigen::VectorXi> branch_to,
        Eigen::Ref<const CplxVect>        yff_eff,
        Eigen::Ref<const CplxVect>        yft_eff,
        Eigen::Ref<const CplxVect>        ytf_eff,
        Eigen::Ref<const CplxVect>        ytt_eff,
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
    // set_gen_v() bus maps (see ScenarioSweepSession's accessors of the same name)
    std::vector<int>    is_vm_fixed_bus() const
    { return std::vector<int>(h_is_vm_fixed_bus_.begin(), h_is_vm_fixed_bus_.end()); }
    std::vector<int>    vc_group_of_bus() const { return h_vc_group_of_bus_; }
    std::vector<int>    vc_group_has_fixed_member() const { return h_vc_group_fixed_; }
    std::vector<double> vc_v_set() const { return h_vc_v_set_; }

    RealVect get_residuals()  const;   // (n_scenarios,)               real
    RealVect get_or_amps()    const;   // (n_scenarios * n_branches,) real
    RealVect get_ex_amps()    const;   // (n_scenarios * n_branches,) real
    BatchTimings get_timings() const { return timings_; }

    // =========================================================================
    // Post-solve PHYSICAL checks (opt-in, see contingency/physical_checks_data
    // .hpp): the per-bus reactive capability (compute_physical_violations,
    // lightsim2grid PR #206 parity) and the droop hvdc P-saturation
    // (compute_physical_violations). Flags / tolerances / capacities live on
    // physical_checks(); mutable, taken into account at the next run().
    // set_bus_q_capability() hands in the plan the reactive check needs (built
    // by lightsim2grid's own build_bus_q_plan through the bridge, or by the
    // caller in array mode). The get_* accessors are synchronous D->H copies
    // and throw unless the last run() had the corresponding flag on.
    // =========================================================================
    PhysicalChecksConfig&       physical_checks()       { return phys_; }
    const PhysicalChecksConfig& physical_checks() const { return phys_; }
    void set_bus_q_capability(const BusQPlanData& plan);
    BusQViolationsResult  get_bus_q_violations()    const;
    BusQViolationsResult  get_bus_q_violations_n()  const;
    HvdcPViolationsResult get_hvdc_p_violations()   const;
    HvdcPViolationsResult get_hvdc_p_violations_n() const;

    // Non-copyable, non-movable (owns CUDA resources via unique_ptr)
    InjectionSweepSession(const InjectionSweepSession&)            = delete;
    InjectionSweepSession& operator=(const InjectionSweepSession&) = delete;
    InjectionSweepSession(InjectionSweepSession&&)                 = delete;
    InjectionSweepSession& operator=(InjectionSweepSession&&)      = delete;
};

#endif // INJECTION_SWEEP_SESSION_HPP