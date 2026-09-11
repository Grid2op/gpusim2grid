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
#include "gen_contingency_data.hpp"          // GenContingencyData
#include "reordering_alg.hpp"
#include "matching_alg.hpp"
#include "pivot_epsilon_alg.hpp"

#include "Eigen/Core"
#include "Eigen/SparseCore"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Forward-declare CUDA-dependent types to keep CUDA headers out of this file.
struct AcPfNrState;
struct LedgerData;
struct ScenarioSweepBatch;
struct ScenarioSweepDeviceData;   // session-owned device buffers (defined in the .cu)
template <typename BatchSource> struct BatchPfDriver;
using ScenarioSweepSolver = BatchPfDriver<ScenarioSweepBatch>;

// =============================================================================
// ScenarioSweepDriverConfig — the construction-time shape of the live batch
// driver. run() compares the current settings against the snapshot taken when
// solver_ was built: any difference means the driver cannot be reused (a
// "cold" run rebuilds it, with a new cuDSS ANALYSIS); otherwise run() only
// swaps the source (new topology, "warm") or the injections ("hot"). The
// config members are plain read/write attributes on the Python side, so a
// snapshot comparison is the only robust way to notice a change.
// =============================================================================
struct ScenarioSweepDriverConfig {
    int    n_scenarios     = -1;
    int    batch_size      = 0;
    int    refactor_period = 1;
    ContingencySolverType strategy = ContingencySolverType::DirectRefactorEvery;
    ReorderingAlg   reordering_alg    = ReorderingAlg::Default;
    MatchingAlg     matching_alg      = MatchingAlg::None;
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default;
    bool   scaling_max_voltage_change = false;
    double max_dVa = 0.5, max_dVm = 0.1;
    bool   mask_mode            = false;   // handle_disconnected_grid
    bool   fixed_batch_capacity = false;
    int    base_state_generation = -1;

    bool operator==(const ScenarioSweepDriverConfig& o) const {
        return n_scenarios == o.n_scenarios && batch_size == o.batch_size
            && refactor_period == o.refactor_period && strategy == o.strategy
            && reordering_alg == o.reordering_alg && matching_alg == o.matching_alg
            && pivot_epsilon_alg == o.pivot_epsilon_alg
            && scaling_max_voltage_change == o.scaling_max_voltage_change
            && max_dVa == o.max_dVa && max_dVm == o.max_dVm
            && mask_mode == o.mask_mode && fixed_batch_capacity == o.fixed_batch_capacity
            && base_state_generation == o.base_state_generation;
    }
    bool operator!=(const ScenarioSweepDriverConfig& o) const { return !(*this == o); }
};

struct ScenarioSweepSession {

    // Destructor defined in the .cu where AcPfNrState/ScenarioSweepSolver are
    // complete types.
    ~ScenarioSweepSession();

    // =========================================================================
    // Owned GPU state
    // =========================================================================
    std::unique_ptr<AcPfNrState>          base_state_;
    std::unique_ptr<ScenarioSweepSolver>  solver_;   // null until the first run(); then PERSISTENT
    std::unique_ptr<ScenarioSweepDeviceData> dev_;   // canonical original-order Sbus / gen_v device buffers

    // =========================================================================
    // Driver persistence (see ScenarioSweepDriverConfig and run()).
    //
    //   *_dirty_            : which inputs changed since the last run().
    //   injections_on_device_ / gen_v_on_device_ : the canonical buffer was
    //                         last filled straight from a device tensor
    //                         (set_injections_dlpack / set_gen_v_dlpack), so
    //                         run() must not overwrite it from the host copies.
    //   fixed_batch_capacity_ : when true, batch_size_ is used verbatim as the
    //                         driver's chunk capacity (no rebalancing over the
    //                         active count), so with batch_size_ >= n_scenarios
    //                         the whole batch is always ONE chunk whatever
    //                         rows get islanded -- what the differentiable
    //                         wrapper needs (the adjoint reads the last chunk's
    //                         Jacobian). Default false keeps the historic
    //                         rebalancing for the plain sweep API.
    //   keep_final_jacobian_ : refill J at the converged V after the NR loop
    //                         (forwarded to the driver; see BatchPfDriver).
    //   run_counter_ etc.   : observability for callers/tests (a torch
    //                         autograd backward checks run_counter_ against
    //                         the forward it belongs to).
    // =========================================================================
    bool injections_dirty_     = false;
    bool topology_dirty_       = false;
    bool gen_v_dirty_          = false;
    bool gen_off_dirty_        = false;
    bool injections_on_device_ = false;
    bool gen_v_on_device_      = false;
    bool fixed_batch_capacity_ = false;
    bool keep_final_jacobian_  = false;
    bool last_run_kept_jacobian_ = false;
    int  run_counter_          = 0;
    int  driver_build_counter_ = 0;
    int  source_build_counter_ = 0;
    int  base_state_generation_ = 0;
    ScenarioSweepDriverConfig driver_cfg_;

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
    CplxVect        h_yff_eff_, h_yft_eff_, h_ytf_eff_, h_ytt_eff_;
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
    // Vm-fixed bus mask (pv ∪ slack_ids, built once at construction) --
    // consulted by set_gen_v() below. See that method's own doc.
    // =========================================================================
    std::vector<char> h_is_vm_fixed_bus_;

    // =========================================================================
    // Host generator target-voltage override (set_gen_v()) -- optional; see
    // that method's own doc. Empty (has_gen_v_ == false) means every row
    // keeps using the grid's own base-case voltage, exactly as before this
    // setter existed.
    // =========================================================================
    Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> gen_v_;
    Eigen::VectorXi gen_bus_;
    bool   has_gen_v_      = false;

    // =========================================================================
    // Host topology data (set_topology()) — one Contingency (and tripped-
    // branch list, for compute_flows' zero-flow step) per scenario, row-
    // aligned with p_mw_/q_mvar_. has_topology_ == false means "run()
    // defaults every scenario to no trips" (pure injection sweep).
    // =========================================================================
    std::vector<Contingency>      contingencies_;
    std::vector<std::vector<int>> tripped_branches_per_scenario_;
    bool has_topology_ = false;

    // =========================================================================
    // Generator contingencies (set_contingency_gens(), lightsim2grid PR #193
    // parity). gen_data_ is the per-generator snapshot the bridge read off the
    // grid (array/tuple mode has none → set_contingency_gens raises).
    // gen_off_ is the (n_scenarios × n_gen) mask, ORIGINAL row order.
    //
    // The buses that lose their LAST local voltage controller in SOME row need
    // a reserved Vm column + Q equation in the shared Jacobian (ledger_extend
    // .hpp's add_switchable_vm_buses); reserved_buses_ is the sorted set the
    // CURRENT base_state_ was built with. run() derives the set the mask needs
    // and rebuilds base_state_ (one base-case setup + cuDSS analysis) whenever
    // the two differ -- growing or shrinking, so an all-False mask is
    // bit-identical to no mask at all. The stored ctor inputs below exist for
    // exactly that rebuild; base_ledger_ is the UNEXTENDED ledger copy.
    // =========================================================================
    using BoolMat = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    GenContingencyData gen_data_;
    bool               has_gen_data_ = false;
    BoolMat            gen_off_;
    bool               has_gen_off_  = false;
    std::vector<int>   reserved_buses_;

    Eigen::SparseMatrix<eigen_cplx_type> Ybus_cm_;
    CplxVect        Vinit_, Sbus_;
    Eigen::VectorXi slack_ids_, pv_, pq_;
    int    max_iter_base_   = 10;
    double tol_base_        = 1e-6;
    int    device_          = -1;
    bool   presolved_v_     = false;
    bool   debug_base_case_ = false;
    std::unique_ptr<LedgerData> base_ledger_;   // null in array/tuple mode

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
        Eigen::Ref<const CplxVect>        yff_eff,
        Eigen::Ref<const CplxVect>        yft_eff,
        Eigen::Ref<const CplxVect>        ytf_eff,
        Eigen::Ref<const CplxVect>        ytt_eff,
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
    // set_injections_device — the device path of set_injections(): d_ptr is a
    // (n_scen × n_bus) row-major PER-UNIT complex buffer (the build's
    // cudaComplexType) on this session's device, e.g. a torch tensor handed
    // over through DLPack (see dlpack_export.cu). One D2D copy into the
    // canonical original-order buffer; producer_stream (a cudaStream_t
    // handle, 0 = none) is waited on through an event first, and the copy is
    // host-synchronized before returning so the caller may free/reuse the
    // source immediately. Fixes n_scenarios().
    // =========================================================================
    void set_injections_device(const void* d_ptr, int n_scen, int n_bus,
                               std::uintptr_t producer_stream);

    // =========================================================================
    // set_topology — one branch-id list per scenario (lines-then-trafos),
    // row-aligned with set_injections(). Requires set_branch_data() first.
    // Optional: if never called, run() defaults every scenario to "no
    // branches tripped".
    // =========================================================================
    void set_topology(const std::vector<std::vector<int>>& branch_ids_per_scenario);

    // =========================================================================
    // set_gen_v -- see InjectionSweepSession::set_gen_v's doc (identical
    // semantics: NOT fed into Sbus, only re-seeds |V| at each generator's own
    // Vm-fixed bus right before that scenario's solve; a disconnected,
    // reactive-only, or remotely voltage-regulating generator's column is
    // silently ignored). May be called repeatedly, and in either order
    // relative to set_injections().
    // =========================================================================
    void set_gen_v(
        Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gen_v,
        Eigen::Ref<const Eigen::VectorXi> gen_bus);

    // Device path of set_gen_v(): d_ptr is a (n_scen × n_gen) row-major real
    // buffer (the build's cuda_real_type) on this device; the Vm-fixed column
    // filter is derived from gen_bus on the host (cheap, O(n_gen)) and the
    // selected columns are gathered on the device at run(). Same stream /
    // sync contract as set_injections_device.
    void set_gen_v_device(const void* d_ptr, int n_scen, int n_gen,
                          Eigen::Ref<const Eigen::VectorXi> gen_bus,
                          std::uintptr_t producer_stream);

    // Drop any gen_v override: every row keeps the grid's base-case voltage
    // again (the state before set_gen_v was ever called).
    void clear_gen_v();

    // =========================================================================
    // set_gen_contingency_data — per-generator snapshot (bridge factory only;
    // see gen_contingency_data.hpp). Enables set_contingency_gens().
    // =========================================================================
    void set_gen_contingency_data(const GenContingencyData& data);

    // =========================================================================
    // set_contingency_gens — (n_scenarios × n_gen) bool mask, row-aligned with
    // set_injections()/set_topology(): True disconnects that generator for
    // that row. Mirrors lightsim2grid's ScenarioSweep::set_contingency_gens:
    //   • the injection side (its P, and its target Q when it does not
    //     regulate voltage, leaving Sbus) is the CALLER's job (the Python
    //     facade does it in set_injections_from_elements);
    //   • the labelling side is done here: when the LAST generator locally
    //     regulating a bus is off, that bus turns PQ for the row (its
    //     reserved Q row is released; still-PV rows identity-pin it);
    //   • the distributed slack is re-weighted per row without the
    //     disconnected participants (the slack BUS SET never changes).
    // Refuses a generator regulating a remote bus or standing on a bus a
    // control group holds (VoltageControl owns those rows/columns). Only
    // stores the mask; the structure is derived (and the base state rebuilt
    // if needed) by the next run().
    // =========================================================================
    void set_contingency_gens(Eigen::Ref<const BoolMat> mask);

    // Current augmented Jacobian dimension of the base state (grows by one per
    // reserved switchable bus) and the reserved buses themselves -- lets a
    // caller observe when run() rebuilt the base state.
    int              dim_J() const;
    std::vector<int> get_reserved_buses() const { return reserved_buses_; }
    bool             has_gen_contingency() const { return has_gen_off_; }

    // =========================================================================
    // run — solves every scenario. A scenario whose topology change
    // disconnects the grid is skipped (NaN residual/voltage) — unless
    // handle_disconnected_grid_ is set, in which case only scenarios stranding
    // the angle reference or a controller bus are left as NaN (the rest solve
    // on their largest connected component, masked buses reported as NaN).
    //
    // Three paths, decided against the live driver (see
    // ScenarioSweepDriverConfig):
    //   cold : no driver yet, or its shape/config changed (n_scenarios,
    //          batch_size, strategy, cuDSS config, base state, ...) → build a
    //          new BatchPfDriver<ScenarioSweepBatch> (allocation + cuDSS
    //          ANALYSIS + first FACTORIZATION on the first iteration).
    //   warm : only the topology / generator mask changed → new
    //          ScenarioSweepBatch (CPU connectivity + patches) swapped into
    //          the live driver; no analysis, REFACTORIZATION only.
    //   hot  : only injections / gen_v changed → one device gather of the new
    //          rows; nothing else touched.
    // The result buffers (v_results_dlpack) are then overwritten IN PLACE
    // across runs (they only move on a cold rebuild).
    // =========================================================================
    void run();

    // =========================================================================
    // solve_JT_batch — batched adjoint (see BatchPfDriver::solve_JT_batch;
    // pointer arguments are device buffers of the documented shapes, nullptr
    // where optional). Requires the last run() to have been made with
    // keep_final_jacobian_ = true (alias mode) or an external J snapshot.
    // Builds the transposed system lazily on the first call.
    // =========================================================================
    void solve_JT_batch(const void* d_rhs_orig, const void* d_J_ext,
                        bool want_gen_v_grad,
                        const void* d_Ybus_ext, const void* d_V_ext_orig,
                        std::uintptr_t producer_stream);

    // Adjoint / structure accessors (see the bindings for the shapes).
    bool adjoint_ready() const;
    int  capacity()      const;   // live driver's chunk capacity (0 before run())
    int  n_active()      const;   // rows actually solved by the last run()
    int  nnz_J()         const;
    int  nnz_Y()         const;
    std::vector<int> p_row_of_bus()     const;
    std::vector<int> q_row_of_bus()     const;
    std::vector<int> theta_col_of_bus() const;
    std::vector<int> vm_col_of_bus()    const;
    std::vector<int> is_vm_fixed_bus()  const;
    Eigen::VectorXi  get_active_to_orig() const;
    std::pair<std::vector<int>, std::vector<int>> j_skeleton() const;   // (outer, inner)

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
    BatchTimings get_timings() const;  // run() timings + cumulative adjoint counters

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

private:
    // (Re)build base_state_ + mask_cfg_ on the base ledger extended with the
    // given switchable buses (empty: the un-extended ledger). Drops solver_
    // first (it references *base_state_).
    void _build_base_state(const std::vector<int>& switchable_buses);

    // Snapshot of the settings the live driver depends on (see
    // ScenarioSweepDriverConfig).
    ScenarioSweepDriverConfig _current_config() const;

    // Make an external stream's pending work visible to `cs` (event record +
    // wait); 0 = nothing to wait for.
    static void _wait_producer(std::uintptr_t producer_stream, void* cs);

    // Derive, from gen_off_, the buses that lose every local controller per
    // row (row_pv_to_pq), the union of those (required, sorted, restricted to
    // buses that need a reserved Vm/Q pair), and the slack participants each
    // row disconnects (row_slack_off). All empty when no mask is set.
    void _prepare_gen_contingency(std::vector<int>&              required,
                                  std::vector<std::vector<int>>& row_pv_to_pq,
                                  std::vector<std::vector<int>>& row_slack_off) const;

    // Per-row distributed-slack weights, [n_scenarios * n_slack] in the base
    // state's participant order; empty when no row re-weights the slack.
    std::vector<cuda_real_type> _row_slack_weights(
        const std::vector<std::vector<int>>& row_slack_off) const;
};

#endif // SCENARIO_SWEEP_SESSION_HPP
