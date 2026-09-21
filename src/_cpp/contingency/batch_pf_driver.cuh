// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BATCH_PF_DRIVER_CUH
#define BATCH_PF_DRIVER_CUH

// =============================================================================
// contingency/batch_pf_driver.cuh
//
// Templated batched-PF chunk driver — shared by N-k contingency analysis and
// the injection sweep.  Parameterised by a small NON-VIRTUAL BatchSource
// policy that owns the per-chunk perturbation:
//
//   ContingencyBatch : tile V + tile Ybus + apply per-element patches.
//                      Sbus is the shared base-case (sbus_stride = 0).
//   InjectionBatch   : tile V only (Ybus tiled once at construction).
//                      Sbus is per-scenario (sbus_stride = n_bus).
//
// All other state (V_batch / Ybus_batch / J / dx / F buffers, cuDSS context,
// SpMV descriptor, linear-solve policy std::variant, residual + branch-flow
// outputs, dimensions, timings) is shared and lives on this class.
//
// The constructor performs:
//   1.  source.cpu_preprocess_ms()  → recorded into t_preprocess_ms_
//   2.  block-diagonal CSR structure upload (outer/inner only)
//   3.  chunk-sized device buffer allocation
//   4.  full-result buffer allocation (V_results + residuals)
//   5.  cuSPARSE block-diagonal SpMV descriptor
//   6.  CudssBatchSolver::initialize  (cuDSS ANALYSIS — once)
//   7.  std::visit policy.initialize_from_base
//   8.  source.initialize(ctx, cs)   — source-specific one-time work
//
// Per-chunk loop (in _solve_chunk):
//   a.  source.prepare_Ybus_batch(ctx, chunk_idx, actual_batch, …)
//   b.  source.prepare_Sbus_batch(ctx, chunk_idx, actual_batch, …)
//   c.  NrIterBuffers built with source.d_Sbus_ptr() + source.sbus_stride()
//   d.  std::visit policy → run_nr_loop<Policy>
//   e.  final SpMV + fill_F + per-element ‖F‖∞ + store-V (+ optional flows)
//
// Member declaration order is the same as the historic
// ContingencyAnalysisSolver: cs FIRST so it is destroyed LAST.
// =============================================================================

#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "../cuda_utils.h"
#include "../cu_complex_utils.h"
#include "../timing_utils.hpp"
#include "../acpf_nr_state.cuh"
#include "../contingency_analysis_helper.hpp"   // ContingencySolverType
#include "bus_q_check_data.hpp"                   // BusQPlanData
#include "gen_p_check_data.hpp"                   // GenPPlanData
#include "strategies/cudss_batch_solver.cuh"
#include "strategies/policy_refactor_every.cuh"
#include "strategies/policy_base_case_factors.cuh"
#include "strategies/policy_iter0_only.cuh"
#include "strategies/policy_refactor_every_n.cuh"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "Eigen/Core"
#include "Eigen/SparseCore"

// -----------------------------------------------------------------------------
// BatchPfDriverContext — non-owning per-chunk view passed to the BatchSource
// hooks.  Plain struct (no templates) to keep BatchSource types decoupled from
// BatchPfDriver's template parameter.
// -----------------------------------------------------------------------------
struct BatchPfDriverContext {
    AcPfNrState&     base;
    cudaComplexType* d_V_batch;             // [batch_size × n_bus]
    cudaComplexType* d_Ybus_values_batch;   // [batch_size × nnz_Y]
    int              batch_size;
    int              n_bus;
    int              nnz_Y;
};

// -----------------------------------------------------------------------------
// BatchAdjoint — the batched transposed system Jᵀ λ = x̄ (one per batch slot),
// built LAZILY by BatchPfDriver::solve_JT_batch on its first call and reused
// by every later call (requirements: nothing exists until the first backward;
// later backward calls only permute values, REFACTORIZE and SOLVE).
//
// cuDSS (0.8) has no transposed-solve mode, so Jᵀ is an explicit second
// uniform-batch system over the SAME capacity: its CSR skeleton is the
// transpose of the shared J skeleton (host counting sort, once), and its
// values are J's values permuted through d_J_to_JT (JT_values[map[i]] =
// J_values[i]) by one kernel per call. Everything is sized by the driver's
// fixed capacity (batch_size_), so it survives replace_source() and dies with
// the driver.
// -----------------------------------------------------------------------------
struct BatchAdjoint {
    CudssBatchSolver solver;                       // own cuDSS context + ANALYSIS

    thrust::device_vector<int>            d_JT_outer;    // [dim_J + 1]
    thrust::device_vector<int>            d_JT_inner;    // [nnz_J]
    thrust::device_vector<int>            d_J_to_JT;     // [nnz_J] position map
    thrust::device_vector<cuda_real_type> d_JT_values;   // [capacity × nnz_J]
    thrust::device_vector<cuda_real_type> d_rhs;         // [capacity × dim_J], slot order
    thrust::device_vector<cuda_real_type> d_sol;         // [capacity × dim_J], slot order
    thrust::device_vector<cuda_real_type> d_sol_full;    // [n_contingencies × dim_J], original order

    // gen_v adjoint (dS/dVm column contraction at Vm-fixed buses), built on
    // the first call that asks for it.
    bool                                   gen_v_ready = false;
    thrust::device_vector<int>             d_Ybus_T_pos;      // [nnz_Y] position of (j,i) for entry (i,j)
    thrust::device_vector<int>             d_p_row_of_bus;    // [n_bus]
    thrust::device_vector<int>             d_q_row_of_bus;    // [n_bus]
    thrust::device_vector<char>            d_is_vm_fixed_bus; // [n_bus]
    thrust::device_vector<cudaComplexType> d_V_ext_slots;     // [capacity × n_bus] snapshot-mode scratch
    thrust::device_vector<cuda_real_type>  d_gvm;             // [capacity × n_bus], slot order
    thrust::device_vector<cuda_real_type>  d_gvm_full;        // [n_contingencies × n_bus], original order

    bool factorized           = false;
    int  factorized_for_solve = -1;   // driver n_solves_ whose own J was last permuted+refactored

    // Counters / timings (cumulative over the driver's life; surfaced through
    // BatchTimings by the sessions).
    int    n_analysis = 0, n_factorize = 0, n_refactorize = 0, n_solve = 0;
    double t_build_ms = 0.;
    TimingEntry t_first_factorize, t_refactorize, t_solve;
};

// =============================================================================
// BatchPfDriver<BatchSource>
// =============================================================================
template <typename BatchSource>
struct BatchPfDriver {

    // -------------------------------------------------------------------------
    // CUDA stream  (declared FIRST — destroyed LAST)
    // -------------------------------------------------------------------------
    CudaStream cs;

    // -------------------------------------------------------------------------
    // Reference to base-case state (not owned; must outlive this object)
    // -------------------------------------------------------------------------
    AcPfNrState& base;

    // -------------------------------------------------------------------------
    // Per-element batch source (owned by value — moved in at construction)
    // -------------------------------------------------------------------------
    BatchSource source_;

    // -------------------------------------------------------------------------
    // Dimensions
    //
    // n_contingencies is the count of batch elements; kept under this name
    // for backwards compatibility with the existing public surface
    // (ContingencyAnalysisSession reads it; BatchTimings exposes it).
    // For the injection sweep it is n_scenarios.
    // -------------------------------------------------------------------------
    int n_contingencies = 0;   // size of the result buffers (full set)
    int n_active_       = 0;    // systems actually solved (disconnected dropped)
    int batch_size_     = 0;
    int nb_iter_        = 0;
    int n_chunks_       = 0;    // ceil(n_active_ / batch_size_)

    // -------------------------------------------------------------------------
    // One-time construction timings (wall-clock ms)
    //
    //   t_analysis_ms_    — cuDSS ANALYSIS + policy init ONLY.
    //   t_context_init_ms_ — cuDSS handle/config/data creation, split out of
    //                       t_analysis_ms_: on the first cuDSS use in a process
    //                       this is dlopen + JIT of the backend, size-
    //                       independent one-time cost, not analysis work.
    //   t_source_init_ms_ — source-specific one-time GPU setup (split out of
    //                       what used to be bundled into t_analysis_ms_): see
    //                       BatchSource::initialize().
    // -------------------------------------------------------------------------
    double t_preprocess_ms_       = 0.;
    double t_alloc_ms_            = 0.;
    double t_analysis_ms_         = 0.;
    double t_context_init_ms_     = 0.;
    double t_source_init_ms_      = 0.;
    double t_branch_data_upload_ms_ = 0.;   // set by set_branch_data()
    double t_violation_setup_ms_    = 0.;   // set by set_violation_limits()

    // -------------------------------------------------------------------------
    // Block-diagonal Ybus structure (outer/inner only — values tiled per chunk)
    // -------------------------------------------------------------------------
    thrust::device_vector<int> d_Ybus_batch_outer;
    thrust::device_vector<int> d_Ybus_batch_inner;

    // -------------------------------------------------------------------------
    // Chunk-sized working buffers
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V_batch;
    thrust::device_vector<cudaComplexType> d_Ybus_values_batch;
    thrust::device_vector<cudaComplexType> d_Ibus_batch;
    thrust::device_vector<cuda_real_type>  d_F_batch;
    thrust::device_vector<cuda_real_type>  d_dx_batch;
    thrust::device_vector<cuda_real_type>  d_J_values_batch;

    // Per-batch-slot augmented-feature running state (empty when the feature is
    // inactive). The shared single-system feature data (slack weights, HVDC/VC
    // arrays, feature positions) lives on `base` and is pointed to directly.
    thrust::device_vector<cuda_real_type>  d_slack_absorbed_batch;  // [batch_size]
    thrust::device_vector<cuda_real_type>  d_vc_q_batch;            // [batch_size * n_vc_ctrl]

    // NR step-scaling (MaxVoltageChange) config + per-batch-slot scratch
    // (empty unless scaling_max_voltage_change_ is on). Each slot gets its own
    // alpha from its own max|dtheta|/max|dvm| -- see run_nr_loop's own doc.
    bool                                    scaling_max_voltage_change_ = false;
    cuda_real_type                          max_dVa_ = static_cast<cuda_real_type>(0.5);
    cuda_real_type                          max_dVm_ = static_cast<cuda_real_type>(0.1);
    thrust::device_vector<cuda_real_type>   d_scale_max_dtheta_batch;  // [batch_size]
    thrust::device_vector<cuda_real_type>   d_scale_max_dvm_batch;     // [batch_size]

    // -------------------------------------------------------------------------
    // Full result buffers
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V_results;
    thrust::device_vector<cuda_real_type>  d_residuals;

    // -------------------------------------------------------------------------
    // Optional branch-flow outputs (only meaningful for contingency-style
    // features that mutate topology).  Both InjectionBatch and ContingencyBatch
    // can co-exist with this: InjectionBatch simply never calls set_branch_data.
    //
    // _has_branch_admittances : the O(n_branches) admittance arrays below
    //                           (+ d_bus_vn_kv) are uploaded and ready. Set by
    //                           upload_branch_admittances(); enough for the
    //                           fused compute_limit_violations kernel.
    // _has_branch_data        : ALSO the dense O(n_contingencies*n_branches)
    //                           d_or_amps_results/d_ex_amps_results are
    //                           allocated (set_branch_data() only). Gates step
    //                           ⑥'s full-batch flow kernel in _solve_chunk —
    //                           unchanged meaning from before this split.
    // -------------------------------------------------------------------------
    bool _has_branch_admittances = false;
    bool _has_branch_data        = false;
    int  n_branches_      = 0;

    thrust::device_vector<int>             d_branch_from;
    thrust::device_vector<int>             d_branch_to;
    thrust::device_vector<cudaComplexType> d_yff_eff;
    thrust::device_vector<cudaComplexType> d_yft_eff;
    thrust::device_vector<cudaComplexType> d_ytf_eff;
    thrust::device_vector<cudaComplexType> d_ytt_eff;
    thrust::device_vector<cuda_real_type>  d_base_current_A;
    thrust::device_vector<cuda_real_type>  d_bus_vn_kv;   // [n_bus], per-bus nominal kV

    thrust::device_vector<cuda_real_type>  d_or_amps_results;
    thrust::device_vector<cuda_real_type>  d_ex_amps_results;

    // -------------------------------------------------------------------------
    // compute_limit_violations (opt-in; fused per-chunk voltage/current check).
    // All sized O(n_contingencies) or O(n_contingencies * K) — bounded, never
    // O(n_contingencies * n_branches) or O(n_contingencies * n_bus). Allocated
    // only by set_violation_limits(); untouched (empty) otherwise.
    // -------------------------------------------------------------------------
    bool           _fused_violations_enabled = false;
    int            violation_capacity_       = 0;   // K
    cuda_real_type violation_tol_            = 0;
    int            n_lines_                  = 0;   // branch ordering split (lines-then-trafos)

    thrust::device_vector<cuda_real_type> d_bus_vmin_kv;         // [n_bus]
    thrust::device_vector<cuda_real_type> d_bus_vmax_kv;         // [n_bus]
    thrust::device_vector<cuda_real_type> d_branch_limit_a1_ka;  // [n_branches]
    thrust::device_vector<cuda_real_type> d_branch_limit_a2_ka;  // [n_branches]

    // Compact per-contingency output, size n_contingencies * K (SoA, matching
    // this file's convention of separate typed device vectors over a packed
    // struct array).
    thrust::device_vector<int>            d_viol_element_type;
    thrust::device_vector<int>            d_viol_element_id;
    thrust::device_vector<int>            d_viol_side;
    thrust::device_vector<int>            d_viol_type;
    thrust::device_vector<cuda_real_type> d_viol_value;
    thrust::device_vector<cuda_real_type> d_viol_limit;
    thrust::device_vector<int>            d_violation_count;      // [n_contingencies]; -1 = not simulated, else 0..K
    thrust::device_vector<int>            d_violation_truncated;  // [n_contingencies]; 0/1

    // TRUE, uncapped per-type violation totals -- [n_contingencies]; -1 = not
    // simulated, else the exact count (independent of violation_capacity/K,
    // unlike d_violation_count above which is capped at K).
    thrust::device_vector<int>            d_violation_count_low_voltage;
    thrust::device_vector<int>            d_violation_count_high_voltage;
    thrust::device_vector<int>            d_violation_count_current;

    // -------------------------------------------------------------------------
    // compute_physical_violations (opt-in; per-bus reactive-capability check,
    // lightsim2grid PR #206 parity -- see bus_q_check_data.hpp and
    // check_bus_q_violations_kernel). Plan arrays are O(n_check), outputs
    // O(n_contingencies * K_q). Allocated only by set_bus_q_check().
    // -------------------------------------------------------------------------
    bool           _bus_q_enabled     = false;
    int            bus_q_n_check_     = 0;
    int            bus_q_capacity_    = 0;   // K_q
    int            bus_q_n_gen_       = 0;   // columns of d_bq_gen_off_ (0 = none)
    cuda_real_type bus_q_tol_mvar_    = 0;
    cuda_real_type bus_q_sn_mva_      = 0;
    cuda_real_type bus_q_residual_tol_ = 0;  // row gate (a non-converged row reports nothing)
    // Non-owning: the session owns the (n_contingencies x n_gen) uint8 mask,
    // ORIGINAL row order (ScenarioSweep generator contingencies); nullptr = none.
    const unsigned char* d_bq_gen_off_ = nullptr;
    double         t_bus_q_setup_ms_  = 0.;

    thrust::device_vector<int>            d_bq_bus_solver, d_bq_n_fixed, d_bq_gen_start, d_bq_gen_id;
    thrust::device_vector<cuda_real_type> d_bq_qmin_fixed, d_bq_qmax_fixed, d_bq_bmin_sum, d_bq_bmax_sum,
                                          d_bq_gen_qmin, d_bq_gen_qmax;
    thrust::device_vector<int>            d_bq_out_bus_id, d_bq_out_type;     // [n_contingencies * K_q]
    thrust::device_vector<cuda_real_type> d_bq_out_value, d_bq_out_limit;
    thrust::device_vector<int>            d_bq_count;        // [n_contingencies]; -1 = never simulated, else 0..K_q
    thrust::device_vector<int>            d_bq_truncated;    // [n_contingencies]; 0/1
    // the base ("n") case's own report (run_bus_q_check_n): one slot
    thrust::device_vector<int>            d_bq_n_bus_id, d_bq_n_type;         // [K_q]
    thrust::device_vector<cuda_real_type> d_bq_n_value, d_bq_n_limit;
    thrust::device_vector<int>            d_bq_n_count, d_bq_n_truncated;     // [1]

    // -------------------------------------------------------------------------
    // compute_physical_violations (opt-in; droop P-saturation check, see
    // check_hvdc_p_violations_kernel). The per-line data is the base state's
    // own HVDC arrays; only the outputs live here, O(n_contingencies * K_p).
    // -------------------------------------------------------------------------
    bool           _hvdc_p_enabled      = false;
    int            hvdc_p_capacity_     = 0;   // K_p
    cuda_real_type hvdc_p_tol_pu_       = 0;   // tol_mw / sn_mva
    cuda_real_type hvdc_p_sn_mva_       = 0;
    cuda_real_type hvdc_p_residual_tol_ = 0;
    double         t_hvdc_p_setup_ms_   = 0.;

    thrust::device_vector<int>            d_hp_out_hvdc_id, d_hp_out_side;    // [n_contingencies * K_p]
    thrust::device_vector<cuda_real_type> d_hp_out_value, d_hp_out_limit;
    thrust::device_vector<int>            d_hp_count;        // [n_contingencies]; -1 = never simulated, else 0..K_p
    thrust::device_vector<int>            d_hp_truncated;    // [n_contingencies]; 0/1
    thrust::device_vector<int>            d_hp_n_hvdc_id, d_hp_n_side;        // [K_p]
    thrust::device_vector<cuda_real_type> d_hp_n_value, d_hp_n_limit;
    thrust::device_vector<int>            d_hp_n_count, d_hp_n_truncated;     // [1]

    // -------------------------------------------------------------------------
    // compute_physical_violations (opt-in; per-machine active power of the
    // distributed slack, lightsim2grid's GenPCheck.hpp parity -- see
    // gen_p_check_data.hpp and check_gen_p_violations_kernel). The plan's
    // participants are regrouped per bus at upload (CSR over the distinct
    // participating buses, d_gp_part_*; d_gp_bus_slot maps an entry to its
    // bus' group, -1 when no participant stands there). Outputs
    // O(n_contingencies * K_g). Allocated only by set_gen_p_check().
    // -------------------------------------------------------------------------
    bool           _gen_p_enabled       = false;
    int            gen_p_n_entries_     = 0;
    int            gen_p_n_part_bus_    = 0;
    int            gen_p_capacity_      = 0;   // K_g
    int            gen_p_n_gen_         = 0;   // columns of d_gp_gen_off_ (0 = none)
    int            gen_p_target_stride_ = 0;   // columns of d_gp_targets (0 = base targets)
    cuda_real_type gen_p_tol_mw_        = 0;
    cuda_real_type gen_p_sn_mva_        = 0;
    cuda_real_type gen_p_residual_tol_  = 0;
    const unsigned char* d_gp_gen_off_  = nullptr;   // non-owning, like d_bq_gen_off_
    double         t_gen_p_setup_ms_    = 0.;

    thrust::device_vector<int>            d_gp_el_type, d_gp_el_id, d_gp_bus_solver, d_gp_bus_slot;
    thrust::device_vector<cuda_real_type> d_gp_weight, d_gp_min_p, d_gp_max_p, d_gp_target_base;
    thrust::device_vector<int>            d_gp_part_bus, d_gp_part_start, d_gp_part_el_type, d_gp_part_el_id;
    thrust::device_vector<cuda_real_type> d_gp_part_weight;
    // per-row set-points of the plan's entries (upload_gen_p_targets), ORIGINAL
    // row order, [n_contingencies * n_entries]; empty = base targets everywhere
    thrust::device_vector<cuda_real_type> d_gp_targets;
    thrust::device_vector<int>            d_gp_out_element_type, d_gp_out_element_id, d_gp_out_type;  // [n_contingencies * K_g]
    thrust::device_vector<cuda_real_type> d_gp_out_value, d_gp_out_limit;
    thrust::device_vector<int>            d_gp_count;        // [n_contingencies]; -1 = never simulated, else 0..K_g
    thrust::device_vector<int>            d_gp_truncated;    // [n_contingencies]; 0/1
    thrust::device_vector<int>            d_gp_n_element_type, d_gp_n_element_id, d_gp_n_type;  // [K_g]
    thrust::device_vector<cuda_real_type> d_gp_n_value, d_gp_n_limit;
    thrust::device_vector<int>            d_gp_n_count, d_gp_n_truncated;     // [1]

    // -------------------------------------------------------------------------
    // cuSPARSE batched SpMV
    // -------------------------------------------------------------------------
    CuSpMV spmv_batch;

    // -------------------------------------------------------------------------
    // Linear solver + NR policy variant
    // -------------------------------------------------------------------------
    CudssBatchSolver linear_solver_;

    std::variant<PolicyRefactorEvery,
                 PolicyBaseCaseFactors,
                 PolicyIter0Only,
                 PolicyRefactorEveryN> policy_;

    // cuDSS analysis config, retained so the lazily built adjoint context
    // (BatchAdjoint) analyses Jᵀ with the same choices as the forward.
    ReorderingAlg   reordering_alg_    = ReorderingAlg::Default;
    MatchingAlg     matching_alg_      = MatchingAlg::None;
    PivotEpsilonAlg pivot_epsilon_alg_ = PivotEpsilonAlg::Default;

    // -------------------------------------------------------------------------
    // Persistence / adjoint state
    //
    //   keep_final_jacobian_ : after the NR loop of a chunk, refill
    //                          d_J_values_batch at the CONVERGED V (the loop
    //                          leaves J(V_{nb_iter-1}) behind). Only one chunk
    //                          may be solved then (only the last chunk's J
    //                          survives in the chunk buffer) -- solve() throws
    //                          otherwise. Set by the differentiable wrapper.
    //   n_solves_            : solve() calls on this driver; the adjoint keys
    //                          its "J already permuted + refactored" cache on it.
    //   adjoint_             : null until the first solve_JT_batch().
    // -------------------------------------------------------------------------
    bool keep_final_jacobian_ = false;
    int  n_solves_            = 0;
    std::unique_ptr<BatchAdjoint> adjoint_;

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------
    //   base_state       — must outlive this object; cuDSS ANALYSIS already done.
    //   source           — moved into source_; its initialize(ctx, cs) is invoked
    //                      once construction reaches the GPU-setup phase.
    //   n_elements       — total contingencies / scenarios in this run.
    //   batch_size       — systems per chunk.
    //   nb_iter          — fixed NR iterations per chunk.
    //   strategy_type    — selects which Policy alternative emplaces into policy_.
    //   refactor_period  — for DirectRefactorEveryN.
    //   reordering_alg   — CUDSS_CONFIG_REORDERING_ALG for the batch ANALYSIS.
    //   matching_alg     — CUDSS_CONFIG_MATCHING_ALG for the batch ANALYSIS.
    //   pivot_epsilon_alg — CUDSS_CONFIG_PIVOT_EPSILON_ALG for the batch ANALYSIS.
    // -------------------------------------------------------------------------
    BatchPfDriver(
        AcPfNrState&              base_state,
        BatchSource               source,
        int                       n_contingencies_in,    // count of batch elements
        int                       batch_size,
        int                       nb_iter,
        ContingencySolverType     strategy_type   = ContingencySolverType::DirectRefactorEvery,
        int                       refactor_period = 1,
        ReorderingAlg             reordering_alg  = ReorderingAlg::Default,
        MatchingAlg               matching_alg    = MatchingAlg::None,
        PivotEpsilonAlg           pivot_epsilon_alg = PivotEpsilonAlg::Default,
        // NR step-scaling (MaxVoltageChange) -- see the members' own doc. Off
        // by default; see AcPfNrState's own doc for what this fixes.
        bool                      scaling_max_voltage_change = false,
        double                    max_dVa = 0.5,
        double                    max_dVm = 0.1);

    ~BatchPfDriver() = default;

    // -------------------------------------------------------------------------
    // solve()  — iterate chunks; fills d_V_results + d_residuals.
    // -------------------------------------------------------------------------
    BatchTimings solve();

    // -------------------------------------------------------------------------
    // replace_source — swap in a NEW BatchSource on a LIVE driver (the "warm"
    // path of ScenarioSweepSession::run(): new topology, same shape). Keeps
    // the cuDSS context + ANALYSIS, the chunk buffers, the SpMV descriptor and
    // the policy state (a policy that already factorized refactorizes next),
    // re-running only the source's own H→D setup. The new source must have
    // been built for this driver's capacity (used_batch_size() == batch_size_)
    // so its chunk ranges line up with the chunk loop; its active count may
    // shrink or grow within n_contingencies (phantom padding handles a short
    // last chunk). A member template so the explicit class instantiations of
    // sources that are not move-assignable do not instantiate it.
    // -------------------------------------------------------------------------
    template <typename S = BatchSource>
    void replace_source(S&& src)
    {
        cs.synchronize();
        if (src.used_batch_size() != batch_size_)
            throw std::runtime_error(
                "[batch_pf] replace_source: the new source was built for a chunk "
                "size of " + std::to_string(src.used_batch_size()) + " but this "
                "driver's capacity is " + std::to_string(batch_size_));
        if (src.n_active() > n_contingencies)
            throw std::runtime_error(
                "[batch_pf] replace_source: more active elements than result slots");
        source_   = std::move(src);
        n_active_ = source_.n_active();
        n_chunks_ = (n_active_ + batch_size_ - 1) / batch_size_;
        t_preprocess_ms_ = source_.cpu_preprocess_ms();

        auto t_source_start = std::chrono::steady_clock::now();
        {
            BatchPfDriverContext ctx = make_context();
            source_.initialize(ctx, cs);
        }
        cs.synchronize();
        t_source_init_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_source_start).count();
        if (adjoint_) adjoint_->factorized_for_solve = -1;
    }

    // -------------------------------------------------------------------------
    // mark_reused — a session reusing this driver for another run() calls
    // this first so the one-time construction costs (allocation, cuDSS
    // ANALYSIS, context creation) are reported once, not on every run; on the
    // "hot" path (no new source at all) the source's own preprocess/upload
    // costs are zeroed too. The absolute numbers are what let a caller (or a
    // test) tell "reused" from "rebuilt".
    // -------------------------------------------------------------------------
    void mark_reused(bool hot)
    {
        t_alloc_ms_        = 0.;
        t_analysis_ms_     = 0.;
        t_context_init_ms_ = 0.;
        if (hot) {
            t_preprocess_ms_  = 0.;
            t_source_init_ms_ = 0.;
        }
    }

    // -------------------------------------------------------------------------
    // solve_JT_batch — batched adjoint solve Jᵀ λ = x̄ per active slot, in
    // ORIGINAL row order both ways (see BatchAdjoint above; first call builds
    // everything). Requires solve() to have run with keep_final_jacobian_ so
    // d_J_values_batch holds the converged J (alias mode), or a caller-owned
    // [capacity × nnz_J] snapshot of those values (d_J_ext, snapshot mode).
    //
    //   d_rhs_orig      : [n_contingencies × dim_J]; non-finite entries → 0.
    //   d_J_ext         : nullptr → use d_J_values_batch.
    //   want_gen_v_grad : also compute the gen_v (Vm-fixed bus) gradient
    //                     contraction into d_gvm_full (see gen_v_adjoint_kernel).
    //   d_Ybus_ext      : snapshot of [capacity × nnz_Y] patched Ybus values
    //                     (nullptr → d_Ybus_values_batch). gen_v only.
    //   d_V_ext_orig    : [n_contingencies × n_bus] converged V in original
    //                     order (nullptr → the chunk's own d_V_batch). gen_v only.
    //   is_vm_fixed_bus : [n_bus] (pv ∪ slack) mask, gen_v only.
    // Results: d_JT_sol_full_ptr() [n_contingencies × dim_J] and, when asked,
    // d_gvm_full_ptr() [n_contingencies × n_bus]; rows of compacted-out
    // (islanded) elements are 0. Synchronizes cs before returning.
    // -------------------------------------------------------------------------
    void solve_JT_batch(const cuda_real_type*    d_rhs_orig,
                        const cuda_real_type*    d_J_ext,
                        bool                     want_gen_v_grad,
                        const cudaComplexType*   d_Ybus_ext,
                        const cudaComplexType*   d_V_ext_orig,
                        const std::vector<char>& is_vm_fixed_bus);

    bool adjoint_ready() const { return static_cast<bool>(adjoint_); }
    const cuda_real_type* d_JT_sol_full_ptr() const {
        return adjoint_ ? thrust::raw_pointer_cast(adjoint_->d_sol_full.data()) : nullptr;
    }
    const cuda_real_type* d_gvm_full_ptr() const {
        return (adjoint_ && !adjoint_->d_gvm_full.empty())
            ? thrust::raw_pointer_cast(adjoint_->d_gvm_full.data()) : nullptr;
    }
    const cuda_real_type* j_values_ptr() const {
        return thrust::raw_pointer_cast(d_J_values_batch.data());
    }
    const cudaComplexType* ybus_values_ptr() const {
        return thrust::raw_pointer_cast(d_Ybus_values_batch.data());
    }

    // -------------------------------------------------------------------------
    // copy_results_to_host  — syncs cs and copies V_results + residuals.
    // -------------------------------------------------------------------------
    void copy_results_to_host(CplxVect& V_out, RealVect& res_out) const;

    // -------------------------------------------------------------------------
    // Branch-flow data  (no-op for features that do not call it)
    //
    // upload_branch_admittances : uploads ONLY the O(n_branches) admittance
    //   arrays (+ d_bus_vn_kv). Sets _has_branch_admittances. Used by
    //   ContingencyAnalysisSession::run() to make branch data available to the
    //   fused compute_limit_violations kernel BEFORE solve() runs its chunk
    //   loop, without allocating the dense flow-result buffers.
    // set_branch_data : calls upload_branch_admittances(), then also allocates
    //   d_or_amps_results/d_ex_amps_results and sets _has_branch_data. Public
    //   signature/behavior unchanged from before this split.
    // -------------------------------------------------------------------------
    void upload_branch_admittances(
        Eigen::Ref<const Eigen::VectorXi> branch_from,
        Eigen::Ref<const Eigen::VectorXi> branch_to,
        Eigen::Ref<const CplxVect>        yff_eff,
        Eigen::Ref<const CplxVect>        yft_eff,
        Eigen::Ref<const CplxVect>        ytf_eff,
        Eigen::Ref<const CplxVect>        ytt_eff,
        Eigen::Ref<const RealVect>        bus_vn_kv,
        double                            sn_mva);

    void set_branch_data(
        Eigen::Ref<const Eigen::VectorXi> branch_from,
        Eigen::Ref<const Eigen::VectorXi> branch_to,
        Eigen::Ref<const CplxVect>        yff_eff,
        Eigen::Ref<const CplxVect>        yft_eff,
        Eigen::Ref<const CplxVect>        ytf_eff,
        Eigen::Ref<const CplxVect>        ytt_eff,
        Eigen::Ref<const RealVect>        bus_vn_kv,
        double                            sn_mva);

    void copy_flow_results_to_host(RealVect& or_amps_out,
                                    RealVect& ex_amps_out) const;

    // -------------------------------------------------------------------------
    // compute_limit_violations: configure per-bus voltage (kV) / per-branch
    // current (kA) limits and the fused kernel's DIVERGENCE tolerance + output
    // capacity K. Requires upload_branch_admittances() (directly, or via
    // set_branch_data()) to have already run. NaN = "not configured" for that
    // element (matches lightsim2grid's convention).
    // -------------------------------------------------------------------------
    void set_violation_limits(
        Eigen::Ref<const RealVect> bus_vmin_kv,
        Eigen::Ref<const RealVect> bus_vmax_kv,
        Eigen::Ref<const RealVect> branch_limit_a1_ka,
        Eigen::Ref<const RealVect> branch_limit_a2_ka,
        double tol,
        int    K,
        int    n_lines);

    // -------------------------------------------------------------------------
    // compute_physical_violations: upload the plan, set the tolerance (MVAr), the
    // per-row output capacity K_q and the row gate (residual_tol: a row whose
    // residual is NaN or above it reports nothing), and (re)seed the outputs
    // -- the -1 "never simulated" sentinel included. Re-callable on a live
    // driver: ScenarioSweep calls it on EVERY run() so a reused driver never
    // shows a previous run's records. d_gen_off is a non-owning pointer to the
    // session's (n_contingencies x n_gen) uint8 mask in ORIGINAL row order
    // (nullptr / 0 = no generator contingencies); it must outlive solve().
    // -------------------------------------------------------------------------
    void set_bus_q_check(const BusQPlanData& plan,
                         double tol_mvar, int K_q, double residual_tol,
                         const unsigned char* d_gen_off, int n_gen);

    // The base ("n") case's own report: the same kernel over the base state's
    // converged V / Ybus / Sbus as a 1-slot batch (no gate, no generator
    // contingency, no result map). Requires set_bus_q_check(). The caller gates
    // it on the base solve's own convergence flag.
    void run_bus_q_check_n();

    // compute_physical_violations: same contract as set_bus_q_check for the droop
    // P-saturation check. tol_mw is converted to pu with sn_mva. The per-line
    // data is the base state's own (n_hvdc may be 0: rows then report count 0).
    void set_hvdc_p_check(double tol_mw, double sn_mva, int K_p, double residual_tol);
    void run_hvdc_p_check_n();

    // compute_physical_violations: same contract as set_bus_q_check for the
    // per-machine active-power check of the distributed slack (an EMPTY plan is
    // fine: rows then report count 0). tol_mw in MW. A plan whose entry count
    // changed drops any per-row set-points uploaded before.
    void set_gen_p_check(const GenPPlanData& plan, double tol_mw, int K_g, double residual_tol,
                         const unsigned char* d_gen_off, int n_gen);
    // Per-row active set-points of the plan's entries (MW, generator convention,
    // NaN = keep the base one), (n_contingencies x n_entries) in ORIGINAL row
    // order; an empty matrix drops them (base targets for every row). Requires
    // set_gen_p_check(). Kept across the re-seeding set_gen_p_check() does on
    // every run (ScenarioSweep), so the caller uploads only when they changed.
    void upload_gen_p_targets(const RealMatRM& targets);
    bool has_gen_p_targets() const { return gen_p_target_stride_ > 0; }
    void run_gen_p_check_n();

    double bus_q_setup_ms()  const { return t_bus_q_setup_ms_; }
    double hvdc_p_setup_ms() const { return t_hvdc_p_setup_ms_; }
    double gen_p_setup_ms()  const { return t_gen_p_setup_ms_; }

    // -------------------------------------------------------------------------
    // DLPack / zero-copy accessors
    // -------------------------------------------------------------------------
    const cudaComplexType* d_V_results_ptr() const {
        return thrust::raw_pointer_cast(d_V_results.data());
    }
    int  n_bus()            const { return base.n_bus; }
    int  device_id()        const { return base.device_id_; }
    int  n_contingencies_() const { return n_contingencies; }
    void synchronize()            { cs.synchronize(); }

    // H→D branch-admittance upload time from the most recent set_branch_data()
    // call (read back by the sessions into BatchTimings::t_branch_data_upload_ms).
    double branch_data_upload_ms() const { return t_branch_data_upload_ms_; }

    // H→D limits-upload + buffer-allocation time from the most recent
    // set_violation_limits() call (read back into BatchTimings::t_violation_setup_ms).
    double violation_setup_ms() const { return t_violation_setup_ms_; }

    // Non-copyable, non-movable
    BatchPfDriver(const BatchPfDriver&)            = delete;
    BatchPfDriver& operator=(const BatchPfDriver&) = delete;
    BatchPfDriver(BatchPfDriver&&)                 = delete;
    BatchPfDriver& operator=(BatchPfDriver&&)      = delete;

    // -------------------------------------------------------------------------
    // Context view  — built once in the ctor and reused per chunk by sources.
    // -------------------------------------------------------------------------
    BatchPfDriverContext make_context();

private:
    void _solve_chunk(int c_start, int actual_batch, BatchTimings& t);
    // First-call setup of the adjoint: skeleton transpose + position map,
    // buffers, cuDSS ANALYSIS of Jᵀ (with the forward's config).
    void _prepare_adjoint();
    // First-call setup of the gen_v contraction data (Ybus transpose-position
    // map, bus→row maps, Vm-fixed mask).
    void _prepare_gen_v_adjoint(const std::vector<char>& is_vm_fixed_bus);
};

#endif // BATCH_PF_DRIVER_CUH