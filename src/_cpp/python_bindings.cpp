// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "timing_utils.hpp"
#include "acpf_nr.hpp"
#include "contingency_analysis_helper.hpp"   // Contingency, Triplet
#include "contingency_analysis_session.hpp"  // ContingencyAnalysisSession
#include "injection_sweep.hpp"               // run_injection_sweep_gpu
#include "injection_sweep_session.hpp"       // InjectionSweepSession
#include "dlpack_export.hpp"                 // export_v_base_dlpack etc.

#ifdef GPUSIM2GRID_HAVE_LS2G
#include "ls2g_bridge.hpp"                   // make_*_session_from_lsgrid
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <tuple>
#include <vector>


PYBIND11_MODULE(_gpusim2grid, m)
{
    m.doc() =
        "Low-level CUDA extension for gpusim2grid: GPU AC power flow, batched "
        "contingency analysis and injection sweeps, plus zero-copy DLPack "
        "export. End users should normally import the Python wrappers from the "
        "gpusim2grid package rather than calling this module directly.";

    // -----------------------------------------------------------------
    // TimingEntry — the building block of AcPfTimings per-phase fields.
    //
    // Exposed to Python so callers can inspect both dimensions:
    //   timings.t_spmv.gpu_ms   — pure GPU timeline (excludes launch overhead)
    //   timings.t_spmv.wall_ms  — wall-clock after GPU sync (>= gpu_ms)
    //   timings.t_spmv.overhead_ms  — difference: driver/launch overhead
    // -----------------------------------------------------------------
    pybind11::class_<TimingEntry>(m, "TimingEntry",
        "Two-dimensional timing of a single phase: the pure GPU-timeline "
        "elapsed time (``gpu_ms``, from CUDA events) and the wall-clock time "
        "after stream synchronization (``wall_ms``). Their difference is "
        "driver/kernel-launch overhead.")
        .def_readonly("gpu_ms",  &TimingEntry::gpu_ms,
                      "GPU-timeline elapsed time (cudaEventElapsedTime), "
                      "excluding kernel-launch and driver overhead (ms)")
        .def_readonly("wall_ms", &TimingEntry::wall_ms,
                      "Wall-clock elapsed time from start() to "
                      "cudaEventSynchronize() completion (ms). "
                      "Always >= gpu_ms.")
        .def_property_readonly("overhead_ms",
                      [](const TimingEntry& e) { return e.wall_ms - e.gpu_ms; },
                      "Driver + kernel-launch overhead: wall_ms - gpu_ms (ms)")
        .def("__repr__", [](const TimingEntry& e) {
            return std::string("TimingEntry(gpu_ms=")
                + std::to_string(e.gpu_ms)
                + ", wall_ms=" + std::to_string(e.wall_ms)
                + ", overhead_ms=" + std::to_string(e.wall_ms - e.gpu_ms)
                + ")";
        });

    // -----------------------------------------------------------------
    // AcPfTimings — timing results returned by acpf_nr_gpu.
    // -----------------------------------------------------------------
    pybind11::class_<AcPfTimings>(m, "AcPfTimings",
        "Per-phase timing breakdown and convergence status for a single AC "
        "Newton-Raphson solve (``acpf_nr_gpu`` / ``AcPfNrSession``). One-time "
        "setup fields are wall-clock only; per-iteration fields are "
        ":class:`TimingEntry` totals accumulated over all NR iterations.")
        // --- one-time setup (wall-clock only) ---
        .def_readonly("t_init_ms",    &AcPfTimings::t_init_ms,
                      "Wall-clock: sparsity build, host uploads, "
                      "cuSPARSE/cuDSS descriptor creation (ms)")
        .def_readonly("t_analyze_ms", &AcPfTimings::t_analyze_ms,
                      "Wall-clock: cuDSS symbolic analysis — done once, "
                      "sparsity pattern of J never changes (ms)")
        // --- per-iteration totals (TimingEntry: gpu_ms + wall_ms) ---
        .def_readonly("t_spmv",      &AcPfTimings::t_spmv,
                      "cuSPARSE SpMV: Ibus = Ybus·V — total across all NR iterations")
        .def_readonly("t_fill_F",    &AcPfTimings::t_fill_F,
                      "fill_FP_kernel + fill_FQ_kernel — total across all NR iterations")
        .def_readonly("t_fill_J",    &AcPfTimings::t_fill_J,
                      "fill_J_kernel: compute J non-zeros from V and Ibus — total")
        .def_readonly("t_first_factorize", &AcPfTimings::t_first_factorize,
                      "cuDSS FACTORIZATION — first NR iteration only (iter 0)")
        .def_readonly("t_refactorize",     &AcPfTimings::t_refactorize,
                      "cuDSS REFACTORIZATION — all subsequent NR iterations (iter > 0) — total")
        .def_readonly("t_solve",     &AcPfTimings::t_solve,
                      "cuDSS SOLVE: dx = J⁻¹·(−F) — total across all NR iterations")
        .def_readonly("t_update_V",  &AcPfTimings::t_update_V,
                      "update_Va_kernel + update_Vm_kernel — total across all NR iterations")
        .def_readonly("t_mismatch",  &AcPfTimings::t_mismatch,
                      "compute the mismatch  ‖F‖∞ -total across all NR iterations")
        // --- scalar metadata ---
        .def_readonly("nb_iter",        &AcPfTimings::nb_iter,
                      "Number of Newton-Raphson iterations executed")
        .def_readonly("n_refactorize",  &AcPfTimings::n_refactorize,
                      "Number of REFACTORIZATION calls (nb_iter - 1)")
        .def_readonly("converged", &AcPfTimings::converged,
                      "True if ‖F‖∞ < tol at exit")
        // --- computed property ---
        .def_property_readonly("t_per_iter",
                      [](const AcPfTimings& t) { return t.t_per_iter(); },
                      "Mean TimingEntry per NR iteration "
                      "(sum of all per-phase totals / nb_iter); "
                      "zero entry if nb_iter == 0")
        .def_property_readonly("t_total",
                      [](const AcPfTimings& t) { return t.t_total(); },
                      "Total time (for all iterations). Only the transfer H->D and D->H are "
                      "omitted. This takes into account symbolic factorization (done once) and all the time for all iterations.")
        // --- repr ---
        .def("__repr__", [](const AcPfTimings & t) {
            auto pi = t.t_per_iter();
            std::string status = t.converged ? "CONVERGED" : "NOT CONVERGED";
            return std::string("AcPfTimings(")
                + status
                + ", nb_iter="         + std::to_string(t.nb_iter)
                + ", t_init_ms="       + std::to_string(t.t_init_ms)
                + ", t_analyze_ms="    + std::to_string(t.t_analyze_ms)
                + ", t_spmv="         + std::to_string(t.t_spmv.wall_ms)
                + ", t_fill_F="       + std::to_string(t.t_fill_F.wall_ms)
                + ", t_fill_J="       + std::to_string(t.t_fill_J.wall_ms)
                + ", t_first_factorize=" + std::to_string(t.t_first_factorize.wall_ms)
                + ", t_refactorize="    + std::to_string(t.t_refactorize.wall_ms)
                + ", n_refactorize="    + std::to_string(t.n_refactorize)
                + ", t_solve="        + std::to_string(t.t_solve.wall_ms)
                + ", t_update_V="     + std::to_string(t.t_update_V.wall_ms)
                + ", t_per_iter_wall_ms=" + std::to_string(pi.wall_ms)
                + ", t_per_iter_gpu_ms="  + std::to_string(pi.gpu_ms)
                + ")";
        });

    // -----------------------------------------------------------------
    // BatchTimings — returned by run_contingency_analysis_gpu and
    // run_injection_sweep_gpu.  Exposed under the legacy alias
    // "ContingencyTimings" as well (see m.attr() below) so that existing
    // Python code continues to work unchanged.
    // -----------------------------------------------------------------
    pybind11::class_<BatchTimings>(m, "BatchTimings",
        "Per-phase timing breakdown for a batched run "
        "(``run_contingency_analysis_gpu`` / ``run_injection_sweep_gpu`` and "
        "their session classes). One-time setup fields are wall-clock only; "
        "per-chunk fields are :class:`TimingEntry` totals accumulated over all "
        "chunks and iterations. Also exposed under the legacy alias "
        "``ContingencyTimings``.")
        // --- one-time setup (wall-clock only) ---
        .def_readonly("t_base_case_ms",  &BatchTimings::t_base_case_ms,
                      "Wall-clock: base-case NR to convergence (AcPfNrState construction) (ms)")
        .def_readonly("t_preprocess_ms", &BatchTimings::t_preprocess_ms,
                      "Wall-clock: CPU resolve_indices + build_flat_patches + build_blockdiag (ms)")
        .def_readonly("t_alloc_ms",      &BatchTimings::t_alloc_ms,
                      "Wall-clock: device buffer allocation + block-diagonal structure upload (ms)")
        .def_readonly("t_analysis_ms",   &BatchTimings::t_analysis_ms,
                      "Wall-clock: cuDSS ANALYSIS on single-system J (uniform-batch context) (ms)")
        .def_readonly("t_copy_flows_to_host_ms", &BatchTimings::t_copy_flows_to_host_ms,
                      "Wall-clock: D→H download of or_amps / ex_amps flow results (ms)")
        // --- per-chunk accumulated totals (TimingEntry: gpu_ms + wall_ms) ---
        .def_readonly("t_tile_V",      &BatchTimings::t_tile_V,
                      "D→D tiling of d_V_base into d_V_batch — total across all chunks")
        .def_readonly("t_tile_Ybus",   &BatchTimings::t_tile_Ybus,
                      "D→D tiling of base Ybus values into d_Ybus_values_batch — total across all chunks")
        .def_readonly("t_patch_Ybus",  &BatchTimings::t_patch_Ybus,
                      "apply_contingencies_kernel: subtract Ybus deltas — total across all chunks")
        .def_readonly("t_spmv",        &BatchTimings::t_spmv,
                      "cuSPARSE block-diagonal SpMV: Ibus_batch = Ybus_batch·V_batch — total")
        .def_readonly("t_fill_F",      &BatchTimings::t_fill_F,
                      "fill_FP_kernel + fill_FQ_kernel — total across all chunks × nb_iter")
        .def_readonly("t_fill_J",      &BatchTimings::t_fill_J,
                      "fill_J_kernel — total across all chunks × nb_iter")
        .def_readonly("t_first_factorize", &BatchTimings::t_first_factorize,
                      "cuDSS FACTORIZATION — single call at chunk 0, iter 0")
        .def_readonly("t_refactorize",     &BatchTimings::t_refactorize,
                      "cuDSS REFACTORIZATION — all subsequent calls (n_chunks × nb_iter − 1) — total")
        .def_readonly("t_solve",       &BatchTimings::t_solve,
                      "cuDSS SOLVE — total across all chunks × nb_iter")
        .def_readonly("t_update_V",    &BatchTimings::t_update_V,
                      "update_Va_kernel + update_Vm_kernel — total across all chunks × nb_iter")
        .def_readonly("t_residual",    &BatchTimings::t_residual,
                      "Post-loop SpMV + fill_F + compute_residuals_kernel — total across all chunks")
        .def_readonly("t_store_V",     &BatchTimings::t_store_V,
                      "D→D copy d_V_batch → d_V_results — total across all chunks")
        .def_readonly("t_flow_computation", &BatchTimings::t_flow_computation,
                      "compute_branch_flows_kernel — total across all chunks (0 if no branch data)")
        // --- metadata ---
        .def_readonly("n_contingencies", &BatchTimings::n_contingencies,
                      "Total number of contingencies")
        .def_readonly("n_chunks",        &BatchTimings::n_chunks,
                      "Number of chunks: ceil(n_contingencies / chunk_size)")
        .def_readonly("chunk_size",      &BatchTimings::chunk_size,
                      "Maximum contingencies per chunk (batch_size parameter)")
        .def_readonly("nb_iter",         &BatchTimings::nb_iter,
                      "Fixed NR iterations per chunk (no convergence check mid-loop)")
        .def_readonly("n_refactorize",   &BatchTimings::n_refactorize,
                      "Number of REFACTORIZATION calls (n_chunks × nb_iter − 1)")
        .def_readonly("n_disconnected",  &BatchTimings::n_disconnected,
                      "Contingencies skipped because they would disconnect the Ybus graph; "
                      "their residuals are set to NaN in the output")
        // --- computed properties ---
        .def_property_readonly("t_chunks_total_wall_ms",
                      [](const BatchTimings& t) { return t.t_chunks_total_wall_ms(); },
                      "Total wall-clock time for all chunk work (excludes one-time setup) (ms)")
        .def_property_readonly("t_per_contingency_ms",
                      [](const BatchTimings& t) { return t.t_per_contingency_ms(); },
                      "Mean wall-clock time per contingency across all chunks (ms)")
        // --- repr ---
        .def("__repr__", [](const BatchTimings& t) {
            return std::string("BatchTimings(")
                + "n_contingencies=" + std::to_string(t.n_contingencies)
                + ", n_chunks="      + std::to_string(t.n_chunks)
                + ", chunk_size="    + std::to_string(t.chunk_size)
                + ", nb_iter="       + std::to_string(t.nb_iter)
                + ", t_base_case_ms="  + std::to_string(t.t_base_case_ms)
                + ", t_analysis_ms="   + std::to_string(t.t_analysis_ms)
                + ", t_spmv_wall="     + std::to_string(t.t_spmv.wall_ms)
                + ", t_fill_F_wall="   + std::to_string(t.t_fill_F.wall_ms)
                + ", t_fill_J_wall="   + std::to_string(t.t_fill_J.wall_ms)
                + ", t_first_factorize_wall=" + std::to_string(t.t_first_factorize.wall_ms)
                + ", t_refactorize_wall="     + std::to_string(t.t_refactorize.wall_ms)
                + ", n_refactorize="          + std::to_string(t.n_refactorize)
                + ", t_solve_wall="    + std::to_string(t.t_solve.wall_ms)
                + ", t_update_V_wall=" + std::to_string(t.t_update_V.wall_ms)
                + ", t_residual_wall=" + std::to_string(t.t_residual.wall_ms)
                + ", t_per_contingency_ms=" + std::to_string(t.t_per_contingency_ms())
                + ")";
        });

    // Legacy Python alias — existing user code referencing
    // gpusim2grid._gpusim2grid.ContingencyTimings continues to work.
    m.attr("ContingencyTimings") = m.attr("BatchTimings");

  // -----------------------------------------------------------------
  // Functions
  // -----------------------------------------------------------------
  m.def("acpf_nr_gpu", acpf_nr_gpu,
        pybind11::arg("out"),
        pybind11::arg("ac_Ybus"),
        pybind11::arg("V"),
        pybind11::arg("Sbus"),
        pybind11::arg("slack_ids"),
        pybind11::arg("slack_weights"),
        pybind11::arg("pv"),
        pybind11::arg("pq"),
        pybind11::arg("max_iter"),
        pybind11::arg("tol"),
        pybind11::arg("nb_solve"),
        pybind11::arg("device") = -1,
        "Run the Newton-Raphson AC power-flow on the GPU.\n\n"
        "Returns an AcPfTimings struct with per-phase timing breakdowns\n"
        "and convergence status. The solution vector `out` is written\n"
        "in-place (resized to n_bus if needed).\n\n"
        "device: CUDA device ordinal (-1 = current device).");

  // -----------------------------------------------------------------
  // AcPfNrSession — stateful single-system NR solver.
  // Keeps the voltage vector on device so it can be exported via DLPack
  // without a host copy.  Use v_dlpack() for zero-copy PyTorch/JAX interop
  // or get_v() to copy back to a NumPy array.
  // -----------------------------------------------------------------
  pybind11::class_<AcPfNrSession, std::shared_ptr<AcPfNrSession>>(
      m, "AcPfNrSession",
      "Stateful single-system AC Newton-Raphson solver.\n\n"
      "Solves the AC power flow to convergence at construction and keeps the "
      "voltage vector and the converged Jacobian factorization on the device, "
      "so results can be exported zero-copy via DLPack (``v_dlpack()``) and the "
      "adjoint system can be solved (``solve_JT_dlpack()``) without re-running "
      "the forward solve. Use ``get_v()`` to copy the voltages back to NumPy.")
    .def(pybind11::init<
           const Eigen::SparseMatrix<eigen_cplx_type>&,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const RealVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const Eigen::VectorXi>,
           int, eigen_real_type, int>(),
         pybind11::arg("Ybus"),
         pybind11::arg("Vinit"),
         pybind11::arg("Sbus"),
         pybind11::arg("slack_ids"),
         pybind11::arg("slack_weights"),
         pybind11::arg("pv"),
         pybind11::arg("pq"),
         pybind11::arg("max_iter"),
         pybind11::arg("tol"),
         pybind11::arg("device") = -1,
         "Construct and immediately solve the base-case AC power flow.\n\n"
         "Ybus          : scipy.sparse complex (n_bus x n_bus) admittance matrix.\n"
         "Vinit         : (n_bus,) complex128 warm-start voltages.\n"
         "Sbus          : (n_bus,) complex128 per-unit complex injections.\n"
         "slack_ids     : (n_slack,) int32 slack bus indices.\n"
         "slack_weights : (n_slack,) float distributed-slack weights.\n"
         "pv, pq        : int32 PV / PQ bus indices.\n"
         "max_iter      : int, maximum NR iterations.\n"
         "tol           : float, convergence tolerance on ||F||inf.\n"
         "device        : CUDA device ordinal (-1 = current device).")
    .def_property_readonly("timings", &AcPfNrSession::timings,
         "AcPfTimings struct with per-phase timing breakdowns.")
    .def("get_v", &AcPfNrSession::get_v,
         "Copy the voltage vector from device to a NumPy array (one D→H transfer).")
    .def("v_dlpack", &export_v_acpfnr_dlpack,
         "Export the voltage vector as a DLPack capsule, shape [n_bus].\n"
         "Zero-copy: the tensor aliases live GPU memory owned by this session.\n"
         "Syncs the solver stream before returning.")
    .def("solve_JT_dlpack", &export_jt_solve_dlpack,
         pybind11::arg("rhs"),
         "Solve Jᵀλ = rhs using the converged LU factorization of Jᵀ.\n"
         "rhs: DLPack capsule, shape [dim_J], real float, on the same device.\n"
         "Returns a DLPack capsule for λ (shape [dim_J], real float).\n"
         "WARNING: the result aliases an internal buffer — clone the tensor\n"
         "before calling solve_JT_dlpack again to avoid overwriting.")
    .def_property_readonly("n_pvpq", &AcPfNrSession::n_pvpq,
        "Number of PV+PQ buses (first block of the NR state vector).")
    .def_property_readonly("n_pq", &AcPfNrSession::n_pq,
        "Number of PQ buses (second block of the NR state vector).")
    .def_property_readonly("dim_J", &AcPfNrSession::dim_J,
        "Jacobian dimension: n_pvpq + n_pq.")
    .def_property_readonly("pvpq", &AcPfNrSession::pvpq,
        "Sorted PV+PQ bus indices (D→H copy, called once per backward pass).")
    .def_property_readonly("pq", &AcPfNrSession::pq,
        "PQ bus indices (D→H copy, called once per backward pass).");

  // -----------------------------------------------------------------
  // ContingencySolverType enum — selects the linear-solve strategy.
  // Registered BEFORE bindings that use it as a default argument
  // (e.g. acpf_nr_gpu_injection's strategy_type kwarg).
  // -----------------------------------------------------------------
  pybind11::enum_<ContingencySolverType>(m, "ContingencySolverType")
      .value("DirectRefactorEvery",   ContingencySolverType::DirectRefactorEvery,
             "Fill J every iteration; FACTORIZE once globally, REFACTORIZE for all "
             "subsequent calls.  Highest accuracy (default).")
      .value("DirectBaseCaseFactors", ContingencySolverType::DirectBaseCaseFactors,
             "Reuse base-case LU factors for all contingencies (no per-contingency "
             "factorize).  Cheapest; approximate (uses base-case J).")
      .value("DirectIter0Only",       ContingencySolverType::DirectIter0Only,
             "Fill J at iter==0 per chunk; FACTORIZE/REFACTORIZE once per chunk; "
             "SOLVE-only for remaining iterations.  Compromise between accuracy and cost.")
      .value("DirectRefactorEveryN",  ContingencySolverType::DirectRefactorEveryN,
             "Fill J every iteration; REFACTORIZE every N NR calls (N set via refactor_period). "
             "N=1 degenerates to DirectRefactorEvery.");

  // -----------------------------------------------------------------
  // acpf_nr_gpu_injection
  //
  // Batched-injection AC power flow sweep using GPU-accelerated Newton-Raphson.
  // Runs n_scenarios power flows in parallel, each with the SAME base Ybus
  // but a DIFFERENT (p_mw, q_mvar) injection profile.
  //
  // Parameters
  // ----------
  // Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq
  //     Same convention as run_contingency_analysis_gpu.  Sbus is the per-unit
  //     complex injection used for the base-case NR (the warm-start).
  // p_mw       : (n_scenarios, n_bus) float64, row-major
  //              Per-scenario active power in MW.
  // q_mvar     : (n_scenarios, n_bus) float64, row-major
  //              Per-scenario reactive power in MVAr.
  // sn_mva     : float — system base apparent power (MVA).  Used internally to
  //              convert (p_mw, q_mvar) → per-unit complex Sbus.
  // batch_size : int — scenarios per GPU chunk.
  // nb_iter    : int — fixed NR iterations per chunk.
  // max_iter_base, tol_base : as in run_contingency_analysis_gpu.
  //
  // Returns
  // -------
  // (V_results, residuals, timings) tuple
  //   V_results : (n_scenarios * n_bus,) complex128 voltages (per-unit)
  //   residuals : (n_scenarios,) float64 ‖F‖∞ residual per scenario
  //   timings   : BatchTimings — same struct used by contingency analysis;
  //               n_disconnected = 0, t_patch_Ybus = 0.
  //
  // strategy_type   : one of ContingencySolverType.{DirectRefactorEvery,
  //                   DirectIter0Only, DirectBaseCaseFactors,
  //                   DirectRefactorEveryN}.  Default: DirectRefactorEvery.
  // refactor_period : only used when strategy_type == DirectRefactorEveryN.
  // -----------------------------------------------------------------
  m.def("acpf_nr_gpu_injection",
    [](const Eigen::SparseMatrix<eigen_cplx_type>&     Ybus,
       Eigen::Ref<const CplxVect>                      Vinit,
       Eigen::Ref<const CplxVect>                      Sbus,
       Eigen::Ref<const Eigen::VectorXi>               slack_ids,
       Eigen::Ref<const RealVect>                      slack_weights,
       Eigen::Ref<const Eigen::VectorXi>               pv,
       Eigen::Ref<const Eigen::VectorXi>               pq,
       Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
       Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
       double sn_mva,
       int batch_size,
       int nb_iter,
       int max_iter_base,
       double tol_base,
       ContingencySolverType strategy_type,
       int refactor_period)
       -> pybind11::tuple
    {
        CplxVect V_out;
        RealVect res_out;
        BatchTimings timings = run_injection_sweep_gpu(
            V_out, res_out,
            Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq,
            p_mw, q_mvar, sn_mva,
            batch_size, nb_iter, max_iter_base, tol_base,
            strategy_type, refactor_period);
        return pybind11::make_tuple(V_out, res_out, timings);
    },
    pybind11::arg("Ybus"),
    pybind11::arg("Vinit"),
    pybind11::arg("Sbus"),
    pybind11::arg("slack_ids"),
    pybind11::arg("slack_weights"),
    pybind11::arg("pv"),
    pybind11::arg("pq"),
    pybind11::arg("p_mw"),
    pybind11::arg("q_mvar"),
    pybind11::arg("sn_mva"),
    pybind11::arg("batch_size"),
    pybind11::arg("nb_iter"),
    pybind11::arg("max_iter_base")   = 10,
    pybind11::arg("tol_base")        = 1e-6,
    pybind11::arg("strategy_type")   = ContingencySolverType::DirectRefactorEvery,
    pybind11::arg("refactor_period") = 1);

  // -----------------------------------------------------------------
  // ContingencyAnalysisSession (exposed as ContingencyAnalysisSession;
  // the Python ContingencyAnalysisSolver wrapper is in gpusim2grid.contingency_analysis)
  // -----------------------------------------------------------------
  pybind11::class_<ContingencyAnalysisSession,
                   std::shared_ptr<ContingencyAnalysisSession>>(
      m, "ContingencyAnalysisSession",
      "Stateful GPU N-k contingency analysis solver (low-level binding).\n\n"
      "Solves the base case once at construction, then evaluates a batch of "
      "contingencies in GPU chunks, reusing the base-case factorization. "
      "Prefer the Python wrapper "
      ":class:`gpusim2grid.contingency_analysis.ContingencyAnalysisSolver`, "
      "which adds string strategy selection, device parsing, and lazy "
      "host-transfer result buffers.\n\n"
      "Lifecycle: set_branch_data() -> build_contingencies() -> run() -> "
      "compute_flows().")
    .def(pybind11::init<
           const Eigen::SparseMatrix<eigen_cplx_type>&,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const RealVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const Eigen::VectorXi>,
           int, int, int, double, int>(),
         pybind11::arg("Ybus"),
         pybind11::arg("Vinit"),
         pybind11::arg("Sbus"),
         pybind11::arg("slack_ids"),
         pybind11::arg("slack_weights"),
         pybind11::arg("pv"),
         pybind11::arg("pq"),
         pybind11::arg("batch_size"),
         pybind11::arg("nb_iter"),
         pybind11::arg("max_iter_base") = 10,
         pybind11::arg("tol_base")      = 1e-6,
         pybind11::arg("device")        = -1,
         "Construct and solve the base case.\n\n"
         "Ybus, Vinit, Sbus, slack_ids, slack_weights, pv, pq : grid arrays "
         "(same convention as AcPfNrSession).\n"
         "batch_size    : contingencies processed per GPU chunk.\n"
         "nb_iter       : fixed NR iterations per chunk (no mid-loop convergence check).\n"
         "max_iter_base : max NR iterations for the base-case solve.\n"
         "tol_base      : ||F||inf tolerance for the base-case solve.\n"
         "device        : CUDA device ordinal (-1 = current device).")
    .def("set_branch_data",
         &ContingencyAnalysisSession::set_branch_data,
         pybind11::arg("branch_from"),
         pybind11::arg("branch_to"),
         pybind11::arg("yff"),
         pybind11::arg("yft"),
         pybind11::arg("ytf"),
         pybind11::arg("ytt"),
         pybind11::arg("bus_vn_kv"),
         pybind11::arg("sn_mva"),
         "Store per-branch pi-model effective admittances (yff, yft, ytf, ytt), "
         "the from/to bus indices, per-bus nominal voltage (kV) and system base "
         "(MVA). Branches are ordered lines-then-trafos. Required before "
         "build_contingencies() and compute_flows().")
    .def("build_contingencies",
         &ContingencyAnalysisSession::build_contingencies,
         pybind11::arg("branch_ids_per_ctg"),
         "Define the contingency set from a list of branch-index lists; each "
         "inner list is the branches tripped in that contingency. Requires "
         "set_branch_data() first.")
    .def("run",   &ContingencyAnalysisSession::run,
         "Solve all contingencies. Fills the device-side voltage and residual "
         "buffers; contingencies that would disconnect the graph get NaN residuals.")
    .def("compute_flows", &ContingencyAnalysisSession::compute_flows,
         "Compute per-branch ampere flows for every contingency from the stored "
         "voltages (tripped branches zeroed). Requires run() and set_branch_data().")
    .def("get_V_results",  &ContingencyAnalysisSession::get_V_results,
         "Copy batch voltages to host: (n_contingencies * n_bus,) complex128.")
    .def("get_residuals",  &ContingencyAnalysisSession::get_residuals,
         "Copy per-contingency ||F||inf residuals to host: (n_contingencies,) float64.")
    .def("get_or_amps",    &ContingencyAnalysisSession::get_or_amps,
         "Copy origin-terminal ampere flows to host: (n_contingencies * n_branches,). "
         "Requires compute_flows().")
    .def("get_ex_amps",    &ContingencyAnalysisSession::get_ex_amps,
         "Copy extremity-terminal ampere flows to host: (n_contingencies * n_branches,). "
         "Requires compute_flows().")
    .def("get_timings",    &ContingencyAnalysisSession::get_timings,
         "Return the :class:`BatchTimings` from the most recent run().")
    .def_property_readonly("n_contingencies", &ContingencyAnalysisSession::n_contingencies,
         "Number of contingencies in the current set.")
    .def_property_readonly("n_bus",           &ContingencyAnalysisSession::n_bus,
         "Number of buses in the grid.")
    .def_property_readonly("n_branches",      &ContingencyAnalysisSession::n_branches,
         "Number of branches (lines + trafos); available after set_branch_data().")
    .def_readwrite("batch_size", &ContingencyAnalysisSession::batch_size_,
                   "Contingencies per GPU chunk (takes effect on the next run())")
    .def_readonly("used_batch_size", &ContingencyAnalysisSession::used_batch_size_,
                   "Batch size effectively used during the last run()")
    .def_readwrite("nb_iter",    &ContingencyAnalysisSession::nb_iter_,
                   "Fixed NR iterations per chunk (takes effect on the next run())")
    .def_readwrite("refactor_period", &ContingencyAnalysisSession::refactor_period_,
                   "Refactor period N for DirectRefactorEveryN strategy (takes effect on the next run())")
    .def_readwrite("strategy_type", &ContingencyAnalysisSession::strategy_type_,
                   "Linear-solve strategy (ContingencySolverType enum; takes effect on the next run())")
    // -------------------------------------------------------------------
    // Zero-copy DLPack exporters.
    // The returned capsule aliases live device memory; calling run() again
    // overwrites it in place.  Clone the tensor before a subsequent run()
    // if a snapshot is needed.
    // -------------------------------------------------------------------
    .def("v_base_dlpack",    &export_v_base_dlpack,
         "Export base-case voltages as DLPack capsule, shape [n_bus].\n"
         "Syncs the base-case stream before returning.")
    .def("v_results_dlpack", &export_v_results_dlpack,
         "Export batch voltages as DLPack capsule, shape [n_contingencies, n_bus].\n"
         "Requires run() to have been called.  Syncs the solver stream.");

  // -----------------------------------------------------------------
  // InjectionSweepSession — stateful batched-injection PF solver.
  // Base-case NR runs once at construction; set_injections() + run() may be
  // called repeatedly to sweep different injection sets reusing that base.
  // -----------------------------------------------------------------
  pybind11::class_<InjectionSweepSession,
                   std::shared_ptr<InjectionSweepSession>>(
      m, "InjectionSweepSession",
      "Stateful GPU batched-injection power flow solver (low-level binding).\n\n"
      "Solves the base case once at construction; set_injections() + run() may "
      "be called repeatedly to sweep different (P, Q) injection sets reusing "
      "that base case. Prefer the Python wrapper "
      ":class:`gpusim2grid.injection_sweep.InjectionSweepSolver`.")
    .def(pybind11::init<
           const Eigen::SparseMatrix<eigen_cplx_type>&,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const CplxVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const RealVect>,
           Eigen::Ref<const Eigen::VectorXi>,
           Eigen::Ref<const Eigen::VectorXi>,
           int, int, int, double, int>(),
         pybind11::arg("Ybus"),
         pybind11::arg("Vinit"),
         pybind11::arg("Sbus"),
         pybind11::arg("slack_ids"),
         pybind11::arg("slack_weights"),
         pybind11::arg("pv"),
         pybind11::arg("pq"),
         pybind11::arg("batch_size"),
         pybind11::arg("nb_iter"),
         pybind11::arg("max_iter_base") = 10,
         pybind11::arg("tol_base")      = 1e-6,
         pybind11::arg("device")        = -1,
         "Construct and solve the base case. Arguments match "
         "ContingencyAnalysisSession, except batch_size counts injection "
         "scenarios per GPU chunk.")
    .def("set_injections",
         &InjectionSweepSession::set_injections,
         pybind11::arg("p_mw"),
         pybind11::arg("q_mvar"),
         pybind11::arg("sn_mva"),
         "Store the (n_scenarios × n_bus) MW / MVAr injection arrays "
         "(converted to per-unit on run()). May be called repeatedly.")
    .def("set_branch_data",
         &InjectionSweepSession::set_branch_data,
         pybind11::arg("branch_from"),
         pybind11::arg("branch_to"),
         pybind11::arg("yff"),
         pybind11::arg("yft"),
         pybind11::arg("ytf"),
         pybind11::arg("ytt"),
         pybind11::arg("bus_vn_kv"),
         pybind11::arg("sn_mva"),
         "Store π-model branch admittances. Must be called before compute_flows().")
    .def("run",            &InjectionSweepSession::run,
         "Solve all injection scenarios. Fills the device-side voltage and "
         "residual buffers. Requires set_injections() first.")
    .def("compute_flows",  &InjectionSweepSession::compute_flows,
         "Compute branch flows for all scenarios from d_V_results. "
         "Requires run() and set_branch_data() to have been called.")
    .def("get_V_results",  &InjectionSweepSession::get_V_results,
         "Copy batch voltages to host: (n_scenarios * n_bus,) complex128.")
    .def("get_residuals",  &InjectionSweepSession::get_residuals,
         "Copy per-scenario ||F||inf residuals to host: (n_scenarios,) float64.")
    .def("get_or_amps",    &InjectionSweepSession::get_or_amps,
         "Copy origin-terminal ampere flows to host: (n_scenarios * n_branches,). "
         "Requires compute_flows().")
    .def("get_ex_amps",    &InjectionSweepSession::get_ex_amps,
         "Copy extremity-terminal ampere flows to host: (n_scenarios * n_branches,). "
         "Requires compute_flows().")
    .def("get_timings",    &InjectionSweepSession::get_timings,
         "Return the :class:`BatchTimings` from the most recent run().")
    .def_property_readonly("n_scenarios", &InjectionSweepSession::n_scenarios,
         "Number of injection scenarios in the current set.")
    .def_property_readonly("n_bus",       &InjectionSweepSession::n_bus,
         "Number of buses in the grid.")
    .def_property_readonly("n_branches",  &InjectionSweepSession::n_branches,
         "Number of branches (lines + trafos); available after set_branch_data().")
    .def_readwrite("batch_size", &InjectionSweepSession::batch_size_,
                   "Scenarios per GPU chunk (takes effect on the next run())")
    .def_readonly("used_batch_size", &InjectionSweepSession::used_batch_size_,
                   "Batch size effectively used during the last run()")
    .def_readwrite("nb_iter",    &InjectionSweepSession::nb_iter_,
                   "Fixed NR iterations per chunk (takes effect on the next run())")
    .def_readwrite("refactor_period", &InjectionSweepSession::refactor_period_,
                   "Refactor period N for DirectRefactorEveryN strategy (takes effect on the next run())")
    .def_readwrite("strategy_type", &InjectionSweepSession::strategy_type_,
                   "Linear-solve strategy (ContingencySolverType enum; takes effect on the next run())")
    // -------------------------------------------------------------------
    // Zero-copy DLPack exporters.
    // -------------------------------------------------------------------
    .def("v_base_dlpack",    &export_v_base_dlpack_inj,
         "Export base-case voltages as DLPack capsule, shape [n_bus].\n"
         "Syncs the base-case stream before returning.")
    .def("v_results_dlpack", &export_v_results_dlpack_inj,
         "Export batch voltages as DLPack capsule, shape [n_scenarios, n_bus].\n"
         "Requires run() to have been called.  Syncs the solver stream.");

    // -----------------------------------------------------------------
    // Zero-copy construction from a solved lightsim2grid LSGrid
    // (only when gpusim2grid was built against lightsim2grid_core)
    // -----------------------------------------------------------------
#ifdef GPUSIM2GRID_HAVE_LS2G
    m.def("_make_ca_session_from_lsgrid",
        [](pybind11::object grid_py, bool init_from_n_powerflow,
           int batch_size, int nb_iter, int max_iter_base, double tol_base,
           int device) {
            ls2g::LSGrid& grid = grid_py.cast<ls2g::LSGrid&>();
            return make_ca_session_from_lsgrid(
                grid, init_from_n_powerflow, batch_size, nb_iter,
                max_iter_base, tol_base, device);
        },
        pybind11::arg("grid"),
        pybind11::arg("init_from_n_powerflow") = true,
        pybind11::arg("batch_size")            = 100,
        pybind11::arg("nb_iter")               = 4,
        pybind11::arg("max_iter_base")         = 10,
        pybind11::arg("tol_base")              = 1e-6,
        pybind11::arg("device")                = -1,
        "Build a ContingencyAnalysisSession directly from a solved lightsim2grid "
        "LSGrid (zero-copy: Ybus/Sbus/V/pv/pq/slack + branch admittances are "
        "read off the C++ object). Branch data is set automatically.");

    m.def("_make_is_session_from_lsgrid",
        [](pybind11::object grid_py, bool init_from_n_powerflow,
           int batch_size, int nb_iter, int max_iter_base, double tol_base,
           int device, bool with_branch_data) {
            ls2g::LSGrid& grid = grid_py.cast<ls2g::LSGrid&>();
            return make_is_session_from_lsgrid(
                grid, init_from_n_powerflow, batch_size, nb_iter,
                max_iter_base, tol_base, device, with_branch_data);
        },
        pybind11::arg("grid"),
        pybind11::arg("init_from_n_powerflow") = true,
        pybind11::arg("batch_size")            = 100,
        pybind11::arg("nb_iter")               = 4,
        pybind11::arg("max_iter_base")         = 10,
        pybind11::arg("tol_base")              = 1e-6,
        pybind11::arg("device")                = -1,
        pybind11::arg("with_branch_data")      = true,
        "Build an InjectionSweepSession directly from a solved lightsim2grid "
        "LSGrid (zero-copy). Branch data is set automatically when requested.");

    m.def("_make_acpf_session_from_lsgrid",
        [](pybind11::object grid_py, int max_iter, double tol, int device) {
            ls2g::LSGrid& grid = grid_py.cast<ls2g::LSGrid&>();
            return make_acpf_session_from_lsgrid(grid, max_iter, tol, device);
        },
        pybind11::arg("grid"),
        pybind11::arg("max_iter") = 10,
        pybind11::arg("tol")      = 1e-8,
        pybind11::arg("device")   = -1,
        "Build a single-system AcPfNrSession from a solved lightsim2grid LSGrid, "
        "solving the same augmented system (distributed slack / extensions) via "
        "the NRLedger read off the C++ object.");

    m.attr("have_ls2g_bridge") = true;
#else
    m.attr("have_ls2g_bridge") = false;
#endif

    // -----------------------------------------------------------------
    // Compilation options — queryable at runtime
    // -----------------------------------------------------------------
#ifdef GPUSIM2GRID_REAL_FLOAT
    m.attr("is_fp32") = true;
#else
    m.attr("is_fp32") = false;
#endif
}