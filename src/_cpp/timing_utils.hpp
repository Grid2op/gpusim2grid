// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <chrono>

// ---------------------------------------------------------------------------
// Helper: duration between two steady_clock time points, in milliseconds.
// Wall-clock helper — used only for t_init_ms / t_analyze_ms.
// ---------------------------------------------------------------------------
static double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// TimingEntry
//
// Holds both the GPU-measured time (via cudaEventElapsedTime) and the
// wall-clock time (via std::chrono) for a single timed phase.
//
//   gpu_ms  — time on the GPU timeline between the two cudaEvent stamps.
//             Excludes kernel-launch overhead and any CPU work in the segment.
//   wall_ms — CPU wall-clock elapsed from start() to stop_ms().
//             Because stop_ms() calls cudaEventSynchronize(), the CPU timer
//             stops *after* the GPU work finishes, so wall_ms >= gpu_ms always.
//             The difference (wall_ms - gpu_ms) is pure driver / launch overhead.
//
// Arithmetic helpers += and + are provided so that per-iteration entries can
// be accumulated into per-phase totals directly:
//   timings.t_spmv += timer.stop_ms();
// ---------------------------------------------------------------------------
struct TimingEntry {
    double gpu_ms  = 0.;
    double wall_ms = 0.;

    TimingEntry& operator+=(const TimingEntry& rhs) {
        gpu_ms  += rhs.gpu_ms;
        wall_ms += rhs.wall_ms;
        return *this;
    }
    TimingEntry operator+(const TimingEntry& rhs) const {
        return {gpu_ms + rhs.gpu_ms, wall_ms + rhs.wall_ms};
    }
    TimingEntry operator/(double d) const {
        return {gpu_ms / d, wall_ms / d};
    }
};

// ---------------------------------------------------------------------------
// AcPfTimings
//
// All durations in milliseconds.
//
// One-time setup fields (plain double — no GPU event available for these):
//   t_init_ms     — wall-clock total: pvpq sort + J sparsity build + H→D uploads
//                   + cuSPARSE descriptor creation.  Mixed CPU/GPU work; an event
//                   pair would only capture the GPU tail, so chrono is used.
//                   Sub-components (sum to ≈ t_init_ms, remainder = cuSPARSE setup):
//     t_build_J_ms  — sub-phase: pvpq sort + build_J_structure() (CPU only):
//                     derives J CSR sparsity pattern and the four scatter maps
//                     (d_map_j11/12/21/22) that link Ybus nonzeros to J entries.
//     t_upload_ms   — sub-phase: all H→D data transfers: Ybus values + CSR
//                     indices, scatter maps, Sbus, V_init, J skeleton arrays.
//   t_analyze_ms  — wall-clock: cuDSS symbolic analysis / reordering (done once).
//   t_prepare_jt_ms — wall-clock: prepare_JT() -- builds Jᵀ via cusparseCsr2cscEx2
//                     and factorizes it with a dedicated cuDSS context, for the
//                     adjoint (backward-pass) solve. Called unconditionally at
//                     the end of construction; comparable in cost to
//                     t_analyze_ms + t_first_factorize since it repeats
//                     equivalent symbolic-analysis + factorization work on the
//                     transposed system.
//
// Per-iteration accumulated totals (TimingEntry: gpu + wall):
//   t_spmv        — cuSPARSE SpMV: Ibus = Ybus·V
//   t_fill_F      — fill_FP_kernel + fill_FQ_kernel
//   t_fill_J      — fill_J_kernel: compute J non-zeros from V and Ibus
//   t_first_factorize — cuDSS FACTORIZATION (iter 0 only)
//   t_refactorize     — cuDSS REFACTORIZATION (iter > 0), accumulated total
//   t_solve       — cuDSS SOLVE: dx = J⁻¹·(−F)
//   t_update_V    — update_Va_kernel + update_Vm_kernel
//
// nb_iter   : number of NR iterations executed
// converged : whether ‖F‖∞ < tol at exit
//
// D→H (wall-clock only):
//   t_copy_v_to_host_ms — copy_V_to_host(): the D→H copy behind get_v().
//
// Note on presolved_v (init_from_n_powerflow): when the caller trusts Vinit as
// already converged, the NR loop is skipped entirely (nb_iter == 0) after one
// validation fill_F/fill_J/FACTORIZE — t_solve/t_update_V/t_refactorize stay 0
// unless an augmented-ledger extension (distributed slack / HVDC / VC) is
// active, in which case one solve + feature-state update still runs to seed
// that extension's running state. The aggregation methods below are plain
// sums of these fields, so they remain correct in either mode; only their
// *composition* differs (dominated by t_first_factorize/t_analyze_ms in the
// presolved_v fast path vs. spread across every NR iteration otherwise).
//
// Aggregation (read-only, computed — no new state):
//   t_cpu_preprocess_ms() — alias for t_build_J_ms (already pure CPU).
//   t_host_to_device_ms() — alias for t_upload_ms (already pure H→D).
//   t_device_to_host_ms() — alias for t_copy_v_to_host_ms.
//   t_gpu_compute_ms()    — t_analyze_ms + all per-iteration GPU-phase wall time.
//   t_grand_total_ms()    — everything above; should match an external
//                           stopwatch wrapped around construction → get_v().
// ---------------------------------------------------------------------------
struct AcPfTimings {
    // --- one-time setup (wall-clock only, mixed CPU/GPU) ---
    double t_init_ms    = 0.;   // total: see sub-fields below
    double t_build_J_ms = 0.;   // sub-phase: J sparsity + scatter maps (CPU)
    double t_upload_ms  = 0.;   // sub-phase: H→D data transfers
    double t_analyze_ms = 0.;   // cuDSS symbolic analysis
    double t_prepare_jt_ms = 0.;   // Jᵀ transpose + cuDSS analyze/factorize (adjoint setup)

    // --- per-iteration totals (GPU + wall) ---
    TimingEntry t_spmv;
    TimingEntry t_fill_F;
    TimingEntry t_fill_J;
    TimingEntry t_first_factorize;   // FACTORIZATION (iter 0 only)
    TimingEntry t_refactorize;       // REFACTORIZATION (iter > 0)
    TimingEntry t_mismatch;
    TimingEntry t_solve;
    TimingEntry t_update_V;

    // --- D→H (wall-clock only) ---
    double t_copy_v_to_host_ms = 0.;   // copy_V_to_host(): D→H copy behind get_v()

    int  nb_iter      = 0;
    int  n_refactorize = 0;   // number of refactorize calls (nb_iter - 1)
    bool converged    = false;

    // Mean time per NR iteration.
    TimingEntry t_per_iter() const {
        if (nb_iter <= 0) return {};
        TimingEntry sum = t_spmv + t_fill_F + t_fill_J
                        + t_first_factorize + t_refactorize + t_solve + t_update_V;
        return sum / static_cast<double>(nb_iter);
    }

    TimingEntry t_total() const {
        TimingEntry res;
        res.wall_ms = t_analyze_ms;
        res += t_spmv + t_fill_F + t_fill_J
             + t_first_factorize + t_refactorize + t_solve + t_update_V;
        return res;
    }

    // Note: t_total() intentionally excludes t_prepare_jt_ms (adjoint setup is
    // not part of the forward-solve "total"); t_gpu_compute_ms()/
    // t_grand_total_ms() below include it since it's real one-time GPU work
    // performed during construction.

    // --- coarse aggregation (read-only; see note above) ---
    double t_cpu_preprocess_ms() const { return t_build_J_ms; }
    // t_init_ms - t_build_J_ms rather than a plain alias for t_upload_ms:
    // t_init_ms is one continuous wall-clock span covering build_J + upload
    // + cuSPARSE/cuDSS handle & descriptor creation (the last of which has no
    // dedicated field), so this is the exact non-overlapping remainder after
    // CPU preprocessing -- keeping t_cpu_preprocess_ms() + t_host_to_device_ms()
    // == t_init_ms exactly, which t_grand_total_ms() below relies on.
    double t_host_to_device_ms() const { return t_init_ms - t_build_J_ms; }
    double t_device_to_host_ms() const { return t_copy_v_to_host_ms; }

    double t_gpu_compute_ms() const {
        return t_analyze_ms + t_prepare_jt_ms
             + (t_spmv + t_fill_F + t_fill_J + t_first_factorize
                + t_refactorize + t_solve + t_update_V + t_mismatch).wall_ms;
    }

    double t_grand_total_ms() const {
        return t_init_ms + t_gpu_compute_ms() + t_copy_v_to_host_ms;
    }
};

// =============================================================================
// Add this struct to timing_utils.hpp after AcPfTimings
// =============================================================================

// ---------------------------------------------------------------------------
// ContingencyTimings
//
// All durations in milliseconds.
//
// One-time setup (wall-clock only — mixed CPU/GPU):
//   t_base_case_ms   — AcPfNrState construction: full base-case NR to convergence,
//                      including J sparsity build, H→D uploads, cuDSS analysis,
//                      and all NR iterations (or, under presolved_v, one
//                      validation fill_F/fill_J/FACTORIZE instead of a loop —
//                      see AcPfTimings' note on presolved_v).  The build_J /
//                      upload sub-components are folded into t_preprocess_ms
//                      and t_alloc_ms below for a complete cross-cutting
//                      picture; t_base_case_solve_only_ms exposes the
//                      non-overlapping remainder (analyze + NR/validation)
//                      so a grand total can be computed without double-
//                      counting the folded share. Kept for convenience/
//                      backwards-compat; grand-total composition uses the
//                      decomposed pieces instead.
//   t_preprocess_ms  — all CPU preprocessing (pure CPU, no GPU work mixed in):
//                        • base-case: pvpq sort + build_J_structure() (scatter maps)
//                        • contingency: resolve_indices + build_flat_patches
//                          + build_blockdiag_csr
//                        • injection: per-scenario Sbus build (P,Q → per-unit complex)
//   t_alloc_ms       — H→D data transfers + device buffer allocation:
//                        • base-case: Ybus values/indices, scatter maps, Sbus,
//                          V_init, J skeleton (from AcPfNrState ctor)
//                        • block-diagonal CSR structure + chunk-sized working
//                          buffers (BatchPfDriver ctor)
//   t_analysis_ms    — cuDSS init + ANALYSIS (symbolic factorization) + policy
//                      init only (NOT source-specific setup — see
//                      t_source_init_ms; base-case analysis is inside
//                      t_base_case_ms / t_base_case_solve_only_ms)
//   t_source_init_ms — source-specific one-time GPU setup, split out of what
//                      used to be bundled into t_analysis_ms: flat-patch/mask
//                      H→D upload (contingency) or full Sbus_all H→D upload +
//                      one-time Ybus D→D tiling (injection)
//   t_branch_data_upload_ms — H→D upload of branch admittances
//                      (set_branch_data(), called from compute_flows())
//   t_base_case_solve_only_ms — non-overlapping remainder of t_base_case_ms:
//                      cuDSS analyze + NR iterations (or presolved_v
//                      validation) only, excluding the build_J/upload share
//                      already folded into t_preprocess_ms/t_alloc_ms
//   t_copy_flows_to_host_ms  — D→H download of or_amps/ex_amps (compute_flows())
//   t_copy_V_to_host_ms      — D→H download behind get_V_results()/.to_numpy()
//   t_copy_residuals_to_host_ms — D→H download behind get_residuals()/.to_numpy()
//
// Per-chunk accumulated totals (TimingEntry: gpu_ms + wall_ms):
//   t_tile_V         — cudaMemcpyAsync: d_V_base tiled into d_V_batch
//   t_tile_Ybus      — cudaMemcpyAsync: base Ybus values tiled into d_Ybus_values_batch
//   t_patch_Ybus     — apply_contingencies_kernel: subtract Ybus deltas
//   t_tile_Sbus      — injection only: D→D Sbus row-slice copy + phantom-pad
//                      (stays zero for contingency, same convention as
//                      t_tile_Ybus/t_patch_Ybus staying zero for injection)
//   t_spmv           — cuSPARSE block-diagonal SpMV: Ibus = Ybus_batch · V_batch
//   t_fill_F         — fill_FP_kernel + fill_FQ_kernel
//   t_fill_J         — fill_J_kernel
//   t_first_factorize — cuDSS FACTORIZATION, single call at chunk 0 / iter 0
//   t_refactorize     — cuDSS REFACTORIZATION, all subsequent calls (accumulated)
//   t_solve          — cuDSS SOLVE
//   t_update_V       — update_Va_kernel + update_Vm_kernel
//   t_residual       — final SpMV + fill_F + compute_residuals_kernel (post-loop)
//   t_store_V        — cudaMemcpyAsync: chunk d_V_batch → d_V_results
//   t_flow_computation — compute_branch_flows_kernel (+ a small incidental H→D
//                      upload of the tripped-branch zero-index list, for
//                      contingency only — kept bundled since it's tiny and
//                      tightly coupled to the flow kernel)
//
// Metadata:
//   n_contingencies  — total number of contingencies
//   n_chunks         — ceil(n_contingencies / chunk_size)
//   chunk_size       — batch_size (maximum, last chunk may be smaller)
//   nb_iter          — fixed NR iterations per chunk
//
// Coarse aggregation (read-only, computed — no new state):
//   t_cpu_preprocess_ms() — alias for t_preprocess_ms (already pure CPU).
//   t_host_to_device_ms() — t_alloc_ms + t_source_init_ms + t_branch_data_upload_ms.
//                           Approximate: t_source_init_ms includes a D→D Ybus
//                           tile for the injection sweep (not host-touching),
//                           bundled here rather than with GPU compute since
//                           it's one-time setup, not a per-chunk kernel.
//   t_device_to_host_ms() — t_copy_flows_to_host_ms + t_copy_V_to_host_ms
//                           + t_copy_residuals_to_host_ms (exact — pure D→H).
//   t_gpu_compute_ms()    — t_base_case_solve_only_ms + t_analysis_ms
//                           + t_chunks_total_wall_ms().
//   t_grand_total_ms()    — sum of the four buckets above; should match an
//                           external stopwatch wrapped around construction →
//                           first run() → compute_flows() → .to_numpy() calls.
//                           Caveat: t_base_case_solve_only_ms (like
//                           t_base_case_ms) is captured once at construction;
//                           a second run() on the same solver (reusing the
//                           base case) will still report it, overstating that
//                           call's true incremental wall time by roughly that
//                           amount.
// ---------------------------------------------------------------------------
struct BatchTimings {
    // --- one-time setup ---
    double t_base_case_ms         = 0.;
    double t_preprocess_ms        = 0.;
    double t_alloc_ms             = 0.;
    double t_analysis_ms          = 0.;
    double t_source_init_ms       = 0.;
    double t_branch_data_upload_ms = 0.;
    double t_base_case_solve_only_ms = 0.;
    double t_copy_flows_to_host_ms = 0.;
    double t_copy_V_to_host_ms     = 0.;
    double t_copy_residuals_to_host_ms = 0.;
    // compute_limit_violations only; zero unless enabled.
    double t_violation_setup_ms         = 0.;   // H->D limits upload + buffer alloc (set_violation_limits())
    double t_copy_violations_to_host_ms = 0.;   // D->H across the 10 get_violation_*() accessors

    // --- per-chunk accumulated (TimingEntry: gpu + wall) ---
    TimingEntry t_tile_V;
    TimingEntry t_tile_Ybus;
    TimingEntry t_patch_Ybus;
    TimingEntry t_tile_Sbus;         // injection only; zero for contingency
    TimingEntry t_spmv;
    TimingEntry t_fill_F;
    TimingEntry t_fill_J;
    TimingEntry t_first_factorize;   // single FACTORIZATION call (chunk 0, iter 0)
    TimingEntry t_refactorize;       // all subsequent REFACTORIZATION calls
    TimingEntry t_solve;
    TimingEntry t_update_V;
    TimingEntry t_residual;
    TimingEntry t_store_V;
    // check_limit_violations_kernel only; zero unless compute_limit_violations
    // is enabled -- see t_store_V for the rest of that phase (nr_mask_v_nan +
    // the V-results store, both unconditional).
    TimingEntry t_violation_check;
    TimingEntry t_flow_computation;

    // --- metadata ---
    //
    // n_contingencies is the count of batch elements; kept under this name
    // for backwards compatibility with existing Python callers.  For the
    // injection sweep it is n_scenarios.
    int n_contingencies  = 0;
    int n_chunks         = 0;
    int chunk_size       = 0;
    int nb_iter          = 0;
    int n_refactorize    = 0;   // number of refactorize calls (n_chunks * nb_iter - 1)
    int n_disconnected   = 0;   // contingencies skipped (would disconnect the grid)

    // Total wall-clock time for all chunks (excludes one-time setup).
    double t_chunks_total_wall_ms() const {
        return (t_tile_V         + t_tile_Ybus      + t_patch_Ybus
              + t_tile_Sbus      + t_spmv           + t_fill_F
              + t_fill_J         + t_first_factorize + t_refactorize
              + t_solve          + t_update_V       + t_residual
              + t_store_V        + t_violation_check + t_flow_computation).wall_ms;
    }

    // Mean wall time per contingency (across all chunks).
    double t_per_contingency_ms() const {
        return n_contingencies > 0
            ? t_chunks_total_wall_ms() / n_contingencies
            : 0.;
    }

    // --- coarse aggregation (read-only; see note above) ---
    double t_cpu_preprocess_ms() const { return t_preprocess_ms; }

    double t_host_to_device_ms() const {
        return t_alloc_ms + t_source_init_ms + t_branch_data_upload_ms
             + t_violation_setup_ms;
    }

    double t_device_to_host_ms() const {
        return t_copy_flows_to_host_ms + t_copy_V_to_host_ms
             + t_copy_residuals_to_host_ms + t_copy_violations_to_host_ms;
    }

    double t_gpu_compute_ms() const {
        return t_base_case_solve_only_ms + t_analysis_ms + t_chunks_total_wall_ms();
    }

    double t_grand_total_ms() const {
        return t_cpu_preprocess_ms() + t_host_to_device_ms()
             + t_gpu_compute_ms()    + t_device_to_host_ms();
    }
};

// Backwards-compatibility alias: existing C++ code that referenced
// ContingencyTimings continues to compile unchanged.
using ContingencyTimings = BatchTimings;

#endif // TIMING_UTILS_H