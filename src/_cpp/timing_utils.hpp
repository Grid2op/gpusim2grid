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
// ---------------------------------------------------------------------------
struct AcPfTimings {
    // --- one-time setup (wall-clock only, mixed CPU/GPU) ---
    double t_init_ms    = 0.;   // total: see sub-fields below
    double t_build_J_ms = 0.;   // sub-phase: J sparsity + scatter maps (CPU)
    double t_upload_ms  = 0.;   // sub-phase: H→D data transfers
    double t_analyze_ms = 0.;   // cuDSS symbolic analysis
 
    // --- per-iteration totals (GPU + wall) ---
    TimingEntry t_spmv;
    TimingEntry t_fill_F;
    TimingEntry t_fill_J;
    TimingEntry t_first_factorize;   // FACTORIZATION (iter 0 only)
    TimingEntry t_refactorize;       // REFACTORIZATION (iter > 0)
    TimingEntry t_mismatch;
    TimingEntry t_solve;
    TimingEntry t_update_V;

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
//                      and all NR iterations.  The sub-components below are folded
//                      into t_preprocess_ms and t_alloc_ms for a complete picture.
//   t_preprocess_ms  — all CPU preprocessing:
//                        • base-case: pvpq sort + build_J_structure() (scatter maps)
//                        • contingency: resolve_indices + build_flat_patches
//                          + build_blockdiag_csr
//   t_alloc_ms       — all H→D data transfers + device buffer allocation:
//                        • base-case: Ybus values/indices, scatter maps, Sbus,
//                          V_init, J skeleton (from AcPfNrState ctor)
//                        • contingency: flat-patch arrays, block-diag structure,
//                          chunk-sized working buffers
//   t_analysis_ms    — cuDSS ANALYSIS on single-system J (uniform-batch context,
//                      contingency solver only; base-case analysis is inside
//                      t_base_case_ms)
//
// Per-chunk accumulated totals (TimingEntry: gpu_ms + wall_ms):
//   t_tile_V         — cudaMemcpyAsync: d_V_base tiled into d_V_batch
//   t_tile_Ybus      — cudaMemcpyAsync: base Ybus values tiled into d_Ybus_values_batch
//   t_patch_Ybus     — apply_contingencies_kernel: subtract Ybus deltas
//   t_spmv           — cuSPARSE block-diagonal SpMV: Ibus = Ybus_batch · V_batch
//   t_fill_F         — fill_FP_kernel + fill_FQ_kernel
//   t_fill_J         — fill_J_kernel
//   t_first_factorize — cuDSS FACTORIZATION, single call at chunk 0 / iter 0
//   t_refactorize     — cuDSS REFACTORIZATION, all subsequent calls (accumulated)
//   t_solve          — cuDSS SOLVE
//   t_update_V       — update_Va_kernel + update_Vm_kernel
//   t_residual       — final SpMV + fill_F + compute_residuals_kernel (post-loop)
//   t_store_V        — cudaMemcpyAsync: chunk d_V_batch → d_V_results
//
// Metadata:
//   n_contingencies  — total number of contingencies
//   n_chunks         — ceil(n_contingencies / chunk_size)
//   chunk_size       — batch_size (maximum, last chunk may be smaller)
//   nb_iter          — fixed NR iterations per chunk
// ---------------------------------------------------------------------------
struct BatchTimings {
    // --- one-time setup ---
    double t_base_case_ms         = 0.;
    double t_preprocess_ms        = 0.;
    double t_alloc_ms             = 0.;
    double t_analysis_ms          = 0.;
    double t_copy_flows_to_host_ms = 0.;

    // --- per-chunk accumulated (TimingEntry: gpu + wall) ---
    TimingEntry t_tile_V;
    TimingEntry t_tile_Ybus;
    TimingEntry t_patch_Ybus;
    TimingEntry t_spmv;
    TimingEntry t_fill_F;
    TimingEntry t_fill_J;
    TimingEntry t_first_factorize;   // single FACTORIZATION call (chunk 0, iter 0)
    TimingEntry t_refactorize;       // all subsequent REFACTORIZATION calls
    TimingEntry t_solve;
    TimingEntry t_update_V;
    TimingEntry t_residual;
    TimingEntry t_store_V;
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
              + t_spmv           + t_fill_F          + t_fill_J
              + t_first_factorize + t_refactorize    + t_solve
              + t_update_V       + t_residual        + t_store_V
              + t_flow_computation).wall_ms;
    }

    // Mean wall time per contingency (across all chunks).
    double t_per_contingency_ms() const {
        return n_contingencies > 0
            ? t_chunks_total_wall_ms() / n_contingencies
            : 0.;
    }
};

// Backwards-compatibility alias: existing C++ code that referenced
// ContingencyTimings continues to compile unchanged.
using ContingencyTimings = BatchTimings;

#endif // TIMING_UTILS_H