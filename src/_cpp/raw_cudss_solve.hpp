// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef RAW_CUDSS_SOLVE_H
#define RAW_CUDSS_SOLVE_H

#include <string>
#include <vector>

#include "reordering_alg.hpp"
#include "matching_alg.hpp"
#include "pivot_epsilon_alg.hpp"

// Solve J*dx = rhs with gpusim2grid's own cuDSS wrapper (analyze -> factorize
// -> solve, CudssContext/CudssDescriptor from cuda_utils.h), completely
// decoupled from any power-flow/grid construction: J is supplied directly as
// a CSR triplet. Meant for validating cuDSS itself against an arbitrary
// sparse system -- e.g. a (J, F) pair dumped via AcPfNrSession::get_J() /
// get_F() -- without rebuilding a grid/session at all. See
// repro_cudss_bug_standalone.py.
//
//   dim            matrix dimension (J is dim x dim)
//   indptr         CSR row pointer, size dim+1
//   indices        CSR column indices, size nnz
//   data           CSR values, size nnz (narrowed to cuda_real_type internally,
//                  same as the rest of the codebase)
//   rhs            right-hand side, size dim
//   device         CUDA device ordinal, or -1 to use the current device
//   reordering_alg CUDSS_CONFIG_REORDERING_ALG choice for the ANALYSIS phase
//   matching_alg   CUDSS_CONFIG_MATCHING_ALG choice for the ANALYSIS phase
//   pivot_epsilon_alg CUDSS_CONFIG_PIVOT_EPSILON_ALG choice for the ANALYSIS phase
//   ir_n_steps     CUDSS_CONFIG_IR_N_STEPS, iterative refinement steps per SOLVE
//
// Returns dx (host, double), size dim.
std::vector<double> solve_cudss_raw(
    int dim,
    const std::vector<int>& indptr,
    const std::vector<int>& indices,
    const std::vector<double>& data,
    const std::vector<double>& rhs,
    int device = -1,
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
    int ir_n_steps = 0);

// Result of benchmark_cudss_batch_raw(): wall-clock ms (stream-synchronized)
// of each cuDSS phase on a uniform batch, the factor statistics cuDSS reports
// after ANALYSIS, and a sanity check of the returned solutions.
struct CudssBatchBenchResult {
    int       dim = 0, nnz = 0, batch_size = 0;
    double    context_init_ms = 0.;   // handle/config/data creation
    double    analysis_ms     = 0.;   // CUDSS_PHASE_ANALYSIS (once)
    double    factorize_ms    = 0.;   // first CUDSS_PHASE_FACTORIZATION
    double    refactorize_ms  = 0.;   // mean CUDSS_PHASE_REFACTORIZATION
    double    solve_ms        = 0.;   // mean CUDSS_PHASE_SOLVE
    int       n_refactorize   = 0;
    long long lu_nnz               = -1;  // CUDSS_DATA_LU_NNZ
    long long mem_device_permanent = -1;  // CUDSS_DATA_MEMORY_ESTIMATES[0..3], bytes
    long long mem_device_peak      = -1;
    long long mem_host_permanent   = -1;
    long long mem_host_peak        = -1;
    double    max_rel_residual = 0.;      // max over batch slots of ||A x - b||inf / ||b||inf
    int       n_nonfinite_slots = 0;      // slots whose solution has a NaN/inf
    int       n_groups        = 1;
    double    group_cycle_ms  = 0.;       // mean per group of rebind + REFACTORIZATION + SOLVE,
                                          // one stream sync per cycle over the groups
    int         n_handles       = 1;
    std::string handle_launch   = "serial";
    double      handle_setup_ms = 0.;     // creation + ANALYSIS + FACTORIZATION of handles 1..n-1
    double      handle_round_ms = 0.;     // mean wall ms of one round over all the handles
};

// Time gpusim2grid's own CudssBatchSolver (same code path, same batch-mode
// environment variables, same cuDSS config knobs as the NR driver) on an
// arbitrary CSR matrix replicated batch_size times with identical values:
// ANALYSIS, one FACTORIZATION, n_refactorize REFACTORIZATION + SOLVE pairs.
// rhs = A * 1, so the exact solution is all ones. For sizing what a Jacobian
// that no session can build yet (e.g. a structural superset) would cost.
// n_groups > 1 (uniform mode): the buffers hold n_groups slices of batch_size
// matrices and the handle is cycled over them (CudssBatchSolver::bind_slice,
// then REFACTORIZATION + SOLVE per slice), as the continuous schedule's tail
// does; group g's rhs is (g + 1)·A·1, so a slice solved against another
// slice's data fails the residual check.
// n_handles > 1 (exclusive with n_groups > 1): n_handles independent handles
// of batch_size matrices, handle h on slice h, each its own ANALYSIS. One
// round = for every handle, rhs reset (device copy) + REFACTORIZATION + SOLVE;
// handle_launch says how: "serial" (one stream, one host thread), "streams"
// (one stream per handle, one host thread, one sync over all of them per
// round) or "threads" (one stream and one host thread per handle, each running
// its own rounds with its own sync, the wall clock spanning all of them).
CudssBatchBenchResult benchmark_cudss_batch_raw(
    int dim,
    const std::vector<int>& indptr,
    const std::vector<int>& indices,
    const std::vector<double>& data,
    int batch_size,
    int n_refactorize = 4,
    int device = -1,
    ReorderingAlg reordering_alg = ReorderingAlg::Default,
    MatchingAlg matching_alg = MatchingAlg::None,
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default,
    int ir_n_steps = 0,
    int n_groups = 1,
    int n_handles = 1,
    const std::string& handle_launch = "serial");

// Result of calibrate_tail_groups(): what one Newton round's linear solve
// costs on the full batch and on one group of the continuous tail.
struct TailCalibration {
    int    batch_size = 0, group_size = 0, n_repeat = 0;
    double t_full_ms  = 0.;   // mean wall ms of REFACTORIZATION + SOLVE, batch_size matrices
    double t_group_ms = 0.;   // the same, group_size matrices
    int    max_groups = 0;    // floor(t_full_ms / t_group_ms); 0 when group_size >= batch_size
};

struct AcPfNrState;

// The value of tail_max_groups the continuous schedule's own first round
// could measure (docs/dev_notes/outer_loops_performance.md, step 3): the base
// case's J tiled batch_size times, factorized once, then n_repeat
// REFACTORIZATION + SOLVE on a uniform handle of batch_size matrices and on
// one of group_size matrices, each stream-synchronized; the tail's groups pay
// as long as ceil(n_live / G) * t_group < t_full, i.e. up to max_groups
// groups. cuDSS only: the tail also runs the Newton kernels on fewer slots,
// so the value is conservative. Allocates batch_size systems (values,
// right-hand sides, solutions) and their factors for the duration of the call.
TailCalibration calibrate_tail_groups(
    const AcPfNrState& base, int batch_size, int group_size, int n_repeat,
    ReorderingAlg reordering_alg, MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg, int ir_n_steps);

#endif  // RAW_CUDSS_SOLVE_H
