// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef RAW_CUDSS_SOLVE_H
#define RAW_CUDSS_SOLVE_H

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
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default);

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
};

// Time gpusim2grid's own CudssBatchSolver (same code path, same batch-mode
// environment variables, same cuDSS config knobs as the NR driver) on an
// arbitrary CSR matrix replicated batch_size times with identical values:
// ANALYSIS, one FACTORIZATION, n_refactorize REFACTORIZATION + SOLVE pairs.
// rhs = A * 1, so the exact solution is all ones. For sizing what a Jacobian
// that no session can build yet (e.g. a structural superset) would cost.
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
    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default);

#endif  // RAW_CUDSS_SOLVE_H
