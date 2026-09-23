// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "raw_cudss_solve.hpp"
#include "cuda_utils.h"
#include "dtypes.hpp"
#include "contingency/strategies/cudss_batch_solver.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#define CHK_CUDA_RAW(call)                                                    \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(                                        \
                std::string("[solve_cudss_raw] CUDA error: ")                 \
                + cudaGetErrorString(_e));                                    \
    } while (0)

#define CHK_DSS_RAW(call)                                                     \
    do {                                                                      \
        cudssStatus_t _s = (call);                                            \
        if (_s != CUDSS_STATUS_SUCCESS)                                       \
            throw std::runtime_error(                                        \
                std::string("[solve_cudss_raw] cuDSS error, status=")         \
                + std::to_string(_s));                                        \
    } while (0)

std::vector<double> solve_cudss_raw(
    int dim,
    const std::vector<int>& indptr,
    const std::vector<int>& indices,
    const std::vector<double>& data,
    const std::vector<double>& rhs,
    int device,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg)
{
    if (dim <= 0)
        throw std::runtime_error("solve_cudss_raw: dim must be > 0");
    if (static_cast<int>(indptr.size()) != dim + 1)
        throw std::runtime_error("solve_cudss_raw: indptr size must be dim+1");
    if (indices.size() != data.size())
        throw std::runtime_error("solve_cudss_raw: indices and data must have the same size");
    if (static_cast<int>(rhs.size()) != dim)
        throw std::runtime_error("solve_cudss_raw: rhs size must equal dim");
    const int nnz = static_cast<int>(data.size());

    if (device >= 0) {
        int count = 0;
        CHK_CUDA_RAW(cudaGetDeviceCount(&count));
        if (device >= count)
            throw std::runtime_error(
                "solve_cudss_raw: device index " + std::to_string(device)
                + " out of range (device count = " + std::to_string(count) + ")");
        CHK_CUDA_RAW(cudaSetDevice(device));
    }

    CudaStream cs;

    // Narrow to cuda_real_type (float or double per build), same as the rest
    // of the codebase (see AcPfNrSession::get_J() / solve_cudss()).
    std::vector<cuda_real_type> h_data(data.begin(), data.end());
    std::vector<cuda_real_type> h_rhs(rhs.begin(), rhs.end());

    thrust::device_vector<int>            d_outer(indptr.begin(), indptr.end());
    thrust::device_vector<int>            d_inner(indices.begin(), indices.end());
    thrust::device_vector<cuda_real_type> d_values(h_data.begin(), h_data.end());
    thrust::device_vector<cuda_real_type> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<cuda_real_type> d_sol(static_cast<size_t>(dim), cuda_real_type(0));

    CudssContext dss;
    CHK_DSS_RAW(cudssCreate(&dss.handle));
    CHK_DSS_RAW(cudssSetStream(dss.handle, cs));
    CHK_DSS_RAW(cudssConfigCreate(&dss.config));
    CHK_DSS_RAW(cudssDataCreate(dss.handle, &dss.data));
    dss.set_reordering_alg(reordering_alg);
    dss.set_matching_alg(matching_alg);
    dss.set_pivot_epsilon_alg(pivot_epsilon_alg);

    CudssDescriptor dss_A, dss_x, dss_b;
    dss_A.create_csr(dim, nnz,
        thrust::raw_pointer_cast(d_outer.data()),
        thrust::raw_pointer_cast(d_inner.data()),
        thrust::raw_pointer_cast(d_values.data()));
    dss_x.create_dn(dim, thrust::raw_pointer_cast(d_sol.data()));
    dss_b.create_dn(dim, thrust::raw_pointer_cast(d_rhs.data()));

    // Same three-call sequence gpusim2grid's real NR pipeline uses (analyze
    // -> factorize -> solve), on the exact same CudssContext/CudssDescriptor
    // wrapper -- see cuda_utils.h and the analyze() fix in this file's commit.
    dss.analyze(dss_A, dss_x, dss_b);
    dss.factorize(dss_A, dss_x, dss_b);
    dss.solve(dss_A, dss_x, dss_b);
    cs.synchronize();

    thrust::host_vector<cuda_real_type> h_sol(d_sol);
    return std::vector<double>(h_sol.begin(), h_sol.end());
}

CudssBatchBenchResult benchmark_cudss_batch_raw(
    int dim,
    const std::vector<int>& indptr,
    const std::vector<int>& indices,
    const std::vector<double>& data,
    int batch_size,
    int n_refactorize,
    int device,
    ReorderingAlg reordering_alg,
    MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg)
{
    if (dim <= 0 || batch_size <= 0 || n_refactorize < 0)
        throw std::runtime_error(
            "benchmark_cudss_batch_raw: dim and batch_size must be > 0, n_refactorize >= 0");
    if (static_cast<int>(indptr.size()) != dim + 1)
        throw std::runtime_error("benchmark_cudss_batch_raw: indptr size must be dim+1");
    if (indices.size() != data.size())
        throw std::runtime_error("benchmark_cudss_batch_raw: indices and data must have the same size");
    const int nnz = static_cast<int>(data.size());
    if (indptr[static_cast<size_t>(dim)] != nnz)
        throw std::runtime_error("benchmark_cudss_batch_raw: indptr[dim] must equal nnz");
    if (device >= 0) CHK_CUDA_RAW(cudaSetDevice(device));

    const size_t B = static_cast<size_t>(batch_size);
    const size_t n = static_cast<size_t>(dim), m = static_cast<size_t>(nnz);

    // b = A * 1 (exact solution: ones), computed once on the host.
    std::vector<double> b(n, 0.);
    for (size_t r = 0; r < n; ++r)
        for (int k = indptr[r]; k < indptr[r + 1]; ++k)
            b[r] += data[static_cast<size_t>(k)];

    std::vector<cuda_real_type> h_vals(B * m), h_rhs(B * n);
    for (size_t i = 0; i < B; ++i) {
        std::copy(data.begin(), data.end(), h_vals.begin() + i * m);
        std::copy(b.begin(), b.end(), h_rhs.begin() + i * n);
    }

    CudaStream cs;
    thrust::device_vector<int>            d_outer(indptr.begin(), indptr.end());
    thrust::device_vector<int>            d_inner(indices.begin(), indices.end());
    thrust::device_vector<cuda_real_type> d_vals(h_vals.begin(), h_vals.end());
    thrust::device_vector<cuda_real_type> d_rhs0(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<cuda_real_type> d_rhs(B * n), d_x(B * n, cuda_real_type(0));

    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    CudssBatchBenchResult res;
    res.dim = dim; res.nnz = nnz; res.batch_size = batch_size;

    CudssBatchSolver solver;
    cs.synchronize();
    auto t0 = clk::now();
    solver.initialize(batch_size, dim, nnz,
                      thrust::raw_pointer_cast(d_outer.data()),
                      thrust::raw_pointer_cast(d_inner.data()),
                      thrust::raw_pointer_cast(d_vals.data()),
                      thrust::raw_pointer_cast(d_rhs.data()),
                      thrust::raw_pointer_cast(d_x.data()),
                      cs, reordering_alg, matching_alg, pivot_epsilon_alg);
    solver.set_values(thrust::raw_pointer_cast(d_vals.data()));
    solver.prepare_factorization();   // non-uniform modes: the deferred ANALYSIS
    cs.synchronize();
    res.context_init_ms = solver.context_init_ms();
    res.analysis_ms     = ms_since(t0) - res.context_init_ms;
    solver.refresh_factor_stats();
    const CudssFactorStats& fs = solver.factor_stats();
    res.lu_nnz               = fs.lu_nnz;
    res.mem_device_permanent = fs.mem_device_permanent;
    res.mem_device_peak      = fs.mem_device_peak;
    res.mem_host_permanent   = fs.mem_host_permanent;
    res.mem_host_peak        = fs.mem_host_peak;

    t0 = clk::now();
    solver.factor();
    cs.synchronize();
    res.factorize_ms = ms_since(t0);

    // cuDSS may use b as workspace: restore it before every SOLVE (untimed).
    auto run_solve = [&]() {
        thrust::copy(d_rhs0.begin(), d_rhs0.end(), d_rhs.begin());
        cs.synchronize();
        auto ts = clk::now();
        solver.solve();
        cs.synchronize();
        return ms_since(ts);
    };
    double t_solve = run_solve();
    int n_solve = 1;
    for (int it = 0; it < n_refactorize; ++it) {
        t0 = clk::now();
        solver.refactor();
        cs.synchronize();
        res.refactorize_ms += ms_since(t0);
        t_solve += run_solve();
        ++n_solve;
    }
    res.n_refactorize = n_refactorize;
    if (n_refactorize > 0) res.refactorize_ms /= n_refactorize;
    res.solve_ms = t_solve / n_solve;

    // Sanity check of every slot's solution against b (host, double).
    thrust::host_vector<cuda_real_type> h_x(d_x);
    double bnorm = 0.;
    for (double v : b) bnorm = std::max(bnorm, std::abs(v));
    if (bnorm == 0.) bnorm = 1.;
    for (size_t i = 0; i < B; ++i) {
        const cuda_real_type* x = &h_x[i * n];
        bool finite = true;
        for (size_t r = 0; r < n && finite; ++r) finite = std::isfinite(x[r]);
        if (!finite) { ++res.n_nonfinite_slots; continue; }
        double worst = 0.;
        for (size_t r = 0; r < n; ++r) {
            double ax = 0.;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k)
                ax += data[static_cast<size_t>(k)] * static_cast<double>(x[indices[static_cast<size_t>(k)]]);
            worst = std::max(worst, std::abs(ax - b[r]));
        }
        res.max_rel_residual = std::max(res.max_rel_residual, worst / bnorm);
    }
    return res;
}
