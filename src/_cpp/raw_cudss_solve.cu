// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "raw_cudss_solve.hpp"
#include "cuda_utils.h"
#include "dtypes.hpp"
#include "contingency/strategies/cudss_batch_solver.cuh"
#include "acpf_nr_state.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <cudss.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

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
    PivotEpsilonAlg pivot_epsilon_alg,
    int ir_n_steps)
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
    dss.set_ir_n_steps(ir_n_steps);

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
    PivotEpsilonAlg pivot_epsilon_alg,
    int ir_n_steps,
    int n_groups,
    int n_handles,
    const std::string& handle_launch)
{
    if (dim <= 0 || batch_size <= 0 || n_refactorize < 0 || n_groups <= 0 || n_handles <= 0)
        throw std::runtime_error(
            "benchmark_cudss_batch_raw: dim, batch_size, n_groups and n_handles must be > 0, "
            "n_refactorize >= 0");
    if (n_groups > 1 && n_handles > 1)
        throw std::runtime_error("benchmark_cudss_batch_raw: n_groups and n_handles are exclusive");
    if (handle_launch != "serial" && handle_launch != "streams" && handle_launch != "threads")
        throw std::runtime_error(
            "benchmark_cudss_batch_raw: handle_launch must be 'serial', 'streams' or 'threads', got '"
            + handle_launch + "'");
    if (static_cast<int>(indptr.size()) != dim + 1)
        throw std::runtime_error("benchmark_cudss_batch_raw: indptr size must be dim+1");
    if (indices.size() != data.size())
        throw std::runtime_error("benchmark_cudss_batch_raw: indices and data must have the same size");
    const int nnz = static_cast<int>(data.size());
    if (indptr[static_cast<size_t>(dim)] != nnz)
        throw std::runtime_error("benchmark_cudss_batch_raw: indptr[dim] must equal nnz");
    if (device >= 0) CHK_CUDA_RAW(cudaSetDevice(device));

    const size_t B = static_cast<size_t>(batch_size);
    const size_t G = static_cast<size_t>(n_groups);
    const size_t H = static_cast<size_t>(n_handles);
    const size_t NS = std::max(G, H);   // slices of batch_size matrices in the buffers
    const size_t n = static_cast<size_t>(dim), m = static_cast<size_t>(nnz);

    // b = A * 1 (exact solution: ones), computed once on the host. Slice g's
    // right-hand side is (g + 1)·b, so its exact solution is g + 1: a slice
    // solved against another slice's b (a rebinding that did not take) shows.
    std::vector<double> b(n, 0.);
    for (size_t r = 0; r < n; ++r)
        for (int k = indptr[r]; k < indptr[r + 1]; ++k)
            b[r] += data[static_cast<size_t>(k)];

    std::vector<cuda_real_type> h_vals(NS * B * m), h_rhs(NS * B * n);
    for (size_t i = 0; i < NS * B; ++i) {
        const double scale = static_cast<double>(i / B + 1);
        std::copy(data.begin(), data.end(), h_vals.begin() + i * m);
        for (size_t r = 0; r < n; ++r)
            h_rhs[i * n + r] = static_cast<cuda_real_type>(scale * b[r]);
    }

    CudaStream cs;
    thrust::device_vector<int>            d_outer(indptr.begin(), indptr.end());
    thrust::device_vector<int>            d_inner(indices.begin(), indices.end());
    thrust::device_vector<cuda_real_type> d_vals(h_vals.begin(), h_vals.end());
    thrust::device_vector<cuda_real_type> d_rhs0(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<cuda_real_type> d_rhs(NS * B * n), d_x(NS * B * n, cuda_real_type(0));
    cuda_real_type* vals = thrust::raw_pointer_cast(d_vals.data());
    cuda_real_type* rhs  = thrust::raw_pointer_cast(d_rhs.data());
    cuda_real_type* x    = thrust::raw_pointer_cast(d_x.data());

    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    CudssBatchBenchResult res;
    res.dim = dim; res.nnz = nnz; res.batch_size = batch_size; res.n_groups = n_groups;
    res.n_handles = n_handles; res.handle_launch = handle_launch;

    CudssBatchSolver solver;
    cs.synchronize();
    auto t0 = clk::now();
    solver.initialize(batch_size, dim, nnz,
                      thrust::raw_pointer_cast(d_outer.data()),
                      thrust::raw_pointer_cast(d_inner.data()),
                      vals, rhs, x,
                      cs, reordering_alg, matching_alg, pivot_epsilon_alg, ir_n_steps);
    solver.set_values(vals);
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

    // Point the handle at group g (n_groups > 1 only: uniform mode).
    auto bind = [&](size_t g) {
        if (G > 1) solver.bind_slice(vals + g * B * m, rhs + g * B * n, x + g * B * n);
    };
    // cuDSS may use b as workspace: restore it before every SOLVE (untimed).
    auto restore_rhs = [&]() {
        thrust::copy(d_rhs0.begin(), d_rhs0.end(), d_rhs.begin());
        cs.synchronize();
    };

    restore_rhs();
    t0 = clk::now();
    solver.factor();
    cs.synchronize();
    res.factorize_ms = ms_since(t0);

    // Per-phase timings, stream-synchronized after every call: the first
    // group only (the one the handle was analysed and factorized on).
    auto run_solve = [&]() {
        restore_rhs();
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

    // The cycle a tail round does: for every group, rebind + REFACTORIZATION
    // + SOLVE, one stream synchronization at the end of the cycle (what the
    // NR loop pays, host launch overhead included). Every slice's solution
    // is then the last cycle's.
    const int n_cycles = std::max(1, n_refactorize);
    double t_cycles = 0.;
    for (int it = 0; it < n_cycles; ++it) {
        restore_rhs();
        t0 = clk::now();
        for (size_t g = 0; g < G; ++g) {
            bind(g);
            solver.refactor();
            solver.solve();
        }
        cs.synchronize();
        t_cycles += ms_since(t0);
    }
    bind(0);
    res.group_cycle_ms = t_cycles / (static_cast<double>(n_cycles) * static_cast<double>(G));

    // Independent handles (n_handles > 1): handle 0 is the one above, on cs
    // and slice 0; handle h owns slice h, its own stream unless "serial".
    if (H > 1) {
        const bool own_streams = handle_launch != "serial";
        const cuda_real_type* rhs0 = thrust::raw_pointer_cast(d_rhs0.data());
        std::vector<CudaStream> streams(own_streams ? H - 1 : 0);
        std::vector<cudaStream_t> st(H, cs);
        for (size_t h = 1; h < H && own_streams; ++h) st[h] = streams[h - 1];
        std::vector<std::unique_ptr<CudssBatchSolver>> hs(H);
        restore_rhs();
        t0 = clk::now();
        for (size_t h = 1; h < H; ++h) {
            hs[h] = std::make_unique<CudssBatchSolver>();
            hs[h]->initialize(batch_size, dim, nnz,
                              thrust::raw_pointer_cast(d_outer.data()),
                              thrust::raw_pointer_cast(d_inner.data()),
                              vals + h * B * m, rhs + h * B * n, x + h * B * n,
                              st[h], reordering_alg, matching_alg, pivot_epsilon_alg, ir_n_steps);
            hs[h]->set_values(vals + h * B * m);
            hs[h]->prepare_factorization();
            hs[h]->factor();
        }
        for (cudaStream_t s : st) CHK_CUDA_RAW(cudaStreamSynchronize(s));
        res.handle_setup_ms = ms_since(t0);
        auto handle = [&](size_t h) -> CudssBatchSolver& { return h == 0 ? solver : *hs[h]; };

        // One handle's share of a round. The rhs copy stands for the NR
        // loop's refill of F (cuDSS may use b as workspace).
        auto round = [&](size_t h) {
            CHK_CUDA_RAW(cudaMemcpyAsync(rhs + h * B * n, rhs0 + h * B * n,
                                         B * n * sizeof(cuda_real_type),
                                         cudaMemcpyDeviceToDevice, st[h]));
            handle(h).refactor();
            handle(h).solve();
        };

        if (handle_launch == "threads") {
            int dev = 0;
            CHK_CUDA_RAW(cudaGetDevice(&dev));
            std::atomic<bool> go{false};
            std::vector<std::exception_ptr> err(H);
            std::vector<std::thread> workers;
            for (size_t h = 0; h < H; ++h)
                workers.emplace_back([&, h]() {
                    try {
                        CHK_CUDA_RAW(cudaSetDevice(dev));
                        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                        for (int it = 0; it < n_cycles; ++it) {
                            round(h);
                            CHK_CUDA_RAW(cudaStreamSynchronize(st[h]));
                        }
                    } catch (...) { err[h] = std::current_exception(); }
                });
            t0 = clk::now();
            go.store(true, std::memory_order_release);
            for (auto& w : workers) w.join();
            res.handle_round_ms = ms_since(t0) / n_cycles;
            for (auto& e : err) if (e) std::rethrow_exception(e);
        } else {
            t0 = clk::now();
            for (int it = 0; it < n_cycles; ++it) {
                for (size_t h = 0; h < H; ++h) round(h);
                for (size_t h = 0; h < (own_streams ? H : 1); ++h)
                    CHK_CUDA_RAW(cudaStreamSynchronize(st[h]));
            }
            res.handle_round_ms = ms_since(t0) / n_cycles;
        }
    }

    // Sanity check of every slot's solution against its b (host, double).
    thrust::host_vector<cuda_real_type> h_x(d_x);
    double bnorm = 0.;
    for (double v : b) bnorm = std::max(bnorm, std::abs(v));
    if (bnorm == 0.) bnorm = 1.;
    for (size_t i = 0; i < NS * B; ++i) {
        const double scale = static_cast<double>(i / B + 1);
        const cuda_real_type* xi = &h_x[i * n];
        bool finite = true;
        for (size_t r = 0; r < n && finite; ++r) finite = std::isfinite(xi[r]);
        if (!finite) { ++res.n_nonfinite_slots; continue; }
        double worst = 0.;
        for (size_t r = 0; r < n; ++r) {
            double ax = 0.;
            for (int k = indptr[r]; k < indptr[r + 1]; ++k)
                ax += data[static_cast<size_t>(k)] * static_cast<double>(xi[indices[static_cast<size_t>(k)]]);
            worst = std::max(worst, std::abs(ax - scale * b[r]));
        }
        res.max_rel_residual = std::max(res.max_rel_residual, worst / (scale * bnorm));
    }
    return res;
}

// dst[b * n + i] = src[i], b < batch
__global__ static void tile_values_kernel(cuda_real_type* __restrict__ dst,
                                          const cuda_real_type* __restrict__ src,
                                          int n, int batch)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<ptrdiff_t>(n) * batch) return;
    dst[tid] = src[tid % n];
}

TailCalibration calibrate_tail_groups(
    const AcPfNrState& base, int batch_size, int group_size, int n_repeat,
    ReorderingAlg reordering_alg, MatchingAlg matching_alg,
    PivotEpsilonAlg pivot_epsilon_alg, int ir_n_steps)
{
    if (batch_size < 1 || group_size < 1 || n_repeat < 1)
        throw std::invalid_argument(
            "calibrate_tail_groups: batch_size, group_size and n_repeat must be >= 1");
    const int dim = base.dim_J, nnz = base.nnz_J;
    const thrust::device_vector<cuda_real_type>& J =
        base.d_J_values_base.empty() ? base.d_J_values : base.d_J_values_base;
    if (dim <= 0 || static_cast<int>(J.size()) < nnz)
        throw std::runtime_error("calibrate_tail_groups: the base state holds no Jacobian");

    TailCalibration res;
    res.batch_size = batch_size;
    res.group_size = group_size;
    res.n_repeat   = n_repeat;

    const size_t S = static_cast<size_t>(batch_size);
    CudaStream cs;
    thrust::device_vector<cuda_real_type> d_vals(S * nnz), d_rhs(S * dim), d_x(S * dim);
    const int BS_T = 256;
    tile_values_kernel<<<static_cast<unsigned>((S * nnz + BS_T - 1) / BS_T), BS_T, 0, static_cast<cudaStream_t>(cs)>>>(
        thrust::raw_pointer_cast(d_vals.data()), thrust::raw_pointer_cast(J.data()), nnz, batch_size);
    CHK_CUDA_RAW(cudaGetLastError());
    int* outer = const_cast<int*>(thrust::raw_pointer_cast(base.d_J_outer.data()));
    int* inner = const_cast<int*>(thrust::raw_pointer_cast(base.d_J_inner.data()));

    using clk = std::chrono::steady_clock;
    // one handle of B matrices on the first B systems: factorized once, then
    // the mean of n_repeat REFACTORIZATION + SOLVE (b refilled untimed: cuDSS
    // may use it as workspace)
    auto time_handle = [&](int B) {
        CudssBatchSolver solver;
        solver.initialize(B, dim, nnz, outer, inner,
                          thrust::raw_pointer_cast(d_vals.data()),
                          thrust::raw_pointer_cast(d_rhs.data()),
                          thrust::raw_pointer_cast(d_x.data()),
                          cs, reordering_alg, matching_alg, pivot_epsilon_alg, ir_n_steps);
        thrust::fill(thrust::cuda::par.on(static_cast<cudaStream_t>(cs)), d_rhs.begin(), d_rhs.end(), cuda_real_type(1));
        solver.factor();
        solver.solve();
        cs.synchronize();
        double total = 0.;
        for (int it = 0; it < n_repeat; ++it) {
            thrust::fill(thrust::cuda::par.on(static_cast<cudaStream_t>(cs)), d_rhs.begin(), d_rhs.end(), cuda_real_type(1));
            cs.synchronize();
            const auto t0 = clk::now();
            solver.refactor();
            solver.solve();
            cs.synchronize();
            total += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        }
        return total / n_repeat;
    };
    res.t_full_ms  = time_handle(batch_size);
    res.t_group_ms = time_handle(std::min(group_size, batch_size));
    res.max_groups = (group_size < batch_size && res.t_group_ms > 0.)
        ? static_cast<int>(std::floor(res.t_full_ms / res.t_group_ms)) : 0;
    return res;
}
