// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// warmup.cu — pay CUDA/cuSPARSE/cuDSS one-time init outside any timed region.
// See warmup.hpp for the rationale.
// =============================================================================

#include "warmup.hpp"

#include "cuda_utils.h"        // CudaStream, CudssContext, CudssDescriptor
#include "cu_complex_utils.h"  // cudaComplexType, CUDA_C_TYPE, h_cplx_one/zero
#include "dtypes.hpp"          // cuda_real_type
#include "timing_utils.hpp"    // ms_since

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cudss.h>

#include <thrust/device_vector.h>

#include <chrono>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define CHK_CUDA_WU(call)                                                     \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
            throw std::runtime_error(                                         \
                std::string("[warmup] CUDA error: ")                          \
                + cudaGetErrorString(_e));                                    \
    } while (0)

#define CHK_CSP_WU(call)                                                      \
    do {                                                                      \
        cusparseStatus_t _s = (call);                                         \
        if (_s != CUSPARSE_STATUS_SUCCESS)                                    \
            throw std::runtime_error(                                         \
                std::string("[warmup] cuSPARSE error, status=")               \
                + std::to_string(static_cast<int>(_s)));                      \
    } while (0)

#define CHK_DSS_WU(call)                                                      \
    do {                                                                      \
        cudssStatus_t _s = (call);                                            \
        if (_s != CUDSS_STATUS_SUCCESS)                                       \
            throw std::runtime_error(                                         \
                std::string("[warmup] cuDSS error, status=")                  \
                + std::to_string(static_cast<int>(_s)));                      \
    } while (0)

namespace {

// -----------------------------------------------------------------------------
// One cuSPARSE complex CSR SpMV on a 2x2 diagonal matrix — the same call shape
// (const CSR + const dense x + dense y, CUSPARSE_SPMV_ALG_DEFAULT) the NR loop
// uses for Ibus = Ybus·V, so the kernels it needs are resident afterwards.
// -----------------------------------------------------------------------------
void warm_cusparse(cudaStream_t cs)
{
    constexpr int n = 2, nnz = 2;

    const std::vector<int> h_outer{0, 1, 2};
    const std::vector<int> h_inner{0, 1};
    std::vector<cudaComplexType> h_vals(nnz);
    std::vector<cudaComplexType> h_x(n);
    for (int i = 0; i < nnz; ++i)
        h_vals[i] = CudaFunHelper::my_make_cuComplex(cuda_real_type(2), cuda_real_type(0));
    for (int i = 0; i < n; ++i)
        h_x[i] = CudaFunHelper::my_make_cuComplex(cuda_real_type(1), cuda_real_type(0));

    thrust::device_vector<int>             d_outer(h_outer.begin(), h_outer.end());
    thrust::device_vector<int>             d_inner(h_inner.begin(), h_inner.end());
    thrust::device_vector<cudaComplexType> d_vals(h_vals.begin(), h_vals.end());
    thrust::device_vector<cudaComplexType> d_x(h_x.begin(), h_x.end());
    thrust::device_vector<cudaComplexType> d_y(static_cast<size_t>(n));

    cusparseHandle_t      handle = nullptr;
    cusparseConstSpMatDescr_t mat = nullptr;
    cusparseConstDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t      vec_y = nullptr;
    void*  buf      = nullptr;
    size_t buf_size = 0;

    CHK_CSP_WU(cusparseCreate(&handle));
    CHK_CSP_WU(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    CHK_CSP_WU(cusparseSetStream(handle, cs));

    CHK_CSP_WU(cusparseCreateConstCsr(
        &mat, n, n, nnz,
        thrust::raw_pointer_cast(d_outer.data()),
        thrust::raw_pointer_cast(d_inner.data()),
        thrust::raw_pointer_cast(d_vals.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_TYPE));
    CHK_CSP_WU(cusparseCreateConstDnVec(
        &vec_x, n, thrust::raw_pointer_cast(d_x.data()), CUDA_C_TYPE));
    CHK_CSP_WU(cusparseCreateDnVec(
        &vec_y, n, thrust::raw_pointer_cast(d_y.data()), CUDA_C_TYPE));

    CHK_CSP_WU(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_cplx_one, mat, vec_x, &h_cplx_zero, vec_y,
        CUDA_C_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));
    if (buf_size > 0) CHK_CUDA_WU(cudaMalloc(&buf, buf_size));

    CHK_CSP_WU(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_cplx_one, mat, vec_x, &h_cplx_zero, vec_y,
        CUDA_C_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, buf));

    CHK_CUDA_WU(cudaStreamSynchronize(cs));

    if (buf) cudaFree(buf);
    if (vec_y) cusparseDestroyDnVec(vec_y);
    if (vec_x) cusparseDestroyDnVec(vec_x);
    if (mat)   cusparseDestroySpMat(mat);
    if (handle) cusparseDestroy(handle);
}

// -----------------------------------------------------------------------------
// One cuDSS analyze -> factorize -> solve on a 2x2 diagonal system.
//
// ubatch_size > 1 exercises the uniform-batch path used by
// ContingencyAnalysisGPU / InjectionSweepGPU; ubatch_size == 1 (config left
// untouched) exercises the single-system path used by AcPfGPU. Both are warmed
// because they can pull in different kernels.
// -----------------------------------------------------------------------------
void warm_cudss(cudaStream_t cs, int ubatch_size)
{
    constexpr int dim = 2, nnz = 2;
    const size_t nb = static_cast<size_t>(ubatch_size);

    const std::vector<int> h_outer{0, 1, 2};
    const std::vector<int> h_inner{0, 1};

    // Structure is shared across the batch; values/rhs/solution are per system.
    thrust::device_vector<int>            d_outer(h_outer.begin(), h_outer.end());
    thrust::device_vector<int>            d_inner(h_inner.begin(), h_inner.end());
    thrust::device_vector<cuda_real_type> d_vals(nb * nnz, cuda_real_type(2));
    thrust::device_vector<cuda_real_type> d_rhs (nb * dim, cuda_real_type(1));
    thrust::device_vector<cuda_real_type> d_sol (nb * dim, cuda_real_type(0));

    CudssContext dss;
    CHK_DSS_WU(cudssCreate(&dss.handle));
    CHK_DSS_WU(cudssSetStream(dss.handle, cs));
    CHK_DSS_WU(cudssConfigCreate(&dss.config));
    CHK_DSS_WU(cudssDataCreate(dss.handle, &dss.data));

    CudssDescriptor A, x, b;
    A.create_csr(dim, nnz,
                 thrust::raw_pointer_cast(d_outer.data()),
                 thrust::raw_pointer_cast(d_inner.data()),
                 thrust::raw_pointer_cast(d_vals.data()));
    x.create_dn(dim, thrust::raw_pointer_cast(d_sol.data()));
    b.create_dn(dim, thrust::raw_pointer_cast(d_rhs.data()));

    if (ubatch_size > 1)
        CHK_DSS_WU(cudssConfigSet(dss.config, CUDSS_CONFIG_UBATCH_SIZE,
                                  &ubatch_size, sizeof(ubatch_size)));

    dss.analyze(A, x, b);
    dss.factorize(A, x, b);
    dss.solve(A, x, b);

    CHK_CUDA_WU(cudaStreamSynchronize(cs));
}

}  // namespace

// =============================================================================
// warmup
// =============================================================================
double warmup(int device)
{
    auto t0 = std::chrono::steady_clock::now();

    // Idempotent by design: callers put this at the top of a benchmark loop or
    // a test, and repeating the cuDSS analyze/factorize/solve below is not
    // free (fixed per-call overhead, ~20 ms even on a 2x2). Once the process
    // has warmed a device, there is nothing left to initialize on it.
    // Deliberately not reset by cudaDeviceReset() -- gpusim2grid never calls
    // it, and a caller that does can simply not rely on this.
    static std::set<int> warmed;
    static std::mutex    warmed_mutex;
    {
        std::lock_guard<std::mutex> guard(warmed_mutex);
        if (warmed.count(device)) return ms_since(t0);
    }

    if (device >= 0) {
        int count = 0;
        CHK_CUDA_WU(cudaGetDeviceCount(&count));
        if (device >= count)
            throw std::runtime_error(
                "warmup: device index " + std::to_string(device)
                + " out of range (device count = " + std::to_string(count) + ")");
        CHK_CUDA_WU(cudaSetDevice(device));
    }

    // Forces CUDA-context creation on the current device before anything else
    // (cudaFree(0) is the canonical no-op that does it).
    CHK_CUDA_WU(cudaFree(nullptr));

    {
        CudaStream cs;
        warm_cusparse(cs);
        warm_cudss(cs, /*ubatch_size=*/1);
        warm_cudss(cs, /*ubatch_size=*/2);
    }

    {
        std::lock_guard<std::mutex> guard(warmed_mutex);
        warmed.insert(device);
    }
    return ms_since(t0);
}
