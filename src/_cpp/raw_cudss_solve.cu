// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "raw_cudss_solve.hpp"
#include "cuda_utils.h"
#include "dtypes.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cudss.h>

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
    int device)
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
