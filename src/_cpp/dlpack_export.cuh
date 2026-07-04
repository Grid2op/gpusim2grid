// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// dlpack_export.cuh  —  Low-level DLPack helpers (no CUDA/Thrust/pybind11)
//
// Included by dlpack_export.cu (nvcc) where complete types are available.
// Do NOT include this directly from python_bindings.cpp — use dlpack_export.hpp.
// =============================================================================

#pragma once

#include "dlpack/dlpack.h"
#include "dtypes.hpp"
#include "cu_complex_utils.h"

#include <memory>

// Compile-time sanity checks on the complex layout assumed by DLPack.
static_assert(std::is_same<cuda_real_type, float>::value ||
              std::is_same<cuda_real_type, double>::value,
              "cuda_real_type must be float or double");
static_assert(sizeof(cudaComplexType) == 2 * sizeof(cuda_real_type),
              "cudaComplexType must be packed (no padding between re/im)");

// 64 for FP32 build (complex64), 128 for FP64 (complex128).
static constexpr int kDLPackComplexBits =
    static_cast<int>(sizeof(cuda_real_type)) * 2 * 8;

// 32 for FP32 build (float32), 64 for FP64 (float64).
static constexpr int kDLPackRealBits =
    static_cast<int>(sizeof(cuda_real_type)) * 8;

// =============================================================================
// DLPackCtx — heap-allocated owner + shape storage.
// The shape array MUST outlive the DLManagedTensor (shape ptr is stored inside
// dl_tensor.shape); keeping it in this struct ensures that.
// =============================================================================
struct DLPackCtx {
    std::shared_ptr<void> owner;   // keeps the session alive
    int64_t               shape[2];
};

// =============================================================================
// dlpack_deleter — release the wrapper + ctx.  Never frees dl_tensor.data
// (owned by the thrust::device_vector inside the session).
// =============================================================================
inline void dlpack_deleter(DLManagedTensor* self) {
    delete static_cast<DLPackCtx*>(self->manager_ctx);
    delete self;
}

// =============================================================================
// make_dl_tensor
//   data      — raw device pointer (not freed by deleter)
//   device_id — CUDA ordinal where the memory lives
//   shape_0   — first dimension
//   shape_1   — second dimension; 0 → ndim=1 (1-D tensor)
//   owner     — held by DLPackCtx; drops session refcount when released
// =============================================================================
inline DLManagedTensor* make_dl_tensor(
    void*                 data,
    int                   device_id,
    int64_t               shape_0,
    int64_t               shape_1,
    std::shared_ptr<void> owner)
{
    auto* ctx     = new DLPackCtx;
    ctx->owner    = std::move(owner);
    ctx->shape[0] = shape_0;
    ctx->shape[1] = shape_1;

    int ndim = (shape_1 == 0) ? 1 : 2;

    auto* mt = new DLManagedTensor;
    mt->dl_tensor.data        = data;
    mt->dl_tensor.device      = {kDLCUDA, device_id};
    mt->dl_tensor.ndim        = ndim;
    mt->dl_tensor.dtype       = {kDLComplex,
                                  static_cast<uint8_t>(kDLPackComplexBits),
                                  1};
    mt->dl_tensor.shape       = ctx->shape;
    mt->dl_tensor.strides     = nullptr;    // compact row-major
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx           = ctx;
    mt->deleter               = dlpack_deleter;
    return mt;
}

// =============================================================================
// make_dl_tensor_real
//   Real-valued (float32 or float64) variant of make_dl_tensor.
//   Used for solve_JT_dlpack where the solution vector is real, not complex.
//   Always 1-D (no shape_1 parameter).
// =============================================================================
inline DLManagedTensor* make_dl_tensor_real(
    void*                 data,
    int                   device_id,
    int64_t               n,
    std::shared_ptr<void> owner)
{
    auto* ctx     = new DLPackCtx;
    ctx->owner    = std::move(owner);
    ctx->shape[0] = n;
    ctx->shape[1] = 0;

    auto* mt = new DLManagedTensor;
    mt->dl_tensor.data        = data;
    mt->dl_tensor.device      = {kDLCUDA, device_id};
    mt->dl_tensor.ndim        = 1;
    mt->dl_tensor.dtype       = {kDLFloat,
                                  static_cast<uint8_t>(kDLPackRealBits),
                                  1};
    mt->dl_tensor.shape       = ctx->shape;
    mt->dl_tensor.strides     = nullptr;    // compact row-major
    mt->dl_tensor.byte_offset = 0;
    mt->manager_ctx           = ctx;
    mt->deleter               = dlpack_deleter;
    return mt;
}