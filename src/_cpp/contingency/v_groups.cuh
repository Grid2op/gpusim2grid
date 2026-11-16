// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef V_GROUPS_CUH
#define V_GROUPS_CUH

// =============================================================================
// contingency/v_groups.cuh
//
// Starting voltages by group (ContingencyAnalysisGPU.strategy_warm_start): a row
// of group g >= 0 starts from row g of a (n_groups x n_bus) buffer -- typically a
// contingency's converged voltages, shared by all its strategy rows -- its
// non-finite entries (buses its contingency masked) from the base case; group -1
// starts from the base case. Used by ContingencyBatch (set_v_groups, load_slots);
// self-contained so tests/cpp can exercise it alone.
//
//   v_groups_by_active  — host: the group of every ACTIVE row, from the group of
//                         every ORIGINAL row and the active -> original map
//   launch_gather_groups_to_slots — device: fill the loaded slots' starting V
// =============================================================================

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../cu_complex_utils.h"   // cudaComplexType, CudaFunHelper

// The group of every ACTIVE row: group_of_orig[active_to_orig[a]]. Throws
// std::runtime_error when group_of_orig does not have one entry per original row
// (n_total) or a group of an active row is outside [-1, n_groups).
inline std::vector<int> v_groups_by_active(const std::vector<int>& group_of_orig,
                                           const std::vector<int>& active_to_orig,
                                           int n_total, int n_groups)
{
    if (static_cast<int>(group_of_orig.size()) != n_total)
        throw std::runtime_error("v_groups_by_active: " + std::to_string(group_of_orig.size())
                                 + " groups for " + std::to_string(n_total) + " rows");
    std::vector<int> by_active(active_to_orig.size());
    for (std::size_t a = 0; a < active_to_orig.size(); ++a) {
        const int orig = active_to_orig[a];
        if (orig < 0 || orig >= n_total)
            throw std::runtime_error("v_groups_by_active: active row " + std::to_string(a)
                                     + " maps to row " + std::to_string(orig) + ", outside [0, "
                                     + std::to_string(n_total) + ")");
        const int g = group_of_orig[static_cast<std::size_t>(orig)];
        if (g < -1 || g >= n_groups)
            throw std::runtime_error("v_groups_by_active: group " + std::to_string(g)
                                     + " outside [-1, " + std::to_string(n_groups) + ")");
        by_active[a] = g;
    }
    return by_active;
}

// dst[d_slot[i]] = the starting V of row d_row[i] (an ACTIVE row, -1 = a phantom
// slot): row g = group_of_active[row] >= 0 of groups, a non-finite entry of it
// replaced by base; base for a phantom slot or group -1. One thread per (slot, bus).
template <typename C>
__global__ void gather_groups_to_slots_kernel(C* __restrict__ dst,
                                              const C* __restrict__ groups,
                                              const int* __restrict__ group_of_active,
                                              const C* __restrict__ base,
                                              const int* __restrict__ d_slot,
                                              const int* __restrict__ d_row,
                                              int n, int n_bus)
{
    const std::ptrdiff_t tid = static_cast<std::ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::ptrdiff_t i   = tid / n_bus;
    const int            b   = static_cast<int>(tid % n_bus);
    if (i >= n) return;
    const int row = d_row[i];
    const int g   = row >= 0 ? group_of_active[row] : -1;
    C v = base[b];
    if (g >= 0) {
        const C w = groups[static_cast<std::ptrdiff_t>(g) * n_bus + b];
        if (isfinite(CudaFunHelper::my_cuCreal(w)) && isfinite(CudaFunHelper::my_cuCimag(w))) v = w;
    }
    dst[static_cast<std::ptrdiff_t>(d_slot[i]) * n_bus + b] = v;
}

// Launch gather_groups_to_slots_kernel for n loaded slots of n_bus buses (no-op on
// empty input); returns the launch error, if any.
inline cudaError_t launch_gather_groups_to_slots(cudaComplexType* dst, const cudaComplexType* groups,
                                                 const int* group_of_active, const cudaComplexType* base,
                                                 const int* d_slot, const int* d_row,
                                                 int n, int n_bus, cudaStream_t cs)
{
    if (n <= 0 || n_bus <= 0) return cudaSuccess;
    constexpr int block = 256;
    const long long total = static_cast<long long>(n) * n_bus;
    const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    gather_groups_to_slots_kernel<cudaComplexType><<<grid, block, 0, cs>>>(
        dst, groups, group_of_active, base, d_slot, d_row, n, n_bus);
    return cudaGetLastError();
}

#endif  // V_GROUPS_CUH
