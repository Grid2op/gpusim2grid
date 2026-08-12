// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BATCH_DIMS_CHECK_HPP
#define BATCH_DIMS_CHECK_HPP

#include <limits>
#include <stdexcept>
#include <string>

// =============================================================================
// batch_dims_check.hpp
//
// gpusim2grid's batch kernels (fill_J_kernel, fill_FP_kernel/fill_FQ_kernel,
// update_Va_kernel/update_Vm_kernel, the MultiSlack/HVDC/VoltageControl
// feature kernels, compute_branch_flows_kernel, check_limit_violations_kernel,
// …) and the host-side build_blockdiag_csr() all compute a batch slot's flat
// buffer offset as a plain 32-bit `int` product -- e.g.
//   const int J_base = b * nnz_J;                        (fill_J_kernel)
//   h_batch_outer[b * n_bus + r] = single_outer[r] + b*nnz; (build_blockdiag_csr)
//
// The buffers those offsets index into ARE allocated correctly (every
// resize() in BatchPfDriver's constructor casts through size_t first), so a
// batch too large to fit in device memory fails with a normal bad_alloc --
// but once (batch_size or n_contingencies) * stride exceeds INT32_MAX, the
// *offset itself* silently wraps in 32-bit int arithmetic before that OOM
// check is ever reached. The kernel then writes outside the buffer: a CUDA
// illegal-memory-access crash, not a catchable Python exception. On a large
// enough grid (large nnz_J / dim_J / n_bus) this triggers at a batch size far
// below anything that would exhaust GPU memory, so "reduce batch_size until
// it fits" does not reliably avoid it -- the caller needs a hard error at the
// actual 32-bit boundary instead.
//
// check_batch_stride() is the single guard used at every call site about to
// size or launch a batch computation over a (count, stride) pair like the
// ones above. Call it BEFORE any allocation/launch for that pair.
// =============================================================================
inline void check_batch_stride(const std::string& context,
                                const std::string& count_name, long long count,
                                const std::string& stride_name, long long stride)
{
    constexpr long long kInt32Max = std::numeric_limits<int>::max();
    if (count <= 0 || stride <= 0) return;
    // Divide first so the overflow check itself never overflows.
    if (count > kInt32Max / stride) {
        throw std::runtime_error(
            "[" + context + "] " + count_name + " (" + std::to_string(count) +
            ") * " + stride_name + " (" + std::to_string(stride) + ") exceeds "
            "INT32_MAX (" + std::to_string(kInt32Max) + "). gpusim2grid's batch "
            "kernels index their buffers with 32-bit ints, so this combination "
            "would silently wrap the computed offset and corrupt GPU memory "
            "instead of raising a clean error. Reduce " + count_name +
            " (e.g. run smaller batch_size chunks) so that " + count_name +
            " * " + stride_name + " stays within INT32_MAX.");
    }
}

#endif  // BATCH_DIMS_CHECK_HPP
