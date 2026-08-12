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
// gpusim2grid's own batch kernels and launch-site grid-size arithmetic
// (fill_J_kernel's J_base, the fill_FP/FQ/update_Va/Vm/MultiSlack/HVDC/
// VoltageControl kernels, the nr_grid_size() launch helper in
// nr_iter_step.cuh) were widened to 64-bit (ptrdiff_t/long long) offset and
// thread-index arithmetic -- see the "tid/b widened to ptrdiff_t" comments
// throughout acpf_nr_kernels.cu, violation_kernels.cu, driver.cuh and
// nr_iter_step.cuh. That fix lifted the batch_size * {nnz_J, dim_J} ceiling
// from an artificial INT32_MAX wraparound (which used to silently write
// outside the buffer -- a CUDA illegal-memory-access crash, not a catchable
// Python exception) up to whatever GPU memory actually allows.
//
// Two things this guard still has to catch even after that fix:
//
//   1. cuDSS itself. cudssMatrixCreateCsr takes int64_t nrows/ncols/nnz (the
//      single-system dims, which stay far below INT32_MAX regardless of
//      batch_size), but cuDSS's own internal indexing into the *batched*
//      value buffer (d_J_values_batch, read via CUDSS_CONFIG_UBATCH_SIZE
//      during FACTORIZATION/REFACTORIZATION/SOLVE) is closed-source -- there
//      is no public confirmation it is 64-bit-safe at batch_size * nnz_J
//      scales beyond INT32_MAX. Until that is verified empirically on real
//      hardware, this guard keeps rejecting the combination rather than
//      risk relying on unverified library behavior.
//
//   2. The cuSPARSE block-diagonal SpMV path. build_blockdiag_csr()
//      (contingency_analysis_helper.cpp) builds ONE literal block-diagonal
//      Ybus matrix of size (batch_size*n_bus)^2 with batch_size*nnz_Y
//      non-zeros, and BatchPfDriver's constructor hands its outer/inner
//      arrays to cusparseCreateConstCsr with CUSPARSE_INDEX_32I explicitly.
//      That is a hard, unavoidable 32-bit index requirement in the current
//      implementation -- widening gpusim2grid's own arithmetic cannot lift
//      it; only migrating to CUSPARSE_INDEX_64I (a separate, larger change
//      to the SpMV descriptor setup) could. batch_size * {n_bus, nnz_Y}
//      genuinely cannot exceed INT32_MAX today.
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
            "INT32_MAX (" + std::to_string(kInt32Max) + "). This combination "
            "hits a hard 32-bit index limit somewhere in the batch pipeline "
            "(cuSPARSE's block-diagonal SpMV descriptor, or cuDSS's own "
            "uniform-batch value-buffer indexing -- see batch_dims_check.hpp's "
            "own doc for which). Reduce " + count_name +
            " (e.g. run smaller batch_size chunks) so that " + count_name +
            " * " + stride_name + " stays within INT32_MAX.");
    }
}

#endif  // BATCH_DIMS_CHECK_HPP
