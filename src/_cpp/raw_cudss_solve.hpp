// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef RAW_CUDSS_SOLVE_H
#define RAW_CUDSS_SOLVE_H

#include <vector>

// Solve J*dx = rhs with gpusim2grid's own cuDSS wrapper (analyze -> factorize
// -> solve, CudssContext/CudssDescriptor from cuda_utils.h), completely
// decoupled from any power-flow/grid construction: J is supplied directly as
// a CSR triplet. Meant for validating cuDSS itself against an arbitrary
// sparse system -- e.g. a (J, F) pair dumped via AcPfNrSession::get_J() /
// get_F() -- without rebuilding a grid/session at all. See
// repro_cudss_bug_standalone.py.
//
//   dim      matrix dimension (J is dim x dim)
//   indptr   CSR row pointer, size dim+1
//   indices  CSR column indices, size nnz
//   data     CSR values, size nnz (narrowed to cuda_real_type internally,
//            same as the rest of the codebase)
//   rhs      right-hand side, size dim
//   device   CUDA device ordinal, or -1 to use the current device
//
// Returns dx (host, double), size dim.
std::vector<double> solve_cudss_raw(
    int dim,
    const std::vector<int>& indptr,
    const std::vector<int>& indices,
    const std::vector<double>& data,
    const std::vector<double>& rhs,
    int device = -1);

#endif  // RAW_CUDSS_SOLVE_H
