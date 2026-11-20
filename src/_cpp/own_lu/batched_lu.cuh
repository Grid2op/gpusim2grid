// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// own_lu/batched_lu.cuh -- numeric part of the batched LU of our own
// =============================================================================
//
// Streams values along an own_lu::LuPlan (lu_plan.hpp) for a batch of B
// matrices sharing its pattern. The caller's buffers keep gpusim2grid's layout,
// slot after slot (values [B × nnz], right-hand sides / solutions [B × n]);
// inside, the factor values are interleaved by matrix (v[entry·B + b]: a warp
// is 32 matrices on the same entry, index loads broadcast, value loads
// coalesced), with tiled transposes in and out.
//
// No pivoting: the pivots are the diagonal of the plan's order (a static-pivot
// refactorization, every call). A pivot smaller than eps · max|A_b| is clamped
// to it, as cuDSS's PIVOT_EPSILON, and counted per matrix (clamped_counts()):
// with lightsim2grid's constrained order none is expected; with another order
// the structurally zero pivots of the rows the outer loops switch by value are
// clamped, and iterative refinement (refine()) recovers the solution.
//
// Active prefix (set_active(nb)): every call then works on the first nb
// matrices only -- the slots past it are neither read nor written, their
// factors and solutions stay as they were -- with the buffers' layout (stride
// B) unchanged. The continuous tail (BatchPfDriver::_compact_tail moves the
// live rows to the front) runs on it instead of a second, smaller solver.
//
// Kernels: gpu-security-analysis' engine/kernels/lu.cu (RTE-AssistFlux),
// templated on the real type, plus the clamp counter and the refinement.
// Not thread-safe; every launch on the stream given at init().
//
// Original batched LU (algorithm, symbolic plan and kernels): Ralf Cortes Do
// Nascimento (RTE), contributed from gpu-security-analysis.
// =============================================================================

#ifndef GPUSIM2GRID_OWN_LU_BATCHED_LU_CUH
#define GPUSIM2GRID_OWN_LU_BATCHED_LU_CUH

#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "lu_plan.hpp"

namespace own_lu {

// The default relative pivot clamp (GPUSIM2GRID_OWN_LU_EPS overrides it):
// 1e-12 in FP64; FP32 cannot resolve that (a clamped pivot then gives garbage
// factors and refinement diverges), so 1e-5 there.
constexpr double default_pivot_eps()
{
    return sizeof(cuda_real_type) == 4 ? 1e-5 : 1e-12;
}

class BatchedLu {
public:
    // eps: relative pivot clamp (0 = none). split_chain / split_items: a level
    // with fewer than split_items items whose longest chain of contributions
    // exceeds split_chain shares each item over K warps (K = 2 ... 32).
    void init(const LuPlan& plan, int batch, cudaStream_t stream, double eps = default_pivot_eps(),
              int split_chain = 16, int split_items = 256);

    // values: [B × nnz] slot-major, the pattern of the plan's A (CSR order);
    // the first active() matrices are read.
    void factorize(const cuda_real_type* values);
    // b, x: [B × n] slot-major. x may alias b (but then b is gone: no refine()).
    void solve(const cuda_real_type* b, cuda_real_type* x);
    // x += LU^-1 (b - A x), `steps` times; A = values on (indptr, indices),
    // device CSR of the plan's pattern. Needs b intact: x must not alias b
    // (std::invalid_argument when steps > 0).
    void refine(const cuda_real_type* values, const int* indptr, const int* indices,
                const cuda_real_type* b, cuda_real_type* x, int steps);

    // Matrices [0, nb) only for the next calls (1 <= nb <= batch(); init()
    // sets batch()). The others' factors and solutions are left untouched.
    void set_active(int nb);
    int active() const { return nb_; }

    // Whether the fused triangular-solve kernel (a run of narrow levels in one 1024-thread block)
    // launches at that block size on the current device; init() fuses nothing otherwise.
    static bool fused_solve_supported();

    // Pivots clamped per active matrix by the last factorize() (host sync).
    std::vector<int> clamped_counts() const;

    // The triangular solves' plan (diagnostic, for the tests): the launches one solve() makes
    // (forward then backward; a fused run of levels is one), and whether a fused run holds a
    // level split over several warps (k > 1)
    int solve_launches() const { return static_cast<int>(fw_steps_.size() + bw_steps_.size()); }
    bool fused_run_has_split_level() const;

    int n() const { return n_; }
    int batch() const { return B_; }
    int n_ent() const { return n_ent_; }
    int levels() const { return levels_; }
    long long contributions() const { return contributions_; }
    bool initialized() const { return B_ > 0; }

private:
    struct Launch { int offset, count, k; bool is_u; };
    // a triangular solve's steps: one launch, or a run of consecutive narrow levels (each fitting one
    // pass of a 32-warp block: count * k <= 32) in one fused kernel, levels [first, last) of the run
    // tables (offset, count, k each)
    struct SolveStep { int launch = -1, first = 0, last = 0; };
    std::vector<SolveStep> fw_steps_, bw_steps_;
    thrust::device_vector<int> d_fw_run_, d_bw_run_;   // [3 x levels]: offset, count, k

    int n_ = 0, nnz_ = 0, n_ent_ = 0, B_ = 0, nb_ = 0, levels_ = 0;
    long long contributions_ = 0;
    double eps_ = 0.;
    cudaStream_t stream_ = nullptr;

    thrust::device_vector<int> d_a_ent_, d_inv_, d_rp_, d_dp_, d_ci_;
    thrust::device_vector<int> d_tgt_, d_cptr_, d_dg_, d_cl_, d_cu_;
    thrust::device_vector<int> d_fw_rows_, d_bw_rows_;
    std::vector<Launch> f_launch_, fw_launch_, bw_launch_;

    thrust::device_vector<cuda_real_type> d_v_;    // factors, interleaved [n_ent × B]
    thrust::device_vector<cuda_real_type> d_xi_;   // solve work, interleaved [n × B]
    thrust::device_vector<cuda_real_type> d_tol_;  // per matrix clamp
    thrust::device_vector<int>            d_clamped_;
    thrust::device_vector<cuda_real_type> d_r_, d_d_;  // refinement scratch [B × n] (the prefix used)
};

}  // namespace own_lu

#endif  // GPUSIM2GRID_OWN_LU_BATCHED_LU_CUH
