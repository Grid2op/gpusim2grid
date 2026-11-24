// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// fused_lu/top_lu.cuh -- own_lu with its trailing block dense, in shared memory
// =============================================================================
//
// own_lu::BatchedLu's contract and layout (values / b / x slot-major, factors
// interleaved by matrix, static pivots clamped and counted, an active prefix,
// refine()), with one change: the pivots of the last elimination levels -- T,
// every pivot at level >= L0, chosen as large as one block's shared memory
// holds densely -- are no longer one launch (two in the factorization) per
// level. T is closed: an L / U entry of a pivot in T only reaches its etree
// ancestors, all in T, so
//   factorize: the levels below L0 as own_lu, then one launch for what T x T
//     receives from outside T (one item per T x T entry, as own_lu's items, only
//     their contributions through pivots not in T), then one block per matrix
//     factoring T x T densely in shared memory (right-looking, a barrier per
//     pivot), written back into the factors;
//   solve: the rows outside T by levels as own_lu (fused runs included); T's
//     forward and backward one block per matrix each, x_T and T's factor values
//     in shared memory, a barrier per column.
// On PtFige's J the trailing ~100 pivots each sit alone on their level: ~400 of
// the ~766 dependent launches of a factorization + solve + refinement step.
// Prototype (untracked): kernels copied from own_lu/batched_lu.cu.
// =============================================================================

#ifndef GPUSIM2GRID_FUSED_LU_TOP_LU_CUH
#define GPUSIM2GRID_FUSED_LU_TOP_LU_CUH

#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "../own_lu/lu_plan.hpp"

namespace top_lu {

class TopLu {
public:
    // max_top: largest |T| (0 = as many as one block's shared memory holds densely; GPUSIM2GRID_TOP_LU_M
    // overrides). Other arguments as own_lu::BatchedLu::init.
    void init(const own_lu::LuPlan& plan, int batch, cudaStream_t stream, double eps, int max_top = 0,
              int split_chain = 16, int split_items = 256);
    void factorize(const cuda_real_type* values);
    // x is written by the refine() that follows (always: CudssBatchSolver calls both, steps may be 0) or by
    // flush(): until then it holds the previous values
    void solve(const cuda_real_type* b, cuda_real_type* x);
    void flush();
    void refine(const cuda_real_type* values, const int* indptr, const int* indices, const cuda_real_type* b,
                cuda_real_type* x, int steps);
    void set_active(int nb);
    int active() const { return nb_; }
    std::vector<int> clamped_counts() const;

    // 0: dense top when T is not empty; 1: never (own_lu's levels for everything, the same factors)
    void set_levels_only(bool v) { levels_only_ = v; }

    int top_size(int c = 0) const { return tops_[c].m_; }
    int top_level(int c = 0) const { return tops_[c].l0_; }
    int launches_factor() const;
    int launches_solve() const;

private:
    struct Launch { int offset, count, k; bool is_u; int mc = 0; };  // mc: longest chain
    struct SolveStep { int launch = -1, first = 0, last = 0; };
    struct Sched {  // a triangular solve's level launches, narrow runs fused
        std::vector<Launch> launches;
        std::vector<SolveStep> steps;
        thrust::device_vector<int> d_rows, d_run;
    };
    void run_solve(const Sched& s, bool forward, cuda_real_type* xi);

    int n_ = 0, nnz_ = 0, n_ent_ = 0, B_ = 0, nb_ = 0;
    double eps_ = 0.;
    bool levels_only_ = false;
    // the factorization transposes J once, into d_ai_ (A positions interleaved): the factors start from it,
    // the clamps come from its max, and refine() computes its residual on it, in the interleaved layout, right
    // after the solve that left b (d_bi_) and x (d_xi_) there
    // device memory without initialization (thrust's device_vector fills on resize): for buffers always written
    // before they are read
    struct RawBuf {
        cuda_real_type* p = nullptr;
        size_t n = 0;
        RawBuf() = default;
        RawBuf(const RawBuf&) = delete;
        RawBuf& operator=(const RawBuf&) = delete;
        ~RawBuf() { if (p) cudaFree(p); }
        void alloc(size_t m) {
            if (m == n) return;
            if (p) cudaFree(p);
            p = nullptr;
            n = m;
            if (m && cudaMalloc(&p, m * sizeof(cuda_real_type)) != cudaSuccess) throw std::runtime_error("[top_lu] cudaMalloc");
        }
        cuda_real_type* data() { return p; }
        const cuda_real_type* data() const { return p; }
    };
    RawBuf d_ai_, d_bi_, d_ri_, d_xi_;
    thrust::device_vector<unsigned long long> d_amax_;
    thrust::device_vector<int> d_src_, d_ra_ptr_, d_ra_p_, d_ra_col_;
    long long fact_gen_ = 0, solve_gen_ = -1;
    bool interleaved_ok_ = false, out_pending_ = false;
    const cuda_real_type *last_b_ = nullptr, *last_values_ = nullptr;
    cuda_real_type* last_x_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int split_chain_ = 16, split_items_ = 256;

    thrust::device_vector<int> d_a_ent_, d_inv_, d_rp_, d_dp_, d_ci_;
    thrust::device_vector<int> d_tgt_, d_cptr_, d_dg_, d_cl_, d_cu_;
    std::vector<Launch> f_all_;  // every factor step
    Sched fw_all_, bw_all_;

    // A trailing block T (the pivots of level >= l0_) and everything that depends on it. Two are built: tops_[0]
    // as large as one block's shared memory holds (few active matrices: its kernels replace the most launches),
    // tops_[1] capped (GPUSIM2GRID_TOP_LU_M_LARGE, default 104: with many active matrices its one-block-per-matrix
    // factorization costs more than the launches it replaces). factorize() picks one by the active count
    // (GPUSIM2GRID_TOP_LU_NB, default 768), the solves use the factorization's.
    struct Top {
        int m_ = 0, l0_ = 0, sf = 0, sw = 0, sb = 0;
        std::vector<Launch> f_bottom_;  // the factor steps below l0_
        Sched fw_rest_, bw_rest_;       // the rows outside T
        thrust::device_vector<int> d_t_rows_;                                 // T, ascending (plan numbering)
        thrust::device_vector<int> d_oa_tgt_, d_oa_src_, d_oa_cptr_, d_oa_cl_, d_oa_cu_;  // T x T items, outside part
        int n_oa_ = 0;
        thrust::device_vector<int> d_top_ent_;                                // T x T pattern entries (q order)
        thrust::device_vector<int> d_lslot_, d_uslot_, d_pp_ptr_, d_pslot_;   // the sparse T factorization's slots
        int n_top_ = 0;
        thrust::device_vector<int> d_rk_ptr_, d_rk_;          // per local pivot k: local rows i > k of L(:, k)
        thrust::device_vector<int> d_fo_ptr_, d_fo_ent_, d_fo_col_;  // per local row: its L entries outside T
        thrust::device_vector<int> d_fl_ent_;
        thrust::device_vector<int> d_bu_ptr_, d_bu_row_, d_bu_ent_;  // per local column k: rows i < k of U(:, k)
        thrust::device_vector<int> d_dg_ent_;
        int n_fl_ = 0, n_bu_ = 0;
        // T x T matrix-major (tm[b * n_top + q]): written by top_outside, factored in place by top_factor, read by
        // the T solves (contiguous per matrix; the interleaved v strides by the batch)
        thrust::device_vector<cuda_real_type> d_tm_;
        thrust::device_vector<int> d_fl_q_, d_bu_q_, d_dg_q_;
    };
    Top tops_[2];
    int cur_ = 0, top_nb_ = 768;
    int fuse_nb_ = 0;  // fused solve runs up to this many active matrices (GPUSIM2GRID_TOP_LU_FUSE_NB; issue #44)
    void solve_core(const Top& t, cuda_real_type* xi);

    thrust::device_vector<cuda_real_type> d_v_, d_tol_, d_r_, d_d_;
    thrust::device_vector<int> d_clamped_;
};

}  // namespace top_lu

#endif  // GPUSIM2GRID_FUSED_LU_TOP_LU_CUH
