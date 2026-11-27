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

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "../dtypes.hpp"
#include "../own_lu/lu_plan.hpp"
#include "../own_lu/batched_lu.cuh"   // own_lu_refine_all

namespace top_lu {

// A symbolic (read-only once built) device array whose copies share the storage: TopLu::copy_symbolic_from
// hands a new TopLu the template's arrays without copying device memory (~3 ms of device-to-device copies
// per driver build on PtFige). assign() builds new storage, as thrust's does; data() feeds raw().
template <class T>
struct SharedDev {
    std::shared_ptr<thrust::device_vector<T>> p = std::make_shared<thrust::device_vector<T>>();
    template <class It>
    void assign(It b, It e) { p = std::make_shared<thrust::device_vector<T>>(b, e); }
    void assign(size_t n, const T& v) { p = std::make_shared<thrust::device_vector<T>>(n, v); }
    auto data() const { return p->data(); }
    size_t size() const { return p->size(); }
    bool empty() const { return p->empty(); }
};

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
        SharedDev<int> d_rows, d_run;
    };
    // The factors a triangular solve reads: this instance's (one per lane) or the reference's (one for every lane,
    // shared: the low-rank first steps), over `nb` lanes of the interleaved x
    struct Factors { const cuda_real_type* v; const cuda_real_type* tm; bool shared; int nb; };
    void run_solve(const Sched& s, bool forward, cuda_real_type* xi, const int* mask, const Factors& f);
    // solve() on the matrices `mask` selects (clamp counts > 0; nullptr: all): refine()'s, as own_lu's
    // with_lowrank: the main solve (the low-rank lanes too); false: a refinement's correction (factorized lanes)
    void solve_impl(const cuda_real_type* b, cuda_real_type* x, const int* mask, bool with_lowrank);
    bool refine_all_ = false;   // own_lu::own_lu_refine_all()
    int batch_clamp_total() const;   // d_clamped_[0], host-synchronizing

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
    SharedDev<int> d_src_, d_ra_ptr_, d_ra_p_, d_ra_col_;
    long long fact_gen_ = 0, solve_gen_ = -1;
    bool interleaved_ok_ = false, out_pending_ = false;
    const cuda_real_type *last_b_ = nullptr, *last_values_ = nullptr;
    cuda_real_type* last_x_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int split_chain_ = 16, split_items_ = 256;

    SharedDev<int> d_a_ent_, d_inv_, d_rp_, d_dp_, d_ci_;
    SharedDev<int> d_tgt_, d_cptr_, d_dg_, d_cl_, d_cu_;
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
        SharedDev<int> d_t_rows_;                                 // T, ascending (plan numbering)
        SharedDev<int> d_oa_tgt_, d_oa_src_, d_oa_cptr_, d_oa_cl_, d_oa_cu_;  // T x T items, outside part
        int n_oa_ = 0;
        SharedDev<int> d_top_ent_;                                // T x T pattern entries (q order)
        SharedDev<int> d_lslot_, d_uslot_, d_pp_ptr_, d_pslot_;   // the sparse T factorization's slots
        int n_top_ = 0;
        SharedDev<int> d_rk_ptr_, d_rk_;          // per local pivot k: local rows i > k of L(:, k)
        SharedDev<int> d_fo_ptr_, d_fo_ent_, d_fo_col_;  // per local row: its L entries outside T
        SharedDev<int> d_fl_ent_;
        SharedDev<int> d_bu_ptr_, d_bu_row_, d_bu_ent_;  // per local column k: rows i < k of U(:, k)
        SharedDev<int> d_dg_ent_;
        int n_fl_ = 0, n_bu_ = 0;
        // T x T matrix-major (tm[b * n_top + q]): written by top_outside, factored in place by top_factor, read by
        // the T solves (contiguous per matrix; the interleaved v strides by the batch)
        thrust::device_vector<cuda_real_type> d_tm_;
        SharedDev<int> d_fl_q_, d_bu_q_, d_dg_q_;
    };
    Top tops_[2];
    int cur_ = 0, top_nb_ = 768;
    int fuse_nb_ = 0;  // fused solve runs up to this many active matrices (GPUSIM2GRID_TOP_LU_FUSE_NB; issue #44)
    void solve_core(const Top& t, cuda_real_type* xi, const int* mask, const Factors& f);
    Factors own_factors(int nb) const;

    // ---- low-rank first steps (GPUSIM2GRID_OWN_LU_LOWRANK=1, prototype; docs/dev_notes/n1_big_rounds_bandwidth.md,
    //      section 8) ----
    // A matrix whose rows are the reference's but for at most lr_K_ of them (a contingency row at its first Newton
    // iteration: the base case's J but for the rows of its tripped branch's buses) is not factorized: its solve is
    // the reference's (one factorization, shared) corrected by the Woodbury identity,
    //   x = z - W_R (I + D W_R)^-1 D z,  z = A_ref^-1 b,  D = (A - A_ref)[R, :],  W_R = A_ref^-1[:, R].
    // The reference: the entrywise majority of three matrices of the first factorization (ref_: a TopLu of one
    // matrix), its inverse computed once (d_winv_). factorize() classifies the matrices -- sampled entries, then
    // every row, then the capacitance matrix I + D W_R factorized with pivots of comparable size -- and puts
    // those that do not fit first in the interleaved layout (lanes [0, nreg_): factorized as before), the
    // candidates after them (lanes [nreg_, nreg_ + nc_)). solve(): a candidate whose right-hand side is the
    // reference's but for at most lr_Kf_ rows needs no solve (z = z_ref + columns of the inverse: x is z_ref plus
    // a combination of columns); the others are solved on the reference's factors.
    bool lr_on_ = false, is_ref_ = false;
    bool lr_ready_ = false, lr_failed_ = false, lr_inv_ready_ = false, lr_stats_ = false;
    int lr_K_ = 12;        // most modified rows of a candidate
    int lr_ref_[3] = {0, 1, 2};   // the matrices of the first factorization the reference is the majority of
    int lr_min_ = 32;      // fewest candidates worth the path
    double lr_max_ratio_ = 1e8;   // largest pivot ratio of the capacitance matrix
    int nreg_ = 0, nc_ = 0;
    bool lane_map_ = false;       // d_perm_ (lane -> matrix) in use since the last factorize()
    // the pending output holds low-rank lanes: the first lr_nB_ of them solved on the reference's factors
    // (interleaved, corrected when written out), the lr_nA_ others by columns (their right-hand side is the
    // reference's but for a few rows: x = z_ref + a combination of columns of A_ref^-1, nothing solved)
    bool pending_lr_ = false;
    int lr_nB_ = 0, lr_nA_ = 0;
    bool lr_f_ready_ = false;
    int lr_Kf_ = 8;        // most rows of a candidate's right-hand side off the reference's, for the columns mode
    int force_cur_ = -1;          // the trailing block to use whatever the active count (the reference's)
    std::unique_ptr<TopLu> ref_;
    SharedDev<int> d_jptr_, d_jcol_, d_samp_;   // J's rows (positions), columns; the sampled positions
    int n_samp_ = 0;
    SharedDev<int> d_tpos_, d_lev_;   // an entry's transposed one (the reference inverse solves A^T); a pivot's level
    int n_lev_ = 0;
    RawBuf d_jref_, d_winv_, d_lr_S_, d_lr_y_, d_fref_, d_zref_, d_lr_coef_;
    thrust::device_vector<int> d_kind_, d_lr_rows_, d_lr_cnt_, d_perm_, d_lr_piv_, d_slot_mask_;
    thrust::device_vector<int> d_lr_cntf_, d_lr_frows_, d_lr_ncol_, d_lr_colrow_, d_lr_cold_;
    std::vector<int> h_cntf_, h_tail_;
    std::vector<int> h_kind_, h_perm_;
    void lr_setup_reference(const cuda_real_type* values);
    void lr_classify(const cuda_real_type* values);
    void lr_build_inverse();
    void lr_solve_candidates(int nB);
    // the candidates whose right-hand side does not fit the columns mode: how many, their lanes put first
    int lr_prepare_rhs(const cuda_real_type* b);
    // the pending solution to x: the factorized lanes (+ `add` under `mask`: the refined one), the low-rank lanes
    void write_out(cuda_real_type* x, const cuda_real_type* add, const int* mask);

    // init() = init_symbolic() (a function of the plan, the parameters and the device alone: the
    // schedules, the item lists, both trailing blocks) + alloc_batch() (the buffers sized by the
    // batch). The symbolic part is kept per plan, process-wide, and copied into a new TopLu on the
    // same plan instead of rebuilt (~10 ms of host work + ~100 uploads on PtFige, at every driver
    // build): copy_symbolic_from copies every member init_symbolic sets, and nothing else.
    void init_symbolic(const own_lu::LuPlan& plan, int max_top, int split_chain, int split_items);
    void alloc_batch(int batch);
    void copy_symbolic_from(const TopLu& o);
    void set_smem_attributes() const;

    RawBuf d_v_;   // the factors: every entry written by the factorization before it is read
    thrust::device_vector<cuda_real_type> d_tol_, d_r_, d_d_;
    thrust::device_vector<int> d_clamped_;
};

}  // namespace top_lu

#endif  // GPUSIM2GRID_FUSED_LU_TOP_LU_CUH
