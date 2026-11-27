// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// fused_lu/top_lu.cu -- see top_lu.cuh. The level kernels are own_lu/batched_lu.cu's, copied (prototype);
// top_outside / top_factor / top_fwd / top_bwd are new.

#include "top_lu.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <array>
#include <utility>
#include <list>
#include <mutex>

namespace top_lu {

namespace {

constexpr int kThreads = 256;

#define OWN_LU_CHECK(call)                                                                    \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess)                                                                \
            throw std::runtime_error(std::string("[own_lu] ") + #call + ": " + cudaGetErrorString(e_)); \
    } while (0)

// thread = (item, matrix), matrix fastest, over the first nb matrices (the
// active prefix) of a batch of stride B
// Programmatic dependent launch (TopLu::launch): a kernel may start before its predecessor in the stream has
// finished; it waits here, before touching anything the predecessor produces or reads (a no-op otherwise)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TOP_PDL_WAIT() cudaGridDependencySynchronize()
#else
#define TOP_PDL_WAIT() do {} while (0)
#endif

#define LU_ITEM_AND_MATRIX                                                  \
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (long long)n_items * nb) return;                            \
    const int it = (int)(idx / nb);                                        \
    const int b = (int)(idx - (long long)it * nb);

// Item targets: entry t, or ~t for a fill entry (its sum starts at 0).
#define LU_TARGET(raw) ((raw) < 0 ? ~(raw) : (raw))
#define LU_START(raw, v, t, B, b) ((raw) < 0 ? T(0) : (v)[(size_t)(t) * (B) + (b)])

// Refinement on the matrices whose factorization clamped a pivot only (refine(); GPUSIM2GRID_OWN_LU_IR=all:
// every matrix): the solve kernels take the per-matrix clamp counts as a mask and leave a matrix with none
// alone (nullptr: every matrix, the plain solve). `b` is always < nb where it is read.
#define LU_MASKED(b) (mask != nullptr && mask[b] <= 0)

// A factor entry of matrix b in a solve kernel: one per matrix (interleaved), or the reference's, the same for
// every lane (kSh: the low-rank first steps solve on one shared factorization)
#define LU_VI(e) (kSh ? (size_t)(e) : (size_t)(e) * B + b)

template <class T>
__device__ __forceinline__ T clamp_pivot(T s, T e, int* clamped, int b)
{
    if (fabs(s) < e) {
        atomicAdd(clamped + b, 1);
        atomicAdd(clamped - 1, 1);   // the batch's total (d_clamped_[0])
        return s < T(0) ? -e : e;
    }
    return s;
}

// U step of a level: row p's entries, the pivot clamped to +-tol[b].
template <class T>
__global__ void lu_left_u(const int* __restrict__ tgt, const int* __restrict__ src, const T* __restrict__ ai, const int* __restrict__ cptr,
                          const int* __restrict__ cl, const int* __restrict__ cu,
                          const int* __restrict__ dg, const int n_items, const int B, const int nb,
                          const T* __restrict__ tol, int* __restrict__ clamped, T* __restrict__ v)
{
    TOP_PDL_WAIT();
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = (src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b]);
    {   // two partial sums: twice the loads in flight per thread
        T s2 = T(0);
        int c = cptr[it];
        const int ce = cptr[it + 1];
        for (; c + 1 < ce; c += 2) {
            s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
            s2 = fma(-v[(size_t)cl[c + 1] * B + b], v[(size_t)cu[c + 1] * B + b], s2);
        }
        if (c < ce) s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
        s += s2;
    }
    if (t == dg[it]) s = clamp_pivot(s, tol[b], clamped, b);
    v[(size_t)t * B + b] = s;
}

// L step of a level: column p's entries, divided by the (clamped) pivot.
template <class T>
__global__ void lu_left_l(const int* __restrict__ tgt, const int* __restrict__ src, const T* __restrict__ ai, const int* __restrict__ cptr,
                          const int* __restrict__ cl, const int* __restrict__ cu,
                          const int* __restrict__ dg, const int n_items, const int B, const int nb,
                          T* __restrict__ v)
{
    TOP_PDL_WAIT();
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = (src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b]);
    {   // two partial sums: twice the loads in flight per thread
        T s2 = T(0);
        int c = cptr[it];
        const int ce = cptr[it + 1];
        for (; c + 1 < ce; c += 2) {
            s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
            s2 = fma(-v[(size_t)cl[c + 1] * B + b], v[(size_t)cu[c + 1] * B + b], s2);
        }
        if (c < ce) s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
        s += s2;
    }
    v[(size_t)t * B + b] = s / v[(size_t)dg[it] * B + b];
}

// Forward (unit L): x[i] -= sum L[i,k] x[k]; backward: x[i] = (x[i] - sum U[i,j] x[j]) / U[i,i].
template <class T, bool kSh = false>
__global__ void lu_fwd(const int* __restrict__ rows, const int* __restrict__ rp,
                       const int* __restrict__ dp, const int* __restrict__ ci, const int n_items,
                       const int B, const int nb, const T* __restrict__ v, T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    LU_ITEM_AND_MATRIX
    if (LU_MASKED(b)) return;
    const int i = rows[it];
    T s = x[(size_t)i * B + b];
    {
        T s2 = T(0);
        int e = rp[i];
        const int ee = dp[i];
        for (; e + 1 < ee; e += 2) {
            s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
            s2 = fma(-v[LU_VI((e + 1))], x[(size_t)ci[e + 1] * B + b], s2);
        }
        if (e < ee) s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
        s += s2;
    }
    x[(size_t)i * B + b] = s;
}

template <class T, bool kSh = false>
__global__ void lu_bwd(const int* __restrict__ rows, const int* __restrict__ rp,
                       const int* __restrict__ dp, const int* __restrict__ ci, const int n_items,
                       const int B, const int nb, const T* __restrict__ v, T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    LU_ITEM_AND_MATRIX
    if (LU_MASKED(b)) return;
    const int i = rows[it];
    T s = x[(size_t)i * B + b];
    {
        T s2 = T(0);
        int e = dp[i] + 1;
        const int ee = rp[i + 1];
        for (; e + 1 < ee; e += 2) {
            s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
            s2 = fma(-v[LU_VI((e + 1))], x[(size_t)ci[e + 1] * B + b], s2);
        }
        if (e < ee) s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
        s += s2;
    }
    x[(size_t)i * B + b] = s / v[LU_VI(dp[i])];
}

// Layout changes, 32 x 32 tiles (block 32 x 8), matrices b < nb only.
// in:  dst[map[j] * B + b] = src[b * ncols + j]; out: dst[b * ncols + j] = src[map[j] * B + b]
template <class T>
__global__ void lu_tr_in(const T* __restrict__ src, const int* __restrict__ map, const int ncols,
                         const int B, const int nb, T* __restrict__ dst)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)b * ncols + j];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) dst[(size_t)map[j] * B + b] = t[threadIdx.x][r];
    }
}

template <class T>
__global__ void lu_tr_out(const T* __restrict__ src, const int* __restrict__ map, const int ncols,
                          const int B, const int nb, T* __restrict__ dst, const int* __restrict__ perm)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)map[j] * B + b];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) dst[(size_t)(perm ? perm[b] : b) * ncols + j] = t[threadIdx.x][r];
    }
}

// tol[b] = eps * max_j |values[b * nnz + j]| (one block of 256 per matrix)
template <class T>
__global__ void lu_scaled_tol(const T* __restrict__ values, const int nnz, const T eps, T* __restrict__ tol)
{
    TOP_PDL_WAIT();
    __shared__ T red[256];
    const T* row = values + (size_t)blockIdx.x * nnz;
    T m = T(0);
    for (int j = threadIdx.x; j < nnz; j += blockDim.x) m = fmax(m, fabs(row[j]));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) tol[blockIdx.x] = eps * red[0];
}

// Split variants for the narrow levels near the root: block = 32 consecutive
// matrices (lane) x K warps, one item per blockIdx.x; warp w sums the
// contributions w, w + K, ...; partial sums added in warp order (deterministic).
// red holds the K warps' partial sums only: dynamic shared memory of lu_split_smem<T>(K) bytes. A static
// [32][33] array (8.4 KB whatever K) capped these kernels at 10 blocks per SM, i.e. 42% occupancy with
// K = 2 (Nsight Compute, PtFige N-1 at batch 4096: the split levels at 72-76% of peak DRAM vs ~88% for
// the unsplit ones).
#define LU_SPLIT_THREAD                                         \
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5; \
    const int K = blockDim.x >> 5;                              \
    const int it = blockIdx.x;                                  \
    const int b = blockIdx.y * 32 + lane;   /* b < nb: active */ \
    extern __shared__ __align__(16) unsigned char lu_split_sm[]; \
    T (*red)[33] = reinterpret_cast<T (*)[33]>(lu_split_sm);

template <class T>
constexpr size_t lu_split_smem(int k) { return static_cast<size_t>(k) * 33 * sizeof(T); }

template <class T>
__device__ __forceinline__ T lu_split_sum(T acc, const int lane, const int part, const int K, T (*red)[33])
{
    red[part][lane] = acc;
    __syncthreads();
    T s = T(0);
    if (part == 0)
        for (int w = 0; w < K; ++w) s += red[w][lane];
    return s;
}

template <class T>
__global__ void lu_left_split(const int* __restrict__ tgt, const int* __restrict__ src, const T* __restrict__ ai, const int* __restrict__ cptr,
                              const int* __restrict__ cl, const int* __restrict__ cu,
                              const int* __restrict__ dg, const int B, const int nb, const int is_u,
                              const T* __restrict__ tol, int* __restrict__ clamped, T* __restrict__ v)
{
    TOP_PDL_WAIT();
    LU_SPLIT_THREAD
    T acc = T(0);
    if (b < nb)
        for (int c = cptr[it] + part; c < cptr[it + 1]; c += K)
            acc = fma(v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= nb) return;
    const int t = LU_TARGET(tgt[it]), d = dg[it];
    T s = (src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b]) - sum;
    if (is_u) {
        if (t == d) s = clamp_pivot(s, tol[b], clamped, b);
        v[(size_t)t * B + b] = s;
    } else {
        v[(size_t)t * B + b] = s / v[(size_t)d * B + b];
    }
}

template <class T, bool kSh = false>
__global__ void lu_fwd_split(const int* __restrict__ rows, const int* __restrict__ rp,
                             const int* __restrict__ dp, const int* __restrict__ ci, const int B,
                             const int nb, const T* __restrict__ v, T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < nb && !LU_MASKED(b))
        for (int e = rp[i] + part; e < dp[i]; e += K) acc = fma(v[LU_VI(e)], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= nb || LU_MASKED(b)) return;
    x[(size_t)i * B + b] -= sum;
}

template <class T, bool kSh = false>
__global__ void lu_bwd_split(const int* __restrict__ rows, const int* __restrict__ rp,
                             const int* __restrict__ dp, const int* __restrict__ ci, const int B,
                             const int nb, const T* __restrict__ v, T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < nb && !LU_MASKED(b))
        for (int e = dp[i] + 1 + part; e < rp[i + 1]; e += K)
            acc = fma(v[LU_VI(e)], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= nb || LU_MASKED(b)) return;
    x[(size_t)i * B + b] = (x[(size_t)i * B + b] - sum) / v[LU_VI(dp[i])];
}

constexpr int kRunThreads = 1024;   // lu_solve_run's block: 32 matrices (lanes) x 32 warps

// A run of narrow levels of a triangular solve in one kernel: block = 32 consecutive matrices (lane) x
// 32 warps, the levels in order with a block barrier between them (a matrix depends on its own rows
// only). In a level of k parts, warps item * k ... item * k + k - 1 take item `item`; every value is
// computed as lu_fwd / lu_bwd (k = 1) or lu_fwd_split / lu_bwd_split (k > 1) do: the same bits.
template <class T, bool kForward, bool kSh = false>
__global__ void __launch_bounds__(kRunThreads) lu_solve_run(const int* __restrict__ rows, const int* __restrict__ run, const int first,
                             const int last, const int n_lv, const int* __restrict__ rp, const int* __restrict__ dp,
                             const int* __restrict__ ci, const int B, const int nb, const T* __restrict__ v,
                             T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int b = blockIdx.x * 32 + lane;
    __shared__ T red[32][33];
    for (int l = first; l < last; ++l) {
        const int offset = run[l], count = run[n_lv + l], K = run[2 * n_lv + l];
        const int item = warp / K, part = warp - item * K;
        const bool active = item < count && b < nb && !LU_MASKED(b);
        const int i = item < count ? rows[offset + item] : 0;
        if (K == 1) {
            if (active) {
                T s = x[(size_t)i * B + b];
                if (kForward) {
                    for (int e = rp[i]; e < dp[i]; ++e) s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
                    x[(size_t)i * B + b] = s;
                } else {
                    for (int e = dp[i] + 1; e < rp[i + 1]; ++e) s = fma(-v[LU_VI(e)], x[(size_t)ci[e] * B + b], s);
                    x[(size_t)i * B + b] = s / v[LU_VI(dp[i])];
                }
            }
        } else {
            T acc = T(0);
            if (active) {
                if (kForward) {
                    for (int e = rp[i] + part; e < dp[i]; e += K) acc = fma(v[LU_VI(e)], x[(size_t)ci[e] * B + b], acc);
                } else {
                    for (int e = dp[i] + 1 + part; e < rp[i + 1]; e += K)
                        acc = fma(v[LU_VI(e)], x[(size_t)ci[e] * B + b], acc);
                }
            }
            red[warp][lane] = acc;
            __syncthreads();
            if (part == 0 && active) {
                T sum = T(0);
                for (int w = 0; w < K; ++w) sum += red[item * K + w][lane];
                if (kForward) x[(size_t)i * B + b] -= sum;
                else x[(size_t)i * B + b] = (x[(size_t)i * B + b] - sum) / v[LU_VI(dp[i])];
            }
        }
        __syncthreads();   // the level's rows before the next level reads them
    }
}

// r[b, i] = rhs[b, i] - (A_b x_b)[i], slot-major (thread = (matrix, row), row fastest)
template <class T>
__global__ void lu_residual(const T* __restrict__ values, const int* __restrict__ indptr,
                            const int* __restrict__ indices, const int n, const int nnz, const int B,
                            const T* __restrict__ rhs, const T* __restrict__ x, T* __restrict__ r, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * B) return;
    const int b = (int)(idx / n), i = (int)(idx - (long long)b * n);
    if (LU_MASKED(b)) { r[idx] = T(0); return; }
    const T* a = values + (size_t)b * nnz;
    const T* xb = x + (size_t)b * n;
    T s = rhs[idx];
    for (int p = indptr[i]; p < indptr[i + 1]; ++p) s = fma(-a[p], xb[indices[p]], s);
    r[idx] = s;
}

template <class T>
__global__ void lu_axpy(const long long m, const int n, const T* __restrict__ d, T* __restrict__ x,
                        const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m && !LU_MASKED(idx / n)) x[idx] += d[idx];
}

inline unsigned blocks_for(long long work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

template <class V>
auto raw(const V& v) { return thrust::raw_pointer_cast(v.data()); }
template <class V>
auto rawm(V& v) { return thrust::raw_pointer_cast(v.data()); }

// ---- the trailing block T ----

// T x T items: their contributions through pivots outside T (own_lu's item, no clamp, no division), into the
// matrix-major copy of T x T (tm[b * n_top + q]): a 32 items x 32 matrices tile, computed matrix-fast (coalesced
// reads), written item-fast through shared memory (items sorted by q: coalesced writes)
template <class T>
__global__ void top_outside(const int* __restrict__ oq, const int* __restrict__ src, const T* __restrict__ ai,
                            const int* __restrict__ cptr, const int* __restrict__ cl, const int* __restrict__ cu,
                            const int n_items, const int n_top, const int B, const int nb, const T* __restrict__ v,
                            T* __restrict__ tm)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int it = j0 + r, b = b0 + threadIdx.x;
        if (it < n_items && b < nb) {
            T s = src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b];
            for (int c = cptr[it]; c < cptr[it + 1]; ++c) s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
            t[r][threadIdx.x] = s;
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, it = j0 + threadIdx.x;
        if (it < n_items && b < nb) tm[(size_t)b * n_top + oq[it]] = t[threadIdx.x][r];
    }
}

// T rows: x_T[i] -= L(i, outside T) x_outside, block = 32 matrices (lanes) x K warps, one row per blockIdx.x
template <class T, bool kSh = false>
__global__ void top_fo(const int* __restrict__ trows, const int* __restrict__ fo_ptr, const int* __restrict__ fo_ent,
                       const int* __restrict__ fo_col, const int B, const int nb, const T* __restrict__ v, T* __restrict__ xi, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    __shared__ T red[32][33];
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5, K = blockDim.x >> 5, i = blockIdx.x;
    const int b = blockIdx.y * 32 + lane;
    T acc = T(0);
    if (b < nb && !LU_MASKED(b))
        for (int a = fo_ptr[i] + part; a < fo_ptr[i + 1]; a += K) acc = fma(v[LU_VI(fo_ent[a])], xi[(size_t)fo_col[a] * B + b], acc);
    red[part][lane] = acc;
    __syncthreads();
    if (part == 0 && b < nb && !LU_MASKED(b)) {
        T s = T(0);
        for (int w = 0; w < K; ++w) s += red[w][lane];
        xi[(size_t)trows[i] * B + b] -= s;
    }
}

constexpr int kTopFactorThreads = 512, kTopSolveThreads = 256;

// One block per matrix: T x T in shared memory, sparse (its pattern's entries in tm's order, q), right-looking
// with static pivots clamped as own_lu. Step k: thread 0 clamps the pivot (slot dq[k]) and takes its reciprocal;
// then the updates of the pattern pairs of column k -- target slot pslot[pp_ptr[k] + a r + c] from L slot lslot[a]
// and U slot uslot[c] of its r rows -- and the scaling of column k - 1 (deferred: column k is read unscaled during
// step k). L entries end divided by their pivot, U entries keep it (own_lu's layout).
template <class T>
__global__ void __launch_bounds__(kTopFactorThreads)
top_factor(const int m, const int n_top, const int* __restrict__ dq, const int* __restrict__ rk_ptr,
           const int* __restrict__ lslot, const int* __restrict__ uslot, const int* __restrict__ pp_ptr,
           const int* __restrict__ pslot, const int B, const int nb, const T* __restrict__ tol, int* __restrict__ clamped,
           T* __restrict__ tm)
{
    TOP_PDL_WAIT();
    extern __shared__ __align__(16) unsigned char sm[];
    T* D = reinterpret_cast<T*>(sm);
    T* inv = D + n_top;
    T* pv = inv + m;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb) return;
    T* tb = tm + (size_t)b * n_top;
    for (int q = tid; q < n_top; q += bd) D[q] = tb[q];
    const T e = tol[b];
    __syncthreads();
    for (int k = 0; k < m; ++k) {
        if (tid == 0) {
            T p = D[dq[k]];
            if (fabs(p) < e) {
                atomicAdd(clamped + b, 1);
                atomicAdd(clamped - 1, 1);   // the batch's total (d_clamped_[0])
                p = p < T(0) ? -e : e;
            }
            pv[k] = p;
            inv[k] = T(1) / p;
        }
        __syncthreads();
        const T ik = inv[k];
        if (k > 0) {
            const T ip = inv[k - 1];
            for (int a = rk_ptr[k - 1] + tid; a < rk_ptr[k]; a += bd) D[lslot[a]] *= ip;
        }
        const int r0 = rk_ptr[k], r = rk_ptr[k + 1] - r0, p0 = pp_ptr[k];
        for (int t = tid; t < r * r; t += bd) {
            const int ai = t / r, aj = t - ai * r;
            const int ts = pslot[p0 + t];
            D[ts] = fma(-D[lslot[r0 + ai]] * ik, D[uslot[r0 + aj]], D[ts]);
        }
        __syncthreads();
    }
    for (int a = rk_ptr[m - 1] + tid; a < rk_ptr[m]; a += bd) D[lslot[a]] *= inv[m - 1];
    for (int k = tid; k < m; k += bd) D[dq[k]] = pv[k];
    __syncthreads();
    for (int q = tid; q < n_top; q += bd) tb[q] = D[q];
}

template <class T>
__device__ __forceinline__ T top_warp_sum(T s)
{
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

// One block per matrix: T's forward (unit L) on T x T (top_fo took the columns outside T), column-oriented:
// x_T[i] -= L(i, k) x_T[k] for the pattern rows of column k, a barrier per column; L from the matrix-major tm
template <class T, bool kSh = false>
__global__ void __launch_bounds__(kTopSolveThreads)
top_fwd(const int m, const int n_top, const int* __restrict__ trows, const int* __restrict__ rk_ptr, const int* __restrict__ rk,
        const int* __restrict__ fl_q, const int n_fl, const int B, const int nb, const T* __restrict__ tm, T* __restrict__ xi, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    extern __shared__ __align__(16) unsigned char sm[];
    T* xT = reinterpret_cast<T*>(sm);
    T* Ls = xT + m;
    int* kp = reinterpret_cast<int*>(Ls + n_fl);
    int* kr = kp + m + 1;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb || LU_MASKED(b)) return;   // the whole block: before any barrier
    const T* tb = tm + (kSh ? (size_t)0 : (size_t)b * n_top);
    for (int q = tid; q < n_fl; q += bd) {
        Ls[q] = tb[fl_q[q]];
        kr[q] = rk[q];
    }
    for (int i = tid; i <= m; i += bd) kp[i] = rk_ptr[i];
    for (int i = tid; i < m; i += bd) xT[i] = xi[(size_t)trows[i] * B + b];
    __syncthreads();
    for (int k = 0; k < m; ++k) {
        const T xk = xT[k];
        for (int a = kp[k] + tid; a < kp[k + 1]; a += bd) xT[kr[a]] = fma(-Ls[a], xk, xT[kr[a]]);
        __syncthreads();
    }
    for (int i = tid; i < m; i += bd) xi[(size_t)trows[i] * B + b] = xT[i];
}

// One block per matrix: T's backward, column-oriented: x_k = x_T[k] / U(k, k) (reciprocals taken once), then
// x_T[i] -= U(i, k) x_k for the pattern rows i < k of column k, a barrier per column. U(T, outside) is empty.
template <class T, bool kSh = false>
__global__ void __launch_bounds__(kTopSolveThreads)
top_bwd(const int m, const int n_top, const int* __restrict__ trows, const int* __restrict__ bu_ptr, const int* __restrict__ bu_row,
        const int* __restrict__ bu_q, const int n_bu, const int* __restrict__ dg_q, const int B, const int nb,
        const T* __restrict__ tm, T* __restrict__ xi, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    extern __shared__ __align__(16) unsigned char sm[];
    T* xT = reinterpret_cast<T*>(sm);
    T* xo = xT + m;
    T* dinv = xo + m;
    T* Us = dinv + m;
    int* kp = reinterpret_cast<int*>(Us + n_bu);
    int* kr = kp + m + 1;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb || LU_MASKED(b)) return;   // the whole block: before any barrier
    const T* tb = tm + (kSh ? (size_t)0 : (size_t)b * n_top);
    for (int q = tid; q < n_bu; q += bd) {
        Us[q] = tb[bu_q[q]];
        kr[q] = bu_row[q];
    }
    for (int i = tid; i <= m; i += bd) kp[i] = bu_ptr[i];
    for (int i = tid; i < m; i += bd) {
        xT[i] = xi[(size_t)trows[i] * B + b];
        dinv[i] = T(1) / tb[dg_q[i]];
    }
    __syncthreads();
    for (int k = m - 1; k >= 0; --k) {
        const T xk = xT[k] * dinv[k];
        if (tid == 0) xo[k] = xk;
        for (int a = kp[k] + tid; a < kp[k + 1]; a += bd) xT[kr[a]] = fma(-Us[a], xk, xT[kr[a]]);
        __syncthreads();
    }
    for (int i = tid; i < m; i += bd) xi[(size_t)trows[i] * B + b] = xo[i];
}


// J -> its interleaved copy (A positions, matrix fastest) and max |A_b| (bits of a non-negative double: their
// order is the values'), grid-striding over column tiles so that a block adds one atomic per matrix
template <class T>
__global__ void lu_tr_in_vals(const T* __restrict__ src, const int ncols, const int B, const int nb, T* __restrict__ dst,
                              unsigned long long* __restrict__ amax, const int* __restrict__ perm)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    __shared__ double mx[8][32];
    const int b0 = blockIdx.y * 32;
    double m = 0.;
    for (int j0 = blockIdx.x * 32; j0 < ncols; j0 += gridDim.x * 32) {
        for (int r = threadIdx.y; r < 32; r += 8) {
            const int b = b0 + r, j = j0 + threadIdx.x;
            if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)(perm ? perm[b] : b) * ncols + j];
        }
        __syncthreads();
        for (int r = threadIdx.y; r < 32; r += 8) {
            const int j = j0 + r, b = b0 + threadIdx.x;
            if (b < nb && j < ncols) {
                const T x = t[threadIdx.x][r];
                dst[(size_t)j * B + b] = x;
                m = fmax(m, fabs(static_cast<double>(x)));
            }
        }
        __syncthreads();
    }
    mx[threadIdx.y][threadIdx.x] = m;
    __syncthreads();
    if (threadIdx.y == 0 && b0 + threadIdx.x < nb) {
        for (int r = 1; r < 8; ++r) m = fmax(m, mx[r][threadIdx.x]);
        atomicMax(amax + b0 + threadIdx.x, static_cast<unsigned long long>(__double_as_longlong(m)));
    }
}

template <class T>
__global__ void lu_tol_from_max(const unsigned long long* __restrict__ amax, const double eps, const int nb, T* __restrict__ tol)
{
    TOP_PDL_WAIT();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) tol[b] = static_cast<T>(eps * __longlong_as_double(static_cast<long long>(amax[b])));
}

// b -> two interleaved copies (the right-hand side kept for the refinement, the one the solve overwrites)
template <class T>
__global__ void lu_tr_in2(const T* __restrict__ src, const int* __restrict__ map, const int ncols, const int B, const int nb,
                          T* __restrict__ dst, T* __restrict__ dst2, const int* __restrict__ perm)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)(perm ? perm[b] : b) * ncols + j];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) {
            const size_t o = (size_t)map[j] * B + b;
            dst[o] = t[threadIdx.x][r];
            dst2[o] = t[threadIdx.x][r];
        }
    }
}

// out: dst[b * ncols + j] = src[map[j] * B + b] + src2[map[j] * B + b] (the refined solution)
template <class T>
__global__ void lu_tr_out_add(const T* __restrict__ src, const T* __restrict__ src2, const int* __restrict__ map,
                              const int ncols, const int B, const int nb, T* __restrict__ dst, const int* __restrict__ mask, const int* __restrict__ perm)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) {
            const size_t o = (size_t)map[j] * B + b;
            t[r][threadIdx.x] = LU_MASKED(b) ? src[o] : src[o] + src2[o];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) dst[(size_t)(perm ? perm[b] : b) * ncols + j] = t[threadIdx.x][r];
    }
}

// r = b - A x, everything interleaved (permuted rows, matrix fastest): thread = (row, matrix)
template <class T>
__global__ void lu_residual_i(const int* __restrict__ ra_ptr, const int* __restrict__ ra_p, const int* __restrict__ ra_col,
                              const int n, const int B, const int nb, const T* __restrict__ ai, const T* __restrict__ bi,
                              const T* __restrict__ xi, T* __restrict__ ri, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * nb) return;
    const int i = (int)(idx / nb), b = (int)(idx - (long long)i * nb);
    if (LU_MASKED(b)) { ri[(size_t)i * B + b] = T(0); return; }
    T s = bi[(size_t)i * B + b];
    for (int k = ra_ptr[i]; k < ra_ptr[i + 1]; ++k) s = fma(-ai[(size_t)ra_p[k] * B + b], xi[(size_t)ra_col[k] * B + b], s);
    ri[(size_t)i * B + b] = s;
}

template <class T>
__global__ void lu_axpy_i(const int n, const int B, const int nb, const T* __restrict__ d, T* __restrict__ x, const int* __restrict__ mask)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * nb) return;
    const int i = (int)(idx / nb), b = (int)(idx - (long long)i * nb);
    if (LU_MASKED(b)) return;
    x[(size_t)i * B + b] += d[(size_t)i * B + b];
}

// ---- low-rank first steps (top_lu.cuh) ----

constexpr int kLrMaxK = 16;   // the capacitance system of a candidate lives in a thread's local arrays

__device__ __forceinline__ bool lr_same(double x, double y) { return __double_as_longlong(x) == __double_as_longlong(y); }
__device__ __forceinline__ bool lr_same(float x, float y) { return __float_as_int(x) == __float_as_int(y); }

// The reference: the entrywise majority of three matrices (the base case's J wherever at most one of the three rows
// changed the entry)
template <class T>
__global__ void lr_majority(const T* __restrict__ a, const T* __restrict__ b, const T* __restrict__ c, const int nnz,
                            T* __restrict__ ref)
{
    TOP_PDL_WAIT();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    const T x = a[i], y = b[i];
    ref[i] = (lr_same(x, y) || lr_same(x, c[i])) ? x : y;
}

// kind[b] = 0 when at most K of the sampled positions (one per sampled row) of matrix b differ from the reference
// (it may be a candidate: lr_diff looks at every row), else -1; cnt[b] = 0; tot: how many were kept
template <class T>
__global__ void lr_sample(const T* __restrict__ values, const T* __restrict__ ref, const int* __restrict__ samp, const int ns,
                          const int nnz, const int nb, const int K, int* __restrict__ kind, int* __restrict__ cnt,
                          int* __restrict__ tot)
{
    TOP_PDL_WAIT();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const T* a = values + (size_t)b * nnz;
    int d = 0;
    for (int s = 0; s < ns; ++s) d += lr_same(a[samp[s]], ref[samp[s]]) ? 0 : 1;
    kind[b] = d <= K ? 0 : -1;
    cnt[b] = 0;
    if (d <= K) atomicAdd(tot, 1);
}

// The rows of matrix b that differ from the reference's (bitwise), for the matrices the samples kept: block = (256
// rows, matrix); cnt[b] counts them, the first K go to rows (in no order: lr_capacitance sorts them)
template <class T>
__global__ void lr_diff(const T* __restrict__ values, const T* __restrict__ ref, const int* __restrict__ jptr, const int n,
                        const int nnz, const int K, const int* __restrict__ kind, int* __restrict__ cnt, int* __restrict__ rows)
{
    TOP_PDL_WAIT();
    const int b = blockIdx.y;
    if (kind[b] < 0) return;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const T* a = values + (size_t)b * nnz;
    bool d = false;
    for (int p = jptr[r]; p < jptr[r + 1]; ++p) d = d || !lr_same(a[p], ref[p]);
    if (d) {
        const int i = atomicAdd(cnt + b, 1);
        if (i < K) rows[(size_t)b * K + i] = r;
    }
}

// The capacitance matrix of a candidate, S = I + D W_R with D = (A_b - A_ref)[R, :] and W_R = A_ref^-1[:, R]
// (winv[inv[r] * n + j] = A_ref^-1[j, r]), factorized in place with partial pivoting (whole rows swapped: piv, as
// LAPACK's getrf). kind[b] = its number of rows, or -1 (factorized like the others) with more than K rows, a zero
// or non-finite pivot, or pivots spread over more than max_ratio.
template <class T>
__global__ void lr_capacitance(const T* __restrict__ values, const T* __restrict__ ref, const int* __restrict__ jptr,
                               const int* __restrict__ jcol, const int* __restrict__ inv, const T* __restrict__ winv,
                               const int n, const int nnz, const int nb, const int K, const T max_ratio, int* __restrict__ kind,
                               const int* __restrict__ cnt, int* __restrict__ rows, T* __restrict__ S, int* __restrict__ piv)
{
    TOP_PDL_WAIT();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb || kind[b] < 0) return;
    const int k = cnt[b];
    if (k > K) { kind[b] = -1; return; }
    int* rw = rows + (size_t)b * K;
    for (int a = 1; a < k; ++a) {   // the atomics' order is not reproducible: ascending rows
        const int x = rw[a];
        int c = a - 1;
        while (c >= 0 && rw[c] > x) { rw[c + 1] = rw[c]; --c; }
        rw[c + 1] = x;
    }
    T* Sb = S + (size_t)b * K * K;
    int* pb = piv + (size_t)b * K;
    const T* a = values + (size_t)b * nnz;
    for (int i = 0; i < k; ++i) {
        const int r = rw[i];
        for (int c = 0; c < k; ++c) {
            const T* wc = winv + (size_t)inv[rw[c]] * n;
            T s = i == c ? T(1) : T(0);
            for (int p = jptr[r]; p < jptr[r + 1]; ++p) s = fma(a[p] - ref[p], wc[jcol[p]], s);
            Sb[i * K + c] = s;
        }
    }
    T pmax = T(0), pmin = T(0);
    bool ok = true;
    for (int c = 0; c < k && ok; ++c) {
        int im = c;
        T vm = fabs(Sb[c * K + c]);
        for (int i = c + 1; i < k; ++i) {
            const T x = fabs(Sb[i * K + c]);
            if (x > vm) { vm = x; im = i; }
        }
        pb[c] = im;
        if (!(vm > T(0)) || isinf(vm)) { ok = false; break; }   // also a NaN column
        if (im != c)
            for (int j = 0; j < k; ++j) {
                const T x = Sb[c * K + j];
                Sb[c * K + j] = Sb[im * K + j];
                Sb[im * K + j] = x;
            }
        if (c == 0 || vm > pmax) pmax = vm;
        if (c == 0 || vm < pmin) pmin = vm;
        const T inv = T(1) / Sb[c * K + c];
        for (int i = c + 1; i < k; ++i) {
            const T l = Sb[i * K + c] * inv;
            Sb[i * K + c] = l;
            for (int j = c + 1; j < k; ++j) Sb[i * K + j] = fma(-l, Sb[c * K + j], Sb[i * K + j]);
        }
    }
    kind[b] = (ok && !(pmin * max_ratio < pmax)) ? k : -1;
}

// y = S^-1 (D z) of the candidates, z their reference solve (interleaved, lane c of xi); y per lane
template <class T>
__global__ void lr_coeff(const T* __restrict__ values, const T* __restrict__ ref, const int* __restrict__ jptr,
                         const int* __restrict__ jcol, const int* __restrict__ inv, const T* __restrict__ xi, const int B,
                         const int nnz, const int nc, const int K, const int* __restrict__ perm, const int* __restrict__ kind,
                         const int* __restrict__ rows, const T* __restrict__ S, const int* __restrict__ piv, T* __restrict__ y)
{
    TOP_PDL_WAIT();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int b = perm[c], k = kind[b];
    const int* rw = rows + (size_t)b * K;
    const T* Sb = S + (size_t)b * K * K;
    const int* pb = piv + (size_t)b * K;
    const T* a = values + (size_t)b * nnz;
    T t[kLrMaxK];
    for (int i = 0; i < k; ++i) {
        const int r = rw[i];
        T s = T(0);
        for (int p = jptr[r]; p < jptr[r + 1]; ++p) s = fma(a[p] - ref[p], xi[(size_t)inv[jcol[p]] * B + c], s);
        t[i] = s;
    }
    for (int i = 0; i < k; ++i) {
        const int im = pb[i];
        const T x = t[i];
        t[i] = t[im];
        t[im] = x;
    }
    for (int i = 1; i < k; ++i) {
        T s = t[i];
        for (int j = 0; j < i; ++j) s = fma(-Sb[i * K + j], t[j], s);
        t[i] = s;
    }
    for (int i = k - 1; i >= 0; --i) {
        T s = t[i];
        for (int j = i + 1; j < k; ++j) s = fma(-Sb[i * K + j], t[j], s);
        t[i] = s / Sb[i * K + i];
    }
    for (int i = 0; i < k; ++i) y[(size_t)c * K + i] = t[i];
}

// The candidates' solution: dst[matrix, j] = z[j] - sum_a y_a A_ref^-1[j, r_a], z interleaved (lane c of src), the
// inverse's columns contiguous (32 x 32 tiles as lu_tr_out)
template <class T>
__global__ void lu_tr_out_lr(const T* __restrict__ src, const int* __restrict__ map, const int ncols, const int B,
                             const int nc, T* __restrict__ dst, const int* __restrict__ perm, const int* __restrict__ kind,
                             const int* __restrict__ rows, const T* __restrict__ y, const T* __restrict__ winv, const int K)
{
    TOP_PDL_WAIT();
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, c0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, c = c0 + threadIdx.x;
        if (c < nc && j < ncols) t[r][threadIdx.x] = src[(size_t)map[j] * B + c];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int c = c0 + r, j = j0 + threadIdx.x;
        if (c < nc && j < ncols) {
            const int b = perm[c], k = kind[b];
            T s = t[threadIdx.x][r];
            for (int a = 0; a < k; ++a) s = fma(-y[(size_t)c * K + a], winv[(size_t)map[rows[(size_t)b * K + a]] * ncols + j], s);
            dst[(size_t)b * ncols + j] = s;
        }
    }
}

constexpr int kLrMaxKf = 16;

// The rows of a candidate's right-hand side that differ from the reference's (bitwise): block = (256 rows, candidate
// lane c of matrix perm[c]); cntf[c] counts them, the first Kf go to frows (lr_coeff_cols sorts them)
template <class T>
__global__ void lr_fdiff(const T* __restrict__ rhs, const T* __restrict__ fref, const int n, const int Kf,
                         const int* __restrict__ perm, int* __restrict__ cntf, int* __restrict__ frows)
{
    TOP_PDL_WAIT();
    const int c = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    if (!lr_same(rhs[(size_t)perm[c] * n + r], fref[r])) {
        const int i = atomicAdd(cntf + c, 1);
        if (i < Kf) frows[(size_t)c * Kf + i] = r;
    }
}

// The columns mode of a candidate (lane c of the columns' lanes): b = f_ref + db (db on the rows F), so z = A_ref^-1 b = z_ref + W_F db and
// x = z - W_R y, y = S^-1 (D z): the coefficients of the columns W_R (-y) and W_F (db), D z computed from z_ref and
// the entries of W_F at D's columns.
template <class T>
__global__ void lr_coeff_cols(const T* __restrict__ values, const T* __restrict__ ref, const int* __restrict__ jptr,
                              const int* __restrict__ jcol, const int* __restrict__ inv, const T* __restrict__ rhs,
                              const T* __restrict__ fref, const T* __restrict__ zref, const T* __restrict__ winv, const int n,
                              const int nnz,
                              const int nc, const int K, const int Kf, const int* __restrict__ perm,
                              const int* __restrict__ kind, const int* __restrict__ rows, const T* __restrict__ S,
                              const int* __restrict__ piv, const int* __restrict__ cntf, int* __restrict__ frows,
                              const int* __restrict__ cold, int* __restrict__ ncol, int* __restrict__ colrow,
                              T* __restrict__ coef)
{
    TOP_PDL_WAIT();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc) return;
    const int co = cold[c];   // the candidate's lane when its right-hand side was compared (cntf, frows)
    const int b = perm[c], k = kind[b], kf = cntf[co];
    const int* rw = rows + (size_t)b * K;
    const T* Sb = S + (size_t)b * K * K;
    const int* pb = piv + (size_t)b * K;
    const T* a = values + (size_t)b * nnz;
    int* fr = frows + (size_t)co * Kf;
    for (int i = 1; i < kf; ++i) {   // ascending rows: the atomics' order is not reproducible
        const int x = fr[i];
        int j = i - 1;
        while (j >= 0 && fr[j] > x) { fr[j + 1] = fr[j]; --j; }
        fr[j + 1] = x;
    }
    T db[kLrMaxKf];
    for (int f = 0; f < kf; ++f) db[f] = rhs[(size_t)b * n + fr[f]] - fref[fr[f]];
    T t[kLrMaxK];
    for (int i = 0; i < k; ++i) {
        const int r = rw[i];
        T s = T(0);
        for (int p = jptr[r]; p < jptr[r + 1]; ++p) {
            const int col = jcol[p];
            T z = zref[col];
            for (int f = 0; f < kf; ++f) z = fma(winv[(size_t)inv[fr[f]] * n + col], db[f], z);
            s = fma(a[p] - ref[p], z, s);
        }
        t[i] = s;
    }
    for (int i = 0; i < k; ++i) {
        const int im = pb[i];
        const T x = t[i];
        t[i] = t[im];
        t[im] = x;
    }
    for (int i = 1; i < k; ++i) {
        T s = t[i];
        for (int j = 0; j < i; ++j) s = fma(-Sb[i * K + j], t[j], s);
        t[i] = s;
    }
    for (int i = k - 1; i >= 0; --i) {
        T s = t[i];
        for (int j = i + 1; j < k; ++j) s = fma(-Sb[i * K + j], t[j], s);
        t[i] = s / Sb[i * K + i];
    }
    const int CW = K + Kf;
    int* cr = colrow + (size_t)c * CW;
    T* cf = coef + (size_t)c * CW;
    for (int i = 0; i < k; ++i) { cr[i] = inv[rw[i]]; cf[i] = -t[i]; }   // the inverse's columns by pivot
    for (int f = 0; f < kf; ++f) { cr[k + f] = inv[fr[f]]; cf[k + f] = db[f]; }
    ncol[c] = k + kf;
}

// The candidates' solution in the columns mode: dst[matrix, j] = z_ref[j] + sum_q coef_q winv[row_q * n + j]
template <class T>
__global__ void lr_out_cols(const T* __restrict__ zref, const T* __restrict__ winv, const int n, const int nc, const int CW,
                            const int* __restrict__ perm, const int* __restrict__ ncol, const int* __restrict__ colrow,
                            const T* __restrict__ coef, T* __restrict__ dst)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * nc) return;
    const int c = (int)(idx / n), j = (int)(idx - (long long)c * n);
    const int* cr = colrow + (size_t)c * CW;
    const T* cf = coef + (size_t)c * CW;
    T s = zref[j];
    const int m = ncol[c];
    for (int q = 0; q < m; ++q) s = fma(cf[q], winv[(size_t)cr[q] * n + j], s);
    dst[(size_t)perm[c] * n + j] = s;
}

// The reference's factors complete in v: the trailing block's entries, kept apart (tm), copied in
template <class T>
__global__ void lr_densify(const int* __restrict__ top_ent, const T* __restrict__ tm, const int n_top, T* __restrict__ v)
{
    TOP_PDL_WAIT();
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n_top) v[top_ent[q]] = tm[q];
}

// A_ref^-1, every column at once: X[p * n + j] = A_ref^-1[j, order[p]], i.e. row p of X is the inverse's column
// order[p], contiguous. Lane j (an unknown) solves the transposed system A^T x = e_j on the factors (A = L U:
// U^T w = e, L^T x = w), an entry of U^T / L^T being the transposed entry's value (tpos).
// lr_inv_walk: w, zero but on the path from pivot inv[j] to the root of the elimination tree (thread = lane):
// w_i = (e_i - sum_{k < i} U(k, i) w_k) / U(i, i), each w_k pushed to its row's entries -- ancestors, all on the path,
// one per level: the accumulators stay in the thread, indexed by level. X zeroed beforehand (off the path).
constexpr int kLrLev = 320;
template <class T>
__global__ void lr_inv_walk(const int* __restrict__ inv, const int* __restrict__ lev, const int n_lev, const int* __restrict__ rp,
                            const int* __restrict__ dp, const int* __restrict__ ci, const T* __restrict__ v, const int n,
                            T* __restrict__ X)
{
    TOP_PDL_WAIT();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    T acc[kLrLev];
    for (int l = 0; l < n_lev; ++l) acc[l] = T(0);
    int i = inv[j];
    acc[lev[i]] = T(1);
    for (;;) {
        const T w = acc[lev[i]] / v[dp[i]];
        X[(size_t)i * n + j] = w;
        const int e0 = dp[i] + 1, e1 = rp[i + 1];
        if (e0 >= e1) break;   // the root
        for (int e = e0; e < e1; ++e) {
            const int l = lev[ci[e]];
            acc[l] = fma(-v[e], w, acc[l]);
        }
        i = ci[e0];   // the parent: the first entry of U's row
    }
}

// lr_inv_bwd: one level of L^T x = w (the levels downwards: a row's operands are its ancestors), thread =
// (row, lane), lanes side by side: x_i = w_i - sum_{k > i} L(k, i) x_k
template <class T>
__global__ void lr_inv_bwd(const int* __restrict__ rows, const int n_items, const int* __restrict__ rp,
                           const int* __restrict__ dp, const int* __restrict__ ci, const int* __restrict__ tpos,
                           const T* __restrict__ v, const int n, T* __restrict__ X)
{
    TOP_PDL_WAIT();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n_items * n) return;
    const int it = (int)(idx / n), j = (int)(idx - (long long)it * n);
    const int i = rows[it];
    T s = X[(size_t)i * n + j], s2 = T(0);
    int e = dp[i] + 1;
    const int ee = rp[i + 1];
    if (e >= ee) return;   // a root: x = w
    for (; e + 1 < ee; e += 2) {
        s = fma(-v[tpos[e]], X[(size_t)ci[e] * n + j], s);
        s2 = fma(-v[tpos[e + 1]], X[(size_t)ci[e + 1] * n + j], s2);
    }
    if (e < ee) s = fma(-v[tpos[e]], X[(size_t)ci[e] * n + j], s);
    X[(size_t)i * n + j] = s + s2;
}

// diagnostic (GPUSIM2GRID_OWN_LU_LOWRANK_STATS): out[2 b] = max_i |rhs - A x|, out[2 b + 1] = max_i |rhs|, one block per matrix
template <class T>
__global__ void lr_dbg_residual(const T* __restrict__ values, const int* __restrict__ indptr, const int* __restrict__ indices,
                                const int n, const int nnz, const T* __restrict__ rhs, const T* __restrict__ x, double* __restrict__ out)
{
    TOP_PDL_WAIT();
    __shared__ double red[2][256];
    const int b = blockIdx.x;
    const T* a = values + (size_t)b * nnz;
    const T* xb = x + (size_t)b * n;
    double mr = 0., mb = 0.;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        T s = rhs[(size_t)b * n + i];
        mb = fmax(mb, fabs((double)s));
        for (int p = indptr[i]; p < indptr[i + 1]; ++p) s = fma(-a[p], xb[indices[p]], s);
        const double r = fabs((double)s);
        if (!(r <= mr)) mr = r;   // a NaN residual stays
    }
    red[0][threadIdx.x] = mr; red[1][threadIdx.x] = mb;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < blockDim.x; ++k) {
            if (!(red[0][k] <= mr)) mr = red[0][k];
            mb = fmax(mb, red[1][k]);
        }
        out[2 * b] = mr; out[2 * b + 1] = mb;
    }
}

// mask by matrix of a refinement with a lane map: the factorized lanes' clamp counts (1 without them: every one),
// 0 for the low-rank lanes
__global__ void lr_slot_mask(const int* __restrict__ perm, const int* __restrict__ clamped, const int nreg, const int nb,
                             int* __restrict__ slot_mask)
{
    TOP_PDL_WAIT();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nb) return;
    slot_mask[perm[c]] = c < nreg ? (clamped != nullptr ? clamped[c] : 1) : 0;
}

// Warps per item when few matrices are active: the GPU is mostly idle, so a level whose items carry a real chain
// of contributions is split over warps (each summing every k-th) whatever its width, the chain cut ~4 x k fold;
// with many matrices the plan's static choice stays (own_lu's).
inline int split_for(int static_k, int count, int mc, int nb)
{
    if (static_cast<long long>(count) * nb > 8192 || mc <= 8) return static_k;
    int k = 1;
    while (k < 32 && 4 * k < mc) k <<= 1;
    return std::max(static_k, k);
}

}  // namespace




// Every launch of TopLu, with programmatic dependent launch (GPUSIM2GRID_TOP_LU_PDL=0: off): the launch of a
// kernel overlaps its predecessor's drain, the kernel waiting in TOP_PDL_WAIT for its results
template <typename... KArgs, typename... Args>
inline void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args&&... args)
{
    static const bool on = [] { const char* e = std::getenv("GPUSIM2GRID_TOP_LU_PDL"); return !(e && e[0] == '0'); }();
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = on ? 1 : 0;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    OWN_LU_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

namespace {
// The symbolic templates (TopLu::init): matched on the plan's content, the parameters, the
// environment variables init_symbolic reads and the device. Never freed at exit (deliberately
// leaked: their device memory would be freed after the CUDA context in a static destructor).
struct TopTemplateKey {
    int n = 0, nnz = 0, n_ent = 0, max_top = 0, split_chain = 0, split_items = 0, device = 0;
    std::vector<int> order, rp, ci, a_ent;
    std::string env;
};
std::string top_env_string()
{
    std::string s;
    for (const char* v : {"GPUSIM2GRID_TOP_LU_M", "GPUSIM2GRID_TOP_LU_M_LARGE", "GPUSIM2GRID_TOP_LU_NB",
                          "GPUSIM2GRID_TOP_LU_FUSE_NB"}) {
        const char* e = std::getenv(v);
        s += e ? e : "-";
        s += '|';
    }
    return s;
}
}  // namespace

void TopLu::init(const own_lu::LuPlan& plan, int batch, cudaStream_t stream, double eps, int max_top,
                 int split_chain, int split_items)
{
    if (batch <= 0) throw std::invalid_argument("[top_lu::TopLu::init] batch must be > 0");
    static const bool cache_on = [] {
        const char* e = std::getenv("GPUSIM2GRID_OWN_LU_PLAN_CACHE");
        return !(e && e[0] == '0');
    }();
    if (!cache_on) {
        init_symbolic(plan, max_top, split_chain, split_items);
    } else {
        TopTemplateKey key;
        key.n = plan.n; key.nnz = plan.nnz; key.n_ent = plan.n_ent;
        key.max_top = max_top; key.split_chain = split_chain; key.split_items = split_items;
        OWN_LU_CHECK(cudaGetDevice(&key.device));
        key.env = top_env_string();
        static std::mutex mu;
        static std::list<std::pair<TopTemplateKey, TopLu*>> cache;   // most recent first
        std::lock_guard<std::mutex> lock(mu);
        auto match = [&](const TopTemplateKey& k) {   // the cheap fields first, the vectors only then
            return k.n == key.n && k.nnz == key.nnz && k.n_ent == key.n_ent && k.max_top == key.max_top
                && k.split_chain == key.split_chain && k.split_items == key.split_items && k.device == key.device
                && k.env == key.env && k.order == plan.order && k.rp == plan.rp && k.ci == plan.ci
                && k.a_ent == plan.a_ent;
        };
        auto it = cache.begin();
        while (it != cache.end() && !match(it->first)) ++it;
        if (it != cache.end()) {
            cache.splice(cache.begin(), cache, it);
            copy_symbolic_from(*cache.front().second);
            set_smem_attributes();
        } else {
            init_symbolic(plan, max_top, split_chain, split_items);
            key.order = plan.order; key.rp = plan.rp; key.ci = plan.ci; key.a_ent = plan.a_ent;
            TopLu* t = new TopLu();
            t->copy_symbolic_from(*this);
            cache.emplace_front(std::move(key), t);
            if (cache.size() > 4) { delete cache.back().second; cache.pop_back(); }
        }
    }
    eps_ = eps;
    stream_ = stream;
    refine_all_ = own_lu::own_lu_refine_all();
    alloc_batch(batch);
    // the low-rank first steps (GPUSIM2GRID_OWN_LU_LOWRANK=1): FP64 only, J row-major, A_ref^-1 within its budget
    lr_on_ = lr_ready_ = lr_failed_ = lr_inv_ready_ = lr_f_ready_ = false;
    ref_.reset();
    auto env_on = [](const char* name) { const char* e = std::getenv(name); return e != nullptr && *e != '\0' && *e != '0'; };
    if (!is_ref_ && env_on("GPUSIM2GRID_OWN_LU_LOWRANK") && sizeof(cuda_real_type) == 8 && batch >= 3
        && d_jptr_.size() == static_cast<size_t>(n_) + 1) {
        double max_mb = 4096.;
        if (const char* e = std::getenv("GPUSIM2GRID_OWN_LU_LOWRANK_MAX_MB")) max_mb = std::atof(e);
        if (const char* e = std::getenv("GPUSIM2GRID_OWN_LU_LOWRANK_K")) lr_K_ = std::min(std::max(std::atoi(e), 1), kLrMaxK);
        if (const char* e = std::getenv("GPUSIM2GRID_OWN_LU_LOWRANK_MIN")) lr_min_ = std::max(std::atoi(e), 1);
        lr_stats_ = env_on("GPUSIM2GRID_OWN_LU_LOWRANK_STATS");
        if (static_cast<double>(n_) * n_ * sizeof(cuda_real_type) <= max_mb * 1048576.) {
            lr_on_ = true;
            ref_ = std::make_unique<TopLu>();
            ref_->is_ref_ = true;
            ref_->init(plan, 1, stream, eps, max_top, split_chain, split_items);
            // the reference's trailing block: the one a large batch uses, whatever the candidates' number
            ref_->force_cur_ = (batch > top_nb_ && tops_[1].m_ > 0) ? 1 : 0;
            const size_t K = static_cast<size_t>(lr_K_);
            d_kind_.assign(B_, -1);
            d_lr_cnt_.assign(B_ + 1, 0);
            d_lr_rows_.assign(B_ * K, 0);
            d_lr_piv_.assign(B_ * K, 0);
            d_perm_.assign(B_, 0);
            d_slot_mask_.assign(B_, 0);
            d_lr_S_.alloc(B_ * K * K);
            d_lr_y_.alloc(B_ * K);
            if (const char* e = std::getenv("GPUSIM2GRID_OWN_LU_LOWRANK_KF")) lr_Kf_ = std::min(std::max(std::atoi(e), 0), kLrMaxKf);
            const size_t Kf = static_cast<size_t>(lr_Kf_);
            d_lr_cntf_.assign(B_ + 1, 0);
            d_lr_frows_.assign(B_ * std::max<size_t>(Kf, 1), 0);
            d_lr_ncol_.assign(B_, 0);
            d_lr_cold_.assign(B_, 0);
            d_lr_colrow_.assign(B_ * (K + Kf), 0);
            d_lr_coef_.alloc(B_ * (K + Kf));
        }
    }
}

void TopLu::copy_symbolic_from(const TopLu& o)
{
    n_ = o.n_; nnz_ = o.nnz_; n_ent_ = o.n_ent_;
    split_chain_ = o.split_chain_; split_items_ = o.split_items_;
    d_src_ = o.d_src_; d_ra_ptr_ = o.d_ra_ptr_; d_ra_p_ = o.d_ra_p_; d_ra_col_ = o.d_ra_col_;
    d_a_ent_ = o.d_a_ent_; d_inv_ = o.d_inv_; d_rp_ = o.d_rp_; d_dp_ = o.d_dp_; d_ci_ = o.d_ci_;
    d_tgt_ = o.d_tgt_; d_cptr_ = o.d_cptr_; d_dg_ = o.d_dg_; d_cl_ = o.d_cl_; d_cu_ = o.d_cu_;
    f_all_ = o.f_all_; fw_all_ = o.fw_all_; bw_all_ = o.bw_all_;
    for (int c = 0; c < 2; ++c) {
        tops_[c] = o.tops_[c];
        tops_[c].d_tm_.clear();   // batch-sized: alloc_batch
    }
    top_nb_ = o.top_nb_; fuse_nb_ = o.fuse_nb_;
    d_jptr_ = o.d_jptr_; d_jcol_ = o.d_jcol_; d_samp_ = o.d_samp_; n_samp_ = o.n_samp_;
    d_tpos_ = o.d_tpos_; d_lev_ = o.d_lev_; n_lev_ = o.n_lev_;
    cur_ = 0;
    fact_gen_ = 0; solve_gen_ = -1;
    interleaved_ok_ = false; out_pending_ = false;
    last_b_ = nullptr; last_values_ = nullptr; last_x_ = nullptr;
}

void TopLu::set_smem_attributes() const
{
    using T = cuda_real_type;
    const int sf = std::max(tops_[0].sf, tops_[1].sf), sw = std::max(tops_[0].sw, tops_[1].sw),
              sb = std::max(tops_[0].sb, tops_[1].sb);
    if (sf > 0) OWN_LU_CHECK(cudaFuncSetAttribute(top_factor<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sf));
    if (sw > 0) OWN_LU_CHECK(cudaFuncSetAttribute(top_fwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw));
    if (sb > 0) OWN_LU_CHECK(cudaFuncSetAttribute(top_bwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb));
    if (sw > 0) OWN_LU_CHECK(cudaFuncSetAttribute(top_fwd<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw));
    if (sb > 0) OWN_LU_CHECK(cudaFuncSetAttribute(top_bwd<T, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb));
}

void TopLu::alloc_batch(int batch)
{
    using T = cuda_real_type;
    B_ = nb_ = nreg_ = batch;
    nc_ = 0;
    lane_map_ = false;
    // no zeroing: every factor entry is written by the factorization before anything reads it (a fill
    // entry's sum starts from 0 in the kernels, LU_START): 4.6 GB of memset per driver build on PtFige
    d_v_.alloc(static_cast<size_t>(n_ent_) * B_);
    d_ai_.alloc(static_cast<size_t>(nnz_) * B_);  // written by the transpose before any read
    for (Top& t : tops_) t.d_tm_.assign(static_cast<size_t>(std::max(t.n_top_, 1)) * B_, T(0));
    d_bi_.alloc(static_cast<size_t>(n_) * B_);
    d_ri_.alloc(static_cast<size_t>(n_) * B_);
    d_amax_.assign(B_, 0ULL);
    d_xi_.alloc(static_cast<size_t>(n_) * B_);
    d_tol_.assign(B_, T(0));
    d_clamped_.assign(B_ + 1, 0);   // [0]: the batch's clamp total (clamped[-1] in the kernels), then one per matrix
    d_r_.clear();
    d_d_.clear();
    OWN_LU_CHECK(cudaDeviceSynchronize());
}

void TopLu::init_symbolic(const own_lu::LuPlan& plan, int max_top, int split_chain, int split_items)
{
    using T = cuda_real_type;
    n_ = plan.n;
    nnz_ = plan.nnz;
    n_ent_ = plan.n_ent;
    split_chain_ = split_chain;
    split_items_ = split_items;
    const int n = n_;
    const std::vector<int>&rp = plan.rp, &dp = plan.dp, &ci = plan.ci;

    auto up = [](SharedDev<int>& d, const std::vector<int>& h) {
        if (h.empty()) d.assign(1, 0);
        else d.assign(h.begin(), h.end());
    };
    up(d_a_ent_, plan.a_ent);
    up(d_inv_, plan.inv_order);
    up(d_rp_, rp);
    up(d_dp_, dp);
    up(d_ci_, ci);

    auto k_of = [&](int count, int max_chain) {
        if (split_chain <= 0 || count >= split_items) return 1;
        const int k = (max_chain + split_chain - 1) / split_chain;
        if (k <= 1) return 1;
        int p = 1;
        while (p < k) p <<= 1;
        return std::min(p, 32);
    };
    // a triangular solve's schedule from rows in level order and their slices (offset, count, max_chain)
    auto make_sched = [&](Sched& s, const std::vector<int>& rows, const std::vector<std::array<int, 3>>& slices) {
        s.launches.clear();
        s.steps.clear();
        for (const auto& sl : slices) s.launches.push_back({sl[0], sl[1], k_of(sl[1], sl[2]), false, sl[2]});
        const int n_lv = static_cast<int>(s.launches.size());
        std::vector<int> run(3 * static_cast<size_t>(std::max(n_lv, 1)), 0);
        for (int l = 0; l < n_lv; ++l) {
            run[l] = s.launches[l].offset;
            run[n_lv + l] = s.launches[l].count;
            run[2 * n_lv + l] = s.launches[l].k;
        }
        auto narrow = [&](int l) { return static_cast<long long>(s.launches[l].count) * s.launches[l].k <= 32; };
        for (int l = 0; l < n_lv;) {
            int e = l;
            while (e < n_lv && narrow(e)) ++e;
            if (e - l >= 2) {
                // a fused run's block has 32 warps whatever its levels: give each level all of them (warps per row
                // = 32 / rows, a power of 2, no more than its longest row needs), its chains split that much more
                for (int q = l; q < e; ++q) {
                    const auto& Lq = s.launches[q];
                    int k = 1;
                    while (2 * k * Lq.count <= 32 && 2 * k <= std::max(1, Lq.mc)) k <<= 1;
                    run[2 * n_lv + q] = std::max(Lq.k, k);
                }
                s.steps.push_back(SolveStep{-1, l, e});
                l = e;
            } else {
                s.steps.push_back(SolveStep{l, 0, 0});
                ++l;
            }
        }
        up(s.d_rows, rows);
        up(s.d_run, run);
    };
    auto group = [&](const std::vector<int>& lev, const std::vector<char>& keep, const std::vector<int>& len,
                     std::vector<int>& rows, std::vector<std::array<int, 3>>& slices) {
        rows.clear();
        slices.clear();
        for (int i = 0; i < n; ++i)
            if (keep[i]) rows.push_back(i);
        std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) { return lev[a] < lev[b]; });
        for (size_t a = 0; a < rows.size();) {
            size_t b = a;
            int mc = 0;
            while (b < rows.size() && lev[rows[b]] == lev[rows[a]]) mc = std::max(mc, len[rows[b++]]);
            slices.push_back({static_cast<int>(a), static_cast<int>(b - a), mc});
            a = b;
        }
    };

    // levels (the plan's): factorization / forward, backward
    std::vector<int> level(n, 0), fw_len(n), bw_len(n);
    for (int i = 0; i < n; ++i) {
        int mx = -1;
        for (int e = rp[i]; e < dp[i]; ++e) mx = std::max(mx, level[ci[e]]);
        level[i] = mx + 1;
        fw_len[i] = dp[i] - rp[i];
        bw_len[i] = rp[i + 1] - dp[i] - 1;
    }
    const int H = n ? *std::max_element(level.begin(), level.end()) + 1 : 0;
    std::vector<char> all(n, 1);
    {
        std::vector<int> rows, blev(n, 0);
        std::vector<std::array<int, 3>> sl;
        group(level, all, fw_len, rows, sl);
        make_sched(fw_all_, rows, sl);
        for (int i = n - 1; i >= 0; --i) {
            int mx = -1;
            for (int e = dp[i] + 1; e < rp[i + 1]; ++e) mx = std::max(mx, blev[ci[e]]);
            blev[i] = mx + 1;
        }
        group(blev, all, bw_len, rows, sl);
        make_sched(bw_all_, rows, sl);
    }
    // the factorization's items with explicit sources: the plan's (target entry, its A position or -1 for a
    // fill entry: start 0), then, at the end of the U step of its row's level, an item for every U entry
    // without one: a copy of its A value (no contribution: own_lu leaves the transposed A value there; the
    // factors start from the interleaved copy of A here, so the entry must be written), or a 0 for an entry
    // of the symmetrized pattern with neither an A value nor a contribution -- the backward solve reads it,
    // and the factors are not zero-filled (PtFige's J has one such entry)
    std::vector<int> erow(n_ent_);
    for (int i = 0; i < n; ++i)
        for (int e = rp[i]; e < rp[i + 1]; ++e) erow[e] = i;
    struct Step { int offset, count, mc; bool is_u; };
    std::vector<int> it_tgt, it_src, it_cptr{0}, it_cl, it_cu, it_dg;
    it_tgt.reserve(plan.f_tgt.size() + n);
    it_src.reserve(plan.f_tgt.size() + n);
    it_dg.reserve(plan.f_tgt.size() + n);
    it_cptr.reserve(plan.f_tgt.size() + n + 1);
    it_cl.reserve(plan.cl.size());
    it_cu.reserve(plan.cu.size());
    std::vector<Step> steps;
    {
        auto target = [](int raw) { return raw < 0 ? ~raw : raw; };
        std::vector<int> a_of(n_ent_, -1);
        for (int p = 0; p < nnz_; ++p) a_of[plan.a_ent[p]] = p;
        std::vector<char> has_item(n_ent_, 0);
        for (int raw : plan.f_tgt) has_item[target(raw)] = 1;
        std::vector<int> ex_ptr(H + 1, 0), ex;  // per level, the U entries of A without an item (rows ascending)
        for (int r = 0; r < n; ++r)
            for (int e = dp[r] + 1; e < rp[r + 1]; ++e)
                if (!has_item[e]) ++ex_ptr[level[r] + 1];
        for (int L = 0; L < H; ++L) ex_ptr[L + 1] += ex_ptr[L];
        ex.resize(ex_ptr[H]);
        {
            std::vector<int> fill(ex_ptr.begin(), ex_ptr.end() - 1);
            for (int r = 0; r < n; ++r)
                for (int e = dp[r] + 1; e < rp[r + 1]; ++e)
                    if (!has_item[e]) ex[fill[level[r]]++] = e;
        }
        for (const auto& st : plan.factor_steps) {
            Step ns{static_cast<int>(it_tgt.size()), 0, st.max_chain, st.is_u};
            for (int it = st.offset; it < st.offset + st.count; ++it) {
                const int t = target(plan.f_tgt[it]);
                it_tgt.push_back(t);
                it_src.push_back(a_of[t]);
                it_dg.push_back(plan.f_dg[it]);
                for (int c = plan.f_cptr[it]; c < plan.f_cptr[it + 1]; ++c) {
                    it_cl.push_back(plan.cl[c]);
                    it_cu.push_back(plan.cu[c]);
                }
                it_cptr.push_back(static_cast<int>(it_cl.size()));
            }
            if (st.is_u) {
                const int t0 = target(plan.f_tgt[st.offset]);
                const int lv = level[std::min(erow[t0], ci[t0])];
                for (int x = ex_ptr[lv]; x < ex_ptr[lv + 1]; ++x) {
                    const int e = ex[x];
                    it_tgt.push_back(e);
                    it_src.push_back(a_of[e]);
                    it_dg.push_back(dp[erow[e]]);
                    it_cptr.push_back(static_cast<int>(it_cl.size()));
                }
            }
            ns.count = static_cast<int>(it_tgt.size()) - ns.offset;
            steps.push_back(ns);
        }
    }
    up(d_tgt_, it_tgt);
    up(d_src_, it_src);
    up(d_cptr_, it_cptr);
    up(d_dg_, it_dg);
    up(d_cl_, it_cl);
    up(d_cu_, it_cu);
    f_all_.clear();
    for (const auto& st : steps) f_all_.push_back({st.offset, st.count, k_of(st.count, st.mc), st.is_u, st.mc});
    // the interleaved residual: per permuted row, its A entries (position, permuted column)
    {
        std::vector<int> ra_ptr(n + 1, 0), ra_p(nnz_), ra_col(nnz_);
        for (int p = 0; p < nnz_; ++p) ++ra_ptr[erow[plan.a_ent[p]] + 1];
        for (int i = 0; i < n; ++i) ra_ptr[i + 1] += ra_ptr[i];
        {
            std::vector<int> fill(ra_ptr.begin(), ra_ptr.end() - 1);
            for (int p = 0; p < nnz_; ++p) {  // p ascending within each row: the order the residual sums in
                const int q = fill[erow[plan.a_ent[p]]]++;
                ra_p[q] = p;
                ra_col[q] = ci[plan.a_ent[p]];
            }
        }
        up(d_ra_ptr_, ra_ptr);
        up(d_ra_p_, ra_p);
        up(d_ra_col_, ra_col);
    }

    // J's own rows (the low-rank first steps compare matrices row by row): a position's row and column from
    // its factor entry; one sampled position per sampled row. Left empty when the positions are not row-major.
    {
        std::vector<int> jptr(n + 1, 0), jcol(nnz_), samp;
        bool row_major = true;
        int prev = 0;
        for (int p = 0; p < nnz_; ++p) {
            const int e = plan.a_ent[p];
            const int r = plan.order[erow[e]];
            jcol[p] = plan.order[ci[e]];
            if (r < prev) row_major = false;
            prev = r;
            ++jptr[r + 1];
        }
        for (int i = 0; i < n; ++i) jptr[i + 1] += jptr[i];
        if (!row_major) jptr.assign(1, 0);
        else {
            const int ns = std::min(64, n);
            for (int s2 = 0; s2 < ns; ++s2) {
                const int p = jptr[static_cast<long long>(s2) * n / ns];
                if (p < nnz_ && (samp.empty() || samp.back() != p)) samp.push_back(p);
            }
        }
        n_samp_ = static_cast<int>(samp.size());
        up(d_jptr_, jptr);
        up(d_jcol_, jcol);
        up(d_samp_, samp);
    }
    {
        std::vector<int> tpos(n_ent_);
        for (int i = 0; i < n; ++i)
            for (int e = rp[i]; e < rp[i + 1]; ++e) {   // (i, j) -> (j, i): the pattern is symmetric
                const int j = ci[e];
                const auto b = ci.begin() + rp[j], en = ci.begin() + rp[j + 1];
                const auto it = std::lower_bound(b, en, i);
                if (it == en || *it != i) throw std::logic_error("[top_lu] the factors' pattern is not symmetric");
                tpos[e] = static_cast<int>(it - ci.begin());
            }
        up(d_tpos_, tpos);
        up(d_lev_, level);
        n_lev_ = H;
    }

    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_M")) max_top = std::atoi(e);
    int cap_large = 104;
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_M_LARGE")) cap_large = std::atoi(e);
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_NB")) top_nb_ = std::atoi(e);
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_FUSE_NB")) fuse_nb_ = std::atoi(e);
    auto build_top = [&](Top& t, int cap) {
        // T: the pivots of level >= L0, L0 the lowest level whose T x T (stored sparsely: its pattern's entries) and the
        // T solves' lists fit one block's shared memory
        int optin = 0, dev = 0;
        OWN_LU_CHECK(cudaGetDevice(&dev));
        OWN_LU_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        std::vector<int> lcount(n, 0);  // L entries per column
        for (int i = 0; i < n; ++i)
            for (int e = rp[i]; e < dp[i]; ++e) ++lcount[ci[e]];
        std::vector<std::vector<int>> by_lev(H);
        for (int i = 0; i < n; ++i) by_lev[level[i]].push_back(i);
        t.m_ = 0;
        t.l0_ = H;
        {
            std::vector<char> mark(n, 0);
            long long mm = 0, nl = 0, nnz_t = 0;
            const long long sz = static_cast<long long>(sizeof(T));
            for (int L = H - 1; L >= 0; --L) {
                for (int i : by_lev[L]) mark[i] = 1;
                for (int i : by_lev[L])
                    for (int e = rp[i]; e < rp[i + 1]; ++e)
                        if (mark[ci[e]]) nnz_t += level[ci[e]] > L ? 2 : 1;  // (i, c) and, c above, its mirror (c, i)
                mm += static_cast<long long>(by_lev[L].size());
                for (int i : by_lev[L]) nl += lcount[i];
                const long long sf = (nnz_t + 2 * mm) * sz;                      // top_factor
                const long long sw = (mm + nl) * sz + (mm + 1 + nl) * 4;         // top_fwd
                const long long sb = (3 * mm + nl) * sz + (mm + 1 + nl) * 4;     // top_bwd
                if (std::max(sf, std::max(sw, sb)) > optin || (cap > 0 && mm > cap)) break;
                t.m_ = static_cast<int>(mm);
                t.l0_ = L;
            }
        }
        if (t.m_ < 2) { t.m_ = 0; t.l0_ = H; }
        std::vector<char> in_t(n, 0);
        std::vector<int> trows, tpos(n, -1);
        for (int i = 0; i < n; ++i)
            if (level[i] >= t.l0_) { in_t[i] = 1; tpos[i] = static_cast<int>(trows.size()); trows.push_back(i); }
        for (int i = 0; i < n; ++i) {  // closure: L(:, k in T) and U(k in T, :) stay in T
            for (int e = rp[i]; e < dp[i]; ++e)
                if (in_t[ci[e]] && !in_t[i]) throw std::logic_error("[top_lu] T not closed (L)");
            if (in_t[i])
                for (int e = dp[i] + 1; e < rp[i + 1]; ++e)
                    if (!in_t[ci[e]]) throw std::logic_error("[top_lu] T not closed (U)");
        }
        // factor steps below L0; the T x T items' outside contributions
        t.f_bottom_.clear();
        std::vector<int> oa_tgt, oa_src, oa_cptr{0}, oa_cl, oa_cu;
        for (size_t s = 0; s < steps.size(); ++s) {
            const auto& st = steps[s];
            auto top_item = [&](int it) {
                const int t = it_tgt[it];
                return in_t[std::min(erow[t], ci[t])] != 0;
            };
            const bool top = top_item(st.offset);
            for (int it = st.offset; it < st.offset + st.count; ++it)
                if (top_item(it) != top) throw std::logic_error("[top_lu] a factor step straddles L0");
            if (!top) { t.f_bottom_.push_back(f_all_[s]); continue; }
            for (int it = st.offset; it < st.offset + st.count; ++it) {
                oa_tgt.push_back(it_tgt[it]);
                oa_src.push_back(it_src[it]);
                for (int c = it_cptr[it]; c < it_cptr[it + 1]; ++c)
                    if (!in_t[ci[it_cl[c]]]) {
                        oa_cl.push_back(it_cl[c]);
                        oa_cu.push_back(it_cu[c]);
                    }
                oa_cptr.push_back(static_cast<int>(oa_cl.size()));
            }
        }

        // T x T: entries -> dense slots; column lists of L (rows below) and U (rows above)
        const int m = t.m_;
        std::vector<int> top_ent, dg_ent(m), fo_ptr{0}, fo_ent, fo_col;
        std::vector<std::vector<std::pair<int, int>>> lcol(m), ucol(m);
        for (int a = 0; a < m; ++a) {
            const int r = trows[a];
            dg_ent[a] = dp[r];
            for (int e = rp[r]; e < rp[r + 1]; ++e) {
                const int c = ci[e];
                if (!in_t[c]) {  // L(r, c outside T)
                    fo_ent.push_back(e);
                    fo_col.push_back(c);
                    continue;
                }
                top_ent.push_back(e);
                if (c < r) lcol[tpos[c]].push_back({a, e});
                if (c > r) ucol[tpos[c]].push_back({a, e});
            }
            fo_ptr.push_back(static_cast<int>(fo_ent.size()));
        }
        std::vector<int> rk_ptr{0}, rk, fl_ent, bu_ptr{0}, bu_row, bu_ent;
        for (int k = 0; k < m; ++k) {
            std::sort(lcol[k].begin(), lcol[k].end());
            std::sort(ucol[k].begin(), ucol[k].end());
            for (const auto& [i, e] : lcol[k]) { rk.push_back(i); fl_ent.push_back(e); }
            for (const auto& [i, e] : ucol[k]) { bu_row.push_back(i); bu_ent.push_back(e); }
            rk_ptr.push_back(static_cast<int>(rk.size()));
            bu_ptr.push_back(static_cast<int>(bu_row.size()));
        }
        t.n_top_ = static_cast<int>(top_ent.size());
        t.n_fl_ = static_cast<int>(rk.size());
        t.n_bu_ = static_cast<int>(bu_row.size());
        // positions in the matrix-major tm (q = index in top_ent) of the L / U lists and the diagonals; the T x T items
        // sorted by q (coalesced tile writes)
        {
            std::vector<int> q_of(n_ent_, -1);
            for (int q = 0; q < t.n_top_; ++q) q_of[top_ent[q]] = q;
            std::vector<int> fl_q, bu_q, dg_q(m);
            for (int e : fl_ent) fl_q.push_back(q_of[e]);
            for (int e : bu_ent) bu_q.push_back(q_of[e]);
            for (int a = 0; a < m; ++a) dg_q[a] = q_of[dg_ent[a]];
            up(t.d_fl_q_, fl_q);
            up(t.d_bu_q_, bu_q);
            up(t.d_dg_q_, dg_q);
            // the sparse factorization's slots: per local pivot k, its pattern rows R_k (rk, local, ascending): L slot
            // (i, k), U slot (k, i) (the pattern is symmetric), and the r x r update targets (i_a, i_c)
            std::vector<int> lslot, uslot, pp_ptr{0}, pslot;
            std::vector<int> qmap(static_cast<size_t>(m) * m, -1);  // local (row, column) -> q
            for (int q = 0; q < t.n_top_; ++q)
                qmap[static_cast<size_t>(tpos[erow[top_ent[q]]]) * m + tpos[ci[top_ent[q]]]] = q;
            auto qe = [&](int r, int c) {
                const int q = qmap[static_cast<size_t>(tpos[r]) * m + tpos[c]];
                if (q < 0) throw std::logic_error("[top_lu] T x T pattern not closed");
                return q;
            };
            for (int k = 0; k < m; ++k) {
                const int gk = trows[k];
                for (int a = rk_ptr[k]; a < rk_ptr[k + 1]; ++a) {
                    const int gi = trows[rk[a]];
                    lslot.push_back(qe(gi, gk));
                    uslot.push_back(qe(gk, gi));
                }
                for (int a = rk_ptr[k]; a < rk_ptr[k + 1]; ++a)
                    for (int c = rk_ptr[k]; c < rk_ptr[k + 1]; ++c) pslot.push_back(qe(trows[rk[a]], trows[rk[c]]));
                pp_ptr.push_back(static_cast<int>(pslot.size()));
            }
            up(t.d_lslot_, lslot);
            up(t.d_uslot_, uslot);
            up(t.d_pp_ptr_, pp_ptr);
            up(t.d_pslot_, pslot);
            std::vector<int> ord(oa_tgt.size());
            std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int x, int y) { return q_of[oa_tgt[x]] < q_of[oa_tgt[y]]; });
            std::vector<int> o_q, o_src, o_cptr{0}, o_cl, o_cu;
            for (int it : ord) {
                if (q_of[oa_tgt[it]] < 0) throw std::logic_error("[top_lu] a T x T item outside the T x T pattern");
                o_q.push_back(q_of[oa_tgt[it]]);
                o_src.push_back(oa_src[it]);
                for (int c = oa_cptr[it]; c < oa_cptr[it + 1]; ++c) { o_cl.push_back(oa_cl[c]); o_cu.push_back(oa_cu[c]); }
                o_cptr.push_back(static_cast<int>(o_cl.size()));
            }
            t.n_oa_ = static_cast<int>(o_q.size());
            up(t.d_oa_tgt_, o_q);
            up(t.d_oa_src_, o_src);
            up(t.d_oa_cptr_, o_cptr);
            up(t.d_oa_cl_, o_cl);
            up(t.d_oa_cu_, o_cu);
        }
        up(t.d_t_rows_, trows);
        up(t.d_top_ent_, top_ent);
        up(t.d_rk_ptr_, rk_ptr);
        up(t.d_rk_, rk);
        up(t.d_fl_ent_, fl_ent);
        up(t.d_fo_ptr_, fo_ptr);
        up(t.d_fo_ent_, fo_ent);
        up(t.d_fo_col_, fo_col);
        up(t.d_bu_ptr_, bu_ptr);
        up(t.d_bu_row_, bu_row);
        up(t.d_bu_ent_, bu_ent);
        up(t.d_dg_ent_, dg_ent);

        // the rows outside T: forward by the plan's levels, backward by their levels among themselves
        {
            std::vector<char> rest(n);
            for (int i = 0; i < n; ++i) rest[i] = !in_t[i];
            std::vector<int> rows, blev(n, 0);
            std::vector<std::array<int, 3>> sl;
            group(level, rest, fw_len, rows, sl);
            make_sched(t.fw_rest_, rows, sl);
            for (int i = n - 1; i >= 0; --i) {
                if (in_t[i]) continue;
                int mx = -1;
                for (int e = dp[i] + 1; e < rp[i + 1]; ++e)
                    if (!in_t[ci[e]]) mx = std::max(mx, blev[ci[e]]);
                blev[i] = mx + 1;
            }
            group(blev, rest, bw_len, rows, sl);
            make_sched(t.bw_rest_, rows, sl);
        }
        if (m > 0) {
            const int sf = static_cast<int>((static_cast<long long>(t.n_top_) + 2 * m) * sizeof(T));
            const int sw = static_cast<int>((m + t.n_fl_) * sizeof(T) + (m + 1 + t.n_fl_) * 4);  // top_fwd
            const int sb = static_cast<int>((3 * m + t.n_bu_) * sizeof(T) + (m + 1 + t.n_bu_) * 4);
            t.sf = sf;
            t.sw = sw;
            t.sb = sb;
        }
    };
    build_top(tops_[0], max_top);
    build_top(tops_[1], max_top > 0 ? std::min(max_top, cap_large) : cap_large);
    set_smem_attributes();
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_STATS"); e != nullptr && *e && *e != '0') {
        auto dump = [](const char* what, const std::vector<Launch>& ls) {
            std::fprintf(stderr, "[top_lu] %s: %zu launches\n", what, ls.size());
            for (size_t l = 0; l < ls.size(); ++l)
                std::fprintf(stderr, "  %3zu count %6d chain %4d k %2d%s\n", l, ls[l].count, ls[l].mc, ls[l].k,
                             static_cast<long long>(ls[l].count) * ls[l].k <= 32 ? " narrow" : "");
        };
        dump("factor below L0", tops_[0].f_bottom_);
        dump("forward outside T", tops_[0].fw_rest_.launches);
        dump("backward outside T", tops_[0].bw_rest_.launches);
    }
}

int TopLu::launches_factor() const
{
    const bool dense = tops_[0].m_ > 0 && !levels_only_;
    return 3 + static_cast<int>(dense ? tops_[0].f_bottom_.size() + 2 : f_all_.size());
}

int TopLu::launches_solve() const
{
    const bool dense = tops_[0].m_ > 0 && !levels_only_;
    return 2 + static_cast<int>(dense ? tops_[0].fw_rest_.steps.size() + tops_[0].bw_rest_.steps.size() + 3
                                      : fw_all_.steps.size() + bw_all_.steps.size());
}

void TopLu::set_active(int nb)
{
    if (nb < 1 || nb > B_) throw std::invalid_argument("[top_lu::TopLu::set_active] need 1 <= nb <= batch");
    if (nb == nb_) return;
    nb_ = nb;
    nreg_ = nb;   // until the next factorize(): no lane map
    nc_ = 0;
    lane_map_ = false;
}

TopLu::Factors TopLu::own_factors(int nb) const
{
    return Factors{d_v_.data(), raw(tops_[cur_].d_tm_), false, nb};
}

// ---- low-rank first steps ----

namespace {
// GPUSIM2GRID_OWN_LU_LOWRANK_STATS: wall time of a phase, the stream drained before and after (diagnostic only)
struct LrPhase {
    const char* what; cudaStream_t s; bool on; std::chrono::steady_clock::time_point t0;
    LrPhase(const char* w, cudaStream_t st, bool o) : what(w), s(st), on(o)
    {
        if (on) { cudaStreamSynchronize(s); t0 = std::chrono::steady_clock::now(); }
    }
    ~LrPhase()
    {
        if (!on) return;
        cudaStreamSynchronize(s);
        std::fprintf(stderr, "[top_lu]   %-22s %7.3f ms\n", what,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
};
}  // namespace

// The reference: the entrywise majority of three matrices of the first factorization -- the first, the middle and
// the last one: far apart in the caller's order, so unlikely to change the same rows (the first three contingencies
// of an N-1 often trip branches of one bus) --, factorized once. A reference whose factorization clamps a pivot is
// not the matrix the identity needs: the path stays off.
void TopLu::lr_setup_reference(const cuda_real_type* values)
{
    using T = cuda_real_type;
    if (nb_ < 3) return;
    d_jref_.alloc(static_cast<size_t>(nnz_));
    lr_ref_[0] = 0; lr_ref_[1] = nb_ / 2; lr_ref_[2] = nb_ - 1;
    pdl_launch(lr_majority<T>, dim3(blocks_for(nnz_)), dim3(kThreads), 0, stream_, values + static_cast<size_t>(lr_ref_[0]) * nnz_,
               values + static_cast<size_t>(lr_ref_[1]) * nnz_, values + static_cast<size_t>(lr_ref_[2]) * nnz_, nnz_,
               d_jref_.data());
    {
        LrPhase ph_("reference factorize", stream_, lr_stats_);
        ref_->factorize(d_jref_.data());
    }
    if (ref_->batch_clamp_total() != 0) {
        lr_failed_ = true;
        if (lr_stats_) std::fprintf(stderr, "[top_lu] low rank: the reference clamps a pivot, path off\n");
        return;
    }
    lr_ready_ = true;
}

// A_ref^-1 (winv[p * n + j] = A_ref^-1[j, order[p]]) from the reference's factors, its trailing block's entries
// first copied into them (the kernels read one array): the forward part along each unit vector's path, the
// backward part level by level, every column at once.
void TopLu::lr_build_inverse()
{
    using T = cuda_real_type;
    if (n_lev_ > kLrLev) { lr_failed_ = true; return; }
    try {
        d_winv_.alloc(static_cast<size_t>(n_) * n_);
    } catch (const std::exception&) {
        cudaGetLastError();
        lr_failed_ = true;
        return;
    }
    const Top& t = ref_->tops_[ref_->cur_];
    if (t.m_ > 0 && !ref_->levels_only_)
        pdl_launch(lr_densify<T>, dim3((t.n_top_ + 255) / 256), dim3(256), 0, stream_, raw(t.d_top_ent_), raw(t.d_tm_), t.n_top_,
                   ref_->d_v_.data());
    const T* v = ref_->d_v_.data();
    T* X = d_winv_.data();
    const int* rp = raw(d_rp_); const int* dp = raw(d_dp_); const int* ci = raw(d_ci_); const int* tpos = raw(d_tpos_);
    {
        LrPhase ph_("inverse: zero", stream_, lr_stats_);
        OWN_LU_CHECK(cudaMemsetAsync(X, 0, sizeof(T) * static_cast<size_t>(n_) * n_, stream_));
    }
    {
        LrPhase ph_("inverse: forward", stream_, lr_stats_);
        pdl_launch(lr_inv_walk<T>, dim3((n_ + 31) / 32), dim3(32), 0, stream_, raw(d_inv_), raw(d_lev_), n_lev_, rp, dp, ci, v, n_, X);
    }
    LrPhase ph_("inverse: backward", stream_, lr_stats_);
    for (auto L = fw_all_.launches.rbegin(); L != fw_all_.launches.rend(); ++L)
        pdl_launch(lr_inv_bwd<T>, dim3(blocks_for(static_cast<long long>(L->count) * n_)), dim3(kThreads), 0, stream_,
                   raw(fw_all_.d_rows) + L->offset, L->count, rp, dp, ci, tpos, v, n_, X);
    lr_inv_ready_ = true;
}

// Which matrices of this factorization take the low-rank path: those whose sampled entries are the reference's
// (but K at most), then whose rows are (but K at most), then whose capacitance matrix factorizes with pivots of
// comparable size. Sets the lanes (d_perm_: the others first, in order, then the candidates, in order).
void TopLu::lr_classify(const cuda_real_type* values)
{
    using T = cuda_real_type;
    const int nb = nb_, K = lr_K_;
    int* kind = rawm(d_kind_);
    int* cnt = rawm(d_lr_cnt_);
    OWN_LU_CHECK(cudaMemsetAsync(cnt + B_, 0, sizeof(int), stream_));
    pdl_launch(lr_sample<T>, dim3((nb + 255) / 256), dim3(256), 0, stream_, values, d_jref_.data(), raw(d_samp_), n_samp_,
               nnz_, nb, K, kind, cnt, cnt + B_);
    int n0 = 0;
    OWN_LU_CHECK(cudaMemcpyAsync(&n0, cnt + B_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    if (n0 < lr_min_) return;
    if (!lr_inv_ready_) {
        LrPhase ph_("inverse", stream_, lr_stats_);
        lr_build_inverse();
        if (lr_failed_) return;
    }
    LrPhase ph_diff("diff + capacitance", stream_, lr_stats_);
    pdl_launch(lr_diff<T>, dim3((n_ + 255) / 256, nb), dim3(256), 0, stream_, values, d_jref_.data(), raw(d_jptr_), n_, nnz_,
               K, kind, cnt, rawm(d_lr_rows_));
    pdl_launch(lr_capacitance<T>, dim3((nb + 31) / 32), dim3(32), 0, stream_, values, d_jref_.data(), raw(d_jptr_),
               raw(d_jcol_), raw(d_inv_), d_winv_.data(), n_, nnz_, nb, K, static_cast<T>(lr_max_ratio_), kind, cnt, rawm(d_lr_rows_),
               d_lr_S_.data(), rawm(d_lr_piv_));
    h_kind_.resize(static_cast<size_t>(nb));
    OWN_LU_CHECK(cudaMemcpyAsync(h_kind_.data(), kind, sizeof(int) * nb, cudaMemcpyDeviceToHost, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    int nc = 0;
    for (int b = 0; b < nb; ++b) nc += h_kind_[static_cast<size_t>(b)] >= 0;
    if (lr_stats_) {
        int hist[kLrMaxK + 1] = {0};
        for (int b = 0; b < nb; ++b)
            if (h_kind_[static_cast<size_t>(b)] >= 0) ++hist[h_kind_[static_cast<size_t>(b)]];
        std::fprintf(stderr, "[top_lu] low rank: %d matrices, %d kept by the samples, %d candidates, by rows:", nb, n0, nc);
        for (int k = 0; k <= K; ++k)
            if (hist[k]) std::fprintf(stderr, " %d:%d", k, hist[k]);
        std::fprintf(stderr, "\n");
    }
    if (nc < lr_min_) return;
    h_perm_.resize(static_cast<size_t>(nb));
    int r = 0, c = nb - nc;
    for (int b = 0; b < nb; ++b) h_perm_[static_cast<size_t>(h_kind_[static_cast<size_t>(b)] >= 0 ? c++ : r++)] = b;
    OWN_LU_CHECK(cudaMemcpyAsync(rawm(d_perm_), h_perm_.data(), sizeof(int) * nb, cudaMemcpyHostToDevice, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));   // h_perm_ may be rewritten by the next call
    nreg_ = nb - nc;
    nc_ = nc;
    lane_map_ = true;
}

// The columns mode's test and data: the reference right-hand side (the majority of the reference matrix's three
// rows: set at the first solve) and its solution z_ref, then each candidate's rows off it. The
// candidates with more than lr_Kf_ of them are solved on the reference's factors: their lanes come first
// (the candidates' lanes hold nothing yet), d_lr_cold_ keeping where each one was when compared.
int TopLu::lr_prepare_rhs(const cuda_real_type* b)
{
    using T = cuda_real_type;
    if (lr_Kf_ <= 0 || (!lr_f_ready_ && nb_ <= lr_ref_[2])) return nc_;
    if (!lr_f_ready_) {
        d_fref_.alloc(static_cast<size_t>(n_));
        d_zref_.alloc(static_cast<size_t>(n_));
        pdl_launch(lr_majority<T>, dim3(blocks_for(n_)), dim3(kThreads), 0, stream_, b + static_cast<size_t>(lr_ref_[0]) * n_,
                   b + static_cast<size_t>(lr_ref_[1]) * n_, b + static_cast<size_t>(lr_ref_[2]) * n_, n_, d_fref_.data());
        ref_->solve(d_fref_.data(), d_zref_.data());
        ref_->flush();
        lr_f_ready_ = true;
    }
    int* cntf = rawm(d_lr_cntf_);
    OWN_LU_CHECK(cudaMemsetAsync(cntf, 0, sizeof(int) * nc_, stream_));
    pdl_launch(lr_fdiff<T>, dim3((n_ + 255) / 256, nc_), dim3(256), 0, stream_, b, d_fref_.data(), n_, lr_Kf_,
               raw(d_perm_) + nreg_, cntf, rawm(d_lr_frows_));
    h_cntf_.resize(static_cast<size_t>(nc_));
    OWN_LU_CHECK(cudaMemcpyAsync(h_cntf_.data(), cntf, sizeof(int) * nc_, cudaMemcpyDeviceToHost, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    int nB = 0;
    for (int c = 0; c < nc_; ++c) nB += h_cntf_[static_cast<size_t>(c)] > lr_Kf_;
    // lanes: the solved candidates, then the columns ones (both in order); h_tail_: [matrices | former lanes]
    h_tail_.resize(2 * static_cast<size_t>(nc_));
    int iB = 0, iA = nB;
    for (int c = 0; c < nc_; ++c) {
        const int at = h_cntf_[static_cast<size_t>(c)] > lr_Kf_ ? iB++ : iA++;
        h_tail_[static_cast<size_t>(at)] = h_perm_[static_cast<size_t>(nreg_ + c)];
        h_tail_[static_cast<size_t>(nc_ + at)] = c;
    }
    for (int c = 0; c < nc_; ++c) h_perm_[static_cast<size_t>(nreg_ + c)] = h_tail_[static_cast<size_t>(c)];
    OWN_LU_CHECK(cudaMemcpyAsync(rawm(d_perm_) + nreg_, h_tail_.data(), sizeof(int) * nc_, cudaMemcpyHostToDevice, stream_));
    OWN_LU_CHECK(cudaMemcpyAsync(rawm(d_lr_cold_), h_tail_.data() + nc_, sizeof(int) * nc_, cudaMemcpyHostToDevice, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));   // h_tail_ may be rewritten by the next call
    if (lr_stats_) std::fprintf(stderr, "[top_lu] low rank: %d candidates solved on the reference, %d by columns\n", nB, nc_ - nB);
    return nB;
}

// The candidates' lanes: z on the reference's factors, then the coefficients of the correction (lu_tr_out_lr
// applies it when the solution is written out)
void TopLu::lr_solve_candidates(int nB)
{
    using T = cuda_real_type;
    const int rc = ref_->cur_;
    const Factors f{ref_->d_v_.data(), raw(ref_->tops_[rc].d_tm_), true, nB};
    T* xc = d_xi_.data() + nreg_;
    {
        LrPhase ph_("solve: shared", stream_, lr_stats_);
        solve_core(tops_[rc], xc, nullptr, f);
    }
    LrPhase ph_("solve: coefficients", stream_, lr_stats_);
    pdl_launch(lr_coeff<T>, dim3((nB + 31) / 32), dim3(32), 0, stream_, last_values_, d_jref_.data(), raw(d_jptr_),
               raw(d_jcol_), raw(d_inv_), xc, B_, nnz_, nB, lr_K_, raw(d_perm_) + nreg_, raw(d_kind_), raw(d_lr_rows_),
               d_lr_S_.data(), raw(d_lr_piv_), d_lr_y_.data());
}

void TopLu::write_out(cuda_real_type* x, const cuda_real_type* add, const int* mask)
{
    using T = cuda_real_type;
    const int* perm = lane_map_ ? raw(d_perm_) : nullptr;
    dim3 tblock(32, 8);
    LrPhase ph_("solve: x out", stream_, lr_stats_ && lr_on_);
    if (nreg_ > 0) {
        dim3 tgrid((n_ + 31) / 32, (nreg_ + 31) / 32);
        if (add != nullptr)
            pdl_launch(lu_tr_out_add<T>, dim3(tgrid), dim3(tblock), 0, stream_, d_xi_.data(), add, raw(d_inv_), n_, B_, nreg_,
                       x, mask, perm);
        else
            pdl_launch(lu_tr_out<T>, dim3(tgrid), dim3(tblock), 0, stream_, d_xi_.data(), raw(d_inv_), n_, B_, nreg_, x, perm);
    }
    if (pending_lr_ && lr_nB_ > 0) {
        dim3 tgrid((n_ + 31) / 32, (lr_nB_ + 31) / 32);
        pdl_launch(lu_tr_out_lr<T>, dim3(tgrid), dim3(tblock), 0, stream_, d_xi_.data() + nreg_, raw(d_inv_), n_, B_, lr_nB_, x,
                   raw(d_perm_) + nreg_, raw(d_kind_), raw(d_lr_rows_), d_lr_y_.data(), d_winv_.data(), lr_K_);
    }
    if (pending_lr_ && lr_nA_ > 0) {
        pdl_launch(lr_out_cols<T>, dim3(blocks_for(static_cast<long long>(n_) * lr_nA_)), dim3(kThreads), 0, stream_,
                   d_zref_.data(), d_winv_.data(), n_, lr_nA_, lr_K_ + lr_Kf_, raw(d_perm_) + nreg_ + lr_nB_, raw(d_lr_ncol_),
                   raw(d_lr_colrow_), d_lr_coef_.data(), x);
    }
    out_pending_ = false;
    pending_lr_ = false;
}

void TopLu::factorize(const cuda_real_type* values)
{
    using T = cuda_real_type;
    lane_map_ = false;
    nreg_ = nb_;
    nc_ = 0;
    if (lr_on_ && !lr_failed_) {
        LrPhase ph_("classification (all)", stream_, lr_stats_);
        if (!lr_ready_) lr_setup_reference(values);
        if (lr_ready_) lr_classify(values);
    }
    LrPhase ph_f("factorization", stream_, lr_stats_ && lr_on_);
    const int B = B_, nb = nreg_;   // the lanes factorized
    const int* perm = lane_map_ ? raw(d_perm_) : nullptr;
    T* v = rawm(d_v_);
    T* ai = rawm(d_ai_);
    T* tol = rawm(d_tol_);
    int* clamped = rawm(d_clamped_) + 1;   // per lane; clamped[-1]: the batch's total
    OWN_LU_CHECK(cudaMemsetAsync(clamped - 1, 0, sizeof(int) * (nb_ + 1), stream_));
    OWN_LU_CHECK(cudaMemsetAsync(rawm(d_amax_), 0, sizeof(unsigned long long) * nb_, stream_));
    cur_ = force_cur_ >= 0 ? force_cur_ : ((nb > top_nb_ && tops_[1].m_ > 0) ? 1 : 0);  // the solves that follow use the same T
    ++fact_gen_;
    last_values_ = values;
    if (nb == 0) {
        OWN_LU_CHECK(cudaGetLastError());
        return;
    }
    {   // J -> interleaved A + max |A_b|, then the clamps
        const int tiles = (nnz_ + 31) / 32;
        dim3 grid(std::min(tiles, 1024), (nb + 31) / 32), block(32, 8);
        pdl_launch(lu_tr_in_vals<T>, dim3(grid), dim3(block), 0, stream_, values, nnz_, B, nb, ai, rawm(d_amax_), perm);
        pdl_launch(lu_tol_from_max<T>, dim3((nb + 255) / 256), dim3(256), 0, stream_, raw(d_amax_), eps_, nb, tol);
    }
    const int* tgt = raw(d_tgt_); const int* src = raw(d_src_); const int* cptr = raw(d_cptr_);
    const int* dg = raw(d_dg_); const int* cl = raw(d_cl_); const int* cu = raw(d_cu_);
    Top& t = tops_[cur_];
    const bool dense = t.m_ > 0 && !levels_only_;
    for (const Launch& L : dense ? t.f_bottom_ : f_all_) {
        const int k = split_for(L.k, L.count, L.mc, nb);
        if (k > 1) {
            dim3 grid(L.count, (nb + 31) / 32);
            pdl_launch(lu_left_split<T>, dim3(grid), dim3(32 * k), lu_split_smem<T>(k), stream_, tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu,
                                                             dg + L.offset, B, nb, L.is_u ? 1 : 0, tol, clamped, v);
        } else if (L.is_u) {
            pdl_launch(lu_left_u<T>, dim3(blocks_for((long long)L.count * nb)), dim3(kThreads), 0, stream_,
                tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, nb, tol, clamped, v);
        } else {
            pdl_launch(lu_left_l<T>, dim3(blocks_for((long long)L.count * nb)), dim3(kThreads), 0, stream_,
                tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, nb, v);
        }
    }
    if (dense) {
        pdl_launch(top_outside<T>, dim3(dim3((t.n_oa_ + 31) / 32, (nb + 31) / 32)), dim3(dim3(32, 8)), 0, stream_,
            raw(t.d_oa_tgt_), raw(t.d_oa_src_), ai, raw(t.d_oa_cptr_), raw(t.d_oa_cl_), raw(t.d_oa_cu_), t.n_oa_, t.n_top_, B, nb, v, rawm(t.d_tm_));
        const int m = t.m_;
        const size_t sf = (static_cast<size_t>(t.n_top_) + 2 * m) * sizeof(T);
        pdl_launch(top_factor<T>, dim3(nb), dim3(kTopFactorThreads), sf, stream_, m, t.n_top_, raw(t.d_dg_q_), raw(t.d_rk_ptr_), raw(t.d_lslot_),
                                                               raw(t.d_uslot_), raw(t.d_pp_ptr_), raw(t.d_pslot_), B, nb, tol,
                                                               clamped, rawm(t.d_tm_));
    }
    OWN_LU_CHECK(cudaGetLastError());
}

// a solve kernel on this instance's factors or on shared ones (Factors::shared)
#define TOP_LAUNCH_SH(K, ...)                                \
    do {                                                     \
        if (f.shared) pdl_launch(K<T, true>, __VA_ARGS__);   \
        else pdl_launch(K<T, false>, __VA_ARGS__);           \
    } while (0)

void TopLu::run_solve(const Sched& s, bool forward, cuda_real_type* xi, const int* mask, const Factors& f)
{
    using T = cuda_real_type;
    const int B = B_, nb = f.nb;
    const T* v = f.v;
    const int* rp = raw(d_rp_); const int* dp = raw(d_dp_); const int* ci = raw(d_ci_);
    const int n_lv = static_cast<int>(s.launches.size());
    auto level = [&](const Launch& L) {
        const int* rows = raw(s.d_rows) + L.offset;
        const int k = split_for(L.k, L.count, L.mc, nb);
        if (forward) {
            if (k > 1) TOP_LAUNCH_SH(lu_fwd_split, dim3(dim3(L.count, (nb + 31) / 32)), dim3(32 * k), lu_split_smem<T>(k), stream_, rows, rp, dp, ci, B, nb, v, xi, mask);
            else TOP_LAUNCH_SH(lu_fwd, dim3(blocks_for((long long)L.count * nb)), dim3(kThreads), 0, stream_, rows, rp, dp, ci, L.count, B, nb, v, xi, mask);
        } else {
            if (k > 1) TOP_LAUNCH_SH(lu_bwd_split, dim3(dim3(L.count, (nb + 31) / 32)), dim3(32 * k), lu_split_smem<T>(k), stream_, rows, rp, dp, ci, B, nb, v, xi, mask);
            else TOP_LAUNCH_SH(lu_bwd, dim3(blocks_for((long long)L.count * nb)), dim3(kThreads), 0, stream_, rows, rp, dp, ci, L.count, B, nb, v, xi, mask);
        }
    };
    // a fused run is one block per 32 matrices: with many active matrices it leaves most of the GPU idle (issue
    // #44: one launch per level is faster there), so it is used only up to fuse_nb_ active matrices
    const bool fuse = nb <= fuse_nb_;
    for (const SolveStep& st : s.steps) {
        if (st.launch < 0) {
            if (!fuse) {
                for (int l = st.first; l < st.last; ++l) level(s.launches[static_cast<size_t>(l)]);
                continue;
            }
            const dim3 grid((nb + 31) / 32), block(kRunThreads);
            if (forward) {
                if (f.shared) pdl_launch(lu_solve_run<T, true, true>, dim3(grid), dim3(block), 0, stream_, raw(s.d_rows), raw(s.d_run), st.first, st.last, n_lv, rp, dp, ci, B, nb, v, xi, mask);
                else pdl_launch(lu_solve_run<T, true, false>, dim3(grid), dim3(block), 0, stream_, raw(s.d_rows), raw(s.d_run), st.first, st.last, n_lv, rp, dp, ci, B, nb, v, xi, mask);
            } else {
                if (f.shared) pdl_launch(lu_solve_run<T, false, true>, dim3(grid), dim3(block), 0, stream_, raw(s.d_rows), raw(s.d_run), st.first, st.last, n_lv, rp, dp, ci, B, nb, v, xi, mask);
                else pdl_launch(lu_solve_run<T, false, false>, dim3(grid), dim3(block), 0, stream_, raw(s.d_rows), raw(s.d_run), st.first, st.last, n_lv, rp, dp, ci, B, nb, v, xi, mask);
            }
            continue;
        }
        level(s.launches[static_cast<size_t>(st.launch)]);
    }
}

void TopLu::solve_core(const Top& t, cuda_real_type* xi, const int* mask, const Factors& f)
{
    using T = cuda_real_type;
    const int B = B_, nb = f.nb;
    if (nb <= 0) return;
    if (t.m_ > 0 && !levels_only_) {
        const int m = t.m_;
        run_solve(t.fw_rest_, true, xi, mask, f);
        const size_t sw = (m + t.n_fl_) * sizeof(T) + (m + 1 + t.n_fl_) * 4;
        TOP_LAUNCH_SH(top_fo, dim3(dim3(m, (nb + 31) / 32)), dim3(32 * 8), 0, stream_, raw(t.d_t_rows_), raw(t.d_fo_ptr_), raw(t.d_fo_ent_), raw(t.d_fo_col_),
                                                                 B, nb, f.v, xi, mask);
        TOP_LAUNCH_SH(top_fwd, dim3(nb), dim3(kTopSolveThreads), sw, stream_, m, t.n_top_, raw(t.d_t_rows_), raw(t.d_rk_ptr_), raw(t.d_rk_), raw(t.d_fl_q_), t.n_fl_,
                                                           B, nb, f.tm, xi, mask);
        const size_t sb = (3 * m + t.n_bu_) * sizeof(T) + (m + 1 + t.n_bu_) * 4;
        TOP_LAUNCH_SH(top_bwd, dim3(nb), dim3(kTopSolveThreads), sb, stream_, m, t.n_top_, raw(t.d_t_rows_), raw(t.d_bu_ptr_), raw(t.d_bu_row_), raw(t.d_bu_q_),
                                                           t.n_bu_, raw(t.d_dg_q_), B, nb, f.tm, xi, mask);
        run_solve(t.bw_rest_, false, xi, mask, f);
    } else {
        run_solve(fw_all_, true, xi, mask, f);
        run_solve(bw_all_, false, xi, mask, f);
    }
}

#undef TOP_LAUNCH_SH

void TopLu::solve(const cuda_real_type* b, cuda_real_type* x) { solve_impl(b, x, nullptr, true); }

void TopLu::solve_impl(const cuda_real_type* b, cuda_real_type* x, const int* mask, bool with_lowrank)
{
    using T = cuda_real_type;
    const int B = B_, nb = nb_;
    T* xi = rawm(d_xi_);
    const int* perm = lane_map_ ? raw(d_perm_) : nullptr;
    int nB = 0, nA = 0;   // the low-rank lanes: solved on the reference's factors, by columns
    if (with_lowrank && nc_ > 0) {
        LrPhase ph_("solve: rhs rows", stream_, lr_stats_);
        nB = lr_prepare_rhs(b);
        nA = nc_ - nB;
    }
    const int nin = nreg_ + nB;   // the lanes whose right-hand side is solved on
    if (nin > 0) {
        LrPhase ph_("solve: b in", stream_, lr_stats_ && lr_on_);
        dim3 tgrid((n_ + 31) / 32, (nin + 31) / 32), tblock(32, 8);
        pdl_launch(lu_tr_in2<T>, dim3(tgrid), dim3(tblock), 0, stream_, b, raw(d_inv_), n_, B, nin, rawm(d_bi_), xi, perm);
    }
    {
        LrPhase ph_("solve: factorized", stream_, lr_stats_ && lr_on_);
        solve_core(tops_[cur_], xi, mask, own_factors(nreg_));
    }
    if (nB > 0) lr_solve_candidates(nB);
    if (nA > 0) {
        LrPhase ph_("solve: column coeffs", stream_, lr_stats_);
        pdl_launch(lr_coeff_cols<T>, dim3((nA + 31) / 32), dim3(32), 0, stream_, last_values_, d_jref_.data(), raw(d_jptr_),
                   raw(d_jcol_), raw(d_inv_), b, d_fref_.data(), d_zref_.data(), d_winv_.data(), n_, nnz_, nA, lr_K_, lr_Kf_,
                   raw(d_perm_) + nreg_ + nB, raw(d_kind_), raw(d_lr_rows_), d_lr_S_.data(), raw(d_lr_piv_), raw(d_lr_cntf_),
                   rawm(d_lr_frows_), raw(d_lr_cold_) + nB, rawm(d_lr_ncol_), rawm(d_lr_colrow_), d_lr_coef_.data());
    }
    pending_lr_ = nB + nA > 0;
    lr_nB_ = nB;
    lr_nA_ = nA;
    // x is written back by refine() (once, with the correction: x + d) or by flush(); solve() leaves it pending
    out_pending_ = true;
    // the interleaved b and x stay valid for refine() while nothing was factorized in between and this is the
    // first solve on these factors (another solve could follow on other values: refine then reads them)
    interleaved_ok_ = solve_gen_ != fact_gen_;
    solve_gen_ = fact_gen_;
    last_b_ = b;
    last_x_ = x;
    OWN_LU_CHECK(cudaGetLastError());
}

void TopLu::refine(const cuda_real_type* values, const int* indptr, const int* indices, const cuda_real_type* b,
                   cuda_real_type* x, int steps)
{
    using T = cuda_real_type;
    if (lr_stats_ && lr_on_ && out_pending_ && x == last_x_ && b == last_b_ && std::getenv("GPUSIM2GRID_OWN_LU_LOWRANK_CHECK")) {   // diagnostic: every matrix's residual
        flush();
        thrust::device_vector<double> d_out(2 * static_cast<size_t>(nb_));
        pdl_launch(lr_dbg_residual<T>, dim3(nb_), dim3(256), 0, stream_, values, indptr, indices, n_, nnz_, b, x, rawm(d_out));
        OWN_LU_CHECK(cudaStreamSynchronize(stream_));
        std::vector<double> h(2 * static_cast<size_t>(nb_));
        OWN_LU_CHECK(cudaMemcpy(h.data(), rawm(d_out), sizeof(double) * h.size(), cudaMemcpyDeviceToHost));
        double worst[2] = {0., 0.};
        int bad[2] = {0, 0}, cnt[2] = {0, 0};
        for (int m = 0; m < nb_; ++m) {
            const int lr = lane_map_ && h_kind_[static_cast<size_t>(m)] >= 0 ? 1 : 0;
            const double rel = h[2 * m] / std::max(h[2 * m + 1], 1e-300);
            ++cnt[lr];
            if (!(rel <= worst[lr])) worst[lr] = rel;
            if (!(rel <= 1e-8)) ++bad[lr];
        }
        std::fprintf(stderr, "[top_lu] residuals max|b - A x| / max|b|: factorized %d lanes worst %.2e (%d above 1e-8), "
                     "low rank %d lanes worst %.2e (%d above 1e-8)\n", cnt[0], worst[0], bad[0], cnt[1], worst[1], bad[1]);
    }
    if (steps <= 0) { flush(); return; }
    if (static_cast<const void*>(x) == static_cast<const void*>(b))
        throw std::invalid_argument("top_lu::TopLu::refine: x aliases b");
    // Nothing clamped anywhere in the batch (the usual case): nothing to refine, and not even the masked
    // correction's launch chain (~160 nearly empty kernels a round) -- one 4-byte read of the batch's clamp
    // total, after a host sync the round needs soon anyway. The low-rank lanes are never refined (nothing of
    // theirs was factorized, hence clamped).
    if (nreg_ == 0 || (!refine_all_ && batch_clamp_total() == 0)) { flush(); return; }
    const int B = B_, nb = nreg_;
    const long long mi = static_cast<long long>(n_) * nb;
    // only the lanes whose factorization clamped a pivot (the others' x is left as solve() gave it)
    const int* mask = refine_all_ ? nullptr : raw(d_clamped_) + 1;
    if (interleaved_ok_ && b == last_b_ && x == last_x_ && values == last_values_) {
        // in the interleaved layout: r = b - A x on the copy of A the factorization transposed, the correction
        // solved in place, x + d written back by the last transpose
        T* xi = rawm(d_xi_);
        T* ri = rawm(d_ri_);
        for (int s = 0; s < steps; ++s) {
            pdl_launch(lu_residual_i<T>, dim3(blocks_for(mi)), dim3(kThreads), 0, stream_, raw(d_ra_ptr_), raw(d_ra_p_), raw(d_ra_col_), n_, B, nb,
                                                                       raw(d_ai_), raw(d_bi_), xi, ri, mask);
            solve_core(tops_[cur_], ri, mask, own_factors(nb));
            if (s + 1 < steps) pdl_launch(lu_axpy_i<T>, dim3(blocks_for(mi)), dim3(kThreads), 0, stream_, n_, B, nb, ri, xi, mask);
            else write_out(x, ri, mask);
        }
        interleaved_ok_ = false;
        OWN_LU_CHECK(cudaGetLastError());
        return;
    }
    flush();  // the slot-major path reads x
    if (d_r_.size() != static_cast<size_t>(n_) * B_) {
        d_r_.resize(static_cast<size_t>(n_) * B_);
        d_d_.resize(static_cast<size_t>(n_) * B_);
    }
    const long long mm = static_cast<long long>(n_) * nb_;
    const int* smask = mask;   // by matrix: the lanes' through the map
    if (lane_map_) {
        pdl_launch(lr_slot_mask, dim3((nb_ + 255) / 256), dim3(256), 0, stream_, raw(d_perm_), mask, nreg_, nb_, rawm(d_slot_mask_));
        smask = raw(d_slot_mask_);
    }
    for (int s = 0; s < steps; ++s) {
        pdl_launch(lu_residual<T>, dim3(blocks_for(mm)), dim3(kThreads), 0, stream_, values, indptr, indices, n_, nnz_, nb_, b, x, rawm(d_r_), smask);
        solve_impl(rawm(d_r_), rawm(d_d_), mask, false);
        flush();
        pdl_launch(lu_axpy<T>, dim3(blocks_for(mm)), dim3(kThreads), 0, stream_, mm, n_, raw(d_d_), x, smask);
    }
    interleaved_ok_ = false;
    OWN_LU_CHECK(cudaGetLastError());
}

void TopLu::flush()
{
    if (!out_pending_) return;
    write_out(last_x_, nullptr, nullptr);
    OWN_LU_CHECK(cudaGetLastError());
}

int TopLu::batch_clamp_total() const
{
    int total = 0;
    OWN_LU_CHECK(cudaMemcpyAsync(&total, raw(d_clamped_), sizeof(int), cudaMemcpyDeviceToHost, stream_));
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    return total;
}

std::vector<int> TopLu::clamped_counts() const
{
    std::vector<int> h(nb_, 0);
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    if (!lane_map_) {
        OWN_LU_CHECK(cudaMemcpy(h.data(), raw(d_clamped_) + 1, sizeof(int) * nb_, cudaMemcpyDeviceToHost));
        return h;
    }
    std::vector<int> lanes(static_cast<size_t>(std::max(nreg_, 1)), 0);   // the factorized lanes' counts, by matrix
    if (nreg_ > 0) OWN_LU_CHECK(cudaMemcpy(lanes.data(), raw(d_clamped_) + 1, sizeof(int) * nreg_, cudaMemcpyDeviceToHost));
    for (int c = 0; c < nreg_; ++c) h[static_cast<size_t>(h_perm_[static_cast<size_t>(c)])] = lanes[static_cast<size_t>(c)];
    return h;
}

}  // namespace top_lu
