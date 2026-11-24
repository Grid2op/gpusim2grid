// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// fused_lu/top_lu.cu -- see top_lu.cuh. The level kernels are own_lu/batched_lu.cu's, copied (prototype);
// top_outside / top_factor / top_fwd / top_bwd are new.

#include "top_lu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <array>

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
#define LU_ITEM_AND_MATRIX                                                  \
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (long long)n_items * nb) return;                            \
    const int it = (int)(idx / nb);                                        \
    const int b = (int)(idx - (long long)it * nb);

// Item targets: entry t, or ~t for a fill entry (its sum starts at 0).
#define LU_TARGET(raw) ((raw) < 0 ? ~(raw) : (raw))
#define LU_START(raw, v, t, B, b) ((raw) < 0 ? T(0) : (v)[(size_t)(t) * (B) + (b)])

template <class T>
__device__ __forceinline__ T clamp_pivot(T s, T e, int* clamped, int b)
{
    if (fabs(s) < e) {
        atomicAdd(clamped + b, 1);
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
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = (src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b]);
    for (int c = cptr[it]; c < cptr[it + 1]; ++c)
        s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
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
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = (src[it] < 0 ? T(0) : ai[(size_t)src[it] * B + b]);
    for (int c = cptr[it]; c < cptr[it + 1]; ++c)
        s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
    v[(size_t)t * B + b] = s / v[(size_t)dg[it] * B + b];
}

// Forward (unit L): x[i] -= sum L[i,k] x[k]; backward: x[i] = (x[i] - sum U[i,j] x[j]) / U[i,i].
template <class T>
__global__ void lu_fwd(const int* __restrict__ rows, const int* __restrict__ rp,
                       const int* __restrict__ dp, const int* __restrict__ ci, const int n_items,
                       const int B, const int nb, const T* __restrict__ v, T* __restrict__ x)
{
    LU_ITEM_AND_MATRIX
    const int i = rows[it];
    T s = x[(size_t)i * B + b];
    for (int e = rp[i]; e < dp[i]; ++e) s = fma(-v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], s);
    x[(size_t)i * B + b] = s;
}

template <class T>
__global__ void lu_bwd(const int* __restrict__ rows, const int* __restrict__ rp,
                       const int* __restrict__ dp, const int* __restrict__ ci, const int n_items,
                       const int B, const int nb, const T* __restrict__ v, T* __restrict__ x)
{
    LU_ITEM_AND_MATRIX
    const int i = rows[it];
    T s = x[(size_t)i * B + b];
    for (int e = dp[i] + 1; e < rp[i + 1]; ++e) s = fma(-v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], s);
    x[(size_t)i * B + b] = s / v[(size_t)dp[i] * B + b];
}

// Layout changes, 32 x 32 tiles (block 32 x 8), matrices b < nb only.
// in:  dst[map[j] * B + b] = src[b * ncols + j]; out: dst[b * ncols + j] = src[map[j] * B + b]
template <class T>
__global__ void lu_tr_in(const T* __restrict__ src, const int* __restrict__ map, const int ncols,
                         const int B, const int nb, T* __restrict__ dst)
{
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
                          const int B, const int nb, T* __restrict__ dst)
{
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)map[j] * B + b];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) dst[(size_t)b * ncols + j] = t[threadIdx.x][r];
    }
}

// tol[b] = eps * max_j |values[b * nnz + j]| (one block of 256 per matrix)
template <class T>
__global__ void lu_scaled_tol(const T* __restrict__ values, const int nnz, const T eps, T* __restrict__ tol)
{
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
#define LU_SPLIT_THREAD                                         \
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5; \
    const int K = blockDim.x >> 5;                              \
    const int it = blockIdx.x;                                  \
    const int b = blockIdx.y * 32 + lane;   /* b < nb: active */ \
    __shared__ T red[32][33];

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

template <class T>
__global__ void lu_fwd_split(const int* __restrict__ rows, const int* __restrict__ rp,
                             const int* __restrict__ dp, const int* __restrict__ ci, const int B,
                             const int nb, const T* __restrict__ v, T* __restrict__ x)
{
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < nb)
        for (int e = rp[i] + part; e < dp[i]; e += K) acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= nb) return;
    x[(size_t)i * B + b] -= sum;
}

template <class T>
__global__ void lu_bwd_split(const int* __restrict__ rows, const int* __restrict__ rp,
                             const int* __restrict__ dp, const int* __restrict__ ci, const int B,
                             const int nb, const T* __restrict__ v, T* __restrict__ x)
{
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < nb)
        for (int e = dp[i] + 1 + part; e < rp[i + 1]; e += K)
            acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= nb) return;
    x[(size_t)i * B + b] = (x[(size_t)i * B + b] - sum) / v[(size_t)dp[i] * B + b];
}

constexpr int kRunThreads = 1024;   // lu_solve_run's block: 32 matrices (lanes) x 32 warps

// A run of narrow levels of a triangular solve in one kernel: block = 32 consecutive matrices (lane) x
// 32 warps, the levels in order with a block barrier between them (a matrix depends on its own rows
// only). In a level of k parts, warps item * k ... item * k + k - 1 take item `item`; every value is
// computed as lu_fwd / lu_bwd (k = 1) or lu_fwd_split / lu_bwd_split (k > 1) do: the same bits.
template <class T, bool kForward>
__global__ void __launch_bounds__(kRunThreads) lu_solve_run(const int* __restrict__ rows, const int* __restrict__ run, const int first,
                             const int last, const int n_lv, const int* __restrict__ rp, const int* __restrict__ dp,
                             const int* __restrict__ ci, const int B, const int nb, const T* __restrict__ v,
                             T* __restrict__ x)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int b = blockIdx.x * 32 + lane;
    __shared__ T red[32][33];
    for (int l = first; l < last; ++l) {
        const int offset = run[l], count = run[n_lv + l], K = run[2 * n_lv + l];
        const int item = warp / K, part = warp - item * K;
        const bool active = item < count && b < nb;
        const int i = item < count ? rows[offset + item] : 0;
        if (K == 1) {
            if (active) {
                T s = x[(size_t)i * B + b];
                if (kForward) {
                    for (int e = rp[i]; e < dp[i]; ++e) s = fma(-v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], s);
                    x[(size_t)i * B + b] = s;
                } else {
                    for (int e = dp[i] + 1; e < rp[i + 1]; ++e) s = fma(-v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], s);
                    x[(size_t)i * B + b] = s / v[(size_t)dp[i] * B + b];
                }
            }
        } else {
            T acc = T(0);
            if (active) {
                if (kForward) {
                    for (int e = rp[i] + part; e < dp[i]; e += K) acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
                } else {
                    for (int e = dp[i] + 1 + part; e < rp[i + 1]; e += K)
                        acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
                }
            }
            red[warp][lane] = acc;
            __syncthreads();
            if (part == 0 && active) {
                T sum = T(0);
                for (int w = 0; w < K; ++w) sum += red[item * K + w][lane];
                if (kForward) x[(size_t)i * B + b] -= sum;
                else x[(size_t)i * B + b] = (x[(size_t)i * B + b] - sum) / v[(size_t)dp[i] * B + b];
            }
        }
        __syncthreads();   // the level's rows before the next level reads them
    }
}

// r[b, i] = rhs[b, i] - (A_b x_b)[i], slot-major (thread = (matrix, row), row fastest)
template <class T>
__global__ void lu_residual(const T* __restrict__ values, const int* __restrict__ indptr,
                            const int* __restrict__ indices, const int n, const int nnz, const int B,
                            const T* __restrict__ rhs, const T* __restrict__ x, T* __restrict__ r)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * B) return;
    const int b = (int)(idx / n), i = (int)(idx - (long long)b * n);
    const T* a = values + (size_t)b * nnz;
    const T* xb = x + (size_t)b * n;
    T s = rhs[idx];
    for (int p = indptr[i]; p < indptr[i + 1]; ++p) s = fma(-a[p], xb[indices[p]], s);
    r[idx] = s;
}

template <class T>
__global__ void lu_axpy(const long long m, const T* __restrict__ d, T* __restrict__ x)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) x[idx] += d[idx];
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
template <class T>
__global__ void top_fo(const int* __restrict__ trows, const int* __restrict__ fo_ptr, const int* __restrict__ fo_ent,
                       const int* __restrict__ fo_col, const int B, const int nb, const T* __restrict__ v, T* __restrict__ xi)
{
    __shared__ T red[32][33];
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5, K = blockDim.x >> 5, i = blockIdx.x;
    const int b = blockIdx.y * 32 + lane;
    T acc = T(0);
    if (b < nb)
        for (int a = fo_ptr[i] + part; a < fo_ptr[i + 1]; a += K) acc = fma(v[(size_t)fo_ent[a] * B + b], xi[(size_t)fo_col[a] * B + b], acc);
    red[part][lane] = acc;
    __syncthreads();
    if (part == 0 && b < nb) {
        T s = T(0);
        for (int w = 0; w < K; ++w) s += red[w][lane];
        xi[(size_t)trows[i] * B + b] -= s;
    }
}

constexpr int kTopFactorThreads = 512, kTopSolveThreads = 256;

// One block per matrix: T x T dense in shared memory (m x m), right-looking with static pivots clamped as
// own_lu. Step k: thread 0 clamps the pivot and takes its reciprocal; then the update of the pairs of the
// pattern rows of column k (kr[kp[k] .. kp[k+1]), local) and the scaling of column k - 1 (deferred: column k
// is read unscaled during step k). L entries end divided by their pivot, U entries keep it (own_lu's layout).
template <class T>
__global__ void __launch_bounds__(kTopFactorThreads)
top_factor(const int m, const int n_top, const int* __restrict__ top_pos,
           const int* __restrict__ rk_ptr, const int* __restrict__ rk, const int n_rk, const int B, const int nb,
           const T* __restrict__ tol, int* __restrict__ clamped, T* __restrict__ tm)
{
    extern __shared__ __align__(16) unsigned char sm[];
    T* D = reinterpret_cast<T*>(sm);
    T* inv = D + m * m;
    T* pv = inv + m;
    int* kp = reinterpret_cast<int*>(pv + m);
    int* kr = kp + m + 1;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb) return;
    for (int i = tid; i < m * m; i += bd) D[i] = T(0);
    for (int i = tid; i <= m; i += bd) kp[i] = rk_ptr[i];
    for (int i = tid; i < n_rk; i += bd) kr[i] = rk[i];
    __syncthreads();
    T* tb = tm + (size_t)b * n_top;
    for (int q = tid; q < n_top; q += bd) D[top_pos[q]] = tb[q];
    const T e = tol[b];
    __syncthreads();
    for (int k = 0; k < m; ++k) {
        if (tid == 0) {
            T p = D[k * m + k];
            if (fabs(p) < e) {
                atomicAdd(clamped + b, 1);
                p = p < T(0) ? -e : e;
            }
            pv[k] = p;
            inv[k] = T(1) / p;
        }
        __syncthreads();
        const T ik = inv[k];
        if (k > 0) {
            const T ip = inv[k - 1];
            for (int a = kp[k - 1] + tid; a < kp[k]; a += bd) D[kr[a] * m + k - 1] *= ip;
        }
        const int r0 = kp[k], r = kp[k + 1] - r0;
        for (int t = tid; t < r * r; t += bd) {
            const int ai = t / r, aj = t - ai * r;
            const int i = kr[r0 + ai], j = kr[r0 + aj];
            D[i * m + j] = fma(-D[i * m + k] * ik, D[k * m + j], D[i * m + j]);
        }
        __syncthreads();
    }
    for (int a = kp[m - 1] + tid; a < kp[m]; a += bd) D[kr[a] * m + m - 1] *= inv[m - 1];
    for (int k = tid; k < m; k += bd) D[k * m + k] = pv[k];
    __syncthreads();
    for (int q = tid; q < n_top; q += bd) tb[q] = D[top_pos[q]];
}

template <class T>
__device__ __forceinline__ T top_warp_sum(T s)
{
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

// One block per matrix: T's forward (unit L) on T x T (top_fo took the columns outside T), column-oriented:
// x_T[i] -= L(i, k) x_T[k] for the pattern rows of column k, a barrier per column; L from the matrix-major tm
template <class T>
__global__ void __launch_bounds__(kTopSolveThreads)
top_fwd(const int m, const int n_top, const int* __restrict__ trows, const int* __restrict__ rk_ptr, const int* __restrict__ rk,
        const int* __restrict__ fl_q, const int n_fl, const int B, const int nb, const T* __restrict__ tm, T* __restrict__ xi)
{
    extern __shared__ __align__(16) unsigned char sm[];
    T* xT = reinterpret_cast<T*>(sm);
    T* Ls = xT + m;
    int* kp = reinterpret_cast<int*>(Ls + n_fl);
    int* kr = kp + m + 1;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb) return;
    const T* tb = tm + (size_t)b * n_top;
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
template <class T>
__global__ void __launch_bounds__(kTopSolveThreads)
top_bwd(const int m, const int n_top, const int* __restrict__ trows, const int* __restrict__ bu_ptr, const int* __restrict__ bu_row,
        const int* __restrict__ bu_q, const int n_bu, const int* __restrict__ dg_q, const int B, const int nb,
        const T* __restrict__ tm, T* __restrict__ xi)
{
    extern __shared__ __align__(16) unsigned char sm[];
    T* xT = reinterpret_cast<T*>(sm);
    T* xo = xT + m;
    T* dinv = xo + m;
    T* Us = dinv + m;
    int* kp = reinterpret_cast<int*>(Us + n_bu);
    int* kr = kp + m + 1;
    const int b = blockIdx.x, tid = threadIdx.x, bd = blockDim.x;
    if (b >= nb) return;
    const T* tb = tm + (size_t)b * n_top;
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
                              unsigned long long* __restrict__ amax)
{
    __shared__ T t[32][33];
    __shared__ double mx[8][32];
    const int b0 = blockIdx.y * 32;
    double m = 0.;
    for (int j0 = blockIdx.x * 32; j0 < ncols; j0 += gridDim.x * 32) {
        for (int r = threadIdx.y; r < 32; r += 8) {
            const int b = b0 + r, j = j0 + threadIdx.x;
            if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)b * ncols + j];
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
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) tol[b] = static_cast<T>(eps * __longlong_as_double(static_cast<long long>(amax[b])));
}

// b -> two interleaved copies (the right-hand side kept for the refinement, the one the solve overwrites)
template <class T>
__global__ void lu_tr_in2(const T* __restrict__ src, const int* __restrict__ map, const int ncols, const int B, const int nb,
                          T* __restrict__ dst, T* __restrict__ dst2)
{
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) t[r][threadIdx.x] = src[(size_t)b * ncols + j];
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
                              const int ncols, const int B, const int nb, T* __restrict__ dst)
{
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < nb && j < ncols) {
            const size_t o = (size_t)map[j] * B + b;
            t[r][threadIdx.x] = src[o] + src2[o];
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < nb && j < ncols) dst[(size_t)b * ncols + j] = t[threadIdx.x][r];
    }
}

// r = b - A x, everything interleaved (permuted rows, matrix fastest): thread = (row, matrix)
template <class T>
__global__ void lu_residual_i(const int* __restrict__ ra_ptr, const int* __restrict__ ra_p, const int* __restrict__ ra_col,
                              const int n, const int B, const int nb, const T* __restrict__ ai, const T* __restrict__ bi,
                              const T* __restrict__ xi, T* __restrict__ ri)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * nb) return;
    const int i = (int)(idx / nb), b = (int)(idx - (long long)i * nb);
    T s = bi[(size_t)i * B + b];
    for (int k = ra_ptr[i]; k < ra_ptr[i + 1]; ++k) s = fma(-ai[(size_t)ra_p[k] * B + b], xi[(size_t)ra_col[k] * B + b], s);
    ri[(size_t)i * B + b] = s;
}

template <class T>
__global__ void lu_axpy_i(const int n, const int B, const int nb, const T* __restrict__ d, T* __restrict__ x)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)n * nb) return;
    const int i = (int)(idx / nb), b = (int)(idx - (long long)i * nb);
    x[(size_t)i * B + b] += d[(size_t)i * B + b];
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



void TopLu::init(const own_lu::LuPlan& plan, int batch, cudaStream_t stream, double eps, int max_top,
                 int split_chain, int split_items)
{
    using T = cuda_real_type;
    if (batch <= 0) throw std::invalid_argument("[top_lu::TopLu::init] batch must be > 0");
    n_ = plan.n;
    nnz_ = plan.nnz;
    n_ent_ = plan.n_ent;
    B_ = nb_ = batch;
    eps_ = eps;
    stream_ = stream;
    split_chain_ = split_chain;
    split_items_ = split_items;
    const int n = n_;
    const std::vector<int>&rp = plan.rp, &dp = plan.dp, &ci = plan.ci;

    auto up = [](thrust::device_vector<int>& d, const std::vector<int>& h) {
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
            if (e - l >= 2) { s.steps.push_back(SolveStep{-1, l, e}); l = e; }
            else { s.steps.push_back(SolveStep{l, 0, 0}); ++l; }
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
    // fill entry: start 0), then, at the end of the U step of its row's level, a copy item for every U entry of A
    // without one (no contribution: own_lu leaves the transposed A value there; the factors start from the
    // interleaved copy of A here, so the entry must be written)
    std::vector<int> erow(n_ent_);
    for (int i = 0; i < n; ++i)
        for (int e = rp[i]; e < rp[i + 1]; ++e) erow[e] = i;
    struct Step { int offset, count, mc; bool is_u; };
    std::vector<int> it_tgt, it_src, it_cptr{0}, it_cl, it_cu, it_dg;
    std::vector<Step> steps;
    {
        auto target = [](int raw) { return raw < 0 ? ~raw : raw; };
        std::vector<int> a_of(n_ent_, -1);
        for (int p = 0; p < nnz_; ++p) a_of[plan.a_ent[p]] = p;
        std::vector<char> has_item(n_ent_, 0);
        for (int raw : plan.f_tgt) has_item[target(raw)] = 1;
        std::vector<std::vector<int>> extra(H);
        for (int r = 0; r < n; ++r)
            for (int e = dp[r] + 1; e < rp[r + 1]; ++e)
                if (!has_item[e] && a_of[e] >= 0) extra[level[r]].push_back(e);
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
                for (int e : extra[level[std::min(erow[t0], ci[t0])]]) {
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
        std::vector<std::vector<std::pair<int, int>>> rows(n);
        for (int p = 0; p < nnz_; ++p) rows[erow[plan.a_ent[p]]].push_back({p, ci[plan.a_ent[p]]});
        std::vector<int> ra_ptr{0}, ra_p, ra_col;
        for (int i = 0; i < n; ++i) {
            for (const auto& [p, c] : rows[i]) { ra_p.push_back(p); ra_col.push_back(c); }
            ra_ptr.push_back(static_cast<int>(ra_p.size()));
        }
        up(d_ra_ptr_, ra_ptr);
        up(d_ra_p_, ra_p);
        up(d_ra_col_, ra_col);
    }

    // T: the pivots of level >= L0, L0 the lowest level whose dense block (+ lists) fits one block
    int optin = 0, dev = 0;
    OWN_LU_CHECK(cudaGetDevice(&dev));
    OWN_LU_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_M")) max_top = std::atoi(e);
    std::vector<int> lcount(n, 0);  // L entries per column
    for (int i = 0; i < n; ++i)
        for (int e = rp[i]; e < dp[i]; ++e) ++lcount[ci[e]];
    std::vector<long long> per_lev(H, 0), lc_lev(H, 0);
    for (int i = 0; i < n; ++i) {
        ++per_lev[level[i]];
        lc_lev[level[i]] += lcount[i];
    }
    m_ = 0;
    l0_ = H;
    {
        long long mm = 0, nl = 0;
        for (int L = H - 1; L >= 0; --L) {
            mm += per_lev[L];
            nl += lc_lev[L];
            const long long smem = (mm * mm + 2 * mm) * static_cast<long long>(sizeof(T)) + (mm + 1 + nl) * 4;
            if (smem > optin || (max_top > 0 && mm > max_top)) break;
            m_ = static_cast<int>(mm);
            l0_ = L;
        }
    }
    if (m_ < 2) { m_ = 0; l0_ = H; }
    std::vector<char> in_t(n, 0);
    std::vector<int> trows, tpos(n, -1);
    for (int i = 0; i < n; ++i)
        if (level[i] >= l0_) { in_t[i] = 1; tpos[i] = static_cast<int>(trows.size()); trows.push_back(i); }
    for (int i = 0; i < n; ++i) {  // closure: L(:, k in T) and U(k in T, :) stay in T
        for (int e = rp[i]; e < dp[i]; ++e)
            if (in_t[ci[e]] && !in_t[i]) throw std::logic_error("[top_lu] T not closed (L)");
        if (in_t[i])
            for (int e = dp[i] + 1; e < rp[i + 1]; ++e)
                if (!in_t[ci[e]]) throw std::logic_error("[top_lu] T not closed (U)");
    }
    // factor steps below L0; the T x T items' outside contributions
    f_bottom_.clear();
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
        if (!top) { f_bottom_.push_back(f_all_[s]); continue; }
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
    const int m = m_;
    std::vector<int> top_ent, top_pos, dg_ent(m), fo_ptr{0}, fo_ent, fo_col;
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
            top_pos.push_back(a * m + tpos[c]);
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
    n_top_ = static_cast<int>(top_ent.size());
    n_fl_ = static_cast<int>(rk.size());
    n_bu_ = static_cast<int>(bu_row.size());
    // positions in the matrix-major tm (q = index in top_ent) of the L / U lists and the diagonals; the T x T items
    // sorted by q (coalesced tile writes)
    {
        std::vector<int> q_of(n_ent_, -1);
        for (int q = 0; q < n_top_; ++q) q_of[top_ent[q]] = q;
        std::vector<int> fl_q, bu_q, dg_q(m);
        for (int e : fl_ent) fl_q.push_back(q_of[e]);
        for (int e : bu_ent) bu_q.push_back(q_of[e]);
        for (int a = 0; a < m; ++a) dg_q[a] = q_of[dg_ent[a]];
        up(d_fl_q_, fl_q);
        up(d_bu_q_, bu_q);
        up(d_dg_q_, dg_q);
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
        n_oa_ = static_cast<int>(o_q.size());
        up(d_oa_tgt_, o_q);
        up(d_oa_src_, o_src);
        up(d_oa_cptr_, o_cptr);
        up(d_oa_cl_, o_cl);
        up(d_oa_cu_, o_cu);
    }
    up(d_t_rows_, trows);
    up(d_top_ent_, top_ent);
    up(d_top_pos_, top_pos);
    up(d_rk_ptr_, rk_ptr);
    up(d_rk_, rk);
    up(d_fl_ent_, fl_ent);
    up(d_fo_ptr_, fo_ptr);
    up(d_fo_ent_, fo_ent);
    up(d_fo_col_, fo_col);
    up(d_bu_ptr_, bu_ptr);
    up(d_bu_row_, bu_row);
    up(d_bu_ent_, bu_ent);
    up(d_dg_ent_, dg_ent);

    // the rows outside T: forward by the plan's levels, backward by their levels among themselves
    {
        std::vector<char> rest(n);
        for (int i = 0; i < n; ++i) rest[i] = !in_t[i];
        std::vector<int> rows, blev(n, 0);
        std::vector<std::array<int, 3>> sl;
        group(level, rest, fw_len, rows, sl);
        make_sched(fw_rest_, rows, sl);
        for (int i = n - 1; i >= 0; --i) {
            if (in_t[i]) continue;
            int mx = -1;
            for (int e = dp[i] + 1; e < rp[i + 1]; ++e)
                if (!in_t[ci[e]]) mx = std::max(mx, blev[ci[e]]);
            blev[i] = mx + 1;
        }
        group(blev, rest, bw_len, rows, sl);
        make_sched(bw_rest_, rows, sl);
    }
    if (m > 0) {
        const int sf = static_cast<int>((static_cast<long long>(m) * m + 2 * m) * sizeof(T) + (m + 1 + n_fl_) * 4);
        const int sw = static_cast<int>((m + n_fl_) * sizeof(T) + (m + 1 + n_fl_) * 4);  // top_fwd
        const int sb = static_cast<int>((3 * m + n_bu_) * sizeof(T) + (m + 1 + n_bu_) * 4);
        OWN_LU_CHECK(cudaFuncSetAttribute(top_factor<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sf));
        OWN_LU_CHECK(cudaFuncSetAttribute(top_fwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sw));
        OWN_LU_CHECK(cudaFuncSetAttribute(top_bwd<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, sb));
    }
    if (const char* e = std::getenv("GPUSIM2GRID_TOP_LU_STATS"); e != nullptr && *e && *e != '0') {
        auto dump = [](const char* what, const std::vector<Launch>& ls) {
            std::fprintf(stderr, "[top_lu] %s: %zu launches\n", what, ls.size());
            for (size_t l = 0; l < ls.size(); ++l)
                std::fprintf(stderr, "  %3zu count %6d chain %4d k %2d%s\n", l, ls[l].count, ls[l].mc, ls[l].k,
                             static_cast<long long>(ls[l].count) * ls[l].k <= 32 ? " narrow" : "");
        };
        dump("factor below L0", f_bottom_);
        dump("forward outside T", fw_rest_.launches);
        dump("backward outside T", bw_rest_.launches);
    }
    d_v_.assign(static_cast<size_t>(n_ent_) * B_, T(0));
    d_ai_.assign(static_cast<size_t>(nnz_) * B_, T(0));
    d_tm_.assign(static_cast<size_t>(std::max(n_top_, 1)) * B_, T(0));
    d_bi_.assign(static_cast<size_t>(n_) * B_, T(0));
    d_ri_.assign(static_cast<size_t>(n_) * B_, T(0));
    d_amax_.assign(B_, 0ULL);
    d_xi_.assign(static_cast<size_t>(n_) * B_, T(0));
    d_tol_.assign(B_, T(0));
    d_clamped_.assign(B_, 0);
    d_r_.clear();
    d_d_.clear();
    OWN_LU_CHECK(cudaDeviceSynchronize());
}

int TopLu::launches_factor() const
{
    const bool dense = m_ > 0 && !levels_only_;
    return 3 + static_cast<int>(dense ? f_bottom_.size() + 2 : f_all_.size());
}

int TopLu::launches_solve() const
{
    const bool dense = m_ > 0 && !levels_only_;
    return 2 + static_cast<int>(dense ? fw_rest_.steps.size() + bw_rest_.steps.size() + 3
                                      : fw_all_.steps.size() + bw_all_.steps.size());
}

void TopLu::set_active(int nb)
{
    if (nb < 1 || nb > B_) throw std::invalid_argument("[top_lu::TopLu::set_active] need 1 <= nb <= batch");
    nb_ = nb;
}

void TopLu::factorize(const cuda_real_type* values)
{
    using T = cuda_real_type;
    const int B = B_, nb = nb_;
    T* v = rawm(d_v_);
    T* ai = rawm(d_ai_);
    T* tol = rawm(d_tol_);
    int* clamped = rawm(d_clamped_);
    OWN_LU_CHECK(cudaMemsetAsync(clamped, 0, sizeof(int) * nb, stream_));
    OWN_LU_CHECK(cudaMemsetAsync(rawm(d_amax_), 0, sizeof(unsigned long long) * nb, stream_));
    {   // J -> interleaved A + max |A_b|, then the clamps
        const int tiles = (nnz_ + 31) / 32;
        dim3 grid(std::min(tiles, 64), (nb + 31) / 32), block(32, 8);
        lu_tr_in_vals<T><<<grid, block, 0, stream_>>>(values, nnz_, B, nb, ai, rawm(d_amax_));
        lu_tol_from_max<T><<<(nb + 255) / 256, 256, 0, stream_>>>(raw(d_amax_), eps_, nb, tol);
    }
    const int* tgt = raw(d_tgt_); const int* src = raw(d_src_); const int* cptr = raw(d_cptr_);
    const int* dg = raw(d_dg_); const int* cl = raw(d_cl_); const int* cu = raw(d_cu_);
    const bool dense = m_ > 0 && !levels_only_;
    for (const Launch& L : dense ? f_bottom_ : f_all_) {
        const int k = split_for(L.k, L.count, L.mc, nb);
        if (k > 1) {
            dim3 grid(L.count, (nb + 31) / 32);
            lu_left_split<T><<<grid, 32 * k, 0, stream_>>>(tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu,
                                                             dg + L.offset, B, nb, L.is_u ? 1 : 0, tol, clamped, v);
        } else if (L.is_u) {
            lu_left_u<T><<<blocks_for((long long)L.count * nb), kThreads, 0, stream_>>>(
                tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, nb, tol, clamped, v);
        } else {
            lu_left_l<T><<<blocks_for((long long)L.count * nb), kThreads, 0, stream_>>>(
                tgt + L.offset, src + L.offset, ai, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, nb, v);
        }
    }
    if (dense) {
        top_outside<T><<<dim3((n_oa_ + 31) / 32, (nb + 31) / 32), dim3(32, 8), 0, stream_>>>(
            raw(d_oa_tgt_), raw(d_oa_src_), ai, raw(d_oa_cptr_), raw(d_oa_cl_), raw(d_oa_cu_), n_oa_, n_top_, B, nb, v, rawm(d_tm_));
        const int m = m_;
        const size_t sf = (static_cast<size_t>(m) * m + 2 * m) * sizeof(T) + (m + 1 + n_fl_) * 4;
        top_factor<T><<<nb, kTopFactorThreads, sf, stream_>>>(m, n_top_, raw(d_top_pos_), raw(d_rk_ptr_),
                                                               raw(d_rk_), n_fl_, B, nb, tol, clamped, rawm(d_tm_));
    }
    ++fact_gen_;
    last_values_ = values;
    OWN_LU_CHECK(cudaGetLastError());
}

void TopLu::run_solve(const Sched& s, bool forward, cuda_real_type* xi)
{
    using T = cuda_real_type;
    const int B = B_, nb = nb_;
    const T* v = raw(d_v_);
    const int* rp = raw(d_rp_); const int* dp = raw(d_dp_); const int* ci = raw(d_ci_);
    const int n_lv = static_cast<int>(s.launches.size());
    for (const SolveStep& st : s.steps) {
        if (st.launch < 0) {
            if (forward)
                lu_solve_run<T, true><<<(nb + 31) / 32, kRunThreads, 0, stream_>>>(raw(s.d_rows), raw(s.d_run), st.first,
                                                                                 st.last, n_lv, rp, dp, ci, B, nb, v, xi);
            else
                lu_solve_run<T, false><<<(nb + 31) / 32, kRunThreads, 0, stream_>>>(raw(s.d_rows), raw(s.d_run), st.first,
                                                                                  st.last, n_lv, rp, dp, ci, B, nb, v, xi);
            continue;
        }
        const Launch& L = s.launches[static_cast<size_t>(st.launch)];
        const int* rows = raw(s.d_rows) + L.offset;
        const int k = split_for(L.k, L.count, L.mc, nb);
        if (forward) {
            if (k > 1) lu_fwd_split<T><<<dim3(L.count, (nb + 31) / 32), 32 * k, 0, stream_>>>(rows, rp, dp, ci, B, nb, v, xi);
            else lu_fwd<T><<<blocks_for((long long)L.count * nb), kThreads, 0, stream_>>>(rows, rp, dp, ci, L.count, B, nb, v, xi);
        } else {
            if (k > 1) lu_bwd_split<T><<<dim3(L.count, (nb + 31) / 32), 32 * k, 0, stream_>>>(rows, rp, dp, ci, B, nb, v, xi);
            else lu_bwd<T><<<blocks_for((long long)L.count * nb), kThreads, 0, stream_>>>(rows, rp, dp, ci, L.count, B, nb, v, xi);
        }
    }
}

void TopLu::solve_core(cuda_real_type* xi)
{
    using T = cuda_real_type;
    const int B = B_, nb = nb_;
    if (m_ > 0 && !levels_only_) {
        const int m = m_;
        run_solve(fw_rest_, true, xi);
        const size_t sw = (m + n_fl_) * sizeof(T) + (m + 1 + n_fl_) * 4;
        top_fo<T><<<dim3(m, (nb + 31) / 32), 32 * 8, 0, stream_>>>(raw(d_t_rows_), raw(d_fo_ptr_), raw(d_fo_ent_), raw(d_fo_col_),
                                                                 B, nb, raw(d_v_), xi);
        top_fwd<T><<<nb, kTopSolveThreads, sw, stream_>>>(m, n_top_, raw(d_t_rows_), raw(d_rk_ptr_), raw(d_rk_), raw(d_fl_q_), n_fl_,
                                                           B, nb, raw(d_tm_), xi);
        const size_t sb = (3 * m + n_bu_) * sizeof(T) + (m + 1 + n_bu_) * 4;
        top_bwd<T><<<nb, kTopSolveThreads, sb, stream_>>>(m, n_top_, raw(d_t_rows_), raw(d_bu_ptr_), raw(d_bu_row_), raw(d_bu_q_),
                                                           n_bu_, raw(d_dg_q_), B, nb, raw(d_tm_), xi);
        run_solve(bw_rest_, false, xi);
    } else {
        run_solve(fw_all_, true, xi);
        run_solve(bw_all_, false, xi);
    }
}

void TopLu::solve(const cuda_real_type* b, cuda_real_type* x)
{
    using T = cuda_real_type;
    const int B = B_, nb = nb_;
    T* xi = rawm(d_xi_);
    dim3 tgrid((n_ + 31) / 32, (nb + 31) / 32), tblock(32, 8);
    lu_tr_in2<T><<<tgrid, tblock, 0, stream_>>>(b, raw(d_inv_), n_, B, nb, rawm(d_bi_), xi);
    solve_core(xi);
    lu_tr_out<T><<<tgrid, tblock, 0, stream_>>>(xi, raw(d_inv_), n_, B, nb, x);
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
    if (steps <= 0) return;
    if (static_cast<const void*>(x) == static_cast<const void*>(b))
        throw std::invalid_argument("top_lu::TopLu::refine: x aliases b");
    const int B = B_, nb = nb_;
    const long long mm = static_cast<long long>(n_) * nb_;
    if (interleaved_ok_ && b == last_b_ && x == last_x_ && values == last_values_) {
        // in the interleaved layout: r = b - A x on the copy of A the factorization transposed, the correction
        // solved in place, x + d written back by the last transpose
        T* xi = rawm(d_xi_);
        T* ri = rawm(d_ri_);
        dim3 tgrid((n_ + 31) / 32, (nb + 31) / 32), tblock(32, 8);
        for (int s = 0; s < steps; ++s) {
            lu_residual_i<T><<<blocks_for(mm), kThreads, 0, stream_>>>(raw(d_ra_ptr_), raw(d_ra_p_), raw(d_ra_col_), n_, B, nb,
                                                                       raw(d_ai_), raw(d_bi_), xi, ri);
            solve_core(ri);
            if (s + 1 < steps) lu_axpy_i<T><<<blocks_for(mm), kThreads, 0, stream_>>>(n_, B, nb, ri, xi);
            else lu_tr_out_add<T><<<tgrid, tblock, 0, stream_>>>(xi, ri, raw(d_inv_), n_, B, nb, x);
        }
        interleaved_ok_ = false;
        OWN_LU_CHECK(cudaGetLastError());
        return;
    }
    if (d_r_.size() != static_cast<size_t>(n_) * B_) {
        d_r_.resize(static_cast<size_t>(n_) * B_);
        d_d_.resize(static_cast<size_t>(n_) * B_);
    }
    for (int s = 0; s < steps; ++s) {
        lu_residual<T><<<blocks_for(mm), kThreads, 0, stream_>>>(values, indptr, indices, n_, nnz_, nb_, b, x, rawm(d_r_));
        solve(rawm(d_r_), rawm(d_d_));
        lu_axpy<T><<<blocks_for(mm), kThreads, 0, stream_>>>(mm, raw(d_d_), x);
    }
    interleaved_ok_ = false;
    OWN_LU_CHECK(cudaGetLastError());
}

std::vector<int> TopLu::clamped_counts() const
{
    std::vector<int> h(nb_, 0);
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    OWN_LU_CHECK(cudaMemcpy(h.data(), raw(d_clamped_), sizeof(int) * nb_, cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace top_lu
