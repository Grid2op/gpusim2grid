// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// own_lu/batched_lu.cu -- see batched_lu.cuh. The kernels are
// gpu-security-analysis' engine/kernels/lu.cu, templated on the real type; the
// U steps also count the pivots they clamp, and refine() is new.

#include "batched_lu.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace own_lu {

namespace {

constexpr int kThreads = 256;

#define OWN_LU_CHECK(call)                                                                    \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess)                                                                \
            throw std::runtime_error(std::string("[own_lu] ") + #call + ": " + cudaGetErrorString(e_)); \
    } while (0)

// thread = (item, matrix), matrix fastest
#define LU_ITEM_AND_MATRIX                                                  \
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= (long long)n_items * B) return;                             \
    const int it = (int)(idx / B);                                         \
    const int b = (int)(idx - (long long)it * B);

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
__global__ void lu_left_u(const int* __restrict__ tgt, const int* __restrict__ cptr,
                          const int* __restrict__ cl, const int* __restrict__ cu,
                          const int* __restrict__ dg, const int n_items, const int B,
                          const T* __restrict__ tol, int* __restrict__ clamped, T* __restrict__ v)
{
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = LU_START(tgt[it], v, t, B, b);
    for (int c = cptr[it]; c < cptr[it + 1]; ++c)
        s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
    if (t == dg[it]) s = clamp_pivot(s, tol[b], clamped, b);
    v[(size_t)t * B + b] = s;
}

// L step of a level: column p's entries, divided by the (clamped) pivot.
template <class T>
__global__ void lu_left_l(const int* __restrict__ tgt, const int* __restrict__ cptr,
                          const int* __restrict__ cl, const int* __restrict__ cu,
                          const int* __restrict__ dg, const int n_items, const int B,
                          T* __restrict__ v)
{
    LU_ITEM_AND_MATRIX
    const int t = LU_TARGET(tgt[it]);
    T s = LU_START(tgt[it], v, t, B, b);
    for (int c = cptr[it]; c < cptr[it + 1]; ++c)
        s = fma(-v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], s);
    v[(size_t)t * B + b] = s / v[(size_t)dg[it] * B + b];
}

// Forward (unit L): x[i] -= sum L[i,k] x[k]; backward: x[i] = (x[i] - sum U[i,j] x[j]) / U[i,i].
template <class T>
__global__ void lu_fwd(const int* __restrict__ rows, const int* __restrict__ rp,
                       const int* __restrict__ dp, const int* __restrict__ ci, const int n_items,
                       const int B, const T* __restrict__ v, T* __restrict__ x)
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
                       const int B, const T* __restrict__ v, T* __restrict__ x)
{
    LU_ITEM_AND_MATRIX
    const int i = rows[it];
    T s = x[(size_t)i * B + b];
    for (int e = dp[i] + 1; e < rp[i + 1]; ++e) s = fma(-v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], s);
    x[(size_t)i * B + b] = s / v[(size_t)dp[i] * B + b];
}

// Layout changes, 32 x 32 tiles (block 32 x 8).
// in:  dst[map[j] * B + b] = src[b * ncols + j]; out: dst[b * ncols + j] = src[map[j] * B + b]
template <class T>
__global__ void lu_tr_in(const T* __restrict__ src, const int* __restrict__ map, const int ncols,
                         const int B, T* __restrict__ dst)
{
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < B && j < ncols) t[r][threadIdx.x] = src[(size_t)b * ncols + j];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < B && j < ncols) dst[(size_t)map[j] * B + b] = t[threadIdx.x][r];
    }
}

template <class T>
__global__ void lu_tr_out(const T* __restrict__ src, const int* __restrict__ map, const int ncols,
                          const int B, T* __restrict__ dst)
{
    __shared__ T t[32][33];
    const int j0 = blockIdx.x * 32, b0 = blockIdx.y * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int j = j0 + r, b = b0 + threadIdx.x;
        if (b < B && j < ncols) t[r][threadIdx.x] = src[(size_t)map[j] * B + b];
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int b = b0 + r, j = j0 + threadIdx.x;
        if (b < B && j < ncols) dst[(size_t)b * ncols + j] = t[threadIdx.x][r];
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
    const int b = blockIdx.y * 32 + lane;                       \
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
__global__ void lu_left_split(const int* __restrict__ tgt, const int* __restrict__ cptr,
                              const int* __restrict__ cl, const int* __restrict__ cu,
                              const int* __restrict__ dg, const int B, const int is_u,
                              const T* __restrict__ tol, int* __restrict__ clamped, T* __restrict__ v)
{
    LU_SPLIT_THREAD
    T acc = T(0);
    if (b < B)
        for (int c = cptr[it] + part; c < cptr[it + 1]; c += K)
            acc = fma(v[(size_t)cl[c] * B + b], v[(size_t)cu[c] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= B) return;
    const int t = LU_TARGET(tgt[it]), d = dg[it];
    T s = LU_START(tgt[it], v, t, B, b) - sum;
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
                             const T* __restrict__ v, T* __restrict__ x)
{
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < B)
        for (int e = rp[i] + part; e < dp[i]; e += K) acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= B) return;
    x[(size_t)i * B + b] -= sum;
}

template <class T>
__global__ void lu_bwd_split(const int* __restrict__ rows, const int* __restrict__ rp,
                             const int* __restrict__ dp, const int* __restrict__ ci, const int B,
                             const T* __restrict__ v, T* __restrict__ x)
{
    LU_SPLIT_THREAD
    const int i = rows[it];
    T acc = T(0);
    if (b < B)
        for (int e = dp[i] + 1 + part; e < rp[i + 1]; e += K)
            acc = fma(v[(size_t)e * B + b], x[(size_t)ci[e] * B + b], acc);
    const T sum = lu_split_sum(acc, lane, part, K, red);
    if (part != 0 || b >= B) return;
    x[(size_t)i * B + b] = (x[(size_t)i * B + b] - sum) / v[(size_t)dp[i] * B + b];
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

}  // namespace

void BatchedLu::init(const LuPlan& plan, int batch, cudaStream_t stream, double eps,
                     int split_chain, int split_items)
{
    if (batch <= 0) throw std::invalid_argument("[own_lu::BatchedLu::init] batch must be > 0");
    n_ = plan.n;
    nnz_ = plan.nnz;
    n_ent_ = plan.n_ent;
    B_ = batch;
    levels_ = plan.levels;
    contributions_ = plan.contributions;
    eps_ = eps;
    stream_ = stream;

    auto up = [](thrust::device_vector<int>& d, const std::vector<int>& h) { d.assign(h.begin(), h.end()); };
    up(d_a_ent_, plan.a_ent);
    up(d_inv_, plan.inv_order);
    up(d_rp_, plan.rp);
    up(d_dp_, plan.dp);
    up(d_ci_, plan.ci);
    up(d_tgt_, plan.f_tgt);
    up(d_cptr_, plan.f_cptr);
    up(d_dg_, plan.f_dg);
    up(d_cl_, plan.cl.empty() ? std::vector<int>{0} : plan.cl);
    up(d_cu_, plan.cu.empty() ? std::vector<int>{0} : plan.cu);
    up(d_fw_rows_, plan.fw_rows);
    up(d_bw_rows_, plan.bw_rows);

    auto k_of = [&](const LevelSlice& s) {
        if (split_chain <= 0 || s.count >= split_items) return 1;
        const int k = (s.max_chain + split_chain - 1) / split_chain;
        if (k <= 1) return 1;
        int p = 1;
        while (p < k) p <<= 1;
        return std::min(p, 32);
    };
    f_launch_.clear(); fw_launch_.clear(); bw_launch_.clear();
    for (const auto& s : plan.factor_steps) f_launch_.push_back({s.offset, s.count, k_of(s), s.is_u});
    for (const auto& s : plan.fw_levels) fw_launch_.push_back({s.offset, s.count, k_of(s), false});
    for (const auto& s : plan.bw_levels) bw_launch_.push_back({s.offset, s.count, k_of(s), false});

    // entries never computed (symmetrized fill without contributions) stay 0
    d_v_.assign(static_cast<size_t>(n_ent_) * B_, cuda_real_type(0));
    d_xi_.assign(static_cast<size_t>(n_) * B_, cuda_real_type(0));
    d_tol_.assign(B_, cuda_real_type(0));
    d_clamped_.assign(B_, 0);
    d_r_.clear();
    d_d_.clear();
    OWN_LU_CHECK(cudaDeviceSynchronize());
}

void BatchedLu::factorize(const cuda_real_type* values)
{
    using T = cuda_real_type;
    const int B = B_;
    T* v = rawm(d_v_);
    T* tol = rawm(d_tol_);
    int* clamped = rawm(d_clamped_);
    OWN_LU_CHECK(cudaMemsetAsync(clamped, 0, sizeof(int) * B, stream_));
    if (eps_ > 0.)
        lu_scaled_tol<T><<<B, 256, 0, stream_>>>(values, nnz_, static_cast<T>(eps_), tol);
    {
        dim3 grid((nnz_ + 31) / 32, (B + 31) / 32), block(32, 8);
        lu_tr_in<T><<<grid, block, 0, stream_>>>(values, raw(d_a_ent_), nnz_, B, v);
    }
    const int* tgt = raw(d_tgt_); const int* cptr = raw(d_cptr_); const int* dg = raw(d_dg_);
    const int* cl = raw(d_cl_);   const int* cu = raw(d_cu_);
    for (const Launch& L : f_launch_) {
        if (L.k > 1) {
            dim3 grid(L.count, (B + 31) / 32);
            lu_left_split<T><<<grid, 32 * L.k, 0, stream_>>>(tgt + L.offset, cptr + L.offset, cl, cu, dg + L.offset,
                                                             B, L.is_u ? 1 : 0, tol, clamped, v);
        } else if (L.is_u) {
            lu_left_u<T><<<blocks_for((long long)L.count * B), kThreads, 0, stream_>>>(
                tgt + L.offset, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, tol, clamped, v);
        } else {
            lu_left_l<T><<<blocks_for((long long)L.count * B), kThreads, 0, stream_>>>(
                tgt + L.offset, cptr + L.offset, cl, cu, dg + L.offset, L.count, B, v);
        }
    }
    OWN_LU_CHECK(cudaGetLastError());
}

void BatchedLu::solve(const cuda_real_type* b, cuda_real_type* x)
{
    using T = cuda_real_type;
    const int B = B_;
    const T* v = raw(d_v_);
    T* xi = rawm(d_xi_);
    dim3 tgrid((n_ + 31) / 32, (B + 31) / 32), tblock(32, 8);
    lu_tr_in<T><<<tgrid, tblock, 0, stream_>>>(b, raw(d_inv_), n_, B, xi);
    const int* rp = raw(d_rp_); const int* dp = raw(d_dp_); const int* ci = raw(d_ci_);
    for (const Launch& L : fw_launch_) {
        const int* rows = raw(d_fw_rows_) + L.offset;
        if (L.k > 1)
            lu_fwd_split<T><<<dim3(L.count, (B + 31) / 32), 32 * L.k, 0, stream_>>>(rows, rp, dp, ci, B, v, xi);
        else
            lu_fwd<T><<<blocks_for((long long)L.count * B), kThreads, 0, stream_>>>(rows, rp, dp, ci, L.count, B, v, xi);
    }
    for (const Launch& L : bw_launch_) {
        const int* rows = raw(d_bw_rows_) + L.offset;
        if (L.k > 1)
            lu_bwd_split<T><<<dim3(L.count, (B + 31) / 32), 32 * L.k, 0, stream_>>>(rows, rp, dp, ci, B, v, xi);
        else
            lu_bwd<T><<<blocks_for((long long)L.count * B), kThreads, 0, stream_>>>(rows, rp, dp, ci, L.count, B, v, xi);
    }
    lu_tr_out<T><<<tgrid, tblock, 0, stream_>>>(xi, raw(d_inv_), n_, B, x);
    OWN_LU_CHECK(cudaGetLastError());
}

void BatchedLu::refine(const cuda_real_type* values, const int* indptr, const int* indices,
                       const cuda_real_type* b, cuda_real_type* x, int steps)
{
    using T = cuda_real_type;
    if (steps <= 0) return;
    const long long m = static_cast<long long>(n_) * B_;
    if (d_r_.size() != static_cast<size_t>(m)) {
        d_r_.resize(m);
        d_d_.resize(m);
    }
    for (int s = 0; s < steps; ++s) {
        lu_residual<T><<<blocks_for(m), kThreads, 0, stream_>>>(values, indptr, indices, n_, nnz_, B_, b, x,
                                                                rawm(d_r_));
        solve(rawm(d_r_), rawm(d_d_));
        lu_axpy<T><<<blocks_for(m), kThreads, 0, stream_>>>(m, raw(d_d_), x);
    }
    OWN_LU_CHECK(cudaGetLastError());
}

std::vector<int> BatchedLu::clamped_counts() const
{
    std::vector<int> h(B_, 0);
    OWN_LU_CHECK(cudaStreamSynchronize(stream_));
    OWN_LU_CHECK(cudaMemcpy(h.data(), raw(d_clamped_), sizeof(int) * B_, cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace own_lu
