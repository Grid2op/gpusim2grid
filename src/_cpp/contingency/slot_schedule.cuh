// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SLOT_SCHEDULE_CUH
#define SLOT_SCHEDULE_CUH

// =============================================================================
// contingency/slot_schedule.cuh
//
// What a BatchSource needs to fill an arbitrary set of batch slots with
// arbitrary rows (the load path of both schedulers, see
// docs/dev_notes/continuous_batching.md):
//
//   SlotLoadView  : the slots being loaded now and the ACTIVE row each one
//                   receives (-1 = the base case: a phantom slot). The chunked
//                   schedule loads every slot of chunk c (rows c*S.., phantom
//                   tail); the continuous one the slots its last round freed.
//   SlotTableView : the active row every slot holds after the load (what the
//                   per-iteration streams -- the masks -- are built from).
//   kernels       : copy rows of a per-row array into a slot list (row -1 =
//                   a base row, or left alone), fill a slot list with a value,
//                   and the per-slot slack_absorbed initialisation.
//   PinnedBuffer  : page-locked host staging, so the per-round H->D / D->H of
//                   the scheduler never goes through a pageable copy.
//
// The kernels are templates on purpose: this header is included by several
// translation units and a non-template __global__ defined here would be
// emitted in each of them.
// =============================================================================

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../cu_complex_utils.h"
#include "../nr_iter_step.cuh"   // BS, nr_grid_size

// -----------------------------------------------------------------------------
// Views
// -----------------------------------------------------------------------------
struct SlotLoadView {
    int        n      = 0;         // slots loaded
    const int* d_slot = nullptr;   // [n] slot ids (device)
    const int* d_row  = nullptr;   // [n] active row, -1 = base case (device)
    const int* h_slot = nullptr;   // the same two lists on the host
    const int* h_row  = nullptr;
    int        chunk  = -1;        // chunked schedule: the chunk being loaded
                                   // (slot i <- row chunk*S + i); -1 otherwise
    int        n_real = 0;         // rows >= 0 among the n
};

struct SlotTableView {
    int        S          = 0;
    const int* d_slot_row = nullptr;   // [S] active row of every slot, -1 = phantom
    const int* h_slot_row = nullptr;
    int        chunk      = -1;        // as SlotLoadView::chunk
};

// The slot moves of a continuous run's tail compaction (BatchPfDriver::
// _compact_tail): slot from[i]'s per-slot state is copied to slot to[i]. The
// from and to sets are disjoint (live slots past n_live into phantom slots
// before it), so every copy can run at once, in place.
struct SlotMoveView {
    int        n      = 0;
    const int* d_from = nullptr;   // [n] (device)
    const int* d_to   = nullptr;
    const int* h_from = nullptr;   // the same two lists on the host
    const int* h_to   = nullptr;
};

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

// dst[slot[i] * width + j] = src[row[i] * width + j] when src and row[i] >= 0,
// else base[j]; a slot whose value would come from a null base is left alone.
// One thread per (i, j).
template <typename T>
__global__ void gather_rows_to_slots_kernel(T* __restrict__ dst,
                                            const T* __restrict__ src,
                                            const T* __restrict__ base,
                                            const int* __restrict__ d_slot,
                                            const int* __restrict__ d_row,
                                            int n, int width)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t i   = tid / width;
    const int       j   = static_cast<int>(tid % width);
    if (i >= n) return;
    const int row = d_row[i];
    const ptrdiff_t at = static_cast<ptrdiff_t>(d_slot[i]) * width + j;
    if (src != nullptr && row >= 0)
        dst[at] = src[static_cast<ptrdiff_t>(row) * width + j];
    else if (base != nullptr)
        dst[at] = base[j];
}

// buf[to[i] * width + j] = buf[from[i] * width + j]; the two slot sets are
// disjoint (SlotMoveView).
template <typename T>
__global__ void move_slot_rows_kernel(T* __restrict__ buf,
                                      const int* __restrict__ d_from,
                                      const int* __restrict__ d_to,
                                      int n, int width)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t i   = tid / width;
    const int       j   = static_cast<int>(tid % width);
    if (i >= n) return;
    buf[static_cast<ptrdiff_t>(d_to[i]) * width + j] = buf[static_cast<ptrdiff_t>(d_from[i]) * width + j];
}

// dst[row[s] * width + j] = src[s * width + j] for every slot s < n whose row[s]
// is >= 0 (an eviction map: the slot's leaving row), the others left alone.
template <typename T>
__global__ void scatter_slots_to_rows_kernel(T* __restrict__ dst,
                                             const T* __restrict__ src,
                                             const int* __restrict__ d_row,
                                             int n, int width)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t s   = tid / width;
    const int       j   = static_cast<int>(tid % width);
    if (s >= n) return;
    const int row = d_row[s];
    if (row < 0) return;
    dst[static_cast<ptrdiff_t>(row) * width + j] = src[s * width + j];
}

// dst[slot[i] * width + j] = value
template <typename T>
__global__ void fill_slots_kernel(T* __restrict__ dst, T value,
                                  const int* __restrict__ d_slot, int n, int width)
{
    const ptrdiff_t tid = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const ptrdiff_t i   = tid / width;
    const int       j   = static_cast<int>(tid % width);
    if (i >= n) return;
    dst[static_cast<ptrdiff_t>(d_slot[i]) * width + j] = value;
}

// dst[i] = value, i < n
template <typename T>
__global__ void fill_value_kernel(T* __restrict__ dst, T value, ptrdiff_t n)
{
    const ptrdiff_t i = static_cast<ptrdiff_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = value;
}

// slack_absorbed[slot] = Re(Σ Sbus_slot) for the listed slots -- the same sum,
// in the same order, as init_slack_absorbed_kernel (acpf_nr_kernels.cu).
template <typename C, typename R>
__global__ void init_slack_absorbed_slots_kernel(R* __restrict__ d_slack_absorbed,
                                                 const C* __restrict__ d_Sbus,
                                                 int sbus_stride, int n_bus,
                                                 const int* __restrict__ d_slot, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int b = d_slot[i];
    const C* sb = d_Sbus + static_cast<size_t>(b) * sbus_stride;
    R s = static_cast<R>(0.);
    for (int k = 0; k < n_bus; ++k) s += CudaFunHelper::my_cuCreal(sb[k]);
    d_slack_absorbed[b] = s;
}

// -----------------------------------------------------------------------------
// The check before a solve's first step (continuous schedule, outer loops):
// lightsim2grid's NRAlgo::newton checks its residual before stepping, after
// recalibrating a distributed slack's slack_absorbed from it, so an inner solve
// whose start is already converged (a re-solve after the loops moved the
// units' targets, typically) takes no step.
// -----------------------------------------------------------------------------

// For each slot with d_check[slot] != 0: NRSystem::calibrate_slack_absorbed /
// MultiSlack::absorb_balance -- the sum of the P rows of F (the negated
// mismatch, masked rows already 0) over the total participation weight goes to
// slack_absorbed, which closes the active power balance at the current V
// (adjust_slack_mismatch_kernel takes sa·w out of F: the sum is affine in sa).
// A degenerate participation (total weight <= 1e-7, lightsim2grid's
// _tol_equal_float) leaves it as it is. One block per slot.
template <typename R>
__global__ void calibrate_slack_absorbed_slots_kernel(R* __restrict__ d_slack_absorbed,
                                                      const R* __restrict__ d_F,
                                                      const int* __restrict__ d_p_rows, int n_p,
                                                      const R* __restrict__ d_slack_w, int w_stride,
                                                      int n_slack, int dim_J,
                                                      const int* __restrict__ d_check, int n_slots)
{
    const int b = blockIdx.x;
    if (b >= n_slots || d_check[b] == 0) return;
    __shared__ R s_sum[BS];
    const R* F = d_F + static_cast<size_t>(b) * dim_J;
    R acc = static_cast<R>(0.);
    for (int k = threadIdx.x; k < n_p; k += blockDim.x) acc += F[d_p_rows[k]];
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int h = blockDim.x / 2; h > 0; h /= 2) {
        if (threadIdx.x < h) s_sum[threadIdx.x] += s_sum[threadIdx.x + h];
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    const R* w = d_slack_w + static_cast<size_t>(b) * w_stride;
    R total_w = static_cast<R>(0.);
    for (int k = 0; k < n_slack; ++k) total_w += w[k];
    if (!(fabs(static_cast<double>(total_w)) > 1e-7)) return;
    d_slack_absorbed[b] += s_sum[0] / total_w;
}

// d_skip[slot] = the slot is checked and its residual is already below tol.
template <typename R>
__global__ void skip_converged_slots_kernel(int* __restrict__ d_skip,
                                            const int* __restrict__ d_check,
                                            const R* __restrict__ d_res,
                                            double tol, int n_slots)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    d_skip[s] = (d_check[s] != 0 && static_cast<double>(d_res[s]) < tol) ? 1 : 0;
}

// -----------------------------------------------------------------------------
// Launchers (no-ops on empty input)
// -----------------------------------------------------------------------------
template <typename T>
inline void launch_gather_rows_to_slots(T* dst, const T* src, const T* base,
                                        const SlotLoadView& L, int width, cudaStream_t cs)
{
    if (L.n <= 0 || width <= 0) return;
    gather_rows_to_slots_kernel<T><<<nr_grid_size((long long)L.n * width, BS), BS, 0, cs>>>(
        dst, src, base, L.d_slot, L.d_row, L.n, width);
}

template <typename T>
inline void launch_scatter_slots_to_rows(T* dst, const T* src, const int* d_row, int n, int width,
                                         cudaStream_t cs)
{
    if (dst == nullptr || n <= 0 || width <= 0) return;
    scatter_slots_to_rows_kernel<T><<<nr_grid_size((long long)n * width, BS), BS, 0, cs>>>(
        dst, src, d_row, n, width);
}

template <typename T>
inline void launch_fill_slots(T* dst, T value, const SlotLoadView& L, int width, cudaStream_t cs)
{
    if (L.n <= 0 || width <= 0) return;
    fill_slots_kernel<T><<<nr_grid_size((long long)L.n * width, BS), BS, 0, cs>>>(
        dst, value, L.d_slot, L.n, width);
}

// Copy the rows of a [S × width] per-slot buffer along the moves (no-op on a
// null / empty buffer).
template <typename T>
inline void launch_move_slot_rows(T* buf, int width, const SlotMoveView& M, cudaStream_t cs)
{
    if (buf == nullptr || M.n <= 0 || width <= 0) return;
    move_slot_rows_kernel<T><<<nr_grid_size((long long)M.n * width, BS), BS, 0, cs>>>(
        buf, M.d_from, M.d_to, M.n, width);
}

template <typename T>
inline void launch_fill_value(T* dst, T value, ptrdiff_t n, cudaStream_t cs)
{
    if (n <= 0) return;
    fill_value_kernel<T><<<nr_grid_size((long long)n, BS), BS, 0, cs>>>(dst, value, n);
}

// -----------------------------------------------------------------------------
// PinnedBuffer — page-locked host array (cudaMallocHost), RAII, non-copyable.
// -----------------------------------------------------------------------------
template <typename T>
struct PinnedBuffer {
    T*          ptr  = nullptr;
    std::size_t size = 0;

    PinnedBuffer() = default;
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr(o.ptr), size(o.size) { o.ptr = nullptr; o.size = 0; }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept
    {
        if (this != &o) {
            release();
            ptr = o.ptr; size = o.size;
            o.ptr = nullptr; o.size = 0;
        }
        return *this;
    }

    void resize(std::size_t n)
    {
        if (n == size) return;
        release();
        if (n == 0) return;
        void* p = nullptr;
        const cudaError_t err = cudaMallocHost(&p, n * sizeof(T));
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("PinnedBuffer: cudaMallocHost: ")
                                     + cudaGetErrorString(err));
        ptr  = static_cast<T*>(p);
        size = n;
    }
    void release()
    {
        if (ptr) cudaFreeHost(ptr);
        ptr  = nullptr;
        size = 0;
    }
    T&       operator[](std::size_t i)       { return ptr[i]; }
    const T& operator[](std::size_t i) const { return ptr[i]; }
};

#endif  // SLOT_SCHEDULE_CUH
