// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// launch_gather_groups_to_slots (contingency/v_groups.cuh): the starting V the
// strategy warm start writes into the loaded slots, in the build's precision.
// Needs a GPU (skipped, code 77, without one).

#include <cmath>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "check.hpp"
#include "contingency/v_groups.cuh"

namespace {

using C = cudaComplexType;

constexpr int N_BUS = 3;

C cplx(double re, double im) { return C{static_cast<decltype(C{}.x)>(re), static_cast<decltype(C{}.x)>(im)}; }

template <typename T>
T* to_device(const std::vector<T>& h)
{
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

// Run the gather into an (n_slots x N_BUS) batch prefilled with a sentinel; return it.
std::vector<C> gather(const std::vector<C>& groups, const std::vector<int>& group_of_active,
                      const std::vector<C>& base, const std::vector<int>& slots,
                      const std::vector<int>& rows, int n_slots)
{
    std::vector<C> h_dst(static_cast<size_t>(n_slots) * N_BUS, cplx(-9., -9.));
    C* d_dst = to_device(h_dst);
    C* d_groups = to_device(groups);
    int* d_gofa = to_device(group_of_active);
    C* d_base = to_device(base);
    int* d_slot = to_device(slots);
    int* d_row = to_device(rows);
    const cudaError_t e = launch_gather_groups_to_slots(d_dst, d_groups, d_gofa, d_base, d_slot, d_row,
                                                        static_cast<int>(slots.size()), N_BUS, nullptr);
    CHECK(e == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(C), cudaMemcpyDeviceToHost);
    for (void* p : {static_cast<void*>(d_dst), static_cast<void*>(d_groups), static_cast<void*>(d_gofa),
                    static_cast<void*>(d_base), static_cast<void*>(d_slot), static_cast<void*>(d_row)})
        cudaFree(p);
    return h_dst;
}

bool same(const C& a, const C& b) { return a.x == b.x && a.y == b.y; }

// two groups, a base case; row r of the batch is ACTIVE row r
const std::vector<C> GROUPS{cplx(1.01, -0.1), cplx(1.02, -0.2), cplx(1.03, -0.3),   // group 0
                            cplx(0.97, 0.1), cplx(0.98, 0.2), cplx(0.99, 0.3)};      // group 1
const std::vector<C> BASE{cplx(1., 0.), cplx(1., -0.05), cplx(1., -0.07)};

}  // namespace

TEST_CASE("a row of group g starts from row g of the groups")
{
    const auto out = gather(GROUPS, {1, 0}, BASE, {0, 1}, {0, 1}, 2);
    for (int b = 0; b < N_BUS; ++b) {
        CHECK(same(out[0 * N_BUS + b], GROUPS[1 * N_BUS + b]));
        CHECK(same(out[1 * N_BUS + b], GROUPS[0 * N_BUS + b]));
    }
}

TEST_CASE("group -1 and a phantom slot (row -1) start from the base case")
{
    const auto out = gather(GROUPS, {-1}, BASE, {0, 1}, {0, -1}, 2);
    for (int b = 0; b < N_BUS; ++b) {
        CHECK(same(out[0 * N_BUS + b], BASE[b]));
        CHECK(same(out[1 * N_BUS + b], BASE[b]));
    }
}

TEST_CASE("a non-finite entry of a group (a masked bus) starts from the base case")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<C> groups = GROUPS;
    groups[0 * N_BUS + 1] = cplx(nan, nan);   // group 0, bus 1
    groups[0 * N_BUS + 2] = cplx(1., inf);    // group 0, bus 2: one part is enough
    const auto out = gather(groups, {0}, BASE, {0}, {0}, 1);
    CHECK(same(out[0], groups[0]));
    CHECK(same(out[1], BASE[1]));
    CHECK(same(out[2], BASE[2]));
}

TEST_CASE("only the listed slots are written, in any order")
{
    // 4 slots; load slot 3 <- active row 1 (group 0) and slot 1 <- active row 0 (group 1)
    const auto out = gather(GROUPS, {1, 0}, BASE, {3, 1}, {1, 0}, 4);
    for (int b = 0; b < N_BUS; ++b) {
        CHECK(same(out[3 * N_BUS + b], GROUPS[0 * N_BUS + b]));
        CHECK(same(out[1 * N_BUS + b], GROUPS[1 * N_BUS + b]));
        CHECK(same(out[0 * N_BUS + b], cplx(-9., -9.)));   // untouched
        CHECK(same(out[2 * N_BUS + b], cplx(-9., -9.)));
    }
}

TEST_CASE("an empty load is a no-op")
{
    CHECK(launch_gather_groups_to_slots(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, N_BUS, nullptr)
          == cudaSuccess);
}

GPUSIM2GRID_TEST_MAIN(true)
