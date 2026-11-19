// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// own_lu::BatchedLu's fused triangular solve (lu_solve_run: a run of narrow levels in one 1024-thread
// kernel): it can always be launched at that size, and gives the bits of the per-level kernels
// (GPUSIM2GRID_OWN_LU_FUSE=0) -- on a chain of single-row levels and on an arrow matrix, whose
// dense last row is shared over several warps (the split path). Needs a GPU.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "check.hpp"
#include "own_lu/batched_lu.cuh"
#include "own_lu/lu_plan.hpp"

using own_lu::BatchedLu;
using T = cuda_real_type;

namespace {

struct Csr {
    int n = 0;
    std::vector<int> indptr, indices;
};

// tridiagonal (every level of the solves one row: all narrow), plus, with `arrow`, a dense last row
// and column (its forward row has n - 1 contributions: split over several warps)
Csr pattern(int n, bool arrow)
{
    Csr a;
    a.n = n;
    a.indptr.push_back(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const bool band = std::abs(i - j) <= 1;
            const bool edge = arrow && (i == n - 1 || j == n - 1);
            if (band || edge) a.indices.push_back(j);
        }
        a.indptr.push_back(static_cast<int>(a.indices.size()));
    }
    return a;
}

// B diagonally dominant matrices on the pattern (each its own values), slot-major
std::vector<T> values(const Csr& a, int B)
{
    const int nnz = static_cast<int>(a.indices.size());
    std::vector<T> v(static_cast<size_t>(B) * nnz);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < a.n; ++i)
            for (int p = a.indptr[i]; p < a.indptr[i + 1]; ++p) {
                const int j = a.indices[p];
                v[static_cast<size_t>(b) * nnz + p] =
                    i == j ? T(4 * a.n + b) : T(-1.0 - 0.01 * ((i * 7 + j * 3 + b) % 11));
            }
    return v;
}

// the solutions of A_b x = rhs_b, with the fused kernel or (fuse = false) one launch per level
std::vector<T> solve(const Csr& a, int B, bool fuse)
{
    if (fuse) unsetenv("GPUSIM2GRID_OWN_LU_FUSE");
    else setenv("GPUSIM2GRID_OWN_LU_FUSE", "0", 1);
    std::vector<int> order(static_cast<size_t>(a.n));
    for (int i = 0; i < a.n; ++i) order[static_cast<size_t>(i)] = i;
    const own_lu::LuPlan plan = own_lu::build_lu_plan(a.n, a.indptr, a.indices, order);
    BatchedLu lu;
    lu.init(plan, B, nullptr);
    unsetenv("GPUSIM2GRID_OWN_LU_FUSE");
    const std::vector<T> hv = values(a, B);
    std::vector<T> hb(static_cast<size_t>(B) * a.n);
    for (size_t k = 0; k < hb.size(); ++k) hb[k] = T(1.0 + 0.1 * static_cast<double>(k % 13));
    thrust::device_vector<T> dv(hv.begin(), hv.end()), db(hb.begin(), hb.end()), dx(hb.size());
    lu.factorize(thrust::raw_pointer_cast(dv.data()));
    lu.solve(thrust::raw_pointer_cast(db.data()), thrust::raw_pointer_cast(dx.data()));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    thrust::host_vector<T> hx = dx;
    return std::vector<T>(hx.begin(), hx.end());
}

bool same_bits(const std::vector<T>& x, const std::vector<T>& y)
{
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

// |A_b x_b - rhs_b|_inf over the batch
double residual(const Csr& a, int B, const std::vector<T>& x)
{
    const int nnz = static_cast<int>(a.indices.size());
    const std::vector<T> v = values(a, B);
    double worst = 0.;
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < a.n; ++i) {
            double s = 0.;
            for (int p = a.indptr[i]; p < a.indptr[i + 1]; ++p)
                s += static_cast<double>(v[static_cast<size_t>(b) * nnz + p]) *
                     static_cast<double>(x[static_cast<size_t>(b) * a.n + a.indices[p]]);
            const size_t k = static_cast<size_t>(b) * a.n + i;
            worst = std::fmax(worst, std::fabs(s - (1.0 + 0.1 * static_cast<double>(k % 13))));
        }
    return worst;
}

const double TOL = sizeof(T) == 4 ? 1e-3 : 1e-10;

}  // namespace

TEST_CASE("the fused solve kernel launches at its block size")
{
    CHECK(BatchedLu::fused_solve_supported());
}

TEST_CASE("chain of single-row levels: fused = per level, bit for bit")
{
    const Csr a = pattern(50, false);
    for (int B : {1, 31, 40, 64}) {
        const std::vector<T> fused = solve(a, B, true), plain = solve(a, B, false);
        CHECK(same_bits(fused, plain));
        CHECK(residual(a, B, fused) < TOL);
    }
}

TEST_CASE("arrow matrix (a split level in the run): fused = per level, bit for bit")
{
    const Csr a = pattern(40, true);
    for (int B : {1, 40}) {
        const std::vector<T> fused = solve(a, B, true), plain = solve(a, B, false);
        CHECK(same_bits(fused, plain));
        CHECK(residual(a, B, fused) < TOL);
    }
}

GPUSIM2GRID_TEST_MAIN(true)
