// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GPUSIM2GRID_TESTS_CHECK_HPP
#define GPUSIM2GRID_TESTS_CHECK_HPP

// =============================================================================
// tests/cpp/check.hpp -- a minimal, dependency-free test harness for the C++/CUDA
// unit tests (no download needed: build machines may be offline).
//
//   TEST_CASE(name) { CHECK(cond); CHECK_CLOSE(a, b, tol); CHECK_THROWS(expr); }
//   int main() { return run_tests(); }          // or GPUSIM2GRID_TEST_MAIN(needs_gpu)
//
// A failed check prints file:line and the expression, the test goes on; the
// process returns 1 if any check failed. A test executable that needs a GPU
// returns SKIP_CODE (77, CTest's SKIP_RETURN_CODE) on a machine without one.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace g2g_test {

inline constexpr int SKIP_CODE = 77;

struct Registry {
    std::vector<std::pair<std::string, std::function<void()>>> tests;
    int failures = 0;
    static Registry& get()
    {
        static Registry r;
        return r;
    }
};

struct Register {
    Register(const char* name, std::function<void()> fn) { Registry::get().tests.emplace_back(name, std::move(fn)); }
};

inline void fail(const char* file, int line, const std::string& what)
{
    ++Registry::get().failures;
    std::fprintf(stderr, "%s:%d: FAILED: %s\n", file, line, what.c_str());
}

inline bool has_gpu()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

inline int run_tests(bool needs_gpu)
{
    if (needs_gpu && !has_gpu()) {
        std::printf("no CUDA device: skipped\n");
        return SKIP_CODE;
    }
    Registry& r = Registry::get();
    for (auto& [name, fn] : r.tests) {
        const int before = r.failures;
        fn();
        std::printf("[%s] %s\n", r.failures == before ? "  ok  " : " FAIL ", name.c_str());
    }
    std::printf("%zu tests, %d failed checks\n", r.tests.size(), r.failures);
    return r.failures == 0 ? 0 : 1;
}

}  // namespace g2g_test

#define G2G_CAT_(a, b) a##b
#define G2G_CAT(a, b) G2G_CAT_(a, b)

#define TEST_CASE(name)                                                                            \
    static void G2G_CAT(g2g_test_fn_, __LINE__)();                                                 \
    static const g2g_test::Register G2G_CAT(g2g_test_reg_, __LINE__)(name, G2G_CAT(g2g_test_fn_, __LINE__)); \
    static void G2G_CAT(g2g_test_fn_, __LINE__)()

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) g2g_test::fail(__FILE__, __LINE__, #cond);                                    \
    } while (0)

// CHECK with a note of where (a row, a seed) in the failure message
#define CHECK_MSG(cond, msg)                                                                       \
    do {                                                                                           \
        if (!(cond)) g2g_test::fail(__FILE__, __LINE__, std::string(#cond) + " -- " + (msg));     \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                                                     \
    do {                                                                                           \
        const double g2g_a_ = static_cast<double>(a), g2g_b_ = static_cast<double>(b);             \
        if (!(std::fabs(g2g_a_ - g2g_b_) <= static_cast<double>(tol)))                             \
            g2g_test::fail(__FILE__, __LINE__,                                                     \
                           std::string(#a " == " #b ": ") + std::to_string(g2g_a_) + " vs "       \
                               + std::to_string(g2g_b_));                                          \
    } while (0)

#define CHECK_THROWS(expr)                                                                         \
    do {                                                                                           \
        bool g2g_thrown_ = false;                                                                  \
        try {                                                                                      \
            (void)(expr);                                                                          \
        } catch (...) {                                                                            \
            g2g_thrown_ = true;                                                                    \
        }                                                                                          \
        if (!g2g_thrown_) g2g_test::fail(__FILE__, __LINE__, "no exception from " #expr);          \
    } while (0)

#define GPUSIM2GRID_TEST_MAIN(needs_gpu)                                                           \
    int main() { return g2g_test::run_tests(needs_gpu); }

#endif  // GPUSIM2GRID_TESTS_CHECK_HPP
