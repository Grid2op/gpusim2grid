// Configure-time probe (see MatchLs2gEigenAbi.cmake): compiles only if nvcc's host compiler,
// with the flags under test, gives Eigen the SIMD/alignment settings lightsim2grid_core was
// built with (passed in as G2G_EXPECT_*).
#include <Eigen/Core>

#ifndef __CUDA_ARCH__
#ifdef EIGEN_VECTORIZE_AVX
#define G2G_HAS_AVX 1
#else
#define G2G_HAS_AVX 0
#endif
#ifdef EIGEN_VECTORIZE_AVX2
#define G2G_HAS_AVX2 1
#else
#define G2G_HAS_AVX2 0
#endif
#ifdef EIGEN_VECTORIZE_AVX512
#define G2G_HAS_AVX512 1
#else
#define G2G_HAS_AVX512 0
#endif
#ifdef EIGEN_VECTORIZE_FMA
#define G2G_HAS_FMA 1
#else
#define G2G_HAS_FMA 0
#endif
static_assert(EIGEN_MAX_ALIGN_BYTES == G2G_EXPECT_ALIGN, "EIGEN_MAX_ALIGN_BYTES differs");
static_assert(G2G_HAS_AVX == G2G_EXPECT_AVX, "AVX differs");
static_assert(G2G_HAS_AVX2 == G2G_EXPECT_AVX2, "AVX2 differs");
static_assert(G2G_HAS_AVX512 == G2G_EXPECT_AVX512, "AVX512 differs");
static_assert(G2G_HAS_FMA == G2G_EXPECT_FMA, "FMA differs");
#endif

int main() { return 0; }
