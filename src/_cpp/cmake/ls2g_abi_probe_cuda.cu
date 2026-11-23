// Configure-time probe (see MatchLs2gEigenAbi.cmake): compiles only if nvcc's host compiler,
// with the flags under test, gives Eigen the SIMD/alignment settings lightsim2grid_core was
// built with (passed in as G2G_EXPECT_*).
//
// Eigen never vectorizes in a CUDA translation unit, host pass included (EIGEN_GPUCC skips the
// whole EIGEN_VECTORIZE_* block of ConfigureVectorization.h), so EIGEN_VECTORIZE_AVX & co. are
// undefined here whatever -march says: testing them failed every build against a
// lightsim2grid_core compiled with AVX. The ISA is therefore read off the host compiler's own
// macros, mapped the way ConfigureVectorization.h maps them for a non-CUDA file (GCC/Clang,
// no SYCL), i.e. what Eigen would turn on with these flags. EIGEN_MAX_ALIGN_BYTES -- what
// a CUDA TU actually shares with lightsim2grid_core: object layout, aligned (de)allocation --
// does follow the ISA under nvcc and is compared as is.
#include <Eigen/Core>

#ifndef __CUDA_ARCH__
#if defined(__AVX512F__)
#define G2G_HAS_AVX512 1
#else
#define G2G_HAS_AVX512 0
#endif
#if defined(__AVX2__) || G2G_HAS_AVX512
#define G2G_HAS_AVX2 1
#else
#define G2G_HAS_AVX2 0
#endif
#if defined(__AVX__) || G2G_HAS_AVX2
#define G2G_HAS_AVX 1
#else
#define G2G_HAS_AVX 0
#endif
#if defined(__FMA__) || G2G_HAS_AVX512
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
