// Configure-time probe (see MatchLs2gEigenAbi.cmake): prints, as two CMake lists, the
// Eigen SIMD/alignment ABI tag lightsim2grid_core was compiled with (a real call into the
// library) and the one this compiler gives with the flags under test (evaluated here).
#include <cstdio>

#include <Ls2gAbiTag.hpp>

static void print_tag(const char* name, const ls2g::Ls2gAbiTag& t)
{
    std::printf("%s=%u;%u.%u.%u.%u;%d;%d;%d;%d;%d;%d\n", name, t.eigen_max_align_bytes,
                t.eigen_world_version, t.eigen_major_version, t.eigen_minor_version,
                t.eigen_patch_version, int(t.vectorize_sse), int(t.vectorize_avx),
                int(t.vectorize_avx2), int(t.vectorize_avx512), int(t.vectorize_fma),
                int(t.vectorize_neon));
}

int main()
{
    print_tag("CORE", ls2g::core_abi_tag());
    print_tag("SELF", ls2g::ls2g_current_abi_tag());
    return 0;
}
