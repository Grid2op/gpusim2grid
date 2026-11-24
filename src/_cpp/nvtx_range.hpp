// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// nvtx_range.hpp -- NVTX ranges for Nsight Systems (nsys profile --trace=nvtx).
//
// Header-only NVTX3 from the CUDA toolkit; a no-op where its headers are
// missing (PyPI's CUDA wheels ship none). Without a profiler attached a range
// costs a few ns.
// =============================================================================

#ifndef GPUSIM2GRID_NVTX_RANGE_HPP
#define GPUSIM2GRID_NVTX_RANGE_HPP

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define GPUSIM2GRID_HAVE_NVTX 1
#  endif
#endif

namespace g2g_nvtx {

inline void push(const char* name)
{
#ifdef GPUSIM2GRID_HAVE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}

inline void pop()
{
#ifdef GPUSIM2GRID_HAVE_NVTX
    nvtxRangePop();
#endif
}

// One range for the life of the object.
struct Range {
    explicit Range(const char* name) { push(name); }
    ~Range() { pop(); }
    Range(const Range&) = delete;
    Range& operator=(const Range&) = delete;
};

}  // namespace g2g_nvtx

#endif  // GPUSIM2GRID_NVTX_RANGE_HPP
