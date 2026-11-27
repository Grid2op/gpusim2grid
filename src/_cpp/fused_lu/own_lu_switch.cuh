// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// fused_lu/own_lu_switch.cuh -- own_lu::BatchedLu or top_lu::TopLu behind one interface
// =============================================================================
//
// Prototype hook (untracked): CudssBatchSolver holds this instead of a
// BatchedLu when the file exists (__has_include). GPUSIM2GRID_OWN_LU_TOP=1 at
// init() picks TopLu (own_lu with its trailing block dense in shared memory,
// top_lu.cuh); anything else keeps own_lu::BatchedLu, unchanged.
// =============================================================================

#ifndef GPUSIM2GRID_FUSED_LU_OWN_LU_SWITCH_CUH
#define GPUSIM2GRID_FUSED_LU_OWN_LU_SWITCH_CUH

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../own_lu/batched_lu.cuh"
#include "top_lu.cuh"

namespace top_lu {

class OwnLuSwitch {
public:
    void init(const own_lu::LuPlan& plan, int batch, cudaStream_t stream, double eps)
    {
        const char* e = std::getenv("GPUSIM2GRID_OWN_LU_TOP");
        top_on_ = e != nullptr && *e != '\0' && *e != '0';
        n_ent_ = plan.n_ent;
        if (top_on_) {
            top_.init(plan, batch, stream, eps);
            std::fprintf(stderr, "[own_lu] GPUSIM2GRID_OWN_LU_TOP: trailing block |T| = %d (levels >= %d of %d), "
                         "launches factor %d solve %d\n", top_.top_size(), top_.top_level(), plan.levels,
                         top_.launches_factor(), top_.launches_solve());
        } else {
            own_.init(plan, batch, stream, eps);
        }
        init_ = true;
    }
    void factorize(const cuda_real_type* values) { top_on_ ? top_.factorize(values) : own_.factorize(values); }
    void solve(const cuda_real_type* b, cuda_real_type* x) { top_on_ ? top_.solve(b, x) : own_.solve(b, x); }
    void refine(const cuda_real_type* values, const int* indptr, const int* indices, const cuda_real_type* b,
                cuda_real_type* x, int steps)
    {
        top_on_ ? top_.refine(values, indptr, indices, b, x, steps) : own_.refine(values, indptr, indices, b, x, steps);
    }
    void set_active(int nb) { top_on_ ? top_.set_active(nb) : own_.set_active(nb); }
    int active() const { return top_on_ ? top_.active() : own_.active(); }
    std::vector<int> clamped_counts() const { return top_on_ ? top_.clamped_counts() : own_.clamped_counts(); }
    int n_ent() const { return n_ent_; }
    bool initialized() const { return init_; }

private:
    own_lu::BatchedLu own_;
    TopLu top_;
    bool top_on_ = false, init_ = false;
    int n_ent_ = 0;
};

}  // namespace top_lu

#endif  // GPUSIM2GRID_FUSED_LU_OWN_LU_SWITCH_CUH
