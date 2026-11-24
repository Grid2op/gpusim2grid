// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef OWN_BRANCH_FLOWS_CUH
#define OWN_BRANCH_FLOWS_CUH

// =============================================================================
// contingency/own_branch_flows.cuh
//
// override_own_branch_flows — the flows (A, both sides) of the branches a row holds its own
// block of (Contingency::own_branches: a strategy's transformer values and bus moves in a
// contingency analysis, set_line_parameters / set_trafo_parameters in a scenario sweep),
// written over what compute_branch_flows_kernel computed from the shared *_eff blocks. Run
// after the zeroing of the rows' tripped branches: a branch a row opens at one end keeps its
// other end's current. Shared by ContingencyAnalysisSession and ScenarioSweepSession.
// =============================================================================

#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "../acpf_nr_kernels.cuh"           // override_branch_flows_kernel
#include "../contingency_analysis_helper.hpp"   // Contingency, RowBranchBlock
#include "../cuda_utils.h"                 // upload_h2d
#include "../cu_complex_utils.h"

template <class Driver>
void override_own_branch_flows(Driver& solver, const std::vector<Contingency>& contingencies, int n_bus,
                               const char* who)
{
    // the flows are those of the rows the last run solved, over the branches the driver holds: a
    // block outside them (rows rebuilt since, branch data set after the blocks) would be written
    // past the flow buffers
    const int n_run = solver.n_contingencies, n_bra = solver.n_branches_;
    std::vector<int> rows, branches, ft;
    std::vector<cudaComplexType> blk;
    std::vector<cuda_real_type> base;
    for (size_t c = 0; c < contingencies.size(); ++c)
        for (const RowBranchBlock& ob : contingencies[c].own_branches) {
            if (static_cast<long long>(c) >= n_run)
                throw std::runtime_error(std::string(who) + ": row " + std::to_string(c) + " has its own branch blocks "
                                         "but the last run solved " + std::to_string(n_run) + " rows: run() first");
            if (ob.branch >= n_bra)
                throw std::runtime_error(std::string(who) + ": branch " + std::to_string(ob.branch) + " of row "
                                         + std::to_string(c) + "'s own blocks is not one of the "
                                         + std::to_string(n_bra) + " branches");
            rows.push_back(static_cast<int>(c));
            branches.push_back(ob.branch);
            for (int q = 0; q < 4; ++q)
                blk.push_back(CudaFunHelper::my_make_cuComplex(static_cast<cuda_real_type>(ob.y[q].real()),
                                                               static_cast<cuda_real_type>(ob.y[q].imag())));
            ft.push_back(ob.from);
            ft.push_back(ob.to);
            base.push_back(static_cast<cuda_real_type>(ob.base_or_A));
            base.push_back(static_cast<cuda_real_type>(ob.base_ex_A));
        }
    if (rows.empty()) return;
    const cudaStream_t cs = solver.cs;
    thrust::device_vector<int> d_rows, d_br, d_ft;
    thrust::device_vector<cudaComplexType> d_blk;
    thrust::device_vector<cuda_real_type> d_base;
    upload_h2d(d_rows, rows.data(), rows.size(), cs);
    upload_h2d(d_br, branches.data(), branches.size(), cs);
    upload_h2d(d_blk, blk.data(), blk.size(), cs);
    upload_h2d(d_ft, ft.data(), ft.size(), cs);
    upload_h2d(d_base, base.data(), base.size(), cs);
    const int n = static_cast<int>(rows.size());
    constexpr int bs = 256;
    override_branch_flows_kernel<<<static_cast<unsigned>((n + bs - 1) / bs), bs, 0, cs>>>(
        thrust::raw_pointer_cast(solver.d_V_results.data()),
        thrust::raw_pointer_cast(d_rows.data()), thrust::raw_pointer_cast(d_br.data()),
        thrust::raw_pointer_cast(d_blk.data()),
        thrust::raw_pointer_cast(solver.d_branch_from.data()),
        thrust::raw_pointer_cast(solver.d_branch_to.data()),
        thrust::raw_pointer_cast(solver.d_base_current_A.data()),
        thrust::raw_pointer_cast(solver.d_base_current_ex_A.data()),
        thrust::raw_pointer_cast(solver.d_or_amps_results.data()),
        thrust::raw_pointer_cast(solver.d_ex_amps_results.data()),
        n_bus, solver.n_branches_, n,
        thrust::raw_pointer_cast(d_ft.data()), thrust::raw_pointer_cast(d_base.data()));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(who) + ": " + cudaGetErrorString(e));
    cudaStreamSynchronize(cs);   // the staging vectors die here
}

#endif  // OWN_BRANCH_FLOWS_CUH
