// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GEN_V_OVERRIDE_HPP
#define GEN_V_OVERRIDE_HPP

// =============================================================================
// contingency/gen_v_override.hpp
//
// GenVOverride — host-side precomputed per-scenario generator target-voltage
// data, shared by InjectionBatch and ScenarioSweepBatch's prepare_Ybus_batch
// (see acpf_nr_kernels.cuh's apply_gen_v_kernel). Built once per run() by
// build_gen_v_override() from the session's (n_scenarios x n_gen) gen_v
// matrix (InjectionSweepSession::set_gen_v / ScenarioSweepSession::set_gen_v),
// filtered down to just the generators whose OWN bus is Vm-fixed (PV or
// slack) — see set_gen_v()'s doc for why that filter is both necessary (a
// PQ-bus reseed would just be overwritten by the very next NR iteration) and
// sufficient (a VoltageControl-regulated remote bus is PQ-classified with its
// own border row, so it is never in pv/slack and is correctly left alone).
// Empty (k_active() == 0) is a legal, cheap "no override configured" state.
// =============================================================================

#include "../dtypes.hpp"

#include <vector>

struct GenVOverride {
    std::vector<int>            h_active_bus;   // [k_active] AC-solver bus id
    std::vector<cuda_real_type> h_gen_v_all;    // [n_rows * k_active], row-major

    int k_active() const { return static_cast<int>(h_active_bus.size()); }
};

// gen_v           : (n_rows x n_gen) target vm_pu, row-major, NaN = leave unset
// gen_bus         : (n_gen,) AC-solver bus id per generator, -1 = disconnected
// is_vm_fixed_bus : (n_bus,) truthy where the bus's own magnitude is not an NR
//                   unknown (PV ∪ slack) — see set_gen_v()'s doc
template <typename RealMat, typename IntVec, typename CharVec>
GenVOverride build_gen_v_override(const RealMat& gen_v,
                                  const IntVec& gen_bus,
                                  const CharVec& is_vm_fixed_bus)
{
    GenVOverride out;
    const int n_gen = static_cast<int>(gen_bus.size());
    const int n_bus = static_cast<int>(is_vm_fixed_bus.size());

    std::vector<int> cols;
    cols.reserve(static_cast<size_t>(n_gen));
    for (int g = 0; g < n_gen; ++g) {
        const int bus = gen_bus[g];
        if (bus >= 0 && bus < n_bus && is_vm_fixed_bus[bus]) cols.push_back(g);
    }
    if (cols.empty()) return out;

    out.h_active_bus.reserve(cols.size());
    for (int g : cols) out.h_active_bus.push_back(gen_bus[g]);

    const int n_rows = static_cast<int>(gen_v.rows());
    const size_t k = cols.size();
    out.h_gen_v_all.resize(static_cast<size_t>(n_rows) * k);
    for (int r = 0; r < n_rows; ++r)
        for (size_t j = 0; j < k; ++j)
            out.h_gen_v_all[static_cast<size_t>(r) * k + j] =
                static_cast<cuda_real_type>(gen_v(r, cols[j]));

    return out;
}

#endif // GEN_V_OVERRIDE_HPP
