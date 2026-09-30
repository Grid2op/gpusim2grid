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
// filtered down to the generators whose REGULATED bus is driven by a set-point:
// a Vm-fixed bus (PV or slack without a Vm unknown -- a PQ-bus reseed would
// just be overwritten by the very next NR iteration), or the regulated bus of
// a VoltageControl group, whose bordered row |V_reg| + s.Q - v_set = 0 then
// takes the column's value as that row's v_set (per slot, see
// apply_gen_vset_kernel). A remote regulator's set-point lives there, not at
// its own bus: keying it on its own bus silently dropped it.
// Empty (k_active() == 0) is a legal, cheap "no override configured" state.
// =============================================================================

#include "../dtypes.hpp"

#include <vector>

struct GenVOverride {
    std::vector<int>            h_active_bus;   // [k_active] AC-solver bus id
    // [k_active] VoltageControl group whose v_set this column drives, -1 for a
    // column that reseeds a Vm-fixed bus (see build_gen_v_override)
    std::vector<int>            h_active_vc_group;
    std::vector<cuda_real_type> h_gen_v_all;    // [n_rows * k_active], row-major

    int k_active() const { return static_cast<int>(h_active_bus.size()); }
    bool has_vc_columns() const {
        for (int g : h_active_vc_group) if (g >= 0) return true;
        return false;
    }
};

// Which generator columns a gen_v matrix actually drives, and how. gen_bus[g]
// is the AC-solver bus generator g REGULATES (InjectionElements.gen_v_bus, -1
// for none). That bus is either
//   * Vm-fixed (PV / slack with no Vm unknown in the ledger): the column
//     reseeds |V| there (vc_group -1), or
//   * the regulated bus of a VoltageControl group (vc_group_of_bus[bus] >= 0):
//     the column sets that group's per-row v_set (and reseeds |V| there too,
//     as a start value -- lightsim2grid's set_vm writes the regulated bus).
// Anything else (a PQ bus nothing borders) is not driven.
template <typename IntVec, typename CharVec>
void gen_v_active_columns(const IntVec& gen_bus,
                          const CharVec& is_vm_fixed_bus,
                          const std::vector<int>& vc_group_of_bus,
                          std::vector<int>& cols,
                          std::vector<int>& bus_out,
                          std::vector<int>& group_out)
{
    cols.clear(); bus_out.clear(); group_out.clear();
    const int n_gen = static_cast<int>(gen_bus.size());
    const int n_bus = static_cast<int>(is_vm_fixed_bus.size());
    for (int g = 0; g < n_gen; ++g) {
        const int bus = gen_bus[g];
        if (bus < 0 || bus >= n_bus) continue;
        const int grp = (static_cast<int>(vc_group_of_bus.size()) > bus)
                        ? vc_group_of_bus[static_cast<size_t>(bus)] : -1;
        if (grp < 0 && !is_vm_fixed_bus[bus]) continue;
        cols.push_back(g);
        bus_out.push_back(bus);
        group_out.push_back(grp);
    }
}

// gen_v           : (n_rows x n_gen) target vm_pu, row-major, NaN = leave unset
// gen_bus         : (n_gen,) AC-solver bus each generator regulates, -1 = none
// is_vm_fixed_bus : (n_bus,) truthy where the bus's magnitude is not an NR
//                   unknown (PV ∪ slack, no Vm column) — see set_gen_v()'s doc
// vc_group_of_bus : (n_bus,) VoltageControl group regulating the bus, or -1
//                   (empty: no VoltageControl)
// The two per-bus maps gen_v_active_columns reads, built once per session.
//   is_vm_fixed_bus : pv ∪ slack_ids, minus every bus that owns a Vm unknown
//                     in the (un-extended) ledger -- a slack bus nothing pins
//                     locally keeps a free Vm + Q equation in lightsim2grid's
//                     formulation (VoltageControlPlan's free_vm_slack), so a
//                     reseed there is overwritten by the first NR step.
//   vc_group_of_bus : per bus, the VoltageControl group regulating it, or -1.
template <typename IdxVec>
void build_gen_v_bus_maps(int n_bus, const IdxVec& pv, const IdxVec& slack_ids,
                          const std::vector<int>& vm_col_of_bus,
                          const std::vector<int>& vc_reg_bus,
                          std::vector<char>& is_vm_fixed_bus,
                          std::vector<int>& vc_group_of_bus)
{
    is_vm_fixed_bus.assign(static_cast<size_t>(n_bus), 0);
    for (Eigen::Index i = 0; i < pv.size(); ++i) {
        const int b = pv(i);
        if (b >= 0 && b < n_bus) is_vm_fixed_bus[static_cast<size_t>(b)] = 1;
    }
    for (Eigen::Index i = 0; i < slack_ids.size(); ++i) {
        const int b = slack_ids(i);
        if (b >= 0 && b < n_bus) is_vm_fixed_bus[static_cast<size_t>(b)] = 1;
    }
    for (int b = 0; b < n_bus && b < static_cast<int>(vm_col_of_bus.size()); ++b)
        if (vm_col_of_bus[static_cast<size_t>(b)] >= 0) is_vm_fixed_bus[static_cast<size_t>(b)] = 0;
    vc_group_of_bus.assign(static_cast<size_t>(n_bus), -1);
    for (int g = 0; g < static_cast<int>(vc_reg_bus.size()); ++g) {
        const int b = vc_reg_bus[static_cast<size_t>(g)];
        if (b >= 0 && b < n_bus) vc_group_of_bus[static_cast<size_t>(b)] = g;
    }
}

template <typename RealMat, typename IntVec, typename CharVec>
GenVOverride build_gen_v_override(const RealMat& gen_v,
                                  const IntVec& gen_bus,
                                  const CharVec& is_vm_fixed_bus,
                                  const std::vector<int>& vc_group_of_bus = {})
{
    GenVOverride out;
    std::vector<int> cols;
    gen_v_active_columns(gen_bus, is_vm_fixed_bus, vc_group_of_bus,
                         cols, out.h_active_bus, out.h_active_vc_group);
    if (cols.empty()) return out;

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
