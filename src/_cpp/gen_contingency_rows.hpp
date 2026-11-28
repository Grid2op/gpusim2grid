// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// gen_contingency_rows.hpp — what the generators each row disconnects need,
// row by row (mirrors lightsim2grid's BaseBatchSweep::_maybe_prepare_gen_contingency
// and _row_slack_weights), shared by ScenarioSweepSession (its
// set_contingency_gens mask) and ContingencyAnalysisSession (its contingencies
// tripping generators). `gen_off(r, g)`: whether row r disconnects generator g.
// =============================================================================

#ifndef GEN_CONTINGENCY_ROWS_HPP
#define GEN_CONTINGENCY_ROWS_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "dtypes.hpp"
#include "gen_contingency_data.hpp"
#include "ledger_data.hpp"
#include "slack_weight_rows.hpp"

namespace gen_contingency {

// The elements one row takes out: generators, storage units, hvdc lines (any order, duplicates
// allowed, ids out of range ignored).
struct RowOff {
    std::vector<int> gens, stos, hvdcs;
    void clear() { gens.clear(); stos.clear(); hvdcs.clear(); }
};

// `row_off(r, off)`: fills `off` (cleared) with what row r takes out.
// `required`: the buses some row takes their LAST local voltage controller off
// and that own no Vm unknown / Q equation in the base ledger (they need a
// reserved pair, see ledger_extend.hpp's add_switchable_vm_buses), sorted;
// `row_pv_to_pq`: per row, the buses it releases (sorted); `row_slack_off`: per
// row, the participating generators it takes out of the distributed slack (sorted).
// A row only visits the buses of the elements it takes out: most rows take out none
// (a contingency analysis of branch trips with a few generator rows among them).
template <class RowOffFn>
void prepare_rows(const GenContingencyData& gd, int n_rows, RowOffFn row_off,
                  const LedgerData* base_ledger,
                  std::vector<int>& required,
                  std::vector<std::vector<int>>& row_pv_to_pq,
                  std::vector<std::vector<int>>& row_slack_off)
{
    required.clear();
    row_pv_to_pq.assign(static_cast<size_t>(std::max(n_rows, 0)), std::vector<int>());
    row_slack_off.assign(static_cast<size_t>(std::max(n_rows, 0)), std::vector<int>());
    const int n_gen = gd.n_gen;

    // Which bus does each LOCAL voltage controller pin? (generators, storage units and
    // the converter stations holding their own bus)
    struct BusControllers { std::vector<int> gens, stos, hvdcs; };
    std::map<int, BusControllers> controllers;
    for (int g = 0; g < n_gen; ++g) {
        if (!gd.status[g] || !gd.local_vreg[g] || gd.on_group_bus[g]) continue;   // (a group holds its bus)
        const int b = gd.bus[g];
        if (b < 0) continue;
        controllers[b].gens.push_back(g);
    }
    for (int s = 0; s < gd.n_storage; ++s) {
        const int b = gd.storage_vreg_bus[static_cast<size_t>(s)];
        if (b >= 0) controllers[b].stos.push_back(s);
    }
    for (int h = 0; h < gd.n_hvdc; ++h)
        for (int side = 0; side < 2; ++side) {
            const int b = gd.station_vreg_bus[static_cast<size_t>(2 * h + side)];
            if (b >= 0) controllers[b].hvdcs.push_back(h);
        }

    const auto sorted_ids = [](std::vector<int>& v, int n) {
        v.erase(std::remove_if(v.begin(), v.end(), [n](int x) { return x < 0 || x >= n; }), v.end());
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    const auto has = [](const std::vector<int>& v, int x) { return std::binary_search(v.begin(), v.end(), x); };
    std::set<int> req;
    RowOff off;
    std::vector<int> cand;   // the buses of the row's elements that hold a local controller
    for (int r = 0; r < n_rows; ++r) {
        off.clear();
        row_off(r, off);
        sorted_ids(off.gens, n_gen);
        sorted_ids(off.stos, gd.n_storage);
        sorted_ids(off.hvdcs, gd.n_hvdc);
        if (off.gens.empty() && off.stos.empty() && off.hvdcs.empty()) continue;
        cand.clear();
        for (int g : off.gens) cand.push_back(gd.bus[g]);
        for (int s : off.stos) cand.push_back(gd.storage_vreg_bus[static_cast<size_t>(s)]);
        for (int h : off.hvdcs)
            for (int side = 0; side < 2; ++side) cand.push_back(gd.station_vreg_bus[static_cast<size_t>(2 * h + side)]);
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        for (int b : cand) {
            const auto it = controllers.find(b);
            if (it == controllers.end()) continue;
            bool all_off = true;
            for (int g : it->second.gens) if (!has(off.gens, g)) { all_off = false; break; }
            if (all_off) for (int s : it->second.stos) if (!has(off.stos, s)) { all_off = false; break; }
            if (all_off) for (int h : it->second.hvdcs) if (!has(off.hvdcs, h)) { all_off = false; break; }
            if (all_off) {
                row_pv_to_pq[static_cast<size_t>(r)].push_back(b);
                req.insert(b);
            }
        }
        for (int g : off.gens)
            if (gd.status[g] && gd.slack_participant[g])
                row_slack_off[static_cast<size_t>(r)].push_back(g);
    }

    // Only a bus that owns no Vm unknown / Q equation in the base ledger needs
    // a reserved pair (add_switchable_vm_buses skips the others anyway, so
    // this keeps `required` comparable to the reserved set).
    for (int b : req) {
        if (base_ledger &&
            (base_ledger->vm_col_of_bus[static_cast<size_t>(b)] >= 0 ||
             base_ledger->q_row_of_bus[static_cast<size_t>(b)] >= 0))
            continue;
        required.push_back(b);
    }
    std::sort(required.begin(), required.end());
}

// Why the batch cannot take this generator out of a row (empty: it can): a controller
// of a voltage-control group (a remote one, or on a bus a group holds) leaves its group
// by value, what neither the physical checks nor the gradients know how to read yet.
inline std::string gen_unsupported(const GenContingencyData& gd, int g, bool physical, bool adjoint)
{
    if (!gd.remote_vreg[g] && !gd.on_group_bus[g]) return std::string();
    if (physical)
        return "disconnects generator " + std::to_string(g) + ", a controller of a voltage-control group: "
               "the physical checks cannot leave it out of a row yet";
    if (adjoint)
        return "disconnects generator " + std::to_string(g) + ", a controller of a voltage-control group: "
               "the gradients do not support it yet";
    return std::string();
}

// The controllers of the voltage-control groups these elements are (positions in the ledger's
// controller list, sorted).
inline std::vector<int> tripped_controllers(const LedgerData& ld, const std::vector<int>& gens,
                                            const std::vector<int>& svcs, const std::vector<int>& hvdcs)
{
    std::vector<int> out;
    if (gens.empty() && svcs.empty() && hvdcs.empty()) return out;
    const auto named = [](const std::vector<int>& ids, int id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    for (size_t c = 0; c < ld.vc_kind.size() && c < ld.vc_elem_id.size(); ++c) {
        const int kind = ld.vc_kind[c], id = ld.vc_elem_id[c];
        // VoltageControlSolverData::Kind: GEN, SVC, HVDC_SIDE_1, HVDC_SIDE_2, STORAGE
        if ((kind == 0 && named(gens, id)) || (kind == 1 && named(svcs, id)) || ((kind == 2 || kind == 3) && named(hvdcs, id)))
            out.push_back(static_cast<int>(c));
    }
    return out;
}

// Why a row cannot lose these controllers (empty: it can): a frozen one holds its output in
// Sbus, and a group left with nobody regulating frees its regulated bus -- not a slack bus,
// whose magnitude nothing would hold any more.
inline std::string tripped_controllers_unsupported(const LedgerData& ld, const std::vector<int>& tripped)
{
    if (tripped.empty()) return std::string();
    const auto is_tripped = [&](int c) { return std::binary_search(tripped.begin(), tripped.end(), c); };
    for (int c : tripped) {
        const size_t cc = static_cast<size_t>(c);
        if (cc >= ld.vc_held.size() || !ld.vc_held[cc]) continue;
        // a held SVC is an idle voltage monitor of the outer loops, held at 0: tripped, it is
        // pinned at 0 all the same (and VoltageMonitoring never switches it on)
        const bool idle_svc = cc < ld.vc_kind.size() && ld.vc_kind[cc] == 1 &&
                              (cc >= ld.vc_q_held.size() || ld.vc_q_held[cc] == 0.);
        if (!idle_svc)
            return "disconnects controller " + std::to_string(c) + " of a voltage-control group, held at a reactive limit";
    }
    for (int g = 0; g < ld.vc_n_groups(); ++g) {
        const int bus = ld.vc_reg_bus[static_cast<size_t>(g)];
        if (std::find(ld.vc_free_vm_slack_buses.begin(), ld.vc_free_vm_slack_buses.end(), bus) == ld.vc_free_vm_slack_buses.end())
            continue;
        const int first = ld.vc_grp_start[static_cast<size_t>(g)], cnt = ld.vc_grp_count[static_cast<size_t>(g)];
        bool one_tripped = false, someone_left = false;
        for (int c = first; c < first + cnt; ++c) {
            const bool held = static_cast<size_t>(c) < ld.vc_held.size() && ld.vc_held[static_cast<size_t>(c)];
            if (is_tripped(c)) one_tripped = one_tripped || !held;   // a held one regulated nothing
            else if (!held) someone_left = true;
        }
        if (one_tripped && !someone_left)
            return "disconnects every controller of the group regulating a slack bus (solver bus " + std::to_string(bus) + ")";
    }
    return std::string();
}

// What a row has lost among the controllers: those it trips and those on a bus it masks.
inline std::vector<int> lost_controllers(const LedgerData& ld, const std::vector<int>& tripped,
                                         const std::vector<int>& masked_buses)
{
    std::vector<int> out = tripped;
    for (size_t c = 0; c < ld.vc_bus.size(); ++c)
        if (std::find(masked_buses.begin(), masked_buses.end(), ld.vc_bus[c]) != masked_buses.end())
            out.push_back(static_cast<int>(c));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// What a row holds by value (SlotVcHold; (controller, Q pu) sorted by controller): in every
// group it loses a controller of (`lost`, lost_controllers), the lost ones at 0 and the ones
// it pins (Contingency::vc_pinned_ctrl) at their targets -- the holding formulation then picks
// the group's reference among the others, and build_mask_entries leaves those pins to it
// (MaskConfig::vc_hold_by_value). A pin wins over a loss, as the pin streams did.
inline std::vector<std::pair<int, double>> held_by_value(const LedgerData& ld, const std::vector<int>& lost,
                                                         const std::vector<std::pair<int, double>>& pins)
{
    std::map<int, double> held;
    if (lost.empty()) return {};
    std::set<int> groups;
    for (int c : lost) {
        if (c < 0 || c >= static_cast<int>(ld.vc_group.size())) continue;
        groups.insert(ld.vc_group[static_cast<size_t>(c)]);
        held[c] = 0.;
    }
    for (const auto& pc : pins)
        if (pc.first >= 0 && pc.first < static_cast<int>(ld.vc_group.size()) &&
            groups.count(ld.vc_group[static_cast<size_t>(pc.first)]))
            held[pc.first] = pc.second;
    return std::vector<std::pair<int, double>>(held.begin(), held.end());
}

// Per-row normalised slack weights ([n_rows * n_slack], ORIGINAL order) without
// the participating generators each row takes out (row_slack_off), on the base
// participant layout (`slack_bus`: the bus of each slack index). The storage
// units taking part in the slack keep their share on every row (no row
// disconnects one). Empty when no row takes anything out nor gives weights of its own.
// `ref_bus` takes the whole share of a row that leaves no participant.
// `over` (nullable): the raw weights rows give some units (ScenarioSweep
// set_gen_slack_weights / set_storage_slack_weights, lightsim2grid PR #303) -- a
// positive one makes a connected unit a participant, 0 takes it out; `gen_off(r, g)`
// says whether row r disconnects generator g (a unit only the row's weight brings in
// is not in row_slack_off: one the row disconnects takes no share all the same).
template <class StoOff, class GenOff = std::nullptr_t>
std::vector<cuda_real_type> row_slack_weights(const GenContingencyData& gd,
                                                     const std::vector<std::vector<int>>& row_slack_off,
                                                     int n_rows, int n_bus,
                                                     const std::vector<int>& slack_bus,
                                                     const std::vector<cuda_real_type>& base_w,
                                                     int ref_bus, const std::string& who, StoOff sto_off,
                                                     const SlackWeightRows* over = nullptr,
                                                     GenOff gen_off = nullptr)
{
    std::vector<cuda_real_type> out;
    const int n_slack = static_cast<int>(slack_bus.size());
    if (n_slack <= 0) return out;
    // whether the row takes a storage unit of the slack out
    const auto row_sto_off = [&](int r) {
        for (int s = 0; s < gd.n_storage; ++s)
            if (gd.storage_bus[static_cast<size_t>(s)] >= 0 && sto_off(r, s)) return true;
        return false;
    };
    const auto row_weighted = [&](int r) { return over != nullptr && over->has_row(r); };
    const auto gen_off_row = [&](int r, int g) {
        if constexpr (std::is_same_v<GenOff, std::nullptr_t>) { (void)r; (void)g; return false; }
        else return static_cast<bool>(gen_off(r, g));
    };
    bool any = false;
    for (const auto& v : row_slack_off) if (!v.empty()) { any = true; break; }
    for (int r = 0; r < n_rows && !any; ++r) any = row_sto_off(r) || row_weighted(r);
    if (!any) return out;

    out.assign(static_cast<size_t>(n_rows) * n_slack, static_cast<cuda_real_type>(0.));
    std::vector<double> w(static_cast<size_t>(n_bus));
    std::vector<char>   off(static_cast<size_t>(gd.n_gen));
    for (int r = 0; r < n_rows; ++r) {
        cuda_real_type* row = out.data() + static_cast<ptrdiff_t>(r) * n_slack;
        const bool sto_out = row_sto_off(r);
        const bool weighted = row_weighted(r);
        if (row_slack_off[static_cast<size_t>(r)].empty() && !sto_out && !weighted) {
            std::copy(base_w.begin(), base_w.end(), row);
            continue;
        }
        std::fill(off.begin(), off.end(), 0);
        for (int g : row_slack_off[static_cast<size_t>(r)]) off[static_cast<size_t>(g)] = 1;
        std::fill(w.begin(), w.end(), 0.0);
        double sum = 0.0;
        const std::vector<double>& sto_w = gd.storage_slack_weight_bus;
        if (weighted && !over->sto.empty()) {
            // per unit: the row's weight where it gives one (any connected unit), its own otherwise
            for (int s = 0; s < gd.n_storage; ++s) {
                double ws;
                int b;
                if (over->sto.has(r, s)) {
                    ws = over->sto.of(r, s);
                    b = static_cast<size_t>(s) < gd.storage_any_bus.size() ? gd.storage_any_bus[static_cast<size_t>(s)] : -1;
                    if (!(ws >= slack_weight_rows::MIN_WEIGHT)) continue;
                } else {
                    b = gd.storage_bus[static_cast<size_t>(s)];
                    ws = gd.storage_slack_weight[static_cast<size_t>(s)];
                }
                if (b < 0 || sto_off(r, s)) continue;
                w[static_cast<size_t>(b)] += ws;
                sum += ws;
            }
        } else if (!sto_out) {
            for (int b = 0; b < n_bus && b < static_cast<int>(sto_w.size()); ++b) {
                w[static_cast<size_t>(b)] = sto_w[static_cast<size_t>(b)];
                sum += sto_w[static_cast<size_t>(b)];
            }
        } else {
            for (int s = 0; s < gd.n_storage; ++s) {
                const int b = gd.storage_bus[static_cast<size_t>(s)];
                if (b < 0 || sto_off(r, s)) continue;
                w[static_cast<size_t>(b)] += gd.storage_slack_weight[static_cast<size_t>(s)];
                sum += gd.storage_slack_weight[static_cast<size_t>(s)];
            }
        }
        for (int g = 0; g < gd.n_gen; ++g) {
            if (!gd.status[g] || off[static_cast<size_t>(g)]) continue;
            double wg = gd.slack_weight[g];
            if (weighted && over->gen.has(r, g)) {
                wg = over->gen.of(r, g);
                if (!(wg >= slack_weight_rows::MIN_WEIGHT) || gen_off_row(r, g)) continue;
            } else if (!gd.slack_participant[g]) {
                continue;
            }
            const int b = gd.bus[g];
            if (b < 0) continue;
            w[static_cast<size_t>(b)] += wg;
            sum += wg;
        }
        if (std::abs(sum) < 1e-12) {
            // Every participant off: the reference bus keeps the whole share
            // (the angle reference is a property of the batch, picked once).
            std::fill(w.begin(), w.end(), 0.0);
            if (ref_bus >= 0 && ref_bus < n_bus) w[static_cast<size_t>(ref_bus)] = 1.0;
        } else {
            for (double& x : w) x /= sum;
        }
        for (int k = 0; k < n_slack; ++k) {
            const int b = slack_bus[static_cast<size_t>(k)];
            row[k] = static_cast<cuda_real_type>(w[static_cast<size_t>(b)]);
            w[static_cast<size_t>(b)] = 0.0;   // consumed
        }
        // A survivor bus that is not a base participant cannot exist (the
        // survivors are a subset of the base participants, and a row's own
        // weights are checked against the slack buses before); the degenerate
        // "reference gets 1" case can, when the reference carries no base
        // weight (several slacks) -- refuse rather than silently drop it.
        for (int b = 0; b < n_bus; ++b)
            if (w[static_cast<size_t>(b)] != 0.0)
                throw std::runtime_error(
                    who + ": row " + std::to_string(r) + " re-weights "
                    "the distributed slack onto bus " + std::to_string(b) + ", which "
                    "owns no slack column entry in the base case (a reference "
                    "bus with zero base weight). Keep at least one participating "
                    "generator connected on that row.");
    }
    return out;
}

}  // namespace gen_contingency

#endif  // GEN_CONTINGENCY_ROWS_HPP
