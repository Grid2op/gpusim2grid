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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dtypes.hpp"
#include "gen_contingency_data.hpp"
#include "ledger_data.hpp"

namespace gen_contingency {

// `required`: the buses some row takes their LAST local voltage controller off
// and that own no Vm unknown / Q equation in the base ledger (they need a
// reserved pair, see ledger_extend.hpp's add_switchable_vm_buses), sorted;
// `row_pv_to_pq`: per row, the buses it releases (sorted); `row_slack_off`: per
// row, the participating generators it takes out of the distributed slack.
template <class GenOff, class StoOff, class HvdcOff>
void prepare_rows(const GenContingencyData& gd, int n_rows, GenOff gen_off, StoOff sto_off, HvdcOff hvdc_off,
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
        if (!gd.status[g] || !gd.local_vreg[g]) continue;
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

    std::set<int> req;
    for (int r = 0; r < n_rows; ++r) {
        for (const auto& kv : controllers) {
            bool all_off = true;
            for (int g : kv.second.gens) if (!gen_off(r, g)) { all_off = false; break; }
            if (all_off) for (int s : kv.second.stos) if (!sto_off(r, s)) { all_off = false; break; }
            if (all_off) for (int h : kv.second.hvdcs) if (!hvdc_off(r, h)) { all_off = false; break; }
            if (all_off) {
                row_pv_to_pq[static_cast<size_t>(r)].push_back(kv.first);
                req.insert(kv.first);
            }
        }
        for (int g = 0; g < n_gen; ++g)
            if (gen_off(r, g) && gd.status[g] && gd.slack_participant[g])
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

// Why the batch cannot take this generator out of a row (empty: it can): a remote
// controller, or a bus a control group holds -- remote generator, SVC, HVDC station --
// lives in the VoltageControl extension, whose own Jacobian rows and columns nothing here
// reserves or releases.
inline std::string gen_unsupported(const GenContingencyData& gd, int g)
{
    if (gd.remote_vreg[g])
        return "disconnects generator " + std::to_string(g) + ", which regulates the voltage of a remote bus";
    if (gd.on_group_bus[g])
        return "disconnects generator " + std::to_string(g) + ", which stands on a bus a control group holds";
    return std::string();
}

// Per-row normalised slack weights ([n_rows * n_slack], ORIGINAL order) without
// the participating generators each row takes out (row_slack_off), on the base
// participant layout (`slack_bus`: the bus of each slack index). The storage
// units taking part in the slack keep their share on every row (no row
// disconnects one). Empty when no row takes anything out. `ref_bus` takes the
// whole share of a row that leaves no participant.
template <class StoOff>
std::vector<cuda_real_type> row_slack_weights(const GenContingencyData& gd,
                                                     const std::vector<std::vector<int>>& row_slack_off,
                                                     int n_rows, int n_bus,
                                                     const std::vector<int>& slack_bus,
                                                     const std::vector<cuda_real_type>& base_w,
                                                     int ref_bus, const std::string& who, StoOff sto_off)
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
    bool any = false;
    for (const auto& v : row_slack_off) if (!v.empty()) { any = true; break; }
    for (int r = 0; r < n_rows && !any; ++r) any = row_sto_off(r);
    if (!any) return out;

    out.assign(static_cast<size_t>(n_rows) * n_slack, static_cast<cuda_real_type>(0.));
    std::vector<double> w(static_cast<size_t>(n_bus));
    std::vector<char>   off(static_cast<size_t>(gd.n_gen));
    for (int r = 0; r < n_rows; ++r) {
        cuda_real_type* row = out.data() + static_cast<ptrdiff_t>(r) * n_slack;
        const bool sto_out = row_sto_off(r);
        if (row_slack_off[static_cast<size_t>(r)].empty() && !sto_out) {
            std::copy(base_w.begin(), base_w.end(), row);
            continue;
        }
        std::fill(off.begin(), off.end(), 0);
        for (int g : row_slack_off[static_cast<size_t>(r)]) off[static_cast<size_t>(g)] = 1;
        std::fill(w.begin(), w.end(), 0.0);
        double sum = 0.0;
        const std::vector<double>& sto_w = gd.storage_slack_weight_bus;
        if (!sto_out) {
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
            if (!gd.status[g] || !gd.slack_participant[g] || off[static_cast<size_t>(g)]) continue;
            const int b = gd.bus[g];
            if (b < 0) continue;
            w[static_cast<size_t>(b)] += gd.slack_weight[g];
            sum += gd.slack_weight[g];
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
        // survivors are a subset of the base participants); the degenerate
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
