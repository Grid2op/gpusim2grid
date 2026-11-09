// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// element_trip_data.hpp — what tripping a shunt, a load or a generator changes
// (lightsim2grid's add_contingency / set_contingency_shunts / _loads parity)
// =============================================================================
//
// Host-only snapshot read off a solved lightsim2grid grid by the bridge
// (extract_element_trip_data, ls2g_bridge.cpp):
//
//   - a shunt is an admittance: tripping it takes its diagonal Ybus coefficient
//     (y, per unit, the very value lightsim2grid's ShuntContainer stamps) out,
//     one more triplet of the row's Ybus patch;
//   - a load is an injection: tripping it gives back what it draws, +(p + jq)/sn
//     at its bus, to the row's Sbus;
//   - a generator too: -(p + jq)/sn at its bus, its reactive setpoint only when
//     it does not regulate voltage (GenContingencyData::vreg_on / target_q_mvar,
//     its PV -> PQ side and slack re-weighting being GenContingencyData's).
//
// An element off in the base grid (bus -1) is in neither the matrix nor the
// injection: tripping it changes nothing. All bus ids are AC-solver numbering.
// =============================================================================

#ifndef ELEMENT_TRIP_DATA_HPP
#define ELEMENT_TRIP_DATA_HPP

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ElementTripData {
    double sn_mva = 1.;

    // shunts
    int n_shunt = 0;
    std::vector<int>    shunt_bus;          // [n_shunt] solver bus, -1 when off
    std::vector<double> shunt_y_re;         // [n_shunt] the diagonal coefficient, pu
    std::vector<double> shunt_y_im;
    std::vector<double> shunt_p_mw;         // [n_shunt] its active power at 1 pu (the slack pre-pass)
    std::vector<char>   shunt_regulating;   // [n_shunt] its sections regulate a voltage (ShuntVoltageControl)

    // loads
    int n_load = 0;
    std::vector<int>    load_bus;           // [n_load] solver bus, -1 when off
    std::vector<double> load_p_mw;          // [n_load]
    std::vector<double> load_q_mvar;        // [n_load]

    // generators (their bus, regulation and reactive setpoint are GenContingencyData's)
    int n_gen = 0;
    std::vector<double> gen_target_p_mw;    // [n_gen]

    // static generators: an injection only
    int n_sgen = 0;
    std::vector<int>    sgen_bus;           // [n_sgen] solver bus, -1 when off
    std::vector<double> sgen_p_mw;          // [n_sgen]
    std::vector<double> sgen_q_mvar;        // [n_sgen]

    // storage units, load convention (what StorageContainer::_fillSbus takes)
    int n_storage = 0;
    std::vector<int>    storage_bus;        // [n_storage] solver bus, -1 when off
    std::vector<double> storage_p_mw;       // [n_storage]
    std::vector<double> storage_q_mvar;     // [n_storage] 0 for one regulating its bus
    std::vector<char>   storage_on_group;   // [n_storage] regulates a bus a control group holds
    std::vector<char>   storage_checked;    // [n_storage] the physical checks read it (it regulates or shares the slack)

    // static var compensators
    int n_svc = 0;
    std::vector<int>    svc_bus;            // [n_svc] solver bus, -1 when off
    std::vector<double> svc_b0_pu;          // [n_svc] the diagonal susceptance it stamps, whatever its mode
    std::vector<double> svc_q_mvar;         // [n_svc] the reactive injection of one in REACTIVE_POWER mode, else 0
    std::vector<char>   svc_voltage;        // [n_svc] in voltage mode: a controller of its control group
    std::vector<char>   svc_standby;        // [n_svc] carries a standby automaton
    std::vector<char>   svc_frozen;         // [n_svc] an outer loop froze it at a reactive limit

    // HVDC lines: both converter stations, per line [2 * n_hvdc], side 1 then side 2
    int n_hvdc = 0;
    std::vector<char>   hvdc_on;            // [n_hvdc]
    std::vector<char>   hvdc_droop;         // [n_hvdc] in AC emulation: its active power is in the Newton
    std::vector<char>   hvdc_regulating;    // [n_hvdc] one of its stations regulates a voltage
    std::vector<char>   hvdc_on_group;      // [n_hvdc] one of its stations regulates a bus a control group holds
    std::vector<int>    hvdc_station_bus;   // solver bus, -1 when off or outside the solved grid
    std::vector<double> hvdc_station_p_mw;  // generator convention
    std::vector<double> hvdc_station_q_mvar;// 0 for one regulating its bus

    bool empty() const {
        return n_shunt == 0 && n_load == 0 && n_gen == 0 && n_sgen == 0 && n_storage == 0 && n_svc == 0 && n_hvdc == 0;
    }
};

// The elements one row trips beyond its branches, shunts, loads and generators
// (lightsim2grid PR #292), each list sorted.
struct MoreTrips {
    std::vector<int> sgens, storages, svcs, hvdcs;
    bool any() const { return !sgens.empty() || !storages.empty() || !svcs.empty() || !hvdcs.empty(); }
};

namespace element_trips {

inline void check_ids(const std::vector<int>& ids, int n, const char* kind, const char* who)
{
    for (int id : ids)
        if (id < 0 || id >= n)
            throw std::runtime_error(std::string(who) + ": no " + kind + " with id " +
                                     std::to_string(id) + " (the grid has " + std::to_string(n) + ")");
}

// One (bus, y_re, y_im) triplet per tripped shunt that is in the base matrix:
// the diagonal coefficient to subtract.
struct DiagTriplet { int bus; double re; double im; };
inline std::vector<DiagTriplet> shunt_triplets(const ElementTripData& d, const std::vector<int>& shunt_ids)
{
    std::vector<DiagTriplet> out;
    for (int s : shunt_ids) {
        const int b = d.shunt_bus[static_cast<size_t>(s)];
        if (b < 0) continue;
        out.push_back({b, d.shunt_y_re[static_cast<size_t>(s)], d.shunt_y_im[static_cast<size_t>(s)]});
    }
    return out;
}

// One (bus, 0, b0) triplet per tripped SVC with a b0 in the base matrix: the very
// coefficient SvcContainer::_fillYbus stamps, whatever its mode.
inline std::vector<DiagTriplet> svc_triplets(const ElementTripData& d, const std::vector<int>& svc_ids)
{
    std::vector<DiagTriplet> out;
    for (int s : svc_ids) {
        const int b = d.svc_bus[static_cast<size_t>(s)];
        const double b0 = d.svc_b0_pu[static_cast<size_t>(s)];
        if (b >= 0 && b0 != 0.) out.push_back({b, 0., b0});
    }
    return out;
}

inline void check_ids(const MoreTrips& t, const ElementTripData& d, const char* who)
{
    check_ids(t.sgens, d.n_sgen, "static generator", who);
    check_ids(t.storages, d.n_storage, "storage unit", who);
    check_ids(t.svcs, d.n_svc, "static var compensator", who);
    check_ids(t.hvdcs, d.n_hvdc, "hvdc line", who);
}

// The entries (positions in `hvdc_id`, the droop lines of the Newton) these lines are.
inline std::vector<int> droop_entries(const std::vector<int>& hvdc_id, const std::vector<int>& hvdcs)
{
    std::vector<int> out;
    for (int h : hvdcs) {
        const auto it = std::find(hvdc_id.begin(), hvdc_id.end(), h);
        if (it != hvdc_id.end()) out.push_back(static_cast<int>(it - hvdc_id.begin()));
    }
    return out;
}

// Why the batch cannot take these elements out of a row (empty: it can).
inline std::string more_unsupported(const ElementTripData& d, const MoreTrips& t, bool outer, bool physical)
{
    for (int s : t.storages) {
        const size_t k = static_cast<size_t>(s);
        if (d.storage_bus[k] < 0) continue;
        if (d.storage_on_group[k])
            return "disconnects storage unit " + std::to_string(s) + ", which regulates a bus a control group holds";
        if (physical && d.storage_checked[k])
            return "disconnects storage unit " + std::to_string(s) + ": the physical checks cannot leave it out of a row yet";
    }
    for (int s : t.svcs) {
        const size_t k = static_cast<size_t>(s);
        if (d.svc_bus[k] < 0) continue;
        if (d.svc_standby[k]) return "disconnects static var compensator " + std::to_string(s) + ", which carries a standby automaton";
        if (d.svc_frozen[k]) return "disconnects static var compensator " + std::to_string(s) + ", which an outer loop froze at a reactive limit";
        if (d.svc_voltage[k]) return "disconnects static var compensator " + std::to_string(s) + ", which regulates a voltage";
    }
    for (int h : t.hvdcs) {
        const size_t k = static_cast<size_t>(h);
        if (!d.hvdc_on[k]) continue;
        if (physical && d.hvdc_droop[k])
            return "disconnects hvdc line " + std::to_string(h) + ", which runs in AC emulation: the physical checks cannot leave it out of a row yet";
        if (d.hvdc_on_group[k]) return "disconnects hvdc line " + std::to_string(h) + ", a station of which stands in a control group";
        if (outer && d.hvdc_regulating[k])
            return "disconnects hvdc line " + std::to_string(h) + ", a station of which regulates a voltage: the outer loops cannot leave it out yet";
        if (physical && d.hvdc_regulating[k])
            return "disconnects hvdc line " + std::to_string(h) + ", a station of which regulates a voltage: the physical checks cannot leave it out of a row yet";
    }
    return std::string();
}

// What the tripped loads and generators add to the row's Sbus, (bus, dS pu),
// one entry per bus, sorted. `gen_vreg_on` / `gen_target_q_mvar` /
// `gen_bus`: GenContingencyData's (bus -1: off).
inline std::vector<std::pair<int, std::complex<double>>> sbus_delta(
    const ElementTripData& d, const std::vector<int>& load_ids, const std::vector<int>& gen_ids,
    const std::vector<int>& gen_bus, const std::vector<char>& gen_vreg_on,
    const std::vector<double>& gen_target_q_mvar, const MoreTrips* more = nullptr)
{
    std::vector<std::pair<int, std::complex<double>>> out;
    for (int l : load_ids) {
        const int b = d.load_bus[static_cast<size_t>(l)];
        if (b < 0) continue;
        out.emplace_back(b, std::complex<double>(d.load_p_mw[static_cast<size_t>(l)],
                                                 d.load_q_mvar[static_cast<size_t>(l)]) / d.sn_mva);
    }
    for (int g : gen_ids) {
        const int b = gen_bus[static_cast<size_t>(g)];
        if (b < 0) continue;
        const double q = gen_vreg_on[static_cast<size_t>(g)] ? 0. : gen_target_q_mvar[static_cast<size_t>(g)];
        out.emplace_back(b, -std::complex<double>(d.gen_target_p_mw[static_cast<size_t>(g)], q) / d.sn_mva);
    }
    if (more != nullptr) {
        for (int s : more->sgens) {
            const size_t k = static_cast<size_t>(s);
            if (d.sgen_bus[k] >= 0)
                out.emplace_back(d.sgen_bus[k], -std::complex<double>(d.sgen_p_mw[k], d.sgen_q_mvar[k]) / d.sn_mva);
        }
        for (int s : more->storages) {   // load convention: given back
            const size_t k = static_cast<size_t>(s);
            if (d.storage_bus[k] >= 0)
                out.emplace_back(d.storage_bus[k], std::complex<double>(d.storage_p_mw[k], d.storage_q_mvar[k]) / d.sn_mva);
        }
        for (int s : more->svcs) {
            const size_t k = static_cast<size_t>(s);
            if (d.svc_bus[k] >= 0 && d.svc_q_mvar[k] != 0.)
                out.emplace_back(d.svc_bus[k], -std::complex<double>(0., d.svc_q_mvar[k]) / d.sn_mva);
        }
        for (int h : more->hvdcs) {
            if (!d.hvdc_on[static_cast<size_t>(h)]) continue;
            // in AC emulation the line's active power is the Newton's, not in Sbus
            const bool p_in_sbus = !d.hvdc_droop[static_cast<size_t>(h)];
            for (size_t k = 2 * static_cast<size_t>(h); k < 2 * static_cast<size_t>(h) + 2; ++k)
                if (d.hvdc_station_bus[k] >= 0)
                    out.emplace_back(d.hvdc_station_bus[k],
                                     -std::complex<double>(p_in_sbus ? d.hvdc_station_p_mw[k] : 0.,
                                                           d.hvdc_station_q_mvar[k]) / d.sn_mva);
        }
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<int, std::complex<double>>& a, const std::pair<int, std::complex<double>>& b) {
                  return a.first < b.first; });
    std::vector<std::pair<int, std::complex<double>>> merged;
    for (const auto& e : out) {
        if (!merged.empty() && merged.back().first == e.first) merged.back().second += e.second;
        else merged.push_back(e);
    }
    return merged;
}

// The active power (MW, generator convention) the tripped loads and the tripped
// shunts NOT on a masked bus take out of the balance: the slack pre-pass' term
// for a row's element trips, its island sum being read off the row's OWN Sbus
// (base + sbus_delta, where a tripped load or generator nets to 0) and
// subtracting every masked bus' shunts, and the tripped generators being its
// own term (see slack_redistribution::lost_mw).
template <class BusMasked>
double trip_lost_mw(const ElementTripData& d, const std::vector<int>& load_ids,
                    const std::vector<int>& shunt_ids, BusMasked bus_masked)
{
    double lost = 0.;
    for (int l : load_ids) {
        const int b = d.load_bus[static_cast<size_t>(l)];
        if (b < 0) continue;
        lost -= d.load_p_mw[static_cast<size_t>(l)];
    }
    for (int s : shunt_ids) {
        const int b = d.shunt_bus[static_cast<size_t>(s)];
        if (b < 0 || bus_masked(b)) continue;
        lost -= d.shunt_p_mw[static_cast<size_t>(s)];
    }
    return lost;
}

// The active power (MW, generator convention) these elements take out of the balance,
// each counted once where it stood in the base grid: lightsim2grid's term (e).
inline double more_lost_mw(const ElementTripData& d, const MoreTrips& t)
{
    double lost = 0.;
    for (int s : t.sgens) if (d.sgen_bus[static_cast<size_t>(s)] >= 0) lost += d.sgen_p_mw[static_cast<size_t>(s)];
    for (int s : t.storages) if (d.storage_bus[static_cast<size_t>(s)] >= 0) lost -= d.storage_p_mw[static_cast<size_t>(s)];
    for (int h : t.hvdcs) {
        if (!d.hvdc_on[static_cast<size_t>(h)]) continue;
        for (size_t k = 2 * static_cast<size_t>(h); k < 2 * static_cast<size_t>(h) + 2; ++k)
            if (d.hvdc_station_bus[k] >= 0) lost += d.hvdc_station_p_mw[k];
    }
    return lost;
}

}  // namespace element_trips

#endif  // ELEMENT_TRIP_DATA_HPP
