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

    bool empty() const { return n_shunt == 0 && n_load == 0 && n_gen == 0; }
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

// What the tripped loads and generators add to the row's Sbus, (bus, dS pu),
// one entry per bus, sorted. `gen_vreg_on` / `gen_target_q_mvar` /
// `gen_bus`: GenContingencyData's (bus -1: off).
inline std::vector<std::pair<int, std::complex<double>>> sbus_delta(
    const ElementTripData& d, const std::vector<int>& load_ids, const std::vector<int>& gen_ids,
    const std::vector<int>& gen_bus, const std::vector<char>& gen_vreg_on,
    const std::vector<double>& gen_target_q_mvar)
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

}  // namespace element_trips

#endif  // ELEMENT_TRIP_DATA_HPP
