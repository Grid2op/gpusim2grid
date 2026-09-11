// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// gen_contingency_data.hpp — per-generator metadata for ScenarioSweepSession::
// set_contingency_gens (lightsim2grid PR #193 parity)
// =============================================================================
//
// Host-only snapshot read off a solved lightsim2grid grid by the bridge
// (extract_gen_contingency_data, ls2g_bridge.cpp) -- everything the session
// needs to turn a (n_scenarios x n_gen) "disconnect this generator for this
// row" mask into (a) the buses that lose their LAST local voltage controller
// in some row (and therefore need a reserved Vm column + Q equation, see
// ledger_extend.hpp), (b) per-row PV pins, (c) per-row distributed-slack
// weights re-derived without the disconnected participants. The injection
// side (the generator's P / target Q leaving Sbus) is done in Python, where
// Sbus is assembled (see _ls2g_utils.build_bus_injections).
//
// All bus ids are AC-solver numbering; -1 for a generator that is off or sits
// on a bus the solver dropped.
// =============================================================================

#ifndef GEN_CONTINGENCY_DATA_HPP
#define GEN_CONTINGENCY_DATA_HPP

#include <vector>

struct GenContingencyData {
    int n_gen = 0;
    std::vector<int>    bus;               // [n_gen] solver bus, -1 when off / unmapped
    std::vector<char>   status;            // [n_gen] connected
    std::vector<char>   local_vreg;        // [n_gen] pins the voltage of its OWN bus (PV)
    std::vector<char>   remote_vreg;       // [n_gen] regulates ANOTHER bus (VoltageControl)
    std::vector<char>   on_group_bus;      // [n_gen] its bus is held by a control group
    std::vector<char>   slack_participant; // [n_gen] distributed-slack participant (weight != 0)
    std::vector<double> slack_weight;      // [n_gen] raw (un-normalised) slack weight
    std::vector<char>   vreg_on;           // [n_gen] voltage regulator on (its Q is NOT in Sbus)
    std::vector<double> target_q_mvar;     // [n_gen] reactive setpoint of a non-regulating one

    bool empty() const { return n_gen == 0; }
};

#endif  // GEN_CONTINGENCY_DATA_HPP
