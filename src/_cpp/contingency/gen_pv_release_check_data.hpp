// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GEN_PV_RELEASE_CHECK_DATA_HPP
#define GEN_PV_RELEASE_CHECK_DATA_HPP

// =============================================================================
// contingency/gen_pv_release_check_data.hpp — host-side "plan" of the PQ -> PV
// release check (compute_physical_violations, lightsim2grid's
// GenPvReleaseCheck.hpp parity, PR #216)
// =============================================================================
//
// OpenLoadFlow's ReactiveLimits outer loop has two directions. The PV -> PQ
// one (a regulating machine asked for more reactive power than it has) is the
// bus reactive-capability check (LOW_Q / HIGH_Q). This is its mirror image: a
// PQ generator the caller flagged as pinned at a reactive limit by an outer
// loop (lightsim2grid's per-generator can_be_pv, LSGrid::set_gen_can_be_pv),
// sitting at its min_q (resp. max_q), whose REGULATED bus sits below (resp.
// above) the target it would hold, absorbs (resp. produces) too much for that
// target: the loop would let it regulate again. Reported on the GENERATOR as
// LOW_VOLTAGE_AT_MIN_Q (9) / HIGH_VOLTAGE_AT_MAX_Q (10), value the regulated
// bus' voltage and limit the target, both in kV of that bus. Nothing is
// switched back.
//
// Which machines can be reported, and at which limit each sits, is decided
// ONCE (no batch axis varies a reactive set-point) -- by the bridge through
// lightsim2grid's own build_gen_pv_release_plan (extract_gen_pv_release_plan_
// from_lsgrid, ls2g_bridge.cpp: flagged, connected, not regulating, a reactive
// range of at least 1 MVAr, a target_q within tol_mva of one of its limits),
// or handed in as raw arrays by a caller in array/tuple mode. Per row only the
// regulated voltage, the row's own target (a sweep's gen_v, see
// set_gen_pv_release_targets), the mask and a generator contingency vary; the
// device evaluates every row (check_gen_pv_release_violations_kernel).
//
// One entry per candidate machine:
//   gen_id         the generator id (the column of a generator-contingency mask)
//   reg_bus_solver the bus it would regulate, SOLVER numbering
//   gen_bus_solver its own bus, SOLVER numbering (a row that masks it strands
//                  the machine: it releases nothing, like a disconnected one)
//   at_min         1 = pinned at min_q (reported when BELOW the target),
//                  0 = at max_q (reported when ABOVE)
//   target_vm_pu   the grid's own target (a row may hand its own)
//   vn_kv          nominal voltage of the regulated bus (value / limit in kV)
//
//   el_type        OPTIONAL (empty = every entry a GENERATOR): the element the
//                  entry is reported on, 5 = GENERATOR, 7 = SVC
//                  (ViolationElementType); gen_id is then its svc id.
//   standby        OPTIONAL (empty = none): 1 for an entry of the standby SVC
//                  check below, 0 for a release.
//
// An SVC entry is either of two lightsim2grid checks, neither varied by a row
// (a row's own targets are ignored for them) nor disconnected by a contingency
// (the generator mask does not apply):
//   - the release of an SVC an outer loop froze at a reactive limit
//     (LSGrid::set_svc_can_be_pv, lightsim2grid's GenPvReleaseCheck.hpp): the
//     generators' test verbatim, one entry, reported as LOW_VOLTAGE_AT_MIN_Q
//     (9) / HIGH_VOLTAGE_AT_MAX_Q (10) on the SVC;
//   - the switch on of an idle SVC flagged as carrying a standby automaton
//     (LSGrid::set_svc_standby, SvcStandbyCheck.hpp), which OpenLoadFlow's
//     MonitoringVoltageOuterLoop switches to voltage control once the voltage
//     of the bus it regulates leaves the automaton's [low, high] thresholds:
//     TWO entries with standby = 1, at_min = 1 with target_vm_pu the low
//     threshold and at_min = 0 with the high one -- the release test again --
//     reported as LOW_VOLTAGE_SVC_STANDBY (11) / HIGH_VOLTAGE_SVC_STANDBY (12).
//
// Deliberately NOT carried, compared to lightsim2grid's plan: the grid bus id
// of the regulated bus and the element names (gpusim2grid's records carry
// neither).
//
// CUDA-free on purpose: included by the session headers, which the host
// compiler builds into ls2g_bridge.cpp.
// =============================================================================

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <Eigen/Core>

#include "../dtypes.hpp"

struct GenPvReleasePlanData {
    // ViolationElementType codes of the entries (limit_violation_types.hpp)
    static constexpr int EL_GENERATOR = 5;
    static constexpr int EL_SVC       = 7;

    int             n_entries = 0;
    Eigen::VectorXi gen_id;          // [n_entries] generator id, or svc id for an SVC entry
    Eigen::VectorXi reg_bus_solver;  // [n_entries]
    Eigen::VectorXi gen_bus_solver;  // [n_entries]
    Eigen::VectorXi at_min;          // [n_entries] 1 = at min_q, 0 = at max_q
    RealVect        target_vm_pu;    // [n_entries] the grid's own target
    RealVect        vn_kv;           // [n_entries] nominal kV of the regulated bus
    Eigen::VectorXi el_type;         // [n_entries] EL_GENERATOR / EL_SVC, or empty = all generators
    Eigen::VectorXi standby;         // [n_entries] 1 = a standby SVC check entry, or empty = none

    bool empty() const { return n_entries == 0; }
    bool is_svc(int k) const { return el_type.size() != 0 && el_type(k) == EL_SVC; }
    bool is_standby(int k) const { return standby.size() != 0 && standby(k) != 0; }

    // Structural checks only (sizes / index ranges); throws std::runtime_error.
    void validate(int n_bus) const {
        auto fail = [&](const std::string& what) {
            throw std::runtime_error("GenPvReleasePlanData: " + what);
        };
        if (n_entries < 0) fail("n_entries must be >= 0");
        auto need = [&](long long got, const char* name) {
            if (got != n_entries) {
                std::ostringstream m;
                m << name << " has size " << got << ", expected " << n_entries;
                fail(m.str());
            }
        };
        need(gen_id.size(),         "gen_id");
        need(reg_bus_solver.size(), "reg_bus_solver");
        need(gen_bus_solver.size(), "gen_bus_solver");
        need(at_min.size(),         "at_min");
        need(target_vm_pu.size(),   "target_vm_pu");
        need(vn_kv.size(),          "vn_kv");
        if (el_type.size() != 0) need(el_type.size(), "el_type");
        if (standby.size() != 0) need(standby.size(), "standby");
        for (int k = 0; k < n_entries; ++k) {
            std::ostringstream m;
            if (gen_id(k) < 0) {
                m << "gen_id[" << k << "] must be >= 0";
                fail(m.str());
            }
            if (reg_bus_solver(k) < 0 || reg_bus_solver(k) >= n_bus ||
                gen_bus_solver(k) < 0 || gen_bus_solver(k) >= n_bus) {
                m << "entry " << k << ": a bus is outside [0, n_bus=" << n_bus << ")";
                fail(m.str());
            }
            if (at_min(k) != 0 && at_min(k) != 1) {
                m << "at_min[" << k << "] must be 0 or 1";
                fail(m.str());
            }
            if (el_type.size() != 0 && el_type(k) != EL_GENERATOR && el_type(k) != EL_SVC) {
                m << "el_type[" << k << "] must be " << EL_GENERATOR << " (GENERATOR) or "
                  << EL_SVC << " (SVC)";
                fail(m.str());
            }
            if (standby.size() != 0 && standby(k) != 0 && (standby(k) != 1 || !is_svc(k))) {
                m << "standby[" << k << "] must be 0, or 1 on an SVC entry";
                fail(m.str());
            }
            if (!std::isfinite(static_cast<double>(target_vm_pu(k))) ||
                !(static_cast<double>(target_vm_pu(k)) > 0.)) {
                m << "target_vm_pu[" << k << "] must be finite and > 0";
                fail(m.str());
            }
            if (!std::isfinite(static_cast<double>(vn_kv(k))) ||
                !(static_cast<double>(vn_kv(k)) > 0.)) {
                m << "vn_kv[" << k << "] must be finite and > 0";
                fail(m.str());
            }
        }
    }
};

#endif  // GEN_PV_RELEASE_CHECK_DATA_HPP
