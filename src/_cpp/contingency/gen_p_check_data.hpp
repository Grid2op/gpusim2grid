// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef GEN_P_CHECK_DATA_HPP
#define GEN_P_CHECK_DATA_HPP

// =============================================================================
// contingency/gen_p_check_data.hpp — host-side "plan" of the per-machine
// active-power check of the distributed slack (compute_physical_violations,
// lightsim2grid's GenPCheck.hpp parity: generators AND storage units)
// =============================================================================
//
// A flattened, device-friendly copy of lightsim2grid's gen_p_check::GenPPlan
// (batch_algorithm/GenPCheck.hpp): which machines carrying the distributed
// slack can be reported at all, and everything a row needs to work out the
// active power each one produced. Built ONCE per session -- by the bridge
// through lightsim2grid's own build_gen_p_plan (extract_gen_p_plan_from_
// lsgrid, ls2g_bridge.cpp), or handed in as raw arrays by a caller in
// array/tuple mode (set_gen_p_capability) -- and uploaded to the driver, which
// evaluates every row on the device (check_gen_p_violations_kernel).
//
// WHAT IS CHECKED. lightsim2grid solves the distributed slack INSIDE the
// Newton system (MultiSlack): the absorbed power is an unknown, shared out by
// fixed participation factors that know nothing about limits. A participating
// machine's converged active power is therefore
//
//     p = target_p + its share of what its bus had to make up
//
// and can land beyond max_p_mw or below min_p_mw -- the condition
// OpenLoadFlow's DistributedSlack outer loop acts on. Per MACHINE (the active
// split is the caller's own participation factors, not a convention), for
// generators and storage units alike: the two families take a share under
// the same rule, so the share of a machine is a fraction of the RAW
// participation of its whole bus, both families included -- which is why the
// plan carries every participant, limits or not, and not only the ones that
// can be reported.
//
// Two lists (mirroring upstream's `gens` / `participants`):
//
//   * entries [n_entries] -- the machines that CAN be reported: connected,
//     participating with a nonzero weight, given at least one finite limit.
//     el_type is 5 (GENERATOR) or 6 (STORAGE), el_id the container id (the
//     column of a ScenarioSweep generator-contingency mask for a generator),
//     bus_solver the bus it stands on, slack_weight its own RAW factor,
//     min_p_mw / max_p_mw its limits (NaN = none on that side) and
//     target_p_mw the grid's own active set-point -- all in MW and in the
//     GENERATOR convention (a storage unit's load-convention target is
//     negated on the way in, upstream's `target_sign`).
//   * participants [n_part] -- every machine that takes a share, limits or
//     not, with the same (el_type, el_id, bus_solver, slack_weight) fields.
//
// The device evaluates, per row, the raw participation of a bus as the sum of
// the weights of the participants that stand on it and are still live in
// that row (not on a masked bus, not a generator the row disconnected). That
// is exactly upstream's `w_norm(bus) * total_raw_w` (the normalised per-bus
// weight the solver was given, un-normalised again by the row's total), since
// the row's normalised weights ARE those raw sums over the live participants,
// renormalised -- and it needs no per-row weight vector on the device.
//
// The active power a bus' slack machines produced on top of their targets is
// the RAW active residual real(V . conj(Ybus . V) - Sbus) of the slot's
// patched Ybus, plus the angle-droop hvdc flows leaving that bus (a station's
// published injection is -p_flow, so it must not be charged to the slack
// machines): what lightsim2grid's `mis_bus.real() - slack_absorbed * w`
// reconstructs. No MultiSlack state is needed.
//
// Deliberately NOT carried, compared to lightsim2grid's plan: the element
// names (gpusim2grid's records carry none).
//
// CUDA-free on purpose: included by the session headers, which the host
// compiler builds into ls2g_bridge.cpp.
// =============================================================================

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <Eigen/Core>

#include "../dtypes.hpp"

// (n_rows x n_cols) row-major real matrix, the shape every per-row input of the
// sessions has (set_injections, set_gen_v, set_gen_p_targets)
using RealMatRM = Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct GenPPlanData {
    // ---- the machines that can be reported --------------------------------
    int             n_entries = 0;
    Eigen::VectorXi el_type;        // [n_entries] 5 = GENERATOR, 6 = STORAGE
    Eigen::VectorXi el_id;          // [n_entries] container id of that family
    Eigen::VectorXi bus_solver;     // [n_entries]
    RealVect        slack_weight;   // [n_entries] raw participation factor
    RealVect        min_p_mw;       // [n_entries] NaN = no lower limit
    RealVect        max_p_mw;       // [n_entries] NaN = no upper limit
    RealVect        target_p_mw;    // [n_entries] base set-point, generator convention
    // ---- ... and everything that takes a share ----------------------------
    int             n_part = 0;
    Eigen::VectorXi part_el_type;   // [n_part]
    Eigen::VectorXi part_el_id;     // [n_part]
    Eigen::VectorXi part_bus_solver;// [n_part]
    RealVect        part_weight;    // [n_part] raw participation factor
    double          sn_mva = 100.0;

    bool empty() const { return n_entries == 0; }

    // Structural checks only (sizes / index ranges); throws std::runtime_error.
    void validate(int n_bus) const {
        std::ostringstream exc_;
        auto fail = [&](const std::string& what) {
            exc_ << "GenPPlanData: " << what;
            throw std::runtime_error(exc_.str());
        };
        if (n_entries < 0) fail("n_entries must be >= 0");
        if (n_part < 0) fail("n_part must be >= 0");
        auto need = [&](long long got, long long want, const char* name) {
            if (got != want) {
                std::ostringstream m;
                m << name << " has size " << got << ", expected " << want;
                fail(m.str());
            }
        };
        need(el_type.size(),         n_entries, "el_type");
        need(el_id.size(),           n_entries, "el_id");
        need(bus_solver.size(),      n_entries, "bus_solver");
        need(slack_weight.size(),    n_entries, "slack_weight");
        need(min_p_mw.size(),        n_entries, "min_p_mw");
        need(max_p_mw.size(),        n_entries, "max_p_mw");
        need(target_p_mw.size(),     n_entries, "target_p_mw");
        need(part_el_type.size(),    n_part,    "part_el_type");
        need(part_el_id.size(),      n_part,    "part_el_id");
        need(part_bus_solver.size(), n_part,    "part_bus_solver");
        need(part_weight.size(),     n_part,    "part_weight");
        auto check_machine = [&](int type, int id, int bus, const char* who, int k) {
            std::ostringstream m;
            if (type != 5 && type != 6) {
                m << who << "[" << k << "]: el_type must be 5 (GENERATOR) or 6 (STORAGE), got " << type;
                fail(m.str());
            }
            if (id < 0) {
                m << who << "[" << k << "]: el_id must be >= 0";
                fail(m.str());
            }
            if (bus < 0 || bus >= n_bus) {
                m << who << "[" << k << "]: bus_solver = " << bus << " is outside [0, n_bus=" << n_bus << ")";
                fail(m.str());
            }
        };
        for (int k = 0; k < n_entries; ++k) {
            check_machine(el_type(k), el_id(k), bus_solver(k), "entries", k);
            if (!std::isfinite(static_cast<double>(slack_weight(k))))
                fail("slack_weight must be finite");
            if (!std::isfinite(static_cast<double>(target_p_mw(k))))
                fail("target_p_mw must be finite");
        }
        for (int p = 0; p < n_part; ++p) {
            check_machine(part_el_type(p), part_el_id(p), part_bus_solver(p), "participants", p);
            if (!std::isfinite(static_cast<double>(part_weight(p))))
                fail("part_weight must be finite");
        }
        if (!(sn_mva > 0.)) fail("sn_mva must be > 0");
    }
};

#endif  // GEN_P_CHECK_DATA_HPP
