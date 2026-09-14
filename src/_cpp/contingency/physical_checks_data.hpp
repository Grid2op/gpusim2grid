// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PHYSICAL_CHECKS_DATA_HPP
#define PHYSICAL_CHECKS_DATA_HPP

// =============================================================================
// contingency/physical_checks_data.hpp — the two opt-in post-solve "physical"
// checks, as the three batch sessions expose them
// =============================================================================
//
//   - per-bus reactive capability (lightsim2grid PR #206 parity; LOW_Q /
//     HIGH_Q on a BUS)
//   - droop HVDC P-saturation (OpenLoadFlow's HvdcAcEmulationLimits;
//     HVDC_P_SATURATION on an HVDC)
//
// Both say the same kind of thing (ViolationCategory::PHYSICAL): the converged
// solution assumes a control that the equipment cannot actually hold. Neither
// enforces anything; they only report. Hence ONE opt-in for the category,
// compute_physical_violations (lightsim2grid's name), one tolerance in MVA and
// one capacity, in one configuration object (PhysicalChecksConfig, held by
// each session and bound to Python once); the two result shapes are merged
// into a single PHYSICAL-only list per row on the Python side. The work itself is in BatchPfDriver (set_bus_q_check /
// set_hvdc_p_check + the two kernels of violation_kernels.cu) and the session
// glue in physical_checks_impl.cuh.
//
// CUDA-free on purpose: included by the session headers, which the host
// compiler builds into ls2g_bridge.cpp.
// =============================================================================

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <Eigen/Core>

#include "../dtypes.hpp"
#include "bus_q_check_data.hpp"

// Per-row records of the bus reactive-capability check, flat SoA:
//   bus_id/type/value/limit : [n_rows * capacity] (row r owns [r*capacity, r*capacity + count[r]))
//   count                   : [n_rows]; -1 = never simulated (compacted out), else 0..capacity
//   truncated               : [n_rows]; 1 when more than `capacity` buses violated
// type is 5 (LOW_Q) or 6 (HIGH_Q); bus_id the SOLVER bus id; value/limit MVAr.
// For the base ("n") case n_rows == 1.
struct BusQViolationsResult {
    Eigen::VectorXi bus_id, type;
    RealVect        value, limit;
    Eigen::VectorXi count, truncated;
    int             capacity = 0;
};

// Same layout for the droop P-saturation check: hvdc_id is the GRID hvdc id,
// side 1 (saturates 1->2) or 2 (saturates 2->1), value/limit MW. Every record
// is element type HVDC (4), violation type HVDC_P_SATURATION (7).
struct HvdcPViolationsResult {
    Eigen::VectorXi hvdc_id, side;
    RealVect        value, limit;
    Eigen::VectorXi count, truncated;
    int             capacity = 0;
};

struct PhysicalChecksConfig {
    // ONE opt-in for the whole category (lightsim2grid's compute_physical_
    // violations): every record it produces has ViolationCategory::PHYSICAL,
    // whatever the element -- a bus' reactive capability today, an hvdc line's
    // droop saturation, the next physical limit tomorrow with no new flag.
    bool   compute_physical_violations = false;
    // slack on every comparison, in MVA (MVAr for the reactive check, MW for the
    // active one); upstream default
    double physical_violation_tol_mva = 1e-4;
    // records kept per row AND per check (bounds each output at n_rows * capacity)
    int    physical_violation_capacity = 16;
    // the routing the reactive check needs (set_bus_q_capability); the hvdc check
    // needs nothing beyond the base state
    BusQPlanData bus_q_plan;
    bool   has_bus_q_plan = false;
    bool   has_result     = false;   // a run() with the flag on has happened

    // Setters mirror lightsim2grid's: a no-op when unchanged, otherwise the
    // previously computed report is dropped (it was made under other settings).
    void set_compute_physical_violations(bool v) {
        if (v == compute_physical_violations) return;
        compute_physical_violations = v;
        has_result = false;
    }
    void set_physical_violation_tol_mva(double v) {
        if (!(v >= 0.) || !std::isfinite(v)) {
            std::ostringstream exc_;
            exc_ << "physical_violation_tol_mva: the tolerance should be a finite, "
                    "non-negative number of MVA (got " << v << ").";
            throw std::runtime_error(exc_.str());
        }
        if (v != physical_violation_tol_mva) has_result = false;
        physical_violation_tol_mva = v;
    }
    void set_physical_violation_capacity(int k) {
        if (k <= 0) throw std::runtime_error("physical_violation_capacity must be > 0");
        if (k != physical_violation_capacity) has_result = false;
        physical_violation_capacity = k;
    }
    void set_bus_q_plan(const BusQPlanData& plan, int n_bus) {
        plan.validate(n_bus);
        bus_q_plan = plan;
        has_bus_q_plan = true;
        has_result = false;
    }
};

#endif  // PHYSICAL_CHECKS_DATA_HPP
