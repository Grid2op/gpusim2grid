// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef BUS_Q_CHECK_DATA_HPP
#define BUS_Q_CHECK_DATA_HPP

// =============================================================================
// contingency/bus_q_check_data.hpp — host-side "plan" of the per-bus reactive-
// capability check (compute_physical_violations, lightsim2grid PR #206 parity)
// =============================================================================
//
// A flattened, device-friendly copy of lightsim2grid's bus_q_check::BusQPlan
// (batch_algorithm/BusQCheck.hpp): which buses are checked at all, and what
// holds each one. Built ONCE per session -- by the bridge through
// lightsim2grid's own build_bus_q_plan (extract_bus_q_plan_from_lsgrid,
// ls2g_bridge.cpp), or handed in as raw arrays by a caller in array/tuple mode
// (set_bus_q_capability) -- and uploaded to the driver, which then evaluates
// every row on the device (check_bus_q_violations_kernel).
//
// Per checked bus k (solver numbering):
//   - qmin_fixed_mvar / qmax_fixed_mvar : the capability that never changes
//     from row to row -- the voltage-regulating HVDC converter stations and
//     storage units standing on the bus ([min_q, max_q] MVAr each, summed);
//   - n_fixed : how many such always-live machines hold the bus (stations +
//     storage units + voltage-mode SVCs), so that a row can tell "nothing holds this bus any
//     more" (nb_live == 0 -> not checked) once its generators are off;
//   - bmin_sum_pu / bmax_sum_pu : the voltage-mode SVCs' susceptance range,
//     summed (pu, base sn_mva). An SVC's capability is a susceptance, so what
//     it is worth in MVAr is re-evaluated at each row's own |V|:
//     q = b . |V|^2 . sn_mva (generator convention);
//   - gen_start[k] .. gen_start[k+1] : CSR slice of the voltage-regulating
//     generators holding the bus, each with its container id (the column of a
//     ScenarioSweep generator-contingency mask: a generator a row disconnects
//     leaves the sum) and its fixed [min_q, max_q] in MVAr.
//
// Deliberately NOT carried, compared to lightsim2grid's plan: the controller
// positions (`ctrl_pos`) and the substation name. The GPU compares against the
// RAW reactive residual imag(V . conj(Ybus . V) - Sbus), which is what
// lightsim2grid's mis_bus + sum(Q_c) reconstructs (see the kernel's doc), so
// no VoltageControl state is needed; and gpusim2grid's violation records
// carry no names.
//
// CUDA-free on purpose: included by the session headers, which the host
// compiler builds into ls2g_bridge.cpp.
// =============================================================================

#include <sstream>
#include <stdexcept>
#include <Eigen/Core>

#include "../dtypes.hpp"

struct BusQPlanData {
    int             n_check = 0;
    Eigen::VectorXi bus_solver;                        // [n_check]
    RealVect        qmin_fixed_mvar, qmax_fixed_mvar;  // [n_check] HVDC station + storage unit sums (MVAr)
    Eigen::VectorXi n_fixed;                           // [n_check] stations + storage units + svcs (always live)
    RealVect        bmin_sum_pu, bmax_sum_pu;          // [n_check] SVC susceptance sums (pu)
    Eigen::VectorXi gen_start;                         // [n_check + 1] CSR
    Eigen::VectorXi gen_id;                            // [nnz] generator container id
    RealVect        gen_qmin_mvar, gen_qmax_mvar;      // [nnz] MVAr
    double          sn_mva = 100.0;

    bool empty() const { return n_check == 0; }
    int  n_gen_entries() const { return static_cast<int>(gen_id.size()); }

    // Structural checks only (sizes / index ranges); throws std::runtime_error.
    void validate(int n_bus) const {
        std::ostringstream exc_;
        auto fail = [&](const std::string& what) {
            exc_ << "BusQPlanData: " << what;
            throw std::runtime_error(exc_.str());
        };
        if (n_check < 0) fail("n_check must be >= 0");
        auto need = [&](long long got, long long want, const char* name) {
            if (got != want) {
                std::ostringstream m;
                m << name << " has size " << got << ", expected " << want;
                fail(m.str());
            }
        };
        need(bus_solver.size(),      n_check,     "bus_solver");
        need(qmin_fixed_mvar.size(), n_check,     "qmin_fixed_mvar");
        need(qmax_fixed_mvar.size(), n_check,     "qmax_fixed_mvar");
        need(n_fixed.size(),         n_check,     "n_fixed");
        need(bmin_sum_pu.size(),     n_check,     "bmin_sum_pu");
        need(bmax_sum_pu.size(),     n_check,     "bmax_sum_pu");
        need(gen_start.size(),       n_check + 1, "gen_start");
        const long long nnz = gen_id.size();
        need(gen_qmin_mvar.size(), nnz, "gen_qmin_mvar");
        need(gen_qmax_mvar.size(), nnz, "gen_qmax_mvar");
        if (n_check > 0) {
            if (gen_start(0) != 0) fail("gen_start[0] must be 0");
            for (int k = 0; k < n_check; ++k) {
                if (gen_start(k + 1) < gen_start(k)) fail("gen_start must be non-decreasing");
                if (bus_solver(k) < 0 || bus_solver(k) >= n_bus) {
                    std::ostringstream m;
                    m << "bus_solver[" << k << "] = " << bus_solver(k)
                      << " is outside [0, n_bus=" << n_bus << ")";
                    fail(m.str());
                }
                if (n_fixed(k) < 0) fail("n_fixed must be >= 0");
            }
            if (gen_start(n_check) != nnz) fail("gen_start[n_check] must equal gen_id.size()");
        } else if (nnz != 0) {
            fail("gen_id must be empty when n_check == 0");
        }
        for (long long p = 0; p < nnz; ++p)
            if (gen_id(p) < 0) fail("gen_id entries must be >= 0");
        if (!(sn_mva > 0.)) fail("sn_mva must be > 0");
    }
};

#endif  // BUS_Q_CHECK_DATA_HPP
