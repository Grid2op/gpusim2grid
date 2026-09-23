// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef PHYSICAL_CHECKS_IMPL_CUH
#define PHYSICAL_CHECKS_IMPL_CUH

// =============================================================================
// contingency/physical_checks_impl.cuh — session-side glue of the three post-
// solve physical checks (see physical_checks_data.hpp), shared by the three
// batch sessions. Templated on the driver (BatchPfDriver<Source>).
// =============================================================================

#include <chrono>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "physical_checks_data.hpp"
#include "../timing_utils.hpp"

namespace physical_checks {

struct SetupTimes { double bus_q_ms = 0.; double hvdc_p_ms = 0.; double gen_p_ms = 0.; };

// Before solve(): hand the driver what the three checks need and (re)seed
// their outputs, then produce the base ("n") case's own report when the base
// solve converged. Re-callable on a live driver (ScenarioSweep's persistent
// one): that is what resets the -1 "never simulated" sentinels between runs.
// gen_p_targets (nullptr / empty = none) are the per-row set-points of the
// active check, uploaded only when `gen_p_targets_dirty` or the driver has
// none yet (a cold driver).
template <class Driver>
SetupTimes before_solve(PhysicalChecksConfig& cfg, Driver& drv, const char* who,
                        double residual_tol, double sn_mva, bool base_converged,
                        const unsigned char* d_gen_off, int n_gen,
                        const RealMatRM* gen_p_targets = nullptr,
                        bool gen_p_targets_dirty = false)
{
    SetupTimes t;
    if (!cfg.compute_physical_violations) return t;
    if (!cfg.has_bus_q_plan)
        throw std::runtime_error(std::string(who) +
            ": compute_physical_violations requires the per-bus reactive capability -- call "
            "set_bus_q_capability() first (the *GPU facades do it from a lightsim2grid grid "
            "through set_bus_q_capability_from_grid(); an EMPTY plan is accepted when no "
            "machine regulates a voltage).");
    drv.set_bus_q_check(cfg.bus_q_plan, cfg.physical_violation_tol_mva,
                        cfg.physical_violation_capacity, residual_tol, d_gen_off, n_gen);
    if (base_converged) drv.run_bus_q_check_n();
    t.bus_q_ms = drv.bus_q_setup_ms();
    drv.set_hvdc_p_check(cfg.physical_violation_tol_mva, sn_mva,
                         cfg.physical_violation_capacity, residual_tol);
    if (base_converged) drv.run_hvdc_p_check_n();
    t.hvdc_p_ms = drv.hvdc_p_setup_ms();
    // the active check: an unset plan is an empty one (nothing was given
    // limits), see PhysicalChecksConfig::gen_p_plan
    drv.set_gen_p_check(cfg.gen_p_plan, cfg.physical_violation_tol_mva,
                        cfg.physical_violation_capacity, residual_tol, d_gen_off, n_gen);
    const bool has_targets = (gen_p_targets != nullptr) && gen_p_targets->size() > 0;
    if (has_targets && (gen_p_targets_dirty || !drv.has_gen_p_targets()))
        drv.upload_gen_p_targets(*gen_p_targets);
    else if (!has_targets && drv.has_gen_p_targets())
        drv.upload_gen_p_targets(RealMatRM());
    if (base_converged) drv.run_gen_p_check_n();
    t.gen_p_ms = drv.gen_p_setup_ms();
    return t;
}

// After solve(): fold the setup times (the plain `timings_ = solve()`
// assignment would drop them) and mark the report as available.
inline void after_solve(PhysicalChecksConfig& cfg, BatchTimings& tm, const SetupTimes& t)
{
    tm.t_physical_setup_ms += t.bus_q_ms + t.hvdc_p_ms + t.gen_p_ms;
    cfg.has_result = cfg.compute_physical_violations;
}

namespace detail {
template <class T>
Eigen::VectorXi to_int(const thrust::device_vector<T>& d) {
    thrust::host_vector<T> h = d;
    Eigen::VectorXi out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<int>(h[i]);
    return out;
}
template <class T>
RealVect to_real(const thrust::device_vector<T>& d) {
    thrust::host_vector<T> h = d;
    RealVect out(static_cast<Eigen::Index>(h.size()));
    for (size_t i = 0; i < h.size(); ++i) out(static_cast<Eigen::Index>(i)) = static_cast<eigen_real_type>(h[i]);
    return out;
}
inline double ms_since_(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace detail

// D->H copy of the bus reactive-capability report (batch rows, or the base
// "n" case). Synchronizes the driver's stream; the copy time is accumulated
// into t_acc (BatchTimings::t_copy_violations_to_host_ms).
template <class Driver>
BusQViolationsResult fetch_bus_q(const PhysicalChecksConfig& cfg, const Driver& drv,
                                 bool n_case, const char* who, double& t_acc)
{
    if (!cfg.has_result)
        throw std::runtime_error(std::string(who) +
            ": the physical checks were not requested. Set "
            "compute_physical_violations = True before run() to use this feature.");
    drv.cs.synchronize();
    auto t0 = std::chrono::steady_clock::now();
    BusQViolationsResult r;
    r.capacity  = drv.bus_q_capacity_;
    r.bus_id    = detail::to_int (n_case ? drv.d_bq_n_bus_id    : drv.d_bq_out_bus_id);
    r.type      = detail::to_int (n_case ? drv.d_bq_n_type      : drv.d_bq_out_type);
    r.value     = detail::to_real(n_case ? drv.d_bq_n_value     : drv.d_bq_out_value);
    r.limit     = detail::to_real(n_case ? drv.d_bq_n_limit     : drv.d_bq_out_limit);
    r.count     = detail::to_int (n_case ? drv.d_bq_n_count     : drv.d_bq_count);
    r.truncated = detail::to_int (n_case ? drv.d_bq_n_truncated : drv.d_bq_truncated);
    t_acc += detail::ms_since_(t0);
    return r;
}

template <class Driver>
HvdcPViolationsResult fetch_hvdc_p(const PhysicalChecksConfig& cfg, const Driver& drv,
                                   bool n_case, const char* who, double& t_acc)
{
    if (!cfg.has_result)
        throw std::runtime_error(std::string(who) +
            ": the physical checks were not requested. Set "
            "compute_physical_violations = True before run() to use this feature.");
    drv.cs.synchronize();
    auto t0 = std::chrono::steady_clock::now();
    HvdcPViolationsResult r;
    r.capacity  = drv.hvdc_p_capacity_;
    r.hvdc_id   = detail::to_int (n_case ? drv.d_hp_n_hvdc_id   : drv.d_hp_out_hvdc_id);
    r.side      = detail::to_int (n_case ? drv.d_hp_n_side      : drv.d_hp_out_side);
    r.value     = detail::to_real(n_case ? drv.d_hp_n_value     : drv.d_hp_out_value);
    r.limit     = detail::to_real(n_case ? drv.d_hp_n_limit     : drv.d_hp_out_limit);
    r.count     = detail::to_int (n_case ? drv.d_hp_n_count     : drv.d_hp_count);
    r.truncated = detail::to_int (n_case ? drv.d_hp_n_truncated : drv.d_hp_truncated);
    t_acc += detail::ms_since_(t0);
    return r;
}

template <class Driver>
GenPViolationsResult fetch_gen_p(const PhysicalChecksConfig& cfg, const Driver& drv,
                                 bool n_case, const char* who, double& t_acc)
{
    if (!cfg.has_result)
        throw std::runtime_error(std::string(who) +
            ": the physical checks were not requested. Set "
            "compute_physical_violations = True before run() to use this feature.");
    drv.cs.synchronize();
    auto t0 = std::chrono::steady_clock::now();
    GenPViolationsResult r;
    r.capacity     = drv.gen_p_capacity_;
    r.element_type = detail::to_int (n_case ? drv.d_gp_n_element_type : drv.d_gp_out_element_type);
    r.element_id   = detail::to_int (n_case ? drv.d_gp_n_element_id   : drv.d_gp_out_element_id);
    r.type         = detail::to_int (n_case ? drv.d_gp_n_type         : drv.d_gp_out_type);
    r.value        = detail::to_real(n_case ? drv.d_gp_n_value        : drv.d_gp_out_value);
    r.limit        = detail::to_real(n_case ? drv.d_gp_n_limit        : drv.d_gp_out_limit);
    r.count        = detail::to_int (n_case ? drv.d_gp_n_count        : drv.d_gp_count);
    r.truncated    = detail::to_int (n_case ? drv.d_gp_n_truncated    : drv.d_gp_truncated);
    t_acc += detail::ms_since_(t0);
    return r;
}

}  // namespace physical_checks

#endif  // PHYSICAL_CHECKS_IMPL_CUH
