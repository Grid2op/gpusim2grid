// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/device_tables.hpp -- what the device outer-loop model reads, on the host
// (docs/outer_loops_on_device.md, §4.6).
//
// Filled by Ls2gOuterModel::device_tables() from its own plans, so the device
// model starts from exactly the data the host model runs on; uploaded once by
// the device model. Plain vectors, CUDA-free and lightsim2grid-free: the bridge
// fills them, the .cu reads them. Variable-length lists are CSR (`*_ptr` of size
// n + 1 into the flat array). Units: pu for powers on the solver side, MW / MVAr
// for the containers', as the host model has them.
// =============================================================================

#ifndef GPUSIM2GRID_OUTER_DEVICE_TABLES_HPP
#define GPUSIM2GRID_OUTER_DEVICE_TABLES_HPP

#include <vector>

namespace outer {

// The loops, in the grid's order (Ls2gOuterModel's Kind and LoopDecl)
enum DeviceLoopKind : int {
    DL_DISTRIBUTED_SLACK = 0, DL_HVDC_LIMITS = 1, DL_VOLTAGE_MONITORING = 2, DL_REACTIVE_LIMITS = 3,
};

struct DeviceTables {
    // ---- sizes, driver parameters ----------------------------------------------------------
    int n_bus = 0, n_gen = 0, n_sto = 0, n_svc = 0;
    double sn = 1.;
    int slack_bus = -1;
    int max_outer_iterations = 0;
    // (Layout::watch_bus) solver bus -> index in the gathered arrays, -1 not watched
    std::vector<int> widx;
    int n_watch = 0;

    // ---- the loops, in order, and their parameters ---------------------------------------------
    std::vector<int> loop_kind;            // DeviceLoopKind
    double ds_mismatch_mw = 1., ds_eps_mw = 1e-3, ds_moved_fraction = 0.9;
    int    ds_fail = 1, ds_run_without_participant = 0;
    int    rl_max_switch = 3, rl_robust = 1;
    double rl_max_mismatch = 1e-4, rl_base_mva = 100.;
    double rl_vmin = 0.5, rl_vmax = 2., rl_margin = 1.02, rl_restart = 1.;
    int    robust_mode = 0;                // OuterLoopDriverParams::voltage_remote_control_robust_mode

    // ---- generators ---------------------------------------------------------------------------
    std::vector<double> gen_target_p;      // the grid's target P (MW), the "in force" base
    std::vector<double> gen_min_q, gen_max_q, gen_target_q, gen_raw_target_q;   // MVAr, raw NaN: none
    std::vector<int>    gen_vreg_on;
    // capability curves (LSGrid::gen_curves_): per generator its points sorted by p
    std::vector<int>    curve_ptr;         // [n_gen + 1]
    std::vector<double> curve_p, curve_min_q, curve_max_q;
    // the ReactiveLimits buses (solver ids) each generator reaches (gen_rl_bus_)
    std::vector<int>    gen_rl_bus_ptr, gen_rl_bus;
    // ---- storage units, SVCs, hvdc stations ----------------------------------------------------
    std::vector<double> sto_target_p;      // load convention (MW)
    std::vector<double> svc_b_min, svc_b_max;
    std::vector<double> station_min_q1, station_max_q1, station_min_q2, station_max_q2;   // per hvdc id

    // ---- ReactiveLimits: the grid's plan (rl0_), one entry per controller bus ------------------
    int rl_n = 0;
    int rl_n_bus_q = 0;     // its first entries are bus_q_check's (a mask drops them), the monitors after
    int rl_has_monitor = 0;
    std::vector<int>    rl_bus, rl_local, rl_reg_bus, rl_group, rl_monitor_svc, rl_pin, rl_has_base, rl_has_storage;
    std::vector<double> rl_min_q, rl_max_q, rl_target_vm, rl_nominal_v, rl_target_q, rl_base_min_q, rl_base_max_q;
    // its entry's elements (BusQEntry): CSR per controller bus
    std::vector<int> rl_gen_ptr, rl_gen, rl_svc_ptr, rl_svc, rl_sto_ptr, rl_sto, rl_ctrl_ptr, rl_ctrl;
    // a slot's start (RlPlan::state0): target P (MW), frozen Q, state, switch count, realistic
    std::vector<double> rl0_target_p, rl0_frozen_q;
    std::vector<int>    rl0_state, rl0_nb_pv_pq, rl0_realistic;

    // ---- VoltageControl (ledger order) -----------------------------------------------------------
    int n_ctrl = 0, n_grp = 0;
    std::vector<int>    vc_bus, vc_kind, vc_elem, vc_group, vc_plan_held;
    std::vector<double> vc_slope, vc_q_plan, vc_v_set;
    std::vector<int>    vc_at_bus_ptr, vc_at_bus;    // per solver bus, its controllers
    std::vector<int>    svc_ctrl;                    // svc id -> its held controller, -1

    // ---- droop hvdc (ledger order) and the HvdcLimits plan (the lines in AC emulation) -----------
    int n_hvdc = 0;
    std::vector<int>    hvdc_bus1, hvdc_bus2, hvdc_status0, hvdc_id;
    std::vector<double> hvdc_p0, hvdc_k, hvdc_lf1, hvdc_lf2, hvdc_r, hvdc_pmax12, hvdc_pmax21;

    // the HvdcAcEmulationLimits plan (hvdc_plan0_: the lines in AC emulation, every one linear)
    std::vector<int>    hl_hvdc_id, hl_bus1, hl_bus2;
    std::vector<double> hl_p0_mw, hl_k_mw_per_rad, hl_pmax12_mw, hl_pmax21_mw;
    // the VoltageMonitoring plan (monitors_: the standby SVCs the plan holds) and their set-points
    std::vector<int>    mon_svc, mon_reg_bus, mon_svc_bus;
    std::vector<double> mon_low_vm, mon_high_vm, mon_vn_kv;
    std::vector<double> svc_standby_low_target, svc_standby_high_target;   // per svc id, pu (NaN: none)
    std::vector<int>    held_svcs;           // sorted svc ids the plan holds
    std::vector<int>    svc_reg_is_own_bus;  // per svc id: regulated bus == its own bus

    // ---- DistributedSlack: the grid's participants (units0_), in upstream's order ---------------
    std::vector<int>    ds_el_id, ds_kind, ds_bus, ds_in_slack;
    std::vector<double> ds_weight, ds_min_mw, ds_max_mw;

    int n_pin = 0;            // switchable buses (Layout::pin_bus)
    std::vector<int> pin_bus;
    // the buses a slot's Sbus correction may touch, sorted: what target_delta and freeze_q are
    // kept densely over on the device (the DistributedSlack units' buses, the local ReactiveLimits
    // buses, every bus a loaded slot already had an entry on)
    std::vector<int> corr_bus;
};

// One slot right after the host model loaded a row in it (Ls2gOuterModel::load): what the
// device model uploads. Indices into the grid's plans (rl_*, ds_*, hl_*) with a "present"
// flag: a masked row's plan is the grid's without its masked buses, in the same order.
struct DeviceSlotImage {
    int row = -1;
    std::vector<int> active;
    int last_fixing = -1;
    int check_unrealistic = 0;
    std::vector<unsigned char> masked_bus;      // [n_bus]
    std::vector<unsigned char> pq;              // [n_pin]
    std::vector<double> held_q, svc_on;         // [n_ctrl]
    std::vector<int>    hvdc_regime;            // [n_hvdc]
    std::vector<double> gen_target_p;           // [n_gen], NaN untouched
    std::vector<double> sto_target_p;           // [n_sto]
    std::vector<int>    delta_bus;              // target_delta, sorted
    std::vector<double> delta_p, delta_q;
    std::vector<int>    freeze_bus;             // freeze_q, sorted
    std::vector<double> freeze_q;
    std::vector<int>    vm_reset_bus;
    std::vector<double> vm_reset_vm;
    int lim_base = 1;
    std::vector<int>    lim_dirty_bus;          // the buses whose limits a moved generator dirtied
    std::vector<unsigned char> rl_present;      // [rl_n]
    std::vector<double> rl_target_p, rl_frozen_q;
    std::vector<int>    rl_state, rl_nb_pv_pq, rl_realistic;
    std::vector<unsigned char> ds_present;      // [n units0]
    std::vector<double> ds_injection, ds_current, ds_initial;
    std::vector<unsigned char> hl_present;      // [hvdc plan lines]
    std::vector<int>    hl_frozen_dir;
};

// What the device model asks of the host one (implemented by Ls2gOuterModel): its tables, once,
// and the state of a slot it just loaded.
class DeviceHostSide {
public:
    virtual ~DeviceHostSide() = default;
    virtual DeviceTables device_tables() const = 0;
    // throws NotImplementedError-worthy std::domain_error when the slot holds something the
    // device model does not run yet (a row-level plan, own units)
    virtual void device_slot_image(int slot, DeviceSlotImage& img) const = 0;
};

}  // namespace outer

#endif  // GPUSIM2GRID_OUTER_DEVICE_TABLES_HPP
