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

#include <string>
#include <vector>

namespace outer {

// The loops, in the grid's order (Ls2gOuterModel's Kind and LoopDecl)
enum DeviceLoopKind : int {
    DL_DISTRIBUTED_SLACK = 0, DL_HVDC_LIMITS = 1, DL_VOLTAGE_MONITORING = 2, DL_REACTIVE_LIMITS = 3,
    DL_SHUNT_VOLTAGE_CONTROL = 4, DL_PHASE_CONTROL = 5, DL_TRANSFORMER_VOLTAGE_CONTROL = 6,
};

struct DeviceTables {
    // ---- sizes, driver parameters ----------------------------------------------------------
    int n_bus = 0, n_gen = 0, n_sto = 0, n_svc = 0, n_load = 0;
    double sn = 1.;
    int slack_bus = -1;
    int max_outer_iterations = 0;
    // (Layout::watch_bus) solver bus -> index in the gathered arrays, -1 not watched
    std::vector<int> widx;
    int n_watch = 0;

    // ---- the loops, in order, and their parameters ---------------------------------------------
    std::vector<int> loop_kind;            // DeviceLoopKind
    std::vector<std::string> loop_name;    // their names (the trace's)
    int keep_trace = 0;                    // Ls2gModelOptions::keep_trace
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
    // a distributed-slack Newton (NROuterDist_*): its slack_absorbed column's buses (Layout::slack_bus),
    // each one's index by solver bus (-1: none), the reference's; whether DistributedSlack shares in it
    int dist = 0, ds_reserved = 0, n_slack = 0, ref_slack = -1;
    // ShuntControl's controllers (the regulating shunts of a bus): bus, shunts (CSR, flattened
    // index j), each shunt's id, sections range, grid count, status and B per count 0..max (pu,
    // CSR); ShuntVoltageControl's groups (the grid's): their controllers (entry, -1: none) and buses
    int n_sc = 0, n_sc_sh = 0, n_svg = 0;
    std::vector<int>    sc_bus, sc_sh_ptr, sh_id, sh_min, sh_max, sh_count0, sh_status, sh_has, sh_b_ptr;
    std::vector<double> sh_b;
    std::vector<int>    svg_ptr, svg_e, svg_bus;
    // BranchControl's transformers (the ledger's order): trafo id, phase / ratio column, ends, sign,
    // ratio exponent, the grid's taps (pos0 / rpos0, has_*: a tap changer), the taps a retap accepts
    // ([plo, phi] x [rlo, rhi]) and at each (rpos, pos), row-major from (rlo, plo): ratio, pi block
    // (4 complex, re/im) and shift (CSR bc_tab_ptr, in tab entries); each phase tap's shift (CSR)
    int n_bc = 0;
    std::vector<int>    bc_trafo, bc_phase, bc_ratio, bc_b1, bc_b2, bc_has_ptc, bc_has_rtc, bc_pos0, bc_rpos0;
    std::vector<int>    bc_plo, bc_phi, bc_rlo, bc_rhi, bc_tab_ptr, bc_alpha_ptr;
    std::vector<double> bc_sign, bc_rexp, bc_tab_ratio, bc_tab_block, bc_tab_shift, bc_alpha;
    // PhaseControl's shifters (the grid's): trafo, its entry, mode (0 active power, 1 current limiter),
    // regulated side (1 / 2), that side's nominal kV, its current target (A)
    int n_pc = 0;
    std::vector<int>    pc_trafo, pc_e, pc_mode, pc_side;
    std::vector<double> pc_vn_kv, pc_target;
    // TransformerVoltageControl: its groups (the grid's: regulated bus, target and half deadband in pu,
    // transformers CSR as group-transformer index gt -> trafo id and BranchControl entry), the
    // grid's branches (lines then trafos: solver ends, -1 open; trafo id or -1), each trafo's gt,
    // the switchable index of each bus; use_initial_tap_position
    int has_tvc = 0, n_tvg = 0, n_gt = 0, n_br = 0, tv_use_initial = 0;
    std::vector<int>    tvg_ptr, tvg_bus, gt_trafo, gt_e, br_b1, br_b2, br_trafo, trafo_gt, pin_of_bus;
    std::vector<double> tvg_target, tvg_hdb;
    std::vector<int>    slack_idx_of_bus;
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
    // the row's own values of the grid plan's buses (a row disconnecting generators, setting
    // voltages: its own plan, mapped onto the grid's), [rl_n]; and the generators its plan drops
    std::vector<double> rl_target_q, rl_target_vm;
    std::vector<unsigned char> rl_has_base;
    std::vector<unsigned char> rl_goff;         // [rl_gen CSR] a bus' generator the row does not have there
    std::vector<unsigned char> ctrl_off;        // [n_ctrl] the controllers its plan drops (tripped)
    // the row's own targets (set_row_targets: OuterRowView's), NaN where it sets none
    std::vector<double> row_gen_p, row_sto_p;   // [n_gen], [n_sto]
    std::vector<double> rl_target_p, rl_frozen_q;
    std::vector<int>    rl_state, rl_nb_pv_pq, rl_realistic;
    std::vector<unsigned char> ds_present;      // [n units0]
    std::vector<double> ds_injection, ds_current, ds_initial;
    std::vector<int> ds_bus;                    // [n units0] its bus in the row (a moved generator's new one)
    // a distributed-slack Newton's: the units out of its share (and at which bound), its last
    // slack_absorbed and whether the units took it, the target Q that share moved unseen
    unsigned char ds_in_newton = 0, sa_shared = 0;
    std::vector<unsigned char> ds_parked, ds_parked_high;   // [n units0]
    double sa = 0., unsolved_q_mvar = 0.;
    // ShuntControl / ShuntVoltageControl: the section counts [n_sc_sh], the pending sets, their B,
    // the controls on (1) / off (0) / untouched (-1) [n_sc]; the grid's groups this row hides [n_svg]
    std::vector<int> sc_counts, sc_ctl;
    std::vector<unsigned char> sc_pending, sv_hidden;
    std::vector<double> sc_b_new;
    // BranchControl / PhaseControl: each transformer's state (out of the row, taps, block, requests,
    // controls; BranchControl::Entry), the grid's shifters this row has and those that regulate
    std::vector<unsigned char> bc_out, bc_moved, pc_present, pc_regulates;
    std::vector<int> bc_pos, bc_rpos, bc_req_tap, bc_req_rtap, bc_ctl, bc_rctl;
    std::vector<double> bc_a_tap, bc_rho_tap, bc_tap;   // bc_tap [8 n_bc] re/im
    // TransformerVoltageControl: its step, its voltage limit (kV), the groups it hides, each group
    // transformer's switch and ratio ranges (TvcRatio: initial, min, max, shared_*), what decides a
    // ReactiveLimits bus' freeze that the row fixes (candidate, target |V| (NaN: the solve's), its
    // load (pu)) and whether it is frozen (at which |V|), the buses suspended, the branches closed
    int tvc_step = 0;
    double tvc_limit = 0.;
    std::vector<unsigned char> tv_hidden, tv_enabled, tv_cand, tv_frozen, tv_susp, br_closed;
    std::vector<double> tv_ratios, tv_tvm, tv_load, tv_frozen_vm;
    // the adjoint's (keep_final_jacobian): the units parked / their output as the last Newton solve
    // had them, [n units0]; each generator's capability slope recorded when its bus froze (pu per
    // MW, slope_set: one recorded), [n_gen]
    std::vector<unsigned char> ds_solved_parked;
    std::vector<double> ds_solved_mw;
    std::vector<unsigned char> slope_set;
    std::vector<double> slope;
    std::vector<unsigned char> hl_present;      // [hvdc plan lines]
    std::vector<int>    hl_frozen_dir;
    double lost_mw = 0.;                        // what DistributedSlack's share_lost_p counted
    std::vector<std::string> trace;             // the decisions its load recorded (keep_trace)
};

// What the adjoint needs of a row's end (Ls2gOuterModel::adjoint_info), as the device read it off
// the row's slot: the controls of its last solve, its units, its frozen buses' slopes
struct DeviceAdjointState {
    std::vector<unsigned char> pin, plan_held;   // [n_pin], [n_ctrl]
    std::vector<double> held_q;                  // [n_ctrl]
    unsigned char ds_in_newton = 0;
    std::vector<unsigned char> ds_present, ds_solved_parked;   // [n units0]
    std::vector<double> ds_solved_mw, ds_initial_mw;
    std::vector<int> rl_present, rl_state;       // [rl_n]
    std::vector<int> slope_k0, slope_gen, slope_ctrl;   // in the plan's bus order, then each bus' order
    std::vector<double> slope;
};

// Where a row's loops ended, for the rows resuming from it (Ls2gOuterModel::keep_point's
// ResumePoint), as the device model reads it off the row's slot at its end.
struct DevicePoint {
    int from_row = -1;                          // the row's own resume point (-1: the grid's / none)
    std::vector<double> gen_p, sto_p;           // [n_gen], [n_sto] MW, NaN untouched
    std::vector<int> hvdc_id, hvdc_regime;      // the lines out of their linear regime
    std::vector<int> svc_id;                    // the standby SVCs switched on, and their set-point
    std::vector<double> svc_target;
    std::vector<int> frozen_bus, frozen_state, frozen_realistic;   // sorted by bus
    std::vector<double> v_re, v_im;             // V at the frozen buses (same order)
    std::vector<int> parked_kind, parked_el, parked_high;   // the units a distributed-slack Newton left parked
    bool has_sections = false;                  // ShuntVoltageControl ran: the section counts [n_sc_sh]
    std::vector<int> sections;
    bool has_phase = false, has_ratio = false;  // PhaseControl / TransformerVoltageControl ran
    std::vector<int> bc_pos, bc_rpos;           // [n_bc] its taps
    std::vector<unsigned char> bc_out;
    double lost_mw = 0.;
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
    // a row the device ran ended with its loops stable: its point, for the rows resuming from it
    virtual void device_import_point(int row, const DevicePoint& p) = 0;
    // the rows' overrides (before their run): the device tables' ReactiveLimits plan grows by the buses
    // the rows' moved generators bring; true when it changed (the tables are to be rebuilt)
    virtual bool device_prepare_rows(const RowOverrides& ov) = 0;
    // adjoint_info, from what the device read off the row's slot at its end
    virtual void device_adjoint_info(const DeviceAdjointState& a, RowResult& r) const = 0;
    // row_blocks from a finished row's taps ([n_bc] each)
    virtual void device_row_blocks(const int* pos, const int* rpos, std::vector<int>& trafo,
                                   std::vector<double>& block) const = 0;
};

}  // namespace outer

#endif  // GPUSIM2GRID_OUTER_DEVICE_TABLES_HPP
