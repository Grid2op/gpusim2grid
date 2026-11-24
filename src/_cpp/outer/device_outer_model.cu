// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/device_outer_model.cu -- the outer loops' decisions on the device
// (outer_loops_backend='device'; docs/outer_loops_on_device.md, §4.6).
//
// One thread per slot runs the port of Ls2gOuterModel::on_solved_impl / drive /
// finish, of the DistributedSlack (single slack), AcHvdcAcEmulationLimits,
// VoltageMonitoring and ReactiveLimits checks, of lightsim2grid's
// slack_redistribution::distribute and capability curves, and of materialize.
// Every __device__ function below is named after the host function it follows,
// statement for statement, and sums in the same order; this file is compiled
// without FMA contraction (CMakeLists.txt), so a decision on a threshold goes the
// same way as on the host. The lockstep tests (tests/python/outer_device/
// test_device_model.py) hold the two models to the same controls after every call.
// =============================================================================

#include "device_outer_model.hpp"
#include "../ledger_data.hpp"   // HVDC_STATUS_TRIPPED
#include "../nvtx_range.hpp"
#include "../cuda_utils.h"      // upload_h2d_grow
#include "../contingency/slot_schedule.cuh"   // move_slot_rows_kernel

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <sstream>
#include <memory>
#include <type_traits>
#include <string>
#include <vector>

namespace outer {
namespace {

constexpr int STABLE = 0, UNSTABLE = 1, FAILED = 2, CHECK_AGAIN = 3;   // OuterLoopStatus
constexpr int VC_GEN = 0, VC_SVC = 1, VC_HVDC_1 = 2, VC_HVDC_2 = 3;     // VoltageControlSolverData::Kind
// device error codes (a case the host scan should have refused, or a capacity)
constexpr int kNoTap = -2147483647 - 1;   // RatioTapControl: no move requested (INT_MIN)
constexpr int ERR_NOT_WATCHED = 1, ERR_CORR_BUS = 2, ERR_VM_RESET_CAP = 3, ERR_NO_PIN = 4, ERR_TRACE_CAP = 5;
// the decision trace's actions (Ls2gOuterModel::record's strings, in this order: kActionNames)
enum TraceAction : int {
    TA_DISTRIBUTE, TA_FAIL_RESIDUE, TA_UNITS_MOVED, TA_PARK, TA_UNPARK, TA_TARGET_Q_FOLLOWED, TA_RELEASE,
    TA_SATURATE, TA_SWITCH_ON, TA_KEPT_PQ_GROUP_HOLDS, TA_KEPT_PQ, TA_KEPT_PV_STRONGEST, TA_PV_TO_PQ_UNREALISTIC,
    TA_PV_TO_PQ, TA_KEPT_PQ_MAX_SWITCH, TA_PQ_TO_PV, TA_LIMIT_MOVED, TA_SETTLED, TA_ROUND_SECTIONS,
    TA_ROUND_TAP, TA_KEPT_TAP, TA_MOVE_TAP, TA_ROUND_TO_RANGE, TA_SHARE_LOST_P
};
const char* const kActionNames[] = {
    "DISTRIBUTE", "FAIL_RESIDUE", "UNITS_MOVED", "PARK", "UNPARK", "TARGET_Q_FOLLOWED", "RELEASE",
    "SATURATE", "SWITCH_ON", "KEPT_PQ_GROUP_HOLDS", "KEPT_PQ", "KEPT_PV_STRONGEST", "PV_TO_PQ_UNREALISTIC",
    "PV_TO_PQ", "KEPT_PQ_MAX_SWITCH", "PQ_TO_PV", "LIMIT_MOVED", "SETTLED", "ROUND_SECTIONS",
    "ROUND_TAP", "KEPT_TAP", "MOVE_TAP", "ROUND_TO_RANGE", "SHARE_LOST_P"};

void check_cuda(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string("device outer model: ") + what + ": " + cudaGetErrorString(e));
}

template <typename T> T* raw(thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
template <typename T> const T* raw(const thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }

// ---- what a kernel reads and writes, as raw pointers --------------------------------------------
struct Dev {
    // the per-slot pools are slot-major: slot s's fields of one pool are one row of w_* elements
    size_t w_int, w_u8, w_f64;
    // sizes and parameters
    int n_bus, n_gen, n_sto, n_watch, n_loops, n_pin, n_ctrl, n_grp, n_hvdc, rl_n, n_units, n_hl, n_mon, nc, vcap;
    int rl_n_bus_q;
    int slack_bus, max_outer, robust_mode;
    int dist, n_slack, ref_slack;
    int n_sc, n_sc_sh, n_svg, n_bc, n_pc;
    const int *bc_trafo, *bc_phase, *bc_b1, *bc_b2, *bc_has_ptc, *bc_plo, *bc_phi, *bc_rlo, *bc_rhi, *bc_tab_ptr, *bc_alpha_ptr;
    const double *bc_sign, *bc_rexp, *bc_tab_ratio, *bc_tab_block, *bc_tab_shift, *bc_alpha;
    const int *pc_trafo, *pc_e, *pc_mode, *pc_side;
    int has_tvc, n_tvg, n_gt, n_br, n_line, tv_use_initial;
    const int *bc_ratio, *tvg_ptr, *tvg_bus, *gt_trafo, *gt_e, *br_b1, *br_b2, *br_trafo, *trafo_gt, *pin_of_bus;
    const double *tvg_target, *tvg_hdb;
    const double *pc_vn_kv, *pc_target;
    const int *sc_bus, *sc_sh_ptr, *sh_min, *sh_max, *sh_count0, *sh_status, *sh_b_ptr, *svg_ptr, *svg_e, *svg_bus;
    const double* sh_b;
    int keep_adj;                  // keep what the adjoint needs (OuterHook::set_keep_adjoint)
    int trace, tcap;               // keep the decision trace, its per-slot capacity between flushes
    const double* mon_vn;          // [n_mon] kV
    const int* slack_idx;          // [n_bus] index in the slack_absorbed column's buses, -1
    double sn;
    double ds_mismatch_mw, ds_eps_mw, ds_moved_fraction;
    int ds_fail;
    int rl_max_switch, rl_robust;
    double rl_max_mismatch, rl_base_mva, rl_vmin, rl_vmax, rl_margin, rl_restart;
    // static tables
    const int *widx, *loop_kind;
    const double *gen_tp0, *gen_min_q, *gen_max_q, *gen_target_q, *gen_raw_q;
    const int *gen_vreg, *curve_ptr, *gen_rl_ptr, *gen_rl_bus;
    const double *curve_p, *curve_min, *curve_max, *sto_tp0, *svc_b_min, *svc_b_max;
    const double *st_min1, *st_max1, *st_min2, *st_max2;
    const int *rl_bus, *rl_local, *rl_reg, *rl_group, *rl_mon, *rl_pin, *rl_has_base;
    const double *rl_min_q, *rl_max_q, *rl_tvm, *rl_nv, *rl_tq, *rl_bmin, *rl_bmax;
    const int *rl_gen_ptr, *rl_gen, *rl_svc_ptr, *rl_svc, *rl_ctrl_ptr, *rl_ctrl;
    const int *vc_bus, *vc_kind, *vc_elem, *vc_group, *vc_plan_held, *vc_at_ptr, *vc_at, *svc_ctrl;
    const double *vc_slope, *vc_q_plan, *vc_v_set;
    const int *hv_b1, *hv_b2, *hv_st0;
    const double *hv_p0, *hv_k, *hv_lf1, *hv_lf2, *hv_r, *hv_pmax12, *hv_pmax21;
    const unsigned char *hvdc_at_bus;
    const int *hl_id, *hl_b1, *hl_b2, *hl_entry;
    const double *hl_p0, *hl_k, *hl_pmax12, *hl_pmax21;
    const int *mon_svc, *mon_reg, *mon_bus;
    const double *mon_low, *mon_high, *svc_low_t, *svc_high_t;
    const int *ds_el, *ds_kind, *ds_bus;
    const double *ds_w, *ds_min, *ds_max;
    // a row's lost power shared on the device (DeviceTables::rp_*): the n point's moved units, their
    // buses and moves, its V at the watched buses ([2 n_watch], a Row's first fields), each
    // ReactiveLimits bus' frozen state there; ReactiveLimits' storage units per bus (CSR)
    int ds_reserved, rp_n, n_rp_gen, n_rp_sto;
    const int *rp_gen_bus, *rp_sto_bus, *rl_rp_state, *rl_sto_ptr, *rl_sto, *rl_has_sto;
    const double *rp_gen_move, *rp_sto_move, *rp_v;
    const int *corr_bus, *corr_of_bus;
    // per slot: the driver state
    int *row, *it, *last_unstable, *last_status, *first, *changed, *nb_outer, *nb_passes, *total_nr, *old_total;
    int *nr_steps, *v_version, *solver_ok, *unrealistic, *check_unreal, *pass_check_unreal, *last_fixing, *n_active;
    int *last_loop, *lim_base, *vmr_n, *err, *p_recheck;
    int *active, *loop_iter, *done_at;
    unsigned char *masked, *lim_dirty, *pq;
    double *held_q, *svc_on;
    int *hvdc_regime;
    double *gen_tp, *sto_tp, *lost_mw, *row_gp, *row_sp;
    unsigned char *ds_inn, *ds_park, *ds_parkh, *sa_sh;
    double *sa, *unsq, *c_slack_w;
    unsigned char *ds_spark, *slp_set;
    double *ds_smw, *slp;
    int *tr_n, *tr_li, *tr_loop, *tr_act, *tr_el, *tr_outer;
    int *sc_cnt, *sc_ctl, *s_ord, *s_cnt;
    int *bc_pos, *bc_rpos, *bc_rq, *bc_rrq, *bc_ctl, *bc_rctl;
    unsigned char *bc_out, *bc_mov, *bc_rt, *pc_pres, *pc_reg;
    double *bc_atap, *bc_rtap, *bc_tap;
    int *tv_step, *tv_par;
    unsigned char *tv_hid, *tv_en, *tv_cand, *tv_fz, *tv_susp, *tv_closed, *tv_pv;
    double *tv_lim, *tv_rat, *tv_tvm, *tv_load, *tv_fzv;
    unsigned char *sc_pend, *sv_hid;
    double *sc_bnew, *s_mag;
    unsigned char *tr_taken, *s_gh;
    double *tr_val, *tr_lim, *s_tv, *s_tl, *s_hp;
    unsigned char *dset, *fset;
    double *dre, *dim, *fq;
    int *vmr_bus;
    double *vmr_vm;
    unsigned char *rl_present, *rl_hb_s, *rl_goff, *ctrl_off;
    double *rl_tp, *rl_fq, *rl_tq_s, *rl_tvm_s;
    int *rl_state, *rl_nb, *rl_real;
    unsigned char *ds_present;
    double *ds_inj, *ds_cur, *ds_ini;
    // each unit's bus in the slot's row: the grid's (ds_bus), a generator the row moves on its new one
    // (the host's ds_unit_bus)
    int *ds_ubus;
    unsigned char *hl_present;
    int *hl_dir;
    // per slot: the controls (materialize)
    unsigned char *c_pin, *c_plan_held;
    double *c_held_q, *c_v_set;
    int *c_hvdc_status, *c_sb_n, *c_sb_bus;
    double *c_sb_dp, *c_sb_dq;
    // per slot: scratch
    unsigned char *s_hold, *s_slope;
    int *s_pq_k, *s_pq_lo, *s_pv_k, *s_mv_k;
    unsigned char *s_pq_real;
    int *s_cls;
    double *s_qlim;
    double *s_pq_lim;
    double *s_lo, *s_hi, *s_new;
    unsigned char *s_act, *s_sat;
    int *s_hl;
    // per row: the results
    int *r_status, *r_unreal, *r_failed, *r_outer, *r_nr;
    double *r_unit_mw;
    int *r_hvdc;
    int *r_sc;                     // [rows x n_sc_sh] the section counts a row ends with
    int *r_pos, *r_rpos;           // [rows x n_bc] the taps a row ends with
};

__host__ __device__ inline size_t pstride(const Dev& d, const int*) { return d.w_int; }
__host__ __device__ inline size_t pstride(const Dev& d, const unsigned char*) { return d.w_u8; }
__host__ __device__ inline size_t pstride(const Dev& d, const double*) { return d.w_f64; }
// slot s's block of a per-slot field (its width is the field's, the row stride the pool's)
#define SLOT(arr, w) (d.arr + static_cast<size_t>(s) * pstride(d, d.arr))
// slot s's value of a one-element per-slot field
#define S1(name) (*SLOT(name, 1))

// ---- the gathered quantities of one converged slot (RowView) -------------------------------------
struct Row {
    const double* g;   // [6 n_watch + n_ctrl + 2 n_bc + n_sc + 2], the driver's oc_gather layout
    int nw, nctrl, nbc;
    __device__ int w(const Dev& d, int s, int b) const {
        const int k = (b >= 0 && b < d.n_bus) ? d.widx[b] : -1;
        if (k < 0) S1(err) = ERR_NOT_WATCHED;
        return k < 0 ? 0 : k;
    }
    __device__ double v_re(int k) const { return g[k]; }
    __device__ double v_im(int k) const { return g[nw + k]; }
    __device__ double s_re(int k) const { return g[2 * nw + k]; }
    __device__ double s_im(int k) const { return g[3 * nw + k]; }
    __device__ double sb_re(int k) const { return g[4 * nw + k]; }
    __device__ double sb_im(int k) const { return g[5 * nw + k]; }
    __device__ double vc_q(int c) const { return g[6 * nw + c]; }
    __device__ double sc_b(int e) const { return g[6 * nw + nctrl + 2 * nbc + e]; }
    __device__ double bc_a(int e) const { return g[6 * nw + nctrl + e]; }
    __device__ double bc_rho(int e) const { return g[6 * nw + nctrl + nbc + e]; }
};

__device__ double vm_of(const Dev& d, int s, const Row& rv, int b)
{
    const int k = rv.w(d, s, b);
    return hypot(rv.v_re(k), rv.v_im(k));
}
__device__ double va_of(const Dev& d, int s, const Row& rv, int b)
{
    const int k = rv.w(d, s, b);
    return atan2(rv.v_im(k), rv.v_re(k));
}
__device__ double norm_of(const Dev& d, int s, const Row& rv, int b)
{
    const int k = rv.w(d, s, b);
    return rv.v_re(k) * rv.v_re(k) + rv.v_im(k) * rv.v_im(k);
}

// ---- lightsim2grid: capability curves and the forced target Q (LSGrid.cpp, v1.3.1) ---------------
__device__ bool curve_at(const Dev& d, int g, double p_mw, double& min_q, double& max_q)
{
    const int a0 = d.curve_ptr[g], n = d.curve_ptr[g + 1] - a0;
    if (n < 2 || !isfinite(p_mw)) return false;
    int i = 0;
    while (i + 2 < n && p_mw > d.curve_p[a0 + i + 1]) ++i;
    const double ap = d.curve_p[a0 + i], bp = d.curve_p[a0 + i + 1];
    const double amin = d.curve_min[a0 + i], bmin = d.curve_min[a0 + i + 1];
    const double amax = d.curve_max[a0 + i], bmax = d.curve_max[a0 + i + 1];
    if (bp == ap) return false;
    const double t = (p_mw - ap) / (bp - ap);
    min_q = amin + (bmin - amin) * t;
    max_q = amax + (bmax - amax) * t;
    if (min_q > max_q) min_q = max_q = (min_q + max_q) / 2.;
    return true;
}
// LSGrid::gen_limit_slopes_at_outer_target (curve_at's slopes): 0 without a curve or a target in force
__device__ double gen_limit_slope(const Dev& d, int g, double p_mw, bool max)
{
    const int a0 = d.curve_ptr[g], n = d.curve_ptr[g + 1] - a0;
    if (n < 2 || !isfinite(p_mw)) return 0.;
    int i = 0;
    while (i + 2 < n && p_mw > d.curve_p[a0 + i + 1]) ++i;
    const double ap = d.curve_p[a0 + i], bp = d.curve_p[a0 + i + 1];
    const double amin = d.curve_min[a0 + i], bmin = d.curve_min[a0 + i + 1];
    const double amax = d.curve_max[a0 + i], bmax = d.curve_max[a0 + i + 1];
    if (bp == ap) return 0.;
    const double t = (p_mw - ap) / (bp - ap);
    const double min_q = amin + (bmin - amin) * t, max_q = amax + (bmax - amax) * t;
    double dmin = (bmin - amin) / (bp - ap), dmax = (bmax - amax) / (bp - ap);
    if (min_q > max_q) dmin = dmax = (dmin + dmax) / 2.;
    return max ? dmax : dmin;
}
// LSGrid::gen_limits_at_outer_target
__device__ void gen_limits(const Dev& d, int g, double p, double& lo, double& hi)
{
    lo = d.gen_min_q[g];
    hi = d.gen_max_q[g];
    if (!isfinite(p)) return;
    double l, h;
    if (curve_at(d, g, p, l, h)) { lo = l; hi = h; }
}
// LSGrid::gen_target_q_at_outer_target
__device__ double gen_target_q_at(const Dev& d, int g, double p)
{
    const double target_q = d.gen_target_q[g];
    const double raw = d.gen_raw_q[g];
    if (!isfinite(raw) || !isfinite(p)) return target_q;
    double lo, hi;
    gen_limits(d, g, p, lo, hi);
    double q = raw;
    if (q < lo) q = lo;
    else if (q > hi) q = hi;
    return q;
}

// ---- Ls2gOuterModel's helpers --------------------------------------------------------------------
// the loops' target, else the row's own (set_row_targets), else the grid's
__device__ double gen_in_force(const Dev& d, int s, int g)
{
    const double v = SLOT(gen_tp, d.n_gen)[g];
    if (!isnan(v)) return v;
    const double r = SLOT(row_gp, d.n_gen)[g];
    return isnan(r) ? d.gen_tp0[g] : r;
}
__device__ double sto_in_force(const Dev& d, int s, int i)
{
    const double v = SLOT(sto_tp, d.n_sto)[i];
    if (!isnan(v)) return v;
    const double r = SLOT(row_sp, d.n_sto)[i];
    return isnan(r) ? d.sto_tp0[i] : r;
}
// OuterGridView::gen_target_p_override: the loops' value, else the row's, else NaN
__device__ double gen_override(const Dev& d, int s, int g)
{
    if (g < 0) return CUDART_NAN;
    const double v = SLOT(gen_tp, d.n_gen)[g];
    return isnan(v) ? SLOT(row_gp, d.n_gen)[g] : v;
}
__device__ int corr_index(const Dev& d, int s, int bus)
{
    const int ci = (bus >= 0 && bus < d.n_bus) ? d.corr_of_bus[bus] : -1;
    if (ci < 0) S1(err) = ERR_CORR_BUS;
    return ci;
}
__device__ void add_injection(const Dev& d, int s, int bus, double dre, double dim)
{
    if (dre == 0. && dim == 0.) return;
    const int ci = corr_index(d, s, bus);
    if (ci < 0) return;
    unsigned char* set = SLOT(dset, d.nc);
    double* re = SLOT(dre, d.nc);
    double* im = SLOT(dim, d.nc);
    if (!set[ci]) { set[ci] = 1; re[ci] = 0.; im[ci] = 0.; }
    re[ci] += dre;
    im[ci] += dim;
}
__device__ void freeze_set(const Dev& d, int s, int b, double q)
{
    const int ci = corr_index(d, s, b);
    if (ci < 0) return;
    SLOT(fset, d.nc)[ci] = 1;
    SLOT(fq, d.nc)[ci] = q;
}
__device__ void freeze_erase(const Dev& d, int s, int b)
{
    const int ci = (b >= 0 && b < d.n_bus) ? d.corr_of_bus[b] : -1;
    if (ci >= 0) SLOT(fset, d.nc)[ci] = 0;
}
__device__ void reset_vm(const Dev& d, int s, int bus, double vm)
{
    int& n = S1(vmr_n);
    if (n >= d.vcap) { S1(err) = ERR_VM_RESET_CAP; return; }
    SLOT(vmr_bus, d.vcap)[n] = bus;
    SLOT(vmr_vm, d.vcap)[n] = vm;
    ++n;
}
// Ls2gOuterModel::record: one decision of the loop being checked (tr_li), flushed by export_points
__device__ void rec(const Dev& d, int s, int act, bool taken, int element, double value, double limit)
{
    if (!d.trace) return;
    int& n = S1(tr_n);
    if (n >= d.tcap) { S1(err) = ERR_TRACE_CAP; return; }
    SLOT(tr_loop, d.tcap)[n] = S1(tr_li);
    SLOT(tr_act, d.tcap)[n] = act;
    SLOT(tr_taken, d.tcap)[n] = taken ? 1 : 0;
    SLOT(tr_el, d.tcap)[n] = element;
    SLOT(tr_val, d.tcap)[n] = value;
    SLOT(tr_lim, d.tcap)[n] = limit;
    SLOT(tr_outer, d.tcap)[n] = S1(nb_outer);
    ++n;
}
// OuterInjections::move_gen_target_p
__device__ double move_gen(const Dev& d, int s, int g, int bus, double mw)
{
    const double delta = mw - gen_in_force(d, s, g);
    double ds_re = delta / d.sn, ds_im = 0.;
    const bool pq = !d.gen_vreg[g];
    const double q_before = pq ? gen_target_q_at(d, g, gen_override(d, s, g)) : 0.;
    SLOT(gen_tp, d.n_gen)[g] = mw;
    for (int j = d.gen_rl_ptr[g]; j < d.gen_rl_ptr[g + 1]; ++j) SLOT(lim_dirty, d.n_bus)[d.gen_rl_bus[j]] = 1;
    if (pq) ds_im += (gen_target_q_at(d, g, gen_override(d, s, g)) - q_before) / d.sn;
    add_injection(d, s, bus, ds_re, ds_im);
    return delta;
}
__device__ double move_storage(const Dev& d, int s, int i, int bus, double mw)
{
    const double delta = sto_in_force(d, s, i) - mw;   // load convention
    SLOT(sto_tp, d.n_sto)[i] = mw;
    add_injection(d, s, bus, delta / d.sn, 0.);
    return delta;
}
// hvdc_recv / hvdc_flows
__device__ double hvdc_recv(double p_ctrl_abs, bool side1_ctrl, double lf1, double lf2, double r)
{
    const double lf_ctrl = side1_ctrl ? lf1 : lf2;
    const double lf_recv = side1_ctrl ? lf2 : lf1;
    const double line_in = (1. - lf_ctrl) * p_ctrl_abs;
    return (1. - lf_recv) * (line_in - r * line_in * line_in);
}
__device__ void hvdc_flows(int st, double raw, double lf1, double lf2, double r, double pmax12, double pmax21,
                           double& p1, double& p2)
{
    if (st == HVDC_STATUS_TRIPPED) { p1 = 0.; p2 = 0.; return; }
    if (st == 0) {
        if (raw >= 0.) { p1 = raw; p2 = -hvdc_recv(raw, true, lf1, lf2, r); }
        else           { p1 = -hvdc_recv(-raw, false, lf1, lf2, r); p2 = -raw; }
    } else if (st > 0) { p1 = pmax12; p2 = -hvdc_recv(pmax12, true, lf1, lf2, r); }
    else               { p1 = -hvdc_recv(pmax21, false, lf1, lf2, r); p2 = pmax21; }
}
// Ls2gOuterModel::mis_bus: V.conj(Ybus.V) - Sbus + the droop flows - the controllers' Q, with the
// frozen Q put back (against the slot's controls, as materialize last wrote them)
__device__ void mis_bus(const Dev& d, int s, const Row& rv, int b, double& mre, double& mim)
{
    const int k = rv.w(d, s, b);
    mre = rv.s_re(k) - rv.sb_re(k);
    mim = rv.s_im(k) - rv.sb_im(k);
    const bool hvdc_here = b >= 0 && b < d.n_bus && d.hvdc_at_bus[b];
    for (int e = 0; hvdc_here && e < d.n_hvdc; ++e) {
        const int b1 = d.hv_b1[e], b2 = d.hv_b2[e];
        if (b1 != b && b2 != b) continue;
        const double raw = d.hv_p0[e] + d.hv_k[e] * (va_of(d, s, rv, b1) - va_of(d, s, rv, b2));
        double p1, p2;
        hvdc_flows(SLOT(c_hvdc_status, d.n_hvdc)[e], raw, d.hv_lf1[e], d.hv_lf2[e], d.hv_r[e], d.hv_pmax12[e],
                   d.hv_pmax21[e], p1, p2);
        if (b1 == b) mre += p1;
        if (b2 == b) mre += p2;
    }
    for (int j = (b >= 0 && b < d.n_bus) ? d.vc_at_ptr[b] : 0; b >= 0 && b < d.n_bus && j < d.vc_at_ptr[b + 1]; ++j) {
        const int c = d.vc_at[j];
        const double q_in_sbus = SLOT(c_plan_held, d.n_ctrl)[c] ? d.vc_q_plan[c] : 0.;
        mim -= rv.vc_q(c) - q_in_sbus;
    }
    const int ci = (b >= 0 && b < d.n_bus) ? d.corr_of_bus[b] : -1;
    if (ci >= 0 && SLOT(fset, d.nc)[ci]) mim += SLOT(fq, d.nc)[ci];
    // the distributed slack's share (MultiSlack::adjust_mismatch), which the fold moved into the
    // targets after the solve and so left in the mismatch (the weights the solve ran with)
    if (S1(sa_sh) && b >= 0 && b < d.n_bus) {
        const int k = d.slack_idx[b];
        if (k >= 0) mre += S1(sa) * SLOT(c_slack_w, d.n_slack)[k];
    }
}

// ---- a distributed-slack Newton (Ls2gOuterModel::ds_share / ds_fold) -------------------------------
// the units the Newton shares on (unparked, a weight, a bus of the slack_absorbed column), their total weight
__device__ bool ds_shares(const Dev& d, int s, int k, double& w)
{
    w = 0.;
    if (!S1(ds_inn) || !SLOT(ds_present, d.n_units)[k]) return false;
    w = SLOT(ds_park, d.n_units)[k] ? 0. : d.ds_w[k];
    const int b = SLOT(ds_ubus, d.n_units)[k];
    return w > 0. && b >= 0 && b < d.n_bus && d.slack_idx[b] >= 0;
}
__device__ double ds_share(const Dev& d, int s)
{
    double total = 0., w;
    for (int k = 0; k < d.n_units; ++k) if (ds_shares(d, s, k, w)) total += w;
    return total;
}
// OuterLoopAlgo::_fold_slack_share: each unit's share of the solve's slack_absorbed into its target
__device__ void ds_fold(const Dev& d, int s, double sa)
{
    S1(sa) = sa;
    S1(sa_sh) = 0;
    if (sa == 0.) return;
    const double total = ds_share(d, s);
    if (!(total > 0.)) return;
    S1(sa_sh) = 1;
    const unsigned char* dset = SLOT(dset, d.nc);
    const double* dim = SLOT(dim, d.nc);
    double w;
    for (int k = 0; k < d.n_units; ++k) {
        if (!ds_shares(d, s, k, w)) continue;
        const int el = d.ds_el[k], bus = SLOT(ds_ubus, d.n_units)[k];
        const int ci = (bus >= 0 && bus < d.n_bus) ? d.corr_of_bus[bus] : -1;
        const double q_before = (ci >= 0 && dset[ci]) ? dim[ci] : 0.;
        const double mw = -sa * d.sn * w / total;   // generation convention
        if (d.ds_kind[k] == 0) move_gen(d, s, el, bus, gen_in_force(d, s, el) + mw);
        else move_storage(d, s, el, bus, sto_in_force(d, s, el) - mw);   // load convention
        const double q_after = (ci >= 0 && dset[ci]) ? dim[ci] : 0.;
        S1(unsq) += fabs(q_after - q_before) * d.sn;
    }
}

// ---- lightsim2grid's slack_redistribution::distribute (no overshoot), over the slot's units ------
// writes s_new (the new injections) and s_sat; returns not_distributed_mw, `all_saturated`
__device__ double distribute(const Dev& d, int s, double mismatch_mw, double eps_mw, bool& all_saturated)
{
    const unsigned char* present = SLOT(ds_present, d.n_units);
    const double* inj = SLOT(ds_inj, d.n_units);
    double* lo = SLOT(s_lo, d.n_units);
    double* hi = SLOT(s_hi, d.n_units);
    double* nmw = SLOT(s_new, d.n_units);
    unsigned char* active = SLOT(s_act, d.n_units);
    unsigned char* sat = SLOT(s_sat, d.n_units);
    all_saturated = false;
    int nb = 0;
    for (int k = 0; k < d.n_units; ++k) {
        sat[k] = 0;
        active[k] = 0;
        if (!present[k]) continue;
        ++nb;
        nmw[k] = inj[k];
    }
    if (nb == 0 || fabs(mismatch_mw) <= eps_mw) return 0.;
    const double inf = CUDART_INF;
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        active[k] = 1;
        lo[k] = isfinite(d.ds_min[k]) ? d.ds_min[k] : -inf;
        hi[k] = isfinite(d.ds_max[k]) ? d.ds_max[k] : inf;
        if (inj[k] < 0.) hi[k] = fmin(hi[k], 0.);
        else lo[k] = fmax(lo[k], 0.);
    }
    double remaining = mismatch_mw;
    int nb_active = nb, rounds = 0;
    while (nb_active > 0 && fabs(remaining) > eps_mw && rounds <= nb + 1) {
        ++rounds;
        double factor_sum = 0.;
        for (int k = 0; k < d.n_units; ++k) if (active[k]) factor_sum += d.ds_w[k];
        if (factor_sum <= 0.) break;
        double done = 0.;
        for (int k = 0; k < d.n_units; ++k) {
            if (!active[k]) continue;
            const double old = nmw[k];
            double cand = old + remaining * d.ds_w[k] / factor_sum;
            if (remaining > 0. && cand >= hi[k]) {
                cand = old > hi[k] ? old : hi[k];
                active[k] = 0; sat[k] = 1; --nb_active;
            } else if (remaining < 0. && cand <= lo[k]) {
                cand = old < lo[k] ? old : lo[k];
                active[k] = 0; sat[k] = 1; --nb_active;
            }
            done += cand - old;
            nmw[k] = cand;
        }
        remaining -= done;
    }
    // the units are DistributedSlack's own participants: never in the solve's slack (in_slack false)
    if (nb_active == 0) {
        all_saturated = true;
        for (int k = 0; k < d.n_units; ++k) sat[k] = 0;
    }
    return remaining;
}

// SlotState::ds_mark_solved: the units parked and their output as the next solve has them
__device__ void ds_mark_solved(const Dev& d, int s)
{
    for (int k = 0; k < d.n_units; ++k) {
        SLOT(ds_spark, d.n_units)[k] = SLOT(ds_park, d.n_units)[k];
        SLOT(ds_smw, d.n_units)[k] = SLOT(ds_cur, d.n_units)[k];
    }
}

// ---- DistributedSlackLoop::_check_in_newton ---------------------------------------------------------
__device__ int check_ds_in_newton(const Dev& d, int s, const Row& rv)
{
    const unsigned char* present = SLOT(ds_present, d.n_units);
    double* cur = SLOT(ds_cur, d.n_units);
    const double* inj = SLOT(ds_inj, d.n_units);
    // where the solve left the units: the fold moved their targets by their share
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        cur[k] = d.ds_kind[k] == 0 ? gen_in_force(d, s, d.ds_el[k]) : -sto_in_force(d, s, d.ds_el[k]);
    }
    // the solve just checked ran with the units parked as the last check left them
    if (d.keep_adj) ds_mark_solved(d, s);
    double mre, mim;
    mis_bus(d, s, rv, d.slack_bus, mre, mim);
    double mismatch = mre * d.sn;
    if (isnan(mismatch)) mismatch = 0.;
    double remaining = mismatch;
    int n_units = 0;
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        ++n_units;
        remaining += cur[k] - inj[k];
    }
    bool all_sat;
    const double not_distributed = distribute(d, s, remaining, d.ds_eps_mw, all_sat);
    unsigned char* sat = SLOT(s_sat, d.n_units);
    if (all_sat) for (int k = 0; k < d.n_units; ++k) if (present[k]) sat[k] = 1;
    const double residue = n_units == 0 ? remaining : not_distributed;
    if (d.ds_fail && fabs(residue) > d.ds_eps_mw) {
        rec(d, s, TA_FAIL_RESIDUE, true, -1, residue, d.ds_eps_mw);
        return FAILED;
    }
    double moved = 0.;
    const double* nmw = SLOT(s_new, d.n_units);
    unsigned char* parked = SLOT(ds_park, d.n_units);
    unsigned char* parked_high = SLOT(ds_parkh, d.n_units);
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        // a unit at a bound leaves the Newton's share; one back inside its bounds rejoins it
        if ((sat[k] ? 1 : 0) != parked[k]) rec(d, s, sat[k] ? TA_PARK : TA_UNPARK, true, d.ds_el[k], cur[k], nmw[k]);
        const int ub = SLOT(ds_ubus, d.n_units)[k];
        moved += fabs(d.ds_kind[k] == 0 ? move_gen(d, s, d.ds_el[k], ub, nmw[k])
                                        : move_storage(d, s, d.ds_el[k], ub, -nmw[k]));
        cur[k] = nmw[k];
        parked[k] = sat[k] ? 1 : 0;
        parked_high[k] = remaining > 0. ? 1 : 0;
    }
    const double threshold = d.ds_moved_fraction * d.ds_eps_mw;
    bool unstable = moved > threshold;
    rec(d, s, TA_UNITS_MOVED, unstable, -1, moved, threshold);
    const double unsolved_q = S1(unsq);
    S1(unsq) = 0.;
    if (unsolved_q > threshold) {
        rec(d, s, TA_TARGET_Q_FOLLOWED, true, -1, unsolved_q, threshold);
        unstable = true;
    }
    // re-solved with the new weights
    if (unstable && d.keep_adj) ds_mark_solved(d, s);
    return unstable ? UNSTABLE : STABLE;
}

// ---- DistributedSlackLoop::_check (a single-slack Newton) -----------------------------------------
__device__ int check_ds(const Dev& d, int s, const Row& rv)
{
    if (S1(ds_inn)) {
        int n_present = 0;
        for (int k = 0; k < d.n_units; ++k) n_present += SLOT(ds_present, d.n_units)[k] ? 1 : 0;
        if (n_present > 0) return check_ds_in_newton(d, s, rv);
    }
    double mre, mim;
    mis_bus(d, s, rv, d.slack_bus, mre, mim);
    const double mismatch = mre * d.sn;
    const bool triggered = fabs(mismatch) > d.ds_mismatch_mw && fabs(mismatch) > d.ds_eps_mw;
    rec(d, s, TA_DISTRIBUTE, triggered, -1, mismatch, fmax(d.ds_mismatch_mw, d.ds_eps_mw));
    if (!triggered) return STABLE;
    const unsigned char* present = SLOT(ds_present, d.n_units);
    double* cur = SLOT(ds_cur, d.n_units);
    const double* inj = SLOT(ds_inj, d.n_units);
    double remaining = mismatch;
    int n_units = 0;
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        ++n_units;
        remaining += cur[k] - inj[k];
    }
    bool all_sat;
    const double not_distributed = distribute(d, s, remaining, d.ds_eps_mw, all_sat);
    const double residue = n_units == 0 ? remaining : not_distributed;
    if (d.ds_fail && fabs(residue) > d.ds_eps_mw) {
        rec(d, s, TA_FAIL_RESIDUE, true, -1, residue, d.ds_eps_mw);
        return FAILED;
    }
    double moved = 0.;
    const double* nmw = SLOT(s_new, d.n_units);
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        const int ub = SLOT(ds_ubus, d.n_units)[k];
        moved += fabs(d.ds_kind[k] == 0 ? move_gen(d, s, d.ds_el[k], ub, nmw[k])
                                        : move_storage(d, s, d.ds_el[k], ub, -nmw[k]));
        cur[k] = nmw[k];
    }
    const bool unstable = moved > d.ds_moved_fraction * d.ds_eps_mw;
    rec(d, s, TA_UNITS_MOVED, unstable, -1, moved, d.ds_moved_fraction * d.ds_eps_mw);
    return unstable ? UNSTABLE : STABLE;
}

// ---- HvdcAcEmulationLimitsLoop::_check (lightsim2grid's check_hvdc_p_violations, tol 0) ------------
__device__ int check_hvdc(const Dev& d, int s, const Row& rv)
{
    const unsigned char* present = SLOT(hl_present, d.n_hl);
    int* dir = SLOT(hl_dir, d.n_hl);
    int* regime_of = SLOT(s_hl, d.n_hl);        // the regime its violation asks, 99: none
    const unsigned char* masked = SLOT(masked, d.n_bus);
    bool any = false;
    for (int k = 0; k < d.n_hl; ++k) {
        regime_of[k] = 99;
        if (!present[k]) continue;
        const int b1 = d.hl_b1[k], b2 = d.hl_b2[k];
        if (b1 >= d.n_bus || b2 >= d.n_bus) continue;
        if ((b1 >= 0 && masked[b1]) || (b2 >= 0 && masked[b2])) continue;
        const double p_mw = d.hl_p0[k] + d.hl_k[k] * (va_of(d, s, rv, b1) - va_of(d, s, rv, b2));
        if (!isfinite(p_mw)) continue;
        if (d.trace) SLOT(s_hp, d.n_hl)[k] = p_mw;
        const double pmax12 = d.hl_pmax12[k], pmax21 = d.hl_pmax21[k];
        if (dir[k] != 0) {
            const bool inside = (p_mw > 0. && p_mw < pmax12) || (p_mw < 0. && -p_mw < pmax21);
            if (inside) regime_of[k] = 0;                         // HVDC_AC_EMULATION_RELEASE
            else if (dir[k] > 0 && p_mw < 0.) regime_of[k] = -1;  // HIGH_P, side 2
            else if (dir[k] < 0 && p_mw > 0.) regime_of[k] = 1;   // HIGH_P, side 1
        } else if (isfinite(pmax12) && p_mw > pmax12) {
            regime_of[k] = 1;
        } else if (isfinite(pmax21) && -p_mw > pmax21) {
            regime_of[k] = -1;
        }
        any = any || regime_of[k] != 99;
    }
    if (!any) return STABLE;
    bool changed = false;
    for (int v = 0; v < d.n_hl; ++v) {           // the violations, in the plan's order
        const int regime = regime_of[v];
        if (regime == 99) continue;
        // the violation's flow and limit (check_hvdc_p_violations'), before any regime changes
        const double p_mw = d.trace ? SLOT(s_hp, d.n_hl)[v] : 0.;
        const bool fwd = regime > 0 || (regime == 0 && dir[v] > 0);
        const double value = fwd ? p_mw : -p_mw, limit = fwd ? d.hl_pmax12[v] : d.hl_pmax21[v];
        bool acted = false;
        for (int k = 0; k < d.n_hl; ++k) {
            if (!present[k] || d.hl_id[k] != d.hl_id[v] || dir[k] == regime) continue;
            const int e = d.hl_entry[k];
            if (e < 0) continue;                  // no regime control for it
            dir[k] = regime;
            SLOT(hvdc_regime, d.n_hvdc)[e] = regime;
            changed = true;
            acted = true;
        }
        rec(d, s, regime == 0 ? TA_RELEASE : TA_SATURATE, acted, d.hl_id[v], value, limit);
    }
    return changed ? UNSTABLE : STABLE;
}

// ---- VoltageMonitoringLoop::_check (lightsim2grid's check_svc_standby_violations, tol 0) -----------
__device__ int check_vm(const Dev& d, int s, const Row& rv)
{
    const unsigned char* masked = SLOT(masked, d.n_bus);
    double* svc_on = SLOT(svc_on, d.n_ctrl);
    bool changed = false;
    for (int m = 0; m < d.n_mon; ++m) {
        const int svc = d.mon_svc[m];
        const int c = d.svc_ctrl[svc];
        if (c >= 0 && isfinite(svc_on[c])) continue;     // switched on
        const int reg = d.mon_reg[m], own = d.mon_bus[m];
        if (reg < 0 || reg >= d.n_bus) continue;
        if (masked[reg]) continue;
        if (own >= 0 && own < d.n_bus && masked[own]) continue;
        const double vm = vm_of(d, s, rv, reg);
        if (!isfinite(vm)) continue;
        double target, thr;
        if (vm < d.mon_low[m]) { target = d.svc_low_t[svc]; thr = d.mon_low[m]; }
        else if (vm > d.mon_high[m]) { target = d.svc_high_t[svc]; thr = d.mon_high[m]; }
        else continue;
        const bool can = isfinite(target) && c >= 0;
        rec(d, s, TA_SWITCH_ON, can, svc, vm * d.mon_vn[m], thr * d.mon_vn[m]);
        if (!can) continue;
        svc_on[c] = target;
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

// ---- ReactiveLimitsLoop ----------------------------------------------------------------------------
__device__ bool rl_at_base(const Dev& d, int s, int k)
{
    const int b = d.rl_bus[k];
    return SLOT(rl_hb_s, d.rl_n)[k] && S1(lim_base) && b >= 0 && b < d.n_bus && !SLOT(lim_dirty, d.n_bus)[b];
}
__device__ double rl_bus_q(const Dev& d, int s, const Row& rv, int k)
{
    double mre, mim;
    mis_bus(d, s, rv, d.rl_bus[k], mre, mim);
    double q = mim;
    if (!d.rl_local[k])
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j)
            if (!SLOT(ctrl_off, d.n_ctrl)[d.rl_ctrl[j]]) q += rv.vc_q(d.rl_ctrl[j]);
    return q * d.sn;
}
__device__ void rl_limits(const Dev& d, int s, const Row* rv, int k, double& q_min, double& q_max)
{
    const bool at_base = rl_at_base(d, s, k);
    if (at_base) {
        q_min = d.rl_bmin[k];
        q_max = d.rl_bmax[k];
        if (d.rl_local[k] || rv == nullptr) return;
    }
    if (!at_base) {
        q_min = d.rl_min_q[k];
        q_max = d.rl_max_q[k];
        for (int j = d.rl_gen_ptr[k]; j < d.rl_gen_ptr[k + 1]; ++j) {
            const int gid = d.rl_gen[j];
            if (SLOT(rl_goff, 0)[j]) continue;   // the row's plan does not have it on this bus
            double lo, hi;
            gen_limits(d, gid, gen_override(d, s, gid), lo, hi);
            q_min += lo;
            q_max += hi;
        }
    }
    if (d.rl_svc_ptr[k] == d.rl_svc_ptr[k + 1] || rv == nullptr) return;
    const double v2_sn = norm_of(d, s, *rv, d.rl_bus[k]) * d.sn;
    for (int j = d.rl_svc_ptr[k]; j < d.rl_svc_ptr[k + 1]; ++j) {
        const int sid = d.rl_svc[j];
        q_min += d.svc_b_min[sid] * v2_sn;
        q_max += d.svc_b_max[sid] * v2_sn;
    }
}
__device__ double rl_controller_limit(const Dev& d, int s, const Row& rv, int c, bool max)
{
    const int el = d.vc_elem[c];
    switch (d.vc_kind[c]) {
        case VC_GEN: {
            double lo, hi;
            gen_limits(d, el, gen_override(d, s, el), lo, hi);
            return max ? hi : lo;
        }
        case VC_SVC: {
            const double v2_sn = norm_of(d, s, rv, d.vc_bus[c]) * d.sn;
            return (max ? d.svc_b_max[el] : d.svc_b_min[el]) * v2_sn;
        }
        case VC_HVDC_1: return max ? d.st_max1[el] : d.st_min1[el];
        case VC_HVDC_2: return max ? d.st_max2[el] : d.st_min2[el];
        default: return 0.;
    }
}
// rl_groups_holding, into s_hold
__device__ void rl_groups_holding(const Dev& d, int s)
{
    unsigned char* holding = SLOT(s_hold, d.n_grp);
    unsigned char* sloped = SLOT(s_slope, d.n_grp);
    const double* held_q = SLOT(held_q, d.n_ctrl);
    const double* svc_on = SLOT(svc_on, d.n_ctrl);
    for (int g = 0; g < d.n_grp; ++g) { holding[g] = 0; sloped[g] = 0; }
    for (int c = 0; c < d.n_ctrl; ++c) {
        const int g = d.vc_group[c];
        if (d.vc_slope[c] != 0.) sloped[g] = 1;
        if (isfinite(held_q[c])) continue;
        if (d.vc_plan_held[c]) {
            const bool switched_on = d.vc_kind[c] == VC_SVC && isfinite(svc_on[c]);
            if (!switched_on) continue;
        }
        holding[g] = 1;
    }
    for (int g = 0; g < d.n_grp; ++g) holding[g] = holding[g] && !sloped[g];
}
// the slopes recorded for bus k's generators forgotten (st.rl_slopes.erase(k))
__device__ void rl_clear_slopes(const Dev& d, int s, int k)
{
    unsigned char* set = SLOT(slp_set, d.n_gen);
    if (!d.rl_local[k]) {
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) {
            const int c = d.rl_ctrl[j];
            if (d.vc_kind[c] == VC_GEN) set[d.vc_elem[c]] = 0;
        }
    } else {
        for (int j = d.rl_gen_ptr[k]; j < d.rl_gen_ptr[k + 1]; ++j)
            if (!SLOT(rl_goff, 0)[j]) set[d.rl_gen[j]] = 0;
    }
}
// a generator frozen at a limit of its curve follows it: what the adjoint adds to its gen_p gradient
__device__ void rl_record_slope(const Dev& d, int s, int g, bool max)
{
    const double slope = gen_limit_slope(d, g, gen_override(d, s, g), max);
    if (slope == 0.) return;
    SLOT(slp_set, d.n_gen)[g] = 1;
    SLOT(slp, d.n_gen)[g] = slope / d.sn;
}
__device__ void rl_freeze(const Dev& d, int s, const Row& rv, int k, double q_mvar, int state)
{
    double* held_q = SLOT(held_q, d.n_ctrl);
    if (d.keep_adj) rl_clear_slopes(d, s, k);
    if (!d.rl_local[k]) {
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) {
            const int c = d.rl_ctrl[j];
            if (SLOT(ctrl_off, d.n_ctrl)[c]) continue;   // the row trips it: not in its plan
            double q = rl_controller_limit(d, s, rv, c, state > 0);
            if (d.vc_kind[c] == VC_GEN) {
                if (SLOT(rl_real, d.rl_n)[k]) q = d.gen_target_q[d.vc_elem[c]];   // the robust mode's
                else if (d.keep_adj) rl_record_slope(d, s, d.vc_elem[c], state > 0);
            }
            held_q[c] = q / d.sn;
        }
        SLOT(rl_state, d.rl_n)[k] = state;
        SLOT(rl_fq, d.rl_n)[k] = q_mvar;
        return;
    }
    if (!SLOT(rl_real, d.rl_n)[k] && d.keep_adj)
        for (int j = d.rl_gen_ptr[k]; j < d.rl_gen_ptr[k + 1]; ++j)
            if (!SLOT(rl_goff, 0)[j]) rl_record_slope(d, s, d.rl_gen[j], state > 0);
    if (d.rl_pin[k] < 0) { S1(err) = ERR_NO_PIN; return; }
    freeze_set(d, s, d.rl_bus[k], q_mvar / d.sn);
    SLOT(pq, d.n_pin)[d.rl_pin[k]] = 1;
    SLOT(rl_state, d.rl_n)[k] = state;
    SLOT(rl_fq, d.rl_n)[k] = q_mvar;
}
__device__ void rl_release(const Dev& d, int s, int k)
{
    if (d.keep_adj) rl_clear_slopes(d, s, k);
    if (!d.rl_local[k]) {
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j)
            if (!SLOT(ctrl_off, d.n_ctrl)[d.rl_ctrl[j]]) SLOT(held_q, d.n_ctrl)[d.rl_ctrl[j]] = CUDART_NAN;
    } else {
        freeze_erase(d, s, d.rl_bus[k]);
        SLOT(pq, d.n_pin)[d.rl_pin[k]] = 0;
        reset_vm(d, s, d.rl_bus[k], SLOT(rl_tvm_s, d.rl_n)[k]);
    }
    SLOT(rl_state, d.rl_n)[k] = 0;
    SLOT(rl_real, d.rl_n)[k] = 0;
    SLOT(rl_fq, d.rl_n)[k] = 0.;
}
// "a is stronger than b" (Ls2gOuterModel::check_rl's keep-the-strongest-PV order)
__device__ bool rl_stronger(const Dev& d, int s, int a, int b)
{
    if (d.rl_nv[a] != d.rl_nv[b]) return d.rl_nv[a] > d.rl_nv[b];
    const double pa = SLOT(rl_tp, d.rl_n)[a], pb = SLOT(rl_tp, d.rl_n)[b];
    if (pa != pb) return pa > pb;
    return d.rl_bus[a] < d.rl_bus[b];
}
// ReactiveLimitsLoop::_evaluate of one controller bus k (Ls2gOuterModel::check_rl's loop body), into s_cls[k] /
// s_qlim[k]: 0 not checked, 1 / 2 PV -> PQ at its min / max, 3 / 4 the robust mode's (at its
// target Q), 5 stays PV, 6 PQ -> PV, 7 kept PQ, 8 kept PQ with its limit moved. Reads only: the
// buses of a slot are evaluated in parallel (s_hold computed before).
__device__ void eval_rl_bus(const Dev& d, int s, const Row& rv, int k)
{
    int& cls = SLOT(s_cls, d.rl_n)[k];
    double& qlim = SLOT(s_qlim, d.rl_n)[k];
    cls = 0;
    if (!SLOT(rl_present, d.rl_n)[k]) return;
    // a bus whose voltage control another loop suspended (TransformerVoltageControl)
    if (d.has_tvc && SLOT(tv_susp, d.n_bus)[d.rl_bus[k]]) return;
    const double eps = d.rl_max_mismatch * d.rl_base_mva;
    const double v_low = d.rl_vmin * d.rl_margin;
    const double v_high = d.rl_vmax / d.rl_margin;
    double target_vm = SLOT(rl_tvm_s, d.rl_n)[k];
    if (d.rl_mon[k] >= 0) {
        const int c = d.svc_ctrl[d.rl_mon[k]];
        if (c < 0 || !isfinite(SLOT(svc_on, d.n_ctrl)[c])) return;   // still idle
        target_vm = SLOT(svc_on, d.n_ctrl)[c];
    }
    double q_min, q_max;
    rl_limits(d, s, &rv, k, q_min, q_max);
    const int state = SLOT(rl_state, d.rl_n)[k];
    if (state == 0) {
        const double q = rl_bus_q(d, s, rv, k);
        if (d.trace) SLOT(s_tv, d.rl_n)[k] = q;   // the record's value; its limit is qlim
        if (q < q_min - eps) { cls = 1; qlim = q_min; }
        else if (q > q_max + eps) { cls = 2; qlim = q_max; }
        else if (d.rl_robust && d.rl_reg[k] != d.rl_bus[k]) {
            const double v = vm_of(d, s, rv, d.rl_bus[k]);
            if (v < v_low)       { cls = 3; qlim = SLOT(rl_tq_s, d.rl_n)[k]; }
            else if (v > v_high) { cls = 4; qlim = SLOT(rl_tq_s, d.rl_n)[k]; }
            else cls = 5;
        } else {
            cls = 5;
        }
        return;
    }
    const bool held_by_group = d.rl_group[k] >= 0 && SLOT(s_hold, d.n_grp)[d.rl_group[k]];
    const double vm_read = vm_of(d, s, rv, d.rl_reg[k]);
    const double vm = held_by_group ? target_vm : vm_read;
    if (d.trace) {   // the records' value and limit (kV), and whether a group holds the bus
        const double vn = d.rl_nv[k];
        SLOT(s_tl, d.rl_n)[k] = target_vm * vn;
        SLOT(s_gh, d.rl_n)[k] = held_by_group ? 1 : 0;
        SLOT(s_tv, d.rl_n)[k] = (state < 0 ? vm < target_vm : vm > target_vm) ? vm * vn : vm_read * vn;
    }
    if (state < 0 ? vm < target_vm : vm > target_vm) { cls = 6; return; }
    const double q_lim = state < 0 ? q_min : q_max;
    cls = (!SLOT(rl_real, d.rl_n)[k] && fabs(q_lim - SLOT(rl_fq, d.rl_n)[k]) > eps) ? 8 : 7;
}

// ReactiveLimitsLoop::_check on the evaluated buses (Ls2gOuterModel::check_rl's second half; one thread): the
// lists in bus order, as the host builds them
__device__ int check_rl_apply(const Dev& d, int s, const Row& rv)
{
    int* pq_k = SLOT(s_pq_k, d.rl_n);
    int* pq_lo = SLOT(s_pq_lo, d.rl_n);
    unsigned char* pq_real = SLOT(s_pq_real, d.rl_n);
    double* pq_lim = SLOT(s_pq_lim, d.rl_n);
    int* pv_k = SLOT(s_pv_k, d.rl_n);
    int* mv_k = SLOT(s_mv_k, d.rl_n);
    const int* cls = SLOT(s_cls, d.rl_n);
    const double* qlim = SLOT(s_qlim, d.rl_n);
    int n_pq = 0, n_pv = 0, n_mv = 0, remaining_pv = 0;
    for (int k = 0; k < d.rl_n; ++k) {
        switch (cls[k]) {
            case 1: case 2: case 3: case 4:
                pq_k[n_pq] = k; pq_lo[n_pq] = (cls[k] == 1 || cls[k] == 3) ? 1 : 0;
                pq_real[n_pq] = cls[k] >= 3 ? 1 : 0; pq_lim[n_pq] = qlim[k]; ++n_pq;
                break;
            case 5: ++remaining_pv; break;
            case 6: pv_k[n_pv++] = k; break;
            case 8: mv_k[n_mv++] = k; break;
            default: break;
        }
    }
    const double v_low = d.rl_vmin * d.rl_margin;
    const double v_high = d.rl_vmax / d.rl_margin;
    bool changed = false;
    // the records: the value each bus was evaluated with, its limit (a PV bus' qlim, a frozen one's s_tl)
    auto rec_k = [&](int act, bool taken, int k, double limit) {
        rec(d, s, act, taken, d.rl_bus[k], SLOT(s_tv, d.rl_n)[k], limit);
    };
    if (d.trace)
        for (int k = 0; k < d.rl_n; ++k)
            if (cls[k] == 7 || cls[k] == 8)
                rec_k(SLOT(s_gh, d.rl_n)[k] ? TA_KEPT_PQ_GROUP_HOLDS : TA_KEPT_PQ, false, k, SLOT(s_tl, d.rl_n)[k]);
    int skip = -1;
    if (n_pq > 0 && remaining_pv == 0) {
        skip = 0;
        for (int j = 1; j < n_pq; ++j) if (rl_stronger(d, s, pq_k[j], pq_k[skip])) skip = j;
        rec_k(TA_KEPT_PV_STRONGEST, false, pq_k[skip], pq_lim[skip]);
    }
    for (int j = 0; j < n_pq; ++j) {
        if (j == skip) continue;
        const int k = pq_k[j];
        rec_k(pq_real[j] ? TA_PV_TO_PQ_UNREALISTIC : TA_PV_TO_PQ, true, k, pq_lim[j]);
        SLOT(rl_real, d.rl_n)[k] = pq_real[j];
        rl_freeze(d, s, rv, k, pq_lim[j], pq_lo[j] ? -1 : +1);
        ++SLOT(rl_nb, d.rl_n)[k];
        if (d.rl_robust && d.rl_reg[k] != d.rl_bus[k]) {
            const double v = vm_of(d, s, rv, d.rl_bus[k]);
            if (pq_real[j] || v < v_low || v > v_high) reset_vm(d, s, d.rl_bus[k], d.rl_restart);
        }
        changed = true;
    }
    for (int j = 0; j < n_pv; ++j) {
        const int k = pv_k[j];
        if (SLOT(rl_nb, d.rl_n)[k] >= d.rl_max_switch) {
            rec_k(TA_KEPT_PQ_MAX_SWITCH, false, k, SLOT(s_tl, d.rl_n)[k]);
            continue;
        }
        rec_k(TA_PQ_TO_PV, true, k, SLOT(s_tl, d.rl_n)[k]);
        rl_release(d, s, k);
        changed = true;
    }
    for (int j = 0; j < n_mv; ++j) {
        const int k = mv_k[j];
        double q_min, q_max;
        rl_limits(d, s, &rv, k, q_min, q_max);
        const int state = SLOT(rl_state, d.rl_n)[k];
        rec(d, s, TA_LIMIT_MOVED, true, d.rl_bus[k], state < 0 ? q_min : q_max, SLOT(rl_fq, d.rl_n)[k]);
        rl_freeze(d, s, rv, k, state < 0 ? q_min : q_max, state);
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

// ---- ShuntVoltageControlLoop (Ls2gOuterModel::svc_dispatch / sc_set_sections / check_svc) -------------
__device__ double sh_b_at(const Dev& d, int j, int count) { return d.sh_b[d.sh_b_ptr[j] + count]; }
// the susceptance b over controller e's shunts: the largest first (stable), each its share of what is
// left, rounded to its closest section (the current count unless another is strictly closer), into s_cnt
__device__ void svc_dispatch(const Dev& d, int s, int e, double b)
{
    const int j0 = d.sc_sh_ptr[e], m = d.sc_sh_ptr[e + 1] - j0;
    int* ord = SLOT(s_ord, d.n_sc_sh) + j0;
    int* cnt = SLOT(s_cnt, d.n_sc_sh) + j0;
    double* mag = SLOT(s_mag, d.n_sc_sh) + j0;
    for (int i = 0; i < m; ++i) {
        const int j = j0 + i;
        mag[i] = fabs(sh_b_at(d, j, d.sh_max[j]) - sh_b_at(d, j, d.sh_min[j]));
        cnt[i] = d.sh_count0[j];
        // insertion into the order, after every one at least as large: std::stable_sort's
        int p = i;
        while (p > 0 && mag[ord[p - 1]] < mag[i]) { ord[p] = ord[p - 1]; --p; }
        ord[p] = i;
    }
    double residue = b;
    int remaining = m;
    for (int q = 0; q < m; ++q) {
        const int i = ord[q], j = j0 + i;
        const double share = residue / static_cast<double>(remaining--);
        int best = cnt[i];
        double best_distance = fabs(share - sh_b_at(d, j, best));
        for (int count = d.sh_min[j]; count <= d.sh_max[j]; ++count) {
            const double distance = fabs(share - sh_b_at(d, j, count));
            if (distance < best_distance) { best = count; best_distance = distance; }
        }
        cnt[i] = best;
        residue -= sh_b_at(d, j, best);
    }
}
// ShuntControl::set_sections: the counts (clamped) and the susceptance they give, applied once
__device__ void sc_set_sections(const Dev& d, int s, int e)
{
    const int j0 = d.sc_sh_ptr[e], j1 = d.sc_sh_ptr[e + 1];
    const int* cnt = SLOT(s_cnt, d.n_sc_sh);
    int* counts = SLOT(sc_cnt, d.n_sc_sh);
    double b = 0.;
    for (int j = j0; j < j1; ++j) {
        const int count = max(0, min(cnt[j], d.sh_max[j]));
        counts[j] = count;
        if (d.sh_status[j]) b += sh_b_at(d, j, count);
    }
    SLOT(sc_bnew, d.n_sc)[e] = b;
    SLOT(sc_pend, d.n_sc)[e] = 1;
}
__device__ int check_svc(const Dev& d, int s, const Row& rv, int iteration)
{
    if (iteration != 0 || d.n_svg == 0) return STABLE;
    // nothing to do without a controller the inner algorithm can act on
    bool supported = false;
    for (int k = 0; k < d.svg_ptr[d.n_svg]; ++k) supported = supported || d.svg_e[k] >= 0;
    if (!supported) return STABLE;
    bool any = false;
    for (int g = 0; g < d.n_svg; ++g) {
        if (SLOT(sv_hid, d.n_svg)[g]) continue;
        for (int k = d.svg_ptr[g]; k < d.svg_ptr[g + 1]; ++k) {
            const int e = d.svg_e[k];
            if (e < 0) continue;
            SLOT(sc_ctl, d.n_sc)[e] = 0;
            const double b = rv.sc_b(e);
            svc_dispatch(d, s, e, b);
            sc_set_sections(d, s, e);
            // the susceptance the Newton solved for (pu), rounded to sections
            rec(d, s, TA_ROUND_SECTIONS, true, d.svg_bus[k], b, CUDART_NAN);
            any = true;
        }
    }
    return any ? UNSTABLE : STABLE;
}

// ---- BranchControl (Ls2gOuterModel::bc_retap / bc_block_at / bc_current) ----------------------------
// std::complex<double>'s arithmetic, written out (the same operations, so the same bits)
struct Cx { double re, im; };
__device__ Cx cx_mul(Cx a, Cx b) { return Cx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
__device__ Cx cx_scale(Cx a, double f) { return Cx{a.re * f, a.im * f}; }
__device__ Cx cx_add(Cx a, Cx b) { return Cx{a.re + b.re, a.im + b.im}; }
// BranchControl::_retap (set_tap / set_ratio_tap ignore a position off the table)
__device__ void bc_retap(const Dev& d, int s, int e, int rpos, int pos)
{
    int* cur_pos = SLOT(bc_pos, d.n_bc);
    int* cur_rpos = SLOT(bc_rpos, d.n_bc);
    // (without a tap changer its range is its one position: any other is off the table)
    if (pos != cur_pos[e] && (!d.bc_has_ptc[e] || pos < d.bc_plo[e] || pos > d.bc_phi[e])) pos = cur_pos[e];
    if (rpos != cur_rpos[e] && (rpos < d.bc_rlo[e] || rpos > d.bc_rhi[e])) rpos = cur_rpos[e];
    cur_pos[e] = pos;
    cur_rpos[e] = rpos;
    const int np = d.bc_phi[e] - d.bc_plo[e] + 1;
    const int k = d.bc_tab_ptr[e] + (rpos - d.bc_rlo[e]) * np + (pos - d.bc_plo[e]);
    if (d.bc_has_ptc[e]) SLOT(bc_atap, d.n_bc)[e] = d.bc_tab_shift[k];
    SLOT(bc_rtap, d.n_bc)[e] = d.bc_tab_ratio[k];
    double* tap = SLOT(bc_tap, 8 * d.n_bc) + 8 * e;
    for (int q = 0; q < 8; ++q) tap[q] = d.bc_tab_block[8 * k + q];
}
// BranchControl::_block_at: the block at shift a and ratio rho, from the one at the taps
__device__ void bc_block_at(const Dev& d, int s, int e, double a, double rho, Cx blk[4])
{
    if (SLOT(bc_out, d.n_bc)[e]) { for (int k = 0; k < 4; ++k) blk[k] = Cx{0., 0.}; return; }
    const double* tap = SLOT(bc_tap, 8 * d.n_bc) + 8 * e;
    const double dd = d.bc_sign[e] * (a - SLOT(bc_atap, d.n_bc)[e]);
    const Cx rot{cos(dd), sin(dd)};
    const double f = pow(rho / SLOT(bc_rtap, d.n_bc)[e], d.bc_rexp[e]);
    blk[0] = cx_scale(cx_scale(Cx{tap[0], tap[1]}, f), f);
    blk[1] = cx_mul(cx_scale(Cx{tap[2], tap[3]}, f), rot);
    blk[2] = cx_mul(cx_scale(Cx{tap[4], tap[5]}, f), Cx{rot.re, -rot.im});
    blk[3] = Cx{tap[6], tap[7]};
}
// BranchControl::current on the block stamped now: |I| on `side` (pu) and its derivative with the shift
__device__ void bc_current(const Dev& d, int s, const Row& rv, int e, int side, double& i_pu, double& di_da)
{
    Cx y[4];
    bc_block_at(d, s, e, rv.bc_a(e), rv.bc_rho(e), y);
    const int k1 = rv.w(d, s, d.bc_b1[e]), k2 = rv.w(d, s, d.bc_b2[e]);
    const Cx V1{rv.v_re(k1), rv.v_im(k1)}, V2{rv.v_re(k2), rv.v_im(k2)};
    const double sg = d.bc_sign[e];
    Cx I, dI;
    if (side == 1) {
        I = cx_add(cx_mul(y[0], V1), cx_mul(y[1], V2));
        dI = cx_mul(cx_mul(cx_scale(Cx{0., 1.}, sg), y[1]), V2);
    } else {
        I = cx_add(cx_mul(y[2], V1), cx_mul(y[3], V2));
        dI = cx_mul(cx_mul(cx_scale(Cx{-0., -1.}, sg), y[2]), V1);
    }
    i_pu = hypot(I.re, I.im);
    di_da = i_pu > 0. ? (I.re * dI.re - (-I.im) * dI.im) / i_pu : 0.;
}

// ---- PhaseControlLoop::_check (Ls2gOuterModel::check_pc) ---------------------------------------------
__device__ double bc_alpha_at(const Dev& d, int e, int pos) { return d.bc_alpha[d.bc_alpha_ptr[e] + pos - d.bc_plo[e]]; }
__device__ int check_pc(const Dev& d, int s, const Row& rv, int iteration)
{
    const unsigned char* present = SLOT(pc_pres, d.n_pc);
    bool any = false;
    for (int k = 0; k < d.n_pc; ++k) any = any || (present[k] && d.pc_e[k] >= 0);
    if (!any) return STABLE;
    int* pos_of = SLOT(bc_pos, d.n_bc);
    if (iteration == 0) {
        // the active power controllers switched off, their shift rounded to the closest tap
        for (int k = 0; k < d.n_pc; ++k) {
            if (!present[k] || d.pc_mode[k] != 0 || !SLOT(pc_reg, d.n_pc)[k]) continue;
            const int e = d.pc_e[k];
            if (e < 0) continue;
            SLOT(bc_ctl, d.n_bc)[e] = 0;
            const double a = rv.bc_a(e);
            int best = pos_of[e];
            double best_distance = fabs(a - bc_alpha_at(d, e, best));
            for (int pos = d.bc_plo[e]; pos <= d.bc_phi[e]; ++pos) {
                const double distance = fabs(a - bc_alpha_at(d, e, pos));
                if (distance < best_distance) { best = pos; best_distance = distance; }
            }
            SLOT(bc_mov, d.n_bc)[e] = 1;
            SLOT(bc_rq, d.n_bc)[e] = best;
            rec(d, s, TA_ROUND_TAP, true, d.pc_trafo[k], a, bc_alpha_at(d, e, best));
        }
        return CHECK_AGAIN;
    }
    // the current limiters above their limit move one tap, the way that lowers the current
    bool moved = false;
    for (int k = 0; k < d.n_pc; ++k) {
        if (!present[k] || d.pc_mode[k] != 1) continue;
        const int e = d.pc_e[k];
        if (e < 0) continue;
        double i_pu, di_da;
        bc_current(d, s, rv, e, d.pc_side[k], i_pu, di_da);
        const double i_a = i_pu * d.sn * 1000. / (sqrt(3.) * d.pc_vn_kv[k]);
        const double target = d.pc_target[k];
        if (!(target < i_a)) {
            rec(d, s, TA_KEPT_TAP, false, d.pc_trafo[k], i_a, target);
            continue;
        }
        const bool increase = !(di_da > 0.);
        const double a = rv.bc_a(e);
        int pos = pos_of[e];
        const int old_pos = pos;
        if (pos < d.bc_phi[e]) {
            const double next = bc_alpha_at(d, e, pos + 1);
            if ((increase && next > a) || (!increase && next < a)) ++pos;
        }
        if (pos > d.bc_plo[e]) {
            const double previous = bc_alpha_at(d, e, pos - 1);
            if ((increase && previous > a) || (!increase && previous < a)) --pos;
        }
        rec(d, s, TA_MOVE_TAP, pos != old_pos, d.pc_trafo[k], i_a, target);
        if (pos != old_pos) {
            SLOT(bc_mov, d.n_bc)[e] = 1;
            SLOT(bc_rq, d.n_bc)[e] = pos;
            moved = true;
        }
    }
    return moved ? UNSTABLE : STABLE;
}

// ---- TransformerVoltageControlLoop (Ls2gOuterModel::check_tvc and its helpers) -------------------------
__device__ double bc_ratio_at(const Dev& d, int e, int rpos, int pos)
{
    const int np = d.bc_phi[e] - d.bc_plo[e] + 1;
    return d.bc_tab_ratio[d.bc_tab_ptr[e] + (rpos - d.bc_rlo[e]) * np + (pos - d.bc_plo[e])];
}
__device__ bool tvc_reserved(const Dev& d, int gt) { const int e = d.gt_e[gt]; return e >= 0 && d.bc_ratio[e]; }
__device__ void tvc_set_on(const Dev& d, int s, int gt, bool on)
{
    SLOT(tv_en, d.n_gt)[gt] = on ? 1 : 0;
    if (tvc_reserved(d, gt)) SLOT(bc_rctl, d.n_bc)[d.gt_e[gt]] = on ? 1 : 0;
}
// PiModelArray.roundR1ToClosestTap, at the phase tap the solve has it at
__device__ int tvc_closest_tap(const Dev& d, int s, int e, double value)
{
    const int ppos = SLOT(bc_pos, d.n_bc)[e];
    int best = SLOT(bc_rpos, d.n_bc)[e];
    double best_distance = fabs(value - bc_ratio_at(d, e, best, ppos));
    for (int pos = d.bc_rlo[e]; pos <= d.bc_rhi[e]; ++pos) {
        const double distance = fabs(value - bc_ratio_at(d, e, pos, ppos));
        if (distance < best_distance) { best = pos; best_distance = distance; }
    }
    return best;
}
__device__ bool tvc_voltage_control_off(const Dev& d, int s, int bus)
{
    const int k = (bus >= 0 && bus < d.n_bus) ? d.pin_of_bus[bus] : -1;
    return (k >= 0 && SLOT(pq, d.n_pin)[k]) || SLOT(tv_susp, d.n_bus)[bus];
}
// disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage (the candidates the row fixed)
__device__ void tvc_freeze(const Dev& d, int s, const Row& rv)
{
    for (int k = 0; k < d.rl_n_bus_q; ++k) {
        if (!SLOT(rl_present, d.rl_n)[k] || !SLOT(tv_cand, d.rl_n)[k]) continue;
        const int b = d.rl_bus[k];
        if (tvc_voltage_control_off(d, s, b)) continue;
        const bool local = d.rl_local[k];
        if (!local) {
            bool held = false;
            for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) {
                const int c = d.rl_ctrl[j];
                if (!SLOT(ctrl_off, d.n_ctrl)[c]) held = held || isfinite(SLOT(held_q, d.n_ctrl)[c]);
            }
            if (held) continue;
        }
        const double tvm0 = SLOT(tv_tvm, d.rl_n)[k];
        const double target_vm = isnan(tvm0) ? vm_of(d, s, rv, b) : tvm0;
        const double load = SLOT(tv_load, d.rl_n)[k];
        if (local) {
            double mre, mim;
            mis_bus(d, s, rv, b, mre, mim);
            freeze_set(d, s, b, mim - load);
            const int pin = d.pin_of_bus[b];
            if (pin >= 0) SLOT(pq, d.n_pin)[pin] = 1;
        } else {
            bool first = true;
            for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) {
                const int c = d.rl_ctrl[j];
                if (SLOT(ctrl_off, d.n_ctrl)[c]) continue;
                double q = rv.vc_q(c);
                if (first) q -= load;
                first = false;
                SLOT(held_q, d.n_ctrl)[c] = q;
            }
        }
        SLOT(tv_susp, d.n_bus)[b] = 1;
        SLOT(tv_fz, d.rl_n)[k] = 1;
        SLOT(tv_fzv, d.rl_n)[k] = target_vm;
    }
}
__device__ void tvc_release(const Dev& d, int s)
{
    for (int k = 0; k < d.rl_n; ++k) {
        if (!SLOT(tv_fz, d.rl_n)[k]) continue;
        const int b = d.rl_bus[k];
        if (d.rl_local[k]) {
            freeze_erase(d, s, b);
            const int pin = d.pin_of_bus[b];
            if (pin >= 0) SLOT(pq, d.n_pin)[pin] = 0;
            reset_vm(d, s, b, SLOT(tv_fzv, d.rl_n)[k]);
        } else {
            for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j)
                if (!SLOT(ctrl_off, d.n_ctrl)[d.rl_ctrl[j]]) SLOT(held_q, d.n_ctrl)[d.rl_ctrl[j]] = CUDART_NAN;
        }
        SLOT(tv_susp, d.n_bus)[b] = 0;
        SLOT(tv_fz, d.rl_n)[k] = 0;
    }
}
// LfNetwork.fixTransformerVoltageControls: a transformer whose other side, once every transformer
// switched on is taken out, keeps no PV bus is switched off
__device__ int tvc_root(int* parent, int x)
{
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
}
__device__ void tvc_fix_controls(const Dev& d, int s)
{
    int* parent = SLOT(tv_par, d.n_bus);
    unsigned char* pv = SLOT(tv_pv, d.n_bus);
    const unsigned char* closed = SLOT(tv_closed, d.n_br);
    for (int b = 0; b < d.n_bus; ++b) { parent[b] = b; pv[b] = 0; }
    for (int br = 0; br < d.n_br; ++br) {
        if (!closed[br]) continue;
        const int tr = d.br_trafo[br];
        if (tr >= 0 && d.trafo_gt[tr] >= 0 && SLOT(tv_en, d.n_gt)[d.trafo_gt[tr]]) continue;
        const int a = d.br_b1[br], b = d.br_b2[br];
        if (a < 0 || b < 0 || a >= d.n_bus || b >= d.n_bus) continue;
        parent[tvc_root(parent, a)] = tvc_root(parent, b);
    }
    for (int k = 0; k < d.rl_n_bus_q; ++k) {
        if (!SLOT(rl_present, d.rl_n)[k] || !d.rl_local[k]) continue;
        if (tvc_voltage_control_off(d, s, d.rl_bus[k])) continue;
        pv[tvc_root(parent, d.rl_bus[k])] = 1;
    }
    for (int g = 0; g < d.n_tvg; ++g)
        for (int gt = d.tvg_ptr[g]; gt < d.tvg_ptr[g + 1]; ++gt) {
            if (!SLOT(tv_en, d.n_gt)[gt]) continue;
            const int bi = d.n_line + d.gt_trafo[gt];   // its branch (lines first)
            if (!closed[bi]) continue;
            const int b1 = d.br_b1[bi], b2 = d.br_b2[bi], bus = d.tvg_bus[g];
            int other;
            if (bus == b1) other = b2;
            else if (bus == b2) other = b1;
            else continue;
            if (other < 0 || other >= d.n_bus || !pv[tvc_root(parent, other)]) tvc_set_on(d, s, gt, false);
        }
}
__device__ int check_tvc(const Dev& d, int s, const Row& rv)
{
    int& step = S1(tv_step);
    if (step == 2 || d.n_tvg == 0) return STABLE;
    bool any = false;
    for (int gt = 0; gt < d.n_gt; ++gt) any = any || tvc_reserved(d, gt);
    if (!any) return STABLE;
    const unsigned char* hidden = SLOT(tv_hid, d.n_tvg);
    unsigned char* en = SLOT(tv_en, d.n_gt);
    double* rat = SLOT(tv_rat, 6 * d.n_gt);   // initial, min, max, shared_min, shared_max, shared_initial
    if (step == 0) {
        bool need_run = false;
        for (int g = 0; g < d.n_tvg; ++g) {
            if (hidden[g]) continue;
            const double v = vm_of(d, s, rv, d.tvg_bus[g]);
            const bool outside = !(fabs(d.tvg_target[g] - v) <= d.tvg_hdb[g]);
            rec(d, s, TA_SWITCH_ON, outside, d.tvg_bus[g], fabs(d.tvg_target[g] - v), d.tvg_hdb[g]);
            if (!outside) continue;
            for (int gt = d.tvg_ptr[g]; gt < d.tvg_ptr[g + 1]; ++gt) {
                if (!tvc_reserved(d, gt)) continue;
                tvc_set_on(d, s, gt, true);
                need_run = true;
            }
        }
        // TransformerRatioManager: the ratios of the transformers switched on
        for (int g = 0; g < d.n_tvg; ++g) {
            int n_on = 0;
            double a = 0., b = 0., sum_min = 0., sum_max = 0.;
            for (int gt = d.tvg_ptr[g]; gt < d.tvg_ptr[g + 1]; ++gt) {
                if (!en[gt]) continue;
                ++n_on;
                const int e = d.gt_e[gt];
                double* r = rat + 6 * gt;
                const int ppos = SLOT(bc_pos, d.n_bc)[e];
                r[0] = rv.bc_rho(e);
                r[1] = CUDART_INF;
                r[2] = -CUDART_INF;
                for (int pos = d.bc_rlo[e]; pos <= d.bc_rhi[e]; ++pos) {
                    const double value = bc_ratio_at(d, e, pos, ppos);
                    r[1] = fmin(r[1], value);
                    r[2] = fmax(r[2], value);
                }
                if (r[2] != r[1]) {
                    a += (r[2] - r[0]) / (r[2] - r[1]);
                    b += (r[0] - r[1]) / (r[2] - r[1]);
                } else {
                    a += 1.;
                    b += 1.;
                }
                sum_min += r[1];
                sum_max += r[2];
            }
            if (n_on == 0) continue;
            const double nn = static_cast<double>(n_on);
            for (int gt = d.tvg_ptr[g]; gt < d.tvg_ptr[g + 1]; ++gt) {
                if (!en[gt]) continue;
                double* r = rat + 6 * gt;
                if (d.tv_use_initial) {
                    r[3] = sum_min / nn;
                    r[4] = sum_max / nn;
                    r[5] = (r[3] * a + r[4] * b) / nn;
                } else {
                    r[3] = r[1];
                    r[4] = r[2];
                    r[5] = r[0];
                }
            }
        }
        if (!need_run) {
            step = 2;
            return STABLE;
        }
        tvc_freeze(d, s, rv);
        tvc_fix_controls(d, s);
        step = 1;
        return CHECK_AGAIN;
    }
    // CONTROL: the ratios out of their range rounded to their extreme tap
    bool out_of_range = false;
    for (int gt = 0; gt < d.n_gt; ++gt) {
        if (!en[gt]) continue;
        const int e = d.gt_e[gt];
        const double* r = rat + 6 * gt;
        const double value = rv.bc_rho(e);
        if (value < r[3] || value > r[4]) {
            const double edge = value > r[4] ? r[4] : r[3];
            rec(d, s, TA_ROUND_TO_RANGE, true, d.gt_trafo[gt], value, edge);
            SLOT(bc_rrq, d.n_bc)[e] = tvc_closest_tap(d, s, e, edge);
            tvc_set_on(d, s, gt, false);
            out_of_range = true;
        }
    }
    if (!out_of_range) {
        // updateContinuousRatio, then every transformer rounded and switched off
        for (int g = 0; g < d.n_tvg; ++g) {
            if (hidden[g]) continue;
            for (int gt = d.tvg_ptr[g]; gt < d.tvg_ptr[g + 1]; ++gt) {
                if (!tvc_reserved(d, gt)) continue;
                const int e = d.gt_e[gt];
                double value = rv.bc_rho(e);
                if (en[gt] && d.tv_use_initial) {
                    const double* r = rat + 6 * gt;
                    value = value >= r[5] ? r[0] + (value - r[5]) * (r[2] - r[0]) / (r[4] - r[5])
                                          : r[0] - (r[5] - value) * (r[0] - r[1]) / (r[5] - r[3]);
                }
                const int position = tvc_closest_tap(d, s, e, value);
                SLOT(bc_rrq, d.n_bc)[e] = position;
                rec(d, s, TA_ROUND_TAP, true, d.gt_trafo[gt], value, static_cast<double>(position));
                tvc_set_on(d, s, gt, false);
            }
        }
        tvc_release(d, s);
        step = 2;
    }
    return CHECK_AGAIN;
}

__device__ int check(const Dev& d, int s, const Row& rv, int loop, int iteration)
{
    switch (d.loop_kind[loop]) {
        case 0: return check_ds(d, s, rv);
        case 1: return check_hvdc(d, s, rv);
        case 2: return check_vm(d, s, rv);
        case 4: return check_svc(d, s, rv, iteration);
        case 5: return check_pc(d, s, rv, iteration);
        case 6: return check_tvc(d, s, rv);
        // ReactiveLimits: drive evaluates its buses in parallel first (eval_rl_bus, check_rl_apply)
        default: return STABLE;
    }
}

// ---- materialize: the slot's controls --------------------------------------------------------------
// materialize, split for a block: materialize_par on every thread t of nt (the per-element copies, each element
// written by one thread exactly as the sequential loop writes it), a barrier, then materialize_seq on one thread
// (what depends on the order: a group's set-point overridden by its SVCs, the compacted correction list, the
// retaps, the slack weights summed per bus). materialize = both on one thread.
__device__ void materialize_par(const Dev& d, int s, int t, int nt)
{
    const unsigned char* pq = SLOT(pq, d.n_pin);
    unsigned char* pin = SLOT(c_pin, d.n_pin);
    for (int k = t; k < d.n_pin; k += nt) pin[k] = pq[k] ? 0 : 1;
    const double* held_q = SLOT(held_q, d.n_ctrl);
    const double* svc_on = SLOT(svc_on, d.n_ctrl);
    double* c_held = SLOT(c_held_q, d.n_ctrl);
    unsigned char* plan_held = SLOT(c_plan_held, d.n_ctrl);
    double* v_set = SLOT(c_v_set, d.n_grp);
    for (int g = t; g < d.n_grp; g += nt) v_set[g] = d.vc_v_set[g];
    for (int j = t; j < d.n_ctrl; j += nt) {
        c_held[j] = held_q[j];
        plan_held[j] = (d.vc_plan_held[j] && !(d.vc_kind[j] == VC_SVC && isfinite(svc_on[j]))) ? 1 : 0;
    }
    const int* regime = SLOT(hvdc_regime, d.n_hvdc);
    int* status = SLOT(c_hvdc_status, d.n_hvdc);
    for (int e = t; e < d.n_hvdc; e += nt) status[e] = regime[e] == 2 ? d.hv_st0[e] : regime[e];
}
__device__ void materialize_seq(const Dev& d, int s)
{
    {   // a plan-held SVC switched on sets its group's v_set (in controller order: the last one wins)
        const double* svc_on = SLOT(svc_on, d.n_ctrl);
        double* v_set = SLOT(c_v_set, d.n_grp);
        for (int j = 0; j < d.n_ctrl; ++j)
            if (d.vc_plan_held[j] && d.vc_kind[j] == VC_SVC && isfinite(svc_on[j])) v_set[d.vc_group[j]] = svc_on[j];
    }
    // target_delta + i.freeze_q, per bus, both over corr_bus (sorted)
    const unsigned char* dset = SLOT(dset, d.nc);
    const unsigned char* fset = SLOT(fset, d.nc);
    const double* dre = SLOT(dre, d.nc);
    const double* dim = SLOT(dim, d.nc);
    const double* fq = SLOT(fq, d.nc);
    int* sb_bus = SLOT(c_sb_bus, d.nc);
    double* sb_dp = SLOT(c_sb_dp, d.nc);
    double* sb_dq = SLOT(c_sb_dq, d.nc);
    int n = 0;
    for (int ci = 0; ci < d.nc; ++ci) {
        if (!dset[ci] && !fset[ci]) continue;
        double re = dset[ci] ? dre[ci] : 0.;
        double im = dset[ci] ? dim[ci] : 0.;
        if (fset[ci]) { re += 0.; im += fq[ci]; }
        sb_bus[n] = d.corr_bus[ci];
        sb_dp[n] = re;
        sb_dq[n] = im;
        ++n;
    }
    S1(c_sb_n) = n;
    // BranchControl (NROuterInner::apply_controls): the phase taps the loops moved (again before every
    // solve), the ratio taps once, each retap flagged for the driver
    for (int e = 0; e < d.n_bc; ++e) {
        unsigned char& rt = SLOT(bc_rt, d.n_bc)[e];
        rt = 0;
        if (SLOT(bc_out, d.n_bc)[e]) continue;
        if (SLOT(bc_mov, d.n_bc)[e]) { bc_retap(d, s, e, SLOT(bc_rpos, d.n_bc)[e], SLOT(bc_rq, d.n_bc)[e]); rt = 1; }
        if (SLOT(bc_rrq, d.n_bc)[e] != kNoTap) { bc_retap(d, s, e, SLOT(bc_rrq, d.n_bc)[e], SLOT(bc_pos, d.n_bc)[e]); rt = 1; }
    }
    // the distributed slack's weights (NROuterInner::apply_controls): each unit's normalised weight
    // on its bus, the reference alone at 1 when none takes a share
    if (d.dist) {
        double* w_out = SLOT(c_slack_w, d.n_slack);
        for (int k = 0; k < d.n_slack; ++k) w_out[k] = 0.;
        const double total = ds_share(d, s);
        if (total > 0.) {
            double w;
            for (int k = 0; k < d.n_units; ++k)
                if (ds_shares(d, s, k, w)) w_out[d.slack_idx[SLOT(ds_ubus, d.n_units)[k]]] += w / total;
        } else if (d.ref_slack >= 0) {
            w_out[d.ref_slack] = 1.;
        }
    }
}
__device__ void materialize(const Dev& d, int s)
{
    materialize_par(d, s, 0, 1);
    materialize_seq(d, s);
}

// ---- finish ----------------------------------------------------------------------------------------
__device__ int finish_scalar(const Dev& d, int s, int inner_status)   // finish, but its per-element results
{
    int status;
    if (S1(last_status) == FAILED) status = FAILED;
    else status = S1(nb_outer) < d.max_outer ? STABLE : UNSTABLE;
    int row_status;
    if (S1(solver_ok)) row_status = status == STABLE ? 0 : status == FAILED ? 4 : 5;
    else row_status = S1(unrealistic) ? 6 : inner_status;
    const int r = S1(row);
    if (r >= 0) {
        d.r_status[r] = status;
        d.r_unreal[r] = S1(unrealistic);
        d.r_failed[r] = status == FAILED ? S1(last_loop) : -1;
        d.r_outer[r] = S1(nb_outer);
        d.r_nr[r] = S1(nr_steps);
    }
    return row_status;
}
// the finished row's per-element results, thread t of nt (finish_scalar has run; any order: one write each)
__device__ void finish_arrays(const Dev& d, int s, int t, int nt)
{
    const int r = S1(row);
    if (r < 0) return;
    for (int k = t; k < d.n_units; k += nt)
        d.r_unit_mw[static_cast<size_t>(r) * d.n_units + k] =
            d.ds_kind[k] == 0 ? SLOT(gen_tp, d.n_gen)[d.ds_el[k]] : SLOT(sto_tp, d.n_sto)[d.ds_el[k]];
    for (int e = t; e < d.n_hvdc; e += nt) d.r_hvdc[static_cast<size_t>(r) * d.n_hvdc + e] = SLOT(hvdc_regime, d.n_hvdc)[e];
    for (int j = t; j < d.n_sc_sh; j += nt) d.r_sc[static_cast<size_t>(r) * d.n_sc_sh + j] = SLOT(sc_cnt, d.n_sc_sh)[j];
    for (int e = t; e < d.n_bc; e += nt) {
        d.r_pos[static_cast<size_t>(r) * d.n_bc + e] = SLOT(bc_pos, d.n_bc)[e];
        d.r_rpos[static_cast<size_t>(r) * d.n_bc + e] = SLOT(bc_rpos, d.n_bc)[e];
    }
}

// ---- drive: the for loop over the loops of a pass, from loop i --------------------------------------
// returns 0 Resolve (the caller materializes), 1 Done (row_status set; the caller writes finish_arrays)
// 2: paused before a ReactiveLimits check, whose buses the block evaluates first (p_recheck keeps
// what to resume with: nothing of that check has run yet); `rl_ready`: they are evaluated
__device__ int drive(const Dev& d, int s, const Row* rv, int& row_status, int inner_status, bool recheck,
                     bool unreal_flag, bool& rl_ready)
{
    const int n = S1(n_active);
    int* active = SLOT(active, d.n_loops);
    int* done_at = SLOT(done_at, d.n_loops);
    int* loop_iter = SLOT(loop_iter, d.n_loops);
    for (;;) {
        const bool rechecking = recheck;
        const int i = S1(it);
        const bool stop = !recheck && (i >= n || i == S1(last_unstable) || !S1(solver_ok) ||
                                       S1(last_status) == FAILED || S1(nb_outer) >= d.max_outer);
        recheck = false;
        if (stop) {
            if (S1(total_nr) > S1(old_total) && S1(solver_ok) && S1(last_status) != FAILED &&
                S1(nb_outer) < d.max_outer) {
                S1(pass_check_unreal) = S1(last_fixing) < 0;
                S1(old_total) = S1(total_nr);
                ++S1(nb_passes);
                S1(it) = 0;
                continue;
            }
            row_status = finish_scalar(d, s, inner_status);   // the caller writes finish_arrays
            return 1;
        }
        const int loop = active[i];
        S1(tr_li) = loop;
        const bool settled = !rechecking && done_at[i] == S1(v_version);
        int status = STABLE;
        if (!settled && d.loop_kind[loop] == 3) {
            if (!rl_ready) { S1(p_recheck) = rechecking ? 1 : 0; return 2; }
            rl_ready = false;
            status = check_rl_apply(d, s, *rv);
        } else if (!settled) {
            status = check(d, s, *rv, loop, loop_iter[i]);
        }
        if (!settled) {
            S1(last_status) = status;
            S1(last_loop) = loop;
        }
        if (status == UNSTABLE || status == CHECK_AGAIN) {
            S1(changed) = status == UNSTABLE;
            ++loop_iter[i];
            S1(last_unstable) = i;
            S1(check_unreal) = S1(pass_check_unreal);
            return 0;   // the caller materializes the controls (a block: materialize_par then materialize_seq)
        }
        // STABLE or FAILED: the loop's do-while ends here
        if (!S1(pass_check_unreal) && i == S1(last_fixing) && S1(solver_ok) && rv != nullptr && unreal_flag) {
            S1(unrealistic) = 1;
            S1(solver_ok) = 0;
        }
        if (i == S1(last_fixing)) S1(pass_check_unreal) = 1;
        S1(it) = i + 1;
    }
}

// ---- on_solved_impl, one thread per listed slot ----------------------------------------------------
// on_solved_impl, one block per listed slot: thread 0 runs the driver logic; whenever it reaches a
// ReactiveLimits check (drive returns 2) the block evaluates that slot's buses in parallel, then
// thread 0 resumes. The buses are independent and each one's sums stay those of one thread, so
// the decisions are the one-thread port's.
__global__ void on_solved_kernel(Dev d, const int* slots, int n, const int* converged, const int* nr_iter,
                                 const int* inner_status, const int* grow, const double* gather, int stride,
                                 int* next_out, int* status_out)
{
    const int i = blockIdx.x;
    if (i >= n) return;   // uniform over the block
    const int s = slots[i];
    const bool conv = converged[i] != 0;
    // a converged slot's gathered row (the driver gathers the converged slots only, in order)
    const double* g = gather + static_cast<size_t>(conv ? grow[i] : 0) * stride;
    const bool unreal_flag = conv && g[stride - 1] != 0.;
    Row rv{g, d.n_watch, d.n_ctrl, d.n_bc};
    const Row* rvp = conv ? &rv : nullptr;
    __shared__ int sh_next;
    int row_status = inner_status[i];   // thread 0's
    if (threadIdx.x == 0) {
        bool rl_ready = false;
        S1(total_nr) += max(nr_iter[i], 1);
        S1(nr_steps) += nr_iter[i];
        bool ok = conv;
        if (ok && S1(check_unreal) && unreal_flag) { S1(unrealistic) = 1; ok = false; }
        S1(solver_ok) = ok;
        // OuterLoopAlgo::_solve: a distributed-slack Newton's share into the units' targets
        if (d.dist && conv) ds_fold(d, s, g[stride - 2]);
        int next;
        if (S1(first)) {
            S1(first) = 0;
            if (!S1(solver_ok)) {
                row_status = finish_scalar(d, s, inner_status[i]);
                next = 1;
            } else {
                S1(pass_check_unreal) = S1(last_fixing) < 0;
                S1(old_total) = S1(total_nr);
                ++S1(nb_passes);
                S1(it) = 0;
                next = drive(d, s, rvp, row_status, inner_status[i], false, unreal_flag, rl_ready);
            }
        } else {
            ++S1(nb_outer);
            bool again = true;
            if (nr_iter[i] > 0) {
                ++S1(v_version);
            } else if (S1(solver_ok) && S1(changed)) {
                SLOT(done_at, d.n_loops)[S1(it)] = S1(v_version);
                S1(last_status) = STABLE;
                S1(tr_li) = SLOT(active, d.n_loops)[S1(it)];
                rec(d, s, TA_SETTLED, true, -1, 0., 0.);
                again = false;
            }
            if (again && S1(solver_ok) && S1(nb_outer) < d.max_outer) {
                next = drive(d, s, rvp, row_status, inner_status[i], true, unreal_flag, rl_ready);
            } else {
                if (!S1(pass_check_unreal) && S1(it) == S1(last_fixing) && S1(solver_ok) && conv && unreal_flag) {
                    S1(unrealistic) = 1;
                    S1(solver_ok) = 0;
                }
                if (S1(it) == S1(last_fixing)) S1(pass_check_unreal) = 1;
                ++S1(it);
                next = drive(d, s, rvp, row_status, inner_status[i], false, unreal_flag, rl_ready);
            }
        }
        sh_next = next;
    }
    __syncthreads();
    while (sh_next == 2) {   // paused before a ReactiveLimits check
        if (threadIdx.x == 0) rl_groups_holding(d, s);
        __syncthreads();
        for (int k = threadIdx.x; k < d.rl_n; k += blockDim.x) eval_rl_bus(d, s, rv, k);
        __syncthreads();
        if (threadIdx.x == 0) {
            bool rl_ready = true;
            sh_next = drive(d, s, rvp, row_status, inner_status[i], S1(p_recheck) != 0, unreal_flag, rl_ready);
        }
        __syncthreads();
    }
    // what drive left to the block: the controls of a row to re-solve, the results of a finished one
    if (sh_next == 0) {
        materialize_par(d, s, threadIdx.x, blockDim.x);
        __syncthreads();
        if (threadIdx.x == 0) materialize_seq(d, s);
    } else if (sh_next == 1) {
        finish_arrays(d, s, threadIdx.x, blockDim.x);
    }
    if (threadIdx.x != 0) return;
    next_out[i] = sh_next;
    status_out[i] = row_status;
    // a case the host scan should have refused: next 2, the error code as the status
    if (S1(err) != 0) { next_out[i] = 2; status_out[i] = S1(err); }
}

// the listed slots' controls into the driver's arrays (DeviceOuterHook::write_controls)
__global__ void write_controls_kernel(Dev d, const int* slots, int n, DeviceOuterHook::ControlTargets t)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = slots[i];
    const size_t ss = static_cast<size_t>(s);
    for (int k = 0; k < d.n_pin; ++k) t.pin[ss * d.n_pin + k] = SLOT(c_pin, d.n_pin)[k];
    for (int j = 0; j < d.n_ctrl; ++j) {
        t.held_q[ss * d.n_ctrl + j] = SLOT(c_held_q, d.n_ctrl)[j];
        t.plan_held[ss * d.n_ctrl + j] = SLOT(c_plan_held, d.n_ctrl)[j];
    }
    for (int g = 0; g < d.n_grp; ++g) t.v_set[ss * d.n_grp + g] = SLOT(c_v_set, d.n_grp)[g];
    for (int e = 0; e < d.n_hvdc; ++e) t.hvdc[ss * d.n_hvdc + e] = SLOT(c_hvdc_status, d.n_hvdc)[e];
    const size_t b0 = static_cast<size_t>(i) * d.nc;
    const int nsb = S1(c_sb_n);
    for (int k = 0; k < d.nc; ++k) {
        t.ds_slot[b0 + k] = s;
        t.ds_bus[b0 + k] = k < nsb ? SLOT(c_sb_bus, d.nc)[k] : -1;
        t.ds_p[b0 + k] = k < nsb ? SLOT(c_sb_dp, d.nc)[k] : 0.;
        t.ds_q[b0 + k] = k < nsb ? SLOT(c_sb_dq, d.nc)[k] : 0.;
    }
    const size_t v0 = static_cast<size_t>(i) * d.vcap;
    const int nvr = S1(vmr_n);
    for (int k = 0; k < d.vcap; ++k) {
        t.vr_slot[v0 + k] = s;
        t.vr_bus[v0 + k] = k < nvr ? SLOT(vmr_bus, d.vcap)[k] : -1;
        t.vr_vm[v0 + k] = k < nvr ? SLOT(vmr_vm, d.vcap)[k] : 0.;
    }
    S1(vmr_n) = 0;   // one-shot (OuterHook::clear_vm_reset)
    if (t.slack_w != nullptr)
        for (int k = 0; k < d.n_slack; ++k) t.slack_w[ss * d.n_slack + k] = SLOT(c_slack_w, d.n_slack)[k];
    // BranchControl (one block of n_bc per listed slot): out of the row, retapped (once), the block at
    // the taps (4 complex), its shift and ratio, the controls
    if (t.bc_out != nullptr) {
        const size_t c0 = static_cast<size_t>(i) * d.n_bc;
        for (int e = 0; e < d.n_bc; ++e) {
            t.bc_out[c0 + e] = SLOT(bc_out, d.n_bc)[e];
            t.bc_retap[c0 + e] = SLOT(bc_rt, d.n_bc)[e];
            for (int q = 0; q < 8; ++q) t.bc_tap[8 * (c0 + e) + q] = SLOT(bc_tap, 8 * d.n_bc)[8 * e + q];
            t.bc_a_tap[c0 + e] = SLOT(bc_atap, d.n_bc)[e];
            t.bc_rho_tap[c0 + e] = SLOT(bc_rtap, d.n_bc)[e];
            t.bc_ctl[c0 + e] = static_cast<signed char>(SLOT(bc_ctl, d.n_bc)[e]);
            t.bc_rctl[c0 + e] = static_cast<signed char>(SLOT(bc_rctl, d.n_bc)[e]);
            SLOT(bc_rt, d.n_bc)[e] = 0;          // applied
            SLOT(bc_rrq, d.n_bc)[e] = kNoTap;    // RatioTapControl::take_tap: once
        }
    }
    // ShuntControl (one block of n_sc per listed slot): the sections set (once), their B, the controls
    if (t.sc_set != nullptr) {
        const size_t c0 = static_cast<size_t>(i) * d.n_sc;
        for (int e = 0; e < d.n_sc; ++e) {
            t.sc_set[c0 + e] = SLOT(sc_pend, d.n_sc)[e];
            t.sc_b_new[c0 + e] = SLOT(sc_bnew, d.n_sc)[e];
            t.sc_ctl[c0 + e] = static_cast<signed char>(SLOT(sc_ctl, d.n_sc)[e]);
            SLOT(sc_pend, d.n_sc)[e] = 0;   // ShuntSectionControl::take_sections: applied once
        }
    }
}

__global__ void materialize_kernel(Dev d, const int* slots, int n)   // a block per listed slot
{
    const int i = blockIdx.x;
    if (i >= n) return;   // the whole block
    const int s = slots[i];
    materialize_par(d, s, threadIdx.x, blockDim.x);
    __syncthreads();
    if (threadIdx.x == 0) materialize_seq(d, s);
}

// ---- the model -------------------------------------------------------------------------------------
// A slot row of the three pools, on the host: what a load stages, what controls() reads back.
struct HostRow {
    std::vector<int> i;
    std::vector<unsigned char> u;
    std::vector<double> f;
};

class DeviceOuterModel final : public OuterHook, public DeviceOuterHook {
public:
    DeviceOuterModel(std::shared_ptr<OuterHook> host, DeviceHostSide* side);
    ~DeviceOuterModel() override {
        if (prof_on_ && prof_rows_ > 0)
            std::fprintf(stderr, "[GPUSIM2GRID_OC_PROFILE] device host-loaded rows %ld: host load %.1f ms, image %.1f ms, "
                         "fill %.1f ms, upload %.1f ms (%lld entries); pool widths int %zu u8 %zu f64 %zu\n", prof_rows_, prof_ms_[0],
                         prof_ms_[1], prof_ms_[2], prof_ms_[3], prof_dif_, w_int_, w_u8_, w_f64_);
        if (prof_on_ && prof_rows_ > 0)
            std::fprintf(stderr, "[GPUSIM2GRID_OC_PROFILE] device host-loaded rows by kind: resuming plain %ld, resuming masked %ld, "
                         "resuming other %ld, masked %ld, other %ld, phantom %ld\n", prof_why_[0], prof_why_[1], prof_why_[2],
                         prof_why_[3], prof_why_[4], prof_why_[5]);
    }
    const Layout& layout() const override { return host_->layout(); }
    void set_keep_adjoint(bool keep) override {
        keep_adj_ = keep;
        dev_.keep_adj = keep ? 1 : 0;
        adj_.assign(keep ? static_cast<size_t>(std::max(n_rows_, 0)) : 0, RowResult());
        host_->set_keep_adjoint(keep);
    }
    void begin(int n_slots, int n_rows) override;
    void load(int slot, int row) override;
    void move(int from, int to) override;
    const SlotControls& controls(int slot) const override;
    void clear_vm_reset(int slot) override;
    void row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const override;
    Next on_solved(int slot, bool converged, int nr_iter, int inner_status, const SlotData* data,
                   int& row_status) override;
    const RowResult& row_result(int row) const override;
    int n_rows() const override { return n_rows_; }
    void set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p) override {
        // a row with targets of its own loads through the host model (never from a template); its
        // result is rebuilt from them (row_base_)
        own_targets_.assign(static_cast<size_t>(std::max(n_rows, 0)), 0);
        const size_t ng = static_cast<size_t>(t_.n_gen), ns = static_cast<size_t>(t_.n_sto);
        for (size_t r = 0; r < own_targets_.size(); ++r) {
            bool own = false;
            for (size_t k = 0; ng > 0 && k < ng && (r + 1) * ng <= gen_p.size(); ++k) own = own || !std::isnan(gen_p[r * ng + k]);
            for (size_t k = 0; ns > 0 && k < ns && (r + 1) * ns <= storage_p.size(); ++k) own = own || !std::isnan(storage_p[r * ns + k]);
            own_targets_[r] = own ? 1 : 0;
        }
        host_->set_row_targets(n_rows, std::move(gen_p), std::move(storage_p));
    }
    // the session's buses grew by spare buses (topology actions) or went back: the host model
    // takes them, the device tables are rebuilt from it (the pools follow at the next begin)
    void set_spare_buses(const LedgerData& ledger, const std::vector<int>& spare_grid_bus) override {
        host_->set_spare_buses(ledger, spare_grid_bus);
        n_spare_ = static_cast<int>(spare_grid_bus.size());
        init_tables();
    }
    void keep_row_points(bool keep) override {
        keep_points_ = keep;
        host_->keep_row_points(keep);
    }
    void set_row_resume(std::vector<int> point_of_row) override {
        resume_of_row_ = point_of_row;
        referenced_.assign(resume_of_row_.size(), 0);
        for (int k : resume_of_row_) if (k >= 0 && static_cast<size_t>(k) < referenced_.size()) referenced_[static_cast<size_t>(k)] = 1;
        host_->set_row_resume(std::move(point_of_row));
    }
    void export_points(int n, const int* h_slots, const int* h_rows, const int* h_next, const int* h_status,
                       const int* h_grow, const double* d_gather, int stride) override;
    // a finished row's adjoint view (adjoint_info) from its slot, read back
    void adjoint_of(const HostRow& hr, RowResult& out) const;
    void set_row_overrides(RowOverrides ov) override {
        // the device tables' ReactiveLimits plan follows the rows' moved generators (before begin)
        if (side_->device_prepare_rows(ov)) init_tables();
        note_plain_rows(ov);
        host_->set_row_overrides(std::move(ov));
    }
    // DeviceOuterHook
    void load_rows(const std::vector<int>& slots, const std::vector<int>& rows) override;
    long long n_host_loads() const override { return n_host_loads_; }
    void decide(int n, const int* d_slots, const int* d_conv, const int* d_nr, const int* d_inner, const int* d_grow,
                const double* d_gather, int stride, int* d_next, int* d_status, void* stream) override;
    void set_stream(void* stream) override { stream_ = static_cast<cudaStream_t>(stream); }
    void move_slots(int n, const int* d_from, const int* d_to, const int* h_from, const int* h_to) override;
    // every entry of the block is written by write_controls_kernel (k < nc): none without a
    // correction bus, and the driver then launches no Sbus correction
    int sbus_block() const override { return nc_; }
    int vm_reset_block() const override { return vcap_; }
    void write_controls(int n, const int* d_slots, const ControlTargets& t, void* stream) override {
        if (n == 0) return;
        write_controls_kernel<<<(n + 63) / 64, 64, 0, static_cast<cudaStream_t>(stream)>>>(dev_, d_slots, n, t);
        check_cuda(cudaGetLastError(), "write_controls");
    }

private:
    struct PerSlot;   // the per-slot device arrays, see alloc_slots
    void build_dev();
    void alloc_slots(int S);
    void init_tables();
    void alloc_rows(int R);
    void fill_row(const DeviceSlotImage& img, HostRow& hr);
    void note_plain_rows(const RowOverrides& ov);
    // a row resuming from another row's point loads through the host model (its point is there)
    bool resumes(int row) const {
        return row >= 0 && static_cast<size_t>(row) < resume_of_row_.size() && resume_of_row_[static_cast<size_t>(row)] >= 0;
    }
    // nothing in RowOverrides changes the row's start (whatever point it resumes from)
    bool own_targets(int row) const {
        return row >= 0 && static_cast<size_t>(row) < own_targets_.size() && own_targets_[static_cast<size_t>(row)];
    }
    bool plain_overrides(int row) const {
        return row >= 0 && !own_targets(row) && (static_cast<size_t>(row) >= plain_.size() || plain_[static_cast<size_t>(row)])
               && (!has_loss(row) || loss_on_device(row));
    }
    // a lost power to share (RowOverrides::lost_p_mw, redistribute_slack; the session hands every row its
    // loss, 0 MW included)
    bool has_loss(int row) const {
        return row >= 0 && static_cast<size_t>(row) < row_loss_.size() && !std::isnan(row_loss_[static_cast<size_t>(row)]);
    }
    // shared on the device after the row's template copy (share_lost_kernel): from the n point or from
    // scratch, not from another row's point (it shares beyond that point's loss), and not after an n solve
    // that left units of a distributed-slack Newton parked (their keep-parked move follows the share)
    bool loss_on_device(int row) const { return !resumes(row) && !(t_.ds_reserved && t_.rp_n_parked > 0); }
    // a row whose load the template's is, bit for bit: it shares nothing (a loss of exactly 0 that
    // strands no unit counts nothing, as no loss) -- the rows the template may be captured from
    bool shares_nothing(int row) const {
        return !has_loss(row) || (row_loss_[static_cast<size_t>(row)] == 0. && !has_masks(row) && !any_off_unit_moved_);
    }
    // a plain row with a loss and no mask: the template, then its share (the masked rows' path, no mask)
    int edit_source(int row) const {
        if (has_masks(row)) return mask_source(row);
        return (tmpl_row_ && has_loss(row) && plain(row)) ? -1 : -2;
    }
    bool plain(int row) const { return plain_overrides(row) && !resumes(row); }
    // a row resuming from point k with plain overrides: a copy of point k's template once it exists
    // the point templates, row `idx` of three device buffers (one row per point, appended)
    struct PointTemplate {
        std::unique_ptr<HostRow> host;
        int idx = -1;
        // the buses (not spare) its capturing row masked, sorted: a row is copied from it only if it masks
        // them all (its contingency's island, typically); the mask kernel then removes the row's others
        std::vector<int> masks;
    };
    std::map<int, PointTemplate> point_tmpl_;
    thrust::device_vector<int> pt_i_;
    thrust::device_vector<unsigned char> pt_u_;
    thrust::device_vector<double> pt_f_;
    int n_pt_ = 0;
    void add_point_template(int point, const HostRow& hr);
    void compare_rows_with(const HostRow& hr, const HostRow& tmpl, int row) const;
    void compare_with_template(const HostRow& hr, int row) const;
    void compare_rows(const HostRow& host, const HostRow& dev, int row) const;
    bool mask_from_template(int row) const { return mask_source(row) != -2; }
    // the template a masked row is loaded from: -1 the plain rows', k >= 0 point template k's (a row
    // resuming from that point), -2 none (loaded through the host model)
    int mask_source(int row) const;
    void read_row(int slot, HostRow& hr) const;
    void set_int(int slot, const char* name, int v);
    int get_int(int slot, const char* name) const;
    void check_errors(const std::vector<int>& slots) const;
    void bind(int*& f, size_t base) { f = raw(p_int_) + base; }
    void bind(unsigned char*& f, size_t base) { f = raw(p_u8_) + base; }
    void bind(double*& f, size_t base) { f = raw(p_f64_) + base; }

    std::shared_ptr<OuterHook> host_;
    DeviceHostSide* side_;
    DeviceTables t_;
    int n_units_ = 0, n_hl_ = 0, n_mon_ = 0, nc_ = 0, vcap_ = 0, n_loops_ = 0, stride_ = 0;
    // the decision trace (keep_outer_trace): per-slot capacity between two flushes, the scratch widths
    int tcap_ = 0, trn_ = 0, trh_ = 0;
    int tvr_ = 0, tvb_ = 0;   // TransformerVoltageControl's per-bus / per-ReactiveLimits-bus widths (0 without it)
    std::vector<std::vector<std::string>> trace_;   // per row
    void flush_trace(int n, const int* h_slots, const int* h_rows);
    std::vector<int> corr_of_bus_h_;
    // static tables
    thrust::device_vector<int> i_widx_, i_loop_kind_, i_gen_vreg_, i_curve_ptr_, i_gen_rl_ptr_, i_gen_rl_bus_;
    thrust::device_vector<double> d_gen_tp0_, d_gen_min_q_, d_gen_max_q_, d_gen_tq_, d_gen_raw_q_;
    thrust::device_vector<double> d_curve_p_, d_curve_min_, d_curve_max_, d_sto_tp0_, d_svc_bmin_, d_svc_bmax_;
    thrust::device_vector<double> d_st_min1_, d_st_max1_, d_st_min2_, d_st_max2_;
    thrust::device_vector<int> i_rl_bus_, i_rl_local_, i_rl_reg_, i_rl_group_, i_rl_mon_, i_rl_pin_, i_rl_has_base_;
    thrust::device_vector<double> d_rl_min_q_, d_rl_max_q_, d_rl_tvm_, d_rl_nv_, d_rl_tq_, d_rl_bmin_, d_rl_bmax_;
    thrust::device_vector<int> i_rl_gen_ptr_, i_rl_gen_, i_rl_svc_ptr_, i_rl_svc_, i_rl_ctrl_ptr_, i_rl_ctrl_;
    thrust::device_vector<int> i_vc_bus_, i_vc_kind_, i_vc_elem_, i_vc_group_, i_vc_plan_held_, i_vc_at_ptr_, i_vc_at_, i_svc_ctrl_;
    thrust::device_vector<double> d_vc_slope_, d_vc_q_plan_, d_vc_v_set_;
    thrust::device_vector<int> i_hv_b1_, i_hv_b2_, i_hv_st0_;
    thrust::device_vector<double> d_hv_p0_, d_hv_k_, d_hv_lf1_, d_hv_lf2_, d_hv_r_, d_hv_pmax12_, d_hv_pmax21_;
    thrust::device_vector<unsigned char> u_hvdc_at_bus_;
    thrust::device_vector<int> i_hl_id_, i_hl_b1_, i_hl_b2_, i_hl_entry_;
    thrust::device_vector<double> d_hl_p0_, d_hl_k_, d_hl_pmax12_, d_hl_pmax21_;
    thrust::device_vector<int> i_mon_svc_, i_mon_reg_, i_mon_bus_;
    thrust::device_vector<double> d_mon_low_, d_mon_high_, d_svc_low_t_, d_svc_high_t_, d_mon_vn_;
    thrust::device_vector<int> i_ds_el_, i_ds_kind_, i_ds_bus_, i_slack_idx_;
    thrust::device_vector<int> i_sc_bus_, i_sc_sh_ptr_, i_sh_min_, i_sh_max_, i_sh_count0_, i_sh_status_, i_sh_b_ptr_;
    thrust::device_vector<int> i_svg_ptr_, i_svg_e_, i_svg_bus_;
    thrust::device_vector<double> d_sh_b_;
    thrust::device_vector<int> r_sc_, r_pos_, r_rpos_;
    thrust::device_vector<int> i_bc_trafo_, i_bc_phase_, i_bc_b1_, i_bc_b2_, i_bc_has_ptc_, i_bc_plo_, i_bc_phi_, i_bc_rlo_,
        i_bc_rhi_, i_bc_tab_ptr_, i_bc_alpha_ptr_, i_pc_trafo_, i_pc_e_, i_pc_mode_, i_pc_side_;
    thrust::device_vector<int> i_bc_ratio_, i_tvg_ptr_, i_tvg_bus_, i_gt_trafo_, i_gt_e_, i_br_b1_, i_br_b2_, i_br_trafo_,
        i_trafo_gt_, i_pin_of_bus_;
    thrust::device_vector<double> d_tvg_target_, d_tvg_hdb_;
    thrust::device_vector<double> d_bc_sign_, d_bc_rexp_, d_bc_tab_ratio_, d_bc_tab_block_, d_bc_tab_shift_, d_bc_alpha_,
        d_pc_vn_kv_, d_pc_target_;
    thrust::device_vector<double> d_ds_w_, d_ds_min_, d_ds_max_;
    thrust::device_vector<int> i_rp_gen_bus_, i_rp_sto_bus_, i_rl_rp_state_, i_rl_sto_ptr_, i_rl_sto_, i_rl_has_sto_;
    thrust::device_vector<double> d_rp_gen_move_, d_rp_sto_move_, d_rp_v_;
    thrust::device_vector<int> i_corr_bus_, i_corr_of_bus_;
    // per slot / per row: one int, one byte and one double pool each, carved by offsets
    struct Field { int kind; size_t width; size_t off; };   // kind 0 int, 1 uchar, 2 double
    std::vector<Field> slot_fields_;
    std::map<std::string, Field> field_of_;
    thrust::device_vector<int> p_int_;
    thrust::device_vector<unsigned char> p_u8_;
    thrust::device_vector<double> p_f64_;
    size_t w_int_ = 0, w_u8_ = 0, w_f64_ = 0;   // per-slot widths of the three pools
    thrust::device_vector<int> r_int_;
    thrust::device_vector<double> r_f64_;
    int S_ = 0, n_rows_ = 0;
    // every copy and launch of the model, on the driver's stream (set_stream; legacy default until)
    cudaStream_t stream_ = nullptr;
    Dev dev_{};
    // the template targets every row starts from (checked at each load)
    std::vector<double> gen_tp_tmpl_, sto_tp_tmpl_;
    bool have_tmpl_ = false;
    // per point rows resume from: the targets those rows start from (gen, storage)
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> point_base_;
    // per row whose start differs from its reference (template / point): its own (gen, storage)
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> row_base_;
    // M3 step 2: the slot a row that changes nothing starts from, loaded once through the host
    // model and copied on the device for the next ones (GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE=1:
    // every such row also loaded through the host and compared)
    bool keep_points_ = false;                // OuterHook::keep_row_points
    bool keep_adj_ = false;                   // OuterHook::set_keep_adjoint
    // GPUSIM2GRID_OC_PROFILE: the host-loaded rows' parts (host load, image, fill, upload), ms
    bool prof_on_ = [] { const char* e = std::getenv("GPUSIM2GRID_OC_PROFILE"); return e && *e && *e != '0'; }();
    double prof_ms_[4] = {0., 0., 0., 0.};
    long prof_rows_ = 0;
    long long prof_dif_ = 0;   // the uploaded differences of the host-loaded rows (entries)
    long long n_host_loads_ = 0;   // since begin(): rows loaded through the host model
    // the spare buses (topology actions): the session's last n_spare_ buses. A plain row leaves them
    // empty, so its masks there change nothing the loops start from
    int n_spare_ = 0;
    bool spare_bus(int b) const { return b >= t_.n_bus - n_spare_; }
    bool has_masks(int row) const {
        return row >= 0 && static_cast<size_t>(row) < masked_rows_.size() && !masked_rows_[static_cast<size_t>(row)].empty();
    }
    // a template from a row's host load: its masks on spare buses cleared (a copy gets its own row's masks,
    // mask_rows_kernel); those elsewhere kept (real_masks: a copy's row must mask them too)
    HostRow unmasked(const HostRow& hr, int row) const;
    std::vector<int> real_masks(int row) const {
        std::vector<int> out;
        if (has_masks(row))
            for (int b : masked_rows_[static_cast<size_t>(row)]) if (!spare_bus(b)) out.push_back(b);
        return out;
    }
    // a row resuming from a point, starting as the point's template would: nothing but masks of its own
    bool point_row(int row) const {
        return resumes(row) && !own_targets(row) && (plain_overrides(row) ||
               (static_cast<size_t>(row) < mask_only_.size() && mask_only_[static_cast<size_t>(row)]));
    }
    long prof_why_[6] = {0, 0, 0, 0, 0, 0};
    std::vector<RowResult> adj_;              // per row: its adjoint view (pq_buses ... gen_p_slope_*)
    std::vector<int> resume_of_row_;          // OuterHook::set_row_resume
    std::vector<double> row_loss_;            // [rows] RowOverrides::lost_p_mw, NaN: none
    bool any_off_unit_moved_ = false;         // the n point moved a unit off in the grid (DeviceTables::rp_*_bus -2)
    std::vector<char> referenced_;            // per row: another row resumes from its point
    std::vector<char> own_targets_;           // per row: set_row_targets gave it targets of its own
    std::vector<char> plain_;                 // per row: nothing in RowOverrides changes its start
    std::vector<char> mask_only_;             // per row: only its masked buses do
    std::vector<std::vector<int>> masked_rows_;   // per row: its masked buses, sorted
    std::unique_ptr<HostRow> tmpl_row_;
    thrust::device_vector<int> tmpl_i_;
    thrust::device_vector<unsigned char> tmpl_u_;
    thrust::device_vector<double> tmpl_f_;
    bool check_tmpl_ = false;
    // host mirrors
    mutable std::vector<SlotControls> ctl_;
    mutable std::vector<RowResult> results_;
    mutable std::vector<char> fin_built_;
    // the row pools (r_int_ ... r_rpos_) on the host, downloaded once after a decide() changed them:
    // row_result / row_trafo_blocks of every row read them there (one copy per pool, not per row)
    mutable std::vector<int> h_r_int_, h_r_sc_, h_r_pos_, h_r_rpos_;
    mutable std::vector<double> h_r_f64_;
    mutable bool rows_on_host_ = false;
    void sync_rows_host() const;
    const int* host_of(const int* dev) const;
    const double* host_of(const double* dev) const;
    // staging
    thrust::device_vector<int> st_slots_, st_conv_, st_nr_, st_inner_, st_grow_, st_next_, st_status_;
    // load_rows' lists and staging, kept across calls (an allocation would sync the device)
    thrust::device_vector<int> ld_slots_, ld_tmpl_, ld_mask_, ld_point_;
    thrust::device_vector<double> ld_loss_;   // the losses of the rows load_rows edits from the template
    // the phantom rows' template (a base-case load: row -1)
    bool phantom_ready_ = false;
    std::unique_ptr<HostRow> ph_host_;
    thrust::device_vector<int> ph_i_, ld_phantom_;
    thrust::device_vector<unsigned char> ph_u_;
    thrust::device_vector<double> ph_f_;
    // host-loaded rows: the baseline row (host / device) and each call's differences from it
    std::unique_ptr<HostRow> base_row_;
    thrust::device_vector<int> base_i_;
    thrust::device_vector<unsigned char> base_u_;
    thrust::device_vector<double> base_f_;
    std::vector<long long> dif_i_pos_, dif_u_pos_, dif_f_pos_;
    // the host-loaded rows of a call diffed against their point's template (not the baseline): their
    // slots and template rows (a resumed row differs from the baseline almost everywhere)
    std::vector<int> up_pslot_, up_pidx_;
    thrust::device_vector<int> ld_up_ps_, ld_up_pi_;
    std::vector<int> dif_i_val_;
    std::vector<unsigned char> dif_u_val_;
    std::vector<double> dif_f_val_;
    thrust::device_vector<long long> ld_dpos_i_, ld_dpos_u_, ld_dpos_f_;
    thrust::device_vector<int> ld_dval_i_;
    thrust::device_vector<unsigned char> ld_dval_u_;
    thrust::device_vector<double> ld_dval_f_;
    thrust::device_vector<double> st_gather_;
};

template <typename T>
thrust::device_vector<T> up(const std::vector<T>& v)
{
    return v.empty() ? thrust::device_vector<T>(1) : thrust::device_vector<T>(v.begin(), v.end());
}

DeviceOuterModel::DeviceOuterModel(std::shared_ptr<OuterHook> host, DeviceHostSide* side)
    : host_(std::move(host)), side_(side)
{
    init_tables();
}

// the tables off the host model (its buses as they are: the base ones, or with the spares
// set_spare_buses appended), uploaded; the slot pools are sized by begin()
void DeviceOuterModel::init_tables()
{
    try {
        t_ = side_->device_tables();
    } catch (const std::domain_error& e) {
        throw DeviceNotImplemented(e.what());
    }
    const Layout& L = host_->layout();
    n_units_ = static_cast<int>(t_.ds_el_id.size());
    n_hl_ = static_cast<int>(t_.hl_hvdc_id.size());
    n_mon_ = static_cast<int>(t_.mon_svc.size());
    nc_ = static_cast<int>(t_.corr_bus.size());
    n_loops_ = static_cast<int>(t_.loop_kind.size());
    if (t_.has_tvc) { tvr_ = t_.rl_n; tvb_ = t_.n_bus; }
    if (t_.keep_trace) {
        // one on_solved checks each active loop at most once (the buffer is flushed after every
        // decide): a re-check of loop L runs L..n-1, and the new pass it may start stops at
        // last_unstable == L, as Ls2gOuterModel::drive. DistributedSlack's few records and a PARK per
        // unit, a record per hvdc line and monitor, ReactiveLimits' up to two per bus (KEPT_PQ and
        // LIMIT_MOVED), the SETTLED record in the 16
        tcap_ = 2 * t_.rl_n + 2 * n_units_ + n_hl_ + n_mon_ + t_.n_pc + t_.n_tvg + 2 * t_.n_gt +
                (t_.svg_ptr.empty() ? 0 : t_.svg_ptr.back()) + 16;
        trn_ = t_.rl_n;
        trh_ = n_hl_;
    }
    vcap_ = t_.rl_n + t_.n_pin + 8;
    stride_ = 6 * t_.n_watch + t_.n_ctrl + 2 * L.n_bc + L.n_sc + 2;
    corr_of_bus_h_.assign(static_cast<size_t>(t_.n_bus), -1);
    for (int k = 0; k < nc_; ++k) corr_of_bus_h_[static_cast<size_t>(t_.corr_bus[static_cast<size_t>(k)])] = k;
    // hvdc: the plan's line -> its droop entry, the buses with a droop end
    std::vector<int> hl_entry;
    for (int k = 0; k < n_hl_; ++k) {
        const auto it = std::find(t_.hvdc_id.begin(), t_.hvdc_id.end(), t_.hl_hvdc_id[static_cast<size_t>(k)]);
        hl_entry.push_back(it == t_.hvdc_id.end() ? -1 : static_cast<int>(it - t_.hvdc_id.begin()));
    }
    std::vector<unsigned char> at_bus(static_cast<size_t>(t_.n_bus), 0);
    for (int e = 0; e < t_.n_hvdc; ++e)
        for (int b : {t_.hvdc_bus1[static_cast<size_t>(e)], t_.hvdc_bus2[static_cast<size_t>(e)]})
            if (b >= 0 && b < t_.n_bus) at_bus[static_cast<size_t>(b)] = 1;
    i_widx_ = up(t_.widx); i_loop_kind_ = up(t_.loop_kind);
    i_gen_vreg_ = up(t_.gen_vreg_on); i_curve_ptr_ = up(t_.curve_ptr);
    i_gen_rl_ptr_ = up(t_.gen_rl_bus_ptr); i_gen_rl_bus_ = up(t_.gen_rl_bus);
    d_gen_tp0_ = up(t_.gen_target_p); d_gen_min_q_ = up(t_.gen_min_q); d_gen_max_q_ = up(t_.gen_max_q);
    d_gen_tq_ = up(t_.gen_target_q); d_gen_raw_q_ = up(t_.gen_raw_target_q);
    d_curve_p_ = up(t_.curve_p); d_curve_min_ = up(t_.curve_min_q); d_curve_max_ = up(t_.curve_max_q);
    d_sto_tp0_ = up(t_.sto_target_p); d_svc_bmin_ = up(t_.svc_b_min); d_svc_bmax_ = up(t_.svc_b_max);
    d_st_min1_ = up(t_.station_min_q1); d_st_max1_ = up(t_.station_max_q1);
    d_st_min2_ = up(t_.station_min_q2); d_st_max2_ = up(t_.station_max_q2);
    i_rl_bus_ = up(t_.rl_bus); i_rl_local_ = up(t_.rl_local); i_rl_reg_ = up(t_.rl_reg_bus); i_rl_group_ = up(t_.rl_group);
    i_rl_mon_ = up(t_.rl_monitor_svc); i_rl_pin_ = up(t_.rl_pin); i_rl_has_base_ = up(t_.rl_has_base);
    d_rl_min_q_ = up(t_.rl_min_q); d_rl_max_q_ = up(t_.rl_max_q); d_rl_tvm_ = up(t_.rl_target_vm);
    d_rl_nv_ = up(t_.rl_nominal_v); d_rl_tq_ = up(t_.rl_target_q); d_rl_bmin_ = up(t_.rl_base_min_q); d_rl_bmax_ = up(t_.rl_base_max_q);
    i_rl_gen_ptr_ = up(t_.rl_gen_ptr); i_rl_gen_ = up(t_.rl_gen); i_rl_svc_ptr_ = up(t_.rl_svc_ptr); i_rl_svc_ = up(t_.rl_svc);
    i_rl_ctrl_ptr_ = up(t_.rl_ctrl_ptr); i_rl_ctrl_ = up(t_.rl_ctrl);
    i_vc_bus_ = up(t_.vc_bus); i_vc_kind_ = up(t_.vc_kind); i_vc_elem_ = up(t_.vc_elem); i_vc_group_ = up(t_.vc_group);
    i_vc_plan_held_ = up(t_.vc_plan_held); i_vc_at_ptr_ = up(t_.vc_at_bus_ptr); i_vc_at_ = up(t_.vc_at_bus);
    i_svc_ctrl_ = up(t_.svc_ctrl);
    d_vc_slope_ = up(t_.vc_slope); d_vc_q_plan_ = up(t_.vc_q_plan); d_vc_v_set_ = up(t_.vc_v_set);
    i_hv_b1_ = up(t_.hvdc_bus1); i_hv_b2_ = up(t_.hvdc_bus2); i_hv_st0_ = up(t_.hvdc_status0);
    d_hv_p0_ = up(t_.hvdc_p0); d_hv_k_ = up(t_.hvdc_k); d_hv_lf1_ = up(t_.hvdc_lf1); d_hv_lf2_ = up(t_.hvdc_lf2);
    d_hv_r_ = up(t_.hvdc_r); d_hv_pmax12_ = up(t_.hvdc_pmax12); d_hv_pmax21_ = up(t_.hvdc_pmax21);
    u_hvdc_at_bus_ = up(at_bus);
    i_hl_id_ = up(t_.hl_hvdc_id); i_hl_b1_ = up(t_.hl_bus1); i_hl_b2_ = up(t_.hl_bus2); i_hl_entry_ = up(hl_entry);
    d_hl_p0_ = up(t_.hl_p0_mw); d_hl_k_ = up(t_.hl_k_mw_per_rad); d_hl_pmax12_ = up(t_.hl_pmax12_mw); d_hl_pmax21_ = up(t_.hl_pmax21_mw);
    i_mon_svc_ = up(t_.mon_svc); i_mon_reg_ = up(t_.mon_reg_bus); i_mon_bus_ = up(t_.mon_svc_bus);
    d_mon_low_ = up(t_.mon_low_vm); d_mon_high_ = up(t_.mon_high_vm); d_mon_vn_ = up(t_.mon_vn_kv);
    d_svc_low_t_ = up(t_.svc_standby_low_target); d_svc_high_t_ = up(t_.svc_standby_high_target);
    i_ds_el_ = up(t_.ds_el_id); i_ds_kind_ = up(t_.ds_kind); i_ds_bus_ = up(t_.ds_bus);
    i_slack_idx_ = up(t_.slack_idx_of_bus);
    i_sc_bus_ = up(t_.sc_bus); i_sc_sh_ptr_ = up(t_.sc_sh_ptr); i_sh_min_ = up(t_.sh_min); i_sh_max_ = up(t_.sh_max);
    i_sh_count0_ = up(t_.sh_count0); i_sh_status_ = up(t_.sh_status); i_sh_b_ptr_ = up(t_.sh_b_ptr);
    i_svg_ptr_ = up(t_.svg_ptr); i_svg_e_ = up(t_.svg_e); i_svg_bus_ = up(t_.svg_bus); d_sh_b_ = up(t_.sh_b);
    i_bc_trafo_ = up(t_.bc_trafo); i_bc_phase_ = up(t_.bc_phase); i_bc_b1_ = up(t_.bc_b1); i_bc_b2_ = up(t_.bc_b2);
    i_bc_has_ptc_ = up(t_.bc_has_ptc); i_bc_plo_ = up(t_.bc_plo); i_bc_phi_ = up(t_.bc_phi); i_bc_rlo_ = up(t_.bc_rlo);
    i_bc_rhi_ = up(t_.bc_rhi); i_bc_tab_ptr_ = up(t_.bc_tab_ptr); i_bc_alpha_ptr_ = up(t_.bc_alpha_ptr);
    d_bc_sign_ = up(t_.bc_sign); d_bc_rexp_ = up(t_.bc_rexp); d_bc_tab_ratio_ = up(t_.bc_tab_ratio);
    d_bc_tab_block_ = up(t_.bc_tab_block); d_bc_tab_shift_ = up(t_.bc_tab_shift); d_bc_alpha_ = up(t_.bc_alpha);
    i_pc_trafo_ = up(t_.pc_trafo); i_pc_e_ = up(t_.pc_e); i_pc_mode_ = up(t_.pc_mode); i_pc_side_ = up(t_.pc_side);
    d_pc_vn_kv_ = up(t_.pc_vn_kv); d_pc_target_ = up(t_.pc_target);
    i_bc_ratio_ = up(t_.bc_ratio); i_tvg_ptr_ = up(t_.tvg_ptr); i_tvg_bus_ = up(t_.tvg_bus); i_gt_trafo_ = up(t_.gt_trafo);
    i_gt_e_ = up(t_.gt_e); i_br_b1_ = up(t_.br_b1); i_br_b2_ = up(t_.br_b2); i_br_trafo_ = up(t_.br_trafo);
    i_trafo_gt_ = up(t_.trafo_gt); i_pin_of_bus_ = up(t_.pin_of_bus); d_tvg_target_ = up(t_.tvg_target); d_tvg_hdb_ = up(t_.tvg_hdb);
    d_ds_w_ = up(t_.ds_weight); d_ds_min_ = up(t_.ds_min_mw); d_ds_max_ = up(t_.ds_max_mw);
    i_rp_gen_bus_ = up(t_.rp_gen_bus); i_rp_sto_bus_ = up(t_.rp_sto_bus); i_rl_rp_state_ = up(t_.rl_rp_state);
    i_rl_sto_ptr_ = up(t_.rl_sto_ptr); i_rl_sto_ = up(t_.rl_sto); i_rl_has_sto_ = up(t_.rl_has_storage);
    d_rp_gen_move_ = up(t_.rp_gen_move); d_rp_sto_move_ = up(t_.rp_sto_move); d_rp_v_ = up(t_.rp_v);
    i_corr_bus_ = up(t_.corr_bus); i_corr_of_bus_ = up(corr_of_bus_h_);
    any_off_unit_moved_ = std::count(t_.rp_gen_bus.begin(), t_.rp_gen_bus.end(), -2) > 0
                          || std::count(t_.rp_sto_bus.begin(), t_.rp_sto_bus.end(), -2) > 0;
}

// The per-slot fields, in one place: name -> (pool, width). The order is the Dev members'.
#define G2G_SLOT_FIELDS(X)                                                                              \
    X(row, 0, 1) X(it, 0, 1) X(last_unstable, 0, 1) X(last_status, 0, 1) X(first, 0, 1) X(changed, 0, 1)   \
    X(nb_outer, 0, 1) X(nb_passes, 0, 1) X(total_nr, 0, 1) X(old_total, 0, 1) X(nr_steps, 0, 1)            \
    X(v_version, 0, 1) X(solver_ok, 0, 1) X(unrealistic, 0, 1) X(check_unreal, 0, 1)                      \
    X(pass_check_unreal, 0, 1) X(last_fixing, 0, 1) X(n_active, 0, 1) X(last_loop, 0, 1) X(lim_base, 0, 1) \
    X(vmr_n, 0, 1) X(err, 0, 1) X(p_recheck, 0, 1)                                                      \
    X(active, 0, n_loops_) X(loop_iter, 0, n_loops_) X(done_at, 0, n_loops_)                             \
    X(masked, 1, t_.n_bus) X(lim_dirty, 1, t_.n_bus) X(pq, 1, t_.n_pin)                                  \
    X(held_q, 2, t_.n_ctrl) X(svc_on, 2, t_.n_ctrl) X(hvdc_regime, 0, t_.n_hvdc)                         \
    X(gen_tp, 2, t_.n_gen) X(sto_tp, 2, t_.n_sto) X(lost_mw, 2, 1)                                       \
    X(row_gp, 2, t_.n_gen) X(row_sp, 2, t_.n_sto)                                                        \
    X(ds_inn, 1, 1) X(ds_park, 1, n_units_) X(ds_parkh, 1, n_units_) X(sa_sh, 1, 1) X(sa, 2, 1)            \
    X(unsq, 2, 1) X(c_slack_w, 2, t_.n_slack)                                                            \
    X(ds_spark, 1, n_units_) X(ds_smw, 2, n_units_) X(slp_set, 1, t_.n_gen) X(slp, 2, t_.n_gen)          \
    X(tr_n, 0, 1) X(tr_li, 0, 1) X(tr_loop, 0, tcap_) X(tr_act, 0, tcap_) X(tr_el, 0, tcap_)            \
    X(tr_outer, 0, tcap_) X(tr_taken, 1, tcap_) X(tr_val, 2, tcap_) X(tr_lim, 2, tcap_)                 \
    X(s_gh, 1, trn_) X(s_tv, 2, trn_) X(s_tl, 2, trn_) X(s_hp, 2, trh_)                                  \
    X(sc_cnt, 0, t_.n_sc_sh) X(sc_ctl, 0, t_.n_sc) X(sc_pend, 1, t_.n_sc) X(sc_bnew, 2, t_.n_sc)          \
    X(sv_hid, 1, t_.n_svg) X(s_ord, 0, t_.n_sc_sh) X(s_cnt, 0, t_.n_sc_sh) X(s_mag, 2, t_.n_sc_sh)        \
    X(bc_pos, 0, t_.n_bc) X(bc_rpos, 0, t_.n_bc) X(bc_rq, 0, t_.n_bc) X(bc_rrq, 0, t_.n_bc)               \
    X(bc_ctl, 0, t_.n_bc) X(bc_rctl, 0, t_.n_bc) X(bc_out, 1, t_.n_bc) X(bc_mov, 1, t_.n_bc)              \
    X(bc_rt, 1, t_.n_bc) X(pc_pres, 1, t_.n_pc) X(pc_reg, 1, t_.n_pc) X(bc_atap, 2, t_.n_bc)              \
    X(bc_rtap, 2, t_.n_bc) X(bc_tap, 2, 8 * t_.n_bc)                                                     \
    X(tv_step, 0, 1) X(tv_lim, 2, 1) X(tv_hid, 1, t_.n_tvg) X(tv_en, 1, t_.n_gt) X(tv_rat, 2, 6 * t_.n_gt)  \
    X(tv_cand, 1, tvr_) X(tv_tvm, 2, tvr_) X(tv_load, 2, tvr_) X(tv_fz, 1, tvr_) X(tv_fzv, 2, tvr_)       \
    X(tv_susp, 1, tvb_) X(tv_par, 0, tvb_) X(tv_pv, 1, tvb_) X(tv_closed, 1, t_.n_br)                    \
    X(dset, 1, nc_) X(fset, 1, nc_) X(dre, 2, nc_) X(dim, 2, nc_) X(fq, 2, nc_)                          \
    X(vmr_bus, 0, vcap_) X(vmr_vm, 2, vcap_)                                                             \
    X(rl_present, 1, t_.rl_n) X(rl_tp, 2, t_.rl_n) X(rl_fq, 2, t_.rl_n) X(rl_state, 0, t_.rl_n)            \
    X(rl_nb, 0, t_.rl_n) X(rl_real, 0, t_.rl_n) X(rl_tq_s, 2, t_.rl_n) X(rl_tvm_s, 2, t_.rl_n)             \
    X(rl_hb_s, 1, t_.rl_n) X(rl_goff, 1, t_.rl_gen.size()) X(ctrl_off, 1, t_.n_ctrl)                     \
    X(ds_present, 1, n_units_) X(ds_inj, 2, n_units_) X(ds_cur, 2, n_units_) X(ds_ini, 2, n_units_)      \
    X(ds_ubus, 0, n_units_)                                                                              \
    X(hl_present, 1, n_hl_) X(hl_dir, 0, n_hl_)                                                          \
    X(c_pin, 1, t_.n_pin) X(c_plan_held, 1, t_.n_ctrl) X(c_held_q, 2, t_.n_ctrl) X(c_v_set, 2, t_.n_grp)  \
    X(c_hvdc_status, 0, t_.n_hvdc) X(c_sb_n, 0, 1) X(c_sb_bus, 0, nc_) X(c_sb_dp, 2, nc_) X(c_sb_dq, 2, nc_) \
    X(s_hold, 1, t_.n_grp) X(s_slope, 1, t_.n_grp) X(s_pq_k, 0, t_.rl_n) X(s_pq_lo, 0, t_.rl_n)            \
    X(s_pv_k, 0, t_.rl_n) X(s_mv_k, 0, t_.rl_n) X(s_pq_real, 1, t_.rl_n) X(s_pq_lim, 2, t_.rl_n)           \
    X(s_lo, 2, n_units_) X(s_hi, 2, n_units_) X(s_new, 2, n_units_) X(s_act, 1, n_units_)                \
    X(s_sat, 1, n_units_) X(s_hl, 0, n_hl_) X(s_cls, 0, t_.rl_n) X(s_qlim, 2, t_.rl_n)

template <typename P> struct KindOf;
template <> struct KindOf<int*> { static constexpr int v = 0; };
template <> struct KindOf<unsigned char*> { static constexpr int v = 1; };
template <> struct KindOf<double*> { static constexpr int v = 2; };

void DeviceOuterModel::alloc_slots(int S)
{
    S_ = S;
    slot_fields_.clear();
    field_of_.clear();
    w_int_ = w_u8_ = w_f64_ = 0;
    // each field's offset inside its pool's slot row: slot s's value k at pool[s * w_pool + off + k]
#define G2G_ADD(name, kind, width) {                                                     \
        static_assert(KindOf<decltype(Dev::name)>::v == (kind), "pool of " #name);        \
        const size_t w = static_cast<size_t>(std::max(1, static_cast<int>(width)));        \
        size_t& acc = (kind) == 0 ? w_int_ : (kind) == 1 ? w_u8_ : w_f64_;                \
        slot_fields_.push_back(Field{kind, w, acc});                                       \
        field_of_[#name] = slot_fields_.back();                                            \
        acc += w; }
    G2G_SLOT_FIELDS(G2G_ADD)
#undef G2G_ADD
    p_int_.assign(w_int_ * static_cast<size_t>(S), 0);
    p_u8_.assign(w_u8_ * static_cast<size_t>(S), 0);
    p_f64_.assign(w_f64_ * static_cast<size_t>(S), 0.);
    dev_.w_int = w_int_; dev_.w_u8 = w_u8_; dev_.w_f64 = w_f64_;
    size_t fi = 0;
#define G2G_BIND(name, kind, width) { bind(dev_.name, slot_fields_[fi++].off); }
    G2G_SLOT_FIELDS(G2G_BIND)
#undef G2G_BIND
    ctl_.assign(static_cast<size_t>(S), SlotControls());
}

void DeviceOuterModel::alloc_rows(int R)
{
    n_rows_ = R;
    const size_t r = static_cast<size_t>(std::max(R, 1));
    // RowResult's defaults: status -1 (never ran), no iteration, every hvdc line KEEP (2)
    std::vector<int> r0(r * (5 + static_cast<size_t>(std::max(t_.n_hvdc, 1))), 0);
    std::fill(r0.begin(), r0.begin() + static_cast<ptrdiff_t>(r), -1);                  // r_status
    std::fill(r0.begin() + static_cast<ptrdiff_t>(2 * r), r0.begin() + static_cast<ptrdiff_t>(3 * r), -1);   // r_failed
    std::fill(r0.begin() + static_cast<ptrdiff_t>(5 * r), r0.end(), 2);                 // r_hvdc
    r_int_.assign(r0.begin(), r0.end());
    r_f64_.assign(r * static_cast<size_t>(std::max(n_units_, 1)), 0.);
    dev_.r_status = raw(r_int_);
    dev_.r_unreal = dev_.r_status + r;
    dev_.r_failed = dev_.r_unreal + r;
    dev_.r_outer = dev_.r_failed + r;
    dev_.r_nr = dev_.r_outer + r;
    dev_.r_hvdc = dev_.r_nr + r;
    dev_.r_unit_mw = raw(r_f64_);
    r_sc_.assign(r * static_cast<size_t>(std::max(t_.n_sc_sh, 1)), 0);
    dev_.r_sc = raw(r_sc_);
    r_pos_.assign(r * static_cast<size_t>(std::max(t_.n_bc, 1)), 0);
    r_rpos_.assign(r * static_cast<size_t>(std::max(t_.n_bc, 1)), 0);
    dev_.r_pos = raw(r_pos_);
    dev_.r_rpos = raw(r_rpos_);
    results_.assign(r, RowResult());
    adj_.assign(keep_adj_ ? r : 0, RowResult());
    trace_.assign(t_.keep_trace ? r : 0, {});
    fin_built_.assign(r, 0);
    rows_on_host_ = false;
}

void DeviceOuterModel::build_dev()
{
    Dev& d = dev_;
    d.n_bus = t_.n_bus; d.n_gen = t_.n_gen; d.n_sto = t_.n_sto; d.n_watch = t_.n_watch; d.n_loops = n_loops_;
    d.n_pin = t_.n_pin; d.n_ctrl = t_.n_ctrl; d.n_grp = t_.n_grp; d.n_hvdc = t_.n_hvdc; d.rl_n = t_.rl_n;
    d.rl_n_bus_q = t_.rl_n_bus_q;
    d.n_units = n_units_; d.n_hl = n_hl_; d.n_mon = n_mon_; d.nc = nc_; d.vcap = vcap_;
    d.n_sc = t_.n_sc; d.n_sc_sh = t_.n_sc_sh; d.n_svg = t_.n_svg; d.n_bc = host_->layout().n_bc; d.n_pc = t_.n_pc;
    d.bc_trafo = raw(i_bc_trafo_); d.bc_phase = raw(i_bc_phase_); d.bc_b1 = raw(i_bc_b1_); d.bc_b2 = raw(i_bc_b2_);
    d.bc_has_ptc = raw(i_bc_has_ptc_); d.bc_plo = raw(i_bc_plo_); d.bc_phi = raw(i_bc_phi_); d.bc_rlo = raw(i_bc_rlo_);
    d.bc_rhi = raw(i_bc_rhi_); d.bc_tab_ptr = raw(i_bc_tab_ptr_); d.bc_alpha_ptr = raw(i_bc_alpha_ptr_);
    d.bc_sign = raw(d_bc_sign_); d.bc_rexp = raw(d_bc_rexp_); d.bc_tab_ratio = raw(d_bc_tab_ratio_);
    d.bc_tab_block = raw(d_bc_tab_block_); d.bc_tab_shift = raw(d_bc_tab_shift_); d.bc_alpha = raw(d_bc_alpha_);
    d.pc_trafo = raw(i_pc_trafo_); d.pc_e = raw(i_pc_e_); d.pc_mode = raw(i_pc_mode_); d.pc_side = raw(i_pc_side_);
    d.pc_vn_kv = raw(d_pc_vn_kv_); d.pc_target = raw(d_pc_target_);
    d.has_tvc = t_.has_tvc; d.n_tvg = t_.n_tvg; d.n_gt = t_.n_gt; d.n_br = t_.n_br;
    d.n_line = t_.n_br - static_cast<int>(t_.trafo_gt.size()); d.tv_use_initial = t_.tv_use_initial;
    d.bc_ratio = raw(i_bc_ratio_); d.tvg_ptr = raw(i_tvg_ptr_); d.tvg_bus = raw(i_tvg_bus_); d.gt_trafo = raw(i_gt_trafo_);
    d.gt_e = raw(i_gt_e_); d.br_b1 = raw(i_br_b1_); d.br_b2 = raw(i_br_b2_); d.br_trafo = raw(i_br_trafo_);
    d.trafo_gt = raw(i_trafo_gt_); d.pin_of_bus = raw(i_pin_of_bus_); d.tvg_target = raw(d_tvg_target_); d.tvg_hdb = raw(d_tvg_hdb_);
    d.sc_bus = raw(i_sc_bus_); d.sc_sh_ptr = raw(i_sc_sh_ptr_); d.sh_min = raw(i_sh_min_); d.sh_max = raw(i_sh_max_);
    d.sh_count0 = raw(i_sh_count0_); d.sh_status = raw(i_sh_status_); d.sh_b_ptr = raw(i_sh_b_ptr_);
    d.svg_ptr = raw(i_svg_ptr_); d.svg_e = raw(i_svg_e_); d.svg_bus = raw(i_svg_bus_); d.sh_b = raw(d_sh_b_);
    d.dist = t_.dist; d.n_slack = t_.n_slack; d.ref_slack = t_.ref_slack; d.slack_idx = raw(i_slack_idx_);
    d.slack_bus = t_.slack_bus; d.max_outer = t_.max_outer_iterations; d.robust_mode = t_.robust_mode; d.sn = t_.sn;
    d.ds_mismatch_mw = t_.ds_mismatch_mw; d.ds_eps_mw = t_.ds_eps_mw; d.ds_moved_fraction = t_.ds_moved_fraction;
    d.ds_fail = t_.ds_fail; d.rl_max_switch = t_.rl_max_switch; d.rl_robust = t_.rl_robust;
    d.rl_max_mismatch = t_.rl_max_mismatch; d.rl_base_mva = t_.rl_base_mva; d.rl_vmin = t_.rl_vmin; d.rl_vmax = t_.rl_vmax;
    d.rl_margin = t_.rl_margin; d.rl_restart = t_.rl_restart;
    d.widx = raw(i_widx_); d.loop_kind = raw(i_loop_kind_);
    d.gen_tp0 = raw(d_gen_tp0_); d.gen_min_q = raw(d_gen_min_q_); d.gen_max_q = raw(d_gen_max_q_);
    d.gen_target_q = raw(d_gen_tq_); d.gen_raw_q = raw(d_gen_raw_q_);
    d.gen_vreg = raw(i_gen_vreg_); d.curve_ptr = raw(i_curve_ptr_); d.gen_rl_ptr = raw(i_gen_rl_ptr_); d.gen_rl_bus = raw(i_gen_rl_bus_);
    d.curve_p = raw(d_curve_p_); d.curve_min = raw(d_curve_min_); d.curve_max = raw(d_curve_max_);
    d.sto_tp0 = raw(d_sto_tp0_); d.svc_b_min = raw(d_svc_bmin_); d.svc_b_max = raw(d_svc_bmax_);
    d.st_min1 = raw(d_st_min1_); d.st_max1 = raw(d_st_max1_); d.st_min2 = raw(d_st_min2_); d.st_max2 = raw(d_st_max2_);
    d.rl_bus = raw(i_rl_bus_); d.rl_local = raw(i_rl_local_); d.rl_reg = raw(i_rl_reg_); d.rl_group = raw(i_rl_group_);
    d.rl_mon = raw(i_rl_mon_); d.rl_pin = raw(i_rl_pin_); d.rl_has_base = raw(i_rl_has_base_);
    d.rl_min_q = raw(d_rl_min_q_); d.rl_max_q = raw(d_rl_max_q_); d.rl_tvm = raw(d_rl_tvm_); d.rl_nv = raw(d_rl_nv_);
    d.rl_tq = raw(d_rl_tq_); d.rl_bmin = raw(d_rl_bmin_); d.rl_bmax = raw(d_rl_bmax_);
    d.rl_gen_ptr = raw(i_rl_gen_ptr_); d.rl_gen = raw(i_rl_gen_); d.rl_svc_ptr = raw(i_rl_svc_ptr_); d.rl_svc = raw(i_rl_svc_);
    d.rl_ctrl_ptr = raw(i_rl_ctrl_ptr_); d.rl_ctrl = raw(i_rl_ctrl_);
    d.vc_bus = raw(i_vc_bus_); d.vc_kind = raw(i_vc_kind_); d.vc_elem = raw(i_vc_elem_); d.vc_group = raw(i_vc_group_);
    d.vc_plan_held = raw(i_vc_plan_held_); d.vc_at_ptr = raw(i_vc_at_ptr_); d.vc_at = raw(i_vc_at_); d.svc_ctrl = raw(i_svc_ctrl_);
    d.vc_slope = raw(d_vc_slope_); d.vc_q_plan = raw(d_vc_q_plan_); d.vc_v_set = raw(d_vc_v_set_);
    d.hv_b1 = raw(i_hv_b1_); d.hv_b2 = raw(i_hv_b2_); d.hv_st0 = raw(i_hv_st0_);
    d.hv_p0 = raw(d_hv_p0_); d.hv_k = raw(d_hv_k_); d.hv_lf1 = raw(d_hv_lf1_); d.hv_lf2 = raw(d_hv_lf2_);
    d.hv_r = raw(d_hv_r_); d.hv_pmax12 = raw(d_hv_pmax12_); d.hv_pmax21 = raw(d_hv_pmax21_);
    d.hvdc_at_bus = raw(u_hvdc_at_bus_);
    d.hl_id = raw(i_hl_id_); d.hl_b1 = raw(i_hl_b1_); d.hl_b2 = raw(i_hl_b2_); d.hl_entry = raw(i_hl_entry_);
    d.hl_p0 = raw(d_hl_p0_); d.hl_k = raw(d_hl_k_); d.hl_pmax12 = raw(d_hl_pmax12_); d.hl_pmax21 = raw(d_hl_pmax21_);
    d.mon_svc = raw(i_mon_svc_); d.mon_reg = raw(i_mon_reg_); d.mon_bus = raw(i_mon_bus_);
    d.trace = t_.keep_trace; d.tcap = tcap_; d.mon_vn = raw(d_mon_vn_);
    d.mon_low = raw(d_mon_low_); d.mon_high = raw(d_mon_high_); d.svc_low_t = raw(d_svc_low_t_); d.svc_high_t = raw(d_svc_high_t_);
    d.ds_el = raw(i_ds_el_); d.ds_kind = raw(i_ds_kind_); d.ds_bus = raw(i_ds_bus_);
    d.ds_w = raw(d_ds_w_); d.ds_min = raw(d_ds_min_); d.ds_max = raw(d_ds_max_);
    d.ds_reserved = t_.ds_reserved; d.rp_n = t_.rp_n;
    d.n_rp_gen = static_cast<int>(t_.rp_gen_bus.size()); d.n_rp_sto = static_cast<int>(t_.rp_sto_bus.size());
    d.rp_gen_bus = raw(i_rp_gen_bus_); d.rp_sto_bus = raw(i_rp_sto_bus_); d.rl_rp_state = raw(i_rl_rp_state_);
    d.rl_sto_ptr = raw(i_rl_sto_ptr_); d.rl_sto = raw(i_rl_sto_); d.rl_has_sto = raw(i_rl_has_sto_);
    d.rp_gen_move = raw(d_rp_gen_move_); d.rp_sto_move = raw(d_rp_sto_move_); d.rp_v = raw(d_rp_v_);
    d.corr_bus = raw(i_corr_bus_); d.corr_of_bus = raw(i_corr_of_bus_);
}

void DeviceOuterModel::begin(int n_slots, int n_rows)
{
    host_->begin(n_slots, n_rows);
    alloc_slots(n_slots);
    alloc_rows(n_rows);
    build_dev();
    have_tmpl_ = false;
    n_host_loads_ = 0;
    phantom_ready_ = false;
    base_row_.reset();
    point_base_.clear();
    row_base_.clear();
    point_tmpl_.clear();
    n_pt_ = 0;
    tmpl_row_.reset();
    const char* env = std::getenv("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE");
    check_tmpl_ = env != nullptr && env[0] != '\0' && env[0] != '0';
}

void DeviceOuterModel::note_plain_rows(const RowOverrides& ov)
{
    // a row is plain when nothing the loops start from depends on it. The branches it opens (and
    // the buses it masks) reach BranchControl (its transformers out of the row), PhaseControl's
    // shifters, ShuntVoltageControl's groups and TransformerVoltageControl's image (its closed
    // branches, the freeze candidates' step-up test, its groups; with or without a transformer
    // BranchControl reserves): with any of them, such a row loads through the host
    bool topo_dependent = t_.n_bc > 0 || t_.n_sc > 0;
    for (int k : t_.loop_kind)
        topo_dependent = topo_dependent || k == DL_PHASE_CONTROL || k == DL_SHUNT_VOLTAGE_CONTROL ||
                         k == DL_TRANSFORMER_VOLTAGE_CONTROL;
    size_t n = 0;
    for (size_t m : {ov.masked.size(), ov.pq_buses.size(), ov.vc_tripped.size(), ov.hvdc_off.size(),
                     ov.gen_bus.size(), ov.lost_p_mw.size(), ov.branch_off.size()}) n = std::max(n, m);
    const size_t ng = static_cast<size_t>(t_.n_gen), ns = static_cast<size_t>(t_.n_sto), ngr = static_cast<size_t>(t_.n_grp);
    if (ng > 0) n = std::max({n, ov.gen_off.size() / ng, ov.gen_vm.size() / ng});
    if (ns > 0) n = std::max(n, ov.storage_off.size() / ns);
    if (ngr > 0) n = std::max(n, ov.vc_v_set.size() / ngr);
    plain_.assign(n, 1);
    row_loss_.assign(n, std::numeric_limits<double>::quiet_NaN());
    mask_only_.assign(n, 0);
    masked_rows_.assign(n, {});
    auto any_row = [](const auto& m, size_t r, size_t w, auto pred) {
        if (w == 0 || (r + 1) * w > m.size()) return false;
        for (size_t k = r * w; k < (r + 1) * w; ++k) if (pred(m[k])) return true;
        return false;
    };
    // a row of a 0 / 1 mask with something set: an OR over the row, no branch per entry (the
    // masks are dense, one row per generator / load of the grid: thousands of bytes per row)
    auto any_set = [](const std::vector<char>& m, size_t r, size_t w) {
        if (w == 0 || (r + 1) * w > m.size()) return false;
        unsigned char acc = 0;
        for (size_t k = r * w; k < (r + 1) * w; ++k) acc |= static_cast<unsigned char>(m[k]);
        return acc != 0;
    };
    // the loads a row trips: TransformerVoltageControl freezes a bus at mis_bus - the row's own load Q
    // (device_tvc_image's tv_load); nothing else reads them (DistributedSlack takes a row's loss
    // through its lost_p_mw: share_lost_kernel)
    const size_t nl = static_cast<size_t>(t_.n_load);
    const bool loads_matter = t_.has_tvc != 0;
    for (size_t r = 0; r < n; ++r) {
        bool changes = false;
        const bool has_mask = r < ov.masked.size() && !ov.masked[r].empty();
        changes = changes || (r < ov.pq_buses.size() && !ov.pq_buses[r].empty());
        changes = changes || (r < ov.vc_tripped.size() && !ov.vc_tripped[r].empty());
        changes = changes || (r < ov.hvdc_off.size() && !ov.hvdc_off[r].empty());
        changes = changes || (r < ov.gen_bus.size() && !ov.gen_bus[r].empty());
        // its loss (redistribute_slack) is shared on the device after the copy (share_lost_kernel): row_loss_
        row_loss_[r] = r < ov.lost_p_mw.size() ? ov.lost_p_mw[r] : std::numeric_limits<double>::quiet_NaN();
        changes = changes || any_set(ov.gen_off, r, ng);
        changes = changes || any_row(ov.gen_vm, r, ng, [](double v) { return !std::isnan(v); });
        changes = changes || any_set(ov.storage_off, r, ns);
        changes = changes || any_row(ov.vc_v_set, r, ngr, [](double v) { return !std::isnan(v); });
        changes = changes || (loads_matter && any_set(ov.load_off, r, nl));
        if (topo_dependent) {
            const bool opens = r < ov.branch_off.size() && !ov.branch_off[r].empty();
            changes = changes || opens || has_mask;
        }
        // masks on spare buses only: still plain (copied from a template, then masked on the device)
        bool real_mask = false;
        if (has_mask) for (int b : ov.masked[r]) real_mask = real_mask || !spare_bus(b);
        plain_[r] = (changes || real_mask) ? 0 : 1;
        mask_only_[r] = (!changes && has_mask) ? 1 : 0;
        if (has_mask) {
            masked_rows_[r] = ov.masked[r];
            std::sort(masked_rows_[r].begin(), masked_rows_[r].end());
        }
    }
}

// a masked row loads from the template unless the mask touches what the template's load decided
// from the whole grid: DistributedSlack left without a unit off the slack bus or ReactiveLimits
// without a bus (the loop goes out), a bus it froze or reset (a resumed frozen bus the row's plan
// no longer has)
int DeviceOuterModel::mask_source(int row) const
{
    if (row < 0 || own_targets(row) || static_cast<size_t>(row) >= mask_only_.size() || !mask_only_[static_cast<size_t>(row)])
        return -2;
    if (has_loss(row) && !loss_on_device(row)) return -2;
    // the plain template, or the template of the point the row resumes from
    const HostRow* tmpl = nullptr;
    int source = -2;
    if (resumes(row)) {
        auto it = point_tmpl_.find(resume_of_row_[static_cast<size_t>(row)]);
        if (it == point_tmpl_.end()) return -2;
        const std::vector<int>& m = masked_rows_[static_cast<size_t>(row)];
        if (!std::includes(m.begin(), m.end(), it->second.masks.begin(), it->second.masks.end())) return -2;
        tmpl = it->second.host.get();
        source = it->second.idx;
    } else {
        if (!tmpl_row_) return -2;
        tmpl = tmpl_row_.get();
        source = -1;
    }
    const std::vector<int>& m = masked_rows_[static_cast<size_t>(row)];
    auto is_masked = [&](int b) { return std::binary_search(m.begin(), m.end(), b); };
    const auto& ti = tmpl->i;
    const auto& tu = tmpl->u;
    const size_t act = field_of_.at("active").off, n_act = static_cast<size_t>(ti[field_of_.at("n_active").off]);
    for (size_t k = 0; k < n_act; ++k) {
        if (t_.loop_kind[static_cast<size_t>(ti[act + k])] != DL_DISTRIBUTED_SLACK) continue;
        bool off_slack = false;
        for (int u = 0; u < n_units_ && !off_slack; ++u) {
            const int b = t_.ds_bus[static_cast<size_t>(u)];
            off_slack = !is_masked(b) && b != t_.slack_bus;
        }
        if (!off_slack) return -2;
    }
    // ReactiveLimits needed by the template but not by the row: the row's plan is the grid's without
    // its masked buses (rl_plan_masked), the monitors kept, and set_active drops the loop when that
    // leaves nothing -- and with it moves the robust mode's last_fixing. A row whose masks cover every
    // bus_q entry the template has, on a grid without monitors, loads through the host
    const size_t rl_pres = field_of_.at("rl_present").off;
    for (size_t k = 0; k < n_act; ++k) {
        if (t_.loop_kind[static_cast<size_t>(ti[act + k])] != DL_REACTIVE_LIMITS) continue;
        bool needed = t_.rl_has_monitor != 0 || t_.rl_n > t_.rl_n_bus_q;
        for (int e = 0; e < t_.rl_n_bus_q && !needed; ++e) {
            const int b = t_.rl_bus[static_cast<size_t>(e)];
            needed = tu[rl_pres + static_cast<size_t>(e)] && (b < 0 || !is_masked(b));
        }
        if (!needed) return -2;
    }
    const size_t fset = field_of_.at("fset").off;
    for (int ci = 0; ci < nc_; ++ci)
        if (tu[fset + static_cast<size_t>(ci)] && is_masked(t_.corr_bus[static_cast<size_t>(ci)])) return -2;
    const size_t vb = field_of_.at("vmr_bus").off, nv = static_cast<size_t>(ti[field_of_.at("vmr_n").off]);
    for (size_t k = 0; k < nv; ++k) if (is_masked(ti[vb + k])) return -2;
    return source;
}

void DeviceOuterModel::compare_rows(const HostRow& host, const HostRow& dev, int row) const
{
    for (const auto& kv : field_of_) {
        const Field& f = kv.second;
        if (kv.first.rfind("c_", 0) == 0) continue;   // controls: materialize writes them after
        // the decisions the device recorded (its share's), flushed with the round's: the traces are
        // compared as the rows' results
        if (kv.first.rfind("tr_", 0) == 0) continue;
        if (kv.first.rfind("s_", 0) == 0) continue;    // scratch (distribute's): no slot state
        for (size_t k = 0; k < f.width; ++k) {
            const size_t at = f.off + k;
            bool same;
            if (f.kind == 0) same = host.i[at] == dev.i[at];
            else if (f.kind == 1) same = host.u[at] == dev.u[at];
            else same = host.f[at] == dev.f[at] || (std::isnan(host.f[at]) && std::isnan(dev.f[at]));
            if (!same)
                throw std::logic_error("device outer model: masked row " + std::to_string(row) + " loaded on the device "
                                       "differs from the host load in its field " + kv.first + "[" + std::to_string(k) + "]");
        }
    }
}

HostRow DeviceOuterModel::unmasked(const HostRow& hr, int row) const
{
    HostRow out = hr;
    if (!has_masks(row)) return out;
    const size_t off = field_of_.at("masked").off;
    for (int b : masked_rows_[static_cast<size_t>(row)])
        if (spare_bus(b)) out.u[off + static_cast<size_t>(b)] = 0;
    return out;
}

// a point's template appended to the device buffers (grown by doubling: rare reallocations)
void DeviceOuterModel::add_point_template(int point, const HostRow& hr)
{
    PointTemplate& pt = point_tmpl_[point];
    pt.host = std::make_unique<HostRow>(hr);
    pt.idx = n_pt_++;
    auto append = [&](auto& buf, const auto& row, size_t w) {
        using T = typename std::remove_reference<decltype(buf)>::type::value_type;
        const size_t need = static_cast<size_t>(n_pt_) * w;
        if (buf.size() < need) {
            thrust::device_vector<T> grown(std::max(need, 2 * buf.size()));
            if (!buf.empty())
                check_cuda(cudaMemcpyAsync(raw(grown), raw(buf), buf.size() * sizeof(T), cudaMemcpyDeviceToDevice, stream_),
                           "point templates");
            check_cuda(cudaStreamSynchronize(stream_), "point templates");
            buf.swap(grown);
        }
        check_cuda(cudaMemcpyAsync(raw(buf) + static_cast<size_t>(pt.idx) * w, row.data(), w * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_), "point templates");
    };
    append(pt_i_, hr.i, w_int_);
    append(pt_u_, hr.u, w_u8_);
    append(pt_f_, hr.f, w_f64_);
}

void DeviceOuterModel::compare_with_template(const HostRow& hr, int row) const
{
    compare_rows_with(hr, *tmpl_row_, row);
}

void DeviceOuterModel::compare_rows_with(const HostRow& hr, const HostRow& tmpl, int row) const
{
    const size_t row_off = field_of_.at("row").off;
    for (const auto& kv : field_of_) {
        const Field& f = kv.second;
        for (size_t k = 0; k < f.width; ++k) {
            const size_t at = f.off + k;
            bool same;
            if (f.kind == 0) same = at == row_off || hr.i[at] == tmpl.i[at];
            else if (f.kind == 1) same = hr.u[at] == tmpl.u[at];
            else same = hr.f[at] == tmpl.f[at] || (std::isnan(hr.f[at]) && std::isnan(tmpl.f[at]));
            if (!same)
                throw std::logic_error("device outer model: row " + std::to_string(row) + " differs from the template "
                                       "in its field " + kv.first + "[" + std::to_string(k) + "]");
        }
    }
}


template <typename T>
void set_field(std::vector<T>& row, size_t off, const std::vector<T>& v)
{
    std::copy(v.begin(), v.end(), row.begin() + static_cast<ptrdiff_t>(off));
}

void DeviceOuterModel::fill_row(const DeviceSlotImage& img, HostRow& hr)
{
    // the targets every row starts from must be the same (rows setting their own are M4); a
    // phantom slot (no row) takes no resume point and writes no result: it is not compared
    // a row resuming from another row's point starts from that point's targets: the same for every
    // row resuming from it, kept per point (the base of their results)
    auto same = [](const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) return false;
        for (size_t k = 0; k < a.size(); ++k)
            if (!(a[k] == b[k] || (std::isnan(a[k]) && std::isnan(b[k])))) return false;
        return true;
    };
    // The references are taken from the rows the templates are captured from (a plain row; a point's
    // point_row), whose targets every copy shares; any other row keeps its own unless it equals its
    // reference (a row loaded before its reference exists keeps its own)
    const auto own_start = [&](const std::vector<double>* gen, const std::vector<double>* sto) {
        if (gen == nullptr || !same(*gen, img.gen_target_p) || !same(*sto, img.sto_target_p))
            row_base_[img.row] = std::make_pair(img.gen_target_p, img.sto_target_p);
    };
    if (img.row < 0) {
    } else if (resumes(img.row)) {
        const int k = resume_of_row_[static_cast<size_t>(img.row)];
        auto it = point_base_.find(k);
        if (it == point_base_.end() && point_row(img.row))
            point_base_.emplace(k, std::make_pair(img.gen_target_p, img.sto_target_p));
        else if (it == point_base_.end()) own_start(nullptr, nullptr);
        else own_start(&it->second.first, &it->second.second);
    } else if (plain(img.row) && !have_tmpl_) {
        gen_tp_tmpl_ = img.gen_target_p; sto_tp_tmpl_ = img.sto_target_p; have_tmpl_ = true;
    } else {
        // a row starting from its own targets (a generator it disconnects, ...): kept for its result
        own_start(have_tmpl_ ? &gen_tp_tmpl_ : nullptr, have_tmpl_ ? &sto_tp_tmpl_ : nullptr);
    }
    hr.i.assign(w_int_, 0);
    hr.u.assign(w_u8_, 0);
    hr.f.assign(w_f64_, 0.);
    auto I = [&](const char* name) { return field_of_.at(name).off; };
    // the driver state at a row's start (SlotState's defaults, load's settings)
    const std::pair<const char*, int> scalars[] = {
        {"row", img.row}, {"it", 0}, {"last_unstable", -1}, {"last_status", STABLE}, {"first", 1},
        {"changed", 0}, {"nb_outer", 0}, {"nb_passes", 0}, {"total_nr", 0}, {"old_total", 0}, {"nr_steps", 0},
        {"v_version", 0}, {"solver_ok", 1}, {"unrealistic", 0}, {"check_unreal", img.check_unrealistic},
        {"pass_check_unreal", 0}, {"last_fixing", img.last_fixing}, {"n_active", static_cast<int>(img.active.size())},
        {"last_loop", -1}, {"lim_base", img.lim_base}, {"vmr_n", static_cast<int>(img.vm_reset_bus.size())}, {"err", 0}};
    for (const auto& kv : scalars) hr.i[I(kv.first)] = kv.second;
    set_field(hr.i, I("active"), img.active);
    for (int k = 0; k < n_loops_; ++k) hr.i[I("done_at") + static_cast<size_t>(k)] = -1;
    set_field(hr.u, I("masked"), img.masked_bus);
    for (int b : img.lim_dirty_bus) hr.u[I("lim_dirty") + static_cast<size_t>(b)] = 1;
    set_field(hr.u, I("pq"), img.pq);
    set_field(hr.f, I("held_q"), img.held_q);
    set_field(hr.f, I("svc_on"), img.svc_on);
    set_field(hr.i, I("hvdc_regime"), img.hvdc_regime);
    set_field(hr.f, I("gen_tp"), img.gen_target_p);
    set_field(hr.f, I("row_gp"), img.row_gen_p);
    hr.u[I("ds_inn")] = img.ds_in_newton;
    set_field(hr.u, I("ds_park"), img.ds_parked);
    set_field(hr.u, I("ds_parkh"), img.ds_parked_high);
    hr.u[I("sa_sh")] = img.sa_shared;
    hr.f[I("sa")] = img.sa;
    hr.f[I("unsq")] = img.unsolved_q_mvar;
    set_field(hr.u, I("ds_spark"), img.ds_solved_parked);
    set_field(hr.f, I("ds_smw"), img.ds_solved_mw);
    set_field(hr.u, I("slp_set"), img.slope_set);
    set_field(hr.i, I("sc_cnt"), img.sc_counts);
    set_field(hr.i, I("sc_ctl"), img.sc_ctl);
    set_field(hr.u, I("sc_pend"), img.sc_pending);
    set_field(hr.f, I("sc_bnew"), img.sc_b_new);
    set_field(hr.u, I("sv_hid"), img.sv_hidden);
    set_field(hr.i, I("bc_pos"), img.bc_pos); set_field(hr.i, I("bc_rpos"), img.bc_rpos);
    set_field(hr.i, I("bc_rq"), img.bc_req_tap); set_field(hr.i, I("bc_rrq"), img.bc_req_rtap);
    set_field(hr.i, I("bc_ctl"), img.bc_ctl); set_field(hr.i, I("bc_rctl"), img.bc_rctl);
    set_field(hr.u, I("bc_out"), img.bc_out); set_field(hr.u, I("bc_mov"), img.bc_moved);
    set_field(hr.u, I("pc_pres"), img.pc_present); set_field(hr.u, I("pc_reg"), img.pc_regulates);
    set_field(hr.f, I("bc_atap"), img.bc_a_tap); set_field(hr.f, I("bc_rtap"), img.bc_rho_tap);
    set_field(hr.f, I("bc_tap"), img.bc_tap);
    if (t_.has_tvc) {
        hr.i[I("tv_step")] = img.tvc_step;
        hr.f[I("tv_lim")] = img.tvc_limit;
        set_field(hr.u, I("tv_hid"), img.tv_hidden); set_field(hr.u, I("tv_en"), img.tv_enabled);
        set_field(hr.f, I("tv_rat"), img.tv_ratios);
        set_field(hr.u, I("tv_cand"), img.tv_cand); set_field(hr.f, I("tv_tvm"), img.tv_tvm);
        set_field(hr.f, I("tv_load"), img.tv_load); set_field(hr.u, I("tv_fz"), img.tv_frozen);
        set_field(hr.f, I("tv_fzv"), img.tv_frozen_vm); set_field(hr.u, I("tv_susp"), img.tv_susp);
        set_field(hr.u, I("tv_closed"), img.br_closed);
    }
    set_field(hr.f, I("slp"), img.slope);
    set_field(hr.f, I("row_sp"), img.row_sto_p);
    hr.f[I("lost_mw")] = img.lost_mw;
    set_field(hr.f, I("sto_tp"), img.sto_target_p);
    // the Sbus correction, densely over corr_bus
    for (size_t k = 0; k < img.delta_bus.size(); ++k) {
        const int ci = corr_of_bus_h_[static_cast<size_t>(img.delta_bus[k])];
        if (ci < 0) throw DeviceNotImplemented("a row moving targets off the planned buses");
        hr.u[I("dset") + static_cast<size_t>(ci)] = 1;
        hr.f[I("dre") + static_cast<size_t>(ci)] = img.delta_p[k];
        hr.f[I("dim") + static_cast<size_t>(ci)] = img.delta_q[k];
    }
    for (size_t k = 0; k < img.freeze_bus.size(); ++k) {
        const int ci = corr_of_bus_h_[static_cast<size_t>(img.freeze_bus[k])];
        if (ci < 0) throw DeviceNotImplemented("a row freezing a bus off the planned buses");
        hr.u[I("fset") + static_cast<size_t>(ci)] = 1;
        hr.f[I("fq") + static_cast<size_t>(ci)] = img.freeze_q[k];
    }
    if (static_cast<int>(img.vm_reset_bus.size()) > vcap_) throw DeviceNotImplemented("that many |V| resets");
    set_field(hr.i, I("vmr_bus"), img.vm_reset_bus);
    set_field(hr.f, I("vmr_vm"), img.vm_reset_vm);
    set_field(hr.u, I("rl_present"), img.rl_present);
    set_field(hr.f, I("rl_tp"), img.rl_target_p);
    set_field(hr.f, I("rl_fq"), img.rl_frozen_q);
    set_field(hr.i, I("rl_state"), img.rl_state);
    set_field(hr.i, I("rl_nb"), img.rl_nb_pv_pq);
    set_field(hr.i, I("rl_real"), img.rl_realistic);
    set_field(hr.f, I("rl_tq_s"), img.rl_target_q);
    set_field(hr.f, I("rl_tvm_s"), img.rl_target_vm);
    set_field(hr.u, I("rl_hb_s"), img.rl_has_base);
    set_field(hr.u, I("rl_goff"), img.rl_goff);
    set_field(hr.u, I("ctrl_off"), img.ctrl_off);
    set_field(hr.u, I("ds_present"), img.ds_present);
    set_field(hr.i, I("ds_ubus"), img.ds_bus);
    set_field(hr.f, I("ds_inj"), img.ds_injection);
    set_field(hr.f, I("ds_cur"), img.ds_current);
    set_field(hr.f, I("ds_ini"), img.ds_initial);
    set_field(hr.u, I("hl_present"), img.hl_present);
    set_field(hr.i, I("hl_dir"), img.hl_frozen_dir);
}

// the template slot row into each listed slot, then each slot's own row index
template <typename T>
__global__ void template_rows_kernel(T* pool, const T* tmpl, const int* slots, int n, size_t w)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(n) * w) return;
    pool[static_cast<size_t>(slots[t / w]) * w + t % w] = tmpl[t % w];
}
// the differences of n rows from their baseline: entry j at (row k, offset q) = pos[j] / w, pos[j] % w
template <typename T>
__global__ void scatter_diffs_kernel(T* pool, const long long* pos, const T* val, const int* slots, size_t m, size_t w)
{
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const size_t k = static_cast<size_t>(pos[j]) / w, q = static_cast<size_t>(pos[j]) % w;
    pool[static_cast<size_t>(slots[k]) * w + q] = val[j];
}

// row idx[i] of the templates into slot slots[i]
template <typename T>
__global__ void indexed_rows_kernel(T* pool, const T* tmpl, const int* slots, const int* idx, int n, size_t w)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(n) * w) return;
    const size_t i = t / w, k = t % w;
    pool[static_cast<size_t>(slots[i]) * w + k] = tmpl[static_cast<size_t>(idx[i]) * w + k];
}

__global__ void set_rows_kernel(int* pool, const int* slots, const int* rows, int n, size_t w, size_t off)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pool[static_cast<size_t>(slots[i]) * w + off] = rows[i];
}

// A masked row on top of the template (load() of a row whose only change is its masked buses):
// the buses masked, the controllers on them held at 0, its ReactiveLimits plan the grid's without
// the bus_q_check entries on them (rl_plan_masked: the monitors stay), its DistributedSlack units
// without the ones on them. One thread per listed slot; mask_ptr / mask_bus: CSR per listed slot.
__global__ void mask_rows_kernel(Dev d, const int* slots, int n, const int* mask_ptr, const int* mask_bus)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = slots[i];
    unsigned char* masked = SLOT(masked, d.n_bus);
    double* held_q = SLOT(held_q, d.n_ctrl);
    for (int j = mask_ptr[i]; j < mask_ptr[i + 1]; ++j) {
        const int b = mask_bus[j];
        masked[b] = 1;
        for (int q = d.vc_at_ptr[b]; q < d.vc_at_ptr[b + 1]; ++q) held_q[d.vc_at[q]] = 0.;
    }
    // what the mask removes takes the state of an absent entry (the host image's zeros: never read)
    unsigned char* rl_present = SLOT(rl_present, d.rl_n);
    for (int k = 0; k < d.rl_n_bus_q; ++k) {
        if (d.rl_bus[k] < 0 || !masked[d.rl_bus[k]]) continue;
        rl_present[k] = 0;
        SLOT(rl_tp, d.rl_n)[k] = 0.; SLOT(rl_fq, d.rl_n)[k] = 0.;
        SLOT(rl_state, d.rl_n)[k] = 0; SLOT(rl_nb, d.rl_n)[k] = 0; SLOT(rl_real, d.rl_n)[k] = 0;
    }
    unsigned char* ds_present = SLOT(ds_present, d.n_units);
    const int* ubus = SLOT(ds_ubus, d.n_units);
    for (int k = 0; k < d.n_units; ++k) {
        if (ubus[k] < 0 || !masked[ubus[k]]) continue;
        ds_present[k] = 0;
        SLOT(ds_inj, d.n_units)[k] = 0.; SLOT(ds_cur, d.n_units)[k] = 0.; SLOT(ds_ini, d.n_units)[k] = 0.;
    }
}

// A row's lost power (redistribute_slack: RowOverrides::lost_p_mw), on a slot just copied from the
// template (and masked): what the host load does from DistributedSlack's share on, in its order --
// init_loops' DistributedSlack (Ls2gOuterModel::share_lost_p, DistributedSlackLoop::_share_lost_p,
// the units its loss puts at a bound parked around a distributed-slack Newton), its ReactiveLimits
// (each bus' target P, when it comes after DistributedSlack) and resume_loops' ReactiveLimits (the
// buses the n point left frozen, at their limits at the targets the share moved). The template
// shared nothing (a row sets NaN or exactly 0 lost and strands no unit: lost_mw 0), so everything
// else is already the row's. Not for a row resuming from another row's point (it shares beyond that
// point's loss) nor after an n solve that parked units of a distributed-slack Newton (the keep-parked
// move follows the share): load_rows routes those through the host model. One thread per slot.
__global__ void share_lost_kernel(Dev d, const int* slots, const double* lost_p, int n)
{
    // a warp per row: the loops whose iterations are independent (each its own element) run on its lanes, the
    // ordered sums keep their order (lanes filter, the terms are added in index order on every lane), the rest
    // runs on lane 0: the host model's arithmetic, operation for operation
    const int lane = threadIdx.x & 31;
    const int i = static_cast<int>((static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    if (i >= n) return;   // the whole warp
    const double lost_p_mw = lost_p[i];
    if (isnan(lost_p_mw)) return;   // nothing to share: the template's state is the row's
    const int s = slots[i];
    const unsigned char* masked = SLOT(masked, d.n_bus);
    // DistributedSlackLoop::lost_p_to_share_mw: the row's loss, plus the moves of the n point's units
    // it loses (off in the grid or on a bus it masks), at the targets in force
    double counted = lost_p_mw;
    if (d.rp_n) {
        const auto lost = [&](int b) { return b == -2 || (b >= 0 && b < d.n_bus && masked[b]); };
        // the lanes test 32 units at a time; the lost ones' moves are added in index order (every lane the same sum)
        for (int j0 = 0; j0 < d.n_rp_gen; j0 += 32) {
            const int j = j0 + lane;
            const bool l = j < d.n_rp_gen && lost(d.rp_gen_bus[j]);
            const double mv = l ? d.rp_gen_move[j] : 0.;
            for (unsigned m = __ballot_sync(0xffffffffu, l); m != 0u; m &= m - 1u)
                counted += __shfl_sync(0xffffffffu, mv, __ffs(m) - 1);
        }
        for (int j0 = 0; j0 < d.n_rp_sto; j0 += 32) {
            const int j = j0 + lane;
            const bool l = j < d.n_rp_sto && lost(d.rp_sto_bus[j]);
            const double mv = l ? d.rp_sto_move[j] : 0.;
            for (unsigned m = __ballot_sync(0xffffffffu, l); m != 0u; m &= m - 1u)
                counted -= __shfl_sync(0xffffffffu, mv, __ffs(m) - 1);
        }
    }
    if (lane == 0) S1(lost_mw) = counted;
    // the loops the row runs, in order: share_lost_p is DistributedSlack's init
    const int* active = SLOT(active, d.n_loops);
    int pos_ds = -1, pos_rl = -1;
    for (int k = 0; k < S1(n_active); ++k) {
        if (d.loop_kind[active[k]] == DL_DISTRIBUTED_SLACK) pos_ds = k;
        if (d.loop_kind[active[k]] == DL_REACTIVE_LIMITS) pos_rl = k;
    }
    if (pos_ds < 0) return;
    const unsigned char* present = SLOT(ds_present, d.n_units);
    int n_present = 0;
    for (int k = lane; k < d.n_units; k += 32) n_present += present[k] ? 1 : 0;
    n_present = __reduce_add_sync(0xffffffffu, n_present);
    if (n_present == 0 || !(fabs(counted) > d.ds_eps_mw)) return;
    if (lane == 0) {   // DistributedSlack's share and its bookkeeping: sequential
    bool all_sat;
    distribute(d, s, counted, d.ds_eps_mw, all_sat);
    const double* nmw = SLOT(s_new, d.n_units);
    double* cur = SLOT(ds_cur, d.n_units);
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        const int ub = SLOT(ds_ubus, d.n_units)[k];
        if (d.ds_kind[k] == 0) move_gen(d, s, d.ds_el[k], ub, nmw[k]);
        else move_storage(d, s, d.ds_el[k], ub, -nmw[k]);
        // the base (ds_inj) stays the targets the row started from
        cur[k] = nmw[k];
    }
    S1(tr_li) = active[pos_ds];
    rec(d, s, TA_SHARE_LOST_P, true, -1, counted, d.ds_eps_mw);
    // around a distributed-slack Newton, the units its loss put at a bound start parked (every one when
    // all are: distribute clears its flags then)
    if (d.ds_reserved) {
        const unsigned char* sat = SLOT(s_sat, d.n_units);
        for (int k = 0; k < d.n_units; ++k) {
            if (!present[k] || !(all_sat || sat[k])) continue;
            SLOT(ds_park, d.n_units)[k] = 1;
            SLOT(ds_parkh, d.n_units)[k] = counted > 0. ? 1 : 0;
        }
    }
    if (d.keep_adj) ds_mark_solved(d, s);   // the first solve runs with these
    }
    __syncwarp();   // lane 0's targets before the lanes read them
    if (pos_rl < 0) return;
    // ReactiveLimits' init after DistributedSlack's: each bus' target P from the targets in force,
    // but a bus none of whose generators moved (state0's sum; the row sets no storage target)
    const unsigned char* rl_present = SLOT(rl_present, d.rl_n);
    const unsigned char* goff = SLOT(rl_goff, 0);
    if (pos_rl > pos_ds) {
        double* rl_tp = SLOT(rl_tp, d.rl_n);
        for (int k = lane; k < d.rl_n; k += 32) {   // each bus its own target, summed as the host does
            if (!rl_present[k] || (!d.rl_has_sto[k] && rl_at_base(d, s, k))) continue;
            double target_p = 0.;
            for (int j = d.rl_gen_ptr[k]; j < d.rl_gen_ptr[k + 1]; ++j)
                if (!goff[j]) target_p += gen_in_force(d, s, d.rl_gen[j]);
            for (int j = d.rl_sto_ptr[k]; j < d.rl_sto_ptr[k + 1]; ++j) target_p -= sto_in_force(d, s, d.rl_sto[j]);
            rl_tp[k] = target_p;
        }
    }
    __syncwarp();
    // resume_loops: each bus the n point left frozen, at the limit it holds now (or its target Q, the
    // robust mode's), its V the point's -- sequential, lane 0
    if (!d.rp_n || lane != 0) return;
    const Row rv{d.rp_v, d.n_watch, 0, 0};
    for (int k = 0; k < d.rl_n; ++k) {
        const int state = d.rl_rp_state[k];
        if (!rl_present[k] || state == 0) continue;
        double q_min, q_max;
        rl_limits(d, s, &rv, k, q_min, q_max);
        rl_freeze(d, s, rv, k, SLOT(rl_real, d.rl_n)[k] ? SLOT(rl_tq_s, d.rl_n)[k] : (state < 0 ? q_min : q_max), state);
    }
}

void DeviceOuterModel::load_rows(const std::vector<int>& slots_in, const std::vector<int>& rows_in)
{
    if (slots_in.empty()) return;
    // no phantom template yet: the call's first phantom row (row -1) alone first, the others copies
    if (!phantom_ready_ && !check_tmpl_ && slots_in.size() > 1) {
        for (size_t k = 0; k < slots_in.size(); ++k) {
            if (rows_in[k] >= 0) continue;
            int n_ph = 0;
            for (int r : rows_in) n_ph += r < 0;
            if (n_ph < 2) break;
            load_rows({slots_in[k]}, {rows_in[k]});
            std::vector<int> s2, r2;
            for (size_t j = 0; j < slots_in.size(); ++j)
                if (j != k) { s2.push_back(slots_in[j]); r2.push_back(rows_in[j]); }
            load_rows(s2, r2);
            return;
        }
    }
    // no template yet: the first plain row alone through the host (it becomes the template), so
    // that the other plain rows of this call are copied already
    if (!tmpl_row_ && !check_tmpl_ && slots_in.size() > 1) {
        for (size_t k = 0; k < slots_in.size(); ++k) {
            if (!plain(rows_in[k]) || !shares_nothing(rows_in[k])) continue;
            load_rows({slots_in[k]}, {rows_in[k]});
            std::vector<int> s2, r2;
            for (size_t j = 0; j < slots_in.size(); ++j)
                if (j != k) { s2.push_back(slots_in[j]); r2.push_back(rows_in[j]); }
            load_rows(s2, r2);
            return;
        }
    }
    // ... and likewise the first row of each point this call starts several rows from
    if (!check_tmpl_ && slots_in.size() > 1) {
        std::map<int, int> seen;   // point -> rows of this call that would capture its template (point_row)
        for (size_t k = 0; k < slots_in.size(); ++k)
            if (point_row(rows_in[k])) ++seen[resume_of_row_[static_cast<size_t>(rows_in[k])]];
        std::vector<char> first(slots_in.size(), 0);
        bool any = false;
        for (size_t k = 0; k < slots_in.size(); ++k) {
            if (!point_row(rows_in[k])) continue;
            const int pk = resume_of_row_[static_cast<size_t>(rows_in[k])];
            if (point_tmpl_.count(pk) || seen[pk] < 2) continue;
            first[k] = 1;
            seen[pk] = 0;   // only its first row
            any = true;
        }
        if (any) {
            std::vector<int> s1, r1, s2, r2;
            for (size_t k = 0; k < slots_in.size(); ++k) {
                (first[k] ? s1 : s2).push_back(slots_in[k]);
                (first[k] ? r1 : r2).push_back(rows_in[k]);
            }
            load_rows(s1, r1);   // one row per point: each becomes its point's template
            load_rows(s2, r2);
            return;
        }
    }
    // the plain rows once the template exists: copied on the device, no host load
    std::vector<int> slots, rows, t_slots, t_rows, m_slots, m_rows, m_src, p_slots, p_rows, f_slots;
    // a row copied on the device starts with no decision (its template's load recorded none it shares)
    auto clear_trace = [&](const std::vector<int>& rs) {
        if (!t_.keep_trace) return;
        for (int r : rs) if (r >= 0 && static_cast<size_t>(r) < trace_.size()) trace_[static_cast<size_t>(r)].clear();
    };
    for (size_t k = 0; k < slots_in.size(); ++k) {
        const int row = rows_in[k];
        bool from_point = !check_tmpl_ && resumes(row) && plain_overrides(row) && !has_masks(row);
        if (from_point) {
            auto it = point_tmpl_.find(resume_of_row_[static_cast<size_t>(row)]);
            from_point = it != point_tmpl_.end() && it->second.masks.empty();
        }
        if (row < 0 && phantom_ready_ && !check_tmpl_) { f_slots.push_back(slots_in[k]); }
        else if (tmpl_row_ && !check_tmpl_ && plain(row) && !has_masks(row) && shares_nothing(row)) { t_slots.push_back(slots_in[k]); t_rows.push_back(row); }
        else if (from_point)                        { p_slots.push_back(slots_in[k]); p_rows.push_back(row); }
        else if (const int src = edit_source(row); src != -2) {
            m_slots.push_back(slots_in[k]); m_rows.push_back(row); m_src.push_back(src);
        }
        else                                        { slots.push_back(slots_in[k]); rows.push_back(row); }
    }
    clear_trace(t_rows);
    clear_trace(m_rows);
    clear_trace(p_rows);
    // the base-case loads of slots the queue left empty (row -1, always the same): copies of the first
    if (!f_slots.empty()) {
        g2g_nvtx::Range nv("device load: phantom rows");
        const int nf = static_cast<int>(f_slots.size());
        upload_h2d_grow(ld_phantom_, f_slots.data(), f_slots.size(), stream_);   // pageable: staged, f_slots may die
        auto copy = [&](auto& pool, auto& tmpl, size_t w) {
            using T = typename std::remove_reference<decltype(pool)>::type::value_type;
            const size_t total = static_cast<size_t>(nf) * w;
            if (total == 0) return;
            template_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(
                raw(pool), raw(tmpl), raw(ld_phantom_), nf, w);
        };
        copy(p_int_, ph_i_, w_int_);
        copy(p_u8_, ph_u_, w_u8_);
        copy(p_f64_, ph_f_, w_f64_);
        if (nf > 0) materialize_kernel<<<nf, 128, 0, stream_>>>(dev_, raw(ld_phantom_), nf);
        check_cuda(cudaGetLastError(), "phantom load");
    }
    // the rows resuming from a point that has a template: copies of it, grouped by point
    if (!p_slots.empty()) {
        g2g_nvtx::Range nv("device load: rows from their point's template");
        const int np = static_cast<int>(p_slots.size());
        std::vector<int> pack(p_slots);
        pack.insert(pack.end(), p_rows.begin(), p_rows.end());
        for (int r : p_rows) pack.push_back(point_tmpl_.at(resume_of_row_[static_cast<size_t>(r)]).idx);
        upload_h2d_grow(ld_point_, pack.data(), pack.size(), stream_);
        const int* d_ps = raw(ld_point_);
        const int* d_pr = d_ps + np;
        const int* d_pi = d_pr + np;
        auto copy = [&](auto& pool, auto& tmpl, size_t w) {
            using T = typename std::remove_reference<decltype(pool)>::type::value_type;
            const size_t total = static_cast<size_t>(np) * w;
            if (total == 0) return;
            indexed_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(
                raw(pool), raw(tmpl), d_ps, d_pi, np, w);
        };
        copy(p_int_, pt_i_, w_int_);
        copy(p_u8_, pt_u_, w_u8_);
        copy(p_f64_, pt_f_, w_f64_);
        set_rows_kernel<<<(np + 63) / 64, 64, 0, stream_>>>(raw(p_int_), d_ps, d_pr, np, w_int_, field_of_.at("row").off);
        if (np > 0) materialize_kernel<<<np, 128, 0, stream_>>>(dev_, d_ps, np);
        check_cuda(cudaGetLastError(), "point template load");
    }
    // the masked rows and those with a loss to share: the template, then their masks and their share on
    // the device (only the mask lists and the losses go up)
    if (!m_slots.empty()) {
        g2g_nvtx::Range nv("device load: masked rows / rows sharing a loss, from the template");
        const int nm = static_cast<int>(m_slots.size());
        // the plain template's rows first, then those resuming from a point (their point's template)
        std::vector<int> order(static_cast<size_t>(nm));
        for (int k = 0; k < nm; ++k) order[static_cast<size_t>(k)] = k;
        std::stable_partition(order.begin(), order.end(), [&](int k) { return m_src[static_cast<size_t>(k)] < 0; });
        std::vector<int> ms, mr, msrc;
        for (int k : order) { ms.push_back(m_slots[static_cast<size_t>(k)]); mr.push_back(m_rows[static_cast<size_t>(k)]); msrc.push_back(m_src[static_cast<size_t>(k)]); }
        m_slots = ms; m_rows = mr; m_src = msrc;
        std::vector<int> mptr(1, 0), mbus;   // each row's masks, in that order
        for (int r : m_rows) {
            const auto& m = masked_rows_[static_cast<size_t>(r)];
            mbus.insert(mbus.end(), m.begin(), m.end());
            mptr.push_back(static_cast<int>(mbus.size()));
        }
        const int n_plain = static_cast<int>(std::count_if(m_src.begin(), m_src.end(), [](int x) { return x < 0; }));
        std::vector<int> pack(m_slots);
        pack.insert(pack.end(), m_rows.begin(), m_rows.end());
        pack.insert(pack.end(), mptr.begin(), mptr.end());
        pack.insert(pack.end(), mbus.begin(), mbus.end());
        pack.insert(pack.end(), m_src.begin() + n_plain, m_src.end());
        upload_h2d_grow(ld_mask_, pack.data(), pack.size(), stream_);
        const int* d_ms = raw(ld_mask_);
        const int* d_mr = d_ms + nm;
        const int* d_mptr = d_mr + nm;
        const int* d_mbus = d_mptr + nm + 1;
        const int* d_msrc = d_mbus + mbus.size();
        auto copy = [&](auto& pool, auto& tmpl, auto& pts, size_t w) {
            using T = typename std::remove_reference<decltype(pool)>::type::value_type;
            if (w == 0) return;
            if (n_plain > 0)
                template_rows_kernel<T><<<static_cast<unsigned>((static_cast<size_t>(n_plain) * w + 255) / 256), 256, 0, stream_>>>(
                    raw(pool), raw(tmpl), d_ms, n_plain, w);
            const int np = nm - n_plain;
            if (np > 0)
                indexed_rows_kernel<T><<<static_cast<unsigned>((static_cast<size_t>(np) * w + 255) / 256), 256, 0, stream_>>>(
                    raw(pool), raw(pts), d_ms + n_plain, d_msrc, np, w);
        };
        copy(p_int_, tmpl_i_, pt_i_, w_int_);
        copy(p_u8_, tmpl_u_, pt_u_, w_u8_);
        copy(p_f64_, tmpl_f_, pt_f_, w_f64_);
        set_rows_kernel<<<(nm + 63) / 64, 64, 0, stream_>>>(raw(p_int_), d_ms, d_mr, nm, w_int_, field_of_.at("row").off);
        mask_rows_kernel<<<(nm + 63) / 64, 64, 0, stream_>>>(dev_, d_ms, nm, d_mptr, d_mbus);
        check_cuda(cudaGetLastError(), "masked load");
        std::vector<double> loss(static_cast<size_t>(nm));
        bool any_loss = false;
        for (int k = 0; k < nm; ++k) {
            const int r = m_rows[static_cast<size_t>(k)];
            loss[static_cast<size_t>(k)] = has_loss(r) ? row_loss_[static_cast<size_t>(r)] : std::numeric_limits<double>::quiet_NaN();
            any_loss = any_loss || has_loss(r);
        }
        if (any_loss) {
            upload_h2d_grow(ld_loss_, loss.data(), loss.size(), stream_);   // pageable: staged, loss may die
            share_lost_kernel<<<(nm + 3) / 4, 128, 0, stream_>>>(dev_, d_ms, raw(ld_loss_), nm);   // a warp per row
            check_cuda(cudaGetLastError(), "lost power share");
        }
        if (check_tmpl_) {
            check_cuda(cudaStreamSynchronize(stream_), "masked load");   // each one also through the host model, compared field by field
            HostRow hr, dr;
            DeviceSlotImage img;
            for (int k = 0; k < nm; ++k) {
                read_row(m_slots[static_cast<size_t>(k)], dr);
                host_->load(m_slots[static_cast<size_t>(k)], m_rows[static_cast<size_t>(k)]);
                try {   // as the rows loaded through the host model below
                    side_->device_slot_image(m_slots[static_cast<size_t>(k)], img);
                } catch (const std::domain_error& e) {
                    throw DeviceNotImplemented(e.what());
                }
                fill_row(img, hr);
                compare_rows(hr, dr, m_rows[static_cast<size_t>(k)]);
            }
        }
        if (nm > 0) materialize_kernel<<<nm, 128, 0, stream_>>>(dev_, d_ms, nm);
        check_cuda(cudaGetLastError(), "materialize");
    }
    if (!t_slots.empty()) {
        g2g_nvtx::Range nv("device load: template rows");
        const int nt = static_cast<int>(t_slots.size());
        std::vector<int> tpack(t_slots);
        tpack.insert(tpack.end(), t_rows.begin(), t_rows.end());
        upload_h2d_grow(ld_tmpl_, tpack.data(), tpack.size(), stream_);
        const int* d_ts = raw(ld_tmpl_);
        const int* d_tr = d_ts + nt;
        auto copy = [&](auto& pool, auto& tmpl, size_t w) {
            using T = typename std::remove_reference<decltype(pool)>::type::value_type;
            const size_t total = static_cast<size_t>(nt) * w;
            if (total == 0) return;
            template_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(raw(pool), raw(tmpl), d_ts, nt, w);
        };
        copy(p_int_, tmpl_i_, w_int_);
        copy(p_u8_, tmpl_u_, w_u8_);
        copy(p_f64_, tmpl_f_, w_f64_);
        set_rows_kernel<<<(nt + 63) / 64, 64, 0, stream_>>>(raw(p_int_), d_ts, d_tr, nt, w_int_, field_of_.at("row").off);
        if (nt > 0) materialize_kernel<<<nt, 128, 0, stream_>>>(dev_, d_ts, nt);
        check_cuda(cudaGetLastError(), "template load");
    }
    const int n = static_cast<int>(slots.size());
    if (n == 0) return;
    g2g_nvtx::Range nv("device load: rows through the host model");
    // each row uploaded as its differences from a baseline row (the run's first host-loaded one, kept
    // on the device): a row is mostly the grid's, a full upload was hundreds of KB a row
    dif_i_pos_.clear(); dif_i_val_.clear(); dif_u_pos_.clear(); dif_u_val_.clear(); dif_f_pos_.clear(); dif_f_val_.clear();
    up_pslot_.clear(); up_pidx_.clear();
    HostRow hr;
    DeviceSlotImage img;
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };
    for (int k = 0; k < n; ++k) {
        auto t0 = clk::now();
        host_->load(slots[static_cast<size_t>(k)], rows[static_cast<size_t>(k)]);
        if (prof_on_) prof_ms_[0] += ms_since(t0);
        t0 = clk::now();
        try {
            side_->device_slot_image(slots[static_cast<size_t>(k)], img);
        } catch (const std::domain_error& e) {
            throw DeviceNotImplemented(e.what());
        }
        if (prof_on_) prof_ms_[1] += ms_since(t0);
        t0 = clk::now();
        fill_row(img, hr);
        if (prof_on_) prof_ms_[2] += ms_since(t0);
        const int row = rows[static_cast<size_t>(k)];
        if (prof_on_) {   // why through the host: point template capture / masked, its point without a template / other
            const bool res = resumes(row);
            const bool masked = row >= 0 && static_cast<size_t>(row) < mask_only_.size() && mask_only_[static_cast<size_t>(row)];
            const int why = row < 0 ? 5 : res && plain_overrides(row) ? 0 : res && masked ? 1 : res ? 2 : masked ? 3 : 4;
            ++prof_why_[why];
        }
        if (t_.keep_trace && row >= 0 && static_cast<size_t>(row) < trace_.size()) trace_[static_cast<size_t>(row)] = img.trace;
        if (row < 0 && phantom_ready_ && check_tmpl_) compare_rows_with(hr, *ph_host_, row);
        if (row < 0 && !phantom_ready_) {   // the first phantom load: the template of the next ones
            ph_host_ = std::make_unique<HostRow>(hr);
            ph_i_.assign(hr.i.begin(), hr.i.end());
            ph_u_.assign(hr.u.begin(), hr.u.end());
            ph_f_.assign(hr.f.begin(), hr.f.end());
            phantom_ready_ = true;
        }
        if (point_row(row)) {   // the first row of a point: that point's template
            const int pk = resume_of_row_[static_cast<size_t>(row)];
            auto it = point_tmpl_.find(pk);
            if (it == point_tmpl_.end()) {
                add_point_template(pk, unmasked(hr, row));
                point_tmpl_[pk].masks = real_masks(row);
            } else if (check_tmpl_ && !has_masks(row)) {
                compare_rows_with(hr, *it->second.host, row);
            }
        }
        if (plain(row) && shares_nothing(row)) {
            if (!tmpl_row_) {   // the first plain row: the template of the next ones
                tmpl_row_ = std::make_unique<HostRow>(unmasked(hr, row));
                tmpl_i_.assign(tmpl_row_->i.begin(), tmpl_row_->i.end());
                tmpl_u_.assign(tmpl_row_->u.begin(), tmpl_row_->u.end());
                tmpl_f_.assign(tmpl_row_->f.begin(), tmpl_row_->f.end());
            } else if (check_tmpl_ && !has_masks(row)) {
                compare_with_template(hr, row);
            }
        }
        if (!base_row_) {
            base_row_ = std::make_unique<HostRow>(hr);
            base_i_.assign(hr.i.begin(), hr.i.end());
            base_u_.assign(hr.u.begin(), hr.u.end());
            base_f_.assign(hr.f.begin(), hr.f.end());
        }
        // its differences (doubles bitwise: a NaN is the same NaN), as (k * width + offset, value), from
        // its point's template when it resumes from a point that has one (on the device: pt_*), else the baseline
        const HostRow* src = base_row_.get();
        if (resumes(row)) {
            auto it = point_tmpl_.find(resume_of_row_[static_cast<size_t>(row)]);
            if (it != point_tmpl_.end()) {
                src = it->second.host.get();
                up_pslot_.push_back(slots[static_cast<size_t>(k)]);
                up_pidx_.push_back(it->second.idx);
            }
        }
        const HostRow& b = *src;
        for (size_t q = 0; q < w_int_; ++q)
            if (hr.i[q] != b.i[q]) { dif_i_pos_.push_back(static_cast<long long>(k) * static_cast<long long>(w_int_) + static_cast<long long>(q)); dif_i_val_.push_back(hr.i[q]); }
        for (size_t q = 0; q < w_u8_; ++q)
            if (hr.u[q] != b.u[q]) { dif_u_pos_.push_back(static_cast<long long>(k) * static_cast<long long>(w_u8_) + static_cast<long long>(q)); dif_u_val_.push_back(hr.u[q]); }
        for (size_t q = 0; q < w_f64_; ++q)
            if (std::memcmp(&hr.f[q], &b.f[q], sizeof(double)) != 0) {
                dif_f_pos_.push_back(static_cast<long long>(k) * static_cast<long long>(w_f64_) + static_cast<long long>(q));
                dif_f_val_.push_back(hr.f[q]);
            }
    }
    auto t_up = clk::now();
    upload_h2d_grow(ld_slots_, slots.data(), slots.size(), stream_);
    const int* d_sl = raw(ld_slots_);
    const int np_up = static_cast<int>(up_pslot_.size());
    if (np_up > 0) { upload_h2d_grow(ld_up_ps_, up_pslot_.data(), up_pslot_.size(), stream_); upload_h2d_grow(ld_up_pi_, up_pidx_.data(), up_pidx_.size(), stream_); }
    auto apply = [&](auto& pool, auto& base, auto& pt, size_t w, const std::vector<long long>& pos, const auto& val,
                     thrust::device_vector<long long>& d_pos, auto& d_val) {
        using T = typename std::remove_reference<decltype(pool)>::type::value_type;
        const size_t total = static_cast<size_t>(n) * w;
        if (total > 0)
            template_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(
                raw(pool), raw(base), d_sl, n, w);
        // the rows diffed against their point's template start from it instead
        if (np_up > 0 && w > 0)
            indexed_rows_kernel<T><<<static_cast<unsigned>((static_cast<size_t>(np_up) * w + 255) / 256), 256, 0, stream_>>>(
                raw(pool), raw(pt), raw(ld_up_ps_), raw(ld_up_pi_), np_up, w);
        if (pos.empty()) return;
        upload_h2d_grow(d_pos, pos.data(), pos.size(), stream_);
        upload_h2d_grow(d_val, val.data(), val.size(), stream_);
        const size_t m = pos.size();
        scatter_diffs_kernel<T><<<static_cast<unsigned>((m + 255) / 256), 256, 0, stream_>>>(
            raw(pool), raw(d_pos), raw(d_val), d_sl, m, w);
    };
    apply(p_int_, base_i_, pt_i_, w_int_, dif_i_pos_, dif_i_val_, ld_dpos_i_, ld_dval_i_);
    apply(p_u8_, base_u_, pt_u_, w_u8_, dif_u_pos_, dif_u_val_, ld_dpos_u_, ld_dval_u_);
    apply(p_f64_, base_f_, pt_f_, w_f64_, dif_f_pos_, dif_f_val_, ld_dpos_f_, ld_dval_f_);
    if (n > 0) materialize_kernel<<<n, 128, 0, stream_>>>(dev_, raw(ld_slots_), n);
    check_cuda(cudaGetLastError(), "materialize");
    n_host_loads_ += n;
    if (prof_on_) {
        prof_ms_[3] += ms_since(t_up); prof_rows_ += n;
        prof_dif_ += static_cast<long long>(dif_i_pos_.size() + dif_u_pos_.size() + dif_f_pos_.size());
    }
    // stream-ordered: the driver's next work on the same stream sees the loaded slots; the host
    // staging vectors were copied out by the (pageable) uploads before they returned
}

void DeviceOuterModel::load(int slot, int row)
{
    load_rows({slot}, {row});
}

void DeviceOuterModel::move(int from, int to)
{
    host_->move(from, to);
    auto mv = [&](auto& pool, size_t w) {
        using T = typename std::remove_reference<decltype(pool)>::type::value_type;
        check_cuda(cudaMemcpyAsync(raw(pool) + static_cast<size_t>(to) * w, raw(pool) + static_cast<size_t>(from) * w,
                                   w * sizeof(T), cudaMemcpyDeviceToDevice, stream_), "move");
    };
    mv(p_int_, w_int_);
    mv(p_u8_, w_u8_);
    mv(p_f64_, w_f64_);
    set_int(from, "row", -1);
}

__global__ void clear_rows_kernel(int* pool, const int* slots, int n, size_t w, size_t off)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pool[static_cast<size_t>(slots[i]) * w + off] = -1;
}

void DeviceOuterModel::move_slots(int n, const int* d_from, const int* d_to, const int* h_from, const int* h_to)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) host_->move(h_from[i], h_to[i]);
    auto mv = [&](auto& pool, size_t w) {
        using T = typename std::remove_reference<decltype(pool)>::type::value_type;
        const long long total = static_cast<long long>(n) * static_cast<long long>(w);
        move_slot_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(
            raw(pool), d_from, d_to, n, static_cast<int>(w));
    };
    mv(p_int_, w_int_);
    mv(p_u8_, w_u8_);
    mv(p_f64_, w_f64_);
    clear_rows_kernel<<<(n + 63) / 64, 64, 0, stream_>>>(raw(p_int_), d_from, n, w_int_, field_of_.at("row").off);
    check_cuda(cudaGetLastError(), "move_slots");
}

void DeviceOuterModel::set_int(int slot, const char* name, int v)
{
    check_cuda(cudaMemcpyAsync(raw(p_int_) + static_cast<size_t>(slot) * w_int_ + field_of_.at(name).off, &v,
                               sizeof(int), cudaMemcpyHostToDevice, stream_), "upload");
}

int DeviceOuterModel::get_int(int slot, const char* name) const
{
    int v;
    check_cuda(cudaMemcpyAsync(&v, raw(p_int_) + static_cast<size_t>(slot) * w_int_ + field_of_.at(name).off,
                               sizeof(int), cudaMemcpyDeviceToHost, stream_), "download");
    check_cuda(cudaStreamSynchronize(stream_), "download");
    return v;
}

void DeviceOuterModel::clear_vm_reset(int slot)
{
    host_->clear_vm_reset(slot);
    set_int(slot, "vmr_n", 0);
    // RatioTapControl::take_tap: once (and the retaps were applied)
    if (t_.n_bc > 0) {
        std::vector<int> none(static_cast<size_t>(t_.n_bc), kNoTap);
        check_cuda(cudaMemcpyAsync(raw(p_int_) + static_cast<size_t>(slot) * w_int_ + field_of_.at("bc_rrq").off, none.data(),
                                   none.size() * sizeof(int), cudaMemcpyHostToDevice, stream_), "clear");
        check_cuda(cudaMemsetAsync(raw(p_u8_) + static_cast<size_t>(slot) * w_u8_ + field_of_.at("bc_rt").off, 0,
                                   static_cast<size_t>(t_.n_bc), stream_), "clear");
        check_cuda(cudaStreamSynchronize(stream_), "clear");
    }
    // ShuntSectionControl::take_sections: applied once
    if (t_.n_sc > 0)
        check_cuda(cudaMemsetAsync(raw(p_u8_) + static_cast<size_t>(slot) * w_u8_ + field_of_.at("sc_pend").off, 0,
                                   static_cast<size_t>(t_.n_sc), stream_), "clear");
}

void DeviceOuterModel::read_row(int slot, HostRow& hr) const
{
    hr.i.resize(w_int_); hr.u.resize(w_u8_); hr.f.resize(w_f64_);
    check_cuda(cudaMemcpyAsync(hr.i.data(), raw(p_int_) + static_cast<size_t>(slot) * w_int_, w_int_ * sizeof(int),
                               cudaMemcpyDeviceToHost, stream_), "download");
    check_cuda(cudaMemcpyAsync(hr.u.data(), raw(p_u8_) + static_cast<size_t>(slot) * w_u8_, w_u8_,
                               cudaMemcpyDeviceToHost, stream_), "download");
    check_cuda(cudaMemcpyAsync(hr.f.data(), raw(p_f64_) + static_cast<size_t>(slot) * w_f64_, w_f64_ * sizeof(double),
                               cudaMemcpyDeviceToHost, stream_), "download");
    check_cuda(cudaStreamSynchronize(stream_), "download");
}

template <typename T>
std::vector<T> field_vec(const std::vector<T>& row, size_t off, size_t n)
{
    return std::vector<T>(row.begin() + static_cast<ptrdiff_t>(off), row.begin() + static_cast<ptrdiff_t>(off + n));
}

const SlotControls& DeviceOuterModel::controls(int slot) const
{
    HostRow hr;
    read_row(slot, hr);
    auto I = [&](const char* name) { return field_of_.at(name).off; };
    SlotControls& c = ctl_[static_cast<size_t>(slot)];
    const size_t np = static_cast<size_t>(t_.n_pin), nctrl = static_cast<size_t>(t_.n_ctrl);
    c.pin = field_vec(hr.u, I("c_pin"), np);
    c.held_q = field_vec(hr.f, I("c_held_q"), nctrl);
    c.plan_held = field_vec(hr.u, I("c_plan_held"), nctrl);
    c.v_set = field_vec(hr.f, I("c_v_set"), static_cast<size_t>(t_.n_grp));
    c.hvdc_status = field_vec(hr.i, I("c_hvdc_status"), static_cast<size_t>(t_.n_hvdc));
    const size_t n = static_cast<size_t>(hr.i[I("c_sb_n")]);
    c.sbus_bus = field_vec(hr.i, I("c_sb_bus"), n);
    c.sbus_dp = field_vec(hr.f, I("c_sb_dp"), n);
    c.sbus_dq = field_vec(hr.f, I("c_sb_dq"), n);
    const size_t nv = static_cast<size_t>(hr.i[I("vmr_n")]);
    c.vm_reset_bus = field_vec(hr.i, I("vmr_bus"), nv);
    c.vm_reset_vm = field_vec(hr.f, I("vmr_vm"), nv);
    c.bc_out.clear(); c.bc_retap.clear(); c.bc_tap.clear(); c.bc_a_tap.clear(); c.bc_rho_tap.clear();
    const size_t nbc = static_cast<size_t>(t_.n_bc);
    c.bc_out = field_vec(hr.u, I("bc_out"), nbc);
    c.bc_retap.clear();
    for (size_t e = 0; e < nbc; ++e) if (hr.u[I("bc_rt") + e]) c.bc_retap.push_back(static_cast<int>(e));
    c.bc_tap = field_vec(hr.f, I("bc_tap"), 8 * nbc);
    c.bc_a_tap = field_vec(hr.f, I("bc_atap"), nbc);
    c.bc_rho_tap = field_vec(hr.f, I("bc_rtap"), nbc);
    c.bc_ctl.clear(); c.bc_rctl.clear();
    for (size_t e = 0; e < nbc; ++e) {
        c.bc_ctl.push_back(static_cast<signed char>(hr.i[I("bc_ctl") + e]));
        c.bc_rctl.push_back(static_cast<signed char>(hr.i[I("bc_rctl") + e]));
    }
    c.sc_set.clear();
    for (int e = 0; e < t_.n_sc; ++e) if (hr.u[I("sc_pend") + static_cast<size_t>(e)]) c.sc_set.push_back(e);
    c.sc_b_new = field_vec(hr.f, I("sc_bnew"), static_cast<size_t>(t_.n_sc));
    c.sc_ctl.clear();
    for (int e = 0; e < t_.n_sc; ++e) c.sc_ctl.push_back(static_cast<signed char>(hr.i[I("sc_ctl") + static_cast<size_t>(e)]));
    c.slack_w = field_vec(hr.f, I("c_slack_w"), static_cast<size_t>(t_.n_slack));
    return c;
}

void DeviceOuterModel::check_errors(const std::vector<int>& slots) const
{
    for (int s : slots) {
        const int e = get_int(s, "err");
        if (e != 0) throw std::runtime_error(device_outer_error(s, e));
    }
}

Next DeviceOuterModel::on_solved(int slot, bool converged, int nr_iter, int inner_status, const SlotData* data,
                                 int& row_status)
{
    rows_on_host_ = false;
    // the gathered row, in the driver's oc_gather layout
    std::vector<double> g(static_cast<size_t>(stride_), 0.);
    if (converged && data != nullptr) {
        const size_t nw = static_cast<size_t>(t_.n_watch);
        for (size_t k = 0; k < nw; ++k) {
            g[k] = data->v_re[k]; g[nw + k] = data->v_im[k];
            g[2 * nw + k] = data->s_re[k]; g[3 * nw + k] = data->s_im[k];
            g[4 * nw + k] = data->sb_re[k]; g[5 * nw + k] = data->sb_im[k];
        }
        for (int c = 0; c < t_.n_ctrl; ++c) g[6 * nw + static_cast<size_t>(c)] = data->vc_q[c];
        const size_t nbc = static_cast<size_t>(host_->layout().n_bc), o_bc = 6 * nw + static_cast<size_t>(t_.n_ctrl);
        for (size_t e = 0; e < nbc && data->bc_a != nullptr; ++e) { g[o_bc + e] = data->bc_a[e]; g[o_bc + nbc + e] = data->bc_rho[e]; }
        for (int e = 0; e < t_.n_sc && data->sc_b != nullptr; ++e) g[o_bc + 2 * nbc + static_cast<size_t>(e)] = data->sc_b[e];
        g[static_cast<size_t>(stride_ - 2)] = data->slack_absorbed;
        g[static_cast<size_t>(stride_ - 1)] = data->unrealistic ? 1. : 0.;
    }
    st_slots_.assign(1, slot);
    st_conv_.assign(1, converged ? 1 : 0);
    st_nr_.assign(1, nr_iter);
    st_inner_.assign(1, inner_status);
    st_grow_.assign(1, 0);
    upload_h2d_grow(st_gather_, g.data(), g.size(), stream_);
    st_next_.assign(1, 0);
    st_status_.assign(1, 0);
    decide(1, raw(st_slots_), raw(st_conv_), raw(st_nr_), raw(st_inner_), raw(st_grow_), raw(st_gather_), stride_,
           raw(st_next_), raw(st_status_), stream_);
    check_cuda(cudaStreamSynchronize(stream_), "on_solved");
    check_errors({slot});
    const int next = st_next_[0];
    row_status = st_status_[0];
    const int r = get_int(slot, "row");
    if (next == 1 && r >= 0) fin_built_[static_cast<size_t>(r)] = 0;
    const int grow = (converged && data != nullptr) ? 0 : -1;
    export_points(1, &slot, &r, &next, &row_status, &grow, raw(st_gather_), stride_);
    return next == 1 ? Next::Done : Next::Resolve;
}

void DeviceOuterModel::decide(int n, const int* d_slots, const int* d_conv, const int* d_nr, const int* d_inner,
                              const int* d_grow, const double* d_gather, int stride, int* d_next, int* d_status,
                              void* stream)
{
    if (n == 0) return;
    rows_on_host_ = false;   // finished rows write their results
    on_solved_kernel<<<n, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        dev_, d_slots, n, d_conv, d_nr, d_inner, d_grow, d_gather, stride, d_next, d_status);
    check_cuda(cudaGetLastError(), "decide");
}

void DeviceOuterModel::export_points(int n, const int* h_slots, const int* h_rows, const int* h_next,
                                     const int* h_status, const int* h_grow, const double* d_gather, int stride)
{
    if (t_.keep_trace) flush_trace(n, h_slots, h_rows);
    if (!(keep_points_ && !referenced_.empty()) && !keep_adj_) return;
    // keep_point's condition: the loops stable and the row converged (RowStatus Converged); the
    // adjoint's: every row that ends (adjoint_info)
    std::vector<int> sel;
    std::vector<char> want_point, want_adj;
    for (int i = 0; i < n; ++i) {
        const int row = h_rows[i];
        if (h_next[i] != 1 || row < 0) continue;
        const bool pt = keep_points_ && h_status[i] == 0 && static_cast<size_t>(row) < referenced_.size() &&
                        referenced_[static_cast<size_t>(row)] && h_grow[i] >= 0;
        const bool adj = keep_adj_ && static_cast<size_t>(row) < adj_.size();
        if (!pt && !adj) continue;
        sel.push_back(i);
        want_point.push_back(pt ? 1 : 0);
        want_adj.push_back(adj ? 1 : 0);
    }
    if (sel.empty()) return;
    // every selected slot's state and gathered row downloaded together, one sync
    std::vector<HostRow> rows_h(sel.size());
    std::vector<std::vector<double>> gs(sel.size(), std::vector<double>(static_cast<size_t>(stride)));
    for (size_t q = 0; q < sel.size(); ++q) {
        const int i = sel[q];
        const size_t slot = static_cast<size_t>(h_slots[i]);
        HostRow& r = rows_h[q];
        r.i.resize(w_int_); r.u.resize(w_u8_); r.f.resize(w_f64_);
        check_cuda(cudaMemcpyAsync(r.i.data(), raw(p_int_) + slot * w_int_, w_int_ * sizeof(int), cudaMemcpyDeviceToHost, stream_), "export");
        check_cuda(cudaMemcpyAsync(r.u.data(), raw(p_u8_) + slot * w_u8_, w_u8_, cudaMemcpyDeviceToHost, stream_), "export");
        check_cuda(cudaMemcpyAsync(r.f.data(), raw(p_f64_) + slot * w_f64_, w_f64_ * sizeof(double), cudaMemcpyDeviceToHost, stream_), "export");
        if (h_grow[i] >= 0)
            check_cuda(cudaMemcpyAsync(gs[q].data(), d_gather + static_cast<size_t>(h_grow[i]) * stride,
                                       gs[q].size() * sizeof(double), cudaMemcpyDeviceToHost, stream_), "export");
    }
    check_cuda(cudaStreamSynchronize(stream_), "export");
    auto I = [&](const char* name) { return field_of_.at(name).off; };
    for (size_t q = 0; q < sel.size(); ++q) {
        const int row = h_rows[sel[q]];
        const HostRow& hr = rows_h[q];
        const std::vector<double>& g = gs[q];
        if (want_adj[q]) adjoint_of(hr, adj_[static_cast<size_t>(row)]);
        if (!want_point[q]) continue;
        DevicePoint p;
        p.from_row = resumes(row) ? resume_of_row_[static_cast<size_t>(row)] : -1;
        p.gen_p.assign(hr.f.begin() + static_cast<ptrdiff_t>(I("gen_tp")),
                       hr.f.begin() + static_cast<ptrdiff_t>(I("gen_tp") + static_cast<size_t>(t_.n_gen)));
        p.sto_p.assign(hr.f.begin() + static_cast<ptrdiff_t>(I("sto_tp")),
                       hr.f.begin() + static_cast<ptrdiff_t>(I("sto_tp") + static_cast<size_t>(t_.n_sto)));
        for (int e = 0; e < t_.n_hvdc; ++e) {
            const int reg = hr.i[I("hvdc_regime") + static_cast<size_t>(e)];
            if (reg != 2) { p.hvdc_id.push_back(t_.hvdc_id[static_cast<size_t>(e)]); p.hvdc_regime.push_back(reg); }
        }
        for (int c = 0; c < t_.n_ctrl; ++c) {
            const double on = hr.f[I("svc_on") + static_cast<size_t>(c)];
            if (t_.vc_kind[static_cast<size_t>(c)] == VC_SVC && std::isfinite(on)) {
                p.svc_id.push_back(t_.vc_elem[static_cast<size_t>(c)]);
                p.svc_target.push_back(on);
            }
        }
        // the frozen buses, the first entry of a bus (std::map::emplace), sorted by bus
        std::map<int, std::pair<int, int>> frozen;
        for (int k = 0; k < t_.rl_n; ++k) {
            if (!hr.u[I("rl_present") + static_cast<size_t>(k)]) continue;
            const int state = hr.i[I("rl_state") + static_cast<size_t>(k)];
            if (state != 0)
                frozen.emplace(t_.rl_bus[static_cast<size_t>(k)], std::make_pair(state, hr.i[I("rl_real") + static_cast<size_t>(k)]));
        }
        const size_t nw = static_cast<size_t>(t_.n_watch);
        for (const auto& kv : frozen) {
            const int b = kv.first;
            const int w = (b >= 0 && b < t_.n_bus) ? t_.widx[static_cast<size_t>(b)] : -1;
            p.frozen_bus.push_back(b);
            p.frozen_state.push_back(kv.second.first);
            p.frozen_realistic.push_back(kv.second.second);
            // keep_point's v_at holds the watched ones only; ResumePoint::v_at then lacks the others
            p.v_re.push_back(w >= 0 ? g[static_cast<size_t>(w)] : 0.);
            p.v_im.push_back(w >= 0 ? g[nw + static_cast<size_t>(w)] : 0.);
            if (w < 0) {   // keep the host's semantics: a bus without V is not in v_at
                p.v_re.back() = std::numeric_limits<double>::quiet_NaN();
            }
        }
        // the units a distributed-slack Newton left parked (keep_point's, in the units' order)
        for (int k = 0; k < n_units_; ++k) {
            if (!hr.u[I("ds_present") + static_cast<size_t>(k)] || !hr.u[I("ds_park") + static_cast<size_t>(k)]) continue;
            p.parked_kind.push_back(t_.ds_kind[static_cast<size_t>(k)]);
            p.parked_el.push_back(t_.ds_el_id[static_cast<size_t>(k)]);
            p.parked_high.push_back(hr.u[I("ds_parkh") + static_cast<size_t>(k)]);
        }
        // the section counts / taps, when the row ran ShuntVoltageControl / PhaseControl (keep_point's)
        for (int li = 0; li < hr.i[I("n_active")]; ++li) {
            const int kind = t_.loop_kind[static_cast<size_t>(hr.i[I("active") + static_cast<size_t>(li)])];
            p.has_sections = p.has_sections || kind == DL_SHUNT_VOLTAGE_CONTROL;
            p.has_phase = p.has_phase || kind == DL_PHASE_CONTROL;
            p.has_ratio = p.has_ratio || kind == DL_TRANSFORMER_VOLTAGE_CONTROL;
        }
        if (p.has_phase || p.has_ratio) {
            p.bc_pos = field_vec(hr.i, I("bc_pos"), static_cast<size_t>(t_.n_bc));
            p.bc_rpos = field_vec(hr.i, I("bc_rpos"), static_cast<size_t>(t_.n_bc));
            p.bc_out = field_vec(hr.u, I("bc_out"), static_cast<size_t>(t_.n_bc));
        }
        if (p.has_sections)
            p.sections.assign(hr.i.begin() + static_cast<ptrdiff_t>(I("sc_cnt")),
                              hr.i.begin() + static_cast<ptrdiff_t>(I("sc_cnt") + static_cast<size_t>(t_.n_sc_sh)));
        p.lost_mw = hr.f[I("lost_mw")];
        side_->device_import_point(row, p);
    }
}

// the decisions the listed slots recorded since the last flush, appended to their rows' traces
// (Ls2gOuterModel::record's format), the slots' buffers emptied
void DeviceOuterModel::flush_trace(int n, const int* h_slots, const int* h_rows)
{
    if (n <= 0) return;
    const size_t o_n = field_of_.at("tr_n").off;
    std::vector<int> cnt(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        check_cuda(cudaMemcpyAsync(&cnt[static_cast<size_t>(i)], raw(p_int_) + static_cast<size_t>(h_slots[i]) * w_int_ + o_n,
                                   sizeof(int), cudaMemcpyDeviceToHost, stream_), "trace");
    check_cuda(cudaStreamSynchronize(stream_), "trace");
    const size_t tc = static_cast<size_t>(tcap_);
    struct Entries { std::vector<int> loop, act, el, outer; std::vector<unsigned char> taken; std::vector<double> val, lim; };
    std::vector<Entries> e(static_cast<size_t>(n));
    auto get_i = [&](std::vector<int>& out, const char* f, int slot, int c) {
        out.resize(static_cast<size_t>(c));
        check_cuda(cudaMemcpyAsync(out.data(), raw(p_int_) + static_cast<size_t>(slot) * w_int_ + field_of_.at(f).off,
                                   static_cast<size_t>(c) * sizeof(int), cudaMemcpyDeviceToHost, stream_), "trace");
    };
    bool any = false;
    for (int i = 0; i < n; ++i) {
        const int c = cnt[static_cast<size_t>(i)], slot = h_slots[i];
        if (c <= 0) continue;
        any = true;
        Entries& x = e[static_cast<size_t>(i)];
        get_i(x.loop, "tr_loop", slot, c); get_i(x.act, "tr_act", slot, c);
        get_i(x.el, "tr_el", slot, c); get_i(x.outer, "tr_outer", slot, c);
        x.taken.resize(static_cast<size_t>(c)); x.val.resize(static_cast<size_t>(c)); x.lim.resize(static_cast<size_t>(c));
        check_cuda(cudaMemcpyAsync(x.taken.data(), raw(p_u8_) + static_cast<size_t>(slot) * w_u8_ + field_of_.at("tr_taken").off,
                                   static_cast<size_t>(c), cudaMemcpyDeviceToHost, stream_), "trace");
        check_cuda(cudaMemcpyAsync(x.val.data(), raw(p_f64_) + static_cast<size_t>(slot) * w_f64_ + field_of_.at("tr_val").off,
                                   static_cast<size_t>(c) * sizeof(double), cudaMemcpyDeviceToHost, stream_), "trace");
        check_cuda(cudaMemcpyAsync(x.lim.data(), raw(p_f64_) + static_cast<size_t>(slot) * w_f64_ + field_of_.at("tr_lim").off,
                                   static_cast<size_t>(c) * sizeof(double), cudaMemcpyDeviceToHost, stream_), "trace");
        check_cuda(cudaMemsetAsync(raw(p_int_) + static_cast<size_t>(slot) * w_int_ + o_n, 0, sizeof(int), stream_), "trace");
    }
    (void)tc;
    if (!any) return;
    check_cuda(cudaStreamSynchronize(stream_), "trace");
    for (int i = 0; i < n; ++i) {
        const int row = h_rows[i];
        const Entries& x = e[static_cast<size_t>(i)];
        if (x.act.empty() || row < 0 || static_cast<size_t>(row) >= trace_.size()) continue;
        auto& out = trace_[static_cast<size_t>(row)];
        for (size_t k = 0; k < x.act.size(); ++k) {
            std::ostringstream os;
            // a device NaN may carry a sign bit: printed as the host's quiet NaN ("nan")
            auto host_nan = [](double v) { return std::isnan(v) ? std::numeric_limits<double>::quiet_NaN() : v; };
            os << t_.loop_name[static_cast<size_t>(x.loop[k])] << " #" << x.outer[k] << " " << kActionNames[x.act[k]]
               << (x.taken[k] ? "" : " (not taken)") << ", element " << x.el[k] << ", value=" << host_nan(x.val[k])
               << ", limit=" << host_nan(x.lim[k]);
            out.push_back(os.str());
        }
    }
}

void DeviceOuterModel::adjoint_of(const HostRow& hr, RowResult& out) const
{
    auto I = [&](const char* name) { return field_of_.at(name).off; };
    DeviceAdjointState a;
    const size_t np = static_cast<size_t>(t_.n_pin), nctrl = static_cast<size_t>(t_.n_ctrl), nu = static_cast<size_t>(n_units_);
    const size_t rn = static_cast<size_t>(t_.rl_n);
    a.pin = field_vec(hr.u, I("c_pin"), np);
    a.plan_held = field_vec(hr.u, I("c_plan_held"), nctrl);
    a.held_q = field_vec(hr.f, I("c_held_q"), nctrl);
    a.ds_in_newton = hr.u[I("ds_inn")];
    a.ds_present = field_vec(hr.u, I("ds_present"), nu);
    a.ds_solved_parked = field_vec(hr.u, I("ds_spark"), nu);
    a.ds_solved_mw = field_vec(hr.f, I("ds_smw"), nu);
    a.ds_initial_mw = field_vec(hr.f, I("ds_ini"), nu);
    a.rl_present.resize(rn);
    a.rl_state.resize(rn);
    for (size_t k = 0; k < rn; ++k) {
        a.rl_present[k] = hr.u[I("rl_present") + k];
        a.rl_state[k] = hr.i[I("rl_state") + k];
    }
    // the slopes recorded, bus by bus in the plan's order, each bus' generators in its order
    const size_t o_set = I("slp_set"), o_slp = I("slp"), o_goff = I("rl_goff"), o_coff = I("ctrl_off");
    for (int k = 0; k < t_.rl_n; ++k) {
        const size_t kk = static_cast<size_t>(k);
        if (!a.rl_present[kk] || a.rl_state[kk] == 0) continue;
        if (!t_.rl_local[kk]) {
            for (int j = t_.rl_ctrl_ptr[kk]; j < t_.rl_ctrl_ptr[kk + 1]; ++j) {
                const int c = t_.rl_ctrl[static_cast<size_t>(j)];
                if (hr.u[o_coff + static_cast<size_t>(c)] || t_.vc_kind[static_cast<size_t>(c)] != VC_GEN) continue;
                const size_t g = static_cast<size_t>(t_.vc_elem[static_cast<size_t>(c)]);
                if (!hr.u[o_set + g]) continue;
                a.slope_k0.push_back(k); a.slope_gen.push_back(static_cast<int>(g)); a.slope_ctrl.push_back(c);
                a.slope.push_back(hr.f[o_slp + g]);
            }
        } else {
            for (int j = t_.rl_gen_ptr[kk]; j < t_.rl_gen_ptr[kk + 1]; ++j) {
                const size_t g = static_cast<size_t>(t_.rl_gen[static_cast<size_t>(j)]);
                if (hr.u[o_goff + static_cast<size_t>(j)] || !hr.u[o_set + g]) continue;
                a.slope_k0.push_back(k); a.slope_gen.push_back(static_cast<int>(g)); a.slope_ctrl.push_back(-1);
                a.slope.push_back(hr.f[o_slp + g]);
            }
        }
    }
    out = RowResult();
    side_->device_adjoint_info(a, out);
}

void DeviceOuterModel::sync_rows_host() const
{
    if (rows_on_host_) return;
    auto down = [](const auto& d, auto& h) {
        h.resize(d.size());
        if (!h.empty())
            check_cuda(cudaMemcpy(h.data(), thrust::raw_pointer_cast(d.data()), h.size() * sizeof(h[0]),
                                  cudaMemcpyDeviceToHost), "download");
    };
    down(r_int_, h_r_int_);
    down(r_f64_, h_r_f64_);
    down(r_sc_, h_r_sc_);
    down(r_pos_, h_r_pos_);
    down(r_rpos_, h_r_rpos_);
    rows_on_host_ = true;
}

// the host mirror of a pointer into one of the row pools (Dev's r_* fields)
const int* DeviceOuterModel::host_of(const int* dev) const
{
    sync_rows_host();
    const std::pair<const thrust::device_vector<int>*, const std::vector<int>*> pools[] = {
        {&r_int_, &h_r_int_}, {&r_sc_, &h_r_sc_}, {&r_pos_, &h_r_pos_}, {&r_rpos_, &h_r_rpos_}};
    for (const auto& [d, h] : pools) {
        const int* b = thrust::raw_pointer_cast(d->data());
        if (std::less_equal<const int*>()(b, dev) && std::less<const int*>()(dev, b + d->size()))
            return h->data() + (dev - b);
    }
    throw std::logic_error("device outer model: not a row pool");
}

const double* DeviceOuterModel::host_of(const double* dev) const
{
    sync_rows_host();
    const double* b = thrust::raw_pointer_cast(r_f64_.data());
    if (!(std::less_equal<const double*>()(b, dev) && std::less<const double*>()(dev, b + r_f64_.size())))
        throw std::logic_error("device outer model: not a row pool");
    return h_r_f64_.data() + (dev - b);
}

namespace {
// n elements of row `row` of a per-row (host) array
template <typename T>
std::vector<T> row_of(const T* p, int row, size_t width, size_t n)
{
    return std::vector<T>(p + static_cast<size_t>(row) * width, p + static_cast<size_t>(row) * width + n);
}
}  // namespace

void DeviceOuterModel::row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const
{
    trafo.clear();
    block.clear();
    if (row < 0 || row >= n_rows_ || t_.n_bc == 0) return;
    if (host_of(dev_.r_status)[row] < 0) return;   // never ran
    const std::vector<int> pos = row_of(host_of(dev_.r_pos), row, static_cast<size_t>(t_.n_bc), static_cast<size_t>(t_.n_bc));
    const std::vector<int> rpos = row_of(host_of(dev_.r_rpos), row, static_cast<size_t>(t_.n_bc), static_cast<size_t>(t_.n_bc));
    side_->device_row_blocks(pos.data(), rpos.data(), trafo, block);
}

const RowResult& DeviceOuterModel::row_result(int row) const
{
    const size_t r = static_cast<size_t>(row);
    RowResult& res = results_.at(r);
    if (fin_built_[r]) return res;
    fin_built_[r] = 1;
    res = RowResult();
    const Dev& d = dev_;
    res.status = host_of(d.r_status)[row];
    if (res.status < 0) return res;   // never ran (not simulated): the default result
    res.unrealistic = host_of(d.r_unreal)[row] != 0;
    const int failed = host_of(d.r_failed)[row];
    if (failed >= 0 && static_cast<size_t>(failed) < t_.loop_name.size())
        res.failed_loop = t_.loop_name[static_cast<size_t>(failed)];   // the loop's own name, as the host's
    res.nb_outer_iterations = host_of(d.r_outer)[row];
    res.nb_nr_iterations = host_of(d.r_nr)[row];
    if (r < trace_.size()) res.decisions = trace_[r];
    if (r < adj_.size()) {
        const RowResult& a = adj_[r];
        res.pq_buses = a.pq_buses;
        res.vset_lost_groups = a.vset_lost_groups;
        res.gen_p_fixed = a.gen_p_fixed;
        res.gen_p_slope_gen = a.gen_p_slope_gen;
        res.gen_p_slope_row = a.gen_p_slope_row;
        res.gen_p_slope_coeff = a.gen_p_slope_coeff;
    }
    // the generators' and storage units' targets: the rows' common start, the units' own where they moved
    std::vector<double> gen_tp = gen_tp_tmpl_, sto_tp = sto_tp_tmpl_;
    if (resumes(row)) {
        auto it = point_base_.find(resume_of_row_[r]);
        if (it != point_base_.end()) { gen_tp = it->second.first; sto_tp = it->second.second; }
    }
    {
        auto it = row_base_.find(row);
        if (it != row_base_.end()) { gen_tp = it->second.first; sto_tp = it->second.second; }
    }
    const std::vector<double> units = row_of(host_of(d.r_unit_mw), row, static_cast<size_t>(std::max(n_units_, 1)), static_cast<size_t>(n_units_));
    for (int k = 0; k < n_units_; ++k) {
        const size_t el = static_cast<size_t>(t_.ds_el_id[static_cast<size_t>(k)]);
        if (t_.ds_kind[static_cast<size_t>(k)] == 0) { if (el < gen_tp.size()) gen_tp[el] = units[static_cast<size_t>(k)]; }
        else if (el < sto_tp.size()) sto_tp[el] = units[static_cast<size_t>(k)];
    }
    bool any = false;
    for (double v : gen_tp) any = any || !std::isnan(v);
    if (any) res.gens_target_p_mw = gen_tp;
    for (size_t s = 0; s < sto_tp.size(); ++s)
        if (!std::isnan(sto_tp[s])) { res.storage_ids.push_back(static_cast<int>(s)); res.storage_target_p_mw.push_back(sto_tp[s]); }
    // the section counts the loops left, where they differ from the grid's
    if (t_.n_sc_sh > 0) {
        const std::vector<int> sc = row_of(host_of(d.r_sc), row, static_cast<size_t>(t_.n_sc_sh), static_cast<size_t>(t_.n_sc_sh));
        for (int j = 0; j < t_.n_sc_sh; ++j) {
            const size_t jj = static_cast<size_t>(j);
            if (t_.sh_has[jj] && t_.sh_count0[jj] == sc[jj]) continue;
            res.shunt_ids.push_back(t_.sh_id[jj]);
            res.shunt_sections.push_back(sc[jj]);
        }
    }
    // the taps the loops handled, where they differ from the grid's
    if (t_.n_bc > 0) {
        const std::vector<int> pos = row_of(host_of(d.r_pos), row, static_cast<size_t>(t_.n_bc), static_cast<size_t>(t_.n_bc));
        const std::vector<int> rpos = row_of(host_of(d.r_rpos), row, static_cast<size_t>(t_.n_bc), static_cast<size_t>(t_.n_bc));
        for (int e = 0; e < t_.n_bc; ++e) {
            const size_t ee = static_cast<size_t>(e);
            const int tr = t_.bc_trafo[ee];
            if (t_.bc_phase[ee] && !(t_.bc_has_ptc[ee] && t_.bc_pos0[ee] == pos[ee])) { res.phase_tap_ids.push_back(tr); res.phase_tap.push_back(pos[ee]); }
            if (t_.bc_ratio[ee] && !(t_.bc_has_rtc[ee] && t_.bc_rpos0[ee] == rpos[ee])) { res.ratio_tap_ids.push_back(tr); res.ratio_tap.push_back(rpos[ee]); }
        }
    }
    const std::vector<int> hv = row_of(host_of(d.r_hvdc), row, static_cast<size_t>(std::max(t_.n_hvdc, 1)), static_cast<size_t>(t_.n_hvdc));
    for (int e = 0; e < t_.n_hvdc; ++e)
        if (hv[static_cast<size_t>(e)] != 2) { res.hvdc_ids.push_back(t_.hvdc_id[static_cast<size_t>(e)]); res.hvdc_regime.push_back(hv[static_cast<size_t>(e)]); }
    return res;
}

}  // namespace

std::string device_outer_error(int slot, int code)
{
    const char* what = code == ERR_NOT_WATCHED ? "a bus the gather does not watch"
                     : code == ERR_CORR_BUS ? "a Sbus correction off the planned buses"
                     : code == ERR_VM_RESET_CAP ? "too many |V| resets in one solve"
                     : code == ERR_NO_PIN ? "a local controller bus without a reserved Vm column"
                     : code == ERR_TRACE_CAP ? "more decisions in one solve than the trace holds" : "an unknown error";
    return "device outer model, slot " + std::to_string(slot) + ": " + what;
}

void check_device_features(const std::vector<DeviceFeature>& features)
{
    std::string missing;
    for (const DeviceFeature& f : features) {
        if (f.count <= 0 || f.supported) continue;
        if (!missing.empty()) missing += "; ";
        missing += f.name + " (" + std::to_string(f.count) + ")";
    }
    if (!missing.empty()) throw DeviceNotImplemented(missing);
}

std::shared_ptr<OuterHook> make_device_outer_model_from(std::shared_ptr<OuterHook> host, DeviceHostSide* side)
{
    return std::make_shared<DeviceOuterModel>(std::move(host), side);
}

}  // namespace outer
