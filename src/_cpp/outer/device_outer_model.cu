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
#include "../contingency/slot_schedule.cuh"   // move_slot_rows_kernel

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <map>
#include <memory>
#include <type_traits>
#include <string>
#include <vector>

namespace outer {
namespace {

constexpr int STABLE = 0, UNSTABLE = 1, FAILED = 2, CHECK_AGAIN = 3;   // OuterLoopStatus
constexpr int VC_GEN = 0, VC_SVC = 1, VC_HVDC_1 = 2, VC_HVDC_2 = 3;     // VoltageControlSolverData::Kind
// device error codes (a case the host scan should have refused, or a capacity)
constexpr int ERR_NOT_WATCHED = 1, ERR_CORR_BUS = 2, ERR_VM_RESET_CAP = 3, ERR_NO_PIN = 4;

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
    const int *corr_bus, *corr_of_bus;
    // per slot: the driver state
    int *row, *it, *last_unstable, *last_status, *first, *changed, *nb_outer, *nb_passes, *total_nr, *old_total;
    int *nr_steps, *v_version, *solver_ok, *unrealistic, *check_unreal, *pass_check_unreal, *last_fixing, *n_active;
    int *last_loop, *lim_base, *vmr_n, *err, *p_recheck;
    int *active, *loop_iter, *done_at;
    unsigned char *masked, *lim_dirty, *pq;
    double *held_q, *svc_on;
    int *hvdc_regime;
    double *gen_tp, *sto_tp;
    unsigned char *dset, *fset;
    double *dre, *dim, *fq;
    int *vmr_bus;
    double *vmr_vm;
    unsigned char *rl_present;
    double *rl_tp, *rl_fq;
    int *rl_state, *rl_nb, *rl_real;
    unsigned char *ds_present;
    double *ds_inj, *ds_cur, *ds_ini;
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
    const double* g;   // [6 n_watch + n_ctrl + ... + 2], the driver's oc_gather layout
    int nw, nctrl;
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
__device__ double gen_in_force(const Dev& d, int s, int g)
{
    const double v = SLOT(gen_tp, d.n_gen)[g];
    return isnan(v) ? d.gen_tp0[g] : v;
}
__device__ double sto_in_force(const Dev& d, int s, int i)
{
    const double v = SLOT(sto_tp, d.n_sto)[i];
    return isnan(v) ? d.sto_tp0[i] : v;
}
// OuterGridView::gen_target_p_override (a row sets no target of its own in the device scope)
__device__ double gen_override(const Dev& d, int s, int g)
{
    return g < 0 ? CUDART_NAN : SLOT(gen_tp, d.n_gen)[g];
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

// ---- DistributedSlackLoop::_check (a single-slack Newton) -----------------------------------------
__device__ int check_ds(const Dev& d, int s, const Row& rv)
{
    double mre, mim;
    mis_bus(d, s, rv, d.slack_bus, mre, mim);
    const double mismatch = mre * d.sn;
    const bool triggered = fabs(mismatch) > d.ds_mismatch_mw && fabs(mismatch) > d.ds_eps_mw;
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
    if (d.ds_fail && fabs(residue) > d.ds_eps_mw) return FAILED;
    double moved = 0.;
    const double* nmw = SLOT(s_new, d.n_units);
    for (int k = 0; k < d.n_units; ++k) {
        if (!present[k]) continue;
        moved += fabs(d.ds_kind[k] == 0 ? move_gen(d, s, d.ds_el[k], d.ds_bus[k], nmw[k])
                                        : move_storage(d, s, d.ds_el[k], d.ds_bus[k], -nmw[k]));
        cur[k] = nmw[k];
    }
    return moved > d.ds_moved_fraction * d.ds_eps_mw ? UNSTABLE : STABLE;
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
        for (int k = 0; k < d.n_hl; ++k) {
            if (!present[k] || d.hl_id[k] != d.hl_id[v] || dir[k] == regime) continue;
            const int e = d.hl_entry[k];
            if (e < 0) continue;                  // no regime control for it
            dir[k] = regime;
            SLOT(hvdc_regime, d.n_hvdc)[e] = regime;
            changed = true;
        }
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
        double target;
        if (vm < d.mon_low[m]) target = d.svc_low_t[svc];
        else if (vm > d.mon_high[m]) target = d.svc_high_t[svc];
        else continue;
        if (!(isfinite(target) && c >= 0)) continue;
        svc_on[c] = target;
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

// ---- ReactiveLimitsLoop ----------------------------------------------------------------------------
__device__ bool rl_at_base(const Dev& d, int s, int k)
{
    const int b = d.rl_bus[k];
    return d.rl_has_base[k] && S1(lim_base) && b >= 0 && b < d.n_bus && !SLOT(lim_dirty, d.n_bus)[b];
}
__device__ double rl_bus_q(const Dev& d, int s, const Row& rv, int k)
{
    double mre, mim;
    mis_bus(d, s, rv, d.rl_bus[k], mre, mim);
    double q = mim;
    if (!d.rl_local[k])
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) q += rv.vc_q(d.rl_ctrl[j]);
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
__device__ void rl_freeze(const Dev& d, int s, const Row& rv, int k, double q_mvar, int state)
{
    double* held_q = SLOT(held_q, d.n_ctrl);
    if (!d.rl_local[k]) {
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) {
            const int c = d.rl_ctrl[j];
            double q = rl_controller_limit(d, s, rv, c, state > 0);
            if (d.vc_kind[c] == VC_GEN && SLOT(rl_real, d.rl_n)[k]) q = d.gen_target_q[d.vc_elem[c]];   // the robust mode's
            held_q[c] = q / d.sn;
        }
        SLOT(rl_state, d.rl_n)[k] = state;
        SLOT(rl_fq, d.rl_n)[k] = q_mvar;
        return;
    }
    if (d.rl_pin[k] < 0) { S1(err) = ERR_NO_PIN; return; }
    freeze_set(d, s, d.rl_bus[k], q_mvar / d.sn);
    SLOT(pq, d.n_pin)[d.rl_pin[k]] = 1;
    SLOT(rl_state, d.rl_n)[k] = state;
    SLOT(rl_fq, d.rl_n)[k] = q_mvar;
}
__device__ void rl_release(const Dev& d, int s, int k)
{
    if (!d.rl_local[k]) {
        for (int j = d.rl_ctrl_ptr[k]; j < d.rl_ctrl_ptr[k + 1]; ++j) SLOT(held_q, d.n_ctrl)[d.rl_ctrl[j]] = CUDART_NAN;
    } else {
        freeze_erase(d, s, d.rl_bus[k]);
        SLOT(pq, d.n_pin)[d.rl_pin[k]] = 0;
        reset_vm(d, s, d.rl_bus[k], d.rl_tvm[k]);
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
    const double eps = d.rl_max_mismatch * d.rl_base_mva;
    const double v_low = d.rl_vmin * d.rl_margin;
    const double v_high = d.rl_vmax / d.rl_margin;
    double target_vm = d.rl_tvm[k];
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
        if (q < q_min - eps) { cls = 1; qlim = q_min; }
        else if (q > q_max + eps) { cls = 2; qlim = q_max; }
        else if (d.rl_robust && d.rl_reg[k] != d.rl_bus[k]) {
            const double v = vm_of(d, s, rv, d.rl_bus[k]);
            if (v < v_low)       { cls = 3; qlim = d.rl_tq[k]; }
            else if (v > v_high) { cls = 4; qlim = d.rl_tq[k]; }
            else cls = 5;
        } else {
            cls = 5;
        }
        return;
    }
    const bool held_by_group = d.rl_group[k] >= 0 && SLOT(s_hold, d.n_grp)[d.rl_group[k]];
    const double vm_read = vm_of(d, s, rv, d.rl_reg[k]);
    const double vm = held_by_group ? target_vm : vm_read;
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
    int skip = -1;
    if (n_pq > 0 && remaining_pv == 0) {
        skip = 0;
        for (int j = 1; j < n_pq; ++j) if (rl_stronger(d, s, pq_k[j], pq_k[skip])) skip = j;
    }
    for (int j = 0; j < n_pq; ++j) {
        if (j == skip) continue;
        const int k = pq_k[j];
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
        if (SLOT(rl_nb, d.rl_n)[k] >= d.rl_max_switch) continue;
        rl_release(d, s, k);
        changed = true;
    }
    for (int j = 0; j < n_mv; ++j) {
        const int k = mv_k[j];
        double q_min, q_max;
        rl_limits(d, s, &rv, k, q_min, q_max);
        const int state = SLOT(rl_state, d.rl_n)[k];
        rl_freeze(d, s, rv, k, state < 0 ? q_min : q_max, state);
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

__device__ int check(const Dev& d, int s, const Row& rv, int loop)
{
    switch (d.loop_kind[loop]) {
        case 0: return check_ds(d, s, rv);
        case 1: return check_hvdc(d, s, rv);
        case 2: return check_vm(d, s, rv);
        // ReactiveLimits: drive evaluates its buses in parallel first (eval_rl_bus, check_rl_apply)
        default: return STABLE;
    }
}

// ---- materialize: the slot's controls --------------------------------------------------------------
__device__ void materialize(const Dev& d, int s)
{
    const unsigned char* pq = SLOT(pq, d.n_pin);
    unsigned char* pin = SLOT(c_pin, d.n_pin);
    for (int k = 0; k < d.n_pin; ++k) pin[k] = pq[k] ? 0 : 1;
    const double* held_q = SLOT(held_q, d.n_ctrl);
    const double* svc_on = SLOT(svc_on, d.n_ctrl);
    double* c_held = SLOT(c_held_q, d.n_ctrl);
    unsigned char* plan_held = SLOT(c_plan_held, d.n_ctrl);
    double* v_set = SLOT(c_v_set, d.n_grp);
    for (int g = 0; g < d.n_grp; ++g) v_set[g] = d.vc_v_set[g];
    for (int j = 0; j < d.n_ctrl; ++j) {
        c_held[j] = held_q[j];
        plan_held[j] = d.vc_plan_held[j] ? 1 : 0;
        if (d.vc_plan_held[j] && d.vc_kind[j] == VC_SVC && isfinite(svc_on[j])) {
            plan_held[j] = 0;
            v_set[d.vc_group[j]] = svc_on[j];
        }
    }
    const int* regime = SLOT(hvdc_regime, d.n_hvdc);
    int* status = SLOT(c_hvdc_status, d.n_hvdc);
    for (int e = 0; e < d.n_hvdc; ++e) status[e] = regime[e] == 2 ? d.hv_st0[e] : regime[e];
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
}

// ---- finish ----------------------------------------------------------------------------------------
__device__ int finish(const Dev& d, int s, int inner_status)
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
        for (int k = 0; k < d.n_units; ++k)
            d.r_unit_mw[static_cast<size_t>(r) * d.n_units + k] =
                d.ds_kind[k] == 0 ? SLOT(gen_tp, d.n_gen)[d.ds_el[k]] : SLOT(sto_tp, d.n_sto)[d.ds_el[k]];
        for (int e = 0; e < d.n_hvdc; ++e) d.r_hvdc[static_cast<size_t>(r) * d.n_hvdc + e] = SLOT(hvdc_regime, d.n_hvdc)[e];
    }
    return row_status;
}

// ---- drive: the for loop over the loops of a pass, from loop i --------------------------------------
// returns 0 Resolve, 1 Done (row_status set)
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
            row_status = finish(d, s, inner_status);
            return 1;
        }
        const int loop = active[i];
        const bool settled = !rechecking && done_at[i] == S1(v_version);
        int status = STABLE;
        if (!settled && d.loop_kind[loop] == 3) {
            if (!rl_ready) { S1(p_recheck) = rechecking ? 1 : 0; return 2; }
            rl_ready = false;
            status = check_rl_apply(d, s, *rv);
        } else if (!settled) {
            status = check(d, s, *rv, loop);
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
            materialize(d, s);
            return 0;
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
    Row rv{g, d.n_watch, d.n_ctrl};
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
        int next;
        if (S1(first)) {
            S1(first) = 0;
            if (!S1(solver_ok)) {
                row_status = finish(d, s, inner_status[i]);
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
}

__global__ void materialize_kernel(Dev d, const int* slots, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) materialize(d, slots[i]);
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
    DeviceOuterModel(std::shared_ptr<OuterHook> host, const DeviceHostSide* side);
    const Layout& layout() const override { return host_->layout(); }
    void set_keep_adjoint(bool keep) override {
        if (keep) throw DeviceNotImplemented("the outer adjoint (keep_final_jacobian)");
    }
    void begin(int n_slots, int n_rows) override;
    void load(int slot, int row) override;
    void move(int from, int to) override;
    const SlotControls& controls(int slot) const override;
    void clear_vm_reset(int slot) override;
    void row_trafo_blocks(int, std::vector<int>& trafo, std::vector<double>& block) const override {
        trafo.clear(); block.clear();
    }
    Next on_solved(int slot, bool converged, int nr_iter, int inner_status, const SlotData* data,
                   int& row_status) override;
    const RowResult& row_result(int row) const override;
    int n_rows() const override { return n_rows_; }
    void set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p) override {
        for (double v : gen_p) if (!std::isnan(v)) throw DeviceNotImplemented("rows with their own generator targets");
        for (double v : storage_p) if (!std::isnan(v)) throw DeviceNotImplemented("rows with their own storage targets");
        host_->set_row_targets(n_rows, std::move(gen_p), std::move(storage_p));
    }
    void set_row_overrides(RowOverrides ov) override {
        note_plain_rows(ov);
        host_->set_row_overrides(std::move(ov));
    }
    // DeviceOuterHook
    void load_rows(const std::vector<int>& slots, const std::vector<int>& rows) override;
    void decide(int n, const int* d_slots, const int* d_conv, const int* d_nr, const int* d_inner, const int* d_grow,
                const double* d_gather, int stride, int* d_next, int* d_status, void* stream) override;
    void set_stream(void* stream) override { stream_ = static_cast<cudaStream_t>(stream); }
    void move_slots(int n, const int* d_from, const int* d_to, const int* h_from, const int* h_to) override;
    int sbus_block() const override { return std::max(nc_, 1); }
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
    void alloc_rows(int R);
    void fill_row(const DeviceSlotImage& img, HostRow& hr);
    void note_plain_rows(const RowOverrides& ov);
    bool plain(int row) const {
        return row >= 0 && (static_cast<size_t>(row) >= plain_.size() || plain_[static_cast<size_t>(row)]);
    }
    void compare_with_template(const HostRow& hr, int row) const;
    void compare_rows(const HostRow& host, const HostRow& dev, int row) const;
    bool mask_from_template(int row) const;
    void read_row(int slot, HostRow& hr) const;
    void set_int(int slot, const char* name, int v);
    int get_int(int slot, const char* name) const;
    void check_errors(const std::vector<int>& slots) const;
    void bind(int*& f, size_t base) { f = raw(p_int_) + base; }
    void bind(unsigned char*& f, size_t base) { f = raw(p_u8_) + base; }
    void bind(double*& f, size_t base) { f = raw(p_f64_) + base; }

    std::shared_ptr<OuterHook> host_;
    const DeviceHostSide* side_;
    DeviceTables t_;
    int n_units_ = 0, n_hl_ = 0, n_mon_ = 0, nc_ = 0, vcap_ = 0, n_loops_ = 0, stride_ = 0;
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
    thrust::device_vector<double> d_mon_low_, d_mon_high_, d_svc_low_t_, d_svc_high_t_;
    thrust::device_vector<int> i_ds_el_, i_ds_kind_, i_ds_bus_;
    thrust::device_vector<double> d_ds_w_, d_ds_min_, d_ds_max_;
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
    // M3 step 2: the slot a row that changes nothing starts from, loaded once through the host
    // model and copied on the device for the next ones (GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE=1:
    // every such row also loaded through the host and compared)
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
    // staging
    thrust::device_vector<int> st_slots_, st_conv_, st_nr_, st_inner_, st_grow_, st_next_, st_status_;
    // load_rows' lists and staging, kept across calls (an allocation would sync the device)
    thrust::device_vector<int> ld_slots_, ld_tmpl_, ld_mask_, ld_si_;
    thrust::device_vector<unsigned char> ld_su_;
    thrust::device_vector<double> ld_sf_;
    thrust::device_vector<double> st_gather_;
};

template <typename T>
thrust::device_vector<T> up(const std::vector<T>& v)
{
    return v.empty() ? thrust::device_vector<T>(1) : thrust::device_vector<T>(v.begin(), v.end());
}

DeviceOuterModel::DeviceOuterModel(std::shared_ptr<OuterHook> host, const DeviceHostSide* side)
    : host_(std::move(host)), side_(side)
{
    try {
        t_ = side_->device_tables();
    } catch (const std::domain_error& e) {
        throw DeviceNotImplemented(e.what());
    }
    const Layout& L = host_->layout();
    if (L.n_bc > 0) throw DeviceNotImplemented("BranchControl transformers");
    if (L.n_sc > 0) throw DeviceNotImplemented("ShuntControl controllers");
    if (!L.slack_bus.empty()) throw DeviceNotImplemented("a distributed-slack Newton (NROuterDist_*)");
    n_units_ = static_cast<int>(t_.ds_el_id.size());
    n_hl_ = static_cast<int>(t_.hl_hvdc_id.size());
    n_mon_ = static_cast<int>(t_.mon_svc.size());
    nc_ = static_cast<int>(t_.corr_bus.size());
    n_loops_ = static_cast<int>(t_.loop_kind.size());
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
    d_mon_low_ = up(t_.mon_low_vm); d_mon_high_ = up(t_.mon_high_vm);
    d_svc_low_t_ = up(t_.svc_standby_low_target); d_svc_high_t_ = up(t_.svc_standby_high_target);
    i_ds_el_ = up(t_.ds_el_id); i_ds_kind_ = up(t_.ds_kind); i_ds_bus_ = up(t_.ds_bus);
    d_ds_w_ = up(t_.ds_weight); d_ds_min_ = up(t_.ds_min_mw); d_ds_max_ = up(t_.ds_max_mw);
    i_corr_bus_ = up(t_.corr_bus); i_corr_of_bus_ = up(corr_of_bus_h_);
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
    X(gen_tp, 2, t_.n_gen) X(sto_tp, 2, t_.n_sto)                                                        \
    X(dset, 1, nc_) X(fset, 1, nc_) X(dre, 2, nc_) X(dim, 2, nc_) X(fq, 2, nc_)                          \
    X(vmr_bus, 0, vcap_) X(vmr_vm, 2, vcap_)                                                             \
    X(rl_present, 1, t_.rl_n) X(rl_tp, 2, t_.rl_n) X(rl_fq, 2, t_.rl_n) X(rl_state, 0, t_.rl_n)            \
    X(rl_nb, 0, t_.rl_n) X(rl_real, 0, t_.rl_n)                                                          \
    X(ds_present, 1, n_units_) X(ds_inj, 2, n_units_) X(ds_cur, 2, n_units_) X(ds_ini, 2, n_units_)      \
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
    results_.assign(r, RowResult());
    fin_built_.assign(r, 0);
}

void DeviceOuterModel::build_dev()
{
    Dev& d = dev_;
    d.n_bus = t_.n_bus; d.n_gen = t_.n_gen; d.n_sto = t_.n_sto; d.n_watch = t_.n_watch; d.n_loops = n_loops_;
    d.n_pin = t_.n_pin; d.n_ctrl = t_.n_ctrl; d.n_grp = t_.n_grp; d.n_hvdc = t_.n_hvdc; d.rl_n = t_.rl_n;
    d.rl_n_bus_q = t_.rl_n_bus_q;
    d.n_units = n_units_; d.n_hl = n_hl_; d.n_mon = n_mon_; d.nc = nc_; d.vcap = vcap_;
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
    d.mon_low = raw(d_mon_low_); d.mon_high = raw(d_mon_high_); d.svc_low_t = raw(d_svc_low_t_); d.svc_high_t = raw(d_svc_high_t_);
    d.ds_el = raw(i_ds_el_); d.ds_kind = raw(i_ds_kind_); d.ds_bus = raw(i_ds_bus_);
    d.ds_w = raw(d_ds_w_); d.ds_min = raw(d_ds_min_); d.ds_max = raw(d_ds_max_);
    d.corr_bus = raw(i_corr_bus_); d.corr_of_bus = raw(i_corr_of_bus_);
}

void DeviceOuterModel::begin(int n_slots, int n_rows)
{
    host_->begin(n_slots, n_rows);
    alloc_slots(n_slots);
    alloc_rows(n_rows);
    build_dev();
    have_tmpl_ = false;
    tmpl_row_.reset();
    const char* env = std::getenv("GPUSIM2GRID_OUTER_DEVICE_CHECK_TEMPLATE");
    check_tmpl_ = env != nullptr && env[0] != '\0' && env[0] != '0';
}

void DeviceOuterModel::note_plain_rows(const RowOverrides& ov)
{
    // a row is plain when nothing the loops start from depends on it: the branches it opens only
    // reach PhaseControl / TransformerVoltageControl / ShuntVoltageControl and BranchControl,
    // which the device model refuses
    size_t n = 0;
    for (size_t m : {ov.masked.size(), ov.pq_buses.size(), ov.vc_tripped.size(), ov.hvdc_off.size(),
                     ov.gen_bus.size(), ov.lost_p_mw.size()}) n = std::max(n, m);
    const size_t ng = static_cast<size_t>(t_.n_gen), ns = static_cast<size_t>(t_.n_sto), ngr = static_cast<size_t>(t_.n_grp);
    if (ng > 0) n = std::max({n, ov.gen_off.size() / ng, ov.gen_vm.size() / ng});
    if (ns > 0) n = std::max(n, ov.storage_off.size() / ns);
    if (ngr > 0) n = std::max(n, ov.vc_v_set.size() / ngr);
    plain_.assign(n, 1);
    mask_only_.assign(n, 0);
    masked_rows_.assign(n, {});
    auto any_row = [](const auto& m, size_t r, size_t w, auto pred) {
        if (w == 0 || (r + 1) * w > m.size()) return false;
        for (size_t k = r * w; k < (r + 1) * w; ++k) if (pred(m[k])) return true;
        return false;
    };
    // load_off only reaches TransformerVoltageControl; counted anyway, as the conservative choice
    const size_t nl = ov.load_off.empty() || n == 0 ? 0 : ov.load_off.size() / std::max<size_t>(n, 1);
    for (size_t r = 0; r < n; ++r) {
        bool changes = false;
        const bool has_mask = r < ov.masked.size() && !ov.masked[r].empty();
        changes = changes || (r < ov.pq_buses.size() && !ov.pq_buses[r].empty());
        changes = changes || (r < ov.vc_tripped.size() && !ov.vc_tripped[r].empty());
        changes = changes || (r < ov.hvdc_off.size() && !ov.hvdc_off[r].empty());
        changes = changes || (r < ov.gen_bus.size() && !ov.gen_bus[r].empty());
        changes = changes || (r < ov.lost_p_mw.size() && !std::isnan(ov.lost_p_mw[r]));
        changes = changes || any_row(ov.gen_off, r, ng, [](char c) { return c != 0; });
        changes = changes || any_row(ov.gen_vm, r, ng, [](double v) { return !std::isnan(v); });
        changes = changes || any_row(ov.storage_off, r, ns, [](char c) { return c != 0; });
        changes = changes || any_row(ov.vc_v_set, r, ngr, [](double v) { return !std::isnan(v); });
        changes = changes || any_row(ov.load_off, r, nl, [](char c) { return c != 0; });
        plain_[r] = (changes || has_mask) ? 0 : 1;
        mask_only_[r] = (!changes && has_mask) ? 1 : 0;
        if (has_mask) {
            masked_rows_[r] = ov.masked[r];
            std::sort(masked_rows_[r].begin(), masked_rows_[r].end());
        }
    }
}

// a masked row loads from the template unless the mask touches what the template's load decided
// from the whole grid: DistributedSlack left without a unit off the slack bus (the loop goes out),
// a bus it froze or reset (a resumed frozen bus the row's plan no longer has)
bool DeviceOuterModel::mask_from_template(int row) const
{
    if (!tmpl_row_ || row < 0 || static_cast<size_t>(row) >= mask_only_.size() || !mask_only_[static_cast<size_t>(row)])
        return false;
    const std::vector<int>& m = masked_rows_[static_cast<size_t>(row)];
    auto is_masked = [&](int b) { return std::binary_search(m.begin(), m.end(), b); };
    const auto& ti = tmpl_row_->i;
    const auto& tu = tmpl_row_->u;
    const size_t act = field_of_.at("active").off, n_act = static_cast<size_t>(ti[field_of_.at("n_active").off]);
    for (size_t k = 0; k < n_act; ++k) {
        if (t_.loop_kind[static_cast<size_t>(ti[act + k])] != DL_DISTRIBUTED_SLACK) continue;
        bool off_slack = false;
        for (int u = 0; u < n_units_ && !off_slack; ++u) {
            const int b = t_.ds_bus[static_cast<size_t>(u)];
            off_slack = !is_masked(b) && b != t_.slack_bus;
        }
        if (!off_slack) return false;
    }
    const size_t fset = field_of_.at("fset").off;
    for (int ci = 0; ci < nc_; ++ci)
        if (tu[fset + static_cast<size_t>(ci)] && is_masked(t_.corr_bus[static_cast<size_t>(ci)])) return false;
    const size_t vb = field_of_.at("vmr_bus").off, nv = static_cast<size_t>(ti[field_of_.at("vmr_n").off]);
    for (size_t k = 0; k < nv; ++k) if (is_masked(ti[vb + k])) return false;
    return true;
}

void DeviceOuterModel::compare_rows(const HostRow& host, const HostRow& dev, int row) const
{
    for (const auto& kv : field_of_) {
        const Field& f = kv.second;
        if (kv.first.rfind("c_", 0) == 0) continue;   // controls: materialize writes them after
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

void DeviceOuterModel::compare_with_template(const HostRow& hr, int row) const
{
    const size_t row_off = field_of_.at("row").off;
    for (const auto& kv : field_of_) {
        const Field& f = kv.second;
        for (size_t k = 0; k < f.width; ++k) {
            const size_t at = f.off + k;
            bool same;
            if (f.kind == 0) same = at == row_off || hr.i[at] == tmpl_row_->i[at];
            else if (f.kind == 1) same = hr.u[at] == tmpl_row_->u[at];
            else same = hr.f[at] == tmpl_row_->f[at] || (std::isnan(hr.f[at]) && std::isnan(tmpl_row_->f[at]));
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
    if (img.row < 0) {
    } else if (!have_tmpl_) {
        gen_tp_tmpl_ = img.gen_target_p; sto_tp_tmpl_ = img.sto_target_p; have_tmpl_ = true;
    } else {
        auto same = [](const std::vector<double>& a, const std::vector<double>& b) {
            if (a.size() != b.size()) return false;
            for (size_t k = 0; k < a.size(); ++k)
                if (!(a[k] == b[k] || (std::isnan(a[k]) && std::isnan(b[k])))) return false;
            return true;
        };
        if (!same(gen_tp_tmpl_, img.gen_target_p) || !same(sto_tp_tmpl_, img.sto_target_p))
            throw DeviceNotImplemented("rows starting from different generator targets");
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
    set_field(hr.u, I("ds_present"), img.ds_present);
    set_field(hr.f, I("ds_inj"), img.ds_injection);
    set_field(hr.f, I("ds_cur"), img.ds_current);
    set_field(hr.f, I("ds_ini"), img.ds_initial);
    set_field(hr.u, I("hl_present"), img.hl_present);
    set_field(hr.i, I("hl_dir"), img.hl_frozen_dir);
}

// row i of the staging pools -> slot slots[i] of the device pools
template <typename T>
__global__ void scatter_rows_kernel(T* pool, const T* staged, const int* slots, int n, size_t w)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(n) * w) return;
    const size_t i = t / w, k = t % w;
    pool[static_cast<size_t>(slots[i]) * w + k] = staged[t];
}

// a persistent device buffer holding `v` (grown, never shrunk: an allocation would sync the device)
// (pageable host memory: staged before the call returns, `v` may die after)
template <typename T>
void put_buf(thrust::device_vector<T>& d, const std::vector<T>& v, cudaStream_t st)
{
    if (v.empty()) return;
    if (d.size() < v.size()) d.resize(v.size() + v.size() / 2 + 64);   // amortised (a resize fills and syncs)
    check_cuda(cudaMemcpyAsync(raw(d), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, st), "upload");
}

template <typename T>
void scatter(thrust::device_vector<T>& pool, thrust::device_vector<T>& stage, const std::vector<T>& staged,
             const int* d_slots, int n, size_t w, cudaStream_t st)
{
    if (n == 0 || w == 0) return;
    put_buf(stage, staged, st);
    const size_t total = static_cast<size_t>(n) * w;
    scatter_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, st>>>(raw(pool), raw(stage), d_slots, n, w);
    check_cuda(cudaGetLastError(), "scatter");
}

// the template slot row into each listed slot, then each slot's own row index
template <typename T>
__global__ void template_rows_kernel(T* pool, const T* tmpl, const int* slots, int n, size_t w)
{
    const size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(n) * w) return;
    pool[static_cast<size_t>(slots[t / w]) * w + t % w] = tmpl[t % w];
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
    for (int k = 0; k < d.n_units; ++k) {
        if (!masked[d.ds_bus[k]]) continue;
        ds_present[k] = 0;
        SLOT(ds_inj, d.n_units)[k] = 0.; SLOT(ds_cur, d.n_units)[k] = 0.; SLOT(ds_ini, d.n_units)[k] = 0.;
    }
}

void DeviceOuterModel::load_rows(const std::vector<int>& slots_in, const std::vector<int>& rows_in)
{
    if (slots_in.empty()) return;
    // no template yet: the first plain row alone through the host (it becomes the template), so
    // that the other plain rows of this call are copied already
    if (!tmpl_row_ && !check_tmpl_ && slots_in.size() > 1) {
        for (size_t k = 0; k < slots_in.size(); ++k) {
            if (!plain(rows_in[k])) continue;
            load_rows({slots_in[k]}, {rows_in[k]});
            std::vector<int> s2, r2;
            for (size_t j = 0; j < slots_in.size(); ++j)
                if (j != k) { s2.push_back(slots_in[j]); r2.push_back(rows_in[j]); }
            load_rows(s2, r2);
            return;
        }
    }
    // the plain rows once the template exists: copied on the device, no host load
    std::vector<int> slots, rows, t_slots, t_rows, m_slots, m_rows;
    for (size_t k = 0; k < slots_in.size(); ++k) {
        const int row = rows_in[k];
        if (tmpl_row_ && !check_tmpl_ && plain(row)) { t_slots.push_back(slots_in[k]); t_rows.push_back(row); }
        else if (mask_from_template(row))           { m_slots.push_back(slots_in[k]); m_rows.push_back(row); }
        else                                        { slots.push_back(slots_in[k]); rows.push_back(row); }
    }
    // the masked rows: the template, then their masks on the device (only the mask lists go up)
    if (!m_slots.empty()) {
        g2g_nvtx::Range nv("device load: masked rows from the template");
        const int nm = static_cast<int>(m_slots.size());
        std::vector<int> mptr(1, 0), mbus;
        for (int r : m_rows) {
            const auto& m = masked_rows_[static_cast<size_t>(r)];
            mbus.insert(mbus.end(), m.begin(), m.end());
            mptr.push_back(static_cast<int>(mbus.size()));
        }
        std::vector<int> pack(m_slots);
        pack.insert(pack.end(), m_rows.begin(), m_rows.end());
        pack.insert(pack.end(), mptr.begin(), mptr.end());
        pack.insert(pack.end(), mbus.begin(), mbus.end());
        put_buf(ld_mask_, pack, stream_);
        const int* d_ms = raw(ld_mask_);
        const int* d_mr = d_ms + nm;
        const int* d_mptr = d_mr + nm;
        const int* d_mbus = d_mptr + nm + 1;
        auto copy = [&](auto& pool, auto& tmpl, size_t w) {
            using T = typename std::remove_reference<decltype(pool)>::type::value_type;
            const size_t total = static_cast<size_t>(nm) * w;
            if (total == 0) return;
            template_rows_kernel<T><<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream_>>>(raw(pool), raw(tmpl), d_ms, nm, w);
        };
        copy(p_int_, tmpl_i_, w_int_);
        copy(p_u8_, tmpl_u_, w_u8_);
        copy(p_f64_, tmpl_f_, w_f64_);
        set_rows_kernel<<<(nm + 63) / 64, 64, 0, stream_>>>(raw(p_int_), d_ms, d_mr, nm, w_int_, field_of_.at("row").off);
        mask_rows_kernel<<<(nm + 63) / 64, 64, 0, stream_>>>(dev_, d_ms, nm, d_mptr, d_mbus);
        check_cuda(cudaGetLastError(), "masked load");
        if (check_tmpl_) {
            check_cuda(cudaStreamSynchronize(stream_), "masked load");   // each one also through the host model, compared field by field
            HostRow hr, dr;
            DeviceSlotImage img;
            for (int k = 0; k < nm; ++k) {
                read_row(m_slots[static_cast<size_t>(k)], dr);
                host_->load(m_slots[static_cast<size_t>(k)], m_rows[static_cast<size_t>(k)]);
                side_->device_slot_image(m_slots[static_cast<size_t>(k)], img);
                fill_row(img, hr);
                compare_rows(hr, dr, m_rows[static_cast<size_t>(k)]);
            }
        }
        materialize_kernel<<<(nm + 63) / 64, 64, 0, stream_>>>(dev_, d_ms, nm);
        check_cuda(cudaGetLastError(), "materialize");
    }
    if (!t_slots.empty()) {
        g2g_nvtx::Range nv("device load: template rows");
        const int nt = static_cast<int>(t_slots.size());
        std::vector<int> tpack(t_slots);
        tpack.insert(tpack.end(), t_rows.begin(), t_rows.end());
        put_buf(ld_tmpl_, tpack, stream_);
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
        materialize_kernel<<<(nt + 63) / 64, 64, 0, stream_>>>(dev_, d_ts, nt);
        check_cuda(cudaGetLastError(), "template load");
    }
    const int n = static_cast<int>(slots.size());
    if (n == 0) return;
    g2g_nvtx::Range nv("device load: rows through the host model");
    std::vector<int> si(static_cast<size_t>(n) * w_int_);
    std::vector<unsigned char> su(static_cast<size_t>(n) * w_u8_);
    std::vector<double> sf(static_cast<size_t>(n) * w_f64_);
    HostRow hr;
    DeviceSlotImage img;
    for (int k = 0; k < n; ++k) {
        host_->load(slots[static_cast<size_t>(k)], rows[static_cast<size_t>(k)]);
        try {
            side_->device_slot_image(slots[static_cast<size_t>(k)], img);
        } catch (const std::domain_error& e) {
            throw DeviceNotImplemented(e.what());
        }
        fill_row(img, hr);
        const int row = rows[static_cast<size_t>(k)];
        if (plain(row)) {
            if (!tmpl_row_) {   // the first plain row: the template of the next ones
                tmpl_row_ = std::make_unique<HostRow>(hr);
                tmpl_i_.assign(hr.i.begin(), hr.i.end());
                tmpl_u_.assign(hr.u.begin(), hr.u.end());
                tmpl_f_.assign(hr.f.begin(), hr.f.end());
            } else if (check_tmpl_) {
                compare_with_template(hr, row);
            }
        }
        std::copy(hr.i.begin(), hr.i.end(), si.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(k) * w_int_));
        std::copy(hr.u.begin(), hr.u.end(), su.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(k) * w_u8_));
        std::copy(hr.f.begin(), hr.f.end(), sf.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(k) * w_f64_));
    }
    put_buf(ld_slots_, slots, stream_);
    scatter(p_int_, ld_si_, si, raw(ld_slots_), n, w_int_, stream_);
    scatter(p_u8_, ld_su_, su, raw(ld_slots_), n, w_u8_, stream_);
    scatter(p_f64_, ld_sf_, sf, raw(ld_slots_), n, w_f64_, stream_);
    materialize_kernel<<<(n + 63) / 64, 64, 0, stream_>>>(dev_, raw(ld_slots_), n);
    check_cuda(cudaGetLastError(), "materialize");
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
    c.bc_ctl.clear(); c.bc_rctl.clear(); c.sc_set.clear(); c.sc_b_new.clear(); c.sc_ctl.clear(); c.slack_w.clear();
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
        g[static_cast<size_t>(stride_ - 2)] = data->slack_absorbed;
        g[static_cast<size_t>(stride_ - 1)] = data->unrealistic ? 1. : 0.;
    }
    st_slots_.assign(1, slot);
    st_conv_.assign(1, converged ? 1 : 0);
    st_nr_.assign(1, nr_iter);
    st_inner_.assign(1, inner_status);
    st_grow_.assign(1, 0);
    put_buf(st_gather_, g, stream_);
    st_next_.assign(1, 0);
    st_status_.assign(1, 0);
    decide(1, raw(st_slots_), raw(st_conv_), raw(st_nr_), raw(st_inner_), raw(st_grow_), raw(st_gather_), stride_,
           raw(st_next_), raw(st_status_), stream_);
    check_cuda(cudaStreamSynchronize(stream_), "on_solved");
    check_errors({slot});
    const int next = st_next_[0];
    row_status = st_status_[0];
    if (next == 1) {
        const int r = get_int(slot, "row");
        if (r >= 0) fin_built_[static_cast<size_t>(r)] = 0;
    }
    return next == 1 ? Next::Done : Next::Resolve;
}

void DeviceOuterModel::decide(int n, const int* d_slots, const int* d_conv, const int* d_nr, const int* d_inner,
                              const int* d_grow, const double* d_gather, int stride, int* d_next, int* d_status,
                              void* stream)
{
    if (n == 0) return;
    on_solved_kernel<<<n, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        dev_, d_slots, n, d_conv, d_nr, d_inner, d_grow, d_gather, stride, d_next, d_status);
    check_cuda(cudaGetLastError(), "decide");
}

// one element / n elements of a per-row device array
template <typename T>
T get1(const T* p, int i)
{
    T v;
    check_cuda(cudaMemcpy(&v, p + i, sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
template <typename T>
std::vector<T> get(const T* p, int row, size_t width, size_t n)
{
    std::vector<T> v(n);
    if (n > 0)
        check_cuda(cudaMemcpy(v.data(), p + static_cast<size_t>(row) * width, n * sizeof(T), cudaMemcpyDeviceToHost),
                   "download");
    return v;
}

const RowResult& DeviceOuterModel::row_result(int row) const
{
    const size_t r = static_cast<size_t>(row);
    RowResult& res = results_.at(r);
    if (fin_built_[r]) return res;
    fin_built_[r] = 1;
    res = RowResult();
    const Dev& d = dev_;
    res.status = get1(d.r_status, row);
    if (res.status < 0) return res;   // never ran (not simulated): the default result
    res.unrealistic = get1(d.r_unreal, row) != 0;
    const int failed = get1(d.r_failed, row);
    if (failed >= 0) {
        static const char* names[] = {"DistributedSlack", "AcHvdcAcEmulationLimits", "VoltageMonitoring", "ReactiveLimits"};
        res.failed_loop = names[t_.loop_kind[static_cast<size_t>(failed)]];
    }
    res.nb_outer_iterations = get1(d.r_outer, row);
    res.nb_nr_iterations = get1(d.r_nr, row);
    // the generators' and storage units' targets: the rows' common start, the units' own where they moved
    std::vector<double> gen_tp = gen_tp_tmpl_, sto_tp = sto_tp_tmpl_;
    const std::vector<double> units = get(d.r_unit_mw, row, static_cast<size_t>(std::max(n_units_, 1)), static_cast<size_t>(n_units_));
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
    const std::vector<int> hv = get(d.r_hvdc, row, static_cast<size_t>(std::max(t_.n_hvdc, 1)), static_cast<size_t>(t_.n_hvdc));
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
                     : code == ERR_NO_PIN ? "a local controller bus without a reserved Vm column" : "an unknown error";
    return "device outer model, slot " + std::to_string(slot) + ": " + what;
}

std::shared_ptr<OuterHook> make_device_outer_model_from(std::shared_ptr<OuterHook> host, const DeviceHostSide* side)
{
    return std::make_shared<DeviceOuterModel>(std::move(host), side);
}

}  // namespace outer
