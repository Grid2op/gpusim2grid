// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/ls2g_outer_model.cpp -- see ls2g_outer_model.hpp
//
// A port, not a call: lightsim2grid's loop classes act on an OuterContext built
// around a whole NR system and the full V of a row, which would have to come
// back to the host at every outer iteration of every row. Here each loop's
// check reads only what the device gathered for the row (outer_hook.hpp's
// SlotData: V, computed and scheduled power at the watched buses, controllers'
// Q), and its action edits the row's control state (SlotState), which the
// driver uploads before the row is solved again. The code follows the
// upstream files line by line (file names in each section); a difference in
// what a row ends at is a bug here.
// =============================================================================

#include "ls2g_outer_model.hpp"
#include "device_outer_model.hpp"

#include "../ledger_data.hpp"

#include <LSGrid.hpp>
#include <batch_algorithm/BusQCheck.hpp>
#include <batch_algorithm/HvdcPCheck.hpp>
#include <batch_algorithm/SvcStandbyCheck.hpp>
#include <element_container/SlackRedistribution.hpp>
#include <powerflow_algorithm/outer_loop/OuterRowView.hpp>
#include <powerflow_algorithm/outer_loop/DistributedSlackLoop.hpp>
#include <powerflow_algorithm/outer_loop/HvdcAcEmulationLimitsLoop.hpp>
#include <powerflow_algorithm/outer_loop/VoltageMonitoringLoop.hpp>
#include <powerflow_algorithm/outer_loop/ReactiveLimitsLoop.hpp>
#include <powerflow_algorithm/outer_loop/OuterLoopDriver.hpp>
#include <powerflow_algorithm/outer_loop/PhaseControlLoop.hpp>
#include <powerflow_algorithm/outer_loop/TransformerVoltageControlLoop.hpp>
#include <powerflow_algorithm/outer_loop/ShuntVoltageControlLoop.hpp>

#include <algorithm>
#include <array>
#include <tuple>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace outer {

namespace {

using cplx = std::complex<double>;
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

enum class Kind { DistributedSlack, HvdcLimits, VoltageMonitoring, ReactiveLimits, PhaseControl,
                  TransformerVoltageControl, ShuntVoltageControl };
enum Status { STABLE = 0, UNSTABLE = 1, FAILED = 2, CHECK_AGAIN = 3 };   // OuterLoopStatus

// A std::map's interface over a sorted vector (the subset the model uses): the same keys, the
// same order, the same values. A row's maps hold tens of buses, are searched once per bus of
// every check and copied at every load: contiguous storage instead of a tree.
template <class K, class V>
class FlatMap {
public:
    using value_type = std::pair<K, V>;
    using iterator = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;
    iterator begin() { return v_.begin(); }
    iterator end() { return v_.end(); }
    const_iterator begin() const { return v_.begin(); }
    const_iterator end() const { return v_.end(); }
    const_iterator cbegin() const { return v_.cbegin(); }
    const_iterator cend() const { return v_.cend(); }
    bool empty() const { return v_.empty(); }
    size_t size() const { return v_.size(); }
    iterator find(const K& k) {
        auto it = lower(k);
        return (it != v_.end() && it->first == k) ? it : v_.end();
    }
    const_iterator find(const K& k) const {
        auto it = std::lower_bound(v_.begin(), v_.end(), k, [](const value_type& a, const K& b) { return a.first < b; });
        return (it != v_.end() && it->first == k) ? it : v_.end();
    }
    size_t count(const K& k) const { return find(k) != v_.end() ? 1 : 0; }
    V& operator[](const K& k) {
        auto it = lower(k);
        if (it == v_.end() || it->first != k) it = v_.insert(it, value_type(k, V()));
        return it->second;
    }
    size_t erase(const K& k) {
        auto it = find(k);
        if (it == v_.end()) return 0;
        v_.erase(it);
        return 1;
    }
private:
    iterator lower(const K& k) {
        return std::lower_bound(v_.begin(), v_.end(), k, [](const value_type& a, const K& b) { return a.first < b; });
    }
    std::vector<value_type> v_;
};

// GPUSIM2GRID_OC_PROFILE=1 (diagnostic): wall time and calls of the model's parts, printed on
// stderr when the model is destroyed. Off: one predictable branch per scope.
enum ProfSec : int {
    P_ON_SOLVED_RESOLVE, P_ON_SOLVED_DONE, P_DS_FOLD, P_CHECK_DS, P_CHECK_HVDC, P_CHECK_VM, P_CHECK_RL,
    P_CHECK_PC, P_CHECK_TVC, P_CHECK_SVC, P_MATERIALIZE, P_FINISH, P_ADJOINT,
    P_LOAD, P_LOAD_PHANTOM, P_LOAD_RL_PLAN, P_LOAD_BC, P_LOAD_UNITS, P_LOAD_SET_ACTIVE,
    P_LOAD_RESUME_TARGETS, P_LOAD_INIT_LOOPS, P_LOAD_RESUME_LOOPS, P_LOAD_RESET, P_LOAD_INIT_DS, P_LOAD_INIT_RL, P_LOAD_MASKED_ROWS, P_LOAD_DS_NUNITS,
    P_DONE_NO_RESOLVE, P_DNR_FAILED, P_DNR_OK_MASKED, P_DNR_OK_MOVED, P_DNR_OK_BASE,
    P_ACT_FIRST, P_ACT_FIRST_LAST = P_ACT_FIRST + 6, P_ACT_ALL, P_ACT_ALL_LAST = P_ACT_ALL + 6,
    P_DS_STABLE_MOVED, P_RL_RESUME_FROZEN, P_RL_BASE, P_RL_RECOMPUTE, P_RL_NO_BASE, P_RL_PLAN_FILTER, P_RL_PLAN_BUSES, P_N_SEC
};
const char* const kProfName[P_N_SEC] = {
    "on_solved (Resolve)", "on_solved (Done)", "  ds_fold", "  check DistributedSlack", "  check HvdcLimits",
    "  check VoltageMonitoring", "  check ReactiveLimits", "  check PhaseControl", "  check TVC",
    "  check ShuntVoltageControl", "  materialize", "  finish", "    adjoint_info",
    "load (row)", "load (phantom)", "  rl plan", "  BranchControl: opened trafos",
    "  DistributedSlack units", "  set_active", "  resume targets", "  init_loops", "  resume_loops", "  reset the slot", "    init DistributedSlack", "    init ReactiveLimits", "rows that mask buses", "DS units copied",
    "on_solved (Done, no re-solve)", "  no re-solve: failed", "  no re-solve: ok, masks buses",
    "  no re-solve: ok, limits moved", "  no re-solve: ok, base limits",
    "first act: DistributedSlack", "first act: HvdcLimits", "first act: VoltageMonitoring", "first act: ReactiveLimits",
    "first act: PhaseControl", "first act: TVC", "first act: ShuntVoltageControl",
    "acts: DistributedSlack", "acts: HvdcLimits", "acts: VoltageMonitoring", "acts: ReactiveLimits",
    "acts: PhaseControl", "acts: TVC", "acts: ShuntVoltageControl",
    "DS STABLE but moved", "rl buses resumed frozen", "rl bus: base sums", "rl bus: moved, recomputed", "rl bus: no base", "    plan: bus_q entries", "    plan: controller buses"};

struct OcProfile {
    bool on = false;
    double ns[P_N_SEC] = {};
    long long n[P_N_SEC] = {};
    OcProfile() { const char* e = std::getenv("GPUSIM2GRID_OC_PROFILE"); on = e != nullptr && *e && *e != '0'; }
    ~OcProfile()
    {
        if (!on) return;
        std::fprintf(stderr, "[GPUSIM2GRID_OC_PROFILE] %-28s %10s %10s %9s\n", "part", "ms", "calls", "us/call");
        for (int s = 0; s < P_N_SEC; ++s)
            if (n[s] > 0)
                std::fprintf(stderr, "[GPUSIM2GRID_OC_PROFILE] %-28s %10.1f %10lld %9.2f\n", kProfName[s], ns[s] * 1e-6,
                             n[s], ns[s] * 1e-3 / static_cast<double>(n[s]));
    }
};
struct ProfScope {
    OcProfile* p;
    int sec;
    std::chrono::steady_clock::time_point t0;
    ProfScope(OcProfile& prof, int s) : p(prof.on ? &prof : nullptr), sec(s)
    {
        if (p) t0 = std::chrono::steady_clock::now();
    }
    // re-labels the scope before it ends (on_solved: Resolve or Done is known at the end)
    void relabel(int s) { sec = s; }
    ~ProfScope()
    {
        if (!p) return;
        p->ns[sec] += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
        ++p->n[sec];
    }
};

struct LoopDecl {
    Kind kind;
    std::string name;
    // DistributedSlack
    double ds_mismatch_mw = 1., ds_eps_mw = 1e-3, ds_moved_fraction = 0.9;
    bool ds_run_without_participant = false;
    bool   ds_fail = true;
    // ReactiveLimits
    int    rl_max_switch = 3;
    double rl_max_mismatch = 1e-4, rl_base_mva = 100.;
    bool   rl_robust = true;
    double rl_vmin = 0.5, rl_vmax = 2., rl_margin = 1.02, rl_restart = 1.;
    // TransformerVoltageControl: the grid's loop, for its groups(view) and parameters
    std::shared_ptr<const ls2g::TransformerVoltageControlLoop> tvc;
};

// TransformerVoltageControlLoop's per-solve state
struct TvcRatio {   // TransformerRatioManager's, per transformer
    double initial = 1., min = 1., max = 1.;
    double shared_min = 1., shared_max = 1., shared_initial = 1.;
};
struct TvcFrozen {  // a generator bus frozen by INITIAL
    int entry;           // into RlPlan::plan->buses (RlBus::entry)
    int bus_solver;
    bool local;
    double target_vm;
};
struct TvcBranch {  // a branch closed in the row: its solver buses, its trafo id (-1: a line)
    int bus1, bus2, trafo;
};

// ReactiveLimitsLoop::ControllerBus
// what a row changes of a controller bus (ReactiveLimitsLoop::ControllerBus), per slot; the
// rest is RlBus, shared by the slots through their plan
struct RlBusState {
    double target_p = 0.;
    double frozen_q = 0.;
    int state = 0;
    int nb_pv_pq = 0;
    bool realistic = false;
};

// what a generator frozen at a limit of its capability curve follows: the slope its Q follows
// its target P with (ReactiveLimitsLoop::ControllerBus::slopes, OuterQSlope): (gen, controller
// (-1: a local bus)), pu per MW
using RlSlopes = std::vector<std::pair<std::pair<int, int>, double>>;

struct RlBus {
    int entry = -1;          // its entry in the plan's BusQPlan
    int bus_solver = -1;
    bool local = true;
    double min_q = 0., max_q = 0.;
    int reg_bus_solver = -1;
    int group = -1;
    double target_vm = 1.;
    double nominal_v = 0.;
    double target_q = 0.;
    int monitor_svc = -1;    // svc id of a voltage monitor's bus
    // the generator part of its limits (min_q / max_q + the generators' curves) at the base
    // targets -- those of a row that sets none (the resumed ones, or none): grid plan and its
    // masked copies only (a row that reuses them sees exactly these generators)
    bool has_base = false;
    double base_min_q = 0., base_max_q = 0.;
    bool has_storage = false;   // its target P is not in the base value (RlPlan::state0)
    int pin = -1;            // a local bus' switchable index
};

// ReactiveLimits' plan for one kind of row -- the grid's, or a row that masks buses or
// disconnects generators (ReactiveLimitsLoop::_initialize on that row's view): bus_q_check's
// entries plus one per voltage monitor, and the controller buses built on them (their
// target P is the row's, filled at load)
struct RlPlan {
    // the entries RlBus::entry indexes: a row that only masks buses shares the grid's, its
    // buses being the grid's without the masked ones
    std::shared_ptr<const ls2g::bus_q_check::BusQPlan> plan;
    std::vector<RlBus> buses;
    // what a slot starts from, aligned with buses: target_p at the base targets where has_base
    std::vector<RlBusState> state0;
    bool has_monitor = false;
    size_t n_bus_q = 0;    // plan.buses' entries before the monitors: bus_q_check's own plan
};

// everything one row carries through its solves
// Where a solve left the loops, for a row to resume from (lightsim2grid's OuterResumePoint): the
// grid's own last solve (Ls2gModelOptions::resume_from_n), or a row of the same run that ended
// with its loops stable (OuterHook::keep_row_points / set_row_resume) -- a strategy row resumes
// from its contingency's N-1 state, as a second solve on the grid with the contingency applied.
struct ResumePoint {
    std::vector<double> gen_p, sto_p;                    // moved targets, MW, NaN untouched
    ls2g::CplxVect V;                                    // the solve's voltages (the grid's point) ...
    std::vector<std::pair<int, cplx>> v_at;              // ... or at the buses a resuming loop reads, sorted
    std::map<int, int> hvdc;                             // hvdc id -> regime
    std::vector<std::pair<int, double>> svc_on;
    std::map<int, std::pair<int, bool>> frozen;          // solver bus -> (state, realistic)
    std::vector<ls2g::DistributedSlackLoop::ParkedUnit> parked;
    std::vector<int> phase_tap, ratio_tap;               // by trafo id, INT_MIN: not carried
    std::vector<int> sections;                           // by shunt id, INT_MIN: not carried
    // the loss the row counted for its DistributedSlack (lost_p_to_share_mw, 0: none): a row
    // resuming from it shares only what it loses beyond (its contingency's loss is in gen_p)
    double lost_mw = 0.;
};

struct SlotState {
    int row = -1;
    const ResumePoint* rp = nullptr;      // where the row's loops resume from (nullptr: from scratch)
    // the raw slack weights the row gives its units (RowOverrides::gen_slack_w / storage_slack_w,
    // NaN: the grid's "can participate" one; empty: none) -- OuterRowView::Units::slack_weight
    std::vector<double> gen_slack_w, sto_slack_w;
    double lost_mw = 0.;                  // what share_lost_p counted (ResumePoint::lost_mw)
    // ---- the row as its loops see it (OuterRowView; its layout and masked list point
    // into the model and this slot) ---------------------------------------------------------
    ls2g::OuterRowView rowv;
    std::vector<int> masked;              // sorted solver buses the row masks
    std::shared_ptr<const RlPlan> rl;
    std::vector<int> active;              // the loops is_needed on this row (indices into loops_)
    int last_fixing = -1;
    // ---- control state (OuterControls + OuterInjections) -------------------
    std::vector<unsigned char> pq;        // per switchable bus: 1 PQ (unpinned)
    std::vector<double> held_q;           // per controller, pu, NaN not held by value
    std::vector<double> svc_on;           // per controller, the standby target, NaN idle
    std::vector<int> hvdc_regime;         // per ledger hvdc entry, 2 = KEEP
    std::vector<char> hvdc_tripped;       // per ledger hvdc entry: the row trips it (empty: none)
    bool hvdc_line_tripped(int hvdc_id, const std::vector<int>& ids) const {
        if (hvdc_tripped.empty()) return false;
        const auto it = std::find(ids.begin(), ids.end(), hvdc_id);
        return it != ids.end() && hvdc_tripped[static_cast<size_t>(it - ids.begin())] != 0;
    }
    FlatMap<int, cplx> target_delta;      // Sbus_target - the row's Sbus, pu
    FlatMap<int, double> freeze_q;        // Sbus - Sbus_target (imaginary), pu
    std::vector<char> frozen_at;          // per solver bus, 1 when in freeze_q (empty: none yet)
    std::vector<double> gen_target_p;     // OuterInjections::gen_target_p (MW, NaN untouched)
    // RlBus::base_*_q apply: the row starts from the base targets (sets none of its own)
    bool lim_base = false;
    // per solver bus, 1 once a generator of its ReactiveLimits entry moved (empty: none)
    std::vector<char> lim_dirty;
    std::vector<double> storage_target_p;
    std::vector<int> vm_reset_bus;
    std::vector<double> vm_reset_vm;
    // ---- loops' own state ------------------------------------------------------
    std::vector<ls2g::slack_redistribution::Participant> ds_units;
    std::vector<int> ds_unit_bus;
    std::vector<double> ds_current_mw;
    // each unit's target in force before the row's lost-power share moved it (its own set-point:
    // the adjoint's "a unit beyond its bound keeps its own" test)
    std::vector<double> ds_initial_mw;
    // around a distributed-slack Newton (DistributedSlackLoop's in-Newton state): whether the
    // Newton shares the slack on the units (the loop runs on the row), the units at a bound
    // and on which side, those the row's lost-power share put at a bound
    bool ds_in_newton = false;
    std::vector<char> ds_parked, ds_parked_high, ds_lost_sat;
    bool ds_lost_high = false;
    // ds_parked and ds_current_mw as the last Newton solve had them, which its Jacobian (and so
    // the adjoint, adjoint_info) holds: a check that parks a unit with too small a move to
    // re-solve changes ds_parked for no solve (DistributedSlackLoop::solved_parked_ / solved_mw_);
    // kept only when gradients are wanted (keep_adjoint_)
    std::vector<char> ds_solved_parked;
    std::vector<double> ds_solved_mw;
    void ds_mark_solved() { ds_solved_parked = ds_parked; ds_solved_mw = ds_current_mw; }
    // the last solve's slack_absorbed, and whether its units took it (OuterLoopAlgo::
    // _fold_slack_share: then the mismatch keeps `+ slack_absorbed * weight`); the target Q
    // the share moved, which that solve did not see (OuterInjections::unsolved_q_mvar)
    double sa = 0.;
    bool sa_shared = false;
    double unsolved_q_mvar = 0.;
    ls2g::hvdc_p_check::HvdcPPlan hvdc_plan;
    std::vector<std::pair<int, double>> vm_switched_on;
    std::vector<RlBusState> rl_state;     // aligned with rl->buses (empty: the loop is not active)
    FlatMap<int, RlSlopes> rl_slopes;     // by index in rl->buses: the buses frozen with a slope
    std::vector<ls2g::PhaseControlLoop::Shifter> pc_shifters;
    int tvc_step = 0;                     // 0 INITIAL, 1 CONTROL, 2 COMPLETE
    std::vector<ls2g::TransformerVoltageControlLoop::Group> tvc_groups;
    std::vector<char> tvc_enabled, tvc_controller;   // per trafo id
    std::vector<TvcRatio> tvc_ratios;                // per trafo id
    std::vector<TvcFrozen> tvc_frozen;
    std::set<int> suspended;              // OuterControls::suspended (solver buses)
    std::vector<ls2g::ShuntVoltageControlLoop::Group> sv_groups;
    // ---- ShuntControl, per controller (the ledger's order): its shunts' section counts
    // (ShuntControl::Entry::counts), a switch pending (ShuntSectionControl::take_sections)
    // and its susceptance then, the requested control --------------------------------------
    std::vector<std::vector<int>> sc_counts;
    std::vector<char>   sc_pending;
    std::vector<double> sc_b_new;
    std::vector<signed char> sc_ctl;
    // ---- BranchControl, per transformer (the ledger's order): what the slot's system holds,
    // as lightsim2grid's BranchControl::Entry, and what the loops asked of it (its
    // PhaseShifterControl / RatioTapControl) ----------------------------------------------------
    std::vector<unsigned char> bc_out;     // the row does not have it (drop_out_of_row)
    std::vector<int>    bc_pos, bc_rpos;   // the taps it is at
    std::vector<double> bc_a_tap, bc_rho_tap;
    std::vector<cplx>   bc_tap;            // [4 * n_bc] the block at the taps
    std::vector<unsigned char> bc_moved;   // PhaseShifterControl::tap_moved (for the solve)
    std::vector<int>    bc_req_tap;        // ... its requested_tap
    std::vector<int>    bc_req_rtap;       // RatioTapControl: a move once (INT_MIN: none)
    std::vector<signed char> bc_ctl, bc_rctl;   // requested controls, -1 untouched
    // ---- driver state (OuterLoopAlgo::compute_pf) -----------------------------
    std::vector<int> loop_iter; // per active loop: its UNSTABLE count (OuterContext::iteration)
    // a change a re-solve met with no Newton step is done (OuterLoopAlgo::compute_pf, #264):
    // v_version counts the solves that moved V, done_at[i] the version loop i's change was
    // done on (-1: none); changed = the check in flight said UNSTABLE (not CHECK_AGAIN)
    std::vector<int> done_at;
    int  v_version = 0;
    bool changed = false;
    bool first = true;          // the first solve has not ended yet
    int  i = 0;                 // the loop being checked
    int  last_unstable = -1;
    int  last_status = STABLE;
    std::string last_name;
    bool solver_ok = true;
    bool check_unrealistic = false;     // for the solve in flight
    bool pass_check_unrealistic = false;
    int  old_total = 0;
    int  total_nr = 0;          // at least 1 per solve: what tells a pass re-solved
    int  nr_steps = 0;          // the Newton steps actually taken (RowResult::nb_nr_iterations)
    int  nb_outer = 0;
    int  nb_passes = 0;
    bool unrealistic = false;
    std::vector<std::string> trace;
    SlotControls ctl;
};

// the gathered quantities of one converged slot -- or, before a row's first solve,
// only the voltages it starts from (`full`, every bus: a resumed row's)
[[noreturn]] __attribute__((noinline, cold)) void throw_not_watched(int b)
{
    throw std::runtime_error("outer model: bus " + std::to_string(b) + " is not watched");
}

struct RowView {
    const SlotData* d;
    const std::vector<int>* widx;
    const ls2g::CplxVect* full = nullptr;
    const std::vector<std::pair<int, cplx>>* at = nullptr;   // ResumePoint::v_at
    bool has(int b) const { return b >= 0 && b < static_cast<int>(widx->size()) && (*widx)[b] >= 0; }
    int  w(int b) const {
        if (!has(b)) throw_not_watched(b);
        return (*widx)[b];
    }
    cplx V(int b)    const {
        if (full != nullptr) return cplx((*full)(b).real(), (*full)(b).imag());
        if (at != nullptr) {
            const auto it = std::lower_bound(at->begin(), at->end(), b,
                                             [](const std::pair<int, cplx>& e, int key) { return e.first < key; });
            if (it == at->end() || it->first != b) throw_not_watched(b);
            return it->second;
        }
        const int k = w(b);
        return {d->v_re[k], d->v_im[k]};
    }
    cplx S(int b)    const { const int k = w(b); return {d->s_re[k], d->s_im[k]}; }
    cplx Sbus(int b) const { const int k = w(b); return {d->sb_re[k], d->sb_im[k]}; }
    double bc_a(int e)   const { return d->bc_a[e]; }
    double bc_rho(int e) const { return d->bc_rho[e]; }
    double sc_b(int e)   const { return d->sc_b[e]; }
    double vm(int b) const { return std::abs(V(b)); }
};

// HvdcDroopSolverData::recv_pu / flows_pu (gpusim2grid's hvdc_flows_pu, host side)
double hvdc_recv(double p_ctrl_abs, bool side1_ctrl, double lf1, double lf2, double r)
{
    const double lf_ctrl = side1_ctrl ? lf1 : lf2;
    const double lf_recv = side1_ctrl ? lf2 : lf1;
    const double line_in = (1. - lf_ctrl) * p_ctrl_abs;
    return (1. - lf_recv) * (line_in - r * line_in * line_in);
}
void hvdc_flows(int st, double raw, double lf1, double lf2, double r, double pmax12, double pmax21,
                double& p1, double& p2)
{
    if (st == HVDC_STATUS_TRIPPED) { p1 = 0.; p2 = 0.; return; }
    if (st == 0) {
        if (raw >= 0.) { p1 = raw; p2 = -hvdc_recv(raw, true, lf1, lf2, r); }
        else           { p1 = -hvdc_recv(-raw, false, lf1, lf2, r); p2 = -raw; }
    } else if (st > 0) { p1 = pmax12; p2 = -hvdc_recv(pmax12, true, lf1, lf2, r); }
    else               { p1 = -hvdc_recv(pmax21, false, lf1, lf2, r); p2 = pmax21; }
}

class Ls2gOuterModel final : public OuterHook, public DeviceHostSide {
public:
    Ls2gOuterModel(const ls2g::LSGrid& grid, const LedgerData& ledger, const Ls2gModelOptions& opt);

    const Layout& layout() const override { return layout_; }
    void set_keep_adjoint(bool keep) override { keep_adjoint_ = keep; }
    void begin(int n_slots, int n_rows) override;
    void load(int slot, int row) override;
    void move(int from, int to) override;
    const SlotControls& controls(int slot) const override { return slots_[static_cast<size_t>(slot)].ctl; }
    void clear_vm_reset(int slot) override;
    Next on_solved(int slot, bool converged, int nr_iter, int inner_status,
                   const SlotData* data, int& row_status) override;
    Next on_solved_impl(int slot, bool converged, int nr_iter, int inner_status,
                        const SlotData* data, int& row_status);
    const RowResult& row_result(int row) const override;
    void row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const override;
    int n_rows() const override { return n_rows_; }
    void set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p) override;
    void set_row_overrides(RowOverrides overrides) override {
        check_row_slack_weights(overrides);
        ov_ = std::move(overrides);
    }
    // lightsim2grid's _check_slack_weight_rows under outer loops: the buses the DistributedSlack
    // loop's Newton share uses are reserved once for the batch, from the units that can take
    // part in the grid's slack -- a row's positive weight to any other is refused
    void check_row_slack_weights(const RowOverrides& ov) const;
    void keep_row_points(bool keep) override { keep_points_ = keep; }
    void set_row_resume(std::vector<int> point_of_row) override { resume_of_row_ = std::move(point_of_row); }
    void set_spare_buses(const LedgerData& ledger, const std::vector<int>& spare_grid_bus) override;
    // see outer_device_census
    std::vector<DeviceFeature> census() const;
    // the device model's inputs (outer/device_tables.hpp)
    DeviceTables device_tables() const override;
    void device_slot_image(int slot, DeviceSlotImage& img) const override;
    void device_import_point(int row, const DevicePoint& p) override;
    void device_adjoint_info(const DeviceAdjointState& a, RowResult& r) const override;
    void device_row_blocks(const int* pos, const int* rpos, std::vector<int>& trafo,
                           std::vector<double>& block) const override;
    void device_tvc_image(const SlotState& st, DeviceSlotImage& img) const;
    bool device_prepare_rows(const RowOverrides& ov) override;

private:
    // ---- static data -------------------------------------------------------------
    // a copy of the grid: its containers, capability curves and raw targets, what
    // the loops read at run time. Its solver layout is NOT copied (a copy starts
    // cold): everything that needs it is read off the caller's grid at construction.
    std::shared_ptr<ls2g::LSGrid> grid_;
    ls2g::SolverBusLayout sbl_;          // the caller's grid's AC labelling (a copy has none)
    ls2g::OuterGridView view(const SlotState& st) const { return ls2g::OuterGridView(*grid_, &st.rowv); }
    Ls2gModelOptions opt_;
    // the rows' adjoint terms are wanted (set_keep_adjoint): only then are the slopes of the frozen
    // generators and the last solve's parked units recorded, and adjoint_info run
    bool keep_adjoint_ = false;
    // GPUSIM2GRID_OUTER_CHECK_RL_PLAN=1 (diagnostic): every filtered ReactiveLimits plan compared
    // with build_bus_q_plan's, every filtered unit list with collect_units' (throws on a difference)
    bool check_rl_filter_ = [] { const char* e = std::getenv("GPUSIM2GRID_OUTER_CHECK_RL_PLAN"); return e && *e && *e != '0'; }();
    mutable OcProfile prof_;              // GPUSIM2GRID_OC_PROFILE
    Layout layout_;
    double sn_ = 1.;
    int n_bus_ = 0, n_gen_ = 0, n_sto_ = 0;
    // the base buses' tables (set_spare_buses appends the spares to copies of them)
    struct BaseBuses {
        bool saved = false;
        int n_bus = 0;
        ls2g::SolverBusLayout sbl;
        std::vector<int> pin_bus, watch_bus, q_row_of_bus, pin_of_bus, sc_of_bus, slack_idx_of_bus, widx;
        std::vector<std::vector<int>> vc_at_bus;
        std::vector<char> hvdc_at_bus;
    } base_buses_;
    int slack_bus_ = -1;
    std::vector<int> widx_;                  // solver bus -> watch index
    // per generator, the target P its reactive limits were last read at (gen_limits_at_outer_target)
    // and those limits: the curves never change during a run, and most rows read every
    // generator at the same (resumed) target, so the same input gives the same output
    mutable std::vector<double> glim_p_, glim_lo_, glim_hi_;
    mutable std::vector<char> glim_valid_;
    // per solver bus, 1 when a droop hvdc line ends there (mis_bus scans the lines only then)
    std::vector<char> hvdc_at_bus_;
    // per generator, the solver buses of the grid plan's ReactiveLimits entries it is part of
    std::vector<std::vector<int>> gen_rl_bus_;
    // zero-filled n_bus scratch of the HVDC / VoltageMonitoring checks (lightsim2grid's
    // checks take full vectors): only the entries a check writes are reset after it
    mutable ls2g::RealVect hvdc_va_scratch_;
    mutable ls2g::CplxVect vm_v_scratch_;
    // freeze_q's writers keep frozen_at in step (mis_bus looks a bus up only where it is set)
    void freeze_set(SlotState& st, int b, double q) const {
        if (b >= 0 && b < n_bus_) {
            if (st.frozen_at.empty()) st.frozen_at.assign(static_cast<size_t>(n_bus_), 0);
            st.frozen_at[static_cast<size_t>(b)] = 1;
        }
        st.freeze_q[b] = q;
    }
    void freeze_erase(SlotState& st, int b) const {
        st.freeze_q.erase(b);
        if (b >= 0 && b < n_bus_ && !st.frozen_at.empty()) st.frozen_at[static_cast<size_t>(b)] = 0;
    }
    void gen_limits(int g, double p, double& lo, double& hi) const {
        const size_t gg = static_cast<size_t>(g);
        const double c = glim_p_[gg];
        if (glim_valid_[gg] && (c == p || (std::isnan(c) && std::isnan(p)))) {
            lo = glim_lo_[gg];
            hi = glim_hi_[gg];
            return;
        }
        ls2g::real_type l, h;
        grid_->gen_limits_at_outer_target(g, static_cast<ls2g::real_type>(p), l, h);
        glim_p_[gg] = p; glim_lo_[gg] = l; glim_hi_[gg] = h; glim_valid_[gg] = 1;
        lo = l;
        hi = h;
    }
    std::vector<int> pin_of_bus_;            // solver bus -> switchable index
    std::vector<LoopDecl> loops_;            // the grid's loops, in order
    ls2g::OuterLoopDriverParams params_;
    // VoltageControl, ledger order (== the layout's plan)
    std::vector<int> vc_bus_, vc_kind_, vc_elem_, vc_group_, vc_plan_held_;
    std::vector<double> vc_slope_, vc_q_plan_, vc_v_set_;
    std::vector<std::vector<int>> vc_at_bus_;      // controllers standing on each bus
    std::vector<int> grp_start_, grp_count_;
    std::vector<int> vc_v_row_;                    // per group, its voltage row
    std::vector<std::vector<int>> vc_share_row_;   // per group, its sharing rows
    std::vector<int> q_row_of_bus_;
    // droop hvdc, ledger order
    std::vector<int> hvdc_id_, hvdc_bus1_, hvdc_bus2_, hvdc_status0_;
    std::vector<double> hvdc_p0_, hvdc_k_, hvdc_lf1_, hvdc_lf2_, hvdc_r_, hvdc_pmax12_, hvdc_pmax21_;
    // BranchControl, the ledger's order (LedgerData::bc_*)
    int n_bc_ = 0, n_line_ = 0, n_trafo_ = 0, n_load_ = 0;
    std::vector<int> bc_trafo_, bc_b1_, bc_b2_, bc_phase_, bc_ratio_, bc_has_col_;
    std::vector<double> bc_sign_, bc_rexp_;
    std::vector<int> bc_of_trafo_;                 // trafo id -> entry (-1)
    // ShuntControl, the ledger's order (LedgerData::sc_*)
    int n_sc_ = 0;
    std::vector<int> sc_bus_, sc_of_bus_;          // solver bus -> controller (-1)
    std::vector<std::vector<int>> sc_shunts_;
    std::set<int> tvc_buses_;                      // the buses the reserved ratio groups regulate
    // DistributedSlack's participants of a row that disconnects no unit (their injections are
    // rewritten from the row's targets in force: init_loops), and their buses
    std::vector<ls2g::slack_redistribution::Participant> units0_;
    std::vector<int> units0_bus_;
    // NROuterDist_*: the Newton shares the slack itself (dist_), on the units of the
    // DistributedSlack loop when the grid has one (ds_reserved_: OuterControls::
    // slack_share); each Layout::slack_bus' index by solver bus, the reference's
    bool dist_ = false;
    bool ds_reserved_ = false;
    std::vector<int> slack_idx_of_bus_;
    int ref_slack_idx_ = -1;
    // plans
    std::shared_ptr<const RlPlan> rl0_;            // the grid's (a row without masks / units off)
    // the device tables' plan: the grid's, or with the buses the rows' moved generators bring
    // (device_prepare_rows: bus_q entries, then theirs, then the monitors)
    std::shared_ptr<const RlPlan> rl_dev_;
    // the moves rl_dev_ was built for (device_prepare_rows: the same ones keep it)
    std::vector<std::vector<std::pair<int, int>>> rl_dev_moves_;
    bool rl_dev_built_ = false;
    // the buses the rows move DistributedSlack participants to, sorted (device_prepare_rows): the
    // device tables' Sbus correction must reach them, the units' shares go there (ds_unit_bus)
    std::vector<int> ds_dev_move_bus_;
    std::vector<std::vector<int>> gens_of_bus0_;   // per solver bus, the grid's generators there
    const RlPlan& rl_dev() const { return rl_dev_ ? *rl_dev_ : *rl0_; }
    ls2g::hvdc_p_check::HvdcPPlan hvdc_plan0_;     // the lines in AC emulation
    ls2g::svc_standby_check::SvcStandbyPlan monitors_;
    std::vector<int> svc_ctrl_;                    // svc id -> its held controller (-1)
    std::vector<int> held_svcs_;                   // the standby SVCs the plan holds, sorted
    // where the grid's own solve left the loops (Ls2gModelOptions::resume_from_n)
    bool resume_ = false;
    ResumePoint point_n_;
    // the rows' own points (keep_row_points: by row, nullptr while the row has not ended
    // stable) and the point each row resumes from
    bool keep_points_ = false;
    std::vector<std::shared_ptr<const ResumePoint>> row_points_;
    std::vector<int> resume_of_row_;                    // [n_rows] index into row_points_, -1: point_n_ / none
    void keep_point(const SlotState& st, const RowView* rv);
    // take_resume_targets on a row that sets no target and disconnects no unit
    std::vector<double> resume0_gen_target_p_, resume0_sto_target_p_;
    FlatMap<int, cplx> resume0_target_delta_;
    // ---- per run ------------------------------------------------------------------
    int n_rows_ = 0;
    std::vector<double> row_gen_p_, row_sto_p_;    // [n_rows * n_gen], [n_rows * n_sto]
    RowOverrides ov_;                              // the rows' masks, units off, set-points
    std::vector<SlotState> slots_;
    // what every load starts from (built by begin()): the default controls, the grid's taps and
    // sections; copy-assigned, so a slot keeps the capacity of its vectors from row to row
    SlotState slot0_;
    void build_slot0();
    // the finished rows' results: finish() fills what it has at hand and copies the final
    // positions below; their (id, value) lists are built by row_result() at the first read,
    // outside the run (results_ and fin_built_ are a cache)
    mutable std::vector<RowResult> results_;
    mutable std::vector<char> fin_built_;          // [n_rows] results_ holds the lists
    std::vector<char>   fin_done_;                 // [n_rows] the row finished
    std::vector<double> fin_sto_p_;                // [n_rows * n_sto] storage targets (NaN: not moved)
    std::vector<int>    fin_hvdc_;                 // [n_rows * n_hvdc] hvdc regimes (2: KEEP)
    std::vector<int>    fin_bc_pos_, fin_bc_rpos_; // [n_rows * n_bc] phase / ratio taps
    std::vector<int>    fin_sc_;                   // [n_rows * n_sc_sh_] section counts
    std::vector<size_t> sc_off_;                   // [n_sc + 1] first shunt of a controller in fin_sc_
    size_t n_sc_sh_ = 0;
    // the pi block of each transformer the row's loops moved off the grid's taps (row_trafo_blocks)
    void row_blocks(size_t row, std::vector<int>& trafo, std::vector<double>& block) const;

    // ---- helpers ------------------------------------------------------------------
    // OuterInjections::gen_target_p_in_force / storage_target_p_in_force
    double gen_in_force(const SlotState& st, int g) const {
        if (!st.gen_target_p.empty() && !std::isnan(st.gen_target_p[static_cast<size_t>(g)]))
            return st.gen_target_p[static_cast<size_t>(g)];
        return view(st).gen_target_p(g);
    }
    double sto_in_force(const SlotState& st, int s) const {
        if (!st.storage_target_p.empty() && !std::isnan(st.storage_target_p[static_cast<size_t>(s)]))
            return st.storage_target_p[static_cast<size_t>(s)];
        return view(st).storage_target_p(s);
    }
    // OuterGridView::gen_target_p_override: the loops' value, else the row's, else NaN
    double gen_override(const SlotState& st, int g) const {
        static_assert(std::is_same<ls2g::real_type, double>::value, "lightsim2grid built in double");
        // OuterGridView::gen_target_p_override, inlined: the loops' value, else the row's, else NaN
        if (g < 0) return NaN;
        const size_t gg = static_cast<size_t>(g);
        if (gg < st.gen_target_p.size() && !std::isnan(st.gen_target_p[gg])) return st.gen_target_p[gg];
        const auto& row = st.rowv.gens.target_p;
        if (gg < row.size() && !std::isnan(row[gg])) return row[gg];
        return NaN;
    }
    // the plan, units and active loops of the row a slot holds (its view set already)
    std::shared_ptr<const RlPlan> build_rl_plan(const ls2g::OuterGridView& v, const std::vector<int>* gone = nullptr,
                                                const std::vector<int>* hvdc_gone = nullptr, bool filtered = false,
                                                const std::vector<std::pair<int, int>>* moves = nullptr) const;
    // bus_q_check::build_bus_q_plan of a row from the grid's plan: the machines the row disconnects or
    // strands out, the buses left without one dropped, the generators it moves on their new bus
    // false: not filterable (a moved generator is a controller of the plan), build_bus_q_plan's then
    bool filtered_bus_q_plan(const ls2g::OuterGridView& v, ls2g::bus_q_check::BusQPlan& out,
                             const std::vector<std::pair<int, int>>* moves = nullptr) const;
    std::shared_ptr<const RlPlan> rl_plan_masked(const std::vector<int>& masked) const;
    void take_resume_targets(SlotState& st) const;
    // DistributedSlack, at initialize: the row's lost active power shared on its units
    void share_lost_p(SlotState& st, const LoopDecl& L) const;
    // ... around a distributed-slack Newton: unit k's bound (DistributedSlackLoop::_bound),
    // the units a resumed solve parked parked again (_keep_parked), the units the Newton
    // shares on and their total weight (_set_newton_weights + SlackShareControl::set_units:
    // weight > 0, on a reserved bus), the share moved into their targets (_fold_slack_share)
    double ds_bound(const SlotState& st, size_t k, bool high) const;
    void ds_keep_parked(SlotState& st) const;
    double ds_share(const SlotState& st, std::vector<size_t>* units) const;
    void ds_fold(SlotState& st, double sa) const;
    // `gen_w` / `sto_w` (nullable, NaN: the grid's): the weights the row gives its units
    void collect_units(const ls2g::OuterGridView& v, std::vector<ls2g::slack_redistribution::Participant>& units,
                       std::vector<int>& bus, const std::vector<double>* gen_w = nullptr,
                       const std::vector<double>* sto_w = nullptr) const;
    void set_active(SlotState& st) const;
    void add_injection(SlotState& st, int bus, cplx ds) const {
        if (ds == cplx(0., 0.)) return;
        st.target_delta[bus] += ds;
    }
    double move_gen(SlotState& st, int g, int bus, double mw) const;
    double move_storage(SlotState& st, int s, int bus, double mw) const;
    void record(SlotState& st, const std::string& loop, const std::string& action, bool taken,
                int element, double value, double limit) const;
    // whether a participant stands off the slack bus (DistributedSlackLoop::_off_slack_bus)
    bool ds_off_slack_bus(const SlotState& st) const {
        return std::any_of(st.ds_unit_bus.begin(), st.ds_unit_bus.end(), [&](int b) { return b != slack_bus_; });
    }
    void reset_vm(SlotState& st, int bus, double vm) const {
        st.vm_reset_bus.push_back(bus);
        st.vm_reset_vm.push_back(vm);
    }
    // the bus' residual as lightsim2grid's NR publishes it (OuterLoopAlgo::_solve's
    // correction included): V.conj(Ybus.V) - Sbus + the droop flows - the controllers' Q
    cplx mis_bus(const SlotState& st, const RowView& rv, const SlotControls& c, int b) const;
    void materialize(SlotState& st) const;

    // the loops
    void init_loops(SlotState& st) const;
    // OuterLoopAlgo::_take_targets and the loops' initialize with ctx.resume
    void resume_loops(SlotState& st) const;
    int check(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const;
    int check_ds(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    int check_ds_in_newton(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    int check_hvdc(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    int check_vm(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    // whether the slot's row trips controller c (RowOverrides::vc_tripped, sorted)
    bool row_trips_ctrl(const SlotState& st, int c) const {
        if (st.row < 0 || static_cast<size_t>(st.row) >= ov_.vc_tripped.size()) return false;
        const std::vector<int>& t = ov_.vc_tripped[static_cast<size_t>(st.row)];
        return std::binary_search(t.begin(), t.end(), c);
    }
    // ... and the standby SVC svc_id (a held SVC controller of the plan)
    bool row_trips_svc(const SlotState& st, int svc_id) const {
        const int c = (svc_id >= 0 && static_cast<size_t>(svc_id) < svc_ctrl_.size()) ? svc_ctrl_[static_cast<size_t>(svc_id)] : -1;
        return c >= 0 && row_trips_ctrl(st, c);
    }
    int check_rl(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    int check_pc(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const;
    int check_tvc(SlotState& st, const LoopDecl& L, const RowView& rv) const;
    int check_svc(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const;
    // ShuntControl::set_sections (the counts clamped, the susceptance they give)
    void sc_set_sections(SlotState& st, int e, const std::vector<int>& counts) const;
    // ShuntVoltageControlLoop::_dispatch
    std::vector<int> svc_dispatch(const std::vector<int>& ids, double b) const;
    // TransformerVoltageControlLoop's helpers
    void tvc_set_on(SlotState& st, int t, bool on) const;
    int  tvc_closest_tap(const SlotState& st, int t, double value) const;
    std::vector<TvcBranch> tvc_closed_branches(const ls2g::OuterGridView& v) const;
    double tvc_limit(const SlotState& st, const LoopDecl& L) const;
    bool tvc_voltage_control_off(const SlotState& st, int bus) const;
    void tvc_freeze(SlotState& st, const RowView& rv, double limit, const std::vector<TvcBranch>& closed) const;
    void tvc_release(SlotState& st) const;
    void tvc_fix_controls(SlotState& st, const std::vector<TvcBranch>& closed) const;
    int  bc_entry(int trafo) const {
        return (trafo >= 0 && trafo < n_trafo_) ? bc_of_trafo_[static_cast<size_t>(trafo)] : -1;
    }
    // BranchControl helpers: the block at shift a and ratio rho (BranchControl::_block_at),
    // a re-tap's data (BranchControl::_retap), the current through a side (::current)
    void bc_block_at(const SlotState& st, int e, double a, double rho, cplx blk[4]) const;
    void bc_retap(SlotState& st, int e, int rpos, int pos) const;
    void bc_current(const SlotState& st, const RowView& rv, int e, int side, double& i_pu, double& di_da) const;
    // ReactiveLimits helpers
    double rl_bus_q(const SlotState& st, const RowView& rv, const RlBus& bus) const;
    void rl_limits(const SlotState& st, const RowView* rv, const RlBus& bus, double& qmin, double& qmax) const;
    // the row's generators on that bus are where the base targets have them
    bool rl_at_base(const SlotState& st, const RlBus& bus) const {
        return bus.has_base && st.lim_base && bus.bus_solver >= 0 && bus.bus_solver < n_bus_ &&
               (st.lim_dirty.empty() || !st.lim_dirty[static_cast<size_t>(bus.bus_solver)]);
    }
    double rl_controller_limit(const SlotState& st, const RowView& rv, int c, bool max) const;
    std::vector<char> rl_groups_holding(const SlotState& st) const;
    void rl_freeze(SlotState& st, const RowView& rv, size_t k, double q_mvar, int state) const;
    // ReactiveLimitsLoop::_gen_limit_slope: the slope of generator g's limit (max or min) at
    // the target P in force, MVar per MW (0 without a curve or a target a row or loop gave it)
    double rl_gen_limit_slope(const SlotState& st, int g, bool max) const {
        ls2g::real_type dlo = 0., dhi = 0.;
        grid_->gen_limit_slopes_at_outer_target(g, gen_override(st, g), dlo, dhi);
        return max ? dhi : dlo;
    }
    // the adjoint's view of the row's last solve (RowResult::pq_buses ... gen_p_slope_*)
    void adjoint_info(const SlotState& st, RowResult& r) const;
    void rl_release(SlotState& st, size_t k) const;
    // the driver
    // `recheck`: loop i acted and the row was re-solved: runOuterLoop's do-while
    // checks it again before the for loop's stop tests apply
    Next drive(SlotState& st, const RowView& rv, int& row_status, int inner_status, bool recheck = false);
    Next finish(SlotState& st, int& row_status, int inner_status, const RowView* rv = nullptr);
};

// =============================================================================
// Construction: the loops, the plans, the layout
// =============================================================================
Ls2gOuterModel::Ls2gOuterModel(const ls2g::LSGrid& grid, const LedgerData& ledger,
                               const Ls2gModelOptions& opt)
    : grid_(std::make_shared<ls2g::LSGrid>(grid)), opt_(opt)
{
    const ls2g::LSGrid& g = grid;   // solved: its layout is the one the session solves in
    sn_ = static_cast<double>(g.get_sn_mva());
    n_bus_ = ledger.n_bus;
    n_gen_ = g.get_generators().nb();
    glim_p_.assign(static_cast<size_t>(n_gen_), NaN);
    glim_lo_.assign(static_cast<size_t>(n_gen_), 0.);
    glim_hi_.assign(static_cast<size_t>(n_gen_), 0.);
    glim_valid_.assign(static_cast<size_t>(n_gen_), 0);
    hvdc_va_scratch_ = ls2g::RealVect::Zero(n_bus_);
    vm_v_scratch_ = ls2g::CplxVect::Zero(n_bus_);
    n_sto_ = g.get_storages().nb();
    params_ = g.get_outer_loop_driver_params();
    {
        const auto slack = g.get_slack_ids_solver();
        if (slack.size() == 0) throw std::runtime_error("outer model: the grid has no slack bus");
        slack_bus_ = static_cast<int>(slack(0));
    }

    // ---- the loops ------------------------------------------------------------------
    std::vector<LoopDecl> loops;
    for (const auto& lp : g.get_outer_loops()) {
        LoopDecl d;
        d.name = lp->name();
        if (auto* ds = dynamic_cast<const ls2g::DistributedSlackLoop*>(lp.get())) {
            d.kind = Kind::DistributedSlack;
            const auto& p = ds->params();
            d.ds_mismatch_mw = p.slack_bus_p_max_mismatch_mw;
            d.ds_fail = p.fail_on_residue;
            d.ds_eps_mw = p.p_residue_eps_mw;
            d.ds_moved_fraction = p.moved_fraction;
            d.ds_run_without_participant = p.run_without_participant;
        } else if (dynamic_cast<const ls2g::HvdcAcEmulationLimitsLoop*>(lp.get())) {
            d.kind = Kind::HvdcLimits;
        } else if (dynamic_cast<const ls2g::VoltageMonitoringLoop*>(lp.get())) {
            d.kind = Kind::VoltageMonitoring;
        } else if (auto* rl = dynamic_cast<const ls2g::ReactiveLimitsLoop*>(lp.get())) {
            d.kind = Kind::ReactiveLimits;
            const auto& p = rl->params();
            d.rl_max_switch = p.max_pq_pv_switch;
            d.rl_max_mismatch = p.max_reactive_power_mismatch;
            d.rl_base_mva = p.mismatch_base_mva;
            // -1 / NaN: the driver's (ReactiveLimitsLoop::_robust / _realistic_band)
            d.rl_robust = p.robust_mode < 0 ? params_.voltage_remote_control_robust_mode : p.robust_mode != 0;
            d.rl_vmin = std::isnan(p.min_realistic_voltage) ? params_.min_realistic_voltage : p.min_realistic_voltage;
            d.rl_vmax = std::isnan(p.max_realistic_voltage) ? params_.max_realistic_voltage : p.max_realistic_voltage;
            d.rl_margin = p.realistic_voltage_margin;
            d.rl_restart = p.robust_restart_vm_pu;
        } else if (dynamic_cast<const ls2g::PhaseControlLoop*>(lp.get())) {
            d.kind = Kind::PhaseControl;
        } else if (dynamic_cast<const ls2g::ShuntVoltageControlLoop*>(lp.get())) {
            d.kind = Kind::ShuntVoltageControl;
        } else if (auto* tv = dynamic_cast<const ls2g::TransformerVoltageControlLoop*>(lp.get())) {
            d.kind = Kind::TransformerVoltageControl;
            d.tvc = std::make_shared<const ls2g::TransformerVoltageControlLoop>(tv->params());
        } else {
            throw std::runtime_error("gpusim2grid's outer loops: the loop '" + d.name +
                                     "' is not supported (DistributedSlack, AcHvdcAcEmulationLimits, "
                                     "VoltageMonitoring, ReactiveLimits, PhaseControl, TransformerVoltageControl and "
                                     "ShuntVoltageControl are)");
        }
        loops.push_back(d);
    }

    sbl_ = grid.get_ac_layout();
    // the grid seen as a row that changes nothing, on the model's own copy
    SlotState grid_row;
    grid_row.rowv.layout = &sbl_;
    const ls2g::OuterGridView v = view(grid_row);
    const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();

    // where the grid's last solve left its loops
    resume_ = opt.resume_from_n;
    if (resume_) {
#ifndef LS2G_HAS_OUTER_LOOP_STATE_ACCESSORS
        throw std::runtime_error("resuming the rows from the grid's own solve needs a lightsim2grid "
                                 "exposing its loops' state (LS2G_HAS_OUTER_LOOP_STATE_ACCESSORS); "
                                 "use init_from_n_powerflow=False");
#else
        const std::shared_ptr<const ls2g::OuterResumePoint> point = g.get_algo().get_outer_resume_point();
        if (!point)
            throw std::runtime_error("resuming the rows from the grid's own solve: the grid's last solve "
                                     "did not end with its loops stable (solve it with its NROuter_* "
                                     "algorithm first)");
        ResumePoint& pn = point_n_;
        pn.gen_p.assign(point->gen_target_p.begin(), point->gen_target_p.end());
        pn.sto_p.assign(point->storage_target_p.begin(), point->storage_target_p.end());
        pn.V = point->V;
        for (const auto& lp : point->loops) {
            if (auto* h = dynamic_cast<const ls2g::HvdcAcEmulationLimitsLoop*>(lp.get())) {
                for (const auto& line : h->regimes().lines) pn.hvdc[line.hvdc_id] = line.frozen_dir;
            } else if (auto* m = dynamic_cast<const ls2g::VoltageMonitoringLoop*>(lp.get())) {
                for (const auto& on : m->switched_on()) pn.svc_on.emplace_back(on.first, on.second);
            } else if (auto* r = dynamic_cast<const ls2g::ReactiveLimitsLoop*>(lp.get())) {
                for (const auto& fb : r->frozen_buses())
                    pn.frozen.emplace(fb.bus_solver, std::make_pair(fb.state, fb.realistic));  // the first per bus
            } else if (auto* ds = dynamic_cast<const ls2g::DistributedSlackLoop*>(lp.get())) {
                pn.parked = ds->parked_units();   // around a distributed-slack Newton only
            }
        }
        // the tap positions the solve settled on (OuterLoopAlgo::_apply_carried_positions)
        pn.phase_tap.assign(point->phase_tap.begin(), point->phase_tap.end());
        pn.ratio_tap.assign(point->ratio_tap.begin(), point->ratio_tap.end());
        pn.sections.assign(point->sections.begin(), point->sections.end());
#endif
    }

    // ---- VoltageControl, as the ledger has it --------------------------------------------
    const int nc = ledger.vc_n_controllers();
    if (ctrl.n_controllers() != nc || ctrl.n_groups() != ledger.vc_n_groups())
        throw std::runtime_error("outer model: the grid's voltage-control plan is not the ledger's");
    vc_bus_ = ledger.vc_bus; vc_kind_ = ledger.vc_kind; vc_elem_ = ledger.vc_elem_id;
    vc_group_ = ledger.vc_group; vc_slope_ = ledger.vc_slope; vc_v_set_ = ledger.vc_v_set;
    grp_start_ = ledger.vc_grp_start; grp_count_ = ledger.vc_grp_count;
    vc_v_row_ = ledger.vc_v_rows; vc_share_row_ = ledger.vc_share_rows;
    q_row_of_bus_ = ledger.q_row_of_bus;
    vc_plan_held_.assign(static_cast<size_t>(nc), 0);
    vc_q_plan_.assign(static_cast<size_t>(nc), 0.);
    for (int c = 0; c < nc; ++c) {
        vc_plan_held_[static_cast<size_t>(c)] = ctrl.is_held(c) ? 1 : 0;
        if (static_cast<int>(ledger.vc_q_held.size()) == nc) vc_q_plan_[static_cast<size_t>(c)] = ledger.vc_q_held[static_cast<size_t>(c)];
    }
    vc_at_bus_.assign(static_cast<size_t>(n_bus_), {});
    for (int c = 0; c < nc; ++c) vc_at_bus_[static_cast<size_t>(vc_bus_[static_cast<size_t>(c)])].push_back(c);
    svc_ctrl_.assign(static_cast<size_t>(g.get_svcs().nb()), -1);
    for (int c = 0; c < nc; ++c)
        if (vc_kind_[static_cast<size_t>(c)] == ls2g::VoltageControlSolverData::SVC && vc_plan_held_[static_cast<size_t>(c)])
            svc_ctrl_[static_cast<size_t>(vc_elem_[static_cast<size_t>(c)])] = c;

    // ---- BranchControl, as the ledger has it ---------------------------------------------
    n_bc_ = ledger.n_bc();
    n_line_ = g.get_lines().nb();
    n_trafo_ = g.get_trafos().nb();
    n_load_ = g.get_loads().nb();
    bc_trafo_ = ledger.bc_trafo; bc_b1_ = ledger.bc_bus1; bc_b2_ = ledger.bc_bus2;
    bc_phase_ = ledger.bc_phase; bc_ratio_ = ledger.bc_ratio;
    bc_sign_ = ledger.bc_sign; bc_rexp_ = ledger.bc_rexp;
    bc_has_col_.assign(static_cast<size_t>(n_bc_), 0);
    for (int e = 0; e < n_bc_; ++e) bc_has_col_[static_cast<size_t>(e)] = ledger.bc_shift_col[static_cast<size_t>(e)] >= 0;
    bc_of_trafo_.assign(static_cast<size_t>(n_trafo_), -1);
    for (int e = 0; e < n_bc_; ++e) bc_of_trafo_[static_cast<size_t>(bc_trafo_[static_cast<size_t>(e)])] = e;

    // ---- ShuntControl, as the ledger has it ----------------------------------------------
    n_sc_ = ledger.n_sc();
    sc_bus_ = ledger.sc_bus;
    sc_shunts_ = ledger.sc_shunts;
    sc_of_bus_.assign(static_cast<size_t>(n_bus_), -1);
    for (int e = 0; e < n_sc_; ++e) sc_of_bus_[static_cast<size_t>(sc_bus_[static_cast<size_t>(e)])] = e;
    tvc_buses_.insert(ledger.bc_grp_bus.begin(), ledger.bc_grp_bus.end());

    // ---- droop hvdc, as the ledger has it ------------------------------------------------
    hvdc_bus1_ = ledger.hvdc_bus1; hvdc_bus2_ = ledger.hvdc_bus2; hvdc_status0_ = ledger.hvdc_status;
    hvdc_p0_ = ledger.hvdc_p0; hvdc_k_ = ledger.hvdc_k; hvdc_lf1_ = ledger.hvdc_lf1; hvdc_lf2_ = ledger.hvdc_lf2;
    hvdc_r_ = ledger.hvdc_r; hvdc_pmax12_ = ledger.hvdc_pmax12; hvdc_pmax21_ = ledger.hvdc_pmax21;
    hvdc_id_ = ledger.hvdc_id;
    if (hvdc_id_.empty())
        for (int e = 0; e < ledger.n_hvdc(); ++e) hvdc_id_.push_back(e);

    // ---- the plans (each loop's is_needed / initialize, on the grid's own view) ----------
    // the switchable buses first: a local controller bus of the plan must be one
    layout_.pin_bus = ledger.switchable_vm_buses;
    pin_of_bus_.assign(static_cast<size_t>(n_bus_), -1);
    for (size_t k = 0; k < layout_.pin_bus.size(); ++k) pin_of_bus_[static_cast<size_t>(layout_.pin_bus[k])] = static_cast<int>(k);
    // ReactiveLimits: bus_q_check's plan, plus one entry per voltage monitor
    rl0_ = build_rl_plan(v);
    // HvdcAcEmulationLimits: the lines in AC emulation
    {
        ls2g::hvdc_p_check::HvdcPPlan plan;
        ls2g::hvdc_p_check::build_hvdc_p_plan(g, v.id_me_to_solver(), plan);
        for (const auto& line : plan.lines) if (line.frozen_dir == 0) hvdc_plan0_.lines.push_back(line);
    }
    // VoltageMonitoring: the standby SVCs the plan holds
    for (int c = 0; c < ctrl.n_controllers(); ++c)
        if (ctrl.kind(c) == ls2g::VoltageControlSolverData::SVC && ctrl.is_held(c)) held_svcs_.push_back(ctrl.elem_id(c));
    std::sort(held_svcs_.begin(), held_svcs_.end());
    {
        ls2g::svc_standby_check::SvcStandbyPlan plan;
        ls2g::svc_standby_check::build_svc_standby_plan(g, v.id_me_to_solver(), plan);
        for (const auto& e : plan.svcs)
            if (std::binary_search(held_svcs_.begin(), held_svcs_.end(), e.svc_id)) monitors_.svcs.push_back(e);
    }
    loops_ = loops;
    // the robust mode's last loop able to fix an unrealistic state, and is_needed: per row
    // (set_active), since a row disconnecting units changes the participants and the plan
    // DistributedSlack's participants: collected once here, they depend on a row only
    // through the units it disconnects (collecting them was most of a slot's load)
    collect_units(v, units0_, units0_bus_);

    // per solver bus, the generators the grid places there (id order): build_rl_plan's target Q sums
    // of a row moving no generator
    {
        SlotState plain_g;
        plain_g.rowv.layout = &sbl_;
        const ls2g::OuterGridView vg = view(plain_g);
        gens_of_bus0_.assign(static_cast<size_t>(n_bus_), {});
        for (int g = 0; g < n_gen_; ++g) {
            const int b = vg.gen_bus_solver(g);
            if (b >= 0 && b < n_bus_) gens_of_bus0_[static_cast<size_t>(b)].push_back(g);
        }
    }

    // ---- a distributed-slack Newton (NROuterDist_*) ----------------------------------------
    // its slack_absorbed column has an entry on the reference and on every bus its
    // DistributedSlack loop reserved; the loop shares in the Newton when the grid has one
    dist_ = ledger.outer_loops && ledger.has_multislack();
    if (dist_) {
        layout_.slack_bus = ledger.slack_participant_buses;
        slack_idx_of_bus_.assign(static_cast<size_t>(n_bus_), -1);
        for (size_t k = 0; k < layout_.slack_bus.size(); ++k)
            slack_idx_of_bus_[static_cast<size_t>(layout_.slack_bus[k])] = static_cast<int>(k);
        ref_slack_idx_ = slack_idx_of_bus_[static_cast<size_t>(slack_bus_)];
        if (ref_slack_idx_ < 0)
            throw std::runtime_error("outer model: the reference bus has no entry in the slack_absorbed column");
        for (const LoopDecl& d : loops_) ds_reserved_ = ds_reserved_ || d.kind == Kind::DistributedSlack;
    }

    // ---- the layout ------------------------------------------------------------------------
    layout_.n_ctrl = nc;
    layout_.n_grp  = ledger.vc_n_groups();
    layout_.n_hvdc = ledger.n_hvdc();
    layout_.n_bc   = n_bc_;
    layout_.n_sc   = n_sc_;
    std::vector<int> watch;
    watch.push_back(slack_bus_);
    for (const auto& e : rl0_->plan->buses) watch.push_back(e.bus_solver);
    for (const RlBus& b : rl0_->buses) watch.push_back(b.reg_bus_solver);
    for (int c = 0; c < nc; ++c) watch.push_back(vc_bus_[static_cast<size_t>(c)]);
    for (int e = 0; e < layout_.n_hvdc; ++e) { watch.push_back(hvdc_bus1_[static_cast<size_t>(e)]); watch.push_back(hvdc_bus2_[static_cast<size_t>(e)]); }
    for (const auto& line : hvdc_plan0_.lines) { watch.push_back(line.bus1_solver); watch.push_back(line.bus2_solver); }
    for (const auto& m : monitors_.svcs) { watch.push_back(m.reg_bus_solver); watch.push_back(m.svc_bus_solver); }
    for (int e = 0; e < n_bc_; ++e) { watch.push_back(bc_b1_[static_cast<size_t>(e)]); watch.push_back(bc_b2_[static_cast<size_t>(e)]); }
    // the buses the transformers regulate (every group, a row's are among them)
    for (const LoopDecl& d : loops_)
        if (d.kind == Kind::TransformerVoltageControl)
            for (const auto& gr : d.tvc->groups(v)) watch.push_back(gr.bus_solver);
    watch.erase(std::remove_if(watch.begin(), watch.end(), [&](int b) { return b < 0 || b >= n_bus_; }), watch.end());
    std::sort(watch.begin(), watch.end());
    watch.erase(std::unique(watch.begin(), watch.end()), watch.end());
    layout_.watch_bus = watch;
    widx_.assign(static_cast<size_t>(n_bus_), -1);
    for (size_t k = 0; k < watch.size(); ++k) widx_[static_cast<size_t>(watch[k])] = static_cast<int>(k);
    // the unrealistic-voltage check (is_state_unrealistic): every bus with a Vm unknown
    {
        const auto vn_kv = g.get_bus_vn_kv();
        const auto& to_me = v.id_solver_to_me();
        for (int b = 0; b < n_bus_; ++b) {
            if (ledger.vm_col_of_bus[static_cast<size_t>(b)] < 0) continue;
            const int me = to_me[b].cast_int();
            if (me < 0 || vn_kv(me) < params_.min_nominal_voltage_realistic_check) continue;
            layout_.realistic_bus.push_back(b);
        }
    }
    layout_.min_realistic_v = params_.min_realistic_voltage;
    layout_.max_realistic_v = params_.max_realistic_voltage;
    hvdc_at_bus_.assign(static_cast<size_t>(n_bus_), 0);
    for (int e = 0; e < layout_.n_hvdc; ++e)
        for (int b : {hvdc_bus1_[static_cast<size_t>(e)], hvdc_bus2_[static_cast<size_t>(e)]})
            if (b >= 0 && b < n_bus_) hvdc_at_bus_[static_cast<size_t>(b)] = 1;
    // the resumed targets of a row that sets none and disconnects no unit: the same for all
    if (resume_) {
        SlotState plain;
        plain.rowv.layout = &sbl_;
        plain.rp = &point_n_;
        take_resume_targets(plain);
        resume0_gen_target_p_ = plain.gen_target_p;
        resume0_sto_target_p_ = plain.storage_target_p;
        resume0_target_delta_ = plain.target_delta;
    }
    // ReactiveLimits: the generator part of every bus' limits at the base targets (rl_limits on a
    // plain row, the same sums in the same order), and the buses each generator reaches
    {
        auto rl = std::make_shared<RlPlan>(*rl0_);
        SlotState plain;
        plain.rowv.layout = &sbl_;
        plain.rl = rl0_;
        if (resume_) plain.gen_target_p = resume0_gen_target_p_;
        gen_rl_bus_.assign(static_cast<size_t>(n_gen_), {});
        for (size_t k = 0; k < rl->buses.size(); ++k) {
            RlBus& bus = rl->buses[k];
            double mn, mx;
            rl_limits(plain, nullptr, bus, mn, mx);
            bus.base_min_q = mn;
            bus.base_max_q = mx;
            bus.has_base = true;
            const auto& entry = rl->plan->buses[static_cast<size_t>(bus.entry)];
            bus.has_storage = !entry.storage_ids.empty();
            double target_p = 0.;
            for (int gid : entry.gen_ids) target_p += gen_in_force(plain, gid);
            rl->state0[k].target_p = target_p;
            for (int gid : entry.gen_ids)
                if (gid >= 0 && gid < n_gen_ && entry.bus_solver >= 0 && entry.bus_solver < n_bus_) gen_rl_bus_[static_cast<size_t>(gid)].push_back(entry.bus_solver);
        }
        rl0_ = rl;
    }
}

void Ls2gOuterModel::set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p)
{
    if (!gen_p.empty() && gen_p.size() != static_cast<size_t>(n_rows) * n_gen_)
        throw std::runtime_error("outer model: set_row_targets: gen_p must be n_rows x n_gen");
    if (!storage_p.empty() && storage_p.size() != static_cast<size_t>(n_rows) * n_sto_)
        throw std::runtime_error("outer model: set_row_targets: storage_p must be n_rows x n_storage");
    row_gen_p_ = std::move(gen_p);
    row_sto_p_ = std::move(storage_p);
}

void Ls2gOuterModel::begin(int n_slots, int n_rows)
{
    if (!row_gen_p_.empty() && row_gen_p_.size() != static_cast<size_t>(n_rows) * n_gen_)
        throw std::runtime_error("outer model: the rows' generator targets do not match the batch's rows");
    if (!row_sto_p_.empty() && row_sto_p_.size() != static_cast<size_t>(n_rows) * n_sto_)
        throw std::runtime_error("outer model: the rows' storage targets do not match the batch's rows");
    auto check = [&](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("outer model: the rows' ") + what +
                                          " do not match the batch's rows");
    };
    check(ov_.masked.empty() || ov_.masked.size() == static_cast<size_t>(n_rows), "masked buses");
    check(ov_.gen_off.empty() || ov_.gen_off.size() == static_cast<size_t>(n_rows) * n_gen_, "disconnected generators");
    check(ov_.gen_vm.empty() || ov_.gen_vm.size() == static_cast<size_t>(n_rows) * n_gen_, "generator set-points");
    check(ov_.pq_buses.empty() || ov_.pq_buses.size() == static_cast<size_t>(n_rows), "released buses");
    check(ov_.vc_v_set.empty() || ov_.vc_v_set.size() == static_cast<size_t>(n_rows) * layout_.n_grp, "group set-points");
    check(ov_.branch_off.empty() || ov_.branch_off.size() == static_cast<size_t>(n_rows), "opened branches");
    check(ov_.storage_off.empty() || ov_.storage_off.size() == static_cast<size_t>(n_rows) * n_sto_, "disconnected storage units");
    check(ov_.load_off.empty() || ov_.load_off.size() == static_cast<size_t>(n_rows) * n_load_, "disconnected loads");
    check(ov_.lost_p_mw.empty() || ov_.lost_p_mw.size() == static_cast<size_t>(n_rows), "lost active power");
    check(resume_of_row_.empty() || resume_of_row_.size() == static_cast<size_t>(n_rows), "resume points");
    n_rows_ = n_rows;
    slots_.assign(static_cast<size_t>(n_slots), SlotState());
    const size_t nr = static_cast<size_t>(n_rows);
    if (keep_points_) row_points_.assign(nr, nullptr);
    results_.assign(nr, RowResult());
    fin_built_.assign(nr, 0);
    fin_done_.assign(nr, 0);
    sc_off_.assign(static_cast<size_t>(n_sc_) + 1, 0);
    for (int e = 0; e < n_sc_; ++e)
        sc_off_[static_cast<size_t>(e) + 1] = sc_off_[static_cast<size_t>(e)] + sc_shunts_[static_cast<size_t>(e)].size();
    n_sc_sh_ = sc_off_.back();
    fin_sto_p_.assign(nr * static_cast<size_t>(n_sto_), NaN);
    fin_hvdc_.assign(nr * static_cast<size_t>(layout_.n_hvdc), 2);
    fin_bc_pos_.assign(nr * static_cast<size_t>(n_bc_), 0);
    fin_bc_rpos_.assign(nr * static_cast<size_t>(n_bc_), 0);
    fin_sc_.assign(nr * n_sc_sh_, 0);
    build_slot0();
}

// The slot every load starts from: the default control state (OuterControls::reset_states),
// BranchControl at the grid's taps (BranchControl::_reset) with every transformer the grid has
// closed, ShuntControl at the grid's sections (ShuntControl::_reset). A load then applies the
// row's own changes.
void Ls2gOuterModel::build_slot0()
{
    SlotState& st = slot0_;
    st = SlotState();
    st.pq.assign(layout_.pin_bus.size(), 0);
    // a spare bus (set_spare_buses) is PQ unless a row holds it PV (a generator moved onto it)
    const int n_base = base_buses_.saved ? base_buses_.n_bus : n_bus_;
    for (size_t k = 0; k < layout_.pin_bus.size(); ++k)
        if (layout_.pin_bus[k] >= n_base) st.pq[k] = 1;
    st.held_q.assign(static_cast<size_t>(layout_.n_ctrl), NaN);
    st.svc_on.assign(static_cast<size_t>(layout_.n_ctrl), NaN);
    st.hvdc_regime.assign(static_cast<size_t>(layout_.n_hvdc), 2);
    st.rowv.layout = &sbl_;
    {
        const size_t nb = static_cast<size_t>(n_bc_);
        const auto& trafos = grid_->get_trafos();
        const auto& ptc = trafos.get_tap_changers(true);
        const auto& rtc = trafos.get_tap_changers(false);
        const ls2g::OuterGridView v = view(st);
        st.bc_out.assign(nb, 0);
        st.bc_pos.assign(nb, 0); st.bc_rpos.assign(nb, 0);
        st.bc_a_tap.assign(nb, 0.); st.bc_rho_tap.assign(nb, 1.);
        st.bc_tap.assign(4 * nb, cplx(0., 0.));
        st.bc_moved.assign(nb, 0); st.bc_req_tap.assign(nb, 0);
        st.bc_req_rtap.assign(nb, std::numeric_limits<int>::min());
        st.bc_ctl.assign(nb, -1); st.bc_rctl.assign(nb, -1);
        for (size_t e = 0; e < nb; ++e) {
            const int t = bc_trafo_[e];
            int b1 = -1, b2 = -1;
            st.bc_out[e] = !(v.trafo_closed(t, b1, b2) && b1 == bc_b1_[e] && b2 == bc_b2_[e]);
            const ls2g::TrafoInfo info = trafos[t];
            st.bc_pos[e] = ptc.has(t) ? ptc.position(t) : 0;
            st.bc_rpos[e] = rtc.has(t) ? rtc.position(t) : 0;
            st.bc_a_tap[e] = info.shift_rad;
            st.bc_rho_tap[e] = (ptc.has(t) || rtc.has(t)) ? trafos.ratio_at(t, st.bc_rpos[e], st.bc_pos[e]) : info.ratio;
            const auto tap = trafos.pi_block_at(t, st.bc_rpos[e], st.bc_pos[e], info.shift_rad);
            for (int k = 0; k < 4; ++k) st.bc_tap[4 * e + static_cast<size_t>(k)] = cplx(tap[static_cast<size_t>(k)].real(), tap[static_cast<size_t>(k)].imag());
        }
    }
    {
        const auto& shunts = grid_->get_shunts();
        st.sc_counts.assign(static_cast<size_t>(n_sc_), {});
        for (int e = 0; e < n_sc_; ++e)
            for (int sh : sc_shunts_[static_cast<size_t>(e)])
                st.sc_counts[static_cast<size_t>(e)].push_back(shunts.get_section_count(sh));
        st.sc_pending.assign(static_cast<size_t>(n_sc_), 0);
        st.sc_b_new.assign(static_cast<size_t>(n_sc_), 0.);
        st.sc_ctl.assign(static_cast<size_t>(n_sc_), -1);
    }
}

// =============================================================================
// A row enters: default control state, the loops' initialize (not resumed)
// =============================================================================
void Ls2gOuterModel::load(int slot, int row)
{
    ProfScope ps_(prof_, row >= 0 ? P_LOAD : P_LOAD_PHANTOM);
    SlotState& st = slots_[static_cast<size_t>(slot)];
    // the default controls, the grid's taps and sections (build_slot0)
    { ProfScope p(prof_, P_LOAD_RESET); st = slot0_; }
    st.row = row;
    if (row >= 0) {
        int k = static_cast<size_t>(row) < resume_of_row_.size() ? resume_of_row_[static_cast<size_t>(row)] : -1;
        // a point of this run: only once that row ended stable (else the fallback)
        if (k >= 0 && (static_cast<size_t>(k) >= row_points_.size() || !row_points_[static_cast<size_t>(k)])) k = -1;
        st.rp = k >= 0 ? row_points_[static_cast<size_t>(k)].get() : (resume_ ? &point_n_ : nullptr);
    }
    // the row as its loops see it (BaseBatchSweep::_fill_outer_row_view)
    bool changes_units = false;
    const std::vector<int>* vc_gone = nullptr;
    const std::vector<int>* hvdc_gone = nullptr;
    const std::vector<std::pair<int, int>>* gen_moves = nullptr;   // the generators it moves (grid id, bus)
    if (row >= 0) {
        const size_t r = static_cast<size_t>(row);
        auto row_of = [&](const std::vector<double>& m, int n, std::vector<ls2g::real_type>& out) {
            if (m.empty() || n <= 0) return;
            out.assign(m.begin() + static_cast<ptrdiff_t>(r * n), m.begin() + static_cast<ptrdiff_t>((r + 1) * n));
        };
        row_of(row_gen_p_, n_gen_, st.rowv.gens.target_p);
        row_of(row_sto_p_, n_sto_, st.rowv.storages.target_p);
        row_of(ov_.gen_vm, n_gen_, st.rowv.gens.target_vm);
        if (!ov_.gen_off.empty()) {
            st.rowv.gens.off.assign(ov_.gen_off.begin() + static_cast<ptrdiff_t>(r * n_gen_),
                                    ov_.gen_off.begin() + static_cast<ptrdiff_t>((r + 1) * n_gen_));
            for (char o : st.rowv.gens.off) changes_units = changes_units || o != 0;
        }
        if (!ov_.storage_off.empty()) {
            st.rowv.storages.off.assign(ov_.storage_off.begin() + static_cast<ptrdiff_t>(r * n_sto_),
                                        ov_.storage_off.begin() + static_cast<ptrdiff_t>((r + 1) * n_sto_));
            for (char o : st.rowv.storages.off) changes_units = changes_units || o != 0;
        }
        if (!ov_.load_off.empty()) {
            st.rowv.loads.off.assign(ov_.load_off.begin() + static_cast<ptrdiff_t>(r * n_load_),
                                     ov_.load_off.begin() + static_cast<ptrdiff_t>((r + 1) * n_load_));
        }
        if (r < ov_.lost_p_mw.size()) st.rowv.lost_p_mw = ov_.lost_p_mw[r];
        // the slack weights the row gives its units (set_gen_slack_weights): its own DistributedSlack
        // units, and it starts from the grid's targets -- the "n" solve's loops shared its imbalance
        // with the grid's weights, which the row's would then only complete (lightsim2grid's
        // _set_outer_row_resume)
        st.gen_slack_w.clear();
        st.sto_slack_w.clear();
        bool row_weights = false;
        const auto take_w = [&](const std::vector<double>& m, int n, std::vector<double>& out) {
            if (m.empty() || n <= 0) return;
            out.assign(m.begin() + static_cast<ptrdiff_t>(r * n), m.begin() + static_cast<ptrdiff_t>((r + 1) * n));
            bool any = false;
            for (double w : out) any = any || !std::isnan(w);
            if (!any) out.clear();
            row_weights = row_weights || any;
        };
        take_w(ov_.gen_slack_w, n_gen_, st.gen_slack_w);
        take_w(ov_.storage_slack_w, n_sto_, st.sto_slack_w);
        if (row_weights) {
            changes_units = true;
            st.rp = nullptr;
        }
        // a controller the row trips is gone, as one on a masked bus (below)
        if (r < ov_.vc_tripped.size() && !ov_.vc_tripped[r].empty()) {
            for (int c : ov_.vc_tripped[r]) st.held_q[static_cast<size_t>(c)] = 0.;
            vc_gone = &ov_.vc_tripped[r];
            changes_units = true;     // its own ReactiveLimits plan, without them
        }
        if (r < ov_.hvdc_off.size() && !ov_.hvdc_off[r].empty()) {
            hvdc_gone = &ov_.hvdc_off[r];
            changes_units = true;     // its own ReactiveLimits plan, without the line's stations
            st.hvdc_tripped.assign(hvdc_id_.size(), 0);
            for (int h : ov_.hvdc_off[r]) {
                const auto it = std::find(hvdc_id_.begin(), hvdc_id_.end(), h);
                if (it != hvdc_id_.end()) st.hvdc_tripped[static_cast<size_t>(it - hvdc_id_.begin())] = 1;
            }
        }
        if (r < ov_.masked.size() && !ov_.masked[r].empty()) {
            st.masked = ov_.masked[r];
            std::sort(st.masked.begin(), st.masked.end());
            st.rowv.masked = &st.masked;
            // a controller on a masked bus is gone (lightsim2grid's ctrl_masked_): never its
            // group's reference, pinned at 0, so the group regulates through its live ones only
            for (int c = 0; c < layout_.n_ctrl; ++c)
                if (std::binary_search(st.masked.begin(), st.masked.end(), vc_bus_[static_cast<size_t>(c)]))
                    st.held_q[static_cast<size_t>(c)] = 0.;
        }
        // the branches its contingency opens (BaseBatchSweep::_fill_outer_row_view)
        if (r < ov_.branch_off.size() && !ov_.branch_off[r].empty()) {
            st.rowv.lines.off.assign(static_cast<size_t>(n_line_), 0);
            st.rowv.trafos.off.assign(static_cast<size_t>(n_trafo_), 0);
            for (int br : ov_.branch_off[r]) {
                if (br >= 0 && br < n_line_) st.rowv.lines.off[static_cast<size_t>(br)] = 1;
                else if (br >= n_line_ && br < n_line_ + n_trafo_) st.rowv.trafos.off[static_cast<size_t>(br - n_line_)] = 1;
            }
        }
        // the generators it places on another bus (a topological action): the loops see them
        // there (their own ReactiveLimits plan, DistributedSlack units); a regulating one
        // holds its new bus PV at its set-point -- that bus' Q row pinned, |V| reset to it
        if (r < ov_.gen_bus.size() && !ov_.gen_bus[r].empty()) {
            gen_moves = &ov_.gen_bus[r];
            st.rowv.gens.bus_solver.assign(static_cast<size_t>(n_gen_), ls2g::OuterRowView::Units::GRID_BUS);
            const auto& gens = grid_->get_generators();
            for (const auto& [g, bus] : ov_.gen_bus[r]) {
                if (g < 0 || g >= n_gen_ || bus < 0 || bus >= n_bus_)
                    throw std::runtime_error("outer model: generator " + std::to_string(g) + " placed on bus "
                                             + std::to_string(bus) + ", outside the solver buses");
                st.rowv.gens.bus_solver[static_cast<size_t>(g)] = bus;
                if (!gens.get_voltage_regulator_on(g)) continue;
                const int k = pin_of_bus_[static_cast<size_t>(bus)];
                if (k < 0)
                    throw std::runtime_error("outer model: generator " + std::to_string(g) + " moved onto bus "
                                             + std::to_string(bus) + ", which the outer loops' ledger has not reserved");
                st.pq[static_cast<size_t>(k)] = 0;
                reset_vm(st, bus, static_cast<double>(gens.get_target_vm_pu(g)));
            }
            changes_units = true;
        }
        // the switchable buses a generator contingency leaves without a regulator start PQ
        if (r < ov_.pq_buses.size())
            for (int b : ov_.pq_buses[r]) {
                const int k = (b >= 0 && b < n_bus_) ? pin_of_bus_[static_cast<size_t>(b)] : -1;
                if (k < 0)
                    throw std::runtime_error("outer model: a generator contingency releases bus " + std::to_string(b) +
                                             ", which the outer loops' ledger has not reserved");
                st.pq[static_cast<size_t>(k)] = 1;
            }
    }
    // the grid's plan, unless the row changes what it is built from: the units it
    // disconnects, the buses it masks, its machines' voltage set-points (a local bus'
    // target_vm, what its release test compares with); masks alone only filter it
    {
        ProfScope p(prof_, P_LOAD_RL_PLAN);
        st.lim_base = st.rowv.gens.target_p.empty();
    // (a row moving no generator: the grid's plan filtered, build_bus_q_plan's result at a fraction)
    if (changes_units || !st.rowv.gens.target_vm.empty())
        st.rl = build_rl_plan(view(st), vc_gone, hvdc_gone, /*filtered=*/true, gen_moves);
        else if (!st.masked.empty())                          st.rl = rl_plan_masked(st.masked);
        else                                                  st.rl = rl0_;
    }
    // BranchControl: the slot starts at the grid's taps (build_slot0), without the transformers
    // the row opens (drop_out_of_row); no control requested yet (OuterControls::reset_states)
    if (!st.rowv.trafos.off.empty()) {
        ProfScope p(prof_, P_LOAD_BC);
        const ls2g::OuterGridView v = view(st);
        for (size_t e = 0; e < static_cast<size_t>(n_bc_); ++e) {
            int b1 = -1, b2 = -1;
            st.bc_out[e] = !(v.trafo_closed(bc_trafo_[e], b1, b2) && b1 == bc_b1_[e] && b2 == bc_b2_[e]);
        }
    }
    // ShuntControl: the grid's sections, no control requested yet (build_slot0)
    // DistributedSlackLoop's collect_units: the grid's participants, unless the row
    // disconnects units (their injection is the row's target in force: init_loops) or
    // strands buses (a unit on one takes no share)
    {
        ProfScope p(prof_, P_LOAD_UNITS);
        if (prof_.on && !st.masked.empty()) ++prof_.n[P_LOAD_MASKED_ROWS];
        // a unit the row moves that the grid's participants lack (one the grid has disconnected):
        // collect_units' then
        bool units_filterable = st.gen_slack_w.empty() && st.sto_slack_w.empty();
        if (gen_moves != nullptr)
            for (const auto& gb : *gen_moves) {
                if (!(grid_->get_generators().get_can_participate_slack_weight(gb.first) > 0.)) continue;
                bool listed = false;
                for (const auto& u : units0_)
                    listed = listed || (u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR && u.el_id == gb.first);
                units_filterable = units_filterable && listed;
            }
        if (changes_units && units_filterable) {
            // the grid's participants still on and on a live bus, in order, a moved one on its row bus
            // (collect_units' rules; the injections are rewritten by init_loops)
            const ls2g::OuterGridView v = view(st);
            const auto& to_me = v.id_solver_to_me();
            st.ds_units.reserve(units0_.size());
            st.ds_unit_bus.reserve(units0_.size());
            for (size_t k = 0; k < units0_.size(); ++k) {
                const auto& u = units0_[k];
                const bool gen = u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR;
                const int bus = gen ? v.gen_bus_solver(u.el_id) : v.storage_bus_solver(u.el_id);
                if (bus < 0 || v.is_masked(bus)) continue;
                st.ds_units.push_back(u);
                st.ds_unit_bus.push_back(bus);
                if (bus != units0_bus_[k]) st.ds_units.back().bus = to_me[bus].cast_int();
            }
            if (check_rl_filter_) {
                std::vector<ls2g::slack_redistribution::Participant> ref;
                std::vector<int> ref_bus;
                collect_units(v, ref, ref_bus);
                // a limit may be NaN (none): the same NaN on both sides is the same limit
                auto num = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
                auto unit_same = [&](size_t k) {
                    return ref[k].kind == st.ds_units[k].kind && ref[k].el_id == st.ds_units[k].el_id &&
                           ref[k].bus == st.ds_units[k].bus && ref_bus[k] == st.ds_unit_bus[k] &&
                           num(ref[k].weight, st.ds_units[k].weight) && num(ref[k].min_p_mw, st.ds_units[k].min_p_mw) &&
                           num(ref[k].max_p_mw, st.ds_units[k].max_p_mw);
                };
                bool same = ref.size() == st.ds_units.size();
                for (size_t k = 0; same && k < ref.size(); ++k) same = unit_same(k);
                if (!same) {
                    std::ostringstream os;
                    os << "outer model: the filtered DistributedSlack units are not collect_units' (row " << row << "): "
                       << st.ds_units.size() << " filtered vs " << ref.size() << ";";
                    for (size_t k = 0; k < std::max(ref.size(), st.ds_units.size()); ++k) {
                        auto put = [&](const char* who, const std::vector<ls2g::slack_redistribution::Participant>& u,
                                       const std::vector<int>& b) {
                            if (k < u.size())
                                os << " " << who << "(kind " << static_cast<int>(u[k].kind) << " el " << u[k].el_id
                                   << " bus " << u[k].bus << "/" << b[k] << " w " << u[k].weight << " p ["
                                   << u[k].min_p_mw << ", " << u[k].max_p_mw << "])";
                            else os << " " << who << "(none)";
                        };
                        if (k < ref.size() && k < st.ds_units.size() && unit_same(k)) continue;
                        os << " at " << k << ":";
                        put("filtered", st.ds_units, st.ds_unit_bus);
                        put("ref", ref, ref_bus);
                        break;
                    }
                    throw std::runtime_error(os.str());
                }
                st.ds_units = ref;   // the injections as collect_units has them (init_loops rewrites them)
            }
        } else if (changes_units) {
            collect_units(view(st), st.ds_units, st.ds_unit_bus, st.gen_slack_w.empty() ? nullptr : &st.gen_slack_w,
                          st.sto_slack_w.empty() ? nullptr : &st.sto_slack_w);
        } else if (!st.masked.empty()) {
            st.ds_units.reserve(units0_.size());
            st.ds_unit_bus.reserve(units0_.size());
            for (size_t k = 0; k < units0_.size(); ++k) {
                if (std::binary_search(st.masked.begin(), st.masked.end(), units0_bus_[k])) continue;
                st.ds_units.push_back(units0_[k]);
                st.ds_unit_bus.push_back(units0_bus_[k]);
            }
        } else {
            st.ds_units = units0_;
            st.ds_unit_bus = units0_bus_;
        }
    }
    { ProfScope p(prof_, P_LOAD_SET_ACTIVE); set_active(st); }
    st.loop_iter.assign(st.active.size(), 0);
    st.done_at.assign(st.active.size(), -1);
    st.v_version = 0;
    st.changed = false;
    if (row >= 0) {
        if (st.rp != nullptr) {
            ProfScope p(prof_, P_LOAD_RESUME_TARGETS);
            // the point's moves, unless the row sets targets or disconnects units
            if (st.rp != &point_n_ || changes_units || !st.rowv.gens.target_p.empty() || !st.rowv.storages.target_p.empty()) {
                take_resume_targets(st);
            } else {
                st.gen_target_p = resume0_gen_target_p_;
                st.storage_target_p = resume0_sto_target_p_;
                st.target_delta = resume0_target_delta_;
            }
        }
        if (st.rp != nullptr) {
            const ResumePoint& rp = *st.rp;
            // OuterLoopAlgo::_apply_carried_positions: every tap control moved to the position
            // the solve resumed from left it at (a transformer the row opens takes nothing)
            for (int e = 0; e < n_bc_; ++e) {
                const size_t ee = static_cast<size_t>(e);
                if (st.bc_out[ee]) continue;
                const size_t t = static_cast<size_t>(bc_trafo_[ee]);
                const int none = std::numeric_limits<int>::min();
                if (bc_phase_[ee] && t < rp.phase_tap.size() && rp.phase_tap[t] != none) {
                    st.bc_moved[ee] = 1;
                    st.bc_req_tap[ee] = rp.phase_tap[t];
                }
                if (bc_ratio_[ee] && t < rp.ratio_tap.size() && rp.ratio_tap[t] != none)
                    st.bc_req_rtap[ee] = rp.ratio_tap[t];
            }
            // ... and every shunt controller all of whose shunts have a carried count
            for (int e = 0; e < n_sc_; ++e) {
                std::vector<int> counts;
                const int none = std::numeric_limits<int>::min();
                bool all = true;
                for (int sh : sc_shunts_[static_cast<size_t>(e)]) {
                    const int c = static_cast<size_t>(sh) < rp.sections.size() ? rp.sections[static_cast<size_t>(sh)] : none;
                    if (c == none) { all = false; break; }
                    counts.push_back(c);
                }
                if (all && !counts.empty()) sc_set_sections(st, e, counts);
            }
        }
        { ProfScope p(prof_, P_LOAD_INIT_LOOPS); init_loops(st); }
        if (st.rp != nullptr) { ProfScope p(prof_, P_LOAD_RESUME_LOOPS); resume_loops(st); }
        st.check_unrealistic = st.last_fixing < 0;
    }
    materialize(st);
}

void Ls2gOuterModel::take_resume_targets(SlotState& st) const
{
    // OuterLoopAlgo::_take_targets: the targets the loops moved there, but where the row sets
    // its own (on a slot whose injections nothing moved yet)
    const ls2g::OuterGridView v = view(st);
    const ResumePoint& rp = *st.rp;
    for (size_t gid = 0; gid < rp.gen_p.size(); ++gid) {
        const int g = static_cast<int>(gid);
        if (std::isnan(rp.gen_p[gid]) || v.row_sets_gen_target_p(g)) continue;
        const int bus = v.gen_bus_solver(g);
        if (bus >= 0) move_gen(st, g, bus, rp.gen_p[gid]);
    }
    for (size_t sid = 0; sid < rp.sto_p.size(); ++sid) {
        const int s = static_cast<int>(sid);
        if (std::isnan(rp.sto_p[sid]) || v.row_sets_storage_target_p(s)) continue;
        const int bus = v.storage_bus_solver(s);
        if (bus >= 0) move_storage(st, s, bus, rp.sto_p[sid]);
    }
}

std::shared_ptr<const RlPlan> Ls2gOuterModel::rl_plan_masked(const std::vector<int>& masked) const
{
    // build_rl_plan on a row whose only change is its masked buses: bus_q_check's plan skips
    // a masked bus and nothing else depends on the masks, so it is the grid's plan without
    // the buses on masked ones (in the same order); the monitors are kept, as there
    auto rl = std::make_shared<RlPlan>();
    rl->plan = rl0_->plan;
    rl->has_monitor = rl0_->has_monitor;
    const size_t n = rl0_->buses.size();
    rl->buses.reserve(n);
    rl->state0.reserve(n);
    for (size_t k = 0; k < n; ++k) {
        const bool bus_q = k < rl0_->n_bus_q;
        if (bus_q && std::binary_search(masked.begin(), masked.end(), rl0_->buses[k].bus_solver)) continue;
        if (bus_q) ++rl->n_bus_q;
        rl->buses.push_back(rl0_->buses[k]);
        rl->state0.push_back(rl0_->state0[k]);
    }
    return rl;
}

void Ls2gOuterModel::check_row_slack_weights(const RowOverrides& ov) const
{
    const auto check = [&](const std::vector<double>& m, int n, const char* what, const char* setter,
                           const auto& grid_weight) {
        if (m.empty() || n <= 0) return;
        for (size_t k = 0; k < m.size(); ++k) {
            if (!(m[k] >= 1e-7)) continue;   // NaN, or out of the slack (upstream _tol_equal_float)
            const int el = static_cast<int>(k % static_cast<size_t>(n));
            if (grid_weight(el) > 0.) continue;
            throw std::invalid_argument(
                std::string(setter) + ": row " + std::to_string(k / static_cast<size_t>(n)) + " gives " + what + " " +
                std::to_string(el) + " a slack weight, but it takes no part in the grid's distributed slack, with "
                "the outer loops: the units sharing the slack are reserved once for the batch. Make that unit a "
                "participant of the grid's slack (add_gen_slackbus / add_storage_slackbus) and give it 0 in the "
                "rows where it should take no share.");
        }
    };
    const auto& gens = grid_->get_generators();
    const auto& stos = grid_->get_storages();
    check(ov.gen_slack_w, n_gen_, "generator", "set_gen_slack_weights",
          [&gens](int g) { return static_cast<double>(gens.get_can_participate_slack_weight(g)); });
    check(ov.storage_slack_w, n_sto_, "storage unit", "set_storage_slack_weights",
          [&stos](int s) { return static_cast<double>(stos.get_can_participate_slack_weight(s)); });
}

void Ls2gOuterModel::collect_units(const ls2g::OuterGridView& v,
                                   std::vector<ls2g::slack_redistribution::Participant>& units,
                                   std::vector<int>& bus_of, const std::vector<double>* gen_w,
                                   const std::vector<double>* sto_w) const
{
    // the row's weight of a unit where it gives one (OuterGridView::gen_slack_weight), the
    // grid's "can participate" one otherwise
    const auto row_w = [](const std::vector<double>* w, int el, double own) {
        if (w == nullptr || static_cast<size_t>(el) >= w->size()) return own;
        const double x = (*w)[static_cast<size_t>(el)];
        return std::isnan(x) ? own : x;
    };
    // DistributedSlackLoop's collect_units: the participants of this row
    units.clear();
    bus_of.clear();
    const auto& gens = grid_->get_generators();
    const auto& stos = grid_->get_storages();
    const auto& to_me = v.id_solver_to_me();
    auto add = [&](ls2g::slack_redistribution::UnitKind kind, int el, int bus, double w,
                   double inj, double mn, double mx) {
        if (!(w > 0.) || bus < 0 || v.is_masked(bus)) return;
        ls2g::slack_redistribution::Participant u;
        u.kind = kind; u.el_id = el; u.bus = to_me[bus].cast_int(); u.injection_mw = inj;
        u.weight = w; u.min_p_mw = mn; u.max_p_mw = mx; u.in_slack = false;
        units.push_back(u);
        bus_of.push_back(bus);
    };
    for (int el = 0; el < gens.nb(); ++el)
        add(ls2g::slack_redistribution::UnitKind::GENERATOR, el, v.gen_bus_solver(el),
            row_w(gen_w, el, gens.get_can_participate_slack_weight(el)), v.gen_target_p(el), gens.get_min_p(el),
            gens.get_max_p(el));
    for (int el = 0; el < stos.nb(); ++el)
        add(ls2g::slack_redistribution::UnitKind::STORAGE, el, v.storage_bus_solver(el),
            row_w(sto_w, el, stos.get_can_participate_slack_weight(el)), -v.storage_target_p(el), stos.get_min_p(el),
            stos.get_max_p(el));
}

bool Ls2gOuterModel::filtered_bus_q_plan(const ls2g::OuterGridView& v, ls2g::bus_q_check::BusQPlan& out,
                                         const std::vector<std::pair<int, int>>* moves) const
{
    // build_bus_q_plan's rules on the grid's entries (in the grid's bus order): a generator that is
    // a controller of the plan counts while on, any other while a local voltage controller; a
    // storage unit while a local voltage controller; a station while its line is on; an SVC and
    // the controller positions as the grid's; a masked bus, or one left without a machine, out.
    // The generators the row moves (`moves`, grid id -> solver bus) leave their grid bus and join
    // their row bus -- an entry of the grid's, or a bus it gets (nothing else of the plan stands
    // there: a storage unit, station or SVC would have made it one; no controller either)
    const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();
    std::vector<char> gen_is_ctrl(static_cast<size_t>(n_gen_), 0);
    for (int c = 0; c < ctrl.n_controllers(); ++c)
        if (!ctrl.is_held(c) && ctrl.kind(c) == ls2g::VoltageControlSolverData::GEN && ctrl.elem_id(c) >= 0 &&
            ctrl.elem_id(c) < n_gen_)
            gen_is_ctrl[static_cast<size_t>(ctrl.elem_id(c))] = 1;
    std::vector<int> moved;                      // sorted grid ids
    std::map<int, std::vector<int>> arriving;    // solver bus -> the generators the row moves there
    if (moves != nullptr)
        for (const auto& [g, bus] : *moves) {
            if (g < 0 || g >= n_gen_) continue;
            if (gen_is_ctrl[static_cast<size_t>(g)]) return false;   // its controller position stays the layout's
            moved.push_back(g);
            const int b = v.gen_bus_solver(g);
            if (b >= 0 && b < n_bus_) arriving[b].push_back(g);
        }
    std::sort(moved.begin(), moved.end());
    auto counts = [&](int gid) {
        return gen_is_ctrl[static_cast<size_t>(gid)] ? v.gen_on(gid) : v.gen_is_local_voltage_controller(gid);
    };
    out.clear();
    for (size_t k = 0; k < rl0_->n_bus_q; ++k) {
        const auto& g = rl0_->plan->buses[static_cast<size_t>(rl0_->buses[k].entry)];
        if (v.is_masked(g.bus_solver)) { arriving.erase(g.bus_solver); continue; }
        ls2g::bus_q_check::BusQEntry e = g;
        e.gen_ids.clear();
        for (int gid : g.gen_ids)
            if (!std::binary_search(moved.begin(), moved.end(), gid) && counts(gid)) e.gen_ids.push_back(gid);
        if (auto it = arriving.find(g.bus_solver); it != arriving.end()) {
            for (int gid : it->second) if (counts(gid)) e.gen_ids.push_back(gid);
            std::sort(e.gen_ids.begin(), e.gen_ids.end());
            arriving.erase(it);
        }
        e.storage_ids.clear();
        for (int sid : g.storage_ids) if (v.storage_is_local_voltage_controller(sid)) e.storage_ids.push_back(sid);
        e.station_ids.clear();
        for (const auto& st : g.station_ids) if (v.hvdc_on(st.first)) e.station_ids.push_back(st);
        if (e.gen_ids.empty() && e.storage_ids.empty() && e.station_ids.empty() && e.svc_ids.empty()) continue;
        if (!e.ctrl_pos.empty()) out.needs_controller_q = true;
        out.buses.push_back(std::move(e));
    }
    if (arriving.empty()) return true;
    // the buses the moves bring, each in its place in the grid's bus order
    const auto& to_me = v.id_solver_to_me();
    const auto& subs = grid_->get_substations();
    const std::vector<std::string>& sub_names = subs.get_sub_names();
    for (auto& [b, gens] : arriving) {
        if (v.is_masked(b)) continue;
        ls2g::bus_q_check::BusQEntry e;
        e.bus_solver = b;
        e.bus_grid = to_me[b].cast_int();
        for (int gid : gens) if (counts(gid)) e.gen_ids.push_back(gid);
        if (e.gen_ids.empty()) continue;
        std::sort(e.gen_ids.begin(), e.gen_ids.end());
        for (int c = 0; c < ctrl.n_controllers(); ++c)
            if (ctrl.bus(c) == b && !ctrl.is_held(c)) e.ctrl_pos.push_back(c);
        if (!e.ctrl_pos.empty()) out.needs_controller_q = true;
        const int sub_id = subs.sub_id_of_bus(e.bus_grid);
        if (sub_id >= 0 && static_cast<size_t>(sub_id) < sub_names.size()) e.sub_name = sub_names[static_cast<size_t>(sub_id)];
        auto at = std::upper_bound(out.buses.begin(), out.buses.end(), e.bus_grid,
                                   [](int bg, const ls2g::bus_q_check::BusQEntry& x) { return bg < x.bus_grid; });
        out.buses.insert(at, std::move(e));
    }
    return true;
}

std::shared_ptr<const RlPlan> Ls2gOuterModel::build_rl_plan(const ls2g::OuterGridView& v, const std::vector<int>* gone,
                                                            const std::vector<int>* hvdc_gone, bool filtered,
                                                            const std::vector<std::pair<int, int>>* moves) const
{
    auto rl = std::make_shared<RlPlan>();
    // ReactiveLimitsLoop::_initialize: bus_q_check's plan on this view, plus one entry per
    // voltage monitor (a held SVC controller of the plan)
    auto plan = std::make_shared<ls2g::bus_q_check::BusQPlan>();
    rl->plan = plan;
    {
    ProfScope p_filter(prof_, P_RL_PLAN_FILTER);
    if (filtered && !filtered_bus_q_plan(v, *plan, moves)) filtered = false;
    if (filtered) {
        if (check_rl_filter_) {   // GPUSIM2GRID_OUTER_CHECK_RL_PLAN: the filter is build_bus_q_plan's plan
            ls2g::bus_q_check::BusQPlan ref;
            ls2g::bus_q_check::build_bus_q_plan(v, ref);
            bool same = ref.buses.size() == plan->buses.size() && ref.needs_controller_q == plan->needs_controller_q;
            for (size_t k = 0; same && k < ref.buses.size(); ++k) {
                const auto& a = ref.buses[k];
                const auto& b = plan->buses[k];
                same = a.bus_solver == b.bus_solver && a.bus_grid == b.bus_grid && a.gen_ids == b.gen_ids &&
                       a.storage_ids == b.storage_ids && a.station_ids == b.station_ids && a.svc_ids == b.svc_ids &&
                       a.ctrl_pos == b.ctrl_pos && a.sub_name == b.sub_name;
            }
            if (!same) throw std::runtime_error("outer model: the filtered ReactiveLimits plan is not build_bus_q_plan's");
        }
    } else {
        plan->clear();
        ls2g::bus_q_check::build_bus_q_plan(v, *plan);
    }
    }
    ProfScope p_buses(prof_, P_RL_PLAN_BUSES);
    // The controllers the row trips (`gone`, sorted) hold nothing: the plan keeps every
    // controller standing on a bus (a generator it only drops from the machines), and a
    // freeze of that bus would write a limit over their Q = 0.
    // The hvdc lines it trips (`hvdc_gone`, grid ids): their stations hold nothing either,
    // what lightsim2grid #292's OuterRowView::hvdc_off tells its own plan.
    const bool any_gone = gone != nullptr && !gone->empty();
    const bool any_hvdc_gone = hvdc_gone != nullptr && !hvdc_gone->empty();
    if (any_hvdc_gone)
        for (auto& entry : plan->buses)
            entry.station_ids.erase(std::remove_if(entry.station_ids.begin(), entry.station_ids.end(), [&](const auto& station) {
                return std::find(hvdc_gone->begin(), hvdc_gone->end(), station.first) != hvdc_gone->end(); }),
                entry.station_ids.end());
    if (any_gone)
        for (auto& entry : plan->buses) {
            entry.ctrl_pos.erase(std::remove_if(entry.ctrl_pos.begin(), entry.ctrl_pos.end(), [&](int c) {
                return std::binary_search(gone->begin(), gone->end(), c); }), entry.ctrl_pos.end());
            for (int c : *gone)
                if (vc_kind_[static_cast<size_t>(c)] == ls2g::VoltageControlSolverData::SVC)
                    entry.svc_ids.erase(std::remove(entry.svc_ids.begin(), entry.svc_ids.end(), vc_elem_[static_cast<size_t>(c)]),
                                        entry.svc_ids.end());
        }
    if (any_gone || any_hvdc_gone)
        plan->buses.erase(std::remove_if(plan->buses.begin(), plan->buses.end(), [](const ls2g::bus_q_check::BusQEntry& e) {
            return e.gen_ids.empty() && e.storage_ids.empty() && e.station_ids.empty() && e.svc_ids.empty(); }), plan->buses.end());
    rl->n_bus_q = plan->buses.size();
    std::vector<int> monitor_of_entry(plan->buses.size(), -1);
    const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();
    for (int c = 0; c < ctrl.n_controllers(); ++c) {
        if (!ctrl.is_held(c) || ctrl.kind(c) != ls2g::VoltageControlSolverData::SVC) continue;
        if (any_gone && std::binary_search(gone->begin(), gone->end(), c)) continue;   // tripped
        rl->has_monitor = true;
        ls2g::bus_q_check::BusQEntry entry;
        entry.bus_solver = ctrl.bus(c);
        entry.bus_grid = v.id_solver_to_me()[ctrl.bus(c)].cast_int();
        entry.svc_ids.push_back(ctrl.elem_id(c));
        entry.ctrl_pos.push_back(c);
        plan->buses.push_back(entry);
        monitor_of_entry.push_back(ctrl.elem_id(c));
    }
    const auto& gens = grid_->get_generators();
    const auto& stos = grid_->get_storages();
    const auto& hvdcs = grid_->get_dclines();
    const auto vn_kv = grid_->get_bus_vn_kv();
    // the robust mode's: the target Q of every generator ON in this row, summed per bus (filtered: over
    // the grid's generators of each bus and those the row moves there, in id order -- the same sums)
    std::vector<double> target_q_of_bus;
    if (!filtered || check_rl_filter_) {
        target_q_of_bus.assign(v.id_solver_to_me().size(), 0.);
        for (int gid = 0; gid < gens.nb(); ++gid) {
            const int b = v.gen_bus_solver(gid);
            if (b >= 0 && static_cast<size_t>(b) < target_q_of_bus.size())
                target_q_of_bus[static_cast<size_t>(b)] += gens.get_target_q()(gid);
        }
    }
    auto target_q_at = [&](int b) {
        if (!filtered) return target_q_of_bus[static_cast<size_t>(b)];
        // the grid's generators of the bus still there and those the row moves onto it, in id order
        std::vector<int> at;
        if (b >= 0 && static_cast<size_t>(b) < gens_of_bus0_.size())
            for (int gid : gens_of_bus0_[static_cast<size_t>(b)])
                if (v.gen_bus_solver(gid) == b) at.push_back(gid);
        if (moves != nullptr) {
            for (const auto& gb : *moves)
                if (gb.first >= 0 && gb.first < gens.nb() && v.gen_bus_solver(gb.first) == b &&
                    std::find(at.begin(), at.end(), gb.first) == at.end())
                    at.push_back(gb.first);
            std::sort(at.begin(), at.end());
        }
        double q = 0.;
        for (int gid : at) q += gens.get_target_q()(gid);
        if (check_rl_filter_ && std::memcmp(&q, &target_q_of_bus[static_cast<size_t>(b)], sizeof(double)) != 0)
            throw std::runtime_error("outer model: a filtered row's target Q sum is not build_rl_plan's");
        return q;
    };
    for (size_t k = 0; k < rl->plan->buses.size(); ++k) {
        const auto& entry = rl->plan->buses[k];
        RlBus bus;
        bus.entry = static_cast<int>(k);
        bus.bus_solver = entry.bus_solver;
        bus.local = entry.ctrl_pos.empty() && entry.svc_ids.empty();
        bus.monitor_svc = monitor_of_entry[k];
        bus.reg_bus_solver = entry.bus_solver;
        bus.nominal_v = vn_kv(entry.bus_grid);
        // the switchable bus a loop freezing it switches (-1: none reserved, no loop does)
        if (bus.local) bus.pin = pin_of_bus_[static_cast<size_t>(entry.bus_solver)];
        bool has_target = false;
        if (!bus.local) {
            const int gr = ctrl.group(entry.ctrl_pos.front());
            bus.group = gr;
            bus.reg_bus_solver = ctrl.reg_bus(gr);
            bus.target_vm = ctrl.v_set(gr);
            has_target = true;
            const int reg_grid = v.id_solver_to_me()[bus.reg_bus_solver].cast_int();
            if (reg_grid >= 0) bus.nominal_v = vn_kv(reg_grid);
        }
        bus.target_q = target_q_at(entry.bus_solver);
        for (int gid : entry.gen_ids)
            if (!has_target) { bus.target_vm = v.gen_target_vm(gid); has_target = true; }
        for (int sid : entry.storage_ids) {
            bus.min_q += stos.get_min_q(sid);
            bus.max_q += stos.get_max_q(sid);
            if (!has_target) { bus.target_vm = v.storage_target_vm(sid); has_target = true; }
        }
        for (const auto& station : entry.station_ids) {
            bus.min_q += hvdcs.get_station_min_q_mvar(station.first, station.second);
            bus.max_q += hvdcs.get_station_max_q_mvar(station.first, station.second);
            if (!has_target) {
                bus.target_vm = hvdcs.get_station_target_vm_pu(station.first, station.second);
                has_target = true;
            }
        }
        rl->buses.push_back(bus);
    }
    rl->state0.assign(rl->buses.size(), RlBusState());
    return rl;
}

void Ls2gOuterModel::set_active(SlotState& st) const
{
    // OpenLoadFlow's isNeeded filter on this row, and the robust mode's last loop able to
    // fix an unrealistic state
    st.active.clear();
    st.last_fixing = -1;
    const ls2g::OuterGridView v = view(st);
    for (size_t i = 0; i < loops_.size(); ++i) {
        const LoopDecl& d = loops_[i];
        bool needed = false;
        switch (d.kind) {
            case Kind::DistributedSlack:
                // a unit off the slack bus to take the mismatch: the slack bus' own units would
                // only take what it already absorbs (DistributedSlackLoop::_has_participant)
                // Unless asked to run anyway (Params::run_without_participant).
                needed = d.ds_run_without_participant || ds_off_slack_bus(st);
                break;
            case Kind::HvdcLimits:
                // (without the lines the row takes out: nothing to saturate, nothing to release)
                needed = std::any_of(hvdc_plan0_.lines.begin(), hvdc_plan0_.lines.end(), [&](const auto& line) {
                    return !st.hvdc_line_tripped(line.hvdc_id, hvdc_id_); });
                break;
            case Kind::VoltageMonitoring: {
                const auto& svcs = grid_->get_svcs();
                for (int id : held_svcs_)
                    if (!row_trips_svc(st, id) && svcs.get_regulated_bus_id(id) == svcs.get_bus_id()(id).cast_int())
                        needed = true;
                break;
            }
            case Kind::ReactiveLimits:    needed = !st.rl->buses.empty() || st.rl->has_monitor; break;
            case Kind::PhaseControl:      needed = true; break;   // stable at once without a shifter
            // the groups of an active loop are kept for its initialize (init_loops)
            case Kind::TransformerVoltageControl: {
                auto groups = d.tvc->groups(v);
                for (const auto& gr : groups) needed = needed || !gr.hidden;
                if (needed) st.tvc_groups = std::move(groups);
                break;
            }
            case Kind::ShuntVoltageControl: {
                auto groups = ls2g::ShuntVoltageControlLoop::groups(v, tvc_buses_);
                for (const auto& gr : groups) needed = needed || !gr.hidden;
                if (needed) st.sv_groups = std::move(groups);
                break;
            }
        }
        if (needed) st.active.push_back(static_cast<int>(i));
    }
    // (BaseOuterLoop::can_fix_unrealistic_state: ReactiveLimits, TransformerVoltageControl)
    if (params_.voltage_remote_control_robust_mode)
        for (size_t i = 0; i < st.active.size(); ++i) {
            const Kind k = loops_[static_cast<size_t>(st.active[i])].kind;
            if (k == Kind::ReactiveLimits || k == Kind::TransformerVoltageControl) st.last_fixing = static_cast<int>(i);
        }
}

// =============================================================================
// The tail compaction moves a row to another slot: its whole state follows.
// The row view points at the slot's own masked list, re-pointed here.
// =============================================================================
void Ls2gOuterModel::move(int from, int to)
{
    SlotState& dst = slots_[static_cast<size_t>(to)];
    dst = std::move(slots_[static_cast<size_t>(from)]);
    if (dst.rowv.masked != nullptr) dst.rowv.masked = &dst.masked;
    slots_[static_cast<size_t>(from)] = SlotState();
}

void Ls2gOuterModel::clear_vm_reset(int slot)
{
    SlotState& st = slots_[static_cast<size_t>(slot)];
    st.vm_reset_bus.clear();
    st.vm_reset_vm.clear();
    st.ctl.vm_reset_bus.clear();
    st.ctl.vm_reset_vm.clear();
    // RatioTapControl::take_tap, ShuntSectionControl::take_sections: applied once
    std::fill(st.bc_req_rtap.begin(), st.bc_req_rtap.end(), std::numeric_limits<int>::min());
    st.ctl.bc_retap.clear();
    std::fill(st.sc_pending.begin(), st.sc_pending.end(), 0);
    st.ctl.sc_set.clear();
}

void Ls2gOuterModel::init_loops(SlotState& st) const
{
    for (int li : st.active) {
        const LoopDecl& d = loops_[static_cast<size_t>(li)];
        switch (d.kind) {
            case Kind::DistributedSlack: {
                ProfScope p_ds(prof_, P_LOAD_INIT_DS);
                if (prof_.on) prof_.n[P_LOAD_DS_NUNITS] += static_cast<long long>(st.ds_units.size());
                // run without a unit off the slack bus: no participant, its own units would
                // only take what it already absorbs (DistributedSlackLoop::_initialize)
                if (!ds_off_slack_bus(st)) { st.ds_units.clear(); st.ds_unit_bus.clear(); }
                // DistributedSlackLoop::_initialize: each unit (collected by load) from its
                // target in force
                for (auto& u : st.ds_units)
                    u.injection_mw = u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR
                        ? gen_in_force(st, u.el_id) : -sto_in_force(st, u.el_id);
                st.ds_current_mw.resize(st.ds_units.size());
                for (size_t k = 0; k < st.ds_units.size(); ++k) st.ds_current_mw[k] = st.ds_units[k].injection_mw;
                st.ds_initial_mw = st.ds_current_mw;
                st.ds_lost_sat.clear();
                share_lost_p(st, d);
                // every unit takes its share in the Newton to start with, but those a resumed
                // solve kept out, and those the row's lost power put at a bound
                st.ds_parked.assign(st.ds_units.size(), 0);
                st.ds_parked_high.assign(st.ds_units.size(), 0);
                if (ds_reserved_) {
                    st.ds_in_newton = true;
                    if (st.rp != nullptr) ds_keep_parked(st);
                    for (size_t k = 0; k < st.ds_units.size() && k < st.ds_lost_sat.size(); ++k) {
                        if (!st.ds_lost_sat[k]) continue;
                        st.ds_parked[k] = 1;
                        st.ds_parked_high[k] = st.ds_lost_high ? 1 : 0;
                    }
                }
                if (keep_adjoint_) st.ds_mark_solved();   // the first solve runs with these
                break;
            }
            case Kind::HvdcLimits:
                st.hvdc_plan = hvdc_plan0_;   // every line linear
                if (!st.hvdc_tripped.empty())
                    st.hvdc_plan.lines.erase(std::remove_if(st.hvdc_plan.lines.begin(), st.hvdc_plan.lines.end(),
                                                            [&](const auto& line) { return st.hvdc_line_tripped(line.hvdc_id, hvdc_id_); }),
                                             st.hvdc_plan.lines.end());
                break;
            case Kind::VoltageMonitoring:
                st.vm_switched_on.clear();
                break;
            case Kind::ShuntVoltageControl: {
                // ShuntVoltageControlLoop::_initialize (its groups built by set_active): every
                // controller on for the first solve
                for (const auto& gr : st.sv_groups) {
                    if (gr.hidden) continue;
                    for (int bus : gr.controller_buses) {
                        const int e = (bus >= 0 && bus < n_bus_) ? sc_of_bus_[static_cast<size_t>(bus)] : -1;
                        if (e >= 0) st.sc_ctl[static_cast<size_t>(e)] = 1;
                    }
                }
                break;
            }
            case Kind::TransformerVoltageControl: {
                // TransformerVoltageControlLoop::_initialize (its groups built by set_active):
                // every voltage control off
                st.tvc_step = 0;
                st.tvc_frozen.clear();
                st.tvc_enabled.assign(static_cast<size_t>(n_trafo_), 0);
                st.tvc_controller.assign(static_cast<size_t>(n_trafo_), 0);
                st.tvc_ratios.assign(static_cast<size_t>(n_trafo_), TvcRatio());
                for (const auto& gr : st.tvc_groups)
                    for (int t : gr.trafos) st.tvc_controller[static_cast<size_t>(t)] = 1;
                for (int t = 0; t < n_trafo_; ++t) {
                    if (!st.tvc_controller[static_cast<size_t>(t)]) continue;
                    const int e = bc_entry(t);
                    if (e >= 0 && bc_ratio_[static_cast<size_t>(e)]) st.bc_rctl[static_cast<size_t>(e)] = 0;
                }
                break;
            }
            case Kind::PhaseControl: {
                // PhaseControlLoop::_initialize: the active power controllers regulate from the
                // first solve on
                st.pc_shifters = ls2g::PhaseControlLoop::shifters(view(st));
                for (const auto& sh : st.pc_shifters) {
                    if (sh.mode != ls2g::RegulationMode::ACTIVE_POWER || !sh.regulates) continue;
                    const int e = bc_of_trafo_[static_cast<size_t>(sh.trafo)];
                    if (e >= 0 && bc_phase_[static_cast<size_t>(e)]) st.bc_ctl[static_cast<size_t>(e)] = 1;
                }
                break;
            }
            case Kind::ReactiveLimits: {
                ProfScope p_rl(prof_, P_LOAD_INIT_RL);
                st.rl_state = st.rl->state0;
                // state0 holds the target P at the base targets: only what the row moved is summed
                const bool sto_base = st.rowv.storages.target_p.empty();
                for (size_t k = 0; k < st.rl_state.size(); ++k) {
                    const RlBus& bus = st.rl->buses[k];
                    if (sto_base && !bus.has_storage && rl_at_base(st, bus)) continue;
                    const auto& entry = st.rl->plan->buses[static_cast<size_t>(bus.entry)];
                    double target_p = 0.;
                    for (int gid : entry.gen_ids) target_p += gen_in_force(st, gid);
                    for (int sid : entry.storage_ids) target_p -= sto_in_force(st, sid);
                    st.rl_state[k].target_p = target_p;
                }
                break;
            }
        }
    }
}

void Ls2gOuterModel::share_lost_p(SlotState& st, const LoopDecl& L) const
{
    // DistributedSlackLoop::_share_lost_p: the row's loss shared before its first solve, from
    // the targets in force (a resumed row's lost units at the "n" solve's moved targets)
    static const std::vector<ls2g::real_type> none;
    const double counted = ls2g::DistributedSlackLoop::lost_p_to_share_mw(
        view(st), st.rp != nullptr ? st.rp->gen_p : none, st.rp != nullptr ? st.rp->sto_p : none);
    st.lost_mw = std::isnan(counted) ? 0. : counted;
    // a row resumed from another row's point: that row's loss is in the point's targets already
    const double lost = (st.rp != nullptr && st.rp != &point_n_) ? st.lost_mw - st.rp->lost_mw : counted;
    if (st.ds_units.empty() || !(std::abs(lost) > L.ds_eps_mw)) return;
    std::vector<ls2g::real_type> new_mw;
    std::vector<char> saturated;
    const ls2g::slack_redistribution::Report report =
        ls2g::slack_redistribution::distribute(st.ds_units, lost, L.ds_eps_mw, new_mw, saturated);
    // every unit at a bound: distribute clears the flags, all are
    st.ds_lost_sat = report.all_saturated ? std::vector<char>(st.ds_units.size(), 1) : saturated;
    st.ds_lost_high = lost > 0.;
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        auto& u = st.ds_units[k];
        if (u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR) move_gen(st, u.el_id, st.ds_unit_bus[k], new_mw[k]);
        else move_storage(st, u.el_id, st.ds_unit_bus[k], -new_mw[k]);
        // the base (u.injection_mw) stays the targets the row started from: the checks share
        // the cumulative mismatch from there, as OpenLoadFlow resets to initialTargetP
        st.ds_current_mw[k] = new_mw[k];
    }
    record(st, L.name, "SHARE_LOST_P", true, -1, lost, L.ds_eps_mw);
}

double Ls2gOuterModel::ds_bound(const SlotState& st, size_t k, bool high) const
{
    // DistributedSlackLoop::_bound: as slack_redistribution::distribute has it, 0 MW on the
    // side of 0 the unit's initial target is not on; NaN without one
    const auto& u = st.ds_units[k];
    if (high) {
        double hi = u.max_p_mw;
        if (u.injection_mw < 0.) hi = std::isnan(hi) ? 0. : std::min(hi, 0.);
        return hi;
    }
    double lo = u.min_p_mw;
    if (!(u.injection_mw < 0.)) lo = std::isnan(lo) ? 0. : std::max(lo, 0.);
    return lo;
}

void Ls2gOuterModel::ds_keep_parked(SlotState& st) const
{
    // DistributedSlackLoop::_keep_parked: at the bound the solve resumed from parked it at
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        const auto& u = st.ds_units[k];
        for (const auto& parked : st.rp->parked) {
            if (parked.kind != u.kind || parked.el_id != u.el_id) continue;
            const double bound = ds_bound(st, k, parked.high);
            if (std::isnan(bound)) break;
            if (u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR) move_gen(st, u.el_id, st.ds_unit_bus[k], bound);
            else move_storage(st, u.el_id, st.ds_unit_bus[k], -bound);
            st.ds_current_mw[k] = bound;
            st.ds_parked[k] = 1;
            st.ds_parked_high[k] = parked.high ? 1 : 0;
            break;
        }
    }
}

double Ls2gOuterModel::ds_share(const SlotState& st, std::vector<size_t>* units) const
{
    // the units the Newton shares on (DistributedSlackLoop::_set_newton_weights, filtered as
    // SlackShareControl::set_units does) and their total weight
    if (units != nullptr) units->clear();
    if (!st.ds_in_newton) return 0.;
    double total = 0.;
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        const double w = st.ds_parked[k] ? 0. : static_cast<double>(st.ds_units[k].weight);
        const int b = st.ds_unit_bus[k];
        if (!(w > 0.) || b < 0 || b >= n_bus_ || slack_idx_of_bus_[static_cast<size_t>(b)] < 0) continue;
        total += w;
        if (units != nullptr) units->push_back(k);
    }
    return total;
}

void Ls2gOuterModel::ds_fold(SlotState& st, double sa) const
{
    ProfScope ps_(prof_, P_DS_FOLD);
    // OuterLoopAlgo::_fold_slack_share: each unit's share of the solve's slack_absorbed into
    // its target, so that what reads the solve finds the units' outputs in their targets
    st.sa = sa;
    st.sa_shared = false;
    if (sa == 0.) return;
    std::vector<size_t> units;
    const double total = ds_share(st, &units);
    // the reference bus took it all (weight 1): what is left on it, as a single slack leaves
    // it (mis_bus then leaves `slack_absorbed * weight` out)
    if (!(total > 0.)) return;
    st.sa_shared = true;
    for (size_t k : units) {
        const auto& u = st.ds_units[k];
        const int bus = st.ds_unit_bus[k];
        // the target Q of a unit that does not regulate follows its P: that, the solve did not see
        auto it = st.target_delta.find(bus);
        const double q_before = it != st.target_delta.end() ? it->second.imag() : 0.;
        // generation convention; slack_absorbed < 0 when the units produce more
        const double mw = -sa * sn_ * static_cast<double>(u.weight) / total;
        if (u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR) move_gen(st, u.el_id, bus, gen_in_force(st, u.el_id) + mw);
        else move_storage(st, u.el_id, bus, sto_in_force(st, u.el_id) - mw);   // load convention
        it = st.target_delta.find(bus);
        const double q_after = it != st.target_delta.end() ? it->second.imag() : 0.;
        st.unsolved_q_mvar += std::abs(q_after - q_before) * sn_;
    }
}

void Ls2gOuterModel::resume_loops(SlotState& st) const
{
    const ResumePoint& rp = *st.rp;
    RowView rv{nullptr, &widx_, rp.V.size() > 0 ? &rp.V : nullptr, rp.V.size() > 0 ? nullptr : &rp.v_at};
    for (int li : st.active) {
        const LoopDecl& d = loops_[static_cast<size_t>(li)];
        switch (d.kind) {
            case Kind::DistributedSlack:   // its units start from the targets in force: taken above
            case Kind::PhaseControl:       // restarts (its taps are carried: load)
            case Kind::TransformerVoltageControl:
            case Kind::ShuntVoltageControl:
                break;
            case Kind::HvdcLimits:
                // HvdcAcEmulationLimitsLoop::_initialize: each line in the regime the solve
                // resumed from left it in
                for (auto& line : st.hvdc_plan.lines) {
                    auto it = rp.hvdc.find(line.hvdc_id);
                    if (it == rp.hvdc.end() || it->second == 0) continue;
                    const auto e = std::find(hvdc_id_.begin(), hvdc_id_.end(), line.hvdc_id);
                    if (e == hvdc_id_.end()) continue;
                    line.frozen_dir = it->second;
                    st.hvdc_regime[static_cast<size_t>(e - hvdc_id_.begin())] = it->second;
                }
                break;
            case Kind::VoltageMonitoring:
                // a switched-on SVC stays on for good -- unless the row trips it
                st.vm_switched_on.clear();
                for (const auto& on : rp.svc_on) {
                    if (row_trips_svc(st, on.first)) continue;
                    st.vm_switched_on.push_back(on);
                    if (on.first < 0 || static_cast<size_t>(on.first) >= svc_ctrl_.size()) continue;
                    const int c = svc_ctrl_[static_cast<size_t>(on.first)];
                    if (c >= 0) st.svc_on[static_cast<size_t>(c)] = on.second;
                }
                break;
            case Kind::ReactiveLimits:
                // ReactiveLimitsLoop::_initialize: the buses the solve left frozen, each at the
                // limit it holds in THIS solve (or its target Q for the robust mode's)
                for (size_t k = 0; k < st.rl_state.size(); ++k) {
                    const RlBus& bus = st.rl->buses[k];
                    auto it = rp.frozen.find(bus.bus_solver);
                    if (it == rp.frozen.end() || it->second.first == 0) continue;
                    if (prof_.on) ++prof_.n[P_RL_RESUME_FROZEN];
                    st.rl_state[k].realistic = it->second.second;
                    double q_min, q_max;
                    rl_limits(st, &rv, bus, q_min, q_max);
                    rl_freeze(st, rv, k, st.rl_state[k].realistic ? bus.target_q : (it->second.first < 0 ? q_min : q_max),
                              it->second.first);
                }
                break;
        }
    }
}

void Ls2gOuterModel::record(SlotState& st, const std::string& loop, const std::string& action, bool taken,
                            int element, double value, double limit) const
{
    if (!opt_.keep_trace) return;
    std::ostringstream os;
    os << loop << " #" << st.nb_outer << " " << action << (taken ? "" : " (not taken)")
       << ", element " << element << ", value=" << value << ", limit=" << limit;
    st.trace.push_back(os.str());
}

// =============================================================================
// The control state as the device sees it
// =============================================================================
void Ls2gOuterModel::materialize(SlotState& st) const
{
    ProfScope ps_(prof_, P_MATERIALIZE);
    SlotControls& c = st.ctl;
    const size_t np = layout_.pin_bus.size();
    c.pin.assign(np, 1);
    for (size_t k = 0; k < np; ++k) c.pin[k] = st.pq[k] ? 0 : 1;
    c.held_q = st.held_q;
    c.plan_held.assign(static_cast<size_t>(layout_.n_ctrl), 0);
    c.v_set = vc_v_set_;
    // the row's gen_v on a group (BaseBatchSweep::_apply_step_vc_v_set)
    if (st.row >= 0 && !ov_.vc_v_set.empty())
        for (int g = 0; g < layout_.n_grp; ++g) {
            const double vs = ov_.vc_v_set[static_cast<size_t>(st.row) * layout_.n_grp + g];
            if (std::isfinite(vs)) c.v_set[static_cast<size_t>(g)] = vs;
        }
    for (int j = 0; j < layout_.n_ctrl; ++j) {
        const size_t jj = static_cast<size_t>(j);
        c.plan_held[jj] = vc_plan_held_[jj];
        // VoltageControl::release_held_svcs: a standby SVC switched on regulates its group
        if (vc_plan_held_[jj] && vc_kind_[jj] == ls2g::VoltageControlSolverData::SVC && std::isfinite(st.svc_on[jj])) {
            c.plan_held[jj] = 0;
            c.v_set[static_cast<size_t>(vc_group_[jj])] = st.svc_on[jj];
        }
    }
    c.hvdc_status.assign(static_cast<size_t>(layout_.n_hvdc), 0);
    for (int e = 0; e < layout_.n_hvdc; ++e)
        c.hvdc_status[static_cast<size_t>(e)] = st.hvdc_regime[static_cast<size_t>(e)] == 2
            ? hvdc_status0_[static_cast<size_t>(e)] : st.hvdc_regime[static_cast<size_t>(e)];
    for (size_t e = 0; e < st.hvdc_tripped.size(); ++e)
        if (st.hvdc_tripped[e]) c.hvdc_status[e] = HVDC_STATUS_TRIPPED;
    // target_delta + i.freeze_q, per bus: both sorted by bus, merged without a copy (the
    // same sums as adding freeze_q into a copy of target_delta)
    c.sbus_bus.clear(); c.sbus_dp.clear(); c.sbus_dq.clear();
    auto td = st.target_delta.cbegin();
    auto fq = st.freeze_q.cbegin();
    while (td != st.target_delta.cend() || fq != st.freeze_q.cend()) {
        const bool take_td = td != st.target_delta.cend() && (fq == st.freeze_q.cend() || td->first <= fq->first);
        const bool take_fq = fq != st.freeze_q.cend() && (td == st.target_delta.cend() || fq->first <= td->first);
        const int bus = take_td ? td->first : fq->first;
        cplx ds = take_td ? td->second : cplx(0., 0.);
        if (take_fq) ds += cplx(0., fq->second);
        c.sbus_bus.push_back(bus);
        c.sbus_dp.push_back(ds.real());
        c.sbus_dq.push_back(ds.imag());
        if (take_td) ++td;
        if (take_fq) ++fq;
    }
    c.vm_reset_bus = st.vm_reset_bus;
    c.vm_reset_vm = st.vm_reset_vm;
    // BranchControl (NROuterInner::apply_controls): the phase taps the loops moved (again
    // before every solve, PhaseShifterControl::tap_moved being for the whole solve), the ratio
    // taps once (RatioTapControl::take_tap), then the controls they asked for
    c.bc_out = st.bc_out;
    c.bc_retap.clear();
    for (int e = 0; e < n_bc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        if (st.bc_out[ee]) continue;
        bool retap = false;
        if (st.bc_moved[ee]) { bc_retap(st, e, st.bc_rpos[ee], st.bc_req_tap[ee]); retap = true; }
        if (st.bc_req_rtap[ee] != std::numeric_limits<int>::min()) { bc_retap(st, e, st.bc_req_rtap[ee], st.bc_pos[ee]); retap = true; }
        if (retap) c.bc_retap.push_back(e);
    }
    c.bc_tap.assign(8 * static_cast<size_t>(n_bc_), 0.);
    for (size_t k = 0; k < 4 * static_cast<size_t>(n_bc_); ++k) {
        c.bc_tap[2 * k] = st.bc_tap[k].real();
        c.bc_tap[2 * k + 1] = st.bc_tap[k].imag();
    }
    c.bc_a_tap = st.bc_a_tap;
    c.bc_rho_tap = st.bc_rho_tap;
    c.bc_ctl = st.bc_ctl;
    c.bc_rctl = st.bc_rctl;
    // ShuntControl: the sections switched (once), then the controls
    c.sc_set.clear();
    for (int e = 0; e < n_sc_; ++e) if (st.sc_pending[static_cast<size_t>(e)]) c.sc_set.push_back(e);
    c.sc_b_new = st.sc_b_new;
    c.sc_ctl = st.sc_ctl;
    // the distributed slack's weights (NROuterInner::apply_controls): each unit's normalised
    // weight on its bus, the reference alone at 1 when none takes a share
    if (dist_) {
        c.slack_w.assign(layout_.slack_bus.size(), 0.);
        std::vector<size_t> units;
        const double total = ds_share(st, &units);
        if (total > 0.) {
            for (size_t k : units)
                c.slack_w[static_cast<size_t>(slack_idx_of_bus_[static_cast<size_t>(st.ds_unit_bus[k])])] +=
                    static_cast<double>(st.ds_units[k].weight) / total;
        } else {
            c.slack_w[static_cast<size_t>(ref_slack_idx_)] = 1.;
        }
    }
}

void Ls2gOuterModel::sc_set_sections(SlotState& st, int e, const std::vector<int>& counts) const
{
    const size_t ee = static_cast<size_t>(e);
    const auto& ids = sc_shunts_[ee];
    if (counts.size() != ids.size()) return;
    const auto& shunts = grid_->get_shunts();
    double b = 0.;
    for (size_t i = 0; i < counts.size(); ++i) {
        const int sh = ids[i];
        const int count = std::max(0, std::min(counts[i], shunts.get_max_section_count(sh)));
        st.sc_counts[ee][i] = count;
        if (shunts.get_status()[static_cast<size_t>(sh)]) b -= shunts.section_q(sh, count) / sn_;
    }
    st.sc_b_new[ee] = b;
    st.sc_pending[ee] = 1;
}

void Ls2gOuterModel::bc_retap(SlotState& st, int e, int rpos, int pos) const
{
    // BranchControl::_retap (set_tap / set_ratio_tap ignore a position off the table)
    const size_t ee = static_cast<size_t>(e);
    const int t = bc_trafo_[ee];
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    if (pos != st.bc_pos[ee] && (!ptc.has(t) || pos < ptc.low_tap(t) || pos > ptc.high_tap(t))) pos = st.bc_pos[ee];
    if (rpos != st.bc_rpos[ee] && (!rtc.has(t) || rpos < rtc.low_tap(t) || rpos > rtc.high_tap(t))) rpos = st.bc_rpos[ee];
    st.bc_pos[ee] = pos;
    st.bc_rpos[ee] = rpos;
    if (ptc.has(t)) st.bc_a_tap[ee] = ptc.alpha_at(t, pos);
    st.bc_rho_tap[ee] = trafos.ratio_at(t, rpos, pos);
    const auto tap = trafos.pi_block_at(t, rpos, pos, st.bc_a_tap[ee]);
    for (int k = 0; k < 4; ++k) st.bc_tap[4 * ee + static_cast<size_t>(k)] = cplx(tap[static_cast<size_t>(k)].real(), tap[static_cast<size_t>(k)].imag());
}

void Ls2gOuterModel::bc_block_at(const SlotState& st, int e, double a, double rho, cplx blk[4]) const
{
    // BranchControl::_block_at
    const size_t ee = static_cast<size_t>(e);
    if (st.bc_out[ee]) { for (int k = 0; k < 4; ++k) blk[k] = cplx(0., 0.); return; }
    const double d = bc_sign_[ee] * (a - st.bc_a_tap[ee]);
    const cplx rot(std::cos(d), std::sin(d));
    const double f = std::pow(rho / st.bc_rho_tap[ee], bc_rexp_[ee]);
    blk[0] = st.bc_tap[4 * ee] * f * f;
    blk[1] = st.bc_tap[4 * ee + 1] * f * rot;
    blk[2] = st.bc_tap[4 * ee + 2] * f * std::conj(rot);
    blk[3] = st.bc_tap[4 * ee + 3];
}

void Ls2gOuterModel::bc_current(const SlotState& st, const RowView& rv, int e, int side,
                                double& i_pu, double& di_da) const
{
    // BranchControl::current, on the block stamped now
    const size_t ee = static_cast<size_t>(e);
    cplx y[4];
    bc_block_at(st, e, rv.bc_a(e), rv.bc_rho(e), y);
    const cplx V1 = rv.V(bc_b1_[ee]);
    const cplx V2 = rv.V(bc_b2_[ee]);
    const cplx J(0., 1.);
    const cplx I = side == 1 ? y[0] * V1 + y[1] * V2 : y[2] * V1 + y[3] * V2;
    const cplx dI = side == 1 ? J * bc_sign_[ee] * y[1] * V2 : -J * bc_sign_[ee] * y[2] * V1;
    i_pu = std::abs(I);
    di_da = i_pu > 0. ? std::real(std::conj(I) * dI) / i_pu : 0.;
}

cplx Ls2gOuterModel::mis_bus(const SlotState& st, const RowView& rv, const SlotControls& c, int b) const
{
    cplx mis;
    if (rv.full == nullptr) {
        const int k = rv.w(b);
        mis = cplx(rv.d->s_re[k], rv.d->s_im[k]) - cplx(rv.d->sb_re[k], rv.d->sb_im[k]);
    } else {
        mis = rv.S(b) - rv.Sbus(b);
    }
    // the droop flows leaving the bus (Hvdc::adjust_mismatch)
    const bool hvdc_here = b >= 0 && b < n_bus_ && hvdc_at_bus_[static_cast<size_t>(b)];
    for (int e = 0; hvdc_here && e < layout_.n_hvdc; ++e) {
        const int b1 = hvdc_bus1_[static_cast<size_t>(e)], b2 = hvdc_bus2_[static_cast<size_t>(e)];
        if (b1 != b && b2 != b) continue;
        const double raw = hvdc_p0_[static_cast<size_t>(e)] +
                           hvdc_k_[static_cast<size_t>(e)] * (std::arg(rv.V(b1)) - std::arg(rv.V(b2)));
        double p1, p2;
        hvdc_flows(c.hvdc_status[static_cast<size_t>(e)], raw, hvdc_lf1_[static_cast<size_t>(e)],
                   hvdc_lf2_[static_cast<size_t>(e)], hvdc_r_[static_cast<size_t>(e)],
                   hvdc_pmax12_[static_cast<size_t>(e)], hvdc_pmax21_[static_cast<size_t>(e)], p1, p2);
        if (b1 == b) mis += cplx(p1, 0.);
        if (b2 == b) mis += cplx(p2, 0.);
    }
    // the controllers' reactive injection (VoltageControl::adjust_mismatch)
    for (int j : vc_at_bus_[static_cast<size_t>(b)]) {
        const double q_in_sbus = c.plan_held[static_cast<size_t>(j)] ? vc_q_plan_[static_cast<size_t>(j)] : 0.;
        mis -= cplx(0., rv.d->vc_q[j] - q_in_sbus);
    }
    // OuterLoopAlgo::_solve: published against the targets, not the frozen Q
    const bool frozen = (b >= 0 && b < n_bus_) ? (!st.frozen_at.empty() && st.frozen_at[static_cast<size_t>(b)])
                                               : !st.freeze_q.empty();
    if (frozen) {
        auto it = st.freeze_q.find(b);
        if (it != st.freeze_q.end()) mis += cplx(0., it->second);
    }
    // the distributed slack's share (MultiSlack::adjust_mismatch), which the fold moved into
    // the targets after the solve and so left in the mismatch
    if (st.sa_shared && b >= 0 && b < n_bus_) {
        const int k = slack_idx_of_bus_[static_cast<size_t>(b)];
        if (k >= 0 && static_cast<size_t>(k) < c.slack_w.size()) mis += cplx(st.sa * c.slack_w[static_cast<size_t>(k)], 0.);
    }
    return mis;
}

// =============================================================================
// OuterInjections::move_gen_target_p / move_storage_target_p
// =============================================================================
double Ls2gOuterModel::move_gen(SlotState& st, int g, int bus, double mw) const
{
    if (st.gen_target_p.empty()) st.gen_target_p.assign(static_cast<size_t>(n_gen_), NaN);
    const double delta = mw - gen_in_force(st, g);
    cplx ds(delta / sn_, 0.);
    const bool pq = !grid_->get_generators().get_voltage_regulator_on(g);
    const double q_before = pq ? grid_->gen_target_q_at_outer_target(g, gen_override(st, g)) : 0.;
    st.gen_target_p[static_cast<size_t>(g)] = mw;
    if (g >= 0 && static_cast<size_t>(g) < gen_rl_bus_.size() && !gen_rl_bus_[static_cast<size_t>(g)].empty()) {
        if (st.lim_dirty.empty()) st.lim_dirty.assign(static_cast<size_t>(n_bus_), 0);
        for (int b : gen_rl_bus_[static_cast<size_t>(g)]) st.lim_dirty[static_cast<size_t>(b)] = 1;
    }
    if (pq) ds += cplx(0., (grid_->gen_target_q_at_outer_target(g, gen_override(st, g)) - q_before) / sn_);
    add_injection(st, bus, ds);
    return delta;
}

double Ls2gOuterModel::move_storage(SlotState& st, int s, int bus, double mw) const
{
    if (st.storage_target_p.empty()) st.storage_target_p.assign(static_cast<size_t>(n_sto_), NaN);
    const double delta = sto_in_force(st, s) - mw;   // load convention
    st.storage_target_p[static_cast<size_t>(s)] = mw;
    add_injection(st, bus, cplx(delta / sn_, 0.));
    return delta;
}

// =============================================================================
// DistributedSlackLoop::_check
// =============================================================================
int Ls2gOuterModel::check_ds(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    // with no unit at all, the Newton shares nothing: the reference bus' mismatch is checked
    // against the threshold as around a single-slack Newton
    if (st.ds_in_newton && !st.ds_units.empty()) return check_ds_in_newton(st, L, rv);
    // single slack: slack_absorbed is 0, the whole residual of the slack bus
    const double mismatch = mis_bus(st, rv, st.ctl, slack_bus_).real() * sn_;
    const bool triggered = std::abs(mismatch) > L.ds_mismatch_mw && std::abs(mismatch) > L.ds_eps_mw;
    record(st, L.name, "DISTRIBUTE", triggered, -1, mismatch, std::max(L.ds_mismatch_mw, L.ds_eps_mw));
    if (!triggered) return STABLE;
    double remaining = mismatch;
    for (size_t k = 0; k < st.ds_units.size(); ++k) remaining += st.ds_current_mw[k] - st.ds_units[k].injection_mw;
    std::vector<ls2g::real_type> new_mw;
    std::vector<char> saturated;
    const ls2g::slack_redistribution::Report report =
        ls2g::slack_redistribution::distribute(st.ds_units, remaining, L.ds_eps_mw, new_mw, saturated);
    const double residue = st.ds_units.empty() ? remaining : static_cast<double>(report.not_distributed_mw);
    if (L.ds_fail && std::abs(residue) > L.ds_eps_mw) {
        record(st, L.name, "FAIL_RESIDUE", true, -1, residue, L.ds_eps_mw);
        return FAILED;
    }
    double moved = 0.;
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        const auto& u = st.ds_units[k];
        moved += std::abs(u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR
            ? move_gen(st, u.el_id, st.ds_unit_bus[k], new_mw[k])
            : move_storage(st, u.el_id, st.ds_unit_bus[k], -new_mw[k]));
        st.ds_current_mw[k] = new_mw[k];
    }
    const bool unstable = moved > L.ds_moved_fraction * L.ds_eps_mw;
    record(st, L.name, "UNITS_MOVED", unstable, -1, moved, L.ds_moved_fraction * L.ds_eps_mw);
    if (prof_.on && !unstable && moved > 0.) ++prof_.n[P_DS_STABLE_MOVED];
    return unstable ? UNSTABLE : STABLE;
}

// DistributedSlackLoop::_check_in_newton
int Ls2gOuterModel::check_ds_in_newton(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    // where the solve left the units: the fold moved their targets by their share
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        const auto& u = st.ds_units[k];
        st.ds_current_mw[k] = u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR
            ? gen_in_force(st, u.el_id) : -sto_in_force(st, u.el_id);
    }
    // the solve just checked ran with the units parked as the last check left them: this loop
    // is checked again only after a solve
    if (keep_adjoint_) st.ds_mark_solved();
    // the same cumulative sharing from the initial targets as around a single-slack Newton;
    // what the reference bus kept is not nothing only when every unit is parked
    double mismatch = mis_bus(st, rv, st.ctl, slack_bus_).real() * sn_;
    if (std::isnan(mismatch)) mismatch = 0.;
    double remaining = mismatch;
    for (size_t k = 0; k < st.ds_units.size(); ++k) remaining += st.ds_current_mw[k] - st.ds_units[k].injection_mw;
    std::vector<ls2g::real_type> new_mw;
    std::vector<char> saturated;
    const ls2g::slack_redistribution::Report report =
        ls2g::slack_redistribution::distribute(st.ds_units, remaining, L.ds_eps_mw, new_mw, saturated);
    // every unit at a bound: none of them shares any more, the reference bus takes the rest
    if (report.all_saturated) saturated.assign(st.ds_units.size(), 1);
    const double residue = st.ds_units.empty() ? remaining : static_cast<double>(report.not_distributed_mw);
    if (L.ds_fail && std::abs(residue) > L.ds_eps_mw) {
        record(st, L.name, "FAIL_RESIDUE", true, -1, residue, L.ds_eps_mw);
        return FAILED;
    }
    double moved = 0.;
    for (size_t k = 0; k < st.ds_units.size(); ++k) {
        const auto& u = st.ds_units[k];
        const bool is_gen = u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR;
        // a unit at a bound leaves the Newton's share; one back inside its bounds rejoins it
        if (saturated[k] != st.ds_parked[k])
            record(st, L.name, saturated[k] ? "PARK" : "UNPARK", true, u.el_id, st.ds_current_mw[k], new_mw[k]);
        moved += std::abs(is_gen ? move_gen(st, u.el_id, st.ds_unit_bus[k], new_mw[k])
                                 : move_storage(st, u.el_id, st.ds_unit_bus[k], -new_mw[k]));
        st.ds_current_mw[k] = new_mw[k];
    }
    st.ds_parked.assign(saturated.begin(), saturated.end());
    st.ds_parked_high.assign(st.ds_units.size(), remaining > 0. ? 1 : 0);
    // the units moved only when some unit was pushed past a bound, or came back
    const double threshold = L.ds_moved_fraction * L.ds_eps_mw;
    bool unstable = moved > threshold;
    record(st, L.name, "UNITS_MOVED", unstable, -1, moved, threshold);
    // ... or when the Newton's share, moved into the targets after the solve, took a unit's
    // target Q with it: the solve has not seen that one
    const double unsolved_q = st.unsolved_q_mvar;
    st.unsolved_q_mvar = 0.;
    if (unsolved_q > threshold) {
        record(st, L.name, "TARGET_Q_FOLLOWED", true, -1, unsolved_q, threshold);
        unstable = true;
    }
    // re-solved with the new weights; a STABLE check's change is not (unless another loop's
    // re-solve comes after, and this loop is then checked again)
    if (unstable && keep_adjoint_) st.ds_mark_solved();
    return unstable ? UNSTABLE : STABLE;
}

// =============================================================================
// HvdcAcEmulationLimitsLoop::_check
// =============================================================================
int Ls2gOuterModel::check_hvdc(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    ls2g::RealVect& Va = hvdc_va_scratch_;   // all 0 between calls
    for (const auto& line : st.hvdc_plan.lines) {
        Va(line.bus1_solver) = std::arg(rv.V(line.bus1_solver));
        Va(line.bus2_solver) = std::arg(rv.V(line.bus2_solver));
    }
    std::vector<ls2g::LimitViolation> trigger;
    ls2g::hvdc_p_check::check_hvdc_p_violations(st.hvdc_plan, Va, 0., st.rowv.masked, trigger);
    for (const auto& line : st.hvdc_plan.lines) {
        Va(line.bus1_solver) = 0.;
        Va(line.bus2_solver) = 0.;
    }
    if (trigger.empty()) return STABLE;
    bool changed = false;
    for (const auto& v : trigger) {
        if (v.element_type != ls2g::ViolationElementType::HVDC) continue;
        int regime;
        if (v.violation_type == ls2g::LimitViolationType::HIGH_P) regime = v.side == 1 ? 1 : -1;
        else if (v.violation_type == ls2g::LimitViolationType::HVDC_AC_EMULATION_RELEASE) regime = 0;
        else continue;
        bool acted = false;
        for (auto& line : st.hvdc_plan.lines) {
            if (line.hvdc_id != v.element_id || line.frozen_dir == regime) continue;
            const auto it = std::find(hvdc_id_.begin(), hvdc_id_.end(), line.hvdc_id);
            if (it == hvdc_id_.end()) continue;   // no regime control for it
            line.frozen_dir = regime;
            st.hvdc_regime[static_cast<size_t>(it - hvdc_id_.begin())] = regime;
            changed = true;
            acted = true;
        }
        record(st, L.name, regime == 0 ? "RELEASE" : "SATURATE", acted, v.element_id, v.value, v.limit);
    }
    return changed ? UNSTABLE : STABLE;
}

// =============================================================================
// VoltageMonitoringLoop::_check
// =============================================================================
int Ls2gOuterModel::check_vm(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    ls2g::svc_standby_check::SvcStandbyPlan idle;
    for (const auto& e : monitors_.svcs) {
        const int c = svc_ctrl_[static_cast<size_t>(e.svc_id)];
        if (c >= 0 && std::isfinite(st.svc_on[static_cast<size_t>(c)])) continue;   // switched on
        if (c >= 0 && row_trips_ctrl(st, c)) continue;   // the row trips it: no automaton left
        idle.svcs.push_back(e);
    }
    ls2g::CplxVect& V = vm_v_scratch_;   // all 0 between calls
    for (const auto& e : idle.svcs) {
        V(e.reg_bus_solver) = rv.V(e.reg_bus_solver);
        V(e.svc_bus_solver) = rv.V(e.svc_bus_solver);
    }
    std::vector<ls2g::LimitViolation> trigger;
    ls2g::svc_standby_check::check_svc_standby_violations(idle, V, 0., st.rowv.masked, trigger);
    for (const auto& e : idle.svcs) {
        V(e.reg_bus_solver) = ls2g::cplx_type(0., 0.);
        V(e.svc_bus_solver) = ls2g::cplx_type(0., 0.);
    }
    if (trigger.empty()) return STABLE;
    const auto& svcs = grid_->get_svcs();
    bool changed = false;
    for (const auto& v : trigger) {
        if (v.element_type != ls2g::ViolationElementType::SVC) continue;
        double target;
        if (v.violation_type == ls2g::LimitViolationType::LOW_VOLTAGE_SVC_STANDBY) target = svcs.get_standby_low_target_vm_pu(v.element_id);
        else if (v.violation_type == ls2g::LimitViolationType::HIGH_VOLTAGE_SVC_STANDBY) target = svcs.get_standby_high_target_vm_pu(v.element_id);
        else continue;
        const int c = (v.element_id >= 0 && static_cast<size_t>(v.element_id) < svc_ctrl_.size())
                      ? svc_ctrl_[static_cast<size_t>(v.element_id)] : -1;
        const bool can = std::isfinite(target) && c >= 0;
        record(st, L.name, "SWITCH_ON", can, v.element_id, v.value, v.limit);
        if (!can) continue;
        st.svc_on[static_cast<size_t>(c)] = target;
        st.vm_switched_on.emplace_back(v.element_id, target);
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

// =============================================================================
// ReactiveLimitsLoop
// =============================================================================
double Ls2gOuterModel::rl_bus_q(const SlotState& st, const RowView& rv, const RlBus& bus) const
{
    double q = mis_bus(st, rv, st.ctl, bus.bus_solver).imag();
    if (!bus.local)
        for (int pos : st.rl->plan->buses[static_cast<size_t>(bus.entry)].ctrl_pos) q += rv.d->vc_q[pos];
    return q * sn_;
}

void Ls2gOuterModel::rl_limits(const SlotState& st, const RowView* rv, const RlBus& bus,
                               double& q_min, double& q_max) const
{
    // (bus.bus_solver is its entry's bus; a local bus has no controller and no SVC)
    const bool at_base = rl_at_base(st, bus);
    if (at_base) {
        q_min = bus.base_min_q;   // none of its generators moved: the base sums, bit for bit
        q_max = bus.base_max_q;
        if (prof_.on) ++prof_.n[P_RL_BASE];
        if (bus.local || rv == nullptr) return;
    }
    const auto& entry = st.rl->plan->buses[static_cast<size_t>(bus.entry)];
    if (!at_base) {
        if (prof_.on) ++prof_.n[(bus.has_base && st.lim_base) ? P_RL_RECOMPUTE : P_RL_NO_BASE];
        q_min = bus.min_q;
        q_max = bus.max_q;
        for (int gid : entry.gen_ids) {
            double lo, hi;
            gen_limits(gid, gen_override(st, gid), lo, hi);
            q_min += lo;
            q_max += hi;
        }
    }
    if (entry.svc_ids.empty() || rv == nullptr) return;
    const auto& svcs = grid_->get_svcs();
    const double v2_sn = std::norm(rv->V(bus.bus_solver)) * sn_;
    for (int sid : entry.svc_ids) {
        q_min += svcs.get_b_min(sid) * v2_sn;
        q_max += svcs.get_b_max(sid) * v2_sn;
    }
}

double Ls2gOuterModel::rl_controller_limit(const SlotState& st, const RowView& rv, int c, bool max) const
{
    const int el = vc_elem_[static_cast<size_t>(c)];
    switch (vc_kind_[static_cast<size_t>(c)]) {
        case ls2g::VoltageControlSolverData::GEN: {
            double lo, hi;
            gen_limits(el, gen_override(st, el), lo, hi);
            return max ? hi : lo;
        }
        case ls2g::VoltageControlSolverData::SVC: {
            const double v2_sn = std::norm(rv.V(vc_bus_[static_cast<size_t>(c)])) * sn_;
            return (max ? grid_->get_svcs().get_b_max(el) : grid_->get_svcs().get_b_min(el)) * v2_sn;
        }
        case ls2g::VoltageControlSolverData::HVDC_SIDE_1:
        case ls2g::VoltageControlSolverData::HVDC_SIDE_2: {
            const int side = vc_kind_[static_cast<size_t>(c)] == ls2g::VoltageControlSolverData::HVDC_SIDE_1 ? 1 : 2;
            return max ? grid_->get_dclines().get_station_max_q_mvar(el, side)
                       : grid_->get_dclines().get_station_min_q_mvar(el, side);
        }
        default:
            return 0.;
    }
}

std::vector<char> Ls2gOuterModel::rl_groups_holding(const SlotState& st) const
{
    std::vector<char> holding(static_cast<size_t>(layout_.n_grp), 0), sloped(holding.size(), 0);
    for (int c = 0; c < layout_.n_ctrl; ++c) {
        const size_t cc = static_cast<size_t>(c);
        const size_t g = static_cast<size_t>(vc_group_[cc]);
        if (vc_slope_[cc] != 0.) sloped[g] = 1;
        if (std::isfinite(st.held_q[cc])) continue;          // held at a limit by this loop
        if (vc_plan_held_[cc]) {                             // a frozen regulator, or a monitor not switched on
            const bool switched_on = vc_kind_[cc] == ls2g::VoltageControlSolverData::SVC && std::isfinite(st.svc_on[cc]);
            if (!switched_on) continue;
        }
        holding[g] = 1;
    }
    for (size_t g = 0; g < holding.size(); ++g) holding[g] = holding[g] && !sloped[g];
    return holding;
}

void Ls2gOuterModel::rl_freeze(SlotState& st, const RowView& rv, size_t k, double q_mvar, int state) const
{
    const RlBus& bus = st.rl->buses[k];
    RlBusState& rs = st.rl_state[k];
    const int ki = static_cast<int>(k);
    // a generator frozen at a limit of its curve follows it: what the adjoint adds to its gen_p
    // gradient (the robust mode's target Q does not move with P)
    st.rl_slopes.erase(ki);
    if (!bus.local) {
        const auto& entry = st.rl->plan->buses[static_cast<size_t>(bus.entry)];
        for (int c : entry.ctrl_pos) {
            double q = rl_controller_limit(st, rv, c, state > 0);
            if (vc_kind_[static_cast<size_t>(c)] == ls2g::VoltageControlSolverData::GEN) {
                const int g = vc_elem_[static_cast<size_t>(c)];
                if (rs.realistic) {
                    q = grid_->get_generators().get_target_q()(g);   // the robust mode's
                } else if (keep_adjoint_) {
                    const double slope = rl_gen_limit_slope(st, g, state > 0);
                    if (slope != 0.) st.rl_slopes[ki].push_back({{g, c}, slope / sn_});
                }
            }
            st.held_q[static_cast<size_t>(c)] = q / sn_;
        }
        rs.state = state;
        rs.frozen_q = q_mvar;
        return;
    }
    if (!rs.realistic && keep_adjoint_)
        for (int g : st.rl->plan->buses[static_cast<size_t>(bus.entry)].gen_ids) {
            const double slope = rl_gen_limit_slope(st, g, state > 0);
            if (slope != 0.) st.rl_slopes[ki].push_back({{g, -1}, slope / sn_});
        }
    if (bus.pin < 0)
        throw std::runtime_error("outer model: the controller bus " + std::to_string(bus.bus_solver) +
                                 " has no reserved Vm column (is the grid's algorithm an NROuter_* one, "
                                 "solved with its loops?)");
    // Sbus(b).imag = imag(Sbus_target(b)) + q / sn, the bus PQ
    freeze_set(st, bus.bus_solver, q_mvar / sn_);
    st.pq[static_cast<size_t>(bus.pin)] = 1;
    rs.state = state;
    rs.frozen_q = q_mvar;
}

void Ls2gOuterModel::rl_release(SlotState& st, size_t k) const
{
    const RlBus& bus = st.rl->buses[k];
    RlBusState& rs = st.rl_state[k];
    const int ki = static_cast<int>(k);
    if (!bus.local) {
        const auto& entry = st.rl->plan->buses[static_cast<size_t>(bus.entry)];
        for (int c : entry.ctrl_pos) st.held_q[static_cast<size_t>(c)] = NaN;
        rs.state = 0;
        rs.realistic = false;
        rs.frozen_q = 0.;
        st.rl_slopes.erase(ki);
        return;
    }
    st.rl_slopes.erase(ki);
    freeze_erase(st, bus.bus_solver);
    st.pq[static_cast<size_t>(bus.pin)] = 0;
    // a pinned row keeps the magnitude the bus has: back at its set-point
    reset_vm(st, bus.bus_solver, bus.target_vm);
    rs.state = 0;
    rs.realistic = false;
    rs.frozen_q = 0.;
}

int Ls2gOuterModel::check_rl(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    struct Switch { int k; ls2g::LimitViolationType type; double value; double limit; bool realistic; bool group_holds; };
    std::vector<Switch> to_pq, to_pv, kept;
    std::vector<int> moved;
    int remaining_pv = 0;
    // ---- _evaluate ----------------------------------------------------------------
    const double eps = L.rl_max_mismatch * L.rl_base_mva;
    const double v_low = L.rl_vmin * L.rl_margin;
    const double v_high = L.rl_vmax / L.rl_margin;
    const std::vector<char> holding = rl_groups_holding(st);
    for (size_t k = 0; k < st.rl_state.size(); ++k) {
        const RlBus& bus = st.rl->buses[k];
        const RlBusState& rs = st.rl_state[k];
        const int ki = static_cast<int>(k);
        // a bus whose voltage control another loop suspended (TransformerVoltageControl)
        if (!st.suspended.empty() && st.suspended.count(bus.bus_solver)) continue;
        double target_vm = bus.target_vm;
        if (bus.monitor_svc >= 0) {
            const int c = svc_ctrl_[static_cast<size_t>(bus.monitor_svc)];
            if (c < 0 || !std::isfinite(st.svc_on[static_cast<size_t>(c)])) continue;   // still idle
            target_vm = st.svc_on[static_cast<size_t>(c)];
        }
        double q_min, q_max;
        rl_limits(st, &rv, bus, q_min, q_max);
        if (rs.state == 0) {
            const double q = rl_bus_q(st, rv, bus);
            if (q < q_min - eps) {
                to_pq.push_back(Switch{ki, ls2g::LimitViolationType::LOW_Q, q, q_min, false, false});
            } else if (q > q_max + eps) {
                to_pq.push_back(Switch{ki, ls2g::LimitViolationType::HIGH_Q, q, q_max, false, false});
            } else if (L.rl_robust && bus.reg_bus_solver != bus.bus_solver) {
                const double v = rv.vm(bus.bus_solver);
                if (v < v_low)       to_pq.push_back(Switch{ki, ls2g::LimitViolationType::LOW_Q, q, bus.target_q, true, false});
                else if (v > v_high) to_pq.push_back(Switch{ki, ls2g::LimitViolationType::HIGH_Q, q, bus.target_q, true, false});
                else ++remaining_pv;
            } else {
                ++remaining_pv;
            }
            continue;
        }
        const bool held_by_group = bus.group >= 0 && holding[static_cast<size_t>(bus.group)];
        const double vm_read = rv.vm(bus.reg_bus_solver);
        const double vm = held_by_group ? target_vm : vm_read;
        const double vn = bus.nominal_v;
        const auto release = rs.state < 0 ? ls2g::LimitViolationType::LOW_VOLTAGE_AT_MIN_Q
                                           : ls2g::LimitViolationType::HIGH_VOLTAGE_AT_MAX_Q;
        if (rs.state < 0 ? vm < target_vm : vm > target_vm) {
            to_pv.push_back(Switch{ki, release, vm * vn, target_vm * vn, false, false});
            continue;
        }
        kept.push_back(Switch{ki, release, vm_read * vn, target_vm * vn, false, held_by_group});
        const double q_lim = rs.state < 0 ? q_min : q_max;
        if (!rs.realistic && std::abs(q_lim - rs.frozen_q) > eps) moved.push_back(ki);
    }
    // ---- _check ---------------------------------------------------------------------
    bool changed = false;
    auto rec = [&](const char* action, bool taken, const Switch& sw) {
        record(st, L.name, action, taken, st.rl->buses[static_cast<size_t>(sw.k)].bus_solver, sw.value, sw.limit);
    };
    for (const Switch& sw : kept) rec(sw.group_holds ? "KEPT_PQ_GROUP_HOLDS" : "KEPT_PQ", false, sw);
    if (!to_pq.empty() && remaining_pv == 0) {
        auto stronger = [&](const Switch& a, const Switch& b) {
            const RlBus& ba = st.rl->buses[static_cast<size_t>(a.k)];
            const RlBus& bb = st.rl->buses[static_cast<size_t>(b.k)];
            const double pa = st.rl_state[static_cast<size_t>(a.k)].target_p;
            const double pb = st.rl_state[static_cast<size_t>(b.k)].target_p;
            if (ba.nominal_v != bb.nominal_v) return ba.nominal_v > bb.nominal_v;
            if (pa != pb) return pa > pb;
            return ba.bus_solver < bb.bus_solver;
        };
        const auto strongest = std::min_element(to_pq.begin(), to_pq.end(), stronger);
        rec("KEPT_PV_STRONGEST", false, *strongest);
        to_pq.erase(strongest);
    }
    for (const Switch& sw : to_pq) {
        rec(sw.realistic ? "PV_TO_PQ_UNREALISTIC" : "PV_TO_PQ", true, sw);
        const size_t k = static_cast<size_t>(sw.k);
        const RlBus& bus = st.rl->buses[k];
        RlBusState& rs = st.rl_state[k];
        rs.realistic = sw.realistic;
        rl_freeze(st, rv, k, sw.limit, sw.type == ls2g::LimitViolationType::LOW_Q ? -1 : +1);
        ++rs.nb_pv_pq;
        if (L.rl_robust && bus.reg_bus_solver != bus.bus_solver) {
            const double v = rv.vm(bus.bus_solver);
            if (sw.realistic || v < v_low || v > v_high) reset_vm(st, bus.bus_solver, L.rl_restart);
        }
        changed = true;
    }
    for (const Switch& sw : to_pv) {
        const size_t k = static_cast<size_t>(sw.k);
        if (st.rl_state[k].nb_pv_pq >= L.rl_max_switch) {
            rec("KEPT_PQ_MAX_SWITCH", false, sw);
            continue;
        }
        rec("PQ_TO_PV", true, sw);
        rl_release(st, k);
        changed = true;
    }
    for (int k : moved) {
        const size_t ks = static_cast<size_t>(k);
        const RlBus& bus = st.rl->buses[ks];
        const RlBusState& rs = st.rl_state[ks];
        double q_min, q_max;
        rl_limits(st, &rv, bus, q_min, q_max);
        record(st, L.name, "LIMIT_MOVED", true, bus.bus_solver, rs.state < 0 ? q_min : q_max, rs.frozen_q);
        rl_freeze(st, rv, ks, rs.state < 0 ? q_min : q_max, rs.state);
        changed = true;
    }
    return changed ? UNSTABLE : STABLE;
}

int Ls2gOuterModel::check(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const
{
    switch (L.kind) {
        case Kind::DistributedSlack:  { ProfScope p(prof_, P_CHECK_DS);   return check_ds(st, L, rv); }
        case Kind::HvdcLimits:        { ProfScope p(prof_, P_CHECK_HVDC); return check_hvdc(st, L, rv); }
        case Kind::VoltageMonitoring: { ProfScope p(prof_, P_CHECK_VM);   return check_vm(st, L, rv); }
        case Kind::ReactiveLimits:    { ProfScope p(prof_, P_CHECK_RL);   return check_rl(st, L, rv); }
        case Kind::PhaseControl:      { ProfScope p(prof_, P_CHECK_PC);   return check_pc(st, L, rv, iteration); }
        case Kind::TransformerVoltageControl: { ProfScope p(prof_, P_CHECK_TVC); return check_tvc(st, L, rv); }
        case Kind::ShuntVoltageControl: { ProfScope p(prof_, P_CHECK_SVC); return check_svc(st, L, rv, iteration); }
    }
    return STABLE;
}

// =============================================================================
// TransformerVoltageControlLoop (AFTER_GENERATOR_VOLTAGE_CONTROL)
// =============================================================================
void Ls2gOuterModel::tvc_set_on(SlotState& st, int t, bool on) const
{
    st.tvc_enabled[static_cast<size_t>(t)] = on ? 1 : 0;
    const int e = bc_entry(t);
    if (e >= 0 && bc_ratio_[static_cast<size_t>(e)]) st.bc_rctl[static_cast<size_t>(e)] = on ? 1 : 0;
}

int Ls2gOuterModel::tvc_closest_tap(const SlotState& st, int t, double value) const
{
    // PiModelArray.roundR1ToClosestTap: the current position unless another is strictly closer;
    // at the phase tap the solve has it at (a phase shifter's, the grid's otherwise)
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    const int e = bc_entry(t);
    const int ppos = (e >= 0 && bc_phase_[static_cast<size_t>(e)]) ? st.bc_pos[static_cast<size_t>(e)]
                                                                  : (ptc.has(t) ? ptc.position(t) : 0);
    int best = st.bc_rpos[static_cast<size_t>(e)];
    double best_distance = std::abs(value - trafos.ratio_at(t, best, ppos));
    for (int pos = rtc.low_tap(t); pos <= rtc.high_tap(t); ++pos) {
        const double distance = std::abs(value - trafos.ratio_at(t, pos, ppos));
        if (distance < best_distance) { best = pos; best_distance = distance; }
    }
    return best;
}

std::vector<TvcBranch> Ls2gOuterModel::tvc_closed_branches(const ls2g::OuterGridView& v) const
{
    std::vector<TvcBranch> res;
    int b1, b2;
    for (int el = 0; el < n_line_; ++el)
        if (v.line_closed(el, b1, b2)) res.push_back(TvcBranch{b1, b2, -1});
    for (int el = 0; el < n_trafo_; ++el)
        if (v.trafo_closed(el, b1, b2)) res.push_back(TvcBranch{b1, b2, el});
    return res;
}

double Ls2gOuterModel::tvc_limit(const SlotState& st, const LoopDecl& L) const
{
    const double p = L.tvc->params().max_controlled_nominal_voltage;
    if (p >= 0.) return p;
    // GeneratorVoltageControlManager.computeDefaultMinNominalVoltageLimit
    double res = std::numeric_limits<ls2g::real_type>::min();
    const ls2g::OuterGridView v = view(st);
    const auto vn_kv = grid_->get_bus_vn_kv();
    const auto& to_me = v.id_solver_to_me();
    for (const auto& gr : st.tvc_groups) {
        bool valid = true;
        for (int t : gr.trafos) {
            int b1, b2;
            if (!v.trafo_closed(t, b1, b2) || vn_kv(to_me[b1].cast_int()) == vn_kv(to_me[b2].cast_int())) {
                valid = false;
                break;
            }
        }
        if (valid) res = std::max(res, static_cast<double>(vn_kv(gr.bus_grid)));
    }
    return res;
}

bool Ls2gOuterModel::tvc_voltage_control_off(const SlotState& st, int bus) const
{
    // a controller bus whose voltage control a loop took away: switched PQ, or suspended
    const int k = (bus >= 0 && bus < n_bus_) ? pin_of_bus_[static_cast<size_t>(bus)] : -1;
    return (k >= 0 && st.pq[static_cast<size_t>(k)]) || st.suspended.count(bus) > 0;
}

void Ls2gOuterModel::tvc_freeze(SlotState& st, const RowView& rv, double limit,
                                const std::vector<TvcBranch>& closed) const
{
    // disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage: a controller bus frozen at
    // its bus' injection, which OpenLoadFlow takes as the generation (its load counted twice)
    const ls2g::OuterGridView v = view(st);
    const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();
    const auto& to_me = v.id_solver_to_me();
    const auto vn_kv = grid_->get_bus_vn_kv();
    const auto& trafos = grid_->get_trafos();
    std::vector<char> on_controller(static_cast<size_t>(vn_kv.size()), 0);
    for (int t = 0; t < n_trafo_; ++t) {
        if (!st.tvc_controller[static_cast<size_t>(t)]) continue;
        for (const int b : {trafos.get_bus_side_1(t).cast_int(), trafos.get_bus_side_2(t).cast_int()})
            if (b >= 0 && b < static_cast<int>(vn_kv.size())) on_controller[static_cast<size_t>(b)] = 1;
    }
    std::vector<double> min_connected(static_cast<size_t>(to_me.size()), NaN);
    for (const TvcBranch& br : closed) {
        const double vv = std::max(vn_kv(to_me[br.bus1].cast_int()), vn_kv(to_me[br.bus2].cast_int()));
        for (const int b : {br.bus1, br.bus2}) {
            double& m = min_connected[static_cast<size_t>(b)];
            m = std::isnan(m) ? vv : std::min(m, vv);
        }
    }
    std::vector<double> load_q(static_cast<size_t>(n_bus_), 0.);
    const int nb_loads = grid_->get_loads().nb();
    for (int l = 0; l < nb_loads; ++l) {
        const int b = v.load_bus_solver(l);
        if (b >= 0 && b < n_bus_) load_q[static_cast<size_t>(b)] += v.load_target_q(l) / sn_;
    }
    const auto& buses = st.rl->plan->buses;
    for (size_t k = 0; k < st.rl->n_bus_q; ++k) {
        const auto& entry = buses[static_cast<size_t>(st.rl->buses[k].entry)];
        const bool local = entry.ctrl_pos.empty() && entry.svc_ids.empty();
        int controlled_grid = entry.bus_grid;
        if (!local) controlled_grid = to_me[ctrl.reg_bus(ctrl.group(entry.ctrl_pos.front()))].cast_int();
        if (controlled_grid < 0 || vn_kv(controlled_grid) > limit) continue;
        const int b = entry.bus_solver;
        if (tvc_voltage_control_off(st, b)) continue;
        if (!local) {
            bool held = false;
            for (int c : entry.ctrl_pos) held = held || std::isfinite(st.held_q[static_cast<size_t>(c)]);
            if (held) continue;
        }
        // GeneratorVoltageControlManager.hasStepUpTransformers
        bool step_up;
        if (!entry.station_ids.empty()) step_up = true;
        else if (on_controller[static_cast<size_t>(entry.bus_grid)]) step_up = false;
        else {
            const double min_v = min_connected[static_cast<size_t>(entry.bus_solver)];
            step_up = min_v > vn_kv(entry.bus_grid) && min_v > limit;
        }
        if (step_up) continue;
        double target_vm = rv.vm(b);
        if (!entry.gen_ids.empty()) target_vm = v.gen_target_vm(entry.gen_ids.front());
        else if (!entry.storage_ids.empty()) target_vm = v.storage_target_vm(entry.storage_ids.front());
        const double load = load_q[static_cast<size_t>(b)];
        if (local) {
            // the regulating units' output now (they are out of Sbus): the bus' residual
            const double q_gen = mis_bus(st, rv, st.ctl, b).imag();
            freeze_set(st, b, q_gen - load);
            const int pin = pin_of_bus_[static_cast<size_t>(b)];
            if (pin >= 0) st.pq[static_cast<size_t>(pin)] = 1;
        } else {
            bool first = true;
            for (int c : entry.ctrl_pos) {
                double q = rv.d->vc_q[c];
                if (first) q -= load;
                first = false;
                st.held_q[static_cast<size_t>(c)] = q;
            }
        }
        st.suspended.insert(b);
        // (the plan's entry, not k: a row masking buses shares the grid's plan, its buses fewer)
        st.tvc_frozen.push_back(TvcFrozen{st.rl->buses[k].entry, b, local, target_vm});
    }
}

void Ls2gOuterModel::tvc_release(SlotState& st) const
{
    for (const TvcFrozen& f : st.tvc_frozen) {
        if (f.local) {
            freeze_erase(st, f.bus_solver);
            const int pin = pin_of_bus_[static_cast<size_t>(f.bus_solver)];
            if (pin >= 0) st.pq[static_cast<size_t>(pin)] = 0;
            reset_vm(st, f.bus_solver, f.target_vm);
        } else {
            for (int c : st.rl->plan->buses[static_cast<size_t>(f.entry)].ctrl_pos) st.held_q[static_cast<size_t>(c)] = NaN;
        }
        st.suspended.erase(f.bus_solver);
    }
    st.tvc_frozen.clear();
}

void Ls2gOuterModel::tvc_fix_controls(SlotState& st, const std::vector<TvcBranch>& closed) const
{
    // LfNetwork.fixTransformerVoltageControls: a transformer whose other side, once every
    // transformer switched on is taken out, keeps no PV bus is switched off
    std::vector<int> parent(static_cast<size_t>(n_bus_));
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    auto join = [&](int a, int b) {
        if (a < 0 || b < 0 || a >= n_bus_ || b >= n_bus_) return;
        parent[static_cast<size_t>(root(a))] = root(b);
    };
    for (const TvcBranch& br : closed) {
        if (br.trafo >= 0 && st.tvc_enabled[static_cast<size_t>(br.trafo)]) continue;
        join(br.bus1, br.bus2);
    }
    std::set<int> with_pv;
    const auto& buses = st.rl->plan->buses;
    for (size_t k = 0; k < st.rl->n_bus_q; ++k) {
        const auto& entry = buses[static_cast<size_t>(st.rl->buses[k].entry)];
        if (!(entry.ctrl_pos.empty() && entry.svc_ids.empty())) continue;
        if (tvc_voltage_control_off(st, entry.bus_solver)) continue;
        with_pv.insert(root(entry.bus_solver));
    }
    const ls2g::OuterGridView v = view(st);
    for (const auto& gr : st.tvc_groups)
        for (int t : gr.trafos) {
            if (!st.tvc_enabled[static_cast<size_t>(t)]) continue;
            int b1, b2;
            if (!v.trafo_closed(t, b1, b2)) continue;
            int other;
            if (gr.bus_solver == b1) other = b2;
            else if (gr.bus_solver == b2) other = b1;
            else continue;
            if (!with_pv.count(root(other))) tvc_set_on(st, t, false);
        }
}

int Ls2gOuterModel::check_tvc(SlotState& st, const LoopDecl& L, const RowView& rv) const
{
    if (st.tvc_step == 2 || st.tvc_groups.empty()) return STABLE;
    // nothing to do without a transformer the inner algorithm can act on (a reserved control)
    auto reserved = [&](int t) { const int e = bc_entry(t); return e >= 0 && bc_ratio_[static_cast<size_t>(e)]; };
    bool any = false;
    for (const auto& gr : st.tvc_groups) for (int t : gr.trafos) any = any || reserved(t);
    if (!any) return STABLE;
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    const auto& tp = L.tvc->params();
    auto ratio_now = [&](int t) { return rv.bc_rho(bc_entry(t)); };
    auto phase_pos = [&](int t) {
        const int e = bc_entry(t);
        return (e >= 0 && bc_phase_[static_cast<size_t>(e)]) ? st.bc_pos[static_cast<size_t>(e)]
                                                             : (ptc.has(t) ? ptc.position(t) : 0);
    };

    if (st.tvc_step == 0) {
        bool need_run = false;
        for (const auto& gr : st.tvc_groups) {
            if (gr.hidden) continue;
            const double v = rv.vm(gr.bus_solver);
            const bool outside = !(std::abs(gr.target - v) <= gr.half_deadband);
            record(st, L.name, "SWITCH_ON", outside, gr.bus_solver, std::abs(gr.target - v), gr.half_deadband);
            if (!outside) continue;
            for (int t : gr.trafos) {
                if (!reserved(t)) continue;   // handled_ratio
                tvc_set_on(st, t, true);
                need_run = true;
            }
        }
        // TransformerRatioManager: the ratios of the transformers switched on
        for (const auto& gr : st.tvc_groups) {
            std::vector<int> on;
            for (int t : gr.trafos) if (st.tvc_enabled[static_cast<size_t>(t)]) on.push_back(t);
            if (on.empty()) continue;
            double a = 0., b = 0., sum_min = 0., sum_max = 0.;
            for (int t : on) {
                TvcRatio& r = st.tvc_ratios[static_cast<size_t>(t)];
                const int ppos = ptc.has(t) ? phase_pos(t) : 0;
                r.initial = ratio_now(t);
                r.min = std::numeric_limits<double>::infinity();
                r.max = -std::numeric_limits<double>::infinity();
                for (int pos = rtc.low_tap(t); pos <= rtc.high_tap(t); ++pos) {
                    const double value = trafos.ratio_at(t, pos, ppos);
                    r.min = std::min(r.min, value);
                    r.max = std::max(r.max, value);
                }
                if (r.max != r.min) {
                    a += (r.max - r.initial) / (r.max - r.min);
                    b += (r.initial - r.min) / (r.max - r.min);
                } else {
                    a += 1.;
                    b += 1.;
                }
                sum_min += r.min;
                sum_max += r.max;
            }
            const double n = static_cast<double>(on.size());
            for (int t : on) {
                TvcRatio& r = st.tvc_ratios[static_cast<size_t>(t)];
                if (tp.use_initial_tap_position) {
                    r.shared_min = sum_min / n;
                    r.shared_max = sum_max / n;
                    r.shared_initial = (r.shared_min * a + r.shared_max * b) / n;
                } else {
                    r.shared_min = r.min;
                    r.shared_max = r.max;
                    r.shared_initial = r.initial;
                }
            }
        }
        if (!need_run) {
            st.tvc_step = 2;
            return STABLE;
        }
        const std::vector<TvcBranch> closed = tvc_closed_branches(view(st));
        tvc_freeze(st, rv, tvc_limit(st, L), closed);
        tvc_fix_controls(st, closed);
        st.tvc_step = 1;
        return CHECK_AGAIN;   // the next check is the next step
    }

    // CONTROL: the ratios out of their range rounded to their extreme tap
    bool out_of_range = false;
    for (const auto& gr : st.tvc_groups)
        for (int t : gr.trafos) {
            if (!st.tvc_enabled[static_cast<size_t>(t)]) continue;
            const TvcRatio& r = st.tvc_ratios[static_cast<size_t>(t)];
            const double value = ratio_now(t);
            if (value < r.shared_min || value > r.shared_max) {
                const double edge = value > r.shared_max ? r.shared_max : r.shared_min;
                record(st, L.name, "ROUND_TO_RANGE", true, t, value, edge);
                st.bc_req_rtap[static_cast<size_t>(bc_entry(t))] = tvc_closest_tap(st, t, edge);
                tvc_set_on(st, t, false);
                out_of_range = true;
            }
        }
    if (!out_of_range) {
        // updateContinuousRatio, then every transformer rounded and switched off
        for (const auto& gr : st.tvc_groups) {
            if (gr.hidden) continue;
            for (int t : gr.trafos) {
                if (!reserved(t)) continue;   // handled_ratio
                double value = ratio_now(t);
                if (st.tvc_enabled[static_cast<size_t>(t)] && tp.use_initial_tap_position) {
                    const TvcRatio& r = st.tvc_ratios[static_cast<size_t>(t)];
                    value = value >= r.shared_initial
                        ? r.initial + (value - r.shared_initial) * (r.max - r.initial) / (r.shared_max - r.shared_initial)
                        : r.initial - (r.shared_initial - value) * (r.initial - r.min) / (r.shared_initial - r.shared_min);
                }
                const int position = tvc_closest_tap(st, t, value);
                st.bc_req_rtap[static_cast<size_t>(bc_entry(t))] = position;
                record(st, L.name, "ROUND_TAP", true, t, value, static_cast<double>(position));
                tvc_set_on(st, t, false);
            }
        }
        tvc_release(st);
        st.tvc_step = 2;
    }
    // in any case, the loop must run again: its next check is its next step
    return CHECK_AGAIN;
}

// =============================================================================
// ShuntVoltageControlLoop (WITH_GENERATOR_VOLTAGE_CONTROL)
// =============================================================================
std::vector<int> Ls2gOuterModel::svc_dispatch(const std::vector<int>& ids, double b) const
{
    const auto& shunts = grid_->get_shunts();
    auto b_at = [&](int sh, int count) { return -shunts.section_q(sh, count) / sn_; };
    // the largest first (the B between its fewest sections and its most), a stable order
    std::vector<size_t> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> magnitude(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int sh = ids[i];
        magnitude[i] = std::abs(b_at(sh, shunts.get_max_section_count(sh)) - b_at(sh, shunts.get_min_section_count(sh)));
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return magnitude[a] > magnitude[c]; });
    std::vector<int> counts(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) counts[i] = shunts.get_section_count(ids[i]);
    double residue = b;
    size_t remaining = ids.size();
    for (size_t i : order) {
        const int sh = ids[i];
        const double share = residue / static_cast<double>(remaining--);
        // roundBToClosestSection: the current count unless another is strictly closer
        int best = counts[i];
        double best_distance = std::abs(share - b_at(sh, best));
        for (int count = shunts.get_min_section_count(sh); count <= shunts.get_max_section_count(sh); ++count) {
            const double distance = std::abs(share - b_at(sh, count));
            if (distance < best_distance) { best = count; best_distance = distance; }
        }
        counts[i] = best;
        residue -= b_at(sh, best);
    }
    return counts;
}

int Ls2gOuterModel::check_svc(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const
{
    if (iteration != 0 || st.sv_groups.empty()) return STABLE;
    auto entry = [&](int bus) { return (bus >= 0 && bus < n_bus_) ? sc_of_bus_[static_cast<size_t>(bus)] : -1; };
    // nothing to do without a controller the inner algorithm can act on
    bool supported = false;
    for (const auto& gr : st.sv_groups) for (int bus : gr.controller_buses) supported = supported || entry(bus) >= 0;
    if (!supported) return STABLE;
    bool any = false;
    for (const auto& gr : st.sv_groups) {
        if (gr.hidden) continue;
        for (size_t c = 0; c < gr.controller_buses.size(); ++c) {
            const int bus = gr.controller_buses[c];
            const int e = entry(bus);
            if (e < 0) continue;
            st.sc_ctl[static_cast<size_t>(e)] = 0;
            const double b = rv.sc_b(e);
            sc_set_sections(st, e, svc_dispatch(gr.shunts[c], b));
            // the susceptance the Newton solved for (pu), rounded to sections
            record(st, L.name, "ROUND_SECTIONS", true, bus, b, NaN);
            any = true;
        }
    }
    return any ? UNSTABLE : STABLE;
}

// =============================================================================
// PhaseControlLoop::_check
// =============================================================================
int Ls2gOuterModel::check_pc(SlotState& st, const LoopDecl& L, const RowView& rv, int iteration) const
{
    // nothing to do without a phase shifter the inner algorithm can act on (one the batch's
    // system handles: its controls are those of the grid's solve)
    auto entry = [&](int trafo) {
        const int e = bc_of_trafo_[static_cast<size_t>(trafo)];
        return (e >= 0 && bc_phase_[static_cast<size_t>(e)]) ? e : -1;
    };
    bool any = false;
    for (const auto& sh : st.pc_shifters) any = any || entry(sh.trafo) >= 0;
    if (!any) return STABLE;
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    if (iteration == 0) {
        // the active power controllers are switched off, their shift rounded to the closest
        // tap (the current one unless another is strictly closer)
        for (const auto& sh : st.pc_shifters) {
            if (sh.mode != ls2g::RegulationMode::ACTIVE_POWER || !sh.regulates) continue;
            const int e = entry(sh.trafo);
            if (e < 0) continue;
            const size_t ee = static_cast<size_t>(e);
            st.bc_ctl[ee] = 0;
            const double a = rv.bc_a(e);
            int best = st.bc_pos[ee];
            double best_distance = std::abs(a - ptc.alpha_at(sh.trafo, best));
            for (int pos = ptc.low_tap(sh.trafo); pos <= ptc.high_tap(sh.trafo); ++pos) {
                const double distance = std::abs(a - ptc.alpha_at(sh.trafo, pos));
                if (distance < best_distance) { best = pos; best_distance = distance; }
            }
            st.bc_moved[ee] = 1;
            st.bc_req_tap[ee] = best;
            record(st, L.name, "ROUND_TAP", true, sh.trafo, a, ptc.alpha_at(sh.trafo, best));
        }
        // OpenLoadFlow re-solves whenever there is a phase shifter, limiters included; the
        // next check (the limiters') runs whatever this one moved
        return CHECK_AGAIN;
    }
    // the current limiters above their limit move one tap, the way that lowers the current
    bool moved = false;
    const auto vn_kv = grid_->get_bus_vn_kv();
    for (const auto& sh : st.pc_shifters) {
        if (sh.mode != ls2g::RegulationMode::CURRENT_LIMITER) continue;
        const int e = entry(sh.trafo);
        if (e < 0) continue;
        const size_t ee = static_cast<size_t>(e);
        double i_pu, di_da;
        bc_current(st, rv, e, sh.side, i_pu, di_da);
        const int bus = (sh.side == 1 ? trafos.get_bus_side_1(sh.trafo) : trafos.get_bus_side_2(sh.trafo)).cast_int();
        const double i_a = i_pu * sn_ * 1000. / (std::sqrt(3.) * vn_kv(bus));
        if (!(ptc.target(sh.trafo) < i_a)) {
            record(st, L.name, "KEPT_TAP", false, sh.trafo, i_a, ptc.target(sh.trafo));
            continue;
        }
        // PiModelArray.shiftOneTapPositionToChangeA1
        const bool increase = !(di_da > 0.);
        const double a = rv.bc_a(e);
        int pos = st.bc_pos[ee];
        const int old_pos = pos;
        if (pos < ptc.high_tap(sh.trafo)) {
            const double next = ptc.alpha_at(sh.trafo, pos + 1);
            if ((increase && next > a) || (!increase && next < a)) ++pos;
        }
        if (pos > ptc.low_tap(sh.trafo)) {
            const double previous = ptc.alpha_at(sh.trafo, pos - 1);
            if ((increase && previous > a) || (!increase && previous < a)) --pos;
        }
        record(st, L.name, "MOVE_TAP", pos != old_pos, sh.trafo, i_a, ptc.target(sh.trafo));
        if (pos != old_pos) {
            st.bc_moved[ee] = 1;
            st.bc_req_tap[ee] = pos;
            moved = true;
        }
    }
    return moved ? UNSTABLE : STABLE;
}

// =============================================================================
// The driver: OuterLoopAlgo::compute_pf, resumed at every inner solve's end
// =============================================================================
Next Ls2gOuterModel::on_solved(int slot, bool converged, int nr_iter, int inner_status,
                               const SlotData* data, int& row_status)
{
    ProfScope ps(prof_, P_ON_SOLVED_RESOLVE);
    const Next next = on_solved_impl(slot, converged, nr_iter, inner_status, data, row_status);
    if (next == Next::Done) {
        const SlotState& st = slots_[static_cast<size_t>(slot)];
        ps.relabel(st.nb_outer == 0 ? P_DONE_NO_RESOLVE : P_ON_SOLVED_DONE);
        if (prof_.on && st.nb_outer == 0) {   // what a device filter could have judged
            if (!st.solver_ok || st.last_status == FAILED) ++prof_.n[P_DNR_FAILED];
            else {
                if (!st.masked.empty()) ++prof_.n[P_DNR_OK_MASKED];
                ++prof_.n[st.lim_dirty.empty() ? P_DNR_OK_BASE : P_DNR_OK_MOVED];
            }
        }
    }
    return next;
}

Next Ls2gOuterModel::on_solved_impl(int slot, bool converged, int nr_iter, int inner_status,
                                    const SlotData* data, int& row_status)
{
    SlotState& st = slots_[static_cast<size_t>(slot)];
    // OuterLoopAlgo::_solve's epilogue: a solve counts for at least one iteration
    st.total_nr += std::max(nr_iter, 1);
    st.nr_steps += nr_iter;
    bool ok = converged;
    if (ok && st.check_unrealistic && data != nullptr && data->unrealistic) {
        st.unrealistic = true;
        ok = false;
    }
    st.solver_ok = ok;
    // OuterLoopAlgo::_solve: a distributed-slack Newton's share into the units' targets
    if (dist_ && converged && data != nullptr) ds_fold(st, data->slack_absorbed);
    RowView rv{data, &widx_};
    if (st.first) {
        st.first = false;
        if (!st.solver_ok) return finish(st, row_status, inner_status);
        // the first pass
        st.pass_check_unrealistic = st.last_fixing < 0;
        st.old_total = st.total_nr;
        ++st.nb_passes;
        st.i = 0;
        return drive(st, rv, row_status, inner_status);
    }
    // a re-solve after loop i acted: runOuterLoop's do-while goes on
    ++st.nb_outer;
    bool again = true;
    if (nr_iter > 0) {
        ++st.v_version;
    } else if (st.solver_ok && st.changed) {
        // no Newton step: V did not move and the change sits in the residuals, within
        // tolerance; checked again, the loop would act again for what it already did
        // (DistributedSlack moving its units twice for one mismatch). Its change is done,
        // until a solve moves V.
        st.done_at[static_cast<size_t>(st.i)] = st.v_version;
        st.last_status = STABLE;
        record(st, loops_[static_cast<size_t>(st.active[static_cast<size_t>(st.i)])].name, "SETTLED", true, -1, 0., 0.);
        again = false;
    }
    if (again && st.solver_ok && st.nb_outer < params_.max_outer_iterations)
        return drive(st, rv, row_status, inner_status, /*recheck=*/true);
    // the loop's do-while ends: its epilogue, then the next loop (which stops the pass)
    if (!st.pass_check_unrealistic && st.i == st.last_fixing && st.solver_ok && data != nullptr && data->unrealistic) {
        st.unrealistic = true;
        st.solver_ok = false;
    }
    if (st.i == st.last_fixing) st.pass_check_unrealistic = true;
    ++st.i;
    return drive(st, rv, row_status, inner_status);
}

Next Ls2gOuterModel::drive(SlotState& st, const RowView& rv, int& row_status, int inner_status, bool recheck)
{
    const int n = static_cast<int>(st.active.size());
    for (;;) {
        // the for loop over the loops of a pass, from loop i
        const bool rechecking = recheck;   // inside loop i's do-while: always checked
        const bool stop = !recheck &&
                          (st.i >= n || st.i == st.last_unstable || !st.solver_ok ||
                           st.last_status == FAILED || st.nb_outer >= params_.max_outer_iterations);
        recheck = false;
        if (stop) {
            // the pass is over: another one while a re-solve added Newton iterations
            if (st.total_nr > st.old_total && st.solver_ok && st.last_status != FAILED &&
                st.nb_outer < params_.max_outer_iterations) {
                st.pass_check_unrealistic = st.last_fixing < 0;
                st.old_total = st.total_nr;
                ++st.nb_passes;
                st.i = 0;
                continue;
            }
            return finish(st, row_status, inner_status, &rv);
        }
        const LoopDecl& L = loops_[static_cast<size_t>(st.active[static_cast<size_t>(st.i)])];
        // a loop whose change is done is not checked again until a solve moves V
        const bool settled = !rechecking && st.done_at[static_cast<size_t>(st.i)] == st.v_version;
        const int status = settled ? STABLE : check(st, L, rv, st.loop_iter[static_cast<size_t>(st.i)]);
        if (!settled) {
            st.last_status = status;
            st.last_name = L.name;
        }
        if (status == UNSTABLE || status == CHECK_AGAIN) {
            st.changed = status == UNSTABLE;
            if (prof_.on) {
                const int kk = static_cast<int>(L.kind);
                ++prof_.n[P_ACT_ALL + kk];
                if (st.nb_outer == 0) ++prof_.n[P_ACT_FIRST + kk];
            }
            ++st.loop_iter[static_cast<size_t>(st.i)];
            st.last_unstable = st.i;
            st.check_unrealistic = st.pass_check_unrealistic;
            materialize(st);
            return Next::Resolve;
        }
        // STABLE or FAILED: the loop's do-while ends here
        if (!st.pass_check_unrealistic && st.i == st.last_fixing && st.solver_ok && rv.d != nullptr && rv.d->unrealistic) {
            st.unrealistic = true;
            st.solver_ok = false;
        }
        if (st.i == st.last_fixing) st.pass_check_unrealistic = true;
        ++st.i;
    }
}

Next Ls2gOuterModel::finish(SlotState& st, int& row_status, int inner_status, const RowView* rv)
{
    ProfScope ps_(prof_, P_FINISH);
    int status;
    if (st.last_status == FAILED) status = FAILED;
    else status = st.nb_outer < params_.max_outer_iterations ? STABLE : UNSTABLE;
    if (st.solver_ok) {
        row_status = status == STABLE ? 0 /*Converged*/ : status == FAILED ? 4 /*OuterFailed*/ : 5 /*OuterUnstable*/;
    } else {
        row_status = st.unrealistic ? 6 /*Unrealistic*/ : inner_status;
    }
    if (st.row >= 0) {
        RowResult& r = results_[static_cast<size_t>(st.row)];
        r = RowResult();
        r.status = status;
        r.unrealistic = st.unrealistic;
        if (status == FAILED) r.failed_loop = st.last_name;
        r.nb_outer_iterations = st.nb_outer;
        r.nb_nr_iterations = st.nr_steps;
        // the final positions; row_result() lists where they differ from the grid's
        const size_t row = static_cast<size_t>(st.row);
        if (!st.storage_target_p.empty())
            std::copy(st.storage_target_p.begin(), st.storage_target_p.end(), fin_sto_p_.begin() + static_cast<ptrdiff_t>(row * n_sto_));
        std::copy(st.hvdc_regime.begin(), st.hvdc_regime.end(), fin_hvdc_.begin() + static_cast<ptrdiff_t>(row * layout_.n_hvdc));
        if (n_bc_ > 0) {
            std::copy(st.bc_pos.begin(), st.bc_pos.end(), fin_bc_pos_.begin() + static_cast<ptrdiff_t>(row * n_bc_));
            std::copy(st.bc_rpos.begin(), st.bc_rpos.end(), fin_bc_rpos_.begin() + static_cast<ptrdiff_t>(row * n_bc_));
        }
        for (int e = 0; e < n_sc_; ++e) {
            const size_t ee = static_cast<size_t>(e);
            const auto& c = st.sc_counts[ee];
            const size_t n = std::min(c.size(), sc_off_[ee + 1] - sc_off_[ee]);
            std::copy(c.begin(), c.begin() + static_cast<ptrdiff_t>(n), fin_sc_.begin() + static_cast<ptrdiff_t>(row * n_sc_sh_ + sc_off_[ee]));
        }
        fin_done_[row] = 1;
        fin_built_[row] = 0;
        r.decisions = std::move(st.trace);
        if (keep_adjoint_) adjoint_info(st, r);
        if (keep_points_ && status == STABLE && st.solver_ok && row < row_points_.size()) keep_point(st, rv);
        // the row is over: its slot's targets (reset at its next load) become the result's
        r.gens_target_p_mw = std::move(st.gen_target_p);
    }
    return Next::Done;
}

void Ls2gOuterModel::set_spare_buses(const LedgerData& ledger, const std::vector<int>& spare_grid_bus)
{
    // the buses change (and the grid's plans with them): the device tables' plan is rebuilt by the
    // next device_prepare_rows
    rl_dev_.reset();
    rl_dev_built_ = false;
    BaseBuses& b = base_buses_;
    if (!b.saved) {
        b.n_bus = n_bus_;
        b.sbl = sbl_;
        b.pin_bus = layout_.pin_bus;
        b.watch_bus = layout_.watch_bus;
        b.q_row_of_bus = q_row_of_bus_;
        b.pin_of_bus = pin_of_bus_;
        b.sc_of_bus = sc_of_bus_;
        b.slack_idx_of_bus = slack_idx_of_bus_;
        b.widx = widx_;
        b.vc_at_bus = vc_at_bus_;
        b.hvdc_at_bus = hvdc_at_bus_;
        b.saved = true;
    }
    // back to the base ...
    n_bus_ = b.n_bus;
    sbl_ = b.sbl;
    layout_.pin_bus = b.pin_bus;
    layout_.watch_bus = b.watch_bus;
    q_row_of_bus_ = b.q_row_of_bus;
    pin_of_bus_ = b.pin_of_bus;
    sc_of_bus_ = b.sc_of_bus;
    slack_idx_of_bus_ = b.slack_idx_of_bus;
    widx_ = b.widx;
    vc_at_bus_ = b.vc_at_bus;
    hvdc_at_bus_ = b.hvdc_at_bus;
    const int m = static_cast<int>(spare_grid_bus.size());
    if (m > 0) {
        // ... then the spares appended: each its own grid bus in the labelling (an empty busbar
        // of the base grid, as lightsim2grid's union layout keeps it), PQ, no slack share
        const int n0 = b.n_bus;
        if (ledger.n_bus != n0 + m)
            throw std::runtime_error("outer model: the ledger has " + std::to_string(ledger.n_bus) + " buses, not "
                                     + std::to_string(n0) + " + " + std::to_string(m) + " spares");
        using SolverId = std::decay_t<decltype(std::as_const(sbl_.id_me_to_solver)[0])>;
        using GlobalId = std::decay_t<decltype(std::as_const(sbl_.id_solver_to_me)[0])>;
        for (int k = 0; k < m; ++k) {
            const int me = spare_grid_bus[static_cast<size_t>(k)];
            if (me < 0 || static_cast<size_t>(me) >= sbl_.id_me_to_solver.size()
                    || sbl_.id_me_to_solver[me].cast_int() != ls2g::BaseConstants::_deactivated_bus_id)
                throw std::runtime_error("outer model: spare bus " + std::to_string(n0 + k) + " on grid bus "
                                         + std::to_string(me) + ", which is not an empty bus of the grid");
            sbl_.id_me_to_solver[me] = SolverId(n0 + k);
            sbl_.id_solver_to_me.push_back(GlobalId(me));
            sbl_.bus_pq.push_back(SolverId(n0 + k));
        }
        for (ls2g::RealVect* w : {&sbl_.slack_weights, &sbl_.slack_raw_weights})
            if (w->size() == n0) { w->conservativeResize(n0 + m); w->tail(m).setZero(); }
        n_bus_ = n0 + m;
        q_row_of_bus_ = ledger.q_row_of_bus;
        // switchable: the base ones, then the spares (PQ by default, see build_slot0)
        if (ledger.switchable_vm_buses.size() != b.pin_bus.size() + static_cast<size_t>(m)
                || !std::equal(b.pin_bus.begin(), b.pin_bus.end(), ledger.switchable_vm_buses.begin()))
            throw std::runtime_error("outer model: the ledger's switchable buses are not the base ones and the spares");
        layout_.pin_bus = ledger.switchable_vm_buses;
        pin_of_bus_.assign(static_cast<size_t>(n_bus_), -1);
        for (size_t k = 0; k < layout_.pin_bus.size(); ++k)
            pin_of_bus_[static_cast<size_t>(layout_.pin_bus[k])] = static_cast<int>(k);
        sc_of_bus_.resize(static_cast<size_t>(n_bus_), -1);
        if (!slack_idx_of_bus_.empty()) slack_idx_of_bus_.resize(static_cast<size_t>(n_bus_), -1);
        vc_at_bus_.resize(static_cast<size_t>(n_bus_));
        hvdc_at_bus_.resize(static_cast<size_t>(n_bus_), 0);
        // watched: a moved generator's ReactiveLimits reads its new bus
        widx_.resize(static_cast<size_t>(n_bus_), -1);
        for (int k = 0; k < m; ++k) {
            widx_[static_cast<size_t>(n0 + k)] = static_cast<int>(layout_.watch_bus.size());
            layout_.watch_bus.push_back(n0 + k);
        }
    }
    hvdc_va_scratch_ = ls2g::RealVect::Zero(n_bus_);
    vm_v_scratch_ = ls2g::CplxVect::Zero(n_bus_);
}

// Where the row's loops ended (stable), as lightsim2grid's OuterResumePoint has a solve's: the
// targets they moved (the point the row resumed from where it moved none), the hvdc regimes,
// the SVCs switched on, the frozen buses (with their voltage: rl_limits' SVC part), the parked
// units, the positions the active tap and section loops settled on, the loss it counted.
void Ls2gOuterModel::keep_point(const SlotState& st, const RowView* rv)
{
    auto pt = std::make_shared<ResumePoint>();
    const ResumePoint* from = st.rp;
    auto merged = [](const std::vector<double>& own, const std::vector<double>* base) {
        std::vector<double> out = own.empty() && base != nullptr ? *base : own;
        if (base != nullptr && !own.empty())
            for (size_t i = 0; i < out.size() && i < base->size(); ++i)
                if (std::isnan(out[i])) out[i] = (*base)[i];
        return out;
    };
    pt->gen_p = merged(st.gen_target_p, from != nullptr ? &from->gen_p : nullptr);
    pt->sto_p = merged(st.storage_target_p, from != nullptr ? &from->sto_p : nullptr);
    for (size_t e = 0; e < st.hvdc_regime.size() && e < hvdc_id_.size(); ++e)
        if (st.hvdc_regime[e] != 2) pt->hvdc[hvdc_id_[e]] = st.hvdc_regime[e];
    pt->svc_on = st.vm_switched_on;
    if (st.rl)
        for (size_t k = 0; k < st.rl_state.size() && k < st.rl->buses.size(); ++k)
            if (st.rl_state[k].state != 0)
                pt->frozen.emplace(st.rl->buses[k].bus_solver, std::make_pair(st.rl_state[k].state, st.rl_state[k].realistic));
    if (rv != nullptr)
        for (const auto& fb : pt->frozen)
            if (rv->has(fb.first)) pt->v_at.emplace_back(fb.first, rv->V(fb.first));
    for (size_t k = 0; k < st.ds_parked.size() && k < st.ds_units.size(); ++k)
        if (st.ds_parked[k])
            pt->parked.push_back({st.ds_units[k].kind, st.ds_units[k].el_id,
                                  k < st.ds_parked_high.size() && st.ds_parked_high[k] != 0});
    bool pc = false, tvc = false, svc = false;
    for (int li : st.active) {
        const Kind k = loops_[static_cast<size_t>(li)].kind;
        pc = pc || k == Kind::PhaseControl;
        tvc = tvc || k == Kind::TransformerVoltageControl;
        svc = svc || k == Kind::ShuntVoltageControl;
    }
    const int none = std::numeric_limits<int>::min();
    if (pc || tvc) {
        pt->phase_tap.assign(static_cast<size_t>(n_trafo_), none);
        pt->ratio_tap.assign(static_cast<size_t>(n_trafo_), none);
        for (int e = 0; e < n_bc_; ++e) {
            const size_t ee = static_cast<size_t>(e);
            if (st.bc_out[ee]) continue;
            const size_t t = static_cast<size_t>(bc_trafo_[ee]);
            if (pc && bc_phase_[ee]) pt->phase_tap[t] = st.bc_pos[ee];
            if (tvc && bc_ratio_[ee]) pt->ratio_tap[t] = st.bc_rpos[ee];
        }
    }
    if (svc && n_sc_ > 0) {
        pt->sections.assign(static_cast<size_t>(grid_->get_shunts().nb()), none);
        for (int e = 0; e < n_sc_; ++e) {
            const auto& sh = sc_shunts_[static_cast<size_t>(e)];
            const auto& c = st.sc_counts[static_cast<size_t>(e)];
            for (size_t j = 0; j < sh.size() && j < c.size(); ++j) pt->sections[static_cast<size_t>(sh[j])] = c[j];
        }
    }
    pt->lost_mw = st.lost_mw;
    row_points_[static_cast<size_t>(st.row)] = std::move(pt);
}

void Ls2gOuterModel::adjoint_info(const SlotState& st, RowResult& r) const
{
    ProfScope ps_(prof_, P_ADJOINT);
    // the controls of the row's last solve (materialized for it: a check that ends the row
    // changes none)
    const SlotControls& c = st.ctl;
    for (size_t k = 0; k < c.pin.size(); ++k)
        if (!c.pin[k]) r.pq_buses.push_back(layout_.pin_bus[k]);
    // VoltageControl, may_hold_ formulation: a controller held by value or by the plan
    auto held = [&](int j) {
        return std::isfinite(c.held_q[static_cast<size_t>(j)]) || c.plan_held[static_cast<size_t>(j)] != 0;
    };
    auto ref_active = [&](int g) {
        for (int off = 0; off < grp_count_[static_cast<size_t>(g)]; ++off)
            if (!held(grp_start_[static_cast<size_t>(g)] + off)) return off;
        return -1;
    };
    for (int g = 0; g < layout_.n_grp; ++g)
        if (ref_active(g) < 0) r.vset_lost_groups.push_back(g);
    // VoltageControl::hold_row: the row pinning a held controller at its output
    auto hold_row = [&](int j) {
        if (j < 0 || j >= layout_.n_ctrl || !held(j)) return -1;
        const int g = vc_group_[static_cast<size_t>(j)];
        if (static_cast<size_t>(g) >= vc_v_row_.size()) return -1;
        const auto row_of = [&](int k) {
            return k == 0 ? vc_v_row_[static_cast<size_t>(g)] : vc_share_row_[static_cast<size_t>(g)][static_cast<size_t>(k - 1)];
        };
        const int off = j - grp_start_[static_cast<size_t>(g)];
        const int ref = ref_active(g);
        return (ref >= 0 && off == 0) ? row_of(ref) : row_of(off);
    };
    // DistributedSlackLoop::_adjoint_terms: a generator parked at a bound sits there whatever
    // its set-point -- unless that bound is its own set-point (one already beyond a limit). Its
    // own: from before the row's lost-power share, which moves the initial targets the loop
    // starts from (injection_mw) -- a unit that share put at its bound is held there too.
    // As the last Newton solve had them: a unit parked by a check that did not re-solve still
    // shared the slack in that solve
    if (st.ds_in_newton)
        for (size_t k = 0; k < st.ds_units.size() && k < st.ds_solved_parked.size() &&
                           k < st.ds_solved_mw.size() && k < st.ds_initial_mw.size(); ++k) {
            const auto& u = st.ds_units[k];
            if (!st.ds_solved_parked[k] || u.kind != ls2g::slack_redistribution::UnitKind::GENERATOR) continue;
            if (st.ds_solved_mw[k] == st.ds_initial_mw[k]) continue;
            r.gen_p_fixed.push_back(u.el_id);
        }
    // ReactiveLimitsLoop::_adjoint_terms (OuterLoopAlgo::get_adjoint_solve_info: a unit whose P
    // reaches nothing has no slope either)
    for (const auto& frozen : st.rl_slopes) {   // by index in rl->buses: the buses in their order
        const size_t k = static_cast<size_t>(frozen.first);
        if (st.rl_state[k].state == 0) continue;
        const RlBus& bus = st.rl->buses[k];
        for (const auto& sl : frozen.second) {
            const int g = sl.first.first, ctrl = sl.first.second;
            if (std::find(r.gen_p_fixed.begin(), r.gen_p_fixed.end(), g) != r.gen_p_fixed.end()) continue;
            const int row = ctrl >= 0 ? hold_row(ctrl) : q_row_of_bus_[static_cast<size_t>(bus.bus_solver)];
            if (row < 0) continue;
            r.gen_p_slope_gen.push_back(g);
            r.gen_p_slope_row.push_back(row);
            r.gen_p_slope_coeff.push_back(sl.second);
        }
    }
}

// A finished row's positions where they differ from the grid's (BaseBatchSweep::_record_row_outer)
const RowResult& Ls2gOuterModel::row_result(int row) const
{
    const size_t rr = static_cast<size_t>(row);
    RowResult& r = results_.at(rr);
    if (!fin_done_[rr] || fin_built_[rr]) return r;
    fin_built_[rr] = 1;
    for (int s = 0; s < n_sto_; ++s) {
        const double p = fin_sto_p_[rr * static_cast<size_t>(n_sto_) + static_cast<size_t>(s)];
        if (!std::isnan(p)) { r.storage_ids.push_back(s); r.storage_target_p_mw.push_back(p); }
    }
    for (int e = 0; e < layout_.n_hvdc; ++e) {
        const int reg = fin_hvdc_[rr * static_cast<size_t>(layout_.n_hvdc) + static_cast<size_t>(e)];
        if (reg != 2) { r.hvdc_ids.push_back(hvdc_id_[static_cast<size_t>(e)]); r.hvdc_regime.push_back(reg); }
    }
    // the section counts the loops left
    const auto& shunts = grid_->get_shunts();
    for (int e = 0; e < n_sc_; ++e)
        for (size_t i = 0; i < sc_shunts_[static_cast<size_t>(e)].size(); ++i) {
            const int sh = sc_shunts_[static_cast<size_t>(e)][i];
            const int c = fin_sc_[rr * n_sc_sh_ + sc_off_[static_cast<size_t>(e)] + i];
            if (shunts.has_sections(sh) && shunts.get_section_count(sh) == c) continue;
            r.shunt_ids.push_back(sh);
            r.shunt_sections.push_back(c);
        }
    // the tap positions the loops handled
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    for (int e = 0; e < n_bc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        const int t = bc_trafo_[ee];
        const int pos = fin_bc_pos_[rr * static_cast<size_t>(n_bc_) + ee];
        const int rpos = fin_bc_rpos_[rr * static_cast<size_t>(n_bc_) + ee];
        if (bc_phase_[ee] && !(ptc.has(t) && ptc.position(t) == pos)) { r.phase_tap_ids.push_back(t); r.phase_tap.push_back(pos); }
        if (bc_ratio_[ee] && !(rtc.has(t) && rtc.position(t) == rpos)) { r.ratio_tap_ids.push_back(t); r.ratio_tap.push_back(rpos); }
    }
    return r;
}

// the blocks a finished row's flows use (BaseBatchSweep::_record_row_trafo_blocks)
void Ls2gOuterModel::row_blocks(size_t row, std::vector<int>& trafo, std::vector<double>& block) const
{
    trafo.clear();
    block.clear();
    if (!fin_done_[row]) return;
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    for (int e = 0; e < n_bc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        const int t = bc_trafo_[ee];
        const int pos = fin_bc_pos_[row * static_cast<size_t>(n_bc_) + ee];
        const int rpos0 = fin_bc_rpos_[row * static_cast<size_t>(n_bc_) + ee];
        const bool ph = bc_phase_[ee] && !(ptc.has(t) && ptc.position(t) == pos);
        const bool ra = bc_ratio_[ee] && !(rtc.has(t) && rtc.position(t) == rpos0);
        if (!ph && !ra) continue;
        const int rpos = ra ? rpos0 : (rtc.has(t) ? rtc.position(t) : 0);
        const int ppos = ph ? pos : (ptc.has(t) ? ptc.position(t) : 0);
        const double shift = ptc.has(t) ? ptc.alpha_at(t, ppos) : trafos[t].shift_rad;
        const auto blk = trafos.pi_block_at(t, rpos, ppos, shift);
        trafo.push_back(t);
        for (int k = 0; k < 4; ++k) { block.push_back(blk[static_cast<size_t>(k)].real()); block.push_back(blk[static_cast<size_t>(k)].imag()); }
    }
}

void Ls2gOuterModel::row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const
{
    if (row < 0 || row >= n_rows_) throw std::out_of_range("outer model: row_trafo_blocks: no such row");
    row_blocks(static_cast<size_t>(row), trafo, block);
}

}  // namespace

bool ls2g_outer_resume_available()
{
#ifdef LS2G_HAS_OUTER_LOOP_STATE_ACCESSORS
    return true;
#else
    return false;
#endif
}

std::shared_ptr<OuterHook> make_ls2g_outer_model(const ls2g::LSGrid& grid, const LedgerData& ledger,
                                                 const Ls2gModelOptions& options)
{
    return std::make_shared<Ls2gOuterModel>(grid, ledger, options);
}

// =============================================================================
// The device backend's feature census (docs/outer_loops_on_device.md, §4.2)
// =============================================================================
std::vector<DeviceFeature> Ls2gOuterModel::census() const
{
    std::vector<DeviceFeature> out;
    // what outer_loops_backend='device' runs (device_outer_model.cu); every other feature refuses it
    static const std::set<std::string> device_runs = {
        "loop DistributedSlack", "loop AcHvdcAcEmulationLimits", "loop VoltageMonitoring", "loop ReactiveLimits",
        "ReactiveLimits local controller buses", "ReactiveLimits voltage-control group buses (remote regulators)",
        "ReactiveLimits robust mode on group buses", "ReactiveLimits buses with SVCs",
        "ReactiveLimits buses with storage units", "ReactiveLimits buses with HVDC stations",
        "ReactiveLimits voltage-monitor buses (standby SVCs once switched on)",
        "generators with a reactive capability curve", "generators with forceTargetQInReactiveLimits",
        "held controllers (frozen regulators kept in their group)", "DistributedSlack units",
        "HVDC lines in AC emulation (AcHvdcAcEmulationLimits)", "standby SVCs (VoltageMonitoring)",
        "resumed rows (init_from_n_powerflow)", "resumed frozen local buses", "resumed frozen group buses",
        "resumed switched-on SVCs", "resumed saturated HVDC lines", "distributed-slack Newton (NROuterDist_*)",
        "resumed parked DistributedSlack units", "loop ShuntVoltageControl", "ShuntControl controllers",
        "resumed section positions", "loop PhaseControl", "BranchControl transformers",
        "resumed tap positions", "loop TransformerVoltageControl"};
    auto add = [&](const char* name, size_t count) {
        out.push_back(DeviceFeature{name, static_cast<int>(count), device_runs.count(name) > 0});
    };
    size_t n_kind[7] = {0, 0, 0, 0, 0, 0, 0};
    for (const LoopDecl& d : loops_) ++n_kind[static_cast<int>(d.kind)];
    add("loop DistributedSlack", n_kind[static_cast<int>(Kind::DistributedSlack)]);
    add("loop AcHvdcAcEmulationLimits", n_kind[static_cast<int>(Kind::HvdcLimits)]);
    add("loop VoltageMonitoring", n_kind[static_cast<int>(Kind::VoltageMonitoring)]);
    add("loop ReactiveLimits", n_kind[static_cast<int>(Kind::ReactiveLimits)]);
    add("loop PhaseControl", n_kind[static_cast<int>(Kind::PhaseControl)]);
    add("loop TransformerVoltageControl", n_kind[static_cast<int>(Kind::TransformerVoltageControl)]);
    add("loop ShuntVoltageControl", n_kind[static_cast<int>(Kind::ShuntVoltageControl)]);
    add("distributed-slack Newton (NROuterDist_*)", dist_ ? 1 : 0);
    bool robust = false;
    for (const LoopDecl& d : loops_) robust = robust || (d.kind == Kind::ReactiveLimits && d.rl_robust);

    // ReactiveLimits' controller buses (the grid's plan)
    size_t rl_local = 0, rl_remote = 0, rl_svc = 0, rl_storage = 0, rl_station = 0, rl_monitor = 0;
    for (const RlBus& b : rl0_->buses) {
        if (b.monitor_svc >= 0) { ++rl_monitor; continue; }
        const auto& e = rl0_->plan->buses[static_cast<size_t>(b.entry)];
        if (b.local) ++rl_local; else ++rl_remote;
        if (!e.svc_ids.empty()) ++rl_svc;
        if (!e.storage_ids.empty()) ++rl_storage;
        if (!e.station_ids.empty()) ++rl_station;
    }
    add("ReactiveLimits local controller buses", rl_local);
    add("ReactiveLimits voltage-control group buses (remote regulators)", rl_remote);
    add("ReactiveLimits robust mode on group buses", robust && rl_remote > 0 ? 1 : 0);
    add("ReactiveLimits buses with SVCs", rl_svc);
    add("ReactiveLimits buses with storage units", rl_storage);
    add("ReactiveLimits buses with HVDC stations", rl_station);
    add("ReactiveLimits voltage-monitor buses (standby SVCs once switched on)", rl_monitor);
    // limits that move with a generator's target P, a target Q that follows it
    size_t curves = 0, force_q = 0;
    {
        const auto raw_q = grid_->get_gen_raw_target_q();
        for (int g = 0; g < n_gen_; ++g) {
            ls2g::real_type lo, hi;
            if (grid_->gen_limits_at(g, 0., lo, hi)) ++curves;
            if (raw_q.size() == n_gen_ && !std::isnan(static_cast<double>(raw_q(g)))) ++force_q;
        }
    }
    add("generators with a reactive capability curve", curves);
    add("generators with forceTargetQInReactiveLimits", force_q);
    size_t held = 0;
    for (int c = 0; c < layout_.n_ctrl; ++c)
        if (vc_plan_held_[static_cast<size_t>(c)] && vc_kind_[static_cast<size_t>(c)] != ls2g::VoltageControlSolverData::SVC) ++held;
    add("held controllers (frozen regulators kept in their group)", held);

    // the other loops' plans
    add("DistributedSlack units", units0_.size());
    add("HVDC lines in AC emulation (AcHvdcAcEmulationLimits)", hvdc_plan0_.lines.size());
    add("standby SVCs (VoltageMonitoring)", monitors_.svcs.size());
    add("BranchControl transformers", static_cast<size_t>(n_bc_));
    add("ShuntControl controllers", static_cast<size_t>(n_sc_));

    // where the rows resume from (init_from_n_powerflow=True)
    size_t fr_local = 0, fr_remote = 0, hv_sat = 0;
    if (resume_) {
        for (const auto& kv : point_n_.frozen) {
            bool local = true;
            for (const RlBus& b : rl0_->buses)
                if (b.bus_solver == kv.first) { local = b.local; break; }
            if (local) ++fr_local; else ++fr_remote;
        }
        for (const auto& kv : point_n_.hvdc) hv_sat += kv.second != 0;
    }
    add("resumed rows (init_from_n_powerflow)", resume_ ? 1 : 0);
    add("resumed frozen local buses", fr_local);
    add("resumed frozen group buses", fr_remote);
    add("resumed switched-on SVCs", resume_ ? point_n_.svc_on.size() : 0);
    add("resumed saturated HVDC lines", hv_sat);
    add("resumed parked DistributedSlack units", resume_ ? point_n_.parked.size() : 0);
    add("resumed tap positions", resume_ ? point_n_.phase_tap.size() + point_n_.ratio_tap.size() : 0);
    add("resumed section positions", resume_ ? point_n_.sections.size() : 0);
    return out;
}

std::shared_ptr<OuterHook> make_device_outer_model(const ls2g::LSGrid& grid, const LedgerData& ledger,
                                                   const Ls2gModelOptions& options)
{
    auto host = std::make_shared<Ls2gOuterModel>(grid, ledger, options);
    check_device_features(host->census());
    DeviceHostSide* side = host.get();
    return make_device_outer_model_from(std::move(host), side);
}

// keep_point, from what the device read off the row's slot (DevicePoint: its targets, frozen buses,
// switched-on SVCs, hvdc regimes, parked units, taps and sections), merged with the point it resumed
// from where the slot holds nothing of its own
void Ls2gOuterModel::device_import_point(int row, const DevicePoint& p)
{
    if (!keep_points_ || row < 0 || static_cast<size_t>(row) >= row_points_.size()) return;
    auto pt = std::make_shared<ResumePoint>();
    const ResumePoint* from = p.from_row >= 0 && static_cast<size_t>(p.from_row) < row_points_.size()
        ? row_points_[static_cast<size_t>(p.from_row)].get() : (resume_ ? &point_n_ : nullptr);
    auto merged = [](const std::vector<double>& own, const std::vector<double>* base) {
        std::vector<double> out = own.empty() && base != nullptr ? *base : own;
        if (base != nullptr && !own.empty())
            for (size_t i = 0; i < out.size() && i < base->size(); ++i)
                if (std::isnan(out[i])) out[i] = (*base)[i];
        return out;
    };
    bool any_gen = false, any_sto = false;
    for (double v : p.gen_p) any_gen = any_gen || !std::isnan(v);
    for (double v : p.sto_p) any_sto = any_sto || !std::isnan(v);
    static const std::vector<double> none;
    pt->gen_p = merged(any_gen ? p.gen_p : none, from != nullptr ? &from->gen_p : nullptr);
    pt->sto_p = merged(any_sto ? p.sto_p : none, from != nullptr ? &from->sto_p : nullptr);
    for (size_t k = 0; k < p.hvdc_id.size(); ++k) pt->hvdc[p.hvdc_id[k]] = p.hvdc_regime[k];
    for (size_t k = 0; k < p.svc_id.size(); ++k) pt->svc_on.emplace_back(p.svc_id[k], p.svc_target[k]);
    for (size_t k = 0; k < p.frozen_bus.size(); ++k) {
        pt->frozen.emplace(p.frozen_bus[k], std::make_pair(p.frozen_state[k], p.frozen_realistic[k] != 0));
        if (!std::isnan(p.v_re[k])) pt->v_at.emplace_back(p.frozen_bus[k], cplx(p.v_re[k], p.v_im[k]));
    }
    if (p.has_phase || p.has_ratio) {
        const int none = std::numeric_limits<int>::min();
        pt->phase_tap.assign(static_cast<size_t>(n_trafo_), none);
        pt->ratio_tap.assign(static_cast<size_t>(n_trafo_), none);
        for (int e = 0; e < n_bc_; ++e) {
            const size_t ee = static_cast<size_t>(e);
            if (ee >= p.bc_out.size() || p.bc_out[ee]) continue;
            const size_t tr = static_cast<size_t>(bc_trafo_[ee]);
            if (p.has_phase && bc_phase_[ee]) pt->phase_tap[tr] = p.bc_pos[ee];
            if (p.has_ratio && bc_ratio_[ee]) pt->ratio_tap[tr] = p.bc_rpos[ee];
        }
    }
    if (p.has_sections) {
        pt->sections.assign(static_cast<size_t>(grid_->get_shunts().nb()), std::numeric_limits<int>::min());
        size_t j = 0;
        for (int e = 0; e < n_sc_; ++e)
            for (int sh : sc_shunts_[static_cast<size_t>(e)])
                if (j < p.sections.size()) pt->sections[static_cast<size_t>(sh)] = p.sections[j++];
    }
    for (size_t k = 0; k < p.parked_el.size(); ++k)
        pt->parked.push_back({p.parked_kind[k] == 0 ? ls2g::slack_redistribution::UnitKind::GENERATOR
                                                    : ls2g::slack_redistribution::UnitKind::STORAGE,
                              p.parked_el[k], p.parked_high[k] != 0});
    pt->lost_mw = p.lost_mw;
    row_points_[static_cast<size_t>(row)] = std::move(pt);
}

// TransformerVoltageControl's part of a slot image: its state, and what its freeze reads that the
// row fixes (tvc_freeze's static half, evaluated now: the candidates, their target |V| and load)
void Ls2gOuterModel::device_tvc_image(const SlotState& st, DeviceSlotImage& img) const
{
    const LoopDecl* L = nullptr;
    for (const LoopDecl& d : loops_) if (d.kind == Kind::TransformerVoltageControl) L = &d;
    if (L == nullptr) return;
    SlotState plain_tv;
    plain_tv.rowv.layout = &sbl_;
    const ls2g::OuterGridView v0 = view(plain_tv);
    const auto grid_groups = L->tvc->groups(v0);
    // the row's groups: the grid's, some hidden
    img.tv_hidden.assign(grid_groups.size(), 1);
    if (!st.tvc_groups.empty()) {
        if (st.tvc_groups.size() != grid_groups.size())
            throw std::domain_error("a row whose transformer voltage groups are not the grid's");
        for (size_t g = 0; g < grid_groups.size(); ++g) {
            if (st.tvc_groups[g].bus_solver != grid_groups[g].bus_solver || st.tvc_groups[g].trafos != grid_groups[g].trafos)
                throw std::domain_error("a row whose transformer voltage groups are not the grid's");
            img.tv_hidden[g] = st.tvc_groups[g].hidden ? 1 : 0;
        }
    }
    img.tvc_step = st.tvc_step;
    for (const auto& gr : grid_groups)
        for (int tr : gr.trafos) {
            const size_t tt = static_cast<size_t>(tr);
            img.tv_enabled.push_back(tt < st.tvc_enabled.size() ? st.tvc_enabled[tt] : 0);
            const TvcRatio r = tt < st.tvc_ratios.size() ? st.tvc_ratios[tt] : TvcRatio();
            for (double x : {r.initial, r.min, r.max, r.shared_min, r.shared_max, r.shared_initial}) img.tv_ratios.push_back(x);
        }
    // the row's branches: the grid's ends where closed (a row moving one to other buses is refused)
    const ls2g::OuterGridView v = view(st);
    int b1, b2, g1, g2;
    std::vector<TvcBranch> closed;
    for (int el = 0; el < n_line_ + n_trafo_; ++el) {
        const bool trafo = el >= n_line_;
        const int id = trafo ? el - n_line_ : el;
        const bool c = trafo ? v.trafo_closed(id, b1, b2) : v.line_closed(id, b1, b2);
        const bool c0 = trafo ? v0.trafo_closed(id, g1, g2) : v0.line_closed(id, g1, g2);
        if (c && (!c0 || b1 != g1 || b2 != g2))
            throw std::domain_error("a row closing a branch the grid has open, or moving one");
        img.br_closed.push_back(c ? 1 : 0);
        if (c) closed.push_back(TvcBranch{b1, b2, trafo ? id : -1});
    }
    // the freeze's static half (tvc_freeze, the dynamic tests left to the device)
    const size_t rn = rl_dev().buses.size();
    img.tv_cand.assign(rn, 0); img.tv_tvm.assign(rn, NaN); img.tv_load.assign(rn, 0.);
    img.tv_frozen.assign(rn, 0); img.tv_frozen_vm.assign(rn, 0.);
    img.tv_susp.assign(static_cast<size_t>(n_bus_), 0);
    for (int b : st.suspended) if (b >= 0 && b < n_bus_) img.tv_susp[static_cast<size_t>(b)] = 1;
    img.tvc_limit = st.tvc_groups.empty() ? 0. : tvc_limit(st, *L);
    if (st.rl) {
        const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();
        const auto& to_me = v.id_solver_to_me();
        const auto vn_kv = grid_->get_bus_vn_kv();
        const auto& trafos = grid_->get_trafos();
        std::vector<char> on_controller(static_cast<size_t>(vn_kv.size()), 0);
        for (int tr = 0; tr < n_trafo_; ++tr) {
            if (static_cast<size_t>(tr) >= st.tvc_controller.size() || !st.tvc_controller[static_cast<size_t>(tr)]) continue;
            for (const int b : {trafos.get_bus_side_1(tr).cast_int(), trafos.get_bus_side_2(tr).cast_int()})
                if (b >= 0 && b < static_cast<int>(vn_kv.size())) on_controller[static_cast<size_t>(b)] = 1;
        }
        std::vector<double> min_connected(static_cast<size_t>(to_me.size()), NaN);
        for (const TvcBranch& br : closed) {
            const double vv = std::max(vn_kv(to_me[br.bus1].cast_int()), vn_kv(to_me[br.bus2].cast_int()));
            for (const int b : {br.bus1, br.bus2}) {
                double& m = min_connected[static_cast<size_t>(b)];
                m = std::isnan(m) ? vv : std::min(m, vv);
            }
        }
        std::vector<double> load_q(static_cast<size_t>(n_bus_), 0.);
        for (int l = 0; l < grid_->get_loads().nb(); ++l) {
            const int b = v.load_bus_solver(l);
            if (b >= 0 && b < n_bus_) load_q[static_cast<size_t>(b)] += v.load_target_q(l) / sn_;
        }
        const double limit = img.tvc_limit;
        const auto& buses = st.rl->plan->buses;
        size_t k0 = 0;
        for (size_t k = 0; k < st.rl->buses.size(); ++k) {
            // the row bus' index in the grid plan (the same bus and monitor)
            while (k0 < rn && !(rl_dev().buses[k0].bus_solver == st.rl->buses[k].bus_solver &&
                                rl_dev().buses[k0].monitor_svc == st.rl->buses[k].monitor_svc)) ++k0;
            if (k0 == rn) throw std::domain_error("a row whose ReactiveLimits plan is not a filter of the grid's");
            const size_t kk = k0++;
            if (k >= st.rl->n_bus_q) continue;
            const auto& entry = buses[static_cast<size_t>(st.rl->buses[k].entry)];
            const bool local = entry.ctrl_pos.empty() && entry.svc_ids.empty();
            int controlled_grid = entry.bus_grid;
            if (!local) controlled_grid = to_me[ctrl.reg_bus(ctrl.group(entry.ctrl_pos.front()))].cast_int();
            if (controlled_grid < 0 || vn_kv(controlled_grid) > limit) continue;
            bool step_up;
            if (!entry.station_ids.empty()) step_up = true;
            else if (on_controller[static_cast<size_t>(entry.bus_grid)]) step_up = false;
            else {
                const double min_v = min_connected[static_cast<size_t>(entry.bus_solver)];
                step_up = min_v > vn_kv(entry.bus_grid) && min_v > limit;
            }
            if (step_up) continue;
            img.tv_cand[kk] = 1;
            if (!entry.gen_ids.empty()) img.tv_tvm[kk] = v.gen_target_vm(entry.gen_ids.front());
            else if (!entry.storage_ids.empty()) img.tv_tvm[kk] = v.storage_target_vm(entry.storage_ids.front());
            img.tv_load[kk] = load_q[static_cast<size_t>(entry.bus_solver)];
        }
        // the buses frozen now (a resumed or loaded row's)
        for (const TvcFrozen& f : st.tvc_frozen) {
            for (size_t q = 0; q < rn; ++q)
                if (rl_dev().buses[q].bus_solver == f.bus_solver && rl_dev().buses[q].monitor_svc < 0) {
                    img.tv_frozen[q] = 1;
                    img.tv_frozen_vm[q] = f.target_vm;
                    break;
                }
        }
    }
}

// the buses the rows' moved generators bring to ReactiveLimits (a generator placed on another bus:
// that bus a controller bus of its plan), appended to the grid's bus_q entries for the device
// tables; each such bus with every generator a row moves there (sorted by id, the plans' order).
// A generator a row moves onto a bus the grid's plan already has (a re-merge) joins that bus'
// entry: the rows not moving it there have it off (device_slot_image's rl_goff). Each row's plan
// is the grid's filtered (filtered_bus_q_plan, as its load builds it); the same moves as last
// time keep the tables.
bool Ls2gOuterModel::device_prepare_rows(const RowOverrides& ov)
{
    const bool had = rl_dev_ != nullptr;
    bool any_move = false;
    for (const auto& m : ov.gen_bus) any_move = any_move || !m.empty();
    if (!any_move) {
        rl_dev_moves_.clear();
        rl_dev_built_ = false;
        rl_dev_.reset();
        const bool had_ds = !ds_dev_move_bus_.empty();
        ds_dev_move_bus_.clear();
        return had || had_ds;
    }
    if (rl_dev_built_ && ov.gen_bus == rl_dev_moves_) return false;
    rl_dev_moves_ = ov.gen_bus;
    rl_dev_built_ = true;
    // the buses DistributedSlack participants are moved to (load's filtered units: a moved one on
    // its row bus), whether or not ReactiveLimits brings them too
    std::vector<int> ds_move_bus;
    for (const auto& moves : ov.gen_bus)
        for (const auto& gb : moves) {
            if (gb.second < 0) continue;
            for (size_t k = 0; k < units0_.size(); ++k)
                if (units0_[k].kind == ls2g::slack_redistribution::UnitKind::GENERATOR && units0_[k].el_id == gb.first &&
                    units0_bus_[k] != gb.second)
                    ds_move_bus.push_back(gb.second);
        }
    std::sort(ds_move_bus.begin(), ds_move_bus.end());
    ds_move_bus.erase(std::unique(ds_move_bus.begin(), ds_move_bus.end()), ds_move_bus.end());
    const bool ds_changed = ds_move_bus != ds_dev_move_bus_;
    ds_dev_move_bus_ = std::move(ds_move_bus);
    std::map<int, std::pair<RlBus, ls2g::bus_q_check::BusQEntry>> extra;
    std::map<size_t, std::vector<int>> joining;   // grid bus_q entry k -> generators rows move there
    for (size_t r = 0; r < ov.gen_bus.size(); ++r) {
        if (ov.gen_bus[r].empty()) continue;
        SlotState st;
        st.rowv.layout = &sbl_;
        st.rowv.gens.bus_solver.assign(static_cast<size_t>(n_gen_), ls2g::OuterRowView::Units::GRID_BUS);
        for (const auto& gb : ov.gen_bus[r])
            if (gb.first >= 0 && gb.first < n_gen_) st.rowv.gens.bus_solver[static_cast<size_t>(gb.first)] = gb.second;
        const auto plan = build_rl_plan(view(st), nullptr, nullptr, /*filtered=*/true, &ov.gen_bus[r]);
        for (size_t k = 0; k < plan->n_bus_q; ++k) {
            const RlBus& rb = plan->buses[k];
            const auto& entry = plan->plan->buses[static_cast<size_t>(rb.entry)];
            size_t q = 0;
            while (q < rl0_->n_bus_q && rl0_->buses[q].bus_solver != rb.bus_solver) ++q;
            if (q < rl0_->n_bus_q) {
                const auto& grid_ids = rl0_->plan->buses[static_cast<size_t>(rl0_->buses[q].entry)].gen_ids;
                for (int g : entry.gen_ids)
                    if (std::find(grid_ids.begin(), grid_ids.end(), g) == grid_ids.end()) joining[q].push_back(g);
                continue;
            }
            auto it = extra.find(rb.bus_solver);
            if (it == extra.end()) {
                extra.emplace(rb.bus_solver, std::make_pair(rb, entry));
            } else {
                auto& ids = it->second.second.gen_ids;
                for (int g : entry.gen_ids) if (std::find(ids.begin(), ids.end(), g) == ids.end()) ids.push_back(g);
                std::sort(ids.begin(), ids.end());
            }
        }
    }
    if (extra.empty() && joining.empty()) {
        rl_dev_.reset();
        return had || ds_changed;
    }
    auto plan = std::make_shared<ls2g::bus_q_check::BusQPlan>(*rl0_->plan);
    auto rl = std::make_shared<RlPlan>();
    rl->plan = plan;
    rl->has_monitor = rl0_->has_monitor;
    // the grid's bus_q entries (their plan entries kept) and the moved generators' buses, in grid
    // bus order -- every row's plan lists its buses that way (build_bus_q_plan walks the grid's
    // buses), so each is a subsequence of this one in its own order (device_slot_image's search,
    // the decisions trace's order); a real grid's spare buses sit between its in-service ones --,
    // then the monitors
    std::vector<ls2g::bus_q_check::BusQEntry> entries(plan->buses.begin(), plan->buses.begin() + static_cast<ptrdiff_t>(rl0_->n_bus_q));
    for (auto& [k, gens] : joining) {
        auto& ids = entries[static_cast<size_t>(rl0_->buses[k].entry)].gen_ids;
        for (int g : gens) if (std::find(ids.begin(), ids.end(), g) == ids.end()) ids.push_back(g);
        std::sort(ids.begin(), ids.end());
    }
    struct Placed { int bus_grid; RlBus bus; RlBusState state; };
    std::vector<Placed> placed;
    for (size_t k = 0; k < rl0_->n_bus_q; ++k)
        placed.push_back(Placed{entries[static_cast<size_t>(rl0_->buses[k].entry)].bus_grid, rl0_->buses[k], rl0_->state0[k]});
    for (auto& kv : extra) {
        RlBus b = kv.second.first;
        b.entry = static_cast<int>(entries.size());
        b.has_base = false;
        entries.push_back(kv.second.second);
        placed.push_back(Placed{kv.second.second.bus_grid, b, RlBusState()});
    }
    std::stable_sort(placed.begin(), placed.end(), [](const Placed& a, const Placed& b) { return a.bus_grid < b.bus_grid; });
    for (Placed& p : placed) { rl->buses.push_back(p.bus); rl->state0.push_back(p.state); }
    rl->n_bus_q = rl->buses.size();
    for (size_t k = rl0_->n_bus_q; k < rl0_->buses.size(); ++k) {
        RlBus b = rl0_->buses[k];
        entries.push_back(plan->buses[static_cast<size_t>(b.entry)]);
        b.entry = static_cast<int>(entries.size()) - 1;
        rl->buses.push_back(b);
        rl->state0.push_back(rl0_->state0[k]);
    }
    plan->buses = std::move(entries);
    rl_dev_ = rl;
    return true;
}

void Ls2gOuterModel::device_row_blocks(const int* pos, const int* rpos, std::vector<int>& trafo,
                                      std::vector<double>& block) const
{
    trafo.clear();
    block.clear();
    const auto& trafos = grid_->get_trafos();
    const auto& ptc = trafos.get_tap_changers(true);
    const auto& rtc = trafos.get_tap_changers(false);
    for (int e = 0; e < n_bc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        const int t = bc_trafo_[ee];
        const bool ph = bc_phase_[ee] && !(ptc.has(t) && ptc.position(t) == pos[e]);
        const bool ra = bc_ratio_[ee] && !(rtc.has(t) && rtc.position(t) == rpos[e]);
        if (!ph && !ra) continue;
        const int rp = ra ? rpos[e] : (rtc.has(t) ? rtc.position(t) : 0);
        const int pp = ph ? pos[e] : (ptc.has(t) ? ptc.position(t) : 0);
        const double shift = ptc.has(t) ? ptc.alpha_at(t, pp) : trafos[t].shift_rad;
        const auto blk = trafos.pi_block_at(t, rp, pp, shift);
        trafo.push_back(t);
        for (int k = 0; k < 4; ++k) { block.push_back(blk[static_cast<size_t>(k)].real()); block.push_back(blk[static_cast<size_t>(k)].imag()); }
    }
}

// adjoint_info line by line, on the state the device read off the row's slot (the grid plan's buses
// and units, those of the row present)
void Ls2gOuterModel::device_adjoint_info(const DeviceAdjointState& a, RowResult& r) const
{
    for (size_t k = 0; k < a.pin.size(); ++k)
        if (!a.pin[k]) r.pq_buses.push_back(layout_.pin_bus[k]);
    auto held = [&](int j) {
        return std::isfinite(a.held_q[static_cast<size_t>(j)]) || a.plan_held[static_cast<size_t>(j)] != 0;
    };
    auto ref_active = [&](int g) {
        for (int off = 0; off < grp_count_[static_cast<size_t>(g)]; ++off)
            if (!held(grp_start_[static_cast<size_t>(g)] + off)) return off;
        return -1;
    };
    for (int g = 0; g < layout_.n_grp; ++g)
        if (ref_active(g) < 0) r.vset_lost_groups.push_back(g);
    auto hold_row = [&](int j) {
        if (j < 0 || j >= layout_.n_ctrl || !held(j)) return -1;
        const int g = vc_group_[static_cast<size_t>(j)];
        if (static_cast<size_t>(g) >= vc_v_row_.size()) return -1;
        const auto row_of = [&](int k) {
            return k == 0 ? vc_v_row_[static_cast<size_t>(g)] : vc_share_row_[static_cast<size_t>(g)][static_cast<size_t>(k - 1)];
        };
        const int off = j - grp_start_[static_cast<size_t>(g)];
        const int ref = ref_active(g);
        return (ref >= 0 && off == 0) ? row_of(ref) : row_of(off);
    };
    if (a.ds_in_newton)
        for (size_t k = 0; k < units0_.size() && k < a.ds_present.size(); ++k) {
            if (!a.ds_present[k] || !a.ds_solved_parked[k]) continue;
            if (units0_[k].kind != ls2g::slack_redistribution::UnitKind::GENERATOR) continue;
            if (a.ds_solved_mw[k] == a.ds_initial_mw[k]) continue;
            r.gen_p_fixed.push_back(units0_[k].el_id);
        }
    for (size_t q = 0; q < a.slope_k0.size(); ++q) {
        const size_t k0 = static_cast<size_t>(a.slope_k0[q]);
        if (!a.rl_present[k0] || a.rl_state[k0] == 0) continue;
        const int g = a.slope_gen[q], ctrl = a.slope_ctrl[q];
        if (std::find(r.gen_p_fixed.begin(), r.gen_p_fixed.end(), g) != r.gen_p_fixed.end()) continue;
        const int row = ctrl >= 0 ? hold_row(ctrl) : q_row_of_bus_[static_cast<size_t>(rl_dev().buses[k0].bus_solver)];
        if (row < 0) continue;
        r.gen_p_slope_gen.push_back(g);
        r.gen_p_slope_row.push_back(row);
        r.gen_p_slope_coeff.push_back(a.slope[q]);
    }
}

std::vector<DeviceFeature> outer_device_census(const ls2g::LSGrid& grid, const LedgerData& ledger,
                                               const Ls2gModelOptions& options)
{
    return Ls2gOuterModel(grid, ledger, options).census();
}

// =============================================================================
// The device model's inputs (docs/outer_loops_on_device.md, §4.6)
// =============================================================================
DeviceTables Ls2gOuterModel::device_tables() const
{
    DeviceTables t;
    t.n_bus = n_bus_; t.n_gen = n_gen_; t.n_sto = n_sto_; t.n_load = n_load_;
    const auto& svcs = grid_->get_svcs();
    const auto& gens = grid_->get_generators();
    t.n_svc = svcs.nb();
    t.sn = sn_;
    t.slack_bus = slack_bus_;
    t.max_outer_iterations = params_.max_outer_iterations;
    t.robust_mode = params_.voltage_remote_control_robust_mode ? 1 : 0;
    t.widx = widx_;
    t.n_watch = static_cast<int>(layout_.watch_bus.size());
    t.keep_trace = opt_.keep_trace ? 1 : 0;
    for (const LoopDecl& d : loops_) {
        t.loop_name.push_back(d.name);
        switch (d.kind) {
            case Kind::DistributedSlack:
                t.loop_kind.push_back(DL_DISTRIBUTED_SLACK);
                t.ds_mismatch_mw = d.ds_mismatch_mw; t.ds_eps_mw = d.ds_eps_mw;
                t.ds_moved_fraction = d.ds_moved_fraction; t.ds_fail = d.ds_fail;
                t.ds_run_without_participant = d.ds_run_without_participant;
                break;
            case Kind::HvdcLimits:        t.loop_kind.push_back(DL_HVDC_LIMITS); break;
            case Kind::VoltageMonitoring: t.loop_kind.push_back(DL_VOLTAGE_MONITORING); break;
            case Kind::ShuntVoltageControl: t.loop_kind.push_back(DL_SHUNT_VOLTAGE_CONTROL); break;
            case Kind::PhaseControl: t.loop_kind.push_back(DL_PHASE_CONTROL); break;
            case Kind::TransformerVoltageControl:
                t.loop_kind.push_back(DL_TRANSFORMER_VOLTAGE_CONTROL);
                t.has_tvc = 1;
                t.tv_use_initial = d.tvc->params().use_initial_tap_position ? 1 : 0;
                break;
            case Kind::ReactiveLimits:
                t.loop_kind.push_back(DL_REACTIVE_LIMITS);
                t.rl_max_switch = d.rl_max_switch; t.rl_robust = d.rl_robust;
                t.rl_max_mismatch = d.rl_max_mismatch; t.rl_base_mva = d.rl_base_mva;
                t.rl_vmin = d.rl_vmin; t.rl_vmax = d.rl_vmax; t.rl_margin = d.rl_margin; t.rl_restart = d.rl_restart;
                break;
            default:
                throw std::domain_error("the loop " + d.name);
        }
    }
    // generators and storage units: the base targets in force (a row that sets none)
    SlotState plain;
    plain.rowv.layout = &sbl_;
    const ls2g::OuterGridView v = view(plain);
    const auto raw_q = grid_->get_gen_raw_target_q();
    for (int g = 0; g < n_gen_; ++g) {
        t.gen_target_p.push_back(static_cast<double>(v.gen_target_p(g)));
        t.gen_min_q.push_back(static_cast<double>(gens.get_min_q(g)));
        t.gen_max_q.push_back(static_cast<double>(gens.get_max_q(g)));
        t.gen_target_q.push_back(static_cast<double>(gens.get_target_q()(g)));
        t.gen_raw_target_q.push_back(raw_q.size() == n_gen_ ? static_cast<double>(raw_q(g)) : NaN);
        t.gen_vreg_on.push_back(gens.get_voltage_regulator_on(g) ? 1 : 0);
    }
    {
        // LSGrid::set_gen_capability_curves's storage, read back through its state
        const ls2g::LSGrid::StateRes state = grid_->get_state();
        const auto& cid = std::get<ls2g::LSGrid::GEN_CURVE_GEN_ID_ID>(state);
        const auto& cp = std::get<ls2g::LSGrid::GEN_CURVE_P_ID>(state);
        const auto& cmin = std::get<ls2g::LSGrid::GEN_CURVE_MIN_Q_ID>(state);
        const auto& cmax = std::get<ls2g::LSGrid::GEN_CURVE_MAX_Q_ID>(state);
        std::vector<std::vector<std::array<double, 3>>> curves(static_cast<size_t>(n_gen_));
        for (size_t k = 0; k < cid.size(); ++k)
            if (cid[k] >= 0 && cid[k] < n_gen_)
                curves[static_cast<size_t>(cid[k])].push_back({static_cast<double>(cp[k]), static_cast<double>(cmin[k]),
                                                              static_cast<double>(cmax[k])});
        t.curve_ptr.push_back(0);
        for (auto& c : curves) {
            std::stable_sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });
            for (const auto& pt : c) { t.curve_p.push_back(pt[0]); t.curve_min_q.push_back(pt[1]); t.curve_max_q.push_back(pt[2]); }
            t.curve_ptr.push_back(static_cast<int>(t.curve_p.size()));
        }
    }
    t.gen_rl_bus_ptr.push_back(0);
    for (const auto& buses : gen_rl_bus_) {
        t.gen_rl_bus.insert(t.gen_rl_bus.end(), buses.begin(), buses.end());
        t.gen_rl_bus_ptr.push_back(static_cast<int>(t.gen_rl_bus.size()));
    }
    for (int s = 0; s < n_sto_; ++s) t.sto_target_p.push_back(static_cast<double>(v.storage_target_p(s)));
    for (int sid = 0; sid < t.n_svc; ++sid) {
        t.svc_b_min.push_back(static_cast<double>(svcs.get_b_min(sid)));
        t.svc_b_max.push_back(static_cast<double>(svcs.get_b_max(sid)));
        t.svc_standby_low_target.push_back(static_cast<double>(svcs.get_standby_low_target_vm_pu(sid)));
        t.svc_standby_high_target.push_back(static_cast<double>(svcs.get_standby_high_target_vm_pu(sid)));
        t.svc_reg_is_own_bus.push_back(svcs.get_regulated_bus_id(sid) == svcs.get_bus_id()(sid).cast_int() ? 1 : 0);
    }
    t.held_svcs = held_svcs_;
    const auto& dcl = grid_->get_dclines();
    for (int h = 0; h < dcl.nb(); ++h) {
        t.station_min_q1.push_back(static_cast<double>(dcl.get_station_min_q_mvar(h, 1)));
        t.station_max_q1.push_back(static_cast<double>(dcl.get_station_max_q_mvar(h, 1)));
        t.station_min_q2.push_back(static_cast<double>(dcl.get_station_min_q_mvar(h, 2)));
        t.station_max_q2.push_back(static_cast<double>(dcl.get_station_max_q_mvar(h, 2)));
    }
    // ReactiveLimits: the grid's plan
    t.rl_n = static_cast<int>(rl_dev().buses.size());
    t.rl_n_bus_q = static_cast<int>(rl_dev().n_bus_q);
    t.rl_has_monitor = rl_dev().has_monitor ? 1 : 0;
    for (auto* p : {&t.rl_gen_ptr, &t.rl_svc_ptr, &t.rl_sto_ptr, &t.rl_ctrl_ptr}) p->push_back(0);
    for (size_t k = 0; k < rl_dev().buses.size(); ++k) {
        const RlBus& b = rl_dev().buses[k];
        t.rl_bus.push_back(b.bus_solver); t.rl_local.push_back(b.local ? 1 : 0);
        t.rl_reg_bus.push_back(b.reg_bus_solver); t.rl_group.push_back(b.group);
        t.rl_monitor_svc.push_back(b.monitor_svc); t.rl_pin.push_back(b.pin);
        t.rl_has_base.push_back(b.has_base ? 1 : 0); t.rl_has_storage.push_back(b.has_storage ? 1 : 0);
        t.rl_min_q.push_back(b.min_q); t.rl_max_q.push_back(b.max_q); t.rl_target_vm.push_back(b.target_vm);
        t.rl_nominal_v.push_back(b.nominal_v); t.rl_target_q.push_back(b.target_q);
        t.rl_base_min_q.push_back(b.base_min_q); t.rl_base_max_q.push_back(b.base_max_q);
        const auto& e = rl_dev().plan->buses[static_cast<size_t>(b.entry)];
        t.rl_gen.insert(t.rl_gen.end(), e.gen_ids.begin(), e.gen_ids.end());
        t.rl_svc.insert(t.rl_svc.end(), e.svc_ids.begin(), e.svc_ids.end());
        t.rl_sto.insert(t.rl_sto.end(), e.storage_ids.begin(), e.storage_ids.end());
        t.rl_ctrl.insert(t.rl_ctrl.end(), e.ctrl_pos.begin(), e.ctrl_pos.end());
        t.rl_gen_ptr.push_back(static_cast<int>(t.rl_gen.size()));
        t.rl_svc_ptr.push_back(static_cast<int>(t.rl_svc.size()));
        t.rl_sto_ptr.push_back(static_cast<int>(t.rl_sto.size()));
        t.rl_ctrl_ptr.push_back(static_cast<int>(t.rl_ctrl.size()));
        const RlBusState& s0 = rl_dev().state0[k];
        t.rl0_target_p.push_back(s0.target_p); t.rl0_frozen_q.push_back(s0.frozen_q);
        t.rl0_state.push_back(s0.state); t.rl0_nb_pv_pq.push_back(s0.nb_pv_pq); t.rl0_realistic.push_back(s0.realistic ? 1 : 0);
        // the state the n point left it frozen in (resume_loops re-freezes it there)
        const auto fz = resume_ ? point_n_.frozen.find(b.bus_solver) : point_n_.frozen.end();
        t.rl_rp_state.push_back(fz == point_n_.frozen.end() ? 0 : fz->second.first);
    }
    // VoltageControl
    t.n_ctrl = layout_.n_ctrl; t.n_grp = layout_.n_grp;
    t.vc_bus = vc_bus_; t.vc_kind = vc_kind_; t.vc_elem = vc_elem_; t.vc_group = vc_group_;
    t.vc_plan_held = vc_plan_held_; t.vc_slope = vc_slope_; t.vc_q_plan = vc_q_plan_; t.vc_v_set = vc_v_set_;
    t.vc_at_bus_ptr.push_back(0);
    for (const auto& cs : vc_at_bus_) {
        t.vc_at_bus.insert(t.vc_at_bus.end(), cs.begin(), cs.end());
        t.vc_at_bus_ptr.push_back(static_cast<int>(t.vc_at_bus.size()));
    }
    t.svc_ctrl = svc_ctrl_;
    // droop hvdc and the HvdcLimits plan
    t.n_hvdc = layout_.n_hvdc;
    t.hvdc_bus1 = hvdc_bus1_; t.hvdc_bus2 = hvdc_bus2_; t.hvdc_status0 = hvdc_status0_; t.hvdc_id = hvdc_id_;
    t.hvdc_p0 = hvdc_p0_; t.hvdc_k = hvdc_k_; t.hvdc_lf1 = hvdc_lf1_; t.hvdc_lf2 = hvdc_lf2_; t.hvdc_r = hvdc_r_;
    t.hvdc_pmax12 = hvdc_pmax12_; t.hvdc_pmax21 = hvdc_pmax21_;
    for (const auto& line : hvdc_plan0_.lines) {
        t.hl_hvdc_id.push_back(line.hvdc_id); t.hl_bus1.push_back(line.bus1_solver); t.hl_bus2.push_back(line.bus2_solver);
        t.hl_p0_mw.push_back(static_cast<double>(line.p0_mw)); t.hl_k_mw_per_rad.push_back(static_cast<double>(line.k_mw_per_rad));
        t.hl_pmax12_mw.push_back(static_cast<double>(line.pmax_1to2_mw)); t.hl_pmax21_mw.push_back(static_cast<double>(line.pmax_2to1_mw));
    }
    for (const auto& m : monitors_.svcs) {
        t.mon_svc.push_back(m.svc_id); t.mon_reg_bus.push_back(m.reg_bus_solver); t.mon_svc_bus.push_back(m.svc_bus_solver);
        t.mon_low_vm.push_back(static_cast<double>(m.low_vm_pu)); t.mon_high_vm.push_back(static_cast<double>(m.high_vm_pu));
        t.mon_vn_kv.push_back(static_cast<double>(m.vn_kv));
    }
    // DistributedSlack's participants
    for (size_t k = 0; k < units0_.size(); ++k) {
        const auto& u = units0_[k];
        t.ds_el_id.push_back(u.el_id);
        t.ds_kind.push_back(u.kind == ls2g::slack_redistribution::UnitKind::GENERATOR ? 0 : 1);
        t.ds_bus.push_back(units0_bus_[k]);
        t.ds_in_slack.push_back(u.in_slack ? 1 : 0);
        t.ds_weight.push_back(static_cast<double>(u.weight));
        t.ds_min_mw.push_back(static_cast<double>(u.min_p_mw));
        t.ds_max_mw.push_back(static_cast<double>(u.max_p_mw));
        if (u.overshoot_mw != 0.) throw std::domain_error("a DistributedSlack unit with an overshoot");
    }
    // what a row's loss reads of the n point (share_lost_p: DistributedSlackLoop::lost_p_to_share_mw
    // on a row that sets no target, the same moves in the same order; resume_loops' V)
    if (resume_) {
        SlotState plain;
        plain.rowv.layout = &sbl_;
        plain.rp = &point_n_;
        const ls2g::OuterGridView v = view(plain);
        t.rp_n = 1;
        t.rp_n_parked = static_cast<int>(point_n_.parked.size());
        for (size_t gid = 0; gid < point_n_.gen_p.size(); ++gid) {
            const int g = static_cast<int>(gid);
            if (std::isnan(point_n_.gen_p[gid])) continue;
            t.rp_gen_bus.push_back(v.gen_on(g) ? v.gen_bus_solver(g) : -2);
            t.rp_gen_move.push_back(point_n_.gen_p[gid] - v.gen_target_p(g));
        }
        for (size_t sid = 0; sid < point_n_.sto_p.size(); ++sid) {
            const int s = static_cast<int>(sid);
            if (std::isnan(point_n_.sto_p[sid])) continue;
            t.rp_sto_bus.push_back(v.storage_on(s) ? v.storage_bus_solver(s) : -2);
            t.rp_sto_move.push_back(point_n_.sto_p[sid] - v.storage_target_p(s));
        }
        const size_t nw = layout_.watch_bus.size();
        t.rp_v.assign(2 * nw, NaN);
        for (size_t k = 0; k < nw; ++k) {
            const int b = layout_.watch_bus[k];
            cplx vb(NaN, NaN);
            if (point_n_.V.size() > 0) {
                if (b >= 0 && b < static_cast<int>(point_n_.V.size())) vb = cplx(point_n_.V(b).real(), point_n_.V(b).imag());
            } else {
                const auto it = std::lower_bound(point_n_.v_at.begin(), point_n_.v_at.end(), b,
                                                 [](const std::pair<int, cplx>& e, int key) { return e.first < key; });
                if (it != point_n_.v_at.end() && it->first == b) vb = it->second;
            }
            t.rp_v[k] = vb.real();
            t.rp_v[nw + k] = vb.imag();
        }
    }
    // ShuntControl's controllers and ShuntVoltageControl's groups
    {
        const auto& shunts = grid_->get_shunts();
        t.n_sc = n_sc_;
        t.sc_sh_ptr.push_back(0);
        t.sh_b_ptr.push_back(0);
        for (int e = 0; e < n_sc_; ++e) {
            t.sc_bus.push_back(sc_bus_[static_cast<size_t>(e)]);
            for (int sh : sc_shunts_[static_cast<size_t>(e)]) {
                t.sh_id.push_back(sh);
                t.sh_min.push_back(shunts.get_min_section_count(sh));
                t.sh_max.push_back(shunts.get_max_section_count(sh));
                t.sh_count0.push_back(shunts.get_section_count(sh));
                t.sh_status.push_back(shunts.get_status()[static_cast<size_t>(sh)] ? 1 : 0);
                t.sh_has.push_back(shunts.has_sections(sh) ? 1 : 0);
                for (int count = 0; count <= shunts.get_max_section_count(sh); ++count)
                    t.sh_b.push_back(-static_cast<double>(shunts.section_q(sh, count)) / sn_);
                t.sh_b_ptr.push_back(static_cast<int>(t.sh_b.size()));
            }
            t.sc_sh_ptr.push_back(static_cast<int>(t.sh_id.size()));
        }
        t.n_sc_sh = static_cast<int>(t.sh_id.size());
        bool has_sv = false;
        for (const LoopDecl& d : loops_) has_sv = has_sv || d.kind == Kind::ShuntVoltageControl;
        t.svg_ptr.push_back(0);
        if (has_sv) {
            SlotState plain_sv;
            plain_sv.rowv.layout = &sbl_;
            for (const auto& gr : ls2g::ShuntVoltageControlLoop::groups(view(plain_sv), tvc_buses_)) {
                for (size_t c = 0; c < gr.controller_buses.size(); ++c) {
                    const int bus = gr.controller_buses[c];
                    const int e = (bus >= 0 && bus < n_bus_) ? sc_of_bus_[static_cast<size_t>(bus)] : -1;
                    if (e >= 0 && gr.shunts[c] != sc_shunts_[static_cast<size_t>(e)])
                        throw std::domain_error("a shunt voltage group whose shunts are not its controller's");
                    t.svg_e.push_back(e);
                    t.svg_bus.push_back(bus);
                }
                t.svg_ptr.push_back(static_cast<int>(t.svg_e.size()));
            }
        }
        t.n_svg = static_cast<int>(t.svg_ptr.size()) - 1;
    }
    // BranchControl's transformers: what a retap reads (BranchControl::_retap), per (ratio tap, phase tap)
    {
        const auto& trafos = grid_->get_trafos();
        const auto& ptc = trafos.get_tap_changers(true);
        const auto& rtc = trafos.get_tap_changers(false);
        t.n_bc = n_bc_;
        t.bc_tab_ptr.push_back(0);
        t.bc_alpha_ptr.push_back(0);
        for (int e = 0; e < n_bc_; ++e) {
            const size_t ee = static_cast<size_t>(e);
            const int tr = bc_trafo_[ee];
            const bool hp = ptc.has(tr), hr = rtc.has(tr);
            const int pos0 = hp ? ptc.position(tr) : 0, rpos0 = hr ? rtc.position(tr) : 0;
            const int plo = hp ? ptc.low_tap(tr) : pos0, phi = hp ? ptc.high_tap(tr) : pos0;
            const int rlo = hr ? rtc.low_tap(tr) : rpos0, rhi = hr ? rtc.high_tap(tr) : rpos0;
            t.bc_trafo.push_back(tr); t.bc_phase.push_back(bc_phase_[ee] ? 1 : 0); t.bc_ratio.push_back(bc_ratio_[ee] ? 1 : 0);
            t.bc_b1.push_back(bc_b1_[ee]); t.bc_b2.push_back(bc_b2_[ee]);
            t.bc_sign.push_back(bc_sign_[ee]); t.bc_rexp.push_back(bc_rexp_[ee]);
            t.bc_has_ptc.push_back(hp ? 1 : 0); t.bc_has_rtc.push_back(hr ? 1 : 0);
            t.bc_pos0.push_back(pos0); t.bc_rpos0.push_back(rpos0);
            t.bc_plo.push_back(plo); t.bc_phi.push_back(phi); t.bc_rlo.push_back(rlo); t.bc_rhi.push_back(rhi);
            const double shift0 = static_cast<double>(trafos[tr].shift_rad);
            for (int rpos = rlo; rpos <= rhi; ++rpos)
                for (int pos = plo; pos <= phi; ++pos) {
                    const double a = hp ? static_cast<double>(ptc.alpha_at(tr, pos)) : shift0;
                    t.bc_tab_ratio.push_back(static_cast<double>(trafos.ratio_at(tr, rpos, pos)));
                    t.bc_tab_shift.push_back(a);
                    const auto blk = trafos.pi_block_at(tr, rpos, pos, a);
                    for (int k = 0; k < 4; ++k) {
                        t.bc_tab_block.push_back(static_cast<double>(blk[static_cast<size_t>(k)].real()));
                        t.bc_tab_block.push_back(static_cast<double>(blk[static_cast<size_t>(k)].imag()));
                    }
                }
            t.bc_tab_ptr.push_back(static_cast<int>(t.bc_tab_ratio.size()));
            for (int pos = plo; pos <= phi; ++pos)
                t.bc_alpha.push_back(hp ? static_cast<double>(ptc.alpha_at(tr, pos)) : shift0);
            t.bc_alpha_ptr.push_back(static_cast<int>(t.bc_alpha.size()));
        }
        // PhaseControl's shifters (the grid's)
        bool has_pc = false;
        for (const LoopDecl& d : loops_) has_pc = has_pc || d.kind == Kind::PhaseControl;
        if (has_pc) {
            SlotState plain_pc;
            plain_pc.rowv.layout = &sbl_;
            const auto vn_kv = grid_->get_bus_vn_kv();
            for (const auto& sh : ls2g::PhaseControlLoop::shifters(view(plain_pc))) {
                t.pc_trafo.push_back(sh.trafo);
                const int e = bc_of_trafo_[static_cast<size_t>(sh.trafo)];
                t.pc_e.push_back((e >= 0 && bc_phase_[static_cast<size_t>(e)]) ? e : -1);
                t.pc_mode.push_back(sh.mode == ls2g::RegulationMode::ACTIVE_POWER ? 0
                                    : sh.mode == ls2g::RegulationMode::CURRENT_LIMITER ? 1 : 2);
                t.pc_side.push_back(sh.side);
                const int bus = (sh.side == 1 ? trafos.get_bus_side_1(sh.trafo) : trafos.get_bus_side_2(sh.trafo)).cast_int();
                t.pc_vn_kv.push_back(bus >= 0 ? static_cast<double>(vn_kv(bus)) : NaN);
                t.pc_target.push_back(ptc.has(sh.trafo) ? static_cast<double>(ptc.target(sh.trafo)) : NaN);
            }
        }
        t.n_pc = static_cast<int>(t.pc_trafo.size());
        // TransformerVoltageControl's groups and the grid's branches
        t.pin_of_bus = pin_of_bus_;
        t.tvg_ptr.push_back(0);
        if (t.has_tvc) {
            const LoopDecl* L = nullptr;
            for (const LoopDecl& d : loops_) if (d.kind == Kind::TransformerVoltageControl) L = &d;
            SlotState plain_tv;
            plain_tv.rowv.layout = &sbl_;
            const ls2g::OuterGridView v0 = view(plain_tv);
            t.trafo_gt.assign(static_cast<size_t>(n_trafo_), -1);
            for (const auto& gr : L->tvc->groups(v0)) {
                t.tvg_bus.push_back(gr.bus_solver);
                t.tvg_target.push_back(static_cast<double>(gr.target));
                t.tvg_hdb.push_back(static_cast<double>(gr.half_deadband));
                for (int tr : gr.trafos) {
                    if (t.trafo_gt[static_cast<size_t>(tr)] >= 0)
                        throw std::domain_error("a transformer in two transformer voltage groups");
                    t.trafo_gt[static_cast<size_t>(tr)] = static_cast<int>(t.gt_trafo.size());
                    t.gt_trafo.push_back(tr);
                    t.gt_e.push_back(bc_entry(tr));
                }
                t.tvg_ptr.push_back(static_cast<int>(t.gt_trafo.size()));
            }
            int b1, b2;
            for (int el = 0; el < n_line_; ++el) {
                const bool c = v0.line_closed(el, b1, b2);
                t.br_b1.push_back(c ? b1 : -1); t.br_b2.push_back(c ? b2 : -1); t.br_trafo.push_back(-1);
            }
            for (int el = 0; el < n_trafo_; ++el) {
                const bool c = v0.trafo_closed(el, b1, b2);
                t.br_b1.push_back(c ? b1 : -1); t.br_b2.push_back(c ? b2 : -1); t.br_trafo.push_back(el);
            }
        }
        t.n_tvg = static_cast<int>(t.tvg_ptr.size()) - 1;
        t.n_gt = static_cast<int>(t.gt_trafo.size());
        t.n_br = static_cast<int>(t.br_b1.size());
    }
    t.dist = dist_ ? 1 : 0;
    t.ds_reserved = ds_reserved_ ? 1 : 0;
    t.n_slack = static_cast<int>(layout_.slack_bus.size());
    t.ref_slack = ref_slack_idx_;
    t.slack_idx_of_bus = dist_ ? slack_idx_of_bus_ : std::vector<int>(static_cast<size_t>(n_bus_), -1);
    t.n_pin = static_cast<int>(layout_.pin_bus.size());
    t.pin_bus = layout_.pin_bus;
    // the buses a Sbus correction may touch
    std::vector<int> corr(units0_bus_.begin(), units0_bus_.end());
    corr.insert(corr.end(), ds_dev_move_bus_.begin(), ds_dev_move_bus_.end());
    for (const RlBus& b : rl_dev().buses) if (b.local && b.bus_solver >= 0) corr.push_back(b.bus_solver);
    for (const auto& kv : resume0_target_delta_) corr.push_back(kv.first);
    if (resume_) for (const auto& kv : point_n_.frozen) corr.push_back(kv.first);
    std::sort(corr.begin(), corr.end());
    corr.erase(std::unique(corr.begin(), corr.end()), corr.end());
    t.corr_bus = corr;
    return t;
}

void Ls2gOuterModel::device_slot_image(int slot, DeviceSlotImage& img) const
{
    const SlotState& st = slots_[static_cast<size_t>(slot)];
    img = DeviceSlotImage();
    img.row = st.row;
    img.lost_mw = st.lost_mw;
    img.row_gen_p.assign(static_cast<size_t>(n_gen_), NaN);
    img.row_sto_p.assign(static_cast<size_t>(n_sto_), NaN);
    for (size_t g = 0; g < st.rowv.gens.target_p.size() && g < img.row_gen_p.size(); ++g) img.row_gen_p[g] = st.rowv.gens.target_p[g];
    for (size_t k = 0; k < st.rowv.storages.target_p.size() && k < img.row_sto_p.size(); ++k) img.row_sto_p[k] = st.rowv.storages.target_p[k];
    img.trace = st.trace;
    img.active = st.active;
    // BranchControl / PhaseControl
    for (int e = 0; e < n_bc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        img.bc_out.push_back(st.bc_out[ee]); img.bc_moved.push_back(st.bc_moved[ee]);
        img.bc_pos.push_back(st.bc_pos[ee]); img.bc_rpos.push_back(st.bc_rpos[ee]);
        img.bc_req_tap.push_back(st.bc_req_tap[ee]); img.bc_req_rtap.push_back(st.bc_req_rtap[ee]);
        img.bc_ctl.push_back(st.bc_ctl[ee]); img.bc_rctl.push_back(st.bc_rctl[ee]);
        img.bc_a_tap.push_back(st.bc_a_tap[ee]); img.bc_rho_tap.push_back(st.bc_rho_tap[ee]);
        for (int k = 0; k < 4; ++k) {
            img.bc_tap.push_back(st.bc_tap[4 * ee + static_cast<size_t>(k)].real());
            img.bc_tap.push_back(st.bc_tap[4 * ee + static_cast<size_t>(k)].imag());
        }
    }
    {
        bool has_pc = false;
        for (const LoopDecl& d : loops_) has_pc = has_pc || d.kind == Kind::PhaseControl;
        if (has_pc) {
            SlotState plain_pc;
            plain_pc.rowv.layout = &sbl_;
            const auto grid_sh = ls2g::PhaseControlLoop::shifters(view(plain_pc));
            img.pc_present.assign(grid_sh.size(), 0);
            img.pc_regulates.assign(grid_sh.size(), 0);
            // the row's shifters: those of the grid, some left out (in order), regulating or not
            size_t k0 = 0;
            for (const auto& sh : st.pc_shifters) {
                while (k0 < grid_sh.size() && !(grid_sh[k0].trafo == sh.trafo && grid_sh[k0].mode == sh.mode &&
                                                grid_sh[k0].side == sh.side)) ++k0;
                if (k0 == grid_sh.size()) throw std::domain_error("a row whose phase shifters are not the grid's");
                img.pc_present[k0] = 1;
                img.pc_regulates[k0] = sh.regulates ? 1 : 0;
                ++k0;
            }
        }
    }
    // TransformerVoltageControl
    device_tvc_image(st, img);
    // ShuntControl / ShuntVoltageControl
    for (int e = 0; e < n_sc_; ++e) {
        const size_t ee = static_cast<size_t>(e);
        const auto& c = st.sc_counts[ee];
        for (size_t i = 0; i < sc_shunts_[ee].size(); ++i) img.sc_counts.push_back(i < c.size() ? c[i] : 0);
        img.sc_pending.push_back(st.sc_pending[ee] ? 1 : 0);
        img.sc_b_new.push_back(st.sc_b_new[ee]);
        img.sc_ctl.push_back(st.sc_ctl[ee]);
    }
    {
        // the grid's groups, the ones this row hides (a row whose groups differ otherwise is refused)
        bool has_sv = false;
        for (const LoopDecl& d : loops_) has_sv = has_sv || d.kind == Kind::ShuntVoltageControl;
        if (has_sv) {
            SlotState plain_sv;
            plain_sv.rowv.layout = &sbl_;
            const auto grid_groups = ls2g::ShuntVoltageControlLoop::groups(view(plain_sv), tvc_buses_);
            img.sv_hidden.assign(grid_groups.size(), 1);
            if (!st.sv_groups.empty()) {
                if (st.sv_groups.size() != grid_groups.size())
                    throw std::domain_error("a row whose shunt voltage groups are not the grid's");
                for (size_t g = 0; g < grid_groups.size(); ++g) {
                    if (st.sv_groups[g].controller_buses != grid_groups[g].controller_buses ||
                        st.sv_groups[g].shunts != grid_groups[g].shunts)
                        throw std::domain_error("a row whose shunt voltage groups are not the grid's");
                    img.sv_hidden[g] = st.sv_groups[g].hidden ? 1 : 0;
                }
            }
        }
    }
    img.last_fixing = st.last_fixing;
    img.check_unrealistic = st.check_unrealistic ? 1 : 0;
    img.masked_bus.assign(static_cast<size_t>(n_bus_), 0);
    for (int b : st.masked) if (b >= 0 && b < n_bus_) img.masked_bus[static_cast<size_t>(b)] = 1;
    img.pq = st.pq;
    img.held_q = st.held_q;
    img.svc_on = st.svc_on;
    img.hvdc_regime = st.hvdc_regime;
    img.gen_target_p = st.gen_target_p.empty() ? std::vector<double>(static_cast<size_t>(n_gen_), NaN) : st.gen_target_p;
    img.sto_target_p = st.storage_target_p.empty() ? std::vector<double>(static_cast<size_t>(n_sto_), NaN) : st.storage_target_p;
    for (const auto& kv : st.target_delta) {
        img.delta_bus.push_back(kv.first); img.delta_p.push_back(kv.second.real()); img.delta_q.push_back(kv.second.imag());
    }
    for (const auto& kv : st.freeze_q) { img.freeze_bus.push_back(kv.first); img.freeze_q.push_back(kv.second); }
    img.vm_reset_bus = st.vm_reset_bus;
    img.vm_reset_vm = st.vm_reset_vm;
    img.lim_base = st.lim_base ? 1 : 0;
    for (size_t b = 0; b < st.lim_dirty.size(); ++b) if (st.lim_dirty[b]) img.lim_dirty_bus.push_back(static_cast<int>(b));
    // ReactiveLimits: the grid's plan, the row's buses present. A masked row's plan is a filter of
    // it (the same entries); a row's own plan (generators disconnected, voltages set) is mapped onto
    // it bus by bus: the same buses in the same order, each with the same members but generators,
    // its generators a subsequence of the grid bus' ones (the dropped ones: gen_off), its own
    // target Q / target |V| / base flag
    const RlPlan& P = rl_dev();
    const size_t rn = P.buses.size();
    img.rl_present.assign(rn, 0);
    img.rl_target_p.assign(rn, 0.); img.rl_frozen_q.assign(rn, 0.);
    img.rl_state.assign(rn, 0); img.rl_nb_pv_pq.assign(rn, 0); img.rl_realistic.assign(rn, 0);
    img.rl_target_q.resize(rn); img.rl_target_vm.resize(rn); img.rl_has_base.resize(rn);
    // each bus' first generator position in the tables' CSR (device_tables' order)
    std::vector<size_t> gen_pos(rn + 1, 0);
    for (size_t k = 0; k < rn; ++k) {
        img.rl_target_q[k] = P.buses[k].target_q;
        img.rl_target_vm[k] = P.buses[k].target_vm;
        img.rl_has_base[k] = P.buses[k].has_base ? 1 : 0;
        gen_pos[k + 1] = gen_pos[k] + P.plan->buses[static_cast<size_t>(P.buses[k].entry)].gen_ids.size();
    }
    img.rl_goff.assign(gen_pos[rn], 0);
    img.ctrl_off.assign(vc_kind_.size(), 0);
    if (st.rl) {
        const bool own = st.rl->plan != P.plan;
        size_t k0 = 0;
        for (size_t j = 0; j < st.rl->buses.size(); ++j) {
            const RlBus& rb = st.rl->buses[j];
            if (!own) {
                while (k0 < rn && P.buses[k0].entry != rb.entry) ++k0;
            } else {
                while (k0 < rn && !(P.buses[k0].bus_solver == rb.bus_solver &&
                                    P.buses[k0].monitor_svc == rb.monitor_svc)) ++k0;
            }
            if (k0 == rn) throw std::domain_error("a row whose ReactiveLimits plan is not a filter of the grid's");
            if (own) {
                const RlBus& gb = P.buses[k0];
                const auto& ge = P.plan->buses[static_cast<size_t>(gb.entry)];
                const auto& re = st.rl->plan->buses[static_cast<size_t>(rb.entry)];
                const char* diff = gb.local != rb.local ? "local" : gb.group != rb.group ? "group"
                    : gb.reg_bus_solver != rb.reg_bus_solver ? "regulated bus" : gb.pin != rb.pin ? "pin"
                    : gb.nominal_v != rb.nominal_v ? "nominal voltage" : gb.min_q != rb.min_q || gb.max_q != rb.max_q ? "fixed limits"
                    : ge.storage_ids != re.storage_ids ? "storage units" : ge.station_ids != re.station_ids ? "hvdc stations"
                    : ge.svc_ids != re.svc_ids ? "svcs" : nullptr;
                if (diff != nullptr)
                    throw std::domain_error(std::string("a row whose ReactiveLimits plan changes more than its generators (") +
                                            diff + " of bus " + std::to_string(rb.bus_solver) + ")");
                // its generators: the table bus' less the ones the row does not have there (per position)
                size_t q = 0;
                for (size_t i = 0; i < ge.gen_ids.size(); ++i) {
                    const bool kept = q < re.gen_ids.size() && re.gen_ids[q] == ge.gen_ids[i];
                    if (kept) ++q;
                    img.rl_goff[gen_pos[k0] + i] = kept ? 0 : 1;
                }
                if (q != re.gen_ids.size())
                    throw std::domain_error("a row whose ReactiveLimits bus gains a generator");
                // its controllers: the grid bus' without the ones the row trips (build_rl_plan's gone)
                q = 0;
                for (int c : ge.ctrl_pos) {
                    const bool kept = q < re.ctrl_pos.size() && re.ctrl_pos[q] == c;
                    if (kept) ++q;
                    else if (c >= 0 && static_cast<size_t>(c) < img.ctrl_off.size()) img.ctrl_off[static_cast<size_t>(c)] = 1;
                }
                if (q != re.ctrl_pos.size())
                    throw std::domain_error("a row whose ReactiveLimits bus gains a controller");
                img.rl_target_q[k0] = rb.target_q;
                img.rl_target_vm[k0] = rb.target_vm;
                img.rl_has_base[k0] = rb.has_base ? 1 : 0;
            }
            img.rl_present[k0] = 1;
            const RlBusState& rs = j < st.rl_state.size() ? st.rl_state[j] : st.rl->state0[j];
            img.rl_target_p[k0] = rs.target_p; img.rl_frozen_q[k0] = rs.frozen_q; img.rl_state[k0] = rs.state;
            img.rl_nb_pv_pq[k0] = rs.nb_pv_pq; img.rl_realistic[k0] = rs.realistic ? 1 : 0;
            ++k0;
        }
    }
    // DistributedSlack: the grid's units, the row's present (a masked row filters them)
    const size_t un = units0_.size();
    img.ds_present.assign(un, 0);
    img.ds_injection.assign(un, 0.); img.ds_current.assign(un, 0.); img.ds_initial.assign(un, 0.);
    img.ds_parked.assign(un, 0); img.ds_parked_high.assign(un, 0);
    img.ds_solved_parked.assign(un, 0); img.ds_solved_mw.assign(un, 0.);
    img.ds_bus = units0_bus_;   // an absent unit's is never read
    img.slope_set.assign(static_cast<size_t>(n_gen_), 0); img.slope.assign(static_cast<size_t>(n_gen_), 0.);
    for (const auto& frozen : st.rl_slopes)
        for (const auto& sl : frozen.second) {
            const int g = sl.first.first;
            if (g < 0 || g >= n_gen_) continue;
            img.slope_set[static_cast<size_t>(g)] = 1;
            img.slope[static_cast<size_t>(g)] = sl.second;
        }
    img.ds_in_newton = st.ds_in_newton ? 1 : 0;
    img.sa = st.sa;
    img.sa_shared = st.sa_shared ? 1 : 0;
    img.unsolved_q_mvar = st.unsolved_q_mvar;
    {
        size_t k0 = 0;
        for (size_t j = 0; j < st.ds_units.size(); ++j) {
            const auto& u = st.ds_units[j];
            while (k0 < un && !(units0_[k0].el_id == u.el_id && units0_[k0].kind == u.kind)) ++k0;
            if (k0 == un) throw std::domain_error("a row with its own DistributedSlack units");
            img.ds_present[k0] = 1;
            // a generator the row moves takes its share on its new bus (collect_units' solver bus)
            if (j < st.ds_unit_bus.size()) img.ds_bus[k0] = st.ds_unit_bus[j];
            img.ds_injection[k0] = static_cast<double>(u.injection_mw);
            if (j < st.ds_current_mw.size()) img.ds_current[k0] = st.ds_current_mw[j];
            if (j < st.ds_initial_mw.size()) img.ds_initial[k0] = st.ds_initial_mw[j];
            if (j < st.ds_parked.size()) img.ds_parked[k0] = st.ds_parked[j] ? 1 : 0;
            if (j < st.ds_parked_high.size()) img.ds_parked_high[k0] = st.ds_parked_high[j] ? 1 : 0;
            if (j < st.ds_solved_parked.size()) img.ds_solved_parked[k0] = st.ds_solved_parked[j] ? 1 : 0;
            if (j < st.ds_solved_mw.size()) img.ds_solved_mw[k0] = st.ds_solved_mw[j];
            ++k0;
        }
    }
    // HvdcLimits: the grid's lines in AC emulation, the row's present
    const size_t hn = hvdc_plan0_.lines.size();
    img.hl_present.assign(hn, 0);
    img.hl_frozen_dir.assign(hn, 0);
    {
        size_t k0 = 0;
        for (const auto& line : st.hvdc_plan.lines) {
            while (k0 < hn && hvdc_plan0_.lines[k0].hvdc_id != line.hvdc_id) ++k0;
            if (k0 == hn) throw std::domain_error("a row whose hvdc plan is not a filter of the grid's");
            img.hl_present[k0] = 1;
            img.hl_frozen_dir[k0] = line.frozen_dir;
            ++k0;
        }
    }
}

}  // namespace outer
