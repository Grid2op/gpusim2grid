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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
enum Status { STABLE = 0, UNSTABLE = 1, FAILED = 2 };   // OuterLoopStatus

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
    P_LOAD, P_LOAD_PHANTOM, P_LOAD_RL_PLAN, P_LOAD_BC, P_LOAD_SC, P_LOAD_UNITS, P_LOAD_SET_ACTIVE,
    P_LOAD_RESUME_TARGETS, P_LOAD_INIT_LOOPS, P_LOAD_RESUME_LOOPS, P_LOAD_RESET, P_LOAD_INIT_DS, P_LOAD_INIT_RL, P_LOAD_MASKED_ROWS, P_LOAD_DS_NUNITS, P_RL_BASE, P_RL_RECOMPUTE, P_RL_NO_BASE, P_N_SEC
};
const char* const kProfName[P_N_SEC] = {
    "on_solved (Resolve)", "on_solved (Done)", "  ds_fold", "  check DistributedSlack", "  check HvdcLimits",
    "  check VoltageMonitoring", "  check ReactiveLimits", "  check PhaseControl", "  check TVC",
    "  check ShuntVoltageControl", "  materialize", "  finish", "    adjoint_info",
    "load (row)", "load (phantom)", "  rl plan", "  BranchControl reset", "  ShuntControl reset",
    "  DistributedSlack units", "  set_active", "  resume targets", "  init_loops", "  resume_loops", "  reset the slot", "    init DistributedSlack", "    init ReactiveLimits", "rows that mask buses", "DS units copied", "rl bus: base sums", "rl bus: moved, recomputed", "rl bus: no base"};

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
    int entry;
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
struct SlotState {
    int row = -1;
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
    bool first = true;          // the first solve has not ended yet
    int  i = 0;                 // the loop being checked
    int  last_unstable = -1;
    int  last_status = STABLE;
    std::string last_name;
    bool solver_ok = true;
    bool check_unrealistic = false;     // for the solve in flight
    bool pass_check_unrealistic = false;
    int  old_total = 0;
    int  total_nr = 0;
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
    bool has(int b) const { return b >= 0 && b < static_cast<int>(widx->size()) && (*widx)[b] >= 0; }
    int  w(int b) const {
        if (!has(b)) throw_not_watched(b);
        return (*widx)[b];
    }
    cplx V(int b)    const {
        if (full != nullptr) return cplx((*full)(b).real(), (*full)(b).imag());
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
    if (st == 0) {
        if (raw >= 0.) { p1 = raw; p2 = -hvdc_recv(raw, true, lf1, lf2, r); }
        else           { p1 = -hvdc_recv(-raw, false, lf1, lf2, r); p2 = -raw; }
    } else if (st > 0) { p1 = pmax12; p2 = -hvdc_recv(pmax12, true, lf1, lf2, r); }
    else               { p1 = -hvdc_recv(pmax21, false, lf1, lf2, r); p2 = pmax21; }
}

class Ls2gOuterModel final : public OuterHook {
public:
    Ls2gOuterModel(const ls2g::LSGrid& grid, const LedgerData& ledger, const Ls2gModelOptions& opt);

    const Layout& layout() const override { return layout_; }
    void begin(int n_slots, int n_rows) override;
    void load(int slot, int row) override;
    void move(int from, int to) override;
    const SlotControls& controls(int slot) const override { return slots_[static_cast<size_t>(slot)].ctl; }
    void clear_vm_reset(int slot) override;
    Next on_solved(int slot, bool converged, int nr_iter, int inner_status,
                   const SlotData* data, int& row_status) override;
    Next on_solved_impl(int slot, bool converged, int nr_iter, int inner_status,
                        const SlotData* data, int& row_status);
    const RowResult& row_result(int row) const override { return results_.at(static_cast<size_t>(row)); }
    void row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const override;
    int n_rows() const override { return n_rows_; }
    void set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p) override;
    void set_row_overrides(RowOverrides overrides) override { ov_ = std::move(overrides); }

private:
    // ---- static data -------------------------------------------------------------
    // a copy of the grid: its containers, capability curves and raw targets, what
    // the loops read at run time. Its solver layout is NOT copied (a copy starts
    // cold): everything that needs it is read off the caller's grid at construction.
    std::shared_ptr<ls2g::LSGrid> grid_;
    ls2g::SolverBusLayout sbl_;          // the caller's grid's AC labelling (a copy has none)
    ls2g::OuterGridView view(const SlotState& st) const { return ls2g::OuterGridView(*grid_, &st.rowv); }
    Ls2gModelOptions opt_;
    mutable OcProfile prof_;              // GPUSIM2GRID_OC_PROFILE
    Layout layout_;
    double sn_ = 1.;
    int n_bus_ = 0, n_gen_ = 0, n_sto_ = 0;
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
    ls2g::hvdc_p_check::HvdcPPlan hvdc_plan0_;     // the lines in AC emulation
    ls2g::svc_standby_check::SvcStandbyPlan monitors_;
    std::vector<int> svc_ctrl_;                    // svc id -> its held controller (-1)
    std::vector<int> held_svcs_;                   // the standby SVCs the plan holds, sorted
    // where the grid's own solve left the loops (Ls2gModelOptions::resume_from_n)
    bool resume_ = false;
    std::vector<double> resume_gen_p_, resume_sto_p_;   // moved targets, MW, NaN untouched
    ls2g::CplxVect resume_V_;
    std::map<int, int> resume_hvdc_;                    // hvdc id -> regime
    std::vector<std::pair<int, double>> resume_svc_on_;
    std::map<int, std::pair<int, bool>> resume_frozen_; // solver bus -> (state, realistic)
    std::vector<ls2g::DistributedSlackLoop::ParkedUnit> resume_parked_;
    std::vector<int> resume_phase_tap_, resume_ratio_tap_;   // by trafo id, INT_MIN: not carried
    std::vector<int> resume_sections_;                       // by shunt id, INT_MIN: not carried
    // take_resume_targets on a row that sets no target and disconnects no unit
    std::vector<double> resume0_gen_target_p_, resume0_sto_target_p_;
    FlatMap<int, cplx> resume0_target_delta_;
    // ---- per run ------------------------------------------------------------------
    int n_rows_ = 0;
    std::vector<double> row_gen_p_, row_sto_p_;    // [n_rows * n_gen], [n_rows * n_sto]
    RowOverrides ov_;                              // the rows' masks, units off, set-points
    std::vector<SlotState> slots_;
    std::vector<RowResult> results_;
    std::vector<std::vector<int>>    row_blk_trafo_;   // row_trafo_blocks, per row
    std::vector<std::vector<double>> row_blk_;

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
    std::shared_ptr<const RlPlan> build_rl_plan(const ls2g::OuterGridView& v) const;
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
    void collect_units(const ls2g::OuterGridView& v, std::vector<ls2g::slack_redistribution::Participant>& units,
                       std::vector<int>& bus) const;
    void set_active(SlotState& st) const;
    void add_injection(SlotState& st, int bus, cplx ds) const {
        if (ds == cplx(0., 0.)) return;
        st.target_delta[bus] += ds;
    }
    double move_gen(SlotState& st, int g, int bus, double mw) const;
    double move_storage(SlotState& st, int s, int bus, double mw) const;
    void record(SlotState& st, const std::string& loop, const std::string& action, bool taken,
                int element, double value, double limit) const;
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
    Next finish(SlotState& st, int& row_status, int inner_status);
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
        resume_gen_p_.assign(point->gen_target_p.begin(), point->gen_target_p.end());
        resume_sto_p_.assign(point->storage_target_p.begin(), point->storage_target_p.end());
        resume_V_ = point->V;
        for (const auto& lp : point->loops) {
            if (auto* h = dynamic_cast<const ls2g::HvdcAcEmulationLimitsLoop*>(lp.get())) {
                for (const auto& line : h->regimes().lines) resume_hvdc_[line.hvdc_id] = line.frozen_dir;
            } else if (auto* m = dynamic_cast<const ls2g::VoltageMonitoringLoop*>(lp.get())) {
                for (const auto& on : m->switched_on()) resume_svc_on_.emplace_back(on.first, on.second);
            } else if (auto* r = dynamic_cast<const ls2g::ReactiveLimitsLoop*>(lp.get())) {
                for (const auto& fb : r->frozen_buses())
                    resume_frozen_.emplace(fb.bus_solver, std::make_pair(fb.state, fb.realistic));  // the first per bus
            } else if (auto* ds = dynamic_cast<const ls2g::DistributedSlackLoop*>(lp.get())) {
                resume_parked_ = ds->parked_units();   // around a distributed-slack Newton only
            }
        }
        // the tap positions the solve settled on (OuterLoopAlgo::_apply_carried_positions)
        resume_phase_tap_.assign(point->phase_tap.begin(), point->phase_tap.end());
        resume_ratio_tap_.assign(point->ratio_tap.begin(), point->ratio_tap.end());
        resume_sections_.assign(point->sections.begin(), point->sections.end());
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
    check(ov_.load_off.empty() || ov_.load_off.size() == static_cast<size_t>(n_rows) * n_load_, "disconnected loads");
    check(ov_.lost_p_mw.empty() || ov_.lost_p_mw.size() == static_cast<size_t>(n_rows), "lost active power");
    n_rows_ = n_rows;
    slots_.assign(static_cast<size_t>(n_slots), SlotState());
    results_.assign(static_cast<size_t>(n_rows), RowResult());
    row_blk_trafo_.assign(static_cast<size_t>(n_rows), {});
    row_blk_.assign(static_cast<size_t>(n_rows), {});
}

// =============================================================================
// A row enters: default control state, the loops' initialize (not resumed)
// =============================================================================
void Ls2gOuterModel::load(int slot, int row)
{
    ProfScope ps_(prof_, row >= 0 ? P_LOAD : P_LOAD_PHANTOM);
    SlotState& st = slots_[static_cast<size_t>(slot)];
    { ProfScope p(prof_, P_LOAD_RESET); st = SlotState(); }
    st.row = row;
    st.pq.assign(layout_.pin_bus.size(), 0);
    st.held_q.assign(static_cast<size_t>(layout_.n_ctrl), NaN);
    st.svc_on.assign(static_cast<size_t>(layout_.n_ctrl), NaN);
    st.hvdc_regime.assign(static_cast<size_t>(layout_.n_hvdc), 2);
    // the row as its loops see it (BaseBatchSweep::_fill_outer_row_view)
    st.rowv.layout = &sbl_;
    bool changes_units = false;
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
        if (!ov_.load_off.empty()) {
            st.rowv.loads.off.assign(ov_.load_off.begin() + static_cast<ptrdiff_t>(r * n_load_),
                                     ov_.load_off.begin() + static_cast<ptrdiff_t>((r + 1) * n_load_));
        }
        if (r < ov_.lost_p_mw.size()) st.rowv.lost_p_mw = ov_.lost_p_mw[r];
        if (r < ov_.masked.size() && !ov_.masked[r].empty()) {
            st.masked = ov_.masked[r];
            std::sort(st.masked.begin(), st.masked.end());
            st.rowv.masked = &st.masked;
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
    if (changes_units || !st.rowv.gens.target_vm.empty()) st.rl = build_rl_plan(view(st));
        else if (!st.masked.empty())                          st.rl = rl_plan_masked(st.masked);
        else                                                  st.rl = rl0_;
    }
    // BranchControl: what the slot's system holds at the grid's taps (BranchControl::_reset),
    // without the transformers the row opens (drop_out_of_row); no control requested yet
    // (OuterControls::reset_states)
    {
        ProfScope p(prof_, P_LOAD_BC);
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
    // ShuntControl: the grid's sections (ShuntControl::_reset), no control requested yet
    {
        ProfScope p(prof_, P_LOAD_SC);
        const auto& shunts = grid_->get_shunts();
        st.sc_counts.assign(static_cast<size_t>(n_sc_), {});
        for (int e = 0; e < n_sc_; ++e)
            for (int sh : sc_shunts_[static_cast<size_t>(e)])
                st.sc_counts[static_cast<size_t>(e)].push_back(shunts.get_section_count(sh));
        st.sc_pending.assign(static_cast<size_t>(n_sc_), 0);
        st.sc_b_new.assign(static_cast<size_t>(n_sc_), 0.);
        st.sc_ctl.assign(static_cast<size_t>(n_sc_), -1);
    }
    // DistributedSlackLoop's collect_units: the grid's participants, unless the row
    // disconnects units (their injection is the row's target in force: init_loops) or
    // strands buses (a unit on one takes no share)
    {
        ProfScope p(prof_, P_LOAD_UNITS);
        if (prof_.on && !st.masked.empty()) ++prof_.n[P_LOAD_MASKED_ROWS];
        if (changes_units) {
            collect_units(view(st), st.ds_units, st.ds_unit_bus);
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
    if (row >= 0) {
        if (resume_) {
            ProfScope p(prof_, P_LOAD_RESUME_TARGETS);
            // the grid's moves, unless the row sets targets or disconnects units
            if (changes_units || !st.rowv.gens.target_p.empty() || !st.rowv.storages.target_p.empty()) {
                take_resume_targets(st);
            } else {
                st.gen_target_p = resume0_gen_target_p_;
                st.storage_target_p = resume0_sto_target_p_;
                st.target_delta = resume0_target_delta_;
            }
        }
        if (resume_) {
            // OuterLoopAlgo::_apply_carried_positions: every tap control moved to the position
            // the solve resumed from left it at (a transformer the row opens takes nothing)
            for (int e = 0; e < n_bc_; ++e) {
                const size_t ee = static_cast<size_t>(e);
                if (st.bc_out[ee]) continue;
                const size_t t = static_cast<size_t>(bc_trafo_[ee]);
                const int none = std::numeric_limits<int>::min();
                if (bc_phase_[ee] && t < resume_phase_tap_.size() && resume_phase_tap_[t] != none) {
                    st.bc_moved[ee] = 1;
                    st.bc_req_tap[ee] = resume_phase_tap_[t];
                }
                if (bc_ratio_[ee] && t < resume_ratio_tap_.size() && resume_ratio_tap_[t] != none)
                    st.bc_req_rtap[ee] = resume_ratio_tap_[t];
            }
            // ... and every shunt controller all of whose shunts have a carried count
            for (int e = 0; e < n_sc_; ++e) {
                std::vector<int> counts;
                const int none = std::numeric_limits<int>::min();
                bool all = true;
                for (int sh : sc_shunts_[static_cast<size_t>(e)]) {
                    const int c = static_cast<size_t>(sh) < resume_sections_.size() ? resume_sections_[static_cast<size_t>(sh)] : none;
                    if (c == none) { all = false; break; }
                    counts.push_back(c);
                }
                if (all && !counts.empty()) sc_set_sections(st, e, counts);
            }
        }
        { ProfScope p(prof_, P_LOAD_INIT_LOOPS); init_loops(st); }
        if (resume_) { ProfScope p(prof_, P_LOAD_RESUME_LOOPS); resume_loops(st); }
        st.check_unrealistic = st.last_fixing < 0;
    }
    materialize(st);
}

void Ls2gOuterModel::take_resume_targets(SlotState& st) const
{
    // OuterLoopAlgo::_take_targets: the targets the loops moved there, but where the row sets
    // its own (on a slot whose injections nothing moved yet)
    const ls2g::OuterGridView v = view(st);
    for (size_t gid = 0; gid < resume_gen_p_.size(); ++gid) {
        const int g = static_cast<int>(gid);
        if (std::isnan(resume_gen_p_[gid]) || v.row_sets_gen_target_p(g)) continue;
        const int bus = v.gen_bus_solver(g);
        if (bus >= 0) move_gen(st, g, bus, resume_gen_p_[gid]);
    }
    for (size_t sid = 0; sid < resume_sto_p_.size(); ++sid) {
        const int s = static_cast<int>(sid);
        if (std::isnan(resume_sto_p_[sid]) || v.row_sets_storage_target_p(s)) continue;
        const int bus = v.storage_bus_solver(s);
        if (bus >= 0) move_storage(st, s, bus, resume_sto_p_[sid]);
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

void Ls2gOuterModel::collect_units(const ls2g::OuterGridView& v,
                                   std::vector<ls2g::slack_redistribution::Participant>& units,
                                   std::vector<int>& bus_of) const
{
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
            gens.get_can_participate_slack_weight(el), v.gen_target_p(el), gens.get_min_p(el), gens.get_max_p(el));
    for (int el = 0; el < stos.nb(); ++el)
        add(ls2g::slack_redistribution::UnitKind::STORAGE, el, v.storage_bus_solver(el),
            stos.get_can_participate_slack_weight(el), -v.storage_target_p(el), stos.get_min_p(el), stos.get_max_p(el));
}

std::shared_ptr<const RlPlan> Ls2gOuterModel::build_rl_plan(const ls2g::OuterGridView& v) const
{
    auto rl = std::make_shared<RlPlan>();
    // ReactiveLimitsLoop::_initialize: bus_q_check's plan on this view, plus one entry per
    // voltage monitor (a held SVC controller of the plan)
    auto plan = std::make_shared<ls2g::bus_q_check::BusQPlan>();
    rl->plan = plan;
    ls2g::bus_q_check::build_bus_q_plan(v, *plan);
    rl->n_bus_q = plan->buses.size();
    std::vector<int> monitor_of_entry(plan->buses.size(), -1);
    const ls2g::VoltageControlSolverData& ctrl = v.layout().voltage_control.controllers();
    for (int c = 0; c < ctrl.n_controllers(); ++c) {
        if (!ctrl.is_held(c) || ctrl.kind(c) != ls2g::VoltageControlSolverData::SVC) continue;
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
    // the robust mode's: the target Q of every generator ON in this row, summed per bus
    std::vector<double> target_q_of_bus(v.id_solver_to_me().size(), 0.);
    for (int gid = 0; gid < gens.nb(); ++gid) {
        const int b = v.gen_bus_solver(gid);
        if (b >= 0 && static_cast<size_t>(b) < target_q_of_bus.size())
            target_q_of_bus[static_cast<size_t>(b)] += gens.get_target_q()(gid);
    }
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
        bus.target_q = target_q_of_bus[static_cast<size_t>(entry.bus_solver)];
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
                needed = std::any_of(st.ds_unit_bus.begin(), st.ds_unit_bus.end(),
                                     [&](int b) { return b != slack_bus_; });
                break;
            case Kind::HvdcLimits:        needed = !hvdc_plan0_.empty(); break;
            case Kind::VoltageMonitoring: {
                const auto& svcs = grid_->get_svcs();
                for (int id : held_svcs_)
                    if (svcs.get_regulated_bus_id(id) == svcs.get_bus_id()(id).cast_int()) needed = true;
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
                    if (resume_) ds_keep_parked(st);
                    for (size_t k = 0; k < st.ds_units.size() && k < st.ds_lost_sat.size(); ++k) {
                        if (!st.ds_lost_sat[k]) continue;
                        st.ds_parked[k] = 1;
                        st.ds_parked_high[k] = st.ds_lost_high ? 1 : 0;
                    }
                }
                break;
            }
            case Kind::HvdcLimits:
                st.hvdc_plan = hvdc_plan0_;   // every line linear
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
    const double lost = ls2g::DistributedSlackLoop::lost_p_to_share_mw(
        view(st), resume_ ? resume_gen_p_ : none, resume_ ? resume_sto_p_ : none);
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
        u.injection_mw = new_mw[k];
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
        for (const auto& parked : resume_parked_) {
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
    RowView rv{nullptr, &widx_, &resume_V_};
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
                    auto it = resume_hvdc_.find(line.hvdc_id);
                    if (it == resume_hvdc_.end() || it->second == 0) continue;
                    const auto e = std::find(hvdc_id_.begin(), hvdc_id_.end(), line.hvdc_id);
                    if (e == hvdc_id_.end()) continue;
                    line.frozen_dir = it->second;
                    st.hvdc_regime[static_cast<size_t>(e - hvdc_id_.begin())] = it->second;
                }
                break;
            case Kind::VoltageMonitoring:
                // a switched-on SVC stays on for good
                st.vm_switched_on = resume_svc_on_;
                for (const auto& on : resume_svc_on_) {
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
                    auto it = resume_frozen_.find(bus.bus_solver);
                    if (it == resume_frozen_.end() || it->second.first == 0) continue;
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
    if (st.ds_in_newton) return check_ds_in_newton(st, L, rv);
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
                } else {
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
    if (!rs.realistic)
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
        st.tvc_frozen.push_back(TvcFrozen{static_cast<int>(k), b, local, target_vm});
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
        return UNSTABLE;
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
    // in any case, the loop must run again
    return UNSTABLE;
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
        // OpenLoadFlow re-solves whenever there is a phase shifter, limiters included
        return UNSTABLE;
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
    if (next == Next::Done) ps.relabel(P_ON_SOLVED_DONE);
    return next;
}

Next Ls2gOuterModel::on_solved_impl(int slot, bool converged, int nr_iter, int inner_status,
                                    const SlotData* data, int& row_status)
{
    SlotState& st = slots_[static_cast<size_t>(slot)];
    // OuterLoopAlgo::_solve's epilogue: a solve counts for at least one iteration
    st.total_nr += std::max(nr_iter, 1);
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
    if (st.solver_ok && st.nb_outer < params_.max_outer_iterations)
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
            return finish(st, row_status, inner_status);
        }
        const LoopDecl& L = loops_[static_cast<size_t>(st.active[static_cast<size_t>(st.i)])];
        const int status = check(st, L, rv, st.loop_iter[static_cast<size_t>(st.i)]);
        st.last_status = status;
        st.last_name = L.name;
        if (status == UNSTABLE) {
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

Next Ls2gOuterModel::finish(SlotState& st, int& row_status, int inner_status)
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
        r.nb_nr_iterations = st.total_nr;
        for (size_t s = 0; s < st.storage_target_p.size(); ++s)
            if (!std::isnan(st.storage_target_p[s])) { r.storage_ids.push_back(static_cast<int>(s)); r.storage_target_p_mw.push_back(st.storage_target_p[s]); }
        for (size_t e = 0; e < st.hvdc_regime.size(); ++e)
            if (st.hvdc_regime[e] != 2) { r.hvdc_ids.push_back(hvdc_id_[e]); r.hvdc_regime.push_back(st.hvdc_regime[e]); }
        // the section counts the loops left, where they differ from the grid's
        {
            const auto& shunts = grid_->get_shunts();
            for (int e = 0; e < n_sc_; ++e)
                for (size_t i = 0; i < sc_shunts_[static_cast<size_t>(e)].size(); ++i) {
                    const int sh = sc_shunts_[static_cast<size_t>(e)][i];
                    const int c = st.sc_counts[static_cast<size_t>(e)][i];
                    if (shunts.has_sections(sh) && shunts.get_section_count(sh) == c) continue;
                    r.shunt_ids.push_back(sh);
                    r.shunt_sections.push_back(c);
                }
        }
        // the positions the loops handled, where they differ from the grid's
        // (BaseBatchSweep::_record_row_outer), and the blocks the row's flows use
        // (_record_row_trafo_blocks)
        {
            const auto& trafos = grid_->get_trafos();
            const auto& ptc = trafos.get_tap_changers(true);
            const auto& rtc = trafos.get_tap_changers(false);
            auto& bt = row_blk_trafo_[static_cast<size_t>(st.row)];
            auto& bb = row_blk_[static_cast<size_t>(st.row)];
            bt.clear();
            bb.clear();
            for (int e = 0; e < n_bc_; ++e) {
                const size_t ee = static_cast<size_t>(e);
                const int t = bc_trafo_[ee];
                const bool ph = bc_phase_[ee] && !(ptc.has(t) && ptc.position(t) == st.bc_pos[ee]);
                const bool ra = bc_ratio_[ee] && !(rtc.has(t) && rtc.position(t) == st.bc_rpos[ee]);
                if (ph) { r.phase_tap_ids.push_back(t); r.phase_tap.push_back(st.bc_pos[ee]); }
                if (ra) { r.ratio_tap_ids.push_back(t); r.ratio_tap.push_back(st.bc_rpos[ee]); }
                if (!ph && !ra) continue;
                const int rpos = ra ? st.bc_rpos[ee] : (rtc.has(t) ? rtc.position(t) : 0);
                const int ppos = ph ? st.bc_pos[ee] : (ptc.has(t) ? ptc.position(t) : 0);
                const double shift = ptc.has(t) ? ptc.alpha_at(t, ppos) : trafos[t].shift_rad;
                const auto blk = trafos.pi_block_at(t, rpos, ppos, shift);
                bt.push_back(t);
                for (int k = 0; k < 4; ++k) { bb.push_back(blk[static_cast<size_t>(k)].real()); bb.push_back(blk[static_cast<size_t>(k)].imag()); }
            }
        }
        r.decisions = std::move(st.trace);
        adjoint_info(st, r);
        // the row is over: its slot's targets (reset at its next load) become the result's
        r.gens_target_p_mw = std::move(st.gen_target_p);
    }
    return Next::Done;
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
    // starts from (injection_mw) -- a unit that share put at its bound is held there too
    if (st.ds_in_newton)
        for (size_t k = 0; k < st.ds_units.size() && k < st.ds_parked.size() && k < st.ds_initial_mw.size(); ++k) {
            const auto& u = st.ds_units[k];
            if (!st.ds_parked[k] || u.kind != ls2g::slack_redistribution::UnitKind::GENERATOR) continue;
            if (st.ds_current_mw[k] == st.ds_initial_mw[k]) continue;
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

void Ls2gOuterModel::row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const
{
    trafo = row_blk_trafo_.at(static_cast<size_t>(row));
    block = row_blk_.at(static_cast<size_t>(row));
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

}  // namespace outer
