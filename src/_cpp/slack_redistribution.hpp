// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// slack_redistribution.hpp — OpenLoadFlow-style bounded redistribution of the
// active power a batch row loses (option `redistribute_slack`, lightsim2grid
// PR #216 parity: SlackRedistribution.hpp + BaseBatchSweep::
// _prepare_slack_redistribution)
// =============================================================================
//
// The distributed slack of the Newton solve shares whatever imbalance a row
// leaves by fixed per-bus weights, with no limit: a row that loses a big
// generator (a ScenarioSweep generator contingency, or an island cut off in
// handle_disconnected_grid mode) pushes the remaining machines past their
// max_p, where OpenLoadFlow's DistributedSlack outer loop stops each one at its
// bound and re-shares the excess. The part of that imbalance known BEFORE the
// solve -- the set-points of what the row takes out -- is shared here the way
// OLF does it (distribute), once per run() for every row, on the host:
//
//   * what a row loses (lost_mw, generator convention) is
//       (a) the active set-point of every generator the row disconnects, plus
//       (b) the net injection of the buses it masks: sn * sum Re(Sbus_row)
//           over them (generators the row keeps, static generators, minus
//           loads and storage units, the non-droop hvdc stations -- all in the
//           row's Sbus) minus their shunts' active power at 1 pu, which the
//           AC Sbus does not carry. Upstream sums the same terms element by
//           element; the total is the same up to rounding. (A droop hvdc end
//           is never masked here: stranding one skips the row.)
//   * the participants are the slack units (generators, then storage units,
//     by id) left in the main component and not disconnected by the row, with
//     a weight above 1e-7, in GENERATOR convention;
//   * distribute() shares lost_mw on them, clamped to [min_p, max_p] and never
//     crossing 0 MW; the row then gets, per bus, dP = new - old set-point
//     (added to its Sbus) and its saturated units leave its distributed slack
//     (row weights re-derived without them), so the Newton solve only shares
//     what is left (the change in the losses) on the units that can still move.
//
// Nothing the size of n_rows x n_bus is built; a row that loses nothing keeps
// every entry empty and solves exactly as without the option.
//
// CUDA-free on purpose: included by the session headers, which the host
// compiler builds into ls2g_bridge.cpp.
// =============================================================================

#ifndef SLACK_REDISTRIBUTION_HPP
#define SLACK_REDISTRIBUTION_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "dtypes.hpp"

// Element codes of the units (ViolationElementType's, like GenPPlanData).
constexpr int SLACK_UNIT_GENERATOR = 5;
constexpr int SLACK_UNIT_STORAGE   = 6;

// Everything the pre-pass needs, read off a solved grid by the bridge
// (extract_slack_redistribution_data) or handed in as arrays in tuple mode.
struct SlackRedistributionData {
    // ---- the distributed-slack participants, generators by id then storage
    // units by id (upstream order): connected, flagged slack, |weight| > 1e-12,
    // on a bus of the solved system
    int             n_units = 0;
    Eigen::VectorXi kind;          // [n_units] 5 GENERATOR / 6 STORAGE
    Eigen::VectorXi el_id;         // [n_units] container id
    Eigen::VectorXi bus_solver;    // [n_units]
    RealVect        weight;        // [n_units] raw participation factor
    RealVect        min_p_mw;      // [n_units] NaN = unbounded below
    RealVect        max_p_mw;      // [n_units] NaN = unbounded above
    RealVect        target_p_mw;   // [n_units] grid's set-point, GENERATOR convention
    // [n_units] 1: a participant of the Newton solve's distributed slack too (a saturated
    // one leaves it); 0: of this pre-pass only -- a unit lightsim2grid flags "can
    // participate in the slack" (LSGrid::set_gen_can_participate_slack: left out of the
    // slack only because it sat at an active limit), which never enters the solve's
    // slack weights. Empty: every unit is in the slack.
    Eigen::VectorXi in_slack;
    // ---- every generator: whether it is in the solved system (its solver bus,
    // -1 otherwise) and its grid set-point -- what a row that disconnects it
    // loses when the row gives no set-point of its own
    int             n_gen = 0;
    Eigen::VectorXi gen_bus_solver;   // [n_gen]
    RealVect        gen_target_p_mw;  // [n_gen]
    int             n_sto = 0;        // storage container size (masks are sized on it)
    // ---- per solver bus, the active power of its shunts at 1 pu (MW, load
    // convention): not in the AC Sbus, lost with the bus all the same
    RealVect        shunt_p_mw;       // [n_bus]
    double          sn_mva = 100.0;

    bool empty() const { return n_units == 0; }
    bool unit_in_slack(int k) const { return in_slack.size() == 0 || in_slack(k) != 0; }

    void validate(int n_bus) const {
        auto fail = [&](const std::string& what) {
            throw std::runtime_error("SlackRedistributionData: " + what);
        };
        auto need = [&](long long got, long long want, const char* name) {
            if (got != want) {
                std::ostringstream m;
                m << name << " has size " << got << ", expected " << want;
                fail(m.str());
            }
        };
        if (n_units < 0 || n_gen < 0 || n_sto < 0) fail("negative sizes");
        need(kind.size(),        n_units, "kind");
        need(el_id.size(),       n_units, "el_id");
        need(bus_solver.size(),  n_units, "bus_solver");
        need(weight.size(),      n_units, "weight");
        need(min_p_mw.size(),    n_units, "min_p_mw");
        need(max_p_mw.size(),    n_units, "max_p_mw");
        need(target_p_mw.size(), n_units, "target_p_mw");
        if (in_slack.size() != 0) need(in_slack.size(), n_units, "in_slack");
        need(gen_bus_solver.size(),  n_gen, "gen_bus_solver");
        need(gen_target_p_mw.size(), n_gen, "gen_target_p_mw");
        need(shunt_p_mw.size(),  n_bus, "shunt_p_mw");
        for (int k = 0; k < n_units; ++k) {
            std::ostringstream m;
            const int t = kind(k);
            if (t != SLACK_UNIT_GENERATOR && t != SLACK_UNIT_STORAGE) {
                m << "kind[" << k << "] must be 5 (GENERATOR) or 6 (STORAGE), got " << t;
                fail(m.str());
            }
            const int lim = (t == SLACK_UNIT_GENERATOR) ? n_gen : n_sto;
            if (el_id(k) < 0 || el_id(k) >= lim) {
                m << "el_id[" << k << "] = " << el_id(k) << " is outside its container";
                fail(m.str());
            }
            if (bus_solver(k) < 0 || bus_solver(k) >= n_bus) {
                m << "bus_solver[" << k << "] is outside [0, n_bus=" << n_bus << ")";
                fail(m.str());
            }
            if (!std::isfinite(static_cast<double>(weight(k))) ||
                !std::isfinite(static_cast<double>(target_p_mw(k)))) {
                m << "unit " << k << ": weight and target_p_mw must be finite";
                fail(m.str());
            }
        }
        for (int g = 0; g < n_gen; ++g)
            if (gen_bus_solver(g) >= n_bus) fail("gen_bus_solver outside [-1, n_bus)");
        if (!(sn_mva > 0.)) fail("sn_mva must be > 0");
    }
};

namespace slack_redistribution {

// A unit taking part in the redistribution (generator convention, MW).
struct Participant {
    int    unit;       // index into SlackRedistributionData's units
    double injection_mw;
    double weight;
    double min_p_mw;   // NaN: unbounded
    double max_p_mw;   // NaN: unbounded
};

// What the redistribution did, per row (lightsim2grid's SlackRedistributionReport).
struct Report {
    double mismatch_mw = 0.;         // what had to be shared (> 0: units inject more)
    int    nb_participants = 0;
    int    nb_saturated = 0;         // units that reached a bound (and left the slack)
    int    nb_rounds = 0;            // 0: nothing was shared
    double not_distributed_mw = 0.;  // what no unit could take (all saturated)
    bool   all_saturated = false;    // every unit hit its bound: none left the slack
};

constexpr double DEFAULT_EPS_MW   = 1e-6;   // upstream default_eps_mw
constexpr double POOL_MIN_WEIGHT  = 1e-7;   // upstream BaseConstants::_tol_equal_float
constexpr double MOVED_EPS_MW     = 1e-7;   // a set-point "moved" (upstream _tol_equal_float)

// Verbatim port of lightsim2grid's slack_redistribution::distribute (OLF's
// GenerationActivePowerDistributionStep): share `mismatch_mw` on `units`, in
// their order, proportionally to their weight, each clamped to its bounds and
// never crossing 0 MW; a clamped unit leaves the pool and what it could not
// take is shared again. When EVERY unit hits its bound, `saturated` is cleared
// and all_saturated set (they all stay in the slack).
inline Report distribute(const std::vector<Participant>& units, double mismatch_mw, double eps_mw,
                         std::vector<double>& new_injection_mw, std::vector<char>& saturated)
{
    const std::size_t nb = units.size();
    Report report;
    report.mismatch_mw = mismatch_mw;
    report.nb_participants = static_cast<int>(nb);
    new_injection_mw.resize(nb);
    saturated.assign(nb, 0);
    for (std::size_t k = 0; k < nb; ++k) new_injection_mw[k] = units[k].injection_mw;
    if (nb == 0 || std::abs(mismatch_mw) <= eps_mw) return report;

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> lo(nb), hi(nb);
    std::vector<char> active(nb, 1);
    for (std::size_t k = 0; k < nb; ++k) {
        lo[k] = std::isfinite(units[k].min_p_mw) ? units[k].min_p_mw : -inf;
        hi[k] = std::isfinite(units[k].max_p_mw) ? units[k].max_p_mw : inf;
        // the sign of the injection is kept: 0 MW is a bound on the side the unit is not on
        if (units[k].injection_mw < 0.) hi[k] = std::min(hi[k], 0.);
        else lo[k] = std::max(lo[k], 0.);
    }

    double remaining = mismatch_mw;
    std::size_t nb_active = nb;
    while (nb_active > 0 && std::abs(remaining) > eps_mw && report.nb_rounds <= static_cast<int>(nb) + 1) {
        ++report.nb_rounds;
        double factor_sum = 0.;
        for (std::size_t k = 0; k < nb; ++k) if (active[k]) factor_sum += units[k].weight;
        if (factor_sum <= 0.) break;
        double done = 0.;
        for (std::size_t k = 0; k < nb; ++k) {
            if (!active[k]) continue;
            const double old = new_injection_mw[k];
            double cand = old + remaining * units[k].weight / factor_sum;
            if (remaining > 0. && cand >= hi[k]) {
                // never DEcrease a unit already above its max (same rule for the min)
                cand = old > hi[k] ? old : hi[k];
                active[k] = 0;
                saturated[k] = 1;
                --nb_active;
            } else if (remaining < 0. && cand <= lo[k]) {
                cand = old < lo[k] ? old : lo[k];
                active[k] = 0;
                saturated[k] = 1;
                --nb_active;
            }
            done += cand - old;
            new_injection_mw[k] = cand;
        }
        remaining -= done;
    }
    report.not_distributed_mw = remaining;
    for (std::size_t k = 0; k < nb; ++k) if (saturated[k]) ++report.nb_saturated;
    if (nb_active == 0) {
        report.all_saturated = true;
        saturated.assign(nb, 0);
    }
    return report;
}

// What the pre-pass decided for ONE row.
struct RowResult {
    std::vector<std::pair<int, double>> dp_pu;   // (solver bus, dP in pu), one per bus, sorted
    std::vector<std::pair<int, double>> moved;   // (unit index, new set-point MW, gen. convention), sorted
    std::vector<int>                    saturated_units;   // unit indices, sorted
    Report                              report;
    bool empty() const { return dp_pu.empty() && saturated_units.empty(); }
};

// The active power (MW, generator convention) a row loses: (a) the set-point of
// every generator in the solved system the row disconnects (gen_off(g),
// gen_p_row(g) its row set-point), plus (b) the net injection of its masked
// buses (`masked`, solver ids) read off its own Sbus (sbus_p_mw(b), MW) minus
// their shunts' active power.
template <class GenOff, class GenPRow, class SbusPMw>
double lost_mw(const SlackRedistributionData& d, const std::vector<int>& masked,
               GenOff gen_off, GenPRow gen_p_row, SbusPMw sbus_p_mw)
{
    double lost = 0.;
    for (int g = 0; g < d.n_gen; ++g) {
        if (d.gen_bus_solver(g) < 0) continue;
        if (!gen_off(g)) continue;
        lost += gen_p_row(g);
    }
    for (int b : masked)
        lost += sbus_p_mw(b) - static_cast<double>(d.shunt_p_mw(b));
    return lost;
}

// The pre-pass of one row: share `lost` on the units that stay in the main
// component (bus_masked(b) false) and that the row does not disconnect
// (gen_off(g) false for a generator), at their row set-point (gen_p_row(g) for
// a generator, the grid's own for a storage unit), and turn the result into
// the row's Sbus correction and saturated units.
template <class BusMasked, class GenOff, class GenPRow>
RowResult redistribute_row(const SlackRedistributionData& d, double lost,
                           BusMasked bus_masked, GenOff gen_off, GenPRow gen_p_row,
                           double eps_mw = DEFAULT_EPS_MW)
{
    RowResult out;
    out.report.mismatch_mw = lost;
    if (std::abs(lost) <= eps_mw) return out;

    std::vector<Participant> units;
    units.reserve(static_cast<size_t>(d.n_units));
    for (int k = 0; k < d.n_units; ++k) {
        if (!(static_cast<double>(d.weight(k)) > POOL_MIN_WEIGHT)) continue;
        if (bus_masked(d.bus_solver(k))) continue;
        const bool is_gen = d.kind(k) == SLACK_UNIT_GENERATOR;
        if (is_gen && gen_off(d.el_id(k))) continue;
        Participant p;
        p.unit         = k;
        p.injection_mw = is_gen ? gen_p_row(d.el_id(k)) : static_cast<double>(d.target_p_mw(k));
        p.weight       = static_cast<double>(d.weight(k));
        p.min_p_mw     = static_cast<double>(d.min_p_mw(k));
        p.max_p_mw     = static_cast<double>(d.max_p_mw(k));
        units.push_back(p);
    }
    if (units.empty()) return out;   // the row's own fallback stays (see row_slack_weights)

    std::vector<double> new_inj;
    std::vector<char>   saturated;
    out.report = distribute(units, lost, eps_mw, new_inj, saturated);
    for (size_t i = 0; i < units.size(); ++i) {
        const Participant& p = units[i];
        const double dp_mw = new_inj[i] - p.injection_mw;
        if (std::abs(dp_mw) > MOVED_EPS_MW) {
            out.moved.emplace_back(p.unit, new_inj[i]);
            out.dp_pu.emplace_back(d.bus_solver(p.unit), dp_mw / d.sn_mva);
        }
        // a pre-pass-only unit was never in the solve's slack: nothing to leave
        if (saturated[i] && d.unit_in_slack(p.unit)) out.saturated_units.push_back(p.unit);
    }
    // one entry per bus: merge the units sharing one
    std::sort(out.dp_pu.begin(), out.dp_pu.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.first < b.first; });
    std::vector<std::pair<int, double>> merged;
    merged.reserve(out.dp_pu.size());
    for (const auto& bd : out.dp_pu) {
        if (!merged.empty() && merged.back().first == bd.first) merged.back().second += bd.second;
        else merged.push_back(bd);
    }
    out.dp_pu.swap(merged);
    return out;
}

// A row's normalised distributed-slack weights on the base participant layout
// (`slack_bus`, the bus of each slack index), without the units `excluded`
// (unit indices: the generators the row disconnects and the units its
// pre-pass saturated) -- lightsim2grid's get_slack_weights_solver_without. When
// nothing is left, the reference bus takes the whole share. Throws when a
// survivor stands on a bus with no slack index (it cannot: the survivors are a
// subset of the base participants) or the reference has none.
inline void row_slack_weights(const SlackRedistributionData& d, const std::vector<char>& excluded,
                              int n_bus, const std::vector<int>& slack_bus, int ref_bus,
                              int row_for_errors, std::vector<double>& w_scratch,
                              cuda_real_type* out_row)
{
    w_scratch.assign(static_cast<size_t>(n_bus), 0.0);
    double sum = 0.0;
    for (int k = 0; k < d.n_units; ++k) {
        if (excluded[static_cast<size_t>(k)]) continue;
        if (!d.unit_in_slack(k)) continue;   // a pre-pass-only unit: not in the solve's slack
        w_scratch[static_cast<size_t>(d.bus_solver(k))] += static_cast<double>(d.weight(k));
        sum += static_cast<double>(d.weight(k));
    }
    if (std::abs(sum) < 1e-12) {
        std::fill(w_scratch.begin(), w_scratch.end(), 0.0);
        if (ref_bus >= 0 && ref_bus < n_bus) w_scratch[static_cast<size_t>(ref_bus)] = 1.0;
    } else {
        for (double& x : w_scratch) x /= sum;
    }
    const int n_slack = static_cast<int>(slack_bus.size());
    for (int k = 0; k < n_slack; ++k) {
        const int b = slack_bus[static_cast<size_t>(k)];
        out_row[k] = static_cast<cuda_real_type>(w_scratch[static_cast<size_t>(b)]);
        w_scratch[static_cast<size_t>(b)] = 0.0;   // consumed
    }
    for (int b = 0; b < n_bus; ++b)
        if (w_scratch[static_cast<size_t>(b)] != 0.0)
            throw std::runtime_error(
                "slack redistribution: row " + std::to_string(row_for_errors) + " re-weights "
                "the distributed slack onto bus " + std::to_string(b) + ", which owns no "
                "slack column entry in the base case.");
}

// The pre-pass of every row (ORIGINAL order): skipped(r) rows are left empty;
// masked(r) the row's masked solver buses; gen_off(r, g) whether it
// disconnects generator g; gen_p_row(r, g) its set-point of generator g (MW);
// sbus_p_mw(r, b) the real part of its Sbus at bus b, in MW.
template <class Skipped, class Masked, class GenOff, class GenPRow, class SbusPMw>
std::vector<RowResult> prepass_all(const SlackRedistributionData& d, int n_rows, int n_bus,
                                   Skipped skipped, Masked masked, GenOff gen_off,
                                   GenPRow gen_p_row, SbusPMw sbus_p_mw)
{
    std::vector<RowResult> out(static_cast<size_t>(std::max(n_rows, 0)));
    std::vector<char> is_masked(static_cast<size_t>(n_bus), 0);
    for (int r = 0; r < n_rows; ++r) {
        if (skipped(r)) continue;
        const std::vector<int>& m = masked(r);
        auto g_off = [&](int g) { return gen_off(r, g); };
        auto g_p   = [&](int g) { return gen_p_row(r, g); };
        const double lost = lost_mw(d, m, g_off, g_p, [&](int b) { return sbus_p_mw(r, b); });
        for (int b : m) is_masked[static_cast<size_t>(b)] = 1;
        out[static_cast<size_t>(r)] = redistribute_row(
            d, lost, [&](int b) { return b >= 0 && b < n_bus && is_masked[static_cast<size_t>(b)] != 0; },
            g_off, g_p);
        for (int b : m) is_masked[static_cast<size_t>(b)] = 0;
    }
    return out;
}

// Per-row normalised slack weights ([n_rows * n_slack], ORIGINAL order) without
// the generators a row disconnects (gen_off(r, g)) and the units its pre-pass
// saturated (rows[r].saturated_units; `rows` may be empty). Empty when no row
// takes anything out: every slot then keeps the base weights.
template <class GenOff>
std::vector<cuda_real_type> build_row_weights(const SlackRedistributionData& d,
                                              const std::vector<RowResult>& rows,
                                              int n_rows, GenOff gen_off, int n_bus,
                                              const std::vector<int>& slack_bus,
                                              const std::vector<cuda_real_type>& base_w,
                                              int ref_bus)
{
    std::vector<cuda_real_type> out;
    const int n_slack = static_cast<int>(slack_bus.size());
    if (n_slack <= 0 || n_rows <= 0) return out;
    std::vector<char> excluded(static_cast<size_t>(d.n_units), 0);
    std::vector<double> w_scratch;
    bool any = false;
    for (int r = 0; r < n_rows; ++r) {
        std::fill(excluded.begin(), excluded.end(), 0);
        bool row_any = false;
        for (int k = 0; k < d.n_units; ++k)
            if (d.unit_in_slack(k) && d.kind(k) == SLACK_UNIT_GENERATOR && gen_off(r, d.el_id(k))) {
                excluded[static_cast<size_t>(k)] = 1;
                row_any = true;
            }
        if (static_cast<size_t>(r) < rows.size())
            for (int k : rows[static_cast<size_t>(r)].saturated_units) {
                if (!d.unit_in_slack(k)) continue;   // e.g. handed in by set_external_slack_saturation
                excluded[static_cast<size_t>(k)] = 1;
                row_any = true;
            }
        if (!row_any) continue;
        if (!any) {
            // first row taking something out: every row so far keeps the base
            out.resize(static_cast<size_t>(n_rows) * n_slack);
            for (int q = 0; q < n_rows; ++q)
                std::copy(base_w.begin(), base_w.end(), out.begin() + static_cast<ptrdiff_t>(q) * n_slack);
            any = true;
        }
        row_slack_weights(d, excluded, n_bus, slack_bus, ref_bus, r, w_scratch,
                          out.data() + static_cast<ptrdiff_t>(r) * n_slack);
    }
    return out;
}

}  // namespace slack_redistribution

// The active-power check's view of the pre-pass (upstream's _row_target_p /
// _row_takes_no_share): per row, the set-points it moved -- written over the
// caller's own targets (`user_targets`, (n_rows x n_entries), NaN = base; may
// be empty) into `targets_out` for the plan's entries -- and the units it
// saturated, which produce that set-point and take no share (`ns_gen` /
// `ns_sto`, uint8 (n_rows x n_gen) / (n_rows x n_sto), cleared when none).
// Returns false (outputs cleared) when the pre-pass moved nothing anywhere.
inline bool gen_p_check_inputs(const SlackRedistributionData& d,
                               const std::vector<slack_redistribution::RowResult>& rows,
                               const Eigen::VectorXi& plan_el_type, const Eigen::VectorXi& plan_el_id,
                               const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic,
                                                   Eigen::RowMajor>& user_targets,
                               Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic,
                                             Eigen::RowMajor>& targets_out,
                               std::vector<unsigned char>& ns_gen,
                               std::vector<unsigned char>& ns_sto)
{
    targets_out.resize(0, 0);
    ns_gen.clear();
    ns_sto.clear();
    const int n_rows = static_cast<int>(rows.size());
    bool any_moved = false, any_sat = false;
    for (const auto& r : rows) {
        if (!r.moved.empty()) any_moved = true;
        if (!r.saturated_units.empty()) any_sat = true;
    }
    if (!any_moved && !any_sat) return false;

    const Eigen::Index n_entries = plan_el_type.size();
    if (any_moved && n_entries > 0) {
        std::vector<int> gen_entry(static_cast<size_t>(d.n_gen), -1), sto_entry(static_cast<size_t>(d.n_sto), -1);
        for (Eigen::Index k = 0; k < n_entries; ++k) {
            const int id = plan_el_id(k);
            if (plan_el_type(k) == SLACK_UNIT_GENERATOR && id >= 0 && id < d.n_gen) gen_entry[static_cast<size_t>(id)] = static_cast<int>(k);
            if (plan_el_type(k) == SLACK_UNIT_STORAGE && id >= 0 && id < d.n_sto) sto_entry[static_cast<size_t>(id)] = static_cast<int>(k);
        }
        if (user_targets.rows() == n_rows && user_targets.cols() == n_entries)
            targets_out = user_targets;
        else
            targets_out = Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Constant(
                n_rows, n_entries, std::numeric_limits<eigen_real_type>::quiet_NaN());
        for (int r = 0; r < n_rows; ++r)
            for (const auto& um : rows[static_cast<size_t>(r)].moved) {
                const int k = um.first;
                const int id = d.el_id(k);
                const int e = (d.kind(k) == SLACK_UNIT_GENERATOR) ? gen_entry[static_cast<size_t>(id)]
                                                                  : sto_entry[static_cast<size_t>(id)];
                if (e >= 0) targets_out(r, e) = static_cast<eigen_real_type>(um.second);
            }
    }
    if (any_sat) {
        ns_gen.assign(static_cast<size_t>(n_rows) * d.n_gen, 0);
        ns_sto.assign(static_cast<size_t>(n_rows) * d.n_sto, 0);
        for (int r = 0; r < n_rows; ++r)
            for (int k : rows[static_cast<size_t>(r)].saturated_units) {
                const int id = d.el_id(k);
                if (d.kind(k) == SLACK_UNIT_GENERATOR)
                    ns_gen[static_cast<size_t>(r) * d.n_gen + id] = 1;
                else
                    ns_sto[static_cast<size_t>(r) * d.n_sto + id] = 1;
            }
    }
    return true;
}

// What the pre-pass did on each row of the last run() (ORIGINAL order), the
// batch counterpart of lightsim2grid's SlackRedistributionReport. A row that
// was skipped or lost nothing has mismatch_mw == 0 and nb_rounds == 0.
struct SlackRedistributionReport {
    RealVect        mismatch_mw;          // what had to be shared (> 0: units inject more)
    RealVect        not_distributed_mw;   // what no unit could take
    Eigen::VectorXi nb_participants;
    Eigen::VectorXi nb_saturated;         // units that reached a bound (left the slack)
    Eigen::VectorXi nb_rounds;
    Eigen::VectorXi all_saturated;        // 1: every unit hit its bound, none left the slack
};

inline SlackRedistributionReport make_slack_redistribution_report(
    const std::vector<slack_redistribution::RowResult>& rows)
{
    SlackRedistributionReport r;
    const Eigen::Index n = static_cast<Eigen::Index>(rows.size());
    r.mismatch_mw = RealVect::Zero(n);
    r.not_distributed_mw = RealVect::Zero(n);
    r.nb_participants = Eigen::VectorXi::Zero(n);
    r.nb_saturated = Eigen::VectorXi::Zero(n);
    r.nb_rounds = Eigen::VectorXi::Zero(n);
    r.all_saturated = Eigen::VectorXi::Zero(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const slack_redistribution::Report& rep = rows[static_cast<size_t>(i)].report;
        r.mismatch_mw(i)        = static_cast<eigen_real_type>(rep.mismatch_mw);
        r.not_distributed_mw(i) = static_cast<eigen_real_type>(rep.not_distributed_mw);
        r.nb_participants(i)    = rep.nb_participants;
        r.nb_saturated(i)       = rep.nb_saturated;
        r.nb_rounds(i)          = rep.nb_rounds;
        r.all_saturated(i)      = rep.all_saturated ? 1 : 0;
    }
    return r;
}

#endif  // SLACK_REDISTRIBUTION_HPP
