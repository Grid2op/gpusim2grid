// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/outer_hook.hpp -- what the continuous scheduler and an outer-loop model
// exchange (docs/dev_notes/continuous_batching.md, Part B).
//
// The model (outer/ls2g_outer_model.cpp, built by the lightsim2grid bridge)
// runs lightsim2grid's NROuter_* driver and loops, row by row, on the host; the
// device runs the inner Newton solves. A row stays in its slot from its first
// load to its end: when its loops act, the model rewrites the slot's CONTROL
// STATE (SlotControls) and the driver re-solves the row in place, from the V
// it converged to.
//
// Pure host types, no CUDA, no lightsim2grid: included by the driver, the
// session and the bridge alike.
// =============================================================================

#ifndef GPUSIM2GRID_OUTER_HOOK_HPP
#define GPUSIM2GRID_OUTER_HOOK_HPP

#include <string>
#include <vector>

namespace outer {

// The device layout of the controls, fixed for the life of a model (solver
// bus numbering, the ledger's order).
struct Layout {
    // The switchable buses: a Vm column and a Q row the outer loops reserved, the
    // Q row identity-pinned while the bus is PV. Sorted.
    std::vector<int> pin_bus;
    int n_ctrl = 0;   // VoltageControl controllers (ledger order)
    int n_grp  = 0;   // VoltageControl groups
    int n_hvdc = 0;   // droop hvdc entries (ledger order)
    int n_bc   = 0;   // BranchControl's transformers (LedgerData::bc_*, its order)
    int n_sc   = 0;   // ShuntControl's controllers (LedgerData::sc_*, its order)
    // The buses whose V, computed power V.conj(Ybus.V) and scheduled power (the
    // slot's Sbus) are copied back for each converged slot: everything the loops read.
    std::vector<int> watch_bus;
    // The unrealistic-voltage check (OuterLoopDriverParams): the buses whose
    // magnitude is a Newton unknown and whose nominal voltage is at least the
    // check's threshold. A switchable bus is a candidate only while PQ in its slot.
    std::vector<int> realistic_bus;
    double min_realistic_v = 0.5;
    double max_realistic_v = 2.;
};

// The control state of one slot: what its loops changed in its equations,
// apart from its inputs. Written to the device whenever it changes.
struct SlotControls {
    std::vector<unsigned char> pin;        // [pin_bus] 1: PV (Q row pinned)
    std::vector<double>        held_q;     // [n_ctrl] pu, NaN: not held by value
    std::vector<unsigned char> plan_held;  // [n_ctrl] the plan's held controllers still held
    std::vector<double>        v_set;      // [n_grp] pu
    std::vector<int>           hvdc_status;// [n_hvdc] droop regime: 0 linear, +-1 saturated
    // Sparse correction of the row's own Sbus (pu): moved targets, frozen Q.
    std::vector<int>           sbus_bus;
    std::vector<double>        sbus_dp, sbus_dq;
    // One-shot |V| resets (pu), applied at the next (re)load of the slot.
    std::vector<int>           vm_reset_bus;
    std::vector<double>        vm_reset_vm;
    // BranchControl, per transformer [n_bc] (lightsim2grid's NROuterInner::apply_controls,
    // replayed on the slot before each solve, in its order):
    //  - bc_out: 1 = the row does not have it (its contingency opened it): nothing of it is
    //    stamped in the slot's Ybus, it is never patched (BranchControl::drop_out_of_row);
    //  - a re-tap (one-shot, bc_retap lists the transformers): its tap block bc_tap
    //    [4 complex as re, im: y11 y12 y21 y22], shift bc_a_tap and ratio bc_rho_tap; the
    //    shift and ratio unknowns (and their targets) go back to the tap's;
    //  - then the requested controls, bc_ctl (phase: P = target) and bc_rctl (ratio: the
    //    group's voltage), -1 untouched, 0 off (its target: where the unknown is now), 1 on.
    std::vector<unsigned char> bc_out;
    std::vector<int>           bc_retap;
    std::vector<double>        bc_tap;          // [8 * n_bc]
    std::vector<double>        bc_a_tap, bc_rho_tap;
    std::vector<signed char>   bc_ctl, bc_rctl;
    // ShuntControl, per controller [n_sc]: the sections the loop switched (one-shot, sc_set
    // lists the controllers; their susceptance sc_b_new, pu), then the requested control
    // (sc_ctl, -1 untouched, 0 off -- its target where B is --, 1 on)
    std::vector<int>           sc_set;
    std::vector<double>        sc_b_new;
    std::vector<signed char>   sc_ctl;
};

// What the device gathered for one converged slot, as host doubles, indexed
// like Layout::watch_bus (vc_q like the controllers).
struct SlotData {
    const double* v_re  = nullptr;
    const double* v_im  = nullptr;
    const double* s_re  = nullptr;   // computed power, V.conj(Ybus.V)
    const double* s_im  = nullptr;
    const double* sb_re = nullptr;   // the slot's scheduled power (its Sbus)
    const double* sb_im = nullptr;
    const double* vc_q  = nullptr;   // controllers' reactive injection, pu
    const double* bc_a  = nullptr;   // BranchControl's shift and ratio unknowns [n_bc]
    const double* bc_rho = nullptr;
    const double* sc_b  = nullptr;   // ShuntControl's susceptances [n_sc]
    bool unrealistic = false;        // some Layout::realistic_bus outside the band
};

enum class Next : int { Resolve = 0, Done = 1 };

// What the rows change beyond their injections, as the loops see it (lightsim2grid's
// OuterRowView); every member optional, empty = no row changes it.
struct RowOverrides {
    // per row, the solver buses handle_disconnected_grid masks
    std::vector<std::vector<int>> masked;
    // n_rows x n_gen, row-major: 1 = the row disconnects the generator (set_contingency_gens)
    std::vector<char> gen_off;
    // n_rows x n_gen, row-major: the generators' voltage set-points, pu (NaN: the grid's)
    std::vector<double> gen_vm;
    // per row, the switchable buses it starts PQ (a generator contingency took their last
    // regulator)
    std::vector<std::vector<int>> pq_buses;
    // n_rows x n_grp, row-major: the VoltageControl groups' v_set the row's gen_v gives
    // (NaN: the grid's)
    std::vector<double> vc_v_set;
    // per row, the branches its contingency opens (the lines, then the transformers:
    // n_lines + trafo id)
    std::vector<std::vector<int>> branch_off;
};

// Mirrors lightsim2grid's OuterRowResult (BaseBatchSweep.hpp).
struct RowResult {
    int  status = -1;            // OuterLoopStatus: 0 STABLE, 1 UNSTABLE, 2 FAILED; -1 never ran
    bool unrealistic = false;
    std::string failed_loop;     // the loop's name when status == FAILED
    int  nb_outer_iterations = 0;
    int  nb_nr_iterations = 0;   // summed over every inner solve, at least 1 per solve
    // what the loops moved, sparse: only what differs from the row's own values
    std::vector<int>    gen_ids;
    std::vector<double> gen_target_p_mw;
    std::vector<int>    storage_ids;
    std::vector<double> storage_target_p_mw;   // load convention, as the container
    std::vector<int>    hvdc_ids;
    std::vector<int>    hvdc_regime;
    // the tap positions and section counts they settled on, where they differ from the grid's
    std::vector<int>    phase_tap_ids, phase_tap;
    std::vector<int>    ratio_tap_ids, ratio_tap;
    std::vector<int>    shunt_ids, shunt_sections;
    // the decision trace (OuterLoopStats::decisions), when the model keeps it
    std::vector<std::string> decisions;
};

class OuterHook {
public:
    virtual ~OuterHook() = default;

    virtual const Layout& layout() const = 0;
    // A new run of `n_slots` slots over rows [0, n_rows) (original numbering).
    virtual void begin(int n_slots, int n_rows) = 0;
    // A row (original index, -1 for a phantom slot) enters slot `slot`: its loops
    // start from the default control state.
    virtual void load(int slot, int row) = 0;
    virtual const SlotControls& controls(int slot) const = 0;
    // The row in slot `from` now lives in slot `to` (the tail compaction of a
    // continuous run moves the live rows into the first slots): everything the
    // model keeps for it follows; `from` is left as an empty phantom slot.
    virtual void move(int from, int to) = 0;
    // The driver applied the slot's one-shot edits (|V| resets, re-taps).
    virtual void clear_vm_reset(int slot) = 0;
    // The pi block of every transformer the row's loops moved off the grid's taps, at its
    // final taps (lightsim2grid's _record_row_trafo_blocks): what its flows are computed
    // with. `trafo` grid ids; `block` 8 doubles per transformer (re, im of y11 y12 y21 y22).
    virtual void row_trafo_blocks(int row, std::vector<int>& trafo, std::vector<double>& block) const = 0;
    // The inner solve of the row in `slot` ended after `nr_iter` iterations, converged
    // (`data` holds what the device gathered) or not (`inner_status`, a RowStatus).
    // Resolve: the row stays, controls(slot) is its new state. Done: `row_status`
    // receives its final RowStatus.
    virtual Next on_solved(int slot, bool converged, int nr_iter, int inner_status,
                           const SlotData* data, int& row_status) = 0;
    virtual const RowResult& row_result(int row) const = 0;
    virtual int n_rows() const = 0;
    // The rows' own active targets, MW, row-major (NaN or empty: the grid's): the
    // generators', and the storage units' in the load convention -- what the
    // loops start from (DistributedSlack's units, ReactiveLimits' capability
    // curves). Their effect on Sbus is the caller's (the row's injections).
    virtual void set_row_targets(int n_rows, std::vector<double> gen_p, std::vector<double> storage_p) = 0;
    // The rows' other changes (see RowOverrides); replaces the previous ones.
    virtual void set_row_overrides(RowOverrides overrides) = 0;
};

}  // namespace outer

#endif  // GPUSIM2GRID_OUTER_HOOK_HPP
