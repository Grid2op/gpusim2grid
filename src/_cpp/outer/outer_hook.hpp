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
    bool unrealistic = false;        // some Layout::realistic_bus outside the band
};

enum class Next : int { Resolve = 0, Done = 1 };

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
    // The driver applied the slot's |V| resets.
    virtual void clear_vm_reset(int slot) = 0;
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
};

}  // namespace outer

#endif  // GPUSIM2GRID_OUTER_HOOK_HPP
