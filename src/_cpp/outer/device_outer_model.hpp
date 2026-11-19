// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// outer/device_outer_model.hpp -- the outer loops' decisions on the device
// (outer_loops_backend='device'; docs/outer_loops_on_device.md, §4.6).
//
// An outer::OuterHook like Ls2gOuterModel, whose per-slot state lives in device
// arrays and whose decisions (the passes, the four default loops' checks, the
// controls) run in kernels, one thread per slot, following the host model line
// by line. A row is loaded by the host model it wraps (its own load, then the
// slot state uploaded); the tables come from that model's plans. CUDA-free
// header: the class is defined in the .cu.
// =============================================================================

#ifndef GPUSIM2GRID_DEVICE_OUTER_MODEL_HPP
#define GPUSIM2GRID_DEVICE_OUTER_MODEL_HPP

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "outer_hook.hpp"
#include "device_tables.hpp"

namespace outer {

// A feature outer_loops_backend='device' does not implement yet: the Python bindings translate it
// to NotImplementedError (the whole batch is refused, nothing falls back on the host).
struct DeviceNotImplemented : std::logic_error {
    explicit DeviceNotImplemented(const std::string& what)
        : std::logic_error("outer_loops_backend='device' does not implement yet: " + what +
                           ". Use outer_loops_backend='host'.") {}
};

// One feature of a grid's outer loops, as the model sees it (docs/outer_loops_on_device.md,
// §4.2): how many of it the grid has (0: none) and whether outer_loops_backend='device'
// implements it. A feature the grid has and the device does not implement refuses 'device'.
struct DeviceFeature {
    std::string name;
    int count = 0;
    bool supported = false;
};

// Throws DeviceNotImplemented naming every feature the grid has (count > 0) and the device does not
// implement, with its count; nothing when there is none.
void check_device_features(const std::vector<DeviceFeature>& features);

// The batched entry points (docs/outer_loops_on_device.md, M3): what the driver calls instead of the
// per-slot OuterHook ones when its hook is a device model. Device pointers; `stream` a cudaStream_t.
class DeviceOuterHook {
public:
    virtual ~DeviceOuterHook() = default;
    // the stream every copy and launch of the model goes on (the driver's)
    virtual void set_stream(void* stream) = 0;
    // the rows enter the slots (OuterHook::load of each, the slot states uploaded together)
    virtual void load_rows(const std::vector<int>& slots, const std::vector<int>& rows) = 0;
    // the rows load_rows set up through the host model since begin() (the others: copies on the device)
    virtual long long n_host_loads() const = 0;
    // every listed slot's inner solve ended (OuterHook::on_solved of each): converged or not, its
    // Newton steps and inner status; a converged one's gathered row (the driver's oc_gather layout,
    // `stride` doubles) is row d_grow[i] of d_gather. Writes the next step (0 Resolve, 1 Done) and the
    // row status of each. Asynchronous on `stream`.
    virtual void decide(int n, const int* d_slots, const int* d_conv, const int* d_nr, const int* d_inner,
                        const int* d_grow, const double* d_gather, int stride, int* d_next, int* d_status,
                        void* stream) = 0;
    // The controls of the listed slots into the driver's per-slot arrays (rows of Layout widths):
    // the dense ones written in place; the Sbus correction and the |V| resets as fixed blocks per
    // listed slot (sbus_block() / vm_reset_block() entries, every one written, unused ones with bus -1;
    // a block may be empty) the driver applies with its own kernels. The resets are one-shot: cleared.
    // Asynchronous on `stream`.
    struct ControlTargets {
        unsigned char* pin = nullptr;
        double* held_q = nullptr;
        unsigned char* plan_held = nullptr;
        double* v_set = nullptr;
        int* hvdc = nullptr;
        int* ds_slot = nullptr;  int* ds_bus = nullptr;  double* ds_p = nullptr;  double* ds_q = nullptr;
        int* vr_slot = nullptr;  int* vr_bus = nullptr;  double* vr_vm = nullptr;
        double* slack_w = nullptr;   // [S x n_slack] a distributed-slack Newton's weights (nullptr: none)
        // ShuntControl, one block of n_sc per listed slot (nullptr: none): sections set (once), their
        // susceptance (pu), the controls (1 on, 0 off, -1 untouched)
        unsigned char* sc_set = nullptr;  double* sc_b_new = nullptr;  signed char* sc_ctl = nullptr;
        // BranchControl, one block of n_bc per listed slot (nullptr: none): out of the row, retapped,
        // the block at the taps (4 complex as re/im pairs), its shift and ratio, the controls
        unsigned char* bc_out = nullptr;  unsigned char* bc_retap = nullptr;  double* bc_tap = nullptr;
        double* bc_a_tap = nullptr;  double* bc_rho_tap = nullptr;  signed char* bc_ctl = nullptr;  signed char* bc_rctl = nullptr;
    };
    // the tail compaction's moves together (OuterHook::move of each pair; the two slot sets are
    // disjoint): device lists d_from / d_to and the same on the host
    virtual void move_slots(int n, const int* d_from, const int* d_to, const int* h_from, const int* h_to) = 0;
    // the slots decide() just finished (next 1): a row other rows resume from (set_row_resume) that
    // ended stable and converged has its point read off its slot and its gathered row (row
    // h_grow[i] of d_gather) and handed to the host model (keep_point), before the slot is reused
    virtual void export_points(int n, const int* h_slots, const int* h_rows, const int* h_next, const int* h_status,
                               const int* h_grow, const double* d_gather, int stride) = 0;
    virtual int sbus_block() const = 0;
    virtual int vm_reset_block() const = 0;
    virtual void write_controls(int n, const int* d_slots, const ControlTargets& t, void* stream) = 0;
};

// The message of a slot's device error (decide's next 2: its status is the code)
std::string device_outer_error(int slot, int code);

// The device model over `host` (a model implementing both interfaces, Ls2gOuterModel; `side` is
// the same object). Throws DeviceNotImplemented for a grid the device does not run yet.
std::shared_ptr<OuterHook> make_device_outer_model_from(std::shared_ptr<OuterHook> host, DeviceHostSide* side);

}  // namespace outer

#endif  // GPUSIM2GRID_DEVICE_OUTER_MODEL_HPP
