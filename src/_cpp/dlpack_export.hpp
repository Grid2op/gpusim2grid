// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// dlpack_export.hpp — declarations for DLPack capsule exporters.
// Include this from python_bindings.cpp (g++).
// Definitions live in dlpack_export.cu (nvcc).
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>
#include <cstdint>
#include <memory>

#include "Eigen/Core"

// Forward-declare session types — complete type not needed here.
struct AcPfNrSession;
struct ContingencyAnalysisSession;
struct InjectionSweepSession;
struct ScenarioSweepSession;

// AcPfNrSession voltage (shape [n_bus]) — syncs state_->cs.
pybind11::capsule export_v_acpfnr_dlpack(
    std::shared_ptr<AcPfNrSession> self);

// AcPfNrSession adjoint solve: Jᵀλ = rhs.
// rhs_capsule: DLPack "dltensor" capsule, shape [dim_J], real float, on device.
// Returns: DLPack capsule for λ (shape [dim_J], real float).
// WARNING: the returned capsule aliases an internal buffer — clone before the
// next solve_JT_dlpack call to avoid overwriting.
pybind11::capsule export_jt_solve_dlpack(
    std::shared_ptr<AcPfNrSession> self,
    pybind11::capsule rhs_capsule);

// Base-case voltage (shape [n_bus]) — syncs base_state_->cs.
pybind11::capsule export_v_base_dlpack(
    std::shared_ptr<ContingencyAnalysisSession> self);

pybind11::capsule export_v_base_dlpack_inj(
    std::shared_ptr<InjectionSweepSession> self);

// Batch result voltages — requires run() to have been called first.
// ContingencyAnalysisSession: shape [n_contingencies, n_bus]
// InjectionSweepSession:      shape [n_scenarios,     n_bus]
pybind11::capsule export_v_results_dlpack(
    std::shared_ptr<ContingencyAnalysisSession> self);

pybind11::capsule export_v_results_dlpack_inj(
    std::shared_ptr<InjectionSweepSession> self);

// ScenarioSweepSession:       shape [n_scenarios,     n_bus]
pybind11::capsule export_v_base_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self);

pybind11::capsule export_v_results_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self);

// -----------------------------------------------------------------------------
// ScenarioSweepSession differentiable-path interop (see _batch_pf.py).
//
// Importers CONSUME the capsule (renamed to "used_dltensor", deleter called
// once the D2D copy is host-synchronized), validate dtype / device / shape /
// contiguity, and hand the device pointer to the session. producer_stream is
// the cudaStream_t handle (as an integer) the tensor was produced on --
// torch.cuda.current_stream().cuda_stream -- or 0.
// -----------------------------------------------------------------------------
// (n_scenarios, n_bus) complex, per-unit Sbus.
void import_injections_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self,
    pybind11::capsule capsule,
    std::uintptr_t producer_stream);

// (n_scenarios, n_gen) real vm_pu; gen_bus (n_gen,) AC-solver bus per generator.
void import_gen_v_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self,
    pybind11::capsule capsule,
    Eigen::Ref<const Eigen::VectorXi> gen_bus,
    std::uintptr_t producer_stream);

// Batched adjoint: rhs (n_scenarios, dim_J) real; optional snapshots
// j_values (capacity, nnz_J) real, ybus_values (capacity, nnz_Y) complex,
// v (n_scenarios, n_bus) complex (pybind11::none() when absent). Returns
// (lambda capsule [n_scenarios, dim_J], gvm capsule [n_scenarios, n_bus] or
// None); both alias driver buffers overwritten by the next call -- clone.
pybind11::tuple export_solve_jt_batch_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self,
    pybind11::capsule rhs_capsule,
    pybind11::object  j_values_capsule,
    pybind11::object  ybus_values_capsule,
    pybind11::object  v_capsule,
    bool              want_gen_v_grad,
    std::uintptr_t    producer_stream);

// Chunk-buffer aliases for the snapshot mode: [capacity, nnz_J] real /
// [capacity, nnz_Y] complex, active-slot order. Clone before the next run().
pybind11::capsule export_j_values_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self);
pybind11::capsule export_ybus_values_dlpack_ss(
    std::shared_ptr<ScenarioSweepSession> self);