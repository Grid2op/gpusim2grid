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
#include <memory>

// Forward-declare session types — complete type not needed here.
struct AcPfNrSession;
struct ContingencyAnalysisSession;
struct InjectionSweepSession;

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