// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// dlpack_export.cu — DLPack capsule exporter implementations.
//
// Compiled by nvcc so the complete CUDA types (AcPfNrState, BatchPfDriver,
// thrust::device_vector) are fully visible.
//
// Lifetime contract
// -----------------
//   Each returned capsule captures a shared_ptr<Session> as the owner.
//   The session stays alive as long as any consumer holds the DLTensor.
//   The deleter frees ONLY the DLManagedTensor wrapper + DLPackCtx — it
//   never frees dl_tensor.data (owned by a thrust::device_vector inside
//   the session's solver).
//
//   WARNING: calling run() again overwrites d_V_results in-place.  Clone
//   the tensor before a subsequent run() if a snapshot is needed.
// =============================================================================

#include "dlpack_export.hpp"
#include "dlpack_export.cuh"

#include "acpf_nr_state.cuh"
#include "acpf_nr.hpp"
#include "contingency_analysis_session.hpp"
#include "injection_sweep_session.hpp"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/contingency_batch.cuh"
#include "contingency/batch_sources/injection_batch.cuh"

#include <pybind11/pybind11.h>
#include <stdexcept>

// =============================================================================
// Capsule destructor — registered as the PyCapsule destructor.
// Calls the DLManagedTensor deleter only if the framework has not yet
// consumed the capsule (name still "dltensor").  If already renamed to
// "used_dltensor" by from_dlpack(), the framework owns it — do nothing.
// =============================================================================
static void capsule_destructor(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* mt = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        if (mt && mt->deleter) mt->deleter(mt);
    }
}

// =============================================================================
// AcPfNrSession exporter
// =============================================================================

pybind11::capsule
export_v_acpfnr_dlpack(std::shared_ptr<AcPfNrSession> self)
{
    self->state_->cs.synchronize();
    void*   ptr = const_cast<void*>(
        static_cast<const void*>(self->state_->d_V_ptr()));
    int64_t n   = static_cast<int64_t>(self->state_->n_bus);
    int     dev = self->state_->device_id_;
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, n, 0, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

// =============================================================================
// AcPfNrSession adjoint solve exporter
// =============================================================================

pybind11::capsule
export_jt_solve_dlpack(std::shared_ptr<AcPfNrSession> self,
                       pybind11::capsule              rhs_capsule)
{
    // Unpack the incoming DLPack capsule.  The capsule must still be named
    // "dltensor" (i.e. not yet consumed by torch.from_dlpack).
    auto* rhs_mt = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(rhs_capsule.ptr(), "dltensor"));
    if (!rhs_mt)
        throw std::runtime_error(
            "solve_JT_dlpack: invalid or already-consumed DLPack capsule "
            "(expected name \"dltensor\")");

    const auto* rhs_ptr =
        static_cast<const cuda_real_type*>(rhs_mt->dl_tensor.data);

    // Sync stream, solve, sync again so the result is visible to the caller.
    self->state_->cs.synchronize();
    self->state_->solve_JT(
        rhs_ptr,
        thrust::raw_pointer_cast(self->state_->d_JT_sol.data()));
    self->state_->cs.synchronize();

    // Export d_JT_sol as a real-float DLPack capsule.
    // The DLPackCtx captures the session shared_ptr as owner, keeping the
    // device buffer alive as long as the consumer holds the capsule or the
    // tensor derived from it.
    void*   ptr   = thrust::raw_pointer_cast(self->state_->d_JT_sol.data());
    int64_t n     = static_cast<int64_t>(self->state_->dim_J);
    int     dev   = self->state_->device_id_;
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt    = make_dl_tensor_real(ptr, dev, n, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

// =============================================================================
// ContingencyAnalysisSession exporters
// =============================================================================

pybind11::capsule
export_v_base_dlpack(std::shared_ptr<ContingencyAnalysisSession> self)
{
    self->base_state_->cs.synchronize();
    void*   ptr = const_cast<void*>(
        static_cast<const void*>(self->base_state_->d_V_ptr()));
    int64_t n   = static_cast<int64_t>(self->base_state_->n_bus);
    int     dev = self->base_state_->device_id_;
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, n, 0, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

pybind11::capsule
export_v_results_dlpack(std::shared_ptr<ContingencyAnalysisSession> self)
{
    if (!self->solver_)
        throw std::runtime_error(
            "ContingencyAnalysisSession: run() must be called before "
            "v_results_dlpack()");
    self->solver_->synchronize();
    void*   ptr = const_cast<void*>(
        static_cast<const void*>(self->solver_->d_V_results_ptr()));
    int64_t nc  = static_cast<int64_t>(self->solver_->n_contingencies_());
    int64_t nb  = static_cast<int64_t>(self->solver_->n_bus());
    int     dev = self->solver_->device_id();
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, nc, nb, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

// =============================================================================
// InjectionSweepSession exporters
// =============================================================================

pybind11::capsule
export_v_base_dlpack_inj(std::shared_ptr<InjectionSweepSession> self)
{
    self->base_state_->cs.synchronize();
    void*   ptr = const_cast<void*>(
        static_cast<const void*>(self->base_state_->d_V_ptr()));
    int64_t n   = static_cast<int64_t>(self->base_state_->n_bus);
    int     dev = self->base_state_->device_id_;
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, n, 0, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

pybind11::capsule
export_v_results_dlpack_inj(std::shared_ptr<InjectionSweepSession> self)
{
    if (!self->solver_)
        throw std::runtime_error(
            "InjectionSweepSession: run() must be called before "
            "v_results_dlpack()");
    self->solver_->synchronize();
    void*   ptr = const_cast<void*>(
        static_cast<const void*>(self->solver_->d_V_results_ptr()));
    int64_t ns  = static_cast<int64_t>(self->solver_->n_contingencies_());
    int64_t nb  = static_cast<int64_t>(self->solver_->n_bus());
    int     dev = self->solver_->device_id();
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, ns, nb, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}