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
#include "scenario_sweep_session.hpp"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/contingency_batch.cuh"
#include "contingency/batch_sources/injection_batch.cuh"
#include "contingency/batch_sources/scenario_sweep_batch.cuh"

#include <pybind11/pybind11.h>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

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

// =============================================================================
// ScenarioSweepSession exporters
// =============================================================================

pybind11::capsule
export_v_base_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self)
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
export_v_results_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self)
{
    if (!self->solver_)
        throw std::runtime_error(
            "ScenarioSweepSession: run() must be called before "
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

// =============================================================================
// ScenarioSweepSession differentiable-path importers / exporters
// =============================================================================

namespace {

// A borrowed view of an incoming DLPack capsule: validated pointer + the
// managed tensor, consumed (renamed + deleter run) by release().
struct DLInput {
    DLManagedTensor* mt   = nullptr;
    const void*      data = nullptr;
    PyObject*        capsule = nullptr;

    void release() {
        if (!mt) return;
        // Protocol: the consumer renames the capsule so the producer's
        // destructor never double-frees, then calls the deleter itself.
        PyCapsule_SetName(capsule, "used_dltensor");
        if (mt->deleter) mt->deleter(mt);
        mt = nullptr;
    }
};

// Validate a "dltensor" capsule: on this device, dtype (code, bits), ndim and
// shape (a -1 entry accepts any extent), compact row-major (strides null or
// matching). Returns the view; the caller must release() it once the data
// has been consumed (host-synchronized copy).
DLInput check_dl_input(pybind11::handle capsule, const char* what,
                       int device_id, uint8_t dtype_code, int dtype_bits,
                       std::initializer_list<int64_t> shape)
{
    DLInput in;
    in.capsule = capsule.ptr();
    if (!PyCapsule_IsValid(in.capsule, "dltensor"))
        throw std::runtime_error(
            std::string(what) + ": invalid or already-consumed DLPack capsule "
            "(expected name \"dltensor\")");
    in.mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(in.capsule, "dltensor"));
    const DLTensor& t = in.mt->dl_tensor;
    if (t.device.device_type != kDLCUDA || t.device.device_id != device_id)
        throw std::runtime_error(
            std::string(what) + ": the tensor must live on CUDA device "
            + std::to_string(device_id) + " (the session's device)");
    if (t.dtype.code != dtype_code || t.dtype.bits != dtype_bits || t.dtype.lanes != 1)
        throw std::runtime_error(
            std::string(what) + ": wrong dtype -- expected "
            + (dtype_code == kDLComplex ? "complex" : "float")
            + std::to_string(dtype_bits) + " (this build's precision)");
    const int ndim = static_cast<int>(shape.size());
    if (t.ndim != ndim)
        throw std::runtime_error(
            std::string(what) + ": expected a " + std::to_string(ndim) + "-D tensor, got "
            + std::to_string(t.ndim) + "-D");
    int d = 0;
    for (int64_t expected : shape) {
        if (expected >= 0 && t.shape[d] != expected)
            throw std::runtime_error(
                std::string(what) + ": dimension " + std::to_string(d) + " has extent "
                + std::to_string(t.shape[d]) + ", expected " + std::to_string(expected));
        ++d;
    }
    if (t.strides) {
        int64_t expected = 1;
        for (int k = t.ndim - 1; k >= 0; --k) {
            if (t.shape[k] > 1 && t.strides[k] != expected)
                throw std::runtime_error(
                    std::string(what) + ": the tensor must be contiguous (row-major); "
                    "call .contiguous() first");
            expected *= t.shape[k];
        }
    }
    in.data = static_cast<const char*>(t.data) + t.byte_offset;
    return in;
}

}  // namespace

void import_injections_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self,
                                 pybind11::capsule capsule,
                                 std::uintptr_t producer_stream)
{
    const int dev   = self->base_state_->device_id_;
    const int n_bus = self->base_state_->n_bus;
    DLInput in = check_dl_input(capsule, "set_injections_dlpack", dev,
                                kDLComplex, kDLPackComplexBits, {-1, n_bus});
    const int n_scen = static_cast<int>(in.mt->dl_tensor.shape[0]);
    try {
        self->set_injections_device(in.data, n_scen, n_bus, producer_stream);
    } catch (...) {
        in.release();
        throw;
    }
    in.release();
}

void import_gen_v_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self,
                            pybind11::capsule capsule,
                            Eigen::Ref<const Eigen::VectorXi> gen_bus,
                            std::uintptr_t producer_stream)
{
    const int dev   = self->base_state_->device_id_;
    const int n_gen = static_cast<int>(gen_bus.size());
    DLInput in = check_dl_input(capsule, "set_gen_v_dlpack", dev,
                                kDLFloat, kDLPackRealBits, {-1, n_gen});
    const int n_scen = static_cast<int>(in.mt->dl_tensor.shape[0]);
    try {
        self->set_gen_v_device(in.data, n_scen, n_gen, gen_bus, producer_stream);
    } catch (...) {
        in.release();
        throw;
    }
    in.release();
}

pybind11::tuple
export_solve_jt_batch_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self,
                                pybind11::capsule rhs_capsule,
                                pybind11::object  j_values_capsule,
                                pybind11::object  ybus_values_capsule,
                                pybind11::object  v_capsule,
                                bool              want_gen_v_grad,
                                std::uintptr_t    producer_stream)
{
    if (!self->solver_)
        throw std::runtime_error(
            "ScenarioSweepSession: run() must be called before solve_JT_batch_dlpack()");
    const int dev    = self->base_state_->device_id_;
    const int n_scen = self->solver_->n_contingencies_();
    const int n_bus  = self->base_state_->n_bus;
    const int dim_J  = self->base_state_->dim_J;
    const int nnz_J  = self->base_state_->nnz_J;
    const int nnz_Y  = self->base_state_->nnz_Y;
    const int cap    = self->solver_->batch_size_;

    DLInput rhs = check_dl_input(rhs_capsule, "solve_JT_batch_dlpack(rhs)", dev,
                                 kDLFloat, kDLPackRealBits, {n_scen, dim_J});
    DLInput jv, yv, vv;
    std::vector<DLInput*> inputs{&rhs};
    try {
        if (!j_values_capsule.is_none()) {
            jv = check_dl_input(j_values_capsule, "solve_JT_batch_dlpack(j_values)", dev,
                                kDLFloat, kDLPackRealBits, {cap, nnz_J});
            inputs.push_back(&jv);
        }
        if (!ybus_values_capsule.is_none()) {
            yv = check_dl_input(ybus_values_capsule, "solve_JT_batch_dlpack(ybus_values)", dev,
                                kDLComplex, kDLPackComplexBits, {cap, nnz_Y});
            inputs.push_back(&yv);
        }
        if (!v_capsule.is_none()) {
            vv = check_dl_input(v_capsule, "solve_JT_batch_dlpack(v)", dev,
                                kDLComplex, kDLPackComplexBits, {n_scen, n_bus});
            inputs.push_back(&vv);
        }
        self->solve_JT_batch(rhs.data, jv.data, want_gen_v_grad, yv.data, vv.data,
                             producer_stream);
    } catch (...) {
        for (DLInput* p : inputs) p->release();
        throw;
    }
    // solve_JT_batch host-synchronizes before returning: the inputs may go.
    for (DLInput* p : inputs) p->release();

    auto owner = std::static_pointer_cast<void>(self);
    void* lam_ptr = const_cast<void*>(
        static_cast<const void*>(self->solver_->d_JT_sol_full_ptr()));
    auto* lam_mt = make_dl_tensor_real(lam_ptr, dev, n_scen, owner, dim_J);
    pybind11::capsule lam_cap(lam_mt, "dltensor", capsule_destructor);

    pybind11::object gvm_obj = pybind11::none();
    if (want_gen_v_grad) {
        void* gvm_ptr = const_cast<void*>(
            static_cast<const void*>(self->solver_->d_gvm_full_ptr()));
        auto* gvm_mt = make_dl_tensor_real(gvm_ptr, dev, n_scen, owner, n_bus);
        gvm_obj = pybind11::capsule(gvm_mt, "dltensor", capsule_destructor);
    }
    return pybind11::make_tuple(lam_cap, gvm_obj);
}

pybind11::capsule
export_j_values_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self)
{
    if (!self->solver_)
        throw std::runtime_error(
            "ScenarioSweepSession: run() must be called before j_values_dlpack()");
    self->solver_->synchronize();
    void*   ptr = const_cast<void*>(static_cast<const void*>(self->solver_->j_values_ptr()));
    int     dev = self->solver_->device_id();
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor_real(ptr, dev, self->solver_->batch_size_, std::move(owner),
                                      self->base_state_->nnz_J);
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}

pybind11::capsule
export_ybus_values_dlpack_ss(std::shared_ptr<ScenarioSweepSession> self)
{
    if (!self->solver_)
        throw std::runtime_error(
            "ScenarioSweepSession: run() must be called before ybus_values_dlpack()");
    self->solver_->synchronize();
    void*   ptr = const_cast<void*>(static_cast<const void*>(self->solver_->ybus_values_ptr()));
    int     dev = self->solver_->device_id();
    auto    owner = std::static_pointer_cast<void>(self);
    auto*   mt  = make_dl_tensor(ptr, dev, self->solver_->batch_size_,
                                 self->base_state_->nnz_Y, std::move(owner));
    return pybind11::capsule(mt, "dltensor", capsule_destructor);
}