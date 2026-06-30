// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INJECTION_BATCH_CUH
#define INJECTION_BATCH_CUH

// =============================================================================
// contingency/batch_sources/injection_batch.cuh
//
// InjectionBatch — BatchSource policy for the dense-injection PF sweep.
//
// Inputs (host side, captured at construction):
//   h_Sbus_all : (n_scenario × n_bus) per-unit complex injections, row-major.
//                Built upstream from (p_mw, q_mvar, sn_mva) by the Python entry
//                point.  Uploaded to d_Sbus_all in initialize().
//
// Per-chunk behaviour
// -------------------
//   prepare_Ybus_batch :
//     ① tile base d_V into d_V_batch                  (batch_size D→D copies)
//     (no Ybus mutation — base Ybus has already been tiled into
//      d_Ybus_values_batch once during initialize().)
//
//   prepare_Sbus_batch :
//     ② cudaMemcpyAsync the (actual_batch × n_bus) row-slice of d_Sbus_all
//        into d_Sbus_batch.
//     ③ Phantom-pad remaining slots (batch_size - actual_batch) with
//        base.d_Sbus, so phantom NR is a converged no-op.
//
// sbus_stride = n_bus → fill_FP/FQ kernels read d_Sbus[b * n_bus + bus],
// one row per scenario.
// =============================================================================

#include <chrono>
#include <cstddef>
#include <thrust/device_vector.h>
#include <vector>

#include "../../dtypes.hpp"
#include "../../cuda_utils.h"
#include "../../cu_complex_utils.h"
#include "../../timing_utils.hpp"
#include "../../acpf_nr_state.cuh"

struct BatchPfDriverContext;
struct NrIterBuffers;

inline double ib_ms_since(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

struct InjectionBatch {

    // -------------------------------------------------------------------------
    // sbus_stride = n_bus  → per-scenario row indexing in fill_FP/FQ kernels.
    // -------------------------------------------------------------------------
    static int sbus_stride(int n_bus) { return n_bus; }

    // -------------------------------------------------------------------------
    // Host-side per-scenario Sbus (n_scenario × n_bus complex per-unit)
    // -------------------------------------------------------------------------
    std::vector<cudaComplexType> h_Sbus_all_;
    int                          n_scenarios_ = 0;

    // -------------------------------------------------------------------------
    // Device-side full Sbus + per-chunk Sbus
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_Sbus_all;     // n_scenario × n_bus
    thrust::device_vector<cudaComplexType> d_Sbus_batch;   // batch_size × n_bus

    // Preprocess time (CPU build of h_Sbus_all_).  Filled by the caller (the
    // public entry point), since the host-side complex build happens there.
    double t_preprocess_ms = 0.0;

    // -------------------------------------------------------------------------
    // Constructor (host-only).
    //   h_Sbus_all  : moved in — (n_scenario × n_bus) cudaComplexType, row-major.
    //   n_scenarios : the leading dimension.
    //   preprocess_ms : optional caller-side timing for h_Sbus_all assembly.
    // -------------------------------------------------------------------------
    InjectionBatch(std::vector<cudaComplexType>&& h_Sbus_all,
                   int                            n_scenarios,
                   double                         preprocess_ms = 0.0)
        : h_Sbus_all_(std::move(h_Sbus_all))
        , n_scenarios_(n_scenarios)
        , t_preprocess_ms(preprocess_ms)
    {}

    InjectionBatch(InjectionBatch&&) noexcept = default;
    InjectionBatch(const InjectionBatch&) = delete;
    InjectionBatch& operator=(const InjectionBatch&) = delete;
    InjectionBatch& operator=(InjectionBatch&&) = delete;

    // -------------------------------------------------------------------------
    // initialize  — runs after BatchPfDriver allocates its chunk buffers.
    //
    //   • Upload d_Sbus_all from h_Sbus_all_.
    //   • Allocate d_Sbus_batch (batch_size × n_bus).
    //   • Tile base d_Ybus_values into d_Ybus_values_batch ONCE.  Ybus is
    //     never modified per chunk in the injection sweep — fill_J reads
    //     identical base values from every batch slot.
    // -------------------------------------------------------------------------
    void initialize(BatchPfDriverContext& ctx, cudaStream_t cs);

    // -------------------------------------------------------------------------
    // prepare_Ybus_batch — tile V only.  Ybus is permanent.
    // -------------------------------------------------------------------------
    void prepare_Ybus_batch(BatchPfDriverContext& ctx,
                            int                  /*chunk_idx*/,
                            int                  /*actual_batch*/,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&  t);

    // -------------------------------------------------------------------------
    // prepare_Sbus_batch — copy the row-slice + phantom-pad with base.d_Sbus.
    // -------------------------------------------------------------------------
    void prepare_Sbus_batch(BatchPfDriverContext& ctx,
                            int                  chunk_idx,
                            int                  actual_batch,
                            cudaStream_t         cs,
                            CudaTimer&           timer,
                            BatchTimings&  t);

    // -------------------------------------------------------------------------
    // d_Sbus_ptr  — returns the per-batch Sbus pointer.
    // -------------------------------------------------------------------------
    const cudaComplexType* d_Sbus_ptr(const BatchPfDriverContext& /*ctx*/) const {
        return thrust::raw_pointer_cast(d_Sbus_batch.data());
    }

    // -------------------------------------------------------------------------
    // Active-set interface (BatchSource concept).  The injection sweep has no
    // concept of disconnection — every scenario is solved, so the active set is
    // the full set and the result map is the identity (nullptr → fast path).
    // -------------------------------------------------------------------------
    int        n_active()     const { return n_scenarios_; }
    const int* d_result_map() const { return nullptr; }

    // BatchSource concept: the injection sweep never masks buses (topology is
    // fixed), so this is a no-op that leaves the NrIterBuffers mask fields at
    // their off defaults.
    void fill_mask_buffers(NrIterBuffers& /*buf*/, int /*chunk_idx*/,
                           const int* /*d_J_outer*/) const {}

    double cpu_preprocess_ms() const { return t_preprocess_ms; }
};

#endif // INJECTION_BATCH_CUH