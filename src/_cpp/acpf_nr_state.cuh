// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// acpf_nr_state.cuh  —  Stateful GPU Newton-Raphson AC power flow
// =============================================================================
//
// Motivation
// ----------
// The original acpf_nr_gpu() function bundled setup, NR convergence, and
// teardown in one monolithic call.  Contingency analysis needs to:
//
//   (a) reuse the cuDSS symbolic factorisation (ANALYSIS paid once)
//   (b) reset d_V to the base-case solution for each contingency
//   (c) access the J sparsity skeleton and scatter maps directly
//
// AcPfNrState separates these concerns:
//
//   • Construction   = Phase 0: CPU setup + device upload + ANALYSIS + NR
//   • After ctor     = all GPU buffers valid, cuDSS context alive
//   • d_V_base       = converged base-case voltages (FP32 complex)
//   • d_J_values_base= converged base-case J values  (FP32 real)
//   • dss*           = live cuDSS context (ANALYSIS already done)
//
// The public acpf_nr_gpu() free function becomes a thin wrapper that
// constructs an AcPfNrState, copies V_out back to host, and returns timings.
//
// Thread safety: NOT thread-safe.  One instance per CUDA stream / call site.
//
// Member declaration order
// ------------------------
// Members are destroyed in reverse declaration order.  CuSpMV and CudssContext
// hold handles bound to the stream, so CudaStream cs must be declared FIRST
// so that it is destroyed LAST and outlives every handle bound to it.
//
// =============================================================================

#ifndef ACPF_NR_STATE_CUH
#define ACPF_NR_STATE_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dtypes.hpp"
#include "cuda_utils.h"       // CudaStream, CudssContext, CudssDescriptor,
                               // CuSpMV, CudaTimer
#include "cu_complex_utils.h" // cudaComplexType, CudaFunHelper, CUDA_C_TYPE …
#include "timing_utils.hpp"   // AcPfTimings, TimingEntry

struct LedgerData;            // ledger_data.hpp (host); optional augmented-J input

// =============================================================================
// AcPfNrState
// =============================================================================
struct AcPfNrState {

    // -------------------------------------------------------------------------
    // CUDA stream
    //   All cuSPARSE SpMV, cuDSS phases, Thrust reductions, and custom CUDA
    //   kernels are enqueued on cs.
    //
    //   IMPORTANT — declared FIRST: C++ destroys members in reverse declaration
    //   order, so cs is destroyed LAST, after spmv and dss whose handles are
    //   bound to it.  A stream must not be destroyed before the handles using it.
    // -------------------------------------------------------------------------
    CudaStream cs;

    // -------------------------------------------------------------------------
    // Device ordinal recorded unconditionally after cudaSetDevice in ctor.
    // -------------------------------------------------------------------------
    int device_id_ = 0;

    // -------------------------------------------------------------------------
    // Scalar dimensions  (CPU-side, set in constructor, never modified)
    // -------------------------------------------------------------------------
    int n_bus  = 0;
    int n_pv   = 0;
    int n_pq   = 0;
    int n_pvpq = 0;
    int dim_J  = 0;   // n_pvpq + n_pq
    int nnz_Y  = 0;
    int nnz_J  = 0;

    // -------------------------------------------------------------------------
    // Bus-index vectors (device)
    // -------------------------------------------------------------------------
    thrust::device_vector<int> d_pvpq;   // sorted pvpq bus indices
    thrust::device_vector<int> d_pq;     // pq bus indices

    // -------------------------------------------------------------------------
    // NRLedger pair lists (device) — define the augmented Jacobian layout.
    //   Each (bus, row/col) pair: which solver bus owns the equation/unknown and
    //   its J row/column. Built once (trivial ledger from pv/pq for a feature-free
    //   grid; the augmented ledger from lightsim2grid once extensions are active)
    //   and shared, single-system, across every batch slot — exactly like the
    //   scatter maps. Counts n_p/n_q/n_theta/n_vm may exceed n_pvpq/n_pq when an
    //   extension adds rows/columns.
    // -------------------------------------------------------------------------
    int n_p = 0, n_q = 0, n_theta = 0, n_vm = 0;
    thrust::device_vector<int> d_p_buses,     d_p_rows;      // P equations
    thrust::device_vector<int> d_q_buses,     d_q_rows;      // Q equations
    thrust::device_vector<int> d_theta_buses, d_theta_cols;  // theta unknowns
    thrust::device_vector<int> d_vm_buses,    d_vm_cols;     // vm unknowns

    // -------------------------------------------------------------------------
    // MultiSlack (distributed slack in the Jacobian) — slack_col < 0 when absent.
    //   d_slack_absorbed is the per-batch-slot running state (size 1 for the
    //   single-system base solve; tiled to batch_size by the batch driver).
    //   The other arrays are shared single-system (length n_slack).
    // -------------------------------------------------------------------------
    int slack_col = -1;
    int n_slack   = 0;
    thrust::device_vector<int>            d_slack_prow;       // [n_slack] P row of each slack
    thrust::device_vector<cuda_real_type> d_slack_w;          // [n_slack] slack weight
    thrust::device_vector<int>            d_slack_feat_pos;   // [n_slack] J pos (p_row, slack_col)
    thrust::device_vector<cuda_real_type> d_slack_absorbed;   // [1] base-solve running state

    // -------------------------------------------------------------------------
    // Ybus CSR (device, complex cuda_real_type)
    // -------------------------------------------------------------------------
    thrust::device_vector<int>             d_Ybus_outer;   // size n_bus+1
    thrust::device_vector<int>             d_Ybus_inner;   // size nnz_Y
    thrust::device_vector<cudaComplexType> d_Ybus_values;  // size nnz_Y

    // -------------------------------------------------------------------------
    // Working vectors (device)
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V;     // current voltages
    thrust::device_vector<cudaComplexType> d_Ibus;  // cuSPARSE SpMV output
    thrust::device_vector<cudaComplexType> d_Sbus;  // scheduled injections

    thrust::device_vector<cuda_real_type>  d_F;     // mismatch RHS,  size dim_J
    thrust::device_vector<cuda_real_type>  d_dx;    // NR correction, size dim_J

    // -------------------------------------------------------------------------
    // Jacobian CSR skeleton (device, real cuda_real_type)
    //   Sparsity pattern is FIXED across all NR iterations and contingencies.
    // -------------------------------------------------------------------------
    thrust::device_vector<int>            d_J_outer;   // size dim_J+1
    thrust::device_vector<int>            d_J_inner;   // size nnz_J
    thrust::device_vector<cuda_real_type> d_J_values;  // current numeric values

    // -------------------------------------------------------------------------
    // Scatter maps  (device, size nnz_Y each)
    //   d_map_j{11,12,21,22}[k]  = position in d_J_values[] where Ybus nnz k
    //   contributes to sub-block J11/J12/J21/J22 respectively (-1 = absent).
    //   Precomputed once from Ybus_rm (RowMajor) by build_J_structure().
    // -------------------------------------------------------------------------
    thrust::device_vector<int> d_map_j11;
    thrust::device_vector<int> d_map_j12;
    thrust::device_vector<int> d_map_j21;
    thrust::device_vector<int> d_map_j22;

    // -------------------------------------------------------------------------
    // Base-case snapshots  (saved after NR convergence; device-to-device copy)
    //   ContingencyAnalysisSolver tiles d_V_base into its batch voltage array
    //   and optionally seeds d_J_values from d_J_values_base.
    // -------------------------------------------------------------------------
    thrust::device_vector<cudaComplexType> d_V_base;          // converged V
    thrust::device_vector<cuda_real_type>  d_J_values_base;   // converged J values

    // -------------------------------------------------------------------------
    // cuSPARSE SpMV descriptor  (Ybus · d_V → d_Ibus)
    //   Uses CUSPARSE_POINTER_MODE_HOST so alpha/beta live on the CPU stack
    //   (h_cplx_one / h_cplx_zero from cu_complex_utils.h).
    //   No CudaDeviceScalar needed.
    // -------------------------------------------------------------------------
    CuSpMV spmv;

    // -------------------------------------------------------------------------
    // cuDSS context — KEPT ALIVE after construction
    //   ANALYSIS has been executed; dss_A/dss_b/dss_x point into d_J_* and
    //   d_F / d_dx.  ContingencyAnalysisSolver calls FACTORIZATION /
    //   REFACTORIZATION / SOLVE directly on these objects, amortising the
    //   symbolic factorisation across the entire contingency batch.
    // -------------------------------------------------------------------------
    CudssContext    dss;    // handle + config + data
    CudssDescriptor dss_A;  // sparse J descriptor
    CudssDescriptor dss_b;  // dense F descriptor (RHS)
    CudssDescriptor dss_x;  // dense dx descriptor (solution)

    // -------------------------------------------------------------------------
    // Jᵀ CSR and cuDSS context for the adjoint solve  (built at end of ctor)
    //
    // cuDSS 0.7 exposes CUDSS_CONFIG_SOLVE_MODE for a future transpose mode,
    // but only value 0 (forward solve) is currently supported.  We build Jᵀ
    // via cusparseCsr2cscEx2 (CSC of J == CSR of Jᵀ for a square matrix) and
    // factor it with a separate CudssContext reused across all backward calls.
    // -------------------------------------------------------------------------
    thrust::device_vector<int>            d_JT_outer;   // size dim_J+1
    thrust::device_vector<int>            d_JT_inner;   // size nnz_J
    thrust::device_vector<cuda_real_type> d_JT_values;  // size nnz_J
    thrust::device_vector<cuda_real_type> d_JT_rhs;     // work RHS, size dim_J
    thrust::device_vector<cuda_real_type> d_JT_sol;     // solution, size dim_J

    CudssContext    dss_T;   // separate cuDSS context for Jᵀ
    CudssDescriptor dss_AT;  // sparse Jᵀ descriptor
    CudssDescriptor dss_bT;  // dense RHS descriptor
    CudssDescriptor dss_xT;  // dense solution descriptor

    // -------------------------------------------------------------------------
    // Timings captured during construction
    // -------------------------------------------------------------------------
    AcPfTimings timings;

    // =========================================================================
    // Constructor
    //   Performs the complete Phase 0:
    //     1. CPU preprocessing  (pvpq, Ybus_rm, J skeleton, scatter maps)
    //     2. Device uploads
    //     3. cuSPARSE + cuDSS descriptors + stream binding
    //     4. ANALYSIS  (reordering + symbolic factorisation, paid once)
    //     5. Newton-Raphson loop to convergence
    //     6. Device-to-device snapshot of d_V_base and d_J_values_base
    //
    //   Throws std::runtime_error on any CUDA / cuDSS / cuSPARSE error.
    // =========================================================================
    AcPfNrState(
        const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
        Eigen::Ref<const CplxVect>                  Vinit,
        Eigen::Ref<const CplxVect>                  Sbus,
        Eigen::Ref<const Eigen::VectorXi>           pv,
        Eigen::Ref<const Eigen::VectorXi>           pq,
        int                                         max_iter,
        eigen_real_type                             tol,
        int                                         device = -1,
        // Optional augmented-J description read off a solved lightsim2grid grid.
        // nullptr → build the trivial feature-free ledger from pv/pq (Phase 1).
        const LedgerData*                           ledger = nullptr
    );

    // =========================================================================
    // Destructor
    //   All members are RAII (CudaStream, CudaEvent, CuSpMV, CudssContext,
    //   CudssDescriptor, thrust::device_vectors) — default destructor is
    //   correct.  Destruction order is reverse declaration order, so cs
    //   (declared last) is destroyed after all handles bound to it.
    // =========================================================================
    ~AcPfNrState() = default;

    // =========================================================================
    // copy_V_to_host
    //   Copies d_V back to a host Eigen vector (one D→H transfer).
    //   Called by the thin acpf_nr_gpu() wrapper and by the contingency solver
    //   to retrieve the base-case voltages for correctness validation.
    // =========================================================================
    void copy_V_to_host(Eigen::Ref<CplxVect> V_out) const;

    // =========================================================================
    // d_V_ptr — raw device pointer into d_V for zero-copy DLPack export.
    // =========================================================================
    const cudaComplexType* d_V_ptr() const {
        return thrust::raw_pointer_cast(d_V.data());
    }

    // =========================================================================
    // Jᵀ adjoint helpers
    //   prepare_JT() — called once at the end of the constructor.
    //   solve_JT()   — called each backward pass; reuses the factored Jᵀ.
    //   Both pointers are device pointers of size dim_J.
    // =========================================================================
    void prepare_JT();
    void solve_JT(const cuda_real_type* d_rhs, cuda_real_type* d_sol);

    // =========================================================================
    // Non-copyable, non-movable
    //   cuSPARSE / cuDSS descriptors hold raw pointers into the thrust vectors.
    //   Moving those vectors would silently dangle those pointers.
    // =========================================================================
    AcPfNrState(const AcPfNrState&)            = delete;
    AcPfNrState& operator=(const AcPfNrState&) = delete;
    AcPfNrState(AcPfNrState&&)                 = delete;
    AcPfNrState& operator=(AcPfNrState&&)      = delete;
};

#endif  // ACPF_NR_STATE_H