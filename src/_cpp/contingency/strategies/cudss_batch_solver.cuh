// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// contingency/strategies/cudss_batch_solver.cuh
// =============================================================================
//
// CudssBatchSolver — owns the cuDSS batch context and the three matrix/vector
// descriptors (A, x, b).  Exposes four operations:
//
//   initialize()   — create handle/config/data, build descriptors, run ANALYSIS
//   set_values()   — update the J values pointer in dss_A_ (CPU call, no GPU work)
//   factor()       — CUDSS_PHASE_FACTORIZATION  (first numeric factorization)
//   refactor()     — CUDSS_PHASE_REFACTORIZATION (subsequent refactorizations)
//   solve()        — CUDSS_PHASE_SOLVE
//   refresh_factor_stats() — read CUDSS_DATA_LU_NNZ / CUDSS_DATA_MEMORY_ESTIMATES
//                    of the last ANALYSIS into factor_stats()
//
// This struct owns only the linear-algebra machinery.  It holds no state about
// *when* to factorize — that is left entirely to the policy that uses it.
//
// Batch mode (experimental, selected by environment variable)
// -----------------------------------------------------------
// The caller always hands over the same contiguous buffers: J values
// [ubatch × nnz_J], F and dx [ubatch × dim_J], slot after slot, and one
// single-system CSR skeleton. How cuDSS is told about them is picked at
// initialize() time:
//
//   Uniform    (default)                  CUDSS_CONFIG_UBATCH_SIZE: one
//              ANALYSIS of the shared pattern, reused by every slot.
//   BlockDiag  GPUSIM2GRID_USE_BLOCKDIAG=1  one plain CSR matrix of size
//              ubatch·dim_J whose blocks are the slots (the buffers already
//              have that layout); one ANALYSIS over the whole matrix.
//   NonUniform GPUSIM2GRID_USE_BATCH_MODE=1 cudssMatrixCreateBatchCsr /
//              BatchDn: a "non-uniform" batch of ubatch independent matrices
//              that happen to share one skeleton (every row-pointer and
//              column-index entry aliases the single-system arrays).
//
// The two variables are read on every initialize(), so they take effect on
// the next driver build (every run() for contingency analysis / injection
// sweep; a cold rebuild for ScenarioSweepSession). Setting both is an error.
// A one-line notice goes to stderr the first time a non-default mode is used.
//
// Re-analysis per chunk (BlockDiag / NonUniform only): the systems of one chunk
// have nothing to do with those of the next, so ANALYSIS is redone for every
// chunk, on that chunk's own J values. begin_chunk() (called by run_nr_loop)
// only marks it pending; it runs in prepare_factorization() -- called by the
// policies right before they start their factorization timer, and again
// (no-op by then) from factor()/refactor() -- so the analysis wall time lands
// in analysis_ms(), not in the factorization timings. The first factorization
// after an analysis is a FACTORIZATION even when refactor() is asked for. The
// ANALYSIS of initialize() is deferred the same way (J values are not filled
// yet at that point). A policy that never refactorizes
// (direct_base_case_factors) keeps its base-case factors for every chunk.
//
// Destruction order
// -----------------
// Declared members top-to-bottom; destroyed bottom-to-top by C++ rules.
// dss_ must be declared first so it is destroyed last: CudssContext::~CudssContext
// calls cudssDataDestroy(handle, data) which requires a live handle. The
// storage the descriptors reference (block-diagonal skeleton, batch pointer
// and size arrays) is declared before the descriptors so it outlives them.
//
// Thread safety: NOT thread-safe.  One instance per solver / CUDA stream.
// =============================================================================

#ifndef CUDSS_BATCH_SOLVER_CUH
#define CUDSS_BATCH_SOLVER_CUH

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cudss.h>

#include <thrust/device_vector.h>

#include "../../cuda_utils.h"  // CudssContext, CudssDescriptor
#include "../../dtypes.hpp"     // cuda_real_type, CUDSS_R_TYPE

enum class CudssBatchMode { Uniform, BlockDiag, NonUniform };

// What cuDSS reports about the factors after an ANALYSIS (cudssDataGet).
// -1 = not queried yet, or not reported by this cuDSS build/mode. In Uniform
// mode lu_nnz is the fill-in of ONE system (the shared pattern), while the
// memory estimates cover the whole batch as cuDSS allocates it.
struct CudssFactorStats {
    long long lu_nnz                 = -1;  // CUDSS_DATA_LU_NNZ
    long long mem_device_permanent   = -1;  // CUDSS_DATA_MEMORY_ESTIMATES[0], bytes
    long long mem_device_peak        = -1;  //                            [1]
    long long mem_host_permanent     = -1;  //                            [2]
    long long mem_host_peak          = -1;  //                            [3]
};

inline const char* cudss_batch_mode_name(CudssBatchMode m)
{
    switch (m) {
        case CudssBatchMode::BlockDiag:  return "block_diag";
        case CudssBatchMode::NonUniform: return "non_uniform";
        default:                         return "uniform";
    }
}

// True when the variable is set to anything but "", "0", "false", "False".
inline bool cudss_env_flag(const char* name)
{
    const char* v = std::getenv(name);
    if (!v) return false;
    const std::string s(v);
    return !(s.empty() || s == "0" || s == "false" || s == "False" || s == "FALSE");
}

inline CudssBatchMode cudss_batch_mode_from_env()
{
    const bool blockdiag = cudss_env_flag("GPUSIM2GRID_USE_BLOCKDIAG");
    const bool batch     = cudss_env_flag("GPUSIM2GRID_USE_BATCH_MODE");
    if (blockdiag && batch)
        throw std::runtime_error(
            "[CudssBatchSolver] GPUSIM2GRID_USE_BLOCKDIAG and "
            "GPUSIM2GRID_USE_BATCH_MODE are mutually exclusive");
    if (blockdiag) return CudssBatchMode::BlockDiag;
    if (batch)     return CudssBatchMode::NonUniform;
    return CudssBatchMode::Uniform;
}

// Block-diagonal CSR skeleton of `batch` copies of (outer, inner), written on
// the device (defined in batch_pf_driver.cu, next to blockdiag_csr_kernel which
// it launches). Async on cs.
void cudss_blockdiag_csr(int n, int nnz, int batch,
                         const int* outer, const int* inner,
                         int* batch_outer, int* batch_inner,
                         cudaStream_t cs);

struct CudssBatchSolver {

    // =========================================================================
    // cuDSS batch context (owned)
    //
    // Declaration order (destruction is reverse):
    //   dss_b_ (last declared) → destroyed first
    //   dss_x_
    //   dss_A_
    //   batch storage referenced by the descriptors
    //   dss_  (first declared) → destroyed last
    // =========================================================================
    CudssContext    dss_;

    CudssBatchMode mode_ = CudssBatchMode::Uniform;
    int ubatch_ = 0, dim_J_ = 0, nnz_J_ = 0;
    cudaStream_t cs_ = nullptr;

    // Deferred / per-chunk ANALYSIS bookkeeping (see the file header).
    bool   analysis_pending_          = false;
    bool   factorized_since_analysis_ = false;
    int    n_analysis_                = 0;
    double t_analysis_total_ms_       = 0.;

    // BlockDiag: the ubatch·dim_J skeleton.
    thrust::device_vector<int> d_bd_outer_;
    thrust::device_vector<int> d_bd_inner_;

    // NonUniform: host size arrays (cuDSS keeps the pointers) and device
    // arrays of device pointers, one entry per batch member.
    std::vector<int>             h_nrows_;
    std::vector<int>             h_nnz_;
    std::vector<int>             h_ncols_dn_;   // all 1 (one right-hand side)
    thrust::device_vector<void*> d_ptr_outer_;
    thrust::device_vector<void*> d_ptr_inner_;
    thrust::device_vector<void*> d_ptr_values_;
    thrust::device_vector<void*> d_ptr_x_;
    thrust::device_vector<void*> d_ptr_b_;
    cuda_real_type*              values_base_ = nullptr;

    CudssDescriptor dss_A_;
    CudssDescriptor dss_x_;
    CudssDescriptor dss_b_;

    // Wall-clock ms of the handle/config/data creation inside initialize();
    // see context_init_ms() below.
    double t_context_init_ms_ = 0.;

    CudssFactorStats factor_stats_;

    // =========================================================================
    // initialize()
    //
    // Creates the cuDSS context, builds the three descriptors (in the batch
    // mode picked from the environment, see the file header), and runs the
    // one-time ANALYSIS phase (reordering + symbolic factorization).
    //
    // Parameters (all device pointers unless noted):
    //   ubatch_size       — number of systems in the batch
    //   dim_J             — Jacobian dimension (n_pvpq + n_pq)
    //   nnz_J             — number of non-zeros in J (per system)
    //   d_J_outer         — J row-pointer array  (single-system, size dim_J+1)
    //   d_J_inner         — J column-index array (single-system, size nnz_J)
    //   d_J_values_batch  — J values batch buffer (ubatch_size * nnz_J)
    //   d_F_batch         — F (RHS) batch buffer  (ubatch_size * dim_J)
    //   d_dx_batch        — dx (solution) batch buffer (ubatch_size * dim_J)
    //   cs                — CUDA stream on which cuDSS will be bound
    //   reordering_alg    — CUDSS_CONFIG_REORDERING_ALG choice (default: cuDSS's own default)
    //   matching_alg      — CUDSS_CONFIG_MATCHING_ALG choice (default: cuDSS's own default, matching off)
    //   pivot_epsilon_alg — CUDSS_CONFIG_PIVOT_EPSILON_ALG choice (default: cuDSS's own default)
    //
    // Async: ANALYSIS is launched on cs; the caller must synchronize cs after
    // this call (or before consuming any results).
    // =========================================================================
    void initialize(int ubatch_size, int dim_J, int nnz_J,
                    int*            d_J_outer,
                    int*            d_J_inner,
                    cuda_real_type* d_J_values_batch,
                    cuda_real_type* d_F_batch,
                    cuda_real_type* d_dx_batch,
                    cudaStream_t    cs,
                    ReorderingAlg   reordering_alg = ReorderingAlg::Default,
                    MatchingAlg     matching_alg = MatchingAlg::None,
                    PivotEpsilonAlg pivot_epsilon_alg = PivotEpsilonAlg::Default)
    {
        auto chk = [](cudssStatus_t s, const char* msg) {
            if (s != CUDSS_STATUS_SUCCESS)
                throw std::runtime_error(
                    std::string("[CudssBatchSolver::initialize] ") + msg
                    + ": cuDSS status=" + std::to_string(static_cast<int>(s)));
        };

        mode_   = cudss_batch_mode_from_env();
        cs_     = cs;
        ubatch_ = ubatch_size;
        dim_J_  = dim_J;
        nnz_J_  = nnz_J;
        announce_mode();

        // Both alternatives address the whole batch with 32-bit indices
        // (block-diagonal offsets; cuDSS caps a non-uniform batch's aggregate
        // rows/non-zeros at INT_MAX).
        if (mode_ != CudssBatchMode::Uniform
            && (static_cast<long long>(ubatch_size) * nnz_J > INT_MAX
                || static_cast<long long>(ubatch_size) * dim_J >= INT_MAX))
            throw std::runtime_error(
                std::string("[CudssBatchSolver::initialize] batch mode '")
                + cudss_batch_mode_name(mode_) + "': batch_size * nnz_J = "
                + std::to_string(static_cast<long long>(ubatch_size) * nnz_J)
                + " exceeds the 32-bit index limit; lower batch_size");

        // Context creation is timed separately from ANALYSIS below: on a
        // process' first cuDSS call it dlopens + JITs the backend (tens of ms,
        // independent of problem size), which would otherwise inflate what
        // callers read as symbolic-analysis cost. See context_init_ms().
        auto t_ctx_start = std::chrono::steady_clock::now();

        chk(cudssCreate(&dss_.handle),                "cudssCreate");
        chk(cudssSetStream(dss_.handle, cs),          "cudssSetStream");
        chk(cudssConfigCreate(&dss_.config),          "cudssConfigCreate");
        chk(cudssDataCreate(dss_.handle, &dss_.data), "cudssDataCreate");

        t_context_init_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_ctx_start).count();

        switch (mode_) {
        case CudssBatchMode::Uniform:
            dss_A_.create_csr(dim_J, nnz_J, d_J_outer, d_J_inner, d_J_values_batch);
            dss_x_.create_dn(dim_J, d_dx_batch);
            dss_b_.create_dn(dim_J, d_F_batch);
            chk(cudssConfigSet(dss_.config, CUDSS_CONFIG_UBATCH_SIZE,
                               &ubatch_size, sizeof(ubatch_size)), "cudssConfigSet");
            break;

        case CudssBatchMode::BlockDiag: {
            const long long n_big   = static_cast<long long>(ubatch_size) * dim_J;
            const long long nnz_big = static_cast<long long>(ubatch_size) * nnz_J;
            d_bd_outer_.resize(static_cast<size_t>(n_big + 1));
            d_bd_inner_.resize(static_cast<size_t>(nnz_big));
            cudss_blockdiag_csr(dim_J, nnz_J, ubatch_size, d_J_outer, d_J_inner,
                                thrust::raw_pointer_cast(d_bd_outer_.data()),
                                thrust::raw_pointer_cast(d_bd_inner_.data()), cs);
            dss_A_.create_csr(n_big, nnz_big,
                              thrust::raw_pointer_cast(d_bd_outer_.data()),
                              thrust::raw_pointer_cast(d_bd_inner_.data()),
                              d_J_values_batch);
            dss_x_.create_dn(n_big, d_dx_batch);
            dss_b_.create_dn(n_big, d_F_batch);
            break;
        }

        case CudssBatchMode::NonUniform: {
            const size_t B = static_cast<size_t>(ubatch_size);
            h_nrows_.assign(B, dim_J);
            h_nnz_.assign(B, nnz_J);
            h_ncols_dn_.assign(B, 1);

            std::vector<void*> h_outer(B, d_J_outer), h_inner(B, d_J_inner);
            std::vector<void*> h_x(B), h_b(B);
            for (size_t i = 0; i < B; ++i) {
                h_x[i] = d_dx_batch + i * static_cast<size_t>(dim_J);
                h_b[i] = d_F_batch  + i * static_cast<size_t>(dim_J);
            }
            d_ptr_outer_.assign(h_outer.begin(), h_outer.end());
            d_ptr_inner_.assign(h_inner.begin(), h_inner.end());
            d_ptr_x_.assign(h_x.begin(), h_x.end());
            d_ptr_b_.assign(h_b.begin(), h_b.end());
            upload_value_pointers(d_J_values_batch);

            chk(cudssMatrixCreateBatchCsr(
                    &dss_A_.desc, ubatch_size,
                    h_nrows_.data(), h_nrows_.data(), h_nnz_.data(),
                    raw_ptrs(d_ptr_outer_), nullptr,
                    raw_ptrs(d_ptr_inner_), raw_ptrs(d_ptr_values_),
                    CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_TYPE,
                    CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                "cudssMatrixCreateBatchCsr");
            chk(cudssMatrixCreateBatchDn(
                    &dss_x_.desc, ubatch_size,
                    h_nrows_.data(), h_ncols_dn_.data(), h_nrows_.data(),
                    raw_ptrs(d_ptr_x_), CUDSS_R_32I, CUDSS_R_TYPE,
                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateBatchDn(x)");
            chk(cudssMatrixCreateBatchDn(
                    &dss_b_.desc, ubatch_size,
                    h_nrows_.data(), h_ncols_dn_.data(), h_nrows_.data(),
                    raw_ptrs(d_ptr_b_), CUDSS_R_32I, CUDSS_R_TYPE,
                    CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateBatchDn(b)");
            break;
        }
        }

        dss_.set_reordering_alg(reordering_alg);
        dss_.set_matching_alg(matching_alg);
        dss_.set_pivot_epsilon_alg(pivot_epsilon_alg);

        if (mode_ == CudssBatchMode::Uniform) {
            dss_.analyze(dss_A_, dss_x_, dss_b_);
            n_analysis_ = 1;
        } else {
            analysis_pending_ = true;
        }
    }

    // =========================================================================
    // begin_chunk()          — BlockDiag / NonUniform: mark ANALYSIS pending.
    // prepare_factorization() — run a pending ANALYSIS (host-synchronized,
    //                          wall-clock timed into analysis_ms()).
    // =========================================================================
    void begin_chunk()
    {
        if (mode_ != CudssBatchMode::Uniform) analysis_pending_ = true;
    }

    void prepare_factorization()
    {
        if (!analysis_pending_) return;
        cudaStreamSynchronize(cs_);
        auto t0 = std::chrono::steady_clock::now();
        dss_.analyze(dss_A_, dss_x_, dss_b_);
        cudaStreamSynchronize(cs_);
        t_analysis_total_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        ++n_analysis_;
        analysis_pending_          = false;
        factorized_since_analysis_ = false;
        refresh_factor_stats();
    }

    // =========================================================================
    // refresh_factor_stats() — query the fill-in (LU non-zeros) and memory
    // estimates of the last ANALYSIS. Host-synchronizes the stream (the values
    // are only final once ANALYSIS has run); call it outside timed regions.
    // Uniform mode: the driver calls it after initialize()'s ANALYSIS; the
    // other modes refresh it after every per-chunk ANALYSIS (last one wins).
    // A query cuDSS rejects leaves its field at -1.
    // =========================================================================
    void refresh_factor_stats()
    {
        if (!dss_.handle || !dss_.data) return;
        cudaStreamSynchronize(cs_);
        size_t written = 0;
        int64_t lu_nnz = -1;
        if (cudssDataGet(dss_.handle, dss_.data, CUDSS_DATA_LU_NNZ,
                         &lu_nnz, sizeof(lu_nnz), &written) == CUDSS_STATUS_SUCCESS)
            factor_stats_.lu_nnz = static_cast<long long>(lu_nnz);
        int64_t mem[16];
        for (auto& v : mem) v = -1;
        if (cudssDataGet(dss_.handle, dss_.data, CUDSS_DATA_MEMORY_ESTIMATES,
                         mem, sizeof(mem), &written) == CUDSS_STATUS_SUCCESS) {
            factor_stats_.mem_device_permanent = static_cast<long long>(mem[0]);
            factor_stats_.mem_device_peak      = static_cast<long long>(mem[1]);
            factor_stats_.mem_host_permanent   = static_cast<long long>(mem[2]);
            factor_stats_.mem_host_peak        = static_cast<long long>(mem[3]);
        }
    }

    const CudssFactorStats& factor_stats() const { return factor_stats_; }

    // Cumulative wall ms of the ANALYSIS runs done by prepare_factorization()
    // (0 in Uniform mode, whose single ANALYSIS the driver times itself).
    double analysis_ms() const { return t_analysis_total_ms_; }
    int    n_analysis()  const { return n_analysis_; }

    // =========================================================================
    // set_values()
    //
    // Notifies cuDSS of the current J values pointer.  CPU call — no GPU work
    // is launched (NonUniform re-uploads its pointer array only when the base
    // pointer actually changed, which no current caller does).  Must be called
    // before the first factor() or refactor() when new J values have been
    // written into d_J_values_batch.
    // =========================================================================
    void set_values(cuda_real_type* d_J_values_batch)
    {
        if (mode_ != CudssBatchMode::NonUniform) {
            dss_A_.set_values(d_J_values_batch);
            return;
        }
        if (d_J_values_batch == values_base_) return;
        upload_value_pointers(d_J_values_batch);
        cudssStatus_t s = cudssMatrixSetBatchValues(dss_A_.desc, raw_ptrs(d_ptr_values_));
        if (s != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(
                "[CudssBatchSolver::set_values] cudssMatrixSetBatchValues: cuDSS status="
                + std::to_string(static_cast<int>(s)));
    }

    // =========================================================================
    // factor()   — CUDSS_PHASE_FACTORIZATION  (first numeric factorization)
    // refactor() — CUDSS_PHASE_REFACTORIZATION (subsequent refactorizations)
    // solve()    — CUDSS_PHASE_SOLVE
    //
    // All three are async on the stream bound at initialize() time.
    // =========================================================================
    // Wall-clock ms spent creating the cuDSS handle/config/data in
    // initialize() — pure library/context setup, no ANALYSIS. Dominated by
    // first-touch dlopen + JIT on the first cuDSS use in the process, ~0
    // afterwards.
    double context_init_ms() const { return t_context_init_ms_; }

    CudssBatchMode mode() const { return mode_; }

    void factor()
    {
        prepare_factorization();
        dss_.factorize(dss_A_, dss_x_, dss_b_);
        factorized_since_analysis_ = true;
    }
    void refactor()
    {
        prepare_factorization();
        if (factorized_since_analysis_)
            dss_.refactorize(dss_A_, dss_x_, dss_b_);
        else
            dss_.factorize(dss_A_, dss_x_, dss_b_);
        factorized_since_analysis_ = true;
    }
    void solve()    { dss_.solve      (dss_A_, dss_x_, dss_b_); }

    // Non-copyable, non-movable (cuDSS descriptors hold raw device pointers).
    CudssBatchSolver() = default;
    CudssBatchSolver(const CudssBatchSolver&)            = delete;
    CudssBatchSolver& operator=(const CudssBatchSolver&) = delete;
    CudssBatchSolver(CudssBatchSolver&&)                 = delete;
    CudssBatchSolver& operator=(CudssBatchSolver&&)      = delete;

private:
    static const void* const* raw_ptrs(thrust::device_vector<void*>& v)
    {
        return static_cast<const void* const*>(
            static_cast<void*>(thrust::raw_pointer_cast(v.data())));
    }

    void upload_value_pointers(cuda_real_type* base)
    {
        std::vector<void*> h(static_cast<size_t>(ubatch_));
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = base + i * static_cast<size_t>(nnz_J_);
        d_ptr_values_.assign(h.begin(), h.end());
        values_base_ = base;
    }

    void announce_mode() const
    {
        if (mode_ == CudssBatchMode::Uniform) return;
        static bool announced[3] = {false, false, false};
        bool& done = announced[static_cast<int>(mode_)];
        if (done) return;
        done = true;
        std::fprintf(stderr,
                     "[gpusim2grid] cuDSS batch mode: %s (set by environment; "
                     "experimental)\n", cudss_batch_mode_name(mode_));
    }
};

#endif // CUDSS_BATCH_SOLVER_CUH
