// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// =============================================================================
// injection_sweep.cu
//
// Public C++ entry point for the batched-injection power flow sweep.
// Mirrors run_contingency_analysis_gpu in structure but instantiates
// BatchPfDriver<InjectionBatch> with a dense per-scenario Sbus.
// =============================================================================

#include "injection_sweep.hpp"
#include "acpf_nr_state.cuh"
#include "contingency/batch_pf_driver.cuh"
#include "contingency/batch_sources/injection_batch.cuh"
#include "cu_complex_utils.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
inline double inj_ms_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}
}

BatchTimings run_injection_sweep_gpu(
    CplxVect&                                   V_results_out,
    RealVect&                                   residuals_out,
    const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
    Eigen::Ref<const CplxVect>                  Vinit,
    Eigen::Ref<const CplxVect>                  Sbus_base,
    Eigen::Ref<const Eigen::VectorXi>           slack_ids,
    Eigen::Ref<const RealVect>                  slack_weights,
    Eigen::Ref<const Eigen::VectorXi>           pv,
    Eigen::Ref<const Eigen::VectorXi>           pq,
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p_mw,
    Eigen::Ref<const Eigen::Matrix<eigen_real_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> q_mvar,
    double                                      sn_mva,
    int                                         batch_size,
    int                                         nb_iter,
    int                                         max_iter_base,
    double                                      tol_base,
    ContingencySolverType                       strategy_type,
    int                                         refactor_period)
{
    (void)slack_ids;
    (void)slack_weights;

    // -------------------------------------------------------------------------
    // Shape validation
    // -------------------------------------------------------------------------
    const int n_bus = static_cast<int>(Sbus_base.size());
    const int n_scen = static_cast<int>(p_mw.rows());

    if (q_mvar.rows() != p_mw.rows() || q_mvar.cols() != p_mw.cols()) {
        throw std::runtime_error(
            "run_injection_sweep_gpu: p_mw and q_mvar must have the same shape");
    }
    if (p_mw.cols() != n_bus) {
        throw std::runtime_error(
            "run_injection_sweep_gpu: p_mw / q_mvar second dim must equal n_bus");
    }
    if (n_scen <= 0) {
        throw std::runtime_error(
            "run_injection_sweep_gpu: n_scenarios must be > 0");
    }
    if (sn_mva <= 0.0) {
        throw std::runtime_error(
            "run_injection_sweep_gpu: sn_mva must be > 0");
    }
    if (batch_size <= 0 || nb_iter <= 0) {
        throw std::runtime_error(
            "run_injection_sweep_gpu: batch_size and nb_iter must be > 0");
    }

    // -------------------------------------------------------------------------
    // Build host-side per-unit complex Sbus_all  ((n_scenarios × n_bus) row-major)
    //   Sbus_all[s, b] = (p_mw(s, b) + 1j * q_mvar(s, b)) / sn_mva
    // -------------------------------------------------------------------------
    auto t_host_start = std::chrono::steady_clock::now();
    std::vector<cudaComplexType> h_Sbus_all(
        static_cast<size_t>(n_scen) * static_cast<size_t>(n_bus));
    const double inv_sn = 1.0 / sn_mva;
    for (int s = 0; s < n_scen; ++s) {
        for (int b = 0; b < n_bus; ++b) {
            const double p_pu = p_mw(s, b)   * inv_sn;
            const double q_pu = q_mvar(s, b) * inv_sn;
            h_Sbus_all[static_cast<size_t>(s) * n_bus + b] =
                CudaFunHelper::my_make_cuComplex(
                    static_cast<cuda_real_type>(p_pu),
                    static_cast<cuda_real_type>(q_pu));
        }
    }
    const double t_host_ms = inj_ms_since(t_host_start);

    // -------------------------------------------------------------------------
    // RowMajor Ybus (required by AcPfNrState's J-structure builder)
    // -------------------------------------------------------------------------
    Eigen::SparseMatrix<eigen_cplx_type, Eigen::RowMajor> Ybus_rm = Ybus;

    // -------------------------------------------------------------------------
    // Base-case NR (using the base Sbus_base — same convention as the rest of
    // the stack: per-unit complex).
    // -------------------------------------------------------------------------
    auto t_base_start = std::chrono::steady_clock::now();
    AcPfNrState base_state(Ybus, Vinit, Sbus_base, pv, pq,
                           max_iter_base,
                           static_cast<eigen_real_type>(tol_base));
    const double t_base_case_ms = inj_ms_since(t_base_start);

    // -------------------------------------------------------------------------
    // Compute the effective per-chunk batch size (same rebalancing rule as the
    // contingency entry point: balance the last chunk so no chunk is wastefully
    // small).
    // -------------------------------------------------------------------------
    const size_t n_elem    = static_cast<size_t>(n_scen);
    const size_t max_bs    = static_cast<size_t>(batch_size);
    const size_t n_chunks  = (n_elem + max_bs - 1) / max_bs;
    const int    used_bs   = static_cast<int>(
        n_chunks > 0 ? (n_elem + n_chunks - 1) / n_chunks : n_elem);

    // -------------------------------------------------------------------------
    // Construct the InjectionBatch source and the BatchPfDriver
    // -------------------------------------------------------------------------
    InjectionBatch source(std::move(h_Sbus_all), n_scen, t_host_ms);

    BatchPfDriver<InjectionBatch> solver(
        base_state,
        std::move(source),
        n_scen,
        Ybus_rm.outerIndexPtr(),
        Ybus_rm.innerIndexPtr(),
        used_bs,
        nb_iter,
        strategy_type,
        refactor_period);

    BatchTimings timings = solver.solve();
    solver.copy_results_to_host(V_results_out, residuals_out);

    timings.t_base_case_ms = t_base_case_ms;
    timings.n_disconnected = 0;
    return timings;
}