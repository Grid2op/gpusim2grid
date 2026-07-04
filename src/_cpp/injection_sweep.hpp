// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INJECTION_SWEEP_HPP
#define INJECTION_SWEEP_HPP

// =============================================================================
// injection_sweep.hpp  —  Public C++ entry point for the batched-injection
// power flow sweep.
//
// Inputs
// ------
// Ybus, Vinit, Sbus_base, pv, pq, slack_ids, slack_weights
//     Same convention as run_contingency_analysis_gpu — used to build the
//     base-case NR state (AcPfNrState).  Sbus_base is the per-unit complex
//     injection vector used during base-case convergence.
//
// p_mw, q_mvar  (host buffers, n_scenarios × n_bus, row-major float64)
//     Per-scenario active and reactive power in PHYSICAL UNITS (MW / MVAr).
//     Internally divided by sn_mva to form per-unit complex Sbus for the
//     batched NR loop (consistent with the existing Sbus convention used by
//     the rest of the stack).
//
// sn_mva  (system base apparent power)
//     Used for the MW/MVAr → pu conversion.
//
// batch_size, nb_iter, max_iter_base, tol_base
//     Same semantics as in the contingency analysis entry point.
//
// Outputs
// -------
// V_results_out  : (n_scenarios * n_bus,) complex per-unit voltages, row-major.
// residuals_out  : (n_scenarios,) ‖F‖∞ residual per scenario after nb_iter NR.
// Return value   : BatchTimings — same struct used by the contingency
//                  entry point.  For this feature:
//                     t_patch_Ybus = 0  (no per-chunk Ybus mutation)
//                     n_disconnected = 0
//                     t_tile_Ybus  = 0 per chunk (one-time tile during
//                                   construction; folded into t_analysis_ms).
// =============================================================================

#include "dtypes.hpp"
#include "timing_utils.hpp"
#include "contingency_analysis_helper.hpp"   // ContingencySolverType

#include "Eigen/Core"
#include "Eigen/SparseCore"

BatchTimings run_injection_sweep_gpu(
    CplxVect&                                   V_results_out,   // (n_scen * n_bus,)
    RealVect&                                   residuals_out,   // (n_scen,)
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
    int                                         max_iter_base   = 10,
    double                                      tol_base        = 1e-6,
    ContingencySolverType                       strategy_type   = ContingencySolverType::DirectRefactorEvery,
    int                                         refactor_period = 1
);

#endif // INJECTION_SWEEP_HPP