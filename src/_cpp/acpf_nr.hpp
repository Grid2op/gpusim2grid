// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef ACPFNR_H
#define ACPFNR_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"

#include "dtypes.hpp"
#include "timing_utils.hpp"

#include <memory>
#include <iostream>
#include <vector>

struct AcPfNrState;  // complete type in acpf_nr_state.cuh (nvcc only)
struct LedgerData;   // complete type in ledger_data.hpp (augmented-J description)

AcPfTimings acpf_nr_gpu(
    Eigen::Ref<CplxVect> out,
    const Eigen::SparseMatrix<eigen_cplx_type> & ac_Ybus,
    Eigen::Ref<const CplxVect> V,
    Eigen::Ref<const CplxVect> Sbus,
    Eigen::Ref<const Eigen::VectorXi> slack_ids,
    Eigen::Ref<const RealVect> slack_weights,
    Eigen::Ref<const Eigen::VectorXi> pv,
    Eigen::Ref<const Eigen::VectorXi> pq,
    int max_iter,
    eigen_real_type tol,
    int nb_solve,
    int device = -1
);

// =============================================================================
// AcPfNrSession — stateful wrapper around AcPfNrState that keeps the device
// voltage vector alive so it can be exported via DLPack without a host copy.
// =============================================================================
struct AcPfNrSession {
    std::shared_ptr<AcPfNrState> state_;

    AcPfNrSession(
        const Eigen::SparseMatrix<eigen_cplx_type>& Ybus,
        Eigen::Ref<const CplxVect>                  Vinit,
        Eigen::Ref<const CplxVect>                  Sbus,
        Eigen::Ref<const Eigen::VectorXi>           slack_ids,
        Eigen::Ref<const RealVect>                  slack_weights,
        Eigen::Ref<const Eigen::VectorXi>           pv,
        Eigen::Ref<const Eigen::VectorXi>           pq,
        int                                         max_iter,
        eigen_real_type                             tol,
        int                                         device = -1,
        const LedgerData*                           ledger = nullptr,
        bool                                         presolved_v = false
    );

    AcPfTimings timings() const;

    // Copy d_V from device to a host Eigen vector (one D→H transfer).
    CplxVect get_v() const;

    // Accessors for adjoint-solve metadata (used by Python backward pass).
    int n_pvpq() const;
    int n_pq()   const;
    int dim_J()  const;
    std::vector<int> pvpq() const;  // D→H copy of sorted pvpq indices
    std::vector<int> pq()   const;  // D→H copy of pq indices

    // Bus-indexed NRLedger maps (host, size n_bus, -1 sentinel when the bus
    // owns no such row/col). Used by the differentiable adjoint path to
    // project/scatter gradients per-bus instead of assuming the trivial
    // bare-system pvpq/pq positional layout — see ledger_data.hpp.
    std::vector<int> p_row_of_bus()     const;
    std::vector<int> q_row_of_bus()     const;
    std::vector<int> theta_col_of_bus() const;
    std::vector<int> vm_col_of_bus()    const;
};

#endif  // ACPFNR_H