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
        int                                         device = -1
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
};

#endif  // ACPFNR_H