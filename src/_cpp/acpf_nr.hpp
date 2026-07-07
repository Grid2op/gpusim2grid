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
#include "reordering_alg.hpp"

#include <memory>
#include <iostream>
#include <tuple>
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
        bool                                         presolved_v = false,
        // DEBUG ONLY -- see AcPfNrState's own doc. No-op unless presolved_v=true.
        bool                                         diag_stop_before_state_correction = false,
        // CUDSS_CONFIG_REORDERING_ALG choice; see AcPfNrState's own doc.
        ReorderingAlg                                reordering_alg = ReorderingAlg::Default
    );

    AcPfTimings timings() const;

    // Copy d_V from device to a host Eigen vector (one D→H transfer).
    CplxVect get_v() const;

    // The mismatch RHS gpusim2grid actually computed on the GPU (D->H copy of
    // d_F), widened to double. In the normal flow this is F after any
    // has_ext_state correction (should be ~0 at convergence); when the
    // session was built with diag_stop_before_state_correction=true, it is
    // the RAW F(Vinit, state=0), letting an external solver (e.g. scipy) redo
    // the correction step independently of cuDSS on the exact same data --
    // see the VC+MultiSlack singular-solve investigation.
    std::vector<double> get_F() const;

    // DEBUG: solve J*dx = rhs using gpusim2grid's OWN cuDSS context (the same
    // handle/config/data and the SAME factorization of J already computed at
    // construction -- J itself is NOT re-uploaded or re-factorized here).
    // Uploads `rhs` into d_F (overwriting whatever get_F() would have
    // returned), calls dss.solve(dss_A, dss_x, dss_b), and returns dx (D->H
    // copy of d_dx). Meant to be compared directly against
    // scipy.sparse.linalg.spsolve(get_J(), rhs) on the exact same (J, rhs) --
    // see repro_cudss_bug.py / the VC+MultiSlack singular-solve investigation.
    std::vector<double> solve_cudss(const std::vector<double>& rhs) const;

    // DEBUG: close the loop on solve_cudss()/scipy -- compute J*dx - rhs (a
    // plain host-side CSR matvec against get_J()'s own sparsity/values, no
    // GPU/cuSPARSE involved) and return the residual vector. A correct dx
    // (e.g. from scipy) gives ~0 everywhere; cuDSS's own (broken) dx on this
    // system does not. dx must have length dim_J.
    std::vector<double> residual(const std::vector<double>& dx, const std::vector<double>& rhs) const;

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

    // The augmented Jacobian gpusim2grid actually built and factorized on the
    // GPU (D→H copy), in the SAME RowMajor CSR convention as the rest of this
    // codebase: (indptr, indices, data), indptr size dim_J+1, indices/data
    // size nnz_J. Reconstruct in Python with
    // ``scipy.sparse.csr_matrix((data, indices, indptr), shape=(dim_J, dim_J))``.
    // Values are widened to double regardless of the FP32/FP64 build.
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>> get_J() const;
};

#endif  // ACPFNR_H