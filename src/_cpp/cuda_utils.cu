// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "cuda_utils.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"


void _init_Ybus_device(
    thrust::device_vector<int>& d_cst_Ybus_outer,
    thrust::device_vector<int>& d_cst_Ybus_inner,
    thrust::device_vector<cudaComplexType>& d_cst_Ybus_values,
    const Eigen::SparseMatrix<eigen_cplx_type> & Ybus
)
{
    const int num_rows = Ybus.rows();
    const int num_cols = Ybus.cols();
    const int nnz = Ybus.nonZeros();
    const int * p_Ybus_outer = Ybus.outerIndexPtr();
    const int * p_Ybus_inner = Ybus.innerIndexPtr();

    d_cst_Ybus_outer = thrust::device_vector<int>(num_rows + 1);
    thrust::copy(p_Ybus_outer, p_Ybus_outer + (num_rows + 1), d_cst_Ybus_outer.begin());

    d_cst_Ybus_inner = thrust::device_vector<int>(nnz);
    thrust::copy(p_Ybus_inner, p_Ybus_inner + nnz, d_cst_Ybus_inner.begin());

    _init_vector_device_cplx(d_cst_Ybus_values,  CplxVect::Map(Ybus.valuePtr(), nnz));
}