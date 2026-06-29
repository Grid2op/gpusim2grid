// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef DTYPES_H
#define DTYPES_H

#include <complex>
#include "Eigen/Core"

typedef float real_type;  // type for real numbers: can be changed if installed from source


// ---------------------------------------------------------------------------
// Floating-point precision
//
// Controlled at build time by the GPUSIM2GRID_REAL_FLOAT preprocessor macro,
// which is set by CMake when -DUSE_FLOAT_PRECISION=ON is passed (or when the
// CUDA_REAL_FLOAT environment variable is set at configure time).
//
//   FP64 (default):  cuda_real_type = double
//   FP32 (opt-in):   cuda_real_type = float
// ---------------------------------------------------------------------------
#ifdef GPUSIM2GRID_REAL_FLOAT
typedef float cuda_real_type;
#else
typedef double cuda_real_type;
#endif
typedef double eigen_real_type;  // type for real number eigen side

typedef std::complex<eigen_real_type> eigen_cplx_type;  // type for complex number

typedef Eigen::Matrix<eigen_real_type, Eigen::Dynamic, 1> EigenPythonNumType;  // Eigen::VectorXd
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVect;
typedef Eigen::Matrix<eigen_real_type, Eigen::Dynamic, 1> RealVect;
typedef Eigen::Matrix<eigen_cplx_type, Eigen::Dynamic, 1> CplxVect;

#endif  // DTYPES_H