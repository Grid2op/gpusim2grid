#ifndef CUCOMPLEX_UTIL_H
#define CUCOMPLEX_UTIL_H

#include "dtypes.hpp"
#include <cuComplex.h>
#include <cudss.h>

// which type of complex value to use in cuda
// there might be a simpler things to do here ?
template<bool>
struct CudaIsDoubleComplexType;

template<> struct CudaIsDoubleComplexType<true>{
    typedef double _real_type;
    const static cudaDataType_t  CUDA_C_TYPE  = CUDA_C_64F;
    const static cudssDataType_t CUDSS_R_TYPE = CUDSS_R_64F;
    typedef std::complex<double> _cplx_type;
    typedef cuDoubleComplex complexType;

    __host__ __device__ static complexType my_cuConj(complexType x){return cuConj(x);}
    __host__ __device__ static _real_type my_cuCreal(complexType x){return cuCreal(x);}
    __host__ __device__ static _real_type my_cuCimag(complexType x){return cuCimag(x);}
    __host__ __device__ static _real_type my_cuCabs(complexType x){return cuCabs(x);}

    __host__ __device__ static complexType my_cuCmul(complexType x, complexType y){return cuCmul(x, y);}
    __host__ __device__ static complexType my_cuCadd(complexType x, complexType y){return cuCadd(x, y);}
    __host__ __device__ static complexType my_cuCsub(complexType x, complexType y){return cuCsub(x, y);}
    __host__ __device__ static complexType my_cuCdiv(complexType x, complexType y){return cuCdiv(x, y);}
    __host__ __device__ static complexType my_make_cuComplex(_real_type x, _real_type y){return make_cuDoubleComplex(x, y);}

    __host__ __device__ static _real_type my_atan2(_real_type x, _real_type y) {return atan2(x, y);}
    __host__ __device__ static _real_type my_cos(_real_type x) {return cos(x);}
    __host__ __device__ static _real_type my_sin(_real_type x) {return sin(x);}
    // __host__ __device__ static complexType my_make_cuComplex(_cplx_type z){return make_cuDoubleComplex(z.real(), z.imag());}
};

template<> struct CudaIsDoubleComplexType<false>{
    typedef float _real_type;
    const static cudaDataType_t  CUDA_C_TYPE  = CUDA_C_32F;
    const static cudssDataType_t CUDSS_R_TYPE = CUDSS_R_32F;
    typedef std::complex<float> _cplx_type;
    typedef cuFloatComplex complexType;

    __host__ __device__ static complexType my_cuConj(complexType x){return cuConjf(x);}
    __host__ __device__ static _real_type my_cuCreal(complexType x){return cuCrealf(x);}
    __host__ __device__ static _real_type my_cuCimag(complexType x){return cuCimagf(x);}
    __host__ __device__ static _real_type my_cuCabs(complexType x){return cuCabsf(x);}

    __host__ __device__ static complexType my_cuCmul(complexType x, complexType y){return cuCmulf(x, y);}
    __host__ __device__ static complexType my_cuCadd(complexType x, complexType y){return cuCaddf(x, y);}
    __host__ __device__ static complexType my_cuCsub(complexType x, complexType y){return cuCsubf(x, y);}
    __host__ __device__ static complexType my_cuCdiv(complexType x, complexType y){return cuCdivf(x, y);}
    __host__ __device__ static complexType my_make_cuComplex(_real_type x, _real_type y){return make_cuFloatComplex(x, y);}
    __host__ __device__ static _real_type my_atan2(_real_type x, _real_type y) {return atan2f(x, y);}
    __host__ __device__ static _real_type my_cos(_real_type x) {return cosf(x);}
    __host__ __device__ static _real_type my_sin(_real_type x) {return sinf(x);}
    // __host__ __device__ static complexType my_make_cuComplex(_cplx_type z){return make_cuFloatComplex(z.real(), z.imag());}
};

using CudaFunHelper = CudaIsDoubleComplexType<std::is_same<double, cuda_real_type>::value>;
using cudaComplexType = CudaFunHelper::complexType;
const static cudaDataType_t  CUDA_C_TYPE  = CudaFunHelper::CUDA_C_TYPE;
const static cudssDataType_t CUDSS_R_TYPE = CudaFunHelper::CUDSS_R_TYPE;


// helper for functions

template<typename T>
__host__ __device__ __inline__ T my_cuConj(T x) = delete;

template<typename T>
__host__ __device__ __inline__ T my_cuCMul(T x, T y) = delete;

template<typename T>
__host__ __device__ __inline__ T my_cuCAdd(T x, T y) = delete;

template<typename T>
__host__ __device__ __inline__ T my_cuCSub(T x, T y) = delete;

template<typename cudaCplxType, typename cudaRealType>
__host__ __device__ __inline__ cudaCplxType my_make_cuComplex(cudaRealType x, cudaRealType y) = delete;

template<typename cudaCplxType, typename EigencplxType>
__host__ __device__ __inline__ cudaCplxType my_make_cuComplex_eigen(EigencplxType x) = delete;


const static __device__ cudaComplexType d_cplx_one = {
    static_cast<cuda_real_type>(1.),
    static_cast<cuda_real_type>(0.)
};
const static __device__ cudaComplexType d_cplx_zero = {
    static_cast<cuda_real_type>(0.),
    static_cast<cuda_real_type>(0.)
};
const static __host__ cudaComplexType h_cplx_one = {
    static_cast<cuda_real_type>(1.),
    static_cast<cuda_real_type>(0.)
};
const static __host__ cudaComplexType h_cplx_zero = {
    static_cast<cuda_real_type>(0.),
    static_cast<cuda_real_type>(0.)
};

// definition for cuDoubleComplex
// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_cuConj(cuDoubleComplex x){return cuConj(x);}

// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_cuCMul(cuDoubleComplex x, cuDoubleComplex y){return cuCmul(x, y);}

// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_cuCAdd(cuDoubleComplex x, cuDoubleComplex y){return cuCadd(x, y);}

// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_cuCSub(cuDoubleComplex x, cuDoubleComplex y){return cuCsub(x, y);}

// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_make_cuComplex(double x, double y){return make_cuDoubleComplex(x, y);}

// template<>
// __host__ __device__ __inline__ cuDoubleComplex my_make_cuComplex_eigen(const std::complex<double> x){return make_cuDoubleComplex(x.real(), x.imag());}


// definition for cuFloatComplex
// template<>
// __host__ __device__ __inline__ cuFloatComplex my_cuConj(cuFloatComplex x){return cuConjf(x);}

// template<>
// __host__ __device__ __inline__ cuFloatComplex my_cuCMul(cuFloatComplex x, cuFloatComplex y){return cuCmulf(x, y);}

// template<>
// __host__ __device__ __inline__ cuFloatComplex my_cuCAdd(cuFloatComplex x, cuFloatComplex y){return cuCaddf(x, y);}

// template<>
// __host__ __device__ __inline__ cuFloatComplex my_cuCSub(cuFloatComplex x, cuFloatComplex y){return cuCsubf(x, y);}

// template<>
// __host__ __device__ __inline__ cuFloatComplex my_make_cuComplex(float x, float y){return make_cuFloatComplex(x, y);}

// template<>
// __host__ __device__ __inline__ cuFloatComplex my_make_cuComplex_eigen(const std::complex<float> x){return make_cuFloatComplex(x.real(), x.imag());}

# endif  // CUCOMPLEX_UTIL_H
