#ifndef MY_CUDA_UTILS_H
#define MY_CUDA_UTILS_H

#include "cu_complex_utils.h"
#include "timing_utils.hpp"
#include "reordering_alg.hpp"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <stdexcept>          // std::runtime_error
#include <string>             // std::string

// generic cuda / cudss
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cudss.h>

// for cuda memory management
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>


// Eigen
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseLU"


typedef thrust::device_vector<int> device_vector_int;
typedef thrust::device_vector<cudaComplexType> device_vector_cplx;


void _init_Ybus_device(
    thrust::device_vector<int>& d_cst_Ybus_outer,
    thrust::device_vector<int>& d_cst_Ybus_inner,
    thrust::device_vector<cudaComplexType>& d_cst_Ybus_values,
    const Eigen::SparseMatrix<eigen_cplx_type> & Ybus
);

// template<class cudaType, class EigenType, class FunctorTransform>
// void _init_vector_device(
//     thrust::device_vector<cudaType>& d_vect,
//     const EigenType & host_vect,
//     const FunctorTransform & functor_tranform
// )
// {
//     const int size = host_vect.size();
//     d_vect = thrust::device_vector<cudaType>(size);
//     thrust::transform(thrust::device, &host_vect(0), &host_vect(0) + size,  d_vect.begin(), functor_tranform);
// }

// static void _init_vector_device_cplx(
//     thrust::device_vector<cudaComplexType>& d_vect,
//     Eigen::Ref<const CplxVect> host_vect
// )
// {
//     const int size = host_vect.size();
//     Eigen::Matrix<cudaComplexType, Eigen::Dynamic, 1> convert_to_cuda_type(size);
//     for(size_t i = 0; i < size; i++)
//     {
//         eigen_cplx_type z = host_vect(i);
//         convert_to_cuda_type(i) = CudaFunHelper::my_make_cuComplex(
//             static_cast<cuda_real_type>(z.real()),
//             static_cast<cuda_real_type>(z.imag())
//         );
//     }
//     auto tmp_vect = thrust::host_vector<cudaComplexType>(convert_to_cuda_type.begin(), convert_to_cuda_type.end());
//     d_vect = tmp_vect;
// }


// static void _init_vector_device_real(
//     thrust::device_vector<cuda_real_type>& d_vect,
//     Eigen::Ref<const CplxVect> host_vect
// )
// {
//     const int size = host_vect.size();
//     Eigen::Matrix<cuda_real_type, Eigen::Dynamic, 1> convert_to_cuda_type(size);
//     for(size_t i = 0; i < size; i++)
//     {
//         eigen_cplx_type z = host_vect(i);
//         convert_to_cuda_type(i) = static_cast<cuda_real_type>(z.real());
//     }
//     auto tmp_vect = thrust::host_vector<cuda_real_type>(convert_to_cuda_type.begin(), convert_to_cuda_type.end());
//     d_vect = tmp_vect;
// }

// struct EigenComplexToCudaComplex
// {
//     EigenComplexToCudaComplex(){}
//     __host__ __device__
//     cudaComplexType operator()(const eigen_real_type & r, const  eigen_real_type & i) const {
//         return  CudaFunHelper::my_make_cuComplex(static_cast<cuda_real_type>(r), static_cast<cuda_real_type>(i));
//     }
// };

// struct EigenRealToCudaComplex
// {
//     EigenRealToCudaComplex(){}
//     __host__ __device__
//     cudaComplexType operator()(const eigen_real_type & r) const {
//         return  CudaFunHelper::my_make_cuComplex(static_cast<cuda_real_type>(r), static_cast<cuda_real_type>(0.));
//     }
// };

// -----------------------------------------------------------------
// Upload functors: Eigen complex host → CUDA device types
// -----------------------------------------------------------------
struct EigenCplxToCudaCplx {
    __host__ cudaComplexType operator()(const eigen_cplx_type& z) const {
        return CudaFunHelper::my_make_cuComplex(
            static_cast<cuda_real_type>(z.real()),
            static_cast<cuda_real_type>(z.imag()));
    }
};

struct EigenCplxToCudaReal {
    __host__ cuda_real_type operator()(const eigen_cplx_type& z) const {
        return static_cast<cuda_real_type>(z.real());
    }
};

// -----------------------------------------------------------------
// Unified helper: transform n host elements and upload to device.
// No intermediate Eigen::Matrix or thrust::host_vector needed —
// the transform is fused into the H2D copy by thrust.
//
// Precondition: h_vec is contiguous (inner stride == 1), which is
// always true for CplxVect (Eigen column-vector dense storage).
// -----------------------------------------------------------------
template <typename DeviceT, typename Functor>
static void _upload_eigen_cplx(
    thrust::device_vector<DeviceT>& d_vec,
    Eigen::Ref<const CplxVect>      h_vec,
    Functor                          f,
    cudaStream_t                     stream = nullptr)   // default = stream 0
{
    const int n = h_vec.size();
    d_vec.resize(n);
    thrust::copy(
        thrust::cuda::par.on(stream),                    // <-- bind stream
        thrust::make_transform_iterator(h_vec.data(),     f),
        thrust::make_transform_iterator(h_vec.data() + n, f),
        d_vec.begin());
}

static void _init_vector_device_cplx(
    thrust::device_vector<cudaComplexType>& d_vec,
    Eigen::Ref<const CplxVect>              h_vec,
    cudaStream_t                            stream = nullptr)
{
    _upload_eigen_cplx(d_vec, h_vec, EigenCplxToCudaCplx(), stream);
}

static void _init_vector_device_real(
    thrust::device_vector<cuda_real_type>& d_vec,
    Eigen::Ref<const CplxVect>             h_vec,
    cudaStream_t                           stream = nullptr)
{
    _upload_eigen_cplx(d_vec, h_vec, EigenCplxToCudaReal(), stream);
}

// ---------------------------------------------------------------------------
// RAII wrapper for a single cudss descriptor (cudssMatrix_t).
// ---------------------------------------------------------------------------
struct CudssDescriptor {
    cudssMatrix_t desc = nullptr;

    ~CudssDescriptor() {
        if (desc) cudssMatrixDestroy(desc);
    }

    CudssDescriptor() = default;
    CudssDescriptor(const CudssDescriptor&) = delete;
    CudssDescriptor& operator=(const CudssDescriptor&) = delete;

    CudssDescriptor(CudssDescriptor&& other) noexcept
        : desc(other.desc) { other.desc = nullptr; }

    CudssDescriptor& operator=(CudssDescriptor&& other) noexcept {
        if (this != &other) {
            if (desc) cudssMatrixDestroy(desc);
            desc       = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    // Dense column vector (ncols=1, ld=nrows, CUDSS_R_TYPE, COL_MAJOR).
    void create_dn(int64_t nrows, const void* data) {
        _check(cudssMatrixCreateDn(&desc, nrows, 1, nrows,
                                   data, CUDSS_R_TYPE, CUDSS_LAYOUT_COL_MAJOR),
               "create_dn");
    }

    // Square CSR matrix (row_end=nullptr, int32 offsets+indices, CUDSS_R_TYPE,
    // MTYPE_GENERAL, MVIEW_FULL, BASE_ZERO).
    void create_csr(int64_t n, int64_t nnz,
                    const void* row_ptr, const void* col_idx, const void* values) {
        _check(cudssMatrixCreateCsr(&desc, n, n, nnz,
                                    row_ptr, nullptr, col_idx, values,
                                    CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_TYPE,
                                    CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
               "create_csr");
    }

    void set_values(const void* values) {
        _check(cudssMatrixSetValues(desc, values), "set_values");
    }

private:
    static void _check(cudssStatus_t s, const char* caller) {
        if (s != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(
                std::string("CudssDescriptor::") + caller + " failed: status=" + std::to_string(s));
    }
};

// ---------------------------------------------------------------------------
// Maps the CUDA-free ReorderingAlg (reordering_alg.hpp) to the real cuDSS
// enum. Default matches cuDSS's own implicit default (nothing set) exactly.
// ---------------------------------------------------------------------------
inline cudssReorderingAlg_t to_cudss_reordering_alg(ReorderingAlg alg) {
    switch (alg) {
        case ReorderingAlg::BtfColamd:        return CUDSS_REORDERING_ALG_BTF_COLAMD;
        case ReorderingAlg::Colamd:           return CUDSS_REORDERING_ALG_COLAMD;
        case ReorderingAlg::Amd:              return CUDSS_REORDERING_ALG_AMD;
        case ReorderingAlg::NestedDissection: return CUDSS_REORDERING_ALG_NESTED_DISSECTION;
        case ReorderingAlg::None:             return CUDSS_REORDERING_ALG_NONE;
        default:                              return CUDSS_REORDERING_ALG_DEFAULT;
    }
}

// ---------------------------------------------------------------------------
// RAII wrapper for the cudss solver context: handle, config and data.
// ---------------------------------------------------------------------------
struct CudssContext {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;

    ~CudssContext() {
        if (data)   cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }

    CudssContext() = default;
    CudssContext(const CudssContext&) = delete;
    CudssContext& operator=(const CudssContext&) = delete;

    CudssContext(CudssContext&& other) noexcept
        : handle(other.handle), config(other.config), data(other.data) {
        other.handle = nullptr; other.config = nullptr; other.data = nullptr;
    }

    CudssContext& operator=(CudssContext&& other) noexcept {
        if (this != &other) {
            if (data)   cudssDataDestroy(handle, data);
            if (config) cudssConfigDestroy(config);
            if (handle) cudssDestroy(handle);
            handle = other.handle; config = other.config; data = other.data;
            other.handle = nullptr; other.config = nullptr; other.data = nullptr;
        }
        return *this;
    }

    // Must be called after cudssConfigCreate() and before analyze() to take
    // effect (CUDSS_CONFIG_REORDERING_ALG is read during CUDSS_PHASE_ANALYSIS).
    void set_reordering_alg(ReorderingAlg alg) {
        cudssReorderingAlg_t v = to_cudss_reordering_alg(alg);
        cudssStatus_t s = cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
                                         &v, sizeof(v));
        if (s != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(
                std::string("CudssContext::set_reordering_alg failed: status=")
                + std::to_string(static_cast<int>(s)));
    }

    void analyze(const CudssDescriptor& A, const CudssDescriptor& x, const CudssDescriptor& b) {
        // CUDSS_PHASE_ANALYSIS == CUDSS_PHASE_REORDERING | CUDSS_PHASE_SYMBOLIC_FACTORIZATION
        // (cudss_data_types.h). A single combined call is cuDSS's documented usage
        // (see cudss_reordering_phase.cpp); executing the two sub-phases separately
        // and then the combined phase on top redundantly reorders+symbolic-factorizes
        // twice, which can leave workspace sizing / permutation state inconsistent.
        _execute(CUDSS_PHASE_ANALYSIS, A, x, b, "analyze");
    }
    void factorize(const CudssDescriptor& A, const CudssDescriptor& x, const CudssDescriptor& b) {
        _execute(CUDSS_PHASE_FACTORIZATION, A, x, b, "factorize");
    }
    void refactorize(const CudssDescriptor& A, const CudssDescriptor& x, const CudssDescriptor& b) {
        _execute(CUDSS_PHASE_REFACTORIZATION, A, x, b, "refactorize");
    }
    void solve(const CudssDescriptor& A, const CudssDescriptor& x, const CudssDescriptor& b) {
        _execute(CUDSS_PHASE_SOLVE, A, x, b, "solve");
    }

private:
    void _execute(cudssPhase_t phase,
                  const CudssDescriptor& A, const CudssDescriptor& x, const CudssDescriptor& b,
                  const char* caller) {
        cudssStatus_t s = cudssExecute(handle, phase, config, data, A.desc, x.desc, b.desc);
        if (s != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(
                std::string("CudssContext::") + caller + " failed: status=" + std::to_string(s));
    }
};

// ---------------------------------------------------------------------------
// Helper macro
// ---------------------------------------------------------------------------
#define CHECK_CUDSS(call, msg)                                           \
    do {                                                                 \
        cudssStatus_t _s = (call);                                       \
        if (_s != CUDSS_STATUS_SUCCESS) {                                \
            std::cerr << (msg) << " (status=" << _s << ")\n";           \
            return {};                                                    \
        }                                                                \
    } while(0)

// ---------------------------------------------------------------------------
// cuSPARSE work buffer
// ---------------------------------------------------------------------------
struct CuBuffer {
    void*  ptr  = nullptr;
    size_t size = 0;
    ~CuBuffer() { if (ptr) cudaFree(ptr); }
    CuBuffer() = default;
    CuBuffer(const CuBuffer&) = delete;
    CuBuffer& operator=(const CuBuffer&) = delete;
};

// ---------------------------------------------------------------------------
// cuSPARSE handle + descriptors for one SpMV  (Y_csr · x_dn → y_dn)
// ---------------------------------------------------------------------------
struct CuSpMV {
    cusparseHandle_t          handle  = nullptr;
    cusparseConstSpMatDescr_t mat     = nullptr;
    cusparseConstDnVecDescr_t vec_x   = nullptr;
    cusparseDnVecDescr_t      vec_y   = nullptr;
    CuBuffer                  buf;

    ~CuSpMV() {
        if (vec_y)  cusparseDestroyDnVec(vec_y);
        if (vec_x)  cusparseDestroyDnVec(vec_x);
        if (mat)    cusparseDestroySpMat(mat);
        if (handle) cusparseDestroy(handle);
    }
    CuSpMV() = default;
    CuSpMV(const CuSpMV&) = delete;
    CuSpMV& operator=(const CuSpMV&) = delete;

    void spmv() {
        cusparseStatus_t status = cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_cplx_one, mat, vec_x,
            &h_cplx_zero, vec_y,
            CUDA_C_TYPE, CUSPARSE_SPMV_ALG_DEFAULT, buf.ptr);
        if (status != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error(
                std::string("cusparseSpMV failed: ") + cusparseGetErrorString(status));
    }
};

// ---------------------------------------------------------------------------
// CudaDeviceScalar
// ---------------------------------------------------------------------------
template <typename T>
struct CudaDeviceScalar {
    T* ptr = nullptr;

    explicit CudaDeviceScalar(const T& host_value) {
        cudaError_t err = cudaMalloc(&ptr, sizeof(T));
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaDeviceScalar cudaMalloc: ") + cudaGetErrorString(err));
        err = cudaMemcpy(ptr, &host_value, sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(ptr); ptr = nullptr;
            throw std::runtime_error(
                std::string("CudaDeviceScalar cudaMemcpy: ") + cudaGetErrorString(err));
        }
    }
    CudaDeviceScalar() = default;
    ~CudaDeviceScalar() { if (ptr) cudaFree(ptr); }
    CudaDeviceScalar(const CudaDeviceScalar&) = delete;
    CudaDeviceScalar& operator=(const CudaDeviceScalar&) = delete;
    CudaDeviceScalar(CudaDeviceScalar&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
    CudaDeviceScalar& operator=(CudaDeviceScalar&& o) noexcept {
        if (this != &o) { if (ptr) cudaFree(ptr); ptr = o.ptr; o.ptr = nullptr; }
        return *this;
    }
    T* get() const { return ptr; }
};

// ---------------------------------------------------------------------------
// Stream-aware device memory helpers
//
// These three functions replace thrust::copy/fill when a non-default stream
// must be used.  thrust::copy with thrust::cuda::par.on(s) routes ALL copies
// through the GPU code path regardless of where the source pointer lives:
// raw host pointers are indistinguishable from device pointers at the thrust
// dispatch level, so H→D copies are silently treated as D→D and fail with
// cudaErrorInvalidValue.  cudaMemcpyAsync is explicit about the direction and
// is the correct primitive for stream-ordered data movement.
//
//   upload_h2d  — resize dst to n, then H→D on stream
//   copy_d2d    — resize dst to match src, then D→D on stream
//   zero_d      — cudaMemsetAsync to 0 on stream (dst must already be sized)
//
// All three throw std::runtime_error on CUDA failure.
// ---------------------------------------------------------------------------
template<typename T>
void upload_h2d(thrust::device_vector<T>& dst,
                const T*                  src,
                std::size_t               n,
                cudaStream_t              stream)
{
    dst.resize(n);
    cudaError_t err = cudaMemcpyAsync(
        thrust::raw_pointer_cast(dst.data()), src,
        n * sizeof(T), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("upload_h2d: ") + cudaGetErrorString(err));
}

template<typename T>
void copy_d2d(thrust::device_vector<T>&       dst,
              const thrust::device_vector<T>& src,
              cudaStream_t                     stream)
{
    dst.resize(src.size());
    cudaError_t err = cudaMemcpyAsync(
        thrust::raw_pointer_cast(dst.data()),
        thrust::raw_pointer_cast(src.data()),
        src.size() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("copy_d2d: ") + cudaGetErrorString(err));
}

template<typename T>
void zero_d(thrust::device_vector<T>& dst,
            cudaStream_t               stream)
{
    cudaError_t err = cudaMemsetAsync(
        thrust::raw_pointer_cast(dst.data()), 0,
        dst.size() * sizeof(T), stream);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("zero_d: ") + cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// CudaEvent
//
// RAII wrapper for a single cudaEvent_t.
//
// Usage examples:
//
//   // As a standalone synchronisation point:
//   CudaEvent ev;
//   ev.record(stream);           // insert marker into stream
//   ev.synchronize();            // CPU blocks until marker is reached
//
//   // As a fence between two kernels on the same stream (e_va_done pattern):
//   CudaEvent gate;
//   gate.record(stream);
//   kernel_A<<<..., stream>>>();
//   gate.synchronize();          // CPU ensures A is done before launching B
//   kernel_B<<<..., stream>>>();
//
//   // Elapsed time between two events:
//   CudaEvent start, stop;
//   start.record(stream);
//   kernel<<<..., stream>>>();
//   stop.record(stream);
//   stop.synchronize();
//   float ms = stop.elapsed_since(start);
//
// The implicit conversion operator lets a CudaEvent be passed directly
// wherever a cudaEvent_t is expected (e.g. legacy API calls).
// ---------------------------------------------------------------------------
struct CudaEvent {
    cudaEvent_t ev = nullptr;

    CudaEvent() {
        cudaError_t err = cudaEventCreate(&ev);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaEvent: cudaEventCreate: ") + cudaGetErrorString(err));
    }

    ~CudaEvent() {
        if (ev) cudaEventDestroy(ev);
    }

    // Non-copyable (owns a unique CUDA handle)
    CudaEvent(const CudaEvent&)            = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Movable
    CudaEvent(CudaEvent&& o) noexcept : ev(o.ev) { o.ev = nullptr; }
    CudaEvent& operator=(CudaEvent&& o) noexcept {
        if (this != &o) {
            if (ev) cudaEventDestroy(ev);
            ev = o.ev; o.ev = nullptr;
        }
        return *this;
    }

    // Record a marker into stream s (nullptr = default stream).
    void record(cudaStream_t s = nullptr) {
        cudaError_t err = cudaEventRecord(ev, s);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaEvent::record: ") + cudaGetErrorString(err));
    }

    // Block the calling CPU thread until this event has been reached on the GPU.
    void synchronize() const {
        cudaError_t err = cudaEventSynchronize(ev);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaEvent::synchronize: ") + cudaGetErrorString(err));
    }

    // Elapsed GPU time in milliseconds between `start` and this event.
    // Both events must have been recorded and the stop event must be complete
    // (call synchronize() on *this before calling elapsed_since).
    float elapsed_since(const CudaEvent& start) const {
        float ms = 0.f;
        cudaError_t err = cudaEventElapsedTime(&ms, start.ev, ev);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaEvent::elapsed_since: ") + cudaGetErrorString(err));
        return ms;
    }

    // Implicit conversion so a CudaEvent can be passed where cudaEvent_t is
    // expected without spelling out .ev everywhere.
    operator cudaEvent_t() const { return ev; }
};

// ---------------------------------------------------------------------------
// CudaTimer
//
// RAII wrapper that times a GPU work segment with both GPU-event precision
// and CPU wall-clock precision.  Returns a TimingEntry{gpu_ms, wall_ms}.
//
//   gpu_ms  — cudaEventElapsedTime between start and stop events.
//             Excludes kernel-launch overhead and any CPU work in the segment.
//   wall_ms — std::chrono wall-clock from start() to the cudaEventSynchronize()
//             inside stop_ms().  Always >= gpu_ms; the difference is driver /
//             launch overhead.
//
// Both _start and _stop are CudaEvent members — no manual destroy needed.
// The explicit stream parameter (set at construction) is forwarded to every
// CudaEvent::record() call so all timing stays on the same stream.
//
// Usage (unchanged from the original):
//   CudaTimer timer(cs);       // bind to stream cs
//   timer.start();
//   launch_kernel<<<..., cs>>>();
//   timings.t_spmv += timer.stop_ms();
// ---------------------------------------------------------------------------
struct CudaTimer {
    CudaEvent    _start;           // records the beginning of the timed segment
    CudaEvent    _stop;            // records the end   of the timed segment
    cudaStream_t _stream{nullptr}; // stream on which work is recorded
    std::chrono::steady_clock::time_point _wall_start;

    explicit CudaTimer(cudaStream_t s = nullptr) : _stream(s) {}

    // Default destructor is correct: CudaEvent members clean themselves up.
    ~CudaTimer() = default;

    // Non-copyable (CudaEvent members are non-copyable)
    CudaTimer(const CudaTimer&)            = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    void start() {
        _start.record(_stream);
        _wall_start = std::chrono::steady_clock::now();
    }

    // Records the stop event, synchronises (so wall time is accurate),
    // then returns {gpu_ms, wall_ms}.
    TimingEntry stop_ms() {
        _stop.record(_stream);
        _stop.synchronize();   // CPU waits for GPU — wall_ms is now valid
        double wall_ms = ms_since(_wall_start);
        return {static_cast<double>(_stop.elapsed_since(_start)), wall_ms};
    }
};

// ---------------------------------------------------------------------------
// CudaStream
// ---------------------------------------------------------------------------
struct CudaStream {
    cudaStream_t stream = nullptr;

    CudaStream() {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaStream: ") + cudaGetErrorString(err));
    }
    ~CudaStream() { if (stream) { cudaStreamDestroy(stream); stream = nullptr; } }

    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& o) noexcept : stream(o.stream) { o.stream = nullptr; }
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) {
            if (stream) cudaStreamDestroy(stream);
            stream = o.stream; o.stream = nullptr;
        }
        return *this;
    }

    void synchronize() const {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CudaStream::synchronize: ") + cudaGetErrorString(err));
    }

    operator cudaStream_t() const { return stream; }
};

#endif  // MY_CUDA_UTILS_H
