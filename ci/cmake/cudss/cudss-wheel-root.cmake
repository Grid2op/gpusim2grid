# Sets _cudss_root to the cuDSS pip wheel's prefix (<site-packages>/nvidia/cuXX):
# $ENV{CUDSS_WHEEL_ROOT} if set, else the wheel matching nvcc's CUDA major
# (nvidia-cudss-cu12 for CUDA 12, -cu13 for CUDA 13) in Python_EXECUTABLE.

set(_cudss_root "$ENV{CUDSS_WHEEL_ROOT}")
if(NOT _cudss_root)
    string(REGEX MATCH "^[0-9]+" _cudss_cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import nvidia.cu${_cudss_cuda_major} as m; print(list(m.__path__)[0])"
        OUTPUT_VARIABLE _cudss_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()
