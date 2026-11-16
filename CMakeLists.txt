cmake_minimum_required(VERSION 3.15...4.0)
message("Using CMake version ${CMAKE_VERSION}")

# ---------------------------------------------------------------------------
# The CUDA toolkit from PyPI, without any environment variable.
#
# An isolated build of the sdist (`pip install gpusim2grid`) gets PyPI's CUDA 13
# toolkit as a build requirement (pyproject.toml): one prefix
# <site-packages>/nvidia/cu13 holding bin/nvcc, include/ and lib/. When nothing
# names a toolkit (CUDACXX, CMAKE_CUDA_COMPILER, CUDAToolkit_ROOT), use that one
# if the building Python has it; otherwise CMake's own search (nvcc on PATH) as
# before. Must run before project(), which picks the CUDA compiler.
# scikit-build-core passes Python_EXECUTABLE.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL ""
   AND NOT DEFINED CUDAToolkit_ROOT AND "$ENV{CUDAToolkit_ROOT}" STREQUAL ""
   AND DEFINED Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import nvidia.cu13 as m; print(list(m.__path__)[0])"
        OUTPUT_VARIABLE _g2g_wheel_cuda
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _g2g_wheel_cuda_res
        ERROR_QUIET)
    if(_g2g_wheel_cuda_res EQUAL 0 AND EXISTS "${_g2g_wheel_cuda}/bin/nvcc")
        message(STATUS "[gpusim2grid] CUDA toolkit from PyPI: ${_g2g_wheel_cuda}")
        set(CMAKE_CUDA_COMPILER "${_g2g_wheel_cuda}/bin/nvcc" CACHE FILEPATH "")
        set(CUDAToolkit_ROOT "${_g2g_wheel_cuda}" CACHE PATH "")
    endif()
endif()

# ---------------------------------------------------------------------------
# GPU architectures, when nothing sets them (CMAKE_CUDA_ARCHITECTURES, CUDAARCHS):
# the GPUs of this machine (nvidia-smi), else every major architecture CUDA 12/13
# both support, the newest also as PTX. Without this, nvcc's default (sm_75 PTX
# only for CUDA 13) is JIT-compiled at every first launch.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    find_program(_g2g_nvidia_smi nvidia-smi)
    set(_g2g_archs "")
    if(_g2g_nvidia_smi)
        execute_process(
            COMMAND "${_g2g_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _g2g_caps
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _g2g_smi_res
            ERROR_QUIET)
        if(_g2g_smi_res EQUAL 0 AND _g2g_caps)
            string(REPLACE "." "" _g2g_caps "${_g2g_caps}")
            string(REGEX REPLACE "[\r\n]+" ";" _g2g_caps "${_g2g_caps}")
            list(REMOVE_DUPLICATES _g2g_caps)
            foreach(_c IN LISTS _g2g_caps)
                if(_c MATCHES "^[0-9]+$")
                    list(APPEND _g2g_archs "${_c}-real")
                endif()
            endforeach()
        endif()
    endif()
    if(NOT _g2g_archs)
        set(_g2g_archs 75-real 80-real 86-real 89-real 90-real 100-real 120)
    endif()
    message(STATUS "[gpusim2grid] CUDA architectures: ${_g2g_archs} "
                   "(set CMAKE_CUDA_ARCHITECTURES to choose)")
    set(CMAKE_CUDA_ARCHITECTURES "${_g2g_archs}" CACHE STRING "")
endif()

# _gpusim2grid is the name of the PYBIND11_MODULE
project(gpusim2grid LANGUAGES CXX CUDA)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# define this variable (used when this is installed using cmake directly
# without 'pip install XXX')
if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME ${CMAKE_CURRENT_BINARY_DIR}/gpusim2grid)
endif(NOT DEFINED SKBUILD_PROJECT_NAME)

# ---------------------------------------------------------------------------
# Floating-point precision selection
#
# Priority (highest to lowest):
#   1. Environment variable CUDA_REAL_FLOAT  → force FP32
#   2. Environment variable CUDA_REAL_DOUBLE → force FP64
#   3. CMake option -DUSE_FLOAT_PRECISION=ON/OFF (default OFF = FP64)
#
# Example usage:
#   cmake -DUSE_FLOAT_PRECISION=ON ..          # FP32 via CMake option
#   CUDA_REAL_FLOAT=1 pip install .            # FP32 via env var
#   CUDA_REAL_DOUBLE=1 pip install .           # FP64 via env var (explicit)
# ---------------------------------------------------------------------------
option(USE_FLOAT_PRECISION
       "Use FP32 (float) for CUDA/Eigen real types instead of FP64 (double)"
       OFF)

if(DEFINED ENV{CUDA_REAL_FLOAT})
    set(USE_FLOAT_PRECISION ON)
    message(STATUS "[gpusim2grid] CUDA_REAL_FLOAT env var detected → forcing FP32")
elseif(DEFINED ENV{CUDA_REAL_DOUBLE})
    set(USE_FLOAT_PRECISION OFF)
    message(STATUS "[gpusim2grid] CUDA_REAL_DOUBLE env var detected → forcing FP64")
endif()

if(USE_FLOAT_PRECISION)
    add_compile_definitions(GPUSIM2GRID_REAL_FLOAT)
    message(STATUS "[gpusim2grid] Precision: FP32 (float)")
else()
    message(STATUS "[gpusim2grid] Precision: FP64 (double)")
endif()

add_subdirectory(src)
