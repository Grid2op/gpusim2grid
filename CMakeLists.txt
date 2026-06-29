cmake_minimum_required(VERSION 3.15...4.0)
message("Using CMake version ${CMAKE_VERSION}")

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
