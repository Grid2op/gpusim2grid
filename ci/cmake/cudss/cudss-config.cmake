# Minimal CMake package config for the cuDSS *pip wheel* (nvidia-cudss-cu12/-cu13).
#
# NVIDIA's tarball / apt packages of cuDSS ship a real cudss-config.cmake; the
# pip wheel ships only the headers and the library (libcudss.so.1 for cuDSS 1.x,
# libcudss.so.0 for 0.x), so find_package(cudss) can't find it. This shim lets
# `find_package(cudss 0.8.0 REQUIRED)` in
# src/_cpp/CMakeLists.txt resolve against the wheel. Used by CI
# (.github/workflows/build.yml); point cudss_DIR (or cudss_ROOT) at this
# directory and CUDSS_WHEEL_ROOT at the wheel's <site-packages>/nvidia/cuXX
# (otherwise the wheel matching nvcc's CUDA major is used).

if(TARGET cudss)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cudss-wheel-root.cmake")

find_path(cudss_INCLUDE_DIR cudss.h HINTS "${_cudss_root}/include" NO_DEFAULT_PATH)
# The versioned names first: the soname is the file's own name.
find_file(cudss_LIBRARY NAMES libcudss.so.1 libcudss.so.0 libcudss.so
          HINTS "${_cudss_root}/lib" NO_DEFAULT_PATH)

if(NOT cudss_INCLUDE_DIR OR NOT cudss_LIBRARY)
    set(cudss_FOUND FALSE)
    set(cudss_NOT_FOUND_MESSAGE
        "cuDSS pip wheel not found under '${_cudss_root}' (set CUDSS_WHEEL_ROOT)")
    return()
endif()

get_filename_component(_cudss_soname "${cudss_LIBRARY}" NAME)
add_library(cudss SHARED IMPORTED)
set_target_properties(cudss PROPERTIES
    IMPORTED_LOCATION "${cudss_LIBRARY}"
    IMPORTED_SONAME "${_cudss_soname}"
    INTERFACE_INCLUDE_DIRECTORIES "${cudss_INCLUDE_DIR}")
set(cudss_INCLUDE_DIRS "${cudss_INCLUDE_DIR}")
set(cudss_LIBRARIES cudss)
set(cudss_FOUND TRUE)
