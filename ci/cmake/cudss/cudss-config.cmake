# Minimal CMake package config for the cuDSS *pip wheel* (nvidia-cudss-cu12).
#
# NVIDIA's tarball / apt packages of cuDSS ship a real cudss-config.cmake; the
# pip wheel ships only the headers and libcudss.so.0, so find_package(cudss)
# can't find it. This shim lets `find_package(cudss 0.8.0 REQUIRED)` in
# src/_cpp/CMakeLists.txt resolve against the wheel. Used by CI
# (.github/workflows/build.yml); point cudss_DIR (or cudss_ROOT) at this
# directory and CUDSS_WHEEL_ROOT at the wheel's <site-packages>/nvidia/cu12.

if(TARGET cudss)
    return()
endif()

set(_cudss_root "$ENV{CUDSS_WHEEL_ROOT}")
if(NOT _cudss_root)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import nvidia.cu12, os; print(list(nvidia.cu12.__path__)[0])"
        OUTPUT_VARIABLE _cudss_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()

find_path(cudss_INCLUDE_DIR cudss.h HINTS "${_cudss_root}/include" NO_DEFAULT_PATH)
find_file(cudss_LIBRARY NAMES libcudss.so libcudss.so.0
          HINTS "${_cudss_root}/lib" NO_DEFAULT_PATH)

if(NOT cudss_INCLUDE_DIR OR NOT cudss_LIBRARY)
    set(cudss_FOUND FALSE)
    set(cudss_NOT_FOUND_MESSAGE
        "cuDSS pip wheel not found under '${_cudss_root}' (set CUDSS_WHEEL_ROOT)")
    return()
endif()

add_library(cudss SHARED IMPORTED)
set_target_properties(cudss PROPERTIES
    IMPORTED_LOCATION "${cudss_LIBRARY}"
    IMPORTED_SONAME "libcudss.so.0"
    INTERFACE_INCLUDE_DIRECTORIES "${cudss_INCLUDE_DIR}")
set(cudss_INCLUDE_DIRS "${cudss_INCLUDE_DIR}")
set(cudss_LIBRARIES cudss)
set(cudss_FOUND TRUE)
