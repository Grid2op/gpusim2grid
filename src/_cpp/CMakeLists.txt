
project(_gpusim2grid LANGUAGES CXX CUDA)
# No version in find_package: cuDSS's version file only accepts a request of its own
# major version, so `find_package(cudss 0.8.0)` would reject 1.x. Check the floor here.
find_package(cudss REQUIRED)
if(cudss_VERSION AND cudss_VERSION VERSION_LESS 0.8.0)
    message(FATAL_ERROR "cuDSS >= 0.8.0 required, found ${cudss_VERSION}")
endif()
message(STATUS "cuDSS ${cudss_VERSION} found in ${cudss_DIR}")
# A CUDA toolkit installed from PyPI (``cuda-toolkit[nvcc,...]``, CUDA 13: one
# prefix <site-packages>/nvidia/cu13 holding bin/nvcc, include/ and lib/) ships
# each library under its soname only (libcudart.so.13, libcusparse.so.12), not
# the unversioned libcudart.so that FindCUDAToolkit's find_library looks for.
# Point it at the versioned files then; a toolkit from NVIDIA's installer or
# container has the unversioned names and is left alone.
get_filename_component(_g2g_cuda_lib "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_g2g_cuda_lib "${_g2g_cuda_lib}/../lib" ABSOLUTE)
foreach(_g2g_l cudart cublas cublasLt cusolver cusparse nvJitLink)
    if(NOT EXISTS "${_g2g_cuda_lib}/lib${_g2g_l}.so")
        file(GLOB _g2g_f "${_g2g_cuda_lib}/lib${_g2g_l}.so.[0-9]*")
        if(_g2g_f)
            list(SORT _g2g_f)
            list(GET _g2g_f 0 _g2g_f)
            set(CUDA_${_g2g_l}_LIBRARY "${_g2g_f}" CACHE FILEPATH "")
            if(_g2g_l STREQUAL "cudart")
                set(CUDA_CUDART "${_g2g_f}" CACHE FILEPATH "")
            endif()
        endif()
    endif()
endforeach()
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudart cusolver cusparse)
# The zero-copy lightsim2grid bridge is compiled only when lightsim2grid_core
# was found at configure time (see ../CMakeLists.txt). It is a CUDA-free .cpp.
set(_gpusim2grid_ls2g_sources "")
if(GPUSIM2GRID_LS2G_CORE_FOUND)
    list(APPEND _gpusim2grid_ls2g_sources ls2g_bridge.cpp outer/ls2g_outer_model.cpp)
endif()

# gpusim2grid_core: the solvers as a plain C++/CUDA shared library, with no Python in it,
# so that a C++ program (or a JNI library) can use them directly. Its public headers
# (installed under gpusim2grid/include) do not include any CUDA header. The Python module
# below is only the bindings and the DLPack export, linked against it.
add_library(
    gpusim2grid_core SHARED
    cuda_utils.cu
    acpf_nr.cu
    contingency_analysis_helper.cpp
    ledger_extend.cpp
    acpf_nr_kernels.cu
    outer/outer_kernels.cu
    outer/branch_control_kernels.cu
    contingency_analysis_session.cu
    contingency/batch_pf_driver.cu
    contingency/violation_kernels.cu
    contingency/batch_sources/contingency_batch.cu
    contingency/batch_sources/injection_batch.cu
    contingency/batch_sources/scenario_sweep_batch.cu
    injection_sweep.cu
    injection_sweep_session.cu
    scenario_sweep_session.cu
    raw_cudss_solve.cu
    warmup.cu
    own_lu/lu_plan.cpp
    own_lu/lu_order.cpp
    own_lu/batched_lu.cu
    ${_gpusim2grid_ls2g_sources})
add_library(gpusim2grid::core ALIAS gpusim2grid_core)
set_target_properties(gpusim2grid_core PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME core)
target_include_directories(gpusim2grid_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${SKBUILD_PROJECT_NAME}/include>)
# the CUDA runtime is linked statically (CMake's default for CUDA targets), as before the split:
# a shared libcudart from the toolkit would clash with the one torch loads
target_link_libraries(gpusim2grid_core PUBLIC CUDA::cusparse cudss)
if(USE_FLOAT_PRECISION)
    target_compile_definitions(gpusim2grid_core PUBLIC GPUSIM2GRID_REAL_FLOAT)
endif()

pybind11_add_module(
    _gpusim2grid
    python_bindings.cpp
    dlpack_export.cu)
target_link_libraries(_gpusim2grid PRIVATE gpusim2grid_core)
# _gpusim2grid.so NEEDs libgpusim2grid_core.so, installed next to it (and, with the bridge,
# liblightsim2grid_core.so: its rpath is set below).
if(NOT APPLE)
    set_property(TARGET _gpusim2grid APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
else()
    set_property(TARGET _gpusim2grid APPEND PROPERTY INSTALL_RPATH "@loader_path")
endif()

if(GPUSIM2GRID_LS2G_CORE_FOUND)
    target_link_libraries(gpusim2grid_core PUBLIC lightsim2grid::core)
    target_compile_definitions(gpusim2grid_core PUBLIC GPUSIM2GRID_HAVE_LS2G)

    # Mirror lightsim2grid_core's -march=native / -O3 onto gpusim2grid_core (PUBLIC: the module and any C++ consumer inherit it).
    #
    # lightsim2grid_core and libgpusim2grid_core.so are two *separate* shared
    # libraries, and ls2g_bridge.cpp exchanges dynamic-size Eigen objects with
    # it by value (grid.get_J_solver() -> Eigen::SparseMatrix<...>,
    # grid.get_bus_vn_kv()/get_limit_a1_ka()/etc. -> RealVect/CplxVect).
    # -march=native changes which SIMD ISA Eigen sees enabled, which changes
    # EIGEN_MAX_ALIGN_BYTES and thus how Eigen aligns/allocates/frees its
    # dynamic-size matrices; an object allocated under one alignment
    # assumption and freed under another silently corrupts the heap the
    # moment it crosses this boundary. This is a real, previously-hit bug in
    # lightsim2grid itself (see its CHANGELOG.rst and
    # docs/solver_plugin.rst / examples/cmake/MatchLightsim2gridBuildFlags.cmake,
    # whose recipe this block follows) -- not a hypothetical one.
    #
    # lightsim2grid_core_MARCH_NATIVE / _O3_OPTIM are exported by its CMake
    # config (lightsim2grid_coreConfig.cmake.in) with exactly this in mind.
    if(NOT MSVC)
        if(lightsim2grid_core_MARCH_NATIVE)
            message(STATUS "[gpusim2grid] lightsim2grid_core was built with "
                           "-march=native -- matching it on gpusim2grid_core "
                           "(required, see lightsim2grid's docs/solver_plugin.rst)")
            target_compile_options(gpusim2grid_core PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-march=native>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
        endif()
        if(lightsim2grid_core_O3_OPTIM)
            message(STATUS "[gpusim2grid] lightsim2grid_core was built with "
                           "-O3 -- matching it on gpusim2grid_core")
            target_compile_options(gpusim2grid_core PUBLIC
                $<$<COMPILE_LANGUAGE:CXX>:-O3>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)
        endif()
    endif()

    # libgpusim2grid_core.so NEEDs liblightsim2grid_core.so. Both packages install
    # side by side under site-packages, so a relocatable $ORIGIN-relative rpath
    # to the lightsim2grid package dir lets the loader find it without relying
    # on import order or LD_LIBRARY_PATH.
    if(NOT APPLE)
        set_property(TARGET gpusim2grid_core APPEND PROPERTY
            INSTALL_RPATH "$ORIGIN/../lightsim2grid")
        set_property(TARGET gpusim2grid_core APPEND PROPERTY
            BUILD_RPATH "$ORIGIN/../lightsim2grid")
        set_property(TARGET _gpusim2grid APPEND PROPERTY
            INSTALL_RPATH "$ORIGIN/../lightsim2grid")
    else()
        set_property(TARGET _gpusim2grid APPEND PROPERTY
            INSTALL_RPATH "@loader_path/../lightsim2grid")
        set_property(TARGET gpusim2grid_core APPEND PROPERTY
            INSTALL_RPATH "@loader_path/../lightsim2grid")
    endif()

    # LSGrid.hpp transitively includes KLUSolver.hpp because lightsim2grid_core
    # exports -DKLU_SOLVER_AVAILABLE. That header needs the SuiteSparse headers
    # (cs.h, klu.h, amd.h, colamd.h, btf.h, SuiteSparse_config.h). We must
    # compile the bridge *with* KLU_SOLVER_AVAILABLE to match the ABI of the
    # loaded lightsim2grid_cpp module, so we need these headers. lightsim2grid
    # >= 1.3.0 ships them with the core; older versions do not, and they have
    # to come from somewhere else.
    #
    # Search order (first hit wins):
    #   1. SUITESPARSE_INCLUDE_DIR (CMake var or env var): a single directory
    #      containing klu.h et al. directly (flat layout, as a system package
    #      or a hand-merged include dir would produce).
    #   1b. The ones lightsim2grid_core ships (lightsim2grid >= 1.3.0), already
    #      on lightsim2grid::core's include path.
    #   2. The SuiteSparse checkout next to the lightsim2grid *source* this
    #      Python interpreter's lightsim2grid was built from (per-module
    #      Include/ layout, e.g. a local lightsim2grid clone's SuiteSparse
    #      submodule). Auto-detected from `lightsim2grid.__file__`; override
    #      with -DGPUSIM2GRID_SUITESPARSE_DIR=... if that guess is wrong.
    #   3. The system SuiteSparse dev package (e.g. Debian/Ubuntu's
    #      libsuitesparse-dev), headers flat under <prefix>/include/suitesparse/.
    #   4. Nothing found: fail configure now with a clear message instead of
    #      letting the build fail later on a bare "cs.h: No such file".
    if(NOT DEFINED SUITESPARSE_INCLUDE_DIR AND DEFINED ENV{SUITESPARSE_INCLUDE_DIR})
        set(SUITESPARSE_INCLUDE_DIR "$ENV{SUITESPARSE_INCLUDE_DIR}")
    endif()

    set(GPUSIM2GRID_SUITESPARSE_FOUND FALSE)

    # 1. Explicit flat override.
    if(SUITESPARSE_INCLUDE_DIR AND EXISTS "${SUITESPARSE_INCLUDE_DIR}/klu.h")
        target_include_directories(gpusim2grid_core PUBLIC
            $<BUILD_INTERFACE:${SUITESPARSE_INCLUDE_DIR}>)
        message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge "
                       "(SUITESPARSE_INCLUDE_DIR): ${SUITESPARSE_INCLUDE_DIR}")
        set(GPUSIM2GRID_SUITESPARSE_FOUND TRUE)
    elseif(SUITESPARSE_INCLUDE_DIR)
        message(STATUS "[gpusim2grid] SUITESPARSE_INCLUDE_DIR="
                       "${SUITESPARSE_INCLUDE_DIR} does not contain klu.h "
                       "directly; ignoring it and trying the next option.")
    endif()

    # 1b. lightsim2grid >= 1.3.0 installs them with the core and puts them on
    #     lightsim2grid::core's include path: nothing to add.
    if(NOT GPUSIM2GRID_SUITESPARSE_FOUND)
        get_target_property(_ls2g_incs lightsim2grid::core INTERFACE_INCLUDE_DIRECTORIES)
        foreach(_d IN LISTS _ls2g_incs)
            if(EXISTS "${_d}/klu.h")
                message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge "
                               "(shipped with lightsim2grid_core): ${_d}")
                set(GPUSIM2GRID_SUITESPARSE_FOUND TRUE)
            endif()
        endforeach()
    endif()

    # 2. Auto-detect the lightsim2grid source checkout's own SuiteSparse
    #    submodule: <dir of lightsim2grid.__file__>/../SuiteSparse. This is
    #    where a local editable/source build of lightsim2grid keeps it, with
    #    each module's headers under its own Include/ subdir.
    if(NOT GPUSIM2GRID_SUITESPARSE_FOUND AND NOT DEFINED GPUSIM2GRID_SUITESPARSE_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                    "import lightsim2grid, os; print(os.path.dirname(os.path.dirname(lightsim2grid.__file__)))"
            OUTPUT_VARIABLE GPUSIM2GRID_LS2G_SRC_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _ls2g_src_dir_result
            ERROR_QUIET)
        if(_ls2g_src_dir_result EQUAL 0 AND EXISTS "${GPUSIM2GRID_LS2G_SRC_DIR}/SuiteSparse")
            set(GPUSIM2GRID_SUITESPARSE_DIR "${GPUSIM2GRID_LS2G_SRC_DIR}/SuiteSparse")
        endif()
    endif()

    if(NOT GPUSIM2GRID_SUITESPARSE_FOUND AND EXISTS "${GPUSIM2GRID_SUITESPARSE_DIR}/KLU/Include/klu.h")
        target_include_directories(gpusim2grid_core PUBLIC
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/CXSparse/Include>
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/KLU/Include>
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/AMD/Include>
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/COLAMD/Include>
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/BTF/Include>
            $<BUILD_INTERFACE:${GPUSIM2GRID_SUITESPARSE_DIR}/SuiteSparse_config>)
        message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge "
                       "(lightsim2grid source checkout): ${GPUSIM2GRID_SUITESPARSE_DIR}")
        set(GPUSIM2GRID_SUITESPARSE_FOUND TRUE)
    endif()

    # 3. System SuiteSparse dev package.
    if(NOT GPUSIM2GRID_SUITESPARSE_FOUND)
        find_path(GPUSIM2GRID_SYSTEM_SUITESPARSE_INCLUDE_DIR
            NAMES klu.h
            PATH_SUFFIXES suitesparse)
        if(GPUSIM2GRID_SYSTEM_SUITESPARSE_INCLUDE_DIR)
            target_include_directories(gpusim2grid_core PUBLIC
                $<BUILD_INTERFACE:${GPUSIM2GRID_SYSTEM_SUITESPARSE_INCLUDE_DIR}>)
            message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge "
                           "(system install): ${GPUSIM2GRID_SYSTEM_SUITESPARSE_INCLUDE_DIR}")
            set(GPUSIM2GRID_SUITESPARSE_FOUND TRUE)
        endif()
    endif()

    # 4. Nothing worked: fail now rather than at compile time.
    if(NOT GPUSIM2GRID_SUITESPARSE_FOUND)
        message(FATAL_ERROR
            "[gpusim2grid] SuiteSparse headers not found (cs.h/klu.h). Tried, "
            "in order: SUITESPARSE_INCLUDE_DIR (env var or -D, flat dir "
            "containing klu.h directly), the ones lightsim2grid_core ships "
            "(lightsim2grid >= 1.3.0), the SuiteSparse checkout next to "
            "the lightsim2grid this Python's lightsim2grid was built from "
            "(auto-detected, override with -DGPUSIM2GRID_SUITESPARSE_DIR=... "
            "at a per-module Include/ layout), and the system SuiteSparse dev "
            "package (e.g. libsuitesparse-dev on Debian/Ubuntu). Install one "
            "of these, or point one of the two variables above at it.")
    endif()
endif()

install(TARGETS _gpusim2grid DESTINATION ${SKBUILD_PROJECT_NAME})

# gpusim2grid_core for C++ consumers: the library next to the module, its headers under
# gpusim2grid/include, and a CMake package (gpusim2grid.get_cmake_dir()) giving the
# imported target gpusim2grid::core. A consumer that also links lightsim2grid::core must
# match its -march / -O3 flags (they are part of the target's interface, see above).
include(CMakePackageConfigHelpers)
set(GPUSIM2GRID_CORE_CMAKEDIR ${SKBUILD_PROJECT_NAME}/share/cmake/gpusim2grid_core)
install(TARGETS gpusim2grid_core EXPORT gpusim2grid_coreTargets
    LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
    RUNTIME DESTINATION ${SKBUILD_PROJECT_NAME}
    ARCHIVE DESTINATION ${SKBUILD_PROJECT_NAME})
# The headers and the CMake package only with the lightsim2grid bridge: without it the
# headers would need the vendored Eigen, which is not installed.
if(GPUSIM2GRID_LS2G_CORE_FOUND)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${SKBUILD_PROJECT_NAME}/include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
        PATTERN "dlpack" EXCLUDE PATTERN "dlpack_export.hpp" EXCLUDE)
    install(EXPORT gpusim2grid_coreTargets NAMESPACE gpusim2grid::
        DESTINATION ${GPUSIM2GRID_CORE_CMAKEDIR})
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpusim2grid_coreConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/gpusim2grid_coreConfig.cmake
        INSTALL_DESTINATION ${GPUSIM2GRID_CORE_CMAKEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gpusim2grid_coreConfig.cmake
        DESTINATION ${GPUSIM2GRID_CORE_CMAKEDIR})
endif()
