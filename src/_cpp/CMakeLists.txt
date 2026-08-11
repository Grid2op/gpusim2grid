
project(_gpusim2grid LANGUAGES CXX CUDA)
find_package(cudss 0.8.0 REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudart cusolver cusparse)
# The zero-copy lightsim2grid bridge is compiled only when lightsim2grid_core
# was found at configure time (see ../CMakeLists.txt). It is a CUDA-free .cpp.
set(_gpusim2grid_ls2g_sources "")
if(GPUSIM2GRID_LS2G_CORE_FOUND)
    list(APPEND _gpusim2grid_ls2g_sources ls2g_bridge.cpp)
endif()

pybind11_add_module(
    _gpusim2grid
    python_bindings.cpp
    cuda_utils.cu
    acpf_nr.cu
    contingency_analysis_helper.cpp
    acpf_nr_kernels.cu
    contingency_analysis_session.cu
    contingency/batch_pf_driver.cu
    contingency/violation_kernels.cu
    contingency/batch_sources/injection_batch.cu
    contingency/batch_sources/scenario_sweep_batch.cu
    injection_sweep.cu
    injection_sweep_session.cu
    scenario_sweep_session.cu
    dlpack_export.cu
    raw_cudss_solve.cu
    warmup.cu
    ${_gpusim2grid_ls2g_sources})
target_link_libraries(_gpusim2grid PRIVATE CUDA::cusparse cudss)

if(GPUSIM2GRID_LS2G_CORE_FOUND)
    target_link_libraries(_gpusim2grid PRIVATE lightsim2grid::core)
    target_compile_definitions(_gpusim2grid PRIVATE GPUSIM2GRID_HAVE_LS2G)

    # _gpusim2grid.so NEEDs liblightsim2grid_core.so. Both packages install
    # side by side under site-packages, so a relocatable $ORIGIN-relative rpath
    # to the lightsim2grid package dir lets the loader find it without relying
    # on import order or LD_LIBRARY_PATH.
    if(NOT APPLE)
        set_property(TARGET _gpusim2grid APPEND PROPERTY
            INSTALL_RPATH "$ORIGIN/../lightsim2grid")
        set_property(TARGET _gpusim2grid APPEND PROPERTY
            BUILD_RPATH "$ORIGIN/../lightsim2grid")
    else()
        set_property(TARGET _gpusim2grid APPEND PROPERTY
            INSTALL_RPATH "@loader_path/../lightsim2grid")
    endif()

    # LSGrid.hpp transitively includes KLUSolver.hpp because lightsim2grid_core
    # exports -DKLU_SOLVER_AVAILABLE. That header needs the SuiteSparse headers
    # (cs.h, klu.h, amd.h, colamd.h, btf.h, SuiteSparse_config.h), which the
    # installed lightsim2grid_core package does NOT ship. We must compile the
    # bridge *with* KLU_SOLVER_AVAILABLE to match the ABI of the loaded
    # lightsim2grid_cpp module, so we need these headers from somewhere else.
    # (Upstream fix: lightsim2grid_core should bundle these headers or keep KLU
    #  out of its public header surface — see docs/lightsim2grid_requirements.md.)
    #
    # Search order (first hit wins):
    #   1. SUITESPARSE_INCLUDE_DIR (CMake var or env var): a single directory
    #      containing klu.h et al. directly (flat layout, as a system package
    #      or a hand-merged include dir would produce).
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
        target_include_directories(_gpusim2grid PRIVATE "${SUITESPARSE_INCLUDE_DIR}")
        message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge "
                       "(SUITESPARSE_INCLUDE_DIR): ${SUITESPARSE_INCLUDE_DIR}")
        set(GPUSIM2GRID_SUITESPARSE_FOUND TRUE)
    elseif(SUITESPARSE_INCLUDE_DIR)
        message(STATUS "[gpusim2grid] SUITESPARSE_INCLUDE_DIR="
                       "${SUITESPARSE_INCLUDE_DIR} does not contain klu.h "
                       "directly; ignoring it and trying the next option.")
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
        target_include_directories(_gpusim2grid PRIVATE
            "${GPUSIM2GRID_SUITESPARSE_DIR}/CXSparse/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/KLU/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/AMD/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/COLAMD/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/BTF/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/SuiteSparse_config")
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
            target_include_directories(_gpusim2grid PRIVATE
                "${GPUSIM2GRID_SYSTEM_SUITESPARSE_INCLUDE_DIR}")
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
            "containing klu.h directly), the SuiteSparse checkout next to "
            "the lightsim2grid this Python's lightsim2grid was built from "
            "(auto-detected, override with -DGPUSIM2GRID_SUITESPARSE_DIR=... "
            "at a per-module Include/ layout), and the system SuiteSparse dev "
            "package (e.g. libsuitesparse-dev on Debian/Ubuntu). Install one "
            "of these, or point one of the two variables above at it.")
    endif()
endif()

install(TARGETS _gpusim2grid DESTINATION ${SKBUILD_PROJECT_NAME})
