
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
    injection_sweep.cu
    injection_sweep_session.cu
    dlpack_export.cu
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
    # lightsim2grid_cpp module, so we point at the vendored SuiteSparse
    # submodule. Override with -DGPUSIM2GRID_SUITESPARSE_DIR=... if needed.
    # (Upstream fix: lightsim2grid_core should bundle these headers or keep KLU
    #  out of its public header surface — see docs/lightsim2grid_requirements.md.)
    if(NOT DEFINED GPUSIM2GRID_SUITESPARSE_DIR)
        set(GPUSIM2GRID_SUITESPARSE_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/../../_lightsim2grid/SuiteSparse")
    endif()
    if(EXISTS "${GPUSIM2GRID_SUITESPARSE_DIR}/KLU/Include/klu.h")
        target_include_directories(_gpusim2grid PRIVATE
            "${GPUSIM2GRID_SUITESPARSE_DIR}/CXSparse/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/KLU/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/AMD/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/COLAMD/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/BTF/Include"
            "${GPUSIM2GRID_SUITESPARSE_DIR}/SuiteSparse_config")
        message(STATUS "[gpusim2grid] SuiteSparse headers for ls2g bridge: "
                       "${GPUSIM2GRID_SUITESPARSE_DIR}")
    else()
        message(WARNING "[gpusim2grid] SuiteSparse headers not found at "
                        "${GPUSIM2GRID_SUITESPARSE_DIR}; the ls2g bridge may "
                        "fail to compile (cs.h/klu.h). Set "
                        "-DGPUSIM2GRID_SUITESPARSE_DIR=...")
    endif()
endif()

install(TARGETS _gpusim2grid DESTINATION ${SKBUILD_PROJECT_NAME})
