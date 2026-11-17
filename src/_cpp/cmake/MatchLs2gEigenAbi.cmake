# g2g_match_ls2g_eigen_abi(<target> [<flag>...])
#
# Makes <target> (PUBLIC) compile with the Eigen SIMD/alignment settings lightsim2grid_core
# was built with, whatever this compiler enables by default. The <flag>s are the ISA flags
# <target> already gets for both languages (-march=native when the core has it); the probes
# start from them.
#
# Mirroring lightsim2grid_core_MARCH_NATIVE is not enough: when lightsim2grid was built
# without -march=native, what it got is its own compiler's default ISA, and ours may differ.
# The PyPI wheel is built for baseline x86-64 (SSE, 16-byte alignment) while the compiler
# of a GitHub ubuntu-26.04 runner turned AVX2 + FMA on (32-byte alignment), and the
# extension then refused to import (Ls2gAbiTag check in python_bindings.cpp).
#
# So the core's tag is read off the library itself (ls2g::core_abi_tag(), a probe linked
# against lightsim2grid::core and run), compared with what this compiler gives, and the
# x86 ISA flags that differ are turned on/off (-m<isa> / -mno-<isa>) for C++ and for nvcc's
# host compiler alike. The result is checked again for both languages; a mismatch that no
# such flag can fix (another Eigen version, another architecture) stops the configuration.
# When cross-compiling the probe cannot run: nothing is changed, the import-time check
# still guards.
function(g2g_match_ls2g_eigen_abi target)
    if(MSVC)
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(WARNING "[gpusim2grid] cross-compiling: cannot check that the Eigen "
                        "SIMD/alignment settings match lightsim2grid_core's")
        return()
    endif()
    set(_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    if(NOT _dir)  # CMake < 3.17
        set(_dir "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    endif()

    # Runs the probe with extra C++ flags <flags>; sets <core>/<self> to the two tags.
    function(_g2g_abi_probe flags out_core out_self)
        try_run(_g2g_abi_run _g2g_abi_compile
            "${CMAKE_CURRENT_BINARY_DIR}/ls2g_abi_probe"
            "${_dir}/ls2g_abi_probe.cpp"
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${lightsim2grid_core_INCLUDE_DIRS}"
            COMPILE_DEFINITIONS ${flags}
            LINK_LIBRARIES lightsim2grid::core
            COMPILE_OUTPUT_VARIABLE _out
            RUN_OUTPUT_VARIABLE _run)
        if(NOT _g2g_abi_compile OR NOT _g2g_abi_run EQUAL 0)
            message(FATAL_ERROR "[gpusim2grid] could not build / run the Eigen ABI probe "
                                "against lightsim2grid_core (flags: '${flags}'):\n${_out}\n${_run}")
        endif()
        string(REGEX MATCH "CORE=([^\n]*)" _ "${_run}")
        set(${out_core} "${CMAKE_MATCH_1}" PARENT_SCOPE)
        string(REGEX MATCH "SELF=([^\n]*)" _ "${_run}")
        set(${out_self} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endfunction()

    set(_base ${ARGN})
    _g2g_abi_probe("${_base}" _core _self)
    message(STATUS "[gpusim2grid] Eigen ABI tag (align;version;sse;avx;avx2;avx512;fma;neon): "
                   "lightsim2grid_core [${_core}], this compiler [${_self}]")

    set(_flags "")
    if(NOT _core STREQUAL _self)
        # positions in the tag: 0 align, 1 version, 2 sse, 3 avx, 4 avx2, 5 avx512, 6 fma, 7 neon
        foreach(_f "5;avx512f" "6;fma" "4;avx2" "3;avx")
            list(GET _f 0 _i)
            list(GET _f 1 _isa)
            list(GET _core ${_i} _c)
            list(GET _self ${_i} _s)
            if(_c AND NOT _s)
                list(APPEND _flags "-m${_isa}")
            elseif(_s AND NOT _c)
                list(APPEND _flags "-mno-${_isa}")
            endif()
        endforeach()
        if(_flags)
            _g2g_abi_probe("${_base};${_flags}" _core2 _self2)
        else()
            set(_self2 "${_self}")
        endif()
        if(NOT _core STREQUAL _self2)
            message(FATAL_ERROR
                "[gpusim2grid] this compiler cannot match lightsim2grid_core's Eigen "
                "SIMD/alignment settings (align;version;sse;avx;avx2;avx512;fma;neon): "
                "lightsim2grid_core [${_core}], gpusim2grid [${_self2}] with '${_flags}'. "
                "Loading both would corrupt the heap; build lightsim2grid and gpusim2grid "
                "with the same Eigen and -march flags (lightsim2grid's docs/solver_plugin.rst, "
                "\"Matching build flags\").")
        endif()
        message(STATUS "[gpusim2grid] matching lightsim2grid_core's Eigen SIMD settings "
                       "with: ${_flags}")
        target_compile_options(${target} PUBLIC
            "$<$<COMPILE_LANGUAGE:CXX>:${_flags}>")
    endif()

    # nvcc's host compiler (which may not be CMAKE_CXX_COMPILER) must agree too.
    list(GET _core 0 _align)
    list(GET _core 3 _avx)
    list(GET _core 4 _avx2)
    list(GET _core 5 _avx512)
    list(GET _core 6 _fma)
    set(_xflags "")
    foreach(_fl IN LISTS _base _flags)
        list(APPEND _xflags "-Xcompiler=${_fl}")
    endforeach()
    try_compile(_g2g_abi_cuda_ok "${CMAKE_CURRENT_BINARY_DIR}/ls2g_abi_probe_cuda"
        "${_dir}/ls2g_abi_probe_cuda.cu"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${lightsim2grid_core_INCLUDE_DIRS}"
                    "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        COMPILE_DEFINITIONS ${_xflags} -DG2G_EXPECT_ALIGN=${_align}
            -DG2G_EXPECT_AVX=${_avx} -DG2G_EXPECT_AVX2=${_avx2}
            -DG2G_EXPECT_AVX512=${_avx512} -DG2G_EXPECT_FMA=${_fma}
        OUTPUT_VARIABLE _out)
    if(NOT _g2g_abi_cuda_ok)
        message(FATAL_ERROR
            "[gpusim2grid] nvcc's host compiler does not give Eigen lightsim2grid_core's "
            "SIMD/alignment settings [${_core}] (extra flags: '${_xflags}'). Use the same "
            "host compiler as CMAKE_CXX_COMPILER (CMAKE_CUDA_HOST_COMPILER).\n${_out}")
    endif()
    if(_flags)
        set(_xflags "")
        foreach(_fl IN LISTS _flags)
            list(APPEND _xflags "-Xcompiler=${_fl}")
        endforeach()
        target_compile_options(${target} PUBLIC
            "$<$<COMPILE_LANGUAGE:CUDA>:${_xflags}>")
    endif()
endfunction()
