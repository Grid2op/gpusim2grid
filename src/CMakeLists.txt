# ---------------------------------------------------------------------------
# Eigen resolution
#
# Prefer the Eigen that ships inside lightsim2grid_core's install tree: building
# gpusim2grid against the *same* Eigen lightsim2grid uses avoids subtle ABI /
# version mismatches when the two exchange Eigen types in C++. Fall back to the
# vendored copy (src/eigen/) when lightsim2grid_core is not discoverable.
#
# LIGHTSIM2GRID_CMAKE_DIR is auto-detected by asking the Python interpreter
# CMake is configuring against (Python_EXECUTABLE, resolved in the top-level
# CMakeLists.txt) for lightsim2grid.get_cmake_dir(). lightsim2grid is a
# build-system requirement (see pyproject.toml [build-system].requires), so it
# is importable there without any extra setup: plain `pip install .` /
# `uv pip install .` is enough, no CMAKE_ARGS needed.
#
# Override only if you need a *different* lightsim2grid install than the one
# on Python_EXECUTABLE's path:
#   CMAKE_ARGS="-DLIGHTSIM2GRID_CMAKE_DIR=$(python -c \
#     'import lightsim2grid; print(lightsim2grid.get_cmake_dir())')"
# ---------------------------------------------------------------------------
if(NOT DEFINED LIGHTSIM2GRID_CMAKE_DIR OR LIGHTSIM2GRID_CMAKE_DIR STREQUAL "")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import lightsim2grid; print(lightsim2grid.get_cmake_dir())"
        OUTPUT_VARIABLE LIGHTSIM2GRID_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _ls2g_cmake_dir_result
        ERROR_QUIET
    )
    if(NOT _ls2g_cmake_dir_result EQUAL 0)
        set(LIGHTSIM2GRID_CMAKE_DIR "")
    else()
        message(STATUS "[gpusim2grid] auto-detected lightsim2grid cmake dir: "
                       "${LIGHTSIM2GRID_CMAKE_DIR}")
    endif()
endif()

find_package(lightsim2grid_core CONFIG QUIET
    HINTS "${LIGHTSIM2GRID_CMAKE_DIR}")

if(lightsim2grid_core_FOUND)
    message(STATUS "[gpusim2grid] lightsim2grid_core found; using its Eigen "
                   "(${lightsim2grid_core_INCLUDE_DIRS})")
    include_directories("${lightsim2grid_core_INCLUDE_DIRS}")
    set(GPUSIM2GRID_LS2G_CORE_FOUND TRUE CACHE INTERNAL
        "lightsim2grid_core was found at configure time")
else()
    message(STATUS "[gpusim2grid] lightsim2grid_core not found; using vendored "
                   "Eigen (src/eigen/)")
    include_directories("eigen")
endif()

add_subdirectory(_cpp)
