# ---------------------------------------------------------------------------
# Eigen resolution
#
# Prefer the Eigen that ships inside lightsim2grid_core's install tree: building
# gpusim2grid against the *same* Eigen lightsim2grid uses avoids subtle ABI /
# version mismatches when the two exchange Eigen types in C++. Fall back to the
# vendored copy (src/eigen/) when lightsim2grid_core is not discoverable.
#
# Build hint to locate it:
#   CMAKE_ARGS="-DLIGHTSIM2GRID_CMAKE_DIR=$(python -c \
#     'import lightsim2grid; print(lightsim2grid.get_cmake_dir())')"
# ---------------------------------------------------------------------------
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
