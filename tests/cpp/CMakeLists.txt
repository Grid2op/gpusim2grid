# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# C++/CUDA unit tests (GPUSIM2GRID_BUILD_TESTS=ON), run with ctest. No external
# framework: tests/cpp/check.hpp. A test needing a GPU exits 77 (skipped) without one.
# One executable per source, named after the file; sources may sit in a topic folder.
set(_g2g_cpp_tests
    test_v_groups_host test_v_groups_gather test_union_pattern test_connectivity_moves
    outer_device/test_device_features
    own_lu/test_fused_solve)
foreach(_src ${_g2g_cpp_tests})
    get_filename_component(_t ${_src} NAME)
    add_executable(${_t} ${_src}.cu)
    target_link_libraries(${_t} PRIVATE gpusim2grid_core)
    target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${_t} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
                                           BUILD_RPATH "$<TARGET_FILE_DIR:gpusim2grid_core>")
    add_test(NAME ${_t} COMMAND ${_t})
    set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE 77 LABELS cpp)
endforeach()
