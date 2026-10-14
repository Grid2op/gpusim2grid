# Version file for the cuDSS pip-wheel shim (see cudss-config.cmake): reads the
# version from cudss.h so `find_package(cudss 0.8.0)` checks the real wheel.

include("${CMAKE_CURRENT_LIST_DIR}/cudss-wheel-root.cmake")

set(PACKAGE_VERSION "0.0.0")
if(EXISTS "${_cudss_root}/include/cudss.h")
    file(STRINGS "${_cudss_root}/include/cudss.h" _cudss_ver_lines
         REGEX "^#define CUDSS_VERSION_(MAJOR|MINOR|PATCH) +[0-9]+")
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH "CUDSS_VERSION_${_part} +([0-9]+)" _m "${_cudss_ver_lines}")
        set(_cudss_${_part} "${CMAKE_MATCH_1}")
    endforeach()
    set(PACKAGE_VERSION "${_cudss_MAJOR}.${_cudss_MINOR}.${_cudss_PATCH}")
endif()

if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
