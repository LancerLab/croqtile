cmake_minimum_required(VERSION 3.22)
project(Choreo VERSION 0.0.2)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define package version
# set(CHOREO_PACKAGE_VERSION "core" CACHE STRING "Choose the package version: standalone or core")

# use ccache if exists
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found, compilation will not be cached")
endif()

# allow diagnostics-color
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fdiagnostics-color)
endif()

# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/tools")

set(SYMBOLIC_DIR "${CMAKE_SOURCE_DIR}/extern/ginac")
set(CLN_DIR "${SYMBOLIC_DIR}/cln-1.3.7/install")
set(GINAC_DIR "${SYMBOLIC_DIR}/ginac-1.8.7/install")

set(FLEX_PATH "${TOOLCHAIN_DIR}/bin/flex")
set(BISON_PATH "${TOOLCHAIN_DIR}/bin/bison")
set(BISON_ENV "BISON_PKGDATADIR=${TOOLCHAIN_DIR}/shared/bison/")

add_compile_options(-Wall -Wextra -g  -D__CHOREO_FACTOR_DIR__="${TOOLCHAIN_DIR}" -D__CHOREO_CUDA_DIR__="${TOOLCHAIN_DIR}" -D__CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}")

option(PUBLIC_PACKAGE "Build for Public Release Package")

if (PUBLIC_PACKAGE)
  add_compile_options(-D__CHOREO_INSTALLATION_PACKAGE__)
endif()

# do not warn about strict-aliasing
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-fno-strict-aliasing)
endif()

# Include generated files
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CLN_DIR}/include ${GINAC_DIR}/include)

# add code sources and rules
add_subdirectory(lib)

# preprocessor
add_executable(copp
  utils/choreo_preprocess.cpp
)

# Add the main executable
add_executable(choreo
  utils/choreo_main.cpp
)
target_include_directories(choreo PRIVATE ${CMAKE_SOURCE_DIR})

# Link the library to the executable
target_link_libraries(choreo PRIVATE
  parser_obj
  scanner_obj
  cl_obj
  codegen
  ${GINAC_DIR}/lib/libginac.a
  ${CLN_DIR}/lib/libcln.a
  gmp
)

target_link_libraries(copp PRIVATE
  parser_obj
  scanner_obj
  codegen
  ${GINAC_DIR}/lib/libginac.a
  ${CLN_DIR}/lib/libcln.a
  gmp
)

# message(STATUS "CHOREO_PACKAGE_VERSION is set to: ${CHOREO_PACKAGE_VERSION}")

# Check if VERSION.txt exists
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/VERSION.txt")
  message(FATAL_ERROR "VERSION.txt not found!")
endif()

# Read the version from VERSION.txt
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" CPACK_PACKAGE_VERSION)
# Strip any leading/trailing whitespace or newlines
string(STRIP "${CPACK_PACKAGE_VERSION}" CPACK_PACKAGE_VERSION)

# Minimum required glibc version
set(MIN_GLIBC_VERSION "2.27")

# Create the DEBIAN/preinst script
file(WRITE ${CMAKE_BINARY_DIR}/preinst
"#!/bin/bash

# Minimum required glibc version
MIN_GLIBC_VERSION=\"${MIN_GLIBC_VERSION}\"

# Check if 'topscc' is installed
if ! dpkg -s topscc > /dev/null 2>&1; then
    echo \"Error: The 'topscc' package is not installed. Please install it first.\"
    exit 1
fi

exit 0
")

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/preinst")
install(PROGRAMS ${CMAKE_BINARY_DIR}/preinst DESTINATION DEBIAN)

# Set package information
set(CPACK_PACKAGE_NAME "choreo")
#set(CPACK_PACKAGE_VERSION "1.0.0.alpha")
set(CPACK_PACKAGE_DESCRIPTION "Installed package for choreo")
set(CPACK_PACKAGE_VENDOR "ERA Developers")

# Set the packaging generator (DEB, TGZ, ZIP, RPM)
set(CPACK_GENERATOR "DEB;TGZ;ZIP")

# Set the maintainer and contact info
set(CPACK_PACKAGE_CONTACT "heng.shi@enflame-tech.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "heng.shi@enflame-tech.com")

set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Installed package for choreo")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27), libgmp10 (>= 6.1.2)")

# Set the install prefix
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/package")

# Install rules for core version
# install(DIRECTORY ${GINAC_DIR}/include/ginac DESTINATION include USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "*")
# install(DIRECTORY ${CLN_DIR}/include/cln DESTINATION include USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "*")
#install(DIRECTORY ${GINAC_DIR}/lib/ DESTINATION lib USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "libginac.a")
#install(DIRECTORY ${CLN_DIR}/lib/ DESTINATION lib USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "libcln.a")
install(FILES ${CMAKE_SOURCE_DIR}/utils/choreo.h DESTINATION include)
install(TARGETS choreo copp
  RUNTIME DESTINATION bin
  # mute for core version
  # LIBRARY DESTINATION lib
  # ARCHIVE DESTINATION lib
)
# install(FILES ${CMAKE_BINARY_DIR}/choreo_header.inc DESTINATION include)
# install(FILES ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc DESTINATION include)

# # Handle different package versions
# if(CHOREO_PACKAGE_VERSION STREQUAL "standalone")
#     message(STATUS "Building Standalone Version")
#     # install(DIRECTORY ${GINAC_DIR}/include DESTINATION include)
#     # install(DIRECTORY ${GINAC_DIR}/lib DESTINATION lib)
#     # install(DIRECTORY ${CLN_DIR}/include DESTINATION include)
#     # install(DIRECTORY ${CLN_DIR}/lib DESTINATION lib)
#     # install(DIRECTORY ${BISON_PATH}/include DESTINATION include)
#     # install(DIRECTORY ${BISON_PATH}/lib DESTINATION lib)
#     # Install the executable and library
#     install(TARGETS choreo copp codegen
#       RUNTIME DESTINATION bin
#       LIBRARY DESTINATION lib
#       ARCHIVE DESTINATION lib
#     )
#     # Install generated headers
#     install(FILES choreo_header.inc choreo_cuda_header.inc DESTINATION include)
#
# elseif(CHOREO_PACKAGE_VERSION STREQUAL "core")
#     message(STATUS "Building Core Version")
#     install(TARGETS choreo copp codegen
#       RUNTIME DESTINATION bin
#       LIBRARY DESTINATION lib
#       ARCHIVE DESTINATION lib
#     )
#     install(FILES choreo_header.inc choreo_cuda_header.inc DESTINATION include)
#
# else()
#     message(FATAL_ERROR "Invalid package version selected. Please choose 'standalone' or 'core'.")
# endif()

include(CPack)
