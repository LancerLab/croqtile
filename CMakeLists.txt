cmake_minimum_required(VERSION 3.22)

# Read version from VERSION.txt
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" VERSION_FROM_FILE)
string(STRIP "${VERSION_FROM_FILE}" VERSION_FROM_FILE)
# Extract major.minor.patch for CMake project version
string(REGEX MATCH "^[0-9]+\.[0-9]+\.[0-9]+" PROJECT_VERSION_ONLY "${VERSION_FROM_FILE}")

project(Choreo VERSION ${PROJECT_VERSION_ONLY})

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

option(ENABLE_CUTE "Enable CuTe code generation" ON)
option(ENABLE_CUDA "Enable CUDA code generation" ON)

if (ENABLE_CUTE)
endif()

if(ENABLE_CUTE)
  add_compile_options(-D__CHOREO_BUILD_TARGET_CUTE__)
  if (NOT ENABLE_CUDA)
    message(STATUS "CuTe requires CUDA - automatically enabling CUDA support")
    set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA code generation" FORCE)
  endif()
endif()

# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/extern")

set(SYMBOLIC_DIR "${CMAKE_SOURCE_DIR}/extern/ginac")
set(CLN_DIR "${SYMBOLIC_DIR}/cln-1.3.7/install")
set(GINAC_DIR "${SYMBOLIC_DIR}/ginac-1.8.7/install")

set(FLEX_PATH "${TOOLCHAIN_DIR}/bin/flex")
set(BISON_PATH "${TOOLCHAIN_DIR}/bin/bison")
set(BISON_ENV "BISON_PKGDATADIR=${TOOLCHAIN_DIR}/shared/bison/")

# CUDA_HOME as a cache variable with environment fallback
if(DEFINED ENV{CUDA_HOME})
  set(CUDA_HOME $ENV{CUDA_HOME} CACHE PATH "Path to CUDA installation")
else()
  set(CUDA_HOME "/usr/local/cuda" CACHE PATH "Path to CUDA installation")
endif()

add_compile_options(-Wall -Wextra -g)

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
include_directories(${CMAKE_SOURCE_DIR}/runtime)
include_directories(${CLN_DIR}/include ${GINAC_DIR}/include)

# add code sources and rules
add_subdirectory(lib)

# preprocessor executable
add_executable(copp tools/copp/choreo_preprocess.cpp)
# choreo compiler executable
add_executable(choreo tools/choreo/choreo_main.cpp)

set(EXTERN_LIBS
  ${GINAC_DIR}/lib/libginac.a
  ${CLN_DIR}/lib/libcln.a
  gmp
  stdc++fs
)

# Link the library to the executable
target_link_libraries(choreo PRIVATE
  parse
  core
  codegen
  support
  ${EXTERN_LIBS}
)

target_link_libraries(copp PRIVATE
  parse
  core
  codegen
  support
  ${EXTERN_LIBS}
)

# choreo sdk library
add_library(choreo-dev STATIC
  $<TARGET_OBJECTS:parse>
  $<TARGET_OBJECTS:core>
  $<TARGET_OBJECTS:codegen>
  $<TARGET_OBJECTS:support>
  tools/sdk/public_api.cpp
)

# Exclude from default build
set_target_properties(choreo-dev PROPERTIES EXCLUDE_FROM_ALL TRUE)

# message(STATUS "CHOREO_PACKAGE_VERSION is set to: ${CHOREO_PACKAGE_VERSION}")

# Check if VERSION.txt exists
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/VERSION.txt")
  message(FATAL_ERROR "VERSION.txt not found!")
endif()


# Set the install prefix
install(FILES ${CMAKE_SOURCE_DIR}/runtime/choreo.h
  DESTINATION include
  COMPONENT rt_headers
)

install(TARGETS choreo copp
  RUNTIME DESTINATION bin
  COMPONENT executables
)

# SDK headers installation - only if SDK exists
install(DIRECTORY ${CMAKE_SOURCE_DIR}/lib/
  DESTINATION include/choreo
  COMPONENT sdk_headers
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "sym_replace.hpp" EXCLUDE
)

install(
  FILES
    ${CMAKE_BINARY_DIR}/choreo_header.inc
    ${CMAKE_BINARY_DIR}/parser.tab.hh
  DESTINATION include/choreo
  COMPONENT sdk_headers
)

install(TARGETS choreo-dev ARCHIVE
  DESTINATION lib
  COMPONENT libraries
)

# --- CPack setup ---
#set(CPACK_COMPONENTS_ALL sdk_headers libraries rt_headers executables)

# Read the version from VERSION.txt
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" CPACK_PACKAGE_VERSION)
# Strip any leading/trailing whitespace or newlines
string(STRIP "${CPACK_PACKAGE_VERSION}" CPACK_PACKAGE_VERSION)

# Common metadata
set(CPACK_PACKAGE_NAME "choreo")
set(CPACK_PACKAGE_VENDOR "ERA Developers")
set(CPACK_PACKAGE_CONTACT "heng.shi@enflame-tech.com")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "heng.shi@enflame-tech.com")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")

# Where final artifacts go
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/package")

# Install prefix inside packages
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

# Use your install() rules as the source; component-aware packaging
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

# You can leave CPACK_GENERATOR empty here and pass -G on the command,
# or pick defaults now:
set(CPACK_GENERATOR "DEB;TGZ;ZIP")

# Generate the standard CPackConfig.cmake now
include(CPack)

configure_file(${CMAKE_SOURCE_DIR}/cmake/CPackCompiler.cmake.in
               ${CMAKE_BINARY_DIR}/CPackCompiler.cmake @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/cmake/CPackSDK.cmake.in
               ${CMAKE_BINARY_DIR}/CPackSDK.cmake @ONLY)

# Set standalone packaging mode if CMake variable STANDALONE=ON
option(STANDALONE "Build standalone package (contains extern and ginac/cln)" OFF)
if(STANDALONE)
  set(CHOREO_STANDALONE_PACKAGE ON)
  message(STATUS "Building STANDALONE package: extern/bin, extern/include, extern/lib, extern/shared, and ginac/cln are all contained.")
else()
  set(CHOREO_STANDALONE_PACKAGE OFF)
  message(STATUS "Building CORE package: Only contains the main program and header files, excluding extern and ginac/cln.")
endif()

if(CHOREO_STANDALONE_PACKAGE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27), libgmp10 (>= 6.1.2)")
  install(DIRECTORY ${GINAC_DIR}/include/ DESTINATION include USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CLN_DIR}/include/ DESTINATION include USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${GINAC_DIR}/lib/ DESTINATION lib USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "libginac.a")
  install(DIRECTORY ${CLN_DIR}/lib/ DESTINATION lib USE_SOURCE_PERMISSIONS FILES_MATCHING PATTERN "libcln.a")
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/bin/ DESTINATION extern/bin USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/include/ DESTINATION extern/include USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/lib/ DESTINATION extern/lib USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/shared/ DESTINATION extern/shared USE_SOURCE_PERMISSIONS)
endif()

# Choreo Compiler Package: runtime headers + binaries
add_custom_target(package-compiler
  DEPENDS choreo copp
  COMMAND ${CMAKE_CPACK_COMMAND}
    -D CPACK_PROJECT_CONFIG_FILE=${CMAKE_BINARY_DIR}/CPackCompiler.cmake
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Building standalone compiler package"
)

# SDK Package: headers + static lib
add_custom_target(package-sdk
  DEPENDS choreo-dev
  COMMAND ${CMAKE_CPACK_COMMAND}
    -D CPACK_PROJECT_CONFIG_FILE=${CMAKE_BINARY_DIR}/CPackSDK.cmake
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Building SDK package (headers + static lib)"
)
