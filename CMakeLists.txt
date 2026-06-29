cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Read version from VERSION.txt
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" VERSION_FROM_FILE)
string(STRIP "${VERSION_FROM_FILE}" VERSION_FROM_FILE)
# Extract major.minor.patch for CMake project version
string(REGEX MATCH "^[0-9]+\.[0-9]+\.[0-9]+" PROJECT_VERSION_ONLY "${VERSION_FROM_FILE}")

project(Choreo VERSION ${PROJECT_VERSION_ONLY})

# Function to check compiler compatibility
function(check_compiler_compatibility)
  set(COMPATIBLE_COMPILER FALSE)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC detected
    message(STATUS "Detected GCC version: ${CMAKE_CXX_COMPILER_VERSION}")

    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
      set(COMPATIBLE_COMPILER TRUE PARENT_SCOPE)
      message(STATUS "OK: GCC 9.0+ with C++17 support confirmed")
    else()
      message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old. GCC 9.0+ required for full C++17 support.")
    endif()

  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang detected (includes AppleClang)
    message(STATUS "Detected Clang version: ${CMAKE_CXX_COMPILER_VERSION}")

    # Clang has good C++17 support since version 5.0, but we'll be conservative
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "5.0")
      set(COMPATIBLE_COMPILER TRUE PARENT_SCOPE)
      message(STATUS "OK: Clang 5.0+ with C++17 support confirmed")
    else()
      message(FATAL_ERROR "Clang ${CMAKE_CXX_COMPILER_VERSION} is too old. Clang 5.0+ recommended for C++17 support.")
    endif()

  else()
    # Other compilers (MSVC, Intel, etc.)
    message(WARNING "Unknown compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(WARNING "C++17 support cannot be automatically verified.")
    message(WARNING "Please ensure your compiler supports C++17.")
    set(COMPATIBLE_COMPILER TRUE PARENT_SCOPE) # Assume it works, but warn user
  endif()
endfunction()

# Check compiler compatibility
check_compiler_compatibility()

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find threads package (not needed for Emscripten)
if(NOT EMSCRIPTEN)
  find_package(Threads REQUIRED)
endif()

# use ccache if exists
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found, compilation will not be cached")
endif()

option(CHOREO_DEFAULT_TARGET "cute")

# allow diagnostics-color
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fdiagnostics-color)
endif()

# ============================
# Cascading selection + validation
# ============================

if(EMSCRIPTEN)
  set(ENABLE_CUDA OFF CACHE BOOL "" FORCE)
  set(ENABLE_CUTE OFF CACHE BOOL "" FORCE)
  set(ENABLE_HIP OFF CACHE BOOL "" FORCE)
  set(CHOREO_DEFAULT_TARGET "mock" CACHE STRING "" FORCE)
endif()

# Track if user explicitly set via -D before we create cache entries
set(_enable_cuda_user_set FALSE)
set(_enable_cute_user_set FALSE)
foreach(var IN ITEMS ENABLE_CUDA ENABLE_CUTE)
  foreach(arg IN LISTS CMAKE_CACHE_ARGS)
    if("${arg}" MATCHES "^${var}=")
      set(_${var_lower}_user_set TRUE)
      break()
    endif()
  endforeach()
endforeach()

# Public cache options (defaults: CUDA ON, CUTE OFF)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUTE "Enable CUTE/CUTLASS support (requires CUDA)" ON)
option(CHOREO_FAST_COMPILE_DEFAULT "Default --fast-compile to ON for faster nvcc builds" OFF)

# ============================
# Project selection (semicolon-separated list, like LLVM_ENABLE_PROJECTS)
# Available: choreo, coir, co-mock, co-web
# ============================
set(CROQ_PROJECT "choreo;coir" CACHE STRING
  "Semicolon-separated list of projects to build (choreo;coir;co-mock;co-web)")

list(FIND CROQ_PROJECT "choreo" _idx)
if(_idx GREATER_EQUAL 0)
  set(CHOREO_BUILD_CHOREO ON)
else()
  set(CHOREO_BUILD_CHOREO OFF)
endif()

list(FIND CROQ_PROJECT "coir" _idx)
if(_idx GREATER_EQUAL 0)
  set(CHOREO_BUILD_COIR ON)
else()
  set(CHOREO_BUILD_COIR OFF)
endif()

list(FIND CROQ_PROJECT "co-mock" _idx)
if(_idx GREATER_EQUAL 0)
  set(CHOREO_BUILD_CO_MOCK ON)
else()
  set(CHOREO_BUILD_CO_MOCK OFF)
endif()

list(FIND CROQ_PROJECT "co-web" _idx)
if(_idx GREATER_EQUAL 0)
  set(CHOREO_BUILD_CO_WEB ON)
else()
  set(CHOREO_BUILD_CO_WEB OFF)
endif()

# ============================
# Target selection (semicolon-separated list)
# "all" builds every backend under lib/Target/.
# Directory names (gpu, cpu, amdgpu, hetero, ...) and aliases defined in
# each backend's target.aliases file are accepted. Unknown names are
# silently ignored.
# ============================
set(CROQ_TARGET "all" CACHE STRING
  "Semicolon-separated list of target backends to build (default: all)")

# Resolve CROQ_TARGET to directory-level backend names.
# Each backend directory under lib/Target/ may contain a target.aliases file
# listing additional user-facing names (one per line), e.g. "cute" -> GPU.
# This is auto-discovered so no internal names leak into root CMakeLists.txt.
function(_croq_resolve_targets INPUT_LIST OUT_VAR)
  # Discover all backend directories
  file(GLOB _all_backends
    RELATIVE "${CMAKE_SOURCE_DIR}/lib/Target"
    "${CMAKE_SOURCE_DIR}/lib/Target/*/CMakeLists.txt")
  set(_backend_dirs "")
  foreach(_b ${_all_backends})
    get_filename_component(_dir "${_b}" DIRECTORY)
    list(APPEND _backend_dirs "${_dir}")
  endforeach()

  # Build alias -> directory mapping
  set(_alias_map "")
  foreach(_dir ${_backend_dirs})
    string(TOLOWER "${_dir}" _dir_lower)
    list(APPEND _alias_map "${_dir_lower}=${_dir}")
    set(_alias_file "${CMAKE_SOURCE_DIR}/lib/Target/${_dir}/target.aliases")
    if(EXISTS "${_alias_file}")
      file(STRINGS "${_alias_file}" _aliases)
      foreach(_a ${_aliases})
        string(STRIP "${_a}" _a)
        if(_a AND NOT _a MATCHES "^#")
          string(TOLOWER "${_a}" _a_lower)
          list(APPEND _alias_map "${_a_lower}=${_dir}")
        endif()
      endforeach()
    endif()
  endforeach()

  set(_resolved "")
  foreach(_t ${INPUT_LIST})
    string(TOLOWER "${_t}" _t)
    if(_t STREQUAL "all")
      set(_resolved "${_backend_dirs}")
      break()
    endif()
    set(_found FALSE)
    foreach(_entry ${_alias_map})
      string(REGEX MATCH "^([^=]+)=(.*)" _ "${_entry}")
      if("${CMAKE_MATCH_1}" STREQUAL "${_t}")
        list(APPEND _resolved "${CMAKE_MATCH_2}")
        set(_found TRUE)
        break()
      endif()
    endforeach()
    if(NOT _found)
      message(STATUS "CROQ_TARGET: unknown target '${_t}', ignoring.")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _resolved)
  set(${OUT_VAR} "${_resolved}" PARENT_SCOPE)
endfunction()

_croq_resolve_targets("${CROQ_TARGET}" CROQ_ENABLED_TARGETS)
message(STATUS "CROQ_TARGET = ${CROQ_TARGET}")
message(STATUS "Enabled targets: ${CROQ_ENABLED_TARGETS}")

# Per-target booleans (CROQ_TARGET_<UPPER>) for use in toolchain gating.
# Discovered from lib/Target/ directories so no internal names appear here.
file(GLOB _all_target_cmakelists
  RELATIVE "${CMAKE_SOURCE_DIR}/lib/Target"
  "${CMAKE_SOURCE_DIR}/lib/Target/*/CMakeLists.txt")
foreach(_f ${_all_target_cmakelists})
  get_filename_component(_d "${_f}" DIRECTORY)
  string(TOUPPER "${_d}" _d_upper)
  set(CROQ_TARGET_${_d_upper} OFF)
endforeach()
foreach(_backend ${CROQ_ENABLED_TARGETS})
  string(TOUPPER "${_backend}" _backend_upper)
  set(CROQ_TARGET_${_backend_upper} ON)
endforeach()

# Allow users to provide preferred defaults (empty by default)
#set(DEFAULT_CUDA_HOME "${CMAKE_SOURCE_DIR}/extern/portable_cuda" CACHE PATH "Optional default CUDA path (used if CLI and ENV are invalid)")
set(DEFAULT_CUDA_HOME "/usr/local/cuda" CACHE PATH "Optional default CUDA path (used if CLI and ENV are invalid)")

# Skip CUDA/CUTE detection entirely when GPU target is not enabled
if(NOT CROQ_TARGET_GPU)
  set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
  set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE/CUTLASS support" FORCE)
endif()

# Skip HIP/ROCm detection when AMDGPU target is not enabled
if(NOT CROQ_TARGET_AMDGPU)
  set(ENABLE_HIP OFF CACHE BOOL "Enable HIP/ROCm support" FORCE)
endif()

# Find CUDA package to get version info
if(NOT EMSCRIPTEN AND CROQ_TARGET_GPU)
  find_package(CUDAToolkit QUIET)
endif()

if(CUDAToolkit_FOUND)
  set(_cuda_with_cute4_support TRUE)
  set(DEFAULT_CUTE_HOME "")
  # Determine CUTLASS path based on CUDA version
  if(CUDAToolkit_VERSION VERSION_LESS 12.0)
    # CUDA 11.x or earlier
    set(_cuda_with_cute4_support FALSE)
  else()
    # CUDA 12.x or later
    set(DEFAULT_CUTE_HOME "${CMAKE_SOURCE_DIR}/extern/cutlass")
  endif()
endif()

set(CUDA_HOME "")
set(CUTE_HOME "")

# ----- helpers -----
function(_looks_like_valid_cuda ROOT OUT_VALID)
  set(ok FALSE)
  if(ROOT AND EXISTS "${ROOT}")
    if(EXISTS "${ROOT}/bin/nvcc" OR
       EXISTS "${ROOT}/bin/nvcc.exe" OR
       EXISTS "${ROOT}/include/cuda.h")
      set(ok TRUE)
    endif()
  endif()
  set(${OUT_VALID} ${ok} PARENT_SCOPE)
endfunction()

function(_looks_like_valid_cute ROOT OUT_VALID)
  set(ok FALSE)
  if(ROOT AND EXISTS "${ROOT}")
    if(EXISTS "${ROOT}/include/cute/tensor.hpp" OR
       EXISTS "${ROOT}/include/cute/cute.hpp"   OR
       EXISTS "${ROOT}/include/cutlass/cutlass.h")
      set(ok TRUE)
    endif()
  endif()
  set(${OUT_VALID} ${ok} PARENT_SCOPE)
endfunction()

# ============================
# CUDA: cascade & validate
# ============================
set(_cuda_candidates "")
if(NOT "${CUDA_HOME}" STREQUAL "")
  # 1) CLI cache variable
  set(_cuda_candidates "${CUDA_HOME}")
elseif(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
  # 2) Environment variable
  set(_cuda_candidates "$ENV{CUDA_HOME}")
elseif(NOT "${DEFAULT_CUDA_HOME}" STREQUAL "")
  # 3) Optional/default fallbacks (edit these as you like)
  set(_cuda_candidates "${DEFAULT_CUDA_HOME}")
endif()

set(_cuda_root "")
set(_cuda_root_valid FALSE)
if(NOT "${_cuda_candidates}" STREQUAL "")
  _looks_like_valid_cuda("${_cuda_candidates}" _ok)
  if(_ok)
    set(_cuda_root "${_cuda_candidates}")
    set(_cuda_root_valid TRUE)
  endif()
endif()

# Auto behaviour vs explicit enable
if(NOT _cuda_root_valid)
  if(_enable_cuda_user_set)
    # User forced ON, but nothing valid
    message(FATAL_ERROR
      "ENABLE_CUDA=ON but no valid CUDA found.\n"
      "Tried (in order):\n  -DCUDA_HOME='${CUDA_HOME}'\n  ENV{CUDA_HOME}='$ENV{CUDA_HOME}'\n"
      "  DEFAULT_CUDA_HOME='${DEFAULT_CUDA_HOME}' and common defaults.\n"
      "Please set -DCUDA_HOME to a valid CUDA install or -DENABLE_CUDA=OFF.")
  else()
    # Not explicitly set force OFF
    if(ENABLE_CUDA)
      message(STATUS "No valid CUDA found. Auto-disabling ENABLE_CUDA.")
      set(ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
    endif()
  endif()
endif()

message(STATUS "ENABLE_CUDA = ${ENABLE_CUDA}")
if(ENABLE_CUDA)
  message(STATUS "Using CUDA from: ${_cuda_root}.")
  find_program(NVCC nvcc HINTS ${_cuda_root}/bin)
	execute_process(
		COMMAND ${NVCC} --version
		OUTPUT_VARIABLE NVCC_OUT
		OUTPUT_STRIP_TRAILING_WHITESPACE
		)

	string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _ "${NVCC_OUT}")
	set(CUDA_VERSION_MAJOR "${CMAKE_MATCH_1}")
	set(CUDA_VERSION_MINOR "${CMAKE_MATCH_2}")
endif()

# ============================
# CUTE/CUTLASS: cascade & validate (requires CUDA)
# ============================
set(_cute_candidates "")
if(NOT "${CUTE_HOME}" STREQUAL "")
  set(_cute_candidates "${CUTE_HOME}")
elseif(DEFINED ENV{CUTE_HOME} AND NOT "$ENV{CUTE_HOME}" STREQUAL "")
  set(_cute_candidates "$ENV{CUTE_HOME}")
elseif(NOT "${DEFAULT_CUTE_HOME}" STREQUAL "")
  set(_cute_candidates "${DEFAULT_CUTE_HOME}")
endif()

set(_cute_root "")
set(_cute_root_valid FALSE)
if(NOT "${_cute_candidates}" STREQUAL "")
  _looks_like_valid_cute("${_cute_candidates}" _ok)
  if(_ok)
    set(_cute_root "${_cute_candidates}")
    set(_cute_root_valid TRUE)
  endif()
endif()

if(ENABLE_CUTE)
  if(NOT ENABLE_CUDA OR NOT _cuda_root_valid)
    if(_enable_cute_user_set)
      message(FATAL_ERROR
        "ENABLE_CUTE=ON requires a working CUDA toolchain.\n"
        "Fix CUDA first (set a valid -DCUDA_HOME or ENV{CUDA_HOME}).")
    else()
      message(STATUS "No valid CUDA found. Auto-disabling ENABLE_CUTE.")
      set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE support" FORCE)
    endif()
  elseif(NOT _cute_root_valid)
    if(NOT _cuda_with_cute4_support)
      message(STATUS "CUDA version too low for CUTE 4. Auto-disabling ENABLE_CUTE.")
      set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE/CUTLASS support" FORCE)
    else()
      include(${CMAKE_SOURCE_DIR}/cmake/CutlassBootstrap.cmake)
      if(NOT _cute_root_valid)
        message(STATUS "CUTLASS not available. Auto-disabling ENABLE_CUTE.")
        set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE/CUTLASS support" FORCE)
      endif()
    endif()
  endif()
endif()

message(STATUS "ENABLE_CUTE = ${ENABLE_CUTE}")
if(ENABLE_CUTE)
  message(STATUS "Using CUTE/CUTLASS from: ${_cute_root}")
endif()

# ============================
# HIP/ROCm: cascade & validate (only when AMDGPU target enabled)
# ============================
option(ENABLE_HIP "Enable HIP/ROCm support for AMD GPUs" ON)

set(DEFAULT_ROCM_HOME "/opt/rocm" CACHE PATH "Optional default ROCm path")

if(NOT CROQ_TARGET_AMDGPU)
  set(ENABLE_HIP OFF CACHE BOOL "" FORCE)
endif()

function(_looks_like_valid_rocm ROOT OUT_VALID)
  set(ok FALSE)
  if(ROOT AND EXISTS "${ROOT}")
    if(EXISTS "${ROOT}/bin/hipcc" AND
       EXISTS "${ROOT}/include/hip/hip_runtime.h")
      set(ok TRUE)
    endif()
  endif()
  set(${OUT_VALID} ${ok} PARENT_SCOPE)
endfunction()

set(_rocm_candidates "")
if(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
  set(_rocm_candidates "$ENV{ROCM_HOME}")
elseif(NOT "${DEFAULT_ROCM_HOME}" STREQUAL "")
  set(_rocm_candidates "${DEFAULT_ROCM_HOME}")
endif()

set(_rocm_root "")
set(_rocm_root_valid FALSE)
if(NOT "${_rocm_candidates}" STREQUAL "")
  _looks_like_valid_rocm("${_rocm_candidates}" _ok)
  if(_ok)
    set(_rocm_root "${_rocm_candidates}")
    set(_rocm_root_valid TRUE)
  endif()
endif()

if(NOT _rocm_root_valid)
  if(ENABLE_HIP)
    message(STATUS "No valid ROCm found. Auto-disabling ENABLE_HIP.")
    set(ENABLE_HIP OFF CACHE BOOL "Enable HIP/ROCm support" FORCE)
  endif()
endif()

message(STATUS "ENABLE_HIP = ${ENABLE_HIP}")
if(ENABLE_HIP)
  message(STATUS "Using ROCm from: ${_rocm_root}")
  find_program(HIPCC hipcc HINTS ${_rocm_root}/bin)
  execute_process(
    COMMAND ${HIPCC} --version
    OUTPUT_VARIABLE HIPCC_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "HIP version: ([0-9]+)\\.([0-9]+)" _ "${HIPCC_OUT}")
  set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(HIP_VERSION_MINOR "${CMAKE_MATCH_2}")
  message(STATUS "HIP version: ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")
endif()

# Dummy target to make configuration succeed when appropriate
add_library(dummy INTERFACE)


# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/extern")

# Flex and Bison: find in toolchain/system, or auto-build from source
if(NOT EMSCRIPTEN)
  include(${CMAKE_SOURCE_DIR}/cmake/FlexBisonBootstrap.cmake)
else()
  include(${CMAKE_SOURCE_DIR}/cmake/EmscriptenBootstrap.cmake)
endif()

# CUDA_HOME as a cache variable with environment fallback
if(DEFINED ENV{CUDA_HOME})
  set(CUDA_HOME $ENV{CUDA_HOME} CACHE PATH "Path to CUDA installation")
else()
  set(CUDA_HOME "/usr/local/cuda" CACHE PATH "Path to CUDA installation")
endif()

add_compile_options(-Wall -Wextra)

if(EMSCRIPTEN)
  # Choreo is developed against GCC; suppress Clang-specific pedantic warnings
  # that do not indicate actual bugs when cross-compiling to WASM.
  add_compile_options(
    -Wno-delete-non-virtual-dtor
    -Wno-unused-private-field
    -Wno-constant-conversion
    -Wno-overloaded-virtual
    -Wno-unused-lambda-capture
    -Wno-inconsistent-missing-override
    -Wno-mismatched-tags
    -Wno-tautological-constant-out-of-range-compare
    -Wno-unused-but-set-variable
    -Wno-unused-function
    -Wno-parentheses
    "-Wno-#warnings"
  )
endif()

option(PUBLIC_PACKAGE "Build for Public Release Package")

if (PUBLIC_PACKAGE)
  add_compile_options(-D__CHOREO_INSTALLATION_PACKAGE__)
endif()

if (CHOREO_FAST_COMPILE_DEFAULT)
  add_compile_options(-DCHOREO_FAST_COMPILE_DEFAULT=1)
endif()

# do not warn about strict-aliasing
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-fno-strict-aliasing)
endif()

# Include generated files
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/runtime)

# add code sources and rules
add_subdirectory(lib)

# additional tools (co-mock, etc.)
add_subdirectory(tools)

if(NOT EMSCRIPTEN)

set(EXTERN_LIBS
  stdc++fs
)

if(ENABLE_CUDA)
  find_library(CUDART_LIB cudart HINTS ${_cuda_root}/lib64 ${_cuda_root}/targets/x86_64-linux/lib)
  if(CUDART_LIB)
    list(APPEND EXTERN_LIBS ${CUDART_LIB})
  else()
    message(WARNING "CUDA runtime library (libcudart) not found. Auto-detection may not work.")
  endif()
endif()

if(ENABLE_HIP)
  find_library(AMDHIP64_LIB amdhip64 HINTS ${_rocm_root}/lib ${_rocm_root}/lib64)
  if(AMDHIP64_LIB)
    list(APPEND EXTERN_LIBS ${AMDHIP64_LIB})
  else()
    message(WARNING "HIP runtime library (libamdhip64) not found.")
  endif()
endif()

if(CHOREO_BUILD_CHOREO)
  add_executable(copp tools/copp/choreo_preprocess.cpp)
  add_executable(choreo tools/choreo/choreo_main.cpp)

  add_dependencies(choreo parser_deps)
  add_dependencies(copp parser_deps)

  target_link_libraries(choreo PRIVATE
    parse core codegen support pp
    -Wl,--whole-archive targets -Wl,--no-whole-archive
    ${EXTERN_LIBS} Threads::Threads
  )

  target_link_libraries(copp PRIVATE
    parse core codegen support pp
    -Wl,--whole-archive targets -Wl,--no-whole-archive
    ${EXTERN_LIBS} Threads::Threads
  )
endif()

endif() # NOT EMSCRIPTEN

if(NOT EMSCRIPTEN AND CHOREO_BUILD_CHOREO)
# ============================
# Choreo SDK library
# ============================
set(CHOREO_SDK_VERSION "${VERSION_FROM_FILE}")

add_library(choreo-dev STATIC
  $<TARGET_OBJECTS:parse>
  $<TARGET_OBJECTS:core>
  $<TARGET_OBJECTS:codegen>
  $<TARGET_OBJECTS:support>
  $<TARGET_OBJECTS:pp>
  $<TARGET_OBJECTS:targets>
  tools/sdk/public_api.cpp
)

target_compile_definitions(choreo-dev PRIVATE
  CHOREO_SDK_VERSION="${CHOREO_SDK_VERSION}"
)

target_include_directories(choreo-dev PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tools/sdk>
  $<INSTALL_INTERFACE:include/choreo>
)

set_target_properties(choreo-dev PROPERTIES
  EXCLUDE_FROM_ALL TRUE
  OUTPUT_NAME choreo-dev
)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/VERSION.txt")
  message(FATAL_ERROR "VERSION.txt not found!")
endif()

# ============================
# Install rules
# ============================
install(FILES ${CMAKE_SOURCE_DIR}/runtime/choreo.h
  DESTINATION include
  COMPONENT rt_headers
)

install(TARGETS choreo copp
  RUNTIME DESTINATION bin
  COMPONENT executables
)

if(CHOREO_BUILD_CO_MOCK)
  install(TARGETS co-mock
    RUNTIME DESTINATION bin
    COMPONENT executables
  )
endif()

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
    ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
    ${CMAKE_BINARY_DIR}/parser.tab.hh
  DESTINATION include/choreo
  COMPONENT sdk_headers
)

install(FILES ${CMAKE_SOURCE_DIR}/tools/sdk/choreo_sdk.hpp
  DESTINATION include/choreo
  COMPONENT sdk_headers
)

install(TARGETS choreo-dev ARCHIVE
  DESTINATION lib
  COMPONENT libraries
)

include(CMakePackageConfigHelpers)
set(CHOREO_INSTALL_INCLUDEDIR include)
set(CHOREO_INSTALL_LIBDIR lib)

configure_package_config_file(
  ${CMAKE_SOURCE_DIR}/cmake/ChoreoConfig.cmake.in
  ${CMAKE_BINARY_DIR}/ChoreoConfig.cmake
  INSTALL_DESTINATION lib/cmake/Choreo
  PATH_VARS CHOREO_INSTALL_INCLUDEDIR CHOREO_INSTALL_LIBDIR
)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/ChoreoConfigVersion.cmake.in
  ${CMAKE_BINARY_DIR}/ChoreoConfigVersion.cmake
  @ONLY
)

install(
  FILES
    ${CMAKE_BINARY_DIR}/ChoreoConfig.cmake
    ${CMAKE_BINARY_DIR}/ChoreoConfigVersion.cmake
  DESTINATION lib/cmake/Choreo
  COMPONENT libraries
)

# --- CPack setup ---
file(READ "${CMAKE_SOURCE_DIR}/VERSION.txt" CPACK_PACKAGE_VERSION)
string(STRIP "${CPACK_PACKAGE_VERSION}" CPACK_PACKAGE_VERSION)

set(CPACK_PACKAGE_NAME "choreo")
set(CPACK_PACKAGE_VENDOR "Choreo Developers")
set(CPACK_PACKAGE_CONTACT "https://github.com/tbd/choreo/issues")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/package")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_GENERATOR "DEB;TGZ;ZIP")

include(CPack)

configure_file(${CMAKE_SOURCE_DIR}/cmake/CPackCompiler.cmake.in
               ${CMAKE_BINARY_DIR}/CPackCompiler.cmake @ONLY)

configure_file(${CMAKE_SOURCE_DIR}/cmake/CPackSDK.cmake.in
               ${CMAKE_BINARY_DIR}/CPackSDK.cmake @ONLY)

option(STANDALONE "Build standalone package (contains extern)" OFF)
if(STANDALONE)
  set(CHOREO_STANDALONE_PACKAGE ON)
  message(STATUS "Building STANDALONE package: extern/bin, extern/include, extern/lib, extern/shared, and are all contained.")
else()
  set(CHOREO_STANDALONE_PACKAGE OFF)
  message(STATUS "Building CORE package: Only contains the main program and header files, excluding extern.")
endif()

if(CHOREO_STANDALONE_PACKAGE)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27)")
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/bin/ DESTINATION extern/bin USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/include/ DESTINATION extern/include USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/lib/ DESTINATION extern/lib USE_SOURCE_PERMISSIONS)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/extern/shared/ DESTINATION extern/shared USE_SOURCE_PERMISSIONS)
endif()

add_custom_target(package-compiler
  DEPENDS choreo copp
  COMMAND ${CMAKE_CPACK_COMMAND}
    -D CPACK_PROJECT_CONFIG_FILE=${CMAKE_BINARY_DIR}/CPackCompiler.cmake
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Building standalone compiler package"
)

add_custom_target(package-sdk
  DEPENDS choreo-dev
  COMMAND ${CMAKE_CPACK_COMMAND}
    -D CPACK_PROJECT_CONFIG_FILE=${CMAKE_BINARY_DIR}/CPackSDK.cmake
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Building SDK package (headers + static lib)"
)

endif() # NOT EMSCRIPTEN AND CHOREO_BUILD_CHOREO
