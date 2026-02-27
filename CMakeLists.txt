cmake_minimum_required(VERSION 3.22)

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
      message(STATUS "✓ GCC 9.0+ with C++17 support confirmed")
    else()
      message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old. GCC 9.0+ required for full C++17 support.")
    endif()

  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang detected (includes AppleClang)
    message(STATUS "Detected Clang version: ${CMAKE_CXX_COMPILER_VERSION}")

    # Clang has good C++17 support since version 5.0, but we'll be conservative
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "5.0")
      set(COMPATIBLE_COMPILER TRUE PARENT_SCOPE)
      message(STATUS "✓ Clang 5.0+ with C++17 support confirmed")
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

# Find threads package
find_package(Threads REQUIRED)

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

# Allow users to provide preferred defaults (empty by default)
#set(DEFAULT_CUDA_HOME "${CMAKE_SOURCE_DIR}/extern/portable_cuda" CACHE PATH "Optional default CUDA path (used if CLI and ENV are invalid)")
set(DEFAULT_CUDA_HOME "/usr/local/cuda" CACHE PATH "Optional default CUDA path (used if CLI and ENV are invalid)")

# Find CUDA package to get version info
find_package(CUDAToolkit)

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
  # 1) CLI cache variable
  set(_cute_candidates "${CUTE_HOME}")
elseif(DEFINED ENV{CUTE_HOME} AND NOT "$ENV{CUTE_HOME}" STREQUAL "")
  # 2) Environment variable
  set(_cute_candidates "$ENV{CUTE_HOME}")
elseif(NOT "${DEFAULT_CUTE_HOME}" STREQUAL "")
  # 3) Optional/default
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

# Enforce CUDA dependency + cascade outcome
if(ENABLE_CUTE)
  # If user wants CUTE, CUDA must be valid
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
    if(_enable_cute_user_set)
      message(FATAL_ERROR
        "ENABLE_CUTE=ON but no valid CUTE/CUTLASS found.\n"
        "Tried (in order):\n  -DCUTE_HOME='${CUTE_HOME}'\n  ENV{CUTE_HOME}='$ENV{CUTE_HOME}'\n"
        "  DEFAULT_CUTE_HOME='${DEFAULT_CUTE_HOME}'.")
    elseif(NOT _cuda_with_cute4_support)
      message(STATUS "CUDA version is low. Auto-disabling ENABLE_CUTE.")
      set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE/CUTLASS support (requires CUDA)" FORCE)
    else()
      # CUTE enabled but came from cache default? (rare)
      # Be conservative and turn it OFF if nothing valid
      message(STATUS "No valid CUTE/CUTLASS found. Auto-disabling ENABLE_CUTE.")
      set(ENABLE_CUTE OFF CACHE BOOL "Enable CUTE/CUTLASS support (requires CUDA)" FORCE)
    endif()
  endif()

  # If user provided an invalid CUTE_HOME but CUTE is OFF, warn:
  if(NOT _cute_root_valid AND (NOT "${CUTE_HOME}" STREQUAL "" OR DEFINED ENV{CUTE_HOME}))
    message(STATUS "CUTE is OFF; ignoring invalid CUTE_HOME candidates.")
  endif()
endif()

message(STATUS "ENABLE_CUTE = ${ENABLE_CUTE}")
if(ENABLE_CUTE)
  message(STATUS "Using CUTE/CUTLASS from: ${_cute_root}")
endif()

# Dummy target to make configuration succeed when appropriate
add_library(dummy INTERFACE)


# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/extern")

# flex and bison are required
set(TOOLCHAIN_BIN_DIR "${TOOLCHAIN_DIR}/bin/")
set(TOOLCHAIN_BISON_DATA_DIR "${TOOLCHAIN_DIR}/shared/bison/")

find_program(BISON_EXECUTABLE
  NAMES bison
  HINTS "${TOOLCHAIN_BIN_DIR}"
)

find_program(FLEX_EXECUTABLE
  NAMES flex
  HINTS "${TOOLCHAIN_BIN_DIR}"
)

if(NOT FLEX_EXECUTABLE OR NOT BISON_EXECUTABLE)
  message(FATAL_ERROR "Missing build tools! Bison: ${BISON_EXECUTABLE}, Flex: ${FLEX_EXECUTABLE}")
endif()

if(BISON_EXECUTABLE)
  execute_process(
    COMMAND ${BISON_EXECUTABLE} --version
    OUTPUT_VARIABLE BISON_VER_OUTPUT
  )
# Extract version string (e.g., "3.8.2")
  if(BISON_VER_OUTPUT MATCHES "bison.* ([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
    set(BISON_VERSION_FOUND ${CMAKE_MATCH_1})
    if(BISON_VERSION_FOUND VERSION_LESS "3.8")
        message(FATAL_ERROR "Bison 3.8 or higher is required. Found version: ${BISON_VERSION_FOUND} at ${BISON_EXECUTABLE}")
    endif()
  else()
    message(WARNING "Could not determine Bison version. Build may fail.")
  endif()
endif()

if(BISON_EXECUTABLE AND TOOLCHAIN_DIR)
# Check if the found bison is inside the toolchain directory
  cmake_path(IS_PREFIX TOOLCHAIN_DIR "${BISON_EXECUTABLE}" IS_LOCAL_BISON)

  if(IS_LOCAL_BISON AND EXISTS "${TOOLCHAIN_BISON_DATA_DIR}")
    set(BISON_ENV_CMD "BISON_PKGDATADIR=${TOOLCHAIN_BISON_DATA_DIR}")
    message(STATUS "Using local Bison environment with PKGDATADIR")
  endif()
endif()

if(NOT BISON_EXECUTABLE OR NOT FLEX_EXECUTABLE)
	message(FATAL_ERROR "Bison or Flex not found! Please check TOOLCHAIN_DIR or install them.")
endif()

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

# add code sources and rules
add_subdirectory(lib)

# preprocessor executable
add_executable(copp tools/copp/choreo_preprocess.cpp)
# choreo compiler executable
add_executable(choreo tools/choreo/choreo_main.cpp)

set(EXTERN_LIBS
  stdc++fs
)

# Add CUDA runtime library if CUDA is enabled
if(ENABLE_CUDA)
  # Try to find libcudart in standard CUDA locations
  find_library(CUDART_LIB cudart HINTS ${_cuda_root}/lib64 ${_cuda_root}/targets/x86_64-linux/lib)
  if(CUDART_LIB)
    list(APPEND EXTERN_LIBS ${CUDART_LIB})
  else()
    message(WARNING "CUDA runtime library (libcudart) not found. Auto-detection may not work.")
  endif()
endif()

# Link the library to the executable
target_link_libraries(choreo PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  ${EXTERN_LIBS}
  Threads::Threads
)

target_link_libraries(copp PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  ${EXTERN_LIBS}
  Threads::Threads
)

# choreo sdk library
add_library(choreo-dev STATIC
  $<TARGET_OBJECTS:parse>
  $<TARGET_OBJECTS:core>
  $<TARGET_OBJECTS:codegen>
  $<TARGET_OBJECTS:support>
  $<TARGET_OBJECTS:pp>
  $<TARGET_OBJECTS:targets>
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
    ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
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
set(CPACK_PACKAGE_VENDOR "Choreo Developers")
set(CPACK_PACKAGE_CONTACT "https://github.com/tbd/choreo/issues")
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
