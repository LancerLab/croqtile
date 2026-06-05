# LLVMBootstrap.cmake
# On-demand LLVM/MLIR dependency for the CoIR tooling.
#
# Reads tools/coir/llvm-dep.conf for URL, tarball name, and MD5.
# When LLVM/MLIR is not found in extern/llvm-project/, downloads it
# automatically (or errors out if COIR_AUTO_DOWNLOAD_LLVM is OFF).
#
# Sets: MLIR_DIR, LLVM_DIR, LLVM_TABLEGEN_EXE, MLIR_TABLEGEN_EXE

set(_LLVM_ROOT "${CMAKE_SOURCE_DIR}/extern/llvm-project")

# --- Parse llvm-dep.conf ---
set(_DEP_CONF "${CMAKE_SOURCE_DIR}/tools/coir/llvm-dep.conf")
if(NOT EXISTS "${_DEP_CONF}")
  message(FATAL_ERROR
    "Missing ${_DEP_CONF} -- cannot determine LLVM source.\n"
    "This file should contain LLVM_URL, LLVM_TAR, LLVM_MD5, LLVM_SHASH.")
endif()
file(STRINGS "${_DEP_CONF}" _dep_lines REGEX "^[A-Z_]+=")
foreach(_line ${_dep_lines})
  string(REGEX MATCH "^([A-Z_]+)=(.*)" _ "${_line}")
  set(COIR_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
endforeach()

option(COIR_AUTO_DOWNLOAD_LLVM
  "Automatically download LLVM/MLIR if not found" ON)

# --- Locate or download LLVM/MLIR ---
set(_LLVM_CMAKE_DIR "${_LLVM_ROOT}/lib/cmake")

if(NOT EXISTS "${_LLVM_CMAKE_DIR}/mlir/MLIRConfig.cmake")
  if(NOT COIR_AUTO_DOWNLOAD_LLVM)
    message(FATAL_ERROR
      "LLVM/MLIR not found at ${_LLVM_ROOT}.\n"
      "CoIR tooling requires a pre-built LLVM/MLIR installation.\n"
      "Run: make setup-coir-deps\n"
      "Or set -DCOIR_AUTO_DOWNLOAD_LLVM=ON to download automatically.\n"
      "Or manually extract the LLVM tarball into extern/llvm-project/")
  endif()

  message(STATUS "CoIR: LLVM/MLIR not found, downloading (${COIR_LLVM_SHASH})...")
  set(_TAR_PATH "${CMAKE_SOURCE_DIR}/extern/${COIR_LLVM_TAR}")

  if(NOT EXISTS "${_TAR_PATH}")
    file(DOWNLOAD "${COIR_LLVM_URL}" "${_TAR_PATH}"
      SHOW_PROGRESS
      EXPECTED_MD5 "${COIR_LLVM_MD5}"
      STATUS _dl_status
      TIMEOUT 600)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      file(REMOVE "${_TAR_PATH}")
      message(FATAL_ERROR
        "Failed to download LLVM/MLIR: ${_dl_msg}\n"
        "URL: ${COIR_LLVM_URL}\n"
        "You can download manually and place at: ${_TAR_PATH}")
    endif()
  endif()

  message(STATUS "CoIR: Extracting LLVM/MLIR into ${_LLVM_ROOT}...")
  file(MAKE_DIRECTORY "${_LLVM_ROOT}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_TAR_PATH}"
    WORKING_DIRECTORY "${_LLVM_ROOT}"
    RESULT_VARIABLE _extract_result)
  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract LLVM tarball: ${_TAR_PATH}")
  endif()

  # Handle tarballs that extract with a top-level directory
  if(NOT EXISTS "${_LLVM_CMAKE_DIR}/mlir/MLIRConfig.cmake")
    file(GLOB _subdirs "${_LLVM_ROOT}/*/lib/cmake/mlir/MLIRConfig.cmake")
    if(_subdirs)
      list(GET _subdirs 0 _found_config)
      get_filename_component(_inner_root "${_found_config}" DIRECTORY)
      get_filename_component(_inner_root "${_inner_root}" DIRECTORY)
      get_filename_component(_inner_root "${_inner_root}" DIRECTORY)
      get_filename_component(_inner_root "${_inner_root}" DIRECTORY)
      file(GLOB _inner_contents "${_inner_root}/*")
      foreach(_item ${_inner_contents})
        get_filename_component(_name "${_item}" NAME)
        file(RENAME "${_item}" "${_LLVM_ROOT}/${_name}")
      endforeach()
    endif()
  endif()

  file(REMOVE "${_TAR_PATH}")

  if(NOT EXISTS "${_LLVM_CMAKE_DIR}/mlir/MLIRConfig.cmake")
    message(FATAL_ERROR
      "LLVM/MLIR download succeeded but MLIRConfig.cmake not found.\n"
      "Expected at: ${_LLVM_CMAKE_DIR}/mlir/MLIRConfig.cmake\n"
      "The tarball layout may have changed.")
  endif()
  message(STATUS "CoIR: LLVM/MLIR installed to ${_LLVM_ROOT}")
endif()

# --- Configure LLVM/MLIR ---
list(PREPEND CMAKE_PREFIX_PATH "${_LLVM_ROOT}")
set(MLIR_DIR "${_LLVM_CMAKE_DIR}/mlir" CACHE PATH "MLIR CMake directory")
set(LLVM_DIR "${_LLVM_CMAKE_DIR}/llvm" CACHE PATH "LLVM CMake directory")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "CoIR: Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "CoIR: Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

if(NOT LLVM_TABLEGEN_EXE)
  set(LLVM_TABLEGEN_EXE "${_LLVM_ROOT}/bin/llvm-tblgen")
endif()
if(NOT MLIR_TABLEGEN_EXE)
  set(MLIR_TABLEGEN_EXE "${_LLVM_ROOT}/bin/mlir-tblgen")
endif()
