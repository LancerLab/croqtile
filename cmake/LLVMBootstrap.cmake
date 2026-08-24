# LLVMBootstrap.cmake
# On-demand LLVM/MLIR dependency for the CoIR tooling.
#
# Reads LLVM_* entries from cmake/deps.conf for URL, tarball name, and MD5.
# When LLVM/MLIR is not found in extern/llvm-project/, obtains it
# automatically (or errors out if COIR_AUTO_DOWNLOAD_LLVM is OFF) using
# this fallback chain:
#   1. internal FTP mirror ($FTP_SERVER env var, see cmake/DepMirror.cmake)
#   2. public release tarball (LLVM_URL in cmake/deps.conf)
#   3. build from source (scripts/build_llvm.sh) -- slow, last resort
# An existing installation whose version does not match LLVM_SHASH in
# cmake/deps.conf is discarded and re-obtained through the same chain.
#
# Sets: MLIR_DIR, LLVM_DIR, LLVM_TABLEGEN_EXE, MLIR_TABLEGEN_EXE

set(_LLVM_ROOT "${CMAKE_SOURCE_DIR}/extern/llvm-project")

# --- Read LLVM settings from cmake/deps.conf ---
set(_DEP_CONF "${CMAKE_SOURCE_DIR}/cmake/deps.conf")
if(NOT EXISTS "${_DEP_CONF}")
  message(FATAL_ERROR
    "Missing ${_DEP_CONF} -- cannot determine LLVM source.\n"
    "This file should contain LLVM_URL, LLVM_TAR, LLVM_MD5, LLVM_SHASH.")
endif()
file(STRINGS "${_DEP_CONF}" _dep_lines REGEX "^LLVM_[A-Z_0-9]+=")
foreach(_line ${_dep_lines})
  string(REGEX MATCH "^([A-Z_0-9]+)=(.*)" _ "${_line}")
  set(COIR_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
endforeach()

option(COIR_AUTO_DOWNLOAD_LLVM
  "Automatically download LLVM/MLIR if not found" ON)

# --- Locate or download LLVM/MLIR ---
set(_LLVM_CMAKE_DIR "${_LLVM_ROOT}/lib/cmake")

# Verify that an existing LLVM/MLIR installation matches the version
# requested in cmake/deps.conf. A stale installation (e.g. LLVM 18 left
# behind after a version bump) will otherwise produce confusing API
# mismatch errors during the CoIR build.
set(_LLVM_VERSION_MISMATCH FALSE)
if(EXISTS "${_LLVM_CMAKE_DIR}/llvm/LLVMConfig.cmake")
  file(STRINGS "${_LLVM_CMAKE_DIR}/llvm/LLVMConfig.cmake" _llvm_config_version_line
    REGEX "^set\\(LLVM_PACKAGE_VERSION ")
  if(_llvm_config_version_line)
    string(REGEX REPLACE "^set\\(LLVM_PACKAGE_VERSION ([^)]+)\\).*" "\\1"
      _installed_llvm_version "${_llvm_config_version_line}")
    string(STRIP "${_installed_llvm_version}" _installed_llvm_version)

    # Expected version is only comparable when LLVM_SHASH is a release tag
    # (e.g. "llvmorg-21.1.0" or a bare "X.Y.Z"). When LLVM_SHASH is a git
    # commit hash (e.g. "d752c5b"), LLVM_PACKAGE_VERSION (e.g. "21.0.0git")
    # cannot be compared against it, so skip the mismatch check entirely.
    set(_expected_llvm_version "")
    if(DEFINED COIR_LLVM_SHASH)
      string(REGEX REPLACE "^llvmorg-" "" _expected_llvm_version "${COIR_LLVM_SHASH}")
      string(STRIP "${_expected_llvm_version}" _expected_llvm_version)
      if(NOT _expected_llvm_version MATCHES "^[0-9]+([.][0-9]+)+")
        # Not a semantic version (e.g. a commit hash); nothing to compare.
        set(_expected_llvm_version "")
      endif()
    endif()

    if(_expected_llvm_version AND
        NOT _installed_llvm_version VERSION_EQUAL _expected_llvm_version)
      message(STATUS
        "CoIR: installed LLVM ${_installed_llvm_version} does not match "
        "requested ${_expected_llvm_version}")
      set(_LLVM_VERSION_MISMATCH TRUE)
    endif()
  endif()
endif()

if(_LLVM_VERSION_MISMATCH)
  if(NOT COIR_AUTO_DOWNLOAD_LLVM)
    message(FATAL_ERROR
      "LLVM/MLIR at ${_LLVM_ROOT} is version ${_installed_llvm_version}, "
      "but cmake/deps.conf requests ${_expected_llvm_version}.\n"
      "Set -DCOIR_AUTO_DOWNLOAD_LLVM=ON to replace it automatically, "
      "or manually extract the correct LLVM tarball into ${_LLVM_ROOT}.")
  endif()
  message(STATUS
    "CoIR: removing stale LLVM/MLIR installation at ${_LLVM_ROOT}...")
  file(REMOVE_RECURSE "${_LLVM_ROOT}")
endif()

if(NOT EXISTS "${_LLVM_CMAKE_DIR}/mlir/MLIRConfig.cmake")
  if(NOT COIR_AUTO_DOWNLOAD_LLVM)
    message(FATAL_ERROR
      "LLVM/MLIR not found at ${_LLVM_ROOT}.\n"
      "CoIR tooling requires a pre-built LLVM/MLIR installation.\n"
      "Set -DCOIR_AUTO_DOWNLOAD_LLVM=ON to download automatically,\n"
      "or manually extract the LLVM tarball into extern/llvm-project/")
  endif()

  message(STATUS "CoIR: LLVM/MLIR not found, downloading (${COIR_LLVM_SHASH})...")
  set(_TAR_PATH "${CMAKE_SOURCE_DIR}/extern/${COIR_LLVM_TAR}")

  include("${CMAKE_SOURCE_DIR}/cmake/DepMirror.cmake")
  dep_validate_cached_archive(
    _cached_archive_valid "${_TAR_PATH}" "${COIR_LLVM_MD5}")
  if(NOT _cached_archive_valid)
    dep_download_order(_download_url _dependency_fallback_url
      "${COIR_LLVM_URL}" "${COIR_LLVM_TAR}" llvm)
    set(_dl_args "${_download_url}" "${_TAR_PATH}"
      SHOW_PROGRESS STATUS _dl_status TIMEOUT 600)
    file(DOWNLOAD ${_dl_args})
    list(GET _dl_status 0 _dl_code)
    if(_dl_code EQUAL 0 AND COIR_LLVM_MD5)
      file(MD5 "${_TAR_PATH}" _actual_md5)
      if(NOT _actual_md5 STREQUAL COIR_LLVM_MD5)
        file(REMOVE "${_TAR_PATH}")
        set(_dl_code 1)
        set(_dl_msg "MD5 mismatch: expected ${COIR_LLVM_MD5}, got ${_actual_md5}")
      endif()
    endif()
    if(NOT _dl_code EQUAL 0)
      # In FTP-first mode, retry the configured primary URL before its
      # separately configured fallback URL.
      if(NOT _download_url STREQUAL COIR_LLVM_URL
          AND _dependency_fallback_url)
        message(STATUS
          "CoIR: trying configured URL ${_dependency_fallback_url}")
        set(_dl_args "${_dependency_fallback_url}" "${_TAR_PATH}"
          SHOW_PROGRESS STATUS _dl_status TIMEOUT 600)
        file(DOWNLOAD ${_dl_args})
        list(GET _dl_status 0 _dl_code)
        if(_dl_code EQUAL 0 AND COIR_LLVM_MD5)
          file(MD5 "${_TAR_PATH}" _actual_md5)
          if(NOT _actual_md5 STREQUAL COIR_LLVM_MD5)
            file(REMOVE "${_TAR_PATH}")
            set(_dl_code 1)
            set(_dl_msg
              "MD5 mismatch: expected ${COIR_LLVM_MD5}, "
              "got ${_actual_md5}")
          endif()
        endif()
      endif()
    endif()
    if(NOT _dl_code EQUAL 0)
      # Primary download failed -- try fallback URL if configured
      list(GET _dl_status 1 _dl_msg)
      file(REMOVE "${_TAR_PATH}")
      if(DEFINED COIR_LLVM_FALLBACK_URL AND NOT "${COIR_LLVM_FALLBACK_URL}" STREQUAL "")
        message(WARNING
          "CoIR: LLVM primary download failed (${_dl_msg})\n"
          "  Primary URL: ${COIR_LLVM_URL}\n"
          "  Trying fallback URL: ${COIR_LLVM_FALLBACK_URL}")
        set(_dl_args "${COIR_LLVM_FALLBACK_URL}" "${_TAR_PATH}"
          SHOW_PROGRESS STATUS _dl_status TIMEOUT 600)
        file(DOWNLOAD ${_dl_args})
        list(GET _dl_status 0 _dl_code)
        if(_dl_code EQUAL 0 AND DEFINED COIR_LLVM_FALLBACK_MD5
            AND NOT "${COIR_LLVM_FALLBACK_MD5}" STREQUAL "")
          file(MD5 "${_TAR_PATH}" _actual_md5)
          if(NOT _actual_md5 STREQUAL COIR_LLVM_FALLBACK_MD5)
            file(REMOVE "${_TAR_PATH}")
            set(_dl_code 1)
            set(_dl_msg "MD5 mismatch: expected ${COIR_LLVM_FALLBACK_MD5}, got ${_actual_md5}")
          endif()
        endif()
      endif()
      # Still failed -- try FTP_SERVER mirror if set
      if(NOT _dl_code EQUAL 0 AND _download_url STREQUAL COIR_LLVM_URL)
        if(_dependency_fallback_url)
          message(STATUS
            "CoIR: trying dependency fallback ${_dependency_fallback_url}")
          set(_dl_args "${_dependency_fallback_url}" "${_TAR_PATH}"
            SHOW_PROGRESS STATUS _dl_status TIMEOUT 600)
          file(DOWNLOAD ${_dl_args})
          list(GET _dl_status 0 _dl_code)
          if(_dl_code EQUAL 0 AND COIR_LLVM_MD5)
            file(MD5 "${_TAR_PATH}" _actual_md5)
            if(NOT _actual_md5 STREQUAL COIR_LLVM_MD5)
              file(REMOVE "${_TAR_PATH}")
              set(_dl_code 1)
              set(_dl_msg
                "MD5 mismatch: expected ${COIR_LLVM_MD5}, "
                "got ${_actual_md5}")
            endif()
          endif()
        endif()
      endif()
    endif()
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      file(REMOVE "${_TAR_PATH}")
      # Deepest fallback: build LLVM/MLIR from source.  This is slow, so
      # it runs only after the FTP mirror and public tarballs failed.
      message(WARNING
        "CoIR: LLVM/MLIR download failed (${_dl_msg})\n"
        "  URL: ${COIR_LLVM_URL}\n"
        "  Falling back to building ${COIR_LLVM_SHASH} from source; "
        "this can take a long time.")
      execute_process(
        COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/build_llvm.sh"
                "${COIR_LLVM_SHASH}"
        RESULT_VARIABLE _llvm_build_result)
      if(NOT _llvm_build_result EQUAL 0)
        message(FATAL_ERROR
          "Failed to obtain LLVM/MLIR by download or source build.\n"
          "You can download manually and place at: ${_TAR_PATH}\n"
          "or run: bash scripts/build_llvm.sh ${COIR_LLVM_SHASH}")
      endif()
      set(_llvm_built_from_source TRUE)
    endif()
  endif()

  if(NOT _llvm_built_from_source)
    message(STATUS "CoIR: Extracting LLVM/MLIR into ${_LLVM_ROOT}...")
    file(MAKE_DIRECTORY "${_LLVM_ROOT}")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf "${_TAR_PATH}"
      WORKING_DIRECTORY "${_LLVM_ROOT}"
      RESULT_VARIABLE _extract_result)
    if(NOT _extract_result EQUAL 0)
      file(REMOVE "${_TAR_PATH}")
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
  endif()

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

set(Clang_DIR "${_LLVM_CMAKE_DIR}/clang" CACHE PATH "Clang CMake directory")
find_package(Clang CONFIG)
if(Clang_FOUND)
  message(STATUS "CoIR: Found Clang at ${Clang_DIR}")
endif()

message(STATUS "CoIR: Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "CoIR: Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)
include(TableGen)

# Ninja < 1.10 cannot handle depfile mode when CMake adds implicit
# outputs (| ${cmake_ninja_workdir}...) for IDE support.  The depslog
# rejects them with "multiple outputs aren't (yet?) supported".
# Override tablegen() to fall back to globbing .td files instead of
# using DEPFILE when the installed Ninja is too old.
if(CMAKE_GENERATOR MATCHES "Ninja")
  execute_process(
    COMMAND ninja --version
    OUTPUT_VARIABLE _ninja_ver
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_ninja_ver VERSION_LESS "1.10")
    message(STATUS "CoIR: Ninja ${_ninja_ver} < 1.10 -- "
      "disabling tablegen depfile mode to avoid depslog error")
    function(tablegen project ofn)
      cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})

      if(NOT ${project}_TABLEGEN_EXE)
        message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
      endif()

      file(GLOB local_tds "*.td")
      file(GLOB_RECURSE global_tds
        "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
      set(additional_cmdline -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn})

      if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
        set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
      else()
        set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
          ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
      endif()

      get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
      list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
      list(REMOVE_ITEM tblgen_includes "")
      list(TRANSFORM tblgen_includes PREPEND -I)

      set(tablegen_exe ${${project}_TABLEGEN_EXE})
      set(tablegen_depends ${${project}_TABLEGEN_TARGET} ${tablegen_exe})

      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
        COMMAND ${tablegen_exe} ${ARG_UNPARSED_ARGUMENTS}
          -I ${CMAKE_CURRENT_SOURCE_DIR}
          ${tblgen_includes}
          ${LLVM_TABLEGEN_FLAGS}
          ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
          --write-if-changed
          ${additional_cmdline}
        DEPENDS ${ARG_DEPENDS} ${tablegen_depends}
          ${local_tds} ${global_tds}
          ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
          ${LLVM_TARGET_DEPENDS}
        COMMENT "Building ${ofn}...")

      set_property(DIRECTORY APPEND
        PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})
      set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT}
        ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
      set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES GENERATED 1)
    endfunction()
  endif()
endif()

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
