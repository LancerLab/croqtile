# FlexBisonBootstrap.cmake
# Auto-download prebuilt flex and bison when not found in the environment.
#
# Usage: include(cmake/FlexBisonBootstrap.cmake)
#
# Sets: FLEX_EXECUTABLE, BISON_EXECUTABLE, BISON_ENV_CMD, IS_LOCAL_BISON
# Downloads to: ${CMAKE_SOURCE_DIR}/extern/bin/ and extern/shared/bison/

set(_FB_TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/extern")
set(_FB_BIN_DIR "${_FB_TOOLCHAIN_DIR}/bin")
set(_FB_BISON_DATA_DIR "${_FB_TOOLCHAIN_DIR}/shared/bison")

# Required minimum versions
set(_FB_BISON_MIN_VERSION "3.8")
set(_FB_FLEX_MIN_VERSION "2.6")

# Prebuilt archive URL and SHA (linux-x86_64 static builds)
set(_FB_BISON_URL "https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz")
set(_FB_FLEX_URL "https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz")

# Download timeout (seconds)
set(FLEX_BISON_DOWNLOAD_TIMEOUT 120 CACHE STRING
  "Timeout in seconds for downloading flex/bison source archives")

# ---------- helper: check bison version ----------
function(_fb_check_bison_version BISON_EXE OUT_OK)
  set(_ok FALSE)
  if(BISON_EXE AND EXISTS "${BISON_EXE}")
    execute_process(
      COMMAND "${BISON_EXE}" --version
      OUTPUT_VARIABLE _ver_out
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0 AND _ver_out MATCHES "([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
      if(CMAKE_MATCH_1 VERSION_GREATER_EQUAL "${_FB_BISON_MIN_VERSION}")
        set(_ok TRUE)
      endif()
    endif()
  endif()
  set(${OUT_OK} ${_ok} PARENT_SCOPE)
endfunction()

# ---------- helper: check flex version ----------
function(_fb_check_flex_version FLEX_EXE OUT_OK)
  set(_ok FALSE)
  if(FLEX_EXE AND EXISTS "${FLEX_EXE}")
    execute_process(
      COMMAND "${FLEX_EXE}" --version
      OUTPUT_VARIABLE _ver_out
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _rc)
    if(_rc EQUAL 0 AND _ver_out MATCHES "([0-9]+\\.[0-9]+(\\.[0-9]+)?)")
      if(CMAKE_MATCH_1 VERSION_GREATER_EQUAL "${_FB_FLEX_MIN_VERSION}")
        set(_ok TRUE)
      endif()
    endif()
  endif()
  set(${OUT_OK} ${_ok} PARENT_SCOPE)
endfunction()

# ---------- helper: build from source ----------
function(_fb_build_from_source TOOL_NAME TAR_URL DEST_DIR)
  set(_stamp "${DEST_DIR}/.${TOOL_NAME}_built")
  if(EXISTS "${_stamp}")
    return()
  endif()

  include(FetchContent)
  message(STATUS "Downloading ${TOOL_NAME} source (timeout: ${FLEX_BISON_DOWNLOAD_TIMEOUT}s)...")

  set(_src_dir "${CMAKE_BINARY_DIR}/_fb_${TOOL_NAME}_src")
  set(_archive "${CMAKE_BINARY_DIR}/_fb_${TOOL_NAME}.tar.gz")

  file(DOWNLOAD "${TAR_URL}" "${_archive}"
    TIMEOUT ${FLEX_BISON_DOWNLOAD_TIMEOUT}
    STATUS _dl_status
    SHOW_PROGRESS)

  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    list(GET _dl_status 1 _dl_msg)
    message(FATAL_ERROR
      "Failed to download ${TOOL_NAME}: ${_dl_msg}\n"
      "URL: ${TAR_URL}\n"
      "Install ${TOOL_NAME} manually or check network connectivity.")
  endif()

  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_src_dir}")

  file(GLOB _extracted_dirs "${_src_dir}/${TOOL_NAME}-*")
  list(GET _extracted_dirs 0 _build_dir)

  message(STATUS "Building ${TOOL_NAME} from source in ${_build_dir}...")

  execute_process(
    COMMAND ./configure --prefix=${DEST_DIR} --disable-shared
    WORKING_DIRECTORY "${_build_dir}"
    RESULT_VARIABLE _cfg_rc
    OUTPUT_QUIET ERROR_QUIET)

  if(NOT _cfg_rc EQUAL 0)
    message(FATAL_ERROR
      "${TOOL_NAME} configure failed (exit ${_cfg_rc}).\n"
      "Install ${TOOL_NAME} >= ${_FB_${TOOL_NAME}_MIN_VERSION} manually.")
  endif()

  include(ProcessorCount)
  ProcessorCount(_nproc)
  if(_nproc EQUAL 0)
    set(_nproc 2)
  endif()

  execute_process(
    COMMAND make -j${_nproc}
    WORKING_DIRECTORY "${_build_dir}"
    RESULT_VARIABLE _make_rc
    OUTPUT_QUIET ERROR_QUIET)

  if(NOT _make_rc EQUAL 0)
    message(FATAL_ERROR
      "${TOOL_NAME} build failed (exit ${_make_rc}).\n"
      "Install ${TOOL_NAME} >= ${_FB_${TOOL_NAME}_MIN_VERSION} manually.")
  endif()

  execute_process(
    COMMAND make install
    WORKING_DIRECTORY "${_build_dir}"
    RESULT_VARIABLE _inst_rc
    OUTPUT_QUIET ERROR_QUIET)

  if(NOT _inst_rc EQUAL 0)
    message(FATAL_ERROR "${TOOL_NAME} install failed (exit ${_inst_rc}).")
  endif()

  file(WRITE "${_stamp}" "built")
  file(REMOVE "${_archive}")
  file(REMOVE_RECURSE "${_src_dir}")
endfunction()


# =====================
# Main logic
# =====================

# 1. Try to find bison in toolchain, then system
find_program(BISON_EXECUTABLE NAMES bison HINTS "${_FB_BIN_DIR}")
_fb_check_bison_version("${BISON_EXECUTABLE}" _bison_ok)

if(NOT _bison_ok)
  # Try system bison
  find_program(_sys_bison NAMES bison)
  _fb_check_bison_version("${_sys_bison}" _sys_bison_ok)
  if(_sys_bison_ok)
    set(BISON_EXECUTABLE "${_sys_bison}" CACHE FILEPATH "Bison executable" FORCE)
    set(_bison_ok TRUE)
  endif()
endif()

if(NOT _bison_ok)
  message(STATUS "No suitable bison (>= ${_FB_BISON_MIN_VERSION}) found. Building from source...")
  _fb_build_from_source("bison" "${_FB_BISON_URL}" "${_FB_TOOLCHAIN_DIR}")
  set(BISON_EXECUTABLE "${_FB_BIN_DIR}/bison" CACHE FILEPATH "Bison executable" FORCE)
  _fb_check_bison_version("${BISON_EXECUTABLE}" _bison_ok)
  if(NOT _bison_ok)
    message(FATAL_ERROR "Failed to build a working bison >= ${_FB_BISON_MIN_VERSION}")
  endif()
endif()

# 2. Try to find flex in toolchain, then system
find_program(FLEX_EXECUTABLE NAMES flex HINTS "${_FB_BIN_DIR}")
_fb_check_flex_version("${FLEX_EXECUTABLE}" _flex_ok)

if(NOT _flex_ok)
  find_program(_sys_flex NAMES flex)
  _fb_check_flex_version("${_sys_flex}" _sys_flex_ok)
  if(_sys_flex_ok)
    set(FLEX_EXECUTABLE "${_sys_flex}" CACHE FILEPATH "Flex executable" FORCE)
    set(_flex_ok TRUE)
  endif()
endif()

if(NOT _flex_ok)
  message(STATUS "No suitable flex (>= ${_FB_FLEX_MIN_VERSION}) found. Building from source...")
  _fb_build_from_source("flex" "${_FB_FLEX_URL}" "${_FB_TOOLCHAIN_DIR}")
  set(FLEX_EXECUTABLE "${_FB_BIN_DIR}/flex" CACHE FILEPATH "Flex executable" FORCE)
  _fb_check_flex_version("${FLEX_EXECUTABLE}" _flex_ok)
  if(NOT _flex_ok)
    message(FATAL_ERROR "Failed to build a working flex >= ${_FB_FLEX_MIN_VERSION}")
  endif()
endif()

# 3. Set BISON_ENV_CMD for local bison data dir
set(IS_LOCAL_BISON FALSE)
set(BISON_ENV_CMD "")

file(TO_CMAKE_PATH "${_FB_TOOLCHAIN_DIR}" _toolchain_norm)
file(TO_CMAKE_PATH "${BISON_EXECUTABLE}" _bison_norm)
string(LENGTH "${_toolchain_norm}" _tc_len)
string(SUBSTRING "${_bison_norm}" 0 ${_tc_len} _bison_prefix)
if(_bison_prefix STREQUAL _toolchain_norm)
  set(IS_LOCAL_BISON TRUE)
  # Check both the legacy path (shared/bison) and standard GNU path (share/bison)
  set(_FB_BISON_DATA_DIR_GNU "${_FB_TOOLCHAIN_DIR}/share/bison")
  if(EXISTS "${_FB_BISON_DATA_DIR}")
    set(BISON_ENV_CMD "BISON_PKGDATADIR=${_FB_BISON_DATA_DIR}")
    message(STATUS "Using local Bison data dir: ${_FB_BISON_DATA_DIR}")
  elseif(EXISTS "${_FB_BISON_DATA_DIR_GNU}")
    set(BISON_ENV_CMD "BISON_PKGDATADIR=${_FB_BISON_DATA_DIR_GNU}")
    message(STATUS "Using local Bison data dir: ${_FB_BISON_DATA_DIR_GNU}")
  endif()
endif()

# 4. Report
execute_process(COMMAND "${BISON_EXECUTABLE}" --version OUTPUT_VARIABLE _bv OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND "${FLEX_EXECUTABLE}" --version OUTPUT_VARIABLE _fv OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX MATCH "[^\n]+" _bv_line "${_bv}")
string(REGEX MATCH "[^\n]+" _fv_line "${_fv}")
message(STATUS "Flex:  ${FLEX_EXECUTABLE} (${_fv_line})")
message(STATUS "Bison: ${BISON_EXECUTABLE} (${_bv_line})")
