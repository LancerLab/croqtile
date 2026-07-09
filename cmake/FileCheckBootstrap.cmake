# FileCheckBootstrap.cmake
# Ensure the FileCheck utility (from LLVM) is available for lit.sh tests.
#
# Search order:
#   1. extern/bin/FileCheck               (already present from a prior kit)
#   2. extern/llvm-project/bin/FileCheck  (from CoIR LLVM download)
#   3. Download from FILECHECK_URL in cmake/deps.conf (prebuilt kit)
#   4. Download from LLVM_URL in cmake/deps.conf, extract bin/FileCheck
#   5. System PATH (FileCheck, FileCheck-18, ..., FileCheck-14)
#
# When CoIR is enabled (default), LLVM is downloaded by LLVMBootstrap.cmake
# and FileCheck is found at extern/llvm-project/bin/FileCheck (strategy 2).
# For choreo-only builds, strategy 3 or 4 provides FileCheck.
# System PATH is the last resort -- it may find a Python port that is not
# fully compatible, so download-based strategies are preferred.
#
# This file must be included AFTER add_subdirectory(tools) so that
# LLVMBootstrap has already run when CoIR is enabled.
#
# Usage: include(cmake/FileCheckBootstrap.cmake)
# Sets: FILECHECK_EXECUTABLE (cached)

set(_FC_EXTERN_BIN "${CMAKE_SOURCE_DIR}/extern/bin/FileCheck")
set(_FC_LLVM_BIN "${CMAKE_SOURCE_DIR}/extern/llvm-project/bin/FileCheck")

macro(_fc_found _path _source)
  set(FILECHECK_EXECUTABLE "${_path}"
    CACHE FILEPATH "FileCheck executable" FORCE)
  message(STATUS "FileCheck: ${FILECHECK_EXECUTABLE}${_source}")
  return()
endmacro()

# 1. extern/bin/FileCheck
if(EXISTS "${_FC_EXTERN_BIN}")
  _fc_found("${_FC_EXTERN_BIN}" "")
endif()

# 2. extern/llvm-project/bin/FileCheck (available after LLVMBootstrap)
if(EXISTS "${_FC_LLVM_BIN}")
  _fc_found("${_FC_LLVM_BIN}" "")
endif()

# --- Read deps.conf ---
set(_FC_URL "")
set(_FC_MD5 "")
set(_FC_LLVM_URL "")

set(_FC_CONF "${CMAKE_SOURCE_DIR}/cmake/deps.conf")
if(EXISTS "${_FC_CONF}")
  file(STRINGS "${_FC_CONF}" _fc_lines
    REGEX "^(FILECHECK|LLVM)_[A-Z_0-9]+=")
  foreach(_line ${_fc_lines})
    if(_line MATCHES "^FILECHECK_([A-Z_0-9]+)=(.*)")
      set(_FC_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
    elseif(_line MATCHES "^LLVM_URL=(.*)")
      set(_FC_LLVM_URL "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()

# 3. Download from FILECHECK_URL (prebuilt kit containing bin/FileCheck)
if(_FC_URL)
  set(_archive "${CMAKE_BINARY_DIR}/_filecheck_kit.tar.gz")
  set(_dl_args "${_FC_URL}" "${_archive}" TIMEOUT 120 STATUS _dl_status)
  if(_FC_MD5)
    list(APPEND _dl_args EXPECTED_MD5 "${_FC_MD5}")
  endif()

  message(STATUS "Downloading FileCheck kit...")
  file(DOWNLOAD ${_dl_args})

  list(GET _dl_status 0 _dl_code)
  if(_dl_code EQUAL 0)
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/extern/bin")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf "${_archive}"
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/extern"
      RESULT_VARIABLE _ext_rc ERROR_QUIET)
    file(REMOVE "${_archive}")
    if(_ext_rc EQUAL 0 AND EXISTS "${_FC_EXTERN_BIN}")
      execute_process(COMMAND chmod +x "${_FC_EXTERN_BIN}" ERROR_QUIET)
      _fc_found("${_FC_EXTERN_BIN}" "")
    endif()
  else()
    list(GET _dl_status 1 _dl_msg)
    # Try FTP_SERVER mirror if set
    include(cmake/DepMirror.cmake)
    get_filename_component(_fc_tarball "${_FC_URL}" NAME)
    dep_mirror_fallback(_mirror_url "${_fc_tarball}" filecheck)
    if(_mirror_url)
      message(STATUS "FileCheck: trying mirror ${_mirror_url}")
      set(_dl_args "${_mirror_url}" "${_archive}" TIMEOUT 120 STATUS _dl_status)
      file(DOWNLOAD ${_dl_args})
      list(GET _dl_status 0 _dl_code)
      if(_dl_code EQUAL 0)
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/extern/bin")
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E tar xf "${_archive}"
          WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/extern"
          RESULT_VARIABLE _ext_rc ERROR_QUIET)
        file(REMOVE "${_archive}")
        if(_ext_rc EQUAL 0 AND EXISTS "${_FC_EXTERN_BIN}")
          execute_process(COMMAND chmod +x "${_FC_EXTERN_BIN}" ERROR_QUIET)
          _fc_found("${_FC_EXTERN_BIN}" "")
        endif()
      else()
        message(STATUS "FileCheck mirror download failed")
        file(REMOVE "${_archive}")
      endif()
    else()
      message(STATUS "FileCheck kit download failed: ${_dl_msg}")
      file(REMOVE "${_archive}")
    endif()
  endif()
endif()

# 4. Fallback: download LLVM release and extract just bin/FileCheck.
#    Used for choreo-only builds where no prebuilt kit is configured.
if(_FC_LLVM_URL)
  message(STATUS
    "Downloading LLVM release to extract FileCheck"
    " (one-time, may take a few minutes)...")

  get_filename_component(_llvm_ext "${_FC_LLVM_URL}" EXT)
  set(_llvm_archive "${CMAKE_BINARY_DIR}/_filecheck_llvm${_llvm_ext}")

  file(DOWNLOAD "${_FC_LLVM_URL}" "${_llvm_archive}"
    TIMEOUT 900 STATUS _dl_status2 SHOW_PROGRESS)

  list(GET _dl_status2 0 _dl2_code)
  if(_dl2_code EQUAL 0)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar tf "${_llvm_archive}"
      OUTPUT_VARIABLE _tar_listing ERROR_QUIET RESULT_VARIABLE _list_rc)
    if(_list_rc EQUAL 0)
      string(REGEX MATCH "([^\n]*/bin/FileCheck)\n" _ "${_tar_listing}")
      set(_fc_in_tar "${CMAKE_MATCH_1}")
      if(_fc_in_tar)
        execute_process(
          COMMAND ${CMAKE_COMMAND} -E tar xf "${_llvm_archive}"
            "${_fc_in_tar}"
          WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/extern"
          RESULT_VARIABLE _ext2_rc ERROR_QUIET)
        if(_ext2_rc EQUAL 0)
          set(_extracted "${CMAKE_SOURCE_DIR}/extern/${_fc_in_tar}")
          if(EXISTS "${_extracted}")
            file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/extern/bin")
            file(RENAME "${_extracted}" "${_FC_EXTERN_BIN}")
            execute_process(
              COMMAND chmod +x "${_FC_EXTERN_BIN}" ERROR_QUIET)
            string(REGEX MATCH "^([^/]+)/" _ "${_fc_in_tar}")
            if(CMAKE_MATCH_1)
              set(_top "${CMAKE_SOURCE_DIR}/extern/${CMAKE_MATCH_1}")
              if(IS_DIRECTORY "${_top}")
                file(REMOVE_RECURSE "${_top}")
              endif()
            endif()
          endif()
        endif()
      endif()
    endif()
    file(REMOVE "${_llvm_archive}")
    if(EXISTS "${_FC_EXTERN_BIN}")
      _fc_found("${_FC_EXTERN_BIN}" " (from LLVM release)")
    endif()
  else()
    list(GET _dl_status2 1 _dl2_msg)
    message(STATUS "LLVM download for FileCheck failed: ${_dl2_msg}")
    file(REMOVE "${_llvm_archive}")
  endif()
endif()

# 5. System PATH (last resort; may find a Python port that is not
#    fully compatible with LLVM's FileCheck)
find_program(_fc_sys NAMES
  FileCheck FileCheck-18 FileCheck-17 FileCheck-16
  FileCheck-15 FileCheck-14)
if(_fc_sys)
  _fc_found("${_fc_sys}" " (system)")
endif()

message(STATUS
  "FileCheck not found. Tests requiring FileCheck may fail."
  " Install llvm-dev (apt install llvm-dev) to provide it.")
