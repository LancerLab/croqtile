# GTestBootstrap.cmake
# Auto-download GoogleTest source when not found in extern/gtest/.
#
# Reads GTEST_* entries from cmake/deps.conf for URL, tarball, and MD5.
# The standalone tests (tests/standalone/) build gtest from source
# via their own Makefile; this module only ensures the source tree exists.
#
# Usage: include(cmake/GTestBootstrap.cmake)

set(_GTEST_ROOT "${CMAKE_SOURCE_DIR}/extern/gtest")
set(_GTEST_MARKER "${_GTEST_ROOT}/googletest/src/gtest-all.cc")

# --- Read settings from cmake/deps.conf ---
set(_GTEST_URL "")
set(_GTEST_TAR "")
set(_GTEST_MD5 "")

set(_GTEST_CONF "${CMAKE_SOURCE_DIR}/cmake/deps.conf")
if(EXISTS "${_GTEST_CONF}")
  file(STRINGS "${_GTEST_CONF}" _gtest_lines REGEX "^GTEST_[A-Z_0-9]+=")
  foreach(_line ${_gtest_lines})
    string(REGEX MATCH "^GTEST_([A-Z_0-9]+)=(.*)" _ "${_line}")
    set(_GTEST_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endforeach()
endif()

if(EXISTS "${_GTEST_MARKER}")
  return()
endif()

if(NOT _GTEST_URL)
  message(STATUS
    "GoogleTest not found at ${_GTEST_ROOT} and no GTEST_URL in deps.conf."
    " Standalone tests will not be available.")
  return()
endif()

set(_archive "${CMAKE_BINARY_DIR}/_gtest_download.tar.gz")

set(_dl_args "${_GTEST_URL}" "${_archive}"
  TIMEOUT 120
  STATUS _dl_status)
if(_GTEST_MD5)
  list(APPEND _dl_args EXPECTED_MD5 "${_GTEST_MD5}")
endif()

message(STATUS "Downloading GoogleTest...")
file(DOWNLOAD ${_dl_args})

list(GET _dl_status 0 _dl_code)
if(NOT _dl_code EQUAL 0)
  list(GET _dl_status 1 _dl_msg)
  # Try FTP_SERVER mirror if set
  include(cmake/DepMirror.cmake)
  dep_mirror_fallback(_mirror_url "${_GTEST_TAR}" gtest)
  if(_mirror_url)
    message(STATUS "GTest: trying mirror ${_mirror_url}")
    set(_dl_args "${_mirror_url}" "${_archive}"
      TIMEOUT 120 STATUS _dl_status)
    file(DOWNLOAD ${_dl_args})
    list(GET _dl_status 0 _dl_code)
  endif()
endif()
if(NOT _dl_code EQUAL 0)
  list(GET _dl_status 1 _dl_msg)
  message(STATUS "GoogleTest download failed: ${_dl_msg}")
  file(REMOVE "${_archive}")
  return()
endif()

file(MAKE_DIRECTORY "${_GTEST_ROOT}")
message(STATUS "Extracting GoogleTest to ${_GTEST_ROOT}...")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf "${_archive}"
  WORKING_DIRECTORY "${_GTEST_ROOT}"
  RESULT_VARIABLE _ext_rc)

file(REMOVE "${_archive}")

if(NOT _ext_rc EQUAL 0)
  message(STATUS "GoogleTest extraction failed (exit ${_ext_rc})")
  return()
endif()

# GitHub archives extract to googletest-VERSION/; flatten if needed.
if(NOT EXISTS "${_GTEST_MARKER}")
  file(GLOB _nested_dirs "${_GTEST_ROOT}/googletest-*")
  if(_nested_dirs)
    list(GET _nested_dirs 0 _nested)
    file(GLOB _nested_contents "${_nested}/*")
    foreach(_item ${_nested_contents})
      get_filename_component(_name "${_item}" NAME)
      file(RENAME "${_item}" "${_GTEST_ROOT}/${_name}")
    endforeach()
    file(REMOVE_RECURSE "${_nested}")
  endif()
endif()

if(EXISTS "${_GTEST_MARKER}")
  message(STATUS "GoogleTest: ${_GTEST_ROOT}")
else()
  message(STATUS "GoogleTest extraction succeeded but source not found"
    " at expected path.")
endif()
