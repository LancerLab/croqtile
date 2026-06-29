# CutlassBootstrap.cmake
# Auto-download CUTLASS when ENABLE_CUTE is ON and the library is missing.
#
# Reads CUTLASS_* entries from cmake/deps.conf for URL, version, and MD5.
#
# Sets: _cute_root, _cute_root_valid

set(_CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/extern/cutlass")

# --- Read CUTLASS settings from cmake/deps.conf ---
set(_CUTLASS_CONF "${CMAKE_SOURCE_DIR}/cmake/deps.conf")
set(CUTLASS_VERSION "")
set(CUTLASS_URL "")
set(CUTLASS_TAR "")
set(CUTLASS_MD5 "")

if(EXISTS "${_CUTLASS_CONF}")
  file(STRINGS "${_CUTLASS_CONF}" _cutlass_lines REGEX "^CUTLASS_[A-Z_0-9]+=")
  foreach(_line ${_cutlass_lines})
    string(REGEX MATCH "^([A-Z_0-9]+)=(.*)" _ "${_line}")
    set(${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endforeach()
endif()

if(NOT CUTLASS_URL)
  message(WARNING "No CUTLASS_URL in deps.conf; cannot auto-download CUTLASS.")
  return()
endif()

option(CUTLASS_AUTO_DOWNLOAD
  "Automatically download CUTLASS if not found" ON)

_looks_like_valid_cute("${_CUTLASS_ROOT}" _cutlass_present)

if(NOT _cutlass_present)
  if(NOT CUTLASS_AUTO_DOWNLOAD)
    message(STATUS "CUTLASS not found at ${_CUTLASS_ROOT}. "
      "Set -DCUTLASS_AUTO_DOWNLOAD=ON or provide CUTE_HOME.")
    return()
  endif()

  message(STATUS "CUTLASS not found, downloading ${CUTLASS_VERSION}...")
  set(_CUTLASS_TAR_PATH "${CMAKE_SOURCE_DIR}/extern/${CUTLASS_TAR}")

  if(NOT EXISTS "${_CUTLASS_TAR_PATH}")
    set(_dl_args "${CUTLASS_URL}" "${_CUTLASS_TAR_PATH}"
      SHOW_PROGRESS STATUS _dl_status TIMEOUT 300)
    if(CUTLASS_MD5)
      list(APPEND _dl_args EXPECTED_MD5 "${CUTLASS_MD5}")
    endif()
    file(DOWNLOAD ${_dl_args})
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      file(REMOVE "${_CUTLASS_TAR_PATH}")
      message(WARNING
        "Failed to download CUTLASS: ${_dl_msg}\n"
        "URL: ${CUTLASS_URL}\n"
        "CUTE support will be disabled.")
      return()
    endif()
  endif()

  message(STATUS "Extracting CUTLASS into ${_CUTLASS_ROOT}...")
  file(MAKE_DIRECTORY "${_CUTLASS_ROOT}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf "${_CUTLASS_TAR_PATH}"
    WORKING_DIRECTORY "${_CUTLASS_ROOT}"
    RESULT_VARIABLE _extract_result)
  if(NOT _extract_result EQUAL 0)
    message(WARNING "Failed to extract CUTLASS tarball.")
    file(REMOVE "${_CUTLASS_TAR_PATH}")
    return()
  endif()

  # GitHub tarballs extract into cutlass-X.Y.Z/; flatten into extern/cutlass/
  _looks_like_valid_cute("${_CUTLASS_ROOT}" _ok_after_extract)
  if(NOT _ok_after_extract)
    file(GLOB _cutlass_subdirs "${_CUTLASS_ROOT}/*/include/cute")
    if(_cutlass_subdirs)
      list(GET _cutlass_subdirs 0 _inner_cute)
      get_filename_component(_inner_root "${_inner_cute}" DIRECTORY)
      get_filename_component(_inner_root "${_inner_root}" DIRECTORY)
      file(GLOB _inner_contents "${_inner_root}/*")
      foreach(_item ${_inner_contents})
        get_filename_component(_name "${_item}" NAME)
        file(RENAME "${_item}" "${_CUTLASS_ROOT}/${_name}")
      endforeach()
    endif()
  endif()

  file(REMOVE "${_CUTLASS_TAR_PATH}")

  _looks_like_valid_cute("${_CUTLASS_ROOT}" _cutlass_final)
  if(NOT _cutlass_final)
    message(WARNING
      "CUTLASS download succeeded but headers not found.\n"
      "Expected include/cute/ in ${_CUTLASS_ROOT}")
    return()
  endif()
  message(STATUS "CUTLASS ${CUTLASS_VERSION} installed to ${_CUTLASS_ROOT}")
endif()

set(_cute_root "${_CUTLASS_ROOT}")
set(_cute_root_valid TRUE)
