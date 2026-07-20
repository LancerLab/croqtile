# PythonBootstrap.cmake
# On-demand Python + pybind11 dependency for the co-py tooling.
#
# Reads PYBIND11_* entries from cmake/deps.conf for URL, tarball name, and MD5.
# When pybind11 is not found in extern/pybind11/, downloads it automatically
# (or errors out if CROPY_AUTO_DOWNLOAD_PYBIND11 is OFF).
#
# Sets: pybind11 targets available for pybind11_add_module()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(_PYBIND11_ROOT "${CMAKE_SOURCE_DIR}/extern/pybind11")

# --- Read PYBIND11 settings from cmake/deps.conf ---
set(_DEP_CONF "${CMAKE_SOURCE_DIR}/cmake/deps.conf")
if(NOT EXISTS "${_DEP_CONF}")
  message(FATAL_ERROR
    "Missing ${_DEP_CONF} -- cannot determine pybind11 source.\n"
    "This file should contain PYBIND11_URL, PYBIND11_TAR, PYBIND11_MD5.")
endif()
file(STRINGS "${_DEP_CONF}" _dep_lines REGEX "^PYBIND11_[A-Z_0-9]+=")
foreach(_line ${_dep_lines})
  string(REGEX MATCH "^([A-Z_0-9]+)=(.*)" _ "${_line}")
  set(CROPY_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
endforeach()

option(CROPY_AUTO_DOWNLOAD_PYBIND11
  "Automatically download pybind11 if not found" ON)

# --- Locate or download pybind11 ---
if(NOT EXISTS "${_PYBIND11_ROOT}/CMakeLists.txt")
  if(NOT CROPY_AUTO_DOWNLOAD_PYBIND11)
    message(FATAL_ERROR
      "pybind11 not found at ${_PYBIND11_ROOT}.\n"
      "co-py requires pybind11.\n"
      "Set -DCROPY_AUTO_DOWNLOAD_PYBIND11=ON to download automatically,\n"
      "or manually extract the pybind11 tarball into extern/pybind11/")
  endif()

  message(STATUS "co-py: pybind11 not found, downloading (${CROPY_PYBIND11_VERSION})...")
  set(_TAR_PATH "${CMAKE_SOURCE_DIR}/extern/${CROPY_PYBIND11_TAR}")

  include("${CMAKE_SOURCE_DIR}/cmake/DepMirror.cmake")
  dep_validate_cached_archive(
    _cached_archive_valid "${_TAR_PATH}" "${CROPY_PYBIND11_MD5}")
  if(NOT _cached_archive_valid)
    dep_download_order(_download_url _fallback_url
      "${CROPY_PYBIND11_URL}" "${CROPY_PYBIND11_TAR}" pybind11)
    set(_dl_args "${_download_url}" "${_TAR_PATH}"
      SHOW_PROGRESS STATUS _dl_status TIMEOUT 120)
    file(DOWNLOAD ${_dl_args})
    list(GET _dl_status 0 _dl_code)
    if(_dl_code EQUAL 0 AND CROPY_PYBIND11_MD5)
      file(MD5 "${_TAR_PATH}" _actual_md5)
      if(NOT _actual_md5 STREQUAL CROPY_PYBIND11_MD5)
        file(REMOVE "${_TAR_PATH}")
        set(_dl_code 1)
        set(_dl_msg "MD5 mismatch: expected ${CROPY_PYBIND11_MD5}, got ${_actual_md5}")
      endif()
    endif()
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      if(_fallback_url)
        message(STATUS "pybind11: trying fallback ${_fallback_url}")
        set(_dl_args "${_fallback_url}" "${_TAR_PATH}"
          SHOW_PROGRESS STATUS _dl_status TIMEOUT 120)
        file(DOWNLOAD ${_dl_args})
        list(GET _dl_status 0 _dl_code)
        if(_dl_code EQUAL 0 AND CROPY_PYBIND11_MD5)
          file(MD5 "${_TAR_PATH}" _actual_md5)
          if(NOT _actual_md5 STREQUAL CROPY_PYBIND11_MD5)
            file(REMOVE "${_TAR_PATH}")
            set(_dl_code 1)
            set(_dl_msg
              "MD5 mismatch: expected ${CROPY_PYBIND11_MD5}, "
              "got ${_actual_md5}")
          endif()
        endif()
      endif()
    endif()
    if(NOT _dl_code EQUAL 0)
      list(GET _dl_status 1 _dl_msg)
      file(REMOVE "${_TAR_PATH}")
      message(FATAL_ERROR
        "Failed to download pybind11: ${_dl_msg}\n"
        "URL: ${CROPY_PYBIND11_URL}\n"
        "You can download manually and place at: ${_TAR_PATH}")
    endif()
  endif()

  message(STATUS "co-py: extracting pybind11 into ${_PYBIND11_ROOT}...")
  file(MAKE_DIRECTORY "${_PYBIND11_ROOT}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xf "${_TAR_PATH}"
    WORKING_DIRECTORY "${_PYBIND11_ROOT}"
    RESULT_VARIABLE _extract_result)
  if(NOT _extract_result EQUAL 0)
    file(REMOVE "${_TAR_PATH}")
    message(FATAL_ERROR "Failed to extract pybind11 tarball: ${_TAR_PATH}")
  endif()

  # Handle tarballs that extract with a top-level directory (e.g. pybind11-2.13.1/)
  if(NOT EXISTS "${_PYBIND11_ROOT}/CMakeLists.txt")
    file(GLOB _subdirs "${_PYBIND11_ROOT}/*/CMakeLists.txt")
    if(_subdirs)
      list(GET _subdirs 0 _found_cmake)
      get_filename_component(_inner_root "${_found_cmake}" DIRECTORY)
      file(GLOB _inner_contents "${_inner_root}/*")
      foreach(_item ${_inner_contents})
        get_filename_component(_name "${_item}" NAME)
        file(RENAME "${_item}" "${_PYBIND11_ROOT}/${_name}")
      endforeach()
      file(REMOVE_RECURSE "${_inner_root}")
    endif()
  endif()

  file(REMOVE "${_TAR_PATH}")

  if(NOT EXISTS "${_PYBIND11_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
      "pybind11 download succeeded but CMakeLists.txt not found.\n"
      "Expected at: ${_PYBIND11_ROOT}/CMakeLists.txt\n"
      "The tarball layout may have changed.")
  endif()
  message(STATUS "co-py: pybind11 installed to ${_PYBIND11_ROOT}")
endif()

# --- Configure pybind11 ---
add_subdirectory(${_PYBIND11_ROOT} ${CMAKE_BINARY_DIR}/pybind11)
message(STATUS "co-py: pybind11 ${pybind11_VERSION} ready")
message(STATUS "co-py: Python ${Python_VERSION} (${Python_EXECUTABLE})")
