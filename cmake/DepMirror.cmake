# DepMirror.cmake
# Selects the download order for dependency URLs. When the FTP_SERVER
# environment variable is set, it is used first with a URL of the form:
#
#   ftp://${FTP_SERVER}/<category>/<tarball>
#
# Usage:
#   include("${CMAKE_SOURCE_DIR}/cmake/DepMirror.cmake")
#   dep_mirror_fallback(<out_var> <tarball> <category>)
#   dep_download_order(<primary_out> <fallback_out>
#     <configured_url> <tarball> <category>)
#   dep_validate_cached_archive(<valid_out> <archive> <expected_md5>)
#
# Returns an empty string when FTP_SERVER is unset or empty.

function(dep_mirror_fallback out_var tarball category)
  if(DEFINED ENV{FTP_SERVER} AND NOT "$ENV{FTP_SERVER}" STREQUAL "")
    set(${out_var} "ftp://$ENV{FTP_SERVER}/${category}/${tarball}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(dep_validate_cached_archive valid_out archive expected_md5)
  if(NOT EXISTS "${archive}")
    set(${valid_out} FALSE PARENT_SCOPE)
    return()
  endif()

  if(expected_md5)
    file(MD5 "${archive}" _actual_md5)
    if(NOT _actual_md5 STREQUAL expected_md5)
      message(STATUS
        "Removing invalid cached archive ${archive}: "
        "expected MD5 ${expected_md5}, got ${_actual_md5}")
      file(REMOVE "${archive}")
      set(${valid_out} FALSE PARENT_SCOPE)
      return()
    endif()
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar tf "${archive}"
    RESULT_VARIABLE _archive_result
    OUTPUT_QUIET
    ERROR_QUIET)
  if(NOT _archive_result EQUAL 0)
    message(STATUS
      "Removing unreadable cached archive ${archive}")
    file(REMOVE "${archive}")
    set(${valid_out} FALSE PARENT_SCOPE)
    return()
  endif()

  set(${valid_out} TRUE PARENT_SCOPE)
endfunction()

function(dep_download_order primary_out fallback_out configured_url tarball category)
  dep_mirror_fallback(_mirror_url "${tarball}" "${category}")

  if(_mirror_url)
    set(_primary_url "${_mirror_url}")
    set(_fallback_url "${configured_url}")
    message(STATUS "${category}: using FTP_SERVER first")
  else()
    set(_primary_url "${configured_url}")
    set(_fallback_url "")
  endif()

  set(${primary_out} "${_primary_url}" PARENT_SCOPE)
  set(${fallback_out} "${_fallback_url}" PARENT_SCOPE)
endfunction()
