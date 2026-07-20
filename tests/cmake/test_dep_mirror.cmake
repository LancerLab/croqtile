set(ENV{FTP_SERVER} "mirror.example.test")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/DepMirror.cmake")

function(assert_equal actual expected label)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${label}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

dep_download_order(primary fallback
  "https://upstream.example/gtest.tar.gz" "gtest.tar.gz" gtest)
assert_equal("${primary}"
  "ftp://mirror.example.test/gtest/gtest.tar.gz" "FTP primary")
assert_equal("${fallback}" "https://upstream.example/gtest.tar.gz"
  "configured fallback")

unset(ENV{FTP_SERVER})
dep_download_order(primary fallback
  "https://upstream.example/llvm.tar.gz" "llvm.tar.gz" llvm)
assert_equal("${primary}" "https://upstream.example/llvm.tar.gz"
  "configured primary")
assert_equal("${fallback}" "" "no fallback")

set(_test_dir "${CMAKE_CURRENT_LIST_DIR}/../../build/dep-mirror-test")
set(_test_archive "${_test_dir}/archive.tar")
file(MAKE_DIRECTORY "${_test_dir}")
file(WRITE "${_test_dir}/fixture.txt" "valid archive fixture")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar cf "${_test_archive}" fixture.txt
  WORKING_DIRECTORY "${_test_dir}"
  RESULT_VARIABLE _create_result)
if(NOT _create_result EQUAL 0)
  message(FATAL_ERROR "Failed to create cached archive fixture")
endif()
file(MD5 "${_test_archive}" _test_md5)

dep_validate_cached_archive(
  cached_valid "${_test_archive}" "${_test_md5}")
assert_equal("${cached_valid}" "TRUE" "valid cached archive")

dep_validate_cached_archive(
  cached_valid "${_test_archive}" "00000000000000000000000000000000")
assert_equal("${cached_valid}" "FALSE" "invalid cached archive")
if(EXISTS "${_test_archive}")
  message(FATAL_ERROR "Invalid cached archive was not removed")
endif()

file(WRITE "${_test_archive}" "truncated archive fixture")
dep_validate_cached_archive(cached_valid "${_test_archive}" "")
assert_equal("${cached_valid}" "FALSE" "unreadable cached archive")
if(EXISTS "${_test_archive}")
  message(FATAL_ERROR "Unreadable cached archive was not removed")
endif()
file(REMOVE_RECURSE "${_test_dir}")

message(STATUS "Dependency mirror tests passed")
