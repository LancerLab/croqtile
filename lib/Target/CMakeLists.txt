file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo.h CHOREO_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_types.h CHOREO_TYPES_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_types_cute.h CHOREO_TYPES_CUTE_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_cute.h CHOREO_CUTE_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_cc.h CHOREO_CC_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/precompiled/choreo_device_api.h CHOREO_DEVICE_API_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/precompiled/choreo_precompiled.cu CHOREO_PRECOMPILED_CU_CONTENT)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_types_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_types_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_types_cute_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_types_cute_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_cute_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_cc_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_cc_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_device_api_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_device_api_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_precompiled_cu_template.in
  ${CMAKE_BINARY_DIR}/choreo_precompiled_cu.inc
  @ONLY
)

# Dependency targets for configured headers.
# Defined here (not inside backends) so they are available regardless of
# which backends CROQ_TARGET enables.
add_custom_target(
  gpu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  ${CMAKE_BINARY_DIR}/choreo_types_header.inc
  ${CMAKE_BINARY_DIR}/choreo_types_cute_header.inc
  ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
)

add_custom_target(
  cpu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  ${CMAKE_BINARY_DIR}/choreo_types_header.inc
  ${CMAKE_BINARY_DIR}/choreo_cc_header.inc
)

add_library(targets STATIC target_stub.cpp)

file(GLOB TARGET_BACKENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(backend IN LISTS TARGET_BACKENDS)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${backend}
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${backend}/CMakeLists.txt)
    list(FIND CROQ_ENABLED_TARGETS "${backend}" _target_idx)
    if(_target_idx LESS 0)
      message(STATUS "Target backend '${backend}' skipped (not in CROQ_TARGET)")
      continue()
    endif()
    add_subdirectory(${backend})
    target_sources(targets PRIVATE $<TARGET_OBJECTS:${backend}Target>)
  endif()
endforeach()
