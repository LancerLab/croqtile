file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo.h CHOREO_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_cute.h CHOREO_CUTE_HEADER_CONTENT)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  @ONLY
)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_cute_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_cute_header.inc
  @ONLY
)

add_library(targets STATIC)

file(GLOB TARGET_BACKENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(backend IN LISTS TARGET_BACKENDS)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${backend}
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${backend}/CMakeLists.txt)
    add_subdirectory(${backend})
    # Each backend must define a object library named `${backend}Target`
    target_sources(targets PRIVATE $<TARGET_OBJECTS:${backend}Target>)
  endif()
endforeach()
