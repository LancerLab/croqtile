add_custom_target(
  cpu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  ${CMAKE_BINARY_DIR}/choreo_types_header.inc
  ${CMAKE_BINARY_DIR}/choreo_cc_header.inc
)

set(CPU_TARGET_SOURCES
  cc_codegen.cpp
  cc_target.cpp
)

add_library(CPUTarget OBJECT ${CPU_TARGET_SOURCES})
add_dependencies(CPUTarget cpu_deps)

target_include_directories(CPUTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
)
