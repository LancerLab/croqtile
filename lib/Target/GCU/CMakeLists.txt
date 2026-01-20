file(READ ${CMAKE_CURRENT_SOURCE_DIR}/factor_script.sh FACTOR_SCRIPT_CONTENT)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/factor_script_template.in
  ${CMAKE_BINARY_DIR}/factor_script.inc
  @ONLY
)

add_custom_target(
  gcu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/factor_script.inc
  ${CMAKE_BINARY_DIR}/choreo_header.inc
)

set(GCU_TARGET_SOURCES
  topscc_target.cpp
  topscc_preprocess.cpp
  topscc_codegen.cpp
  factor_target.cpp
  factor_codegen.cpp
  factor_codegen_types.cpp
)

add_library(GCUTarget OBJECT ${GCU_TARGET_SOURCES})
add_dependencies(GCUTarget gpu_deps)

target_include_directories(GCUTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
)

target_compile_definitions(GCUTarget PRIVATE
  __CHOREO_FACTOR_DIR__="${TOOLCHAIN_DIR}"
  __CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}"
)

if(EXISTS "${TOOLCHAIN_DIR}/lib/libacoreop.bc")
  target_compile_definitions(GCUTarget PRIVATE
    __CHOREO_GCU_ACORE_DIR__="${TOOLCHAIN_DIR}"
  )
endif()
