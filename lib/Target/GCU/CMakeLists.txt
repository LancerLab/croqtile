file(READ ${CMAKE_CURRENT_SOURCE_DIR}/runtime/topscc_header.h TOPSCC_HEADER_CONTENT)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/topscc_header_template.in
  ${CMAKE_BINARY_DIR}/topscc_header.inc
  @ONLY
)

if(EXISTS "${CMAKE_SOURCE_DIR}/runtime/gcu/lib_fallback.h")
  file(READ ${CMAKE_SOURCE_DIR}/runtime/gcu/lib_fallback.h LIB_FALLBACK_HEADER_CONTENT)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/acore_runtime_template.in
    ${CMAKE_BINARY_DIR}/acore_runtime.inc
    @ONLY
  )
endif()

add_custom_target(
  gcu_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  ${CMAKE_BINARY_DIR}/choreo_types_header.inc
  ${CMAKE_BINARY_DIR}/topscc_header.inc
)

set(GCU_TARGET_SOURCES
  topscc_target.cpp
  topscc_preprocess.cpp
  topscc_codegen.cpp
  lower_libcall.cpp
)

add_library(GCUTarget OBJECT ${GCU_TARGET_SOURCES})
add_dependencies(GCUTarget gpu_deps)

target_include_directories(GCUTarget
  PUBLIC include
  PRIVATE ${CMAKE_SOURCE_DIR}
)

target_compile_definitions(GCUTarget PRIVATE
  __CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}"
)

if(NOT EXISTS "${TOOLCHAIN_DIR}/lib/libacoreop.bc")
  message(WARNING "acore library not found at ${TOOLCHAIN_DIR}/lib/libacoreop.bc. "
    "MMA and __lib_* codegen will be disabled.")
else()
  target_compile_definitions(GCUTarget PRIVATE
    __CHOREO_GCU_ACORE_DIR__="${TOOLCHAIN_DIR}"
  )
endif()
