add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/scanner.yy.cc
  COMMAND ${FLEX_PATH} -o ${CMAKE_BINARY_DIR}/scanner.yy.cc ${CMAKE_SOURCE_DIR}/lib/scanner.l
  DEPENDS ${CMAKE_SOURCE_DIR}/lib/scanner.l
  COMMENT "Generating scanner.yy.cc with flex"
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_BINARY_DIR}/parser.tab.hh
  COMMAND ${BISON_ENV} ${BISON_PATH} -t -d -o ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_SOURCE_DIR}/lib/parser.yy
  DEPENDS ${CMAKE_SOURCE_DIR}/lib/parser.yy
  COMMENT "Generating parser.tab.cc and parser.tab.hh with bison"
)

file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo.h CHOREO_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/runtime/choreo_cuda.h CHOREO_CUDA_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/scripts/factor_script.sh FACTOR_SCRIPT_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/scripts/cuda_script.sh CUDA_SCRIPT_CONTENT)

configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_header.inc
  @ONLY
)
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/choreo_cuda_header_template.in
  ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc
  @ONLY
)
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/factor_script_template.in
  ${CMAKE_BINARY_DIR}/factor_script.inc
  @ONLY
)
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/cuda_script_template.in
  ${CMAKE_BINARY_DIR}/cuda_script.inc
  @ONLY
)

add_custom_target(
  parser_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/scanner.yy.cc
  ${CMAKE_BINARY_DIR}/parser.tab.cc
  ${CMAKE_BINARY_DIR}/parser.tab.hh
)

add_custom_target(
  codegen_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/cuda_script.inc
  ${CMAKE_BINARY_DIR}/factor_script.inc
  ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc
  ${CMAKE_BINARY_DIR}/choreo_header.inc
)

set(PARSER_SOURCES
  ${CMAKE_BINARY_DIR}/parser.tab.cc
  ${CMAKE_BINARY_DIR}/scanner.yy.cc
)

set(CORE_SOURCES
  ast.cpp
  earlysema.cpp
  semacheck.cpp
  shapeinfer.cpp
  symtab.cpp
  sym_replace.cpp
  typeinfer.cpp
  types.cpp
  valno.cpp
  visitor.cpp
  liveness_analysis.cpp
  mem_reuse.cpp
)

# Define the source files for your codegen library
set(CODEGEN_SOURCES
  codegen.cpp
  codegen_factor.cpp
  codegen_factor_types.cpp
  codegen_topscc.cpp
  codegen_cuda.cpp
  codegen_cuda_types.cpp
  codegen_cute.cpp
)

set(CL_SOURCES
  command_line.cpp
)

# Add the libraries
add_library(parse OBJECT ${PARSER_SOURCES})
add_library(core OBJECT ${CORE_SOURCES})
add_library(codegen OBJECT ${CODEGEN_SOURCES})
add_library(support OBJECT ${CL_SOURCES})

add_dependencies(parse parser_deps)
add_dependencies(codegen codegen_deps)

target_include_directories(parse PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(core PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(codegen PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(support PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})

target_compile_definitions(codegen PRIVATE
  __CHOREO_FACTOR_DIR__="${TOOLCHAIN_DIR}"
  __CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}"
)

if(ENABLE_CUDA)
  target_include_directories(codegen PRIVATE ${CUDA_HOME}/include)
  link_directories(${CUDA_HOME}/lib64 ${CUDA_HOME}/targets/x86_64-linux/lib)

  target_compile_definitions(codegen PRIVATE
    __CHOREO_CUDA_DIR__=${CUDA_HOME}
  )
endif()

if(ENABLE_CUTE)
  target_include_directories(codegen PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include  # for lightweight helpers (optional)
  )

  target_compile_definitions(codegen PRIVATE
    __CHOREO_CUTE_DIR__=${CUTLASS_DIR}
  )
endif()

if(EXISTS "${TOOLCHAIN_DIR}/lib/libacoreop.bc")
  target_compile_definitions(codegen PRIVATE
    __CHOREO_GCU_ACORE_DIR__="${TOOLCHAIN_DIR}"
  )
endif()
