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

file(READ ${CMAKE_SOURCE_DIR}/utils/choreo.h CHOREO_HEADER_CONTENT)
file(READ ${CMAKE_SOURCE_DIR}/utils/choreo_cuda.h CHOREO_CUDA_HEADER_CONTENT)
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
  generate_sources ALL
  DEPENDS 
  ${CMAKE_BINARY_DIR}/scanner.yy.cc 
  ${CMAKE_BINARY_DIR}/parser.tab.cc 
  ${CMAKE_BINARY_DIR}/parser.tab.hh 
  ${CMAKE_BINARY_DIR}/cuda_script.inc
  ${CMAKE_BINARY_DIR}/factor_script.inc
  ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc
  ${CMAKE_BINARY_DIR}/choreo_header.inc
)

# Define the source files for your codegen library
set(CODEGEN_SOURCES
  ast.cpp
  codegen_cuda.cpp
  codegen_cuda_types.cpp
  codegen_factor.cpp
  codegen_factor_types.cpp
  codegen_topscc.cpp
  earlysema.cpp
  semacheck.cpp
  sym_replace.cpp
  typeinfer.cpp
  types.cpp
  valno.cpp
  visitor.cpp
)

add_library(parser_obj OBJECT ${CMAKE_BINARY_DIR}/parser.tab.cc)
target_include_directories(parser_obj PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_dependencies(parser_obj generate_sources)

add_library(scanner_obj OBJECT ${CMAKE_BINARY_DIR}/scanner.yy.cc)
target_include_directories(scanner_obj PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_dependencies(scanner_obj generate_sources)

# Add the codegen library
add_library(codegen STATIC 
  ${CODEGEN_SOURCES}
)
target_include_directories(codegen PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_link_libraries(codegen PRIVATE
  ${CLN_DIR}/lib/libcln.so 
  ${GINAC_DIR}/lib/libginac.so
)

target_compile_definitions(codegen PRIVATE
    __CHOREO_FACTOR_DIR__="${TOOLCHAIN_DIR}"
    __CHOREO_CUDA_DIR__="${TOOLCHAIN_DIR}"
    __CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}"
)

