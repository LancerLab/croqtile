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

add_subdirectory(Target)

add_custom_target(
  parser_deps ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/scanner.yy.cc
  ${CMAKE_BINARY_DIR}/parser.tab.cc
  ${CMAKE_BINARY_DIR}/parser.tab.hh
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
  typeinfer.cpp
  types.cpp
  valno.cpp
  visitor.cpp
  liveness_analysis.cpp
  loop_vectorize.cpp
  mem_reuse.cpp
  diversity_analysis.cpp
  scalar_evolution.cpp
  pipeline.cpp
  target_registry.cpp
  target_utils.cpp
  target.cpp
)

# Define the source files for your codegen library
set(CODEGEN_SOURCES
  codegen.cpp
  codegen_utils.cpp
)

set(CL_SOURCES
  command_line.cpp
)

set(PP_SOURCES
  preprocess.cpp
)

# Add the libraries
add_library(parse OBJECT ${PARSER_SOURCES})
add_library(core OBJECT ${CORE_SOURCES})
add_library(codegen OBJECT ${CODEGEN_SOURCES})
add_library(support OBJECT ${CL_SOURCES})
add_library(pp OBJECT ${PP_SOURCES})

set(CHOREO_COMPONENTS
  parse
  core
  codegen
  targets
  pp
  support
)

add_dependencies(parse parser_deps)

target_include_directories(parse PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(core PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(codegen PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(support PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})
target_include_directories(pp PRIVATE ${WORK_DIR} ${CMAKE_SOURCE_DIR})

target_compile_definitions(support PRIVATE
  __CHOREO_DEFAULT_TARGET__="${CHOREO_DEFAULT_TARGET}"
)
