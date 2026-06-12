if(NOT EMSCRIPTEN)
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/scanner.yy.cc
    COMMAND ${FLEX_EXECUTABLE} -o ${CMAKE_BINARY_DIR}/scanner.yy.cc ${CMAKE_SOURCE_DIR}/lib/scanner.l
    DEPENDS ${CMAKE_SOURCE_DIR}/lib/scanner.l
    COMMENT "Generating scanner.yy.cc with flex"
  )

  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_BINARY_DIR}/parser.tab.hh
    COMMAND ${BISON_ENV_CMD} ${BISON_EXECUTABLE} -t -d -o ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_SOURCE_DIR}/lib/parser.yy
    DEPENDS ${CMAKE_SOURCE_DIR}/lib/parser.yy
    COMMENT "Generating parser.tab.cc and parser.tab.hh with bison"
  )
else()
  set(_NATIVE_BUILD ${CMAKE_SOURCE_DIR}/build)
  if(NOT EXISTS ${_NATIVE_BUILD}/parser.tab.cc)
    message(FATAL_ERROR
      "Emscripten builds require pre-generated parser files.\n"
      "Run 'make build' first to generate them in build/.")
  endif()
  configure_file(${_NATIVE_BUILD}/scanner.yy.cc ${CMAKE_BINARY_DIR}/scanner.yy.cc COPYONLY)
  configure_file(${_NATIVE_BUILD}/parser.tab.cc ${CMAKE_BINARY_DIR}/parser.tab.cc COPYONLY)
  configure_file(${_NATIVE_BUILD}/parser.tab.hh ${CMAKE_BINARY_DIR}/parser.tab.hh COPYONLY)

  # FlexLexer.h is needed by scanner.hpp but not shipped with Emscripten.
  # Copy from the native build or known system paths.
  set(_FLEX_LEXER_CANDIDATES
    ${_NATIVE_BUILD}/FlexLexer.h
    /usr/include/FlexLexer.h
    /usr/local/include/FlexLexer.h
    ${CMAKE_SOURCE_DIR}/extern/include/FlexLexer.h
  )
  set(_FLEX_LEXER_FOUND FALSE)
  foreach(_candidate ${_FLEX_LEXER_CANDIDATES})
    if(EXISTS ${_candidate})
      configure_file(${_candidate} ${CMAKE_BINARY_DIR}/FlexLexer.h COPYONLY)
      set(_FLEX_LEXER_FOUND TRUE)
      break()
    endif()
  endforeach()
  if(NOT _FLEX_LEXER_FOUND)
    message(FATAL_ERROR "FlexLexer.h not found. Install flex or copy FlexLexer.h to extern/include/.")
  endif()
endif()

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
  assess.cpp
  assert_site.cpp
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
  sys_utils.cpp
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
