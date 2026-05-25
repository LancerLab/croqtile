if(NOT EMSCRIPTEN)
  message(STATUS "co-web: skipped (requires Emscripten cross-compilation)")
  return()
endif()

add_executable(co-web
  wasm_api.cpp
  ${CMAKE_SOURCE_DIR}/tools/co-mock/mock_interp.cpp
  ${CMAKE_SOURCE_DIR}/tools/co-mock/mock_memory.cpp
  ${CMAKE_SOURCE_DIR}/tools/co-mock/debugger.cpp
)

set_target_properties(co-web PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  SUFFIX ".js"
)

target_include_directories(co-web PRIVATE
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/runtime
  ${CMAKE_SOURCE_DIR}/tools/co-mock
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(co-web PRIVATE
  __CHOREO_DEFAULT_TARGET__="cc"
  CHOREO_SDK_VERSION="${CHOREO_SDK_VERSION}"
)

target_link_libraries(co-web PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
)

set_target_properties(co-web PROPERTIES
  LINK_FLAGS "\
    --bind \
    -s MODULARIZE=1 \
    -s EXPORT_NAME='ChoreoModule' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ENVIRONMENT='web,worker' \
    -s NO_EXIT_RUNTIME=1 \
    -s EXPORTED_RUNTIME_METHODS=['FS'] \
    -s FORCE_FILESYSTEM=1 \
    -s INITIAL_MEMORY=33554432 \
  "
)
