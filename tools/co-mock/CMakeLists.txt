add_executable(co-mock
  mock_main.cpp
  mock_interp.cpp
  mock_memory.cpp
  debugger.cpp
)

set_target_properties(co-mock PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_include_directories(co-mock PRIVATE
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(co-mock PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  ${EXTERN_LIBS}
  Threads::Threads
)
