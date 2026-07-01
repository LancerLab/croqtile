# tools/co-py/CMakeLists.txt -- croqtile-python bindings
if(EMSCRIPTEN)
  message(STATUS "co-py: skipped (not supported under Emscripten)")
  return()
endif()

include(${CMAKE_SOURCE_DIR}/cmake/PythonBootstrap.cmake)

# Parent object libraries need PIC for shared library linking
set_target_properties(parse core codegen support pp targets
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_core NO_EXTRAS
  src/croqtile/_core.cpp
  ${CMAKE_SOURCE_DIR}/tools/sdk/public_api.cpp
)

target_compile_definitions(_core PRIVATE
  CHOREO_SDK_VERSION="${VERSION_FROM_FILE}")

target_link_libraries(_core PRIVATE
  parse core codegen support pp targets Threads::Threads)

target_include_directories(_core PRIVATE
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/runtime
  ${CMAKE_SOURCE_DIR}/tools/sdk
)

# Copy .so next to Python source for PYTHONPATH dev workflow
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_core>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/croqtile/
)
