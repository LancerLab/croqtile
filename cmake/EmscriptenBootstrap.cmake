# EmscriptenBootstrap.cmake
# Auto-download Emscripten SDK when building co-web (WASM target).
#
# Usage: include(cmake/EmscriptenBootstrap.cmake) inside if(EMSCRIPTEN) block.
#
# This module is only relevant for Emscripten cross-compilation.
# It validates the Emscripten toolchain is functional and reports its version.
#
# The Emscripten SDK must be installed BEFORE cmake runs (emcmake sets up
# CMAKE_TOOLCHAIN_FILE). This module only verifies and reports -- actual
# installation is handled by CMake's ExternalProject or the user.

if(NOT EMSCRIPTEN)
  return()
endif()

# emcmake sets CMAKE_C_COMPILER to emcc; use it to report version.
execute_process(
  COMMAND "${CMAKE_C_COMPILER}" --version
  OUTPUT_VARIABLE _emcc_ver
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _rc)

if(_rc EQUAL 0)
  string(REGEX MATCH "[^\n]+" _emcc_line "${_emcc_ver}")
  message(STATUS "Emscripten: ${_emcc_line}")
else()
  message(STATUS "Emscripten: version unknown (emcmake configured)")
endif()
