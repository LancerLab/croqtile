cmake_minimum_required(VERSION 3.22)
project(Choreo VERSION 0.0.2)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# use ccache if exists
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found, compilation will not be cached")
endif()

# allow diagnotics-color
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fdiagnostics-color)
endif()

# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/tools")

set(SYMBOLIC_DIR "${CMAKE_SOURCE_DIR}/extern/ginac")
set(CLN_DIR "${SYMBOLIC_DIR}/cln-1.3.7/install")
set(GINAC_DIR "${SYMBOLIC_DIR}/ginac-1.8.7/install")

set(FLEX_PATH "${TOOLCHAIN_DIR}/bin/flex")
set(BISON_PATH "${TOOLCHAIN_DIR}/bin/bison")
set(BISON_ENV "BISON_PKGDATADIR=${TOOLCHAIN_DIR}/shared/bison/")

add_compile_options(-Wall -Wextra -g  -D__CHOREO_FACTOR_DIR__="${TOOLCHAIN_DIR}" -D__CHOREO_CUDA_DIR__="${TOOLCHAIN_DIR}" -D__CHOREO_TOPSCC_DIR__="${TOOLCHAIN_DIR}")

# Include generated files
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CLN_DIR}/include ${GINAC_DIR}/include)

# add code sources and rules
add_subdirectory(lib)

# preprocessor
add_executable(copp
  utils/choreo_preprocess.cpp
)

# Add the main executable
add_executable(choreo
  utils/choreo_main.cpp
)
target_include_directories(choreo PRIVATE ${CMAKE_SOURCE_DIR})

# Link the library to the executable
# porting from config -Wl,-rpath -Wl,$(GINAC_DIR)/install/lib
target_link_libraries(choreo PRIVATE
  parser_obj
  scanner_obj
  cl_obj
  codegen
  ${CLN_DIR}/lib/libcln.so
  ${GINAC_DIR}/lib/libginac.so
)

target_link_libraries(copp PRIVATE
  parser_obj
  scanner_obj
  codegen
  ${CLN_DIR}/lib/libcln.so
  ${GINAC_DIR}/lib/libginac.so
)

# Install the executable and library
install(TARGETS choreo copp codegen
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Install generated headers
install(FILES choreo_header.inc choreo_cuda_header.inc
        DESTINATION include)

