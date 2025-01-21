cmake_minimum_required(VERSION 3.22)
project(Choreo VERSION 0.0.2)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the build directory (this is automatically handled by CMake)
set(BUILD_DIR ${CMAKE_BINARY_DIR}/build)

set(TOOLCHAIN_DIR "${CMAKE_SOURCE_DIR}/tools")

set(FLEX_PATH "${TOOLCHAIN_DIR}/bin/flex")
set(BISON_PATH "${TOOLCHAIN_DIR}/bin/bison")
set(BISON_ENV "BISON_PKGDATADIR=${TOOLCHAIN_DIR}/shared/bison/")


# Include generated files
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/utils)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/scanner.yy.cc
  COMMAND ${FLEX_PATH} -o ${CMAKE_BINARY_DIR}/scanner.yy.cc ${CMAKE_SOURCE_DIR}/lib/scanner.l
  DEPENDS ${CMAKE_SOURCE_DIR}/lib/scanner.l
  COMMENT "Generating scanner.yy.cc with flex"
)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_BINARY_DIR}/parser.tab.hh ${CMAKE_BINARY_DIR}/location.hh
  COMMAND ${BISON_ENV} ${BISON_PATH} -t -d -o ${CMAKE_BINARY_DIR}/parser.tab.cc ${CMAKE_SOURCE_DIR}/lib/parser.yy
  DEPENDS ${CMAKE_SOURCE_DIR}/lib/parser.yy
  COMMENT "Generating parser.tab.cc and parser.tab.hh with bison"
)

# Generate choreo_header.inc
# add_custom_command(
#   OUTPUT ${CMAKE_BINARY_DIR}/choreo_header.inc
#   COMMAND echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
#   COMMAND echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
#   COMMAND echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
#   COMMAND cat ${CMAKE_SOURCE_DIR}/utils/choreo.h >> $@
#   COMMAND echo ")\";" >> $@
#   COMMAND echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@
#   DEPENDS ${CMAKE_SOURCE_DIR}/utils/choreo.h
# )
#
# # Generate choreo_cuda_header.inc
# add_custom_command(
#   OUTPUT ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc
#   COMMAND echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
#   COMMAND echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
#   COMMAND echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
#   COMMAND cat ${CMAKE_SOURCE_DIR}/utils/choreo_cuda.h >> $@
#   COMMAND echo ")\";" >> $@
#   COMMAND echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@
#   DEPENDS ${CMAKE_SOURCE_DIR}/utils/choreo_cuda.h
# )
#
# # Generate factor_script.inc
# add_custom_command(
#   OUTPUT ${CMAKE_BINARY_DIR}/factor_script.inc
#   COMMAND echo "#ifndef __CHOREO_FACTOR_SCRIPT_H__" > $@
#   COMMAND echo "#define __CHOREO_FACTOR_SCRIPT_H__" >> $@
#   COMMAND echo -n "static const char* __factor_script_as_string = R\"__co_factor__(" >> $@
#   COMMAND cat ${CMAKE_SOURCE_DIR}/scripts/factor_script.sh >> $@
#   COMMAND echo ")__co_factor__\";" >> $@
#   COMMAND echo "#endif // __CHOREO_FACTOR_SCRIPT_H__" >> $@
#   DEPENDS ${CMAKE_SOURCE_DIR}/scripts/factor_script.sh
# )

# Generate cuda_script.inc
# add_custom_command(
#   OUTPUT ${CMAKE_BINARY_DIR}/cuda_script.inc
#   COMMAND ${CMAKE_COMMAND} -E echo "#ifndef __CHOREO_CUDA_SCRIPT_H__" > $@
#   COMMAND ${CMAKE_COMMAND} -E echo "#define __CHOREO_CUDA_SCRIPT_H__" >> $@
#   COMMAND ${CMAKE_COMMAND} -E echo "static const char* __cuda_script_as_string = R__co_cuda__" >> $@
#   COMMAND ${CMAKE_COMMAND} -E cat ${CMAKE_SOURCE_DIR}/scripts/cuda_script.sh >> $@
#   COMMAND echo "\)" >> $@
#   COMMAND ${CMAKE_COMMAND} -E echo "__co_cuda__;" >> $@
#   COMMAND ${CMAKE_COMMAND} -E echo "#endif // __CHOREO_CUDA_SCRIPT_H__" >> $@
#   DEPENDS ${CMAKE_SOURCE_DIR}/scripts/cuda_script.sh
# )
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
#
# add_custom_command(
#   OUTPUT ${CMAKE_BINARY_DIR}/cuda_script.inc
#   COMMAND ${CMAKE_COMMAND} -E echo "Generated CUDA script header"
#   DEPENDS ${CMAKE_SOURCE_DIR}/scripts/cuda_script.sh
#   COMMENT "Generating CUDA script header"
# )

add_custom_target(
  generate_sources ALL
  DEPENDS 
  ${CMAKE_BINARY_DIR}/scanner.yy.cc 
  ${CMAKE_BINARY_DIR}/parser.tab.cc 
  ${CMAKE_BINARY_DIR}/parser.tab.hh 
  ${CMAKE_BINARY_DIR}/location.hh
  ${CMAKE_BINARY_DIR}/cuda_script.inc
  ${CMAKE_BINARY_DIR}/factor_script.inc
  ${CMAKE_BINARY_DIR}/choreo_cuda_header.inc
  ${CMAKE_BINARY_DIR}/choreo_header.inc
)

# # Define the source files for your codegen library
# set(CODEGEN_SOURCES
#   lib/ast.cpp
#   lib/codegen_cuda.cpp
#   lib/codegen_cuda_types.cpp
#   lib/codegen_factor.cpp
#   lib/codegen_factor_types.cpp
#   lib/codegen_topscc.cpp
#   lib/earlysema.cpp
#   lib/semacheck.cpp
#   lib/sym_replace.cpp
#   lib/typeinfer.cpp
#   lib/types.cpp
#   lib/valno.cpp
#   lib/visitor.cpp
# )
#
# # Add the codegen library
# add_library(codegen STATIC 
#   ${CODEGEN_SOURCES}
# )
#
# # Add the executable
# add_executable(choreo
#   utils/choreo_main.cpp
#   utils/choreo_preprocess.cpp
#   ${CODEGEN_SOURCES}
# )
#
# # Link the library to the executable
# target_link_libraries(choreo codegen)
#
# # Specify the include directories for the executable target
# target_include_directories(choreo PRIVATE ${WORK_DIR} ${SRC_DIR})
#
# # Install the executable and library
# install(TARGETS choreo codegen
#   RUNTIME DESTINATION bin
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
# )
#
# # Install generated headers
# install(FILES choreo_header.inc choreo_cuda_header.inc
#         DESTINATION include)
#
