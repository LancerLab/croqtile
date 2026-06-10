# tools/coir/CMakeLists.txt
# CoIR tooling -- MLIR-based IR layer for Choreo
#
# Builds three tools:
#   coir-gen     : Choreo AST -> CoIR MLIR translation
#   coir-opt     : CoIR MLIR optimizer (pass pipeline)
#   coir-codegen : CoIR MLIR -> C++ source emission

if(EMSCRIPTEN)
  message(STATUS "coir: skipped (not supported under Emscripten)")
  return()
endif()

# ------------------------------------------------------------------
# LLVM/MLIR dependency (via LLVMBootstrap)
# ------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/LLVMBootstrap.cmake)

# ------------------------------------------------------------------
# Local include paths
# ------------------------------------------------------------------
set(COIR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(COIR_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR})

# include/Dialect/CoIR/ holds source headers that #include "CoIR/Foo.h.inc"
# Generated .inc files go to ${COIR_GEN_DIR}/CoIR/ so that relative include works.
include_directories(${COIR_INCLUDE_DIR})
include_directories(${COIR_INCLUDE_DIR}/Dialect)
include_directories(${COIR_GEN_DIR})

# ------------------------------------------------------------------
# TableGen: generate C++ from CoIR ODS definitions
# ------------------------------------------------------------------

# --- Dialect ---
set(LLVM_TARGET_DEFINITIONS ${COIR_INCLUDE_DIR}/Dialect/CoIR/CoIRDialect.td)
mlir_tablegen(CoIR/CoIRDialect.h.inc -gen-dialect-decls -dialect coir
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIRDialect.cpp.inc -gen-dialect-defs -dialect coir
  -I ${COIR_INCLUDE_DIR}/Dialect)

# --- Attributes ---
set(LLVM_TARGET_DEFINITIONS ${COIR_INCLUDE_DIR}/Dialect/CoIR/CoIRAttrs.td)
mlir_tablegen(CoIR/CoIRAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=coir
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIRAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=coir
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIREnums.h.inc -gen-enum-decls
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIREnums.cpp.inc -gen-enum-defs
  -I ${COIR_INCLUDE_DIR}/Dialect)

# --- Ops ---
set(LLVM_TARGET_DEFINITIONS ${COIR_INCLUDE_DIR}/Dialect/CoIR/CoIROps.td)
mlir_tablegen(CoIR/CoIROps.h.inc -gen-op-decls
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIROps.cpp.inc -gen-op-defs
  -I ${COIR_INCLUDE_DIR}/Dialect)

# --- Types ---
set(LLVM_TARGET_DEFINITIONS ${COIR_INCLUDE_DIR}/Dialect/CoIR/CoIRTypes.td)
mlir_tablegen(CoIR/CoIRTypes.h.inc -gen-typedef-decls
  -I ${COIR_INCLUDE_DIR}/Dialect)
mlir_tablegen(CoIR/CoIRTypes.cpp.inc -gen-typedef-defs
  -I ${COIR_INCLUDE_DIR}/Dialect)

# --- Passes ---
set(LLVM_TARGET_DEFINITIONS ${COIR_INCLUDE_DIR}/Dialect/CoIR/Passes.td)
mlir_tablegen(CoIR/Passes.h.inc -gen-pass-decls -name CoIR
  -I ${COIR_INCLUDE_DIR}/Dialect)

add_custom_target(CoIRTableGen DEPENDS
  ${COIR_GEN_DIR}/CoIR/CoIRDialect.h.inc
  ${COIR_GEN_DIR}/CoIR/CoIRDialect.cpp.inc
  ${COIR_GEN_DIR}/CoIR/CoIRAttrs.h.inc
  ${COIR_GEN_DIR}/CoIR/CoIRAttrs.cpp.inc
  ${COIR_GEN_DIR}/CoIR/CoIREnums.h.inc
  ${COIR_GEN_DIR}/CoIR/CoIREnums.cpp.inc
  ${COIR_GEN_DIR}/CoIR/CoIROps.h.inc
  ${COIR_GEN_DIR}/CoIR/CoIROps.cpp.inc
  ${COIR_GEN_DIR}/CoIR/CoIRTypes.h.inc
  ${COIR_GEN_DIR}/CoIR/CoIRTypes.cpp.inc
  ${COIR_GEN_DIR}/CoIR/Passes.h.inc
)

# ------------------------------------------------------------------
# CoIR Dialect library
# ------------------------------------------------------------------
add_library(MLIRCoIRDialect STATIC
  lib/IR/CoIRDialect.cpp
  lib/IR/CoIROps.cpp
  lib/IR/CoIRTypes.cpp
)
add_dependencies(MLIRCoIRDialect CoIRTableGen)
target_include_directories(MLIRCoIRDialect PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRDialect PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRAsyncDialect
  MLIRGPUDialect
  MLIRSupport
)

# ------------------------------------------------------------------
# CoIR Transforms library (target-agnostic lowering passes)
# ------------------------------------------------------------------
add_library(MLIRCoIRTransforms STATIC
  lib/Transforms/ClassifyCopies.cpp
  lib/Transforms/LowerMMA.cpp
  lib/Transforms/LowerCopy.cpp
)
add_dependencies(MLIRCoIRTransforms CoIRTableGen)
target_include_directories(MLIRCoIRTransforms PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRTransforms PUBLIC
  MLIRCoIRDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRArithDialect
  MLIRGPUDialect
  MLIRSCFDialect
)

# ------------------------------------------------------------------
# CoIR CodeGen GPU library (CUDA/CUTE emission)
# ------------------------------------------------------------------
add_library(MLIRCoIRCodeGenGPU STATIC
  lib/CodeGen/GPU/EmitCUDA.cpp
)
add_dependencies(MLIRCoIRCodeGenGPU CoIRTableGen)
target_include_directories(MLIRCoIRCodeGenGPU PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRCodeGenGPU PUBLIC
  MLIRCoIRDialect
  MLIRCoIRTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRArithDialect
  MLIRGPUDialect
  MLIRSCFDialect
)

# ------------------------------------------------------------------
# coir-opt: MLIR optimizer for CoIR dialect
# ------------------------------------------------------------------
add_executable(coir-opt lib/Opt/CoIROpt.cpp)
target_include_directories(coir-opt PRIVATE
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(coir-opt PRIVATE
  MLIRCoIRDialect
  MLIRCoIRTransforms
  MLIRCoIRCodeGenGPU
  MLIROptLib
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRVectorDialect
  MLIRSupport
  MLIRParser
  LLVMSupport
)

# ------------------------------------------------------------------
# coir-codegen: CoIR -> target source emission (--target selects backend)
# ------------------------------------------------------------------
add_executable(coir-codegen lib/CodeGen/GPU/CuteCodeGen.cpp lib/Driver/Pipeline.cpp)
target_include_directories(coir-codegen PRIVATE
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/CodeGen
)
target_link_libraries(coir-codegen PRIVATE
  MLIRCoIRDialect
  MLIRCoIRTransforms
  -Wl,--whole-archive MLIRCoIRCodeGenGPU -Wl,--no-whole-archive
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRSupport
  MLIRParser
  LLVMSupport
)

# ------------------------------------------------------------------
# Additional codegen targets (auto-discovered from lib/CodeGen/*/)
# ------------------------------------------------------------------
# Auto-discover extra codegen backends (beyond GPU)
file(GLOB _extra_codegen_dirs
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/CodeGen/*/CMakeLists.txt")
foreach(_extra_cmake ${_extra_codegen_dirs})
  get_filename_component(_dir "${_extra_cmake}" DIRECTORY)
  get_filename_component(_name "${_dir}" NAME)
  if(NOT _name STREQUAL "GPU")
    add_subdirectory("${_dir}")
    target_link_libraries(coir-codegen PRIVATE
      -Wl,--whole-archive MLIRCoIRCodeGen${_name} -Wl,--no-whole-archive)
    target_include_directories(coir-codegen PRIVATE "${_dir}")
  endif()
endforeach()


# ------------------------------------------------------------------
# co2ir: Choreo AST -> CoIR MLIR translation
# ------------------------------------------------------------------
add_executable(co2ir
  lib/ASTIRGen/CoIRGen.cpp
  lib/ASTIRGen/ASTCoIRGen.cpp
)
target_include_directories(co2ir PRIVATE
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ASTIRGen
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/runtime
  ${CMAKE_BINARY_DIR}
)
add_dependencies(co2ir CoIRTableGen)
target_link_libraries(co2ir PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  MLIRCoIRDialect
  MLIRCoIRTransforms
  MLIRCoIRCodeGenGPU
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRSupport
  LLVMSupport
  stdc++fs
  Threads::Threads
)

# ------------------------------------------------------------------
# cocc: end-to-end CoIR compiler driver (.co -> target code)
# ------------------------------------------------------------------

add_executable(cocc
  lib/Driver/CoCC.cpp
  lib/Driver/Pipeline.cpp
  lib/ASTIRGen/ASTCoIRGen.cpp
)
target_include_directories(cocc PRIVATE
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/ASTIRGen
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/runtime
  ${CMAKE_BINARY_DIR}
)
add_dependencies(cocc CoIRTableGen)
target_link_libraries(cocc PRIVATE
  parse
  core
  codegen
  support
  pp
  -Wl,--whole-archive targets MLIRCoIRCodeGenGPU -Wl,--no-whole-archive
  MLIRCoIRDialect
  MLIRCoIRTransforms
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRSupport
  LLVMSupport
  stdc++fs
  Threads::Threads
)

# Link auxiliary codegen targets (whole-archive to pull in static registrations)
foreach(_extra_cmake ${_extra_codegen_dirs})
  get_filename_component(_dir "${_extra_cmake}" DIRECTORY)
  get_filename_component(_name "${_dir}" NAME)
  if(NOT _name STREQUAL "GPU")
    target_link_libraries(cocc PRIVATE
      -Wl,--whole-archive MLIRCoIRCodeGen${_name} -Wl,--no-whole-archive)
    target_include_directories(cocc PRIVATE "${_dir}")
  endif()
endforeach()

# ------------------------------------------------------------------
# Install rules
# ------------------------------------------------------------------
install(TARGETS co2ir coir-opt coir-codegen cocc
  RUNTIME DESTINATION bin
  COMPONENT executables
)
