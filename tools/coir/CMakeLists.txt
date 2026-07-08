# tools/coir/CMakeLists.txt
# CoIR tooling -- MLIR-based IR layer for Choreo
#
# Builds tools:
#   co2ir    : Choreo AST -> CoIR MLIR translation
#   coir-opt : CoIR MLIR optimizer (pass pipeline)
#   cocc     : End-to-end CoIR compiler driver

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
# Common MLIR link libraries (shared across all CoIR targets)
# ------------------------------------------------------------------
set(COIR_MLIR_LIBS
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRVectorDialect
  MLIRSupport
  MLIRPass
  MLIRParser
  MLIRTransformUtils
  LLVMSupport
)

# ------------------------------------------------------------------
# lib/IR -- CoIR Dialect library
# ------------------------------------------------------------------
file(GLOB _COIR_IR_SRCS lib/IR/*.cpp)
add_library(MLIRCoIRDialect STATIC ${_COIR_IR_SRCS})
add_dependencies(MLIRCoIRDialect CoIRTableGen)
target_include_directories(MLIRCoIRDialect PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRDialect PUBLIC
  MLIRIR MLIRFuncDialect MLIRAsyncDialect MLIRGPUDialect MLIRSupport
  MLIRLoopLikeInterface
)

# ------------------------------------------------------------------
# lib/Transforms -- target-agnostic lowering passes
# ------------------------------------------------------------------
file(GLOB _COIR_TRANSFORMS_SRCS lib/Transforms/*.cpp)
add_library(MLIRCoIRTransforms STATIC ${_COIR_TRANSFORMS_SRCS})
add_dependencies(MLIRCoIRTransforms CoIRTableGen)
target_include_directories(MLIRCoIRTransforms PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRTransforms PUBLIC
  MLIRCoIRDialect MLIRIR MLIRPass MLIRTransformUtils MLIRSupport
  MLIRArithDialect MLIRGPUDialect MLIRSCFDialect
)

# ------------------------------------------------------------------
# lib/Driver -- Pipeline implementation (shared by tools)
# ------------------------------------------------------------------
file(GLOB _COIR_DRIVER_SRCS lib/Driver/Pipeline.cpp)
add_library(MLIRCoIRDriver STATIC ${_COIR_DRIVER_SRCS})
add_dependencies(MLIRCoIRDriver CoIRTableGen)
target_include_directories(MLIRCoIRDriver PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRDriver PUBLIC
  MLIRCoIRDialect MLIRCoIRTransforms MLIRIR MLIRPass MLIRSupport
)

# ------------------------------------------------------------------
# lib/CodeGen -- target-specific code generation (auto-discovered)
# ------------------------------------------------------------------
# Each sub-directory under lib/CodeGen/ with *.cpp files produces a library
# named MLIRCoIRCodeGen<DirName>.
set(COIR_CODEGEN_LIBS "")

file(GLOB _codegen_subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/lib/CodeGen
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/CodeGen/*")
foreach(_subdir ${_codegen_subdirs})
  set(_subdir_path "${CMAKE_CURRENT_SOURCE_DIR}/lib/CodeGen/${_subdir}")
  if(IS_DIRECTORY "${_subdir_path}")
    file(GLOB _subdir_srcs "${_subdir_path}/*.cpp")
    # Skip directories with a CMakeLists.txt (they self-manage via add_subdirectory)
    if(EXISTS "${_subdir_path}/CMakeLists.txt")
      add_subdirectory("${_subdir_path}")
      list(APPEND COIR_CODEGEN_LIBS "MLIRCoIRCodeGen${_subdir}")
    elseif(_subdir_srcs)
      add_library(MLIRCoIRCodeGen${_subdir} STATIC ${_subdir_srcs})
      add_dependencies(MLIRCoIRCodeGen${_subdir} CoIRTableGen)
      target_include_directories(MLIRCoIRCodeGen${_subdir} PUBLIC
        ${COIR_INCLUDE_DIR}
        ${COIR_INCLUDE_DIR}/Dialect
        ${COIR_GEN_DIR}
        ${_subdir_path}
      )
      target_link_libraries(MLIRCoIRCodeGen${_subdir} PUBLIC
        MLIRCoIRDialect MLIRCoIRTransforms MLIRCoIRDriver
        MLIRIR MLIRPass MLIRSupport MLIRArithDialect MLIRGPUDialect MLIRSCFDialect
        MLIRMemRefDialect
      )
      list(APPEND COIR_CODEGEN_LIBS "MLIRCoIRCodeGen${_subdir}")
    endif()
  endif()
endforeach()

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
  MLIROptLib
  ${COIR_MLIR_LIBS}
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRControlFlowDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRGPUToNVVMTransforms
  MLIRGPUPipelines
  MLIRGPUToGPURuntimeTransforms
  MLIRIndexToLLVM
  MLIRMemRefToLLVM
  MLIRNVVMToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
  MLIRGPUTransforms
  MLIRFuncAllExtensions
  MLIRNVVMToLLVMIRTranslation
)
# Link all codegen libs (for pass registration)
foreach(_lib ${COIR_CODEGEN_LIBS})
  target_link_libraries(coir-opt PRIVATE
    -Wl,--whole-archive ${_lib} -Wl,--no-whole-archive)
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
  parse core codegen support pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  MLIRCoIRDialect MLIRCoIRTransforms MLIRCoIRDriver
  ${COIR_MLIR_LIBS}
  stdc++fs Threads::Threads
)
# Link codegen libs for emission support
foreach(_lib ${COIR_CODEGEN_LIBS})
  target_link_libraries(co2ir PRIVATE
    -Wl,--whole-archive ${_lib} -Wl,--no-whole-archive)
endforeach()
# Release links pull in core .o files with huge DWARF sections (e.g.
# assess.cpp.o).  GNU ld can fail merging them with MLIR static libs
# (.LASF undefined reference in .debug_info).  Strip debug at link for
# these executables only; choreo/copp and Debug builds are unaffected.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options(co2ir PRIVATE "-Wl,--strip-debug")
endif()

# ------------------------------------------------------------------
# cocc: end-to-end CoIR compiler driver (.co -> target code)
# ------------------------------------------------------------------
add_executable(cocc
  lib/Driver/CoCC.cpp
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
  parse core codegen support pp
  -Wl,--whole-archive targets -Wl,--no-whole-archive
  MLIRCoIRDialect MLIRCoIRTransforms MLIRCoIRDriver
  ${COIR_MLIR_LIBS}
  stdc++fs Threads::Threads
)
# Link all codegen libs (whole-archive for static registrations)
foreach(_lib ${COIR_CODEGEN_LIBS})
  target_link_libraries(cocc PRIVATE
    -Wl,--whole-archive ${_lib} -Wl,--no-whole-archive)
endforeach()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options(cocc PRIVATE "-Wl,--strip-debug")
endif()

# ------------------------------------------------------------------
# Install rules
# ------------------------------------------------------------------
install(TARGETS co2ir coir-opt cocc
  RUNTIME DESTINATION bin
  COMPONENT executables
)
