file(GLOB _GPU_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(MLIRCoIRCodeGenGPU STATIC ${_GPU_SRCS})
add_dependencies(MLIRCoIRCodeGenGPU CoIRTableGen)
target_include_directories(MLIRCoIRCodeGenGPU PUBLIC
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(MLIRCoIRCodeGenGPU PUBLIC
  MLIRCoIRCodeGenCommon
  MLIRCoIRDialect
  MLIRCoIRTransforms
  MLIRCoIRDriver

  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRArithDialect
  MLIRGPUDialect
  MLIRSCFDialect
  MLIRMemRefDialect

  # GPU->NVVM->LLVM lowering chain
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRGPUToNVVMTransforms
  MLIRGPUPipelines
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRIndexToLLVM
  MLIRMemRefToLLVM
  MLIRNVVMToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts

  # MLIR -> LLVM IR translation
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation

  # NVPTX backend
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo

  # Dialect extensions
  MLIRFuncAllExtensions

  # LLVM core for TargetMachine / PassManager
  LLVMTarget
  LLVMCore
  LLVMMC
  LLVMMCParser
  LLVMCodeGen
  LLVMSelectionDAG
  LLVMAsmPrinter
  LLVMAnalysis
  LLVMBitWriter
  LLVMTransformUtils
  LLVMScalarOpts
  LLVMipo
  LLVMObjCARCOpts
  LLVMInstCombine
  LLVMCodeGenTypes

  # x86 backend for host object emission
  LLVMX86CodeGen
  LLVMX86Desc
  LLVMX86Info
  LLVMX86AsmParser
)

# Clang-as-library for in-memory host C++ compilation.
if(Clang_FOUND)
  target_link_libraries(MLIRCoIRCodeGenGPU PUBLIC
    clangFrontend
    clangCodeGen
    clangSerialization
    clangParse
    clangSema
    clangAST
    clangLex
    clangBasic
    clangDriver
    clangEdit
    clangAnalysis
    clangSupport
  )
  target_compile_definitions(MLIRCoIRCodeGenGPU PUBLIC COIR_HAS_CLANG=1)
endif()
