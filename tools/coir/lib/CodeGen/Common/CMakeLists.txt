add_library(MLIRCoIRCodeGenCommon STATIC
  ../CoIREmitterBase.cpp
  ../CoIRKernelLowering.cpp
)

target_include_directories(MLIRCoIRCodeGenCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${COIR_INCLUDE_DIR}
)

target_link_libraries(MLIRCoIRCodeGenCommon PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRIR
  LLVMCore
  LLVMSupport
)
