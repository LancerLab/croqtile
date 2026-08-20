add_library(MLIRCoIRCodeGenCommon STATIC
  ../CoIREmitterBase.cpp
  ../CoIRKernelLowering.cpp
)

# CoIREmitterBase.cpp includes generated TableGen headers
# (Dialect/CoIR/CoIRDialect.h -> "CoIR/CoIRDialect.h.inc"), so this library
# must build after CoIRTableGen has produced them.
add_dependencies(MLIRCoIRCodeGenCommon CoIRTableGen)

target_include_directories(MLIRCoIRCodeGenCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
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
