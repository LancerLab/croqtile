add_library(MLIRCoIRCodeGenCPU STATIC EmitCC.cpp)
add_dependencies(MLIRCoIRCodeGenCPU CoIRTableGen)
target_include_directories(MLIRCoIRCodeGenCPU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRCodeGenCPU PUBLIC
  MLIRCoIRCodeGenCommon
  MLIRCoIRDialect
  MLIRCoIRDriver
  MLIRArithDialect
  MLIRSCFDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
