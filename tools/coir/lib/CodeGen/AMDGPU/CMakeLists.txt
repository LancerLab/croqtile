add_library(MLIRCoIRCodeGenAMDGPU STATIC EmitHIP.cpp)
add_dependencies(MLIRCoIRCodeGenAMDGPU CoIRTableGen)
target_include_directories(MLIRCoIRCodeGenAMDGPU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${COIR_INCLUDE_DIR}
  ${COIR_INCLUDE_DIR}/Dialect
  ${COIR_GEN_DIR}
)
target_link_libraries(MLIRCoIRCodeGenAMDGPU PRIVATE
  MLIRCoIRDialect
  MLIRCoIRDriver
  MLIRArithDialect
  MLIRSCFDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
