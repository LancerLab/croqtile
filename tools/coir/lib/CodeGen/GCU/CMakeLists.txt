add_library(MLIRCoIRCodeGenGCU STATIC EmitTopscc.cpp)
target_include_directories(MLIRCoIRCodeGenGCU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../include
)
target_link_libraries(MLIRCoIRCodeGenGCU PRIVATE
  MLIRCoIRCodeGenCommon
  MLIRCoIRDialect
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
