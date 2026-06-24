#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "CodeGen/GPU/NativePipeline.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/Passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::NVVM::NVVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::vector::VectorDialect>();
  registry.insert<coir::CoIRDialect>();

  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);
  mlir::func::registerAllExtensions(registry);

  coir::registerCoIRPasses();
  mlir::registerPass(coir::createConvertToGPUPass);

  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertGpuOpsToNVVMOps();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerGpuKernelOutliningPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "coir-opt\n", registry));
}
