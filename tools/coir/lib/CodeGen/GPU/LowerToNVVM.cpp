//===- LowerToNVVM.cpp - Chain upstream GPU->NVVM->LLVM passes -----------===//
//
// Provides a helper that runs the full CoIR -> GPU -> NVVM -> LLVM dialect
// lowering pipeline. Called by NativePipeline to prepare IR for PTX emission.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/NativePipeline.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace coir {
namespace gpu {

static void ensureDialectExtensions(mlir::MLIRContext &ctx) {
  mlir::DialectRegistry registry;
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::NVVM::registerConvertGpuToNVVMInterface(registry);
  mlir::func::registerAllExtensions(registry);
  ctx.appendDialectRegistry(registry);
}

bool lowerToNVVM(mlir::ModuleOp module) {
  mlir::MLIRContext &ctx = *module.getContext();
  ensureDialectExtensions(ctx);
  mlir::PassManager pm(&ctx);
  pm.enableVerifier(true);

  pm.addPass(coir::createConvertToGPUPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  return mlir::succeeded(pm.run(module));
}

} // namespace gpu
} // namespace coir
