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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

struct GpuToNVVMWithSCFPass
    : public mlir::OperationPass<mlir::gpu::GPUModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuToNVVMWithSCFPass)
  GpuToNVVMWithSCFPass()
      : OperationPass(mlir::TypeID::get<GpuToNVVMWithSCFPass>()) {}
  llvm::StringRef getName() const override { return "GpuToNVVMWithSCF"; }
  llvm::StringRef getArgument() const override {
    return "coir-gpu-to-nvvm-with-scf";
  }
  llvm::StringRef getDescription() const override {
    return "GPU+SCF to NVVM/LLVM with unified type converter";
  }
  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<GpuToNVVMWithSCFPass>();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    mlir::MLIRContext &ctx = *gpuModule.getContext();

    mlir::LowerToLLVMOptions llvmOpts(&ctx);
    llvmOpts.useBarePtrCallConv = false;
    mlir::LLVMTypeConverter converter(&ctx, llvmOpts);
    mlir::configureGpuToNVVMTypeConverter(converter);

    mlir::ConversionTarget target(ctx);
    mlir::configureGpuToNVVMConversionLegality(target);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::NVVM::NVVMDialect>();
    target.addIllegalDialect<mlir::gpu::GPUDialect>();
    target.addLegalOp<mlir::gpu::GPUModuleOp>();
    target.addLegalOp<mlir::gpu::YieldOp>();

    mlir::RewritePatternSet patterns(&ctx);
    mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
    mlir::populateGpuWMMAToNVVMConversionPatterns(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);

    if (mlir::failed(mlir::applyPartialConversion(gpuModule, target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

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
      std::make_unique<GpuToNVVMWithSCFPass>());
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
