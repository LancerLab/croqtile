//===- EmitPTX.cpp - Translate LLVM dialect to PTX via NVPTX backend ------===//
//
// Translates an MLIR module (in LLVM dialect) to LLVM IR, then uses the NVPTX
// TargetMachine to emit PTX assembly. Part of the native GPU codegen pipeline.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/NativePipeline.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace coir {
namespace gpu {

static void ensureNVPTXTargetInitialized() {
  static bool initialized = false;
  if (!initialized) {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    initialized = true;
  }
}

static std::unique_ptr<llvm::TargetMachine>
createNVPTXTargetMachine(llvm::StringRef arch) {
  ensureNVPTXTargetInitialized();

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget("nvptx64-nvidia-cuda", error);
  if (!target) {
    llvm::errs() << "NVPTX target not found: " << error << "\n";
    return nullptr;
  }

  llvm::TargetOptions opts;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      "nvptx64-nvidia-cuda", arch, "+ptx78", opts, llvm::Reloc::PIC_));
}

std::string emitPTX(mlir::ModuleOp gpuModule, llvm::StringRef arch) {
  mlir::MLIRContext &ctx = *gpuModule.getContext();

  mlir::registerBuiltinDialectTranslation(ctx);
  mlir::registerLLVMDialectTranslation(ctx);
  mlir::registerNVVMDialectTranslation(ctx);
  mlir::registerGPUDialectTranslation(ctx);

  llvm::LLVMContext llvmCtx;
  auto llvmMod = mlir::translateModuleToLLVMIR(gpuModule, llvmCtx);
  if (!llvmMod) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR\n";
    return "";
  }

  auto tm = createNVPTXTargetMachine(arch);
  if (!tm) return "";

  llvmMod->setTargetTriple("nvptx64-nvidia-cuda");
  llvmMod->setDataLayout(tm->createDataLayout());

  llvm::SmallVector<char> ptxBytes;
  llvm::raw_svector_ostream ptxStream(ptxBytes);
  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, ptxStream, nullptr,
                              llvm::CodeGenFileType::AssemblyFile)) {
    llvm::errs() << "NVPTX backend cannot emit assembly\n";
    return "";
  }
  pm.run(*llvmMod);

  return std::string(ptxBytes.begin(), ptxBytes.end());
}

std::string emitPTXFromCoIR(mlir::ModuleOp module, llvm::StringRef arch) {
  mlir::ModuleOp clone = module.clone();

  if (!lowerToNVVM(clone)) {
    llvm::errs() << "GPU->NVVM lowering failed\n";
    clone->erase();
    return "";
  }

  // Collect all gpu.module ops (one per kernel after outlining).
  llvm::SmallVector<mlir::gpu::GPUModuleOp> gpuMods;
  clone.walk([&](mlir::gpu::GPUModuleOp m) { gpuMods.push_back(m); });

  if (gpuMods.empty()) {
    llvm::errs() << "No gpu.module found after lowering\n";
    clone->erase();
    return "";
  }

  // Merge all gpu.module bodies into a single temporary module for
  // translation to a single PTX compilation unit.
  mlir::OpBuilder builder(clone.getContext());
  auto tempModule = builder.create<mlir::ModuleOp>(clone.getLoc());
  llvm::DenseSet<llvm::StringRef> seenSymbols;
  builder.setInsertionPointToStart(tempModule.getBody());
  for (auto gpuMod : gpuMods) {
    for (auto &op : llvm::make_early_inc_range(
             gpuMod.getBody()->getOperations())) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      if (auto sym = op.getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName())) {
        if (!seenSymbols.insert(sym.getValue()).second)
          continue;
      }
      op.moveBefore(tempModule.getBody(), tempModule.getBody()->end());
    }
  }

  if (getenv("COIR_DUMP_TEMP_MODULE"))
    tempModule.dump();
  std::string ptx = emitPTX(tempModule, arch);
  tempModule->erase();
  clone->erase();
  return ptx;
}

} // namespace gpu
} // namespace coir
