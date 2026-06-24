//===- Pipeline.cpp - CoIR Pipeline class implementation ------------------===//
//
// Implements CoIR::Pipeline (EmitCoIR, Optimize, Lower, EmitSource,
// CompileBinary) and the module-level target attribute helpers.
//
// The pipeline has shared phases (Optimize, Lower) that run for every
// target, followed by target-specific codegen delegated to CodeGen backends.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace CoIR {

void StampTargetOnModule(mlir::ModuleOp module, llvm::StringRef target,
                         llvm::StringRef arch, llvm::StringRef mma_target,
                         bool has_tma, bool has_dma) {
  auto *ctx = module.getContext();
  if (!target.empty())
    module->setAttr("coir.target", mlir::StringAttr::get(ctx, target));
  if (!arch.empty())
    module->setAttr("coir.arch", mlir::StringAttr::get(ctx, arch));
  if (!mma_target.empty())
    module->setAttr("coir.mma_target", mlir::StringAttr::get(ctx, mma_target));
  module->setAttr("coir.has_tma", mlir::BoolAttr::get(ctx, has_tma));
  module->setAttr("coir.has_dma", mlir::BoolAttr::get(ctx, has_dma));
}

llvm::StringRef GetMMATarget(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::StringAttr>("coir.mma_target"))
    return attr.getValue();
  return {};
}

bool HasTMA(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::BoolAttr>("coir.has_tma"))
    return attr.getValue();
  return false;
}

bool HasDMA(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::BoolAttr>("coir.has_dma"))
    return attr.getValue();
  return false;
}

llvm::StringRef GetArch(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::StringAttr>("coir.arch"))
    return attr.getValue();
  return {};
}

void Pipeline::EmitCoIR(llvm::raw_ostream &os) {
  mlir::OpPrintingFlags flags;
  module_.print(os, flags);
  os << "\n";
}

bool Pipeline::Optimize() {
  // Stamp cost threshold as module attribute for EstimateAssertCost to read.
  module_->setAttr("coir.cost_threshold",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(&ctx_, 32), cost_threshold_));
  mlir::PassManager pm(&ctx_);
  pm.addPass(coir::createHoistAssertionsPass());
  pm.addPass(coir::createEstimateAssertCostPass());
  if (collect_stats_)
    pm.addPass(coir::createCollectAssertStatsPass());
  if (mlir::failed(pm.run(module_)))
    return false;
  return true;
}

bool Pipeline::Lower() {
  mlir::PassManager pm(&ctx_);
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return false;
  pm.addPass(coir::createClassifyCopiesPass());
  pm.addPass(coir::createLowerDMADescPass());
  pm.addPass(coir::createHoistDMAConfigPass());
  pm.addPass(coir::createLowerMMAPass());
  pm.addPass(coir::createLowerCopyPass());
  return mlir::succeeded(pm.run(module_));
}

int Pipeline::EmitSource(llvm::StringRef target, bool script,
                         llvm::StringRef output_path,
                         llvm::StringRef arch) {
  auto codegen = CodeGenRegistry::Create(target.str());
  if (!codegen) {
    llvm::errs() << "error: no code generator registered for target '"
                 << target << "'\n";
    return 1;
  }

  if (!codegen->Lower(module_)) {
    llvm::errs() << "error: target '" << target << "' lowering failed\n";
    return 1;
  }

  llvm::raw_ostream *os = &llvm::outs();
  std::unique_ptr<llvm::raw_fd_ostream> file_os;

  if (!output_path.empty()) {
    std::error_code ec;
    file_os = std::make_unique<llvm::raw_fd_ostream>(output_path, ec);
    if (ec) {
      llvm::errs() << "error: cannot open '" << output_path << "': "
                   << ec.message() << "\n";
      return 1;
    }
    os = file_os.get();
  }

  if (script)
    return codegen->EmitScript(module_, arch, *os);
  return codegen->EmitSource(module_, arch, *os);
}

int Pipeline::CompileBinary(llvm::StringRef target, llvm::StringRef output_path,
                            llvm::StringRef arch) {
  auto codegen = CodeGenRegistry::Create(target.str());
  if (!codegen) {
    llvm::errs() << "error: no code generator for target '" << target
                 << "' (binary mode requires a CodeGen backend)\n";
    return 1;
  }

  if (!codegen->Lower(module_)) {
    llvm::errs() << "error: target '" << target << "' lowering failed\n";
    return 1;
  }

  return codegen->Compile(module_, arch, output_path);
}

} // namespace CoIR
