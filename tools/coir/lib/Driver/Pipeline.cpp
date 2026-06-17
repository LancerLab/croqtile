//===- Pipeline.cpp - CoIR Pipeline class implementation ------------------===//
//
// Implements CoIR::Pipeline (EmitCoIR, Optimize, Lower, EmitSource) and
// the module-level target attribute helpers.
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
                         bool has_tma) {
  auto *ctx = module.getContext();
  if (!target.empty())
    module->setAttr("coir.target", mlir::StringAttr::get(ctx, target));
  if (!arch.empty())
    module->setAttr("coir.arch", mlir::StringAttr::get(ctx, arch));
  if (!mma_target.empty())
    module->setAttr("coir.mma_target", mlir::StringAttr::get(ctx, mma_target));
  module->setAttr("coir.has_tma", mlir::BoolAttr::get(ctx, has_tma));
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
  // TODO: CoIR optimization passes (CSE, canonicalize, etc.)
  return true;
}

bool Pipeline::Lower() {
  mlir::PassManager pm(&ctx_);
  pm.addPass(coir::createClassifyCopiesPass());
  pm.addPass(coir::createLowerDMADescPass());
  pm.addPass(coir::createHoistDMAConfigPass());
  pm.addPass(coir::createLowerMMAPass());
  pm.addPass(coir::createLowerCopyPass());
  return mlir::succeeded(pm.run(module_));
}

int Pipeline::EmitSource(llvm::StringRef target, bool script,
                         llvm::StringRef output_path) {
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

  auto emitter = EmitterRegistry::Create(target.str());
  if (!emitter) {
    llvm::errs() << "error: no emitter registered for target '" << target
                 << "'\n";
    return 1;
  }

  if (script)
    emitter->EmitScript(module_, *os);
  else
    emitter->EmitSource(module_, *os);
  return 0;
}

} // namespace CoIR
