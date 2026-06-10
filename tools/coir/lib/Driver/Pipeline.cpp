//===- Pipeline.cpp - CoIR Pipeline class implementation ------------------===//
//
// Implements CoIR::Pipeline (EmitCoIR, Optimize, Lower, EmitSource).
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

namespace CoIR {

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
