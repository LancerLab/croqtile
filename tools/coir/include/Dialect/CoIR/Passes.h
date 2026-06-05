//===- Passes.h - CoIR dialect passes ---------------------------*- C++ -*-===//
//
// Pass declarations for CoIR dialect transformations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_COIR_PASSES_H
#define DIALECT_COIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace coir {

std::unique_ptr<mlir::Pass> createClassifyCopiesPass();
std::unique_ptr<mlir::Pass> createLowerMMAPass();
std::unique_ptr<mlir::Pass> createLowerCopyPass();
std::unique_ptr<mlir::Pass> createEmitCUDAPass();

#define GEN_PASS_REGISTRATION
#include "CoIR/Passes.h.inc"

} // namespace coir

#endif // DIALECT_COIR_PASSES_H
