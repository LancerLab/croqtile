#ifndef COIR_CODEGEN_GCU_EMIT_TOPSCC_H
#define COIR_CODEGEN_GCU_EMIT_TOPSCC_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace coir {
void emitTopscc(mlir::ModuleOp module, llvm::raw_ostream& os);
} // namespace coir

#endif
