//===- EmitHostStubs.h - Host entry stub generation from CoIR ----*- C++ -*-===//
//
// Generates C++ host entry function text for each coir.kernel op, using
// metadata attrs captured during ASTCoIRGen (param names, attrs, refs,
// element types).  Stubs use the CUDA Driver API since the kernel is
// loaded from a PTX/cubin module at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_GPU_EMITHOSTSTUBS_H
#define CODEGEN_GPU_EMITHOSTSTUBS_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace coir {
namespace gpu {

/// Walk every coir.kernel in |module| and emit a C++ host entry stub for
/// each one.  The generated text uses CUDA Driver API calls and matches
/// the Choreo spanned_view / spanned_data calling convention.
std::string emitHostStubs(mlir::ModuleOp module);

} // namespace gpu
} // namespace coir

#endif // CODEGEN_GPU_EMITHOSTSTUBS_H
