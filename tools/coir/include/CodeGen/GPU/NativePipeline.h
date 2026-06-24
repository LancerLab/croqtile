//===- NativePipeline.h - GPU native compilation API ------------*- C++ -*-===//
//
// GPU-target-specific compilation pipeline APIs. These are intentionally
// separated from the dialect-level Passes.h to keep target concerns isolated.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_GPU_NATIVEPIPELINE_H
#define CODEGEN_GPU_NATIVEPIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace coir {

/// Create the ConvertToGPU pass (CoIR -> GPU/memref/arith).
std::unique_ptr<mlir::Pass> createConvertToGPUPass();

namespace gpu {

/// Run the full CoIR -> GPU -> NVVM -> LLVM dialect pipeline.
/// Returns true on success.
bool lowerToNVVM(mlir::ModuleOp module);

/// Translate an MLIR module (LLVM dialect) to PTX assembly string.
std::string emitPTX(mlir::ModuleOp gpuModule, llvm::StringRef arch);

/// Full pipeline: CoIR -> GPU -> NVVM -> LLVM -> PTX.
/// Clones the module internally.
std::string emitPTXFromCoIR(mlir::ModuleOp module, llvm::StringRef arch);

} // namespace gpu
} // namespace coir

#endif // CODEGEN_GPU_NATIVEPIPELINE_H
