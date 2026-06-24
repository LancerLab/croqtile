//===- CompilePipeline.h - Full in-memory NVPTX compile --------*- C++ -*-===//
//
// Orchestrates the full in-memory compilation pipeline:
//   1. CoIR -> PTX (device lowering)
//   2. Generate host stubs from kernel metadata
//   3. Combine stubs + user host code -> host LLVM IR (via clang)
//   4. Embed PTX string in host IR
//   5. Emit host object + link -> executable
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_GPU_COMPILEPIPELINE_H
#define CODEGEN_GPU_COMPILEPIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include <string>

namespace coir {
namespace gpu {

int compileToExecutable(mlir::ModuleOp module, llvm::StringRef arch,
                        llvm::StringRef outputPath,
                        const char *typesHeader, const char *runtimeHeader,
                        llvm::StringRef cudaHome);

void embedPTXInHostModule(llvm::Module &hostMod, llvm::StringRef ptxString,
                          llvm::StringRef globalName);

bool emitHostObjectFile(llvm::Module &hostMod, llvm::StringRef outputPath);

bool linkHostExecutable(llvm::StringRef objectPath,
                        llvm::StringRef executablePath,
                        llvm::StringRef cudaHome);

} // namespace gpu
} // namespace coir

#endif // CODEGEN_GPU_COMPILEPIPELINE_H
