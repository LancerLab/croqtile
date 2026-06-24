//===- HostCompile.h - In-memory host C++ -> LLVM IR via clang ---*- C++ -*-===//
//
// Compiles a C++ source string to an LLVM IR Module in-memory, using
// clang as a library (CompilerInstance + EmitLLVMOnlyAction).
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_GPU_HOSTCOMPILE_H
#define CODEGEN_GPU_HOSTCOMPILE_H

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <vector>

namespace coir {
namespace gpu {

struct HostCompileOptions {
  std::string cudaHome;
  std::vector<std::pair<std::string, std::string>> virtualFiles;
  std::string standard = "-std=c++17";
};

/// Compile |source| (plain C++) to an LLVM IR Module using clang in-process.
/// |virtualFiles| are (filename, contents) pairs injected into an in-memory
/// VFS so headers like choreo_types.h / choreo.h are available.
/// Returns nullptr on failure (diagnostics printed to stderr).
std::unique_ptr<llvm::Module>
compileHostToLLVM(llvm::StringRef source, llvm::LLVMContext &ctx,
                  const HostCompileOptions &opts = {});

} // namespace gpu
} // namespace coir

#endif // CODEGEN_GPU_HOSTCOMPILE_H
