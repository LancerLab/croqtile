//===- CodeGen.h - CoIR target code generation interface ---------*- C++ -*-===//
//
// Base class and registry for CoIR target-specific code generation.
//
// Each target registers a CodeGen subclass that
// owns its entire pipeline from CoIR lowering through code emission:
//
//   Lower()       -- target-specific lowering passes on the CoIR module
//   EmitSource()  -- emit target source code to a stream
//   EmitScript()  -- emit self-contained build+run script
//   Compile()     -- full in-memory compilation to binary
//
//===----------------------------------------------------------------------===//

#ifndef COIR_TARGET_CODEGEN_H
#define COIR_TARGET_CODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace CoIR {

/// Base class for CoIR target code generators. Each target implements its
/// own lowering and emission pipeline.
class CodeGen {
public:
  virtual ~CodeGen() = default;

  /// Run target-specific lowering passes on the module.
  /// Called after shared optimization passes. The module still contains
  /// high-level CoIR ops; this method applies whatever target-specific
  /// transformations are needed before emission.
  /// Returns true on success.
  virtual bool Lower(mlir::ModuleOp /*module*/) { return true; }

  /// Emit target source code to the output stream.
  virtual int EmitSource(mlir::ModuleOp module, llvm::StringRef arch,
                         llvm::raw_ostream &os) = 0;

  /// Emit a self-contained build+run script. Default calls EmitSource().
  virtual int EmitScript(mlir::ModuleOp module, llvm::StringRef arch,
                         llvm::raw_ostream &os) {
    return EmitSource(module, arch, os);
  }

  /// Full in-memory compilation to a binary at |outputPath|.
  virtual int Compile(mlir::ModuleOp /*module*/, llvm::StringRef /*arch*/,
                      llvm::StringRef /*outputPath*/) {
    llvm::errs() << "Compile() not implemented for this target\n";
    return 1;
  }
};

/// Registry for target-specific code generators.
struct CodeGenRegistry {
  using FactoryFn = std::function<std::unique_ptr<CodeGen>()>;

  static void Register(const std::string &name, FactoryFn factory) {
    GetFactories()[name] = std::move(factory);
  }

  static std::unique_ptr<CodeGen> Create(const std::string &target) {
    auto &m = GetFactories();
    auto it = m.find(target);
    if (it == m.end()) return nullptr;
    return it->second();
  }

private:
  static std::unordered_map<std::string, FactoryFn> &GetFactories() {
    static std::unordered_map<std::string, FactoryFn> registry;
    return registry;
  }
};

} // namespace CoIR

#endif // COIR_TARGET_CODEGEN_H
