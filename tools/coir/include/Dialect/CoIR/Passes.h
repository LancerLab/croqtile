//===- Passes.h - CoIR dialect passes ---------------------------*- C++ -*-===//
//
// Pass declarations for CoIR dialect transformations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_COIR_PASSES_H
#define DIALECT_COIR_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// MLIR dialect passes live in lowercase `coir` namespace (MLIR convention).
namespace coir {

std::unique_ptr<mlir::Pass> createClassifyCopiesPass();
std::unique_ptr<mlir::Pass> createLowerDMADescPass();
std::unique_ptr<mlir::Pass> createHoistDMAConfigPass();
std::unique_ptr<mlir::Pass> createLowerMMAPass();
std::unique_ptr<mlir::Pass> createLowerCopyPass();
std::unique_ptr<mlir::Pass> createEmitCUDAPass();

void emitCUDA(mlir::ModuleOp module, llvm::raw_ostream &os);

#define GEN_PASS_REGISTRATION
#include "CoIR/Passes.h.inc"

} // namespace coir

// High-level CoIR APIs use PascalCase `CoIR` namespace (Choreo convention).
namespace CoIR {

/// Script generation context -- set by the driver before calling Emit with
/// script=true. Provides target-independent infrastructure (headers, build
/// env) so emitters don't depend on Choreo headers directly.
struct ScriptContext {
  const char *types_header = nullptr;
  const char *runtime_header = nullptr;
  std::string build_env;
  std::string arch_override;

  static ScriptContext &Get() {
    static ScriptContext ctx;
    return ctx;
  }
};

/// Base class for CoIR target emitters. Subclasses implement target-specific
/// code generation with a uniform interface.
class Emitter {
public:
  virtual ~Emitter() = default;

  /// Emit self-contained script (compile+run) to the output stream.
  virtual void EmitScript(mlir::ModuleOp module, llvm::raw_ostream &os) = 0;

  /// Emit raw source code to the output stream.
  virtual void EmitSource(mlir::ModuleOp module, llvm::raw_ostream &os) = 0;
};

/// Registry for target-specific emitters. Emitters self-register via static
/// initialization.
struct EmitterRegistry {
  using FactoryFn = std::function<std::unique_ptr<Emitter>()>;

  static void Register(const std::string &name, FactoryFn factory) {
    GetFactories()[name] = std::move(factory);
  }

  static std::unique_ptr<Emitter> Create(const std::string &target) {
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

/// CoIR compilation pipeline -- encapsulates opt -> lower -> emit flow.
class Pipeline {
public:
  Pipeline(mlir::ModuleOp module, mlir::MLIRContext &ctx)
      : module_(module), ctx_(ctx) {}

  /// Print the CoIR MLIR module (like clang -emit-llvm).
  void EmitCoIR(llvm::raw_ostream &os);

  /// Run CoIR optimization passes. Returns true on success.
  bool Optimize();

  /// Run CoIR lowering passes. Returns true on success.
  bool Lower();

  /// Emit source code for the given target.
  /// @param target Target name (e.g. "cute")
  /// @param script If true, emit self-contained build+run script
  /// @param output_path Explicit output path (empty = stdout)
  /// @returns 0 on success, non-zero on error.
  int EmitSource(llvm::StringRef target, bool script,
                 llvm::StringRef output_path);

  mlir::ModuleOp GetModule() { return module_; }

private:
  mlir::ModuleOp module_;
  mlir::MLIRContext &ctx_;
};

} // namespace CoIR

#endif // DIALECT_COIR_PASSES_H
