//===- CodeGen.h - CoIR target code generation interface ---------*- C++ -*-===//
//
// Base class and registry for CoIR target-specific code generation.
//
// Class hierarchy for source-text emitters:
//
//   CoIR::CodeGen                  -- abstract pipeline interface
//     CoIR::CodeGen (direct)       -- used by NVPTXCodeGen (MLIR->PTX path)
//     coir::CoIREmitterBase        -- unified base for IR-walking emitters;
//                                    combines CodeGen pipeline + op visitor +
//                                    script/host-code helpers
//       CUDAEmitter                -- CUDA/CUTE target
//       HIPEmitter                 -- HIP/AMD target
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

/// Script generation context -- set by the driver before codegen.
/// Provides target-independent infrastructure (headers, build env)
/// so emitters don't depend on Choreo headers directly.
struct ScriptContext {
  const char *types_header = nullptr;
  const char *runtime_header = nullptr;
  const char *types_cute_header = nullptr;
  const char *cute_header = nullptr;
  std::string build_env;
  std::string target_setup;
  std::string arch_override;
  std::string cuda_home;

  static ScriptContext &Get() {
    static ScriptContext ctx;
    return ctx;
  }
};

/// Base class for CoIR target code generators.
class CodeGen {
public:
  virtual ~CodeGen() = default;

  virtual bool Lower(mlir::ModuleOp /*module*/) { return true; }

  virtual int EmitSource(mlir::ModuleOp module, llvm::StringRef arch,
                         llvm::raw_ostream &os) = 0;

  virtual int EmitScript(mlir::ModuleOp module, llvm::StringRef arch,
                         llvm::raw_ostream &os) {
    return EmitSource(module, arch, os);
  }

  virtual int Compile(mlir::ModuleOp /*module*/, llvm::StringRef /*arch*/,
                      llvm::StringRef /*outputPath*/) {
    llvm::errs() << "Compile() not implemented for this target\n";
    return 1;
  }

  // -- Shared script helpers (used by all source-text targets) --

  /// Emit the user's host-code block attached to the module.
  static void emitHostCode(mlir::ModuleOp module, llvm::raw_ostream &os) {
    auto attr = module->getAttrOfType<mlir::StringAttr>("coir.host_code");
    if (attr) os << "\n" << attr.getValue() << "\n";
  }

  /// Emit common script prologue: shebang, tmpdir, embedded Choreo headers.
  static void emitScriptPrologue(llvm::raw_ostream &os,
                                 llvm::StringRef comment,
                                 llvm::StringRef tmpSuffix = "");

  /// Emit the --execute block common to all script targets.
  static void emitScriptExecuteBlock(llvm::raw_ostream &os) {
    os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
    os << "  shift\n";
    os << "  \"$BINFILE\" \"$@\"\n";
    os << "fi\n";
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
