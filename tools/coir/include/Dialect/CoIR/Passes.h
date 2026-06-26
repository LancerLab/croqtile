//===- Passes.h - CoIR dialect passes ---------------------------*- C++ -*-===//
//
// Pass declarations for CoIR dialect transformations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_COIR_PASSES_H
#define DIALECT_COIR_PASSES_H

#include "Target/CodeGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

// MLIR dialect passes live in lowercase `coir` namespace (MLIR convention).
namespace coir {

std::unique_ptr<mlir::Pass> createClassifyCopiesPass();
std::unique_ptr<mlir::Pass> createLowerDMADescPass();
std::unique_ptr<mlir::Pass> createHoistDMAConfigPass();
std::unique_ptr<mlir::Pass> createLowerMMAPass();
std::unique_ptr<mlir::Pass> createLowerCopyPass();
std::unique_ptr<mlir::Pass> createHoistAssertionsPass();
std::unique_ptr<mlir::Pass> createEstimateAssertCostPass();
std::unique_ptr<mlir::Pass> createCollectAssertStatsPass();
std::unique_ptr<mlir::Pass> createEmitCUDAPass();
std::unique_ptr<mlir::Pass> createEmitHIPPass();

void emitCUDA(mlir::ModuleOp module, llvm::raw_ostream &os);
void emitHIP(mlir::ModuleOp module, llvm::raw_ostream &os);

#define GEN_PASS_REGISTRATION
#include "CoIR/Passes.h.inc"

} // namespace coir

// High-level CoIR APIs use PascalCase `CoIR` namespace (Choreo convention).
namespace CoIR {

/// Stamp target metadata onto a CoIR module as MLIR attributes.
/// Passes read these attributes instead of carrying per-pass options.
///
/// Module attributes set:
///   "coir.target"     -- target backend name (e.g. "cute")
///   "coir.arch"       -- architecture string (e.g. "sm_90a")
///   "coir.mma_target" -- MMA lowering strategy: "wgmma", "mma_sync",
///                        "ukernel", or "" (no MMA)
///   "coir.has_tma"    -- whether the target supports TMA
///   "coir.has_dma"    -- whether the target needs DMA for global<->local
void StampTargetOnModule(mlir::ModuleOp module, llvm::StringRef target,
                         llvm::StringRef arch, llvm::StringRef mma_target,
                         bool has_tma, bool has_dma);

/// Read MMA target strategy from module attributes.
/// Returns empty string if not set.
llvm::StringRef GetMMATarget(mlir::ModuleOp module);

/// Read whether the target has TMA from module attributes.
bool HasTMA(mlir::ModuleOp module);

/// Read whether the target needs hardware DMA for global<->local transfers.
bool HasDMA(mlir::ModuleOp module);

/// Read the architecture string from module attributes.
llvm::StringRef GetArch(mlir::ModuleOp module);

/// Script generation context -- set by the driver before calling Emit with
/// script=true. Provides target-independent infrastructure (headers, build
/// env) so emitters don't depend on Choreo headers directly.
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

/// CoIR compilation pipeline -- encapsulates opt -> lower -> codegen flow.
///
/// The pipeline has three phases:
///   1. Optimize() -- target-independent optimizations (HoistAssertions)
///   2. Lower()    -- shared CoIR lowering (DMA/MMA classification, hoisting,
///                    lowering). Runs for all targets.
///   3. EmitSource() / CompileBinary() -- target-specific codegen via
///                    CodeGenRegistry. Each target's CodeGen can run
///                    additional target-specific lowering before emission.
class Pipeline {
public:
  Pipeline(mlir::ModuleOp module, mlir::MLIRContext &ctx,
           int costThreshold = 4, bool collectStats = false)
      : module_(module), ctx_(ctx), cost_threshold_(costThreshold),
        collect_stats_(collectStats) {}

  /// Print the CoIR MLIR module (like clang -emit-llvm).
  void EmitCoIR(llvm::raw_ostream &os);

  /// Run safety instrumentation passes (always, not skippable by --no-opt):
  ///   1. HoistAssertions   -- LICM for assertions (SSA dominance)
  ///   2. EstimateAssertCost -- stamp cost/enabled attrs
  ///   3. CollectAssertStats -- collect stats (if collect_stats_)
  bool InstrumentSafety();

  /// Run CoIR optimization passes (skipped by --no-opt / -O0).
  /// Currently empty; future home for real optimizations.
  bool Optimize();

  /// Run shared CoIR lowering passes (DMA/MMA classification and hoisting).
  /// These are target-independent and benefit all backends.
  bool Lower();

  /// Lower + emit source code for the given target.
  /// Looks up a CodeGen from CodeGenRegistry, runs its target-specific
  /// Lower(), then emits.
  int EmitSource(llvm::StringRef target, bool script,
                 llvm::StringRef output_path,
                 llvm::StringRef arch = "");

  /// Lower + compile to a native binary via the target's CodeGen.
  int CompileBinary(llvm::StringRef target, llvm::StringRef output_path,
                    llvm::StringRef arch = "");

  mlir::ModuleOp GetModule() { return module_; }
  mlir::MLIRContext &GetContext() { return ctx_; }

private:
  mlir::ModuleOp module_;
  mlir::MLIRContext &ctx_;
  int cost_threshold_;
  bool collect_stats_;
};

} // namespace CoIR

#endif // DIALECT_COIR_PASSES_H
