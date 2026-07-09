//===- CoIREmitterBase.h - Unified base for CoIR emitters --------*- C++ -*-===//
//
// Combined base class for CoIR-to-source-text emitters. Inherits from
// CoIR::CodeGen to provide the full pipeline interface (EmitSource,
// EmitScript) alongside the op-level visitor pattern and shared helpers.
//
// Target emitters inherit CoIREmitterBase and override virtual hooks
// for type emission, target-specific ops, kernel/host entry generation,
// and script compilation commands.
//
//===----------------------------------------------------------------------===//

#ifndef COIR_CODEGEN_COIREMITTERBASE_H
#define COIR_CODEGEN_COIREMITTERBASE_H

#include "Target/CodeGen.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace coir {

/// Unified base for CoIR source-text emitters. Combines:
///   - CoIR::CodeGen pipeline interface (EmitSource/EmitScript)
///   - MLIR op visitor pattern (emitOp dispatcher + virtual hooks)
///   - Shared state (output stream, indentation, value naming)
///   - Common op implementations (arith, SCF, tensor load/store)
///   - Script generation helpers (inherited from CodeGen)
class CoIREmitterBase : public CoIR::CodeGen {
public:
  CoIREmitterBase() = default;
  ~CoIREmitterBase() override = default;

  // -- CodeGen pipeline interface --
  // Derived classes implement emitModule() which is called by EmitSource.
  // EmitSource and EmitScript have default implementations that call
  // emitModule and emitUserCppCode respectively; targets override EmitScript
  // to add their compiler-specific script.

  int EmitSource(mlir::ModuleOp module, llvm::StringRef arch,
                 llvm::raw_ostream &os) override;

  // -- Module emission (pure virtual, each target implements) --
  virtual void emitModule(mlir::ModuleOp module, llvm::raw_ostream &os) = 0;

protected:
  // -- Emission state (set per EmitSource call) --
  llvm::raw_ostream *os_ = nullptr;
  unsigned indent = 0;
  llvm::DenseMap<mlir::Value, std::string> valueNames;
  llvm::DenseMap<unsigned, std::string> returnParamNames;
  llvm::DenseSet<mlir::Value> returnValues;
  unsigned nextId = 0;
  std::string lastSpmName;
  bool dynSpmEmitted = false;

  llvm::raw_ostream &os() { return *os_; }

  /// Reset emission state for a new module.
  void resetState();

  // -- Indentation --
  std::string getIndent();
  void incIndent();
  void decIndent();

  // -- Value naming --
  std::string getName(mlir::Value v);

  // -- Type emission (must be overridden per target) --
  virtual std::string emitType(mlir::Type ty) = 0;

  // -- Element type (shared default, overridable) --
  virtual std::string emitElementType(mlir::Type ty);

  // -- Tensor helpers --
  int64_t getTensorBytes(coir::TensorType tty);
  void emitLinearIndex(mlir::ValueRange indices, coir::TensorType tty);

  // -- Return value prescan --
  void prescanReturnValues(KernelOp kernel);

  // -- Visitor-pattern op dispatch (template method) --
  virtual void emitOp(mlir::Operation *op);

  // -- Common ops (concrete in base) --
  void emitConstant(mlir::arith::ConstantOp op);
  bool emitArithBinOp(mlir::Operation *op);
  bool emitCmpOp(mlir::Operation *op);
  void emitIfOp(mlir::scf::IfOp op);
  void emitBreak(coir::CoIRBreakOp op);
  void emitContinue(coir::CoIRContinueOp op);
  void emitSelect(mlir::arith::SelectOp op);
  void emitTensorLoadElem(TensorLoadElemOp op);
  void emitTensorStoreElem(TensorStoreElemOp op);

  // -- Overridable ops with default impls --
  virtual void emitWhileOp(mlir::scf::WhileOp op);
  virtual void emitCoIRWhileOp(coir::CoIRWhileOp op);
  virtual void emitForeach(ForeachOp op);
  virtual void emitYield(YieldOp op);
  virtual void emitTensorAlloc(TensorAllocOp op);
  virtual void emitTensorTile(TensorTileOp op);

  // -- Pure virtual target-specific ops --
  virtual void emitParallel(ParallelOp op) = 0;
  virtual void emitHostEntry(KernelOp kernel) = 0;
  virtual void emitBarrier(BarrierOp op) = 0;
  virtual void emitWait(WaitOp op) = 0;
  virtual void emitDmaCopy(DmaCopyOp op) = 0;
  virtual void emitDMAConstDesc(DMAConstDescOp op) = 0;
  virtual void emitDMAPrefetch(DMADescPrefetchOp op) = 0;
  virtual void emitDMARuntimeDesc(DMADescRuntimeOp op) = 0;
  virtual void emitDMAInvoke(DMAInvokeOp op) = 0;
  virtual void emitMMAFill(MMAFillOp op) = 0;
  virtual void emitMMALoad(MMALoadOp op) = 0;
  virtual void emitMMAExec(MMAExecOp op) = 0;
  virtual void emitMMAStore(MMAStoreOp op) = 0;

  // -- Target-specific op hooks (no-op by default) --
  virtual void emitFutureRotate(FutureRotateOp op) { (void)op; }
  virtual void emitAsyncUndef(AsyncUndefOp op) { (void)op; }
  virtual void emitKernelReturn(KernelReturnOp op) { (void)op; }
  virtual void emitTensorReduceElem(TensorReduceElemOp op) { (void)op; }
  virtual void emitAtomic(AtomicOp op) { (void)op; }

  // -- Fallback for unhandled ops (override to add target ops) --
  virtual void emitOpFallback(mlir::Operation *op);

  // -- Alloc qualifier hook --
  virtual std::string getAllocQualifier(coir::TensorType tty);
  virtual bool needsTMAAlignment(coir::TensorType tty);

  // -- Target capability queries (override per target/arch) --
  virtual bool hasGroupLevel() const { return false; }
  virtual bool supportsFP8() const { return false; }
  virtual bool supportsFP6() const { return false; }
  virtual bool supportsFP4() const { return false; }
  virtual bool supportsLaunchBounds() const { return false; }
  virtual bool supportsMaxNreg() const { return false; }
};

} // namespace coir

#endif // COIR_CODEGEN_COIREMITTERBASE_H
