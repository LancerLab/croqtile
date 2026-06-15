//===- HoistDMAConfig.cpp - Hoist loop-invariant DMA descriptor setup -----===//
//
// LICM for DMA descriptor pipeline ops: moves dma.const.desc and
// dma.prefetch.desc above coir.foreach loops when their operands are
// defined outside the loop.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace coir {
#define GEN_PASS_DECL_HOISTDMACONFIG
#define GEN_PASS_DEF_HOISTDMACONFIG
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

static bool isDefinedOutsideLoop(Value v, ForeachOp loop) {
  if (!v.getDefiningOp())
    return true;
  return !loop->isAncestor(v.getDefiningOp());
}

static bool allOperandsOutsideLoop(Operation *op, ForeachOp loop) {
  for (auto operand : op->getOperands()) {
    if (!isDefinedOutsideLoop(operand, loop))
      return false;
  }
  return true;
}

struct HoistDMAConfigPass
    : public ::coir::impl::HoistDMAConfigBase<HoistDMAConfigPass> {
  using HoistDMAConfigBase::HoistDMAConfigBase;

  void runOnOperation() override {
    getOperation()->walk([&](ForeachOp foreachOp) {
      bool changed = true;
      while (changed) {
        changed = false;
        llvm::SmallVector<Operation *, 4> toHoist;

        foreachOp.getBody().walk([&](Operation *op) {
          if (!isa<DMAConstDescOp, DMADescPrefetchOp, DMACheckOp>(op))
            return;
          if (allOperandsOutsideLoop(op, foreachOp))
            toHoist.push_back(op);
        });

        for (auto *op : toHoist) {
          op->moveBefore(foreachOp);
          changed = true;
        }
      }
    });
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createHoistDMAConfigPass() {
  return std::make_unique<HoistDMAConfigPass>();
}
} // namespace coir
