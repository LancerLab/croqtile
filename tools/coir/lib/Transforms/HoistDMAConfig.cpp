//===- HoistDMAConfig.cpp - Hoist loop-invariant DMA descriptor setup -----===//
//
// Uses MLIR's LoopLikeOpInterface LICM to hoist DMA descriptor pipeline
// ops (const.desc, prefetch.desc, check) above loop nests when their
// operands are loop-invariant.  Works with arbitrarily nested loops.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Pass/Pass.h"

namespace coir {
#define GEN_PASS_DECL_HOISTDMACONFIG
#define GEN_PASS_DEF_HOISTDMACONFIG
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

struct HoistDMAConfigPass
    : public ::coir::impl::HoistDMAConfigBase<HoistDMAConfigPass> {
  using HoistDMAConfigBase::HoistDMAConfigBase;

  void runOnOperation() override {
    getOperation()->walk([&](LoopLikeOpInterface loopLike) {
      moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          [&](Value value, Region *) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          [](Operation *op, Region *) {
            return isa<DMAConstDescOp, DMADescPrefetchOp, DMACheckOp>(op);
          },
          [](Operation *op, Region *region) {
            op->moveBefore(region->getParentOp());
          });
    });
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createHoistDMAConfigPass() {
  return std::make_unique<HoistDMAConfigPass>();
}
} // namespace coir
