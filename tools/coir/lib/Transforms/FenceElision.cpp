//===- FenceElision.cpp - Drop redundant directional fences ---------------===//
//
// Drops a directional coir.fence that is immediately preceded (ignoring
// memory-effect-free descriptor/config ops) by an equal fence. A fence orders
// the issuing agent's prior accesses, so a second fence of the same kind with
// no intervening memory effect publishes nothing new and is redundant.
//
// The pass runs unconditionally in the lowering pipeline (even at -O0),
// because fences are correctness annotations that lower to hardware barrier
// instructions; they are not optional optimizations.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/StringRef.h"

namespace coir {
#define GEN_PASS_DECL_FENCEELISION
#define GEN_PASS_DEF_FENCEELISION
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

/// Equality of the four-fold axes. Two fences are redundant iff they agree on
/// all of space, entity, order, and scope.
static bool sameFence(FenceOp a, FenceOp b) {
  return a.getSpace() == b.getSpace() && a.getEntity() == b.getEntity() &&
         a.getOrder() == b.getOrder() && a.getScope() == b.getScope();
}

/// Drop `op` if the nearest preceding operation (skipping memory-effect-free
/// descriptor/config ops) is an equal fence.
struct RedundantFencePattern : public OpRewritePattern<FenceOp> {
  using OpRewritePattern<FenceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FenceOp op,
                                PatternRewriter& rewriter) const override {
    // Walk backward within the block. Effect-free ops (const.desc, check,
    // prefetch.desc, runtime.desc) do not order memory, so they are skipped;
    // any other operation is a memory boundary that ends the search.
    for (Operation* prev = op->getPrevNode(); prev != nullptr;
         prev = prev->getPrevNode()) {
      if (mlir::isMemoryEffectFree(prev)) continue;
      if (auto prevFence = dyn_cast<FenceOp>(prev)) {
        if (sameFence(prevFence, op)) {
          rewriter.eraseOp(op);
          return success();
        }
      }
      break;
    }
    return failure();
  }
};

struct FenceElisionPass
    : public ::coir::impl::FenceElisionBase<FenceElisionPass> {
  using FenceElisionBase::FenceElisionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RedundantFencePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createFenceElisionPass() {
  return std::make_unique<FenceElisionPass>();
}
} // namespace coir
