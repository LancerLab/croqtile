//===- LowerDMADesc.cpp - Decompose copy ops into DMA descriptor pipeline -===//
//
// Rewrites coir.data.copy inside coir.foreach loops into the DMA descriptor
// pipeline: const.desc + prefetch.desc + runtime.desc + invoke.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace coir {
#define GEN_PASS_DECL_LOWERDMADESC
#define GEN_PASS_DEF_LOWERDMADESC
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

struct DecomposeDataCopyInLoop : public OpRewritePattern<DataCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DataCopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    auto foreachOp = op->getParentOfType<ForeachOp>();
    if (!foreachOp)
      return failure();

    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType = llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType)
      return failure();

    auto &foreachBody = foreachOp.getBody();
    Value iv = foreachBody.getArgument(0);

    Value srcBase = op.getSource();
    llvm::SmallVector<Value> offsets;
    bool srcDependsOnIV = false;

    if (auto tileOp = srcBase.getDefiningOp<TensorTileOp>()) {
      srcBase = tileOp.getSource();
      for (auto idx : tileOp.getIndices()) {
        offsets.push_back(idx);
        if (idx == iv)
          srcDependsOnIV = true;
      }
    } else {
      auto *defOp = srcBase.getDefiningOp();
      if (defOp) {
        for (auto operand : defOp->getOperands()) {
          if (operand == iv) {
            srcDependsOnIV = true;
            break;
          }
        }
      }
    }

    if (!srcDependsOnIV)
      return failure();

    auto descType = coir::DMADescType::get(rewriter.getContext());
    auto descRtType = coir::FinalDMADescType::get(rewriter.getContext());
    auto tokenType = coir::AsyncTokenType::get(rewriter.getContext());
    Location loc = op.getLoc();

    auto kindAttr = coir::DMAKindAttr::get(
        rewriter.getContext(),
        offsets.empty() ? coir::DMAKind::Copy : coir::DMAKind::Slice);

    rewriter.setInsertionPoint(foreachOp);
    auto constDesc = rewriter.create<DMAConstDescOp>(
        loc, descType, srcBase, op.getDest(), kindAttr);
    auto prefetch = rewriter.create<DMADescPrefetchOp>(
        loc, descRtType, constDesc.getOut());

    rewriter.setInsertionPoint(op);

    Value rtDesc;
    if (!offsets.empty()) {
      auto runtimeDesc = rewriter.create<DMADescRuntimeOp>(
          loc, descRtType, prefetch.getOut(), offsets);
      rtDesc = runtimeDesc.getOut();
    } else {
      rtDesc = prefetch.getOut();
    }

    auto invoke = rewriter.create<DMAInvokeOp>(loc, tokenType, rtDesc);
    auto wait = rewriter.create<WaitOp>(loc, invoke.getDone());
    (void)wait;

    if (auto tileOp = op.getSource().getDefiningOp<TensorTileOp>()) {
      rewriter.eraseOp(op);
      if (tileOp->use_empty())
        rewriter.eraseOp(tileOp);
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

struct LowerDMADescPass
    : public ::coir::impl::LowerDMADescBase<LowerDMADescPass> {
  using LowerDMADescBase::LowerDMADescBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeDataCopyInLoop>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createLowerDMADescPass() {
  return std::make_unique<LowerDMADescPass>();
}
} // namespace coir
