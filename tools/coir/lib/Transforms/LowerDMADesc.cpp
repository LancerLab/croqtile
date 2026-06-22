//===- LowerDMADesc.cpp - Decompose DMA copy ops into descriptor pipeline -===//
//
// Target-driven decomposition of coir.dma.copy into the DMA descriptor
// pipeline (const.desc -> prefetch.desc -> runtime.desc -> invoke).
//
// The pass reads the module attribute "coir.has_dma" to decide whether
// to apply descriptor decomposition. Also matches coir.data.copy for
// backward compatibility with standalone tests that bypass ClassifyCopies.
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

/// Decompose a DMA copy inside a foreach loop into the descriptor pipeline.
/// Works on both DmaCopyOp (post-ClassifyCopies) and DataCopyOp (standalone).
template <typename CopyOpTy>
struct DecomposeCopyInLoop : public OpRewritePattern<CopyOpTy> {
  using OpRewritePattern<CopyOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOpTy op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    auto foreachOp = op->template getParentOfType<ForeachOp>();
    if (!foreachOp)
      return failure();

    auto srcType =
        llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType =
        llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType)
      return failure();

    auto &foreachBody = foreachOp.getBody();
    Value iv = foreachBody.getArgument(0);

    Value srcBase = op.getSource();
    llvm::SmallVector<Value> offsets;
    bool srcDependsOnIV = false;

    if (auto tileOp = srcBase.template getDefiningOp<TensorTileOp>()) {
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

    // Replace the original copy's token (if any) with the invoke's token.
    // DmaCopyOp always produces a token; DataCopyOp optionally does.
    if (op.getToken() && !op.getToken().use_empty())
      rewriter.replaceAllUsesWith(op.getToken(), invoke.getDone());
    else
      rewriter.create<WaitOp>(loc, invoke.getDone());

    auto tileOp =
        op.getSource().template getDefiningOp<TensorTileOp>();
    rewriter.eraseOp(op);
    if (tileOp && tileOp->use_empty())
      rewriter.eraseOp(tileOp);

    return success();
  }
};

/// Check whether the target uses a descriptor-based DMA engine.
/// Uses coir.has_dma module attribute (set by the driver from target features).
static bool targetUsesDescriptorDMA(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return true; // standalone test: assume descriptor DMA

  auto hasDMA =
      module->getAttrOfType<BoolAttr>("coir.has_dma");
  if (!hasDMA)
    return true; // no attr: assume descriptor DMA (for tests)

  return hasDMA.getValue();
}

struct LowerDMADescPass
    : public ::coir::impl::LowerDMADescBase<LowerDMADescPass> {
  using LowerDMADescBase::LowerDMADescBase;

  void runOnOperation() override {
    if (!targetUsesDescriptorDMA(getOperation()))
      return;

    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeCopyInLoop<DmaCopyOp>>(ctx);
    patterns.add<DecomposeCopyInLoop<DataCopyOp>>(ctx);

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
