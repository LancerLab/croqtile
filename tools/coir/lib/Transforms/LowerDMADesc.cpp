//===- LowerDMADesc.cpp - Decompose DMA/TMA copy ops into descriptor pipeline
//
// Decompose coir.dma.copy and coir.tma.copy into the unified DMA descriptor
// pipeline:   const.desc -> prefetch.desc -> [runtime.desc] -> invoke
//
// This pass only handles decomposition (structural rewriting). Hoisting of
// loop-invariant ops (const.desc, prefetch) is done by HoistDMAConfig which
// runs immediately after.
//
// Precondition: input IR should use coir.dma.copy / coir.tma.copy directly.
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

#include <type_traits>

namespace coir {
#define GEN_PASS_DECL_LOWERDMADESC
#define GEN_PASS_DEF_LOWERDMADESC
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

static bool isGlobalSharedCopy(coir::TensorType srcType,
                               coir::TensorType dstType) {
  int32_t srcMS = srcType.getMemorySpace();
  int32_t dstMS = dstType.getMemorySpace();
  return (srcMS <= 0 && dstMS == 1) || (srcMS == 1 && dstMS <= 0);
}

/// Decompose a DmaCopyOp or TmaCopyOp into the descriptor pipeline in-place.
/// Extracts tile offsets from TensorTileOp if present.
/// Hoisting is left to HoistDMAConfig.
template <typename CopyOpTy>
struct DecomposeCopy : public OpRewritePattern<CopyOpTy> {
  using OpRewritePattern<CopyOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOpTy op,
                                PatternRewriter &rewriter) const override {
    auto srcType =
        llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType =
        llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType)
      return failure();

    if (!isGlobalSharedCopy(srcType, dstType))
      return failure();

    // Extract base tensors and tile offsets.
    Value srcBase = op.getSource();
    Value dstBase = op.getDest();
    llvm::SmallVector<Value> offsets;
    bool hasOffsets = false;

    if (auto tileOp = srcBase.template getDefiningOp<TensorTileOp>()) {
      srcBase = tileOp.getSource();
      for (auto idx : tileOp.getIndices())
        offsets.push_back(idx);
      hasOffsets = true;
    }
    if (auto tileOp = dstBase.template getDefiningOp<TensorTileOp>()) {
      dstBase = tileOp.getSource();
      if (!hasOffsets) {
        for (auto idx : tileOp.getIndices())
          offsets.push_back(idx);
        hasOffsets = true;
      }
    }

    // For TMA copies, tile offsets become runtime.desc coordinates.
    // For DMA copies with tile offsets, also decompose: the base geometry
    // goes into const.desc (hoistable) and offsets go into runtime.desc.
    constexpr bool isTMA = std::is_same_v<CopyOpTy, TmaCopyOp>;

    auto descType = coir::DMADescType::get(rewriter.getContext());
    auto descRtType = coir::FinalDMADescType::get(rewriter.getContext());
    auto tokenType = coir::AsyncTokenType::get(rewriter.getContext());
    Location loc = op.getLoc();

    auto kindAttr = coir::DMAKindAttr::get(
        rewriter.getContext(),
        hasOffsets ? coir::DMAKind::Slice : coir::DMAKind::Copy);

    auto constDesc = rewriter.create<DMAConstDescOp>(
        loc, descType, srcBase, dstBase, kindAttr,
        isTMA ? rewriter.getUnitAttr() : nullptr);
    auto prefetch = rewriter.create<DMADescPrefetchOp>(
        loc, descRtType, constDesc.getOut());

    Value invokeDesc;
    if (hasOffsets && !offsets.empty()) {
      auto runtimeDesc = rewriter.create<DMADescRuntimeOp>(
          loc, descRtType, prefetch.getOut(), offsets);
      invokeDesc = runtimeDesc.getOut();
    } else {
      invokeDesc = prefetch.getOut();
    }

    auto invoke = rewriter.create<DMAInvokeOp>(loc, tokenType, invokeDesc);

    if (op.getToken() && !op.getToken().use_empty())
      rewriter.replaceAllUsesWith(op.getToken(), invoke.getDone());
    else
      rewriter.create<WaitOp>(loc, invoke.getDone());

    auto srcTileOp =
        op.getSource().template getDefiningOp<TensorTileOp>();
    auto dstTileOp =
        op.getDest().template getDefiningOp<TensorTileOp>();
    rewriter.eraseOp(op);
    if (srcTileOp && srcTileOp->use_empty())
      rewriter.eraseOp(srcTileOp);
    if (dstTileOp && dstTileOp->use_empty())
      rewriter.eraseOp(dstTileOp);

    return success();
  }
};

struct LowerDMADescPass
    : public ::coir::impl::LowerDMADescBase<LowerDMADescPass> {
  using LowerDMADescBase::LowerDMADescBase;

  void runOnOperation() override {
    // Gate: require target DMA/TMA capability.
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      module = getOperation()->getParentOfType<ModuleOp>();
    if (module) {
      auto hasDMA = module->getAttrOfType<BoolAttr>("coir.has_dma");
      auto hasTMA = module->getAttrOfType<BoolAttr>("coir.has_tma");
      bool active = (!hasDMA && !hasTMA) || // standalone test
                    (hasDMA && hasDMA.getValue()) ||
                    (hasTMA && hasTMA.getValue());
      if (!active)
        return;
    }

    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DecomposeCopy<DmaCopyOp>>(ctx);
    patterns.add<DecomposeCopy<TmaCopyOp>>(ctx);

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
