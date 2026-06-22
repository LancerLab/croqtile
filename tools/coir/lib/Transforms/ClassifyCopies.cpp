//===- ClassifyCopies.cpp - Lower data.copy to specialized copy ops --------===//
//
// Classifies coir.data.copy into dma.copy, tma.copy, or element.copy
// based on source/destination memory spaces and target capabilities
// read from module attributes.
//
// Target capabilities (set by the driver from Target::HasTMA()/HasDMA()):
//   coir.has_tma  -- target supports TMA (e.g. SM90+)
//   coir.has_dma  -- target has DMA engine for cross-level transfers
//                    (global<->shared, global<->local, shared<->local)
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "coir-classify-copies"

namespace coir {
#define GEN_PASS_DECL_CLASSIFYCOPIES
#define GEN_PASS_DEF_CLASSIFYCOPIES
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

static int32_t getMemSpace(coir::TensorType ty) {
  return ty.getMemorySpace();
}

static bool isGlobal(int32_t ms) { return ms <= 0; }
static bool isShared(int32_t ms) { return ms == 1; }
static bool isLocal(int32_t ms) { return ms == 2; }

struct ClassifyDataCopy : public OpRewritePattern<DataCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  bool hasTMA;
  bool hasDMA;
  ClassifyDataCopy(MLIRContext *ctx, bool hasTMA, bool hasDMA)
      : OpRewritePattern(ctx), hasTMA(hasTMA), hasDMA(hasDMA) {}

  LogicalResult matchAndRewrite(DataCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType = llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType)
      return failure();

    int32_t srcMS = getMemSpace(srcType);
    int32_t dstMS = getMemSpace(dstType);
    Location loc = op.getLoc();

    bool crossLevel =
        (isGlobal(srcMS) && isShared(dstMS)) ||
        (isShared(srcMS) && isGlobal(dstMS));
    bool globalLocal =
        (isGlobal(srcMS) && isLocal(dstMS)) ||
        (isLocal(srcMS) && isGlobal(dstMS));
    bool sharedLocal =
        (isShared(srcMS) && isLocal(dstMS)) ||
        (isLocal(srcMS) && isShared(dstMS));

    bool needsHwCopy =
        crossLevel ||
        (hasDMA && globalLocal) ||
        (hasDMA && sharedLocal);

    if (needsHwCopy && hasTMA) {
      auto newOp = rewriter.create<TmaCopyOp>(loc,
          coir::AsyncTokenType::get(rewriter.getContext()),
          op.getSource(), op.getDest());
      if (op.getToken())
        rewriter.replaceOp(op, newOp.getToken());
      else
        rewriter.eraseOp(op);
      return success();
    }

    if (needsHwCopy) {
      auto newOp = rewriter.create<DmaCopyOp>(loc,
          coir::AsyncTokenType::get(rewriter.getContext()),
          op.getSource(), op.getDest());
      if (op.getToken())
        rewriter.replaceOp(op, newOp.getToken());
      else
        rewriter.eraseOp(op);
      return success();
    }

    rewriter.create<ElementCopyOp>(loc, op.getSource(), op.getDest());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Resolve target capabilities from module attributes.
/// These attributes are set by the driver via Target::HasTMA()/HasDMA(),
/// which query the target's feature table. For standalone coir-opt tests,
/// embed coir.has_tma/coir.has_dma directly in the test IR.
static void resolveTargetCaps(Operation *op, bool &hasTMA, bool &hasDMA) {
  hasTMA = false;
  hasDMA = false;

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return;

  if (auto attr = module->getAttrOfType<BoolAttr>("coir.has_tma"))
    hasTMA = attr.getValue();
  if (auto attr = module->getAttrOfType<BoolAttr>("coir.has_dma"))
    hasDMA = attr.getValue();
}

struct ClassifyCopiesPass
    : public ::coir::impl::ClassifyCopiesBase<ClassifyCopiesPass> {
  using ClassifyCopiesBase::ClassifyCopiesBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    bool hasTMA = false;
    bool hasDMA = false;
    resolveTargetCaps(getOperation(), hasTMA, hasDMA);

    RewritePatternSet patterns(ctx);
    patterns.add<ClassifyDataCopy>(ctx, hasTMA, hasDMA);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createClassifyCopiesPass() {
  return std::make_unique<ClassifyCopiesPass>();
}
} // namespace coir
