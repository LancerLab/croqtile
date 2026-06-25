//===- ClassifyCopies.cpp - Validate DMA/TMA copy ops against target ------===//
//
// Validates that coir.tma.copy ops are supported by the target (emits an
// error if not). DmaCopyOp is always valid since thread-cooperative copy
// is a form of DMA in GPU environments.
//
// Also handles legacy coir.data.copy ops from older IR files by classifying
// them into the appropriate specialized op based on memory spaces and caps.
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

/// Handle legacy DataCopyOp (for backward compatibility with older IR).
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

    bool needsHwCopy = crossLevel || hasDMA;
    if (op.getKind() != coir::DMAKind::Copy)
      needsHwCopy = true;

    if (needsHwCopy && hasTMA && crossLevel) {
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
          op.getSource(), op.getDest(),
          op.getKindAttr(),
          op.getPadLowAttr(), op.getPadHighAttr(),
          op.getPadValueAttr(), op.getTransposePermAttr());
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

    // Hard check: reject tma.copy if target lacks TMA.
    if (!hasTMA) {
      bool hasTmaCopyOps = false;
      getOperation()->walk([&](TmaCopyOp op) {
        op.emitOpError(
            "requires TMA support but target does not provide it; "
            "use dma.copy instead or target a TMA-capable architecture");
        hasTmaCopyOps = true;
      });
      if (hasTmaCopyOps)
        return signalPassFailure();
    }

    // Classify legacy DataCopyOp (backward compat with older IR).
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
