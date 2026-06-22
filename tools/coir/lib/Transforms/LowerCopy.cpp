//===- LowerCopy.cpp - Lower specialized copy ops to transfer patterns ----===//
//
// Expands coir.dma.copy, coir.tma.copy, and coir.element.copy into
// the corresponding low-level transfer and synchronization sequences.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace coir {
#define GEN_PASS_DECL_LOWERCOPY
#define GEN_PASS_DEF_LOWERCOPY
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

// Element copy: annotate with "lowered" to signal readiness for codegen.
struct LowerElementCopy : public OpRewritePattern<ElementCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ElementCopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    if (!srcType)
      return failure();

    int64_t totalElems = 1;
    for (auto d : srcType.getShape())
      totalElems *= d;
    op->setAttr("total_elements",
                rewriter.getI64IntegerAttr(totalElems));
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// DMA copy: annotate with transfer metadata.
// In codegen, this becomes cp.async.bulk with mbarrier arrive/wait.
struct LowerDmaCopy : public OpRewritePattern<DmaCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DmaCopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    if (!srcType)
      return failure();

    int64_t totalBytes = 1;
    for (auto d : srcType.getShape())
      totalBytes *= d;
    unsigned elemBits = srcType.getElementType().getIntOrFloatBitWidth();
    totalBytes = totalBytes * elemBits / 8;

    op->setAttr("transfer_bytes", rewriter.getI64IntegerAttr(totalBytes));
    op->setAttr("mechanism", rewriter.getStringAttr("cp_async"));
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// TMA copy: annotate with TMA descriptor metadata.
// In codegen, this becomes TMA descriptor creation + copy_tma_desc invoke.
struct LowerTmaCopy : public OpRewritePattern<TmaCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TmaCopyOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    if (!srcType)
      return failure();

    int64_t totalBytes = 1;
    for (auto d : srcType.getShape())
      totalBytes *= d;
    unsigned elemBits = srcType.getElementType().getIntOrFloatBitWidth();
    totalBytes = totalBytes * elemBits / 8;

    op->setAttr("transfer_bytes", rewriter.getI64IntegerAttr(totalBytes));
    op->setAttr("mechanism", rewriter.getStringAttr("tma"));
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

struct LowerCopyPass : public ::coir::impl::LowerCopyBase<LowerCopyPass> {
  using LowerCopyBase::LowerCopyBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerElementCopy>(ctx);
    patterns.add<LowerDmaCopy>(ctx);
    patterns.add<LowerTmaCopy>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createLowerCopyPass() {
  return std::make_unique<LowerCopyPass>();
}
} // namespace coir
