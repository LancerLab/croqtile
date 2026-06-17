//===- ClassifyCopies.cpp - Lower data.copy to specialized copy ops --------===//
//
// Classifies coir.data.copy operations into dma.copy, tma.copy, or
// thread.copy based on source/destination memory spaces and target arch.
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
static bool isRegisterOrLocal(int32_t ms) { return ms >= 2; }

struct ClassifyDataCopy : public OpRewritePattern<DataCopyOp> {
  using OpRewritePattern::OpRewritePattern;

  bool hasTMA;
  ClassifyDataCopy(MLIRContext *ctx, bool hasTMA)
      : OpRewritePattern(ctx), hasTMA(hasTMA) {}

  LogicalResult matchAndRewrite(DataCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType = llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType)
      return failure();

    int32_t srcMS = getMemSpace(srcType);
    int32_t dstMS = getMemSpace(dstType);
    Location loc = op.getLoc();

    bool needsAsync =
        (isGlobal(srcMS) && isShared(dstMS)) ||
        (isShared(srcMS) && isGlobal(dstMS));

    if (needsAsync && hasTMA) {
      auto newOp = rewriter.create<TmaCopyOp>(loc,
          coir::AsyncTokenType::get(rewriter.getContext()),
          op.getSource(), op.getDest());
      if (op.getToken())
        rewriter.replaceOp(op, newOp.getToken());
      else
        rewriter.eraseOp(op);
      return success();
    }

    if (needsAsync) {
      auto newOp = rewriter.create<DmaCopyOp>(loc,
          coir::AsyncTokenType::get(rewriter.getContext()),
          op.getSource(), op.getDest());
      if (op.getToken())
        rewriter.replaceOp(op, newOp.getToken());
      else
        rewriter.eraseOp(op);
      return success();
    }

    rewriter.create<ThreadCopyOp>(loc, op.getSource(), op.getDest());
    rewriter.eraseOp(op);
    return success();
  }
};

// Infer TMA capability from --target-arch string.
// This mirrors Target::HasTMA() for standalone coir-opt usage.
static bool inferHasTMAFromArch(llvm::StringRef arch) {
  return arch.contains("sm_9") || arch.contains("sm_100");
}

struct ClassifyCopiesPass
    : public ::coir::impl::ClassifyCopiesBase<ClassifyCopiesPass> {
  using ClassifyCopiesBase::ClassifyCopiesBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    // Module attribute (set by the driver via Target::HasTMA)
    // is authoritative. Fall back to --target-arch for coir-opt.
    bool hasTMA = false;
    if (auto module = dyn_cast<ModuleOp>(getOperation()))
      hasTMA = CoIR::HasTMA(module);
    else if (auto pm = getOperation()->getParentOfType<ModuleOp>())
      hasTMA = CoIR::HasTMA(pm);
    if (!hasTMA)
      hasTMA = inferHasTMAFromArch(targetArch);

    RewritePatternSet patterns(ctx);
    patterns.add<ClassifyDataCopy>(ctx, hasTMA);

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
