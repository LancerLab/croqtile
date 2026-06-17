//===- LowerMMA.cpp - Lower coir.mma ops to target MMA instructions ------===//
//
// Converts high-level coir.mma.* operations into sequences of lower-level
// operations targeting the selected GPU MMA instruction set.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace coir {
#define GEN_PASS_DECL_LOWERMMA
#define GEN_PASS_DEF_LOWERMMA
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

// MMA fill: zero-initialize accumulator fragment.
// Lowered to arith.constant + coir.mma.fill_lowered (marker op)
// that will be consumed by codegen.
struct LowerMMAFill : public OpRewritePattern<MMAFillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAFillOp op,
                                PatternRewriter &rewriter) const override {
    auto fragType = op.getResult().getType();
    // Keep the fill as-is since it already represents the low-level operation.
    // The value is already a scalar constant that will become a register fill.
    // Mark it with an attribute to signal it's been lowered.
    if (op->hasAttr("lowered"))
      return failure();
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// MMA load: load from shared memory tensor tile into MMA fragment.
// On SM90: this maps to ldmatrix/wgmma load from shared memory
// On SM80: this maps to wmma::load_matrix_sync
struct LowerMMALoad : public OpRewritePattern<MMALoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MMALoadOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// MMA exec: perform the matrix multiply-accumulate.
// The target MMA strategy is read from the module attribute "coir.mma_target",
// which is set per-target by the driver (e.g. "wgmma", "mma_sync", "ukernel").
// When running via coir-opt, the --target-arch option is used as fallback.
struct LowerMMAExec : public OpRewritePattern<MMAExecOp> {
  std::string mmaTarget;
  LowerMMAExec(MLIRContext *ctx, StringRef mmaTarget)
      : OpRewritePattern(ctx), mmaTarget(mmaTarget.str()) {}

  LogicalResult matchAndRewrite(MMAExecOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();

    if (!mmaTarget.empty())
      op->setAttr("target", rewriter.getStringAttr(mmaTarget));

    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// MMA store: store result from MMA fragment to shared memory tensor.
// On SM90: stmatrix
// On SM80: wmma::store_matrix_sync
struct LowerMMAStore : public OpRewritePattern<MMAStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("lowered"))
      return failure();
    op->setAttr("lowered", rewriter.getUnitAttr());
    return success();
  }
};

// Infer MMA target from --target-arch string.
// This mirrors Target::MMATargetName() for standalone coir-opt usage
// where the Choreo target interface is not available.
static std::string inferMMATargetFromArch(llvm::StringRef arch) {
  if (arch.empty()) return "";
  if (arch.contains("sm_9")) return "wgmma";
  if (arch.contains("sm_")) return "mma_sync";
  return "ukernel";
}

struct LowerMMAPass : public ::coir::impl::LowerMMABase<LowerMMAPass> {
  using LowerMMABase::LowerMMABase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    // Module attribute (set by the driver via Target::MMATargetName)
    // is authoritative. Fall back to --target-arch for coir-opt.
    std::string mmaTarget;
    if (auto module = dyn_cast<ModuleOp>(getOperation()))
      mmaTarget = CoIR::GetMMATarget(module).str();
    else if (auto pm = getOperation()->getParentOfType<ModuleOp>())
      mmaTarget = CoIR::GetMMATarget(pm).str();
    if (mmaTarget.empty())
      mmaTarget = inferMMATargetFromArch(targetArch);

    RewritePatternSet patterns(ctx);
    patterns.add<LowerMMAFill>(ctx);
    patterns.add<LowerMMALoad>(ctx);
    patterns.add<LowerMMAExec>(ctx, mmaTarget);
    patterns.add<LowerMMAStore>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createLowerMMAPass() {
  return std::make_unique<LowerMMAPass>();
}
} // namespace coir
