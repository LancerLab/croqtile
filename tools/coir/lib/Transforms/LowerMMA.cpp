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

// Infer K dimension from the LHS MMALoadOp's source tensor.
// For a matmul C[M,N] += A[M,K] * B[K,N], the LHS load source is [M,K].
static int64_t inferKFromLHS(MMAExecOp op) {
  auto lhsLoad = op.getLhs().getDefiningOp<MMALoadOp>();
  if (!lhsLoad)
    return 0;
  auto srcType = dyn_cast<coir::TensorType>(lhsLoad.getSource().getType());
  if (!srcType || srcType.getShape().size() < 2)
    return 0;
  return srcType.getShape().back();
}

// Determine if an MxNxK config maps to CTMMA (inline PTX mma.sync) rather than
// WMMA. CTMMA configs are m16n8kX and m8n8kX families.
static bool isCTMMAConfig(int64_t M, int64_t N, int64_t K) {
  if (M == 16 && N == 8 && (K == 4 || K == 8 || K == 16 || K == 32 || K == 64))
    return true;
  if (M == 8 && N == 8 && (K == 4 || K == 16 || K == 32 || K == 128))
    return true;
  return false;
}

// Compute register counts matching CuTe atom DRegisters/ARegisters/BRegisters.
// SM70 8x8x4 uses 4-wide quad groups (8 regs per thread for f32 accum).
// SM80+ m16n8/m8n8 shapes use standard M*N/32 for D/C.

static int64_t getRegNumD(int64_t M, int64_t N, int64_t K,
                          mlir::Type accumType) {
  if (M == 8 && N == 8 && K == 4) {
    if (accumType.isF32()) return 8;
    return 4; // f16 accum: uint32_t[4]
  }
  int64_t base = M * N / 32;
  // f16/bf16 accumulators pack 2 values per uint32_t register
  if (accumType.isF16() || accumType.isBF16())
    return base / 2;
  return base;
}

static int64_t getRegNumA(int64_t M, int64_t N, int64_t K) {
  if (M == 8 && N == 8 && K == 4) return 2;
  if (M == 16 && N == 8 && K == 4) return 2;
  if (M == 16 && N == 8 && K == 8) return 2;
  if (M == 16 && N == 8 && K == 16) return 4;
  if (M == 16 && N == 8 && K == 32) return 8;
  if (M == 8 && N == 8 && K == 16) return 1;
  if (M == 8 && N == 8 && K == 32) return 2;
  return M * K / 64;
}

static int64_t getRegNumB(int64_t M, int64_t N, int64_t K) {
  if (M == 8 && N == 8 && K == 4) return 2;
  if (M == 16 && N == 8 && K == 4) return 1;
  if (M == 16 && N == 8 && K == 8) return 1;
  if (M == 16 && N == 8 && K == 16) return 2;
  if (M == 16 && N == 8 && K == 32) return 4;
  if (M == 8 && N == 8 && K == 16) return 1;
  if (M == 8 && N == 8 && K == 32) return 2;
  return N * K / 64;
}

// Map MLIR types to the CuTe type letter for the FMA atom name.
static std::string typeToFMALetter(mlir::Type t) {
  if (t.isF32()) return "F32";
  if (t.isF16()) return "F16";
  if (t.isBF16()) return "BF16";
  if (t.isInteger(8)) return "S8";
  if (t.isInteger(32)) return "S32";
  return "F32";
}

// Get SM generation for the atom prefix.
static std::string getArchPrefix(int64_t M, int64_t N, int64_t K) {
  if (M == 8 && N == 8 && K == 4) return "SM70";
  return "SM80";
}

// Build the full CuTe FMA atom name (e.g., "cute::SM80_16x8x8_F32F16F16F32_TN").
static std::string buildFMAAtomName(int64_t M, int64_t N, int64_t K,
                                    mlir::Type accumType, mlir::Type aType,
                                    mlir::Type bType) {
  std::string prefix = getArchPrefix(M, N, K);
  std::string d_str = typeToFMALetter(accumType);
  // f32 operands with K=4 or K=8 (and M=16, N=8) are TF32 MMA atoms
  bool isTF32 = aType.isF32() && M == 16 && N == 8 && (K == 4 || K == 8);
  std::string a_str = isTF32 ? "TF32" : typeToFMALetter(aType);
  std::string b_str = isTF32 ? "TF32" : typeToFMALetter(bType);
  std::string c_str = d_str;

  return "cute::" + prefix + "_" + std::to_string(M) + "x" +
         std::to_string(N) + "x" + std::to_string(K) + "_" + d_str + a_str +
         b_str + c_str + "_TN";
}

// MMA exec: perform the matrix multiply-accumulate.
// The target MMA strategy is read from the module attribute "coir.mma_target",
// which is set per-target by the driver (e.g. "wgmma", "mma_sync", "ukernel").
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

    // For mma_sync target, detect CTMMA configs and stamp atom metadata.
    if (mmaTarget == "mma_sync") {
      // Derive M, N, K from operand fragment shapes (more reliable than accum):
      // lhs is [M, K], rhs is [K, N].
      auto lhsType = cast<coir::MMAFragType>(op.getLhs().getType());
      auto rhsType = cast<coir::MMAFragType>(op.getRhs().getType());
      auto lhsShape = lhsType.getShape();
      auto rhsShape = rhsType.getShape();
      int64_t M = lhsShape.size() > 0 ? lhsShape[0] : 16;
      int64_t K = inferKFromLHS(op);
      int64_t N = rhsShape.size() > 1 ? rhsShape[1] : (rhsShape.size() > 0 ? rhsShape[0] : 16);
      if (K == 0)
        K = lhsShape.size() > 1 ? lhsShape[1] : 16;

      if (K > 0 && isCTMMAConfig(M, N, K)) {
        // Infer element types from fragment types.
        auto fragType = cast<coir::MMAFragType>(op.getResult().getType());
        mlir::Type accumType = fragType.getElementType();
        mlir::Type aType = lhsType.getElementType();
        mlir::Type bType = rhsType.getElementType();

        std::string policyName =
            "CUTE_MMA_M" + std::to_string(M) + "N" + std::to_string(N) +
            "K" + std::to_string(K);
        std::string fmaAtom = buildFMAAtomName(M, N, K, accumType, aType, bType);

        op.setMmaAtomNameAttr(rewriter.getStringAttr(policyName));
        op.setKDimAttr(rewriter.getI64IntegerAttr(K));
        op.setRegNumAAttr(rewriter.getI64IntegerAttr(getRegNumA(M, N, K)));
        op.setRegNumBAttr(rewriter.getI64IntegerAttr(getRegNumB(M, N, K)));
        op.setRegNumDAttr(rewriter.getI64IntegerAttr(getRegNumD(M, N, K, accumType)));
        // Also store the FMA CuTe atom for emission.
        op->setAttr("fma_atom", rewriter.getStringAttr(fmaAtom));
      }
    }

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

struct LowerMMAPass : public ::coir::impl::LowerMMABase<LowerMMAPass> {
  using LowerMMABase::LowerMMABase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    // Read coir.mma_target from the module (set by the driver via
    // Target::MMATargetName()). For standalone coir-opt tests, embed
    // coir.mma_target directly in the test IR.
    std::string mmaTarget;
    if (auto module = dyn_cast<ModuleOp>(getOperation()))
      mmaTarget = CoIR::GetMMATarget(module).str();
    else if (auto pm = getOperation()->getParentOfType<ModuleOp>())
      mmaTarget = CoIR::GetMMATarget(pm).str();

    if (mmaTarget.empty()) {
      return;
    }

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
