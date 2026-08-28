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

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/Passes.h"

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
                                PatternRewriter& rewriter) const override {
    auto srcType = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstType = llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcType || !dstType) return failure();

    if (!isGlobalSharedCopy(srcType, dstType)) return failure();

    // Extract base tensors and tile offsets.
    // Convention: TensorTileOp indices = [offsets..., dynDimVals...]
    // where offsets has `rank` elements and dynDimVals has numDynDims.
    // Tile indices are multiplied by tile sizes to get element offsets.
    Value srcBase = op.getSource();
    Value dstBase = op.getDest();
    llvm::SmallVector<Value> srcOffsets;
    llvm::SmallVector<Value> dstOffsets;

    // Materialize a tensor's per-dim extent as an index SSA value. Anchor
    // tiles (result shape `1x1x...`) carry only the chunk index, so the
    // element offset must be scaled by the DMA counterpart's actual per-dim
    // chunk extent (e.g. `K/2` for a dynamic `i0.span / #idx` chunk).
    auto materializeDimExtent = [&](Value tensor, unsigned dim) -> Value {
      while (auto t = tensor.getDefiningOp<TensorTileOp>())
        tensor = t.getSource();
      auto tty = llvm::cast<coir::TensorType>(tensor.getType());
      if (!tty.isDynamicDim(dim))
        return rewriter.create<mlir::arith::ConstantIndexOp>(
            op.getLoc(), tty.getShape()[dim]);
      auto resolve = [&](mlir::OperandRange dynDims) -> Value {
        unsigned dynIdx = 0;
        for (unsigned i = 0; i < tty.getShape().size(); ++i) {
          if (!tty.isDynamicDim(i))
            continue;
          if (i == dim)
            return dynDims[dynIdx];
          ++dynIdx;
        }
        llvm_unreachable("dynamic dim not found in operands");
      };
      if (auto alloc = tensor.getDefiningOp<TensorAllocOp>())
        return resolve(alloc.getDynamicDims());
      if (auto bind = tensor.getDefiningOp<TensorBindDimsOp>())
        return resolve(bind.getDynamicDims());
      return nullptr;
    };

    // Scale a tile index to an element offset. Tiles with a static chunk size
    // (>1) scale by that size; anchor tiles (size 1) scale by the counterpart's
    // dim extent, which may be dynamic.
    auto scaleTileIndex = [&](Value idx, int64_t tileDim, Value counterpart,
                              unsigned i) -> Value {
      if (tileDim > 1) {
        auto tileSize = rewriter.create<mlir::arith::ConstantIndexOp>(
            op.getLoc(), tileDim);
        return rewriter.create<mlir::arith::MulIOp>(op.getLoc(), idx, tileSize);
      }
      auto cTy = llvm::dyn_cast<coir::TensorType>(counterpart.getType());
      if (!cTy || i >= cTy.getShape().size())
        return idx;
      Value extent = materializeDimExtent(counterpart, i);
      if (!extent)
        return idx;
      if (auto c = extent.getDefiningOp<mlir::arith::ConstantIndexOp>())
        if (c.value() == 1)
          return idx;
      return rewriter.create<mlir::arith::MulIOp>(op.getLoc(), idx, extent);
    };

    if (auto tileOp = srcBase.template getDefiningOp<TensorTileOp>()) {
      srcBase = tileOp.getSource();
      auto tileTy = llvm::cast<coir::TensorType>(tileOp.getResult().getType());
      unsigned rank = tileTy.getRank();
      auto tileShape = tileTy.getShape();
      auto allIdx = tileOp.getIndices();
      bool elementOffset = tileOp->hasAttr("coir.element_offset");
      for (unsigned i = 0; i < std::min((unsigned)allIdx.size(), rank); ++i) {
        Value idx = allIdx[i];
        // Element-offset tiles (chained subspan/modspan) carry the offset
        // already multiplied by the strides, so pass through unchanged.
        if (elementOffset) {
          srcOffsets.push_back(idx);
          continue;
        }
        // Multiply tile index by tile size to get the per-dim offset.
        if (i < tileShape.size())
          idx = scaleTileIndex(idx, tileShape[i], op.getDest(), i);
        srcOffsets.push_back(idx);
      }
    }
    if (auto tileOp = dstBase.template getDefiningOp<TensorTileOp>()) {
      dstBase = tileOp.getSource();
      auto tileTy = llvm::cast<coir::TensorType>(tileOp.getResult().getType());
      unsigned rank = tileTy.getRank();
      auto tileShape = tileTy.getShape();
      auto allIdx = tileOp.getIndices();
      bool elementOffset = tileOp->hasAttr("coir.element_offset");
      for (unsigned i = 0; i < std::min((unsigned)allIdx.size(), rank); ++i) {
        Value idx = allIdx[i];
        if (elementOffset) {
          dstOffsets.push_back(idx);
          continue;
        }
        if (i < tileShape.size())
          idx = scaleTileIndex(idx, tileShape[i], op.getSource(), i);
        dstOffsets.push_back(idx);
      }
    }
    bool hasOffsets = !srcOffsets.empty() || !dstOffsets.empty();

    // For TMA copies, tile offsets become runtime.desc coordinates.
    // For DMA copies with tile offsets, also decompose: the base geometry
    // goes into const.desc (hoistable) and offsets go into runtime.desc.
    constexpr bool isTMA = std::is_same_v<CopyOpTy, TmaCopyOp>;

    auto descType = coir::DMADescType::get(rewriter.getContext());
    auto descRtType = coir::FinalDMADescType::get(rewriter.getContext());
    auto tokenType = coir::AsyncTokenType::get(rewriter.getContext());
    Location loc = op.getLoc();

    // Determine kind: tile offsets imply Slice (sub-region access).
    // TmaCopyOp has no kind attribute - derive entirely from structure.
    // DmaCopyOp carries a kind attr (default Copy) - override to Slice when
    // tiled.
    auto kind = coir::DMAKind::Copy;
    if constexpr (!isTMA) kind = op.getKind();
    if (hasOffsets && kind == coir::DMAKind::Copy) kind = coir::DMAKind::Slice;
    auto kindAttr = coir::DMAKindAttr::get(rewriter.getContext(), kind);

    mlir::IntegerAttr swizAttr;
    mlir::UnitAttr zfillAttr;
    if constexpr (isTMA) {
      if (auto sb = op.getSwizzleBytes())
        swizAttr = rewriter.getI64IntegerAttr(*sb);
      if (op.getZfill()) zfillAttr = rewriter.getUnitAttr();
    }
    auto constDesc = rewriter.create<DMAConstDescOp>(
        loc, descType, srcBase, dstBase, kindAttr,
        isTMA ? rewriter.getUnitAttr() : nullptr, swizAttr, zfillAttr);

    // Record tile presence for emission-side config dispatch.
    bool hasSrcTile =
        op.getSource().template getDefiningOp<TensorTileOp>() != nullptr;
    bool hasDstTile =
        op.getDest().template getDefiningOp<TensorTileOp>() != nullptr;
    if (hasSrcTile) constDesc->setAttr("src_tiled", rewriter.getUnitAttr());
    if (hasDstTile) constDesc->setAttr("dst_tiled", rewriter.getUnitAttr());
    // Store tile shape for slice_deslice emission (the actual data transfer
    // size).
    if (hasSrcTile || hasDstTile) {
      auto tileOp = hasSrcTile
                        ? op.getSource().template getDefiningOp<TensorTileOp>()
                        : op.getDest().template getDefiningOp<TensorTileOp>();
      auto tileShape =
          llvm::cast<coir::TensorType>(tileOp.getResult().getType()).getShape();
      llvm::SmallVector<int64_t> shapeVec(tileShape.begin(), tileShape.end());
      constDesc->setAttr("tile_shape", rewriter.getDenseI64ArrayAttr(shapeVec));
    }

    // Forward pad/transpose attributes from the original DmaCopyOp.
    if constexpr (!isTMA) {
      for (const char* attrName :
           {"pad_low", "pad_high", "pad_value", "transpose_perm"}) {
        if (auto attr = op->getAttr(attrName))
          constDesc->setAttr(attrName, attr);
      }
    }
    auto prefetch =
        rewriter.create<DMADescPrefetchOp>(loc, descRtType, constDesc.getOut());

    Value invokeDesc;
    if (hasOffsets) {
      // Chain: prefetch -> [src_runtime_desc] -> [dst_runtime_desc] -> invoke
      Value chain = prefetch.getOut();
      if (!srcOffsets.empty()) {
        auto srcRt = rewriter.create<DMADescRuntimeOp>(loc, descRtType, chain,
                                                       srcOffsets);
        chain = srcRt.getOut();
      }
      if (!dstOffsets.empty()) {
        auto dstRt = rewriter.create<DMADescRuntimeOp>(loc, descRtType, chain,
                                                       dstOffsets);
        dstRt->setAttr("dst_offsets", rewriter.getUnitAttr());
        chain = dstRt.getOut();
      }
      invokeDesc = chain;
    } else {
      invokeDesc = prefetch.getOut();
    }

    llvm::SmallVector<Value> dynDims;
    llvm::SmallVector<int64_t> copyShapeVec;
    if (hasOffsets) {
      auto srcTy = llvm::dyn_cast<coir::TensorType>(op.getSource().getType());
      auto dstTy = llvm::dyn_cast<coir::TensorType>(op.getDest().getType());
      if (srcTy && dstTy &&
          (srcTy.hasDynamicShape() || dstTy.hasDynamicShape())) {
        // Use the tile shape (the more constrained type from the tiled
        // operand). For loads, source is tiled; for stores, dest is tiled.
        auto srcTileOp = op.getSource().template getDefiningOp<TensorTileOp>();
        auto dstTileOp = op.getDest().template getDefiningOp<TensorTileOp>();
        llvm::ArrayRef<int64_t> shapeRef;
        TensorTileOp activeTileOp = nullptr;
        if (srcTileOp) {
          shapeRef = srcTy.getShape();
          activeTileOp = srcTileOp;
        } else if (dstTileOp) {
          shapeRef = dstTy.getShape();
          activeTileOp = dstTileOp;
        } else {
          shapeRef =
              srcTy.hasDynamicShape() ? srcTy.getShape() : dstTy.getShape();
        }
        copyShapeVec.assign(shapeRef.begin(), shapeRef.end());
        unsigned numDyn = 0;
        for (auto d : shapeRef)
          if (mlir::ShapedType::isDynamic(d)) numDyn++;
        if (activeTileOp && numDyn > 0) {
          auto indices = activeTileOp.getIndices();
          unsigned rank = shapeRef.size();
          if (indices.size() >= rank + numDyn) {
            for (unsigned i = rank; i < rank + numDyn; ++i)
              dynDims.push_back(indices[i]);
          }
        }
      }
    }

    auto invoke = rewriter.create<DMAInvokeOp>(
        loc, /*done=*/mlir::Type(tokenType), /*desc=*/invokeDesc,
        /*dyn_dims=*/mlir::ValueRange(dynDims),
        /*thr_layout=*/nullptr, /*val_layout=*/nullptr,
        /*copy_atom=*/nullptr, /*need_pred=*/nullptr,
        /*prediction=*/nullptr, /*swizzle=*/nullptr,
        /*copy_shape=*/nullptr);
    if (!copyShapeVec.empty())
      invoke->setAttr("copy_shape",
                      rewriter.getDenseI64ArrayAttr(copyShapeVec));

    if (op.getToken() && !op.getToken().use_empty())
      rewriter.replaceAllUsesWith(op.getToken(), invoke.getDone());
    else
      rewriter.create<WaitOp>(loc, invoke.getDone());

    auto srcTileOp = op.getSource().template getDefiningOp<TensorTileOp>();
    auto dstTileOp = op.getDest().template getDefiningOp<TensorTileOp>();
    rewriter.eraseOp(op);
    if (srcTileOp && srcTileOp->use_empty()) rewriter.eraseOp(srcTileOp);
    if (dstTileOp && dstTileOp->use_empty()) rewriter.eraseOp(dstTileOp);

    return success();
  }
};

struct LowerDMADescPass
    : public ::coir::impl::LowerDMADescBase<LowerDMADescPass> {
  using LowerDMADescBase::LowerDMADescBase;

  void runOnOperation() override {
    // Gate: require target DMA/TMA capability.
    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module) module = getOperation()->getParentOfType<ModuleOp>();
    if (module) {
      auto hasDMA = module->getAttrOfType<BoolAttr>("coir.has_dma");
      auto hasTMA = module->getAttrOfType<BoolAttr>("coir.has_tma");
      bool active = (!hasDMA && !hasTMA) || // standalone test
                    (hasDMA && hasDMA.getValue()) ||
                    (hasTMA && hasTMA.getValue());
      if (!active) return;
    }

    auto* ctx = &getContext();
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
