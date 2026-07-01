//===- PlanDMACopy.cpp - Plan tiled copy strategy for DMA invoke ops ------===//
//
// Analyzes DMAInvokeOp instances (non-TMA) and stamps tiled-copy parameters
// (thread layout, value layout, copy atom, predication) when cooperative copy
// is profitable. EmitCUDA reads these attrs to emit choreo::tiled_copy<...>
// instead of choreo::naive_copy.
//
// Runs after LowerDMADesc + HoistDMAConfig, before LowerCopy.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "coir-plan-dma-copy"

namespace coir {
#define GEN_PASS_DECL_PLANDMACOPY
#define GEN_PASS_DEF_PLANDMACOPY
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

struct TiledCopyConfig {
  int64_t thrRows = 1;
  int64_t thrCols = 1;
  int64_t valRows = 1;
  int64_t valCols = 1;
  std::string atomName;
  bool needPred = false;
  int64_t predDim0 = 0;
  int64_t predDim1 = 0;
  bool swizzle = false;
};

static int64_t getThreadCount(DMAInvokeOp invoke) {
  auto *parent = invoke->getParentOp();
  while (parent) {
    if (auto parallel = dyn_cast<ParallelOp>(parent)) {
      auto level = parallel.getLevel();
      if (level == ParallelLevel::THREAD || level == ParallelLevel::GROUP) {
        auto bounds = parallel.getBounds();
        int64_t count = 1;
        for (auto b : bounds)
          count *= b;
        if (level == ParallelLevel::GROUP) {
          auto *outer = parallel->getParentOp();
          while (outer) {
            if (auto outerP = dyn_cast<ParallelOp>(outer)) {
              if (outerP.getLevel() == ParallelLevel::THREAD) {
                auto ob = outerP.getBounds();
                for (auto b : ob)
                  count *= b;
                break;
              }
            }
            outer = outer->getParentOp();
          }
        }
        return count;
      }
    }
    parent = parent->getParentOp();
  }

  auto *op = invoke->getParentOp();
  while (op) {
    if (auto kernel = dyn_cast<KernelOp>(op)) {
      if (auto lb = kernel->getAttrOfType<DenseI64ArrayAttr>("launch_bounds")) {
        auto arr = lb.asArrayRef();
        int64_t threads = 1;
        for (auto v : arr)
          threads *= v;
        return threads;
      }
      break;
    }
    op = op->getParentOp();
  }
  return 0;
}

static bool isGlobalToShared(coir::TensorType srcType,
                             coir::TensorType dstType) {
  return srcType.getMemorySpace() <= 0 && dstType.getMemorySpace() == 1;
}

static bool isSharedToGlobal(coir::TensorType srcType,
                             coir::TensorType dstType) {
  return srcType.getMemorySpace() == 1 && dstType.getMemorySpace() <= 0;
}

static size_t getElemBits(Type elemType) {
  if (elemType.isF16() || elemType.isBF16())
    return 16;
  if (elemType.isF32() || elemType.isInteger(32))
    return 32;
  if (elemType.isF64() || elemType.isInteger(64))
    return 64;
  if (elemType.isInteger(8))
    return 8;
  if (elemType.isInteger(16))
    return 16;
  return 32;
}

static std::string selectCopyAtom(size_t alignBits, bool cpAsync) {
  if (cpAsync) {
    switch (alignBits) {
    case 128:
      return "cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<"
             "cute::uint128_t>, ELEM>";
    case 64:
      return "cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<"
             "cute::uint64_t>, ELEM>";
    case 32:
      return "cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<"
             "cute::uint32_t>, ELEM>";
    default:
      return "";
    }
  }
  switch (alignBits) {
  case 128:
    return "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>"
           ", ELEM>";
  case 64:
    return "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<64>"
           ", ELEM>";
  case 32:
    return "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<32>"
           ", ELEM>";
  case 16:
    return "cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<16>"
           ", ELEM>";
  default:
    return "cute::Copy_Atom<cute::UniversalCopy<ELEM>, ELEM>";
  }
}

static std::optional<TiledCopyConfig>
searchTiledConfig(int64_t boxM, int64_t boxN, int64_t threadsCount,
                  size_t elemBits, bool isG2S) {
  if (threadsCount < 2 || boxN < 1 || boxM < 1)
    return std::nullopt;

  size_t maxVf = 128 / elemBits;
  bool cpAsync = isG2S && (threadsCount % 32 == 0);

  for (size_t vf = maxVf; vf >= 1; vf /= 2) {
    if (static_cast<size_t>(boxN) % vf != 0)
      continue;

    int64_t bestThrM = 0, bestThrN = 0, bestValM = 0, bestValN = 0;
    bool bestPred = false;
    size_t bestEffVec = 0;
    bool found = false;

    for (int64_t thrN = 1;
         thrN <= threadsCount &&
         static_cast<size_t>(thrN) * vf <= static_cast<size_t>(boxN);
         thrN *= 2) {
      if (threadsCount % thrN != 0 || boxN % thrN != 0)
        continue;
      int64_t thrM = threadsCount / thrN;
      int64_t valN = boxN / thrN;
      if (valN % static_cast<int64_t>(vf) != 0)
        continue;

      bool thisPred = false;
      int64_t effBoxM = boxM;
      if (boxM % thrM != 0) {
        thisPred = true;
        effBoxM = ((boxM + thrM - 1) / thrM) * thrM;
      }

      int64_t maxVm = std::max(
          int64_t{1}, static_cast<int64_t>(128 / elemBits) / valN);
      int64_t vm = maxVm;
      while (vm > 1 && effBoxM % (thrM * vm) != 0)
        vm /= 2;
      if (effBoxM % (thrM * vm) != 0)
        continue;

      size_t effVec =
          std::min({static_cast<size_t>(valN) * elemBits, vf * elemBits,
                    size_t{128}});

      bool better = !found || (!thisPred && bestPred) ||
                    (thisPred == bestPred && effVec > bestEffVec) ||
                    (thisPred == bestPred && effVec == bestEffVec &&
                     thrN > bestThrN);
      if (better) {
        bestThrM = thrM;
        bestThrN = thrN;
        bestValM = vm;
        bestValN = valN;
        bestPred = thisPred;
        bestEffVec = effVec;
        found = true;
      }
    }

    if (!found)
      continue;

    size_t alignBits = vf * elemBits;
    std::string atom = selectCopyAtom(alignBits, cpAsync);
    if (atom.empty())
      atom = selectCopyAtom(alignBits, false);
    if (atom.empty())
      continue;

    TiledCopyConfig cfg;
    cfg.thrRows = bestThrM;
    cfg.thrCols = bestThrN;
    cfg.valRows = bestValM;
    cfg.valCols = bestValN;
    cfg.atomName = atom;
    cfg.needPred = bestPred;
    if (bestPred) {
      cfg.predDim0 = boxM;
      cfg.predDim1 = boxN;
    }
    cfg.swizzle = false;
    return cfg;
  }
  return std::nullopt;
}

struct PlanDMACopyPass
    : public ::coir::impl::PlanDMACopyBase<PlanDMACopyPass> {
  using PlanDMACopyBase::PlanDMACopyBase;

  void runOnOperation() override {
    getOperation()->walk([&](DMAInvokeOp invoke) {
      if (invoke.getThrLayout())
        return;

      Value descVal = invoke.getDesc();
      DMAConstDescOp constDesc = nullptr;

      if (auto rtOp = descVal.getDefiningOp<DMADescRuntimeOp>()) {
        auto prefetch = rtOp.getIn().getDefiningOp<DMADescPrefetchOp>();
        if (prefetch)
          constDesc = prefetch.getIn().getDefiningOp<DMAConstDescOp>();
      } else if (auto prefetch =
                     descVal.getDefiningOp<DMADescPrefetchOp>()) {
        constDesc = prefetch.getIn().getDefiningOp<DMAConstDescOp>();
      }

      if (!constDesc)
        return;
      if (constDesc.getTma())
        return;

      auto srcType = dyn_cast<coir::TensorType>(constDesc.getSource().getType());
      auto dstType = dyn_cast<coir::TensorType>(constDesc.getDest().getType());
      if (!srcType || !dstType)
        return;

      bool g2s = isGlobalToShared(srcType, dstType);
      bool s2g = isSharedToGlobal(srcType, dstType);
      if (!g2s && !s2g)
        return;

      auto tileType = g2s ? dstType : srcType;
      auto shape = tileType.getShape();
      if (shape.empty())
        return;

      // Only apply tiled copy to 2D tensors for now. 3D+ tensors need
      // rank-reduction (batch loop peeling) which is not yet implemented.
      if (shape.size() != 2)
        return;

      int64_t boxM = shape[0];
      int64_t boxN = shape[1];

      if (boxN <= 0 || boxM <= 0)
        return;

      int64_t threads = getThreadCount(invoke);
      if (threads < 2)
        return;

      size_t elemBits = getElemBits(tileType.getElementType());

      auto cfg = searchTiledConfig(boxM, boxN, threads, elemBits, g2s);
      if (!cfg)
        return;

      auto *ctx = invoke.getContext();
      invoke.setThrLayoutAttr(
          DenseI64ArrayAttr::get(ctx, {cfg->thrRows, cfg->thrCols}));
      invoke.setValLayoutAttr(
          DenseI64ArrayAttr::get(ctx, {cfg->valRows, cfg->valCols}));
      invoke.setCopyAtomAttr(StringAttr::get(ctx, cfg->atomName));
      invoke.setNeedPredAttr(BoolAttr::get(ctx, cfg->needPred));
      if (cfg->needPred) {
        invoke.setPredictionAttr(
            DenseI64ArrayAttr::get(ctx, {cfg->predDim0, cfg->predDim1}));
      }
      invoke.setSwizzleAttr(BoolAttr::get(ctx, cfg->swizzle));
    });
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createPlanDMACopyPass() {
  return std::make_unique<PlanDMACopyPass>();
}
} // namespace coir
