//===- EstimateAssertCost.cpp - Stamp cost metadata on coir.assert ops ----===//
//
// Walks all coir.assert ops and stamps estimated_cost (raw trip-count
// product), cost_class (classified tier), and enabled (threshold gate).
//
// Run after coir-hoist-assertions so cost reflects the final position.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace coir {
#define GEN_PASS_DECL_ESTIMATEASSERTCOST
#define GEN_PASS_DEF_ESTIMATEASSERTCOST
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

constexpr uint64_t kDynamicTripCount = 100;
constexpr uint64_t kMediumCostThreshold = 50;
constexpr uint64_t kHighCostThreshold = 500;

static uint64_t estimateNestingCost(AssertOp assertOp, KernelOp kernel) {
  uint64_t cost = 1;
  auto *current = assertOp->getParentOp();
  while (current && current != kernel.getOperation()) {
    if (auto foreachOp = dyn_cast<ForeachOp>(current)) {
      auto ub = foreachOp.getUpperBound();
      if (auto cst = ub.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
          cost *= intAttr.getInt();
        else
          cost *= kDynamicTripCount;
      } else {
        cost *= kDynamicTripCount;
      }
    } else if (isa<scf::WhileOp>(current) || isa<CoIRWhileOp>(current)) {
      cost *= kDynamicTripCount;
    }
    current = current->getParentOp();
  }
  return cost;
}

static AssertCost categorizeCost(uint64_t cost) {
  if (cost >= kHighCostThreshold) return AssertCost::HIGH;
  if (cost >= kMediumCostThreshold) return AssertCost::MEDIUM;
  if (cost == 1) return AssertCost::ENTRY;
  return AssertCost::LOW;
}

struct EstimateAssertCostPass
    : public ::coir::impl::EstimateAssertCostBase<EstimateAssertCostPass> {
  using EstimateAssertCostBase::EstimateAssertCostBase;

  void runOnOperation() override {
    // Read threshold from module attribute (set by Pipeline), or fall back to
    // the pass option (for standalone coir-opt usage).
    int threshold = costThreshold;
    if (auto attr = getOperation()->getAttrOfType<IntegerAttr>(
            "coir.cost_threshold"))
      threshold = attr.getInt();

    getOperation()->walk([&](AssertOp assertOp) {
      auto kernel = assertOp->getParentOfType<KernelOp>();
      uint64_t cost = kernel ? estimateNestingCost(assertOp, kernel) : 1;
      auto tier = categorizeCost(cost);
      bool enabled = static_cast<int>(tier) < threshold;
      auto *ctx = assertOp.getContext();
      assertOp->setAttr("estimated_cost",
                        IntegerAttr::get(IntegerType::get(ctx, 64), cost));
      assertOp->setAttr("cost_class", AssertCostAttr::get(ctx, tier));
      assertOp->setAttr("enabled", BoolAttr::get(ctx, enabled));
    });
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEstimateAssertCostPass() {
  return std::make_unique<EstimateAssertCostPass>();
}
} // namespace coir
