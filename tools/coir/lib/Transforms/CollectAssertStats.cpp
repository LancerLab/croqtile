//===- CollectAssertStats.cpp - Collect assertion statistics ---------------===//
//
// Walks coir.assert ops and accumulates per-tier, per-usage, and
// enabled/disabled counts into CCtx().GetAssessmentStats().
// Does not modify the IR. Run after coir-estimate-assert-cost.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/Passes.h"

#include "context.hpp"

#include "mlir/Pass/Pass.h"

namespace coir {
#define GEN_PASS_DECL_COLLECTASSERTSTATS
#define GEN_PASS_DEF_COLLECTASSERTSTATS
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

struct CollectAssertStatsPass
    : public ::coir::impl::CollectAssertStatsBase<CollectAssertStatsPass> {
  using CollectAssertStatsBase::CollectAssertStatsBase;

  void runOnOperation() override {
    auto &s = Choreo::CCtx().GetAssessmentStats();

    // Only collect cost-classification and enabled/disabled counts here.
    // Base stats (runtime_total, per-usage runtime) are already counted by
    // CollectSemaStats from the assessor log, avoiding double-counting for
    // assertions that appear in both the log and as coir.assert ops.
    getOperation()->walk([&](AssertOp op) {
      if (auto costAttr =
              op->getAttrOfType<AssertCostAttr>("cost_class")) {
        switch (costAttr.getValue()) {
        case AssertCost::ENTRY:  s.runtime_entry++; break;
        case AssertCost::LOW:    s.runtime_low++; break;
        case AssertCost::MEDIUM: s.runtime_medium++; break;
        case AssertCost::HIGH:   s.runtime_high++; break;
        }
      }

      if (auto ea = op->getAttrOfType<BoolAttr>("enabled")) {
        if (ea.getValue()) s.runtime_enabled++;
        else s.runtime_disabled++;
      }
    });
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createCollectAssertStatsPass() {
  return std::make_unique<CollectAssertStatsPass>();
}
} // namespace coir
