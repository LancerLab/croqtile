//===- HoistAssertions.cpp - Hoist runtime assertions to earliest site ----===//
//
// Target-agnostic assertion hoisting pass for CoIR.
//
// Promotes coir.assert ops from USE -> HOIST or ENTRY based on SSA
// def-chain analysis:
//   - ENTRY: all condition inputs are kernel arguments (host-side check)
//   - HOIST: condition is loop-invariant w.r.t. enclosing loop
//   - USE:   condition depends on loop IV or is inside conditional
//
// Mirrors Choreo's AssertSite pass but operates on MLIR SSA graph.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace coir {
#define GEN_PASS_DECL_HOISTASSERTIONS
#define GEN_PASS_DEF_HOISTASSERTIONS
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

static bool isDefinedOutsideRegion(Value v, Region* region) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return !region->isAncestor(arg.getOwner()->getParent());
  return !region->isAncestor(v.getDefiningOp()->getParentRegion());
}

static bool allInputsFromKernelArgs(Value cond, KernelOp kernel) {
  SmallVector<Value> worklist;
  DenseSet<Value> visited;
  worklist.push_back(cond);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner()->getParentOp() == kernel.getOperation()) continue;
      return false;
    }

    auto* defOp = v.getDefiningOp();
    if (!defOp) return false;

    if (isa<arith::ConstantOp>(defOp)) continue;

    for (Value operand : defOp->getOperands()) worklist.push_back(operand);
  }
  return true;
}

static bool isInsideConditional(Operation* op) {
  auto* parent = op->getParentOp();
  while (parent) {
    if (isa<scf::IfOp>(parent)) return true;
    parent = parent->getParentOp();
  }
  return false;
}

static void collectDefChain(Value root, Region* boundary,
                            SmallVectorImpl<Operation*>& opsToMove) {
  SmallVector<Value> worklist;
  DenseSet<Operation*> visited;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto* defOp = v.getDefiningOp();
    if (!defOp) continue;
    if (!boundary->isAncestor(defOp->getParentRegion())) continue;
    if (!visited.insert(defOp).second) continue;
    opsToMove.push_back(defOp);
    for (Value operand : defOp->getOperands()) worklist.push_back(operand);
  }
}

static void moveOpsBeforeInTopoOrder(SmallVectorImpl<Operation*>& ops,
                                     Block* destBlock,
                                     Block::iterator destPoint) {
  DenseSet<Operation*> opSet(ops.begin(), ops.end());
  SmallVector<Operation*> sorted;
  DenseSet<Operation*> emitted;

  std::function<void(Operation*)> visit = [&](Operation* op) {
    if (!opSet.count(op) || emitted.count(op)) return;
    for (Value operand : op->getOperands())
      if (auto* defOp = operand.getDefiningOp()) visit(defOp);
    sorted.push_back(op);
    emitted.insert(op);
  };

  for (auto* op : ops) visit(op);

  for (auto* op : sorted) op->moveBefore(destBlock, destPoint);
}

static Operation* findOutermostInvariantLoop(AssertOp assertOp) {
  Operation* bestLoop = nullptr;
  auto* current = assertOp->getParentOp();

  while (current) {
    if (auto foreachOp = dyn_cast<ForeachOp>(current)) {
      Region& loopRegion = foreachOp.getBody();
      bool allOutside = true;

      SmallVector<Value> worklist;
      DenseSet<Value> visited;
      worklist.push_back(assertOp.getCondition());

      while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (!visited.insert(v).second) continue;
        if (!isDefinedOutsideRegion(v, &loopRegion)) {
          allOutside = false;
          break;
        }
        if (auto* defOp = v.getDefiningOp()) {
          if (!isa<arith::ConstantOp>(defOp))
            for (Value operand : defOp->getOperands())
              worklist.push_back(operand);
        }
      }

      if (allOutside)
        bestLoop = current;
      else
        break;
    }
    current = current->getParentOp();
  }
  return bestLoop;
}

struct HoistAssertionsPass
    : public ::coir::impl::HoistAssertionsBase<HoistAssertionsPass> {
  using HoistAssertionsBase::HoistAssertionsBase;

  void runOnOperation() override {
    SmallVector<AssertOp> asserts;
    getOperation()->walk([&](AssertOp op) { asserts.push_back(op); });

    for (auto assertOp : asserts) {
      if (assertOp.getSite() == AssertSite::ENTRY) continue;

      if (isInsideConditional(assertOp)) continue;

      auto kernel = assertOp->getParentOfType<KernelOp>();
      if (!kernel) continue;

      if (allInputsFromKernelArgs(assertOp.getCondition(), kernel)) {
        assertOp.setSiteAttr(
            AssertSiteAttr::get(assertOp.getContext(), AssertSite::ENTRY));

        auto& entryBlock = kernel.getBody().front();
        SmallVector<Operation*> chain;
        collectDefChain(assertOp.getCondition(), &kernel.getBody(), chain);
        chain.push_back(assertOp);
        moveOpsBeforeInTopoOrder(chain, &entryBlock, entryBlock.begin());
        continue;
      }

      auto* targetLoop = findOutermostInvariantLoop(assertOp);
      if (targetLoop) {
        assertOp.setSiteAttr(
            AssertSiteAttr::get(assertOp.getContext(), AssertSite::HOIST));
        SmallVector<Operation*> chain;
        collectDefChain(assertOp.getCondition(),
                        &targetLoop->getParentOfType<KernelOp>().getBody(),
                        chain);
        chain.push_back(assertOp);
        moveOpsBeforeInTopoOrder(chain, targetLoop->getBlock(),
                                 targetLoop->getIterator());
      }
    }
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createHoistAssertionsPass() {
  return std::make_unique<HoistAssertionsPass>();
}
} // namespace coir
