//===- HoistAssertions.cpp - Hoist runtime assertions to earliest site ----===//
//
// SSA post-dominator based assertion hoisting for CoIR.
//
// Algorithm (computeHoistTarget):
//   1. Collect condition def-chain to find all dependencies.
//   2. Walk up from assertion's parent ops toward the kernel:
//      - scf.if    -> transparent (post-dom lift through)
//      - ForeachOp -> opaque (exit only if all deps are outside)
//      - scf.while -> opaque (never exit -- conservative)
//      - coir.while-> opaque (never exit -- conservative)
//      - KernelOp  -> ENTRY
//   3. Move assertion + def-chain to the computed target.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "coir-hoist-assertions"

namespace coir {
#define GEN_PASS_DECL_HOISTASSERTIONS
#define GEN_PASS_DEF_HOISTASSERTIONS
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

static bool isDefinedOutsideRegion(Value v, Region *region) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return !region->isAncestor(arg.getOwner()->getParent());
  return !region->isAncestor(v.getDefiningOp()->getParentRegion());
}

static bool allDepsMovableOutOfRegion(Value cond, Region *region) {
  SmallVector<Value> worklist;
  DenseSet<Value> visited;
  worklist.push_back(cond);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    if (isDefinedOutsideRegion(v, region)) continue;
    auto *defOp = v.getDefiningOp();
    if (!defOp) return false;
    if (isa<arith::ConstantOp>(defOp)) continue;
    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return true;
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
    auto *defOp = v.getDefiningOp();
    if (!defOp) return false;
    if (isa<arith::ConstantOp>(defOp)) continue;
    for (Value operand : defOp->getOperands()) worklist.push_back(operand);
  }
  return true;
}

struct HoistTarget {
  Operation *insertBefore = nullptr;
  Block *destBlock = nullptr;
  AssertSite site = AssertSite::USE;
};

static HoistTarget computeHoistTarget(AssertOp assertOp, KernelOp kernel) {
  HoistTarget target;
  Value cond = assertOp.getCondition();

  Operation *bestInsertBefore = nullptr;
  Block *bestBlock = nullptr;
  bool hoistedPastSomething = false;
  bool stoppedEarly = false;

  auto *current = assertOp->getParentOp();
  while (current && current != kernel.getOperation()) {
    if (isa<scf::IfOp>(current)) {
      bestInsertBefore = current;
      bestBlock = current->getBlock();
      hoistedPastSomething = true;
    } else if (auto foreachOp = dyn_cast<ForeachOp>(current)) {
      Region &loopRegion = foreachOp.getBody();
      if (allDepsMovableOutOfRegion(cond, &loopRegion)) {
        bestInsertBefore = current;
        bestBlock = current->getBlock();
        hoistedPastSomething = true;
      } else {
        stoppedEarly = true;
        break;
      }
    } else if (isa<scf::WhileOp>(current) || isa<CoIRWhileOp>(current)) {
      stoppedEarly = true;
      break;
    }
    current = current->getParentOp();
  }

  if (!stoppedEarly && allInputsFromKernelArgs(cond, kernel)) {
    target.destBlock = &kernel.getBody().front();
    target.insertBefore = nullptr;
    target.site = AssertSite::ENTRY;
    return target;
  }

  if (hoistedPastSomething && bestInsertBefore) {
    target.insertBefore = bestInsertBefore;
    target.destBlock = bestBlock;
    target.site = AssertSite::HOIST;
  }

  return target;
}

static void collectDefChain(Value root, Region *boundary,
                            SmallVectorImpl<Operation *> &opsToMove) {
  SmallVector<Value> worklist;
  DenseSet<Operation *> visited;
  worklist.push_back(root);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    auto *defOp = v.getDefiningOp();
    if (!defOp) continue;
    if (!boundary->isAncestor(defOp->getParentRegion())) continue;
    if (!visited.insert(defOp).second) continue;
    opsToMove.push_back(defOp);
    for (Value operand : defOp->getOperands()) worklist.push_back(operand);
  }
}

/// Move (or clone) ops to destBlock before destPoint, in topological order.
/// Ops that have users outside the move set are cloned instead of moved,
/// so that existing references remain valid.
static void moveOpsBeforeInTopoOrder(SmallVectorImpl<Operation *> &ops,
                                     Block *destBlock,
                                     Block::iterator destPoint) {
  DenseSet<Operation *> opSet(ops.begin(), ops.end());
  SmallVector<Operation *> sorted;
  DenseSet<Operation *> emitted;

  std::function<void(Operation *)> visit = [&](Operation *op) {
    if (!opSet.count(op) || emitted.count(op)) return;
    for (Value operand : op->getOperands())
      if (auto *defOp = operand.getDefiningOp()) visit(defOp);
    sorted.push_back(op);
    emitted.insert(op);
  };

  for (auto *op : ops) visit(op);
  for (auto *op : sorted) {
    bool hasExternalUsers = llvm::any_of(
        op->getUsers(), [&](Operation *u) { return !opSet.count(u); });
    if (hasExternalUsers) {
      OpBuilder b(destBlock, destPoint);
      auto *clone = b.clone(*op);
      for (auto &use : llvm::make_early_inc_range(op->getUses()))
        if (opSet.count(use.getOwner()))
          use.set(clone->getResult(0));
    } else {
      op->moveBefore(destBlock, destPoint);
    }
  }
}

struct HoistAssertionsPass
    : public ::coir::impl::HoistAssertionsBase<HoistAssertionsPass> {
  using HoistAssertionsBase::HoistAssertionsBase;

  void runOnOperation() override {
    SmallVector<AssertOp> asserts;
    getOperation()->walk([&](AssertOp op) { asserts.push_back(op); });

    // Fold away assertions whose predicates are statically evaluable.
    // Covers both direct constant-true and cmpi on two constants.
    for (auto it = asserts.begin(); it != asserts.end();) {
      auto assertOp = *it;
      Value cond = assertOp.getCondition();
      std::optional<bool> staticVal;

      if (auto constOp = cond.getDefiningOp<arith::ConstantOp>()) {
        if (auto bv = dyn_cast<IntegerAttr>(constOp.getValue()))
          staticVal = bv.getValue().getBoolValue();
      } else if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>()) {
        APInt lhs, rhs;
        if (matchPattern(cmpOp.getLhs(), mlir::m_ConstantInt(&lhs)) &&
            matchPattern(cmpOp.getRhs(), mlir::m_ConstantInt(&rhs)))
          staticVal = arith::applyCmpPredicate(cmpOp.getPredicate(), lhs, rhs);
      }

      if (staticVal && *staticVal) {
        assertOp.erase();
        it = asserts.erase(it);
        continue;
      }
      ++it;
    }

    for (auto assertOp : asserts) {
      if (assertOp.getSite() == AssertSite::ENTRY) continue;

      auto kernel = assertOp->getParentOfType<KernelOp>();
      if (!kernel) continue;

      auto target = computeHoistTarget(assertOp, kernel);

      LLVM_DEBUG(llvm::dbgs() << "HOIST: msg=\""
                              << assertOp.getMessage() << "\" -> site="
                              << (target.site == AssertSite::ENTRY ? "ENTRY"
                                  : target.site == AssertSite::HOIST ? "HOIST"
                                  : "USE") << "\n");

      if (target.site == AssertSite::ENTRY) {
        assertOp.setSiteAttr(
            AssertSiteAttr::get(assertOp.getContext(), AssertSite::ENTRY));
        auto &entryBlock = kernel.getBody().front();
        SmallVector<Operation *> chain;
        collectDefChain(assertOp.getCondition(), &kernel.getBody(), chain);
        chain.push_back(assertOp);
        moveOpsBeforeInTopoOrder(chain, &entryBlock, entryBlock.begin());
      } else if (target.site == AssertSite::HOIST && target.insertBefore) {
        assertOp.setSiteAttr(
            AssertSiteAttr::get(assertOp.getContext(), AssertSite::HOIST));
        SmallVector<Operation *> chain;
        collectDefChain(assertOp.getCondition(), &kernel.getBody(), chain);
        chain.push_back(assertOp);
        moveOpsBeforeInTopoOrder(chain, target.destBlock,
                                 target.insertBefore->getIterator());
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
