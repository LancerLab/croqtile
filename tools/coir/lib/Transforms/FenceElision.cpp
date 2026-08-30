//===- FenceElision.cpp - Drop redundant fences ---------------------------===//
//
// Drops a coir.fence that is made redundant by an earlier, stronger fence.
// A fence orders the issuing agent's prior accesses; a later fence publishes
// nothing new when a stronger-or-equal fence already executed on every path
// to it with no intervening memory effect that the later fence would order.
//
// Redundancy is decided by a four-axis subsumption lattice over
// (space, entity, order, scope):
//
//  - order:  seq_cst >= acq_rel >= {release, acquire} (release/acquire are
//            incomparable); equal always subsumes.
//  - entity: all >= {threads, dma, tma, mma}; a threads (generic proxy) fence
//            never subsumes an engine (async proxy) fence and vice versa.
//  - scope:  coarser subsumes finer (device > cluster > block > groupx4 >
//            group > thread); none (auto) matches only itself.
//  - space:  equal-only (TODO: global >= shared >= local for thread fences,
//            pending per-target confirmation).
//
// Availability is a forward must-analysis over each region's block CFG:
//  - state = set of available fence guarantees, kept subsumption-maximal;
//  - a fence adds itself (dropping entries it subsumes);
//  - memory-effect-free ops and fences are transparent (fences publish
//    nothing new, so they neither kill nor end the search);
//  - an op with a Write effect kills guarantees with a release component
//    (new unpublished data), an op with a Read effect kills guarantees with
//    an acquire component (the read may itself need the later fence's
//    ordering against subsequent reads); ops without memory-effect
//    information (waits, barriers, ...) kill everything;
//  - the pre-op state propagates into the entry blocks of nested non-loop
//    regions (an scf.if branch executes, when it executes, in the pre-if
//    state); loop bodies get an empty entry state because the pre-loop state
//    does not hold at region entry on later iterations;
//  - block entry state = meet (path intersection) over predecessors.
//
// Batch removal is sound: justification is transitive (C >= A >= B with
// kill-free paths implies C >= B), so a fence justified by another fence
// that is itself removed remains justified by the root of the chain.
//
// The pass runs unconditionally in the lowering pipeline (even at -O0),
// because fences are correctness annotations that lower to hardware barrier
// instructions; they are not optional optimizations.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace coir {
#define GEN_PASS_DECL_FENCEELISION
#define GEN_PASS_DEF_FENCEELISION
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

/// One available fence guarantee: a fence of this kind executed on every path
/// to the current program point, with no intervening kill.
struct FencePoint {
  TensorMemorySpace space;
  FenceEntity entity;
  FenceOrder order;
  ParallelLevel scope;

  bool operator==(const FencePoint& o) const {
    return space == o.space && entity == o.entity && order == o.order &&
           scope == o.scope;
  }
};

static bool orderSubsumes(FenceOrder a, FenceOrder b) {
  if (a == b) return true;
  if (a == FenceOrder::SeqCst) return true;
  if (a == FenceOrder::AcqRel)
    return b == FenceOrder::Release || b == FenceOrder::Acquire;
  return false;
}

static bool scopeSubsumes(ParallelLevel a, ParallelLevel b) {
  if (a == b) return true;
  // Coarseness rank; NONE (auto) and SEQ are outside the lattice and match
  // only themselves (handled above).
  auto rank = [](ParallelLevel s) -> int {
    switch (s) {
    case ParallelLevel::THREAD: return 0;
    case ParallelLevel::GROUP: return 1;
    case ParallelLevel::GROUPx4: return 2;
    case ParallelLevel::BLOCK: return 3;
    case ParallelLevel::CLUSTER: return 4;
    case ParallelLevel::DEVICE: return 5;
    default: return -1;
    }
  };
  int ra = rank(a), rb = rank(b);
  return ra >= 0 && rb >= 0 && ra > rb;
}

/// Fence `a` makes a later fence `b` redundant.
static bool subsumes(const FencePoint& a, const FencePoint& b) {
  if (a.space != b.space) return false;
  if (a.entity != b.entity && a.entity != FenceEntity::All) return false;
  return orderSubsumes(a.order, b.order) && scopeSubsumes(a.scope, b.scope);
}

static bool hasReleaseComponent(FenceOrder o) {
  return o == FenceOrder::Release || o == FenceOrder::AcqRel ||
         o == FenceOrder::SeqCst;
}

static bool hasAcquireComponent(FenceOrder o) {
  return o == FenceOrder::Acquire || o == FenceOrder::AcqRel ||
         o == FenceOrder::SeqCst;
}

/// The set of fence guarantees available at a program point, kept
/// subsumption-maximal.
struct FenceState {
  llvm::SmallVector<FencePoint, 4> points;

  bool covers(const FencePoint& p) const {
    return llvm::any_of(points,
                        [&](const FencePoint& q) { return subsumes(q, p); });
  }

  void add(const FencePoint& p) {
    if (covers(p)) return;
    llvm::erase_if(points, [&](const FencePoint& q) { return subsumes(p, q); });
    points.push_back(p);
  }

  void clear() { points.clear(); }

  bool operator==(const FenceState& o) const {
    if (points.size() != o.points.size()) return false;
    for (const FencePoint& p : points)
      if (!llvm::is_contained(o.points, p)) return false;
    return true;
  }

  /// Must-analysis meet: a guarantee holds at a join only if every
  /// predecessor provides a guarantee that subsumes it.
  static FenceState meet(const FenceState& a, const FenceState& b) {
    FenceState r;
    for (const FencePoint& p : a.points)
      if (b.covers(p)) r.add(p);
    for (const FencePoint& p : b.points)
      if (a.covers(p)) r.add(p);
    return r;
  }
};

struct FenceElisionPass
    : public ::coir::impl::FenceElisionBase<FenceElisionPass> {
  using FenceElisionBase::FenceElisionBase;

  void runOnOperation() override {
    for (Region& region : getOperation()->getRegions())
      processRegion(region, /*entry=*/std::nullopt, /*collect=*/true);
    for (FenceOp fence : redundant_) fence->erase();
    redundant_.clear();
  }

private:
  llvm::SmallVector<FenceOp, 8> redundant_;

  /// Invalidate guarantees that `op`'s memory effects break. Fences and
  /// effect-free ops kill nothing; ops without effect information (waits,
  /// barriers) conservatively kill everything.
  static void applyKill(Operation* op, FenceState& state) {
    if (isa<FenceOp>(op) || mlir::isMemoryEffectFree(op)) return;
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface) {
      state.clear();
      return;
    }
    bool writes = iface.hasEffect<MemoryEffects::Write>();
    bool reads = iface.hasEffect<MemoryEffects::Read>();
    if (!writes && !reads) {
      state.clear();
      return;
    }
    llvm::erase_if(state.points, [&](const FencePoint& p) {
      return (writes && hasReleaseComponent(p.order)) ||
             (reads && hasAcquireComponent(p.order));
    });
  }

  /// Walk one block, threading `state` through its ops. When `collect` is
  /// set, fences covered by the state are recorded as redundant and nested
  /// regions are processed recursively.
  FenceState transferBlock(Block* block, FenceState state, bool collect) {
    for (Operation& op : block->getOperations()) {
      if (auto fence = dyn_cast<FenceOp>(&op)) {
        FencePoint p{fence.getSpace(), fence.getEntity(), fence.getOrder(),
                     fence.getScope()};
        if (collect && state.covers(p)) redundant_.push_back(fence);
        // Even a redundant fence still provides its guarantee until erased;
        // adding it is harmless (add() drops points it subsumes).
        state.add(p);
        continue;
      }
      // Recurse into nested regions before applying the op's own kill, so the
      // pre-op state is visible at region entry. Loop bodies re-execute, so
      // the pre-loop state does not hold at region entry on later
      // iterations: they get an empty entry state (no back-edge reasoning).
      if (collect && op.getNumRegions() > 0) {
        bool isLoop = isa<LoopLikeOpInterface>(&op);
        for (Region& region : op.getRegions())
          processRegion(
              region, isLoop ? std::nullopt : std::optional<FenceState>(state),
              collect);
      }
      applyKill(&op, state);
    }
    return state;
  }

  /// Forward available-fences must-analysis over the region's block CFG,
  /// followed (when `collect` is set) by a redundancy-collection walk.
  void processRegion(Region& region, std::optional<FenceState> entry,
                     bool collect) {
    if (region.empty()) return;

    llvm::DenseMap<Block*, FenceState> inStates, outStates;
    llvm::SmallVector<Block*, 8> worklist;
    for (Block& b : region) worklist.push_back(&b);

    while (!worklist.empty()) {
      Block* b = worklist.pop_back_val();
      FenceState in;
      if (b->isEntryBlock()) {
        in = entry.value_or(FenceState{});
      } else {
        bool first = true;
        for (Block* pred : b->getPredecessors()) {
          auto it = outStates.find(pred);
          const FenceState& predOut =
              it != outStates.end() ? it->second : FenceState{};
          in = first ? predOut : FenceState::meet(in, predOut);
          first = false;
        }
        // Unreachable block (no predecessors): empty state.
      }
      inStates[b] = in;
      FenceState out = transferBlock(b, in, /*collect=*/false);
      auto it = outStates.find(b);
      if (it == outStates.end() || !(it->second == out)) {
        outStates[b] = std::move(out);
        for (Block* succ : b->getSuccessors()) worklist.push_back(succ);
      }
    }

    if (!collect) return;
    for (Block& b : region) transferBlock(&b, inStates[&b], /*collect=*/true);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createFenceElisionPass() {
  return std::make_unique<FenceElisionPass>();
}
} // namespace coir
