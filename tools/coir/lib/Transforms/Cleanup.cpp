//===- Cleanup.cpp - Canonicalize + CSE cleanup of frontend IR ------------===//
//
// Runs the MLIR canonicalizer followed by CSE to remove semantically
// neutral noise emitted by the frontend:
//
//   - arith.index_cast(arith.constant) pairs fold into a single typed
//     constant (the frontend emits integer literals as i32 and then casts
//     them to index for coir.tensor.tile / coir.tensor.load_elem indices).
//   - Duplicate casts of the same value are merged (the frontend re-emits
//     the same cast at every use site).
//   - Identity index_cast round trips (i64 -> index -> i64) produced by
//     the DMA 2^32 size-check assertions collapse back to the original
//     constant.
//   - Trivially dead ops left behind are dropped.
//
// The pass does not hoist code across regions or change control structure.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "coir-cleanup"

namespace coir {
#define GEN_PASS_DECL_CLEANUP
#define GEN_PASS_DEF_CLEANUP
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;

namespace {

struct CleanupPass : public ::coir::impl::CleanupBase<CleanupPass> {
  using CleanupBase::CleanupBase;

  void runOnOperation() override {
    // Schedule canonicalize + CSE through the current pipeline executor.
    // (A nested PassManager::run() would re-initialize dialects and trip
    // the "appending to the dialect registry while in a multi-threaded
    // execution context" assertion.)
    OpPassManager pipeline;
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    if (failed(runPipeline(pipeline, getOperation())))
      signalPassFailure();
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createCleanupPass() {
  return std::make_unique<CleanupPass>();
}
} // namespace coir
