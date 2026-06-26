//===- ClassifyCopies.cpp - Validate DMA/TMA copy ops against target ------===//
//
// Validates that coir.tma.copy ops are supported by the target (emits an
// error if not). DmaCopyOp is always valid since thread-cooperative copy
// is a form of DMA in GPU environments.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "coir-classify-copies"

namespace coir {
#define GEN_PASS_DECL_CLASSIFYCOPIES
#define GEN_PASS_DEF_CLASSIFYCOPIES
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

struct ClassifyCopiesPass
    : public ::coir::impl::ClassifyCopiesBase<ClassifyCopiesPass> {
  using ClassifyCopiesBase::ClassifyCopiesBase;

  void runOnOperation() override {
    bool hasTMA = false;

    auto module = dyn_cast<ModuleOp>(getOperation());
    if (!module)
      module = getOperation()->getParentOfType<ModuleOp>();
    if (module) {
      if (auto attr = module->getAttrOfType<BoolAttr>("coir.has_tma"))
        hasTMA = attr.getValue();
    }

    if (!hasTMA) {
      bool hasTmaCopyOps = false;
      getOperation()->walk([&](TmaCopyOp op) {
        op.emitOpError(
            "requires TMA support but target does not provide it; "
            "use dma.copy instead or target a TMA-capable architecture");
        hasTmaCopyOps = true;
      });
      if (hasTmaCopyOps)
        return signalPassFailure();
    }
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createClassifyCopiesPass() {
  return std::make_unique<ClassifyCopiesPass>();
}
} // namespace coir
