//===- ConvertToGCU.cpp - Lower CoIR ops to GCU-compatible GPU dialect ----===//
//
// Converts CoIR kernel operations into the MLIR GPU dialect and standard
// dialects (memref, arith, scf) in a form suitable for consumption by the
// external GCU compiler (gcu-compiler-opt).
//
// This pass is similar to ConvertToGPU (NVPTX path) but uses GCU address
// spaces and maps DMA copies to memref.dma_start/dma_wait instead of flat
// copy loops, since the GCU compiler pipeline handles DTE lowering.
//
// Handled ops:
//   coir.kernel       -> gpu.module + gpu.func (kernel)
//   coir.parallel     -> gpu.thread_id / gpu.block_id
//   coir.foreach      -> scf.for
//   coir.tensor.alloc -> memref.alloc (with GCU address spaces)
//   coir.tensor.load_elem / store_elem -> memref.load / store
//   coir.tensor.reduce_elem -> memref.load + arith.add + memref.store
//   coir.element.copy -> flat memcpy loop
//   coir.dma.copy     -> memref.dma_start + memref.dma_wait
//   coir.tma.copy     -> memref.dma_start + memref.dma_wait
//   coir.dma.invoke   -> gpu.barrier (sync point)
//   coir.dma.const.desc / prefetch.desc / runtime.desc -> skipped
//   coir.barrier      -> gpu.barrier
//   coir.wait         -> gpu.barrier
//   coir.assert       -> skipped
//   coir.return       -> gpu.return
//   coir.mma.*        -> not yet supported (error)
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"

#include "CodeGen/CoIRKernelLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace coir;
namespace mgpu = mlir::gpu;

namespace {

// GCU address space constants (numeric values matching GCU LLVM backend).
// These are what kurama's -convert-memref-to-gcu expects as input.
// 0 = generic/local, 1 = global, 2 = workgroup (shared), 5 = private.
constexpr int64_t kGCUAddrGlobal = 1;
constexpr int64_t kGCUAddrWorkgroup = 2;

struct ConvertToGCUPass : public mlir::OperationPass<mlir::ModuleOp>,
                          public coir::CoIRKernelLoweringBase {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToGCUPass)
  ConvertToGCUPass() : OperationPass(mlir::TypeID::get<ConvertToGCUPass>()) {}
  llvm::StringRef getName() const override { return "ConvertToGCU"; }
  llvm::StringRef getArgument() const override {
    return "coir-convert-to-gcu";
  }
  llvm::StringRef getDescription() const override {
    return "Lower CoIR kernel ops to GCU-compatible GPU/memref/arith dialects";
  }
  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertToGCUPass>();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mgpu::GPUDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect,
                    func::FuncDialect, LLVM::LLVMDialect>();
  }

  MemRefType convertTensorType(coir::TensorType tty) override {
    int32_t ms = tty.getMemorySpace();
    Attribute addrSpace;
    if (ms == 1)
      addrSpace = IntegerAttr::get(IntegerType::get(tty.getContext(), 64),
                                 kGCUAddrWorkgroup);
    else if (ms == 0)
      addrSpace = IntegerAttr::get(IntegerType::get(tty.getContext(), 64),
                                 kGCUAddrGlobal);
    return MemRefType::get(tty.getShape(), tty.getElementType(),
                           AffineMap{}, addrSpace);
  }

  bool convertTargetOp(OpBuilder &builder, Location loc, Operation &op,
                       KernelConvertCtx &ctx) override {
    if (auto tile = dyn_cast<TensorTileOp>(op)) {
      Value src = ctx.mapping.lookup(tile.getSource());
      ctx.mapping.map(tile.getResult(), src);
      return true;
    }
    if (auto storeTile = dyn_cast<TensorStoreTileOp>(op)) {
      Value src = ctx.mapping.lookup(storeTile.getTile());
      Value dst = ctx.mapping.lookup(storeTile.getDest());
      emitFlatCopyLoop(builder, loc, src, dst);
      return true;
    }
    if (auto tc = dyn_cast<TmaCopyOp>(op)) {
      emitFlatCopyLoop(builder, loc, ctx.mapping.lookup(tc.getSource()),
                       ctx.mapping.lookup(tc.getDest()));
      if (tc.getToken())
        ctx.mapping.map(tc.getToken(), ctx.mapping.lookup(tc.getDest()));
      return true;
    }
    if (isa<DMAInvokeOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return true;
    }
    // Lower coir.call to func.call for external device functions
    if (auto callOp = dyn_cast<coir::CallOp>(op)) {
      auto callee = callOp.getCallee().str();
      // Map operands through the convert context (tensor -> memref)
      SmallVector<Value> mappedOperands;
      SmallVector<Type> argTypes;
      for (auto operand : callOp.getOperands_()) {
        auto mapped = ctx.mapping.lookup(operand);
        mappedOperands.push_back(mapped);
        argTypes.push_back(mapped.getType());
      }

      // Emit func.func private declaration in the gpu.module
      // (must be in gpu.module, not top-level ModuleOp, because
      //  SymbolTable::lookupNearestSymbolFrom stops at gpu.module)
      auto gpuModule =
          builder.getInsertionBlock()->getParentOp()->getParentOfType<
              mgpu::GPUModuleOp>();
      if (!gpuModule) {
        op.emitError("coir.call not inside a gpu.module");
        return false;
      }

      // Check if symbol name is already taken (e.g. by gpu.func)
      if (auto *existingSym =
              mlir::SymbolTable::lookupSymbolIn(gpuModule, callee)) {
        if (!isa<func::FuncOp>(existingSym)) {
          op.emitError("coir.call callee '")
              << callee
              << "' conflicts with existing symbol of type '"
              << existingSym->getName().getStringRef() << "'";
          return false;
        }
      }

      auto existingFn = gpuModule.lookupSymbol<func::FuncOp>(callee);
      if (!existingFn) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(gpuModule.getBody());
        auto fnType = builder.getFunctionType(argTypes, TypeRange{});
        auto fnOp = builder.create<func::FuncOp>(loc, callee, fnType);
        fnOp.setPrivate();
      } else {
        // Verify arg types match the existing declaration
        auto existingArgTypes = existingFn.getArgumentTypes();
        if (existingArgTypes.size() != argTypes.size()) {
          op.emitError("coir.call to '")
              << callee << "' has " << argTypes.size()
              << " operands, but existing declaration has "
              << existingArgTypes.size() << " arguments";
          return false;
        }
        // Note: full type compatibility check deferred to MLIR verifier
      }

      builder.create<func::CallOp>(loc, callee, TypeRange{}, mappedOperands);
      return true;
    }
    // MMA ops should have been caught in runOnOperation validation
    if (isa<MMAExecOp, MMALoadOp, MMAStoreOp, MMAFillOp>(op))
      return true;
    return false;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    bool hasMMA = false;
    module.walk([&](Operation *op) {
      if (isa<MMAExecOp, MMALoadOp, MMAStoreOp, MMAFillOp>(op))
        hasMMA = true;
    });
    if (hasMMA) {
      module.emitError("GCU native codegen does not yet support MMA ops; "
                       "use -t topscc for MMA kernels");
      return signalPassFailure();
    }

    module->setAttr("gpu.container_module",
                    UnitAttr::get(module.getContext()));

    SmallVector<KernelOp> kernels;
    module.walk([&](KernelOp k) { kernels.push_back(k); });
    for (auto kernel : kernels) {
      if (failed(convertKernel(module, kernel)))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createConvertToGCUPass() {
  return std::make_unique<ConvertToGCUPass>();
}
} // namespace coir
