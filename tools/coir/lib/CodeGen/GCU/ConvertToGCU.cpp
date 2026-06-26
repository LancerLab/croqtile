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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace coir;
namespace mgpu = mlir::gpu;

namespace {

// GCU address space constants (numeric values matching GCU LLVM backend).
// These are what kurama's -convert-memref-to-gcu expects as input.
// 0 = generic/local, 1 = global, 2 = workgroup (shared), 5 = private.
constexpr int64_t kGCUAddrGlobal = 1;
constexpr int64_t kGCUAddrWorkgroup = 2;

static mgpu::Dimension dimFromIndex(unsigned i) {
  switch (i) {
  case 0: return mgpu::Dimension::x;
  case 1: return mgpu::Dimension::y;
  case 2: return mgpu::Dimension::z;
  default: return mgpu::Dimension::x;
  }
}

static MemRefType convertTensorType(coir::TensorType tty) {
  int32_t ms = tty.getMemorySpace();
  // CoIR memory spaces: 0=global, 1=shared, 2=local/register
  // Map to GCU numeric address spaces for memref types.
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

struct LaunchDims {
  SmallVector<int64_t, 3> gridDims = {1, 1, 1};
  SmallVector<int64_t, 3> blockDims = {1, 1, 1};
};

static LaunchDims collectLaunchDims(KernelOp kernel) {
  LaunchDims dims;
  kernel.getBody().walk([&](ParallelOp par) {
    auto bounds = par.getBounds();
    auto lvl = par.getLevel();
    SmallVector<int64_t, 3> *target = nullptr;
    if (lvl == coir::ParallelLevel::BLOCK)
      target = &dims.gridDims;
    else if (lvl == coir::ParallelLevel::THREAD)
      target = &dims.blockDims;
    if (!target) return;
    for (unsigned i = 0; i < bounds.size() && i < 3; ++i)
      (*target)[i] = bounds[i];
  });
  return dims;
}

struct ConvertToGCUPass : public mlir::OperationPass<mlir::ModuleOp> {
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
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Check for unsupported MMA ops early.
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

    // Set gpu.container_module attribute for kurama compatibility.
    module->setAttr("gpu.container_module",
                    UnitAttr::get(module.getContext()));

    SmallVector<KernelOp> kernels;
    module.walk([&](KernelOp k) { kernels.push_back(k); });
    for (auto kernel : kernels) {
      if (failed(convertKernel(module, kernel)))
        return signalPassFailure();
    }
  }

private:
  LogicalResult convertKernel(ModuleOp module, KernelOp kernel) {
    OpBuilder builder(module.getContext());
    Location loc = kernel.getLoc();
    auto fnType = kernel.getFunctionType();
    StringRef symName = kernel.getSymName();

    LaunchDims launchDims = collectLaunchDims(kernel);

    SmallVector<Type> gpuArgTypes;
    for (auto inTy : fnType.getInputs()) {
      if (auto tty = dyn_cast<coir::TensorType>(inTy))
        gpuArgTypes.push_back(convertTensorType(tty));
      else
        gpuArgTypes.push_back(inTy);
    }
    for (auto resTy : fnType.getResults()) {
      if (auto tty = dyn_cast<coir::TensorType>(resTy))
        gpuArgTypes.push_back(convertTensorType(tty));
      else
        gpuArgTypes.push_back(resTy);
    }

    std::string moduleName = (symName + "_module").str();
    builder.setInsertionPoint(kernel);
    auto gpuModule = builder.create<mgpu::GPUModuleOp>(loc, moduleName);

    builder.setInsertionPointToStart(gpuModule.getBody());
    auto gpuFuncType = builder.getFunctionType(gpuArgTypes, TypeRange{});
    std::string kernelName = (symName + "_kernel").str();
    auto gpuFunc =
        builder.create<mgpu::GPUFuncOp>(loc, kernelName, gpuFuncType);
    gpuFunc->setAttr(mgpu::GPUDialect::getKernelFuncAttrName(),
                     builder.getUnitAttr());

    gpuModule->setAttr("coir.grid_dims",
                       builder.getDenseI64ArrayAttr(launchDims.gridDims));
    gpuModule->setAttr("coir.block_dims",
                       builder.getDenseI64ArrayAttr(launchDims.blockDims));

    Block &entry = gpuFunc.getBody().front();
    builder.setInsertionPointToStart(&entry);

    IRMapping mapping;
    auto &kernelBody = kernel.getBody();
    if (!kernelBody.empty()) {
      auto kernelArgs = kernelBody.getArguments();
      for (unsigned i = 0; i < kernelArgs.size(); ++i)
        mapping.map(kernelArgs[i], entry.getArgument(i));
    }

    unsigned outArgIdx = fnType.getNumInputs();
    DenseMap<Value, Value> returnAllocMap;
    for (auto &op : kernelBody.front().getOperations()) {
      if (auto ret = dyn_cast<KernelReturnOp>(op)) {
        for (unsigned i = 0; i < ret.getOperands().size(); ++i)
          returnAllocMap[ret.getOperands()[i]] =
              entry.getArgument(outArgIdx + i);
      }
    }

    for (auto &op : kernelBody.front().getOperations())
      convertOp(builder, loc, op, mapping, returnAllocMap);

    if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>())
      builder.create<mgpu::ReturnOp>(loc);

    kernel.erase();
    return success();
  }

  void convertOp(OpBuilder &builder, Location loc, Operation &op,
                 IRMapping &mapping,
                 DenseMap<Value, Value> &returnAllocMap) {
    if (isa<KernelReturnOp>(op)) {
      builder.create<mgpu::ReturnOp>(loc);
      return;
    }
    if (auto alloc = dyn_cast<TensorAllocOp>(op)) {
      convertAlloc(builder, loc, alloc, mapping, returnAllocMap);
      return;
    }
    if (auto par = dyn_cast<ParallelOp>(op)) {
      convertParallel(builder, loc, par, mapping, returnAllocMap);
      return;
    }
    if (auto fe = dyn_cast<ForeachOp>(op)) {
      convertForeach(builder, loc, fe, mapping, returnAllocMap);
      return;
    }
    if (auto tile = dyn_cast<TensorTileOp>(op)) {
      // For now, tile is transparent: maps to the source memref.
      // This is correct for single-chunk kernels (parallel by 1).
      Value src = mapping.lookup(tile.getSource());
      mapping.map(tile.getResult(), src);
      return;
    }
    if (auto storeTile = dyn_cast<TensorStoreTileOp>(op)) {
      Value src = mapping.lookup(storeTile.getTile());
      Value dst = mapping.lookup(storeTile.getDest());
      emitFlatCopyLoop(builder, loc, src, dst);
      return;
    }
    if (auto le = dyn_cast<TensorLoadElemOp>(op)) {
      convertLoadElem(builder, loc, le, mapping);
      return;
    }
    if (auto se = dyn_cast<TensorStoreElemOp>(op)) {
      convertStoreElem(builder, loc, se, mapping);
      return;
    }
    if (auto re = dyn_cast<TensorReduceElemOp>(op)) {
      convertReduceElem(builder, loc, re, mapping);
      return;
    }

    // --- Copy ops ---
    if (auto ec = dyn_cast<ElementCopyOp>(op)) {
      convertElementCopy(builder, loc, ec, mapping);
      return;
    }
    if (auto dc = dyn_cast<DmaCopyOp>(op)) {
      convertDmaCopy(builder, loc, dc, mapping);
      return;
    }
    if (auto tc = dyn_cast<TmaCopyOp>(op)) {
      emitFlatCopyLoop(builder, loc, mapping.lookup(tc.getSource()),
                       mapping.lookup(tc.getDest()));
      if (tc.getToken())
        mapping.map(tc.getToken(), mapping.lookup(tc.getDest()));
      return;
    }

    // DMA descriptor pipeline ops -- no-ops at this level.
    if (isa<DMAConstDescOp>(op) || isa<DMADescPrefetchOp>(op) ||
        isa<DMADescRuntimeOp>(op) || isa<DMACheckOp>(op))
      return;
    if (isa<DMAInvokeOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return;
    }

    // --- Synchronization ---
    if (isa<BarrierOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return;
    }
    if (isa<WaitOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return;
    }

    if (isa<AssertOp>(op))
      return;

    if (isa<YieldOp>(op))
      return;
    builder.clone(op, mapping);
  }

  void convertAlloc(OpBuilder &builder, Location loc, TensorAllocOp alloc,
                    IRMapping &mapping,
                    DenseMap<Value, Value> &returnAllocMap) {
    auto tty = cast<coir::TensorType>(alloc.getResult().getType());
    auto it = returnAllocMap.find(alloc.getResult());
    if (it != returnAllocMap.end()) {
      mapping.map(alloc.getResult(), it->second);
      return;
    }
    auto memTy = convertTensorType(tty);
    auto newAlloc = builder.create<memref::AllocOp>(loc, memTy);
    mapping.map(alloc.getResult(), newAlloc.getResult());
  }

  void convertLoadElem(OpBuilder &builder, Location loc,
                       TensorLoadElemOp loadElem, IRMapping &mapping) {
    Value src = mapping.lookup(loadElem.getSource());
    SmallVector<Value> indices;
    for (auto idx : loadElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    auto loaded = builder.create<memref::LoadOp>(loc, src, indices);
    mapping.map(loadElem.getResult(), loaded.getResult());
  }

  void convertStoreElem(OpBuilder &builder, Location loc,
                        TensorStoreElemOp storeElem, IRMapping &mapping) {
    Value dst = mapping.lookup(storeElem.getDest());
    Value val = mapping.lookup(storeElem.getValue());
    SmallVector<Value> indices;
    for (auto idx : storeElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    builder.create<memref::StoreOp>(loc, val, dst, indices);
  }

  void convertReduceElem(OpBuilder &builder, Location loc,
                         TensorReduceElemOp reduceElem, IRMapping &mapping) {
    Value dst = mapping.lookup(reduceElem.getDest());
    Value val = mapping.lookup(reduceElem.getValue());
    SmallVector<Value> indices;
    for (auto idx : reduceElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    Value old = builder.create<memref::LoadOp>(loc, dst, indices);
    Value sum;
    if (isa<FloatType>(val.getType()))
      sum = builder.create<arith::AddFOp>(loc, old, val);
    else
      sum = builder.create<arith::AddIOp>(loc, old, val);
    builder.create<memref::StoreOp>(loc, sum, dst, indices);
  }

  void convertElementCopy(OpBuilder &builder, Location loc,
                          ElementCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
  }

  // DMA copy -> flat copy loop (for now).
  // TODO: Once GCU address space annotations are properly emitted,
  // switch to memref.dma_start/dma_wait so kurama's -convert-memref-to-gcu
  // can lower them to DTE async ops.
  void convertDmaCopy(OpBuilder &builder, Location loc,
                      DmaCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
    if (copyOp.getToken())
      mapping.map(copyOp.getToken(), dst);
  }

  void emitFlatCopyLoop(OpBuilder &builder, Location loc, Value src,
                        Value dst) {
    auto srcTy = cast<MemRefType>(src.getType());
    int64_t totalElems = 1;
    for (auto dim : srcTy.getShape())
      totalElems *= dim;

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value total = builder.create<arith::ConstantIndexOp>(loc, totalElems);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

    auto elemTy = srcTy.getElementType();
    auto flatTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                  srcTy.getMemorySpace());
    auto dstTy = cast<MemRefType>(dst.getType());
    auto flatDstTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                     dstTy.getMemorySpace());

    auto flatSrc = builder.create<memref::ReinterpretCastOp>(
        loc, flatTy, src, /*offset=*/zero, /*sizes=*/ValueRange{total},
        /*strides=*/ValueRange{one});
    auto flatDst = builder.create<memref::ReinterpretCastOp>(
        loc, flatDstTy, dst, /*offset=*/zero, /*sizes=*/ValueRange{total},
        /*strides=*/ValueRange{one});

    auto loop = builder.create<scf::ForOp>(loc, zero, total, one);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value iv = loop.getInductionVar();
      Value elem = builder.create<memref::LoadOp>(loc, flatSrc, iv);
      builder.create<memref::StoreOp>(loc, elem, flatDst, iv);
    }
  }

  void convertForeach(OpBuilder &builder, Location loc, ForeachOp fe,
                      IRMapping &mapping,
                      DenseMap<Value, Value> &returnAllocMap) {
    auto &body = fe.getBody();
    if (body.empty()) return;
    auto args = body.front().getArguments();

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value ub = mapping.lookup(fe.getUpperBound());
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> initVals;
    for (auto init : fe.getIterArgs())
      initVals.push_back(mapping.lookup(init));

    auto loop = builder.create<scf::ForOp>(loc, zero, ub, step, initVals);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      mapping.map(args[0], loop.getInductionVar());
      for (unsigned i = 0; i < initVals.size(); ++i)
        mapping.map(args[i + 1], loop.getRegionIterArg(i));
      for (auto &op : body.front().getOperations()) {
        if (auto yield = dyn_cast<YieldOp>(op)) {
          if (!initVals.empty()) {
            // Replace the auto-generated empty yield with one carrying values.
            auto &block = *loop.getBody();
            if (auto *term = block.getTerminator())
              term->erase();
            SmallVector<Value> yieldedVals;
            for (auto v : yield.getOperands())
              yieldedVals.push_back(mapping.lookup(v));
            builder.create<scf::YieldOp>(loc, yieldedVals);
          }
          continue;
        }
        convertOp(builder, loc, op, mapping, returnAllocMap);
      }
    }
    for (unsigned i = 0; i < fe.getNumResults(); ++i)
      mapping.map(fe.getResult(i), loop.getResult(i));
  }

  void convertParallel(OpBuilder &builder, Location loc, ParallelOp par,
                       IRMapping &mapping,
                       DenseMap<Value, Value> &returnAllocMap) {
    auto lvl = par.getLevel();
    auto &body = par.getBody();
    if (body.empty()) return;

    auto args = body.getArguments();
    for (unsigned i = 0; i < args.size() && i < 3; ++i) {
      Value id;
      if (lvl == coir::ParallelLevel::THREAD)
        id = builder.create<mgpu::ThreadIdOp>(loc, dimFromIndex(i));
      else if (lvl == coir::ParallelLevel::BLOCK)
        id = builder.create<mgpu::BlockIdOp>(loc, dimFromIndex(i));
      else
        id = builder.create<arith::ConstantIndexOp>(loc, 0);
      mapping.map(args[i], id);
    }

    for (auto &op : body.front().getOperations())
      convertOp(builder, loc, op, mapping, returnAllocMap);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createConvertToGCUPass() {
  return std::make_unique<ConvertToGCUPass>();
}
} // namespace coir
