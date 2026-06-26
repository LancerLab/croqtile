//===- ConvertToGPU.cpp - Lower CoIR ops to GPU dialect -------------------===//
//
// Converts CoIR kernel operations into the MLIR GPU dialect and supporting
// standard dialects (memref, arith, scf). This is the first stage of the
// native GPU codegen pipeline: CoIR -> GPU -> NVVM -> LLVM -> PTX.
//
// This pass runs AFTER the shared CoIR lowering pipeline, so the IR
// contains lowered copy forms (element.copy, dma.copy, tma.copy) and
// DMA descriptor ops rather than raw data.copy.
//
// Handled ops:
//   coir.kernel       -> gpu.module + gpu.func (kernel)
//   coir.parallel     -> gpu.thread_id / gpu.block_id
//   coir.foreach      -> scf.for
//   coir.tensor.alloc -> memref.alloc (with GPU address space for shared)
//   coir.tensor.load_elem / store_elem -> memref.load / store
//   coir.tensor.reduce_elem -> memref.load + arith.add + memref.store
//   coir.element.copy -> flat memcpy loop (scf.for + load + store)
//   coir.dma.copy     -> flat memcpy loop + barrier
//   coir.tma.copy     -> flat memcpy loop + barrier
//   coir.dma.invoke   -> flat memcpy from descriptor source/dest + barrier
//   coir.dma.const.desc / prefetch.desc / runtime.desc -> skipped
//   coir.barrier      -> gpu.barrier
//   coir.wait         -> gpu.barrier (conservative)
//   coir.assert       -> skipped (handled at host level)
//   coir.return       -> gpu.return
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/NativePipeline.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace coir;
namespace mgpu = mlir::gpu;

namespace {

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
  // GPU address spaces: 0=global/generic, 1=shared (workgroup=3 in NVVM).
  Attribute addrSpace;
  if (ms == 1)
    addrSpace = IntegerAttr::get(IntegerType::get(tty.getContext(), 64), 3);
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

struct ConvertToGPUPass : public mlir::OperationPass<mlir::ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToGPUPass)
  ConvertToGPUPass() : OperationPass(mlir::TypeID::get<ConvertToGPUPass>()) {}
  llvm::StringRef getName() const override { return "ConvertToGPU"; }
  llvm::StringRef getArgument() const override {
    return "coir-convert-to-gpu";
  }
  llvm::StringRef getDescription() const override {
    return "Lower CoIR kernel ops to GPU/memref/arith dialects";
  }
  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<ConvertToGPUPass>();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mgpu::GPUDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
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

    // Pre-scan for shared-memory TensorAllocOps and create memref.global
    // ops in the gpu.module so they lower to static .shared PTX buffers
    // instead of runtime malloc.
    unsigned shmIdx = 0;
    DenseMap<Operation*, std::string> shmGlobalNames;
    DenseMap<Operation*, MemRefType> shmGlobalTypes;
    auto &kernelBody = kernel.getBody();
    kernelBody.walk([&](TensorAllocOp alloc) {
      auto tty = cast<coir::TensorType>(alloc.getResult().getType());
      if (tty.getMemorySpace() != 1) return;
      auto memTy = convertTensorType(tty);
      std::string globalName =
          ("__shm_" + symName + "_" + std::to_string(shmIdx++)).str();
      shmGlobalNames[alloc.getOperation()] = globalName;
      shmGlobalTypes[alloc.getOperation()] = memTy;
      builder.setInsertionPointToStart(gpuModule.getBody());
      builder.create<memref::GlobalOp>(
          loc, globalName,
          builder.getStringAttr("private"),
          memTy, Attribute(), /*constant=*/false, IntegerAttr());
    });

    builder.setInsertionPointToEnd(gpuModule.getBody());
    auto gpuFuncType =
        builder.getFunctionType(gpuArgTypes, TypeRange{});
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
      convertOp(builder, loc, op, mapping, returnAllocMap,
                shmGlobalNames, shmGlobalTypes);

    if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>())
      builder.create<mgpu::ReturnOp>(loc);

    kernel.erase();
    return success();
  }

  void convertOp(OpBuilder &builder, Location loc, Operation &op,
                 IRMapping &mapping,
                 DenseMap<Value, Value> &returnAllocMap,
                 DenseMap<Operation*, std::string> &shmGlobalNames,
                 DenseMap<Operation*, MemRefType> &shmGlobalTypes) {
    if (isa<KernelReturnOp>(op)) {
      builder.create<mgpu::ReturnOp>(loc);
      return;
    }
    if (auto alloc = dyn_cast<TensorAllocOp>(op)) {
      convertAlloc(builder, loc, alloc, mapping, returnAllocMap,
                   shmGlobalNames, shmGlobalTypes);
      return;
    }
    if (auto par = dyn_cast<ParallelOp>(op)) {
      convertParallel(builder, loc, par, mapping, returnAllocMap,
                      shmGlobalNames, shmGlobalTypes);
      return;
    }
    if (auto fe = dyn_cast<ForeachOp>(op)) {
      convertForeach(builder, loc, fe, mapping, returnAllocMap,
                     shmGlobalNames, shmGlobalTypes);
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

    // --- Copy ops (after shared lowering) ---
    if (auto ec = dyn_cast<ElementCopyOp>(op)) {
      convertElementCopy(builder, loc, ec, mapping);
      return;
    }
    if (auto dc = dyn_cast<DmaCopyOp>(op)) {
      convertDmaCopy(builder, loc, dc, mapping);
      return;
    }
    if (auto tc = dyn_cast<TmaCopyOp>(op)) {
      convertTmaCopy(builder, loc, tc, mapping);
      return;
    }

    // --- DMA descriptor pipeline ops (from LowerDMADesc) ---
    // The descriptor infrastructure targets hardware DMA engines.
    // On basic NVPTX we handle the invoke as a flat copy; the
    // descriptor-building ops are no-ops.
    if (isa<DMAConstDescOp>(op) || isa<DMADescPrefetchOp>(op) ||
        isa<DMADescRuntimeOp>(op) || isa<DMACheckOp>(op))
      return;
    if (auto invoke = dyn_cast<DMAInvokeOp>(op)) {
      convertDMAInvoke(builder, loc, invoke, mapping);
      return;
    }

    // --- Synchronization ---
    if (auto bar = dyn_cast<BarrierOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return;
    }
    if (isa<WaitOp>(op)) {
      builder.create<mgpu::BarrierOp>(loc);
      return;
    }

    // --- Assertions (handled at host level, skip in device code) ---
    if (isa<AssertOp>(op))
      return;

    // --- DataCopyOp (pre-lowering fallback, should not appear after
    // shared Lower() but kept for robustness) ---
    if (auto dc = dyn_cast<DataCopyOp>(op)) {
      convertDataCopyFallback(builder, loc, dc, mapping);
      return;
    }

    if (isa<YieldOp>(op))
      return;
    builder.clone(op, mapping);
  }

  void convertAlloc(OpBuilder &builder, Location loc, TensorAllocOp alloc,
                    IRMapping &mapping,
                    DenseMap<Value, Value> &returnAllocMap,
                    DenseMap<Operation*, std::string> &shmGlobalNames,
                    DenseMap<Operation*, MemRefType> &shmGlobalTypes) {
    auto tty = cast<coir::TensorType>(alloc.getResult().getType());
    auto it = returnAllocMap.find(alloc.getResult());
    if (it != returnAllocMap.end()) {
      mapping.map(alloc.getResult(), it->second);
      return;
    }
    auto shmIt = shmGlobalNames.find(alloc.getOperation());
    if (shmIt != shmGlobalNames.end()) {
      auto memTy = shmGlobalTypes[alloc.getOperation()];
      auto ref = builder.create<memref::GetGlobalOp>(loc, memTy,
                                                     shmIt->second);
      mapping.map(alloc.getResult(), ref.getResult());
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

  // element.copy -> flat scf.for loop of memref load/store.
  void convertElementCopy(OpBuilder &builder, Location loc,
                          ElementCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
  }

  // dma.copy -> flat copy + barrier (basic GPU fallback).
  void convertDmaCopy(OpBuilder &builder, Location loc,
                      DmaCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
    builder.create<mgpu::BarrierOp>(loc);
    if (copyOp.getToken())
      mapping.map(copyOp.getToken(), dst);
  }

  // tma.copy -> flat copy + barrier (basic GPU fallback).
  void convertTmaCopy(OpBuilder &builder, Location loc,
                      TmaCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
    builder.create<mgpu::BarrierOp>(loc);
    if (copyOp.getToken())
      mapping.map(copyOp.getToken(), dst);
  }

  // dma.invoke -> trace descriptor chain to find src/dst, emit flat copy.
  void convertDMAInvoke(OpBuilder &builder, Location loc,
                        DMAInvokeOp invoke, IRMapping &mapping) {
    // Walk use-def: invoke.desc -> prefetch.in -> const.desc
    Value desc = invoke.getDesc();
    DMAConstDescOp constDesc = nullptr;

    if (auto prefetch = desc.getDefiningOp<DMADescPrefetchOp>())
      constDesc = prefetch.getIn().getDefiningOp<DMAConstDescOp>();
    else if (auto rt = desc.getDefiningOp<DMADescRuntimeOp>()) {
      auto prefetch2 = rt.getIn().getDefiningOp<DMADescPrefetchOp>();
      if (prefetch2)
        constDesc = prefetch2.getIn().getDefiningOp<DMAConstDescOp>();
    }

    if (constDesc) {
      Value src = mapping.lookup(constDesc.getSource());
      Value dst = mapping.lookup(constDesc.getDest());
      emitFlatCopyLoop(builder, loc, src, dst);
    }
    builder.create<mgpu::BarrierOp>(loc);
  }

  // Pre-lowering fallback for data.copy (should not appear after shared
  // Lower() but kept for robustness).
  void convertDataCopyFallback(OpBuilder &builder, Location loc,
                               DataCopyOp copyOp, IRMapping &mapping) {
    Value src = mapping.lookup(copyOp.getSource());
    Value dst = mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
    builder.create<mgpu::BarrierOp>(loc);
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
                      DenseMap<Value, Value> &returnAllocMap,
                      DenseMap<Operation*, std::string> &shmGlobalNames,
                      DenseMap<Operation*, MemRefType> &shmGlobalTypes) {
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
      Block *loopBody = loop.getBody();

      // If ForOp created an implicit yield terminator, erase it.
      // We will emit the correct terminator from the coir.yield.
      if (loopBody->mightHaveTerminator())
        loopBody->getTerminator()->erase();

      builder.setInsertionPointToEnd(loopBody);

      mapping.map(args[0], loop.getInductionVar());

      for (unsigned i = 0; i < initVals.size(); ++i)
        mapping.map(args[i + 1], loop.getRegionIterArg(i));

      for (auto &op : body.front().getOperations()) {
        if (auto yield = dyn_cast<YieldOp>(op)) {
          SmallVector<Value> yieldedVals;
          for (auto v : yield.getOperands())
            yieldedVals.push_back(mapping.lookup(v));
          builder.create<scf::YieldOp>(loc, yieldedVals);
          continue;
        }
        convertOp(builder, loc, op, mapping, returnAllocMap,
                   shmGlobalNames, shmGlobalTypes);
      }
    }

    for (unsigned i = 0; i < fe.getNumResults(); ++i)
      mapping.map(fe.getResult(i), loop.getResult(i));
  }

  void convertParallel(OpBuilder &builder, Location loc, ParallelOp par,
                       IRMapping &mapping,
                       DenseMap<Value, Value> &returnAllocMap,
                       DenseMap<Operation*, std::string> &shmGlobalNames,
                       DenseMap<Operation*, MemRefType> &shmGlobalTypes) {
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
      convertOp(builder, loc, op, mapping, returnAllocMap,
                shmGlobalNames, shmGlobalTypes);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createConvertToGPUPass() {
  return std::make_unique<ConvertToGPUPass>();
}
} // namespace coir
