//===- ConvertToGPU.cpp - Lower CoIR ops to GPU dialect -------------------===//
//
// Converts CoIR kernel operations into the MLIR GPU dialect and supporting
// standard dialects (memref, arith, scf). This is the first stage of the
// native GPU codegen pipeline: CoIR -> GPU -> NVVM -> LLVM -> PTX.
//
// This pass runs AFTER the shared CoIR lowering pipeline, so the IR
// contains lowered copy forms (element.copy, dma.copy, tma.copy) and
// DMA descriptor ops.
//
// Handled ops:
//   coir.kernel       -> gpu.module + gpu.func (kernel)
//   coir.parallel     -> gpu.thread_id / gpu.block_id
//   coir.foreach      -> scf.for
//   coir.tensor.tile  -> memref.subview
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
#include "CodeGen/CoIRKernelLowering.h"
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

struct ConvertToGPUPass : public mlir::OperationPass<mlir::ModuleOp>,
                          public coir::CoIRKernelLoweringBase {
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
      shmGlobalNames_.clear();
      shmGlobalTypes_.clear();
      if (failed(convertKernel(module, kernel)))
        return signalPassFailure();
    }
  }

  MemRefType convertTensorType(coir::TensorType tty) override {
    int32_t ms = tty.getMemorySpace();
    Attribute addrSpace;
    if (ms == 1)
      addrSpace = IntegerAttr::get(IntegerType::get(tty.getContext(), 64), 3);
    return MemRefType::get(tty.getShape(), tty.getElementType(),
                           AffineMap{}, addrSpace);
  }

  void preScanKernel(KernelOp kernel, OpBuilder &builder,
                     mgpu::GPUModuleOp gpuModule) override {
    Location loc = kernel.getLoc();
    StringRef symName = kernel.getSymName();
    unsigned shmIdx = 0;
    auto &kernelBody = kernel.getBody();
    kernelBody.walk([&](TensorAllocOp alloc) {
      auto tty = cast<coir::TensorType>(alloc.getResult().getType());
      if (tty.getMemorySpace() != 1) return;
      auto memTy = convertTensorType(tty);
      std::string globalName =
          ("__shm_" + symName + "_" + std::to_string(shmIdx++)).str();
      shmGlobalNames_[alloc.getOperation()] = globalName;
      shmGlobalTypes_[alloc.getOperation()] = memTy;
      builder.setInsertionPointToStart(gpuModule.getBody());
      builder.create<memref::GlobalOp>(
          loc, globalName,
          builder.getStringAttr("private"),
          memTy, Attribute(), /*constant=*/false, IntegerAttr());
    });
  }

  bool convertTargetOp(OpBuilder &builder, Location loc,
                       Operation &op, KernelConvertCtx &ctx) override {
    if (auto tile = dyn_cast<TensorTileOp>(op)) {
      convertTensorTile(builder, loc, tile, ctx.mapping);
      return true;
    }
    if (auto tc = dyn_cast<TmaCopyOp>(op)) {
      convertTmaCopy(builder, loc, tc, ctx);
      return true;
    }
    if (auto invoke = dyn_cast<DMAInvokeOp>(op)) {
      convertDMAInvoke(builder, loc, invoke, ctx);
      return true;
    }
    return false;
  }

  void convertAlloc(OpBuilder &builder, Location loc,
                    TensorAllocOp alloc, KernelConvertCtx &ctx) override {
    auto tty = cast<coir::TensorType>(alloc.getResult().getType());
    auto it = ctx.returnAllocMap.find(alloc.getResult());
    if (it != ctx.returnAllocMap.end()) {
      ctx.mapping.map(alloc.getResult(), it->second);
      return;
    }
    auto shmIt = shmGlobalNames_.find(alloc.getOperation());
    if (shmIt != shmGlobalNames_.end()) {
      auto memTy = shmGlobalTypes_[alloc.getOperation()];
      auto ref = builder.create<memref::GetGlobalOp>(loc, memTy, shmIt->second);
      ctx.mapping.map(alloc.getResult(), ref.getResult());
      return;
    }
    auto memTy = convertTensorType(tty);
    int32_t ms = tty.getMemorySpace();
    if (ms == 2) {
      auto newAlloc = builder.create<memref::AllocaOp>(loc, memTy);
      ctx.mapping.map(alloc.getResult(), newAlloc.getResult());
    } else {
      auto newAlloc = builder.create<memref::AllocOp>(loc, memTy);
      ctx.mapping.map(alloc.getResult(), newAlloc.getResult());
    }
  }

  void convertDmaCopy(OpBuilder &builder, Location loc,
                      DmaCopyOp copyOp, KernelConvertCtx &ctx) override {
    CoIRKernelLoweringBase::convertDmaCopy(builder, loc, copyOp, ctx);
    builder.create<mgpu::BarrierOp>(loc);
  }

  void emitFlatCopyLoop(OpBuilder &builder, Location loc,
                        Value src, Value dst) override {
    auto srcTy = cast<MemRefType>(src.getType());
    int64_t totalElems = 1;
    for (auto dim : srcTy.getShape())
      totalElems *= dim;

    Value srcOffset = extractOffset(builder, loc, src);
    Value dstOffset = extractOffset(builder, loc, dst);
    Value srcBase = getBaseMemRef(src);
    Value dstBase = getBaseMemRef(dst);

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value total = builder.create<arith::ConstantIndexOp>(loc, totalElems);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

    auto elemTy = srcTy.getElementType();
    auto srcBaseTy = cast<MemRefType>(srcBase.getType());
    auto dstBaseTy = cast<MemRefType>(dstBase.getType());

    auto makeReinterpret = [&](Value base, Value offset,
                               Attribute addrSpace) -> Value {
      bool hasDynOffset =
          !arith::ConstantIndexOp::isBuildableWith(
              builder.getIndexAttr(0), offset.getType()) ||
          (offset.getDefiningOp<arith::ConstantIndexOp>() &&
           offset.getDefiningOp<arith::ConstantIndexOp>().value() != 0) ||
          !offset.getDefiningOp<arith::ConstantIndexOp>();

      int64_t staticOffset = 0;
      if (auto cst = offset.getDefiningOp<arith::ConstantIndexOp>())
        staticOffset = cst.value();
      else
        hasDynOffset = true;

      MemRefType flatTy;
      if (hasDynOffset && staticOffset == 0) {
        auto layout = StridedLayoutAttr::get(builder.getContext(),
                                             ShapedType::kDynamic, {1});
        flatTy = MemRefType::get({totalElems}, elemTy, layout, addrSpace);
      } else if (staticOffset != 0) {
        auto layout = StridedLayoutAttr::get(builder.getContext(),
                                             staticOffset, {1});
        flatTy = MemRefType::get({totalElems}, elemTy, layout, addrSpace);
      } else {
        flatTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                 addrSpace);
      }
      return builder.create<memref::ReinterpretCastOp>(
          loc, flatTy, base, offset,
          ValueRange{total}, ValueRange{one});
    };

    auto flatSrc = makeReinterpret(srcBase, srcOffset,
                                   srcBaseTy.getMemorySpace());
    auto flatDst = makeReinterpret(dstBase, dstOffset,
                                   dstBaseTy.getMemorySpace());

    auto loop = builder.create<scf::ForOp>(loc, zero, total, one);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      Value iv = loop.getInductionVar();
      Value elem = builder.create<memref::LoadOp>(loc, flatSrc, iv);
      builder.create<memref::StoreOp>(loc, elem, flatDst, iv);
    }
  }

private:
  DenseMap<Operation*, std::string> shmGlobalNames_;
  DenseMap<Operation*, MemRefType> shmGlobalTypes_;

  void convertTensorTile(OpBuilder &builder, Location loc,
                         TensorTileOp tileOp, IRMapping &mapping) {
    auto srcTy = cast<coir::TensorType>(tileOp.getSource().getType());
    auto tileTy = cast<coir::TensorType>(tileOp.getResult().getType());
    Value src = mapping.lookup(tileOp.getSource());
    auto srcMemTy = cast<MemRefType>(src.getType());
    auto tileShape = tileTy.getShape();
    auto srcShape = srcTy.getShape();
    auto indices = tileOp.getIndices();

    if (indices.empty()) {
      mapping.map(tileOp.getResult(), src);
      return;
    }

    // Compute flat byte offset: sum(indices[i] * tileShape[i] * stride[i])
    // where stride[i] = product of srcShape[i+1..n-1].
    SmallVector<int64_t> srcStrides(srcShape.size());
    {
      int64_t s = 1;
      for (int i = (int)srcShape.size() - 1; i >= 0; --i) {
        srcStrides[i] = s;
        s *= srcShape[i];
      }
    }

    Value offset = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned i = 0; i < indices.size() && i < srcShape.size(); ++i) {
      Value idx = mapping.lookup(indices[i]);
      int64_t tileDim = (i < tileShape.size()) ? tileShape[i] : 1;
      int64_t elemStride = tileDim * srcStrides[i];
      Value stride = builder.create<arith::ConstantIndexOp>(loc, elemStride);
      Value term = builder.create<arith::MulIOp>(loc, idx, stride);
      offset = builder.create<arith::AddIOp>(loc, offset, term);
    }

    auto tileMemTy = convertTensorType(tileTy);
    int64_t totalTileElems = 1;
    for (auto d : tileShape)
      totalTileElems *= d;

    auto flatTy = MemRefType::get({totalTileElems}, srcMemTy.getElementType(),
                                  AffineMap{}, srcMemTy.getMemorySpace());
    Value totalSize =
        builder.create<arith::ConstantIndexOp>(loc, totalTileElems);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto cast = builder.create<memref::ReinterpretCastOp>(
        loc, flatTy, src, offset, ValueRange{totalSize}, ValueRange{one});
    mapping.map(tileOp.getResult(), cast.getResult());
  }

  void convertTmaCopy(OpBuilder &builder, Location loc,
                      TmaCopyOp copyOp, KernelConvertCtx &ctx) {
    Value src = ctx.mapping.lookup(copyOp.getSource());
    Value dst = ctx.mapping.lookup(copyOp.getDest());
    emitFlatCopyLoop(builder, loc, src, dst);
    builder.create<mgpu::BarrierOp>(loc);
    if (copyOp.getToken())
      ctx.mapping.map(copyOp.getToken(), dst);
  }

  void convertDMAInvoke(OpBuilder &builder, Location loc,
                        DMAInvokeOp invoke, KernelConvertCtx &ctx) {
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
      Value src = ctx.mapping.lookup(constDesc.getSource());
      Value dst = ctx.mapping.lookup(constDesc.getDest());
      emitFlatCopyLoop(builder, loc, src, dst);
    }
    builder.create<mgpu::BarrierOp>(loc);
  }

  Value extractOffset(OpBuilder &builder, Location loc, Value memref) {
    if (auto castOp =
            memref.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto staticOffsets = castOp.getStaticOffsets();
      if (!staticOffsets.empty() && staticOffsets[0] == ShapedType::kDynamic)
        return castOp.getOffsets()[0];
      if (!staticOffsets.empty() && staticOffsets[0] != 0)
        return builder.create<arith::ConstantIndexOp>(loc, staticOffsets[0]);
    }
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }

  Value getBaseMemRef(Value memref) {
    if (auto castOp =
            memref.getDefiningOp<memref::ReinterpretCastOp>())
      return castOp.getSource();
    return memref;
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createConvertToGPUPass() {
  return std::make_unique<ConvertToGPUPass>();
}
} // namespace coir
