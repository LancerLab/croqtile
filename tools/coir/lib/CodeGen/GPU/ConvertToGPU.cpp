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
// Target-specific ops handled here (beyond CoIRKernelLoweringBase):
//   coir.tensor.tile  -> memref.reinterpret_cast (with source strides)
//   coir.tma.copy     -> flat/strided copy loop + barrier
//   coir.dma.invoke   -> flat memcpy from descriptor source/dest + barrier
//   coir.atomic       -> memref.atomic_rmw / generic_atomic_rmw
//   scf.if            -> scf.if (recursive conversion of body ops)
//   coir.mma.fill     -> gpu.subgroup_mma_constant_matrix
//   coir.mma.load     -> gpu.subgroup_mma_load_matrix
//   coir.mma.exec     -> gpu.subgroup_mma_compute
//   coir.mma.store    -> gpu.subgroup_mma_store_matrix
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
      adjustedBlockDims_.clear();
      if (failed(convertKernel(module, kernel)))
        return signalPassFailure();
      if (!adjustedBlockDims_.empty()) {
        module.walk([&](mgpu::GPUModuleOp gpuMod) {
          if (gpuMod->hasAttr("coir.block_dims")) {
            OpBuilder b(module.getContext());
            gpuMod->setAttr("coir.block_dims",
                            b.getDenseI64ArrayAttr(adjustedBlockDims_));
          }
        });
      }
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

    mmaFragRoles_.clear();
    mmaFragTranspose_.clear();
    mmaStoreShmNames_.clear();
    kernelBody.walk([&](MMAFillOp fill) {
      mmaFragRoles_[fill.getResult()] = "COp";
    });
    kernelBody.walk([&](MMAExecOp exec) {
      mmaFragRoles_[exec.getAccumulator()] = "COp";
      mmaFragRoles_[exec.getLhs()] = "AOp";
      mmaFragRoles_[exec.getRhs()] = "BOp";
      mmaFragRoles_[exec.getResult()] = "COp";
      auto layout = exec.getLayout();
      bool aTranspose = (layout == coir::MMALayout::ColCol ||
                         layout == coir::MMALayout::ColRow);
      bool bTranspose = (layout == coir::MMALayout::RowCol ||
                         layout == coir::MMALayout::ColCol);
      mmaFragTranspose_[exec.getLhs()] = aTranspose;
      mmaFragTranspose_[exec.getRhs()] = bTranspose;
    });
    // Compute warps per block, including GROUP-level parallelism.
    int64_t groupWarps = 0;
    bool hasThreadLevel = false;
    LaunchDims baseDims = collectLaunchDims(kernel);
    kernelBody.walk([&](ParallelOp par) {
      auto lvl = par.getLevel();
      if (lvl == ParallelLevel::GROUP ||
          lvl == ParallelLevel::GROUPx4) {
        int64_t nWarps = 1;
        for (auto b : par.getBounds()) nWarps *= b;
        if (lvl == ParallelLevel::GROUPx4) nWarps *= 4;
        groupWarps = nWarps;
      } else if (lvl == ParallelLevel::THREAD) {
        hasThreadLevel = true;
      }
    });
    if (groupWarps > 0 && hasThreadLevel) {
      int64_t tpg = 1;
      for (auto d : baseDims.blockDims) tpg *= d;
      warpsPerBlock_ = groupWarps;
      adjustedBlockDims_ = {groupWarps * tpg};
    } else if (groupWarps > 0) {
      warpsPerBlock_ = groupWarps;
      adjustedBlockDims_ = {groupWarps * 32};
    } else {
      int64_t threadsPerBlock = 1;
      for (auto d : baseDims.blockDims) threadsPerBlock *= d;
      warpsPerBlock_ = std::max<int64_t>(threadsPerBlock / 32, 1);
      adjustedBlockDims_ = {};
    }

    unsigned mmaShmIdx = 0;
    kernelBody.walk([&](MMAStoreOp store) {
      mmaFragRoles_[store.getFragment()] = "COp";
      auto fragTy = cast<coir::MMAFragType>(store.getFragment().getType());
      auto destTy = cast<coir::TensorType>(store.getDest().getType());
      if (fragTy.getElementType() != destTy.getElementType()) {
        auto addrSpace =
            IntegerAttr::get(IntegerType::get(loc.getContext(), 64), 3);
        auto fragShape = fragTy.getShape();
        int64_t elemsPerWarp = 1;
        for (auto d : fragShape) elemsPerWarp *= d;
        auto memTy = MemRefType::get(
            {warpsPerBlock_ * elemsPerWarp}, fragTy.getElementType(),
            AffineMap{}, addrSpace);
        std::string globalName =
            ("__mma_cvt_" + symName + "_" +
             std::to_string(mmaShmIdx++)).str();
        mmaStoreShmNames_[store.getOperation()] = globalName;
        mmaStoreShmElemsPerWarp_[store.getOperation()] = elemsPerWarp;
        builder.setInsertionPointToStart(gpuModule.getBody());
        builder.create<memref::GlobalOp>(
            loc, globalName,
            builder.getStringAttr("private"),
            memTy, Attribute(), /*constant=*/false, IntegerAttr());
      }
    });
  }

  bool convertTargetOp(OpBuilder &builder, Location loc,
                       Operation &op, KernelConvertCtx &ctx) override {
    if (auto par = dyn_cast<ParallelOp>(op)) {
      if (par.getLevel() == ParallelLevel::GROUP ||
          par.getLevel() == ParallelLevel::GROUPx4) {
        convertGroupParallel(builder, loc, par, ctx);
        return true;
      }
    }
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
    if (auto atomicOp = dyn_cast<AtomicOp>(op)) {
      convertAtomic(builder, loc, atomicOp, ctx.mapping);
      return true;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      convertScfIf(builder, loc, ifOp, ctx);
      return true;
    }
    if (auto fill = dyn_cast<MMAFillOp>(op)) {
      convertMMAFill(builder, loc, fill, ctx.mapping);
      return true;
    }
    if (auto load = dyn_cast<MMALoadOp>(op)) {
      convertMMALoad(builder, loc, load, ctx.mapping);
      return true;
    }
    if (auto exec = dyn_cast<MMAExecOp>(op)) {
      convertMMAExec(builder, loc, exec, ctx.mapping);
      return true;
    }
    if (auto store = dyn_cast<MMAStoreOp>(op)) {
      convertMMAStore(builder, loc, store, ctx.mapping);
      return true;
    }
    // Override base load/store/reduce to handle index count mismatches
    // from rank-changing tensor.tile.
    if (auto le = dyn_cast<TensorLoadElemOp>(op)) {
      gpuConvertLoadElem(builder, loc, le, ctx.mapping);
      return true;
    }
    if (auto se = dyn_cast<TensorStoreElemOp>(op)) {
      gpuConvertStoreElem(builder, loc, se, ctx.mapping);
      return true;
    }
    if (auto re = dyn_cast<TensorReduceElemOp>(op)) {
      gpuConvertReduceElem(builder, loc, re, ctx.mapping);
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
      auto ref =
          builder.create<memref::GetGlobalOp>(loc, memTy, shmIt->second);
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
    auto dstTy = cast<MemRefType>(dst.getType());
    int64_t totalElems = 1;
    for (auto dim : srcTy.getShape())
      totalElems *= dim;

    bool srcIsStridedTile = false, dstIsStridedTile = false;
    if (srcTy.getRank() > 1) {
      if (isa<StridedLayoutAttr>(srcTy.getLayout()))
        srcIsStridedTile = true;
      else if (src.getDefiningOp<memref::ReinterpretCastOp>())
        srcIsStridedTile = true;
    }
    if (dstTy.getRank() > 1) {
      if (isa<StridedLayoutAttr>(dstTy.getLayout()))
        dstIsStridedTile = true;
      else if (dst.getDefiningOp<memref::ReinterpretCastOp>())
        dstIsStridedTile = true;
    }

    if (!srcIsStridedTile && !dstIsStridedTile) {
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

      auto makeFlatView = [&](Value base, Value offset,
                              Attribute addrSpace) -> Value {
        bool hasDynOffset = false;
        int64_t staticOff = 0;
        if (auto cst = offset.getDefiningOp<arith::ConstantIndexOp>())
          staticOff = cst.value();
        else
          hasDynOffset = true;

        MemRefType flatTy;
        if (hasDynOffset) {
          auto layout = StridedLayoutAttr::get(builder.getContext(),
                                               ShapedType::kDynamic, {1});
          flatTy = MemRefType::get({totalElems}, elemTy, layout, addrSpace);
        } else if (staticOff != 0) {
          auto layout = StridedLayoutAttr::get(builder.getContext(),
                                               staticOff, {1});
          flatTy = MemRefType::get({totalElems}, elemTy, layout, addrSpace);
        } else {
          flatTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                   addrSpace);
        }
        return builder.create<memref::ReinterpretCastOp>(
            loc, flatTy, base, offset,
            ValueRange{total}, ValueRange{one});
      };

      auto flatSrc = makeFlatView(srcBase, srcOffset,
                                  srcBaseTy.getMemorySpace());
      auto flatDst = makeFlatView(dstBase, dstOffset,
                                  dstBaseTy.getMemorySpace());

      auto loop = builder.create<scf::ForOp>(loc, zero, total, one);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Value iv = loop.getInductionVar();
        Value elem = builder.create<memref::LoadOp>(loc, flatSrc, iv);
        builder.create<memref::StoreOp>(loc, elem, flatDst, iv);
      }
      return;
    }

    // At least one side is a strided multi-dimensional tile.
    auto shape = srcTy.getShape();
    unsigned rank = shape.size();

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> ivs;
    SmallVector<scf::ForOp> loops;

    for (unsigned d = 0; d < rank; ++d) {
      Value ub = builder.create<arith::ConstantIndexOp>(loc, shape[d]);
      auto loop = builder.create<scf::ForOp>(loc, zero, ub, one);
      loops.push_back(loop);
      ivs.push_back(loop.getInductionVar());
      builder.setInsertionPointToStart(loop.getBody());
    }

    Value elem = builder.create<memref::LoadOp>(loc, src, ivs);
    if (dstTy.getRank() == srcTy.getRank()) {
      builder.create<memref::StoreOp>(loc, elem, dst, ivs);
    } else {
      Value linearIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      for (unsigned d = 0; d < rank; ++d) {
        int64_t stride = 1;
        for (unsigned k = d + 1; k < rank; ++k) stride *= shape[k];
        Value s = builder.create<arith::ConstantIndexOp>(loc, stride);
        Value term = builder.create<arith::MulIOp>(loc, ivs[d], s);
        linearIdx = builder.create<arith::AddIOp>(loc, linearIdx, term);
      }
      dst = flattenIfNeeded(builder, loc, dst, 1);
      builder.create<memref::StoreOp>(loc, elem, dst, linearIdx);
    }

    if (!loops.empty())
      builder.setInsertionPointAfter(loops.front());
  }

private:
  DenseMap<Operation*, std::string> shmGlobalNames_;
  DenseMap<Operation*, MemRefType> shmGlobalTypes_;
  DenseMap<Value, StringRef> mmaFragRoles_;
  DenseMap<Value, bool> mmaFragTranspose_;
  DenseMap<Operation*, std::string> mmaStoreShmNames_;
  DenseMap<Operation*, int64_t> mmaStoreShmElemsPerWarp_;
  int64_t warpsPerBlock_ = 1;
  SmallVector<int64_t, 3> adjustedBlockDims_;

  Value flattenIfNeeded(OpBuilder &builder, Location loc, Value memref,
                        unsigned numIndices) {
    auto memTy = cast<MemRefType>(memref.getType());
    if ((unsigned)memTy.getRank() == numIndices)
      return memref;
    int64_t totalElems = 1;
    for (auto d : memTy.getShape()) totalElems *= d;
    Value base = getBaseMemRef(memref);
    Value offset = extractOffset(builder, loc, memref);

    bool hasDynOffset = false;
    int64_t staticOff = 0;
    if (auto cst = offset.getDefiningOp<arith::ConstantIndexOp>())
      staticOff = cst.value();
    else
      hasDynOffset = true;

    MemRefType flatTy;
    if (hasDynOffset) {
      auto layout = StridedLayoutAttr::get(builder.getContext(),
                                           ShapedType::kDynamic, {1});
      flatTy = MemRefType::get({totalElems}, memTy.getElementType(),
                               layout, memTy.getMemorySpace());
    } else if (staticOff != 0) {
      auto layout = StridedLayoutAttr::get(builder.getContext(),
                                           staticOff, {1});
      flatTy = MemRefType::get({totalElems}, memTy.getElementType(),
                               layout, memTy.getMemorySpace());
    } else {
      flatTy = MemRefType::get({totalElems}, memTy.getElementType(),
                               AffineMap{}, memTy.getMemorySpace());
    }

    Value sz = builder.create<arith::ConstantIndexOp>(loc, totalElems);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    return builder.create<memref::ReinterpretCastOp>(
        loc, flatTy, base, offset, ValueRange{sz}, ValueRange{one});
  }

  void gpuConvertLoadElem(OpBuilder &builder, Location loc,
                         TensorLoadElemOp loadElem, IRMapping &mapping) {
    Value src = mapping.lookup(loadElem.getSource());
    SmallVector<Value> indices;
    for (auto idx : loadElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    src = flattenIfNeeded(builder, loc, src, indices.size());
    auto loaded = builder.create<memref::LoadOp>(loc, src, indices);
    mapping.map(loadElem.getResult(), loaded.getResult());
  }

  void gpuConvertStoreElem(OpBuilder &builder, Location loc,
                           TensorStoreElemOp storeElem, IRMapping &mapping) {
    Value dst = mapping.lookup(storeElem.getDest());
    Value val = mapping.lookup(storeElem.getValue());
    SmallVector<Value> indices;
    for (auto idx : storeElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    dst = flattenIfNeeded(builder, loc, dst, indices.size());
    builder.create<memref::StoreOp>(loc, val, dst, indices);
  }

  void gpuConvertReduceElem(OpBuilder &builder, Location loc,
                            TensorReduceElemOp reduceElem,
                            IRMapping &mapping) {
    Value dst = mapping.lookup(reduceElem.getDest());
    Value val = mapping.lookup(reduceElem.getValue());
    SmallVector<Value> indices;
    for (auto idx : reduceElem.getIndices())
      indices.push_back(mapping.lookup(idx));
    dst = flattenIfNeeded(builder, loc, dst, indices.size());
    Value old = builder.create<memref::LoadOp>(loc, dst, indices);
    Value sum;
    if (isa<FloatType>(val.getType()))
      sum = builder.create<arith::AddFOp>(loc, old, val);
    else
      sum = builder.create<arith::AddIOp>(loc, old, val);
    builder.create<memref::StoreOp>(loc, sum, dst, indices);
  }

  mgpu::MMAMatrixType getMMAMatrixType(coir::MMAFragType fragTy,
                                      StringRef role) {
    return mgpu::MMAMatrixType::get(fragTy.getShape(),
                                    fragTy.getElementType(), role);
  }

  bool getFragTranspose(Value coirVal) {
    auto it = mmaFragTranspose_.find(coirVal);
    if (it != mmaFragTranspose_.end())
      return it->second;
    return false;
  }

  StringRef getFragRole(Value coirVal) {
    auto it = mmaFragRoles_.find(coirVal);
    if (it != mmaFragRoles_.end())
      return it->second;
    return "COp";
  }

  void convertMMAFill(OpBuilder &builder, Location loc, MMAFillOp fill,
                      IRMapping &mapping) {
    auto fragTy = cast<MMAFragType>(fill.getResult().getType());
    auto matTy = getMMAMatrixType(fragTy, "COp");
    Value val = mapping.lookup(fill.getValue());
    auto result =
        builder.create<mgpu::SubgroupMmaConstantMatrixOp>(loc, matTy, val);
    mapping.map(fill.getResult(), result.getResult());
  }

  void convertMMALoad(OpBuilder &builder, Location loc, MMALoadOp load,
                      IRMapping &mapping) {
    auto fragTy = cast<MMAFragType>(load.getResult().getType());
    StringRef role = getFragRole(load.getResult());
    auto matTy = getMMAMatrixType(fragTy, role);

    Value src = mapping.lookup(load.getSource());
    auto srcMemTy = cast<MemRefType>(src.getType());

    int64_t leadDim = srcMemTy.getShape().back();

    if (auto strided = dyn_cast<StridedLayoutAttr>(srcMemTy.getLayout())) {
      auto strides = strided.getStrides();
      if (!strides.empty() && strides[0] != ShapedType::kDynamic)
        leadDim = strides[0];
    } else if (auto castOp =
                   src.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto staticStrides = castOp.getStaticStrides();
      if (!staticStrides.empty() &&
          staticStrides[0] != ShapedType::kDynamic)
        leadDim = staticStrides[0];
    }

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto result = builder.create<mgpu::SubgroupMmaLoadMatrixOp>(
        loc, matTy, src, ValueRange{zero, zero},
        builder.getIndexAttr(leadDim),
        /*transpose=*/UnitAttr());
    mapping.map(load.getResult(), result.getResult());
  }

  void convertMMAExec(OpBuilder &builder, Location loc, MMAExecOp exec,
                      IRMapping &mapping) {
    Value acc = mapping.lookup(exec.getAccumulator());
    Value lhs = mapping.lookup(exec.getLhs());
    Value rhs = mapping.lookup(exec.getRhs());

    auto result = builder.create<mgpu::SubgroupMmaComputeOp>(
        loc, acc.getType(), lhs, rhs, acc,
        /*a_transpose=*/UnitAttr(), /*b_transpose=*/UnitAttr());
    mapping.map(exec.getResult(), result.getResult());
  }

  void convertMMAStore(OpBuilder &builder, Location loc, MMAStoreOp store,
                       IRMapping &mapping) {
    Value frag = mapping.lookup(store.getFragment());
    Value dst = mapping.lookup(store.getDest());
    auto dstMemTy = cast<MemRefType>(dst.getType());
    auto fragTy = cast<mgpu::MMAMatrixType>(frag.getType());

    bool needsConversion =
        fragTy.getElementType() != dstMemTy.getElementType();

    if (needsConversion) {
      auto fragShape = fragTy.getShape();
      auto shmNameIt = mmaStoreShmNames_.find(store.getOperation());
      assert(shmNameIt != mmaStoreShmNames_.end());
      int64_t elemsPerWarp = mmaStoreShmElemsPerWarp_[store.getOperation()];
      auto addrSpace =
          builder.getIntegerAttr(builder.getIntegerType(64), 3);

      // Get the full 1D shared buffer (sized for all warps).
      auto globalMemTy = MemRefType::get(
          {warpsPerBlock_ * elemsPerWarp}, fragTy.getElementType(),
          AffineMap{}, addrSpace);
      auto globalRef = builder.create<memref::GetGlobalOp>(
          loc, globalMemTy, shmNameIt->second);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

      // Compute warp index: threadIdx.x / 32
      Value tidX = builder.create<mgpu::ThreadIdOp>(
          loc, builder.getIndexType(), mgpu::Dimension::x);
      Value warpSize = builder.create<arith::ConstantIndexOp>(loc, 32);
      Value warpId = builder.create<arith::DivUIOp>(loc, tidX, warpSize);
      Value elemsPerWarpVal =
          builder.create<arith::ConstantIndexOp>(loc, elemsPerWarp);
      Value warpOffset =
          builder.create<arith::MulIOp>(loc, warpId, elemsPerWarpVal);

      // Store to 2D subview at warp's row offset.
      // Row index = warpId * fragShape[0], col index = 0.
      Value rowsPerWarp =
          builder.create<arith::ConstantIndexOp>(loc, fragShape[0]);
      Value rowOffset =
          builder.create<arith::MulIOp>(loc, warpId, rowsPerWarp);

      auto tmpMemTy = MemRefType::get(
          {warpsPerBlock_ * fragShape[0], fragShape[1]},
          fragTy.getElementType(), AffineMap{}, addrSpace);
      auto globalRef2D = builder.create<memref::ReinterpretCastOp>(
          loc, tmpMemTy, globalRef.getResult(), zero,
          ValueRange{
              builder.create<arith::ConstantIndexOp>(
                  loc, warpsPerBlock_ * fragShape[0]),
              builder.create<arith::ConstantIndexOp>(loc, fragShape[1])},
          ValueRange{
              builder.create<arith::ConstantIndexOp>(loc, fragShape[1]),
              builder.create<arith::ConstantIndexOp>(loc, 1)});

      builder.create<mgpu::SubgroupMmaStoreMatrixOp>(
          loc, frag, globalRef2D, ValueRange{rowOffset, zero},
          builder.getIndexAttr(fragShape.back()),
          /*transpose=*/UnitAttr());
      builder.create<mgpu::BarrierOp>(loc);

      // Read from warp's portion and convert with 2D indexing into dst.
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value nRows =
          builder.create<arith::ConstantIndexOp>(loc, fragShape[0]);
      Value nCols =
          builder.create<arith::ConstantIndexOp>(loc, fragShape[1]);

      auto outerLoop = builder.create<scf::ForOp>(loc, zero, nRows, one);
      {
        OpBuilder::InsertionGuard outerGuard(builder);
        builder.setInsertionPointToStart(outerLoop.getBody());
        Value row = outerLoop.getInductionVar();

        auto innerLoop =
            builder.create<scf::ForOp>(loc, zero, nCols, one);
        {
          OpBuilder::InsertionGuard innerGuard(builder);
          builder.setInsertionPointToStart(innerLoop.getBody());
          Value col = innerLoop.getInductionVar();

          Value linearIdx =
              builder.create<arith::MulIOp>(loc, row, nCols);
          linearIdx =
              builder.create<arith::AddIOp>(loc, linearIdx, col);
          Value srcIdx =
              builder.create<arith::AddIOp>(loc, linearIdx, warpOffset);
          Value elem = builder.create<memref::LoadOp>(
              loc, globalRef, ValueRange{srcIdx});
          Value cvt = builder.create<arith::TruncFOp>(
              loc, dstMemTy.getElementType(), elem);
          builder.create<memref::StoreOp>(loc, cvt, dst,
                                          ValueRange{row, col});
        }
      }
      return;
    }

    int64_t leadDim = dstMemTy.getShape().back();

    if (auto strided = dyn_cast<StridedLayoutAttr>(dstMemTy.getLayout())) {
      auto strides = strided.getStrides();
      if (!strides.empty() && strides[0] != ShapedType::kDynamic)
        leadDim = strides[0];
    } else if (auto castOp =
                   dst.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto staticStrides = castOp.getStaticStrides();
      if (!staticStrides.empty() &&
          staticStrides[0] != ShapedType::kDynamic)
        leadDim = staticStrides[0];
    }

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    builder.create<mgpu::SubgroupMmaStoreMatrixOp>(
        loc, frag, dst, ValueRange{zero, zero},
        builder.getIndexAttr(leadDim),
        /*transpose=*/UnitAttr());
  }

  void convertGroupParallel(OpBuilder &builder, Location loc,
                            ParallelOp par, KernelConvertCtx &ctx) {
    auto &body = par.getBody();
    if (body.empty()) return;
    auto args = body.getArguments();
    auto bounds = par.getBounds();
    int warpScale =
        (par.getLevel() == ParallelLevel::GROUPx4) ? 4 : 1;

    Value tidX = builder.create<mgpu::ThreadIdOp>(
        loc, builder.getIndexType(), mgpu::Dimension::x);
    Value warpSizeVal = builder.create<arith::ConstantIndexOp>(
        loc, 32 * warpScale);
    Value warpId = builder.create<arith::DivUIOp>(loc, tidX, warpSizeVal);

    if (args.size() == 1) {
      ctx.mapping.map(args[0], warpId);
    } else {
      Value remaining = warpId;
      for (int i = (int)args.size() - 1; i >= 0; --i) {
        Value bound =
            builder.create<arith::ConstantIndexOp>(loc, bounds[i]);
        Value dimIdx =
            builder.create<arith::RemUIOp>(loc, remaining, bound);
        ctx.mapping.map(args[i], dimIdx);
        remaining =
            builder.create<arith::DivUIOp>(loc, remaining, bound);
      }
    }

    for (auto &op : body.front().getOperations())
      convertOp(builder, loc, op, ctx);
  }

  void convertTensorTile(OpBuilder &builder, Location loc,
                         TensorTileOp tileOp, IRMapping &mapping) {
    auto srcTy = cast<coir::TensorType>(tileOp.getSource().getType());
    auto tileTy = cast<coir::TensorType>(tileOp.getResult().getType());
    Value src = mapping.lookup(tileOp.getSource());
    auto srcMemTy = cast<MemRefType>(src.getType());
    auto tileShape = tileTy.getShape();
    auto srcShape = srcTy.getShape();
    auto indices = tileOp.getIndices();

    SmallVector<int64_t> srcStrides(srcShape.size());
    {
      int64_t s = 1;
      for (int i = (int)srcShape.size() - 1; i >= 0; --i) {
        srcStrides[i] = s;
        s *= srcShape[i];
      }
    }

    Value offset = builder.create<arith::ConstantIndexOp>(loc, 0);
    if (!indices.empty()) {
      for (unsigned i = 0; i < indices.size() && i < srcShape.size(); ++i) {
        Value idx = mapping.lookup(indices[i]);
        int64_t tileDim = (i < tileShape.size()) ? tileShape[i] : 1;
        int64_t elemStride = tileDim * srcStrides[i];
        Value stride =
            builder.create<arith::ConstantIndexOp>(loc, elemStride);
        Value term = builder.create<arith::MulIOp>(loc, idx, stride);
        offset = builder.create<arith::AddIOp>(loc, offset, term);
      }
    }

    SmallVector<int64_t> tileStrides;
    if (tileShape.size() == srcShape.size()) {
      tileStrides.assign(srcStrides.begin(), srcStrides.end());
    } else {
      tileStrides.resize(tileShape.size());
      int64_t s = 1;
      for (int i = (int)tileShape.size() - 1; i >= 0; --i) {
        tileStrides[i] = s;
        s *= tileShape[i];
      }
    }

    bool hasDynOffset = false;
    int64_t staticOffset = 0;
    if (auto cst = offset.getDefiningOp<arith::ConstantIndexOp>())
      staticOffset = cst.value();
    else
      hasDynOffset = true;

    bool needsLayout = hasDynOffset || staticOffset != 0;
    MemRefType tileMem;
    if (needsLayout) {
      int64_t layoutOffset =
          hasDynOffset ? ShapedType::kDynamic : staticOffset;
      auto layout = StridedLayoutAttr::get(builder.getContext(),
                                           layoutOffset, tileStrides);
      tileMem = MemRefType::get(tileShape, srcMemTy.getElementType(),
                                layout, srcMemTy.getMemorySpace());
    } else {
      tileMem = MemRefType::get(tileShape, srcMemTy.getElementType(),
                                AffineMap{}, srcMemTy.getMemorySpace());
    }

    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      sizes.push_back(
          builder.create<arith::ConstantIndexOp>(loc, tileShape[i]));
      strides.push_back(
          builder.create<arith::ConstantIndexOp>(loc, tileStrides[i]));
    }

    Value base = getBaseMemRef(src);
    Value parentOffset = extractOffset(builder, loc, src);
    if (auto cst = parentOffset.getDefiningOp<arith::ConstantIndexOp>()) {
      if (cst.value() != 0)
        offset = builder.create<arith::AddIOp>(loc, offset, parentOffset);
    } else {
      offset = builder.create<arith::AddIOp>(loc, offset, parentOffset);
    }

    auto cast = builder.create<memref::ReinterpretCastOp>(
        loc, tileMem, base, offset, sizes, strides);
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
    DMADescRuntimeOp rtDesc = nullptr;

    if (auto prefetch = desc.getDefiningOp<DMADescPrefetchOp>())
      constDesc = prefetch.getIn().getDefiningOp<DMAConstDescOp>();
    else if (auto rt = desc.getDefiningOp<DMADescRuntimeOp>()) {
      rtDesc = rt;
      auto prefetch2 = rt.getIn().getDefiningOp<DMADescPrefetchOp>();
      if (prefetch2)
        constDesc = prefetch2.getIn().getDefiningOp<DMAConstDescOp>();
    }

    if (constDesc) {
      Value src = ctx.mapping.lookup(constDesc.getSource());
      Value dst = ctx.mapping.lookup(constDesc.getDest());

      if (rtDesc && !rtDesc.getOffsets().empty()) {
        auto srcCoirTy =
            cast<coir::TensorType>(constDesc.getSource().getType());
        auto dstCoirTy =
            cast<coir::TensorType>(constDesc.getDest().getType());
        bool isLoad = (srcCoirTy.getMemorySpace() <= 0) &&
                      (dstCoirTy.getMemorySpace() == 1);

        Value globalMem = isLoad ? src : dst;
        auto globalCoirTy = isLoad ? srcCoirTy : dstCoirTy;
        auto localCoirTy = isLoad ? dstCoirTy : srcCoirTy;
        auto globalShape = globalCoirTy.getShape();
        auto tileShape = localCoirTy.getShape();

        Value sliced =
            applyTileOffset(builder, loc, globalMem, globalShape,
                            tileShape, rtDesc.getOffsets(), ctx.mapping);
        if (isLoad)
          emitFlatCopyLoop(builder, loc, sliced, dst);
        else
          emitFlatCopyLoop(builder, loc, src, sliced);
      } else {
        emitFlatCopyLoop(builder, loc, src, dst);
      }
    }
    builder.create<mgpu::BarrierOp>(loc);
  }

  Value applyTileOffset(OpBuilder &builder, Location loc, Value base,
                        ArrayRef<int64_t> baseShape,
                        ArrayRef<int64_t> tileShape,
                        OperandRange coirOffsets, IRMapping &mapping) {
    auto baseTy = cast<MemRefType>(base.getType());

    SmallVector<int64_t> baseStrides(baseShape.size());
    {
      int64_t s = 1;
      for (int i = (int)baseShape.size() - 1; i >= 0; --i) {
        baseStrides[i] = s;
        s *= baseShape[i];
      }
    }

    Value offset = builder.create<arith::ConstantIndexOp>(loc, 0);
    for (unsigned i = 0; i < coirOffsets.size() && i < baseShape.size(); ++i) {
      Value idx = mapping.lookup(coirOffsets[i]);
      int64_t tileDim = (i < tileShape.size()) ? tileShape[i] : 1;
      int64_t elemStride = tileDim * baseStrides[i];
      Value stride = builder.create<arith::ConstantIndexOp>(loc, elemStride);
      Value term = builder.create<arith::MulIOp>(loc, idx, stride);
      offset = builder.create<arith::AddIOp>(loc, offset, term);
    }

    SmallVector<int64_t> tileStrides;
    if (tileShape.size() == baseShape.size())
      tileStrides.assign(baseStrides.begin(), baseStrides.end());
    else {
      tileStrides.resize(tileShape.size());
      int64_t s = 1;
      for (int i = (int)tileShape.size() - 1; i >= 0; --i) {
        tileStrides[i] = s;
        s *= tileShape[i];
      }
    }

    bool hasDynOffset = false;
    int64_t staticOffset = 0;
    if (auto cst = offset.getDefiningOp<arith::ConstantIndexOp>())
      staticOffset = cst.value();
    else
      hasDynOffset = true;

    int64_t layoutOffset =
        hasDynOffset ? ShapedType::kDynamic : staticOffset;
    auto layout = StridedLayoutAttr::get(builder.getContext(),
                                         layoutOffset, tileStrides);
    MemRefType tileMem = MemRefType::get(tileShape, baseTy.getElementType(),
                                         layout, baseTy.getMemorySpace());

    SmallVector<Value> sizes, strides;
    for (unsigned i = 0; i < tileShape.size(); ++i) {
      sizes.push_back(
          builder.create<arith::ConstantIndexOp>(loc, tileShape[i]));
      strides.push_back(
          builder.create<arith::ConstantIndexOp>(loc, tileStrides[i]));
    }

    Value basePtr = getBaseMemRef(base);
    Value parentOffset = extractOffset(builder, loc, base);
    if (auto cst = parentOffset.getDefiningOp<arith::ConstantIndexOp>()) {
      if (cst.value() != 0)
        offset = builder.create<arith::AddIOp>(loc, offset, parentOffset);
    } else {
      offset = builder.create<arith::AddIOp>(loc, offset, parentOffset);
    }

    return builder.create<memref::ReinterpretCastOp>(
        loc, tileMem, basePtr, offset, sizes, strides);
  }

  void convertAtomic(OpBuilder &builder, Location loc, AtomicOp atomicOp,
                     IRMapping &mapping) {
    Value dst = mapping.lookup(atomicOp.getDest());
    Value val = mapping.lookup(atomicOp.getValue());
    SmallVector<Value> indices;
    for (auto idx : atomicOp.getIndices())
      indices.push_back(mapping.lookup(idx));
    dst = flattenIfNeeded(builder, loc, dst, indices.size());

    auto kind = atomicOp.getKind();
    bool isFloat = isa<FloatType>(val.getType());

    using AK = coir::AtomicKind;
    using RMW = arith::AtomicRMWKind;

    auto mapKind = [&]() -> std::optional<RMW> {
      switch (kind) {
      case AK::Add:  return isFloat ? RMW::addf : RMW::addi;
      case AK::Exch: return RMW::assign;
      case AK::Min:  return isFloat ? RMW::minimumf : RMW::mins;
      case AK::Max:  return isFloat ? RMW::maximumf : RMW::maxs;
      case AK::And:  return RMW::andi;
      case AK::Or:   return RMW::ori;
      default:       return std::nullopt;
      }
    };

    auto rmwKind = mapKind();
    if (rmwKind) {
      builder.create<memref::AtomicRMWOp>(loc, val.getType(), *rmwKind, val,
                                          dst, indices);
    } else if (kind == AK::Sub) {
      Value neg;
      if (isFloat)
        neg = builder.create<arith::NegFOp>(loc, val);
      else {
        Value zero = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(val.getType(), 0));
        neg = builder.create<arith::SubIOp>(loc, zero, val);
      }
      builder.create<memref::AtomicRMWOp>(loc, val.getType(), RMW::addi, neg,
                                          dst, indices);
    } else {
      auto genAtomic = builder.create<memref::GenericAtomicRMWOp>(
          loc, dst, indices);
      Block *body = &genAtomic.body().front();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(body);
      Value current = body->getArgument(0);
      Value result;
      if (kind == AK::Xor)
        result = builder.create<arith::XOrIOp>(loc, current, val);
      else
        result = current;
      builder.create<memref::AtomicYieldOp>(loc, result);
    }
  }

  void convertScfIf(OpBuilder &builder, Location loc, scf::IfOp ifOp,
                    KernelConvertCtx &ctx) {
    Value cond = ctx.mapping.lookup(ifOp.getCondition());

    bool hasElse = !ifOp.getElseRegion().empty();
    SmallVector<Type> resultTypes;
    for (auto r : ifOp.getResults())
      resultTypes.push_back(r.getType());

    auto newIf = builder.create<scf::IfOp>(loc, resultTypes, cond, hasElse);

    {
      OpBuilder::InsertionGuard guard(builder);
      Block &thenBlock = newIf.getThenRegion().front();
      if (thenBlock.mightHaveTerminator())
        thenBlock.getTerminator()->erase();
      builder.setInsertionPointToEnd(&thenBlock);

      for (auto &op : ifOp.getThenRegion().front().getOperations()) {
        if (isa<scf::YieldOp>(op)) {
          SmallVector<Value> yieldVals;
          for (auto v : op.getOperands())
            yieldVals.push_back(ctx.mapping.lookup(v));
          builder.create<scf::YieldOp>(loc, yieldVals);
          continue;
        }
        convertOp(builder, loc, op, ctx);
      }
      if (!thenBlock.mightHaveTerminator())
        builder.create<scf::YieldOp>(loc);
    }

    if (hasElse) {
      OpBuilder::InsertionGuard guard(builder);
      Block &elseBlock = newIf.getElseRegion().front();
      if (elseBlock.mightHaveTerminator())
        elseBlock.getTerminator()->erase();
      builder.setInsertionPointToEnd(&elseBlock);

      for (auto &op : ifOp.getElseRegion().front().getOperations()) {
        if (isa<scf::YieldOp>(op)) {
          SmallVector<Value> yieldVals;
          for (auto v : op.getOperands())
            yieldVals.push_back(ctx.mapping.lookup(v));
          builder.create<scf::YieldOp>(loc, yieldVals);
          continue;
        }
        convertOp(builder, loc, op, ctx);
      }
      if (!elseBlock.mightHaveTerminator())
        builder.create<scf::YieldOp>(loc);
    }

    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      ctx.mapping.map(ifOp.getResult(i), newIf.getResult(i));
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
