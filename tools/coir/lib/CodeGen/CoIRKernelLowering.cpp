//===- CoIRKernelLowering.cpp - Shared CoIR->GPU kernel lowering ----------===//
//
// Shared implementations for lowering CoIR kernel ops to GPU MLIR.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/CoIRKernelLowering.h"
#include "Dialect/CoIR/CoIRAttrs.h"

using namespace mlir;
using namespace coir;
namespace mgpu = mlir::gpu;

// ===== Static helpers =====

mgpu::Dimension CoIRKernelLoweringBase::dimFromIndex(unsigned i) {
  switch (i) {
  case 0: return mgpu::Dimension::x;
  case 1: return mgpu::Dimension::y;
  case 2: return mgpu::Dimension::z;
  default: return mgpu::Dimension::x;
  }
}

LaunchDims CoIRKernelLoweringBase::collectLaunchDims(KernelOp kernel) {
  LaunchDims dims;
  kernel.getBody().walk([&](ParallelOp par) {
    auto bounds = par.getBounds();
    auto lvl = par.getLevel();
    SmallVector<int64_t, 3> *target = nullptr;
    if (lvl == ParallelLevel::BLOCK)
      target = &dims.gridDims;
    else if (lvl == ParallelLevel::THREAD)
      target = &dims.blockDims;
    if (!target) return;
    for (unsigned i = 0; i < bounds.size() && i < 3; ++i)
      (*target)[i] = bounds[i];
  });
  return dims;
}

// ===== Kernel conversion =====

LogicalResult CoIRKernelLoweringBase::convertKernel(ModuleOp module,
                                                    KernelOp kernel) {
  OpBuilder builder(module.getContext());
  Location loc = kernel.getLoc();
  auto fnType = kernel.getFunctionType();
  StringRef symName = kernel.getSymName();

  LaunchDims launchDims = collectLaunchDims(kernel);

  SmallVector<Type> gpuArgTypes;
  for (auto inTy : fnType.getInputs()) {
    if (auto tty = dyn_cast<TensorType>(inTy))
      gpuArgTypes.push_back(convertTensorType(tty));
    else
      gpuArgTypes.push_back(inTy);
  }
  for (auto resTy : fnType.getResults()) {
    if (auto tty = dyn_cast<TensorType>(resTy))
      gpuArgTypes.push_back(convertTensorType(tty));
    else
      gpuArgTypes.push_back(resTy);
  }

  std::string moduleName = (symName + "_module").str();
  builder.setInsertionPoint(kernel);
  auto gpuModule = builder.create<mgpu::GPUModuleOp>(loc, moduleName);

  preScanKernel(kernel, builder, gpuModule);

  builder.setInsertionPointToEnd(gpuModule.getBody());
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

  KernelConvertCtx ctx{mapping, returnAllocMap};
  for (auto &op : kernelBody.front().getOperations())
    convertOp(builder, loc, op, ctx);

  if (entry.empty() || !entry.back().hasTrait<OpTrait::IsTerminator>())
    builder.create<mgpu::ReturnOp>(loc);

  kernel.erase();
  return success();
}

// ===== Op dispatch =====

void CoIRKernelLoweringBase::convertOp(OpBuilder &builder, Location loc,
                                       Operation &op, KernelConvertCtx &ctx) {
  if (convertTargetOp(builder, loc, op, ctx))
    return;

  if (isa<KernelReturnOp>(op)) {
    builder.create<mgpu::ReturnOp>(loc);
    return;
  }
  if (auto alloc = dyn_cast<TensorAllocOp>(op)) {
    convertAlloc(builder, loc, alloc, ctx);
    return;
  }
  if (auto par = dyn_cast<ParallelOp>(op)) {
    convertParallel(builder, loc, par, ctx);
    return;
  }
  if (auto fe = dyn_cast<ForeachOp>(op)) {
    convertForeach(builder, loc, fe, ctx);
    return;
  }
  if (auto le = dyn_cast<TensorLoadElemOp>(op)) {
    convertLoadElem(builder, loc, le, ctx.mapping);
    return;
  }
  if (auto se = dyn_cast<TensorStoreElemOp>(op)) {
    convertStoreElem(builder, loc, se, ctx.mapping);
    return;
  }
  if (auto re = dyn_cast<TensorReduceElemOp>(op)) {
    convertReduceElem(builder, loc, re, ctx.mapping);
    return;
  }
  if (auto ec = dyn_cast<ElementCopyOp>(op)) {
    convertElementCopy(builder, loc, ec, ctx.mapping);
    return;
  }
  if (auto dc = dyn_cast<DmaCopyOp>(op)) {
    convertDmaCopy(builder, loc, dc, ctx);
    return;
  }

  // DMA descriptor pipeline ops -- no-ops at this level.
  if (isa<DMAConstDescOp>(op) || isa<DMADescPrefetchOp>(op) ||
      isa<DMADescRuntimeOp>(op) || isa<DMACheckOp>(op))
    return;

  // Synchronization
  if (isa<BarrierOp>(op) || isa<WaitOp>(op)) {
    builder.create<mgpu::BarrierOp>(loc);
    return;
  }

  if (auto rotate = dyn_cast<FutureRotateOp>(op)) {
    auto inputs = rotate.getFutures();
    auto outputs = rotate.getResults();
    unsigned n = inputs.size();
    if (n >= 2) {
      SmallVector<Value> mapped;
      for (auto in : inputs)
        mapped.push_back(ctx.mapping.lookup(in));
      for (unsigned i = 0; i < n; ++i)
        ctx.mapping.map(outputs[i], mapped[(i + 1) % n]);
    } else if (n == 1) {
      ctx.mapping.map(outputs[0], ctx.mapping.lookup(inputs[0]));
    }
    return;
  }

  if (isa<AsyncUndefOp>(op)) {
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    Value cond = ctx.mapping.lookup(ifOp.getCondition());
    SmallVector<Type> resultTypes;
    for (auto ty : ifOp.getResultTypes()) {
      if (auto tty = dyn_cast<TensorType>(ty))
        resultTypes.push_back(convertTensorType(tty));
      else
        resultTypes.push_back(ty);
    }
    bool hasElse = !ifOp.getElseRegion().empty();
    auto newIf = builder.create<scf::IfOp>(loc, resultTypes, cond, hasElse);
    {
      OpBuilder::InsertionGuard guard(builder);

      if (newIf.thenBlock()->mightHaveTerminator())
        newIf.thenBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIf.thenBlock());
      for (auto &thenOp : ifOp.thenBlock()->getOperations()) {
        if (auto yield = dyn_cast<scf::YieldOp>(thenOp)) {
          SmallVector<Value> yieldVals;
          for (auto v : yield.getOperands())
            yieldVals.push_back(ctx.mapping.lookup(v));
          builder.create<scf::YieldOp>(loc, yieldVals);
        } else {
          convertOp(builder, loc, thenOp, ctx);
        }
      }
      if (hasElse) {
        if (newIf.elseBlock()->mightHaveTerminator())
          newIf.elseBlock()->getTerminator()->erase();
        builder.setInsertionPointToEnd(newIf.elseBlock());
        for (auto &elseOp : ifOp.elseBlock()->getOperations()) {
          if (auto yield = dyn_cast<scf::YieldOp>(elseOp)) {
            SmallVector<Value> yieldVals;
            for (auto v : yield.getOperands())
              yieldVals.push_back(ctx.mapping.lookup(v));
            builder.create<scf::YieldOp>(loc, yieldVals);
          } else {
            convertOp(builder, loc, elseOp, ctx);
          }
        }
      }
    }
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      ctx.mapping.map(ifOp.getResult(i), newIf.getResult(i));
    return;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    SmallVector<Value> inits;
    for (auto init : whileOp.getInits())
      inits.push_back(ctx.mapping.lookup(init));

    SmallVector<Type> resultTypes;
    for (auto ty : whileOp.getResultTypes())
      resultTypes.push_back(ty);

    auto newWhile = builder.create<scf::WhileOp>(loc, resultTypes, inits);
    {
      OpBuilder::InsertionGuard guard(builder);

      SmallVector<Type> beforeArgTypes;
      for (auto arg : whileOp.getBeforeArguments())
        beforeArgTypes.push_back(arg.getType());
      SmallVector<Location> beforeLocs(beforeArgTypes.size(), loc);
      Block *beforeBlock = builder.createBlock(
          &newWhile.getBefore(), newWhile.getBefore().end(),
          beforeArgTypes, beforeLocs);
      builder.setInsertionPointToStart(beforeBlock);
      for (unsigned i = 0; i < whileOp.getBeforeArguments().size(); ++i)
        ctx.mapping.map(whileOp.getBeforeArguments()[i],
                        beforeBlock->getArgument(i));
      for (auto &beforeOp : whileOp.getBefore().front().getOperations()) {
        if (auto condOp = dyn_cast<scf::ConditionOp>(beforeOp)) {
          Value cond = ctx.mapping.lookup(condOp.getCondition());
          SmallVector<Value> condArgs;
          for (auto v : condOp.getArgs())
            condArgs.push_back(ctx.mapping.lookup(v));
          builder.create<scf::ConditionOp>(loc, cond, condArgs);
        } else {
          convertOp(builder, loc, beforeOp, ctx);
        }
      }

      SmallVector<Type> afterArgTypes;
      for (auto arg : whileOp.getAfterArguments())
        afterArgTypes.push_back(arg.getType());
      SmallVector<Location> afterLocs(afterArgTypes.size(), loc);
      Block *afterBlock = builder.createBlock(
          &newWhile.getAfter(), newWhile.getAfter().end(),
          afterArgTypes, afterLocs);
      builder.setInsertionPointToStart(afterBlock);
      for (unsigned i = 0; i < whileOp.getAfterArguments().size(); ++i)
        ctx.mapping.map(whileOp.getAfterArguments()[i],
                        afterBlock->getArgument(i));
      for (auto &afterOp : whileOp.getAfter().front().getOperations()) {
        if (auto yield = dyn_cast<scf::YieldOp>(afterOp)) {
          SmallVector<Value> yieldVals;
          for (auto v : yield.getOperands())
            yieldVals.push_back(ctx.mapping.lookup(v));
          builder.create<scf::YieldOp>(loc, yieldVals);
        } else {
          convertOp(builder, loc, afterOp, ctx);
        }
      }
    }
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
      ctx.mapping.map(whileOp.getResult(i), newWhile.getResult(i));
    return;
  }

  if (isa<AssertOp>(op) || isa<YieldOp>(op))
    return;

  builder.clone(op, ctx.mapping);
}

// ===== Shared op converters =====

void CoIRKernelLoweringBase::convertAlloc(OpBuilder &builder, Location loc,
                                          TensorAllocOp alloc,
                                          KernelConvertCtx &ctx) {
  auto tty = cast<TensorType>(alloc.getResult().getType());
  auto it = ctx.returnAllocMap.find(alloc.getResult());
  if (it != ctx.returnAllocMap.end()) {
    ctx.mapping.map(alloc.getResult(), it->second);
    return;
  }
  auto memTy = convertTensorType(tty);
  auto newAlloc = builder.create<memref::AllocOp>(loc, memTy);
  ctx.mapping.map(alloc.getResult(), newAlloc.getResult());
}

void CoIRKernelLoweringBase::convertLoadElem(OpBuilder &builder, Location loc,
                                             TensorLoadElemOp loadElem,
                                             IRMapping &mapping) {
  Value src = mapping.lookup(loadElem.getSource());
  SmallVector<Value> indices;
  for (auto idx : loadElem.getIndices())
    indices.push_back(mapping.lookup(idx));
  if (indices.empty() && cast<MemRefType>(src.getType()).getRank() > 0) {
    src = flattenIfNeeded(builder, loc, src, 1);
    indices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  } else {
    src = flattenIfNeeded(builder, loc, src, indices.size());
  }
  auto loaded = builder.create<memref::LoadOp>(loc, src, indices);
  mapping.map(loadElem.getResult(), loaded.getResult());
}

void CoIRKernelLoweringBase::convertStoreElem(OpBuilder &builder, Location loc,
                                              TensorStoreElemOp storeElem,
                                              IRMapping &mapping) {
  Value dst = mapping.lookup(storeElem.getDest());
  Value val = mapping.lookup(storeElem.getValue());
  SmallVector<Value> indices;
  for (auto idx : storeElem.getIndices())
    indices.push_back(mapping.lookup(idx));
  if (indices.empty() && cast<MemRefType>(dst.getType()).getRank() > 0) {
    dst = flattenIfNeeded(builder, loc, dst, 1);
    indices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  } else {
    dst = flattenIfNeeded(builder, loc, dst, indices.size());
  }
  builder.create<memref::StoreOp>(loc, val, dst, indices);
}

void CoIRKernelLoweringBase::convertReduceElem(
    OpBuilder &builder, Location loc, TensorReduceElemOp reduceElem,
    IRMapping &mapping) {
  Value dst = mapping.lookup(reduceElem.getDest());
  Value val = mapping.lookup(reduceElem.getValue());
  SmallVector<Value> indices;
  for (auto idx : reduceElem.getIndices())
    indices.push_back(mapping.lookup(idx));
  if (indices.empty() && cast<MemRefType>(dst.getType()).getRank() > 0) {
    dst = flattenIfNeeded(builder, loc, dst, 1);
    indices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  } else {
    dst = flattenIfNeeded(builder, loc, dst, indices.size());
  }
  Value old = builder.create<memref::LoadOp>(loc, dst, indices);
  Value sum;
  if (isa<FloatType>(val.getType()))
    sum = builder.create<arith::AddFOp>(loc, old, val);
  else
    sum = builder.create<arith::AddIOp>(loc, old, val);
  builder.create<memref::StoreOp>(loc, sum, dst, indices);
}

// ===== Rank-mismatch helpers =====

Value CoIRKernelLoweringBase::flattenIfNeeded(OpBuilder &builder, Location loc,
                                              Value memref,
                                              unsigned numIndices) {
  auto memTy = cast<MemRefType>(memref.getType());
  if ((unsigned)memTy.getRank() == numIndices)
    return memref;
  int64_t totalElems = 1;
  for (auto d : memTy.getShape())
    totalElems *= d;
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
    flatTy = MemRefType::get({totalElems}, memTy.getElementType(), layout,
                             memTy.getMemorySpace());
  } else if (staticOff != 0) {
    auto layout =
        StridedLayoutAttr::get(builder.getContext(), staticOff, {1});
    flatTy = MemRefType::get({totalElems}, memTy.getElementType(), layout,
                             memTy.getMemorySpace());
  } else {
    flatTy = MemRefType::get({totalElems}, memTy.getElementType(),
                             AffineMap{}, memTy.getMemorySpace());
  }

  Value sz = builder.create<arith::ConstantIndexOp>(loc, totalElems);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  return builder.create<memref::ReinterpretCastOp>(loc, flatTy, base, offset,
                                                   ValueRange{sz},
                                                   ValueRange{one});
}

Value CoIRKernelLoweringBase::getBaseMemRef(Value memref) {
  if (auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>())
    return castOp.getSource();
  return memref;
}

Value CoIRKernelLoweringBase::extractOffset(OpBuilder &builder, Location loc,
                                            Value memref) {
  if (auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto staticOffsets = castOp.getStaticOffsets();
    if (!staticOffsets.empty() && staticOffsets[0] == ShapedType::kDynamic)
      return castOp.getOffsets()[0];
    if (!staticOffsets.empty() && staticOffsets[0] != 0)
      return builder.create<arith::ConstantIndexOp>(loc, staticOffsets[0]);
  }
  return builder.create<arith::ConstantIndexOp>(loc, 0);
}

void CoIRKernelLoweringBase::convertElementCopy(OpBuilder &builder,
                                                Location loc,
                                                ElementCopyOp copyOp,
                                                IRMapping &mapping) {
  Value src = mapping.lookup(copyOp.getSource());
  Value dst = mapping.lookup(copyOp.getDest());
  emitFlatCopyLoop(builder, loc, src, dst);
}

void CoIRKernelLoweringBase::convertDmaCopy(OpBuilder &builder, Location loc,
                                            DmaCopyOp copyOp,
                                            KernelConvertCtx &ctx) {
  Value src = ctx.mapping.lookup(copyOp.getSource());
  Value dst = ctx.mapping.lookup(copyOp.getDest());
  emitFlatCopyLoop(builder, loc, src, dst);
  if (copyOp.getToken())
    ctx.mapping.map(copyOp.getToken(), dst);
}

void CoIRKernelLoweringBase::emitFlatCopyLoop(OpBuilder &builder, Location loc,
                                              Value src, Value dst) {
  auto srcTy = cast<MemRefType>(src.getType());
  int64_t totalElems = 1;
  for (auto dim : srcTy.getShape())
    totalElems *= dim;

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value total = builder.create<arith::ConstantIndexOp>(loc, totalElems);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto elemTy = srcTy.getElementType();
  auto flatSrcTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                   srcTy.getMemorySpace());
  auto dstTy = cast<MemRefType>(dst.getType());
  auto flatDstTy = MemRefType::get({totalElems}, elemTy, AffineMap{},
                                   dstTy.getMemorySpace());

  auto flatSrc = builder.create<memref::ReinterpretCastOp>(
      loc, flatSrcTy, src, zero, ValueRange{total}, ValueRange{one});
  auto flatDst = builder.create<memref::ReinterpretCastOp>(
      loc, flatDstTy, dst, zero, ValueRange{total}, ValueRange{one});

  auto loop = builder.create<scf::ForOp>(loc, zero, total, one);
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value elem = builder.create<memref::LoadOp>(loc, flatSrc, iv);
    builder.create<memref::StoreOp>(loc, elem, flatDst, iv);
  }
}

void CoIRKernelLoweringBase::convertForeach(OpBuilder &builder, Location loc,
                                            ForeachOp fe,
                                            KernelConvertCtx &ctx) {
  auto &body = fe.getBody();
  if (body.empty()) return;
  auto args = body.front().getArguments();

  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = ctx.mapping.lookup(fe.getUpperBound());
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> initVals;
  SmallVector<unsigned> asyncUndefIndices;
  for (unsigned i = 0; i < fe.getIterArgs().size(); ++i) {
    Value init = fe.getIterArgs()[i];
    if (ctx.mapping.contains(init)) {
      initVals.push_back(ctx.mapping.lookup(init));
    } else {
      asyncUndefIndices.push_back(i);
      initVals.push_back(Value());
    }
  }

  if (!asyncUndefIndices.empty()) {
    SmallVector<Value> yieldedVals;
    for (auto &op : body.front().getOperations()) {
      if (auto yield = dyn_cast<YieldOp>(op)) {
        for (auto v : yield.getOperands())
          yieldedVals.push_back(v);
      }
    }
    for (unsigned idx : asyncUndefIndices) {
      Value yieldedVal = yieldedVals[idx];
      if (auto *defOp = yieldedVal.getDefiningOp()) {
        if (auto rotateOp = dyn_cast<FutureRotateOp>(defOp)) {
          Value firstInput = rotateOp.getFutures()[0];
          if (ctx.mapping.contains(firstInput)) {
            Value mapped = ctx.mapping.lookup(firstInput);
            initVals[idx] = mapped;
            ctx.mapping.map(fe.getIterArgs()[idx], mapped);
            continue;
          }
        }
      }
      if (ctx.mapping.contains(yieldedVal)) {
        Value mapped = ctx.mapping.lookup(yieldedVal);
        initVals[idx] = mapped;
        ctx.mapping.map(fe.getIterArgs()[idx], mapped);
      }
    }
    for (unsigned idx : asyncUndefIndices) {
      if (!initVals[idx]) {
        Value dummy = builder.create<arith::ConstantIndexOp>(loc, 0);
        initVals[idx] = dummy;
        ctx.mapping.map(fe.getIterArgs()[idx], dummy);
      }
    }
  }

  auto loop = builder.create<scf::ForOp>(loc, zero, ub, step, initVals);
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *loopBody = loop.getBody();

    if (loopBody->mightHaveTerminator())
      loopBody->getTerminator()->erase();

    builder.setInsertionPointToEnd(loopBody);

    ctx.mapping.map(args[0], loop.getInductionVar());
    for (unsigned i = 0; i < initVals.size(); ++i)
      ctx.mapping.map(args[i + 1], loop.getRegionIterArg(i));

    for (auto &op : body.front().getOperations()) {
      if (auto yield = dyn_cast<YieldOp>(op)) {
        SmallVector<Value> yieldedVals;
        for (auto v : yield.getOperands())
          yieldedVals.push_back(ctx.mapping.lookup(v));
        builder.create<scf::YieldOp>(loc, yieldedVals);
        continue;
      }
      convertOp(builder, loc, op, ctx);
    }
  }

  for (unsigned i = 0; i < fe.getNumResults(); ++i)
    ctx.mapping.map(fe.getResult(i), loop.getResult(i));
}

void CoIRKernelLoweringBase::convertParallel(OpBuilder &builder, Location loc,
                                             ParallelOp par,
                                             KernelConvertCtx &ctx) {
  auto lvl = par.getLevel();
  auto &body = par.getBody();
  if (body.empty()) return;

  auto args = body.getArguments();
  for (unsigned i = 0; i < args.size() && i < 3; ++i) {
    Value id;
    if (lvl == ParallelLevel::THREAD)
      id = builder.create<mgpu::ThreadIdOp>(loc, dimFromIndex(i));
    else if (lvl == ParallelLevel::BLOCK)
      id = builder.create<mgpu::BlockIdOp>(loc, dimFromIndex(i));
    else
      id = builder.create<arith::ConstantIndexOp>(loc, 0);
    ctx.mapping.map(args[i], id);
  }

  for (auto &op : body.front().getOperations())
    convertOp(builder, loc, op, ctx);
}
