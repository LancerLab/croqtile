// ast_coir_gen.cpp -- AST-to-CoIR MLIR translation
#include "ASTCoIRGen.hpp"
#include "codegen_utils.hpp"
#include "symbexpr.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace Choreo;
using namespace CoIR;

namespace {

int64_t EvalToInt(const ValueItem &vi) {
  if (!vi || !vi->IsNumeric()) return 0;
  if (auto *nv = dyn_cast<sbe::NumericValue>(vi.get()))
    return nv->Value();
  return 0;
}

} // namespace

mlir::Type ASTCoIRGen::LowerBaseType(BaseType bt) {
  auto &ctx = IRContext();
  switch (bt) {
  case BaseType::F64: return mlir::Float64Type::get(&ctx);
  case BaseType::F32: return mlir::Float32Type::get(&ctx);
  case BaseType::F16: return mlir::Float16Type::get(&ctx);
  case BaseType::BF16: return mlir::BFloat16Type::get(&ctx);
  case BaseType::F8_E4M3: return mlir::Float8E4M3FNType::get(&ctx);
  case BaseType::F8_E5M2: return mlir::Float8E5M2Type::get(&ctx);
  case BaseType::U32:
    return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Unsigned);
  case BaseType::S32:
    return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signless);
  case BaseType::BOOL: return mlir::IntegerType::get(&ctx, 1);
  default: return mlir::Float32Type::get(&ctx);
  }
}

coir::TensorType
ASTCoIRGen::LowerSpannedType(const ptr<SpannedType> &sty) {
  auto elemType = LowerBaseType(sty->ElementType());
  auto mdspan = sty->s_type;
  auto &shape = mdspan->value;
  llvm::SmallVector<int64_t> dims;
  if (shape.IsValid())
    for (auto &v : shape.Value())
      dims.push_back(EvalToInt(v));

  int32_t memSpace = -1;
  switch (sty->m_type) {
  case Storage::GLOBAL:
  case Storage::DEFAULT:
    memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Global);
    break;
  case Storage::SHARED:
    memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Shared);
    break;
  case Storage::LOCAL:
    memSpace = static_cast<int32_t>(coir::TensorMemorySpace::Local);
    break;
  default: break;
  }

  llvm::SmallVector<int64_t> strides;
  auto &choreoStrides = sty->GetStrides();
  for (auto &sv : choreoStrides) {
    auto val = EvalToInt(sv);
    strides.push_back(val);
  }

  // Omit strides when they describe a dense contiguous layout (row-major)
  bool isDense = true;
  if (!strides.empty() && strides.size() == dims.size()) {
    int64_t expected = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
      if (strides[i] != expected) { isDense = false; break; }
      expected *= dims[i];
    }
  }
  if (isDense)
    strides.clear();

  return coir::TensorType::get(&IRContext(), elemType, dims, memSpace, strides);
}

coir::ParallelLevelAttr
ASTCoIRGen::LowerParallelLevel(ParallelLevel pl) {
  coir::ParallelLevel level;
  switch (pl) {
  case ParallelLevel::BLOCK:   level = coir::ParallelLevel::BLOCK; break;
  case ParallelLevel::GROUP:   level = coir::ParallelLevel::GROUP; break;
  case ParallelLevel::GROUPx4: level = coir::ParallelLevel::GROUPx4; break;
  case ParallelLevel::THREAD:  level = coir::ParallelLevel::THREAD; break;
  case ParallelLevel::CLUSTER: level = coir::ParallelLevel::CLUSTER; break;
  case ParallelLevel::DEVICE:  level = coir::ParallelLevel::DEVICE; break;
  case ParallelLevel::SEQ:     level = coir::ParallelLevel::SEQ; break;
  default: level = coir::ParallelLevel::BLOCK; break;
  }
  return coir::ParallelLevelAttr::get(&IRContext(), level);
}

bool ASTCoIRGen::BeforeVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    PushScope();
  } else if (auto *cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    PushScope();
    CreateKernelOp(*cf);
  } else if (isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n) ||
             isa<AST::WithBlock>(&n)) {
    PushScope();
  }
  return true;
}

bool ASTCoIRGen::InMidVisitImpl(AST::Node &) { return true; }

bool ASTCoIRGen::AfterVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    PopScope();
    if (!suppress_output) {
      mlir::OpPrintingFlags flags;
      IRModule().print(llvm::outs(), flags);
      llvm::outs() << "\n";
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    auto *block = builder.getInsertionBlock();
    if (block && (block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()))
      builder.create<coir::KernelReturnOp>(
          builder.getUnknownLoc(), mlir::ValueRange{});
    PopScope();
  } else if (isa<AST::ParallelBy>(&n)) {
    auto *block = builder.getInsertionBlock();
    if (block && (block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()))
      builder.create<coir::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    auto *parentBlock = block->getParentOp()->getBlock();
    if (parentBlock) builder.setInsertionPointAfter(block->getParentOp());
    PopScope();
  } else if (isa<AST::ForeachBlock>(&n)) {
    auto *block = builder.getInsertionBlock();
    auto *foreachOp = block->getParentOp();

    if (!pendingYields.empty()) {
      auto *terminator = block->getTerminator();
      llvm::SmallVector<mlir::Value> yieldVals;
      for (auto &[name, _] : pendingYields) {
        auto val = LookupValue(name);
        if (val) yieldVals.push_back(val);
      }
      if (terminator)
        terminator->erase();
      builder.setInsertionPointToEnd(block);
      builder.create<coir::YieldOp>(builder.getUnknownLoc(), yieldVals);
    }

    auto *parentBlock = foreachOp->getBlock();
    if (parentBlock) builder.setInsertionPointAfter(foreachOp);

    llvm::SmallVector<std::pair<std::string, mlir::Value>> resultMappings;
    if (auto foreach_ = mlir::dyn_cast<coir::ForeachOp>(foreachOp)) {
      unsigned numRes = foreach_.getNumResults();
      for (unsigned i = 0; i < pendingYields.size(); ++i) {
        if (i < numRes)
          resultMappings.push_back(
              {pendingYields[i].first, foreach_.getResult(i)});
      }
    }
    pendingYields.clear();
    PopScope();
    for (auto &[name, val] : resultMappings)
      UpdateValue(name, val);
  } else if (isa<AST::WithBlock>(&n)) {
    PopScope();
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::Program &) { return true; }

void ASTCoIRGen::CreateKernelOp(AST::ChoreoFunction &cf) {
  auto loc = Loc(cf);
  auto fty = cast<Choreo::FunctionType>(GetSymbolType(cf.name));

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (auto &inTy : fty->in_tys) {
    if (auto sty = dyn_cast<SpannedType>(inTy))
      argTypes.push_back(LowerSpannedType(sty));
    else
      argTypes.push_back(LowerBaseType(inTy->GetBaseType()));
  }

  if (fty->out_ty && fty->out_ty->GetBaseType() != BaseType::VOID) {
    if (auto sty = dyn_cast<SpannedType>(fty->out_ty))
      resultTypes.push_back(LowerSpannedType(sty));
    else
      resultTypes.push_back(LowerBaseType(fty->out_ty->GetBaseType()));
  }

  auto mlirFnType =
      mlir::FunctionType::get(&IRContext(), argTypes, resultTypes);

  builder.setInsertionPointToEnd(IRModule().getBody());
  auto kernelOp = builder.create<coir::KernelOp>(
      loc, mlir::StringAttr::get(&IRContext(), cf.name),
      mlir::TypeAttr::get(mlirFnType));

  auto &bodyRegion = kernelOp.getBody();
  auto *entryBlock = new mlir::Block();
  bodyRegion.push_back(entryBlock);

  for (unsigned i = 0; i < argTypes.size(); ++i)
    entryBlock->addArgument(argTypes[i], loc);

  builder.setInsertionPointToEnd(entryBlock);

  if (cf.f_decl.params) {
    unsigned idx = 0;
    for (auto &param : cf.f_decl.params->values) {
      if (param->HasSymbol() && idx < argTypes.size())
        MapValue(param->sym->name, entryBlock->getArgument(idx));
      ++idx;
    }
  }
}

bool ASTCoIRGen::Visit(AST::ChoreoFunction &) { return true; }

bool ASTCoIRGen::Visit(AST::FunctionDecl &) { return true; }

bool ASTCoIRGen::Visit(AST::ParallelBy &pb) {
  auto loc = Loc(pb);
  auto levelAttr = LowerParallelLevel(pb.GetLevel());

  llvm::SmallVector<int64_t> bounds;
  if (pb.HasSubPVs()) {
    for (auto &bnd : pb.AllBoundExprs())
      if (auto *expr = dyn_cast<AST::Expr>(bnd.get()))
        if (expr->Opts().HasVal())
          bounds.push_back(EvalToInt(expr->Opts().GetVal()));
  } else {
    auto bv = pb.BoundValue();
    bounds.push_back(EvalToInt(bv));
  }

  auto parallelOp = builder.create<coir::ParallelOp>(
      loc, levelAttr,
      mlir::DenseI64ArrayAttr::get(&IRContext(), bounds));

  auto &bodyRegion = parallelOp.getBody();
  auto *block = new mlir::Block();
  bodyRegion.push_back(block);

  auto indexType = mlir::IndexType::get(&IRContext());
  if (pb.HasSubPVs()) {
    for (auto &bpv : pb.AllSubPVs()) {
      auto arg = block->addArgument(indexType, loc);
      if (auto *id = dyn_cast<AST::Identifier>(bpv.get()))
        MapValue(id->name, arg);
    }
    // For 1D parallel, map the BPV name to the single sub-PV arg
    if (pb.AllSubPVs().size() == 1)
      MapValue(pb.BPV()->name, block->getArgument(0));
  } else {
    auto arg = block->addArgument(indexType, loc);
    MapValue(pb.BPV()->name, arg);
  }

  builder.setInsertionPointToEnd(block);
  return true;
}

namespace {
void collectMMAAccNames(AST::Node *node,
                        llvm::SmallVectorImpl<std::string> &names) {
  if (!node) return;
  if (auto *mma = dyn_cast<AST::MMA>(node)) {
    auto &op = *mma->GetOperation();
    if (op.IsKind(AST::MMAOperation::Exec)) {
      auto acc = op.ExecOperand(0);
      if (acc) names.push_back(AST::FragName(acc));
    }
    return;
  }
  if (auto *mn = dyn_cast<AST::MultiNodes>(node)) {
    for (auto &child : mn->values) {
      if (child) collectMMAAccNames(child.get(), names);
    }
    return;
  }
  if (node->HasBody()) {
    auto body = node->GetBody();
    if (body) collectMMAAccNames(body.get(), names);
  }
}
} // namespace

bool ASTCoIRGen::Visit(AST::ForeachBlock &fb) {
  auto loc = Loc(fb);
  auto indexType = mlir::IndexType::get(&IRContext());

  int64_t bound = 1;
  if (fb.ranges && !fb.ranges->values.empty()) {
    auto &firstRange = fb.ranges->values[0];
    if (auto *lr = dyn_cast<AST::LoopRange>(firstRange.get())) {
      if (lr->ubound) {
        if (auto *expr = dyn_cast<AST::Expr>(lr->ubound.get()))
          if (expr->Opts().HasVal())
            bound = EvalToInt(expr->Opts().GetVal());
      }
    }
  }

  llvm::SmallVector<std::string> accNames;
  if (fb.stmts) collectMMAAccNames(fb.stmts.get(), accNames);

  llvm::SmallVector<mlir::Value> iterArgs;
  llvm::SmallVector<mlir::Type> iterTypes;
  llvm::SmallVector<std::string> iterNames;
  for (auto &name : accNames) {
    auto val = LookupValue(name);
    if (val) {
      iterArgs.push_back(val);
      iterTypes.push_back(val.getType());
      iterNames.push_back(name);
    }
  }

  auto ubConst = builder.create<mlir::arith::ConstantIndexOp>(loc, bound);

  auto foreachOp = builder.create<coir::ForeachOp>(
      loc, iterTypes, ubConst.getResult(), iterArgs);

  auto &bodyRegion = foreachOp.getBody();
  auto *block = new mlir::Block();
  bodyRegion.push_back(block);

  auto ivArg = block->addArgument(indexType, loc);

  if (fb.ranges && !fb.ranges->values.empty()) {
    auto &firstRange = fb.ranges->values[0];
    if (auto *lr = dyn_cast<AST::LoopRange>(firstRange.get()))
      MapValue(lr->GetIVName(), ivArg);
  }

  pendingYields.clear();
  for (unsigned i = 0; i < iterNames.size(); ++i) {
    auto iterArg = block->addArgument(iterTypes[i], loc);
    MapValue(iterNames[i], iterArg);
    pendingYields.push_back({iterNames[i], iterArg});
  }

  builder.setInsertionPointToEnd(block);
  builder.create<coir::YieldOp>(loc, mlir::ValueRange{});
  builder.setInsertionPoint(block->getTerminator());

  return true;
}


mlir::Value ASTCoIRGen::EmitExpr(AST::Node &n) {
  auto loc = Loc(n);

  if (auto *id = dyn_cast<AST::Identifier>(&n)) {
    return LookupValue(id->name);
  }

  if (auto *ii = dyn_cast<AST::IntIndex>(&n)) {
    return EmitExpr(*ii->value);
  }

  if (auto *il = dyn_cast<AST::IntLiteral>(&n)) {
    auto ty = mlir::IntegerType::get(&IRContext(), 32,
                                     mlir::IntegerType::Signless);
    int64_t val = std::visit([](auto v) -> int64_t { return v; }, il->value);
    return builder.create<mlir::arith::ConstantIntOp>(loc, val, ty);
  }

  if (auto *fl = dyn_cast<AST::FloatLiteral>(&n)) {
    auto ty = mlir::Float32Type::get(&IRContext());
    float val = std::visit([](auto v) -> float { return static_cast<float>(v); },
                           fl->value);
    return builder.create<mlir::arith::ConstantFloatOp>(
        loc, llvm::APFloat(val), ty);
  }

  if (auto *da = dyn_cast<AST::DataAccess>(&n)) {
    if (da->AccessElement()) {
      auto tensorVal = LookupValue(da->GetDataName());
      if (!tensorVal) return nullptr;
      auto tty = tensorVal.getType().dyn_cast<coir::TensorType>();
      if (!tty) return nullptr;

      llvm::SmallVector<mlir::Value> idxVals;
      for (auto &idx : da->GetIndices()) {
        mlir::Value v = EmitExpr(*idx);
        if (!v) {
          if (auto *idNode = dyn_cast<AST::Identifier>(idx.get()))
            v = LookupValue(idNode->name);
        }
        if (v && !v.getType().isa<mlir::IndexType>())
          v = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IndexType::get(&IRContext()), v);
        if (v) idxVals.push_back(v);
      }

      return builder.create<coir::TensorLoadElemOp>(
          loc, tty.getElementType(), tensorVal, idxVals);
    } else {
      return LookupValue(da->GetDataName());
    }
  }

  if (auto *expr = dyn_cast<AST::Expr>(&n)) {
    if (expr->GetForm() == AST::Expr::Reference)
      return EmitExpr(*expr->GetR());

    if (expr->GetForm() == AST::Expr::Binary) {
      auto lhs = EmitExpr(*expr->GetL());
      auto rhs = EmitExpr(*expr->GetR());
      if (!lhs || !rhs) return nullptr;

      auto resTy = lhs.getType();
      bool isFloat = resTy.isa<mlir::FloatType>();
      auto op = expr->GetOp();

      if (op == Op::Add)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::AddFOp>(loc, lhs,
                                                                      rhs)
                   : (mlir::Value)builder.create<mlir::arith::AddIOp>(loc, lhs,
                                                                      rhs);
      if (op == Op::Sub)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::SubFOp>(loc, lhs,
                                                                      rhs)
                   : (mlir::Value)builder.create<mlir::arith::SubIOp>(loc, lhs,
                                                                      rhs);
      if (op == Op::Mul)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::MulFOp>(loc, lhs,
                                                                      rhs)
                   : (mlir::Value)builder.create<mlir::arith::MulIOp>(loc, lhs,
                                                                      rhs);
      if (op == Op::Div)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::DivFOp>(loc, lhs,
                                                                      rhs)
                   : (mlir::Value)builder.create<mlir::arith::DivSIOp>(loc, lhs,
                                                                       rhs);
    }

    if (expr->GetForm() == AST::Expr::Unary) {
      auto operand = EmitExpr(*expr->GetR());
      if (!operand) return nullptr;
      auto op = expr->GetOp();
      if (op == Op::Sub) {
        bool isFloat = operand.getType().isa<mlir::FloatType>();
        if (isFloat) {
          auto zero = builder.create<mlir::arith::ConstantFloatOp>(
              loc, llvm::APFloat(0.0f), operand.getType().cast<mlir::FloatType>());
          return builder.create<mlir::arith::SubFOp>(loc, zero, operand);
        } else {
          auto zero = builder.create<mlir::arith::ConstantIntOp>(
              loc, 0, operand.getType());
          return builder.create<mlir::arith::SubIOp>(loc, zero, operand);
        }
      }
    }
  }

  return nullptr;
}

bool ASTCoIRGen::Visit(AST::Assignment &asgn) {
  if (!asgn.AssignToDataElement()) {
    if (asgn.IsDecl()) {
      auto rhs = EmitExpr(*asgn.value);
      if (rhs)
        MapValue(asgn.GetName(), rhs);
    }
    return true;
  }

  auto loc = Loc(asgn);
  auto &da = *asgn.da;
  auto tensorVal = LookupValue(da.GetDataName());
  if (!tensorVal) return true;

  auto tty = tensorVal.getType().dyn_cast<coir::TensorType>();
  if (!tty) return true;

  auto rhs = EmitExpr(*asgn.value);
  if (!rhs) return true;

  llvm::SmallVector<mlir::Value> idxVals;
  for (auto &idx : da.GetIndices()) {
    mlir::Value v = EmitExpr(*idx);
    if (!v) {
      if (auto *idNode = dyn_cast<AST::Identifier>(idx.get()))
        v = LookupValue(idNode->name);
    }
    if (v && !v.getType().isa<mlir::IndexType>())
      v = builder.create<mlir::arith::IndexCastOp>(
          loc, mlir::IndexType::get(&IRContext()), v);
    if (v) idxVals.push_back(v);
  }

  builder.create<coir::TensorStoreElemOp>(loc, rhs, tensorVal, idxVals);
  return true;
}

bool ASTCoIRGen::Visit(AST::Return &ret) {
  auto loc = Loc(ret);
  llvm::SmallVector<mlir::Value> retVals;
  if (ret.value) {
    auto val = EmitExpr(*ret.value);
    if (!val) {
      if (auto *id = dyn_cast<AST::Identifier>(ret.value.get()))
        val = LookupValue(id->name);
    }
    if (val) retVals.push_back(val);
  }
  builder.create<coir::KernelReturnOp>(loc, retVals);
  return true;
}

bool ASTCoIRGen::Visit(AST::NamedVariableDecl &nvd) {
  auto symType = GetSymbolType(nvd.GetName());
  if (!symType) return true;

  if (auto sty = dyn_cast<SpannedType>(symType)) {
    auto loc = Loc(nvd);
    auto tty = LowerSpannedType(sty);
    auto allocOp = builder.create<coir::TensorAllocOp>(loc, tty);
    MapValue(nvd.GetName(), allocOp.getResult());
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::DMA &dma) {
  if (dma.IsDummy()) return true;

  auto loc = Loc(dma);

  mlir::Value srcVal = nullptr;
  mlir::Value dstVal = nullptr;

  if (auto *srcChunk = dyn_cast<AST::ChunkAt>(dma.from.get())) {
    srcVal = LookupValue(srcChunk->data->name);
  } else if (auto *srcId = dyn_cast<AST::Identifier>(dma.from.get())) {
    srcVal = LookupValue(srcId->name);
  } else if (auto *srcDA = dyn_cast<AST::DataAccess>(dma.from.get())) {
    srcVal = LookupValue(srcDA->GetDataName());
  }

  if (auto *dstChunk = dyn_cast<AST::ChunkAt>(dma.to.get())) {
    dstVal = LookupValue(dstChunk->data->name);
  } else if (auto *dstId = dyn_cast<AST::Identifier>(dma.to.get())) {
    dstVal = LookupValue(dstId->name);
  } else if (auto *dstDA = dyn_cast<AST::DataAccess>(dma.to.get())) {
    dstVal = LookupValue(dstDA->GetDataName());
  } else if (isa<AST::Memory>(dma.to.get())) {
    // "=> local" pattern: create a local tensor alloc for the destination
    if (srcVal) {
      auto srcTy = srcVal.getType().dyn_cast<coir::TensorType>();
      if (srcTy) {
        auto localTy = coir::TensorType::get(
            &IRContext(), srcTy.getElementType(), srcTy.getShape(),
            static_cast<int32_t>(coir::TensorMemorySpace::Local),
            srcTy.getStrides());
        auto allocOp = builder.create<coir::TensorAllocOp>(loc, localTy);
        dstVal = allocOp.getResult();
      }
    }
  }

  if (!srcVal || !dstVal) return true;

  bool isAsync = dma.IsAsync();
  auto copyOp = builder.create<coir::DataCopyOp>(
      loc, isAsync ? mlir::TypeRange{coir::AsyncTokenType::get(&IRContext())}
                   : mlir::TypeRange{},
      srcVal, dstVal, isAsync ? builder.getUnitAttr() : mlir::UnitAttr{});

  if (!dma.future.empty()) {
    if (isAsync && copyOp.getToken())
      MapValue(dma.future, copyOp.getToken());
    else if (dstVal)
      MapValue(dma.future, dstVal);
  }

  return true;
}

bool ASTCoIRGen::Visit(AST::MMA &n) {
  auto loc = Loc(n);
  auto &op = *n.GetOperation();

  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    auto fillVal = EmitExpr(*op.FillingValue());
    if (!fillVal) return true;

    auto fillType = op.FillingType();
    auto elemTy = (fillType != BaseType::UNKSCALAR)
                      ? LowerBaseType(fillType)
                      : fillVal.getType();

    auto nodeType = n.GetType();
    llvm::SmallVector<int64_t> shape;
    if (auto sty = dyn_cast<SpannedType>(nodeType)) {
      auto &mdspan = sty->s_type->value;
      if (mdspan.IsValid())
        for (auto &v : mdspan.Value())
          shape.push_back(EvalToInt(v));
    }
    if (shape.empty())
      shape = {16, 16};

    auto fragTy = coir::MMAFragType::get(&IRContext(), elemTy, shape);

    if (fillVal.getType() != elemTy) {
      if (elemTy.isa<mlir::FloatType>() && fillVal.getType().isa<mlir::FloatType>())
        fillVal = builder.create<mlir::arith::ExtFOp>(loc, elemTy, fillVal);
      else if (elemTy.isa<mlir::FloatType>())
        fillVal = builder.create<mlir::arith::SIToFPOp>(loc, elemTy, fillVal);
    }

    auto fillOp = builder.create<coir::MMAFillOp>(loc, fragTy, fillVal);
    std::string fragName = AST::FragName(op.FillingTo());
    MapValue(fragName, fillOp.getResult());
    break;
  }

  case AST::MMAOperation::Load:
  case AST::MMAOperation::LoadR: {
    auto chunkAt = op.LoadFrom();
    if (!chunkAt) return true;

    auto srcVal = LookupValue(chunkAt->data->name);
    if (!srcVal) return true;

    auto srcTy = srcVal.getType().dyn_cast<coir::TensorType>();
    if (!srcTy) return true;

    llvm::SmallVector<mlir::Value> idxVals;
    if (chunkAt->HasOperation()) {
      for (auto &sop : chunkAt->AllOperations()) {
        if (auto indices = sop->GetIndices()) {
          for (auto &idx : indices->AllValues()) {
            if (auto *id = dyn_cast<AST::Identifier>(idx.get())) {
              auto v = LookupValue(id->name);
              if (v && !v.getType().isa<mlir::IndexType>())
                v = builder.create<mlir::arith::IndexCastOp>(
                    loc, mlir::IndexType::get(&IRContext()), v);
              if (v) idxVals.push_back(v);
            } else {
              mlir::Value v = EmitExpr(*idx);
              if (v && !v.getType().isa<mlir::IndexType>())
                v = builder.create<mlir::arith::IndexCastOp>(
                    loc, mlir::IndexType::get(&IRContext()), v);
              if (v) idxVals.push_back(v);
            }
          }
        }
      }
    } else if (chunkAt->indices) {
      for (auto &idx : chunkAt->indices->AllValues()) {
        mlir::Value v = EmitExpr(*idx);
        if (v && !v.getType().isa<mlir::IndexType>())
          v = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IndexType::get(&IRContext()), v);
        if (v) idxVals.push_back(v);
      }
    }

    auto nodeType = n.GetType();
    llvm::SmallVector<int64_t> tileShape;
    if (auto sty = dyn_cast<SpannedType>(nodeType)) {
      auto &mdspan = sty->s_type->value;
      if (mdspan.IsValid())
        for (auto &v : mdspan.Value())
          tileShape.push_back(EvalToInt(v));
    }
    if (tileShape.empty())
      tileShape = {16, 16};

    auto tileTy = coir::TensorType::get(
        &IRContext(), srcTy.getElementType(), tileShape,
        static_cast<int32_t>(coir::TensorMemorySpace::Shared),
        llvm::ArrayRef<int64_t>{});
    auto tileOp = builder.create<coir::TensorTileOp>(
        loc, tileTy, srcVal, idxVals);

    auto fragTy = coir::MMAFragType::get(
        &IRContext(), srcTy.getElementType(), tileShape);
    auto loadOp = builder.create<coir::MMALoadOp>(
        loc, fragTy, tileOp.getResult());

    std::string fragName;
    if (op.IsLoadR()) {
      if (auto loadTo = op.LoadTo())
        fragName = AST::FragName(loadTo);
    } else {
      auto future = op.LoadTo();
      if (future)
        fragName = AST::FragName(future);
    }
    if (!fragName.empty())
      MapValue(fragName, loadOp.getResult());
    break;
  }

  case AST::MMAOperation::Exec: {
    auto accExpr = op.ExecOperand(0);
    auto lhsExpr = op.ExecOperand(1);
    auto rhsExpr = op.ExecOperand(2);
    if (!accExpr || !lhsExpr || !rhsExpr) return true;

    auto accVal = LookupValue(AST::FragName(accExpr));
    auto lhsVal = LookupValue(AST::FragName(lhsExpr));
    auto rhsVal = LookupValue(AST::FragName(rhsExpr));
    if (!accVal || !lhsVal || !rhsVal) return true;

    coir::MMALayout layout;
    switch (op.GetMethod()) {
    case AST::MMAOperation::ROW_ROW: layout = coir::MMALayout::RowRow; break;
    case AST::MMAOperation::ROW_COL: layout = coir::MMALayout::RowCol; break;
    case AST::MMAOperation::COL_ROW: layout = coir::MMALayout::ColRow; break;
    case AST::MMAOperation::COL_COL: layout = coir::MMALayout::ColCol; break;
    }
    auto layoutAttr = coir::MMALayoutAttr::get(&IRContext(), layout);

    auto execOp = builder.create<coir::MMAExecOp>(
        loc, accVal.getType(), accVal, lhsVal, rhsVal, layoutAttr);

    std::string accName = AST::FragName(accExpr);
    MapValue(accName, execOp.getResult());
    break;
  }

  case AST::MMAOperation::Store: {
    auto srcExpr = op.StoreFrom();
    auto dstChunk = op.StoreTo();
    if (!srcExpr || !dstChunk) return true;

    auto fragVal = LookupValue(AST::FragName(srcExpr));
    if (!fragVal) return true;

    auto dstVal = LookupValue(dstChunk->data->name);
    if (!dstVal) return true;

    auto dstTy = dstVal.getType().dyn_cast<coir::TensorType>();
    if (!dstTy) return true;

    llvm::SmallVector<mlir::Value> idxVals;
    if (dstChunk->HasOperation()) {
      for (auto &sop : dstChunk->AllOperations()) {
        if (auto indices = sop->GetIndices()) {
          for (auto &idx : indices->AllValues()) {
            mlir::Value v = EmitExpr(*idx);
            if (v && !v.getType().isa<mlir::IndexType>())
              v = builder.create<mlir::arith::IndexCastOp>(
                  loc, mlir::IndexType::get(&IRContext()), v);
            if (v) idxVals.push_back(v);
          }
        }
      }
    } else if (dstChunk->indices) {
      for (auto &idx : dstChunk->indices->AllValues()) {
        mlir::Value v = EmitExpr(*idx);
        if (v && !v.getType().isa<mlir::IndexType>())
          v = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IndexType::get(&IRContext()), v);
        if (v) idxVals.push_back(v);
      }
    }

    auto fragTy = fragVal.getType().cast<coir::MMAFragType>();
    auto tileTy = coir::TensorType::get(
        &IRContext(), dstTy.getElementType(), fragTy.getShape(),
        dstTy.getMemorySpace(), llvm::ArrayRef<int64_t>{});
    auto tileOp = builder.create<coir::TensorTileOp>(
        loc, tileTy, dstVal, idxVals);

    builder.create<coir::MMAStoreOp>(loc, fragVal, tileOp.getResult());
    break;
  }

  default:
    break;
  }

  return true;
}

bool ASTCoIRGen::Visit(AST::CppSourceCode &n) {
  if (n.kind == AST::CppSourceCode::Host ||
      n.kind == AST::CppSourceCode::None) {
    auto existing = IRModule()->getAttrOfType<mlir::StringAttr>("coir.host_code");
    std::string combined = existing ? existing.getValue().str() : "";
    combined += n.GetCode();
    IRModule()->setAttr("coir.host_code",
                        mlir::StringAttr::get(&IRContext(), combined));
  }
  return true;
}
