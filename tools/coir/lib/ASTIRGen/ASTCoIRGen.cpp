// ast_coir_gen.cpp -- AST-to-CoIR MLIR translation
#include "ASTCoIRGen.hpp"
#include "codegen_utils.hpp"
#include "symbexpr.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace Choreo;
using namespace CoIR;

namespace {

int64_t evalToInt(const ValueItem &vi) {
  if (!vi || !vi->IsNumeric()) return 0;
  if (auto *nv = dyn_cast<sbe::NumericValue>(vi.get()))
    return nv->Value();
  return 0;
}

} // namespace

mlir::Type CoIRTranslate::translateBaseType(BaseType bt) {
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
    return mlir::IntegerType::get(&ctx, 32, mlir::IntegerType::Signed);
  case BaseType::BOOL: return mlir::IntegerType::get(&ctx, 1);
  default: return mlir::Float32Type::get(&ctx);
  }
}

coir::TensorType
CoIRTranslate::translateSpannedType(const ptr<SpannedType> &sty) {
  auto elemType = translateBaseType(sty->ElementType());
  auto mdspan = sty->s_type;
  auto &shape = mdspan->value;
  llvm::SmallVector<int64_t> dims;
  if (shape.IsValid())
    for (auto &v : shape.Value())
      dims.push_back(evalToInt(v));

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
  return coir::TensorType::get(&IRContext(), elemType, dims, memSpace);
}

coir::ParallelLevelAttr
CoIRTranslate::translateParallelLevel(ParallelLevel pl) {
  coir::ParallelLevel level;
  switch (pl) {
  case ParallelLevel::BLOCK:   level = coir::ParallelLevel::Block; break;
  case ParallelLevel::GROUP:   level = coir::ParallelLevel::Warp; break;
  case ParallelLevel::GROUPx4: level = coir::ParallelLevel::Warpgroup; break;
  case ParallelLevel::THREAD:  level = coir::ParallelLevel::Thread; break;
  default: level = coir::ParallelLevel::Block; break;
  }
  return coir::ParallelLevelAttr::get(&IRContext(), level);
}

bool CoIRTranslate::BeforeVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n))
    pushScope();
  else if (isa<AST::ChoreoFunction>(&n))
    pushScope();
  else if (isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n) ||
           isa<AST::WithBlock>(&n))
    pushScope();
  return true;
}

bool CoIRTranslate::InMidVisitImpl(AST::Node &) { return true; }

bool CoIRTranslate::AfterVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    popScope();
    outs() << IRModule() << "\n";
  } else if (isa<AST::ChoreoFunction>(&n)) {
    popScope();
  } else if (isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n) ||
             isa<AST::WithBlock>(&n)) {
    popScope();
  }
  return true;
}

bool CoIRTranslate::Visit(AST::Program &) { return true; }

bool CoIRTranslate::Visit(AST::ChoreoFunction &cf) {
  auto loc = Loc(cf);
  auto fty = cast<Choreo::FunctionType>(GetSymbolType(fname));

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (auto &inTy : fty->in_tys) {
    if (auto sty = dyn_cast<SpannedType>(inTy.get()))
      argTypes.push_back(translateSpannedType(
          std::dynamic_pointer_cast<SpannedType>(inTy)));
    else
      argTypes.push_back(translateBaseType(inTy->GetBaseType()));
  }

  if (fty->out_ty && fty->out_ty->GetBaseType() != BaseType::VOID) {
    if (auto sty = dyn_cast<SpannedType>(fty->out_ty.get()))
      resultTypes.push_back(translateSpannedType(
          std::dynamic_pointer_cast<SpannedType>(fty->out_ty)));
    else
      resultTypes.push_back(translateBaseType(fty->out_ty->GetBaseType()));
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
  return true;
}

bool CoIRTranslate::Visit(AST::FunctionDecl &) { return true; }

bool CoIRTranslate::Visit(AST::ParallelBy &pb) {
  auto loc = Loc(pb);
  auto levelAttr = translateParallelLevel(pb.GetLevel());

  llvm::SmallVector<int64_t> bounds;
  if (pb.HasSubPVs()) {
    for (auto &bnd : pb.AllBoundExprs())
      if (auto *expr = dyn_cast<AST::Expr>(bnd.get()))
        if (expr->Opts().HasVal())
          bounds.push_back(evalToInt(expr->Opts().GetVal()));
  } else {
    auto bv = pb.BoundValue();
    bounds.push_back(evalToInt(bv));
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
        mapValue(id->name, arg);
    }
  } else {
    auto arg = block->addArgument(indexType, loc);
    mapValue(pb.BPV()->name, arg);
  }

  builder.setInsertionPointToEnd(block);
  return true;
}

bool CoIRTranslate::Visit(AST::ForeachBlock &fb) {
  auto loc = Loc(fb);
  auto indexType = mlir::IndexType::get(&IRContext());

  int64_t bound = 1;
  if (fb.ranges && !fb.ranges->values.empty()) {
    auto &firstRange = fb.ranges->values[0];
    if (auto *lr = dyn_cast<AST::LoopRange>(firstRange.get())) {
      if (lr->ubound) {
        if (auto *expr = dyn_cast<AST::Expr>(lr->ubound.get()))
          if (expr->Opts().HasVal())
            bound = evalToInt(expr->Opts().GetVal());
      }
    }
  }

  auto ubConst = builder.create<mlir::arith::ConstantIndexOp>(loc, bound);

  auto foreachOp = builder.create<coir::ForeachOp>(
      loc, mlir::TypeRange{}, ubConst.getResult(), mlir::ValueRange{});

  auto &bodyRegion = foreachOp.getBody();
  auto *block = new mlir::Block();
  bodyRegion.push_back(block);

  auto ivArg = block->addArgument(indexType, loc);

  if (fb.ranges && !fb.ranges->values.empty()) {
    auto &firstRange = fb.ranges->values[0];
    if (auto *lr = dyn_cast<AST::LoopRange>(firstRange.get()))
      if (lr->iv)
        mapValue(lr->iv->name, ivArg);
  }

  builder.setInsertionPointToEnd(block);
  builder.create<coir::YieldOp>(loc, mlir::ValueRange{});
  builder.setInsertionPoint(block->getTerminator());

  return true;
}

bool CoIRTranslate::Visit(AST::Assignment &) { return true; }
bool CoIRTranslate::Visit(AST::Return &) { return true; }
bool CoIRTranslate::Visit(AST::NamedVariableDecl &) { return true; }
