// ast_coir_gen.cpp -- AST-to-CoIR MLIR translation
#include "ASTCoIRGen.hpp"
#include "codegen_utils.hpp"
#include "context.hpp"
#include "dmaconf.hpp"
#include "symbexpr.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/StringSet.h"

using namespace Choreo;
using namespace CoIR;

void ASTCoIRGen::EmitAssert(mlir::Location loc, mlir::Value condition,
                            llvm::StringRef message, coir::AssertSite site,
                            coir::AssertUsage usage) {
  auto siteAttr = coir::AssertSiteAttr::get(&IRContext(), site);
  auto usageAttr = coir::AssertUsageAttr::get(&IRContext(), usage);
  builder.create<coir::AssertOp>(loc, condition,
                                 builder.getStringAttr(message),
                                 siteAttr, usageAttr,
                                 mlir::IntegerAttr{},
                                 coir::AssertCostAttr{},
                                 mlir::BoolAttr{});
}

void ASTCoIRGen::BuildAssertionMap() {
  assert_map_.clear();
  for (const auto& ar : FCtx(fname).GetAssessor().GetAssertions())
    assert_map_[ar.node].push_back(&ar);
}

mlir::Value ASTCoIRGen::MaterializeSBE(mlir::Location loc,
                                        const ValueItem& expr) {
  using namespace sbe;
  auto indexType = mlir::IndexType::get(&IRContext());

  if (auto nv = dyn_cast<NumericValue>(expr))
    return builder.create<mlir::arith::ConstantIndexOp>(loc, nv->Value());

  if (auto bv = dyn_cast<BooleanValue>(expr)) {
    auto i1Type = mlir::IntegerType::get(&IRContext(), 1);
    return builder.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(i1Type, bv->Value() ? 1 : 0));
  }

  if (auto sv = dyn_cast<SymbolicValue>(expr)) {
    auto val = LookupValue(UnScopedName(sv->Value()));
    if (!val) return nullptr;
    if (!mlir::isa<mlir::IndexType>(val.getType()))
      val = builder.create<mlir::arith::IndexCastOp>(loc, indexType, val);
    return val;
  }

  if (auto bo = dyn_cast<BinaryOperation>(expr)) {
    auto lhs = MaterializeSBE(loc, bo->GetLeft());
    auto rhs = MaterializeSBE(loc, bo->GetRight());
    if (!lhs || !rhs) return nullptr;

    if (!mlir::isa<mlir::IndexType>(lhs.getType()))
      lhs = builder.create<mlir::arith::IndexCastOp>(loc, indexType, lhs);
    if (!mlir::isa<mlir::IndexType>(rhs.getType()))
      rhs = builder.create<mlir::arith::IndexCastOp>(loc, indexType, rhs);

    switch (bo->GetOpCode()) {
    case OpCode::ADD:
      return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    case OpCode::SUBTRACT:
      return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    case OpCode::MULTIPLY:
      return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    case OpCode::DIVIDE:
      return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    case OpCode::IRES:
      return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
    case OpCode::GE:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
    case OpCode::GT:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
    case OpCode::LT:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
    case OpCode::LE:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
    case OpCode::EQ:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
    case OpCode::NE:
      return builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
    case OpCode::AND: {
      auto i1 = mlir::IntegerType::get(&IRContext(), 1);
      auto l1 = builder.create<mlir::arith::TruncIOp>(loc, i1, lhs);
      auto r1 = builder.create<mlir::arith::TruncIOp>(loc, i1, rhs);
      return builder.create<mlir::arith::AndIOp>(loc, l1, r1);
    }
    case OpCode::OR: {
      auto i1 = mlir::IntegerType::get(&IRContext(), 1);
      auto l1 = builder.create<mlir::arith::TruncIOp>(loc, i1, lhs);
      auto r1 = builder.create<mlir::arith::TruncIOp>(loc, i1, rhs);
      return builder.create<mlir::arith::OrIOp>(loc, l1, r1);
    }
    default:
      return nullptr;
    }
  }

  return nullptr;
}

void ASTCoIRGen::EmitNodeAssertions(AST::Node* node) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = assert_map_.find(node);
  if (it == assert_map_.end()) return;

  for (auto* ar : it->second) {
    auto loc = ToLoc(ar->loc);
    auto predicate = MaterializeSBE(loc, ar->expr);
    if (!predicate) continue;

    auto site = coir::AssertSite::USE;
    auto usage = coir::AssertUsage::UNCLASSIFIED;
    switch (ar->usage_type) {
    case UsageType::ShapeCompatibility:
      usage = coir::AssertUsage::SHAPE_COMPAT; break;
    case UsageType::ElementAccess:
      usage = coir::AssertUsage::ELEMENT_ACCESS; break;
    case UsageType::LoopBound:
      usage = coir::AssertUsage::LOOP_BOUND; break;
    case UsageType::HardwareConstraint:
      usage = coir::AssertUsage::HW_CONSTRAINT; break;
    case UsageType::UnClassified:
      usage = coir::AssertUsage::UNCLASSIFIED; break;
    }

    EmitAssert(loc, predicate, ar->message, site, usage);
  }
}

int64_t ASTCoIRGen::ResolveBoundedVarExtent(llvm::StringRef rvName) {
  auto scoped = InScopeName(rvName.str());
  auto it = bv_map.find(scoped);
  if (it == bv_map.end()) return 0;

  int64_t total = 1;
  for (auto &mappedName : it->second) {
    // First try: look up the MLIR value by the mapped unscoped name
    auto lookupName = UnScopedName(mappedName);
    auto val = LookupValue(lookupName);
    if (!val) val = LookupValue(lookupName + ".data");
    if (val) {
      if (auto tty = val.getType().dyn_cast<coir::TensorType>()) {
        for (auto d : tty.getShape())
          if (d > 0) total *= d;
        continue;
      }
    }
    // Second try: BoundedType from the symbol table
    auto symType = GetSymbolType(UnScopedName(mappedName));
    if (auto bty = dyn_cast<BoundedType>(symType)) {
      if (bty->HasValidBound()) {
        auto &ub = bty->GetUpperBound();
        if (auto v = VIInt(ub)) { total *= *v; continue; }
      }
    }
  }
  return total;
}

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
  case BaseType::U8:
  case BaseType::S8:
    return mlir::IntegerType::get(&ctx, 8);
  case BaseType::U16:
  case BaseType::S16:
    return mlir::IntegerType::get(&ctx, 16);
  case BaseType::U32:
  case BaseType::S32:
    return mlir::IntegerType::get(&ctx, 32);
  case BaseType::S64:
  case BaseType::U64:
    return mlir::IntegerType::get(&ctx, 64);
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
  case Storage::DEFAULT:
    break;
  case Storage::GLOBAL:
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
    BuildAssertionMap();
  } else if (isa<AST::ParallelBy>(&n) || isa<AST::ForeachBlock>(&n) ||
             isa<AST::WithBlock>(&n)) {
    PushScope();
  }
  EmitNodeAssertions(&n);
  return true;
}

bool ASTCoIRGen::InMidVisitImpl(AST::Node &n) {
  if (isa<AST::IfElseBlock>(&n) && !ifMergeStack.empty()) {
    auto &info = ifMergeStack.back();
    auto ifOp = info.ifOp;
    auto &thenBlock = ifOp.getThenRegion().front();
    builder.setInsertionPointToEnd(&thenBlock);

    llvm::SmallVector<mlir::Value> yieldVals;
    for (unsigned i = 0; i < info.modifiedNames.size(); ++i) {
      auto val = LookupValue(info.modifiedNames[i]);
      yieldVals.push_back(val ? val : info.preIfValues[i]);
    }

    if (!thenBlock.empty() &&
        thenBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
      thenBlock.back().erase();
    builder.setInsertionPointToEnd(&thenBlock);
    builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), yieldVals);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  }
  return true;
}

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
  } else if (isa<AST::IfElseBlock>(&n) && !ifMergeStack.empty()) {
    auto info = ifMergeStack.pop_back_val();
    auto ifOp = info.ifOp;
    auto &elseBlock = ifOp.getElseRegion().front();
    builder.setInsertionPointToEnd(&elseBlock);

    llvm::SmallVector<mlir::Value> yieldVals;
    auto *ifElse = dyn_cast<AST::IfElseBlock>(&n);
    if (ifElse && ifElse->HasElse()) {
      for (auto &name : info.modifiedNames) {
        auto val = LookupValue(name);
        yieldVals.push_back(val ? val : info.preIfValues[yieldVals.size()]);
      }
    } else {
      yieldVals = info.preIfValues;
    }

    if (!elseBlock.empty() &&
        elseBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
      elseBlock.back().erase();
    builder.setInsertionPointToEnd(&elseBlock);
    builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), yieldVals);

    builder.setInsertionPointAfter(ifOp);

    for (unsigned i = 0; i < info.modifiedNames.size(); ++i)
      UpdateValue(info.modifiedNames[i], ifOp.getResult(i));
  } else if (isa<AST::WhileBlock>(&n) && !whileMergeStack.empty()) {
    auto info = whileMergeStack.pop_back_val();

    if (info.isCoirWhile) {
      auto coirWhile = mlir::cast<coir::CoIRWhileOp>(info.whileOp);
      auto &bodyBlock = coirWhile.getBodyRegion().front();
      builder.setInsertionPointToEnd(&bodyBlock);

      // If body doesn't end with a terminator, add coir.continue
      if (bodyBlock.empty() ||
          !bodyBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        llvm::SmallVector<mlir::Value> yieldVals;
        for (auto &name : info.iterNames) {
          auto val = LookupValue(name);
          if (val) yieldVals.push_back(val);
        }
        builder.create<coir::CoIRContinueOp>(builder.getUnknownLoc(),
                                              yieldVals);
      }

      PopScope();
      builder.setInsertionPointAfter(coirWhile);
      for (unsigned i = 0; i < info.iterNames.size(); ++i)
        UpdateValue(info.iterNames[i], coirWhile.getResult(i));
    } else {
      auto scfWhile = mlir::cast<mlir::scf::WhileOp>(info.whileOp);
      auto &afterBlock = scfWhile.getAfter().front();
      builder.setInsertionPointToEnd(&afterBlock);

      llvm::SmallVector<mlir::Value> yieldVals;
      for (auto &name : info.iterNames) {
        auto val = LookupValue(name);
        yieldVals.push_back(val);
      }

      if (!afterBlock.empty() &&
          afterBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
        afterBlock.back().erase();
      builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), yieldVals);

      PopScope();
      builder.setInsertionPointAfter(scfWhile);
      for (unsigned i = 0; i < info.iterNames.size(); ++i)
        UpdateValue(info.iterNames[i], scfWhile.getResult(i));
    }
  } else if (isa<AST::WithBlock>(&n)) {
    PopScope();
  } else if (isa<AST::InThreadsBlock>(&n)) {
    auto *block = builder.getInsertionBlock();
    auto *parentOp = block->getParentOp();
    if (parentOp && parentOp->getBlock())
      builder.setInsertionPointAfter(parentOp);
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

  // Attach host-facing metadata so downstream stub generation can
  // reconstruct the Choreo calling convention (spanned_view / spanned_data,
  // auto-memory-shadowing, pass-by-ref copy-back).
  llvm::SmallVector<mlir::Attribute> paramNames, paramAttrs, paramRefs,
      hostElemTypes;
  llvm::SmallVector<mlir::Attribute> paramDims;
  if (cf.f_decl.params) {
    for (auto &p : cf.f_decl.params->values) {
      paramNames.push_back(mlir::StringAttr::get(
          &IRContext(), p->HasSymbol() ? p->sym->name : ""));
      paramAttrs.push_back(
          builder.getI32IntegerAttr(static_cast<int>(p->attr)));
      paramRefs.push_back(builder.getBoolAttr(p->pass_by_ref));
      if (auto sty = dyn_cast<SpannedType>(p->type->GetType())) {
        hostElemTypes.push_back(mlir::StringAttr::get(
            &IRContext(), STR(sty->ElementType())));
        paramDims.push_back(
            builder.getI64IntegerAttr(static_cast<int64_t>(sty->Dims())));
      } else {
        hostElemTypes.push_back(mlir::StringAttr::get(&IRContext(), ""));
        paramDims.push_back(builder.getI64IntegerAttr(0));
      }
    }
  }
  kernelOp->setAttr("coir.param_names",
                     mlir::ArrayAttr::get(&IRContext(), paramNames));
  kernelOp->setAttr("coir.param_attrs",
                     mlir::ArrayAttr::get(&IRContext(), paramAttrs));
  kernelOp->setAttr("coir.param_refs",
                     mlir::ArrayAttr::get(&IRContext(), paramRefs));
  kernelOp->setAttr("coir.host_elem_types",
                     mlir::ArrayAttr::get(&IRContext(), hostElemTypes));
  kernelOp->setAttr("coir.param_dims",
                     mlir::ArrayAttr::get(&IRContext(), paramDims));

  // Output type metadata for stub return signature.
  if (fty->out_ty && fty->out_ty->GetBaseType() != BaseType::VOID) {
    if (auto sty = dyn_cast<SpannedType>(fty->out_ty)) {
      kernelOp->setAttr("coir.host_ret_elem_type",
                         mlir::StringAttr::get(&IRContext(),
                                               STR(sty->ElementType())));
      kernelOp->setAttr("coir.host_ret_dims",
                         builder.getI64IntegerAttr(
                             static_cast<int64_t>(sty->Dims())));
    }
  }

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

void collectScalarIterArgs(AST::Node *node,
                           llvm::SmallVectorImpl<std::string> &names,
                           llvm::StringSet<> &seen) {
  if (!node) return;
  if (auto *asgn = dyn_cast<AST::Assignment>(node)) {
    if (!asgn->AssignToDataElement() && !asgn->IsDecl()) {
      auto &name = asgn->GetName();
      if (!seen.count(name)) {
        names.push_back(name);
        seen.insert(name);
      }
    }
    return;
  }
  if (auto *mn = dyn_cast<AST::MultiNodes>(node)) {
    for (auto &child : mn->values)
      if (child) collectScalarIterArgs(child.get(), names, seen);
    return;
  }
  if (node->HasBody()) {
    auto body = node->GetBody();
    if (body) collectScalarIterArgs(body.get(), names, seen);
  }
}

void collectIfModifiedScalars(AST::Node *thenBody, AST::Node *elseBody,
                              llvm::SmallVectorImpl<std::string> &names,
                              llvm::StringSet<> &seen) {
  if (thenBody) collectScalarIterArgs(thenBody, names, seen);
  if (elseBody) collectScalarIterArgs(elseBody, names, seen);
}

bool hasEarlyExit(AST::Node *node) {
  if (!node) return false;
  if (isa<AST::Break>(node) || isa<AST::Continue>(node)) return true;
  if (auto *mn = dyn_cast<AST::MultiNodes>(node)) {
    for (auto &child : mn->values)
      if (child && hasEarlyExit(child.get())) return true;
    return false;
  }
  if (auto *ifb = dyn_cast<AST::IfElseBlock>(node)) {
    if (ifb->stmts && hasEarlyExit(ifb->stmts.get())) return true;
    if (ifb->else_stmts && hasEarlyExit(ifb->else_stmts.get())) return true;
    return false;
  }
  if (node->HasBody()) {
    auto body = node->GetBody();
    if (body && hasEarlyExit(body.get())) return true;
  }
  return false;
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
      if (bound <= 1) {
        auto rvName = lr->GetRVName();
        auto symType = GetSymbolType(rvName);
        if (auto bty = dyn_cast<BoundedType>(symType)) {
          if (bty->HasValidBound()) {
            auto &ub = bty->GetUpperBound();
            if (ub && ub->IsNumeric())
              bound = EvalToInt(ub);
          }
        }
      }
      if (bound <= 1) {
        auto resolved = ResolveBoundedVarExtent(lr->GetRVName());
        if (resolved > 1) bound = resolved;
      }
      if (bound <= 1) {
        auto rvName = lr->GetRVName();
        auto symType = GetSymbolType(rvName);
        if (auto bty = dyn_cast<BoundedType>(symType)) {
          if (bty->HasValidBound()) {
            auto ubs = bty->GetUpperBounds();
            if (ubs.IsValid()) {
              int64_t total = 1;
              for (auto &ub : ubs.Value())
                if (ub && ub->IsNumeric()) total *= EvalToInt(ub);
              if (total > 1) bound = total;
            }
          }
        }
      }
    }
  }

  llvm::SmallVector<std::string> accNames;
  if (fb.stmts) collectMMAAccNames(fb.stmts.get(), accNames);

  llvm::SmallVector<std::string> scalarNames;
  llvm::StringSet<> scalarSeen;
  if (fb.stmts) collectScalarIterArgs(fb.stmts.get(), scalarNames, scalarSeen);

  llvm::SmallVector<mlir::Value> iterArgs;
  llvm::SmallVector<mlir::Type> iterTypes;
  llvm::SmallVector<std::string> iterNames;
  llvm::StringSet<> iterSeen;
  for (auto &name : accNames) {
    auto val = LookupValue(name);
    if (val && !iterSeen.count(name)) {
      iterArgs.push_back(val);
      iterTypes.push_back(val.getType());
      iterNames.push_back(name);
      iterSeen.insert(name);
    }
  }
  for (auto &name : scalarNames) {
    auto val = LookupValue(name);
    if (val && !iterSeen.count(name)) {
      iterArgs.push_back(val);
      iterTypes.push_back(val.getType());
      iterNames.push_back(name);
      iterSeen.insert(name);
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

  if (auto *sa = dyn_cast<AST::SpanAs>(&n)) {
    auto srcVal = LookupValue(sa->id->name);
    if (!srcVal) return nullptr;
    auto srcTy = mlir::dyn_cast<coir::TensorType>(srcVal.getType());
    if (!srcTy) return srcVal;

    llvm::SmallVector<int64_t> newShape;
    if (sa->list) {
      for (auto &v : sa->list->AllValues()) {
        if (auto *il = dyn_cast<AST::IntLiteral>(v.get())) {
          int64_t val =
              std::visit([](auto x) -> int64_t { return x; }, il->value);
          newShape.push_back(val);
        } else if (auto *expr = dyn_cast<AST::Expr>(v.get())) {
          if (expr->Opts().HasVal())
            newShape.push_back(EvalToInt(expr->Opts().GetVal()));
        }
      }
    }
    if (newShape.empty()) return srcVal;

    auto reshapedTy = coir::TensorType::get(
        &IRContext(), srcTy.getElementType(), newShape,
        srcTy.getMemorySpace(), llvm::ArrayRef<int64_t>{});
    auto tileOp = builder.create<coir::TensorTileOp>(
        loc, reshapedTy, srcVal, mlir::ValueRange{});
    return tileOp.getResult();
  }

  if (auto *da = dyn_cast<AST::DataAccess>(&n)) {
    if (da->AccessElement()) {
      auto tensorVal = LookupValue(da->GetDataName());
      if (!tensorVal) return nullptr;
      auto tty = mlir::dyn_cast<coir::TensorType>(tensorVal.getType());
      if (!tty) return nullptr;

      llvm::SmallVector<mlir::Value> idxVals;
      for (auto &idx : da->GetIndices()) {
        mlir::Value v = EmitExpr(*idx);
        if (!v) {
          if (auto *idNode = dyn_cast<AST::Identifier>(idx.get()))
            v = LookupValue(idNode->name);
        }
        if (v && !mlir::isa<mlir::IndexType>(v.getType()))
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

      if (lhs.getType() != rhs.getType()) {
        if (mlir::isa<mlir::IndexType>(lhs.getType()) &&
            mlir::isa<mlir::IntegerType>(rhs.getType()))
          rhs = builder.create<mlir::arith::IndexCastOp>(loc, lhs.getType(),
                                                          rhs);
        else if (mlir::isa<mlir::IndexType>(rhs.getType()) &&
                 mlir::isa<mlir::IntegerType>(lhs.getType()))
          lhs = builder.create<mlir::arith::IndexCastOp>(loc, rhs.getType(),
                                                          lhs);
      }

      auto resTy = lhs.getType();
      bool isFloat = mlir::isa<mlir::FloatType>(resTy);
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
      if (op == Op::Mod)
        return (mlir::Value)builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
      if (op == Op::Lt)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      if (op == Op::Gt)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      if (op == Op::Eq)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
      if (op == Op::Le)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
      if (op == Op::Ge)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
      if (op == Op::Ne)
        return isFloat
                   ? (mlir::Value)builder.create<mlir::arith::CmpFOp>(
                         loc, mlir::arith::CmpFPredicate::UNE, lhs, rhs)
                   : (mlir::Value)builder.create<mlir::arith::CmpIOp>(
                         loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
    }

    if (expr->GetForm() == AST::Expr::Ternary) {
      auto cond = EmitExpr(*expr->GetC());
      auto trueVal = EmitExpr(*expr->GetL());
      auto falseVal = EmitExpr(*expr->GetR());
      if (!cond || !trueVal || !falseVal) return nullptr;
      if (!cond.getType().isInteger(1)) {
        auto zero = builder.create<mlir::arith::ConstantIntOp>(
            loc, 0, cond.getType());
        cond = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ne, cond, zero);
      }
      return builder.create<mlir::arith::SelectOp>(loc, cond, trueVal,
                                                    falseVal);
    }

    if (expr->GetForm() == AST::Expr::Unary) {
      auto operand = EmitExpr(*expr->GetR());
      if (!operand) return nullptr;
      auto op = expr->GetOp();
      if (op == Op::Sub) {
        bool isFloat = mlir::isa<mlir::FloatType>(operand.getType());
        if (isFloat) {
          auto zero = builder.create<mlir::arith::ConstantFloatOp>(
              loc, llvm::APFloat(0.0f), mlir::cast<mlir::FloatType>(operand.getType()));
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
    auto rhs = EmitExpr(*asgn.value);
    if (rhs) {
      if (asgn.IsDecl())
        MapValue(asgn.GetName(), rhs);
      else
        UpdateValue(asgn.GetName(), rhs);
    }
    return true;
  }

  auto loc = Loc(asgn);
  auto &da = *asgn.da;
  auto tensorVal = LookupValue(da.GetDataName());
  if (!tensorVal) return true;

  auto tty = mlir::dyn_cast<coir::TensorType>(tensorVal.getType());
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
    if (v && !mlir::isa<mlir::IndexType>(v.getType()))
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
  if (lastSpanAsResult) {
    MapValue(nvd.GetName(), lastSpanAsResult);
    lastSpanAsResult = nullptr;
    return true;
  }

  auto symType = GetSymbolType(nvd.GetName());
  if (!symType) return true;

  if (auto sty = dyn_cast<SpannedType>(symType)) {
    auto loc = Loc(nvd);
    auto tty = LowerSpannedType(sty);
    auto allocOp = builder.create<coir::TensorAllocOp>(loc, tty);
    MapValue(nvd.GetName(), allocOp.getResult());
  } else {
    if (nvd.init_expr) {
      auto val = EmitExpr(*nvd.init_expr);
      if (val) MapValue(nvd.GetName(), val);
    } else if (nvd.init_value) {
      auto val = EmitExpr(*nvd.init_value);
      if (val) MapValue(nvd.GetName(), val);
    }
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::SpanAs &sa) {
  auto srcVal = LookupValue(sa.id->name);
  if (!srcVal) return true;
  auto srcTy = mlir::dyn_cast<coir::TensorType>(srcVal.getType());
  if (!srcTy) return true;

  auto loc = Loc(sa);
  llvm::SmallVector<int64_t> newShape;
  if (sa.list) {
    for (auto &v : sa.list->AllValues()) {
      if (auto *il = dyn_cast<AST::IntLiteral>(v.get())) {
        int64_t val =
            std::visit([](auto x) -> int64_t { return x; }, il->value);
        newShape.push_back(val);
      } else if (auto *expr = dyn_cast<AST::Expr>(v.get())) {
        if (expr->Opts().HasVal())
          newShape.push_back(EvalToInt(expr->Opts().GetVal()));
      }
    }
  }
  if (newShape.empty()) return true;

  auto reshapedTy = coir::TensorType::get(
      &IRContext(), srcTy.getElementType(), newShape,
      srcTy.getMemorySpace(), llvm::ArrayRef<int64_t>{});
  auto tileOp = builder.create<coir::TensorTileOp>(
      loc, reshapedTy, srcVal, mlir::ValueRange{});

  if (sa.nid && !sa.nid->name.empty())
    MapValue(sa.nid->name, tileOp.getResult());

  lastSpanAsResult = tileOp.getResult();
  return true;
}

mlir::Value ASTCoIRGen::EmitChunkAtTile(AST::ChunkAt &chunk,
                                        mlir::Value baseVal) {
  auto loc = Loc(chunk);
  auto baseTy = mlir::dyn_cast<coir::TensorType>(baseVal.getType());
  if (!baseTy) return baseVal;

  auto baseShape = baseTy.getShape();

  auto isWildcard = [](AST::Node *idx) -> bool {
    if (auto *id = dyn_cast<AST::Identifier>(idx))
      return id->name == "_" || id->name == "__choreo_no_tiling__";
    if (auto *expr = dyn_cast<AST::Expr>(idx)) {
      if (expr->GetForm() == AST::Expr::Reference) {
        auto *inner = expr->GetR().get();
        if (auto *id = dyn_cast<AST::Identifier>(inner))
          return id->name == "_" || id->name == "__choreo_no_tiling__";
      }
    }
    return false;
  };

  auto emitIdx = [&](AST::Node *idx) -> mlir::Value {
    if (auto *id = dyn_cast<AST::Identifier>(idx)) {
      auto v = LookupValue(id->name);
      if (v) {
        if (!mlir::isa<mlir::IndexType>(v.getType()))
          v = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IndexType::get(&IRContext()), v);
        return v;
      }
    }
    auto v = EmitExpr(*idx);
    if (v && !mlir::isa<mlir::IndexType>(v.getType()))
      v = builder.create<mlir::arith::IndexCastOp>(
          loc, mlir::IndexType::get(&IRContext()), v);
    return v;
  };

  // Use the pre-computed block shape from shape inference when available.
  // ChunkAt::GetBlockShape() gives the actual tile shape after applying
  // tiling operations (e.g., Chunk(4) on a [256] tensor yields [64]).
  llvm::SmallVector<int64_t> blockDims;
  if (chunk.GetBlockShape().IsValid()) {
    auto posVals = chunk.GetBlockShape().PosValList();
    if (posVals)
      for (auto d : *posVals) blockDims.push_back(static_cast<int64_t>(d));
  }

  llvm::SmallVector<mlir::Value> idxVals;
  llvm::SmallVector<int64_t> tileShape;

  if (chunk.HasOperation()) {
    for (auto &sop : chunk.AllOperations()) {
      auto indices = sop->GetIndices();
      if (!indices) continue;
      unsigned dimIdx = 0;
      for (auto &idx : indices->AllValues()) {
        if (isWildcard(idx.get())) {
          tileShape.push_back(dimIdx < baseShape.size()
                                  ? baseShape[dimIdx]
                                  : 1);
          auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          idxVals.push_back(zero);
        } else {
          auto v = emitIdx(idx.get());
          if (v) {
            idxVals.push_back(v);
            if (dimIdx < blockDims.size())
              tileShape.push_back(blockDims[dimIdx]);
            else
              tileShape.push_back(1);
          }
        }
        dimIdx++;
      }
      break;
    }
  } else if (chunk.indices) {
    unsigned dimIdx = 0;
    for (auto &idx : chunk.indices->AllValues()) {
      if (isWildcard(idx.get())) {
        tileShape.push_back(dimIdx < baseShape.size()
                                ? baseShape[dimIdx]
                                : 1);
        auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        idxVals.push_back(zero);
      } else {
        auto v = emitIdx(idx.get());
        if (v) {
          idxVals.push_back(v);
          if (dimIdx < blockDims.size())
            tileShape.push_back(blockDims[dimIdx]);
          else
            tileShape.push_back(1);
        }
      }
      dimIdx++;
    }
  }

  if (idxVals.empty()) return baseVal;

  llvm::SmallVector<int64_t> resultShape;
  for (auto d : tileShape)
    resultShape.push_back(d);
  if (resultShape.empty()) resultShape.push_back(1);

  auto tileTy = coir::TensorType::get(
      &IRContext(), baseTy.getElementType(), resultShape,
      baseTy.getMemorySpace(), llvm::ArrayRef<int64_t>{});
  return builder.create<coir::TensorTileOp>(loc, tileTy, baseVal, idxVals);
}

bool ASTCoIRGen::Visit(AST::DMA &dma) {
  if (dma.IsDummy()) return true;

  auto loc = Loc(dma);

  mlir::Value srcVal = nullptr;
  mlir::Value dstVal = nullptr;

  auto resolveDMAVal = [&](mlir::Value val, llvm::StringRef name) -> mlir::Value {
    if (val && mlir::isa<coir::AsyncTokenType>(val.getType())) {
      auto dataVal = LookupValue((name + ".data").str());
      if (dataVal) return dataVal;
    }
    return val;
  };

  if (auto *srcChunk = dyn_cast<AST::ChunkAt>(dma.from.get())) {
    auto base = LookupValue(srcChunk->data->name);
    base = resolveDMAVal(base, srcChunk->data->name);
    if (base) srcVal = EmitChunkAtTile(*srcChunk, base);
  } else if (auto *srcId = dyn_cast<AST::Identifier>(dma.from.get())) {
    srcVal = LookupValue(srcId->name);
    srcVal = resolveDMAVal(srcVal, srcId->name);
  } else if (auto *srcDA = dyn_cast<AST::DataAccess>(dma.from.get())) {
    srcVal = LookupValue(srcDA->GetDataName());
    srcVal = resolveDMAVal(srcVal, srcDA->GetDataName());
  }

  if (auto *dstChunk = dyn_cast<AST::ChunkAt>(dma.to.get())) {
    auto base = LookupValue(dstChunk->data->name);
    if (base) dstVal = EmitChunkAtTile(*dstChunk, base);
  } else if (auto *dstId = dyn_cast<AST::Identifier>(dma.to.get())) {
    dstVal = LookupValue(dstId->name);
  } else if (auto *dstDA = dyn_cast<AST::DataAccess>(dma.to.get())) {
    dstVal = LookupValue(dstDA->GetDataName());
  } else if (auto *mem = dyn_cast<AST::Memory>(dma.to.get())) {
    if (srcVal) {
      auto srcTy = mlir::dyn_cast<coir::TensorType>(srcVal.getType());
      if (srcTy) {
        auto space = coir::TensorMemorySpace::Local;
        if (mem->Get() == Storage::SHARED)
          space = coir::TensorMemorySpace::Shared;
        auto dstTy = coir::TensorType::get(
            &IRContext(), srcTy.getElementType(), srcTy.getShape(),
            static_cast<int32_t>(space), srcTy.getStrides());
        auto allocOp = builder.create<coir::TensorAllocOp>(loc, dstTy);
        dstVal = allocOp.getResult();
      }
    }
  }

  if (!srcVal || !dstVal) return true;

  auto srcTensor = srcVal.getType().dyn_cast<coir::TensorType>();
  auto dstTensor = dstVal.getType().dyn_cast<coir::TensorType>();
  if (srcTensor && dstTensor) {
    auto srcShape = srcTensor.getShape();
    auto dstShape = dstTensor.getShape();
    int64_t srcElems = 1, dstElems = 1;
    bool srcStatic = true, dstStatic = true;
    for (auto s : srcShape) {
      if (s < 0) { srcStatic = false; break; }
      srcElems *= s;
    }
    for (auto s : dstShape) {
      if (s < 0) { dstStatic = false; break; }
      dstElems *= s;
    }
    if (srcStatic && dstStatic && srcElems > dstElems) {
      auto srcC = builder.create<mlir::arith::ConstantIndexOp>(loc, srcElems);
      auto dstC = builder.create<mlir::arith::ConstantIndexOp>(loc, dstElems);
      auto cmp = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sle, srcC, dstC);
      EmitAssert(loc, cmp, "DMA copy: src elements exceed dst capacity",
                 coir::AssertSite::USE, coir::AssertUsage::SHAPE_COMPAT);
    }
  }

  bool isAsync = dma.IsAsync();

  // Map DMA kind from AST operation string.
  auto kind = coir::DMAKind::Copy;
  if (dma.operation == ".pad") kind = coir::DMAKind::Pad;
  else if (dma.operation == ".transp") kind = coir::DMAKind::Transpose;
  auto kindAttr = coir::DMAKindAttr::get(&IRContext(), kind);

  // Extract pad config.
  mlir::DenseI64ArrayAttr padLowAttr, padHighAttr;
  mlir::Attribute padValueAttr;
  if (kind == coir::DMAKind::Pad && dma.config) {
    if (dma.config->Name() == "pad") {
      auto *pc = static_cast<PadConfig*>(dma.config.get());
      auto extractIntList = [](const ptr<AST::MultiValues> &mv)
          -> llvm::SmallVector<int64_t> {
        llvm::SmallVector<int64_t> vals;
        if (!mv) return vals;
        for (auto &v : mv->AllValues()) {
          AST::Node *n = v.get();
          if (auto *expr = dyn_cast<AST::Expr>(n))
            n = expr->GetR().get();
          if (auto *lit = dyn_cast<AST::IntLiteral>(n))
            vals.push_back(
                std::visit([](auto x) -> int64_t { return x; }, lit->value));
          else
            vals.push_back(0);
        }
        return vals;
      };
      auto low = extractIntList(pc->pad_low);
      auto high = extractIntList(pc->pad_high);
      if (!low.empty())
        padLowAttr = builder.getDenseI64ArrayAttr(low);
      if (!high.empty())
        padHighAttr = builder.getDenseI64ArrayAttr(high);
      if (pc->value) {
        auto *valNode = pc->value->GetR().get();
        if (auto *intLit = dyn_cast<AST::IntLiteral>(valNode))
          padValueAttr = builder.getI64IntegerAttr(
              std::visit([](auto x) -> int64_t { return x; }, intLit->value));
        else if (auto *fLit = dyn_cast<AST::FloatLiteral>(valNode))
          padValueAttr = builder.getF64FloatAttr(
              std::visit([](auto x) -> double { return x; }, fLit->value));
      }
    }
  }

  // Extract transpose permutation.
  mlir::DenseI64ArrayAttr transpPermAttr;
  if (kind == coir::DMAKind::Transpose && dma.config) {
    if (dma.config->Name() == "transpose") {
      auto *tc = static_cast<TransposeConfig*>(dma.config.get());
      llvm::SmallVector<int64_t> perm(tc->dim_values.begin(),
                                       tc->dim_values.end());
      transpPermAttr = builder.getDenseI64ArrayAttr(perm);
    }
  }

  // Emit the appropriate copy op based on user's explicit intent.
  // TMA: user wrote `tma.copy` -> TmaCopyOp (always async).
  // DMA: user wrote `dma.copy` -> DmaCopyOp (always produces token).
  mlir::Value token = nullptr;
  if (dma.IsTMA()) {
    auto tmaCopy = builder.create<coir::TmaCopyOp>(
        loc, coir::AsyncTokenType::get(&IRContext()), srcVal, dstVal);
    token = tmaCopy.getToken();
  } else {
    auto dmaCopy = builder.create<coir::DmaCopyOp>(
        loc, coir::AsyncTokenType::get(&IRContext()), srcVal, dstVal,
        kindAttr, padLowAttr, padHighAttr, padValueAttr, transpPermAttr);
    token = dmaCopy.getToken();
  }

  if (!dma.future.empty()) {
    if (isAsync && token) {
      MapValue(dma.future, token);
      if (dstVal)
        MapValue(dma.future + ".data", dstVal);
    } else if (dstVal) {
      MapValue(dma.future, dstVal);
      MapValue(dma.future + ".data", dstVal);
    }
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
      if (mlir::isa<mlir::FloatType>(elemTy) && mlir::isa<mlir::FloatType>(fillVal.getType()))
        fillVal = builder.create<mlir::arith::ExtFOp>(loc, elemTy, fillVal);
      else if (mlir::isa<mlir::FloatType>(elemTy))
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

    auto srcTy = mlir::dyn_cast<coir::TensorType>(srcVal.getType());
    if (!srcTy) return true;

    llvm::SmallVector<mlir::Value> idxVals;
    if (chunkAt->HasOperation()) {
      for (auto &sop : chunkAt->AllOperations()) {
        if (auto indices = sop->GetIndices()) {
          for (auto &idx : indices->AllValues()) {
            if (auto *id = dyn_cast<AST::Identifier>(idx.get())) {
              auto v = LookupValue(id->name);
              if (v && !mlir::isa<mlir::IndexType>(v.getType()))
                v = builder.create<mlir::arith::IndexCastOp>(
                    loc, mlir::IndexType::get(&IRContext()), v);
              if (v) idxVals.push_back(v);
            } else {
              mlir::Value v = EmitExpr(*idx);
              if (v && !mlir::isa<mlir::IndexType>(v.getType()))
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
        if (v && !mlir::isa<mlir::IndexType>(v.getType()))
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

    coir::MMALayout layout = coir::MMALayout::RowCol;
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

    auto dstTy = mlir::dyn_cast<coir::TensorType>(dstVal.getType());
    if (!dstTy) return true;

    llvm::SmallVector<mlir::Value> idxVals;
    if (dstChunk->HasOperation()) {
      for (auto &sop : dstChunk->AllOperations()) {
        if (auto indices = sop->GetIndices()) {
          for (auto &idx : indices->AllValues()) {
            mlir::Value v = EmitExpr(*idx);
            if (v && !mlir::isa<mlir::IndexType>(v.getType()))
              v = builder.create<mlir::arith::IndexCastOp>(
                  loc, mlir::IndexType::get(&IRContext()), v);
            if (v) idxVals.push_back(v);
          }
        }
      }
    } else if (dstChunk->indices) {
      for (auto &idx : dstChunk->indices->AllValues()) {
        mlir::Value v = EmitExpr(*idx);
        if (v && !mlir::isa<mlir::IndexType>(v.getType()))
          v = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IndexType::get(&IRContext()), v);
        if (v) idxVals.push_back(v);
      }
    }

    auto fragTy = mlir::cast<coir::MMAFragType>(fragVal.getType());
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

bool ASTCoIRGen::Visit(AST::Rotate &rot) {
  auto loc = Loc(rot);
  llvm::SmallVector<std::string> names;
  llvm::SmallVector<mlir::Value> tokens;
  for (auto &node : rot.GetIds()) {
    auto *id = dyn_cast<AST::Identifier>(node.get());
    if (!id) continue;
    auto val = LookupValue(id->name);
    if (!val || !mlir::isa<coir::AsyncTokenType>(val.getType())) continue;
    names.push_back(id->name);
    tokens.push_back(val);
  }
  if (tokens.size() < 2) return true;
  llvm::SmallVector<mlir::Type> resultTypes(tokens.size(),
      coir::AsyncTokenType::get(builder.getContext()));
  auto rotateOp = builder.create<coir::FutureRotateOp>(
      loc, resultTypes, tokens);
  for (unsigned i = 0; i < names.size(); ++i)
    UpdateValue(names[i], rotateOp.getResult(i));
  return true;
}

bool ASTCoIRGen::Visit(AST::Wait &w) {
  if (!w.targets) return true;
  auto loc = Loc(w);
  for (auto &t : w.targets->AllValues()) {
    std::string name;
    std::string subscript;
    if (auto *id = dyn_cast<AST::Identifier>(t.get()))
      name = id->name;
    else if (auto *expr = dyn_cast<AST::Expr>(t.get())) {
      if (expr->op == Op::ElemOf) {
        if (auto *base = dyn_cast<AST::Identifier>(expr->GetL().get()))
          name = base->name;
        if (expr->GetR()) {
          if (auto *si = dyn_cast<AST::Identifier>(expr->GetR().get()))
            subscript = si->name;
          else if (auto *sl = dyn_cast<AST::IntLiteral>(expr->GetR().get()))
            subscript = std::to_string(std::visit(
                [](auto v) -> int64_t { return v; }, sl->value));
        }
      } else if (auto *id = dyn_cast<AST::Identifier>(expr->GetR().get()))
        name = id->name;
    }
    if (name.empty()) continue;
    auto tokenVal = LookupValue(name);
    if (tokenVal && mlir::isa<coir::AsyncTokenType>(tokenVal.getType())) {
      builder.create<coir::WaitOp>(loc, tokenVal);
    } else {
      builder.create<coir::EventWaitOp>(
          loc, builder.getStringAttr(name),
          subscript.empty() ? nullptr : builder.getStringAttr(subscript));
    }
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::Trigger &trig) {
  auto loc = Loc(trig);
  for (auto &t : trig.GetEvents()) {
    std::string name;
    std::string subscript;
    if (auto *expr = dyn_cast<AST::Expr>(t.get())) {
      if (expr->op == Op::ElemOf) {
        if (auto *base = dyn_cast<AST::Identifier>(expr->GetL().get()))
          name = base->name;
        if (expr->GetR()) {
          if (auto *si = dyn_cast<AST::Identifier>(expr->GetR().get()))
            subscript = si->name;
          else if (auto *sl = dyn_cast<AST::IntLiteral>(expr->GetR().get()))
            subscript = std::to_string(std::visit(
                [](auto v) -> int64_t { return v; }, sl->value));
        }
      } else if (auto *id = dyn_cast<AST::Identifier>(expr->GetR().get()))
        name = id->name;
      else if (auto *id2 = dyn_cast<AST::Identifier>(expr->GetL().get()))
        name = id2->name;
    } else if (auto *id = dyn_cast<AST::Identifier>(t.get()))
      name = id->name;
    if (name.empty()) continue;
    builder.create<coir::EventTriggerOp>(
        loc, builder.getStringAttr(name),
        subscript.empty() ? nullptr : builder.getStringAttr(subscript));
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::Synchronize &sync) {
  auto loc = Loc(sync);
  coir::ParallelLevel scope;
  switch (sync.buf_ty) {
  case Storage::SHARED:
    scope = coir::ParallelLevel::BLOCK;
    break;
  case Storage::LOCAL:
    scope = coir::ParallelLevel::GROUP;
    break;
  case Storage::GLOBAL:
    scope = coir::ParallelLevel::DEVICE;
    break;
  default:
    scope = coir::ParallelLevel::BLOCK;
    break;
  }
  builder.create<coir::BarrierOp>(
      loc, coir::ParallelLevelAttr::get(&IRContext(), scope));
  return true;
}

static coir::AtomicKind parseAtomicKind(llvm::StringRef fname) {
  using AK = coir::AtomicKind;
  return llvm::StringSwitch<AK>(fname)
      .Case("__atomic_add", AK::Add)
      .Case("__atomic_sub", AK::Sub)
      .Case("__atomic_exch", AK::Exch)
      .Case("__atomic_min", AK::Min)
      .Case("__atomic_max", AK::Max)
      .Case("__atomic_and", AK::And)
      .Case("__atomic_or", AK::Or)
      .Case("__atomic_xor", AK::Xor)
      .Case("__atomic_cas", AK::CAS)
      .Default(AK::Add);
}

void ASTCoIRGen::emitAtomicCall(AST::Call &call) {
  auto loc = Loc(call);
  auto &fname = call.function->name;
  auto &args = call.GetArguments();

  if (args.size() < 2) return;

  AST::Node *addrNode = args[0].get();
  auto valNode = args[1].get();

  if (auto *expr = dyn_cast<AST::Expr>(addrNode))
    if (expr->GetForm() == AST::Expr::Reference && expr->GetR())
      addrNode = expr->GetR().get();

  if (auto *da = dyn_cast<AST::DataAccess>(addrNode)) {
    auto tensorVal = LookupValue(da->GetDataName());
    if (!tensorVal) return;
    auto tty = mlir::dyn_cast<coir::TensorType>(tensorVal.getType());
    if (!tty) return;

    llvm::SmallVector<mlir::Value> idxVals;
    for (auto &idx : da->GetIndices()) {
      mlir::Value v = EmitExpr(*idx);
      if (!v) {
        if (auto *idNode = dyn_cast<AST::Identifier>(idx.get()))
          v = LookupValue(idNode->name);
      }
      if (v && !mlir::isa<mlir::IndexType>(v.getType()))
        v = builder.create<mlir::arith::IndexCastOp>(
            loc, mlir::IndexType::get(&IRContext()), v);
      if (v) idxVals.push_back(v);
    }
    mlir::Value valVal = EmitExpr(*valNode);
    if (!valVal || !tensorVal) return;

    auto kindAttr = coir::AtomicKindAttr::get(&IRContext(),
                                               parseAtomicKind(fname));
    builder.create<coir::AtomicOp>(loc, kindAttr, valVal, tensorVal,
                                   idxVals, /*compare=*/nullptr);
  }
}

bool ASTCoIRGen::Visit(AST::Call &call) {
  auto loc = Loc(call);
  auto &fname = call.function->name;
  auto &args = call.GetArguments();

  // Atomics lower to TensorReduceElemOp with atomic attribute
  if (call.IsAtomic()) {
    emitAtomicCall(call);
    return true;
  }

  // Skip non-lib BIFs that we don't yet lower (assert, print, arith)
  if (call.IsBIF() && !call.IsLibCall()) return true;

  // Emit arguments as operands
  llvm::SmallVector<mlir::Value> operands;
  for (auto &arg : args) {
    mlir::Value v = EmitExpr(*arg);
    if (!v) {
      if (isa<AST::Identifier>(arg))
        v = LookupValue(
            std::static_pointer_cast<AST::Identifier>(arg)->name);
    }
    if (v) operands.push_back(v);
  }

  // Collect template arguments as string attributes
  llvm::SmallVector<mlir::StringRef> tplStrs;
  llvm::SmallVector<std::string> tplStorage;
  if (call.template_args) {
    for (auto &ta : call.template_args->AllValues()) {
      std::string taStr;
      if (isa<AST::Identifier>(ta))
        taStr = std::static_pointer_cast<AST::Identifier>(ta)->name;
      else if (isa<AST::IntLiteral>(ta))
        taStr = std::to_string(std::visit(
            [](auto v) -> int64_t { return v; },
            std::static_pointer_cast<AST::IntLiteral>(ta)->value));
      else
        taStr = "?";
      tplStorage.push_back(std::move(taStr));
    }
    for (auto &s : tplStorage)
      tplStrs.push_back(s);
  }

  auto callOp = builder.create<coir::CallOp>(
      loc,
      /*result=*/mlir::Type{},
      builder.getStringAttr(fname),
      operands,
      tplStrs.empty() ? nullptr
                      : builder.getStrArrayAttr(tplStrs),
      call.IsLibCall() ? builder.getBoolAttr(true) : nullptr,
      call.IsBIF() ? builder.getBoolAttr(true) : nullptr,
      call.IsExpr() ? builder.getBoolAttr(true) : nullptr);
  (void)callOp;

  return true;
}

bool ASTCoIRGen::Visit(AST::WhileBlock &wb) {
  auto loc = Loc(wb);
  auto pred = wb.GetPred();
  if (!pred) return true;

  if (hasEarlyExit(wb.stmts.get())) {
    llvm::SmallVector<std::string> modNames;
    llvm::StringSet<> modSeen;
    if (wb.stmts) collectScalarIterArgs(wb.stmts.get(), modNames, modSeen);

    llvm::SmallVector<std::string> predNames;
    llvm::StringSet<> predSeen;
    collectScalarIterArgs(pred.get(), predNames, predSeen);
    for (auto &name : predNames) {
      if (!modSeen.count(name)) {
        auto val = LookupValue(name);
        if (val) {
          modNames.push_back(name);
          modSeen.insert(name);
        }
      }
    }

    llvm::SmallVector<mlir::Value> iterArgs;
    llvm::SmallVector<mlir::Type> iterTypes;
    llvm::SmallVector<std::string> validNames;
    for (auto &name : modNames) {
      auto val = LookupValue(name);
      if (val) {
        iterArgs.push_back(val);
        iterTypes.push_back(val.getType());
        validNames.push_back(name);
      }
    }

    auto coirWhile = builder.create<coir::CoIRWhileOp>(
        loc, iterTypes, iterArgs);

    // Condition region
    auto &condRegion = coirWhile.getCondRegion();
    auto *condBlock = new mlir::Block();
    condRegion.push_back(condBlock);
    for (unsigned i = 0; i < iterTypes.size(); ++i)
      condBlock->addArgument(iterTypes[i], loc);

    builder.setInsertionPointToEnd(condBlock);
    for (unsigned i = 0; i < validNames.size(); ++i)
      UpdateValue(validNames[i], condBlock->getArgument(i));

    auto condVal = EmitExpr(*pred);
    if (condVal && !condVal.getType().isInteger(1)) {
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0,
          condVal.getType());
      condVal = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, condVal, zero);
    }
    if (!condVal) {
      condVal = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    llvm::SmallVector<mlir::Value> condArgs;
    for (unsigned i = 0; i < validNames.size(); ++i)
      condArgs.push_back(condBlock->getArgument(i));
    builder.create<coir::CoIRWhileCondOp>(loc, condVal, condArgs);

    // Body region
    auto &bodyRegion = coirWhile.getBodyRegion();
    auto *bodyBlock = new mlir::Block();
    bodyRegion.push_back(bodyBlock);
    for (unsigned i = 0; i < iterTypes.size(); ++i)
      bodyBlock->addArgument(iterTypes[i], loc);

    PushScope();
    for (unsigned i = 0; i < validNames.size(); ++i)
      MapValue(validNames[i], bodyBlock->getArgument(i));

    builder.setInsertionPointToEnd(bodyBlock);

    WhileMergeInfo info;
    info.whileOp = coirWhile;
    info.iterNames = std::move(validNames);
    info.iterTypes = std::move(iterTypes);
    info.isCoirWhile = true;
    whileMergeStack.push_back(std::move(info));

    return true;
  }

  // Structured while (no break/continue) -> scf.while
  llvm::SmallVector<std::string> modNames;
  llvm::StringSet<> modSeen;
  if (wb.stmts) collectScalarIterArgs(wb.stmts.get(), modNames, modSeen);

  // Also include variables used in the predicate that are modified in the body
  // (the loop variable itself)
  llvm::SmallVector<std::string> predNames;
  llvm::StringSet<> predSeen;
  collectScalarIterArgs(pred.get(), predNames, predSeen);
  for (auto &name : predNames) {
    if (!modSeen.count(name)) {
      auto val = LookupValue(name);
      if (val) {
        modNames.push_back(name);
        modSeen.insert(name);
      }
    }
  }

  llvm::SmallVector<mlir::Value> iterArgs;
  llvm::SmallVector<mlir::Type> iterTypes;
  llvm::SmallVector<std::string> validNames;
  for (auto &name : modNames) {
    auto val = LookupValue(name);
    if (val) {
      iterArgs.push_back(val);
      iterTypes.push_back(val.getType());
      validNames.push_back(name);
    }
  }

  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc, iterTypes, iterArgs);

  // -- "before" region: evaluate condition, emit scf.condition --
  auto &beforeRegion = whileOp.getBefore();
  auto *beforeBlock = new mlir::Block();
  beforeRegion.push_back(beforeBlock);
  for (unsigned i = 0; i < iterTypes.size(); ++i)
    beforeBlock->addArgument(iterTypes[i], loc);

  builder.setInsertionPointToEnd(beforeBlock);

  // Map iter_args so EmitExpr can reference the loop variables
  for (unsigned i = 0; i < validNames.size(); ++i)
    UpdateValue(validNames[i], beforeBlock->getArgument(i));

  auto condVal = EmitExpr(*pred);
  if (condVal && !condVal.getType().isInteger(1)) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0,
        condVal.getType());
    condVal = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, condVal, zero);
  }
  if (!condVal) {
    auto c1 = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    condVal = c1;
  }

  llvm::SmallVector<mlir::Value> condArgs;
  for (unsigned i = 0; i < validNames.size(); ++i)
    condArgs.push_back(beforeBlock->getArgument(i));
  builder.create<mlir::scf::ConditionOp>(loc, condVal, condArgs);

  // -- "after" region: body will be filled by visitor --
  auto &afterRegion = whileOp.getAfter();
  auto *afterBlock = new mlir::Block();
  afterRegion.push_back(afterBlock);
  for (unsigned i = 0; i < iterTypes.size(); ++i)
    afterBlock->addArgument(iterTypes[i], loc);

  PushScope();
  for (unsigned i = 0; i < validNames.size(); ++i)
    MapValue(validNames[i], afterBlock->getArgument(i));

  builder.setInsertionPointToEnd(afterBlock);

  WhileMergeInfo info;
  info.whileOp = whileOp;
  info.iterNames = std::move(validNames);
  info.iterTypes = std::move(iterTypes);
  info.isCoirWhile = false;
  whileMergeStack.push_back(std::move(info));

  return true;
}

bool ASTCoIRGen::Visit(AST::Break &br) {
  if (whileMergeStack.empty() || !whileMergeStack.back().isCoirWhile)
    return true;

  auto loc = Loc(br);
  auto &info = whileMergeStack.back();
  llvm::SmallVector<mlir::Value> vals;
  for (auto &name : info.iterNames) {
    auto val = LookupValue(name);
    if (val) vals.push_back(val);
  }
  builder.create<coir::CoIRBreakOp>(loc, vals);
  return true;
}

bool ASTCoIRGen::Visit(AST::Continue &cont) {
  if (whileMergeStack.empty() || !whileMergeStack.back().isCoirWhile)
    return true;

  auto loc = Loc(cont);
  auto &info = whileMergeStack.back();
  llvm::SmallVector<mlir::Value> vals;
  for (auto &name : info.iterNames) {
    auto val = LookupValue(name);
    if (val) vals.push_back(val);
  }
  builder.create<coir::CoIRContinueOp>(loc, vals);
  return true;
}

bool ASTCoIRGen::Visit(AST::IfElseBlock &ifelse) {
  auto loc = Loc(ifelse);
  auto pred = ifelse.GetPredicate();
  if (!pred) return true;

  auto condVal = EmitExpr(*pred);
  if (!condVal) return true;

  if (!condVal.getType().isInteger(1)) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0,
        condVal.getType());
    condVal = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, condVal, zero);
  }

  llvm::SmallVector<std::string> modNames;
  llvm::StringSet<> modSeen;
  collectIfModifiedScalars(ifelse.GetThenBody().get(),
                           ifelse.GetElseBody().get(), modNames, modSeen);

  llvm::SmallVector<mlir::Value> preIfValues;
  llvm::SmallVector<mlir::Type> resultTypes;
  llvm::SmallVector<std::string> validNames;
  for (auto &name : modNames) {
    auto val = LookupValue(name);
    if (val) {
      preIfValues.push_back(val);
      resultTypes.push_back(val.getType());
      validNames.push_back(name);
    }
  }

  auto ifOp = builder.create<mlir::scf::IfOp>(loc, resultTypes,
                                                condVal,
                                                /*withElseRegion=*/true);

  IfMergeInfo info;
  info.ifOp = ifOp;
  info.modifiedNames = std::move(validNames);
  info.preIfValues = std::move(preIfValues);
  ifMergeStack.push_back(std::move(info));

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
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
  } else if (n.kind == AST::CppSourceCode::Device) {
    auto existing =
        IRModule()->getAttrOfType<mlir::StringAttr>("coir.device_code");
    std::string combined = existing ? existing.getValue().str() : "";
    combined += n.GetCode();
    IRModule()->setAttr("coir.device_code",
                        mlir::StringAttr::get(&IRContext(), combined));
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::InThreadsBlock &n) {
  auto loc = Loc(n);

  if (!n.pred) return true;

  mlir::Value predVal = EmitExpr(*n.pred);
  if (!predVal) return true;

  if (!predVal.getType().isInteger(1)) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(
        loc, 0, predVal.getType());
    predVal = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, predVal, zero);
  }

  auto ithOp = builder.create<coir::InThreadsOp>(
      loc, predVal,
      n.async ? builder.getBoolAttr(true) : nullptr,
      n.outer ? builder.getBoolAttr(true) : nullptr);

  auto &bodyRegion = ithOp.getBody();
  auto *block = new mlir::Block();
  bodyRegion.push_back(block);
  builder.setInsertionPointToStart(block);
  return true;
}
