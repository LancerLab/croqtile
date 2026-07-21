// ast_coir_gen.cpp -- AST-to-CoIR MLIR translation
#include "ASTCoIRGen.hpp"
#include "codegen_utils.hpp"
#include "context.hpp"
#include "dmaconf.hpp"
#include "symbexpr.hpp"
#include "symvals.hpp"
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
  for (const auto& ar : FCtx(fname).GetAssessor().GetAssertions()) {
    auto* key = ar.emit_node ? ar.emit_node : ar.node;
    assert_map_[key].push_back(&ar);
  }
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
    auto name = UnScopedName(sv->Value());
    auto val = LookupValue(name);
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

  if (auto to = dyn_cast<TernaryOperation>(expr)) {
    auto pred = MaterializeSBE(loc, to->GetPred());
    auto lhs = MaterializeSBE(loc, to->GetLeft());
    auto rhs = MaterializeSBE(loc, to->GetRight());
    if (!pred || !lhs || !rhs) return nullptr;

    if (!pred.getType().isInteger(1)) {
      auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      if (!mlir::isa<mlir::IndexType>(pred.getType()))
        pred = builder.create<mlir::arith::IndexCastOp>(loc, indexType, pred);
      pred = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::ne, pred, zero);
    }
    if (!mlir::isa<mlir::IndexType>(lhs.getType()))
      lhs = builder.create<mlir::arith::IndexCastOp>(loc, indexType, lhs);
    if (!mlir::isa<mlir::IndexType>(rhs.getType()))
      rhs = builder.create<mlir::arith::IndexCastOp>(loc, indexType, rhs);
    return builder.create<mlir::arith::SelectOp>(loc, pred, lhs, rhs);
  }

  return nullptr;
}

void ASTCoIRGen::EmitNodeAssertions(AST::Node* node) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = assert_map_.find(node);
  if (it == assert_map_.end()) return;

  for (auto* ar : it->second) {
    // Re-evaluate the SBE expression with the current scope's bound
    // information. Loop-scoped variables may have become concretely
    // bounded since the original SemaChecker assessment, allowing
    // compile-time resolution that the initial Assess() could not do.
    if (ar->expr) {
      auto norm = ar->expr->Normalize();
      if (auto bv = VIBool(norm); bv && *bv) {
        auto& stats = CCtx().GetAssessmentStats();
        stats.static_true++;
        stats.runtime_total--;
        switch (ar->usage_type) {
        case UsageType::UnClassified: stats.unclassified_runtime--; break;
        case UsageType::ShapeCompatibility: stats.shape_compat_runtime--; break;
        case UsageType::ElementAccess: stats.elem_access_runtime--; break;
        case UsageType::LoopBound: stats.loop_bound_runtime--; break;
        case UsageType::HardwareConstraint: stats.hw_constraint_runtime--; break;
        }
        continue;
      }
    }

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
      if (auto tty = mlir::dyn_cast<coir::TensorType>(val.getType())) {
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
  if (!vi || !vi->IsNumeric()) return mlir::ShapedType::kDynamic;
  if (auto *nv = dyn_cast<sbe::NumericValue>(vi.get()))
    return nv->Value();
  return mlir::ShapedType::kDynamic;
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
  case BaseType::F6_E2M3: return mlir::Float6E2M3FNType::get(&ctx);
  case BaseType::F6_E3M2: return mlir::Float6E3M2FNType::get(&ctx);
  case BaseType::F4_E2M1: return mlir::Float4E2M1FNType::get(&ctx);
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
  case BaseType::STREAM: return mlir::IndexType::get(&ctx);
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

  // If any dimension is dynamic, cannot compute dense strides -- clear them.
  bool hasDynDim = false;
  for (auto d : dims)
    if (mlir::ShapedType::isDynamic(d)) { hasDynDim = true; break; }

  if (hasDynDim) {
    strides.clear();
  } else {
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
  }

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
  } else if (auto *asgn = dyn_cast<AST::Assignment>(&n)) {
    if (!asgn->AssignToDataElement())
      pendingDmaAssignName = asgn->GetName();
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
    IRModule()->walk([](coir::KernelOp kernel) {
      kernel.walk([](mlir::Block *block) {
        bool seenReturn = false;
        llvm::SmallVector<mlir::Operation *> dead;
        for (auto &op : *block) {
          if (seenReturn) {
            if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
              dead.push_back(&op);
          } else if (mlir::isa<coir::KernelReturnOp>(op))
            seenReturn = true;
        }
        for (auto *op : llvm::reverse(dead)) {
          op->dropAllDefinedValueUses();
          op->dropAllReferences();
          op->erase();
        }
      });
    });
    PopScope();
    if (!suppress_output) {
      mlir::OpPrintingFlags flags;
      IRModule().print(llvm::outs(), flags);
      llvm::outs() << "\n";
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    auto *block = builder.getInsertionBlock();
    if (block) {
      bool insideKernel = false;
      if (auto *parentOp = block->getParentOp())
        for (auto *op = parentOp; op; op = op->getParentOp())
          if (mlir::isa<coir::KernelOp>(op)) { insideKernel = true; break; }
      if (insideKernel &&
          (block->empty() ||
           !block->back().hasTrait<mlir::OpTrait::IsTerminator>()))
        builder.create<coir::KernelReturnOp>(builder.getUnknownLoc(),
                                             mlir::ValueRange{});
    }
    PopScope();
  } else if (isa<AST::ParallelBy>(&n)) {
    auto *block = builder.getInsertionBlock();
    if (block && (block->empty() || !block->back().hasTrait<mlir::OpTrait::IsTerminator>()))
      builder.create<coir::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange{});
    auto *parentBlock = block->getParentOp()->getBlock();
    if (parentBlock) builder.setInsertionPointAfter(block->getParentOp());
    PopScope();
  } else if (isa<AST::ForeachBlock>(&n)) {
    // Close inner nested foreachs (no iter_args) first.
    for (unsigned ri = foreachNestDepth; ri > 1; --ri) {
      auto *block = builder.getInsertionBlock();
      auto *innerOp = block->getParentOp();
      auto *parentBlock = innerOp->getBlock();
      if (parentBlock) builder.setInsertionPointAfter(innerOp);
    }

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
    foreachNestDepth = 0;
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
    if (parentOp && mlir::isa<coir::InThreadsOp>(parentOp)) {
      if (parentOp->getBlock())
        builder.setInsertionPointAfter(parentOp);
    }
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
    if (isa<StreamType>(inTy)) continue;
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
      loc, mlir::TypeRange{},
      cf.name, mlirFnType,
      coir::LaunchBoundsAttr{},
      coir::MaxNregAttr{});

  // Attach host-facing metadata so downstream stub generation can
  // reconstruct the Choreo calling convention (spanned_view / spanned_data,
  // auto-memory-shadowing, pass-by-ref copy-back).
  llvm::SmallVector<mlir::Attribute> paramNames, paramAttrs, paramRefs,
      hostElemTypes;
  llvm::SmallVector<mlir::Attribute> paramDims;
  if (cf.f_decl.params) {
    for (auto &p : cf.f_decl.params->values) {
      if (isa<StreamType>(p->type->GetType())) continue;
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

  // Collect dynamic (symbolic) dimension names from all SpannedType params.
  // Each unique symbolic dim gets an extra index-typed kernel argument so
  // that runtime dimension values are available inside the kernel for
  // foreach bounds, stride computation, and assertion materialization.
  struct DimArgInfo {
    unsigned paramIdx;
    unsigned dimIdx;
    std::string symbolName;
  };
  llvm::SmallVector<DimArgInfo> dimArgs;
  llvm::StringSet<> seenDimSymbols;

  // Track all (param, dim) occurrences per symbol for cross-parameter
  // consistency checks (e.g. foo(f32[M,N] a, f32[N,K] b) => a.shape()[1]==b.shape()[0]).
  struct DimOccurrence {
    unsigned paramIdx;
    unsigned dimIdx;
  };
  llvm::StringMap<llvm::SmallVector<DimOccurrence>> allDimOccurrences;

  if (cf.f_decl.params) {
    unsigned pIdx = 0;
    for (auto &inTy : fty->in_tys) {
      if (isa<StreamType>(inTy)) { ++pIdx; continue; }
      if (auto sty = dyn_cast<SpannedType>(inTy)) {
        auto mdspan = sty->s_type;
        auto &shape = mdspan->value;
        if (shape.IsValid()) {
          unsigned dIdx = 0;
          for (auto &v : shape.Value()) {
            if (v && !v->IsNumeric()) {
              if (auto *sv = dyn_cast<sbe::SymbolicValue>(v.get())) {
                auto name = UnScopedName(sv->Value());
                allDimOccurrences[name].push_back({pIdx, dIdx});
                if (!seenDimSymbols.count(name)) {
                  seenDimSymbols.insert(name);
                  dimArgs.push_back({pIdx, dIdx, name});
                  argTypes.push_back(mlir::IndexType::get(&IRContext()));
                }
              }
            }
            ++dIdx;
          }
        }
      }
      ++pIdx;
    }
  }

  // Rebuild the FunctionType to include extra dim arguments.
  auto mlirFnType2 =
      mlir::FunctionType::get(&IRContext(), argTypes, resultTypes);
  kernelOp.setFunctionTypeAttr(mlir::TypeAttr::get(mlirFnType2));

  auto &bodyRegion = kernelOp.getBody();
  auto *entryBlock = new mlir::Block();
  bodyRegion.push_back(entryBlock);

  for (unsigned i = 0; i < argTypes.size(); ++i)
    entryBlock->addArgument(argTypes[i], loc);

  builder.setInsertionPointToEnd(entryBlock);

  if (cf.f_decl.params) {
    unsigned idx = 0;
    for (auto &param : cf.f_decl.params->values) {
      if (isa<StreamType>(param->type->GetType())) continue;
      if (param->HasSymbol() && idx < argTypes.size())
        MapValue(param->sym->name, entryBlock->getArgument(idx));
      ++idx;
    }
  }

  // Map dynamic dimension symbols to their kernel arguments and record
  // metadata so EmitCUDA can emit the corresponding parameters.
  unsigned baseArgCount = argTypes.size() - dimArgs.size();
  llvm::SmallVector<mlir::Attribute> dimArgAttrs;
  for (unsigned i = 0; i < dimArgs.size(); ++i) {
    auto argVal = entryBlock->getArgument(baseArgCount + i);
    MapValue(dimArgs[i].symbolName, argVal);
    auto dict = mlir::DictionaryAttr::get(&IRContext(), {
      mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "param"),
                           builder.getI64IntegerAttr(dimArgs[i].paramIdx)),
      mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "dim"),
                           builder.getI64IntegerAttr(dimArgs[i].dimIdx)),
      mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "name"),
                           mlir::StringAttr::get(&IRContext(),
                                                 dimArgs[i].symbolName)),
    });
    dimArgAttrs.push_back(dict);
  }
  if (!dimArgAttrs.empty())
    kernelOp->setAttr("coir.dim_args",
                       mlir::ArrayAttr::get(&IRContext(), dimArgAttrs));

  // Emit cross-parameter symbolic dimension consistency checks.
  // For each symbolic dim appearing in 2+ parameters, record pair-wise
  // equality checks so EmitCUDA/EmitHIP can emit host-side runtime_check.
  llvm::SmallVector<mlir::Attribute> dimCheckAttrs;
  for (auto &entry : allDimOccurrences) {
    auto &occs = entry.second;
    if (occs.size() < 2) continue;
    for (size_t i = 1; i < occs.size(); ++i) {
      auto dict = mlir::DictionaryAttr::get(&IRContext(), {
        mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "name"),
                             mlir::StringAttr::get(&IRContext(), entry.first())),
        mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "param0"),
                             builder.getI64IntegerAttr(occs[i - 1].paramIdx)),
        mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "dim0"),
                             builder.getI64IntegerAttr(occs[i - 1].dimIdx)),
        mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "param1"),
                             builder.getI64IntegerAttr(occs[i].paramIdx)),
        mlir::NamedAttribute(mlir::StringAttr::get(&IRContext(), "dim1"),
                             builder.getI64IntegerAttr(occs[i].dimIdx)),
      });
      dimCheckAttrs.push_back(dict);
    }
  }
  if (!dimCheckAttrs.empty())
    kernelOp->setAttr("coir.dim_checks",
                       mlir::ArrayAttr::get(&IRContext(), dimCheckAttrs));

  // Dynamic memory reuse: add mr_offset_* and spm_size kernel args.
  for (auto &[dfName, mri] : FCtx(cf.name).GetAllDynMemReuseInfos()) {
    if (!mri) continue;
    for (auto &[sto, ie] : mri->infos) {
      llvm::SmallVector<mlir::Attribute> mrArgNames;
      for (auto &offArg : ie.offset_args) {
        std::string sanitized = offArg;
        for (size_t pos = 0; (pos = sanitized.find("::", pos)) !=
             std::string::npos; )
          sanitized.replace(pos, 2, "_");
        argTypes.push_back(mlir::IndexType::get(&IRContext()));
        auto arg = entryBlock->addArgument(
            mlir::IndexType::get(&IRContext()), loc);
        MapValue(sanitized, arg);
        mrArgNames.push_back(
            mlir::StringAttr::get(&IRContext(), sanitized));
      }
      // spm_size arg
      argTypes.push_back(mlir::IndexType::get(&IRContext()));
      auto spmSizeArg = entryBlock->addArgument(
          mlir::IndexType::get(&IRContext()), loc);
      MapValue(ie.spm_size, spmSizeArg);

      // Store metadata for emitter
      kernelOp->setAttr("coir.mr_offset_args",
                         mlir::ArrayAttr::get(&IRContext(), mrArgNames));
      kernelOp->setAttr("coir.mr_spm_size_arg",
                         mlir::StringAttr::get(&IRContext(), ie.spm_size));
      // Store chunks info for host-side HeapSimulator emission
      llvm::SmallVector<mlir::Attribute> chunkAttrs;
      for (auto &c : ie.chunks)
        chunkAttrs.push_back(mlir::StringAttr::get(&IRContext(), c));
      kernelOp->setAttr("coir.mr_chunks",
                         mlir::ArrayAttr::get(&IRContext(), chunkAttrs));
      kernelOp->setAttr("coir.mr_chunks_name",
                         mlir::StringAttr::get(&IRContext(), ie.chunks_name));
      kernelOp->setAttr("coir.mr_result_name",
                         mlir::StringAttr::get(&IRContext(), ie.result));
      kernelOp->setAttr("coir.mr_offsets_name",
                         mlir::StringAttr::get(&IRContext(), ie.offsets_name));

      // Const interference matrix for parametric HeapSimulator plan.
      if (ie.n_buffers > 0 && !ie.interference.empty()) {
        kernelOp->setAttr("coir.mr_n_buffers",
                           mlir::IntegerAttr::get(
                               mlir::IntegerType::get(&IRContext(), 64),
                               ie.n_buffers));
        kernelOp->setAttr("coir.mr_alignment",
                           mlir::IntegerAttr::get(
                               mlir::IntegerType::get(&IRContext(), 64),
                               ie.alignment));
        llvm::SmallVector<mlir::Attribute> boolMat;
        for (bool v : ie.interference)
          boolMat.push_back(mlir::BoolAttr::get(&IRContext(), v));
        kernelOp->setAttr("coir.mr_interference",
                           mlir::ArrayAttr::get(&IRContext(), boolMat));
        llvm::SmallVector<mlir::Attribute> sizeExprs;
        for (auto &s : ie.size_exprs)
          sizeExprs.push_back(mlir::StringAttr::get(&IRContext(), s));
        kernelOp->setAttr("coir.mr_size_exprs",
                           mlir::ArrayAttr::get(&IRContext(), sizeExprs));
        llvm::SmallVector<mlir::Attribute> bufIds;
        for (auto &b : ie.buffer_ids)
          bufIds.push_back(mlir::StringAttr::get(&IRContext(), b));
        kernelOp->setAttr("coir.mr_buffer_ids",
                           mlir::ArrayAttr::get(&IRContext(), bufIds));
      }
    }
    // Update function type with the new args
    auto mlirFnType3 =
        mlir::FunctionType::get(&IRContext(), argTypes, resultTypes);
    kernelOp.setFunctionTypeAttr(mlir::TypeAttr::get(mlirFnType3));
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
      mlir::DenseI64ArrayAttr::get(&IRContext(), bounds),
      /*stream=*/nullptr, /*is_async=*/nullptr);

  // For GROUP-level parallel with dynamic bounds, record the param/dim
  // mapping so EmitCUDA can generate the correct block expression.
  if (levelAttr.getValue() == coir::ParallelLevel::GROUP ||
      levelAttr.getValue() == coir::ParallelLevel::GROUPx4) {
    llvm::SmallVector<int64_t> dynParams, dynDims;
    auto kernelOp = parallelOp->getParentOfType<coir::KernelOp>();
    auto dimArgs =
        kernelOp
            ? kernelOp->getAttrOfType<mlir::ArrayAttr>("coir.dim_args")
            : nullptr;
    if (dimArgs) {
      if (pb.HasSubPVs()) {
        for (auto &bnd : pb.AllBoundExprs()) {
          if (auto *expr = dyn_cast<AST::Expr>(bnd.get())) {
            if (expr->Opts().HasVal() &&
                EvalToInt(expr->Opts().GetVal()) ==
                    mlir::ShapedType::kDynamic) {
              if (auto *sv = dyn_cast<sbe::SymbolicValue>(
                      expr->Opts().GetVal().get())) {
                auto name = UnScopedName(sv->Value());
                for (auto da : dimArgs) {
                  auto dict = mlir::cast<mlir::DictionaryAttr>(da);
                  auto n = dict.getAs<mlir::StringAttr>("name");
                  if (n && n.getValue() == name) {
                    dynParams.push_back(
                        mlir::cast<mlir::IntegerAttr>(dict.get("param"))
                            .getInt());
                    dynDims.push_back(
                        mlir::cast<mlir::IntegerAttr>(dict.get("dim"))
                            .getInt());
                    break;
                  }
                }
              }
            }
          }
        }
      } else {
        auto bv = pb.BoundValue();
        if (EvalToInt(bv) == mlir::ShapedType::kDynamic) {
          if (auto *sv = dyn_cast<sbe::SymbolicValue>(bv.get())) {
            auto name = UnScopedName(sv->Value());
            for (auto da : dimArgs) {
              auto dict = mlir::cast<mlir::DictionaryAttr>(da);
              auto n = dict.getAs<mlir::StringAttr>("name");
              if (n && n.getValue() == name) {
                dynParams.push_back(
                    mlir::cast<mlir::IntegerAttr>(dict.get("param"))
                        .getInt());
                dynDims.push_back(
                    mlir::cast<mlir::IntegerAttr>(dict.get("dim"))
                        .getInt());
                break;
              }
            }
          }
        }
      }
    }
    if (!dynParams.empty()) {
      parallelOp->setAttr("coir.dyn_group_bound_param",
                           builder.getDenseI64ArrayAttr(dynParams));
      parallelOp->setAttr("coir.dyn_group_bound_dim",
                           builder.getDenseI64ArrayAttr(dynDims));
    }
  }

  if (pb.HasStream()) {
    std::string streamStr = STR(pb.StreamExpr());
    if (!streamStr.empty())
      parallelOp.setStreamAttr(
          mlir::StringAttr::get(&IRContext(), streamStr));
    if (pb.IsAsync())
      parallelOp.setIsAsyncAttr(mlir::BoolAttr::get(&IRContext(), true));
  }

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

  if (pb.HasLaunchBounds() && pb.GetLevel() == ParallelLevel::BLOCK) {
    auto &lb_args = pb.GetLaunchBoundsArgs();
    int64_t maxThr = 0, minBlk = 0, maxCluster = 0;
    if (lb_args->Count() >= 1)
      if (auto *e = dyn_cast<AST::Expr>(lb_args->ValueAt(0).get()))
        if (e->Opts().HasVal()) maxThr = EvalToInt(e->Opts().GetVal());
    if (lb_args->Count() >= 2)
      if (auto *e = dyn_cast<AST::Expr>(lb_args->ValueAt(1).get()))
        if (e->Opts().HasVal()) minBlk = EvalToInt(e->Opts().GetVal());
    if (lb_args->Count() >= 3)
      if (auto *e = dyn_cast<AST::Expr>(lb_args->ValueAt(2).get()))
        if (e->Opts().HasVal()) maxCluster = EvalToInt(e->Opts().GetVal());
    if (maxThr > 0 || minBlk > 0 || maxCluster > 0) {
      auto lbAttr = coir::LaunchBoundsAttr::get(
          &IRContext(), maxThr, minBlk, maxCluster);
      if (auto kernelOp = parallelOp->getParentOfType<coir::KernelOp>())
        kernelOp.setLaunchBoundsAttr(lbAttr);
    }
  }

  if (pb.HasMaxnreg() && pb.GetLevel() == ParallelLevel::BLOCK) {
    auto *arg = pb.GetMaxnregArg().get();
    if (arg && arg->Opts().HasVal()) {
      int64_t regVal = EvalToInt(arg->Opts().GetVal());
      if (regVal > 0) {
        auto nregAttr = coir::MaxNregAttr::get(&IRContext(), regVal);
        if (auto kernelOp = parallelOp->getParentOfType<coir::KernelOp>())
          kernelOp.setMaxNregAttr(nregAttr);
      }
    }
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

void collectRotateIterArgs(AST::Node *node,
                           llvm::SmallVectorImpl<std::string> &names,
                           llvm::StringSet<> &seen) {
  if (!node) return;
  if (auto *rot = dyn_cast<AST::Rotate>(node)) {
    for (auto &idNode : rot->GetIds()) {
      if (auto *id = dyn_cast<AST::Identifier>(idNode.get())) {
        if (!seen.count(id->name)) {
          names.push_back(id->name);
          seen.insert(id->name);
        }
      }
    }
    return;
  }
  if (auto *mn = dyn_cast<AST::MultiNodes>(node)) {
    for (auto &child : mn->values)
      if (child) collectRotateIterArgs(child.get(), names, seen);
    return;
  }
  if (node->HasBody()) {
    auto body = node->GetBody();
    if (body) collectRotateIterArgs(body.get(), names, seen);
  }
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

int64_t ASTCoIRGen::resolveRangeBound(AST::LoopRange *lr) {
  int64_t bound = 1;
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
  return bound;
}

mlir::Value ASTCoIRGen::resolveRangeUBValue(AST::LoopRange *lr, int64_t bound) {
  auto loc = Loc(*lr);
  mlir::Value ubValue;
  if (bound <= 1) {
    if (lr->ubound) {
      if (auto *expr = dyn_cast<AST::Expr>(lr->ubound.get())) {
        if (expr->Opts().HasVal()) {
          auto &vi = expr->Opts().GetVal();
          if (vi && !vi->IsNumeric()) {
            if (auto *sv = dyn_cast<sbe::SymbolicValue>(vi.get()))
              ubValue = LookupValue(UnScopedName(sv->Value()));
          }
        }
      }
    }
    if (!ubValue) {
      auto rvName = lr->GetRVName();
      auto symType = GetSymbolType(rvName);
      if (auto bty = dyn_cast<BoundedType>(symType)) {
        if (bty->HasValidBound()) {
          auto &ub = bty->GetUpperBound();
          if (ub && !ub->IsNumeric()) {
            if (auto *sv = dyn_cast<sbe::SymbolicValue>(ub.get()))
              ubValue = LookupValue(UnScopedName(sv->Value()));
          }
        }
      }
    }
    if (!ubValue && lr->ubound) {
      auto v = EmitExpr(*lr->ubound);
      if (v) ubValue = v;
      if (ubValue && !mlir::isa<mlir::IndexType>(ubValue.getType()))
        ubValue = builder.create<mlir::arith::IndexCastOp>(
            loc, mlir::IndexType::get(&IRContext()), ubValue);
    }
    if (!ubValue) {
      auto rvName = lr->GetRVName();
      auto symType = GetSymbolType(rvName);
      if (auto bty = dyn_cast<BoundedType>(symType)) {
        if (bty->HasValidBound()) {
          auto &ub = bty->GetUpperBound();
          if (ub && !dyn_cast<sbe::NumericValue>(ub.get())) {
            auto v = MaterializeSBE(loc, ub);
            if (v) {
              if (!mlir::isa<mlir::IndexType>(v.getType()))
                v = builder.create<mlir::arith::IndexCastOp>(
                    loc, mlir::IndexType::get(&IRContext()), v);
              ubValue = v;
            }
          }
        }
      }
    }
  }
  if (!ubValue)
    ubValue = builder.create<mlir::arith::ConstantIndexOp>(loc, bound);
  if (ubValue && !mlir::isa<mlir::IndexType>(ubValue.getType()))
    ubValue = builder.create<mlir::arith::IndexCastOp>(
        loc, mlir::IndexType::get(&IRContext()), ubValue);
  return ubValue;
}

bool ASTCoIRGen::Visit(AST::ForeachBlock &fb) {
  auto loc = Loc(fb);
  auto indexType = mlir::IndexType::get(&IRContext());

  // Collect all range bounds for multi-dim foreach.
  llvm::SmallVector<std::pair<AST::LoopRange*, int64_t>> rangeBounds;
  if (fb.ranges && !fb.ranges->values.empty()) {
    for (auto &rng : fb.ranges->values) {
      if (auto *lr = dyn_cast<AST::LoopRange>(rng.get())) {
        int64_t bound = resolveRangeBound(lr);
        // For multi-dim span, also check GetUpperBounds per dimension
        if (bound <= 1 && rangeBounds.empty()) {
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
        rangeBounds.push_back({lr, bound});
      }
    }
  }
  if (rangeBounds.empty())
    rangeBounds.push_back({nullptr, 1});

  llvm::SmallVector<std::string> accNames;
  if (fb.stmts) collectMMAAccNames(fb.stmts.get(), accNames);

  llvm::SmallVector<std::string> scalarNames;
  llvm::StringSet<> scalarSeen;
  if (fb.stmts) collectScalarIterArgs(fb.stmts.get(), scalarNames, scalarSeen);

  llvm::SmallVector<std::string> rotateNames;
  llvm::StringSet<> rotateSeen;
  if (fb.stmts) collectRotateIterArgs(fb.stmts.get(), rotateNames, rotateSeen);

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
  for (auto &name : rotateNames) {
    if (iterSeen.count(name)) continue;
    auto val = LookupValue(name);
    auto asyncTy = coir::AsyncTokenType::get(&IRContext());
    if (!val) {
      val = builder.create<coir::AsyncUndefOp>(loc, asyncTy).getResult();
      MapValue(name, val);
    }
    iterArgs.push_back(val);
    iterTypes.push_back(val.getType());
    iterNames.push_back(name);
    iterSeen.insert(name);
    auto dataName = name + ".data";
    if (!iterSeen.count(dataName)) {
      auto dataVal = LookupValue(dataName);
      if (dataVal && mlir::isa<coir::TensorType>(dataVal.getType())) {
        iterArgs.push_back(dataVal);
        iterTypes.push_back(dataVal.getType());
        iterNames.push_back(dataName);
        iterSeen.insert(dataName);
      }
    }
  }

  // Emit nested ForeachOps: outermost carries iter_args, inner ones are plain.
  unsigned numRanges = rangeBounds.size();
  for (unsigned ri = 0; ri < numRanges; ++ri) {
    auto *lr = rangeBounds[ri].first;
    int64_t bound = rangeBounds[ri].second;
    mlir::Value ubValue = lr ? resolveRangeUBValue(lr, bound)
                             : builder.create<mlir::arith::ConstantIndexOp>(loc, bound);

    mlir::Value lbValue;
    if (lr && lr->lbound) {
      auto lbExpr = EmitExpr(*lr->lbound);
      if (lbExpr) {
        if (!mlir::isa<mlir::IndexType>(lbExpr.getType()))
          lbExpr = builder.create<mlir::arith::IndexCastOp>(
              loc, indexType, lbExpr);
        lbValue = lbExpr;
        ubValue = builder.create<mlir::arith::SubIOp>(loc, ubValue, lbValue);
      }
    }

    coir::ForeachOp foreachOp;
    if (ri == 0) {
      foreachOp = builder.create<coir::ForeachOp>(
          loc, iterTypes, ubValue, iterArgs);
    } else {
      foreachOp = builder.create<coir::ForeachOp>(
          loc, mlir::TypeRange{}, ubValue, mlir::ValueRange{});
    }

    auto &bodyRegion = foreachOp.getBody();
    auto *block = new mlir::Block();
    bodyRegion.push_back(block);

    auto ivArg = block->addArgument(indexType, loc);

    if (ri == 0) {
      pendingYields.clear();
      for (unsigned i = 0; i < iterNames.size(); ++i) {
        auto iterArg = block->addArgument(iterTypes[i], loc);
        MapValue(iterNames[i], iterArg);
        pendingYields.push_back({iterNames[i], iterArg});
      }
    }

    builder.setInsertionPointToEnd(block);
    builder.create<coir::YieldOp>(loc, mlir::ValueRange{});
    builder.setInsertionPoint(block->getTerminator());

    if (lr) {
      mlir::Value mappedIV = ivArg;
      if (lbValue)
        mappedIV = builder.create<mlir::arith::AddIOp>(loc, ivArg, lbValue);
      MapValue(lr->GetIVName(), mappedIV);
    }
  }

  foreachNestDepth = numRanges;
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
    if (fl->IsFloat64()) {
      auto ty = mlir::Float64Type::get(&IRContext());
      double val = fl->Val_f64();
      return builder.create<mlir::arith::ConstantFloatOp>(
          loc, llvm::APFloat(val), ty);
    }
    auto ty = mlir::Float32Type::get(&IRContext());
    float val = fl->Val_f32();
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
      auto v = LookupValue(da->GetDataName());
      if (!v) v = LookupValue(da->GetDataName() + ".data");
      return v;
    }
  }

  if (auto *ce = dyn_cast<AST::CastExpr>(&n)) {
    auto inner = EmitExpr(*ce->GetR());
    if (!inner) return nullptr;
    if (ce->FromType() == ce->ToType()) return inner;
    auto toBT = ce->ToType();
    bool fromFloat = IsFloatType(ce->FromType());
    bool toFloat = IsFloatType(toBT);
    bool fromInt = IsIntegerType(ce->FromType());
    bool toInt = IsIntegerType(toBT);
    auto getFloatTy = [&](BaseType bt) -> mlir::Type {
      if (bt == BaseType::F16) return mlir::Float16Type::get(&IRContext());
      if (bt == BaseType::F32) return mlir::Float32Type::get(&IRContext());
      if (bt == BaseType::F64) return mlir::Float64Type::get(&IRContext());
      if (bt == BaseType::BF16) return mlir::BFloat16Type::get(&IRContext());
      return {};
    };
    if (fromFloat && toFloat) {
      auto targetTy = getFloatTy(toBT);
      if (targetTy && targetTy != inner.getType()) {
        unsigned fromBits = inner.getType().getIntOrFloatBitWidth();
        unsigned toBits = targetTy.getIntOrFloatBitWidth();
        if (toBits > fromBits)
          return builder.create<mlir::arith::ExtFOp>(loc, targetTy, inner);
        else
          return builder.create<mlir::arith::TruncFOp>(loc, targetTy, inner);
      }
    } else if (fromInt && toFloat) {
      auto targetTy = getFloatTy(toBT);
      if (targetTy) {
        if (mlir::isa<mlir::IndexType>(inner.getType()))
          inner = builder.create<mlir::arith::IndexCastOp>(
              loc, builder.getI32Type(), inner);
        if (IsSignedType(ce->FromType()))
          return builder.create<mlir::arith::SIToFPOp>(loc, targetTy, inner);
        else
          return builder.create<mlir::arith::UIToFPOp>(loc, targetTy, inner);
      }
    } else if (fromFloat && toInt) {
      auto getBits = [](BaseType bt) -> unsigned {
        switch (bt) {
        case BaseType::S8: case BaseType::U8: return 8;
        case BaseType::S16: case BaseType::U16: return 16;
        case BaseType::S64: case BaseType::U64: return 64;
        default: return 32;
        }
      };
      auto intTy = mlir::IntegerType::get(&IRContext(), getBits(toBT));
      if (IsSignedType(toBT))
        return builder.create<mlir::arith::FPToSIOp>(loc, intTy, inner);
      else
        return builder.create<mlir::arith::FPToUIOp>(loc, intTy, inner);
    }
    return inner;
  }

  if (auto *call = dyn_cast<AST::Call>(&n)) {
    if (call->IsArith() || call->IsLibCall()) {
      auto &fname = call->function->name;
      auto &callArgs = call->GetArguments();
      llvm::SmallVector<mlir::Value> operands;
      for (auto &arg : callArgs) {
        mlir::Value v = EmitExpr(*arg);
        if (!v) {
          if (auto name = AST::GetName(*arg))
            v = LookupValue(*name);
        }
        if (v) operands.push_back(v);
      }
      if (operands.empty()) return nullptr;
      mlir::Type resTy = operands[0].getType();
      if (mlir::isa<mlir::IndexType>(resTy))
        resTy = mlir::IntegerType::get(&IRContext(), 32,
                                       mlir::IntegerType::Signless);
      llvm::SmallVector<mlir::StringRef> tplStrs;
      llvm::SmallVector<std::string> tplStorage;
      if (call->template_args) {
        for (auto &ta : call->template_args->AllValues()) {
          if (auto name = AST::GetName(*ta))
            tplStorage.push_back(*name);
          else if (auto lit = dyn_cast<AST::IntLiteral>(ta))
            tplStorage.push_back(STR(*lit));
          else if (auto expr = dyn_cast<AST::Expr>(ta);
                   expr->IsReference() &&
                   isa<AST::DataType>(expr->GetReference()))
            tplStorage.push_back(
                CppTypeName(cast<AST::DataType>(expr->GetReference())
                                ->getBaseType()));
          else if (auto expr = dyn_cast<AST::Expr>(ta);
                   expr->Opts().HasVal())
            tplStorage.push_back(
                STR(EvalToInt(expr->Opts().GetVal())));
          else
            choreo_unreachable("unexpected template argument type in call");
        }
        for (auto &s : tplStorage)
          tplStrs.push_back(s);
      }
      auto callOp = builder.create<coir::CallOp>(
          loc, resTy, builder.getStringAttr(fname), operands,
          tplStrs.empty() ? nullptr : builder.getStrArrayAttr(tplStrs),
          call->IsLibCall() ? builder.getBoolAttr(true) : nullptr,
          call->IsBIF() ? builder.getBoolAttr(true) : nullptr,
          builder.getBoolAttr(true));
      return callOp.getResult();
    }
    // Handle device function calls used as expressions
    // (e.g., call kernel2() in an if condition)
    if (call->IsExpr()) {
      auto &fname = call->function->name;
      auto &callArgs = call->GetArguments();
      llvm::SmallVector<mlir::Value> operands;
      for (auto &arg : callArgs) {
        mlir::Value v = EmitExpr(*arg);
        if (!v) {
          if (auto name = AST::GetName(*arg))
            v = LookupValue(*name);
        }
        if (v) operands.push_back(v);
      }
      // Determine return type from device function declarations
      mlir::Type resTy;
      if (!call->device_functions.empty()) {
        auto &df = call->device_functions[0];
        resTy = LowerBaseType(df->ret_type->GetDataType());
      } else {
        // Default to i32 when device function not resolved
        resTy = mlir::IntegerType::get(&IRContext(), 32,
                                       mlir::IntegerType::Signless);
      }
      llvm::SmallVector<mlir::StringRef> tplStrs;
      llvm::SmallVector<std::string> tplStorage;
      if (call->template_args) {
        for (auto &ta : call->template_args->AllValues()) {
          std::string taStr;
          if (auto name = AST::GetName(*ta))
            taStr = *name;
          else if (auto lit = dyn_cast<AST::IntLiteral>(ta))
            taStr = STR(*lit);
          else if (auto expr = dyn_cast<AST::Expr>(ta);
                   expr->IsReference() &&
                   isa<AST::DataType>(expr->GetReference()))
            taStr = CppTypeName(
                cast<AST::DataType>(expr->GetReference())->getBaseType());
          else if (auto expr = dyn_cast<AST::Expr>(ta);
                   expr->Opts().HasVal())
            taStr = STR(EvalToInt(expr->Opts().GetVal()));
          else
            choreo_unreachable("unexpected template argument type in call");
          tplStorage.push_back(std::move(taStr));
        }
        for (auto &s : tplStorage)
          tplStrs.push_back(s);
      }
      auto callOp = builder.create<coir::CallOp>(
          loc, resTy, builder.getStringAttr(fname), operands,
          tplStrs.empty() ? nullptr : builder.getStrArrayAttr(tplStrs),
          nullptr, // not lib call
          nullptr, // not BIF
          builder.getBoolAttr(true)); // is_expr
      return callOp.getResult();
    }
  }

  if (auto *expr = dyn_cast<AST::Expr>(&n)) {
    if (expr->GetForm() == AST::Expr::Reference)
      return EmitExpr(*expr->GetR());

    if (expr->GetForm() == AST::Expr::Binary) {
      auto op = expr->GetOp();
      if (op == Op::UBoundAdd || op == Op::UBoundSub)
        return EmitExpr(*expr->GetL());
      if (op == Op::UBound) {
        auto lhs = EmitExpr(*expr->GetL());
        auto rhs = EmitExpr(*expr->GetR());
        if (!lhs || !rhs) return nullptr;
        auto *rNode = expr->GetR().get();
        std::string rName;
        if (auto *rid = dyn_cast<AST::Identifier>(rNode))
          rName = rid->name;
        auto rType = rName.empty() ? expr->GetR()->GetType()
                                   : GetSymbolType(rName);
        auto rty = dyn_cast<BoundedType>(rType);
        if (rty && rty->HasValidBound()) {
          int64_t ub = EvalToInt(rty->GetUpperBound());
          if (lhs.getType() != rhs.getType()) {
            if (mlir::isa<mlir::IndexType>(lhs.getType()))
              rhs = builder.create<mlir::arith::IndexCastOp>(
                  loc, lhs.getType(), rhs);
            else if (mlir::isa<mlir::IndexType>(rhs.getType()))
              lhs = builder.create<mlir::arith::IndexCastOp>(
                  loc, rhs.getType(), lhs);
          }
          mlir::Value ubConst;
          if (mlir::isa<mlir::IndexType>(lhs.getType()))
            ubConst = builder.create<mlir::arith::ConstantIndexOp>(loc, ub);
          else
            ubConst = builder.create<mlir::arith::ConstantIntOp>(
                loc, ub, lhs.getType());
          auto mul = builder.create<mlir::arith::MulIOp>(loc, lhs, ubConst);
          return (mlir::Value)builder.create<mlir::arith::AddIOp>(loc, mul,
                                                                   rhs);
        }
        return nullptr;
      }
      if (op == Op::LogicAnd || op == Op::LogicOr) {
        auto lhs = EmitExpr(*expr->GetL());
        auto rhs = EmitExpr(*expr->GetR());
        if (!lhs || !rhs) return nullptr;
        auto toBool = [&](mlir::Value v) -> mlir::Value {
          if (v.getType().isInteger(1)) return v;
          mlir::Value zero;
          if (mlir::isa<mlir::IndexType>(v.getType()))
            zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          else
            zero = builder.create<mlir::arith::ConstantIntOp>(
                loc, 0, v.getType());
          return builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ne, v, zero);
        };
        lhs = toBool(lhs);
        rhs = toBool(rhs);
        if (op == Op::LogicAnd)
          return (mlir::Value)builder.create<mlir::arith::AndIOp>(
              loc, lhs, rhs);
        return (mlir::Value)builder.create<mlir::arith::OrIOp>(
            loc, lhs, rhs);
      }

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
      bool isMMAFrag = mlir::isa<coir::MMAFragType>(resTy);
      bool isFloat = mlir::isa<mlir::FloatType>(resTy);
      if (!isFloat && isMMAFrag) {
        auto fragTy = mlir::cast<coir::MMAFragType>(resTy);
        isFloat = mlir::isa<mlir::FloatType>(fragTy.getElementType());
      }

      if (isMMAFrag) {
        std::string funcName;
        if (op == Op::Add) funcName = "fragment_scalar_elementwise_add";
        else if (op == Op::Sub) funcName = "fragment_scalar_elementwise_sub";
        else if (op == Op::Mul) funcName = "fragment_scalar_elementwise_mul";
        if (!funcName.empty()) {
          llvm::SmallVector<mlir::Value> operands = {lhs, rhs};
          auto callOp = builder.create<coir::CallOp>(
              loc, resTy, funcName, operands,
              mlir::ArrayAttr{},
              mlir::BoolAttr{}, mlir::BoolAttr{}, mlir::BoolAttr{});
          return callOp.getResult();
        }
      }

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

      if (op == Op::UBound) {
        // m # g0 = m * upperBound(g0) + g0
        int64_t ub = 0;
        auto *rhsNode = expr->GetR().get();
        if (auto *rhsId = dyn_cast<AST::Identifier>(rhsNode)) {
          auto symType = GetSymbolType(rhsId->name);
          if (auto bty = dyn_cast<BoundedType>(symType))
            if (bty->HasValidBound())
              if (auto v = VIInt(bty->GetUpperBound()))
                ub = *v;
        }
        if (ub == 0) {
          if (auto *rhsExpr = dyn_cast<AST::Expr>(rhsNode))
            if (rhsExpr->Opts().HasUBound())
              if (auto v = VIInt(rhsExpr->Opts().GetUBound()))
                ub = *v;
        }
        if (ub > 0) {
          auto ubConst = builder.create<mlir::arith::ConstantOp>(
              loc, resTy,
              builder.getIntegerAttr(resTy, ub));
          auto mul = builder.create<mlir::arith::MulIOp>(loc, lhs, ubConst);
          return (mlir::Value)builder.create<mlir::arith::AddIOp>(
              loc, mul, rhs);
        }
        return (mlir::Value)builder.create<mlir::arith::AddIOp>(loc, lhs,
                                                                 rhs);
      }

      if (op == Op::SizeOf) {
        if (!expr->Opts().HasVal())
          choreo_unreachable("SizeOf: no optimized value available");
        return MaterializeSBE(loc, expr->Opts().GetVal());
      }

      if (op == Op::DataOf || op == Op::MDataOf) {
        if (!expr->Opts().HasVal())
          choreo_unreachable("DataOf/MDataOf: no optimized value available");
        return MaterializeSBE(loc, expr->Opts().GetVal());
      }
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
      auto op = expr->GetOp();
      if (op == Op::GetUBound) {
        if (expr->Opts().HasVal()) {
          int64_t val = EvalToInt(expr->Opts().GetVal());
          auto ty = mlir::IntegerType::get(&IRContext(), 32,
                                           mlir::IntegerType::Signless);
          return builder.create<mlir::arith::ConstantIntOp>(loc, val, ty);
        }
        auto *rNode = expr->GetR().get();
        std::string rName;
        if (auto *rid = dyn_cast<AST::Identifier>(rNode))
          rName = rid->name;
        else if (auto *re = dyn_cast<AST::Expr>(rNode))
          if (auto *rid2 = dyn_cast<AST::Identifier>(re->GetR().get()))
            rName = rid2->name;
        if (!rName.empty()) {
          auto symType = GetSymbolType(rName);
          if (auto rty = dyn_cast<BoundedType>(symType)) {
            if (rty->HasValidBound()) {
              int64_t ub = EvalToInt(rty->GetUpperBound());
              auto ty = mlir::IntegerType::get(&IRContext(), 32,
                                               mlir::IntegerType::Signless);
              return builder.create<mlir::arith::ConstantIntOp>(loc, ub, ty);
            }
          }
        }
        return nullptr;
      }
      if (op == Op::LogicNot) {
        auto operand = EmitExpr(*expr->GetR());
        if (!operand) return nullptr;
        if (!operand.getType().isInteger(1)) {
          mlir::Value zero;
          if (mlir::isa<mlir::IndexType>(operand.getType()))
            zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          else
            zero = builder.create<mlir::arith::ConstantIntOp>(
                loc, 0, operand.getType());
          operand = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::ne, operand, zero);
        }
        auto one = builder.create<mlir::arith::ConstantIntOp>(
            loc, 1, operand.getType());
        return (mlir::Value)builder.create<mlir::arith::XOrIOp>(loc, operand,
                                                                 one);
      }
      if (op == Op::DataOf || op == Op::MDataOf) {
        // input_load_s.data parses as dataid_expr -> Unary "dataof" Expr.
        // Resolve by first trying <name>.data, then falling back to <name>.
        auto inner = expr->GetR();
        if (auto name = AST::GetName(*inner)) {
          auto dataName = *name + ".data";
          auto v = LookupValue(dataName);
          if (v) return v;
          v = LookupValue(*name);
          if (v) return v;
        }
        return EmitExpr(*inner);
      }
      auto operand = EmitExpr(*expr->GetR());
      if (!operand) return nullptr;
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

  if (auto *expr = dyn_cast<AST::Expr>(&n)) {
    if (expr->Opts().HasVal()) {
      int64_t val = EvalToInt(expr->Opts().GetVal());
      auto ty = mlir::IntegerType::get(&IRContext(), 32,
                                       mlir::IntegerType::Signless);
      return builder.create<mlir::arith::ConstantIntOp>(loc, val, ty);
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

  auto *block = builder.getInsertionBlock();
  if (block) {
    bool insideKernel = false;
    if (auto *parentOp = block->getParentOp())
      for (auto *op = parentOp; op; op = op->getParentOp())
        if (mlir::isa<coir::KernelOp>(op)) { insideKernel = true; break; }
    if (!insideKernel) return true;
  }

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

    // Register-storage spanned types with init are MMA operand fragments
    if (sty->m_type == Storage::REG && (nvd.init_expr || nvd.init_value)) {
      auto elemTy = LowerBaseType(sty->ElementType());
      llvm::SmallVector<int64_t> shape;
      auto &mdspan = sty->GetShape();
      if (mdspan.IsValid())
        for (auto &v : mdspan.Value())
          shape.push_back(EvalToInt(v));
      if (shape.empty()) shape = {16, 16};

      auto fragTy = coir::MMAFragType::get(&IRContext(), elemTy, shape);
      mlir::Value fillVal;
      if (nvd.init_expr)
        fillVal = EmitExpr(*nvd.init_expr);
      else if (nvd.init_value)
        fillVal = EmitExpr(*nvd.init_value);
      if (!fillVal) return true;

      if (fillVal.getType() != elemTy) {
        if (mlir::isa<mlir::FloatType>(elemTy) &&
            mlir::isa<mlir::FloatType>(fillVal.getType())) {
          unsigned srcBits =
              mlir::cast<mlir::FloatType>(fillVal.getType()).getWidth();
          unsigned dstBits = mlir::cast<mlir::FloatType>(elemTy).getWidth();
          if (dstBits > srcBits)
            fillVal = builder.create<mlir::arith::ExtFOp>(loc, elemTy, fillVal);
          else
            fillVal =
                builder.create<mlir::arith::TruncFOp>(loc, elemTy, fillVal);
        } else if (mlir::isa<mlir::FloatType>(elemTy))
          fillVal = builder.create<mlir::arith::SIToFPOp>(loc, elemTy, fillVal);
      }

      auto fillOp = builder.create<coir::MMAFillOp>(loc, fragTy, fillVal);
      MapValue(nvd.GetName(), fillOp.getResult());
      return true;
    }

    // SPM backing array: skip IR generation, will be emitted by codegen.
    if (nvd.HasNote("spm")) {
      auto &mdspan = sty->GetShape();
      int64_t spmBytes = 1;
      if (mdspan.IsValid())
        for (auto &v : mdspan.Value())
          spmBytes *= EvalToInt(v);
      auto elemTy = LowerBaseType(sty->ElementType());
      spmBytes *= (elemTy.getIntOrFloatBitWidth() / 8);
      pendingSpmSize = spmBytes;
      return true;
    }

    auto tty = LowerSpannedType(sty);
    llvm::SmallVector<mlir::Value> dynDimVals;
    if (tty.hasDynamicShape()) {
      for (unsigned d = 0; d < tty.getRank(); ++d) {
        if (tty.isDynamicDim(d)) {
          auto &shape = sty->GetShape();
          if (shape.IsValid() && d < shape.Value().size()) {
            auto &vi = shape.Value()[d];
            if (auto *sv = dyn_cast<sbe::SymbolicValue>(vi.get())) {
              auto symName = UnScopedName(sv->Value());
              auto val = LookupValue(symName);
              if (val) { dynDimVals.push_back(val); continue; }
            }
          }
          // Fallback: find from kernel index block args
          if (auto kernelOp =
                  builder.getBlock()->getParent()->getParentOfType<
                      coir::KernelOp>()) {
            auto fnTy = kernelOp.getFunctionType();
            for (unsigned a = 0; a < fnTy.getNumInputs(); ++a) {
              if (mlir::isa<mlir::IndexType>(fnTy.getInput(a))) {
                dynDimVals.push_back(
                    kernelOp.getBody().getArgument(a));
                break;
              }
            }
          }
        }
      }
    }
    mlir::TypedAttr initAttr;
    if (nvd.init_value) {
      auto *valNode = nvd.init_value.get();
      if (auto *expr = dyn_cast<AST::Expr>(valNode))
        valNode = expr->GetR().get();
      if (auto *il = dyn_cast<AST::IntLiteral>(valNode)) {
        int64_t v =
            std::visit([](auto x) -> int64_t { return x; }, il->value);
        if (mlir::isa<mlir::FloatType>(tty.getElementType()))
          initAttr = builder.getFloatAttr(tty.getElementType(),
                                          static_cast<double>(v));
        else
          initAttr = builder.getIntegerAttr(tty.getElementType(), v);
      } else if (auto *fl = dyn_cast<AST::FloatLiteral>(valNode)) {
        double v =
            std::visit([](auto x) -> double { return x; }, fl->value);
        initAttr = builder.getFloatAttr(tty.getElementType(), v);
      }
    }
    mlir::StringAttr reuseSpm;
    mlir::IntegerAttr reuseOffset;
    bool isDynReuse = false;
    std::string dynOffsetName;
    if (nvd.HasNote("reuse")) {
      auto spmName = nvd.GetNote("reuse");
      if (!spmName.empty())
        reuseSpm = builder.getStringAttr(spmName);
      if (nvd.HasNote("offset")) {
        auto offStr = nvd.GetNote("offset");
        if (offStr.find("mr_offset") == 0) {
          isDynReuse = true;
          dynOffsetName = offStr;
          reuseOffset = builder.getI64IntegerAttr(-1);
        } else {
          int64_t off = 0;
          try { off = std::stoll(offStr); }
          catch (...) {}
          reuseOffset = builder.getI64IntegerAttr(off);
        }
      }
    }
    auto allocOp = builder.create<coir::TensorAllocOp>(
        loc, tty, dynDimVals, initAttr, reuseSpm, reuseOffset);
    if (reuseOffset && pendingSpmSize > 0) {
      allocOp->setAttr("spm_size",
                       builder.getI64IntegerAttr(pendingSpmSize));
    }
    if (isDynReuse) {
      allocOp->setAttr("dyn_offset_arg",
                       builder.getStringAttr(dynOffsetName));
    }
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
    auto checkName = [](const std::string &n) {
      return n == "_" || n == "__choreo_no_tiling__" ||
             n == "__choreo_parent_dim__";
    };
    if (auto *id = dyn_cast<AST::Identifier>(idx))
      return checkName(id->name);
    if (auto *expr = dyn_cast<AST::Expr>(idx)) {
      if (expr->GetForm() == AST::Expr::Reference) {
        auto *inner = expr->GetR().get();
        if (auto *id = dyn_cast<AST::Identifier>(inner))
          return checkName(id->name);
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

  llvm::SmallVector<mlir::Value> dynDimVals;

  if (chunk.HasOperation()) {
    for (auto &sop : chunk.AllOperations()) {
      if (auto *viewOp = dyn_cast<AST::SOP::View>(sop.get())) {
        auto offsets = viewOp->GetOffsets();
        auto subspan = viewOp->GetSubSpan();
        if (!offsets || !subspan) continue;
        for (auto &off : offsets->AllValues()) {
          auto v = emitIdx(off.get());
          if (v) idxVals.push_back(v);
          else {
            auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            idxVals.push_back(zero);
          }
        }
        unsigned dimIdx = 0;
        for (auto &dim : subspan->AllValues()) {
          if (isWildcard(dim.get())) {
            tileShape.push_back(dimIdx < baseShape.size()
                                    ? baseShape[dimIdx]
                                    : 1);
          } else if (auto *lit = dyn_cast<AST::IntLiteral>(dim.get())) {
            tileShape.push_back(lit->Val());
          } else if (auto *expr = dyn_cast<AST::Expr>(dim.get())) {
            if (expr->GetForm() == AST::Expr::Reference) {
              if (auto *lit = dyn_cast<AST::IntLiteral>(
                      expr->GetR().get()))
                tileShape.push_back(lit->Val());
              else {
                tileShape.push_back(mlir::ShapedType::kDynamic);
                auto v = emitIdx(dim.get());
                if (v) dynDimVals.push_back(v);
              }
            } else {
              tileShape.push_back(mlir::ShapedType::kDynamic);
              auto v = emitIdx(dim.get());
              if (v) dynDimVals.push_back(v);
            }
          } else {
            tileShape.push_back(mlir::ShapedType::kDynamic);
            auto v = emitIdx(dim.get());
            if (v) dynDimVals.push_back(v);
          }
          dimIdx++;
        }
        break;
      }
      if (auto *subspanOp = dyn_cast<AST::SOP::SubSpan>(sop.get())) {
        auto subspan = subspanOp->GetSubSpan();
        auto indices = subspanOp->GetIndices();
        if (!subspan) continue;
        if (indices) {
          for (auto &idx : indices->AllValues()) {
            auto v = emitIdx(idx.get());
            if (v) idxVals.push_back(v);
            else {
              auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              idxVals.push_back(zero);
            }
          }
        } else {
          unsigned rank = subspan->AllValues().size();
          for (unsigned i = 0; i < rank; ++i) {
            auto zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
            idxVals.push_back(zero);
          }
        }
        unsigned dimIdx = 0;
        for (auto &dim : subspan->AllValues()) {
          if (isWildcard(dim.get())) {
            tileShape.push_back(dimIdx < baseShape.size()
                                    ? baseShape[dimIdx]
                                    : 1);
          } else if (auto *lit = dyn_cast<AST::IntLiteral>(dim.get())) {
            tileShape.push_back(lit->Val());
          } else if (auto *expr = dyn_cast<AST::Expr>(dim.get())) {
            if (expr->GetForm() == AST::Expr::Reference) {
              if (auto *lit = dyn_cast<AST::IntLiteral>(
                      expr->GetR().get()))
                tileShape.push_back(lit->Val());
              else {
                tileShape.push_back(mlir::ShapedType::kDynamic);
                auto v = emitIdx(dim.get());
                if (v) dynDimVals.push_back(v);
              }
            } else {
              tileShape.push_back(mlir::ShapedType::kDynamic);
              auto v = emitIdx(dim.get());
              if (v) dynDimVals.push_back(v);
            }
          } else {
            tileShape.push_back(mlir::ShapedType::kDynamic);
            auto v = emitIdx(dim.get());
            if (v) dynDimVals.push_back(v);
          }
          dimIdx++;
        }
        break;
      }
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

  for (auto &v : dynDimVals)
    idxVals.push_back(v);

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
        llvm::SmallVector<mlir::Value> dstDynDims;
        if (dstTy.hasDynamicShape()) {
          if (auto srcAlloc = srcVal.getDefiningOp<coir::TensorAllocOp>())
            dstDynDims.append(srcAlloc.getDynamicDims().begin(),
                              srcAlloc.getDynamicDims().end());
        }
        auto allocOp =
            builder.create<coir::TensorAllocOp>(
                loc, dstTy, dstDynDims,
                mlir::TypedAttr{}, mlir::StringAttr{}, mlir::IntegerAttr{});
        dstVal = allocOp.getResult();
      }
    }
  }

  if (!srcVal || !dstVal) return true;

  // DMA shape-compatibility is already assessed by SemaChecker and
  // materialized via EmitNodeAssertions; no ad-hoc IR-level check needed.

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

  // DMA: user wrote `dma.copy` -> DmaCopyOp (always produces token).
  mlir::Value token = nullptr;
  if (dma.IsTMA()) {
    mlir::IntegerAttr swizAttr;
    auto swizMode = dma.GetSwizzleMode();
    if (swizMode != SwizMode::NONE) {
      int64_t swizBytes = 0;
      switch (swizMode) {
      case SwizMode::B32: swizBytes = 32; break;
      case SwizMode::B64: swizBytes = 64; break;
      case SwizMode::B128: swizBytes = 128; break;
      default: break;
      }
      if (swizBytes > 0)
        swizAttr = builder.getI64IntegerAttr(swizBytes);
    }
    auto tmaCopy = builder.create<coir::TmaCopyOp>(
        loc, coir::AsyncTokenType::get(&IRContext()), srcVal, dstVal,
        swizAttr, dma.IsOOBZeroFill() ? builder.getUnitAttr()
                                      : mlir::UnitAttr());
    token = tmaCopy.getToken();
  } else {
    auto dmaCopy = builder.create<coir::DmaCopyOp>(
        loc, coir::AsyncTokenType::get(&IRContext()), srcVal, dstVal,
        kindAttr, padLowAttr, padHighAttr, padValueAttr, transpPermAttr);
    token = dmaCopy.getToken();
  }

  auto futureName = dma.future;
  if (futureName.empty() && !pendingDmaAssignName.empty())
    futureName = pendingDmaAssignName;
  if (!futureName.empty()) {
    if (isAsync && token) {
      MapValue(futureName, token);
      if (dstVal)
        MapValue(futureName + ".data", dstVal);
    } else if (dstVal) {
      MapValue(futureName, dstVal);
      MapValue(futureName + ".data", dstVal);
    }
  }
  pendingDmaAssignName.clear();

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
    mlir::Type elemTy;
    if (fillType != BaseType::UNKSCALAR)
      elemTy = LowerBaseType(fillType);
    else {
      elemTy = fillVal.getType();
      if (elemTy.isF64())
        elemTy = mlir::Float32Type::get(&IRContext());
    }

    llvm::SmallVector<int64_t> shape;
    // Prefer shape from symbol table (set by ShapeInference)
    std::string fragName = AST::FragName(op.FillingTo());
    auto symType = GetSymbolType(fragName);
    if (symType) {
      if (auto sty = dyn_cast<SpannedType>(symType)) {
        auto &mdspan = sty->GetShape();
        if (mdspan.IsValid())
          for (auto &v : mdspan.Value())
            shape.push_back(EvalToInt(v));
      }
    }
    if (shape.empty()) {
      auto nodeType = n.GetType();
      if (auto sty = dyn_cast<SpannedType>(nodeType)) {
        auto &mdspan = sty->s_type->value;
        if (mdspan.IsValid())
          for (auto &v : mdspan.Value())
            shape.push_back(EvalToInt(v));
      }
    }
    if (shape.empty())
      shape = {16, 16};

    auto fragTy = coir::MMAFragType::get(&IRContext(), elemTy, shape);

    if (fillVal.getType() != elemTy) {
      if (mlir::isa<mlir::FloatType>(elemTy) &&
          mlir::isa<mlir::FloatType>(fillVal.getType())) {
        unsigned fromBits = fillVal.getType().getIntOrFloatBitWidth();
        unsigned toBits = elemTy.getIntOrFloatBitWidth();
        if (toBits > fromBits)
          fillVal = builder.create<mlir::arith::ExtFOp>(loc, elemTy, fillVal);
        else
          fillVal =
              builder.create<mlir::arith::TruncFOp>(loc, elemTy, fillVal);
      } else if (mlir::isa<mlir::FloatType>(elemTy))
        fillVal = builder.create<mlir::arith::SIToFPOp>(loc, elemTy, fillVal);
    }

    auto fillOp = builder.create<coir::MMAFillOp>(loc, fragTy, fillVal);
    MapValue(fragName, fillOp.getResult());
    break;
  }

  case AST::MMAOperation::Load:
  case AST::MMAOperation::LoadR: {
    auto chunkAt = op.LoadFrom();
    if (!chunkAt) return true;

    auto srcVal = LookupValue(chunkAt->data->name);
    if (!srcVal) return true;

    if (mlir::isa<coir::AsyncTokenType>(srcVal.getType())) {
      auto dataVal = LookupValue(chunkAt->data->name + ".data");
      if (dataVal) srcVal = dataVal;
    }

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

    // Infer tile shape from ChunkAt's block shape (preferred) or AST type.
    llvm::SmallVector<int64_t> tileShape;
    if (chunkAt->GetBlockShape().IsValid()) {
      auto posVals = chunkAt->GetBlockShape().PosValList();
      if (posVals)
        for (auto d : *posVals) tileShape.push_back(static_cast<int64_t>(d));
    }
    if (tileShape.empty()) {
      // WGMMA B operand: shared-memory source with no chunkat -> full tensor
      if (srcTy.getMemorySpace() == 1 && idxVals.empty())
        for (auto d : srcTy.getShape())
          tileShape.push_back(d);
    }
    if (tileShape.empty()) {
      auto nodeType = n.GetType();
      if (auto sty = dyn_cast<SpannedType>(nodeType)) {
        auto &mdspan = sty->s_type->value;
        if (mdspan.IsValid())
          for (auto &v : mdspan.Value())
            tileShape.push_back(EvalToInt(v));
      }
    }
    if (tileShape.empty())
      tileShape = {16, 16};

    auto tileTy = coir::TensorType::get(
        &IRContext(), srcTy.getElementType(), tileShape,
        srcTy.getMemorySpace(),
        llvm::ArrayRef<int64_t>{});
    auto tileOp = builder.create<coir::TensorTileOp>(
        loc, tileTy, srcVal, idxVals);

    auto fragTy = coir::MMAFragType::get(
        &IRContext(), srcTy.getElementType(), tileShape);
    mlir::IntegerAttr swizAttr;
    auto swizMode = op.GetSwizzleMode();
    if (swizMode != SwizMode::NONE) {
      int64_t swizBytes = 0;
      switch (swizMode) {
      case SwizMode::B32: swizBytes = 32; break;
      case SwizMode::B64: swizBytes = 64; break;
      case SwizMode::B128: swizBytes = 128; break;
      default: break;
      }
      if (swizBytes > 0)
        swizAttr = builder.getI64IntegerAttr(swizBytes);
    }
    auto loadOp = builder.create<coir::MMALoadOp>(
        loc, fragTy, tileOp.getResult(), /*k_dim=*/nullptr, swizAttr);

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
        loc, accVal.getType(), accVal, lhsVal, rhsVal, layoutAttr,
        /*k_dim=*/nullptr, /*mma_atom_name=*/nullptr,
        /*reg_num_a=*/nullptr, /*reg_num_b=*/nullptr,
        /*reg_num_d=*/nullptr);

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
    // Prefer the actual block shape from ChunkAt over the fragment type shape
    llvm::SmallVector<int64_t> storeShape;
    if (dstChunk->GetBlockShape().IsValid()) {
      auto posVals = dstChunk->GetBlockShape().PosValList();
      if (posVals)
        for (auto d : *posVals) storeShape.push_back(static_cast<int64_t>(d));
    }
    if (storeShape.empty())
      storeShape.assign(fragTy.getShape().begin(), fragTy.getShape().end());

    auto tileTy = coir::TensorType::get(
        &IRContext(), dstTy.getElementType(), storeShape,
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
  // Left-rotate the .data associations to match the token rotation.
  llvm::SmallVector<mlir::Value> dataVals;
  for (auto &name : names)
    dataVals.push_back(LookupValue(name + ".data"));
  for (unsigned i = 0; i < names.size(); ++i) {
    auto rotatedData = dataVals[(i + 1) % names.size()];
    if (rotatedData)
      UpdateValue(names[i] + ".data", rotatedData);
  }
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
            subscript = STR(*sl);
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
            subscript = STR(*sl);
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

  // Allow print/println BIFs through; skip other non-lib BIFs (assert, arith)
  if (call.IsBIF() && !call.IsLibCall() &&
      fname != "print" && fname != "println")
    return true;

  // Device calls used as expressions are handled by EmitExpr
  if (call.IsExpr() && !call.IsArith() && !call.IsLibCall())
    return true;

  // Emit arguments as operands
  llvm::SmallVector<mlir::Value> operands;
  for (auto &arg : args) {
    mlir::Value v = EmitExpr(*arg);
    if (!v) {
      if (auto name = AST::GetName(*arg))
        v = LookupValue(*name);
      // .data fallback (null case): EmitExpr may return nullptr for Unary
      // "dataof" expressions (e.g., input_load_s.data used as a call arg).
      // AST::GetName also fails on dataid_expr nodes. Fall back to
      // resolving the base name and looking up <name>.data directly.
      if (!v) {
        std::string baseName;
        if (auto *expr = dyn_cast<AST::Expr>(arg.get())) {
          if (expr->GetOp() == Op::DataOf || expr->GetOp() == Op::MDataOf) {
            if (auto innerName = AST::GetName(*expr->GetR()))
              baseName = *innerName;
          }
        } else if (auto *da = dyn_cast<AST::DataAccess>(arg.get())) {
          baseName = da->GetDataName();
        }
        if (!baseName.empty()) {
          v = LookupValue(baseName + ".data");
          if (!v) v = LookupValue(baseName);
        }
      }
    }
    // .data fallback: for DMA results, the base name may resolve to a
    // non-tensor (token or element count), while <name>.data holds the
    // actual buffer.
    if (v && !mlir::isa<coir::TensorType>(v.getType())) {
      if (auto name = AST::GetName(*arg)) {
        auto dataName = *name;
        if (!llvm::StringRef(dataName).ends_with(".data"))
          dataName += ".data";
        auto bufVal = LookupValue(dataName);
        if (bufVal && mlir::isa<coir::TensorType>(bufVal.getType()))
          v = bufVal;
      }
    }
    if (v) operands.push_back(v);
  }

  // Collect template arguments as string attributes
  llvm::SmallVector<mlir::StringRef> tplStrs;
  llvm::SmallVector<std::string> tplStorage;
  if (call.template_args) {
    for (auto &ta : call.template_args->AllValues()) {
      std::string taStr;
      if (auto name = AST::GetName(*ta))
        taStr = *name;
      else if (auto lit = dyn_cast<AST::IntLiteral>(ta))
        taStr = STR(*lit);
      else if (auto expr = dyn_cast<AST::Expr>(ta);
               expr->IsReference() &&
               isa<AST::DataType>(expr->GetReference()))
        taStr = CppTypeName(
            cast<AST::DataType>(expr->GetReference())->getBaseType());
      else if (auto expr = dyn_cast<AST::Expr>(ta);
               expr->Opts().HasVal())
        taStr = STR(EvalToInt(expr->Opts().GetVal()));
      else
        choreo_unreachable("unexpected template argument type in call");
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
    auto existing = IRModule()->getAttrOfType<mlir::StringAttr>("coir.user_cpp_code");
    std::string combined = existing ? existing.getValue().str() : "";
    combined += n.GetCode();
    IRModule()->setAttr("coir.user_cpp_code",
                        mlir::StringAttr::get(&IRContext(), combined));
  } else if (n.kind == AST::CppSourceCode::Device) {
    auto existing =
        IRModule()->getAttrOfType<mlir::StringAttr>("coir.explicit_device_code");
    std::string combined = existing ? existing.getValue().str() : "";
    combined += n.GetCode();
    IRModule()->setAttr("coir.explicit_device_code",
                        mlir::StringAttr::get(&IRContext(), combined));
  }
  return true;
}

bool ASTCoIRGen::Visit(AST::InThreadsBlock &n) {
  auto loc = Loc(n);
  if (!n.pred) return true;

  mlir::Value predVal = EmitExpr(*n.pred);
  if (!predVal)
    predVal = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

  if (!predVal.getType().isInteger(1)) {
    mlir::Value zero;
    if (mlir::isa<mlir::IndexType>(predVal.getType()))
      zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    else
      zero = builder.create<mlir::arith::ConstantIntOp>(
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
