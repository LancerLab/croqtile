//===- CoIREmitterBase.cpp - Unified CoIR emitter base --------------------===//
//
// Shared implementations for all CoIR source-text emitters.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/CoIREmitterBase.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace coir;

// ===== CodeGen static helpers (defined here to keep header-only) =====

void CoIR::CodeGen::emitScriptPrologue(llvm::raw_ostream &os,
                                       llvm::StringRef comment,
                                       llvm::StringRef tmpSuffix) {
  auto &sctx = CoIR::ScriptContext::Get();
  bool has_embedded = sctx.types_header && sctx.runtime_header;

  os << "#!/usr/bin/env bash\n";
  os << "# CoIR generated script -- " << comment << "\n";
  os << "set -eo pipefail\n\n";

  os << "TMPDIR=$(mktemp -d /tmp/cocc" << tmpSuffix << "_XXXXXX)\n";
  os << "trap 'rm -rf $TMPDIR' EXIT\n\n";

  if (has_embedded) {
    os << "cat > \"$TMPDIR/choreo_types.h\" << '__COCC_TYPES_HEADER__'\n";
    os << sctx.types_header;
    os << "\n__COCC_TYPES_HEADER__\n\n";

    os << "cat > \"$TMPDIR/choreo.h\" << '__COCC_CHOREO_HEADER__'\n";
    os << sctx.runtime_header;
    os << "\n__COCC_CHOREO_HEADER__\n\n";

    if (sctx.types_cute_header) {
      os << "cat > \"$TMPDIR/choreo_types_cute.h\" "
            "<< '__COCC_TYPES_CUTE_HEADER__'\n";
      os << sctx.types_cute_header;
      os << "\n__COCC_TYPES_CUTE_HEADER__\n\n";
    }
    if (sctx.cute_header) {
      os << "cat > \"$TMPDIR/choreo_cute.h\" << '__COCC_CUTE_HEADER__'\n";
      os << sctx.cute_header;
      os << "\n__COCC_CUTE_HEADER__\n\n";
    }
  } else {
    os << "if [[ -z \"${CHOREO_ROOT:-}\" ]]; then\n";
    os << "  _coir_bin=$(which coir-codegen 2>/dev/null || true)\n";
    os << "  if [[ -n \"$_coir_bin\" ]]; then\n";
    os << "    CHOREO_ROOT=\"$(cd \"$(dirname \"$_coir_bin\")\" && "
          "git rev-parse --show-toplevel 2>/dev/null || true)\"\n";
    os << "  fi\n";
    os << "  if [[ -z \"${CHOREO_ROOT:-}\" ]]; then\n";
    os << "    CHOREO_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && "
          "git rev-parse --show-toplevel 2>/dev/null || "
          "echo \"$(dirname \"${BASH_SOURCE[0]}\")\")\" \n";
    os << "  fi\n";
    os << "fi\n";
    os << "CHOREO_INC=\"${CHOREO_ROOT}/runtime\"\n\n";
  }

  if (!sctx.build_env.empty()) os << sctx.build_env;
  if (!sctx.target_setup.empty()) os << sctx.target_setup << "\n";
}

// ===== CoIREmitterBase -- CodeGen pipeline =====

int CoIREmitterBase::EmitSource(ModuleOp module, llvm::StringRef /*arch*/,
                                llvm::raw_ostream &os) {
  os_ = &os;
  resetState();
  emitModule(module, os);
  emitHostCode(module, os);
  return 0;
}

void CoIREmitterBase::resetState() {
  indent = 0;
  valueNames.clear();
  returnParamNames.clear();
  returnValues.clear();
  nextId = 0;
}

// ===== Indentation =====

std::string CoIREmitterBase::getIndent() {
  return std::string(indent * 2, ' ');
}

void CoIREmitterBase::incIndent() { indent++; }
void CoIREmitterBase::decIndent() { if (indent > 0) indent--; }

// ===== Value naming =====

std::string CoIREmitterBase::getName(Value v) {
  auto it = valueNames.find(v);
  if (it != valueNames.end()) return it->second;
  std::string name = "v" + std::to_string(nextId++);
  valueNames[v] = name;
  return name;
}

// ===== Element type emission =====

std::string CoIREmitterBase::emitElementType(Type ty) {
  if (ty.isF16()) return "half";
  if (ty.isF32()) return "float";
  if (ty.isF64()) return "double";
  if (ty.isInteger(8)) return "uint8_t";
  if (ty.isInteger(16)) return "int16_t";
  if (ty.isInteger(32)) return "int32_t";
  if (ty.isInteger(64)) return "int64_t";
  llvm_unreachable("unsupported element type");
}

// ===== Tensor helpers =====

int64_t CoIREmitterBase::getTensorBytes(TensorType tty) {
  int64_t n = 1;
  for (auto d : tty.getShape()) n *= d;
  Type eTy = tty.getElementType();
  int64_t elemSize = 4;
  if (eTy.isF16() || eTy.isInteger(16)) elemSize = 2;
  else if (eTy.isF64() || eTy.isInteger(64)) elemSize = 8;
  else if (eTy.isInteger(8)) elemSize = 1;
  return n * elemSize;
}

void CoIREmitterBase::emitLinearIndex(ValueRange indices, TensorType tty) {
  auto strides = tty.getStrides();
  auto shape = tty.getShape();
  if (indices.empty()) {
    os() << "0";
    return;
  }
  if (indices.size() == 1 && strides.empty()) {
    os() << getName(indices[0]);
    return;
  }
  llvm::SmallVector<int64_t> effectiveStrides;
  if (!strides.empty()) {
    effectiveStrides.assign(strides.begin(), strides.end());
  } else {
    effectiveStrides.resize(shape.size());
    int64_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      effectiveStrides[i] = s;
      s *= shape[i];
    }
  }
  for (unsigned i = 0; i < indices.size(); ++i) {
    if (i > 0) os() << " + ";
    if (i < effectiveStrides.size() && effectiveStrides[i] != 1)
      os() << getName(indices[i]) << " * " << effectiveStrides[i];
    else
      os() << getName(indices[i]);
  }
}

// ===== Return value prescan =====

void CoIREmitterBase::prescanReturnValues(KernelOp kernel) {
  auto &body = kernel.getBody();
  for (auto &op : body.front().getOperations()) {
    if (auto ret = dyn_cast<KernelReturnOp>(op)) {
      for (unsigned i = 0; i < ret.getOperands().size(); ++i) {
        returnValues.insert(ret.getOperands()[i]);
        valueNames[ret.getOperands()[i]] = returnParamNames[i];
      }
    }
  }
}

// ===== Op dispatch (visitor pattern) =====

void CoIREmitterBase::emitOp(Operation *op) {
  if (auto parallel = dyn_cast<ParallelOp>(op))
    emitParallel(parallel);
  else if (auto foreach_ = dyn_cast<ForeachOp>(op))
    emitForeach(foreach_);
  else if (auto loadElem = dyn_cast<TensorLoadElemOp>(op))
    emitTensorLoadElem(loadElem);
  else if (auto storeElem = dyn_cast<TensorStoreElemOp>(op))
    emitTensorStoreElem(storeElem);
  else if (auto fill = dyn_cast<MMAFillOp>(op))
    emitMMAFill(fill);
  else if (auto load = dyn_cast<MMALoadOp>(op))
    emitMMALoad(load);
  else if (auto exec = dyn_cast<MMAExecOp>(op))
    emitMMAExec(exec);
  else if (auto store = dyn_cast<MMAStoreOp>(op))
    emitMMAStore(store);
  else if (auto constDesc = dyn_cast<DMAConstDescOp>(op))
    emitDMAConstDesc(constDesc);
  else if (auto prefetch = dyn_cast<DMADescPrefetchOp>(op))
    emitDMAPrefetch(prefetch);
  else if (auto rtDesc = dyn_cast<DMADescRuntimeOp>(op))
    emitDMARuntimeDesc(rtDesc);
  else if (auto invoke = dyn_cast<DMAInvokeOp>(op))
    emitDMAInvoke(invoke);
  else if (auto dmaCopy = dyn_cast<DmaCopyOp>(op))
    emitDmaCopy(dmaCopy);
  else if (auto barrier = dyn_cast<BarrierOp>(op))
    emitBarrier(barrier);
  else if (auto wait = dyn_cast<WaitOp>(op))
    emitWait(wait);
  else if (auto rotate = dyn_cast<FutureRotateOp>(op))
    emitFutureRotate(rotate);
  else if (auto atomicOp = dyn_cast<AtomicOp>(op))
    emitAtomic(atomicOp);
  else if (auto reduceElem = dyn_cast<TensorReduceElemOp>(op))
    emitTensorReduceElem(reduceElem);
  else if (auto alloc = dyn_cast<TensorAllocOp>(op))
    emitTensorAlloc(alloc);
  else if (auto tile = dyn_cast<TensorTileOp>(op))
    emitTensorTile(tile);
  else if (auto ret = dyn_cast<KernelReturnOp>(op))
    emitKernelReturn(ret);
  else if (auto yield = dyn_cast<YieldOp>(op))
    emitYield(yield);
  else if (auto constOp = dyn_cast<arith::ConstantOp>(op))
    emitConstant(constOp);
  else if (auto indexCast = dyn_cast<arith::IndexCastOp>(op))
    valueNames[indexCast.getResult()] = getName(indexCast.getIn());
  else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op))
    emitIfOp(ifOp);
  else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op))
    emitWhileOp(whileOp);
  else if (auto coirWhile = dyn_cast<CoIRWhileOp>(op))
    emitCoIRWhileOp(coirWhile);
  else if (isa<CoIRWhileCondOp>(op))
    (void)op;
  else if (auto breakOp = dyn_cast<CoIRBreakOp>(op))
    emitBreak(breakOp);
  else if (auto contOp = dyn_cast<CoIRContinueOp>(op))
    emitContinue(contOp);
  else if (isa<mlir::scf::YieldOp>(op) || isa<mlir::scf::ConditionOp>(op))
    (void)op;
  else if (emitArithBinOp(op)) {}
  else if (emitCmpOp(op)) {}
  else if (auto selectOp = dyn_cast<arith::SelectOp>(op))
    emitSelect(selectOp);
  else if (isa<DMACheckOp>(op))
    (void)op;
  else
    emitOpFallback(op);
}

// ===== Common op implementations =====

void CoIREmitterBase::emitConstant(arith::ConstantOp op) {
  std::string name = getName(op.getResult());
  if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
    os() << getIndent() << "const int " << name << " = "
         << intAttr.getInt() << ";\n";
  } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
    os() << getIndent() << "const " << emitElementType(op.getType())
         << " " << name << " = ";
    llvm::SmallString<16> strVal;
    floatAttr.getValue().toString(strVal, 6, 0);
    os() << strVal << ";\n";
  } else {
    os() << getIndent() << "auto " << name << " = /* constant */;\n";
  }
}

bool CoIREmitterBase::emitArithBinOp(Operation *op) {
  llvm::StringRef opStr;
  if (isa<arith::AddIOp>(op) || isa<arith::AddFOp>(op)) opStr = "+";
  else if (isa<arith::SubIOp>(op) || isa<arith::SubFOp>(op)) opStr = "-";
  else if (isa<arith::MulIOp>(op) || isa<arith::MulFOp>(op)) opStr = "*";
  else if (isa<arith::DivSIOp>(op) || isa<arith::DivFOp>(op)) opStr = "/";
  else if (isa<arith::RemSIOp>(op)) opStr = "%";
  else return false;

  std::string name = getName(op->getResult(0));
  std::string lhs = getName(op->getOperand(0));
  std::string rhs = getName(op->getOperand(1));
  os() << getIndent() << emitType(op->getResult(0).getType()) << " "
       << name << " = " << lhs << " " << opStr << " " << rhs << ";\n";
  return true;
}

bool CoIREmitterBase::emitCmpOp(Operation *op) {
  if (auto cmpI = dyn_cast<arith::CmpIOp>(op)) {
    std::string name = getName(cmpI.getResult());
    std::string lhs = getName(cmpI.getLhs());
    std::string rhs = getName(cmpI.getRhs());
    llvm::StringRef opStr;
    switch (cmpI.getPredicate()) {
    case arith::CmpIPredicate::eq:  opStr = "=="; break;
    case arith::CmpIPredicate::ne:  opStr = "!="; break;
    case arith::CmpIPredicate::slt: opStr = "<"; break;
    case arith::CmpIPredicate::sle: opStr = "<="; break;
    case arith::CmpIPredicate::sgt: opStr = ">"; break;
    case arith::CmpIPredicate::sge: opStr = ">="; break;
    case arith::CmpIPredicate::ult: opStr = "<"; break;
    case arith::CmpIPredicate::ule: opStr = "<="; break;
    case arith::CmpIPredicate::ugt: opStr = ">"; break;
    case arith::CmpIPredicate::uge: opStr = ">="; break;
    }
    os() << getIndent() << "bool " << name << " = (" << lhs << " "
         << opStr << " " << rhs << ");\n";
    return true;
  }
  if (auto cmpF = dyn_cast<arith::CmpFOp>(op)) {
    std::string name = getName(cmpF.getResult());
    std::string lhs = getName(cmpF.getLhs());
    std::string rhs = getName(cmpF.getRhs());
    llvm::StringRef opStr;
    switch (cmpF.getPredicate()) {
    case arith::CmpFPredicate::OEQ: opStr = "=="; break;
    case arith::CmpFPredicate::OGT: opStr = ">"; break;
    case arith::CmpFPredicate::OGE: opStr = ">="; break;
    case arith::CmpFPredicate::OLT: opStr = "<"; break;
    case arith::CmpFPredicate::OLE: opStr = "<="; break;
    default: opStr = "!="; break;
    }
    os() << getIndent() << "bool " << name << " = (" << lhs << " "
         << opStr << " " << rhs << ");\n";
    return true;
  }
  return false;
}

void CoIREmitterBase::emitIfOp(mlir::scf::IfOp op) {
  for (auto res : op.getResults())
    os() << getIndent() << emitType(res.getType()) << " "
         << getName(res) << ";\n";

  os() << getIndent() << "if (" << getName(op.getCondition()) << ") {\n";
  incIndent();
  for (auto &bodyOp : op.getThenRegion().front().getOperations()) {
    if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i)
        os() << getIndent() << getName(op.getResult(i)) << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
    } else {
      emitOp(&bodyOp);
    }
  }
  decIndent();
  os() << getIndent() << "}\n";
  if (!op.getElseRegion().empty()) {
    os() << getIndent() << "else {\n";
    incIndent();
    for (auto &bodyOp : op.getElseRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i)
          os() << getIndent() << getName(op.getResult(i)) << " = "
               << getName(yieldOp.getOperand(i)) << ";\n";
      } else {
        emitOp(&bodyOp);
      }
    }
    decIndent();
    os() << getIndent() << "}\n";
  }
}

void CoIREmitterBase::emitBreak(CoIRBreakOp op) {
  os() << getIndent() << "break;\n";
}

void CoIREmitterBase::emitContinue(CoIRContinueOp op) {
  os() << getIndent() << "continue;\n";
}

void CoIREmitterBase::emitSelect(arith::SelectOp op) {
  std::string name = getName(op.getResult());
  os() << getIndent() << emitType(op.getResult().getType()) << " "
       << name << " = " << getName(op.getCondition()) << " ? "
       << getName(op.getTrueValue()) << " : "
       << getName(op.getFalseValue()) << ";\n";
}

void CoIREmitterBase::emitTensorLoadElem(TensorLoadElemOp op) {
  std::string name = getName(op.getResult());
  std::string src = getName(op.getSource());
  auto tty = cast<TensorType>(op.getSource().getType());
  os() << getIndent() << emitType(op.getResult().getType()) << " " << name
       << " = " << src << "[";
  emitLinearIndex(op.getIndices(), tty);
  os() << "];\n";
}

void CoIREmitterBase::emitTensorStoreElem(TensorStoreElemOp op) {
  std::string dst = getName(op.getDest());
  std::string val = getName(op.getValue());
  auto tty = cast<TensorType>(op.getDest().getType());
  os() << getIndent() << dst << "[";
  emitLinearIndex(op.getIndices(), tty);
  os() << "] = " << val << ";\n";
}

// ===== Overridable ops with defaults =====

void CoIREmitterBase::emitWhileOp(mlir::scf::WhileOp op) {
  auto &beforeBlock = op.getBefore().front();
  auto condOp =
      dyn_cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());

  for (unsigned i = 0; i < op.getInits().size(); ++i)
    valueNames[beforeBlock.getArgument(i)] = getName(op.getInits()[i]);

  llvm::SmallVector<std::string> iterVarNames;
  for (unsigned i = 0; i < op.getInits().size(); ++i)
    iterVarNames.push_back(getName(op.getInits()[i]));

  for (auto &bodyOp : beforeBlock.getOperations()) {
    if (isa<mlir::scf::ConditionOp>(&bodyOp)) continue;
    emitOp(&bodyOp);
  }

  os() << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
  incIndent();

  auto &afterBlock = op.getAfter().front();
  for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
    valueNames[afterBlock.getArgument(i)] = getName(condOp.getArgs()[i]);

  for (auto &bodyOp : afterBlock.getOperations()) {
    if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
      for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
        os() << getIndent() << iterVarNames[i] << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
        valueNames[beforeBlock.getArgument(i)] =
            getName(yieldOp.getOperand(i));
      }
      for (auto &bOp : beforeBlock.getOperations()) {
        if (isa<mlir::scf::ConditionOp>(&bOp)) continue;
        emitOp(&bOp);
      }
    } else {
      emitOp(&bodyOp);
    }
  }

  decIndent();
  os() << getIndent() << "}\n";

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    if (condOp && i < condOp.getArgs().size())
      valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
  }
}

void CoIREmitterBase::emitCoIRWhileOp(CoIRWhileOp op) {
  auto &condBlock = op.getCondRegion().front();
  auto condOp = dyn_cast<CoIRWhileCondOp>(condBlock.getTerminator());

  for (unsigned i = 0; i < op.getInits().size(); ++i)
    valueNames[condBlock.getArgument(i)] = getName(op.getInits()[i]);

  llvm::SmallVector<std::string> iterVarNames;
  for (unsigned i = 0; i < op.getInits().size(); ++i)
    iterVarNames.push_back(getName(op.getInits()[i]));

  for (auto &bodyOp : condBlock.getOperations()) {
    if (isa<CoIRWhileCondOp>(&bodyOp)) continue;
    emitOp(&bodyOp);
  }

  os() << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
  incIndent();

  auto &bodyBlock = op.getBodyRegion().front();
  for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
    valueNames[bodyBlock.getArgument(i)] = getName(condOp.getArgs()[i]);

  for (auto &bodyOp : bodyBlock.getOperations()) {
    if (auto breakOp = dyn_cast<CoIRBreakOp>(&bodyOp)) {
      for (unsigned i = 0; i < breakOp.getOperands().size(); ++i) {
        os() << getIndent() << iterVarNames[i] << " = "
             << getName(breakOp.getOperand(i)) << ";\n";
        valueNames[condBlock.getArgument(i)] =
            getName(breakOp.getOperand(i));
      }
      os() << getIndent() << "break;\n";
    } else if (auto contOp = dyn_cast<CoIRContinueOp>(&bodyOp)) {
      for (unsigned i = 0; i < contOp.getOperands().size(); ++i) {
        os() << getIndent() << iterVarNames[i] << " = "
             << getName(contOp.getOperand(i)) << ";\n";
        valueNames[condBlock.getArgument(i)] =
            getName(contOp.getOperand(i));
      }
      for (auto &cOp : condBlock.getOperations()) {
        if (isa<CoIRWhileCondOp>(&cOp)) continue;
        emitOp(&cOp);
      }
      os() << getIndent() << "continue;\n";
    } else {
      emitOp(&bodyOp);
    }
  }

  decIndent();
  os() << getIndent() << "}\n";

  for (unsigned i = 0; i < op.getNumResults(); ++i) {
    if (condOp && i < condOp.getArgs().size())
      valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
  }
}

void CoIREmitterBase::emitForeach(ForeachOp op) {
  auto &body = op.getBody();
  auto args = body.front().getArguments();
  std::string iv = getName(args[0]);
  std::string ub = getName(op.getUpperBound());

  auto iterArgs = op.getIterArgs();
  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    std::string iterName = getName(args[i + 1]);
    os() << getIndent() << "auto " << iterName << " = "
         << getName(iterArgs[i]) << ";\n";
  }

  os() << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
       << ub << "; ++" << iv << ") {\n";
  incIndent();
  for (auto &bodyOp : body.front().getOperations())
    emitOp(&bodyOp);
  decIndent();
  os() << getIndent() << "}\n";

  for (unsigned i = 0; i < op.getResults().size(); ++i)
    valueNames[op.getResult(i)] = getName(args[i + 1]);
}

void CoIREmitterBase::emitYield(YieldOp op) {
  auto *parentOp = op->getParentOp();
  if (auto foreachOp = dyn_cast<ForeachOp>(parentOp)) {
    auto args = foreachOp.getBody().front().getArguments();
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      auto iterArgName = getName(args[i + 1]);
      auto yieldValName = getName(op.getOperands()[i]);
      if (iterArgName != yieldValName)
        os() << getIndent() << iterArgName << " = " << yieldValName
             << ";\n";
    }
  }
}

void CoIREmitterBase::emitTensorAlloc(TensorAllocOp op) {
  if (returnValues.count(op.getResult()))
    return;

  auto tensorTy = cast<TensorType>(op.getResult().getType());
  std::string name = getName(op.getResult());
  int64_t totalElems = 1;
  for (auto d : tensorTy.getShape()) totalElems *= d;

  std::string qualifier = getAllocQualifier(tensorTy);
  if (needsTMAAlignment(tensorTy)) {
    int64_t totalBytes =
        totalElems * (tensorTy.getElementType().getIntOrFloatBitWidth() / 8);
    os() << getIndent() << qualifier << "alignas(128) unsigned char "
         << name << "_storage[" << totalBytes << "];\n";
    os() << getIndent() << emitElementType(tensorTy.getElementType())
         << "* " << name << " = ("
         << emitElementType(tensorTy.getElementType()) << "*)("
         << name << "_storage);\n";
  } else {
    os() << getIndent() << qualifier
         << emitElementType(tensorTy.getElementType())
         << " " << name << "[" << totalElems << "];\n";
  }
}

void CoIREmitterBase::emitTensorTile(TensorTileOp op) {
  std::string base = getName(op.getSource());
  std::string name = getName(op.getResult());
  auto tty = cast<TensorType>(op.getSource().getType());
  os() << getIndent() << "auto* " << name << " = &" << base << "[";
  emitLinearIndex(op.getIndices(), tty);
  os() << "];\n";
}

// ===== Fallback =====

void CoIREmitterBase::emitOpFallback(Operation *op) {
  os() << getIndent() << "// [unhandled] " << op->getName().getStringRef()
       << "\n";
}

// ===== Alloc qualifier hooks =====

std::string CoIREmitterBase::getAllocQualifier(TensorType tty) {
  return tty.getMemorySpace() == 1 ? "__shared__ " : "";
}

bool CoIREmitterBase::needsTMAAlignment(TensorType tty) {
  return false;
}
