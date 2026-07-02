//===- EmitCC.cpp - Emit portable C++ source from CoIR IR -----------------===//
//
// Walks the module and emits C++ source for CPU execution. Parallel levels
// become nested for-loops, DMA becomes std::memcpy, and MMA falls back to
// software loop nests.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "CodeGen/CoIREmitterBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace coir {
#define GEN_PASS_DECL_EMITCC
#define GEN_PASS_DEF_EMITCC
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

class CCEmitter : public coir::CoIREmitterBase {
public:
  CCEmitter() = default;

  void emitModule(ModuleOp module, llvm::raw_ostream &os) override {
    os_ = &os;
    resetState();
    emitHeader();
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitKernel(kernel);
    }
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitHostEntry(kernel);
    }
  }

  int EmitScript(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    auto &sctx = CoIR::ScriptContext::Get();
    bool has_embedded = sctx.types_header && sctx.runtime_header;

    emitScriptPrologue(os, "compile and execute CPU kernel", "_cc");

    os << "TMPFILE=\"$TMPDIR/kernel.cpp\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_CC_SOURCE__'\n";
    EmitSource(module, "", os);

    os << "\n__COCC_CC_SOURCE__\n\n";
    if (has_embedded) {
      os << "g++ -std=c++17 -O2 -pthread "
            "-I\"$TMPDIR\" -o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    } else {
      os << "g++ -std=c++17 -O2 -pthread "
            "-I\"$CHOREO_INC\" -I\"$TMPDIR\" "
            "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    }
    emitScriptExecuteBlock(os);
    return 0;
  }

private:
  struct EntryAssertion {
    AssertOp op;
  };
  llvm::SmallVector<EntryAssertion> entryAssertions;

  std::string emitType(Type ty) override {
    if (auto tensorTy = dyn_cast<coir::TensorType>(ty))
      return emitElementType(tensorTy.getElementType()) + "*";
    if (isa<coir::AsyncTokenType>(ty))
      return "int";
    if (ty.isIndex())
      return "int";
    if (ty.isBF16())
      return "choreo::bf16";
    if (ty.isF16())
      return "choreo::f16";
    if (ty.isF32())
      return "float";
    if (ty.isF64())
      return "double";
    if (ty.isInteger(1))
      return "bool";
    if (ty.isInteger(8))
      return "uint8_t";
    if (ty.isInteger(16))
      return "int16_t";
    if (ty.isInteger(32))
      return "int";
    if (ty.isInteger(64))
      return "long long";
    return "/* unknown type */";
  }

  std::string emitElementType(Type ty) override {
    if (ty.isBF16()) return "choreo::bf16";
    if (ty.isF16()) return "choreo::f16";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isInteger(8)) return "uint8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int32_t";
    if (ty.isInteger(64)) return "int64_t";
    return CoIREmitterBase::emitElementType(ty);
  }

  void emitHeader() {
    os() << "#define __CHOREO_TARGET_CPU__ 1\n";
    os() << "#include \"choreo.h\"\n";
    os() << "#include <cstring>\n";
    os() << "#include <cstdlib>\n\n";
  }

  std::string emitChoreoType(Type ty, bool asView = true) {
    if (auto tty = dyn_cast<coir::TensorType>(ty)) {
      std::string choreoElem;
      auto eTyML = tty.getElementType();
      if (eTyML.isInteger(8)) choreoElem = "choreo::u8";
      else if (eTyML.isInteger(16)) choreoElem = "choreo::s16";
      else if (eTyML.isInteger(32)) choreoElem = "choreo::s32";
      else if (eTyML.isInteger(64)) choreoElem = "choreo::s64";
      else if (eTyML.isBF16()) choreoElem = "choreo::bf16";
      else if (eTyML.isF32()) choreoElem = "choreo::f32";
      else if (eTyML.isF16()) choreoElem = "choreo::f16";
      else if (eTyML.isF64()) choreoElem = "choreo::f64";
      else choreoElem = "choreo::s32";
      unsigned ndim = tty.getShape().size();
      if (asView)
        return "const choreo::spanned_view<" + choreoElem + ", " +
               std::to_string(ndim) + "> &";
      else
        return "choreo::spanned_data<" + choreoElem + ", " +
               std::to_string(ndim) + ">";
    }
    return "/* unknown */";
  }

  std::string getAllocQualifier(coir::TensorType /*tty*/) override {
    return "";
  }

  bool needsTMAAlignment(coir::TensorType /*tty*/) override { return false; }

  void emitOpFallback(mlir::Operation *op) override {
    if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      os() << getIndent() << "// TMA not supported on CPU target\n";
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else if (auto callOp = dyn_cast<coir::CallOp>(op))
      emitCall(callOp);
    else if (auto ithOp = dyn_cast<coir::InThreadsOp>(op))
      emitInThreads(ithOp);
    else
      CoIREmitterBase::emitOpFallback(op);
  }

  // -- MMA: software fallback (not supported on CPU) --
  void emitMMAFill(MMAFillOp /*op*/) override {
    os() << getIndent() << "// MMA not supported on CPU target\n";
  }
  void emitMMALoad(MMALoadOp /*op*/) override {
    os() << getIndent() << "// MMA not supported on CPU target\n";
  }
  void emitMMAExec(MMAExecOp /*op*/) override {
    os() << getIndent() << "// MMA not supported on CPU target\n";
  }
  void emitMMAStore(MMAStoreOp /*op*/) override {
    os() << getIndent() << "// MMA not supported on CPU target\n";
  }

  // -- DMA descriptor pipeline: no-op on CPU --
  void emitDMAConstDesc(DMAConstDescOp /*op*/) override {}
  void emitDMAPrefetch(DMADescPrefetchOp /*op*/) override {}
  void emitDMARuntimeDesc(DMADescRuntimeOp /*op*/) override {}
  void emitDMAInvoke(DMAInvokeOp op) override {
    os() << getIndent() << "// DMA invoke (descriptor-based, no-op on CPU)\n";
    valueNames[op.getDone()] = "0";
  }

  void emitKernel(KernelOp kernel) {
    entryAssertions.clear();

    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();

    os() << "void __" << symName << "_impl__(";

    auto &body = kernel.getBody();
    unsigned paramIdx = 0;
    if (!body.empty()) {
      auto args = body.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (paramIdx > 0) os() << ", ";
        std::string name = "arg" + std::to_string(paramIdx);
        valueNames[args[i]] = name;
        os() << emitType(fnType.getInput(i)) << " " << name;
        paramIdx++;
      }
    }
    for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
      if (paramIdx > 0) os() << ", ";
      std::string name = "out" + std::to_string(i);
      os() << emitType(fnType.getResult(i)) << " " << name;
      returnParamNames[i] = name;
      paramIdx++;
    }
    os() << ") {\n";
    incIndent();

    prescanReturnValues(kernel);

    for (auto &op : body.front().getOperations())
      emitOp(&op);

    decIndent();
    os() << "}\n\n";
  }

  void emitHostEntry(KernelOp kernel) override {
    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();

    if (numResults == 0) {
      os() << "void " << symName << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << emitChoreoType(fnType.getInput(i), true) << " p" << i;
      }
      os() << ") {\n";
      emitEntryAssertions(kernel);
      os() << "  __" << symName << "_impl__(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty)
          os() << "p" << i << ".data()";
        else
          os() << "p" << i;
      }
      os() << ");\n";
      os() << "}\n\n";
      return;
    }

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    os() << emitChoreoType(fnType.getResult(0), false) << " " << symName
         << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os() << ", ";
      os() << emitChoreoType(fnType.getInput(i), true) << " p" << i;
    }
    os() << ") {\n";

    std::string shapeStr;
    {
      llvm::raw_string_ostream ss(shapeStr);
      ss << "{";
      for (unsigned d = 0; d < resTy.getShape().size(); ++d) {
        if (d > 0) ss << ", ";
        ss << resTy.getShape()[d];
      }
      ss << "}";
    }
    std::string choreoElem;
    auto resElemTy = resTy.getElementType();
    if (resElemTy.isInteger(8)) choreoElem = "choreo::u8";
    else if (resElemTy.isInteger(16)) choreoElem = "choreo::s16";
    else if (resElemTy.isInteger(32)) choreoElem = "choreo::s32";
    else if (resElemTy.isInteger(64)) choreoElem = "choreo::s64";
    else if (resElemTy.isBF16()) choreoElem = "choreo::bf16";
    else if (resElemTy.isF32()) choreoElem = "choreo::f32";
    else if (resElemTy.isF16()) choreoElem = "choreo::f16";
    else if (resElemTy.isF64()) choreoElem = "choreo::f64";
    else choreoElem = "choreo::s32";

    os() << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
         << resTy.getShape().size() << ">(" << shapeStr << ");\n";

    emitEntryAssertions(kernel);
    os() << "  __" << symName << "_impl__(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os() << ", ";
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty)
        os() << "p" << i << ".data()";
      else
        os() << "p" << i;
    }
    os() << ", __result.data());\n";
    os() << "  return __result;\n";
    os() << "}\n\n";
  }

  void emitAssert(AssertOp op) {
    if (auto ea = op->getAttrOfType<BoolAttr>("enabled"))
      if (!ea.getValue()) return;
    auto site = op.getSite();
    auto msg = op.getMessage().str();
    if (site == AssertSite::ENTRY) {
      entryAssertions.push_back({op});
      return;
    }
    os() << getIndent() << "choreo::choreo_assert(" << getName(op.getCondition())
         << ", \"" << msg << "\");\n";
  }

  std::string emitExprInHostScope(Value v, KernelOp kernel,
                                  DenseMap<Value, std::string> &hostNames) {
    auto it = hostNames.find(v);
    if (it != hostNames.end()) return it->second;

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner()->getParentOp() == kernel.getOperation()) {
        std::string name = "p" + std::to_string(arg.getArgNumber());
        hostNames[v] = name;
        return name;
      }
    }

    auto *defOp = v.getDefiningOp();
    if (!defOp) return "/* unknown */";

    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      std::string val;
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
        val = std::to_string(intAttr.getInt());
      else
        val = "/* const */";
      hostNames[v] = val;
      return val;
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      auto lhs = emitExprInHostScope(cmpOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(cmpOp.getRhs(), kernel, hostNames);
      const char *pred = "==";
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq: pred = "=="; break;
      case arith::CmpIPredicate::ne: pred = "!="; break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult: pred = "<"; break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule: pred = "<="; break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt: pred = ">"; break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge: pred = ">="; break;
      }
      std::string result = "(" + lhs + " " + pred + " " + rhs + ")";
      hostNames[v] = result;
      return result;
    }

    if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
      auto lhs = emitExprInHostScope(addOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(addOp.getRhs(), kernel, hostNames);
      return (hostNames[v] = "(" + lhs + " + " + rhs + ")");
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      auto lhs = emitExprInHostScope(mulOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(mulOp.getRhs(), kernel, hostNames);
      return (hostNames[v] = "(" + lhs + " * " + rhs + ")");
    }
    if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      auto lhs = emitExprInHostScope(subOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(subOp.getRhs(), kernel, hostNames);
      return (hostNames[v] = "(" + lhs + " - " + rhs + ")");
    }
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames);
      return (hostNames[v] = "(" + lhs + " / " + rhs + ")");
    }

    return "/* unsupported expr */";
  }

  void emitEntryAssertions(KernelOp kernel) {
    DenseMap<Value, std::string> hostNames;
    for (auto &ea : entryAssertions) {
      auto cond =
          emitExprInHostScope(ea.op.getCondition(), kernel, hostNames);
      os() << "  choreo::runtime_check(" << cond << ", \""
           << ea.op.getMessage() << "\");\n";
    }
  }

  void emitCall(CallOp op) {
    auto callee = op.getCallee().str();
    bool isExpr = op.getIsExpr() && *op.getIsExpr() && op.getResult();

    os() << getIndent();
    if (isExpr) {
      auto resTy = op.getResult().getType();
      os() << emitType(resTy) << " " << getName(op.getResult()) << " = ";
    }
    os() << callee;

    if (auto tplArgs = op.getTemplateArgs()) {
      os() << "<";
      bool first = true;
      for (auto a : *tplArgs) {
        if (!first) os() << ", ";
        first = false;
        os() << mlir::cast<mlir::StringAttr>(a).getValue().str();
      }
      os() << ">";
    }

    os() << "(";
    bool first = true;
    for (auto arg : op.getOperands_()) {
      if (!first) os() << ", ";
      first = false;
      auto ty = arg.getType();
      if (auto tty = mlir::dyn_cast<coir::TensorType>(ty))
        os() << "(" << emitElementType(tty.getElementType()) << "*)"
             << getName(arg);
      else
        os() << getName(arg);
    }
    os() << ");\n";
  }

  void emitInThreads(InThreadsOp op) {
    os() << getIndent() << "if (" << getName(op.getPredicate()) << ") {\n";
    incIndent();
    for (auto &bodyOp : op.getBody().front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitAtomic(AtomicOp op) override {
    auto dstTy = mlir::cast<coir::TensorType>(op.getDest().getType());
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());

    using AK = coir::AtomicKind;
    switch (op.getKind()) {
    case AK::Add:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] += " << val << ";\n";
      break;
    case AK::Sub:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] -= " << val << ";\n";
      break;
    case AK::Exch:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] = " << val << ";\n";
      break;
    case AK::Min:
      os() << getIndent() << "if (" << val << " < " << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "]) " << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] = " << val << ";\n";
      break;
    case AK::Max:
      os() << getIndent() << "if (" << val << " > " << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "]) " << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] = " << val << ";\n";
      break;
    case AK::And:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] &= " << val << ";\n";
      break;
    case AK::Or:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] |= " << val << ";\n";
      break;
    case AK::Xor:
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), dstTy);
      os() << "] ^= " << val << ";\n";
      break;
    case AK::CAS:
      if (auto cmpAttr = op.getCompare()) {
        std::string cmp;
        if (auto intAttr = dyn_cast<IntegerAttr>(*cmpAttr))
          cmp = std::to_string(intAttr.getInt());
        else
          cmp = "/* compare */";
        os() << getIndent() << "if (" << dst << "[";
        emitLinearIndex(op.getIndices(), dstTy);
        os() << "] == " << cmp << ") " << dst << "[";
        emitLinearIndex(op.getIndices(), dstTy);
        os() << "] = " << val << ";\n";
      }
      break;
    }
  }

  void emitParallel(ParallelOp op) override {
    auto level = op.getLevel();
    auto bounds = op.getBounds();
    auto &body = op.getBody();
    auto args = body.getArguments();

    if (level == ParallelLevel::BLOCK || level == ParallelLevel::THREAD ||
        level == ParallelLevel::GROUP) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string iv = getName(args[i]);
        os() << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
             << bounds[i] << "; ++" << iv << ") {\n";
        incIndent();
      }
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      for (unsigned i = 0; i < args.size(); ++i) {
        decIndent();
        os() << getIndent() << "}\n";
      }
    } else {
      for (unsigned i = 0; i < args.size(); ++i)
        valueNames[args[i]] = "0";
      os() << getIndent() << "{\n";
      incIndent();
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      decIndent();
      os() << getIndent() << "}\n";
    }
  }

  void emitDmaCopy(DmaCopyOp op) override {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    if (!srcTy) {
      os() << getIndent() << "// dma.copy (unknown type)\n";
      return;
    }
    auto kind = op.getKind();
    if (kind == DMAKind::Pad) {
      emitCopyWithPad(op);
    } else if (kind == DMAKind::Transpose) {
      emitCopyWithTranspose(op);
    } else {
      int64_t bytes = getTensorBytes(srcTy);
      if (bytes > 0) {
        os() << getIndent() << "std::memcpy(" << getName(op.getDest())
             << ", " << getName(op.getSource()) << ", " << bytes << ");\n";
      }
    }
    valueNames[op.getToken()] = "0";
  }

  void emitCopyWithPad(DmaCopyOp op) {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstTy = dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcTy || !dstTy) return;
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();
    int rank = srcShape.size();
    std::string eTy = emitElementType(srcTy.getElementType());
    std::string srcName = getName(op.getSource());
    std::string dstName = getName(op.getDest());

    std::string padVal = "0";
    if (auto pv = op.getPadValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(*pv))
        padVal = std::to_string(intAttr.getInt());
      else if (auto fpAttr = dyn_cast<FloatAttr>(*pv)) {
        llvm::SmallString<16> s;
        fpAttr.getValue().toString(s, 6, 0);
        padVal = std::string(s);
      }
    }

    int64_t dstElems = 1;
    for (auto d : dstShape) dstElems *= d;
    os() << getIndent() << "for (int __i = 0; __i < " << dstElems
         << "; ++__i) ((" << eTy << "*)" << dstName << ")[__i] = (" << eTy
         << ")" << padVal << ";\n";

    llvm::SmallVector<int64_t> lowVals(rank, 0);
    if (auto pl = op.getPadLow()) {
      auto arr = pl.value();
      for (int i = 0; i < rank && i < (int)arr.size(); ++i)
        lowVals[i] = arr[i];
    }

    int64_t srcElems = 1;
    for (auto d : srcShape) srcElems *= d;

    os() << getIndent() << "for (int __i = 0; __i < " << srcElems
         << "; ++__i) {\n";
    incIndent();
    os() << getIndent() << "int __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os() << getIndent() << "int " << dn << " = __rem / " << stride
             << ";\n";
        os() << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os() << getIndent() << "int " << dn << " = __rem;\n";
      }
    }
    os() << getIndent() << "int __dst_idx = ";
    for (int d = 0; d < rank; ++d) {
      if (d > 0) os() << " + ";
      std::string coord =
          "(__d" + std::to_string(d) + " + " + std::to_string(lowVals[d]) + ")";
      int64_t stride = 1;
      for (int k = d + 1; k < rank; ++k) stride *= dstShape[k];
      if (stride != 1)
        os() << coord << " * " << stride;
      else
        os() << coord;
    }
    os() << ";\n";
    os() << getIndent() << "((" << eTy << "*)" << dstName
         << ")[__dst_idx] = ((" << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitCopyWithTranspose(DmaCopyOp op) {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    auto dstTy = dyn_cast<coir::TensorType>(op.getDest().getType());
    if (!srcTy || !dstTy) return;
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();
    int rank = srcShape.size();
    std::string eTy = emitElementType(srcTy.getElementType());
    std::string srcName = getName(op.getSource());
    std::string dstName = getName(op.getDest());

    llvm::SmallVector<int64_t> perm;
    if (auto tp = op.getTransposePerm()) {
      auto arr = tp.value();
      perm.assign(arr.begin(), arr.end());
    }
    if (perm.empty())
      for (int i = rank - 1; i >= 0; --i) perm.push_back(i);

    int64_t srcElems = 1;
    for (auto d : srcShape) srcElems *= d;

    os() << getIndent() << "for (int __i = 0; __i < " << srcElems
         << "; ++__i) {\n";
    incIndent();
    os() << getIndent() << "int __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os() << getIndent() << "int " << dn << " = __rem / " << stride
             << ";\n";
        os() << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os() << getIndent() << "int " << dn << " = __rem;\n";
      }
    }
    os() << getIndent() << "int __dst_idx = ";
    for (int d = 0; d < rank; ++d) {
      if (d > 0) os() << " + ";
      std::string coord = "__d" + std::to_string(perm[d]);
      int64_t stride = 1;
      for (int k = d + 1; k < rank; ++k) stride *= dstShape[k];
      if (stride != 1)
        os() << coord << " * " << stride;
      else
        os() << coord;
    }
    os() << ";\n";
    os() << getIndent() << "((" << eTy << "*)" << dstName
         << ")[__dst_idx] = ((" << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitElementCopy(ElementCopyOp op) {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    if (!srcTy) {
      os() << getIndent() << "// element.copy (unknown type)\n";
      return;
    }
    int64_t bytes = getTensorBytes(srcTy);
    if (bytes > 0) {
      os() << getIndent() << "std::memcpy(" << getName(op.getDest())
           << ", " << getName(op.getSource()) << ", " << bytes << ");\n";
    }
  }

  void emitBarrier(BarrierOp /*op*/) override {}

  void emitWait(WaitOp /*op*/) override {}

  void emitTensorTile(TensorTileOp op) override {
    std::string name = getName(op.getResult());
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    auto indices = op.getIndices();

    if (indices.empty()) {
      valueNames[op.getResult()] = getName(op.getSource());
      return;
    }

    auto srcShape = srcTy.getShape();
    os() << getIndent() << "auto " << name << " = " << getName(op.getSource());
    os() << " + (";
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) os() << " + ";
      os() << getName(indices[i]);
      int64_t stride = 1;
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      os() << " * " << stride;
    }
    os() << ");\n";
  }

  void emitTensorReduceElem(TensorReduceElemOp op) override {
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    os() << getIndent() << dst << "[";
    emitLinearIndex(op.getIndices(), tty);
    os() << "] += " << val << ";\n";
  }
};

struct EmitCCPass : public ::coir::impl::EmitCCBase<EmitCCPass> {
  using EmitCCBase::EmitCCBase;

  void runOnOperation() override {
    auto module = getOperation();
    CCEmitter emitter;
    emitter.emitModule(module, llvm::outs());
  }
};

static bool registered_cc = [] {
  CoIR::CodeGenRegistry::Register("cc", [] {
    return std::make_unique<CCEmitter>();
  });
  return true;
}();

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitCCPass() {
  return std::make_unique<EmitCCPass>();
}

void emitCC(mlir::ModuleOp module, llvm::raw_ostream &os) {
  CCEmitter emitter;
  emitter.emitModule(module, os);
}
} // namespace coir
