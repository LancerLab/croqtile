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
#include "llvm/ADT/StringSwitch.h"
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
    hasMMAOps = false;
    hasLibCalls = false;
    hasPrintCalls = false;
    scanFeatures(module);
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
      os << "g++ -std=c++17 -O2 -pthread -fopenmp-simd "
            "-I\"$TMPDIR\" -o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    } else {
      os << "g++ -std=c++17 -O2 -pthread -fopenmp-simd "
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
    if (auto fty = dyn_cast<FloatType>(ty)) {
      if (fty.getWidth() == 8) return "uint8_t";
      if (fty.getWidth() == 19) return "float"; // TF32
    }
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
    if (auto fty = dyn_cast<FloatType>(ty)) {
      if (fty.getWidth() == 8) return "uint8_t";
      if (fty.getWidth() == 19) return "float"; // TF32
    }
    if (ty.isInteger(1)) return "bool";
    if (ty.isInteger(8)) return "uint8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int32_t";
    if (ty.isInteger(64)) return "int64_t";
    return CoIREmitterBase::emitElementType(ty);
  }

  bool hasMMAOps = false;
  bool hasLibCalls = false;
  bool hasPrintCalls = false;

  void scanFeatures(ModuleOp module) {
    module.walk([&](Operation *op) {
      if (isa<MMAFillOp, MMALoadOp, MMAExecOp, MMAStoreOp>(op))
        hasMMAOps = true;
      if (auto call = dyn_cast<CallOp>(op)) {
        auto callee = call.getCallee();
        if (callee.starts_with("__lib_") || callee.starts_with("__sqrt") ||
            callee.starts_with("__exp") || callee.starts_with("__log") ||
            callee.starts_with("__rsqrt") || callee.starts_with("__abs") ||
            callee.starts_with("__gelu") || callee.starts_with("__erf") ||
            callee.starts_with("__sin") || callee.starts_with("__cos") ||
            callee.starts_with("__tanh"))
          hasLibCalls = true;
        if (callee == "print" || callee == "println")
          hasPrintCalls = true;
      }
    });
  }

  void emitHeader() {
    os() << "#define __CHOREO_TARGET_CPU__ 1\n";
    os() << "#include \"choreo.h\"\n";
    if (hasMMAOps)
      os() << "#include \"choreo_cc.h\"\n";
    os() << "#include <cstring>\n";
    os() << "#include <cstdlib>\n";
    if (hasLibCalls)
      os() << "#include <cmath>\n";
    if (hasPrintCalls)
      os() << "#include <iostream>\n";
    os() << "\n";
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

  void emitTensorAlloc(TensorAllocOp op) override {
    if (returnValues.count(op.getResult()))
      return;

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    std::string name = getName(op.getResult());

    if (op.getReuseOffsetAttr()) {
      if (lastSpmName.empty()) {
        if (auto spmSizeAttr =
                op->getAttrOfType<mlir::IntegerAttr>("spm_size")) {
          int64_t spmBytes = spmSizeAttr.getInt();
          lastSpmName = "__spm_" + std::to_string(nextId++);
          os() << getIndent() << "alignas(64) unsigned char "
               << lastSpmName << "[" << spmBytes << "];\n";
        }
      }
      int64_t offset = op.getReuseOffset().value_or(0);
      std::string eType = emitElementType(tensorTy.getElementType());
      os() << getIndent() << eType << "* " << name << " = (" << eType
           << "*)((unsigned char*)" << lastSpmName << " + " << offset
           << ");\n";
      return;
    }

    if (op->hasAttr("spm")) {
      int64_t totalBytes = 1;
      for (auto d : tensorTy.getShape()) totalBytes *= d;
      int64_t elemBits = tensorTy.getElementType().getIntOrFloatBitWidth();
      totalBytes *= (elemBits / 8);
      lastSpmName = name;
      os() << getIndent() << "alignas(64) unsigned char " << name << "["
           << totalBytes << "];\n";
      return;
    }

    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape()) totalElems *= d;
    os() << getIndent() << "alignas(64) "
         << emitElementType(tensorTy.getElementType()) << " " << name
         << "[" << totalElems << "];\n";
  }

  void emitForeach(ForeachOp op) override {
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

    if (iterArgs.empty())
      os() << getIndent() << "#pragma omp simd\n";
    os() << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
         << ub << "; ++" << iv << ") {\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os() << getIndent() << "}\n";

    if (auto yieldOp = dyn_cast<YieldOp>(body.front().getTerminator())) {
      auto results = yieldOp.getOperands();
      auto opResults = op.getResults();
      for (unsigned i = 0; i < results.size(); ++i)
        valueNames[opResults[i]] = getName(results[i]);
    }
  }

  void emitKernelReturn(KernelReturnOp op) override {
    auto operands = op.getOperands();
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto it = returnParamNames.find(i);
      if (it != returnParamNames.end()) {
        auto ty = operands[i].getType();
        if (auto tty = dyn_cast<coir::TensorType>(ty)) {
          int64_t bytes = getTensorBytes(tty);
          if (bytes > 0)
            os() << getIndent() << "std::memcpy(" << it->second << ", "
                 << getName(operands[i]) << ", " << bytes << ");\n";
        } else {
          os() << getIndent() << "*" << it->second << " = "
               << getName(operands[i]) << ";\n";
        }
      }
    }
  }

  // -- MMA: software reference via choreo_cc.h --
  void emitMMAFill(MMAFillOp op) override {
    auto resTy = cast<coir::MMAFragType>(op.getResult().getType());
    int64_t count = 1;
    for (auto d : resTy.getShape()) count *= d;
    std::string name = getName(op.getResult());
    std::string eType = emitElementType(resTy.getElementType());
    os() << getIndent() << "alignas(64) " << eType << " " << name << "["
         << count << "];\n";
    os() << getIndent() << "choreo::cc::mma_fill(" << name << ", ("
         << eType << ")" << getName(op.getValue()) << ", " << count
         << ");\n";
  }
  void emitMMALoad(MMALoadOp op) override {
    auto srcTy = cast<coir::TensorType>(op.getSource().getType());
    auto resTy = cast<coir::MMAFragType>(op.getResult().getType());
    int64_t count = 1;
    for (auto d : resTy.getShape()) count *= d;
    std::string eType = emitElementType(resTy.getElementType());
    std::string name = getName(op.getResult());
    os() << getIndent() << "alignas(64) " << eType << " " << name << "["
         << count << "];\n";
    int64_t bytes = getTensorBytes(srcTy);
    if (bytes > 0) {
      os() << getIndent() << "std::memcpy(" << name << ", "
           << getName(op.getSource()) << ", " << bytes << ");\n";
    }
  }
  void emitMMAExec(MMAExecOp op) override {
    auto accTy = cast<coir::MMAFragType>(op.getAccumulator().getType());
    auto lhsTy = cast<coir::MMAFragType>(op.getLhs().getType());
    auto rhsTy = cast<coir::MMAFragType>(op.getRhs().getType());
    auto accShape = accTy.getShape();
    auto lhsShape = lhsTy.getShape();

    int64_t M = accShape.size() >= 2 ? accShape[0] : 1;
    int64_t N = accShape.size() >= 2 ? accShape[1] : accShape[0];
    int64_t K = lhsShape.size() >= 2 ? lhsShape[1] : lhsShape[0];

    std::string accElem = emitElementType(accTy.getElementType());
    std::string lhsElem = emitElementType(lhsTy.getElementType());
    std::string rhsElem = emitElementType(rhsTy.getElementType());

    std::string accName = getName(op.getAccumulator());
    valueNames[op.getResult()] = accName;

    os() << getIndent() << "choreo::cc::mma_exec_row_col<" << accElem
         << ", " << lhsElem << ", " << rhsElem << ", " << accElem << ">("
         << accName << ", " << getName(op.getLhs()) << ", "
         << getName(op.getRhs()) << ", " << accName << ", "
         << M << ", " << N << ", " << K << ");\n";
  }
  void emitMMAStore(MMAStoreOp op) override {
    auto fragTy = cast<coir::MMAFragType>(op.getFragment().getType());
    int64_t elems = 1;
    for (auto d : fragTy.getShape()) elems *= d;
    int64_t bytes = elems * (fragTy.getElementType().getIntOrFloatBitWidth() / 8);
    if (bytes > 0) {
      os() << getIndent() << "std::memcpy(" << getName(op.getDest())
           << ", " << getName(op.getFragment()) << ", " << bytes << ");\n";
    }
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
    auto operands = op.getOperands_();

    if (callee == "print" || callee == "println") {
      emitPrint(op, callee == "println");
      return;
    }

    if (isExpr && emitArithBIF(op))
      return;

    if (callee.substr(0, 6) == "__lib_" && emitLibCall(op))
      return;

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
    for (auto a : operands) {
      if (!first) os() << ", ";
      first = false;
      auto ty = a.getType();
      if (auto tty = mlir::dyn_cast<coir::TensorType>(ty))
        os() << "(" << emitElementType(tty.getElementType()) << "*)"
             << getName(a);
      else
        os() << getName(a);
    }
    os() << ");\n";
  }

  void emitPrint(CallOp op, bool newline) {
    os() << getIndent() << "std::cout";
    for (auto a : op.getOperands_())
      os() << " << " << getName(a);
    if (newline)
      os() << " << \"\\n\"";
    os() << ";\n";
  }

  bool emitArithBIF(CallOp op) {
    auto callee = op.getCallee().str();
    auto operands = op.getOperands_();
    if (operands.empty() || !op.getResult()) return false;

    using llvm::StringSwitch;
    // Unary math BIFs
    std::string mapped = StringSwitch<std::string>(callee)
      .Case("__sqrt", "std::sqrt")
      .Case("__rsqrt", "")
      .Case("__exp", "std::exp")
      .Case("__expm1", "std::expm1")
      .Case("__log", "std::log")
      .Case("__log1p", "std::log1p")
      .Case("__log2", "std::log2")
      .Case("__abs", "std::abs")
      .Case("__fabs", "std::fabs")
      .Case("__ceil", "std::ceil")
      .Case("__floor", "std::floor")
      .Case("__round", "std::round")
      .Case("__sin", "std::sin")
      .Case("__cos", "std::cos")
      .Case("__tan", "std::tan")
      .Case("__asin", "std::asin")
      .Case("__acos", "std::acos")
      .Case("__atan", "std::atan")
      .Case("__sinh", "std::sinh")
      .Case("__cosh", "std::cosh")
      .Case("__tanh", "std::tanh")
      .Case("__erf", "std::erf")
      .Case("__erfc", "std::erfc")
      .Case("__cbrt", "std::cbrt")
      .Default("");

    std::string a0 = getName(operands[0]);
    std::string resName = getName(op.getResult());
    auto resTy = op.getResult().getType();

    if (!mapped.empty()) {
      os() << getIndent() << emitType(resTy) << " " << resName << " = "
           << mapped << "(" << a0 << ");\n";
      return true;
    }
    if (callee == "__rsqrt") {
      os() << getIndent() << emitType(resTy) << " " << resName
           << " = 1.0 / std::sqrt(" << a0 << ");\n";
      return true;
    }
    if (callee == "__gelu") {
      os() << getIndent() << emitType(resTy) << " " << resName << " = ("
           << a0 << " * 0.5 * (1.0 + std::erf(" << a0
           << " * 0.7071067811865476)));\n";
      return true;
    }
    if (callee == "__signbit") {
      os() << getIndent() << emitType(resTy) << " " << resName
           << " = std::signbit(" << a0 << ");\n";
      return true;
    }
    // Binary arith BIFs
    if (operands.size() >= 2) {
      std::string a1 = getName(operands[1]);
      std::string binOp = StringSwitch<std::string>(callee)
        .Case("__max", "std::max")
        .Case("__min", "std::min")
        .Case("__pow", "std::pow")
        .Case("__fmod", "std::fmod")
        .Case("__atan2", "std::atan2")
        .Default("");
      if (!binOp.empty()) {
        os() << getIndent() << emitType(resTy) << " " << resName << " = "
             << binOp << "(" << a0 << ", " << a1 << ");\n";
        return true;
      }
    }
    return false;
  }

  bool emitLibCall(CallOp op) {
    auto callee = op.getCallee().str();
    auto operands = op.getOperands_();
    auto arg = [&](unsigned i) -> std::string { return getName(operands[i]); };
    unsigned argc = operands.size();

    if (callee == "__lib_gemm" && argc >= 5) {
      bool has_bias = (argc >= 6);
      std::string out = arg(0), A = arg(1), B = arg(2);
      unsigned ki = has_bias ? 4 : 3, ni = has_bias ? 5 : 4;
      std::string K = arg(ki), N = arg(ni);
      std::string bias = has_bias ? arg(3) : "nullptr";
      auto outTy = dyn_cast<coir::TensorType>(operands[0].getType());
      std::string eType = outTy ? emitElementType(outTy.getElementType())
                                : "float";
      os() << getIndent() << "choreo::cc::lib_gemm<" << eType << ">("
           << out << ", " << A << ", " << B << ", " << bias << ", "
           << K << ", " << N << ");\n";
      return true;
    }

    // Unary __lib_* (dst, src, num)
    static const std::pair<llvm::StringRef, const char *> unary_map[] = {
      {"__lib_abs", "std::abs"}, {"__lib_sqrt", "std::sqrt"},
      {"__lib_exp", "std::exp"}, {"__lib_log", "std::log"},
      {"__lib_ceil", "std::ceil"}, {"__lib_floor", "std::floor"},
      {"__lib_round", "std::round"}, {"__lib_sin", "std::sin"},
      {"__lib_cos", "std::cos"}, {"__lib_tan", "std::tan"},
      {"__lib_asin", "std::asin"}, {"__lib_acos", "std::acos"},
      {"__lib_atan", "std::atan"}, {"__lib_sinh", "std::sinh"},
      {"__lib_cosh", "std::cosh"}, {"__lib_tanh", "std::tanh"},
      {"__lib_erf", "std::erf"}, {"__lib_erfc", "std::erfc"},
      {"__lib_cbrt", "std::cbrt"},
    };
    for (auto &[name, fn] : unary_map) {
      if (callee == name && argc >= 3) {
        os() << getIndent() << "for (int __i = 0; __i < " << arg(2)
             << "; ++__i) " << arg(0) << "[__i] = " << fn << "("
             << arg(1) << "[__i]);\n";
        return true;
      }
    }
    // Special unary forms
    if (argc >= 3) {
      std::string body;
      if (callee == "__lib_neg")
        body = arg(0) + "[__i] = -" + arg(1) + "[__i]";
      else if (callee == "__lib_rsqrt")
        body = arg(0) + "[__i] = 1.0 / std::sqrt(" + arg(1) + "[__i])";
      else if (callee == "__lib_reciprocal")
        body = arg(0) + "[__i] = 1.0 / " + arg(1) + "[__i]";
      else if (callee == "__lib_sign")
        body = arg(0) + "[__i] = (" + arg(1) + "[__i] > 0) - (" +
               arg(1) + "[__i] < 0)";
      else if (callee == "__lib_relu")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] > 0 ? " +
               arg(1) + "[__i] : 0";
      else if (callee == "__lib_gelu")
        body = arg(0) + "[__i] = " + arg(1) +
               "[__i] * 0.5 * (1.0 + std::erf(" + arg(1) +
               "[__i] * 0.7071067811865476))";
      else if (callee == "__lib_sigmoid")
        body = arg(0) + "[__i] = 1.0 / (1.0 + std::exp(-" +
               arg(1) + "[__i]))";
      else if (callee == "__lib_silu")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] / (1.0 + std::exp(-" +
               arg(1) + "[__i]))";
      if (!body.empty()) {
        os() << getIndent() << "for (int __i = 0; __i < " << arg(2)
             << "; ++__i) " << body << ";\n";
        return true;
      }
    }

    // Binary __lib_* (dst, lhs, rhs, num)
    if (argc >= 4) {
      std::string body;
      if (callee == "__lib_add")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] + " + arg(2) + "[__i]";
      else if (callee == "__lib_sub")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] - " + arg(2) + "[__i]";
      else if (callee == "__lib_mul")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] * " + arg(2) + "[__i]";
      else if (callee == "__lib_div")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] / " + arg(2) + "[__i]";
      else if (callee == "__lib_max")
        body = arg(0) + "[__i] = std::max(" + arg(1) + "[__i], " +
               arg(2) + "[__i])";
      else if (callee == "__lib_min")
        body = arg(0) + "[__i] = std::min(" + arg(1) + "[__i], " +
               arg(2) + "[__i])";
      else if (callee == "__lib_pow")
        body = arg(0) + "[__i] = std::pow(" + arg(1) + "[__i], " +
               arg(2) + "[__i])";
      else if (callee == "__lib_fmod")
        body = arg(0) + "[__i] = std::fmod(" + arg(1) + "[__i], " +
               arg(2) + "[__i])";
      else if (callee == "__lib_atan2")
        body = arg(0) + "[__i] = std::atan2(" + arg(1) + "[__i], " +
               arg(2) + "[__i])";
      else if (callee == "__lib_gt")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] > " + arg(2) + "[__i]";
      else if (callee == "__lib_ge")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] >= " + arg(2) + "[__i]";
      else if (callee == "__lib_lt")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] < " + arg(2) + "[__i]";
      else if (callee == "__lib_le")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] <= " + arg(2) + "[__i]";
      else if (callee == "__lib_eq")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] == " + arg(2) + "[__i]";
      else if (callee == "__lib_ne")
        body = arg(0) + "[__i] = " + arg(1) + "[__i] != " + arg(2) + "[__i]";
      if (!body.empty()) {
        os() << getIndent() << "for (int __i = 0; __i < " << arg(3)
             << "; ++__i) " << body << ";\n";
        return true;
      }
    }

    // Reduce: (dst, src, num, reduce_dim, num_reduce)
    if ((callee == "__lib_reduce_sum" || callee == "__lib_reduce_max" ||
         callee == "__lib_reduce_min" || callee == "__lib_reduce_mean") &&
        argc >= 5) {
      std::string dst = arg(0), src = arg(1), num = arg(2);
      std::string nreduce = arg(4);
      std::string init, accum;
      if (callee == "__lib_reduce_sum" || callee == "__lib_reduce_mean")
        init = "0",
        accum = "__acc += " + src + "[__outer * " + nreduce + " + __inner]";
      else if (callee == "__lib_reduce_max")
        init = src + "[__outer * " + nreduce + "]",
        accum = "__acc = std::max(__acc, " + src +
                "[__outer * " + nreduce + " + __inner])";
      else
        init = src + "[__outer * " + nreduce + "]",
        accum = "__acc = std::min(__acc, " + src +
                "[__outer * " + nreduce + " + __inner])";
      std::string result =
          "{ for (int __outer = 0; __outer < " + num +
          " / " + nreduce +
          "; ++__outer) { auto __acc = " + init +
          "; for (int __inner = 0; __inner < " + nreduce +
          "; ++__inner) { " + accum + "; } " + dst + "[__outer] = __acc";
      if (callee == "__lib_reduce_mean")
        result += " / " + nreduce;
      result += "; } }";
      os() << getIndent() << result << "\n";
      return true;
    }

    // Convert: (dst, src, num)
    if (callee == "__lib_convert" && argc >= 3) {
      os() << getIndent() << "for (int __i = 0; __i < " << arg(2)
           << "; ++__i) " << arg(0) << "[__i] = " << arg(1) << "[__i];\n";
      return true;
    }

    // Where: (dst, cond, x, y, num)
    if (callee == "__lib_where" && argc >= 5) {
      os() << getIndent() << "for (int __i = 0; __i < " << arg(4)
           << "; ++__i) " << arg(0) << "[__i] = " << arg(1) << "[__i] ? "
           << arg(2) << "[__i] : " << arg(3) << "[__i];\n";
      return true;
    }

    return false;
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
