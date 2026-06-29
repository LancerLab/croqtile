//===- EmitHIP.cpp - Emit HIP/C++ source from lowered CoIR IR ------------===//
//
// Walks the module and emits HIP kernel source text. The emitter converts
// CoIR operations into their HIP/C++ equivalents for AMD GPUs.
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
#define GEN_PASS_DECL_EMITHIP
#define GEN_PASS_DEF_EMITHIP
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

class HIPEmitter : public coir::CoIREmitterBase {
public:
  HIPEmitter() = default;

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

    emitScriptPrologue(os, "compile and execute HIP kernel", "_hip");

    if (sctx.build_env.empty()) {
      os << "ROCM_HOME=\"${ROCM_HOME:-/opt/rocm}\"\n";
      os << "HIPCC=\"${ROCM_HOME}/bin/hipcc\"\n";
      os << "if [[ ! -x \"$HIPCC\" ]]; then\n";
      os << "  echo \"Error: hipcc not found at $HIPCC\"; exit 1\n";
      os << "fi\n";
      os << "if [[ -z \"${amdgpu_arch:-}\" ]]; then\n";
      os << "  amdgpu_arch=$(\"${ROCM_HOME}/bin/rocminfo\" 2>/dev/null | "
            "grep -oP 'gfx\\d+' | head -1 || echo \"gfx1030\")\n";
      os << "fi\n\n";
    }

    os << "TMPFILE=\"$TMPDIR/kernel.hip\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_HIP_SOURCE__'\n";
    EmitSource(module, "", os);

    os << "\n__COCC_HIP_SOURCE__\n\n";
    if (has_embedded) {
      os << "\"${HIPCC}\" -std=c++17 --offload-arch=\"${amdgpu_arch}\" "
            "-I\"$TMPDIR\" -o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    } else {
      os << "\"$HIPCC\" -std=c++17 --offload-arch=\"$amdgpu_arch\" "
            "-I\"$CHOREO_INC\" -I\"$TMPDIR\" -o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
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
      return "__hip_bfloat16";
    if (ty.isF16())
      return "__half";
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
    if (ty.isBF16()) return "__hip_bfloat16";
    if (ty.isF16()) return "__half";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isInteger(8)) return "uint8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int32_t";
    if (ty.isInteger(64)) return "int64_t";
    return "/* unknown */";
  }

  void emitHeader() {
    os() << "#define __CHOREO_TARGET_AMDGPU__ 1\n";
    os() << "#include \"choreo.h\"\n";
    os() << "#include <hip/hip_runtime.h>\n\n";
  }

  std::string kernelDeviceName(StringRef name) {
    return ("__" + name + "_kernel__").str();
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

  std::string getAllocQualifier(coir::TensorType tty) override {
    return tty.getMemorySpace() == 1 ? "__shared__ " : "";
  }

  void emitOpFallback(mlir::Operation *op) override {
    using namespace mlir;
    if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      os() << getIndent() << "// ERROR: TMA not supported on HIP target\n";
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else
      CoIREmitterBase::emitOpFallback(op);
  }

  void emitMMAFill(MMAFillOp /*op*/) override {
    os() << getIndent() << "// ERROR: MMA not supported on HIP target\n";
  }
  void emitMMALoad(MMALoadOp /*op*/) override {
    os() << getIndent() << "// ERROR: MMA not supported on HIP target\n";
  }
  void emitMMAExec(MMAExecOp /*op*/) override {
    os() << getIndent() << "// ERROR: MMA not supported on HIP target\n";
  }
  void emitMMAStore(MMAStoreOp /*op*/) override {
    os() << getIndent() << "// ERROR: MMA not supported on HIP target\n";
  }
  void emitDMAConstDesc(DMAConstDescOp /*op*/) override {}
  void emitDMAPrefetch(DMADescPrefetchOp /*op*/) override {}
  void emitDMARuntimeDesc(DMADescRuntimeOp /*op*/) override {}
  void emitDMAInvoke(DMAInvokeOp /*op*/) override {}

  struct LaunchDims {
    llvm::SmallVector<int64_t, 3> grid = {1};
    llvm::SmallVector<int64_t, 3> block = {1};

    std::string gridStr() const {
      if (grid.size() == 1) return std::to_string(grid[0]);
      std::string s = "dim3(";
      for (unsigned i = 0; i < grid.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(grid[i]);
      }
      return s + ")";
    }
    std::string blockStr() const {
      if (block.size() == 1) return std::to_string(block[0]);
      std::string s = "dim3(";
      for (unsigned i = 0; i < block.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(block[i]);
      }
      return s + ")";
    }
  };

  LaunchDims getLaunchDims(KernelOp kernel) {
    LaunchDims dims;
    kernel.getBody().walk([&](ParallelOp par) {
      auto lvl = par.getLevel();
      auto bounds = par.getBounds();
      llvm::SmallVector<int64_t, 3> bv(bounds.begin(), bounds.end());
      if (lvl == coir::ParallelLevel::BLOCK)
        dims.grid = bv;
      else if (lvl == coir::ParallelLevel::THREAD)
        dims.block = bv;
    });
    return dims;
  }

  void emitKernel(KernelOp kernel) {
    entryAssertions.clear();

    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    os() << "__global__ void " << devName << "(";

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
    std::string devName = kernelDeviceName(symName);
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();

    if (numResults == 0) {
      os() << "void " << symName << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << emitChoreoType(fnType.getInput(i), true) << " p" << i;
      }
      os() << ") {\n";

      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty) continue;
        std::string eType = emitElementType(tty.getElementType());
        int64_t bytes = getTensorBytes(tty);
        os() << "  " << eType << "* p" << i << "__device = nullptr;\n";
        os() << "  (void)hipMalloc(&p" << i << "__device, " << bytes
           << "ULL);\n";
        os() << "  (void)hipMemcpy(p" << i << "__device, p" << i << ".data(), "
           << bytes << "ULL, hipMemcpyHostToDevice);\n";
      }

      auto dims = getLaunchDims(kernel);
      emitEntryAssertions(kernel);

      os() << "  " << devName << "<<<" << dims.gridStr() << ", "
         << dims.blockStr() << ">>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "p" << i << "__device";
      }
      os() << ");\n";
      os() << "  (void)hipDeviceSynchronize();\n";

      for (unsigned i = 0; i < numInputs; ++i)
        os() << "  (void)hipFree(p" << i << "__device);\n";
      os() << "}\n\n";
      return;
    }

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    os() << emitChoreoType(fnType.getResult(0), false) << " " << symName << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os() << ", ";
      os() << emitChoreoType(fnType.getInput(i), true) << " p" << i;
    }
    os() << ") {\n";

    std::string eType = emitElementType(resTy.getElementType());

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty) continue;
      std::string inputEType = emitElementType(tty.getElementType());
      int64_t bytes = getTensorBytes(tty);
      os() << "  " << inputEType << "* p" << i << "__device = nullptr;\n";
      os() << "  (void)hipMalloc(&p" << i << "__device, " << bytes << "ULL);\n";
      os() << "  (void)hipMemcpy(p" << i << "__device, p" << i << ".data(), "
         << bytes << "ULL, hipMemcpyHostToDevice);\n";
    }

    int64_t resBytes = getTensorBytes(resTy);
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
    os() << "  " << eType << "* __result__device = nullptr;\n";
    os() << "  (void)hipMalloc(&__result__device, " << resBytes << "ULL);\n";

    auto dims = getLaunchDims(kernel);
    emitEntryAssertions(kernel);

    os() << "  " << devName << "<<<" << dims.gridStr() << ", "
       << dims.blockStr() << ">>>(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os() << ", ";
      os() << "p" << i << "__device";
    }
    os() << ", __result__device);\n";
    os() << "  (void)hipDeviceSynchronize();\n";
    os() << "  (void)hipMemcpy(__result.data(), __result__device, "
       << resBytes << "ULL, hipMemcpyDeviceToHost);\n";

    for (unsigned i = 0; i < numInputs; ++i)
      os() << "  (void)hipFree(p" << i << "__device);\n";
    os() << "  (void)hipFree(__result__device);\n";
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
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(remOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(remOp.getRhs(), kernel, hostNames);
      return (hostNames[v] = "(" + lhs + " % " + rhs + ")");
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

  void emitParallel(ParallelOp op) override {
    auto level = op.getLevel();
    auto bounds = op.getBounds();
    auto &body = op.getBody();
    auto args = body.getArguments();

    os() << getIndent() << "// parallel level="
       << stringifyParallelLevel(level) << " bounds=[";
    for (unsigned i = 0; i < bounds.size(); ++i) {
      if (i > 0) os() << ", ";
      os() << bounds[i];
    }
    os() << "]\n";

    if (level == ParallelLevel::BLOCK) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string dim = i == 0 ? "blockIdx.x" : "blockIdx.y";
        valueNames[args[i]] = dim;
      }
    } else if (level == ParallelLevel::THREAD) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string dim = i == 0 ? "threadIdx.x" : "threadIdx.y";
        valueNames[args[i]] = dim;
      }
    } else {
      for (unsigned i = 0; i < args.size(); ++i)
        valueNames[args[i]] = getName(args[i]);
    }

    os() << getIndent() << "{\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitCooperativeCopy(Value src, Value dst) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    if (!srcTy) {
      os() << getIndent() << "// copy (unknown tensor type)\n";
      return;
    }
    int64_t totalElems = 1;
    for (auto d : srcTy.getShape()) totalElems *= d;
    std::string srcName = getName(src);
    std::string dstName = getName(dst);
    std::string eTy = emitElementType(srcTy.getElementType());

    os() << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << totalElems << "; __i += blockDim.x) {\n";
    incIndent();
    os() << getIndent() << "((" << eTy << "*)" << dstName << ")[__i] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitCopyWithPad(Value src, Value dst,
                       std::optional<ArrayRef<int64_t>> padLow,
                       std::optional<ArrayRef<int64_t>> padHigh,
                       std::optional<Attribute> padValueAttr) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    auto dstTy = dyn_cast<coir::TensorType>(dst.getType());
    if (!srcTy || !dstTy) {
      os() << getIndent() << "// pad copy (unknown tensor type)\n";
      return;
    }
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();
    int rank = srcShape.size();
    std::string eTy = emitElementType(srcTy.getElementType());
    std::string srcName = getName(src);
    std::string dstName = getName(dst);

    std::string padVal = "0";
    if (padValueAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(*padValueAttr))
        padVal = std::to_string(intAttr.getInt());
      else if (auto fpAttr = dyn_cast<FloatAttr>(*padValueAttr)) {
        llvm::SmallString<16> s;
        fpAttr.getValue().toString(s, 6, 0);
        padVal = std::string(s);
      }
    }

    int64_t dstElems = 1;
    for (auto d : dstShape) dstElems *= d;
    os() << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << dstElems << "; __i += blockDim.x) {\n";
    incIndent();
    os() << getIndent() << "((" << eTy << "*)" << dstName << ")[__i] = ("
       << eTy << ")" << padVal << ";\n";
    decIndent();
    os() << getIndent() << "}\n";
    os() << getIndent() << "__syncthreads();\n";

    llvm::SmallVector<int64_t> lowVals(rank, 0);
    if (padLow) {
      auto pl = *padLow;
      for (int i = 0; i < rank && i < (int)pl.size(); ++i)
        lowVals[i] = pl[i];
    }

    int64_t srcElems = 1;
    for (auto d : srcShape) srcElems *= d;

    os() << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << srcElems << "; __i += blockDim.x) {\n";
    incIndent();
    os() << getIndent() << "size_t __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os() << getIndent() << "size_t " << dn << " = __rem / " << stride
           << ";\n";
        os() << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os() << getIndent() << "size_t " << dn << " = __rem;\n";
      }
    }
    os() << getIndent() << "size_t __dst_idx = ";
    for (int d = 0; d < rank; ++d) {
      if (d > 0) os() << " + ";
      std::string coord = "(__d" + std::to_string(d) + " + "
                         + std::to_string(lowVals[d]) + ")";
      int64_t stride = 1;
      for (int k = d + 1; k < rank; ++k) stride *= dstShape[k];
      if (stride != 1)
        os() << coord << " * " << stride;
      else
        os() << coord;
    }
    os() << ";\n";
    os() << getIndent() << "((" << eTy << "*)" << dstName << ")[__dst_idx] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitCopyWithTranspose(Value src, Value dst,
                             std::optional<ArrayRef<int64_t>> permAttr) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    auto dstTy = dyn_cast<coir::TensorType>(dst.getType());
    if (!srcTy || !dstTy) {
      os() << getIndent() << "// transpose copy (unknown tensor type)\n";
      return;
    }
    auto srcShape = srcTy.getShape();
    auto dstShape = dstTy.getShape();
    int rank = srcShape.size();
    std::string eTy = emitElementType(srcTy.getElementType());
    std::string srcName = getName(src);
    std::string dstName = getName(dst);

    llvm::SmallVector<int64_t> perm;
    if (permAttr) {
      auto pa = *permAttr;
      perm.assign(pa.begin(), pa.end());
    }
    if (perm.empty())
      for (int i = rank - 1; i >= 0; --i) perm.push_back(i);

    int64_t srcElems = 1;
    for (auto d : srcShape) srcElems *= d;

    os() << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << srcElems << "; __i += blockDim.x) {\n";
    incIndent();
    os() << getIndent() << "size_t __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os() << getIndent() << "size_t " << dn << " = __rem / " << stride
           << ";\n";
        os() << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os() << getIndent() << "size_t " << dn << " = __rem;\n";
      }
    }
    os() << getIndent() << "size_t __dst_idx = ";
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
    os() << getIndent() << "((" << eTy << "*)" << dstName << ")[__dst_idx] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  template <typename CopyOp>
  void emitCopyDispatch(CopyOp op) {
    auto kind = op.getKind();
    if (kind == DMAKind::Pad) {
      auto padLow = op.getPadLow();
      auto padHigh = op.getPadHigh();
      auto padValue = op.getPadValue();
      std::optional<ArrayRef<int64_t>> pl =
          padLow ? std::optional(padLow.value()) : std::nullopt;
      std::optional<ArrayRef<int64_t>> ph =
          padHigh ? std::optional(padHigh.value()) : std::nullopt;
      std::optional<Attribute> pv =
          padValue ? std::optional<Attribute>(*padValue) : std::nullopt;
      emitCopyWithPad(op.getSource(), op.getDest(), pl, ph, pv);
    } else if (kind == DMAKind::Transpose) {
      auto tp = op.getTransposePerm();
      std::optional<ArrayRef<int64_t>> perm =
          tp ? std::optional(tp.value()) : std::nullopt;
      emitCopyWithTranspose(op.getSource(), op.getDest(), perm);
    } else {
      emitCooperativeCopy(op.getSource(), op.getDest());
    }
  }

  void emitDmaCopy(DmaCopyOp op) override {
    emitCopyDispatch(op);
    os() << getIndent() << "__syncthreads();\n";
    valueNames[op.getToken()] = getName(op.getDest());
  }

  void emitElementCopy(ElementCopyOp op) {
    emitCooperativeCopy(op.getSource(), op.getDest());
  }

  void emitBarrier(BarrierOp op) override {
    auto scope = op.getScope();
    if (scope == ParallelLevel::BLOCK)
      os() << getIndent() << "__syncthreads();\n";
    else
      os() << getIndent() << "// barrier scope="
         << stringifyParallelLevel(scope) << "\n";
  }

  void emitWait(WaitOp /*op*/) override {
    os() << getIndent() << "__syncthreads();\n";
  }

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
    bool isAtomic = op->hasAttr("atomic");
    if (isAtomic) {
      os() << getIndent() << "atomicAdd(&" << dst << "[";
      emitLinearIndex(op.getIndices(), tty);
      os() << "], " << val << ");\n";
    } else {
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), tty);
      os() << "] += " << val << ";\n";
    }
  }

};

struct EmitHIPPass : public ::coir::impl::EmitHIPBase<EmitHIPPass> {
  using EmitHIPBase::EmitHIPBase;

  void runOnOperation() override {
    auto module = getOperation();
    HIPEmitter emitter;
    emitter.emitModule(module, llvm::outs());
  }
};

static bool registered_hip = [] {
  CoIR::CodeGenRegistry::Register("hip", [] {
    return std::make_unique<HIPEmitter>();
  });
  return true;
}();

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitHIPPass() {
  return std::make_unique<EmitHIPPass>();
}

void emitHIP(mlir::ModuleOp module, llvm::raw_ostream &os) {
  HIPEmitter emitter;
  emitter.emitModule(module, os);
}
} // namespace coir

