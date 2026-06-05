// EmitTopscc -- emit topscc C++ from CoIR IR for GCU target.

#include "EmitTopscc.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace coir;

namespace {

class TopsccEmitter {
public:
  TopsccEmitter(llvm::raw_ostream &os) : os(os), indent(0) {}

  void emitModule(ModuleOp module) {
    emitHeader();
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitKernel(kernel);
    }
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitHostWrapper(kernel);
    }
  }

private:
  llvm::raw_ostream &os;
  unsigned indent;
  DenseMap<Value, std::string> valueNames;
  DenseMap<unsigned, std::string> returnParamNames;
  DenseSet<Value> returnValues;
  unsigned nextId = 0;

  std::string getIndent() { return std::string(indent * 2, ' '); }
  void incIndent() { indent++; }
  void decIndent() { if (indent > 0) indent--; }

  std::string getName(Value v) {
    auto it = valueNames.find(v);
    if (it != valueNames.end())
      return it->second;
    std::string name = "v" + std::to_string(nextId++);
    valueNames[v] = name;
    return name;
  }

  std::string emitType(Type ty) {
    if (auto tensorTy = dyn_cast<coir::TensorType>(ty))
      return emitElemType(tensorTy.getElementType()) + "*";
    if (ty.isIndex()) return "int";
    if (ty.isF16()) return "half";
    if (ty.isF32()) return "float";
    if (ty.isInteger(32)) return "int";
    return "/* unknown */";
  }

  std::string emitElemType(Type ty) {
    if (ty.isF16()) return "half";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isInteger(32)) return "int";
    return "/* unknown */";
  }

  void emitHeader() {
    os << "#include <krt/misc.h>\n";
    os << "#include <tcle.h>\n";
    os << "#include <stdint.h>\n";
    os << "#include <tops/topscc_types.h>\n";
    os << "#include <tops.h>\n";
    os << "#include \"tops/tops_runtime.h\"\n";
    os << "#include \"choreo.h\"\n\n";
  }

  void emitKernel(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    os << "__device__ void " << kernel.getSymName() << "(";

    auto &body = kernel.getBody();
    unsigned paramIdx = 0;
    if (!body.empty()) {
      auto args = body.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (paramIdx > 0) os << ", ";
        std::string name = "arg" + std::to_string(paramIdx);
        valueNames[args[i]] = name;
        os << emitType(fnType.getInput(i)) << " " << name;
        paramIdx++;
      }
    }
    for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
      if (paramIdx > 0) os << ", ";
      std::string name = "out" + std::to_string(i);
      os << emitType(fnType.getResult(i)) << " " << name;
      returnParamNames[i] = name;
      paramIdx++;
    }
    os << ") {\n";
    incIndent();

    for (auto &op : body.front().getOperations()) {
      if (auto ret = dyn_cast<KernelReturnOp>(op)) {
        for (unsigned i = 0; i < ret.getOperands().size(); ++i) {
          returnValues.insert(ret.getOperands()[i]);
          valueNames[ret.getOperands()[i]] = returnParamNames[i];
        }
      }
    }

    for (auto &op : body.front().getOperations())
      emitOp(&op);

    decIndent();
    os << "}\n\n";
  }

  int64_t getTensorBytes(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    Type eTy = tty.getElementType();
    int64_t elemSize = 4;
    if (eTy.isF16() || eTy.isInteger(16)) elemSize = 2;
    else if (eTy.isF64() || eTy.isInteger(64)) elemSize = 8;
    else if (eTy.isInteger(8)) elemSize = 1;
    return n * elemSize;
  }

  int64_t getTensorNumElems(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    return n;
  }

  void emitHostWrapper(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();
    if (numResults == 0) return;

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    std::string eType = emitElemType(resTy.getElementType());
    int64_t resN = getTensorNumElems(resTy);
    int64_t resBytes = getTensorBytes(resTy);

    // Emit __global__ wrapper that handles DMA staging
    os << "__global__ void __coir_global_" << name.str() << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      os << emitType(fnType.getInput(i)) << " g_in" << i;
    }
    os << ", " << eType << "* g_out, int N) {\n";
    // Local buffers
    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      int64_t n = tty ? getTensorNumElems(tty) : resN;
      os << "  __local__ " << eType << " l_in" << i << "[" << n << "];\n";
    }
    os << "  __local__ " << eType << " l_out[" << resN << "];\n";
    os << "  tops::private_dte ctx;\n  ctx.init();\n";
    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      int64_t n = tty ? getTensorNumElems(tty) : resN;
      os << "  tops::memcpy(ctx, tops::mdspan(l_in" << i << ", " << n
         << "), tops::mdspan(g_in" << i << ", " << n << "));\n";
    }
    os << "  " << name.str() << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      os << "l_in" << i;
    }
    os << ", l_out);\n";
    os << "  tops::memcpy(ctx, tops::mdspan(g_out, " << resN
       << "), tops::mdspan(l_out, " << resN << "));\n";
    os << "}\n\n";

    // Emit host-callable wrapper using choreo runtime types
    std::string choreoElem;
    if (resTy.getElementType().isInteger(32)) choreoElem = "choreo::s32";
    else if (resTy.getElementType().isF32()) choreoElem = "choreo::f32";
    else if (resTy.getElementType().isF16()) choreoElem = "choreo::f16";
    else choreoElem = "choreo::s32";

    unsigned ndim = resTy.getShape().size();
    os << "choreo::spanned_data<" << choreoElem << ", " << ndim << "> "
       << name.str() << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      auto inTy = dyn_cast<coir::TensorType>(fnType.getInput(i));
      std::string inChoreo = choreoElem;
      unsigned inDim = inTy ? inTy.getShape().size() : ndim;
      os << "const choreo::spanned_view<" << inChoreo << ", " << inDim
         << "> & p" << i;
    }
    os << ") {\n";

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      int64_t bytes = tty ? getTensorBytes(tty) : resBytes;
      os << "  " << eType << "* p" << i << "__device = nullptr;\n";
      os << "  topsMalloc((void**)&p" << i << "__device, " << bytes << "ULL);\n";
      os << "  topsMemcpy(p" << i << "__device, p" << i << ".data(), "
         << bytes << "ULL, topsMemcpyHostToDevice);\n";
    }

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
    os << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
       << ndim << ">(" << shapeStr << ");\n";
    os << "  " << eType << "* __result__device = nullptr;\n";
    os << "  topsMalloc((void**)&__result__device, " << resBytes << "ULL);\n";
    os << "  __coir_global_" << name.str() << "<<<1, 1>>>(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      os << "p" << i << "__device";
    }
    os << ", __result__device, " << resN << ");\n";
    os << "  topsDeviceSynchronize();\n";
    os << "  topsMemcpy(__result.data(), __result__device, "
       << resBytes << "ULL, topsMemcpyDeviceToHost);\n";
    for (unsigned i = 0; i < numInputs; ++i)
      os << "  topsFree(p" << i << "__device);\n";
    os << "  topsFree(__result__device);\n";
    os << "  return __result;\n";
    os << "}\n\n";
  }

  void emitOp(Operation *op) {
    if (auto parallel = dyn_cast<ParallelOp>(op))
      emitParallel(parallel);
    else if (auto foreach_ = dyn_cast<ForeachOp>(op))
      emitForeach(foreach_);
    else if (auto dataCopy = dyn_cast<DataCopyOp>(op))
      emitDataCopy(dataCopy);
    else if (auto loadElem = dyn_cast<TensorLoadElemOp>(op))
      emitLoadElem(loadElem);
    else if (auto storeElem = dyn_cast<TensorStoreElemOp>(op))
      emitStoreElem(storeElem);
    else if (auto alloc = dyn_cast<TensorAllocOp>(op))
      emitAlloc(alloc);
    else if (auto barrier = dyn_cast<BarrierOp>(op))
      emitBarrier(barrier);
    else if (auto wait = dyn_cast<WaitOp>(op))
      (void)wait;
    else if (auto ret = dyn_cast<KernelReturnOp>(op))
      (void)ret;
    else if (auto yield = dyn_cast<YieldOp>(op))
      (void)yield;
    else if (auto constOp = dyn_cast<arith::ConstantOp>(op))
      emitConstant(constOp);
    else if (emitArithBinOp(op)) {}
    else {
      os << getIndent() << "// [unhandled] " << op->getName().getStringRef()
         << "\n";
    }
  }

  void emitParallel(ParallelOp op) {
    auto level = op.getLevel();
    auto bounds = op.getBounds();
    auto &body = op.getBody();
    auto args = body.getArguments();

    os << getIndent() << "// parallel level="
       << stringifyParallelLevel(level) << "\n";

    if (level == ParallelLevel::BLOCK) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string name = "pid_" + std::to_string(nextId++);
        valueNames[args[i]] = name;
        os << getIndent() << "int " << name
           << " = tops::block_idx_x();  // bound=" << bounds[i] << "\n";
      }
      os << getIndent() << "{\n";
      incIndent();
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      decIndent();
      os << getIndent() << "}\n";
    } else {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string name = "pid_" + std::to_string(nextId++);
        valueNames[args[i]] = name;
        os << getIndent() << "for (int " << name << " = 0; " << name
           << " < " << bounds[i] << "; ++" << name << ") {\n";
        incIndent();
      }
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      for (unsigned i = 0; i < args.size(); ++i) {
        decIndent();
        os << getIndent() << "}\n";
      }
    }
  }

  void emitForeach(ForeachOp op) {
    auto &body = op.getBody();
    auto args = body.front().getArguments();
    std::string iv = getName(args[0]);
    std::string ub = getName(op.getUpperBound());

    os << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
       << ub << "; ++" << iv << ") {\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}\n";
  }

  void emitDataCopy(DataCopyOp op) {
    std::string src = getName(op.getSource());
    std::string dst = getName(op.getDest());
    auto srcTy = cast<coir::TensorType>(op.getSource().getType());
    int64_t totalElems = 1;
    for (auto d : srcTy.getShape()) totalElems *= d;
    int64_t elemBytes = 4;
    if (srcTy.getElementType().isF16()) elemBytes = 2;
    else if (srcTy.getElementType().isF64()) elemBytes = 8;

    os << getIndent() << "tcle::dma_copy(" << dst << ", " << src
       << ", " << totalElems * elemBytes << ");  // "
       << totalElems << " elements\n";

    if (op.getToken())
      valueNames[op.getToken()] = "/* dma_token */";
  }

  void emitLinearIndex(mlir::ValueRange indices, coir::TensorType tty) {
    auto strides = tty.getStrides();
    auto shape = tty.getShape();
    if (indices.empty()) {
      os << "0";
      return;
    }
    if (indices.size() == 1 && strides.empty()) {
      os << getName(indices[0]);
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
      if (i > 0) os << " + ";
      if (i < effectiveStrides.size() && effectiveStrides[i] != 1)
        os << getName(indices[i]) << " * " << effectiveStrides[i];
      else
        os << getName(indices[i]);
    }
  }

  void emitLoadElem(TensorLoadElemOp op) {
    std::string name = getName(op.getResult());
    std::string src = getName(op.getSource());
    auto tty = cast<coir::TensorType>(op.getSource().getType());
    os << getIndent() << emitType(op.getResult().getType()) << " " << name
       << " = " << src << "[";
    emitLinearIndex(op.getIndices(), tty);
    os << "];\n";
  }

  void emitStoreElem(TensorStoreElemOp op) {
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    os << getIndent() << dst << "[";
    emitLinearIndex(op.getIndices(), tty);
    os << "] = " << val << ";\n";
  }

  void emitAlloc(TensorAllocOp op) {
    if (returnValues.count(op.getResult())) return;

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape()) totalElems *= d;

    std::string qualifier;
    if (tensorTy.getMemorySpace() ==
        static_cast<int32_t>(coir::TensorMemorySpace::Local))
      qualifier = "__local__ ";

    os << getIndent() << qualifier << emitElemType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";
  }

  void emitBarrier(BarrierOp) {
    os << getIndent() << "tcle::sync();\n";
  }

  void emitConstant(arith::ConstantOp op) {
    std::string name = getName(op.getResult());
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
      os << getIndent() << "const int " << name << " = "
         << intAttr.getInt() << ";\n";
    else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      llvm::SmallString<16> strVal;
      floatAttr.getValue().toString(strVal, 6, 0);
      os << getIndent() << "const " << emitElemType(op.getType())
         << " " << name << " = " << strVal << ";\n";
    }
  }

  bool emitArithBinOp(Operation *op) {
    llvm::StringRef opStr;
    if (isa<arith::AddIOp>(op) || isa<arith::AddFOp>(op))
      opStr = "+";
    else if (isa<arith::SubIOp>(op) || isa<arith::SubFOp>(op))
      opStr = "-";
    else if (isa<arith::MulIOp>(op) || isa<arith::MulFOp>(op))
      opStr = "*";
    else if (isa<arith::DivSIOp>(op) || isa<arith::DivFOp>(op))
      opStr = "/";
    else
      return false;

    std::string name = getName(op->getResult(0));
    std::string lhs = getName(op->getOperand(0));
    std::string rhs = getName(op->getOperand(1));
    os << getIndent() << emitType(op->getResult(0).getType()) << " "
       << name << " = " << lhs << " " << opStr << " " << rhs << ";\n";
    return true;
  }
};

} // namespace

namespace coir {

void emitTopscc(mlir::ModuleOp module, llvm::raw_ostream &os) {
  TopsccEmitter emitter(os);
  emitter.emitModule(module);
}

static void emitScriptHeader(llvm::raw_ostream &os) {
  os << "#!/usr/bin/env bash\n";
  os << "# CoIR generated script -- compile and execute topscc kernel\n";
  os << "set -euo pipefail\n\n";
  os << "TOPSCC=\"${TOPSCC:-topscc}\"\n";
  os << "if ! command -v \"$TOPSCC\" &>/dev/null; then\n";
  os << "  echo \"Error: topscc not found\"; exit 1\n";
  os << "fi\n\n";
  os << "# Find choreo runtime header\n";
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
  os << "CHOREO_RT=\"${CHOREO_ROOT}/runtime\"\n\n";
  os << "TMPFILE=$(mktemp /tmp/coir_XXXXXX.cc)\n";
  os << "BINFILE=\"${TMPFILE%.cc}\"\n";
  os << "trap 'rm -f $TMPFILE $BINFILE' EXIT\n\n";
  os << "cat > \"$TMPFILE\" << '__COIR_TOPSCC_SOURCE__'\n";
}

static void emitScriptFooter(llvm::raw_ostream &os) {
  os << "__COIR_TOPSCC_SOURCE__\n\n";
  os << "# Compile and optionally execute\n";
  os << "\"$TOPSCC\" -std=c++17 -I\"$CHOREO_RT\" "
        "-o \"$BINFILE\" \"$TMPFILE\" -lpthread -ldl -lrt 2>&1\n";
  os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
  os << "  shift\n";
  os << "  \"$BINFILE\" \"$@\"\n";
  os << "fi\n";
}

static void emitHostCode(llvm::raw_ostream &os, ModuleOp module) {
  auto hostCodeAttr = module->getAttrOfType<StringAttr>("coir.host_code");
  if (!hostCodeAttr) return;
  os << "\n" << hostCodeAttr.getValue() << "\n";
}

void emitTopsccScript(mlir::ModuleOp module, llvm::raw_ostream &os) {
  emitScriptHeader(os);
  TopsccEmitter emitter(os);
  emitter.emitModule(module);
  emitHostCode(os, module);
  emitScriptFooter(os);
}

} // namespace coir
