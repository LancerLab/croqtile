//===- EmitCUDA.cpp - Emit CUDA/C++ source from lowered CoIR IR ----------===//
//
// Walks the module and emits CUDA kernel source text. The emitter converts
// CoIR operations into their CUDA/C++ equivalents.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CoIR/Passes.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace coir {
#define GEN_PASS_DECL_EMITCUDA
#define GEN_PASS_DEF_EMITCUDA
#include "CoIR/Passes.h.inc"
} // namespace coir

using namespace mlir;
using namespace coir;

namespace {

class CUDAEmitter {
public:
  CUDAEmitter(llvm::raw_ostream &os) : os(os), indent(0) {}

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
  DenseSet<Value> mmaAccumulators;
  DenseMap<Value, std::string> mmaFragRoles;
  DenseMap<Value, std::string> mmaFragLayouts;
  DenseMap<Value, int64_t> tileStrides;
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

  std::string emitCUDAType(Type ty) {
    if (auto tensorTy = dyn_cast<coir::TensorType>(ty)) {
      std::string result;
      llvm::raw_string_ostream ss(result);
      ss << emitElementType(tensorTy.getElementType()) << "*";
      return result;
    }
    if (auto fragTy = dyn_cast<coir::MMAFragType>(ty)) {
      return "/* mma_frag -- see declaration */";
    }
    if (isa<coir::AsyncTokenType>(ty))
      return "cuda::barrier::arrival_token";
    if (ty.isIndex())
      return "int";
    if (ty.isF16())
      return "half";
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

  std::string emitElementType(Type ty) {
    if (ty.isF16()) return "half";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isInteger(8)) return "uint8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int32_t";
    if (ty.isInteger(64)) return "int64_t";
    return "/* unknown */";
  }

  std::string emitWMMAFragType(coir::MMAFragType fragTy,
                               StringRef role = "accumulator",
                               StringRef layout = "") {
    std::string result;
    llvm::raw_string_ostream ss(result);
    auto shape = fragTy.getShape();
    int64_t M = shape.size() > 0 ? shape[0] : 16;
    int64_t N = shape.size() > 1 ? shape[1] : 16;
    int64_t K = 16;
    ss << "wmma::fragment<wmma::" << role << ", "
       << M << ", " << N << ", " << K << ", "
       << emitElementType(fragTy.getElementType());
    if (!layout.empty())
      ss << ", wmma::" << layout;
    ss << ">";
    return result;
  }

  void prescanMMAFragRoles(mlir::Operation *root) {
    root->walk([&](MMAFillOp fill) {
      mmaFragRoles[fill.getResult()] = "accumulator";
    });
    root->walk([&](MMAExecOp exec) {
      auto layout = exec.getLayout();
      std::string lhsLayout, rhsLayout;
      if (layout == MMALayout::RowCol) {
        lhsLayout = "row_major"; rhsLayout = "col_major";
      } else if (layout == MMALayout::RowRow) {
        lhsLayout = "row_major"; rhsLayout = "row_major";
      } else if (layout == MMALayout::ColCol) {
        lhsLayout = "col_major"; rhsLayout = "col_major";
      } else {
        lhsLayout = "col_major"; rhsLayout = "row_major";
      }
      mmaFragRoles[exec.getLhs()] = "matrix_a";
      mmaFragLayouts[exec.getLhs()] = lhsLayout;
      mmaFragRoles[exec.getRhs()] = "matrix_b";
      mmaFragLayouts[exec.getRhs()] = rhsLayout;
      mmaFragRoles[exec.getAccumulator()] = "accumulator";
      mmaFragRoles[exec.getResult()] = "accumulator";
    });
  }

  void emitHeader() {
    os << "#include <cuda_fp16.h>\n";
    os << "#include <mma.h>\n";
    os << "#include \"choreo.h\"\n";
    os << "using namespace nvcuda;\n\n";
  }

  std::string kernelDeviceName(StringRef name) {
    return ("__" + name + "_kernel__").str();
  }

  void emitKernel(KernelOp kernel) {
    prescanMMAFragRoles(kernel);

    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    os << "__global__ void " << devName << "(";

    auto &body = kernel.getBody();
    unsigned paramIdx = 0;
    if (!body.empty()) {
      auto args = body.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (paramIdx > 0)
          os << ", ";
        std::string name = "arg" + std::to_string(paramIdx);
        valueNames[args[i]] = name;
        os << emitCUDAType(fnType.getInput(i)) << " " << name;
        paramIdx++;
      }
    }
    for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
      if (paramIdx > 0)
        os << ", ";
      std::string name = "out" + std::to_string(i);
      os << emitCUDAType(fnType.getResult(i)) << " " << name;
      returnParamNames[i] = name;
      paramIdx++;
    }
    os << ") {\n";
    incIndent();

    // Pre-scan: identify return values to bind allocs to output params
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

  std::string emitChoreoType(Type ty, bool asView = true) {
    if (auto tty = dyn_cast<coir::TensorType>(ty)) {
      std::string eTy = emitElementType(tty.getElementType());
      std::string choreoElem;
      auto eTyML = tty.getElementType();
      if (eTyML.isInteger(8)) choreoElem = "choreo::u8";
      else if (eTyML.isInteger(16)) choreoElem = "choreo::s16";
      else if (eTyML.isInteger(32)) choreoElem = "choreo::s32";
      else if (eTyML.isInteger(64)) choreoElem = "choreo::s64";
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

  void emitHostWrapper(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();

    if (numResults == 0) {
      os << "void " << symName << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << emitChoreoType(fnType.getInput(i), true) << " p" << i;
      }
      os << ") {\n";

      std::string eType = "int8_t";
      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty) continue;
        eType = emitElementType(tty.getElementType());
        int64_t bytes = getTensorBytes(tty);
        os << "  " << eType << "* p" << i << "__device = nullptr;\n";
        os << "  cudaMalloc(&p" << i << "__device, " << bytes << "ULL);\n";
        os << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), "
           << bytes << "ULL, cudaMemcpyHostToDevice);\n";
      }

      auto dims = getLaunchDims(kernel);

      os << "  " << devName << "<<<" << dims.gridStr() << ", "
         << dims.blockStr() << ">>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << "p" << i << "__device";
      }
      os << ");\n";
      os << "  cudaDeviceSynchronize();\n";

      for (unsigned i = 0; i < numInputs; ++i)
        os << "  cudaFree(p" << i << "__device);\n";
      os << "}\n\n";
      return;
    }

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    os << emitChoreoType(fnType.getResult(0), false) << " " << symName << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      auto &body = kernel.getBody();
      std::string pName = "arg" + std::to_string(i);
      if (!body.empty() && i < body.getArguments().size()) {
        pName = "p" + std::to_string(i);
      }
      os << emitChoreoType(fnType.getInput(i), true) << " " << pName;
    }
    os << ") {\n";

    std::string eType = emitElementType(resTy.getElementType());

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty) continue;
      std::string inputEType = emitElementType(tty.getElementType());
      int64_t bytes = getTensorBytes(tty);
      os << "  " << inputEType << "* p" << i << "__device = nullptr;\n";
      os << "  cudaMalloc(&p" << i << "__device, " << bytes << "ULL);\n";
      os << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), "
         << bytes << "ULL, cudaMemcpyHostToDevice);\n";
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
    else if (resElemTy.isF32()) choreoElem = "choreo::f32";
    else if (resElemTy.isF16()) choreoElem = "choreo::f16";
    else if (resElemTy.isF64()) choreoElem = "choreo::f64";
    else choreoElem = "choreo::s32";

    os << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
       << resTy.getShape().size() << ">(" << shapeStr << ");\n";
    os << "  " << eType << "* __result__device = nullptr;\n";
    os << "  cudaMalloc(&__result__device, " << resBytes << "ULL);\n";

    auto dims = getLaunchDims(kernel);

    os << "  " << devName << "<<<" << dims.gridStr() << ", "
       << dims.blockStr() << ">>>(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      os << "p" << i << "__device";
    }
    os << ", __result__device);\n";
    os << "  cudaDeviceSynchronize();\n";
    os << "  cudaMemcpy(__result.data(), __result__device, "
       << resBytes << "ULL, cudaMemcpyDeviceToHost);\n";

    for (unsigned i = 0; i < numInputs; ++i)
      os << "  cudaFree(p" << i << "__device);\n";
    os << "  cudaFree(__result__device);\n";
    os << "  return __result;\n";
    os << "}\n\n";
  }

  void emitOp(Operation *op) {
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
    else if (auto dataCopy = dyn_cast<DataCopyOp>(op))
      emitDataCopy(dataCopy);
    else if (auto dmaCopy = dyn_cast<DmaCopyOp>(op))
      emitDmaCopy(dmaCopy);
    else if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      emitTmaCopy(tmaCopy);
    else if (auto threadCopy = dyn_cast<ThreadCopyOp>(op))
      emitThreadCopy(threadCopy);
    else if (auto barrier = dyn_cast<BarrierOp>(op))
      emitBarrier(barrier);
    else if (auto wait = dyn_cast<WaitOp>(op))
      emitWait(wait);
    else if (auto reduceElem = dyn_cast<TensorReduceElemOp>(op))
      emitTensorReduceElem(reduceElem);
    else if (auto alloc = dyn_cast<TensorAllocOp>(op))
      emitTensorAlloc(alloc);
    else if (auto tile = dyn_cast<TensorTileOp>(op))
      emitTensorTile(tile);
    else if (auto ret = dyn_cast<KernelReturnOp>(op))
      (void)ret;
    else if (auto yield = dyn_cast<YieldOp>(op))
      emitYield(yield);
    else if (auto constOp = dyn_cast<arith::ConstantOp>(op))
      emitConstant(constOp);
    else if (auto indexCast = dyn_cast<arith::IndexCastOp>(op)) {
      valueNames[indexCast.getResult()] = getName(indexCast.getIn());
    }
    else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op))
      emitIfOp(ifOp);
    else if (isa<mlir::scf::YieldOp>(op))
      (void)op;
    else if (emitArithBinOp(op)) {}
    else if (emitCmpOp(op)) {}
    else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      std::string name = getName(selectOp.getResult());
      os << getIndent() << emitCUDAType(selectOp.getResult().getType()) << " "
         << name << " = " << getName(selectOp.getCondition()) << " ? "
         << getName(selectOp.getTrueValue()) << " : "
         << getName(selectOp.getFalseValue()) << ";\n";
    }
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
       << stringifyParallelLevel(level) << " bounds=[";
    for (unsigned i = 0; i < bounds.size(); ++i) {
      if (i > 0) os << ", ";
      os << bounds[i];
    }
    os << "]\n";

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

    os << getIndent() << "{\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}\n";
  }

  void emitForeach(ForeachOp op) {
    auto &body = op.getBody();
    auto args = body.front().getArguments();
    std::string iv = getName(args[0]);
    std::string ub = getName(op.getUpperBound());

    auto iterArgs = op.getIterArgs();
    for (unsigned i = 0; i < iterArgs.size(); ++i) {
      std::string iterName = getName(args[i + 1]);
      os << getIndent() << "auto " << iterName << " = "
         << getName(iterArgs[i]) << ";\n";
    }

    os << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
       << ub << "; ++" << iv << ") {\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}\n";

    for (unsigned i = 0; i < op.getResults().size(); ++i)
      valueNames[op.getResult(i)] = getName(args[i + 1]);
  }

  void emitMMAFill(MMAFillOp op) {
    auto fragTy = cast<coir::MMAFragType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    mmaAccumulators.insert(op.getResult());
    os << getIndent() << emitWMMAFragType(fragTy, "accumulator")
       << " " << name << ";\n";
    os << getIndent() << "wmma::fill_fragment(" << name << ", "
       << getName(op.getValue()) << ");\n";
  }

  int64_t getSourceLeadingDim(Value v) {
    if (auto tileOp = v.getDefiningOp<TensorTileOp>()) {
      auto srcTy = dyn_cast<coir::TensorType>(tileOp.getSource().getType());
      if (srcTy && srcTy.getShape().size() >= 2)
        return srcTy.getShape().back();
    }
    if (auto tty = dyn_cast<coir::TensorType>(v.getType()))
      if (tty.getShape().size() >= 2)
        return tty.getShape().back();
    return 16;
  }

  void emitMMALoad(MMALoadOp op) {
    auto fragTy = cast<coir::MMAFragType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    auto roleIt = mmaFragRoles.find(op.getResult());
    std::string role = roleIt != mmaFragRoles.end() ? roleIt->second : "matrix_a";
    auto layoutIt = mmaFragLayouts.find(op.getResult());
    std::string layout = layoutIt != mmaFragLayouts.end() ? layoutIt->second : "";
    os << getIndent() << emitWMMAFragType(fragTy, role, layout)
       << " " << name << ";\n";
    int64_t ldm = getSourceLeadingDim(op.getSource());
    os << getIndent() << "wmma::load_matrix_sync(" << name << ", "
       << getName(op.getSource()) << ", " << ldm << ");\n";
  }

  void emitMMAExec(MMAExecOp op) {
    std::string acc = getName(op.getAccumulator());
    valueNames[op.getResult()] = acc;
    os << getIndent() << "wmma::mma_sync(" << acc << ", "
       << getName(op.getLhs()) << ", " << getName(op.getRhs()) << ", "
       << acc << ");\n";
  }

  void emitMMAStore(MMAStoreOp op) {
    int64_t ldm = getSourceLeadingDim(op.getDest());
    os << getIndent() << "wmma::store_matrix_sync("
       << getName(op.getDest()) << ", " << getName(op.getFragment())
       << ", " << ldm << ", wmma::mem_row_major);\n";
  }

  void emitDataCopy(DataCopyOp op) {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    int64_t totalElems = 1;
    if (srcTy)
      for (auto d : srcTy.getShape()) totalElems *= d;

    os << getIndent() << "for (int i = threadIdx.x; i < " << totalElems
       << "; i += blockDim.x)\n";
    incIndent();
    os << getIndent() << getName(op.getDest()) << "[i] = "
       << getName(op.getSource()) << "[i];\n";
    decIndent();
    os << getIndent() << "__syncthreads();\n";

    if (op.getToken())
      valueNames[op.getToken()] = getName(op.getDest());
  }

  void emitDmaCopy(DmaCopyOp op) {
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    int64_t totalElems = 1;
    if (srcTy)
      for (auto d : srcTy.getShape()) totalElems *= d;

    int64_t totalBytes = totalElems * 4;
    if (auto bytes = op->getAttrOfType<IntegerAttr>("transfer_bytes"))
      totalBytes = bytes.getInt();
    else if (srcTy) {
      unsigned bits = srcTy.getElementType().getIntOrFloatBitWidth();
      totalBytes = totalElems * bits / 8;
    }

    os << getIndent() << "for (int i = threadIdx.x; i < " << totalElems
       << "; i += blockDim.x)\n";
    incIndent();
    os << getIndent() << getName(op.getDest()) << "[i] = "
       << getName(op.getSource()) << "[i];\n";
    decIndent();
    os << getIndent() << "__syncthreads();\n";

    valueNames[op.getToken()] = getName(op.getDest());
  }

  void emitTmaCopy(TmaCopyOp op) {
    os << getIndent() << "// TMA copy\n";
    os << getIndent() << "cute::copy(tma_desc, " << getName(op.getSource())
       << ", " << getName(op.getDest()) << ");\n";
    valueNames[op.getToken()] = "/* tma_token */";
  }

  void emitThreadCopy(ThreadCopyOp op) {
    int64_t totalElems = 0;
    if (auto n = op->getAttrOfType<IntegerAttr>("total_elements"))
      totalElems = n.getInt();

    auto srcStrideIt = tileStrides.find(op.getSource());
    auto dstStrideIt = tileStrides.find(op.getDest());
    int64_t srcStride = srcStrideIt != tileStrides.end() ? srcStrideIt->second : 1;
    int64_t dstStride = dstStrideIt != tileStrides.end() ? dstStrideIt->second : 1;

    if (totalElems <= 1) {
      os << getIndent() << getName(op.getDest()) << "[0] = "
         << getName(op.getSource()) << "[0];\n";
    } else if (srcStride == 1 && dstStride == 1) {
      os << getIndent() << "for (int i = threadIdx.x; i < " << totalElems
         << "; i += blockDim.x)\n";
      incIndent();
      os << getIndent() << getName(op.getDest()) << "[i] = "
         << getName(op.getSource()) << "[i];\n";
      decIndent();
    } else {
      os << getIndent() << "for (int i = 0; i < " << totalElems << "; ++i)\n";
      incIndent();
      std::string srcIdx = srcStride != 1 ? "i * " + std::to_string(srcStride) : "i";
      std::string dstIdx = dstStride != 1 ? "i * " + std::to_string(dstStride) : "i";
      os << getIndent() << getName(op.getDest()) << "[" << dstIdx << "] = "
         << getName(op.getSource()) << "[" << srcIdx << "];\n";
      decIndent();
    }
  }

  void emitBarrier(BarrierOp op) {
    auto scope = op.getScope();
    if (scope == ParallelLevel::BLOCK)
      os << getIndent() << "__syncthreads();\n";
    else
      os << getIndent() << "// barrier scope="
         << stringifyParallelLevel(scope) << "\n";
  }

  void emitWait(WaitOp op) {
    os << getIndent() << "__syncthreads();\n";
  }

  void emitTensorAlloc(TensorAllocOp op) {
    if (returnValues.count(op.getResult()))
      return; // bound to output parameter

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    int32_t ms = tensorTy.getMemorySpace();
    std::string qualifier = ms == 1 ? "__shared__ " : "";
    std::string name = getName(op.getResult());

    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape())
      totalElems *= d;

    os << getIndent() << qualifier << emitElementType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";
  }

  void emitTensorTile(TensorTileOp op) {
    std::string name = getName(op.getResult());
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    auto tileTy = dyn_cast<coir::TensorType>(op.getResult().getType());
    auto indices = op.getIndices();

    if (indices.empty()) {
      valueNames[op.getResult()] = getName(op.getSource());
      return;
    }

    auto srcShape = srcTy.getShape();
    auto tileShape = tileTy ? tileTy.getShape() : llvm::ArrayRef<int64_t>{};

    llvm::SmallVector<int64_t> srcStrides(srcShape.size());
    {
      int64_t s = 1;
      for (int i = (int)srcShape.size() - 1; i >= 0; --i) {
        srcStrides[i] = s;
        s *= srcShape[i];
      }
    }

    bool hasWildcard = false;
    int64_t wildcardStride = 1;

    llvm::SmallVector<int64_t> perIdxTileSize(indices.size(), 1);
    if (indices.size() == srcShape.size() &&
        tileShape.size() == srcShape.size()) {
      for (unsigned i = 0; i < indices.size(); ++i)
        perIdxTileSize[i] = tileShape[i];
    } else {
      for (unsigned i = 0; i < indices.size() && i < srcShape.size(); ++i) {
        bool isConst0 = false;
        if (auto constOp = indices[i].getDefiningOp<arith::ConstantOp>())
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
            if (intAttr.getInt() == 0)
              isConst0 = true;

        if (isConst0 && srcShape[i] > 1) {
          perIdxTileSize[i] = srcShape[i];
          hasWildcard = true;
          wildcardStride = srcStrides[i];
        } else {
          perIdxTileSize[i] = 1;
        }
      }
    }

    if (hasWildcard)
      tileStrides[op.getResult()] = wildcardStride;

    os << getIndent() << "auto " << name << " = " << getName(op.getSource());
    os << " + (";
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) os << " + ";
      os << getName(indices[i]);
      int64_t stride = perIdxTileSize[i];
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      os << " * " << stride;
    }
    os << ");\n";
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

  void emitTensorLoadElem(TensorLoadElemOp op) {
    std::string name = getName(op.getResult());
    std::string src = getName(op.getSource());
    auto tty = cast<coir::TensorType>(op.getSource().getType());
    os << getIndent() << emitCUDAType(op.getResult().getType()) << " " << name
       << " = " << src << "[";
    emitLinearIndex(op.getIndices(), tty);
    os << "];\n";
  }

  void emitTensorReduceElem(TensorReduceElemOp op) {
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    bool isAtomic = op->hasAttr("atomic");
    if (isAtomic) {
      os << getIndent() << "atomicAdd(&" << dst << "[";
      emitLinearIndex(op.getIndices(), tty);
      os << "], " << val << ");\n";
    } else {
      os << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), tty);
      os << "] += " << val << ";\n";
    }
  }

  void emitTensorStoreElem(TensorStoreElemOp op) {
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    os << getIndent() << dst << "[";
    emitLinearIndex(op.getIndices(), tty);
    os << "] = " << val << ";\n";
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
    else if (isa<arith::RemSIOp>(op))
      opStr = "%";
    else
      return false;

    std::string name = getName(op->getResult(0));
    std::string lhs = getName(op->getOperand(0));
    std::string rhs = getName(op->getOperand(1));
    os << getIndent() << emitCUDAType(op->getResult(0).getType()) << " "
       << name << " = " << lhs << " " << opStr << " " << rhs << ";\n";
    return true;
  }

  void emitIfOp(mlir::scf::IfOp op) {
    os << getIndent() << "if (" << getName(op.getCondition()) << ") {\n";
    incIndent();
    for (auto &bodyOp : op.getThenRegion().front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}\n";
    if (!op.getElseRegion().empty()) {
      os << getIndent() << "else {\n";
      incIndent();
      for (auto &bodyOp : op.getElseRegion().front().getOperations())
        emitOp(&bodyOp);
      decIndent();
      os << getIndent() << "}\n";
    }
  }

  bool emitCmpOp(Operation *op) {
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
      os << getIndent() << "bool " << name << " = (" << lhs << " " << opStr
         << " " << rhs << ");\n";
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
      os << getIndent() << "bool " << name << " = (" << lhs << " " << opStr
         << " " << rhs << ");\n";
      return true;
    }
    return false;
  }

  void emitYield(YieldOp op) {
    auto *parentOp = op->getParentOp();
    if (auto foreachOp = dyn_cast<ForeachOp>(parentOp)) {
      auto args = foreachOp.getBody().front().getArguments();
      for (unsigned i = 0; i < op.getOperands().size(); ++i) {
        auto iterArgName = getName(args[i + 1]);
        auto yieldValName = getName(op.getOperands()[i]);
        if (iterArgName != yieldValName)
          os << getIndent() << iterArgName << " = " << yieldValName << ";\n";
      }
    }
  }

  void emitConstant(arith::ConstantOp op) {
    std::string name = getName(op.getResult());
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      os << getIndent() << "const int " << name << " = "
         << intAttr.getInt() << ";\n";
    } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      os << getIndent() << "const " << emitElementType(op.getType())
         << " " << name << " = ";
      llvm::SmallString<16> strVal;
      floatAttr.getValue().toString(strVal, 6, 0);
      os << strVal << ";\n";
    } else {
      os << getIndent() << "auto " << name << " = /* constant */;\n";
    }
  }
};

struct EmitCUDAPass : public ::coir::impl::EmitCUDABase<EmitCUDAPass> {
  using EmitCUDABase::EmitCUDABase;

  void runOnOperation() override {
    auto module = getOperation();
    CUDAEmitter emitter(llvm::outs());
    emitter.emitModule(module);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitCUDAPass() {
  return std::make_unique<EmitCUDAPass>();
}

void emitCUDA(mlir::ModuleOp module, llvm::raw_ostream &os) {
  CUDAEmitter emitter(os);
  emitter.emitModule(module);
}
} // namespace coir

namespace {

void emitHostCode(mlir::ModuleOp module, llvm::raw_ostream &os) {
  auto attr = module->getAttrOfType<mlir::StringAttr>("coir.host_code");
  if (attr) os << "\n" << attr.getValue() << "\n";
}

void emitScriptHeader(llvm::raw_ostream &os) {
  auto &sctx = CoIR::ScriptContext::Get();
  bool has_embedded = sctx.types_header && sctx.runtime_header;

  os << "#!/usr/bin/env bash\n";
  os << "# CoIR generated script -- compile and execute kernel\n";
  os << "set -eo pipefail\n\n";

  os << "TMPDIR=$(mktemp -d /tmp/cocc_XXXXXX)\n";
  os << "trap 'rm -rf $TMPDIR' EXIT\n\n";

  if (has_embedded) {
    os << "cat > \"$TMPDIR/choreo_types.h\" << '__COCC_TYPES_HEADER__'\n";
    os << sctx.types_header;
    os << "\n__COCC_TYPES_HEADER__\n\n";

    os << "cat > \"$TMPDIR/choreo.h\" << '__COCC_CHOREO_HEADER__'\n";
    os << sctx.runtime_header;
    os << "\n__COCC_CHOREO_HEADER__\n\n";
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

  if (!sctx.build_env.empty()) {
    os << sctx.build_env;
  } else {
    os << "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n";
    os << "NVCC=\"${CUDA_HOME}/bin/nvcc\"\n";
    os << "if [[ ! -x \"$NVCC\" ]]; then\n";
    os << "  echo \"Error: nvcc not found at $NVCC\"; exit 1\n";
    os << "fi\n";
    os << "if [[ -z \"${gpu_arch:-}\" ]]; then\n";
    os << "  _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader "
          "2>/dev/null | head -1 | tr -d '.' || true)\n";
    os << "  gpu_arch=\"sm_${_cc:-86}\"\n";
    os << "fi\n\n";
  }

  if (!sctx.arch_override.empty())
    os << "gpu_arch=\"" << sctx.arch_override << "\"\n\n";
}

void emitScriptFooter(llvm::raw_ostream &os) {
  auto &sctx = CoIR::ScriptContext::Get();
  bool has_embedded = sctx.types_header && sctx.runtime_header;

  os << "\n__COCC_CUDA_SOURCE__\n\n";

  if (has_embedded) {
    os << "\"${NVCC}\" -std=c++17 -arch=\"${gpu_arch}\" -I\"$TMPDIR\" "
          "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
  } else {
    os << "\"$NVCC\" -std=c++17 -arch=\"$gpu_arch\" -I\"$CHOREO_INC\" "
          "-I\"$TMPDIR\" -o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
  }

  os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
  os << "  shift\n";
  os << "  \"$BINFILE\" \"$@\"\n";
  os << "fi\n";
}

class CUDATargetEmitter : public CoIR::Emitter {
public:
  void EmitScript(mlir::ModuleOp module, llvm::raw_ostream &os) override {
    emitScriptHeader(os);
    os << "TMPFILE=\"$TMPDIR/kernel.cu\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_CUDA_SOURCE__'\n";
    coir::emitCUDA(module, os);
    emitHostCode(module, os);
    emitScriptFooter(os);
  }

  void EmitSource(mlir::ModuleOp module, llvm::raw_ostream &os) override {
    coir::emitCUDA(module, os);
    emitHostCode(module, os);
  }
};

static bool registered_gpu = [] {
  CoIR::EmitterRegistry::Register("cute", [] {
    return std::make_unique<CUDATargetEmitter>();
  });
  return true;
}();

} // namespace
