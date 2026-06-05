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
  }

private:
  llvm::raw_ostream &os;
  unsigned indent;
  DenseMap<Value, std::string> valueNames;
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
      std::string result;
      llvm::raw_string_ostream ss(result);
      ss << "wmma::fragment<wmma::accumulator, "
         << fragTy.getShape()[0] << ", " << fragTy.getShape()[1]
         << ", 16, " << emitElementType(fragTy.getElementType()) << ">";
      return result;
    }
    if (isa<coir::AsyncTokenType>(ty))
      return "cuda::barrier::arrival_token";
    if (ty.isIndex())
      return "int";
    if (ty.isF16())
      return "half";
    if (ty.isF32())
      return "float";
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
    if (ty.isInteger(8)) return "int8_t";
    if (ty.isInteger(32)) return "int32_t";
    return "/* unknown */";
  }

  void emitHeader() {
    os << "#include <cuda_fp16.h>\n";
    os << "#include <mma.h>\n";
    os << "using namespace nvcuda;\n\n";
  }

  void emitKernel(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    os << "__global__ void " << kernel.getSymName() << "(";

    auto &body = kernel.getBody();
    if (!body.empty()) {
      auto args = body.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (i > 0)
          os << ", ";
        std::string name = "arg" + std::to_string(i);
        valueNames[args[i]] = name;
        os << emitCUDAType(fnType.getInput(i)) << " " << name;
      }
    }
    os << ") {\n";
    incIndent();

    for (auto &op : body.front().getOperations())
      emitOp(&op);

    decIndent();
    os << "}\n\n";
  }

  void emitOp(Operation *op) {
    if (auto parallel = dyn_cast<ParallelOp>(op))
      emitParallel(parallel);
    else if (auto foreach_ = dyn_cast<ForeachOp>(op))
      emitForeach(foreach_);
    else if (auto fill = dyn_cast<MMAFillOp>(op))
      emitMMAFill(fill);
    else if (auto load = dyn_cast<MMALoadOp>(op))
      emitMMALoad(load);
    else if (auto exec = dyn_cast<MMAExecOp>(op))
      emitMMAExec(exec);
    else if (auto store = dyn_cast<MMAStoreOp>(op))
      emitMMAStore(store);
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

    if (level == ParallelLevel::Block) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string dim = i == 0 ? "blockIdx.x" : "blockIdx.y";
        valueNames[args[i]] = dim;
      }
    } else if (level == ParallelLevel::Thread) {
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
    os << getIndent() << emitCUDAType(fragTy) << " " << name << ";\n";
    os << getIndent() << "wmma::fill_fragment(" << name << ", "
       << getName(op.getValue()) << ");\n";
  }

  void emitMMALoad(MMALoadOp op) {
    auto fragTy = cast<coir::MMAFragType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    os << getIndent() << emitCUDAType(fragTy) << " " << name << ";\n";
    os << getIndent() << "wmma::load_matrix_sync(" << name << ", "
       << getName(op.getSource()) << ", " << fragTy.getShape()[1] << ");\n";
  }

  void emitMMAExec(MMAExecOp op) {
    std::string res = getName(op.getResult());
    std::string target = "wmma::mma_sync";
    if (auto tAttr = op->getAttrOfType<StringAttr>("target")) {
      if (tAttr.getValue() == "wgmma")
        target = "wgmma::mma_async";
    }
    auto resTy = cast<coir::MMAFragType>(op.getResult().getType());
    os << getIndent() << emitCUDAType(resTy) << " " << res << ";\n";
    os << getIndent() << target << "(" << res << ", "
       << getName(op.getLhs()) << ", " << getName(op.getRhs()) << ", "
       << getName(op.getAccumulator()) << ");\n";
  }

  void emitMMAStore(MMAStoreOp op) {
    auto fragTy = cast<coir::MMAFragType>(op.getFragment().getType());
    os << getIndent() << "wmma::store_matrix_sync("
       << getName(op.getDest()) << ", " << getName(op.getFragment())
       << ", " << fragTy.getShape()[1] << ", wmma::mem_row_major);\n";
  }

  void emitDmaCopy(DmaCopyOp op) {
    os << getIndent() << "// DMA copy: cp.async\n";
    os << getIndent() << "cuda::memcpy_async(" << getName(op.getDest())
       << ", " << getName(op.getSource()) << ", "
       << "/* transfer_bytes */";
    if (auto bytes = op->getAttrOfType<IntegerAttr>("transfer_bytes"))
      os << " " << bytes.getInt();
    os << ");\n";
    valueNames[op.getToken()] = "/* dma_token */";
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
    os << getIndent() << "// thread cooperative copy (" << totalElems
       << " elements)\n";
    os << getIndent() << "for (int i = threadIdx.x; i < " << totalElems
       << "; i += blockDim.x)\n";
    incIndent();
    os << getIndent() << getName(op.getDest()) << "[i] = "
       << getName(op.getSource()) << "[i];\n";
    decIndent();
  }

  void emitBarrier(BarrierOp op) {
    auto scope = op.getScope();
    if (scope == ParallelLevel::Block)
      os << getIndent() << "__syncthreads();\n";
    else
      os << getIndent() << "// barrier scope="
         << stringifyParallelLevel(scope) << "\n";
  }

  void emitWait(WaitOp op) {
    os << getIndent() << "// wait for async op\n";
  }

  void emitTensorAlloc(TensorAllocOp op) {
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
    os << getIndent() << "auto " << name << " = "
       << getName(op.getSource()) << " + /* tile offset */;\n";
  }

  void emitYield(YieldOp op) {
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      os << getIndent() << "// yield -> loop carried\n";
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
} // namespace coir
