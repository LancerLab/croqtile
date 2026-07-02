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
#include "CodeGen/CoIREmitterBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
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

class CUDAEmitter : public coir::CoIREmitterBase {
public:
  CUDAEmitter() = default;

  bool Lower(mlir::ModuleOp module) override {
    mlir::PassManager pm(module.getContext());
    pm.addPass(coir::createPlanDMACopyPass());
    return mlir::succeeded(pm.run(module));
  }

  void emitModule(ModuleOp module, llvm::raw_ostream &os) override {
    os_ = &os;
    resetState();
    hasTMA = false;
    hasDMA = false;
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("coir.has_tma"))
      hasTMA = attr.getValue();
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("coir.has_dma"))
      hasDMA = attr.getValue();

    emitHeader(hasTMA);
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

    emitScriptPrologue(os, "compile and execute kernel");

    if (sctx.build_env.empty()) {
      os << "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n";
      os << "NVCC=\"${CUDA_HOME}/bin/nvcc\"\n";
      os << "if [[ ! -x \"$NVCC\" ]]; then\n";
      os << "  echo \"Error: nvcc not found at $NVCC\"; exit 1\n";
      os << "fi\n";
      os << "if [[ -z \"${gpu_arch:-}\" ]]; then\n";
      os << "  _cc=$(nvidia-smi --query-gpu=compute_cap "
            "--format=csv,noheader "
            "2>/dev/null | head -1 | tr -d '.' || true)\n";
      os << "  gpu_arch=\"sm_${_cc:-86}\"\n";
      os << "fi\n\n";
    }
    if (!sctx.arch_override.empty())
      os << "gpu_arch=\"" << sctx.arch_override << "\"\n\n";

    os << "TMPFILE=\"$TMPDIR/kernel.cu\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_CUDA_SOURCE__'\n";
    EmitSource(module, "", os);

    os << "\n__COCC_CUDA_SOURCE__\n\n";
    if (has_embedded) {
      os << "\"${NVCC}\" -std=c++17 -arch=\"${gpu_arch}\" "
            "--expt-relaxed-constexpr "
            "-I\"$TMPDIR\" -I\"${CUTE_HOME}/include\" "
            "-L\"${CUDA_HOME}/lib64\" -lcuda "
            "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    } else {
      os << "\"$NVCC\" -std=c++17 -arch=\"$gpu_arch\" "
            "--expt-relaxed-constexpr "
            "-I\"$CHOREO_INC\" -I\"$TMPDIR\" -I\"${CUTE_HOME}/include\" "
            "-L\"${CUDA_HOME}/lib64\" -lcuda "
            "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
    }
    emitScriptExecuteBlock(os);
    return 0;
  }

private:
  bool hasTMA = false;
  bool hasDMA = false;
  DenseSet<Value> mmaAccumulators;
  DenseMap<Value, std::string> mmaFragRoles;
  DenseMap<Value, std::string> mmaFragLayouts;
  struct TileLayout {
    llvm::SmallVector<int64_t> shape;
    llvm::SmallVector<int64_t> strides;
  };
  DenseMap<Value, int64_t> tileStrides;
  DenseMap<Value, TileLayout> tileLayouts;

  struct EntryAssertion {
    AssertOp op;
  };
  llvm::SmallVector<EntryAssertion> entryAssertions;

  // DMA descriptor tracking: populated by prescanDescriptors from
  // DMAConstDescOp instances in the kernel body.
  struct DescInfo {
    unsigned index;
    coir::TensorType srcType;
    coir::TensorType dstType;
    coir::DMAKind kind;
    bool isTMA;   // global<->shared with hasTMA
    bool isLoad;  // global->shared direction
    Value constDescResult; // SSA value from DMAConstDescOp
    int64_t swizzleBytes = 0; // 0=none, 32, 64, 128
  };
  llvm::SmallVector<DescInfo> descInfos;
  DenseMap<Value, unsigned> descValueToIndex;
  // Runtime offsets bound per DMAInvokeOp desc operand (from DMADescRuntimeOp)
  DenseMap<Value, SmallVector<Value, 4>> descRuntimeOffsets;
  DenseSet<Value> dmaTokens; // tokens from DMAInvokeOp
  unsigned nextFutureId = 0;

  std::string emitElementType(Type ty) override {
    if (ty.isBF16()) return "__nv_bfloat16";
    return CoIREmitterBase::emitElementType(ty);
  }

  std::string emitType(Type ty) override {
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
    if (ty.isBF16())
      return "__nv_bfloat16";
    if (ty.isF32())
      return "float";
    if (ty.isF64())
      return "double";
    if (isa<mlir::Float8E4M3FNType>(ty))
      return "choreo::f8_e4m3";
    if (isa<mlir::Float8E5M2Type>(ty))
      return "choreo::f8_e5m2";
    if (isa<mlir::Float6E2M3FNType>(ty))
      return "choreo::f6_e2m3";
    if (isa<mlir::Float6E3M2FNType>(ty))
      return "choreo::f6_e3m2";
    if (isa<mlir::Float4E2M1FNType>(ty))
      return "choreo::f4_e2m1";
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

  std::string getAllocQualifier(coir::TensorType tty) override {
    return tty.getMemorySpace() == 1 ? "__shared__ " : "";
  }

  bool needsTMAAlignment(coir::TensorType tty) override {
    return tty.getMemorySpace() == 1 && hasTMA;
  }

  void emitOpFallback(mlir::Operation *op) override {
    if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      emitTmaCopy(tmaCopy);
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else if (auto callOp = dyn_cast<coir::CallOp>(op))
      emitCall(callOp);
    else
      CoIREmitterBase::emitOpFallback(op);
  }

  void emitCall(CallOp op) {
    auto callee = op.getCallee().str();
    bool isExpr = op.getIsExpr() && *op.getIsExpr() && op.getResult();
    bool isArith = op.getIsBif() && *op.getIsBif();

    if (callee.find("fragment_scalar_elementwise_") == 0) {
      std::string opSymbol;
      if (callee == "fragment_scalar_elementwise_add") opSymbol = "+";
      else if (callee == "fragment_scalar_elementwise_sub") opSymbol = "-";
      else if (callee == "fragment_scalar_elementwise_mul") opSymbol = "*";

      auto operands = op.getOperands_();
      auto fragVal = operands[0];
      auto scalarVal = operands[1];
      auto fragTy = mlir::cast<coir::MMAFragType>(fragVal.getType());
      std::string eTy = emitElementType(fragTy.getElementType());

      std::string resName = getName(op.getResult());
      valueNames[op.getResult()] = getName(fragVal);
      os() << getIndent()
           << "choreo::nv_cute::warp_cooperative::fragment_scalar_elementwise("
           << getName(fragVal) << ", " << getName(scalarVal) << ", []("
           << eTy << " a, " << eTy << " b) { return a " << opSymbol
           << " b; });\n";
      return;
    }

    std::string funcName = callee;
    if (isArith) {
      static const llvm::StringMap<std::string> intrinsicMap = {
          {"__fmaf", "fmaf"}, {"__frcp_rn", "__frcp_rn"}};
      auto it = intrinsicMap.find(callee);
      if (it != intrinsicMap.end()) {
        funcName = it->second;
      } else {
        llvm::StringRef ref(callee);
        if (ref.starts_with("__"))
          ref = ref.drop_front(2);
        std::string stripped = ref.str();
        if (stripped == "min" || stripped == "max")
          funcName = "(choreo::nv_cute::numerics::" + stripped + ")";
        else
          funcName = "choreo::nv_cute::numerics::" + stripped;
      }
    }

    os() << getIndent();
    if (isExpr) {
      auto resTy = op.getResult().getType();
      os() << emitType(resTy) << " " << getName(op.getResult()) << " = ";
    }
    os() << funcName;

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

  std::string emitWMMAFragType(coir::MMAFragType fragTy,
                               StringRef role = "accumulator",
                               StringRef layout = "",
                               Value fragVal = nullptr) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    int64_t M = 16, N = 16, K = 16;
    if (fragVal && mmaTileDims.count(fragVal)) {
      auto [tM, tN, tK] = mmaTileDims[fragVal];
      M = tM; N = tN; K = tK;
    } else {
      auto shape = fragTy.getShape();
      M = shape.size() > 0 ? shape[0] : 16;
      N = shape.size() > 1 ? shape[1] : 16;
    }
    std::string elemStr = emitElementType(fragTy.getElementType());
    if (fragTy.getElementType().isF32() && K == 8 && role != "accumulator")
      elemStr = "tf32";
    ss << "wmma::fragment<wmma::" << role << ", "
       << M << ", " << N << ", " << K << ", " << elemStr;
    if (!layout.empty())
      ss << ", wmma::" << layout;
    ss << ">";
    return result;
  }

  DenseMap<Value, std::tuple<int64_t, int64_t, int64_t>> mmaTileDims;
  DenseSet<Value> ctmmaValues;
  bool useWGMMA = false;
  DenseSet<Value> wgmmaOperandFrags; // register-resident A operands for RS

  void prescanMMAFragRoles(mlir::Operation *root) {
    // Detect WGMMA: any GROUPx4 parallel level signals SM90+ warp-group MMA
    useWGMMA = false;
    wgmmaOperandFrags.clear();
    root->walk([&](coir::ParallelOp par) {
      if (par.getLevel() == coir::ParallelLevel::GROUPx4)
        useWGMMA = true;
    });
    if (useWGMMA) {
      root->walk([&](MMAExecOp exec) {
        if (exec.getMmaAtomNameAttr()) return; // skip CTMMA execs
        auto lhsDef = exec.getLhs().getDefiningOp<MMAFillOp>();
        if (lhsDef) wgmmaOperandFrags.insert(lhsDef.getResult());
      });
    }
    root->walk([&](MMAFillOp fill) {
      mmaFragRoles[fill.getResult()] = "accumulator";
    });
    root->walk([&](MMAExecOp exec) {
      std::string lhsLayout = "row_major";
      std::string rhsLayout = "row_major";
      mmaFragRoles[exec.getLhs()] = "matrix_a";
      mmaFragLayouts[exec.getLhs()] = lhsLayout;
      mmaFragRoles[exec.getRhs()] = "matrix_b";
      mmaFragLayouts[exec.getRhs()] = rhsLayout;
      mmaFragRoles[exec.getAccumulator()] = "accumulator";
      mmaFragRoles[exec.getResult()] = "accumulator";
      auto accTy = dyn_cast<coir::MMAFragType>(exec.getAccumulator().getType());
      auto lhsTy = dyn_cast<coir::MMAFragType>(exec.getLhs().getType());
      if (accTy && lhsTy) {
        auto accShape = accTy.getShape();
        auto lhsShape = lhsTy.getShape();
        int64_t M = accShape.size() > 0 ? accShape[0] : 16;
        int64_t N = accShape.size() > 1 ? accShape[1] : 16;
        int64_t K = lhsShape.size() > 1 ? lhsShape[1] : 16;
        auto tile = std::make_tuple(M, N, K);
        mmaTileDims[exec.getLhs()] = tile;
        mmaTileDims[exec.getRhs()] = tile;
        mmaTileDims[exec.getAccumulator()] = tile;
        mmaTileDims[exec.getResult()] = tile;
      }
      // Track all values that feed into CTMMA execs.
      if (exec.getMmaAtomNameAttr()) {
        ctmmaValues.insert(exec.getLhs());
        ctmmaValues.insert(exec.getRhs());
        ctmmaValues.insert(exec.getAccumulator());
        ctmmaValues.insert(exec.getResult());
      }
    });
    // Propagate CTMMA markers through foreach iter_args: if the exec result
    // is yielded as a foreach iter_arg, the initial value (fill) is also CTMMA.
    root->walk([&](coir::ForeachOp foreach) {
      auto iterArgs = foreach.getIterArgs();
      auto &bodyRegion = foreach.getBody();
      auto *bodyBlock = &bodyRegion.front();
      if (bodyBlock->getNumArguments() == 0) return;
      auto yieldOp = dyn_cast<coir::YieldOp>(bodyBlock->getTerminator());
      if (!yieldOp) return;
      auto yieldVals = yieldOp.getOperands();
      for (unsigned i = 0; i < yieldVals.size() && i < iterArgs.size(); ++i) {
        if (ctmmaValues.count(yieldVals[i]) ||
            ctmmaValues.count(bodyBlock->getArgument(i + 1))) {
          ctmmaValues.insert(iterArgs[i]);
          ctmmaValues.insert(bodyBlock->getArgument(i + 1));
          ctmmaValues.insert(yieldVals[i]);
          ctmmaValues.insert(foreach.getResult(i));
        }
        Value candidates[] = {yieldVals[i], bodyBlock->getArgument(i + 1)};
        for (Value v : candidates) {
          if (mmaTileDims.count(v)) {
            auto tile = mmaTileDims[v];
            mmaTileDims[iterArgs[i]] = tile;
            mmaTileDims[bodyBlock->getArgument(i + 1)] = tile;
            mmaTileDims[yieldVals[i]] = tile;
            mmaTileDims[foreach.getResult(i)] = tile;
          }
        }
      }
    });
  }

  void emitHeader(bool withTMA) {
    os() << "#define __CHOREO_TARGET_CUTE__\n";
    os() << "#define __USE_CUDA_TYPE__\n";
    if (withTMA || useWGMMA) {
      os() << "#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90\n";
      os() << "#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__\n";
    }
    os() << "#include \"choreo.h\"\n";
    if (withTMA) {
      os() << "namespace cde = cuda::device::experimental;\n";
    }
    if (useWGMMA) {
      os() << "#include <cutlass/cutlass.h>\n";
      os() << "#include <cutlass/arch/barrier.h>\n";
      os() << "#include <cutlass/arch/reg_reconfig.h>\n";
    }
    os() << "using namespace nvcuda;\n";
    os() << "using namespace choreo;\n\n";
  }

  std::string kernelDeviceName(StringRef name) {
    return ("__" + name + "_kernel__").str();
  }

  void prescanDescriptors(KernelOp kernel) {
    descInfos.clear();
    descValueToIndex.clear();

    kernel.getBody().walk([&](DMAConstDescOp op) {
      auto srcType = dyn_cast<coir::TensorType>(op.getSource().getType());
      auto dstType = dyn_cast<coir::TensorType>(op.getDest().getType());
      if (!srcType || !dstType) return;

      int32_t srcMS = srcType.getMemorySpace();
      int32_t dstMS = dstType.getMemorySpace();

      bool isTMA = op.getTma();
      bool isLoad = (srcMS <= 0) && (dstMS == 1);

      int64_t swizBytes = 0;
      if (auto sb = op.getSwizzleBytes())
        swizBytes = *sb;

      unsigned idx = descInfos.size();
      descInfos.push_back({idx, srcType, dstType, op.getKind(),
                           isTMA, isLoad, op.getOut(), swizBytes});
      descValueToIndex[op.getOut()] = idx;
    });
  }

  void emitKernel(KernelOp kernel) {
    entryAssertions.clear();
    lastSpmName.clear();
    prescanMMAFragRoles(kernel);
    prescanDescriptors(kernel);

    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    os() << "__global__ ";
    if (auto lb = kernel.getLaunchBoundsAttr()) {
      if (lb.getMaxThreadsPerBlock() > 0) {
        os() << "__launch_bounds__(" << lb.getMaxThreadsPerBlock();
        if (lb.getMinBlocksPerMultiprocessor() > 0) {
          os() << ", " << lb.getMinBlocksPerMultiprocessor();
          if (lb.getMaxBlocksPerCluster() > 0)
            os() << ", " << lb.getMaxBlocksPerCluster();
        }
        os() << ") ";
      }
    }
    if (auto nr = kernel.getMaxNregAttr()) {
      if (nr.getValue() > 0)
        os() << "__maxnreg__(" << nr.getValue() << ") ";
    }
    os() << "void " << devName << "(";

    auto &body = kernel.getBody();
    unsigned paramIdx = 0;
    if (!body.empty()) {
      auto args = body.getArguments();
      for (unsigned i = 0; i < args.size(); ++i) {
        if (paramIdx > 0)
          os() << ", ";
        std::string name = "arg" + std::to_string(paramIdx);
        valueNames[args[i]] = name;
        os() << emitType(fnType.getInput(i)) << " " << name;
        paramIdx++;
      }
    }
    for (unsigned i = 0; i < fnType.getNumResults(); ++i) {
      if (paramIdx > 0)
        os() << ", ";
      std::string name = "out" + std::to_string(i);
      os() << emitType(fnType.getResult(i)) << " " << name;
      returnParamNames[i] = name;
      paramIdx++;
    }
    // TMA descriptors as kernel parameters
    unsigned tmaParamCount = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      if (paramIdx > 0)
        os() << ", ";
      os() << "const __grid_constant__ CUtensorMap __choreo_tma_"
         << tmaParamCount << "_tensor_map";
      paramIdx++;
      tmaParamCount++;
    }
    os() << ") {\n";
    incIndent();

    prescanReturnValues(kernel);

    // Emit TMA barriers and atoms at kernel body start (only for TMA loads)
    unsigned tmaIdx = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      if (descInfos[i].isLoad) {
        os() << getIndent() << "__shared__ __align__(8) uint64_t "
           << "choreo_copy_atom_t_" << tmaIdx << "_barrier;\n";
        os() << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
        incIndent();
        os() << getIndent() << "choreo::tma_mbarrier_init(&choreo_copy_atom_t_"
           << tmaIdx << "_barrier, 1);\n";
        decIndent();
        os() << getIndent() << "}\n";
        os() << getIndent() << "__syncthreads();\n";
        os() << getIndent() << "TMAAtom choreo_copy_atom_t_" << tmaIdx
           << "{&choreo_copy_atom_t_" << tmaIdx << "_barrier};\n\n";
      }
      tmaIdx++;
    }

    for (auto &op : body.front().getOperations())
      emitOp(&op);

    decIndent();
    os() << "}\n\n";
  }

  std::string emitChoreoType(Type ty, bool asView = true,
                             llvm::StringRef hostElemHint = "") {
    if (auto tty = dyn_cast<coir::TensorType>(ty)) {
      std::string choreoElem;
      if (!hostElemHint.empty()) {
        choreoElem = "choreo::" + hostElemHint.str();
      } else {
        auto eTyML = tty.getElementType();
        if (eTyML.isInteger(8)) choreoElem = "choreo::u8";
        else if (eTyML.isInteger(16)) choreoElem = "choreo::s16";
        else if (eTyML.isInteger(32)) choreoElem = "choreo::s32";
        else if (eTyML.isInteger(64)) choreoElem = "choreo::s64";
        else if (eTyML.isF32()) choreoElem = "choreo::f32";
        else if (eTyML.isF16()) choreoElem = "choreo::f16";
        else if (eTyML.isBF16()) choreoElem = "choreo::bf16";
        else if (eTyML.isF64()) choreoElem = "choreo::f64";
        else if (isa<mlir::Float8E4M3FNType>(eTyML))
          choreoElem = "choreo::f8_e4m3";
        else if (isa<mlir::Float8E5M2Type>(eTyML))
          choreoElem = "choreo::f8_e5m2";
        else if (isa<mlir::Float6E2M3FNType>(eTyML))
          choreoElem = "choreo::f6_e2m3";
        else if (isa<mlir::Float6E3M2FNType>(eTyML))
          choreoElem = "choreo::f6_e3m2";
        else if (isa<mlir::Float4E2M1FNType>(eTyML))
          choreoElem = "choreo::f4_e2m1";
        else choreoElem = "choreo::s32";
      }
      unsigned ndim = tty.getShape().size();
      if (asView)
        return "const choreo::spanned_view<" + choreoElem + ", " +
               std::to_string(ndim) + "> &";
      else
        return "choreo::spanned_data<" + choreoElem + ", " +
               std::to_string(ndim) + ">";
    }
    if (ty.isInteger(32)) return "int";
    if (ty.isInteger(64)) return "int64_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(8)) return "uint8_t";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isF16()) return "half";
    if (ty.isIndex()) return "int";
    return "/* unknown */";
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

  std::string getStreamName(KernelOp kernel) {
    std::string stream;
    kernel.getBody().walk([&](ParallelOp par) {
      if (par.getLevel() == coir::ParallelLevel::BLOCK && par.getStreamAttr())
        stream = par.getStreamAttr().getValue().str();
    });
    return stream;
  }

  bool isAsyncLaunch(KernelOp kernel) {
    bool async = false;
    kernel.getBody().walk([&](ParallelOp par) {
      if (par.getLevel() == coir::ParallelLevel::BLOCK &&
          par.getIsAsyncAttr() && par.getIsAsyncAttr().getValue())
        async = true;
    });
    return async;
  }

  LaunchDims getLaunchDims(KernelOp kernel) {
    LaunchDims dims;
    int64_t groupWarps = 0;
    bool hasThreadLevel = false;
    kernel.getBody().walk([&](ParallelOp par) {
      auto lvl = par.getLevel();
      auto bounds = par.getBounds();
      llvm::SmallVector<int64_t, 3> bv(bounds.begin(), bounds.end());
      if (lvl == coir::ParallelLevel::BLOCK)
        dims.grid = bv;
      else if (lvl == coir::ParallelLevel::THREAD) {
        dims.block = bv;
        hasThreadLevel = true;
      } else if (lvl == coir::ParallelLevel::GROUP ||
                 lvl == coir::ParallelLevel::GROUPx4) {
        int64_t nWarps = 1;
        for (auto b : bounds) nWarps *= b;
        if (lvl == coir::ParallelLevel::GROUPx4)
          nWarps *= 4;
        groupWarps = nWarps;
      }
    });
    if (groupWarps > 0 && hasThreadLevel) {
      int64_t threadsPerGroup = 1;
      for (auto b : dims.block) threadsPerGroup *= b;
      dims.block = {groupWarps * threadsPerGroup};
    } else if (groupWarps > 0) {
      dims.block = {groupWarps * 32};
    }
    return dims;
  }

  std::string emitTMADataType(Type elemTy) {
    if (elemTy.isF32()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
    if (elemTy.isF16()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
    if (elemTy.isF64()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
    if (elemTy.isInteger(8)) return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (elemTy.isInteger(16)) return "CU_TENSOR_MAP_DATA_TYPE_UINT16";
    if (elemTy.isInteger(32)) return "CU_TENSOR_MAP_DATA_TYPE_UINT32";
    if (isa<mlir::Float8E4M3FNType>(elemTy))
      return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (isa<mlir::Float8E5M2Type>(elemTy))
      return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (isa<mlir::Float6E2M3FNType>(elemTy) ||
        isa<mlir::Float6E3M2FNType>(elemTy) ||
        isa<mlir::Float4E2M1FNType>(elemTy))
      return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (elemTy.isBF16()) return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16";
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
  }

  void emitTMADescriptorSetup(unsigned descIdx, const std::string &dataPtr,
                              coir::TensorType globalType,
                              coir::TensorType tileType,
                              int64_t swizzleBytes = 0) {
    auto shape = globalType.getShape();
    auto elemTy = globalType.getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    unsigned rank = shape.size();

    // Tile shape for box dimensions (transfer size per TMA op)
    auto tileShape = tileType.getShape();

    std::string prefix = "__choreo_tma_" + std::to_string(descIdx);

    // Shape (innermost-first for cuTensorMapEncodeTiled)
    os() << "  uint64_t " << prefix << "_shape[] = {";
    for (int i = (int)rank - 1; i >= 0; --i) {
      if (i < (int)rank - 1) os() << ", ";
      os() << shape[i];
    }
    os() << "};\n";

    // Strides (byte strides, skip innermost -- it's implicit)
    os() << "  uint64_t " << prefix << "_strides[] = {";
    int64_t stride = elemBytes;
    for (int i = (int)rank - 1; i >= 1; --i) {
      if (i < (int)rank - 1) os() << ", ";
      stride *= shape[i];
      os() << stride;
    }
    os() << "};\n";

    // Box shape from the tile type (innermost-first)
    os() << "  uint32_t " << prefix << "_box_shape[] = {";
    for (int i = (int)rank - 1; i >= 0; --i) {
      if (i < (int)rank - 1) os() << ", ";
      int64_t tileDim = (i < (int)tileShape.size()) ? tileShape[i] : 1;
      os() << "(uint32_t)(" << tileDim << ")";
    }
    os() << "};\n";

    // Element strides (always 1)
    os() << "  uint32_t " << prefix << "_elem_strides[] = {";
    for (unsigned i = 0; i < rank; ++i) {
      if (i > 0) os() << ", ";
      os() << "1";
    }
    os() << "};\n";

    // Tensor map
    os() << "  alignas(64) CUtensorMap " << prefix << "_tensor_map{};\n";
    os() << "  CUresult " << prefix << "_res = cuTensorMapEncodeTiled(\n";
    os() << "          &" << prefix << "_tensor_map,\n";
    os() << "          CUtensorMapDataType::" << emitTMADataType(elemTy) << ",\n";
    os() << "          " << rank << ",\n";
    os() << "          " << dataPtr << ",\n";
    os() << "          " << prefix << "_shape,\n";
    os() << "          " << prefix << "_strides,\n";
    os() << "          " << prefix << "_box_shape,\n";
    os() << "          " << prefix << "_elem_strides,\n";
    os() << "          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,\n";
    const char *swizEnum = "CU_TENSOR_MAP_SWIZZLE_NONE";
    if (swizzleBytes == 32) swizEnum = "CU_TENSOR_MAP_SWIZZLE_32B";
    else if (swizzleBytes == 64) swizEnum = "CU_TENSOR_MAP_SWIZZLE_64B";
    else if (swizzleBytes == 128) swizEnum = "CU_TENSOR_MAP_SWIZZLE_128B";
    os() << "          CUtensorMapSwizzle::" << swizEnum << ",\n";
    os() << "          CUtensorMapL2promotion::"
       << "CU_TENSOR_MAP_L2_PROMOTION_L2_128B,\n";
    os() << "          CUtensorMapFloatOOBfill::"
       << "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n";
    os() << "  choreo::abend_true(" << prefix << "_res != CUDA_SUCCESS);\n";
  }

  std::string getTMAGlobalPtr(unsigned tmaIdx, KernelOp kernel) {
    unsigned tmaLoadIdx = 0;
    unsigned tmaStoreIdx = 0;
    unsigned tmaCount = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      if (tmaCount == tmaIdx) {
        if (descInfos[i].isLoad) {
          unsigned fnInputs = kernel.getFunctionType().getNumInputs();
          unsigned argIdx = tmaLoadIdx < fnInputs ? tmaLoadIdx : 0;
          return "p" + std::to_string(argIdx) + "__device";
        } else {
          return "__result__device";
        }
      }
      if (descInfos[i].isLoad)
        tmaLoadIdx++;
      else
        tmaStoreIdx++;
      tmaCount++;
    }
    return "/* unknown TMA ptr */";
  }

  unsigned countTMADescs() {
    unsigned count = 0;
    for (auto &d : descInfos)
      if (d.isTMA) count++;
    return count;
  }

  struct DimArgMeta {
    int64_t paramIdx;
    int64_t dimIdx;
    std::string name;
  };

  struct DimCheckMeta {
    std::string name;
    int64_t param0, dim0;
    int64_t param1, dim1;
  };

  llvm::SmallVector<DimArgMeta> getDimArgs(KernelOp kernel) {
    llvm::SmallVector<DimArgMeta> result;
    auto attr = kernel->getAttrOfType<ArrayAttr>("coir.dim_args");
    if (!attr) return result;
    for (auto a : attr) {
      auto dict = dyn_cast<DictionaryAttr>(a);
      if (!dict) continue;
      DimArgMeta m;
      m.paramIdx = dict.getAs<IntegerAttr>("param").getInt();
      m.dimIdx = dict.getAs<IntegerAttr>("dim").getInt();
      m.name = dict.getAs<StringAttr>("name").getValue().str();
      result.push_back(m);
    }
    return result;
  }

  llvm::SmallVector<DimCheckMeta> getDimChecks(KernelOp kernel) {
    llvm::SmallVector<DimCheckMeta> result;
    auto attr = kernel->getAttrOfType<ArrayAttr>("coir.dim_checks");
    if (!attr) return result;
    for (auto a : attr) {
      auto dict = dyn_cast<DictionaryAttr>(a);
      if (!dict) continue;
      DimCheckMeta m;
      m.name = dict.getAs<StringAttr>("name").getValue().str();
      m.param0 = dict.getAs<IntegerAttr>("param0").getInt();
      m.dim0 = dict.getAs<IntegerAttr>("dim0").getInt();
      m.param1 = dict.getAs<IntegerAttr>("param1").getInt();
      m.dim1 = dict.getAs<IntegerAttr>("dim1").getInt();
      result.push_back(m);
    }
    return result;
  }

  std::string getDynShmemExpr(KernelOp kernel,
                              llvm::ArrayRef<DimArgMeta> dimArgMeta) {
    std::string expr;
    kernel.getBody().walk([&](coir::TensorAllocOp alloc) {
      if (!expr.empty()) return;
      auto tty = mlir::cast<coir::TensorType>(alloc.getResult().getType());
      if (tty.getMemorySpace() != 1) return;
      if (!tty.hasDynamicShape()) return;
      llvm::raw_string_ostream ss(expr);
      for (unsigned d = 0; d < tty.getRank(); ++d) {
        if (d > 0) ss << " * ";
        auto dim = tty.getShape()[d];
        if (mlir::ShapedType::isDynamic(dim)) {
          bool found = false;
          for (auto &da : dimArgMeta) {
            if (da.dimIdx == (int64_t)d) {
              ss << "p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
              found = true;
              break;
            }
          }
          if (!found) ss << "1";
        } else {
          ss << dim;
        }
      }
      unsigned elemBits = tty.getElementType().getIntOrFloatBitWidth();
      if (elemBits > 8)
        ss << " * " << (elemBits / 8);
    });
    return expr;
  }

  void emitHostEntry(KernelOp kernel) override {
    prescanDescriptors(kernel);
    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    unsigned numResults = fnType.getNumResults();
    unsigned numTMA = countTMADescs();
    auto dimArgMeta = getDimArgs(kernel);
    unsigned numOrigInputs = fnType.getNumInputs() - dimArgMeta.size();

    auto streamName = getStreamName(kernel);

    llvm::SmallVector<llvm::StringRef> hostElemHints;
    if (auto attr = kernel->getAttrOfType<mlir::ArrayAttr>("coir.host_elem_types"))
      for (auto a : attr)
        hostElemHints.push_back(cast<mlir::StringAttr>(a).getValue());

    if (numResults == 0) {
      os() << "void " << symName << "(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os() << ", ";
        llvm::StringRef hint = (i < hostElemHints.size()) ? hostElemHints[i] : "";
        os() << emitChoreoType(fnType.getInput(i), true, hint) << " p" << i;
      }
      if (!streamName.empty()) {
        if (numOrigInputs > 0) os() << ", ";
        os() << "cudaStream_t " << streamName;
      }
      os() << ") {\n";

      std::string eType = "int8_t";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty) continue;
        eType = emitElementType(tty.getElementType());
        int64_t bytes = getTensorBytes(tty);
        os() << "  " << eType << "* p" << i << "__device = nullptr;\n";
        if (bytes < 0) {
          os() << "  cudaMalloc(&p" << i << "__device, p" << i
             << ".element_count() * sizeof(" << eType << "));\n";
          os() << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), p"
             << i << ".element_count() * sizeof(" << eType
             << "), cudaMemcpyHostToDevice);\n";
        } else {
          os() << "  cudaMalloc(&p" << i << "__device, " << bytes << "ULL);\n";
          os() << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), "
             << bytes << "ULL, cudaMemcpyHostToDevice);\n";
        }
      }

      // TMA descriptor setup
      unsigned tmaIdx = 0;
      for (unsigned i = 0; i < descInfos.size(); ++i) {
        if (!descInfos[i].isTMA) continue;
        auto globalType = descInfos[i].isLoad ? descInfos[i].srcType
                                              : descInfos[i].dstType;
        auto tileType = descInfos[i].isLoad ? descInfos[i].dstType
                                            : descInfos[i].srcType;
        std::string ptr = getTMAGlobalPtr(tmaIdx, kernel);
        emitTMADescriptorSetup(tmaIdx, ptr, globalType, tileType,
                               descInfos[i].swizzleBytes);
        tmaIdx++;
      }

      auto dims = getLaunchDims(kernel);
      emitEntryAssertions(kernel, numOrigInputs, dimArgMeta);
      emitDimChecks(kernel);
      auto dynShmem = getDynShmemExpr(kernel, dimArgMeta);
      auto streamName = getStreamName(kernel);
      bool asyncLaunch = isAsyncLaunch(kernel);

      os() << "  " << devName << "<<<" << dims.gridStr() << ", "
         << dims.blockStr();
      if (!dynShmem.empty() || !streamName.empty())
        os() << ", " << (dynShmem.empty() ? "0" : dynShmem);
      if (!streamName.empty())
        os() << ", " << streamName;
      // Launch order must match kernel signature:
      //   [input ptrs/scalars] [dim args] [TMA maps]
      os() << ">>>(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os() << ", ";
        if (isa<coir::TensorType>(fnType.getInput(i)))
          os() << "p" << i << "__device";
        else
          os() << "p" << i;
      }
      for (auto &da : dimArgMeta) {
        os() << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
      }
      for (unsigned i = 0; i < numTMA; ++i) {
        os() << ", __choreo_tma_" << i << "_tensor_map";
      }
      os() << ");\n";
      if (!asyncLaunch) {
        if (!streamName.empty())
          os() << "  choreo::abend_true(cudaStreamSynchronize("
             << streamName << "));\n";
        else
          os() << "  cudaDeviceSynchronize();\n";
      }

      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (isa<coir::TensorType>(fnType.getInput(i)))
          os() << "  cudaFree(p" << i << "__device);\n";
      }
      os() << "}\n\n";
      return;
    }

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    llvm::StringRef retHint;
    if (auto retAttr = kernel->getAttrOfType<mlir::StringAttr>("coir.host_ret_elem_type"))
      retHint = retAttr.getValue();
    os() << emitChoreoType(fnType.getResult(0), false, retHint) << " " << symName << "(";
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (i > 0) os() << ", ";
      auto &body = kernel.getBody();
      std::string pName = "arg" + std::to_string(i);
      if (!body.empty() && i < body.getArguments().size()) {
        pName = "p" + std::to_string(i);
      }
      llvm::StringRef hint = (i < hostElemHints.size()) ? hostElemHints[i] : "";
      os() << emitChoreoType(fnType.getInput(i), true, hint) << " " << pName;
    }
    if (!streamName.empty()) {
      if (numOrigInputs > 0) os() << ", ";
      os() << "cudaStream_t " << streamName;
    }
    os() << ") {\n";

    std::string eType = emitElementType(resTy.getElementType());

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty) continue;
      std::string inputEType = emitElementType(tty.getElementType());
      int64_t bytes = getTensorBytes(tty);
      os() << "  " << inputEType << "* p" << i << "__device = nullptr;\n";
      if (bytes < 0) {
        os() << "  cudaMalloc(&p" << i << "__device, p" << i
           << ".element_count() * sizeof(" << inputEType << "));\n";
        os() << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), p"
           << i << ".element_count() * sizeof(" << inputEType
           << "), cudaMemcpyHostToDevice);\n";
      } else {
        os() << "  cudaMalloc(&p" << i << "__device, " << bytes << "ULL);\n";
        os() << "  cudaMemcpy(p" << i << "__device, p" << i << ".data(), "
           << bytes << "ULL, cudaMemcpyHostToDevice);\n";
      }
    }

    int64_t resBytes = getTensorBytes(resTy);
    std::string shapeStr;
    {
      llvm::raw_string_ostream ss(shapeStr);
      ss << "{";
      for (unsigned d = 0; d < resTy.getShape().size(); ++d) {
        if (d > 0) ss << ", ";
        auto dim = resTy.getShape()[d];
        if (mlir::ShapedType::isDynamic(dim)) {
          // Find matching dimArgMeta to get runtime shape
          for (auto &da : dimArgMeta) {
            if (da.dimIdx == (int64_t)d) {
              ss << "p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
              break;
            }
          }
        } else {
          ss << dim;
        }
      }
      ss << "}";
    }
    std::string choreoElem;
    if (!retHint.empty()) {
      choreoElem = "choreo::" + retHint.str();
    } else {
      auto resElemTy = resTy.getElementType();
      if (resElemTy.isInteger(8)) choreoElem = "choreo::u8";
      else if (resElemTy.isInteger(16)) choreoElem = "choreo::s16";
      else if (resElemTy.isInteger(32)) choreoElem = "choreo::s32";
      else if (resElemTy.isInteger(64)) choreoElem = "choreo::s64";
      else if (resElemTy.isF32()) choreoElem = "choreo::f32";
      else if (resElemTy.isF16()) choreoElem = "choreo::f16";
      else if (resElemTy.isBF16()) choreoElem = "choreo::bf16";
      else if (resElemTy.isF64()) choreoElem = "choreo::f64";
      else if (isa<mlir::Float8E4M3FNType>(resElemTy))
        choreoElem = "choreo::f8_e4m3";
      else if (isa<mlir::Float8E5M2Type>(resElemTy))
        choreoElem = "choreo::f8_e5m2";
      else if (isa<mlir::Float6E2M3FNType>(resElemTy))
        choreoElem = "choreo::f6_e2m3";
      else if (isa<mlir::Float6E3M2FNType>(resElemTy))
        choreoElem = "choreo::f6_e3m2";
      else if (isa<mlir::Float4E2M1FNType>(resElemTy))
        choreoElem = "choreo::f4_e2m1";
      else choreoElem = "choreo::s32";
    }

    os() << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
       << resTy.getShape().size() << ">(" << shapeStr << ");\n";
    os() << "  " << eType << "* __result__device = nullptr;\n";
    if (resBytes < 0) {
      os() << "  cudaMalloc(&__result__device, __result.element_count()"
         << " * sizeof(" << eType << "));\n";
    } else {
      os() << "  cudaMalloc(&__result__device, " << resBytes << "ULL);\n";
    }

    // TMA descriptor setup
    unsigned tmaIdx = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      auto globalType = descInfos[i].isLoad ? descInfos[i].srcType
                                            : descInfos[i].dstType;
      auto tileType = descInfos[i].isLoad ? descInfos[i].dstType
                                          : descInfos[i].srcType;
      std::string ptr = getTMAGlobalPtr(tmaIdx, kernel);
      emitTMADescriptorSetup(tmaIdx, ptr, globalType, tileType,
                             descInfos[i].swizzleBytes);
      tmaIdx++;
    }

    auto dims = getLaunchDims(kernel);
    emitEntryAssertions(kernel, numOrigInputs, dimArgMeta);
    emitDimChecks(kernel);
    auto dynShmem = getDynShmemExpr(kernel, dimArgMeta);
    bool asyncLaunch = isAsyncLaunch(kernel);

    os() << "  " << devName << "<<<" << dims.gridStr() << ", "
       << dims.blockStr();
    if (!dynShmem.empty() || !streamName.empty())
      os() << ", " << (dynShmem.empty() ? "0" : dynShmem);
    if (!streamName.empty())
      os() << ", " << streamName;
    // Launch order must match kernel signature:
    //   [input ptrs/scalars] [dim args] [output ptrs] [TMA maps]
    os() << ">>>(";
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (i > 0) os() << ", ";
      if (isa<coir::TensorType>(fnType.getInput(i)))
        os() << "p" << i << "__device";
      else
        os() << "p" << i;
    }
    for (auto &da : dimArgMeta) {
      os() << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
    }
    os() << ", __result__device";
    for (unsigned i = 0; i < numTMA; ++i) {
      os() << ", __choreo_tma_" << i << "_tensor_map";
    }
    os() << ");\n";
    if (!asyncLaunch) {
      if (!streamName.empty())
        os() << "  choreo::abend_true(cudaStreamSynchronize("
             << streamName << "));\n";
      else
        os() << "  cudaDeviceSynchronize();\n";
    }
    if (resBytes < 0) {
      os() << "  cudaMemcpy(__result.data(), __result__device, "
         << "__result.element_count() * sizeof(" << eType
         << "), cudaMemcpyDeviceToHost);\n";
    } else {
      os() << "  cudaMemcpy(__result.data(), __result__device, "
         << resBytes << "ULL, cudaMemcpyDeviceToHost);\n";
    }

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (isa<coir::TensorType>(fnType.getInput(i)))
        os() << "  cudaFree(p" << i << "__device);\n";
    }
    os() << "  cudaFree(__result__device);\n";
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
       << ", \"" << msg << "\");" << "\n";
  }

  std::string emitExprInHostScope(
      Value v, KernelOp kernel, DenseMap<Value, std::string> &hostNames,
      unsigned numOrigInputs, llvm::ArrayRef<DimArgMeta> dimArgMeta) {
    auto it = hostNames.find(v);
    if (it != hostNames.end()) return it->second;

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner()->getParentOp() == kernel.getOperation()) {
        unsigned idx = arg.getArgNumber();
        if (idx < numOrigInputs) {
          std::string name = "p" + std::to_string(idx);
          hostNames[v] = name;
          return name;
        }
        unsigned daIdx = idx - numOrigInputs;
        if (daIdx < dimArgMeta.size()) {
          auto &da = dimArgMeta[daIdx];
          std::string name = "(int)p" + std::to_string(da.paramIdx) +
                             ".shape()[" + std::to_string(da.dimIdx) + "]";
          hostNames[v] = name;
          return name;
        }
        std::string name = "/* arg" + std::to_string(idx) + " */";
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

    if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
      return emitExprInHostScope(castOp.getIn(), kernel, hostNames,
                                 numOrigInputs, dimArgMeta);
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      auto lhs = emitExprInHostScope(cmpOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(cmpOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
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
      auto lhs = emitExprInHostScope(addOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(addOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      std::string result = "(" + lhs + " + " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      auto lhs = emitExprInHostScope(mulOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(mulOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      std::string result = "(" + lhs + " * " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      auto lhs = emitExprInHostScope(subOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(subOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      std::string result = "(" + lhs + " - " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      std::string result = "(" + lhs + " / " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(remOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(remOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      std::string result = "(" + lhs + " % " + rhs + ")";
      hostNames[v] = result;
      return result;
    }

    return "/* unknown */";
  }

  void emitEntryAssertions(KernelOp kernel,
                           unsigned numOrigInputs,
                           llvm::ArrayRef<DimArgMeta> dimArgMeta) {
    DenseMap<Value, std::string> hostNames;
    for (auto &ea : entryAssertions) {
      auto cond = emitExprInHostScope(ea.op.getCondition(), kernel, hostNames,
                                      numOrigInputs, dimArgMeta);
      if (cond.find("/* unknown */") != std::string::npos ||
          cond.find("/* arg") != std::string::npos)
        continue;
      os() << "  choreo::runtime_check(" << cond << ", \""
         << ea.op.getMessage() << "\");\n";
    }
  }

  static std::string ordinal(int n) {
    static const char *suffixes[] = {"th", "st", "nd", "rd", "th"};
    int v = n % 100;
    int idx = (v >= 11 && v <= 13) ? 0 : std::min(v % 10, 4);
    return std::to_string(n) + suffixes[idx];
  }

  void emitDimChecks(KernelOp kernel) {
    auto checks = getDimChecks(kernel);
    for (auto &c : checks) {
      os() << "  choreo::runtime_check("
           << "p" << c.param0 << ".shape()[" << c.dim0 << "]"
           << " == "
           << "p" << c.param1 << ".shape()[" << c.dim1 << "]"
           << ", \"The shapes of the " << ordinal(c.param0 + 1)
           << " parameter (dim: " << c.dim0 << ") and the "
           << ordinal(c.param1 + 1)
           << " parameter (dim: " << c.dim1
           << ") are inconsistent.\");\n";
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
    } else if (level == ParallelLevel::GROUP ||
               level == ParallelLevel::GROUPx4) {
      int warpScale = (level == ParallelLevel::GROUPx4) ? 4 : 1;
      std::string warpId = "(threadIdx.x / " +
                           std::to_string(32 * warpScale) + ")";
      if (args.size() == 1) {
        valueNames[args[0]] = warpId;
      } else {
        for (unsigned i = 0; i < args.size(); ++i) {
          int64_t divisor = 1;
          for (unsigned j = i + 1; j < args.size(); ++j)
            divisor *= bounds[j];
          std::string expr = "(" + warpId + " / " +
                             std::to_string(divisor) + " % " +
                             std::to_string(bounds[i]) + ")";
          valueNames[args[i]] = expr;
        }
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

  bool isCTMMAExec(MMAExecOp exec) {
    return exec.getMmaAtomNameAttr() != nullptr;
  }

  // Find the MMAExecOp that uses this value (through the ctmmaValues set)
  MMAExecOp findCTMMAExec(mlir::Operation *root) {
    MMAExecOp found = nullptr;
    root->walk([&](MMAExecOp exec) {
      if (exec.getMmaAtomNameAttr())
        found = exec;
    });
    return found;
  }

  bool isValueCTMMA(Value v) {
    return ctmmaValues.count(v) > 0;
  }

  void emitMMAFill(MMAFillOp op) override {
    auto fragTy = cast<coir::MMAFragType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    mmaAccumulators.insert(op.getResult());

    if (isValueCTMMA(op.getResult())) {
      auto exec = findCTMMAExec(op->getParentOfType<KernelOp>());
      int64_t regNum = (exec && exec.getRegNumDAttr()) ? exec.getRegNumDAttr().getInt() : 4;
      std::string elemTy = fragTy.getElementType().isF32() ? "float" : "int32_t";
      os() << getIndent() << elemTy << " " << name << "[" << regNum << "];\n";
      os() << getIndent() << "for (int __i = 0; __i < " << regNum
           << "; ++__i) " << name << "[__i] = " << getName(op.getValue())
           << ";\n";
      return;
    }

    if (useWGMMA) {
      auto shape = fragTy.getShape();
      bool isOperandFrag = wgmmaOperandFrags.count(op.getResult()) > 0;
      if (isOperandFrag) {
        int64_t M = shape.size() > 0 ? shape[0] : 16;
        int64_t K = shape.size() > 1 ? shape[1] : 16;
        int64_t fragElems = (M * K) / 128; // per-thread elements for RS
        os() << getIndent() << "half " << name << "[" << fragElems << "];\n";
        os() << getIndent() << "for (int __i = 0; __i < " << fragElems
             << "; ++__i) " << name << "[__i] = choreo::f32_to_f16("
             << getName(op.getValue()) << ");\n";
      } else {
        int64_t M = shape.size() > 0 ? shape[0] : 64;
        int64_t N = shape.size() > 1 ? shape[1] : 64;
        int64_t accRegs = (M * N) / 128; // accumulator regs per thread
        os() << getIndent() << "float " << name << "[" << accRegs << "];\n";
        os() << getIndent() << "for (int __i = 0; __i < " << accRegs
             << "; ++__i) " << name << "[__i] = " << getName(op.getValue())
             << ";\n";
      }
      return;
    }

    os() << getIndent()
       << emitWMMAFragType(fragTy, "accumulator", "", op.getResult())
       << " " << name << ";\n";
    os() << getIndent() << "wmma::fill_fragment(" << name << ", "
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

  void emitMMALoad(MMALoadOp op) override {
    auto fragTy = cast<coir::MMAFragType>(op.getResult().getType());
    std::string name = getName(op.getResult());

    if (isValueCTMMA(op.getResult())) {
      auto exec = findCTMMAExec(op->getParentOfType<KernelOp>());
      if (exec) {
        std::string atomName = exec.getMmaAtomNameAttr().getValue().str();
        auto roleIt = mmaFragRoles.find(op.getResult());
        std::string role = roleIt != mmaFragRoles.end() ? roleIt->second : "matrix_a";
        std::string suffix = (role == "matrix_a") ? "a" : "b";

        std::string srcTensor = emitCuteTensor(op.getSource(),
                                               "_mma_ld_" + std::to_string(nextId++));
        os() << getIndent() << "auto " << name << " = load_fragment_" << suffix
             << "<" << atomName << ">(" << srcTensor << ");\n";
        return;
      }
    }

    if (useWGMMA) {
      const char *swizEnum = "WGMMA_Swizzle::NS";
      if (auto sb = op.getSwizzleBytes()) {
        if (*sb == 32) swizEnum = "WGMMA_Swizzle::B32";
        else if (*sb == 64) swizEnum = "WGMMA_Swizzle::B64";
        else if (*sb == 128) swizEnum = "WGMMA_Swizzle::B128";
      }
      os() << getIndent() << "uint64_t " << name
           << " = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, "
           << swizEnum << ">("
           << getName(op.getSource()) << ");\n";
      return;
    }

    auto roleIt = mmaFragRoles.find(op.getResult());
    std::string role = roleIt != mmaFragRoles.end() ? roleIt->second : "matrix_a";
    auto layoutIt = mmaFragLayouts.find(op.getResult());
    std::string layout = layoutIt != mmaFragLayouts.end() ? layoutIt->second : "";
    os() << getIndent()
       << emitWMMAFragType(fragTy, role, layout, op.getResult())
       << " " << name << ";\n";
    int64_t ldm = getSourceLeadingDim(op.getSource());
    os() << getIndent() << "wmma::load_matrix_sync(" << name << ", "
       << getName(op.getSource()) << ", " << ldm << ");\n";
  }

  void emitMMAExec(MMAExecOp op) override {
    std::string acc = getName(op.getAccumulator());
    valueNames[op.getResult()] = acc;

    if (isCTMMAExec(op)) {
      int64_t regD = op.getRegNumDAttr().getInt();
      int64_t regA = op.getRegNumAAttr().getInt();
      int64_t regB = op.getRegNumBAttr().getInt();
      std::string lhs = getName(op.getLhs());
      std::string rhs = getName(op.getRhs());

      std::string fmaAtom;
      if (auto attr = op->getAttrOfType<StringAttr>("fma_atom"))
        fmaAtom = attr.getValue().str();
      else
        fmaAtom = "cute::SM80_16x8x8_F32F16F16F32_TN";

      os() << getIndent() << fmaAtom << "::fma(";
      for (int64_t i = 0; i < regD; ++i)
        os() << acc << "[" << i << "], ";
      for (int64_t i = 0; i < regA; ++i)
        os() << lhs << "[" << i << "], ";
      for (int64_t i = 0; i < regB; ++i)
        os() << rhs << "[" << i << "], ";
      for (int64_t i = 0; i < regD; ++i) {
        os() << acc << "[" << i << "]";
        if (i < regD - 1) os() << ", ";
      }
      os() << ");\n";
      return;
    }

    if (useWGMMA) {
      std::string lhs = getName(op.getLhs());
      std::string rhs = getName(op.getRhs());

      auto accTy = cast<coir::MMAFragType>(op.getAccumulator().getType());
      auto lhsTy = cast<coir::MMAFragType>(op.getLhs().getType());
      int64_t M = accTy.getShape()[0];
      int64_t N = accTy.getShape()[1];
      int64_t K = lhsTy.getShape().size() > 1 ? lhsTy.getShape()[1] : 16;

      bool isRS = wgmmaOperandFrags.count(op.getLhs()) > 0;
      std::string mode = isRS ? "RS" : "SS";

      os() << getIndent() << "warpgroup_fence_operand(" << acc << ");\n";
      os() << getIndent() << "warpgroup_arrive();\n";
      if (isRS)
        os() << getIndent() << "warpgroup_fence_operand(" << lhs << ");\n";

      os() << getIndent() << "cute::SM90::GMMA::MMA_" << M << "x" << N << "x"
           << K << "_F32F16F16_" << mode
           << "<static_cast<cute::SM90::GMMA::Major>(0), "
              "static_cast<cute::SM90::GMMA::Major>(0)>::fma(\n";
      incIndent();
      // A operand: 4 uint32_t regs from the register fragment
      if (isRS) {
        os() << getIndent();
        int64_t lhsU32 = (M * K) / (128 * 2); // half->uint32 packing: 2 halves per u32
        for (int64_t i = 0; i < lhsU32; ++i) {
          os() << "reinterpret_cast<const uint32_t*>(" << lhs << ")[" << i
               << "]";
          os() << ", ";
        }
        os() << "\n";
      } else {
        os() << getIndent() << lhs << ", ";
      }
      // B operand (descriptor)
      os() << getIndent() << rhs << ",\n";
      // D regs (output accumulator)
      os() << getIndent();
      int64_t accRegs = (M * N) / 128;
      for (int64_t i = 0; i < accRegs; ++i) {
        os() << acc << "[" << i << "]";
        if (i < accRegs - 1) os() << ", ";
      }
      os() << ");\n";
      decIndent();
      os() << getIndent() << "warpgroup_commit_batch();\n";
      os() << getIndent() << "warpgroup_wait<0>();\n";
      os() << getIndent() << "warpgroup_fence_operand(" << acc << ");\n";
      return;
    }

    os() << getIndent() << "wmma::mma_sync(" << acc << ", "
       << getName(op.getLhs()) << ", " << getName(op.getRhs()) << ", "
       << acc << ");\n";
  }

  unsigned mmaStoreCvtIdx = 0;

  void emitMMAStore(MMAStoreOp op) override {
    // CTMMA store: use store_fragment_d helper from choreo_cute.h
    auto fragVal = op.getFragment();
    if (isValueCTMMA(fragVal)) {
      auto exec = findCTMMAExec(op->getParentOfType<KernelOp>());
      if (exec) {
        std::string atomName = exec.getMmaAtomNameAttr().getValue().str();
        // For CTMMA, the actual output tile is MxN (derived from operand shapes)
        auto lhsTy = mlir::cast<coir::MMAFragType>(exec.getLhs().getType());
        auto rhsTy = mlir::cast<coir::MMAFragType>(exec.getRhs().getType());
        int64_t mmaM = lhsTy.getShape()[0];
        int64_t mmaN = rhsTy.getShape().size() > 1 ? rhsTy.getShape()[1] : rhsTy.getShape()[0];
        std::string suffix = "_mma_st_" + std::to_string(nextId++);
        // Emit tensor with correct MxN shape (override the fragment type's shape)
        auto dstTy = mlir::cast<coir::TensorType>(op.getDest().getType());
        std::string elemTy = emitElementType(dstTy.getElementType());
        int32_t ms = dstTy.getMemorySpace();
        int64_t ldm = dstTy.getShape().size() > 1 ? dstTy.getStrides().empty()
            ? dstTy.getShape()[1] : dstTy.getStrides()[0] : 1;
        // Get leading dim from parent tensor shape
        auto srcOp = op.getDest().getDefiningOp();
        if (auto tileOp = mlir::dyn_cast_or_null<coir::TensorTileOp>(srcOp)) {
          auto parentTy = mlir::cast<coir::TensorType>(tileOp.getSource().getType());
          if (parentTy.getShape().size() > 1)
            ldm = parentTy.getShape()[1];
        }
        std::string shapeName = "__shape" + suffix;
        std::string strideName = "__stride" + suffix;
        std::string layoutName = "__layout" + suffix;
        std::string tensorName = "__tensor" + suffix;
        os() << getIndent() << "auto " << shapeName
             << " = cute::make_shape(cute::Int<" << mmaM << ">{}, cute::Int<" << mmaN << ">{});\n";
        os() << getIndent() << "auto " << strideName
             << " = cute::make_stride(cute::Int<" << ldm << ">{}, cute::Int<1>{});\n";
        os() << getIndent() << "auto " << layoutName
             << " = cute::make_layout(" << shapeName << ", " << strideName << ");\n";
        std::string ptrName = getName(op.getDest());
        os() << getIndent() << "auto " << tensorName << " = cute::make_tensor(";
        if (ms <= 0)
          os() << "cute::make_gmem_ptr<" << elemTy << ">((" << elemTy << "*)" << ptrName << ")";
        else if (ms == 1)
          os() << "cute::make_smem_ptr((" << elemTy << "*)" << ptrName << ")";
        else
          os() << "((" << elemTy << "*)" << ptrName << ")";
        os() << ", " << layoutName << ");\n";
        os() << getIndent() << "store_fragment_d<" << atomName << ">("
             << tensorName << ", " << getName(fragVal) << ");\n";
        return;
      }
    }

    if (useWGMMA) {
      auto fragTy = cast<coir::MMAFragType>(op.getFragment().getType());
      auto shape = fragTy.getShape();
      int64_t M = shape.size() > 0 ? shape[0] : 64;
      int64_t N = shape.size() > 1 ? shape[1] : 64;
      int64_t accRegs = (M * N) / 128;
      std::string suffix = "_wgmma_st_" + std::to_string(nextId++);
      auto dstTy = cast<coir::TensorType>(op.getDest().getType());
      std::string elemTy = emitElementType(dstTy.getElementType());
      int64_t ldm = getSourceLeadingDim(op.getDest());
      std::string shapeName = "__shape" + suffix;
      std::string layoutName = "__layout" + suffix;
      std::string tensorName = "__tensor" + suffix;
      os() << getIndent() << "auto " << shapeName
           << " = cute::make_shape(cute::Int<" << M << ">{}, cute::Int<"
           << N << ">{});\n";
      os() << getIndent() << "auto " << layoutName
           << " = cute::make_layout(" << shapeName
           << ", cute::make_stride(cute::Int<" << ldm
           << ">{}, cute::Int<1>{}));\n";
      os() << getIndent() << "auto " << tensorName
           << " = cute::make_tensor(cute::make_gmem_ptr<" << elemTy << ">(("
           << elemTy << "*)" << getName(op.getDest()) << "), " << layoutName
           << ");\n";
      os() << getIndent() << "store_fragment_d<CUTE_WGMMA_M64K16, " << N
           << ">(" << tensorName << ", " << getName(op.getFragment()) << ");\n";
      return;
    }

    int64_t ldm = getSourceLeadingDim(op.getDest());
    auto fragTy = cast<coir::MMAFragType>(op.getFragment().getType());
    auto dstTy = cast<coir::TensorType>(op.getDest().getType());

    if (fragTy.getElementType().isF32() && dstTy.getElementType().isF16()) {
      auto shape = fragTy.getShape();
      int64_t tileM = shape.size() > 0 ? shape[0] : 16;
      int64_t tileN = shape.size() > 1 ? shape[1] : 16;
      int64_t tileElems = tileM * tileN;
      int64_t numWarps = 1;
      auto *parentOp = op->getParentOp();
      while (parentOp) {
        if (auto par = dyn_cast<ParallelOp>(parentOp)) {
          auto lvl = par.getLevel();
          if (lvl == ParallelLevel::GROUP ||
              lvl == ParallelLevel::GROUPx4) {
            numWarps = 1;
            for (auto b : par.getBounds()) numWarps *= b;
            if (lvl == ParallelLevel::GROUPx4) numWarps *= 4;
            break;
          }
        }
        parentOp = parentOp->getParentOp();
      }
      std::string idx = std::to_string(mmaStoreCvtIdx++);
      os() << getIndent() << "{\n";
      incIndent();
      os() << getIndent() << "__shared__ float __mma_cvt_" << idx
           << "[" << numWarps * tileElems << "];\n";
      os() << getIndent() << "float* __mma_cvt_" << idx
           << "_local = __mma_cvt_" << idx
           << " + (threadIdx.x / 32) * " << tileElems << ";\n";
      os() << getIndent() << "wmma::store_matrix_sync(__mma_cvt_" << idx
           << "_local, " << getName(op.getFragment()) << ", " << tileN
           << ", wmma::mem_row_major);\n";
      os() << getIndent() << "__syncwarp();\n";
      os() << getIndent() << "for (int _r = 0; _r < " << tileM
           << "; ++_r)\n";
      incIndent();
      os() << getIndent() << "for (int _c = threadIdx.x % 32; _c < "
           << tileN << "; _c += 32)\n";
      incIndent();
      os() << getIndent() << getName(op.getDest()) << "[_r * " << ldm
           << " + _c] = __float2half(__mma_cvt_" << idx
           << "_local[_r * " << tileN << " + _c]);\n";
      decIndent();
      decIndent();
      decIndent();
      os() << getIndent() << "}\n";
    } else {
      os() << getIndent() << "wmma::store_matrix_sync("
           << getName(op.getDest()) << ", " << getName(op.getFragment())
           << ", " << ldm << ", wmma::mem_row_major);\n";
    }
  }

  TileLayout getLayoutForValue(Value v) {
    auto it = tileLayouts.find(v);
    if (it != tileLayouts.end())
      return it->second;
    auto tty = dyn_cast<coir::TensorType>(v.getType());
    if (!tty) return {};
    auto shape = tty.getShape();
    TileLayout layout;
    layout.shape.assign(shape.begin(), shape.end());
    if (!tty.hasDynamicShape()) {
      layout.strides.resize(shape.size());
      int64_t s = 1;
      for (int i = (int)shape.size() - 1; i >= 0; --i) {
        layout.strides[i] = s;
        s *= shape[i];
      }
    }
    return layout;
  }

  void emitCuteMakeShape(llvm::StringRef varName,
                         const llvm::SmallVector<int64_t> &dims,
                         KernelOp kernel = nullptr) {
    os() << getIndent() << "auto " << varName << " = cute::make_shape(";
    llvm::SmallVector<std::string> dimArgNames;
    if (kernel) {
      auto fnTy = kernel.getFunctionType();
      for (unsigned a = 0; a < fnTy.getNumInputs(); ++a)
        if (fnTy.getInput(a).isIndex())
          dimArgNames.push_back(getName(kernel.getBody().getArgument(a)));
    }
    unsigned dynIdx = 0;
    for (unsigned i = 0; i < dims.size(); ++i) {
      if (i > 0) os() << ", ";
      if (mlir::ShapedType::isDynamic(dims[i])) {
        if (dynIdx < dimArgNames.size())
          os() << dimArgNames[dynIdx];
        else
          os() << "1";
        dynIdx++;
      } else {
        os() << "cute::Int<" << dims[i] << ">{}";
      }
    }
    os() << ");\n";
  }

  void emitCuteMakeStride(llvm::StringRef varName,
                          const llvm::SmallVector<int64_t> &strides) {
    os() << getIndent() << "auto " << varName << " = cute::make_stride(";
    for (unsigned i = 0; i < strides.size(); ++i) {
      if (i > 0) os() << ", ";
      if (mlir::ShapedType::isDynamic(strides[i]))
        os() << "cute::Int<1>{}";
      else
        os() << "cute::Int<" << strides[i] << ">{}";
    }
    os() << ");\n";
  }

  std::string emitCuteTensor(Value v, const std::string &suffix) {
    auto layout = getLayoutForValue(v);
    auto tty = dyn_cast<coir::TensorType>(v.getType());
    int32_t ms = tty ? tty.getMemorySpace() : 0;

    std::string shapeName = "__shape" + suffix;
    std::string strideName = "__stride" + suffix;
    std::string layoutName = "__layout" + suffix;
    std::string tensorName = "__tensor" + suffix;

    KernelOp enclosingKernel = nullptr;
    if (tty && tty.hasDynamicShape())
      if (auto *block = v.getParentBlock())
        enclosingKernel = block->getParent()->getParentOfType<KernelOp>();
    emitCuteMakeShape(shapeName, layout.shape, enclosingKernel);

    // For dynamic tensors, use shape-only layout (CUTE computes row-major strides)
    if (tty && tty.hasDynamicShape()) {
      os() << getIndent() << "auto " << layoutName
         << " = cute::make_layout(" << shapeName << ");\n";
    } else {
      emitCuteMakeStride(strideName, layout.strides);
      os() << getIndent() << "auto " << layoutName
         << " = cute::make_layout(" << shapeName << ", " << strideName
         << ");\n";
    }

    std::string elemTy = tty ? emitElementType(tty.getElementType()) : "int";
    std::string ptrName = getName(v);
    os() << getIndent() << "auto " << tensorName << " = cute::make_tensor(";
    if (ms <= 0)
      os() << "cute::make_gmem_ptr<" << elemTy << ">((" << elemTy << "*)"
         << ptrName << ")";
    else if (ms == 1)
      os() << "cute::make_smem_ptr((" << elemTy << "*)" << ptrName << ")";
    else
      os() << "((" << elemTy << "*)" << ptrName << ")";
    os() << ", " << layoutName << ");\n";

    return tensorName;
  }

  std::string emitCuteTensorFromPtr(const std::string &ptrExpr,
                                    coir::TensorType origType,
                                    llvm::ArrayRef<int64_t> shape,
                                    const std::string &suffix,
                                    KernelOp kernel = nullptr) {
    int32_t ms = origType.getMemorySpace();
    std::string shapeName = "__shape" + suffix;
    std::string strideName = "__stride" + suffix;
    std::string layoutName = "__layout" + suffix;
    std::string tensorName = "__tensor" + suffix;

    llvm::SmallVector<int64_t> shapeVec(shape.begin(), shape.end());
    bool hasDyn = llvm::any_of(shape, mlir::ShapedType::isDynamic);
    emitCuteMakeShape(shapeName, shapeVec, kernel);
    if (hasDyn) {
      os() << getIndent() << "auto " << layoutName
         << " = cute::make_layout(" << shapeName << ");\n";
    } else {
      llvm::SmallVector<int64_t> strides(shape.size());
      int64_t s = 1;
      for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
      }
      emitCuteMakeStride(strideName, strides);
      os() << getIndent() << "auto " << layoutName
         << " = cute::make_layout(" << shapeName << ", " << strideName
         << ");\n";
    }

    std::string elemTy = emitElementType(origType.getElementType());
    os() << getIndent() << "auto " << tensorName << " = cute::make_tensor(";
    if (ms <= 0)
      os() << "cute::make_gmem_ptr<" << elemTy << ">((" << elemTy << "*)"
         << ptrExpr << ")";
    else if (ms == 1)
      os() << "cute::make_smem_ptr((" << elemTy << "*)" << ptrExpr << ")";
    else
      os() << "((" << elemTy << "*)" << ptrExpr << ")";
    os() << ", " << layoutName << ");\n";

    return tensorName;
  }

  void emitNaiveCopy(Value src, Value dst) {
    unsigned id = nextId++;
    std::string srcTensor = emitCuteTensor(src, "_src_" + std::to_string(id));
    std::string dstTensor = emitCuteTensor(dst, "_dst_" + std::to_string(id));
    os() << getIndent() << "choreo::naive_copy(" << srcTensor << ", "
       << dstTensor << ");\n";
  }

  // --- DMA Descriptor Pipeline Emission ---

  void emitDMAConstDesc(DMAConstDescOp op) override {
    // DMAConstDescOp is a compile-time descriptor -- no runtime code emitted.
    // We propagate the desc value index for use by downstream ops.
    auto it = descValueToIndex.find(op.getOut());
    if (it == descValueToIndex.end()) return;
    // Track the prefetch chain: prefetch.desc consumes const.desc result
    valueNames[op.getOut()] = "__desc_" + std::to_string(it->second);
  }

  void emitDMAPrefetch(DMADescPrefetchOp op) override {
    // Prefetch materializes the descriptor. For TMA on GPU, this is a
    // prefetch_tma_descriptor intrinsic. For cp.async DMA, this is a no-op.
    Value in = op.getIn();
    auto it = descValueToIndex.find(in);
    if (it == descValueToIndex.end()) {
      // Walk the chain to find the original const.desc
      valueNames[op.getOut()] = "/* prefetch unknown */";
      return;
    }
    unsigned descIdx = it->second;
    descValueToIndex[op.getOut()] = descIdx;
    valueNames[op.getOut()] = "__desc_" + std::to_string(descIdx);
  }

  void emitDMARuntimeDesc(DMADescRuntimeOp op) override {
    Value in = op.getIn();
    auto it = descValueToIndex.find(in);
    if (it == descValueToIndex.end()) {
      valueNames[op.getOut()] = "/* rt_desc unknown */";
      return;
    }
    unsigned descIdx = it->second;
    descValueToIndex[op.getOut()] = descIdx;
    valueNames[op.getOut()] = "__desc_" + std::to_string(descIdx);
    // Capture runtime offsets for use in TMA coordinate emission.
    SmallVector<Value, 4> offs(op.getOffsets().begin(), op.getOffsets().end());
    descRuntimeOffsets[op.getOut()] = std::move(offs);
  }

  unsigned getTMAIndexForDesc(unsigned descIdx) {
    unsigned tmaIdx = 0;
    for (unsigned i = 0; i < descIdx; ++i) {
      if (descInfos[i].isTMA)
        tmaIdx++;
    }
    return tmaIdx;
  }

  void emitDMAInvoke(DMAInvokeOp op) override {
    Value descVal = op.getDesc();
    auto it = descValueToIndex.find(descVal);
    auto registerDoneToken = [&](Value token) {
      std::string n = getName(token);
      os() << getIndent() << "int " << n << " = 0;\n";
      dmaTokens.insert(token);
    };

    if (it == descValueToIndex.end()) {
      os() << getIndent() << "// DMA invoke (no descriptor info)\n";
      os() << getIndent() << "__syncthreads();\n";
      registerDoneToken(op.getDone());
      return;
    }

    unsigned descIdx = it->second;
    auto &desc = descInfos[descIdx];
    registerDoneToken(op.getDone());

    // Check if PlanDMACopy stamped tiled copy attrs
    if (op.getThrLayout()) {
      emitTiledCopy(op, desc);
      return;
    }

    // Collect runtime offsets if any
    SmallVector<Value, 4> offsets;
    auto offsIt = descRuntimeOffsets.find(descVal);
    if (offsIt != descRuntimeOffsets.end())
      offsets = offsIt->second;

    if (desc.isTMA) {
      emitTMAInvoke(descIdx, desc, offsets);
    } else {
      emitCpAsyncInvoke(descIdx, desc, offsets, op);
    }
  }

  void emitTMAInvoke(unsigned descIdx, const DescInfo &desc,
                     const SmallVector<Value, 4> &offsets) {
    unsigned tmaIdx = getTMAIndexForDesc(descIdx);
    std::string atomName = "choreo_copy_atom_t_" + std::to_string(tmaIdx);
    std::string mapName = "__choreo_tma_" + std::to_string(tmaIdx) +
                          "_tensor_map";

    // Compute TMA coordinates from runtime offsets.
    // Offsets are tile indices; multiply by tile dimensions to get element coords.
    // TMA expects coordinates in innermost-first order.
    auto tileShape = desc.isLoad ? desc.dstType.getShape()
                                 : desc.srcType.getShape();
    auto globalShape = desc.isLoad ? desc.srcType.getShape()
                                   : desc.dstType.getShape();
    unsigned rank = globalShape.size();

    if (desc.isLoad) {
      auto shape = desc.dstType.getShape();
      int64_t elemBits = desc.dstType.getElementType().getIntOrFloatBitWidth();
      int64_t totalElems = 1;
      for (auto d : shape) totalElems *= d;
      int64_t transferBytes = totalElems * elemBits / 8;

      std::string loadFunc = rank <= 2
          ? "choreo::tma_load_2d_shared_cta_global_mbarrier"
          : "choreo::tma_load_3d_shared_cta_global_mbarrier";

      std::string dstName;
      if (auto constOp =
              desc.constDescResult.getDefiningOp<DMAConstDescOp>()) {
        dstName = getName(constOp.getDest());
      } else {
        dstName = "/* unknown dest */";
      }

      os() << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
      incIndent();
      os() << getIndent() << "choreo::tma_mbarrier_expect_tx("
         << atomName << ".ptx_barrier(), " << transferBytes << ");\n";
      os() << getIndent() << loadFunc << "((void*)(" << dstName
         << "), (const void*)&" << mapName << ", "
         << atomName << ".ptx_barrier()";
      // Emit coordinates (innermost-first)
      for (int d = (int)rank - 1; d >= 0; --d) {
        os() << ", ";
        if (d < (int)offsets.size()) {
          int64_t tileDim = tileShape[d];
          if (tileDim > 1)
            os() << getName(offsets[d]) << " * " << tileDim;
          else
            os() << getName(offsets[d]);
        } else {
          os() << "0";
        }
      }
      os() << ");\n";
      decIndent();
      os() << getIndent() << "}\n";

      os() << getIndent() << "choreo::tma_mbarrier_wait_parity("
         << atomName << ".ptx_barrier(), "
         << atomName << ".ptx_phase_bit());\n";
      os() << getIndent() << atomName << ".toggle_ptx_phase();\n";
    } else {
      // S2G TMA store
      std::string srcName;
      if (auto constOp =
              desc.constDescResult.getDefiningOp<DMAConstDescOp>()) {
        srcName = getName(constOp.getSource());
      } else {
        srcName = "/* unknown src */";
      }

      os() << getIndent() << "cde::fence_proxy_async_shared_cta();\n";
      os() << getIndent() << "__syncthreads();\n";
      os() << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
      incIndent();

      auto srcType = desc.srcType;
      std::string storeFunc = rank <= 2
          ? "cde::cp_async_bulk_tensor_2d_shared_to_global"
          : "cde::cp_async_bulk_tensor_3d_shared_to_global";

      os() << getIndent() << storeFunc << "(&" << mapName;
      // Emit coordinates (innermost-first)
      for (int d = (int)rank - 1; d >= 0; --d) {
        os() << ", ";
        if (d < (int)offsets.size()) {
          int64_t tileDim = tileShape[d];
          if (tileDim > 1)
            os() << getName(offsets[d]) << " * " << tileDim;
          else
            os() << getName(offsets[d]);
        } else {
          os() << "0";
        }
      }
      os() << ", " << srcName << ");\n";
      os() << getIndent() << "cde::cp_async_bulk_commit_group();\n";
      os() << getIndent() << "cde::cp_async_bulk_wait_group_read<0>();\n";
      decIndent();
      os() << getIndent() << "}\n";
    }
  }

  void emitTiledCopy(DMAInvokeOp op, const DescInfo &desc) {
    auto thrLayoutArr = *op.getThrLayout();
    auto valLayoutArr = *op.getValLayout();
    auto atomStr = op.getCopyAtomAttr().getValue().str();
    bool needPred = op.getNeedPredAttr() && op.getNeedPredAttr().getValue();
    bool swizzle = op.getSwizzleAttr() && op.getSwizzleAttr().getValue();

    int64_t thrRows = thrLayoutArr[0];
    int64_t thrCols = thrLayoutArr[1];
    int64_t valRows = valLayoutArr[0];
    int64_t valCols = valLayoutArr[1];

    auto constOp = desc.constDescResult.getDefiningOp<DMAConstDescOp>();
    if (!constOp) {
      os() << getIndent() << "// tiled copy: no const desc, fallback\n";
      os() << getIndent() << "__syncthreads();\n";
      return;
    }

    // Resolve element type name for atom template parameter
    auto tileType = desc.isLoad ? desc.dstType : desc.srcType;
    std::string elemTy = emitElementType(tileType.getElementType());

    // Substitute ELEM placeholder in atom name
    std::string atom = atomStr;
    size_t pos = atom.find("ELEM");
    while (pos != std::string::npos) {
      atom.replace(pos, 4, elemTy);
      pos = atom.find("ELEM", pos + elemTy.size());
    }

    // Build source and destination CuTe tensors
    unsigned id = nextId++;
    std::string suffix = "_tc_" + std::to_string(id);

    Value srcVal = constOp.getSource();
    Value dstVal = constOp.getDest();

    // Handle runtime offsets (sliced pointer) if present
    Value descV = op.getDesc();
    SmallVector<Value, 4> offsets;
    auto offsIt = descRuntimeOffsets.find(descV);
    if (offsIt != descRuntimeOffsets.end())
      offsets = offsIt->second;

    std::string srcTensor, dstTensor;
    if (!offsets.empty()) {
      Value globalVal = desc.isLoad ? srcVal : dstVal;
      Value localVal = desc.isLoad ? dstVal : srcVal;
      auto globalTy = cast<coir::TensorType>(globalVal.getType());
      auto localTy = cast<coir::TensorType>(localVal.getType());

      auto globalShape = globalTy.getShape();
      llvm::SmallVector<int64_t> globalStrides(globalShape.size());
      {
        int64_t s = 1;
        for (int i = (int)globalShape.size() - 1; i >= 0; --i) {
          globalStrides[i] = s;
          if (!mlir::ShapedType::isDynamic(globalShape[i]))
            s *= globalShape[i];
        }
      }

      std::string ptrName = getName(globalVal);
      std::string offExpr;
      if (offsets.size() == 1) {
        int64_t tileElems = 1;
        for (auto d : localTy.getShape()) tileElems *= d;
        offExpr = getName(offsets[0]);
        if (tileElems > 1)
          offExpr += " * " + std::to_string(tileElems);
      } else {
        for (unsigned i = 0; i < offsets.size(); ++i) {
          std::string term = getName(offsets[i]);
          if (globalStrides[i] != 1)
            term += " * " + std::to_string(globalStrides[i]);
          if (i == 0) offExpr = term;
          else offExpr += " + " + term;
        }
      }

      std::string slicedPtr = "__tc_ptr_" + std::to_string(id);
      os() << getIndent() << emitElementType(globalTy.getElementType())
         << "* " << slicedPtr << " = ("
         << emitElementType(globalTy.getElementType()) << "*)"
         << ptrName << " + " << offExpr << ";\n";

      if (desc.isLoad) {
        srcTensor = emitCuteTensorFromPtr(slicedPtr, globalTy,
                                          localTy.getShape(), suffix + "_s");
        dstTensor = emitCuteTensor(localVal, suffix + "_d");
      } else {
        srcTensor = emitCuteTensor(localVal, suffix + "_s");
        dstTensor = emitCuteTensorFromPtr(slicedPtr, globalTy,
                                          localTy.getShape(), suffix + "_d");
      }
    } else {
      srcTensor = emitCuteTensor(srcVal, suffix + "_s");
      dstTensor = emitCuteTensor(dstVal, suffix + "_d");
    }

    // Emit the choreo::tiled_copy call
    os() << getIndent() << "choreo::tiled_copy<" << atom << ", "
       << thrRows << ", " << thrCols << ", "
       << valRows << ", " << valCols << ", "
       << (swizzle ? "true" : "false") << ", "
       << (needPred ? "true" : "false") << ", false>("
       << srcTensor << ", " << dstTensor << ", ";

    if (needPred) {
      auto predArr = *op.getPrediction();
      os() << "[&](const auto& __coord) { return cute::elem_less(__coord, "
              "cute::make_shape(cute::Int<"
           << predArr[0] << ">{}, cute::Int<"
           << predArr[1] << ">{})); }";
    } else {
      os() << "[](const auto&) { return true; }";
    }
    os() << ");\n";

    // Emit synchronization: cp.async fence/wait for cp_async atoms, else syncthreads
    bool isCpAsync = atom.find("CP_ASYNC") != std::string::npos;
    if (isCpAsync) {
      os() << getIndent() << "cute::cp_async_fence();\n";
      os() << getIndent() << "cute::cp_async_wait<0>();\n";
      os() << getIndent() << "__syncthreads();\n";
    } else {
      os() << getIndent() << "__syncthreads();\n";
    }
  }

  void emitCpAsyncInvoke(unsigned /*descIdx*/, const DescInfo &desc,
                         const SmallVector<Value, 4> &offsets,
                         DMAInvokeOp invokeOp) {
    // cp.async DMA: cooperative copy with fence/wait
    if (auto constOp =
            desc.constDescResult.getDefiningOp<DMAConstDescOp>()) {
      os() << getIndent() << "// cp.async DMA\n";
      if (!offsets.empty()) {
        Value globalVal = desc.isLoad ? constOp.getSource() : constOp.getDest();
        Value localVal = desc.isLoad ? constOp.getDest() : constOp.getSource();
        auto globalTy = cast<coir::TensorType>(globalVal.getType());
        auto localTy = cast<coir::TensorType>(localVal.getType());

        // Find enclosing kernel for dynamic dim resolution
        KernelOp enclosingKernel = nullptr;
        if (localTy.hasDynamicShape() || globalTy.hasDynamicShape())
          if (auto *block = constOp->getBlock())
            enclosingKernel =
                block->getParent()->getParentOfType<KernelOp>();

        // Compute tile element count (static or runtime expression)
        bool hasDynLocal = localTy.hasDynamicShape();
        std::string tileElemsExpr;
        if (hasDynLocal) {
          llvm::raw_string_ostream ss(tileElemsExpr);
          llvm::SmallVector<std::string> dimArgNames;
          if (enclosingKernel) {
            auto fnTy = enclosingKernel.getFunctionType();
            for (unsigned a = 0; a < fnTy.getNumInputs(); ++a)
              if (fnTy.getInput(a).isIndex())
                dimArgNames.push_back(
                    getName(enclosingKernel.getBody().getArgument(a)));
          }
          unsigned dynIdx = 0;
          for (unsigned d = 0; d < localTy.getRank(); ++d) {
            if (d > 0) ss << " * ";
            auto dim = localTy.getShape()[d];
            if (mlir::ShapedType::isDynamic(dim)) {
              if (dynIdx < dimArgNames.size())
                ss << dimArgNames[dynIdx];
              else
                ss << "1";
              dynIdx++;
            } else {
              ss << dim;
            }
          }
        } else {
          int64_t tileElems = 1;
          for (auto d : localTy.getShape()) tileElems *= d;
          tileElemsExpr = std::to_string(tileElems);
        }

        unsigned id = nextId++;
        std::string ptrName = getName(globalVal);
        std::string elemTy = emitElementType(globalTy.getElementType());

        // Compute linearized offset using global tensor strides.
        auto globalShape = globalTy.getShape();
        llvm::SmallVector<int64_t> globalStrides(globalShape.size());
        {
          int64_t s = 1;
          for (int i = (int)globalShape.size() - 1; i >= 0; --i) {
            globalStrides[i] = s;
            if (!mlir::ShapedType::isDynamic(globalShape[i]))
              s *= globalShape[i];
          }
        }
        std::string offExpr;
        if (offsets.size() == 1) {
          offExpr = getName(offsets[0]);
          if (tileElemsExpr != "1")
            offExpr += " * " + tileElemsExpr;
        } else {
          for (unsigned i = 0; i < offsets.size(); ++i) {
            std::string term = getName(offsets[i]);
            if (globalStrides[i] != 1)
              term += " * " + std::to_string(globalStrides[i]);
            if (i == 0) offExpr = term;
            else offExpr += " + " + term;
          }
        }

        std::string slicedPtr = "__slice_ptr_" + std::to_string(id);
        os() << getIndent() << elemTy << "* " << slicedPtr << " = ("
           << elemTy << "*)" << ptrName << " + " << offExpr << ";\n";

        auto copyShapeAttr = invokeOp.getCopyShapeAttr();
        llvm::ArrayRef<int64_t> copyShape =
            copyShapeAttr ? copyShapeAttr.asArrayRef() : localTy.getShape();
        bool hasDynCopy = llvm::any_of(copyShape, mlir::ShapedType::isDynamic);

        std::string srcTensor, dstTensor;
        std::string suffix = "_" + std::to_string(id);
        if (hasDynCopy) {
          auto dynDims = invokeOp.getDynDims();
          std::string shapeName = "__dma_shape" + suffix;
          os() << getIndent() << "auto " << shapeName << " = cute::make_shape(";
          unsigned dynIdx = 0;
          for (unsigned i = 0; i < copyShape.size(); ++i) {
            if (i > 0) os() << ", ";
            if (mlir::ShapedType::isDynamic(copyShape[i])) {
              if (dynIdx < dynDims.size())
                os() << "(int)" << getName(dynDims[dynIdx]);
              else
                os() << "1";
              dynIdx++;
            } else {
              os() << "cute::Int<" << copyShape[i] << ">{}";
            }
          }
          os() << ");\n";
          std::string layoutName = "__dma_layout" + suffix;
          os() << getIndent() << "auto " << layoutName
             << " = cute::make_layout(" << shapeName << ");\n";

          std::string elemTySrc = emitElementType(globalTy.getElementType());
          if (desc.isLoad) {
            srcTensor = "__dma_src" + suffix;
            os() << getIndent() << "auto " << srcTensor
               << " = cute::make_tensor(cute::make_gmem_ptr<" << elemTySrc
               << ">((" << elemTySrc << "*)" << slicedPtr << "), "
               << layoutName << ");\n";
            dstTensor = "__dma_dst" + suffix;
            std::string elemTyDst = emitElementType(localTy.getElementType());
            os() << getIndent() << "auto " << dstTensor
               << " = cute::make_tensor(cute::make_smem_ptr(("
               << elemTyDst << "*)" << getName(localVal) << "), "
               << layoutName << ");\n";
          } else {
            srcTensor = "__dma_src" + suffix;
            std::string elemTyS = emitElementType(localTy.getElementType());
            os() << getIndent() << "auto " << srcTensor
               << " = cute::make_tensor(cute::make_smem_ptr(("
               << elemTyS << "*)" << getName(localVal) << "), "
               << layoutName << ");\n";
            dstTensor = "__dma_dst" + suffix;
            os() << getIndent() << "auto " << dstTensor
               << " = cute::make_tensor(cute::make_gmem_ptr<" << elemTySrc
               << ">((" << elemTySrc << "*)" << slicedPtr << "), "
               << layoutName << ");\n";
          }
        } else if (desc.isLoad) {
          srcTensor = emitCuteTensorFromPtr(slicedPtr, globalTy,
                                            copyShape, suffix + "_s",
                                            enclosingKernel);
          dstTensor = emitCuteTensor(localVal, suffix + "_d");
        } else {
          srcTensor = emitCuteTensor(localVal, suffix + "_s");
          dstTensor = emitCuteTensorFromPtr(slicedPtr, globalTy,
                                            copyShape, suffix + "_d",
                                            enclosingKernel);
        }
        os() << getIndent() << "choreo::naive_copy(" << srcTensor << ", "
           << dstTensor << ");\n";
      } else {
        emitNaiveCopy(constOp.getSource(), constOp.getDest());
      }
      os() << getIndent() << "__syncthreads();\n";
    } else {
      os() << getIndent() << "// cp.async DMA (unknown source)\n";
      os() << getIndent() << "__syncthreads();\n";
    }
  }

  // --- Fallback handlers for un-lowered copy ops (when LowerDMADesc skips) ---

  void emitDmaCopy(DmaCopyOp op) override {
    emitNaiveCopy(op.getSource(), op.getDest());
    os() << getIndent() << "__syncthreads();\n";
    std::string name = getName(op.getToken());
    os() << getIndent() << "int " << name << " = 0;\n";
    dmaTokens.insert(op.getToken());
  }

  void emitTmaCopy(TmaCopyOp op) {
    emitNaiveCopy(op.getSource(), op.getDest());
    os() << getIndent() << "__syncthreads();\n";
    std::string name = getName(op.getToken());
    os() << getIndent() << "int " << name << " = 0;\n";
    dmaTokens.insert(op.getToken());
  }

  void emitElementCopy(ElementCopyOp op) {
    emitNaiveCopy(op.getSource(), op.getDest());
  }

  void emitBarrier(BarrierOp op) override {
    auto scope = op.getScope();
    if (scope == ParallelLevel::BLOCK)
      os() << getIndent() << "__syncthreads();\n";
    else
      os() << getIndent() << "// barrier scope="
         << stringifyParallelLevel(scope) << "\n";
  }

  void emitWait(WaitOp op) override {
    auto token = op.getToken();
    if (dmaTokens.count(token)) {
      return;
    }
    os() << getIndent() << "__syncthreads();\n";
  }

  void emitAsyncUndef(AsyncUndefOp op) override {
    std::string name = getName(op.getResult());
    os() << getIndent() << "int " << name << " = 0;\n";
    dmaTokens.insert(op.getResult());
  }

  void emitFutureRotate(FutureRotateOp op) override {
    for (auto res : op.getResults()) {
      std::string name = getName(res);
      os() << getIndent() << "int " << name << " = 0;\n";
      dmaTokens.insert(res);
    }
  }

  void emitTensorTile(TensorTileOp op) override {
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

    TileLayout layout;
    layout.shape.assign(perIdxTileSize.begin(), perIdxTileSize.end());
    layout.strides.assign(srcStrides.begin(), srcStrides.end());
    tileLayouts[op.getResult()] = std::move(layout);

    os() << getIndent() << "auto " << name << " = " << getName(op.getSource());
    os() << " + (";
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) os() << " + ";
      os() << getName(indices[i]);
      int64_t stride = perIdxTileSize[i];
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      os() << " * " << stride;
    }
    os() << ");\n";
  }

  void emitAtomic(AtomicOp op) override {
    using AK = coir::AtomicKind;
    llvm::StringRef fnName;
    switch (op.getKind()) {
    case AK::Add:  fnName = "atomicAdd"; break;
    case AK::Sub:  fnName = "atomicSub"; break;
    case AK::Exch: fnName = "atomicExch"; break;
    case AK::Min:  fnName = "atomicMin"; break;
    case AK::Max:  fnName = "atomicMax"; break;
    case AK::And:  fnName = "atomicAnd"; break;
    case AK::Or:   fnName = "atomicOr"; break;
    case AK::Xor:  fnName = "atomicXor"; break;
    case AK::CAS:  fnName = "atomicCAS"; break;
    }
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    os() << getIndent() << fnName << "(&" << getName(op.getDest()) << "[";
    emitLinearIndex(op.getIndices(), tty);
    os() << "], " << getName(op.getValue());
    if (op.getKind() == AK::CAS && op.getCompare())
      os() << ", " << "/* compare */";
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

struct EmitCUDAPass : public ::coir::impl::EmitCUDABase<EmitCUDAPass> {
  using EmitCUDABase::EmitCUDABase;

  void runOnOperation() override {
    auto module = getOperation();
    CUDAEmitter emitter;
    emitter.emitModule(module, llvm::outs());
  }
};

static bool registered_gpu = [] {
  CoIR::CodeGenRegistry::Register("cute", [] {
    return std::make_unique<CUDAEmitter>();
  });
  return true;
}();

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitCUDAPass() {
  return std::make_unique<EmitCUDAPass>();
}

void emitCUDA(mlir::ModuleOp module, llvm::raw_ostream &os) {
  CUDAEmitter emitter;
  emitter.emitModule(module, os);
}
} // namespace coir
