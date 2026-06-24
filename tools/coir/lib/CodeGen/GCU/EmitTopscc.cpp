// EmitTopscc -- emit topscc C++ from CoIR IR for GCU target.

#include "EmitTopscc.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <set>

using namespace mlir;
using namespace coir;

namespace {

class TopsccEmitter {
public:
  TopsccEmitter(llvm::raw_ostream &os) : os(os), indent(0) {}

  void emitModule(ModuleOp module) {
    // Read target arch for per-arch DTE type selection.
    archStr = CoIR::GetArch(module).str();

    // Pre-scan to detect MMA ops so we can include the acore header early.
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op)) {
        kernel.walk([&](Operation *inner) {
          if (isa<MMAFillOp, MMALoadOp, MMAExecOp, MMAStoreOp>(inner))
            hasAcoreCall = true;
        });
      }
    }

    emitHeader();
    if (hasAcoreCall)
      os << "#include <common/acore_op.h>\n\n";

    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        if (hasBlockParallel(kernel))
          emitDeviceFunction(kernel);
    }

    if (!stubCode.empty())
      os << stubCode;

    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitHostFunction(kernel);
    }
  }

private:
  llvm::raw_ostream &os;
  unsigned indent;
  std::string archStr;
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
      return emitType(tensorTy.getElementType()) + "*";
    if (ty.isIndex()) return "int";
    if (ty.isInteger(1)) return "bool";
    if (ty.isF16()) return "__fp16";
    if (ty.isBF16()) return "__bf16";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (ty.isInteger(8)) return "int8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int";
    if (ty.isInteger(64)) return "int64_t";
    return "/* unknown */";
  }

  std::string choreoType(Type ty) {
    if (ty.isF16()) return "choreo::f16";
    if (ty.isBF16()) return "choreo::bf16";
    if (ty.isF32()) return "choreo::f32";
    if (ty.isF64()) return "choreo::f64";
    if (ty.isInteger(8)) return "choreo::s8";
    if (ty.isInteger(16)) return "choreo::s16";
    if (ty.isInteger(32)) return "choreo::s32";
    if (ty.isInteger(64)) return "choreo::s64";
    return "choreo::s32";
  }

  // Portable DTE context type that works across all target architectures.
  // tops_dte_ctx_t is universally available; the choreo.h header typedefs
  // it to the arch-appropriate underlying type when compiled with topscc.
  std::string getDTEType() const { return "tops_dte_ctx_t"; }

  // tops_dte_ctx_t requires explicit .init() on legacy targets (gcu210).
  // On gcu300/400 the type is RAII and init() is a no-op, so always calling
  // it is safe and keeps generated code portable.
  bool needsExplicitInit() const { return true; }

  void emitHeader() {
    os << "#include <stdint.h>\n";
    os << "#include <tops.h>\n";
    os << "#include \"tops/tops_runtime.h\"\n";
    os << "#include \"choreo.h\"\n\n";
  }

  void preCollectStubs(KernelOp kernel) {
    kernel.walk([&](MMAExecOp exec) {
      auto lhsFragTy = cast<coir::MMAFragType>(exec.getLhs().getType());
      auto accFragTy =
          cast<coir::MMAFragType>(exec.getAccumulator().getType());
      int64_t M = lhsFragTy.getShape()[0];
      // Look through mma.load -> tensor.tile to find original M.
      if (auto loadOp = exec.getLhs().getDefiningOp<MMALoadOp>()) {
        if (auto origTy = getOriginalTensorType(loadOp.getSource()))
          M = origTy.getShape()[0];
      }
      getOrEmitStub(M, lhsFragTy.getElementType(),
                    accFragTy.getElementType(), exec.getLayout());
    });
  }

  // Check if kernel return value i is an input argument (return-input pattern).
  int getReturnInputArgIdx(KernelOp kernel, unsigned retIdx) {
    auto &body = kernel.getBody();
    if (body.empty()) return -1;
    for (auto &op : body.front().getOperations()) {
      if (auto ret = dyn_cast<KernelReturnOp>(op)) {
        if (retIdx < ret.getOperands().size()) {
          Value v = ret.getOperands()[retIdx];
          if (auto arg = dyn_cast<BlockArgument>(v))
            return arg.getArgNumber();
        }
      }
    }
    return -1;
  }

  bool hasBlockParallel(KernelOp kernel) {
    bool found = false;
    kernel.walk([&](ParallelOp p) {
      if (p.getLevel() == ParallelLevel::BLOCK) found = true;
    });
    return found;
  }

  void emitDeviceFunction(KernelOp kernel) {
    preCollectStubs(kernel);

    if (!stubDeclCode.empty()) {
      os << stubDeclCode;
      stubDeclCode.clear();
    }

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
      int argIdx = getReturnInputArgIdx(kernel, i);
      if (argIdx >= 0) {
        returnParamNames[i] = "arg" + std::to_string(argIdx);
      } else {
        if (paramIdx > 0) os << ", ";
        std::string name = "out" + std::to_string(i);
        os << emitType(fnType.getResult(i)) << " " << name;
        returnParamNames[i] = name;
        paramIdx++;
      }
    }
    os << ") {\n";
    incIndent();

    for (auto &op : body.front().getOperations()) {
      if (auto ret = dyn_cast<KernelReturnOp>(op)) {
        for (unsigned i = 0; i < ret.getOperands().size(); ++i) {
          returnValues.insert(ret.getOperands()[i]);
          if (getReturnInputArgIdx(kernel, i) < 0)
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
    if (eTy.isF16() || eTy.isBF16() || eTy.isInteger(16)) elemSize = 2;
    else if (eTy.isF64() || eTy.isInteger(64)) elemSize = 8;
    else if (eTy.isInteger(8)) elemSize = 1;
    return n * elemSize;
  }

  int64_t getTensorNumElems(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    return n;
  }

  int64_t getGridDim(KernelOp kernel) {
    auto &body = kernel.getBody();
    if (body.empty()) return 1;
    for (auto &op : body.front().getOperations()) {
      auto parallel = dyn_cast<ParallelOp>(op);
      if (!parallel) continue;
      if (parallel.getLevel() != ParallelLevel::BLOCK) continue;
      auto bounds = parallel.getBounds();
      if (bounds.empty()) continue;
      int64_t grid = 1;
      for (auto b : bounds) grid *= b;
      return grid;
    }
    return 1;
  }

  std::string hostReturnType(FunctionType fnType) {
    if (fnType.getNumResults() == 0) return "void";
    Type resTy = fnType.getResult(0);
    if (auto tty = dyn_cast<coir::TensorType>(resTy)) {
      return "choreo::spanned_data<" +
             choreoType(tty.getElementType()) + ", " +
             std::to_string(tty.getShape().size()) + ">";
    }
    return emitType(resTy);
  }

  void emitHostFunction(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();
    bool needsDevice = hasBlockParallel(kernel);
    auto resTy = numResults > 0
                     ? dyn_cast<coir::TensorType>(fnType.getResult(0))
                     : nullptr;

    // __global__ trampoline -- only when device offload is needed and
    // the result is a tensor requiring D2H copy.
    if (needsDevice && resTy) {
      int retInputIdx = getReturnInputArgIdx(kernel, 0);
      std::string eType = emitType(resTy.getElementType());

      os << "__global__ void __coir_global_" << name.str() << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << emitType(fnType.getInput(i)) << " g_in" << i;
      }
      if (retInputIdx < 0)
        os << ", " << eType << "* g_out, int N";
      os << ") {\n";
      os << "  " << name.str() << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << "g_in" << i;
      }
      if (retInputIdx < 0)
        os << ", g_out";
      os << ");\n";
      os << "}\n\n";
    }

    // Host function signature.
    os << hostReturnType(fnType) << " " << name.str() << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      auto inTy = fnType.getInput(i);
      if (auto tensorTy = dyn_cast<coir::TensorType>(inTy)) {
        unsigned inDim = tensorTy.getShape().size();
        os << "const choreo::spanned_view<"
           << choreoType(tensorTy.getElementType()) << ", "
           << inDim << "> & p" << i;
      } else {
        os << emitType(inTy) << " p" << i;
      }
    }
    os << ") {\n";

    if (needsDevice && resTy) {
      emitDeviceOffloadBody(kernel, resTy);
    } else {
      // No device offload: emit the body directly (the function IS the
      // host function, just like how the AST codegen handles __co__
      // functions without parallel-by).
      auto &body = kernel.getBody();
      if (!body.empty()) {
        incIndent();
        for (auto &op : body.front().getOperations())
          emitOp(&op);
        decIndent();
      }
    }

    os << "}\n\n";
  }

  bool isDeviceGlobal(coir::TensorType tty) {
    return tty.getMemorySpace() ==
           static_cast<int32_t>(coir::TensorMemorySpace::Global);
  }

  void emitDeviceOffloadBody(KernelOp kernel, coir::TensorType resTy) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    int retInputIdx = getReturnInputArgIdx(kernel, 0);
    std::string eType = emitType(resTy.getElementType());
    std::string choreoElem = choreoType(resTy.getElementType());
    unsigned ndim = resTy.getShape().size();
    int64_t resN = getTensorNumElems(resTy);
    int64_t resBytes = getTensorBytes(resTy);
    int64_t gridDim = getGridDim(kernel);

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) {
        std::string inEType = emitType(tty.getElementType());
        os << "  " << inEType << "* p" << i
           << "__device = const_cast<" << inEType << "*>(p" << i
           << ".data());\n";
      } else {
        int64_t bytes = tty ? getTensorBytes(tty) : resBytes;
        std::string inEType = tty ? emitType(tty.getElementType()) : eType;
        os << "  " << inEType << "* p" << i << "__device = nullptr;\n";
        os << "  topsMalloc((void**)&p" << i << "__device, "
           << bytes << "ULL);\n";
        os << "  topsMemcpy(p" << i << "__device, p" << i << ".data(), "
           << bytes << "ULL, topsMemcpyHostToDevice);\n";
      }
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

    if (retInputIdx >= 0) {
      os << "  __coir_global_" << name.str() << "<<<" << gridDim << ", 1>>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << "p" << i << "__device";
      }
      os << ");\n";
      os << "  topsDeviceSynchronize();\n";
      os << "  topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
         << ".data()), p" << retInputIdx << "__device, "
         << resBytes << "ULL, topsMemcpyDeviceToHost);\n";
      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os << "  topsFree(p" << i << "__device);\n";
      }
      os << "  return choreo::copy_as_spanned(p" << retInputIdx
         << ".data(), p" << retInputIdx << ".shape());\n";
    } else {
      os << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
         << ndim << ">(" << shapeStr << ");\n";
      os << "  " << eType << "* __result__device = nullptr;\n";
      os << "  topsMalloc((void**)&__result__device, " << resBytes
         << "ULL);\n";
      os << "  __coir_global_" << name.str() << "<<<" << gridDim
         << ", 1>>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << "p" << i << "__device";
      }
      os << ", __result__device, " << resN << ");\n";
      os << "  topsDeviceSynchronize();\n";
      os << "  topsMemcpy(__result.data(), __result__device, "
         << resBytes << "ULL, topsMemcpyDeviceToHost);\n";
      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os << "  topsFree(p" << i << "__device);\n";
      }
      os << "  topsFree(__result__device);\n";
      os << "  return __result;\n";
    }
  }

  unsigned nextDmaId = 0;
  DenseMap<Value, std::string> dmaCtxNames;
  DenseMap<Value, std::string> asyncFutures;

  bool hasAsyncUses(Value asyncHandle) {
    return asyncHandle && !asyncHandle.use_empty();
  }

  void emitOp(Operation *op) {
    if (auto parallel = dyn_cast<ParallelOp>(op))
      emitParallel(parallel);
    else if (auto foreach_ = dyn_cast<ForeachOp>(op))
      emitForeach(foreach_);
    else if (auto dataCopy = dyn_cast<DataCopyOp>(op))
      emitDataCopy(dataCopy);
    else if (auto dmaCopy = dyn_cast<DmaCopyOp>(op))
      emitDmaCopy(dmaCopy);
    else if (auto constDesc = dyn_cast<DMAConstDescOp>(op))
      emitDMAConstDesc(constDesc);
    else if (auto prefetch = dyn_cast<DMADescPrefetchOp>(op))
      emitDMAPrefetch(prefetch);
    else if (auto rtDesc = dyn_cast<DMADescRuntimeOp>(op))
      emitDMARuntimeDesc(rtDesc);
    else if (auto invoke = dyn_cast<DMAInvokeOp>(op))
      emitDMAInvoke(invoke);
    else if (auto loadElem = dyn_cast<TensorLoadElemOp>(op))
      emitLoadElem(loadElem);
    else if (auto storeElem = dyn_cast<TensorStoreElemOp>(op))
      emitStoreElem(storeElem);
    else if (auto alloc = dyn_cast<TensorAllocOp>(op))
      emitAlloc(alloc);
    else if (auto tile = dyn_cast<TensorTileOp>(op))
      emitTensorTile(tile);
    else if (auto fill = dyn_cast<MMAFillOp>(op))
      emitMMAFill(fill);
    else if (auto load = dyn_cast<MMALoadOp>(op))
      emitMMALoad(load);
    else if (auto exec = dyn_cast<MMAExecOp>(op))
      emitMMAExec(exec);
    else if (auto store = dyn_cast<MMAStoreOp>(op))
      emitMMAStore(store);
    else if (auto barrier = dyn_cast<BarrierOp>(op))
      emitBarrier(barrier);
    else if (auto wait = dyn_cast<WaitOp>(op))
      emitWait(wait);
    else if (auto rotate = dyn_cast<FutureRotateOp>(op))
      emitFutureRotate(rotate);
    else if (auto ret = dyn_cast<KernelReturnOp>(op))
      emitKernelReturn(ret);
    else if (auto yield = dyn_cast<YieldOp>(op))
      emitYield(yield);
    else if (auto check = dyn_cast<DMACheckOp>(op))
      (void)check;
    else if (auto constOp = dyn_cast<arith::ConstantOp>(op))
      emitConstant(constOp);
    else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op))
      emitIfOp(ifOp);
    else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op))
      emitWhileOp(whileOp);
    else if (auto coirWhile = dyn_cast<coir::CoIRWhileOp>(op))
      emitCoIRWhileOp(coirWhile);
    else if (isa<coir::CoIRWhileCondOp>(op))
      {}
    else if (auto breakOp = dyn_cast<coir::CoIRBreakOp>(op))
      emitBreak(breakOp);
    else if (auto contOp = dyn_cast<coir::CoIRContinueOp>(op))
      emitContinue(contOp);
    else if (isa<mlir::scf::YieldOp>(op) || isa<mlir::scf::ConditionOp>(op))
      {}
    else if (auto indexCast = dyn_cast<arith::IndexCastOp>(op))
      valueNames[indexCast.getResult()] = getName(indexCast.getIn());
    else if (auto selectOp = dyn_cast<arith::SelectOp>(op))
      emitSelect(selectOp);
    else if (emitArithBinOp(op)) {}
    else if (emitCmpOp(op)) {}
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

    auto iterArgs = op.getIterArgs();
    for (unsigned i = 0; i < iterArgs.size(); ++i) {
      std::string iterName = getName(args[i + 1]);
      os << getIndent() << "auto " << iterName << " = "
         << getName(iterArgs[i]) << ";\n";
      auto stateIt = acoreStates.find(iterArgs[i]);
      if (stateIt != acoreStates.end()) {
        acoreStates[args[i + 1]] = stateIt->second;
        auto &st = stateIt->second;
        os << getIndent() << "void* " << st.ws_name << "_last_lhs;\n";
        os << getIndent() << "void* " << st.ws_name << "_last_rhs;\n";
      }
    }

    os << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
       << ub << "; ++" << iv << ") {\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}\n";

    for (unsigned i = 0; i < op.getResults().size(); ++i) {
      valueNames[op.getResult(i)] = getName(args[i + 1]);
      auto stateIt = acoreStates.find(args[i + 1]);
      if (stateIt != acoreStates.end())
        acoreStates[op.getResult(i)] = stateIt->second;
    }
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

    os << getIndent() << "auto " << name << " = " << getName(op.getSource());
    if (srcTy && !indices.empty()) {
      os << " + (";
      auto srcShape = srcTy.getShape();
      auto tileShape = tileTy ? tileTy.getShape() : llvm::ArrayRef<int64_t>{};
      for (unsigned i = 0; i < indices.size(); ++i) {
        if (i > 0) os << " + ";
        os << getName(indices[i]);
        int64_t tileDim = (i < tileShape.size()) ? tileShape[i] : 1;
        os << " * " << tileDim;
        for (unsigned j = i + 1; j < srcShape.size(); ++j)
          os << " * " << srcShape[j];
      }
      os << ")";
    }
    os << ";\n";
  }

  // -- MMA emission for GCU (acore::matmul micro-kernel backend) -----------
  //
  // GCU MMA does not use fragment objects. Instead, acore::matmul operates
  // on raw buffer pointers and keeps accumulation state in VACC hardware
  // registers, controlled by acc_flag/store_flag arguments.
  //
  // The deferred exec+store pattern: mma.exec records parameters as pending.
  // When mma.store is seen, the pending exec is emitted with store_flag=1.
  // In a K-loop, each new mma.exec flushes the previous one with store_flag=0.

  struct AcoreAccumState {
    bool first_exec = true;
    std::string ws_name;
    std::string pending_lhs;
    std::string pending_rhs;
    std::string pending_k_dim;
    std::string pending_n_dim;
    std::string stub_name;
    int acc_flag = 0;
    int lt_flag = 0;
    bool has_pending = false;
  };

  DenseMap<Value, AcoreAccumState> acoreStates;
  DenseMap<Value, std::string> mmaFragAddrs;
  std::set<std::string> emittedStubs;
  std::string stubCode;
  std::string stubDeclCode;
  bool hasAcoreCall = false;

  std::string acorePtrType(Type elemTy) {
    if (elemTy.isF16()) return "__fp16";
    if (elemTy.isBF16()) return "__bf16";
    if (elemTy.isF32()) return "float";
    if (elemTy.isInteger(8)) return "char";
    return "void";
  }

  std::string acoreTypeTag(Type elemTy) {
    if (elemTy.isF16()) return "f16";
    if (elemTy.isBF16()) return "bf16";
    if (elemTy.isF32()) return "f32";
    if (elemTy.isInteger(8)) return "s8";
    return "unk";
  }

  std::string stubSignature(const std::string &name,
                            const std::string &outPtr,
                            const std::string &inPtr) {
    return "__device__ void " + name + "(\n"
      "    " + outPtr + "* __restrict__ out,\n"
      "    " + inPtr + "* __restrict__ lhs,\n"
      "    " + inPtr + "* __restrict__ rhs,\n"
      "    int* __restrict__ ws,\n"
      "    int K, int N, int acc, int store, int vab_off, int lt)";
  }

  std::string getOrEmitStub(int64_t M, Type inElemTy, Type outElemTy,
                            coir::MMALayout layout) {
    std::string inPtr = acorePtrType(inElemTy);
    std::string outPtr = acorePtrType(outElemTy);
    std::string fmtStr = (layout == coir::MMALayout::RowCol) ? "MK_KN"
                                                              : "MK_NK";
    std::string name = "__choreo_mma_" + acoreTypeTag(inElemTy) + "_" +
                       acoreTypeTag(outElemTy) + "_M" + std::to_string(M) +
                       "_" + fmtStr;
    if (emittedStubs.count(name)) return name;
    emittedStubs.insert(name);

    std::string sig = stubSignature(name, outPtr, inPtr);
    stubDeclCode += sig + ";\n\n";

    std::string acoreFmt = (layout == coir::MMALayout::RowCol)
                               ? "acore::MK_KN"
                               : "acore::MK_NK";
    llvm::raw_string_ostream s(stubCode);
    s << sig << " {\n"
      << "  acore::matmul<" << M << ", " << acoreFmt << ">(\n"
      << "      out, lhs, rhs, (" << inPtr << "*)nullptr, ws,\n"
      << "      K, N, acc, store, 0, vab_off, lt);\n"
      << "}\n\n";
    return name;
  }

  void emitMMAFill(MMAFillOp op) {
    std::string name = getName(op.getResult());

    AcoreAccumState state;
    state.ws_name = "__mma_ws_" + name;

    os << getIndent() << "int " << state.ws_name << "[2048];\n";
    acoreStates[op.getResult()] = state;
  }

  DenseMap<Value, Value> mmaLoadSources;

  void emitMMALoad(MMALoadOp op) {
    // Look through tensor.tile to find the base buffer address.
    // acore::matmul operates on the full buffer, not a tiled view.
    Value src = op.getSource();
    mmaLoadSources[op.getResult()] = src;
    while (auto tileOp = src.getDefiningOp<TensorTileOp>())
      src = tileOp.getSource();
    mmaFragAddrs[op.getResult()] = getName(src);
  }

  // Walk through tensor.tile chain to find the original untiled tensor.
  // For micro-kernel MMA, acore::matmul needs the full tensor dimensions.
  coir::TensorType getOriginalTensorType(Value v) {
    while (auto tileOp = v.getDefiningOp<TensorTileOp>())
      v = tileOp.getSource();
    if (auto tty = dyn_cast<coir::TensorType>(v.getType()))
      return tty;
    return {};
  }

  bool isInsideForeach(Operation *op) {
    return op->getParentOfType<ForeachOp>() != nullptr;
  }

  std::string getForeachIV(Operation *op) {
    auto foreach_ = op->getParentOfType<ForeachOp>();
    if (!foreach_) return "";
    return getName(foreach_.getBody().front().getArgument(0));
  }

  std::string getForeachUB(Operation *op) {
    auto foreach_ = op->getParentOfType<ForeachOp>();
    if (!foreach_) return "";
    return getName(foreach_.getUpperBound());
  }

  void emitMMAExec(MMAExecOp op) {
    Value accVal = op.getAccumulator();
    auto lhsFragTy = cast<coir::MMAFragType>(op.getLhs().getType());
    auto accFragTy = cast<coir::MMAFragType>(op.getAccumulator().getType());
    auto layout = op.getLayout();

    // For micro-kernel targets, read M/K/N from the original (pre-tile)
    // tensor so that acore::matmul gets the full data dimensions.
    // The hardware micro-kernel handles internal tiling.
    int64_t M = lhsFragTy.getShape()[0];
    int64_t K = lhsFragTy.getShape()[1];
    int64_t N = 0;
    auto rhsFragTy = cast<coir::MMAFragType>(op.getRhs().getType());
    if (layout == coir::MMALayout::RowCol)
      N = rhsFragTy.getShape()[1];
    else
      N = rhsFragTy.getShape()[0];

    auto lhsSrcIt = mmaLoadSources.find(op.getLhs());
    auto rhsSrcIt = mmaLoadSources.find(op.getRhs());
    if (lhsSrcIt != mmaLoadSources.end()) {
      if (auto origTy = getOriginalTensorType(lhsSrcIt->second)) {
        M = origTy.getShape()[0];
        K = origTy.getShape()[1];
      }
    }
    if (rhsSrcIt != mmaLoadSources.end()) {
      if (auto origTy = getOriginalTensorType(rhsSrcIt->second)) {
        if (layout == coir::MMALayout::RowCol)
          N = origTy.getShape()[1];
        else
          N = origTy.getShape()[0];
      }
    }

    Type inElemTy = lhsFragTy.getElementType();
    Type outElemTy = accFragTy.getElementType();

    std::string stub = getOrEmitStub(M, inElemTy, outElemTy, layout);
    hasAcoreCall = true;

    std::string lhsAddr = mmaFragAddrs.count(op.getLhs())
                              ? mmaFragAddrs[op.getLhs()]
                              : getName(op.getLhs());
    std::string rhsAddr = mmaFragAddrs.count(op.getRhs())
                              ? mmaFragAddrs[op.getRhs()]
                              : getName(op.getRhs());

    auto it = acoreStates.find(accVal);
    if (it == acoreStates.end()) {
      AcoreAccumState fresh;
      fresh.ws_name = "__mma_ws_" + getName(accVal);
      os << getIndent() << "int " << fresh.ws_name << "[2048];\n";
      acoreStates[accVal] = fresh;
      it = acoreStates.find(accVal);
    }
    auto &state = it->second;

    std::string inPtr = acorePtrType(inElemTy);
    std::string outPtr = acorePtrType(outElemTy);

    bool inLoop = isInsideForeach(op);
    if (inLoop) {
      // Inside a K-loop: emit exec for all iterations except the last.
      // The last iteration's exec is deferred to mma.store which will
      // emit it with store_flag=1 (compute + writeback in one call).
      std::string iv = getForeachIV(op);
      std::string ub = getForeachUB(op);

      std::string savedLhs = state.ws_name + "_last_lhs";
      std::string savedRhs = state.ws_name + "_last_rhs";
      os << getIndent() << savedLhs << " = (void*)" << lhsAddr << ";\n";
      os << getIndent() << savedRhs << " = (void*)" << rhsAddr << ";\n";

      os << getIndent() << "if (" << iv << " < " << ub << " - 1) {\n";
      incIndent();
      os << getIndent() << stub << "(\n"
         << getIndent() << "    (" << outPtr << "*)nullptr,\n"
         << getIndent() << "    (" << inPtr << "*)" << lhsAddr << ",\n"
         << getIndent() << "    (" << inPtr << "*)" << rhsAddr << ",\n"
         << getIndent() << "    " << state.ws_name << ",\n"
         << getIndent() << "    " << K << ", " << N << ", "
         << "(" << iv << " > 0 ? 1 : 0)"
         << ", 0, 0, (" << iv << " > 0 ? 1 : 0));\n";
      decIndent();
      os << getIndent() << "}\n";

      state.pending_lhs = savedLhs;
      state.pending_rhs = savedRhs;
      state.pending_k_dim = std::to_string(K);
      state.pending_n_dim = std::to_string(N);
      state.stub_name = stub;
      state.acc_flag = 1;
      state.lt_flag = 1;
      state.has_pending = true;
      state.first_exec = false;
    } else {
      // Outside loops: use deferred exec+store pattern.
      // mma.exec just records pending; mma.store emits the call.
      state.pending_lhs = lhsAddr;
      state.pending_rhs = rhsAddr;
      state.pending_k_dim = std::to_string(K);
      state.pending_n_dim = std::to_string(N);
      state.stub_name = stub;
      state.acc_flag = state.first_exec ? 0 : 1;
      state.lt_flag = state.first_exec ? 0 : 1;
      state.has_pending = true;
      state.first_exec = false;
    }

    valueNames[op.getResult()] = getName(accVal);
    acoreStates[op.getResult()] = state;
  }

  void emitMMAStore(MMAStoreOp op) {
    Value fragVal = op.getFragment();
    // Look through tensor.tile to get the base address of the original
    // tensor -- acore::matmul writes to the full output buffer.
    Value destVal = op.getDest();
    while (auto tileOp = destVal.getDefiningOp<TensorTileOp>())
      destVal = tileOp.getSource();
    std::string destAddr = getName(destVal);

    auto it = acoreStates.find(fragVal);
    if (it == acoreStates.end()) {
      os << getIndent() << "// [error] mma.store without prior exec\n";
      return;
    }
    auto &state = it->second;
    if (!state.has_pending) {
      os << getIndent() << "// [error] mma.store: no pending exec\n";
      return;
    }

    auto fragTy = cast<coir::MMAFragType>(fragVal.getType());
    Type outElemTy = fragTy.getElementType();

    // Find the input element type from the exec op chain.
    Type inElemTy = outElemTy;
    if (auto execOp = fragVal.getDefiningOp<MMAExecOp>()) {
      auto lhsFrag = cast<coir::MMAFragType>(execOp.getLhs().getType());
      inElemTy = lhsFrag.getElementType();
    } else if (auto foreachOp = fragVal.getDefiningOp<ForeachOp>()) {
      // Result comes from a foreach loop; walk into the yield to find
      // the exec op.
      auto &body = foreachOp.getBody().front();
      for (auto &bodyOp : body.getOperations()) {
        if (auto execOp = dyn_cast<MMAExecOp>(bodyOp)) {
          auto lhsFrag = cast<coir::MMAFragType>(execOp.getLhs().getType());
          inElemTy = lhsFrag.getElementType();
          break;
        }
      }
    }

    std::string inPtr = acorePtrType(inElemTy);
    std::string outPtr = acorePtrType(outElemTy);

    os << getIndent() << state.stub_name << "(\n"
       << getIndent() << "    (" << outPtr << "*)" << destAddr << ",\n"
       << getIndent() << "    (" << inPtr << "*)" << state.pending_lhs
       << ",\n"
       << getIndent() << "    (" << inPtr << "*)" << state.pending_rhs
       << ",\n"
       << getIndent() << "    " << state.ws_name << ",\n"
       << getIndent() << "    " << state.pending_k_dim << ", "
       << state.pending_n_dim << ", " << state.acc_flag
       << ", 1, 0, " << state.lt_flag << ");\n";

    state.has_pending = false;
  }

  void emitKernelReturn(KernelReturnOp op) {
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      Value val = op.getOperands()[i];
      if (isa<coir::TensorType>(val.getType())) continue;
      os << getIndent() << "return " << getName(val) << ";\n";
    }
  }

  void emitYield(YieldOp op) {
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      auto yieldVal = op.getOperands()[i];
      auto it = acoreStates.find(yieldVal);
      if (it != acoreStates.end()) {
        auto parentForeach = op->getParentOfType<ForeachOp>();
        if (parentForeach) {
          auto iterArgs = parentForeach.getBody().front().getArguments();
          if (i + 1 < iterArgs.size())
            acoreStates[iterArgs[i + 1]] = it->second;
        }
      }
    }
  }

  std::string emitMdspanWithShape(Value tensor) {
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string name = getName(tensor);
    std::string space;
    int32_t ms = tty.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      space = "tops::Private";
    else if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      space = "tops::Shared";
    else
      space = "tops::Global";
    std::string result = "tops::mdspan(" + space + ", (" +
                         emitType(tty.getElementType()) + "*)" + name;
    for (auto dim : tty.getShape())
      result += ", " + std::to_string(dim);
    result += ")";
    return result;
  }

  std::string emitCopyMdspan(Value tensor, int64_t transferElems) {
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string name = getName(tensor);
    std::string space;
    int32_t ms = tty.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      space = "tops::Private";
    else if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      space = "tops::Shared";
    else
      space = "tops::Global";
    return "tops::mdspan(" + space + ", (" +
           emitType(tty.getElementType()) + "*)" + name + ", " +
           std::to_string(transferElems) + ")";
  }

  int64_t tensorElems(Value v) {
    auto tty = cast<coir::TensorType>(v.getType());
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    return n;
  }

  void emitDataCopy(DataCopyOp op) {
    auto kind = op.getKind();
    bool useIndividualShapes = (kind != coir::DMAKind::Copy);
    int64_t srcElems = tensorElems(op.getSource());
    int64_t dstElems = tensorElems(op.getDest());
    int64_t copyElems = std::max(srcElems, dstElems);
    std::string srcMds = useIndividualShapes
        ? emitMdspanWithShape(op.getSource())
        : emitCopyMdspan(op.getSource(), copyElems);
    std::string dstMds = useIndividualShapes
        ? emitMdspanWithShape(op.getDest())
        : emitCopyMdspan(op.getDest(), copyElems);
    bool isAsync = op.getAsync() && hasAsyncUses(op.getToken());
    unsigned id = nextDmaId++;
    std::string ctxName = "__dte_" + std::to_string(id);
    std::string futName = "__fut_" + std::to_string(id);

    os << getIndent() << getDTEType() << " " << ctxName << ";\n";
    os << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_" << id << "\", 0, 0);\n";

    std::string evName = futName + "__event__";

    if (kind == coir::DMAKind::Pad) {
      emitPadArraysFromDataCopy(op, futName);
      std::string padValStr = emitPadValueFromDataCopy(op);
      std::string api = isAsync ? "tops::pad_async" : "tops::pad";
      os << getIndent();
      if (isAsync) os << "tops::event " << evName << " = ";
      os << api << "(*" << futName << ".get_ctx(), "
         << dstMds << ", " << srcMds << ", "
         << futName << "__pad_low__, " << futName << "__pad_high__, "
         << futName << "__pad_mid__, " << padValStr << ");\n";
    } else if (kind == coir::DMAKind::Transpose) {
      emitTransposeLayoutFromDataCopy(op, futName);
      std::string api = isAsync ? "tops::transpose_async" : "tops::transpose";
      os << getIndent();
      if (isAsync) os << "tops::event " << evName << " = ";
      os << api << "(*" << futName << ".get_ctx(), "
         << dstMds << ", " << srcMds << ", "
         << futName << "__layout__);\n";
    } else {
      std::string api = isAsync ? "tops::memcpy_async" : "tops::memcpy";
      os << getIndent();
      if (isAsync) os << "tops::event " << evName << " = ";
      os << api << "(*" << futName << ".get_ctx(), "
         << dstMds << ", " << srcMds << ");\n";
    }

    if (isAsync) {
      os << getIndent() << futName << ".set_event(" << evName << ");\n";
      asyncFutures[op.getToken()] = futName;
    } else {
      os << getIndent() << futName << ".set_nowait();\n";
    }
  }

  void emitPadArraysFromDataCopy(DataCopyOp op, const std::string &futName) {
    auto emitArr = [&](const char *suffix,
                       std::optional<ArrayRef<int64_t>> arr, int rank) {
      os << getIndent() << "unsigned int " << futName << suffix << "[] = {";
      if (arr) {
        for (int i = 0; i < (int)arr->size(); ++i) {
          if (i) os << ", ";
          os << (*arr)[i];
        }
      } else {
        for (int i = 0; i < rank; ++i) {
          if (i) os << ", ";
          os << "0";
        }
      }
      os << "};\n";
    };
    auto srcTy = cast<coir::TensorType>(op.getSource().getType());
    int rank = srcTy.getShape().size();
    emitArr("__pad_low__", op.getPadLow(), rank);
    emitArr("__pad_high__", op.getPadHigh(), rank);
    emitArr("__pad_mid__", std::nullopt, rank);
  }

  std::string emitPadValueFromDataCopy(DataCopyOp op) {
    if (auto intAttr = op.getPadValueAttr().dyn_cast_or_null<IntegerAttr>())
      return std::to_string(intAttr.getInt());
    if (auto fpAttr = op.getPadValueAttr().dyn_cast_or_null<FloatAttr>())
      return std::to_string(fpAttr.getValueAsDouble());
    return "0";
  }

  void emitTransposeLayoutFromDataCopy(DataCopyOp op,
                                        const std::string &futName) {
    os << getIndent() << "int " << futName << "__layout__[] = {";
    if (auto perm = op.getTransposePerm()) {
      for (int i = 0; i < (int)perm->size(); ++i) {
        if (i) os << ", ";
        os << (*perm)[i];
      }
    }
    os << "};\n";
  }

  void emitDmaCopy(DmaCopyOp op) {
    auto kind = op.getKind();
    bool useIndividualShapes = (kind != coir::DMAKind::Copy);
    int64_t srcElems = tensorElems(op.getSource());
    int64_t dstElems = tensorElems(op.getDest());
    int64_t copyElems = std::max(srcElems, dstElems);
    std::string srcMds = useIndividualShapes
        ? emitMdspanWithShape(op.getSource())
        : emitCopyMdspan(op.getSource(), copyElems);
    std::string dstMds = useIndividualShapes
        ? emitMdspanWithShape(op.getDest())
        : emitCopyMdspan(op.getDest(), copyElems);
    bool isAsync = hasAsyncUses(op.getToken());
    unsigned id = nextDmaId++;
    std::string ctxName = "__dte_" + std::to_string(id);
    std::string futName = "__fut_" + std::to_string(id);

    os << getIndent() << getDTEType() << " " << ctxName << ";\n";
    os << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_" << id << "\", 0, 0);\n";

    std::string evName = futName + "__event__";

    if (kind == coir::DMAKind::Pad) {
      emitPadArrays(op, futName);
      std::string padValStr = emitPadValue(op);
      if (isAsync) {
        os << getIndent() << "tops::event " << evName
           << " = tops::pad_async(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ", "
           << futName << "__pad_low__, " << futName << "__pad_high__, "
           << futName << "__pad_mid__, " << padValStr << ");\n";
        os << getIndent() << futName << ".set_event(" << evName << ");\n";
      } else {
        os << getIndent() << "tops::pad(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ", "
           << futName << "__pad_low__, " << futName << "__pad_high__, "
           << futName << "__pad_mid__, " << padValStr << ");\n";
        os << getIndent() << futName << ".set_nowait();\n";
      }
    } else if (kind == coir::DMAKind::Transpose) {
      emitTransposeLayout(op, futName);
      if (isAsync) {
        os << getIndent() << "tops::event " << evName
           << " = tops::transpose_async(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ", "
           << futName << "__layout__);\n";
        os << getIndent() << futName << ".set_event(" << evName << ");\n";
      } else {
        os << getIndent() << "tops::transpose(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ", "
           << futName << "__layout__);\n";
        os << getIndent() << futName << ".set_nowait();\n";
      }
    } else {
      if (isAsync) {
        os << getIndent() << "tops::event " << evName
           << " = tops::memcpy_async(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ");\n";
        os << getIndent() << futName << ".set_event(" << evName << ");\n";
      } else {
        os << getIndent() << "tops::memcpy(*" << futName << ".get_ctx(), "
           << dstMds << ", " << srcMds << ");\n";
        os << getIndent() << futName << ".set_nowait();\n";
      }
    }

    if (isAsync)
      asyncFutures[op.getToken()] = futName;
  }

  void emitPadArrays(DmaCopyOp op, const std::string &futName) {
    auto emitArr = [&](const char *suffix,
                       std::optional<ArrayRef<int64_t>> arr, int rank) {
      os << getIndent() << "unsigned int " << futName << suffix << "[] = {";
      if (arr) {
        for (int i = 0; i < (int)arr->size(); ++i) {
          if (i) os << ", ";
          os << (*arr)[i];
        }
      } else {
        for (int i = 0; i < rank; ++i) {
          if (i) os << ", ";
          os << "0";
        }
      }
      os << "};\n";
    };
    auto srcTy = cast<coir::TensorType>(op.getSource().getType());
    int rank = srcTy.getShape().size();
    emitArr("__pad_low__", op.getPadLow(), rank);
    emitArr("__pad_high__", op.getPadHigh(), rank);
    emitArr("__pad_mid__", std::nullopt, rank);
  }

  std::string emitPadValue(DmaCopyOp op) {
    if (auto intAttr = op.getPadValueAttr().dyn_cast_or_null<IntegerAttr>())
      return std::to_string(intAttr.getInt());
    if (auto fpAttr = op.getPadValueAttr().dyn_cast_or_null<FloatAttr>())
      return std::to_string(fpAttr.getValueAsDouble());
    return "0";
  }

  void emitTransposeLayout(DmaCopyOp op, const std::string &futName) {
    os << getIndent() << "int " << futName << "__layout__[] = {";
    if (auto perm = op.getTransposePerm()) {
      for (int i = 0; i < (int)perm->size(); ++i) {
        if (i) os << ", ";
        os << (*perm)[i];
      }
    }
    os << "};\n";
  }

  void emitWait(WaitOp op) {
    auto it = asyncFutures.find(op.getToken());
    if (it != asyncFutures.end())
      os << getIndent() << it->second << ".wait();\n";
  }

  void emitFutureRotate(FutureRotateOp op) {
    auto inputs = op.getFutures();
    auto outputs = op.getResults();
    SmallVector<std::string> names;
    for (auto in : inputs) {
      auto it = asyncFutures.find(in);
      names.push_back(it != asyncFutures.end() ? it->second : "?");
    }
    os << getIndent() << "choreo::rotate(";
    for (unsigned i = 0; i < names.size(); ++i) {
      if (i) os << ", ";
      os << names[i];
    }
    os << ");\n";
    // Left-rotate the map: output[i] gets the future name of input[(i+1) % n]
    for (unsigned i = 0; i < names.size(); ++i)
      asyncFutures[outputs[i]] = names[(i + 1) % names.size()];
  }

  std::string emitMdspan(Value tensor) {
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string name = getName(tensor);
    auto shape = tty.getShape();
    std::string space;
    int32_t ms = tty.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      space = "tops::Private";
    else if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      space = "tops::Shared";
    else
      space = "tops::Global";

    std::string result = "tops::mdspan(" + space + ", ("
      + emitType(tty.getElementType()) + "*)" + name;
    for (auto d : shape)
      result += ", " + std::to_string(d);
    result += ")";
    return result;
  }

  void emitDMAConstDesc(DMAConstDescOp op) {
    unsigned id = nextDmaId++;
    std::string ctxName = "__dma_" + std::to_string(id);
    std::string futName = "__fut_desc_" + std::to_string(id);
    dmaCtxNames[op.getOut()] = futName;

    os << getIndent() << getDTEType() << " " << ctxName << ";\n";
    os << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_desc_" << id << "\", 0, 0);\n";

    std::string srcMds = emitMdspan(op.getSource());
    std::string dstMds = emitMdspan(op.getDest());

    auto kind = op.getKind();
    if (kind == coir::DMAKind::Copy) {
      os << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Slice) {
      os << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Transpose) {
      os << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Pad) {
      os << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    }
  }

  void emitDMAPrefetch(DMADescPrefetchOp op) {
    auto it = dmaCtxNames.find(op.getIn());
    if (it != dmaCtxNames.end())
      dmaCtxNames[op.getOut()] = it->second;
  }

  void emitDMARuntimeDesc(DMADescRuntimeOp op) {
    auto it = dmaCtxNames.find(op.getIn());
    std::string futName = (it != dmaCtxNames.end()) ? it->second : "__dma_?";
    dmaCtxNames[op.getOut()] = futName;

    auto offsets = op.getOffsets();
    for (unsigned i = 0; i < offsets.size(); ++i) {
      os << getIndent() << futName << ".set_offset(" << i << ", "
         << getName(offsets[i]) << ");\n";
    }
  }

  void emitDMAInvoke(DMAInvokeOp op) {
    auto it = dmaCtxNames.find(op.getDesc());
    std::string ctxName = (it != dmaCtxNames.end()) ? it->second : "__dma_?";

    if (hasAsyncUses(op.getDone())) {
      os << getIndent() << ctxName << ".trigger_only();\n";
      asyncFutures[op.getDone()] = ctxName;
    } else {
      os << getIndent() << ctxName << ".trigger_and_wait();\n";
    }
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

    os << getIndent() << qualifier << emitType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";
  }

  void emitBarrier(BarrierOp) {
    os << getIndent() << "tcle::sync();\n";
  }

  void emitIfOp(mlir::scf::IfOp op) {
    for (auto res : op.getResults())
      os << getIndent() << emitType(res.getType()) << " "
         << getName(res) << ";\n";
    os << getIndent() << "if (" << getName(op.getCondition()) << ") {\n";
    incIndent();
    for (auto &bodyOp : op.getThenRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i)
          os << getIndent() << getName(op.getResult(i)) << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
      } else {
        emitOp(&bodyOp);
      }
    }
    decIndent();
    os << getIndent() << "}\n";
    if (!op.getElseRegion().empty()) {
      os << getIndent() << "else {\n";
      incIndent();
      for (auto &bodyOp : op.getElseRegion().front().getOperations()) {
        if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
          for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i)
            os << getIndent() << getName(op.getResult(i)) << " = "
               << getName(yieldOp.getOperand(i)) << ";\n";
        } else {
          emitOp(&bodyOp);
        }
      }
      decIndent();
      os << getIndent() << "}\n";
    }
  }

  void emitWhileOp(mlir::scf::WhileOp op) {
    auto &beforeBlock = op.getBefore().front();
    auto condOp = dyn_cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());
    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i) {
      std::string name = "wv" + std::to_string(nextId++);
      iterVarNames.push_back(name);
      os << getIndent() << emitType(op.getInits()[i].getType()) << " "
         << name << " = " << getName(op.getInits()[i]) << ";\n";
      valueNames[beforeBlock.getArgument(i)] = name;
    }
    // Emit condition computation and save the condition variable name.
    for (auto &bodyOp : beforeBlock.getOperations()) {
      if (isa<mlir::scf::ConditionOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }
    std::string condName = getName(condOp.getCondition());
    os << getIndent() << "while (" << condName << ") {\n";
    incIndent();
    auto &afterBlock = op.getAfter().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[afterBlock.getArgument(i)] = getName(condOp.getArgs()[i]);
    for (auto &bodyOp : afterBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
          valueNames[beforeBlock.getArgument(i)] = iterVarNames[i];
        }
        // Re-emit condition ops using assignment (not redeclaration).
        for (auto &bOp : beforeBlock.getOperations()) {
          if (isa<mlir::scf::ConditionOp>(&bOp)) continue;
          emitReassignOp(&bOp);
        }
      } else {
        emitOp(&bodyOp);
      }
    }
    decIndent();
    os << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      valueNames[op.getResult(i)] = iterVarNames[i];
  }

  void emitCoIRWhileOp(coir::CoIRWhileOp op) {
    auto &condBlock = op.getCondRegion().front();
    auto condOp = dyn_cast<coir::CoIRWhileCondOp>(condBlock.getTerminator());
    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i) {
      std::string name = "wv" + std::to_string(nextId++);
      iterVarNames.push_back(name);
      os << getIndent() << emitType(op.getInits()[i].getType()) << " "
         << name << " = " << getName(op.getInits()[i]) << ";\n";
      valueNames[condBlock.getArgument(i)] = name;
    }
    for (auto &bodyOp : condBlock.getOperations()) {
      if (isa<coir::CoIRWhileCondOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }
    os << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
    incIndent();
    auto &bodyBlock = op.getBodyRegion().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[bodyBlock.getArgument(i)] = getName(condOp.getArgs()[i]);
    for (auto &bodyOp : bodyBlock.getOperations()) {
      if (auto breakOp = dyn_cast<coir::CoIRBreakOp>(&bodyOp)) {
        for (unsigned i = 0; i < breakOp.getOperands().size(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(breakOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] = iterVarNames[i];
        }
        os << getIndent() << "break;\n";
      } else if (auto contOp = dyn_cast<coir::CoIRContinueOp>(&bodyOp)) {
        for (unsigned i = 0; i < contOp.getOperands().size(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(contOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] = iterVarNames[i];
        }
        for (auto &cOp : condBlock.getOperations()) {
          if (isa<coir::CoIRWhileCondOp>(&cOp)) continue;
          emitReassignOp(&cOp);
        }
        os << getIndent() << "continue;\n";
      } else {
        emitOp(&bodyOp);
      }
    }
    decIndent();
    os << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      valueNames[op.getResult(i)] = iterVarNames[i];
  }

  void emitBreak(coir::CoIRBreakOp) {
    os << getIndent() << "break;\n";
  }

  void emitContinue(coir::CoIRContinueOp) {
    os << getIndent() << "continue;\n";
  }

  void emitSelect(arith::SelectOp op) {
    std::string name = getName(op.getResult());
    os << getIndent() << emitType(op.getResult().getType()) << " "
       << name << " = " << getName(op.getCondition()) << " ? "
       << getName(op.getTrueValue()) << " : "
       << getName(op.getFalseValue()) << ";\n";
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

  // Re-emit an op as assignment (for while-loop condition re-computation).
  // Instead of declaring a new variable, assigns to the existing name.
  void emitReassignOp(Operation *op) {
    if (op->getNumResults() == 0) { emitOp(op); return; }
    auto result = op->getResult(0);
    auto it = valueNames.find(result);
    if (it == valueNames.end()) { emitOp(op); return; }
    std::string existingName = it->second;
    if (auto cmpI = dyn_cast<arith::CmpIOp>(op)) {
      std::string lhs = getName(cmpI.getLhs());
      std::string rhs = getName(cmpI.getRhs());
      llvm::StringRef opStr;
      switch (cmpI.getPredicate()) {
      case arith::CmpIPredicate::eq:  opStr = "=="; break;
      case arith::CmpIPredicate::ne:  opStr = "!="; break;
      case arith::CmpIPredicate::slt: case arith::CmpIPredicate::ult:
        opStr = "<"; break;
      case arith::CmpIPredicate::sle: case arith::CmpIPredicate::ule:
        opStr = "<="; break;
      case arith::CmpIPredicate::sgt: case arith::CmpIPredicate::ugt:
        opStr = ">"; break;
      case arith::CmpIPredicate::sge: case arith::CmpIPredicate::uge:
        opStr = ">="; break;
      }
      os << getIndent() << existingName << " = (" << lhs << " " << opStr
         << " " << rhs << ");\n";
    } else if (auto cmpF = dyn_cast<arith::CmpFOp>(op)) {
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
      os << getIndent() << existingName << " = (" << lhs << " " << opStr
         << " " << rhs << ");\n";
    } else {
      emitOp(op);
    }
  }

  void emitConstant(arith::ConstantOp op) {
    std::string name = getName(op.getResult());
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue()))
      os << getIndent() << "const int " << name << " = "
         << intAttr.getInt() << ";\n";
    else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      llvm::SmallString<16> strVal;
      floatAttr.getValue().toString(strVal, 6, 0);
      os << getIndent() << "const " << emitType(op.getType())
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

struct EmitTopsccPass
    : public PassWrapper<EmitTopsccPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitTopsccPass)
  StringRef getArgument() const override { return "coir-emit-topscc"; }
  StringRef getDescription() const override {
    return "Emit topscc/GCU C++ source from CoIR IR";
  }
  void runOnOperation() override {
    auto module = getOperation();
    TopsccEmitter emitter(llvm::outs());
    emitter.emitModule(module);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitTopsccPass() {
  return std::make_unique<EmitTopsccPass>();
}

void emitTopscc(mlir::ModuleOp module, llvm::raw_ostream &os) {
  TopsccEmitter emitter(os);
  emitter.emitModule(module);
}
} // namespace coir

static mlir::PassRegistration<EmitTopsccPass> reg;

namespace {

void emitHostCode(llvm::raw_ostream &os, ModuleOp module) {
  auto hostCodeAttr = module->getAttrOfType<StringAttr>("coir.host_code");
  if (!hostCodeAttr) return;
  os << "\n" << hostCodeAttr.getValue() << "\n";
}

class TopsccTargetEmitter : public CoIR::Emitter {
public:
  void EmitScript(mlir::ModuleOp module, llvm::raw_ostream &os) override {
    auto &sctx = CoIR::ScriptContext::Get();

    os << "#!/usr/bin/env bash\n";
    os << "# CoIR generated script -- compile and execute topscc kernel\n";
    os << "set -eo pipefail\n\n";

    os << "TMPDIR=$(mktemp -d /tmp/cocc_XXXXXX)\n";
    os << "trap 'rm -rf $TMPDIR' EXIT\n\n";

    if (sctx.types_header) {
      os << "cat > \"$TMPDIR/choreo_types.h\" << '__COCC_TYPES_HEADER__'\n";
      os << sctx.types_header;
      os << "\n__COCC_TYPES_HEADER__\n\n";
    }
    if (sctx.runtime_header) {
      os << "cat > \"$TMPDIR/choreo.h\" << '__COCC_CHOREO_HEADER__'\n";
      os << sctx.runtime_header;
      os << "\n__COCC_CHOREO_HEADER__\n\n";
    }

    if (!sctx.target_setup.empty()) os << sctx.target_setup << "\n";
    if (!sctx.build_env.empty()) os << sctx.build_env;

    os << "TOPSCC=\"${TOPSCC:-topscc}\"\n";
    os << "if ! command -v \"$TOPSCC\" &>/dev/null; then\n";
    os << "  echo \"Error: topscc not found\"; exit 1\n";
    os << "fi\n\n";

    os << "TMPFILE=\"$TMPDIR/kernel.cc\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COIR_TOPSCC_SOURCE__'\n";

    coir::emitTopscc(module, os);
    emitHostCode(os, module);

    os << "\n__COIR_TOPSCC_SOURCE__\n\n";
    os << "\"$TOPSCC\" ${CFLAGS} -D__CHOREO_DMA_DIAGNOSIS__"
          " -I\"$TMPDIR\" -I\"$TMPDIR/topscc\" "
          "-o \"$BINFILE\" \"$TMPFILE\" -lpthread -ldl -lrt 2>&1\n";
    os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
    os << "  shift\n";
    os << "  \"$BINFILE\" \"$@\"\n";
    os << "fi\n";
  }

  void EmitSource(mlir::ModuleOp module, llvm::raw_ostream &os) override {
    coir::emitTopscc(module, os);
    emitHostCode(os, module);
  }
};

static bool registered_topscc = [] {
  CoIR::EmitterRegistry::Register("topscc", [] {
    return std::make_unique<TopsccTargetEmitter>();
  });
  CoIR::EmitterRegistry::Register("gcu", [] {
    return std::make_unique<TopsccTargetEmitter>();
  });
  return true;
}();

} // namespace
