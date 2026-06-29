// EmitTopscc -- emit topscc C++ from CoIR IR for GCU target.

#include "EmitTopscc.h"
#include "Dialect/CoIR/CoIRDialect.h"
#include "Dialect/CoIR/CoIROps.h"
#include "Dialect/CoIR/CoIRTypes.h"
#include "Dialect/CoIR/CoIRAttrs.h"
#include "Dialect/CoIR/Passes.h"
#include "CodeGen/CoIREmitterBase.h"

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

class TopsccEmitter : public coir::CoIREmitterBase {
public:
  TopsccEmitter() = default;

  void emitModule(ModuleOp module, llvm::raw_ostream &out) override {
    os_ = &out;
    resetState();
    // Read target arch for per-arch DTE type selection.
    archStr = CoIR::GetArch(module).str();
    archNum = parseArchNum(archStr);

    // Pre-scan to detect MMA ops or acore calls for auto-include.
    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op)) {
        kernel.walk([&](Operation *inner) {
          if (isa<MMAFillOp, MMALoadOp, MMAExecOp, MMAStoreOp>(inner))
            hasAcoreCall = true;
          if (auto callOp = dyn_cast<CallOp>(inner)) {
            auto callee = callOp.getCallee().str();
            if (callee.find("acore::") == 0 ||
                (callOp.getIsLibCall() && *callOp.getIsLibCall()))
              hasAcoreCall = true;
          }
        });
      }
    }

    emitHeader();
    if (hasAcoreCall)
      os() << "#include <common/acore_op.h>\n\n";

    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        if (hasBlockParallel(kernel))
          emitDeviceFunction(kernel);
    }

    if (!stubCode.empty())
      os() << stubCode;

    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        emitHostEntry(kernel);
    }
  }

  int EmitScript(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    emitScriptPrologue(os, "compile and execute topscc kernel");

    os << "TOPSCC=\"${TOPSCC:-topscc}\"\n";
    os << "if ! command -v \"$TOPSCC\" &>/dev/null; then\n";
    os << "  echo \"Error: topscc not found\"; exit 1\n";
    os << "fi\n\n";

    os << "TMPFILE=\"$TMPDIR/kernel.cc\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COIR_TOPSCC_SOURCE__'\n";
    EmitSource(module, "", os);

    os << "\n__COIR_TOPSCC_SOURCE__\n\n";
    os << "\"$TOPSCC\" ${CFLAGS} -D__CHOREO_DMA_DIAGNOSIS__"
          " -I\"$TMPDIR\" -I\"$TMPDIR/topscc\" "
          "-o \"$BINFILE\" \"$TMPFILE\" -lpthread -ldl -lrt 2>&1\n";
    emitScriptExecuteBlock(os);
    return 0;
  }


private:
  std::string archStr;
  int archNum = 0;

  bool hasGroupLevel() const { return archNum >= 400; }

  int parseArchNum(llvm::StringRef arch) {
    auto s = arch.str();
    auto it = s.rbegin();
    while (it != s.rend() && !std::isdigit(*it)) ++it;
    auto numEnd = it;
    while (it != s.rend() && std::isdigit(*it)) ++it;
    if (it == numEnd) return 0;
    return std::stoi(std::string(it.base(), numEnd.base()));
  }

  std::string emitType(Type ty) override {
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
    os() << "#include <stdint.h>\n";
    os() << "#include <tops.h>\n";
    os() << "#include \"tops/tops_runtime.h\"\n";
    os() << "#include \"choreo.h\"\n\n";
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

  bool hasDeviceParallel(KernelOp kernel) {
    bool found = false;
    kernel.walk([&](ParallelOp p) {
      if (p.getLevel() == ParallelLevel::DEVICE) found = true;
    });
    return found;
  }

  int64_t getDeviceBound(KernelOp kernel) {
    int64_t bound = 1;
    kernel.walk([&](ParallelOp p) {
      if (p.getLevel() == ParallelLevel::DEVICE) {
        auto bounds = p.getBounds();
        for (auto b : bounds) bound *= b;
      }
    });
    return bound;
  }

  struct LaunchConfig {
    SmallVector<int64_t> blockDims;  // BLOCK bounds -> gridDim
    SmallVector<int64_t> groupDims;  // GROUP bounds -> blockDim (gcu400+)
    SmallVector<int64_t> threadDims; // THREAD bounds -> blockDim (gcu300) or
                                     // __thread_dims__ (gcu400+)
  };

  LaunchConfig collectLaunchConfig(KernelOp kernel) {
    LaunchConfig lc;
    kernel.walk([&](ParallelOp p) {
      auto bounds = p.getBounds();
      switch (p.getLevel()) {
      case ParallelLevel::BLOCK:
        for (auto b : bounds) lc.blockDims.push_back(b);
        break;
      case ParallelLevel::GROUP:
        for (auto b : bounds) lc.groupDims.push_back(b);
        break;
      case ParallelLevel::THREAD:
        for (auto b : bounds) lc.threadDims.push_back(b);
        break;
      default:
        break;
      }
    });
    return lc;
  }

  std::string emitDim3(ArrayRef<int64_t> dims) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    auto d = [&](unsigned i) -> int64_t {
      return i < dims.size() ? dims[i] : 1;
    };
    ss << "dim3(" << d(0) << ", " << d(1) << ", " << d(2) << ")";
    return s;
  }

  void emitDeviceFunction(KernelOp kernel) {
    preCollectStubs(kernel);
    bool isMultiDevice = hasDeviceParallel(kernel);

    if (!stubDeclCode.empty()) {
      os() << stubDeclCode;
      stubDeclCode.clear();
    }

    auto fnType = kernel.getFunctionType();

    if (hasGroupLevel()) {
      auto lc = collectLaunchConfig(kernel);
      auto td = [&](unsigned i) -> int64_t {
        return i < lc.threadDims.size() ? lc.threadDims[i] : 1;
      };
      os() << "__thread_dims__(" << td(0) << ", " << td(1) << ", " << td(2)
         << ")\n";
    }

    os() << "__device__ void " << kernel.getSymName() << "(";

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
      int argIdx = getReturnInputArgIdx(kernel, i);
      if (argIdx >= 0) {
        returnParamNames[i] = "arg" + std::to_string(argIdx);
      } else {
        if (paramIdx > 0) os() << ", ";
        std::string name = "out" + std::to_string(i);
        os() << emitType(fnType.getResult(i)) << " " << name;
        returnParamNames[i] = name;
        paramIdx++;
      }
    }
    if (isMultiDevice) {
      if (paramIdx > 0) os() << ", ";
      os() << "int __device_id";
    }
    os() << ") {\n";
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
    os() << "}\n\n";
  }

  int64_t getTensorNumElems(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    return n;
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

  void emitHostEntry(KernelOp kernel) override {
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
      bool isMultiDevice = hasDeviceParallel(kernel);

      if (hasGroupLevel()) {
        auto lc = collectLaunchConfig(kernel);
        auto td = [&](unsigned i) -> int64_t {
          return i < lc.threadDims.size() ? lc.threadDims[i] : 1;
        };
        os() << "__thread_dims__(" << td(0) << ", " << td(1) << ", " << td(2)
           << ")\n";
      }
      os() << "__global__ void __coir_global_" << name.str() << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << emitType(fnType.getInput(i)) << " g_in" << i;
      }
      if (retInputIdx < 0)
        os() << ", " << eType << "* g_out, int N";
      if (isMultiDevice)
        os() << ", int __device_id";
      os() << ") {\n";
      os() << "  " << name.str() << "(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "g_in" << i;
      }
      if (retInputIdx < 0)
        os() << ", g_out";
      if (isMultiDevice)
        os() << ", __device_id";
      os() << ");\n";
      os() << "}\n\n";
    }

    // Host function signature.
    os() << hostReturnType(fnType) << " " << name.str() << "(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os() << ", ";
      auto inTy = fnType.getInput(i);
      if (auto tensorTy = dyn_cast<coir::TensorType>(inTy)) {
        unsigned inDim = tensorTy.getShape().size();
        os() << "const choreo::spanned_view<"
           << choreoType(tensorTy.getElementType()) << ", "
           << inDim << "> & p" << i;
      } else {
        os() << emitType(inTy) << " p" << i;
      }
    }
    os() << ") {\n";

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

    os() << "}\n\n";
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
    auto lc = collectLaunchConfig(kernel);
    std::string gdims = emitDim3(lc.blockDims);
    std::string bdims = hasGroupLevel()
                            ? emitDim3(lc.groupDims)
                            : emitDim3(lc.threadDims);
    bool isMultiDevice = hasDeviceParallel(kernel);
    int64_t devCount = isMultiDevice ? getDeviceBound(kernel) : 1;

    if (isMultiDevice) {
      emitMultiDeviceOffloadBody(kernel, resTy, gdims, bdims, devCount);
      return;
    }

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) {
        std::string inEType = emitType(tty.getElementType());
        os() << "  " << inEType << "* p" << i
           << "__device = const_cast<" << inEType << "*>(p" << i
           << ".data());\n";
      } else {
        int64_t bytes = tty ? getTensorBytes(tty) : resBytes;
        std::string inEType = tty ? emitType(tty.getElementType()) : eType;
        os() << "  " << inEType << "* p" << i << "__device = nullptr;\n";
        os() << "  topsMalloc((void**)&p" << i << "__device, "
           << bytes << "ULL);\n";
        os() << "  topsMemcpy(p" << i << "__device, p" << i << ".data(), "
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
      os() << "  __coir_global_" << name.str() << "<<<" << gdims << ", "
         << bdims << ">>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "p" << i << "__device";
      }
      os() << ");\n";
      os() << "  topsDeviceSynchronize();\n";
      os() << "  topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
         << ".data()), p" << retInputIdx << "__device, "
         << resBytes << "ULL, topsMemcpyDeviceToHost);\n";
      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os() << "  topsFree(p" << i << "__device);\n";
      }
      os() << "  return choreo::copy_as_spanned(p" << retInputIdx
         << ".data(), p" << retInputIdx << ".shape());\n";
    } else {
      os() << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
         << ndim << ">(" << shapeStr << ");\n";
      os() << "  " << eType << "* __result__device = nullptr;\n";
      os() << "  topsMalloc((void**)&__result__device, " << resBytes
         << "ULL);\n";
      os() << "  __coir_global_" << name.str() << "<<<" << gdims << ", "
         << bdims << ">>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "p" << i << "__device";
      }
      os() << ", __result__device, " << resN << ");\n";
      os() << "  topsDeviceSynchronize();\n";
      os() << "  topsMemcpy(__result.data(), __result__device, "
         << resBytes << "ULL, topsMemcpyDeviceToHost);\n";
      for (unsigned i = 0; i < numInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os() << "  topsFree(p" << i << "__device);\n";
      }
      os() << "  topsFree(__result__device);\n";
      os() << "  return __result;\n";
    }
  }

  void emitMultiDeviceOffloadBody(KernelOp kernel, coir::TensorType resTy,
                                  const std::string &gdims,
                                  const std::string &bdims,
                                  int64_t devCount) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    int retInputIdx = getReturnInputArgIdx(kernel, 0);
    std::string eType = emitType(resTy.getElementType());
    std::string choreoElem = choreoType(resTy.getElementType());
    unsigned ndim = resTy.getShape().size();
    int64_t resN = getTensorNumElems(resTy);
    int64_t resBytes = getTensorBytes(resTy);
    std::string dc = std::to_string(devCount);

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

    // Per-device buffer vectors
    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) continue;
      std::string inEType = tty ? emitType(tty.getElementType()) : eType;
      os() << "  std::vector<" << inEType << "*> p" << i
         << "__device_vec(" << dc << ", nullptr);\n";
    }
    if (retInputIdx < 0) {
      os() << "  " << eType << "* __result_buf = (" << eType
         << "*)malloc(" << resBytes << "ULL);\n";
      os() << "  topsHostRegister(__result_buf, " << resBytes
         << "ULL, topsHostRegisterPortable);\n";
      os() << "  std::vector<" << eType << "*> __result__device_vec("
         << dc << ", nullptr);\n";
    }

    // Device loop: topsSetDevice + malloc + H2D + launch
    os() << "  for (int __d = 0; __d < " << dc << "; ++__d) {\n";
    os() << "    topsSetDevice(__d);\n";

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) continue;
      int64_t bytes = tty ? getTensorBytes(tty) : resBytes;
      os() << "    topsMalloc((void**)&p" << i << "__device_vec[__d], "
         << bytes << "ULL);\n";
      os() << "    topsMemcpy(p" << i << "__device_vec[__d], p" << i
         << ".data(), " << bytes << "ULL, topsMemcpyHostToDevice);\n";
    }

    if (retInputIdx < 0) {
      os() << "    topsMalloc((void**)&__result__device_vec[__d], "
         << resBytes << "ULL);\n";
    }

    // Kernel launch with device ID
    os() << "    __coir_global_" << name.str() << "<<<" << gdims << ", "
       << bdims << ">>>(";
    bool first = true;
    for (unsigned i = 0; i < numInputs; ++i) {
      if (!first) os() << ", ";
      first = false;
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty))
        os() << "const_cast<" << emitType(tty.getElementType())
           << "*>(p" << i << ".data())";
      else
        os() << "p" << i << "__device_vec[__d]";
    }
    if (retInputIdx < 0) {
      if (!first) os() << ", ";
      os() << "__result__device_vec[__d], " << resN;
    }
    os() << ", __d);\n";

    os() << "  }\n";

    // Sync loop: topsSetDevice + sync + partial D2H + free.
    // Each device's portion: total / device_count bytes at its own offset.
    int64_t portionElems = resN / devCount;
    int64_t portionBytes = resBytes / devCount;
    os() << "  for (int __sync_d = 0; __sync_d < " << dc << "; ++__sync_d) {\n";
    os() << "    topsSetDevice(__sync_d);\n";
    os() << "    topsDeviceSynchronize();\n";

    if (retInputIdx >= 0) {
      os() << "    topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
         << ".data()) + __sync_d * " << portionElems << ", p" << retInputIdx
         << "__device_vec[__sync_d] + __sync_d * " << portionElems << ", "
         << portionBytes << "ULL, topsMemcpyDeviceToHost);\n";
    } else {
      os() << "    topsMemcpy(__result_buf + __sync_d * " << portionElems
         << ", __result__device_vec[__sync_d] + __sync_d * " << portionElems
         << ", " << portionBytes
         << "ULL, topsMemcpyDeviceToHost);\n";
    }

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) continue;
      os() << "    topsFree(p" << i << "__device_vec[__sync_d]);\n";
    }
    if (retInputIdx < 0)
      os() << "    topsFree(__result__device_vec[__sync_d]);\n";

    os() << "  }\n";

    if (retInputIdx >= 0) {
      os() << "  return choreo::copy_as_spanned(p" << retInputIdx
         << ".data(), p" << retInputIdx << ".shape());\n";
    } else {
      os() << "  auto __result = choreo::copy_as_spanned<" << ndim
         << ">(__result_buf, " << shapeStr << ");\n";
      os() << "  topsHostUnregister(__result_buf);\n";
      os() << "  free(__result_buf);\n";
      os() << "  return __result;\n";
    }
  }

  unsigned nextDmaId = 0;
  DenseMap<Value, std::string> dmaCtxNames;
  DenseMap<Value, std::string> asyncFutures;

  bool hasAsyncUses(Value asyncHandle) {
    return asyncHandle && !asyncHandle.use_empty();
  }

  void emitOpFallback(Operation *op) override {
    if (auto callOp = dyn_cast<coir::CallOp>(op))
      emitCall(callOp);
    else if (auto ithOp = dyn_cast<coir::InThreadsOp>(op))
      emitInThreads(ithOp);
    else if (auto evtTrig = dyn_cast<EventTriggerOp>(op))
      emitEventTrigger(evtTrig);
    else if (auto evtWait = dyn_cast<EventWaitOp>(op))
      emitEventWait(evtWait);
    else
      CoIREmitterBase::emitOpFallback(op);
  }

  void emitParallel(ParallelOp op) override {
    auto level = op.getLevel();
    auto bounds = op.getBounds();
    auto &body = op.getBody();
    auto args = body.getArguments();
    const char *dimName[] = {"x", "y", "z"};

    os() << getIndent() << "// parallel level="
       << stringifyParallelLevel(level) << "\n";

    if (level == ParallelLevel::DEVICE) {
      for (unsigned i = 0; i < args.size(); ++i) {
        valueNames[args[i]] = "__device_id";
      }
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      return;
    }

    bool useHwId = false;
    std::string idPrefix;

    switch (level) {
    case ParallelLevel::BLOCK:
      useHwId = true;
      idPrefix = "__tops_bid_";
      break;
    case ParallelLevel::GROUP:
      useHwId = true;
      idPrefix = "__tops_tid_";
      break;
    case ParallelLevel::THREAD:
      useHwId = true;
      idPrefix = hasGroupLevel() ? "__tops_stid_" : "__tops_tid_";
      break;
    default:
      break;
    }

    if (useHwId) {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string name = "pid_" + std::to_string(nextId++);
        valueNames[args[i]] = name;
        unsigned dim = std::min(i, 2u);
        os() << getIndent() << "int " << name << " = " << idPrefix
           << dimName[dim] << "();  // bound=" << bounds[i] << "\n";
      }
      os() << getIndent() << "{\n";
      incIndent();
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      decIndent();
      os() << getIndent() << "}\n";
    } else {
      for (unsigned i = 0; i < args.size(); ++i) {
        std::string name = "pid_" + std::to_string(nextId++);
        valueNames[args[i]] = name;
        os() << getIndent() << "for (int " << name << " = 0; " << name
           << " < " << bounds[i] << "; ++" << name << ") {\n";
        incIndent();
      }
      for (auto &bodyOp : body.front().getOperations())
        emitOp(&bodyOp);
      for (unsigned i = 0; i < args.size(); ++i) {
        decIndent();
        os() << getIndent() << "}\n";
      }
    }
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
      auto stateIt = acoreStates.find(iterArgs[i]);
      if (stateIt != acoreStates.end()) {
        auto st = stateIt->second;
        acoreStates[args[i + 1]] = st;
        os() << getIndent() << "void* " << st.ws_name << "_last_lhs;\n";
        os() << getIndent() << "void* " << st.ws_name << "_last_rhs;\n";
      }
    }

    os() << getIndent() << "for (int " << iv << " = 0; " << iv << " < "
       << ub << "; ++" << iv << ") {\n";
    incIndent();
    for (auto &bodyOp : body.front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os() << getIndent() << "}\n";

    for (unsigned i = 0; i < op.getResults().size(); ++i) {
      valueNames[op.getResult(i)] = getName(args[i + 1]);
      auto stateIt = acoreStates.find(args[i + 1]);
      if (stateIt != acoreStates.end()) {
        auto st = stateIt->second;
        acoreStates[op.getResult(i)] = st;
      }
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

    os() << getIndent() << "auto " << name << " = " << getName(op.getSource());
    if (srcTy && !indices.empty()) {
      os() << " + (";
      auto srcShape = srcTy.getShape();
      auto tileShape = tileTy ? tileTy.getShape() : llvm::ArrayRef<int64_t>{};
      for (unsigned i = 0; i < indices.size(); ++i) {
        if (i > 0) os() << " + ";
        os() << getName(indices[i]);
        int64_t tileDim = (i < tileShape.size()) ? tileShape[i] : 1;
        os() << " * " << tileDim;
        for (unsigned j = i + 1; j < srcShape.size(); ++j)
          os() << " * " << srcShape[j];
      }
      os() << ")";
    }
    os() << ";\n";
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

  void emitMMAFill(MMAFillOp op) override {
    std::string name = getName(op.getResult());

    AcoreAccumState state;
    state.ws_name = "__mma_ws_" + name;

    os() << getIndent() << "int " << state.ws_name << "[2048];\n";
    acoreStates[op.getResult()] = state;
  }

  DenseMap<Value, Value> mmaLoadSources;

  void emitMMALoad(MMALoadOp op) override {
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

  void emitMMAExec(MMAExecOp op) override {
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

    if (acoreStates.find(accVal) == acoreStates.end()) {
      AcoreAccumState fresh;
      fresh.ws_name = "__mma_ws_" + getName(accVal);
      os() << getIndent() << "int " << fresh.ws_name << "[2048];\n";
      acoreStates[accVal] = fresh;
    }
    auto &state = acoreStates[accVal];

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
      os() << getIndent() << savedLhs << " = (void*)" << lhsAddr << ";\n";
      os() << getIndent() << savedRhs << " = (void*)" << rhsAddr << ";\n";

      os() << getIndent() << "if (" << iv << " < " << ub << " - 1) {\n";
      incIndent();
      os() << getIndent() << stub << "(\n"
         << getIndent() << "    (" << outPtr << "*)nullptr,\n"
         << getIndent() << "    (" << inPtr << "*)" << lhsAddr << ",\n"
         << getIndent() << "    (" << inPtr << "*)" << rhsAddr << ",\n"
         << getIndent() << "    " << state.ws_name << ",\n"
         << getIndent() << "    " << K << ", " << N << ", "
         << "(" << iv << " > 0 ? 1 : 0)"
         << ", 0, 0, (" << iv << " > 0 ? 1 : 0));\n";
      decIndent();
      os() << getIndent() << "}\n";

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
    AcoreAccumState stateCopy = state;
    acoreStates[op.getResult()] = stateCopy;
  }

  void emitMMAStore(MMAStoreOp op) override {
    Value fragVal = op.getFragment();
    // Look through tensor.tile to get the base address of the original
    // tensor -- acore::matmul writes to the full output buffer.
    Value destVal = op.getDest();
    while (auto tileOp = destVal.getDefiningOp<TensorTileOp>())
      destVal = tileOp.getSource();
    std::string destAddr = getName(destVal);

    auto it = acoreStates.find(fragVal);
    if (it == acoreStates.end()) {
      os() << getIndent() << "// [error] mma.store without prior exec\n";
      return;
    }
    auto &state = it->second;
    if (!state.has_pending) {
      os() << getIndent() << "// [error] mma.store: no pending exec\n";
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

    os() << getIndent() << state.stub_name << "(\n"
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

  void emitKernelReturn(KernelReturnOp op) override {
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      Value val = op.getOperands()[i];
      if (isa<coir::TensorType>(val.getType())) continue;
      os() << getIndent() << "return " << getName(val) << ";\n";
    }
  }

  void emitYield(YieldOp op) override {
    for (unsigned i = 0; i < op.getOperands().size(); ++i) {
      auto yieldVal = op.getOperands()[i];
      auto it = acoreStates.find(yieldVal);
      if (it != acoreStates.end()) {
        auto parentForeach = op->getParentOfType<ForeachOp>();
        if (parentForeach) {
          auto iterArgs = parentForeach.getBody().front().getArguments();
          if (i + 1 < iterArgs.size()) {
            auto st = it->second;
            acoreStates[iterArgs[i + 1]] = st;
          }
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

  TensorTileOp getTileDefiningOp(Value v) {
    return v.getDefiningOp<TensorTileOp>();
  }

  std::string emitFullBaseMdspan(TensorTileOp tile) {
    Value base = tile.getSource();
    auto baseTy = cast<coir::TensorType>(base.getType());
    std::string name = getName(base);
    std::string space;
    int32_t ms = baseTy.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      space = "tops::Private";
    else if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      space = "tops::Shared";
    else
      space = "tops::Global";
    std::string result = "tops::mdspan(" + space + ", (" +
                         emitType(baseTy.getElementType()) + "*)" + name;
    for (auto dim : baseTy.getShape())
      result += ", " + std::to_string(dim);
    result += ")";
    return result;
  }

  std::string emitSliceOffsets(TensorTileOp tile, const std::string &prefix) {
    auto tileTy = cast<coir::TensorType>(tile.getResult().getType());
    auto baseTy = cast<coir::TensorType>(tile.getSource().getType());
    auto tileShape = tileTy.getShape();
    auto baseShape = baseTy.getShape();
    auto indices = tile.getIndices();
    std::string arrName = prefix + "__off__";
    os() << getIndent() << "int " << arrName << "[] = {";
    for (unsigned i = 0; i < baseShape.size(); ++i) {
      if (i) os() << ", ";
      if (i < indices.size()) {
        int64_t chunkDim = (i < tileShape.size()) ? tileShape[i] : 1;
        os() << getName(indices[i]) << " * " << chunkDim;
      } else {
        os() << "0";
      }
    }
    os() << "};\n";
    return arrName;
  }

  std::string emitSliceShape(TensorTileOp tile, const std::string &prefix) {
    auto tileTy = cast<coir::TensorType>(tile.getResult().getType());
    std::string arrName = prefix + "__sshape__";
    os() << getIndent() << "unsigned int " << arrName << "[] = {";
    for (unsigned i = 0; i < tileTy.getShape().size(); ++i) {
      if (i) os() << ", ";
      os() << tileTy.getShape()[i];
    }
    os() << "};\n";
    return arrName;
  }

  void emitDmaCopy(DmaCopyOp op) override {
    auto kind = op.getKind();

    auto srcTile = getTileDefiningOp(op.getSource());
    auto dstTile = getTileDefiningOp(op.getDest());

    int64_t srcElems = tensorElems(op.getSource());
    int64_t dstElems = tensorElems(op.getDest());
    int64_t copyElems = std::max(srcElems, dstElems);

    bool isAsync = hasAsyncUses(op.getToken());
    unsigned id = nextDmaId++;
    std::string ctxName = "__dte_" + std::to_string(id);
    std::string futName = "__fut_" + std::to_string(id);

    os() << getIndent() << getDTEType() << " " << ctxName << ";\n";
    os() << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_" << id << "\", 0, 0);\n";

    std::string evName = futName + "__event__";
    std::string apiCall;

    if (kind == coir::DMAKind::Copy) {
      if (srcTile && !dstTile) {
        std::string dstMds = emitMdspanWithShape(op.getDest());
        std::string srcMds = emitFullBaseMdspan(srcTile);
        std::string offArr = emitSliceOffsets(srcTile, futName);
        apiCall = (isAsync ? "tops::slice_async" : "tops::slice");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + offArr + ")";
      } else if (!srcTile && dstTile) {
        std::string srcMds = emitMdspanWithShape(op.getSource());
        std::string dstMds = emitFullBaseMdspan(dstTile);
        std::string offArr = emitSliceOffsets(dstTile, futName);
        apiCall = (isAsync ? "tops::deslice_async" : "tops::deslice");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + offArr + ")";
      } else if (srcTile && dstTile) {
        std::string srcMds = emitFullBaseMdspan(srcTile);
        std::string dstMds = emitFullBaseMdspan(dstTile);
        std::string srcOff = emitSliceOffsets(srcTile, futName + "_s");
        std::string sShape = emitSliceShape(srcTile, futName);
        std::string dstOff = emitSliceOffsets(dstTile, futName + "_d");
        apiCall = (isAsync ? "tops::slice_deslice_async"
                           : "tops::slice_deslice");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + srcOff + ", " + sShape + ", " +
                   dstOff + ")";
      } else {
        std::string srcMds = emitCopyMdspan(op.getSource(), copyElems);
        std::string dstMds = emitCopyMdspan(op.getDest(), copyElems);
        apiCall = (isAsync ? "tops::memcpy_async" : "tops::memcpy");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ")";
      }
    } else if (kind == coir::DMAKind::Pad) {
      emitPadArrays(op, futName);
      std::string padValStr = emitPadValue(op);
      std::string padArgs = futName + "__pad_low__, " + futName +
                            "__pad_high__, " + futName + "__pad_mid__, " +
                            padValStr;
      if (srcTile) {
        std::string dstMds = emitMdspanWithShape(op.getDest());
        std::string srcMds = emitFullBaseMdspan(srcTile);
        std::string offArr = emitSliceOffsets(srcTile, futName);
        std::string sShape = emitSliceShape(srcTile, futName);
        apiCall = (isAsync ? "tops::slice_pad_async" : "tops::slice_pad");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + offArr + ", " + sShape + ", " +
                   padArgs + ")";
      } else {
        std::string srcMds = emitMdspanWithShape(op.getSource());
        std::string dstMds = emitMdspanWithShape(op.getDest());
        apiCall = (isAsync ? "tops::pad_async" : "tops::pad");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + padArgs + ")";
      }
    } else if (kind == coir::DMAKind::Transpose) {
      emitTransposeLayout(op, futName);
      std::string layout = futName + "__layout__";
      if (srcTile && !dstTile) {
        std::string dstMds = emitMdspanWithShape(op.getDest());
        std::string srcMds = emitFullBaseMdspan(srcTile);
        std::string offArr = emitSliceOffsets(srcTile, futName);
        apiCall = (isAsync ? "tops::slice_transpose_async"
                           : "tops::slice_transpose");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + offArr + ", " + layout + ")";
      } else if (!srcTile && dstTile) {
        std::string srcMds = emitMdspanWithShape(op.getSource());
        std::string dstMds = emitFullBaseMdspan(dstTile);
        std::string offArr = emitSliceOffsets(dstTile, futName);
        apiCall = (isAsync ? "tops::transpose_deslice_async"
                           : "tops::transpose_deslice");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + layout + ", " + offArr + ")";
      } else {
        std::string srcMds = emitMdspanWithShape(op.getSource());
        std::string dstMds = emitMdspanWithShape(op.getDest());
        apiCall = (isAsync ? "tops::transpose_async" : "tops::transpose");
        apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                   srcMds + ", " + layout + ")";
      }
    } else {
      std::string srcMds = emitCopyMdspan(op.getSource(), copyElems);
      std::string dstMds = emitCopyMdspan(op.getDest(), copyElems);
      apiCall = (isAsync ? "tops::memcpy_async" : "tops::memcpy");
      apiCall += "(*" + futName + ".get_ctx(), " + dstMds + ", " +
                 srcMds + ")";
    }

    os() << getIndent();
    if (isAsync) os() << "tops::event " << evName << " = ";
    os() << apiCall << ";\n";

    if (isAsync) {
      os() << getIndent() << futName << ".set_event(" << evName << ");\n";
      asyncFutures[op.getToken()] = futName;
    } else {
      os() << getIndent() << futName << ".set_nowait();\n";
    }
  }

  void emitPadArrays(DmaCopyOp op, const std::string &futName) {
    auto emitArr = [&](const char *suffix,
                       std::optional<ArrayRef<int64_t>> arr, int rank) {
      os() << getIndent() << "unsigned int " << futName << suffix << "[] = {";
      if (arr) {
        for (int i = 0; i < (int)arr->size(); ++i) {
          if (i) os() << ", ";
          os() << (*arr)[i];
        }
      } else {
        for (int i = 0; i < rank; ++i) {
          if (i) os() << ", ";
          os() << "0";
        }
      }
      os() << "};\n";
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
    os() << getIndent() << "int " << futName << "__layout__[] = {";
    if (auto perm = op.getTransposePerm()) {
      for (int i = 0; i < (int)perm->size(); ++i) {
        if (i) os() << ", ";
        os() << (*perm)[i];
      }
    }
    os() << "};\n";
  }

  void emitWait(WaitOp op) override {
    auto it = asyncFutures.find(op.getToken());
    if (it != asyncFutures.end())
      os() << getIndent() << it->second << ".wait();\n";
  }

  void emitFutureRotate(FutureRotateOp op) override {
    auto inputs = op.getFutures();
    auto outputs = op.getResults();
    SmallVector<std::string> names;
    for (auto in : inputs) {
      auto it = asyncFutures.find(in);
      names.push_back(it != asyncFutures.end() ? it->second : "?");
    }
    os() << getIndent() << "choreo::rotate(";
    for (unsigned i = 0; i < names.size(); ++i) {
      if (i) os() << ", ";
      os() << names[i];
    }
    os() << ");\n";
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

  void emitDMAConstDesc(DMAConstDescOp op) override {
    unsigned id = nextDmaId++;
    std::string ctxName = "__dma_" + std::to_string(id);
    std::string futName = "__fut_desc_" + std::to_string(id);
    dmaCtxNames[op.getOut()] = futName;

    os() << getIndent() << getDTEType() << " " << ctxName << ";\n";
    os() << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_desc_" << id << "\", 0, 0);\n";

    std::string srcMds = emitMdspan(op.getSource());
    std::string dstMds = emitMdspan(op.getDest());

    auto kind = op.getKind();
    if (kind == coir::DMAKind::Copy) {
      os() << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Slice) {
      os() << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Transpose) {
      os() << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    } else if (kind == coir::DMAKind::Pad) {
      os() << getIndent() << futName << ".configure(" << dstMds << ", "
         << srcMds << ");\n";
    }
  }

  void emitDMAPrefetch(DMADescPrefetchOp op) override {
    auto it = dmaCtxNames.find(op.getIn());
    if (it != dmaCtxNames.end())
      dmaCtxNames[op.getOut()] = it->second;
  }

  void emitDMARuntimeDesc(DMADescRuntimeOp op) override {
    auto it = dmaCtxNames.find(op.getIn());
    std::string futName = (it != dmaCtxNames.end()) ? it->second : "__dma_?";
    dmaCtxNames[op.getOut()] = futName;

    auto offsets = op.getOffsets();
    for (unsigned i = 0; i < offsets.size(); ++i) {
      os() << getIndent() << futName << ".set_offset(" << i << ", "
         << getName(offsets[i]) << ");\n";
    }
  }

  void emitDMAInvoke(DMAInvokeOp op) override {
    auto it = dmaCtxNames.find(op.getDesc());
    std::string ctxName = (it != dmaCtxNames.end()) ? it->second : "__dma_?";

    if (hasAsyncUses(op.getDone())) {
      os() << getIndent() << ctxName << ".trigger_only();\n";
      asyncFutures[op.getDone()] = ctxName;
    } else {
      os() << getIndent() << ctxName << ".trigger_and_wait();\n";
    }
  }




  std::string getAllocQualifier(coir::TensorType tty) override {
    return tty.getMemorySpace() == 1 ? "__local__ " : "";
  }

  void emitTensorAlloc(TensorAllocOp op) override {
    if (returnValues.count(op.getResult())) return;

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    std::string name = getName(op.getResult());
    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape()) totalElems *= d;

    std::string qualifier = getAllocQualifier(tensorTy);
    os() << getIndent() << qualifier << emitType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";
  }

  void emitBarrier(BarrierOp op) override {
    switch (op.getScope()) {
    case coir::ParallelLevel::BLOCK:
      os() << getIndent() << "__syncthreads();\n";
      break;
    case coir::ParallelLevel::GROUP:
      os() << getIndent() << "__syncsubthreads();\n";
      break;
    case coir::ParallelLevel::DEVICE:
      os() << getIndent() << "topsDeviceSynchronize();\n";
      break;
    default:
      os() << getIndent() << "__syncthreads();\n";
      break;
    }
  }

  void emitEventTrigger(EventTriggerOp op) {
    auto name = op.getEventName().str();
    os() << getIndent() << name;
    if (auto sub = op.getSubscript())
      os() << "[" << sub->str() << "]";
    os() << " = true;\n";
  }

  void emitEventWait(EventWaitOp op) {
    auto name = op.getEventName().str();
    std::string ref = name;
    if (auto sub = op.getSubscript())
      ref += "[" + sub->str() + "]";
    os() << getIndent() << "while (" << ref << " == false) continue;\n";
    os() << getIndent() << ref << " = false;\n";
  }


  void emitWhileOp(mlir::scf::WhileOp op) override {
    auto &beforeBlock = op.getBefore().front();
    auto condOp = dyn_cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());
    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i) {
      std::string name = "wv" + std::to_string(nextId++);
      iterVarNames.push_back(name);
      os() << getIndent() << emitType(op.getInits()[i].getType()) << " "
         << name << " = " << getName(op.getInits()[i]) << ";\n";
      valueNames[beforeBlock.getArgument(i)] = name;
    }
    // Emit condition computation and save the condition variable name.
    for (auto &bodyOp : beforeBlock.getOperations()) {
      if (isa<mlir::scf::ConditionOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }
    std::string condName = getName(condOp.getCondition());
    os() << getIndent() << "while (" << condName << ") {\n";
    incIndent();
    auto &afterBlock = op.getAfter().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[afterBlock.getArgument(i)] = getName(condOp.getArgs()[i]);
    for (auto &bodyOp : afterBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          os() << getIndent() << iterVarNames[i] << " = "
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
    os() << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      valueNames[op.getResult(i)] = iterVarNames[i];
  }

  void emitCoIRWhileOp(coir::CoIRWhileOp op) override {
    auto &condBlock = op.getCondRegion().front();
    auto condOp = dyn_cast<coir::CoIRWhileCondOp>(condBlock.getTerminator());
    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i) {
      std::string name = "wv" + std::to_string(nextId++);
      iterVarNames.push_back(name);
      os() << getIndent() << emitType(op.getInits()[i].getType()) << " "
         << name << " = " << getName(op.getInits()[i]) << ";\n";
      valueNames[condBlock.getArgument(i)] = name;
    }
    for (auto &bodyOp : condBlock.getOperations()) {
      if (isa<coir::CoIRWhileCondOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }
    os() << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
    incIndent();
    auto &bodyBlock = op.getBodyRegion().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[bodyBlock.getArgument(i)] = getName(condOp.getArgs()[i]);
    for (auto &bodyOp : bodyBlock.getOperations()) {
      if (auto breakOp = dyn_cast<coir::CoIRBreakOp>(&bodyOp)) {
        for (unsigned i = 0; i < breakOp.getOperands().size(); ++i) {
          os() << getIndent() << iterVarNames[i] << " = "
             << getName(breakOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] = iterVarNames[i];
        }
        os() << getIndent() << "break;\n";
      } else if (auto contOp = dyn_cast<coir::CoIRContinueOp>(&bodyOp)) {
        for (unsigned i = 0; i < contOp.getOperands().size(); ++i) {
          os() << getIndent() << iterVarNames[i] << " = "
             << getName(contOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] = iterVarNames[i];
        }
        for (auto &cOp : condBlock.getOperations()) {
          if (isa<coir::CoIRWhileCondOp>(&cOp)) continue;
          emitReassignOp(&cOp);
        }
        os() << getIndent() << "continue;\n";
      } else {
        emitOp(&bodyOp);
      }
    }
    decIndent();
    os() << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      valueNames[op.getResult(i)] = iterVarNames[i];
  }



  void emitCall(CallOp op) {
    auto callee = op.getCallee().str();
    os() << getIndent() << callee;

    // Template arguments
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

    // Call arguments
    os() << "(";
    bool first = true;
    for (auto arg : op.getOperands_()) {
      if (!first) os() << ", ";
      first = false;
      auto ty = arg.getType();
      if (auto tty = mlir::dyn_cast<coir::TensorType>(ty)) {
        os() << "(" << emitType(tty.getElementType()) << "*)"
           << getName(arg);
      } else {
        os() << getName(arg);
      }
    }
    os() << ");\n";
  }

  void emitInThreads(InThreadsOp op) {
    os() << getIndent() << "if (" << getName(op.getPredicate()) << ") {\n";
    incIndent();
    for (auto &bodyOp : op.getBody().front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os() << getIndent() << "}";
    bool isAsync = op.getAsync() && *op.getAsync();
    bool isOuter = !op.getOuter() || *op.getOuter();
    if (!isAsync && isOuter)
      os() << "\n" << getIndent() << "__syncthreads();";
    os() << "\n";
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

    os() << getIndent() << fnName << "(&" << getName(op.getDest()) << "[";
    bool first = true;
    for (auto idx : op.getIndices()) {
      if (!first) os() << " + ";
      first = false;
      os() << getName(idx);
    }
    os() << "], " << getName(op.getValue());
    if (op.getKind() == AK::CAS && op.getCompare())
      os() << ", " << "/* compare */";
    os() << ");\n";
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
      os() << getIndent() << existingName << " = (" << lhs << " " << opStr
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
      os() << getIndent() << existingName << " = (" << lhs << " " << opStr
         << " " << rhs << ");\n";
    } else {
      emitOp(op);
    }
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
    TopsccEmitter emitter;
    emitter.emitModule(module, llvm::outs());
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitTopsccPass() {
  return std::make_unique<EmitTopsccPass>();
}

void emitTopscc(mlir::ModuleOp module, llvm::raw_ostream &os) {
  TopsccEmitter emitter;
  emitter.emitModule(module, os);
}
} // namespace coir

static mlir::PassRegistration<EmitTopsccPass> reg;

namespace {

static bool registered_topscc = [] {
  CoIR::CodeGenRegistry::Register("topscc", [] {
    return std::make_unique<TopsccEmitter>();
  });
  CoIR::CodeGenRegistry::Register("gcu", [] {
    return std::make_unique<TopsccEmitter>();
  });
  return true;
}();

} // namespace
