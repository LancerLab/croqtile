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

    // GCU local (private) memory requires a compile-time-sized definition for
    // the dynamic memory-reuse pool.  Kernels can override the default via
    // the cocc CFLAGS hook: CFLAGS='-D__CO_DYN_SMEM_SIZE=<bytes>'.
    os() << "#ifndef __CO_DYN_SMEM_SIZE\n"
            "#define __CO_DYN_SMEM_SIZE 262144\n"
            "#endif\n\n";

    emitExplicitDeviceCode(module, out);

    for (auto &op : module.getBody()->getOperations()) {
      if (auto kernel = dyn_cast<KernelOp>(op))
        if (hasBlockParallel(kernel)) {
          spmNames_.clear();
          dynSpmEmitted_ = false;
          emitDeviceFunction(kernel);
        }
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
    auto &sctx = CoIR::ScriptContext::Get();
    os << "\"$TOPSCC\" ${CFLAGS} -D__CHOREO_DMA_DIAGNOSIS__"
          " -Wno-implicitly-unsigned-literal"
          " -I\"$TMPDIR\" -I\"$TMPDIR/topscc\"";
    if (!sctx.source_dir.empty())
      os << " -I\"" << sctx.source_dir << "\"";
    os << " -o \"$BINFILE\" \"$TMPFILE\" -lpthread -ldl -lrt 2>&1\n";
    emitScriptExecuteBlock(os);
    return 0;
  }


private:
  std::string archStr;
  int archNum = 0;

  struct EntryAssertion { AssertOp op; };
  llvm::SmallVector<EntryAssertion> entryAssertions;

  /// SPM pool tracking for memory reuse.
  /// Maps reuse_spm name → __spm_N variable name.  First reuse alloc for
  /// a pool emits the backing array; subsequent allocs emit pointer aliases.
  llvm::DenseMap<llvm::StringRef, std::string> spmNames_;
  /// Whether the dynamic shared-memory extern declaration has been emitted.
  bool dynSpmEmitted_ = false;

  /// When true, emitOp emits host-side code: skips parallel blocks,
  /// emits tensor.alloc as nullptr, etc.
  bool isEmittingHost_ = false;

  /// Values from host-side tensor.alloc (emitted as nullptr).
  /// These should NOT have .data() appended in host print calls.
  llvm::DenseSet<mlir::Value> hostTensorAllocs_;

  /// Names of MR offset kernel parameters (from coir.mr_offset_args).
  /// Used to distinguish kernel params from local buffer offsets.
  llvm::DenseSet<llvm::StringRef> mrOffsetParamNames_;

  /// Names of hoisted param values computed in host function (v* names).
  /// Populated during host-side emission, used in kernel launch.
  llvm::SmallVector<std::string> hoistedParamNames_;

  bool hasGroupLevel() const override { return archNum >= 400; }
  bool supportsFP8() const override { return archNum >= 400; }
  bool supportsFP6() const override { return false; }
  bool supportsFP4() const override { return false; }
  bool supportsLaunchBounds() const override { return false; }
  bool supportsMaxNreg() const override { return false; }

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
    if (ty.isF16()) return "choreo::f16";
    if (ty.isBF16()) return "choreo::bf16";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (isa<mlir::Float8E4M3FNType>(ty) || isa<mlir::Float8E5M2Type>(ty)) {
      if (!supportsFP8()) {
        llvm::errs() << "error: f8 types not supported on " << archStr << "\n";
        return "/* unsupported_f8 */";
      }
      if (isa<mlir::Float8E4M3FNType>(ty)) return "tops::float_e4m3";
      return "tops::float_e5m2";
    }
    if (isa<mlir::Float6E2M3FNType>(ty) || isa<mlir::Float6E3M2FNType>(ty) ||
        isa<mlir::Float4E2M1FNType>(ty)) {
      llvm::errs() << "error: topscc does not support sub-byte float types"
                      " (f6/f4); these require the CUTE target\n";
      return "/* unsupported_narrow_float */";
    }
    if (ty.isInteger(8)) return "int8_t";
    if (ty.isInteger(16)) return "int16_t";
    if (ty.isInteger(32)) return "int";
    if (ty.isInteger(64)) return "int64_t";
    return "/* unknown */";
  }

  std::string emitElementType(Type ty) override {
    if (ty.isF16()) return "choreo::f16";
    if (ty.isBF16()) return "choreo::bf16";
    if (ty.isF32()) return "float";
    if (ty.isF64()) return "double";
    if (isa<mlir::Float8E4M3FNType>(ty) || isa<mlir::Float8E5M2Type>(ty)) {
      if (!supportsFP8()) return "/* unsupported_f8 */";
      if (isa<mlir::Float8E4M3FNType>(ty)) return "tops::float_e4m3";
      return "tops::float_e5m2";
    }
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

  // DTE context type depends on the memory spaces involved:
  // - Global ↔ Shared: use collective DTE (choreo_cdte_priv)
  // - Anything involving Local/Register: use per-thread DTE (choreo_sdte)
  // Uses portable aliases from choreo.h so generated code compiles on
  // all architectures (gcu200 maps both to tops_dte_ctx_t).
  std::string getDTEType(Type srcType, Type dstType) const {
    auto srcMS = 0, dstMS = 0;
    if (auto tty = dyn_cast<coir::TensorType>(srcType))
      srcMS = tty.getMemorySpace();
    if (auto tty = dyn_cast<coir::TensorType>(dstType))
      dstMS = tty.getMemorySpace();
    bool bothGlobalOrShared =
        srcMS <= (int)coir::TensorMemorySpace::Shared &&
        dstMS <= (int)coir::TensorMemorySpace::Shared;
    return bothGlobalOrShared ? "choreo::choreo_cdte_priv" : "choreo::choreo_sdte";
  }

  // tops_dte_ctx_t requires explicit .init() on legacy targets (gcu210).
  // On gcu300/400 the type is RAII and init() is a no-op, so always calling
  // it is safe and keeps generated code portable.
  bool needsExplicitInit() const { return true; }

  void emitHeader() {
    os() << "#include <stdint.h>\n";
    os() << "#include <cstdio>\n";
    os() << "#include <tops.h>\n";
    os() << "#include \"tops/tops_runtime.h\"\n";
    os() << "#include \"choreo.h\"\n";
    os() << "using namespace choreo;\n\n";
    // Force line-buffered stdout so host printf output merges correctly
    // with device stderr when piped (e.g. `2>&1 | FileCheck`).
    os() << "__attribute__((constructor))\n"
            "static void __co_init_stdout() { setvbuf(stdout, NULL, _IOLBF, 0); }\n\n";
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
    std::string streamName;
    bool isAsync = false;
  };

  LaunchConfig collectLaunchConfig(KernelOp kernel) {
    LaunchConfig lc;
    kernel.walk([&](ParallelOp p) {
      auto bounds = p.getBounds();
      switch (p.getLevel()) {
      case ParallelLevel::BLOCK:
        for (auto b : bounds) lc.blockDims.push_back(b);
        if (p.getStreamAttr())
          lc.streamName = p.getStreamAttr().getValue().str();
        if (p.getIsAsyncAttr() && p.getIsAsyncAttr().getValue())
          lc.isAsync = true;
        break;
      case ParallelLevel::GROUP:
        for (auto b : bounds) lc.groupDims.push_back(b);
        break;
      case ParallelLevel::GROUPx4:
        for (auto b : bounds) lc.groupDims.push_back(b * 4);
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

  /// Check if a region contains any ParallelOp (recursively).
  static bool regionHasParallel(Region &region) {
    bool found = false;
    region.walk([&](ParallelOp) { found = true; });
    return found;
  }

  /// Check if an op is a pure computation (no side effects).
  /// Pure ops are safe to emit in both host and device functions.
  static bool isPureOp(Operation *op) {
    return isa<arith::ConstantOp, arith::IndexCastOp, arith::SelectOp,
               arith::ExtSIOp, arith::ExtFOp, arith::TruncIOp,
               arith::TruncFOp, TensorBindDimsOp>(op) ||
           op->hasTrait<mlir::OpTrait::IsCommutative>() ||
           // All standard arith binary/unary ops:
           isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
               arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::DivFOp,
               arith::RemSIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
               arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
               arith::CmpIOp, arith::CmpFOp, arith::NegFOp>(op);
  }

  void emitDeviceFunction(KernelOp kernel) {
    entryAssertions.clear();
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

    os() << "__device__ ";
    if (auto lb = kernel.getLaunchBoundsAttr()) {
      if (lb.getMaxThreadsPerBlock() > 0) {
        if (supportsLaunchBounds()) {
          os() << "__launch_bounds__(" << lb.getMaxThreadsPerBlock();
          if (lb.getMinBlocksPerMultiprocessor() > 0)
            os() << ", " << lb.getMinBlocksPerMultiprocessor();
          os() << ") ";
        } else {
          os() << "/* launch_bounds(" << lb.getMaxThreadsPerBlock();
          if (lb.getMinBlocksPerMultiprocessor() > 0)
            os() << ", " << lb.getMinBlocksPerMultiprocessor();
          os() << ") unsupported */ ";
        }
      }
    }
    if (auto nr = kernel.getMaxNregAttr()) {
      if (nr.getValue() > 0) {
        if (supportsMaxNreg())
          os() << "__maxnreg__(" << nr.getValue() << ") ";
        else
          os() << "/* maxnreg(" << nr.getValue() << ") unsupported */ ";
      }
    }
    os() << "void __choreo_device_" << kernel.getSymName() << "(";

    auto &body = kernel.getBody();
    unsigned paramIdx = 0;

    // Build MR arg name mapping so dynamic reuse offsets are named
    // with their mr_offset_* / spm_size identifiers instead of "argN".
    auto mrOffsets =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_offset_args");
    auto mrSpmSize =
        kernel->getAttrOfType<mlir::StringAttr>("coir.mr_spm_size_arg");
    unsigned numMrOffsets = mrOffsets ? mrOffsets.size() : 0;
    mrOffsetParamNames_.clear();
    if (mrOffsets)
      for (auto a : mrOffsets)
        mrOffsetParamNames_.insert(cast<mlir::StringAttr>(a).getValue());
    unsigned numFnInputs = fnType.getNumInputs();
    unsigned mrBaseIdx = 0;
    if (!body.empty()) {
      auto args = body.getArguments();
      mrBaseIdx = numFnInputs - numMrOffsets - (mrSpmSize ? 1 : 0);
      for (unsigned i = 0; i < numFnInputs && i < args.size(); ++i) {
        if (paramIdx > 0) os() << ", ";
        std::string name;
        if (mrOffsets && i >= mrBaseIdx && i < mrBaseIdx + numMrOffsets) {
          auto offsetIdx = i - mrBaseIdx;
          name = mlir::cast<mlir::StringAttr>(mrOffsets[offsetIdx])
                     .getValue()
                     .str();
        } else if (mrSpmSize && i == numFnInputs - 1) {
          name = mrSpmSize.getValue().str();
        } else {
          name = "arg" + std::to_string(paramIdx);
        }
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

    // Emit dimension variable aliases so hoisted constants can reference
    // them by name (e.g., "L", "N", "M").
    auto dimArgMeta = getDimArgs(kernel);
    if (!body.empty()) {
      auto bodyArgs = body.getArguments();
      auto mrOffsets =
          kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_offset_args");
      auto mrSpmSize =
          kernel->getAttrOfType<mlir::StringAttr>("coir.mr_spm_size_arg");
      unsigned numMrOffsets = mrOffsets ? mrOffsets.size() : 0;
      unsigned mrBaseIdx =
          bodyArgs.size() - numMrOffsets - (mrSpmSize ? 1 : 0);
      unsigned numOrigInputs = mrBaseIdx - dimArgMeta.size();
      for (unsigned i = 0; i < dimArgMeta.size(); ++i) {
        unsigned argIdx = numOrigInputs + i;
        if (argIdx < bodyArgs.size()) {
          std::string argName = valueNames.count(bodyArgs[argIdx])
                                    ? valueNames[bodyArgs[argIdx]]
                                    : "arg" + std::to_string(argIdx);
          os() << getIndent() << "const int " << dimArgMeta[i].name
               << " = " << argName << ";\n";
        }
      }
    }

    // Pre-scan: collect all SPM pool names and emit pool declarations
    // at the function level to ensure they're in scope for all uses.
    spmNames_.clear();
    kernel.walk([&](TensorAllocOp allocOp) {
      if (!allocOp.getReuseOffsetAttr()) return;
      // Skip dynamic offset allocs that ARE kernel params (handled by
      // __dyn_smem path). Also skip allocs with dyn_offset_arg when
      // mrOffsetParamNames_ hasn't been populated (non-device functions).
      if (auto dynArg = allocOp->getAttrOfType<mlir::StringAttr>("dyn_offset_arg"))
        if (mrOffsetParamNames_.empty() || mrOffsetParamNames_.count(dynArg.getValue()))
          return;
      auto poolNameOpt = allocOp.getReuseSpm();
      llvm::StringRef poolName =
          poolNameOpt.has_value() ? *poolNameOpt : "__default_spm";
      if (spmNames_.count(poolName)) return;
      std::string spmVar = "__spm_" + std::to_string(nextId++);
      spmNames_[poolName] = spmVar;
      int64_t spmBytes = 0;
      if (auto spmSizeAttr =
              allocOp->getAttrOfType<mlir::IntegerAttr>("spm_size"))
        spmBytes = spmSizeAttr.getInt();
      auto tensorTy = cast<coir::TensorType>(allocOp.getResult().getType());
      std::string qual = getAllocQualifier(tensorTy);
      os() << getIndent() << qual << "unsigned char "
           << spmVar << "[" << spmBytes << "];\n";
    });

    // Pre-scan: if any dynamic-offset tensor alloc exists, emit the
    // __dyn_smem extern at function level for cross-scope visibility.
    dynSpmEmitted_ = false;
    kernel.walk([&](TensorAllocOp allocOp) {
      if (dynSpmEmitted_) return;
      if (allocOp->getAttrOfType<mlir::StringAttr>("dyn_offset_arg")) {
        auto tensorTy = cast<coir::TensorType>(allocOp.getResult().getType());
        bool isLocal =
            (tensorTy.getMemorySpace() ==
             static_cast<int32_t>(coir::TensorMemorySpace::Local));
        if (isLocal) {
          os() << getIndent()
               << "__local__ unsigned char __dyn_smem[__CO_DYN_SMEM_SIZE];\n";
        } else {
          std::string qual = getAllocQualifier(tensorTy);
          os() << getIndent() << "extern " << qual
               << "unsigned char __dyn_smem[];\n";
        }
        dynSpmEmitted_ = true;
      }
    });

    for (auto &op : body.front().getOperations()) {
      if (auto ret = dyn_cast<KernelReturnOp>(op)) {
        for (unsigned i = 0; i < ret.getOperands().size(); ++i) {
          returnValues.insert(ret.getOperands()[i]);
          if (getReturnInputArgIdx(kernel, i) < 0)
            valueNames[ret.getOperands()[i]] = returnParamNames[i];
        }
      }
    }

    // Device-side emission: parallel blocks, control-flow containing
    // parallel blocks, and pure computations (hoisted by lowering).
    // All other top-level ops are host code and skipped.
    for (auto &op : body.front().getOperations()) {
      if (isa<KernelReturnOp>(&op)) continue;
      if (isa<ParallelOp>(&op)) {
        emitOp(&op);
      } else if (op.getNumRegions() > 0) {
        bool hasPar = false;
        for (auto &region : op.getRegions())
          if (regionHasParallel(region)) { hasPar = true; break; }
        if (hasPar) emitOp(&op);
      } else if (isPureOp(&op) || isa<TensorAllocOp>(&op)) {
        emitOp(&op);
      }
    }

    decIndent();
    os() << "}\n\n";
  }

  /// Walk through tensor.tile ops to find the underlying tensor value.
  Value getTensorDefOp(Value tensor) const {
    while (auto tile = tensor.getDefiningOp<TensorTileOp>())
      tensor = tile.getSource();
    if (auto bind = tensor.getDefiningOp<TensorBindDimsOp>())
      tensor = bind.getSource();
    return tensor;
  }

  /// Emit a single dimension expression.  Returns the static value as a
  /// string or the name of the dynamic SSA value.
  std::string emitDimExpr(Value tensor, unsigned dimIdx) {
    auto tty = cast<coir::TensorType>(tensor.getType());
    int64_t staticDim = tty.getShape()[dimIdx];
    if (!mlir::ShapedType::isDynamic(staticDim))
      return std::to_string(staticDim);

    auto defVal = getTensorDefOp(tensor);

    // Helper to extract the dynIdx-th dynamic dim name from operands.
    auto resolveDyn = [&](mlir::OperandRange dynDims) -> std::string {
      unsigned dynIdx = 0;
      for (unsigned i = 0; i < tty.getShape().size(); ++i) {
        if (tty.isDynamicDim(i)) {
          if (i == dimIdx) {
            assert(dynIdx < dynDims.size() &&
                   "bind_dims operand count mismatch");
            return getName(dynDims[dynIdx]);
          }
          dynIdx++;
        }
      }
      llvm_unreachable("dynamic dim not found in bind_dims operands");
    };

    if (auto alloc = defVal.getDefiningOp<TensorAllocOp>())
      return resolveDyn(alloc.getDynamicDims());
    if (auto bind = defVal.getDefiningOp<TensorBindDimsOp>())
      return resolveDyn(bind.getDynamicDims());

    // Kernel block argument: resolve via dim_args / dim_checks metadata.
    if (auto blockArg = dyn_cast<BlockArgument>(defVal)) {
      // Compute the dynamic-dim index within the base tensor.
      auto baseTy = cast<coir::TensorType>(defVal.getType());
      unsigned dynIdx = 0;
      bool found = false;
      for (unsigned i = 0; i < baseTy.getShape().size(); ++i) {
        if (baseTy.isDynamicDim(i)) {
          if (i == dimIdx) { found = true; break; }
          dynIdx++;
        }
      }
      if (!found)
        return "1"; // static dim that was somehow misidentified

      auto kOp =
          blockArg.getOwner()->getParent()->getParentOfType<KernelOp>();
      if (!kOp)
        return "1";

      auto dimArgMeta = getDimArgs(kOp);
      int64_t paramIdx = blockArg.getArgNumber();

      // Direct match.
      for (auto &da : dimArgMeta) {
        if (da.paramIdx == paramIdx && da.dimIdx == (int64_t)dynIdx)
          return da.name;
      }

      // Transitive match via dim_checks.
      auto dimChecks = getDimChecks(kOp);
      if (!dimChecks.empty()) {
        llvm::SmallSetVector<int64_t, 8> visited;
        llvm::SmallVector<std::pair<int64_t, int64_t>, 8> worklist;
        visited.insert(paramIdx);
        worklist.push_back({paramIdx, (int64_t)dynIdx});
        while (!worklist.empty()) {
          auto [curParam, curDim] = worklist.pop_back_val();
          for (auto &dc : dimChecks) {
            int64_t otherParam = -1, otherDim = -1;
            if (dc.param0 == curParam && dc.dim0 == curDim) {
              otherParam = dc.param1; otherDim = dc.dim1;
            } else if (dc.param1 == curParam && dc.dim1 == curDim) {
              otherParam = dc.param0; otherDim = dc.dim0;
            }
            if (otherParam >= 0 && !visited.count(otherParam)) {
              visited.insert(otherParam);
              worklist.push_back({otherParam, otherDim});
              for (auto &da : dimArgMeta) {
                if (da.paramIdx == otherParam && da.dimIdx == otherDim)
                  return da.name;
              }
            }
          }
        }
      }

      // Fallback: use the first dim_arg (backward-compatible).
      if (!dimArgMeta.empty())
        return dimArgMeta[0].name;

      return "1";
    }

    llvm_unreachable(
        "emitDimExpr: tensor has dynamic dims but no tensor.alloc, "
        "tensor.bind_dims, or kernel block argument defining it");
  }

  /// Emit the full shape of a tensor as comma-separated dim expressions.
  std::string emitTensorShape(Value tensor) {
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string result;
    for (unsigned i = 0; i < tty.getShape().size(); ++i) {
      if (i > 0) result += ", ";
      result += emitDimExpr(tensor, i);
    }
    return result;
  }

  int64_t getTensorNumElems(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) n *= d;
    return n;
  }

  /// Check if a tensor type has any dynamic dimensions.
  bool hasDynamicDims(coir::TensorType tty) {
    return tty.hasDynamicShape();
  }

  /// Emit a runtime expression string for the byte size of a dynamic tensor.
  /// E.g. "p0.shape()[0] * 4" for a 1-D f32 tensor with dynamic dim.
  /// Returns empty string if fully static (caller should use getTensorBytes).
  std::string emitDynamicBytesExpr(coir::TensorType tty, unsigned paramIdx) {
    if (!hasDynamicDims(tty)) return "";
    unsigned elemBytes = tty.getElementType().getIntOrFloatBitWidth() / 8;
    auto shape = tty.getShape();
    std::string result;
    for (unsigned d = 0; d < shape.size(); ++d) {
      if (!result.empty()) result += " * ";
      if (mlir::ShapedType::isDynamic(shape[d]))
        result += "p" + std::to_string(paramIdx) + ".shape()[" +
                  std::to_string(d) + "]";
      else
        result += std::to_string(shape[d]);
    }
    if (elemBytes > 1)
      result += " * " + std::to_string(elemBytes);
    return result;
  }

  /// Emit a runtime expression string for a dynamic shape initializer.
  /// E.g. "{p0.shape()[0]}" for a 1-D tensor with dynamic dim.
  std::string emitDynamicShapeStr(coir::TensorType tty, unsigned paramIdx) {
    auto shape = tty.getShape();
    std::string result = "{";
    for (unsigned d = 0; d < shape.size(); ++d) {
      if (d > 0) result += ", ";
      if (mlir::ShapedType::isDynamic(shape[d]))
        result += "p" + std::to_string(paramIdx) + ".shape()[" +
                  std::to_string(d) + "]";
      else
        result += std::to_string(shape[d]);
    }
    result += "}";
    return result;
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

  struct DimCheckMeta {
    std::string name;
    int64_t param0, dim0;
    int64_t param1, dim1;
  };

  struct DimArgMeta {
    int64_t paramIdx;
    int64_t dimIdx;
    std::string name;
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

  struct MRInfo {
    bool hasDynamicMR = false;
    unsigned numOffsetArgs = 0;
    std::string chunksName;
    std::string resultName;
    std::string offsetsName;
    std::string spmSizeName;
    mlir::ArrayAttr chunks;
    // Const interference matrix (from topscc_codegen optimization).
    unsigned nBuffers = 0;
    unsigned alignment = 512;
    mlir::ArrayAttr interference;
    mlir::ArrayAttr sizeExprs;
    mlir::ArrayAttr bufferIds;
  };

  MRInfo getMRInfo(KernelOp kernel) {
    MRInfo info;
    info.chunks =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_chunks");
    if (!info.chunks) return info;
    info.hasDynamicMR = true;
    auto chunksName =
        kernel->getAttrOfType<mlir::StringAttr>("coir.mr_chunks_name");
    if (chunksName) info.chunksName = chunksName.getValue().str();
    auto resultName =
        kernel->getAttrOfType<mlir::StringAttr>("coir.mr_result_name");
    if (resultName) info.resultName = resultName.getValue().str();
    auto offsetsName =
        kernel->getAttrOfType<mlir::StringAttr>("coir.mr_offsets_name");
    if (offsetsName) info.offsetsName = offsetsName.getValue().str();
    auto spmSizeName =
        kernel->getAttrOfType<mlir::StringAttr>("coir.mr_spm_size_arg");
    if (spmSizeName) info.spmSizeName = spmSizeName.getValue().str();
    auto mrOffsets =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_offset_args");
    if (mrOffsets) info.numOffsetArgs = mrOffsets.size();
    // Const interference matrix (parametric plan).
    auto nBuf =
        kernel->getAttrOfType<mlir::IntegerAttr>("coir.mr_n_buffers");
    if (nBuf) info.nBuffers = nBuf.getInt();
    auto align =
        kernel->getAttrOfType<mlir::IntegerAttr>("coir.mr_alignment");
    if (align) info.alignment = align.getInt();
    info.interference =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_interference");
    info.sizeExprs =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_size_exprs");
    info.bufferIds =
        kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_buffer_ids");
    return info;
  }

  void emitHeapSimulator(const MRInfo &mr) {
    os() << "  // JIT memory reuse\n";
    os() << "  HeapSimulator::Chunks " << mr.chunksName << ";\n";
    for (auto chunkAttr : mr.chunks) {
      os() << "  " << mr.chunksName << ".push_back("
           << mlir::cast<mlir::StringAttr>(chunkAttr).getValue().str()
           << ");\n";
    }
    os() << "  HeapSimulator __mr_sim;\n";
    if (mr.nBuffers > 0 && mr.interference && !mr.interference.empty()) {
      // Parametric plan with const interference matrix.
      std::string imatName = mr.chunksName + "_imat";
      os() << "  std::vector<bool> " << imatName << " = {";
      bool first = true;
      for (auto v : mr.interference) {
        if (!first) os() << ",";
        first = false;
        os() << (mlir::cast<mlir::BoolAttr>(v).getValue() ? "true" : "false");
      }
      os() << "};\n";
      os() << "  HeapSimulator::Result " << mr.resultName
           << " = __mr_sim.Allocate(" << mr.chunksName << ", "
           << mr.alignment << ", " << imatName << ");\n";
    } else {
      os() << "  HeapSimulator::Result " << mr.resultName
           << " = __mr_sim.Allocate(" << mr.chunksName << ", 512);\n";
    }
    os() << "  unsigned " << mr.spmSizeName << " = " << mr.resultName
         << ".heap_size;\n";
    os() << "  unsigned long " << mr.offsetsName << "[" << mr.numOffsetArgs
         << "];\n";
    os() << "  { size_t __idx = 0;\n";
    os() << "    for (const auto& [id, off] : " << mr.resultName
         << ".chunk_offsets)\n";
    os() << "      " << mr.offsetsName << "[__idx++] = off; }\n";
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

    auto mrInfo = getMRInfo(kernel);
    auto dimArgMeta = getDimArgs(kernel);
    unsigned numMrExtra = mrInfo.hasDynamicMR ? mrInfo.numOffsetArgs + 1 : 0;

    // __global__ trampoline — when device offload is needed.
    if (needsDevice) {
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
      bool hasPrevParam = false;
      for (unsigned i = 0; i < numInputs; ++i) {
        if (hasPrevParam) os() << ", ";
        os() << emitType(fnType.getInput(i)) << " g_in" << i;
        hasPrevParam = true;
      }
      if (resTy) {
        int retInputIdx = getReturnInputArgIdx(kernel, 0);
        std::string eType = emitType(resTy.getElementType());
        if (retInputIdx < 0) {
          if (hasPrevParam) os() << ", ";
          os() << eType << "* g_out, int N";
          hasPrevParam = true;
        }
      }
      if (isMultiDevice) {
        if (hasPrevParam) os() << ", ";
        os() << "int __device_id";
      }
      os() << ") {\n";
      os() << "  __choreo_device_" << name.str() << "(";
      bool hasArg = false;
      for (unsigned i = 0; i < numInputs; ++i) {
        if (hasArg) os() << ", ";
        os() << "g_in" << i;
        hasArg = true;
      }
      if (resTy && getReturnInputArgIdx(kernel, 0) < 0) {
        if (hasArg) os() << ", ";
        os() << "g_out";
        hasArg = true;
      }
      if (isMultiDevice) {
        if (hasArg) os() << ", ";
        os() << "__device_id";
      }
      os() << ");\n";
      os() << "}\n\n";
    }

    // Host function signature — only user-visible parameters.
    unsigned numOrigInputs = numInputs - dimArgMeta.size() - numMrExtra;
    if (auto paramNames = kernel->getAttrOfType<ArrayAttr>("coir.param_names"))
      numOrigInputs = std::min(numOrigInputs,
                               static_cast<unsigned>(paramNames.size()));
    llvm::SmallVector<llvm::StringRef> hostElemHints;
    if (auto attr = kernel->getAttrOfType<ArrayAttr>("coir.host_elem_types"))
      for (auto a : attr)
        hostElemHints.push_back(cast<StringAttr>(a).getValue());
    os() << hostReturnType(fnType) << " " << name.str() << "(";
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (i > 0) os() << ", ";
      auto inTy = fnType.getInput(i);
      if (auto tensorTy = dyn_cast<coir::TensorType>(inTy)) {
        unsigned inDim = tensorTy.getShape().size();
        std::string elemStr =
            (i < hostElemHints.size() && !hostElemHints[i].empty())
                ? ("choreo::" + hostElemHints[i].str())
                : choreoType(tensorTy.getElementType());
        os() << "const choreo::spanned_view<"
           << elemStr << ", " << inDim << "> & p" << i;
      } else {
        os() << emitType(inTy) << " p" << i;
      }
    }
    // Add stream parameter if kernel uses a named stream
    auto lcForHost = collectLaunchConfig(kernel);
    if (!lcForHost.streamName.empty())
      os() << ", topsStream_t " << lcForHost.streamName;
    os() << ") {\n";
    incIndent();
    emitDimChecks(kernel);

    // Always declare dimension variables (N, M, L, ...) in the host function
    // so hoisted constants can reference them by name.
    for (auto &da : dimArgMeta) {
      os() << getIndent() << "unsigned " << da.name << " = (int)p"
           << da.paramIdx << ".shape()[" << da.dimIdx << "];\n";
    }

    if (mrInfo.hasDynamicMR)
      emitHeapSimulator(mrInfo);

    if (needsDevice) {
      // Map block args → host param names for host-side op emission.
      auto &body = kernel.getBody();
      auto args = body.getArguments();
      for (unsigned i = 0; i < numOrigInputs && i < args.size(); ++i)
        valueNames[args[i]] = "p" + std::to_string(i);
      for (unsigned i = 0; i < dimArgMeta.size(); ++i) {
        unsigned argIdx = numOrigInputs + i;
        if (argIdx < args.size())
          valueNames[args[argIdx]] = dimArgMeta[i].name;
      }

      // Emit host ops in two phases: pre-parallel, then kernel launch,
      // then post-parallel. This preserves the source ordering.
      isEmittingHost_ = true;
      if (!body.empty()) {
        bool seenParallel = false;
        // Phase 1: emit ops before the first parallel block.
        for (auto &op : body.front().getOperations()) {
          if (isa<KernelReturnOp>(&op)) continue;
          if (isa<ParallelOp>(&op)) { seenParallel = true; break; }
          emitOp(&op);
        }
      }
      isEmittingHost_ = false;

      // Collect hoisted param names: for each hoisted block arg position,
      // find the corresponding v* name from the host-side emission.
      // The hoisted block args correspond to top-level index/i32 ops
      // in the kernel body, in order.
      hoistedParamNames_.clear();
      if (!body.empty()) {
        auto mrOffsets =
            kernel->getAttrOfType<mlir::ArrayAttr>("coir.mr_offset_args");
        unsigned numMrExtra = mrInfo.hasDynamicMR
            ? (mrOffsets ? mrOffsets.size() : 0) + 1 : 0;
        unsigned hoistedStart = numOrigInputs + dimArgMeta.size();
        unsigned hoistedEnd = numInputs - numMrExtra;
        unsigned hoistedIdx = 0;
        for (auto &op : body.front().getOperations()) {
          if (isa<ParallelOp, KernelReturnOp>(&op)) continue;
          for (auto result : op.getResults()) {
            auto ty = result.getType();
            if (ty.isIndex() || ty.isInteger(32) || ty.isInteger(64)) {
              if (hoistedIdx >= hoistedStart && hoistedIdx < hoistedEnd) {
                auto it = valueNames.find(result);
                hoistedParamNames_.push_back(
                    it != valueNames.end() ? it->second : "0");
              }
              ++hoistedIdx;
            }
          }
        }
      }

      emitEntryAssertions(kernel);
      if (resTy)
        emitDeviceOffloadBody(kernel, resTy, mrInfo);
      else
        emitVoidDeviceOffloadBody(kernel, mrInfo);

      // Phase 2: emit ops after the first parallel block.
      isEmittingHost_ = true;
      if (!body.empty()) {
        bool seenParallel = false;
        for (auto &op : body.front().getOperations()) {
          if (isa<ParallelOp>(&op)) { seenParallel = true; continue; }
          if (!seenParallel) continue;
          if (isa<KernelReturnOp>(&op)) continue;
          emitOp(&op);
        }
      }
      isEmittingHost_ = false;
    } else {
      // No device offload: emit the body directly (the function IS the
      // host function, just like how the AST codegen handles __co__
      // functions without parallel-by).
      auto &body = kernel.getBody();
      if (!body.empty()) {
        for (auto &op : body.front().getOperations())
          emitOp(&op);
      }
      emitEntryAssertions(kernel);
    }

    decIndent();
    os() << "}\n\n";
  }

  /// Emit hoisted computation parameters in kernel launch.
  /// These are integer values computed in the host function that the
  /// device function receives as block arguments.
  void emitHoistedLaunchArgs(KernelOp kernel, unsigned numOrigInputs,
                             const llvm::SmallVectorImpl<DimArgMeta> &dimArgMeta,
                             const MRInfo &mr) {
    for (auto &name : hoistedParamNames_)
      os() << ", (int)" << name;
  }

  void emitMRLaunchArgs(const MRInfo &mr) {
    if (!mr.hasDynamicMR) return;
    for (unsigned i = 0; i < mr.numOffsetArgs; ++i)
      os() << ", (int)" << mr.offsetsName << "[" << i << "]";
    os() << ", (int)" << mr.spmSizeName;
  }

  bool isDeviceGlobal(coir::TensorType tty) {
    return tty.getMemorySpace() ==
           static_cast<int32_t>(coir::TensorMemorySpace::Global);
  }

  void emitDeviceOffloadBody(KernelOp kernel, coir::TensorType resTy,
                             const MRInfo &mr) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    auto dimArgMeta = getDimArgs(kernel);
    unsigned numMrExtra = mr.hasDynamicMR ? mr.numOffsetArgs + 1 : 0;
    unsigned numOrigInputs = numInputs - dimArgMeta.size() - numMrExtra;
    if (auto pn = kernel->getAttrOfType<mlir::ArrayAttr>("coir.param_names"))
      numOrigInputs = std::min(numOrigInputs,
                               static_cast<unsigned>(pn.size()));
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

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) {
        std::string inEType = emitType(tty.getElementType());
        os() << "  " << inEType << "* p" << i
           << "__device = const_cast<" << inEType << "*>(p" << i
           << ".data());\n";
      } else if (tty) {
        std::string inEType = emitType(tty.getElementType());
        std::string dynBytes = emitDynamicBytesExpr(tty, i);
        os() << "  " << inEType << "* p" << i << "__device = nullptr;\n";
        if (dynBytes.empty()) {
          int64_t bytes = getTensorBytes(tty);
          os() << "  choreo::abend_true(topsMalloc((void**)&p" << i << "__device, "
             << bytes << "ULL));\n";
          os() << "  choreo::abend_true(topsMemcpy(p" << i << "__device, p" << i << ".data(), "
             << bytes << "ULL, topsMemcpyHostToDevice));\n";
        } else {
          os() << "  choreo::abend_true(topsMalloc((void**)&p" << i << "__device, "
             << dynBytes << "));\n";
          os() << "  choreo::abend_true(topsMemcpy(p" << i << "__device, p" << i << ".data(), "
             << dynBytes << ", topsMemcpyHostToDevice));\n";
        }
      }
    }

    // Build shape string for result tensor; use dynamic expressions for
    // dynamic dims via dimArgMeta.
    std::string shapeStr;
    std::string resDynBytes;
    {
      llvm::raw_string_ostream ss(shapeStr);
      ss << "{";
      for (unsigned d = 0; d < resTy.getShape().size(); ++d) {
        if (d > 0) ss << ", ";
        if (mlir::ShapedType::isDynamic(resTy.getShape()[d])) {
          bool found = false;
          for (auto &da : dimArgMeta) {
            if (da.dimIdx == (int64_t)d) {
              ss << "p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
              found = true;
              break;
            }
          }
          if (!found) ss << "0";
        } else {
          ss << resTy.getShape()[d];
        }
      }
      ss << "}";
      if (hasDynamicDims(resTy)) {
        unsigned elemBytes = resTy.getElementType().getIntOrFloatBitWidth() / 8;
        resDynBytes = "";
        for (unsigned d = 0; d < resTy.getShape().size(); ++d) {
          if (!resDynBytes.empty()) resDynBytes += " * ";
          if (mlir::ShapedType::isDynamic(resTy.getShape()[d])) {
            bool found = false;
            for (auto &da : dimArgMeta) {
              if (da.dimIdx == (int64_t)d) {
                resDynBytes += "p" + std::to_string(da.paramIdx) +
                               ".shape()[" + std::to_string(da.dimIdx) + "]";
                found = true;
                break;
              }
            }
            if (!found) resDynBytes += "0";
          } else {
            resDynBytes += std::to_string(resTy.getShape()[d]);
          }
        }
        if (elemBytes > 1)
          resDynBytes += " * " + std::to_string(elemBytes);
      }
    }

    if (retInputIdx >= 0) {
      os() << "  __coir_global_" << name.str() << "<<<" << gdims << ", "
         << bdims << ">>>(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "p" << i << "__device";
      }
      for (auto &da : dimArgMeta)
        os() << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
      emitHoistedLaunchArgs(kernel, numOrigInputs, dimArgMeta, mr);
      emitMRLaunchArgs(mr);
      os() << ");\n";
      os() << "  choreo::abend_true(topsDeviceSynchronize());\n";
      if (resDynBytes.empty()) {
        os() << "  choreo::abend_true(topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
           << ".data()), p" << retInputIdx << "__device, "
           << resBytes << "ULL, topsMemcpyDeviceToHost));\n";
      } else {
        os() << "  choreo::abend_true(topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
           << ".data()), p" << retInputIdx << "__device, "
           << resDynBytes << ", topsMemcpyDeviceToHost));\n";
      }
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os() << "  choreo::abend_true(topsFree(p" << i << "__device));\n";
      }
      os() << "  return choreo::copy_as_spanned(p" << retInputIdx
         << ".data(), p" << retInputIdx << ".shape());\n";
    } else {
      os() << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
         << ndim << ">(" << shapeStr << ");\n";
      os() << "  " << eType << "* __result__device = nullptr;\n";
      if (resDynBytes.empty()) {
        os() << "  choreo::abend_true(topsMalloc((void**)&__result__device, " << resBytes
           << "ULL));\n";
      } else {
        os() << "  choreo::abend_true(topsMalloc((void**)&__result__device, " << resDynBytes
           << "));\n";
      }
      os() << "  __coir_global_" << name.str() << "<<<" << gdims << ", "
         << bdims << ">>>(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os() << ", ";
        os() << "p" << i << "__device";
      }
      for (auto &da : dimArgMeta)
        os() << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
      emitHoistedLaunchArgs(kernel, numOrigInputs, dimArgMeta, mr);
      emitMRLaunchArgs(mr);
      // Kernel N parameter: dynamic element count or static.
      if (!resDynBytes.empty()) {
        // Strip " * elemBytes" suffix to get element count.
        unsigned elemBytes = resTy.getElementType().getIntOrFloatBitWidth() / 8;
        std::string resDynN = resDynBytes;
        if (elemBytes > 1) {
          auto pos = resDynN.rfind(" * " + std::to_string(elemBytes));
          if (pos != std::string::npos)
            resDynN = resDynN.substr(0, pos);
        }
        os() << ", __result__device, (int)(" << resDynN << "));\n";
      } else {
        os() << ", __result__device, " << resN << ");\n";
      }
      os() << "  choreo::abend_true(topsDeviceSynchronize());\n";
      if (resDynBytes.empty()) {
        os() << "  choreo::abend_true(topsMemcpy(__result.data(), __result__device, "
           << resBytes << "ULL, topsMemcpyDeviceToHost));\n";
      } else {
        os() << "  choreo::abend_true(topsMemcpy(__result.data(), __result__device, "
           << resDynBytes << ", topsMemcpyDeviceToHost));\n";
      }
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && isDeviceGlobal(tty)) continue;
        os() << "  choreo::abend_true(topsFree(p" << i << "__device));\n";
      }
      os() << "  choreo::abend_true(topsFree(__result__device));\n";
      os() << "  return __result;\n";
    }
  }

  void emitVoidDeviceOffloadBody(KernelOp kernel, const MRInfo &mr) {
    auto fnType = kernel.getFunctionType();
    auto name = kernel.getSymName();
    unsigned numInputs = fnType.getNumInputs();
    auto dimArgMeta = getDimArgs(kernel);
    unsigned numMrExtra = mr.hasDynamicMR ? mr.numOffsetArgs + 1 : 0;
    unsigned numOrigInputs = numInputs - dimArgMeta.size() - numMrExtra;
    if (auto pn = kernel->getAttrOfType<mlir::ArrayAttr>("coir.param_names"))
      numOrigInputs = std::min(numOrigInputs,
                               static_cast<unsigned>(pn.size()));
    auto lc = collectLaunchConfig(kernel);
    std::string gdims = emitDim3(lc.blockDims);
    std::string bdims = hasGroupLevel()
                            ? emitDim3(lc.groupDims)
                            : emitDim3(lc.threadDims);
    bool isMultiDevice = hasDeviceParallel(kernel);
    int64_t devCount = isMultiDevice ? getDeviceBound(kernel) : 1;

    // Multi-device void kernel: loop over devices with topsSetDevice.
    if (isMultiDevice) {
      std::string dc = std::to_string(devCount);

      // Allocate per-device buffer vectors for tensor inputs
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty || isDeviceGlobal(tty)) continue;
        std::string inEType = emitType(tty.getElementType());
        os() << "  std::vector<" << inEType << "*> p" << i
           << "__device_vec(" << dc << ", nullptr);\n";
      }

      // Device loop: set device, alloc, H2D, launch
      os() << "  for (int __d = 0; __d < " << dc << "; ++__d) {\n";
      os() << "    choreo::abend_true(topsSetDevice(__d));\n";

      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty || isDeviceGlobal(tty)) continue;
        int64_t bytes = getTensorBytes(tty);
        os() << "    choreo::abend_true(topsMalloc((void**)&p" << i
           << "__device_vec[__d], " << bytes << "ULL));\n";
        os() << "    choreo::abend_true(topsMemcpy(p" << i
           << "__device_vec[__d], p" << i << ".data(), "
           << bytes << "ULL, topsMemcpyHostToDevice));\n";
      }

      os() << "    __coir_global_" << name.str() << "<<<" << gdims << ", "
         << bdims << ">>>(";
      bool first = true;
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (!first) os() << ", ";
        first = false;
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (tty && !isDeviceGlobal(tty))
          os() << "p" << i << "__device_vec[__d]";
        else if (tty)
          os() << "const_cast<" << emitType(tty.getElementType())
             << "*>(p" << i << ".data())";
        else
          os() << "p" << i;
      }
      if (!first) os() << ", ";
      os() << "__d);\n";
      os() << "  }\n";

      // Sync loop: set device, sync, D2H for ref outputs, free
      os() << "  for (int __sync_d = 0; __sync_d < " << dc << "; ++__sync_d) {\n";
      os() << "    choreo::abend_true(topsSetDevice(__sync_d));\n";
      os() << "    choreo::abend_true(topsDeviceSynchronize());\n";

      // D2H for reference output parameters
      auto paramRefs = kernel->getAttrOfType<mlir::ArrayAttr>("coir.param_refs");
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty || isDeviceGlobal(tty)) continue;
        bool isRef = false;
        if (paramRefs && i < paramRefs.size())
          if (auto ba = mlir::dyn_cast<mlir::BoolAttr>(paramRefs[i]))
            isRef = ba.getValue();
        if (!isRef) continue;
        int64_t bytes = getTensorBytes(tty);
        os() << "    choreo::abend_true(topsMemcpy(p" << i << ".data(), p" << i
           << "__device_vec[__sync_d], " << bytes
           << "ULL, topsMemcpyDeviceToHost));\n";
      }

      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty || isDeviceGlobal(tty)) continue;
        os() << "    choreo::abend_true(topsFree(p" << i
           << "__device_vec[__sync_d]));\n";
      }
      os() << "  }\n";
      return;
    }

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty) continue;
      if (isDeviceGlobal(tty)) {
        std::string inEType = emitType(tty.getElementType());
        os() << "  " << inEType << "* p" << i
           << "__device = const_cast<" << inEType << "*>(p" << i
           << ".data());\n";
      } else {
        std::string inEType = emitType(tty.getElementType());
        std::string dynBytes = emitDynamicBytesExpr(tty, i);
        os() << "  " << inEType << "* p" << i << "__device = nullptr;\n";
        if (dynBytes.empty()) {
          int64_t bytes = getTensorBytes(tty);
          os() << "  choreo::abend_true(topsMalloc((void**)&p" << i << "__device, "
             << bytes << "ULL));\n";
          os() << "  choreo::abend_true(topsMemcpy(p" << i << "__device, p" << i << ".data(), "
             << bytes << "ULL, topsMemcpyHostToDevice));\n";
        } else {
          os() << "  choreo::abend_true(topsMalloc((void**)&p" << i << "__device, "
             << dynBytes << "));\n";
          os() << "  choreo::abend_true(topsMemcpy(p" << i << "__device, p" << i << ".data(), "
             << dynBytes << ", topsMemcpyHostToDevice));\n";
        }
      }
    }

    os() << "  __coir_global_" << name.str() << "<<<" << gdims << ", "
       << bdims << ">>>(";
    bool first = true;
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!first) os() << ", ";
      first = false;
      if (tty)
        os() << "p" << i << "__device";
      else
        os() << "p" << i;
    }
    for (auto &da : dimArgMeta)
      os() << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
    emitMRLaunchArgs(mr);
    os() << ");\n";
    os() << "  choreo::abend_true(topsDeviceSynchronize());\n";

    // D2H for reference output parameters (e.g. &C)
    auto paramRefs = kernel->getAttrOfType<mlir::ArrayAttr>("coir.param_refs");
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty || isDeviceGlobal(tty)) continue;
      bool isRef = false;
      if (paramRefs && i < paramRefs.size())
        if (auto ba = mlir::dyn_cast<mlir::BoolAttr>(paramRefs[i]))
          isRef = ba.getValue();
      if (!isRef) continue;
      std::string dynBytes = emitDynamicBytesExpr(tty, i);
      if (dynBytes.empty()) {
        int64_t bytes = getTensorBytes(tty);
        os() << "  choreo::abend_true(topsMemcpy(p" << i << ".data(), p" << i
           << "__device, " << bytes << "ULL, topsMemcpyDeviceToHost));\n";
      } else {
        os() << "  choreo::abend_true(topsMemcpy(p" << i << ".data(), p" << i
           << "__device, " << dynBytes << ", topsMemcpyDeviceToHost));\n";
      }
    }

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty || isDeviceGlobal(tty)) continue;
      os() << "  choreo::abend_true(topsFree(p" << i << "__device));\n";
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
      os() << "  choreo::abend_true(topsHostRegister(__result_buf, " << resBytes
         << "ULL, topsHostRegisterPortable));\n";
      os() << "  std::vector<" << eType << "*> __result__device_vec("
         << dc << ", nullptr);\n";
    }

    // Device loop: topsSetDevice + malloc + H2D + launch
    os() << "  for (int __d = 0; __d < " << dc << "; ++__d) {\n";
    os() << "    choreo::abend_true(topsSetDevice(__d));\n";

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) continue;
      int64_t bytes = tty ? getTensorBytes(tty) : resBytes;
      os() << "    choreo::abend_true(topsMalloc((void**)&p" << i << "__device_vec[__d], "
         << bytes << "ULL));\n";
      os() << "    choreo::abend_true(topsMemcpy(p" << i << "__device_vec[__d], p" << i
         << ".data(), " << bytes << "ULL, topsMemcpyHostToDevice));\n";
    }

    if (retInputIdx < 0) {
      os() << "    choreo::abend_true(topsMalloc((void**)&__result__device_vec[__d], "
         << resBytes << "ULL));\n";
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
    os() << "    choreo::abend_true(topsSetDevice(__sync_d));\n";
    os() << "    choreo::abend_true(topsDeviceSynchronize());\n";

    if (retInputIdx >= 0) {
      os() << "    choreo::abend_true(topsMemcpy(const_cast<" << eType << "*>(p" << retInputIdx
         << ".data()) + __sync_d * " << portionElems << ", p" << retInputIdx
         << "__device_vec[__sync_d] + __sync_d * " << portionElems << ", "
         << portionBytes << "ULL, topsMemcpyDeviceToHost));\n";
    } else {
      os() << "    choreo::abend_true(topsMemcpy(__result_buf + __sync_d * " << portionElems
         << ", __result__device_vec[__sync_d] + __sync_d * " << portionElems
         << ", " << portionBytes
         << "ULL, topsMemcpyDeviceToHost));\n";
    }

    for (unsigned i = 0; i < numInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (tty && isDeviceGlobal(tty)) continue;
      os() << "    choreo::abend_true(topsFree(p" << i << "__device_vec[__sync_d]));\n";
    }
    if (retInputIdx < 0)
      os() << "    choreo::abend_true(topsFree(__result__device_vec[__sync_d]));\n";

    os() << "  }\n";

    if (retInputIdx >= 0) {
      os() << "  return choreo::copy_as_spanned(p" << retInputIdx
         << ".data(), p" << retInputIdx << ".shape());\n";
    } else {
      os() << "  auto __result = choreo::copy_as_spanned<" << ndim
         << ">(__result_buf, " << shapeStr << ");\n";
      os() << "  choreo::abend_true(topsHostUnregister(__result_buf));\n";
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
    else if (auto evtRef = dyn_cast<EventRefOp>(op)) {
      // Emit event ref as an alias to the event variable name.
      std::string name = getName(evtRef.getResult());
      os() << getIndent() << "bool " << name << " = "
           << evtRef.getEventName() << ";\n";
    }
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto storeTile = dyn_cast<TensorStoreTileOp>(op))
      emitTensorStoreTile(storeTile);
    else if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      emitTmaCopyDiagnostic(tmaCopy);
    else
      CoIREmitterBase::emitOpFallback(op);
  }

  void emitParallel(ParallelOp op) override {
    // In host mode, parallel blocks are handled by the kernel launch
    // in emitVoidDeviceOffloadBody/emitDeviceOffloadBody — skip here.
    if (isEmittingHost_) return;
    auto level = op.getLevel();
    auto bounds = op.getBounds();
    auto &body = op.getBody();
    auto args = body.getArguments();
    const char *dimName[] = {"x", "y", "z"};

    os() << getIndent() << "// parallel level="
       << stringifyParallelLevel(level) << "\n";

    // Pre-scan for event ops and declare all events at this scope.
    if (level == ParallelLevel::THREAD) {
      llvm::SmallVector<std::string> eventNames;
      op->walk([&](Operation *innerOp) {
        if (auto ew = dyn_cast<EventWaitOp>(innerOp))
          eventNames.push_back(ew.getEventName().str());
        else if (auto et = dyn_cast<EventTriggerOp>(innerOp))
          eventNames.push_back(et.getEventName().str());
      });
      llvm::DenseSet<StringRef> seen;
      for (auto &name : eventNames) {
        if (declaredEvents.insert(name).second) {
          os() << getIndent() << "__shared__ __volatile__ bool " << name << ";\n";
        }
      }
      if (!eventNames.empty()) {
        os() << getIndent() << "if (__tops_tid_x() == 0) {\n";
        incIndent();
        llvm::DenseSet<StringRef> inited;
        for (auto &name : eventNames) {
          if (inited.insert(name).second)
            os() << getIndent() << name << " = false;\n";
        }
        decIndent();
        os() << getIndent() << "}\n";
        os() << getIndent() << "__syncthreads();\n";
      }
    }

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
    case ParallelLevel::GROUPx4:
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
      auto initVal = iterArgs[i];
      if (mlir::isa<coir::AsyncTokenType>(initVal.getType())) {
        auto futIt = asyncFutures.find(initVal);
        if (futIt != asyncFutures.end()) {
          std::string futName = futIt->second;
          asyncFutures[args[i + 1]] = futName;
          valueNames[args[i + 1]] = futName;
        } else {
          asyncFutures[args[i + 1]] = iterName;
          os() << getIndent() << "choreo::future " << iterName << ";\n";
        }
      } else {
        os() << getIndent() << "auto " << iterName << " = "
           << getName(initVal) << ";\n";
      }
      auto stateIt = acoreStates.find(initVal);
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
      auto futIt = asyncFutures.find(args[i + 1]);
      if (futIt != asyncFutures.end())
        asyncFutures[op.getResult(i)] = futIt->second;
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

    auto srcShape = srcTy.getShape();
    auto tileShape = tileTy ? tileTy.getShape() : llvm::ArrayRef<int64_t>{};

    // Compute row-major source strides
    llvm::SmallVector<int64_t> srcStrides(srcShape.size());
    {
      int64_t s = 1;
      for (int i = (int)srcShape.size() - 1; i >= 0; --i) {
        srcStrides[i] = s;
        s *= srcShape[i];
      }
    }

    // Determine per-index tile size, detecting wildcards (const-0 on large dim)
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
        if (isConst0 && srcShape[i] > 1)
          perIdxTileSize[i] = srcShape[i];
        else
          perIdxTileSize[i] = 1;
      }
    }

    os() << getIndent() << "auto " << name << " = " << getName(op.getSource())
         << " + (";
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) os() << " + ";
      os() << getName(indices[i]);
      // Compute stride = perIdxTileSize[i] * product(srcShape[i+1..])
      // For dynamic dims, emit the runtime variable name instead of
      // the kDynamic sentinel.
      llvm::SmallVector<std::string> factors;
      if (perIdxTileSize[i] != 1)
        factors.push_back(std::to_string(perIdxTileSize[i]));
      for (unsigned j = i + 1; j < srcShape.size(); ++j) {
        if (mlir::ShapedType::isDynamic(srcShape[j])) {
          factors.push_back(emitDimExpr(op.getSource(), j));
        } else if (srcShape[j] != 1) {
          factors.push_back(std::to_string(srcShape[j]));
        }
      }
      if (!factors.empty()) {
        os() << " * ";
        for (unsigned fi = 0; fi < factors.size(); ++fi) {
          if (fi > 0) os() << " * ";
          os() << factors[fi];
        }
      }
    }
    os() << ");\n";
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
      auto parentForeach = op->getParentOfType<ForeachOp>();
      if (!parentForeach) continue;
      auto iterArgs = parentForeach.getBody().front().getArguments();
      if (i + 1 >= iterArgs.size()) continue;
      auto it = acoreStates.find(yieldVal);
      if (it != acoreStates.end()) {
        auto st = it->second;
        acoreStates[iterArgs[i + 1]] = st;
      }
      auto futIt = asyncFutures.find(yieldVal);
      if (futIt != asyncFutures.end())
        asyncFutures[iterArgs[i + 1]] = futIt->second;
    }
  }

  std::string emitMdspanWithShape(Value tensor) {
    auto base = getTensorDefOp(tensor);
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string name = getName(base);
    std::string space;
    int32_t ms = tty.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      space = "tops::Private";
    else if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      space = "tops::Shared";
    else
      space = "tops::Global";
    return "tops::mdspan(" + space + ", (" +
           emitType(tty.getElementType()) + "*)" + name +
           ", " + emitTensorShape(tensor) + ")";
  }

  std::string emitCopyMdspan(Value tensor, const std::string &sizeStr) {
    auto base = getTensorDefOp(tensor);
    auto tty = cast<coir::TensorType>(tensor.getType());
    std::string name = getName(base);
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
           sizeStr + ")";
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
    Value base = getTensorDefOp(tile.getSource());
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
    return "tops::mdspan(" + space + ", (" +
           emitType(baseTy.getElementType()) + "*)" + name +
           ", " + emitTensorShape(base) + ")";
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
        auto idxTy = indices[i].getType();
        // Index-typed values may be 64-bit; braced-init into the int array
        // would then be ill-formed (narrowing), so cast explicitly.  The DTE
        // slice API takes const int*, so offsets must be int anyway.
        bool needCast = !idxTy.isInteger(32);
        if (mlir::ShapedType::isDynamic(chunkDim)) {
          // Dynamic chunk dimension: the runtime value is not known at
          // compile time.  Emit the tile index directly — the DMA runtime
          // resolves the actual dimension size.
          if (needCast) os() << "(int)(";
          os() << getName(indices[i]);
          if (needCast) os() << ")";
        } else {
          if (needCast) os() << "(int)(";
          os() << getName(indices[i]) << " * " << chunkDim;
          if (needCast) os() << ")";
        }
      } else {
        os() << "0";
      }
    }
    os() << "};\n";
    return arrName;
  }

  std::string emitSliceShape(TensorTileOp tile, const std::string &prefix) {
    auto tileTy = cast<coir::TensorType>(tile.getResult().getType());
    auto baseTy = cast<coir::TensorType>(tile.getSource().getType());
    std::string arrName = prefix + "__sshape__";
    os() << getIndent() << "unsigned int " << arrName << "[] = {";
    auto indices = tile.getIndices();
    unsigned baseRank = baseTy.getShape().size();
    unsigned dynIdx = 0;
    for (unsigned i = 0; i < tileTy.getShape().size(); ++i) {
      if (i) os() << ", ";
      int64_t d = tileTy.getShape()[i];
      if (mlir::ShapedType::isDynamic(d)) {
        // Dynamic dim: the size value lives in the tile's extra indices
        // (those beyond the base tensor's rank).  The Nth dynamic result
        // dim corresponds to indices[baseRank + N].
        unsigned idxPos = baseRank + dynIdx;
        if (idxPos < indices.size()) {
          // Cast to unsigned int to avoid narrowing in braced-init.
          os() << "(unsigned int)(" << getName(indices[idxPos]) << ")";
        } else
          os() << d; // fallback
        dynIdx++;
      } else {
        os() << d;
      }
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

    os() << getIndent() << getDTEType(op.getSource().getType(),
                                      op.getDest().getType())
       << " " << ctxName << ";\n";
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
        // Flat copy: resolve dynamic copyElems from the dest tensor's
        // dynamic dim (if available) so both mdspans get the right size.
        std::string flatSizeStr;
        if (mlir::ShapedType::isDynamic(copyElems)) {
          auto dstTty = cast<coir::TensorType>(op.getDest().getType());
          if (dstTty.getShape().size() == 1 && dstTty.isDynamicDim(0))
            flatSizeStr = emitDimExpr(op.getDest(), 0);
        }
        if (flatSizeStr.empty())
          flatSizeStr = std::to_string(copyElems);
        std::string srcMds = emitCopyMdspan(op.getSource(), flatSizeStr);
        std::string dstMds = emitCopyMdspan(op.getDest(), flatSizeStr);
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
      // Other DMA kinds (Slice, etc.): also handle dynamic copyElems.
      std::string otherSizeStr;
      if (mlir::ShapedType::isDynamic(copyElems)) {
        auto dstTty = cast<coir::TensorType>(op.getDest().getType());
        if (dstTty.getShape().size() == 1 && dstTty.isDynamicDim(0))
          otherSizeStr = emitDimExpr(op.getDest(), 0);
      }
      if (otherSizeStr.empty())
        otherSizeStr = std::to_string(copyElems);
      std::string srcMds = emitCopyMdspan(op.getSource(), otherSizeStr);
      std::string dstMds = emitCopyMdspan(op.getDest(), otherSizeStr);
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
    if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(op.getPadValueAttr()))
      return std::to_string(intAttr.getInt());
    if (auto fpAttr = mlir::dyn_cast_or_null<FloatAttr>(op.getPadValueAttr()))
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

  void emitAsyncUndef(AsyncUndefOp op) {
    unsigned id = nextDmaId++;
    std::string futName = "__fut_" + std::to_string(id);
    os() << getIndent() << "choreo::future " << futName << ";\n";
    asyncFutures[op.getResult()] = futName;
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

  // Find the kernel argument name for a dynamic dimension of a tensor.
  // Walks back through TensorTileOp to the kernel block argument and
  // uses dimArgMeta + dim_checks to map it to the corresponding dim arg.
  std::string findDynamicDimName(Value tensor, coir::TensorType /*tty*/,
                                 const std::string & /*mdspanSoFar*/) {
    // Walk up through nested region ops (e.g. coir.foreach) to find
    // the enclosing KernelOp.
    auto *parentOp = tensor.getParentBlock()->getParentOp();
    auto kOp = dyn_cast<KernelOp>(parentOp);
    if (!kOp)
      kOp = parentOp->getParentOfType<KernelOp>();
    if (!kOp) return "0";
    auto dimArgMeta = getDimArgs(kOp);
    auto fnType = kOp.getFunctionType();
    auto mrInfo = getMRInfo(kOp);
    unsigned numMrExtra = mrInfo.hasDynamicMR ? mrInfo.numOffsetArgs + 1 : 0;
    unsigned numOrigInputs = fnType.getNumInputs() - dimArgMeta.size() - numMrExtra;
    if (auto pn = kOp->getAttrOfType<mlir::ArrayAttr>("coir.param_names"))
      numOrigInputs = std::min(numOrigInputs,
                               static_cast<unsigned>(pn.size()));

    // Find which original param this tensor belongs to.
    Value baseTensor = tensor;
    if (auto tileOp = tensor.getDefiningOp<TensorTileOp>())
      baseTensor = tileOp.getSource();
    int64_t thisParam = -1;
    if (auto blockArg = dyn_cast<BlockArgument>(baseTensor))
      thisParam = blockArg.getArgNumber();

    // If not a block argument (e.g., result tensor), use the first
    // available dim arg since result typically shares dims with inputs.
    if (thisParam < 0) {
      if (!dimArgMeta.empty())
        return "arg" + std::to_string(numOrigInputs);
      return "0";
    }

    // Direct match: dim arg for this exact param.
    for (unsigned i = 0; i < dimArgMeta.size(); ++i) {
      if (dimArgMeta[i].paramIdx == thisParam)
        return "arg" + std::to_string(numOrigInputs + i);
    }

    // Transitive match: use dim_checks to find a param that shares the
    // same dynamic dim and has a dim arg. Follow chains up to 3 hops.
    auto dimChecks = kOp->getAttrOfType<ArrayAttr>("coir.dim_checks");
    if (dimChecks) {
      llvm::SmallSetVector<int64_t, 8> visited;
      llvm::SmallVector<int64_t, 8> worklist = {thisParam};
      visited.insert(thisParam);
      while (!worklist.empty()) {
        int64_t curParam = worklist.pop_back_val();
        // Check if this param has a dim arg.
        for (unsigned i = 0; i < dimArgMeta.size(); ++i) {
          if (dimArgMeta[i].paramIdx == curParam)
            return "arg" + std::to_string(numOrigInputs + i);
        }
        // Follow dim_checks to find related params.
        for (auto a : dimChecks) {
          auto dict = dyn_cast<DictionaryAttr>(a);
          if (!dict) continue;
          auto p0 = dict.getAs<IntegerAttr>("param0").getInt();
          auto p1 = dict.getAs<IntegerAttr>("param1").getInt();
          int64_t otherParam = (p0 == curParam) ? p1
                               : (p1 == curParam) ? p0 : -1;
          if (otherParam >= 0 && !visited.count(otherParam)) {
            visited.insert(otherParam);
            worklist.push_back(otherParam);
          }
        }
      }
    }

    // Fallback: use first dim arg if available.
    if (!dimArgMeta.empty())
      return "arg" + std::to_string(numOrigInputs);
    return "0";
  }

  void emitDMAConstDesc(DMAConstDescOp op) override {
    unsigned id = nextDmaId++;
    std::string ctxName = "__dma_" + std::to_string(id);
    std::string futName = "__fut_desc_" + std::to_string(id);
    dmaCtxNames[op.getOut()] = futName;

    os() << getIndent() << getDTEType(op.getSource().getType(),
                                      op.getDest().getType())
       << " " << ctxName << ";\n";
    os() << getIndent() << "choreo::future " << futName << "("
       << ctxName << ", \"dma_desc_" << id << "\", 0, 0);\n";

    std::string srcMds = emitMdspanWithShape(op.getSource());
    std::string dstMds = emitMdspanWithShape(op.getDest());

    bool srcTiled = op->hasAttr("src_tiled");
    bool dstTiled = op->hasAttr("dst_tiled");

    auto kind = op.getKind();

    // Helper: generate inline zero-initialized offset list: (int[]){0, 0, ...}
    auto inlineZeroOffsets = [&](unsigned rank) -> std::string {
      std::string result = "(int[]){";
      for (unsigned i = 0; i < rank; ++i) {
        if (i) result += ", ";
        result += "0";
      }
      result += "}";
      return result;
    };

    // Helper: emit pad arrays from forwarded attributes.
    auto emitDescPadArrays = [&](unsigned rank) {
      auto emitArr = [&](const char *suffix, const char *attrName) {
        os() << getIndent() << "unsigned int " << futName << suffix << "[] = {";
        auto attr = op->getAttrOfType<DenseI64ArrayAttr>(attrName);
        if (attr) {
          for (unsigned i = 0; i < attr.size(); ++i) {
            if (i) os() << ", ";
            os() << attr[i];
          }
        } else {
          for (unsigned i = 0; i < rank; ++i) {
            if (i) os() << ", ";
            os() << "0";
          }
        }
        os() << "};\n";
      };
      emitArr("__pad_low__", "pad_low");
      emitArr("__pad_high__", "pad_high");
      emitArr("__pad_mid__", "pad_mid");
    };

    // Helper: emit pad value from forwarded attribute.
    auto emitDescPadValue = [&]() -> std::string {
      if (auto intAttr = op->getAttrOfType<IntegerAttr>("pad_value"))
        return std::to_string(intAttr.getInt());
      if (auto fpAttr = op->getAttrOfType<FloatAttr>("pad_value"))
        return std::to_string(fpAttr.getValueAsDouble());
      return "0";
    };

    // Helper: emit transpose layout from forwarded attribute.
    auto emitDescTransposeLayout = [&]() {
      os() << getIndent() << "int " << futName << "__layout__[] = {";
      auto perm = op->getAttrOfType<DenseI64ArrayAttr>("transpose_perm");
      if (perm) {
        for (unsigned i = 0; i < perm.size(); ++i) {
          if (i) os() << ", ";
          os() << perm[i];
        }
      }
      os() << "};\n";
    };

    if (kind == coir::DMAKind::Copy || kind == coir::DMAKind::Slice) {
      if (srcTiled && !dstTiled) {
        // configure_slice(dst, src, offsets, pad_value=0)
        auto srcTy = cast<coir::TensorType>(op.getSource().getType());
        os() << getIndent() << futName << ".configure_slice(" << dstMds << ", "
           << srcMds << ", " << inlineZeroOffsets(srcTy.getRank()) << ");\n";
      } else if (!srcTiled && dstTiled) {
        // configure_deslice(dst, src, offsets)
        auto dstTy = cast<coir::TensorType>(op.getDest().getType());
        os() << getIndent() << futName << ".configure_deslice(" << dstMds << ", "
           << srcMds << ", " << inlineZeroOffsets(dstTy.getRank()) << ");\n";
      } else if (srcTiled && dstTiled) {
        // configure_slice_deslice(dst, src, src_offsets, slice_shape, dst_offsets)
        auto srcTy = cast<coir::TensorType>(op.getSource().getType());
        unsigned rank = srcTy.getRank();
        // Use tile_shape attribute (actual transfer size) if available,
        // otherwise fall back to source tensor shape.
        llvm::ArrayRef<int64_t> sliceShape;
        llvm::SmallVector<int64_t> sliceShapeBuf;
        if (auto tsAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("tile_shape")) {
          sliceShape = tsAttr.asArrayRef();
        } else {
          sliceShapeBuf.assign(srcTy.getShape().begin(), srcTy.getShape().end());
          sliceShape = sliceShapeBuf;
        }
        os() << getIndent() << "unsigned int " << futName << "__slice__[] = {";
        for (unsigned i = 0; i < rank; ++i) {
          if (i) os() << ", ";
          os() << (i < sliceShape.size() ? sliceShape[i] : 0);
        }
        os() << "};\n";
        os() << getIndent() << futName << ".configure_slice_deslice(" << dstMds
           << ", " << srcMds << ", " << inlineZeroOffsets(rank) << ", " << futName
           << "__slice__, " << inlineZeroOffsets(rank) << ");\n";
      } else {
        // Plain memcpy.
        os() << getIndent() << futName << ".configure_memcpy(" << dstMds << ", "
           << srcMds << ");\n";
      }
    } else if (kind == coir::DMAKind::Pad) {
      auto srcTy = cast<coir::TensorType>(op.getSource().getType());
      unsigned rank = srcTy.getRank();
      if (srcTiled) {
        // configure_slice_pad(dst, src, offsets, slice_shape, pad_low, pad_high,
        //                  pad_mid, pad_value)
        emitDescPadArrays(rank);
        auto dstTy = cast<coir::TensorType>(op.getDest().getType());
        os() << getIndent() << "unsigned int " << futName << "__slice__[] = {";
        for (unsigned i = 0; i < rank; ++i) {
          if (i) os() << ", ";
          os() << dstTy.getShape()[i];
        }
        os() << "};\n";
        os() << getIndent() << futName << ".configure_slice_pad(" << dstMds << ", "
           << srcMds << ", " << inlineZeroOffsets(rank) << ", " << futName << "__slice__, "
           << futName << "__pad_low__, " << futName << "__pad_high__, "
           << futName << "__pad_mid__, " << emitDescPadValue() << ");\n";
      } else {
        // configure_pad(dst, src, pad_low, pad_high, pad_mid, pad_value)
        emitDescPadArrays(rank);
        os() << getIndent() << futName << ".configure_pad(" << dstMds << ", "
           << srcMds << ", " << futName << "__pad_low__, " << futName
           << "__pad_high__, " << futName << "__pad_mid__, "
           << emitDescPadValue() << ");\n";
      }
    } else if (kind == coir::DMAKind::Transpose) {
      auto srcTy = cast<coir::TensorType>(op.getSource().getType());
      unsigned rank = srcTy.getRank();
      emitDescTransposeLayout();
      if (srcTiled && !dstTiled) {
        os() << getIndent() << futName << ".configure_slice_transpose(" << dstMds
           << ", " << srcMds << ", " << inlineZeroOffsets(rank) << ", " << futName
           << "__layout__);\n";
      } else if (!srcTiled && dstTiled) {
        auto dstTy = cast<coir::TensorType>(op.getDest().getType());
        os() << getIndent() << futName << ".configure_transpose_deslice(" << dstMds
           << ", " << srcMds << ", " << futName << "__layout__, "
           << inlineZeroOffsets(dstTy.getRank()) << ");\n";
      } else {
        os() << getIndent() << futName << ".configure_transpose(" << dstMds << ", "
           << srcMds << ", " << futName << "__layout__);\n";
      }
    } else {
      // Fallback: plain memcpy.
      os() << getIndent() << futName << ".configure_memcpy(" << dstMds << ", "
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

    bool useDstOffset = op->hasAttr("dst_offsets");
    auto offsets = op.getOffsets();
    for (unsigned i = 0; i < offsets.size(); ++i) {
      os() << getIndent() << futName << "."
         << (useDstOffset ? "set_dst_offset" : "set_src_offset")
         << "(" << i << ", " << getName(offsets[i]) << ");\n";
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
    auto ms = tty.getMemorySpace();
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local))
      return "__local__ ";
    if (ms == static_cast<int32_t>(coir::TensorMemorySpace::Shared))
      return "__shared__ ";
    return "";
  }

  void emitReuseInit(TensorAllocOp op, coir::TensorType tty,
                      const std::string &name) {
    auto initAttr = op.getInit();
    if (!initAttr) return;
    auto ms = tty.getMemorySpace();
    bool isLocal =
        (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local));
    std::string space = isLocal ? "tops::Private" : "tops::Shared";
    std::string initVal;
    mlir::TypedAttr attr = *initAttr;
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      initVal = std::to_string(ia.getInt());
    } else if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
      llvm::SmallString<16> buf;
      fa.getValue().toString(buf);
      initVal = std::string(buf);
      if (initVal.find('.') == std::string::npos &&
          initVal.find('e') == std::string::npos)
        initVal += ".0";
      if (tty.getElementType().isF16() || tty.getElementType().isBF16())
        initVal = "(" + emitType(tty.getElementType()) + ")" + initVal;
    }
    std::string mds = "tops::mdspan(" + space + ", (" +
                      emitType(tty.getElementType()) + "*)" + name;
    for (auto d : tty.getShape())
      mds += ", " + std::to_string(d);
    mds += ")";
    unsigned id = nextId++;
    std::string ctxName = "__dte_init_" + std::to_string(id);
    os() << getIndent() << "{\n";
    incIndent();
    os() << getIndent() << "choreo::choreo_sdte " << ctxName << ";\n";
    if (needsExplicitInit())
      os() << getIndent() << ctxName << ".init();\n";
    os() << getIndent() << "tops::memset(" << ctxName << ", " << mds
         << ", " << initVal << ");\n";
    decIndent();
    os() << getIndent() << "}\n";
  }

  void emitTensorAlloc(TensorAllocOp op) override {
    if (returnValues.count(op.getResult())) return;

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    std::string name = getName(op.getResult());

    // Host-side tensor alloc: emit as nullptr (used only for address printing).
    if (isEmittingHost_) {
      os() << getIndent() << emitType(tensorTy.getElementType()) << "* "
           << name << " = nullptr;\n";
      hostTensorAllocs_.insert(op.getResult());
      return;
    }

    // -- Memory reuse: buffer is aliased into a shared SPM at a given offset --
    if (op.getReuseOffsetAttr()) {
      // Dynamic offset: use kernel parameter instead of constant.
      // Only if the offset name is an actual kernel parameter; otherwise
      // fall through to the static offset path (local buffers inside
      // nested inthreads may have dyn_offset_arg set but no corresponding
      // kernel parameter).
      if (auto dynArgAttr =
              op->getAttrOfType<mlir::StringAttr>("dyn_offset_arg")) {
        // Use dynamic path if: (a) mrOffsetParamNames_ is empty (non-device
        // function, all dyn offsets are kernel params), or (b) the name IS
        // a kernel param. Fall through to static path only for local buffers
        // whose dyn_offset_arg is NOT a kernel parameter.
        if (mrOffsetParamNames_.empty() ||
            mrOffsetParamNames_.count(dynArgAttr.getValue())) {
          if (!dynSpmEmitted_) {
            bool isLocal =
                (tensorTy.getMemorySpace() ==
                 static_cast<int32_t>(coir::TensorMemorySpace::Local));
            if (isLocal) {
              os() << getIndent()
                   << "__local__ unsigned char __dyn_smem[__CO_DYN_SMEM_SIZE];"
                      "\n";
              os() << getIndent()
                   << "choreo::choreo_assert(__co__local_spm_size <= "
                      "__CO_DYN_SMEM_SIZE, \"dynamic local memory reuse "
                      "exceeds __CO_DYN_SMEM_SIZE\");\n";
            } else {
              std::string qual = getAllocQualifier(tensorTy);
              os() << getIndent() << "extern " << qual
                   << "unsigned char __dyn_smem[];\n";
            }
            dynSpmEmitted_ = true;
          }
          std::string offName = dynArgAttr.getValue().str();
          std::string eType = emitElementType(tensorTy.getElementType());
          os() << getIndent() << eType << "* " << name << " = (" << eType
               << "*)((unsigned char*)__dyn_smem + " << offName << ");\n";
          emitReuseInit(op, tensorTy, name);
          return;
        }
        // dyn_offset_arg not a kernel param — fall through to static path
      }

      // Static offset: emit pool array on first use, then pointer alias.
      auto poolNameOpt = op.getReuseSpm();
      llvm::StringRef poolName =
          poolNameOpt.has_value() ? *poolNameOpt : "__default_spm";
      std::string &spmVar = spmNames_[poolName];
      if (spmVar.empty()) {
        spmVar = "__spm_" + std::to_string(nextId++);
        int64_t spmBytes = 0;
        if (auto spmSizeAttr =
                op->getAttrOfType<mlir::IntegerAttr>("spm_size"))
          spmBytes = spmSizeAttr.getInt();
        std::string qual = getAllocQualifier(tensorTy);
        os() << getIndent() << qual << "unsigned char "
             << spmVar << "[" << spmBytes << "];\n";
      }
      int64_t offset = op.getReuseOffset().value_or(0);
      std::string eType = emitElementType(tensorTy.getElementType());
      os() << getIndent() << eType << "* " << name << " = (" << eType
           << "*)((unsigned char*)" << spmVar << " + " << offset << ");\n";
      emitReuseInit(op, tensorTy, name);
      return;
    }

    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape()) totalElems *= d;

    std::string qualifier = getAllocQualifier(tensorTy);
    os() << getIndent() << qualifier << emitType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";

    if (auto initAttr = op.getInit()) {
      auto ms = tensorTy.getMemorySpace();
      bool isLocal =
          (ms == static_cast<int32_t>(coir::TensorMemorySpace::Local));
      std::string space = isLocal ? "tops::Private" : "tops::Shared";
      std::string initVal;
      mlir::TypedAttr attr = *initAttr;
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        initVal = std::to_string(ia.getInt());
      } else if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
        llvm::SmallString<16> buf;
        fa.getValue().toString(buf);
        initVal = std::string(buf);
        if (initVal.find('.') == std::string::npos &&
            initVal.find('e') == std::string::npos)
          initVal += ".0";
        if (tensorTy.getElementType().isF16() ||
            tensorTy.getElementType().isBF16())
          initVal = "(" + emitType(tensorTy.getElementType()) + ")" + initVal;
      }
      std::string mds = "tops::mdspan(" + space + ", (" +
                        emitType(tensorTy.getElementType()) + "*)" + name;
      for (auto d : tensorTy.getShape())
        mds += ", " + std::to_string(d);
      mds += ")";

      os() << getIndent() << "{\n";
      incIndent();
      os() << getIndent() << "choreo::choreo_sdte " << name << "__init;\n";
      if (needsExplicitInit())
        os() << getIndent() << name << "__init.init();\n";
      os() << getIndent() << "tops::memset(" << name << "__init, "
           << mds << ", " << initVal << ");\n";
      decIndent();
      os() << getIndent() << "}\n";
    }
  }

  void emitBarrier(BarrierOp op) override {
    switch (op.getScope()) {
    case coir::ParallelLevel::BLOCK:
      os() << getIndent() << "__syncthreads();\n";
      break;
    case coir::ParallelLevel::GROUP:
    case coir::ParallelLevel::GROUPx4:
      os() << getIndent() << "__syncsubthreads();\n";
      break;
    case coir::ParallelLevel::DEVICE:
      os() << getIndent() << "choreo::abend_true(topsDeviceSynchronize());\n";
      break;
    default:
      os() << getIndent() << "__syncthreads();\n";
      break;
    }
  }

  std::string emitExprInHostScope(
      Value v, KernelOp kernel,
      DenseMap<Value, std::string> &hostNames,
      const llvm::SmallVectorImpl<DimArgMeta> &dimArgMeta) {
    auto it = hostNames.find(v);
    if (it != hostNames.end()) return it->second;

    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (arg.getOwner()->getParentOp() == kernel.getOperation()) {
        unsigned idx = arg.getArgNumber();
        // Map dim arg block arguments to host expressions.
        // Dim args come after original inputs; find matching metadata.
        auto fnType = kernel.getFunctionType();
        unsigned numInputs = fnType.getNumInputs();
        if (idx < numInputs) {
          auto inTy = fnType.getInput(idx);
          if (inTy.isIndex() || inTy.isInteger(32) || inTy.isInteger(64)) {
            // This is a dim arg — find its paramIdx/dimIdx.
            unsigned dimArgIdx = 0;
            for (unsigned i = 0; i < numInputs; ++i) {
              auto ty = fnType.getInput(i);
              if (ty.isIndex() || ty.isInteger(32) || ty.isInteger(64)) {
                if (i == idx && dimArgIdx < dimArgMeta.size()) {
                  auto &da = dimArgMeta[dimArgIdx];
                  std::string name = "p" + std::to_string(da.paramIdx) +
                                     ".shape()[" +
                                     std::to_string(da.dimIdx) + "]";
                  hostNames[v] = name;
                  return name;
                }
                ++dimArgIdx;
              }
            }
          }
        }
        std::string name = "p" + std::to_string(idx);
        hostNames[v] = name;
        return name;
      }
    }

    auto *defOp = v.getDefiningOp();
    if (!defOp) return "/* unknown */";

    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        std::string r = std::to_string(intAttr.getInt());
        hostNames[v] = r;
        return r;
      }
    }
    if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
      return emitExprInHostScope(castOp.getIn(), kernel, hostNames, dimArgMeta);

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      auto lhs = emitExprInHostScope(cmpOp.getLhs(), kernel, hostNames, dimArgMeta);
      auto rhs = emitExprInHostScope(cmpOp.getRhs(), kernel, hostNames, dimArgMeta);
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
      auto lhs = emitExprInHostScope(addOp.getLhs(), kernel, hostNames, dimArgMeta);
      auto rhs = emitExprInHostScope(addOp.getRhs(), kernel, hostNames, dimArgMeta);
      std::string result = "(" + lhs + " + " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      auto lhs = emitExprInHostScope(mulOp.getLhs(), kernel, hostNames, dimArgMeta);
      auto rhs = emitExprInHostScope(mulOp.getRhs(), kernel, hostNames, dimArgMeta);
      std::string result = "(" + lhs + " * " + rhs + ")";
      hostNames[v] = result;
      return result;
    }

    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames, dimArgMeta);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames, dimArgMeta);
      std::string result = "(" + lhs + " / " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames, dimArgMeta);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames, dimArgMeta);
      std::string result = "(" + lhs + " / " + rhs + ")";
      hostNames[v] = result;
      return result;
    }

    return "/* unknown */";
  }

  void emitEntryAssertions(KernelOp kernel) {
    auto dimArgMeta = getDimArgs(kernel);
    DenseMap<Value, std::string> hostNames;
    for (auto &ea : entryAssertions) {
      auto cond = emitExprInHostScope(ea.op.getCondition(), kernel, hostNames,
                                      dimArgMeta);
      // TODO: workaround
      // Skip assertions referencing values not resolvable in host scope.
      if (cond.find("/* unknown */") != std::string::npos) continue;
      os() << "  choreo::runtime_check(" << cond << ", \""
         << ea.op.getMessage().str() << "\");\n";
    }
  }

  void emitAssert(AssertOp op) {
    if (auto ea = op->getAttrOfType<BoolAttr>("enabled"))
      if (!ea.getValue()) return;
    auto site = op.getSite();
    if (site == AssertSite::ENTRY) {
      entryAssertions.push_back({op});
      return;
    }
    os() << getIndent() << "choreo::choreo_assert("
       << getName(op.getCondition()) << ", \"" << op.getMessage().str()
       << "\");\n";
  }

  void emitElementCopy(ElementCopyOp op) {
    auto srcTy = cast<coir::TensorType>(op.getSource().getType());
    int64_t totalElems = 1;
    for (auto d : srcTy.getShape()) totalElems *= d;
    std::string src = getName(op.getSource());
    std::string dst = getName(op.getDest());
    std::string idx = "i" + std::to_string(nextId++);
    // Cooperative block-parallel copy: each thread handles a stride
    os() << getIndent() << "for (int " << idx << " = __tops_tid_x(); " << idx
       << " < " << totalElems << "; " << idx << " += __tops_num_threads())\n";
    os() << getIndent() << "  " << dst << "[" << idx << "] = " << src
       << "[" << idx << "];\n";
    os() << getIndent() << "__syncthreads();\n";
  }

  void emitTensorStoreTile(TensorStoreTileOp op) {
    // If the tile came from tensor.tile (pointer alias into dest), the write
    // already went through the alias -- store_tile is a no-op.
    if (op.getTile().getDefiningOp<TensorTileOp>()) return;

    // Otherwise the tile is from an independent allocation (tensor.alloc) and
    // we must copy its contents back into dest at the given offset.
    auto tileTy = cast<coir::TensorType>(op.getTile().getType());
    auto destTy = cast<coir::TensorType>(op.getDest().getType());
    int64_t tileElems = 1;
    for (auto d : tileTy.getShape()) tileElems *= d;

    std::string tile = getName(op.getTile());
    std::string dest = getName(op.getDest());

    auto destShape = destTy.getShape();
    std::string offset;
    auto indices = op.getIndices();
    for (unsigned i = 0; i < indices.size(); ++i) {
      int64_t stride = 1;
      for (unsigned j = i + 1; j < destShape.size(); ++j)
        stride *= destShape[j];
      std::string term = getName(indices[i]);
      if (stride != 1)
        term += " * " + std::to_string(stride);
      if (offset.empty())
        offset = term;
      else
        offset += " + " + term;
    }
    if (offset.empty()) offset = "0";

    std::string idx = "i" + std::to_string(nextId++);
    os() << getIndent() << "for (int " << idx << " = 0; " << idx << " < "
         << tileElems << "; ++" << idx << ")\n";
    os() << getIndent() << "  " << dest << "[" << offset << " + " << idx
         << "] = " << tile << "[" << idx << "];\n";
  }

  void emitTmaCopyDiagnostic(TmaCopyOp op) {
    op.emitError("topscc target does not support coir.tma.copy; "
                 "use coir.dma.copy or lower via ConvertToGCU pass");
    os() << getIndent() << "#error \"coir.tma.copy unsupported on topscc\"\n";
  }

  void emitEventTrigger(EventTriggerOp op) {
    auto name = op.getEventName().str();
    ensureEventDeclared(name);
    os() << getIndent() << name;
    if (auto sub = op.getSubscript())
      os() << "[" << sub->str() << "]";
    os() << " = true;\n";
  }

  void emitEventWait(EventWaitOp op) {
    auto name = op.getEventName().str();
    ensureEventDeclared(name);
    std::string ref = name;
    if (auto sub = op.getSubscript())
      ref += "[" + sub->str() + "]";
    os() << getIndent() << "while (" << ref << " == false) continue;\n";
    os() << getIndent() << ref << " = false;\n";
  }

  // Lazily declare event variables on first use (fallback if not pre-scanned).
  DenseSet<StringRef> declaredEvents;
  void ensureEventDeclared(const std::string &name) {
    // Events should already be declared by the parallel block pre-scan.
    // This is a fallback for events used outside parallel blocks.
    if (declaredEvents.insert(name).second) {
      os() << getIndent() << "__shared__ __volatile__ bool " << name << ";\n";
      os() << getIndent() << "if (__tops_tid_x() == 0) " << name << " = false;\n";
      os() << getIndent() << "__syncthreads();\n";
    }
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
    bool isExpr = op.getIsExpr() && *op.getIsExpr() && op.getResult();
    bool isBif = op.getIsBif() && *op.getIsBif();

    std::string funcName = callee;
    if (isBif) {
      llvm::StringRef ref(callee);
      if (ref.starts_with("__")) {
        // Arithmetic builtins: strip __ prefix and map to tcle:: namespace.
        ref = ref.drop_front(2);
        if (ref == "log")
          funcName = "tcle::ln";
        else if (ref == "pow")
          funcName = "tcle::power";
        else
          funcName = ("tcle::" + ref).str();
      }
      // Non-arith builtins (println, assert, etc.): use name as-is.
    }

    // Handle println/print: map to printf with format string for device code.
    if (callee == "println" || callee == "print") {
      auto args = op.getOperands_();
      std::string fmt;

      // Use the format_str attribute if available (built from AST string
      // literals). Otherwise fall back to a generic type-based format.
      if (auto fmtAttr = op->getAttrOfType<mlir::StringAttr>("format_str")) {
        fmt = fmtAttr.getValue().str();
      } else {
        for (size_t i = 0; i < args.size(); ++i) {
          if (i > 0) fmt += " ";
          auto ty = args[i].getType();
          if (mlir::isa<mlir::FloatType>(ty))
            fmt += "%f";
          else
            fmt += "%lld";
        }
        if (callee == "println") fmt += "\\n";
      }

      // gcu200/210 printf doesn't support %s; use %d with 1/0 instead.
      bool noStringPrintf = (archNum < 300);
      if (noStringPrintf) {
        std::string newFmt;
        for (size_t i = 0; i < fmt.size(); ++i) {
          if (fmt[i] == '%' && i + 1 < fmt.size() && fmt[i + 1] == 's') {
            newFmt += "%d";
            ++i; // skip 's'
          } else {
            newFmt += fmt[i];
          }
        }
        fmt = newFmt;
      }

      os() << getIndent() << "printf(\"" << fmt << "\"";
      // Parse format specifiers to emit type-appropriate casts.
      size_t fmtPos = 0;
      for (auto arg : args) {
        auto ty = arg.getType();
        os() << ", ";
        // Find the next format specifier in the format string.
        while (fmtPos < fmt.size() && fmt[fmtPos] != '%') fmtPos++;
        bool isBoolFmt = false;
        bool isFloatFmt = false;
        bool isPtrFmt = false;
        if (fmtPos < fmt.size()) {
          if (fmtPos + 1 < fmt.size() && fmt[fmtPos + 1] == 'd' && noStringPrintf)
            isBoolFmt = true; // %d was originally %s for bool
          else if (fmtPos + 1 < fmt.size() && fmt[fmtPos + 1] == 's')
            isBoolFmt = true;
          else if (fmtPos + 1 < fmt.size() && fmt[fmtPos + 1] == 'f')
            isFloatFmt = true;
          else if (fmtPos + 1 < fmt.size() && fmt[fmtPos + 1] == 'p')
            isPtrFmt = true;
          fmtPos += 2; // skip %X
        }
        if (isBoolFmt) {
          if (noStringPrintf)
            os() << "(" << getName(arg) << " ? 1 : 0)";
          else if (args.empty() || !arg)
            os() << "\"false\"";
          else
            os() << "(" << getName(arg) << " ? \"true\" : \"false\")";
        } else if (isPtrFmt) {
          if (isEmittingHost_ && mlir::isa<coir::TensorType>(ty)) {
            auto base = getTensorDefOp(arg);
            if (!hostTensorAllocs_.count(base))
              os() << "(void*)" << getName(base) << ".data()";
            else
              os() << "(void*)" << getName(base);
          } else if (mlir::isa<coir::TensorType>(ty)) {
            os() << "(void*)" << getName(getTensorDefOp(arg));
          } else {
            os() << "(void*)" << getName(arg);
          }
        } else if (ty.isF16())
          os() << "f16_to_f32(" << getName(arg) << ")";
        else if (ty.isBF16())
          os() << "(float)" << getName(arg);
        else if (mlir::isa<mlir::FloatType>(ty) || isFloatFmt)
          os() << "(double)" << getName(arg);
        else
          os() << "(long long)" << getName(arg);
      }
      os() << ");\n";
      return;
    }

    os() << getIndent();
    if (isExpr) {
      auto resTy = op.getResult().getType();
      os() << emitType(resTy) << " " << getName(op.getResult()) << " = ";
    }

    os() << funcName;

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
        // bind_dims results are materialized by emitTensorBindDims as
        // aliases to the underlying tensor, so getName() works directly.
        os() << "(" << emitType(tty.getElementType()) << "*)"
             << getName(arg);
        if (isEmittingHost_ && !hostTensorAllocs_.count(arg))
          os() << ".data()";
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

  void emitTensorReduceElem(TensorReduceElemOp op) override {
    std::string dst = getName(op.getDest());
    std::string val = getName(op.getValue());
    auto destTy = cast<coir::TensorType>(op.getDest().getType());
    bool isAtomic = op->hasAttr("atomic");
    if (isAtomic) {
      os() << getIndent() << "atomicAdd(&" << dst << "[";
      emitLinearIndex(op.getIndices(), destTy);
      os() << "], " << val << ");\n";
    } else {
      os() << getIndent() << dst << "[";
      emitLinearIndex(op.getIndices(), destTy);
      os() << "] += " << val << ";\n";
    }
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
    auto destTy = cast<coir::TensorType>(op.getDest().getType());
    emitLinearIndex(op.getIndices(), destTy);
    os() << "], " << getName(op.getValue());
    if (op.getKind() == AK::CAS && op.getCompare()) {
      auto cmpAttr = *op.getCompare();
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cmpAttr))
        os() << ", " << ia.getInt();
      else if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(cmpAttr))
        os() << ", " << fa.getValueAsDouble();
      else
        os() << ", /* compare */";
    }
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
  return true;
}();

} // namespace
