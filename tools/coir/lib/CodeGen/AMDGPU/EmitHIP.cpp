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

class HIPEmitter {
public:
  HIPEmitter(llvm::raw_ostream &os) : os(os), indent(0) {}

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

  struct EntryAssertion {
    AssertOp op;
  };
  llvm::SmallVector<EntryAssertion> entryAssertions;
  bool hasGroupLevel = false;
  int64_t groupThreadScale = 0;
  llvm::DenseSet<mlir::Value> dmaTokens;

  struct DescInfo {
    unsigned idx;
    coir::TensorType srcType;
    coir::TensorType dstType;
    DMAKind kind;
    bool isTMA;
    bool isLoad;
    Value constDescResult;
  };
  llvm::SmallVector<DescInfo> descInfos;
  DenseMap<Value, unsigned> descValueToIndex;
  DenseMap<Value, SmallVector<Value, 4>> descRuntimeOffsets;

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

  std::string emitElementType(Type ty) {
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
    os << "#define __CHOREO_TARGET_AMDGPU__ 1\n";
    os << "#include \"choreo.h\"\n";
    os << "#include <hip/hip_runtime.h>\n\n";
  }

  std::string kernelDeviceName(StringRef name) {
    return ("__" + name + "_kernel__").str();
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
        else if (eTyML.isBF16()) choreoElem = "choreo::bf16";
        else if (eTyML.isF32()) choreoElem = "choreo::f32";
        else if (eTyML.isF16()) choreoElem = "choreo::f16";
        else if (eTyML.isF64()) choreoElem = "choreo::f64";
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
    if (ty.isF16()) return "__half";
    if (ty.isIndex()) return "int";
    return "/* unknown */";
  }

  int64_t getTensorBytes(coir::TensorType tty) {
    int64_t n = 1;
    for (auto d : tty.getShape()) {
      if (mlir::ShapedType::isDynamic(d))
        return -1;
      n *= d;
    }
    Type eTy = tty.getElementType();
    int64_t elemSize = 4;
    if (eTy.isF16() || eTy.isBF16() || eTy.isInteger(16)) elemSize = 2;
    else if (eTy.isF64() || eTy.isInteger(64)) elemSize = 8;
    else if (eTy.isInteger(8)) elemSize = 1;
    return n * elemSize;
  }

  void prescanDescriptors(KernelOp kernel) {
    descInfos.clear();
    descValueToIndex.clear();
    descRuntimeOffsets.clear();

    kernel.getBody().walk([&](DMAConstDescOp op) {
      auto srcType = dyn_cast<coir::TensorType>(op.getSource().getType());
      auto dstType = dyn_cast<coir::TensorType>(op.getDest().getType());
      if (!srcType || !dstType) return;

      int32_t srcMS = srcType.getMemorySpace();
      int32_t dstMS = dstType.getMemorySpace();
      bool isLoad = (srcMS <= 0) && (dstMS == 1);

      unsigned idx = descInfos.size();
      descInfos.push_back(
          {idx, srcType, dstType, op.getKind(), op.getTma(), isLoad,
           op.getOut()});
      descValueToIndex[op.getOut()] = idx;
    });
  }

  void emitDMAConstDesc(DMAConstDescOp op) {
    auto it = descValueToIndex.find(op.getOut());
    if (it == descValueToIndex.end()) return;
    valueNames[op.getOut()] = "__desc_" + std::to_string(it->second);
  }

  void emitDMAPrefetch(DMADescPrefetchOp op) {
    auto it = descValueToIndex.find(op.getIn());
    if (it == descValueToIndex.end()) {
      valueNames[op.getOut()] = "/* prefetch unknown */";
      return;
    }
    unsigned descIdx = it->second;
    descValueToIndex[op.getOut()] = descIdx;
    valueNames[op.getOut()] = "__desc_" + std::to_string(descIdx);
  }

  void emitDMARuntimeDesc(DMADescRuntimeOp op) {
    auto it = descValueToIndex.find(op.getIn());
    if (it == descValueToIndex.end()) {
      valueNames[op.getOut()] = "/* rt_desc unknown */";
      return;
    }
    unsigned descIdx = it->second;
    descValueToIndex[op.getOut()] = descIdx;
    valueNames[op.getOut()] = "__desc_" + std::to_string(descIdx);
    SmallVector<Value, 4> offs(op.getOffsets().begin(), op.getOffsets().end());
    descRuntimeOffsets[op.getOut()] = std::move(offs);
  }

  void emitDMAInvoke(DMAInvokeOp op) {
    dmaTokens.insert(op.getDone());

    Value descVal = op.getDesc();
    auto it = descValueToIndex.find(descVal);
    if (it == descValueToIndex.end()) {
      os << getIndent() << "// DMA invoke (no descriptor info)\n";
      os << getIndent() << "__syncthreads();\n";
      return;
    }

    unsigned descIdx = it->second;
    auto &desc = descInfos[descIdx];
    auto constOp = desc.constDescResult.getDefiningOp<DMAConstDescOp>();
    if (!constOp) {
      os << getIndent() << "__syncthreads();\n";
      return;
    }

    Value srcVal = constOp.getSource();
    Value dstVal = constOp.getDest();

    // Handle runtime offsets: slice into the global tensor
    SmallVector<Value, 4> offsets;
    auto offsIt = descRuntimeOffsets.find(descVal);
    if (offsIt != descRuntimeOffsets.end())
      offsets = offsIt->second;

    if (!offsets.empty()) {
      Value globalVal = desc.isLoad ? srcVal : dstVal;
      Value localVal = desc.isLoad ? dstVal : srcVal;
      auto localTy = dyn_cast<coir::TensorType>(localVal.getType());
      auto globalTy = dyn_cast<coir::TensorType>(globalVal.getType());
      if (localTy && globalTy) {
        int64_t tileElems = 1;
        for (auto d : localTy.getShape()) tileElems *= d;
        std::string elemTy = emitElementType(globalTy.getElementType());
        std::string ptrName = getName(globalVal);
        std::string offExpr = getName(offsets[0]);
        if (tileElems > 1)
          offExpr += " * " + std::to_string(tileElems);
        unsigned id = nextId++;
        std::string slicedPtr = "__slice_ptr_" + std::to_string(id);
        os << getIndent() << elemTy << "* " << slicedPtr << " = ("
           << elemTy << "*)" << ptrName << " + " << offExpr << ";\n";

        // Register sliced pointer as name for the global val in this copy
        std::string origName = getName(globalVal);
        if (desc.isLoad)
          valueNames[srcVal] = slicedPtr;
        else
          valueNames[dstVal] = slicedPtr;
      }
    }

    // Emit the actual copy based on descriptor kind
    switch (desc.kind) {
    case DMAKind::Pad:
      emitCopyWithPad(srcVal, dstVal, std::nullopt, std::nullopt, std::nullopt);
      break;
    case DMAKind::Transpose:
      emitCopyWithTranspose(srcVal, dstVal, std::nullopt);
      break;
    default:
      emitCooperativeCopy(srcVal, dstVal);
      break;
    }
    os << getIndent() << "__syncthreads();\n";

    // Restore original name if we overwrote it
    if (!offsets.empty()) {
      if (desc.isLoad)
        valueNames.erase(srcVal);
      else
        valueNames.erase(dstVal);
    }
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
      } else if (lvl == coir::ParallelLevel::GROUP) {
        int64_t nWarps = 1;
        for (auto b : bounds) nWarps *= b;
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

  void emitKernel(KernelOp kernel) {
    entryAssertions.clear();
    hasGroupLevel = false;
    groupThreadScale = 0;
    dmaTokens.clear();
    prescanDescriptors(kernel);

    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    os << "__global__ ";
    if (auto lb = kernel.getLaunchBoundsAttr()) {
      if (lb.getMaxThreadsPerBlock() > 0) {
        os << "__launch_bounds__(" << lb.getMaxThreadsPerBlock();
        if (lb.getMinBlocksPerMultiprocessor() > 0)
          os << ", " << lb.getMinBlocksPerMultiprocessor();
        os << ") ";
      }
    }
    if (auto nr = kernel.getMaxNregAttr()) {
      if (nr.getValue() > 0)
        os << "__attribute__((amdgpu_num_vgpr(" << nr.getValue()
           << "))) ";
    }
    os << "void " << devName << "(";

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

  std::string getDynShmemExpr(KernelOp kernel,
                              llvm::ArrayRef<DimArgMeta> dimArgMeta = {}) {
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

  void emitHostWrapper(KernelOp kernel) {
    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    unsigned numResults = fnType.getNumResults();
    auto dimArgMeta = getDimArgs(kernel);
    unsigned numOrigInputs = fnType.getNumInputs() - dimArgMeta.size();

    auto streamName = getStreamName(kernel);

    llvm::SmallVector<llvm::StringRef> hostElemHints;
    if (auto attr =
            kernel->getAttrOfType<mlir::ArrayAttr>("coir.host_elem_types"))
      for (auto a : attr)
        hostElemHints.push_back(cast<mlir::StringAttr>(a).getValue());

    if (numResults == 0) {
      os << "void " << symName << "(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os << ", ";
        llvm::StringRef hint =
            (i < hostElemHints.size()) ? hostElemHints[i] : "";
        os << emitChoreoType(fnType.getInput(i), true, hint) << " p" << i;
      }
      if (!streamName.empty()) {
        if (numOrigInputs > 0) os << ", ";
        os << "hipStream_t " << streamName;
      }
      os << ") {\n";

      for (unsigned i = 0; i < numOrigInputs; ++i) {
        auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
        if (!tty) continue;
        std::string eType = emitElementType(tty.getElementType());
        int64_t bytes = getTensorBytes(tty);
        os << "  " << eType << "* p" << i << "__device = nullptr;\n";
        if (bytes < 0) {
          os << "  (void)hipMalloc(&p" << i << "__device, p" << i
             << ".element_count() * sizeof(" << eType << "));\n";
          os << "  (void)hipMemcpy(p" << i << "__device, p" << i
             << ".data(), p" << i << ".element_count() * sizeof(" << eType
             << "), hipMemcpyHostToDevice);\n";
        } else {
          os << "  (void)hipMalloc(&p" << i << "__device, " << bytes
             << "ULL);\n";
          os << "  (void)hipMemcpy(p" << i << "__device, p" << i
             << ".data(), " << bytes << "ULL, hipMemcpyHostToDevice);\n";
        }
      }

      auto dims = getLaunchDims(kernel);
      auto dynShmem = getDynShmemExpr(kernel, dimArgMeta);
      emitEntryAssertions(kernel, numOrigInputs, dimArgMeta);
      emitDimChecks(kernel);
      bool asyncLaunch = isAsyncLaunch(kernel);

      os << "  " << devName << "<<<" << dims.gridStr() << ", "
         << dims.blockStr();
      if (!dynShmem.empty() || !streamName.empty())
        os << ", " << (dynShmem.empty() ? "0" : dynShmem);
      if (!streamName.empty())
        os << ", " << streamName;
      os << ">>>(";
      for (unsigned i = 0; i < numOrigInputs; ++i) {
        if (i > 0) os << ", ";
        os << "p" << i << "__device";
      }
      for (auto &da : dimArgMeta) {
        os << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
      }
      os << ");\n";
      if (!asyncLaunch) {
        if (!streamName.empty())
          os << "  (void)hipStreamSynchronize(" << streamName << ");\n";
        else
          os << "  (void)hipDeviceSynchronize();\n";
      }

      for (unsigned i = 0; i < numOrigInputs; ++i)
        os << "  (void)hipFree(p" << i << "__device);\n";
      os << "}\n\n";
      return;
    }

    auto resTy = dyn_cast<coir::TensorType>(fnType.getResult(0));
    if (!resTy) return;

    llvm::StringRef retHint;
    if (auto attr =
            kernel->getAttrOfType<mlir::StringAttr>("coir.host_ret_type"))
      retHint = attr.getValue();

    os << emitChoreoType(fnType.getResult(0), false, retHint) << " "
       << symName << "(";
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (i > 0) os << ", ";
      llvm::StringRef hint =
          (i < hostElemHints.size()) ? hostElemHints[i] : "";
      os << emitChoreoType(fnType.getInput(i), true, hint) << " p" << i;
    }
    if (!streamName.empty()) {
      if (numOrigInputs > 0) os << ", ";
      os << "hipStream_t " << streamName;
    }
    os << ") {\n";

    std::string eType = emitElementType(resTy.getElementType());

    for (unsigned i = 0; i < numOrigInputs; ++i) {
      auto tty = dyn_cast<coir::TensorType>(fnType.getInput(i));
      if (!tty) continue;
      std::string inputEType = emitElementType(tty.getElementType());
      int64_t bytes = getTensorBytes(tty);
      os << "  " << inputEType << "* p" << i << "__device = nullptr;\n";
      if (bytes < 0) {
        os << "  (void)hipMalloc(&p" << i << "__device, p" << i
           << ".element_count() * sizeof(" << inputEType << "));\n";
        os << "  (void)hipMemcpy(p" << i << "__device, p" << i
           << ".data(), p" << i << ".element_count() * sizeof(" << inputEType
           << "), hipMemcpyHostToDevice);\n";
      } else {
        os << "  (void)hipMalloc(&p" << i << "__device, " << bytes
           << "ULL);\n";
        os << "  (void)hipMemcpy(p" << i << "__device, p" << i << ".data(), "
           << bytes << "ULL, hipMemcpyHostToDevice);\n";
      }
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
    if (!retHint.empty()) {
      choreoElem = "choreo::" + retHint.str();
    } else {
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
    }

    os << "  auto __result = choreo::make_spandata<" << choreoElem << ", "
       << resTy.getShape().size() << ">(" << shapeStr << ");\n";
    os << "  " << eType << "* __result__device = nullptr;\n";
    if (resBytes < 0) {
      os << "  (void)hipMalloc(&__result__device, __result.element_count()"
         << " * sizeof(" << eType << "));\n";
    } else {
      os << "  (void)hipMalloc(&__result__device, " << resBytes
         << "ULL);\n";
    }

    auto dims = getLaunchDims(kernel);
    auto dynShmem = getDynShmemExpr(kernel, dimArgMeta);
    emitEntryAssertions(kernel, numOrigInputs, dimArgMeta);
    emitDimChecks(kernel);
    bool asyncLaunch = isAsyncLaunch(kernel);

    os << "  " << devName << "<<<" << dims.gridStr() << ", "
       << dims.blockStr();
    if (!dynShmem.empty() || !streamName.empty())
      os << ", " << (dynShmem.empty() ? "0" : dynShmem);
    if (!streamName.empty())
      os << ", " << streamName;
    os << ">>>(";
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if (i > 0) os << ", ";
      os << "p" << i << "__device";
    }
    for (auto &da : dimArgMeta) {
      os << ", (int)p" << da.paramIdx << ".shape()[" << da.dimIdx << "]";
    }
    os << ", __result__device);\n";
    if (!asyncLaunch) {
      if (!streamName.empty())
        os << "  (void)hipStreamSynchronize(" << streamName << ");\n";
      else
        os << "  (void)hipDeviceSynchronize();\n";
    }
    if (resBytes < 0) {
      os << "  (void)hipMemcpy(__result.data(), __result__device, "
         << "__result.element_count() * sizeof(" << eType
         << "), hipMemcpyDeviceToHost);\n";
    } else {
      os << "  (void)hipMemcpy(__result.data(), __result__device, "
         << resBytes << "ULL, hipMemcpyDeviceToHost);\n";
    }

    for (unsigned i = 0; i < numOrigInputs; ++i)
      os << "  (void)hipFree(p" << i << "__device);\n";
    os << "  (void)hipFree(__result__device);\n";
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
    else if (isa<MMAFillOp>(op) || isa<MMALoadOp>(op) ||
             isa<MMAExecOp>(op) || isa<MMAStoreOp>(op))
      os << getIndent() << "// ERROR: MMA not supported on HIP target\n";
    else if (auto dmaInv = dyn_cast<DMAInvokeOp>(op))
      emitDMAInvoke(dmaInv);
    else if (auto dmaConst = dyn_cast<DMAConstDescOp>(op))
      emitDMAConstDesc(dmaConst);
    else if (auto dmaPrefetch = dyn_cast<DMADescPrefetchOp>(op))
      emitDMAPrefetch(dmaPrefetch);
    else if (auto dmaRt = dyn_cast<DMADescRuntimeOp>(op))
      emitDMARuntimeDesc(dmaRt);
    else if (isa<DMACheckOp>(op))
      (void)op;
    else if (auto dmaCopy = dyn_cast<DmaCopyOp>(op))
      emitDmaCopy(dmaCopy);
    else if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      os << getIndent() << "// ERROR: TMA not supported on HIP target\n";
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto barrier = dyn_cast<BarrierOp>(op))
      emitBarrier(barrier);
    else if (auto wait = dyn_cast<WaitOp>(op))
      emitWait(wait);
    else if (dyn_cast<FutureRotateOp>(op))
      {}
    else if (auto callOp = dyn_cast<coir::CallOp>(op))
      emitCall(callOp);
    else if (auto ithOp = dyn_cast<coir::InThreadsOp>(op))
      emitInThreads(ithOp);
    else if (auto atomicOp = dyn_cast<coir::AtomicOp>(op))
      emitAtomic(atomicOp);
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
    else if (auto indexCast = dyn_cast<arith::IndexCastOp>(op))
      valueNames[indexCast.getResult()] = getName(indexCast.getIn());
    else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op))
      emitIfOp(ifOp);
    else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op))
      emitWhileOp(whileOp);
    else if (auto coirWhile = dyn_cast<coir::CoIRWhileOp>(op))
      emitCoIRWhileOp(coirWhile);
    else if (isa<coir::CoIRWhileCondOp>(op))
      (void)op;
    else if (isa<coir::CoIRBreakOp>(op))
      os << getIndent() << "break;\n";
    else if (isa<coir::CoIRContinueOp>(op))
      os << getIndent() << "continue;\n";
    else if (isa<mlir::scf::YieldOp>(op) || isa<mlir::scf::ConditionOp>(op))
      (void)op;
    else if (emitArithBinOp(op)) {}
    else if (emitCmpOp(op)) {}
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      std::string name = getName(selectOp.getResult());
      os << getIndent() << emitType(selectOp.getResult().getType()) << " "
         << name << " = " << getName(selectOp.getCondition()) << " ? "
         << getName(selectOp.getTrueValue()) << " : "
         << getName(selectOp.getFalseValue()) << ";\n";
    }
    else {
      os << getIndent() << "// [unhandled] " << op->getName().getStringRef()
         << "\n";
    }
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
    os << getIndent() << "choreo::choreo_assert(" << getName(op.getCondition())
       << ", \"" << msg << "\");\n";
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
      return (hostNames[v] = "(" + lhs + " + " + rhs + ")");
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      auto lhs = emitExprInHostScope(mulOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(mulOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      return (hostNames[v] = "(" + lhs + " * " + rhs + ")");
    }
    if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      auto lhs = emitExprInHostScope(subOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(subOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      return (hostNames[v] = "(" + lhs + " - " + rhs + ")");
    }
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      return (hostNames[v] = "(" + lhs + " / " + rhs + ")");
    }
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(remOp.getLhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      auto rhs = emitExprInHostScope(remOp.getRhs(), kernel, hostNames,
                                     numOrigInputs, dimArgMeta);
      return (hostNames[v] = "(" + lhs + " % " + rhs + ")");
    }

    return "/* unknown */";
  }

  void emitEntryAssertions(KernelOp kernel, unsigned numOrigInputs,
                           llvm::ArrayRef<DimArgMeta> dimArgMeta) {
    DenseMap<Value, std::string> hostNames;
    for (auto &ea : entryAssertions) {
      auto cond = emitExprInHostScope(ea.op.getCondition(), kernel, hostNames,
                                      numOrigInputs, dimArgMeta);
      os << "  choreo::runtime_check(" << cond << ", \""
         << ea.op.getMessage() << "\");\n";
    }
  }

  struct DimCheckMeta {
    std::string name;
    int64_t param0, dim0;
    int64_t param1, dim1;
  };

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
      os << "  choreo::runtime_check("
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

  void emitCall(CallOp op) {
    auto callee = op.getCallee().str();
    bool isExpr = op.getIsExpr() && *op.getIsExpr() && op.getResult();
    bool isArith = op.getIsBif() && *op.getIsBif();

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
        funcName = ref.str();
      }
    }

    os << getIndent();
    if (isExpr) {
      auto resTy = op.getResult().getType();
      os << emitType(resTy) << " " << getName(op.getResult()) << " = ";
    }
    os << funcName;

    if (auto tplArgs = op.getTemplateArgs()) {
      os << "<";
      bool first = true;
      for (auto a : *tplArgs) {
        if (!first) os << ", ";
        first = false;
        os << mlir::cast<mlir::StringAttr>(a).getValue().str();
      }
      os << ">";
    }

    os << "(";
    bool first = true;
    for (auto arg : op.getOperands_()) {
      if (!first) os << ", ";
      first = false;
      auto ty = arg.getType();
      if (auto tty = mlir::dyn_cast<coir::TensorType>(ty))
        os << "(" << emitElementType(tty.getElementType()) << "*)"
           << getName(arg);
      else
        os << getName(arg);
    }
    os << ");\n";
  }

  void emitInThreads(InThreadsOp op) {
    os << getIndent() << "if (" << getName(op.getPredicate()) << ") {\n";
    incIndent();
    for (auto &bodyOp : op.getBody().front().getOperations())
      emitOp(&bodyOp);
    decIndent();
    os << getIndent() << "}";
    bool isAsync = op.getAsync() && *op.getAsync();
    bool isOuter = !op.getOuter() || *op.getOuter();
    if (!isAsync && isOuter)
      os << "\n" << getIndent() << "__syncthreads();";
    os << "\n";
  }

  void emitAtomic(AtomicOp op) {
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

    auto dstTy = mlir::cast<coir::TensorType>(op.getDest().getType());
    os << getIndent() << fnName << "(&" << getName(op.getDest()) << "[";
    emitLinearIndex(op.getIndices(), dstTy);
    os << "], ";
    if (op.getKind() == AK::CAS && op.getCompare()) {
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(*op.getCompare()))
        os << intAttr.getInt() << ", ";
      else if (auto fpAttr = mlir::dyn_cast<FloatAttr>(*op.getCompare())) {
        llvm::SmallString<16> s;
        fpAttr.getValue().toString(s, 6, 0);
        os << std::string(s) << ", ";
      } else
        os << "/* compare */, ";
    }
    os << getName(op.getValue()) << ");\n";
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
      if (hasGroupLevel && groupThreadScale > 1) {
        int64_t threadsPerGroup = 1;
        for (auto b : bounds) threadsPerGroup *= b;
        std::string mod = std::to_string(threadsPerGroup);
        for (unsigned i = 0; i < args.size(); ++i) {
          std::string dim = i == 0
              ? "(threadIdx.x % " + mod + ")"
              : "(threadIdx.y % " + mod + ")";
          valueNames[args[i]] = dim;
        }
      } else {
        for (unsigned i = 0; i < args.size(); ++i) {
          std::string dim = i == 0 ? "threadIdx.x" : "threadIdx.y";
          valueNames[args[i]] = dim;
        }
      }
    } else if (level == ParallelLevel::GROUP) {
      int64_t totalWarps = 1;
      for (auto b : bounds) totalWarps *= b;
      hasGroupLevel = true;
      groupThreadScale = totalWarps;
      int64_t warpSize = 32;
      std::string warpId = "(threadIdx.x / " + std::to_string(warpSize) + ")";
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

  void emitCooperativeCopy(Value src, Value dst) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    if (!srcTy) {
      os << getIndent() << "// copy (unknown tensor type)\n";
      return;
    }
    int64_t totalElems = 1;
    for (auto d : srcTy.getShape()) totalElems *= d;
    std::string srcName = getName(src);
    std::string dstName = getName(dst);
    std::string eTy = emitElementType(srcTy.getElementType());

    os << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << totalElems << "; __i += blockDim.x) {\n";
    incIndent();
    os << getIndent() << "((" << eTy << "*)" << dstName << ")[__i] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os << getIndent() << "}\n";
  }

  void emitCopyWithPad(Value src, Value dst,
                       std::optional<ArrayRef<int64_t>> padLow,
                       std::optional<ArrayRef<int64_t>> /*padHigh*/,
                       std::optional<Attribute> padValueAttr) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    auto dstTy = dyn_cast<coir::TensorType>(dst.getType());
    if (!srcTy || !dstTy) {
      os << getIndent() << "// pad copy (unknown tensor type)\n";
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
    os << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << dstElems << "; __i += blockDim.x) {\n";
    incIndent();
    os << getIndent() << "((" << eTy << "*)" << dstName << ")[__i] = ("
       << eTy << ")" << padVal << ";\n";
    decIndent();
    os << getIndent() << "}\n";
    os << getIndent() << "__syncthreads();\n";

    llvm::SmallVector<int64_t> lowVals(rank, 0);
    if (padLow) {
      auto pl = *padLow;
      for (int i = 0; i < rank && i < (int)pl.size(); ++i)
        lowVals[i] = pl[i];
    }

    int64_t srcElems = 1;
    for (auto d : srcShape) srcElems *= d;

    os << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << srcElems << "; __i += blockDim.x) {\n";
    incIndent();
    os << getIndent() << "size_t __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os << getIndent() << "size_t " << dn << " = __rem / " << stride
           << ";\n";
        os << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os << getIndent() << "size_t " << dn << " = __rem;\n";
      }
    }
    os << getIndent() << "size_t __dst_idx = ";
    for (int d = 0; d < rank; ++d) {
      if (d > 0) os << " + ";
      std::string coord = "(__d" + std::to_string(d) + " + "
                         + std::to_string(lowVals[d]) + ")";
      int64_t stride = 1;
      for (int k = d + 1; k < rank; ++k) stride *= dstShape[k];
      if (stride != 1)
        os << coord << " * " << stride;
      else
        os << coord;
    }
    os << ";\n";
    os << getIndent() << "((" << eTy << "*)" << dstName << ")[__dst_idx] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os << getIndent() << "}\n";
  }

  void emitCopyWithTranspose(Value src, Value dst,
                             std::optional<ArrayRef<int64_t>> permAttr) {
    auto srcTy = dyn_cast<coir::TensorType>(src.getType());
    auto dstTy = dyn_cast<coir::TensorType>(dst.getType());
    if (!srcTy || !dstTy) {
      os << getIndent() << "// transpose copy (unknown tensor type)\n";
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

    os << getIndent() << "for (size_t __i = threadIdx.x; __i < "
       << srcElems << "; __i += blockDim.x) {\n";
    incIndent();
    os << getIndent() << "size_t __rem = __i;\n";
    for (int d = 0; d < rank; ++d) {
      std::string dn = "__d" + std::to_string(d);
      if (d < rank - 1) {
        int64_t stride = 1;
        for (int k = d + 1; k < rank; ++k) stride *= srcShape[k];
        os << getIndent() << "size_t " << dn << " = __rem / " << stride
           << ";\n";
        os << getIndent() << "__rem = __rem % " << stride << ";\n";
      } else {
        os << getIndent() << "size_t " << dn << " = __rem;\n";
      }
    }
    os << getIndent() << "size_t __dst_idx = ";
    for (int d = 0; d < rank; ++d) {
      if (d > 0) os << " + ";
      std::string coord = "__d" + std::to_string(perm[d]);
      int64_t stride = 1;
      for (int k = d + 1; k < rank; ++k) stride *= dstShape[k];
      if (stride != 1)
        os << coord << " * " << stride;
      else
        os << coord;
    }
    os << ";\n";
    os << getIndent() << "((" << eTy << "*)" << dstName << ")[__dst_idx] = (("
       << eTy << "*)" << srcName << ")[__i];\n";
    decIndent();
    os << getIndent() << "}\n";
  }

  template <typename CopyOp>
  void emitCopyDispatch(CopyOp op) {
    auto kind = op.getKind();
    if (kind == DMAKind::Pad) {
      auto padLow = op.getPadLow();
      auto padHigh = op.getPadHigh();
      auto padValue = op.getPadValue();
      std::optional<ArrayRef<int64_t>> pl;
      if (padLow) pl = padLow.value();
      std::optional<ArrayRef<int64_t>> ph;
      if (padHigh) ph = padHigh.value();
      std::optional<Attribute> pv;
      if (padValue) pv = *padValue;
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

  void emitDmaCopy(DmaCopyOp op) {
    emitCopyDispatch(op);
    os << getIndent() << "__syncthreads();\n";
    valueNames[op.getToken()] = getName(op.getDest());
  }

  void emitElementCopy(ElementCopyOp op) {
    emitCooperativeCopy(op.getSource(), op.getDest());
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
    auto token = op.getToken();
    if (dmaTokens.count(token))
      return;
    os << getIndent() << "__syncthreads();\n";
  }

  void emitTensorAlloc(TensorAllocOp op) {
    if (returnValues.count(op.getResult()))
      return;

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    int32_t ms = tensorTy.getMemorySpace();
    std::string name = getName(op.getResult());

    if (tensorTy.hasDynamicShape() && ms == 1 /*shared*/) {
      std::string eType = emitElementType(tensorTy.getElementType());
      os << getIndent() << "extern __shared__ unsigned char __dyn_smem[];\n";
      os << getIndent() << eType << "* " << name
         << " = (" << eType << "*)__dyn_smem;\n";
      return;
    }

    std::string qualifier = ms == 1 ? "__shared__ " : "";
    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape())
      totalElems *= d;

    os << getIndent() << qualifier << emitElementType(tensorTy.getElementType())
       << " " << name << "[" << totalElems << "];\n";
  }

  void emitTensorTile(TensorTileOp op) {
    std::string name = getName(op.getResult());
    auto srcTy = dyn_cast<coir::TensorType>(op.getSource().getType());
    auto indices = op.getIndices();

    if (indices.empty()) {
      valueNames[op.getResult()] = getName(op.getSource());
      return;
    }

    auto srcShape = srcTy.getShape();
    os << getIndent() << "auto " << name << " = " << getName(op.getSource());
    os << " + (";
    for (unsigned i = 0; i < indices.size(); ++i) {
      if (i > 0) os << " + ";
      os << getName(indices[i]);
      int64_t stride = 1;
      for (unsigned j = i + 1; j < srcShape.size(); ++j)
        stride *= srcShape[j];
      os << " * " << stride;
    }
    os << ");\n";
  }

  void emitLinearIndex(mlir::ValueRange indices, coir::TensorType tty) {
    auto strides = tty.getStrides();
    auto shape = tty.getShape();
    if (indices.empty()) { os << "0"; return; }
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
    os << getIndent() << emitType(op.getResult().getType()) << " " << name
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
    if (isa<arith::AddIOp>(op) || isa<arith::AddFOp>(op)) opStr = "+";
    else if (isa<arith::SubIOp>(op) || isa<arith::SubFOp>(op)) opStr = "-";
    else if (isa<arith::MulIOp>(op) || isa<arith::MulFOp>(op)) opStr = "*";
    else if (isa<arith::DivSIOp>(op) || isa<arith::DivFOp>(op)) opStr = "/";
    else if (isa<arith::RemSIOp>(op)) opStr = "%";
    else return false;

    std::string name = getName(op->getResult(0));
    std::string lhs = getName(op->getOperand(0));
    std::string rhs = getName(op->getOperand(1));
    os << getIndent() << emitType(op->getResult(0).getType()) << " "
       << name << " = " << lhs << " " << opStr << " " << rhs << ";\n";
    return true;
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
      } else emitOp(&bodyOp);
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
        } else emitOp(&bodyOp);
      }
      decIndent();
      os << getIndent() << "}\n";
    }
  }

  void emitWhileOp(mlir::scf::WhileOp op) {
    auto &beforeBlock = op.getBefore().front();
    auto condOp =
        dyn_cast<mlir::scf::ConditionOp>(beforeBlock.getTerminator());
    for (unsigned i = 0; i < op.getInits().size(); ++i)
      valueNames[beforeBlock.getArgument(i)] = getName(op.getInits()[i]);
    for (auto &bodyOp : beforeBlock.getOperations()) {
      if (isa<mlir::scf::ConditionOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }
    os << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
    incIndent();
    auto &afterBlock = op.getAfter().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[afterBlock.getArgument(i)] = getName(condOp.getArgs()[i]);
    for (auto &bodyOp : afterBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          os << getIndent() << getName(op.getInits()[i]) << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
          valueNames[beforeBlock.getArgument(i)] =
              getName(yieldOp.getOperand(i));
        }
        for (auto &bOp : beforeBlock.getOperations()) {
          if (isa<mlir::scf::ConditionOp>(&bOp)) continue;
          emitOp(&bOp);
        }
      } else emitOp(&bodyOp);
    }
    decIndent();
    os << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      if (condOp && i < condOp.getArgs().size())
        valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
  }

  void emitCoIRWhileOp(coir::CoIRWhileOp op) {
    auto &condBlock = op.getCondRegion().front();
    auto condOp = dyn_cast<coir::CoIRWhileCondOp>(condBlock.getTerminator());
    for (unsigned i = 0; i < op.getInits().size(); ++i)
      valueNames[condBlock.getArgument(i)] = getName(op.getInits()[i]);
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
        for (unsigned i = 0; i < breakOp.getOperands().size(); ++i)
          valueNames[condBlock.getArgument(i)] =
              getName(breakOp.getOperand(i));
        os << getIndent() << "break;\n";
      } else if (auto contOp = dyn_cast<coir::CoIRContinueOp>(&bodyOp)) {
        for (unsigned i = 0; i < contOp.getOperands().size(); ++i)
          valueNames[condBlock.getArgument(i)] =
              getName(contOp.getOperand(i));
        for (auto &cOp : condBlock.getOperations()) {
          if (isa<coir::CoIRWhileCondOp>(&cOp)) continue;
          emitOp(&cOp);
        }
        os << getIndent() << "continue;\n";
      } else emitOp(&bodyOp);
    }
    decIndent();
    os << getIndent() << "}\n";
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      if (condOp && i < condOp.getArgs().size())
        valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
  }

  bool emitCmpOp(Operation *op) {
    if (auto cmpI = dyn_cast<arith::CmpIOp>(op)) {
      std::string name = getName(cmpI.getResult());
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
      os << getIndent() << "bool " << name << " = ("
         << getName(cmpI.getLhs()) << " " << opStr << " "
         << getName(cmpI.getRhs()) << ");\n";
      return true;
    }
    if (auto cmpF = dyn_cast<arith::CmpFOp>(op)) {
      std::string name = getName(cmpF.getResult());
      llvm::StringRef opStr;
      switch (cmpF.getPredicate()) {
      case arith::CmpFPredicate::OEQ: opStr = "=="; break;
      case arith::CmpFPredicate::OGT: opStr = ">"; break;
      case arith::CmpFPredicate::OGE: opStr = ">="; break;
      case arith::CmpFPredicate::OLT: opStr = "<"; break;
      case arith::CmpFPredicate::OLE: opStr = "<="; break;
      default: opStr = "!="; break;
      }
      os << getIndent() << "bool " << name << " = ("
         << getName(cmpF.getLhs()) << " " << opStr << " "
         << getName(cmpF.getRhs()) << ");\n";
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

struct EmitHIPPass : public ::coir::impl::EmitHIPBase<EmitHIPPass> {
  using EmitHIPBase::EmitHIPBase;

  void runOnOperation() override {
    auto module = getOperation();
    HIPEmitter emitter(llvm::outs());
    emitter.emitModule(module);
  }
};

} // namespace

namespace coir {
std::unique_ptr<mlir::Pass> createEmitHIPPass() {
  return std::make_unique<EmitHIPPass>();
}

void emitHIP(mlir::ModuleOp module, llvm::raw_ostream &os) {
  HIPEmitter emitter(os);
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
  os << "# CoIR generated script -- compile and execute HIP kernel\n";
  os << "set -eo pipefail\n\n";

  os << "TMPDIR=$(mktemp -d /tmp/cocc_hip_XXXXXX)\n";
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
    os << "  _coir_bin=$(which cocc 2>/dev/null || true)\n";
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

  os << "ROCM_HOME=\"${ROCM_HOME:-/opt/rocm}\"\n";
  os << "HIPCC=\"${ROCM_HOME}/bin/hipcc\"\n";
  os << "if [[ ! -x \"$HIPCC\" ]]; then\n";
  os << "  echo \"Error: hipcc not found at $HIPCC\"; exit 1\n";
  os << "fi\n";
  os << "if [[ -z \"${amdgpu_arch:-}\" ]]; then\n";
  os << "  amdgpu_arch=$("
        "\"${ROCM_HOME}/bin/rocminfo\" 2>/dev/null | "
        "grep -oP 'gfx\\d+' | head -1 || echo \"gfx1030\")\n";
  os << "fi\n\n";
}

void emitScriptFooter(llvm::raw_ostream &os) {
  auto &sctx = CoIR::ScriptContext::Get();
  bool has_embedded = sctx.types_header && sctx.runtime_header;

  os << "\n__COCC_HIP_SOURCE__\n\n";

  if (has_embedded) {
    os << "\"${HIPCC}\" -std=c++17 --offload-arch=\"${amdgpu_arch}\" "
          "-I\"$TMPDIR\" "
          "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
  } else {
    os << "\"$HIPCC\" -std=c++17 --offload-arch=\"$amdgpu_arch\" "
          "-I\"$CHOREO_INC\" -I\"$TMPDIR\" "
          "-o \"$BINFILE\" \"$TMPFILE\" 2>&1\n";
  }

  os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
  os << "  shift\n";
  os << "  \"$BINFILE\" \"$@\"\n";
  os << "fi\n";
}

class HIPTargetCodeGen : public CoIR::CodeGen {
public:
  int EmitSource(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    coir::emitHIP(module, os);
    emitHostCode(module, os);
    return 0;
  }

  int EmitScript(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    emitScriptHeader(os);
    os << "TMPFILE=\"$TMPDIR/kernel.hip\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_HIP_SOURCE__'\n";
    coir::emitHIP(module, os);
    emitHostCode(module, os);
    emitScriptFooter(os);
    return 0;
  }
};

static bool registered_hip = [] {
  CoIR::CodeGenRegistry::Register("hip", [] {
    return std::make_unique<HIPTargetCodeGen>();
  });
  return true;
}();

} // namespace
