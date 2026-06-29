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
        emitHostWrapper(kernel);
    }
  }

private:
  llvm::raw_ostream &os;
  unsigned indent;
  bool hasTMA = false;
  bool hasDMA = false;
  DenseMap<Value, std::string> valueNames;
  DenseMap<unsigned, std::string> returnParamNames;
  DenseSet<Value> returnValues;
  DenseSet<Value> mmaAccumulators;
  DenseMap<Value, std::string> mmaFragRoles;
  DenseMap<Value, std::string> mmaFragLayouts;
  struct TileLayout {
    llvm::SmallVector<int64_t> shape;
    llvm::SmallVector<int64_t> strides;
  };
  DenseMap<Value, int64_t> tileStrides;
  DenseMap<Value, TileLayout> tileLayouts;
  unsigned nextId = 0;

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
  };
  llvm::SmallVector<DescInfo> descInfos;
  DenseMap<Value, unsigned> descValueToIndex;
  // Runtime offsets bound per DMAInvokeOp desc operand (from DMADescRuntimeOp)
  DenseMap<Value, SmallVector<Value, 4>> descRuntimeOffsets;
  DenseSet<Value> dmaTokens; // tokens from DMAInvokeOp
  unsigned nextFutureId = 0;

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

  void emitHeader(bool withTMA) {
    os << "#define __CHOREO_TARGET_CUTE__\n";
    os << "#define __USE_CUDA_TYPE__\n";
    if (withTMA) {
      os << "#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90\n";
      os << "#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__\n";
    }
    os << "#include \"choreo.h\"\n";
    if (withTMA) {
      os << "namespace cde = cuda::device::experimental;\n";
    }
    os << "using namespace nvcuda;\n";
    os << "using namespace choreo;\n\n";
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

      unsigned idx = descInfos.size();
      descInfos.push_back({idx, srcType, dstType, op.getKind(),
                           isTMA, isLoad, op.getOut()});
      descValueToIndex[op.getOut()] = idx;
    });
  }

  void emitKernel(KernelOp kernel) {
    entryAssertions.clear();
    prescanMMAFragRoles(kernel);
    prescanDescriptors(kernel);

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
    // TMA descriptors as kernel parameters
    unsigned tmaParamCount = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      if (paramIdx > 0)
        os << ", ";
      os << "const __grid_constant__ CUtensorMap __choreo_tma_"
         << tmaParamCount << "_tensor_map";
      paramIdx++;
      tmaParamCount++;
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

    // Emit TMA barriers and atoms at kernel body start (only for TMA loads)
    unsigned tmaIdx = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      if (descInfos[i].isLoad) {
        os << getIndent() << "__shared__ __align__(8) uint64_t "
           << "choreo_copy_atom_t_" << tmaIdx << "_barrier;\n";
        os << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
        incIndent();
        os << getIndent() << "choreo::tma_mbarrier_init(&choreo_copy_atom_t_"
           << tmaIdx << "_barrier, 1);\n";
        decIndent();
        os << getIndent() << "}\n";
        os << getIndent() << "__syncthreads();\n";
        os << getIndent() << "TMAAtom choreo_copy_atom_t_" << tmaIdx
           << "{&choreo_copy_atom_t_" << tmaIdx << "_barrier};\n\n";
      }
      tmaIdx++;
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

  std::string emitTMADataType(Type elemTy) {
    if (elemTy.isF32()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
    if (elemTy.isF16()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
    if (elemTy.isF64()) return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
    if (elemTy.isInteger(8)) return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (elemTy.isInteger(16)) return "CU_TENSOR_MAP_DATA_TYPE_UINT16";
    if (elemTy.isInteger(32)) return "CU_TENSOR_MAP_DATA_TYPE_UINT32";
    if (auto f8Ty = dyn_cast<mlir::Float8E4M3FNType>(elemTy))
      return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    if (auto f8Ty = dyn_cast<mlir::Float8E5M2Type>(elemTy))
      return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
    return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
  }

  void emitTMADescriptorSetup(unsigned descIdx, const std::string &dataPtr,
                              coir::TensorType globalType,
                              coir::TensorType tileType) {
    auto shape = globalType.getShape();
    auto elemTy = globalType.getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    unsigned rank = shape.size();

    // Tile shape for box dimensions (transfer size per TMA op)
    auto tileShape = tileType.getShape();

    std::string prefix = "__choreo_tma_" + std::to_string(descIdx);

    // Shape (innermost-first for cuTensorMapEncodeTiled)
    os << "  uint64_t " << prefix << "_shape[] = {";
    for (int i = (int)rank - 1; i >= 0; --i) {
      if (i < (int)rank - 1) os << ", ";
      os << shape[i];
    }
    os << "};\n";

    // Strides (byte strides, skip innermost -- it's implicit)
    os << "  uint64_t " << prefix << "_strides[] = {";
    int64_t stride = elemBytes;
    for (int i = (int)rank - 1; i >= 1; --i) {
      if (i < (int)rank - 1) os << ", ";
      stride *= shape[i];
      os << stride;
    }
    os << "};\n";

    // Box shape from the tile type (innermost-first)
    os << "  uint32_t " << prefix << "_box_shape[] = {";
    for (int i = (int)rank - 1; i >= 0; --i) {
      if (i < (int)rank - 1) os << ", ";
      int64_t tileDim = (i < (int)tileShape.size()) ? tileShape[i] : 1;
      os << "(uint32_t)(" << tileDim << ")";
    }
    os << "};\n";

    // Element strides (always 1)
    os << "  uint32_t " << prefix << "_elem_strides[] = {";
    for (unsigned i = 0; i < rank; ++i) {
      if (i > 0) os << ", ";
      os << "1";
    }
    os << "};\n";

    // Tensor map
    os << "  alignas(64) CUtensorMap " << prefix << "_tensor_map{};\n";
    os << "  CUresult " << prefix << "_res = cuTensorMapEncodeTiled(\n";
    os << "          &" << prefix << "_tensor_map,\n";
    os << "          CUtensorMapDataType::" << emitTMADataType(elemTy) << ",\n";
    os << "          " << rank << ",\n";
    os << "          " << dataPtr << ",\n";
    os << "          " << prefix << "_shape,\n";
    os << "          " << prefix << "_strides,\n";
    os << "          " << prefix << "_box_shape,\n";
    os << "          " << prefix << "_elem_strides,\n";
    os << "          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,\n";
    os << "          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,\n";
    os << "          CUtensorMapL2promotion::"
       << "CU_TENSOR_MAP_L2_PROMOTION_L2_128B,\n";
    os << "          CUtensorMapFloatOOBfill::"
       << "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);\n";
    os << "  choreo::abend_true(" << prefix << "_res != CUDA_SUCCESS);\n";
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

  void emitHostWrapper(KernelOp kernel) {
    prescanDescriptors(kernel);
    auto fnType = kernel.getFunctionType();
    auto symName = kernel.getSymName();
    std::string devName = kernelDeviceName(symName);
    unsigned numInputs = fnType.getNumInputs();
    unsigned numResults = fnType.getNumResults();
    unsigned numTMA = countTMADescs();

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

      // TMA descriptor setup
      unsigned tmaIdx = 0;
      for (unsigned i = 0; i < descInfos.size(); ++i) {
        if (!descInfos[i].isTMA) continue;
        auto globalType = descInfos[i].isLoad ? descInfos[i].srcType
                                              : descInfos[i].dstType;
        auto tileType = descInfos[i].isLoad ? descInfos[i].dstType
                                            : descInfos[i].srcType;
        std::string ptr = getTMAGlobalPtr(tmaIdx, kernel);
        emitTMADescriptorSetup(tmaIdx, ptr, globalType, tileType);
        tmaIdx++;
      }

      auto dims = getLaunchDims(kernel);
      emitEntryAssertions(kernel);

      os << "  " << devName << "<<<" << dims.gridStr() << ", "
         << dims.blockStr() << ">>>(";
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i > 0) os << ", ";
        os << "p" << i << "__device";
      }
      for (unsigned i = 0; i < numTMA; ++i) {
        os << ", __choreo_tma_" << i << "_tensor_map";
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

    // TMA descriptor setup
    unsigned tmaIdx = 0;
    for (unsigned i = 0; i < descInfos.size(); ++i) {
      if (!descInfos[i].isTMA) continue;
      auto globalType = descInfos[i].isLoad ? descInfos[i].srcType
                                            : descInfos[i].dstType;
      auto tileType = descInfos[i].isLoad ? descInfos[i].dstType
                                          : descInfos[i].srcType;
      std::string ptr = getTMAGlobalPtr(tmaIdx, kernel);
      emitTMADescriptorSetup(tmaIdx, ptr, globalType, tileType);
      tmaIdx++;
    }

    auto dims = getLaunchDims(kernel);
    emitEntryAssertions(kernel);

    os << "  " << devName << "<<<" << dims.gridStr() << ", "
       << dims.blockStr() << ">>>(";
    for (unsigned i = 0; i < numInputs; ++i) {
      if (i > 0) os << ", ";
      os << "p" << i << "__device";
    }
    os << ", __result__device";
    for (unsigned i = 0; i < numTMA; ++i) {
      os << ", __choreo_tma_" << i << "_tensor_map";
    }
    os << ");\n";
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
    else if (auto constDesc = dyn_cast<DMAConstDescOp>(op))
      emitDMAConstDesc(constDesc);
    else if (auto prefetch = dyn_cast<DMADescPrefetchOp>(op))
      emitDMAPrefetch(prefetch);
    else if (auto rtDesc = dyn_cast<DMADescRuntimeOp>(op))
      emitDMARuntimeDesc(rtDesc);
    else if (auto invoke = dyn_cast<DMAInvokeOp>(op))
      emitDMAInvoke(invoke);
    else if (auto dmaCopy = dyn_cast<DmaCopyOp>(op))
      emitDmaCopy(dmaCopy);
    else if (auto tmaCopy = dyn_cast<TmaCopyOp>(op))
      emitTmaCopy(tmaCopy);
    else if (auto elemCopy = dyn_cast<ElementCopyOp>(op))
      emitElementCopy(elemCopy);
    else if (auto barrier = dyn_cast<BarrierOp>(op))
      emitBarrier(barrier);
    else if (auto wait = dyn_cast<WaitOp>(op))
      emitWait(wait);
    else if (dyn_cast<FutureRotateOp>(op))
      {}
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
    else if (auto indexCast = dyn_cast<arith::IndexCastOp>(op)) {
      valueNames[indexCast.getResult()] = getName(indexCast.getIn());
    }
    else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op))
      emitIfOp(ifOp);
    else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op))
      emitWhileOp(whileOp);
    else if (auto coirWhile = dyn_cast<coir::CoIRWhileOp>(op))
      emitCoIRWhileOp(coirWhile);
    else if (isa<coir::CoIRWhileCondOp>(op))
      (void)op;
    else if (isa<coir::CoIRBreakOp>(op))
      emitBreak(cast<coir::CoIRBreakOp>(op));
    else if (isa<coir::CoIRContinueOp>(op))
      emitContinue(cast<coir::CoIRContinueOp>(op));
    else if (isa<mlir::scf::YieldOp>(op) || isa<mlir::scf::ConditionOp>(op))
      (void)op;
    else if (emitArithBinOp(op)) {}
    else if (emitCmpOp(op)) {}
    else if (auto assertOp = dyn_cast<AssertOp>(op))
      emitAssert(assertOp);
    else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      std::string name = getName(selectOp.getResult());
      os << getIndent() << emitCUDAType(selectOp.getResult().getType()) << " "
         << name << " = " << getName(selectOp.getCondition()) << " ? "
         << getName(selectOp.getTrueValue()) << " : "
         << getName(selectOp.getFalseValue()) << ";\n";
    }
    else if (isa<DMACheckOp>(op))
      (void)op;
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
       << ", \"" << msg << "\");" << "\n";
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
      std::string result = "(" + lhs + " + " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      auto lhs = emitExprInHostScope(mulOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(mulOp.getRhs(), kernel, hostNames);
      std::string result = "(" + lhs + " * " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      auto lhs = emitExprInHostScope(subOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(subOp.getRhs(), kernel, hostNames);
      std::string result = "(" + lhs + " - " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(divOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(divOp.getRhs(), kernel, hostNames);
      std::string result = "(" + lhs + " / " + rhs + ")";
      hostNames[v] = result;
      return result;
    }
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      auto lhs = emitExprInHostScope(remOp.getLhs(), kernel, hostNames);
      auto rhs = emitExprInHostScope(remOp.getRhs(), kernel, hostNames);
      std::string result = "(" + lhs + " % " + rhs + ")";
      hostNames[v] = result;
      return result;
    }

    llvm_unreachable("unsupported op in host-scope expression");
  }

  void emitEntryAssertions(KernelOp kernel) {
    DenseMap<Value, std::string> hostNames;
    for (auto &ea : entryAssertions) {
      auto cond =
          emitExprInHostScope(ea.op.getCondition(), kernel, hostNames);
      os << "  choreo::runtime_check(" << cond << ", \""
         << ea.op.getMessage() << "\");\n";
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

  TileLayout getLayoutForValue(Value v) {
    auto it = tileLayouts.find(v);
    if (it != tileLayouts.end())
      return it->second;
    auto tty = dyn_cast<coir::TensorType>(v.getType());
    if (!tty) return {};
    auto shape = tty.getShape();
    TileLayout layout;
    layout.shape.assign(shape.begin(), shape.end());
    layout.strides.resize(shape.size());
    int64_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      layout.strides[i] = s;
      s *= shape[i];
    }
    return layout;
  }

  void emitCuteMakeShape(llvm::StringRef varName,
                         const llvm::SmallVector<int64_t> &dims) {
    os << getIndent() << "auto " << varName << " = cute::make_shape(";
    for (unsigned i = 0; i < dims.size(); ++i) {
      if (i > 0) os << ", ";
      os << "cute::Int<" << dims[i] << ">{}";
    }
    os << ");\n";
  }

  void emitCuteMakeStride(llvm::StringRef varName,
                          const llvm::SmallVector<int64_t> &strides) {
    os << getIndent() << "auto " << varName << " = cute::make_stride(";
    for (unsigned i = 0; i < strides.size(); ++i) {
      if (i > 0) os << ", ";
      os << "cute::Int<" << strides[i] << ">{}";
    }
    os << ");\n";
  }

  std::string emitCuteTensor(Value v, const std::string &suffix) {
    auto layout = getLayoutForValue(v);
    auto tty = dyn_cast<coir::TensorType>(v.getType());
    int32_t ms = tty ? tty.getMemorySpace() : 0;

    std::string shapeName = "__shape" + suffix;
    std::string strideName = "__stride" + suffix;
    std::string layoutName = "__layout" + suffix;
    std::string tensorName = "__tensor" + suffix;

    emitCuteMakeShape(shapeName, layout.shape);
    emitCuteMakeStride(strideName, layout.strides);
    os << getIndent() << "auto " << layoutName
       << " = cute::make_layout(" << shapeName << ", " << strideName
       << ");\n";

    std::string elemTy = tty ? emitElementType(tty.getElementType()) : "int";
    std::string ptrName = getName(v);
    os << getIndent() << "auto " << tensorName << " = cute::make_tensor(";
    if (ms == 0)
      os << "cute::make_gmem_ptr<" << elemTy << ">((" << elemTy << "*)"
         << ptrName << ")";
    else
      os << "((" << elemTy << "*)" << ptrName << ")";
    os << ", " << layoutName << ");\n";

    return tensorName;
  }

  void emitNaiveCopy(Value src, Value dst) {
    unsigned id = nextId++;
    std::string srcTensor = emitCuteTensor(src, "_src_" + std::to_string(id));
    std::string dstTensor = emitCuteTensor(dst, "_dst_" + std::to_string(id));
    os << getIndent() << "choreo::naive_copy(" << srcTensor << ", "
       << dstTensor << ");\n";
  }

  // --- DMA Descriptor Pipeline Emission ---

  void emitDMAConstDesc(DMAConstDescOp op) {
    // DMAConstDescOp is a compile-time descriptor -- no runtime code emitted.
    // We propagate the desc value index for use by downstream ops.
    auto it = descValueToIndex.find(op.getOut());
    if (it == descValueToIndex.end()) return;
    // Track the prefetch chain: prefetch.desc consumes const.desc result
    valueNames[op.getOut()] = "__desc_" + std::to_string(it->second);
  }

  void emitDMAPrefetch(DMADescPrefetchOp op) {
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

  void emitDMARuntimeDesc(DMADescRuntimeOp op) {
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

  void emitDMAInvoke(DMAInvokeOp op) {
    Value descVal = op.getDesc();
    auto it = descValueToIndex.find(descVal);
    if (it == descValueToIndex.end()) {
      os << getIndent() << "// DMA invoke (no descriptor info)\n";
      os << getIndent() << "__syncthreads();\n";
      dmaTokens.insert(op.getDone());
      return;
    }

    unsigned descIdx = it->second;
    auto &desc = descInfos[descIdx];
    dmaTokens.insert(op.getDone());

    // Collect runtime offsets if any
    SmallVector<Value, 4> offsets;
    auto offsIt = descRuntimeOffsets.find(descVal);
    if (offsIt != descRuntimeOffsets.end())
      offsets = offsIt->second;

    if (desc.isTMA) {
      emitTMAInvoke(descIdx, desc, offsets);
    } else {
      emitCpAsyncInvoke(descIdx, desc);
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

      os << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
      incIndent();
      os << getIndent() << "choreo::tma_mbarrier_expect_tx("
         << atomName << ".ptx_barrier(), " << transferBytes << ");\n";
      os << getIndent() << loadFunc << "((void*)(" << dstName
         << "), (const void*)&" << mapName << ", "
         << atomName << ".ptx_barrier()";
      // Emit coordinates (innermost-first)
      for (int d = (int)rank - 1; d >= 0; --d) {
        os << ", ";
        if (d < (int)offsets.size()) {
          int64_t tileDim = tileShape[d];
          if (tileDim > 1)
            os << getName(offsets[d]) << " * " << tileDim;
          else
            os << getName(offsets[d]);
        } else {
          os << "0";
        }
      }
      os << ");\n";
      decIndent();
      os << getIndent() << "}\n";

      os << getIndent() << "choreo::tma_mbarrier_wait_parity("
         << atomName << ".ptx_barrier(), "
         << atomName << ".ptx_phase_bit());\n";
      os << getIndent() << atomName << ".toggle_ptx_phase();\n";
    } else {
      // S2G TMA store
      std::string srcName;
      if (auto constOp =
              desc.constDescResult.getDefiningOp<DMAConstDescOp>()) {
        srcName = getName(constOp.getSource());
      } else {
        srcName = "/* unknown src */";
      }

      os << getIndent() << "cde::fence_proxy_async_shared_cta();\n";
      os << getIndent() << "__syncthreads();\n";
      os << getIndent() << "if (threadIdx.x == 0 && threadIdx.y == 0) {\n";
      incIndent();

      auto srcType = desc.srcType;
      std::string storeFunc = rank <= 2
          ? "cde::cp_async_bulk_tensor_2d_shared_to_global"
          : "cde::cp_async_bulk_tensor_3d_shared_to_global";

      os << getIndent() << storeFunc << "(&" << mapName;
      // Emit coordinates (innermost-first)
      for (int d = (int)rank - 1; d >= 0; --d) {
        os << ", ";
        if (d < (int)offsets.size()) {
          int64_t tileDim = tileShape[d];
          if (tileDim > 1)
            os << getName(offsets[d]) << " * " << tileDim;
          else
            os << getName(offsets[d]);
        } else {
          os << "0";
        }
      }
      os << ", " << srcName << ");\n";
      os << getIndent() << "cde::cp_async_bulk_commit_group();\n";
      os << getIndent() << "cde::cp_async_bulk_wait_group_read<0>();\n";
      decIndent();
      os << getIndent() << "}\n";
    }
  }

  void emitCpAsyncInvoke(unsigned /*descIdx*/, const DescInfo &desc) {
    // cp.async DMA: cooperative copy with fence/wait
    // Use the original operands from the DMAConstDescOp
    if (auto constOp =
            desc.constDescResult.getDefiningOp<DMAConstDescOp>()) {
      os << getIndent() << "// cp.async DMA\n";
      emitNaiveCopy(constOp.getSource(), constOp.getDest());
      os << getIndent() << "__syncthreads();\n";
    } else {
      os << getIndent() << "// cp.async DMA (unknown source)\n";
      os << getIndent() << "__syncthreads();\n";
    }
  }

  // --- Fallback handlers for un-lowered copy ops (when LowerDMADesc skips) ---

  void emitDmaCopy(DmaCopyOp op) {
    emitNaiveCopy(op.getSource(), op.getDest());
    os << getIndent() << "__syncthreads();\n";
    dmaTokens.insert(op.getToken());
  }

  void emitTmaCopy(TmaCopyOp op) {
    // Should not reach here after LowerDMADesc; fallback to naive copy
    emitNaiveCopy(op.getSource(), op.getDest());
    os << getIndent() << "__syncthreads();\n";
    dmaTokens.insert(op.getToken());
  }

  void emitElementCopy(ElementCopyOp op) {
    emitNaiveCopy(op.getSource(), op.getDest());
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
    if (dmaTokens.count(token)) {
      // Wait already handled inline during DMA invoke emission
      return;
    }
    os << getIndent() << "__syncthreads();\n";
  }

  void emitTensorAlloc(TensorAllocOp op) {
    if (returnValues.count(op.getResult()))
      return; // bound to output parameter

    auto tensorTy = cast<coir::TensorType>(op.getResult().getType());
    int32_t ms = tensorTy.getMemorySpace();
    std::string name = getName(op.getResult());

    int64_t totalElems = 1;
    for (auto d : tensorTy.getShape())
      totalElems *= d;

    if (ms == 1 && hasTMA) {
      // Shared memory with TMA needs 128-byte alignment
      int64_t totalBytes = totalElems *
          (tensorTy.getElementType().getIntOrFloatBitWidth() / 8);
      os << getIndent() << "__shared__ alignas(128) unsigned char "
         << name << "_storage[" << totalBytes << "];\n";
      os << getIndent() << emitElementType(tensorTy.getElementType())
         << "* " << name << " = ("
         << emitElementType(tensorTy.getElementType()) << "*)("
         << name << "_storage);\n";
    } else {
      std::string qualifier = ms == 1 ? "__shared__ " : "";
      os << getIndent() << qualifier << emitElementType(tensorTy.getElementType())
         << " " << name << "[" << totalElems << "];\n";
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
    auto tty = cast<coir::TensorType>(op.getDest().getType());
    os << getIndent() << fnName << "(&" << getName(op.getDest()) << "[";
    emitLinearIndex(op.getIndices(), tty);
    os << "], " << getName(op.getValue());
    if (op.getKind() == AK::CAS && op.getCompare())
      os << ", " << "/* compare */";
    os << ");\n";
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
    // Declare result variables before the if statement
    for (auto res : op.getResults()) {
      os << getIndent() << emitCUDAType(res.getType()) << " "
         << getName(res) << ";\n";
    }

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

    // Map init values to the before-region block args
    for (unsigned i = 0; i < op.getInits().size(); ++i)
      valueNames[beforeBlock.getArgument(i)] = getName(op.getInits()[i]);

    // Declare mutable iter variables
    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i) {
      std::string name = getName(op.getInits()[i]);
      iterVarNames.push_back(name);
    }

    // Emit "before" region ops (except the terminating scf.condition)
    for (auto &bodyOp : beforeBlock.getOperations()) {
      if (isa<mlir::scf::ConditionOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }

    // Emit while (condition)
    os << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
    incIndent();

    // Map condition-forwarded args to after-region block args
    auto &afterBlock = op.getAfter().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[afterBlock.getArgument(i)] =
          getName(condOp.getArgs()[i]);

    // Emit "after" region ops
    for (auto &bodyOp : afterBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&bodyOp)) {
        // Update iter variables and re-emit condition ops
        for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(yieldOp.getOperand(i)) << ";\n";
          valueNames[beforeBlock.getArgument(i)] =
              getName(yieldOp.getOperand(i));
        }
        // Re-emit before-region ops for next iteration condition
        for (auto &bOp : beforeBlock.getOperations()) {
          if (isa<mlir::scf::ConditionOp>(&bOp)) continue;
          emitOp(&bOp);
        }
      } else {
        emitOp(&bodyOp);
      }
    }

    decIndent();
    os << getIndent() << "}\n";

    // Map while results
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      if (condOp && i < condOp.getArgs().size())
        valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
    }
  }

  void emitCoIRWhileOp(coir::CoIRWhileOp op) {
    auto &condBlock = op.getCondRegion().front();
    auto condOp = dyn_cast<coir::CoIRWhileCondOp>(condBlock.getTerminator());

    // Map init values to condition block args
    for (unsigned i = 0; i < op.getInits().size(); ++i)
      valueNames[condBlock.getArgument(i)] = getName(op.getInits()[i]);

    llvm::SmallVector<std::string> iterVarNames;
    for (unsigned i = 0; i < op.getInits().size(); ++i)
      iterVarNames.push_back(getName(op.getInits()[i]));

    // Emit condition region ops (except terminator)
    for (auto &bodyOp : condBlock.getOperations()) {
      if (isa<coir::CoIRWhileCondOp>(&bodyOp)) continue;
      emitOp(&bodyOp);
    }

    os << getIndent() << "while (" << getName(condOp.getCondition())
       << ") {\n";
    incIndent();

    // Map condition-forwarded args to body block args
    auto &bodyBlock = op.getBodyRegion().front();
    for (unsigned i = 0; i < condOp.getArgs().size(); ++i)
      valueNames[bodyBlock.getArgument(i)] =
          getName(condOp.getArgs()[i]);

    // Emit body region ops
    for (auto &bodyOp : bodyBlock.getOperations()) {
      if (auto breakOp = dyn_cast<coir::CoIRBreakOp>(&bodyOp)) {
        for (unsigned i = 0; i < breakOp.getOperands().size(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(breakOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] =
              getName(breakOp.getOperand(i));
        }
        os << getIndent() << "break;\n";
      } else if (auto contOp = dyn_cast<coir::CoIRContinueOp>(&bodyOp)) {
        for (unsigned i = 0; i < contOp.getOperands().size(); ++i) {
          os << getIndent() << iterVarNames[i] << " = "
             << getName(contOp.getOperand(i)) << ";\n";
          valueNames[condBlock.getArgument(i)] =
              getName(contOp.getOperand(i));
        }
        // Re-emit condition computation for next iteration
        for (auto &cOp : condBlock.getOperations()) {
          if (isa<coir::CoIRWhileCondOp>(&cOp)) continue;
          emitOp(&cOp);
        }
        os << getIndent() << "continue;\n";
      } else {
        emitOp(&bodyOp);
      }
    }

    decIndent();
    os << getIndent() << "}\n";

    // Map while results
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      if (condOp && i < condOp.getArgs().size())
        valueNames[op.getResult(i)] = getName(condOp.getArgs()[i]);
    }
  }

  void emitBreak(coir::CoIRBreakOp op) {
    os << getIndent() << "break;\n";
  }

  void emitContinue(coir::CoIRContinueOp op) {
    os << getIndent() << "continue;\n";
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

    if (sctx.types_cute_header) {
      os << "cat > \"$TMPDIR/choreo_types_cute.h\" "
            "<< '__COCC_TYPES_CUTE_HEADER__'\n";
      os << sctx.types_cute_header;
      os << "\n__COCC_TYPES_CUTE_HEADER__\n\n";
    }
    if (sctx.cute_header) {
      os << "cat > \"$TMPDIR/choreo_cute.h\" << '__COCC_CUTE_HEADER__'\n";
      os << sctx.cute_header;
      os << "\n__COCC_CUTE_HEADER__\n\n";
    }
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

  os << "if [[ \"${1:-}\" == \"--execute\" ]]; then\n";
  os << "  shift\n";
  os << "  \"$BINFILE\" \"$@\"\n";
  os << "fi\n";
}

class CUDATargetCodeGen : public CoIR::CodeGen {
public:
  int EmitSource(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    coir::emitCUDA(module, os);
    emitHostCode(module, os);
    return 0;
  }

  int EmitScript(mlir::ModuleOp module, llvm::StringRef /*arch*/,
                 llvm::raw_ostream &os) override {
    emitScriptHeader(os);
    os << "TMPFILE=\"$TMPDIR/kernel.cu\"\n";
    os << "BINFILE=\"$TMPDIR/kernel\"\n\n";
    os << "cat > \"$TMPFILE\" << '__COCC_CUDA_SOURCE__'\n";
    coir::emitCUDA(module, os);
    emitHostCode(module, os);
    emitScriptFooter(os);
    return 0;
  }
};

static bool registered_gpu = [] {
  CoIR::CodeGenRegistry::Register("cute", [] {
    return std::make_unique<CUDATargetCodeGen>();
  });
  return true;
}();

} // namespace
