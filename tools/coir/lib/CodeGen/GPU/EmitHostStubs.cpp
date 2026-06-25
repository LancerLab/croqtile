//===- EmitHostStubs.cpp - Host entry stub generation from CoIR metadata --===//
//
// For each coir.kernel op, emit a __host__ C++ function that:
//   1. Loads the PTX module (via a static helper).
//   2. Shadows host spanned params to device memory (cudaMalloc + H2D).
//   3. Launches the kernel via cuLaunchKernel.
//   4. Copies results back (D2H) and frees shadow buffers.
//   5. Returns a choreo::spanned_data for tensor outputs.
//
// The kernel is resolved by name from the PTX module at runtime, so the
// stubs use CUDA Driver API (cuda.h), not <<<>>> syntax.
//
//===----------------------------------------------------------------------===//

#include "CodeGen/GPU/EmitHostStubs.h"
#include "Dialect/CoIR/CoIROps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

// ParamAttr enum mirrors Choreo::ParamAttr.
enum ParamAttr : int { NONE = 0, SHADOW_TO_GLOBAL = 1, GLOBAL_INPUT = 2 };

struct KernelParamInfo {
  std::string name;
  int attr;
  bool pass_by_ref;
  std::string host_elem_type; // e.g. "s32", "f16" -- empty for scalars
  int64_t dims;               // tensor rank, 0 for scalars
};

struct KernelInfo {
  std::string name;
  std::vector<KernelParamInfo> params;
  std::string ret_elem_type; // empty if void
  int64_t ret_dims = 0;
  int64_t gridDims[3] = {1, 1, 1};
  int64_t blockDims[3] = {1, 1, 1};
};

/// Return the choreo:: qualified C++ type for a choreo element type name.
/// Since the generated stubs include choreo_types.h, choreo::s32 etc. resolve
/// to the same fundamental types (int, float, ...) without a separate mapping.
std::string elemTypeToCpp(llvm::StringRef et) {
  return "choreo::" + et.str();
}

/// Build the choreo::spanned_view<choreo::ET, N> parameter type string.
std::string viewType(llvm::StringRef et, int64_t dims) {
  return "const choreo::spanned_view<choreo::" + et.str() + ", " +
         std::to_string(dims) + ">&";
}

/// Build the choreo::spanned_data<choreo::ET, N> return type string.
std::string dataType(llvm::StringRef et, int64_t dims) {
  return "choreo::spanned_data<choreo::" + et.str() + ", " +
         std::to_string(dims) + ">";
}

std::vector<KernelInfo> collectKernels(ModuleOp module) {
  std::vector<KernelInfo> kernels;
  module.walk([&](coir::KernelOp kop) {
    KernelInfo ki;
    ki.name = kop.getSymName().str();

    auto namesAttr = kop->getAttrOfType<ArrayAttr>("coir.param_names");
    auto attrsAttr = kop->getAttrOfType<ArrayAttr>("coir.param_attrs");
    auto refsAttr = kop->getAttrOfType<ArrayAttr>("coir.param_refs");
    auto elemAttr = kop->getAttrOfType<ArrayAttr>("coir.host_elem_types");
    auto dimsAttr = kop->getAttrOfType<ArrayAttr>("coir.param_dims");

    size_t n = namesAttr ? namesAttr.size() : 0;
    for (size_t i = 0; i < n; ++i) {
      KernelParamInfo pi;
      pi.name = mlir::cast<StringAttr>(namesAttr[i]).getValue().str();
      pi.attr = attrsAttr
                    ? mlir::cast<IntegerAttr>(attrsAttr[i]).getInt()
                    : 0;
      pi.pass_by_ref =
          refsAttr ? mlir::cast<BoolAttr>(refsAttr[i]).getValue() : false;
      pi.host_elem_type =
          elemAttr ? mlir::cast<StringAttr>(elemAttr[i]).getValue().str()
                   : "";
      pi.dims =
          dimsAttr ? mlir::cast<IntegerAttr>(dimsAttr[i]).getInt() : 0;
      ki.params.push_back(std::move(pi));
    }

    if (auto retAttr =
            kop->getAttrOfType<StringAttr>("coir.host_ret_elem_type"))
      ki.ret_elem_type = retAttr.getValue().str();
    if (auto retDims = kop->getAttrOfType<IntegerAttr>("coir.host_ret_dims"))
      ki.ret_dims = retDims.getInt();

    kop.getBody().walk([&](coir::ParallelOp par) {
      auto bounds = par.getBounds();
      auto lvl = par.getLevel();
      int64_t *target = nullptr;
      if (lvl == coir::ParallelLevel::BLOCK)
        target = ki.gridDims;
      else if (lvl == coir::ParallelLevel::THREAD)
        target = ki.blockDims;
      if (!target) return;
      for (unsigned i = 0; i < bounds.size() && i < 3; ++i)
        target[i] = bounds[i];
    });

    kernels.push_back(std::move(ki));
  });
  return kernels;
}

void emitStaticHelper(std::ostringstream &os) {
  os << "// --- CUDA Driver API helper (emitted once) ---\n";
  os << "#include <cuda.h>\n";
  os << "#include <cstdlib>\n";
  os << "#include <cstring>\n";
  os << "#include <iostream>\n";
  os << "\n";
  os << "namespace __coir_rt {\n";
  os << "static CUmodule __ptx_module = nullptr;\n";
  os << "static CUcontext __ctx = nullptr;\n";
  os << "\n";
  os << "static void ensureInit(const char* ptx) {\n";
  os << "  if (__ptx_module) return;\n";
  os << "  CUresult err;\n";
  os << "  err = cuInit(0);\n";
  os << "  if (err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuInit failed: \" << err << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "  CUdevice dev;\n";
  os << "  err = cuDeviceGet(&dev, 0);\n";
  os << "  if (err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuDeviceGet failed: \" << err << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "  err = cuCtxCreate(&__ctx, 0, dev);\n";
  os << "  if (err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuCtxCreate failed: \" << err << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "  err = cuModuleLoadData(&__ptx_module, ptx);\n";
  os << "  if (err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuModuleLoadData failed: \" << err << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "}\n";
  os << "\n";
  os << "static CUfunction getKernel(const char* name) {\n";
  os << "  CUfunction fn;\n";
  os << "  CUresult err = cuModuleGetFunction(&fn, __ptx_module, name);\n";
  os << "  if (err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuModuleGetFunction(\" << name << \") failed: \""
     << " << err << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "  return fn;\n";
  os << "}\n";
  os << "} // namespace __coir_rt\n\n";
}

void emitKernelStub(std::ostringstream &os, const KernelInfo &ki) {
  bool has_tensor_ret =
      !ki.ret_elem_type.empty() && ki.ret_dims > 0;

  // Return type.
  if (has_tensor_ret)
    os << dataType(ki.ret_elem_type, ki.ret_dims);
  else
    os << "void";

  // Function name + params.
  os << " " << ki.name << "(";
  for (size_t i = 0; i < ki.params.size(); ++i) {
    auto &p = ki.params[i];
    if (i > 0) os << ", ";
    if (!p.host_elem_type.empty() && p.dims > 0)
      os << viewType(p.host_elem_type, p.dims) << " " << p.name;
    else
      os << elemTypeToCpp(p.host_elem_type) << " " << p.name;
  }
  os << ") {\n";

  // Lazy init the CUDA driver + PTX module.
  os << "  __coir_rt::ensureInit(__coir_ptx_string);\n";
  std::string ptxKernelName = ki.name + "_kernel";
  os << "  CUfunction __fn = __coir_rt::getKernel(\"" << ptxKernelName
     << "\");\n\n";

  // Shadow host params to device memory.
  for (size_t i = 0; i < ki.params.size(); ++i) {
    auto &p = ki.params[i];
    if (p.host_elem_type.empty() || p.dims == 0)
      continue;
    std::string cppType = elemTypeToCpp(p.host_elem_type);
    std::string devSym = p.name + "__dev";
    if (p.attr == GLOBAL_INPUT) {
      os << "  " << cppType << "* " << devSym << " = const_cast<"
         << cppType << "*>(" << p.name << ".data());\n";
    } else {
      os << "  " << cppType << "* " << devSym << " = nullptr;\n";
      os << "  cuMemAlloc((CUdeviceptr*)&" << devSym << ", "
         << p.name << ".element_count() * sizeof(" << cppType << "));\n";
      os << "  cuMemcpyHtoD((CUdeviceptr)" << devSym << ", "
         << p.name << ".data(), " << p.name << ".element_count() * sizeof("
         << cppType << "));\n";
    }
  }

  // Output buffer.
  if (has_tensor_ret) {
    std::string retCpp = elemTypeToCpp(ki.ret_elem_type);
    // Infer output size from first tensor param (same span convention).
    std::string sizeExpr;
    for (auto &p : ki.params) {
      if (!p.host_elem_type.empty() && p.dims > 0) {
        sizeExpr = p.name + ".element_count()";
        break;
      }
    }
    if (sizeExpr.empty()) sizeExpr = "1";
    os << "  " << retCpp << "* __out_dev = nullptr;\n";
    os << "  cuMemAlloc((CUdeviceptr*)&__out_dev, " << sizeExpr
       << " * sizeof(" << retCpp << "));\n\n";
  }

  // Build kernel args array.  The GPU kernel expects raw pointers + memref
  // ABI fields, but the ConvertToGPU pass lowers tensor args to memref,
  // which in the LLVM lowering becomes (base_ptr, aligned_ptr, offset,
  // sizes..., strides...).  For 1D tensors that is 5 args per param.
  // We pass the device pointers through a void* array with flat layout.
  os << "  // Kernel args: each tensor -> (ptr, ptr, 0, size, 1)\n";
  os << "  size_t __nargs = 0;\n";

  // Count args.
  int argIdx = 0;
  for (auto &p : ki.params) {
    if (!p.host_elem_type.empty() && p.dims > 0)
      argIdx += 2 + 1 + p.dims + p.dims; // base, aligned, offset, sizes, strides
    else
      argIdx += 1;
  }
  if (has_tensor_ret)
    argIdx += 2 + 1 + ki.ret_dims + ki.ret_dims;

  os << "  void* __args[" << argIdx << "];\n";

  // Populate args.
  int idx = 0;
  for (auto &p : ki.params) {
    if (!p.host_elem_type.empty() && p.dims > 0) {
      std::string devSym = p.name + "__dev";
      os << "  static " << elemTypeToCpp(p.host_elem_type) << "* __pa_"
         << p.name << ";\n";
      os << "  __pa_" << p.name << " = " << devSym << ";\n";
      os << "  __args[" << idx << "] = &__pa_" << p.name << ";\n";
      idx++;
      os << "  __args[" << idx << "] = &__pa_" << p.name << ";\n";
      idx++;
      os << "  static int64_t __off_" << p.name << " = 0;\n";
      os << "  __args[" << idx << "] = &__off_" << p.name << ";\n";
      idx++;
      for (int64_t d = 0; d < p.dims; ++d) {
        os << "  static int64_t __sz_" << p.name << "_" << d << " = "
           << p.name << ".shape()[" << d << "];\n";
        os << "  __args[" << idx << "] = &__sz_" << p.name << "_" << d
           << ";\n";
        idx++;
      }
      for (int64_t d = 0; d < p.dims; ++d) {
        // Row-major stride: 1 for the last dimension.
        if (d == p.dims - 1) {
          os << "  static int64_t __st_" << p.name << "_" << d << " = 1;\n";
        } else {
          os << "  static int64_t __st_" << p.name << "_" << d << " = "
             << p.name << ".shape()[" << (d + 1) << "];\n";
        }
        os << "  __args[" << idx << "] = &__st_" << p.name << "_" << d
           << ";\n";
        idx++;
      }
    } else {
      os << "  auto __sc_" << p.name << " = " << p.name << ";\n";
      os << "  __args[" << idx << "] = &__sc_" << p.name << ";\n";
      idx++;
    }
  }

  if (has_tensor_ret) {
    os << "  static " << elemTypeToCpp(ki.ret_elem_type)
       << "* __pa_out;\n";
    os << "  __pa_out = __out_dev;\n";
    os << "  __args[" << idx << "] = &__pa_out;\n";
    idx++;
    os << "  __args[" << idx << "] = &__pa_out;\n";
    idx++;
    os << "  static int64_t __off_out = 0;\n";
    os << "  __args[" << idx << "] = &__off_out;\n";
    idx++;
    std::string refParam;
    for (auto &p : ki.params) {
      if (!p.host_elem_type.empty() && p.dims > 0) {
        refParam = p.name;
        break;
      }
    }
    for (int64_t d = 0; d < ki.ret_dims; ++d) {
      if (!refParam.empty())
        os << "  static int64_t __sz_out_" << d << " = " << refParam
           << ".shape()[" << d << "];\n";
      else
        os << "  static int64_t __sz_out_" << d << " = 1;\n";
      os << "  __args[" << idx << "] = &__sz_out_" << d << ";\n";
      idx++;
    }
    for (int64_t d = 0; d < ki.ret_dims; ++d) {
      if (d == ki.ret_dims - 1) {
        os << "  static int64_t __st_out_" << d << " = 1;\n";
      } else {
        os << "  static int64_t __st_out_" << d << " = "
           << refParam << ".shape()[" << (d + 1) << "];\n";
      }
      os << "  __args[" << idx << "] = &__st_out_" << d << ";\n";
      idx++;
    }
  }

  os << "  __nargs = " << idx << ";\n\n";

  os << "  CUresult __launch_err = cuLaunchKernel(\n";
  os << "      __fn, " << ki.gridDims[0] << ", " << ki.gridDims[1] << ", "
     << ki.gridDims[2] << ", " << ki.blockDims[0] << ", " << ki.blockDims[1]
     << ", " << ki.blockDims[2] << ",\n";
  os << "      0, nullptr, __args, nullptr);\n";
  os << "  if (__launch_err != CUDA_SUCCESS) {\n";
  os << "    std::cerr << \"cuLaunchKernel failed: \" << __launch_err"
     << " << std::endl;\n";
  os << "    std::exit(1);\n";
  os << "  }\n";
  os << "  cuCtxSynchronize();\n\n";

  // Copy-back for pass-by-ref params.
  for (auto &p : ki.params) {
    if (p.host_elem_type.empty() || p.dims == 0) continue;
    if (p.attr == GLOBAL_INPUT) continue;
    if (p.pass_by_ref) {
      std::string cppType = elemTypeToCpp(p.host_elem_type);
      os << "  cuMemcpyDtoH(const_cast<" << cppType << "*>("
         << p.name << ".data()), (CUdeviceptr)" << p.name
         << "__dev, " << p.name << ".element_count() * sizeof(" << cppType
         << "));\n";
    }
  }

  // Copy-back output.
  if (has_tensor_ret) {
    std::string retCpp = elemTypeToCpp(ki.ret_elem_type);
    std::string sizeExpr;
    for (auto &p : ki.params) {
      if (!p.host_elem_type.empty() && p.dims > 0) {
        sizeExpr = p.name + ".element_count()";
        break;
      }
    }
    if (sizeExpr.empty()) sizeExpr = "1";
    if (ki.ret_dims > 1) {
      // Multi-dimensional return: pass individual dimension sizes.
      os << "  auto __result = choreo::make_spandata<choreo::"
         << ki.ret_elem_type << ">(";
      std::string refParam;
      for (auto &p : ki.params) {
        if (!p.host_elem_type.empty() && p.dims > 0) {
          refParam = p.name;
          break;
        }
      }
      for (int64_t d = 0; d < ki.ret_dims; ++d) {
        if (d > 0) os << ", ";
        if (!refParam.empty() && d < ki.ret_dims)
          os << refParam << ".shape()[" << d << "]";
        else
          os << "1";
      }
      os << ");\n";
    } else {
      os << "  auto __result = choreo::make_spandata<choreo::"
         << ki.ret_elem_type << ">(" << sizeExpr << ");\n";
    }
    os << "  cuMemcpyDtoH(__result.data(), (CUdeviceptr)__out_dev, "
       << sizeExpr << " * sizeof(" << retCpp << "));\n";
  }

  // Free shadow device buffers.
  for (auto &p : ki.params) {
    if (p.host_elem_type.empty() || p.dims == 0) continue;
    if (p.attr == GLOBAL_INPUT) continue;
    os << "  cuMemFree((CUdeviceptr)" << p.name << "__dev);\n";
  }
  if (has_tensor_ret)
    os << "  cuMemFree((CUdeviceptr)__out_dev);\n";

  if (has_tensor_ret)
    os << "  return __result;\n";

  os << "}\n\n";
}

} // namespace

std::string coir::gpu::emitHostStubs(ModuleOp module) {
  auto kernels = collectKernels(module);
  if (kernels.empty())
    return "";

  std::ostringstream os;
  os << "// --- Host entry stubs generated by CoIR ---\n\n";
  emitStaticHelper(os);

  for (auto &ki : kernels)
    emitKernelStub(os, ki);

  return os.str();
}
