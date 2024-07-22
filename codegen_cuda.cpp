#include "codegen_cuda.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "cuda_script.inc"
#include "types.hpp"

#ifndef __CHOREO_CUDA_DIR__
#error "missing macro definition of __CHOREO_CUDA_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::CUDA;


bool CUDACodeGen::BeforeVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  return 0;
}

// CLEAN
bool CUDACodeGen::AfterVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  return 0;
}

bool CUDACodeGen::Visit(AST::MultiNodes &) { return true; }
bool CUDACodeGen::Visit(AST::MultiValues &) { return true; }
bool CUDACodeGen::Visit(AST::IntLiteral &) { return true; };
bool CUDACodeGen::Visit(AST::Boolean &) { return true; };
bool CUDACodeGen::Visit(AST::Expr &) { return true; };
bool CUDACodeGen::Visit(AST::MultiDimSpans &) { return true; };
bool CUDACodeGen::Visit(AST::NamedTypeDecl &) { return true; };

bool CUDACodeGen::Visit(AST::NamedVariableDecl &node) {
  __TRACE_EACH_VISIT__(node)
  return true;
};
bool CUDACodeGen::Visit(AST::IntTuple &) { return true; };
bool CUDACodeGen::Visit(AST::Assignment &) { return true; };
bool CUDACodeGen::Visit(AST::IntIndex &) { return true; };
bool CUDACodeGen::Visit(AST::DataType &) { return true; };

bool CUDACodeGen::Visit(AST::Identifier &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool CUDACodeGen::Visit(AST::Parameter &p) {
  __TRACE_EACH_VISIT__(p)
  return true;
}

bool CUDACodeGen::Visit(AST::ParamList &pl) {
  __TRACE_EACH_VISIT__(pl)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::ParallelBy &by) {
  __TRACE_EACH_VISIT__(by)
  return true;
}

bool CUDACodeGen::Visit(AST::WhereBind &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::WithIn &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
};

bool CUDACodeGen::Visit(AST::WithBlock &) { return true; }

bool CUDACodeGen::Visit(AST::Memory &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::DMA &d) {
  __TRACE_EACH_VISIT__(d)
  return true;
}

bool CUDACodeGen::Visit(AST::ChunkAt &) { return true; }

bool CUDACodeGen::Visit(AST::Wait &w) {
  __TRACE_EACH_VISIT__(w)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::Call &c) {
  __TRACE_EACH_VISIT__(c)
  return true;
}

bool CUDACodeGen::Visit(AST::Select &c) {
  __TRACE_EACH_VISIT__(c)
  return true;
}

bool CUDACodeGen::Visit(AST::Return &returnNode) {
  __TRACE_EACH_VISIT__(returnNode)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::ForeachBlock &forNode) {
  __TRACE_EACH_VISIT__(forNode)
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::FunctionDecl &d) {
  __TRACE_EACH_VISIT__(d)
  return true;
}

bool CUDACodeGen::Visit(AST::ChoreoFunction &) { return true; }

bool CUDACodeGen::Visit(AST::CppSourceCode &n) {
  __TRACE_EACH_VISIT__(n)
  return true;
}

bool CUDACodeGen::Visit(AST::Program &) { return true; }

void CUDACodeGen::EmitHostHead(std::ostream &os) {
  os <<
      R"(
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// dependant on the topsruntime
#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"

// choreo header
#include "choreo.h"

using namespace choreo;

namespace {

int64_t SizeOfRankedMemref(size_t rank) {
  return sizeof(topsMemref) +
         rank * sizeof(reinterpret_cast<topsMemref *>(0)->data[0]);
}

struct topsUnrankedMemref CreateUnrankedMemref(void *dev_mem, char *memref_raw,
                                               std::vector<int64_t> shape) {
  struct topsUnrankedMemref unranked_memref;
  unranked_memref.rank = shape.size();
  unranked_memref.ranked_memref = reinterpret_cast<topsMemref *>(memref_raw);

  // Set address
  uint64_t dev_addr = reinterpret_cast<uint64_t>(dev_mem);
  unranked_memref.ranked_memref->high_addr =
      reinterpret_cast<int32_t *>((dev_addr >> 32) & 0xFFFFFFFF);
  unranked_memref.ranked_memref->low_addr =
      reinterpret_cast<int32_t *>(dev_addr & 0xFFFFFFFF);

  // Set offset & shape
  unranked_memref.ranked_memref->offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    unranked_memref.ranked_memref->data[i] = shape[i];
  }

  return unranked_memref;
}

// Nasty data copy. Need optimization together with cuda
template <typename T, int Rank>
static inline std::vector<uint8_t>
ToCUDAData(const spanned_view<T, Rank> &v) {
  return std::vector<uint8_t>((const uint8_t *)(v.data()), v.bytes());
}

template <int N, typename T, typename U>
static inline spanned_data<T, N>
ToSpanned(const std::vector<U> &v, std::initializer_list<int> && shape) {
  return copy_as_spanned<N, T>((T*)v.data(), v.size() * sizeof(U), shape);
}

// must be true
//#define CHECK(a) choreo_assert((a), "", __FILE__, __LINE__)
#define CHECK(a) (a)

} // end anonymous namespace
)";
}

void CUDACodeGen::EmitHostFuncBody(std::ostream &os, const Type &ty,
                                     const std::string &f_n,
                                     const std::string &out_size,
                                     const std::string &out_type,
                                     const Shape &out_shape) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

  // phase 1: create tops executable from a file
  os << "{\n";
  EmitRuntimeCheck(os, ty);
  os << R"(
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(\"" << f_n << "\", std::ios::binary);";
  os << R"(
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create stream
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

)";

  // phase 2: allocate device memory and copy
  std::vector<std::string> device_mems;
  for (auto &p : param_map) {
    auto mem_name = "in_mem" + std::to_string(device_mems.size());
    os << "  void *" << mem_name << " = nullptr;\n";
    os << "  CHECK(topsMalloc(&" << mem_name << ", " << p.second << "));\n";
    os << "  CHECK(topsMemcpy(" << mem_name << ", reinterpret_cast<void *>("
       << p.first << ".data()), " << p.second
       << ", topsMemcpyHostToDevice));\n";
    device_mems.push_back(mem_name);
  }
  os << "  void * device_inputs[] = {" << DelimitedString(device_mems)
     << "};\n\n";

  std::string size_string = ReplaceRuntimeNames(out_size);

  if (!out_size.empty()) {
    os << "  void * out_mem = nullptr;\n";
    os << "  CHECK(topsMalloc(&out_mem, " << size_string << "));\n";
    os << "  void *device_outputs[] = {out_mem};\n";
  }

  std::vector<std::string> inputs;  // cuda input paramters

  os << "\n  // adaption: convert to the cuda parameters\n";
  size_t i = 0;
  std::ostringstream tss;
  for (; i < fty.in_tys.size(); ++i) {
    tss << "  std::unique_ptr<char[]> memref_raw" << i
        << "(new char[SizeOfRankedMemref(";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
      tss << sty->Dims();
    else
      tss << "1";
    tss << ")]);\n";
    tss << "  auto input" << i << " = CreateUnrankedMemref(in_mem" << i
        << ", memref_raw" << i << ".get(), ";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
      tss << LSTR(sty->GetShape());
    } else
      tss << "{1}";
    tss << ");\n";
    inputs.push_back("input" + std::to_string(i));
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    tss << "  std::vector<int64_t> out_shape = {" << RSTR(rty->GetShape())
        << "};\n";
  } else if (!isa<VoidType>(fty.out_ty)) {
    tss << "  std::vector<int64_t> out_shape = {1};\n";
  }
  os << ReplaceRuntimeNames(tss.str(), "(int64_t)");

  if (!void_return) {
    os << "  std::unique_ptr<char[]> memref_raw" << i
       << "(new char[SizeOfRankedMemref(out_shape.size())]);\n";
    os << "  auto output = CreateUnrankedMemref(out_mem, memref_raw" << i
       << ".get(), out_shape);\n";
  }

  // phase 3: Execute the executable and fetch the output
  os << "\n  " << current_fn << "(";
  for (auto &in : inputs) os << "&" << in << ", ";
  os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
  os << "  CHECK(topsStreamSynchronize(stream));\n";

  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (out_shape.IsValid()) {
    out_rank = out_shape.Dims();
    shape_string = ReplaceRuntimeNames(LSTR(out_shape));
  }

  if (!out_size.empty()) {
    os << "  auto res = choreo::make_spandata<" << out_type << ", " << out_rank
       << ">(" << shape_string << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CHECK(topsMemcpy(reinterpret_cast<void *>(res.data()), out_mem,\n";
    os << "                  " << size_string
       << ", topsMemcpyDeviceToHost));\n";
  }

  // phase 4: Free up the resources
  os << "  // Free up the resources\n";
  for (auto &p : device_mems) os << "  topsFree(" << p << ");\n";
  os << "  topsFree(out_mem);\n\n";
  os << "  // TODO: figure out why stream destroying crash some "
        "applications.\n";
  os << "  // topsStreamDestroy(stream);\n";
  os << "  return res;\n";
  os << "}\n";
}

std::string CUDACodeGen::ReplaceRuntimeNames(const std::string &e,
                                               const std::string &prefix,
                                               bool host_code) {
  std::string expr = e;
  for (auto &s : rts_nmap) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      if (host_code)
        expr.replace(pos, s.first.length(), prefix + s.second);
      else
        expr.replace(pos, s.first.length(), "-1");
    }
  }
  return expr;
}

std::string CUDACodeGen::ReplaceDynDimName(const std::string &e) {
  std::string expr = e;
  for (auto &s : rts_nidx) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      std::string dim_value = "dim_(args[" + std::to_string(rts_pidx[s.first]) +
                              "], " + std::to_string(s.second) + ")";
      expr.replace(pos, s.first.length(), dim_value);
    }
  }
  return expr;
}

void CUDACodeGen::EmitRuntimeCheck(std::ostream &os, const Type &ty) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

  assert(fty.in_tys.size() == host_params.size() &&
         "internal error when dealing with the host parameter size.");

  // check if the input shape is as declared in choreo
  if (fty.in_tys.size() == 0) return;

  if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
    auto name = host_params[0];
    size_t count = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<int>(&vi)) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
        os << ", \"shape inconstant on 1st parameter (dim: " << count
           << ").\");\n";
      }
      count++;
    }
  }
  for (size_t i = 1; i < fty.in_tys.size(); ++i) {
    auto name = host_params[i];
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
      size_t count = 0;
      for (auto vi : sty->GetShape().Value()) {
        if (auto vale = dyn_cast<int>(&vi)) {
          auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
          os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
          os << ", \"shape inconstant on " << i + 1
             << "th parameter (dim: " << count << ").\");\n";
        }
        count++;
      }
    }
  }
}

void CUDACodeGen::EmitHostFuncDecl(std::ostream &os, const Type &ty,
                                     const std::string &n, bool decl_only) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);
  assert(host_params.size() == fty.in_tys.size() &&
         "inconsistent parameter count.");

  // emit the return type
  os << HostTypeString(*fty.out_ty, true) << " " << n << "(";

  if (fty.in_tys.size() > 0) {
    if (!decl_only) {
      if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
        param_map.push_back(std::make_pair(
            host_params[0], ReplaceRuntimeNames(sty->ByteSizeExpression())));
      } else
        param_map.push_back(std::make_pair(host_params[0], "1"));
    }
    os << HostTypeString(*fty.in_tys[0]) << " " << host_params[0];
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      if (!decl_only) {
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
          param_map.push_back(std::make_pair(
              host_params[i], ReplaceRuntimeNames(sty->ByteSizeExpression())));
        } else
          param_map.push_back(std::make_pair(host_params[i], "1"));
      }
      os << ", " << HostTypeString(*fty.in_tys[i]) << " " << host_params[i];
    }
  }
  os << ")" << ((decl_only) ? ";\n" : " ");
}

void CUDACodeGen::OutputScript(FunctionType *fty, const std::string &n,
                                 const std::string &out_type,
                                 const std::string &out_size,
                                 const Shape &out_shape) {
  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + n;

  std::string kernel_fn = build_prefix + "_micro_kernel.cpp";
  std::string cuda_fn = build_prefix + "_cuda.cpp";
  std::string cuda_bfn =
      build_path + "/${gcu_target_string}_lib" + current_fn + ".o";
  host_fn = build_prefix + "_host.cpp";
  target_fn = "__choreo_" + n;

  // Generate the host code
  std::string user_code = hs.str();
  hs.clear();

  EmitHostHead(hs);
  if (!user_code.empty()) {
    // user code needs the choreo function decal for call
    EmitHostFuncDecl(hs, *fty, n, true);
    hs << user_code;
  }
  EmitHostFuncDecl(hs, *fty, n);
  EmitHostFuncBody(hs, *fty, cuda_bfn, out_size, out_type, out_shape);

  // backpatch the cuda bin filename
  std::string cuda_src = fs.str();
  if (!alloc_in_fs.str().empty())
    cuda_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(cuda_src, std::string("$$out$$"), output_v);
  ReplaceInString(cuda_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os << "#!/usr/bin/env bash\n\n";
  os << "# This is the choreo generated bash script to compile cuda code\n";

  os << R"script(
  # check the device
  # TODO: improve the target check with more solid code
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
    export TOP_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  elif [[ "$(lspci | grep Tencent)" != "" ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  else
    echo "can not determine the GCU device type."
    exit 1
  fi
)script";
  os << "\n# step 0: set up the environment\n";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n";
  os << "cat <<'EOF' > " << build_path << "/cuda_script.sh\n";
  os << __cuda_script_as_string << "\nEOF\n";
  os << "chmod +x " << build_path << "/cuda_script.sh\n";
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";

  os << "\n# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<'EOF' > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n";

  os << "\n# step 2: write the cuda source code into a temp file\n";
  os << "cuda_src=" << cuda_fn << "\n";
  os << "cat <<'EOF' > ${cuda_src}\n";
  os << cuda_src << "\nEOF\n\n";

  os << "\n# step 3: set the cuda binary file name\n";
  os << "cuda_bin=" << cuda_bfn << "\n";
}
