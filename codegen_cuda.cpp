#include "codegen_cuda.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_cuda_header.inc"
#include "codegen.hpp"
#include "cuda_script.inc"
#include "types.hpp"

#ifndef __CHOREO_CUDA_DIR__
#error "missing macro definition of __CHOREO_CUDA_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::CUDA;

#define __TRACE_EACH_VISIT__(d)       \
  if (trace_visit) {                  \
    os << d.TypeNameString() << ": "; \
    os << "\n";                       \
  }


bool CUDACodeGen::BeforeVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  return 0;
}

// CLEAN
bool CUDACodeGen::AfterVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  if (isa<AST::Program>(&n)) {
    os << "\n# step 4: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "cat <<'EOF' >> ${host_src}\n";
    os << hs.str()
       << "\nEOF\n\n";

    os << "\n# step 5: JIT compile and execute\n";
    os << "# TODO: enable workflow of AOT compilation\n";
    os << "target=" << target_fn << "\n";
    os << "build_path=" << build_path << "\n";
    os << "cuda_script=" << build_path << "/cuda_script.sh\n";
    os << R"(
if command -v nvim &> /dev/null
then
  EDITOR=nvim
else
  EDITOR=less
fi
if [ "$#" -ne 1 ]; then
    echo "    Usage: $0 | --execute           -> compile and execute choreo in cuda
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --list-sources      -> show the tree view of all sources
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
fi
)";
    os << R"(
if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then
)";
    os << "  export cuda_INSTALL=" << STRINGIZE(__CHOREO_cuda_DIR__)
       << "\n# JIT compile and execute\n";
    if (dyn_shaped) os << "VIEW_CONFIG=1 ENABLE_DYNSHAPE=1 ";
    // os << "${cuda_script} ${build_path} ./demos/cuda/test_dir/sgemm_main.cu ${target}\n";
    os << "${cuda_script} ${build_path} ${host_src} ${target}\n";
    os << "diff ${host_src} ./demos/cuda/test_dir/sgemm_main.cu";
    os << R"script(
elif [ "$1" == "--list-sources" ]; then
  tree ${build_path} -L 1
elif [ "$1" == "--statistics" ]; then
  echo ">>>> Line of Code without Choreo"
  wc -l ${cuda_src} ${host_src} ${kernel_src}
  echo ">>>> Line of Code with Choreo"
  wc -l ~/choreo/demo/elementwise_add.co
  # grep -v '^ *//' ~/choreo/demo/elementwise_add.co | wc -l
elif [ "$1" == "--show-kernel" ]; then
  ${EDITOR} ${kernel_src}
elif [ "$1" == "--show-host" ]; then
  ${EDITOR} ${host_src}
elif [ "$1" == "--show-tileflow" ]; then
  ${EDITOR} ${cuda_src}
elif [ "$1" == "--show-choreo" ]; then
  ${EDITOR} ~/choreo/demo/elementwise_add.co
else
    echo "    Usage: $0 | --execute           -> compile and execute choreo in cuda
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --list-sources      -> show the tree view of all sources
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
fi
)script";

  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    entry_fn = f->name;
    current_fn = "__choreo_" + entry_fn;
    auto fty = cast<FunctionType>(f->GetType());
    auto &out_type = fty->out_ty;
    auto out_size = GetByteSizeExprOf(*out_type);
    // fs << "}\n\nMODULE_REGISTER(\"lib" << current_fn << "\", " << current_fn
    //    << ");";  // end the cuda function definition
    // TODO(albert): resolve hardcode
    fs << R"(
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  // blockIdx.y * BLOCKSIZE
  // threadIdx.x++ => x++ | y==
  // x to M

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
)";

    if (auto sty = dyn_cast<SpannedType>(out_type)) {
      OutputScript(fty, f->name, GetBaseTypeStringOf(*out_type), out_size,
                   sty->GetShape());
    } else
      OutputScript(fty, f->name, GetBaseTypeStringOf(*out_type), out_size,
                   Shape() /*invalid shape*/);
    ResetBuffers();
  } 
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
      R"(// choreo header
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <string>
#include <chrono>

#include <choreo_cuda.h>

using namespace choreo;
using namespace choreo::cuda;

// extern "C" void kernel(int * lhs, int * rhs, int * out, int m, int k, int n) {
 /// end of kernel decl

// UPDATE: from kernel-6, it only occupies half of the L1 mem, which is not optimised enough for reuse.
// for kernel-7, let us try a <256,256,256> setting, where occupies 3/4 of L1 MEM

// Analysis 1:
//
// HW: in DORADO: 1 card = 2 clusters = 6 csb = 6 L2 = 24 SIP = 24 L1, each CSB = 8MB, each L1 = 1 MB
//
// GMEM: unchanged from kernel 5 
// SMEM: unchanged from kernel 5 
// LMEM: lhs_load_s=<64x1024>=256KB, rhs_load=<1024x64>=256KB, l2_out=<64x64>=4KB  < 1MB

// Analysis: compute intensity
// kernel 6 calculate <32x32> results per thread requires:
//   32x1024 loads from lhs
//   32x1024 loads from rhs
//   32x32 loads and stores from out
//   => 65 loads + 1 store per result
//
// kernel 7 calculate <256x256> results per thread requires:
//   256x256x4 loads from lhs
//   256x256x4 loads from rhs
//   256x256x4 loads and stores from out
//   => 12 loads and 4 store per result



namespace {

// Nasty data copy. Need optimization together with factor
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

choreo::spanned_data<choreo::f32, 2> ele_add(const choreo::spanned_view<choreo::f32, 2> & in_host0, const choreo::spanned_view<choreo::f32, 2> & in_host1) {
  choreo::runtime_check(in_host0.shape()[0] == 4096, "shape inconstant on 1st parameter (dim: 0).");
  choreo::runtime_check(in_host0.shape()[1] == 4096, "shape inconstant on 1st parameter (dim: 1).");
  choreo::runtime_check(in_host1.shape()[0] == 4096, "shape inconstant on 2th parameter (dim: 0).");
  choreo::runtime_check(in_host1.shape()[1] == 4096, "shape inconstant on 2th parameter (dim: 1).");

  int deviceIdx = 0;
  printf("Running on device %d.\n", deviceIdx);

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C


  float* in_mem0 = nullptr;
  float* in_mem1 = nullptr;
  float* out_mem = nullptr;
  float* out_mem_ref = nullptr;
  CUDACheck(cudaMalloc((void **)&in_mem0, sizeof(float) * in_host0.shape()[0] * in_host0.shape()[1]));
  CUDACheck(cudaMalloc((void **)&in_mem1, sizeof(float) * in_host1.shape()[0] * in_host1.shape()[1]));
  CUDACheck(cudaMalloc((void **)&out_mem, sizeof(float) * in_host0.shape()[0] * in_host1.shape()[1]));
  CUDACheck(cudaMalloc((void **)&out_mem_ref, sizeof(float) * in_host0.shape()[0] * in_host1.shape()[1]));

  CUDACheck(cudaMemcpy(in_mem0, in_host0.data(), sizeof(float) * in_host0.shape()[0] * in_host0.shape()[1], cudaMemcpyHostToDevice));
  CUDACheck(cudaMemcpy(in_mem1, in_host1.data(), sizeof(float) * in_host1.shape()[0] * in_host1.shape()[1], cudaMemcpyHostToDevice));

  // CHECK CORRECTNESS
  auto res_ref = choreo::make_spandata<choreo::f32, 2>({in_host0.shape()[0], in_host1.shape()[1]});
  run_kernel(0, in_host0.shape()[0], in_host1.shape()[1], in_host0.shape()[1], alpha, in_mem0, in_mem1, beta, out_mem_ref, handle);
  CUDACheck(cudaDeviceSynchronize());
  cudaMemcpy(res_ref.data(), out_mem_ref, sizeof(float) * in_host0.shape()[0] * in_host1.shape()[1], cudaMemcpyDeviceToHost);


  cudaEventRecord(beg);

  run_kernel(1, in_host0.shape()[0], in_host1.shape()[1], in_host0.shape()[1], alpha, in_mem0, in_mem1, beta, out_mem, handle);

  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  elapsed_time /= 1000.; // Convert to seconds

  unsigned long flops = 2 * in_host0.shape()[0] * in_host1.shape()[1] * in_host0.shape()[1];
  printf(
      "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
      "(%lu).\n",
      elapsed_time,
      (flops * 1e-9) / elapsed_time, in_host0.shape()[0]);
  fflush(stdout);

  // auto out_host = (float *)malloc(sizeof(float) * m * n);
  auto res = choreo::make_spandata<choreo::f32, 2>({in_host0.shape()[0], in_host1.shape()[1]});
  CUDACheck(cudaMemcpy(res.data(), out_mem, sizeof(float) * in_host0.shape()[0] * in_host1.shape()[1], cudaMemcpyDeviceToHost));

  if (!verify_matrix(res_ref.data(), res.data(), in_host0.shape()[0] * in_host1.shape()[1])) {
    std::cout
        << "Failed to pass the correctness verification against NVIDIA "
           "cuBLAS."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  cudaFree(in_mem0);
  cudaFree(in_mem1);
  cudaFree(out_mem);
  cublasDestroy(handle);

  std::cout
      << "SGEMM passed with correct results "
      << std::endl;

  return res;
};


int main() { /// host program
  unsigned long m, n, k;
  m = 4096;
  n = 4096;
  k = 4096;

  float* a = (float *)malloc(sizeof(float) * m * k);
  float* b = (float *)malloc(sizeof(float) * k * n);
  auto lhs_data = choreo::make_spanview<2>((float*)a, {m, k});
  auto rhs_data = choreo::make_spanview<2>((float*)b, {k, n});

  auto res = ele_add(lhs_data, rhs_data);
  return 0;
})";
}

// void CUDACodeGen::EmitHostFuncBody(std::ostream &os, const Type &ty,
//                                      const std::string &f_n,
//                                      const std::string &out_size,
//                                      const std::string &out_type,
//                                      const Shape &out_shape) {
//   assert(isa<FunctionType>(&ty) && "unexpected type.");
//   auto &fty = *cast<FunctionType>(&ty);
//
//   // phase 1: create tops executable from a file
//   os << "{\n";
//   EmitRuntimeCheck(os, ty);
//   os << R"(
//   std::vector<char> binary;
//   // Read bin file and store to a vector
// )";
//   os << "  std::ifstream ifs(\"" << f_n << "\", std::ios::binary);";
//   os << R"(
//   std::copy(std::istreambuf_iterator<char>(ifs),
//             std::istreambuf_iterator<char>(), std::back_inserter(binary));
//   ifs.close();
//
//   // Create stream
//   topsStream_t stream = nullptr;
//   CHECK(topsStreamCreate(&stream));
//
// )";
//
//   // phase 2: allocate device memory and copy
//   std::vector<std::string> device_mems;
//   for (auto &p : param_map) {
//     auto mem_name = "in_mem" + std::to_string(device_mems.size());
//     os << "  void *" << mem_name << " = nullptr;\n";
//     os << "  CHECK(topsMalloc(&" << mem_name << ", " << p.second << "));\n";
//     os << "  CHECK(topsMemcpy(" << mem_name << ", reinterpret_cast<void *>("
//        << p.first << ".data()), " << p.second
//        << ", topsMemcpyHostToDevice));\n";
//     device_mems.push_back(mem_name);
//   }
//   os << "  void * device_inputs[] = {" << DelimitedString(device_mems)
//      << "};\n\n";
//
//   std::string size_string = ReplaceRuntimeNames(out_size);
//
//   if (!out_size.empty()) {
//     os << "  void * out_mem = nullptr;\n";
//     os << "  CHECK(topsMalloc(&out_mem, " << size_string << "));\n";
//     os << "  void *device_outputs[] = {out_mem};\n";
//   }
//
//   std::vector<std::string> inputs;  // cuda input paramters
//
//   os << "\n  // adaption: convert to the cuda parameters\n";
//   size_t i = 0;
//   std::ostringstream tss;
//   for (; i < fty.in_tys.size(); ++i) {
//     tss << "  std::unique_ptr<char[]> memref_raw" << i
//         << "(new char[SizeOfRankedMemref(";
//     if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
//       tss << sty->Dims();
//     else
//       tss << "1";
//     tss << ")]);\n";
//     tss << "  auto input" << i << " = CreateUnrankedMemref(in_mem" << i
//         << ", memref_raw" << i << ".get(), ";
//     if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
//       tss << LSTR(sty->GetShape());
//     } else
//       tss << "{1}";
//     tss << ");\n";
//     inputs.push_back("input" + std::to_string(i));
//   }
//
//   if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
//     tss << "  std::vector<int64_t> out_shape = {" << RSTR(rty->GetShape())
//         << "};\n";
//   } else if (!isa<VoidType>(fty.out_ty)) {
//     tss << "  std::vector<int64_t> out_shape = {1};\n";
//   }
//   os << ReplaceRuntimeNames(tss.str(), "(int64_t)");
//
//   if (!void_return) {
//     os << "  std::unique_ptr<char[]> memref_raw" << i
//        << "(new char[SizeOfRankedMemref(out_shape.size())]);\n";
//     os << "  auto output = CreateUnrankedMemref(out_mem, memref_raw" << i
//        << ".get(), out_shape);\n";
//   }
//
//   // phase 3: Execute the executable and fetch the output
//   os << "\n  " << current_fn << "(";
//   for (auto &in : inputs) os << "&" << in << ", ";
//   os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
//   os << "  CHECK(topsStreamSynchronize(stream));\n";
//
//   size_t out_rank = 1;
//   std::string shape_string = "{1}";
//   if (out_shape.IsValid()) {
//     out_rank = out_shape.Dims();
//     shape_string = ReplaceRuntimeNames(LSTR(out_shape));
//   }
//
//   if (!out_size.empty()) {
//     os << "  auto res = choreo::make_spandata<" << out_type << ", " << out_rank
//        << ">(" << shape_string << ");\n";
//     os << "  // Copy output data from device to host\n";
//     os << "  CHECK(topsMemcpy(reinterpret_cast<void *>(res.data()), out_mem,\n";
//     os << "                  " << size_string
//        << ", topsMemcpyDeviceToHost));\n";
//   }
//
//   // phase 4: Free up the resources
//   os << "  // Free up the resources\n";
//   for (auto &p : device_mems) os << "  topsFree(" << p << ");\n";
//   os << "  topsFree(out_mem);\n\n";
//   os << "  // TODO: figure out why stream destroying crash some "
//         "applications.\n";
//   os << "  // topsStreamDestroy(stream);\n";
//   os << "  return res;\n";
//   os << "}\n";
// }
//
// std::string CUDACodeGen::ReplaceRuntimeNames(const std::string &e,
//                                                const std::string &prefix,
//                                                bool host_code) {
//   std::string expr = e;
//   for (auto &s : rts_nmap) {
//     size_t pos = 0;
//     while ((pos = expr.find(s.first, pos)) != std::string::npos) {
//       if (host_code)
//         expr.replace(pos, s.first.length(), prefix + s.second);
//       else
//         expr.replace(pos, s.first.length(), "-1");
//     }
//   }
//   return expr;
// }
//
// std::string CUDACodeGen::ReplaceDynDimName(const std::string &e) {
//   std::string expr = e;
//   for (auto &s : rts_nidx) {
//     size_t pos = 0;
//     while ((pos = expr.find(s.first, pos)) != std::string::npos) {
//       std::string dim_value = "dim_(args[" + std::to_string(rts_pidx[s.first]) +
//                               "], " + std::to_string(s.second) + ")";
//       expr.replace(pos, s.first.length(), dim_value);
//     }
//   }
//   return expr;
// }
//
// void CUDACodeGen::EmitRuntimeCheck(std::ostream &os, const Type &ty) {
//   assert(isa<FunctionType>(&ty) && "unexpected type.");
//   auto &fty = *cast<FunctionType>(&ty);
//
//   assert(fty.in_tys.size() == host_params.size() &&
//          "internal error when dealing with the host parameter size.");
//
//   // check if the input shape is as declared in choreo
//   if (fty.in_tys.size() == 0) return;
//
//   if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
//     auto name = host_params[0];
//     size_t count = 0;
//     for (auto vi : sty->GetShape().Value()) {
//       if (auto vale = dyn_cast<int>(&vi)) {
//         auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
//         os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
//         os << ", \"shape inconstant on 1st parameter (dim: " << count
//            << ").\");\n";
//       }
//       count++;
//     }
//   }
//   for (size_t i = 1; i < fty.in_tys.size(); ++i) {
//     auto name = host_params[i];
//     if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
//       size_t count = 0;
//       for (auto vi : sty->GetShape().Value()) {
//         if (auto vale = dyn_cast<int>(&vi)) {
//           auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
//           os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
//           os << ", \"shape inconstant on " << i + 1
//              << "th parameter (dim: " << count << ").\");\n";
//         }
//         count++;
//       }
//     }
//   }
// }
//
// void CUDACodeGen::EmitHostFuncDecl(std::ostream &os, const Type &ty,
//                                      const std::string &n, bool decl_only) {
//   assert(isa<FunctionType>(&ty) && "unexpected type.");
//   auto &fty = *cast<FunctionType>(&ty);
//   assert(host_params.size() == fty.in_tys.size() &&
//          "inconsistent parameter count.");
//
//   // emit the return type
//   os << HostTypeString(*fty.out_ty, true) << " " << n << "(";
//
//   if (fty.in_tys.size() > 0) {
//     if (!decl_only) {
//       if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
//         param_map.push_back(std::make_pair(
//             host_params[0], ReplaceRuntimeNames(sty->ByteSizeExpression())));
//       } else
//         param_map.push_back(std::make_pair(host_params[0], "1"));
//     }
//     os << HostTypeString(*fty.in_tys[0]) << " " << host_params[0];
//     for (size_t i = 1; i < fty.in_tys.size(); ++i) {
//       if (!decl_only) {
//         if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
//           param_map.push_back(std::make_pair(
//               host_params[i], ReplaceRuntimeNames(sty->ByteSizeExpression())));
//         } else
//           param_map.push_back(std::make_pair(host_params[i], "1"));
//       }
//       os << ", " << HostTypeString(*fty.in_tys[i]) << " " << host_params[i];
//     }
//   }
//   os << ")" << ((decl_only) ? ";\n" : " ");
// }

void CUDACodeGen::OutputScript(FunctionType *fty, const std::string &n,
                                 const std::string &out_type,
                                 const std::string &out_size,
                                 const Shape &out_shape) {
  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + n;
  std::string build_prefix_anonymous = build_path + "/__choreo";

  std::string kernel_fn = build_prefix + "_kernel_inlined.cuh";
  std::string cuda_fn = build_prefix_anonymous + "_cuda_kernel.cuh";
  host_fn = build_prefix + "_host.cu";
  target_fn = "__choreo_" + n;

  // Generate the host code
  std::string user_code = hs.str();
  hs.clear();

  EmitHostHead(hs);
  // if (!user_code.empty()) {
  //   // user code needs the choreo function decal for call
  //   EmitHostFuncDecl(hs, *fty, n, true);
  //   hs << user_code;
  // }
  // EmitHostFuncDecl(hs, *fty, n);
  // EmitHostFuncBody(hs, *fty, cuda_bfn, out_size, out_type, out_shape);
  //
  // // backpatch the cuda bin filename
  std::string cuda_src = fs.str();
  if (!alloc_in_fs.str().empty())
    cuda_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(cuda_src, std::string("$$out$$"), output_v);
  ReplaceInString(cuda_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os << "#!/usr/bin/env bash\n\n";
  os << "# This is the choreo generated bash script to compile cuda code\n";

  os << R"script(
#!/bin/bash

set -x

NVCC=nvcc
CUDA_SYS_INCLUDES="-I/usr/local/cuda/include"
CUDA_CHOREO_INCLUDES="-I./demos/cuda/sgemm_ref/"
CUDA_INCLUDES="${CUDA_SYS_INCLUDES} ${CUDA_CHOREO_INCLUDES}"
CUDA_CC="sm_35"
CUDA_ARCH="compute_35"

# TODO, device id
GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

case "${GPU_CC}" in
    3.0)
        CUDA_ARCH="compute_30"
        CUDA_CC="sm_30"
        ;;
    3.5)
        CUDA_ARCH="compute_35"
        CUDA_CC="sm_35"
        ;;
    3.7)
        CUDA_ARCH="compute_37"
        CUDA_CC="sm_37"
        ;;
    5.0)
        CUDA_ARCH="compute_50"
        CUDA_CC="sm_50"
        ;;
    5.2)
        CUDA_ARCH="compute_52"
        CUDA_CC="sm_52"
        ;;
    5.3)
        CUDA_ARCH="compute_53"
        CUDA_CC="sm_53"
        ;;
    6.0)
        CUDA_ARCH="compute_60"
        CUDA_CC="sm_60"
        ;;
    6.1)
        CUDA_ARCH="compute_61"
        CUDA_CC="sm_61"
        ;;
    6.2)
        CUDA_ARCH="compute_62"
        CUDA_CC="sm_62"
        ;;
    7.0)
        CUDA_ARCH="compute_70"
        CUDA_CC="sm_70"
        ;;
    7.2)
        CUDA_ARCH="compute_72"
        CUDA_CC="sm_72"
        ;;
    7.5)
        CUDA_ARCH="compute_75"
        CUDA_CC="sm_75"
        ;;
    8.0)
        CUDA_ARCH="compute_80"
        CUDA_CC="sm_80"
        ;;
    8.6)
        CUDA_ARCH="compute_86"
        CUDA_CC="sm_86"
        ;;
    8.9)
        CUDA_ARCH="compute_89"
        CUDA_CC="sm_89"
        ;;
    9.0)
        CUDA_ARCH="compute_90"
        CUDA_CC="sm_90"
        ;;
    *)
        echo "Unsupported GPU compute capability: ${GPU_CC}"
        exit 1
        ;;
esac

echo "CUDA_ARCH: ${CUDA_ARCH}"
echo "CUDA_CC: ${CUDA_CC}"

)script";
  os << "\n# step 0: set up the environment\n";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n";

  // no need to gen run shell, since cuda compile is simple enough to handle
  os << "cat <<'EOF' > " << build_path << "/cuda_script.sh\n";
  os << __cuda_script_as_string << "\nEOF\n";
  os << "chmod +x " << build_path << "/cuda_script.sh\n";

  os << "cat <<'EOF' > " << build_path << "/choreo_cuda.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";

  // TODO(albert): support INLINED ASM FOR CUDA
  os << "\n# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<'EOF' > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n";

  os << "\n# step 2: write the cuda source code into a temp file\n";
  os << "cuda_src=" << cuda_fn << "\n";
  os << "cat <<'EOF' > ${cuda_src}\n";
  os << cuda_src << "\nEOF\n\n";
}
