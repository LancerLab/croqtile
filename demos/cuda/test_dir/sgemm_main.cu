// choreo header
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
}
