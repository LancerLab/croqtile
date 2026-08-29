
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef __CUDACC__
#pragma nv_diag_suppress 20054
#endif
// include the choreo header;
#define __CHOREO_ENABLE_CUDA_RUNTIME_ENV_CHECK__
#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

#include <cstring>
#include <cstdlib>

// H800 PCIe (Hopper PCIe class) FP16 Tensor Core peak throughput.
#define H800_PCIE_PEAK_F16_TFLOPS 1513
#define H800_PCIE_PEAK_F8_TFLOPS 3026

// Tunable defaults for auto-tuning/profiling workflows.
#define MATMUL_WARP_M 64
// NOTE: M64N128K16 can reduce accumulated error, but slower than N256 in tiling.
// this may suggest we must have multi warpgroups to distribute over M dims
#define MATMUL_WARP_N 64

#define MATMUL_TILE_M 64
#define MATMUL_TILE_K 64
#define MATMUL_WARP_K 16
#define MATMUL_SWIZ 128
#define MATMUL_STAGES 2

#if MATMUL_SWIZ != (2 * MATMUL_TILE_K)
#error "MATMUL_SWIZ must equal 2 * MATMUL_TILE_K for f16 kernel"
#endif

#if MATMUL_SWIZ != 32 && MATMUL_SWIZ != 64 && MATMUL_SWIZ != 128
#error "MATMUL_SWIZ must be one of 32, 64, 128"
#endif

#if MATMUL_WARP_M != 64
#error "MATMUL_WARP_M must be 64 for f16 WGMMA constraints"
#endif

#if MATMUL_WARP_K != 16
#error "MATMUL_WARP_K must be 16 for f16 WGMMA constraints"
#endif

#if MATMUL_TILE_M != MATMUL_WARP_M
#error "MATMUL_TILE_M must equal MATMUL_WARP_M"
#endif

#if MATMUL_WARP_N < 8 || MATMUL_WARP_N > 256 || (MATMUL_WARP_N % 8) != 0
#error "MATMUL_WARP_N must be in [8,256] and divisible by 8 for SM90 WGMMA f16"
#endif

#define MATMUL_DEFAULT_M 4096
#define MATMUL_DEFAULT_N 4096
#define MATMUL_DEFAULT_K 4096

__global__ __launch_bounds__(256, 2) void __choreo_device_matmul(f16 * lhs, f16 * rhs, f16 * output, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  [[maybe_unused]] auto __choreo_device_matmul__ring__ = nullptr;
  { // parallel-by: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:52.12
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_1[32768];
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2]; // shared event barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2]; // shared event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], 129);
    init(&full[1], 129);
    init(&empty[0], 129);
    init(&empty[1], 129);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f16* lhs_load_s = (f16*)(anon_1 + 0);
  f16* rhs_load_s = (f16*)(anon_1 + 16384);
  f16* output_s = (f16*)(anon_1 + 0);
  [[maybe_unused]] auto __choreo_vg4id_x = threadIdx.x / 128;
  [[maybe_unused]] auto __choreo_vtid_x = threadIdx.x % 128;
  unsigned int mc[16];
  uint32_t __frag_init_val0 = broadcast_to_u32(choreo::f32_to_f16(0.000000f));
  for (int idx = 0; idx < 16; ++idx)
    mc[idx] = __frag_init_val0;
  // inthreads: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:61.7
  if ((__choreo_vg4id_x == 0 && __choreo_vtid_x == 0)) {
    // with-in: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:62.9
    {
      int __iv_iv_k = 0;
      // foreach: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:62.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 2 * 4096))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 64), (blockIdx.x * 64), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 2 * 4096))), &__choreo_tma_1_tensor_map, (K / ((K + 63) / 64) * __iv_iv_k), (N / ((N + 63) / 64) * blockIdx.y), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (8192) + (8192));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:71.7
  if ((__choreo_vg4id_x == 1)) {
    // with-in: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:72.9
    {
      int __iv_s = 0;
      // foreach: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:72.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:75.9
    {
      int __iv_iv_k = 0;
      // foreach: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:75.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 63) / 64); ++__iv_iv_k) {
        auto stage = __iv_iv_k % 2;
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        // with-in: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:78.11
        {
          int __iv_iv_warp = 0;
          // foreach: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:78.11
          warpgroup_fence_operand(mc);
          warpgroup_arrive();
          for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
            f16* ma_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((__iv_iv_warp * 16 + __iv_iv_k % 2 * 4096 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            cute::SM90::GMMA::MMA_64x64x16_F16F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15]);
          } // iv_warp
          __iv_iv_warp = 0;
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(mc);
        // trigger event(barrier)  (empty elemof stage) 
        (void)empty[stage].arrive();
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:88.7
  if ((__choreo_vg4id_x == 1)) {
    auto __shape1_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<64>{});
    auto __stride1_output_s = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
    auto __layout1_output_s = cute::make_layout(__shape1_output_s, __stride1_output_s);
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)output_s + 0), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K16, 64>(__tensor1_output_s, reinterpret_cast<f16*>(mc));
    asm volatile("bar.sync 15, 128;\n" ::: "memory");
    future __choreo_anon_fut__0("", 90, 9, output);
    __choreo_anon_fut__0.is_tma = true;
    __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
    cde::fence_proxy_async_shared_cta();
    asm volatile("bar.sync 15, 128;\n" ::: "memory");
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (blockIdx.y * 64), (blockIdx.x * 64), output_s);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
    }
  }
  __syncthreads(); // end inthreads
  } // end parallel-by
}

void matmul(const choreo::spanned_view<choreo::f16, 2> & lhs, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<choreo::f16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = rhs.shape()[0];
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 2nd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == output.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 3rd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 2nd parameter (dim: 0) and the 3rd parameter (dim: 1) are inconsistent.");

  choreo::runtime_check(((static_cast<long long>(M) + 63LL) / 64LL > 0LL), "The 1st bound item of parallelby is invalid: should be greater than 0, tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:52.13");
  choreo::runtime_check(((static_cast<long long>(N) + 63LL) / 64LL > 0LL), "The 2nd bound item of parallelby is invalid: should be greater than 0, tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:52.22");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:62.27");
  choreo::runtime_check((static_cast<long long>(K) / ((static_cast<long long>(K) + 63LL) / 64LL) * (static_cast<long long>(N) / ((static_cast<long long>(N) + 63LL) / 64LL)) <= 4096LL), "DMA to-buffer is too small (((::matmul::K / ((::matmul::K + 63) / 64)) * (::matmul::N / ((::matmul::N + 63) / 64))) > 4096), tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:66.11");
  choreo::runtime_check(((static_cast<long long>(K) + 63LL) / 64LL != 0LL), "zero is detected for the 1st dim of the mdspan inside the with-in statement, tools/sala_real_eval/xcomp_cu_compare/bench_1p1c_small.co:75.27");
  uint64_t __choreo_tma_0_shape[] = {K, M};
  uint64_t __choreo_tma_0_strides[] = {(K * 2)};
  uint32_t __choreo_tma_0_box_shape[] = {(uint32_t)(64), (uint32_t)(64)};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          lhs.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_1_shape[] = {K, N};
  uint64_t __choreo_tma_1_strides[] = {(K * 2)};
  uint32_t __choreo_tma_1_box_shape[] = {(uint32_t)((K / ((K + 63) / 64))), (uint32_t)((N / ((N + 63) / 64)))};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          rhs.data(),
          __choreo_tma_1_shape,
          __choreo_tma_1_strides,
          __choreo_tma_1_box_shape,
          __choreo_tma_1_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_2_shape[] = {N, M};
  uint64_t __choreo_tma_2_strides[] = {(N * 2)};
  uint32_t __choreo_tma_2_box_shape[] = {(uint32_t)(64), (uint32_t)(64)};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          output.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
  dim3 __matmul_gdims0(((M + 63) / 64), ((N + 63) / 64), 1);
  dim3 __matmul_bdims0(256, 1, 1);
  __choreo_device_matmul<<<__matmul_gdims0, __matmul_bdims0>>>(lhs.data(), rhs.data(), output.data(), K, M, N, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
  choreo::abend_true(cudaDeviceSynchronize());
}




int main(int argc, char** argv) {
  bool enable_timing = true;
  bool skip_verify = false;
  double user_flops = -1.0;
  auto is_disable_timing_arg = [](const char* s) {
    const char* t = "--disable-timing";
    int i = 0;
    while (t[i] != '\0' && s[i] == t[i]) ++i;
    return t[i] == '\0' && s[i] == '\0';
  };
  auto is_skip_verify_arg = [](const char* s) {
    const char* t = "--skip-verify";
    int i = 0;
    while (t[i] != '\0' && s[i] == t[i]) ++i;
    return t[i] == '\0' && s[i] == '\0';
  };
  for (int i = 1; i < argc; ++i) {
    if (is_disable_timing_arg(argv[i])) {
      enable_timing = false;
      continue;
    }
    if (is_skip_verify_arg(argv[i])) {
      skip_verify = true;
    }
    if (std::strncmp(argv[i], "--flops=", 8) == 0) {
      user_flops = std::atof(argv[i] + 8);
      continue;
    }
  }

  const char* timing_env = std::getenv("CHOREO_DISABLE_TIMING");
  if (timing_env && timing_env[0] == '1' && timing_env[1] == '\0') {
    enable_timing = false;
  }

  const char* skip_verify_env = std::getenv("CHOREO_SKIP_VERIFY");
  if (skip_verify_env && skip_verify_env[0] == '1' && skip_verify_env[1] == '\0') {
    skip_verify = true;
  }
  
  size_t M = MATMUL_DEFAULT_M;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--m=", 4) == 0) M = std::atol(argv[i] + 4);
    if (std::strncmp(argv[i], "--n=", 4) == 0) N = std::atol(argv[i] + 4);
    if (std::strncmp(argv[i], "--k=", 4) == 0) K = std::atol(argv[i] + 4);
  }
  std::cout << "Problem: M=" << M << " N=" << N << " K=" << K << "\n";

  auto lhs_h = choreo::make_spandata<choreo::f16>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f16>(N, K);
  auto res_h = choreo::make_spandata<choreo::f16>(M, N);
  lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  res_h.fill(0.0f);

  half *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&b_d, N * K * sizeof(half)));
  choreo::abend_true(cudaMalloc(&c_d, M * N * sizeof(half)));
  
  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), N * K * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(c_d, res_h.data(), M * N * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_d = choreo::make_spanview<choreo::f16, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::f16, 2>(b_d, {N, K});
  auto res_d = choreo::make_spanview<choreo::f16, 2>(c_d, {M, N});
  if (enable_timing) {
    int warmup = 10;
    int repeat = 500;
    const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
    const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
    if (warmup_env) {
      int value = std::atoi(warmup_env);
      if (value >= 0) warmup = value;
    }
    if (repeat_env) {
      int value = std::atoi(repeat_env);
      if (value > 0) repeat = value;
    }
    choreo::TimerOption topt;
    topt.warmup = warmup;
    topt.repeat = repeat;
    auto avg_ms = choreo::timing([&]() { matmul(lhs_d, rhs_d, res_d); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";

    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";

    double eff = (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    matmul(lhs_d, rhs_d, res_d);
  }

  choreo::abend_true(cudaMemcpy(res_h.data(), c_d, M * N * sizeof(half), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto res_view = res_h.view();

  float tolerance = 0.05f;
  auto rel_error = [](float ref, float got) {
    float abs_ref = std::abs(ref);
    float denom = abs_ref > 1e-6f ? abs_ref : 1.0f;
    return std::abs(ref - got) / denom;
  };
  // verification
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 256; ++j) {
      float ref = 0.0f;
      for (size_t k = 0; k < lhs_view.shape()[1]; ++k)
        ref += __half2float(lhs_view[i][k] * rhs_view[j][k]);
      float got = __half2float(res_view[i][j]);
      auto delta = rel_error(ref, got);
      if (delta >= tolerance) {
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got << ", delta: " << delta * 100 << "%\n";
      }
      choreo::choreo_assert((delta < tolerance), "values are not equal.");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}

