
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
// include the choreo header;
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

#define __CHOREO_REQUIRED_GPU_DEVICE_SM__ 90

static inline void __choreo_check_cuda_environment__() {
  // ----------- ONE-TIME GUARD -----------
  static bool already_checked = false;
  if (already_checked) return;
  already_checked = true;
  // --------------------------------------

  auto decode_cuda_version =
   [](int v, int& major, int& minor, int& patch) {
    major = v / 1000;
    minor = (v % 1000) / 10;
    patch = v % 10;
  };

  // ----------- Runtime version check -----------
  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA runtime not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                "[choreo] CUDA driver not available: %s\n",
                cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int rMaj, rMin, rPat;
  int dMaj, dMin, dPat;
  decode_cuda_version(runtime_ver, rMaj, rMin, rPat);
  decode_cuda_version(driver_ver, dMaj, dMin, dPat);

  int reqMaj, reqMin, reqPat;
  decode_cuda_version(CUDART_VERSION, reqMaj, reqMin, reqPat);

  if (runtime_ver < CUDART_VERSION) {
    std::fprintf(stderr,
       "[choreo] CUDA runtime too old:\n"
       "  found runtime %d.%d.%d (encoded=%d)\n"
       "  required      %d.%d.%d (encoded=%d)\n",
       rMaj, rMin, rPat, runtime_ver,
       reqMaj, reqMin, reqPat, CUDART_VERSION);
    std::exit(EXIT_FAILURE);
  }

  // Optional: check driver vs runtime mismatch
  if (driver_ver < runtime_ver) {
    std::fprintf(stderr,
       "[choreo] Warning: CUDA driver (%d.%d.%d, encoded=%d) is older than "
       "the CUDA runtime (%d.%d.%d, encoded=%d). This may cause issues.\n",
       dMaj, dMin, dPat, driver_ver,
       rMaj, rMin, rPat, runtime_ver);
  }

  // ----------- Device capability check -----------
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                "[choreo] No CUDA-capable devices found.\n");
    std::exit(EXIT_FAILURE);
  }

  // ----------- Device capability check (selected device) -----------
  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "[choreo] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm = prop.major * 10 + prop.minor;
  if (sm < __CHOREO_REQUIRED_GPU_DEVICE_SM__) {
    std::fprintf(stderr,
        "[choreo] Compute capability too low on device %d (%s):\n"
        "  found SM %d.%d (sm_%d)\n"
        "  required SM >= %d (sm_%d)\n",
        device_id, prop.name,
        prop.major, prop.minor, sm,
        __CHOREO_REQUIRED_GPU_DEVICE_SM__, __CHOREO_REQUIRED_GPU_DEVICE_SM__);
    std::exit(EXIT_FAILURE);
  }

#if 0
  // ----------- Optional success log -----------
  std::fprintf(stderr,
    "[choreo] CUDA environment OK\n"
    "  runtime %d.%d.%d (encoded=%d)\n"
    "  driver  %d.%d.%d (encoded=%d)\n"
    "  device  %d: %s, SM %d.%d (sm_%d)\n",
    rMaj, rMin, rPat, runtime_ver,
    dMaj, dMin, dPat, driver_ver,
    device_id, prop.name, prop.major, prop.minor, sm);
#endif
}

// MoE GEMM FP8→BF16 kernel (ai-tune winner, 2026-03-27, iter012)
// COMPILE:
//   EXTRA_TARGET_CFLAGS="--maxrregcount=72 -Xptxas --allow-expensive-optimizations=true" \
//     ./choreo -gs -t cute -arch=sm_90a <this-file>.co -o /tmp/moe_aitune.cute.result
// RUN:
//   bash /tmp/moe_aitune.cute.result --execute
// TFLOPS: 6.22 (M=384, N=512, K=2048, 256 experts, Poisson routing)
// Regs: 72/thread, 0 spill, 17KB SMEM, 7 blocks/SM, 43.75% occupancy on H100
#include <cstddef>
#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F16_TFLOPS 1513
#define H800_PCIE_PEAK_F8_TFLOPS 3026

// N=64 reduces accumulator from 64 to 32 fp32 regs (vs N=128 baseline).
// BLOCK_SIZE_N matches Qwen 3.5 weight_block_size={128,128} quantization config.
#define MATMUL_WARP_M 64
#define MATMUL_WARP_N 64
#define MATMUL_WARP_K 32

#define MATMUL_TILE_M 64
#define MATMUL_TILE_N 64
#define MATMUL_TILE_K 128

#define MATMUL_SWIZ 128

#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 128
#define SCALE_N_RATIO 2

#if MATMUL_WARP_K != 32
#error "MATMUL_WARP_K must be 32 for f8 WGMMA constraints"
#endif

#if MATMUL_TILE_M != MATMUL_WARP_M
#error "MATMUL_TILE_M must equal MATMUL_WARP_M"
#endif

#define MATMUL_DEFAULT_TOKENS 192
#define TOPK 2
#define MATMUL_DEFAULT_M (MATMUL_DEFAULT_TOKENS * TOPK)
#define MATMUL_DEFAULT_N 512
#define MATMUL_DEFAULT_K 2048


#define EXPERTS 256

__global__ __launch_bounds__(128, 7)
void __choreo_device_moe_gemm_kernel_bf16(f8_e4m3 * lhs, float * scale_a, f8_e4m3 * rhs, float * scale_b, int * expert_offsets, bf16 * output, unsigned DIV_BLK_K, unsigned EXPERTS1, unsigned EXPERT_DIV_BLK_N, unsigned EXPERT_N, unsigned K, unsigned M, unsigned N, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map) {
  auto __choreo_device_moe_gemm_kernel_bf16__ring__ = nullptr;
  { // parallel-by: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:56.18
  auto wg_barrier = cooperative_groups::tiled_partition<128>(cooperative_groups::this_thread_block());
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  __shared__ alignas(1024) unsigned char anon_3[16384];
  auto __choreo_vg4id_x = threadIdx.x / 128;
  auto __choreo_vtid_x = threadIdx.x % 128;
  f8_e4m3* sA = (f8_e4m3*)(anon_3 + 8192);
  f8_e4m3* sB = (f8_e4m3*)(anon_3 + 0);
  int seg_start = *((int*)expert_offsets + blockIdx.x);
  int seg_end = *((int*)expert_offsets + (blockIdx.x + 1));
  int seg_length = (seg_end - seg_start);
  if (seg_end - seg_start <= 0) return;

  // with-in: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:67.5
  {
    int __iv_iv_m = 0;
    // foreach: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:67.5
    for (__iv_iv_m = 0; __iv_iv_m < ((seg_length + 63) / 64); ++__iv_iv_m) {
      int TILE_M = (64 < seg_length - __iv_iv_m * 64 ? 64 : seg_length - __iv_iv_m * 64);
      float mc[32];
      float __frag_init_val0 = 0.000000f;
      for (int idx = 0; idx < 32; ++idx)
        mc[idx] = __frag_init_val0;
      // with-in: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:70.7
      {
        int __iv_iv_k = 0;
        // foreach: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:70.7
        for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
          float mc_scale_frag[32];
          memset(mc_scale_frag, 0, sizeof(mc_scale_frag));
          future __choreo_anon_fut__0("", 71, 9, sA);
          auto __shape1_lhs = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
          auto __stride1_lhs = cute::make_stride(K, cute::Int<1>{});
          auto __layout1_lhs = cute::make_layout(__shape1_lhs, __stride1_lhs);
          auto __tensor1_lhs = cute::make_tensor(cute::make_gmem_ptr<f8_e4m3>((f8_e4m3*)lhs + (__iv_iv_k * 128 + K * (seg_start + __iv_iv_m * 64))), __layout1_lhs);
          auto __shape2_sA = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
          auto __layout2_sA = cute::tile_to_shape(cute::SM90::GMMA::Layout_K_SW128_Atom<f8_e4m3>{}, __shape2_sA);
          auto __tensor2_sA = cute::make_tensor(cute::make_smem_ptr<f8_e4m3>((f8_e4m3*)sA + 0), __layout2_sA);
          choreo::copy_if_g2s<true, f8_e4m3, 16, 8, 1, 16>(__tensor1_lhs, __tensor2_sA, [&](const auto& __coord) { return cute::elem_less(__coord, cute::make_shape(TILE_M, cute::Int<128>{})); });
          __syncthreads();
          auto anon_2 = blockIdx.x * ((N + 63) / 64) + blockIdx.y;
          future __choreo_anon_fut__1("", 74, 9, sB);
          __choreo_anon_fut__1.is_tma = true;
          __choreo_anon_fut__1.set_atom(&choreo_copy_atom_t_0);
          if (__CHOREO_BLOCK_SINGLE__) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(sB, &__choreo_tma_0_tensor_map, (__iv_iv_k * 128), ((blockIdx.y + (N + 63) / 64 * blockIdx.x) * 64), ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier());
            ((TMAAtom*)__choreo_anon_fut__1.get_atom())->token() = cuda::device::barrier_arrive_tx(((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier(), 1, 8192);
          } else {
            ((TMAAtom*)__choreo_anon_fut__1.get_atom())->token() = ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier().arrive();
          }
          ((TMAAtom*)__choreo_anon_fut__1.get_atom())->barrier().wait(std::move(((TMAAtom*)__choreo_anon_fut__1.get_atom())->token()));
          __choreo_anon_fut__1.set_nowait();

          // with-in: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:77.9
          {
            int __iv_iv_warp = 0;
            // foreach: benchmark/performance/moe_gemm/moe_gemm_fp8_bf16_aitune_2026-03-27_iter012.co:77.9
            for (__iv_iv_warp = 0; __iv_iv_warp < 4; ++__iv_iv_warp) {
              f8_e4m3* ma_smem_ptr = (f8_e4m3*)((__iv_iv_warp * 32 + sA));
              uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(ma_smem_ptr);
              f8_e4m3* mb_smem_ptr = (f8_e4m3*)((__iv_iv_warp * 32 + sB));
              uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
              warpgroup_arrive();
              // Note: warpgroup_arrive() should be called once before first WGMMA
              // and warpgroup_wait() should be called once after all WGMMAs
              cute::SM90::GMMA::MMA_64x64x32_F32E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc_scale_frag[0], mc_scale_frag[1], mc_scale_frag[2], mc_scale_frag[3], mc_scale_frag[4], mc_scale_frag[5], mc_scale_frag[6], mc_scale_frag[7], mc_scale_frag[8], mc_scale_frag[9], mc_scale_frag[10], mc_scale_frag[11], mc_scale_frag[12], mc_scale_frag[13], mc_scale_frag[14], mc_scale_frag[15], mc_scale_frag[16], mc_scale_frag[17], mc_scale_frag[18], mc_scale_frag[19], mc_scale_frag[20], mc_scale_frag[21], mc_scale_frag[22], mc_scale_frag[23], mc_scale_frag[24], mc_scale_frag[25], mc_scale_frag[26], mc_scale_frag[27], mc_scale_frag[28], mc_scale_frag[29], mc_scale_frag[30], mc_scale_frag[31]);
            } // iv_warp
            __iv_iv_warp = 0;
          }
          auto sc_a = (__iv_iv_k + DIV_BLK_K * (seg_start + __iv_iv_m * 64) + scale_a);
          float sc_b = *((float*)scale_b + (blockIdx.y / 2 + (N + 127) / 128 * blockIdx.x)*DIV_BLK_K + __iv_iv_k);
          float* mc_scale_a_ptr = (float*)(sc_a);
          float mc_scale_b_val = static_cast<float>(sc_b);
          scale_accumulator<float, float, 64>(reinterpret_cast<float*>(mc), reinterpret_cast<float*>(mc_scale_frag), mc_scale_a_ptr, DIV_BLK_K, mc_scale_b_val);
        } // iv_k
        __iv_iv_k = 0;
      }
      // Finalize WGMMA operations
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      auto __shape3_output = cute::make_shape(TILE_M, cute::Int<64>{});
      auto __stride3_output = cute::make_stride(N, cute::Int<1>{});
      auto __layout3_output = cute::make_layout(__shape3_output, __stride3_output);
      auto __tensor3_output = cute::make_tensor(cute::make_gmem_ptr<bf16>((bf16*)output + (blockIdx.y * 64 + N * (seg_start + __iv_iv_m * 64))), __layout3_output);
      store_fragment_d_mask_row<CUTE_WGMMA_M64K32, 64>(__tensor3_output, reinterpret_cast<float*>(mc), TILE_M);
    } // iv_m
    __iv_iv_m = 0;
  }
  } // end parallel-by
}

void moe_gemm_kernel_bf16(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs, const choreo::spanned_view<choreo::f32, 2> & scale_a, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f32, 2> & scale_b, const choreo::spanned_view<choreo::s32, 1> & expert_offsets, const choreo::spanned_view<choreo::bf16, 2> & output, cudaStream_t s) {
  __choreo_check_cuda_environment__();
  auto &DIV_BLK_K = scale_a.shape()[1];
  auto &EXPERTS1 = expert_offsets.shape()[0];
  auto &EXPERT_DIV_BLK_N = scale_b.shape()[0];
  auto &EXPERT_N = rhs.shape()[0];
  auto &K = lhs.shape()[1];
  auto &M = lhs.shape()[0];
  auto &N = output.shape()[1];
  choreo::runtime_check(scale_a.shape()[1] == scale_b.shape()[1], "The shapes of the 2nd parameter (dim: 1) and the 4th parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[1] == rhs.shape()[1], "The shapes of the 1st parameter (dim: 1) and the 3rd parameter (dim: 1) are inconsistent.");
  choreo::runtime_check(lhs.shape()[0] == scale_a.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(scale_a.shape()[0] == output.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 6th parameter (dim: 0) are inconsistent.");

  uint64_t __choreo_tma_0_shape[] = {K, EXPERT_N};
  uint64_t __choreo_tma_0_strides[] = {K};
  uint32_t __choreo_tma_0_box_shape[] = {128, 64};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          rhs.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  dim3 __moe_gemm_kernel_bf16_gdims0(256, ((N + 63) / 64), 1);
  dim3 __moe_gemm_kernel_bf16_bdims0(128, 1, 1);
  __choreo_device_moe_gemm_kernel_bf16<<<__moe_gemm_kernel_bf16_gdims0, __moe_gemm_kernel_bf16_bdims0, 0, s>>>(lhs.data(), scale_a.data(), rhs.data(), scale_b.data(), expert_offsets.data(), output.data(), DIV_BLK_K, EXPERTS1, EXPERT_DIV_BLK_N, EXPERT_N, K, M, N, __choreo_tma_0_tensor_map);
}





extern "C" void moe_fp8_grouped_gemm_bf16(
    const uint8_t* a,
    const uint8_t* b,
    const float* a_scales,
    const float* b_scales,
    const int32_t* expert_offsets,
    int num_experts,
    int m,
    int n,
    int k,
    int block_size_n,
    int block_size_k,
    int sm_version,
    __nv_bfloat16* out,
    cudaStream_t stream) {
  // printf("moe_fp8_grouped_gemm_bf16 called with m=%d, n=%d, k=%d, num_experts=%d\n", m, n, k, num_experts);
  if (sm_version == 90) {
    int n_blocks = (n + block_size_n - 1) / block_size_n;
    int k_blocks = (k + block_size_k - 1) / block_size_k;

    auto a_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(a, {size_t(m), size_t(k)});
    auto b_ptr = choreo::make_spanview<choreo::f8_e4m3, 2>(b, {size_t(num_experts) * size_t(n), size_t(k)});
    auto a_scales_ptr = choreo::make_spanview<choreo::f32, 2>(a_scales, {size_t(m), size_t(k_blocks)});
    auto b_scales_ptr = choreo::make_spanview<choreo::f32, 2>(b_scales, {size_t(num_experts) * size_t(n_blocks), size_t(k_blocks)});
    auto out_ptr = choreo::make_spanview<choreo::bf16, 2>(out, {size_t(m), size_t(n)});
    auto expert_offsets_ptr = choreo::make_spanview<choreo::s32, 1>(expert_offsets, {size_t(num_experts + 1)});
    moe_gemm_kernel_bf16(a_ptr, a_scales_ptr, b_ptr, b_scales_ptr, expert_offsets_ptr, out_ptr, stream);
  } else {
     printf("moe_fp8_grouped_gemm_bf16 unsupported sm_version %d\n", sm_version);
  }
}

#if 1
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

  size_t top_k = TOPK;
  size_t num_tokens = MATMUL_DEFAULT_TOKENS;
  size_t M = num_tokens * top_k;
  size_t N = MATMUL_DEFAULT_N;
  size_t K = MATMUL_DEFAULT_K;
  // expert configuration
  size_t num_experts = EXPERTS;
  size_t block_size_n = BLOCK_SIZE_N;
  size_t block_size_k = BLOCK_SIZE_K;
  size_t n_blocks = (N + block_size_n - 1) / block_size_n;
  size_t k_blocks = (K + block_size_k - 1) / block_size_k;

  auto lhs_h = choreo::make_spandata<choreo::f8_e4m3>(M, K);
  auto rhs_h = choreo::make_spandata<choreo::f8_e4m3>(num_experts *  N, K);
  auto scale_a_h = choreo::make_spandata<choreo::f32>(M, k_blocks);
  auto scale_b_h = choreo::make_spandata<choreo::f32>(num_experts * n_blocks, k_blocks);
  auto expert_offsets_h = choreo::make_spandata<choreo::s32>(num_experts + 1);
  auto res_h = choreo::make_spandata<choreo::bf16>(M, N);
  cudaStream_t stream = nullptr;

  lhs_h.fill(1.0f);
  // lhs_h.fill_random(0, 2);
  rhs_h.fill_random(0, 2);
  scale_a_h.fill_random(1, 3);
  scale_b_h.fill_random(1, 3);
  res_h.fill(0.0f);

  // Realistic MoE routing: random Poisson-like token distribution (seed=42)
  // Simulates top-k learned router: ~22% of experts get 0 tokens, rest follow Poisson(1.5)
  {
    for (int e = 0; e <= (int)num_experts; ++e) expert_offsets_h[e] = 0;
    uint32_t rng = 0x12345678u;
    for (int row = 0; row < (int)M; ++row) {
      rng = rng * 1664525u + 1013904223u;
      int eid = (int)((rng >> 8) % (uint32_t)num_experts);
      expert_offsets_h[eid + 1]++;
    }
    for (int e = 1; e <= (int)num_experts; ++e)
      expert_offsets_h[e] += expert_offsets_h[e - 1];
  }

  uint8_t *a_d = nullptr, *b_d = nullptr;
  float *a_scale_d = nullptr, *b_scale_d = nullptr;
  int32_t *expert_off_d = nullptr;
  float *out_d = nullptr;
  choreo::abend_true(cudaMalloc(&a_d, M * K * sizeof(uint8_t)));
  choreo::abend_true(cudaMalloc(&b_d, (size_t)num_experts * N * K * sizeof(uint8_t)));
  choreo::abend_true(cudaMalloc(&a_scale_d, M * k_blocks * sizeof(float)));
  choreo::abend_true(cudaMalloc(&b_scale_d, (size_t)num_experts * n_blocks * k_blocks * sizeof(float)));
  choreo::abend_true(cudaMalloc(&expert_off_d, (num_experts + 1) * sizeof(int32_t)));
  choreo::abend_true(cudaMalloc(&out_d, M * N * sizeof(float)));

  choreo::abend_true(cudaMemcpy(a_d, lhs_h.data(), M * K * sizeof(uint8_t), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_d, rhs_h.data(), (size_t)num_experts * N * K * sizeof(uint8_t), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(a_scale_d, scale_a_h.data(), M * k_blocks * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(b_scale_d, scale_b_h.data(), (size_t)num_experts * n_blocks * k_blocks * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(expert_off_d, expert_offsets_h.data(), (num_experts + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(out_d, res_h.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_d = choreo::make_spanview<choreo::f8_e4m3, 2>(a_d, {M, K});
  auto rhs_d = choreo::make_spanview<choreo::f8_e4m3, 2>(b_d, {size_t(num_experts) * N, K});
  auto scale_a_d = choreo::make_spanview<choreo::f32, 2>(a_scale_d, {M, k_blocks});
  auto scale_b_d = choreo::make_spanview<choreo::f32, 2>(b_scale_d, {size_t(num_experts) * n_blocks, k_blocks});
  auto res_d = choreo::make_spanview<choreo::bf16, 2>(out_d, {M, N});
  auto expert_off_d_v = choreo::make_spanview<choreo::s32, 1>(expert_off_d, {size_t(num_experts + 1)});

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
    auto avg_ms = choreo::timing([&]() { moe_gemm_kernel_bf16(lhs_d, scale_a_d, rhs_d, scale_b_d, expert_off_d_v, res_d, stream); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(M) * double(N) * double(K));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    double eff = (tflops / H800_PCIE_PEAK_F8_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    moe_gemm_kernel_bf16(lhs_d, scale_a_d, rhs_d, scale_b_d, expert_off_d_v, res_d, stream);
  }

  choreo::abend_true(cudaMemcpy(res_h.data(), out_d, M * N * sizeof(choreo::bf16), cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  auto lhs_view = lhs_h.view();
  auto rhs_view = rhs_h.view();
  auto res_view = res_h.view();
  auto scale_a_view = scale_a_h.view();
  auto scale_b_view = scale_b_h.view();
  float rel_tolerance = 0.05f;
  float abs_tolerance = 1e-3f;
  size_t mismatch_count = 0;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  int worst_m = -1;
  int worst_n = -1;
  float worst_ref = 0.0f;
  float worst_got = 0.0f;
  auto rel_error = [](float ref, float got) {
    float abs_ref = std::abs(ref);
    float denom = abs_ref > 1e-6f ? abs_ref : 1.0f;
    return std::abs(ref - got) / denom;
  };

  for (int eid = 0; eid < num_experts; ++eid) {
    int seg_start = expert_offsets_h[eid];
    int seg_end = expert_offsets_h[eid + 1];
    for (int m = seg_start; m < seg_end; ++m) {
      for (int n = 0; n < (int)N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < (int)K; ++k) {
          float a_val = to_f32(lhs_view[m][k]);
          float b_val = to_f32(rhs_view[eid * N + n][k]);
          acc += a_val * b_val * scale_a_view[m][k / block_size_k] * scale_b_view[eid * n_blocks + n / block_size_n][k / block_size_k];
        }
        float got = to_f32(res_view[m][n]);
        float abs_error = std::abs(acc - got);
        float delta = rel_error(acc, got);
        bool pass = delta <= rel_tolerance;
        // bool pass = abs_error <= abs_tolerance || delta <= rel_tolerance;

        if (abs_error > max_abs_error) max_abs_error = abs_error;
        if (delta > max_rel_error) {
          max_rel_error = delta;
          worst_m = m;
          worst_n = n;
          worst_ref = acc;
          worst_got = got;
        }

        if (!pass) {
          ++mismatch_count;
          std::cout << "[M=" << m << ", N=" << n << "] " << acc << " <-> " << got
                    << ", abs error: " << abs_error
                    << ", rel error: " << delta * 100 << "%\n";
        }
      }
    }
  }

  std::cout << "Verifier summary: mismatches=" << mismatch_count
            << ", max abs error=" << max_abs_error
            << ", max rel error=" << max_rel_error * 100 << "%";
  if (worst_m >= 0) {
    std::cout << " at [M=" << worst_m << ", N=" << worst_n << "]"
              << " ref=" << worst_ref << ", got=" << worst_got;
  }
  std::cout << "\n";

  if (mismatch_count != 0) {
    std::cerr << "Verification failed\n" << std::endl;
    return 1;
  }
  std::cout << "Test Passed\n" << std::endl;
}
#endif

