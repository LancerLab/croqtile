
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

#include <cstring>
#include <cstdlib>

#define H800_PCIE_PEAK_F8_TFLOPS 3026

#ifndef SPMM_DEFAULT_M
#define SPMM_DEFAULT_M 4096
#endif

#ifndef SPMM_DEFAULT_N
#define SPMM_DEFAULT_N 8192
#endif

#ifndef SPMM_DEFAULT_K
#define SPMM_DEFAULT_K 8192
#endif

#define SPMM_WARP_M 64
#define SPMM_WARP_N 256
#define SPMM_TILE_K 128
#define SPMM_WARP_K 64
#define SPMM_STAGES 2

#define SPMM_PACKED_TILE_K 64
#define SPMM_META_TILE_COLS 4

#define SPMM_LHS_SWIZ 64
#define SPMM_RHS_SWIZ 128

#if SPMM_WARP_M != 64
#error "SPMM_WARP_M must be 64 for SM90 sparse WGMMA constraints"
#endif

#if SPMM_WARP_N < 8 || SPMM_WARP_N > 256 || (SPMM_WARP_N % 8) != 0
#error "SPMM_WARP_N must be in [8,256] and divisible by 8 for SM90 sparse WGMMA"
#endif

#if SPMM_WARP_K != 64
#error "SPMM_WARP_K must be 64 for e4m3 sparse WGMMA constraints"
#endif

#if SPMM_TILE_K != (2 * SPMM_PACKED_TILE_K)
#error "SPMM_TILE_K must equal 2 * SPMM_PACKED_TILE_K"
#endif

#if SPMM_META_TILE_COLS != (SPMM_TILE_K / 32)
#error "SPMM_META_TILE_COLS must equal SPMM_TILE_K / 32 for prepacked sparse metadata"
#endif

template <typename T>
using SparsePolicyWGMMAU32 = choreo::utils::SparseHostPolicy<T, choreo::u32>;

template <typename T>
void init_random_b(choreo::spanned_data<T, 2>& rhs, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t j = 0; j < rhs.shape()[0]; ++j) {
    for (size_t k = 0; k < rhs.shape()[1]; ++k) {
      float value = dist(gen);
      if (std::fabs(value) < 0.1f) value = (value < 0.0f ? -0.25f : 0.25f);
      rhs[j][k] = choreo::utils::from_f32<T>(value);
    }
  }
}

__global__ void __choreo_device_spmm(f8_e4m3 * lhs_packed, unsigned int * lhs_meta, f8_e4m3 * rhs, f16 * output, unsigned K, unsigned M, unsigned META_COLS, unsigned N, unsigned PACKED_K, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  extern __shared__ char __choreo_device_spmm__runtime_shared_buffer__raw[];
  auto __choreo_device_spmm__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_spmm__runtime_shared_buffer__raw));
  { // parallel-by: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:69.12
  __shared__ __align__(8) uint64_t choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    choreo::tma_mbarrier_init(&choreo_copy_atom_t_0_barrier, 1);
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{};
  choreo_copy_atom_t_0.EnablePTXMBarrier(&choreo_copy_atom_t_0_barrier);

  auto anon_1 = (unsigned char*)__choreo_device_spmm__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[2]; // shared event barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[2]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], (blockDim.x - 128) + 1);
    init(&full[1], (blockDim.x - 128) + 1);
    init(&empty[0], (blockDim.x - 128) + 1);
    init(&empty[1], (blockDim.x - 128) + 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f8_e4m3* lhs_load_s = (f8_e4m3*)(anon_1 + 65536);
  f8_e4m3* rhs_load_s = (f8_e4m3*)(anon_1 + 0);
  unsigned int* meta_s = (unsigned int*)(anon_1 + 73728);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  // inthreads: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:76.7
  if ((__choreo_vg4id_x == 0) && __CHOREO_GROUPX4_SINGLE__) {
    // with-in: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:77.9
    {
      int __iv_iv_k = 0;
      // foreach: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:77.9
      for (__iv_iv_k = 0; __iv_iv_k < ((K + 127) / 128); ++__iv_iv_k) {
        int stage = __iv_iv_k % 2;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        choreo::tma_load_2d_shared_cluster_global_mbarrier((void*)(meta_s + ((__iv_iv_k % 2 * 256))), (const void*)&__choreo_tma_0_tensor_map, (uint64_t*)&(full[stage]), (__iv_iv_k * 4), (blockIdx.x * 64));
        choreo::tma_load_2d_shared_cluster_global_mbarrier((void*)(lhs_load_s + ((__iv_iv_k % 2 * 4096))), (const void*)&__choreo_tma_1_tensor_map, (uint64_t*)&(full[stage]), (__iv_iv_k * 64), (blockIdx.x * 64));
        choreo::tma_load_2d_shared_cluster_global_mbarrier((void*)(rhs_load_s + ((__iv_iv_k % 2 * 32768))), (const void*)&__choreo_tma_2_tensor_map, (uint64_t*)&(full[stage]), (__iv_iv_k * 128), (blockIdx.y * 256));
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (1024) + (4096) + (32768));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:96.7
  if ((__choreo_vg4id_x == 1)) {
    unsigned int mc[64];
    uint32_t __frag_init_val0 = broadcast_to_u32(choreo::f32_to_f16(0.000000f));
    for (int idx = 0; idx < 64; ++idx)
      mc[idx] = __frag_init_val0;
    // with-in: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:98.9
    {
      int __iv_s = 0;
      // foreach: benchmark/performance/gemm_sp/gemm_sp_e4m3_iter016_early_empty.co:98.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    {
      const int __sp_tid = threadIdx.x % 128;
      const int __sp_row = ((__sp_tid >> 2) & 7) + ((__sp_tid & 1) << 3) + ((__sp_tid >> 5) << 4);
      const int __sp_u32_col = (__sp_tid >> 1) & 1;
      const int __sp_meta_idx = __sp_row * 4 + __sp_u32_col;
      const int K_ITERS = (K + 127) / 128;

      full[0].wait(full[0].arrive());
      warpgroup_arrive();

      #pragma unroll 1
      for (int __iv_iv_k = 0; __iv_iv_k < K_ITERS; ++__iv_iv_k) {
        const int stage = __iv_iv_k & 1;
        const int next_stage = (__iv_iv_k + 1) & 1;
        const int lhs_off = stage * 4096;
        const int rhs_off = stage * 32768;
        const int meta_off = stage * 256;

        #pragma unroll
        for (int iw = 0; iw < 2; ++iw) {
          f8_e4m3* ma_smem_ptr = (f8_e4m3*)((iw * 32 + lhs_off + lhs_load_s));
          uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B64>(ma_smem_ptr);
          f8_e4m3* mb_smem_ptr = (f8_e4m3*)((iw * 64 + rhs_off + rhs_load_s));
          uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
          uint32_t me = ((uint32_t*)(iw * 2 + meta_off + meta_s))[__sp_meta_idx];
          cute::SM90::GMMA::SPARSE::GMMA_64x256x64_F16E4M3E4M3_SS_TN<>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], me);
        }
        (void)empty[stage].arrive();
        warpgroup_commit_batch();
        if (__iv_iv_k < K_ITERS - 1) {
          warpgroup_wait<1>();
          full[next_stage].wait(full[next_stage].arrive());
          warpgroup_arrive();
        }
      }
      warpgroup_wait<0>();
    }
    f16* output_s = (f16*)(anon_1 + 0);
    auto __shape1_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto __stride1_output_s = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto __layout1_output_s = cute::make_layout(__shape1_output_s, __stride1_output_s);
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)output_s + 0), __layout1_output_s);
    store_fragment_d<CUTE_WGMMA_M64K64, 256>(__tensor1_output_s, reinterpret_cast<f16*>(mc));
    cde::fence_proxy_async_shared_cta();
    if (__CHOREO_GROUPX4_SINGLE__) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map, (blockIdx.y * 256), (blockIdx.x * 64), output_s);
      cde::cp_async_bulk_commit_group();
    }
  } // end inthreads
  } // end parallel-by
}

void spmm(const choreo::spanned_view<choreo::f8_e4m3, 2> & lhs_packed, const choreo::spanned_view<choreo::u32, 2> & lhs_meta, const choreo::spanned_view<choreo::f8_e4m3, 2> & rhs, const choreo::spanned_view<choreo::f16, 2> & output) {
  __choreo_check_cuda_environment__();
  auto &K = rhs.shape()[1];
  auto &M = lhs_packed.shape()[0];
  auto &META_COLS = lhs_meta.shape()[1];
  auto &N = rhs.shape()[0];
  auto &PACKED_K = lhs_packed.shape()[1];
  choreo::runtime_check(lhs_packed.shape()[0] == lhs_meta.shape()[0], "The shapes of the 1st parameter (dim: 0) and the 2nd parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(lhs_meta.shape()[0] == output.shape()[0], "The shapes of the 2nd parameter (dim: 0) and the 4th parameter (dim: 0) are inconsistent.");
  choreo::runtime_check(rhs.shape()[0] == output.shape()[1], "The shapes of the 3rd parameter (dim: 0) and the 4th parameter (dim: 1) are inconsistent.");

  uint64_t __choreo_tma_0_shape[] = {META_COLS, M};
  uint64_t __choreo_tma_0_strides[] = {(META_COLS * 4)};
  uint32_t __choreo_tma_0_box_shape[] = {4, 64};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
          2,
          lhs_meta.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_1_shape[] = {PACKED_K, M};
  uint64_t __choreo_tma_1_strides[] = {PACKED_K};
  uint32_t __choreo_tma_1_box_shape[] = {64, 64};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          lhs_packed.data(),
          __choreo_tma_1_shape,
          __choreo_tma_1_strides,
          __choreo_tma_1_box_shape,
          __choreo_tma_1_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_2_shape[] = {K, N};
  uint64_t __choreo_tma_2_strides[] = {K};
  uint32_t __choreo_tma_2_box_shape[] = {128, 256};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
          2,
          rhs.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_3_shape[] = {N, M};
  uint64_t __choreo_tma_3_strides[] = {(N * 2)};
  uint32_t __choreo_tma_3_box_shape[] = {256, 64};
  uint32_t __choreo_tma_3_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_3_tensor_map{};
  CUresult __choreo_tma_3_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_3_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          output.data(),
          __choreo_tma_3_shape,
          __choreo_tma_3_strides,
          __choreo_tma_3_box_shape,
          __choreo_tma_3_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_3_tensor_map_res != CUDA_SUCCESS);
  dim3 __spmm_gdims0(((M + 63) / 64), ((N + 255) / 256), 1);
  dim3 __spmm_bdims0(256, 1, 1);
  cudaFuncSetAttribute(__choreo_device_spmm, cudaFuncAttributeMaxDynamicSharedMemorySize, 75776 + (128 - 1));
  __choreo_device_spmm<<<__spmm_gdims0, __spmm_bdims0, 75776 + (128 - 1)>>>(lhs_packed.data(), lhs_meta.data(), rhs.data(), output.data(), K, M, META_COLS, N, PACKED_K, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map, __choreo_tma_3_tensor_map);
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
      continue;
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

  size_t m = SPMM_DEFAULT_M;
  size_t n = SPMM_DEFAULT_N;
  size_t k = SPMM_DEFAULT_K;

  choreo::runtime_check((k % 64) == 0,
                        "SPMM_DEFAULT_K must be divisible by 64");

  std::mt19937 gen(42);
  auto lhs_dense_h = choreo::make_spandata<choreo::f8_e4m3>(m, k);
  auto lhs_packed_h = choreo::make_spandata<choreo::f8_e4m3>(m, k / 2);
  auto lhs_meta_h = choreo::make_spandata<choreo::u32>(m, k / 32);
  auto rhs_h = choreo::make_spandata<choreo::f8_e4m3>(n, k);
  auto res_h = choreo::make_spandata<choreo::f16>(m, n);
  SparsePolicyWGMMAU32<choreo::f8_e4m3>::init_structured_sparse_A(lhs_dense_h, gen);
  init_random_b<choreo::f8_e4m3>(rhs_h, gen);
  SparsePolicyWGMMAU32<choreo::f8_e4m3>::encode(lhs_dense_h, lhs_packed_h, lhs_meta_h);
  res_h.fill(0.0f);

  __nv_fp8_e4m3 *lhs_packed_d = nullptr, *rhs_d = nullptr;
  u32 *lhs_meta_d = nullptr;
  half *res_d = nullptr;
  choreo::abend_true(cudaMalloc(&lhs_packed_d, m * (k / 2) * sizeof(__nv_fp8_e4m3)));
  choreo::abend_true(cudaMalloc(&lhs_meta_d, m * (k / 32) * sizeof(u32)));
  choreo::abend_true(cudaMalloc(&rhs_d, n * k * sizeof(__nv_fp8_e4m3)));
  choreo::abend_true(cudaMalloc(&res_d, m * n * sizeof(half)));

  choreo::abend_true(cudaMemcpy(lhs_packed_d, lhs_packed_h.data(),
                                m * (k / 2) * sizeof(__nv_fp8_e4m3),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(lhs_meta_d, lhs_meta_h.data(),
                                m * (k / 32) * sizeof(u32),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(rhs_d, rhs_h.data(),
                                n * k * sizeof(__nv_fp8_e4m3),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(res_d, res_h.data(), m * n * sizeof(half),
                                cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_packed_d_view =
      choreo::make_spanview<choreo::f8_e4m3, 2>(lhs_packed_d, {m, k / 2});
  auto lhs_meta_d_view =
      choreo::make_spanview<choreo::u32, 2>(lhs_meta_d, {m, k / 32});
  auto rhs_d_view = choreo::make_spanview<choreo::f8_e4m3, 2>(rhs_d, {n, k});
  auto res_d_view = choreo::make_spanview<choreo::f16, 2>(res_d, {m, n});

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
    auto avg_ms =
        choreo::timing([&]() { spmm(lhs_packed_d_view, lhs_meta_d_view, rhs_d_view, res_d_view);
                               cudaDeviceSynchronize(); },
                       topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0)
                       ? user_flops
                       : (2.0 * double(m) * double(n) * double(k));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    double eff = (tflops / H800_PCIE_PEAK_F8_TFLOPS) * 100.0;
    std::cout << "HW efficiency: " << eff << "%\n";
  } else {
    spmm(lhs_packed_d_view, lhs_meta_d_view, rhs_d_view, res_d_view);
  }

  choreo::abend_true(cudaMemcpy(res_h.data(), res_d, m * n * sizeof(half),
                                cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  auto lhs_dense_v = lhs_dense_h.view();
  auto rhs_v = rhs_h.view();
  auto res_v = res_h.view();

  float tolerance = 0.5f;
  for (size_t i = 0; i < 128; ++i) {
    for (size_t j = 0; j < 256; ++j) {
      float ref = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) {
        ref += choreo::to_f32(lhs_dense_v[i][kk]) *
               choreo::to_f32(rhs_v[j][kk]);
      }
      float got = __half2float(res_v[i][j]);
      float diff = std::abs(got - ref);
      if (diff > tolerance) {
        std::cout << "mismatch at (" << i << ", " << j << ") gpu=" << got
                  << " ref=" << ref << " diff=" << diff << "\n";
      }
      choreo::choreo_assert(diff < tolerance, "Sparse WGMMA e4m3 failed");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}

