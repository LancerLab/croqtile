
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

#define H800_PCIE_PEAK_F16_TFLOPS 1513

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
#define SPMM_TILE_M 128
#define SPMM_WARP_N 256
#define SPMM_TILE_K 64
#define SPMM_WARP_K 32
#define SPMM_STAGES 3

#define SPMM_PACKED_TILE_K 32
#define SPMM_META_TILE_COLS 2
#define SPMM_META_SUB_COLS 1

#if SPMM_WARP_M != 64
#error "SPMM_WARP_M must be 64 for SM90 sparse WGMMA constraints"
#endif

#if SPMM_TILE_M % SPMM_WARP_M != 0
#error "SPMM_TILE_M must be divisible by SPMM_WARP_M"
#endif

#if SPMM_WARP_K != 32
#error "SPMM_WARP_K must be 32 for f16 sparse WGMMA constraints"
#endif

#if SPMM_TILE_K != (2 * SPMM_PACKED_TILE_K)
#error "SPMM_TILE_K must equal 2 * SPMM_PACKED_TILE_K"
#endif

#if SPMM_META_TILE_COLS != (SPMM_TILE_K / 32)
#error "SPMM_META_TILE_COLS must equal SPMM_TILE_K / 32"
#endif

template <typename T>
using SparsePolicyWGMMA = choreo::utils::SparsePolicyWGMMA<T>;

__global__ void __choreo_device_spmm(f16 * lhs_packed, unsigned int * lhs_meta, f16 * rhs, f16 * output, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map) {
  extern __shared__ char __choreo_device_spmm__runtime_shared_buffer__raw[];
  auto __choreo_device_spmm__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_spmm__runtime_shared_buffer__raw));
  { // parallel-by: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:56.12
  __shared__ cuda::barrier<cuda::thread_scope_block> choreo_copy_atom_t_0_barrier;
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&choreo_copy_atom_t_0_barrier, blockDim.x);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  TMAAtom choreo_copy_atom_t_0{&choreo_copy_atom_t_0_barrier};

  auto anon_4 = (unsigned char*)__choreo_device_spmm__runtime_shared_buffer__;
  __shared__ cuda::barrier<cuda::thread_scope_block> full[3]; // shared event barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> empty[3]; // shared event barrier
  // initialize the event barrier
  if (__CHOREO_BLOCK_SINGLE__) {
    init(&full[0], (blockDim.x - 128) + 1);
    init(&full[1], (blockDim.x - 128) + 1);
    init(&full[2], (blockDim.x - 128) + 1);
    init(&empty[0], (blockDim.x - 128) + 1);
    init(&empty[1], (blockDim.x - 128) + 1);
    init(&empty[2], (blockDim.x - 128) + 1);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  f16* lhs_load_s = (f16*)(anon_4 + 163840);
  f16* rhs_load_s = (f16*)(anon_4 + 0);
  f16* output_s = (f16*)(anon_4 + 98304);
  auto __choreo_vg4id_x = threadIdx.x / 128;
  // inthreads: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:64.7
  if ((__choreo_vg4id_x == 0) && __CHOREO_GROUPX4_SINGLE__) {
    // with-in: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:65.9
    {
      int __iv_iv_k = 0;
      // foreach: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:65.9
      for (__iv_iv_k = 0; __iv_iv_k < 128; ++__iv_iv_k) {
        int stage = __iv_iv_k % 3;
        // wait event(barrier)  (empty elemof stage) 
        empty[stage].wait(empty[stage].arrive());
        cde::cp_async_bulk_tensor_2d_global_to_shared((lhs_load_s + ((__iv_iv_k % 3 * 4096))), &__choreo_tma_0_tensor_map, (__iv_iv_k * 32), (blockIdx.x * 128), full[stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared((rhs_load_s + ((__iv_iv_k % 3 * 16384))), &__choreo_tma_1_tensor_map, (__iv_iv_k * 64), (blockIdx.y * 256), full[stage]);
        // trigger event(barrier)  (full elemof stage) 
        (void)cuda::device::barrier_arrive_tx(full[stage], 1, (8192) + (32768));
      } // iv_k
      __iv_iv_k = 0;
    }
  } // end inthreads
  // inthreads: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:80.7
  if ((__choreo_vg4id_x > 0)) {
    auto anon_3 = __choreo_vg4id_x - 1;
    unsigned int mc[64];
    uint32_t __frag_init_val0 = broadcast_to_u32(choreo::f32_to_f16(0.000000f));
    for (int idx = 0; idx < 64; ++idx)
      mc[idx] = __frag_init_val0;
    // with-in: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:82.9
    {
      int __iv_s = 0;
      // foreach: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:82.9
      #pragma unroll
      for (__iv_s = 0; __iv_s < 3; ++__iv_s) {
        // trigger event(barrier)  (empty elemof s) 
        (void)empty[__iv_s].arrive();
      } // s
      __iv_s = 0;
    }
    // with-in: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:85.9
    {
      int __iv_iv_k = 0;
      // foreach: benchmark/performance/gemm_sp/gemm_sp_f16_aitune_2026-03-24/gemm_sp_f16_iter120_1p2c_3stg.co:85.9
      {
      int __sp_tid = threadIdx.x % 128;
      int __sp_lane = __sp_tid & 31;
      bool __sp_active = ((__sp_lane & 3) < 2);
      int __sp_local_row = ((__sp_tid >> 5) * 16) + (((__sp_tid >> 2) & 7) << 1) + (__sp_tid & 1);
      uint32_t __meta_base_fixed = blockIdx.x * 32768 + (__choreo_vg4id_x - 1) * 16384;
      #pragma unroll 2
      for (__iv_iv_k = 0; __iv_iv_k < 128; ++__iv_iv_k) {
        auto stage = __iv_iv_k % 3;
        // Hoist metadata load before full barrier wait
        uint32_t __meta_w0 = 0, __meta_w1 = 0;
        if (__sp_active) {
          uint2 __meta_vec = __ldg(reinterpret_cast<const uint2*>(&lhs_meta[__meta_base_fixed + __iv_iv_k * 2 + __sp_local_row * 256]));
          __meta_w0 = __meta_vec.x;
          __meta_w1 = __meta_vec.y;
        }
        // wait event(barrier)  (full elemof stage) 
        full[stage].wait(full[stage].arrive());
        {
          // iv_warp=0
          {
            f16* ma_smem_ptr = (f16*)((0 * 16 + (__iv_iv_k % 3 * 4096 + (__choreo_vg4id_x - 1) * 2048) + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B64>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((0 * 32 + __iv_iv_k % 3 * 16384 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            cute::SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], __meta_w0);
            warpgroup_commit_batch();
          }
          // iv_warp=1
          {
            f16* ma_smem_ptr = (f16*)((1 * 16 + (__iv_iv_k % 3 * 4096 + (__choreo_vg4id_x - 1) * 2048) + lhs_load_s));
            uint64_t desc_ma = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B64>(ma_smem_ptr);
            f16* mb_smem_ptr = (f16*)((1 * 32 + __iv_iv_k % 3 * 16384 + rhs_load_s));
            uint64_t desc_mb = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(mb_smem_ptr);
            warpgroup_arrive();
            cute::SM90::GMMA::SPARSE::GMMA_64x256x32_F16F16F16_SS<static_cast<cute::SM90::GMMA::Major>(0), static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_ma, desc_mb, mc[0], mc[1], mc[2], mc[3], mc[4], mc[5], mc[6], mc[7], mc[8], mc[9], mc[10], mc[11], mc[12], mc[13], mc[14], mc[15], mc[16], mc[17], mc[18], mc[19], mc[20], mc[21], mc[22], mc[23], mc[24], mc[25], mc[26], mc[27], mc[28], mc[29], mc[30], mc[31], mc[32], mc[33], mc[34], mc[35], mc[36], mc[37], mc[38], mc[39], mc[40], mc[41], mc[42], mc[43], mc[44], mc[45], mc[46], mc[47], mc[48], mc[49], mc[50], mc[51], mc[52], mc[53], mc[54], mc[55], mc[56], mc[57], mc[58], mc[59], mc[60], mc[61], mc[62], mc[63], __meta_w1);
            warpgroup_commit_batch();
          }
        }
        warpgroup_wait<2>();
        // trigger event(barrier)  (empty elemof stage) 
        (void)empty[stage].arrive();
      } // iv_k
      __iv_iv_k = 0;
      warpgroup_wait<0>();
      }
    }
    warpgroup_wait<0>();
    auto __shape1_output_s = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto __stride1_output_s = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto __layout1_output_s = cute::make_layout(__shape1_output_s, __stride1_output_s);
    auto __tensor1_output_s = cute::make_tensor(cute::make_smem_ptr<f16>((f16*)output_s + ((__choreo_vg4id_x - 1) * 16384)), __layout1_output_s);
    store_fragment_d_stmatrix<CUTE_WGMMA_M64K32, 256>(__tensor1_output_s, reinterpret_cast<f16*>(mc));
  } // end inthreads
  future __choreo_anon_fut__0("", 114, 5);
  __choreo_anon_fut__0.is_tma = true;
  __choreo_anon_fut__0.set_atom(&choreo_copy_atom_t_0);
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  if (__CHOREO_BLOCK_SINGLE__) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_2_tensor_map, (blockIdx.y * 256), (blockIdx.x * 128), output_s);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  } // end parallel-by
}

void spmm(const choreo::spanned_view<choreo::f16, 2> & lhs_packed, const choreo::spanned_view<choreo::u32, 2> & lhs_meta, const choreo::spanned_view<choreo::f16, 2> & rhs, const choreo::spanned_view<choreo::f16, 2> & output) {
  __choreo_check_cuda_environment__();
  choreo::runtime_check(lhs_packed.shape()[0] == 4096, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 0): expect: 4096, but got " + std::to_string(lhs_packed.shape()[0]) + ".");
  choreo::runtime_check(lhs_packed.shape()[1] == 4096, "shape inconsistent on the 1st parameter ('lhs_packed', dim: 1): expect: 4096, but got " + std::to_string(lhs_packed.shape()[1]) + ".");
  choreo::runtime_check(lhs_meta.shape()[0] == 4096, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 0): expect: 4096, but got " + std::to_string(lhs_meta.shape()[0]) + ".");
  choreo::runtime_check(lhs_meta.shape()[1] == 256, "shape inconsistent on the 2nd parameter ('lhs_meta', dim: 1): expect: 256, but got " + std::to_string(lhs_meta.shape()[1]) + ".");
  choreo::runtime_check(rhs.shape()[0] == 8192, "shape inconsistent on the 3rd parameter ('rhs', dim: 0): expect: 8192, but got " + std::to_string(rhs.shape()[0]) + ".");
  choreo::runtime_check(rhs.shape()[1] == 8192, "shape inconsistent on the 3rd parameter ('rhs', dim: 1): expect: 8192, but got " + std::to_string(rhs.shape()[1]) + ".");
  choreo::runtime_check(output.shape()[0] == 4096, "shape inconsistent on the 4th parameter ('output', dim: 0): expect: 4096, but got " + std::to_string(output.shape()[0]) + ".");
  choreo::runtime_check(output.shape()[1] == 8192, "shape inconsistent on the 4th parameter ('output', dim: 1): expect: 8192, but got " + std::to_string(output.shape()[1]) + ".");

  uint64_t __choreo_tma_0_shape[] = {4096, 4096};
  uint64_t __choreo_tma_0_strides[] = {8192};
  uint32_t __choreo_tma_0_box_shape[] = {32, 128};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
          2,
          lhs_packed.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_1_shape[] = {8192, 8192};
  uint64_t __choreo_tma_1_strides[] = {16384};
  uint32_t __choreo_tma_1_box_shape[] = {64, 256};
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
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_2_shape[] = {8192, 4096};
  uint64_t __choreo_tma_2_strides[] = {16384};
  uint32_t __choreo_tma_2_box_shape[] = {256, 128};
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
  dim3 __spmm_gdims0(32, 32, 1);
  dim3 __spmm_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_spmm, cudaFuncAttributeMaxDynamicSharedMemorySize, 188416 + (128 - 1));
  __choreo_device_spmm<<<__spmm_gdims0, __spmm_bdims0, 188416 + (128 - 1)>>>(lhs_packed.data(), lhs_meta.data(), rhs.data(), output.data(), __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map);
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

  size_t m = SPMM_DEFAULT_M, n = SPMM_DEFAULT_N, k = SPMM_DEFAULT_K;
  choreo::runtime_check((k % 64) == 0,
                        "SPMM_DEFAULT_K must be divisible by 64");

  std::mt19937 gen(42);
  auto lhs_dense_h = choreo::make_spandata<choreo::f16>(m, k);
  auto lhs_packed_h = choreo::make_spandata<choreo::f16>(m, k / 2);
  auto lhs_meta_u8_h = choreo::make_spandata<choreo::u8>(m, k / 8);
  auto lhs_meta_h = choreo::make_spandata<choreo::u32>(m, k / 32);
  auto rhs_h = choreo::make_spandata<choreo::f16>(n, k);
  auto res_h = choreo::make_spandata<choreo::f16>(m, n);
  SparsePolicyWGMMA<choreo::f16>::init_structured_sparse_A(lhs_dense_h, gen);
  rhs_h.fill_random(-1.0f, 1.0f);
  SparsePolicyWGMMA<choreo::f16>::encode(lhs_dense_h, lhs_packed_h, lhs_meta_u8_h);
  SparsePolicyWGMMA<choreo::f16>::prepack(lhs_meta_u8_h, lhs_meta_h);
  res_h.fill(0.0f);

  half *lhs_packed_d = nullptr, *rhs_d = nullptr;
  u32 *lhs_meta_d = nullptr;
  half *res_d = nullptr;
  choreo::abend_true(cudaMalloc(&lhs_packed_d, m * (k / 2) * sizeof(half)));
  choreo::abend_true(cudaMalloc(&lhs_meta_d, m * (k / 32) * sizeof(u32)));
  choreo::abend_true(cudaMalloc(&rhs_d, n * k * sizeof(half)));
  choreo::abend_true(cudaMalloc(&res_d, m * n * sizeof(half)));
  choreo::abend_true(cudaMemcpy(lhs_packed_d, lhs_packed_h.data(), m * (k / 2) * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(lhs_meta_d, lhs_meta_h.data(), m * (k / 32) * sizeof(u32), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(rhs_d, rhs_h.data(), n * k * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaMemcpy(res_d, res_h.data(), m * n * sizeof(half), cudaMemcpyHostToDevice));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_packed_d_view = choreo::make_spanview<choreo::f16, 2>(lhs_packed_d, {m, k / 2});
  auto lhs_meta_d_view = choreo::make_spanview<choreo::u32, 2>(lhs_meta_d, {m, k / 32});
  auto rhs_d_view = choreo::make_spanview<choreo::f16, 2>(rhs_d, {n, k});
  auto res_d_view = choreo::make_spanview<choreo::f16, 2>(res_d, {m, n});

  if (enable_timing) {
    choreo::TimerOption topt;
    topt.warmup = 10;
    topt.repeat = 500;
    const char* warmup_env = std::getenv("CHOREO_TIMING_WARMUP");
    const char* repeat_env = std::getenv("CHOREO_TIMING_REPEAT");
    if (warmup_env) {
      int value = std::atoi(warmup_env);
      if (value >= 0) topt.warmup = value;
    }
    if (repeat_env) {
      int value = std::atoi(repeat_env);
      if (value > 0) topt.repeat = value;
    }
    auto avg_ms = choreo::timing([&]() { spmm(lhs_packed_d_view, lhs_meta_d_view, rhs_d_view, res_d_view); cudaDeviceSynchronize(); }, topt);
    std::cout << "Timing avg ms: " << avg_ms << "\n";
    double flops = (user_flops > 0.0) ? user_flops : (2.0 * double(m) * double(n) * double(k));
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;
    std::cout << "TFLOPS: " << tflops << "\n";
    std::cout << "HW efficiency: " << (tflops / H800_PCIE_PEAK_F16_TFLOPS) * 100.0 << "%\n";
  } else {
    spmm(lhs_packed_d_view, lhs_meta_d_view, rhs_d_view, res_d_view);
    choreo::abend_true(cudaDeviceSynchronize());
  }

  if (skip_verify) {
    std::cout << "Test Passed (verify skipped)\n" << std::endl;
    return 0;
  }

  spmm(lhs_packed_d_view, lhs_meta_d_view, rhs_d_view, res_d_view);
  choreo::abend_true(cudaDeviceSynchronize());
  choreo::abend_true(cudaMemcpy(res_h.data(), res_d, m * n * sizeof(half),
                                cudaMemcpyDeviceToHost));
  choreo::abend_true(cudaDeviceSynchronize());

  auto lhs_dense_v = lhs_dense_h.view();
  auto rhs_v = rhs_h.view();
  auto res_v = res_h.view();
  float tolerance = 1.0;
  size_t verify_m = (m < 128) ? m : 128;
  size_t verify_n = (n < 256) ? n : 256;
  for (size_t i = 0; i < verify_m; ++i) {
    for (size_t j = 0; j < verify_n; ++j) {
      float ref = 0.0f;
      for (size_t kk = 0; kk < lhs_dense_v.shape()[1]; ++kk)
        ref += choreo::to_f32(lhs_dense_v[i][kk]) * choreo::to_f32(rhs_v[j][kk]);
      float got = choreo::to_f32(res_v[i][j]);
      float diff = std::abs(got - ref);
      if (diff >= tolerance) {
        std::cout << "[" << i << ", " << j << "] " << ref << " <-> " << got
                  << ", abs delta: " << diff << "\n";
      }
      choreo::choreo_assert((diff < tolerance), "values are not equal.");
    }
  }

  std::cout << "Test Passed\n" << std::endl;
}


