// choreo_precompiled.cu - Precompiled CuTe runtime for fast kernel compilation
//
// This file includes all heavy CuTe/CUTLASS headers and the choreo runtime,
// then provides non-template device wrapper functions for commonly used
// operations. Generated kernel .cu files include only a thin header
// (choreo_device_api.h) with forward declarations, avoiding the expensive
// header processing entirely.
//
// Build once per architecture:
//   nvcc -dc -dlto -gencode arch=compute_90a,code=sm_90a \
//        -std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
//        -D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
//        --expt-relaxed-constexpr \
//        -I${CUTE_HOME}/include \
//        choreo_precompiled.cu -o choreo_precompiled.o

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;

// ---------------------------------------------------------------------------
// WGMMA smem descriptor wrappers
// ---------------------------------------------------------------------------
__device__ uint64_t
__choreo_wgmma_smem_desc_k_ns(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::NS>(
      (f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_k_b32(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B32>(
      (f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_k_b64(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B64>(
      (f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_k_b128(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(
      (f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_mn_ns(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR, WGMMA_Swizzle::NS>(
      (f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_mn_b32(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR,
                              WGMMA_Swizzle::B32>((f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_mn_b64(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR,
                              WGMMA_Swizzle::B64>((f16*)ptr);
}

__device__ uint64_t
__choreo_wgmma_smem_desc_mn_b128(void* ptr) {
  return wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR,
                              WGMMA_Swizzle::B128>((f16*)ptr);
}

// ---------------------------------------------------------------------------
// WGMMA fence/sync primitives
// ---------------------------------------------------------------------------
__device__ void __choreo_warpgroup_arrive() { warpgroup_arrive(); }

__device__ void __choreo_warpgroup_commit_batch() { warpgroup_commit_batch(); }

__device__ void __choreo_warpgroup_wait_0() { warpgroup_wait<0>(); }
__device__ void __choreo_warpgroup_wait_1() { warpgroup_wait<1>(); }
__device__ void __choreo_warpgroup_wait_2() { warpgroup_wait<2>(); }

// ---------------------------------------------------------------------------
// WGMMA fma wrappers (SS variants -- both operands from shared memory)
// The d[] array is passed as float* with the documented register count.
// ---------------------------------------------------------------------------
// MMA_64x64x16 F32 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x64x16_f32f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                          float* d) {
  cute::SM90::GMMA::MMA_64x64x16_F32F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
      d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
      d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]);
}

// MMA_64x128x16 F32 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x128x16_f32f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                           float* d) {
  cute::SM90::GMMA::MMA_64x128x16_F32F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
      d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
      d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
      d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
      d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
      d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
      d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
}

// MMA_64x256x16 F32 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x256x16_f32f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                           float* d) {
  cute::SM90::GMMA::MMA_64x256x16_F32F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
      d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
      d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
      d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
      d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
      d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
      d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63],
      d[64], d[65], d[66], d[67], d[68], d[69], d[70], d[71],
      d[72], d[73], d[74], d[75], d[76], d[77], d[78], d[79],
      d[80], d[81], d[82], d[83], d[84], d[85], d[86], d[87],
      d[88], d[89], d[90], d[91], d[92], d[93], d[94], d[95],
      d[96], d[97], d[98], d[99], d[100], d[101], d[102], d[103],
      d[104], d[105], d[106], d[107], d[108], d[109], d[110], d[111],
      d[112], d[113], d[114], d[115], d[116], d[117], d[118], d[119],
      d[120], d[121], d[122], d[123], d[124], d[125], d[126], d[127]);
}

// MMA_64x64x16 F16 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x64x16_f16f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                          uint32_t* d) {
  cute::SM90::GMMA::MMA_64x64x16_F16F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

// MMA_64x128x16 F16 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x128x16_f16f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                           uint32_t* d) {
  cute::SM90::GMMA::MMA_64x128x16_F16F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
      d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
      d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31]);
}

// MMA_64x256x16 F16 += F16 * F16 (SS, K-major both)
__device__ void
__choreo_wgmma_64x256x16_f16f16f16_ss_kk(uint64_t desc_a, uint64_t desc_b,
                                           uint32_t* d) {
  cute::SM90::GMMA::MMA_64x256x16_F16F16F16_SS<
      static_cast<cute::SM90::GMMA::Major>(0),
      static_cast<cute::SM90::GMMA::Major>(0)>::fma(desc_a, desc_b,
      d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
      d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15],
      d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23],
      d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31],
      d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
      d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47],
      d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55],
      d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
}

// ---------------------------------------------------------------------------
// Fragment store wrappers
// ---------------------------------------------------------------------------
__device__ void
__choreo_store_fragment_d_m64k16_f16_f32(f16* smem, int M, int N,
                                          float* const d) {
  auto shape = cute::make_shape(cute::Int<64>{}, cute::Int<64>{});
  auto stride = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
  auto layout = cute::make_layout(shape, stride);
  auto tensor = cute::make_tensor(cute::make_smem_ptr<f16>(smem), layout);
  store_fragment_d<CUTE_WGMMA_M64K16, 64>(tensor, d);
}

// Flexible N-dimension fragment store (runtime N)
__device__ void
__choreo_store_fragment_d_m64k16_dyn(f16* smem, int N, float* const d) {
  if (N == 64) {
    auto shape = cute::make_shape(cute::Int<64>{}, cute::Int<64>{});
    auto stride = cute::make_stride(cute::Int<64>{}, cute::Int<1>{});
    auto layout = cute::make_layout(shape, stride);
    auto tensor = cute::make_tensor(cute::make_smem_ptr<f16>(smem), layout);
    store_fragment_d<CUTE_WGMMA_M64K16, 64>(tensor, d);
  } else if (N == 128) {
    auto shape = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto stride = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto layout = cute::make_layout(shape, stride);
    auto tensor = cute::make_tensor(cute::make_smem_ptr<f16>(smem), layout);
    store_fragment_d<CUTE_WGMMA_M64K16, 128>(tensor, d);
  } else if (N == 256) {
    auto shape = cute::make_shape(cute::Int<64>{}, cute::Int<256>{});
    auto stride = cute::make_stride(cute::Int<256>{}, cute::Int<1>{});
    auto layout = cute::make_layout(shape, stride);
    auto tensor = cute::make_tensor(cute::make_smem_ptr<f16>(smem), layout);
    store_fragment_d<CUTE_WGMMA_M64K16, 256>(tensor, d);
  }
}

// ---------------------------------------------------------------------------
// TMA copy wrappers
// ---------------------------------------------------------------------------
__device__ void
__choreo_tma_load_2d(void* dst, const CUtensorMap* tma_map,
                     cuda::barrier<cuda::thread_scope_block>& bar,
                     int coord0, int coord1) {
  cde::cp_async_bulk_tensor_2d_global_to_shared(dst, tma_map, coord0, coord1,
                                                 bar);
}

__device__ void
__choreo_tma_store_2d(const CUtensorMap* tma_map, int coord0, int coord1,
                      void* src) {
  cde::cp_async_bulk_tensor_2d_shared_to_global(tma_map, coord0, coord1, src);
}

__device__ void __choreo_tma_store_commit() {
  cde::cp_async_bulk_commit_group();
}

__device__ void __choreo_tma_store_wait_0() {
  cde::cp_async_bulk_wait_group_read<0>();
}

__device__ void __choreo_fence_proxy_async() {
  cde::fence_proxy_async_shared_cta();
}

// ---------------------------------------------------------------------------
// Cooperative groups wrapper
// ---------------------------------------------------------------------------
__device__ auto __choreo_tiled_partition_128() {
  return cooperative_groups::tiled_partition<128>(
      cooperative_groups::this_thread_block());
}

// ---------------------------------------------------------------------------
// Host-side utilities (always compiled)
// ---------------------------------------------------------------------------
void __choreo_check_cuda_environment_impl(int required_sm) {
  static bool already_checked = false;
  if (already_checked)
    return;
  already_checked = true;

  auto decode_cuda_version = [](int v, int& major, int& minor, int& patch) {
    major = v / 1000;
    minor = (v % 1000) / 10;
    patch = v % 10;
  };

  int runtime_ver = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] CUDA runtime not available: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int driver_ver = 0;
  err = cudaDriverGetVersion(&driver_ver);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] CUDA driver not available: %s\n",
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
                 rMaj, rMin, rPat, runtime_ver, reqMaj, reqMin, reqPat,
                 CUDART_VERSION);
    std::exit(EXIT_FAILURE);
  }

  if (driver_ver < runtime_ver) {
    std::fprintf(stderr,
                 "[choreo] Warning: CUDA driver (%d.%d.%d, encoded=%d) is "
                 "older than the CUDA runtime (%d.%d.%d, encoded=%d).\n",
                 dMaj, dMin, dPat, driver_ver, rMaj, rMin, rPat, runtime_ver);
  }

  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "[choreo] No CUDA-capable devices found.\n");
    std::exit(EXIT_FAILURE);
  }

  int device_id = 0;
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[choreo] cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }

  int sm = prop.major * 10 + prop.minor;
  if (sm < required_sm) {
    std::fprintf(
        stderr,
        "[choreo] Compute capability too low on device %d (%s):\n"
        "  found SM %d.%d (sm_%d)\n"
        "  required SM >= %d (sm_%d)\n",
        device_id, prop.name, prop.major, prop.minor, sm, required_sm,
        required_sm);
    std::exit(EXIT_FAILURE);
  }
}
