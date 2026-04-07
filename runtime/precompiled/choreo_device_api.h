// choreo_device_api.h - Thin compatibility header for fast-compile mode
//
// Provides the same API surface as the full choreo runtime + CuTe headers
// but routes calls to precompiled wrapper functions, avoiding the expensive
// CuTe/CUTLASS header processing entirely.
//
// Key idea: this header defines matching type stubs for what the generated
// code uses, so the codegen doesn't need to change.

#ifndef __CHOREO_DEVICE_API_H__
#define __CHOREO_DEVICE_API_H__

// Block cute/tensor.hpp and cutlass headers via include guards.
// This lets choreo.h compile in __CHOREO_TARGET_CUTE__ mode for host types
// without pulling in the heavy CuTe header tree.
#define CUTE_TENSOR_HPP
#define CUTLASS_CUTLASS_H
#define CUTE_ARCH_MMA_SM90_DESC_HPP
#define CUTE_ARCH_MMA_SM90_GMMA_HPP

// Include choreo.h in CuTe mode (for host types: spanned_view, timing, etc.)
// The cute/tensor.hpp include is blocked by the guard above.
#define __CHOREO_TARGET_CUTE__
#ifndef __USE_CUDA_TYPE__
#define __USE_CUDA_TYPE__
#endif
#include "choreo.h"
// Include the choreo_cute.h for device helpers (TMAAtom, future, etc.)
// and MMA policy types -- the blocked cute includes prevent the heavy parts.
#include "choreo_cute.h"

// ---------------------------------------------------------------------------
// Precompiled device wrapper declarations
// ---------------------------------------------------------------------------

// WGMMA smem descriptor wrappers
extern __device__ uint64_t __choreo_wgmma_smem_desc_k_ns(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_k_b32(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_k_b64(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_k_b128(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_mn_ns(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_mn_b32(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_mn_b64(void*);
extern __device__ uint64_t __choreo_wgmma_smem_desc_mn_b128(void*);

// WGMMA fence/sync primitives
extern __device__ void __choreo_warpgroup_arrive();
extern __device__ void __choreo_warpgroup_commit_batch();
extern __device__ void __choreo_warpgroup_wait_0();
extern __device__ void __choreo_warpgroup_wait_1();
extern __device__ void __choreo_warpgroup_wait_2();

// WGMMA fma wrappers -- F32 accumulator, F16 inputs, SS K-major
extern __device__ void __choreo_wgmma_64x64x16_f32f16f16_ss_kk(
    uint64_t, uint64_t, float*);
extern __device__ void __choreo_wgmma_64x128x16_f32f16f16_ss_kk(
    uint64_t, uint64_t, float*);
extern __device__ void __choreo_wgmma_64x256x16_f32f16f16_ss_kk(
    uint64_t, uint64_t, float*);

// WGMMA fma wrappers -- F16 accumulator, F16 inputs, SS K-major
extern __device__ void __choreo_wgmma_64x64x16_f16f16f16_ss_kk(
    uint64_t, uint64_t, uint32_t*);
extern __device__ void __choreo_wgmma_64x128x16_f16f16f16_ss_kk(
    uint64_t, uint64_t, uint32_t*);
extern __device__ void __choreo_wgmma_64x256x16_f16f16f16_ss_kk(
    uint64_t, uint64_t, uint32_t*);

// Fragment store wrappers
extern __device__ void __choreo_store_fragment_d_m64k16_f16_f32(
    f16*, int, int, float* const);
extern __device__ void __choreo_store_fragment_d_m64k16_dyn(
    f16*, int, float* const);

// TMA copy wrappers
extern __device__ void __choreo_tma_load_2d(
    void*, const CUtensorMap*,
    cuda::barrier<cuda::thread_scope_block>&, int, int);
extern __device__ void __choreo_tma_store_2d(
    const CUtensorMap*, int, int, void*);
extern __device__ void __choreo_tma_store_commit();
extern __device__ void __choreo_tma_store_wait_0();
extern __device__ void __choreo_fence_proxy_async();

// Host-side
extern void __choreo_check_cuda_environment_impl(int);

// The GMMA fma types are needed by the generated code. Since we blocked
// the heavy CuTe GMMA header, provide minimal stubs that match the API.
namespace cute {
namespace SM90 {
namespace GMMA {
enum class Major { K = 0, MN = 1 };
} // namespace GMMA
} // namespace SM90
} // namespace cute

#endif // __CHOREO_DEVICE_API_H__
