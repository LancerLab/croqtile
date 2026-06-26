// RUN: coir-opt --coir-classify-copies --coir-lower-copy --coir-emit-hip %s | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// -- Header checks --
// CHECK: #define __CHOREO_TARGET_AMDGPU__ 1
// CHECK: #include "choreo.h"
// CHECK: #include <hip/hip_runtime.h>

// -- All kernel device functions first --
// CHECK: __global__ void __add_kernel__(float* arg0, float* arg1, float* arg2)
// CHECK: arg0[threadIdx.x]
// CHECK: arg1[threadIdx.x]
// CHECK: arg2[threadIdx.x] =
coir.kernel @add(%a: !coir.tensor<256xf32>, %b: !coir.tensor<256xf32>,
                 %c: !coir.tensor<256xf32>) {
  coir.parallel (%tid) in [256] level = #coir.level<thread> {
    %va = coir.tensor.load_elem %a[%tid] : !coir.tensor<256xf32> -> f32
    %vb = coir.tensor.load_elem %b[%tid] : !coir.tensor<256xf32> -> f32
    %vc = arith.addf %va, %vb : f32
    coir.tensor.store_elem %vc, %c[%tid] : f32, !coir.tensor<256xf32>
  }
}

// CHECK: __global__ void __tiled_kernel__
// CHECK: parallel level=block
// CHECK: parallel level=thread
coir.kernel @tiled(%a: !coir.tensor<1024xf32>, %b: !coir.tensor<1024xf32>) {
  coir.parallel (%bid) in [4] level = #coir.level<block> {
    coir.parallel (%tid) in [256] level = #coir.level<thread> {
      %idx = arith.muli %bid, %tid : index
      %va = coir.tensor.load_elem %a[%idx] : !coir.tensor<1024xf32> -> f32
      coir.tensor.store_elem %va, %b[%idx] : f32, !coir.tensor<1024xf32>
    }
  }
}

// CHECK: __global__ void __barrier_test_kernel__
// CHECK: __syncthreads
coir.kernel @barrier_test(%a: !coir.tensor<64xf32>) {
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %va = coir.tensor.load_elem %a[%tid] : !coir.tensor<64xf32> -> f32
    coir.barrier #coir.level<block>
    coir.tensor.store_elem %va, %a[%tid] : f32, !coir.tensor<64xf32>
  }
}

// CHECK: __global__ void __copy_test_kernel__
// CHECK: __shared__ float
// CHECK: for (size_t __i = threadIdx.x; __i <
// CHECK: __syncthreads
coir.kernel @copy_test(%src: !coir.tensor<128xf32>) {
  %dst = coir.tensor.alloc : !coir.tensor<128xf32, shared>
  coir.data.copy %src to %dst : !coir.tensor<128xf32> -> !coir.tensor<128xf32, shared>
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %dst[%tid] : !coir.tensor<128xf32, shared> -> f32
    coir.tensor.store_elem %v, %src[%tid] : f32, !coir.tensor<128xf32>
  }
}

// CHECK: __global__ void __shared_alloc_kernel__
// CHECK: __shared__ float
coir.kernel @shared_alloc(%a: !coir.tensor<32xf32>) {
  %smem = coir.tensor.alloc : !coir.tensor<32xf32, shared>
  coir.parallel (%tid) in [32] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %a[%tid] : !coir.tensor<32xf32> -> f32
    coir.tensor.store_elem %v, %smem[%tid] : f32, !coir.tensor<32xf32, shared>
  }
}

// CHECK: __global__ void __mma_reject_kernel__
// CHECK: ERROR: MMA not supported on HIP target
coir.kernel @mma_reject(
    %a: !coir.tensor<16x16xf16, shared>,
    %c: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  coir.mma.store %acc, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// CHECK: __global__ void __scale_kernel__(float* arg0, float* out0)
coir.kernel @scale(%a: !coir.tensor<64xf32>) -> !coir.tensor<64xf32> {
  %out = coir.tensor.alloc : !coir.tensor<64xf32>
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %va = coir.tensor.load_elem %a[%tid] : !coir.tensor<64xf32> -> f32
    %two = arith.constant 2.0 : f32
    %scaled = arith.mulf %va, %two : f32
    coir.tensor.store_elem %scaled, %out[%tid] : f32, !coir.tensor<64xf32>
  }
  coir.return %out : !coir.tensor<64xf32>
}

// -- Host wrappers (emitted after all kernels) --
// CHECK: void add(
// CHECK: hipMalloc
// CHECK: hipMemcpy{{.*}}hipMemcpyHostToDevice
// CHECK: __add_kernel__<<<
// CHECK: hipDeviceSynchronize
// CHECK: hipFree

// CHECK: void tiled(
// CHECK: __tiled_kernel__<<<4, 256>>>

// CHECK: void barrier_test(
// CHECK: __barrier_test_kernel__<<<

// CHECK: choreo::spanned_data{{.*}} scale(
// CHECK: hipMalloc(&__result__device
// CHECK: hipDeviceSynchronize
// CHECK: hipMemcpy(__result.data(), __result__device
// CHECK: hipMemcpyDeviceToHost
// CHECK: return __result

} // module
