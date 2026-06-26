// RUN: coir-opt --coir-emit-hip %s | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// 2D transpose
// CHECK: __global__ void __transp_2d_kernel__
// CHECK: for (size_t __i = threadIdx.x; __i <
// CHECK: size_t __rem = __i
// CHECK: __d0
// CHECK: __d1
// CHECK: __dst_idx
// CHECK: __syncthreads
coir.kernel @transp_2d(%src: !coir.tensor<4x8xf32>) {
  %dst = coir.tensor.alloc : !coir.tensor<8x4xf32, shared>
  %tok = coir.dma.copy %src to %dst
      { kind = #coir.dma_kind<transpose>,
        transpose_perm = array<i64: 1, 0> }
      : !coir.tensor<4x8xf32> -> !coir.tensor<8x4xf32, shared>
  coir.parallel (%tid) in [32] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %dst[%tid] : !coir.tensor<8x4xf32, shared> -> f32
    coir.tensor.store_elem %v, %src[%tid] : f32, !coir.tensor<4x8xf32>
  }
}

// 3D transpose with custom permutation
// CHECK: __global__ void __transp_3d_kernel__
// CHECK: __d0
// CHECK: __d1
// CHECK: __d2
// CHECK: __dst_idx
coir.kernel @transp_3d(%src: !coir.tensor<2x3x4xf32>) {
  %dst = coir.tensor.alloc : !coir.tensor<4x2x3xf32, shared>
  %tok = coir.dma.copy %src to %dst
      { kind = #coir.dma_kind<transpose>,
        transpose_perm = array<i64: 2, 0, 1> }
      : !coir.tensor<2x3x4xf32> -> !coir.tensor<4x2x3xf32, shared>
  coir.parallel (%tid) in [24] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %dst[%tid] : !coir.tensor<4x2x3xf32, shared> -> f32
    coir.tensor.store_elem %v, %src[%tid] : f32, !coir.tensor<2x3x4xf32>
  }
}

} // module
