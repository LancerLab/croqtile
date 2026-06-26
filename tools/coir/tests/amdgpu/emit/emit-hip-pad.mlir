// RUN: coir-opt --coir-emit-hip %s | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// Pad copy: fill destination with pad value, then copy source into padded region
// CHECK: __global__ void __pad_test_kernel__
// CHECK: for (size_t __i = threadIdx.x; __i <
// CHECK: = (float)0
// CHECK: __syncthreads
// CHECK: for (size_t __i = threadIdx.x; __i <
// CHECK: size_t __rem = __i
// CHECK: __dst_idx
// CHECK: __syncthreads
coir.kernel @pad_test(%src: !coir.tensor<4x4xf32>) {
  %dst = coir.tensor.alloc : !coir.tensor<6x6xf32, shared>
  %tok = coir.dma.copy %src to %dst
      { kind = #coir.dma_kind<pad>,
        pad_low = array<i64: 1, 1>,
        pad_high = array<i64: 1, 1> }
      : !coir.tensor<4x4xf32> -> !coir.tensor<6x6xf32, shared>
  coir.parallel (%tid) in [36] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %dst[%tid] : !coir.tensor<6x6xf32, shared> -> f32
    coir.tensor.store_elem %v, %src[%tid] : f32, !coir.tensor<4x4xf32>
  }
}

// Pad with nonzero fill value
// CHECK: __global__ void __pad_val_kernel__
// CHECK: = (int32_t)-1
coir.kernel @pad_val(%src: !coir.tensor<8xi32>) {
  %dst = coir.tensor.alloc : !coir.tensor<12xi32, shared>
  %tok = coir.dma.copy %src to %dst
      { kind = #coir.dma_kind<pad>,
        pad_low = array<i64: 2>,
        pad_high = array<i64: 2>,
        pad_value = -1 : i64 }
      : !coir.tensor<8xi32> -> !coir.tensor<12xi32, shared>
  coir.parallel (%tid) in [12] level = #coir.level<thread> {
    %v = coir.tensor.load_elem %dst[%tid] : !coir.tensor<12xi32, shared> -> i32
    coir.tensor.store_elem %v, %src[%tid] : i32, !coir.tensor<8xi32>
  }
}

} // module
