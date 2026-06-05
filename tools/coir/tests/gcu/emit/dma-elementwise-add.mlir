// RUN: coir-codegen --target=gcu %s | FileCheck %s

// Verify that a DMA-staged elementwise add emits correct topscc-style code.

// CHECK: #include <krt/misc.h>
// CHECK: __device__ void add_direct
// CHECK: tops::block_idx_x()
// CHECK: tcle::dma_copy
// CHECK: tcle::dma_copy
// CHECK: for
// CHECK: +
// CHECK: tcle::dma_copy

module {
  coir.kernel @add_direct(%lhs: !coir.tensor<128xf32, global>, %rhs: !coir.tensor<128xf32, global>)
      -> !coir.tensor<128xf32, global> {
    %out = coir.tensor.alloc : !coir.tensor<128xf32, global>
    %lhs_local = coir.tensor.alloc : !coir.tensor<64xf32, local>
    %rhs_local = coir.tensor.alloc : !coir.tensor<64xf32, local>
    %out_local = coir.tensor.alloc : !coir.tensor<64xf32, local>
    coir.parallel (%p) in [2] level = #coir.level<block> {
      coir.data.copy %lhs to %lhs_local : !coir.tensor<128xf32, global> -> !coir.tensor<64xf32, local>
      coir.data.copy %rhs to %rhs_local : !coir.tensor<128xf32, global> -> !coir.tensor<64xf32, local>
      %ub = arith.constant 64 : index
      coir.foreach %i in %ub iter_args() {
        %a = coir.tensor.load_elem %lhs_local[%i] : !coir.tensor<64xf32, local> -> f32
        %b = coir.tensor.load_elem %rhs_local[%i] : !coir.tensor<64xf32, local> -> f32
        %sum = arith.addf %a, %b : f32
        coir.tensor.store_elem %sum, %out_local[%i] : f32, !coir.tensor<64xf32, local>
        coir.yield
      }
      coir.data.copy %out_local to %out : !coir.tensor<64xf32, local> -> !coir.tensor<128xf32, global>
      coir.yield
    }
    coir.return %out : !coir.tensor<128xf32, global>
  }
}
