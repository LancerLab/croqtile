// RUN: coir-opt --coir-classify-copies --coir-lower-mma --coir-lower-copy --coir-emit-cuda %s | FileCheck %s

module attributes { "coir.has_tma" = true, "coir.mma_target" = "wgmma" } {

// CHECK: #define __CHOREO_TARGET_CUTE__
// CHECK: #include "choreo.h"
// CHECK: using namespace nvcuda;

// End-to-end test: simple MMA kernel
// CHECK: __global__ void __mma_kernel_kernel__
// CHECK: wmma::fill_fragment
// CHECK: wmma::load_matrix_sync
// CHECK: wmma::load_matrix_sync
// CHECK: wmma::mma_sync
// CHECK: wmma::store_matrix_sync
coir.kernel @mma_kernel(
    %a_tile: !coir.tensor<16x16xf16, shared>,
    %b_tile: !coir.tensor<16x16xf16, shared>,
    %c_tile: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  coir.mma.store %res, %c_tile : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// End-to-end test: copy + barrier + MMA
// CHECK: __global__ void __copy_and_mma_kernel__
// CHECK: choreo::naive_copy
// CHECK: __syncthreads
// CHECK: wmma::fill_fragment
// CHECK: wmma::load_matrix_sync
// CHECK: wmma::mma_sync
// CHECK: wmma::store_matrix_sync
coir.kernel @copy_and_mma(
    %ga: !coir.tensor<128x64xf16>,
    %gb: !coir.tensor<64x128xf16>,
    %c: !coir.tensor<16x16xf32, shared>) {
  %sa = coir.tensor.alloc : !coir.tensor<128x64xf16, shared>
  %sb = coir.tensor.alloc : !coir.tensor<64x128xf16, shared>

  %tok_a = coir.dma.copy %ga to %sa : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  %tok_b = coir.dma.copy %gb to %sb : !coir.tensor<64x128xf16> -> !coir.tensor<64x128xf16, shared>
  coir.wait %tok_a : !coir.async
  coir.wait %tok_b : !coir.async
  coir.barrier #coir.level<block>

  %at = coir.tensor.tile %sa[] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<16x16xf16, shared>
  %bt = coir.tensor.tile %sb[] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x16xf16, shared>

  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %af = coir.mma.load %at : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %bf = coir.mma.load %bt : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %res = coir.mma.exec %acc, %af, %bf {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  coir.mma.store %res, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// End-to-end test: parallel + foreach accumulate
// CHECK: __global__ void __matmul_kernel_kernel__
// CHECK: parallel level=block
// CHECK: for (int
// CHECK: wmma::load_matrix_sync
// CHECK: wmma::mma_sync
// CHECK: wmma::store_matrix_sync
coir.kernel @matmul_kernel(
    %a: !coir.tensor<128x64xf16, shared>,
    %b: !coir.tensor<64x128xf16, shared>,
    %c: !coir.tensor<128x128xf32, shared>) {
  coir.parallel (%bm, %bn) in [2, 2] level = #coir.level<block> {
    %zero = arith.constant 0.0 : f32
    %c4 = arith.constant 4 : index
    %init = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
    %final = coir.foreach %k in %c4 iter_args(%acc = %init) : !coir.mma_frag<16x16xf32> {
      %at = coir.tensor.tile %a[%k] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<16x16xf16, shared>
      %bt = coir.tensor.tile %b[%k] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x16xf16, shared>
      %af = coir.mma.load %at : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
      %bf = coir.mma.load %bt : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
      %r = coir.mma.exec %acc, %af, %bf {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
      coir.yield %r : !coir.mma_frag<16x16xf32>
    }
    coir.mma.store %final, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<128x128xf32, shared>
  }
}

} // module
