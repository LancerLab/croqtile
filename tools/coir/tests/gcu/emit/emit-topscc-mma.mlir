// RUN: coir-opt --coir-lower-mma --coir-emit-topscc %s 2>&1 | FileCheck %s

module attributes { "coir.mma_target" = "ukernel" } {

// CHECK: #include <common/acore_op.h>

// Test: simple MMA kernel emits stub declaration and call
// CHECK: __device__ void __choreo_mma_f16_f32_M16_MK_KN(
// CHECK: __device__ void __choreo_device_mma_simple(
// CHECK: int __mma_ws_{{.*}}[2048]
// CHECK: __choreo_mma_f16_f32_M16_MK_KN(
// CHECK: 0, 1, 0, 0)
coir.kernel @mma_simple(
    %a_tile: !coir.tensor<16x16xf16, shared>,
    %b_tile: !coir.tensor<16x16xf16, shared>,
    %c_tile: !coir.tensor<16x16xf32, shared>) {
  coir.parallel (%p) in [1] level = #coir.level<block> {
    %zero = arith.constant 0.0 : f32
    %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
    %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
    coir.mma.store %res, %c_tile : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
  }
}

// Test: K-loop accumulation with deferred exec+store pattern
// CHECK: __device__ void __choreo_mma_f16_f32_M128_MK_KN(
// CHECK: __device__ void __choreo_device_mma_kloop(
// CHECK: int __mma_ws_{{.*}}[2048]
// CHECK: void* __mma_ws_{{.*}}_last_lhs
// CHECK: void* __mma_ws_{{.*}}_last_rhs
// CHECK: for (int
// CHECK: __choreo_mma_f16_f32_M128_MK_KN(
// CHECK: , 0, 0,
// CHECK: __choreo_mma_f16_f32_M128_MK_KN(
// CHECK: 1, 1, 0, 1)

// Verify stub definitions contain acore::matmul calls
// CHECK: acore::matmul<16, acore::MK_KN>
// CHECK: acore::matmul<128, acore::MK_KN>
coir.kernel @mma_kloop(
    %a: !coir.tensor<128x64xf16, shared>,
    %b: !coir.tensor<64x128xf16, shared>,
    %c: !coir.tensor<128x128xf32, shared>) {
  coir.parallel (%p) in [1] level = #coir.level<block> {
    %zero = arith.constant 0.0 : f32
    %c4 = arith.constant 4 : index
    %init = coir.mma.fill %zero : f32 -> !coir.mma_frag<128x16xf32>
    %final = coir.foreach %k in %c4 iter_args(%acc = %init) : !coir.mma_frag<128x16xf32> {
      %at = coir.tensor.tile %a[%k] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<128x16xf16, shared>
      %bt = coir.tensor.tile %b[%k] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x128xf16, shared>
      %af = coir.mma.load %at : !coir.tensor<128x16xf16, shared> -> !coir.mma_frag<128x16xf16>
      %bf = coir.mma.load %bt : !coir.tensor<16x128xf16, shared> -> !coir.mma_frag<16x128xf16>
      %r = coir.mma.exec %acc, %af, %bf {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<128x16xf32>, !coir.mma_frag<128x16xf16>, !coir.mma_frag<16x128xf16>) -> !coir.mma_frag<128x16xf32>
      coir.yield %r : !coir.mma_frag<128x16xf32>
    }
    coir.mma.store %final, %c : !coir.mma_frag<128x16xf32>, !coir.tensor<128x128xf32, shared>
  }
}

} // module
