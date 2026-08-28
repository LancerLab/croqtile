// RUN: coir-opt --coir-emit-cuda %s | FileCheck %s

// Test fence-order emission on the CUDA target:
//   - seq_cst emits the sequentially-consistent fence.sc.* barrier
//   - release/acquire have no standalone form and collapse to the full
//     acq_rel barrier (__threadfence_block within a CTA, __threadfence
//     device-wide)

module attributes { "coir.has_tma" = false, "coir.has_dma" = false } {

// CHECK: __global__ void __fence_order_kernel__(float* arg0) {
coir.kernel @fence_order(%a: !coir.tensor<64xf32>) {

  // CHECK: asm volatile("fence.sc.gpu;" ::: "memory");
  coir.fence <global> <all> <seq_cst> <block>

  // CHECK: asm volatile("fence.sc.cta;" ::: "memory");
  coir.fence <global> <all> <seq_cst> <thread>

  // CHECK: __threadfence();
  coir.fence <global> <all> <release> <block>

  // CHECK: __threadfence_block();
  coir.fence <global> <all> <acquire> <thread>

  // CHECK: __threadfence_block();
  coir.fence <global> <all> <acq_rel> <thread>
}

}
