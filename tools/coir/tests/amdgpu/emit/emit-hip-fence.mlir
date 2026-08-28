// RUN: coir-opt --coir-emit-hip %s | FileCheck %s

// Test fence-order emission on the AMDGPU target:
//   - release/acquire emit directional __builtin_amdgcn_fence barriers
//   - acq_rel keeps the full __threadfence* forms
//   - seq_cst emits the sequentially-consistent __builtin_amdgcn_fence barrier

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// CHECK: __global__ void __fence_order_kernel__(float* arg0) {
coir.kernel @fence_order(%a: !coir.tensor<64xf32>) {

  // CHECK: __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  coir.fence <global> <all> <release> <block>

  // CHECK: __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  coir.fence <global> <all> <acquire> <thread>

  // CHECK: __threadfence_block();
  coir.fence <global> <all> <acq_rel> <thread>

  // CHECK: __threadfence();
  coir.fence <global> <all> <acq_rel> <block>

  // CHECK: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
  coir.fence <global> <all> <seq_cst> <thread>

  // CHECK: __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
  coir.fence <global> <all> <seq_cst> <block>
}

}
