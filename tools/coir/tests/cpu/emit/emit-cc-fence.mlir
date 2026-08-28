// RUN: coir-opt --coir-emit-cc %s | FileCheck %s

// Test fence-order emission on the CPU target: each order maps to the
// matching std::memory_order for std::atomic_thread_fence.

// CHECK: void __fence_order_impl__(float* arg0) {
coir.kernel @fence_order(%a: !coir.tensor<64xf32>) {

  // CHECK: std::atomic_thread_fence(std::memory_order_release);
  coir.fence <global> <all> <release> <block>

  // CHECK: std::atomic_thread_fence(std::memory_order_acquire);
  coir.fence <global> <all> <acquire> <block>

  // CHECK: std::atomic_thread_fence(std::memory_order_acq_rel);
  coir.fence <global> <all> <acq_rel> <block>

  // CHECK: std::atomic_thread_fence(std::memory_order_seq_cst);
  coir.fence <global> <all> <seq_cst> <block>
}
