// RUN: coir-opt --coir-fence-elision %s | FileCheck %s

// Test 1: two adjacent fences of the same four-fold kind; the second is
// redundant (no intervening memory effect) and is dropped.

// CHECK-LABEL: coir.kernel @dedup_adjacent
coir.kernel @dedup_adjacent() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK-NOT: coir.fence
  coir.fence <shared> <dma> <release> <none>
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 2: adjacent fences of different kinds are both kept.

// CHECK-LABEL: coir.kernel @keep_different_kinds
coir.kernel @keep_different_kinds() {
  // CHECK: coir.fence <local> <dma> <release> <none>
  // CHECK: coir.fence <local> <threads> <release> <none>
  coir.fence <local> <dma> <release> <none>
  coir.fence <local> <threads> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 3: an intervening memory effect (coir.dma.copy) breaks adjacency, so
// the second fence is kept even though its kind matches the first.

// CHECK-LABEL: coir.kernel @keep_intervening_effect
coir.kernel @keep_intervening_effect(
    %a: !coir.tensor<64xf32, shared>,
    %c: !coir.tensor<64xf32, local>) {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK: coir.dma.copy
  // CHECK: coir.fence <shared> <dma> <release> <none>
  coir.fence <shared> <dma> <release> <none>
  %tok = coir.dma.copy %a to %c
    : !coir.tensor<64xf32, shared> -> !coir.tensor<64xf32, local>
  coir.fence <shared> <dma> <release> <none>
  coir.wait %tok : !coir.async
}

// Test 4: an intervening memory-effect-free operation (arith.constant) does
// NOT break adjacency, so the second fence is still dropped.

// CHECK-LABEL: coir.kernel @dedup_effect_free_between
coir.kernel @dedup_effect_free_between() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK-NOT: coir.fence
  coir.fence <shared> <dma> <release> <none>
  %c0 = arith.constant 0 : index
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 5: fences differing in any single axis are not deduplicated.

// CHECK-LABEL: coir.kernel @keep_different_axes
coir.kernel @keep_different_axes() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK: coir.fence <shared> <dma> <acquire> <none>
  // CHECK: coir.fence <global> <dma> <release> <none>
  coir.fence <shared> <dma> <release> <none>
  coir.fence <shared> <dma> <acquire> <none>
  coir.fence <global> <dma> <release> <none>
  coir.barrier #coir.level<block>
}
