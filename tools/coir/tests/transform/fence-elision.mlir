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


// Test 6: order subsumption -- acq_rel subsumes a later release.

// CHECK-LABEL: coir.kernel @sub_order_acq_rel_release
coir.kernel @sub_order_acq_rel_release() {
  // CHECK: coir.fence <shared> <dma> <acq_rel> <none>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  coir.fence <shared> <dma> <acq_rel> <none>
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 7: order subsumption -- seq_cst subsumes a later acq_rel.

// CHECK-LABEL: coir.kernel @sub_order_seq_cst
coir.kernel @sub_order_seq_cst() {
  // CHECK: coir.fence <shared> <dma> <seq_cst> <none>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  coir.fence <shared> <dma> <seq_cst> <none>
  coir.fence <shared> <dma> <acq_rel> <none>
  coir.barrier #coir.level<block>
}

// Test 8: the converse does not hold -- acq_rel does NOT subsume seq_cst.

// CHECK-LABEL: coir.kernel @keep_acq_rel_then_seq_cst
coir.kernel @keep_acq_rel_then_seq_cst() {
  // CHECK: coir.fence <shared> <dma> <acq_rel> <none>
  // CHECK: coir.fence <shared> <dma> <seq_cst> <none>
  coir.fence <shared> <dma> <acq_rel> <none>
  coir.fence <shared> <dma> <seq_cst> <none>
  coir.barrier #coir.level<block>
}

// Test 9: entity subsumption -- `all` subsumes a specific engine fence.

// CHECK-LABEL: coir.kernel @sub_entity_all
coir.kernel @sub_entity_all() {
  // CHECK: coir.fence <shared> <all> <release> <none>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  coir.fence <shared> <all> <release> <none>
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 10: proxy separation -- a threads (generic proxy) fence does NOT
// subsume a tma (async proxy) fence.

// CHECK-LABEL: coir.kernel @keep_threads_vs_tma
coir.kernel @keep_threads_vs_tma() {
  // CHECK: coir.fence <shared> <threads> <release> <none>
  // CHECK: coir.fence <shared> <tma> <release> <none>
  coir.fence <shared> <threads> <release> <none>
  coir.fence <shared> <tma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 11: scope subsumption -- a device-scope fence subsumes a later
// block-scope fence of the same space/entity/order.

// CHECK-LABEL: coir.kernel @sub_scope_device
coir.kernel @sub_scope_device() {
  // CHECK: coir.fence <global> <threads> <acq_rel> <device>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  coir.fence <global> <threads> <acq_rel> <device>
  coir.fence <global> <threads> <acq_rel> <block>
  coir.barrier #coir.level<block>
}

// Test 12: the converse does not hold -- block scope does NOT subsume
// device scope.

// CHECK-LABEL: coir.kernel @keep_scope_block_then_device
coir.kernel @keep_scope_block_then_device() {
  // CHECK: coir.fence <global> <threads> <acq_rel> <block>
  // CHECK: coir.fence <global> <threads> <acq_rel> <device>
  coir.fence <global> <threads> <acq_rel> <block>
  coir.fence <global> <threads> <acq_rel> <device>
  coir.barrier #coir.level<block>
}

// Test 13: intervening fences are transparent -- a fence matching an earlier
// one is dropped even with a different-kind fence in between.

// CHECK-LABEL: coir.kernel @look_through_intervening_fence
coir.kernel @look_through_intervening_fence() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK: coir.fence <global> <threads> <acq_rel> <device>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  coir.fence <shared> <dma> <release> <none>
  coir.fence <global> <threads> <acq_rel> <device>
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 14: cross-block availability -- a fence before scf.if is available in
// both branches; acq_rel subsumes the release and the acquire inside.

// CHECK-LABEL: coir.kernel @fence_before_if
coir.kernel @fence_before_if() {
  // CHECK: coir.fence <shared> <dma> <acq_rel> <none>
  // CHECK-NOT: coir.fence
  // CHECK: coir.barrier
  %cond = arith.constant true
  coir.fence <shared> <dma> <acq_rel> <none>
  scf.if %cond {
    coir.fence <shared> <dma> <release> <none>
  } else {
    coir.fence <shared> <dma> <acquire> <none>
  }
  coir.barrier #coir.level<block>
}

// Test 15: a fence inside one branch is NOT available after the join -- the
// region-holding op is a memory boundary, so the same fence after the if is
// kept.

// CHECK-LABEL: coir.kernel @branch_fence_not_available
coir.kernel @branch_fence_not_available() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK: coir.fence <shared> <dma> <release> <none>
  %cond = arith.constant true
  scf.if %cond {
    coir.fence <shared> <dma> <release> <none>
  }
  coir.fence <shared> <dma> <release> <none>
  coir.barrier #coir.level<block>
}

// Test 16: loop conservatism -- the pre-loop fence does not justify the
// in-loop fence (the body's later iterations re-execute with intervening
// effects), so the inner fence is kept.

// CHECK-LABEL: coir.kernel @loop_keeps_inner_fence
coir.kernel @loop_keeps_inner_fence() {
  // CHECK: coir.fence <shared> <dma> <release> <none>
  // CHECK: coir.fence <shared> <dma> <release> <none>
  %c4 = arith.constant 4 : index
  coir.fence <shared> <dma> <release> <none>
  coir.foreach %i in %c4 {
    coir.fence <shared> <dma> <release> <none>
    coir.yield
  }
  coir.barrier #coir.level<block>
}
