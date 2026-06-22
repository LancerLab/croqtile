// RUN: coir-opt --coir-hoist-assertions %s | FileCheck %s

// Test 1: Assert whose condition only depends on constants (kernel-arg
// invariant) -> hoisted to ENTRY with its def-chain, moved before the loop.
// CHECK-LABEL: coir.kernel @test_hoist_to_entry
coir.kernel @test_hoist_to_entry(
    %src: !coir.tensor<128x64xf32, global>,
    %dst: !coir.tensor<128x64xf32, shared>) {
  %c16 = arith.constant 16 : index
  // The assert and its operand chain should be above the foreach:
  // CHECK: arith.cmpi sle
  // CHECK-NEXT: coir.assert {{.*}} <entry>
  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %c8192 = arith.constant 8192 : index
    %c8192_0 = arith.constant 8192 : index
    %cmp = arith.cmpi sle, %c8192, %c8192_0 : index
    coir.assert %cmp, "shape check" <use> <shape_compat>
    coir.yield
  }
}

// Test 2: Assert that depends on loop IV -> stays at USE inside the loop.
// CHECK-LABEL: coir.kernel @test_stay_at_use
coir.kernel @test_stay_at_use(
    %src: !coir.tensor<128x64xf32, global>) {
  %c16 = arith.constant 16 : index
  // CHECK: coir.foreach
  // CHECK: coir.assert {{.*}} <use>
  coir.foreach %k in %c16 {
    %c64 = arith.constant 64 : index
    %cmp = arith.cmpi slt, %k, %c64 : index
    coir.assert %cmp, "bound check" <use> <loop_bound>
    coir.yield
  }
}

// Test 3: Assert already at ENTRY -> left in place.
// CHECK-LABEL: coir.kernel @test_already_entry
coir.kernel @test_already_entry(
    %src: !coir.tensor<128x64xf32, global>) {
  // CHECK: coir.assert {{.*}} <entry>
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cmp = arith.cmpi sgt, %c128, %c0 : index
  coir.assert %cmp, "positive dim" <entry> <hw_constraint>
}
