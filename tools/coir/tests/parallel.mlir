// RUN: coir-opt %s | FileCheck %s

// Test coir.parallel op
// CHECK-LABEL: coir.kernel @test_parallel
coir.kernel @test_parallel(%a: !coir.tensor<128x256xf16>) {
  // CHECK: coir.parallel (%arg1, %arg2) in [2, 16] level = #coir.level<block>
  coir.parallel (%m, %n) in [2, 16] level = #coir.level<block> {
  }
}

// Test nested parallel ops
// CHECK-LABEL: coir.kernel @test_nested_parallel
coir.kernel @test_nested_parallel(%a: !coir.tensor<128x256xf16>) {
  // CHECK: coir.parallel (%arg1, %arg2) in [2, 16] level = #coir.level<block>
  coir.parallel (%bm, %bn) in [2, 16] level = #coir.level<block> {
    // CHECK: coir.parallel (%arg3) in [4] level = #coir.level<warpgroup>
    coir.parallel (%wg) in [4] level = #coir.level<warpgroup> {
    }
  }
}
