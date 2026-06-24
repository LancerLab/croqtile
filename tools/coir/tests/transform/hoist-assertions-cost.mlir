// RUN: coir-opt --coir-hoist-assertions --coir-estimate-assert-cost="cost-threshold=4" %s | FileCheck %s
// RUN: coir-opt --coir-hoist-assertions --coir-estimate-assert-cost="cost-threshold=0" %s | FileCheck --check-prefix=NONE %s

// Test 1: Assertion inside foreach, but condition is all constants
// -> hoisted to ENTRY at kernel level -> cost=1 (entry block), enabled.
// CHECK-LABEL: coir.kernel @test_cost_entry
// CHECK: coir.assert %{{.*}}, "positive dim" <entry> <hw_constraint> {cost_class = #coir.cost_class<entry>, enabled = true, estimated_cost = 1 : i64}
// CHECK: coir.foreach
// NONE-LABEL: coir.kernel @test_cost_entry
// NONE: coir.assert %{{.*}}, "positive dim" <entry> <hw_constraint> {cost_class = #coir.cost_class<entry>, enabled = false, estimated_cost = 1 : i64}
coir.kernel @test_cost_entry(
    %src: !coir.tensor<128xf32, global>) {
  %c16 = arith.constant 16 : index
  coir.foreach %k in %c16 {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cmp = arith.cmpi sgt, %c128, %c0 : index
    coir.assert %cmp, "positive dim" <use> <hw_constraint>
    coir.yield
  }
}

// Test 2: Assertion inside single foreach with static bound 32.
// cost = 32 -> LOW. Enabled at threshold=4 (all), disabled at threshold=0.
// CHECK-LABEL: coir.kernel @test_cost_single_foreach
// CHECK: coir.foreach
// CHECK: coir.assert %{{.*}}, "bound check" <use> <loop_bound> {cost_class = #coir.cost_class<low>, enabled = true, estimated_cost = 32 : i64}
coir.kernel @test_cost_single_foreach(
    %src: !coir.tensor<128xf32, global>) {
  %c32 = arith.constant 32 : index
  coir.foreach %k in %c32 {
    %c64 = arith.constant 64 : index
    %cmp = arith.cmpi slt, %k, %c64 : index
    coir.assert %cmp, "bound check" <use> <loop_bound>
    coir.yield
  }
}

// Test 3: Assertion inside nested foreach (32 x 64 = 2048).
// cost = 2048 -> HIGH. Enabled at threshold=4, disabled at threshold=0.
// CHECK-LABEL: coir.kernel @test_cost_nested_foreach
// CHECK: coir.assert %{{.*}}, "nested bound" <use> <element_access> {cost_class = #coir.cost_class<high>, enabled = true, estimated_cost = 2048 : i64}
coir.kernel @test_cost_nested_foreach(
    %src: !coir.tensor<128xf32, global>) {
  %c32 = arith.constant 32 : index
  coir.foreach %i in %c32 {
    %c64 = arith.constant 64 : index
    coir.foreach %j in %c64 {
      %cmp = arith.cmpi slt, %j, %c64 : index
      coir.assert %cmp, "nested bound" <use> <element_access>
      coir.yield
    }
    coir.yield
  }
}

// Test 4: Assertion inside foreach with dynamic bound (unknown).
// cost = 100 (default) -> MEDIUM.
// CHECK-LABEL: coir.kernel @test_cost_foreach_dynamic
// CHECK: coir.assert %{{.*}}, "dynamic bound" <use> <loop_bound> {cost_class = #coir.cost_class<medium>, enabled = true, estimated_cost = 100 : i64}
coir.kernel @test_cost_foreach_dynamic(
    %src: !coir.tensor<128xf32, global>,
    %n: index) {
  coir.foreach %k in %n {
    %c64 = arith.constant 64 : index
    %cmp = arith.cmpi slt, %k, %c64 : index
    coir.assert %cmp, "dynamic bound" <use> <loop_bound>
    coir.yield
  }
}

// Test 5: Assertion inside scf.while -> cost = 100 (conservative default).
// CHECK-LABEL: coir.kernel @test_cost_while
// CHECK: coir.assert %{{.*}}, "while bound" <use> <hw_constraint> {cost_class = #coir.cost_class<medium>, enabled = true, estimated_cost = 100 : i64}
coir.kernel @test_cost_while(
    %src: !coir.tensor<128xf32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %res = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c128 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%iv: i32):
    %c100 = arith.constant 100 : index
    %c0_1 = arith.constant 0 : index
    %check = arith.cmpi sgt, %c100, %c0_1 : index
    coir.assert %check, "while bound" <use> <hw_constraint>
    %c1 = arith.constant 1 : i32
    %next = arith.addi %iv, %c1 : i32
    scf.yield %next : i32
  }
  coir.return
}

// Test 6: Threshold filtering -- cost-threshold=1 (ENTRY only).
// Assertion at cost=32 (LOW) should be disabled.
// RUN: coir-opt --coir-hoist-assertions --coir-estimate-assert-cost="cost-threshold=1" %s | FileCheck --check-prefix=ENTRY %s
// ENTRY-LABEL: coir.kernel @test_cost_threshold_disable
// ENTRY: coir.assert %{{.*}}, "disabled bound" <use> <loop_bound> {cost_class = #coir.cost_class<low>, enabled = false, estimated_cost = 32 : i64}
coir.kernel @test_cost_threshold_disable(
    %src: !coir.tensor<128xf32, global>) {
  %c32 = arith.constant 32 : index
  coir.foreach %k in %c32 {
    %c64 = arith.constant 64 : index
    %cmp = arith.cmpi slt, %k, %c64 : index
    coir.assert %cmp, "disabled bound" <use> <loop_bound>
    coir.yield
  }
}

// Test 7: Foreach with static bound 8 -> cost=8 -> LOW.
// CHECK-LABEL: coir.kernel @test_cost_small_foreach
// CHECK: coir.assert %{{.*}}, "small loop" <use> <loop_bound> {cost_class = #coir.cost_class<low>, enabled = true, estimated_cost = 8 : i64}
coir.kernel @test_cost_small_foreach(
    %src: !coir.tensor<128xf32, global>) {
  %c8 = arith.constant 8 : index
  coir.foreach %k in %c8 {
    %c64 = arith.constant 64 : index
    %cmp = arith.cmpi slt, %k, %c64 : index
    coir.assert %cmp, "small loop" <use> <loop_bound>
    coir.yield
  }
}

// Test 8: Foreach (8) x foreach (8) = 64 -> MEDIUM.
// CHECK-LABEL: coir.kernel @test_cost_medium_nested
// CHECK: coir.assert %{{.*}}, "medium nested" <use> <element_access> {cost_class = #coir.cost_class<medium>, enabled = true, estimated_cost = 64 : i64}
coir.kernel @test_cost_medium_nested(
    %src: !coir.tensor<128xf32, global>) {
  %c8 = arith.constant 8 : index
  coir.foreach %i in %c8 {
    coir.foreach %j in %c8 {
      %cmp = arith.cmpi slt, %j, %c8 : index
      coir.assert %cmp, "medium nested" <use> <element_access>
      coir.yield
    }
    coir.yield
  }
}
