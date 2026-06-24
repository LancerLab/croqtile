// RUN: coir-opt --coir-hoist-assertions %s | FileCheck %s

// Test 1: Assert whose condition only depends on constants (kernel-arg
// invariant) -> hoisted to ENTRY with its def-chain, moved before the loop.
// CHECK-LABEL: coir.kernel @test_hoist_to_entry
coir.kernel @test_hoist_to_entry(
    %src: !coir.tensor<128x64xf32, global>,
    %dst: !coir.tensor<128x64xf32, shared>) {
  %c16 = arith.constant 16 : index
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

// Test 4: Assert inside scf.if using foreach IV %k.
// Post-dom lifts past scf.if, but %k pins it in the foreach -> HOIST.
// CHECK-LABEL: coir.kernel @test_if_with_loop_iv
coir.kernel @test_if_with_loop_iv(
    %arg0: !coir.tensor<128xf32, global>) {
  %c16 = arith.constant 16 : index
  // CHECK: coir.foreach
  // The assert should be hoisted out of scf.if to foreach body level.
  // CHECK: arith.cmpi slt
  // CHECK: coir.assert {{.*}} <hoist>
  // CHECK: scf.if
  coir.foreach %k in %c16 {
    %c64 = arith.constant 64 : index
    %guard = arith.cmpi slt, %k, %c64 : index
    scf.if %guard {
      %c32 = arith.constant 32 : index
      %check = arith.cmpi slt, %k, %c32 : index
      coir.assert %check, "guarded bound" <use> <element_access>
      scf.yield
    }
    coir.yield
  }
}

// Test 5: Assert inside scf.if but condition is independent of if and loop
// -> can hoist to ENTRY.
// CHECK-LABEL: coir.kernel @test_independent_in_if
coir.kernel @test_independent_in_if(
    %arg0: !coir.tensor<128xf32, global>) {
  %c16 = arith.constant 16 : index
  // CHECK: coir.assert {{.*}} <entry>
  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %c64 = arith.constant 64 : index
    %guard = arith.cmpi slt, %k, %c64 : index
    scf.if %guard {
      %c100 = arith.constant 100 : index
      %c0 = arith.constant 0 : index
      %check = arith.cmpi sgt, %c100, %c0 : index
      coir.assert %check, "independent check" <use> <hw_constraint>
      scf.yield
    }
    coir.yield
  }
}

// Test 6: Assert uses an scf.if result value.
// Pinned at scf.if level (the result is defined there).
// CHECK-LABEL: coir.kernel @test_if_result_pin
coir.kernel @test_if_result_pin(
    %arg0: !coir.tensor<128xf32, global>) {
  %c16 = arith.constant 16 : index
  coir.foreach %k in %c16 {
    %c64 = arith.constant 64 : index
    %guard = arith.cmpi slt, %k, %c64 : index
    %val = scf.if %guard -> (index) {
      %c1 = arith.constant 1 : index
      scf.yield %c1 : index
    } else {
      %c0 = arith.constant 0 : index
      scf.yield %c0 : index
    }
    %c0_1 = arith.constant 0 : index
    %check = arith.cmpi sgt, %val, %c0_1 : index
    // Uses %val which is defined by scf.if -> cannot hoist past scf.if
    // CHECK: scf.if
    // CHECK: coir.assert {{.*}} <use>
    coir.assert %check, "uses if result" <use> <element_access>
    coir.yield
  }
}

// Test 7: Assert inside scf.while body -> conservative, stays in while.
// CHECK-LABEL: coir.kernel @test_scf_while_conservative
coir.kernel @test_scf_while_conservative(
    %arg0: !coir.tensor<128xf32, global>) {
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
    // CHECK: scf.while
    // CHECK: coir.assert {{.*}} <use>
    coir.assert %check, "inside scf.while" <use> <hw_constraint>
    %c1 = arith.constant 1 : i32
    %next = arith.addi %iv, %c1 : i32
    scf.yield %next : i32
  }
  coir.return
}

// Test 8: Assert inside coir.while body -> conservative, stays in while.
// CHECK-LABEL: coir.kernel @test_coir_while_conservative
coir.kernel @test_coir_while_conservative(
    %arg0: !coir.tensor<128xf32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %res = coir.while (%i = %c0) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %i, %c128 : i32
    coir.while.cond(%cond) %i : i32
  } {
  ^bb0(%ib: i32):
    %c100 = arith.constant 100 : index
    %c0_1 = arith.constant 0 : index
    %check = arith.cmpi sgt, %c100, %c0_1 : index
    // CHECK: coir.while
    // CHECK: coir.assert {{.*}} <use>
    coir.assert %check, "inside coir.while" <use> <hw_constraint>
    %c1 = arith.constant 1 : i32
    %next = arith.addi %ib, %c1 : i32
    coir.continue %next : i32
  }
  coir.return
}

// Test 9: Assert inside nested foreach + scf.if.
// Condition uses only outer foreach args -> hoists past inner foreach
// and scf.if to outer foreach level.
// CHECK-LABEL: coir.kernel @test_nested_hoist
coir.kernel @test_nested_hoist(
    %arg0: !coir.tensor<128xf32, global>) {
  %c8 = arith.constant 8 : index
  coir.foreach %i in %c8 {
    %c4 = arith.constant 4 : index
    coir.foreach %j in %c4 {
      %guard = arith.cmpi slt, %j, %c4 : index
      scf.if %guard {
        %c100 = arith.constant 100 : index
        %c0 = arith.constant 0 : index
        %check = arith.cmpi sgt, %c100, %c0 : index
        // Deps are all constants -> should hoist to ENTRY
        // CHECK: coir.assert {{.*}} <entry>
        coir.assert %check, "nested independent" <use> <hw_constraint>
        scf.yield
      }
      coir.yield
    }
    coir.yield
  }
}
