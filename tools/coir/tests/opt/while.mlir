// RUN: coir-opt %s | FileCheck %s

// Test coir.while, coir.while.cond, coir.break, coir.continue ops.
// CHECK-LABEL: coir.kernel @test_coir_while
coir.kernel @test_coir_while(%buf: !coir.tensor<128xi32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // CHECK: coir.while
  // CHECK: coir.while.cond
  // CHECK: coir.continue
  %res = coir.while (%i = %c0) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %i, %c128 : i32
    coir.while.cond(%cond) %i : i32
  } {
  ^bb0(%ib: i32):
    %c1 = arith.constant 1 : i32
    %next = arith.addi %ib, %c1 : i32
    coir.continue %next : i32
  }
  coir.return
}

// Test coir.break inside scf.if (non-terminator)
// CHECK-LABEL: coir.kernel @test_break
coir.kernel @test_break(%buf: !coir.tensor<128xi32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  // CHECK: coir.break
  // CHECK: coir.continue
  %res = coir.while (%i = %c0) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %i, %c128 : i32
    coir.while.cond(%cond) %i : i32
  } {
  ^bb0(%ib: i32):
    %c10 = arith.constant 10 : i32
    %eq = arith.cmpi eq, %ib, %c10 : i32
    scf.if %eq {
      coir.break %ib : i32
    }
    %c1 = arith.constant 1 : i32
    %next = arith.addi %ib, %c1 : i32
    coir.continue %next : i32
  }
  coir.return
}
