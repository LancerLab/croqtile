// RUN: coir-opt --coir-emit-cuda %s | FileCheck %s

// EmitCUDA test: coir.while with break/continue generates C++ while/break/continue.

module attributes { "coir.target" = "cute", "coir.arch" = "sm_86" } {

// CHECK: while (
// CHECK: break;
// CHECK: continue;
coir.kernel @test_coir_while(%buf: !coir.tensor<128xi32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %res = coir.while (%i = %c0) : (i32) -> (i32) {
    %cond = arith.cmpi slt, %i, %c128 : i32
    coir.while.cond(%cond) %i : i32
  } {
  ^bb0(%ib: i32):
    %c10 = arith.constant 10 : i32
    %should_break = arith.cmpi eq, %ib, %c10 : i32
    scf.if %should_break {
      coir.break %ib : i32
    }
    %c1 = arith.constant 1 : i32
    %next = arith.addi %ib, %c1 : i32
    coir.continue %next : i32
  }
  coir.return
}

}
