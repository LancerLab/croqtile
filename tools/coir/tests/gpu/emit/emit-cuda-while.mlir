// RUN: coir-opt --coir-emit-cuda %s | FileCheck %s

// EmitCUDA test: scf.while generates a C++ while loop.

module attributes { "coir.target" = "cute", "coir.arch" = "sm_86" } {

// CHECK-NOT: [unhandled]
// CHECK: while (
coir.kernel @test_while(%buf: !coir.tensor<128xi32, global>) {
  %c0 = arith.constant 0 : i32
  %c128 = arith.constant 128 : i32
  %res = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c128 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%iv: i32):
    %c1 = arith.constant 1 : i32
    %next = arith.addi %iv, %c1 : i32
    scf.yield %next : i32
  }
  coir.return
}

}
