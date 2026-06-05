// RUN: coir-opt %s | FileCheck %s

// Verify that a simple elementwise add kernel in CoIR MLIR is well-formed
// and roundtrips through coir-opt.

// CHECK: coir.kernel @ele_add
// CHECK-SAME: !coir.tensor<64xi32, global>
// CHECK-SAME: -> !coir.tensor<64xi32, global>
// CHECK: coir.tensor.alloc
// CHECK: coir.parallel
// CHECK-SAME: level = #coir.level<thread>
// CHECK: coir.tensor.load_elem
// CHECK: coir.tensor.load_elem
// CHECK: arith.addi
// CHECK: coir.tensor.store_elem
// CHECK: coir.return

coir.kernel @ele_add(%a: !coir.tensor<64xi32, global>, %b: !coir.tensor<64xi32, global>)
    -> !coir.tensor<64xi32, global> {
  %c = coir.tensor.alloc : !coir.tensor<64xi32, global>
  coir.parallel (%p) in [64] level = #coir.level<thread> {
    %av = coir.tensor.load_elem %a[%p] : !coir.tensor<64xi32, global> -> i32
    %bv = coir.tensor.load_elem %b[%p] : !coir.tensor<64xi32, global> -> i32
    %sum = arith.addi %av, %bv : i32
    coir.tensor.store_elem %sum, %c[%p] : i32, !coir.tensor<64xi32, global>
    coir.yield
  }
  coir.return %c : !coir.tensor<64xi32, global>
}
