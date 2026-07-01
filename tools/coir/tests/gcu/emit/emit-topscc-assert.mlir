// RUN: coir-opt --coir-emit-topscc %s | FileCheck %s
//
// Test that coir.assert emits choreo::choreo_assert for USE-site
// and choreo::runtime_check for ENTRY-site on GCU/topscc.

module attributes { "coir.target" = "topscc", "coir.arch" = "gcu300" } {

coir.kernel @assert_test(
    %src: !coir.tensor<128xf32, global>) -> !coir.tensor<128xf32, shared> {

  // Entry assertion: emitted as runtime_check in host wrapper
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %entry_cmp = arith.cmpi sgt, %c128, %c0 : index
  coir.assert %entry_cmp, "positive dimension" <entry> <hw_constraint>

  // Use-site assertion: emitted inline as choreo_assert
  // CHECK: choreo::choreo_assert(
  // CHECK-SAME: element bound
  %c64 = arith.constant 64 : index
  %use_cmp = arith.cmpi sle, %c64, %c128 : index
  coir.assert %use_cmp, "element bound" <use> <element_access>

  %dst = coir.tensor.alloc : !coir.tensor<128xf32, shared>
  coir.return %dst : !coir.tensor<128xf32, shared>
}

// CHECK: runtime_check(
// CHECK-SAME: positive dimension
}
