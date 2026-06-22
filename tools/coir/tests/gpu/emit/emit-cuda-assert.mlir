// RUN: coir-opt --coir-emit-cuda %s | FileCheck %s

// Test that coir.assert ops are emitted as the correct runtime check API
// based on their site attribute:
//   ENTRY -> choreo::runtime_check (host wrapper, before launch)
//   USE   -> choreo::choreo_assert (device kernel body)

module attributes { "coir.target" = "cute", "coir.arch" = "sm_86" } {

// CHECK: __global__ void __assert_test_kernel__
coir.kernel @assert_test(
    %src: !coir.tensor<128xf32, global>,
    %dst: !coir.tensor<128xf32, shared>) -> !coir.tensor<128xf32, shared> {

  // Entry-site assertion: emitted in host wrapper
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %entry_cmp = arith.cmpi sgt, %c128, %c0 : index
  coir.assert %entry_cmp, "positive dimension" <entry> <hw_constraint>

  // Use-site assertion: emitted in device kernel body
  // CHECK: choreo::choreo_assert
  // CHECK-SAME: element bound
  %c64 = arith.constant 64 : index
  %use_cmp = arith.cmpi sle, %c64, %c128 : index
  coir.assert %use_cmp, "element bound" <use> <element_access>

  coir.return %dst : !coir.tensor<128xf32, shared>
}

// Host wrapper should contain runtime_check for entry assertion
// CHECK: assert_test
// CHECK: runtime_check
// CHECK-SAME: positive dimension
}
