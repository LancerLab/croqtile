// Test that LowerMMA reads coir.mma_target from module attributes
// rather than requiring --target-arch.

// RUN: coir-opt --coir-lower-mma %s | FileCheck %s

// The module declares coir.mma_target = "ukernel"; LowerMMA should use it
// as the primary source of truth (no --target-arch needed).
module attributes {
  "coir.mma_target" = "ukernel",
  "coir.arch" = "custom_arch_v1"
} {

// CHECK-LABEL: coir.kernel @test_module_attr_ukernel
// CHECK: coir.mma.exec {{.*}} target = "ukernel"
coir.kernel @test_module_attr_ukernel(
    %a: !coir.tensor<16x16xf16, shared>,
    %b: !coir.tensor<16x16xf16, shared>,
    %c: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %af = coir.mma.load %a : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %bf = coir.mma.load %b : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %r = coir.mma.exec %acc, %af, %bf {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  coir.mma.store %r, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

} // module
