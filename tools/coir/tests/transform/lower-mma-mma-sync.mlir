// Test LowerMMA with mma_sync target (SM80).
// RUN: coir-opt --coir-lower-mma %s | FileCheck %s

module attributes { "coir.mma_target" = "mma_sync" } {

// CHECK-LABEL: coir.kernel @test_mma_sync
// CHECK: coir.mma.exec {{.*}} target = "mma_sync"
coir.kernel @test_mma_sync(
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
