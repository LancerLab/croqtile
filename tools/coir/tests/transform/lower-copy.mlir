// RUN: coir-opt --coir-lower-copy %s | FileCheck %s
// RUN: coir-opt --coir-classify-copies --coir-lower-copy %s | FileCheck %s --check-prefix=PIPE

module attributes { "coir.has_tma" = true } {

// Test: element.copy gets element count annotation
// CHECK-LABEL: coir.kernel @test_element_copy
// CHECK: coir.element.copy %arg0 to %arg1 {lowered, total_elements = 256 : i64}
coir.kernel @test_element_copy(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, shared>) {
  coir.element.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}

// Test: dma.copy gets transfer metadata
// CHECK-LABEL: coir.kernel @test_dma_copy
// CHECK: coir.dma.copy %arg0 to %arg1 {lowered, mechanism = "cp_async", transfer_bytes = 16384 : i64}
coir.kernel @test_dma_copy(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  %tok = coir.dma.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.wait %tok : !coir.token
}

// Test: tma.copy gets TMA transfer metadata
// CHECK-LABEL: coir.kernel @test_tma_copy
// CHECK: coir.tma.copy %arg0 to %arg1 {lowered, mechanism = "tma", transfer_bytes = 16384 : i64}
coir.kernel @test_tma_copy(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  %tok = coir.tma.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.wait %tok : !coir.token
}

// Test: combined pipeline -- classify then lower
// PIPE-LABEL: coir.kernel @test_classify_then_lower
// PIPE: coir.tma.copy {{.*}} {lowered, mechanism = "tma"
// PIPE: coir.element.copy {{.*}} {lowered, total_elements
coir.kernel @test_classify_then_lower(
    %g: !coir.tensor<128x64xf16>,
    %s: !coir.tensor<128x64xf16, shared>,
    %s2: !coir.tensor<16x16xf16, shared>,
    %s3: !coir.tensor<16x16xf16, shared>) {
  coir.data.copy %g to %s : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.data.copy %s2 to %s3 : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}

} // module
