// ClassifyCopies now validates TMA usage only (DataCopyOp was removed).
// Verify that dma.copy/element.copy pass through unchanged.
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

// CHECK-LABEL: coir.kernel @test_dma_copy_passthrough
// CHECK: coir.dma.copy
coir.kernel @test_dma_copy_passthrough(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  %tok = coir.dma.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.wait %tok : !coir.async
}

// CHECK-LABEL: coir.kernel @test_element_copy_passthrough
// CHECK: coir.element.copy
coir.kernel @test_element_copy_passthrough(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, shared>) {
  coir.element.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}
