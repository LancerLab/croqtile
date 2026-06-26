// Test ClassifyCopies with SM90 module attributes (TMA enabled).
// Verify tma.copy is accepted when has_tma=true.
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

// GPU SM90 module: has_tma=true, has_dma=false
module attributes {coir.target = "cute", coir.arch = "sm_90a",
                   coir.has_tma = true, coir.has_dma = false} {

  // tma.copy is valid with TMA support
  // CHECK-LABEL: coir.kernel @test_tma_copy_sm90
  // CHECK: coir.tma.copy
  coir.kernel @test_tma_copy_sm90(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    %tok = coir.tma.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
    coir.wait %tok : !coir.async
  }

  // dma.copy passes through unchanged
  // CHECK-LABEL: coir.kernel @test_dma_copy_sm90
  // CHECK: coir.dma.copy
  coir.kernel @test_dma_copy_sm90(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    %tok = coir.dma.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
    coir.wait %tok : !coir.async
  }

  // element.copy passes through unchanged
  // CHECK-LABEL: coir.kernel @test_element_copy_sm90
  // CHECK: coir.element.copy
  coir.kernel @test_element_copy_sm90(
      %src: !coir.tensor<64xi32, local>,
      %dst: !coir.tensor<64xi32, local>) {
    coir.element.copy %src to %dst
      : !coir.tensor<64xi32, local> -> !coir.tensor<64xi32, local>
  }
}
