// Test ClassifyCopies driven by module attributes (the target interface path).
// Module attrs are set by the driver from Target::HasTMA()/HasDMA().
//
// GCU target: has_dma=true, has_tma=false
// Verify dma.copy and element.copy pass through; tma.copy would be rejected.
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

module attributes {coir.target = "topscc", coir.arch = "gcu300",
                   coir.has_tma = false, coir.has_dma = true} {

  // dma.copy passes through unchanged
  // CHECK-LABEL: coir.kernel @test_dma_copy_gcu
  // CHECK: coir.dma.copy
  coir.kernel @test_dma_copy_gcu(
      %src: !coir.tensor<64xi32>,
      %dst: !coir.tensor<64xi32, local>) {
    %tok = coir.dma.copy %src to %dst
      : !coir.tensor<64xi32> -> !coir.tensor<64xi32, local>
    coir.wait %tok : !coir.async
  }

  // element.copy passes through unchanged
  // CHECK-LABEL: coir.kernel @test_element_copy_gcu
  // CHECK: coir.element.copy
  coir.kernel @test_element_copy_gcu(
      %src: !coir.tensor<16x16xf16, shared>,
      %dst: !coir.tensor<16x16xf16, local>) {
    coir.element.copy %src to %dst
      : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, local>
  }

  // dma.copy global -> shared
  // CHECK-LABEL: coir.kernel @test_global_to_shared_gcu
  // CHECK: coir.dma.copy
  coir.kernel @test_global_to_shared_gcu(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    %tok = coir.dma.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
    coir.wait %tok : !coir.async
  }
}
