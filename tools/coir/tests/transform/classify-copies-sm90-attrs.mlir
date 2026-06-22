// Test ClassifyCopies with SM90 module attributes (TMA enabled).
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

// GPU SM90 module: has_tma=true, has_dma=false
module attributes {coir.target = "cute", coir.arch = "sm_90a",
                   coir.has_tma = true, coir.has_dma = false} {

  // global -> shared becomes tma.copy (TMA available)
  // CHECK-LABEL: coir.kernel @test_global_to_shared_sm90
  // CHECK: coir.tma.copy
  // CHECK-NOT: coir.dma.copy
  coir.kernel @test_global_to_shared_sm90(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    coir.data.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  }

  // global -> local stays element.copy (no DMA engine on GPU)
  // CHECK-LABEL: coir.kernel @test_global_to_local_sm90
  // CHECK: coir.element.copy
  // CHECK-NOT: coir.dma.copy
  coir.kernel @test_global_to_local_sm90(
      %src: !coir.tensor<64xi32>,
      %dst: !coir.tensor<64xi32, local>) {
    coir.data.copy %src to %dst
      : !coir.tensor<64xi32> -> !coir.tensor<64xi32, local>
  }
}
