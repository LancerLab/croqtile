// Test ClassifyCopies driven by module attributes (the target interface path).
// Module attrs are set by the driver from Target::HasTMA()/HasDMA().
//
// GCU target: has_dma=true, has_tma=false
// DMA engine handles global<->shared, global<->local, AND shared<->local.
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

module attributes {coir.target = "topscc", coir.arch = "gcu300",
                   coir.has_tma = false, coir.has_dma = true} {

  // global -> local becomes dma.copy (DMA engine available)
  // CHECK-LABEL: coir.kernel @test_global_to_local_gcu
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.element.copy
  coir.kernel @test_global_to_local_gcu(
      %src: !coir.tensor<64xi32>,
      %dst: !coir.tensor<64xi32, local>) {
    coir.data.copy %src to %dst
      : !coir.tensor<64xi32> -> !coir.tensor<64xi32, local>
  }

  // local -> global becomes dma.copy
  // CHECK-LABEL: coir.kernel @test_local_to_global_gcu
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.element.copy
  coir.kernel @test_local_to_global_gcu(
      %src: !coir.tensor<64xi32, local>,
      %dst: !coir.tensor<64xi32>) {
    coir.data.copy %src to %dst
      : !coir.tensor<64xi32, local> -> !coir.tensor<64xi32>
  }

  // global -> shared still becomes dma.copy (no TMA on GCU)
  // CHECK-LABEL: coir.kernel @test_global_to_shared_gcu
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.tma.copy
  coir.kernel @test_global_to_shared_gcu(
      %src: !coir.tensor<128x64xf16>,
      %dst: !coir.tensor<128x64xf16, shared>) {
    coir.data.copy %src to %dst
      : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  }

  // shared -> local becomes dma.copy (DMA engine handles shared<->local)
  // CHECK-LABEL: coir.kernel @test_shared_to_local_gcu
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.element.copy
  coir.kernel @test_shared_to_local_gcu(
      %src: !coir.tensor<16x16xf16, shared>,
      %dst: !coir.tensor<16x16xf16, local>) {
    coir.data.copy %src to %dst
      : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, local>
  }

  // local -> shared becomes dma.copy (DMA engine handles shared<->local)
  // CHECK-LABEL: coir.kernel @test_local_to_shared_gcu
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.element.copy
  coir.kernel @test_local_to_shared_gcu(
      %src: !coir.tensor<16x16xf16, local>,
      %dst: !coir.tensor<16x16xf16, shared>) {
    coir.data.copy %src to %dst
      : !coir.tensor<16x16xf16, local> -> !coir.tensor<16x16xf16, shared>
  }
}
