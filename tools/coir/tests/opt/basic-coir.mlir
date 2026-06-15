// RUN: coir-opt %s | FileCheck %s

// Test the DMA descriptor pipeline with target-neutral tensor-based ops.
// CHECK-LABEL: coir.kernel @test_dma_desc
coir.kernel @test_dma_desc(
    %src: !coir.tensor<128x128xf16, global>,
    %dst: !coir.tensor<128x128xf16, local>) {
  // CHECK: coir.dma.const.desc
  %d0 = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
    : !coir.tensor<128x128xf16, global>,
      !coir.tensor<128x128xf16, local> -> !coir.desc
  // CHECK: coir.dma.check
  coir.dma.check %d0 : !coir.desc
  // CHECK: coir.dma.prefetch.desc
  %d1 = coir.dma.prefetch.desc %d0 : !coir.desc -> !coir.desc.rt
  // CHECK: coir.dma.runtime.desc
  %c0 = arith.constant 0 : index
  %d2 = coir.dma.runtime.desc %d1 offsets(%c0) : !coir.desc.rt -> !coir.desc.rt
  // CHECK: coir.dma.invoke
  %t = coir.dma.invoke %d2 : !coir.desc.rt
  coir.wait %t : !coir.token
}
