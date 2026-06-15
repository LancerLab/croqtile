// RUN: coir-opt --coir-lower-dma-desc %s | FileCheck %s

// Test that data.copy inside a foreach with IV-dependent source (via tensor.tile)
// is decomposed into the DMA descriptor pipeline.

// CHECK-LABEL: coir.kernel @test_lower_dma_desc
coir.kernel @test_lower_dma_desc(
    %src: !coir.tensor<1024xf32, global>,
    %buf: !coir.tensor<64xf32, local>) {
  %c16 = arith.constant 16 : index

  // CHECK: %[[DESC:.*]] = coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<slice>}
  // CHECK-SAME: -> !coir.desc
  // CHECK: %[[RT:.*]] = coir.dma.prefetch.desc %[[DESC]]
  // CHECK-SAME: -> !coir.desc.rt

  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %tile = coir.tensor.tile %src[%k]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
    coir.data.copy %tile to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>

    // CHECK: coir.dma.runtime.desc %[[RT]] offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait
    coir.yield
  }
}
