// End-to-end DMA pipeline test for GCU target.
// Runs classify-copies -> lower-dma-desc -> hoist-dma-config in sequence,
// verifying that dma.copy ops inside loops get decomposed into descriptors.
//
// RUN: coir-opt --coir-classify-copies --coir-lower-dma-desc --coir-hoist-dma-config %s | FileCheck %s

module attributes {coir.target = "topscc", coir.arch = "gcu300",
                   coir.has_tma = false, coir.has_dma = true} {

  // data.copy inside foreach -> classified as dma.copy -> decomposed to desc pipeline
  // CHECK-LABEL: coir.kernel @test_gcu_dma_pipeline
  coir.kernel @test_gcu_dma_pipeline(
      %src: !coir.tensor<1024xf32, global>,
      %buf: !coir.tensor<64xf32, local>) {
    %c16 = arith.constant 16 : index

    // CHECK: %[[DESC:.*]] = coir.dma.const.desc
    // CHECK: %[[RT:.*]] = coir.dma.prefetch.desc %[[DESC]]
    // CHECK: coir.foreach
    coir.foreach %k in %c16 {
      %tile = coir.tensor.tile %src[%k]
        : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
      coir.data.copy %tile to %buf
        : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>

      // CHECK: coir.dma.runtime.desc %[[RT]] offsets(%{{.*}})
      // CHECK: coir.dma.invoke
      coir.yield
    }
  }

  // data.copy outside foreach stays as dma.copy (no decomposition needed)
  // CHECK-LABEL: coir.kernel @test_gcu_no_loop
  coir.kernel @test_gcu_no_loop(
      %src: !coir.tensor<64xf32, global>,
      %buf: !coir.tensor<64xf32, local>) {
    // CHECK: coir.dma.copy
    // CHECK-NOT: coir.dma.const.desc
    coir.data.copy %src to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>
  }
}
