// End-to-end DMA pipeline test for GCU target.
// Runs classify-copies -> lower-dma-desc -> hoist-dma-config in sequence.
//
// RUN: coir-opt --coir-classify-copies --coir-lower-dma-desc --coir-hoist-dma-config %s | FileCheck %s

module attributes {coir.target = "topscc", coir.arch = "gcu300",
                   coir.has_tma = false, coir.has_dma = true} {

  // dma.copy global->shared inside foreach
  // CHECK-LABEL: coir.kernel @test_gcu_dma_pipeline
  coir.kernel @test_gcu_dma_pipeline(
      %src: !coir.tensor<1024xf32, global>,
      %buf: !coir.tensor<64xf32, shared>) {
    %c16 = arith.constant 16 : index

    // CHECK: coir.foreach
    coir.foreach %k in %c16 {
      %tile = coir.tensor.tile %src[%k]
        : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
      %tok = coir.dma.copy %tile to %buf
        : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, shared>
      coir.wait %tok : !coir.async

      // CHECK: coir.dma.copy
      coir.yield
    }
  }

  // dma.copy global->local outside foreach
  // CHECK-LABEL: coir.kernel @test_gcu_no_loop
  coir.kernel @test_gcu_no_loop(
      %src: !coir.tensor<64xf32, global>,
      %buf: !coir.tensor<64xf32, local>) {
    // CHECK: coir.dma.copy
    // CHECK-NOT: coir.dma.const.desc
    %tok = coir.dma.copy %src to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>
    coir.wait %tok : !coir.async
  }
}
