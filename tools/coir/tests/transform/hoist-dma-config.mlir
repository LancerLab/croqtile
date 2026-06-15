// RUN: coir-opt --coir-hoist-dma-config %s | FileCheck %s

// Test that dma.const.desc and dma.prefetch.desc are hoisted above foreach
// when their operands are loop-invariant.

// CHECK-LABEL: coir.kernel @test_hoist
coir.kernel @test_hoist(
    %src: !coir.tensor<1024xf32, global>,
    %dst: !coir.tensor<64xf32, local>) {
  %c16 = arith.constant 16 : index

  // CHECK: %[[DESC:.*]] = coir.dma.const.desc
  // CHECK: %[[RT:.*]] = coir.dma.prefetch.desc %[[DESC]]
  // CHECK: coir.foreach

  coir.foreach %k in %c16 {
    %d0 = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
      : !coir.tensor<1024xf32, global>,
        !coir.tensor<64xf32, local> -> !coir.desc
    %d1 = coir.dma.prefetch.desc %d0 : !coir.desc -> !coir.desc.rt
    %d2 = coir.dma.runtime.desc %d1 offsets(%k) : !coir.desc.rt -> !coir.desc.rt
    %t = coir.dma.invoke %d2 : !coir.desc.rt
    coir.wait %t : !coir.token
    coir.yield
  }
}
