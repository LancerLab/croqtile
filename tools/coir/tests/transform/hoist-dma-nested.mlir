// RUN: coir-opt --coir-hoist-dma-config %s | FileCheck %s

// Test that dma.const.desc and dma.prefetch.desc are hoisted through
// multiple levels of nested foreach loops.

// CHECK-LABEL: coir.kernel @test_nested_hoist
coir.kernel @test_nested_hoist(
    %src: !coir.tensor<1024xf32, global>,
    %dst: !coir.tensor<64xf32, local>) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // The const.desc and prefetch.desc should be hoisted above BOTH loops.
  // CHECK: %[[DESC:.*]] = coir.dma.const.desc
  // CHECK: %[[RT:.*]] = coir.dma.prefetch.desc %[[DESC]]
  // CHECK: coir.foreach
  // CHECK:   coir.foreach
  // CHECK:     coir.dma.runtime.desc
  // CHECK:     coir.dma.invoke

  coir.foreach %i in %c4 {
    coir.foreach %k in %c16 {
      %d0 = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<copy>}
        : !coir.tensor<1024xf32, global>,
          !coir.tensor<64xf32, local> -> !coir.desc
      %d1 = coir.dma.prefetch.desc %d0 : !coir.desc -> !coir.desc.rt
      %d2 = coir.dma.runtime.desc %d1 offsets(%k) : !coir.desc.rt -> !coir.desc.rt
      %t = coir.dma.invoke %d2 : !coir.desc.rt
      coir.wait %t : !coir.async
      coir.yield
    }
    coir.yield
  }
}

// Test partial hoisting: inner-loop IV prevents full hoisting of runtime.desc
// but const.desc escapes both loops; prefetch depends on const.desc so it follows.

// CHECK-LABEL: coir.kernel @test_partial_hoist
coir.kernel @test_partial_hoist(
    %src: !coir.tensor<1024xf32, global>,
    %dst: !coir.tensor<64xf32, local>) {
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index

  // const.desc + prefetch should be above both loops;
  // runtime.desc uses %i so it stays inside the outer loop.
  // CHECK: %[[DESC:.*]] = coir.dma.const.desc
  // CHECK: %[[RT:.*]] = coir.dma.prefetch.desc %[[DESC]]
  // CHECK: coir.foreach
  // CHECK:   coir.dma.runtime.desc
  // CHECK:   coir.foreach
  // CHECK:     coir.dma.runtime.desc
  // CHECK:     coir.dma.invoke

  coir.foreach %i in %c4 {
    %d0 = coir.dma.const.desc %src, %dst {kind = #coir.dma_kind<slice>}
      : !coir.tensor<1024xf32, global>,
        !coir.tensor<64xf32, local> -> !coir.desc
    %d1 = coir.dma.prefetch.desc %d0 : !coir.desc -> !coir.desc.rt
    %d1a = coir.dma.runtime.desc %d1 offsets(%i) : !coir.desc.rt -> !coir.desc.rt
    coir.foreach %k in %c16 {
      %d2 = coir.dma.runtime.desc %d1a offsets(%k) : !coir.desc.rt -> !coir.desc.rt
      %t = coir.dma.invoke %d2 : !coir.desc.rt
      coir.wait %t : !coir.async
      coir.yield
    }
    coir.yield
  }
}
