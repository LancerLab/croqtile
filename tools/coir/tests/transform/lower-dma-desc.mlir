// RUN: coir-opt --coir-lower-dma-desc %s | FileCheck %s

// --- Test 1: data.copy path (standalone, no ClassifyCopies) ---

// CHECK-LABEL: coir.kernel @test_data_copy_desc
coir.kernel @test_data_copy_desc(
    %src: !coir.tensor<1024xf32, global>,
    %buf: !coir.tensor<64xf32, local>) {
  %c16 = arith.constant 16 : index

  // CHECK: %[[DESC1:.*]] = coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<slice>}
  // CHECK-SAME: -> !coir.desc
  // CHECK: %[[RT1:.*]] = coir.dma.prefetch.desc %[[DESC1]]
  // CHECK-SAME: -> !coir.desc.rt

  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %tile = coir.tensor.tile %src[%k]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
    coir.data.copy %tile to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>

    // CHECK: coir.dma.runtime.desc %[[RT1]] offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait
    coir.yield
  }
}

// --- Test 2: dma.copy path (post-ClassifyCopies, real pipeline) ---

// CHECK-LABEL: coir.kernel @test_dma_copy_desc
coir.kernel @test_dma_copy_desc(
    %src: !coir.tensor<1024xf32, global>,
    %buf: !coir.tensor<64xf32, local>) {
  %c16 = arith.constant 16 : index

  // CHECK: %[[DESC2:.*]] = coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<slice>}
  // CHECK-SAME: -> !coir.desc
  // CHECK: %[[RT2:.*]] = coir.dma.prefetch.desc %[[DESC2]]
  // CHECK-SAME: -> !coir.desc.rt

  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %tile = coir.tensor.tile %src[%k]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
    %tok = coir.dma.copy %tile to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>
    coir.wait %tok : !coir.token

    // CHECK: coir.dma.runtime.desc %[[RT2]] offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait
    coir.yield
  }
}

// --- Test 3: dma.copy without IV dependency is not decomposed ---

// CHECK-LABEL: coir.kernel @test_no_iv_dep
coir.kernel @test_no_iv_dep(
    %src: !coir.tensor<64xf32, global>,
    %buf: !coir.tensor<64xf32, local>) {
  %c16 = arith.constant 16 : index
  coir.foreach %k in %c16 {
    // Source does not depend on IV -- should stay as dma.copy
    // CHECK: coir.dma.copy
    // CHECK-NOT: coir.dma.const.desc
    %tok = coir.dma.copy %src to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>
    coir.wait %tok : !coir.token
    coir.yield
  }
}
