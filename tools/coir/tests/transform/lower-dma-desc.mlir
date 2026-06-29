// RUN: coir-opt --coir-lower-dma-desc %s | FileCheck %s

// --- Test 1: tma.copy global->shared with tile offsets -> full descriptor pipeline ---

// CHECK-LABEL: coir.kernel @test_tma_copy_desc
coir.kernel @test_tma_copy_desc(
    %src: !coir.tensor<1024xf32, global>,
    %buf: !coir.tensor<64xf32, shared>) {
  %c16 = arith.constant 16 : index

  // CHECK: coir.foreach
  coir.foreach %k in %c16 {
    %tile = coir.tensor.tile %src[%k]
      : !coir.tensor<1024xf32, global> -> !coir.tensor<64xf32, global>
    %tok = coir.tma.copy %tile to %buf
      : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, shared>
    coir.wait %tok : !coir.async

    // CHECK: coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<slice>, tma}
    // CHECK: coir.dma.prefetch.desc
    // CHECK: coir.dma.runtime.desc %{{.*}} offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait
    coir.yield
  }
}

// --- Test 2: dma.copy global->shared without offsets -> descriptor pipeline ---

// CHECK-LABEL: coir.kernel @test_dma_copy_desc
coir.kernel @test_dma_copy_desc(
    %src: !coir.tensor<64xf32, global>,
    %buf: !coir.tensor<64xf32, shared>) {
  // CHECK: coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<copy>}
  // CHECK: coir.dma.prefetch.desc
  // CHECK: coir.dma.invoke
  // CHECK: coir.wait
  %tok = coir.dma.copy %src to %buf
    : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, shared>
  coir.wait %tok : !coir.async
}

// --- Test 3: dma.copy global->shared with tile offsets -> descriptor pipeline ---

// CHECK-LABEL: coir.kernel @test_dma_tiled_copy_desc
coir.kernel @test_dma_tiled_copy_desc(
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

    // CHECK: coir.dma.const.desc %{{.*}}, %{{.*}} {kind = #coir.dma_kind<slice>}
    // CHECK: coir.dma.prefetch.desc
    // CHECK: coir.dma.runtime.desc %{{.*}} offsets(%{{.*}})
    // CHECK: coir.dma.invoke
    // CHECK: coir.wait
    coir.yield
  }
}

// --- Test 4: dma.copy global->local is NOT decomposed (not global<->shared) ---

// CHECK-LABEL: coir.kernel @test_no_decompose_local
coir.kernel @test_no_decompose_local(
    %src: !coir.tensor<64xf32, global>,
    %buf: !coir.tensor<64xf32, local>) {
  // CHECK: coir.dma.copy
  // CHECK-NOT: coir.dma.const.desc
  %tok = coir.dma.copy %src to %buf
    : !coir.tensor<64xf32, global> -> !coir.tensor<64xf32, local>
  coir.wait %tok : !coir.async
}
