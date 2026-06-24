// Default (no attrs): no TMA, no DMA -- global->shared becomes dma.copy,
// global->local becomes element.copy, shared->local becomes element.copy
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s

// CHECK-LABEL: coir.kernel @test_global_to_shared
// CHECK: coir.dma.copy
// CHECK-NOT: coir.tma.copy
coir.kernel @test_global_to_shared(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  coir.data.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
}

// CHECK-LABEL: coir.kernel @test_async_global_to_shared
// CHECK: %[[TOK:.*]] = coir.dma.copy
// CHECK: coir.wait %[[TOK]]
coir.kernel @test_async_global_to_shared(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  %tok = coir.data.copy %src to %dst async : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>, !coir.async
  coir.wait %tok : !coir.async
}

// shared -> shared becomes element.copy
// CHECK-LABEL: coir.kernel @test_shared_to_shared
// CHECK: coir.element.copy
// CHECK-NOT: coir.data.copy
coir.kernel @test_shared_to_shared(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, shared>) {
  coir.data.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}

// shared -> local becomes element.copy (no DMA engine by default)
// CHECK-LABEL: coir.kernel @test_shared_to_local
// CHECK: coir.element.copy
coir.kernel @test_shared_to_local(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, local>) {
  coir.data.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, local>
}

// global -> local becomes element.copy (no DMA engine by default)
// CHECK-LABEL: coir.kernel @test_global_to_local
// CHECK: coir.element.copy
coir.kernel @test_global_to_local(
    %src: !coir.tensor<64xi32>,
    %dst: !coir.tensor<64xi32, local>) {
  coir.data.copy %src to %dst : !coir.tensor<64xi32> -> !coir.tensor<64xi32, local>
}

// local -> global becomes element.copy (no DMA engine by default)
// CHECK-LABEL: coir.kernel @test_local_to_global
// CHECK: coir.element.copy
coir.kernel @test_local_to_global(
    %src: !coir.tensor<64xi32, local>,
    %dst: !coir.tensor<64xi32>) {
  coir.data.copy %src to %dst : !coir.tensor<64xi32, local> -> !coir.tensor<64xi32>
}
