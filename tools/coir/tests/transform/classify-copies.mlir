// Default target-arch is sm_86 (no TMA, no DMA engine):
// RUN: coir-opt --coir-classify-copies %s | FileCheck %s
// SM80 has DMA but no TMA:
// RUN: coir-opt --coir-classify-copies=target-arch=sm_80 %s | FileCheck %s --check-prefix=SM80
// SM90 has both TMA and hardware DMA support:
// RUN: coir-opt --coir-classify-copies=target-arch=sm_90 %s | FileCheck %s --check-prefix=SM90
// Non-GPU targets with DMA engine use dma.copy for global<->local:
// RUN: coir-opt --coir-classify-copies=target-arch=dma_target %s | FileCheck %s --check-prefix=DMA

// Test: global -> shared classification varies by arch
// CHECK-LABEL: coir.kernel @test_global_to_shared
// CHECK: coir.dma.copy
// CHECK-NOT: coir.tma.copy
// SM80-LABEL: coir.kernel @test_global_to_shared
// SM80: coir.dma.copy
// SM80-NOT: coir.tma.copy
// SM90-LABEL: coir.kernel @test_global_to_shared
// SM90: coir.tma.copy
// SM90-NOT: coir.data.copy
// DMA-LABEL: coir.kernel @test_global_to_shared
// DMA: coir.dma.copy
// DMA-NOT: coir.tma.copy
coir.kernel @test_global_to_shared(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  coir.data.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
}

// Test: async global -> shared
// CHECK-LABEL: coir.kernel @test_async_global_to_shared
// CHECK: %[[TOK:.*]] = coir.dma.copy
// CHECK: coir.wait %[[TOK]]
// SM80-LABEL: coir.kernel @test_async_global_to_shared
// SM80: %[[TOK80:.*]] = coir.dma.copy
// SM80: coir.wait %[[TOK80]]
// SM90-LABEL: coir.kernel @test_async_global_to_shared
// SM90: %[[TOK90:.*]] = coir.tma.copy
// SM90: coir.wait %[[TOK90]]
coir.kernel @test_async_global_to_shared(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  %tok = coir.data.copy %src to %dst async : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>, !coir.token
  coir.wait %tok : !coir.token
}

// Test: shared -> shared becomes thread.copy
// CHECK-LABEL: coir.kernel @test_shared_to_shared
// CHECK: coir.thread.copy
// CHECK-NOT: coir.data.copy
coir.kernel @test_shared_to_shared(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, shared>) {
  coir.data.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}

// Test: shared -> local becomes thread.copy (no async needed)
// CHECK-LABEL: coir.kernel @test_shared_to_local
// CHECK: coir.thread.copy
// DMA-LABEL: coir.kernel @test_shared_to_local
// DMA: coir.thread.copy
coir.kernel @test_shared_to_local(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, local>) {
  coir.data.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, local>
}

// Test: global -> local becomes dma.copy on targets with DMA engine
// CHECK-LABEL: coir.kernel @test_global_to_local
// CHECK: coir.thread.copy
// DMA-LABEL: coir.kernel @test_global_to_local
// DMA: coir.dma.copy
// DMA-NOT: coir.thread.copy
coir.kernel @test_global_to_local(
    %src: !coir.tensor<64xi32>,
    %dst: !coir.tensor<64xi32, local>) {
  coir.data.copy %src to %dst : !coir.tensor<64xi32> -> !coir.tensor<64xi32, local>
}

// Test: local -> global becomes dma.copy on targets with DMA engine
// CHECK-LABEL: coir.kernel @test_local_to_global
// CHECK: coir.thread.copy
// DMA-LABEL: coir.kernel @test_local_to_global
// DMA: coir.dma.copy
// DMA-NOT: coir.thread.copy
coir.kernel @test_local_to_global(
    %src: !coir.tensor<64xi32, local>,
    %dst: !coir.tensor<64xi32>) {
  coir.data.copy %src to %dst : !coir.tensor<64xi32, local> -> !coir.tensor<64xi32>
}
