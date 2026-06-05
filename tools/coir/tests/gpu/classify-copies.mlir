// RUN: coir-opt --coir-classify-copies %s | FileCheck %s
// RUN: coir-opt --coir-classify-copies=target-arch=sm_80 %s | FileCheck %s --check-prefix=SM80

// Test: global -> shared becomes tma.copy on SM90
// CHECK-LABEL: coir.kernel @test_global_to_shared
// CHECK: coir.tma.copy
// CHECK-NOT: coir.data.copy
// SM80-LABEL: coir.kernel @test_global_to_shared
// SM80: coir.dma.copy
// SM80-NOT: coir.tma.copy
coir.kernel @test_global_to_shared(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  coir.data.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
}

// Test: async global -> shared becomes tma.copy with token on SM90
// CHECK-LABEL: coir.kernel @test_async_global_to_shared
// CHECK: %[[TOK:.*]] = coir.tma.copy
// CHECK: coir.wait %[[TOK]]
// SM80-LABEL: coir.kernel @test_async_global_to_shared
// SM80: %[[TOK:.*]] = coir.dma.copy
// SM80: coir.wait %[[TOK]]
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

// Test: shared -> register becomes thread.copy
// CHECK-LABEL: coir.kernel @test_shared_to_local
// CHECK: coir.thread.copy
coir.kernel @test_shared_to_local(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, local>) {
  coir.data.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, local>
}
