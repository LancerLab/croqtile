// RUN: coir-opt %s | FileCheck %s

// Test coir.data.copy (sync)
// CHECK-LABEL: coir.kernel @test_data_copy_sync
coir.kernel @test_data_copy_sync(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  // CHECK: coir.data.copy %arg0 to %arg1 : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.data.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
}

// Test coir.data.copy (async)
// CHECK-LABEL: coir.kernel @test_data_copy_async
coir.kernel @test_data_copy_async(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  // CHECK: coir.data.copy %arg0 to %arg1 async : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>, !coir.token
  %tok = coir.data.copy %src to %dst async : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>, !coir.token
  // CHECK: coir.wait
  coir.wait %tok : !coir.token
}

// Test coir.dma.copy
// CHECK-LABEL: coir.kernel @test_dma_copy
coir.kernel @test_dma_copy(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  // CHECK: coir.dma.copy %arg0 to %arg1 : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  %tok = coir.dma.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.wait %tok : !coir.token
}

// Test coir.tma.copy
// CHECK-LABEL: coir.kernel @test_tma_copy
coir.kernel @test_tma_copy(
    %src: !coir.tensor<128x64xf16>,
    %dst: !coir.tensor<128x64xf16, shared>) {
  // CHECK: coir.tma.copy %arg0 to %arg1 : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  %tok = coir.tma.copy %src to %dst : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  coir.wait %tok : !coir.token
}

// Test coir.thread.copy
// CHECK-LABEL: coir.kernel @test_thread_copy
coir.kernel @test_thread_copy(
    %src: !coir.tensor<16x16xf16, shared>,
    %dst: !coir.tensor<16x16xf16, shared>) {
  // CHECK: coir.thread.copy %arg0 to %arg1 : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
  coir.thread.copy %src to %dst : !coir.tensor<16x16xf16, shared> -> !coir.tensor<16x16xf16, shared>
}

// Test coir.barrier
// CHECK-LABEL: coir.kernel @test_barrier
coir.kernel @test_barrier() {
  // CHECK: coir.barrier <block>
  coir.barrier #coir.level<block>
  // CHECK: coir.barrier <group>
  coir.barrier #coir.level<group>
}

// Test combined copy + barrier + MMA pipeline
// CHECK-LABEL: coir.kernel @test_copy_mma_pipeline
coir.kernel @test_copy_mma_pipeline(
    %ga: !coir.tensor<128x64xf16>,
    %gb: !coir.tensor<64x128xf16>,
    %gc: !coir.tensor<128x128xf32>) {
  %sa = coir.tensor.alloc : !coir.tensor<128x64xf16, shared>
  %sb = coir.tensor.alloc : !coir.tensor<64x128xf16, shared>

  // CHECK: coir.dma.copy
  %tok_a = coir.dma.copy %ga to %sa : !coir.tensor<128x64xf16> -> !coir.tensor<128x64xf16, shared>
  // CHECK: coir.dma.copy
  %tok_b = coir.dma.copy %gb to %sb : !coir.tensor<64x128xf16> -> !coir.tensor<64x128xf16, shared>
  coir.wait %tok_a : !coir.token
  coir.wait %tok_b : !coir.token
  // CHECK: coir.barrier
  coir.barrier #coir.level<block>

  %a_tile = coir.tensor.tile %sa[] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<16x16xf16, shared>
  %b_tile = coir.tensor.tile %sb[] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x16xf16, shared>
  %zero = arith.constant 0.0 : f32
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  // CHECK: coir.mma.exec
  %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
}
