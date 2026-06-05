// RUN: coir-opt %s | FileCheck %s

// Test coir.mma.fill
// CHECK-LABEL: coir.kernel @test_mma_fill
coir.kernel @test_mma_fill() {
  %zero = arith.constant 0.0 : f32
  // CHECK: coir.mma.fill %cst : f32 -> !coir.mma_frag<16x16xf32>
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
}

// Test coir.mma.load
// CHECK-LABEL: coir.kernel @test_mma_load
coir.kernel @test_mma_load(%tile: !coir.tensor<16x16xf16, shared>) {
  // CHECK: coir.mma.load %arg0 : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  %frag = coir.mma.load %tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
}

// Test coir.mma.exec
// CHECK-LABEL: coir.kernel @test_mma_exec
coir.kernel @test_mma_exec(
    %acc: !coir.mma_frag<16x16xf32>,
    %a: !coir.mma_frag<16x16xf16>,
    %b: !coir.mma_frag<16x16xf16>) {
  // CHECK: coir.mma.exec %arg0, %arg1, %arg2 {layout = #coir.mma_layout<row_col>}
  // CHECK-SAME: (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  %res = coir.mma.exec %acc, %a, %b {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
}

// Test coir.mma.store
// CHECK-LABEL: coir.kernel @test_mma_store
coir.kernel @test_mma_store(
    %frag: !coir.mma_frag<16x16xf32>,
    %dest: !coir.tensor<16x16xf32, shared>) {
  // CHECK: coir.mma.store %arg0, %arg1 : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
  coir.mma.store %frag, %dest : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// Test full MMA pipeline: fill -> load -> exec -> store
// CHECK-LABEL: coir.kernel @test_mma_pipeline
coir.kernel @test_mma_pipeline(
    %a_tile: !coir.tensor<16x16xf16, shared>,
    %b_tile: !coir.tensor<16x16xf16, shared>,
    %c_tile: !coir.tensor<16x16xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  // CHECK: coir.mma.fill
  %acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  // CHECK: coir.mma.load
  %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  // CHECK: coir.mma.load
  %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
  // CHECK: coir.mma.exec
  %res = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
  // CHECK: coir.mma.store
  coir.mma.store %res, %c_tile : !coir.mma_frag<16x16xf32>, !coir.tensor<16x16xf32, shared>
}

// Test MMA with foreach (accumulation loop)
// CHECK-LABEL: coir.kernel @test_mma_accumulate
coir.kernel @test_mma_accumulate(
    %a: !coir.tensor<128x64xf16, shared>,
    %b: !coir.tensor<64x128xf16, shared>,
    %c: !coir.tensor<128x128xf32, shared>) {
  %zero = arith.constant 0.0 : f32
  %c4 = arith.constant 4 : index
  // CHECK: coir.mma.fill
  %init_acc = coir.mma.fill %zero : f32 -> !coir.mma_frag<16x16xf32>
  // CHECK: coir.foreach
  %final_acc = coir.foreach %k in %c4 iter_args(%acc = %init_acc) : !coir.mma_frag<16x16xf32> {
    %a_tile = coir.tensor.tile %a[%k] : !coir.tensor<128x64xf16, shared> -> !coir.tensor<16x16xf16, shared>
    %b_tile = coir.tensor.tile %b[%k] : !coir.tensor<64x128xf16, shared> -> !coir.tensor<16x16xf16, shared>
    %a_frag = coir.mma.load %a_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    %b_frag = coir.mma.load %b_tile : !coir.tensor<16x16xf16, shared> -> !coir.mma_frag<16x16xf16>
    // CHECK: coir.mma.exec
    %new_acc = coir.mma.exec %acc, %a_frag, %b_frag {layout = #coir.mma_layout<row_col>} : (!coir.mma_frag<16x16xf32>, !coir.mma_frag<16x16xf16>, !coir.mma_frag<16x16xf16>) -> !coir.mma_frag<16x16xf32>
    coir.yield %new_acc : !coir.mma_frag<16x16xf32>
  }
  // CHECK: coir.mma.store
  coir.mma.store %final_acc, %c : !coir.mma_frag<16x16xf32>, !coir.tensor<128x128xf32, shared>
}
