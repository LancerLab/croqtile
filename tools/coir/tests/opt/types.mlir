// RUN: coir-opt %s | FileCheck %s

// Test coir.tensor type parsing and printing
// CHECK-LABEL: func.func @test_tensor_types
func.func @test_tensor_types(
    %a: !coir.tensor<128x64xf16>,
    %b: !coir.tensor<16x16xf32, shared>,
    %c: !coir.tensor<256x256xf16, global>) {
  // CHECK: !coir.tensor<128x64xf16, default>
  // CHECK: !coir.tensor<16x16xf32, shared>
  // CHECK: !coir.tensor<256x256xf16, global>
  func.return
}

// Test coir.mma_frag type parsing and printing
// CHECK-LABEL: func.func @test_frag_types
func.func @test_frag_types(
    %a: !coir.mma_frag<16x16xf16>,
    %b: !coir.mma_frag<8x8xf32>) {
  // CHECK: !coir.mma_frag<16x16xf16>
  // CHECK: !coir.mma_frag<8x8xf32>
  func.return
}

// Test coir.token type
// CHECK-LABEL: func.func @test_token_type
func.func @test_token_type(%t: !coir.async) {
  // CHECK: !coir.async
  func.return
}
