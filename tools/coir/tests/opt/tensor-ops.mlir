// RUN: coir-opt %s | FileCheck %s

// Test coir.tensor.alloc
// CHECK-LABEL: coir.kernel @test_tensor_alloc
coir.kernel @test_tensor_alloc() {
  // CHECK: coir.tensor.alloc : !coir.tensor<128x256xf16, default>
  %t = coir.tensor.alloc : !coir.tensor<128x256xf16>
}

// Test coir.tensor.tile
// CHECK-LABEL: coir.kernel @test_tensor_tile
coir.kernel @test_tensor_tile(%a: !coir.tensor<128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: coir.tensor.tile %arg0[%c0, %c1] : !coir.tensor<128x64xf16, default> -> !coir.tensor<16x16xf16, default>
  %tile = coir.tensor.tile %a[%c0, %c1] : !coir.tensor<128x64xf16> -> !coir.tensor<16x16xf16>
}

// Test coir.tensor.store_tile
// CHECK-LABEL: coir.kernel @test_tensor_store
coir.kernel @test_tensor_store(
    %tile: !coir.tensor<16x16xf16>,
    %dest: !coir.tensor<128x64xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: coir.tensor.store_tile %arg0, %arg1[%c0] : !coir.tensor<16x16xf16, default>, !coir.tensor<128x64xf16, default>
  coir.tensor.store_tile %tile, %dest[%c0] : !coir.tensor<16x16xf16>, !coir.tensor<128x64xf16>
}
