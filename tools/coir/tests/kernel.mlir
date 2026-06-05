// RUN: coir-opt %s | FileCheck %s

// Test coir.kernel op
// CHECK-LABEL: coir.kernel @matmul
// CHECK-SAME: %arg0: !coir.tensor<128x64xf16>
// CHECK-SAME: %arg1: !coir.tensor<64x256xf16>
// CHECK: -> !coir.tensor<128x256xf16>
coir.kernel @matmul(%a: !coir.tensor<128x64xf16>,
                    %b: !coir.tensor<64x256xf16>)
    -> !coir.tensor<128x256xf16> {
  // CHECK: coir.tensor.alloc
  %out = coir.tensor.alloc : !coir.tensor<128x256xf16>
  // CHECK: coir.return
  coir.return %out : !coir.tensor<128x256xf16>
}

// Test void kernel
// CHECK-LABEL: coir.kernel @void_kernel
coir.kernel @void_kernel() {
  coir.return
}
