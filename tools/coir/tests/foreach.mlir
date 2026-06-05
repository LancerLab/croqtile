// RUN: coir-opt %s | FileCheck %s

// Test coir.foreach op
// CHECK-LABEL: coir.kernel @test_foreach
coir.kernel @test_foreach(%a: !coir.tensor<128x64xf16>) {
  %c4 = arith.constant 4 : index
  // CHECK: coir.foreach %arg1 in %c4
  coir.foreach %k in %c4 {
    // CHECK: coir.yield
    coir.yield
  }
}
