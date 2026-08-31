// RUN: coir-opt --coir-convert-to-gcu %s | FileCheck %s

// A select over CoIR tensors must be rebuilt from the converted memref
// operands.  Retaining the original tensor result type makes a following DMA
// consume a non-memref and crashes the GPU/GCU lowering.

module attributes {coir.target = "gcu", coir.arch = "gcu300"} {
  // CHECK-LABEL: gpu.func @select_buffer_kernel
  // CHECK-SAME: i32
  // CHECK-SAME: memref<16xi32, 1>
  coir.kernel @select_buffer(
      %which: i32,
      %lhs: !coir.tensor<16xi32, global>,
      %rhs: !coir.tensor<16xi32, global>,
      %output: !coir.tensor<16xi32, global>) {
    %zero = arith.constant 0 : i32
    %condition = arith.cmpi eq, %which, %zero : i32
    // CHECK: %[[SELECTED:.*]] = arith.select {{.*}} : memref<16xi32, 1>
    %selected = arith.select %condition, %lhs, %rhs
        : !coir.tensor<16xi32, global>
    %done = coir.dma.copy %selected to %output
        : !coir.tensor<16xi32, global> -> !coir.tensor<16xi32, global>
    coir.wait %done : !coir.async
    coir.return
  }
}
