// RUN: coir-opt --coir-convert-to-gpu --gpu-kernel-outlining --convert-gpu-to-nvvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-index-to-llvm --reconcile-unrealized-casts %s | FileCheck %s

// Validates that upstream MLIR passes can consume our ConvertToGPU output
// and produce LLVM dialect with NVVM intrinsics.

// CHECK-NOT: coir.kernel
// CHECK-NOT: coir.parallel
// CHECK-NOT: coir.tensor
// CHECK-NOT: gpu.func
// CHECK-NOT: memref.
// CHECK: llvm.func @ele_add_kernel
// CHECK: nvvm.read.ptx.sreg.tid.x
// CHECK: llvm.load
// CHECK: llvm.load
// CHECK: llvm.add
// CHECK: llvm.store
// CHECK: llvm.return

coir.kernel @ele_add(%a: !coir.tensor<64xi32, global>,
                     %b: !coir.tensor<64xi32, global>)
    -> !coir.tensor<64xi32, global> {
  %c = coir.tensor.alloc : !coir.tensor<64xi32, global>
  coir.parallel (%p) in [64] level = #coir.level<thread> {
    %av = coir.tensor.load_elem %a[%p] : !coir.tensor<64xi32, global> -> i32
    %bv = coir.tensor.load_elem %b[%p] : !coir.tensor<64xi32, global> -> i32
    %sum = arith.addi %av, %bv : i32
    coir.tensor.store_elem %sum, %c[%p] : i32, !coir.tensor<64xi32, global>
    coir.yield
  }
  coir.return %c : !coir.tensor<64xi32, global>
}
