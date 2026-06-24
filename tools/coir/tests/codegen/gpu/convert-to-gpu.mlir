// RUN: coir-opt --coir-convert-to-gpu %s | FileCheck %s

// ---- Test 1: Simple elementwise add with thread-level parallel ----
// CHECK-LABEL: gpu.module @ele_add_module
// CHECK:   gpu.func @ele_add_kernel
// CHECK-SAME: memref<64xi32>
// CHECK-SAME: memref<64xi32>
// CHECK-SAME: memref<64xi32>
// CHECK:     %[[TID:.*]] = gpu.thread_id x
// CHECK:     memref.load
// CHECK:     memref.load
// CHECK:     arith.addi
// CHECK:     memref.store
// CHECK:     gpu.return
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

// ---- Test 2: Void kernel, no args (nil pattern) ----
// CHECK-LABEL: gpu.module @nil_module
// CHECK:   gpu.func @nil_kernel()
// CHECK:     gpu.return
coir.kernel @nil() {
  coir.return
}

// ---- Test 3: Float types ----
// CHECK-LABEL: gpu.module @float_add_module
// CHECK:   gpu.func @float_add_kernel
// CHECK-SAME: memref<64xf32>
// CHECK:     gpu.thread_id x
// CHECK:     arith.addf
// CHECK:     gpu.return
coir.kernel @float_add(%a: !coir.tensor<64xf32, global>,
                       %b: !coir.tensor<64xf32, global>)
    -> !coir.tensor<64xf32, global> {
  %c = coir.tensor.alloc : !coir.tensor<64xf32, global>
  coir.parallel (%p) in [64] level = #coir.level<thread> {
    %av = coir.tensor.load_elem %a[%p] : !coir.tensor<64xf32, global> -> f32
    %bv = coir.tensor.load_elem %b[%p] : !coir.tensor<64xf32, global> -> f32
    %sum = arith.addf %av, %bv : f32
    coir.tensor.store_elem %sum, %c[%p] : f32, !coir.tensor<64xf32, global>
    coir.yield
  }
  coir.return %c : !coir.tensor<64xf32, global>
}

// ---- Test 4: 2D parallel (block + thread nested) ----
// CHECK-LABEL: gpu.module @add_2d_module
// CHECK:   gpu.func @add_2d_kernel
// CHECK-DAG:  gpu.block_id x
// CHECK-DAG:  gpu.block_id y
// CHECK-DAG:  gpu.thread_id x
// CHECK:      gpu.return
coir.kernel @add_2d(%a: !coir.tensor<256x128xi32, global>,
                    %b: !coir.tensor<256x128xi32, global>)
    -> !coir.tensor<256x128xi32, global> {
  %c = coir.tensor.alloc : !coir.tensor<256x128xi32, global>
  coir.parallel (%bi, %bj) in [4, 2] level = #coir.level<block> {
    coir.parallel (%ti) in [128] level = #coir.level<thread> {
      coir.yield
    }
    coir.yield
  }
  coir.return %c : !coir.tensor<256x128xi32, global>
}
