// RUN: coir-opt --coir-classify-copies --coir-lower-copy --coir-emit-hip %s | FileCheck %s

module attributes { "coir.has_tma" = false, "coir.has_dma" = false, "coir.mma_target" = "" } {

// Foreach loop emission
// CHECK: __global__ void __foreach_kernel__
// CHECK: for (int
coir.kernel @foreach(%a: !coir.tensor<256xf32>, %b: !coir.tensor<256xf32>) {
  %c4 = arith.constant 4 : index
  coir.foreach %i in %c4 {
    coir.parallel (%tid) in [64] level = #coir.level<thread> {
      %idx = arith.addi %i, %tid : index
      %va = coir.tensor.load_elem %a[%idx] : !coir.tensor<256xf32> -> f32
      coir.tensor.store_elem %va, %b[%idx] : f32, !coir.tensor<256xf32>
    }
    coir.yield
  }
}

// If-else emission
// CHECK: __global__ void __ifelse_kernel__
// CHECK: if (
// CHECK: else
coir.kernel @ifelse(%a: !coir.tensor<64xf32>, %b: !coir.tensor<64xf32>) {
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %va = coir.tensor.load_elem %a[%tid] : !coir.tensor<64xf32> -> f32
    %zero = arith.constant 0.0 : f32
    %cond = arith.cmpf ogt, %va, %zero : f32
    %result = scf.if %cond -> (f32) {
      %two = arith.constant 2.0 : f32
      %r = arith.mulf %va, %two : f32
      scf.yield %r : f32
    } else {
      %neg = arith.constant -1.0 : f32
      %r2 = arith.mulf %va, %neg : f32
      scf.yield %r2 : f32
    }
    coir.tensor.store_elem %result, %b[%tid] : f32, !coir.tensor<64xf32>
  }
}

// Atomic reduce emission
// CHECK: __global__ void __reduce_kernel__
// CHECK: atomicAdd
coir.kernel @reduce(%a: !coir.tensor<256xf32>, %out: !coir.tensor<1xf32>) {
  coir.parallel (%tid) in [256] level = #coir.level<thread> {
    %va = coir.tensor.load_elem %a[%tid] : !coir.tensor<256xf32> -> f32
    %c0 = arith.constant 0 : index
    coir.tensor.reduce_elem %va, %out[%c0] {atomic} : f32, !coir.tensor<1xf32>
  }
}

} // module
