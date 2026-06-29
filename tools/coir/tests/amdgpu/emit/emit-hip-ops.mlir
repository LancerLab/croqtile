// RUN: coir-opt --coir-emit-hip %s | FileCheck %s

// CHECK: #define __CHOREO_TARGET_AMDGPU__
// CHECK: #include "choreo.h"

// --- AtomicOp ---
// CHECK-LABEL: __atomic_test_kernel__
// CHECK: atomicAdd(&arg0[
// CHECK: atomicMin(&arg0[
// CHECK: atomicExch(&arg0[
coir.kernel @atomic_test(%dst: !coir.tensor<64xi32>) {
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %c1 = arith.constant 1 : i32
    coir.atomic <add> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <min> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <exch> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
  }
}

// --- AtomicOp multi-kind ---
// CHECK-LABEL: __atomic_all_kinds_kernel__
// CHECK: atomicSub
// CHECK: atomicMax
// CHECK: atomicAnd
// CHECK: atomicOr
// CHECK: atomicXor
coir.kernel @atomic_all_kinds(%dst: !coir.tensor<64xi32>) {
  coir.parallel (%tid) in [64] level = #coir.level<thread> {
    %c1 = arith.constant 1 : i32
    coir.atomic <sub> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <max> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <and> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <or> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
    coir.atomic <xor> %c1, %dst[%tid] : i32, !coir.tensor<64xi32>
  }
}
