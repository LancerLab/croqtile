// RUN: coir-opt --coir-cleanup %s | FileCheck %s

// Test 1: i32 literal constants cast to index fold into index constants.
// The frontend emits integer literals as i32 and casts them to index for
// tensor.tile indices; cleanup folds each pair into one typed constant.

// CHECK-LABEL: coir.kernel @fold_literal_casts
// CHECK-NOT: i32 to index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: coir.tensor.tile %{{.*}}[%[[C0]]]
// CHECK: coir.tensor.tile %{{.*}}[%[[C2]]]
// CHECK-NOT: i32 to index
// CHECK: coir.return
coir.kernel @fold_literal_casts(
    %src: !coir.tensor<128x64xf32, global>,
    %buf0: !coir.tensor<32x64xf32, local>,
    %buf2: !coir.tensor<32x64xf32, local>) {
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %i0 = arith.index_cast %c0_i32 : i32 to index
  %i2 = arith.index_cast %c2_i32 : i32 to index
  %t0 = coir.tensor.tile %src[%i0]
    : !coir.tensor<128x64xf32, global> -> !coir.tensor<32x64xf32, global>
  %t2 = coir.tensor.tile %src[%i2]
    : !coir.tensor<128x64xf32, global> -> !coir.tensor<32x64xf32, global>
  %f0 = coir.dma.copy %t0 to %buf0
    : !coir.tensor<32x64xf32, global> -> !coir.tensor<32x64xf32, local>
  %f2 = coir.dma.copy %t2 to %buf2
    : !coir.tensor<32x64xf32, global> -> !coir.tensor<32x64xf32, local>
  coir.wait %f0 : !coir.async
  coir.wait %f2 : !coir.async
  coir.return
}

// Test 2: duplicate casts of the same value are merged by CSE (the
// frontend re-emits the same cast at every use site). The now-identical
// tiles are merged too; casts of different values stay distinct.

// CHECK-LABEL: coir.kernel @cse_duplicate_casts
// CHECK-NOT: i32 to index
// CHECK: %[[O:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NOT: i32 to index
// CHECK: %[[E:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NOT: i32 to index
// CHECK: %[[T:.*]] = coir.tensor.tile %{{.*}}[%[[O]]]
// CHECK: coir.tensor.tile %{{.*}}[%[[E]]]
// CHECK: coir.dma.copy %[[T]] to
// CHECK: coir.dma.copy %[[T]] to
// CHECK-NOT: i32 to index
// CHECK: coir.return
coir.kernel @cse_duplicate_casts(
    %src: !coir.tensor<128x64xf32, global>,
    %buf1: !coir.tensor<16x64xf32, local>,
    %buf2: !coir.tensor<16x64xf32, local>,
    %buf3: !coir.tensor<16x64xf32, local>,
    %start: i32,
    %end: i32) {
  %o1 = arith.index_cast %start : i32 to index
  %o2 = arith.index_cast %start : i32 to index
  %e1 = arith.index_cast %end : i32 to index
  %t1 = coir.tensor.tile %src[%o1]
    : !coir.tensor<128x64xf32, global> -> !coir.tensor<16x64xf32, global>
  %t2 = coir.tensor.tile %src[%o2]
    : !coir.tensor<128x64xf32, global> -> !coir.tensor<16x64xf32, global>
  %t3 = coir.tensor.tile %src[%e1]
    : !coir.tensor<128x64xf32, global> -> !coir.tensor<16x64xf32, global>
  %f1 = coir.dma.copy %t1 to %buf1
    : !coir.tensor<16x64xf32, global> -> !coir.tensor<16x64xf32, local>
  %f2 = coir.dma.copy %t2 to %buf2
    : !coir.tensor<16x64xf32, global> -> !coir.tensor<16x64xf32, local>
  %f3 = coir.dma.copy %t3 to %buf3
    : !coir.tensor<16x64xf32, global> -> !coir.tensor<16x64xf32, local>
  coir.wait %f1 : !coir.async
  coir.wait %f2 : !coir.async
  coir.wait %f3 : !coir.async
  coir.return
}

// Test 3: identity index_cast round trips (i64 -> index -> i64) from the
// DMA 2^32 size-check collapse back to the original i64 constant, while
// the semantically needed index -> i64 cast of the computed size stays.

// CHECK-LABEL: coir.kernel @fold_index_cast_round_trip
// CHECK-NOT: i64 to index
// CHECK-DAG: arith.constant 4294967296 : i64
// CHECK-DAG: arith.constant 256 : index
// CHECK: arith.cmpi slt
// CHECK-NOT: i64 to index
// CHECK: coir.return
coir.kernel @fold_index_cast_round_trip(%dim: index) {
  %c256 = arith.constant 256 : index
  %bytes = arith.muli %dim, %c256 : index
  %c4294967296 = arith.constant 4294967296 : i64
  %bi = arith.index_cast %c4294967296 : i64 to index
  %bb = arith.index_cast %bi : index to i64
  %sz = arith.index_cast %bytes : index to i64
  %cmp = arith.cmpi slt, %sz, %bb : i64
  coir.assert %cmp, "The size of data transferred by DMA cannot exceed 2^32." <use> <hw_constraint>
  coir.return
}
