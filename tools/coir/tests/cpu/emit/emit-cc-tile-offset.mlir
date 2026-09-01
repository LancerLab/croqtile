// RUN: coir-opt --coir-emit-cc %s | FileCheck %s

// Test coir.tensor.tile emission on the CPU target:
//   - element_offset tile: leading indices are pre-multiplied element offsets,
//     summed as a flat offset (no per-dim stride scaling).
//   - plain tile: raw chunk indices, scaled by their row stride.

// CHECK-LABEL: void __tile_offset_impl__(
coir.kernel @tile_offset(%src: !coir.tensor<4x64xi32>) -> !coir.tensor<4x64xi32> {

  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index

  // element_offset: base + (5 + 0), NOT base + (5 * 64 + 0).
  // CHECK: auto [[EO:v[0-9]+]] = {{.*}} + ([[A:v[0-9]+]] + [[B:v[0-9]+]]);
  %tile = coir.tensor.tile %src[%c5, %c0] {coir.element_offset} : !coir.tensor<4x64xi32> -> !coir.tensor<1x64xi32>
  %v = coir.tensor.load_elem %tile[%c0] : !coir.tensor<1x64xi32> -> i32
  coir.tensor.store_elem %v, %src[%c5, %c0] : i32, !coir.tensor<4x64xi32>

  // plain: base + (7 * 64).
  // CHECK: auto [[PL:v[0-9]+]] = {{.*}} + ([[C:v[0-9]+]] * 64);
  %plain = coir.tensor.tile %src[%c7] : !coir.tensor<4x64xi32> -> !coir.tensor<1xi32>
  %p = coir.tensor.load_elem %plain[] : !coir.tensor<1xi32> -> i32
  coir.tensor.store_elem %p, %src[%c5, %c7] : i32, !coir.tensor<4x64xi32>

  coir.return %src : !coir.tensor<4x64xi32>
}
