// RUN: coir-opt --coir-cleanup %s | FileCheck %s

func.func @reload_after_store(%buffer: !coir.tensor<4xi32, global>,
                              %index: index, %value: i32) -> (i32, i32) {
  %before = coir.tensor.load_elem %buffer[%index] : !coir.tensor<4xi32, global> -> i32
  coir.tensor.store_elem %value, %buffer[%index] : i32, !coir.tensor<4xi32, global>
  %after = coir.tensor.load_elem %buffer[%index] : !coir.tensor<4xi32, global> -> i32
  return %before, %after : i32, i32
}

// CHECK-LABEL: func.func @reload_after_store
// CHECK: [[BEFORE:%[a-zA-Z0-9_]+]] = coir.tensor.load_elem
// CHECK: coir.tensor.store_elem
// CHECK: [[AFTER:%[a-zA-Z0-9_]+]] = coir.tensor.load_elem
// CHECK: return [[BEFORE]], [[AFTER]] : i32, i32
