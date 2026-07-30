// RUN: coir-opt %s | FileCheck %s

// Test coir.tensor.bind_dims op: parsing, printing, and round-tripping.
// bind_dims binds explicit index SSA values to a tensor's dynamic (`?`)
// dimensions in left-to-right order.  The printed form inlines the SSA
// values directly in the tensor type for readability; the internal
// storage still uses `?` sentinels.
//
// The bind_dims custom format uses `tensor<...>` (no `!coir.` prefix)
// with comma-separated dimensions to avoid ambiguity between SSA value
// names and the `x` dimension separator.

// --- single dynamic dim ------------------------------------------------
// CHECK-LABEL: coir.kernel @bind_dims_single_dyn
coir.kernel @bind_dims_single_dyn(
    %t: !coir.tensor<128x?xf32, default>,
    %n: index) {
  // CHECK: coir.tensor.bind_dims(%{{.*}}) %{{.*}} : tensor<128,%{{.*}},f32, default>
  %bound = coir.tensor.bind_dims(%n) %t : tensor<128,%n,f32, default>
  coir.return
}

// --- multiple dynamic dims ---------------------------------------------
// CHECK-LABEL: coir.kernel @bind_dims_multi_dyn
coir.kernel @bind_dims_multi_dyn(
    %t: !coir.tensor<?x?xi32, shared>,
    %m: index,
    %n: index) {
  // CHECK: coir.tensor.bind_dims(%{{.*}}, %{{.*}}) %{{.*}} : tensor<%{{.*}},%{{.*}},i32, shared>
  %bound = coir.tensor.bind_dims(%m, %n) %t : tensor<%m,%n,i32, shared>
  coir.return
}

// --- downstream use (tensor.tile through bind_dims) --------------------
// CHECK-LABEL: coir.kernel @bind_dims_downstream_use
coir.kernel @bind_dims_downstream_use(
    %t: !coir.tensor<?x?xf16, global>,
    %m: index,
    %n: index) {
  %bound = coir.tensor.bind_dims(%m, %n) %t : tensor<%m,%n,f16, global>
  %c0 = arith.constant 0 : index
  // Note: tensor.tile operand types use the standard !coir.tensor<...> printer
  // which emits ? for dynamic dims.  Only bind_dims inlines SSA refs.
  // CHECK: coir.tensor.bind_dims(%{{.*}}, %{{.*}}) %{{.*}} : tensor<%{{.*}},%{{.*}},f16, global>
  // CHECK: coir.tensor.tile %{{.*}}[%{{.*}}] : !coir.tensor<?x?xf16, global> -> !coir.tensor<16x16xf16, global>
  %tile = coir.tensor.tile %bound[%c0] : !coir.tensor<?x?xf16, global> -> !coir.tensor<16x16xf16, global>
  coir.return
}

// --- type preservation (dynamic dims visible as SSA refs) ---------------
// CHECK-LABEL: coir.kernel @bind_dims_preserves_type
coir.kernel @bind_dims_preserves_type(
    %t: !coir.tensor<64x?x?xf32, default>,
    %m: index,
    %n: index) {
  // CHECK: coir.tensor.bind_dims(%{{.*}}, %{{.*}}) %{{.*}} : tensor<64,%{{.*}},%{{.*}},f32, default>
  %bound = coir.tensor.bind_dims(%m, %n) %t : tensor<64,%m,%n,f32, default>
  %c0 = arith.constant 0 : index
  // CHECK: coir.tensor.tile %{{.*}}[%{{.*}}] : !coir.tensor<64x?x?xf32, default> -> !coir.tensor<16x16xf32, default>
  %tile = coir.tensor.tile %bound[%c0] : !coir.tensor<64x?x?xf32, default> -> !coir.tensor<16x16xf32, default>
  coir.return
}
