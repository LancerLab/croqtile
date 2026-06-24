// RUN: coir-opt --coir-emit-cuda %s | FileCheck %s

module attributes { "coir.target" = "cute", "coir.arch" = "sm_86" } {

// CHECK: __global__ void __enabled_test_kernel__
coir.kernel @enabled_test(
    %src: !coir.tensor<128xf32, global>,
    %dst: !coir.tensor<128xf32, shared>) -> !coir.tensor<128xf32, shared> {

  // Enabled USE assertion -> emitted as choreo_assert in device code
  %c128 = arith.constant 128 : index
  %c64 = arith.constant 64 : index
  %use_cmp = arith.cmpi sle, %c64, %c128 : index
  // CHECK: choreo::choreo_assert({{.*}}, "enabled use")
  coir.assert %use_cmp, "enabled use" <use> <element_access>
      {estimated_cost = 32 : i64,
       cost_class = #coir.cost_class<low>,
       enabled = true}

  // Disabled USE assertion -> NOT emitted as choreo_assert
  %c0 = arith.constant 0 : index
  %dis_cmp = arith.cmpi sgt, %c128, %c0 : index
  // CHECK-NOT: choreo::choreo_assert({{.*}}, "disabled use")
  coir.assert %dis_cmp, "disabled use" <use> <element_access>
      {estimated_cost = 2048 : i64,
       cost_class = #coir.cost_class<high>,
       enabled = false}

  coir.return %dst : !coir.tensor<128xf32, shared>
}

// CHECK: __global__ void __entry_filter_test_kernel__
coir.kernel @entry_filter_test(
    %src: !coir.tensor<128xf32, global>,
    %dst: !coir.tensor<128xf32, shared>) -> !coir.tensor<128xf32, shared> {

  // Enabled ENTRY assertion -> emitted as runtime_check in host wrapper
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %entry_cmp = arith.cmpi sgt, %c128, %c0 : index
  coir.assert %entry_cmp, "enabled entry" <entry> <hw_constraint>
      {estimated_cost = 1 : i64,
       cost_class = #coir.cost_class<entry>,
       enabled = true}

  // Disabled ENTRY assertion -> NOT emitted in host wrapper
  %c64 = arith.constant 64 : index
  %dis_entry = arith.cmpi sgt, %c64, %c0 : index
  coir.assert %dis_entry, "disabled entry" <entry> <hw_constraint>
      {estimated_cost = 1 : i64,
       cost_class = #coir.cost_class<entry>,
       enabled = false}

  coir.return %dst : !coir.tensor<128xf32, shared>
}

// Host wrapper: enabled entry emitted, disabled entry NOT emitted
// CHECK: entry_filter_test
// CHECK: runtime_check({{.*}}, "enabled entry")
// CHECK-NOT: runtime_check({{.*}}, "disabled entry")

}
