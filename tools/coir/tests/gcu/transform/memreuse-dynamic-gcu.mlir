// Dynamic memory reuse — ConvertToGCU path.
// Verifies that coir.tensor.alloc ops with dyn_offset_arg produce:
//   1. A single memref.alloc<?xi8, 2> (dynamic pool) at the gpu.func entry
//   2. memref.view ops with dynamic byte offsets into that pool
//
// RUN: coir-opt --coir-convert-to-gcu %s | FileCheck %s

module attributes {coir.target = "gcu", coir.arch = "gcu300"} {

  // CHECK-LABEL: gpu.func @test_dyn_reuse_kernel
  coir.kernel @test_dyn_reuse(
      %src: !coir.tensor<1024xbf16, global>,
      %mr_off_0: index,
      %mr_off_1: index,
      %spm_size: index)
      attributes {
        coir.mr_offset_args = ["mr_off_0", "mr_off_1"],
        coir.mr_spm_size_arg = "spm_size"
      } {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // Dynamic reuse: offset via kernel arg "mr_off_0".
      // CHECK: %[[POOL:.*]] = memref.alloc(%{{.*}}) : memref<?xi8, 2>
      // CHECK: memref.view %[[POOL]][%{{.*}}] : memref<?xi8, 2> to memref<128xbf16, 2>
      %buf1 = coir.tensor.alloc {
          reuse_offset = -1 : i64,
          dyn_offset_arg = "mr_off_0"}
        : !coir.tensor<128xbf16, shared>

      // Second dynamic reuse: different offset "mr_off_1", same pool.
      // CHECK: memref.view %[[POOL]][%{{.*}}] : memref<?xi8, 2> to memref<64xbf16, 2>
      %buf2 = coir.tensor.alloc {
          reuse_offset = -1 : i64,
          dyn_offset_arg = "mr_off_1"}
        : !coir.tensor<64xbf16, shared>
      coir.yield
    }
    coir.return
    // CHECK-NOT: memref.alloc() : memref<?xi8, 2>
    // CHECK-NOT: memref.alloc() : memref
  }

}
