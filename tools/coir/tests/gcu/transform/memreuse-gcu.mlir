// Static memory reuse — ConvertToGCU path.
// Verifies that coir.tensor.alloc ops with reuse_spm/reuse_offset produce:
//   1. A single memref.alloc (i8 pool) at the top of the gpu.func
//   2. memref.view ops for each reuse alloc, aliasing into the pool at the
//      correct byte offset
//
// RUN: coir-opt --coir-convert-to-gcu %s | FileCheck %s

module attributes {coir.target = "gcu", coir.arch = "gcu300"} {

  // CHECK-LABEL: gpu.func @test_reuse_pool_kernel
  coir.kernel @test_reuse_pool(
      %src: !coir.tensor<1024xbf16, global>,
      %dst: !coir.tensor<1024xbf16, global>) {
    coir.parallel (%tid) in [5] level = #coir.level<thread> {
      %c8 = arith.constant 8 : index
      coir.foreach %k in %c8 {
        // Each alloc reuses "pool0" at a different offset.
        // CHECK: memref.alloc() : memref<256xi8, 2>
        // CHECK: memref.view{{.*}} : memref<256xi8, 2> to memref<128xbf16, 2>
        %buf1 = coir.tensor.alloc {
            reuse_spm = "pool0",
            reuse_offset = 0 : i64,
            spm_size = 256 : i64}
          : !coir.tensor<128xbf16, shared>
        // CHECK: memref.view{{.*}} : memref<256xi8, 2> to memref<64xbf16, 2>
        %buf2 = coir.tensor.alloc {
            reuse_spm = "pool0",
            reuse_offset = 128 : i64,
            spm_size = 256 : i64}
          : !coir.tensor<64xbf16, shared>
        coir.yield
      }
      coir.yield
    }
    coir.return
  }

  // CHECK-LABEL: gpu.func @test_multi_pool_kernel
  // Verifies that two distinct pools produce two independent memref.alloc ops.
  coir.kernel @test_multi_pool(
      %src: !coir.tensor<1024xbf16, global>) {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // Pool A.
      %a1 = coir.tensor.alloc {
          reuse_spm = "poolA",
          reuse_offset = 0 : i64,
          spm_size = 512 : i64}
        : !coir.tensor<256xbf16, shared>
      // Pool B.
      %b1 = coir.tensor.alloc {
          reuse_spm = "poolB",
          reuse_offset = 0 : i64,
          spm_size = 128 : i64}
        : !coir.tensor<64xbf16, shared>
      coir.yield
    }
    coir.return
    // CHECK-DAG: memref.alloc() : memref<512xi8, 2>
    // CHECK-DAG: memref.alloc() : memref<128xi8, 2>
  }

  // CHECK-LABEL: gpu.func @test_no_reuse_kernel
  // Verifies that allocs without reuse_spm still produce standalone allocs.
  coir.kernel @test_no_reuse(
      %src: !coir.tensor<1024xbf16, global>) {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // CHECK: memref.alloc() : memref<128xbf16, 2>
      %buf = coir.tensor.alloc
          : !coir.tensor<128xbf16, shared>
      coir.yield
    }
    coir.return
  }
}
