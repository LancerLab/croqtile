// Static and dynamic memory reuse — TopsccEmitter path.
// Verifies that coir.tensor.alloc ops with reuse_spm/reuse_offset produce:
//   1. A single __shared__ unsigned char __spm_N[bytes] pool
//   2. Typed pointer aliases into that pool at the correct offset
//
// RUN: coir-opt --coir-emit-topscc %s | FileCheck %s

module attributes {"coir.target" = "topscc", "coir.arch" = "gcu300"} {

  // CHECK-LABEL: void test_static_reuse
  coir.kernel @test_static_reuse(
      %src: !coir.tensor<1024xbf16, global>) {
    %c8 = arith.constant 8 : index
    coir.foreach %k in %c8 {
      // First reuse alloc triggers pool declaration.
      // CHECK: __shared__ unsigned char [[POOL:__spm_[0-9]+]][256];
      // CHECK: __bf16* {{.+}} = (__bf16*)((unsigned char*)[[POOL]] + 0);
      %buf1 = coir.tensor.alloc {
          reuse_spm = "pool0",
          reuse_offset = 0 : i64,
          spm_size = 256 : i64}
        : !coir.tensor<128xbf16, shared>

      // Second reuse alloc uses the same pool at offset 128.
      // CHECK: __bf16* {{.+}} = (__bf16*)((unsigned char*)[[POOL]] + 128);
      %buf2 = coir.tensor.alloc {
          reuse_spm = "pool0",
          reuse_offset = 128 : i64,
          spm_size = 256 : i64}
        : !coir.tensor<64xbf16, shared>
      coir.yield
    }
    coir.return
  }

  // CHECK-LABEL: void test_multi_pool
  coir.kernel @test_multi_pool(
      %src: !coir.tensor<1024xbf16, global>) {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // Pool A — float type.
      // CHECK: __shared__ unsigned char [[POOLA:__spm_[0-9]+]][512];
      // CHECK: float* {{.+}} = (float*)((unsigned char*)[[POOLA]] + 0);
      %a1 = coir.tensor.alloc {
          reuse_spm = "poolA",
          reuse_offset = 0 : i64,
          spm_size = 512 : i64}
        : !coir.tensor<128xf32, shared>

      // Pool B — half type.
      // CHECK: __shared__ unsigned char [[POOLB:__spm_[0-9]+]][128];
      // CHECK: __fp16* {{.+}} = (__fp16*)((unsigned char*)[[POOLB]] + 0);
      %b1 = coir.tensor.alloc {
          reuse_spm = "poolB",
          reuse_offset = 0 : i64,
          spm_size = 128 : i64}
        : !coir.tensor<64xf16, shared>
      coir.yield
    }
    coir.return
  }

  // CHECK-LABEL: void test_dynamic_reuse
  // Dynamic memory reuse: offsets come from kernel args.
  coir.kernel @test_dynamic_reuse(
      %src: !coir.tensor<1024xbf16, global>,
      %mr_off_0: index,
      %spm_size_val: index)
      attributes {
        coir.mr_offsets_name = "mr_offsets",
        coir.mr_spm_size_arg = "spm_size_val"
      } {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // Dynamic offset: emit extern __shared__ and pointer alias.
      // CHECK: extern __shared__ unsigned char __dyn_smem[];
      // CHECK: __bf16* {{.+}} = (__bf16*)((unsigned char*)__dyn_smem + mr_off_0);
      %buf = coir.tensor.alloc {
          reuse_offset = -1 : i64,
          dyn_offset_arg = "mr_off_0"}
        : !coir.tensor<128xbf16, shared>
      coir.yield
    }
    coir.return
  }

  // CHECK-LABEL: void test_no_reuse
  // Alloc without reuse attributes: standalone array.
  coir.kernel @test_no_reuse(
      %src: !coir.tensor<1024xbf16, global>) {
    %c4 = arith.constant 4 : index
    coir.foreach %k in %c4 {
      // CHECK: __shared__ __bf16 {{.+}}[128]
      // CHECK-NOT: __spm_
      // CHECK-NOT: __dyn_smem
      %buf = coir.tensor.alloc
          : !coir.tensor<128xbf16, shared>
      coir.yield
    }
    coir.return
  }
}
